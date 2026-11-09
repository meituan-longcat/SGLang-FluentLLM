"""Packed host-KV eligibility and selected-cache operands for native SFAD."""
import torch


def packed_host_enabled(args):
    def value(name, default=None):
        return args.get(name, default) if isinstance(args, dict) else getattr(args, name, default)
    return (
        value("device") == "npu"
        and value("disaggregation_mode") == "decode"
        and bool(value("offload_cache", False))
        and value("offload_cache_storage", "host") == "host"
        and bool(value("npu_dsa_kv_int8", False))
    )


def consume_packed_selection(coordinator, attention_id, topk_indices,
                             update_timing="before_attention", *, original_valid_chunks=None):
    """Bind the CLI pair's gathered rows without changing KVP query order.

    Gen supplies positional selectors, including holes; Gather and the current
    row patch write to those positions. Keep the full 2048-row span visible to
    SFAD. The authoritative packed pool is not an operand returned here.
    """
    if (coordinator is None or not getattr(coordinator, "packed_kv", False)
            or not coordinator.has_selection_context(attention_id)):
        raise RuntimeError("Packed offload requires a live packed selection context")
    context = coordinator.get_selection_context(attention_id)
    if context.source_topk is not topk_indices:
        raise RuntimeError("Packed selection context must share the CLI2 TopK tensor")
    rows = topk_indices.shape[0]
    selected = context.k_nope
    if (getattr(context, "packed_row_bytes", None) != 672
            or selected.dtype != torch.int8 or selected.dim() != 4
            or selected.shape[1:] != (128, 1, 672)
            or selected.shape[0] < rows * 16 or context.k_rope is not None):
        raise RuntimeError("Packed selection requires INT8 page128/row672 KV")
    if (rows < 1 or rows > 64 or topk_indices.shape != (rows, 1, 2048)
            or context.topk_indices.shape != topk_indices.shape
            or context.topk_indices.dtype != torch.int32
            or context.actual_seq_qlen.shape != (rows,)
            or context.actual_kv_seq.shape != (rows,)
            or context.actual_seq_qlen.dtype != torch.int32
            or context.actual_kv_seq.dtype != torch.int32):
        raise RuntimeError("Packed SFAD selection metadata must have one request per query")
    # The coordinator publishes this operand with the current-row patch's GE
    # completion dependency, shared by both SFAD and hot-cache update.
    blocks = coordinator.selection_block_table(rows)
    # A late valid position after holes must still be scanned. Gen/Gather have
    # already applied causal/current-window masking before this remapping.
    if original_valid_chunks is None:
        valid_chunks = torch.full((rows,), 16, dtype=torch.int32, device=topk_indices.device)
    else:
        if original_valid_chunks.shape != (rows,) or original_valid_chunks.dtype != torch.int32:
            raise RuntimeError("Packed SFAD requires the original per-query INT32 chunk plan")
        # Preserve the OFF path's reduction groups after Gen masks future
        # current slots. An extra empty group can change floating-point rounding.
        valid_chunks = original_valid_chunks
    bound = (selected, context.topk_indices, blocks, context.actual_seq_qlen,
             context.actual_kv_seq, valid_chunks)
    if update_timing == "before_attention":
        coordinator.clear_temp_tensors(attention_id)
    return bound


# Compatibility helper from upstream; hot-cache decode consumes its paired context.
def materialize_packed_selection(packed, topk, block_table, kv_lengths,
                                 query_lengths, *, causal):
    """Gather whole packed rows and convert metadata to one request/query.

    ``topk`` retains its original query order, including KVP communication
    order. Rank 0's current-token causal mask is applied before remapping;
    other ranks contain history only. Invalid/padded indices read safe row 0
    but remain -1 in the attention selector and cannot affect the result.
    """
    if packed.dtype != torch.int8 or packed.dim() != 4 or packed.shape[1:3] != (128, 1):
        raise ValueError("Packed host selection requires INT8 PA_BSND page128")
    if packed.shape[-1] != 672:
        raise ValueError("Packed host offload requires 672-byte rows")
    total, heads, count = topk.shape
    batch = kv_lengths.numel()
    if heads != 1 or count != 2048 or batch == 0 or total > 64:
        raise ValueError("Unsupported packed host query/selector geometry")
    if query_lengths.numel() != batch or block_table.shape[0] < batch:
        raise ValueError("Packed host sequence metadata does not match")
    device = topk.device
    query_id = torch.arange(total, device=device, dtype=torch.int32)
    # SFAD takes cumulative query ends, including unequal/padded requests.
    request = torch.sum(query_id[:, None] >= query_lengths[None, :], dim=1, dtype=torch.int32)
    request = torch.minimum(request, torch.full_like(request, batch - 1))
    starts = torch.cat((torch.zeros_like(query_lengths[:1]), query_lengths[:-1]))
    query_start = torch.index_select(starts, 0, request)
    query_index = query_id - query_start
    q_len = torch.index_select(query_lengths, 0, request) - query_start
    kv_len = torch.index_select(kv_lengths, 0, request)
    if causal:
        kv_len = kv_len - (q_len - 1 - query_index).clamp_min(0)
    logical = topk.reshape(total, count)
    valid = (logical >= 0) & (logical < kv_len[:, None]) & (query_index < q_len)[:, None]
    max_blocks = block_table.shape[1]
    valid = valid & (logical // 128 < max_blocks)
    safe_block = logical.clamp_min(0) // 128
    safe_block = torch.minimum(safe_block, torch.full_like(safe_block, max_blocks - 1))
    request_blocks = torch.index_select(block_table, 0, request)
    physical_page = torch.gather(request_blocks, 1, safe_block.to(torch.int64))
    valid = valid & (physical_page >= 0) & (physical_page < packed.shape[0])
    physical = physical_page * 128 + logical.clamp_min(0) % 128
    physical = torch.where(valid, physical, torch.zeros_like(physical))
    selected = torch.index_select(packed.view(-1, 672), 0, physical.reshape(-1))
    selected = selected.view(total * (count // 128), 128, 1, 672)
    positions = torch.arange(count, device=device, dtype=torch.int32)[None, :].expand(total, count)
    selected_indices = torch.where(valid, positions, torch.full_like(positions, -1)).unsqueeze(1)
    selected_blocks = torch.arange(total * (count // 128), device=device, dtype=torch.int32).view(total, count // 128)
    selected_query_lengths = torch.arange(1, total + 1, device=device, dtype=torch.int32)
    selected_kv_lengths = torch.full((total,), count, device=device, dtype=torch.int32)
    # Scan all selector chunks. Holes introduced by the causal mask are kept;
    # counting valid rows alone would not bound the last non-empty chunk.
    valid_chunks = torch.full((total,), count // 128, device=device, dtype=torch.int32)
    return (selected, selected_indices, selected_blocks, selected_query_lengths,
            selected_kv_lengths, valid_chunks)
