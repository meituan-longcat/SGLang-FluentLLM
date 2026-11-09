"""Selected packed-KV transport for native sparse Prefill.

Main-model queries share one deduplicated KV union and one SFAP call.
Every query retains all TopK positions (including duplicates), and current
extend values never pass through INT8 quantization.
All ranks run the same collectives, including ranks with no local queries.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist

PAGE = 128
TOPK = 2048
WIDTH = 576
MAX_SFAP_ROWS = 10000 * PAGE
_native_plan_streams = {}


def check_selected_capacity(count_matrix, max_rows):
    """All ranks check the already exchanged counts before sending KV requests."""
    largest = max((sum(row) for row in count_matrix), default=0)
    if largest > max_rows:
        raise ValueError(
            f"Selected-KV union needs {largest} historical rows; SFAP permits {max_rows} after reserving current tokens (10000 pages total)."
        )


@dataclass
class SelectedKvPrefillState:
    query: torch.Tensor
    packed_pool: torch.Tensor
    extend: torch.Tensor


def prepare_selected_kv(q, q_rope, kv_cache, key, rope):
    packed = kv_cache[0] if isinstance(kv_cache, (list, tuple)) else kv_cache
    if packed.dtype != torch.int8:
        raise ValueError("Selected-KV Prefill requires a packed INT8 cache")
    return SelectedKvPrefillState(
        torch.cat((q, q_rope), -1).contiguous(),
        packed.reshape(-1, packed.shape[-1]),
        torch.cat((key.reshape(-1, 512), rope.reshape(-1, 64)), -1),
    )


@dataclass
class PackedRowRoute:
    incoming: torch.Tensor
    sends: list
    receives: list


def prepare_packed_row_route(slots, group, *, count_matrix, max_rows):
    """Exchange owner-local row addresses once; reuse the route for paired KV."""
    world = group.world_size
    device = slots.device
    # The small matrix is already exchanged while index building finishes.
    matrix = count_matrix.cpu().view(world, world).tolist()
    check_selected_capacity(matrix, max_rows)
    rank = group.rank_in_group
    sends = matrix[rank]
    receives = [matrix[i][rank] for i in range(world)]
    sorted_local = slots[: sum(sends)]
    # Nonempty splits keep HCCL's zero-peer cases on a supported path.
    if all(sends):
        request = sorted_local
    else:
        send_offsets = [sum(sends[:i]) for i in range(world)]
        request = torch.zeros(sum(max(n, 1) for n in sends), dtype=sorted_local.dtype, device=device)
        cursor = 0
        for i, n in enumerate(sends):
            request[cursor : cursor + n].copy_(sorted_local[send_offsets[i] : send_offsets[i] + n])
            cursor += max(n, 1)
    incoming = torch.empty(sum(max(n, 1) for n in receives), dtype=request.dtype, device=device)
    dist.all_to_all_single(
        incoming, request, output_split_sizes=[max(n, 1) for n in receives], input_split_sizes=[max(n, 1) for n in sends], group=group.device_group
    )
    return PackedRowRoute(incoming, sends, receives)


def fetch_packed_rows(packed_pool, group, route):
    """Fetch all selected rows in one AllToAll, retaining owner order."""
    payload = packed_pool.index_select(0, route.incoming)
    output_splits = [max(n, 1) for n in route.sends]
    result = packed_pool.new_empty((sum(output_splits), packed_pool.shape[-1]))
    dist.all_to_all_single(
        result.view(torch.bfloat16),
        payload.view(torch.bfloat16),
        output_split_sizes=output_splits,
        input_split_sizes=[max(n, 1) for n in route.receives],
        group=group.device_group,
    )
    del payload
    if all(route.sends):
        return result
    # Empty peers exchange one dummy row. Only those cases need compaction;
    # otherwise the collective already wrote directly into the final buffer.
    parts = [part[:n] for part, n in zip(result.split(output_splits), route.sends) if n]
    if not parts:
        return result[:0]
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)


def prepare_native_selected_metadata(batch, slice_pos):
    """Batch-only metadata; prepare before Indexer and reuse across its layers."""
    prefix_cpu = tuple(int(x) for x in batch.extend_prefix_lens_cpu)
    extend_cpu = tuple(int(x) for x in batch.extend_seq_lens_cpu)
    lo, hi = slice_pos
    rows = batch.req_pool_indices
    table = batch.req_to_token_pool.req_to_token
    signature = (prefix_cpu, extend_cpu, lo, hi, id(rows), id(table))
    cached = getattr(batch, "_native_selected_metadata", None)
    if cached is not None and cached[0] == signature:
        return cached[1]
    device = rows.device
    seq_cpu = [i for i, count in enumerate(extend_cpu) for _ in range(count)]
    scan_cpu = [min(TOPK, pre + pos + 1) for pre, count in zip(prefix_cpu, extend_cpu) for pos in range(count)]
    seq = torch.tensor(seq_cpu[lo:hi], dtype=torch.int32, device=device)
    scan = torch.tensor(scan_cpu[lo:hi], dtype=torch.int32, device=device)
    prefix = torch.tensor(prefix_cpu, dtype=torch.int32, device=device)
    offsets = torch.tensor([sum(extend_cpu[:i]) for i in range(len(extend_cpu))], dtype=torch.int32, device=device)
    num_pages = max(1, (max(prefix_cpu, default=0) + PAGE - 1) // PAGE)
    columns = torch.arange(num_pages, device=device, dtype=torch.int64) * PAGE
    page_bases = table[rows.long()[:, None], columns[None, :]].int().contiguous()
    meta = (seq, scan, prefix, offsets, page_bases)
    batch._native_selected_metadata = (signature, meta)
    return meta


def begin_native_selected_plan(topk, packed_pool, batch, group, slice_pos, stage_stream, route_cache):
    """Submit planning/counts only: do not wait on the host before Q/K enqueue."""
    if not any(batch.extend_prefix_lens_cpu):
        # Every rank agrees from CPU metadata: there is no historical KV to route.
        return
    if "native_indices" in route_cache or "pending_plan" in route_cache:
        return
    logical = topk.reshape(-1, TOPK).to(torch.int32).contiguous()
    seq, _, prefix, offsets, page_bases = prepare_native_selected_metadata(batch, slice_pos)
    ready = torch.npu.Event()
    ready.record(torch.npu.current_stream())
    # Counts only depend on Lookup/Mark. Finish Project/Remap on the plan
    # stream while HCCL exchanges counts on the existing staging stream.
    plan_stream = _native_plan_streams.get(logical.device)
    if plan_stream is None:
        plan_stream = torch.npu.Stream(device=logical.device)
        _native_plan_streams[logical.device] = plan_stream
    args = (logical, page_bases, seq, prefix, offsets, group.world_size, packed_pool.shape[0], batch.token_to_kv_pool.reserved_page_num)
    with torch.npu.stream(plan_stream):
        plan_stream.wait_event(ready)
        for tensor in args[:5]:
            tensor.record_stream(plan_stream)
        workspace = torch.ops.custom.npu_sparse_kv_build_index_start(*args)
        counts = workspace[3][:, 0].view(group.world_size, 6).sum(1).long()
        counts_ready = torch.npu.Event()
        counts_ready.record(plan_stream)
    with torch.npu.stream(stage_stream):
        stage_stream.wait_event(counts_ready)
        counts.record_stream(stage_stream)
        matrix = torch.empty(group.world_size**2, dtype=torch.int64, device=logical.device)
        dist.all_gather_into_tensor(matrix, counts, group=group.device_group)
    with torch.npu.stream(plan_stream):
        unique, indices = torch.ops.custom.npu_sparse_kv_build_index_finish(*args, workspace)
        planned = torch.npu.Event()
        planned.record(plan_stream)
    unique.record_stream(stage_stream)
    route_cache["pending_plan"] = (unique, indices, matrix, planned)


def native_selected_pages(state, group, route_cache, extend_ready):
    """Plan once; consume each attention's own packed historical KV."""
    if "native_indices" not in route_cache:
        unique, indices, matrix, planned = route_cache.pop("pending_plan")
        torch.npu.current_stream().wait_event(planned)
        route = prepare_packed_row_route(
            unique,
            group,
            count_matrix=matrix,
            max_rows=MAX_SFAP_ROWS - state.extend.shape[0],
        )
        route_cache.update(route=route, native_indices=indices)
    route, indices = route_cache["route"], route_cache["native_indices"]
    prefetched = route_cache.pop("prefetched", None)
    if prefetched is None:
        packed = fetch_packed_rows(state.packed_pool, group, route)
    else:
        pool_pointer, packed, ready = prefetched
        if pool_pointer != state.packed_pool.data_ptr():
            raise RuntimeError("Selected-KV prefetch belongs to a different attention pool")
        torch.npu.current_stream().wait_event(ready)
        packed.record_stream(torch.npu.current_stream())
    torch.npu.current_stream().wait_event(extend_ready)
    state.extend.record_stream(torch.npu.current_stream())
    pages = torch.ops.custom.npu_sparse_kv_materialize(
        packed.view(-1, 1, packed.shape[-1]),
        state.extend.contiguous(),
    )
    return pages, indices


def prefetch_native_selected_history(route_cache, batch, group, stage_stream):
    """One packed lookahead only; BF16 expansion waits for the next attention."""
    next_pool = route_cache.pop("next_pool", None)
    if next_pool is None:
        return
    pool, layer_id = next_pool
    with torch.npu.stream(stage_stream):
        batch.token_to_kv_pool.wait_hicache_load(layer_id)
        packed = fetch_packed_rows(pool, group, route_cache["route"])
        ready = torch.npu.Event()
        ready.record(stage_stream)
    route_cache["prefetched"] = (pool.data_ptr(), packed, ready)


def selected_kv_attention(state, topk, batch, scaling, group, slice_pos, stage_stream, *, route_cache=None):
    """Materialize the selected union and run one shared SFAP per attention."""
    logical = topk.reshape(-1, TOPK).to(torch.int32)
    device = state.query.device
    lo, hi = slice_pos
    if hi - lo != logical.shape[0] or state.query.shape[0] != logical.shape[0]:
        raise ValueError("Selected-KV query/TopK/SP metadata mismatch")
    seq, scan, _, offsets, _ = prepare_native_selected_metadata(batch, slice_pos)
    if route_cache is None:
        route_cache = {}
    if not any(batch.extend_prefix_lens_cpu):
        # Every rank has an empty historical set. Current BF16 rows are already
        # replicated by the SP prolog, so no historical transport is needed.
        if state.extend.shape[0] > MAX_SFAP_ROWS:
            raise ValueError("Current tokens exceed SFAP capacity (10000 pages)")
        if not logical.shape[0]:
            return state.query.new_empty((state.query.shape[1], 0, 512))
        indices = route_cache.get("cold_indices")
        if indices is None:
            if len(batch.extend_seq_lens_cpu) == 1:
                indices = logical.contiguous()
            else:
                row_offsets = offsets[seq.long()]
                indices = (logical + row_offsets[:, None]).masked_fill(logical < 0, -1).contiguous()
            route_cache["cold_indices"] = indices
        if state.extend.shape[0] % PAGE == 0 and state.extend.is_contiguous():
            pages = state.extend.view(-1, PAGE, WIDTH)
        else:
            pages = torch.ops.custom.npu_sparse_kv_materialize(
                state.packed_pool.new_empty((0, 1, state.packed_pool.shape[-1])), state.extend.contiguous()
            )
        route_cache.pop("next_pool", None)
    else:
        compute = torch.npu.current_stream()
        input_ready = torch.npu.Event()
        input_ready.record(compute)
        begin_native_selected_plan(logical, state.packed_pool, batch, group, slice_pos, stage_stream, route_cache)
        with torch.npu.stream(stage_stream):
            pages, indices = native_selected_pages(state, group, route_cache, input_ready)
            transport_ready = torch.npu.Event()
            transport_ready.record(stage_stream)
        # Empty SP ranks must join transport before returning/reusing storage.
        compute.wait_event(transport_ready)
        if not logical.shape[0]:
            prefetch_native_selected_history(route_cache, batch, group, stage_stream)
            return state.query.new_empty((state.query.shape[1], 0, 512))
        pages.record_stream(compute)
        indices.record_stream(compute)
    table = torch.arange(pages.shape[0], dtype=torch.int32, device=device).view(1, -1)
    sequence = torch.zeros(logical.shape[0], dtype=torch.int32, device=device)
    output = torch.ops.custom.npu_sparse_mla_attention_prefill(
        state.query,
        pages,
        indices,
        scan,
        table,
        sequence,
        scaling,
        True,
    )
    # Queue lookahead after first-attention transport, before output collective.
    prefetch_native_selected_history(route_cache, batch, group, stage_stream)
    return output
