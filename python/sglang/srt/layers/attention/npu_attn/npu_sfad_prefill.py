"""Native Prefill SFAD staging and logical-to-physical TopK mapping.

This module is deliberately isolated from Decode SFAD.  The persistent cache
stays sharded Page-SoA; only the pages required by the current Prefill batch
are materialized into a rank-major, replicated BF16 PA workspace.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from sglang.srt.distributed import (
    get_attn_sp_token_slice_pos,
    get_attn_tp_group,
)

PAGE_SIZE = 128
ROW_WIDTH = 576
LATENT_WIDTH = 512
ROPE_WIDTH = 64
TOPK = 2048


@dataclass
class SfadPrefillPagePlan:
    local_pages: torch.Tensor
    rank_counts: list[int]
    prefix_page_inv_perm: torch.Tensor
    prefix_pages_per_seq: list[int]
    extend_pages_per_seq: list[int]
    prefix_lens: list[int]
    extend_lens: list[int]
    page_table: torch.Tensor
    query_seq: torch.Tensor
    scan_count: torch.Tensor
    workspace: Optional[torch.Tensor] = None

    @property
    def prefix_page_count(self) -> int:
        return sum(self.prefix_pages_per_seq)


@dataclass
class SfadPrefillState:
    query: torch.Tensor
    plan: SfadPrefillPagePlan
    local_prefix_pages: torch.Tensor
    extend_pages: torch.Tensor
    gathered_pages: Optional[torch.Tensor] = None


@dataclass
class SfadPrefillLocalPrefixState:
    """This rank's cached-prefix pages, staged as BF16 but not gathered."""

    plan: SfadPrefillPagePlan
    local_prefix_pages: torch.Tensor


@dataclass
class SfadPrefillPrefixState:
    """Final workspace with its rank-major prefix filled by Page AllGather.

    The extend tail is reserved here and filled before Attention consumes it.
    """

    plan: SfadPrefillPagePlan
    gathered_pages: torch.Tensor


_STAGE_STREAM = None


def get_sfad_prefill_stage_stream():
    global _STAGE_STREAM
    if _STAGE_STREAM is None:
        _STAGE_STREAM = torch.npu.Stream()
    return _STAGE_STREAM


def _ceil_pages(tokens: int) -> int:
    return (tokens + PAGE_SIZE - 1) // PAGE_SIZE


def _build_page_plan(forward_batch, *, full_cache=False) -> SfadPrefillPagePlan:
    metadata = forward_batch.attn_metadata
    cache_key = "mtp_sfad_full_cache_plan" if full_cache else "sfad_prefill_page_plan"
    cached = getattr(metadata, cache_key, None)
    if cached is not None:
        return cached

    pool = forward_batch.token_to_kv_pool
    if pool.page_size != PAGE_SIZE:
        raise RuntimeError(f"Native SFAD Prefill requires page_size=128, got {pool.page_size}.")

    prefix_lens = [int(x) for x in forward_batch.extend_prefix_lens_cpu]
    extend_lens = [int(x) for x in forward_batch.extend_seq_lens_cpu]
    if max((p + e for p, e in zip(prefix_lens, extend_lens)), default=0) > 10000 * PAGE_SIZE:
        raise ValueError("Prefill SFAP supports at most 10000 pages (1280000 tokens) per request.")
    query_prefix_lens, query_extend_lens = prefix_lens, extend_lens
    if full_cache:
        # Cached MTP consumes quantized history AND quantized current rows.
        # Preserve that contract while gathering complete pages, not token rows.
        prefix_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
        extend_lens = [0] * len(extend_lens)
    prefix_pages_per_seq = [_ceil_pages(x) for x in prefix_lens]
    # Extend staging follows logical sequence page boundaries.  When prefix
    # ends mid-page, its last gathered page is merged with current BF16 extend
    # rows into one transient boundary page.
    extend_pages_per_seq = [_ceil_pages((prefix % PAGE_SIZE) + extend) if extend else 0 for prefix, extend in zip(prefix_lens, extend_lens)]

    global_page_parts = []
    for seq_id, prefix_len in enumerate(prefix_lens):
        if prefix_len == 0:
            continue
        slots = forward_batch.req_to_token_pool.req_to_token[forward_batch.req_pool_indices[seq_id]][:prefix_len]
        if prefix_len < 0 or slots.numel() != prefix_len:
            raise RuntimeError(
                "Native SFAD Prefill prefix length is outside the request slot row: "
                f"seq={seq_id}, prefix_len={prefix_len}, available_slots={slots.numel()}."
            )
        # Prefix restoration and KV allocation start at a page boundary and
        # keep slots contiguous within each page. Sample each page's first
        # slot, including a partial final page, in logical sequence order.
        # Avoid the NPU RightShift + UniqueConsecutive path, which can fail
        # under concurrent execution with TASK_QUEUE_ENABLE=2.
        pages = torch.div(slots[::PAGE_SIZE], PAGE_SIZE, rounding_mode="floor")
        global_page_parts.append(pages)

    device = forward_batch.req_to_token_pool.req_to_token.device
    if global_page_parts:
        global_pages = torch.cat(global_page_parts).to(torch.int64)
    else:
        global_pages = torch.empty(0, dtype=torch.int64, device=device)

    if global_pages.numel() == 0:
        local_pages = global_pages
        rank_counts = [0] * get_attn_tp_group().world_size
        inv_perm = global_pages
    else:
        if bool((global_pages < pool.reserved_page_num).any().item()):
            raise RuntimeError("Native SFAD Prefill does not accept reserved cache pages.")
        owner = pool.get_rank_from_global_page_id(global_pages).to(torch.int64)
        rank_counts = torch.bincount(owner, minlength=get_attn_tp_group().world_size).to(torch.int32).tolist()
        perm = torch.argsort(owner.to(torch.float32), stable=True)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(global_pages.numel(), device=global_pages.device)
        # Keep the native SFAD staging path off the shared helper's NPU
        # RightShift kernel, which can fail under queued concurrent execution.
        local_pages = (
            torch.div(
                global_pages[owner == pool.kvp_rank] - pool.reserved_page_num,
                pool.attn_tp_group_size,
                rounding_mode="floor",
            )
            + pool.reserved_page_num
        )
        if local_pages.numel() != rank_counts[pool.kvp_rank]:
            raise RuntimeError("Native SFAD Prefill page ownership metadata is inconsistent.")

    # Map each sequence's logical page id directly to the physical page in
    # the rank-major gathered workspace.  This is tiny O(batch * pages)
    # metadata; the 2048-wide token mapping stays fused in the SFAD kernel.
    total_pages_per_seq = [_ceil_pages(prefix + extend) for prefix, extend in zip(prefix_lens, extend_lens)]
    max_pages = max(total_pages_per_seq, default=1)
    page_table = torch.full(
        (len(extend_lens), max(max_pages, 1)),
        -1,
        dtype=torch.int32,
        device=device,
    )
    prefix_cursor = 0
    extend_cursor = sum(prefix_pages_per_seq)
    for seq_id, (prefix_len, prefix_pages, extend_pages, total_pages) in enumerate(
        zip(
            prefix_lens,
            prefix_pages_per_seq,
            extend_pages_per_seq,
            total_pages_per_seq,
        )
    ):
        full_prefix_pages = prefix_len // PAGE_SIZE
        if full_prefix_pages:
            page_table[seq_id, :full_prefix_pages] = inv_perm[prefix_cursor : prefix_cursor + full_prefix_pages].to(torch.int32)
        if extend_pages:
            page_table[seq_id, full_prefix_pages:total_pages] = torch.arange(
                extend_cursor,
                extend_cursor + extend_pages,
                dtype=torch.int32,
                device=device,
            )
        elif prefix_pages > full_prefix_pages:
            page_table[seq_id, full_prefix_pages] = inv_perm[prefix_cursor + full_prefix_pages].to(torch.int32)
        prefix_cursor += prefix_pages
        extend_cursor += extend_pages

    global_query_seq = torch.repeat_interleave(
        torch.arange(len(query_extend_lens), device=device, dtype=torch.int32),
        torch.tensor(query_extend_lens, device=device, dtype=torch.int32),
    )
    global_query_position = (
        torch.cat([torch.arange(1, length + 1, device=device, dtype=torch.int32) for length in query_extend_lens if length])
        if any(query_extend_lens)
        else torch.empty(0, dtype=torch.int32, device=device)
    )
    global_prefix_per_query = torch.repeat_interleave(
        torch.tensor(query_prefix_lens, device=device, dtype=torch.int32),
        torch.tensor(query_extend_lens, device=device, dtype=torch.int32),
    )
    global_scan_count = torch.clamp(global_prefix_per_query + global_query_position, max=TOPK)

    # Query and reverse-redistributed TopK are sharded by Attention SP. Keep
    # per-query metadata in that same local range; page_table remains global
    # because it is indexed by the global sequence id in query_seq.
    slice_start, slice_end = get_attn_sp_token_slice_pos(forward_batch)
    global_query_count = global_query_seq.numel()
    if slice_start < 0 or slice_end < slice_start or slice_end > global_query_count:
        raise RuntimeError(
            f"Native SFAD Prefill received an invalid Attention SP slice: [{slice_start}, {slice_end}) for {global_query_count} queries."
        )
    query_seq = global_query_seq[slice_start:slice_end]
    scan_count = global_scan_count[slice_start:slice_end]

    plan = SfadPrefillPagePlan(
        local_pages=local_pages,
        rank_counts=rank_counts,
        prefix_page_inv_perm=inv_perm,
        prefix_pages_per_seq=prefix_pages_per_seq,
        extend_pages_per_seq=extend_pages_per_seq,
        prefix_lens=prefix_lens,
        extend_lens=extend_lens,
        page_table=page_table.contiguous(),
        query_seq=query_seq.contiguous(),
        scan_count=scan_count.contiguous(),
    )
    setattr(metadata, cache_key, plan)
    return plan


def _allocate_mtp_prefix_scratch(kv_cache, forward_batch, plan):
    """Reuse batch-owned buffers across MTP heads on the same compute stream."""
    packed_pool = kv_cache[0] if isinstance(kv_cache, (list, tuple)) else kv_cache
    if packed_pool.dtype != torch.int8:
        return None
    pages = plan.local_pages.numel()
    # Collective inputs may remain pending-free while backend events await
    # polling. Reuse one input with explicit producer/consumer ordering instead
    # of repeatedly retaining a full local prefix behind these events.
    buffers = getattr(forward_batch, "_sfad_prefill_local_workspaces", None)
    if buffers is None:
        buffers = forward_batch._sfad_prefill_local_workspaces = {}
    key = (packed_pool.device, torch.npu.current_stream().npu_stream)
    fused = buffers.get(key)
    if fused is None or fused.shape[0] < pages:
        buffers.pop(key, None)
        fused = None
        fused = torch.empty((pages, PAGE_SIZE, ROW_WIDTH), dtype=torch.bfloat16, device=packed_pool.device)
        buffers[key] = fused
    # All dequantization outputs are allocated on the owner stream as well.
    # Reuse them on staging rather than accumulating its separate tensor cache.
    chunks = getattr(forward_batch, "_sfad_prefill_chunk_workspaces", None)
    if chunks is None:
        chunks = forward_batch._sfad_prefill_chunk_workspaces = {}
    chunk_key = key + (packed_pool.shape[-1],)
    chunk = chunks.get(chunk_key)
    scratch_pages = min(pages, 2048)
    if chunk is None or chunk[0].shape[0] < scratch_pages:
        chunks.pop(chunk_key, None)
        chunk = None
        chunk = (
            packed_pool.new_empty((scratch_pages, PAGE_SIZE, 1, packed_pool.shape[-1])),
            torch.empty((scratch_pages * PAGE_SIZE, 1, LATENT_WIDTH), dtype=torch.bfloat16, device=packed_pool.device),
            torch.empty((scratch_pages * PAGE_SIZE, 1, ROPE_WIDTH), dtype=torch.bfloat16, device=packed_pool.device),
        )
        chunks[chunk_key] = chunk
    packed, latent, rope = chunk
    return packed[:scratch_pages], fused[:pages], latent[: scratch_pages * PAGE_SIZE], rope[: scratch_pages * PAGE_SIZE]


def _stage_local_prefix_pages(
    kv_cache,
    plan: SfadPrefillPagePlan,
    scratch=None,
) -> torch.Tensor:
    """Return this rank's prefix pages as fused BF16 [P,128,576]."""
    local_pages = plan.local_pages.long()
    packed_pool = kv_cache[0] if isinstance(kv_cache, (list, tuple)) else kv_cache

    if packed_pool.dtype == torch.int8:
        row_bytes = packed_pool.shape[-1]
        packed_pages = packed_pool.reshape(-1, PAGE_SIZE, 1, row_bytes)
        count = local_pages.numel()
        if scratch is None:
            packed_scratch = None
            fused = torch.empty((count, PAGE_SIZE, ROW_WIDTH), dtype=torch.bfloat16, device=packed_pool.device)
        else:
            packed_scratch, fused, latent_scratch, rope_scratch = scratch
        # Bound selection/dequantization temporaries to 262144 rows. The
        # AllGather input remains complete; neither layout nor arithmetic changes.
        for start in range(0, count, 2048):
            end = min(start + 2048, count)
            indices = local_pages[start:end]
            if packed_scratch is None:
                selected = packed_pages.index_select(0, indices)
            else:
                selected = packed_scratch[: end - start]
                torch.index_select(packed_pages, 0, indices, out=selected)
            if scratch is None:
                latent, rope = torch.ops.custom.npu_packed_kv_dequant(selected.view(-1, 1, row_bytes))
            else:
                rows = (end - start) * PAGE_SIZE
                latent, rope = latent_scratch[:rows], rope_scratch[:rows]
                torch.ops.custom.npu_packed_kv_dequant_out(selected.view(-1, 1, row_bytes), latent, rope)
            torch.cat((latent, rope), dim=-1, out=fused[start:end].view(-1, 1, ROW_WIDTH))
            del latent, rope, selected
        return fused

    if not isinstance(kv_cache, (list, tuple)):
        fused = packed_pool.reshape(-1, PAGE_SIZE, 1, ROW_WIDTH)
        return fused.index_select(0, local_pages).squeeze(2).contiguous()

    if len(kv_cache) == 1:
        fused = kv_cache[0].reshape(-1, PAGE_SIZE, 1, ROW_WIDTH)
        return fused.index_select(0, local_pages).squeeze(2).contiguous()

    latent_pool, rope_pool = kv_cache
    latent = latent_pool.reshape(-1, PAGE_SIZE, 1, LATENT_WIDTH).index_select(0, local_pages)
    rope = rope_pool.reshape(-1, PAGE_SIZE, 1, ROPE_WIDTH).index_select(0, local_pages)
    return torch.cat((latent, rope), dim=-1).squeeze(2).contiguous()


def _pack_extend_pages(
    latent: torch.Tensor,
    rope: torch.Tensor,
    prefix_lens: list[int],
    extend_lens: list[int],
) -> torch.Tensor:
    fused = torch.cat(
        (
            latent.reshape(-1, LATENT_WIDTH),
            rope.reshape(-1, ROPE_WIDTH),
        ),
        dim=-1,
    )
    parts = []
    offset = 0
    for prefix_len, length in zip(prefix_lens, extend_lens):
        current = fused[offset : offset + length]
        offset += length
        leading = prefix_len % PAGE_SIZE
        padded = _ceil_pages(leading + length) * PAGE_SIZE
        current = torch.nn.functional.pad(current, (0, 0, leading, padded - leading - length))
        parts.append(current.view(-1, PAGE_SIZE, ROW_WIDTH))
    if offset != fused.shape[0]:
        raise RuntimeError(f"Native SFAD Prefill extend rows mismatch: {offset} vs {fused.shape[0]}.")
    if not parts:
        return fused.new_empty((0, PAGE_SIZE, ROW_WIDTH))
    return torch.cat(parts, dim=0).contiguous()


def prepare_prefill_pages_local(
    query_latent: torch.Tensor,
    query_rope: torch.Tensor,
    kv_cache,
    extend_latent: torch.Tensor,
    extend_rope: torch.Tensor,
    forward_batch,
    prefetched_prefix: Optional[SfadPrefillPrefixState] = None,
) -> SfadPrefillState:
    plan = _build_page_plan(forward_batch)
    if prefetched_prefix is not None:
        if prefetched_prefix.plan is not plan:
            raise RuntimeError("Native SFAD Prefill prefix prefetch metadata does not match the current batch.")
        gathered_pages = prefetched_prefix.gathered_pages
        # The local shard was consumed by the completed prefix AllGather and
        # does not need to stay live through workspace construction.
        local_prefix = gathered_pages.new_empty((0, PAGE_SIZE, ROW_WIDTH))
    else:
        local_prefix = _stage_local_prefix_pages(kv_cache, plan)
        gathered_pages = None
    extend_pages = _pack_extend_pages(extend_latent, extend_rope, plan.prefix_lens, plan.extend_lens)
    query = torch.cat((query_latent, query_rope), dim=-1).contiguous()
    if query.dtype != torch.bfloat16:
        raise RuntimeError(f"Native SFAD Prefill requires BF16 query input, got {query.dtype}.")
    if query.shape[-1] != ROW_WIDTH:
        raise RuntimeError(f"Native SFAD Prefill query width mismatch: expected {ROW_WIDTH}, got {query.shape[-1]}.")
    if query.shape[0] != plan.query_seq.numel():
        raise RuntimeError(f"Native SFAD Prefill local query/SP metadata mismatch: {query.shape[0]} vs {plan.query_seq.numel()}.")
    return SfadPrefillState(
        query=query,
        plan=plan,
        local_prefix_pages=local_prefix,
        extend_pages=extend_pages,
        gathered_pages=gathered_pages,
    )


def prepare_prefill_prefix_pages_local(
    kv_cache,
    forward_batch,
) -> SfadPrefillLocalPrefixState:
    """Prepare the Page-AllGather input without submitting a collective."""
    plan = _build_page_plan(forward_batch)
    return SfadPrefillLocalPrefixState(
        plan=plan,
        local_prefix_pages=_stage_local_prefix_pages(kv_cache, plan),
    )


def prepare_mtp_prefill_pages(query_latent, query_rope, kv_cache, extend_latent, extend_rope, forward_batch):
    """Gather once into final pages; preserve MTP cold/cached KV semantics.

    MTP heads run serially on compute. Share scratch storage, never KV contents,
    across those heads in this batch. No cross-batch/global buffer is retained.
    """
    cached = any(int(n) for n in forward_batch.extend_prefix_lens_cpu)
    plan = _build_page_plan(forward_batch, full_cache=cached)
    page_count = plan.prefix_page_count + sum(plan.extend_pages_per_seq)
    workspace = getattr(forward_batch, "_mtp_sfad_workspace", None)
    if workspace is None or workspace.shape[0] < page_count:
        workspace = torch.empty((page_count, PAGE_SIZE, ROW_WIDTH), dtype=torch.bfloat16, device=query_latent.device)
        forward_batch._mtp_sfad_workspace = workspace
    plan.workspace = workspace[:page_count]
    if cached:
        scratch = _allocate_mtp_prefix_scratch(kv_cache, forward_batch, plan)
        local = _stage_local_prefix_pages(kv_cache, plan, scratch)
        extend = workspace.new_empty((0, PAGE_SIZE, ROW_WIDTH))
    else:
        local = workspace.new_empty((0, PAGE_SIZE, ROW_WIDTH))
        extend = _pack_extend_pages(extend_latent, extend_rope, plan.prefix_lens, plan.extend_lens)
    state = SfadPrefillState(
        query=torch.cat((query_latent, query_rope), dim=-1).contiguous(),
        plan=plan,
        local_prefix_pages=local,
        extend_pages=extend,
    )
    submit_prefill_page_gather(state)
    return state


def reserve_prefill_workspace(forward_batch) -> None:
    """Reserve one batch workspace on compute before publishing input_ready.

    Staging writes reuse it only after the caller's existing compute-ready
    events. Reserve before Indexer is queued: an extra owner-stream event at
    Page AllGather submission would serialize the gather after Indexer.
    """
    plan = _build_page_plan(forward_batch)
    if plan.workspace is None:
        shape = (
            plan.prefix_page_count + sum(plan.extend_pages_per_seq),
            PAGE_SIZE,
            ROW_WIDTH,
        )
        plan.workspace = torch.empty(shape, dtype=torch.bfloat16, device=plan.local_pages.device)


def _gather_prefill_pages(
    local_prefix_pages: torch.Tensor,
    plan: SfadPrefillPagePlan,
) -> torch.Tensor:
    """Gather directly into the final workspace, avoiding a full-prefix copy."""
    prefix_count = plan.prefix_page_count
    # The extend size is known before its values are ready. Reserve the tail
    # now so prefix prefetch can keep its original submission boundary.
    # Production reserves this on compute before its input-ready event. The
    # fallback supports standalone, single-stream callers of this helper.
    if plan.workspace is None:
        plan.workspace = local_prefix_pages.new_empty((prefix_count + sum(plan.extend_pages_per_seq), PAGE_SIZE, ROW_WIDTH))
    workspace = plan.workspace
    if not prefix_count:
        # Cold Prefill intentionally submits no KV collective.
        return workspace

    prefix = workspace[:prefix_count]
    group = get_attn_tp_group()
    if group.world_size == 1:
        prefix.copy_(local_prefix_pages)
    elif max(plan.rank_counts) == min(plan.rank_counts):
        group.all_gather_into_tensor(prefix, local_prefix_pages)
    else:
        import torch_npu  # noqa: F401 - registers device/operator implementations

        # Use the same uneven collective as NpuCommunicator.all_gather,
        # including support for ranks owning no pages.
        torch_npu.distributed.all_gather_into_tensor_uneven(
            prefix,
            local_prefix_pages,
            plan.rank_counts,
            group=group.device_group,
        )
    return workspace


def submit_prefill_prefix_page_gather(
    state: SfadPrefillLocalPrefixState,
    collective_done_event=None,
) -> SfadPrefillPrefixState:
    """Gather a prepared local prefix and publish its collective boundary."""
    gathered_pages = _gather_prefill_pages(state.local_prefix_pages, state.plan)

    if collective_done_event is not None:
        collective_done_event.record(torch.npu.current_stream())

    return SfadPrefillPrefixState(
        plan=state.plan,
        gathered_pages=gathered_pages,
    )


def prefetch_prefill_prefix_pages(
    kv_cache,
    forward_batch,
) -> SfadPrefillPrefixState:
    """Stage and gather cached BF16 prefix before a CLI-reuse Attention."""
    return submit_prefill_prefix_page_gather(prepare_prefill_prefix_pages_local(kv_cache, forward_batch))


def submit_prefill_page_gather(
    state: SfadPrefillState,
    collective_done_event=None,
) -> None:
    if state.gathered_pages is None:
        state.gathered_pages = _gather_prefill_pages(state.local_prefix_pages, state.plan)
    prefix_count = state.plan.prefix_page_count
    gathered_prefix = state.gathered_pages[:prefix_count]

    # The reverse TopK AllToAll shares this communicator. Publish the
    # collective boundary before local boundary-page merge/workspace assembly,
    # allowing those local kernels to overlap the reverse AllToAll.
    if collective_done_event is not None:
        collective_done_event.record(torch.npu.current_stream())

    # A non-aligned cached prefix shares one logical page with the first
    # current extend rows.  Copy only the cached prefix fragment into the
    # transient BF16 page; current rows stay exactly pre-quantization BF16.
    prefix_cursor = 0
    extend_cursor = 0
    for prefix_len, prefix_pages, extend_len, extend_pages in zip(
        state.plan.prefix_lens,
        state.plan.prefix_pages_per_seq,
        state.plan.extend_lens,
        state.plan.extend_pages_per_seq,
    ):
        boundary = prefix_len % PAGE_SIZE
        if boundary and extend_len:
            source_page = state.plan.prefix_page_inv_perm[prefix_cursor + prefix_pages - 1].long()
            state.extend_pages[extend_cursor, :boundary].copy_(gathered_prefix[source_page, :boundary])
        prefix_cursor += prefix_pages
        extend_cursor += extend_pages
    # Copy only the new rows. The caller records page_ready after this write;
    # Attention must keep waiting for it, not just collective_done_event.
    state.gathered_pages[prefix_count:].copy_(state.extend_pages)


def sfad_topk_metadata(
    topk_indices: torch.Tensor,
    plan: SfadPrefillPagePlan,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return compact inputs for in-kernel logical-to-physical mapping."""
    if topk_indices.shape[-1] != TOPK:
        raise RuntimeError(f"Native SFAD Prefill TopK width mismatch: expected {TOPK}, got {topk_indices.shape[-1]}.")
    logical = topk_indices.reshape(-1, TOPK).to(torch.int32).contiguous()
    if plan.query_seq.numel() != logical.shape[0]:
        raise RuntimeError(f"Native SFAD Prefill query/sequence metadata mismatch: {plan.query_seq.numel()} vs {logical.shape[0]}.")
    return logical, plan.scan_count, plan.page_table, plan.query_seq
