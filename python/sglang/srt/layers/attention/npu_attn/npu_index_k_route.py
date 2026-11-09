"""Batch-scoped routing of paged Index-K to its Query consumers.

The plan preserves the TND input of get_non_zero_q_meta. Only page metadata
is sorted; no Indexer scores or TopK results are involved. Physical pages
may be arbitrarily ordered. One small host transfer supplies HCCL splits;
all token address expansion and per-layer data movement stay on device.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist


def required_prefixes(seq_lens, extend_lens, query_splits):
    """Query partitions are contiguous in concatenated request order."""
    if len(seq_lens) != len(extend_lens) or sum(query_splits) != sum(extend_lens):
        raise ValueError("Index-K query partition does not cover the batch")
    if any(q < 0 or n < q for n, q in zip(seq_lens, extend_lens)) or any(
        s < 0 for s in query_splits
    ):
        raise ValueError("Invalid Index-K sequence lengths")
    result = []
    start = 0
    for size in query_splits:
        end = start + size
        req_start = 0
        needed = []
        for n, q in zip(seq_lens, extend_lens):
            req_end = req_start + q
            overlap = min(end, req_end) - max(start, req_start)
            needed.append(n - q + min(end, req_end) - req_start if overlap > 0 else 0)
            req_start = req_end
        result.append(needed)
        start = end
    return result


def _expand_pages(bases, lengths, total):
    """Expand variable final-page lengths without a device scalar read."""
    if total == 0:
        return bases.new_empty(0)
    # Explicit int32 accumulation avoids AICPU Cumsum; keep indices int64.
    starts = lengths.cumsum(0, dtype=torch.int32).to(torch.int64) - lengths
    offsets = bases - starts
    jumps = torch.cat((offsets[:1], offsets[1:] - offsets[:-1]))
    # Segment starts are unique. A prefix sum expands these boundary deltas
    # in linear device work; NPU repeat_interleave is costly for many pages.
    expanded = bases.new_zeros(total)
    expanded.index_copy_(0, starts, jumps)
    return expanded.cumsum(0, dtype=torch.int32).to(torch.int64) + torch.arange(
        total, dtype=torch.int64, device=bases.device
    )


@dataclass
class IndexKRoute:
    send_indices: torch.Tensor
    recv_inverse: torch.Tensor
    send_counts: list[int]
    recv_counts: list[int]

    def exchange(self, cache, group):
        width = cache.shape[-1]
        send = cache.reshape(-1, width).index_select(0, self.send_indices)
        recv = cache.new_empty((sum(self.recv_counts), width))
        dist.all_to_all_single(
            recv,
            send,
            output_split_sizes=self.recv_counts,
            input_split_sizes=self.send_counts,
            group=group,
        )
        return recv.index_select(0, self.recv_inverse).view(-1, 1, width)


def build_index_k_route(
    slot_rows, seq_lens, extend_lens, query_splits, page_size, reserved, rank
):
    world = len(query_splits)
    needed = required_prefixes(seq_lens, extend_lens, query_splits)
    total_needed = sum(map(sum, needed))
    # Shared-request batches benefit from broadcast. Avoid duplicating a
    # full prefix in the sender buffer merely to emulate AllGather.
    if world == 1 or total_needed * 4 >= world * sum(seq_lens) * 3:
        return None
    if total_needed > torch.iinfo(torch.int32).max:
        raise ValueError(
            "Index-K routed token count exceeds the int32 cumulative-offset range"
        )
    if len(slot_rows) != len(seq_lens) or not slot_rows:
        raise ValueError("Index-K slot rows do not cover the batch")
    device = slot_rows[0].device
    page_parts, lengths_cpu, destinations = [], [], []
    dst_page_counts = []
    for dst, prefixes in enumerate(needed):
        count = 0
        for row, n in zip(slot_rows, prefixes):
            if not n:
                continue
            if row.numel() < n:
                raise ValueError("Index-K prefix exceeds its allocated slot row")
            pages = (n + page_size - 1) // page_size
            page_parts.append(
                torch.div(row[:n:page_size], page_size, rounding_mode="floor").long()
            )
            lengths_cpu.extend([page_size] * (pages - 1) + [(n - 1) % page_size + 1])
            destinations.extend([dst] * pages)
            count += pages
        dst_page_counts.append(count)
    if not page_parts:
        return None
    pages = torch.cat(page_parts)
    lengths = torch.tensor(lengths_cpu, device=device, dtype=torch.int64)
    dsts = torch.tensor(destinations, device=device, dtype=torch.int64)
    owners = (pages - reserved) % world
    groups = dsts * world + owners
    # Keep stable grouping for the page permutations. The NPU statistics
    # kernel accumulates private histograms without SearchSorted or atomics.
    sorted_groups, order = torch.sort(groups.to(torch.float32), stable=True)
    if device.type == "npu":
        # The flash package loads its per-operator libraries. Keep stable sort
        # for the send/receive permutations; statistics need no permutation.
        import flash_npu_kernel  # noqa: F401

        control_tensor = torch.ops.flash.npu_index_k_route_stats(
            groups, lengths, pages, world * world, reserved
        )
    else:
        # CPU reference path for route validation and non-NPU callers.
        boundaries = torch.searchsorted(
            sorted_groups,
            torch.arange(world * world + 1, device=device, dtype=torch.float32),
        )
        cumulative = torch.cat(
            (
                lengths.new_zeros(1),
                lengths[order].cumsum(0, dtype=torch.int32).to(torch.int64),
            )
        )
        ends = cumulative[boundaries]
        counts = ends[1:] - ends[:-1]
        page_counts = boundaries[1:] - boundaries[:-1]
        control_tensor = torch.cat(
            (counts, page_counts, (pages < reserved).any().long().view(1))
        )
    control = control_tensor.cpu().tolist()
    if control[-1]:
        raise ValueError("Index-K routing encountered an unallocated page")
    counts_cpu = control[: world * world]
    page_counts_cpu = control[world * world : -1]
    # Stable grouping retains request/page order within each sender.
    selected = []
    cursor = 0
    for group_id, count in enumerate(page_counts_cpu):
        if group_id % world == rank:
            selected.append(order[cursor : cursor + count])
        cursor += count
    send_pages = torch.cat(selected)
    send_counts = [counts_cpu[dst * world + rank] for dst in range(world)]
    local_bases = ((pages[send_pages] - reserved) // world + reserved) * page_size
    send_indices = _expand_pages(local_bases, lengths[send_pages], sum(send_counts))

    first = sum(dst_page_counts[:rank])
    last = first + dst_page_counts[rank]
    recv_counts = counts_cpu[rank * world : (rank + 1) * world]
    recv_order = order[first:last] - first
    recv_lengths = lengths[first:last]
    sorted_lengths = recv_lengths[recv_order]
    recv_bases = torch.empty_like(recv_order)
    recv_bases[recv_order] = (
        sorted_lengths.cumsum(0, dtype=torch.int32).to(torch.int64) - sorted_lengths
    )
    recv_inverse = _expand_pages(recv_bases, recv_lengths, sum(recv_counts))
    return IndexKRoute(send_indices, recv_inverse, send_counts, recv_counts)
