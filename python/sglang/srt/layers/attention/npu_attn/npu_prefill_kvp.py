"""Page-granular KVP metadata for contiguous, page-aligned Prefill slot rows."""

import torch


def build_prefill_kvp_metadata(pool, slot_rows, seq_lens):
    """Return local slots, rank counts, inverse permutation and per-request local lengths."""
    page_size = pool.page_size
    reserved = pool.reserved_page_num
    world = pool.attn_tp_group_size
    rank = pool.kvp_rank
    if len(slot_rows) != len(seq_lens) or not slot_rows:
        raise ValueError("slot rows and lengths must describe the same nonempty batch")
    lengths = [int(n) for n in seq_lens]
    if any(n < 0 or row.numel() < n for row, n in zip(slot_rows, lengths)):
        raise ValueError("Prefill length exceeds its slot row")
    if sum(lengths) > torch.iinfo(torch.int32).max:
        raise ValueError("Prefill token count exceeds the int32 cumulative-offset range")
    device = slot_rows[0].device
    page_counts = [(n + page_size - 1) // page_size for n in lengths]
    page_lengths_cpu = [min(page_size, n - j * page_size) for n, pages in zip(lengths, page_counts) for j in range(pages)]
    if not page_lengths_cpu:
        return (
            torch.empty(0, dtype=slot_rows[0].dtype, device=device),
            [0] * world,
            torch.empty(0, dtype=torch.int64, device=device),
            [0] * len(lengths),
        )
    pages = torch.cat([torch.div(row[:n:page_size], page_size, rounding_mode="floor") for row, n in zip(slot_rows, lengths)]).long()
    owners = torch.remainder(pages - reserved, world)
    page_lengths = torch.tensor(page_lengths_cpu, device=device, dtype=torch.int64)
    rank_counts = torch.zeros(world, device=device, dtype=torch.int64)
    rank_counts.scatter_add_(0, owners, page_lengths)
    ends = []
    cursor = 0
    for count in page_counts:
        ends.append(cursor + count - 1 if count else 0)
        cursor += count
    # The caller requires host rank counts. Include partial-page owners and
    # the reserved-page check in this same small synchronization.
    control = torch.cat((rank_counts, owners[torch.tensor(ends, device=device)], (pages < reserved).any().long().view(1))).tolist()
    if control[-1]:
        raise ValueError("Prefill slot rows must contain allocated non-reserved pages")
    counts = [int(x) for x in control[:world]]
    last_owners = control[world:-1]
    perm = torch.argsort(owners.to(torch.float32), stable=True)
    sorted_lengths = page_lengths[perm]
    page_dest = torch.empty_like(perm)
    # Explicit int32 accumulation avoids AICPU Cumsum; keep indices int64.
    page_dest[perm] = sorted_lengths.cumsum(0, dtype=torch.int32).to(torch.int64) - sorted_lengths
    offsets = torch.arange(page_size, device=device, dtype=torch.int64)
    inverse = torch.empty(sum(lengths), device=device, dtype=torch.int64)
    locals_ = []
    local_seq_lens = []
    page_start = token_start = 0
    for n, count, last_owner in zip(lengths, page_counts, last_owners):
        if not count:
            local_seq_lens.append(0)
            continue
        page_end = page_start + count
        # Expand one sequence at a time. No full-batch token mask or token
        # sort is materialized; only the final inverse map is batch-sized.
        for first in range(page_start, page_end, 2048):
            last = min(first + 2048, page_end)
            begin = (first - page_start) * page_size
            size = min((last - first) * page_size, n - begin)
            inverse[token_start + begin : token_start + begin + size].copy_((page_dest[first:last, None] + offsets).flatten()[:size])
        owned_pages = pages[page_start:page_end][owners[page_start:page_end] == rank]
        local_len = owned_pages.numel() * page_size
        if last_owner == rank:
            local_len -= (-n) % page_size
        local_seq_lens.append(local_len)
        local_pages = torch.div(owned_pages - reserved, world, rounding_mode="floor") + reserved
        locals_.append((local_pages[:, None] * page_size + offsets).flatten()[:local_len])
        page_start = page_end
        token_start += n
    local = torch.cat(locals_).to(slot_rows[0].dtype)
    if local.numel() != counts[rank]:
        raise RuntimeError("Page-granular KVP token count disagrees with rank count")
    return local, counts, inverse, local_seq_lens
