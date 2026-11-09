import torch
import torch.nn.functional as F


def _get_data_from_pa_cache(key, block_table, act_s2):
    _, block_size, n2, d = key.shape
    if n2 != 1:
        raise ValueError("n2 only support 1")
    need_blcok_num = (act_s2 + block_size - 1) // block_size
    act_s2_align = need_blcok_num * block_size
    out = torch.zeros((act_s2_align, d), dtype=key.dtype, device=key.device)
    for i in range(need_blcok_num):
        out[i * block_size : (i + 1) * block_size, :] = key[block_table[i], ...].reshape(block_size, d)

    return out[:act_s2, :]


def get_block_kv(key, start, seq_len, indices, block_size):
    now_key = key[start : start + seq_len].squeeze(1)  # [seq_len, D]
    d = now_key.shape[-1]

    pad_len = (block_size - seq_len % block_size) % block_size
    padded_k = F.pad(now_key, [0, 0, 0, pad_len], value=0)

    padded_k = padded_k.reshape(-1, block_size, d)
    valid = (seq_len + block_size - 1) // block_size

    valid_indices = indices[:valid]
    valid_indices = valid_indices[valid_indices >= 0]

    cur_key = padded_k[valid_indices]  # [k, block_size, D]
    cur_key = cur_key.reshape(-1, d)  # [k*block_size, D]

    if pad_len > 0:
        cur_key = cur_key[:-pad_len]

    return cur_key  # [实际token数, D]


def level2_lighting_indxer(
    query,
    pooled_max,
    pooled_min,
    weights,
    cu_seq_lengths_query,
    cu_seq_lengths_key,
    cu_pooled_seq_lengths_key,
    sparse_count=256,
    init_num=16,
    local_num=1024,
    block_size=128,
    pooled_block_size=16,
):
    total_q, _, _ = query.shape

    batch_size = cu_seq_lengths_query.shape[0] - 1
    sub_blocks_per_block = block_size // pooled_block_size  # 8
    init_block_num = (init_num + block_size - 1) // block_size
    local_block_num = (local_num + block_size - 1) // block_size + 1
    block_indices = torch.full((total_q, sparse_count), -1, dtype=torch.int32, device=query.device)
    for batch_id in range(batch_size):
        q_start = cu_seq_lengths_query[batch_id]
        q_end = cu_seq_lengths_query[batch_id + 1]
        act_s1 = q_end - q_start
        if act_s1 == 0:
            continue
        k_start = cu_seq_lengths_key[batch_id]
        k_end = cu_seq_lengths_key[batch_id + 1]
        act_s2 = k_end - k_start
        if act_s2 == 0:
            continue
        pooled_start = cu_pooled_seq_lengths_key[batch_id]
        pooled_end = cu_pooled_seq_lengths_key[batch_id + 1]

        q_batch = query[q_start:q_end].to(torch.float32)
        k_max_batch = pooled_max[pooled_start:pooled_end].to(torch.float32).squeeze(1)  # T,D
        k_min_batch = pooled_min[pooled_start:pooled_end].to(torch.float32).squeeze(1)  # T,D
        weight_batch = weights[q_start:q_end].to(torch.float32)

        qxk = torch.einsum(
            "tnd,kd->tnk",
            torch.clamp(q_batch, min=0),
            k_max_batch,
        )
        qxk = qxk + torch.einsum(
            "tnd,kd->tnk",
            torch.clamp(q_batch, max=0),
            k_min_batch,
        )
        # qxk_paged = qxk.reshape(
        #     act_s1, q_batch.shape[1], -1, sub_blocks_per_block
        # ).sum(-1)
        # [T, L/128]
        # sub_block_scores = (weight_batch.unsqueeze(-1) * F.relu(qxk_paged)).sum(1)
        sub_block_scores = (weight_batch.unsqueeze(-1) * F.relu(qxk)).sum(1)
        sub_block_scores = sub_block_scores.reshape(act_s1, -1, sub_blocks_per_block).sum(-1)

        # mask&3s
        seqlens_k = torch.arange(act_s2 - act_s1 + 1, act_s2 + 1).to(query.device)
        for i in range(act_s1):
            valid_page = (seqlens_k[i] + block_size - 1) // block_size
            sub_block_scores[i, valid_page:] = float("-inf")
            init_page = min(init_block_num, valid_page)
            sub_block_scores[i, :init_page] = float("inf")
            local_page = min(local_block_num, valid_page - init_page)
            sub_block_scores[i, valid_page - local_page : valid_page] = float("inf")
            use_pages = min(valid_page, sparse_count)
            _, idxs = torch.topk(sub_block_scores[i], k=use_pages, dim=0)
            level1_idxs, _ = idxs.sort(dim=0)
            block_indices[q_start + i, :use_pages] = level1_idxs

    return block_indices


# key tnd
def _lightning_indexer(
    query,
    key,
    weights,
    actual_seq_lengths_query,
    actual_seq_lengths_key,
    layout_query="BSND",
    sparse_count=2048,
    sparse_mode=3,
    init_num=16,
    local_num=1024,
    block_indices=None,
    block_size=128,
    block_table=None,
    block_chunk_size=16,
):
    batch_size = query.shape[0]
    if layout_query == "TND":
        batch_size = actual_seq_lengths_query.shape[0]
    out_shape = list(query.shape)
    n2 = key.shape[1] if block_table is None else key.shape[2]
    d = query.shape[-1]
    n1 = query.shape[-2]
    out_shape[-1] = sparse_count
    out_shape[-2] = n2
    # 初始化为全-1
    out = torch.zeros(out_shape, dtype=torch.int32, device=query.device).reshape(-1, n2, sparse_count) - 1
    act_s1 = 0
    act_s2 = 0
    process_q_len = 0
    process_k_len = 0  # TND key 格式时的偏移量
    for batch_id in range(batch_size):
        if actual_seq_lengths_query is None:
            act_s1 = query.shape[1]
        else:
            if layout_query == "TND":  # TND格式时actual_seq_lengths_query为前缀和
                act_s1 = actual_seq_lengths_query[batch_id] - process_q_len
            else:
                act_s1 = actual_seq_lengths_query[batch_id]
        if act_s1 == 0:
            continue
        act_s2 = actual_seq_lengths_key[batch_id]
        now_q = query.reshape(-1, n1, d)[process_q_len : process_q_len + act_s1, :, :].transpose(0, 1).to(torch.float32)
        now_weights = weights.reshape(-1, n1, 1)[process_q_len : process_q_len + act_s1, :, :].transpose(0, 1).to(torch.float32)

        if block_indices is not None:
            seq_key = key[process_k_len : process_k_len + act_s2, :, :].reshape(-1, d).to(torch.float32)  # [act_s2*n2, d]
            remainder = act_s2 % block_size
            pad_amount = (block_size - remainder) % block_size
            if pad_amount > 0:
                seq_key = F.pad(seq_key, (0, 0, 0, pad_amount), value=0)  # [padded, d]
            num_blocks = seq_key.shape[0] // block_size
            pad_k = seq_key.reshape(num_blocks, block_size, d)  # [num_blocks, block_size, d]

            seq_bi = block_indices[process_q_len : process_q_len + act_s1]  # [act_s1, count]
            q_for_einsum = now_q.transpose(0, 1)  # [act_s1, n1, d]
            weight_scale = now_weights.squeeze(-1).transpose(0, 1)  # [act_s1, n1]
            reduce_out_chunks = []
            chunk_size = max(1, block_chunk_size)

            for chunk_start in range(0, seq_bi.shape[1], chunk_size):
                chunk_end = min(chunk_start + chunk_size, seq_bi.shape[1])
                seq_bi_chunk = seq_bi[:, chunk_start:chunk_end]
                safe_bi_chunk = seq_bi_chunk.clamp(min=0, max=num_blocks - 1)
                level2_k_chunk = pad_k[safe_bi_chunk]  # [act_s1, chunk, block_size, d]
                invalid_mask_chunk = seq_bi_chunk < 0
                if invalid_mask_chunk.any():
                    level2_k_chunk = level2_k_chunk.masked_fill(invalid_mask_chunk.unsqueeze(-1).unsqueeze(-1), 0)
                level2_k_chunk = level2_k_chunk.reshape(act_s1, -1, d)  # [act_s1, chunk*block_size, d]
                level2_qxk_chunk = torch.einsum("tnd,tld->tnl", q_for_einsum, level2_k_chunk)
                reduce_out_chunks.append(torch.einsum("tnl,tn->tl", F.relu(level2_qxk_chunk), weight_scale))

            reduce_out = (
                torch.cat(reduce_out_chunks, dim=1) if reduce_out_chunks else torch.empty((act_s1, 0), dtype=now_q.dtype, device=now_q.device)
            )
        else:
            now_block_table = block_table[batch_id, :]
            now_k = _get_data_from_pa_cache(key, now_block_table, act_s2).transpose(0, 1).to(torch.float32)
            # n1,s1,d @ d,s2 -> n1,s1,s2
            # relu_out = torch.maximum(torch.matmul(now_q, now_k), torch.tensor(0))
            reduce_out = torch.sum(F.relu(torch.matmul(now_q, now_k)) * now_weights, dim=0)
        # weight_out = relu_out * now_weights
        # # n1,s1,s2 -> s1,s2
        # reduce_out = torch.sum(weight_out, dim=0)

        if init_num > 0:
            init_end = min(init_num, reduce_out.shape[1])
            reduce_out[:, :init_end] = float("inf")
        if local_num > 0:
            local_start = max(0, reduce_out.shape[1] - local_num)
            reduce_out[:, local_start:] = float("inf")

        # sparse场景下三角置为-inf
        tmp_s1 = reduce_out.shape[0]
        tmp_s2 = reduce_out.shape[1]
        if sparse_mode == 3:
            for i in range(tmp_s1):
                reduce_out[-1 - i, tmp_s2 - i :] = float("-inf")

        _, sorted_indices = torch.sort(reduce_out, dim=1, descending=True)
        return_s2 = min(sparse_count, tmp_s2)
        out[process_q_len : process_q_len + act_s1, 0, :return_s2] = sorted_indices.to(torch.int32)[:, :return_s2]

        process_k_len += act_s2
        process_q_len += act_s1

    out = out.reshape(out_shape)
    return out


def block_pool_past_key_states(
    past_key_states: torch.Tensor,  # [T, N, D]
    cu_seq_lengths_key: torch.Tensor,  # [B+1], cumsum 格式
    block_size: int = 128,
    page_size: int = 128,
    is_minmax: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert past_key_states.dim() == 3, f"Expected 3D tensor, got {past_key_states.dim()}D"
    _, num_heads, head_dim = past_key_states.shape
    batch_size = cu_seq_lengths_key.shape[0] - 1
    device = past_key_states.device
    pooled_max_list = []
    pooled_min_list = []
    pooled_avg_list = []
    pooled_lens = []
    for i in range(batch_size):
        start = cu_seq_lengths_key[i]
        end = cu_seq_lengths_key[i + 1]
        seq_len = end - start
        if seq_len == 0:
            pooled_lens.append(0)
            continue

        kv = past_key_states[start:end]
        pooled_max_blocks = []
        pooled_min_blocks = []
        pooled_avg_blocks = []

        padded_k = F.pad(
            kv.squeeze(1),
            [0, 0, 0, (page_size - kv.shape[0] % page_size) % page_size],
            value=0,
        )

        full_kv = padded_k.reshape(-1, block_size, num_heads, head_dim)
        if is_minmax:
            pooled_max_blocks.append(full_kv.max(dim=1).values)  # [num_full_blocks, N, D]
            pooled_min_blocks.append(full_kv.min(dim=1).values)  # [num_full_blocks, N, D]

            pooled_max_list.append(torch.cat(pooled_max_blocks, dim=0))  # [num_blocks, N, D]
            pooled_min_list.append(torch.cat(pooled_min_blocks, dim=0))  # [num_blocks, N, D]
            pooled_lens.append(pooled_max_list[-1].shape[0])
        else:
            pooled_avg_blocks.append(full_kv.mean(dim=1))
            pooled_avg_list.append(torch.cat(pooled_avg_blocks, dim=0))
            pooled_lens.append(pooled_avg_list[-1].shape[0])

    if is_minmax:
        pooled_max = torch.cat(pooled_max_list, dim=0).to(device).contiguous()
        pooled_min = torch.cat(pooled_min_list, dim=0).to(device).contiguous()
    else:
        pooled_avg = torch.cat(pooled_avg_list, dim=0).to(device).contiguous()
    cu_pooled_seq_lengths = torch.cat(
        [
            torch.tensor([0], dtype=cu_seq_lengths_key.dtype),
            torch.tensor(pooled_lens, dtype=cu_seq_lengths_key.dtype),
        ]
    ).to(device)
    cu_pooled_seq_lengths = torch.cumsum(cu_pooled_seq_lengths, dim=0).to(device)

    if is_minmax:
        return pooled_max, pooled_min, cu_pooled_seq_lengths
    else:
        return pooled_avg, pooled_avg, cu_pooled_seq_lengths


def convert_block2token(flat_token_indices, selected_block_indices, block_size=128):
    mask = flat_token_indices < 0
    flat_token_indices = flat_token_indices.clamp(min=0)
    block_rank = torch.div(flat_token_indices, block_size, rounding_mode="floor").to(torch.int32)
    token_offset = torch.remainder(flat_token_indices, block_size)

    parent_block_indices = torch.gather(selected_block_indices.to(torch.int64), dim=1, index=block_rank.squeeze(1))
    token_indices = parent_block_indices * block_size + token_offset
    mask = mask | (parent_block_indices < 0)
    token_indices.masked_fill_(mask, -1)
    return token_indices
