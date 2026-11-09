import math

import torch
import triton
import triton.language as tl
from transformers.models.llama.modeling_llama import repeat_kv


def _select_attention_torch_aligned(
    q: torch.Tensor,  # [batch_size, q_len, num_q_heads, qk_head_dim]
    k: torch.Tensor,  # [batch_size, kv_len, num_kv_heads, qk_head_dim]
    v: torch.Tensor,  # [batch_size, kv_len, num_kv_heads, v_head_dim]
    select_indices: torch.Tensor,  # [batch_size, q_len, num_group_heads, num_selected_blocks]
    sm_scale: float,
    select_size: int,
    num_slc_score_heads: int,
) -> torch.Tensor:  # [batch_size, q_len, qh, vd]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    select_indices = select_indices.transpose(1, 2)
    b, num_q_heads, q_len, _ = q.shape
    _, num_kv_heads, kv_len, _ = v.shape

    k = repeat_kv(k, num_q_heads // num_kv_heads)
    v = repeat_kv(v, num_q_heads // num_kv_heads)
    select_indices = repeat_kv(select_indices, num_q_heads // num_slc_score_heads)

    s = (q @ k.transpose(-1, -2)) * sm_scale
    mask = None

    if 1 == q_len < kv_len:  # generating:
        causal_mask = torch.ones(q_len, kv_len, dtype=torch.int32, device=q.device).bool()
    else:
        causal_mask = torch.ones(q_len, kv_len, dtype=torch.int32, device=q.device).tril(kv_len - q_len).bool()
    num_select_blocks = math.ceil(kv_len / select_size)
    select_mask = torch.zeros(b, num_q_heads, q_len, num_select_blocks + 1, dtype=torch.int32, device=q.device)
    select_mask.scatter_(-1, select_indices.to(torch.int64), 1)
    select_mask = select_mask.repeat_interleave(select_size, -1)[..., :kv_len].contiguous().bool()
    mask = causal_mask[None, None, :, :] & select_mask
    s.masked_fill_(mask.logical_not(), float("-inf"))
    p = s.softmax(-1, dtype=torch.float32).to(s.dtype)
    o = p @ v
    return o.transpose(1, 2)


def select_attention_torch(
    q: torch.Tensor,  # [total_q_len, num_q_heads, qk_head_dim]
    k_cache: torch.Tensor,  # [kv_cache_size, num_kv_heads, qk_head_dim]
    v_cache: torch.Tensor,  # [kv_cache_size, num_kv_heads, v_head_dim]
    kv_indptr: torch.Tensor,  # [kv_cache_size + 1]
    kv_indices: torch.Tensor,  # [total_kv_len]
    q_indptr: torch.Tensor,  # [batch_size + 1]
    select_indices: torch.Tensor,  # [total_q_len, num_slc_score_heads, top_k]
    sm_scale: float,
    select_size: int,
):
    total_q_len, num_q_heads, qk_head_dim = q.shape
    _, _, v_head_dim = v_cache.shape
    _, num_slc_score_heads, _ = select_indices.shape

    o = q.new_zeros(total_q_len, num_q_heads, v_head_dim)
    for q_start, q_end, kv_start, kv_end in zip(
        q_indptr[:-1],
        q_indptr[1:],
        kv_indptr[:-1],
        kv_indptr[1:],
    ):
        per_seq_kv_indices = kv_indices[kv_start:kv_end]
        per_seq_k = k_cache[per_seq_kv_indices]
        per_seq_v = v_cache[per_seq_kv_indices]
        kv_len = kv_end - kv_start
        q_len = q_end - q_start
        num_select_blocks = math.ceil(kv_len / select_size)
        per_seq_select_indices = select_indices[q_start:q_end]
        per_seq_select_indices = per_seq_select_indices[:, :, :num_select_blocks]
        per_seq_o = _select_attention_torch_aligned(
            q=q[q_start:q_end].unsqueeze(0),
            k=per_seq_k.unsqueeze(0),
            v=per_seq_v.unsqueeze(0),
            select_indices=per_seq_select_indices.unsqueeze(0),
            sm_scale=sm_scale,
            select_size=select_size,
            num_slc_score_heads=num_slc_score_heads,
        )

        o[q_start:q_end] = per_seq_o.squeeze(0)[-q_len:]
    return o


@triton.jit
def _select_attention_decode_kernel(
    q_ptr,  # Q: b x qh x hd
    kcache_ptr,  # [kvcache_len, kh, hd]
    vcache_ptr,  # [kvcache_len, kh, hd]
    t_ptr,  # topk_idx: b x kh x k
    o_ptr,  # O: b x qh x hd
    kv_indptr,
    kv_indices,
    NUM_HEADS_IN_GROUP,
    HEAD_DIM,
    TOPK,
    slc_block_size,
    # sm_scale
    sm_scale,
    # stride
    stride_qb,
    stride_qh,
    stride_qd,
    stride_kl,
    stride_kh,
    stride_kd,
    stride_vl,
    stride_vh,
    stride_vd,
    stride_tb,
    stride_th,
    stride_tk,
    stride_ob,
    stride_oh,
    stride_od,
    # META parameters
    BLOCK_SIZE_K: tl.constexpr,  # for slc block size
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    """
    multi head attn kernel
    """
    # get batch id and head id
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_kh = pid_h // NUM_HEADS_IN_GROUP
    # get kv_len
    kv_start_idx = tl.load(kv_indptr + pid_b)
    kv_end_idx = tl.load(kv_indptr + pid_b + 1)
    kv_len = kv_end_idx - kv_start_idx
    # topk_indices[b, h]
    t_ptr = t_ptr + pid_b * stride_tb + pid_kh * stride_th
    # real block num may be less than topk
    real_slc_block_num = tl.minimum((kv_len + slc_block_size - 1) // slc_block_size, TOPK)
    # real_slc_block_num = tl.sum(
    #     tl.where((topk_idx >= 0) & (topk_idx <= (kv_len - 1) // slc_block_size), 1, 0),
    #     axis=0,
    # )
    off_d = tl.arange(0, BLOCK_SIZE_D)
    mask_d = off_d < HEAD_DIM
    off_q = pid_b * stride_qb + pid_h * stride_qh + off_d
    # [hd]
    q = tl.load(q_ptr + off_q, mask=mask_d, other=0.0)
    # BLOCK_SIZE_K is for select block size
    off_k = tl.arange(0, BLOCK_SIZE_K)
    mask_k = off_k < slc_block_size
    # rowmax and rowsum
    m_i = tl.full((1,), float("-inf"), dtype=tl.float32)
    l_i = tl.full((1,), 0, dtype=tl.float32)
    acc_o = tl.full((BLOCK_SIZE_D,), 0, dtype=tl.float32)
    # sparse attention
    for _i in range(real_slc_block_num):
        # get current block start index
        c = tl.load(t_ptr).to(tl.int32)
        c = c * slc_block_size
        t_ptr = t_ptr + stride_tk
        # load k
        off_n = c + tl.arange(0, BLOCK_SIZE_K)
        kv_loc = tl.load(
            kv_indices + kv_start_idx + off_n,
            mask=(off_n < kv_len) & mask_k,
            other=0,
        )
        # [BLOCK_SIZE_K, hd]
        off_kcache = kv_loc[:, None] * stride_kl + pid_kh * stride_kh + off_d[None, :]
        k = tl.load(
            kcache_ptr + off_kcache,
            mask=(off_n[:, None] < kv_len) & mask_k[:, None] & mask_d[None, :],
            other=0.0,
        )
        # [1, hd] * [BLOCK_SIZE_K, hd] = [BLOCK_SIZE_K, hd] -> [BLOCK_SIZE_K]
        # tl.sum -> float32
        qk = tl.sum(q[None, :] * k, axis=1)
        qk *= sm_scale
        qk = tl.where((off_n < kv_len) & mask_k, qk, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 0))
        p = tl.exp(qk - m_ij)

        off_vcache = kv_loc[:, None] * stride_vl + pid_kh * stride_vh + off_d[None, :]
        # [BLOCK_SIZE_K, hd]
        v = tl.load(
            vcache_ptr + off_vcache,
            mask=(off_n[:, None] < kv_len) & mask_k[:, None] & mask_d[None, :],
            other=0.0,
        )
        re_scale = tl.exp(m_i - m_ij)
        l_i = l_i * re_scale + tl.sum(p, 0)
        acc_o *= re_scale
        acc_o += tl.sum(p[:, None] * v, 0)
        # is_nan = tl.math.isnan(acc_o)
        # if tl.max(is_nan):
        #     tl.device_print("c", c)
        #     tl.device_print("p", p)
        m_i = m_ij

    # final scale
    acc_o = acc_o / l_i
    off_o = pid_b * stride_ob + pid_h * stride_oh + off_d
    tl.store(o_ptr + off_o, acc_o.to(o_ptr.dtype.element_ty), mask=mask_d)


def select_attention_decode_triton(
    q: torch.Tensor,  # [bs, qh, hd]
    k_cache: torch.Tensor,  # [total_len, kh, hd]
    v_cache: torch.Tensor,
    select_indices: torch.Tensor,  # [bs, kh, topk]
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    sm_scale: float,
    select_size: int,
):
    seq_num, qh, hd = q.shape
    _, kh, _ = k_cache.shape
    # MHA kernel, but can compute GQA as substitute
    _, num_slc_score_heads, topk = select_indices.shape
    assert kh == num_slc_score_heads
    assert qh % kh == 0
    num_head_in_group = qh // kh
    topk = select_indices.shape[-1]
    o = torch.zeros_like(q)
    grid = (seq_num, qh)
    BLOCK_SIZE_K = triton.next_power_of_2(select_size)
    BLOCK_SIZE_D = triton.next_power_of_2(hd)
    BLOCK_SIZE_T = triton.next_power_of_2(topk)
    _select_attention_decode_kernel[grid](
        q,
        k_cache,
        v_cache,
        select_indices,
        o,
        kv_indptr,
        kv_indices,
        num_head_in_group,
        hd,
        topk,
        select_size,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        select_indices.stride(0),
        select_indices.stride(1),
        select_indices.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        BLOCK_SIZE_K,
        BLOCK_SIZE_D,
        BLOCK_SIZE_T,
    )
    return o


@triton.jit
def _select_attention_decode_splitk_kernel(
    q_ptr,  # Q: b x qh x hd
    kcache_ptr,  # [kvcache_len, kh, hd]
    vcache_ptr,  # [kvcache_len, kh, hd]
    t_ptr,  # topk_idx: b x kh x k
    o_ptr,  # O: b x qh x hd
    lse_ptr,
    kv_indptr,
    kv_indices,
    NUM_HEADS_IN_GROUP,
    NUM_HEADS_IN_SCORE_GROUP,
    HEAD_DIM,
    V_HEAD_DIM,
    TOPK,
    slc_block_size,
    # sm_scale
    sm_scale,
    # stride
    stride_qb,
    stride_qh,
    stride_qd,
    stride_kl,
    stride_kh,
    stride_kd,
    stride_vl,
    stride_vh,
    stride_vd,
    stride_tb,
    stride_th,
    stride_tk,
    stride_ob,
    stride_oh,
    stride_ok,
    stride_od,
    stride_lseb,
    stride_lseh,
    stride_lsek,
    # META parameters
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # for slc block size
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_VD: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    """
    multi head attn kernel
    """
    # get batch id and head id
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_split_kv = tl.program_id(2)
    pid_kh = pid_h // NUM_HEADS_IN_GROUP
    pid_sh = pid_h // NUM_HEADS_IN_SCORE_GROUP
    # get kv_len
    kv_start_idx = tl.load(kv_indptr + pid_b)
    kv_end_idx = tl.load(kv_indptr + pid_b + 1)
    kv_len = kv_end_idx - kv_start_idx
    # real block num may be less than topk
    real_slc_block_num = tl.minimum((kv_len + slc_block_size - 1) // slc_block_size, TOPK)
    slc_block_num_per_split = tl.cdiv(real_slc_block_num, NUM_KV_SPLITS)
    split_slc_block_start = slc_block_num_per_split * pid_split_kv
    split_slc_block_end = tl.minimum(split_slc_block_start + slc_block_num_per_split, real_slc_block_num)
    if split_slc_block_end <= split_slc_block_start:
        return

    off_d = tl.arange(0, BLOCK_SIZE_D)
    mask_d = off_d < HEAD_DIM
    off_vd = tl.arange(0, BLOCK_SIZE_VD)
    mask_vd = off_vd < V_HEAD_DIM
    off_q = pid_b * stride_qb + pid_h * stride_qh + off_d
    # [hd]
    q = tl.load(q_ptr + off_q, mask=mask_d, other=0.0)
    # BLOCK_SIZE_K is for select block size
    off_k = tl.arange(0, BLOCK_SIZE_K)
    mask_k = off_k < slc_block_size
    # rowmax and rowsum
    m_i = -float("inf")
    l_i = 0.0
    acc_o = tl.zeros((BLOCK_SIZE_VD,), dtype=tl.float32)
    # topk_indices[b, h]
    t_ptr = t_ptr + pid_b * stride_tb + pid_sh * stride_th + split_slc_block_start * stride_tk
    # sparse attention
    for _i in range(split_slc_block_start, split_slc_block_end):
        # get current block start index
        c = tl.load(t_ptr).to(tl.int32)
        c = c * slc_block_size
        t_ptr = t_ptr + stride_tk
        # load k
        off_n = c + tl.arange(0, BLOCK_SIZE_K)
        kv_loc = tl.load(
            kv_indices + kv_start_idx + off_n,
            mask=(off_n < kv_len) & mask_k,
            other=0,
        )
        # [BLOCK_SIZE_K, hd]
        off_kcache = kv_loc[:, None] * stride_kl + pid_kh * stride_kh + off_d[None, :]
        k = tl.load(
            kcache_ptr + off_kcache,
            mask=(off_n[:, None] < kv_len) & mask_k[:, None] & mask_d[None, :],
            other=0.0,
        )
        # [1, hd] * [BLOCK_SIZE_K, hd] = [BLOCK_SIZE_K, hd] -> [BLOCK_SIZE_K]
        # tl.sum -> float32
        qk = tl.sum(q[None, :] * k, axis=1)
        qk *= sm_scale
        qk = tl.where((off_n < kv_len) & mask_k, qk, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 0))
        p = tl.exp(qk - m_ij)

        off_vcache = kv_loc[:, None] * stride_vl + pid_kh * stride_vh + off_vd[None, :]
        # [BLOCK_SIZE_K, hd]
        v = tl.load(
            vcache_ptr + off_vcache,
            mask=(off_n[:, None] < kv_len) & mask_k[:, None] & mask_vd[None, :],
            other=0.0,
        )
        re_scale = tl.exp(m_i - m_ij)
        l_i = l_i * re_scale + tl.sum(p, 0)
        acc_o *= re_scale
        acc_o += tl.sum(p[:, None] * v, 0)
        # is_nan = tl.math.isnan(acc_o)
        # if tl.max(is_nan):
        #     tl.device_print("c", c)
        #     tl.device_print("p", p)
        m_i = m_ij

    # final scale
    acc_o = acc_o / l_i
    off_o = pid_b * stride_ob + pid_h * stride_oh + pid_split_kv * stride_ok + off_vd
    tl.store(o_ptr + off_o, acc_o.to(o_ptr.dtype.element_ty), mask=mask_vd)
    tl.store(
        lse_ptr + pid_b * stride_lseb + pid_h * stride_lseh + pid_split_kv * stride_lsek,
        m_i + tl.log(l_i),
    )


@triton.jit
def _decode_splitk_stage2_kernel(
    a_ptr,
    stride_ab,
    stride_ah,
    stride_ak,
    stride_ad,
    lse_ptr,
    stride_lseb,
    stride_lseh,
    stride_lsek,
    o_ptr,
    stride_ob,
    stride_oh,
    stride_od,
    HEAD_DIM,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    off_d = tl.arange(0, BLOCK_D)
    mask_d = off_d < HEAD_DIM
    e_sum = 1.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    a_ptr = a_ptr + pid_b * stride_ab + pid_h * stride_ah
    for i in range(NUM_KV_SPLITS):
        # load lse
        off_lse = pid_b * stride_lseb + pid_h * stride_lseh + i * stride_lsek
        lse = tl.load(lse_ptr + off_lse)
        new_e_max = tl.maximum(e_max, lse)

        re_scale = tl.exp(e_max - new_e_max)
        exp_lse = tl.exp(lse - new_e_max)
        a = tl.load(a_ptr + i * stride_ak + off_d, mask=mask_d, other=0.0)
        acc = acc * re_scale + a * exp_lse
        e_sum = e_sum * re_scale + exp_lse

        e_max = new_e_max

    off_o = pid_b * stride_ob + pid_h * stride_oh + off_d
    tl.store(o_ptr + off_o, acc / e_sum, mask=mask_d)


def select_attention_decode_splitk_triton(
    q: torch.Tensor,  # [bs, qh, hd]
    k_cache: torch.Tensor,  # [total_len, kh, hd]
    v_cache: torch.Tensor,
    select_indices: torch.Tensor,  # [bs, num_slc_score_heads, topk]
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    sm_scale: float,
    select_size: int,
    num_kv_splits: int,
):
    seq_num, qh, hd = q.shape
    _, _, vhd = v_cache.shape
    _, kh, _ = k_cache.shape
    # MHA kernel, but can compute GQA as substitute
    assert qh % kh == 0
    num_head_in_group = qh // kh
    _, num_slc_score_heads, topk = select_indices.shape
    assert qh % num_slc_score_heads == 0
    num_head_in_score_group = qh // num_slc_score_heads
    attn_logits = torch.zeros(
        (seq_num, qh, num_kv_splits, vhd),
        dtype=q.dtype,
        device=q.device,
    )
    lses = torch.full(
        (seq_num, qh, num_kv_splits),
        -torch.inf,
        dtype=torch.float32,
        device=q.device,
    )
    grid = (seq_num, qh, num_kv_splits)

    BLOCK_SIZE_K = triton.next_power_of_2(select_size)
    BLOCK_SIZE_D = triton.next_power_of_2(hd)
    BLOCK_SIZE_VD = triton.next_power_of_2(vhd)
    BLOCK_SIZE_T = triton.next_power_of_2(topk)
    _select_attention_decode_splitk_kernel[grid](
        q,
        k_cache,
        v_cache,
        select_indices,
        attn_logits,
        lses,
        kv_indptr,
        kv_indices,
        num_head_in_group,
        num_head_in_score_group,
        hd,
        vhd,
        topk,
        select_size,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        select_indices.stride(0),
        select_indices.stride(1),
        select_indices.stride(2),
        *attn_logits.stride(),
        *lses.stride(),
        num_kv_splits,
        BLOCK_SIZE_K,
        BLOCK_SIZE_D,
        BLOCK_SIZE_VD,
        BLOCK_SIZE_T,
    )
    o = torch.empty(
        (seq_num, qh, vhd),
        dtype=q.dtype,
        device=q.device,
    )
    _decode_splitk_stage2_kernel[(seq_num, qh)](
        attn_logits, *attn_logits.stride(), lses, *lses.stride(), o, *o.stride(), vhd, num_kv_splits, BLOCK_SIZE_VD
    )
    # lse = lses.logsumexp(-1)
    # o = torch.einsum("bhkd,bhk->bhd", attn_logits, torch.exp(lses).to(attn_logits.dtype))
    # o = o / torch.exp(lse)[:, :, None]
    return o
