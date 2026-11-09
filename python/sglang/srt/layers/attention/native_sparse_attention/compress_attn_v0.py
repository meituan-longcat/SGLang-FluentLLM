import math

import torch
import triton
import triton.language as tl


@triton.jit
def _compress_attn_decode(
    q_ptr,  # Q: b x qh x hd
    kcache_ptr,  # [kvcache_len, kh, hd]
    vcache_ptr,  # [kvcache_len, kh, hd]
    kv_loc_ptr,
    cu_com_blocks,
    score_ptr,
    o_ptr,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_kl,
    stride_kh,
    stride_kd,
    stride_vl,
    stride_vh,
    stride_vd,
    stride_sco_l,
    stride_sco_h,
    stride_ob,
    stride_oh,
    stride_od,
    sm_scale,
    BLOCK_D: tl.constexpr = 128,
    BLOCK_L: tl.constexpr = 64,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    com_start_idx = tl.load(cu_com_blocks + pid_b)
    com_end_idx = tl.load(cu_com_blocks + pid_b + 1)
    com_blocks = com_end_idx - com_start_idx
    if com_blocks == 0:
        return
    off_d = tl.arange(0, BLOCK_D)
    off_q = pid_b * stride_qb + pid_h * stride_qh + off_d
    # [hd]
    q = tl.load(q_ptr + off_q)
    loops = (com_blocks + BLOCK_L - 1) // BLOCK_L
    m_i = tl.full((1,), float("-inf"), dtype=tl.float32)
    l_i = tl.full((1,), 0, dtype=tl.float32)
    acc_o = tl.full((BLOCK_D,), 0, dtype=tl.float32)
    for i in range(loops):
        off_n = i * BLOCK_L + tl.arange(0, BLOCK_L)
        # [BLOCK_L]
        kv_loc = tl.load(
            kv_loc_ptr + com_start_idx + off_n,
            mask=(off_n < com_blocks),
            other=0,
        )
        off_kcache = kv_loc[:, None] * stride_kl + pid_h * stride_kh + off_d[None, :]
        k = tl.load(
            kcache_ptr + off_kcache,
            mask=off_n[:, None] < com_blocks,
            other=0.0,
        )
        # [BLOCK_L]
        qk = tl.sum(q[None, :] * k, axis=1)
        qk *= sm_scale
        # store qk
        score_off = (com_start_idx + off_n) * stride_sco_l + pid_h * stride_sco_h
        tl.store(score_ptr + score_off, qk, mask=off_n < com_blocks)
        qk = tl.where(off_n < com_blocks, qk, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 0))
        p = tl.exp(qk - m_ij)

        off_vcache = kv_loc[:, None] * stride_vl + pid_h * stride_vh + off_d[None, :]
        v = tl.load(
            vcache_ptr + off_vcache,
            mask=(off_n[:, None] < com_blocks),
            other=0.0,
        )
        re_scale = tl.exp(m_i - m_ij)
        l_i = l_i * re_scale + tl.sum(p, 0)
        acc_o *= re_scale
        acc_o += tl.sum(p[:, None] * v, 0)
        m_i = m_ij

    acc_o = acc_o / l_i
    off_o = pid_b * stride_ob + pid_h * stride_oh + off_d
    tl.store(o_ptr + off_o, acc_o)
    for i in range(loops):
        off_n = i * BLOCK_L + tl.arange(0, BLOCK_L)
        score_off = (com_start_idx + off_n) * stride_sco_l + pid_h * stride_sco_h
        qk = tl.load(score_ptr + score_off, mask=off_n < com_blocks, other=0)
        qk = tl.where(off_n < com_blocks, qk, float("-inf"))
        scores = tl.exp(qk - m_i) / l_i
        tl.store(score_ptr + score_off, scores, mask=off_n < com_blocks)


def compress_attn_decode_triton(
    q: torch.Tensor,
    kcache: torch.Tensor,  # [total_len, kh, hd]
    vcache: torch.Tensor,
    cu_com_blocks,
    compressed_kv_buffer_loc,  # []
):
    bs, qh, hd = q.shape
    flat_com_score = torch.empty(cu_com_blocks[-1].item(), qh, dtype=torch.float32, device=q.device)
    o = torch.zeros_like(q)
    sm_scale = 1 / math.sqrt(hd)
    _compress_attn_decode[(bs, qh)](
        q,
        kcache,
        vcache,
        compressed_kv_buffer_loc,
        cu_com_blocks,
        flat_com_score,
        o,
        *q.stride(),
        *kcache.stride(),
        *vcache.stride(),
        *flat_com_score.stride(),
        *o.stride(),
        sm_scale,
        BLOCK_D=hd,
    )
    return o, flat_com_score
