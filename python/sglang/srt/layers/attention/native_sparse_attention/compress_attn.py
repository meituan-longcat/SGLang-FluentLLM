import math
import torch
import triton
import triton.language as tl
from typing import Tuple
from transformers.models.llama.modeling_llama import repeat_kv
from sglang.srt.layers.attention.triton_ops.decode_attention import decode_attention_fwd
from flash_attn_3.flash_attn_interface import flash_attn_with_kvcache

from einops import rearrange

def _compress_attention_with_score_torch_aligned(
    q: torch.Tensor, # [batch_size, q_len, qh, qkd]
    compressed_k: torch.Tensor, # [batch_size, kv_len, kvh, qkd]
    compressed_v: torch.Tensor, # [batch_size, kv_len, kvh, vd]
    real_seq_len: int,
    kernel_size: int,
    stride: int,
    sm_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]: # [batch_size, q_len, qh, vd], [batch_size, q_len, qh, kv_len]
    q = q.transpose(1, 2)
    compressed_k = compressed_k.transpose(1, 2)
    compressed_v = compressed_v.transpose(1, 2)
    _, qh, m, _ = q.shape
    _, kh, n, _ = compressed_v.shape

    compressed_k = repeat_kv(compressed_k, qh // kh)
    compressed_v = repeat_kv(compressed_v, qh // kh)

    s = (q @ compressed_k.transpose(-1, -2)) * sm_scale
    # Find the last value of real k_idx contained in block_k_idx
    if m == 1 and m < real_seq_len:  # using kv cache
        q_idx = torch.tensor([real_seq_len - 1], device=q.device, dtype=torch.int32)
    else:
        q_idx = torch.arange(m, device=q.device, dtype=torch.int32)
    block_idx = torch.arange(n, device=q.device, dtype=torch.int32)
    k_idx = block_idx * stride + kernel_size - 1
    mask = q_idx[:, None] < k_idx[None, :]
    s.masked_fill_(mask[None, None, :, :], torch.finfo(q.dtype).min)
    p = s.softmax(-1, dtype=torch.float32).to(s.dtype)
    o = p @ compressed_v
    # When q_idx is in the first block and not at the block end, the number of blocks visible to q is 0, need to mask the output
    o.masked_fill_((q_idx < (kernel_size - 1))[None, None, :, None], 0)
    return o.transpose(1, 2), p


def _compress_attention_with_score_torch(
        q: torch.Tensor, # [total_q_len, num_q_heads, qk_head_dim]
        compressed_k_cache: torch.Tensor, # [kv_size, num_kv_heads, qk_head_dim]
        compressed_v_cache: torch.Tensor, # [kv_size, num_kv_heads, v_head_dim]
        compressed_kv_indptr: torch.Tensor, # [batch_size + 1]
        compressed_kv_indices: torch.Tensor, 
        kv_lens: torch.Tensor, # [batch_size]
        q_indptr: torch.Tensor, # [batch_size + 1]
        score_indptr: torch.Tensor, # [batch_size + 1]
        kernel_size: int,
        stride: int,
        sm_scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

    total_q_len, num_q_heads, qk_head_dim = q.shape
    _, _, v_head_dim = compressed_v_cache.shape
    total_scores_len = score_indptr[-1]

    o = q.new_empty(total_q_len, num_q_heads, v_head_dim)
    p = q.new_empty(num_q_heads, total_scores_len)

    for q_start, q_end, compressed_kv_start, compressed_kv_end, score_start, score_end, kv_len in zip(
        q_indptr[:-1], q_indptr[1:],
        compressed_kv_indptr[:-1], compressed_kv_indptr[1:],
        score_indptr[:-1], score_indptr[1:],
        kv_lens,
    ):
        q_len = q_end - q_start
        p_len = score_end - score_start
        if q_len == 1:
            per_seq_q = q[q_start:q_end]
        else:
            per_seq_q = q.new_zeros((kv_len, num_q_heads, qk_head_dim))
            per_seq_q[-q_len:] = q[q_start:q_end]
        per_seq_compressed_kv_indices = compressed_kv_indices[compressed_kv_start:compressed_kv_end]

        cached_compressed_k = compressed_k_cache[per_seq_compressed_kv_indices] # [compressed_kv_len, num_kv_heads, qk_head_dim]
        cached_compressed_v = compressed_v_cache[per_seq_compressed_kv_indices] # [compressed_kv_len, num_kv_heads, v_head_dim]

        per_seq_o, per_seq_p = _compress_attention_with_score_torch_aligned(
            q=per_seq_q.unsqueeze(0),
            compressed_k=cached_compressed_k.unsqueeze(0),
            compressed_v=cached_compressed_v.unsqueeze(0),
            real_seq_len=kv_len.item(),
            kernel_size=kernel_size,
            stride=stride,
            sm_scale=sm_scale,
        )

        o[q_start:q_end] = per_seq_o.squeeze(0)[-q_len:]
        p[:,score_start:score_end]= per_seq_p.squeeze(0)[:,-q_len:,:].view(num_q_heads, p_len)

    o = o.view(-1, num_q_heads, v_head_dim)
    return o, p.transpose(0, 1)


@triton.jit
def _compute_select_score_kernel(AP, SP,
                          ap_stride_b, ap_stride_h, ap_stride_n, ap_stride_m,
                          sp_stride_b, sp_stride_h, sp_stride_n, sp_stride_k,
                          kernel_size, stride, 
                          select_size, num_selcct_blocks, top_n, num_inital, num_local,
                          return_p: tl.constexpr,
                          B, N, M, KH, num_com_block,
                          BLOCK_SIZE_K: tl.constexpr,
                          BLOCK_SIZE_N: tl.constexpr=16,
                          CHUNK_N: tl.constexpr=128
                            ):
    """
    AP: probs [B, KH, N, num_com_block]
    SP: select_probs [B, KH, N, num_selcct_blocks]
    FInd: forward index [B, KH, N, top_n]
    """
    off_bh = tl.cast(tl.program_id(0), tl.int64)
    off_h = off_bh % KH
    off_b = off_bh // KH
    start_n = tl.cast(tl.program_id(1), tl.int64) * CHUNK_N \
            + tl.program_id(2) * BLOCK_SIZE_N
    if start_n >= N:
        return
    off_n = start_n + tl.arange(0, BLOCK_SIZE_N)

    # AP[b, h]
    AP += off_b * ap_stride_b + off_h * ap_stride_h
    SP += off_b * sp_stride_b + off_h * sp_stride_h

    acc_p = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)
    select_idx = tl.arange(0, BLOCK_SIZE_K)

    select_start = select_idx * select_size
    select_end = select_start + select_size
    # compress_start = stride - kernel_size 
    # num_loops = (select_size + 2 * (kernel_size - stride) - kernel_size) // stride + 1
    num_loops = (select_size + kernel_size - stride) // stride
    # compress_idx: [-1 3 7 11 ...]
    compress_idx = (select_idx * select_size - kernel_size) // stride + 1
    compress_start = compress_idx * stride
    for i in range(num_loops):
        compress_end = compress_start + kernel_size
        area = tl.minimum(compress_end, select_end) - tl.maximum(compress_start, select_start)
        w = area / stride
        mask = (compress_idx >= 0) & (compress_idx < num_com_block)
        # If compress_idx == -1, this will load 0
        p = tl.load(AP + off_n[:, None] * ap_stride_n + compress_idx[None, :] * ap_stride_m, 
                    mask=(off_n[:, None] < N) & mask[None, :], other=0.) * w
        acc_p += p
        compress_idx += 1
        compress_start += stride
        
    if return_p:
        # acc_p = tl.where(tl.arange(0, BLOCK_SIZE_K)[None, :] == (off_n // select_size)[:, None], 9999, acc_p)
        tl.store(SP + off_n[:, None] * sp_stride_n + select_idx[None, :] * sp_stride_k, 
                  acc_p, mask=(off_n[:, None] < N) & (select_idx[None, :] < num_selcct_blocks))


def _compute_select_score_triton(
    compress_score, 
    kernel_size, 
    stride, 
    select_size, 
    top_k, 
    kv_len, 
    num_slc_score_heads, 
    slc_att_num_init_blocks=1, 
    slc_att_num_local_blocks=2,
):
    """
    launch _compute_select_probs kernel
    used in prefill, adapted from mtmlp/megatron-mlp feature/nsa-triton-zwexp-mgt-serving
    """
    B, QH, q_len, _ = compress_score.shape
    assert QH % num_slc_score_heads == 0, f"{QH=} {num_slc_score_heads=}"

    num_selcct_blocks = triton.cdiv(kv_len, select_size)
    top_k = min(num_selcct_blocks, top_k)

    BLOCK_SIZE_K = triton.next_power_of_2(num_selcct_blocks)
    select_probs = torch.zeros(B, num_slc_score_heads, q_len, num_selcct_blocks, device=compress_score.device, dtype=torch.float32)
    # indices = torch.empty(B, KH, N, num_selcct_blocks, dtype=torch.int32, device=probs.device,)
    BLOCK_SIZE_N = 32
    if q_len > 8192:
        BLOCK_SIZE_N = 8
    elif q_len >= 1024 * 32:
        BLOCK_SIZE_N = 4
    elif q_len >= 1024 * 64:
        BLOCK_SIZE_N = 2
    grid=lambda meta: (B * num_slc_score_heads, triton.cdiv(q_len, meta['CHUNK_N']), triton.cdiv(meta['CHUNK_N'], meta['BLOCK_SIZE_N']))
    kwargs = {"BLOCK_SIZE_N": BLOCK_SIZE_N, "num_warps": 4, "num_stages": 4}
    num_com_block = (kv_len - kernel_size) // stride + 1
    _compute_select_score_kernel[grid](
        compress_score, select_probs,
        *compress_score.stride(),
        *(select_probs.stride()),
        kernel_size, stride, 
        select_size, num_selcct_blocks, top_k, slc_att_num_init_blocks, slc_att_num_local_blocks,
        True,
        B, q_len, kv_len, num_slc_score_heads, num_com_block,
        BLOCK_SIZE_K,
        **kwargs
    )
    return select_probs

def _transform_score_torch_aligned(
    compress_score: torch.Tensor, # [b, qh, qlen, compressed_kv_len]
    kv_len: int,
    kernel_size: int,
    stride: int,
    select_size: int,
    top_k: int,
    slc_att_num_init_blocks: int,
    slc_att_num_local_blocks: int,
    num_slc_score_heads: int,
    virtual_k_group_agg_type: str = "sum",
):
    keep_prob_value = 999999
    b, qh, m, _ = compress_score.shape
    assert qh >= num_slc_score_heads, f"qheads must be greater than slc_score_heads, {qh=} {num_slc_score_heads=}"
    n = kv_len
    # m == n because padding for now, even for chunked-prefill
    assert m == n, f"{m=} {n=} {compress_score.shape=}"
    
    # [B, KH, qlen, slc_block_num]
    select_probs = _compute_select_score_triton(
        compress_score=compress_score,
        kernel_size=kernel_size,
        stride=stride,
        select_size=select_size,
        top_k=top_k,
        kv_len=kv_len,
        num_slc_score_heads=qh,
        slc_att_num_init_blocks=slc_att_num_init_blocks,
        slc_att_num_local_blocks=slc_att_num_local_blocks,
    )
    g = qh // num_slc_score_heads
    select_probs = rearrange(
        select_probs, "b (sh g) m cn -> b sh g m cn", g=g
    )
    if virtual_k_group_agg_type == "max":
        select_probs = select_probs.max(dim=2)[0]
    else:
        select_probs = select_probs.sum(2)

    if slc_att_num_init_blocks > 0:
        rows = torch.arange(n)[:,None]
        cols = torch.arange(slc_att_num_init_blocks)[None].expand(n, -1)
        # For each token, init block cannot exceed the number of select blocks
        cols = torch.clamp_max(cols, rows // select_size)
        select_probs[:, :, rows, cols] = keep_prob_value
    if slc_att_num_local_blocks > 0:
        off1 = torch.arange(n) // select_size
        off2 = torch.arange(slc_att_num_local_blocks)
        off = torch.clamp_min(off1[:,None] - off2[None], 0)
        rows = torch.arange(n)[:,None]
        select_probs[:, :, rows, off] = keep_prob_value
    return select_probs


def _fill_topk_idx_torch_aligned(
    select_probs: torch.Tensor, # [b, h, qlen, slc_block_num]
    kv_len: int,
    select_size: int,
    top_k: int,
):
    _, _, m, _ = select_probs.shape
    n = kv_len

    num_selcct_blocks = math.ceil(n / select_size)
    if (1 == m < n):  # generating:
        _, indices = torch.topk(select_probs, k=min(num_selcct_blocks, top_k),
                                    dim=-1)  # [b, kh, m, num_selcct_blocks]
        indices[:, :, 0, (n - 1) // select_size + 1:] = num_selcct_blocks
    else:
        _, indices = torch.topk(select_probs, k=min(num_selcct_blocks, top_k),
                                    dim=-1)  # [b, kh, m, num_selcct_blocks]
        for start in range(0, n, select_size):
            # Use num_selcct_blocks as mask, later scatter operation will uniformly distribute masked positions to this location
            indices[:, :, start:start + select_size, (start + select_size) // select_size:] = num_selcct_blocks
    return indices


def _transform_score_to_topk_torch(
    compress_score: torch.Tensor, # [total_score_len, num_q_heads]
    kv_lens: torch.Tensor, # [batch_size]
    q_indptr: torch.Tensor, # [batch_size + 1]
    compress_score_indptr: torch.Tensor, # [batch_size + 1]
    kernel_size: int,
    stride: int,
    select_size: int,
    top_k: int,
    slc_att_num_init_blocks: int,
    slc_att_num_local_blocks: int,
    num_slc_score_heads: int,
    virtual_k_group_agg_type: str,
):
    _ , num_q_heads = compress_score.shape
    total_q_len = q_indptr[-1]
    topk_idx = q_indptr.new_zeros(total_q_len, num_slc_score_heads, top_k)

    for q_start, q_end, compress_score_start, compress_score_end, kv_len in zip(
        q_indptr[:-1], q_indptr[1:],
        compress_score_indptr[:-1], compress_score_indptr[1:],
        kv_lens
    ):
        q_len = q_end - q_start
        per_seq_compress_score = compress_score[compress_score_start:compress_score_end,:].view(q_len, -1, num_q_heads).transpose(1,2)
        padded_per_seq_compress_score = compress_score.new_zeros(
            (kv_len, *per_seq_compress_score.shape[1:])
        )
        padded_per_seq_compress_score[-q_len:,:,:] = per_seq_compress_score
        per_seq_select_score = _transform_score_torch_aligned(
            compress_score=padded_per_seq_compress_score.transpose(0,1).unsqueeze(0),
            kv_len=kv_len.item(),
            kernel_size=kernel_size,
            stride=stride,
            select_size=select_size,
            top_k=top_k,
            slc_att_num_init_blocks=slc_att_num_init_blocks,
            slc_att_num_local_blocks=slc_att_num_local_blocks,
            num_slc_score_heads=num_slc_score_heads,
            virtual_k_group_agg_type=virtual_k_group_agg_type,
        )
        per_seq_topk_idx = _fill_topk_idx_torch_aligned(
            select_probs=per_seq_select_score,
            kv_len=kv_len.item(),
            select_size=select_size,
            top_k=top_k,
        )
        topk_idx[q_start:q_end,:, :per_seq_topk_idx.shape[-1]] = per_seq_topk_idx.squeeze(0).transpose(0,1)[-q_len:,:,:]

    return topk_idx

def compress_attention_torch(
    q: torch.Tensor, # [total_q_len, num_q_heads, qk_head_dim]
    compressed_k_cache: torch.Tensor, # [kv_size, num_kv_heads, qk_head_dim]
    compressed_v_cache: torch.Tensor, # [kv_size, num_kv_heads, v_head_dim]
    compressed_kv_indptr: torch.Tensor, # [batch_size + 1]
    compressed_kv_indices: torch.Tensor, 
    kv_lens: torch.Tensor, # [batch_size]
    q_indptr: torch.Tensor, # [batch_size + 1]
    compress_score_indptr: torch.Tensor, # [batch_size + 1]
    sm_scale: float,
    kernel_size: int,
    stride: int,
    select_size: int,
    top_k: int,
    slc_att_num_init_blocks: int,
    slc_att_num_local_blocks: int,
    num_slc_score_heads: int,
    virtual_k_group_agg_type: str = "sum",
)-> Tuple[torch.Tensor, torch.Tensor]: # o [total_q_len, num_q_heads, v_head_dim], topk_idx [total_q_len, num_slc_score_heads, top_k]

    o, compress_score = _compress_attention_with_score_torch(
        q=q,
        compressed_k_cache=compressed_k_cache,
        compressed_v_cache=compressed_v_cache,
        compressed_kv_indptr=compressed_kv_indptr,
        compressed_kv_indices=compressed_kv_indices,
        kv_lens=kv_lens,
        q_indptr=q_indptr,
        score_indptr=compress_score_indptr,
        kernel_size=kernel_size,
        stride=stride,
        sm_scale=sm_scale,
    )

    topk_idx = _transform_score_to_topk_torch(
        compress_score=compress_score,
        kv_lens = kv_lens,
        q_indptr=q_indptr,
        compress_score_indptr=compress_score_indptr,
        kernel_size=kernel_size,
        stride=stride,
        select_size=select_size,
        top_k=top_k,
        slc_att_num_init_blocks=slc_att_num_init_blocks,
        slc_att_num_local_blocks=slc_att_num_local_blocks,
        num_slc_score_heads=num_slc_score_heads,
        virtual_k_group_agg_type=virtual_k_group_agg_type
    )

    return o, topk_idx

@triton.jit
def _get_attention_score_decode_kernel(
    q_ptr, # [batch_size, num_q_heads, head_dim]
    k_buffer, # [buffer_size, num_kv_heads, head_dim]
    lse_ptr, # [batch_size, num_q_heads]
    score_ptr, # [total_k_len, num_q_heads, num_kv_heads]
    # seqlens
    k_indptr, # [batch_size+1]
    k_indices, # [total_k_len]
    # shape
    NUM_SHARE_Q_HEADS,
    HEAD_DIM,
    # sm_scale
    sm_scale,
    # stride
    stride_qn,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_ln,
    stride_lh,
    stride_sn,
    stride_sh,
    # META parameters
    BLOCK_SIZE_K: tl.constexpr,  # k block size
    BLOCK_SIZE_D: tl.constexpr,  # head dim block size
):
    # get batch id and head id
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_kh = pid_h // NUM_SHARE_Q_HEADS
    # init head dim offset
    offs_d = tl.arange(0, BLOCK_SIZE_D)
    mask_d = offs_d < HEAD_DIM
    # init q ptr and load q
    offs_q = pid_b * stride_qn + pid_h * stride_qh + offs_d
    q = tl.load(q_ptr + offs_q, mask=mask_d, other=0.0)
    # inti lse ptr and load lse
    offs_l = pid_b * stride_ln + pid_h * stride_lh
    lse = tl.load(lse_ptr + offs_l)
    # get q k start and len after rmpad
    k_start = tl.load(k_indptr + pid_b)
    k_end = tl.load(k_indptr + pid_b + 1)
    k_len = k_end - k_start
    k_blocks = tl.cdiv(k_len, BLOCK_SIZE_K)
    for k_block_id in range(k_blocks):
        k_block_start = k_start + k_block_id * BLOCK_SIZE_K
        # load k_loc
        offs_k = k_block_start + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < k_end
        kv_loc = tl.load(k_indices + offs_k, mask=mask_k, other=0)
        # init k ptr and mask and load k
        mask_buf_k = mask_k[:, None] & mask_d[None, :]
        offs_buf_k = kv_loc[:, None] * stride_kn + pid_kh * stride_kh + offs_d[None, :]               
        k = tl.load(k_buffer + offs_buf_k, mask=mask_buf_k, other=0.0)
        # compute qk 
        qk = tl.sum(q[None, :] * k, -1) * sm_scale 
        # compute score
        score = tl.exp(qk - lse)
        # save output
        offs_s = offs_k * stride_sn + pid_h * stride_sh
        tl.store(score_ptr + offs_s, score.to(score_ptr.dtype.element_ty), mask=mask_k)


def _get_attention_score_decode_triton(
    q: torch.Tensor,  # [total_query_len, num_q_heads, head_dim]
    k_buffer: torch.Tensor,  # [kv_buffer_size, num_kv_heads, head_dim]
    lse: torch.Tensor,  # [total_query_len, num_q_heads]
    score: torch.Tensor, # [total_key_len, num_q_heads]
    k_indptr: torch.Tensor, # [batch_size + 1]
    k_indices: torch.Tensor, # [total_key_len]
    sm_scale: float,
) -> torch.Tensor:
    # shape
    batch_size, num_q_heads, head_dim = q.shape
    _, num_k_heads, _ = k_buffer.shape
    # gqa
    assert num_q_heads % num_k_heads == 0
    num_share_q_heads = num_q_heads // num_k_heads
    # launch kernel
    BLOCK_SIZE_K = 64
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
    _get_attention_score_decode_kernel[(batch_size, num_q_heads)](
        q_ptr=q,
        k_buffer=k_buffer,
        lse_ptr=lse,
        score_ptr=score,
        k_indptr=k_indptr,
        k_indices=k_indices,
        NUM_SHARE_Q_HEADS=num_share_q_heads,
        HEAD_DIM=head_dim,
        sm_scale=sm_scale,
        stride_qn=q.stride(0),
        stride_qh=q.stride(1),
        stride_kn=k_buffer.stride(0),
        stride_kh=k_buffer.stride(1),
        stride_ln=lse.stride(0),
        stride_lh=lse.stride(1),
        stride_sn=score.stride(0),
        stride_sh=score.stride(1),
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        # num_warps=8,
        # num_stages=3,
    )

def _attention_with_score_decode_triton(
    q,
    k_buffer,
    v_buffer,
    o,
    score,
    kv_indptr,
    kv_indices,
    req_to_token,
    sm_scale,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Flash attention for decode with score.

    Args:
        q (torch.Tensor): query, shape [batch_size, num_q_heads, qk_head_dim]
        k_buffer (torch.Tensor): key, shape [kv_buffer_size, num_kv_heads, qk_head_dim]
        v_buffer (torch.Tensor): value, shape [kv_buffer_size, num_kv_heads, v_head_dim]
        o (torch.Tensor): output, shape [batch_size, num_q_heads, v_head_dim]
        score (torch.Tensor): attention softmax log-sum-exp, shape [total_kv_len, num_q_heads]
        kv_indptr (torch.Tensor): key value indptr, shape [batch_size + 1]
        kv_indices (torch.Tensor): key value indices, shape [total_kv_len]
        req_to_token (torch.Tensor): request to tokens, shape [batch_size, max_kv_len]
        sm_scale (float): softmax scale
    """
    batch_size, num_q_heads, qk_head_dim = q.shape

    if qk_head_dim <=256:
        out, lse, *_ = flash_attn_with_kvcache(
            q=q,
            k_cache=k_buffer.unsqueeze(1),
            v_cache=v_buffer.unsqueeze(1),
            page_table = req_to_token,
            cache_seqlens=(kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32),
            cu_seqlens_q=torch.arange(0, batch_size + 1, dtype=torch.int32, device=kv_indptr.device),
            cu_seqlens_k_new=kv_indptr.to(torch.int32),
            max_seqlen_q=1,
            softmax_scale=sm_scale,
            causal=False,
            window_size=(-1, -1),
            return_softmax_lse = True,
        )
        o[:] = out
        lse = lse.transpose(0, 1) # [batch_size, num_q_heads]
    else:
        max_kv_splits = 8
        _, _, v_head_dim = v_buffer.shape

        attn_logits = torch.zeros(
            (batch_size, num_q_heads, max_kv_splits, v_head_dim),
            dtype=torch.float32,
            device=q.device,
        )
        # When k_len < num_kv_splits, ignore positions where computation did not occur when calculating logsumexp
        attn_lse = torch.full(
            (batch_size, num_q_heads, max_kv_splits), -torch.inf,
            dtype=torch.float32,
            device=q.device,
        )
        # todo: use get_num_kv_splits_triton or tune
        num_kv_splits = torch.full(
            (batch_size,), 8,
            dtype=torch.int32, device=q.device
        )
        decode_attention_fwd(
            q=q,
            k_buffer=k_buffer,
            v_buffer=v_buffer,
            o=o,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            attn_logits=attn_logits,
            attn_lse=attn_lse,
            num_kv_splits=num_kv_splits,
            max_kv_splits=max_kv_splits,
            sm_scale=sm_scale,
        )
        lse = attn_lse.logsumexp(-1) # [batch_size, num_q_heads]

    _get_attention_score_decode_triton(
        q=q,
        k_buffer=k_buffer,
        lse=lse,
        score=score,
        k_indptr=kv_indptr,
        k_indices=kv_indices,
        sm_scale=sm_scale,
    )

@triton.jit
def _transform_score_decode_kernel(
    flat_com_score, # [com_block_num, nh]
    flat_slc_score, # [slc_block_num, nh]
    stride_com_b, stride_com_h,
    stride_slc_b, stride_slc_h,
    cu_com_blocks, cu_slc_blocks,
    com_stride, com_size, slc_size,
    num_head,
    slc_init_blocks, BLOCK_SLC_I: tl.constexpr,
    slc_local_blocks, BLOCK_SLC_L: tl.constexpr,
    BH: tl.constexpr=32, # for num_head
    BN: tl.constexpr=128
):
    pid0 = tl.program_id(0) # seq_i
    com_block_start = tl.load(cu_com_blocks+pid0)
    com_block_end=  tl.load(cu_com_blocks+pid0+1)
    com_block_num = com_block_end - com_block_start
    if com_block_num == 0:
        return
    # If there is a com block, there must be an slc block
    slc_block_start = tl.load(cu_slc_blocks+pid0)
    slc_block_end = tl.load(cu_slc_blocks+pid0+1)
    slc_block_num = slc_block_end - slc_block_start
    # loops = (slc_size + 2 * (com_size - com_stride) - com_size) // com_stride + 1
    loops = (slc_size + com_size - com_stride) // com_stride
    head_ids = tl.arange(0, BH)
    mask_head = head_ids < num_head
    slc_iters = (slc_block_num+BN-1) // BN
    for i in range(slc_iters):
        slc_ids = i * BN + tl.arange(0, BN)
        mask_slc = slc_ids < slc_block_num
        slc_start = slc_ids * slc_size
        slc_end = slc_start + slc_size
        acc_p = tl.zeros((BN, BH), dtype=tl.float32)
        # com_id means block_id, com_start means the starting position of this block in original kv
        com_ids = (slc_ids * slc_size - com_size) // com_stride + 1
        com_start = com_ids * com_stride
        for _ in range(loops):
            com_end = com_start + com_size
            area = tl.minimum(com_end, slc_end) - tl.maximum(com_start, slc_start)
            w = area / com_stride
            mask_com = (com_ids >= 0) & (com_ids < com_block_num)
            p = tl.load(
                flat_com_score + (com_block_start + com_ids[:,None])*stride_com_b + head_ids[None,:]*stride_com_h,
                mask=mask_com[:,None] & mask_head[None,:], other=0
            )
            acc_p += p * w[:,None]
            com_ids += 1
            com_start += com_stride

        tl.store(
            flat_slc_score + (slc_block_start + slc_ids[:,None])*stride_slc_b + head_ids[None,:]*stride_slc_h,
            acc_p,
            mask = mask_slc[:,None] & mask_head[None,:]
        )
    # set init block and local block
    for i in range(slc_iters):
        slc_ids = i * BN + tl.arange(0, BN)
        mask1 = slc_ids < tl.minimum(slc_block_num, slc_init_blocks)
        mask2 = slc_ids >= tl.maximum(0, slc_block_num - slc_local_blocks)
        mask3 = slc_ids < slc_block_num
        mask = (mask1 | mask2) & mask3
        tl.store(
            flat_slc_score + (slc_block_start + slc_ids[:,None]) * stride_slc_b + head_ids[None,:]*stride_slc_h,
            9999,
            mask=mask[:,None] & mask_head[None,:]
        )


def _transform_score_decode_triton(
    compress_score: torch.Tensor, # [com_blocks, nh]
    num_slc_score_heads,
    compressed_kv_indptr, 
    select_indptr,
    stride, 
    kernel_size, 
    select_size,
    slc_att_num_init_blocks, 
    slc_att_num_local_blocks,
    select_score: torch.Tensor,
):
    bs = compressed_kv_indptr.shape[0] -1

    grid = (bs,)
    _transform_score_decode_kernel[grid](
        compress_score, select_score,
        *compress_score.stride(),
        *select_score.stride(),
        compressed_kv_indptr, 
        select_indptr,
        stride, 
        kernel_size, 
        select_size,
        num_slc_score_heads,
        slc_att_num_init_blocks, 
        triton.next_power_of_2(slc_att_num_init_blocks),
        slc_att_num_local_blocks, 
        triton.next_power_of_2(slc_att_num_local_blocks),
        BH=triton.next_power_of_2(num_slc_score_heads)
    )
    return select_score


@triton.jit
def _fill_topkidx_decode_kernel(
    slc_score_ptr,  # [total_slc_blocks, nh]
    cu_slc_blocks,  # [bs+1]
    topk_idx_ptr,       # [bs, nh, topk]
    stride_sco_l, stride_sco_h,
    stride_idx_b, stride_idx_h, stride_idx_d,
    top_k: tl.constexpr,
    BLOCK_L: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    slc_start_idx = tl.load(cu_slc_blocks + pid_b)
    slc_end_idx = tl.load(cu_slc_blocks + pid_b + 1)
    slc_block_num = slc_end_idx - slc_start_idx
    slc_score_ptr = slc_score_ptr + slc_start_idx * stride_sco_l + pid_h * stride_sco_h
    off = tl.arange(0, BLOCK_L)
    scores = tl.load(slc_score_ptr+off*stride_sco_l, mask=off < slc_block_num, other=float("-inf"))
    loops = tl.minimum(top_k, slc_block_num)
    topk_idx_ptr = topk_idx_ptr + pid_b * stride_idx_b + pid_h *stride_idx_h
    for i in range(loops):
        max_idx = tl.argmax(scores, axis=0)
        scores = tl.where(off == max_idx, float("-inf"), scores)
        tl.store(topk_idx_ptr + i*stride_idx_d, max_idx)


def _fill_topkidx_decode_triton(
    select_score,
    select_indptr,
    topk_idx,
    max_select_len,
):
    bs, num_slc_score_heads, top_k = topk_idx.shape

    _fill_topkidx_decode_kernel[(bs, num_slc_score_heads)](
        select_score, 
        select_indptr,
        topk_idx,
        *select_score.stride(),
        *topk_idx.stride(),
        top_k,
        # BLOCK_L=triton.next_power_of_2(torch.max(slc_blocks).item())
        BLOCK_L=triton.next_power_of_2(max_select_len)
    )

def compress_attention_decode_triton(
    q: torch.Tensor, # [total_q_len, num_q_heads, qk_head_dim]
    compressed_k_cache: torch.Tensor, # [kv_size, num_kv_heads, qk_head_dim]
    compressed_v_cache: torch.Tensor, # [kv_size, num_kv_heads, v_head_dim]
    compress_score_buffer: torch.Tensor, # [total_compressed_kv_len, num_q_heads]
    select_score_buffer: torch.Tensor, # [total_select_len, num_q_heads]
    compressed_kv_indptr: torch.Tensor, # [batch_size + 1]
    compressed_kv_indices: torch.Tensor, 
    req_to_compressed_token: torch.Tensor, # [batch_size, max_kv_len]
    select_indptr: torch.Tensor, # [batch_size]
    sm_scale: float,
    kernel_size: int,
    stride: int,
    select_size: int,
    top_k: int,
    slc_att_num_init_blocks: int,
    slc_att_num_local_blocks: int,
    num_slc_score_heads: int,
    virtual_k_group_agg_type: str,
    max_context_len: int,
)-> Tuple[torch.Tensor, torch.Tensor]: # o [total_q_len, num_q_heads, v_head_dim], topk_idx [total_q_len, num_slc_score_heads, top_k]

    bs, num_q_heads, _ = q.shape
    assert num_q_heads >= num_slc_score_heads, f"qheads must be greater than slc_score_heads, {num_q_heads=} {num_slc_score_heads=}"
    _, _, v_head_dim = compressed_v_cache.shape

    o = q.new_empty((bs, num_q_heads, v_head_dim))

    _attention_with_score_decode_triton(
        q=q,
        k_buffer=compressed_k_cache,
        v_buffer=compressed_v_cache,
        o=o,
        score=compress_score_buffer,
        kv_indptr=compressed_kv_indptr,
        kv_indices=compressed_kv_indices,
        req_to_token=req_to_compressed_token,
        sm_scale=sm_scale,
    )

    _transform_score_decode_triton(
        compress_score = compress_score_buffer,
        num_slc_score_heads = num_q_heads,
        compressed_kv_indptr = compressed_kv_indptr,
        select_indptr = select_indptr,
        stride = stride,
        kernel_size = kernel_size,
        select_size = select_size,
        slc_att_num_init_blocks = slc_att_num_init_blocks,
        slc_att_num_local_blocks = slc_att_num_local_blocks,
        select_score = select_score_buffer,
    )

    g = num_q_heads // num_slc_score_heads
    select_score_buffer = rearrange(
        select_score_buffer, "b (sh g) -> b sh g", g=g
    )
    if virtual_k_group_agg_type == "max":
        select_score_buffer = select_score_buffer.max(dim=2)[0]
    else:
        select_score_buffer = select_score_buffer.sum(2)

    topk_idx = select_indptr.new_zeros(bs, num_slc_score_heads, top_k)
    max_select_len = math.ceil(max_context_len / select_size)
    _fill_topkidx_decode_triton(
        select_score=select_score_buffer, 
        select_indptr=select_indptr, 
        topk_idx=topk_idx, 
        max_select_len=max_select_len
    )

    return o, topk_idx
