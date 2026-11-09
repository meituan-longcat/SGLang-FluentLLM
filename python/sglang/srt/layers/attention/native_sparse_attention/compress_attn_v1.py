import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.triton_ops.decode_attention import decode_attention_fwd


def get_compressed_seqlens(seqlens: torch.Tensor, kernel_size: int, kernel_stride: int):
    """
    Compute seqlens after compression.

    Args:
        seqlens (torch.Tensor): [num_batch_seqs]
        kernel_size (int): compressed block token num
        kernel_stride (int): compressed block stride token num
    Returns:
        seqblocks (torch.Tensor): [num_batch_seqs]
    """
    seqblocks = torch.clamp_min((seqlens - kernel_size) // kernel_stride + 1, 0)
    return seqblocks


def get_compressed_buffer_loc(
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
):
    """
    Get the compressed key and value location in the kv cache.

    Args:
        forward_batch (ForwardBatch): batch info
    Returns:
        compressed_kv_buffer_loc (torch.Tensor): [total_block_len]
    """
    seq_lens_block = get_compressed_seqlens(
        seqlens=seq_lens,
        kernel_size=kernel_size,
        kernel_stride=kernel_stride,
    )

    max_blocks = seq_lens_block.max()
    num_seqs = len(seq_lens_block)

    all_block_indices = torch.arange(max_blocks, device=seq_lens_block.device).unsqueeze(0).expand(num_seqs, -1)  # [num_seqs, max_blocks]

    token_indices = all_block_indices * kernel_stride

    valid_mask = all_block_indices < seq_lens_block.unsqueeze(1)
    seq_ids, block_ids = valid_mask.nonzero(as_tuple=True)

    compressed_kv_buffer_loc = req_to_token[req_pool_indices[seq_ids], token_indices[seq_ids, block_ids]] // kernel_stride

    return compressed_kv_buffer_loc


@triton.jit
def _get_attention_score_kernel(
    q_ptr,  # [batch_size, num_q_heads, head_dim]
    k_buffer,  # [buffer_size, num_kv_heads, head_dim]
    lse_ptr,  # [batch_size, num_q_heads]
    score_ptr,  # [total_k_len, num_q_heads, num_kv_heads]
    # seqlens
    k_indptr,  # [batch_size+1]
    k_indices,  # [total_k_len]
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


def get_attention_score(
    q: torch.Tensor,  # [total_query_len, num_q_heads, head_dim]
    k_buffer: torch.Tensor,  # [kv_buffer_size, num_kv_heads, head_dim]
    lse: torch.Tensor,  # [total_query_len, num_q_heads]
    score: torch.Tensor,  # [total_key_len, num_q_heads]
    k_indptr: torch.Tensor,  # [batch_size + 1]
    k_indices: torch.Tensor,  # [total_key_len]
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
    _get_attention_score_kernel[(batch_size, num_q_heads)](
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


def compress_attention_decode(
    q,
    k_buffer,
    v_buffer,
    o,
    score,
    kv_indptr,
    kv_indices,
    sm_scale,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Flash attention for decode.

    Args:
        q (torch.Tensor): query, shape [batch_size, num_q_heads, qk_head_dim]
        k_buffer (torch.Tensor): key, shape [kv_buffer_size, num_kv_heads, qk_head_dim]
        v_buffer (torch.Tensor): value, shape [kv_buffer_size, num_kv_heads, v_head_dim]
        o (torch.Tensor): output, shape [batch_size, num_q_heads, v_head_dim]
        score (torch.Tensor): attention softmax log-sum-exp, shape [total_kv_len, num_q_heads]
        kv_indptr (torch.Tensor): key value indptr, shape [batch_size + 1]
        kv_indices (torch.Tensor): key value indices, shape [total_kv_len]
        attn_logits (torch.Tensor): attention logits, shape [batch_size, num_q_heads, max_kv_len]
        sm_scale (float): softmax scale, default to 1/sqrt(head_dim)
    """
    num_kv_splits = 8
    batch_size, num_q_heads, qk_head_dim = q.shape
    _, num_kv_heads, v_head_dim = v_buffer.shape

    attn_logits = torch.empty(
        (batch_size, num_q_heads, num_kv_splits, v_head_dim + 1),
        dtype=torch.float32,
        device=q.device,
    )
    attn_logits[:, :, :, :v_head_dim] = 0
    attn_logits[
        :, :, :, v_head_dim
    ] = -torch.inf  # When k_len < num_kv_splits, ignore positions where computation did not occur when calculating logsumexp
    decode_attention_fwd(
        q=q,
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        o=o,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        attn_logits=attn_logits,
        num_kv_splits=num_kv_splits,
        sm_scale=sm_scale,
    )
    lse_splits = attn_logits[:, :, :, v_head_dim]  # [batch_size, num_q_heads, num_kv_splits]
    lse = lse_splits.logsumexp(-1)  # [batch_size, num_q_heads]

    # k = k_buffer.new_empty((total_kv_len, num_kv_heads, qk_head_dim))
    # for i in range(batch_size):
    #     kv_start = kv_indptr[i]
    #     kv_end = kv_indptr[i+1]
    #     per_seq_kv_indices = kv_indices[kv_start:kv_end]
    #     k[kv_start:kv_end] = k_buffer[per_seq_kv_indices]
    #     cu_seqlens_q[i+1] = i+1

    get_attention_score(
        q=q,
        k_buffer=k_buffer,
        lse=lse,
        score=score,
        k_indptr=kv_indptr,
        k_indices=kv_indices,
        sm_scale=sm_scale,
    )

    # for i in range(batch_size):
    #     kv_slice = slice(kv_indptr[i], kv_indptr[i+1])
    #     seq_k = k_buffer[kv_indices[kv_slice]].transpose(0, 1)
    #     seq_k = seq_k.repeat_interleave(num_q_heads // num_kv_heads, dim=0)
    #     seq_probs = ((q[i][:,None,:] @ seq_k.transpose(-1, -2)).squeeze(1) * sm_scale - lse.to(torch.bfloat16)[i][:, None]).exp().T
    #     assert torch.allclose(score[kv_slice], seq_probs, rtol=0.05, atol=0.05)
