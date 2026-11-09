from typing import Optional

import torch
import triton
import triton.language as tl


# Triton implementation
@triton.jit
def _act_quant_kernel(
    X_ptr,
    Y_ptr,
    S_ptr,
    M,
    N,
    group_size: tl.constexpr,
    round_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel for activation quantization.

    Each block processes BLOCK_M rows and group_size columns.
    """
    # Get block IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # FP8 constants
    fp8_min = -448.0
    fp8_max = 448.0
    fp8_max_inv = 1.0 / fp8_max

    # Calculate row and column offsets
    row_start = pid_m * BLOCK_M
    col_start = pid_n * group_size

    # Create offset arrays
    rows = row_start + tl.arange(0, BLOCK_M)
    cols = col_start + tl.arange(0, BLOCK_N)

    # Mask for valid rows and columns
    row_mask = rows < M
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]

    # Load input data
    x_ptrs = X_ptr + rows[:, None] * N + cols[None, :]
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Compute absolute max along columns (group_size dimension) for each row
    x_abs = tl.abs(x)
    amax = tl.max(x_abs, axis=1)  # Shape: (BLOCK_M,)

    # Clamp amax to avoid division by zero
    amax = tl.maximum(amax, 1e-4)

    # Compute scale
    if round_scale:
        # Fast round scale using bit manipulation approximation
        # This is a simplified version - the exact bit manipulation is harder in Triton
        # Using log2 + ceil + pow2 as approximation
        log_val = tl.log2(amax * fp8_max_inv)
        log_ceil = tl.ceil(log_val)
        scale = tl.exp2(log_ceil)
    else:
        scale = amax * fp8_max_inv

    # Quantize: y = clamp(x / scale, fp8_min, fp8_max)
    scale_broadcast = scale[:, None]
    y = x / scale_broadcast
    y = tl.minimum(tl.maximum(y, fp8_min), fp8_max)

    # Store quantized output
    y_ptrs = Y_ptr + rows[:, None] * N + cols[None, :]
    tl.store(y_ptrs, y, mask=mask)

    # Store scales
    s_cols = pid_n
    s_ptrs = S_ptr + rows * (N // group_size) + s_cols
    s_mask = row_mask
    tl.store(s_ptrs, scale, mask=s_mask)


def act_quant(x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization with Triton.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.
        scale_fmt (Optional[str], optional): The format of the scale. Default is None.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.size(-1) % block_size == 0, f"Last dimension size must be divisible by block_size (block_size={block_size})"

    # Flatten all dims except last
    N = x.size(-1)
    x_flat = x.view(-1, N)
    M = x_flat.size(0)

    # Allocate output tensors
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    y_flat = y.view(-1, N)
    s = x.new_empty(*x.size()[:-1], N // block_size, dtype=torch.float32)
    s_flat = s.view(-1, N // block_size)

    if x.shape[0] == 0:
        return y, s
    # Launch kernel
    BLOCK_M = 32
    BLOCK_N = block_size
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, block_size))
    round_scale = scale_fmt is not None

    _act_quant_kernel[grid](
        x_flat,
        y_flat,
        s_flat,
        M,
        N,
        group_size=block_size,
        round_scale=round_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_stages=0 if round_scale else 2,
    )

    return y, s


@triton.jit
def triton_li_rag_kernel(
    q_ptr,  # [s, h, d]
    w_ptr,  # [s, h]
    k_ptr,  # [l, d]
    r_ptr,  # [s, l]
    s_q,
    s_k,
    q_stride_s,
    q_stride_h,
    q_stride_d,
    w_stride_s,
    w_stride_h,
    k_stride_l,
    k_stride_d,
    r_stride_s,
    r_stride_l,
    h: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    q_num_per_m: tl.constexpr = BM // h

    off_s = pid * q_num_per_m + tl.arange(0, q_num_per_m)
    mask_s = off_s < s_q
    off_h = tl.arange(0, h)
    off_d = tl.arange(0, BK)
    # [BK, q_num_per_m, h]
    q = tl.load(
        q_ptr + off_s[None, :, None] * q_stride_s + off_h[None, None, :] * q_stride_h + off_d[:, None, None] * q_stride_d,
        mask=mask_s[None, :, None],
        other=0.0,
    )
    q = tl.reshape(q, (BK, BM))
    w = tl.load(
        w_ptr + off_s[:, None] * w_stride_s + off_h[None, :] * w_stride_h,
        mask=mask_s[:, None],
        other=0.0,
    )
    w = tl.reshape(w, (BM,))

    s_k_loops = tl.cdiv(s_k, BN)
    off_n = tl.arange(0, BN)
    for _i in tl.range(s_k_loops, num_stages=3):
        mask_n = off_n < s_k
        # [BN, BK]
        k = tl.load(k_ptr + off_n[:, None] * k_stride_l + off_d[None, :] * k_stride_d, mask=mask_n[:, None], other=0.0)
        # [BN, BM]
        s = tl.dot(k, q)
        s = tl.maximum(s, 0.0)
        s = s * w[None, :]
        # [BN, q_num_per_m]
        r = tl.reshape(s, (BN, q_num_per_m, h)).sum(axis=2)
        tl.store(r_ptr + off_s[None, :] * r_stride_s + off_n[:, None] * r_stride_l, r, mask=mask_s[None, :] & mask_n[:, None])
        off_n += BN


def indexer_qk_rag(
    query: torch.Tensor,  # [s, h, d]
    weights: torch.Tensor,  # [s, h]
    key: torch.Tensor,  # [l, d]
    logits: torch.Tensor,  # [s, max_l]
):
    s, h, d = query.shape
    s_, h_ = weights.shape
    kv_len, d_ = key.shape
    assert s == s_ and h == h_ and d == d_ and logits.shape[0] == s
    assert (d > 0) & ((d & (d - 1)) == 0)  # d is power of 2
    assert (h > 0) & ((h & (h - 1)) == 0)  # h is power of 2
    BM = 128
    BN = 128
    BK = d
    assert BM % h == 0
    q_num_per_m = BM // h
    grid = (triton.cdiv(s, q_num_per_m),)
    triton_li_rag_kernel[grid](
        query,
        weights,
        key,
        logits,
        s,
        kv_len,
        *query.stride(),
        *weights.stride(),
        *key.stride(),
        *logits.stride(),
        h,
        BM=BM,
        BN=BN,
        BK=BK,
    )
