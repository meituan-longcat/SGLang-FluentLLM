import math

import torch
import triton
import triton.language as tl

from sglang.srt.layers.linear import LinearBase


def cumsum_with_padding(
    lens: torch.Tensor,  # [batch_size]
) -> torch.Tensor:  # [batch_size + 1]
    (bs,) = lens.shape
    indptr = lens.new_zeros(bs + 1)
    torch.cumsum(lens, 0, out=indptr[1:])
    return indptr


def get_compressed_kv_indptr(
    kv_indptr: torch.Tensor,  # [batch_size + 1]
    kernel_size: int,
    stride: int,
) -> torch.Tensor:  # [batch_size + 1]
    kv_lens = kv_indptr[1:] - kv_indptr[:-1]
    compressed_kv_lens = torch.clamp_min((kv_lens - kernel_size) // stride + 1, 0)
    compressed_kv_indptr = cumsum_with_padding(compressed_kv_lens)
    return compressed_kv_indptr


def get_compressed_kv_indices(
    kv_indptr: torch.Tensor,  # [batch_size + 1]
    compressed_kv_indptr: torch.Tensor,  # [batch_size + 1]
    kv_indices: torch.Tensor,
    stride: int,
) -> torch.Tensor:
    compressed_kv_indices = kv_indices.new_empty(compressed_kv_indptr[-1])
    for kv_start, kv_end, compressed_kv_start, compressed_kv_end in zip(
        kv_indptr[:-1], kv_indptr[1:], compressed_kv_indptr[:-1], compressed_kv_indptr[1:]
    ):
        per_seq_kv_indices = kv_indices[kv_start:kv_end]
        compressed_kv_len = compressed_kv_end - compressed_kv_start
        compressed_kv_indices_loc = torch.arange(0, compressed_kv_len, device=kv_indices.device)
        kv_indices_loc = compressed_kv_indices_loc * stride
        per_seq_compressed_kv_indices = per_seq_kv_indices[kv_indices_loc] // stride
        compressed_kv_indices[compressed_kv_start:compressed_kv_end] = per_seq_compressed_kv_indices
    return compressed_kv_indices


def _gate_compress_kernel_torch(
    block: torch.Tensor,  # [..., kernel_size, D]
    gate_proj: LinearBase,  # kernel_size * D -> kernel_size
    cmp_rescale_factor: float = 0,
) -> torch.Tensor:  # [..., D]
    gate_o, _ = gate_proj(block.flatten(-2, -1))
    if cmp_rescale_factor > 0:
        gate_o *= cmp_rescale_factor
    gate_score = gate_o.softmax(dim=-1)
    compressed = (gate_score.unsqueeze(-1) * block).sum(-2)
    return compressed


def _gate_compress_torch_aligned(
    x: torch.Tensor,  # [bs, seq_len, num_heads, head_dim]
    gate_proj: LinearBase,  # kernel_size * head_dim -> kernel_size
    kernel_size: int,
    stride: int,
    cmp_rescale_factor: float = 0,
) -> torch.Tensor:  # [bs, num_blocks, num_heads, head_dim]
    x = x.transpose(1, 2)
    B, H, N, D = x.shape
    num_blocks = (N - kernel_size) // stride + 1

    if num_blocks < 1:
        return torch.zeros(B, 0, H, D, dtype=x.dtype, device=x.device)

    # [bs, h, num_blocks, kernel_size, D]
    block_x = torch.cat(
        [
            torch.roll(x, shifts=-1 * idx * stride, dims=-2)[:, :, : num_blocks * stride].reshape(B, H, num_blocks, stride, -1)[
                :, :, :, : min(stride, kernel_size - idx * stride)
            ]
            for idx in range(math.ceil(kernel_size / stride))
        ],
        axis=-2,
    )

    compress_x = _gate_compress_kernel_torch(
        block=block_x,
        gate_proj=gate_proj,
        cmp_rescale_factor=cmp_rescale_factor,
    )

    return compress_x.transpose(1, 2)


def gate_compress_torch(
    kv_buffer: torch.Tensor,  # [kv_size, num_heads, head_dim]
    compressed_kv_buffer: torch.Tensor,  # [compressed_kv_size, num_heads, head_dim]
    gate_proj: LinearBase,  # kernel_size * head_dim -> head_dim
    kv_indptr: torch.Tensor,  # [batch_size + 1]
    compressed_kv_indptr: torch.Tensor,  # [batch_size + 1]
    kv_indices: torch.Tensor,
    compressed_kv_indices: torch.Tensor,
    kernel_size: int,
    stride: int,
    cmp_rescale_factor: float = 0,
):
    """
    Pepare the compressed blocks from buffer into compressed buffer.

    Args:
        forward_batch (ForwardBatch): batch info
        buffer (torch.Tensor): [buffer_size, num_heads, head_dim]
        compressed_buffer (torch.Tensor): [compressed_buffer_size, num_heads, head_dim]
    """

    for kv_start, kv_end, compressed_kv_start, compressed_kv_end in zip(
        kv_indptr[:-1], kv_indptr[1:], compressed_kv_indptr[:-1], compressed_kv_indptr[1:]
    ):
        per_seq_kv_indices = kv_indices[kv_start:kv_end]
        per_seq_compressed_kv_indices = compressed_kv_indices[compressed_kv_start:compressed_kv_end]
        per_seq_kv = kv_buffer[per_seq_kv_indices]
        per_seq_compressed_kv = _gate_compress_torch_aligned(
            x=per_seq_kv.unsqueeze(0),
            gate_proj=gate_proj,
            kernel_size=kernel_size,
            stride=stride,
            cmp_rescale_factor=cmp_rescale_factor,
        )
        compressed_kv_buffer[per_seq_compressed_kv_indices] = per_seq_compressed_kv.squeeze(0)


def gate_compress_decode_torch(
    kv_buffer: torch.Tensor,  # [kv_size, num_heads, head_dim]
    compressed_kv_buffer: torch.Tensor,  # [compressed_kv_size, num_heads, head_dim]
    gate_proj: LinearBase,  # kernel_size * head_dim -> head_dim
    kv_lens: torch.Tensor,  # [batch_size]
    req_pool_indices: torch.Tensor,  # [batch_size]
    req_to_token: torch.Tensor,  # [pool_size, max_kv_len]
    kernel_size: int,
    stride: int,
):
    start_positions = kv_lens - kernel_size  # [batch_size]
    kernel_range = torch.arange(kernel_size, device=kv_lens.device)  # [kernel_size]
    buffer_indices = start_positions[:, None] + kernel_range[None, :]  # [batch_size, kernel_size]

    token_positions = torch.gather(
        input=req_to_token[req_pool_indices],
        dim=1,
        index=buffer_indices.clamp_min(0),
    )  # [batch_size, kernel_size]

    blocks = kv_buffer[token_positions]  # [batch_size, kernel_size, num_heads, head_dim]
    blocks = blocks.transpose(1, 2)  # [batch_size, num_heads, kernel_size, head_dim]

    compressed = _gate_compress_kernel_torch(
        block=blocks,
        gate_proj=gate_proj,
    )  # [batch_size, num_heads, head_dim]

    valid_mask = (kv_lens >= kernel_size) & ((kv_lens - kernel_size) % stride == 0)  # [batch_size]
    compressed_positions = token_positions[:, 0] // stride  # [batch_size]

    compressed_kv_buffer[compressed_positions] = torch.where(
        valid_mask[:, None, None],  # broadcasted to [batch_size, num_heads, head_dim]
        compressed,
        compressed_kv_buffer[compressed_positions],
    )


# TODO: softmax of triton 3.3 have not dim param, following are the code from triton 3.4
@triton.jit
def _softmax_kernel(x, dim=None, keep_dims=False, ieee_rounding=False):
    if dim is None:
        _dim: tl.constexpr = 0
    else:
        _dim: tl.constexpr = dim
    z = x - tl.max(x, _dim, keep_dims=keep_dims)
    num = tl.exp(z)
    den = tl.sum(num, _dim, keep_dims=keep_dims)
    return tl.fdiv(num, den, ieee_rounding)


@triton.jit
def _gate_compress_decode_kernel(
    kv_cache_ptr,
    compressed_kv_cache_ptr,
    gate_weight_ptr,
    kv_lens_ptr,
    req_pool_indices_ptr,
    req_to_token_ptr,
    # strides
    kv_buffer_stride_s,
    kv_buffer_stride_h,
    kv_buffer_stride_d,
    compressed_kv_buffer_stride_s,
    compressed_kv_buffer_stride_h,
    compressed_kv_buffer_stride_d,
    gate_weight_stride_o,
    gate_weight_stride_i,
    req_to_token_stride_b,
    req_to_token_stride_t,
    # shapes
    num_heads,
    head_dim,
    kernel_stride,
    # constants
    cmp_rescale_factor: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_id = tl.program_id(0)
    head_block_id = tl.program_id(1)

    kv_len = tl.load(kv_lens_ptr + batch_id)
    kernel_start_pos = kv_len - KERNEL_SIZE

    # Early exit if compression is not applicable
    # Need:
    #   1) enough tokens (>=kernel_size)
    #   2) aligned position for stride
    if not ((kernel_start_pos >= 0) & (kernel_start_pos % kernel_stride == 0)):
        return

    req_pool_index = tl.load(req_pool_indices_ptr + batch_id)

    # Current head block range: [h_start, h_start + BLOCK_H)
    h_start = head_block_id * BLOCK_H
    h_offsets = h_start + tl.arange(0, BLOCK_H)
    h_mask = h_offsets < num_heads

    k_arange = tl.arange(0, KERNEL_SIZE)

    gate_scores = tl.zeros((BLOCK_H, KERNEL_SIZE), dtype=tl.float32)

    # Phase 1: Compute gate scores via blocked matrix multiplication
    # Equivalent pseudo-code without blocking:
    #   kv_block = kv_cache[h_offsets, kernel_kv_indices, :] # [BLOCK_H, KERNEL_SIZE, head_dim]
    #   kv_flattened = kv_block.reshape(BLOCK_H, KERNEL_SIZE * head_dim)
    #   gate_scores = kv_flattened @ gate_weight # [BLOCK_H, KERNEL_SIZE]
    for k_block_start in tl.range(0, KERNEL_SIZE, BLOCK_K):
        k_block_offsets = k_block_start + tl.arange(0, BLOCK_K)
        k_block_mask = k_block_offsets < KERNEL_SIZE

        # Get token indices for current kernel-block
        k_block_kv_indices = tl.load(
            req_to_token_ptr + req_pool_index * req_to_token_stride_b + (kernel_start_pos + k_block_offsets) * req_to_token_stride_t,
            mask=k_block_mask,
        )  # [BLOCK_K]
        # Feeling this is a bit unnecessary, just split BLOCK_K, splitting headdim has no use case?
        for d_block_start in tl.range(0, head_dim, BLOCK_D):
            d_block_offsets = d_block_start + tl.arange(0, BLOCK_D)
            d_block_mask = d_block_offsets < head_dim

            kv_k_d_block = tl.load(
                kv_cache_ptr
                + h_offsets[:, None, None] * kv_buffer_stride_h
                + k_block_kv_indices[None, :, None] * kv_buffer_stride_s
                + d_block_offsets[None, None, :] * kv_buffer_stride_d,
                mask=h_mask[:, None, None] & k_block_mask[None, :, None] & d_block_mask[None, None, :],
            )  # [BLOCK_H, BLOCK_K, BLOCK_D]
            kv_k_d_block = tl.reshape(kv_k_d_block, (BLOCK_H, BLOCK_K * BLOCK_D))

            # Compute flat offsets for weight matrix addressing
            w_kd_block_offsets = tl.ravel(k_block_offsets[:, None] * head_dim + d_block_offsets[None, :])  # [BLOCK_K * BLOCK_D]
            w_kd_block_mask = tl.ravel(k_block_mask[:, None] & d_block_mask[None, :])  # [BLOCK_K * BLOCK_D]

            w_k_d_block = tl.load(
                gate_weight_ptr + w_kd_block_offsets[:, None] * gate_weight_stride_i + k_arange[None, :] * gate_weight_stride_o,
                mask=w_kd_block_mask[:, None],
            )  # [BLOCK_K * BLOCK_D, KERNEL_SIZE]

            gate_scores = tl.dot(kv_k_d_block, w_k_d_block, acc=gate_scores)
    if cmp_rescale_factor > 0:
        gate_scores *= cmp_rescale_factor
    gate_scores = _softmax_kernel(gate_scores, dim=1, keep_dims=True)  # [BLOCK_H, KERNEL_SIZE]

    k_offsets = kernel_start_pos + k_arange
    # Need to re-fetch kv cache and scores to do element-wise multiplication then reduce
    k_kv_indices = tl.load(
        req_to_token_ptr + req_pool_index * req_to_token_stride_b + k_offsets * req_to_token_stride_t,
    )

    kernel_start_pos_kv_index = tl.load(req_to_token_ptr + req_pool_index * req_to_token_stride_b + kernel_start_pos * req_to_token_stride_t)
    compressed_kv_cache_index = kernel_start_pos_kv_index // kernel_stride

    # Phase 2: Apply gate compression via weighted sum
    # Equivalent pseudo-code without blocking:
    #   kv_block = kv_cache[h_offsets, kernel_kv_indices, :] # [BLOCK_H, KERNEL_SIZE, head_dim]
    #   compressed_kv = sum(gate_scores[:, :, None] * kv_block, axis=1) # [BLOCK_H, head_dim]
    #   compressed_kv_cache[compressed_index, h_offsets, :] = compressed_kv
    for d_block_start in tl.range(0, head_dim, BLOCK_D):
        d_block_offsets = d_block_start + tl.arange(0, BLOCK_D)
        d_block_mask = d_block_offsets < head_dim

        kv_k_d_block = tl.load(
            kv_cache_ptr
            + h_offsets[:, None, None] * kv_buffer_stride_h
            + k_kv_indices[None, :, None] * kv_buffer_stride_s
            + d_block_offsets[None, None, :] * kv_buffer_stride_d,
            mask=h_mask[:, None, None] & d_block_mask[None, None, :],
        )  # [BLOCK_H, KERNEL_SIZE, BLOCK_D]

        d_block_compressed_kv = tl.sum(kv_k_d_block * gate_scores[:, :, None], axis=1)  # [BLOCK_H, BLOCK_D]

        tl.store(
            compressed_kv_cache_ptr
            + compressed_kv_cache_index * compressed_kv_buffer_stride_s
            + h_offsets[:, None] * compressed_kv_buffer_stride_h
            + d_block_offsets[None, :] * compressed_kv_buffer_stride_d,
            d_block_compressed_kv.to(compressed_kv_cache_ptr.dtype.element_ty),
            mask=h_mask[:, None] & d_block_mask[None, :],
        )


def gate_compress_decode_triton(
    kv_cache: torch.Tensor,  # [kv_size, num_heads, head_dim]
    compressed_kv_cache: torch.Tensor,  # [compressed_kv_size, num_heads, head_dim]
    gate_weight: torch.Tensor,  # [kernel_size, kernel_size * head_dim]
    kv_lens: torch.Tensor,  # [batch_size]
    req_pool_indices: torch.Tensor,  # [batch_size]
    req_to_token: torch.Tensor,  # [pool_size, max_kv_len]
    kernel_size: int,
    stride: int,
    cmp_rescale_factor: float = 0,
):
    """
    Equvalent to gate_compress_decode_torch
    When compression is triggered at current position, fetch kv_cache to compress and fill compressed kv into compressed_kv_cache
    kernel_size: compress block size
    stride: compress block stride
    """
    (batch_size,) = kv_lens.shape
    _, num_heads, head_dim = kv_cache.shape

    # TODO: resolve this assert
    assert kernel_size == triton.next_power_of_2(kernel_size)
    assert kernel_size >= 16

    BLOCK_H = 16
    BLOCK_K = 1
    BLOCK_D = max(16, min(128, triton.next_power_of_2(head_dim)))

    num_head_blocks = triton.cdiv(num_heads, BLOCK_H)

    grid = (
        batch_size,
        num_head_blocks,
    )

    _gate_compress_decode_kernel[grid](
        kv_cache_ptr=kv_cache,
        compressed_kv_cache_ptr=compressed_kv_cache,
        gate_weight_ptr=gate_weight,
        kv_lens_ptr=kv_lens,
        req_pool_indices_ptr=req_pool_indices,
        req_to_token_ptr=req_to_token,
        kv_buffer_stride_s=kv_cache.stride(0),
        kv_buffer_stride_h=kv_cache.stride(1),
        kv_buffer_stride_d=kv_cache.stride(2),
        compressed_kv_buffer_stride_s=compressed_kv_cache.stride(0),
        compressed_kv_buffer_stride_h=compressed_kv_cache.stride(1),
        compressed_kv_buffer_stride_d=compressed_kv_cache.stride(2),
        gate_weight_stride_o=gate_weight.stride(0),
        gate_weight_stride_i=gate_weight.stride(1),
        req_to_token_stride_b=req_to_token.stride(0),
        req_to_token_stride_t=req_to_token.stride(1),
        num_heads=num_heads,
        head_dim=head_dim,
        kernel_stride=stride,
        cmp_rescale_factor=cmp_rescale_factor,
        KERNEL_SIZE=kernel_size,
        BLOCK_H=BLOCK_H,
        BLOCK_K=BLOCK_K,
        BLOCK_D=BLOCK_D,
    )
