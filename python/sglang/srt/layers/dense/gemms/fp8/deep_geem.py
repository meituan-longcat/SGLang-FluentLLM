from typing import List, Optional

import torch
import triton

from sglang.srt.utils import supports_custom_op, is_npu

__is_npu__ = is_npu()
if not __is_npu__:
    import deep_gemm

from sglang.srt.layers.dense.gemms.fp8.triton import triton_w8a8_block_fp8_linear
from sglang.srt.layers.dense.gemms.fp8.fp8_kernel import sglang_per_token_group_quant_fp8

from sglang.srt.env import global_server_args_dict


def dense_deep_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: List[int],
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """This function performs matrix multiplication with block-wise quantization.

    It takes two input tensors `A` and `B` with scales `As` and `Bs`.
    The output is returned in the specified `output_dtype`.

    Args:
        A: The input tensor, e.g., activation.
        B: The input tensor, e.g., weight.
        As: The per-token-group quantization scale for `A`.
        Bs: The per-block quantization scale for `B`.
        block_size: The block size for per-block quantization. It should be 2-dim, e.g., [128, 128].
        output_dytpe: The dtype of the returned tensor.

    Returns:
        torch.Tensor: The result of matmul.
    """
    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]

    assert A.shape[-1] == B.shape[-1]
    assert A.shape[:-1] == As.shape[:-1] and A.is_contiguous()
    assert triton.cdiv(A.shape[-1], block_k) == As.shape[-1], f"{A.shape=} {block_k=} {As.shape=}"

    assert B.ndim == 2 and B.is_contiguous() and Bs.ndim == 2
    N, K = B.shape
    assert triton.cdiv(N, block_n) == Bs.shape[0]
    assert triton.cdiv(K, block_k) == Bs.shape[1]

    C_shape = A.shape[:-1] + (N,)
    C = A.new_empty(C_shape, dtype=output_dtype)

    # deepgemm only support bf16
    assert C.dtype == torch.bfloat16, f"C.dtype: {C.dtype} != torch.bfloat16"
    deep_gemm.gemm_fp8_fp8_bf16_nt((A, As), (B, Bs), C, not global_server_args_dict["disable_pdl"])
    return C


def deepgemm_w8a8_block_fp8_linear_with_fallback(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: List[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    output_dtype = None,
) -> torch.Tensor:
    if output_dtype is None:
        output_dtype = input.dtype

    dtype_supported = output_dtype == torch.bfloat16

    # TODO: https://github.com/sgl-project/sglang/pull/6890#issuecomment-2943395737
    shape_supported = weight.shape[0] % 64 == 0 and weight.shape[1] % 128 == 0

    if not (shape_supported and dtype_supported):
        # fall back to triton
        return triton_w8a8_block_fp8_linear(
            input, weight, block_size, weight_scale, input_scale, bias
        )

    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[0]]

    if input_scale is None:
        q_input, x_scale = sglang_per_token_group_quant_fp8(
            input_2d,
            block_size[1],
            column_major_scales=True,
            scale_tma_aligned=True,
            # scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            scale_ue8m0=False,
        )
    else:
        q_input, x_scale = input_2d, input_scale

    output = dense_deep_gemm(
        q_input, weight, x_scale, weight_scale, block_size, output_dtype=output_dtype
    )
    if bias is not None:
        output += bias
    return output.to(dtype=output_dtype).view(*output_shape)

