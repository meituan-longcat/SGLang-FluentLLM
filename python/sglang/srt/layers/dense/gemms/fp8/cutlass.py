from typing import List, Optional

import math
import torch

from sglang.srt.utils import is_npu

from sglang.srt.layers.dense.gemms.fp8.triton import triton_w8a8_block_fp8_linear

__is_npu__ = is_npu()
if not __is_npu__:
    import flashinfer
    from flashinfer.sgl_fp8_quantization import sgl_fp8_quantize_1x128

# use_vllm_cutlass_w8a8_fp8_kernel = get_bool_env_var("USE_VLLM_CUTLASS_W8A8_FP8_KERNEL")
# use_triton_w8a8_fp8_kernel = get_bool_env_var("USE_TRITON_W8A8_FP8_KERNEL")

# Input scaling factors are no longer optional in _scaled_mm starting
# from pytorch 2.5. Allocating a dummy tensor to pass as input_scale
TORCH_DEVICE_IDENTITY = None


def cutlass_w8a8_block_fp8_linear_with_fallback(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: List[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    *args,
    **kwargs,
) -> torch.Tensor:
    assert input_scale is None

    # TODO: add more robust shape check here
    shape_supported = weight.shape[0] % 128 == 0 and weight.shape[1] % 128 == 0

    if not shape_supported:
        # fallback to triton
        return triton_w8a8_block_fp8_linear(input, weight, block_size, weight_scale, input_scale, bias)

    input_2d = input.view(-1, input.shape[-1])
    M = input_2d.shape[0]
    K = input_2d.shape[1]
    m_padded = math.ceil(M / 4) * 4
    k_blocks = K // 128
    output_shape = [M, weight.shape[0]]

    gemm_output = torch.empty(M, weight.shape[0], dtype=torch.bfloat16, device="cuda")
    mat_a = torch.empty(m_padded, K, dtype=torch.float8_e4m3fn, device="cuda")
    scales = torch.empty(k_blocks, m_padded, dtype=torch.float32, device="cuda")
    # Reuqires [k_blocks, M] shape with row-major scale, and the gemm will treat it as [M, k_blocks] col-major
    # and mat_a is [M, K] with row-major, weight should be [N, K] with row-major
    sgl_fp8_quantize_1x128(input_2d, mat_a, scales, use_ue8m0=False)
    flashinfer.fp8_blockwise_scaled_mm_sm89(gemm_output, mat_a[:M], weight, scales, weight_scale)
    if bias is not None:
        gemm_output += bias
    return gemm_output.to(dtype=input_2d.dtype).view(*output_shape)
