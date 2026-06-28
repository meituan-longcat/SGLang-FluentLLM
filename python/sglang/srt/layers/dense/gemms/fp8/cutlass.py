from typing import List, Optional

import torch

from sglang.srt.utils import get_bool_env_var, is_npu

__is_npu__ = is_npu()
if not __is_npu__:
    from flashinfer import fp8_blockwise_scaled_mm

# use_vllm_cutlass_w8a8_fp8_kernel = get_bool_env_var("USE_VLLM_CUTLASS_W8A8_FP8_KERNEL")
# use_triton_w8a8_fp8_kernel = get_bool_env_var("USE_TRITON_W8A8_FP8_KERNEL")

# Input scaling factors are no longer optional in _scaled_mm starting
# from pytorch 2.5. Allocating a dummy tensor to pass as input_scale
TORCH_DEVICE_IDENTITY = None


def cutlass_block_fp8_supported() -> bool:
    if not get_bool_env_var("SGLANG_SUPPORT_CUTLASS_BLOCK_FP8"):
        return False
    major, minor = torch.cuda.get_device_capability()
    sm_version = major * 10 + minor
    cuda_version = tuple(map(int, torch.version.cuda.split(".")))
    if cuda_version >= (12, 0) and sm_version >= 90:
        return True
    return False


CUTLASS_BLOCK_FP8_SUPPORTED = cutlass_block_fp8_supported()


def cutlass_w8a8_block_fp8_linear_with_fallback(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: List[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert input_scale is None

    # TODO: add more robust shape check here
    shape_supported = weight.shape[0] % 128 == 0 and weight.shape[1] % 128 == 0

    if not shape_supported:
        # fallback to triton
        return triton_w8a8_block_fp8_linear(
            input, weight, block_size, weight_scale, input_scale, bias
        )

    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[0]]

    q_input, x_scale = per_token_group_quant_fp8(
        input_2d, block_size[1], column_major_scales=True
    )
    output = fp8_blockwise_scaled_mm(
        q_input, weight.T, x_scale, weight_scale.T, out_dtype=input_2d.dtype
    )
    if bias is not None:
        output += bias
    return output.to(dtype=input_2d.dtype).view(*output_shape)
