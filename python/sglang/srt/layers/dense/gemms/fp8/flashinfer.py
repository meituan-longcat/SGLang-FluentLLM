from typing import List, Optional

import torch

from sglang.srt.layers.dense.gemms.fp8.fp8_kernel import sglang_per_token_group_quant_fp8
from sglang.srt.utils import is_sm100_supported

VLLM_AVAILABLE = False

from sglang.srt.layers.dense.gemms.fp8.fp8_kernel import (
    is_fp8_fnuz,
)
from sglang.srt.utils import (
    get_bool_env_var,
    get_device_capability,
    is_cuda,
    is_flashinfer_available,
    is_hip,
)

_is_hip = is_hip()
_is_cuda = is_cuda()
_is_fp8_fnuz = is_fp8_fnuz()

use_vllm_cutlass_w8a8_fp8_kernel = get_bool_env_var("USE_VLLM_CUTLASS_W8A8_FP8_KERNEL")
use_triton_w8a8_fp8_kernel = get_bool_env_var("USE_TRITON_W8A8_FP8_KERNEL")

# Input scaling factors are no longer optional in _scaled_mm starting
# from pytorch 2.5. Allocating a dummy tensor to pass as input_scale
TORCH_DEVICE_IDENTITY = None


def use_rowwise_torch_scaled_mm():
    _TORCH_VERSION = torch.__version__.split("+")[0]
    try:
        _TORCH_VERSION_TUPLE = tuple(map(int, _TORCH_VERSION.split(".")[:3]))
    except ValueError:
        _TORCH_VERSION_TUPLE = (0, 0, 0)
    if _is_hip:
        # The condition to determine if it is on a platform that supports
        # torch._scaled_mm rowwise feature.
        # The condition is determined once as the operations
        # are time consuming.
        return get_device_capability() >= (9, 4) and _TORCH_VERSION_TUPLE >= (2, 7, 0)
    return False


USE_ROWWISE_TORCH_SCALED_MM = use_rowwise_torch_scaled_mm()


ENABLE_FLASHINFER_GEMM = get_bool_env_var("SGLANG_ENABLE_FLASHINFER_GEMM") and is_sm100_supported() and is_flashinfer_available()
if ENABLE_FLASHINFER_GEMM:
    from flashinfer.gemm import gemm_fp8_nt_groupwise


def flashinfer_gemm_w8a8_block_fp8_linear(
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

    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[0]]

    q_input, x_scale = sglang_per_token_group_quant_fp8(input_2d, block_size[1], column_major_scales=True)
    # TRTLLM requires column-major scaling factors
    output = gemm_fp8_nt_groupwise(
        q_input,
        weight,
        x_scale,
        weight_scale,
        out_dtype=input_2d.dtype,
        backend="trtllm",
    )

    if bias is not None:
        output += bias

    return output.to(dtype=input_2d.dtype).view(*output_shape)
