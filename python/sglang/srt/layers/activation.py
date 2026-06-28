# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Fused operators for activation layers."""

from dataclasses import dataclass
import triton
import triton.language as tl
from sglang.srt.utils import get_colorful_logger
from typing import Optional

import torch

import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.utils import is_cuda_available, is_npu

if is_cuda_available():
    from flashinfer import gelu_and_mul, gelu_tanh_and_mul, silu_and_mul

if is_npu():
    import torch_npu
else:
    import flashinfer

from sglang.srt.custom_op import CustomOp
from sglang.srt.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.utils import set_weight_attrs,is_npu

logger = get_colorful_logger(__name__)

class SiluAndMul(CustomOp):
    def get_tma_aligned_scale(self, x):
        aligned_size = (x.shape[-2] + 3) // 4 * 4
        x_s = torch.empty(
            x.shape[:-2]
            + (x.shape[-1] // 128, aligned_size),
            device=x.device,
            dtype=torch.float32,
        ).permute(-1, -2)[: x.shape[-2], :]
        return x_s

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

    def forward_npu(self, x: torch.Tensor) -> torch.Tensor:
        return torch_npu.npu_swiglu(x, dim = -1)

    def forward_cuda(self, x: torch.Tensor, fp8_out: bool = False) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        if fp8_out:
            out = torch.empty(output_shape, dtype=torch.float8_e4m3fn, device=x.device)
            scale = self.get_tma_aligned_scale(out)
            out, scale = flashinfer.activation.silu_and_mul_fuse_block_quant(x, scale, out, enable_pdl=True)
            return out, scale
        else:
            out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
            silu_and_mul(x, out)
            return out


class GeluAndMul(CustomOp):
    def __init__(self, approximate="tanh"):
        super().__init__()
        self.approximate = approximate

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return F.gelu(x[..., :d], approximate=self.approximate) * x[..., d:]

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        if self.approximate == "tanh":
            gelu_tanh_and_mul(x, out)
        elif self.approximate == "none":
            gelu_and_mul(x, out)
        else:
            raise RuntimeError("GeluAndMul only support tanh or none")
        return out


class QuickGELU(CustomOp):
    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(zhyncs): Implement the CUDA kernel for QuickGELU in sgl-kernel
        return self.forward_native(x)


class ScaledActivation(nn.Module):
    """An activation function with post-scale parameters.

    This is used for some quantization methods like AWQ.
    """

    def __init__(
        self,
        act_module: nn.Module,
        intermediate_size: int,
        input_is_parallel: bool = True,
        params_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.act = act_module
        self.input_is_parallel = input_is_parallel
        if input_is_parallel:
            tp_size = get_tensor_model_parallel_world_size()
            intermediate_size_per_partition = divide(intermediate_size, tp_size)
        else:
            intermediate_size_per_partition = intermediate_size
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.scales = nn.Parameter(
            torch.empty(intermediate_size_per_partition, dtype=params_dtype)
        )
        set_weight_attrs(self.scales, {"weight_loader": self.weight_loader})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x) / self.scales

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if self.input_is_parallel:
            tp_rank = get_tensor_model_parallel_rank()
            shard_size = param_data.shape[0]
            start_idx = tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


_ACTIVATION_REGISTRY = {
    "gelu": nn.GELU(),
    "gelu_pytorch_tanh": nn.GELU(approximate="tanh"),
}


def get_act_fn(
    act_fn_name: str,
    quant_config: Optional[QuantizationConfig] = None,
    intermediate_size: Optional[int] = None,
    input_is_parallel: bool = True,
    params_dtype: Optional[torch.dtype] = None,
) -> nn.Module:
    """Get an activation function by name."""
    act_fn_name = act_fn_name.lower()
    if act_fn_name not in _ACTIVATION_REGISTRY:
        raise ValueError(f"Activation function {act_fn_name!r} is not supported.")

    act_fn = _ACTIVATION_REGISTRY[act_fn_name]
    if quant_config is not None and act_fn_name in quant_config.get_scaled_act_names():
        if intermediate_size is None:
            raise ValueError(
                "intermediate_size must be specified for scaled "
                "activation functions."
            )
        return ScaledActivation(
            act_fn, intermediate_size, input_is_parallel, params_dtype
        )
    return act_fn

@triton.jit
def clip(x, limit, clip_lower: tl.constexpr):
    res = tl.minimum(x, limit)
    if clip_lower:
        res = tl.maximum(-limit, res)
    return res


@triton.jit
def compute_swiglu(gelu, linear, scale, alpha, limit):
    gelu = gelu.to(tl.float32) * scale
    if limit is not None:
        gelu = clip(gelu, limit, clip_lower=False)
    linear = linear.to(tl.float32) * scale
    if limit is not None:
        linear = clip(linear, limit, clip_lower=True)

    s = gelu / (1 + tl.exp(-alpha * gelu))

    return tl.fma(s, linear, s)  # (s * (linear + 1))


@triton.jit(repr=lambda _: "_swiglu")
def swiglu_fn(input, alpha, limit, exclusive_sum, local_num_experts):
    begin = exclusive_sum[0]
    end = exclusive_sum[local_num_experts]
    input = input[begin: end]

    gelu, linear = tl.split(tl.reshape(input, (input.shape[0], input.shape[1] // 2, 2)))
    return compute_swiglu(gelu, linear, 1.0, alpha, limit)


def swiglu(x, alpha: float = 1.702, limit: float = 7.0):
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    # Clamp the input values
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    # Note we add an extra bias of 1 to the linear layer
    return out_glu * (x_linear + 1)


@triton.jit
def add_bias_to_segments_kernel(
    A_ptr,              # Pointer to tensor A (shape [M, N])
    bias_ptr,           # Pointer to bias (shape [E, N])
    exclusive_sum_ptr,  # Pointer to exclusive prefix sum (length E+1)
    M, N, E,            # Dimensions
    stride_A0,          # Stride of A along dim 0
    stride_A1,          # Stride of A along dim 1
    stride_bias0,       # Stride of bias along dim 0
    stride_bias1,       # Stride of bias along dim 1
    BLOCK_SIZE: tl.constexpr  # Block size for parallelization
):
    # Program ID for parallelization
    pid = tl.program_id(0)
    
    # Find which segment this block is responsible for
    # We'll use binary search to find the segment containing pid
    low = 0
    high = E
    while low < high:
        mid = (low + high) // 2
        mid_val = tl.load(exclusive_sum_ptr + mid)
        if mid_val <= pid:
            low = mid + 1
        else:
            high = mid
    segment_idx = low - 1
    
    # Get the start and end of this segment
    start = tl.load(exclusive_sum_ptr + segment_idx)
    end = tl.load(exclusive_sum_ptr + segment_idx + 1)
    
    # Only proceed if this block is within the segment bounds
    if pid < end:
        # Get the bias for this segment
        bias_offset = segment_idx * stride_bias0
        bias_row = bias_ptr + bias_offset
        
        # Process BLOCK_SIZE elements at a time
        for k in range(0, N, BLOCK_SIZE):
            # Create a mask for valid elements in this block
            col_offsets = k + tl.arange(0, BLOCK_SIZE)
            mask = col_offsets < N
            
            # Load bias values for this block
            bias_vals = tl.load(bias_row + col_offsets * stride_bias1, mask=mask)
            
            # Compute A's memory offsets
            a_row = A_ptr + pid * stride_A0
            a_col = col_offsets * stride_A1
            a_offsets = a_row + a_col
            
            # Load, add bias, and store back
            a_vals = tl.load(a_offsets, mask=mask)
            a_vals += bias_vals
            tl.store(a_offsets, a_vals, mask=mask)


def add_bias_to_segments(A, bias, exclusive_sum):
    """
    A: tensor of shape [M, N]
    bias: tensor of shape [E, N]
    exclusive_sum: tensor of length E+1
    """
    assert A.dim() == 2 and bias.dim() == 2
    M, N = A.shape
    E = bias.shape[0]
    assert exclusive_sum.shape[0] == E + 1
    
    # Configure block size (can be tuned for your hardware)
    BLOCK_SIZE = 128
    
    # Grid size is the number of rows in A
    grid = (M,)
    
    # Launch the kernel
    add_bias_to_segments_kernel[grid](
        A, bias, exclusive_sum,
        M, N, E,
        A.stride(0), A.stride(1),
        bias.stride(0), bias.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )


@dataclass
class SwigluArg:
    alpha: float
    limit: float
