from typing import Callable, Any, Dict, List, Optional, Union

import torch
from torch.nn import Parameter
import torch_npu


from sglang.srt.layers.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from sglang.srt.layers.quantization.compressed_tensors import FusedWeightScaleSupported
from sglang.srt.layers.quantization.compressed_tensors.schemes import CompressedTensorsScheme

BEFORE_INIT = 0
AFTER_INIT = 1
WEIGHT_BITS = 8
ACTIVATION_BITS = 8

ASCEND_ALIGN_BYTES = 512
INT_8_BYTES = 1





# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Optional

import torch
from torch.nn import Parameter



class CompressedTensorsW8A8Int8(CompressedTensorsScheme):

    def __init__(
        self, strategy: str, is_static_input_scheme: bool, input_symmetric: bool
    ):
        self.strategy = FusedWeightScaleSupported(strategy)
        self.is_static_input_scheme = is_static_input_scheme
        self.input_symmetric = input_symmetric

    @classmethod
    def get_min_capability(cls) -> int:
        # lovelace and up
        return 89

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight
        weight_scale = layer.weight_scale

        if getattr(layer, 'throw_dequant', False):
            weight_scale = weight_scale.to(torch.float32)
        weight_offset = layer.weight_offset
        weight = torch_npu.npu_format_cast(weight.t().contiguous(), 29)
        layer.weight = Parameter(weight, requires_grad=False)

        layer.weight_scale = Parameter(weight_scale.view(-1), requires_grad=False)
        layer.weight_offset = Parameter(weight_offset.view(-1).float(), requires_grad=False)
        return

    def create_weights(self, layer: torch.nn.Module, output_partition_sizes: List[int],
                    input_size_per_partition: int, params_dtype: torch.dtype, weight_loader: Callable,
                    **kwargs):
        self.logical_widths = output_partition_sizes

        weight = ModelWeightParameter(data=torch.empty(sum(output_partition_sizes),
                                                    input_size_per_partition, dtype=torch.int8),
                                    input_dim=1, output_dim=0, weight_loader=weight_loader)

        layer.register_parameter("weight", weight)

        if self.strategy == FusedWeightScaleSupported.TENSOR:
            weight_scale = PerTensorScaleParameter(data=torch.empty(
                len(output_partition_sizes), dtype=torch.float32 if params_dtype == torch.float16 else torch.bfloat16),
                weight_loader=weight_loader)
            weight_offset = None
        else:
            weight_scale = ChannelQuantScaleParameter(
                data=torch.empty((sum(output_partition_sizes), 1),
                                dtype=torch.float32 if params_dtype == torch.float16 else torch.bfloat16),
                output_dim=0,
                weight_loader=weight_loader)
            weight_offset = ChannelQuantScaleParameter(
                data=torch.zeros((sum(output_partition_sizes), 1),
                                dtype=torch.float32 if params_dtype == torch.float16 else torch.bfloat16),
                output_dim=0,
                weight_loader=weight_loader)

        layer.register_parameter("weight_scale", weight_scale)
        layer.register_parameter("weight_offset", weight_offset)

        setattr(layer, "init_state", BEFORE_INIT)

        self.empty_out = torch.empty(1, dtype=params_dtype)
        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig
        CompressedTensorsConfig.weight_bits = WEIGHT_BITS
        CompressedTensorsConfig.activation_bits = ACTIVATION_BITS

    def apply_weights(self, layer: torch.nn.Module,
                    x: torch.Tensor,
                    bias: Optional[torch.Tensor],
                    hcom: Optional[str] = None,
                    use_mm_all_reduce_op: bool = False,
                    inner_gather: bool = False,
                    enable_weight_transpsoe: bool = False
                    ) -> Union[torch.Tensor, Dict[str, Any]]:

        # activation per-token dynamic quant
        if isinstance(x, (list, tuple)):
            x_int8, pertoken_scale = x
        else:
            smooth_scales = getattr(layer, 'smooth_scale', None)
            smooth_scales = smooth_scales.to(torch.bfloat16) if smooth_scales is not None else None
            x_int8, pertoken_scale = torch_npu.npu_dynamic_quant(x, smooth_scales=smooth_scales)

        throw_dequant = getattr(layer, 'throw_dequant', False)
        if x_int8.size(0) == 0:
            if throw_dequant:
                return torch.zeros([0, layer.weight.size(1)], dtype=torch.int32, device=x_int8.device), pertoken_scale
            else:
                return torch.zeros([0, layer.weight.size(1)], dtype=torch.bfloat16, device=x_int8.device)

        if use_mm_all_reduce_op:
            assert not throw_dequant, "unsupport throw dequant when using mm all reduce op"
            if pertoken_scale is not None:
                pertoken_scale = pertoken_scale.view(-1)
            out = torch_npu.npu_mm_all_reduce_base(x_int8, layer.weight, hcom, reduce_op='sum',
                                                dequant_scale=layer.weight_scale, pertoken_scale=pertoken_scale)
        else:
            if throw_dequant:
                assert bias is None
                return (torch_npu.npu_quant_matmul(x_int8, layer.weight, layer.weight_scale,
                                                   bias=None, output_dtype=torch.int32),
                        pertoken_scale)
            out = torch_npu.npu_quant_matmul(x_int8, layer.weight, layer.weight_scale,
                                            offset=None,
                                            pertoken_scale=pertoken_scale,
                                            bias=bias,
                                            output_dtype=torch.bfloat16)
        return out
