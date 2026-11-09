from __future__ import annotations

from typing import List, Optional

import torch
from torch.nn.parameter import Parameter

from sglang.srt.layers.parameter import ChannelQuantScaleParameter, ModelWeightParameter
from sglang.srt.layers.quantization.base_config import (
    LinearMethodBase,
)
from sglang.srt.layers.dense.gemms.fp8.fp8_kernel import (
    fp8_dtype,
    is_fp8_fnuz,
    per_token_group_quant_fp8,
)
from sglang.srt.layers.dense.gemms.fp8.fp8_utils import (
    apply_fp8_linear,
    cutlass_fp8_supported,
    input_to_float8,
    normalize_e4m3fn_to_e4m3fnuz,
)


_is_fp8_fnuz = is_fp8_fnuz()


class W8A8Fp8LinearMethod(LinearMethodBase):
    def __init__(self, quantization_config: W8A8Fp8Config):
        self.quantization_config = quantization_config
        self.cutlass_fp8_supported = cutlass_fp8_supported()

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight

        if self.quantization_config.is_checkpoint_fp8_serialized:
            weight_scale = layer.weight_scale.detach()
            # If checkpoint offline quantized with w8a8_fp8, load the weight and weight_scale directly.
            if _is_fp8_fnuz:
                weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(weight=weight, weight_scale=weight_scale)

            layer.weight = Parameter(weight.t(), requires_grad=False)
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)
        else:
            # If checkpoint not offline quantized, quantize the weights with per-channel quantization.
            if self.cutlass_fp8_supported:
                # if cutlass supported, we use cutlass_scaled_mm
                # which requires per-channel quantization on weight
                qweight, weight_scale = per_token_group_quant_fp8(layer.weight, layer.weight.shape[-1])
                weight_scale = weight_scale.t().contiguous()
            else:
                # if cutlass not supported, we fall back to use torch._scaled_mm
                # which requires per tensor quantization on weight
                qweight, weight_scale = input_to_float8(layer.weight, dtype=fp8_dtype)

            # Update the layer with the new values.
            layer.weight = Parameter(qweight.t(), requires_grad=False)
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)
            layer.input_scale = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        weight_dtype = torch.float8_e4m3fn if self.quantization_config.is_checkpoint_fp8_serialized else params_dtype

        weight_loader = extra_weight_attrs.get("weight_loader")
        self.logical_widths = output_partition_sizes

        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=weight_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        if self.quantization_config.is_checkpoint_fp8_serialized:
            weight_scale = ChannelQuantScaleParameter(
                data=torch.empty((sum(output_partition_sizes), 1), dtype=torch.float32),
                output_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight_scale", weight_scale)
        else:
            layer.weight_scale = None

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        return apply_fp8_linear(
            x,
            layer.weight,
            layer.weight_scale,
            bias=bias,
            cutlass_fp8_supported=self.cutlass_fp8_supported,
        )
