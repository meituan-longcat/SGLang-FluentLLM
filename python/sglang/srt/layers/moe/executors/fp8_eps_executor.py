from typing import Optional
from functools import partial

import torch

from sglang.srt.layers.moe.executors.eps_mixin import EPSMixin
from sglang.srt.layers.activation import SwigluArg
from sglang.srt.layers.moe.config import  EPConfig
from sglang.srt.layers.activation import add_bias_to_segments, swiglu

from eps.executor import silu

import flashinfer

from sglang.srt.env import global_server_args_dict


class FP8EPSExecutor(EPSMixin):
    def __init__(
        self,
        gemm,
        ep_config: EPConfig, 
        activation: str,
        swiglu_arg: Optional[SwigluArg] = None
    ) -> None:
        super().__init__(ep_config)

        self.gate_up_gemm = partial(gemm, use_pdl=not global_server_args_dict["disable_pdl"])
        self.down_gemm = partial(gemm, use_pdl=not global_server_args_dict["disable_pdl"])
        self.activation = activation
        self.swiglu_arg = swiglu_arg

    def compute(
        self,
        layer,
        input: torch.Tensor,
        exclusive_sum: torch.Tensor,
        num_tokens_hint: int,
    ) -> torch.Tensor:
        M, hidden_size = input.size()
        device = input.device
        inter_size_x2 = layer.w13_weight.shape[1]
        num_groups = layer.w13_weight.shape[0]
        max_shape_m = (M + 3) // 4 * 4
        max_shape_m_padded = (M + num_groups * 31) // 32 * 32
        gate_up_output = torch.empty(
            (M, inter_size_x2), device=device, dtype=torch.bfloat16
        )
        input_fp8 = torch.empty(
            (M, hidden_size), device=device, dtype=torch.float8_e4m3fn
        )
        input_scale = torch.empty((hidden_size // 128, max_shape_m_padded), dtype=torch.float32, device=device).permute(-1, -2)
        flashinfer.quantization.quant_1x128(
            input, input_fp8, input_scale, exclusive_sum, num_groups, max_shape_m, max_shape_m_padded, hidden_size
        )
        self.gate_up_gemm((input_fp8, input_scale), (layer.w13_weight, layer.w13_weight_scale_inv), gate_up_output, exclusive_sum)

        if getattr(layer, "w13_weight_bias", None) is not None:
            add_bias_to_segments(gate_up_output, layer.w13_weight_bias, exclusive_sum)

        if self.activation == "silu":
            down_input = silu(gate_up_output, exclusive_sum, num_tokens_hint)
        elif self.activation == "swiglu":
            down_input = swiglu(gate_up_output, self.swiglu_arg.alpha, self.swiglu_arg.limit)
        else:
            raise RuntimeError(f"Not supported {self.activation=}")

        output = torch.empty(
            (M, hidden_size), device=device, dtype=torch.bfloat16
        )
        down_input_fp8 = torch.empty(
            (M, inter_size_x2 // 2), device=device, dtype=torch.float8_e4m3fn
        )
        down_input_scale = torch.empty((inter_size_x2 // 2 // 128, max_shape_m_padded), dtype=torch.float32, device=device).permute(-1, -2)
        flashinfer.quantization.quant_1x128(
            down_input, down_input_fp8, down_input_scale, exclusive_sum, num_groups, max_shape_m, max_shape_m_padded, inter_size_x2 // 2
        )
        self.down_gemm((down_input_fp8, down_input_scale), (layer.w2_weight, layer.w2_weight_scale_inv), output, exclusive_sum)
        if getattr(layer, "w2_weight_bias", None) is not None:
            add_bias_to_segments(output, layer.w2_weight_bias, exclusive_sum)

        return output
