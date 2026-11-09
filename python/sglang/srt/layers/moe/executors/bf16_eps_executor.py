from typing import Optional

import torch

from sglang.srt.layers.moe.executors.eps_executor_base import EPSExecutorBase
from sglang.srt.layers.moe.config import EPConfig
from sglang.srt.ops.activation import SwigluArg, add_bias_to_segments, swiglu

from eps.executor import silu


class BF16EPSExecutor(EPSExecutorBase):
    def __init__(self, gemm, ep_config: EPConfig, activation: str, swiglu_arg: Optional[SwigluArg] = None) -> None:
        super().__init__(ep_config)

        self.gate_up_gemm = gemm
        self.down_gemm = gemm
        self.activation = activation
        self.swiglu_arg = swiglu_arg

    def compute(
        self,
        layer,
        input: torch.Tensor,
        exclusive_sum: torch.Tensor,
        num_tokens_hint: int,
    ) -> torch.Tensor:
        gate_up_output = self.gate_up_gemm.run(input, layer.w13_weight, None, exclusive_sum, num_tokens_hint)
        if getattr(layer, "w13_weight_bias", None) is not None:
            add_bias_to_segments(gate_up_output, layer.w13_weight_bias, exclusive_sum)

        if self.activation == "silu":
            down_input = silu(gate_up_output, exclusive_sum, num_tokens_hint)
        elif self.activation == "swiglu":
            down_input = swiglu(gate_up_output, self.swiglu_arg.alpha, self.swiglu_arg.limit)
        else:
            raise RuntimeError(f"Not supported {self.activation=}")

        output = self.down_gemm.run(down_input, layer.w2_weight, None, exclusive_sum, num_tokens_hint)

        if getattr(layer, "w2_weight_bias", None) is not None:
            add_bias_to_segments(output, layer.w2_weight_bias, exclusive_sum)

        return output
