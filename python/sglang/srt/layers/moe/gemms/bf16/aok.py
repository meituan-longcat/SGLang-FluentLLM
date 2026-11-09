from typing import Optional

from eps.executor import AokGroupedGemm

from sglang.srt.layers.moe.config import DispatcherType, EPConfig
from sglang.srt.ops.activation import SwigluArg
from sglang.srt.layers.moe.executors.bf16_deepep_executor import BF16DeepEPExecutor
from sglang.srt.layers.moe.executors.bf16_eps_executor import BF16EPSExecutor
from sglang.srt.layers.moe.layouts.unquant import UnquantMoELayout


class AokGroupedGemmWrapper:
    def __init__(self, top_k, num_experts, layer, quant_config):
        self.top_k = top_k
        self.num_experts = num_experts
        num_local_experts, _, hidden_size = layer.w13_weight.shape
        self.hidden_size = hidden_size

        self.gemm = AokGroupedGemm(num_local_experts)

    @staticmethod
    def get_layout_cls():
        return UnquantMoELayout

    def get_executor(
        self,
        dispatcher_type: DispatcherType,
        activation: str,
        swiglu_arg: Optional[SwigluArg] = None,
    ):
        ep_config = EPConfig.from_gemm_wrapper(self)
        if dispatcher_type == DispatcherType.EPS:
            return BF16EPSExecutor(self.gemm, ep_config, activation, swiglu_arg)
        if dispatcher_type == DispatcherType.DEEPEP:
            return BF16DeepEPExecutor(
                ep_config,
                activation,
                swiglu_arg,
            )
        raise RuntimeError("To implement")
