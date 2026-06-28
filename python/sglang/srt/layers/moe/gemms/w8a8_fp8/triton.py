from functools import partial
from typing import Optional


from sglang.srt.layers.moe.layouts.w8a8_fp8 import Fp8MoEPerChannelQuantLayout
from sglang.srt.layers.moe.executors.triton_executor import TritonExecutor
from sglang.srt.layers.moe.gemms.triton_common import invoke_fused_moe_kernel
from sglang.srt.layers.moe.gemms.triton_config import try_get_optimal_moe_config
from sglang.srt.layers.moe.config import DispatcherType
from sglang.srt.layers.activation import SwigluArg
import triton.language as tl


class TritonGemmWrapper:
    def __init__(self, top_k, num_experts, layer, quant_config):
        num_local_experts, intermediate_size_x2, hidden_size = layer.w13_weight.shape
        intermediate_size = intermediate_size_x2 // 2

        assert num_experts == num_local_experts, f"{num_experts=} != {num_local_experts=}"
        filter_expert = True

        compute_type = tl.bfloat16
        apply_router_weight_on_input = False

        common = dict(
            compute_type=compute_type,
            use_fp8_w8a8=True,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=True,
            block_shape=None,
            filter_expert=filter_expert,
        )
        gemm = partial(
            invoke_fused_moe_kernel,
            **common
        )
        self.gate_up_gemm = partial(
            gemm,
            A_scale=None,
            B_scale=layer.w13_weight_scale,
            mul_routed_weight=apply_router_weight_on_input,
            top_k=top_k,
        )
        self.down_gemm = partial(
            gemm,
            A_scale=None,
            B_scale=layer.w2_weight_scale,
            mul_routed_weight=not apply_router_weight_on_input,
            top_k=1,
        )

        padding_size = 0
        block_shape = None
        self.get_config_func = partial(
            try_get_optimal_moe_config,
            (num_local_experts, intermediate_size * 2, hidden_size),
            (num_local_experts, hidden_size, intermediate_size - padding_size),
            top_k,
            "fp8_w8a8",
            block_shape=block_shape,
            return_down_config=True,
        )

    @staticmethod
    def get_layout_cls():
        return Fp8MoEPerChannelQuantLayout

    def get_executor(self, dispatcher_type, activation: str, swiglu_arg: Optional[SwigluArg] = None):
        if dispatcher_type == DispatcherType.TP:
            return TritonExecutor(self.gate_up_gemm, self.down_gemm, self.get_config_func, activation, swiglu_arg)
        else:
            raise RuntimeError("To implement")
