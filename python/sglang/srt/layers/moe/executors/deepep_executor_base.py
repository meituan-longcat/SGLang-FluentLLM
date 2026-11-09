import torch

from sglang.srt.layers.moe.config import EPConfig
from sglang.srt.layers.moe.dispatcher.deep_ep import DeepEPDispatcher
from sglang.srt.model_executor.forward_batch_info import ForwardMode


class DeepEPExecutorBase:
    """Base class for DeepEP-based MoE executors.

    Analogous to EPSExecutorBase: forward = dispatch + compute + combine.
    Subclasses implement forward_normal() and forward_low_latency() for
    different GEMM backends (e.g. deep_gemm masked/contiguous).
    """

    def __init__(
        self,
        ep_config: EPConfig,
        dispatch_use_fp8: bool = True,
    ):
        self.ep_config = ep_config
        self.deepep_dispatcher = DeepEPDispatcher(
            ep_config,
            dispatch_use_fp8=dispatch_use_fp8,
        )

    def forward(
        self,
        layer,
        hidden_states: torch.Tensor,
        topk_output,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
    ):
        assert hidden_states.dtype == torch.bfloat16, f"hidden_states.dtype: {hidden_states.dtype}"

        topk_idx = topk_output.topk_ids
        topk_weights = topk_output.topk_weights

        # determine deepep mode: low_latency (DECODE) or normal (EXTEND)
        deepep_mode = ForwardMode.EXTEND if max_num_tokens_per_gpu > self.ep_config.low_latency_max_num_tokens_per_gpu else ForwardMode.DECODE

        moe_origin_input = hidden_states

        # dispatch
        self.deepep_dispatcher.dispatch_a(hidden_states, topk_idx, topk_weights, deepep_mode)
        (
            recv_hidden_with_scales,
            recv_topk_idx,
            recv_topk_weights,
            _,  # reorder_topk_ids
            num_recv_tokens_per_expert_list,
            _,  # seg_indptr
            masked_m,
        ) = self.deepep_dispatcher.dispatch_b()
        if isinstance(recv_hidden_with_scales, tuple):
            recv_hidden, input_scales = recv_hidden_with_scales
        else:
            recv_hidden, input_scales = recv_hidden_with_scales, None

        # compute
        output = self.compute(
            layer,
            num_global_tokens,
            deepep_mode,
            recv_hidden,
            input_scales,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            masked_m,
        )

        # combine
        combine_weights = topk_weights.to(torch.float32)
        if deepep_mode == ForwardMode.EXTEND:
            combine_weights = (combine_weights, recv_topk_weights)
        self.deepep_dispatcher.combine_a(
            output,
            topk_idx.to(torch.int64),
            combine_weights,
            deepep_mode,
            moe_origin_input,
        )
        return self.deepep_dispatcher.combine_b()
