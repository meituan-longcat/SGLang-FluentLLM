import torch

from sglang.srt.layers.moe.dispatcher.fast_ep import FastEP
from sglang.srt.layers.moe.config import EPConfig


class EPSExecutorBase:
    def __init__(self, ep_config: EPConfig):
        self.ep_config = ep_config
        self.fast_ep = FastEP(ep_config)

    def forward(
        self,
        layer,
        hidden_states: torch.Tensor,
        topk_output,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
    ):
        assert hidden_states.dtype == torch.bfloat16, f"hidden_states.dtype: {hidden_states.dtype}"
        num_tokens = hidden_states.shape[0]

        assert max_num_tokens_per_gpu <= self.ep_config.low_latency_max_num_tokens_per_gpu, (
            f"{max_num_tokens_per_gpu=}, {self.ep_config.low_latency_max_num_tokens_per_gpu=}"
        )

        expert_x, exclusive_sum = self.fast_ep.dispatch(hidden_states, topk_output.topk_ids, num_global_tokens)

        num_tokens_hint = max(1, num_global_tokens * self.ep_config.top_k // self.ep_config.world_size)
        expert_y = self.compute(layer, expert_x, exclusive_sum, num_tokens_hint)

        result = self.fast_ep.combine(num_tokens, topk_output.topk_ids, topk_output.topk_weights, expert_y, hidden_states, num_global_tokens)
        return result
