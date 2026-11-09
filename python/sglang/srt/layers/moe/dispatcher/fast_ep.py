import torch
from eps.fast_ep import AllToAll

from sglang.srt.distributed.parallel_state import get_eps_communicator
from sglang.srt.layers.moe.config import EPConfig

_FAST_EP = None


class FastEP:
    def __init__(self, config: EPConfig):
        self.config = config

    def construct_fast_ep(self):
        comm = get_eps_communicator()
        _FAST_EP = AllToAll(
            self.config.top_k,
            self.config.num_experts,
            self.config.hidden_size,
            self.config.low_latency_max_num_tokens_per_gpu * self.config.world_size,
            comm.data_ptr(),
        )
        return _FAST_EP

    def dispatch(self, dp_x: torch.Tensor, indices: torch.Tensor, num_global_tokens: int):
        global _FAST_EP
        if _FAST_EP is None:
            _FAST_EP = self.construct_fast_ep()

        assert indices.dtype == torch.int32, "indices dtype should be torch.int32"

        num_local_experts = self.config.num_experts // self.config.world_size
        exclusive_sum = torch.empty(
            num_local_experts + 1,
            dtype=torch.int32,
            device=dp_x.device,
        )
        expert_x = torch.empty(
            (num_global_tokens * self.config.top_k, self.config.hidden_size),
            dtype=dp_x.dtype,
            device=dp_x.device,
        )
        _FAST_EP.dispatch(
            out_exclusive_sum=exclusive_sum,
            out_expert_x=expert_x,
            dp_x=dp_x,
            indices=indices,
            num_global_tokens=num_global_tokens,
        )
        return expert_x, exclusive_sum

    def combine(
        self,
        num_tokens,
        indices: torch.Tensor,
        weights: torch.Tensor,
        expert_y: torch.Tensor,
        dp_x: torch.Tensor,
        num_global_tokens: int,
    ):
        assert indices.dtype == torch.int32, f"indices dtype: {indices.dtype} != torch.int32"

        # Combine
        y = torch.empty(
            (num_tokens, self.config.hidden_size),
            dtype=expert_y.dtype,
            device=expert_y.device,
        )

        _FAST_EP.combine(
            out_tokens=y,
            weights=weights,
            expert_y=expert_y,
            dp_x=dp_x,
            num_global_tokens=num_global_tokens,
        )

        return y
