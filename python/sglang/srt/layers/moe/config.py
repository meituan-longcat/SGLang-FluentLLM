from dataclasses import dataclass
from enum import Enum

import torch

from sglang.srt.distributed import get_moe_ep_group, get_moe_expert_parallel_rank, get_moe_expert_parallel_world_size
from sglang.srt.env import global_server_args_dict


class DispatcherType(Enum):
    EPS = "EPS"
    DEEPEP = "DEEPEP"
    TP = "TP"


@dataclass
class EPConfig:
    top_k: int
    num_experts: int
    low_latency_max_num_tokens_per_gpu: int
    max_num_tokens_per_gpu: int
    hidden_size: int
    rank: int
    world_size: int
    group: torch.distributed.ProcessGroup
    params_dtype: torch.dtype

    @staticmethod
    def from_gemm_wrapper(gemm_wrapper):
        chunked_prefill_size = global_server_args_dict["chunked_prefill_size"]
        max_num_tokens_per_gpu = chunked_prefill_size // global_server_args_dict["attn_tp_size"]
        low_latency_max_num_tokens_per_gpu = global_server_args_dict["low_latency_max_num_tokens_per_gpu"]

        ep_config = EPConfig(
            top_k=gemm_wrapper.top_k,
            num_experts=gemm_wrapper.num_experts,
            low_latency_max_num_tokens_per_gpu=low_latency_max_num_tokens_per_gpu,
            max_num_tokens_per_gpu=max_num_tokens_per_gpu,
            hidden_size=gemm_wrapper.hidden_size,
            rank=get_moe_expert_parallel_rank(),
            world_size=get_moe_expert_parallel_world_size(),
            group=get_moe_ep_group().device_group,
            params_dtype=torch.bfloat16,
        )
        return ep_config
