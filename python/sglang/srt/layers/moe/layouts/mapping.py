from typing import List, Tuple, Optional

from sglang.srt.distributed import get_moe_expert_parallel_rank, get_moe_expert_parallel_world_size
from sglang.srt.managers.eplb.expert_location import get_global_expert_location_metadata


def make_expert_params_mapping(
    ckpt_gate_proj_name: str,
    ckpt_down_proj_name: str,
    ckpt_up_proj_name: str,
    num_experts: int,
) -> List[Tuple[str, str, int, str]]:
    mappings = []

    physical_to_logical_map = get_global_expert_location_metadata().physical_to_logical_map_cpu

    num_local_experts = num_experts // get_moe_expert_parallel_world_size()
    start_expert = num_local_experts * get_moe_expert_parallel_rank()

    for local_expert_id in range(num_local_experts):
        expert_id = physical_to_logical_map[0][start_expert + local_expert_id]  # FIXME

        shard_id = "w1"
        param_name = "experts.w13_"
        weight_name = f"experts.{expert_id}.{ckpt_gate_proj_name}."
        mappings.append((param_name, weight_name, local_expert_id, shard_id))

        shard_id = "w3"
        param_name = "experts.w13_"
        weight_name = f"experts.{expert_id}.{ckpt_up_proj_name}."
        mappings.append((param_name, weight_name, local_expert_id, shard_id))

        shard_id = "w2"
        param_name = "experts.w2_"
        weight_name = f"experts.{expert_id}.{ckpt_down_proj_name}."
        mappings.append((param_name, weight_name, local_expert_id, shard_id))

    return mappings


def make_expert_params_mapping_fused(
    ckpt_gate_up_proj_name: str,
    ckpt_down_proj_name: str,
    ckpt_gate_up_proj_bias_name: str,
    ckpt_down_proj_bias_name: str,
):
    return [
        ("experts.w13_weight", f"experts.{ckpt_gate_up_proj_name}", "w13"),
        (
            "experts.w13_weight_bias",
            f"experts.{ckpt_gate_up_proj_bias_name}",
            "w13",
        ),
        ("experts.w2_weight", f"experts.{ckpt_down_proj_name}", "w2"),
        ("experts.w2_weight_bias", f"experts.{ckpt_down_proj_bias_name}", "w2"),
    ]


def make_expert_input_scale_params_mapping(
    num_experts: int,
) -> List[Tuple[str, str, int, str]]:
    # (param_name, weight_name, expert_id, shard_id)
    return [
        (
            "experts.w13_" if shard_id in ["w1", "w3"] else "experts.w2_",
            f"experts.{expert_id}.{shard_id}.",
            expert_id,
            shard_id,
        )
        for expert_id in range(num_experts)
        for shard_id in ["w1", "w2", "w3"]
    ]


# Naive mapping, will adapt for EPLB later
def map_global_expert_id_to_local_expert_id(expert_id: int, expert_map_cpu) -> int:
    expert_id
