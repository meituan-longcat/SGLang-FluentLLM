"""Fused MoE kernel."""
from typing import Any, Dict, Optional, Tuple, Callable

import torch
import torch_npu
import torch.distributed as dist
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

from sglang.srt.utils import CustomFormatter
import logging
def get_colorful_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(CustomFormatter())

    logger.addHandler(ch)
    return logger


logger = get_colorful_logger(__name__)

_MAX_NUM_TOKEN=100000

def fused_topk(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    topk_weights, topk_ids, row_idx = torch_npu.npu_moe_gating_top_k_softmax(gating_output, k=topk)

    if renormalize:
        topk_weights /= topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids, row_idx


# This is used by the Deepseek-V2 and Deepseek-V3 model
def grouped_topk(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    e_score_correction_bias: Optional[torch.Tensor] = None
):
    gating_output = gating_output.float()
    # scores = torch.softmax(gating_output, dim=-1)
    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    if e_score_correction_bias is not None:
        scores = scores + e_score_correction_bias.unsqueeze(0)
    num_token = scores.shape[0]
    group_scores = scores.view(num_token, num_expert_group,
                               -1).max(dim=-1).values  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1,
                           sorted=False)[1]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = group_mask.unsqueeze(-1).expand(
        num_token, num_expert_group,
        scores.shape[-1] // num_expert_group).reshape(num_token, -1)  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
    topk_weights, topk_ids = torch.topk(tmp_scores,
                                        k=topk,
                                        dim=-1,
                                        sorted=False)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    topk_ids = topk_ids.int()
    # adapt add row_idx
    row_idx = torch.arange(topk_ids.numel(), device=topk_ids.device, dtype=topk_ids.dtype)
    row_idx = row_idx.reshape(topk_ids.shape[1], topk_ids.shape[0]).transpose(1, 0).contiguous()
    # adapt end

    return topk_weights, topk_ids, row_idx

def fused_experts_allgather_ep(hidden_states: torch.Tensor,
                               w13: torch.Tensor,
                               w2: torch.Tensor,
                               topk_weights: torch.Tensor,
                               topk_ids: torch.Tensor,
                               row_idx: torch.Tensor,
                               n_routed_experts: int,
                               local_expert_indices: list,
                               best_expert_tokens: Optional[torch.Tensor] = None
                               ):
    expert_parallel_size = get_tensor_model_parallel_world_size()

    # if expert_parallel_size >= 1:
    num_tokens, hidden_size = hidden_states.shape
    global_local_mask = (topk_ids >= local_expert_indices[0]) & \
                        (topk_ids <= local_expert_indices[-1])
    non_global_local_mask = (~global_local_mask).to(torch.int32)
    global_local_mask = global_local_mask.to(torch.int32)

    topk_ids -= get_tensor_model_parallel_rank() * n_routed_experts

    local_topk_ids_mask_with_max = topk_ids * global_local_mask + non_global_local_mask * n_routed_experts

    # sorted_tokens[expanded_src_to_dst_row] = hidden_states.unsqueeze(0).repeat(6,1,1).reshape(42,-1)
    # expanded_expert_idx[expanded_src_to_dst_row] = local_topk_ids_mask_with_max.transpose(0,1).reshape(-1)
    # sorted_tokens[expanded_src_to_dst_row].reshape(6,7,-1).transpose(0,1) = [0,0,0,0,0,0,0,1,1,1,1...]
    sorted_tokens, expanded_src_to_dst_row, expanded_expert_idx = \
        torch_npu.npu_moe_init_routing(hidden_states, row_idx, local_topk_ids_mask_with_max, _MAX_NUM_TOKEN)

    if expanded_expert_idx.shape[0] > 8192:
        expert_tokens = torch_npu.npu_moe_compute_expert_tokens(expanded_expert_idx, n_routed_experts + 1)
        expert_tokens = expert_tokens[:-1]
    else:
        expert_tokens = torch_npu.npu_moe_compute_expert_tokens(expanded_expert_idx, n_routed_experts)

    if best_expert_tokens is not None:
        expert_tokens = best_expert_tokens

    expert_tokens = expert_tokens.to(torch.int64)
    gate_up_proj_tmp = torch_npu.npu_grouped_matmul(
        [sorted_tokens], [w13], group_list=expert_tokens, group_type=0, group_list_type=0, split_item=3)[0]
    gate_up_proj = torch_npu.npu_swiglu(gate_up_proj_tmp)
    out = torch_npu.npu_grouped_matmul(
        [gate_up_proj], [w2], group_list=expert_tokens, group_type=0, group_list_type=0, split_item=3)[0]
    if out.shape[0] < 12288:
        out[expanded_expert_idx == n_routed_experts] = 0
    else:
        out[expert_tokens[-1]:] = 0

    topk_weights = topk_weights.to(out.dtype)
    output = torch_npu.npu_moe_finalize_routing(out, None, None, None, topk_weights,
                                                expanded_src_to_dst_row, topk_ids)
    # out[expanded_src_to_dst_row].reshape(6,7,-1).transpose(0,1)[0].mul(topk_weights[0].view(-1,1)).sum(0)
    return output

def fused_experts_w8a8_allgather_ep(hidden_states: torch.Tensor,
                                    pertoken_scale: torch.Tensor,
                                    w1: torch.Tensor,
                                    w2: torch.Tensor,
                                    w1_scale: torch.Tensor,
                                    w2_scale: torch.Tensor,
                                    topk_weights: torch.Tensor,
                                    topk_ids: torch.Tensor,
                                    n_routed_experts: int,
                                    max_num_deployed_expert_per_rank:int #ENABLE_OMNI_PLANNER
                                    ):
    batch_size, hidden_size = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_size)
    n_total_expert = n_routed_experts * get_tensor_model_parallel_world_size()

    experts_start_idx = get_tensor_model_parallel_rank() * max_num_deployed_expert_per_rank  #ENABLE_OMNI_PLANNER
    experts_end_idx = experts_start_idx + n_routed_experts
    expert_range = [experts_start_idx, experts_end_idx]

    # sorted_tokens = hidden_states.unsqueeze(1).repeat(1, 6, 1).reshape(42,-1)[expanded_x_idx]
    # dynamic_quant_scale = pertoken_scale.unsqueeze(1).repeat(1,6).reshape(42)[expanded_x_idx]
    # [torch.argsort(expanded_x_idx)]
    sorted_tokens, expanded_x_idx, expert_tokens, dynamic_quant_scale = torch_npu.npu_moe_init_routing_v2(
        hidden_states, topk_ids, scale=pertoken_scale, offset=None, active_num=topk_ids.numel(), expert_capacity=-1, expert_num=n_total_expert, drop_pad_mode=0, expert_tokens_num_type=1, expert_tokens_num_flag=True, quant_mode=-1,active_expert_range=expert_range, row_idx_type=1)

    sorted_topk_weight = torch.index_select(topk_weights.reshape(-1), 0, expanded_x_idx)
    row_index = expanded_x_idx // topk_ids.shape[-1]
    row_index = row_index.to(torch.int64)
    share_input = torch.zeros((batch_size // get_tensor_model_parallel_world_size(), hidden_size), dtype=torch.bfloat16,
                              device="npu")
    scale_2 = torch.ones((n_routed_experts, w1_scale.shape[-1] // 2), dtype=torch.float32, device="npu")

    gate_up_proj = torch_npu.npu_grouped_matmul([sorted_tokens], [w1], bias=None, group_list=expert_tokens,
                                                split_item=3, output_dtype=torch.int32, group_type=0,
                                                group_list_type=1)[0]

    gate_up_proj, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(gate_up_proj, weight_scale=w1_scale, activation_scale=dynamic_quant_scale, bias=None, quant_scale=scale_2, quant_offset=None, group_index=expert_tokens, activate_left=True, quant_mode=1)
    # v2 lite's moe_intermediate_size = 1408, hidden_size=2048
    # v3's moe_intermediate_size = 2048, hidden_size=7168
    # RuntimeError: call aclnnGroupedMatmulFinalizeRoutingWeightNz failed
    # weight KDim is 2048 but is 1408, and NDim is 7168 but is 2048.
    if hidden_size == 7168:
        # for v3 w8a8
        output = torch_npu.npu_grouped_matmul_finalize_routing(gate_up_proj, w2, expert_tokens, scale=w2_scale, bias=None, pertoken_scale=pertoken_scale,
                                                               shared_input=share_input, logit=sorted_topk_weight, row_index=row_index, output_bs=batch_size,
                                                               shared_input_weight=1.0, group_list_type=1, shared_input_offset=0).to(torch.bfloat16)
    else:
        # for v2 lite w8a8
        out = torch_npu.npu_grouped_matmul([gate_up_proj], [w2], scale=[w2_scale], per_token_scale=[pertoken_scale.float()],
                                           bias=None, group_list=expert_tokens, split_item=3, output_dtype=torch.float16,
                                           group_type=0,
                                           group_list_type=1)[0]
        out = out.to(torch.float32)
        output = torch.zeros((batch_size, hidden_size), dtype=torch.float, device="npu")
        out_tmp = out.mul(sorted_topk_weight.view(-1, 1))
        for i in range(expert_tokens.sum().item()):
            output[row_index[i]] += out_tmp[i]
        output = output.to(torch.bfloat16)
    return output
