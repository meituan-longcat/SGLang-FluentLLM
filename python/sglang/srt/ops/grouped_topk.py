# Copyright 2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Grouped top-k expert routing, with and without e_score_correction_bias.

Used by the DeepSeek V2/V3/R1 and GLM-5 series models. The biased variants
implement noaux_tc routing: the correction bias only affects expert
selection, the returned weights are the raw (sigmoid) scores.
"""

import math
from typing import Optional

import torch

from sglang.srt.managers.eplb.expert_location_dispatch import (
    ExpertLocationDispatchInfo,
    topk_ids_logical_to_physical,
)
from sglang.srt.utils import get_compiler_backend, is_cuda, is_npu

_is_cuda = is_cuda()
_is_npu = is_npu()

if not _is_npu:
    from flashinfer import moe_fused_gate


def is_power_of_two(n):
    return n > 0 and math.log2(n).is_integer()


def _mask_topk_ids_padded_region(
    topk_ids: torch.Tensor,
    num_token_non_padded: Optional[torch.Tensor] = None,
):
    if num_token_non_padded is None:
        return
    indices = torch.arange(0, topk_ids.shape[0], device=topk_ids.device)
    topk_ids[indices >= num_token_non_padded, :] = -1


# This is used by the Deepseek V2/V3/R1 series models
@torch.compile(dynamic=True, backend=get_compiler_backend())
def grouped_topk_gpu(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    scores = torch.softmax(gating_output, dim=-1)
    # NPU compiler limitation
    if _is_npu and scores.dtype == torch.bfloat16:
        scores = scores.to(torch.float16)
    num_token = scores.shape[0]
    num_experts = scores.shape[1]
    group_scores = scores.view(num_token, num_expert_group, -1).max(dim=-1).values  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = group_mask.unsqueeze(-1).expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group).reshape(num_token, -1)  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
    # TODO: NPU can't support directly evaluating a comparison for now
    topk_weights, topk_ids = torch.topk(
        tmp_scores,
        k=topk,
        dim=-1,
        sorted=num_fused_shared_experts > 0,
    )
    if num_fused_shared_experts:
        topk_ids[:, -1] = torch.randint(
            low=num_experts,
            high=num_experts + num_fused_shared_experts,
            size=(topk_ids.size(0),),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )
        topk_weights[:, -1] = topk_weights[:, :-1].sum(dim=-1) / routed_scaling_factor

    if renormalize:
        topk_weights_sum = topk_weights.sum(dim=-1, keepdim=True) if num_fused_shared_experts == 0 else topk_weights[:, :-1].sum(dim=-1, keepdim=True)
        topk_weights = topk_weights / topk_weights_sum

    if apply_routed_scaling_factor_on_output:
        topk_weights *= routed_scaling_factor

    topk_weights, topk_ids = topk_weights.to(torch.float32), topk_ids.to(torch.int32)
    topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
    _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)
    return topk_weights, topk_ids


@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=_is_npu)
def biased_grouped_topk_impl(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = 1.0,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    scores = gating_output.sigmoid()
    num_token = scores.shape[0]
    num_experts = scores.shape[1]
    scores_for_choice = scores.view(num_token, -1) + correction_bias.unsqueeze(0)
    group_scores = scores_for_choice.view(num_token, num_expert_group, -1).topk(2, dim=-1)[0].sum(dim=-1)  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = group_mask.unsqueeze(-1).expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group).reshape(num_token, -1)  # [n, e]
    tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))  # [n, e]
    # TODO: NPU can't support directly evaluating a comparison for now
    _, topk_ids = torch.topk(
        tmp_scores,
        k=topk,
        dim=-1,
        sorted=num_fused_shared_experts > 0,
    )
    topk_weights = scores.gather(1, topk_ids)

    if num_fused_shared_experts:
        topk_ids[:, -1] = torch.randint(
            low=num_experts,
            high=num_experts + num_fused_shared_experts,
            size=(topk_ids.size(0),),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )
        topk_weights[:, -1] = topk_weights[:, :-1].sum(dim=-1) / routed_scaling_factor

    if renormalize:
        topk_weights_sum = topk_weights.sum(dim=-1, keepdim=True) if num_fused_shared_experts == 0 else topk_weights[:, :-1].sum(dim=-1, keepdim=True)
        topk_weights = topk_weights / topk_weights_sum

    if apply_routed_scaling_factor_on_output:
        topk_weights *= routed_scaling_factor

    topk_weights, topk_ids = topk_weights.to(torch.float32), topk_ids.to(torch.int32)
    topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
    _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)
    return topk_weights, topk_ids


@torch.compile(dynamic=True, backend=get_compiler_backend())
def _biased_grouped_topk_postprocess(topk_ids, expert_location_dispatch_info, num_token_non_padded):
    topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
    _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)
    return topk_ids


def biased_grouped_topk_gpu(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
):
    assert routed_scaling_factor is not None, "routed_scaling_factor is required for biased_grouped_topk"
    # TODO: moe_fused_gate kernel is not supported for num_fused_shared_experts > 0 now.
    if (
        _is_cuda
        # moe_fused_gate kernel ensure that num_experts/num_expert_group
        # does not exceed MAX_VPT=32 now. And when kernel can handle MAX_VPT > 32,
        # we can remove this assertion.
        and gating_output.shape[1] // num_expert_group <= 32
        and is_power_of_two(correction_bias.shape[0])
    ):
        topk_weights, topk_ids = moe_fused_gate(
            gating_output.to(dtype=torch.float32),
            correction_bias,
            num_expert_group,
            topk_group,
            topk,
            num_fused_shared_experts,
            routed_scaling_factor,
            apply_routed_scaling_factor_on_output,
        )
        # TODO merge into kernel
        if (expert_location_dispatch_info is not None) or (num_token_non_padded is not None):
            topk_ids = _biased_grouped_topk_postprocess(topk_ids, expert_location_dispatch_info, num_token_non_padded)
        return topk_weights, topk_ids
    else:
        return biased_grouped_topk_impl(
            hidden_states,
            gating_output,
            correction_bias,
            topk,
            renormalize,
            num_expert_group,
            topk_group,
            num_fused_shared_experts=num_fused_shared_experts,
            routed_scaling_factor=routed_scaling_factor,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
            apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
        )


grouped_topk = grouped_topk_gpu
biased_grouped_topk = biased_grouped_topk_gpu
