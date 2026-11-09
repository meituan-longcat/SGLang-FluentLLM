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
"""Ungrouped softmax top-k expert routing (e.g. Qwen3-MoE style)."""

from typing import Optional

import torch
import torch.nn.functional as F

from sglang.srt.managers.eplb.expert_location_dispatch import (
    ExpertLocationDispatchInfo,
    topk_ids_logical_to_physical,
)
from sglang.srt.ops.grouped_topk import _mask_topk_ids_padded_region
from sglang.srt.utils import is_npu

if not is_npu():
    from flashinfer import topk_softmax


def fused_topk_native(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    correction_bias: torch.Tensor = None,
):
    if correction_bias is not None:
        n_routed_experts = gating_output.shape[-1]
        scores = gating_output.softmax(dim=-1)
        scores_for_choice = scores.view(-1, n_routed_experts) + correction_bias.unsqueeze(0)
        topk_ids = torch.topk(scores_for_choice, k=topk, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_ids)
    else:
        assert hidden_states.shape[0] == gating_output.shape[0], f"Number of tokens mismatch, {hidden_states.shape=} vs {gating_output.shape=}"
        M, _ = hidden_states.shape
        topk_weights = torch.empty(M, topk, dtype=torch.float32, device=hidden_states.device)
        topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
        topk_weights = F.softmax(gating_output.float(), dim=-1)
        topk_weights, topk_ids = torch.topk(topk_weights, topk, dim=-1)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    M, _ = hidden_states.shape

    topk_weights = torch.empty(M, topk, dtype=torch.float32, device=hidden_states.device)
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)

    topk_softmax(
        topk_weights,
        topk_ids,
        gating_output,
        renormalize,
    )

    topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
    _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)
    return topk_weights, topk_ids
