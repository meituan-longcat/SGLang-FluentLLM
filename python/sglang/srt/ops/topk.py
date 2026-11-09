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

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, NamedTuple, Optional, Protocol, runtime_checkable

import torch
from typing_extensions import TypeGuard

from sglang.srt.managers.eplb import expert_location_dispatch
from sglang.srt.managers.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.managers.eplb.expert_location_dispatch import ExpertLocationDispatchInfo, topk_ids_logical_to_physical
from sglang.srt.ops.custom_op import CustomOp
from sglang.srt.ops.grouped_topk import (
    biased_grouped_topk,
    grouped_topk,
)
from sglang.srt.utils import is_cuda, is_npu

_is_cuda = is_cuda()
_is_npu = is_npu()

logger = logging.getLogger(__name__)

if not _is_npu:
    import flashinfer


class TopKOutputFormat(Enum):
    STANDARD = auto()
    BYPASSED = auto()

    def is_standard(self) -> bool:
        return self == TopKOutputFormat.STANDARD

    def is_bypassed(self) -> bool:
        return self == TopKOutputFormat.BYPASSED


@dataclass
class TopKConfig:
    top_k: int
    layer_id: int = None
    use_grouped_topk: bool = False
    topk_group: Optional[int] = None
    num_expert_group: Optional[int] = None
    renormalize: bool = True
    num_fused_shared_experts: int = 0
    custom_routing_function: Optional[Callable] = None
    correction_bias: Optional[torch.Tensor] = None
    torch_native: bool = False
    routed_scaling_factor: Optional[float] = None
    apply_routed_scaling_factor_on_output: bool = False
    output_format: Optional[TopKOutputFormat] = None
    zero_expert_num: Optional[int] = 0
    topk_indices_dtype: Optional[torch.dtype] = torch.int32


class StandardTopKOutput(NamedTuple):
    """Standard top-k output format."""

    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    router_logits: torch.Tensor

    @property
    def format(self) -> TopKOutputFormat:
        return TopKOutputFormat.STANDARD


class BypassedTopKOutput(NamedTuple):
    """Bypassed top-k output format."""

    hidden_states: torch.Tensor
    router_logits: torch.Tensor
    topk_config: TopKConfig
    num_token_non_padded: Optional[torch.Tensor] = None
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None

    @property
    def format(self) -> TopKOutputFormat:
        return TopKOutputFormat.BYPASSED


@runtime_checkable
class TopKOutput(Protocol):
    """Protocol for top-k outputs in different formats."""

    @property
    def format(self) -> TopKOutputFormat:
        """The format of the output."""
        ...


if not _is_npu:

    class TopKOutputChecker:
        @staticmethod
        def format_is_standard(topk_output: TopKOutput) -> TypeGuard[StandardTopKOutput]:
            return topk_output.format.is_standard()

        # commit 25cf2e6bcd7ea54e353387de75781129e4f9459b “remove triton_kernels (#518)”中删除了@staticmethod，疑为误删。
        @staticmethod
        def format_is_bypassed(topk_output: TopKOutput) -> TypeGuard[BypassedTopKOutput]:
            return topk_output.format.is_bypassed()


class TopK(CustomOp):
    def __init__(
        self,
        top_k: int,
        layer_id: int,
        *,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        renormalize: bool = True,
        num_fused_shared_experts: int = 0,
        custom_routing_function: Optional[Callable] = None,
        correction_bias: Optional[torch.Tensor] = None,
        routed_scaling_factor: Optional[float] = None,
        apply_routed_scaling_factor_on_output: Optional[bool] = False,
        output_format: Optional[TopKOutputFormat] = None,
        zero_expert_num: Optional[int] = 0,
        topk_indices_dtype=torch.int32,
    ):
        super().__init__()

        if use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None

        self.topk_config = TopKConfig(
            top_k=top_k,
            layer_id=layer_id,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            num_fused_shared_experts=num_fused_shared_experts,
            custom_routing_function=custom_routing_function,
            correction_bias=correction_bias,
            routed_scaling_factor=routed_scaling_factor,
            apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
            output_format=output_format,
            zero_expert_num=zero_expert_num,
            topk_indices_dtype=topk_indices_dtype,
        )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        num_token_non_padded: Optional[torch.Tensor] = None,
        expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    ) -> TopKOutput:
        output_format = self.topk_config.output_format if self.topk_config.output_format is not None else TopKOutputFormat.STANDARD

        if output_format == TopKOutputFormat.BYPASSED:
            return BypassedTopKOutput(
                hidden_states=hidden_states,
                router_logits=router_logits,
                topk_config=self.topk_config,
                num_token_non_padded=num_token_non_padded,
                expert_location_dispatch_info=expert_location_dispatch_info,
            )
        else:
            self.topk_config.torch_native = False
            return select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                topk_config=self.topk_config,
                num_token_non_padded=num_token_non_padded,
                expert_location_dispatch_info=expert_location_dispatch_info,
            )

    def empty_topk_output(self, device: torch.device) -> TopKOutput:
        topk = self.topk_config.top_k - self.topk_config.num_fused_shared_experts
        topk_weights = torch.empty((0, topk), dtype=torch.float32, device=device)
        topk_idx = torch.full((0, topk), -1, dtype=torch.int32, device=device)
        router_logits = torch.empty((0, topk), dtype=torch.float32, device=device)
        return StandardTopKOutput(topk_weights, topk_idx, router_logits)


# ------------------------------- TopK implementation -------------------------------------
def select_experts(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    topk_config: TopKConfig,
    *,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
) -> StandardTopKOutput:
    top_k = topk_config.top_k
    layer_id = topk_config.layer_id
    use_grouped_topk = topk_config.use_grouped_topk
    topk_group = topk_config.topk_group
    num_expert_group = topk_config.num_expert_group
    renormalize = topk_config.renormalize
    num_fused_shared_experts = topk_config.num_fused_shared_experts
    custom_routing_function = topk_config.custom_routing_function
    correction_bias = topk_config.correction_bias
    torch_native = topk_config.torch_native
    routed_scaling_factor = topk_config.routed_scaling_factor
    apply_routed_scaling_factor_on_output = topk_config.apply_routed_scaling_factor_on_output

    router_logits, correction_bias = expert_location_dispatch.transform_select_experts_inputs(
        router_logits=router_logits,
        correction_bias=correction_bias,
        info=expert_location_dispatch_info,
    )

    assert not torch_native

    # DeepSeek V2/V3/R1 series models use grouped_top_k
    if use_grouped_topk:
        assert topk_group is not None
        assert num_expert_group is not None
        if correction_bias is None:
            topk_weights, topk_ids = grouped_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                num_fused_shared_experts=num_fused_shared_experts,
                routed_scaling_factor=routed_scaling_factor,
                num_token_non_padded=num_token_non_padded,
                expert_location_dispatch_info=expert_location_dispatch_info,
                apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
            )
        else:
            topk_weights, topk_ids = biased_grouped_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                correction_bias=correction_bias,
                topk=top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                num_fused_shared_experts=num_fused_shared_experts,
                routed_scaling_factor=routed_scaling_factor,
                num_token_non_padded=num_token_non_padded,
                expert_location_dispatch_info=expert_location_dispatch_info,
                apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
            )
    elif topk_config.zero_expert_num > 0:
        assert correction_bias is not None
        # routing_flash always folds routed_scaling_factor into the weights, so
        # the caller must not multiply the expert output by it again.
        assert apply_routed_scaling_factor_on_output, "routing_flash applies routed_scaling_factor in topk weights unconditionally"
        num_tokens = router_logits.shape[0]
        topk_ids = torch.empty(num_tokens, top_k, device=router_logits.device, dtype=topk_config.topk_indices_dtype)
        topk_weights = torch.empty(num_tokens, top_k, device=router_logits.device, dtype=torch.float32)
        num_real_experts = router_logits.shape[1] - topk_config.zero_expert_num
        flashinfer.routing_flash(router_logits, correction_bias, topk_ids, topk_weights, num_real_experts, routed_scaling_factor, False)
        expert_location_dispatch_info = ExpertLocationDispatchInfo.init_new(
            layer_id=layer_id,
        )
        zero_expert_mask = topk_ids == -1
        topk_ids = topk_ids_logical_to_physical(
            topk_ids,
            info=expert_location_dispatch_info,
            num_experts=num_real_experts,
        )
        topk_ids[zero_expert_mask] = -1
    else:
        assert num_token_non_padded is None, "num_token_non_padded is not yet supported in custom_routing_function"
        assert expert_location_dispatch_info is None
        assert not apply_routed_scaling_factor_on_output, "Not implemented"
        topk_weights, topk_ids = custom_routing_function(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=top_k,
            renormalize=renormalize,
        )

    get_global_expert_distribution_recorder().on_select_experts(topk_ids=topk_ids)

    return StandardTopKOutput(topk_weights, topk_ids, router_logits)
