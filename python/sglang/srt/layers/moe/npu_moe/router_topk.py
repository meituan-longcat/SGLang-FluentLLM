from typing import Any, Optional, Tuple

import torch
import torch_npu
from torch.nn import Parameter

from sglang.srt.managers.expert_location_dispatch import (
    ExpertLocationDispatchInfo,
    topk_ids_logical_to_physical,
)


class LongCatTopK(torch.nn.Module):
    """
    Top-K专家选择. For NPU only.
    """
    def __init__(
            self,
            topk: int,
            e_score_correction_bias: Parameter,
            routed_scaling_factor: float = 1.0,
            num_experts: int = 0,
            router_dtype: Any = torch.float32,
            layer_idx: int = -1):
        super().__init__()
        self.topk = topk  # Top-K的K值（如2、8）
        self.e_score_correction_bias = e_score_correction_bias
        self.routed_scaling_factor = routed_scaling_factor  # 权重缩放因子
        self.router_dtype = router_dtype
        self.layer_idx = layer_idx
        self.num_experts = num_experts

    def forward(
        self,
        router_logits: torch.Tensor,
        input_dtype: Any,
        need_probs: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        NPU Top-K核心逻辑：筛选Top-K专家索引，提取对应权重
        """
        expert_weights, expert_topks, router_score = torch_npu.npu_moe_gating_top_k(
            router_logits.float(),
            k=self.topk,
            bias=self.e_score_correction_bias,
            k_group = 1,
            group_count = 1,
            group_select_mode = 0,  # 专家分组时，按照组内最大值对group进行排序
            renorm=0,  # 0=用softmax归一化
            norm_type = 0, # 0: softmax;  1: sigmoid
            out_flag = True, # 是否输出norm函数中间结果
            routed_scaling_factor=self.routed_scaling_factor,
            eps=1e-20
        )

        # 生成概率矩阵（用于后续token分配，仅Top-K专家有非零权重）
        probs = None
        if need_probs:
            probs = torch.zeros_like(router_score)
            probs.scatter_(dim=-1, index=expert_topks.to(torch.int64), src=expert_weights)
            probs = probs.T.contiguous()
            # 当前probs只支持和hidden_states相同dtype
            probs = probs.to(input_dtype)

        expert_location_dispatch_info = ExpertLocationDispatchInfo.init_new(
            layer_id=self.layer_idx,
        )
        expert_topks = topk_ids_logical_to_physical(
            expert_topks,
            info=expert_location_dispatch_info,
            num_experts=self.num_experts,
        )
        return expert_topks, expert_weights, probs  # expert_topks=Top-K专家索引，expert_weights=对应权重

class Router(torch.nn.Module):
    """
    Router for using tokens choose top-k experts assignment.

    This router uses the same mechanism as in Switch Transformer (https://arxiv.org/abs/2101.03961) and V-MoE
    (https://arxiv.org/abs/2106.05974): tokens choose their top experts. Items are sorted by router_probs and then
    routed to their choice of expert until the expert's expert_capacity is reached. **There is no guarantee that each
    token is processed by an expert**, or that each expert receives at least one token.

    """

    def __init__(
            self,
            config,
            layer_idx: int,
            num_experts: int,
            router_dtype: Any = torch.float32):
        super().__init__()
        self.layer_idx = layer_idx
        self.router_dtype = router_dtype
        self.classifier = torch.nn.Linear(
            config.hidden_size,
            num_experts,
            bias=config.router_bias,
            dtype=self.router_dtype,
        )
        self.e_score_correction_bias = Parameter(
            torch.empty(num_experts, device=torch.npu.current_device(), dtype=torch.float32))
        self.jitter_noise = getattr(config, "router_jitter_noise", None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Computes token scores, token passed to which expert(mask) and moe_loss(if using)
        Args:
            hidden_states [s, b, h]
        """
        hidden_states = hidden_states.to(self.router_dtype)
        if self.jitter_noise:
            distrib_lower_bound = 1.0 - self.jitter_noise
            distrib_upper_bound = 1.0 + self.jitter_noise
            uniform_distrib = torch.rand(
                hidden_states.shape,
                device=hidden_states.device,
                dtype=self.router_dtype,
            )
            uniform_distrib = uniform_distrib * (
                distrib_lower_bound - distrib_upper_bound
            )
            uniform_distrib = uniform_distrib + distrib_upper_bound
            hidden_states *= uniform_distrib

        return self.classifier(hidden_states)  # [s, b, e]
