from typing import Any, Optional

import torch
import torch_npu
from torch.nn import Parameter

from sglang.srt.distributed import get_attn_ep_group
from sglang.srt.env import global_server_args_dict
from sglang.srt.managers.eplb.expert_location_dispatch import (
    ExpertLocationDispatchInfo,
    topk_ids_logical_to_physical,
)
from sglang.srt.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


def _get_fused_moe_topk_op():
    return getattr(
        torch.ops.custom,
        "npu_moe_gating_top_k_expert_map",
        None,
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
        layer_idx: int = -1,
        num_physical_experts: Optional[int] = None,
    ):
        super().__init__()
        self.topk = topk  # Top-K的K值（如2、8）
        self.e_score_correction_bias = e_score_correction_bias
        self.routed_scaling_factor = routed_scaling_factor  # 权重缩放因子
        self.router_dtype = router_dtype
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.num_physical_experts = num_physical_experts

        # process() must map every logical zero/padding expert to the physical
        # zero-expert sentinel even when EPLB is disabled. Build the identity
        # map once so the Prefill hot path does not allocate or copy a tensor.
        prefill_identity_expert_map = None
        if num_physical_experts is not None:
            prefill_identity_expert_map = torch.arange(
                num_experts + 1,
                dtype=torch.int32,
                device="cpu",
            )
            prefill_identity_expert_map[-1] = num_physical_experts
            prefill_identity_expert_map = prefill_identity_expert_map.to(device=e_score_correction_bias.device)
        self.register_buffer(
            "_prefill_identity_expert_map",
            prefill_identity_expert_map,
            persistent=False,
        )

    def _validate_expert_map(self, expert_map: torch.Tensor) -> None:
        if expert_map.dim() != 1:
            raise RuntimeError(f"Fused MoE TopK requires a one-dimensional expert map, got shape={tuple(expert_map.shape)}.")
        if expert_map.shape[0] != self.num_experts + 1:
            raise RuntimeError(f"Fused MoE TopK expert-map length mismatch: expected {self.num_experts + 1}, got {expert_map.shape[0]}.")
        if expert_map.dtype != torch.int32:
            raise RuntimeError(f"Fused MoE TopK requires an int32 expert map, got {expert_map.dtype}.")

    def _get_fused_expert_map(
        self,
        dispatch_info: Optional[ExpertLocationDispatchInfo],
        *,
        allow_identity: bool,
    ) -> Optional[torch.Tensor]:
        if dispatch_info is None:
            expert_map = self._prefill_identity_expert_map if allow_identity else None
        elif dispatch_info.ep_dispatch_algorithm == "static_with_zero_expert":
            expert_map = dispatch_info.partial_logical_to_rank_dispatch_physical_map
            if expert_map is None:
                raise RuntimeError("static_with_zero_expert requires a logical-to-physical expert map.")
        else:
            expert_map = None

        if expert_map is not None:
            self._validate_expert_map(expert_map)
        return expert_map

    def _run_gating_topk(
        self,
        router_logits: torch.Tensor,
        *,
        expert_map: Optional[torch.Tensor] = None,
        out_flag: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        bool,
    ]:
        if router_logits.shape[0] == 0:
            output_shape = (0, self.topk)
            expert_weights = router_logits.new_empty(
                output_shape,
                dtype=torch.float32,
            )
            expert_topks = router_logits.new_empty(
                output_shape,
                dtype=torch.int32,
            )
            router_score = router_logits.new_empty(router_logits.shape, dtype=torch.float32) if out_flag else None
            return (
                expert_weights,
                expert_topks,
                router_score,
                expert_map is not None,
            )

        fused_op = _get_fused_moe_topk_op()
        fused_disabled = "moe_topk" in (global_server_args_dict.get("npu_disabled_fused_kernels") or ())
        if expert_map is not None and fused_op is not None and not fused_disabled:
            expert_weights, expert_topks, router_score = fused_op(
                router_logits.float(),
                expert_map.contiguous(),
                self.e_score_correction_bias,
                k=self.topk,
                k_group=1,
                group_count=1,
                group_select_mode=0,
                renorm=0,
                norm_type=0,
                out_flag=False,
                routed_scaling_factor=self.routed_scaling_factor,
                eps=1e-20,
            )
            return expert_weights, expert_topks, router_score, True

        if expert_map is not None and fused_op is None and not fused_disabled:
            logger.warning_once("npu_moe_gating_top_k_expert_map is unavailable; falling back to separate TopK and expert mapping.")

        expert_weights, expert_topks, router_score = torch_npu.npu_moe_gating_top_k(
            router_logits.float(),
            k=self.topk,
            bias=self.e_score_correction_bias,
            k_group=1,
            group_count=1,
            group_select_mode=0,
            renorm=0,
            norm_type=0,
            out_flag=out_flag,
            routed_scaling_factor=self.routed_scaling_factor,
            eps=1e-20,
        )
        return expert_weights, expert_topks, router_score, False

    def forward(
        self, router_logits: torch.Tensor, input_dtype: Any, need_probs: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        NPU Top-K核心逻辑：筛选Top-K专家索引，提取对应权重
        """
        expert_location_dispatch_info = ExpertLocationDispatchInfo.init_new(
            layer_id=self.layer_idx,
        )
        expert_map = self._get_fused_expert_map(
            expert_location_dispatch_info,
            allow_identity=False,
        )
        (
            expert_weights,
            expert_topks,
            router_score,
            expert_ids_are_mapped,
        ) = self._run_gating_topk(
            router_logits,
            expert_map=expert_map if not need_probs else None,
            out_flag=need_probs,
        )

        if expert_ids_are_mapped:
            return expert_topks, expert_weights, None

        # 生成概率矩阵（用于后续token分配，仅Top-K专家有非零权重）
        probs = None
        if need_probs:
            if router_score is None:
                raise RuntimeError("MoE TopK did not return router scores required for probability output.")
            probs = torch.zeros_like(router_score)
            probs.scatter_(dim=-1, index=expert_topks.to(torch.int64), src=expert_weights)
            probs = probs.T.contiguous()
            # 当前probs只支持和hidden_states相同dtype
            probs = probs.to(input_dtype)

        expert_topks = topk_ids_logical_to_physical(
            expert_topks,
            info=expert_location_dispatch_info,
            num_experts=self.num_experts,
        )
        return expert_topks, expert_weights, probs  # expert_topks=Top-K专家索引，expert_weights=对应权重

    def process(
        self,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.num_physical_experts is None:
            raise RuntimeError("LongCatTopK.process() requires num_physical_experts to be set.")

        expert_location_dispatch_info = ExpertLocationDispatchInfo.init_new(
            layer_id=self.layer_idx,
        )

        expert_map = self._get_fused_expert_map(
            expert_location_dispatch_info,
            allow_identity=True,
        )
        (
            expert_weights,
            expert_topks,
            _,
            expert_ids_are_mapped,
        ) = self._run_gating_topk(
            router_logits,
            expert_map=expert_map,
            out_flag=False,
        )
        if expert_ids_are_mapped:
            return expert_topks, expert_weights

        zero_expert_mask = expert_topks >= self.num_experts

        expert_topks = topk_ids_logical_to_physical(
            expert_topks,
            info=expert_location_dispatch_info,
            num_experts=self.num_experts,
        )
        expert_topks = torch.where(
            zero_expert_mask,
            torch.tensor(self.num_physical_experts, device=expert_topks.device, dtype=expert_topks.dtype),
            expert_topks,
        )
        return expert_topks, expert_weights


class DeepSeekTopK(torch.nn.Module):
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
        layer_idx: int = -1,
        num_expert_group=1,
        topk_group=1,
        renormalize=True,
        scoring_func="sigmod",
        num_fused_shared_experts=0,
    ):
        super().__init__()
        self.topk = topk  # Top-K的K值（如2、8）
        self.e_score_correction_bias = e_score_correction_bias
        self.routed_scaling_factor = routed_scaling_factor  # 权重缩放因子
        self.router_dtype = router_dtype
        self.layer_idx = layer_idx
        self.num_experts = num_experts

        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        assert renormalize
        self.num_fused_shared_experts = num_fused_shared_experts

        if scoring_func == "softmax":
            self.norm_type = 0
        elif scoring_func == "sigmoid":
            self.norm_type = 1
        else:
            raise AssertionError("Unknown scoring function")

    def forward(
        self,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        NPU Top-K核心逻辑：筛选Top-K专家索引，提取对应权重
        """
        expert_weights, expert_topks, router_score = torch_npu.npu_moe_gating_top_k(
            router_logits.float(),
            k=self.topk,
            bias=self.e_score_correction_bias,
            k_group=self.topk_group,
            group_count=self.num_expert_group,
            group_select_mode=1,  # 表示取组内Top2的专家进行得分累加，作为专家组得分
            renorm=0,  # 0=用softmax归一化
            norm_type=self.norm_type,  # 0: softmax;  1: sigmoid
            out_flag=True,  # 是否输出norm函数中间结果
            routed_scaling_factor=self.routed_scaling_factor,
            eps=1e-20,
        )

        expert_location_dispatch_info = ExpertLocationDispatchInfo.init_new(
            layer_id=self.layer_idx,
        )
        expert_topks = topk_ids_logical_to_physical(
            expert_topks,
            info=expert_location_dispatch_info,
            num_experts=self.num_experts,
        )
        return expert_topks, expert_weights


class Router(torch.nn.Module):
    """
    Router for using tokens choose top-k experts assignment.

    This router uses the same mechanism as in Switch Transformer (https://arxiv.org/abs/2101.03961) and V-MoE
    (https://arxiv.org/abs/2106.05974): tokens choose their top experts. Items are sorted by router_probs and then
    routed to their choice of expert until the expert's expert_capacity is reached. **There is no guarantee that each
    token is processed by an expert**, or that each expert receives at least one token.

    """

    def __init__(self, config, layer_idx: int, num_experts: int, router_dtype: Any = torch.float32):
        super().__init__()
        from sglang.srt.env import global_server_args_dict

        self.layer_idx = layer_idx
        self.router_dtype = router_dtype
        # Split the router GEMM along its reduction (K/hidden) dim over the
        # attn tp group: every rank keeps only its K-slice of the classifier
        # weight, computes partial logits from the matching K-slice of the
        # gathered hidden states, and a reduce_scatter sums the partials and
        # hands every rank the logits of its own token shard.
        self.split_k = global_server_args_dict.get("npu_split_k_router", False)
        self.tp_size = 1
        self.tp_rank = 0
        in_features = config.hidden_size
        if self.split_k:
            self.comm_group = get_attn_ep_group()
            self.tp_size = self.comm_group.world_size
            self.tp_rank = self.comm_group.rank_in_group
            assert config.hidden_size % self.tp_size == 0
            in_features = config.hidden_size // self.tp_size
        self.classifier = torch.nn.Linear(
            in_features,
            num_experts,
            bias=getattr(config, "router_bias", None),
            dtype=self.router_dtype,
        )
        if self.split_k:
            self._register_k_split_weight_loader(in_features)
        self.e_score_correction_bias = Parameter(torch.empty(num_experts, device=torch.npu.current_device(), dtype=torch.float32))
        self.jitter_noise = getattr(config, "router_jitter_noise", None)
        if self.split_k:
            assert self.jitter_noise is None

    def _register_k_split_weight_loader(self, k_len: int):
        # The checkpoint stores the full [E, H] router weight, while every
        # rank's classifier only keeps its K-dim slice [E, H/tp].
        k_start = self.tp_rank * k_len

        def weight_loader(param, loaded_weight):
            loaded_weight = loaded_weight.narrow(1, k_start, k_len)
            assert loaded_weight.shape == param.shape, f"router weight shard shape mismatch: {loaded_weight.shape=} {param.shape=}"
            param.data.copy_(loaded_weight)

        self.classifier.weight.weight_loader = weight_loader

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_ctrl: bool = False,
    ):
        """
        Computes token scores, token passed to which expert(mask) and moe_loss(if using)
        Args:
            hidden_states [s, b, h]
        """
        if not self.split_k:
            hidden_states = hidden_states.to(self.router_dtype)

            if self.jitter_noise:
                distrib_lower_bound = 1.0 - self.jitter_noise
                distrib_upper_bound = 1.0 + self.jitter_noise
                uniform_distrib = torch.rand(
                    hidden_states.shape,
                    device=hidden_states.device,
                    dtype=self.router_dtype,
                )
                uniform_distrib = uniform_distrib * (distrib_lower_bound - distrib_upper_bound)
                uniform_distrib = uniform_distrib + distrib_upper_bound
                hidden_states *= uniform_distrib

            router_logits = self.classifier(hidden_states)  # [s, b, e]
            if return_ctrl:
                return router_logits, router_logits
            return router_logits
        else:
            k_len = self.classifier.in_features
            assert hidden_states.shape[0] % self.tp_size == 0 and hidden_states.shape[1] == self.tp_size * k_len
            k_start = self.tp_rank * k_len
            # [T, H/tp] @ [H/tp, E], bias is added once after the reduction
            split_k_hidden_states = torch.ops.custom.npu_router_input_slice_cast(hidden_states, k_start, k_len)
            # todo sport jitter_noise
            split_scores = self.classifier(split_k_hidden_states)
            router_logits = self.comm_group.reduce_scatter(split_scores)
            if return_ctrl:
                return router_logits, split_scores
            return router_logits
