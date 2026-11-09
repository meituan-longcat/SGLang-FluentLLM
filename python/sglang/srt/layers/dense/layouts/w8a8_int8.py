from __future__ import annotations

from typing import (
    List,
    Optional,
    Tuple,
    Union,
)

import torch
from torch.nn.parameter import Parameter

from sglang.srt.layers.quantization import W8A8Int8Config
from sglang.srt.layers.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
)
from sglang.srt.layers.quantization.base_config import (
    LinearMethodBase,
)
from sglang.srt.layers.dense.gemms.int8.int8_kernel import per_token_quant_int8
from sglang.srt.utils import (
    cpu_has_amx_support,
    is_cpu,
    is_cuda,
    is_npu,
)


_is_cuda = is_cuda()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
if _is_cuda:
    try:
        from flashinfer import int8_scaled_mm
    except ImportError as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"Failed to import int8_scaled_mm from flashinfer: {e}")
        raise
_is_npu = is_npu()

if _is_npu:
    import torch_npu


# func refers to RMSNorm.__init__
def npu_wrapper_rmsnorm_init(func):
    def init(self, hidden_size: int, **extra_args) -> None:
        func(self, hidden_size, **extra_args)
        self.ignore_anti = True
        # The Ascend w8a8_int8 quantization requires adding a bias in rmsnorm
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size), requires_grad=False)

    return init


# func refers to RMSNorm.forward_oot
def npu_wrapper_rmsnorm_forward(func):
    def _rmsnorm_forward_oot(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if not x.is_contiguous():
            x = x.contiguous()
        if residual is not None:
            out, _, residual_out = torch_npu.npu_add_rms_norm(residual, x, self.weight.data, self.variance_epsilon)
            out = out + self.bias
            return out.to(x.dtype), residual_out

        out = torch_npu.npu_rms_norm(x, self.weight.data, self.variance_epsilon)[0]
        out = out + self.bias
        return out.to(x.dtype)

    return _rmsnorm_forward_oot


def npu_fused_experts(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
):
    original_shape = hidden_states.shape
    original_dtype = hidden_states.dtype
    scale_dtype = original_dtype if original_dtype == torch.bfloat16 else torch.float32
    if len(original_shape) == 3:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    num_tokens = hidden_states.shape[0]
    num_experts = w13.shape[0]
    row_idx_len = num_tokens * top_k
    row_idx = torch.arange(0, row_idx_len, dtype=torch.int32, device=topk_weights.device).view(top_k, -1).permute(1, 0).contiguous()
    hidden_states, expanded_row_idx, expanded_expert_idx = torch_npu.npu_moe_init_routing(
        hidden_states, row_idx=row_idx, expert_idx=topk_ids, active_num=num_tokens
    )
    expert_tokens = torch_npu.npu_moe_compute_expert_tokens(expanded_expert_idx, num_experts)
    expert_tokens = expert_tokens.to(torch.int64)
    # gmm1: gate_up_proj
    hidden_states, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)
    hidden_states = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w13],
        scale=[w13_scale.to(scale_dtype)],
        per_token_scale=[pertoken_scale],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]
    # act_fn: swiglu
    hidden_states = torch_npu.npu_swiglu(hidden_states)
    hidden_states, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)
    # gmm2: down_proj
    hidden_states = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w2],
        scale=[w2_scale.to(scale_dtype)],
        per_token_scale=[pertoken_scale],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]

    final_hidden_states = torch_npu.npu_moe_finalize_routing(
        hidden_states,
        skip1=None,
        skip2=None,
        bias=None,
        scales=topk_weights,
        expanded_src_to_dst_row=expanded_row_idx,
        export_for_source_row=topk_ids,
    )
    if len(original_shape) == 3:
        final_hidden_states = final_hidden_states.view(original_shape)
    return final_hidden_states


class W8A8Int8LinearMethod(LinearMethodBase):
    def __init__(self, quantization_config: W8A8Int8Config):
        self.quantization_config = quantization_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight = Parameter(layer.weight.t(), requires_grad=False)
        layer.weight_scale = Parameter(layer.weight_scale.data, requires_grad=False)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):

        weight_loader = extra_weight_attrs.get("weight_loader")
        self.logical_widths = output_partition_sizes

        weight = ModelWeightParameter(
            data=torch.empty(sum(output_partition_sizes), input_size_per_partition, dtype=torch.int8),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        weight_scale = ChannelQuantScaleParameter(
            data=torch.empty((sum(output_partition_sizes), 1), dtype=torch.float32),
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        x_q, x_scale = per_token_quant_int8(x)

        x_q_2d = x_q.view(-1, x_q.shape[-1])
        x_scale_2d = x_scale.view(-1, x_scale.shape[-1])
        output_shape = [*x_q.shape[:-1], layer.weight.shape[1]]

        output = int8_scaled_mm(
            x_q_2d,
            layer.weight,
            x_scale_2d,
            layer.weight_scale,
            out_dtype=x.dtype,
            bias=bias,
        )

        return output.view(output_shape)
