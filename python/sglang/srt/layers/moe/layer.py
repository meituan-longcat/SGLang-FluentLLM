
import torch

from sglang.srt.layers.moe.gemms import get_gemm_cls
from sglang.srt.layers.moe.config import DispatcherType
from sglang.srt.distributed import get_moe_expert_parallel_world_size, get_moe_tensor_parallel_world_size
from sglang.srt.env import global_server_args_dict
from sglang.srt.layers.activation import SwigluArg
from sglang.srt.layers.moe.gemms.wna16 import WNA16MoEGemmWrapper


class MoELayer(torch.nn.Module):
    def __init__(
        self,
        top_k,
        num_experts,
        hidden_size,
        intermediate_size,
        quant_config,
        layer_index,
        prefix: str = "",
        zero_expert_type: str = "",
        activation = "silu",
        activation_alpha = None,
        swiglu_limit = None,
        with_bias = False,
    ):
        super().__init__()
        self.layer_index = layer_index
        self.prefix = prefix
        self.num_experts = num_experts
        self.ep_num_redundant_experts = global_server_args_dict["ep_num_redundant_experts"]
        self.zero_expert_type = zero_expert_type
        self.activation = activation
        self.swiglu_arg = None
        if self.activation == "swiglu":
            self.swiglu_arg = SwigluArg(alpha=activation_alpha, limit=swiglu_limit)

        num_local_experts = num_experts // get_moe_expert_parallel_world_size()
        intermediate_size_per_partition = intermediate_size // get_moe_tensor_parallel_world_size()

        is_ep = get_moe_expert_parallel_world_size() > 1
        dispatcher_type = DispatcherType.EPS if is_ep else DispatcherType.TP

        GEMMWrapperCls = get_gemm_cls(quant_config, is_ep, self.prefix)

        LayoutCls = GEMMWrapperCls.get_layout_cls()
        self.layout = LayoutCls(quant_config)
        self.layout.create_weights(
            self,
            num_local_experts,
            hidden_size,
            intermediate_size_per_partition,
            torch.get_default_dtype(),
            with_bias=with_bias
        )
        self.apply_routed_scaling_factor_on_output = not isinstance(GEMMWrapperCls, WNA16MoEGemmWrapper)

        self.executor = GEMMWrapperCls(top_k, num_experts, self, quant_config).get_executor(dispatcher_type, self.activation, self.swiglu_arg)

    def forward_zero_experts(self, topk_output):
        zero_expert_limit = self.num_experts 
        if self.ep_num_redundant_experts is not None:
            zero_expert_limit = zero_expert_limit - self.ep_num_redundant_experts

        normal_expert_mask = topk_output.topk_ids >= zero_expert_limit
        topk_output.topk_ids[normal_expert_mask] = -1
        if self.zero_expert_type == "copy":
            topk_output.topk_weights[normal_expert_mask] = 1.0
        if self.zero_expert_type == "drop":
            topk_output.topk_weights[normal_expert_mask] = 0.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_output,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
    ):
        #if self.zero_expert_type:
        #    self.forward_zero_experts(topk_output)
        return self.executor.forward(self, hidden_states, topk_output, num_global_tokens, max_num_tokens_per_gpu)
