from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import nn
import torch_npu
from torch.nn import Parameter

from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear
)
from sglang.srt.distributed import get_mlp_tp_group
from sglang.srt.layers.quantization.base_config import QuantizationConfig


class NPUParallelMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
    ):
        super().__init__()
        self.quant_config = quant_config
        self.mlp_tp_group = get_mlp_tp_group()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=bias,
            gather_output=False,
            quant_config=quant_config,
            outside_tp_group=self.mlp_tp_group,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=bias,
            input_is_parallel=True,
            reduce_results=False,
            quant_config=quant_config,
            outside_tp_group=self.mlp_tp_group,
        )

        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.gate_up_proj.throw_dequant = True

    def act_fn(self, x):
        if self.quant_config is not None and isinstance(x, (List, Tuple)):
            x, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(x[0], weight_scale=self.gate_up_proj.weight_scale.to(torch.float),
                                                            group_index=None, activation_scale=x[1], activate_left=True, quant_mode=1,
                                                            quant_scale=self.down_proj.smooth_scale)
            return x, pertoken_scale
        else:
            return torch_npu.npu_swiglu(x, dim=-1)

    def prefetch_gateup(self, dependency):
        torch_npu.npu_prefetch(self.gate_up_proj.weight, dependency,
                               self.gate_up_proj.weight.numel() * self.gate_up_proj.weight.element_size())

    def prefetch_down(self, dependency):
        torch_npu.npu_prefetch(self.down_proj.weight, dependency,
                               self.down_proj.weight.numel() * self.down_proj.weight.element_size())

    def pre_comm(self, hidden_states, output_split_sizes=None):
        output = []
        tensors = hidden_states if isinstance(hidden_states, (list, tuple)) else [hidden_states]
        for tensor in tensors:
            tensor = tensor.view(-1, tensor.size(-1)) if len(tensor.size()) > 2 else tensor
            out_tensor=self.mlp_tp_group.all_gather(tensor, dim=0, output_split_sizes=output_split_sizes)
            output.append(out_tensor)
        output = output if isinstance(hidden_states, (list, tuple)) else output[0]
        return output

    def pre_quant(self, hidden_states):
        return torch_npu.npu_dynamic_quant(hidden_states, smooth_scales=self.gate_up_proj.smooth_scale)

    def pre_compact_allgather(self, hidden_states, world_len_list, comm_tensor: torch.Tensor=None):
        # for query len with large range in dp prefill, cooperate with post_compact_reduce_scatter
        pass

    def forward(self,
                hidden_states,
                pertoken_scale=None,
                reduce_type=None,  # 'skip', 'all_reduce' or 'reduce_scatter', default is 'all_reduce'
                return_ctrl=False,
                input_split_sizes=None):
        if pertoken_scale is not None:
            gate_up, _ = self.gate_up_proj((hidden_states, pertoken_scale))
        else:
            gate_up, _ = self.gate_up_proj(hidden_states)
        x = self.act_fn(gate_up)
        gate_down, _ = self.down_proj(x)
        output = self.post_comm(gate_down, reduce_type, input_split_sizes)
        if return_ctrl:
            # second/third out for prefetch weight dependency
            return output, gate_up[0] if isinstance(gate_up, (List, Tuple)) else gate_up, gate_down
        return output

    def post_comm(self, hidden_states, reduce_type=None, input_split_sizes=None):
        if reduce_type == 'skip':
            return hidden_states
        reduce_type = reduce_type or 'all_reduce'
        assert reduce_type in ['all_reduce', 'reduce_scatter']
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        if reduce_type == 'reduce_scatter':
            return self.mlp_tp_group.reduce_scatter(hidden_states, input_split_sizes=input_split_sizes)
        else:
            # not used
            return self.mlp_tp_group.all_reduce(hidden_states)

    def post_compact_reduce_scatter(self, hidden_states, world_len_list, comm_tensor: torch.Tensor=None):
        # for query len with large range in dp prefill, cooperate with pre_compact_allgather
        pass

    def process_weights_after_loading(self, layer: torch.nn.Module):
        if self.quant_config is not None:
            if hasattr(self.down_proj, 'smooth_scale'):
                self.down_proj.smooth_scale = Parameter(self.down_proj.smooth_scale.reshape(1, -1), requires_grad=False)
            else:
                self.down_proj.smooth_scale = None
            if hasattr(self.gate_up_proj, 'smooth_scale'):
                self.gate_up_proj.smooth_scale = Parameter(self.gate_up_proj.smooth_scale.data.to(torch.bfloat16),
                                                           requires_grad=False)
            else:
                self.gate_up_proj.smooth_scale = None
