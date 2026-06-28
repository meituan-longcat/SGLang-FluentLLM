from typing import Optional

import torch
from torch import nn

from sglang.srt.distributed.parallel_strategy import DenseParallelStategy
from sglang.srt.env import global_server_args_dict
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.dp_attention import (
    get_dense_tp_group
)
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.utils import should_ignore_quant_layer
from sglang.srt.utils import add_prefix


class ParallelMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layout = global_server_args_dict["dense_parallel_strategy"]
        # For TP MOE, dense uses TP
        if not global_server_args_dict["enable_ep_moe"] or self.layout == DenseParallelStategy.TENSOR_PARALLEL:
            tp_group = get_dense_tp_group()
            self.gate_up_proj = MergedColumnParallelLinear(
                hidden_size,
                [intermediate_size] * 2,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("gate_up_proj", prefix),
                outside_tp_group=tp_group,
                )
            self.down_proj = RowParallelLinear(
                intermediate_size,
                hidden_size,
                bias=False,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("down_proj", prefix),
                outside_tp_group=tp_group,
            )
        else:
            self.gate_up_proj = ReplicatedLinear(
                hidden_size, intermediate_size * 2, bias=False, quant_config=quant_config, prefix=add_prefix("gate_up_proj", prefix)
            )
            self.down_proj = ReplicatedLinear(
                intermediate_size, hidden_size, bias=False, quant_config=quant_config, prefix=add_prefix("down_proj", prefix)
            )

        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )

        self.gateup_unquanted = self.gate_up_proj.quant_config is None or should_ignore_quant_layer(
            prefix=self.gate_up_proj.prefix,
            ignored_layers=getattr(self.gate_up_proj.quant_config, "ignored_layers", [])
        )
        self.down_unquanted = self.down_proj.quant_config is None or should_ignore_quant_layer(
            prefix=self.down_proj.prefix,
            ignored_layers=getattr(self.down_proj.quant_config, "ignored_layers", [])
        )
        self.act_fn = SiluAndMul()

    def forward(self, x, block_scale=None):
        if x.shape[0] == 0:
            return x
        if block_scale is not None:
            gate_up, _ = self.gate_up_proj(x, block_scale, torch.bfloat16)
        else:
            gate_up, _ = self.gate_up_proj(x)
        if self.down_unquanted:
            x = self.act_fn(gate_up)
            x, _ = self.down_proj(x)
        else:
            x, scale = self.act_fn(gate_up, True)
            x, _ = self.down_proj(x, scale)
        return x
