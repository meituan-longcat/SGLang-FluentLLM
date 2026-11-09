from __future__ import annotations

from functools import partial

import torch
from torch.nn import Module

from sglang.srt.layers.moe.layouts.load_functions import load_per_channel_weight_scale
from sglang.srt.layers.moe.layouts.common import MoELayout
from sglang.srt.layers.quantization import FusedMoeWeightScaleSupported
from sglang.srt.distributed import get_moe_tensor_parallel_rank
from sglang.srt.utils import set_weight_attrs


class PerChannelQuantLayout(MoELayout):
    def __init__(self, quant_config):
        super().__init__()
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: Module,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        super().create_weights(
            layer,
            num_local_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )
        self.create_weight_scales(
            layer,
            num_local_experts,
            hidden_size,
            intermediate_size_per_partition,
            **extra_weight_attrs,
        )

    def create_weight_scales(
        self,
        layer: Module,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        **extra_weight_attrs,
    ):
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(num_local_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_local_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        weight_loader = partial(load_per_channel_weight_scale, tp_rank=get_moe_tensor_parallel_rank(), do_transpose=False)  # FIXME
        set_weight_attrs(w13_weight_scale, {"weight_loader": weight_loader})
        set_weight_attrs(w2_weight_scale, {"weight_loader": weight_loader})

        w13_input_scale = None
        w2_input_scale = None
        layer.register_parameter("w13_input_scale", w13_input_scale)
        layer.register_parameter("w2_input_scale", w2_input_scale)

        assert not self.quant_config.is_checkpoint_fp8_serialized

        extra_weight_attrs.update({"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value})
