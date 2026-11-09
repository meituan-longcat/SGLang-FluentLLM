from __future__ import annotations

import torch
from torch.nn import Module

from sglang.srt.layers.moe.layouts.w8a8 import PerChannelQuantLayout
from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8Config


class Fp8MoEPerChannelQuantLayout(PerChannelQuantLayout):
    def __init__(self, quant_config: W8A8Int8Config):
        super().__init__(quant_config)

    def create_weights(
        self,
        layer: Module,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        assert not self.quant_config.is_checkpoint_fp8_serialized
        params_dtype = torch.float8_e4m3fn

        super().create_weights(
            layer,
            num_local_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )
