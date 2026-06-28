from __future__ import annotations

import importlib.util
from typing import List, Optional

import torch
from torch.nn.parameter import Parameter

from sglang.srt.layers.quantization.base_config import (
    LinearMethodBase,
)
from sglang.srt.utils import is_npu
from sglang.srt.utils import (
    set_weight_attrs,
)

__is_npu__ = is_npu()
if __is_npu__:
    import torch_npu

has_triton_kernels = importlib.util.find_spec("triton_kernels") is not None


class UnquantizedLinearMethod(LinearMethodBase):
    """Linear method without quantization."""

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
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        return

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        enable_weight_transpsoe: bool = False
    ) -> torch.Tensor:
        cur_weight = layer.weight
        if not enable_weight_transpsoe:
            cur_weight = cur_weight.permute(1, 0)
        out = torch.matmul(x, cur_weight)
        return out if bias is None else out + bias
