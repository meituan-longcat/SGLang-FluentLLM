# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.

from typing import Optional

import torch
import torchair
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge import attr
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.custom.npu_moe_gating_top_k_expert_map.default)
def convert_npu_moe_gating_top_k_expert_map(
    x: Tensor,
    expert_map: Tensor,
    bias: Optional[Tensor] = None,
    *,
    k: int = 1,
    k_group: int = 1,
    group_count: int = 1,
    group_select_mode: int = 0,
    renorm: int = 0,
    norm_type: int = 0,
    out_flag: bool = False,
    routed_scaling_factor: float = 1.0,
    eps: float = 1e-20,
    meta_outputs: TensorSpec = None,
):
    return torchair.ge.custom_op(
        "MoeGatingTopKExpertMap",
        inputs={"x": x, "bias": bias, "expert_map": expert_map},
        attrs={
            "k": attr.Int(k),
            "k_group": attr.Int(k_group),
            "group_count": attr.Int(group_count),
            "group_select_mode": attr.Int(group_select_mode),
            "renorm": attr.Int(renorm),
            "norm_type": attr.Int(norm_type),
            "out_flag": attr.Bool(out_flag),
            "routed_scaling_factor": attr.Float(routed_scaling_factor),
            "eps": attr.Float(eps),
        },
        outputs=["y", "expert_idx", "out"],
    )
