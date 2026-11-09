# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
import torchair
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge import attr
from torchair.ge._ge_graph import DataType, Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.custom.mlp_split_swiglu.default)
def convert_mlp_split_swiglu(
    x: Tensor,
    group_list: Tensor,
    *,
    local_exp_start: int = 0,
    local_exp_end: int = -1,
    meta_outputs: TensorSpec = None,
):
    group_list = ge.Cast(group_list, dst_type=DataType.DT_INT32)
    return torchair.ge.custom_op(
        "MlpSplitSwiglu",
        inputs={
            "x": x,
            "group_list": group_list,
        },
        attrs={
            "local_expert_start": attr.Int(local_exp_start),
            "local_expert_end": attr.Int(local_exp_end),
        },
        outputs=["out"],
    )
