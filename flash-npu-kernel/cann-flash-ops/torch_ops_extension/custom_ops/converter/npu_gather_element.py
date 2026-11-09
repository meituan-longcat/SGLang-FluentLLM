# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
import torchair
from torchair._ge_concrete_graph.ge_converter.converter_utils import display_ge_type
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair._ge_concrete_graph.utils import dtype_promote
from torchair.ge import attr
from torchair.ge._ge_graph import DataType, Tensor, TensorSpec


def check_support_dtype(self: Tensor):
    supported_dtypes = {
        DataType.DT_FLOAT16,
        DataType.DT_BF16,
        DataType.DT_FLOAT,
        DataType.DT_INT32,
    }
    if self.dtype not in supported_dtypes:
        raise RuntimeError(
            "torch.ops.custom.npu_gather_element.default expects input dtype "
            "DT_FLOAT16, DT_BF16, DT_FLOAT, or DT_INT32, but got "
            f"{display_ge_type(self.dtype)}"
        )


@register_fx_node_ge_converter(torch.ops.custom.npu_gather_element.default)
def convert_npu_gather_element(
    self: Tensor,
    dim: int,
    index: Tensor,
    *,
    meta_outputs: TensorSpec = None,
):
    check_support_dtype(self)
    index = dtype_promote(index, target_dtype=DataType.DT_INT32)
    return torchair.ge.custom_op(
        "GatherElementsV2",
        inputs={"x": self, "index": index},
        attrs={"dim": attr.Int(dim)},
        outputs=["y"],
    )
