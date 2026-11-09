# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.

from typing import Optional

import torch
import torchair
from torchair.ge import attr
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import DataType, Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.custom.flat_quant.default)
def convert_flat_quant(
    x: Tensor,
    kronecker_p1: Tensor,
    kronecker_p2: Tensor,
    *,
    clip_ratio: float = 1.0,
    dst_dtype: int = 16,
    dst_type_max: float = 0.0,
    group_list: Optional[Tensor] = None,
    group_list_type: int = 0,
    clear_padding: bool = False,
    meta_outputs: TensorSpec = None,
):
    inputs = {"x": x, "kronecker_p1": kronecker_p1, "kronecker_p2": kronecker_p2}
    if group_list is not None:
        inputs["group_list"] = group_list

    ge_dst_dtype = 29 if dst_dtype == 16 else 40
    out, quant_scale = torchair.ge.custom_op(
        "FlatQuant",
        inputs=inputs,
        attrs={
            "clip_ratio": attr.Float(clip_ratio),
            "dst_dtype": attr.Int(ge_dst_dtype),
            "dst_type_max": attr.Float(dst_type_max),
            "group_list_type": attr.Int(group_list_type),
            "clear_padding": attr.Bool(clear_padding),
        },
        outputs=["out", "quant_scale"],
    )
    if dst_dtype == 16:
        shape = ge.Div(ge.Shape(out), ge.Const([1, 1, 8], dtype=DataType.DT_INT32))
        packed = ge.Bitcast(
            ge.Reshape(
                out,
                ge.ConcatV2([shape, ge.Const([8], dtype=DataType.DT_INT32)], concat_dim=0, N=2),
            ),
            type=DataType.DT_INT32,
        )
        return ge.Reshape(packed, shape), quant_scale

    shape = ge.Div(ge.Shape(out), ge.Const([1, 2], dtype=DataType.DT_INT32))
    packed = ge.Bitcast(
        ge.Reshape(
            out,
            ge.ConcatV2([shape, ge.Const([2], dtype=DataType.DT_INT32)], concat_dim=0, N=2),
        ),
        type=DataType.DT_UINT8,
    )
    return ge.Reshape(packed, shape), ge.Bitcast(quant_scale, type=DataType.DT_UINT8)
