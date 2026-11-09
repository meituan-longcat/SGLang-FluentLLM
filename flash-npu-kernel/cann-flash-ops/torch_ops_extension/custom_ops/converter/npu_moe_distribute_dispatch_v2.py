# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.

from typing import Optional

import torch
import torchair
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair._ge_concrete_graph.ge_converter.converter_utils import ge_dtype_to_ge_proto_dtype
from torchair.ge import attr
from torchair.ge._ge_graph import (
    DataType,
    Tensor,
    TensorSpec,
    torch_dtype_value_to_ge_proto_type,
    torch_dtype_value_to_ge_type,
)


@register_fx_node_ge_converter(torch.ops.custom.npu_moe_distribute_dispatch_v2.default)
def convert_npu_moe_distribute_dispatch_v2(
    x: Tensor,
    expert_ids: Tensor,
    group_ep: str,
    ep_world_size: int,
    ep_rank_id: int,
    moe_expert_num: int,
    *,
    scales: Optional[Tensor] = None,
    x_active_mask: Optional[Tensor] = None,
    expert_scales: Optional[Tensor] = None,
    elastic_info: Optional[Tensor] = None,
    group_tp: str = "",
    tp_world_size: int = 0,
    tp_rank_id: int = 0,
    expert_shard_type: int = 0,
    shared_expert_num: int = 1,
    shared_expert_rank_num: int = 0,
    quant_mode: int = 0,
    global_bs: int = 0,
    expert_token_nums_type: int = 1,
    comm_alg: str = "",
    zero_expert_num: int = 0,
    copy_expert_num: int = 0,
    const_expert_num: int = 0,
    y_dtype: Optional[int] = None,
    x_dtype: Optional[int] = None,
    scales_dtype: Optional[int] = None,
    meta_outputs: TensorSpec = None,
):
    if x_dtype is not None:
        x = ge.Bitcast(x, type=torch_dtype_value_to_ge_type(x_dtype))
        x.desc.dtype = torch_dtype_value_to_ge_proto_type(x_dtype)
    if scales_dtype is not None:
        scales = ge.Bitcast(scales, type=torch_dtype_value_to_ge_type(scales_dtype))
        scales.desc.dtype = torch_dtype_value_to_ge_proto_type(scales_dtype)

    expand_x_dtype = DataType.DT_INT8
    if quant_mode == 0:
        expand_x_dtype = x.dtype
    elif y_dtype is not None:
        expand_x_dtype = torch_dtype_value_to_ge_type(y_dtype)

    outputs = torchair.ge.custom_op(
        "MoeDistributeDispatchV2",
        inputs={
            "x": x,
            "expert_ids": expert_ids,
            "scales": scales,
            "x_active_mask": x_active_mask,
            "expert_scales": expert_scales,
            "elastic_info": elastic_info,
        },
        attrs={
            "group_ep": attr.Str(group_ep),
            "ep_world_size": attr.Int(ep_world_size),
            "ep_rank_id": attr.Int(ep_rank_id),
            "moe_expert_num": attr.Int(moe_expert_num),
            "group_tp": attr.Str(group_tp),
            "tp_world_size": attr.Int(tp_world_size),
            "tp_rank_id": attr.Int(tp_rank_id),
            "expert_shard_type": attr.Int(expert_shard_type),
            "shared_expert_num": attr.Int(shared_expert_num),
            "shared_expert_rank_num": attr.Int(shared_expert_rank_num),
            "quant_mode": attr.Int(quant_mode),
            "global_bs": attr.Int(global_bs),
            "expert_token_nums_type": attr.Int(expert_token_nums_type),
            "comm_alg": attr.Str(comm_alg),
            "zero_expert_num": attr.Int(zero_expert_num),
            "copy_expert_num": attr.Int(copy_expert_num),
            "const_expert_num": attr.Int(const_expert_num),
            "y_dtype": attr.Int(expand_x_dtype),
        },
        outputs=[
            "expand_x",
            "dynamic_scales",
            "assist_info_for_combine",
            "expert_token_nums",
            "ep_recv_count",
            "tp_recv_count",
            "expand_scales",
        ],
    )
    outputs[0].desc.dtype = ge_dtype_to_ge_proto_dtype(expand_x_dtype)
    dynamic_scales_dtype = DataType.DT_FLOAT
    if quant_mode == 0 and x.dtype not in (DataType.DT_FLOAT16, DataType.DT_BF16) and scales is not None:
        dynamic_scales_dtype = scales.dtype
    elif quant_mode == 4:
        dynamic_scales_dtype = DataType.DT_FLOAT8_E8M0
    outputs[1].desc.dtype = ge_dtype_to_ge_proto_dtype(dynamic_scales_dtype)
    return outputs
