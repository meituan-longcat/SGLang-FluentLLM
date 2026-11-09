# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.

from typing import Optional

import torch
import torchair
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge import attr
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.custom.npu_moe_distribute_combine_v2.default)
def convert_npu_moe_distribute_combine_v2(
    expand_x: Tensor,
    expert_ids: Tensor,
    assist_info_for_combine: Tensor,
    ep_send_counts: Tensor,
    expert_scales: Tensor,
    group_ep: str,
    ep_world_size: int,
    ep_rank_id: int,
    moe_expert_num: int,
    *,
    tp_send_counts: Optional[Tensor] = None,
    x_active_mask: Optional[Tensor] = None,
    expand_scales: Optional[Tensor] = None,
    shared_expert_x: Optional[Tensor] = None,
    elastic_info: Optional[Tensor] = None,
    ori_x: Optional[Tensor] = None,
    const_expert_alpha_1: Optional[Tensor] = None,
    const_expert_alpha_2: Optional[Tensor] = None,
    const_expert_v: Optional[Tensor] = None,
    group_tp: str = "",
    tp_world_size: int = 0,
    tp_rank_id: int = 0,
    expert_shard_type: int = 0,
    shared_expert_num: int = 1,
    shared_expert_rank_num: int = 0,
    global_bs: int = 0,
    comm_quant_mode: int = 0,
    comm_alg: str = "",
    zero_expert_num: int = 0,
    copy_expert_num: int = 0,
    const_expert_num: int = 0,
    meta_outputs: TensorSpec = None,
):
    return torchair.ge.custom_op(
        "MoeDistributeCombineV2",
        inputs={
            "expand_x": expand_x,
            "expert_ids": expert_ids,
            "assist_info_for_combine": assist_info_for_combine,
            "ep_send_counts": ep_send_counts,
            "expert_scales": expert_scales,
            "tp_send_counts": tp_send_counts,
            "x_active_mask": x_active_mask,
            "activation_scale": None,
            "weight_scale": None,
            "group_list": None,
            "expand_scales": expand_scales,
            "shared_expert_x": shared_expert_x,
            "elastic_info": elastic_info,
            "ori_x": ori_x,
            "const_expert_alpha_1": const_expert_alpha_1,
            "const_expert_alpha_2": const_expert_alpha_2,
            "const_expert_v": const_expert_v,
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
            "global_bs": attr.Int(global_bs),
            "out_dtype": attr.Int(0),
            "comm_quant_mode": attr.Int(comm_quant_mode),
            "group_list_type": attr.Int(0),
            "comm_alg": attr.Str(comm_alg),
            "zero_expert_num": attr.Int(zero_expert_num),
            "copy_expert_num": attr.Int(copy_expert_num),
            "const_expert_num": attr.Int(const_expert_num),
        },
        outputs=["x"],
    )
