# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)
import torch
import torch_npu
import torchair
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.library import Library, impl
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType, auto_convert_to_tensor, TensorType
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, \
    BOOL, Support
from torchair._ge_concrete_graph.utils import dtype_promote
from torchair.ge import attr, Tensor, DataType


@auto_convert_to_tensor(
    [False, False, False, False, False, False, False, False, False, False],
    [False, False, False, False, False, False, False, False, False, False])
def GatherSelectionKvCache(selection_k_rope: Tensor,
                           selection_kv_cache: Tensor,
                           selection_kv_block_table: Tensor,
                           selection_kv_block_status: Tensor,
                           selection_topk_indices: Tensor,
                           full_k_rope: Tensor,
                           full_kv_cache: Tensor,
                           full_kv_block_table: Tensor,
                           full_kv_actual_seq: Tensor,
                           full_q_actual_seq: Tensor,
                           *,
                           selection_topk_block_size: int = 64,
                           dependenices=[],
                           node_name=None):

    # process input
    inputs = {
        "selection_k_rope": selection_k_rope,
        "selection_kv_cache": selection_kv_cache,
        "selection_kv_block_table": selection_kv_block_table,
        "selection_kv_block_status": selection_kv_block_status,
        "selection_topk_indices": selection_topk_indices,
        "full_k_rope": full_k_rope,
        "full_kv_cache": full_kv_cache,
        "full_kv_block_table": full_kv_block_table,
        "full_kv_actual_seq": full_kv_actual_seq,
        "full_q_actual_seq": full_q_actual_seq
    }

    # process attrs
    attrs = {
        "selection_topk_block_size": attr.Int(selection_topk_block_size)
    }

    # process outputs
    outputs = [
        "selection_k_rope",
        "selection_kv_cache",
        "selection_kv_block_table",
        "selection_kv_block_status",
        "selection_kv_actual_seq"
    ]

    return torchair.ge.custom_op("GatherSelectionKvCache", inputs=inputs, attrs=attrs, outputs=outputs)


# 根据命名空间获取原地算子注册的Library，如您的算子通过op-plugin注册，则其命名空间为npu，否则使用您注册算子时的命名空间
# （您通过torch.ops.xxx.{your_op_name}调用算子时的xxx就是该算子的命名空间）
m = Library("custom", "FRAGMENT")


@impl(m, "npu_gather_selection_kv_cache", "Functionalize")
def custom_npu_gather_selection_kv_cache_func(
    selection_k_rope: Tensor, 
    selection_kv_cache: Tensor, 
    selection_kv_block_table: Tensor, 
    selection_kv_block_status: Tensor,
    selection_topk_indices: Tensor, 
    full_k_rope: Tensor, 
    full_kv_cache: Tensor, 
    full_kv_block_table: Tensor, 
    full_kv_actual_seq: Tensor,
    full_q_actual_seq: Tensor, 
    *, 
    selection_topk_block_size: int = 64):

    (
        selection_kv_actual_seq_out, 
        selection_k_rope_inplace, 
        selection_kv_cache_inplace, 
        selection_kv_block_table_inplace, 
        selection_kv_block_status_inplace
    ) = torch.ops.custom.npu_gather_selection_kv_cache_functional(
        selection_k_rope,
        selection_kv_cache,
        selection_kv_block_table,
        selection_kv_block_status,
        selection_topk_indices,
        full_k_rope,
        full_kv_cache,
        full_kv_block_table,
        full_kv_actual_seq,
        full_q_actual_seq,
        selection_topk_block_size=selection_topk_block_size
    )

    selection_k_rope.copy_(selection_k_rope_inplace)
    selection_kv_cache.copy_(selection_kv_cache_inplace)
    selection_kv_block_table.copy_(selection_kv_block_table_inplace)
    selection_kv_block_status.copy_(selection_kv_block_status_inplace)

    return selection_kv_actual_seq_out


# 为自定义算子注册converter，用于torch.compile场景成图

# 注意：meta_outputs形参名为固定写法，若写错会影响ge节点的输出dtype与shape推导
@register_fx_node_ge_converter(torch.ops.custom.npu_gather_selection_kv_cache.default)
def convert_npu_gather_selection_kv_cache(
    selection_k_rope: Tensor, 
    selection_kv_cache: Tensor, 
    selection_kv_block_table: Tensor, 
    selection_kv_block_status: Tensor,
    selection_topk_indices: Tensor, 
    full_k_rope: Tensor, 
    full_kv_cache: Tensor, 
    full_kv_block_table: Tensor, 
    full_kv_actual_seq: Tensor,
    full_q_actual_seq: Tensor, 
    *, 
    selection_topk_block_size: int = 64,
    meta_outputs: Any = None):

    selection_k_rope_copy = ge.TensorMove(selection_k_rope)
    selection_kv_cache_copy = ge.TensorMove(selection_kv_cache)
    selection_kv_block_table_copy = ge.TensorMove(selection_kv_block_table)
    selection_kv_block_status_copy = ge.TensorMove(selection_kv_block_status)

    return torchair.ge.custom_op(
        "GatherSelectionKvCache",
        inputs={
                "selection_k_rope": selection_k_rope_copy,
                "selection_kv_cache": selection_kv_cache_copy,
                "selection_kv_block_table": selection_kv_block_table_copy,
                "selection_kv_block_status": selection_kv_block_status_copy,
                "selection_topk_indices": selection_topk_indices,
                "full_k_rope": full_k_rope,
                "full_kv_cache": full_kv_cache,
                "full_kv_block_table": full_kv_block_table,
                "full_kv_actual_seq": full_kv_actual_seq,
                "full_q_actual_seq": full_q_actual_seq
                }, 
        attrs={
                "selection_topk_block_size": attr.Int(selection_topk_block_size)
                }, 
        outputs=[
                "selection_k_rope",
                "selection_kv_cache",
                "selection_kv_block_table",
                "selection_kv_block_status",
                "selection_kv_actual_seq"
                ]
    )


#注意：meta_outputs形参名为固定写法，若写错会影响ge节点的输出dtype与shape推导
@register_fx_node_ge_converter(torch.ops.custom.npu_gather_selection_kv_cache_functional.default)
def convert_npu_gather_selection_kv_cache_functional(
    selection_k_rope: Tensor, 
    selection_kv_cache: Tensor, 
    selection_kv_block_table: Tensor, 
    selection_kv_block_status: Tensor,
    selection_topk_indices: Tensor, 
    full_k_rope: Tensor, 
    full_kv_cache: Tensor, 
    full_kv_block_table: Tensor, 
    full_kv_actual_seq: Tensor,
    full_q_actual_seq: Tensor, 
    *, 
    selection_topk_block_size: int = 64,
    meta_outputs: Any = None):

    selection_k_rope_copy = ge.TensorMove(selection_k_rope)
    selection_kv_cache_copy = ge.TensorMove(selection_kv_cache)
    selection_kv_block_table_copy = ge.TensorMove(selection_kv_block_table)
    selection_kv_block_status_copy = ge.TensorMove(selection_kv_block_status)

    (
        selection_k_rope_out,
        selection_kv_cache_out,
        selection_kv_block_table_out,
        selection_kv_block_status_out,
        selection_kv_actual_seq
    ) = GatherSelectionKvCache(
        selection_k_rope_copy,
        selection_kv_cache_copy,
        selection_kv_block_table_copy,
        selection_kv_block_status_copy,
        selection_topk_indices,
        full_k_rope,
        full_kv_cache,
        full_kv_block_table,
        full_kv_actual_seq,
        full_q_actual_seq,
        selection_topk_block_size=selection_topk_block_size
    )

    return (
        selection_kv_actual_seq,
        selection_k_rope_out,
        selection_kv_cache_out,
        selection_kv_block_table_out,
        selection_kv_block_status_out
    )