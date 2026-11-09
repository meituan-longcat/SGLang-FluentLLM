# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, \
    BOOL, Support
from torchair._ge_concrete_graph.utils import dtype_promote
from torchair.ge import attr


def register(namespace):
    # 为自定义算子注册 converter，用于 torch.compile 图模式成图。
    # 输出名必须与 op_host/update_cur_kv_to_selection_def.cpp 的 this->Output(...) 完全一致：
    #   selection_k_nope / selection_k_rope 为 inplace 输出（与输入同名，GE 写回），
    #   kernel_done 为图返回值标记。
    def convert_npu_update_cur_kv_to_selection(
        cur_k_nope: Tensor,
        cur_k_rope: Tensor,
        cur_kv_write_slots: Tensor,
        selection_k_nope: Tensor,
        selection_k_rope: Tensor,
        selection_kv_block_table: Tensor,
        *,
        meta_outputs: Any = None):
        selection_k_nope_out, selection_k_rope_out, _ = torchair.ge.custom_op(
            "UpdateCurKvToSelection",
            inputs={
                "cur_k_nope": cur_k_nope,
                "cur_k_rope": cur_k_rope,
                "cur_kv_write_slots": cur_kv_write_slots,
                "selection_k_nope": selection_k_nope,
                "selection_k_rope": selection_k_rope,
                "selection_kv_block_table": selection_kv_block_table,
            },
            attrs={},
            outputs=["selection_k_nope", "selection_k_rope", "kernel_done"]
        )
        # Match the ordinary schema's single completion Tensor (not the three
        # physical GE outputs), with the same ordering semantics as the native op.
        return ge.Identity(ge.Const([0], dtype=DataType.DT_INT32),
                           dependencies=[selection_k_nope_out])


    # 根据命名空间获取原地算子注册的Library（与 ops_def_registration.cpp 的 TORCH_LIBRARY(fluent_sfa1, m) 对应）
    m = Library(namespace, "FRAGMENT")


    @impl(m, "npu_update_cur_kv_to_selection", "Functionalize")
    def custom_npu_update_cur_kv_to_selection_func(
        cur_k_nope: Tensor,
        cur_k_rope: Tensor,
        cur_kv_write_slots: Tensor,
        selection_k_nope: Tensor,
        selection_k_rope: Tensor,
        selection_kv_block_table: Tensor):

        (
            kernel_done,
            selection_k_nope_inplace,
            selection_k_rope_inplace
        ) = getattr(torch.ops, namespace).npu_update_cur_kv_to_selection_functional(
            cur_k_nope,
            cur_k_rope,
            cur_kv_write_slots,
            selection_k_nope,
            selection_k_rope,
            selection_kv_block_table
        )

        selection_k_nope.copy_(selection_k_nope_inplace)
        selection_k_rope.copy_(selection_k_rope_inplace)

        return kernel_done


    # 为 functional 变体注册 converter（供 Functionalize impl 成图使用）
    # 注意：
    # 1. torchair.ge.custom_op 按 outputs 列表顺序返回元组，
    #    此处必须手动重排为与 torch functional 签名一致的返回顺序 (kernel_done, selection_k_nope, selection_k_rope)
    # 2. kernel_done 为图返回值标记，kernel 不写其值。
    #    直调路径由 C++ at::zeros 预置 0；GE 图路径输出内存未初始化，故 converter 直接用 GE 常量 0 构造，
    #    保证 torch.compile 链路返回值恒 0，与 op_host 契约/ST 断言一致。
    def convert_npu_update_cur_kv_to_selection_functional(
        cur_k_nope: Tensor,
        cur_k_rope: Tensor,
        cur_kv_write_slots: Tensor,
        selection_k_nope: Tensor,
        selection_k_rope: Tensor,
        selection_kv_block_table: Tensor,
        *,
        meta_outputs: Any = None):
        selection_k_nope_out, selection_k_rope_out, _ = torchair.ge.custom_op(
            "UpdateCurKvToSelection",
            inputs={
                "cur_k_nope": cur_k_nope,
                "cur_k_rope": cur_k_rope,
                "cur_kv_write_slots": cur_kv_write_slots,
                "selection_k_nope": selection_k_nope,
                "selection_k_rope": selection_k_rope,
                "selection_kv_block_table": selection_kv_block_table,
            },
            attrs={},
            outputs=["selection_k_nope", "selection_k_rope", "kernel_done"]
        )
        # Separate reviewed dependency correction: the completion value must depend
        # on the actual updating op, not on an independent constant. Both cache
        # outputs have the same producer; one control edge orders both writes.
        kernel_done = ge.Identity(
            ge.Const([0], dtype=DataType.DT_INT32),
            dependencies=[selection_k_nope_out],
        )
        return kernel_done, selection_k_nope_out, selection_k_rope_out

    register_fx_node_ge_converter(getattr(torch.ops, namespace).npu_update_cur_kv_to_selection.default)(convert_npu_update_cur_kv_to_selection)
    register_fx_node_ge_converter(getattr(torch.ops, namespace).npu_update_cur_kv_to_selection_functional.default)(convert_npu_update_cur_kv_to_selection_functional)
    return m
