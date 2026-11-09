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
    # update_cache 内核为真·functional：读 input、写独立 output GM（Output 名 *_out，与 Input 不同 →
    # GE 分配独立新 GM，绝不 ref-alias/原地写 input）。故 converter 直接把 input 喂 GE 节点、返回 4 个
    # 全新输出即可——不需要 ge.TensorMove，也不需要 Functionalize 胶水（算子本就非原地，无回写语义）。
    # 输出名必须与 op_host/update_cache_def.cpp 的 this->Output(...) 完全一致（*_out）。
    def convert_npu_update_cache(
        selection_topk_indices: Tensor,
        selection_k_rope: Tensor,
        selection_kv_cache: Tensor,
        cache_k_rope: Tensor,
        cache_kv_cache: Tensor,
        cache_status: Tensor,
        cache_age: Tensor,
        selection_kv_block_table: Tensor,
        write_slot: Tensor,
        step: Tensor,
        *,
        meta_outputs: Any = None):
        return torchair.ge.custom_op(
            "UpdateCache",
            inputs={
                "selection_topk_indices": selection_topk_indices,
                "selection_k_rope": selection_k_rope,
                "selection_kv_cache": selection_kv_cache,
                "cache_k_rope": cache_k_rope,
                "cache_kv_cache": cache_kv_cache,
                "cache_status": cache_status,
                "cache_age": cache_age,
                "selection_kv_block_table": selection_kv_block_table,
                "write_slot": write_slot,
                "step": step,
            },
            outputs=["cache_k_rope_out", "cache_kv_cache_out", "cache_status_out", "cache_age_out"]
        )

    register_fx_node_ge_converter(getattr(torch.ops, namespace).npu_update_cache.default)(convert_npu_update_cache)
