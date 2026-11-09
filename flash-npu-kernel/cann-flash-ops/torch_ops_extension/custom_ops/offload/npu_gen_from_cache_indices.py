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
    # 输出名必须与 op_host/gen_from_cache_indices_def.cpp 的 this->Output(...) 完全一致。
    def convert_npu_gen_from_cache_indices(
        selection_topk_indices: Tensor,
        cache_status: Tensor,
        full_kv_actual_seq: Tensor,
        full_q_actual_seq: Tensor,
        *,
        selection_topk_block_size: int = 1,
        enable_topk_indices_opt: int = 0,
        full_kv_block_table: Optional[Tensor] = None,
        full_kv_block_size: int = 128,
        meta_outputs: Any = None):
        inputs = {
            "selection_topk_indices": selection_topk_indices,
            "cache_status": cache_status,
            "full_kv_actual_seq": full_kv_actual_seq,
            "full_q_actual_seq": full_q_actual_seq,
        }
        if full_kv_block_table is not None:
            inputs["full_kv_block_table"] = full_kv_block_table
        return torchair.ge.custom_op(
            "GenFromCacheIndices",
            inputs=inputs,
            attrs={
                "selection_topk_block_size": attr.Int(selection_topk_block_size),
                "enable_topk_indices_opt": attr.Int(enable_topk_indices_opt),
                "full_kv_block_size": attr.Int(full_kv_block_size),
            },
            outputs=["from_cache_indices", "selection_kv_actual_seq", "write_slot",
                     "unified_slots_in_lru_or_cache"]
        )

    register_fx_node_ge_converter(getattr(torch.ops, namespace).npu_gen_from_cache_indices.default)(convert_npu_gen_from_cache_indices)
