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
    # 输出名必须与 op_host/gather_by_cache_indices_def.cpp 的 this->Output(...) 完全一致。
    def convert_npu_gather_by_cache_indices(
        cache_k_rope: Tensor,
        cache_kv_cache: Tensor,
        from_cache_indices: Tensor,
        selection_topk_indices: Tensor,
        full_k_rope: Tensor,
        full_kv_cache: Tensor,
        full_kv_block_table: Tensor,
        selection_kv_block_table: Tensor,
        write_slot: Tensor,
        actual_seq_lengths_kv: Tensor,
        actual_seq_qlen: Tensor,
        *,
        selection_topk_block_size: int = 1,
        selection_kv_block_num: int = 0,
        selection_kv_block_size: int = 128,
        enable_slots_contiguous: int = 0,
        meta_outputs: Any = None):
        return torchair.ge.custom_op(
            "GatherByCacheIndices",
            inputs={
                "cache_k_rope": cache_k_rope,
                "cache_kv_cache": cache_kv_cache,
                "from_cache_indices": from_cache_indices,
                "selection_topk_indices": selection_topk_indices,
                "full_k_rope": full_k_rope,
                "full_kv_cache": full_kv_cache,
                "full_kv_block_table": full_kv_block_table,
                "selection_kv_block_table": selection_kv_block_table,
                "write_slot": write_slot,
                "actual_seq_lengths_kv": actual_seq_lengths_kv,
                "actual_seq_qlen": actual_seq_qlen,
            },
            attrs={
                "selection_topk_block_size": attr.Int(selection_topk_block_size),
                "selection_kv_block_num": attr.Int(selection_kv_block_num),
                "selection_kv_block_size": attr.Int(selection_kv_block_size),
                "enable_slots_contiguous": attr.Int(enable_slots_contiguous),
            },
            outputs=["selection_k_rope", "selection_kv_cache", "cur_kv_write_slots"]
        )

    register_fx_node_ge_converter(getattr(torch.ops, namespace).npu_gather_by_cache_indices.default)(convert_npu_gather_by_cache_indices)
