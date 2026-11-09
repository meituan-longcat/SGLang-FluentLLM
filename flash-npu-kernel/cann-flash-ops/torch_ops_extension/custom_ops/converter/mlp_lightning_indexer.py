# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.

from typing import (
    Any,
    Callable,
    ContextManager,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import torch
import torch_npu
import torchair
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, BF16, F64, I32, I16, I64, \
I8, U8, BOOL, Support
from torchair._ge_concrete_graph.utils import dtype_promote
from torchair.ge import attr


@register_fx_node_ge_converter(torch.ops.custom.mlp_lightning_indexer.default)
def convert_mlp_lightning_indexer(
        query: Tensor,
        key: Tensor,
        weights: Tensor,
        *,
        cur_seq_lengths_query: Tensor = None,
        cur_seq_lengths_key: Tensor = None,
        block_table: Tensor = None,
        init_tensor: Tensor = None,
        local_tensor: Tensor = None,
        layout_query: str = "BSND",
        layout_key: str = "PA_BSND",
        sparse_count: int = 2048,
        kv_block_len: int = 1,
        q_block_len: int = 1,
        init_num: int = 0,
        local_num: int = 0,
        sparse_mode: int = 3,
        pre_tokens: int = 9223372036854775807,
        next_tokens: int = 9223372036854775807,
        return_value: bool = False,
        meta_outputs: TensorSpec = None):
        return torchair.ge.custom_op(
        "MlpLightningIndexer",
        inputs={"query": query,
                "key": key,
                "weights": weights,
                "cur_seq_lengths_query": cur_seq_lengths_query,
                "cur_seq_lengths_key": cur_seq_lengths_key,
                "block_table": block_table,
                "init_tensor": init_tensor,
                "local_tensor": local_tensor,
        },
        attrs={"layout_query": attr.Str(layout_query),
                "layout_key": attr.Str(layout_key),
                "sparse_count": attr.Int(sparse_count),
                "block_len": attr.Int(kv_block_len),
                "q_block_len": attr.Int(q_block_len),
                "init_num": attr.Int(init_num),
                "local_num": attr.Int(local_num),
                "sparse_mode": attr.Int(sparse_mode),
                "pre_tokens": attr.Int(pre_tokens),
                "next_tokens": attr.Int(next_tokens),
                "return_value": attr.Bool(return_value),
        },
        outputs=["sparse_indices",
                "sparse_values",]
        )
