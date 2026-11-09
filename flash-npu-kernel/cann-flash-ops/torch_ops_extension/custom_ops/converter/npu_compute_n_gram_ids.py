# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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

@register_fx_node_ge_converter(torch.ops.custom.npu_compute_n_gram_ids.default)
def convert_npu_compute_n_gram_ids(
    oe_weights: Tensor,
    oe_mods: Tensor,
    exclusive_oe_embeder_size_sums: Tensor,
    tokens: Tensor,
    exclusive_req_len_sums: Tensor,
    oe_token_table: Tensor,
    row_indices: Tensor,
    column_starts: Tensor,
    batch_size: int,
    oe_n: int,
    oe_k: int,
    max_context_len: int,
    eos_token_id: int,
    meta_outputs: Any = None):

    return torchair.ge.custom_op(
        "ComputeNGramIds",
        inputs={"oe_weights": oe_weights,
                "oe_mods": oe_mods,
                "exclusive_oe_embeder_size_sums": exclusive_oe_embeder_size_sums,
                "tokens": tokens,
                "exclusive_req_len_sums": exclusive_req_len_sums,
                "oe_token_table": oe_token_table,
                "row_indices": row_indices,
                "column_starts": column_starts,
                },
        attrs={"batch_size": attr.Int(batch_size),
               "oe_n": attr.Int(oe_n),
               "oe_k": attr.Int(oe_k),
               "max_context_len": attr.Int(max_context_len),
               "eos_token_id": attr.Int(eos_token_id),
               },
        outputs=['oe_n_gram_ids']
    )
