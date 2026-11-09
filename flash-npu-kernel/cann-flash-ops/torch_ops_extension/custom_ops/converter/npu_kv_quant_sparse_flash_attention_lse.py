# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Any

import torch
import torchair as tng
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge import attr
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.custom.npu_kv_quant_sparse_flash_attention_lse.default)
def convert_npu_kv_quant_sparse_flash_attention_lse(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    sparse_indices: Tensor,
    key_dequant_scale: Tensor = None,
    value_dequant_scale: Tensor = None,
    block_table: Tensor = None,
    actual_seq_lengths_query: Tensor = None,
    actual_seq_lengths_kv: Tensor = None,
    scale_value: float = 1.0,
    key_quant_mode: int = 2,
    value_quant_mode: int = 2,
    sparse_block_size: int = 1,
    layout_query: str = "BSND",
    layout_kv: str = "BSND",
    sparse_mode: int = 3,
    pre_tokens: int = 9223372036854775807,
    next_tokens: int = 9223372036854775807,
    attention_mode: int = 2,
    quant_scale_repo_mode: int = 1,
    tile_size: int = 128,
    rope_head_dim: int = 64,
    return_softmax_lse: bool = True,
    softmax_lse_tnd: bool = False,
    meta_outputs: TensorSpec = None,
):
    # CANN 9 graph execution rejects zero-sized custom-op outputs in
    # always-zero-copy mode. The schema always exposes the two LSE outputs, so
    # materialize them in graph mode and let callers ignore them when not needed.
    graph_return_softmax_lse = True
    return tng.ge.custom_op(
        "KvQuantSparseFlashAttentionLse",
        inputs={
            "query": query,
            "key": key,
            "value": value,
            "sparse_indices": sparse_indices,
            "key_dequant_scale": key_dequant_scale,
            "value_dequant_scale": value_dequant_scale,
            "block_table": block_table,
            "actual_seq_lengths_query": actual_seq_lengths_query,
            "actual_seq_lengths_kv": actual_seq_lengths_kv,
        },
        attrs={
            "scale_value": attr.Float(scale_value),
            "key_quant_mode": attr.Int(key_quant_mode),
            "value_quant_mode": attr.Int(value_quant_mode),
            "sparse_block_size": attr.Int(sparse_block_size),
            "layout_query": attr.Str(layout_query),
            "layout_kv": attr.Str(layout_kv),
            "sparse_mode": attr.Int(sparse_mode),
            "pre_tokens": attr.Int(pre_tokens),
            "next_tokens": attr.Int(next_tokens),
            "attention_mode": attr.Int(attention_mode),
            "quant_scale_repo_mode": attr.Int(quant_scale_repo_mode),
            "tile_size": attr.Int(tile_size),
            "rope_head_dim": attr.Int(rope_head_dim),
            "return_softmax_lse": attr.Bool(graph_return_softmax_lse),
            "softmax_lse_tnd": attr.Bool(softmax_lse_tnd),
        },
        outputs=["attention_out", "softmax_max", "softmax_sum"],
    )
