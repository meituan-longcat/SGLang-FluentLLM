# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").

from typing import Optional

import torch
import torchair
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge import attr
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.custom.mlp_sparse_lightning_indexer.default)
def convert_mlp_sparse_lightning_indexer(
    query: Tensor,
    key: Tensor,
    weights: Tensor,
    *,
    cur_seq_lengths_query: Optional[Tensor] = None,
    cur_seq_lengths_key: Optional[Tensor] = None,
    block_table: Optional[Tensor] = None,
    sparse_index: Optional[Tensor] = None,
    layout_query: str = "BSND",
    layout_key: str = "PA_BSND",
    sparse_count: int = 2048,
    init_num: int = 0,
    local_num: int = 0,
    sparse_mode: int = 3,
    pre_tokens: int = 9223372036854775807,
    next_tokens: int = 9223372036854775807,
    return_value: bool = False,
    sparse_index_block_size: int = 128,
    meta_outputs: TensorSpec = None,
):
    return torchair.ge.custom_op(
        "MlpSparseLightningIndexer",
        inputs={
            "query": query,
            "key": key,
            "weights": weights,
            "cur_seq_lengths_query": cur_seq_lengths_query,
            "cur_seq_lengths_key": cur_seq_lengths_key,
            "block_table": block_table,
            "sparse_index": sparse_index,
        },
        attrs={
            "layout_query": attr.Str(layout_query),
            "layout_key": attr.Str(layout_key),
            "sparse_count": attr.Int(sparse_count),
            "init_num": attr.Int(init_num),
            "local_num": attr.Int(local_num),
            "sparse_mode": attr.Int(sparse_mode),
            "pre_tokens": attr.Int(pre_tokens),
            "next_tokens": attr.Int(next_tokens),
            "return_value": attr.Bool(return_value),
            "sparse_index_block_size": attr.Int(sparse_index_block_size),
        },
        outputs=["sparse_indices", "sparse_values"],
    )
