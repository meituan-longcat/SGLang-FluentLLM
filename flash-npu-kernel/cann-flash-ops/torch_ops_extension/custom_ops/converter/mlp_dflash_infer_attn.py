# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.

from typing import Optional

import torch
import torchair
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge import attr
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.custom.mlp_dflash_infer_attn.default)
def convert_mlp_dflash_infer_attn(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    actual_seq_lengths: Optional[Tensor] = None,
    actual_seq_lengths_kv: Optional[Tensor] = None,
    block_table: Optional[Tensor] = None,
    num_heads: int = 1,
    scale: float = 1.0,
    pre_tokens: int = 2147483647,
    next_tokens: int = 16,
    input_layout: str = "TND",
    num_key_value_heads: int = 0,
    sparse_mode: int = 0,
    inner_precise: int = 0,
    block_size: int = 0,
    softmax_lse_flag: bool = False,
    meta_outputs: TensorSpec = None,
):
    del sparse_mode, inner_precise, meta_outputs
    return torchair.ge.custom_op(
        "MlpDflashInferAttn",
        inputs={
            "query": query,
            "key": key,
            "value": value,
            "block_table": block_table,
            "actual_seq_lengths": actual_seq_lengths,
            "actual_seq_lengths_kv": actual_seq_lengths_kv,
        },
        attrs={
            "num_heads": attr.Int(num_heads),
            "scale": attr.Float(scale),
            "pre_tokens": attr.Int(pre_tokens),
            "next_tokens": attr.Int(next_tokens),
            "input_layout": attr.Str(input_layout),
            "num_key_value_heads": attr.Int(num_key_value_heads),
            "block_size": attr.Int(block_size),
            "softmax_lse_flag": attr.Bool(softmax_lse_flag),
        },
        outputs=["attention_out", "softmax_lse"],
    )
