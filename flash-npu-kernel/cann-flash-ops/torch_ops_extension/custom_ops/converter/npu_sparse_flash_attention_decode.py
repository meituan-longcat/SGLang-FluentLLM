from typing import Optional
import torch
import torchair
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge import attr
from torchair.ge._ge_graph import Tensor, TensorSpec

@register_fx_node_ge_converter(torch.ops.custom.npu_sparse_flash_attention_decode.default)
def convert_npu_sparse_flash_attention_decode(
    query: Tensor, key: Tensor, value: Tensor, sparse_indices: Tensor,
    scale_value: float, *, valid_chunks: Tensor,
    block_table: Optional[Tensor] = None,
    actual_seq_lengths_query: Optional[Tensor] = None,
    actual_seq_lengths_kv: Optional[Tensor] = None,
    query_rope: Optional[Tensor] = None, key_rope: Optional[Tensor] = None,
    sparse_block_size: int = 1, layout_query: str = "TND",
    layout_kv: str = "PA_BSND", sparse_mode: int = 3,
    pre_tokens: int = (1 << 63) - 1, next_tokens: int = (1 << 63) - 1,
    attention_mode: int = 2, return_softmax_lse: bool = True,
    sparse_prep_mode: int = 0,
    meta_outputs: TensorSpec = None):
    if attention_mode != 2:
        raise ValueError(
            f"attention_mode must be 2 (MLA), but got {attention_mode}"
        )
    if sparse_prep_mode not in (0, 1, 2):
        raise ValueError(
            "sparse_prep_mode must be 0 (full scan), 1 "
            "(valid_chunks bounded scan), or 2 (trusted-bound holes), "
            f"but got {sparse_prep_mode}"
        )
    return torchair.ge.custom_op("SparseFlashAttentionDecode", inputs={
        "query": query, "key": key, "value": value,
        "sparse_indices": sparse_indices, "valid_chunks": valid_chunks,
        "block_table": block_table,
        "actual_seq_lengths_query": actual_seq_lengths_query,
        "actual_seq_lengths_kv": actual_seq_lengths_kv,
        "query_rope": query_rope, "key_rope": key_rope}, attrs={
        "scale_value": attr.Float(scale_value),
        "sparse_block_size": attr.Int(sparse_block_size),
        "layout_query": attr.Str(layout_query), "layout_kv": attr.Str(layout_kv),
        "sparse_mode": attr.Int(sparse_mode), "pre_tokens": attr.Int(pre_tokens),
        "next_tokens": attr.Int(next_tokens), "attention_mode": attr.Int(attention_mode),
        "return_softmax_lse": attr.Bool(return_softmax_lse),
        "sparse_prep_mode": attr.Int(sparse_prep_mode)},
        outputs=["attention_out", "softmax_max", "softmax_sum"])
