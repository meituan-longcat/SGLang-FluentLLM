import torch
import torchair
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge import attr
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(
    torch.ops.custom.npu_sparse_mla_attention_prefill.default
)
def convert_npu_sparse_mla_attention_prefill(
    query: Tensor,
    gathered_kv_pages: Tensor,
    logical_sparse_rows: Tensor,
    scan_count: Tensor,
    page_table: Tensor,
    query_seq: Tensor,
    scale_value: float,
    head_major_output: bool = False,
    meta_outputs: TensorSpec = None,
):
    return torchair.ge.custom_op(
        "SparseMlaAttentionPrefill",
        inputs={
            "query": query,
            "gathered_kv_pages": gathered_kv_pages,
            "logical_sparse_rows": logical_sparse_rows,
            "scan_count": scan_count,
            "page_table": page_table,
            "query_seq": query_seq,
        },
        attrs={
            "scale_value": attr.Float(scale_value),
            "head_major_output": attr.Bool(head_major_output),
        },
        outputs=["attention_out"],
    )
