import torch
import torchair
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair._ge_concrete_graph.utils import dtype_promote
from torchair.ge import attr
from torchair.ge._ge_graph import DataType, Tensor, TensorSpec


@register_fx_node_ge_converter(
    torch.ops.custom.npu_select_local_topk_indices.default
)
def convert_npu_select_local_topk_indices(
    topk_indices_local: Tensor,
    topk_pos: Tensor,
    kvp_rank: int,
    *,
    meta_outputs: TensorSpec = None,
):
    topk_pos = dtype_promote(topk_pos, target_dtype=DataType.DT_INT32)
    return torchair.ge.custom_op(
        "SelectLocalTopkIndices",
        inputs={
            "topk_indices_local": topk_indices_local,
            "topk_pos": topk_pos,
        },
        attrs={"kvp_rank": attr.Int(kvp_rank)},
        outputs=["topk_indices", "valid_chunks"],
    )
