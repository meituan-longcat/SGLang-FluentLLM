import torch
import torchair
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(
    torch.ops.custom.npu_kvp_attention_merge.default
)
def convert_npu_kvp_attention_merge(
    attention_out: Tensor,
    lse: Tensor,
    *,
    meta_outputs: TensorSpec = None,
):
    return torchair.ge.custom_op(
        "KvpAttentionMerge",
        inputs={"attention_out": attention_out, "lse": lse},
        outputs=["y"],
    )
