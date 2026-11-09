import torch
import torchair
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.custom.npu_moe_expert_id_map.default)
def convert_npu_moe_expert_id_map(
    logical_ids: Tensor,
    expert_map: Tensor,
    *,
    meta_outputs: TensorSpec = None,
):
    return torchair.ge.custom_op(
        "MoeExpertIdMap",
        inputs={
            "logical_ids": logical_ids,
            "expert_map": expert_map,
        },
        outputs=["physical_ids"],
    )
