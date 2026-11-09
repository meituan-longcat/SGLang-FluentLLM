import torch
import torchair
from torchair.ge import attr
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(
    torch.ops.custom.npu_router_input_slice_cast.default
)
def convert_npu_router_input_slice_cast(
    x: Tensor,
    start: int,
    length: int,
    *,
    meta_outputs: TensorSpec = None,
):
    return torchair.ge.custom_op(
        "RouterInputSliceCast",
        inputs={"x": x},
        attrs={
            "start": attr.Int(start),
            "length": attr.Int(length),
        },
        outputs=["y"],
    )
