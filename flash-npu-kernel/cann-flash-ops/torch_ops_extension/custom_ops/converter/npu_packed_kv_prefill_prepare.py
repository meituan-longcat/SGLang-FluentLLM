import torch
import torchair
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge import attr
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(
    torch.ops.custom.npu_packed_kv_prefill_prepare.default
)
def convert_npu_packed_kv_prefill_prepare(
    latent_cache: Tensor,
    gamma: Tensor,
    cos: Tensor,
    sin: Tensor,
    owner_rows: Tensor,
    *,
    epsilon: float = 1e-6,
    packed_row_bytes: int = 656,
    meta_outputs: TensorSpec = None,
):
    return torchair.ge.custom_op(
        "PackedKvPrefillPrepare",
        inputs={
            "latent_cache": latent_cache,
            "gamma": gamma,
            "cos": cos,
            "sin": sin,
            "owner_rows": owner_rows,
        },
        attrs={
            "epsilon": attr.Float(epsilon),
            "packed_row_bytes": attr.Int(packed_row_bytes),
        },
        outputs=["key", "key_rope", "owner_packed"],
    )
