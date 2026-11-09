"""
Torch-native implementation for FusedMoE. This is used for torch.compile.
It is based on https://github.com/pytorch-labs/gpt-fast/blob/32971d3129541c5bfb4f715abc33d1c5f408d204/mixtral-moe/model.py#L204
"""

import torch
from torch.nn import functional as F


def fused_moe_forward_native(
    layer: torch.nn.Module,
    dispatch_output,
) -> torch.Tensor:

    x, topk_output = dispatch_output
    moe_runner_config = layer.moe_runner_config

    if moe_runner_config.apply_router_weight_on_input:
        raise NotImplementedError()

    topk_weights, topk_ids, _ = topk_output

    w13_weights = layer.w13_weight[topk_ids]
    w1_weights, w3_weights = torch.chunk(w13_weights, 2, dim=2)
    w2_weights = layer.w2_weight[topk_ids]
    x1 = torch.einsum("ti,taoi -> tao", x, w1_weights)
    if moe_runner_config.activation == "silu":
        x1 = F.silu(x1)
    elif moe_runner_config.activation == "gelu":
        x1 = F.gelu(x1)
    else:
        raise ValueError(f"Unsupported activation: {moe_runner_config.activation=}")
    x3 = torch.einsum("ti, taoi -> tao", x, w3_weights)
    expert_outs = torch.einsum("tao, taio -> tai", (x1 * x3), w2_weights)
    return torch.einsum("tai,ta -> ti", expert_outs, topk_weights.to(expert_outs.dtype))
