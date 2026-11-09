# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.

from typing import Any, Optional

import torch
import torchair
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge import attr
from torchair.ge._ge_graph import Tensor


def _causal_conv1d(
    x: Tensor,
    weight: Tensor,
    conv_state: Tensor,
    bias: Optional[Tensor],
    query_start_loc: Optional[Tensor],
    cache_indices: Optional[Tensor],
    initial_state_mode: Optional[Tensor],
    num_accepted_tokens: Optional[Tensor],
    write_indices: Optional[Tensor],
    activation_mode: int,
    pad_slot_id: int,
    run_mode: int,
):
    # GE materializes noncontiguous graph inputs before invoking this converter's
    # op, but their FakeTensor strides still describe the original arena view.
    # Never use those strides against the copied storage of a mutable state.
    if not conv_state.meta.is_contiguous():
        raise NotImplementedError(
            "CausalConv1d strided state requires eager or native NPUGraph; "
            "TorchAir GE materializes noncontiguous inputs"
        )
    return torchair.ge.custom_op(
        "CausalConv1d",
        inputs={
            "x": x,
            "weight": weight,
            "bias": bias,
            "convStates": conv_state,
            "queryStartLoc": query_start_loc,
            "cacheIndices": cache_indices,
            "initialStateMode": initial_state_mode,
            "numAcceptedTokens": num_accepted_tokens,
            "writeIndices": write_indices,
        },
        attrs={
            "activationMode": attr.Int(activation_mode),
            "padSlotId": attr.Int(pad_slot_id),
            "runMode": attr.Int(run_mode),
            "stateSlotStride": attr.Int(conv_state.meta.stride(0)),
        },
        outputs=["y"],
    )


@register_fx_node_ge_converter(torch.ops.custom.npu_causal_conv1d.default)
def convert_npu_causal_conv1d(
    x: Tensor,
    weight: Tensor,
    conv_state: Tensor,
    *,
    bias: Optional[Tensor] = None,
    query_start_loc: Optional[Tensor] = None,
    cache_indices: Optional[Tensor] = None,
    write_indices: Optional[Tensor] = None,
    initial_state_mode: Optional[Tensor] = None,
    num_accepted_tokens: Optional[Tensor] = None,
    activation_mode: int = 1,
    pad_slot_id: int = -1,
    run_mode: int = 0,
    meta_outputs: Any = None,
):
    return _causal_conv1d(
        x,
        weight,
        conv_state,
        bias,
        query_start_loc,
        cache_indices,
        initial_state_mode,
        num_accepted_tokens,
        write_indices,
        activation_mode,
        pad_slot_id,
        run_mode,
    )
