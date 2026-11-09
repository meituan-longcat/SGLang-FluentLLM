import functools
from typing import Optional

import torch

from sglang.srt.layers.moe.gemms.triton_common import moe_align_block_size, moe_sum_reduce_torch_compile, moe_sum_reduce_triton
from sglang.srt.ops.activation import SwigluArg, silu_and_mul


try:
    from triton.tools.tensor_descriptor import TensorDescriptor

    _support_tensor_descriptor = True
except Exception:
    _support_tensor_descriptor = False


def support_tensor_descriptor():
    return _support_tensor_descriptor


@functools.lru_cache()
def _moe_use_tma():
    return support_tensor_descriptor()


class TritonExecutor:
    def __init__(self, gate_up_gemm, down_gemm, get_config_func, activation: str, swiglu_arg: Optional[SwigluArg] = None):
        self.gate_up_gemm = gate_up_gemm
        self.down_gemm = down_gemm
        self.get_config_func = get_config_func
        self.activation = activation
        self.swiglu_arg = swiglu_arg

    def forward(self, layer, hidden_states, topk_output, num_global_tokens, max_num_tokens_per_gpu):
        assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"

        routed_scaling_factor = None
        topk_ids = topk_output.topk_ids

        M = hidden_states.shape[0]
        E, N, hidden_size = layer.w13_weight.shape
        topk = topk_ids.shape[1]
        device = hidden_states.device
        dtype = hidden_states.dtype

        config, (down_config, max_block_m) = self.get_config_func(M=M)

        gate_up_moe_use_tma = _moe_use_tma() and config is not None and config.pop("USE_TMA", False)

        down_moe_use_tma = _moe_use_tma() and down_config is not None and down_config.pop("USE_TMA", False)

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(topk_ids, config["BLOCK_SIZE_M"], E)

        max_num_active_experts = min(M * topk, E + 1)
        padded_tokens = max_num_active_experts * (config["BLOCK_SIZE_M"] - 1) if down_moe_use_tma else 0
        intermediate_cache1 = torch.empty(
            (M * topk + padded_tokens, N),
            device=device,
            dtype=dtype,
        )
        intermediate_cache2 = torch.empty(
            (M * topk + padded_tokens, N // 2),
            device=device,
            dtype=dtype,
        )
        intermediate_cache3 = torch.empty(
            (M, topk, hidden_size),
            device=device,
            dtype=dtype,
        )

        self.gate_up_gemm(
            A=hidden_states,
            B=layer.w13_weight,
            bias=None,
            C=intermediate_cache1,
            topk_weights=topk_output.topk_weights,
            topk_ids=topk_ids,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            config=config,
            a_use_tma=False,
            b_use_tma=gate_up_moe_use_tma,
            c_sorted=down_moe_use_tma,
        )

        if self.activation == "silu":
            assert self.activation is not None
            silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
        else:
            raise ValueError(f"Unsupported activation: {self.activation=}")

        self.down_gemm(
            A=intermediate_cache2,
            B=layer.w2_weight,
            bias=None,
            C=intermediate_cache3,
            topk_weights=topk_output.topk_weights,
            topk_ids=topk_ids,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            config=down_config,
            a_use_tma=down_moe_use_tma,
            b_use_tma=down_moe_use_tma,
        )

        if routed_scaling_factor is None:
            routed_scaling_factor = 1.0

        out_hidden_states = torch.empty_like(hidden_states)
        # combine
        # According to micro benchmark results, torch.compile can get better performance for small token.
        if M <= 32:
            moe_sum_reduce_torch_compile(
                intermediate_cache3.view(*intermediate_cache3.shape),
                out_hidden_states,
                routed_scaling_factor,
            )
        else:
            moe_sum_reduce_triton(
                intermediate_cache3.view(*intermediate_cache3.shape),
                out_hidden_states,
                routed_scaling_factor,
            )

        return out_hidden_states
