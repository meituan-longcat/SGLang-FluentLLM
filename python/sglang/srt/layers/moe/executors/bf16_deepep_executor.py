from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl

from deep_gemm_oss import (
    m_grouped_bf16_gemm_nt_contiguous,
    m_grouped_bf16_gemm_nt_masked,
)
from deep_gemm_oss.utils import get_mk_alignment_for_contiguous_layout
import flashinfer

from sglang.srt.layers.moe.executors.deepep_executor_base import DeepEPExecutorBase
from sglang.srt.layers.moe.executors.fp8_deepep_executor import ep_gather
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.ops.activation import silu_and_mul, swiglu


@triton.jit
def _fwd_kernel_ep_scatter_1_no_scale(
    num_recv_tokens_per_expert,
    expert_start_loc,
    m_indices,
    num_experts: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_EXPERT_NUM: tl.constexpr,
):
    cur_expert = tl.program_id(0)

    offset_cumsum = tl.arange(0, BLOCK_EXPERT_NUM)
    tokens_per_expert = tl.load(
        num_recv_tokens_per_expert + offset_cumsum,
        mask=offset_cumsum < num_experts,
        other=0,
    )
    cumsum = tl.cumsum(tokens_per_expert) - tokens_per_expert
    tl.store(expert_start_loc + offset_cumsum, cumsum, mask=offset_cumsum < num_experts)

    cur_expert_start = tl.sum(
        tl.where(offset_cumsum < cur_expert, tokens_per_expert, 0),
        axis=0,
    )
    cur_expert_token_num = tl.load(num_recv_tokens_per_expert + cur_expert)

    m_indices_start_ptr = m_indices + cur_expert_start
    off_expert = tl.arange(0, BLOCK_E)

    for start_m in tl.range(0, cur_expert_token_num, BLOCK_E, num_stages=4):
        tl.store(
            m_indices_start_ptr + start_m + off_expert,
            cur_expert,
        )


@triton.jit
def _fwd_kernel_ep_scatter_2_no_scale(
    total_token_num,
    expert_start_loc,
    recv_x,
    recv_x_stride0,
    recv_x_stride1,
    recv_topk,
    recv_topk_stride0,
    recv_topk_stride1,
    output_tensor,
    output_tensor_stride0,
    output_tensor_stride1,
    output_index,
    output_index_stride0,
    output_index_stride1,
    topk_num: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    HIDDEN_SIZE_PAD: tl.constexpr,
):
    start_token_id = tl.program_id(0)
    grid_num = tl.num_programs(0)

    offset_in = tl.arange(0, HIDDEN_SIZE_PAD)
    mask = offset_in < HIDDEN_SIZE

    for token_id_int32 in range(start_token_id, total_token_num, grid_num):
        token_id = token_id_int32.to(tl.int64)
        to_copy = tl.load(recv_x + token_id * recv_x_stride0 + offset_in, mask=mask)

        for topk_index_int32 in tl.range(0, topk_num, 1, num_stages=4):
            topk_index = topk_index_int32.to(tl.int64)
            expert_id = tl.load(recv_topk + token_id * recv_topk_stride0 + topk_index)
            if expert_id >= 0:
                dest_token_index_int32 = tl.atomic_add(expert_start_loc + expert_id, 1)
                dest_token_index = dest_token_index_int32.to(tl.int64)
                tl.store(
                    output_index + token_id * output_index_stride0 + topk_index,
                    dest_token_index_int32,
                )
                output_tensor_ptr = output_tensor + dest_token_index * output_tensor_stride0
                tl.store(output_tensor_ptr + offset_in, to_copy, mask=mask)


@torch.no_grad()
def ep_scatter_no_scale(
    recv_x: torch.Tensor,
    recv_topk: torch.Tensor,
    num_recv_tokens_per_expert: torch.Tensor,
    expert_start_loc: torch.Tensor,
    output_tensor: torch.Tensor,
    m_indices: torch.Tensor,
    output_index: torch.Tensor,
):
    block_e = 128  # token num of per expert is aligned to 128
    num_warps = 8
    num_experts = num_recv_tokens_per_expert.shape[0]
    hidden_size = recv_x.shape[1]
    grid = num_experts

    assert m_indices.shape[0] % block_e == 0

    _fwd_kernel_ep_scatter_1_no_scale[(grid,)](
        num_recv_tokens_per_expert,
        expert_start_loc,
        m_indices,
        num_experts=num_experts,
        num_warps=num_warps,
        BLOCK_E=block_e,
        BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
    )

    grid = min(recv_topk.shape[0], 1024 * 8)

    _fwd_kernel_ep_scatter_2_no_scale[(grid,)](
        recv_topk.shape[0],
        expert_start_loc,
        recv_x,
        recv_x.stride(0),
        recv_x.stride(1),
        recv_topk,
        recv_topk.stride(0),
        recv_topk.stride(1),
        output_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        output_index,
        output_index.stride(0),
        output_index.stride(1),
        topk_num=recv_topk.shape[1],
        num_warps=num_warps,
        HIDDEN_SIZE=hidden_size,
        HIDDEN_SIZE_PAD=triton.next_power_of_2(hidden_size),
    )


class BF16DeepEPExecutor(DeepEPExecutorBase):
    def __init__(self, ep_config, activation: str, swiglu_arg=None):
        super().__init__(
            ep_config,
            dispatch_use_fp8=False,
        )
        self.normal_gemm = m_grouped_bf16_gemm_nt_contiguous
        self.low_latency_gemm = m_grouped_bf16_gemm_nt_masked
        self.activation = activation
        self.swiglu_arg = swiglu_arg

    @staticmethod
    def _align(x: int, alignment: int) -> int:
        return ((x + alignment - 1) // alignment) * alignment

    def _get_m_alignment(self) -> int:
        return int(get_mk_alignment_for_contiguous_layout())

    def _prepare_contiguous_inputs(
        self,
        recv_hidden: torch.Tensor,
        recv_topk_idx: torch.Tensor,
        num_recv_tokens_per_expert: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = recv_hidden.device
        m_alignment = self._get_m_alignment()
        padded_counts_list = [self._align(count, m_alignment) for count in num_recv_tokens_per_expert]
        total_tokens = sum(padded_counts_list)
        routed_counts = torch.tensor(
            num_recv_tokens_per_expert,
            dtype=torch.int32,
            device=device,
        )
        padded_counts = torch.tensor(
            padded_counts_list,
            dtype=torch.int32,
            device=device,
        )
        expert_start = torch.cumsum(padded_counts, dim=0) - padded_counts

        expert_x = recv_hidden.new_zeros((total_tokens, recv_hidden.shape[1]))
        m_indices = torch.empty((total_tokens,), device=device, dtype=torch.int32)
        output_index = torch.full_like(recv_topk_idx, -1)

        expert_start_loc = expert_start.clone()

        ep_scatter_no_scale(
            recv_hidden,
            recv_topk_idx,
            padded_counts,
            expert_start_loc,
            expert_x,
            m_indices,
            output_index,
        )

        row_ids = torch.arange(total_tokens, device=device, dtype=torch.int32)
        row_is_valid = (row_ids - expert_start[m_indices.long()]) < routed_counts[m_indices.long()]

        return expert_x, m_indices, output_index, row_is_valid

    @staticmethod
    def _add_contiguous_bias(
        output: torch.Tensor,
        bias: Optional[torch.Tensor],
        m_indices: torch.Tensor,
        row_is_valid: torch.Tensor,
    ) -> None:
        if bias is None or output.shape[0] == 0:
            return
        if row_is_valid.any():
            output[row_is_valid] += bias[m_indices[row_is_valid].long()]

    @staticmethod
    def _zero_invalid_rows(output: torch.Tensor, row_is_valid: torch.Tensor) -> None:
        invalid = ~row_is_valid
        if invalid.any():
            output[invalid] = 0

    def _run_contiguous(
        self,
        layer,
        expert_x: torch.Tensor,
        m_indices: torch.Tensor,
        row_is_valid: torch.Tensor,
    ) -> torch.Tensor:
        gate_up_output = torch.empty(
            (expert_x.shape[0], layer.w13_weight.shape[1]),
            device=expert_x.device,
            dtype=expert_x.dtype,
        )
        self.normal_gemm(
            expert_x,
            layer.w13_weight,
            gate_up_output,
            m_indices,
        )
        self._add_contiguous_bias(
            gate_up_output,
            getattr(layer, "w13_weight_bias", None),
            m_indices,
            row_is_valid,
        )
        self._zero_invalid_rows(gate_up_output, row_is_valid)

        if self.activation == "silu":
            down_input = torch.empty(
                (gate_up_output.shape[0], gate_up_output.shape[1] // 2),
                device=gate_up_output.device,
                dtype=gate_up_output.dtype,
            )
            silu_and_mul(gate_up_output, down_input)
        elif self.activation == "swiglu":
            down_input = swiglu(
                gate_up_output,
                self.swiglu_arg.alpha,
                self.swiglu_arg.limit,
            )
        else:
            raise RuntimeError(f"Not supported {self.activation=}")
        self._zero_invalid_rows(down_input, row_is_valid)

        output = torch.empty(
            (expert_x.shape[0], layer.w2_weight.shape[1]),
            device=expert_x.device,
            dtype=expert_x.dtype,
        )
        self.normal_gemm(
            down_input,
            layer.w2_weight,
            output,
            m_indices,
        )
        self._add_contiguous_bias(
            output,
            getattr(layer, "w2_weight_bias", None),
            m_indices,
            row_is_valid,
        )
        return output

    def _run_masked(
        self,
        layer,
        packed_hidden: torch.Tensor,
        masked_m: torch.Tensor,
        expected_m: int,
    ) -> torch.Tensor:
        masked_m = masked_m.to(device=packed_hidden.device, dtype=torch.int32)
        num_experts = packed_hidden.shape[0]
        gate_up_output = torch.empty(
            (
                num_experts,
                packed_hidden.shape[1],
                layer.w13_weight.shape[1],
            ),
            device=packed_hidden.device,
            dtype=packed_hidden.dtype,
        )
        self.low_latency_gemm(
            packed_hidden,
            layer.w13_weight,
            gate_up_output,
            masked_m,
            expected_m,
        )
        if getattr(layer, "w13_weight_bias", None) is not None:
            gate_up_output += layer.w13_weight_bias.unsqueeze(1)

        if self.activation == "silu":
            down_input = torch.empty(
                gate_up_output.shape[:-1] + (gate_up_output.shape[-1] // 2,),
                device=gate_up_output.device,
                dtype=gate_up_output.dtype,
            )
            flashinfer.activation.silu_and_mul_ep(
                gate_up_output,
                masked_m,
                expected_m,
                num_experts,
                out=down_input,
            )
        elif self.activation == "swiglu":
            down_input = swiglu(
                gate_up_output,
                self.swiglu_arg.alpha,
                self.swiglu_arg.limit,
            )
        else:
            raise RuntimeError(f"Not supported {self.activation=}")

        output = torch.empty(
            (
                packed_hidden.shape[0],
                packed_hidden.shape[1],
                layer.w2_weight.shape[1],
            ),
            device=packed_hidden.device,
            dtype=packed_hidden.dtype,
        )
        self.low_latency_gemm(
            down_input,
            layer.w2_weight,
            output,
            masked_m,
            expected_m,
        )
        if getattr(layer, "w2_weight_bias", None) is not None:
            output += layer.w2_weight_bias.unsqueeze(1)
        return output

    def _gather_normal_output(
        self,
        down_output: torch.Tensor,
        recv_topk_idx: torch.Tensor,
        recv_topk_weights: torch.Tensor,
        output_index: torch.Tensor,
    ) -> torch.Tensor:
        gathered = torch.empty(
            (recv_topk_idx.shape[0], down_output.shape[1]),
            device=down_output.device,
            dtype=down_output.dtype,
        )
        ep_gather(
            down_output,
            recv_topk_idx,
            recv_topk_weights,
            output_index,
            gathered,
        )
        return gathered

    def forward_normal(
        self,
        layer,
        recv_hidden: torch.Tensor,
        recv_topk_idx: torch.Tensor,
        recv_topk_weights: torch.Tensor,
        num_recv_tokens_per_expert: List[int],
    ) -> torch.Tensor:
        if num_recv_tokens_per_expert is None:
            return recv_hidden

        expert_x, m_indices, output_index, row_is_valid = self._prepare_contiguous_inputs(
            recv_hidden,
            recv_topk_idx,
            num_recv_tokens_per_expert,
        )
        if expert_x.shape[0] == 0:
            return recv_hidden

        down_output = self._run_contiguous(layer, expert_x, m_indices, row_is_valid)
        return self._gather_normal_output(
            down_output,
            recv_topk_idx,
            recv_topk_weights,
            output_index,
        )

    def forward_low_latency(
        self,
        layer,
        packed_hidden: torch.Tensor,
        recv_count: torch.Tensor,
        expected_m: int,
    ) -> torch.Tensor:
        if packed_hidden.shape[0] == 0:
            return torch.zeros_like(packed_hidden)
        return self._run_masked(
            layer,
            packed_hidden,
            recv_count,
            expected_m,
        )

    def compute(
        self,
        layer,
        num_global_tokens,
        deepep_forward_mode,
        hidden_states,
        input_scales,
        topk_idx,
        topk_weights,
        num_recv_tokens_per_expert_list,
        masked_m,
    ):
        del input_scales
        if deepep_forward_mode == ForwardMode.DECODE:
            expected_m = max(
                1,
                num_global_tokens * layer.top_k // layer.num_experts,
            )
            return self.forward_low_latency(
                layer,
                hidden_states,
                masked_m,
                expected_m,
            )
        return self.forward_normal(
            layer,
            hidden_states,
            topk_idx,
            topk_weights,
            num_recv_tokens_per_expert_list,
        )
