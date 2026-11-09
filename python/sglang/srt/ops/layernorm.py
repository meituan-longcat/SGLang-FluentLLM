# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Fused operators for normalization layers."""

from typing import Optional, Union

import torch
import torch.nn as nn

from sglang.srt.distributed import GroupCoordinator
from sglang.srt.utils import get_colorful_logger, is_cuda_available, is_npu

if is_cuda_available():
    from flashinfer import fused_add_rmsnorm, gemma_fused_add_rmsnorm, gemma_rmsnorm, rmsnorm

if is_npu():
    import torch_npu

from sglang.srt.ops.custom_op import CustomOp
from sglang.srt.utils import is_npu

logger = get_colorful_logger(__name__)


class RMSNorm(CustomOp):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        inplace: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # There might be no tokens here
        if x.shape[0] == 0:
            if residual is not None:
                return x, residual
            else:
                return x
        if residual is not None:
            assert not inplace, "fused_add_rmsnorm does not support inplace operation"
            fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
            return x, residual
        out = rmsnorm(x, self.weight.data, self.variance_epsilon, out=x if inplace else None)
        return out

    def forward_npu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        inplace: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # Same as above
        if x.shape[0] == 0:
            if residual is not None:
                return x, residual
            else:
                return x
        if residual is not None:
            x, _, residual = torch_npu.npu_add_rms_norm(x, residual, self.weight, self.variance_epsilon)
            return x, residual
        return torch_npu.npu_rms_norm(x, self.weight.data, self.variance_epsilon)[0]

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        inplace: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # Same as above
        if x.shape[0] == 0:
            if residual is not None:
                return x, residual
            else:
                return x
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = (x * self.weight).to(orig_dtype)
        if residual is None:
            return x
        else:
            return x, residual

    def forward_with_allreduce_fusion(
        self,
        group: GroupCoordinator,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        fuse_block_quant_fp8: bool = False,
        residual_reduce_scattered: bool = False,
        max_sm_to_use: Optional[int] = None,
        trigger_completion_at_end: bool = False,
        has_partial_norm_out: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward method with allreduce fusion, prioritizing flashinfer fused operations
        """
        from sglang.srt.distributed.flashinfer_comm_fusion import (
            flashinfer_allreduce_residual_rmsnorm,
        )

        if residual is not None:
            from sglang.srt.env import global_server_args_dict

            if group.world_size > 1:
                residual_for_fusion = residual
                # FlashInfer assumes residual is already RS-sharded.
                if not residual_reduce_scattered:
                    world_size = group.world_size
                    rank = group.rank_in_group
                    total_tokens = residual.shape[0]
                    token_counts = [total_tokens // world_size + (1 if i < total_tokens % world_size else 0) for i in range(world_size)]
                    token_start = sum(token_counts[:rank])
                    token_end = token_start + token_counts[rank]
                    residual_for_fusion = residual[token_start:token_end]
                fused_result = flashinfer_allreduce_residual_rmsnorm(
                    input_tensor=x,
                    residual=residual_for_fusion,
                    weight=self.weight,
                    group=group,
                    eps=self.variance_epsilon,
                    max_token_num=global_server_args_dict["flashinfer_comm_max_num_tokens"],
                    block_quant_fp8=fuse_block_quant_fp8,
                    max_sm_to_use=max_sm_to_use,
                    trigger_completion_at_end=trigger_completion_at_end,
                    has_partial_norm_out=has_partial_norm_out,
                )
                if fused_result[0] is not None:
                    return fused_result

        result = self.forward(x, residual)
        if isinstance(result, tuple):
            return result[0], result[1], None
        return result, None, None

    def forward_with_reducescatter_fusion(
        self,
        group: GroupCoordinator,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        fuse_block_quant_fp8: bool = False,
        add_in: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward method with reducescatter fusion, prioritizing flashinfer fused operations
        """
        from sglang.srt.distributed.flashinfer_comm_fusion import (
            flashinfer_reducescatter_residual_rmsnorm,
        )

        if residual is not None:
            from sglang.srt.env import global_server_args_dict

            if group.world_size > 1:
                fused_result = flashinfer_reducescatter_residual_rmsnorm(
                    input_tensor=x,
                    residual=residual,
                    weight=self.weight,
                    group=group,
                    eps=self.variance_epsilon,
                    max_token_num=global_server_args_dict["flashinfer_comm_max_num_tokens"],
                    use_oneshot=True,
                    block_quant_fp8=fuse_block_quant_fp8,
                    add_in=add_in,
                )
                if fused_result[0] is not None:
                    return fused_result

        result = self.forward(x, residual)
        if isinstance(result, tuple):
            return result[0], result[1], None
        return result, None, None


class GemmaRMSNorm(CustomOp):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if x.shape[0] == 0:
            if residual is not None:
                return x, residual
            else:
                return x
        orig_dtype = x.dtype
        if residual is not None:
            x = x + residual
            residual = x

        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x * (1.0 + self.weight.float())
        x = x.to(orig_dtype)
        return x if residual is None else (x, residual)

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if x.shape[0] == 0:
            if residual is not None:
                return x, residual
            else:
                return x
        if residual is not None:
            gemma_fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
            return x, residual
        out = gemma_rmsnorm(x, self.weight.data, self.variance_epsilon)
        return out


if not (is_cuda_available() or is_npu()):
    logger.info("sgl-kernel is not available on Non-NV platforms. Fallback to other kernel libraries.")
    from vllm.model_executor.ops.layernorm import RMSNorm


class FusedRMSNorm(nn.Module):
    """Fused RMSNorm layer for normalizing two tensors simultaneously.

    This layer wraps two independent RMSNorm layers (q_a and kv_a) and performs
    fused normalization during forward pass. The RMSNorm layers are passed in as
    parameters, allowing reuse of existing normalization layers.
    """

    def __init__(
        self,
        q_a_norm: RMSNorm,
        kv_a_norm: RMSNorm,
    ) -> None:
        super().__init__()
        self.q_a_norm = q_a_norm
        self.kv_a_norm = kv_a_norm

    @property
    def weight_q_a(self) -> nn.Parameter:
        """Expose weight_q_a from q_a_norm for backward compatibility."""
        return self.q_a_norm.weight

    @property
    def weight_kv_a(self) -> nn.Parameter:
        """Expose weight_kv_a from kv_a_norm for backward compatibility."""
        return self.kv_a_norm.weight

    def forward(
        self,
        input_q_a: torch.Tensor,
        input_kv_a: torch.Tensor,
        output_q_a: Optional[torch.Tensor] = None,
        output_kv_a: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize two tensors in parallel using fused computation.

        Args:
            input_q_a: Q tensor to normalize
            input_kv_a: KV tensor to normalize

        Returns:
            Tuple of (normalized_q_a, normalized_kv_a)
        """
        from flashinfer.norm import _rmsnorm_fused_parallel

        _rmsnorm_fused_parallel(
            input1=input_q_a,
            weight1=self.weight_q_a,
            output1=output_q_a if output_q_a is not None else input_q_a,
            input2=input_kv_a,
            weight2=self.weight_kv_a,
            output2=output_kv_a if output_kv_a is not None else input_kv_a,
            eps=self.q_a_norm.variance_epsilon,
        )
        return input_q_a, input_kv_a

    def forward_with_allgather_fusion(
        self,
        group: GroupCoordinator,
        qkv: torch.Tensor,
        total_num_tokens: int,
        fuse_block_quant_fp8: bool = False,
        trigger_completion_at_end: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward method with allgather fusion, performing allgather + dual RMSNorm + optional FP8 block quantization.

        This method uses trtllm_allgather_fusion to fuse allgather communication with dual RMSNorm computation
        and optional FP8 block-wise quantization in a single kernel launch.

        Args:
            qkv: Input tensor to allgather, shape [num_token_current_rank, q_lora_rank + kv_lora_rank + qk_rope_head_dim]
            fuse_block_quant_fp8: Whether to perform FP8 block-wise quantization on the first norm output
            trigger_completion_at_end: Whether to trigger completion event at the end of kernel

        Returns:
            Tuple of (allgather_out, quant_out, k_nope, block_scale):
                - allgather_out: Gathered tensor, shape [num_token_all_group, hidden_dim]
                - quant_out: FP8 quantized first norm output (q_contiguous), None if fuse_block_quant_fp8=False
                - k_nope: Second norm output
                - block_scale: Quantization scales, None if fuse_block_quant_fp8=False
        """
        from sglang.srt.distributed.flashinfer_comm_fusion import flashinfer_allgather_dual_rmsnorm
        from sglang.srt.env import global_server_args_dict

        if group.world_size > 1:
            fused_result = flashinfer_allgather_dual_rmsnorm(
                qkv=qkv,
                total_num_tokens=total_num_tokens,
                group=group,
                weight_q_a=self.weight_q_a,
                weight_kv_a=self.weight_kv_a,
                eps_q=self.q_a_norm.variance_epsilon,
                eps_kv=self.kv_a_norm.variance_epsilon,
                max_token_num=global_server_args_dict["flashinfer_comm_max_num_tokens"],
                block_quant_fp8=fuse_block_quant_fp8,
                trigger_completion_at_end=trigger_completion_at_end,
                fp32_acc=False,
            )
            if fused_result[0] is not None:
                return fused_result

        q_lora_rank = self.weight_q_a.shape[0]
        kv_lora_rank = self.weight_kv_a.shape[0]
        q = qkv[..., :q_lora_rank]
        k_nope = qkv[..., q_lora_rank : q_lora_rank + kv_lora_rank]
        q_contiguous = torch.empty_like(q)
        if q.shape[0] > 0:
            self.forward(input_q_a=q, input_kv_a=k_nope, output_q_a=q_contiguous)

        return qkv, q_contiguous, k_nope, None
