from typing import Optional

import torch

from sglang.srt.configs import FLASHConfig
from sglang.srt.distributed import get_moe_ep_group, get_moe_expert_parallel_rank, get_moe_expert_parallel_world_size
from sglang.srt.distributed.parallel_strategy import AttnParallelStrategy, DenseParallelStategy
from sglang.srt.env import global_server_args_dict
from sglang.srt.layers.attention.dp_attention import get_attention_tp_group, get_dense_tp_group
from sglang.srt.layers.moe.config import EPConfig
from sglang.srt.layers.moe.gemms import get_gemm_cls
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.models.longcat_flash import FLASHDecoderLayer, FLASHForCausalLM
from sglang.srt.utils import get_colorful_logger, is_sm90_supported

if is_sm90_supported():
    from sglang.srt.layers.moe.executors.bf16_deepep_executor import BF16DeepEPExecutor
    from sglang.srt.layers.moe.executors.fp8_deepep_executor import FP8DeepEPExecutor
    from sglang.srt.layers.moe.gemms.bf16.aok import AokGroupedGemmWrapper
    from sglang.srt.layers.moe.gemms.fp8.fire import FireGroupedGemmWrapper

logger = get_colorful_logger(__name__)

# low latency - ll uses many SMs, so launch send as early as possible and launch recv as late as possible
# first attn |          router + dispatch send   | dispatch recv + moe gemm + combine send + recv
#            |      dense gemm + second attn pre |         second attn post + dense gemm


# intranode / internode
# first attn | router |     dispatch      |            moe gemm           | notify + combine
#            |   ag   | dense gemm + rs   |    ag + second attn + rs + ag |  dense gemm + rs
class FLASHDecoderLayerOverlap(FLASHDecoderLayer):
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        tp_num_tokens: int,
        next_layer_input_norm=None,
        input_is_sharded: bool = False,
        input_block_scale: Optional[torch.Tensor] = None,
        can_run_flashinfer_fusion: bool = False,
        can_run_next_flashinfer_fusion: bool = False,
        current_layer_attn_1_quanted: bool = False,
        next_layer_attn_0_quanted: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_global_tokens, max_num_tokens_per_gpu = forward_batch.get_num_tokens(tp_num_tokens)
        deepep_mode = (
            ForwardMode.EXTEND if max_num_tokens_per_gpu > global_server_args_dict["low_latency_max_num_tokens_per_gpu"] else ForwardMode.DECODE
        )

        if forward_batch.forward_mode.is_idle():
            return self._forward_idle(
                positions,
                hidden_states,
                forward_batch,
                residual,
                tp_num_tokens,
                num_global_tokens,
                max_num_tokens_per_gpu,
                deepep_mode,
            )

        if deepep_mode == ForwardMode.DECODE and can_run_flashinfer_fusion:
            return self._forward_decode_fused_entry(
                positions,
                hidden_states,
                forward_batch,
                residual,
                tp_num_tokens,
                num_global_tokens,
                input_is_sharded,
                input_block_scale,
                current_layer_attn_1_quanted,
                can_run_next_flashinfer_fusion,
                next_layer_input_norm,
                next_layer_attn_0_quanted,
            )

        if deepep_mode == ForwardMode.DECODE:
            return self._forward_decode_entry(
                positions,
                hidden_states,
                forward_batch,
                residual,
                tp_num_tokens,
                num_global_tokens,
                input_is_sharded,
                input_block_scale,
                can_run_flashinfer_fusion,
                can_run_next_flashinfer_fusion,
                next_layer_input_norm,
                next_layer_attn_0_quanted,
            )

        if deepep_mode == ForwardMode.EXTEND and can_run_flashinfer_fusion:
            return self._forward_extend_fused_entry(
                positions,
                hidden_states,
                forward_batch,
                residual,
                tp_num_tokens,
                num_global_tokens,
                input_is_sharded,
                input_block_scale,
                current_layer_attn_1_quanted,
                can_run_next_flashinfer_fusion,
                next_layer_input_norm,
                next_layer_attn_0_quanted,
            )

        if deepep_mode == ForwardMode.EXTEND:
            return self._forward_extend_entry(
                positions,
                hidden_states,
                forward_batch,
                residual,
                tp_num_tokens,
                num_global_tokens,
                input_is_sharded,
                input_block_scale,
                can_run_flashinfer_fusion,
                can_run_next_flashinfer_fusion,
                next_layer_input_norm,
                next_layer_attn_0_quanted,
            )

    def _forward_decode_fused_entry(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        tp_num_tokens: int,
        num_global_tokens: int,
        input_is_sharded: bool,
        input_block_scale: Optional[torch.Tensor],
        current_layer_attn_1_quanted: bool,
        can_run_next_flashinfer_fusion: bool,
        next_layer_input_norm,
        next_layer_attn_0_quanted: bool,
    ):
        current_stream, hidden_states, residual = self._run_first_attention(
            positions,
            hidden_states,
            forward_batch,
            residual,
            input_is_sharded,
            input_block_scale,
            can_run_flashinfer_fusion=True,
        )
        hidden_states_mlp, residual, block_scale_for_dense, moe_input_hidden_states = self._post_first_attention_fused(
            hidden_states, residual, input_is_sharded
        )
        return self._forward_decode_overlap_core_fused(
            positions,
            forward_batch,
            residual,
            tp_num_tokens,
            num_global_tokens,
            current_stream,
            hidden_states_mlp,
            block_scale_for_dense,
            moe_input_hidden_states,
            current_layer_attn_1_quanted,
            can_run_next_flashinfer_fusion,
            next_layer_input_norm,
            next_layer_attn_0_quanted,
        )

    def _forward_decode_entry(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        tp_num_tokens: int,
        num_global_tokens: int,
        input_is_sharded: bool,
        input_block_scale: Optional[torch.Tensor],
        can_run_flashinfer_fusion: bool,
        can_run_next_flashinfer_fusion: bool,
        next_layer_input_norm,
        next_layer_attn_0_quanted: bool,
    ):
        current_stream, hidden_states, residual = self._run_first_attention(
            positions,
            hidden_states,
            forward_batch,
            residual,
            input_is_sharded,
            input_block_scale,
            can_run_flashinfer_fusion=can_run_flashinfer_fusion,
        )
        hidden_states_mlp, residual, block_scale_for_dense, moe_input_hidden_states = self._post_first_attention(
            hidden_states, residual, tp_num_tokens
        )
        return self._forward_decode_overlap_core(
            positions,
            forward_batch,
            residual,
            tp_num_tokens,
            num_global_tokens,
            current_stream,
            hidden_states_mlp,
            block_scale_for_dense,
            moe_input_hidden_states,
            can_run_next_flashinfer_fusion,
            next_layer_input_norm,
            next_layer_attn_0_quanted,
        )

    def _forward_extend_fused_entry(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        tp_num_tokens: int,
        num_global_tokens: int,
        input_is_sharded: bool,
        input_block_scale: Optional[torch.Tensor],
        current_layer_attn_1_quanted: bool,
        can_run_next_flashinfer_fusion: bool,
        next_layer_input_norm,
        next_layer_attn_0_quanted: bool,
    ):
        current_stream, hidden_states, residual = self._run_first_attention(
            positions,
            hidden_states,
            forward_batch,
            residual,
            input_is_sharded,
            input_block_scale,
            can_run_flashinfer_fusion=True,
        )
        hidden_states_mlp, residual, block_scale_for_dense, moe_input_hidden_states = self._post_first_attention_fused(
            hidden_states, residual, input_is_sharded
        )
        return self._forward_extend_overlap_core_fused(
            positions,
            forward_batch,
            residual,
            tp_num_tokens,
            num_global_tokens,
            current_stream,
            hidden_states_mlp,
            block_scale_for_dense,
            moe_input_hidden_states,
            current_layer_attn_1_quanted,
            can_run_next_flashinfer_fusion,
            next_layer_input_norm,
            next_layer_attn_0_quanted,
        )

    def _forward_extend_entry(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        tp_num_tokens: int,
        num_global_tokens: int,
        input_is_sharded: bool,
        input_block_scale: Optional[torch.Tensor],
        can_run_flashinfer_fusion: bool,
        can_run_next_flashinfer_fusion: bool,
        next_layer_input_norm,
        next_layer_attn_0_quanted: bool,
    ):
        current_stream, hidden_states, residual = self._run_first_attention(
            positions,
            hidden_states,
            forward_batch,
            residual,
            input_is_sharded,
            input_block_scale,
            can_run_flashinfer_fusion=can_run_flashinfer_fusion,
        )
        hidden_states_mlp, residual, block_scale_for_dense, moe_input_hidden_states = self._post_first_attention(
            hidden_states, residual, tp_num_tokens
        )
        return self._forward_extend_overlap_core(
            positions,
            forward_batch,
            residual,
            tp_num_tokens,
            num_global_tokens,
            current_stream,
            hidden_states_mlp,
            block_scale_for_dense,
            moe_input_hidden_states,
            can_run_next_flashinfer_fusion,
            next_layer_input_norm,
            next_layer_attn_0_quanted,
        )

    def _run_first_attention(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        input_is_sharded: bool,
        input_block_scale: Optional[torch.Tensor],
        can_run_flashinfer_fusion: bool,
    ):
        current_stream = torch.cuda.current_stream()
        hidden_states, residual = self._prepare_first_attention_input(hidden_states, residual, input_is_sharded)
        hidden_states = self.self_attn[0](
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            comm_manager=self.mlp_branch_decoder_comm_manager[0],
            block_scale=input_block_scale if input_is_sharded else None,
            can_run_flashinfer_fusion=can_run_flashinfer_fusion,
        )
        return current_stream, hidden_states, residual

    def _prepare_first_attention_input(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        input_is_sharded: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not input_is_sharded:
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm[0](hidden_states)
            else:
                hidden_states, residual = self.input_layernorm[0](hidden_states, residual)
        return hidden_states, residual

    def _post_first_attention_fused(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        input_is_sharded: bool,
    ):
        # moe_input_hidden_states is partial (reduce-scattered pattern) bf16 input
        # for router gemm, while hidden_states_mlp is block-wise fp8.
        return self.post_attention_layernorm[0].forward_with_allreduce_fusion(
            get_attention_tp_group(),
            hidden_states,
            residual,
            fuse_block_quant_fp8=not self.dense_0_unqaunted,
            residual_reduce_scattered=input_is_sharded,
            trigger_completion_at_end=False,
            has_partial_norm_out=True,
        )

    def _post_first_attention(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        tp_num_tokens: int,
    ):
        hidden_states, residual = self.moe_branch_decoder_comm_manager.post_attn_comm(hidden_states, residual, tp_num_tokens)
        hidden_states, residual = self.post_attention_layernorm[0](hidden_states, residual)
        return hidden_states, residual, None, hidden_states

    def _prepare_moe_inputs(self, moe_input_hidden_states: torch.Tensor):
        hidden_states_moe = moe_input_hidden_states.clone()
        router_logits = self.mlp.router(hidden_states_moe)
        topk_out = self.mlp.topk(hidden_states_moe, router_logits)
        expert_indices = topk_out.topk_ids
        expert_scales = topk_out.topk_weights
        if self.config.zero_expert_type is not None and hidden_states_moe.shape[0] > 0:
            normal_expert_mask = expert_indices >= self.mlp.num_experts
            expert_indices[normal_expert_mask] = -1
            if self.config.zero_expert_type == "copy":
                expert_scales[normal_expert_mask] = 1.0
            if self.config.zero_expert_type == "drop":
                expert_scales[normal_expert_mask] = 0.0
        return (
            hidden_states_moe,
            expert_indices,
            expert_scales,
            hidden_states_moe,
        )

    @staticmethod
    def _split_absorb_qkv_outputs(absorb_outputs):
        if len(absorb_outputs) == 2:
            q, k = absorb_outputs
            return q, k, None
        if len(absorb_outputs) == 3:
            q, k, topk_indices = absorb_outputs
            return q, k, topk_indices
        raise ValueError(f"Unexpected output count from forward_absorb_qkv_proj: {len(absorb_outputs)}")

    @staticmethod
    def _forward_absorb_attn_o_proj_compat(
        attn,
        q,
        k,
        forward_batch: ForwardBatch,
        comm_manager,
        topk_indices=None,
    ):
        if topk_indices is not None:
            return attn.forward_absorb_attn_o_proj(q, k, forward_batch, topk_indices)
        return attn.forward_absorb_attn_o_proj(q, k, forward_batch, comm_manager)

    @staticmethod
    def _split_dispatch_hidden_and_scales(recv_hidden_with_scales):
        if isinstance(recv_hidden_with_scales, tuple):
            return recv_hidden_with_scales
        return recv_hidden_with_scales, None

    def _prepare_second_attention_input_fused(
        self,
        hidden_states_mlp: torch.Tensor,
        residual: torch.Tensor,
        current_layer_attn_1_quanted: bool,
    ):
        return self.input_layernorm[1].forward_with_reducescatter_fusion(
            get_attention_tp_group(),
            hidden_states_mlp,
            residual,
            fuse_block_quant_fp8=current_layer_attn_1_quanted,
        )

    def _prepare_second_attention_input(
        self,
        hidden_states_mlp: torch.Tensor,
        residual: torch.Tensor,
        tp_num_tokens: int,
        forward_batch: ForwardBatch,
    ):
        hidden_states_mlp, residual = self.mlp_branch_decoder_comm_manager[0].post_mlp_comm(hidden_states_mlp, residual, tp_num_tokens, forward_batch)
        return self.input_layernorm[1](hidden_states_mlp, residual) + (None,)

    def _post_second_attention_fused(self, hidden_states_mlp: torch.Tensor, residual: torch.Tensor):
        return self.post_attention_layernorm[1].forward_with_allreduce_fusion(
            get_attention_tp_group(),
            hidden_states_mlp,
            residual,
            fuse_block_quant_fp8=not self.dense_1_unqaunted,
            residual_reduce_scattered=True,
        )

    def _post_second_attention(
        self,
        hidden_states_mlp: torch.Tensor,
        residual: torch.Tensor,
        tp_num_tokens: int,
    ):
        hidden_states_mlp, residual = self.mlp_branch_decoder_comm_manager[1].post_attn_comm(hidden_states_mlp, residual, tp_num_tokens)
        hidden_states_mlp, residual = self.post_attention_layernorm[1](hidden_states_mlp, residual)
        return hidden_states_mlp, residual, None

    def _finalize_overlap_output(
        self,
        hidden_states_mlp: torch.Tensor,
        hidden_states_moe: torch.Tensor,
        residual: torch.Tensor,
        can_run_next_flashinfer_fusion: bool,
        next_layer_input_norm,
        next_layer_attn_0_quanted: bool,
    ):
        output_block_scale = None
        if can_run_next_flashinfer_fusion:
            hidden_states, residual, output_block_scale = next_layer_input_norm.forward_with_reducescatter_fusion(
                get_dense_tp_group(),
                hidden_states_mlp,
                residual,
                fuse_block_quant_fp8=next_layer_attn_0_quanted,
                add_in=hidden_states_moe,
            )
        else:
            hidden_states = hidden_states_mlp + hidden_states_moe

        return hidden_states, residual, output_block_scale

    @staticmethod
    def _should_run_idle_mlp_branch():
        return (
            global_server_args_dict["attn_parallel_strategy"] == AttnParallelStrategy.DATA_PARALLEL
            and global_server_args_dict["dense_parallel_strategy"] == DenseParallelStategy.TENSOR_PARALLEL
        )

    def _forward_decode_overlap_core(
        self,
        positions,
        forward_batch,
        residual,
        tp_num_tokens,
        num_global_tokens,
        current_stream,
        hidden_states_mlp,
        block_scale_for_dense,
        moe_input_hidden_states,
        can_run_next_flashinfer_fusion,
        next_layer_input_norm,
        next_layer_attn_0_quanted,
    ):
        deepep_mode = ForwardMode.DECODE
        # overlap stage 1 (ll): router + dispatch send | ag + dense gemm + rs + second attn pre
        # router + dispatch send
        # self.alt_stream.wait_stream(current_stream)
        # with torch.cuda.stream(self.alt_stream):
        (
            hidden_states_moe,
            expert_indices,
            expert_scales,
            moe_origin_input,
        ) = self._prepare_moe_inputs(moe_input_hidden_states)
        self.deepep_dispatcher.dispatch_a(hidden_states_moe, expert_indices, expert_scales, deepep_mode)
        # ag
        hidden_states_mlp, _, _ = self.mlp_branch_decoder_comm_manager[0].pre_mlp_comm(hidden_states_mlp.clone(), forward_batch, tp_num_tokens)
        # dense gemm
        hidden_states_mlp = self.mlps[0](hidden_states_mlp, block_scale=block_scale_for_dense)

        # rs
        hidden_states_mlp, residual, block_scale = self._prepare_second_attention_input(hidden_states_mlp, residual, tp_num_tokens, forward_batch)

        # second attn pre
        attn = self.self_attn[1]
        comm_manager = self.mlp_branch_decoder_comm_manager[1]
        if attn.no_absorb(forward_batch):
            q, k, v = attn.forward_normal_chunked_kv_prepare(positions, hidden_states_mlp, forward_batch, comm_manager, block_scale)
        else:
            Q, K, topk_indices = self._split_absorb_qkv_outputs(
                attn.forward_absorb_qkv_proj(
                    hidden_states_mlp,
                    positions,
                    forward_batch,
                    comm_manager,
                    block_scale,
                    can_run_flashinfer_fusion=False,
                )
            )
        # self.alt_stream.wait_stream(current_stream)
        (
            recv_hidden_with_scales,
            topk_idx,
            topk_weights,
            _,  # reorder_topk_ids,
            num_recv_tokens_per_expert_list,
            _,  # seg_indptr,
            masked_m,
        ) = self.deepep_dispatcher.dispatch_b()
        hidden_states_moe, input_scales = self._split_dispatch_hidden_and_scales(recv_hidden_with_scales)

        hidden_states_moe = self.deepep_executor.compute(
            self.mlp.experts,
            num_global_tokens,
            deepep_mode,
            hidden_states_moe,
            input_scales,
            topk_idx,
            topk_weights,
            num_recv_tokens_per_expert_list,
            masked_m,
        )
        self.deepep_dispatcher.combine_a(hidden_states_moe, topk_idx, topk_weights, deepep_mode, moe_origin_input)

        # overlap stage 2 (ll): dispatch recv + moe gemm + combine send + recv | second attn post + rs + ag + dense gemm + rs
        # second attn post
        attn = self.self_attn[1]
        comm_manager = self.mlp_branch_decoder_comm_manager[1]
        if attn.no_absorb(forward_batch):
            hidden_states_mlp = attn.forward_normal_chunked_kv_core(q, k, v, comm_manager, forward_batch)
        else:
            hidden_states_mlp = self._forward_absorb_attn_o_proj_compat(attn, Q, K, forward_batch, comm_manager, topk_indices)

        # rs
        hidden_states_mlp, residual, block_scale = self._post_second_attention(hidden_states_mlp, residual, tp_num_tokens)

        # ag
        # dense gemm
        hidden_states_mlp, _, _ = self.mlp_branch_decoder_comm_manager[1].pre_mlp_comm(hidden_states_mlp, forward_batch, tp_num_tokens)
        hidden_states_mlp = self.mlps[1](hidden_states_mlp, block_scale)

        # rs
        if not can_run_next_flashinfer_fusion:
            hidden_states_mlp, residual = self.mlp_branch_decoder_comm_manager[1].post_mlp_comm(
                hidden_states_mlp, residual, tp_num_tokens, forward_batch
            )

        # dispatch recv + moe gemm + combine send
        # with torch.cuda.stream(self.alt_stream):

        # self.alt_stream.wait_stream(current_stream)
        # with torch.cuda.stream(self.alt_stream):
        hidden_states_moe = self.deepep_dispatcher.combine_b()

        # current_stream.wait_stream(self.alt_stream)
        return self._finalize_overlap_output(
            hidden_states_mlp,
            hidden_states_moe,
            residual,
            can_run_next_flashinfer_fusion,
            next_layer_input_norm,
            next_layer_attn_0_quanted,
        )

    def _forward_decode_overlap_core_fused(
        self,
        positions,
        forward_batch,
        residual,
        tp_num_tokens,
        num_global_tokens,
        current_stream,
        hidden_states_mlp,
        block_scale_for_dense,
        moe_input_hidden_states,
        current_layer_attn_1_quanted,
        can_run_next_flashinfer_fusion,
        next_layer_input_norm,
        next_layer_attn_0_quanted,
    ):
        deepep_mode = ForwardMode.DECODE
        # overlap stage 1 (ll, fused path): router + dispatch send | dense gemm + fused second-attn pre
        # router + dispatch send
        self.alt_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.alt_stream):
            (
                hidden_states_moe,
                expert_indices,
                expert_scales,
                moe_origin_input,
            ) = self._prepare_moe_inputs(moe_input_hidden_states)
            self.deepep_dispatcher.dispatch_a(hidden_states_moe, expert_indices, expert_scales, deepep_mode)

        # dense gemm
        hidden_states_mlp = self.mlps[0](hidden_states_mlp, block_scale=block_scale_for_dense)
        # fused second attn pre
        hidden_states_mlp, residual, block_scale = self._prepare_second_attention_input_fused(
            hidden_states_mlp, residual, current_layer_attn_1_quanted
        )

        # second attn pre
        attn = self.self_attn[1]
        comm_manager = self.mlp_branch_decoder_comm_manager[1]
        if attn.no_absorb(forward_batch):
            q, k, v = attn.forward_normal_chunked_kv_prepare(positions, hidden_states_mlp, forward_batch, comm_manager, block_scale)
        else:
            Q, K, topk_indices = self._split_absorb_qkv_outputs(
                attn.forward_absorb_qkv_proj(
                    hidden_states_mlp,
                    positions,
                    forward_batch,
                    comm_manager,
                    block_scale,
                    can_run_flashinfer_fusion=True,
                )
            )
        self.alt_stream.wait_stream(current_stream)

        # overlap stage 2 (ll, fused path): dispatch recv + moe gemm + combine send + recv | fused second-attn post + dense gemm + rs
        # fused second attn post
        if attn.no_absorb(forward_batch):
            hidden_states_mlp = attn.forward_normal_chunked_kv_core(q, k, v, comm_manager, forward_batch)
        else:
            hidden_states_mlp = self._forward_absorb_attn_o_proj_compat(attn, Q, K, forward_batch, comm_manager, topk_indices)

        hidden_states_mlp, residual, block_scale, _ = self._post_second_attention_fused(hidden_states_mlp, residual)

        # dense gemm
        hidden_states_mlp = self.mlps[1](hidden_states_mlp, block_scale)

        # rs
        if not can_run_next_flashinfer_fusion:
            hidden_states_mlp, residual = self.mlp_branch_decoder_comm_manager[1].post_mlp_comm(
                hidden_states_mlp, residual, tp_num_tokens, forward_batch
            )

        # dispatch recv + moe gemm + combine send
        with torch.cuda.stream(self.alt_stream):
            (
                recv_hidden_with_scales,
                topk_idx,
                topk_weights,
                _,
                num_recv_tokens_per_expert_list,
                _,
                masked_m,
            ) = self.deepep_dispatcher.dispatch_b()
            hidden_states_moe, input_scales = self._split_dispatch_hidden_and_scales(recv_hidden_with_scales)

            hidden_states_moe = self.deepep_executor.compute(
                self.mlp.experts,
                num_global_tokens,
                deepep_mode,
                hidden_states_moe,
                input_scales,
                topk_idx,
                topk_weights,
                num_recv_tokens_per_expert_list,
                masked_m,
            )
            self.deepep_dispatcher.combine_a(hidden_states_moe, topk_idx, topk_weights, deepep_mode, moe_origin_input)

        self.alt_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.alt_stream):
            hidden_states_moe = self.deepep_dispatcher.combine_b()

        current_stream.wait_stream(self.alt_stream)
        return self._finalize_overlap_output(
            hidden_states_mlp,
            hidden_states_moe,
            residual,
            can_run_next_flashinfer_fusion,
            next_layer_input_norm,
            next_layer_attn_0_quanted,
        )

    def _forward_extend_overlap_core(
        self,
        positions,
        forward_batch,
        residual,
        tp_num_tokens,
        num_global_tokens,
        current_stream,
        hidden_states_mlp,
        block_scale_for_dense,
        moe_input_hidden_states,
        can_run_next_flashinfer_fusion,
        next_layer_input_norm,
        next_layer_attn_0_quanted,
    ):
        deepep_mode = ForwardMode.EXTEND
        # overlap stage 1 (normal): router | ag
        # router
        self.alt_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.alt_stream):
            (
                hidden_states_moe,
                expert_indices,
                expert_scales,
                moe_origin_input,
            ) = self._prepare_moe_inputs(moe_input_hidden_states)
        # ag
        hidden_states_mlp, _, _ = self.mlp_branch_decoder_comm_manager[0].pre_mlp_comm(hidden_states_mlp.clone(), forward_batch, tp_num_tokens)
        self.alt_stream.wait_stream(current_stream)

        # overlap stage 2 (normal): dispatch | dense gemm + rs
        with torch.cuda.stream(self.alt_stream):
            # dispatch
            self.deepep_dispatcher.dispatch_a(hidden_states_moe, expert_indices, expert_scales, forward_mode=deepep_mode)
            (
                recv_hidden_with_scales,
                recv_topk_idx,
                recv_topk_weights,
                _,  # reorder_topk_ids,
                num_recv_tokens_per_expert_list,
                _,  # seg_indptr,
                _,
            ) = self.deepep_dispatcher.dispatch_b()
            hidden_states_moe, input_scales = self._split_dispatch_hidden_and_scales(recv_hidden_with_scales)
        # dense gemm
        hidden_states_mlp = self.mlps[0](hidden_states_mlp, block_scale=block_scale_for_dense)
        self.alt_stream.wait_stream(current_stream)

        # overlap stage 3 (normal): moe gemm | rs + ag + second attn + rs + ag
        # moe gemm
        with torch.cuda.stream(self.alt_stream):
            hidden_states_moe = self.deepep_executor.compute(
                self.mlp.experts,
                num_global_tokens,
                deepep_mode,
                hidden_states_moe,
                input_scales,
                recv_topk_idx,
                recv_topk_weights,
                num_recv_tokens_per_expert_list,
                None,
            )

        # rs
        hidden_states_mlp, residual, block_scale = self._prepare_second_attention_input(hidden_states_mlp, residual, tp_num_tokens, forward_batch)

        # ag + second attn
        hidden_states_mlp = self.self_attn[1](
            positions=positions,
            hidden_states=hidden_states_mlp,
            forward_batch=forward_batch,
            comm_manager=self.mlp_branch_decoder_comm_manager[1],
            block_scale=block_scale,
            can_run_flashinfer_fusion=False,
        )

        # rs
        hidden_states_mlp, residual, block_scale = self._post_second_attention(hidden_states_mlp, residual, tp_num_tokens)
        # ag
        hidden_states_mlp, _, _ = self.mlp_branch_decoder_comm_manager[1].pre_mlp_comm(hidden_states_mlp, forward_batch, tp_num_tokens)
        self.alt_stream.wait_stream(current_stream)

        # overlap stage 4 (normal): notify + combine | dense gemm + rs
        # combine
        with torch.cuda.stream(self.alt_stream):
            self.deepep_dispatcher.combine_a(
                hidden_states_moe, expert_indices.to(torch.int64), (expert_scales.to(torch.float32), recv_topk_weights), deepep_mode, moe_origin_input
            )
            hidden_states_moe = self.deepep_dispatcher.combine_b()
        # dense gemm
        hidden_states_mlp = self.mlps[1](hidden_states_mlp, block_scale)

        if not can_run_next_flashinfer_fusion:
            hidden_states_mlp, residual = self.mlp_branch_decoder_comm_manager[1].post_mlp_comm(
                hidden_states_mlp, residual, tp_num_tokens, forward_batch
            )

        current_stream.wait_stream(self.alt_stream)
        return self._finalize_overlap_output(
            hidden_states_mlp,
            hidden_states_moe,
            residual,
            can_run_next_flashinfer_fusion,
            next_layer_input_norm,
            next_layer_attn_0_quanted,
        )

    def _forward_extend_overlap_core_fused(
        self,
        positions,
        forward_batch,
        residual,
        tp_num_tokens,
        num_global_tokens,
        current_stream,
        hidden_states_mlp,
        block_scale_for_dense,
        moe_input_hidden_states,
        current_layer_attn_1_quanted,
        can_run_next_flashinfer_fusion,
        next_layer_input_norm,
        next_layer_attn_0_quanted,
    ):
        deepep_mode = ForwardMode.EXTEND
        # overlap stage 1 (normal, fused path): router only; standalone ag is fused away
        self.alt_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.alt_stream):
            (
                hidden_states_moe,
                expert_indices,
                expert_scales,
                moe_origin_input,
            ) = self._prepare_moe_inputs(moe_input_hidden_states)

        # overlap stage 2 (normal, fused path): dispatch | dense gemm
        with torch.cuda.stream(self.alt_stream):
            self.deepep_dispatcher.dispatch_a(hidden_states_moe, expert_indices, expert_scales, forward_mode=deepep_mode)
            (
                recv_hidden_with_scales,
                recv_topk_idx,
                recv_topk_weights,
                _,
                num_recv_tokens_per_expert_list,
                _,
                _,
            ) = self.deepep_dispatcher.dispatch_b()
            hidden_states_moe, input_scales = self._split_dispatch_hidden_and_scales(recv_hidden_with_scales)

        hidden_states_mlp = self.mlps[0](hidden_states_mlp, block_scale=block_scale_for_dense)
        hidden_states_mlp, residual, block_scale = self._prepare_second_attention_input_fused(
            hidden_states_mlp, residual, current_layer_attn_1_quanted
        )
        self.alt_stream.wait_stream(current_stream)

        # overlap stage 3 (normal, fused path): moe gemm | fused second-attn path
        with torch.cuda.stream(self.alt_stream):
            hidden_states_moe = self.deepep_executor.compute(
                self.mlp.experts,
                num_global_tokens,
                deepep_mode,
                hidden_states_moe,
                input_scales,
                recv_topk_idx,
                recv_topk_weights,
                num_recv_tokens_per_expert_list,
                None,
            )

        hidden_states_mlp = self.self_attn[1](
            positions=positions,
            hidden_states=hidden_states_mlp,
            forward_batch=forward_batch,
            comm_manager=self.mlp_branch_decoder_comm_manager[1],
            block_scale=block_scale,
            can_run_flashinfer_fusion=True,
        )

        hidden_states_mlp, residual, block_scale, _ = self._post_second_attention_fused(hidden_states_mlp, residual)

        self.alt_stream.wait_stream(current_stream)

        # overlap stage 4 (normal, fused path): notify + combine | dense gemm; standalone rs is fused away
        with torch.cuda.stream(self.alt_stream):
            self.deepep_dispatcher.combine_a(
                hidden_states_moe,
                expert_indices.to(torch.int64),
                (expert_scales.to(torch.float32), recv_topk_weights),
                deepep_mode,
                moe_origin_input,
            )
            hidden_states_moe = self.deepep_dispatcher.combine_b()

        hidden_states_mlp = self.mlps[1](hidden_states_mlp, block_scale)

        if not can_run_next_flashinfer_fusion:
            hidden_states_mlp, residual = self.mlp_branch_decoder_comm_manager[1].post_mlp_comm(
                hidden_states_mlp, residual, tp_num_tokens, forward_batch
            )

        current_stream.wait_stream(self.alt_stream)
        return self._finalize_overlap_output(
            hidden_states_mlp,
            hidden_states_moe,
            residual,
            can_run_next_flashinfer_fusion,
            next_layer_input_norm,
            next_layer_attn_0_quanted,
        )

    def _forward_idle(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        tp_num_tokens: int,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
        deepep_mode: ForwardMode,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """An idle rank may still receive EP-routed tokens from peer ranks."""
        if deepep_mode == ForwardMode.DECODE:
            return self._forward_idle_decode(
                positions,
                hidden_states,
                forward_batch,
                residual,
                tp_num_tokens,
                num_global_tokens,
                max_num_tokens_per_gpu,
            )
        return self._forward_idle_extend(
            positions,
            hidden_states,
            forward_batch,
            residual,
            tp_num_tokens,
            num_global_tokens,
            max_num_tokens_per_gpu,
        )

    def _forward_idle_decode(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        tp_num_tokens: int,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        moe_origin_input = hidden_states
        topk_out = self.mlp.topk.empty_topk_output(hidden_states.device)
        expert_indices, expert_scales = topk_out.topk_ids, topk_out.topk_weights

        should_run_mlp_branch = self._should_run_idle_mlp_branch()

        self.deepep_dispatcher.dispatch_a(hidden_states, expert_indices, expert_scales, ForwardMode.DECODE)

        if should_run_mlp_branch:
            # ag
            hidden_states_mlp, _, _ = self.mlp_branch_decoder_comm_manager[0].pre_mlp_comm(hidden_states.clone(), forward_batch, tp_num_tokens)
            # dense gemm
            hidden_states_mlp = self.mlps[0](hidden_states_mlp)

            # rs
            hidden_states_mlp, residual = self.mlp_branch_decoder_comm_manager[0].post_mlp_comm(
                hidden_states_mlp, residual, tp_num_tokens, forward_batch
            )

        # if should_run_mlp_branch:
        #     hidden_states, residual = self.forward_mlp_branch(
        #         self.mlp_branch_decoder_comm_manager,
        #         positions,
        #         hidden_states,
        #         residual,
        #         forward_batch,
        #         num_global_tokens,
        #         max_num_tokens_per_gpu,
        #         tp_num_tokens,
        #     )

        (
            recv_hidden_with_scales,
            topk_idx,
            topk_weights,
            _,  # reorder_topk_ids
            num_recv_tokens_per_expert_list,
            _,  # seg_indptr
            masked_m,
        ) = self.deepep_dispatcher.dispatch_b()
        hidden_states_moe, input_scales = self._split_dispatch_hidden_and_scales(recv_hidden_with_scales)
        hidden_states_moe = self.deepep_executor.compute(
            self.mlp.experts,
            num_global_tokens,
            ForwardMode.DECODE,
            hidden_states_moe,
            input_scales,
            topk_idx,
            topk_weights,
            num_recv_tokens_per_expert_list,
            masked_m,
        )
        self.deepep_dispatcher.combine_a(
            hidden_states_moe,
            expert_indices.to(torch.int64),
            expert_scales.to(torch.float32),
            ForwardMode.DECODE,
            moe_origin_input,
        )

        if should_run_mlp_branch:
            # ag
            hidden_states_mlp, _, _ = self.mlp_branch_decoder_comm_manager[1].pre_mlp_comm(hidden_states_mlp, forward_batch, tp_num_tokens)
            hidden_states_mlp = self.mlps[1](hidden_states_mlp)

            # rs
            hidden_states_mlp, residual = self.mlp_branch_decoder_comm_manager[1].post_mlp_comm(
                hidden_states_mlp, residual, tp_num_tokens, forward_batch
            )

        hidden_states_moe = self.deepep_dispatcher.combine_b()

        return hidden_states_moe, residual, None

    def _forward_idle_extend(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        tp_num_tokens: int,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del positions, max_num_tokens_per_gpu

        moe_origin_input = hidden_states
        topk_out = self.mlp.topk.empty_topk_output(hidden_states.device)
        expert_indices, expert_scales = topk_out.topk_ids, topk_out.topk_weights
        should_run_mlp_branch = self._should_run_idle_mlp_branch()

        if should_run_mlp_branch:
            hidden_states, _, _ = self.mlp_branch_decoder_comm_manager[0].pre_mlp_comm(hidden_states.clone(), forward_batch, tp_num_tokens)

        self.deepep_dispatcher.dispatch_a(moe_origin_input, expert_indices, expert_scales, ForwardMode.EXTEND)
        (
            recv_hidden_with_scales,
            recv_topk_idx,
            recv_topk_weights,
            _,  # reorder_topk_ids
            num_recv_tokens_per_expert_list,
            _,  # seg_indptr
            _,
        ) = self.deepep_dispatcher.dispatch_b()
        hidden_states_moe, input_scales = self._split_dispatch_hidden_and_scales(recv_hidden_with_scales)

        if should_run_mlp_branch:
            hidden_states = self.mlps[0](hidden_states)
            hidden_states, residual = self.mlp_branch_decoder_comm_manager[0].post_mlp_comm(hidden_states, residual, tp_num_tokens, forward_batch)

        hidden_states_moe = self.deepep_executor.compute(
            self.mlp.experts,
            num_global_tokens,
            ForwardMode.EXTEND,
            hidden_states_moe,
            input_scales,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            None,
        )

        if should_run_mlp_branch:
            hidden_states, _, _ = self.mlp_branch_decoder_comm_manager[1].pre_mlp_comm(hidden_states, forward_batch, tp_num_tokens)

        self.deepep_dispatcher.combine_a(
            hidden_states_moe,
            expert_indices.to(torch.int64),
            (expert_scales.to(torch.float32), recv_topk_weights),
            ForwardMode.EXTEND,
            moe_origin_input,
        )
        hidden_states_moe = self.deepep_dispatcher.combine_b()

        if should_run_mlp_branch:
            hidden_states = self.mlps[1](hidden_states)
            hidden_states, residual = self.mlp_branch_decoder_comm_manager[1].post_mlp_comm(hidden_states, residual, tp_num_tokens, forward_batch)
            hidden_states = hidden_states + hidden_states_moe
        else:
            hidden_states = hidden_states_moe

        return hidden_states, residual, None

    def _init_deepep(self):
        chunked_prefill_size = global_server_args_dict["chunked_prefill_size"]
        max_num_tokens_per_gpu = chunked_prefill_size // global_server_args_dict["attn_tp_size"]
        config = EPConfig(
            top_k=self.mlp.topk.topk_config.top_k,
            num_experts=self.mlp.num_experts,
            low_latency_max_num_tokens_per_gpu=global_server_args_dict["low_latency_max_num_tokens_per_gpu"],
            max_num_tokens_per_gpu=max_num_tokens_per_gpu,
            hidden_size=self.mlp.hidden_size,
            rank=get_moe_expert_parallel_rank(),
            world_size=get_moe_expert_parallel_world_size(),
            group=get_moe_ep_group().device_group,
            params_dtype=torch.bfloat16,
        )
        gemm_cls = get_gemm_cls(
            self.mlp.quant_config,
            is_ep=True,
            prefix=self.mlp.experts.prefix,
        )
        if gemm_cls is FireGroupedGemmWrapper:
            self.deepep_executor = FP8DeepEPExecutor(config)
        elif gemm_cls is AokGroupedGemmWrapper:
            self.deepep_executor = BF16DeepEPExecutor(
                config,
                self.mlp.experts.activation,
                self.mlp.experts.swiglu_arg,
            )
        else:
            raise RuntimeError(f"Unsupported DeepEP overlap GEMM wrapper: {gemm_cls.__name__}")
        self.deepep_dispatcher = self.deepep_executor.deepep_dispatcher


class FLASHForCausalLMOverlap(FLASHForCausalLM):
    def __init__(self, config: FLASHConfig, quant_config: Optional[QuantizationConfig] = None) -> None:
        super().__init__(config, quant_config)

        for layer_idx in range(self.model.start_layer, self.model.end_layer):
            layer = self.model.layers[layer_idx]
            # monkey patch
            layer.__class__ = FLASHDecoderLayerOverlap
            layer._init_deepep()
            layer.config = config
            layer.global_rank = torch.distributed.get_rank()


EntryClass = [FLASHForCausalLMOverlap]
