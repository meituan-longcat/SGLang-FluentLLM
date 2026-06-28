from typing import Optional, Tuple, List

import torch

from sglang.srt.configs import FLASHConfig
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_world_group,
)
from sglang.srt.layers.dp_attention import get_attention_tp_group, get_dense_tp_group
from sglang.srt.layers.moe.config import EPConfig
from sglang.srt.layers.moe.dispatcher.deep_ep import DeepEPDispatcher
from sglang.srt.layers.moe.executors.deep_ep_executor import DeepExecutor
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.env import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.models.longcat_flash import FLASHDecoderLayer, FLASHForCausalLM
from sglang.srt.distributed.parallel_strategy import AttnParallelStrategy, DenseParallelStategy
from sglang.srt.layers.quantization.utils import should_ignore_quant_layer
from sglang.srt.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

MEMORY_THRESHOLD_GB = 0

# low latency
# The router is followed by point-wise ops (e.g., Top-K); overlap these with dense GEMM to mask their latency.
# Recv has synchronization; launch at appropriate positions without preempting GEMM's SM.
# first attn |          router + dispatch send   |    dispatch recv + moe gemm + combine send | combine recv
#            | first mlp dense + second attn pre |       second attn post + second mlp dense  |

# intranode
# first attn | router |  dispatch  |           moe gemm              | notify + combine
#            |   ag   | dense gemm | rs + ag + second attn + rs + ag |  dense gemm + rs
class FLASHDecoderLayerOverlap(FLASHDecoderLayer):
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        tp_num_tokens: int,
        next_layer_input_norm = None,
        input_is_sharded: bool = False,
        input_block_scale: Optional[torch.Tensor] = None,
        use_fused_comm: bool = False,
        next_layer_attn_0_quanted: bool = False,
        layers_to_capture: List = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        num_global_tokens, max_num_tokens_per_gpu = forward_batch.get_num_tokens(
            tp_num_tokens
        )
        is_eagle3_layer_capture = (self.layer_id + 1) not in layers_to_capture if layers_to_capture is not None else False
        is_after_eagle3_layer_capture = self.layer_id not in layers_to_capture if layers_to_capture is not None else False

        if not forward_batch.forward_mode.is_idle():
            # Whether to use flashinfer all_reduce_residual_norm with fp8 block quant fusion, only support block_size 128 for now
            can_run_next_flashinfer_fusion=(use_fused_comm and next_layer_input_norm is not None) if is_eagle3_layer_capture else False
            
            current_stream = torch.cuda.current_stream()
            deepep_mode = (
                ForwardMode.EXTEND
                if max_num_tokens_per_gpu
                > global_server_args_dict["low_latency_max_num_tokens_per_gpu"]
                else ForwardMode.DECODE
            )
            # first input layernorm
            if not input_is_sharded or not is_after_eagle3_layer_capture:
                if residual is None:
                    residual = hidden_states
                    hidden_states = self.input_layernorm[0](hidden_states)
                else:
                    hidden_states, residual = self.input_layernorm[0](
                        hidden_states, residual
                    )

            # first attn
            hidden_states = self.self_attn[0](
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                comm_manager=self.mlp_branch_decoder_comm_manager[0],
                block_scale=input_block_scale if is_after_eagle3_layer_capture else None,
                can_run_flashinfer_fusion=use_fused_comm if is_after_eagle3_layer_capture else False
            )

            hidden_states_mlp = None
            block_scale_for_dense = None
            moe_input_hidden_states = None

            if use_fused_comm:
                # moe_input_hidden_states is partial (reduce-scattered pattern) bf16 input for router gemm.
                # hidden_states_mlp is block-wise fp8
                hidden_states_mlp , residual, block_scale_for_dense, moe_input_hidden_states = self.post_attention_layernorm[0].forward_with_allreduce_fusion(
                    get_attention_tp_group(),
                    hidden_states,
                    residual,
                    fuse_block_quant_fp8=not self.dense_0_unqaunted,
                    residual_reduce_scattered=input_is_sharded,
                    trigger_completion_at_end=False,
                    has_partial_norm_out=True
                )
            else:
                hidden_states, residual = (
                    self.moe_branch_decoder_comm_manager.post_attn_comm(
                        hidden_states, residual, tp_num_tokens
                    )
                )
                # first post attention layernorm
                hidden_states, residual = self.post_attention_layernorm[0](
                    hidden_states, residual
                )
                hidden_states_mlp = hidden_states
                moe_input_hidden_states = hidden_states

            # ################################## overlapped period ##################################

            if deepep_mode == ForwardMode.DECODE:

                # overlap stage 1 : router + dispatch send | first mlp dense + second attn pre
                moe_origin_input = None
                # router + dispatch send
                self.alt_stream.wait_stream(current_stream)
                with torch.cuda.stream(self.alt_stream):
                    hidden_states_moe = moe_input_hidden_states.clone()

                    router_logits = self.mlp.router(hidden_states_moe)
                    topk_out = self.mlp.topk(
                        hidden_states_moe, router_logits
                    )
                    expert_indices, expert_scales = topk_out.topk_ids, topk_out.topk_weights

                    moe_origin_input = hidden_states_moe

                    self._dispatch_a(
                        hidden_states_moe, expert_indices, expert_scales, deepep_mode
                    )

                # first mlp dense
                if not use_fused_comm:
                    hidden_states_mlp, _, _ = self.mlp_branch_decoder_comm_manager[0]. \
                        pre_mlp_comm(hidden_states_mlp.clone(), forward_batch, tp_num_tokens)
                hidden_states_mlp = self.mlps[0](hidden_states_mlp, block_scale=block_scale_for_dense)
                block_scale = None
                if use_fused_comm if is_after_eagle3_layer_capture else False:
                    hidden_states_mlp, residual, block_scale = self.input_layernorm[1].forward_with_reducescatter_fusion(
                        get_attention_tp_group(), hidden_states_mlp, residual, fuse_block_quant_fp8=not self.attn_1_unquanted
                    )
                else:
                    hidden_states_mlp, residual = self.mlp_branch_decoder_comm_manager[0]. \
                        post_mlp_comm(hidden_states_mlp, residual, tp_num_tokens, forward_batch)
                    
                    # second input layernorm
                    hidden_states_mlp, residual = self.input_layernorm[1](
                        hidden_states_mlp, residual
                    )

                # second attn pre
                attn = self.self_attn[1]
                comm_manager = self.mlp_branch_decoder_comm_manager[1]
                if attn.no_absorb(forward_batch):
                    q, k, v = attn.forward_normal_chunked_kv_prepare(
                        positions, hidden_states_mlp, forward_batch, comm_manager, block_scale)
                else:
                    Q, K = attn.forward_absorb_qkv_proj(
                        hidden_states_mlp, positions, forward_batch, comm_manager, block_scale, can_run_flashinfer_fusion=use_fused_comm if is_after_eagle3_layer_capture else False)

                # overlap stage 2 : dispatch recv + moe gemm + combine send | second attn post + second mlp dense
                self.alt_stream.wait_stream(current_stream)
                # second attn post
                attn = self.self_attn[1]
                comm_manager = self.mlp_branch_decoder_comm_manager[1]
                if attn.no_absorb(forward_batch):
                    hidden_states_mlp = attn.forward_normal_chunked_kv_core(q, k, v, comm_manager, forward_batch)
                else:
                    hidden_states_mlp = attn.forward_absorb_attn_o_proj(Q, K, forward_batch, comm_manager)

                block_scale  = None

                if use_fused_comm if is_after_eagle3_layer_capture else False:
                    hidden_states_mlp, residual, block_scale, _ = \
                        self.post_attention_layernorm[1].forward_with_allreduce_fusion(
                            get_attention_tp_group(),
                            hidden_states_mlp,
                            residual,
                            fuse_block_quant_fp8=not self.dense_1_unqaunted,
                            residual_reduce_scattered=True
                        )
                else:
                    hidden_states_mlp, residual = self.mlp_branch_decoder_comm_manager[1]. \
                        post_attn_comm(hidden_states_mlp, residual, tp_num_tokens)
                    # second_post_attention_layernorm
                    hidden_states_mlp, residual = self.post_attention_layernorm[1](hidden_states_mlp, residual)

                # dispatch recv + moe gemm + combine send
                with torch.cuda.stream(self.alt_stream):
                    (
                        (hidden_states_moe, input_scales),
                        topk_idx,
                        topk_weights,
                        _,  # reorder_topk_ids,
                        num_recv_tokens_per_expert_list,
                        _,  # seg_indptr,
                        masked_m,
                    ) = self._dispatch_b()

                    hidden_states_moe = self.forward_routed_experts(
                        num_global_tokens,
                        deepep_mode,
                        hidden_states_moe,
                        input_scales,
                        topk_idx,
                        topk_weights,
                        num_recv_tokens_per_expert_list,
                        masked_m,
                    )
                    # [TODO] ensure issue before the second mlp dense
                    self._combine_a(hidden_states_moe, topk_idx, topk_weights, deepep_mode, moe_origin_input)

                # second mlp dense
                if not (use_fused_comm if is_after_eagle3_layer_capture else False):
                    hidden_states_mlp, _, _ = self.mlp_branch_decoder_comm_manager[1]. \
                        pre_mlp_comm(hidden_states_mlp, forward_batch, tp_num_tokens)
                hidden_states_mlp = self.mlps[1](hidden_states_mlp, block_scale)

                can_run_next_flashinfer_fusion=(use_fused_comm and next_layer_input_norm is not None) if is_eagle3_layer_capture else False
                if not can_run_next_flashinfer_fusion:
                    hidden_states_mlp, residual = self.mlp_branch_decoder_comm_manager[1]. \
                        post_mlp_comm(hidden_states_mlp, residual, tp_num_tokens, forward_batch)

                # step 4: combine recv
                self.alt_stream.wait_stream(current_stream)
                with torch.cuda.stream(self.alt_stream):
                    hidden_states_moe = self._combine_b()
                current_stream.wait_stream(self.alt_stream)
                output_block_scale = None
                if can_run_next_flashinfer_fusion:
                    hidden_states, residual, output_block_scale = next_layer_input_norm.forward_with_reducescatter_fusion(
                        get_dense_tp_group(), hidden_states_mlp, residual, fuse_block_quant_fp8=next_layer_attn_0_quanted, add_in=hidden_states_moe,
                    )
                else:
                    hidden_states = hidden_states_mlp + hidden_states_moe

                return hidden_states, residual, output_block_scale
            
            else:

                # global MEMORY_THRESHOLD_GB
                # if MEMORY_THRESHOLD_GB == 0:
                #     MEMORY_THRESHOLD_GB = set_memory_threshold(self.topk, hidden_states.shape[-1])
                # check_and_clear_cache(MEMORY_THRESHOLD_GB)

                # overlap stage 1: router | ag
                moe_origin_input = None
                self.alt_stream.wait_stream(current_stream)
                # router
                with torch.cuda.stream(self.alt_stream):
                
                    hidden_states_moe = hidden_states.clone()

                    router_logits = self.mlp.router(hidden_states_moe)
                    topk_out = self.mlp.topk(
                        hidden_states_moe, router_logits
                    )
                    expert_indices, expert_scales = topk_out.topk_ids, topk_out.topk_weights

                    moe_origin_input = hidden_states_moe

                    if(
                        self.config.zero_expert_type is not None
                        and hidden_states_moe.shape[0] > 0
                    ):
                        normal_expert_mask = expert_indices >= self.mlp.num_experts
                        expert_indices[normal_expert_mask] = -1
                        if self.config.zero_expert_type == "copy":
                            expert_scales[normal_expert_mask] = 1.0
                        if self.config.zero_expert_type == "drop":
                            expert_indices[normal_expert_mask] = 0.0
                # ag
                hidden_states_mlp, _, _ = self.mlp_branch_decoder_comm_manager[0]. \
                    pre_mlp_comm(hidden_states.clone(), forward_batch, tp_num_tokens)
                
                # overlap stage 2: dispatch | dense gemm
                self.alt_stream.wait_stream(current_stream)
                # first mlp dense
                with torch.cuda.stream(self.alt_stream):
                    # dispatch
                    self._dispatch_a(
                        hidden_states_moe, expert_indices, expert_scales, forward_mode=deepep_mode
                    )
                    (
                        (hidden_states_moe, input_scales),
                        recv_topk_idx,
                        recv_topk_weights,
                        _,  # reorder_topk_ids,
                        num_recv_tokens_per_expert_list,
                        _,  # seg_indptr,
                        _,
                    ) = self._dispatch_b()
                hidden_states_mlp = self.mlps[0](hidden_states_mlp)
                
                # overlap stage 3: moe gemm | rs + ag + second attn + rs + ag
                self.alt_stream.wait_stream(current_stream)
                # moe gemm
                with torch.cuda.stream(self.alt_stream):
                    hidden_states_moe = self.forward_routed_experts(
                        num_global_tokens,
                        deepep_mode,
                        hidden_states_moe,
                        input_scales,
                        recv_topk_idx,
                        recv_topk_weights,
                        num_recv_tokens_per_expert_list,
                        None
                    )

                # rs
                hidden_states_mlp, residual = self.mlp_branch_decoder_comm_manager[0]. \
                    post_mlp_comm(hidden_states_mlp, residual, tp_num_tokens, forward_batch)
                # second input layernorm
                hidden_states_mlp, residual = self.input_layernorm[1](
                    hidden_states_mlp, residual
                )
                # ag + second attn
                hidden_states_mlp = self.self_attn[1](
                    positions=positions,
                    hidden_states=hidden_states_mlp,
                    forward_batch=forward_batch,
                    comm_manager=self.mlp_branch_decoder_comm_manager[1]
                )
                # rs
                hidden_states_mlp, residual = self.mlp_branch_decoder_comm_manager[1]. \
                    post_attn_comm(hidden_states_mlp, residual, tp_num_tokens)
                # second post attention layernorm
                hidden_states_mlp, residual = self.post_attention_layernorm[1](
                    hidden_states_mlp, residual
                )
                # ag
                hidden_states_mlp, _, _ = self.mlp_branch_decoder_comm_manager[1]. \
                    pre_mlp_comm(hidden_states_mlp, forward_batch, tp_num_tokens)
                
                # overlap stage 4: combine | dense gemm + rs
                self.alt_stream.wait_stream(current_stream)
                # combine
                with torch.cuda.stream(self.alt_stream):
                    self._combine_a(
                        hidden_states_moe, expert_indices.to(torch.int64), (expert_scales.to(torch.float32), recv_topk_weights), deepep_mode, moe_origin_input
                    )
                    hidden_states_moe = self._combine_b()
                # second mlp dense
                hidden_states_mlp = self.mlps[1](hidden_states_mlp)
                
                # rs
                hidden_states_mlp, residual = self.mlp_branch_decoder_comm_manager[1]. \
                    post_mlp_comm(hidden_states_mlp, residual, tp_num_tokens, forward_batch)

                current_stream.wait_stream(self.alt_stream)
                hidden_states = hidden_states_mlp + hidden_states_moe

                return hidden_states, residual, None
        else:
            moe_hidden_states = self.forward_mlp(
                self.moe_branch_decoder_comm_manager,
                self.mlp,
                hidden_states,
                residual,
                forward_batch,
                num_global_tokens,
                max_num_tokens_per_gpu,
                tp_num_tokens
            )
            if (
                global_server_args_dict["attn_parallel_strategy"] == AttnParallelStrategy.DATA_PARALLEL
                and
                global_server_args_dict["dense_parallel_strategy"] == DenseParallelStategy.TENSOR_PARALLEL
            ):
                hidden_states, residual = self.forward_mlp_branch(self.mlp_branch_decoder_comm_manager, positions, hidden_states, residual, forward_batch, num_global_tokens, max_num_tokens_per_gpu, tp_num_tokens)
                # do emtpy tensor calculation, not None
                hidden_states = hidden_states + moe_hidden_states
            else:
                hidden_states = moe_hidden_states

            return hidden_states, residual, None
            
    def _dispatch_a(self, hidden_states, topk_idx, tok_weights, forward_mode):
        self.deepep_dispatcher.dispatch_a(
            hidden_states, topk_idx, tok_weights, forward_mode
        )

    def _dispatch_b(self):
        return self.deepep_dispatcher.dispatch_b()

    def _combine_a(self, hidden_states, topk_idx, tok_weights, forward_mode, moe_origin_input):
        self.deepep_dispatcher.combine_a(
            hidden_states, topk_idx, tok_weights, forward_mode, moe_origin_input
        )

    def _combine_b(self):
        return self.deepep_dispatcher.combine_b()

    def forward_routed_experts(
        self,
        num_global_tokens,
        deepep_forward_mode,
        hidden_states,
        input_scales,
        topk_idx,
        topk_weights,
        num_recv_tokens_per_expert_list,
        masked_m,
    ):
        not_quant = self.quant_config is None or should_ignore_quant_layer(
            prefix=self.mlp.experts.prefix,
            ignored_layers=getattr(self.quant_config, "ignored_layers", [])
        )
        executor = DeepExecutor(
                self.mlp.experts.w13_weight,
                self.mlp.experts.w13_weight_scale_inv if not not_quant else None,
                self.mlp.experts.w2_weight,
                self.mlp.experts.w2_weight_scale_inv if not not_quant else None,
            )

        if deepep_forward_mode == ForwardMode.DECODE:
            expected_m = max(
                1,
                num_global_tokens
                * self.mlp.topk.topk_config.top_k
                // self.mlp.num_experts,
            )
            num_tokens_hint = max(1, num_global_tokens * self.mlp.topk.topk_config.top_k // get_tensor_model_parallel_world_size())
            routed_experts_output = executor.forward_low_latency(
                hidden_states, input_scales, masked_m, expected_m, num_tokens_hint
            )
        else:
            routed_experts_output = executor.forward_normal(
                hidden_states,
                input_scales,
                topk_idx,
                topk_weights,
                num_recv_tokens_per_expert_list,
            )
        return routed_experts_output

    def _init_deepep_dispatcher(self):
        config = EPConfig(
            top_k=self.mlp.topk.topk_config.top_k,
            num_experts=self.mlp.num_experts,
            low_latency_max_num_tokens_per_gpu=global_server_args_dict[
                "low_latency_max_num_tokens_per_gpu"
            ],
            max_num_tokens_per_gpu=1024,
            hidden_size=self.mlp.hidden_size,
            rank=get_tensor_model_parallel_rank(),
            world_size=get_tensor_model_parallel_world_size(),
            group=get_world_group().device_group,
            params_dtype=torch.bfloat16,
        )
        return DeepEPDispatcher(config)


class FLASHForCausalLMOverlap(FLASHForCausalLM):
    def __init__(
        self, config: FLASHConfig, quant_config: Optional[QuantizationConfig] = None
    ) -> None:
        super().__init__(config, quant_config)

        for layer in self.model.layers:
            # monkey patch
            layer.__class__ = FLASHDecoderLayerOverlap
            layer.deepep_dispatcher = layer._init_deepep_dispatcher()
            layer.config = config
            layer.global_rank = torch.distributed.get_rank()

EntryClass = [FLASHForCausalLMOverlap]
