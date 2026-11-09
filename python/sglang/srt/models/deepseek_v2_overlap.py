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

# Adapted from:
# https://github.com/vllm-project/vllm/blob/fb6af8bc086328ca6659e72d11ffd4309ce4de22/vllm/model_executor/models/deepseek_v2.py
"""Inference-only DeepseekV2 model."""

from typing import Optional

import torch
from transformers import PretrainedConfig

from sglang.srt.env import global_server_args_dict
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode, MicroBatches
from sglang.srt.models import deepseek_v2
from sglang.srt.utils import add_prefix, get_colorful_logger, is_npu, is_sm90_supported

_is_npu = is_npu()

if not _is_npu and is_sm90_supported():
    # from sglang.srt.layers.moe.ep_moe.deep import DeepEPMode
    from deep_ep import Buffer

    from sglang.srt.tbo.tbo_executor import Orchestration, TBODeepEPDispatchers, TBOExecutor, tbo_stage

logger = get_colorful_logger(__name__)


class DeepseekV2Model(deepseek_v2.DeepseekV2Model):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)
        self.first_k_dense_replace = config.first_k_dense_replace
        self.tbo_executors = self._init_tbo_executors()
        self.tbo_deep_dispatchers = None  # lazy init

    def _init_tbo_executors(self) -> "dict[DeepEPMode, TBOExecutor]":  # noqa: F821
        orchestration_low_latency_mode = self.orchestrate_low_latency_mode()
        orchestration_normal_mode = self.orchestrate_normal_mode()
        tbo_executors = {
            DeepEPMode.low_latency: TBOExecutor(orchestration_low_latency_mode, self.layers[self.first_k_dense_replace :]),  # noqa: F821
            DeepEPMode.normal: TBOExecutor(orchestration_normal_mode, self.layers[self.first_k_dense_replace :]),  # noqa: F821
        }
        return tbo_executors

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        micro_batches: Optional[MicroBatches] = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        tp_num_tokens = hidden_states.shape[0]
        residual = None

        hidden_states, residual = self.forward_normal(hidden_states, residual, positions, forward_batch, tp_num_tokens, 0, self.first_k_dense_replace)

        if not forward_batch.can_run_tbo:
            hidden_states, residual = self.forward_normal(
                hidden_states, residual, positions, forward_batch, tp_num_tokens, self.first_k_dense_replace, len(self.layers)
            )
            if not forward_batch.forward_mode.is_idle():
                hidden_states, _ = self.norm(hidden_states, residual)
                hidden_states, _ = self.layers[-1].decoder_comm_manager.post_final_norm_comm(hidden_states, residual, tp_num_tokens)
        else:
            tbo_executor = self._get_tbo_executor(tp_num_tokens, forward_batch)
            hidden_states = tbo_executor.forward_overlap(
                micro_batches=micro_batches,
                hidden_states=hidden_states,
                residual=residual,
                positions=positions,
                forward_batch=forward_batch,
                tp_num_tokens=tp_num_tokens,
                layers=self.layers[self.first_k_dense_replace :],
            )

        return hidden_states

    def forward_normal(self, hidden_states, residual, positions, forward_batch, tp_num_tokens, layer_id_start, layer_id_end):
        for i in range(layer_id_start, layer_id_end):
            layer = self.layers[i]
            hidden_states, residual = layer(positions, hidden_states, forward_batch, residual, tp_num_tokens=tp_num_tokens)
        return hidden_states, residual

    def _get_tbo_executor(self, tp_num_tokens, forward_batch) -> "TBOExecutor":
        _, max_num_tokens_per_gpu = forward_batch.get_num_tokens(tp_num_tokens)
        if max_num_tokens_per_gpu // 2 > global_server_args_dict["low_latency_max_num_tokens_per_gpu"]:
            return self.tbo_executors[DeepEPMode.normal]  # noqa: F821
        else:
            return self.tbo_executors[DeepEPMode.low_latency]  # noqa: F821

    def tbo_pre_process(self, micro_batches: MicroBatches, hidden_states, residual, positions, forward_batch, tp_num_tokens, layers):
        hidden_states, residual = layers[0].input_layer_norm_fn(hidden_states, residual)
        hidden_states = layers[0].decoder_comm_manager.pre_attn_comm(hidden_states, tp_num_tokens)
        residual = layers[0].decoder_comm_manager.pre_attn_comm(residual, tp_num_tokens)

        split_idx = micro_batches.token_split_index
        hidden_states_a = hidden_states[:split_idx]
        hidden_states_b = hidden_states[split_idx:]
        residual_a = residual[:split_idx]
        residual_b = residual[split_idx:]
        micro_batches[0].hidden_states = hidden_states_a
        micro_batches[0].residual = residual_a
        micro_batches[1].hidden_states = hidden_states_b
        micro_batches[1].residual = residual_b
        return [
            {
                "micro_batch": micro_batches[0],
            },
            {
                "micro_batch": micro_batches[1],
            },
        ]

    def tbo_post_process(self, context, layers):
        hidden_states_a = context[0]["hidden_states"]
        residual_a = context[0]["residual"]
        hidden_states_b = context[1]["hidden_states"]
        residual_b = context[1]["residual"]

        hidden_states_a, residual_a = self.norm(hidden_states_a, residual_a)
        hidden_states_b, residual_b = self.norm(hidden_states_b, residual_b)

        last_layer = layers[-1]
        micro_batch_a = context[0]["micro_batch"]
        micro_batch_b = context[1]["micro_batch"]
        hidden_states_a, _ = last_layer.decoder_comm_manager.post_final_norm_comm(hidden_states_a, residual_a, micro_batch_a.tp_num_tokens)
        hidden_states_b, _ = last_layer.decoder_comm_manager.post_final_norm_comm(hidden_states_b, residual_b, micro_batch_b.tp_num_tokens)
        hidden_states = torch.cat([hidden_states_a, hidden_states_b], dim=0)
        return hidden_states

    def orchestrate_low_latency_mode(self):
        @tbo_stage(provides=["q_nope_out", "k_nope", "q_pe", "k_pe", "residual"])
        def stage_1(micro_batch, layer, hidden_states=None, residual=None, first_run=True):
            """
            input_norm, qkv
            """
            hidden_states = micro_batch.hidden_states if hidden_states is None else hidden_states
            residual = micro_batch.residual if residual is None else residual
            if not first_run:
                tp_num_tokens = micro_batch.tp_num_tokens
                hidden_states, residual = layer.input_layer_norm_fn(hidden_states, residual)
                hidden_states = layer.decoder_comm_manager.pre_attn_comm(hidden_states, tp_num_tokens)

            positions = micro_batch.forward_batch.positions
            q_nope_out, k_nope, q_pe, k_pe = layer.self_attn.forward_absorb_qkv_proj(hidden_states, positions)

            return q_nope_out, k_nope, q_pe, k_pe, residual

        @tbo_stage(provides=["hidden_states", "residual", "expert_indices", "expert_scales", "first_run"])
        def stage_2(q_nope_out, k_nope, q_pe, k_pe, residual, micro_batch, layer, first_run=True):
            """
            attn, o_proj, post_norm, gate
            """
            hidden_states = layer.self_attn.forward_absorb_attn_o_proj(q_nope_out, k_nope, q_pe, k_pe, micro_batch.forward_batch)

            hidden_states, residual = layer.decoder_comm_manager.post_attn_comm_for_tbo(
                hidden_states, residual, micro_batch.tp_num_tokens, is_first_tbo_attn=first_run
            )

            hidden_states, residual = layer.post_attention_layernorm(hidden_states, residual)

            router_logits = layer.mlp.gate(hidden_states)
            expert_indices, expert_scales = layer.mlp.experts.select_routed_experts(hidden_states, router_logits)

            first_run = False
            return hidden_states, residual, expert_indices, expert_scales, first_run

        @tbo_stage(provides=["shared_experts_output"])
        def stage_3(hidden_states, expert_indices, expert_scales, micro_batch, layer):
            """
            dispatch_send, shared experts
            """
            self.dispatch_send(hidden_states, expert_indices, expert_scales, ForwardMode.DECODE, micro_batch.index, layer)
            shared_experts_output = layer.mlp.experts.forward_shared_experts(hidden_states)
            return shared_experts_output

        @tbo_stage()
        def stage_4(micro_batch, layer):
            """
            dispatch_recv, routed experts, combine_send
            """
            (hidden_states, input_scales, topk_idx, topk_weights, num_recv_tokens_per_expert_list, masked_m) = self.dispatch_recv(micro_batch.index)
            routed_experts_output = self.forward_routed_experts(
                micro_batch, ForwardMode.DECODE, hidden_states, input_scales, topk_idx, topk_weights, num_recv_tokens_per_expert_list, masked_m, layer
            )
            self.combine_send(routed_experts_output, topk_idx, topk_weights, ForwardMode.DECODE, micro_batch.index)

        @tbo_stage(provides=["hidden_states"])
        def stage_5(shared_experts_output, micro_batch, layer):
            """
            combine_recv
            """
            hidden_states = self.combine_recv(micro_batch.index)
            hidden_states = hidden_states + shared_experts_output
            return hidden_states

        return Orchestration(
            stages=[stage_1, stage_2, stage_3, stage_4, stage_5], delta_stage=2, pre_process=self.tbo_pre_process, post_process=self.tbo_post_process
        )

    def orchestrate_normal_mode(self):
        @tbo_stage(provides=["attn_output", "residual", "first_run"])
        def stage_1(micro_batch, layer, residual=None, attn_output=None, first_run=True):
            """
            input_norm, attn, post_norm, gate, dispatch_send
            """
            if first_run:
                hidden_states = micro_batch.hidden_states
                residual = micro_batch.residual
            else:
                shared_experts_output = layer.mlp.experts.forward_shared_experts(attn_output)
                hidden_states = self.combine_recv(micro_batch.index)
                hidden_states = hidden_states + shared_experts_output

                hidden_states, residual = layer.input_layer_norm_fn(hidden_states, residual)
                tp_num_tokens = micro_batch.tp_num_tokens
                hidden_states = layer.decoder_comm_manager.pre_attn_comm(hidden_states, tp_num_tokens)

            # attn
            positions = micro_batch.forward_batch.positions
            attn_output = layer.self_attn(positions, hidden_states, micro_batch.forward_batch)

            attn_output, residual = layer.decoder_comm_manager.post_attn_comm_for_tbo(
                attn_output, residual, micro_batch.tp_num_tokens, is_first_tbo_attn=first_run
            )
            attn_output, residual = layer.post_attention_layernorm(hidden_states, residual)

            # gate, select experts
            router_logits = layer.mlp.gate(attn_output)
            expert_indices, expert_scales = layer.mlp.experts.select_routed_experts(attn_output, router_logits)

            # dispatch send
            self.dispatch_send(attn_output, expert_indices, expert_scales, ForwardMode.EXTEND, micro_batch.index, layer)

            first_run = False
            return attn_output, residual, first_run

        @tbo_stage(provides=["hidden_states"])
        def stage_2(micro_batch, layer, attn_output=None):
            """
            dispatch_recv, routed experts, combine_send
            """
            (hidden_states, input_scales, topk_idx, topk_weights, num_recv_tokens_per_expert_list, masked_m) = self.dispatch_recv(micro_batch.index)

            routed_experts_output = self.forward_routed_experts(
                micro_batch, ForwardMode.EXTEND, hidden_states, input_scales, topk_idx, topk_weights, num_recv_tokens_per_expert_list, masked_m, layer
            )

            self.combine_send(routed_experts_output, topk_idx, topk_weights, ForwardMode.EXTEND, micro_batch.index)

            if layer.layer_id == len(self.layers) - 1:
                shared_experts_output = layer.mlp.experts.forward_shared_experts(attn_output)
                hidden_states = self.combine_recv(micro_batch.index)
                hidden_states = hidden_states + shared_experts_output
                return hidden_states

            return attn_output

        device_properties = torch.cuda.get_device_properties(device="cuda")
        total_num_sms = device_properties.multi_processor_count
        deep_gemm_num_sms = total_num_sms - Buffer.num_sms
        return Orchestration(
            stages=[stage_1, stage_2],
            delta_stage=0,
            pre_process=self.tbo_pre_process,
            post_process=self.tbo_post_process,
            deep_gemm_num_sms=deep_gemm_num_sms,
        )

    def dispatch_send(self, hidden_states, expert_indices, expert_scales, forward_mode, index, layer):
        if self.tbo_deep_dispatchers is None:
            experts = layer.mlp.experts
            self.tbo_deep_dispatchers = TBODeepEPDispatchers(experts.top_k, experts.num_experts, experts.hidden_size)
        self.tbo_deep_dispatchers[index].dispatch_a(hidden_states, expert_indices, expert_scales, forward_mode)

    def dispatch_recv(self, index):
        (
            (hidden_states, input_scales),
            topk_idx,
            topk_weights,
            _,  # reorder_topk_ids,
            num_recv_tokens_per_expert_list,
            _,  # seg_indptr,
            masked_m,
        ) = self.tbo_deep_dispatchers[index].dispatch_b()
        return hidden_states, input_scales, topk_idx, topk_weights, num_recv_tokens_per_expert_list, masked_m

    def combine_send(self, hidden_states, expert_indices, expert_scales, forward_mode, index):
        self.tbo_deep_dispatchers[index].combine_a(hidden_states, expert_indices, expert_scales, forward_mode)

    def combine_recv(self, index):
        return self.tbo_deep_dispatchers[index].combine_b()

    def forward_routed_experts(
        self, micro_batch, deepep_forward_mode, hidden_states, input_scales, topk_idx, topk_weights, num_recv_tokens_per_expert_list, masked_m, layer
    ):
        executor = layer.mlp.experts.construct_executor()

        if deepep_forward_mode == ForwardMode.DECODE:
            forward_batch = micro_batch.forward_batch
            tp_num_tokens = micro_batch.tp_num_tokens
            num_global_tokens, _ = forward_batch.get_num_tokens(tp_num_tokens)
            expected_m = max(1, num_global_tokens * layer.mlp.experts.top_k // layer.mlp.experts.num_experts)
            routed_experts_output = executor.forward_low_latency(hidden_states, input_scales, masked_m, expected_m)
        else:
            routed_experts_output = executor.forward_normal(hidden_states, input_scales, topk_idx, topk_weights, num_recv_tokens_per_expert_list)
        return routed_experts_output


class DeepseekV2ForCausalLMOverlap(deepseek_v2.DeepseekV2ForCausalLM):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        model = DeepseekV2Model(config, quant_config, prefix=add_prefix("model", prefix))
        super().__init__(config, model, quant_config, prefix)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        micro_batches: Optional[MicroBatches] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch, micro_batches)
        return self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch)


class DeepseekV3ForCausalLMOverlap(DeepseekV2ForCausalLMOverlap):
    pass


EntryClass = [DeepseekV2ForCausalLMOverlap, DeepseekV3ForCausalLMOverlap]
