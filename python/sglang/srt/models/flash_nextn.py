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

"""Inference-only DeepSeek NextN Speculative Decoding."""
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from sglang.srt.utils import add_prefix, bind_or_assign
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsProcessor

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from sglang.srt.layers.dp_attention import get_attention_tp_group, get_attention_tp_size, get_attention_tp_rank
from sglang.srt.env import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek_v2 import DeepseekV2MLP, DeepseekV2AttentionMLA, DeepseekV3ForCausalLM, DecoderCommMananger
from sglang.srt.models.deepseek_v32 import DeepseekV32MLA
from sglang.srt.models.longcat_flash import FLASHDecoderLayer as FlASHScmoeDecoderLayer
from sglang.srt.utils import is_hip
from sglang.srt.layers.dense.gemms.fp8.fp8_utils import block_dequant
from sglang.srt.layers.moe.layouts.mapping import make_expert_params_mapping
from sglang.srt.configs.model_config import is_dsa

is_hip_ = is_hip()

class FlASHDenseDecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        is_nextn: bool = False,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.enable_dp_attention = global_server_args_dict["enable_dp_attention"]
        self.is_longcat_flash = config.model_type == "flash"

        if self.enable_dp_attention:
            self.tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = get_tensor_model_parallel_world_size()
            self.tp_group = get_tp_group()

        AttnImpl = DeepseekV32MLA if is_dsa(config) else DeepseekV2AttentionMLA

        self.self_attn = AttnImpl(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=(
                config.q_lora_rank if hasattr(config, "q_lora_rank") else None
            ),
            kv_lora_rank=config.kv_lora_rank,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            layer_id=layer_id,
            prefix=add_prefix("self_attn", prefix),
            reduce_attn_results=False,
            alt_stream=alt_stream,
        )

        self.mlp = DeepseekV2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.ffn_hidden_size if hasattr(config, "ffn_hidden_size") else config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.attn_tp_group = get_attention_tp_group()
        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()
        self.layer_id = layer_id
        self.decoder_comm_manager = DecoderCommMananger(
            layer_id=self.layer_id,
            attn_parallel_strategy=global_server_args_dict["attn_parallel_strategy"],
            dense_parallel_strategy=global_server_args_dict["dense_parallel_strategy"],
            moe_parallel_strategy=global_server_args_dict["moe_parallel_strategy"],
            is_moe_layer=False,
            num_layers=config.num_hidden_layers
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        tp_num_tokens: int
    ) -> torch.Tensor:
        if not forward_batch.forward_mode.is_idle():
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)

            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                comm_manager=self.decoder_comm_manager
            )

            hidden_states, residual = self.decoder_comm_manager.post_attn_comm(
                hidden_states, residual, tp_num_tokens)

            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )

            hidden_states = self.forward_mlp(
                hidden_states,
                residual,
                forward_batch,
                tp_num_tokens,
            )
        else:
            hidden_states = self.forward_mlp(
                hidden_states,
                residual,
                forward_batch,
                tp_num_tokens,
            )
        return hidden_states, residual

    def input_layer_norm_fn(self, hidden_states, residual):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        return hidden_states, residual

    def forward_mlp(
        self,
        hidden_states,
        residual,
        forward_batch,
        tp_num_tokens,
    ):
        hidden_states, start_idx, end_idx = self.decoder_comm_manager.pre_mlp_comm(
            hidden_states, forward_batch, tp_num_tokens
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states, residual = self.decoder_comm_manager.post_mlp_comm(
            hidden_states, residual, tp_num_tokens, forward_batch
        )
        if start_idx is not None and end_idx is not None:
            hidden_states = hidden_states[start_idx:end_idx]
        return hidden_states

class FLASHModelNextN(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.nextn_use_scmoe = config.nextn_use_scmoe
        self.vocab_size = config.vocab_size
        if global_server_args_dict["draft_use_oe"]:
            self.use_over_embedding = True
            self.embed_tokens=None
        else:
            self.use_over_embedding = False
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                enable_tp=not global_server_args_dict["enable_dp_attention"],
            )

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.eh_proj = ReplicatedLinear(
            2 * config.hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("eh_proj", "")
        )
        self.alt_stream = torch.cuda.Stream()
        if self.nextn_use_scmoe:
            self.decoder = FlASHScmoeDecoderLayer(config, 0, quant_config=quant_config, alt_stream=self.alt_stream)
        else:
            self.decoder=FlASHDenseDecoderLayer(
                config, 0, quant_config=quant_config, is_nextn=True, alt_stream=self.alt_stream
            )

        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if getattr(config, 'use_over_embedding', False):
            self.enable_over_embedding = config.use_over_embedding
        else:
            self.enable_over_embedding = False

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            if self.use_over_embedding:
                hidden_states=self.embed_tokens(input_ids, forward_batch, is_draft=True)
            else:
                hidden_states=self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        if hidden_states.shape[0] > 0:
            hidden_states, _ = self.eh_proj(
                torch.cat(
                    (
                        self.enorm(hidden_states),
                        self.hnorm(forward_batch.spec_info.hidden_states),
                    ),
                    dim=-1,
                )
            )

        residual=None
        tp_num_tokens=hidden_states.shape[0]
        forward_batch.tp_num_tokens=tp_num_tokens
        hidden_states, residual=self.decoder(
            positions, hidden_states, forward_batch, residual, tp_num_tokens=tp_num_tokens
        )

        if not forward_batch.forward_mode.is_idle():
            hidden_states, _=self.final_layernorm(hidden_states, residual)
            if self.nextn_use_scmoe:
                hidden_states, _=self.decoder.mlp_branch_decoder_comm_manager[1].post_final_norm_comm(hidden_states,
                                                                                                      residual,
                                                                                                      tp_num_tokens)
            else:
                hidden_states, _=self.decoder.decoder_comm_manager.post_final_norm_comm(hidden_states, residual,
                                                                                        tp_num_tokens)
        return hidden_states


class FLASHForCausalLMNextN(DeepseekV3ForCausalLM):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.nextn_use_scmoe = config.nextn_use_scmoe
        self.quant_config = None if "mtp" in getattr(config, "disable_quant_module", []) else quant_config
        self.model = FLASHModelNextN(config, self.quant_config)

        if global_server_args_dict["enable_dp_attention"]:
            self.lm_head = ReplicatedLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
            )
            self.logits_processor = LogitsProcessor(config, skip_all_gather=True)
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=self.quant_config,
            )
            self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        if hasattr(self.config, "num_nextn_predict_layers"):
            num_nextn_layers = self.config.num_nextn_predict_layers
            assert num_nextn_layers == 1, "Only 1 nextn layer is supportted"
            assert num_nextn_layers == self.config.num_hidden_layers
        else:
            num_nextn_layers = 1

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0, None),
            ("gate_up_proj", "up_proj", 1, None),
        ]

        if not is_dsa(self.config):
            stacked_params_mapping.extend([
                ("fused_qkv_a_proj_with_mqa", "q_a_proj", None, 0),
                ("fused_qkv_a_proj_with_mqa", "kv_a_proj_with_mqa", None, self.config.q_lora_rank),
            ])
        else:
            # change stack order
            stacked_params_mapping.extend([
                ("fused_qkv_a_proj_with_mqa", "q_a_proj", None, 0),
                ("fused_qkv_a_proj_with_mqa", "indexer.wk", None, self.config.q_lora_rank),
                ("fused_qkv_a_proj_with_mqa", "kv_a_proj_with_mqa", None, self.config.q_lora_rank+self.config.index_head_dim),
            ])

        def no_tp_load(param, loaded_weight, *args):
            return default_weight_loader(param, loaded_weight)

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts if hasattr(self.config, "n_routed_experts") else self.config.num_experts[0],
        )

        nextn_layer_prefix = "model.layers.0"
        nextn_spec_weight_names = [
            "shared_head.norm",
            "eh_proj",
            "enorm",
            "hnorm",
        ]
        if self.config.model_type == "flash":
            nextn_spec_weight_names.append("final_layernorm")

        new_to_old_names_mapping = {
            "model.mtp.embed_tokens.weight": "embed_tokens.weight",
            "model.mtp.layers.0.eh_proj.weight": "eh_proj.weight",
            "model.mtp.layers.0.eh_proj.weight_scale_inv": "eh_proj.weight_scale_inv",
            "model.mtp.layers.0.eh_proj.weight_scale": "eh_proj.weight_scale",
            "model.mtp.layers.0.enorm.m.weight": "enorm.weight",
            "model.mtp.layers.0.hnorm.m.weight": "hnorm.weight",
            "model.mtp.layers.0.input_layernorm.weight": "layers.0.input_layernorm.weight",
            "model.mtp.layers.0.post_attention_layernorm.weight": "layers.0.post_attention_layernorm.weight",
            "model.mtp.layers.0.self_attn.kv_a_layernorm.weight": "layers.0.self_attn.kv_a_layernorm.weight",
            "model.mtp.layers.0.self_attn.kv_a_proj_with_mqa.weight": "layers.0.self_attn.kv_a_proj_with_mqa.weight",
            "model.mtp.layers.0.self_attn.kv_a_proj_with_mqa.weight_scale_inv": "layers.0.self_attn.kv_a_proj_with_mqa.weight_scale_inv",
            "model.mtp.layers.0.self_attn.kv_a_proj_with_mqa.weight_scale": "layers.0.self_attn.kv_a_proj_with_mqa.weight_scale",
            "model.mtp.layers.0.self_attn.kv_b_proj.weight": "layers.0.self_attn.kv_b_proj.weight",
            "model.mtp.layers.0.self_attn.kv_b_proj.weight_scale_inv": "layers.0.self_attn.kv_b_proj.weight_scale_inv",
            "model.mtp.layers.0.self_attn.kv_b_proj.weight_scale": "layers.0.self_attn.kv_b_proj.weight_scale",
            "model.mtp.layers.0.self_attn.o_proj.weight": "layers.0.self_attn.o_proj.weight",
            "model.mtp.layers.0.self_attn.o_proj.weight_scale_inv": "layers.0.self_attn.o_proj.weight_scale_inv",
            "model.mtp.layers.0.self_attn.o_proj.weight_scale": "layers.0.self_attn.o_proj.weight_scale",
            "model.mtp.layers.0.self_attn.q_a_layernorm.weight": "layers.0.self_attn.q_a_layernorm.weight",
            "model.mtp.layers.0.self_attn.q_a_proj.weight": "layers.0.self_attn.q_a_proj.weight",
            "model.mtp.layers.0.self_attn.q_a_proj.weight_scale_inv": "layers.0.self_attn.q_a_proj.weight_scale_inv",
            "model.mtp.layers.0.self_attn.q_a_proj.weight_scale": "layers.0.self_attn.q_a_proj.weight_scale",
            "model.mtp.layers.0.self_attn.q_b_proj.weight": "layers.0.self_attn.q_b_proj.weight",
            "model.mtp.layers.0.self_attn.q_b_proj.weight_scale_inv": "layers.0.self_attn.q_b_proj.weight_scale_inv",
            "model.mtp.layers.0.self_attn.q_b_proj.weight_scale": "layers.0.self_attn.q_b_proj.weight_scale",
            "model.mtp.layers.0.transformer_layer.mlp.down_proj.weight": "layers.0.mlp.down_proj.weight",
            "model.mtp.layers.0.transformer_layer.mlp.down_proj.weight_scale_inv": "layers.0.mlp.down_proj.weight_scale_inv",
            "model.mtp.layers.0.transformer_layer.mlp.down_proj.weight_scale": "layers.0.mlp.down_proj.weight_scale",
            "model.mtp.layers.0.transformer_layer.mlp.gate_proj.weight": "layers.0.mlp.gate_proj.weight",
            "model.mtp.layers.0.transformer_layer.mlp.gate_proj.weight_scale_inv": "layers.0.mlp.gate_proj.weight_scale_inv",
            "model.mtp.layers.0.transformer_layer.mlp.gate_proj.weight_scale": "layers.0.mlp.gate_proj.weight_scale",
            "model.mtp.layers.0.transformer_layer.mlp.up_proj.weight": "layers.0.mlp.up_proj.weight",
            "model.mtp.layers.0.transformer_layer.mlp.up_proj.weight_scale_inv": "layers.0.mlp.up_proj.weight_scale_inv",
            "model.mtp.layers.0.transformer_layer.mlp.up_proj.weight_scale": "layers.0.mlp.up_proj.weight_scale",
            "model.mtp.layers.0.norm.weight": "layers.0.final_layernorm.weight",
            "model.mtp.norm.weight": "layers.0.final_layernorm.weight",
        }
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if ".mtp." not in name and global_server_args_dict["draft_model_path_use_base"]:
                continue
            if name in new_to_old_names_mapping:
                name = new_to_old_names_mapping[name]
            if name.startswith("layers.0"):
                name = "model." + name
            if name.startswith("enorm") or name.startswith("hnorm") or name.startswith("eh_proj"):
                name = nextn_layer_prefix + "." + name
            if not name.startswith(nextn_layer_prefix):
                continue

            # Use shared head and embed weights from target model
            if "shared_head.head" in name or "embed_tokens" in name:
                continue

            is_decoder = True
            # For nextn specific weights
            for weight_name in nextn_spec_weight_names:
                if weight_name in name:
                    name = name.replace(nextn_layer_prefix, "model")
                    is_decoder = False
                    break
            # For decoder layer weights
            if is_decoder:
                name = name.replace(nextn_layer_prefix, "model.decoder")

            if "rotary_emb.inv_freq" in name:
                continue
            for param_name, weight_name, shard_id, begin_size in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(
                    param, "weight_loader", no_tp_load
                )
                if begin_size is not None and name.endswith(".weight_scale_inv"):
                    begin_size = begin_size // self.config.quantization_config["weight_block_size"][0]
                if "fused_qkv_a_proj_with_mqa" in name:
                    weight_loader(param, loaded_weight, shard_id, begin_size)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if "mlp.experts." in name:
                    for mapping in expert_params_mapping:
                        param_name, weight_name, local_expert_id, shard_id = mapping
                        if weight_name not in name:
                            continue
                        name = name.replace(weight_name, param_name)
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        weight_loader(
                            param,
                            loaded_weight,
                            shard_id=shard_id,
                            local_expert_id=local_expert_id,
                        )
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)

        self.post_load_weights()

    def post_load_weights(self):
        if self.nextn_use_scmoe:
            for i in range(2):
                self_attn = self.model.decoder.self_attn[i]
                if hasattr(self.quant_config, "weight_block_size") and self_attn.kv_b_proj.weight.dtype in (
                    torch.float8_e4m3fn,
                    torch.float8_e4m3fnuz,
                ):
                    weight_block_size = self.quant_config.weight_block_size
                    if weight_block_size is not None:
                        assert hasattr(self_attn.kv_b_proj, "weight_scale_inv")
                        dtype = torch.get_default_dtype()
                        w = block_dequant(
                            self_attn.kv_b_proj.weight,
                            self_attn.kv_b_proj.weight_scale_inv,
                            weight_block_size
                        ).to(dtype)
                else:
                    w = self_attn.kv_b_proj.weight

                w_kc, w_vc = w.unflatten(
                    0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
                ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)

                self_attn.w_kc = bind_or_assign(
                    self_attn.w_kc, w_kc.transpose(1, 2).contiguous().transpose(1, 2)
                )
                self_attn.w_vc = bind_or_assign(
                    self_attn.w_vc, w_vc.contiguous().transpose(1, 2)
                )
                if self.config.mla_scale_q_lora:
                    self_attn.q_a_layernorm.weight.data *= (self.config.hidden_size / self.config.q_lora_rank) ** 0.5
                if self.config.mla_scale_kv_lora:
                    self_attn.kv_a_layernorm.weight.data *= (self.config.hidden_size / self.config.kv_lora_rank) ** 0.5
        else:
            self_attn = self.model.decoder.self_attn
            if hasattr(self.quant_config, "weight_block_size") and self_attn.kv_b_proj.weight.dtype in (
                torch.float8_e4m3fn,
                torch.float8_e4m3fnuz,
            ):
                weight_block_size = self.quant_config.weight_block_size
                if weight_block_size is not None:
                    dtype = torch.get_default_dtype()
                    w = block_dequant(
                        self_attn.kv_b_proj.weight,
                        self_attn.kv_b_proj.weight_scale_inv,
                        weight_block_size
                    ).to(dtype)
                else:
                    w = self_attn.kv_b_proj.weight
            else:
                w = self_attn.kv_b_proj.weight
            w_kc, w_vc = w.unflatten(
                0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
            ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
            self_attn.w_kc = bind_or_assign(
                self_attn.w_kc, w_kc.transpose(1, 2).contiguous().transpose(1, 2)
            )
            self_attn.w_vc = bind_or_assign(
                self_attn.w_vc, w_vc.contiguous().transpose(1, 2)
            )
            if self.config.mla_scale_q_lora:
                self_attn.q_a_layernorm.weight.data *= (self.config.hidden_size / self.config.q_lora_rank) ** 0.5
            if self.config.mla_scale_kv_lora:
                self_attn.kv_a_layernorm.weight.data *= (self.config.hidden_size / self.config.kv_lora_rank) ** 0.5

EntryClass = [FLASHForCausalLMNextN]
