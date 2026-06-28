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
from transformers import PretrainedConfig

from sglang.srt.layers.moe.layouts.mapping import make_expert_params_mapping
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsProcessor

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.env import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek_v2 import DeepseekV2DecoderLayer, DeepseekV3ForCausalLM
from sglang.srt.utils import is_hip
from sglang.srt.layers.dense.gemms.fp8.fp8_utils import block_dequant
from sglang.srt.layers.utils import (
    FLLM_IS_CP, CP_METADATA, cp_split_and_rebuild_data, cp_all_gather_rerange_output
)
from sglang.srt.configs.model_config import is_dsa

is_hip_ = is_hip()


class DeepseekModelNextN(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            enable_tp=not global_server_args_dict["enable_dp_attention"],
        )

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)

        self.alt_stream = torch.cuda.Stream()
        self.decoder = DeepseekV2DecoderLayer(
            config, 0, quant_config=quant_config, is_nextn=True, alt_stream=self.alt_stream
        )

        self.shared_head = nn.Module()
        self.shared_head.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        hidden_states = self.eh_proj(
            torch.cat(
                (
                    self.enorm(hidden_states),
                    self.hnorm(forward_batch.spec_info.hidden_states),
                ),
                dim=-1,
            )
        )

        residual = None
        tp_num_tokens = hidden_states.shape[0]
        forward_batch.tp_num_tokens = tp_num_tokens
        if CP_METADATA:
            hidden_states = cp_split_and_rebuild_data(hidden_states, CP_METADATA.value.split_list, CP_METADATA.value.zigzag_index)
            positions = cp_split_and_rebuild_data(positions, CP_METADATA.value.split_list, CP_METADATA.value.zigzag_index)
        hidden_states, residual = self.decoder(
            positions, hidden_states, forward_batch, residual, tp_num_tokens=tp_num_tokens
        )

        if not forward_batch.forward_mode.is_idle():
            hidden_states, _ = self.shared_head.norm(hidden_states, residual)
            if not FLLM_IS_CP:
                hidden_states, _ = self.decoder.decoder_comm_manager.post_final_norm_comm(hidden_states, residual, tp_num_tokens)
        if CP_METADATA:
            hidden_states = cp_all_gather_rerange_output(hidden_states, CP_METADATA.value, self.decoder.decoder_comm_manager.attn_rsag)
        return hidden_states


class DeepseekV3ForCausalLMNextN(DeepseekV3ForCausalLM):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.quant_config = quant_config

        self.model = DeepseekModelNextN(config, quant_config)

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
                quant_config=quant_config,
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
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Fuse q_a_proj and kv_a_proj_with_mqa along output dimension when q_lora_rank is not None
        fuse_qkv_a_proj = hasattr(self.config, "q_lora_rank") and (
            self.config.q_lora_rank is not None
        )
        cached_a_proj = {} if fuse_qkv_a_proj else None

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, local_expert_id, shard_id)
        expert_params_mapping = make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
        )

        nextn_spec_weight_names = [
            "shared_head.norm",
            "eh_proj",
            "enorm",
            "hnorm",
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if hasattr(self.config, "num_nextn_predict_layers"):
                num_nextn_layers = self.config.num_nextn_predict_layers
                assert num_nextn_layers == 1, "Only 1 nextn layer is supportted"
                nextn_layer_prefix = "model.layers.0"
                if num_nextn_layers != self.config.num_hidden_layers:
                    if name.startswith("model.layers"):
                        name_list = name.split(".")
                        if (
                            len(name_list) >= 3
                            and int(name_list[2]) >= self.config.num_hidden_layers
                        ):
                            nextn_layer_prefix = "model.layers." + str(name_list[2])
                        else:
                            continue
                if not name.startswith(nextn_layer_prefix):
                    continue
            else:
                raise ValueError("num_nextn_predict_layers is not in the config")

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
            for param_name, weight_name, shard_id in stacked_params_mapping:
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
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if ("mlp.experts." in name):
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
                    
                    if (
                        fuse_qkv_a_proj and (
                            "q_a_proj" in name 
                            or "kv_a_proj_with_mqa" in name
                            or "indexer.wk" in name
                        )
                    ):
                        quant_block_size = self.quant_config.weight_block_size[0]
                        begin_size_mp = {
                            "q_a_proj": 0,
                            "indexer_wk": self.config.q_lora_rank,
                            "kv_a_proj_with_mqa": self.config.q_lora_rank,
                        }
                        if is_dsa(self.config):
                            begin_size_mp["kv_a_proj_with_mqa"] += self.config.index_head_dim

                        if "q_a_proj" in name:
                            param = params_dict[name.replace("q_a_proj", "fused_qkv_a_proj_with_mqa")]
                            weight_loader = param.weight_loader
                            begin_size = begin_size_mp["q_a_proj"]
                        elif "kv_a_proj_with_mqa" in name:
                            param = params_dict[name.replace("kv_a_proj_with_mqa", "fused_qkv_a_proj_with_mqa")]
                            weight_loader = param.weight_loader
                            begin_size = begin_size_mp["kv_a_proj_with_mqa"]
                        elif "indexer.wk" in name:
                            param = params_dict[name.replace("indexer.wk", "fused_qkv_a_proj_with_mqa")]
                            weight_loader = param.weight_loader
                            begin_size = begin_size_mp["indexer_wk"]
                        if 'scale_inv' in name:
                            begin_size //= quant_block_size
                        weight_loader(param, loaded_weight, begin_size=begin_size)
                    else:
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
        self.post_load_weights()

    def post_load_weights(self):
        self_attn = self.model.decoder.self_attn
        if hasattr(self.quant_config, "weight_block_size") and (self.quant_config.weight_block_size is not None) and self_attn.kv_b_proj.weight.dtype in (
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        ):
            weight_block_size = self.quant_config.weight_block_size
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
        self_attn.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
        self_attn.w_vc = w_vc.contiguous().transpose(1, 2)


EntryClass = [DeepseekV3ForCausalLMNextN]
