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
# Adapted from
# https://github.com/SafeAILab/EAGLE/blob/main/eagle/model/cnets.py
"""Inference-only LLaMA-EAGLE model compatible with HuggingFace weights."""

from collections.abc import Iterable
from typing import Optional

import torch
from torch import nn
from transformers import LlamaConfig

from sglang.srt.env import global_server_args_dict
from sglang.srt.layers.embedding.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.sampler.logits_processor import LogitsProcessor
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.llama import LlamaDecoderLayer, LlamaForCausalLM
from sglang.srt.ops.layernorm import RMSNorm
from sglang.srt.utils import add_prefix, get_colorful_logger

logger = get_colorful_logger(__name__)


class LlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(
        self,
        config: LlamaConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, layer_id, quant_config, prefix)

        # Skip the input_layernorm
        # https://github.com/SafeAILab/EAGLE/blob/35c78f6cdc19a73e05cf5c330b4c358dad970c6a/eagle/model/cnets.py#L427
        # if layer_id == 0:
        #     del self.input_layernorm
        #     setattr(self, "input_layernorm", lambda x: x)


class LlamaModel(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            enable_tp=not global_server_args_dict["enable_dp_attention"],
        )

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.eh_proj = ReplicatedLinear(
            2 * config.hidden_size, config.hidden_size, bias=False, quant_config=quant_config, prefix=add_prefix("eh_proj", "")
        )

        self.decoder = LlamaDecoderLayer(config, 0, quant_config=quant_config)

        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids) if input_embeds is None else input_embeds

        hidden_states, _ = self.eh_proj(torch.cat((self.enorm(hidden_states), self.hnorm(forward_batch.spec_info.hidden_states)), dim=-1))

        residual = None
        # tp_num_tokens = hidden_states.shape[0]
        hidden_states, residual = self.decoder(
            # positions, hidden_states, forward_batch, residual, tp_num_tokens=tp_num_tokens
            positions,
            hidden_states,
            forward_batch,
            residual,
        )

        if not forward_batch.forward_mode.is_idle():
            hidden_states, _ = self.final_layernorm(hidden_states, residual)
            # hidden_states, _ = self.decoder.decoder_comm_manager.post_final_norm_comm(hidden_states, residual, tp_num_tokens)
        return hidden_states


class LlamaForCausalLMNextN(LlamaForCausalLM):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.quant_config = quant_config
        self.model = LlamaModel(config, quant_config=quant_config)

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
        return self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        print("load_weights_eagle")

        if hasattr(self.config, "num_nextn_predict_layers"):
            num_nextn_layers = self.config.num_nextn_predict_layers
            assert num_nextn_layers == 1, "Only 1 nextn layer is supportted"

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        new_to_old_names_mapping = {
            # "model.mtp.embed_tokens.weight": "model.embed_tokens.weigh",
            "model.mtp.norm.weight": "model.final_layernorm.weight",
            "model.mtp.layers.0.enorm.m.weight": "model.enorm.weight",
            "model.mtp.layers.0.hnorm.m.weight": "model.hnorm.weight",
            "model.mtp.layers.0.eh_proj.weight": "model.eh_proj.weight",
            "model.mtp.layers.0.eh_proj.weight_scale_inv": "model.eh_proj.weight_scale_inv",
            "model.mtp.layers.0.input_layernorm.weight": "model.decoder.input_layernorm.weight",
            "model.mtp.layers.0.post_attention_layernorm.weight": "model.decoder.post_attention_layernorm.weight",
            "model.mtp.layers.0.self_attn.q_norm.weight": "model.decoder.self_attn.q_norm.weight",
            "model.mtp.layers.0.self_attn.k_norm.weight": "model.decoder.self_attn.k_norm.weight",
            "model.mtp.layers.0.self_attn.q_proj.weight": "model.decoder.self_attn.q_proj.weight",
            "model.mtp.layers.0.self_attn.q_proj.weight_scale_inv": "model.decoder.self_attn.q_proj.weight_scale_inv",
            "model.mtp.layers.0.self_attn.k_proj.weight": "model.decoder.self_attn.k_proj.weight",
            "model.mtp.layers.0.self_attn.k_proj.weight_scale_inv": "model.decoder.self_attn.k_proj.weight_scale_inv",
            "model.mtp.layers.0.self_attn.v_proj.weight": "model.decoder.self_attn.v_proj.weight",
            "model.mtp.layers.0.self_attn.v_proj.weight_scale_inv": "model.decoder.self_attn.v_proj.weight_scale_inv",
            "model.mtp.layers.0.self_attn.o_proj.weight": "model.decoder.self_attn.o_proj.weight",
            "model.mtp.layers.0.self_attn.o_proj.weight_scale_inv": "model.decoder.self_attn.o_proj.weight_scale_inv",
            "model.mtp.layers.0.transformer_layer.mlp.down_proj.weight": "model.decoder.mlp.down_proj.weight",
            "model.mtp.layers.0.transformer_layer.mlp.down_proj.weight_scale_inv": "model.decoder.mlp.down_proj.weight_scale_inv",
            "model.mtp.layers.0.transformer_layer.mlp.gate_proj.weight": "model.decoder.mlp.gate_proj.weight",
            "model.mtp.layers.0.transformer_layer.mlp.gate_proj.weight_scale_inv": "model.decoder.mlp.gate_proj.weight_scale_inv",
            "model.mtp.layers.0.transformer_layer.mlp.up_proj.weight": "model.decoder.mlp.up_proj.weight",
            "model.mtp.layers.0.transformer_layer.mlp.up_proj.weight_scale_inv": "model.decoder.mlp.up_proj.weight_scale_inv",
        }

        # named_parameters
        """
            'model.embed_tokens.weight'
            'model.enorm.weight'
            'model.hnorm.weight'
            'model.eh_proj.weight'
            'model.decoder.self_attn.q_norm.weight'
            'model.decoder.self_attn.k_norm.weight'
            'model.decoder.self_attn.qkv_proj.weight'
            'model.decoder.self_attn.o_proj.weight'
            'model.decoder.mlp.gate_up_proj.weight'
            'model.decoder.mlp.down_proj.weight'
            'model.decoder.input_layernorm.weight'
            'model.decoder.post_attention_layernorm.weight'
            'model.final_layernorm.weight'
            'lm_head.weight'
        """

        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            if ".mtp." not in name:
                continue
            # Use shared head and embed weights from target model
            if "shared_head.head" in name or "embed_tokens" in name:
                continue
            if name in new_to_old_names_mapping:
                name = new_to_old_names_mapping[name]

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
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
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


class MeituanQwen3ForCausalLMNextN(LlamaForCausalLMNextN):
    pass


class LongcatLiteForCausalLMNextN(LlamaForCausalLMNextN):
    pass


EntryClass = [
    LlamaForCausalLMNextN,
    # meituan
    MeituanQwen3ForCausalLMNextN,
    LongcatLiteForCausalLMNextN,
]
