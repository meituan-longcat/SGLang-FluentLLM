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

"""Inference-only Longcat-EAGLE3 model compatible with HuggingFace weights."""

from collections.abc import Iterable
from typing import Optional

import torch
from torch import nn
from transformers import LlamaConfig

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from sglang.srt.env import global_server_args_dict
from sglang.srt.layers.attention.dp_attention import get_attention_tp_group, get_attention_tp_rank, get_attention_tp_size
from sglang.srt.layers.embedding.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.sampler.logits_processor import LogitsProcessor
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.deepseek_v2 import DecoderCommMananger, DeepseekV2AttentionMLA, DeepseekV2MLP, DeepseekV3ForCausalLM
from sglang.srt.ops.layernorm import RMSNorm
from sglang.srt.utils import add_prefix


class LongcatDecoderLayerEagle3NextN(nn.Module):
    """Single decoder layer combining Longcat MLA attention and MLP for EAGLE-3 inference."""

    def __init__(
        self,
        config: LlamaConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.enable_dp_attention = global_server_args_dict["enable_dp_attention"]

        if self.enable_dp_attention:
            self.tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = get_tensor_model_parallel_world_size()
            self.tp_group = get_tp_group()

        # Attention layer - input is concatenated [embeds, hidden_states] so input size is 2 * hidden_size
        self.self_attn = DeepseekV2AttentionMLA(
            config=config,
            hidden_size=2 * self.hidden_size,  # Input is concatenated
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=getattr(config, "qk_nope_head_dim", 64),
            qk_rope_head_dim=getattr(config, "qk_rope_head_dim", 64),
            v_head_dim=getattr(config, "v_head_dim", 128),
            q_lora_rank=getattr(config, "q_lora_rank", None),
            kv_lora_rank=getattr(config, "kv_lora_rank", 512),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            layer_id=layer_id,
            prefix=add_prefix("self_attn", prefix),
            reduce_attn_results=False,
        )

        # MLP layer
        self.mlp = DeepseekV2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=getattr(config, "intermediate_size", config.hidden_size * 4),
            hidden_act=getattr(config, "hidden_act", "silu"),
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

        # Layer norms
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
            num_layers=1,  # EAGLE-3 has only 1 layer
        )

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            positions: Position indices
            embeds: Input embeddings
            hidden_states: Hidden states from previous layer
            forward_batch: Forward batch info
            residual: Residual connection
        """
        residual = hidden_states
        embeds = self.input_layernorm(embeds)
        hidden_states = self.hidden_norm(hidden_states)

        # Concatenate input embeddings and hidden states as in the original
        hidden_states = torch.cat([embeds, hidden_states], dim=-1)

        # Self Attention
        hidden_states = self.self_attn(
            positions=positions, hidden_states=hidden_states, forward_batch=forward_batch, comm_manager=self.decoder_comm_manager
        )

        hidden_states, residual = self.decoder_comm_manager.post_attn_comm(hidden_states, residual, hidden_states.shape[0])

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # Fully Connected
        hidden_states, start_idx, end_idx = self.decoder_comm_manager.pre_mlp_comm(hidden_states, forward_batch, hidden_states.shape[0])
        hidden_states = self.mlp(hidden_states)
        hidden_states, residual = self.decoder_comm_manager.post_mlp_comm(hidden_states, residual, hidden_states.shape[0], forward_batch)
        if start_idx is not None and end_idx is not None:
            hidden_states = hidden_states[start_idx:end_idx]

        return hidden_states, residual


class LongcatModelEagle3NextN(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )

        # Projection layer for concatenated hidden states (3 layers -> 1 layer)
        if hasattr(config, "target_hidden_size"):
            self.fc = ReplicatedLinear(
                config.target_hidden_size * 3, config.hidden_size, bias=False, quant_config=quant_config, prefix=add_prefix("fc", prefix)
            )
        else:
            self.fc = ReplicatedLinear(
                config.hidden_size * 3, config.hidden_size, bias=False, quant_config=quant_config, prefix=add_prefix("fc", prefix)
            )

        self.midlayer = LongcatDecoderLayerEagle3NextN(config, 0, quant_config, prefix)

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        embeds = self.embed_tokens(input_ids) if input_embeds is None else input_embeds

        # Get hidden states from spec_info and project them
        hidden_states = forward_batch.spec_info.hidden_states
        if hidden_states.shape[-1] != embeds.shape[-1]:
            hidden_states, _ = self.fc(hidden_states)

        residual = None
        hidden_states, residual = self.midlayer(
            positions,
            embeds,
            hidden_states,
            forward_batch,
            residual,
        )

        hidden_states_to_logits, hidden_states_to_aux = self.norm(hidden_states, residual)

        # For draft decode, we capture the hidden state before norm
        return hidden_states_to_logits, [hidden_states_to_aux]


class LongcatForCausalLMEagle3NextN(DeepseekV3ForCausalLM):
    """
    LongcatForCausalLMEagle3NextN model that implements EAGLE-3 speculative decoding
    using DeepSeekV2 MLA attention and MLP components in a Longcat architecture
    optimized for inference.
    """

    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.quant_config = quant_config

        if self.config.num_hidden_layers != 1:
            raise ValueError("EAGLE3 currently only supports 1 layer")

        self.model = LongcatModelEagle3NextN(config, quant_config=quant_config, prefix=add_prefix("model", prefix))

        # Llama 3.2 1B Instruct set tie_word_embeddings to True
        # Llama 3.1 8B Instruct set tie_word_embeddings to False
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.draft_vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )

        self.logits_processor = LogitsProcessor(config)
        self.capture_aux_hidden_states = True

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load weights from checkpoint, handling name mapping and vocab buffers."""
        for name, loaded_weight in weights:
            if "d2t" in name:
                # d2t stores diffs between draft id and target id
                self.hot_token_id = loaded_weight + torch.arange(loaded_weight.shape[0])

            if "d2t" not in name and "t2d" not in name and "lm_head" not in name:
                new_name = f"model.{name}"
                super().load_weights([(new_name, loaded_weight)])
            elif "lm_head" in name:
                super().load_weights([(name, loaded_weight)])

    def get_hot_token_id(self):
        """Get hot token IDs for EAGLE-3 speculative decoding."""
        return self.hot_token_id


EntryClass = [LongcatForCausalLMEagle3NextN]
