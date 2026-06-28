# Adapted from
# https://github.com/SafeAILab/EAGLE/blob/main/eagle/model/cnets.py
"""Inference-only LLaMA-EAGLE model compatible with HuggingFace weights."""

from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig

from sglang.srt.layers.dense.mlp import ParallelMLP
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import QKVParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader

from sglang.srt.models.llama import LlamaDecoderLayer, LlamaForCausalLM
from sglang.srt.utils import add_prefix, get_colorful_logger

from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.layers.dp_attention import get_attention_tp_group, get_dense_tp_group

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

        # override qkv
        self.self_attn.qkv_proj = QKVParallelLinear(
            2 * self.hidden_size,
            self.self_attn.head_dim,
            self.self_attn.total_num_heads,
            self.self_attn.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )

        if config.model_type == "llama4_text":
            inter_size = config.intermediate_size_mlp
        else:
            inter_size = config.intermediate_size

        self.mlp = ParallelMLP(
            config.hidden_size, inter_size, config.hidden_act, quant_config, prefix=f"{prefix}.mlp",
        )

        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward_low_latency(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        tp_num_tokens: int = 0,
        final_norm: RMSNorm = None
    ):
        residual = hidden_states
        embeds = self.input_layernorm(embeds)
        hidden_states = self.hidden_norm(hidden_states)
        hidden_states = torch.cat([embeds, hidden_states], dim=-1)

        # Self Attention
        hidden_states = self.decoder_comm_manager.pre_attn_comm(hidden_states, tp_num_tokens)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        block_scale = None
        hidden_states, residual, block_scale, _ = self.post_attention_layernorm.forward_with_allreduce_fusion(
            get_attention_tp_group(),
            hidden_states,
            residual,
            fuse_block_quant_fp8=not self.mlp.gateup_unquanted
        )
        hidden_states = self.mlp(hidden_states, block_scale)
        hidden_states, residual, _, _ = final_norm.forward_with_allreduce_fusion(
            get_dense_tp_group(),
            hidden_states,
            residual,
            fuse_block_quant_fp8=False
        )
        return hidden_states, residual

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        tp_num_tokens: int = 0,
        final_norm: RMSNorm = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if tp_num_tokens < global_server_args_dict["flashinfer_comm_max_num_tokens"]:
            return self.forward_low_latency(
                positions,
                embeds,
                hidden_states,
                forward_batch,
                residual,
                tp_num_tokens,
                final_norm
            )

        residual = hidden_states
        embeds = self.input_layernorm(embeds)
        hidden_states = self.hidden_norm(hidden_states)

        hidden_states = torch.cat([embeds, hidden_states], dim=-1)

        # Self Attention
        hidden_states = self.decoder_comm_manager.pre_attn_comm(hidden_states, tp_num_tokens)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states, residual = self.decoder_comm_manager.post_attn_comm(hidden_states, residual, tp_num_tokens)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # Fully Connected
        hidden_states, start_idx, end_idx = self.decoder_comm_manager.pre_mlp_comm(
            hidden_states, forward_batch, tp_num_tokens
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states, residual = self.decoder_comm_manager.post_mlp_comm(
            hidden_states, residual, tp_num_tokens, forward_batch
        )

        return hidden_states, residual


class LlamaModel(nn.Module):
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
            enable_tp=not global_server_args_dict["enable_dp_attention"],
        )
        self.midlayer = LlamaDecoderLayer(config, 0, quant_config, prefix)
        self.num_fc_input_dim = len(config.eagle_aux_hidden_state_layer_ids) if hasattr(config, "eagle_aux_hidden_state_layer_ids") else 3
        self.fc = torch.nn.Linear(config.hidden_size * self.num_fc_input_dim, config.hidden_size)

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            embeds = self.embed_tokens(input_ids)
        else:
            embeds = input_embeds

        hidden_states = forward_batch.spec_info.hidden_states
        if hidden_states.shape[-1] != embeds.shape[-1]:
            hidden_states = self.fc(hidden_states)

        # embeds.shape[0] is 0 in idle batch run, set hidden_states eq to embeds
        if embeds.shape[0] == 0:
            hidden_states = embeds.clone()

        residual = None
        tp_num_tokens = hidden_states.shape[0]
        hidden_states, residual = self.midlayer(
            positions,
            embeds,
            hidden_states,
            forward_batch,
            residual,
            tp_num_tokens,
            self.norm
        )
        if tp_num_tokens < global_server_args_dict["flashinfer_comm_max_num_tokens"]:
            hidden_states_to_logits, hidden_states_to_aux = hidden_states, residual
        else:
            hidden_states_to_logits, hidden_states_to_aux = self.norm(
                hidden_states, residual
            )

            hidden_states_to_logits, _ = self.midlayer.decoder_comm_manager.post_final_norm_comm(hidden_states_to_logits, None, tp_num_tokens)
            hidden_states_to_aux, _ = self.midlayer.decoder_comm_manager.post_final_norm_comm(hidden_states_to_aux, None, tp_num_tokens)

        # For draft decode, we capture the hidden state before norm
        return hidden_states_to_logits, [hidden_states_to_aux]


class LlamaForCausalLMEagle3(LlamaForCausalLM):
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

        self.model = LlamaModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
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
        self.hot_token_id = None

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        params_dict = dict(self.named_parameters())
        # Define the parameter mapping for stacked parameters
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        for name, loaded_weight in weights:
            if "d2t" in name:
                # d2t stores diffs between draft id and target id
                self.hot_token_id = loaded_weight + torch.arange(loaded_weight.shape[0])
                continue

            if "t2d" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param_name = f"model.{name}" if name not in params_dict else name
                if param_name in params_dict:
                    param = params_dict[param_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Handle regular parameters
                param_name = name if name in params_dict else f"model.{name}"
                if param_name in params_dict:
                    param = params_dict[param_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)

    def get_hot_token_id(self):
        return self.hot_token_id


EntryClass = [LlamaForCausalLMEagle3]
