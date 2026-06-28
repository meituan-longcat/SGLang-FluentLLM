# Adapted from
# https://github.com/SafeAILab/EAGLE/blob/main/eagle/model/cnets.py
"""Inference-only LLaMA-EAGLE model compatible with HuggingFace weights."""

from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig

from sglang.srt.distributed import (
    get_ep_group,
    get_attn_tp_group,
    get_attn_tp_world_size,
    GroupCoordinator
)
from sglang.srt.layers.dense.mlp import ParallelMLP
from sglang.srt.layers.dp_attention import (
    get_mlp_tp_group,
    get_draft_attention_tp_group,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import QKVParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.npu_moe.ep_metadata import EPMetadata
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader

from sglang.srt.models.llama import LlamaAttention, LlamaDecoderLayer, LlamaForCausalLM
from sglang.srt.utils import add_prefix, get_colorful_logger

from sglang.srt.managers.schedule_batch import global_server_args_dict

logger = get_colorful_logger(__name__)


@torch.inference_mode()
def mlp_tp_recuce_scatter(hidden_states: torch.Tensor,
                          ep_metadata: EPMetadata, ep_group: GroupCoordinator, mlp_tp_group: GroupCoordinator, attn_tp_group: Optional[GroupCoordinator] = None
                          ) ->torch.Tensor:
    if attn_tp_group is None:
        attn_tp_group = get_attn_tp_group()

    if attn_tp_group.world_size == mlp_tp_group.world_size or attn_tp_group.world_size == 1:
        return mlp_tp_group.all_reduce(hidden_states)

    if ep_group is None:
        ep_group = get_ep_group()

    hidden_states = hidden_states.view(-1, hidden_states.size(-1))
    bs, h = hidden_states.shape

    ep_world_query_lens = ep_metadata.ep_world_query_lens
    mlp_tp_in_ep_start = ep_group.rank_in_group // mlp_tp_group.world_size * mlp_tp_group.world_size
    mlp_tp_in_ep_end = (ep_group.rank_in_group // mlp_tp_group.world_size + 1) *  mlp_tp_group.world_size
    mlp_tp_world_query_lens = ep_world_query_lens[mlp_tp_in_ep_start: mlp_tp_in_ep_end]
    padded_hs, partial_hs, \
        partial_global_padded_hs, global_padded_hs_trans, \
        global_padded_hs = ep_metadata.get_all_gather_buffer()

    mlp_tp_global_padded_hs = global_padded_hs.view(-1, *global_padded_hs.shape[-2:])[:mlp_tp_group.world_size]
    # test mlp_tp_recuce_scatter,
    # mlp_tp_world_query_lens=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1],
    # hidden_states.shape=torch.Size([66, 6144]),
    # mlp_tp_global_padded_hs.shape=torch.Size([16, 6, 6144])
    if not ep_metadata.no_padding:
        pos_start = 0
        for i, l in enumerate(mlp_tp_world_query_lens):
            mlp_tp_global_padded_hs[i, :l, :] = hidden_states[pos_start : pos_start + l, :]
            pos_start += l
    else:
        # need copy?
        mlp_tp_global_padded_hs.copy_(hidden_states.view(mlp_tp_group.world_size, -1, h))

    up_dim = (ep_metadata.decode_only_across_ep and padded_hs.dim() == 2 and mlp_tp_global_padded_hs.dim() == 3) or (mlp_tp_group.world_size == 1)
    if up_dim:
        padded_hs = padded_hs.view(1, padded_hs.shape[-2], padded_hs.shape[-1])
    mlp_tp_group.reduce_scatter_tensor(padded_hs, mlp_tp_global_padded_hs)
    if up_dim:
        padded_hs = padded_hs.view(padded_hs.shape[-2], padded_hs.shape[-1])

    if not ep_metadata.no_padding:
        hidden_states = padded_hs[:ep_world_query_lens[ep_group.rank_in_group]]
    else:
        # need clone?
        hidden_states = padded_hs.clone()

    return hidden_states


def mlp_tp_recuce_scatter_in_graph_mode(hidden_states: torch.Tensor, mlp_tp_group: GroupCoordinator) ->torch.Tensor:
    if get_attn_tp_world_size() == mlp_tp_group.world_size:
        return mlp_tp_group.all_reduce(hidden_states)
    hidden_states = hidden_states.view(-1, hidden_states.size(-1))
    return mlp_tp_group.reduce_scatter(hidden_states)


def mlp_tp_allgather_in_graph_mode(hidden_states: torch.Tensor, mlp_tp_group: GroupCoordinator) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if get_attn_tp_world_size() == mlp_tp_group.world_size:
        return hidden_states
    hidden_states = hidden_states.view(-1, hidden_states.size(-1))
    hidden_states = mlp_tp_group.all_gather(hidden_states, 0)
    hidden_states = hidden_states.squeeze(0)
    return hidden_states


# attn -> mlp, attn dp > mlp dp: allgather, attn dp < mlp dp: split
@torch.inference_mode()
def mlp_tp_allgather(hidden_states: torch.Tensor,
                     forward_batch: ForwardBatch, ep_group: Optional[GroupCoordinator] = None, mlp_tp_group: Optional[GroupCoordinator] = None, attn_tp_group: Optional[GroupCoordinator] = None
                     ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if ep_group is None:
        ep_group = get_ep_group()
    if forward_batch.can_run_with_graph:
        return mlp_tp_allgather_in_graph_mode(hidden_states, mlp_tp_group)
    ep_metadata = forward_batch.ep_metadata
    attn_tp_word_size = get_attn_tp_world_size()
    if attn_tp_group != None:
        attn_tp_word_size = attn_tp_group.world_size
    if get_attn_tp_world_size() == mlp_tp_group.world_size or attn_tp_group.world_size == 1:
        return hidden_states
    # TODO: keep output shape the same with input
    hidden_states = hidden_states.view(-1, hidden_states.size(-1))
    bs, h = hidden_states.shape

    ep_world_query_lens = ep_metadata.ep_world_query_lens
    mlp_tp_in_ep_start = ep_group.rank_in_group // mlp_tp_group.world_size * mlp_tp_group.world_size
    mlp_tp_in_ep_end = (ep_group.rank_in_group // mlp_tp_group.world_size + 1) *  mlp_tp_group.world_size
    mlp_tp_world_query_lens = ep_world_query_lens[mlp_tp_in_ep_start: mlp_tp_in_ep_end]
    padded_hs, partial_hs, \
        partial_global_padded_hs, global_padded_hs_trans, \
        global_padded_hs = ep_metadata.get_all_gather_buffer()

    if not ep_metadata.no_padding:
        if bs != 0:
            padded_hs[0:bs] = hidden_states
    else:
        # need copy?
        if bs != 0:
            padded_hs.copy_(hidden_states)

    mlp_tp_global_padded_hs = global_padded_hs.view(-1, *global_padded_hs.shape[-2:])[:mlp_tp_group.world_size]

    mlp_tp_group.all_gather_into_tensor(mlp_tp_global_padded_hs, padded_hs)

    if not ep_metadata.no_padding:
        hidden_states = torch.cat([mlp_tp_global_padded_hs[i, :l] for i, l in enumerate(mlp_tp_world_query_lens)], dim=0)
    else:
        # need clone?
        hidden_states = mlp_tp_global_padded_hs.clone().view(-1, h)

    hidden_states = hidden_states.squeeze(0)

    return hidden_states


class LlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(
        self,
        config: LlamaConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, layer_id, quant_config, prefix)

        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        rope_is_neox_style = getattr(config, "rope_is_neox_style", True)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False
        )
        self.self_attn = LlamaAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            rope_is_neox_style=rope_is_neox_style,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            bias=attention_bias,
            outside_tp_group=get_draft_attention_tp_group(),
        )

        # override qkv
        self.self_attn.qkv_proj = QKVParallelLinear(
            2 * self.hidden_size,
            self.self_attn.head_dim,
            self.self_attn.total_num_heads,
            self.self_attn.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
            outside_tp_group=get_draft_attention_tp_group(),
        )

        if config.model_type == "llama4_text":
            inter_size = config.intermediate_size_mlp
        else:
            inter_size = config.intermediate_size

        self.mlp = ParallelMLP(
            config.hidden_size, inter_size, config.hidden_act, quant_config, prefix=f"{prefix}.mlp",
        )

        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        tp_num_tokens: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        residual = hidden_states
        embeds = self.input_layernorm(embeds)
        hidden_states = self.hidden_norm(hidden_states)

        hidden_states = torch.cat([embeds, hidden_states], dim=-1)

        # will skip attn
        if hidden_states.shape[0] == 0:
            hidden_states = embeds

        # Self Attention
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        hidden_states = mlp_tp_allgather(hidden_states, forward_batch, None, get_mlp_tp_group(), get_draft_attention_tp_group())
        hidden_states = self.mlp(hidden_states)
        if forward_batch.can_run_with_graph:
            hidden_states = mlp_tp_recuce_scatter_in_graph_mode(hidden_states, get_mlp_tp_group())
        else:
            hidden_states = mlp_tp_recuce_scatter(hidden_states, forward_batch.ep_metadata, None, get_mlp_tp_group(), get_draft_attention_tp_group())

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
            tp_num_tokens
        )

        hidden_states_to_logits, hidden_states_to_aux = self.norm(
            hidden_states, residual
        )

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
