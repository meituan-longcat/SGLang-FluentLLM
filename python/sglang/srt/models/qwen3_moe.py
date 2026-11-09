# Adapted from qwen2_moe.py

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


"""Inference-only Qwen3MoE model compatible with HuggingFace weights."""

from collections.abc import Iterable
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.configs import Qwen3MoeConfig
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from sglang.srt.distributed.decoder_comm_manager import DecoderCommMananger
from sglang.srt.distributed.parallel_strategy import DenseParallelStategy, MoeParallelStrategy
from sglang.srt.env import global_server_args_dict
from sglang.srt.layers.attention.dp_attention import (
    get_attention_tp_group,
    get_attention_tp_rank,
    get_attention_tp_size,
    get_dense_tp_group,
)
from sglang.srt.layers.attention.radix_attention import RadixAttention
from sglang.srt.layers.embedding.rotary_embedding import get_rope
from sglang.srt.layers.embedding.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.moe.layouts.mapping import make_expert_params_mapping
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.sampler.logits_processor import LogitsProcessor
from sglang.srt.managers.eplb.expert_distribution import (
    get_global_expert_distribution_recorder,
)
from sglang.srt.managers.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.ops.activation import SiluAndMul
from sglang.srt.ops.layernorm import RMSNorm
from sglang.srt.ops.topk import TopK, TopKOutputFormat
from sglang.srt.utils import LazyValue, add_prefix, get_colorful_logger, is_npu

_is_npu = is_npu()

if not _is_npu:
    from sglang.srt.layers.moe.layer import MoELayer

logger = get_colorful_logger(__name__)


class Qwen3MoeMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.tp_size, self.tp_rank = None, None
        self.layout = global_server_args_dict["dense_parallel_strategy"]
        self.moe_layout = global_server_args_dict["moe_parallel_strategy"]
        if self.layout == DenseParallelStategy.TENSOR_PARALLEL and self.moe_layout == MoeParallelStrategy.TENSOR_PARALLEL:
            self.tp_group = get_dense_tp_group()
            self.gate_up_proj = MergedColumnParallelLinear(
                hidden_size,
                [intermediate_size] * 2,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("gate_up_proj", prefix),
                outside_tp_group=self.tp_group,
            )
            self.down_proj = RowParallelLinear(
                intermediate_size,
                hidden_size,
                bias=False,
                reduce_results=reduce_results,
                quant_config=quant_config,
                prefix=add_prefix("down_proj", prefix),
                outside_tp_group=self.tp_group,
            )
        else:
            self.gate_up_proj = ReplicatedLinear(
                hidden_size,
                intermediate_size * 2,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("gate_up_proj", prefix),
            )
            self.down_proj = ReplicatedLinear(
                intermediate_size,
                hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("down_proj", prefix),
            )

        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        if x.shape[0] == 0:
            return x
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(
        self,
        config: Qwen3MoeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        layer_index: int = -1,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_index = layer_index
        self.tp_size = get_tensor_model_parallel_world_size()

        if self.tp_size > config.num_experts:
            raise ValueError(f"Tensor parallel size {self.tp_size} is greater than the number of experts {config.num_experts}.")

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            params_dtype=torch.float32,
            prefix=add_prefix("gate", prefix),
        )
        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            layer_id=layer_index,
            renormalize=config.norm_topk_prob,
            use_grouped_topk=False,
            output_format=TopKOutputFormat.STANDARD,
        )

        MoEImpl = MoELayer
        self.experts = MoEImpl(
            top_k=config.num_experts_per_tok,
            num_experts=config.num_experts + global_server_args_dict["ep_num_redundant_experts"],
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            layer_index=layer_index,
            prefix=prefix,
        )
        if getattr(config, "shared_expert_intermediate_size", 0) > 0:
            self.shared_expert = Qwen3MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_expert", prefix),
            )
            self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)
        else:
            self.shared_expert = None
            self.shared_expert_gate = None

    def get_moe_routed_weights(self):
        return [x.data for name, x in self.experts.named_parameters() if name not in ["correction_bias"] and "shared_experts" not in name]

    def forward(self, hidden_states: torch.Tensor, num_global_tokens: int, max_num_tokens_per_gpu: int) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        shared_output = None
        if self.shared_expert is not None:
            shared_output = self.shared_expert(hidden_states)
            if hidden_states.shape[0] > 0 and self.shared_expert_gate is not None:
                shared_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_output

        router_logits, _ = self.gate(hidden_states.to(torch.float32))
        topk_output = self.topk(hidden_states, router_logits) if hidden_states.shape[0] > 0 else self.topk.empty_topk_output(hidden_states.device)

        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            topk_output=topk_output,
            num_global_tokens=num_global_tokens,
            max_num_tokens_per_gpu=max_num_tokens_per_gpu,
        )

        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output

        return final_hidden_states.view(num_tokens, hidden_dim)


class Qwen3MoeAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        head_dim: Optional[int] = None,
        rms_norm_eps: float = 1e-06,
        attention_bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()
        self.total_num_heads = num_heads
        assert self.total_num_heads % self.attn_tp_size == 0
        self.num_heads = self.total_num_heads // self.attn_tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= self.attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % self.attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert self.attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.attn_tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            outside_tp_group=get_attention_tp_group(),
            prefix=add_prefix("qkv_proj", prefix),
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            reduce_results=False,
            outside_tp_group=get_attention_tp_group(),
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
        )

        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def _apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q_by_head = q.reshape(-1, self.head_dim).contiguous()
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.reshape(-1, self.head_dim).contiguous()
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        return q, k

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self._apply_qk_norm(q, k)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3MoeConfig,
        layer_id: int,
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

        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        rms_norm_eps = config.rms_norm_eps
        attention_bias = config.attention_bias
        self.self_attn = Qwen3MoeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            attention_bias=attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )

        # Note: Qwen/Qwen2-57B-A14B-Instruct does not have
        # `mlp_only_layers` in the config.
        mlp_only_layers = [] if not hasattr(config, "mlp_only_layers") else config.mlp_only_layers
        if (layer_id not in mlp_only_layers) and (config.num_experts > 0 and (layer_id + 1) % config.decoder_sparse_step == 0):
            self.mlp = Qwen3MoeSparseMoeBlock(
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                layer_index=layer_id,
            )
            self.is_moe_layer = True
        else:
            self.mlp = Qwen3MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
            self.is_moe_layer = False

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
            is_moe_layer=self.is_moe_layer,
            num_layers=config.num_hidden_layers,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        tp_num_tokens: int,
    ) -> torch.Tensor:

        num_global_tokens, max_num_tokens_per_gpu = forward_batch.get_num_tokens(tp_num_tokens)

        if not forward_batch.forward_mode.is_idle():
            # Self Attention
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)
            hidden_states = self.decoder_comm_manager.pre_attn_comm(hidden_states, tp_num_tokens)
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

            hidden_states, residual = self.decoder_comm_manager.post_attn_comm(hidden_states, residual, tp_num_tokens)

            # Fully Connected
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

            hidden_states = self.forward_mlp(hidden_states, residual, forward_batch, num_global_tokens, max_num_tokens_per_gpu, tp_num_tokens)
        else:
            hidden_states = self.forward_mlp(hidden_states, residual, forward_batch, num_global_tokens, max_num_tokens_per_gpu, tp_num_tokens)
        return hidden_states, residual

    def forward_mlp(self, hidden_states, residual, forward_batch, num_global_tokens, max_num_tokens_per_gpu, tp_num_tokens):
        hidden_states, start_idx, end_idx = self.decoder_comm_manager.pre_mlp_comm(hidden_states, forward_batch, tp_num_tokens)
        if isinstance(self.mlp, Qwen3MoeSparseMoeBlock):
            hidden_states = self.mlp(hidden_states, num_global_tokens, max_num_tokens_per_gpu)
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states, residual = self.decoder_comm_manager.post_mlp_comm(hidden_states, residual, tp_num_tokens, forward_batch)
        if start_idx is not None and end_idx is not None:
            hidden_states = hidden_states[start_idx:end_idx]
        return hidden_states


class Qwen3MoeModel(nn.Module):
    def __init__(
        self,
        config: Qwen3MoeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            enable_tp=not global_server_args_dict["enable_dp_attention"],
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.layers = nn.ModuleList(
            [
                Qwen3MoeDecoderLayer(config, layer_id, quant_config=quant_config, prefix=add_prefix(f"layers.{layer_id}", prefix))
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # For EAGLE3 support
        self.layers_to_capture = []

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        tp_num_tokens = hidden_states.shape[0]

        residual = None
        aux_hidden_states = []
        for i in range(len(self.layers)):
            if i in self.layers_to_capture:
                aux_hidden_states.append(hidden_states + residual if residual is not None else hidden_states)
            with get_global_expert_distribution_recorder().with_current_layer(i):
                layer = self.layers[i]
                hidden_states, residual = layer(positions, hidden_states, forward_batch, residual, tp_num_tokens)
        if not forward_batch.forward_mode.is_idle():
            hidden_states, _ = self.norm(hidden_states, residual)
            hidden_states, _ = layer.decoder_comm_manager.post_final_norm_comm(hidden_states, residual, tp_num_tokens)
        if len(aux_hidden_states) == 0:
            return hidden_states

        return hidden_states, aux_hidden_states


class Qwen3MoeForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: Qwen3MoeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = Qwen3MoeModel(config, quant_config, prefix=add_prefix("model", prefix))
        if global_server_args_dict["enable_dp_attention"]:
            self.lm_head = ReplicatedLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                prefix=add_prefix("lm_head", prefix),
            )
            self.logits_processor = LogitsProcessor(config, skip_all_gather=True)
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
            self.logits_processor = LogitsProcessor(config)
        self._routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: layer.mlp.get_moe_routed_weights()
                for layer_id, layer in enumerate(self.model.layers)
                if isinstance(layer.mlp, Qwen3MoeDecoderLayer)
            }
        )
        self.capture_aux_hidden_states = False

    @property
    def routed_experts_weights_of_layer(self):
        return self._routed_experts_weights_of_layer.value

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
            aux_hidden_states,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params_mapping = make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
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
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
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
                    if name not in params_dict:
                        continue

                    if name in params_dict:
                        param = params_dict[name]
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, loaded_weight)
                    else:
                        logger.warning(f"Parameter {name} not found in params_dict")

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[list[int]] = None):
        self.capture_aux_hidden_states = True
        if layer_ids is None:
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [
                2,
                num_layers // 2,
                num_layers - 3,
            ]  # Specific layers for EAGLE3 support
        else:
            self.model.layers_to_capture = [val + 1 for val in layer_ids]

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_experts,
            num_groups=None,
        )


EntryClass = Qwen3MoeForCausalLM
