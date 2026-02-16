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


"""Inference-only GptOss model compatible with HuggingFace weights."""

from collections.abc import Iterable
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.managers.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.managers.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.moe.layer import MoELayer

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.dp_attention import (
    get_attention_tp_group,
    get_attention_tp_rank,
    get_attention_tp_size,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.env import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix, get_colorful_logger, make_layers
from sglang.srt.distributed.decoder_comm_manager import DecoderCommMananger


class GptOssConfig(PretrainedConfig):
    model_type = "gpt_oss"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


logger = get_colorful_logger(__name__)


# Aligned with HF's implementation, using sliding window inclusive with the last token
# SGLang assumes exclusive
def get_attention_sliding_window_size(config):
    return config.sliding_window - 1


def routing_function(hidden_states, gating_output, topk, renormalize):
    experts = torch.topk(gating_output, k=topk, dim=-1, sorted=True)
    expert_weights = torch.nn.functional.softmax(experts.values.to(torch.float32), dim=1)
    expert_indices = experts.indices.to(torch.int32)
    return expert_weights, expert_indices


class GptOssSparseMoeBlock(nn.Module):
    def __init__(
        self,
        config: GptOssConfig,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        layer_index: int = -1,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_index = layer_index
        self.tp_size = get_tensor_model_parallel_world_size()
        self.hidden_size = hidden_size
        self.activation = config.hidden_act
        self.activation_alpha = getattr(config, "hidden_act_alpha", 1.702)
        self.swiglu_limit = config.swiglu_limit
        self.num_experts = num_experts + global_server_args_dict["ep_num_redundant_experts"]
        self.quant_config = quant_config
        if self.tp_size > config.num_local_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_local_experts}."
            )

        self.experts = MoELayer(
            top_k=top_k,
            num_experts=self.num_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant_config=self.quant_config,
            layer_index=self.layer_index,
            prefix=add_prefix("experts", prefix),
            activation="swiglu",
            activation_alpha=self.activation_alpha,
            swiglu_limit=self.swiglu_limit,
            with_bias=True
        )

        self.topk = TopK(
            top_k=top_k,
            custom_routing_function=routing_function,
            output_format=TopKOutputFormat.STANDARD,
            topk_indices_dtype=torch.int64 if global_server_args_dict["enable_deep_ep"] else torch.int32
        )

        self.router = ReplicatedLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=True,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
            params_dtype=config.torch_dtype,
        )

    def forward(
        self, hidden_states: torch.Tensor, num_global_tokens: int, max_num_tokens_per_gpu: int
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.router(hidden_states)
        if hidden_states.shape[0] > 0:
            topk_output = self.topk(hidden_states, router_logits)
        else:
            topk_output = self.topk.empty_topk_output(hidden_states.device)

        routed_expert_output = self.experts(
            hidden_states=hidden_states,
            topk_output=topk_output,
            num_global_tokens=num_global_tokens,
            max_num_tokens_per_gpu=max_num_tokens_per_gpu,
        )

        return routed_expert_output.view(num_tokens, hidden_dim)

    def get_moe_weights(self):
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"]
        ]


class GptOssAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        head_dim: Optional[int] = None,
        rms_norm_eps: float = 1e-06,
        attention_bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        sliding_window_size: int = -1,  # if -1, normal attention, else, window attention.
        layer_type: str = "",
        params_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.sliding_window_size = sliding_window_size

        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        self.total_num_heads = num_heads
        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.tp_rank = get_tensor_model_parallel_rank()

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            params_dtype=params_dtype,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )

        self.sinks = nn.Parameter(
            torch.empty(self.num_heads, dtype=params_dtype), requires_grad=False
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,
            params_dtype=params_dtype,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        assert layer_type in {"sliding_attention", "full_attention"}
        use_sliding_window = layer_type == "sliding_attention"
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=(sliding_window_size if use_sliding_window else -1),
        )
        self.layer_id = layer_id

    def forward_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        if hidden_states.shape[0] == 0:
            return hidden_states, forward_batch, None
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        inner_state = q, k, v, forward_batch
        return None, forward_batch, inner_state

    def forward_core(self, intermediate_state):
        hidden_states, forward_batch, inner_state = intermediate_state
        if inner_state is None:
            return hidden_states
        attn_output = self.attn(*inner_state, sinks=self.sinks)
        output, _ = self.o_proj(attn_output)
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        s = self.forward_prepare(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        return self.forward_core(s)


class GptOssDecoderLayer(nn.Module):
    def __init__(
        self,
        config: GptOssConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        sliding_window_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        rms_norm_eps = config.rms_norm_eps
        attention_bias = config.attention_bias

        if sliding_window_size is None:
            self.sliding_window_size = get_attention_sliding_window_size(self.config)
        else:
            self.sliding_window_size = sliding_window_size

        self.self_attn = GptOssAttention(
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
            prefix=add_prefix("self_attn", prefix),
            sliding_window_size=self.sliding_window_size,
            layer_type=config.layer_types[layer_id],
            params_dtype=config.torch_dtype,
        )

        self.mlp = GptOssSparseMoeBlock(
            config=config,
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant_config=quant_config,
            layer_index=layer_id,
            prefix=add_prefix(f"mlp", prefix)
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
            layer_id=layer_id,
            attn_parallel_strategy=global_server_args_dict["attn_parallel_strategy"],
            dense_parallel_strategy=global_server_args_dict["dense_parallel_strategy"],
            moe_parallel_strategy=global_server_args_dict["moe_parallel_strategy"],
            is_moe_layer=True,
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
        num_global_tokens, max_num_tokens_per_gpu = forward_batch.get_num_tokens(tp_num_tokens)

        if not forward_batch.forward_mode.is_idle():
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

            hidden_states, residual = self.decoder_comm_manager.post_attn_comm(
                hidden_states, residual, tp_num_tokens)

            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )

            hidden_states = self.forward_mlp(
                hidden_states,
                residual,
                forward_batch,
                num_global_tokens,
                max_num_tokens_per_gpu,
                tp_num_tokens,
            )
        else:
            hidden_states = self.forward_mlp(
                hidden_states,
                residual,
                forward_batch,
                num_global_tokens,
                max_num_tokens_per_gpu,
                tp_num_tokens,
            )
        return hidden_states, residual

    def forward_mlp(
        self,
        hidden_states,
        residual,
        forward_batch,
        num_global_tokens,
        max_num_tokens_per_gpu,
        tp_num_tokens,
    ):
        hidden_states, start_idx, end_idx = self.decoder_comm_manager.pre_mlp_comm(
            hidden_states, forward_batch, tp_num_tokens
        )

        hidden_states = self.mlp(hidden_states, num_global_tokens, max_num_tokens_per_gpu)

        hidden_states, residual = self.decoder_comm_manager.post_mlp_comm(
            hidden_states, residual, tp_num_tokens, forward_batch
        )
        if start_idx is not None and end_idx is not None:
            hidden_states = hidden_states[start_idx:end_idx]
        return hidden_states


class GptOssModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        decoder_layer_type: type[nn.Module] = GptOssDecoderLayer,
    ) -> None:
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            enable_tp=not global_server_args_dict["enable_dp_attention"],
            prefix=add_prefix("embed_tokens", prefix),
        )

        # Use the provided decoder layer type or default to GptOssDecoderLayer
        decoder_layer_type = decoder_layer_type or GptOssDecoderLayer
        self.layers = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: decoder_layer_type(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=add_prefix("layers", prefix),
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        tp_num_tokens = hidden_states.shape[0]
        forward_batch.tp_num_tokens = tp_num_tokens
        residual = None
        for i in range(len(self.layers)):
            with get_global_expert_distribution_recorder().with_current_layer(i):
                layer = self.layers[i]
                hidden_states, residual = layer(
                    positions, hidden_states, forward_batch, residual, tp_num_tokens=tp_num_tokens
                )
        if not forward_batch.forward_mode.is_idle():
            assert residual is not None
            hidden_states, _ = self.norm(hidden_states, residual)
            hidden_states, _ = layer.decoder_comm_manager.post_final_norm_comm(hidden_states, residual, tp_num_tokens)
        return hidden_states


class GptOssForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: GptOssConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = GptOssModel(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            # quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
        )

        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
        )

    def _get_default_weight_mapping(self):
        """Generate default weight name mapping for GptOss safetensors."""
        weight_mapping = {}

        # Map router weights to gate
        weight_mapping["embedding.weight"] = "model.embed_tokens.weight"
        weight_mapping["unembedding.weight"] = "lm_head.weight"
        weight_mapping["norm.scale"] = "model.norm.weight"
        for layer_id in range(self.config.num_hidden_layers):
            weight_mapping[f"block.{layer_id}.attn.q_proj.weight"] = (
                f"model.layers.{layer_id}.self_attn.q_proj.weight"
            )
            weight_mapping[f"block.{layer_id}.attn.q_proj.bias"] = (
                f"model.layers.{layer_id}.self_attn.q_proj.bias"
            )

            weight_mapping[f"block.{layer_id}.attn.k_proj.weight"] = (
                f"model.layers.{layer_id}.self_attn.k_proj.weight"
            )
            weight_mapping[f"block.{layer_id}.attn.k_proj.bias"] = (
                f"model.layers.{layer_id}.self_attn.k_proj.bias"
            )

            weight_mapping[f"block.{layer_id}.attn.v_proj.weight"] = (
                f"model.layers.{layer_id}.self_attn.v_proj.weight"
            )
            weight_mapping[f"block.{layer_id}.attn.v_proj.bias"] = (
                f"model.layers.{layer_id}.self_attn.v_proj.bias"
            )

            weight_mapping[f"block.{layer_id}.attn.out.weight"] = (
                f"model.layers.{layer_id}.self_attn.o_proj.weight"
            )
            weight_mapping[f"block.{layer_id}.attn.out.bias"] = (
                f"model.layers.{layer_id}.self_attn.o_proj.bias"
            )
            weight_mapping[f"block.{layer_id}.attn.sinks"] = (
                f"model.layers.{layer_id}.self_attn.sinks"
            )
            weight_mapping[f"block.{layer_id}.attn.norm.scale"] = (
                f"model.layers.{layer_id}.input_layernorm.weight"
            )

            weight_mapping[f"block.{layer_id}.mlp.gate.weight"] = (
                f"model.layers.{layer_id}.mlp.router.weight"
            )
            weight_mapping[f"block.{layer_id}.mlp.gate.bias"] = (
                f"model.layers.{layer_id}.mlp.router.bias"
            )
            weight_mapping[f"block.{layer_id}.mlp.norm.scale"] = (
                f"model.layers.{layer_id}.post_attention_layernorm.weight"
            )
            weight_mapping[f"block.{layer_id}.mlp.experts.gate_up_proj"] = (
                f"model.layers.{layer_id}.mlp.experts.gate_up_proj"
            )
            weight_mapping[f"block.{layer_id}.mlp.gate_up_proj_bias"] = (
                f"model.layers.{layer_id}.mlp.experts.gate_up_proj_bias"
            )
            weight_mapping[f"block.{layer_id}.mlp.down_proj"] = (
                f"model.layers.{layer_id}.mlp.experts.mlp2_weight"
            )
            weight_mapping[f"block.{layer_id}.mlp.down_proj_bias"] = (
                f"model.layers.{layer_id}.mlp.experts.mlp2_bias"
            )

        return weight_mapping

    def load_weights(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
        is_nextn: bool = False,
        weight_name_mapping: dict = None,
    ):
        quant_config_name = (
            self.quant_config.get_name() if self.quant_config is not None else None
        )
        assert quant_config_name != "mxfp4"
        assert not is_nextn

        self._load_normal_weights(weights, weight_name_mapping=weight_name_mapping)

    def _load_normal_weights(
        self,
        weights,    # It's an iterator, pay attention to whether the iterator is invalidated.
        weight_name_mapping: dict,
    ):
        tp_rank = get_tensor_model_parallel_rank()
        weights = sorted(weights, key=lambda x: x[0])  # Sort by name for consistency

        # Use provided weight name mapping if available, otherwise use default
        if weight_name_mapping is None:
            weight_name_mapping = self._get_default_weight_mapping()
        else:
            # Merge with default mapping
            default_mapping = self._get_default_weight_mapping()
            default_mapping.update(weight_name_mapping)
            weight_name_mapping = default_mapping

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        def make_expert_params_mapping_fused(
            ckpt_gate_up_proj_name: str,
            ckpt_down_proj_name: str,
            ckpt_gate_up_proj_bias_name: str,
            ckpt_down_proj_bias_name: str,
        ):
            return [
                ("experts.w13_weight", f"experts.{ckpt_gate_up_proj_name}", "w13"),
                (
                    "experts.w13_weight_bias",
                    f"experts.{ckpt_gate_up_proj_bias_name}",
                    "w13",
                ),
                ("experts.w2_weight", f"experts.{ckpt_down_proj_name}", "w2"),
                ("experts.w2_weight_bias", f"experts.{ckpt_down_proj_bias_name}", "w2"),
            ]
        expert_params_mapping = (
            make_expert_params_mapping_fused(
                ckpt_gate_up_proj_name="gate_up_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_gate_up_proj_bias_name="gate_up_proj_bias",
                ckpt_down_proj_bias_name="down_proj_bias",
            )
        )

        params_dict = dict(self.named_parameters())
        params_checker = {k: False for k, v in params_dict.items()}

        for name, loaded_weight in weights:
            loaded_weight = _WeightCreator.maybe_materialize(loaded_weight)

            # Apply weight name mapping if provided
            if weight_name_mapping and name in weight_name_mapping:
                name = weight_name_mapping[name]

            if "rotary_emb.inv_freq" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:

                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue

                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                params_checker[name] = True
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    if "bias" not in name:
                        # [E, K, N] -> [E, N, K]
                        loaded_weight = loaded_weight.transpose(-2, -1)

                    # Only support EP
                    local_num_experts = param.shape[0]
                    assert local_num_experts * get_tensor_model_parallel_world_size() == loaded_weight.shape[0]
                    local_experts = loaded_weight[local_num_experts * tp_rank: local_num_experts * (tp_rank + 1)]
                    if loaded_weight.dtype == torch.float8_e5m2:
                        default_weight_loader(param, local_experts.to(torch.bfloat16))
                    else:
                        default_weight_loader(param, local_experts)
                    params_checker[name] = True
                    break
                else:
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name not in params_dict:
                        continue
                    if name in params_dict.keys():
                        param = params_dict[name]
                        if "sinks" in name:
                            start = tp_rank * param.numel()
                            param.data.copy_(
                                loaded_weight[start : start + param.numel()]
                            )
                        else:
                            weight_loader = getattr(
                                param, "weight_loader", default_weight_loader
                            )
                            weight_loader(param, loaded_weight)
                        params_checker[name] = True
                    else:
                        logger.warning(f"Parameter {name} not found in params_dict")
            if tp_rank == 0:
                logger.info(f"{name=} loaded.")

        not_loaded_params = []
        for k, v in params_checker.items():
            if not v and ("weight_scale" not in k) and ("input_scale" not in k):
                not_loaded_params.append(k)

        if tp_rank == 0:
            if len(not_loaded_params) > 0:
                raise Exception(f"Not all parameters loaded: {not_loaded_params=}")
            else:
                logger.info("All parameters loaded successfully.")

        self.routed_experts_weights_of_layer = {
            layer_id: self.model.layers[layer_id].mlp.get_moe_weights()
            for layer_id in range(len(self.model.layers))
        }

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def get_attention_sliding_window_size(self):
        return get_attention_sliding_window_size(self.config)

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_local_experts,
            num_groups=None,
        )


class _WeightCreator:
    def __init__(self, fn):
        self._fn = fn

    @staticmethod
    def maybe_materialize(obj):
        if isinstance(obj, _WeightCreator):
            output = obj._fn()
            obj._fn = None
            return output

        return obj


EntryClass = GptOssForCausalLM
