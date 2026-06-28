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

from typing import Any, Dict, Iterable, Optional, Tuple
import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.layers.moe.layouts.mapping import make_expert_params_mapping
from sglang.srt.layers.moe.npu_moe.layer import NpuEPMoE
from sglang.srt.configs.model_config import (
    is_dsa
)
from sglang.srt.layers.utils import (
    get_layer_id,
    FLLM_IS_CP, CP_METADATA, cp_split_and_rebuild_data, cp_all_gather_rerange_output
)

from sglang.srt.distributed import (
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm, FusedRMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.env import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.managers.expert_location import ModelConfigForExpertLocation
from sglang.srt.utils import add_prefix, LazyValue, get_available_gpu_memory
from sglang.srt.distributed.parallel_strategy import DenseParallelStategy
from sglang.srt.layers.dp_attention import get_attention_tp_group, get_attention_tp_size, get_attention_tp_rank
from sglang.srt.distributed.decoder_comm_manager import DecoderCommMananger
from sglang.srt.managers.expert_distribution import (
    get_global_expert_distribution_recorder,
)
from sglang.srt.models.deepseek_v32 import DeepseekV32MLA

from sglang.srt.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


class DeepseekV2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        is_shared_expert: bool = False
    ) -> None:
        super().__init__()
        self.layout = global_server_args_dict["dense_parallel_strategy"]
        # For TP MOE, shared expert uses TP
        if (is_shared_expert and not global_server_args_dict["enable_ep_moe"]) or (not is_shared_expert and self.layout == DenseParallelStategy.TENSOR_PARALLEL):
            # TODO npu support torch. all_gather_torch and reduce_scatter_torch
            self.tp_group = get_tp_group()
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
                reduce_results=False,  # Communication is handled externally and manually controlled
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
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        if x.shape[0] == 0:
            return x
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class MoEGate(nn.Module):
    def __init__(self, config, prefix: str = ""):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((config.n_routed_experts, config.hidden_size))
        )
        if config.topk_method == "noaux_tc":
            self.e_score_correction_bias = nn.Parameter(
                torch.empty((config.n_routed_experts), dtype=torch.float32)
            )
        else:
            self.e_score_correction_bias = None

    def forward(self, hidden_states, comm_manager=None):
        logits = F.linear(hidden_states, self.weight, None)
        return logits.to(torch.bfloat16)

class NpuDeepseekV2MoE(nn.Module):
    def __init__(self, config: PretrainedConfig, quant_config: Optional[QuantizationConfig] = None, layer_index: int = -1):
        super().__init__()
        self.layer_index = layer_index
        self.ep_size = get_tensor_model_parallel_world_size()
        self.n_shared_experts = config.n_shared_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        if self.ep_size > config.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.n_routed_experts}."
            )

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        if not global_server_args_dict["enable_ep_moe"]:
            raise ValueError("NPU only support EP MoE now")

        self.gate = MoEGate(config=config)

        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                is_shared_expert=True
            )

        self.experts = NpuEPMoE(
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            correction_bias=self.gate.e_score_correction_bias,
            routed_scaling_factor=self.routed_scaling_factor,
            gate_layer=self.gate,
        )

    def forward(self, x, max_num_tokens_per_gpu) -> torch.Tensor:
        hidden_states = x
        shared_experts_out = self.shared_experts(hidden_states)
        experts_out = self.experts(x, max_num_tokens_per_gpu)

        return shared_experts_out + experts_out


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:

    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0

class DeepseekV2AttentionMLA(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        layer_id=None,
        prefix: str = "",
        reduce_attn_results=True,
        skip_rope: bool = False,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        if FLLM_IS_CP:
            self.attn_tp_group = None
            self.num_local_heads = num_heads
        else:
            self.attn_tp_group = get_attention_tp_group()
            assert num_heads % self.attn_tp_group.world_size == 0
            self.num_local_heads = num_heads // self.attn_tp_group.world_size
        self.scaling = self.qk_head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.config = config
        self.attention_backend = global_server_args_dict["attention_backend"]
        self.cli_factor = getattr(config, "cli_factor", 1)
        self.prefix = prefix

        # modification to rope_scaling must be done early enough, b/c e.g. Indexer needs it
        if rope_scaling:
            rope_scaling["rope_type"] = "deepseek_yarn"

        if self.q_lora_rank is not None:
            self.q_a_proj = ReplicatedLinear(
                self.hidden_size,
                self.q_lora_rank,
                bias=False,
                quant_config=quant_config,
            )

            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("kv_a_proj_with_mqa", prefix),
            )

            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("q_b_proj", prefix),
                outside_tp_group=self.attn_tp_group,
                disable_parallel=FLLM_IS_CP,
            )
        else:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("q_proj", prefix),
                outside_tp_group=self.attn_tp_group,
                disable_parallel=FLLM_IS_CP,
            )

            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("kv_a_proj_with_mqa", prefix),
            )

        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("kv_b_proj", prefix),
            outside_tp_group=self.attn_tp_group,
            disable_parallel=FLLM_IS_CP,
        )
        # O projection.
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=reduce_attn_results,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
            outside_tp_group=self.attn_tp_group,
            disable_parallel=FLLM_IS_CP,
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)

        # Fusion layer
        if self.q_lora_rank is not None:
            self.fused_qk_layernorm = FusedRMSNorm(
                self.q_a_layernorm,
                self.kv_a_layernorm,
            )

        if not skip_rope:
            self.rotary_emb = get_rope(
                qk_rope_head_dim,
                rotary_dim=qk_rope_head_dim,
                max_position=max_position_embeddings,
                base=rope_theta,
                rope_scaling=rope_scaling,
                is_neox_style=False,
            )

            if rope_scaling:
                mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
                scaling_factor = rope_scaling["factor"]
                mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
                self.scaling = self.scaling * mscale * mscale
            else:
                self.rotary_emb.forward = self.rotary_emb.forward_cuda
        else:
            self.rotary_emb = None

        if global_server_args_dict["enable_mla_l1_5_cache"]:
            num_q_heads = self.num_heads
        else:
            num_q_heads = self.num_local_heads
        self.attn_mqa = RadixAttention(
            num_q_heads,
            self.kv_lora_rank + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=1,
            layer_id=layer_id,
            v_head_dim=self.kv_lora_rank,
        )

        self.attn_mha = RadixAttention(
            self.num_local_heads,
            self.qk_nope_head_dim + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=self.num_local_heads,
            layer_id=layer_id,
            v_head_dim=self.v_head_dim,
        )

        self.w_kc = None
        self.w_vc = None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        comm_manager: DecoderCommMananger
    ) -> torch.Tensor:
        def no_absorb() -> bool:
            # TODO: support mla
            if forward_batch.forward_mode.is_decode():
                return False
            else:
                return True

        if no_absorb():
            return self.forward_normal_npu(positions, hidden_states, forward_batch)
        else:
            return self.forward_absorb_npu(positions, hidden_states, forward_batch)

    def forward_normal_npu(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if self.q_lora_rank is not None:
            q = self.q_a_proj(hidden_states)[0]
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
        _, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        kv_a, _ = latent_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        latent_cache = latent_cache.unsqueeze(1)
        kv_a = self.kv_a_layernorm(kv_a.contiguous())
        kv = self.kv_b_proj(kv_a)[0]
        kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope = kv[..., : self.qk_nope_head_dim]
        v = kv[..., self.qk_nope_head_dim :]
        k_pe = latent_cache[:, :, self.kv_lora_rank :]
        if self.rotary_emb is not None:
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        q[..., self.qk_nope_head_dim :] = q_pe
        k = torch.empty_like(q)
        k[..., : self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim :] = k_pe

        latent_cache[:, :, : self.kv_lora_rank] = kv_a.unsqueeze(1)
        latent_cache[:, :, self.kv_lora_rank :] = k_pe

        # Save latent cache
        forward_batch.token_to_kv_pool.set_kv_buffer(
            self.attn_mha, forward_batch.out_cache_loc, latent_cache, None
        )
        attn_output = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)
        attn_output = attn_output.reshape(-1, self.num_local_heads * self.v_head_dim)
        output, _ = self.o_proj(attn_output)
        return output

    def forward_absorb_npu(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        q_len = hidden_states.shape[0]
        q_input = hidden_states.new_empty(
            q_len, self.num_local_heads, self.kv_lora_rank + self.qk_rope_head_dim
        )
        if self.q_lora_rank is not None:
            q = self.q_a_proj(hidden_states)[0]
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)

        q_input[..., : self.kv_lora_rank] = q_nope_out.transpose(0, 1)

        latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        v_input = latent_cache[..., : self.kv_lora_rank]
        v_input = self.kv_a_layernorm(v_input.contiguous()).unsqueeze(1)
        k_input = latent_cache.unsqueeze(1)
        k_input[..., : self.kv_lora_rank] = v_input
        k_pe = k_input[..., self.kv_lora_rank:]
        if self.rotary_emb is not None:
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        q_input[..., self.kv_lora_rank:] = q_pe
        k_input[..., self.kv_lora_rank:] = k_pe

        attn_output = self.attn_mqa(q_input, k_input, v_input, forward_batch)
        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        attn_bmm_output = torch.bmm(attn_output.transpose(0, 1), self.w_vc)
        attn_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
        output, _ = self.o_proj(attn_output)

        return output


class DeepseekV2DecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        is_nextn: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.enable_dp_attention = global_server_args_dict["enable_dp_attention"]

        if is_nextn or (
            config.n_routed_experts is not None
            and layer_id >= config.first_k_dense_replace
            and layer_id % config.moe_layer_freq == 0
        ):
            self.is_moe_layer = True
        else:
            self.is_moe_layer = False

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
            quant_config=None if "self_attn" in getattr(config, "disable_quant_module", []) else quant_config,
            layer_id=layer_id,
            prefix=add_prefix("self_attn", prefix),
            reduce_attn_results=False,
        )

        if self.is_moe_layer:
            self.mlp = NpuDeepseekV2MoE(config=config, quant_config=quant_config, layer_index=layer_id)
        else:
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.ffn_hidden_size if hasattr(config, "ffn_hidden_size") else config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=None if "dense_mlp" in getattr(config, "disable_quant_module", []) else quant_config,
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
            is_moe_layer=self.is_moe_layer,
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
        num_global_tokens,
        max_num_tokens_per_gpu,
        tp_num_tokens,
    ):
        if self.is_moe_layer:
            hidden_states = self.mlp(hidden_states, max_num_tokens_per_gpu)
            return hidden_states

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


class DeepseekV2Model(nn.Module):

    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.padding_id = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            enable_tp=not global_server_args_dict["enable_dp_attention"],
        )
        if FLLM_IS_CP:
            self.cp_size = get_attention_tp_size()
            self.cp_rank = get_attention_tp_rank()
        # config.num_hidden_layers = 5; self.start_layer,self.end_layer = 0, 5
        self.layers = nn.ModuleList(
            [
                DeepseekV2DecoderLayer(
                    config,
                    layer_id,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{layer_id}", prefix),
                )
                for layer_id in range(config.num_hidden_layers)
            ]
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
        if CP_METADATA:
            hidden_states = cp_split_and_rebuild_data(hidden_states, CP_METADATA.value.split_list, CP_METADATA.value.zigzag_index)
            positions = cp_split_and_rebuild_data(positions, CP_METADATA.value.split_list, CP_METADATA.value.zigzag_index)
        residual = None
        for i in range(len(self.layers)):
            with get_global_expert_distribution_recorder().with_current_layer(i):
                layer = self.layers[i]
                hidden_states, residual = layer(
                    positions, hidden_states, forward_batch, residual, tp_num_tokens=tp_num_tokens
                )
        if not forward_batch.forward_mode.is_idle():
            hidden_states, _ = self.norm(hidden_states, residual)
            if not FLLM_IS_CP:
                hidden_states, _ = layer.decoder_comm_manager.post_final_norm_comm(hidden_states, residual, tp_num_tokens)
        if CP_METADATA:
            hidden_states = cp_all_gather_rerange_output(hidden_states, CP_METADATA.value, layer.decoder_comm_manager.attn_rsag)
        return hidden_states


class DeepseekV2ForCausalLM(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        model: Optional[DeepseekV2Model] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = model if model is not None else DeepseekV2Model(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
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
            lambda: {}
        )

    @property
    def routed_experts_weights_of_layer(self):
        return self._routed_experts_weights_of_layer.value

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

    def get_param(self, params_dict, name):
        if name in params_dict:
            return params_dict[name]

        if "language_model." in name:
            name = name.replace("language_model.", "")
            if name in params_dict:
                return params_dict[name]

        logger.warning(f"The {name} is not in the model.")
        return None

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, local_expert_id, shard_id)
        expert_params_mapping = make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
        )

        # Fuse q_a_proj and kv_a_proj_with_mqa along output dimension when q_lora_rank is not None
        fuse_qkv_a_proj = hasattr(self.config, "q_lora_rank") and (
            self.config.q_lora_rank is not None
        )
        cached_a_proj = {} if fuse_qkv_a_proj else None

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue
            # TODO(HandH1998): Modify it when nextn is supported.
            if hasattr(self.config, "num_nextn_predict_layers"):
                num_nextn_layers = self.config.num_nextn_predict_layers
                if num_nextn_layers > 0 and name.startswith("model.layers"):
                    name_list = name.split(".")
                    if (
                        len(name_list) >= 3
                        and int(name_list[2]) >= self.config.num_hidden_layers
                    ):
                        continue
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
                param = self.get_param(params_dict, name)
                if param is None:
                    continue
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
                        param = self.get_param(params_dict, name)
                        if param is None:
                            continue
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

                    if "q_a_proj" in name and name not in params_dict:
                        name = name.replace("q_a_proj", "q_proj")
                    param = self.get_param(params_dict, name)
                    if param is None:
                        continue
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)

        self.post_load_weights()


    def post_load_weights(self):
        for layer_id in range(self.config.num_hidden_layers):
            self_attn = self.model.layers[layer_id].self_attn
            if hasattr(self.quant_config, "weight_block_size") and self_attn.kv_b_proj.weight.dtype in (
                torch.float8_e4m3fn,
                torch.float8_e4m3fnuz,
            ):
                weight_block_size = self.quant_config.weight_block_size
                if weight_block_size is not None:
                    raise NotImplementedError("NPU不支持FP8量化")
            else:
                w = self_attn.kv_b_proj.weight

            w_kc, w_vc = w.unflatten(
                0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
            ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
            self_attn.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
            self_attn.w_vc = w_vc.contiguous().transpose(1, 2)

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        if self.pp_skip_mtp:
            return
        gpu_id = torch.npu.current_device()
        device = "npu"
        logger.info(
            f"release mtp's embedding weight start. "
            f"type={type(self.model).__name__}, "
            f"avail mem={get_available_gpu_memory(device, gpu_id):.2f} GB"
        )
        if not self.model.embed_tokens:
            self.model.embed_tokens = VocabParallelEmbedding(
                self.model.vocab_size,
                self.model.hidden_size,
                enable_tp=not global_server_args_dict["enable_dp_attention"],
            )
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.npu.empty_cache()
        torch.npu.synchronize()
        logger.info(
            f"release mtp's embedding weight end. "
            f"type={type(self.model).__name__}, "
            f"avail mem={get_available_gpu_memory(device, gpu_id):.2f} GB"
        )

    def set_oe_and_head(self, over_embedding, head):
        del self.model.embed_tokens
        self.model.embed_tokens = over_embedding
        self.lm_head.weight = head
        torch.npu.empty_cache()
        torch.npu.synchronize()


    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_routed_experts,
            num_groups=config.n_group,
        )


class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):
    pass

class DeepseekV32ForCausalLM(DeepseekV2ForCausalLM):
    pass

EntryClass = [DeepseekV2ForCausalLM, DeepseekV3ForCausalLM, DeepseekV32ForCausalLM]
