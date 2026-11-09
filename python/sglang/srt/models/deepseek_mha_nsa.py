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
"""Inference-only Deepseek MHA NSA model."""

import math
from collections.abc import Iterable
from typing import Any, Optional

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from sglang.srt.distributed.decoder_comm_manager import DecoderCommMananger
from sglang.srt.env import global_server_args_dict
from sglang.srt.layers.attention.dp_attention import get_attention_tp_group, get_attention_tp_rank, get_attention_tp_size
from sglang.srt.layers.attention.native_sparse_attention.nsa_mla import NativeSparseAttentionMLA
from sglang.srt.layers.attention.radix_attention import RadixAttention
from sglang.srt.layers.dense.gemms.fp8.fp8_utils import block_dequant
from sglang.srt.layers.embedding.rotary_embedding import get_rope
from sglang.srt.layers.embedding.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.moe.layer import MoELayer
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.sampler.logits_processor import LogitsProcessor
from sglang.srt.managers.eplb.expert_distribution import (
    get_global_expert_distribution_recorder,
)
from sglang.srt.managers.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA, DeepseekV2MLP, DeepseekV2MoE
from sglang.srt.models.qwen3_nsa import NativeSparseAttention
from sglang.srt.ops.layernorm import RMSNorm
from sglang.srt.utils import LazyValue, add_prefix


class DeepseekV3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        DeepseekV3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class DeepseekMha(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        layer_id=None,
        attention_bias: bool = False,
        prefix: str = "",
        reduce_attn_results: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id

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
        self.head_dim = hidden_size // self.total_num_heads
        assert self.head_dim * self.total_num_heads == self.hidden_size, "hidden_size must be divisible by num_heads"

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
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=attention_bias,
            reduce_results=reduce_attn_results,
            quant_config=quant_config,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
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
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=self.layer_id,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class DeepseekMhaNSA(DeepseekMha):
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        layer_id=None,
        attention_bias: bool = False,
        prefix: str = "",
        reduce_attn_results: bool = True,
    ) -> None:
        super().__init__(
            config=config,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            layer_id=layer_id,
            attention_bias=attention_bias,
            prefix=prefix,
            reduce_attn_results=reduce_attn_results,
        )

        self.kernel_size = config.kernel_size
        self.stride = config.stride
        assert self.kernel_size % self.stride == 0
        assert math.log2(self.stride).is_integer()

        self.post_compress_key_rope = getattr(config, "post_compress_key_rope", False)
        assert not self.post_compress_key_rope
        self.apply_additive_intra_block_position_emb = config.apply_additive_intra_block_position_emb
        assert not self.apply_additive_intra_block_position_emb
        self.head_share_cmp_att_weight = config.head_share_cmp_att_weight
        assert self.head_share_cmp_att_weight
        self.compress_type = getattr(config, "compress_type", "weighted")
        assert self.compress_type == "gated"

        self.select_size = config.select_size
        assert self.select_size >= self.kernel_size
        assert self.select_size % self.stride == 0

        self.top_n = config.top_n
        self.slc_att_num_init_blocks = config.slc_att_num_init_blocks
        self.slc_att_num_local_blocks = config.slc_att_num_local_blocks
        self.window_size = config.window_size

        head_share_att_gate_weight = config.head_share_att_gate_weight
        assert head_share_att_gate_weight
        gate_type = getattr(config, "gate_type", "sigmoid")
        assert gate_type == "sigmoid"
        gate_feature = getattr(config, "gate_feature", "query")
        assert gate_feature == "attention"
        gate_mask = getattr(config, "gate_mask", "111")
        assert gate_mask == "111"

        assert config.select_size >= config.kernel_size
        assert config.kernel_size % config.stride == 0
        assert config.select_size % config.stride == 0
        assert math.log2(config.stride).is_integer()

        self.attn = NativeSparseAttention(
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            qk_head_dim=self.head_dim,
            v_head_dim=self.head_dim,
            scaling=self.scaling,
            sliding_window_size=self.window_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
            select_size=self.select_size,
            top_n=self.top_n,
            slc_att_num_init_blocks=self.slc_att_num_init_blocks,
            slc_att_num_local_blocks=self.slc_att_num_local_blocks,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )


def rearrange_pe(e):
    old_shape = e.shape
    d = old_shape[-1]
    e = e.view(*old_shape[:-1], d // 2, 2).transpose(-1, -2)
    return e.reshape(old_shape)


class DeepseekNSAWithMLA(DeepseekV2AttentionMLA):
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
        rope_scaling: Optional[dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        layer_id=None,
        prefix: str = "",
        reduce_attn_results=True,
        alt_stream: Optional[torch.cuda.Stream] = None,
        skip_rope: bool = False,
    ):
        super().__init__(
            config=config,
            hidden_size=hidden_size,
            num_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            layer_id=layer_id,
            prefix=prefix,
            reduce_attn_results=reduce_attn_results,
            alt_stream=alt_stream,
            skip_rope=skip_rope,
        )
        del self.attn_mqa
        del self.attn_mha
        # del self.kv_b_proj
        # self.k_b_proj = ColumnParallelLinear(
        #     self.kv_lora_rank,
        #     self.num_heads * self.qk_nope_head_dim,
        #     bias=False,
        #     quant_config=quant_config,
        #     prefix=add_prefix("k_b_proj", prefix),
        #     tp_rank=self.attn_tp_rank,
        #     tp_size=self.attn_tp_size,
        # )

        # self.v_b_proj = ColumnParallelLinear(
        #     self.kv_lora_rank,
        #     self.num_heads * self.v_head_dim,
        #     bias=False,
        #     quant_config=quant_config,
        #     prefix=add_prefix("v_b_proj", prefix),
        #     tp_rank=self.attn_tp_rank,
        #     tp_size=self.attn_tp_size,
        # )
        # for precision check
        del self.kv_a_layernorm
        self.kv_a_layernorm = DeepseekV3RMSNorm(self.kv_lora_rank)
        # ========= nsa related =========
        self.nsa_gate_mask = [int(c) for c in config.nsa_gate_mask]
        self.nsa_gate_type = config.nsa_gate_type
        self.stride = config.stride
        self.kernel_size = config.kernel_size
        self.select_size = config.select_size
        self.top_n = config.top_n
        self.slc_att_num_init_blocks = config.slc_att_num_init_blocks
        self.slc_att_num_local_blocks = config.slc_att_num_local_blocks
        self.window_size = getattr(config, "window_size", 512)
        self.num_slc_score_heads = self.num_local_heads // config.virtual_k_group_size
        self.virtual_k_group_agg_type = config.virtual_k_group_agg_type
        self.head_not_share_att_gate_weight = getattr(config, "head_not_share_att_gate_weight", False)

        self.attn = NativeSparseAttentionMLA(
            num_q_heads=self.num_local_heads,
            num_score_heads=1,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            scaling=self.scaling,
            sliding_window_size=self.window_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
            select_size=self.select_size,
            top_n=self.top_n,
            slc_att_num_init_blocks=self.slc_att_num_init_blocks,
            slc_att_num_local_blocks=self.slc_att_num_local_blocks,
            layer_id=layer_id,
            quant_config=quant_config,
            kv_b_proj=self.kv_b_proj,
            num_slc_score_heads=self.num_slc_score_heads,
            virtual_k_group_agg_type=self.virtual_k_group_agg_type,
            nsa_gate_mask=self.nsa_gate_mask,
            nsa_gate_type=self.nsa_gate_type,
            gate_weight_head_not_share=self.head_not_share_att_gate_weight,
            config=config,
            prefix=add_prefix("nsa", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        comm_manager: DecoderCommMananger,
        block_scale: Optional[torch.Tensor] = None,
        can_run_flashinfer_fusion: bool = False,
    ) -> torch.Tensor:

        def no_absorb():
            return not forward_batch.forward_mode.is_decode() and sum(forward_batch.extend_prefix_lens_cpu) == 0

        if no_absorb():
            q_nope, q_pe, k_nope, k_pe, v, lora_kv = self.forward_normal_prologue(positions, hidden_states, forward_batch, comm_manager)

            k_pe = k_pe.view(-1, 1, self.qk_rope_head_dim)
            output = self.attn(
                q=torch.cat([q_nope, q_pe], dim=-1),
                k=torch.cat([k_nope, k_pe.expand(-1, self.num_local_heads, -1)], dim=-1),
                v=v,
                forward_batch=forward_batch,
                save_kv_cache=False,
            )
        else:
            q_nope_out, k_nope, q_pe, k_pe = self.forward_absorb_qkv_proj(hidden_states, positions, forward_batch, comm_manager)
            q_pe, k_pe = map(rearrange_pe, [q_pe, k_pe])
            output = self.attn(
                q=q_pe,
                k=k_pe,
                v=k_nope,
                forward_batch=forward_batch,
                qv=q_nope_out,
            )
        output, _ = self.o_proj(output)
        return output

    def forward_normal_prologue(
        self, positions: torch.Tensor, hidden_states: torch.Tensor, forward_batch: ForwardBatch, comm_manager: DecoderCommMananger
    ) -> torch.Tensor:
        if self.q_lora_rank is not None:
            qkv = self.fused_qkv_a_proj_with_mqa(hidden_states)[0]
            qkv = comm_manager.pre_attn_comm(qkv, forward_batch.tp_num_tokens)
            q, latent_cache = qkv.split([self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1)
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            hidden_states = comm_manager.pre_attn_comm(hidden_states, forward_batch.tp_num_tokens)
            q = self.q_proj(hidden_states)[0].view(-1, self.num_local_heads, self.qk_head_dim)
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]

        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        kv_a, k_pe = latent_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_a = self.kv_a_layernorm(kv_a)
        # k_nope = self.k_b_proj(kv_a)[0]
        # v = self.v_b_proj(kv_a)[0]
        # k_nope = k_nope.view(-1, self.num_local_heads, self.qk_nope_head_dim)
        # v = v.view(-1, self.num_local_heads, self.v_head_dim)
        kv = self.kv_b_proj(kv_a)[0]
        kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope = kv[..., : self.qk_nope_head_dim]
        v = kv[..., self.qk_nope_head_dim :]

        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        q_pe, k_pe = map(rearrange_pe, [q_pe, k_pe])

        latent_cache = latent_cache.unsqueeze(1)
        latent_cache[:, :, : self.kv_lora_rank] = kv_a.unsqueeze(1)
        latent_cache[:, :, self.kv_lora_rank :] = k_pe.unsqueeze(1).clone()

        # Save latent cache
        ttkv_pool: MLATokenToKVPool = forward_batch.token_to_kv_pool
        ttkv_pool.kv_buffer[self.layer_id][forward_batch.out_cache_loc] = latent_cache

        return q_nope, q_pe, k_nope, k_pe, v, kv_a


class DeepseekMhaNsaDecoderLayer(nn.Module):
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

        if self.enable_dp_attention:
            self.tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = get_tensor_model_parallel_world_size()
            self.tp_group = get_tp_group()
        if is_nextn or (config.n_routed_experts is not None and layer_id >= config.first_k_dense_replace and layer_id % config.moe_layer_freq == 0):
            self.is_moe_layer = True
        else:
            self.is_moe_layer = False

        use_nsa = getattr(config, "use_nsa", False)
        use_nsa_mla = getattr(config, "use_nsa_mla", False)
        if use_nsa_mla:
            self.self_attn = DeepseekNSAWithMLA(
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=config.q_lora_rank,
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
        else:
            attn_type = DeepseekMhaNSA if use_nsa else DeepseekMha
            self.self_attn = attn_type(
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                max_position_embeddings=max_position_embeddings,
                quant_config=quant_config,
                layer_id=layer_id,
                attention_bias=config.attention_bias,
                prefix=add_prefix("self_attn", prefix),
                reduce_attn_results=False,
            )

        if self.is_moe_layer:
            self.mlp = DeepseekV2MoE(
                config=config,
                quant_config=quant_config,
                layer_index=layer_id,
                prefix=add_prefix("mlp", prefix),
                alt_stream=alt_stream,
            )
        else:
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
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
        self, positions: torch.Tensor, hidden_states: torch.Tensor, forward_batch: ForwardBatch, residual: Optional[torch.Tensor], tp_num_tokens: int
    ) -> torch.Tensor:

        num_global_tokens, max_num_tokens_per_gpu = forward_batch.get_num_tokens(tp_num_tokens)

        if not forward_batch.forward_mode.is_idle():
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)

            hidden_states = self.self_attn(
                positions=positions, hidden_states=hidden_states, forward_batch=forward_batch, comm_manager=self.decoder_comm_manager
            )

            hidden_states, residual = self.decoder_comm_manager.post_attn_comm(hidden_states, residual, tp_num_tokens)

            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

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
        hidden_states, start_idx, end_idx = self.decoder_comm_manager.pre_mlp_comm(hidden_states, forward_batch, tp_num_tokens)
        hidden_states = self.mlp(hidden_states, num_global_tokens, max_num_tokens_per_gpu) if self.is_moe_layer else self.mlp(hidden_states)
        hidden_states, residual = self.decoder_comm_manager.post_mlp_comm(hidden_states, residual, tp_num_tokens)
        if start_idx is not None and end_idx is not None:
            hidden_states = hidden_states[start_idx:end_idx]
        return hidden_states


class DeepseekMhaNsaModel(nn.Module):
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
        self.alt_stream = torch.cuda.Stream()
        self.layers = nn.ModuleList(
            [
                DeepseekMhaNsaDecoderLayer(
                    config,
                    layer_id,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{layer_id}", prefix),
                    alt_stream=self.alt_stream,
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
        residual = None
        for i in range(len(self.layers)):
            with get_global_expert_distribution_recorder().with_current_layer(i):
                # hids.append(hidden_states+residual if residual is not None else hidden_states.clone())
                layer = self.layers[i]
                hidden_states, residual = layer(positions, hidden_states, forward_batch, residual, tp_num_tokens=tp_num_tokens)
        if not forward_batch.forward_mode.is_idle():
            hidden_states, _ = self.norm(hidden_states, residual)
            # hids.append(hidden_states.clone())
            hidden_states, _ = layer.decoder_comm_manager.post_final_norm_comm(hidden_states, residual, tp_num_tokens)
        # all_hid.append(hids)
        return hidden_states


class DeepseekMhaNsaForCausalLM(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        model: Optional[DeepseekMhaNsaModel] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = model if model is not None else DeepseekMhaNsaModel(config, quant_config, prefix=add_prefix("model", prefix))
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
                if isinstance(layer.mlp, DeepseekV2MoE)
            }
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch)
        return self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch)

    def get_attention_sliding_window_size(self):
        return getattr(self.config, "window_size", None)

    @property
    def routed_experts_weights_of_layer(self):
        return self._routed_experts_weights_of_layer.value

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        name_mapping = {
            "compress_attn": "attn.compress_attn",
            "compress_key": "compress_kv",  # kv_lora_rank
            "compress_value": "compress_k_pe",  # q_pe
            "gate_fusion.gate_weight": "attn.gate_fusion.gate_weight.weight",
        }

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        MoEImpl = DeepLayer if global_server_args_dict["enable_deep_ep"] else MoELayer if global_server_args_dict["enable_ep_moe"] else FusedMoE  # noqa: F821
        expert_params_mapping = MoEImpl.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
        )

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            for source_name, target_name in name_mapping.items():
                if source_name in name:
                    name = name.replace(source_name, target_name)
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
                # to avoid stack 'gate_proj' in compress_attn or gate_fusion
                if "compress_attn" in name or "gate_fusion" in name:
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
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name.endswith("gate_fusion.gate_proj.weight"):
                        loaded_weight = loaded_weight.squeeze(0)
                    if "gate_fusion.gate_weight" in name and len(loaded_weight.shape) > 2:
                        # [qh, gate_num, gate_num*hd] -> [gate_num*gate_num*hd, qh]``
                        loaded_weight = loaded_weight.flatten(1).transpose(0, 1)
                    if "q_a_proj" in name and name not in params_dict:
                        name = name.replace("q_a_proj", "q_proj")
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
        # weight transpose for absorb
        if getattr(self.config, "use_nsa_mla", False):
            for layer_id in range(self.config.num_hidden_layers):
                self_attn: DeepseekNSAWithMLA = self.model.layers[layer_id].self_attn
                if hasattr(self.quant_config, "weight_block_size") and self_attn.kv_b_proj.weight.dtype in (
                    torch.float8_e4m3fn,
                    torch.float8_e4m3fnuz,
                ):
                    weight_block_size = self.quant_config.weight_block_size
                    if weight_block_size is not None:
                        assert hasattr(self_attn.kv_b_proj, "weight_scale_inv")
                        dtype = torch.get_default_dtype()
                        w = block_dequant(self_attn.kv_b_proj.weight, self_attn.kv_b_proj.weight_scale_inv, weight_block_size).to(dtype)
                else:
                    w = self_attn.kv_b_proj.weight

                w_kc, w_vc = w.unflatten(0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)).split(
                    [self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1
                )
                self_attn.w_kc = w_kc.contiguous()
                self_attn.w_vc = w_vc.contiguous().transpose(1, 2)
                self_attn.attn.w_vc = self_attn.w_vc
                if self.config.mla_scale_q_lora:
                    self_attn.q_a_layernorm.weight.data *= (self.config.hidden_size / self.config.q_lora_rank) ** 0.5
                if self.config.mla_scale_kv_lora:
                    self_attn.kv_a_layernorm.weight.data *= (self.config.hidden_size / self.config.kv_lora_rank) ** 0.5

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def init_cuda_graph_state_nsa(self, max_bs, max_num_tokens, max_context_len):
        q_heads = self.config.num_attention_heads // get_attention_tp_size()
        NativeSparseAttention.init_cuda_graph_state_nsa(max_bs, max_num_tokens, max_context_len, self.config.stride, q_heads, self.config.select_size)

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_routed_experts,
            num_groups=None,
        )


EntryClass = DeepseekMhaNsaForCausalLM
