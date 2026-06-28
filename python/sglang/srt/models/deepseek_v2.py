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

from flashinfer import merge_state, dsv3_router_gemm, dsv3_fused_a_gemm
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.layers.moe.layer import MoELayer
from sglang.srt.layers.moe.layouts.mapping import make_expert_params_mapping
from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
from sglang.srt.configs.model_config import (
    is_dsa
)
from sglang.srt.layers.utils import (
    get_layer_id,
    FLLM_IS_CP, CP_METADATA, cp_split_and_rebuild_data, cp_all_gather_rerange_output
)
from sglang.srt.utils import get_device_sm, is_cuda, is_sm90_supported

from sglang.srt.distributed import (
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.attention.chunker import get_attns, get_casual_attn, get_chunks, get_streamed_attn, get_streamed_kv_indices
from sglang.srt.layers.dense.gemms.fp8.fp8_utils import block_dequant
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
from sglang.srt.utils import add_prefix, LazyValue
from sglang.srt.distributed.parallel_strategy import DenseParallelStategy
from sglang.srt.layers.dp_attention import get_attention_tp_group, get_attention_tp_size, get_attention_tp_rank, get_dense_tp_group
from sglang.srt.distributed.decoder_comm_manager import DecoderCommMananger
from sglang.srt.managers.expert_distribution import (
    get_global_expert_distribution_recorder,
)
from sglang.srt.models.utils import (
    create_fused_set_kv_buffer_arg,
    enable_fused_set_kv_buffer,
)
from sglang.srt.models.deepseek_v32 import DeepseekV32MLA
from sglang.srt.layers.quantization.utils import should_ignore_quant_layer

from sglang.srt.utils import get_colorful_logger

_is_cuda = is_cuda()
_device_sm = get_device_sm()

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
        if is_sm90_supported() and hidden_states.shape[0] > 0:
            logits = dsv3_router_gemm(
                hidden_states, self.weight, out_dtype=torch.float32
            )
        else:
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
            gate_layer=self.gate
        )

    def forward(self, x, max_num_tokens_per_gpu) -> torch.Tensor:
        hidden_states = x
        shared_experts_out = self.shared_experts(hidden_states)
        experts_out = self.experts(x, max_num_tokens_per_gpu)

        return shared_experts_out + experts_out


class DeepseekV2MoE(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        layer_index: int = -1,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        self.layer_index = layer_index
        self.tp_size = get_tensor_model_parallel_world_size()
        self.n_shared_experts = config.n_shared_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.alt_stream = alt_stream
        if self.tp_size > config.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.n_routed_experts}."
            )

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        self.gate = MoEGate(config=config, prefix=add_prefix("gate", prefix))

        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("shared_experts", prefix),
                is_shared_expert=True
            )

        MoEImpl = MoELayer
        self.experts = MoEImpl(
            top_k=config.num_experts_per_tok,
            num_experts=config.n_routed_experts + global_server_args_dict["ep_num_redundant_experts"],
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            layer_index=layer_index,
            prefix=prefix
        )

        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            renormalize=config.norm_topk_prob,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            num_fused_shared_experts=0,
            topk_group=config.topk_group,
            correction_bias=self.gate.e_score_correction_bias,
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=self.experts.apply_routed_scaling_factor_on_output,
            output_format=TopKOutputFormat.STANDARD,
        )

    def get_moe_routed_weights(self):
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"] and "shared_experts" not in name
        ]

    def forward(self, hidden_states: torch.Tensor, num_global_tokens: int, max_num_tokens_per_gpu: int) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)

        # router_logits: (num_tokens, n_experts)
        router_logits = self.gate(hidden_states)
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
        if not self.experts.apply_routed_scaling_factor_on_output:
            routed_expert_output *= self.routed_scaling_factor

        shared_output = None
        with torch.cuda.stream(self.alt_stream):
            if self.n_shared_experts is not None and num_tokens > 0:
                shared_output = self.shared_experts(hidden_states)
        current_stream.wait_stream(self.alt_stream)

        if shared_output is not None:
            final_hidden_states = routed_expert_output + shared_output
        else:
            final_hidden_states = routed_expert_output
        return final_hidden_states.view(num_tokens, hidden_dim)


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
        alt_stream: Optional[torch.cuda.Stream] = None,
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
        self.alt_stream = alt_stream
        self.attention_backend = global_server_args_dict["attention_backend"]
        self.cli_factor = getattr(config, "cli_factor", 1)
        self.prefix = prefix

        # modification to rope_scaling must be done early enough, b/c e.g. Indexer needs it
        if rope_scaling:
            rope_scaling["rope_type"] = "deepseek_yarn"

        if self.q_lora_rank is not None:
            self.fused_qkv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("fused_qkv_a_proj_with_mqa", prefix),
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
            is_neox_style = not getattr(config, "rope_interleave", True)
            self.rotary_emb = get_rope(
                qk_rope_head_dim,
                rotary_dim=qk_rope_head_dim,
                max_position=max_position_embeddings,
                base=rope_theta,
                rope_scaling=rope_scaling,
                is_neox_style=is_neox_style,
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

        self.use_fused_set_kv_buffer = enable_fused_set_kv_buffer() and self.rotary_emb is not None

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

        has_fused_proj = hasattr(self, "fused_qkv_a_proj_with_mqa")

        is_packed_weight = (
            has_fused_proj
            and hasattr(self.fused_qkv_a_proj_with_mqa.quant_method, "quant_config")
            and self.fused_qkv_a_proj_with_mqa.quant_method.quant_config.get_name()
            in {"awq", "awq_marlin", "moe_wna16"}
        )
        self.use_min_latency_fused_a_gemm = (
            has_fused_proj
            and not is_packed_weight
            and self.fused_qkv_a_proj_with_mqa.weight.dtype == torch.bfloat16
            and self.fused_qkv_a_proj_with_mqa.weight.shape[0] == 2112
            and self.fused_qkv_a_proj_with_mqa.weight.shape[1] in [7168, 6144]
            and _is_cuda
            and _device_sm >= 90
        )
        self.dense_1_unqaunted = self.check_unquanted(self.q_b_proj) if hasattr(self, 'q_b_proj') else True

    def check_unquanted(self, module) -> bool:
        return module.quant_config is None or should_ignore_quant_layer(
            module.prefix, ignored_layers=getattr(module.quant_config, "ignored_layers", [])
        )

    def no_absorb(self, forward_batch: ForwardBatch) -> bool:
        if global_server_args_dict["enable_flashinfer_mla"]:
            # Flashinfer MLA: Do not absorb when enabling ragged prefill
            # Target verify and draft extend have few tokens, go through absorb.
            return (
                not global_server_args_dict["flashinfer_mla_disable_ragged"]
                and forward_batch.forward_mode.is_extend()
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend()
                and not forward_batch.captureing_prefill_graph
                # and sum(forward_batch.extend_prefix_lens_cpu) == 0
            )
        else:
            # Triton: Use normal computation for prefill and use weight absorption for extend/decode
            return (
                forward_batch.forward_mode.is_extend()
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend()
                and sum(forward_batch.extend_prefix_lens_cpu) == 0
                and not forward_batch.captureing_prefill_graph
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
        if self.no_absorb(forward_batch):
            if self.attention_backend == "duo_attn":
                if forward_batch.attn_backend.layers_type[self.attn_mha.layer_id]: # Full layer.
                    return self.forward_normal_chunked(positions, hidden_states, forward_batch, comm_manager)
                else: # Streaming layer.
                    return self.forward_normal_streamed(positions, hidden_states, forward_batch, comm_manager)
            else:
                return self.forward_normal_chunked(positions, hidden_states, forward_batch, comm_manager, block_scale)
        else:
            return self.forward_absorb(positions, hidden_states, forward_batch, comm_manager, block_scale, can_run_flashinfer_fusion)

    def forward_normal(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        comm_manager: DecoderCommMananger
    ) -> torch.Tensor:
        if self.q_lora_rank is not None:
            qkv = self.fused_qkv_a_proj_with_mqa(hidden_states)[0]
            qkv = comm_manager.pre_attn_comm(qkv, forward_batch.tp_num_tokens)
            q, latent_cache = qkv.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
            )
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]

        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        kv_a, k_pe = latent_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_a = self.kv_a_layernorm(kv_a)
        kv = self.kv_b_proj(kv_a)[0]
        kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope = kv[..., : self.qk_nope_head_dim]
        v = kv[..., self.qk_nope_head_dim :]
        if self.rotary_emb is not None:
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

        latent_cache = latent_cache.unsqueeze(1)
        latent_cache[:, :, : self.kv_lora_rank] = kv_a.unsqueeze(1)
        latent_cache[:, :, self.kv_lora_rank :] = k_pe.unsqueeze(1).clone()

        # Save latent cache
        forward_batch.token_to_kv_pool.set_kv_buffer(
            self.attn_mha, forward_batch.out_cache_loc, latent_cache, None
        )

        q[..., self.qk_nope_head_dim :] = q_pe
        k = torch.empty_like(q)
        k[..., : self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim :] = k_pe.unsqueeze(1)
        attn_output = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)

        attn_output = attn_output.reshape(-1, self.num_local_heads * self.v_head_dim)
        output, _ = self.o_proj(attn_output)
        return output

    def forward_absorb(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        comm_manager: DecoderCommMananger,
        block_scale: Optional[torch.Tensor] = None,
        can_run_flashinfer_fusion: bool = False,
    ) -> torch.Tensor:
        Q, K = self.forward_absorb_qkv_proj(hidden_states, positions, forward_batch, comm_manager, block_scale, can_run_flashinfer_fusion)
        output = self.forward_absorb_attn_o_proj(Q, K, forward_batch, comm_manager)
        return output

    def forward_absorb_qkv_proj(
        self,
        hidden_states,
        positions,
        forward_batch,
        comm_manager: DecoderCommMananger,
        block_scale: Optional[torch.Tensor] = None,
        can_run_flashinfer_fusion: Optional[bool] = None,
    ):
        if self.q_lora_rank is not None:
            if (
                (not isinstance(hidden_states, tuple))
                and hidden_states.shape[0] > 0
                and hidden_states.shape[0] <= 16
                and self.use_min_latency_fused_a_gemm
            ):
                qkv = dsv3_fused_a_gemm(
                    hidden_states, self.fused_qkv_a_proj_with_mqa.weight.T
                )
            else:
                if block_scale is not None:
                    qkv = self.fused_qkv_a_proj_with_mqa(hidden_states, block_scale, torch.bfloat16)[0]
                else:
                    qkv = self.fused_qkv_a_proj_with_mqa(hidden_states)[0]
            if can_run_flashinfer_fusion and self.layer_id != 0:
                qkv, q_contiguous, k_nope, block_scale = self.fused_qk_layernorm.forward_with_allgather_fusion(
                    get_attention_tp_group(),
                    qkv,
                    forward_batch.tp_num_tokens,
                    fuse_block_quant_fp8=not self.dense_1_unqaunted,
                )
                latent_cache = qkv[..., self.q_lora_rank:]
                q = self.q_b_proj(q_contiguous, block_scale, torch.bfloat16)[0].view(-1, self.num_local_heads, self.qk_head_dim)
            else:
                qkv = comm_manager.pre_attn_comm(qkv, forward_batch.tp_num_tokens)
                q, latent_cache = qkv.split(
                    [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
                )
                k_nope = latent_cache[..., : self.kv_lora_rank]

                # fused layernorm
                q_contiguous = torch.empty_like(q)
                if q.shape[0] > 0:
                    self.fused_qk_layernorm(input_q_a=q, input_kv_a=k_nope, output_q_a=q_contiguous)

                q = self.q_b_proj(q_contiguous)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            hidden_states = comm_manager.pre_attn_comm(hidden_states, forward_batch.tp_num_tokens)
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
            k_nope = latent_cache[..., : self.kv_lora_rank]
            self.kv_a_layernorm(k_nope, inplace=True).unsqueeze(1)

        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        batch_size = q_nope.shape[0]
        Q = torch.empty(
            batch_size, self.num_local_heads, self.kv_lora_rank + self.qk_rope_head_dim,
            dtype=q_nope.dtype, device=q_nope.device
        )
        # k_nope do the RMSNorm inplace, so latent_cache contain k_nope after norm and k_pe *before* rotate
        K = latent_cache.unsqueeze(1)
        q_nope_out_view = Q[..., : self.kv_lora_rank]
        torch.bmm(q_nope.transpose(0, 1), self.w_kc, out=q_nope_out_view.transpose(0, 1))
        # Apply RoPE directly on Q and K slices
        if self.rotary_emb is not None and q_nope.shape[0]>0:
            self.rotary_emb(
                positions,
                q_pe,
                K[..., self.kv_lora_rank :],
                fused_set_kv_buffer_arg=(
                    create_fused_set_kv_buffer_arg(
                        value=K[..., : self.kv_lora_rank],
                        layer=self.attn_mqa,
                        forward_batch=forward_batch,
                    )
                    if self.use_fused_set_kv_buffer
                    else None
                ),
                output_q_rope = Q[..., self.kv_lora_rank :],
            )
        else:
            Q[..., self.kv_lora_rank :] = q_pe

        if global_server_args_dict["enable_mla_l1_5_cache"]:
            # allgather q from all tp rank to get a full head Q
            Q = get_attention_tp_group().all_gather(Q, dim=1)

        return Q, K

    def merge_attn_lse_for_tp(self, attn_output, lse, comm_manager):
        # attn_output: (q_len, num_q_heads, h)
        # lse: (q_len, num_q_heads)
        attn_tp_group = get_attention_tp_group()
        q_len, num_q_heads = lse.shape
        gathered_lse = torch.zeros(
            (attn_tp_group.world_size, q_len, num_q_heads),
            device=lse.device,
            dtype=lse.dtype,
        )
        torch.distributed.all_gather_into_tensor(gathered_lse, lse, group=attn_tp_group.device_group, async_op=False)

        # Global logsumexp over TP ranks for mathematically correct attention merge.
        # local lse = log(sum_j exp(score_j)_local_rank)
        # global lse = log(sum_r exp(local_lse_r))
        full_seq_lse = torch.logsumexp(gathered_lse, dim=0)

        # use full_seq_lse to correct attn_output
        lse_correction = torch.exp(lse - full_seq_lse)
        attn_output *= lse_correction.unsqueeze(-1)

        # Reduce-scatter must shard by head dimension, not by flattened
        # token-head order. Otherwise q_len > 1 can mix tokens across shards.
        h = attn_output.shape[-1]
        attn_output = attn_output.transpose(0, 1).contiguous()  # (num_q_heads, q_len, h)
        attn_output = attn_output.reshape(num_q_heads, q_len * h)
        attn_output = attn_tp_group.reduce_scatter(attn_output)
        attn_output = attn_output.reshape(self.num_local_heads, q_len, h).transpose(0, 1).contiguous()
        return attn_output.reshape(-1, h)


    def forward_absorb_attn_o_proj(
        self, Q, K,
        forward_batch: ForwardBatch,
        comm_manager: DecoderCommMananger,
    ):
        q_nope_out = Q[..., : self.kv_lora_rank]
        q_pe = Q[..., self.kv_lora_rank :]
        k_nope = K[..., : self.kv_lora_rank]
        k_pe = K[..., self.kv_lora_rank :]

        if (
            self.attention_backend == "flashinfer_mla" or
            (self.attention_backend == "flashmla" and forward_batch.forward_mode == ForwardMode.EXTEND)
        ):
            attn_output, lse = self.attn_mqa(
                q_nope_out, k_nope, k_nope, forward_batch, q_pe=q_pe, k_pe= k_pe, \
                    save_kv_cache=not self.use_fused_set_kv_buffer
            )
        else:
            attn_output, *rest = self.attn_mqa(
                Q, K, k_nope, forward_batch,
                save_kv_cache=not self.use_fused_set_kv_buffer
            )
            lse = rest[0] if rest else None

        if getattr(forward_batch.token_to_kv_pool, "enable_mla_l1_5_cache", False):
            attn_output = self.merge_attn_lse_for_tp(attn_output, lse, comm_manager)

        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        attn_bmm_output = torch.bmm(attn_output.transpose(0, 1), self.w_vc)
        attn_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
        output, _ = self.o_proj(attn_output)
        return output

    def forward_normal_chunked(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        comm_manager: DecoderCommMananger,
        block_scale: Optional[torch.Tensor] = None,
    ):
        q, k, v = self.forward_normal_chunked_kv_prepare(positions, hidden_states, forward_batch, comm_manager, block_scale)
        output = self.forward_normal_chunked_kv_core(q, k, v, comm_manager, forward_batch)
        return output

    def forward_normal_chunked_kv_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        comm_manager: DecoderCommMananger,
        block_scale: Optional[torch.Tensor] = None,
    ):
        # In normal mha, the k and v tensors will become overly large when the prefix length is long.
        # To avoid this, we split the kv cache into chunks and process them one after another.
        # Since mha is compute friendly, the for loop induced here will not introduce significant overhead.
        # The top comments in https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/mla/common.py
        # will be helpful for understanding the purpose of this function.
        if self.q_lora_rank is not None:
            if block_scale is not None:
                qkv = self.fused_qkv_a_proj_with_mqa(hidden_states, block_scale, torch.bfloat16)[0]
            else:
                qkv = self.fused_qkv_a_proj_with_mqa(hidden_states)[0]
            qkv = comm_manager.pre_attn_comm(qkv, forward_batch.tp_num_tokens)
            q, latent_cache = qkv.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
            )
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            hidden_states = comm_manager.pre_attn_comm(hidden_states, forward_batch.tp_num_tokens)
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        _, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        kv_a, _ = latent_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        latent_cache = latent_cache.unsqueeze(1)
        kv_a = self.kv_a_layernorm(kv_a.contiguous())
        kv = self.kv_b_proj(kv_a)[0]
        kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope = kv[..., : self.qk_nope_head_dim]
        v = kv[..., self.qk_nope_head_dim :]
        k_pe = latent_cache[:, :, self.kv_lora_rank :]

        if self.rotary_emb is not None:
            self.rotary_emb(
                positions,
                q_pe,
                k_pe,
                fused_set_kv_buffer_arg=(
                    create_fused_set_kv_buffer_arg(
                        value=kv_a.unsqueeze(1),
                        layer=self.attn_mha,
                        forward_batch=forward_batch,
                    )
                    if self.use_fused_set_kv_buffer
                    else None
                )
            )

        q[..., self.qk_nope_head_dim :] = q_pe
        k = torch.empty_like(q)
        k[..., : self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim :] = k_pe
        # BF16 cache no longer needs explicit set_kv_buffer

        if not self.use_fused_set_kv_buffer:
            latent_cache[:, :, : self.kv_lora_rank] = kv_a.unsqueeze(1)
            latent_cache[:, :, self.kv_lora_rank :] = k_pe
            # Save latent cache
            forward_batch.token_to_kv_pool.set_kv_buffer(
                self.attn_mha, forward_batch.out_cache_loc, latent_cache, None
            )

        return q, k, v

    def forward_normal_chunked_kv_core(
            self, q, k, v,
            comm_manager: DecoderCommMananger,
            forward_batch: ForwardBatch
        ):
        if forward_batch.casual_chunk_attn is None:
            forward_batch.casual_chunk_attn = get_casual_attn(
                self.num_local_heads,
                self.num_local_heads,
                self.qk_nope_head_dim + self.qk_rope_head_dim,
                self.v_head_dim,
                q.dtype,
                q.device,
                forward_batch.extend_seq_lens,
                forward_batch.extend_seq_lens,
                getattr(forward_batch.attn_backend, "step_counter", None),
                forward_batch.extend_seq_lens_cpu,
                forward_batch.extend_seq_lens_cpu
            )

        if forward_batch.chunk_attns is None:
            chunks, forward_batch.chunk_kv_indices_list, chunks_cpu = get_chunks(
                forward_batch.extend_prefix_lens,
                forward_batch.extend_prefix_lens_cpu,
                req_to_token=forward_batch.req_to_token_pool.req_to_token,
                req_pool_indices=forward_batch.req_pool_indices,
                token_to_kv_pool=forward_batch.token_to_kv_pool,
            )
            forward_batch.chunk_attns = get_attns(
                self.num_local_heads,
                self.num_local_heads,
                self.qk_nope_head_dim + self.qk_rope_head_dim,
                self.v_head_dim,
                q.dtype,
                q.device,
                forward_batch.extend_seq_lens,
                chunks,
                chunks_cpu,
                forward_batch.extend_seq_lens_cpu,
            )

        attn_output = self._chunked_attn_mha(
            q=q,
            k=k,
            v=v,
            token_to_kv_pool=forward_batch.token_to_kv_pool,
            chunk_kv_indices_list=forward_batch.chunk_kv_indices_list,
            attns_list=forward_batch.chunk_attns,
            casual_attn=forward_batch.casual_chunk_attn,
            comm_manager=comm_manager,
        )

        attn_output = attn_output.reshape(-1, self.num_local_heads * self.v_head_dim)
        output, _ = self.o_proj(attn_output)
        return output

    def _gather_kv_from_tp_group(self, token_to_kv_pool, comm_manager, chunk_kv_indices, device):
        if token_to_kv_pool.quant_method == "per_token_head":
            raise NotImplementedError
        else:
            chunk_kv_indices, _, local_indices, per_rank_count, _, inv_perm = chunk_kv_indices
            latent_cache = token_to_kv_pool.get_key_contiguous(self.attn_mha.layer_id, local_indices)
            if latent_cache is None:
                latent_cache = torch.empty(
                    token_to_kv_pool.empty_cache_shape(),
                    dtype=token_to_kv_pool.store_dtype,
                    device=device
                )
            cache_shape = latent_cache.shape[1:]
            num_tokens = chunk_kv_indices.shape[0]

            full_latent_cache = comm_manager.attn_rsag.all_gather(
                latent_cache.view(latent_cache.shape[0], math.prod(cache_shape)),
                token_list_in_group=per_rank_count
            ).view(num_tokens, *cache_shape)

            full_latent_cache = full_latent_cache[inv_perm]

            return full_latent_cache

    def _load_kv_cache(self, token_to_kv_pool, comm_manager, chunk_kv_indices, device):
        if hasattr(token_to_kv_pool, 'enable_mla_l1_5_cache') and token_to_kv_pool.enable_mla_l1_5_cache:
            latent_cache = self._gather_kv_from_tp_group(token_to_kv_pool, comm_manager, chunk_kv_indices, device)
            kv_a_normed, k_pe = latent_cache.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
        else:
            kv_a_normed, k_pe = token_to_kv_pool.get_key_split_contiguous(
                self.attn_mha.layer_id, chunk_kv_indices
            )
        kv_a_normed = kv_a_normed.squeeze(1).contiguous()
        kv = self.kv_b_proj(kv_a_normed)[0]
        kv = kv.view(
            -1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        v = kv[..., self.qk_nope_head_dim :]
        k_nope = kv[..., : self.qk_nope_head_dim]

        k = torch.empty(
            (
                k_nope.shape[0],
                self.num_local_heads,
                self.qk_nope_head_dim + self.qk_rope_head_dim,
            ),
            dtype=v.dtype,
            device=v.device,
        )
        k[..., : self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim :] = k_pe
        return k, v

    def _chunked_attn_mha(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        token_to_kv_pool,
        chunk_kv_indices_list,
        # List[ChunkFlashInferAttn]
        attns_list,
        # ChunkFlashInferAttn
        casual_attn,
        comm_manager: DecoderCommMananger,
    ) -> torch.Tensor:

        compute_stream = torch.cuda.current_stream()
        gather_kv_stream = self.alt_stream
        # gather_kv_stream = compute_stream
        if gather_kv_stream is None:
            self.alt_stream = torch.cuda.Stream()
            gather_kv_stream = self.alt_stream
        gather_kv_stream.wait_stream(compute_stream)

        accum_output, accum_lse = None, None
        with torch.cuda.stream(compute_stream):
            accum_output, accum_lse = casual_attn.forward_extend(q, k, v, self.attn_mha.scaling, self.attn_mha.logit_cap)

        if len(chunk_kv_indices_list) == 0:
            return accum_output

        # Double-buffer KV tensors so loading and compute can overlap safely.
        # Each slot keeps its (k, v) alive until compute is done with it.
        # Overlap sketch (gather_kv_stream vs compute_stream):
        #   gather:  wait(compute) -> load -> ready -> load -> ready -> wait -> load -> ready -> ...
        #   compute:                           wait -> attn ->  done -> wait -> attn -> done -> ...
        buffer_slots = 2
        kv_buffers = [None] * buffer_slots
        kv_ready_events = [torch.cuda.Event() for _ in range(len(chunk_kv_indices_list))]
        kv_done_events = [torch.cuda.Event() for _ in range(buffer_slots)]

        for chunk_idx in range(0, len(chunk_kv_indices_list)):
            chunk_kv_indices = chunk_kv_indices_list[chunk_idx]
            slot = chunk_idx % buffer_slots

            with torch.cuda.stream(gather_kv_stream):
                # If this slot was used before, wait until compute stream is done.
                # make sure the computing kv buffer is not overwritten by newly loaded kv
                if kv_buffers[slot] is not None:
                    kv_done_events[slot].wait()
                if isinstance(chunk_kv_indices, torch.Tensor):
                    chunk_kv_indices.record_stream(gather_kv_stream)
                k_next, v_next = self._load_kv_cache(
                    token_to_kv_pool, comm_manager, chunk_kv_indices, q.device
                )
                kv_buffers[slot] = (k_next, v_next)
                kv_ready_events[chunk_idx].record(stream=gather_kv_stream)

            with torch.cuda.stream(compute_stream):
                attn = attns_list[chunk_idx]

                kv_ready_events[chunk_idx].wait()
                k, v = kv_buffers[slot]
                output, lse = attn.forward_extend(q, k, v, self.attn_mha.scaling, self.attn_mha.logit_cap)
                accum_output, accum_lse = merge_state(accum_output, accum_lse, output, lse)
                kv_done_events[slot].record(stream=compute_stream)

        return accum_output

    def forward_normal_streamed(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        comm_manager
    ):
        # Reuse the prepare func of chunked.
        q, k, v = self.forward_normal_chunked_kv_prepare(positions, hidden_states, forward_batch, comm_manager)
        output = self.forward_normal_streamed_kv_core(q, k, v, forward_batch)
        return output

    def forward_normal_streamed_kv_core(self, q, k, v, forward_batch: ForwardBatch):
        # Reused across layers.
        if forward_batch.streamed_attn is None:
            forward_batch.streamed_kv_indices, streamed_lens = get_streamed_kv_indices(
                forward_batch.extend_seq_lens.to(torch.int32),
                forward_batch.extend_seq_lens_cpu,
                forward_batch.seq_lens.to(torch.int32),
                forward_batch.seq_lens_cpu,
                forward_batch.req_to_token_pool.req_to_token,
                forward_batch.req_pool_indices,
                not any(forward_batch.extend_prefix_lens_cpu),
            )
            forward_batch.streamed_attn = get_streamed_attn(
                self.num_local_heads,
                self.num_local_heads,
                self.qk_nope_head_dim + self.qk_rope_head_dim,
                self.v_head_dim,
                q.dtype,
                forward_batch.extend_seq_lens,
                streamed_lens,
                getattr(forward_batch.attn_backend, "streaming_info", None),
                getattr(forward_batch.attn_backend, "layers_head_mask_type", None),
                getattr(forward_batch.attn_backend, "step_counter", None),
            )

        attn_output = self._streamed_attn_mha(
            q=q,
            k=k,
            v=v,
            token_to_kv_pool=forward_batch.token_to_kv_pool,
            streamed_kv_indices=forward_batch.streamed_kv_indices,
            streamed_attn=forward_batch.streamed_attn,
        )

        attn_output = attn_output.reshape(-1, self.num_local_heads * self.v_head_dim)
        output, _ = self.o_proj(attn_output)
        return output

    def _streamed_attn_mha(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        token_to_kv_pool,
        streamed_kv_indices,
        streamed_attn,
    ) -> torch.Tensor:
        if streamed_kv_indices is not None:
            # Fetch latent cache from memory pool with precomputed streamed kv indices
            latent_cache_buf = token_to_kv_pool.get_key_buffer(
                self.attn_mha.layer_id
            )
            latent_cache = latent_cache_buf[streamed_kv_indices].contiguous()

            kv_a_normed, k_pe = latent_cache.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            kv_a_normed = kv_a_normed.squeeze(1).contiguous()
            kv = self.kv_b_proj(kv_a_normed)[0]
            kv = kv.view(
                -1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            v = kv[..., self.qk_nope_head_dim :]
            k_nope = kv[..., : self.qk_nope_head_dim]

            k = torch.empty(
                (
                    k_nope.shape[0],
                    self.num_local_heads,
                    self.qk_nope_head_dim + self.qk_rope_head_dim,
                ),
                dtype=v.dtype,
                device=v.device,
            )
            k[..., : self.qk_nope_head_dim] = k_nope
            k[..., self.qk_nope_head_dim :] = k_pe

        output = streamed_attn.forward_extend(q, k, v, self.attn_mha.scaling, self.attn_mha.layer_id)

        return output


class DeepseekV2DecoderLayer(nn.Module):

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
            alt_stream=alt_stream,
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
        hidden_states, start_idx, end_idx = self.decoder_comm_manager.pre_mlp_comm(
            hidden_states, forward_batch, tp_num_tokens
        )
        if self.is_moe_layer:
            hidden_states = self.mlp(hidden_states, num_global_tokens, max_num_tokens_per_gpu)
        else:
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
        self.alt_stream = torch.cuda.Stream()
        if FLLM_IS_CP:
            self.cp_size = get_attention_tp_size()
            self.cp_rank = get_attention_tp_rank()
        # config.num_hidden_layers = 50; self.start_layer,self.end_layer = 0, 50
        self.layers = nn.ModuleList(
            [
                DeepseekV2DecoderLayer(
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
            lambda: {
                layer_id: layer.mlp.get_moe_routed_weights()
                for layer_id, layer in enumerate(self.model.layers)
                if isinstance(layer.mlp, DeepseekV2MoE)
            }
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

                    if (
                        fuse_qkv_a_proj
                        and (
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
            self_attn.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
            self_attn.w_vc = w_vc.contiguous().transpose(1, 2)

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def set_oe_and_head(self, over_embedding, head):
        del self.model.embed_tokens
        self.model.embed_tokens = over_embedding
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_routed_experts,
            num_groups=config.n_group,
        )


class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):
    pass


EntryClass = [DeepseekV2ForCausalLM, DeepseekV3ForCausalLM]
