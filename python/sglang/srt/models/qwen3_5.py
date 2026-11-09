# ruff: noqa: N801
import enum
import logging
from collections.abc import Iterable
from typing import Any, Optional

import torch
import triton
import triton.language as tl
from torch import nn

from sglang.srt.configs.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from sglang.srt.distributed import (
    get_moe_expert_parallel_rank,
    get_moe_expert_parallel_world_size,
    get_pp_group,
)
from sglang.srt.distributed.decoder_comm_manager import DecoderCommMananger
from sglang.srt.distributed.model_tensor_tracer import get_load_number_layers
from sglang.srt.env import global_server_args_dict
from sglang.srt.layers.attention.dp_attention import (
    get_attention_tp_group,
    get_attention_tp_rank,
    get_attention_tp_size,
)
from sglang.srt.layers.attention.fla.layernorm_gated import RMSNorm as RMSNormGated
from sglang.srt.layers.attention.mamba.mamba import mamba_v2_sharded_weight_loader
from sglang.srt.layers.attention.radix_attention import RadixAttention
from sglang.srt.layers.embedding.rotary_embedding import get_rope
from sglang.srt.layers.embedding.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.moe.layouts.mapping import make_expert_params_mapping
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.sampler.logits_processor import LogitsProcessor
from sglang.srt.managers.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from sglang.srt.models.qwen2_moe import Qwen2MoeMLP
from sglang.srt.models.qwen3_moe import Qwen3MoeSparseMoeBlock
from sglang.srt.ops.layernorm import GemmaRMSNorm
from sglang.srt.utils import add_prefix, is_cuda, is_npu, make_layers, set_weight_attrs
from sglang.srt.utils.common import divide

if not is_npu():
    pass


logger = logging.getLogger(__name__)
_is_cuda = is_cuda()


# g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
@triton.jit
def fused_gdn_gating_kernel(
    g,
    A_log,
    a,
    dt_bias,
    seq_len,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off
    mask = head_off < NUM_HEADS
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a = tl.load(a + off, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x)
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> torch.Tensor:
    batch, num_heads = a.shape
    seq_len = 1
    grid = (batch, seq_len, triton.cdiv(num_heads, 8))
    g = torch.empty_like(a, dtype=torch.float32)
    fused_gdn_gating_kernel[grid](g, A_log, a, dt_bias, seq_len, num_heads, beta, threshold, 8, num_warps=1)
    return g


class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(
        self,
        config: Qwen3_5MoeTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        layer_id: int = 0,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.attn_tp_rank = get_attention_tp_rank()
        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_group = get_attention_tp_group()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.alt_stream = alt_stream

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_id = layer_id
        self.activation = config.hidden_act
        self.layer_norm_epsilon = config.rms_norm_eps

        self.split_arg_list_qkvz = [(self.key_dim + self.key_dim + self.value_dim) // self.attn_tp_size, self.value_dim // self.attn_tp_size]
        self.split_arg_list_ba = [
            self.num_v_heads // self.attn_tp_size,
            self.num_v_heads // self.attn_tp_size,
        ]

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size, output_size=self.conv_dim, bias=False, quant_config=None, outside_tp_group=self.attn_tp_group
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)
        # projection of the input hidden states

        self.in_proj_qkvz = MergedColumnParallelLinear(
            self.hidden_size,
            [self.key_dim, self.key_dim, self.value_dim, self.value_dim],
            bias=False,
            quant_config=quant_config,
            outside_tp_group=self.attn_tp_group,
        )
        self.in_proj_ba = MergedColumnParallelLinear(
            self.hidden_size, [self.num_v_heads, self.num_v_heads], bias=False, quant_config=None, outside_tp_group=self.attn_tp_group
        )

        query_key_settings = (self.key_dim, 0, False)
        value_settings = (self.value_dim, 0, False)

        delattr(self.conv1d.weight, "weight_loader")
        set_weight_attrs(
            self.conv1d.weight,
            {
                "weight_loader": mamba_v2_sharded_weight_loader(
                    [
                        query_key_settings,
                        query_key_settings,
                        value_settings,
                    ],
                    self.attn_tp_size,
                    self.attn_tp_rank,
                )
            },
        )

        # selective projection used to make dt, B and C input dependent

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads // self.attn_tp_size))

        A = torch.empty(divide(self.num_v_heads, self.attn_tp_size), dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(0)})
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        self.norm = RMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            group_size=None,
            norm_before_gate=True,
            device=torch.cuda.current_device(),
            dtype=config.torch_dtype,
        )

        self.out_proj = RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            input_is_parallel=True,
            reduce_results=False,
            outside_tp_group=self.attn_tp_group,
        )

    def fix_query_key_value_ordering(self, mixed_qkvz, mixed_ba):
        (mixed_qkv, z) = torch.split(mixed_qkvz, self.split_arg_list_qkvz, dim=-1)
        (b, a) = torch.split(mixed_ba, self.split_arg_list_ba, dim=-1)
        z = z.reshape(z.size(0), -1, self.head_v_dim)

        return mixed_qkv, z, b, a

    def _forward_input_proj(self, hidden_states: torch.Tensor):
        DUAL_STREAM_TOKEN_THRESHOLD = 1024
        seq_len, _ = hidden_states.shape
        if seq_len < DUAL_STREAM_TOKEN_THRESHOLD:
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
            with torch.cuda.stream(self.alt_stream):
                projected_states_ba, _ = self.in_proj_ba(hidden_states)
            current_stream.wait_stream(self.alt_stream)
        else:
            projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
            projected_states_ba, _ = self.in_proj_ba(hidden_states)
        return projected_states_qkvz, projected_states_ba

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        seq_len, _ = hidden_states.shape
        projected_states_qkvz, projected_states_ba = self._forward_input_proj(hidden_states)
        mixed_qkv, z, b, a = self.fix_query_key_value_ordering(projected_states_qkvz, projected_states_ba)

        # 2. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))

        kwargs = {
            "mixed_qkv": mixed_qkv,
            "conv_weights": conv_weights,
            "bias": self.conv1d.bias,
            "activation": self.activation,
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
            "attention_tp_size": self.attn_tp_size,
            "head_k_dim": self.head_k_dim,
            "head_v_dim": self.head_v_dim,
            "a": a,
            "b": b,
            "A_log": self.A_log,
            "dt_bias": self.dt_bias,
            "layer_id": self.layer_id,
            "seq_len": seq_len,
            "z": z,
        }

        core_attn_out = forward_batch.attn_backend.forward(
            q=None,
            k=None,
            v=None,
            layer=None,
            forward_batch=forward_batch,
            **kwargs,
        )

        z_shape_og = z.shape
        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(*core_attn_out.shape[:-2], -1)

        output, _ = self.out_proj(core_attn_out)
        return output


class Qwen3_5HybridLinearDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3_5MoeTextConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.linear_attn = Qwen3_5GatedDeltaNet(config, quant_config, layer_id, alt_stream)

        # Qwen3Next all layers are sparse and have no nextn now
        self.is_layer_sparse = True
        self.layer_id = layer_id

        if self.is_layer_sparse:
            self.mlp = Qwen3MoeSparseMoeBlock(config=config, quant_config=quant_config, layer_index=layer_id)
        else:
            self.mlp = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
            )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.decoder_comm_manager = DecoderCommMananger(
            layer_id=self.layer_id,
            attn_parallel_strategy=global_server_args_dict["attn_parallel_strategy"],
            dense_parallel_strategy=global_server_args_dict["dense_parallel_strategy"],
            moe_parallel_strategy=global_server_args_dict["moe_parallel_strategy"],
            is_moe_layer=self.is_layer_sparse,
            num_layers=config.num_hidden_layers,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        **kwargs,
    ):
        forward_batch: ForwardBatch = kwargs.get("forward_batch")
        tp_num_tokens = kwargs.get("tp_num_tokens")
        num_global_tokens, max_num_tokens_per_gpu = forward_batch.get_num_tokens(tp_num_tokens)

        if not forward_batch.forward_mode.is_idle():
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)

            hidden_states = self.decoder_comm_manager.pre_attn_comm(hidden_states, tp_num_tokens)

            hidden_states = self.linear_attn(
                hidden_states,
                forward_batch,
            )

            hidden_states, residual = self.decoder_comm_manager.post_attn_comm(hidden_states, residual, tp_num_tokens)

            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        hidden_states = self.forward_mlp(hidden_states, residual, forward_batch, num_global_tokens, max_num_tokens_per_gpu, tp_num_tokens)

        return hidden_states, residual

    def forward_mlp(self, hidden_states, residual, forward_batch, num_global_tokens, max_num_tokens_per_gpu, tp_num_tokens):
        hidden_states, start_idx, end_idx = self.decoder_comm_manager.pre_mlp_comm(hidden_states, forward_batch, tp_num_tokens)
        hidden_states = self.mlp(hidden_states, num_global_tokens, max_num_tokens_per_gpu) if self.is_layer_sparse else self.mlp(hidden_states)
        hidden_states, residual = self.decoder_comm_manager.post_mlp_comm(hidden_states, residual, tp_num_tokens, forward_batch)
        if start_idx is not None and end_idx is not None:
            hidden_states = hidden_states[start_idx:end_idx]
        return hidden_states


class Qwen3_5HybridAttentionDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3_5MoeTextConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.attn_tp_rank = get_attention_tp_rank()
        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_group = get_attention_tp_group()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % self.attn_tp_size == 0
        self.num_heads = self.total_num_heads // self.attn_tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= self.attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % self.attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert self.attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.attn_tp_size)
        self.head_dim = config.head_dim or (self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.layer_id = layer_id

        if hasattr(config, "rope_parameters"):
            self.rope_scaling = getattr(config, "rope_parameters", None)
        else:
            self.rope_scaling = getattr(config, "rope_scaling", None)
        self.rope_theta = self.rope_scaling.get("rope_theta", 10000)
        self.partial_rotary_factor = self.rope_scaling.get("partial_rotary_factor", 1.0)

        self.attn_output_gate = getattr(config, "attn_output_gate", True)
        if self.attn_output_gate:
            logger.warning_once("using attn output gate!")

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            rope_scaling=self.rope_scaling,
            base=self.rope_theta,
            partial_rotary_factor=self.partial_rotary_factor,
            is_neox_style=True,
            dtype=torch.get_default_dtype(),  # see impl of get_rope
        )

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
            outside_tp_group=self.attn_tp_group,
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
            outside_tp_group=self.attn_tp_group,
        )

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
        )

        # Qwen3Next all layers are sparse and have no nextn now
        self.is_layer_sparse = True

        if self.is_layer_sparse:
            self.mlp = Qwen3MoeSparseMoeBlock(config=config, quant_config=quant_config, layer_index=layer_id)
        else:
            self.mlp = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
            )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.decoder_comm_manager = DecoderCommMananger(
            layer_id=self.layer_id,
            attn_parallel_strategy=global_server_args_dict["attn_parallel_strategy"],
            dense_parallel_strategy=global_server_args_dict["dense_parallel_strategy"],
            moe_parallel_strategy=global_server_args_dict["moe_parallel_strategy"],
            is_moe_layer=self.is_layer_sparse,
            num_layers=config.num_hidden_layers,
        )
        self.alt_stream = alt_stream

    def _apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # overlap qk norm
        if self.alt_stream is not None:
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            q_by_head = q.reshape(-1, self.head_dim)
            q_by_head = self.q_norm(q_by_head)
            with torch.cuda.stream(self.alt_stream):
                k_by_head = k.reshape(-1, self.head_dim)
                k_by_head = self.k_norm(k_by_head)
            current_stream.wait_stream(self.alt_stream)
        else:
            q_by_head = q.reshape(-1, self.head_dim)
            q_by_head = self.q_norm(q_by_head)
            k_by_head = k.reshape(-1, self.head_dim)
            k_by_head = self.k_norm(k_by_head)
        q = q_by_head.view(q.shape)
        k = k_by_head.view(k.shape)
        return q, k

    def self_attention(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)

        if self.attn_output_gate:
            q_gate, k, v = qkv.split([self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
            orig_shape = q_gate.shape[:-1]
            q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(*orig_shape, -1)
            gate = gate.reshape(*orig_shape, -1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q, k = self._apply_qk_norm(q, k)

        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v, forward_batch)

        if self.attn_output_gate:
            gate = torch.sigmoid(gate)
            attn_output = attn_output * gate

        output, _ = self.o_proj(attn_output)
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ):
        tp_num_tokens = kwargs.get("tp_num_tokens")
        num_global_tokens, max_num_tokens_per_gpu = forward_batch.get_num_tokens(tp_num_tokens)

        if not forward_batch.forward_mode.is_idle():
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)
            hidden_states = self.decoder_comm_manager.pre_attn_comm(hidden_states, tp_num_tokens)

            hidden_states = self.self_attention(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
            hidden_states, residual = self.decoder_comm_manager.post_attn_comm(hidden_states, residual, tp_num_tokens)
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        hidden_states = self.forward_mlp(hidden_states, residual, forward_batch, num_global_tokens, max_num_tokens_per_gpu, tp_num_tokens)

        return hidden_states, residual

    def forward_mlp(self, hidden_states, residual, forward_batch, num_global_tokens, max_num_tokens_per_gpu, tp_num_tokens):
        hidden_states, start_idx, end_idx = self.decoder_comm_manager.pre_mlp_comm(hidden_states, forward_batch, tp_num_tokens)
        hidden_states = self.mlp(hidden_states, num_global_tokens, max_num_tokens_per_gpu) if self.is_layer_sparse else self.mlp(hidden_states)
        hidden_states, residual = self.decoder_comm_manager.post_mlp_comm(hidden_states, residual, tp_num_tokens, forward_batch)
        if start_idx is not None and end_idx is not None:
            hidden_states = hidden_states[start_idx:end_idx]
        return hidden_states


ALL_DECODER_LAYER_TYPES = {
    # Need two classes because linear and fullattn have different weights and compute logic.
    "attention": Qwen3_5HybridAttentionDecoderLayer,
    "linear_attention": Qwen3_5HybridLinearDecoderLayer,
}


class Qwen3_5MoeModel(nn.Module):
    def __init__(
        self,
        config: Qwen3_5MoeTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        alt_stream = torch.cuda.Stream() if _is_cuda else None

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            enable_tp=not global_server_args_dict["enable_dp_attention"],
            prefix=add_prefix("embed_tokens", prefix),
        )

        def get_layer(idx: int, prefix: str):
            layer_class = ALL_DECODER_LAYER_TYPES[config.layers_block_type[idx]]
            return layer_class(
                config,
                idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            )

        self.layers = make_layers(config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers")

        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.infer_count = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        # mamba_cache_params: MambaCacheParams,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # pass a sequence index tensor, that is required for
        # proper continuous batching computation including
        # chunked prefill
        hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        tp_num_tokens = hidden_states.shape[0]

        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                layer_id=i,
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                forward_batch=forward_batch,
                tp_num_tokens=tp_num_tokens,
            )

        if not forward_batch.forward_mode.is_idle():
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)
            hidden_states, _ = layer.decoder_comm_manager.post_final_norm_comm(hidden_states, residual, tp_num_tokens)

        return hidden_states


class HybridLayerType(enum.Enum):
    full_attention = "attention"
    swa_attention = "swa_attention"
    linear_attention = "linear_attention"
    mamba2 = "mamba"


class Qwen3_5MoeForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: Qwen3_5MoeTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        if get_load_number_layers() != 0:
            config.num_hidden_layers = get_load_number_layers()
        super().__init__()
        self.config = config
        self.pp_group = get_pp_group()
        assert self.pp_group.is_first_rank and self.pp_group.is_last_rank
        self.quant_config = quant_config
        self.model = Qwen3_5MoeModel(config, quant_config, prefix=add_prefix("model", prefix))
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
        self.lm_head = self.lm_head.float()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        hidden_states = self.model(input_ids, positions, forward_batch, inputs_embeds)

        return self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch)

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def get_param(self, params_dict, name):
        if name in params_dict:
            return params_dict[name]

        if "language_model." in name:
            name = name.replace("language_model.", "")
            if name in params_dict:
                return params_dict[name]

        if ".visua" not in name:
            logger.debug(f"The {name} is not in the model.")
        return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]], is_mtp: bool = False) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        num_v_heads = self.config.linear_num_value_heads
        num_k_heads = self.config.linear_num_key_heads
        head_k_dim = self.config.linear_key_head_dim
        head_v_dim = self.config.linear_value_head_dim
        key_dim = head_k_dim * num_k_heads
        value_dim = head_v_dim * num_v_heads
        stacked_params_mapping.extend(
            [
                ("in_proj_qkvz", "in_proj_z", 3),
                ("in_proj_ba", "in_proj_b", 0),
                ("in_proj_ba", "in_proj_a", 1),
            ]
        )
        in_proj_qkv_stacked_params_mapping = [
            # (param_name, weight_name, shard_id)
            ("in_proj_qkvz", "in_proj_qkv", 0),
            ("in_proj_qkvz", "in_proj_qkv", 1),
            ("in_proj_qkvz", "in_proj_qkv", 2),
        ]
        qkv_size = [0, key_dim, key_dim * 2, key_dim * 2 + value_dim]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )
        fused_expert_params_mapping = [
            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
            ("experts.w2_weight", "experts.down_proj", 0, "w2"),
        ]

        def load_fused_expert_weights(
            name: str,
            params_dict: dict,
            loaded_weight: torch.Tensor,
            shard_id: str,
            num_experts: int,
        ):
            num_local_experts = num_experts // get_moe_expert_parallel_world_size()
            start_expert = num_local_experts * get_moe_expert_parallel_rank()
            param = self.get_param(params_dict, name)
            if param is None:
                return False
            weight_loader = param.weight_loader
            # let ep moe layer to gracefully handle expert_ids that do not belong to local moe rank
            for expert_id in range(num_local_experts):
                local_expert_id = start_expert + expert_id
                curr_expert_weight = loaded_weight[local_expert_id]
                weight_loader(
                    param,
                    curr_expert_weight,
                    shard_id=shard_id,
                    local_expert_id=expert_id,
                )
            return True

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            is_fused_expert = False
            if is_mtp:
                if "mtp" not in name:
                    continue

                if name in [
                    "mtp.fc.weight",
                    "mtp.pre_fc_norm_embedding.weight",
                    "mtp.pre_fc_norm_hidden.weight",
                ]:
                    name = name.replace("mtp.", "")
                else:
                    name = name.replace("mtp", "model")

            if not is_mtp and "mtp" in name:
                continue

            if "rotary_emb.inv_freq" in name:
                continue

            if ".self_attn." in name:
                name = name.replace(".self_attn", "")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if "experts.gate_up_proj" in name or "experts.down_proj" in name:
                    is_fused_expert = True
                    expert_params_mapping = fused_expert_params_mapping
                if weight_name not in name:
                    continue

                # TODO(fix mtp loading)
                if "mlp.experts" in name:
                    continue

                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip layers on other devices.
                # if is_pp_missing_parameter(name, self):
                #     continue
                param = self.get_param(params_dict, name)
                if param is None:
                    continue
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for param_name, weight_name, shard_id in in_proj_qkv_stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    if "in_proj_qkv" in name:
                        begin_size = qkv_size[shard_id]
                        end_size = qkv_size[shard_id + 1]
                        if name.endswith(".weight_scale_inv"):
                            begin_size = begin_size // self.config.quantization_config["weight_block_size"][0]
                            end_size = end_size // self.config.quantization_config["weight_block_size"][0]
                        current_loaded_weight = loaded_weight[begin_size:end_size, :]
                    else:
                        continue
                    replace_name = name.replace(weight_name, param_name)
                    param = self.get_param(params_dict, replace_name)
                    if param is None:
                        continue
                    weight_loader = param.weight_loader
                    weight_loader(param, current_loaded_weight, shard_id)
                    if shard_id >= 2:
                        break
                else:
                    if "mlp.experts." in name:
                        for mapping in expert_params_mapping:
                            param_name, weight_name, expert_id, shard_id = mapping
                            if weight_name not in name:
                                continue
                            name_mapped = name.replace(weight_name, param_name)
                            if is_fused_expert:
                                if "experts.gate_up_proj" in name:
                                    loaded_weight = loaded_weight.chunk(2, dim=-2)
                                    load_fused_expert_weights(
                                        name_mapped,
                                        params_dict,
                                        loaded_weight[0],
                                        "w1",
                                        self.config.num_experts,
                                    )
                                    load_fused_expert_weights(
                                        name_mapped,
                                        params_dict,
                                        loaded_weight[1],
                                        "w3",
                                        self.config.num_experts,
                                    )
                                else:
                                    load_fused_expert_weights(
                                        name_mapped,
                                        params_dict,
                                        loaded_weight,
                                        "w2",
                                        self.config.num_experts,
                                    )
                                name = name_mapped
                            else:
                                if (name.endswith(".bias") or name.endswith("_bias")) and name not in params_dict:
                                    continue
                                param = self.get_param(params_dict, name_mapped)
                                if param is None:
                                    continue
                                weight_loader = param.weight_loader
                                weight_loader(
                                    param,
                                    loaded_weight,
                                    shard_id=shard_id,
                                    local_expert_id=expert_id,
                                )
                            break
                    else:
                        # Skip loading extra bias for GPTQ models.
                        if name.endswith(".bias") and name not in params_dict:
                            continue
                        # if is_pp_missing_parameter(name, self):
                        #     continue

                        param = self.get_param(params_dict, name)
                        if param is None:
                            continue
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        try:
                            weight_loader(param, loaded_weight)
                        except Exception as e:
                            logger.warning(f"Error loading {name}: {e}")
            loaded_params.add(name)
        return loaded_params

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_experts,
            num_groups=None,
        )


EntryClass = Qwen3_5MoeForCausalLM
