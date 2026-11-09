# Adapted from: https://github.com/vllm-project/vllm/blob/0384aa7150c4c9778efca041ffd1beb3ad2bd694/vllm/model_executor/models/kimi_linear.py

import os
from collections.abc import Iterable
from typing import Optional

import torch
from einops import rearrange
from torch import nn

from sglang.srt.configs.kimi_linear import KimiLinearConfig
from sglang.srt.distributed import (
    get_tensor_model_parallel_world_size,
)
from sglang.srt.distributed.decoder_comm_manager import DecoderCommMananger
from sglang.srt.env import global_server_args_dict
from sglang.srt.layers.attention.fla.kda import FusedRMSNormGated
from sglang.srt.layers.embedding.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.moe.layer import MoELayer
from sglang.srt.layers.moe.layouts.mapping import make_expert_params_mapping
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.sampler.logits_processor import LogitsProcessor
from sglang.srt.managers.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
    sharded_weight_loader,
)
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA as KimiMLAAttention
from sglang.srt.models.llama import LlamaMLP as KimiMLP
from sglang.srt.ops.activation import SiluAndMul
from sglang.srt.ops.layernorm import RMSNorm
from sglang.srt.ops.topk import TopK, TopKOutputFormat
from sglang.srt.utils import add_prefix, make_layers, set_weight_attrs
from sglang.srt.utils.common import divide


class KimiSharedExpertMLP(nn.Module):
    """Shared expert MLP matching the active MoE parallel strategy."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. Only silu is supported for now.")
        if global_server_args_dict["enable_ep_moe"]:
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
        else:
            self.gate_up_proj = MergedColumnParallelLinear(
                hidden_size,
                [intermediate_size, intermediate_size],
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("gate_up_proj", prefix),
            )
            self.down_proj = RowParallelLinear(
                intermediate_size,
                hidden_size,
                bias=False,
                reduce_results=False,
                quant_config=quant_config,
                prefix=add_prefix("down_proj", prefix),
            )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 0:
            return x
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class KimiMoE(nn.Module):
    def __init__(
        self,
        config: KimiLinearConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        layer_idx: int = 0,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        moe_intermediate_size = config.moe_intermediate_size
        num_experts = config.num_experts
        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = config.routed_scaling_factor
        self.num_shared_experts = config.num_shared_experts
        self.layer_idx = layer_idx
        self.alt_stream = alt_stream

        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {config.hidden_act}. Only silu is supported for now.")

        # Gate always runs at half / full precision for now.
        self.gate = ReplicatedLinear(
            hidden_size,
            num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

        self.gate.e_score_correction_bias = nn.Parameter(torch.empty((num_experts), dtype=torch.float32))

        self.num_experts = config.n_routed_experts + global_server_args_dict["ep_num_redundant_experts"]
        self.experts = MoELayer(
            top_k=config.num_experts_per_token,
            num_experts=self.num_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            layer_index=self.layer_idx,
            prefix=add_prefix("experts", prefix),
        )

        self.topk = TopK(
            top_k=config.num_experts_per_token,
            layer_id=self.layer_idx,
            use_grouped_topk=config.use_grouped_topk,
            num_expert_group=config.num_expert_group,
            topk_group=config.topk_group,
            renormalize=config.moe_renormalize,
            correction_bias=self.gate.e_score_correction_bias,
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=self.experts.apply_routed_scaling_factor_on_output,
            output_format=TopKOutputFormat.STANDARD,
            topk_indices_dtype=torch.int64 if global_server_args_dict.get("enable_deep_ep", False) else torch.int32,
        )

        if self.num_shared_experts is not None:
            intermediate_size = moe_intermediate_size * self.num_shared_experts
            self.shared_experts = KimiSharedExpertMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("shared_experts", prefix),
            )

    def forward(self, hidden_states: torch.Tensor, num_global_tokens: int, max_num_tokens_per_gpu: int) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)
        shared_output = None
        DUAL_STREAM_TOKEN_THRESHOLD = 1024
        if (
            self.alt_stream is not None
            and self.num_shared_experts is not None
            and hidden_states.shape[0] > 0
            and hidden_states.shape[0] <= DUAL_STREAM_TOKEN_THRESHOLD
        ):
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)

            shared_output = self.shared_experts(hidden_states.clone())

            with torch.cuda.stream(self.alt_stream):
                router_logits, _ = self.gate(hidden_states)
                router_logits = router_logits.to(torch.float32)
                if hidden_states.shape[0] > 0:
                    topk_output = self.topk(hidden_states, router_logits)
                else:
                    topk_output = self.topk.empty_topk_output(hidden_states.device)
                final_hidden_states = self.experts(
                    hidden_states=hidden_states,
                    topk_output=topk_output,
                    num_global_tokens=num_global_tokens,
                    max_num_tokens_per_gpu=max_num_tokens_per_gpu,
                )

            current_stream.wait_stream(self.alt_stream)
        else:
            if self.num_shared_experts is not None:
                shared_output = self.shared_experts(hidden_states)
            router_logits, _ = self.gate(hidden_states)
            router_logits = router_logits.to(torch.float32)
            topk_output = self.topk(hidden_states, router_logits) if hidden_states.shape[0] > 0 else self.topk.empty_topk_output(hidden_states.device)
            final_hidden_states = self.experts(
                hidden_states=hidden_states,
                topk_output=topk_output,
                num_global_tokens=num_global_tokens,
                max_num_tokens_per_gpu=max_num_tokens_per_gpu,
            )

        if not self.experts.apply_routed_scaling_factor_on_output:
            final_hidden_states *= self.routed_scaling_factor

        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output

        return final_hidden_states.view(num_tokens, hidden_size)


class KimiDeltaAttention(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        config: KimiLinearConfig,
        quant_config: Optional[QuantizationConfig] = None,
        rms_norm_eps: float = 1e-5,
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.hidden_size = hidden_size
        self.config = config
        self.head_dim = config.linear_attn_config["head_dim"]
        self.num_heads = config.linear_attn_config["num_heads"]
        self.use_fgb_kda = str(getattr(config, "linear_method", "")).upper() == "FGBKDA"
        # Optional safe-gate bound via env (default None -> standard softplus gate).
        # Example: FLA_KDA_LOWER_BOUND=-5.0
        _lb = os.getenv("FLA_KDA_LOWER_BOUND")
        self.lower_bound = float(_lb) if _lb is not None and _lb != "" else None
        self.layer_idx = layer_idx
        self.prefix = prefix
        assert self.num_heads % self.tp_size == 0
        self.local_num_heads = divide(self.num_heads, self.tp_size)

        projection_size = self.head_dim * self.num_heads
        self.conv_size = config.linear_attn_config["short_conv_kernel_size"]

        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            projection_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size,
            projection_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size,
            projection_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.v_proj",
        )

        self.f_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.f_a_proj",
        )

        self.f_b_proj = ColumnParallelLinear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.f_b_proj",
        )
        self.dt_bias = nn.Parameter(torch.empty(divide(projection_size, self.tp_size), dtype=torch.float32))

        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        if self.use_fgb_kda:
            projection_size = self.head_dim * self.num_heads
            self.b_proj = nn.Sequential(
                ReplicatedLinear(
                    self.hidden_size,
                    self.head_dim,
                    bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.b_proj.0",
                ),
                ColumnParallelLinear(
                    self.head_dim,
                    projection_size,
                    bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.b_proj.1",
                ),
            )
        else:
            self.b_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.b_proj",
            )

        self.q_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.q_conv1d",
        )
        self.k_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.k_conv1d",
        )
        self.v_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.v_conv1d",
        )
        # unsqueeze to fit conv1d weights shape into the linear weights shape.
        # Can't do this in `weight_loader` since it already exists in
        # `ColumnParallelLinear` and `set_weight_attrs`
        # doesn't allow to override it
        self.q_conv1d.weight.data = self.q_conv1d.weight.data.unsqueeze(1)
        self.k_conv1d.weight.data = self.k_conv1d.weight.data.unsqueeze(1)
        self.v_conv1d.weight.data = self.v_conv1d.weight.data.unsqueeze(1)

        self.A_log = nn.Parameter(torch.empty(1, 1, self.local_num_heads, 1, dtype=torch.float32))
        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(2)})

        self.g_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.g_a_proj",
        )
        self.g_b_proj = ColumnParallelLinear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.g_b_proj",
        )
        self.o_norm = FusedRMSNormGated(self.head_dim, eps=rms_norm_eps, activation="sigmoid")
        self.o_proj = RowParallelLinear(
            projection_size, self.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.o_proj", reduce_results=False
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        comm_manager: Optional[DecoderCommMananger] = None,
        block_scale: Optional[torch.Tensor] = None,
        can_run_flashinfer_fusion: bool = False,
    ) -> torch.Tensor:
        # Align with DeepseekV2AttentionMLA: accept unused MLA fusion kwargs and
        # do pre_attn_comm inside so callers can use a single attn call path.
        if comm_manager is not None:
            hidden_states = comm_manager.pre_attn_comm(hidden_states, forward_batch.tp_num_tokens)

        q_proj_states = self.q_proj(hidden_states)[0]
        k_proj_states = self.k_proj(hidden_states)[0]
        v_proj_states = self.v_proj(hidden_states)[0]

        q_conv_weights = self.q_conv1d.weight.view(self.q_conv1d.weight.size(0), self.q_conv1d.weight.size(2))
        k_conv_weights = self.k_conv1d.weight.view(self.k_conv1d.weight.size(0), self.k_conv1d.weight.size(2))
        v_conv_weights = self.v_conv1d.weight.view(self.v_conv1d.weight.size(0), self.v_conv1d.weight.size(2))

        kwargs = {
            "q_proj_states": q_proj_states,
            "k_proj_states": k_proj_states,
            "v_proj_states": v_proj_states,
            "q_conv_weights": q_conv_weights,
            "k_conv_weights": k_conv_weights,
            "v_conv_weights": v_conv_weights,
            "q_conv_bias": self.q_conv1d.bias,
            "k_conv_bias": self.k_conv1d.bias,
            "v_conv_bias": self.v_conv1d.bias,
            "dt_bias": self.dt_bias,
            "b_proj": self.b_proj,
            "f_a_proj": self.f_a_proj,
            "f_b_proj": self.f_b_proj,
            "A_log": self.A_log,
            "head_dim": self.head_dim,
            "hidden_states": hidden_states,
            "layer_id": self.layer_idx,
            "use_fgb_kda": self.use_fgb_kda,
        }
        if self.lower_bound is not None:
            kwargs["lower_bound"] = self.lower_bound
        core_attn_out = forward_batch.attn_backend.forward(
            q=None,
            k=None,
            v=None,
            layer=None,
            forward_batch=forward_batch,
            **kwargs,
        )

        g_proj_states = self.g_b_proj(self.g_a_proj(hidden_states)[0])[0]
        g = rearrange(g_proj_states, "... (h d) -> ... h d", d=self.head_dim)
        core_attn_out = self.o_norm(core_attn_out, g)
        core_attn_out = rearrange(core_attn_out, "1 n h d -> n (h d)")

        return self.o_proj(core_attn_out)[0]


class KimiDecoderLayer(nn.Module):
    def __init__(
        self,
        config: KimiLinearConfig,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.alt_stream = alt_stream
        self.is_kda = config.is_kda_layer(layer_idx)
        self.is_moe = config.is_moe

        if config.is_kda_layer(layer_idx):
            self.self_attn = KimiDeltaAttention(
                layer_idx=layer_idx,
                hidden_size=config.hidden_size,
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            self.self_attn = KimiMLAAttention(
                layer_id=layer_idx,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                config=config,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                skip_rope=True,
                reduce_attn_results=False,
            )
        self.is_layer_sparse = (
            self.is_moe and config.num_experts is not None and layer_idx >= config.first_k_dense_replace and layer_idx % config.moe_layer_freq == 0
        )

        if self.is_layer_sparse:
            self.block_sparse_moe = KimiMoE(
                config=config,
                quant_config=quant_config,
                layer_idx=layer_idx,
                prefix=f"{prefix}.mlp",
                alt_stream=self.alt_stream,
            )
            self.mlp = self.block_sparse_moe
        else:
            self.mlp = KimiMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.layer_id = layer_idx

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
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        tp_num_tokens: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_global_tokens, max_num_tokens_per_gpu = forward_batch.get_num_tokens(tp_num_tokens)

        # Self Attention
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

            # Fully Connected
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


class KimiLinearModel(nn.Module):
    def __init__(
        self,
        config: KimiLinearConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        self.config = config

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.alt_stream = torch.cuda.Stream()

        self.layers = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: KimiDecoderLayer(
                layer_idx=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=self.alt_stream,
            ),
            prefix=add_prefix("layers", prefix),
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        world_size = get_tensor_model_parallel_world_size()
        assert config.num_attention_heads % world_size == 0, "num_attention_heads must be divisible by world_size"

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        residual = None

        tp_num_tokens = hidden_states.shape[0]
        forward_batch.tp_num_tokens = tp_num_tokens

        # TODO: capture aux hidden states
        aux_hidden_states = []
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                residual=residual,
                tp_num_tokens=tp_num_tokens,
            )
        if not forward_batch.forward_mode.is_idle():
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)
            hidden_states, _ = layer.decoder_comm_manager.post_final_norm_comm(hidden_states, residual, tp_num_tokens)

        if len(aux_hidden_states) == 0:
            return hidden_states

        return hidden_states, aux_hidden_states


class KimiLinearForCausalLM(nn.Module):
    def __init__(
        self,
        config: KimiLinearConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = KimiLinearModel(config, quant_config, prefix=add_prefix("model", prefix))
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )
        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(config=config, logit_scale=logit_scale)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            inputs_embeds,
        )
        return self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch)

    @torch.no_grad()
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        if self.config.is_moe:
            # Params for weights, fp8 weight scales, fp8 activation scales
            # (param_name, weight_name, local_expert_id, shard_id)
            expert_params_mapping = make_expert_params_mapping(
                ckpt_gate_proj_name="w1",
                ckpt_down_proj_name="w2",
                ckpt_up_proj_name="w3",
                num_experts=self.config.num_experts,
            )
        else:
            expert_params_mapping = []
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for args in weights:
            name, loaded_weight = args[:2]
            kwargs = args[2] if len(args) > 2 else {}
            if "rotary_emb.inv_freq" in name:
                continue

            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue

            if (
                self.config.is_moe
                and ".experts." in name
                and any(
                    s in name
                    for s in (
                        ".w1.",
                        ".w2.",
                        ".w3.",
                        ".w1.weight",
                        ".w2.weight",
                        ".w3.weight",
                    )
                )
            ):
                loaded_expert = False
                for param_name, weight_name, local_expert_id, shard_id in expert_params_mapping:
                    if weight_name not in name:
                        continue

                    mapped_name = name.replace(weight_name, param_name)
                    candidate_names = [mapped_name]

                    target_name = next((n for n in candidate_names if n in params_dict), None)
                    if target_name is None:
                        # If this rank doesn't own this expert_id, we simply skip it.
                        continue

                    param = params_dict[target_name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        shard_id=shard_id,
                        local_expert_id=local_expert_id,
                    )
                    loaded_params.add(target_name)
                    loaded_expert = True
                    break

                # Whether loaded or not, never fall through to generic loader for experts.
                # If not loaded, it likely belongs to another EP rank.
                if loaded_expert:
                    continue
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
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
                # if is_pp_missing_parameter(name, self):
                #     continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict and not self.config.is_linear_attn:  # noqa: E501
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                # if is_pp_missing_parameter(name, self):
                #     continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, **kwargs)
            loaded_params.add(name)

        for layer_id in self.config.full_attention_layer_ids:
            self_attn = self.model.layers[layer_id].self_attn
            w_kc, w_vc = self_attn.kv_b_proj.weight.unflatten(0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)).split(
                [self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1
            )
            self_attn.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
            self_attn.w_vc = w_vc.contiguous().transpose(1, 2)
            if hasattr(self_attn.kv_b_proj, "weight_scale"):
                self_attn.w_scale = self_attn.kv_b_proj.weight_scale
        self.requires_grad_(False)

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_experts,
            num_groups=None,
        )


EntryClass = KimiLinearForCausalLM
