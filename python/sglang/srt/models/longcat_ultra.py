import re
from collections.abc import Iterable
from typing import Any, Optional

import torch
from torch import nn

from sglang.srt.configs import ShortcutConfig
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from sglang.srt.distributed.decoder_comm_manager import DecoderCommMananger
from sglang.srt.distributed.parallel_strategy import DenseParallelStategy
from sglang.srt.env import global_server_args_dict
from sglang.srt.layers.attention.dp_attention import (
    get_attention_tp_group,
    get_attention_tp_rank,
    get_attention_tp_size,
    get_dense_tp_rank,
    get_dense_tp_size,
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
from sglang.srt.utils import LazyValue, is_npu

if not is_npu():
    from sglang.srt.layers.moe.layer import MoELayer


class LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layout = global_server_args_dict["dense_parallel_strategy"]
        # For TP MOE, dense uses TP
        if not global_server_args_dict["enable_ep_moe"] or self.layout == DenseParallelStategy.TENSOR_PARALLEL:
            tp_rank = get_dense_tp_rank()
            tp_size = get_dense_tp_size()
            self.gate_up_proj = MergedColumnParallelLinear(
                hidden_size, [intermediate_size] * 2, bias=False, quant_config=quant_config, tp_size=tp_size, tp_rank=tp_rank
            )
            self.down_proj = RowParallelLinear(
                intermediate_size, hidden_size, bias=False, quant_config=quant_config, reduce_results=False, tp_rank=tp_rank, tp_size=tp_size
            )
        else:
            self.gate_up_proj = ReplicatedLinear(hidden_size, intermediate_size * 2, bias=False, quant_config=quant_config)
            self.down_proj = ReplicatedLinear(intermediate_size, hidden_size, bias=False, quant_config=quant_config)

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


class LongcatMoe(nn.Module):
    def __init__(
        self,
        config: ShortcutConfig,
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

        # Gate always runs at half / full precision for now.
        self.rounter_params_dtype = params_dtype
        if config.router_dtype == "float32":
            self.rounter_params_dtype = torch.float32
        self.gate = ReplicatedLinear(
            hidden_size,
            num_experts,
            bias=False,
            params_dtype=self.rounter_params_dtype,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        MoEImpl = MoELayer if global_server_args_dict["enable_ep_moe"] else FusedMoE  # noqa: F821
        self.experts = MoEImpl(
            num_experts=num_experts + global_server_args_dict["ep_num_redundant_experts"],
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            layer_id=self.layer_index,
            params_dtype=params_dtype,
            renormalize=False,
            quant_config=quant_config,
            routed_scaling_factor=getattr(config, "routed_scaling_factor", None),
        )

    def get_moe_routed_weights(self):
        return [x.data for name, x in self.experts.named_parameters() if name not in ["correction_bias"] and "shared_experts" not in name]

    def forward(self, hidden_states: torch.Tensor, num_global_tokens: int, max_num_tokens_per_gpu: int) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (num_tokens, n_experts)
        orig_param_dtype = hidden_states.dtype
        if self.rounter_params_dtype == torch.float32:
            hidden_states = hidden_states.to(torch.float32)
        router_logits, _ = self.gate(hidden_states)
        router_logits = router_logits.to(orig_param_dtype)
        hidden_states = hidden_states.to(orig_param_dtype)
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            num_global_tokens=num_global_tokens,
            max_num_tokens_per_gpu=max_num_tokens_per_gpu,
        )

        return final_hidden_states.view(num_tokens, hidden_dim)


class LlamaAttention(nn.Module):
    def __init__(
        self,
        config: ShortcutConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict[str, Any]] = None,
        rope_is_neox_style: bool = True,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        bias: bool = False,
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
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.total_num_heads)
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
            bias=bias,
            quant_config=quant_config,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=bias,
            quant_config=quant_config,
            reduce_results=False,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=rope_is_neox_style,
        )

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
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


class ShortcutMoEDecoderLayer(nn.Module):
    def __init__(
        self,
        config: ShortcutConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = config.original_max_position_embeddings
        rope_is_neox_style = getattr(config, "rope_is_neox_style", True)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.enable_dp_attention = global_server_args_dict["enable_dp_attention"]

        if self.enable_dp_attention:
            self.tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = get_tensor_model_parallel_world_size()
            self.tp_group = get_tp_group()
        attention_bias = getattr(config, "attention_bias", False) or getattr(config, "bias", False)
        if hasattr(config, "moe_intermediate_size"):
            self.moe_intermediate_size = config.moe_intermediate_size
        elif hasattr(config, "expert_ffn_hidden_size"):
            self.moe_intermediate_size = config.expert_ffn_hidden_size
        else:
            self.moe_intermediate_size = config.intermediate_size

        self.self_attn = nn.ModuleList(
            [
                LlamaAttention(
                    config=config,
                    hidden_size=self.hidden_size,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=config.num_key_value_heads,
                    layer_id=layer_id * 2 + i,
                    rope_theta=rope_theta,
                    rope_scaling=rope_scaling,
                    rope_is_neox_style=rope_is_neox_style,
                    max_position_embeddings=max_position_embeddings,
                    quant_config=None if "self_attn" in getattr(config, "disable_quant_module", []) else quant_config,
                    prefix=f"{prefix}.self_attn",
                    bias=attention_bias,
                )
                for i in range(2)
            ]
        )
        self.input_layernorm = nn.ModuleList([RMSNorm(config.hidden_size, eps=config.rms_norm_eps) for i in range(2)])
        self.post_attention_layernorm = nn.ModuleList([RMSNorm(config.hidden_size, eps=config.rms_norm_eps) for i in range(2)])
        self.mlps = nn.ModuleList(
            [
                LlamaMLP(
                    hidden_size=self.hidden_size,
                    intermediate_size=config.intermediate_size,
                    hidden_act="silu",
                    quant_config=None if "mlps" in getattr(config, "disable_quant_module", []) else quant_config,
                    prefix=f"{prefix}.mlps",
                )
                for i in range(2)
            ]
        )
        self.mlp = LongcatMoe(
            config=config,
            num_experts=config.num_experts[layer_id],
            top_k=config.moe_topk,
            hidden_size=config.hidden_size,
            layer_index=layer_id,
            intermediate_size=self.moe_intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
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
            is_moe_layer=False,
            num_layers=config.num_hidden_layers,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        tp_num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_global_tokens, max_num_tokens_per_gpu = forward_batch.get_num_tokens(tp_num_tokens)
        if not forward_batch.forward_mode.is_idle():
            # first_input_layernorm
            residual = hidden_states
            hidden_states = self.input_layernorm[0](hidden_states)

            # first_attn
            hidden_states = self.decoder_comm_manager.pre_attn_comm(hidden_states, tp_num_tokens, is_second_attn=False)
            hidden_states = self.self_attn[0](
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
            self.decoder_comm_manager.is_moe_layer = True
            hidden_states, residual = self.decoder_comm_manager.post_attn_comm(hidden_states, residual, tp_num_tokens, is_first_attn=True)
            hidden_states = residual + hidden_states
            residual = hidden_states

            # first_post_attention_layernorm
            hidden_states = self.post_attention_layernorm[0](hidden_states)

            # moe
            moe_hidden_states = self.forward_mlp(
                self.mlp,
                hidden_states,
                residual,
                forward_batch,
                num_global_tokens,
                max_num_tokens_per_gpu,
                tp_num_tokens,
            )

            # first_mlp
            self.decoder_comm_manager.is_moe_layer = False
            hidden_states = self.forward_mlp(
                self.mlps[0],
                hidden_states,
                residual,
                forward_batch,
                num_global_tokens,
                max_num_tokens_per_gpu,
                tp_num_tokens,
            )

            hidden_states = residual + hidden_states

            # second_input_layernorm
            residual = hidden_states
            hidden_states = self.input_layernorm[1](hidden_states)

            # second_attn
            hidden_states = self.decoder_comm_manager.pre_attn_comm(hidden_states, tp_num_tokens, is_second_attn=True)
            hidden_states = self.self_attn[1](
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
            hidden_states, residual = self.decoder_comm_manager.post_attn_comm(hidden_states, residual, tp_num_tokens, is_first_attn=False)
            hidden_states = residual + hidden_states

            residual = hidden_states
            # second_post_attention_layernorm
            hidden_states = self.post_attention_layernorm[1](hidden_states)

            # second_mlp
            hidden_states = self.forward_mlp(
                self.mlps[1],
                hidden_states,
                residual,
                forward_batch,
                num_global_tokens,
                max_num_tokens_per_gpu,
                tp_num_tokens,
            )
            hidden_states = hidden_states + moe_hidden_states
            hidden_states = residual + hidden_states
        else:
            # moe
            self.decoder_comm_manager.is_moe_layer = True
            hidden_states = self.forward_mlp(
                self.mlp,
                hidden_states,
                residual,
                forward_batch,
                num_global_tokens,
                max_num_tokens_per_gpu,
                tp_num_tokens,
            )
            self.decoder_comm_manager.is_moe_layer = False

        return hidden_states, residual

    def forward_mlp(
        self,
        mlp,
        hidden_states,
        residual,
        forward_batch,
        num_global_tokens,
        max_num_tokens_per_gpu,
        tp_num_tokens,
    ):
        hidden_states, start_idx, end_idx = self.decoder_comm_manager.pre_mlp_comm(hidden_states, forward_batch, tp_num_tokens)
        hidden_states = mlp(hidden_states, num_global_tokens, max_num_tokens_per_gpu) if isinstance(mlp, LongcatMoe) else mlp(hidden_states)
        hidden_states, residual = self.decoder_comm_manager.post_mlp_comm(hidden_states, residual, tp_num_tokens, forward_batch)
        if start_idx is not None and end_idx is not None:
            hidden_states = hidden_states[start_idx:end_idx]
        return hidden_states


class ShortcutMoEModel(nn.Module):
    def __init__(
        self,
        config: ShortcutConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            enable_tp=not global_server_args_dict["enable_dp_attention"],
        )
        self.layers = nn.ModuleList(
            [
                ShortcutMoEDecoderLayer(
                    config,
                    i,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers",
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids) if input_embeds is None else input_embeds.type_as(self.embed_tokens.weight)
        tp_num_tokens = hidden_states.shape[0]
        residual = None
        for i in range(len(self.layers)):
            with get_global_expert_distribution_recorder().with_current_layer(i):
                layer = self.layers[i]
                hidden_states, residual = layer(positions, hidden_states, forward_batch, residual, tp_num_tokens=tp_num_tokens)

        if not forward_batch.forward_mode.is_idle():
            hidden_states = self.norm(hidden_states)
            hidden_states, _ = layer.decoder_comm_manager.post_final_norm_comm(hidden_states, residual, tp_num_tokens)
        return hidden_states


class ShortcutMoEForCausalLM(nn.Module):
    def __init__(
        self,
        config: ShortcutConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = ShortcutMoEModel(config, quant_config=quant_config, prefix="model")
        if global_server_args_dict["enable_dp_attention"]:
            self.lm_head = ReplicatedLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
            )
            self.logits_processor = LogitsProcessor(config, skip_all_gather=True)
        else:
            self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, quant_config=quant_config)
            self.logits_processor = LogitsProcessor(config)
        self._routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: layer.mlp.get_moe_routed_weights() for layer_id, layer in enumerate(self.model.layers) if isinstance(layer.mlp, LongcatMoe)
            }
        )

    @property
    def routed_experts_weights_of_layer(self):
        return self._routed_experts_weights_of_layer.value

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch)
        return self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch)

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        assert len(set(config.num_experts)) == 1

        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_experts[0],
            num_groups=None,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        def rename_attn_weight(name):
            # model.layers.0.self_attn.qkv_proj.0.weight > model.layers.0.self_attn.0.qkv_proj.weight
            configurable_parts = ["qkv_proj", "o_proj"]
            for configurable_part in configurable_parts:
                pattern = rf"(.*?)\.({re.escape(configurable_part)})\.(\d+)\.(.*)"
                match = re.match(pattern, name)
                if match:
                    str_prefix = match.group(1)
                    str_weight = match.group(2)
                    str_num_weight = match.group(3)
                    str_suffix = match.group(4)
                    name = f"{str_prefix}.{str_num_weight}.{str_weight}.{str_suffix}"
            return name

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        MoEImpl = MoELayer if global_server_args_dict["enable_ep_moe"] else FusedMoE  # noqa: F821
        expert_params_mapping = MoEImpl.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts[0],
        )

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp" in name and "mlps" not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if (name.endswith(".bias") or name.endswith("_bias")) and name not in params_dict:
                    continue
                name = rename_attn_weight(name)
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

                    if (name.endswith(".bias") or name.endswith("_bias")) and name not in params_dict:
                        continue
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
                    if (name.endswith(".bias") or name.endswith("_bias")) and name not in params_dict:
                        continue
                    # Skip loading kv_scale from ckpts towards new design.
                    if name.endswith(".kv_scale") and name not in params_dict:
                        continue
                    if name is None:
                        continue
                    if name.endswith("router.classifier.weight"):
                        name = name.replace("router.classifier", "gate")
                    name = rename_attn_weight(name)
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)


EntryClass = ShortcutMoEForCausalLM
