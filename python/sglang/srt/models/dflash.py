import logging
from collections.abc import Iterable
from typing import Optional

import torch
from torch import nn

from sglang.srt.layers.attention.dp_attention import (
    get_attention_tp_group,
    get_attention_tp_rank,
    get_attention_tp_size,
)
from sglang.srt.layers.attention.radix_attention import AttentionType, RadixAttention
from sglang.srt.layers.embedding.rotary_embedding import get_rope
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
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.sampler.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.env import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.llama import LlamaDecoderLayer
from sglang.srt.ops.layernorm import RMSNorm
from sglang.srt.utils import add_prefix, device_synchronize

logger = logging.getLogger(__name__)


class DFlashAttention(nn.Module):
    def __init__(self, config, layer_id: int, quant_config, prefix: str) -> None:
        super().__init__()
        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()
        self.attn_tp_group = get_attention_tp_group()

        hidden_size = int(config.hidden_size)
        self.hidden_size = hidden_size
        self.total_num_heads = int(config.num_attention_heads)
        self.total_num_kv_heads = int(getattr(config, "num_key_value_heads", self.total_num_heads))
        assert self.total_num_heads % self.attn_tp_size == 0
        assert self.total_num_kv_heads % self.attn_tp_size == 0 and self.total_num_kv_heads >= self.attn_tp_size

        self.num_heads = self.total_num_heads // self.attn_tp_size
        self.num_kv_heads = self.total_num_kv_heads // self.attn_tp_size
        self.head_dim = int(getattr(config, "head_dim", hidden_size // self.total_num_heads))
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))
        attention_bias = bool(getattr(config, "attention_bias", False))

        self.with_qk_norm = False
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        self.q_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_heads * self.head_dim,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
            outside_tp_group=self.attn_tp_group,
        )
        self.kv_proj = MergedColumnParallelLinear(
            hidden_size,
            [
                self.total_num_kv_heads * self.head_dim,
                self.total_num_kv_heads * self.head_dim,
            ],
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_proj",
            outside_tp_group=self.attn_tp_group,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
            reduce_results=False,
            outside_tp_group=self.attn_tp_group,
        )

        rope_params = getattr(config, "rope_parameters", None)
        if rope_params is not None:
            if not isinstance(rope_params, dict):
                rope_params = dict(getattr(rope_params, "__dict__", {})) or {}
            rope_theta = rope_params.get("rope_theta", getattr(config, "rope_theta", 10000))
            rope_scaling = {k: v for k, v in rope_params.items() if k != "rope_theta"} or None
        else:
            rope_theta = getattr(config, "rope_theta", 10000)
            rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = config.original_max_position_embeddings
        rope_is_neox_style = bool(getattr(config, "rope_is_neox_style", True))
        max_position_embeddings = int(getattr(config, "max_position_embeddings", 8192))
        if rope_scaling:
            if rope_scaling["rope_type"] == "deepseek_yarn":
                rope_scaling["rope_type"] = "yarn"
                max_position_embeddings = rope_scaling["factor"] * rope_scaling["original_max_position_embeddings"]
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=rope_is_neox_style,
        )
        # SWA sink: block 的 Q 只 attend [前 context_sink_size] + [末 sliding_window+block]。
        use_sliding_window = bool(getattr(config, "use_sliding_window", False))
        sliding_window_size = (
            int(getattr(config, "sliding_window", None) or -1) if use_sliding_window else -1
        )
        dflash_config = getattr(config, "dflash_config", None) or {}
        context_sink_size = int(
            dflash_config.get("context_sink_size", getattr(config, "sliding_window_sink", 16))
        )
        self.use_swa_sink_mode = (
            use_sliding_window and sliding_window_size > 0 and context_sink_size > 0
        )
        self.sliding_window_size = sliding_window_size
        self.context_sink_size = context_sink_size

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            attn_type=AttentionType.ENCODER_ONLY,  # 块内双向,非因果
            sliding_window_size=sliding_window_size,
        )
        self._rope_dummy_q: Optional[torch.Tensor] = None

    def _apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q_by_head = self.q_norm(q.reshape(-1, self.head_dim)).view(q.shape)
        k_by_head = self.k_norm(k.reshape(-1, self.head_dim)).view(k.shape)
        return q_by_head, k_by_head

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states
        q, _ = self.q_proj(hidden_states)
        kv, _ = self.kv_proj(hidden_states)
        k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
        if self.with_qk_norm:
            q, k = self._apply_qk_norm(q, k)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output

    def kv_proj_only(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        kv, _ = self.kv_proj(hidden_states)
        k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
        return k, v

    def apply_k_norm(self, k: torch.Tensor) -> torch.Tensor:
        if not self.with_qk_norm:
            return k
        k_by_head = k.reshape(-1, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        return k_by_head.view_as(k)

    def apply_k_rope(self, positions: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        # dummy_q 必须是零值: GPU 融合 rotary kernel 会把未初始化值带进 k。
        # k 形状跨步恒定 ([bs*block, num_kv_heads, head_dim]), 复用 buffer 免每步分配。
        dummy_q = self._rope_dummy_q
        if dummy_q is None or dummy_q.shape != k.shape:
            self._rope_dummy_q = dummy_q = torch.zeros_like(k)
        else:
            dummy_q.zero_()
        _, k = self.rotary_emb(positions, dummy_q, k)
        return k


class DFlashDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config, layer_id: int, quant_config, prefix: str) -> None:
        super().__init__(config, layer_id, quant_config, prefix)
        self.self_attn = DFlashAttention(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        tp_num_tokens: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
            hidden_states, residual = self.decoder_comm_manager.post_attn_comm(hidden_states, residual, tp_num_tokens)
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        hidden_states, start_index, end_index = self.decoder_comm_manager.pre_mlp_comm(hidden_states, forward_batch, tp_num_tokens)
        hidden_states = self.mlp(hidden_states)
        hidden_states, residual = self.decoder_comm_manager.post_mlp_comm(hidden_states, residual, tp_num_tokens, forward_batch)
        if start_index is not None and end_index is not None:
            hidden_states = hidden_states[start_index:end_index]
        return hidden_states, residual


class DFlashDraftModel(nn.Module):
    def __init__(self, config, quant_config: Optional[QuantizationConfig] = None, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        # draft config 写 disable_quant_module: ["dflash"] 可整体回退 bf16
        self.quant_config = None if "dflash" in getattr(config, "disable_quant_module", []) else quant_config
        quant_config = self.quant_config

        hidden_size = int(config.hidden_size)
        num_layers = int(config.num_hidden_layers)
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
            enable_tp=not global_server_args_dict["enable_dp_attention"],
        )

        self.layers = nn.ModuleList(
            [
                DFlashDecoderLayer(
                    config=config,
                    layer_id=i,
                    quant_config=quant_config,
                    prefix=f"layers.{i}",
                )
                for i in range(num_layers)
            ]
        )
        # draft 不切 PP,attn 范围=全部 draft 层。
        self.attn_start_layer = 0
        self.attn_end_layer = num_layers
        for layer in self.layers:
            layer.decoder_comm_manager.enable_qcp = False
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        self.dflash_config = getattr(config, "dflash_config", None)
        assert self.dflash_config is not None

        self.target_layer_ids = self.dflash_config.get("target_layer_ids", getattr(config, "target_layer_ids", None))
        if self.target_layer_ids is None:
            raise RuntimeError("DFLASH target_layer_ids is None")

        self.num_context_features = len(self.target_layer_ids)
        # fc 不切 TP。用 ReplicatedLinear 而非裸 nn.Linear, 否则量化 ckpt 的 fc.weight_scale 无处安放。
        self.fc = ReplicatedLinear(
            self.num_context_features * hidden_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("fc", prefix),
        )
        self.hidden_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        if self._use_replicated_lm_head():
            self.lm_head = ReplicatedLinear(config.hidden_size, config.vocab_size, bias=False)
            self.logits_processor = LogitsProcessor(config, skip_all_gather=True)
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=None,  # 权重与 target 共享, 不量化
                prefix=add_prefix("lm_head", prefix),
            )
            self.logits_processor = LogitsProcessor(config)

        self.block_size = int(global_server_args_dict["speculative_num_draft_tokens"])
        mask_token_id = self.dflash_config.get("mask_token_id", getattr(config, "mask_token_id", None))
        self.mask_token_id = int(mask_token_id) if mask_token_id is not None else None

    def _use_replicated_lm_head(self) -> bool:
        # lm_head 是否走 ReplicatedLinear(DP) 而非 ParallelLMHead。NPU 子类覆写此判据。
        return bool(global_server_args_dict["enable_dp_attention"])

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        fc_out, _ = self.fc(target_hidden)
        return self.hidden_norm(fc_out)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> LogitsProcessorOutput:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        residual = None
        tp_num_tokens = hidden_states.shape[0]
        forward_batch.tp_num_tokens = tp_num_tokens
        for layer in self.layers:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
                tp_num_tokens,
            )

        if not forward_batch.forward_mode.is_idle():
            hidden_states, _ = self.norm(hidden_states, residual)
            if layer is not None:
                hidden_states, _ = layer.decoder_comm_manager.post_final_norm_comm(hidden_states, residual, tp_num_tokens)

        return self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch, None)

    def get_embed_and_head(self):
        return self.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.embed_tokens.weight
        del self.lm_head.weight
        self.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        device_synchronize()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            ("kv_proj", "k_proj", 0),
            ("kv_proj", "v_proj", 1),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        has_qk_norm = False
        # resolve_param_name 返回 None 会被静默 continue, 须记录以便诊断权重名不匹配。
        skipped_names: list[str] = []

        def resolve_param_name(name: str) -> Optional[str]:
            if name in params_dict:
                return name
            if name.startswith("model."):
                stripped = name[len("model.") :]
                if stripped in params_dict:
                    return stripped
            elif f"model.{name}" in params_dict:
                return f"model.{name}"
            return None

        for name, loaded_weight in weights:
            if "q_norm" in name or "k_norm" in name:
                has_qk_norm = True
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if f".{weight_name}." not in name:
                    continue
                resolved = resolve_param_name(name.replace(weight_name, param_name))
                if resolved is None:
                    skipped_names.append(name)
                    continue
                param = params_dict[resolved]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                resolved = resolve_param_name(name)
                if resolved is None:
                    skipped_names.append(name)
                    continue
                param = params_dict[resolved]
                # fc.weight 的维度由抓层数 K 决定,不匹配时给出明确报错。
                if resolved.endswith("fc.weight") and tuple(loaded_weight.shape) != tuple(param.shape):
                    raise ValueError(
                        f"DFLASH fc.weight 维度不匹配(抓层数 K 不同)。期望 {tuple(param.shape)},实际 {tuple(loaded_weight.shape)},来自 '{name}'。"
                    )
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

        # 权重名不匹配会被静默 continue, 留一条告警避免无声加载失败。
        if skipped_names:
            logger.warning(
                f"DFLASH load_weights 跳过 {len(skipped_names)} 个未匹配权重: {skipped_names[:8]}"
            )

        if has_qk_norm:
            for layer in self.layers:
                layer.self_attn.with_qk_norm = True


EntryClass = DFlashDraftModel
