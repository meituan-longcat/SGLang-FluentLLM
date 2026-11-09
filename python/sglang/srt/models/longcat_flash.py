import re
from collections.abc import Iterable
from typing import Optional, Union

import torch
from flashinfer import dsv3_router_gemm
from torch import nn

from sglang.srt.configs import FLASHConfig
from sglang.srt.configs.model_config import dsa_layer_has_indexer, is_dsa
from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from sglang.srt.distributed.decoder_comm_manager import DecoderCommMananger
from sglang.srt.distributed.model_tensor_tracer import get_load_number_layers
from sglang.srt.distributed.parallel_strategy import AttnParallelStrategy, DenseParallelStategy
from sglang.srt.env import global_server_args_dict
from sglang.srt.layers.attention.dp_attention import (
    get_attention_tp_group,
    get_attention_tp_rank,
    get_attention_tp_size,
    get_dense_tp_group,
    get_dense_tp_size,
)
from sglang.srt.layers.dense.gemms.fp8.fp8_utils import block_dequant
from sglang.srt.layers.dense.mlp import ParallelMLP
from sglang.srt.layers.embedding.over_embedding import FusedOverEmbedding
from sglang.srt.layers.embedding.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.moe.layer import MoELayer
from sglang.srt.layers.moe.layouts.mapping import make_expert_params_mapping
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.utils import should_ignore_quant_layer
from sglang.srt.layers.sampler.logits_processor import LogitsProcessor
from sglang.srt.layers.utils import (
    CP_METADATA,
    cp_all_gather_rerange_output,
    cp_split_and_rebuild_data,
    get_layer_id,
    maybe_get_longcat_attn_layer_id,
)
from sglang.srt.managers.eplb.expert_distribution import (
    get_global_expert_distribution_recorder,
)
from sglang.srt.managers.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek_mha_nsa import DeepseekNSAWithMLA
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
from sglang.srt.models.deepseek_v32 import DeepseekV32MLA
from sglang.srt.models.kimi_linear import KimiDeltaAttention
from sglang.srt.ops.layernorm import RMSNorm
from sglang.srt.ops.topk import TopK, TopKOutputFormat
from sglang.srt.utils import (
    LazyValue,
    add_prefix,
    bind_or_assign,
    get_colorful_logger,
    make_layers,
)

logger = get_colorful_logger(__name__)


class LongcatRouter(nn.Module):
    def __init__(self, config, prefix: str = ""):
        super().__init__()
        assert not config.router_bias

        n_routed_experts = config.n_routed_experts + config.zero_expert_num
        params_dtype = torch.bfloat16 if config.router_dtype == "bfloat16" else torch.float
        self.classifier = ReplicatedLinear(
            config.hidden_size,
            n_routed_experts,
            bias=getattr(config, "router_bias", None),
            params_dtype=params_dtype,
            quant_config=None,
            prefix=add_prefix("classifier", prefix),
        )
        self.e_score_correction_bias = nn.Parameter(torch.zeros(n_routed_experts, dtype=torch.float32))

    def forward(self, hidden_states, rsag=None):
        if not global_server_args_dict["enable_ep_moe"]:
            token_list_in_group = rsag.get_token_dist(hidden_states.shape[0])
            rank = get_attention_tp_rank()
            local_num_tokens = token_list_in_group[rank]
            local_token_offset = sum(token_list_in_group[:rank])
            start = local_token_offset
            end = start + local_num_tokens
            hidden_states = hidden_states[start:end].contiguous()
        # TODO: dsv3_router_gemm use for low latency mode
        if False:
            logits = dsv3_router_gemm(hidden_states, self.classifier.weight, out_dtype=torch.float32)
        else:
            logits = self.classifier(hidden_states.float())[0].to(torch.float32)
        if not global_server_args_dict["enable_ep_moe"]:
            logits = rsag.all_gather(logits.to(torch.bfloat16), token_list_in_group=token_list_in_group)
        return logits.to(torch.float32)


class LongcatMoe(nn.Module):
    def __init__(
        self,
        config: FLASHConfig,
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
        self.zero_expert_type = config.zero_expert_type
        self.routed_scaling_factor = config.routed_scaling_factor
        # Gate always runs at full precision for now.
        self.router = LongcatRouter(config=config, prefix=add_prefix("gate", prefix))
        self.num_experts = num_experts + global_server_args_dict["ep_num_redundant_experts"]
        self.quant_config = quant_config

        self.experts = MoELayer(
            top_k=top_k,
            num_experts=self.num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            quant_config=quant_config,
            layer_index=self.layer_index,
            prefix=prefix,
            zero_expert_type=self.zero_expert_type,
        )

        self.topk = TopK(
            top_k=top_k,
            layer_id=self.layer_index,
            renormalize=False,
            num_fused_shared_experts=0,
            correction_bias=self.router.e_score_correction_bias,
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=self.experts.apply_routed_scaling_factor_on_output,
            output_format=TopKOutputFormat.STANDARD,
            zero_expert_num=config.zero_expert_num,
            topk_indices_dtype=torch.int64 if global_server_args_dict["enable_deep_ep"] else torch.int32,
        )

    def get_moe_routed_weights(self):
        return [x.data for name, x in self.experts.named_parameters() if name not in ["correction_bias"] and "shared_experts" not in name]

    def forward(self, hidden_states: torch.Tensor, num_global_tokens: int, max_num_tokens_per_gpu: int, decoder_comm_manager=None) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (num_tokens, n_experts)
        router_logits = self.router(hidden_states, decoder_comm_manager.attn_rsag)
        topk_output = self.topk(hidden_states, router_logits) if hidden_states.shape[0] > 0 else self.topk.empty_topk_output(hidden_states.device)

        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            topk_output=topk_output,
            num_global_tokens=num_global_tokens,
            max_num_tokens_per_gpu=max_num_tokens_per_gpu,
        )

        if not self.experts.apply_routed_scaling_factor_on_output:
            final_hidden_states *= self.routed_scaling_factor

        return final_hidden_states.view(num_tokens, hidden_dim)


class FLASHDecoderLayer(nn.Module):
    def __init__(
        self,
        config: FLASHConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        kv_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.quant_config = quant_config
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = config.original_max_position_embeddings
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.enable_dp_attention = global_server_args_dict["enable_dp_attention"]

        if self.enable_dp_attention:
            self.tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = get_tensor_model_parallel_world_size()
            self.tp_group = get_tp_group()

        self.intermediate_size = config.ffn_hidden_size if hasattr(config, "ffn_hidden_size") else config.intermediate_size
        if hasattr(config, "moe_intermediate_size"):
            self.moe_intermediate_size = config.moe_intermediate_size
        elif hasattr(config, "expert_ffn_hidden_size"):
            self.moe_intermediate_size = config.expert_ffn_hidden_size
        else:
            self.moe_intermediate_size = self.intermediate_size

        use_nsa_mla = getattr(config, "use_nsa_mla", False)
        self.kv_stream = kv_stream
        if is_dsa(config):
            AttnImpl = DeepseekV32MLA
        elif use_nsa_mla:
            AttnImpl = DeepseekNSAWithMLA
        else:
            AttnImpl = DeepseekV2AttentionMLA
        self.use_hybrid_linear_kda = bool(getattr(config, "use_hybrid_linear_kda", False))
        self.full_attention_layer_ids = set(getattr(config, "full_attention_layer_ids", []))
        self.self_attn = nn.ModuleList([])
        for i in range(2):
            global_attn_layer_id = layer_id * 2 + i
            use_full_attn = (not self.use_hybrid_linear_kda) or (global_attn_layer_id in self.full_attention_layer_ids)
            if use_full_attn:
                attn = AttnImpl(
                    config=config,
                    hidden_size=self.hidden_size,
                    num_heads=config.num_attention_heads,
                    qk_nope_head_dim=config.qk_nope_head_dim,
                    qk_rope_head_dim=config.qk_rope_head_dim,
                    v_head_dim=config.v_head_dim,
                    q_lora_rank=(config.q_lora_rank if hasattr(config, "q_lora_rank") else None),
                    kv_lora_rank=config.kv_lora_rank,
                    rope_theta=rope_theta,
                    rope_scaling=rope_scaling,
                    max_position_embeddings=max_position_embeddings,
                    quant_config=None if "self_attn" in getattr(config, "disable_quant_module", []) else quant_config,
                    layer_id=global_attn_layer_id,
                    reduce_attn_results=False,
                    skip_rope=bool(getattr(config, "kda_nope", False)),
                    prefix=add_prefix(f"self_attn.{i}", prefix),
                    alt_stream=kv_stream,
                )
            else:
                attn = KimiDeltaAttention(
                    layer_idx=global_attn_layer_id,
                    hidden_size=self.hidden_size,
                    config=config,
                    quant_config=None if "self_attn" in getattr(config, "disable_quant_module", []) else quant_config,
                    rms_norm_eps=config.rms_norm_eps,
                    prefix=add_prefix(f"self_attn.{i}", prefix),
                )
            self.self_attn.append(attn)
        self.input_layernorm = nn.ModuleList([RMSNorm(config.hidden_size, eps=config.rms_norm_eps) for i in range(2)])
        self.post_attention_layernorm = nn.ModuleList([RMSNorm(config.hidden_size, eps=config.rms_norm_eps) for i in range(2)])
        self.mlps = nn.ModuleList(
            [
                ParallelMLP(
                    hidden_size=self.hidden_size,
                    intermediate_size=self.intermediate_size,
                    hidden_act="silu",
                    quant_config=None if "mlps" in getattr(config, "disable_quant_module", []) else quant_config,
                    prefix=add_prefix(f"mlps.{i}", prefix),
                )
                for i in range(2)
            ]
        )
        self.topk = config.moe_topk if hasattr(config, "moe_topk") else config.num_experts_per_tok
        self.mlp = LongcatMoe(
            config=config,
            num_experts=config.n_routed_experts if hasattr(config, "n_routed_experts") else config.num_experts[layer_id],
            top_k=config.moe_topk if hasattr(config, "moe_topk") else config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=self.moe_intermediate_size,
            quant_config=quant_config,
            layer_index=layer_id,
            prefix=add_prefix("mlp", prefix),
        )

        self.layer_id = layer_id
        self.alt_stream = alt_stream
        self.attn_tp_group = get_attention_tp_group()
        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()
        self.moe_branch_decoder_comm_manager = DecoderCommMananger(
            layer_id=self.layer_id,
            attn_parallel_strategy=global_server_args_dict["attn_parallel_strategy"],
            dense_parallel_strategy=global_server_args_dict["dense_parallel_strategy"],
            moe_parallel_strategy=global_server_args_dict["moe_parallel_strategy"],
            is_moe_layer=True,
            num_layers=config.num_hidden_layers,
        )
        self.mlp_branch_decoder_comm_manager = [
            DecoderCommMananger(
                layer_id=self.layer_id * 2 + i,
                attn_parallel_strategy=global_server_args_dict["attn_parallel_strategy"],
                dense_parallel_strategy=global_server_args_dict["dense_parallel_strategy"],
                moe_parallel_strategy=global_server_args_dict["moe_parallel_strategy"],
                is_moe_layer=False,
                num_layers=config.num_hidden_layers,
            )
            for i in range(2)
        ]
        self.attn_0_unquanted = (
            self.check_unquanted(self.self_attn[0].fused_qkv_a_proj_with_mqa) if hasattr(self.self_attn[0], "fused_qkv_a_proj_with_mqa") else True
        )
        self.attn_1_unquanted = (
            self.check_unquanted(self.self_attn[1].fused_qkv_a_proj_with_mqa) if hasattr(self.self_attn[1], "fused_qkv_a_proj_with_mqa") else True
        )
        self.dense_0_unqaunted = self.check_unquanted(self.mlps[0].gate_up_proj)
        self.dense_1_unqaunted = self.check_unquanted(self.mlps[1].gate_up_proj)

    def check_unquanted(self, module) -> bool:
        return module.quant_config is None or should_ignore_quant_layer(
            module.prefix, ignored_layers=getattr(module.quant_config, "ignored_layers", [])
        )

    @staticmethod
    def _can_fuse_attn_input_fp8(attn) -> bool:
        # if attn is using dsa, then we can't fuse the quanted input
        return not getattr(attn, "use_dsa", False)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        tp_num_tokens: int,
        next_layer_input_norm: Optional[RMSNorm] = None,
        input_is_sharded: bool = False,
        input_block_scale: Optional[torch.Tensor] = None,
        can_run_flashinfer_fusion: bool = False,
        can_run_next_flashinfer_fusion: bool = False,
        current_layer_attn_1_quanted: bool = False,
        next_layer_attn_0_quanted: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        num_global_tokens, max_num_tokens_per_gpu = forward_batch.get_num_tokens(tp_num_tokens)

        if not forward_batch.forward_mode.is_idle():
            # first_input_layernorm
            if not input_is_sharded:
                if residual is None:
                    residual = hidden_states
                    hidden_states = self.input_layernorm[0](hidden_states)
                else:
                    hidden_states, residual = self.input_layernorm[0](hidden_states, residual)

            # first_attn
            hidden_states = self.self_attn[0](
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                comm_manager=self.mlp_branch_decoder_comm_manager[0],
                block_scale=input_block_scale if input_is_sharded else None,
                can_run_flashinfer_fusion=can_run_flashinfer_fusion,
            )

            # rs
            hidden_states, residual = self.moe_branch_decoder_comm_manager.post_attn_comm(hidden_states, residual, tp_num_tokens)

            # first_post_attention_layernorm
            overlap_input_hidden_states, overlap_input_residual = self.post_attention_layernorm[0](hidden_states, residual)

            if self.alt_stream is not None:
                current_stream = torch.cuda.current_stream()
                self.alt_stream.wait_stream(current_stream)
                # moe
                moe_hidden_states = self.forward_mlp(
                    self.moe_branch_decoder_comm_manager,
                    self.mlp,
                    overlap_input_hidden_states,
                    overlap_input_residual,
                    forward_batch,
                    num_global_tokens,
                    max_num_tokens_per_gpu,
                    tp_num_tokens,
                )
                with torch.cuda.stream(self.alt_stream):
                    hidden_states, residual = self.forward_mlp_branch(
                        self.mlp_branch_decoder_comm_manager,
                        positions,
                        overlap_input_hidden_states,
                        overlap_input_residual,
                        forward_batch,
                        num_global_tokens,
                        max_num_tokens_per_gpu,
                        tp_num_tokens,
                        can_run_flashinfer_fusion=can_run_flashinfer_fusion,
                        can_run_next_flashinfer_fusion=can_run_next_flashinfer_fusion,
                        current_layer_attn_1_quanted=current_layer_attn_1_quanted,
                    )
                current_stream.wait_stream(self.alt_stream)
            else:
                # moe
                moe_hidden_states = self.forward_mlp(
                    self.moe_branch_decoder_comm_manager,
                    self.mlp,
                    overlap_input_hidden_states,
                    overlap_input_residual,
                    forward_batch,
                    num_global_tokens,
                    max_num_tokens_per_gpu,
                    tp_num_tokens,
                )
                hidden_states, residual = self.forward_mlp_branch(
                    self.mlp_branch_decoder_comm_manager,
                    positions,
                    overlap_input_hidden_states,
                    overlap_input_residual,
                    forward_batch,
                    num_global_tokens,
                    max_num_tokens_per_gpu,
                    tp_num_tokens,
                    can_run_flashinfer_fusion=can_run_flashinfer_fusion,
                    can_run_next_flashinfer_fusion=can_run_next_flashinfer_fusion,
                    current_layer_attn_1_quanted=current_layer_attn_1_quanted,
                )

            output_block_scale = None
            if can_run_next_flashinfer_fusion:
                hidden_states, residual, output_block_scale = next_layer_input_norm.forward_with_reducescatter_fusion(
                    get_dense_tp_group(),
                    hidden_states,
                    residual,
                    fuse_block_quant_fp8=next_layer_attn_0_quanted,
                    add_in=moe_hidden_states,
                )
            else:
                hidden_states = hidden_states + moe_hidden_states
        else:
            # moe
            moe_hidden_states = self.forward_mlp(
                self.moe_branch_decoder_comm_manager,
                self.mlp,
                hidden_states,
                residual,
                forward_batch,
                num_global_tokens,
                max_num_tokens_per_gpu,
                tp_num_tokens,
            )
            if (
                global_server_args_dict["attn_parallel_strategy"] == AttnParallelStrategy.DATA_PARALLEL
                and global_server_args_dict["dense_parallel_strategy"] == DenseParallelStategy.TENSOR_PARALLEL
            ):
                hidden_states, residual = self.forward_mlp_branch(
                    self.mlp_branch_decoder_comm_manager,
                    positions,
                    hidden_states,
                    residual,
                    forward_batch,
                    num_global_tokens,
                    max_num_tokens_per_gpu,
                    tp_num_tokens,
                )
                hidden_states = hidden_states + moe_hidden_states
            else:
                hidden_states = moe_hidden_states
            output_block_scale = None

        return hidden_states, residual, output_block_scale

    def forward_mlp_branch(
        self,
        decoder_comm_manager,
        positions,
        hidden_states,
        residual,
        forward_batch,
        num_global_tokens,
        max_num_tokens_per_gpu,
        tp_num_tokens,
        can_run_flashinfer_fusion: bool = False,
        can_run_next_flashinfer_fusion: bool = False,
        current_layer_attn_1_quanted: bool = False,
    ):
        # first_mlp
        hidden_states = self.forward_mlp(
            decoder_comm_manager[0],
            self.mlps[0],
            hidden_states,
            residual,
            forward_batch,
            num_global_tokens,
            max_num_tokens_per_gpu,
            tp_num_tokens,
            skip_post_comm=can_run_flashinfer_fusion,
        )
        block_scale = None
        if not forward_batch.forward_mode.is_idle():
            if isinstance(self.self_attn[1], KimiDeltaAttention):
                can_run_flashinfer_fusion = False
                can_run_next_flashinfer_fusion = False
            if can_run_flashinfer_fusion:
                # second_input_layernorm
                hidden_states, residual, block_scale = self.input_layernorm[1].forward_with_reducescatter_fusion(
                    get_attention_tp_group(),
                    hidden_states,
                    residual,
                    fuse_block_quant_fp8=current_layer_attn_1_quanted,
                )
                # second_attn
                hidden_states = self.self_attn[1](
                    positions=positions,
                    hidden_states=hidden_states,
                    forward_batch=forward_batch,
                    comm_manager=decoder_comm_manager[1],
                    block_scale=block_scale,
                    can_run_flashinfer_fusion=can_run_flashinfer_fusion,
                )
            else:
                # second_input_layernorm
                hidden_states, residual = self.input_layernorm[1](hidden_states, residual)
                # second_attn
                hidden_states = self.self_attn[1](
                    positions=positions,
                    hidden_states=hidden_states,
                    forward_batch=forward_batch,
                    comm_manager=decoder_comm_manager[1],
                    can_run_flashinfer_fusion=can_run_flashinfer_fusion,
                )

            if can_run_flashinfer_fusion:
                hidden_states, residual, block_scale, _ = self.post_attention_layernorm[1].forward_with_allreduce_fusion(
                    get_attention_tp_group(),
                    hidden_states,
                    residual,
                    fuse_block_quant_fp8=not self.dense_1_unqaunted,
                    residual_reduce_scattered=True,
                    max_sm_to_use=69,
                    trigger_completion_at_end=True,
                )
            else:
                hidden_states, residual = decoder_comm_manager[1].post_attn_comm(hidden_states, residual, tp_num_tokens)
                # second_post_attention_layernorm
                hidden_states, residual = self.post_attention_layernorm[1](hidden_states, residual)
        # second_mlp
        hidden_states = self.forward_mlp(
            decoder_comm_manager[1],
            self.mlps[1],
            hidden_states,
            residual,
            forward_batch,
            num_global_tokens,
            max_num_tokens_per_gpu,
            tp_num_tokens,
            block_scale,
            skip_pre_comm=can_run_flashinfer_fusion,
            skip_post_comm=can_run_next_flashinfer_fusion,
        )

        return hidden_states, residual

    def forward_mlp(
        self,
        decoder_comm_manager,
        mlp,
        hidden_states,
        residual,
        forward_batch,
        num_global_tokens,
        max_num_tokens_per_gpu,
        tp_num_tokens,
        block_scale=None,
        skip_pre_comm: bool = False,
        skip_post_comm: bool = False,
    ):
        if skip_pre_comm:
            # means that we are using flashinfer allreduce fusion
            # here, skip this pre mlp comm
            start_idx = None
            end_idx = None
        else:
            # ag (mlp 0)
            hidden_states, start_idx, end_idx = decoder_comm_manager.pre_mlp_comm(hidden_states, forward_batch, tp_num_tokens)
        if isinstance(mlp, LongcatMoe):
            hidden_states = mlp(hidden_states, num_global_tokens, max_num_tokens_per_gpu, decoder_comm_manager)
        else:
            hidden_states = mlp(hidden_states, block_scale)
        # should be reduce_scatterd in next layer's input_layernorm if skiped
        if not skip_post_comm:
            hidden_states, residual = decoder_comm_manager.post_mlp_comm(hidden_states, residual, tp_num_tokens, forward_batch)

        if start_idx is not None and end_idx is not None:
            hidden_states = hidden_states[start_idx:end_idx]
        return hidden_states


class FLASHModel(nn.Module):
    def __init__(
        self,
        config: FLASHConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.pp_group = get_pp_group()
        self.enable_over_embedding = False
        if config.use_over_embedding:
            self.enable_over_embedding = True
        if config.use_over_embedding:
            self.over_embedding = FusedOverEmbedding(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                over_embedding_m=config.over_embedding_m,
                over_embedding_k=config.oe_split_num,
                over_embedding_n=config.oe_neighbor_num,
                over_embedding_base=config.over_embedding_base,
                embp=config.embp,
                oe_ignore_tokens=config.oe_ignore_tokens,
                eos_token_id=config.eos_token_id,
            )
        else:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                enable_tp=not global_server_args_dict["enable_dp_attention"],
            )
        self.alt_stream = None if self.tp_mode() else torch.cuda.Stream()
        self.kv_stream = torch.cuda.Stream()
        # used for debug
        # config.num_hidden_layers = 3; self.start_layer,self.end_layer = 0, 3
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: FLASHDecoderLayer(
                config,
                idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=self.alt_stream,
                kv_stream=self.kv_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
            return_tuple=True,
            return_indices=(1, 3),
            return_kwarg_names=("input_block_scale",),
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers_to_capture = []
        self.config = config

    def tp_mode(self) -> bool:
        return (
            global_server_args_dict["attn_parallel_strategy"] == AttnParallelStrategy.TENSOR_PARALLEL
            and global_server_args_dict["dense_parallel_strategy"] == DenseParallelStategy.TENSOR_PARALLEL
            and not global_server_args_dict["enable_ep_moe"]
        )

    def can_run_flashinfer_fusion(self, forward_batch):
        return (
            (forward_batch.input_ids.shape[0] <= global_server_args_dict["flashinfer_comm_max_num_tokens"])
            and global_server_args_dict["attn_parallel_strategy"] == AttnParallelStrategy.TENSOR_PARALLEL
            and global_server_args_dict["dense_parallel_strategy"] == DenseParallelStategy.TENSOR_PARALLEL
            and global_server_args_dict["enable_ep_moe"]
            and get_attention_tp_size() > 1
            and get_dense_tp_size() > 1
            and not getattr(self.config, "use_hybrid_linear_kda", False)
            and not forward_batch.forward_mode.is_idle()
            and not global_server_args_dict["enable_dsa_qcp"]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.over_embedding(input_ids, forward_batch) if self.enable_over_embedding else self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            if CP_METADATA:
                positions = cp_split_and_rebuild_data(positions, CP_METADATA.value.split_list, CP_METADATA.value.zigzag_index)
                # forward_batch.out_cache_loc
            residual = None
            block_scale = None
        else:
            pp_proxy_tensors = forward_batch.pp_proxy_tensors
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]
            block_scale = pp_proxy_tensors["input_block_scale"]
            if CP_METADATA:
                positions = cp_split_and_rebuild_data(positions, CP_METADATA.value.split_list, CP_METADATA.value.zigzag_index)

        tp_num_tokens = input_ids.shape[0]
        forward_batch.tp_num_tokens = tp_num_tokens
        aux_hidden_states = []
        use_fused_comm = self.can_run_flashinfer_fusion(forward_batch)
        # Track fusion state between layers

        for i in range(len(self.layers)):
            with get_global_expert_distribution_recorder().with_current_layer(i):
                layer = self.layers[i]

                if use_fused_comm:
                    # Determine next layer's input norm for fusion
                    if i + 1 >= self.start_layer and i + 1 < self.end_layer:
                        next_layer_input_norm = self.layers[i + 1].input_layernorm[0]
                        next_layer = self.layers[i + 1]
                    else:
                        next_layer_input_norm = next_layer = None
                    previous_layer_is_capture_boundary = i in self.layers_to_capture
                    current_layer_is_capture_boundary = (i + 1) in self.layers_to_capture
                    input_is_sharded = (i != self.start_layer) and not previous_layer_is_capture_boundary
                    can_run_layer_flashinfer_fusion = use_fused_comm and not previous_layer_is_capture_boundary
                    current_layer_attn_1_quanted = (
                        can_run_layer_flashinfer_fusion and not layer.attn_1_unquanted and layer._can_fuse_attn_input_fp8(layer.self_attn[1])
                    )
                    can_run_next_flashinfer_fusion = use_fused_comm and next_layer_input_norm is not None and not current_layer_is_capture_boundary
                    next_layer = None if i == len(self.layers) - 1 else self.layers[i + 1]
                    next_layer_attn_0_quanted = (
                        can_run_next_flashinfer_fusion
                        and not next_layer.attn_0_unquanted
                        and next_layer._can_fuse_attn_input_fp8(next_layer.self_attn[0])
                    )
                    hidden_states, residual, block_scale = layer(
                        positions,
                        hidden_states,
                        forward_batch,
                        residual,
                        tp_num_tokens=tp_num_tokens,
                        next_layer_input_norm=next_layer_input_norm,
                        input_is_sharded=input_is_sharded,
                        input_block_scale=block_scale,
                        can_run_flashinfer_fusion=can_run_layer_flashinfer_fusion,
                        can_run_next_flashinfer_fusion=can_run_next_flashinfer_fusion,
                        current_layer_attn_1_quanted=current_layer_attn_1_quanted,
                        next_layer_attn_0_quanted=next_layer_attn_0_quanted,
                    )
                else:
                    hidden_states, residual, _ = layer(
                        positions,
                        hidden_states,
                        forward_batch,
                        residual,
                        tp_num_tokens=tp_num_tokens,
                        next_layer_input_norm=None,
                        input_is_sharded=False,
                        input_block_scale=None,
                        can_run_flashinfer_fusion=False,
                        can_run_next_flashinfer_fusion=False,
                        current_layer_attn_1_quanted=False,
                    )

                if i + 1 in self.layers_to_capture:
                    aux_hidden_state = hidden_states.clone()
                    aux_hidden_state += residual
                    if i + 1 != self.config.num_hidden_layers:
                        aux_hidden_state = self.layers[i + 1].mlp_branch_decoder_comm_manager[0].pre_attn_comm(aux_hidden_state, tp_num_tokens)
                    else:
                        aux_hidden_state, _ = layer.mlp_branch_decoder_comm_manager[1].post_final_norm_comm(aux_hidden_state, residual, tp_num_tokens)
                    aux_hidden_states.append(aux_hidden_state)

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                    "input_block_scale": block_scale,
                }
            )

        if not forward_batch.forward_mode.is_idle():
            hidden_states, _ = self.norm(hidden_states, residual)
            if not CP_METADATA:
                hidden_states, _ = layer.mlp_branch_decoder_comm_manager[1].post_final_norm_comm(hidden_states, residual, tp_num_tokens)
        if CP_METADATA:
            hidden_states = cp_all_gather_rerange_output(hidden_states, CP_METADATA.value, layer.moe_branch_decoder_comm_manager.attn_rsag)
        if len(aux_hidden_states) == 0:
            return hidden_states
        return hidden_states, aux_hidden_states


class FLASHForCausalLM(nn.Module):
    def __init__(
        self,
        config: FLASHConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        if get_load_number_layers() != 0:
            config.num_hidden_layers = get_load_number_layers()
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = FLASHModel(config, quant_config=quant_config, prefix="model")
        self.enable_over_embedding = config.use_over_embedding
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
        self.capture_aux_hidden_states = False
        self.pp_group = get_pp_group()
        self.attn_start_layer = self.model.start_layer * 2
        self.attn_end_layer = self.model.end_layer * 2

    @property
    def routed_experts_weights_of_layer(self):
        return self._routed_experts_weights_of_layer.value

    @property
    def device_local_routed_experts_weights_of_layer(self):
        return None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch)

        aux_hidden_states = None
        if self.capture_aux_hidden_states and isinstance(hidden_states, tuple):
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            return self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states)
        else:
            return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id, begin_size)
            ("gate_up_proj", "gate_proj", 0, None),
            ("gate_up_proj", "up_proj", 1, None),
        ]
        if not is_dsa(self.config):
            stacked_params_mapping.extend(
                [
                    ("fused_qkv_a_proj_with_mqa", "q_a_proj", None, 0),
                    ("fused_qkv_a_proj_with_mqa", "kv_a_proj_with_mqa", None, self.config.q_lora_rank),
                ]
            )
        else:
            # change stack order
            stacked_params_mapping.extend(
                [
                    ("fused_qkv_a_proj_with_mqa", "q_a_proj", None, 0),
                    ("fused_qkv_a_proj_with_mqa", "indexer.wk", None, self.config.q_lora_rank),
                    ("fused_qkv_a_proj_with_mqa", "kv_a_proj_with_mqa", None, self.config.q_lora_rank + self.config.index_head_dim),
                ]
            )
        name_mapping = {
            "compress_attn": "attn.compress_attn",
            "compress_key": "compress_kv",  # kv_lora_rank
            "compress_value": "compress_k_pe",  # q_pe
            "gate_fusion.gate_weight": "attn.gate_fusion.gate_weight.weight",
        }

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts + global_server_args_dict["ep_num_redundant_experts"]
            if hasattr(self.config, "n_routed_experts")
            else self.config.num_experts[0] + global_server_args_dict["ep_num_redundant_experts"],
        )
        is_final = False
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "lm_head.weight" in name:
                is_final = True
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (layer_id < self.model.start_layer or layer_id >= self.model.end_layer)
            ):
                continue
            if get_load_number_layers() != 0:
                match = re.search(r"\d+", name)
                if match:
                    number_layer = match.group()
                    if int(number_layer) >= get_load_number_layers():
                        continue
            if "rotary_emb.inv_freq" in name:
                continue
            if ".linear_core." in name:
                name = name.replace(".linear_core.", ".")
            if ".f_proj.0." in name:
                name = name.replace(".f_proj.0.", ".f_a_proj.")
            if ".f_proj.1." in name:
                name = name.replace(".f_proj.1.", ".f_b_proj.")
            if ".g_proj.0." in name:
                name = name.replace(".g_proj.0.", ".g_a_proj.")
            if ".g_proj.1." in name:
                name = name.replace(".g_proj.1.", ".g_b_proj.")
            for source_name, target_name in name_mapping.items():
                if source_name in name:
                    name = name.replace(source_name, target_name)
            for param_name, weight_name, shard_id, begin_size in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp" in name and "mlps" not in name:
                    continue
                # [NSA]: to avoid stack 'gate_proj' in compress_attn or gate_fusion
                if "compress_attn" in name or "gate_fusion" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if (name.endswith(".bias") or name.endswith("_bias")) and name not in params_dict:
                    continue
                # Skip mtp
                if ".mtp." in name:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                if (
                    is_dsa(self.config)
                    and not dsa_layer_has_indexer(self.config, maybe_get_longcat_attn_layer_id(name))
                    and weight_name == "kv_a_proj_with_mqa"
                ):
                    begin_size -= self.config.index_head_dim
                if begin_size is not None and name.endswith(".weight_scale_inv"):
                    begin_size = begin_size // self.config.quantization_config["weight_block_size"][0]
                if "fused_qkv_a_proj_with_mqa" in name:
                    weight_loader(param, loaded_weight, shard_id, begin_size)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if "mlp.experts." in name:
                    for mapping in expert_params_mapping:
                        param_name, weight_name, local_expert_id, shard_id = mapping
                        if weight_name not in name:
                            continue
                        name = name.replace(weight_name, param_name)
                        # Skip mtp
                        if ".mtp." in name:
                            continue
                        if (name.endswith(".bias") or name.endswith("_bias")) and name not in params_dict:
                            continue
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
                    # Skip loading kv_scale from ckpts towards new design.
                    if name.endswith(".kv_scale") and name not in params_dict:
                        continue
                    ########## NSA special weight trans start ##########
                    if name.endswith("gate_fusion.gate_proj.weight"):
                        loaded_weight = loaded_weight.squeeze(0)
                    if "gate_fusion.gate_weight" in name and len(loaded_weight.shape) > 2:
                        # [qh, gate_num, gate_num*hd] -> [gate_num*gate_num*hd, qh]``
                        loaded_weight = loaded_weight.flatten(1).transpose(0, 1)
                    if "q_a_proj" in name and name not in params_dict:
                        name = name.replace("q_a_proj", "q_proj")
                    ########## NSA special weight trans end ##########
                    # Skip mtp
                    if ".mtp." in name:
                        continue
                    if self.enable_over_embedding:
                        if not self.model.pp_group.is_first_rank and any(
                            oe_key in name for oe_key in [".embed_tokens.", ".oe_embed_tokens", ".oe_embed_proj", ".ngram_embeddings"]
                        ):
                            continue
                        if ".embed_tokens." in name:
                            name = "model.over_embedding.word_embeder.weight"
                        if ".oe_embed_tokens" in name:
                            self.model.over_embedding.load_weight(None, name, loaded_weight)
                            continue
                        if ".oe_embed_proj" in name:
                            self.model.over_embedding.load_weight(None, name, loaded_weight)
                            continue
                        if ".ngram_embeddings" in name:
                            self.model.over_embedding.load_weight(None, name, loaded_weight)
                            continue
                    if name is None:
                        continue
                    try:
                        param = params_dict[name]
                    except Exception as e:
                        print(f"name: {name}")
                        print(f"keys: {params_dict.keys()}")
                        raise e
                    # Some longcat hybrid checkpoints store KDA A_log in 1D/2D
                    # (e.g. [num_heads] or [1, num_heads]), while runtime
                    # parameter layout is [1, 1, num_heads, 1].
                    if name.endswith(".A_log") and param.dim() == 4 and loaded_weight.dim() <= 2:
                        loaded_weight = loaded_weight.reshape(1, 1, -1, 1)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
        if is_final:
            self.post_load_weights()

    def post_load_weights(self):
        # weight transpose for absorb
        for layer_id in range(self.config.num_hidden_layers):
            if hasattr(self.model, "start_layer") and (layer_id < self.model.start_layer or layer_id >= self.model.end_layer):
                continue
            for i in range(2):
                self_attn: Union[DeepseekV2AttentionMLA, DeepseekNSAWithMLA, KimiDeltaAttention] = self.model.layers[layer_id].self_attn[i]
                if isinstance(self_attn, KimiDeltaAttention):
                    continue
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
                self_attn.w_kc = bind_or_assign(
                    self_attn.w_kc,
                    w_kc.transpose(1, 2).contiguous().transpose(1, 2),
                )
                self_attn.w_vc = bind_or_assign(self_attn.w_vc, w_vc.contiguous().transpose(1, 2))
                if isinstance(self_attn, DeepseekNSAWithMLA):
                    self_attn.attn.w_vc = self_attn.w_vc
                if self.config.mla_scale_q_lora:
                    self_attn.q_a_layernorm.weight.data *= (self.config.hidden_size / self.config.q_lora_rank) ** 0.5
                if self.config.mla_scale_kv_lora:
                    self_attn.kv_a_layernorm.weight.data *= (self.config.hidden_size / self.config.kv_lora_rank) ** 0.5

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_routed_experts,
            num_groups=None,
        )

    def get_embed_and_head(self):
        if not self.model.enable_over_embedding:
            return self.model.embed_tokens.weight, self.lm_head.weight
        else:
            return self.model.over_embedding, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[list[int]] = None):
        if layer_ids is None:
            self.capture_aux_hidden_states = True
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [2, num_layers // 2, num_layers - 3]
        else:
            self.capture_aux_hidden_states = True
            self.model.layers_to_capture = [val + 1 for val in layer_ids]


EntryClass = FLASHForCausalLM
