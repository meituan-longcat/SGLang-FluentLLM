
from typing import Any, Dict, Optional

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.configs.model_config import (
    get_nsa_index_head_dim,
    get_nsa_index_n_heads,
    get_nsa_index_topk,
    is_dsa,
)
from sglang.srt.layers.attention.dsa.nsa_indexer import Indexer, IndexerBf16
from sglang.srt.layers.utils import (
    FLLM_IS_CP,
    CP_METADATA, cp_all_gather_rerange_output
)
from sglang.srt.utils import get_device_sm, is_cuda, is_npu

_is_npu = is_npu()
_is_cuda = is_cuda()
_device_sm = get_device_sm()

if not _is_npu:
    from flashinfer import dsv3_fused_a_gemm

from sglang.srt.layers.layernorm import RMSNorm, FusedRMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.env import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import add_prefix
from sglang.srt.layers.dp_attention import get_attention_tp_group
from sglang.srt.distributed.decoder_comm_manager import DecoderCommMananger

from sglang.srt.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0

class DeepseekV32MLA(nn.Module):

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
        self.index_head_dim = config.index_head_dim

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
        assert alt_stream is not None
        self.alt_stream = alt_stream
        self.attention_backend = global_server_args_dict["attention_backend"]
        self.cli_factor = getattr(config, "cli_factor", 1)
        self.prefix = prefix

        # modification to rope_scaling must be done early enough, b/c e.g. Indexer needs it
        if rope_scaling:
            rope_scaling["rope_type"] = "deepseek_yarn"

        if self.q_lora_rank is not None:
            if not _is_npu:
                self.fused_qkv_a_proj_with_mqa = ReplicatedLinear(
                    self.hidden_size,
                    self.q_lora_rank + self.index_head_dim + self.kv_lora_rank + self.qk_rope_head_dim,
                    bias=False,
                    quant_config=quant_config,
                    prefix=add_prefix("fused_qkv_a_proj_with_mqa", prefix),
                )
            else:
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

        self.use_dsa = is_dsa(config)
        if self.use_dsa:
            IndexerImpl = Indexer
            is_neox_style = not getattr(config, "indexer_rope_interleave", False)
            if config.architectures[0] in ["FLASHForCausalLM", "FLASHForCausalLMNextN"]:
                IndexerImpl = IndexerBf16
                is_neox_style = False
            self.indexer = IndexerImpl(
                hidden_size=hidden_size,
                index_n_heads=get_nsa_index_n_heads(config),
                index_head_dim=get_nsa_index_head_dim(config),
                rope_head_dim=qk_rope_head_dim,
                index_topk=get_nsa_index_topk(config),
                index_k_norm_type=getattr(config, "index_k_norm_type", "layer"),
                q_lora_rank=q_lora_rank,
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
                scale_fmt="ue8m0",
                block_size=128,
                rope_scaling=rope_scaling,
                is_neox_style=is_neox_style,
                prefix=add_prefix("indexer", prefix),
                quant_config=quant_config,
                layer_id=layer_id,
                alt_stream=alt_stream,
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

        # Fused layer
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

        self.attn_mqa = RadixAttention(
            self.num_local_heads,
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

        if _is_npu:
            self.forward = self.forward_npu

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


    def no_absorb(self, forward_batch: ForwardBatch) -> bool:
        return not self.use_dsa
        # if global_server_args_dict["enable_flashinfer_mla"]:
            # not global_server_args_dict["flashinfer_mla_disable_ragged"]
            # and forward_batch.forward_mode.is_extend()
            # and not forward_batch.forward_mode.is_target_verify()
            # and not forward_batch.forward_mode.is_draft_extend()
            # and not self.use_dsa
            # and not forward_batch.captureing_prefill_graph
         
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        comm_manager: DecoderCommMananger,
        block_scale: Optional[torch.Tensor] = None,
        can_run_flashinfer_fusion: bool = False,
    ) -> torch.Tensor:
        # alway absorb for now
        return self.forward_absorb(positions, hidden_states, forward_batch, comm_manager, block_scale, can_run_flashinfer_fusion)

    def forward_absorb(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        comm_manager: DecoderCommMananger,
        block_scale: Optional[torch.Tensor] = None,
        can_run_flashinfer_fusion: bool = False,
    ) -> torch.Tensor:
        Q, K, topk_indices = self.forward_absorb_qkv_proj(hidden_states, positions, forward_batch, comm_manager, block_scale, can_run_flashinfer_fusion)

        output = self.forward_absorb_attn_o_proj(Q, K, forward_batch, topk_indices)

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
        q_lora = None
        hidden_states = comm_manager.pre_attn_comm(hidden_states, forward_batch.tp_num_tokens)
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
        # self.indexer.get_head_gate(hidden_states, out)
        # qkv = comm_manager.pre_attn_comm(qkv, forward_batch.tp_num_tokens)
        q, index_k, latent_cache = qkv.split(
            [self.q_lora_rank, self.index_head_dim, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
        )
        k_nope = latent_cache[..., : self.kv_lora_rank]

        # fused layernorm
        q_contiguous = torch.empty_like(q)
        if q.shape[0] > 0:
            self.fused_qk_layernorm(input_q_a=q, input_kv_a=k_nope, output_q_a=q_contiguous)

        # q_lora needed by indexer
        if self.use_dsa:
            q_lora = q_contiguous

        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)

        indexer_forward = lambda: self.indexer(
            x=hidden_states,
            q_lora=q_lora,
            index_k=index_k,
            positions=positions,
            forward_batch=forward_batch,
            layer_id=self.layer_id,
            comm_manager=comm_manager,
        )

        if "FLASHForCausalLMNextN" in self.config.architectures[0]:
            # hack for MTP3H
            if global_server_args_dict["is_multi_head_eagle"]:
                if forward_batch.topk_indices is None:
                    topk_indices = indexer_forward()
                    forward_batch.topk_indices = topk_indices
                else:
                    topk_indices = forward_batch.topk_indices
            else:
                topk_indices = indexer_forward()
        else:
            if self.cli_factor <= 1 or self.layer_id % self.cli_factor == 0:
                topk_indices = indexer_forward()
            else:
                topk_indices = forward_batch.topk_indices
            if self.cli_factor > 1:
                forward_batch.topk_indices = topk_indices

        with torch.cuda.stream(self.alt_stream):
            q = self.q_b_proj(q_contiguous)[0].view(-1, self.num_local_heads, self.qk_head_dim)

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
                    fused_set_kv_buffer_arg=None,
                    output_q_rope = Q[..., self.kv_lora_rank :],
                )

        current_stream.wait_stream(self.alt_stream)

        if CP_METADATA:
            # alt stream may cause err, because main stream indexer also all_gather
            K = cp_all_gather_rerange_output(K.squeeze(1), CP_METADATA.get(), comm_manager.attn_rsag).unsqueeze(1)

        return Q, K, topk_indices

    def forward_absorb_attn_o_proj(
        self, Q, K,
        forward_batch: ForwardBatch,
        topk_indices=None
    ):
        q_nope_out = Q[..., : self.kv_lora_rank]
        q_pe = Q[..., self.kv_lora_rank :]
        k_nope = K[..., : self.kv_lora_rank]
        k_pe = K[..., self.kv_lora_rank :]

        assert self.attention_backend == "dsa"
        attn_output = self.attn_mqa(
            q_nope_out, k_nope, k_nope, forward_batch,
            q_pe=q_pe,
            k_pe=k_pe,
            topk_indices=topk_indices
        )
        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        attn_bmm_output = torch.bmm(attn_output.transpose(0, 1), self.w_vc)
        attn_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
        output, _ = self.o_proj(attn_output)
        return output
