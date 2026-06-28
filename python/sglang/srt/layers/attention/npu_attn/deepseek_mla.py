from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch_npu
import copy

from sglang.srt.configs import FLASHConfig
from sglang.srt.layers.attention.npu_attn.flash_attn import CacheConfig
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.dense.layouts.unquant import UnquantizedLinearMethod
from sglang.srt.configs.model_config import (
    is_deepseek_nsa,
    is_dsa,
    get_nsa_index_head_dim,
    get_nsa_index_topk,
    get_nsa_index_n_heads
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    ReplicatedLinear,
    LinearBase
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.distributed import (
    get_attn_tp_world_size,
    get_attn_tp_group,
    get_attn_o_proj_tp_group,
    get_ep_group,
    get_attn_sp_token_slice_pos,
)
from sglang.srt.layers.attention.dsa.nsa_indexer import Indexer
from sglang.srt.env import ENV, global_server_args_dict
from sglang.srt.utils import get_colorful_logger
logger = get_colorful_logger(__name__)


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math

    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class DeepseekV2MLAAttention(torch.nn.Module):
    """
    Main reference: DeepseekV2 paper, and FlashInfer Implementation
    (https://arxiv.org/abs/2405.04434 and https://github.com/flashinfer-ai/flashinfer/pull/551).

    For more info see MLACommonImpl in: vllm/attention/backends/mla/utils.py
    """

    def __init__(
        self,
        config: Union[FLASHConfig, Any],
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: Optional[int],
        kv_lora_rank: int,
        bias: bool = False,
        rope_theta: float = 10000,
        rms_norm_eps: float = 1.0e-6,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        layer_id=None,
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        global v_head_dim_global
        global hidden_size_global
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        hidden_size_global = hidden_size
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        v_head_dim_global = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.bias = bias
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.rope_scaling = rope_scaling # None
        self.max_position_embeddings = max_position_embeddings
        self.cache_config = cache_config
        self.quant_config = quant_config
        self.reduce_results = reduce_results
        self.config = config
        self.enable_nsa = is_deepseek_nsa(config)
        self.npu_disable_dsa_head_parallel = self.enable_nsa and global_server_args_dict["npu_disable_dsa_head_parallel"] and global_server_args_dict["disaggregation_mode"] == "prefill"

        self.attn_tp_size = get_attn_tp_world_size()
        assert num_heads % self.attn_tp_size == 0
        self.num_local_heads = num_heads // self.attn_tp_size
        self.scaling = self.qk_head_dim**-0.5
        if rope_scaling != None:
            assert self.rope_scaling["rope_type"] == "deepseek_yarn"
            mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
            scaling_factor = rope_scaling["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        attn_tp_group = None if self.attn_tp_size == 1 else get_attn_tp_group()
        assert self.q_lora_rank is not None  # assert MLA
        # no merged qkv_a when using npu, need add for gpu
        self.q_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.q_lora_rank,
            bias=False,
            quant_config=self.quant_config,
            enable_weight_transpsoe=True,
            enable_weight_nz=True
        )
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=self.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=bias,
            quant_config=quant_config,
            enable_weight_transpsoe=True,
            enable_weight_nz=True,
            outside_tp_group=None if self.npu_disable_dsa_head_parallel else attn_tp_group,
            disable_parallel=True,
        )

        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=self.quant_config,
            enable_weight_transpsoe=True,
            enable_weight_nz=True
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=self.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=bias,
            quant_config=quant_config,
            outside_tp_group=None if self.npu_disable_dsa_head_parallel else attn_tp_group,
            disable_parallel=True,
        )

        # different o_proj tp mode
        # todo
        self.o_proj_tp_group = None
        if global_server_args_dict["npu_o_proj_tp_size"] > 0:
            self.o_proj_tp_group = get_attn_o_proj_tp_group()
        else:
            self.o_proj_tp_group = attn_tp_group

        split_n = False  # Set True only for attn tp size == 1
        if split_n and self.attn_tp_size == 1:
            assert global_server_args_dict["npu_o_proj_tp_size"] > 0
            # when tp_size > 1, need add all2all before o_proj, not support now
            self.o_proj = ColumnParallelLinear(
                self.num_heads * self.v_head_dim,
                self.hidden_size,
                bias=bias,
                quant_config=quant_config,
                outside_tp_group=None if self.npu_disable_dsa_head_parallel else self.o_proj_tp_group,
                disable_parallel=True,
                enable_weight_nz=True,
                enable_weight_transpsoe=True,
                gather_output=False  # use all2all
            )
        else:
            self.o_proj = RowParallelLinear(
                self.num_heads * self.v_head_dim,
                self.hidden_size,
                bias=bias,
                quant_config=quant_config,
                reduce_results=reduce_results,  # need skip in forward process
                outside_tp_group=None if self.npu_disable_dsa_head_parallel else self.o_proj_tp_group,
                disable_parallel=True,
                enable_weight_nz=True,
                enable_weight_transpsoe=True
            )

        # for longcat extra prolog scale parameter
        self.mla_scale_q_lora = (self.hidden_size / self.q_lora_rank) ** 0.5
        self.mla_scale_kv_lora = (self.hidden_size / self.kv_lora_rank) ** 0.5
        self.mla_scale_v = None

        
        self.attn_mqa = RadixAttention(
            self.num_local_heads,
            self.kv_lora_rank + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=1,
            layer_id=layer_id,
            v_head_dim=self.kv_lora_rank,
            is_npu_dsa=True if self.enable_nsa else False,
        )

        self.attn_mha = RadixAttention(
            self.num_local_heads,
            self.qk_nope_head_dim + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=self.num_local_heads,
            layer_id=layer_id,
            v_head_dim=self.v_head_dim,
            is_npu_dsa=True if self.enable_nsa else False,
        )

        self.w_kc = None
        self.w_vc = None

        if self.enable_nsa:
            self.index_topk = getattr(config, 'index_topk', 128)
            self.cli_factor = getattr(config, 'cli_factor', 1)
            self.indexer = Indexer(
                hidden_size=self.hidden_size,
                index_n_heads=get_nsa_index_n_heads(config),
                index_head_dim=get_nsa_index_head_dim(config),
                rope_head_dim=self.qk_rope_head_dim,
                index_topk=self.index_topk,
                index_k_norm_type="rms",
                q_lora_rank=self.q_lora_rank,
                max_position_embeddings=self.max_position_embeddings,
                rope_theta=self.rope_theta,
                layer_id=layer_id,
                rope_scaling=self.rope_scaling,
                config=config,
                quant_config=quant_config,
                scale_fmt=None
            )

    def forword_absorb_prepare(
            self,
            rotary_emb,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            kv_cache: torch.Tensor,
            forward_batch: ForwardBatch):
        if len(hidden_states.size()) == 2:
            hidden_states = torch.unsqueeze(hidden_states, dim=1)
        kv_cache, kr_cache = kv_cache
        if len(kv_cache.size()) == 3:
            kv_cache = torch.unsqueeze(kv_cache, dim=2)
            kr_cache = torch.unsqueeze(kr_cache, dim=2)

        cache_mode = "PA_BSND" if global_server_args_dict["npu_disable_kv_nz"] else "PA_NZ"
        prolog_v3 = hasattr(torch_npu, 'npu_mla_prolog_v3')
        mla_prolog = torch.ops.npu.npu_mla_prolog
        kwargs = dict()
        if prolog_v3:
            mla_prolog = torch_npu.npu_mla_prolog_v3
            kwargs = dict(
                qc_qr_scale=self.mla_scale_q_lora,
                kc_scale=self.mla_scale_kv_lora
            )
        mla_output = mla_prolog(
            token_x=hidden_states, # (B,S,He)
            weight_dq=self.q_a_proj.weight,# (He,Hcq), (7168,1536)
            weight_uq_qr=self.q_b_proj.weight, # (Hcq,N*(D+Dr)), (1536, 64*(128+64))
            weight_uk=self.kv_b_proj.W_UK_T, # (N,D,Hckv), (64, 128, 512)
            weight_dkv_kr=self.kv_a_proj_with_mqa.weight, # (He,Hckv+Dr), (7168, 576)
            rmsnorm_gamma_cq=self.q_a_layernorm.weight,
            rmsnorm_gamma_ckv=self.kv_a_layernorm.weight,
            rmsnorm_epsilon_cq=self.rms_norm_eps,
            rmsnorm_epsilon_ckv=self.rms_norm_eps,
            rope_sin=rotary_emb[1], # (B,S,Dr)
            rope_cos=rotary_emb[0], # (B,S,Dr)
            cache_index=forward_batch.attn_metadata.slot_mapping.reshape(-1, 1), # (B,S)
            kv_cache=kv_cache, # (BlockNum, BlockSize, Nkv, Hckv)
            kr_cache=kr_cache, # (BlockNum, BlockSize, Nkv, Dr)
            dequant_scale_x=None,
            dequant_scale_w_dq=None,
            dequant_scale_w_uq_qr=None,
            dequant_scale_w_dkv_kr=None,
            smooth_scales_cq=None,
            quant_scale_ckv=None,
            cache_mode=cache_mode,
            **kwargs)
        q_nope, q_rope = mla_output[:2]
        return q_nope, q_rope, kv_cache, kr_cache, forward_batch

    def forward_absorb_core(self, q_nope, q_rope, k_nope, k_rope, forward_batch, return_ctrl=False, topk_indices=None):
        layout = 'TND_NTD'
        attn_output = self.attn_mqa(
            q_nope,
            k_nope,
            k_nope,
            forward_batch,
            q_pe=q_rope,
            k_pe=k_rope,
            topk_indices=topk_indices,
            layout=layout,
            absorbed=True)

        if layout not in ['TND_NTD', 'BSND_NBSD']:
            attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank).transpose(0, 1)
        # Multiply (N, B, L) x (N, L, V) -> (N, B, V)
        attn_output = torch.bmm(attn_output, self.kv_b_proj.W_UV).transpose(0, 1)
        # Convert from (N, B, V) to (B, N * V)
        output = self.forward_attn_dp_o_proj(attn_output, return_ctrl=return_ctrl)
        return output

    def forward_normal_chunked_kv_prepare(
            self,
            rotary_emb,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            kv_cache: torch.Tensor,
            forward_batch: ForwardBatch,
            gather_qkv: bool=False,
            reduce_type: str='reduce_scatter',
            extra_input: Any=None):
        if extra_input is not None:
            q, latent_cache = extra_input
        else:
            q = self.q_a_proj(hidden_states)[0]
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
            if gather_qkv:
                if not self.npu_disable_dsa_head_parallel:
                    q = get_attn_tp_group().gather_tensor(q, forward_batch.global_sp_num_tokens)
                latent_cache = get_attn_tp_group().gather_tensor(latent_cache, forward_batch.global_sp_num_tokens)
        attn_metadata = forward_batch.attn_metadata
        num_prefill_tokens = attn_metadata.num_prefill_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens

        cos, sin = rotary_emb
        cos = cos.view(-1, 1, 1, self.qk_rope_head_dim)
        sin = sin.view(-1, 1, 1, self.qk_rope_head_dim)


        q_num_heads = self.num_local_heads
        q_cos = cos
        q_sin = sin

        if self.npu_disable_dsa_head_parallel:
            q_num_heads = self.num_heads
            slice_start, slice_end = get_attn_sp_token_slice_pos(forward_batch)
            q_cos = cos[slice_start:slice_end]
            q_sin = sin[slice_start:slice_end]
            if extra_input is not None:
                q = q[slice_start:slice_end]

        # mla去掉padding
        q = self.q_a_layernorm(q)
        q = q * self.mla_scale_q_lora
        q = self.q_b_proj(q)[0].view(-1, q_num_heads, self.qk_head_dim)

        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = q_pe.unsqueeze(2)
        q_pe = torch_npu.npu_interleave_rope(q_pe, q_cos, q_sin)  # BNSD
        q_pe = q_pe.squeeze(2)  # BSH

        # kv [N, 1, 576]
        if len(kv_cache) == 1:
            kv_pool = kv_cache[0]
            kv_nope_pool = kv_pool[..., :self.kv_lora_rank].view(-1, 128, 1, 512)
            k_pe_pool = kv_pool[..., self.kv_lora_rank:].view(-1, 128, 1, 64)
        else:
            kv_nope_pool = kv_cache[0].view(-1, 128, 1, 512)
            k_pe_pool = kv_cache[1].view(-1, 128, 1, 64)

        # k_cache形状为 [block_num, block_size, 1, rope_size]，ckv_cache形状为[block_num, block_size, 1, rms_size]
        # index形状为[batch_size * seq_len]，index里的值表示每个token的偏移。
        # k_pe:BNS,64 kv_a:BNS, 512, kv_states:bnsd, cos,sin:bnsd,kv cache:bsnd
        if len(kv_cache) == 1:
            kv_nope, k_pe = torch.split(latent_cache.view(-1, 576), [512, 64], dim=-1)
            kv_nope = self.kv_a_layernorm_absorb_scale(kv_nope)
            k_pe = k_pe.view(-1, 1, 1, 64)
            k_pe = torch_npu.npu_interleave_rope(k_pe, cos, sin)
            kv = torch.concat((kv_nope, k_pe.view(-1, 64)), dim=-1)
            torch_npu.npu_scatter_nd_update_(kv_pool.reshape(-1, 576), attn_metadata.slot_mapping.reshape(-1, 1),
                                             kv.reshape(-1, 576))
        else:
            cache_mode = "PA" if global_server_args_dict["npu_disable_kv_nz"] else "PA_NZ"
            _, _, _, _ = torch_npu.npu_kv_rmsnorm_rope_cache(
                latent_cache.view(-1, 1, 1, 576),  # bnsd
                self.kv_a_layernorm_absorb_scale.weight,
                cos.view(-1, 1, 1, self.qk_rope_head_dim),
                sin.view(-1, 1, 1, self.qk_rope_head_dim),
                attn_metadata.slot_mapping,
                k_pe_pool,
                kv_nope_pool,
                k_rope_scale=None,
                c_kv_scale=None,
                k_rope_offset=None, c_kv_offset=None,
                epsilon=self.kv_a_layernorm.variance_epsilon,
                cache_mode=cache_mode,
                is_output_kv=False)

        output = [(num_prefill_tokens, num_decode_tokens)]
        if num_prefill_tokens > 0:
            prefill_q_nope = q_nope[:num_prefill_tokens]
            prefill_q_pe = q_pe[:num_prefill_tokens]
            if len(kv_cache) == 1:
                kv = kv_pool.reshape(-1, self.kv_lora_rank + self.qk_rope_head_dim
                                     ).index_select(0, attn_metadata.kv_index_list)
                kv_a, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            else:
                if not global_server_args_dict["npu_disable_kv_nz"]:
                    KVCACHE_NZ_DIM = 16  # for half
                    block_num, block_size, _, _ = kv_nope_pool.shape
                    if attn_metadata.kv_block_index_list is not None:
                        mini_kv_nope_pool=kv_nope_pool.index_select(0, attn_metadata.kv_block_index_list)
                        mini_kv_pe_pool=k_pe_pool.index_select(0, attn_metadata.kv_block_index_list)
                        kv_cache_a=(mini_kv_nope_pool
                                    .view(-1, 1, self.kv_lora_rank//KVCACHE_NZ_DIM, block_size, KVCACHE_NZ_DIM))
                        kv_cache_pe=(mini_kv_pe_pool
                                     .view(-1, 1, self.qk_rope_head_dim//KVCACHE_NZ_DIM, block_size, KVCACHE_NZ_DIM))
                    else:
                        kv_cache_a = kv_nope_pool.view(
                            block_num, 1, self.kv_lora_rank // KVCACHE_NZ_DIM, block_size, KVCACHE_NZ_DIM)
                        kv_cache_pe = k_pe_pool.view(
                            block_num, 1, self.qk_rope_head_dim // KVCACHE_NZ_DIM, block_size, KVCACHE_NZ_DIM)
                    kv_nope_pool = kv_cache_a.transpose(1, 3)
                    k_pe_pool = kv_cache_pe.transpose(1, 3)
                kv_a = kv_nope_pool.reshape(-1, self.kv_lora_rank).index_select(0, attn_metadata.kv_index_list)
                k_pe = k_pe_pool.reshape(-1, self.qk_rope_head_dim).index_select(0, attn_metadata.kv_index_list)

                if global_server_args_dict.get("enable_mla_l1_5_cache", False):
                    rank_counts = attn_metadata.per_rank_count
                    inv_perm = attn_metadata.inv_perm
                    # 2. Each rank loads only its own KV tokens using local indices
                    kv_a_local = kv_a
                    k_pe_local = k_pe

                    # 3. Variable-size all_gather (each rank may own different number of tokens)
                    attn_tp_group = get_attn_tp_group()
                    tp_size = attn_tp_group.world_size
                    device_group = attn_tp_group.device_group

                    kv_a_gathered = [
                        torch.empty(rank_counts[r], self.kv_lora_rank,
                                    dtype=kv_a_local.dtype, device=kv_a_local.device)
                        for r in range(tp_size)
                    ]
                    k_pe_gathered = [
                        torch.empty(rank_counts[r], self.qk_rope_head_dim,
                                    dtype=k_pe_local.dtype, device=k_pe_local.device)
                        for r in range(tp_size)
                    ]

                    torch.distributed.all_gather(kv_a_gathered, kv_a_local, group=device_group)
                    torch.distributed.all_gather(k_pe_gathered, k_pe_local, group=device_group)

                    # 4. Concatenate + restore original token order via inv_perm
                    kv_a = torch.cat(kv_a_gathered, dim=0)[inv_perm]
                    k_pe = torch.cat(k_pe_gathered, dim=0)[inv_perm]

            if self.enable_nsa:
                #TND [bs,kv_n,dim]
                k_nope = kv_a.contiguous().view(-1, 1, self.kv_lora_rank)
                k_pe = k_pe.contiguous().view(-1, 1, self.qk_rope_head_dim)
                v = k_nope
                prefill_q_nope = prefill_q_nope.transpose(0, 1)
                prefill_q_nope = torch.bmm(prefill_q_nope, self.kv_b_proj.W_UK_T)
                prefill_q_nope = prefill_q_nope.transpose(0, 1)
            else:
                kv_a = kv_a.contiguous().view(-1, self.kv_lora_rank)
                k_pe = k_pe.contiguous().view(-1, 1, self.qk_rope_head_dim).repeat(1, self.num_local_heads, 1)

                # TODO group attention
                kv = self.kv_b_proj(kv_a)[0].view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
                k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

            output.append((prefill_q_nope, k_nope, v, prefill_q_pe, k_pe))

        if num_decode_tokens > 0:
            decode_q_nope = q_nope[num_prefill_tokens:num_prefill_tokens + num_decode_tokens]
            decode_q_pe = q_pe[num_prefill_tokens:num_prefill_tokens + num_decode_tokens]
            decode_q_nope = decode_q_nope.transpose(0, 1)
            decode_ql_nope = torch.bmm(decode_q_nope, self.kv_b_proj.W_UK_T)
            decode_ql_nope = decode_ql_nope.transpose(0, 1)
            output.append((decode_ql_nope, kv_cache[0], kv_cache[0], decode_q_pe, kv_cache[1]))
        return output, forward_batch, reduce_type

    def forward_normal_chunked_kv_core(self, kv_data, forward_batch: ForwardBatch, reduce_type, topk_indices=None):
        # after attention, add padding
        assert len(kv_data) >= 1
        num_prefill_tokens, num_decode_tokens = kv_data[0]
        output_size = num_prefill_tokens + num_decode_tokens
        if num_decode_tokens > 0:
            output = torch.empty(output_size,
                                 self.v_head_dim * self.num_local_heads,
                                 device=kv_data[1][0].device,
                                 dtype=kv_data[1][0].dtype)
        if num_prefill_tokens > 0:
            assert len(kv_data) >= 2
            q, k, v, q_pe, k_pe = kv_data[1]
            layout = 'TND'
            q_num_heads = q.shape[1]
            attn_output = self.attn_mha(
                q,
                k,
                v,
                forward_batch,
                q_pe=q_pe,
                k_pe=k_pe,
                topk_indices=topk_indices,
                layout=layout)
            if self.enable_nsa:
                # 512 -> 128
                attn_output_transposed = attn_output.transpose(0, 1)  # [N, B*S, L]
                attn_output = torch.bmm(attn_output_transposed, self.kv_b_proj.W_UV)  # [N, B*S, V]
                attn_output = attn_output.transpose(0, 1).contiguous()
            if num_decode_tokens > 0:
                output[:num_prefill_tokens] = attn_output.view(-1, self.v_head_dim * q_num_heads)
            else:
                output = attn_output.view(-1, self.v_head_dim * q_num_heads)
        if num_decode_tokens > 0:
            assert len(kv_data) >= 3
            q, k, v, q_pe, k_pe = kv_data[2]
            layout = 'TND'
            attn_output = self.attn_mqa(
                q,
                k,
                v,
                forward_batch,
                q_pe=q_pe,
                k_pe=k_pe,
                layout=layout,
                absorbed=True)

            if layout not in ['TND_NTD', 'BSND_NBSD']:
                attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank).transpose(0, 1)
            # Multiply (N, B, L) x (N, L, V) -> (N, B, V)
            output[num_prefill_tokens:num_prefill_tokens + num_decode_tokens] = torch.bmm(attn_output, self.kv_b_proj.W_UV)
        attn_sp_token_num=None if reduce_type=='skip' else get_attn_tp_group().get_local_sp_token_num(forward_batch.global_sp_num_tokens)
        o_proj_output = self.forward_attn_tp_o_proj(output, reduce_type, attn_sp_token_num)
        return o_proj_output

    def forward_attn_tp_o_proj(self, attn_out, reduce_type, attn_sp_token_num=None):
        if global_server_args_dict["npu_o_proj_tp_size"] == 0 or global_server_args_dict["npu_o_proj_tp_size"] == self.attn_tp_size:
            if reduce_type == 'skip':
                return self.o_proj(attn_out, force_skip_reduce_results = True)[0]
            elif reduce_type == 'reduce_scatter':
                out = self.o_proj(attn_out, force_skip_reduce_results = True)[0]
                return self.post_comm(out, reduce_type, input_split_sizes=attn_sp_token_num)
            else:
                raise ValueError(f'o_project tp not support reduce type {reduce_type}')

        assert get_attn_o_proj_tp_group().world_size == 1, "only support o_proj tp1 on prefill"
        if self.npu_disable_dsa_head_parallel:
            return self.o_proj(attn_out, force_skip_reduce_results = True)[0]
        h = attn_out.shape[1]
        attn_tp_size = get_attn_tp_group().world_size
        attn_tp_rank = get_attn_tp_group().rank_in_group
        output_flat = attn_out.reshape(-1)
        output_all2all = torch.empty([attn_sp_token_num[attn_tp_rank]*h*attn_tp_size,],
                                     dtype=output_flat.dtype,
                                     device=output_flat.device)

        input_split_sizes=[x*h for x in attn_sp_token_num]
        output_split_sizes=[h * attn_sp_token_num[attn_tp_rank]] * attn_tp_size
        # logger.info(f"{output_flat.shape=} {input_split_sizes=} {output_split_sizes=} {attn_sp_token_num=}")
        torch.distributed.all_to_all_single(output_all2all, output_flat,
                                            group=get_attn_tp_group().device_group,
                                            input_split_sizes=input_split_sizes,
                                            output_split_sizes=output_split_sizes,)
        # logger.info(f"{output_all2all.shape} {input_split_sizes=} {output_split_sizes=}")

        #  shape '[8, 1, 1024]' is invalid for input of size 4096
        output = (output_all2all.reshape(attn_tp_size, attn_sp_token_num[attn_tp_rank], h)
                  .permute(1, 0, 2).contiguous().reshape(attn_sp_token_num[attn_tp_rank], h*attn_tp_size))
        o_proj_output = self.o_proj(output, force_skip_reduce_results = True)[0]
        return o_proj_output

    def post_comm(self, o_proj_output, reduce_type='reduce_scatter', input_split_sizes=None):
        if reduce_type == 'reduce_scatter':
            if not self.npu_disable_dsa_head_parallel:
                return self.o_proj_tp_group.reduce_scatter(o_proj_output, input_split_sizes=input_split_sizes)
            else:
                return o_proj_output
        else:
            raise ValueError(f'o_project tp not support reduce type {reduce_type}')

    def forward_attn_dp_o_proj(self, attn_out, return_ctrl=False):
        if global_server_args_dict["npu_o_proj_tp_size"] <= 0:
            o_proj_output = self.o_proj(attn_out.reshape(attn_out.shape[0], -1))[0]
            return o_proj_output, None
        bs = attn_out.shape[0]

        if isinstance(self.o_proj, ColumnParallelLinear):
            # dp -> tp n -> dp: all_gather(bs) -> o_proj(k, n/tp) -> all2all(split n -> split bs)
            attn_out = self.o_proj_tp_group.all_gather(attn_out, dim=0)

            o_proj_output = self.o_proj(attn_out.reshape(attn_out.shape[0], -1))[0]

            output = torch.empty([bs * self.hidden_size], dtype=attn_out.dtype, device=attn_out.device)
            torch.distributed.all_to_all_single(output, o_proj_output.reshape(-1), group=self.o_proj_tp_group.device_group)
            output = output.view(-1, bs, o_proj_output.shape[-1]).transpose(0, 1).reshape(bs, hidden_size_global)
        else:
            # dp -> tp k -> dp: all2all(split bs -> split k) -> o_proj(k/tp, n) -> reduce_scatter(partial output -> split bs)
            # reshape for matmul+tranpose fusion
            if get_attn_tp_group().world_size == 1:
                attn_out_tmp = attn_out.reshape(bs, 1, -1).view(
                        bs, global_server_args_dict["npu_o_proj_tp_size"], (self.num_local_heads * v_head_dim_global) // global_server_args_dict["npu_o_proj_tp_size"]
                    ).transpose(0, 1).reshape(-1)
                attn_out_k = torch.empty([bs * self.num_local_heads * v_head_dim_global], dtype=attn_out.dtype, device=attn_out.device)
                torch.distributed.all_to_all_single(attn_out_k, attn_out_tmp, group=self.o_proj_tp_group.device_group)
                attn_out = attn_out_k.view(bs * global_server_args_dict["npu_o_proj_tp_size"], -1)

                o_proj_output = self.o_proj(attn_out, force_skip_reduce_results=True)[0]

                output = torch.empty([bs, hidden_size_global], dtype=attn_out.dtype, device=attn_out.device)
                torch.distributed.reduce_scatter_tensor(output, o_proj_output, group=self.o_proj_tp_group.device_group)
            elif get_attn_tp_group().world_size == get_attn_o_proj_tp_group().world_size:
                output=self.o_proj(attn_out.reshape(attn_out.shape[0], self.num_local_heads * v_head_dim_global), force_skip_reduce_results=True)[0]
                torch.distributed.all_reduce(output, group=self.o_proj_tp_group.device_group)
                o_proj_output=output
            else:
                assert False, "not support now"
        if return_ctrl:
            return output, o_proj_output
        return output

    def forward(
            self,
            rotary_emb,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            kv_cache: torch.Tensor,
            forward_batch: ForwardBatch,
            gather_qkv: bool=False,
            reduce_type: str='reduce_scatter',
            return_ctrl: bool=False,
            extra_input: Any=None,
            atten_id: int=0,):
        topk_indices = None
        if self.enable_nsa:
            topk_indices = self.indexer_fn(positions, hidden_states, forward_batch, gather_qkv, extra_input, atten_id)
        if forward_batch.all_decode_or_idle:
            s = self.forword_absorb_prepare(
                rotary_emb, positions, hidden_states, kv_cache, forward_batch)
            return self.forward_absorb_core(*s, return_ctrl=return_ctrl, topk_indices=topk_indices)
        else:
            if forward_batch.forward_mode.is_idle():
                return hidden_states
            s = self.forward_normal_chunked_kv_prepare(
                rotary_emb, positions, hidden_states, kv_cache, forward_batch, gather_qkv, reduce_type, extra_input)
            output = self.forward_normal_chunked_kv_core(*s, topk_indices=topk_indices)
            if return_ctrl:
                # prefill has no ctrl tensor to return
                return output, None
            return output

    def indexer_fn(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            forward_batch: ForwardBatch,
            gather_qkv: bool=False,
            extra_input: Any=None,
            atten_id: int=0,):
        if "FLASHForCausalLMNextN" in self.config.architectures[0]:
            # hack for MTP3H
            if global_server_args_dict["is_multi_head_eagle"]:
                if forward_batch.topk_indices is None:
                    topk_indices = self.get_topk_indcies(positions, hidden_states, forward_batch, self.layer_id, gather_qkv, extra_input)
                    forward_batch.topk_indices = topk_indices
                else:
                    topk_indices = forward_batch.topk_indices
            else:
                topk_indices = self.get_topk_indcies(positions, hidden_states, forward_batch, self.layer_id, gather_qkv, extra_input)
        else:
            # add for cli. two ds-atten
            layer_id = self.layer_id * 2 if not atten_id else self.layer_id * 2 + 1
            if self.cli_factor <=1 or layer_id % self.cli_factor == 0:
                topk_indices = self.get_topk_indcies(positions, hidden_states, forward_batch, layer_id, gather_qkv, extra_input)
            else:
                topk_indices = forward_batch.topk_indices
            if self.cli_factor > 1:
                forward_batch.topk_indices = topk_indices
        return topk_indices

    def get_topk_indcies(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            forward_batch: ForwardBatch,
            layer_id: int,
            gather_qkv: bool=False,
            extra_input: Any=None):
        if extra_input is not None:
            q_lora, _ = extra_input
            global_sp_num_tokens = forward_batch.global_sp_num_tokens
            attn_sp_token_nums = get_attn_tp_group().get_local_sp_token_num(global_sp_num_tokens)
            hidden_states = get_attn_tp_group().all_gather(hidden_states, dim=0, output_split_sizes=attn_sp_token_nums)
        else:
            q_lora = self.q_a_proj(hidden_states)[0]
            if gather_qkv:
                hidden_states = get_attn_tp_group().gather_tensor(hidden_states, forward_batch.global_sp_num_tokens)
                q_lora = get_attn_tp_group().gather_tensor(q_lora, forward_batch.global_sp_num_tokens)
        q_lora = self.q_a_layernorm(q_lora)
        if not forward_batch.all_decode_or_idle:
            attn_metadata = forward_batch.attn_metadata
            num_prefill_tokens = attn_metadata.num_prefill_tokens
            num_decode_tokens = attn_metadata.num_decode_tokens
            no_padding_input_len = num_prefill_tokens + num_decode_tokens
            hidden_states = hidden_states[:no_padding_input_len]
            q_lora = q_lora[:no_padding_input_len]

        topk_indices = self.indexer(
            x=hidden_states,
            q_lora=q_lora,
            positions=positions,
            forward_batch=forward_batch,
            layer_id=layer_id,  # Use stored layer_id if available
        )

        return topk_indices

    def process_weights_after_loading(self, not_used: torch.nn.Module):

        def get_layer_weight(layer):
            WEIGHT_NAMES = ("weight", "qweight", "weight_packed")
            for attr in WEIGHT_NAMES:
                if hasattr(layer, attr):
                    return getattr(layer, attr)
            raise AttributeError(
                f"Layer '{layer}' has no recognized weight attribute:"
                f" {WEIGHT_NAMES}.")

        def get_and_maybe_dequant_weights(layer: LinearBase):
            if not isinstance(layer.quant_method, UnquantizedLinearMethod):
                # NOTE: This should only be used offline, since it's O(N^3)
                eye = torch.eye(layer.input_size_per_partition,
                                dtype=act_dtype,
                                device=get_layer_weight(layer).device)
                dequant_weights = layer.quant_method.apply(layer,
                                                           eye,
                                                           bias=None)
                del eye
                # standardize to (output, input)
                return dequant_weights.T
            return layer.weight

        # we currently do not have quantized bmm's which are needed for
        # `W_UV` and `W_UK_T`, we we just store fp16/bf16 copies and perform
        # the bmm's in 16-bit, the extra memory overhead of this is fairly low
        num_heads = self.num_local_heads
        if self.npu_disable_dsa_head_parallel:
            num_heads = self.num_heads
        kv_b_proj_weight = get_and_maybe_dequant_weights(self.kv_b_proj).T
        assert kv_b_proj_weight.shape == (
            self.kv_lora_rank,
            num_heads * (self.qk_nope_head_dim + self.v_head_dim)), (
                f"{kv_b_proj_weight.shape=}, "
                f"{self.kv_lora_rank=}, "
                f"{num_heads=}, "
                f"{self.qk_nope_head_dim=}, "
                f"{self.v_head_dim=}")
        kv_b_proj_weight = kv_b_proj_weight.view(
            self.kv_lora_rank,
            num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )

        W_UK, W_UV = kv_b_proj_weight.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Convert from (L, N, V) to (N, L, V)
        W_UV = W_UV.transpose(0, 1).contiguous()
        # Convert from (L, N, P) to (N, P, L)
        W_UK_T = W_UK.permute(1, 2, 0).contiguous()

        # tensor here need to be capsulated adapting for graph mode
        # Capsulated as parameter may cause additional weight update in p2p
        if not hasattr(self.kv_b_proj, 'W_UV'):
            self.kv_b_proj.register_parameter("W_UV", torch.nn.Parameter(W_UV))
        else:
            self.kv_b_proj.W_UV.data = W_UV
        if not hasattr(self.kv_b_proj, 'W_UK_T'):
            self.kv_b_proj.register_parameter("W_UK_T", torch.nn.Parameter(W_UK_T))
        else:
            self.kv_b_proj.W_UK_T.data = W_UK_T
        self.kv_a_layernorm_absorb_scale = copy.deepcopy(self.kv_a_layernorm)
        self.kv_a_layernorm_absorb_scale.weight.data = self.kv_a_layernorm.weight.data * self.mla_scale_kv_lora

    def prefetch_full(self, dependency):
        torch_npu.npu_prefetch(self.kv_a_proj_with_mqa.weight, dependency,
                               self.kv_a_proj_with_mqa.weight.numel() * self.kv_a_proj_with_mqa.weight.element_size())
        torch_npu.npu_prefetch(self.q_a_proj.weight, dependency,
                               self.q_a_proj.weight.numel() * self.q_a_proj.weight.element_size())
        torch_npu.npu_prefetch(self.q_b_proj.weight, dependency,
                               self.q_b_proj.weight.numel() * self.q_b_proj.weight.element_size())

    def prefetch_half(self, dependency):
        torch_npu.npu_prefetch(self.q_a_proj.weight, dependency,
                               self.q_a_proj.weight.numel() * self.q_a_proj.weight.element_size())
        torch_npu.npu_prefetch(self.q_b_proj.weight, dependency,
                               self.q_b_proj.weight.numel() * self.q_b_proj.weight.element_size() // 2)
