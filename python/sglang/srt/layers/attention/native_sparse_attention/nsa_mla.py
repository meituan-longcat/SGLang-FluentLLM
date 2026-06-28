import math
from typing import Optional, List

import torch
from torch import nn
# from torch.profiler import record_function
from transformers import PretrainedConfig

from sglang.srt.layers.attention.native_sparse_attention.compress_attn_v1 import (
    get_compressed_buffer_loc,
    get_compressed_seqlens,
)

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import add_prefix
from sglang.srt.models.qwen3_nsa import (
    CompressAttn, SelectiveAttn, Compress,
    AttentionGateFusion,
    load_cu_seqlens, load_kv_cache
)
from sglang.srt.layers.attention.native_sparse_attention.compress_attn import (
    _compress_attention_with_score_torch_aligned,
    _transform_score_torch_aligned,
    _fill_topk_idx_torch_aligned,
)
from sglang.srt.layers.attention.native_sparse_attention.select_attn import (
    _select_attention_torch_aligned
)
from flash_attn_3.flash_attn_interface import flash_attn_varlen_func, flash_attn_with_kvcache

class CompressAttnMLA(nn.Module):
    def __init__(
        self,
        num_q_heads: int,
        qk_head_dim: int,
        v_head_dim: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        kernel_size: int,
        stride: int,
        select_size: int,
        top_n: int,
        slc_att_num_init_blocks: int,
        slc_att_num_local_blocks: int,
        scaling: float,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        kv_b_proj: nn.Module = None,
        num_slc_score_heads: int=1,
        virtual_k_group_agg_type: str="max",
        cmp_gate_rescale: bool = False,
        prefix: str = "",
    ):
        super().__init__()

        self.mqa = CompressAttn(
            num_q_heads=num_q_heads,
            num_kv_heads=1,
            qk_head_dim=kv_lora_rank+qk_rope_head_dim,
            v_head_dim=kv_lora_rank,
            kernel_size=kernel_size,
            stride=stride,
            select_size=select_size,
            top_n=top_n,
            slc_att_num_init_blocks=slc_att_num_init_blocks,
            slc_att_num_local_blocks=slc_att_num_local_blocks,
            num_slc_score_heads=num_slc_score_heads,
            virtual_k_group_agg_type=virtual_k_group_agg_type,
            scaling=scaling,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=prefix
        )
        del self.mqa.compress_key
        del self.mqa.compress_value
        
        self.num_q_heads = num_q_heads
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.select_size = select_size
        self.top_k = top_n
        self.slc_att_num_init_blocks = slc_att_num_init_blocks
        self.slc_att_num_local_blocks = slc_att_num_local_blocks
        self.sm_scale = scaling
        self.layer_id = layer_id
        self.kv_b_proj = kv_b_proj
        self.num_slc_score_heads=num_slc_score_heads
        self.virtual_k_group_agg_type = virtual_k_group_agg_type
        self.cmp_gate_rescale = cmp_gate_rescale
        self.compress_kv = Compress(
            head_dim=kv_lora_rank,
            kernel_size=kernel_size,
            stride=stride,
            quant_config=quant_config,
            cmp_gate_rescale=cmp_gate_rescale,
            prefix=add_prefix("compress_kv", prefix),
        )
        self.compress_k_pe = Compress(
            head_dim=qk_rope_head_dim,
            kernel_size=kernel_size,
            stride=stride,
            quant_config=quant_config,
            cmp_gate_rescale=cmp_gate_rescale,
            prefix=add_prefix("compress_k_pe", prefix),
        )
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_batch: ForwardBatch,
        qv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        key_buffer, _ = forward_batch.token_to_kv_pool.get_kv_buffer(self.layer_id)
        compressed_key_buffer, _ = forward_batch.token_to_kv_pool.get_compressed_kv_buffer(self.layer_id)

        self.compress_kv(
            forward_batch=forward_batch,
            buffer=key_buffer[..., :self.kv_lora_rank],
            compressed_buffer=compressed_key_buffer[..., :self.kv_lora_rank],
        )
        self.compress_k_pe(
            forward_batch=forward_batch,
            buffer=key_buffer[..., self.kv_lora_rank:],
            compressed_buffer=compressed_key_buffer[..., self.kv_lora_rank:],
        )

        o, topk_idx = self._compress_attention(
            q=q,
            k=k,
            v=v,
            forward_batch=forward_batch,
            qv=qv,
        )

        return o, topk_idx

    def _compress_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_batch: ForwardBatch,
        qv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if qv is None:
            o, topk_idx = self._compress_attention_normal(
                q=q,
                uncompressed_k=k,
                uncompressed_v=v,
                forward_batch=forward_batch,
            )
        else:
            q_pe = q
            q_nope_out = qv
            o, topk_idx = self.mqa.compress_attention(
                q=torch.cat([q_nope_out, q_pe], dim=-1),
                forward_batch=forward_batch,
            )

        return o, topk_idx

    def _compress_attention_normal(
        self,
        q: torch.Tensor,
        uncompressed_k: torch.Tensor,
        uncompressed_v: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        # Need to first up-dimension the compress kv pool stuff before doing attn
        seq_lens_kv = forward_batch.seq_lens
        if forward_batch.forward_mode.is_decode():
            seqlens_q = seq_lens_kv.new_ones(seq_lens_kv.shape)
        else:
            seqlens_q = forward_batch.extend_seq_lens
        seq_lens_block = get_compressed_seqlens(
            seqlens=seq_lens_kv,
            kernel_size=self.kernel_size,
            kernel_stride=self.stride,
        )
        seq_lens_p = seqlens_q * seq_lens_block
        cu_seq_lens_q = torch.cat([seqlens_q.new_zeros(1), torch.cumsum(seqlens_q, dim=0)])
        cu_seq_lens_kv = torch.cat([seq_lens_kv.new_zeros(1), torch.cumsum(seq_lens_kv, dim=0)])
        cu_seq_lens_block = torch.cat([seq_lens_block.new_zeros(1), torch.cumsum(seq_lens_block, dim=0)])
        cu_seq_lens_p = torch.cat([seq_lens_p.new_zeros(1), torch.cumsum(seq_lens_p, dim=0)])

        compressed_kv_buffer_loc = get_compressed_buffer_loc(
            seq_lens=seq_lens_kv,
            req_pool_indices=forward_batch.req_pool_indices,
            req_to_token=forward_batch.req_to_token_pool.req_to_token,
            kernel_size=self.kernel_size,
            kernel_stride=self.stride,
        )
        compressed_key_buffer, _ = forward_batch.token_to_kv_pool.get_compressed_kv_buffer(self.layer_id)

        k, v = load_kv_cache(forward_batch, self.layer_id)

        q = q.view(-1, self.num_q_heads, self.qk_head_dim)
        k = k.view(-1, 1, self.qk_head_dim)
        v = v.view(-1, 1, self.v_head_dim)

        total_q_len = cu_seq_lens_q[-1]
        total_scores_len = cu_seq_lens_p[-1]

        o = q.new_empty(total_q_len, self.num_q_heads, self.v_head_dim)
        p = q.new_empty(self.num_q_heads, total_scores_len)
        topk_idx = q.new_zeros(total_q_len, self.num_slc_score_heads, self.top_k)

        for q_start, q_end, kv_start, kv_end, block_start, block_end, p_start, p_end in zip(
            cu_seq_lens_q[:-1], cu_seq_lens_q[1:],
            cu_seq_lens_kv[:-1], cu_seq_lens_kv[1:],
            cu_seq_lens_block[:-1], cu_seq_lens_block[1:],
            cu_seq_lens_p[:-1], cu_seq_lens_p[1:],
        ):
            per_seq_k = k[kv_start:kv_end]
            per_seq_v = v[kv_start:kv_end]
            kv_len = kv_end - kv_start
            kv_len = kv_len.item()
            q_len = q_end - q_start
            p_len = p_end - p_start
            if q_len == 1:
                per_seq_q = q[q_start:q_end]
            else:
                per_seq_q = q.new_zeros((kv_len, self.num_q_heads, self.qk_head_dim))
                per_seq_q[-q_len:] = q[q_start:q_end]
            per_seq_compressed_kv_buffer_loc = compressed_kv_buffer_loc[block_start:block_end]
            # [com_block_num, nh, hd]
            cached_kv = compressed_key_buffer[per_seq_compressed_kv_buffer_loc]
            # [..., 576] -> [..., nh, 128], [..., 64]
            cached_cmp_k_nope, cached_cmp_k_pe, cached_cmp_v =\
                self.cached_lora_kv_to_kv(cached_kv)
            
            com_block_num = per_seq_compressed_kv_buffer_loc.shape[0]
            cached_cmp_k = q.new_empty(com_block_num, self.num_q_heads, self.qk_head_dim)
            cached_cmp_k[..., :self.qk_nope_head_dim] = cached_cmp_k_nope
            cached_cmp_k[..., self.qk_nope_head_dim:] = cached_cmp_k_pe
            # compute mha
            per_seq_o, per_seq_p = _compress_attention_with_score_torch_aligned(
                per_seq_q.unsqueeze(0),
                cached_cmp_k.unsqueeze(0),
                cached_cmp_v.unsqueeze(0),
                kv_len,
                self.kernel_size,
                self.stride,
                self.sm_scale
            )
            o[q_start:q_end] = per_seq_o.squeeze(0)[-q_len:]
            p[:,p_start:p_end]= per_seq_p.squeeze(0)[:,-q_len:,:].view(self.num_q_heads, p_len)

            per_seq_select_score = _transform_score_torch_aligned(
                compress_score=per_seq_p,
                kv_len=kv_len,
                kernel_size=self.kernel_size,
                stride=self.stride,
                select_size=self.select_size,
                top_k=self.top_k,
                slc_att_num_init_blocks=self.slc_att_num_init_blocks,
                slc_att_num_local_blocks=self.slc_att_num_local_blocks,
                num_slc_score_heads=self.num_slc_score_heads,
                virtual_k_group_agg_type=self.virtual_k_group_agg_type
            )
            per_seq_topk_idx = _fill_topk_idx_torch_aligned(
                select_probs=per_seq_select_score,
                kv_len=kv_len,
                select_size=self.select_size,
                top_k=self.top_k,
            )
            topk_idx[q_start:q_end,:, :per_seq_topk_idx.shape[-1]] = per_seq_topk_idx.squeeze(0).transpose(0,1)[-q_len:,:,:]

        o = o.view(-1, self.num_q_heads * self.v_head_dim)
        return o, topk_idx
    
    def cached_lora_kv_to_kv(self, lora_kv: torch.Tensor):
        kv_a, k_pe = lora_kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        # k_b_proj, v_b_proj = self.kv_b_proj
        # k_nope = k_b_proj(kv_a)[0]
        # v = v_b_proj(kv_a)[0]
        # k_nope = k_nope.view(-1, self.num_q_heads, self.qk_nope_head_dim)
        # v = v.view(-1, self.num_q_heads, self.v_head_dim)
        kv = self.kv_b_proj(kv_a)[0]
        kv = kv.view(-1, self.num_q_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope = kv[..., : self.qk_nope_head_dim]
        v = kv[..., self.qk_nope_head_dim :]
        return k_nope, k_pe, v


class SelectiveAttnMLA(nn.Module):
    def __init__(
        self,
        num_q_heads: int,
        num_slc_score_heads: int,
        qk_head_dim: int,
        v_head_dim: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        select_size: int,
        scaling: float,
        layer_id: int,
    ) -> None:
        super().__init__()

        self.mqa = SelectiveAttn(
            num_q_heads=num_q_heads,
            num_kv_heads=1,
            qk_head_dim=kv_lora_rank+qk_rope_head_dim,
            v_head_dim=kv_lora_rank,
            select_size=select_size,
            scaling=scaling,
            layer_id=layer_id,
        )

        self.num_q_heads = num_q_heads
        self.qk_head_dim = qk_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.select_size = select_size
        self.sm_scale = scaling
        self.num_slc_score_heads=num_slc_score_heads

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        selected_indices: torch.Tensor,
        forward_batch: ForwardBatch,
        qv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        if qv is None:
            o = self.forward_normal(
                q=q,
                k=k,
                v=v,
                selected_indices=selected_indices,
                forward_batch=forward_batch,
            )
        else:
            q_pe = q
            q_nope_out = qv
            o = self.mqa(
                q=torch.cat([q_nope_out, q_pe], dim=-1),
                select_indices=selected_indices,
                forward_batch=forward_batch,
            )

        return o
    
    def forward_normal(
        self,
        q: torch.Tensor,
        k,
        v,
        selected_indices, # [qlen, score_head, topk]
        forward_batch: ForwardBatch,
    ):
        """
        Don't take kv cache, directly use the up-dimensioned kv to compute
        """
        # top_indices = top_indices.transpose(0, 1)
        cu_seqlens_q, cu_seqlens_kv = load_cu_seqlens(forward_batch)
        total_q_len = cu_seqlens_q[-1]

        q = q.view(-1, self.num_q_heads, self.qk_nope_head_dim + self.qk_rope_head_dim)
        o = v.new_zeros(total_q_len, self.num_q_heads, self.v_head_dim)

        for q_start, q_end, kv_start, kv_end in zip(
            cu_seqlens_q[:-1], cu_seqlens_q[1:], cu_seqlens_kv[:-1], cu_seqlens_kv[1:],
        ):
            per_seq_k = k[kv_start:kv_end]
            per_seq_v = v[kv_start:kv_end]
            kv_len = kv_end - kv_start
            per_seq_q = q[q_start:q_end]
            per_seq_topk_idx = selected_indices[q_start:q_end]
            num_selcct_blocks = math.ceil(kv_len / self.select_size)
            per_seq_topk_idx = per_seq_topk_idx[:, :, :num_selcct_blocks]
            per_seq_o = _select_attention_torch_aligned(
                per_seq_q[None], # [1, qlen, qh, qknope+qkrope]
                per_seq_k[None],
                per_seq_v[None],
                per_seq_topk_idx[None],
                sm_scale=self.sm_scale,
                select_size=self.select_size,
                num_slc_score_heads=self.num_slc_score_heads,
            )
            o[q_start:q_end] = per_seq_o.squeeze(0)

        o = o.view(-1, self.num_q_heads * self.v_head_dim)
        return o

class SlidingWindowAttnMLA(nn.Module):
    def __init__(
        self,
        num_q_heads: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        sliding_window_size: int,
        scaling: float,
        layer_id: int,
    ) -> None:
        super().__init__()

        self.num_q_heads = num_q_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.sliding_window_size = sliding_window_size
        self.scaling=scaling
        self.layer_id = layer_id

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_batch: ForwardBatch,
        qv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if qv is None:
            o = self.forward_normal(
                q=q,
                k=k,
                v=v,
                forward_batch=forward_batch,
            )
        else:
            o = self.forward_absorb(
                q=q,
                forward_batch=forward_batch,
                qv=qv,
            )

        return o
        
    def forward_absorb(
        self,
        q: torch.Tensor,
        forward_batch: ForwardBatch,
        qv: torch.Tensor,
    ):
        seq_num, = forward_batch.seq_lens.shape

        cache_seqlens = forward_batch.seq_lens.to(torch.int32)
        cache_batch_idx = forward_batch.req_pool_indices.to(torch.int32)

        if forward_batch.forward_mode.is_decode():
            causal = False
            max_seqlen_q=1
            cu_seqlens_q = torch.arange(0, seq_num + 1, dtype=torch.int32, device=cache_seqlens.device)
        else:
            causal = True
            max_seqlen_q = max(forward_batch.extend_seq_lens_cpu)
            cu_seqlens_q = torch.nn.functional.pad(
                torch.cumsum(forward_batch.extend_seq_lens, dim=0, dtype=torch.int32), (1, 0)
            )

        # reference: https://github.com/Dao-AILab/flash-attention/issues/1471
        o = flash_attn_with_kvcache(
            q=q,
            k_cache=forward_batch.token_to_kv_pool.get_key_buffer(self.layer_id)[..., self.kv_lora_rank:].unsqueeze(1),
            v_cache=forward_batch.token_to_kv_pool.get_value_buffer(self.layer_id).unsqueeze(1),
            qv=qv,
            page_table=forward_batch.req_to_token_pool.req_to_token,
            cache_seqlens=cache_seqlens,
            cache_batch_idx=cache_batch_idx,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            softmax_scale=self.scaling,
            causal=causal,
            window_size=(self.sliding_window_size, 0),
        )

        o = o.view(-1, self.num_q_heads * self.kv_lora_rank)
        return o
    
    def forward_normal(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        """
        Don't take kv cache, directly use the up-dimensioned kv to compute
        """
        assert not forward_batch.forward_mode.is_decode()
        assert sum(forward_batch.extend_prefix_lens_cpu) == 0

        cu_seqlens = torch.nn.functional.pad(
            torch.cumsum(forward_batch.seq_lens, dim=0, dtype=torch.int32), (1, 0)
        )
        max_seqlen = forward_batch.seq_lens.max().item()

        o = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=self.scaling,
            causal=True,
            window_size=(self.sliding_window_size, 0),
        )

        o = o.view(-1, self.num_q_heads * self.v_head_dim)
        return o


class NativeSparseAttentionMLA(nn.Module):
    def __init__(
        self,
        num_q_heads: int,
        num_score_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        kv_lora_rank: int,
        scaling: float,
        sliding_window_size: int,
        kernel_size: int,
        stride: int,
        select_size: int,
        top_n: int,
        slc_att_num_init_blocks: int,
        slc_att_num_local_blocks: int,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        kv_b_proj: nn.Module = None,
        num_slc_score_heads: int=1,
        virtual_k_group_agg_type: str="max",
        nsa_gate_mask: List = [1,1,1],
        nsa_gate_type: str = "softmax",
        gate_weight_head_not_share: bool = False,
        config: PretrainedConfig = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        assert select_size >= kernel_size
        assert kernel_size % stride == 0
        assert select_size % stride == 0
        assert math.log2(stride).is_integer()
        # q_heads is also local_heads
        self.num_q_heads = num_q_heads
        self.num_score_heads = num_score_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.v_head_dim = v_head_dim
        self.scaling = scaling
        self.layer_id = layer_id
        self.kv_b_proj = kv_b_proj
        self.nsa_gate_mask = nsa_gate_mask
        self.config = config
        self.cmp_gate_rescale = getattr(config, "cmp_gate_rescale", False)
        self.fusion_gate_rescale = getattr(config, "fusion_gate_rescale", False)
        qk_head_dim = qk_nope_head_dim+qk_rope_head_dim

        self.compress_attn = CompressAttnMLA(
            num_q_heads=num_q_heads,
            qk_head_dim=qk_head_dim,
            v_head_dim=v_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            kernel_size=kernel_size,
            stride=stride,
            select_size=select_size,
            top_n=top_n,
            slc_att_num_init_blocks=slc_att_num_init_blocks,
            slc_att_num_local_blocks=slc_att_num_local_blocks,
            scaling=scaling,
            layer_id=layer_id,
            quant_config=quant_config,
            kv_b_proj=self.kv_b_proj,
            num_slc_score_heads=num_slc_score_heads,
            virtual_k_group_agg_type=virtual_k_group_agg_type,
            cmp_gate_rescale=self.cmp_gate_rescale,
            prefix=add_prefix("compress_attn", prefix),
        )

        self.select_attn = SelectiveAttnMLA(
            num_q_heads=num_q_heads,
            num_slc_score_heads=num_slc_score_heads,
            qk_head_dim=qk_head_dim,
            v_head_dim=v_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            select_size=select_size,
            scaling=scaling,
            layer_id=layer_id,
        )

        self.window_attn = SlidingWindowAttnMLA(
            num_q_heads=num_q_heads,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            sliding_window_size=sliding_window_size,
            scaling=scaling,
            layer_id=layer_id,
        )

        self.gate_fusion = AttentionGateFusion(
            num_q_heads=num_q_heads,
            v_head_dim=v_head_dim,
            quant_config=quant_config,
            nsa_gate_mask=nsa_gate_mask,
            nsa_gate_type=nsa_gate_type,
            gate_weight_head_not_share=gate_weight_head_not_share,
            fusion_gate_rescale=self.fusion_gate_rescale,
            prefix=add_prefix("gate_fusion", prefix),
        )

        self.w_kc = None
        self.w_vc = None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        qv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Native Sparse Attention MLA forward pass with dual execution modes.
        
        === Execution Mode Details ===
        
        1. **Standard MHA Mode** (qv=None):
           Input tensor shapes:
           • q: [total_q_len, num_q_heads, qk_nope_head_dim + qk_rope_head_dim]
             Complete query tensor containing both content (nope) and position (rope) components
             
           • k: [total_kv_len, num_q_heads, qk_nope_head_dim + qk_rope_head_dim] 
             Key tensor for eache query heads, contains both content and position components
             
           • v: [total_kv_len, num_q_heads, v_head_dim]
             Value tensors for each query heads, full dimensionality after LoRA projection
        
        2. **Absorb MQA Mode** (qv!=None):
           Input tensor shapes:
           • q: [total_q_len, num_q_heads, qk_rope_head_dim] 
             Position-only query component (q_pe)
             
           • k: [total_kv_len, 1, qk_rope_head_dim]
             Position-only key component (k_pe), shared across query heads
             
           • v: [total_kv_len, 1, kv_lora_rank]
             LoRA-compressed value tensor (lora_kv), shared across query heads
             
           • qv: [total_q_len, num_q_heads, kv_lora_rank]
             Same as q_nope_out, pre-computed content for absorb mode
        
        Args:
            q, k, v, qv: Input tensors with mode-dependent shapes as described above
            forward_batch: Batch processing context containing memory pools and sequence metadata
            save_kv_cache: Whether to cache key-value pairs for subsequent attention steps
            
        Returns:
            torch.Tensor: Attention output [total_q_len, num_q_heads * v_head_dim]
        """

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                self, forward_batch.out_cache_loc, torch.cat([v, k], dim=-1), None
            )

        o_list = []
        cmp_o = select_o = window_o = None
        if self.nsa_gate_mask[0]:
            cmp_o, topk_idx = self.compress_attn(
                q=q,
                k=k,
                v=v,
                forward_batch=forward_batch,
                qv=qv,
            )
        o_list.append(cmp_o)
        if self.nsa_gate_mask[1]:
            select_o = self.select_attn(
                q=q,
                k=k,
                v=v,
                selected_indices=topk_idx,
                forward_batch=forward_batch,
                qv=qv,
            )
        o_list.append(select_o)
        if self.nsa_gate_mask[2]:
            window_o = self.window_attn(
                q=q,
                k=k,
                v=v,
                forward_batch=forward_batch,
                qv=qv,
            )
        o_list.append(window_o)
        num_q_tokens = q.shape[0]
        o_list = list(map(
            lambda e: None if e is None else \
                e.view(num_q_tokens, self.num_q_heads, -1),
            o_list
        ))
        if qv is not None:
            o_list = list(map(
                # [qh, b, 512] @ [qh, 512, 128] -> [qh, b, 128] -> [b, qh, 128]
                lambda e: None if e is None else \
                    torch.bmm(e.transpose(0, 1), self.w_vc).transpose(0, 1),
                o_list
            ))
        o = self.gate_fusion(*o_list)
        return o
