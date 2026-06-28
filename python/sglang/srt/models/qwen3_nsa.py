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

# Adapted from llama2.py
# Modify details for the adaptation of Qwen2 model.
"""Inference-only Qwen2 model compatible with HuggingFace weights."""
import math
from typing import Any, Dict, Iterable, Optional, Tuple, List

import torch
from torch import nn
# from torch.profiler import record_function
from transformers import PretrainedConfig

from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferAttnBackend,
    WrapperDispatch,
    create_flashinfer_kv_indices_triton,
)
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.attention.native_sparse_attention.select_attn import (
    select_attention_torch,
    select_attention_decode_splitk_triton
)
from sglang.srt.layers.attention.native_sparse_attention.compress_attn import (
    compress_attention_torch,
    compress_attention_decode_triton,
)
from sglang.srt.layers.attention.native_sparse_attention.compress_kv import (
    cumsum_with_padding,
    get_compressed_kv_indptr,
    get_compressed_kv_indices,
    gate_compress_torch,
    gate_compress_decode_triton,
)
from flash_attn_3.flash_attn_interface import flash_attn_with_kvcache

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear, RowParallelLinear
from sglang.srt.layers.dp_attention import get_attention_tp_size, get_attention_tp_rank
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.mem_cache.memory_pool import NativeSparseMHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen2 import Qwen2MLP as Qwen3MLP
from sglang.srt.models.qwen2 import Qwen2Model
from sglang.srt.models.qwen3 import Qwen3Attention
from sglang.srt.utils import add_prefix


Qwen3NsaConfig = None
layer_id = -1
max_bs = -1
# max_total_num_tokens = 121024 # for 1*h800 v2lite
max_total_num_tokens = -1
max_context_len = 32768
# tensor for cuda graph
compress_kv_indices = \
flat_com_score = flat_slc_score = \
cuda_graph_kv_indices = None

def load_cu_seqlens(
    forward_batch: ForwardBatch,
):
    seqlens_kv = forward_batch.seq_lens
    if forward_batch.forward_mode.is_decode():
        seqlens_q = seqlens_kv.new_ones(seqlens_kv.shape)
    else:
        seqlens_q = forward_batch.extend_seq_lens

    cu_seqlens_q = seqlens_q.new_zeros(seqlens_q.shape[0] + 1)
    cu_seqlens_kv = seqlens_kv.new_zeros(seqlens_kv.shape[0] + 1)
    cu_seqlens_q[1:] = torch.cumsum(seqlens_q, dim=0)
    cu_seqlens_kv[1:] = torch.cumsum(seqlens_kv, dim=0)
    return cu_seqlens_q, cu_seqlens_kv

def load_kv_cache(
    forward_batch: ForwardBatch,
    layer_id: int
):
    cached_kv_idx = torch.cat([
        forward_batch.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices[seq_idx],
            :kv_len,
        ]
        for seq_idx, kv_len in enumerate(forward_batch.seq_lens)
    ])

    cached_k = forward_batch.token_to_kv_pool.get_key_buffer(layer_id)[cached_kv_idx]
    cached_v = forward_batch.token_to_kv_pool.get_value_buffer(layer_id)[cached_kv_idx]

    return cached_k, cached_v

# ref: https://github.com/mdy666/Qwen-Native-Sparse-Attention/tree/master
class Compress(torch.nn.Module):
    def __init__(
        self,
        head_dim: int,
        kernel_size: int,
        stride: int,
        quant_config: Optional[QuantizationConfig] = None,
        cmp_gate_rescale:bool = False,
        prefix: str = "",
    ):
        super().__init__()
        self.head_dim = head_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.gate_proj = ReplicatedLinear(
            self.kernel_size * self.head_dim,
            self.kernel_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_proj", prefix),
        )
        self.cmp_gate_rescale = cmp_gate_rescale
        self.cmp_rescale_factor = 0
        if cmp_gate_rescale:
            self.cmp_rescale_factor = (kernel_size * head_dim) ** -0.5

    def forward(
            self,
            forward_batch: ForwardBatch,
            buffer: torch.Tensor,
            compressed_buffer: torch.Tensor,
        ):
        """
        Pepare the compressed blocks from buffer into compressed buffer.

        Args:
            forward_batch (ForwardBatch): batch info
            buffer (torch.Tensor): [buffer_size, num_heads, head_dim]
            compressed_buffer (torch.Tensor): [compressed_buffer_size, num_heads, head_dim]
        """
        if forward_batch.forward_mode.is_decode():
            return self.forward_decode(
                forward_batch=forward_batch,
                buffer=buffer,
                compressed_buffer=compressed_buffer,
            )
        else:
            return self.forward_extend(
                forward_batch=forward_batch,
                buffer=buffer,
                compressed_buffer=compressed_buffer,
            )

    def forward_extend(
            self,
            forward_batch: ForwardBatch,
            buffer: torch.Tensor,
            compressed_buffer: torch.Tensor,
        ):
        """
        Pepare the compressed blocks from buffer into compressed buffer.

        Args:
            forward_batch (ForwardBatch): batch info
            buffer (torch.Tensor): [buffer_size, num_heads, head_dim]
            compressed_buffer (torch.Tensor): [compressed_buffer_size, num_heads, head_dim]
        """
        kv_lens = forward_batch.seq_lens
        kv_indptr = cumsum_with_padding(kv_lens)
        req_to_token = forward_batch.req_to_token_pool.req_to_token
        kv_indices = req_to_token.new_zeros(kv_lens.sum().item())
        bs = forward_batch.batch_size

        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token_ptr = req_to_token,
            req_pool_indices_ptr = forward_batch.req_pool_indices,
            page_kernel_lens_ptr  = forward_batch.seq_lens,
            kv_indptr = kv_indptr,
            kv_start_idx = None,
            kv_indices_ptr = kv_indices,
            req_to_token_ptr_stride = req_to_token.stride(0),
        )

        compressed_kv_indptr = get_compressed_kv_indptr(
            kv_indptr=kv_indptr,
            kernel_size=self.kernel_size,
            stride=self.stride,
        )
        compressed_kv_indices = get_compressed_kv_indices(
            kv_indptr=kv_indptr,
            compressed_kv_indptr=compressed_kv_indptr,
            kv_indices=kv_indices,
            stride=self.stride,
        )

        gate_compress_torch(
            kv_buffer=buffer,
            compressed_kv_buffer=compressed_buffer,
            gate_proj=self.gate_proj,
            kv_indptr=kv_indptr,
            compressed_kv_indptr=compressed_kv_indptr,
            kv_indices=kv_indices,
            compressed_kv_indices=compressed_kv_indices,
            kernel_size=self.kernel_size,
            stride=self.stride,
            cmp_rescale_factor=self.cmp_rescale_factor,
        )

    def forward_decode(
            self,
            forward_batch: ForwardBatch,
            buffer: torch.Tensor,
            compressed_buffer: torch.Tensor,
        ):
        """
        Pepare the compressed blocks from buffer into compressed buffer.

        Args:
            forward_batch (ForwardBatch): batch info
            buffer (torch.Tensor): [buffer_size, num_heads, head_dim]
            compressed_buffer (torch.Tensor): [compressed_buffer_size, num_heads, head_dim]
        """
        gate_compress_decode_triton(
            kv_cache=buffer,
            compressed_kv_cache=compressed_buffer,
            gate_weight=self.gate_proj.weight,
            kv_lens =forward_batch.seq_lens,
            req_pool_indices=forward_batch.req_pool_indices,
            req_to_token=forward_batch.req_to_token_pool.req_to_token,
            kernel_size=self.kernel_size,
            stride=self.stride,
            cmp_rescale_factor=self.cmp_rescale_factor,
        )


class CompressAttn(torch.nn.Module):
    def __init__(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        qk_head_dim: int,
        v_head_dim: int,
        kernel_size: int,
        stride: int,
        select_size: int,
        top_n: int,
        slc_att_num_init_blocks: int,
        slc_att_num_local_blocks: int,
        num_slc_score_heads: int,
        virtual_k_group_agg_type: str,
        scaling: float,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.select_size = select_size
        self.top_k = top_n
        self.slc_att_num_init_blocks = slc_att_num_init_blocks
        self.slc_att_num_local_blocks = slc_att_num_local_blocks
        self.num_slc_score_heads = num_slc_score_heads
        self.virtual_k_group_agg_type = virtual_k_group_agg_type
        self.sm_scale = scaling
        self.layer_id = layer_id

        self.compress_key = Compress(
            head_dim=qk_head_dim,
            kernel_size=kernel_size,
            stride=stride,
            quant_config=quant_config,
            prefix=add_prefix("compress_key", prefix),
        )
        self.compress_value = Compress(
            head_dim=v_head_dim,
            kernel_size=kernel_size,
            stride=stride,
            quant_config=quant_config,
            prefix=add_prefix("compress_value", prefix),
        )

    def forward(
            self,
            q: torch.Tensor,
            forward_batch: ForwardBatch,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the Compress Attention module.

        Args:
            q (torch.Tensor): [total_q_len, num_q_heads * qk_head_dim]
            forward_batch (ForwardBatch): batch info
        Returns:
            o (torch.Tensor): [total_q_len, num_q_heads * v_head_dim]
            topk_idx (torch.Tensor): [total_q_len, num_slc_score_heads, top_k]
        """
        self.compress_key(
            forward_batch=forward_batch,
            buffer=forward_batch.token_to_kv_pool.get_key_buffer(self.layer_id),
            compressed_buffer=forward_batch.token_to_kv_pool.get_compressed_key_buffer(self.layer_id)
        )
        self.compress_value(
            forward_batch=forward_batch,
            buffer=forward_batch.token_to_kv_pool.get_value_buffer(self.layer_id),
            compressed_buffer=forward_batch.token_to_kv_pool.get_compressed_value_buffer(self.layer_id)
        )

        o, topk_idx = self.compress_attention(
            q=q,
            forward_batch=forward_batch,
        )

        return o, topk_idx

    def compress_attention(
            self,
            q: torch.Tensor,
            forward_batch: ForwardBatch,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the Compress Attention module.

        Args:
            q (torch.Tensor): [total_q_len, num_q_heads, qk_head_dim]
            forward_batch (ForwardBatch): batch info
        Returns:
            o (torch.Tensor): [total_q_len, num_q_heads, v_head_dim]
            topk_idx (torch.Tensor): [total_q_len, num_slc_score_heads, top_k]
        """ 
        q = q.view(-1, self.num_q_heads, self.qk_head_dim)

        if forward_batch.forward_mode.is_decode():
            o, topk_idx = self.compress_attention_decode(
                q=q,
                forward_batch=forward_batch,
            )
        else:
            o, topk_idx = self.compress_attention_extend(
                q=q,
                forward_batch=forward_batch,
            )
        
        return o, topk_idx

    def compress_attention_extend(
            self,
            q: torch.Tensor,
            forward_batch: ForwardBatch,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the Compress Attention module.

        Args:
            q (torch.Tensor): [total_q_len, num_q_heads, qk_head_dim]
            forward_batch (ForwardBatch): batch info
        Returns:
            o (torch.Tensor): [total_q_len, num_q_heads, v_head_dim]
            topk_idx (torch.Tensor): [total_q_len, num_slc_score_heads, top_k]
        """ 
        kv_lens = forward_batch.seq_lens
        kv_indptr = cumsum_with_padding(kv_lens)
        req_to_token = forward_batch.req_to_token_pool.req_to_token
        kv_indices = req_to_token.new_zeros(kv_lens.sum().item())
        bs = forward_batch.batch_size

        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token_ptr = req_to_token,
            req_pool_indices_ptr = forward_batch.req_pool_indices,
            page_kernel_lens_ptr  = forward_batch.seq_lens,
            kv_indptr = kv_indptr,
            kv_start_idx = None,
            kv_indices_ptr = kv_indices,
            req_to_token_ptr_stride = req_to_token.stride(0),
        )

        compressed_kv_indptr = get_compressed_kv_indptr(
            kv_indptr=kv_indptr,
            kernel_size=self.kernel_size,
            stride=self.stride,
        )
        compressed_kv_indices = get_compressed_kv_indices(
            kv_indptr=kv_indptr,
            compressed_kv_indptr=compressed_kv_indptr,
            kv_indices=kv_indices,
            stride=self.stride,
        )

        if forward_batch.forward_mode.is_decode():
            q_lens = kv_lens.new_ones(kv_lens.shape)
        else:
            q_lens = forward_batch.extend_seq_lens
        q_indptr = cumsum_with_padding(q_lens)
        
        compressed_kv_lens = compressed_kv_indptr[1:] - compressed_kv_indptr[:-1]
        compress_score_lens = q_lens * compressed_kv_lens
        compress_score_indptr = cumsum_with_padding(compress_score_lens)

        o, topk_idx = compress_attention_torch(
            q=q,
            compressed_k_cache=forward_batch.token_to_kv_pool.get_compressed_key_buffer(self.layer_id),
            compressed_v_cache=forward_batch.token_to_kv_pool.get_compressed_value_buffer(self.layer_id),
            compressed_kv_indptr=compressed_kv_indptr,
            compressed_kv_indices=compressed_kv_indices,
            kv_lens=kv_lens,
            q_indptr=q_indptr,
            compress_score_indptr=compress_score_indptr,
            sm_scale=self.sm_scale,
            kernel_size=self.kernel_size,
            stride=self.stride,
            select_size=self.select_size,
            top_k=self.top_k,
            slc_att_num_init_blocks=self.slc_att_num_init_blocks,
            slc_att_num_local_blocks=self.slc_att_num_local_blocks,
            num_slc_score_heads=self.num_slc_score_heads,
            virtual_k_group_agg_type=self.virtual_k_group_agg_type
        )
        
        o = o.view(-1, self.num_q_heads * self.v_head_dim)
        return o, topk_idx

    def compress_attention_decode(
            self,
            q: torch.Tensor,
            forward_batch: ForwardBatch,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the Compress Attention module.

        Args:
            q (torch.Tensor): [batch_size, num_q_heads, qk_head_dim]
            forward_batch (ForwardBatch): batch info
        Returns:
            o (torch.Tensor): [batch_size, num_q_heads, v_head_dim]
            topk_idx (torch.Tensor): [batch_size, num_slc_score_heads, top_k]
        """

        kv_lens = forward_batch.seq_lens
        kv_indptr = cumsum_with_padding(kv_lens)
        req_to_token = forward_batch.req_to_token_pool.req_to_token

        compressed_kv_indptr = get_compressed_kv_indptr(
            kv_indptr=kv_indptr,
            kernel_size=self.kernel_size,
            stride=self.stride,
        )
        compressed_kv_lens = (compressed_kv_indptr[1:] - compressed_kv_indptr[:-1]).to(kv_lens.dtype)

        req_to_token = forward_batch.req_to_token_pool.req_to_token
        req_to_compressed_token = req_to_token.new_zeros((req_to_token.shape[0], req_to_token.shape[1] // self.stride))
        req_to_compressed_token[forward_batch.req_pool_indices] = req_to_token[
            forward_batch.req_pool_indices,
            ::self.stride,
        ] // self.stride # [batch_size, block_size]

        if compress_kv_indices is None:
            compressed_kv_indices = req_to_compressed_token.new_zeros(compressed_kv_indptr[-1])
        else:
            compressed_kv_indices = compress_kv_indices

        create_flashinfer_kv_indices_triton[(forward_batch.batch_size,)](
            req_to_token_ptr = req_to_compressed_token,
            req_pool_indices_ptr = forward_batch.req_pool_indices,
            page_kernel_lens_ptr  = compressed_kv_lens,
            kv_indptr = compressed_kv_indptr,
            kv_start_idx = None,
            kv_indices_ptr = compressed_kv_indices,
            req_to_token_ptr_stride = req_to_compressed_token.stride(0),
        )

        if flat_com_score is None:
            compress_score_buffer = q.new_zeros((compressed_kv_indptr[-1], self.num_q_heads), dtype=torch.float32)
        else:
            compress_score_buffer = flat_com_score

        select_lens = torch.ceil(kv_lens / self.select_size).to(kv_lens.dtype)
        select_indptr = cumsum_with_padding(select_lens)

        if flat_slc_score is None:
            local_flat_slc_score = compress_score_buffer.new_zeros(select_lens.sum().item(), self.num_q_heads)
        else:
            local_flat_slc_score = flat_slc_score

        o, topk_idx = compress_attention_decode_triton(
            q=q,
            compressed_k_cache=forward_batch.token_to_kv_pool.get_compressed_key_buffer(self.layer_id),
            compressed_v_cache=forward_batch.token_to_kv_pool.get_compressed_value_buffer(self.layer_id),
            compress_score_buffer=compress_score_buffer,
            select_score_buffer=local_flat_slc_score,
            compressed_kv_indptr=compressed_kv_indptr,
            compressed_kv_indices=compressed_kv_indices,
            req_to_compressed_token=req_to_compressed_token[forward_batch.req_pool_indices],
            select_indptr=select_indptr,
            sm_scale=self.sm_scale,
            kernel_size=self.kernel_size,
            stride=self.stride,
            select_size=self.select_size,
            top_k=self.top_k,
            slc_att_num_init_blocks=self.slc_att_num_init_blocks,
            slc_att_num_local_blocks=self.slc_att_num_local_blocks,
            num_slc_score_heads=self.num_slc_score_heads,
            virtual_k_group_agg_type=self.virtual_k_group_agg_type,
            max_context_len=max_context_len,
        )

        o = o.view(-1, self.num_q_heads * self.v_head_dim)
        return o, topk_idx


class SelectiveAttn(nn.Module):
    def __init__(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        qk_head_dim: int,
        v_head_dim: int,
        select_size: int,
        scaling: float,
        layer_id: int,
    ) -> None:
        super().__init__()
        
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.select_size = select_size
        self.sm_scale = scaling
        self.layer_id = layer_id

    def forward(
        self,
        q: torch.Tensor,
        select_indices: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """
        Forward pass for the Selective Attention module.

        Args:
            q (torch.Tensor): [total_q_len, num_q_heads * qk_head_dim]
            select_indices (torch.Tensor): [total_q_len, num_kv_heads, top_k]
            forward_batch (ForwardBatch): batch info
        Returns:
            o (torch.Tensor): [total_q_len, num_q_heads, v_head_dim]
        """
        if forward_batch.forward_mode.is_decode():
            return self.forward_decode(
                q=q,
                select_indices=select_indices,
                forward_batch=forward_batch,
            )
        else:
            return self.forward_extend(
                q=q,
                select_indices=select_indices,
                forward_batch=forward_batch,
            )

    def forward_extend(
        self,
        q: torch.Tensor,
        select_indices: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """
        Forward pass for the Selective Attention module.

        Args:
            q (torch.Tensor): [total_q_len, num_q_heads * qk_head_dim]
            select_indices (torch.Tensor): [total_q_len, num_slc_score_heads, top_k]
            forward_batch (ForwardBatch): batch info
        Returns:
            o (torch.Tensor): [total_q_len, num_q_heads, v_head_dim]
        """
        kv_lens = forward_batch.seq_lens
        kv_indptr = cumsum_with_padding(kv_lens)

        req_to_token = forward_batch.req_to_token_pool.req_to_token
        kv_indices = req_to_token.new_zeros((forward_batch.seq_lens_sum,))
        create_flashinfer_kv_indices_triton[(forward_batch.batch_size,)](
            req_to_token_ptr = req_to_token,
            req_pool_indices_ptr = forward_batch.req_pool_indices,
            page_kernel_lens_ptr  = kv_lens,
            kv_indptr = kv_indptr,
            kv_start_idx = None,
            kv_indices_ptr = kv_indices,
            req_to_token_ptr_stride = req_to_token.stride(0),
        )

        if forward_batch.forward_mode.is_decode():
            q_lens = kv_lens.new_ones(kv_lens.shape)
        else:
            q_lens = forward_batch.extend_seq_lens
        q_indptr = cumsum_with_padding(q_lens)

        q = q.view(-1, self.num_q_heads, self.qk_head_dim)

        o = select_attention_torch(
            q=q,
            k_cache=forward_batch.token_to_kv_pool.get_key_buffer(self.layer_id),
            v_cache=forward_batch.token_to_kv_pool.get_value_buffer(self.layer_id),
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            q_indptr=q_indptr,
            select_indices=select_indices,
            sm_scale=self.sm_scale,
            select_size=self.select_size,
        )

        o = o.view(-1, self.num_q_heads * self.v_head_dim).to(q.dtype)
        return o

    def forward_decode(
        self,
        q: torch.Tensor,
        select_indices: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """
        Forward pass for the Selective Attention module.

        Args:
            q (torch.Tensor): [total_q_len, num_q_heads * qk_head_dim]
            select_indices (torch.Tensor): [total_q_len, num_kv_heads, top_k]
            forward_batch (ForwardBatch): batch info
        Returns:
            o (torch.Tensor): [total_q_len, num_q_heads, v_head_dim]
        """
        bs = len(forward_batch.req_pool_indices)
        kv_indptr = cumsum_with_padding(forward_batch.seq_lens)
        if cuda_graph_kv_indices is None:
            kv_indices = torch.empty(
                forward_batch.seq_lens_sum, dtype=torch.int32, device=q.device
            )
        else:
            kv_indices = cuda_graph_kv_indices

        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token_ptr =  forward_batch.req_to_token_pool.req_to_token,
            req_pool_indices_ptr = forward_batch.req_pool_indices,
            page_kernel_lens_ptr  = forward_batch.seq_lens,
            kv_indptr = kv_indptr,
            kv_start_idx = None,
            kv_indices_ptr = kv_indices,
            req_to_token_ptr_stride = forward_batch.req_to_token_pool.req_to_token.stride(0),
        )

        o = select_attention_decode_splitk_triton(
            q=q.view(-1, self.num_q_heads, self.qk_head_dim),
            k_cache=forward_batch.token_to_kv_pool.get_key_buffer(self.layer_id),
            v_cache=forward_batch.token_to_kv_pool.get_value_buffer(self.layer_id),
            select_indices=select_indices,
            kv_indptr=kv_indptr, 
            kv_indices=kv_indices,
            sm_scale=self.sm_scale,
            select_size=self.select_size,
            num_kv_splits=8
        )
        o = o.view(-1, self.num_q_heads * self.v_head_dim)
        return o


class SlidingWindowAttn(nn.Module):
    def __init__(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        qk_head_dim: int,
        v_head_dim: int,
        sliding_window_size: int,
        scaling: float,
        layer_id: int,
    ):
        super().__init__()

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.sliding_window_size = sliding_window_size
        self.scaling = scaling
        self.layer_id = layer_id

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        
        q = q.view(-1, self.num_q_heads, self.qk_head_dim)
        k = k.view(-1, self.num_kv_heads, self.qk_head_dim)
        v = v.view(-1, self.num_kv_heads, self.v_head_dim)

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                self, forward_batch.out_cache_loc, k, v
            )

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

        o = flash_attn_with_kvcache(
            q=q,
            k_cache=forward_batch.token_to_kv_pool.get_key_buffer(self.layer_id).unsqueeze(1),
            v_cache=forward_batch.token_to_kv_pool.get_value_buffer(self.layer_id).unsqueeze(1),
            page_table=forward_batch.req_to_token_pool.req_to_token,
            cache_seqlens=cache_seqlens,
            cache_batch_idx=cache_batch_idx,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            softmax_scale=self.scaling,
            causal=causal,
            window_size=(self.sliding_window_size, 0),
        )

        o = o.view(-1, self.num_q_heads * self.v_head_dim)
        return o


class AttentionGateFusion(nn.Module):
    def __init__(
        self,
        num_q_heads: int,
        v_head_dim: int,
        quant_config: Optional[QuantizationConfig] = None,
        nsa_gate_mask: List = [1,1,1],
        nsa_gate_type: str = "softmax",
        gate_weight_head_not_share: bool = False,
        fusion_gate_rescale: bool = False,
        prefix: str = '',
    ):
        super().__init__()

        self.num_q_heads = num_q_heads
        self.v_head_dim = v_head_dim

        # cmp_o, select_o, window_o
        self.nsa_gate_mask = nsa_gate_mask
        self.nsa_gate_type = nsa_gate_type
        self.gate_num = gate_num = sum(self.nsa_gate_mask)
        # TODO: Change to based on head_not_share_att_gate_weight
        self.tp_size = get_attention_tp_size()
        self.tp_rank = get_attention_tp_rank()
        if gate_weight_head_not_share:
            self.gate_weight = RowParallelLinear(
                input_size=num_q_heads * self.tp_size,
                output_size=3 * v_head_dim * 3,
                bias=False,
                reduce_results=False,
                prefix=add_prefix("gate_weight", prefix),
                tp_rank=self.tp_rank,
                tp_size=self.tp_size
            )
            # self.gate_weight = torch.nn.Parameter(
            #     torch.empty(num_q_heads, gate_num, v_head_dim * gate_num),
            #     requires_grad=False,
            # )
        self.fusion_gate_rescale_factor = 0
        if fusion_gate_rescale:
            self.fusion_gate_rescale_factor = (v_head_dim * gate_num) ** -0.5

    def forward(
        self,
        o_com_att: torch.Tensor,
        o_slc_att: torch.Tensor,
        o_sw_att: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the Gate Fusion module.

        Args:
            o_com_att (torch.Tensor): [num_batch_tokens, num_q_heads * v_head_dim]
            o_slc_att (torch.Tensor): [num_batch_tokens, num_q_heads * v_head_dim]
            o_sw_att (torch.Tensor): [num_batch_tokens, num_q_heads * v_head_dim]
        Returns:
            o (torch.Tensor): [num_batch_tokens, num_q_heads * v_head_dim]
        """
        a = o_com_att
        b = o_slc_att
        c = o_sw_att
        # [S, H, 3, D]
        gate_input = torch.stack(
            [x for i, x in enumerate([a, b, c]) if self.nsa_gate_mask[i] == 1],
            dim=-2
        )
        # [S, H, 3, D] -> [S, H, 3*D] -> [S, H, 1, 3*D]
        gate_feature = gate_input.flatten(start_dim=-2, end_dim=-1)[:, :, None, :]
        gate_weight = self.gate_weight.weight
        gate_weight = gate_weight.transpose(0, 1).reshape(self.num_q_heads, self.gate_num, -1)
        # hack for no swa
        gate_weight = gate_weight[:, :gate_input.shape[-2], :gate_feature.shape[-1]]
        # [S, H, 1, 3*D] @ [H, 3*D, 3] -> [S,H,1,3] -> [S,H,3]
        gate_score = torch.matmul(gate_feature, gate_weight.transpose(-1, -2)).squeeze(-2)
        if self.fusion_gate_rescale_factor > 0:
            gate_score *= self.fusion_gate_rescale_factor
        if self.nsa_gate_type == 'softmax':
            gate_score = gate_score.softmax(dim=-1, dtype=torch.float32).to(gate_input.dtype)
        elif self.nsa_gate_type == 'sigmoid':
            gate_score = gate_score.sigmoid().to(gate_input.dtype)
        else:
            raise ValueError(f"nsa_gate_type {self.nsa_gate_type} not supported")
        # [S,H,3,1] * [S, H, 3, D]
        gate_output = (gate_score.unsqueeze(-1) * gate_input).sum(-2)

        gate_output = gate_output.view(-1, self.num_q_heads * self.v_head_dim)
        return gate_output


class NativeSparseAttention(nn.Module):

    def __init__(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        qk_head_dim: int,
        v_head_dim: int,
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
        prefix: str = "",
    ) -> None:
        super().__init__()

        assert select_size >= kernel_size
        assert kernel_size % stride == 0
        assert select_size % stride == 0
        assert math.log2(stride).is_integer()

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.layer_id = layer_id

        self.window_attn = RadixAttention(
            num_heads=num_q_heads,
            head_dim=qk_head_dim,
            scaling=scaling,
            num_kv_heads=num_kv_heads,
            layer_id=layer_id,
            v_head_dim=v_head_dim,
            sliding_window_size=sliding_window_size,
        )

        self.compress_attn = CompressAttn(
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            qk_head_dim=qk_head_dim,
            v_head_dim=v_head_dim,
            kernel_size=kernel_size,
            stride=stride,
            select_size=select_size,
            num_slc_score_heads=num_kv_heads,
            top_n=top_n,
            slc_att_num_init_blocks=slc_att_num_init_blocks,
            slc_att_num_local_blocks=slc_att_num_local_blocks,
            scaling=scaling,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("compress_attn", prefix),
        )

        self.select_attn = SelectiveAttn(
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            qk_head_dim=qk_head_dim,
            v_head_dim=v_head_dim,
            select_size=select_size,
            scaling=scaling,
            layer_id=layer_id,
        )

        self.gate_fusion = AttentionGateFusion(
            num_q_heads=num_q_heads,
            v_head_dim=v_head_dim,
            quant_config=quant_config,
            prefix=add_prefix("gate_fusion", prefix),
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """
        Forward pass for the NSA Attention module.

        Args:
            q (torch.Tensor): [num_batch_tokens, num_q_heads * qk_head_dim]
            k (torch.Tensor): [num_batch_tokens, num_kv_heads * qk_head_dim]
            v (torch.Tensor): [num_batch_tokens, num_kv_heads * v_head_dim]
        Returns:
            o (torch.Tensor): [num_batch_tokens, num_q_heads * v_head_dim]
        """

        assert type(forward_batch.attn_backend) in [ FlashInferAttnBackend ], \
            f"only flashinfer support window attn, but got {forward_batch.attn_backend}"
        assert forward_batch.attn_backend.dispatch_reason == WrapperDispatch.SLIDING_WINDOW
        assert type(forward_batch.token_to_kv_pool) in [ NativeSparseMHATokenToKVPool ]

        window_o = self.window_attn(q, k, v, forward_batch, save_kv_cache=True)

        cmp_o, topk_idx = self.compress_attn(q, forward_batch)
        select_o = self.select_attn(q, topk_idx, forward_batch)

        o = self.gate_fusion(cmp_o, select_o, window_o)
        return o

    @staticmethod
    def init_cuda_graph_state_nsa(
        in_max_bs: int,
        in_max_total_num_tokens: int,
        in_max_context_len: int,
        stride, num_q_heads, select_size, 
    ):
        global max_bs, max_total_num_tokens, max_context_len,\
            compress_kv_indices, \
            flat_com_score, flat_slc_score, \
            cuda_graph_kv_indices
        if max_bs != -1: return
        max_bs = in_max_bs
        max_total_num_tokens = in_max_total_num_tokens
        max_context_len = in_max_context_len
        print(f"NSA init_cuda_graph_state_nsa {max_total_num_tokens=} {max_bs=}")
        with torch.device("cuda"):
            compress_kv_indices = torch.zeros(
                max_total_num_tokens // stride, dtype=torch.int32
            )
            flat_com_score = torch.zeros(
                max_total_num_tokens // stride, num_q_heads,
                dtype=torch.float32
            )
            flat_slc_score = torch.zeros(
                max_total_num_tokens // select_size + max_bs, num_q_heads,
                dtype=torch.float32
            )
            cuda_graph_kv_indices = torch.zeros(
                max_total_num_tokens, dtype=torch.int32
            )


class Qwen3NSA(Qwen3Attention):
    def __init__(
        self,
        config: Qwen3NsaConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 1000000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        head_dim: Optional[int] = None,
        max_position_embeddings: int = 32768,
        quant_config: Optional[QuantizationConfig] = None,
        rms_norm_eps: float = None,
        attention_bias: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            rms_norm_eps=rms_norm_eps,
            attention_bias=attention_bias,
            prefix=prefix,
        )

        kernel_size = config.kernel_size
        stride = config.stride
        post_compress_key_rope = getattr(config, 'post_compress_key_rope', False)
        assert not post_compress_key_rope
        apply_additive_intra_block_position_emb = config.apply_additive_intra_block_position_emb
        assert not apply_additive_intra_block_position_emb
        head_share_cmp_att_weight = config.head_share_cmp_att_weight
        assert head_share_cmp_att_weight
        compress_type = getattr(config, 'compress_type', 'weighted')
        assert compress_type == 'gated'
        select_size = config.select_size
        top_n = config.top_n
        slc_att_num_init_blocks = config.slc_att_num_init_blocks
        slc_att_num_local_blocks = config.slc_att_num_local_blocks
        sliding_window_size = config.window_size
        head_share_att_gate_weight = config.head_share_att_gate_weight
        assert head_share_att_gate_weight
        gate_type = getattr(config, 'gate_type', 'sigmoid')
        assert gate_type == 'sigmoid'
        gate_feature = getattr(config, 'gate_feature', 'query')
        assert gate_feature == 'attention'
        gate_mask = getattr(config, 'gate_mask', '111')
        assert gate_mask == '111'

        self.attn = NativeSparseAttention(
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            qk_head_dim=self.head_dim,
            v_head_dim=self.head_dim,
            scaling=self.scaling,
            sliding_window_size=sliding_window_size,
            kernel_size=kernel_size,
            stride=stride,
            select_size=select_size,
            top_n=top_n,
            slc_att_num_init_blocks=slc_att_num_init_blocks,
            slc_att_num_local_blocks=slc_att_num_local_blocks,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )


class Qwen3NsaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3NsaConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 32768)
        head_dim = getattr(config, "head_dim", None)
        self.self_attn = Qwen3NSA(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=config.attention_bias,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            # prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

class Qwen3NsaModel(Qwen2Model):
    def __init__(
        self,
        config: Qwen3NsaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config=config,
            quant_config=quant_config,
            # prefix=prefix,
            decoder_layer_type=Qwen3NsaDecoderLayer,
        )

class Qwen3NsaForCausalLM(nn.Module):
    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = Qwen3NsaModel(config, quant_config=quant_config)
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size, config.hidden_size, quant_config=quant_config
            )
        self.logits_processor = LogitsProcessor(config)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def get_attention_sliding_window_size(self):
        return getattr(self.config, 'window_size', None)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
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
            "gate_fusion.gate_weight": "attn.gate_fusion.gate_proj.weight",
        }

        '''
        'model.layers.0.self_attn.q_norm.weight',
        'model.layers.0.self_attn.k_norm.weight',
        'model.layers.0.self_attn.qkv_proj.weight',
        'model.layers.0.self_attn.o_proj.weight',
        'model.layers.0.self_attn.attn.compress_attn.compress_key.gate_proj.weight',
        'model.layers.0.self_attn.attn.compress_attn.compress_value.gate_proj.weight',
        'model.layers.0.self_attn.attn.gate_fusion.gate_proj.weight',
        'model.layers.0.mlp.gate_up_proj.weight',
        'model.layers.0.mlp.down_proj.weight',
        'model.layers.0.input_layernorm.weight',
        'model.layers.0.post_attention_layernorm.weight',
        '''
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            if name.startswith("model.vision_tower") and name not in params_dict:
                continue

            for source_name, target_name in name_mapping.items():
                if source_name in name:
                    name = name.replace(source_name, target_name)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
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
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name.endswith("gate_fusion.gate_proj.weight"):
                    loaded_weight = loaded_weight.squeeze(0)
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)


EntryClass = Qwen3NsaForCausalLM


