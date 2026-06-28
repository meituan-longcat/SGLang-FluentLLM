from dataclasses import dataclass
from typing import Optional
from flashinfer import BatchPrefillWithRaggedKVCacheWrapper
import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.flash_attention_backend import FlashAttentionMetadata
from sglang.srt.env import global_server_args_dict
from flash_attn_3.flash_attn_interface import flash_attn_varlen_func

from sglang.srt.utils import get_colorful_logger
logger = get_colorful_logger(__name__)

@triton.jit
def create_chunked_cache_kv_indices(
    req_to_token_ptr,  # (max_batch, max_context_len,)
    req_pool_indices_ptr,  # (batch_size,)
    chunk_start_idx_ptr,  # (batch_size,)
    chunk_seq_lens_ptr,  # (batch_size,)
    chunk_cu_seq_lens_ptr,  # (batch_size + 1,)
    chunk_kv_indices_ptr,  # (num_chunk_tokens,)
    req_to_token_ptr_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)

    # find the req pool idx, this is for batch to token
    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    chunk_kv_indices_offset = tl.load(chunk_cu_seq_lens_ptr + pid)

    # get the token positions of current chunk
    chunk_start_pos = tl.load(chunk_start_idx_ptr + pid).to(tl.int32)
    chunk_seq_len = tl.load(chunk_seq_lens_ptr + pid).to(tl.int32)

    num_loop = tl.cdiv(chunk_seq_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < chunk_seq_len
        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + chunk_start_pos
            + offset,
            mask=mask,
        )
        tl.store(
            chunk_kv_indices_ptr + chunk_kv_indices_offset + offset, data, mask=mask
        )

def get_max_chunk_capacity():
    return global_server_args_dict["mla_max_chunk_capacity"]

# Here we suppose the length of each chunk is equal
# For example, if we have 4 sequences with seq length [256, 512, 768, 1024], chunk_len = 256
# num_chunks = cdiv(1024, 256) = 4
# chunk_starts = [[0, 0, 0, 0], [256, 256, 256, 256], [512, 512, 512, 512], [768, 768, 768, 768]]
# chunk_ends = [[256, 256, 256, 256], [256, 512, 512, 512], [256, 512, 768, 768], [256, 512, 768, 1024]]
# chunk_seq_lens = [[256, 256, 256, 256], [0, 256, 256, 256], [0, 0, 256, 256], [0, 0, 0, 256]]
# TODO: Implement a better way to allocate chunk lengths that uses memory spaces more efficiently.
"""
        seq0 seq1 seq2 seq3
chunk0   --   --   --   --
chunk1   --   --   --   --
chunk2   --   --   --   --
chunk3   --   --   --   --
"""
# starts, ends, len_in_chunk, cu_seq_lens, all satisfy the above layout
@dataclass
class Chunks(object):
    starts: torch.Tensor
    ends: torch.Tensor
    len_in_chunk: torch.Tensor
    _cu_seq_lens: Optional[torch.Tensor] = None

    def cu_seq_lens(self):
        if self._cu_seq_lens is None:
            num_chunks = self.starts.shape[0]
            bs = self.starts.shape[1]
            result = torch.zeros(num_chunks, bs + 1, device=self.starts.device, dtype=torch.int32)
            result[:, 1:] = self.len_in_chunk.cumsum(dim=1).to(torch.int32)
            self._cu_seq_lens = result

        return self._cu_seq_lens 


def chunking(prefix_lens: torch.Tensor, num_chunks, batch_size, chunk_len):
    starts = (
        torch.arange(num_chunks, device=prefix_lens.device, dtype=torch.int32)
        .unsqueeze(1)
        .expand(-1, batch_size)
        * chunk_len
    )
    ends = torch.min(prefix_lens.unsqueeze(0), starts + chunk_len).to(torch.int32)

    chunks = Chunks(
        starts=starts,
        ends=ends,
        len_in_chunk=(ends - starts).clamp(min=0).to(torch.int32)
    )
    return chunks


# Called before each attention module if using chunked kv cache for prefill
# Some of the codes are adapted from https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/mla/common.py
def get_chunks(
    prefix_lens, 
    prefix_lens_cpu, 
    req_to_token, 
    req_pool_indices,
    token_to_kv_pool):
    device: torch.device = prefix_lens.device
    batch_size = len(prefix_lens_cpu)

    chunk_capacity = get_max_chunk_capacity()
    chunk_len = chunk_capacity // batch_size
    num_chunks = (max(prefix_lens_cpu) + chunk_len - 1) // chunk_len

    # Here we compute chunk lens twice to avoid stream sync, once on gpu and once on cpu.
    chunks = chunking(prefix_lens, num_chunks, batch_size, chunk_len)
    chunks_cpu = chunking(torch.tensor(prefix_lens_cpu), num_chunks, batch_size, chunk_len)

    num_tokens_per_forward = chunks_cpu.len_in_chunk.sum(dim=1).tolist()
    # assert max(num_tokens_per_forward) <= get_max_chunk_capacity()

    chunk_kv_indices_list = []
    for idx in range(num_chunks):
        chunk_kv_indices = torch.empty(num_tokens_per_forward[idx], dtype=torch.int32, device=device)
        create_chunked_cache_kv_indices[(batch_size,)](
            req_to_token,
            req_pool_indices,
            chunks.starts[idx],
            chunks.len_in_chunk[idx],
            chunks.cu_seq_lens()[idx],
            chunk_kv_indices,
            req_to_token.shape[1],
        )
        
        if (
            hasattr(token_to_kv_pool, 'enable_mla_l1_5_cache')
            and token_to_kv_pool.enable_mla_l1_5_cache
        ):
            mask, local_indices, per_rank_count, perm, inv_perm = (
                token_to_kv_pool.global_loc_to_local_mapping(
                    chunk_kv_indices,
                )
            )
            chunk_kv_indices = (
                chunk_kv_indices,
                mask,
                local_indices,
                per_rank_count,
                perm,
                inv_perm,
            )

        chunk_kv_indices_list.append(chunk_kv_indices)

    return chunks, chunk_kv_indices_list, chunks_cpu


# Reuse this workspace buffer across all flashinfer wrappers
global_workspace_buffer = None

class ChunkerBase(object):
    def __init__(
        self,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        v_head_dim,
        dtype,
        device,
        causal: bool,
        step_counter,
    ):
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim
        self.q_data_type = dtype
        self.causal = causal
        self.step_counter = step_counter
        self.device = device

    def plan(self, *args, **kwargs):
        """
        For Metadata Preparation
        """
        raise NotImplementedError("NotImplemented!")

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scaling,
        logits_soft_cap,
    ):
        raise NotImplementedError("NotImplemented")


class ChunkFlashAttn3(ChunkerBase):
    def __init__(
        self,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        v_head_dim,
        dtype,
        device,
        causal,
        step_counter,
    ):
        super().__init__(
            num_qo_heads,
            num_kv_heads,
            head_dim,
            v_head_dim,
            dtype,
            device,
            causal,
            step_counter,
        )
        self.metadata = None

    def plan(self, q_lens: torch.Tensor, kv_lens: torch.Tensor, q_lens_cpu: list, kv_lens_cpu: list):
        metadata = FlashAttentionMetadata()
        metadata.cache_seqlens_int32 = kv_lens.to(torch.int32)
        metadata.max_seq_len_k = max(kv_lens_cpu)
        metadata.cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(kv_lens, dim=0, dtype=torch.int32), (1, 0)
        )
        metadata.cu_seqlens_q = torch.nn.functional.pad(
            torch.cumsum(q_lens, dim=0, dtype=torch.int32), (1, 0)
        )
        metadata.max_seq_len_q = max(q_lens_cpu)
        self.metadata = metadata
        return self

    def forward_extend(self, q, k, v, scaling, logits_soft_cap=None):
        if self.causal and self.step_counter is not None:
            self.step_counter.record_cache()
        output, lse, *rest = flash_attn_varlen_func(
            q=q.view(-1, self.num_qo_heads, self.head_dim),
            k=k.view(-1, self.num_kv_heads, self.head_dim).to(q.dtype),
            v=v.view(-1, self.num_kv_heads, self.v_head_dim).to(q.dtype),
            cu_seqlens_q=self.metadata.cu_seqlens_q,
            cu_seqlens_k=self.metadata.cu_seqlens_k,
            max_seqlen_q=self.metadata.max_seq_len_q,
            max_seqlen_k=self.metadata.max_seq_len_k,
            softmax_scale=scaling,
            causal=self.causal,
            return_attn_probs=True,
        )
        # NOTE: lse must be transposed when use fa3
        return output, lse.T.contiguous()


class ChunkFlashInferAttn(ChunkerBase):
    def __init__(
        self,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        v_head_dim,
        dtype,
        device,
        causal: bool,
        step_counter,
    ):
        super().__init__(
            num_qo_heads,
            num_kv_heads,
            head_dim,
            v_head_dim,
            dtype,
            device,
            causal,
            step_counter,
        )
        flashinfer_workspace_size = 512 * 1024 * 1024
        # Allocate buffers
        global global_workspace_buffer
        if global_workspace_buffer is None:
            global_workspace_buffer = torch.empty(
                flashinfer_workspace_size,
                dtype=torch.uint8,
                device=device,
            )
        self.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
            global_workspace_buffer, "NHD"
        )

    def plan(self, q_lens, kv_lens, *args, **kwargs):
        bs = len(q_lens)

        qo_indptr = torch.zeros(bs + 1, device=q_lens.device, dtype=torch.int32)
        qo_indptr[1:] = torch.cumsum(q_lens, dim=0)

        kv_indptr = torch.zeros(bs + 1, device=kv_lens.device, dtype=torch.int32)
        kv_indptr[1:] = torch.cumsum(kv_lens, dim=0)

        self.prefill_wrapper_ragged.begin_forward(
            qo_indptr,
            kv_indptr,
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            self.v_head_dim,
            q_data_type=self.q_data_type,
        )
        return self

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scaling,
        logits_soft_cap
    ):
        if self.causal and self.step_counter is not None:
            self.step_counter.record_cache()

        q = q.contiguous()
        o1, s1 = self.prefill_wrapper_ragged.forward_return_lse(
            q.view(-1, self.num_qo_heads, self.head_dim),
            k.view(-1, self.num_kv_heads, self.head_dim),
            v.view(-1, self.num_kv_heads, self.v_head_dim),
            causal=self.causal,
            sm_scale=scaling,
            logits_soft_cap=logits_soft_cap,
        )
        return o1, s1

def get_attns(num_qo_heads, num_kv_heads, head_dim, v_head_dim, dtype, device, q_lens, chunks: Chunks, chunks_cpu: Chunks, q_lens_cpu):
    attns = []
    backend = global_server_args_dict["chunker_backend"]
    AttnImpl = None
    if backend == "flashinfer":
        AttnImpl = ChunkFlashInferAttn
    elif backend == "fa3":
        AttnImpl = ChunkFlashAttn3
    else:
        raise RuntimeError(f"Not Supported backend: {backend}")

    for chunk_idx in range(chunks.starts.shape[0]):
        attns.append(
            AttnImpl(num_qo_heads, num_kv_heads, head_dim, v_head_dim, dtype, device, False, None).plan(q_lens, chunks.len_in_chunk[chunk_idx], q_lens_cpu, chunks_cpu.len_in_chunk[chunk_idx])
        )
    return attns

def get_casual_attn(num_qo_heads, num_kv_heads, head_dim, v_head_dim, dtype, device, q_lens, kv_lens, step_counter, q_lens_cpu, kv_lens_cpu):
    backend = global_server_args_dict["chunker_backend"]
    if backend == "flashinfer":
        return ChunkFlashInferAttn(num_qo_heads, num_kv_heads, head_dim, v_head_dim, dtype, device, True, step_counter).plan(q_lens, kv_lens, q_lens_cpu, kv_lens_cpu)
    elif backend == "fa3":
        return ChunkFlashAttn3(num_qo_heads, num_kv_heads, head_dim, v_head_dim, dtype, device, True, step_counter).plan(q_lens, kv_lens, q_lens_cpu, kv_lens_cpu)
    else:
        raise RuntimeError(f"Not Supported backend: {backend}")


@triton.jit
def create_streamed_cache_kv_indices(
    req_to_token_ptr,  # (max_batch, max_context_len,)
    req_pool_indices_ptr,  # (batch_size,)
    extend_seq_lens_ptr,  # (batch_size,)
    seq_lens_ptr,  # (batch_size,)
    streamed_sink_bounds_ptr,  # (batch_size,)
    streamed_recent_bounds_ptr,  # (batch_size,)
    streamed_lens_ptr,  # (batch_size,)
    streamed_cu_lens_ptr,  # (batch_size + 1,)
    streamed_kv_indices_ptr,  # (num_streamed_tokens,)
    req_to_token_ptr_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 128
    pid = tl.program_id(axis=0)

    # find the req pool idx, this is for batch to token
    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    seq_len = tl.load(seq_lens_ptr + pid)
    streamed_len = tl.load(streamed_lens_ptr + pid)
    streamed_kv_indices_offset = tl.load(streamed_cu_lens_ptr + pid)

    # get the token positions of current stream
    streamed_sink_bound = tl.load(streamed_sink_bounds_ptr + pid).to(tl.int32)
    streamed_recent_bound = tl.load(streamed_recent_bounds_ptr + pid).to(tl.int32)

    num_loop = tl.cdiv(streamed_sink_bound, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < streamed_sink_bound
        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + offset,
            mask=mask,
        )
        tl.store(
            streamed_kv_indices_ptr + streamed_kv_indices_offset + offset, data, mask=mask
        )

    num_loop = tl.cdiv(seq_len - streamed_recent_bound, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = streamed_recent_bound + offset < seq_len
        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + streamed_recent_bound
            + offset,
            mask=mask,
        )
        tl.store(
            streamed_kv_indices_ptr + streamed_kv_indices_offset + streamed_sink_bound + offset, data, mask=mask
        )


def get_streamed_kv_indices(extend_seq_lens, extend_seq_lens_cpu, seq_lens, seq_lens_cpu, req_to_token, req_pool_indices, is_prefill):
    if is_prefill: # Prefill, early exit to avoid redundant kv indices fetching.
        return None, seq_lens
    
    device: torch.device = seq_lens.device
    batch_size = len(seq_lens_cpu)

    # FIXME: fixed to 1 sink block and 7 recent blocks here.
    streamed_sink_bounds = torch.min(seq_lens, torch.full_like(seq_lens, 128))
    streamed_recent_bounds = torch.max(streamed_sink_bounds, ((seq_lens - extend_seq_lens + 127) // 128 - 7) * 128)
    streamed_lens = streamed_sink_bounds + seq_lens - streamed_recent_bounds
    streamed_cu_lens = torch.zeros(batch_size + 1, device=device, dtype=torch.int32)
    streamed_cu_lens[1: ] = streamed_lens.cumsum(dim=0)
    num_tokens_per_forward = streamed_lens.sum().item()

    streamed_kv_indices = torch.empty(num_tokens_per_forward, dtype=torch.int32, device=device)
    create_streamed_cache_kv_indices[(batch_size,)](
        req_to_token,
        req_pool_indices,
        extend_seq_lens,
        seq_lens,
        streamed_sink_bounds,
        streamed_recent_bounds,
        streamed_lens,
        streamed_cu_lens,
        streamed_kv_indices,
        req_to_token.shape[1],
    )

    return streamed_kv_indices, streamed_lens

from duo_flash_attn_interface import flash_attn_varlen_func as duo_flash_attn_varlen_func

class StreamedFlashAttn(object):
    def __init__(self, num_qo_heads, num_kv_heads, head_dim, v_head_dim, dtype, q_lens, kv_lens, streaming_info, head_mask_type, step_counter):
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim
        self.q_data_type = dtype
        self.q_lens = q_lens
        self.max_q_len = torch.max(self.q_lens).item()
        bs = len(q_lens)
        self.cu_q_lens = torch.zeros(bs + 1, device=q_lens.device, dtype=torch.int32)
        self.cu_q_lens[1: ] = torch.cumsum(q_lens, dim=0)
        self.kv_lens = kv_lens
        self.max_kv_len = torch.max(self.kv_lens).item()
        self.cu_kv_lens = torch.zeros(bs + 1, device=kv_lens.device, dtype=torch.int32)
        self.cu_kv_lens[1: ] = torch.cumsum(kv_lens, dim=0)
        self.streaming_info = streaming_info
        self.head_mask_type = head_mask_type
        self.step_counter = step_counter

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scaling,
        layer_id,
    ):
        if self.step_counter is not None:
            self.step_counter.record_cache()

        o = duo_flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=self.cu_q_lens,
            cu_seqlens_k=self.cu_kv_lens,
            head_mask_type=self.head_mask_type[layer_id],
            streaming_info=self.streaming_info,
            max_seqlen_q=self.max_q_len,
            max_seqlen_k=self.max_kv_len,
            softmax_scale=scaling,
            causal=True,
        )
        return o
    

def get_streamed_attn(num_qo_heads, num_kv_heads, head_dim, v_head_dim, dtype, q_lens, kv_lens, streaming_info, head_mask_type, step_counter):
    return StreamedFlashAttn(num_qo_heads, num_kv_heads, head_dim, v_head_dim, dtype, q_lens, kv_lens, streaming_info, head_mask_type, step_counter)
