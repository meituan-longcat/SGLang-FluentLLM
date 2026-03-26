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
"""
Store information about a forward batch.

The following is the flow of data structures for a batch:

ScheduleBatch -> ModelWorkerBatch -> ForwardBatch

- ScheduleBatch is managed by `scheduler.py::Scheduler`.
  It contains high-level scheduling data. Most of the data is on the CPU.
- ModelWorkerBatch is managed by `tp_worker.py::TpModelWorker`.
  It is a subset of `ScheduleBatch` that only contains data related to the model forward on GPU.
  It will be transformed from CPU scheduler to GPU model runner.
- ForwardBatch is managed by `model_runner.py::ModelRunner`.
  It contains low-level tensor data. Most of the data consists of GPU tensors.
  ForwardBatch has positions attribute, ForwardBatch.init_new constructs positions
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, auto
from typing import TYPE_CHECKING, Any, List, Optional, Union, Dict

import torch
import triton
import triton.language as tl

from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.utils import get_compiler_backend, split_array_by_half_sum, is_npu, check_memory_debug

from sglang.srt.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

_is_npu = is_npu()

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.managers.req import Req
    from sglang.srt.managers.schedule_batch import ModelWorkerBatch
    from sglang.srt.mem_cache.memory_pool import BaseTokenToKVPool, ReqToTokenPool
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
    from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


class ForwardMode(IntEnum):
    # Extend a sequence. The KV cache of the beginning part of the sequence is already computed (e.g., system prompt).
    EXTEND = auto()
    # Decode one token.
    DECODE = auto()
    # Contains both EXTEND and DECODE when doing chunked prefill.
    MIXED = auto()
    # No sequence to forward. For data parallel attention, some workers wil be IDLE if no sequence are allocated.
    IDLE = auto()

    # Used in speculative decoding: verify a batch in the target model.
    TARGET_VERIFY = auto()
    # Used in speculative decoding: extend a batch in the draft model.
    DRAFT_EXTEND = auto()

    # A dummy first batch to start the pipeline for overlap scheduler.
    # It is now used for triggering the sampling_info_done event for the first prefill batch.
    DUMMY_FIRST = auto()

    def is_extend(self):
        return (
            self == ForwardMode.EXTEND
            or self == ForwardMode.MIXED
            or self == ForwardMode.DRAFT_EXTEND
            or self == self.TARGET_VERIFY
        )

    def is_decode(self):
        return self == ForwardMode.DECODE

    def is_mixed(self):
        return self == ForwardMode.MIXED

    def is_idle(self):
        return self == ForwardMode.IDLE

    def is_target_verify(self):
        return self == ForwardMode.TARGET_VERIFY

    def is_draft_extend(self):
        return self == ForwardMode.DRAFT_EXTEND

    def is_cuda_graph(self):
        return (
            self == ForwardMode.DECODE
            or self == ForwardMode.TARGET_VERIFY
            or self == ForwardMode.IDLE
        )

    def is_dummy_first(self):
        return self == ForwardMode.DUMMY_FIRST

    def is_decode_or_idle(self):
        return self == ForwardMode.DECODE or self == ForwardMode.IDLE

    def is_decode_or_target_verify(self):
        return self.is_decode() or self.is_target_verify()


class CaptureHiddenMode(IntEnum):
    NULL = auto()
    # Capture hidden states of all tokens.
    FULL = auto()
    # Capture a hidden state of the last token.
    LAST = auto()

    def need_capture(self):
        return self != CaptureHiddenMode.NULL

    def is_full(self):
        return self == CaptureHiddenMode.FULL

    def is_last(self):
        return self == CaptureHiddenMode.LAST


@dataclass
class MicroBatches:
    micro_batches: List[MicroBatch]
    seq_split_index: int
    token_split_index: int

    def __getitem__(self, index):
        return self.micro_batches[index]

    def __len__(self):
        return len(self.micro_batches)


@dataclass
class MicroBatch:
    index: int
    tp_num_tokens: int
    forward_batch: ForwardBatch
    hidden_states: Optional[torch.Tensor] = None
    residual: Optional[torch.Tensor] = None


@dataclass
class ForwardBatch:
    """Store all inputs of a forward pass."""

    # The forward mode
    forward_mode: ForwardMode
    # The batch size
    batch_size: int
    # The input ids
    input_ids: torch.Tensor
    # The indices of requests in the req_to_token_pool
    req_pool_indices: torch.Tensor
    # The sequence length
    seq_lens: torch.Tensor

    # The sum of all sequence lengths
    seq_lens_sum: int

    # Optional seq_lens on cpu
    seq_lens_cpu: Optional[torch.Tensor] = None

    draft_input_ids: Optional[torch.Tensor] = None

    # For logprob
    return_logprob: bool = False
    top_logprobs_nums: Optional[List[int]] = None
    token_ids_logprobs: Optional[List[List[int]]] = None

    # Position information
    positions: torch.Tensor = None

    # For extend
    extend_num_tokens: Optional[int] = None
    extend_seq_lens: Optional[torch.Tensor] = None
    extend_seq_lens_cpu: Optional[List[int]] = None

    extend_prefix_lens: Optional[torch.Tensor] = None
    extend_prefix_lens_cpu: Optional[List[int]] = None

    extend_start_loc: Optional[torch.Tensor] = None
    extend_logprob_start_lens_cpu: Optional[List[int]] = None
    extend_input_logprob_token_ids_gpu: Optional[torch.Tensor] = None

    # For input embeddings
    input_embeds: Optional[torch.tensor] = None
    input_multi_ids: Optional[torch.Tensor] = None
    input_extra_infos: Optional[Dict] = None
    # Multimodal models typically complete multimodal sampling before returning text_logits,
    # storing results temporarily here to avoid redundant sampling
    temp_multi_ids: Optional[torch.Tensor] = None
    # For over embedding: batch_size * (n-1), n is the maximum n-gram embedding
    # When decoding, besides the current token, we also need previous tokens to compute n-gram ids
    # All token information for all requests [max_running_req, context_len]
    oe_token_table: Optional[torch.Tensor] = None
    oe_column_starts: Optional[torch.Tensor] = None
    oe_req_lens: Optional[torch.Tensor] = None
    oe_out_column_starts: Optional[torch.Tensor] = None
    oe_out_req_lens: Optional[torch.Tensor] = None

    # Sampling info
    sampling_info: SamplingBatchInfo = None

    # Attention backend
    out_cache_loc: torch.Tensor = None
    req_to_token_pool: ReqToTokenPool = None
    token_to_kv_pool: BaseTokenToKVPool = None
    attn_backend: AttentionBackend = None
    new_tokens_to_compute: torch.Tensor = None
    new_tokens_total: int = None

    # For DP attention
    global_num_tokens: Optional[List[int]] = None  # e.g. dp = 4, attn-tp = 2, [A, A, B, B, C, C, D, D]
    gathered_buffer: Optional[torch.Tensor] = None
    all_decode_or_idle: bool = False
    can_run_tbo: bool = False

    # Speculative decoding
    spec_info: Optional[Union[EagleVerifyInput, EagleDraftInput]] = None
    spec_algorithm: SpeculativeAlgorithm = None
    capture_hidden_mode: CaptureHiddenMode = None
    spec_num_steps: int = 0

    # For padding
    padded_static_len: int = -1  # -1 if not padded

    # For Qwen2-VL
    mrope_positions: torch.Tensor = None

    global_batch_size: List[int] = None

    chunk_attns: Optional[List] = None
    casual_chunk_attn: Optional[Any] = None
    streamed_attn: Optional[Any] = None

    # Block table cache for attention backends
    block_table_cache: Optional[torch.Tensor] = None

    # The indices of requests in the req_to_token_pool cpu
    req_pool_indices_cpu: Optional[List[int]] = None
    reqs: List[Req] = None

    captureing_prefill_graph: bool = False

    @classmethod
    def init_new(
        cls,
        batch: ModelWorkerBatch,
        model_runner: ModelRunner,
    ):
        device = model_runner.device
        extend_input_logprob_token_ids_gpu = None
        if batch.extend_input_logprob_token_ids is not None:
            extend_input_logprob_token_ids_gpu = (
                batch.extend_input_logprob_token_ids.to(device, non_blocking=True)
            )
        ret = cls(
            forward_mode=batch.forward_mode,
            batch_size=len(batch.seq_lens),
            input_ids=batch.input_ids,
            draft_input_ids=batch.draft_input_ids,
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens,
            seq_lens_sum=batch.seq_lens_sum,
            return_logprob=batch.return_logprob,
            top_logprobs_nums=batch.top_logprobs_nums,
            token_ids_logprobs=batch.token_ids_logprobs,
            global_num_tokens=batch.global_num_tokens,
            all_decode_or_idle=batch.all_decode_or_idle,
            can_run_tbo=batch.can_run_tbo,
            sampling_info=batch.sampling_info,
            req_to_token_pool=model_runner.req_to_token_pool,
            token_to_kv_pool=model_runner.token_to_kv_pool,
            attn_backend=model_runner.attn_backend,
            spec_algorithm=batch.spec_algorithm,
            spec_info=batch.spec_info,
            capture_hidden_mode=batch.capture_hidden_mode,
            input_embeds=batch.input_embeds,
            oe_token_table=batch.oe_token_table,
            oe_column_starts=torch.empty(len(batch.seq_lens), dtype=torch.int32, device=device),
            oe_req_lens=torch.empty(len(batch.seq_lens), dtype=torch.int32, device=device),
            oe_out_column_starts=torch.empty(len(batch.seq_lens), dtype=torch.int32, device=device),
            oe_out_req_lens=torch.empty(len(batch.seq_lens), dtype=torch.int32, device=device),
            extend_input_logprob_token_ids_gpu=extend_input_logprob_token_ids_gpu,
            new_tokens_to_compute=batch.new_tokens_to_compute,
            new_tokens_total=batch.new_tokens_total,
            spec_num_steps=model_runner.spec_num_steps,
            global_batch_size=batch.global_batch_size,
            input_multi_ids=batch.input_multi_ids,
            reqs=batch.reqs,
        )

        if ret.global_num_tokens is not None:
            max_len = max(ret.global_num_tokens)
            ret.gathered_buffer = torch.zeros(
                (max_len * model_runner.tp_size, model_runner.model_config.hidden_size),
                dtype=model_runner.dtype,
                device=device,
            )

        if ret.forward_mode.is_idle():
            ret.positions = torch.empty((0,), device=device)
            ret.set_out_cache_loc()
            return ret

        # Override the positions with spec_info
        if (
            ret.spec_info is not None
            and getattr(ret.spec_info, "positions", None) is not None
        ):
            ret.positions = ret.spec_info.positions

        # Get seq_lens_cpu if needed
        if ret.seq_lens_cpu is None:
            ret.seq_lens_cpu = batch.seq_lens_cpu

        # Init position information
        if ret.forward_mode.is_decode():
            if ret.positions is None:
                ret.positions = clamp_position(batch.seq_lens)
        else:
            ret.extend_seq_lens = torch.tensor(
                batch.extend_seq_lens, dtype=torch.int32
            ).to(device, non_blocking=True)
            ret.extend_prefix_lens = torch.tensor(
                batch.extend_prefix_lens, dtype=torch.int32
            ).to(device, non_blocking=True)
            ret.extend_num_tokens = batch.extend_num_tokens
            if not _is_npu:
                positions, ret.extend_start_loc = compute_position_triton(
                    ret.extend_prefix_lens,
                    ret.extend_seq_lens,
                    ret.extend_num_tokens,
                )
            else:
                positions, ret.extend_start_loc = compute_position_torch(
                    ret.extend_prefix_lens, ret.extend_seq_lens
                )
            if ret.positions is None:
                ret.positions = positions
            ret.extend_prefix_lens_cpu = batch.extend_prefix_lens
            ret.extend_seq_lens_cpu = batch.extend_seq_lens
            ret.extend_logprob_start_lens_cpu = batch.extend_logprob_start_lens

        if model_runner.model_is_mrope:
            ret._compute_mrope_positions(model_runner, batch)

        ret.set_out_cache_loc()

        if hasattr(batch, "request_cache_input"):
            input_dict = batch.request_cache_input
            setattr(ret, f"request_cache_input", input_dict)
        return ret

    def get_out_cache_loc_kernel_wrapper(self, bs, out_cache_loc):
        if not _is_npu:
            get_out_cache_loc_kernel[(bs,)](
                out_cache_loc_ptr=out_cache_loc,
                req_to_token_ptr=self.req_to_token_pool.req_to_token,
                req_pool_indices_ptr=self.req_pool_indices,
                new_compute_lens_ptr=self.new_tokens_to_compute,
                cache_lens_ptr=self.req_to_token_pool.verified_lens,
                req_to_token_ptr_stride=self.req_to_token_pool.req_to_token.shape[1]
            )
        else:
            # TODO for npu
            batch_size = self.req_pool_indices.shape[0]
            device = self.req_pool_indices.device

            # Cumulative sum
            cumsum_starts = torch.cat([
                torch.tensor([0], device=device),
                self.new_tokens_to_compute.cumsum(0)[:-1]
            ])

            for i in range(batch_size):
                new_compute_len = self.new_tokens_to_compute[i].item()
                if new_compute_len == 0:
                    continue
                req_index = self.req_pool_indices[i].item()
                cache_len = self.req_to_token_pool.verified_lens[req_index].item()
                cumsum_start = cumsum_starts[i].item()
                req_tokens = self.req_to_token_pool.req_to_token[req_index, cache_len:cache_len + new_compute_len]
                out_cache_loc[cumsum_start:cumsum_start + new_compute_len] = req_tokens

    def get_num_tokens(self, tp_num_tokens: int):
        if self.global_num_tokens is not None:
            num_global_tokens = sum(self.global_num_tokens) // get_attention_tp_size()
            max_num_tokens_per_gpu = (
                max(self.global_num_tokens) + get_attention_tp_size() - 1
            ) // get_attention_tp_size()
        else:
            num_global_tokens = tp_num_tokens
            max_num_tokens_per_gpu = (
                tp_num_tokens + get_attention_tp_size() - 1
            ) // get_attention_tp_size()
        return num_global_tokens, max_num_tokens_per_gpu

    def set_out_cache_loc(self):
        """
        Here we obtain the actual write positions from req_to_token_pool, not necessarily the positions allocated in this scheduler.

        Note: The update of verified_lens is critical for obtaining write positions, please pay special attention to it.
        """
        out_cache_loc = torch.zeros(
            size=(self.new_tokens_total,),
            device=self.req_to_token_pool.req_to_token.device,
            dtype=torch.int32
        )
        if self.forward_mode.is_idle():
            # idle batch doesn't actually write to KV cache, return all zeros
            self.out_cache_loc = out_cache_loc
            return
        if self.forward_mode == ForwardMode.EXTEND and self.extend_prefix_lens is not None:
            self.req_to_token_pool.verified_lens[self.req_pool_indices] = self.extend_prefix_lens
        bs = self.batch_size
        self.get_out_cache_loc_kernel_wrapper(bs, out_cache_loc)
        self.out_cache_loc = out_cache_loc
        # Increment slot reference count by 1 for server idle check
        if check_memory_debug():
            self.token_to_kv_pool.token_slot_refs[self.out_cache_loc] += 1

    def split_micro_batch(self) -> Union[MicroBatches, None]:
        if not self.can_run_tbo:
            return None

        seq_split_idx, token_split_idx = self._compute_tbo_split_idx()

        fb1 = ForwardBatch(
            forward_mode=self.forward_mode,
            batch_size=seq_split_idx,
            input_ids=self.input_ids[:token_split_idx],
            positions=self.positions[:token_split_idx],
            req_pool_indices=self.req_pool_indices[:seq_split_idx],
            seq_lens=self.seq_lens[:seq_split_idx],
            seq_lens_cpu=self.seq_lens_cpu[:seq_split_idx] if self.seq_lens_cpu is not None else None,
            out_cache_loc=self.out_cache_loc[:token_split_idx],
            seq_lens_sum=sum(self.seq_lens[:seq_split_idx]),
            global_num_tokens=self.global_num_tokens,
            all_decode_or_idle=self.all_decode_or_idle,
            token_to_kv_pool=self.token_to_kv_pool,
            attn_backend=self.attn_backend.tbo_attn_backends[0],
            spec_info=self.spec_info,
            sampling_info=self.sampling_info,
            return_logprob=self.return_logprob,
            top_logprobs_nums=self.top_logprobs_nums[:seq_split_idx] if self.top_logprobs_nums else None,
            token_ids_logprobs=self.token_ids_logprobs[:seq_split_idx] if self.token_ids_logprobs else None,
            extend_num_tokens=self.extend_seq_lens[:seq_split_idx].sum().item() if self.extend_seq_lens is not None else None,
            extend_seq_lens=self.extend_seq_lens[:seq_split_idx] if self.extend_seq_lens is not None else None,
            extend_seq_lens_cpu=self.extend_seq_lens_cpu[:seq_split_idx] if self.extend_seq_lens_cpu is not None else None,
            extend_prefix_lens=self.extend_prefix_lens[:seq_split_idx] if self.extend_prefix_lens is not None else None,
            extend_prefix_lens_cpu=self.extend_prefix_lens_cpu[:seq_split_idx] if self.extend_prefix_lens_cpu is not None else None,
            extend_logprob_start_lens_cpu=self.extend_logprob_start_lens_cpu[:seq_split_idx] if self.extend_logprob_start_lens_cpu else None,
            capture_hidden_mode=self.capture_hidden_mode,
        )

        fb2 = ForwardBatch(
            forward_mode=self.forward_mode,
            batch_size=self.batch_size - seq_split_idx,
            input_ids=self.input_ids[token_split_idx:],
            positions=self.positions[token_split_idx:],
            req_pool_indices=self.req_pool_indices[seq_split_idx:],
            seq_lens=self.seq_lens[seq_split_idx:],
            seq_lens_cpu=self.seq_lens_cpu[seq_split_idx:] if self.seq_lens_cpu is not None else None,
            out_cache_loc=self.out_cache_loc[token_split_idx:],
            seq_lens_sum=sum(self.seq_lens[seq_split_idx:]),
            global_num_tokens=self.global_num_tokens,
            all_decode_or_idle=self.all_decode_or_idle,
            token_to_kv_pool=self.token_to_kv_pool,
            attn_backend=self.attn_backend.tbo_attn_backends[1],
            spec_info=self.spec_info,
            sampling_info=self.sampling_info,
            return_logprob=self.return_logprob,
            top_logprobs_nums=self.top_logprobs_nums[seq_split_idx:] if self.top_logprobs_nums else None,
            token_ids_logprobs=self.token_ids_logprobs[seq_split_idx:] if self.token_ids_logprobs else None,
            extend_num_tokens=self.extend_seq_lens[seq_split_idx:].sum().item() if self.extend_seq_lens is not None else None,
            extend_seq_lens=self.extend_seq_lens[seq_split_idx:] if self.extend_seq_lens is not None else None,
            extend_seq_lens_cpu=self.extend_seq_lens_cpu[seq_split_idx:] if self.extend_seq_lens_cpu is not None else None,
            extend_prefix_lens=self.extend_prefix_lens[seq_split_idx:] if self.extend_prefix_lens is not None else None,
            extend_prefix_lens_cpu=self.extend_prefix_lens_cpu[seq_split_idx:] if self.extend_prefix_lens_cpu is not None else None,
            extend_logprob_start_lens_cpu=self.extend_logprob_start_lens_cpu[seq_split_idx:] if self.extend_logprob_start_lens_cpu else None,
            capture_hidden_mode=self.capture_hidden_mode,
        )

        micro_batch_1 = MicroBatch(
            index=0,
            tp_num_tokens=fb1.input_ids.shape[0],
            forward_batch=fb1
        )

        micro_batch_2 = MicroBatch(
            index=1,
            tp_num_tokens=fb2.input_ids.shape[0],
            forward_batch=fb2
        )

        return MicroBatches(
            micro_batches=[micro_batch_1, micro_batch_2],
            seq_split_index=seq_split_idx,
            token_split_index=token_split_idx
        )

    def _compute_tbo_split_idx(self):
        if self.forward_mode == ForwardMode.EXTEND:
            seq_split_idx = split_array_by_half_sum(self.seq_lens_cpu)
            token_split_idx = sum(self.extend_seq_lens_cpu[:seq_split_idx])
        else:
            seq_split_idx = self.batch_size // 2
            token_split_idx = seq_split_idx * self.spec_info.draft_token_num \
                if self.spec_info else seq_split_idx
        return seq_split_idx, token_split_idx


def compute_position_triton(
    extend_prefix_lens: torch.Tensor, extend_seq_lens: torch.Tensor, extend_seq_lens_sum
):
    """Compute positions. It is a fused version of `compute_position_torch`."""
    batch_size = extend_seq_lens.shape[0]
    positions = torch.empty(
        extend_seq_lens_sum, dtype=torch.int64, device=extend_seq_lens.device
    )
    extend_start_loc = torch.empty(
        batch_size, dtype=torch.int32, device=extend_seq_lens.device
    )
    has_prefix = extend_prefix_lens.shape[0] == batch_size
    # Launch kernel
    compute_position_kernel[(batch_size,)](
        positions,
        extend_start_loc,
        extend_prefix_lens,
        extend_seq_lens,
        has_prefix
    )

    return positions, extend_start_loc


@triton.jit
def compute_position_kernel(
    positions,
    extend_start_loc,
    extend_prefix_lens,
    extend_seq_lens,
    has_prefix: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(0).to(tl.int64)

    prefix_len = tl.load(extend_prefix_lens + pid) if has_prefix else 0
    seq_len = tl.load(extend_seq_lens + pid)

    # NOTE: This can be slow for large bs
    cumsum_start = tl.cast(0, tl.int64)
    for i in range(pid):
        cumsum_start += tl.load(extend_seq_lens + i)

    num_loop = tl.cdiv(seq_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        tl.store(
            positions + cumsum_start + offset,
            prefix_len + offset,
            mask=offset < seq_len,
        )
    tl.store(extend_start_loc + pid, cumsum_start)


def compute_position_torch(
    extend_prefix_lens: torch.Tensor, extend_seq_lens: torch.Tensor
):
    # There is a list comparison here, which causes CPU-GPU synchronization
    positions = torch.concat(
        [
            torch.arange(
                prefix_len, prefix_len + extend_len, device=extend_prefix_lens.device
            )
            for prefix_len, extend_len in zip(extend_prefix_lens, extend_seq_lens)
        ],
        axis=0,
    )
    extend_start_loc = torch.zeros_like(extend_seq_lens)
    extend_start_loc[1:] = torch.cumsum(extend_seq_lens[:-1], dim=0)
    return positions.to(torch.int64), extend_start_loc


@torch.compile(dynamic=True, backend=get_compiler_backend())
def clamp_position(seq_lens):
    return torch.clamp((seq_lens - 1), min=0).to(torch.int64)

@triton.jit
def get_out_cache_loc_kernel(
    out_cache_loc_ptr,
    req_to_token_ptr,
    req_pool_indices_ptr,
    new_compute_lens_ptr,
    cache_lens_ptr,
    req_to_token_ptr_stride: tl.constexpr,
):
    """
    Used during forward_batch initialization to determine KV cache write positions based on
    historical valid KV cache length and the number of tokens to compute this time.
    This process doesn't depend on the allocation logic in the scheduling process.
    During allocation, we only need to ensure that the slots allocated for requests are sufficient.

    This kernel should be executed on the forward_thread.

    Args:
        out_cache_loc_ptr: Pointer to store results, size is batch_size
        req_to_token_ptr: Pointer to get KV cache slot
        req_pool_indices_ptr: Pointer to get request index
        new_compute_lens_ptr: Pointer to get the number of tokens to compute this time
        cache_lens_ptr: Pointer to get historical KV cache length
        req_to_token_ptr_stride: Constant to get stride of req_to_token_ptr
    """
    pid = tl.program_id(0)
    BLOCK_SIZE: tl.constexpr = 512

    new_compute_len = tl.load(new_compute_lens_ptr + pid)
    req_index = tl.load(req_pool_indices_ptr + pid)
    cache_len = tl.load(cache_lens_ptr + req_index)
    req_to_token_start_loc = req_index * req_to_token_ptr_stride + cache_len

    num_loop = tl.cdiv(new_compute_len, BLOCK_SIZE)
    cumsum_start = tl.cast(0, tl.int32)
    for i in range(pid):
        cumsum_start += tl.load(new_compute_lens_ptr + i)
    
    # 0 means padding position
    if req_index == 0:
        for i in range(num_loop):
            offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
            mask = offset < new_compute_len
            zero_values = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
            tl.store(out_cache_loc_ptr + cumsum_start + offset, zero_values, mask=mask)
    else:
        for i in range(num_loop):
            offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
            mask = offset < new_compute_len
            data = tl.load(req_to_token_ptr + req_to_token_start_loc + offset, mask=mask)
            tl.store(
                out_cache_loc_ptr + cumsum_start + offset,
                data,
                mask=mask,
            )
