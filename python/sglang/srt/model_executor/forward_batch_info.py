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
import flash_npu_kernel
import triton.language as tl

from sglang.srt.env import ENV, global_server_args_dict
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.moe.npu_moe.ep_metadata import EPMetadata
from sglang.srt.utils import get_compiler_backend, split_array_by_half_sum, is_npu, check_memory_debug
from sglang.srt.distributed import get_ep_group

from sglang.srt.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

__is_npu__ = is_npu()

if __is_npu__:
    import torch_npu

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.layers.attention.npu_attn.flash_attn import AttentionMetadata
    from sglang.srt.managers.req import Req
    from sglang.srt.managers.schedule_batch import ModelWorkerBatch
    from sglang.srt.mem_cache.memory_pool import BaseTokenToKVPool, ReqToTokenPool
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
    from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput, EagleDraftOutput
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

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{type(self).__name__}.{self.name}"

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

class PPProxyTensors:
    # adapted from https://github.com/vllm-project/vllm/blob/d14e98d924724b284dc5eaf8070d935e214e50c0/vllm/sequence.py#L1103
    tensors: Dict[str, torch.Tensor]

    def __init__(self, tensors):
        # manually define this function, so that
        # Dynamo knows `IntermediateTensors()` comes from this file.
        # Otherwise, dataclass will generate this function by evaluating
        # a string, and we will lose the information about the source file.
        self.tensors = tensors

    def __getitem__(self, key: Union[str, slice]):
        if isinstance(key, str):
            return self.tensors[key]
        elif isinstance(key, slice):
            return self.__class__({k: v[key] for k, v in self.tensors.items()})

    def __setitem__(self, key: str, value: torch.Tensor):
        self.tensors[key] = value

    def __len__(self):
        return len(self.tensors)

    def __eq__(self, other: object):
        return isinstance(other, self.__class__) and self

    def __repr__(self) -> str:
        return f"PPProxyTensors(tensors={self.tensors})"


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
    global_sp_num_tokens: Optional[List[int]] = None
    gathered_buffer: Optional[torch.Tensor] = None
    all_decode_or_idle: bool = False
    can_run_tbo: bool = False

    # Speculative decoding
    spec_info: Optional[Union[EagleVerifyInput, EagleDraftInput, EagleDraftOutput]] = None
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

    exclude_prefill: Optional[bool] = False

    can_run_all2all: Optional[bool] = False

    can_run_with_graph: Optional[bool] = False

    captureing_prefill_graph: bool = False
    attn_metadata: AttentionMetadata = None

    ep_metadata: EPMetadata = None
    pp_proxy_tensors: Optional[PPProxyTensors] = None

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
            pp_proxy_tensors=batch.pp_proxy_tensors,
        )

        # ForwardBatch to EPMetadata
        if __is_npu__ and not ret.forward_mode.is_target_verify():
            ret.set_npu_ep_metadata(model_runner)

        if not __is_npu__:
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
            if not __is_npu__:
                ret.extend_seq_lens = torch.tensor(
                    batch.extend_seq_lens, dtype=torch.int32
                ).to(device, non_blocking=True)
                ret.extend_prefix_lens = torch.tensor(
                    batch.extend_prefix_lens, dtype=torch.int32
                ).to(device, non_blocking=True)
                ret.extend_num_tokens = batch.extend_num_tokens
                positions, ret.extend_start_loc = compute_position_triton(
                    ret.extend_prefix_lens,
                    ret.extend_seq_lens,
                    ret.extend_num_tokens,
                )
            else:
                extend_seq_lens = torch.tensor(
                    batch.extend_seq_lens, dtype=torch.int32, pin_memory=True
                )
                extend_prefix_lens = torch.tensor(
                    batch.extend_prefix_lens, dtype=torch.int32, pin_memory=True
                )
                ret.extend_seq_lens = extend_seq_lens.to(device, non_blocking=True)
                ret.extend_prefix_lens = extend_prefix_lens.to(device, non_blocking=True)
                ret.extend_num_tokens = batch.extend_num_tokens

                positions, ret.extend_start_loc = compute_position_torch(
                    extend_prefix_lens, extend_seq_lens, device
                )
            if ret.positions is None:
                ret.positions = positions
            ret.extend_prefix_lens_cpu = batch.extend_prefix_lens
            ret.extend_seq_lens_cpu = batch.extend_seq_lens
            ret.extend_logprob_start_lens_cpu = batch.extend_logprob_start_lens

        if model_runner.model_is_mrope:
            ret._compute_mrope_positions(model_runner, batch)

        ret.set_out_cache_loc()
        return ret

    def set_npu_ep_metadata(self, model_runner):
        assert __is_npu__, "set_npu_ep_metadata should only be called on NPU devices"

        if global_server_args_dict["npu_disable_all_gather"]:
            self.exclude_prefill = True
            self.can_run_all2all = ENV.npu_enable_all2all_comm
            self.all_decode_or_idle = True
        else:
            # ForwardBatch to EPMetadata
            decode_only_without_graph = self.forward_mode.is_decode() or self.forward_mode.is_idle()
            decode_only = decode_only_without_graph and ENV.npu_enable_graph
            if not (ENV.npu_enable_graph and self.all_decode_or_idle) and global_server_args_dict["disaggregation_mode"] != "prefill":
                if self.input_ids.shape[0] == 0:
                    ep_metadata = EPMetadata(model_runner.dtype, model_runner.model_config.hidden_size, 1,
                                            decode_only, decode_only_without_graph, self.input_ids.device)
                else:
                    ep_metadata = EPMetadata(model_runner.dtype, model_runner.model_config.hidden_size, self.input_ids.shape[0],
                                                decode_only, decode_only_without_graph, self.input_ids.device)
                self.ep_metadata = ep_metadata
            exclude_prefill = self.all_decode_or_idle
            can_run_all2all = exclude_prefill and ENV.npu_enable_all2all_comm
            self.exclude_prefill = exclude_prefill
            self.can_run_all2all = can_run_all2all

    def get_out_cache_loc_kernel_wrapper(self, bs, out_cache_loc):
        if not __is_npu__:
            get_out_cache_loc_kernel[(bs,)](
                out_cache_loc_ptr=out_cache_loc,
                req_to_token_ptr=self.req_to_token_pool.req_to_token,
                req_pool_indices_ptr=self.req_pool_indices,
                new_compute_lens_ptr=self.new_tokens_to_compute,
                cache_lens_ptr=self.req_to_token_pool.verified_lens,
                req_to_token_ptr_stride=self.req_to_token_pool.req_to_token.shape[1]
            )
        else:
            new_compute_lens = self.new_tokens_to_compute
            if ENV.npu_enable_get_out_cache:
                torch.ops.flash.npu_get_out_cache_loc(self.req_to_token_pool.req_to_token, self.req_pool_indices.to(torch.int32), new_compute_lens,
                                                self.req_to_token_pool.verified_lens, out_cache_loc, bs)
            else:
                cumsum_offsets = torch.zeros_like(new_compute_lens)
                torch.cumsum(new_compute_lens[:-1], dim=0, out=cumsum_offsets[1:])
                cache_starts = self.req_to_token_pool.verified_lens[self.req_pool_indices]
                row_indices = torch.repeat_interleave(self.req_pool_indices, new_compute_lens)
                col_indices = (cache_starts.repeat_interleave(new_compute_lens) +
                               torch.arange(new_compute_lens.sum(), device=new_compute_lens.device) -
                               cumsum_offsets.repeat_interleave(new_compute_lens))
                out_cache_loc[:col_indices.size(0)] = self.req_to_token_pool.req_to_token[row_indices, col_indices]

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

        if hasattr(self.token_to_kv_pool, "enable_mla_l1_5_cache") and self.token_to_kv_pool.enable_mla_l1_5_cache:
            # out_cache_loc will be modified inplace
            # nonlocal pages all set to page 0, local pages are remapped to local indices
            _, _, _, _, _ = self.token_to_kv_pool.global_loc_to_local_mapping(
                out_cache_loc
            )
            self.out_cache_loc = out_cache_loc
        else:
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

    def _pad_tensor_to_size(self, tensor: torch.Tensor, size: int, *, value: int = 0):
        if value == 0:
            return torch.cat(
                [tensor, tensor.new_zeros(size - tensor.shape[0], *tensor.shape[1:], dtype=tensor.dtype)],
                dim=0,
            )
        else:
            return torch.cat(
                [
                    tensor,
                    tensor.new_full((size - tensor.shape[0], *tensor.shape[1:]), value, dtype=tensor.dtype),
                ],
                dim=0,
            )

    def _pad_list_to_size(self, l: list, size: int):
        for _ in range(size - len(l)):
            l.append(0)
        return l

    def padding_for_graph_mtp(self, model_runner: ModelRunner, max_padding_size):
        global_num_tokens = self.global_num_tokens
        mtpn_factor = model_runner.server_args.speculative_num_steps + 1
        if not global_server_args_dict["npu_disable_all_gather"]:
            max_num_tokens = max(global_num_tokens)
            assert max_num_tokens <= max_padding_size * mtpn_factor, f"max num tokens ({max_num_tokens}) should less equal than max padding size({max_padding_size * mtpn_factor}) in graph mode"

        sync_group_size = get_ep_group().world_size
        global_num_tokens = [max_padding_size] * sync_group_size

        if self.forward_mode.is_target_verify():
            setattr(self, "ori_bs", self.batch_size)
            self.batch_size = max_padding_size

        if self.forward_mode.is_idle():
            self.extend_logprob_start_lens_cpu = []
            self.forward_mode = ForwardMode.TARGET_VERIFY
            setattr(self, "ori_bs", self.batch_size)
            self.batch_size = max_padding_size

            bs = self.batch_size
            self.extend_seq_lens = torch.full((bs,), 0, device="npu", dtype=torch.int32)
            self.new_tokens_to_compute = torch.full((bs,), mtpn_factor, device="npu", dtype=torch.int32)
            if model_runner.model_config.use_over_embedding:
                self.oe_token_table=model_runner.oe_token_table
                self.oe_column_starts=torch.zeros(bs, dtype=torch.int32, device="npu")
                self.oe_req_lens=torch.zeros(bs, dtype=torch.int32, device="npu")
                self.oe_out_column_starts=torch.zeros(bs, dtype=torch.int32, device="npu")
                self.oe_out_req_lens=torch.zeros(bs, dtype=torch.int32, device="npu")

            from sglang.srt.speculative.eagle_utils import EagleVerifyInput

            draft_token_num = model_runner.server_args.speculative_num_draft_tokens
            self.spec_info = EagleVerifyInput(
                draft_token=torch.zeros(bs * draft_token_num, device="npu"),
                positions=torch.zeros(bs * draft_token_num, device="npu"),

                draft_token_num=draft_token_num,
                spec_steps=model_runner.server_args.speculative_num_steps,
                capture_hidden_mode=CaptureHiddenMode.FULL,
                is_all_greedy=True,
                cumulated_scaling_penalties=torch.ones(
                    (bs, model_runner.model_config.vocab_size),
                    dtype=torch.float32,
                    device="npu",
                ),
            )
            self.capture_hidden_mode = CaptureHiddenMode.FULL
        else:
            bs = self.batch_size

            self.spec_info.cumulated_scaling_penalties = self._pad_tensor_to_size(
                self.spec_info.cumulated_scaling_penalties, bs, value=1)

        # padding
        if self.input_ids is None:
            self.input_ids = torch.empty(0, dtype=torch.int32, device=model_runner.device)
        self.input_ids = self._pad_tensor_to_size(self.input_ids, bs*mtpn_factor).to(torch.int64)
        self.spec_info.draft_token = self._pad_tensor_to_size(self.spec_info.draft_token, bs*mtpn_factor)

        self.req_pool_indices = self._pad_tensor_to_size(self.req_pool_indices, bs)

        seq_len_fill_value = (
            model_runner.attn_backend.get_cuda_graph_seq_len_fill_value()
        )
        self.seq_lens_sum = self.seq_lens_sum + seq_len_fill_value * (
            bs - self.seq_lens.shape[0]
        )
        self.seq_lens = self._pad_tensor_to_size(
            self.seq_lens, bs, value=seq_len_fill_value
        )
        if self.seq_lens_cpu is not None:
            self.seq_lens_cpu = self._pad_tensor_to_size(
                self.seq_lens_cpu, bs, value=seq_len_fill_value
            )

        self.out_cache_loc = self._pad_tensor_to_size(self.out_cache_loc, bs*mtpn_factor)
        self.positions = self._pad_tensor_to_size(self.positions, bs*mtpn_factor).to(torch.int32)
        self.global_num_tokens = global_num_tokens
        self.new_tokens_to_compute = self._pad_tensor_to_size(self.new_tokens_to_compute, bs)

        if self.extend_seq_lens is not None:
            self.extend_seq_lens = self._pad_tensor_to_size(self.extend_seq_lens, bs)

        self.extend_seq_lens_cpu=None

        self.extend_logprob_start_lens_cpu = [0] * bs  # bugfix

        if model_runner.model_config.use_over_embedding:
            self.oe_column_starts=self._pad_tensor_to_size(self.oe_column_starts, bs)
            self.oe_req_lens=self._pad_tensor_to_size(self.oe_req_lens, bs)
            self.oe_out_column_starts=self._pad_tensor_to_size(self.oe_out_column_starts, bs)
            self.oe_out_req_lens=self._pad_tensor_to_size(self.oe_out_req_lens, bs)

    def padding_for_npu_graph(self, model_runner: ModelRunner, max_padding_size):
        global_num_tokens = self.global_num_tokens
        sync_group_size = len(global_num_tokens)
        max_num_tokens = max(global_num_tokens)
        assert max_num_tokens <= max_padding_size, "max num tokens should less equal than max padding size in graph mode"
        global_num_tokens = [max_padding_size] * sync_group_size

        if self.forward_mode.is_idle():
            self.forward_mode = ForwardMode.DECODE

            if model_runner.model_config.use_over_embedding:
                self.oe_token_table=model_runner.oe_token_table
                self.oe_column_starts=torch.zeros(max_padding_size, dtype=torch.int32, device="npu")
                self.oe_req_lens=torch.zeros(max_padding_size, dtype=torch.int32, device="npu")
                self.oe_out_column_starts=torch.zeros(max_padding_size, dtype=torch.int32, device="npu")
                self.oe_out_req_lens=torch.zeros(max_padding_size, dtype=torch.int32, device="npu")

        if self.forward_mode.is_decode():
            setattr(self, "ori_bs", self.batch_size)
            self.batch_size = max_padding_size

        bs=self.batch_size

        # padding
        self.input_ids = self._pad_tensor_to_size(self.input_ids, bs)
        self.req_pool_indices = self._pad_tensor_to_size(self.req_pool_indices, bs)

        seq_len_fill_value = (
            model_runner.attn_backend.get_cuda_graph_seq_len_fill_value()
        )
        self.seq_lens_sum = self.seq_lens_sum + seq_len_fill_value * (
            bs - self.seq_lens.shape[0]
        )
        self.seq_lens = self._pad_tensor_to_size(
            self.seq_lens, bs, value=seq_len_fill_value
        )
        if self.seq_lens_cpu is not None:
            self.seq_lens_cpu = self._pad_tensor_to_size(
                self.seq_lens_cpu, bs, value=seq_len_fill_value
            )

        self.out_cache_loc = self._pad_tensor_to_size(self.out_cache_loc, bs)
        self.positions = self._pad_tensor_to_size(self.positions, bs).to(torch.int32)
        self.global_num_tokens = global_num_tokens

        if self.extend_seq_lens is not None:
            self.extend_seq_lens = self._pad_tensor_to_size(self.extend_seq_lens, bs)

        if model_runner.model_config.use_over_embedding:
            self.oe_column_starts=self._pad_tensor_to_size(self.oe_column_starts, bs)
            self.oe_req_lens=self._pad_tensor_to_size(self.oe_req_lens, bs)
            self.oe_out_column_starts=self._pad_tensor_to_size(self.oe_out_column_starts, bs)
            self.oe_out_req_lens=self._pad_tensor_to_size(self.oe_out_req_lens, bs)

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
    extend_prefix_lens: torch.Tensor, extend_seq_lens: torch.Tensor, device
):
    # There is a list comparison here, which causes CPU-GPU synchronization
    positions = torch.concat(
        [
            torch.arange(
                prefix_len, prefix_len + extend_len, device=device
            )
            for prefix_len, extend_len in zip(extend_prefix_lens, extend_seq_lens)
        ],
        axis=0,
    )
    extend_start_loc = torch.zeros_like(extend_seq_lens)
    extend_start_loc[1:] = torch.cumsum(extend_seq_lens[:-1], dim=0)
    return positions.to(torch.int64), extend_start_loc.pin_memory().to(device, non_blocking=True)

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
