from __future__ import annotations

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
Store information about requests and batches.

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

import dataclasses
import threading
from typing import TYPE_CHECKING, Callable, List, Optional, Union, Dict

import torch
import triton
import triton.language as tl

from sglang.global_config import global_config
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.disaggregation.decode_schedule_batch_mixin import (
    ScheduleBatchDisaggregationDecodeMixin,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.oe_utils import OverEmbeddingInfo
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.memory_pool import (
    BaseTokenToKVPool,
    ReqToTokenPool,
    HybridReqToTokenPool,
)
from sglang.srt.mem_cache.allocator import KVAllocator
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_colorful_logger
from sglang.srt.managers.req import Req


if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

from sglang.srt.env import global_server_args_dict

logger = get_colorful_logger(__name__)

bid = 0


def collect_group_specs(requests):
    from collections import defaultdict
    group_sizes = {}
    for req in requests:
        group_name, group_size, _ = req.get_group_specs()
        if group_name is None:
            continue
        if group_name in group_sizes:
            if group_sizes[group_name] != group_size:
                logger.warning(
                    f"Group '{group_name}' has inconsistent group_size: "
                    f"expected {group_sizes[group_name]}, got {group_size}"
                )
        else:
            group_sizes[group_name] = group_size
    return group_sizes


def filter_incomplete_groups(reqs, keep_indices):
    from collections import defaultdict

    if not keep_indices:
        return keep_indices

    group_counts = defaultdict(int)
    group_sizes = {}

    for idx in keep_indices:
        group_name, group_size, _ = reqs[idx].get_group_specs()
        if group_name is None:
            continue
        group_counts[group_name] += 1
        if group_name in group_sizes:
            if group_sizes[group_name] != group_size:
                logger.warning(
                    f"Group '{group_name}' has inconsistent group_size: "
                    f"expected {group_sizes[group_name]}, got {group_size}"
                )
        else:
            group_sizes[group_name] = group_size

    incomplete_groups = {
        name for name, count in group_counts.items()
        if count != group_sizes[name]
    }

    if not incomplete_groups:
        return keep_indices

    logger.warning(f"Incomplete Filter Found: {incomplete_groups=}, {reqs=}")
    filtered = []
    for idx in keep_indices:
        group_name, _, _ = reqs[idx].get_group_specs()
        if group_name is None or group_name not in incomplete_groups:
            filtered.append(idx)
    return filtered


@dataclasses.dataclass
class ScheduleBatch(ScheduleBatchDisaggregationDecodeMixin):
    """Store all information of a batch on the scheduler."""

    # Request, memory pool, and cache
    reqs: List[Req]
    req_to_token_pool: ReqToTokenPool = None
    kv_allocator: KVAllocator = None
    token_to_kv_pool: BaseTokenToKVPool = None
    tree_cache: BasePrefixCache = None

    # Batch configs
    model_config: ModelConfig = None
    forward_mode: ForwardMode = None
    enable_overlap: bool = False

    # Events
    launch_done: Optional[threading.Event] = None

    # Sampling info
    sampling_info: SamplingBatchInfo = None
    next_batch_sampling_info: SamplingBatchInfo = None

    # Batched arguments to model runner
    input_ids: torch.Tensor = None  # shape: [b], int32
    input_multi_ids: Optional[torch.Tensor] = None  # shape: [b, mm_heads], int32
    token_table: Optional[torch.Tensor] = None
    # oe_info contains all information needed for over embedding, corresponding to the OverEmbeddingInfo class.
    # Noted to avoid circular dependencies
    oe_info: Optional[OverEmbeddingInfo] = None
    draft_input_ids: torch.Tensor = None  # shape: [b], int32
    input_embeds: torch.Tensor = None  # shape: [b, hidden_size], float32
    req_pool_indices: torch.Tensor = None  # shape: [b], int32

    seq_lens: torch.Tensor = None  # shape: [b], int64
    output_ids: torch.Tensor = None  # shape: [b], int32
    output_multi_ids: torch.Tensor = None  # shape: [b], int32

    # The sum of all sequence lengths
    seq_lens_sum: int = None

    # For DP attention
    global_num_tokens: Optional[List[int]] = (
        None  # e.g. dp = 4, attn-tp = 2, [A, A, B, B, C, C, D, D]
    )
    global_num_tokens_for_logprob: Optional[List[int]] = None
    all_decode_or_idle: bool = False
    can_run_tbo: bool = False

    # For processing logprobs
    return_logprob: bool = False
    top_logprobs_nums: Optional[List[int]] = None
    token_ids_logprobs: Optional[List[List[int]]] = None

    # For extend and mixed chunekd prefill
    prefix_lens: List[int] = None
    extend_lens: List[int] = None
    extend_num_tokens: int = None
    decoding_reqs: List[Req] = None
    extend_logprob_start_lens: List[int] = None
    # It comes empty list if logprob is not required.
    extend_input_logprob_token_ids: Optional[torch.Tensor] = None

    # Stream
    has_stream: bool = False

    # Has grammar
    has_grammar: bool = False

    # Device
    device: str = "cuda"

    # Speculative decoding
    spec_algorithm: SpeculativeAlgorithm = None
    spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]] = None
    draft_token_num: Optional[int] = 0
    spec_num_steps: Optional[int] = 0
    # Reserve multiple positions for speculative decoding
    reserve_num_tokens_init: int = None

    # Enable custom logit processor
    enable_custom_logit_processor: bool = False

    # Whether to return hidden states
    return_hidden_states: bool = False

    global_batch_size: List[int] = None
    # set aux data for Disaggregation
    disagg_set_aux_fn: Optional[
        Callable[[torch.Tensor, LogitsProcessorOutput], None]
    ] = None
    # hicache pointer for synchronizing data loading from CPU to GPU
    hicache_consumer_index: int = -1

    @classmethod
    def init_new(
        cls,
        reqs: List[Req],
        req_to_token_pool: ReqToTokenPool,
        kv_allocator: KVAllocator,
        token_to_kv_pool: BaseTokenToKVPool,
        tree_cache: BasePrefixCache,
        model_config: ModelConfig,
        enable_overlap: bool,
        spec_algorithm: SpeculativeAlgorithm,
        enable_custom_logit_processor: bool,
        reserve_num_tokens_init: int = 0,
        draft_token_num: int = 0,
        spec_num_steps: int = 0,
    ):
        return cls(
            reqs=reqs,
            req_to_token_pool=req_to_token_pool,
            kv_allocator=kv_allocator,
            token_to_kv_pool=token_to_kv_pool,
            tree_cache=tree_cache,
            model_config=model_config,
            enable_overlap=enable_overlap,
            return_logprob=any(req.return_logprob for req in reqs),
            has_stream=any(req.stream for req in reqs),
            has_grammar=any(req.grammar for req in reqs),
            device=req_to_token_pool.device,
            spec_algorithm=spec_algorithm,
            enable_custom_logit_processor=enable_custom_logit_processor,
            return_hidden_states=any(req.return_hidden_states for req in reqs),
            reserve_num_tokens_init=reserve_num_tokens_init,
            draft_token_num=draft_token_num,
            spec_num_steps=spec_num_steps,
        )

    def batch_size(self):
        return len(self.reqs)

    def is_empty(self):
        return len(self.reqs) == 0

    def alloc_req_slots(self, num_reqs: int):
        if isinstance(self.req_to_token_pool, HybridReqToTokenPool):
            req_pool_indices = self.req_to_token_pool.alloc(num_reqs, self.reqs)
        else:
            req_pool_indices = self.req_to_token_pool.alloc(num_reqs)
        if req_pool_indices is None:
            raise RuntimeError(
                "alloc_req_slots runs out of memory. "
                "Please set a smaller number for `--max-running-requests`. "
                f"{self.req_to_token_pool.available_size()=}, "
                f"{num_reqs=}, "
            )
        return req_pool_indices

    def alloc_token_slots(self, req_pool_index: int, num_tokens: int):
        out_cache_loc = self.kv_allocator.alloc(req_pool_index, num_tokens, self.req_to_token_pool.alloced_lens[req_pool_index].item())

        if out_cache_loc is None:
            if self.tree_cache is not None:
                logger.debug(
                    f"[evict] before evict evict_tokens={num_tokens} evictable_size={self.tree_cache.evictable_size()}"
                )
                need_page_num = (
                    num_tokens + self.kv_allocator.page_size - 1
                ) // self.kv_allocator.page_size
                self.tree_cache.evict(need_page_num, self.kv_allocator.free)
                logger.debug(
                    f"[evict] after evict evictable_size={self.tree_cache.evictable_size()}"
                )
                out_cache_loc = self.kv_allocator.alloc(req_pool_index, num_tokens, self.req_to_token_pool.alloced_lens[req_pool_index].item())
                logger.debug(
                    f"[evict] {out_cache_loc=} after evict"
                )

            if out_cache_loc is None:
                phase_str = "Prefill" if self.forward_mode.is_extend() else "Decode"
                logger.error(
                    f"{phase_str} out of memory. Try to lower your batch size.\n"
                    f"Try to allocate {num_tokens} tokens.\n"
                    f"Avaliable tokens: {self.kv_allocator.available_size() + self.tree_cache.evictable_size()}\n"
                )
                if self.tree_cache is not None:
                    self.tree_cache.pretty_print()
                exit(1)

        return out_cache_loc

    def assign_req_to_token_pool_wrapper(self, bs, extend_lens, out_cache_loc):
        if (
            global_server_args_dict["attention_backend"] != "torch_native"
            and global_server_args_dict["attention_backend"] != "npu_mla"
        ) and global_server_args_dict["attention_backend"] != "torch_native_mla":
            assign_req_to_token_pool[(bs,)](
                self.req_pool_indices,
                self.req_to_token_pool.req_to_token,
                self.req_to_token_pool.alloced_lens[self.req_pool_indices],
                self.req_to_token_pool.alloced_lens[self.req_pool_indices]
                + extend_lens,
                out_cache_loc,
                self.req_to_token_pool.req_to_token.shape[1],
                triton.next_power_of_2(bs),
            )
        else:
            pt = 0
            for i in range(bs):
                alloced_len = self.req_to_token_pool.alloced_lens[
                    self.req_pool_indices[i]
                ]
                end_loc = alloced_len + extend_lens[i]
                self.req_to_token_pool.write(
                    (self.req_pool_indices[i], slice(alloced_len, end_loc)),
                    out_cache_loc[pt : pt + extend_lens[i]],
                )
                pt += extend_lens[i]

    def prepare_for_extend(self):
        self.forward_mode = ForwardMode.EXTEND

        bs = len(self.reqs)
        reqs = self.reqs
        # Without speculation, reserve_num_tokens_init is 0
        for req in reqs:
            req.reserve_num_tokens = self.reserve_num_tokens_init
        input_ids = [r.fill_ids[r.prefix_len :] for r in reqs]
        if "multi_ids" in global_server_args_dict["mm_mode"]:
            input_multi_ids = [r.fill_multi_ids[r.prefix_len :] for r in reqs]
        else:
            input_multi_ids = None
        # The reason we need draft_fill_ids is that Eagle needs input_ids[1:] + base model's first
        # output token as prefill input. Considering chunked prefill, doing this in Eagle's
        # prepare_for_extend phase like SGLang is incorrect.
        draft_input_ids = [r.draft_fill_ids[r.prefix_len :] for r in reqs]
        extend_num_tokens = sum(len(ids) for ids in input_ids)
        seq_lens = []
        pre_lens = []

        for req in reqs:
            if req.is_retracted:
                assert req.req_pool_idx is None, "retracted req's req_pool_idx should be None!"

        # Allocate memory
        no_chunk_bs = len([req for req in reqs if req.req_pool_idx is None])
        # only assign new req's pool index
        req_pool_indices = self.alloc_req_slots(no_chunk_bs)

        batch_input_embeds = []
        extend_input_logprob_token_ids = []
        out_cache_loc_list = []

        token_level_offsets = torch.arange(self.kv_allocator.page_size, device=self.device)
        for i, req in enumerate(reqs):
            if req.req_pool_idx is None:
                # if req is not chunked, assign a new req_pool_idx
                req.req_pool_idx = req_pool_indices[i]
                req.oe_init = False
            else:
                req_pool_indices.insert(i, req.req_pool_idx)
            pre_len, seq_len = req.prefix_len, len(req.fill_ids)
            if pre_len > 0:
                # prefix_page_ids is got from `init_next_round_input`
                self.kv_allocator.req_to_page[
                    req.req_pool_idx, : len(req.prefix_page_ids)
                ] = req.prefix_page_ids
                start_locs = req.prefix_page_ids * self.kv_allocator.page_size
                prefix_token_slots = (
                    start_locs[:, None] + token_level_offsets
                ).flatten()
                self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, : len(req.prefix_page_ids) * self.kv_allocator.page_size
                ] = prefix_token_slots
                logger.debug(
                    f"{req.prefix_page_ids=} {pre_len=} {self.req_to_token_pool.alloced_lens[req.req_pool_idx]=}"
                )
                self.req_to_token_pool.alloced_lens[req.req_pool_idx] = pre_len

            out_cache_loc = self.alloc_token_slots(req.req_pool_idx, seq_len - pre_len)
            out_cache_loc_list.append(out_cache_loc)

            seq_lens.append(seq_len)
            assert seq_len - pre_len == req.extend_input_len, (
                f"{req.rid=}, {seq_len=}, {pre_len=}, {req.extend_input_len=} {req.prefix_len=} {len(req.prefix_page_ids)=}"
            )

            # If input_embeds are available, store them
            if req.fill_input_embeds is not None:
                chunked_input_embeds = req.fill_input_embeds[req.prefix_len:]
                batch_input_embeds.extend(chunked_input_embeds)

            # Calculate cached_tokens: include device cache hit, host cache hit, and storage prefetch hit
            device_cache_hit = pre_len
            host_cache_hit = req.host_hit_length

            req.cached_tokens += device_cache_hit - req.already_computed # TODO: need to complete the cache length refresh logic
            logger.debug(f"Cache hit: rid={req.rid} device={device_cache_hit} host={host_cache_hit} {req.already_computed=} cached={req.cached_tokens} input_len={len(req.origin_input_ids)}")

            req.already_computed = seq_len
            req.is_retracted = False
            pre_lens.append(pre_len)
            # Compute the relative logprob_start_len in an extend batch
            if req.logprob_start_len >= pre_len:
                req.extend_logprob_start_len = min(
                    req.logprob_start_len - pre_len,
                    req.extend_input_len,
                    req.seqlen - 1,
                )
            else:
                req.extend_logprob_start_len = 0

            if self.return_logprob:
                # Find input logprob token ids.
                # First, find a global index within origin_input_ids and slide it by 1
                # to compute input logprobs. It is because you need the next token
                # to compute input logprobs. E.g., (chunk size 2)
                #
                # input_logprobs = [1, 2, 3, 4]
                # fill_ids = [1, 2]
                # extend_input_logprob_token_id = [2, 3]
                #
                # Note that it can also overflow. In this case, we pad it with 0.
                # input_logprobs = [1, 2, 3, 4]
                # fill_ids = [3, 4]
                # extend_input_logprob_token_id = [4, 0]
                global_start_idx, global_end_idx = (
                    req.prefix_len,
                    len(req.fill_ids),
                )
                # Apply logprob_start_len
                if global_start_idx < req.logprob_start_len:
                    global_start_idx = req.logprob_start_len

                logprob_token_ids = req.origin_input_ids[
                    global_start_idx + 1 : global_end_idx + 1
                ]
                extend_input_logprob_token_ids.extend(logprob_token_ids)

                # We will need req.extend_input_len - req.extend_logprob_start_len number of
                # tokens, and logprob_token_ids is for input logprob, so pad the rest of them by 0.
                extend_input_logprob_token_ids.extend(
                    [0]
                    * (
                        req.extend_input_len
                        - req.extend_logprob_start_len
                        - len(logprob_token_ids)
                    )
                )

        if self.return_logprob:
            extend_input_logprob_token_ids = torch.tensor(
                extend_input_logprob_token_ids
            )
        else:
            extend_input_logprob_token_ids = None

        # Set fields
        self.input_ids = torch.tensor(sum(input_ids, []), dtype=torch.int32).to(
            self.device, non_blocking=True
        )
        if input_multi_ids is not None:
            self.input_multi_ids = torch.tensor(sum(input_multi_ids, []), dtype=torch.int32).to(
                self.device, non_blocking=True
            )
        self.draft_input_ids = [
            torch.tensor(tokens, dtype=torch.int32).to(self.device, non_blocking=True)
            for tokens in draft_input_ids
        ]
        self.req_pool_indices = torch.tensor(req_pool_indices, dtype=torch.int64).to(
            self.device, non_blocking=True
        )
        self.seq_lens = torch.tensor(seq_lens, dtype=torch.int64).to(
            self.device, non_blocking=True
        )
        if batch_input_embeds:
            self.input_embeds = torch.tensor(batch_input_embeds).to(
                self.device, non_blocking=True
            )
            assert len(self.input_embeds.shape) == 2, f"{self.input_embeds.shape}\n{batch_input_embeds}"
        else:
            self.input_embeds = None

        out_cache_loc = torch.concat(out_cache_loc_list).to(
            self.device, non_blocking=True
        )
        self.seq_lens_sum = sum(seq_lens)
        if self.return_logprob:
            self.top_logprobs_nums = [r.top_logprobs_num for r in reqs]
            self.token_ids_logprobs = [r.token_ids_logprob for r in reqs]
        self.extend_num_tokens = extend_num_tokens
        self.prefix_lens = [r.prefix_len for r in reqs]
        self.extend_lens = [r.extend_input_len for r in reqs]
        self.extend_logprob_start_lens = [r.extend_logprob_start_len for r in reqs]
        self.extend_input_logprob_token_ids = extend_input_logprob_token_ids

        # Write to req_to_token_pool
        pre_lens = torch.tensor(pre_lens, dtype=torch.int32).to(
            self.device, non_blocking=True
        )
        extend_lens = torch.tensor(self.extend_lens, dtype=torch.int32).to(
            self.device, non_blocking=True
        )
        self.assign_req_to_token_pool_wrapper(bs, extend_lens, out_cache_loc)
        # TODO: some tensors can be reused for ForwardBatchInfo (e.g., extend_lens, cumsum_start)

        # Update alloced_lens
        self.req_to_token_pool.alloced_lens[self.req_pool_indices] += extend_lens
        if self.spec_algorithm.is_eagle():
            # Pre-allocate a segment of positions for draft decode
            self.prealloc_for_draft_decode()

        # Build sampling info
        self.sampling_info = SamplingBatchInfo.from_schedule_batch(
            self,
            self.model_config.vocab_size,
        )

    def mix_with_running(self, running_batch: "ScheduleBatch"):
        self.forward_mode = ForwardMode.MIXED
        running_bs = running_batch.batch_size()

        for req in running_batch.reqs:
            req.fill_ids = req.origin_input_ids + req.output_ids
            if "multi_ids" in global_server_args_dict["mm_mode"]:
                req.fill_multi_ids = req.origin_input_multi_ids + req.output_multi_ids
            req.extend_input_len = 1

        input_ids = torch.cat([self.input_ids, running_batch.input_ids])
        out_cache_loc = torch.cat([self.out_cache_loc, running_batch.out_cache_loc])

        self.merge_batch(running_batch)
        self.input_ids = input_ids
        if "multi_ids" in global_server_args_dict["mm_mode"]:
            self.input_multi_ids = torch.cat([self.input_multi_ids, running_batch.input_multi_ids])
        self.out_cache_loc = out_cache_loc

        # For overlap scheduler, the output_ids has one step delay
        delta = 0 if self.enable_overlap else -1

        # NOTE: prefix_len is the length of what has been cached
        # NOTE: but we don't cache at each decode step
        self.prefix_lens.extend(
            [
                len(r.origin_input_ids) + len(r.output_ids) + delta
                for r in running_batch.reqs
            ]
        )
        self.extend_lens.extend([1] * running_bs)
        self.extend_num_tokens += running_bs
        # TODO (lianmin): Revisit this. It should be seq_len - 1
        self.extend_logprob_start_lens.extend([0] * running_bs)

    def check_decode_mem(self, buf_multiplier=1):
        bs = len(self.reqs) * buf_multiplier
        if self.kv_allocator.available_pages() >= bs:
            return True

        self.tree_cache.evict(bs, self.kv_allocator.free)

        if self.kv_allocator.available_pages() >= bs:
            return True

        return False

    def retract_decode(self, server_args: ServerArgs, delay_req_pool_release=False):
        """Retract the decoding requests when there is not enough memory."""
        sorted_indices = [i for i in range(len(self.reqs))]
        # TODO(lsyin): improve retraction policy for radix cache
        # For spec decoding, filter_batch API can only filter
        # requests from the back, so we can only retract from the back.
        # TODO(sang): Clean up finish path and support better retract
        # policy.
        if not server_args.speculative_algorithm:
            sorted_indices.sort(
                key=lambda i: (
                    len(self.reqs[i].output_ids),
                    -len(self.reqs[i].origin_input_ids),
                ),
                reverse=True,
            )

        def get_required_pages(num_reqs: int):
            headroom_for_spec_decode = 0
            if server_args.speculative_algorithm:
                if server_args.speculative_algorithm == "PLD":
                    # PLD only needs memory for speculative_num_draft_tokens
                    headroom_for_spec_decode += server_args.speculative_num_draft_tokens
                else: # for EAGLE
                    headroom_for_spec_decode += (
                        server_args.speculative_eagle_topk
                        * server_args.speculative_num_steps
                        + server_args.speculative_num_draft_tokens
                    )

            num_pages = (
                global_config.retract_decode_steps
                + headroom_for_spec_decode
                + self.kv_allocator.page_size
                - 1
            ) // self.kv_allocator.page_size
            return num_reqs * num_pages

        retracted_reqs = []
        seq_lens_cpu = self.seq_lens.cpu().numpy()
        first_iter = True
        while (
            self.kv_allocator.available_pages()
            < get_required_pages(len(sorted_indices))
            or first_iter
        ):
            if len(sorted_indices) == 1:
                # Corner case: only one request left
                assert self.kv_allocator.available_pages() > 0, (
                    "No space left for only one request"
                )
                break

            first_iter = False
            idx = sorted_indices.pop()
            req = self.reqs[idx]
            retracted_reqs.append(req)

            if server_args.disaggregation_mode == "decode":
                logger.info(f"[ScheduleBatch][retract_decode] retracting {req.rid}")
                req.offload_kv_cache(self.req_to_token_pool, self.token_to_kv_pool)

            if isinstance(self.tree_cache, ChunkCache):
                # Deprecated
                token_indices = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, : seq_lens_cpu[idx]
                ]
                self.kv_allocator.free(req.req_pool_idx, token_indices)
                self.req_to_token_pool.free(req.req_pool_idx)
                if req.rid in self.tree_cache.entries:
                    del self.tree_cache.entries[req.rid]
            else:
                # TODO: apply more fine-grained retraction
                page_size = self.kv_allocator.page_size
                logger.info(
                    f"retracting, rid={req.rid}, req_pool_idx={req.req_pool_idx} {req.prefix_len=}, {seq_lens_cpu[idx]=}"
                )
                # release the last node in this function
                self.tree_cache.cache_finished_req(req, delay_req_pool_release=delay_req_pool_release)

                # NOTE(lsyin): we should use the newly evictable memory instantly.
                def ceil_dev(a: int, b: int):
                    return (a + b - 1) // b

                residual_size = (
                    ceil_dev(
                        len(sorted_indices) * global_config.retract_decode_steps,
                        page_size,
                    )
                    - self.kv_allocator.available_size()
                )
                residual_size = max(0, residual_size)
                self.tree_cache.evict(residual_size, None)
                self.kv_allocator.free_group_end()
            req.reset_for_retract(delay_req_pool_release=delay_req_pool_release)

            if len(retracted_reqs) == 0:
                # Corner case: only one request left
                raise ValueError(
                    "Failed to retract any request. No space left for only one request."
                )

        self.filter_batch(keep_indices=sorted_indices)

        # Reqs in batch are filtered
        total_decoded_tokens = sum(len(r.output_ids) for r in self.reqs)
        total_max_new_tokens = sum(r.sampling_params.max_new_tokens for r in self.reqs)

        new_estimate_ratio = (
            total_decoded_tokens + global_config.retract_decode_steps * len(self.reqs)
        ) / total_max_new_tokens
        new_estimate_ratio = min(1.0, new_estimate_ratio)

        return retracted_reqs, new_estimate_ratio

    def prepare_for_idle(self):
        self.forward_mode = ForwardMode.IDLE
        self.input_ids = torch.empty(0, dtype=torch.int32, device=self.device)
        self.input_multi_ids = torch.empty((0, 1), dtype=torch.int32).to(self.device, non_blocking=True)
        self.seq_lens = torch.empty(0, dtype=torch.int64, device=self.device)
        self.req_pool_indices = torch.empty(0, dtype=torch.int64, device=self.device)
        self.seq_lens_sum = 0
        self.extend_num_tokens = 0
        self.sampling_info = SamplingBatchInfo.from_schedule_batch(
            self,
            self.model_config.vocab_size,
        )

    def prepare_for_decode(self):
        # Happens in get_next_batch_to_run, overlapped with model forward, one extra token of KV cache is allocated below
        self.forward_mode = ForwardMode.DECODE
        if self.sampling_info.penalizer_orchestrator.is_required:
            if self.enable_overlap:
                # TODO: this can be slow, optimize this.
                delayed_output_ids = torch.tensor(
                    [
                        (
                            req.output_ids[-1]
                            if len(req.output_ids)
                            else req.origin_input_ids[-1]
                        )
                        for req in self.reqs
                    ],
                    dtype=torch.int64,
                    device=self.device,
                )
                self.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                    delayed_output_ids
                )
            else:
                self.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                    self.output_ids.to(torch.int64)
                )

        self.input_ids = self.output_ids
        self.input_multi_ids = self.output_multi_ids
        self.output_ids = None
        self.output_multi_ids = None

        # Alloc mem
        out_cache_loc_list = []
        # 1 is for the original decode input token, reserve_num_tokens is for speculative decode
        # These positions may be overwritten in the next iteration due to rejection.
        # The number of positions to reserve each time is modified based on the acceptance length.
        num_alloced_tokens_list = []
        for i, req in enumerate(self.reqs):
            if self.spec_algorithm.is_none():
                out_cache_loc_list.append(self.alloc_token_slots(req.req_pool_idx, 1))
                num_alloced_tokens_list.append(1)
            else:
                # For PD warmup
                if req.reserve_num_tokens == 0:
                    req.reserve_num_tokens = self.draft_token_num
                if (
                    self.req_to_token_pool.alloced_lens[req.req_pool_idx].item()
                    > self.model_config.context_len + self.draft_token_num
                ):
                    # Allocation has reached the maximum length the current request may use, no more allocation
                    out_cache_loc_list.append(
                        self.alloc_token_slots(req.req_pool_idx, 0)
                    )
                    num_alloced_tokens_list.append(0)
                    continue
                out_cache_loc_list.append(
                    self.alloc_token_slots(req.req_pool_idx, req.reserve_num_tokens)
                )
                num_alloced_tokens_list.append(req.reserve_num_tokens)
        num_alloced_tokens_device = torch.tensor(
            num_alloced_tokens_list, dtype=torch.int64, pin_memory=True
        )
        num_alloced_tokens_device = num_alloced_tokens_device.to(
            self.device, non_blocking=True
        )
        out_cache_loc = torch.concat(out_cache_loc_list)
        out_cache_loc = out_cache_loc.to(self.device, non_blocking=True)

        bs = len(self.reqs)
        self.assign_req_to_token_pool_wrapper(
            bs, num_alloced_tokens_device, out_cache_loc
        )
        self.req_to_token_pool.alloced_lens[
            self.req_pool_indices
        ] += num_alloced_tokens_device
        if self.spec_algorithm.is_none():
            if self.enable_overlap:
                # Do not use in-place operations in the overlap mode
                self.seq_lens = self.seq_lens + 1
            else:
                # A faster in-place version
                self.seq_lens.add_(1)
            self.seq_lens_sum += len(self.reqs)

    def prealloc_for_draft_decode(self, is_disaggregation_decode: bool = False):
        """Pre-allocate a segment of slots for draft decode"""
        if self.enable_overlap:
            # Conceptually, each allocation during speculation + overlap is preparing for the next batch's launch.
            # Therefore, at the beginning, reserve enough space at the end of prefill for the next round's verify and draft decode.
            # Then, each time adjust the reserved space based on acceptance length to prevent allocation divergence causing insufficient space.
            # The reserved space for draft decode will always be overwritten by valid tokens in the next verify.
            # Initially allocate spec_num_steps, subsequent allocations are not needed.
            num_tokens_pre_alloc = self.draft_token_num + (self.spec_num_steps - 1)
        else:
            # Synchronously, each allocation is for the current batch's launch. Here we allocate spec_num_steps
            # extra slots to reserve enough space for draft decode.
            if self.spec_num_steps > 1:
                num_tokens_pre_alloc = self.spec_num_steps - 1
            else:
                return
        out_cache_loc_list = []
        req_indices = []
        for i, req in enumerate(self.reqs):
            # TODO: consider tree attention
            # End of prefill or PD disaggregation mocked prefill
            if req.draft_fill_ids[-1] == -1 or is_disaggregation_decode:
                out_cache_loc_list.append(
                    self.alloc_token_slots(req.req_pool_idx, num_tokens_pre_alloc)
                )
                req_indices.append(req.req_pool_idx)
        bs = len(req_indices)
        if len(out_cache_loc_list) == 0:
            return
        out_cache_loc = torch.concat(out_cache_loc_list)
        out_cache_loc = out_cache_loc.to(self.device, non_blocking=True)
        req_indices = torch.tensor(req_indices, dtype=torch.int32).to(
            self.device, non_blocking=True
        )
        start_offsets = self.req_to_token_pool.alloced_lens[req_indices]
        end_offsets = start_offsets + num_tokens_pre_alloc
        assign_req_to_token_pool[(bs,)](
            req_indices,
            self.req_to_token_pool.req_to_token,
            start_offsets,
            end_offsets,
            out_cache_loc,
            self.req_to_token_pool.req_to_token.shape[1],
            triton.next_power_of_2(bs),
        )
        self.req_to_token_pool.alloced_lens[req_indices] += num_tokens_pre_alloc

    def filter_batch(
        self,
        chunked_req_to_exclude: Optional[Req] = None,
        keep_indices: Optional[List[int]] = None,
    ):
        if keep_indices is None:
            keep_indices = [
                i
                for i in range(len(self.reqs))
                if not self.reqs[i].finished()
                and self.reqs[i] is not chunked_req_to_exclude
            ]

            keep_indices = filter_incomplete_groups(self.reqs, keep_indices)

        if keep_indices is None or len(keep_indices) == 0:
            # Filter out all requests
            self.reqs = []
            return

        if len(keep_indices) == len(self.reqs):
            # No need to filter
            return

        keep_indices_device = torch.tensor(keep_indices, dtype=torch.int64).to(
            self.device, non_blocking=True
        )

        self.reqs = [self.reqs[i] for i in keep_indices]
        self.req_pool_indices = self.req_pool_indices[keep_indices_device]
        self.seq_lens = self.seq_lens[keep_indices_device]
        self.seq_lens_sum = self.seq_lens.sum().item()
        if self.output_ids is not None:
            self.output_ids = self.output_ids[keep_indices_device]
        if self.output_multi_ids is not None:
            self.output_multi_ids = self.output_multi_ids[keep_indices_device]
        self.return_logprob = any(req.return_logprob for req in self.reqs)
        if self.return_logprob:
            self.top_logprobs_nums = [self.top_logprobs_nums[i] for i in keep_indices]
            self.token_ids_logprobs = [self.token_ids_logprobs[i] for i in keep_indices]
        else:
            self.top_logprobs_nums = None
            self.token_ids_logprobs = None

        self.has_stream = any(req.stream for req in self.reqs)
        self.has_grammar = any(req.grammar for req in self.reqs)

        self.sampling_info.filter_batch(keep_indices, keep_indices_device)
        if self.spec_info:
            self.spec_info.filter_batch(keep_indices_device)

    def update_reserve_num_tokens(self, accept_lengths_cpu: List[int]):
        """Update the number of slots to allocate based on the last acceptance length"""
        if self.spec_algorithm.is_none():
            return
        bs = len(self.reqs)
        if bs == 0:
            return
        for i, req in enumerate(self.reqs):
            reuse_slots_num = self.draft_token_num - accept_lengths_cpu[i]
            req.reserve_num_tokens = self.draft_token_num - reuse_slots_num

    def merge_batch(self, other: "ScheduleBatch"):
        # Penalizer orchestrator must be merged before Batch.reqs is merged. This is because
        # orchestrator.merge() depends on Batch.reqs during preparation of each penalizers, so it
        # needs to be called with pre-merged Batch.reqs.
        self.sampling_info.merge_batch(other.sampling_info)

        self.req_pool_indices = torch.concat(
            [self.req_pool_indices, other.req_pool_indices]
        )
        self.seq_lens = torch.concat([self.seq_lens, other.seq_lens])
        self.seq_lens_sum += other.seq_lens_sum
        if self.output_ids is not None:
            self.output_ids = torch.concat([self.output_ids, other.output_ids])
        if self.output_multi_ids is not None:
            self.output_multi_ids = torch.concat([self.output_multi_ids, other.output_multi_ids])
        if self.return_logprob and other.return_logprob:
            self.top_logprobs_nums.extend(other.top_logprobs_nums)
            self.token_ids_logprobs.extend(other.token_ids_logprobs)
        elif self.return_logprob:
            self.top_logprobs_nums.extend([0] * len(other.reqs))
            self.token_ids_logprobs.extend([None] * len(other.reqs))
        elif other.return_logprob:
            self.top_logprobs_nums = [0] * len(self.reqs) + other.top_logprobs_nums
            self.token_ids_logprobs = [None] * len(self.reqs) + other.token_ids_logprobs
        self.reqs.extend(other.reqs)

        self.return_logprob |= other.return_logprob
        self.has_stream |= other.has_stream
        self.has_grammar |= other.has_grammar
        self.return_hidden_states |= other.return_hidden_states

        if self.spec_info:
            self.spec_info.merge_batch(other.spec_info)

    def get_model_worker_batch(self):
        if self.forward_mode.is_decode_or_idle():
            extend_seq_lens = extend_prefix_lens = extend_logprob_start_lens = None
        else:
            extend_seq_lens = self.extend_lens
            extend_prefix_lens = self.prefix_lens
            extend_logprob_start_lens = self.extend_logprob_start_lens

        seq_lens_cpu = self.seq_lens.cpu()

        if self.sampling_info:
            if self.has_grammar:
                self.sampling_info.grammars = [req.grammar for req in self.reqs]
            else:
                self.sampling_info.grammars = None

        self.set_new_tokens_info()

        global bid
        bid += 1
        return ModelWorkerBatch(
            bid=bid,
            forward_mode=self.forward_mode,
            input_ids=self.input_ids,
            draft_input_ids=self.draft_input_ids,
            req_pool_indices=self.req_pool_indices,
            seq_lens=self.seq_lens,
            seq_lens_sum=self.seq_lens_sum,
            return_logprob=self.return_logprob,
            top_logprobs_nums=self.top_logprobs_nums,
            token_ids_logprobs=self.token_ids_logprobs,
            global_num_tokens=self.global_num_tokens,
            global_num_tokens_for_logprob=self.global_num_tokens_for_logprob,
            all_decode_or_idle=self.all_decode_or_idle,
            can_run_tbo=self.can_run_tbo,
            seq_lens_cpu=seq_lens_cpu,
            extend_num_tokens=self.extend_num_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_logprob_start_lens=extend_logprob_start_lens,
            sampling_info=self.sampling_info,
            input_embeds=self.input_embeds,
            oe_token_table=self.token_table,
            spec_algorithm=self.spec_algorithm,
            spec_info=self.spec_info,
            hicache_consumer_index=self.hicache_consumer_index,
            capture_hidden_mode=(
                CaptureHiddenMode.FULL
                if self.return_hidden_states
                else (
                    getattr(
                        self.spec_info, "capture_hidden_mode", CaptureHiddenMode.NULL
                    )
                    if self.spec_info
                    else CaptureHiddenMode.NULL
                )
            ),
            extend_input_logprob_token_ids=self.extend_input_logprob_token_ids,
            new_tokens_to_compute=self.new_tokens_to_compute,
            new_tokens_total=self.new_tokens_total,
            global_batch_size=self.global_batch_size,
            launch_done=self.launch_done,
            disagg_set_aux_fn=self.disagg_set_aux_fn,
            input_multi_ids=self.input_multi_ids,
            reqs=self.reqs,
        )

    def set_new_tokens_info(self):
        batch_size = len(self.reqs)
        if self.forward_mode.is_extend():
            self._set_extend_new_tokens_info(batch_size)
        elif self.forward_mode.is_decode():
            self._set_decode_new_tokens_info(batch_size)
        elif self.forward_mode.is_idle():
            self._set_idle_new_tokens_info()
        else:
            raise RuntimeError(f"Not Supported forward model: {self.forward_mode}")

    def _set_extend_new_tokens_info(self, batch_size):
        if self.forward_mode.is_target_verify():
            tokens_list = [self.draft_token_num] * batch_size
            self.new_tokens_to_compute = torch.tensor(
                tokens_list, dtype=torch.int32, pin_memory=True
            ).to(self.device, non_blocking=True)
            self.new_tokens_total = sum(tokens_list)
        else:
            extend_lens = torch.tensor(self.extend_lens, dtype=torch.int32).to(
                self.device, non_blocking=True
            )
            self.new_tokens_to_compute = extend_lens
            self.new_tokens_total = self.extend_num_tokens

    def _set_decode_new_tokens_info(self, batch_size):
        tokens_list = [1] * batch_size
        self.new_tokens_to_compute = torch.tensor(
            tokens_list, device=self.device, dtype=torch.int32
        )
        self.new_tokens_total = sum(tokens_list)

    def _set_idle_new_tokens_info(self):
        self.new_tokens_to_compute = torch.tensor(
            [], device=self.device, dtype=torch.int32
        )
        self.new_tokens_total = 0

    def copy(self):
        # Only contain fields that will be used by process_batch_result
        return ScheduleBatch(
            reqs=self.reqs,
            model_config=self.model_config,
            forward_mode=self.forward_mode,
            return_logprob=self.return_logprob,
            decoding_reqs=self.decoding_reqs,
            spec_algorithm=self.spec_algorithm,
            enable_custom_logit_processor=self.enable_custom_logit_processor,
            draft_token_num=self.draft_token_num,
        )

    def __str__(self):
        return (
            f"ScheduleBatch(forward_mode={self.forward_mode.name}, "
            f"#req={(len(self.reqs))})"
        )


@dataclasses.dataclass
class ModelWorkerBatch:
    # The batch id
    bid: int
    # The forward mode
    forward_mode: ForwardMode
    # The input ids
    input_ids: torch.Tensor
    draft_input_ids: torch.Tensor
    # The indices of requests in the req_to_token_pool
    req_pool_indices: torch.Tensor
    # The sequence length
    seq_lens: torch.Tensor
    seq_lens_cpu: Optional[torch.Tensor]

    # Tokens that need to be computed
    new_tokens_to_compute: torch.Tensor
    new_tokens_total: int

    # The sum of all sequence lengths
    seq_lens_sum: int

    # For logprob
    return_logprob: bool
    top_logprobs_nums: Optional[List[int]]
    token_ids_logprobs: Optional[List[List[int]]]

    # For DP attention
    global_num_tokens: Optional[
        List[int]
    ]  # e.g. dp = 4, attn-tp = 2, [A, A, B, B, C, C, D, D]
    global_num_tokens_for_logprob: Optional[List[int]]
    all_decode_or_idle: bool
    can_run_tbo: bool

    # For extend
    extend_num_tokens: Optional[int]
    extend_seq_lens: Optional[List[int]]
    extend_prefix_lens: Optional[List[int]]
    extend_logprob_start_lens: Optional[List[int]]
    extend_input_logprob_token_ids: Optional[torch.Tensor]

    # Sampling info
    sampling_info: SamplingBatchInfo

    # The input Embeds
    input_embeds: Optional[torch.tensor] = None
    # over embedding input ids
    oe_token_table: Optional[torch.Tensor] = None
    # Speculative decoding
    spec_algorithm: SpeculativeAlgorithm = None
    spec_info: Optional[Union[EagleVerifyInput, EagleDraftInput]] = None
    # If set, the output of the batch contains the hidden states of the run.
    capture_hidden_mode: CaptureHiddenMode = None

    global_batch_size: List[int] = None
    # Overlap event
    launch_done: Optional[threading.Event] = None

    # hicache pointer for synchronizing data loading from CPU to GPU
    hicache_consumer_index: int = -1

    # set aux data for Disaggregation
    disagg_set_aux_fn: Optional[
        Callable[[torch.Tensor, LogitsProcessorOutput], None]
    ] = None

    input_multi_ids: Optional[torch.Tensor] = None

    reqs: List[Req] = None

    def __str__(self):
        return f"bid={self.bid} mode={self.forward_mode.name}"


@triton.jit
def assign_req_to_token_pool(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    pool_len: tl.constexpr,
    bs_upper: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    kv_start = tl.load(start_offset + pid)
    kv_end = tl.load(end_offset + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len

    # Get the offset for reading out_cache
    length_offset = tl.arange(0, bs_upper)
    start = tl.load(start_offset + length_offset, mask=length_offset < pid)
    end = tl.load(end_offset + length_offset, mask=length_offset < pid)
    out_offset = tl.sum(end - start, axis=0)

    out_cache_ptr = out_cache_loc + out_offset

    save_offset = tl.arange(0, BLOCK_SIZE) + kv_start
    load_offset = tl.arange(0, BLOCK_SIZE)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = save_offset < kv_end
        data = tl.load(out_cache_ptr + load_offset, mask=mask)
        tl.store(token_pool + save_offset, data, mask=mask)
        save_offset += BLOCK_SIZE
        load_offset += BLOCK_SIZE
