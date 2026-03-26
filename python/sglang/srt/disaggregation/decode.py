"""
Life cycle of a request in the decode server

1. PreallocQueue:
    a. Initialize a receiver for each request
    b. The request handshakes first, and pre-allocate kv once there is available kv.
    c. Move the request to TransferQueue.

2. TransferQueue:
    a. Poll the receiver to check the transfer state
    b. If the transfer has finished, move the request to waiting queue

3. WaitingQueue:
    a. Use the requests in the queue to construct a PrebuiltExtendBatch
    b. Skip the prefill forward but only populate metadata

4. RunningBatch:
    a. Merge the resolved PrebuiltExtendBatch into running batch to run decoding
"""
from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import torch
from torch.distributed import ProcessGroup

from sglang.srt.mem_cache.allocator import KVAllocator
from sglang.srt.disaggregation.base import BaseKVManager, BaseKVReceiver, KVArgs, KVPoll
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    FAKE_BOOTSTRAP_HOST,
    KVClassType,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    get_kv_class,
    is_mla_backend,
    kv_to_page_indices,
    poll_and_all_reduce,
    prepare_abort,
    kv_to_page_num,
)
from sglang.srt.managers.req import ABORT_CODE, FINISH_ABORT, RequestStage
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import (
    BaseTokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import get_colorful_logger
from sglang.srt.metrics.collector import KVTransferMetricsCollector

logger = get_colorful_logger(__name__)

# Constants
DEFAULT_RESERVED_DECODE_TOKENS = 512
FP16_DTYPE_SIZE = 2

if TYPE_CHECKING:
    from sglang.srt.managers.req import Req
    from sglang.srt.managers.scheduler import Scheduler


@dataclass
class DecodeRequest:
    req: Req
    kv_receiver: BaseKVReceiver
    waiting_for_input: bool = False
    metadata_buffer_index: int = -1

class DecodePreallocQueue:
    """
    Store the requests that are preallocating.
    """

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool: BaseTokenToKVPool,
        kv_allocator: KVAllocator,
        draft_token_to_kv_pool: Optional[KVCache],
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        metadata_buffers: MetadataBuffers,
        scheduler: Scheduler,
        transfer_queue: DecodeTransferQueue,
        tree_cache: BasePrefixCache,
        gloo_group: ProcessGroup,
        tp_rank: int,
        world_size: int,
        bootstrap_port: int,
        max_total_num_tokens: int,
        transfer_backend: TransferBackend,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool = token_to_kv_pool
        self.kv_allocator = kv_allocator
        self.draft_token_to_kv_pool = draft_token_to_kv_pool
        self.is_mla_backend = is_mla_backend(self.token_to_kv_pool)
        self.draft_is_mla_backend = is_mla_backend(self.draft_token_to_kv_pool)
        self.metadata_buffers = metadata_buffers
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.scheduler = scheduler
        self.transfer_queue = transfer_queue
        self.tree_cache = tree_cache  # enable decode prefix cache
        self.gloo_group = gloo_group
        self.tp_rank = tp_rank
        self.tp_size = world_size
        self.bootstrap_port = bootstrap_port
        self.max_total_num_tokens = max_total_num_tokens

        self.num_reserved_decode_tokens = int(
            os.environ.get("SGLANG_NUM_RESERVED_DECODE_TOKENS", str(DEFAULT_RESERVED_DECODE_TOKENS))
        )

        # Queue for requests pending pre-allocation
        self.queue: List[DecodeRequest] = []
        self.retracted_queue: List[Req] = []
        self.transfer_backend = transfer_backend
        self.kv_manager = self._init_kv_manager()
        self.device = token_to_kv_pool.device  # Add device attribute

        if hasattr(scheduler, 'enable_metrics') and scheduler.enable_metrics:
            labels = {
                'model_name': scheduler.server_args.served_model_name,
                'app_key': scheduler.server_args.app_key,
            }
            self.kv_transfer_metrics = KVTransferMetricsCollector(labels, scheduler.server_args.metrics_reporters)
        else:
            self.kv_transfer_metrics = None

    def _init_kv_manager(self) -> BaseKVManager:
        kv_args = KVArgs()
        kv_args.engine_rank = self.tp_rank
        kv_data_ptrs, kv_data_lens, kv_item_lens = (
            self.token_to_kv_pool.get_contiguous_buf_infos()
        )
        # [[layer0buf0, layer0buf1...], [layer1buf0, layer1buf1...], ...]
        offsets = self.token_to_kv_pool.get_layerwise_buf_info_offsets()
        target_layer_num = self.token_to_kv_pool.layer_num
        draft_layer_num = 0 if self.draft_token_to_kv_pool is None else self.draft_token_to_kv_pool.layer_num
        if self.draft_token_to_kv_pool is not None:
            draft_kv_data_ptrs, draft_kv_data_lens, draft_kv_item_lens = (
                self.draft_token_to_kv_pool.get_contiguous_buf_infos()
            )
            draft_offsets = self.draft_token_to_kv_pool.get_layerwise_buf_info_offsets(len(kv_data_ptrs))
            kv_data_ptrs += draft_kv_data_ptrs
            kv_data_lens += draft_kv_data_lens
            kv_item_lens += draft_kv_item_lens
            offsets += draft_offsets

        kv_args.kv_data_ptrs = kv_data_ptrs
        kv_args.kv_data_lens = kv_data_lens
        kv_args.kv_item_lens = kv_item_lens
        kv_args.offsets = offsets
        kv_args.target_layer_num = target_layer_num
        kv_args.draft_layer_num = draft_layer_num

        kv_args.aux_data_ptrs, kv_args.aux_data_lens, kv_args.aux_item_lens, kv_args.other_output_offset_idx = (
            self.metadata_buffers.get_buf_infos()
        )
        kv_args.ib_device = self.scheduler.server_args.disaggregation_ib_device
        kv_args.gpu_id = self.scheduler.gpu_id
        kv_manager_class = get_kv_class(self.transfer_backend, KVClassType.MANAGER)
        kv_manager = kv_manager_class(
            kv_args,
            DisaggregationMode.DECODE,
            self.scheduler.server_args,
            self.is_mla_backend,
            self.draft_is_mla_backend
        )
        return kv_manager

    def _check_if_req_exceed_kv_capacity(self, req: Req) -> bool:
        if len(req.origin_input_ids) > self.max_total_num_tokens:
            message = f"Request {req.rid} exceeds the maximum number of tokens: {len(req.origin_input_ids)} > {self.max_total_num_tokens}"
            logger.error(message)
            prepare_abort(req, message)
            self.scheduler.stream_output([req], req.return_logprob)
            return True
        return False

    def add(self, req: Req, is_retracted: bool = False) -> None:
        """Add a request to the pending queue."""
        if self._check_if_req_exceed_kv_capacity(req):
            return

        if is_retracted:
            self.retracted_queue.append(req)
        else:
            if req.bootstrap_host == FAKE_BOOTSTRAP_HOST:
                # Fake transfer for warmup reqs
                kv_receiver_class = get_kv_class(
                    TransferBackend.FAKE, KVClassType.RECEIVER
                )
            else:
                kv_receiver_class = get_kv_class(
                    self.transfer_backend, KVClassType.RECEIVER
                )
            kv_receiver = kv_receiver_class(
                mgr=self.kv_manager,
                bootstrap_addr=f"{req.bootstrap_host}:{req.bootstrap_port}",
                bootstrap_room=req.bootstrap_room,
            )

            if self.scheduler.global_rank == 0:
                req.add_latency(RequestStage.DECODE_PREPARE)
            self.queue.append(
                DecodeRequest(req=req, kv_receiver=kv_receiver, waiting_for_input=False)
            )

    def extend(self, reqs: List[Req], is_retracted: bool = False) -> None:
        """Add a request to the pending queue."""
        for req in reqs:
            self.add(req, is_retracted=is_retracted)
    def resume_retracted_reqs(self) -> List[Req]:
        # TODO refactor scheduling part, reuse with unified engine logic as much as possible

        # allocate memory
        resumed_reqs = []
        indices_to_remove = set()
        allocatable_tokens = self._allocatable_tokens(count_retracted=False)

        for i, req in enumerate(self.retracted_queue):
            if req.to_abort:
                prepare_abort(
                    req,
                    "abort retracted req",
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                    err_type=ABORT_CODE.TransferFailed,
                )
                self.scheduler.stream_output(
                    [req], req.return_logprob
                )
                self.kv_allocator.free_group_end()
                indices_to_remove.add(i)
                continue
            if self.req_to_token_pool.available_size() <= 0:
                logger.debug(
                    f"[resume_retracted_reqs] no available token pool: {self.req_to_token_pool.available_size()}"
                )
                break

            required_tokens_for_request = self._ceil_tokens_with_page_size(
                len(req.origin_input_ids)
                + len(req.output_ids)
                + self.num_reserved_decode_tokens,
                self.kv_allocator.page_size
            )
            prefix_len = self._match_prefix_and_lock(req, req.origin_input_ids)
            required_tokens_for_request -= prefix_len

            if required_tokens_for_request > allocatable_tokens:
                # Try eviction before breaking
                if self.tree_cache is not None:
                    evictable_size = self.tree_cache.evictable_size()
                    if evictable_size > 0:
                        needed_tokens = required_tokens_for_request - allocatable_tokens
                        need_page_num = (needed_tokens + self.kv_allocator.page_size - 1) // self.kv_allocator.page_size
                        logger.debug(
                            f"[resume_retracted_reqs] trying evict: needed_tokens={needed_tokens} "
                            f"allocatable_tokens={allocatable_tokens} evictable_size={evictable_size} "
                            f"need_page_num={need_page_num} for req {req.rid=}"
                        )
                        evict_pages = self.tree_cache.evict(need_page_num, self.kv_allocator.free)
                        # Recalculate allocatable_tokens after eviction
                        allocatable_tokens += evict_pages * self.kv_allocator.page_size
                        # If still not enough space, break
                        if required_tokens_for_request > allocatable_tokens:
                            logger.debug(
                                f"[resume_retracted_reqs] still not enough space after eviction "
                                f"for req {req.rid=}, breaking"
                            )
                            break
                    else:
                        logger.debug(
                            f"[resume_retracted_reqs] no evictable space for req {req.rid=}, breaking"
                        )
                        break
                else:
                    logger.debug(
                        f"[resume_retracted_reqs] "
                        f"no available kv pool: {self.kv_allocator.available_size()} "
                        f"allocatable_tokens: {allocatable_tokens} "
                        f"required_tokens_for_request: {required_tokens_for_request}"
                    )
                    break
            logger.debug(
                f"[DecodePreallocQueue]before resumed req {req} requeired: {required_tokens_for_request}"
            )
            resumed_reqs.append(req)
            indices_to_remove.add(i)
            req.is_retracted = False
            req.extend_input_len = len(req.fill_ids) - req.prefix_len
            kv_loc = self._pre_alloc(req)
            self.req_to_token_pool.alloced_lens[req.req_pool_idx] += kv_loc.shape[0] + req.prefix_len
            allocatable_tokens -= required_tokens_for_request
            # load from cpu, release cpu copy
            req.load_kv_cache(self.req_to_token_pool, self.token_to_kv_pool)
            logger.debug(f"[DecodePreallocQueue]resumed req {req}")

        self.retracted_queue = [
            entry
            for i, entry in enumerate(self.retracted_queue)
            if i not in indices_to_remove
        ]
        return resumed_reqs

    def _update_handshake_waiters(self) -> None:
        if not self.queue:
            return

        if all(decode_req.waiting_for_input for decode_req in self.queue):
            return

        polls = poll_and_all_reduce(
            [decode_req.kv_receiver for decode_req in self.queue], self.gloo_group
        )

        for i, (decode_req, poll) in enumerate(zip(self.queue, polls)):
            if poll == KVPoll.Bootstrapping:
                pass
            elif poll == KVPoll.WaitingForInput:
                decode_req.waiting_for_input = True
            elif poll == KVPoll.Failed:
                error_message = f"Decode handshake failed for request rank={self.tp_rank} {decode_req.req.rid=} {decode_req.req.bootstrap_room=}"
                try:
                    decode_req.kv_receiver.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                logger.error(error_message)
                prepare_abort(
                    decode_req.req,
                    error_message,
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                    err_type=ABORT_CODE.TransferFailed,
                )
            else:
                raise ValueError(f"Unexpected poll case: {poll}")

    def _ceil_tokens_with_page_size(self, tokens, page_size):
        return page_size * ((tokens + page_size - 1) // page_size)

    def pop_preallocated(self) -> List[DecodeRequest]:
        """Pop requests from the queue for pre-allocation."""
        self._update_handshake_waiters()
        preallocated_reqs = []
        indices_to_remove = set()
        # We need to make sure that the sum of inflight tokens and allocatable tokens is greater than maximum input+output length of each inflight request
        # Otherwise it is possible for one request running decode out of memory, while all other requests are in the transfer queue that cannot be retracted.
        retractable_tokens = 0
        if self.scheduler.running_batch is not None:
            retractable_tokens = sum(
                len(r.origin_input_ids) + len(r.output_ids)
                for r in self.scheduler.running_batch.reqs
            )
        allocatable_tokens = self._allocatable_tokens(
            retractable_tokens=retractable_tokens, count_retracted=True
        )
        # First, remove all failed requests from the queue
        for i, decode_req in enumerate(self.queue):
            if isinstance(decode_req.req.finished_reason, FINISH_ABORT) or decode_req.req.to_abort:
                self.scheduler.stream_output(
                    [decode_req.req], decode_req.req.return_logprob
                )
                indices_to_remove.add(i)

        for i, decode_req in enumerate(self.queue):
            if i in indices_to_remove:
                continue

            if not decode_req.waiting_for_input:
                continue

            if self.req_to_token_pool.available_size() <= 0:
                break

            if self.req_to_metadata_buffer_idx_allocator.available_size() <= 0:
                break


            req = decode_req.req
            # Match prefix and lock corresponding tree node
            prefix_len = self._match_prefix_and_lock(
                req, req.origin_input_ids
            )
            #Since actual consideration is based on page, need ceil here
            required_tokens_for_request = self._ceil_tokens_with_page_size(
                len(req.origin_input_ids) + self.num_reserved_decode_tokens, self.token_to_kv_pool.page_size
            )
            required_tokens_for_request -= prefix_len
            max_required_tokens = max(
                required_tokens_for_request,
                req.extend_input_len
                + req.sampling_params.max_new_tokens
                - retractable_tokens - prefix_len
            )

            if max_required_tokens > allocatable_tokens:
                # Try eviction before breaking
                if self.tree_cache is not None and self.tree_cache.evictable_size() > 0:
                    needed_tokens = max_required_tokens - allocatable_tokens
                    need_page_num = (needed_tokens + self.kv_allocator.page_size - 1) // self.kv_allocator.page_size
                    evict_pages = self.tree_cache.evict(need_page_num, self.kv_allocator.free)
                    logger.debug(f"[prealloc] [evict] {need_page_num=}, {evict_pages=}")
                    # Recalculate allocatable_tokens after eviction
                    allocatable_tokens += evict_pages * self.kv_allocator.page_size
                    # If still not enough space, break
                    if max_required_tokens > allocatable_tokens:
                        break
                else:
                    break

            kv_loc = self._pre_alloc(decode_req.req)
            self.req_to_token_pool.alloced_lens[decode_req.req.req_pool_idx] += kv_loc.shape[0] + decode_req.req.prefix_len

            logger.debug(
                f"[DecodePreallocQueue][pop] enough cache for req={decode_req.req.rid} allocatable_tokens={allocatable_tokens} "
                f"required_tokens_for_request={required_tokens_for_request} "
                f"retractable={retractable_tokens} "
                f"prefix_len={decode_req.req.prefix_len} "
                f"cur_token_finish_needed={decode_req.req.extend_input_len + decode_req.req.sampling_params.max_new_tokens - retractable_tokens} "
                f"alloced_lens = {self.req_to_token_pool.alloced_lens[decode_req.req.req_pool_idx]}"
            )

            kv_indices = (
                self.req_to_token_pool.req_to_token[decode_req.req.req_pool_idx][
                    decode_req.req.prefix_len: len(decode_req.req.origin_input_ids)
                ]
                .cpu()
                .numpy()
                .astype(np.int64)
            )

            decode_req.metadata_buffer_index = (
                self.req_to_metadata_buffer_idx_allocator.alloc()
            )
            assert decode_req.metadata_buffer_index is not None
            page_indices = kv_to_page_indices(
                kv_indices, self.token_to_kv_pool.page_size
            )

            allocatable_tokens -= required_tokens_for_request
            if self.kv_transfer_metrics:
                num_kv_indices = len(decode_req.req.origin_input_ids)
                page_size = self.token_to_kv_pool.page_size
                num_pages = kv_to_page_num(num_kv_indices, page_size)
                transfer_size_bytes = num_pages * self.token_to_kv_pool.page_size_bytes
                self.kv_transfer_metrics.log_kv_transfer_size(transfer_size_bytes)

            decode_req.kv_receiver.init(page_indices, decode_req.metadata_buffer_index, decode_prefix_len = decode_req.req.prefix_len)
            if self.scheduler.global_rank == 0:
                decode_req.req.add_latency(RequestStage.DECODE_BOOTSTRAP)
            preallocated_reqs.append(decode_req)
            indices_to_remove.add(i)

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        return preallocated_reqs

    def _allocatable_tokens(
        self, retractable_tokens: Optional[int] = None, count_retracted: bool = True
    ) -> int:
        need_space_for_single_req = 0
        if self.scheduler.running_batch is not None:
            need_space_for_single_req = (
                max(
                    [
                        x.sampling_params.max_new_tokens
                        + len(x.origin_input_ids)
                        - retractable_tokens
                        for x in self.scheduler.running_batch.reqs
                    ]
                )
                if retractable_tokens is not None
                and len(self.scheduler.running_batch.reqs) > 0
                else 0
            )
        elif len(self.transfer_queue.queue) > 0:
            need_space_for_single_req = max(
                [
                    decode_req.req.sampling_params.max_new_tokens
                    + len(decode_req.req.origin_input_ids)
                    for decode_req in self.transfer_queue.queue
                ]
            )

        reserve_token_size = self.num_reserved_decode_tokens * (
            len(self.scheduler.running_batch.reqs)
            if self.scheduler.running_batch
            else 0 + len(self.transfer_queue.queue) + len(self.scheduler.waiting_queue)
        )
        # Insufficient allocatable_tokens will cause req on prefill instance to stay in KVPoll.Bootstrapping state
        reserve_num_pages = (reserve_token_size + self.kv_allocator.page_size - 1) // self.kv_allocator.page_size
        available_pages = self.kv_allocator.available_size()
        need_pages_for_single_req = (need_space_for_single_req + self.kv_allocator.page_size - 1) // self.kv_allocator.page_size
        allocatable_pages = available_pages - max(
            reserve_num_pages,
            # make sure each request can finish if reach max_tokens with all other requests retracted
            need_pages_for_single_req,
        )
        allocatable_tokens = allocatable_pages * self.kv_allocator.page_size

        # Note: if the last fake extend just finishes, and we enter `pop_preallocated` immediately in the next iteration
        #       the extend batch is not in any queue, so we need to explicitly add the tokens slots here
        if (
            self.scheduler.last_batch
            and self.scheduler.last_batch.forward_mode.is_extend()
        ):
            allocatable_tokens -= self.num_reserved_decode_tokens * len(
                self.scheduler.last_batch.reqs
            )

        if count_retracted:
            allocatable_tokens -= sum(
                [
                    len(req.origin_input_ids)
                    + len(req.output_ids)
                    + self.num_reserved_decode_tokens
                    for req in self.retracted_queue
                ]
            )
        return allocatable_tokens

    def _match_prefix_and_lock(self, req: Req, input_ids: List[int]) -> Tuple[torch.Tensor, int, Optional[object]]:
        """Match prefix and lock the corresponding tree node

        Returns:
            Tuple of (matched_prefix_pages, prefix_len, matched_last_node)
        """
        self.tree_cache.dec_lock_ref(req.last_node)
        req.last_node = None
        match_result = self.tree_cache.match_prefix(input_ids, req=req)
        (matched_prefix_pages, prefix_len, matched_last_node) = (
            match_result.device_indices, match_result.device_prefix_length, match_result.last_device_node)
        # Lock the matched prefix to prevent eviction
        if prefix_len > 0:
            self.tree_cache.inc_lock_ref(matched_last_node)

        req.prefix_len = prefix_len
        req.prefix_page_ids = matched_prefix_pages
        req.last_node = matched_last_node
        return prefix_len

    def _pre_alloc(self, req: Req) -> torch.Tensor:
        """Pre-allocate memory for req_to_token and token_kv_pool"""
        assert req.req_pool_idx is None, f"req_pool_idx shoule be None when pre_alloc, but get {req.req_pool_idx}"
        req_pool_indices = self.req_to_token_pool.alloc(1)
        assert req_pool_indices is not None

        req.req_pool_idx = req_pool_indices[0]
        req.oe_init = False

        total_len = len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
        matched_prefix_pages = req.prefix_page_ids
        prefix_len = req.prefix_len
        page_size = self.kv_allocator.page_size
        prefix_kv_locs = torch.tensor([], dtype=torch.int32, device=self.device)
        kv_loc = torch.tensor([], dtype=torch.int32, device=self.device)

        # Handle prefix pages from locked last_node
        if prefix_len > 0:
            new_tokens_needed = total_len - prefix_len
            # Ensure torch.arange is on the same device as matched_prefix_pages
            arange_device = matched_prefix_pages.device if matched_prefix_pages.numel() > 0 else self.device
            prefix_kv_locs = (matched_prefix_pages.unsqueeze(1) * page_size + torch.arange(page_size, device=arange_device)).flatten()
            assert len(prefix_kv_locs) == prefix_len

            # This is essential for radix_cache to work correctly
            prefix_page_count = (prefix_len + page_size - 1) // page_size
            self.kv_allocator.req_to_page[
                req.req_pool_idx, :prefix_page_count
            ] = matched_prefix_pages

        else:
            new_tokens_needed = total_len

        alloced_len = self.req_to_token_pool.alloced_lens[req.req_pool_idx].item()
        if prefix_len > alloced_len:
            alloced_len = prefix_len
        # Allocate memory for new tokens
        if new_tokens_needed > 0:

            kv_loc = self.kv_allocator.alloc(
                req.req_pool_idx,
                new_tokens_needed,
                alloced_len
            )

            assert kv_loc is not None, f"Failed to allocate {new_tokens_needed} tokens for request {req.rid} {self.tree_cache.evictable_size()=} {self.tree_cache.protected_size()=} {self.kv_allocator.available_size()=}"
            self.kv_allocator.token_slot_refs[kv_loc] += 1

        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(0, total_len)), torch.cat([prefix_kv_locs, kv_loc])
        )
        # populate metadata
        req.fill_ids = req.origin_input_ids + req.output_ids
        req.extend_input_len = len(req.fill_ids) - req.prefix_len

        return  kv_loc


class DecodeTransferQueue:
    """
    Store the requests that is polling kv
    """

    def __init__(
        self,
        gloo_group: ProcessGroup,
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        metadata_buffers: MetadataBuffers,
        scheduler: Scheduler,
        tree_cache: BasePrefixCache,
    ):
        self.queue: List[DecodeRequest] = []
        self.gloo_group = gloo_group
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.metadata_buffers = metadata_buffers
        self.scheduler = scheduler
        self.tree_cache = tree_cache

    def add(self, decode_req: DecodeRequest) -> None:
        self.queue.append(decode_req)

    def extend(self, decode_reqs: List[DecodeRequest]) -> None:
        self.queue.extend(decode_reqs)

    def pop_transferred(self) -> List[DecodeRequest]:
        if not self.queue:
            return []

        polls = poll_and_all_reduce(
            [decode_req.kv_receiver for decode_req in self.queue], self.gloo_group
        )

        transferred_reqs = []
        indices_to_remove = set()
        for i, (decode_req, poll) in enumerate(zip(self.queue, polls)):
            if poll == KVPoll.Failed or decode_req.req.to_abort:
                error_message = f"Decode transfer failed for request rank={self.scheduler.tp_rank} {decode_req.req.rid=} {decode_req.req.bootstrap_room=}"
                try:
                    decode_req.kv_receiver.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                logger.error(error_message)
                prepare_abort(
                    decode_req.req,
                    error_message,
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                    err_type=ABORT_CODE.TransferFailed,
                )
                self.scheduler.stream_output(
                    [decode_req.req], decode_req.req.return_logprob
                )

                # Clean up KV cache to prevent memory leak
                self.tree_cache.cache_finished_req(decode_req.req)
                self.tree_cache.token_to_kv_pool_allocator.free_group_end()
                indices_to_remove.add(i)
                continue
            elif poll == KVPoll.Success:
                idx = decode_req.metadata_buffer_index
                (
                    output_id,
                    output_token_logprobs_val,
                    output_token_logprobs_idx,
                    output_top_logprobs_val,
                    output_top_logprobs_idx,
                    hidden_states,
                    cached_tokens,
                ) = self.metadata_buffers.get_buf(idx)

                next_token_ids = self.scheduler.tp_worker.model_runner.forward_postprocess_for_pd_decode(
                    decode_req.req,
                    output_id,
                    hidden_states,
                )
                decode_req.req.output_ids.append(next_token_ids[0].item())

                decode_req.req.cached_tokens = cached_tokens[0].item()

                if decode_req.req.return_logprob:
                    decode_req.req.output_token_logprobs_val.append(
                        output_token_logprobs_val[0].item()
                    )
                    decode_req.req.output_token_logprobs_idx.append(
                        output_token_logprobs_idx[0].item()
                    )
                    decode_req.req.output_top_logprobs_val.append(
                        output_top_logprobs_val[
                            : decode_req.req.top_logprobs_num
                        ].tolist()
                    )
                    decode_req.req.output_top_logprobs_idx.append(
                        output_top_logprobs_idx[
                            : decode_req.req.top_logprobs_num
                        ].tolist()
                    )

                # ===== DEBUG: Print KV cache sequence after transfer =====
                # self._debug_print_kv_cache(decode_req.req)
                # =========================================================

                if hasattr(decode_req.kv_receiver, "clear"):
                    decode_req.kv_receiver.clear()
                    decode_req.kv_receiver = None  # Ensure receiver is properly cleaned up

                # special handling for sampling_params.max_new_tokens == 1
                decode_req.req.check_finished()
                if decode_req.req.finished():
                    self.scheduler.stream_output(
                        [decode_req.req], decode_req.req.return_logprob
                    )
                    self.tree_cache.cache_finished_req(decode_req.req)
                    self.tree_cache.token_to_kv_pool_allocator.free_group_end()
                else:
                    transferred_reqs.append(decode_req.req)
                indices_to_remove.add(i)
            elif poll in [
                KVPoll.Bootstrapping,
                KVPoll.WaitingForInput,
                KVPoll.Transferring,
            ]:
                pass
            else:
                raise ValueError(f"Unexpected poll case: {poll}")

        for i in indices_to_remove:
            idx = self.queue[i].metadata_buffer_index
            assert idx != -1
            if self.scheduler.global_rank == 0:
                self.queue[i].req.add_latency(RequestStage.DECODE_TRANSFERRED)
            self.req_to_metadata_buffer_idx_allocator.free(idx)

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        return transferred_reqs

    def _debug_print_kv_cache(self, req: Req):
        """Debug function to print KV cache sequence by averaging K values at each position"""
        try:
            req_pool_idx = req.req_pool_idx
            if req_pool_idx is None:
                logger.warning(f"[DEBUG_KV] req {req.rid} has no req_pool_idx")
                return

            # Access req_to_token_pool and token_to_kv_pool through scheduler
            req_to_token_pool = self.scheduler.req_to_token_pool
            token_to_kv_pool = self.scheduler.token_to_kv_pool

            # Get the token indices for this request
            seq_len = len(req.origin_input_ids) + len(req.output_ids)
            token_indices = req_to_token_pool.req_to_token[req_pool_idx][:seq_len]

            # Get K cache from first layer (layer 0) as representative
            # Try to get k_buffer, support both MHA and MLA architectures
            k_cache_layer0 = None
            if hasattr(token_to_kv_pool, 'k_buffer'):
                # MHATokenToKVPool
                k_cache_layer0 = token_to_kv_pool.get_key_buffer(0)
            elif hasattr(token_to_kv_pool, 'get_key_buffer'):
                # Try generic get_key_buffer method
                try:
                    k_cache_layer0 = token_to_kv_pool.get_key_buffer(0)
                except:
                    pass

            if k_cache_layer0 is not None:
                # Extract K values for this request's tokens
                k_values = k_cache_layer0[token_indices]  # [seq_len, head_num, head_dim]

                # Average across heads and dimensions to get one value per position
                k_avg_per_pos = k_values.mean(dim=[1, 2]).cpu().tolist()  # [seq_len]

                # Format output
                logger.info(f"[DEBUG_KV] req={req.rid} prefix_len={req.prefix_len} seq_len={seq_len}")
                logger.info(f"[DEBUG_KV] req={req.rid} origin_input_ids={req.origin_input_ids}")
                logger.info(f"[DEBUG_KV] req={req.rid} output_ids={req.output_ids}")
                logger.info(f"[DEBUG_KV] req={req.rid} token_indices={token_indices.cpu().tolist()}")

                # Print K average values (all positions)
                k_avg_str = ", ".join([f"{v:.4f}" for v in k_avg_per_pos])
                logger.info(f"[DEBUG_KV] req={req.rid} K_avg_values=[{k_avg_str}]")

                # Also print prefix vs non-prefix separately if there's a prefix
                if req.prefix_len > 0:
                    prefix_k_avg = k_avg_per_pos[:req.prefix_len]
                    non_prefix_k_avg = k_avg_per_pos[req.prefix_len:]
                    prefix_str = ", ".join([f"{v:.4f}" for v in prefix_k_avg])
                    non_prefix_str = ", ".join([f"{v:.4f}" for v in non_prefix_k_avg])
                    logger.info(f"[DEBUG_KV] req={req.rid} PREFIX_K_avg=[{prefix_str}]")
                    logger.info(f"[DEBUG_KV] req={req.rid} NON_PREFIX_K_avg=[{non_prefix_str}]")
            else:
                logger.warning(f"[DEBUG_KV] Cannot access K cache buffer, token_to_kv_pool type: {type(token_to_kv_pool).__name__}")

        except Exception as e:
            logger.error(f"[DEBUG_KV] Error printing KV cache for req {req.rid}: {e}")
            import traceback
            logger.error(f"[DEBUG_KV] Traceback: {traceback.format_exc()}")


class SchedulerDisaggregationDecodeMixin:

    def _prepare_idle_batch_and_run(self, batch, delay_process=False):
        batch = self.prepare_dp_attn_batch(batch)
        result = None
        if batch:
            result = self.run_batch(batch)
            if not delay_process:
                self.process_batch_result(batch, result)
        return batch, result

    @torch.no_grad()
    def event_loop_normal_disagg_decode(self: Scheduler):
        """A normal scheduler loop for decode worker in disaggregation mode."""

        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            # polling and allocating kv cache
            self.process_decode_queue()

            # Check hierarchical cache events (write_through completion, load_back completion, etc.)
            if self.enable_hierarchical_cache:
                self.tree_cache.check_hicache_events()

            batch = self.get_next_disagg_decode_batch_to_run()
            self.cur_batch = batch

            if batch:
                # Generate fake extend output.
                if batch.forward_mode.is_extend():
                    # Note: Logprobs should be handled on the prefill engine.
                    self.stream_output(
                        batch.reqs, any(req.return_logprob for req in batch.reqs)
                    )
                    if self.server_args.enable_dp_attention:
                        self._prepare_idle_batch_and_run(None)
                else:
                    if self.server_args.enable_dp_attention:
                        self.prepare_dp_attn_batch(batch)
                    result = self.run_batch(batch)
                    self.process_batch_result(batch, result)
            elif self.server_args.enable_dp_attention:
                batch, _ = self._prepare_idle_batch_and_run(None)

            if batch is None and (
                len(self.waiting_queue)
                + len(self.disagg_decode_transfer_queue.queue)
                + len(self.disagg_decode_prealloc_queue.queue)
                == 0
            ):
                # When server is idle, do self-check and re-init some states
                self.check_memory()
                self.new_token_ratio = self.init_new_token_ratio

            if batch is None:
                self.log_idle_stats()

            self.last_batch = batch

    @torch.no_grad()
    def event_loop_overlap_disagg_decode(self: Scheduler):
        result_queue = deque()
        self.last_batch: Optional[ScheduleBatch] = None
        self.last_batch_in_queue = False  # last batch is modified in-place, so we need another variable to track if it's extend

        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            # polling and allocating kv cache
            self.process_decode_queue()

            # Check hierarchical cache events (write_through completion, load_back completion, etc.)
            if self.enable_hierarchical_cache:
                self.tree_cache.check_hicache_events()

            batch = self.get_next_disagg_decode_batch_to_run()
            self.cur_batch = batch
            last_batch_in_queue = False

            if batch:
                # Generate fake extend output.
                if batch.forward_mode.is_extend():
                    # Note: Logprobs should be handled on the prefill engine.
                    self.stream_output(
                        batch.reqs, any(req.return_logprob for req in batch.reqs)
                    )
                    if self.server_args.enable_dp_attention:
                        batch_, result = self._prepare_idle_batch_and_run(
                            None, delay_process=True
                        )
                        if batch_:
                            result_queue.append((batch_.copy(), result))
                            last_batch_in_queue = True
                else:
                    if self.server_args.enable_dp_attention:
                        self.prepare_dp_attn_batch(batch)
                    result = self.run_batch(batch)
                    result_queue.append((batch.copy(), result))

                    if (self.last_batch is None) or (not self.last_batch_in_queue):
                        # Create a dummy first batch to start the pipeline for overlap schedule.
                        # It is now used for triggering the sampling_info_done event.
                        tmp_batch = ScheduleBatch(
                            reqs=None,
                            forward_mode=ForwardMode.DUMMY_FIRST,
                            next_batch_sampling_info=(
                                self.tp_worker.cur_sampling_info
                                if self.draft_worker is None
                                else self.draft_worker.cur_sampling_info
                            ),
                        )
                        self.set_next_batch_sampling_info_done(tmp_batch)
                    last_batch_in_queue = True

            elif self.server_args.enable_dp_attention:
                batch, result = self._prepare_idle_batch_and_run(
                    None, delay_process=True
                )
                if batch:
                    result_queue.append((batch.copy(), result))
                    last_batch_in_queue = True

            # Process the results of the previous batch but skip if the last batch is extend
            if self.last_batch and self.last_batch_in_queue:
                tmp_batch, tmp_result = result_queue.popleft()
                tmp_batch.next_batch_sampling_info = (
                    self.tp_worker.cur_sampling_info
                    if self.draft_worker is None
                    else self.draft_worker.cur_sampling_info
                )
                self.process_batch_result(tmp_batch, tmp_result)
            elif self.running_batch is not None and self.draft_worker:
                # In overlap + MTP mode, correct reserve num tokens here to ensure
                # sufficient allocation
                for req in self.running_batch.reqs:
                    req.reserve_num_tokens = self.server_args.speculative_num_draft_tokens

            if batch is None and (
                len(self.waiting_queue)
                + len(self.disagg_decode_transfer_queue.queue)
                + len(self.disagg_decode_prealloc_queue.queue)
                == 0
            ):
                # When the server is idle, do self-check and re-init some states
                self.check_memory()
                self.new_token_ratio = self.init_new_token_ratio

            if batch is None:
                self.log_idle_stats()

            self.last_batch = batch
            self.last_batch_in_queue = last_batch_in_queue

    def get_next_disagg_decode_batch_to_run(
        self: Scheduler,
    ) -> Optional[Tuple[ScheduleBatch, bool]]:
        """Create fake completed prefill if possible and merge with running batch"""
        # Merge the prefill batch into the running batch
        last_batch = self.last_batch
        if last_batch and last_batch.forward_mode.is_extend():
            # chunked prefill doesn't happen in decode instance.
            assert self.chunked_req is None
            # Filter finished batches.
            last_batch.filter_batch()
            if not last_batch.is_empty():
                if self.running_batch is None or self.running_batch.is_empty():
                    self.running_batch = last_batch
                else:
                    # merge running_batch with prefill batch
                    self.running_batch.merge_batch(last_batch)

        new_prebuilt_batch = self.get_new_prebuilt_batch()

        ret: Optional[ScheduleBatch] = None
        if new_prebuilt_batch:
            ret = new_prebuilt_batch
        else:
            if self.running_batch is None or self.running_batch.is_empty():
                ret = None
            else:
                self.running_batch = self.update_running_batch(self.running_batch, self.last_batch)
                ret = (
                    self.running_batch
                    if (
                        self.running_batch is not None
                        and not self.running_batch.is_empty()
                    )
                    else None
                )
        self.update_oe_info(ret)
        return ret

    def get_new_prebuilt_batch(self: Scheduler) -> Optional[ScheduleBatch]:
        """Create a schedulebatch for fake completed prefill"""
        if self.grammar_queue:
            self.move_ready_grammar_requests()

        if len(self.waiting_queue) == 0:
            return None

        curr_batch_size = self.running_batch.batch_size() if self.running_batch else 0

        batch_size = min(self.req_to_token_pool.size, self.max_running_requests)

        num_not_used_batch = batch_size - curr_batch_size
        logger.debug(f"[Scheduler][ConstrcutBatch] adding_batch:{num_not_used_batch}")

        # pop req from waiting queue
        can_run_list: List[Req] = []
        waiting_queue: List[Req] = []

        for i in range(len(self.waiting_queue)):
            req = self.waiting_queue[i]
            # we can only add at least `num_not_used_batch` new batch to the running queue
            if i < num_not_used_batch:
                can_run_list.append(req)
                logger.debug(
                    f"[Scheduler][ConstrcutBatch] adding req {req} to running queue"
                )
                if self.global_rank == 0:
                    req.add_latency(RequestStage.DECODE_WAITING)
                req.init_next_round_input()  # prealloc has already prefix matched, no need to match again here
            else:
                waiting_queue.append(req)

        self.waiting_queue = waiting_queue
        if len(can_run_list) == 0:
            return None

        # construct a schedule batch with those requests and mark as decode
        draft_token_num = (
            self.server_args.speculative_num_draft_tokens if self.draft_worker else 0
        )
        spec_num_steps = (
            self.server_args.speculative_num_steps if self.draft_worker else 0
        )
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.kv_allocator,
            self.token_to_kv_pool,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            self.server_args.enable_custom_logit_processor,
            self.reserve_num_tokens,
            draft_token_num,
            spec_num_steps,
        )

        # construct fake completed prefill
        new_batch.prepare_for_prebuilt_extend()
        new_batch.process_prebuilt_extend(self.server_args, self.model_config)

        return new_batch


    def process_decode_queue(self: Scheduler):
        # try to resume retracted requests if there are enough space for another `num_reserved_decode_tokens` decode steps
        # 1. Retracted requests have the highest priority
        resumed_reqs = self.disagg_decode_prealloc_queue.resume_retracted_reqs()
        self.waiting_queue.extend(resumed_reqs)
        if len(self.disagg_decode_prealloc_queue.retracted_queue) > 0:
            # if there are still retracted requests, we do not allocate new requests
            # but we allow reqs whose kv cache have already been allocated to run
            logger.debug(
                f"[Scheduler][process_decode_queue] there are still {len(self.disagg_decode_prealloc_queue.retracted_queue)} requests in retracted queue"
            )
        else:
            # 2. Newly arrived Reqs have the lowest priority
            req_conns = self.disagg_decode_prealloc_queue.pop_preallocated()
            self.disagg_decode_transfer_queue.extend(req_conns)

        # 3. Allocated reqs have the second highest priority
        alloc_reqs = (
            self.disagg_decode_transfer_queue.pop_transferred()
        )  # the requests which kv has arrived
        self.waiting_queue.extend(alloc_reqs)
