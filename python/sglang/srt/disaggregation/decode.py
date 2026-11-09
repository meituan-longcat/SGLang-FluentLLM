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

import contextlib
import gc
import os
import time
from collections import deque
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.distributed import ProcessGroup

from sglang.srt.disaggregation.base import BaseKVManager, BaseKVReceiver, KVArgs, KVPoll
from sglang.srt.disaggregation.kv_manager_queue_mixin import get_transfer_buffer_device_types
from sglang.srt.disaggregation.utils import (
    FAKE_BOOTSTRAP_HOST,
    DisaggregationMode,
    KVClassType,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    get_kv_class,
    is_mla_backend,
    kv_to_page_indices,
    kv_to_page_num,
    poll_and_all_reduce,
    prepare_abort,
)
from sglang.srt.env import envs
from sglang.srt.managers.req import AbortCode, FinishAbort, RequestStage
from sglang.srt.managers.scheduling.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    HybridReqToTokenPool,
)
from sglang.srt.mem_cache.memory_pool.host_memory_pool import RankLocalCpuCopy
from sglang.srt.mem_cache.memory_pool.kv_allocator import KVAllocator
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.observability.metrics.collector import KVTransferMetricsCollector
from sglang.srt.utils import get_colorful_logger
from sglang.srt.utils.common import configure_gc_warning, freeze_gc

logger = get_colorful_logger(__name__)

# Constants
DEFAULT_RESERVED_DECODE_TOKENS = 512
CLIP_MAX_NEW_TOKEN = envs.SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION.get()
FP16_DTYPE_SIZE = 2

if TYPE_CHECKING:
    from sglang.srt.managers.req import Req
    from sglang.srt.managers.scheduling.scheduler import Scheduler
    from sglang.srt.mem_cache.memory_pool import (
        BaseTokenToKVPool,
        ReqToTokenPool,
    )


@dataclass
class DecodeRequest:
    req: Req
    kv_receiver: BaseKVReceiver
    waiting_for_input: bool = False
    metadata_buffer_index: int = -1
    abort_initiated: bool = False


@dataclass
class _RankAllocPlan:
    rank_alloc_num_tokens: int
    page2rank_new: list
    alloced_num_tokens: int


class DecodePreallocQueue:
    """
    Store the requests that are preallocating.
    """

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool: BaseTokenToKVPool,
        kv_allocator: KVAllocator,
        draft_token_to_kv_pools: list[BaseTokenToKVPool | None],
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        metadata_buffers: MetadataBuffers,
        scheduler: Scheduler,
        transfer_queue: DecodeTransferQueue,
        tree_cache: RadixCache,
        gloo_group: ProcessGroup,
        tp_rank: int,
        tp_size: int,
        bootstrap_port: int,
        max_total_num_tokens: int,
        transfer_backend: TransferBackend,
        scheduler_comm_device: torch.device,
        pp_rank: int,
        pp_size: int,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool = token_to_kv_pool
        self.kv_allocator = kv_allocator
        self.draft_token_to_kv_pools = draft_token_to_kv_pools
        self.is_mla_backend = is_mla_backend(self.token_to_kv_pool)
        if self.draft_token_to_kv_pools is None:
            self.draft_is_mla_backend = self.is_mla_backend
        else:
            self.draft_is_mla_backend = is_mla_backend(self.draft_token_to_kv_pools[0])
        self.metadata_buffers = metadata_buffers
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.scheduler = scheduler
        self.transfer_queue = transfer_queue
        self.tree_cache = tree_cache  # enable decode prefix cache
        self.gloo_group = gloo_group
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.bootstrap_port = bootstrap_port
        self.max_total_num_tokens = max_total_num_tokens

        self.num_reserved_decode_tokens = int(os.environ.get("SGLANG_NUM_RESERVED_DECODE_TOKENS", str(DEFAULT_RESERVED_DECODE_TOKENS)))

        # Queue for requests pending pre-allocation
        self.queue: list[DecodeRequest] = []
        self.retracted_queue: list[Req] = []
        self._waiting_poll_calls = 0
        self.transfer_backend = transfer_backend
        self.kv_manager = self._init_kv_manager()
        self.device = token_to_kv_pool.device  # Add device attribute

        if hasattr(scheduler, "enable_metrics") and scheduler.enable_metrics:
            labels = {
                "model_name": scheduler.server_args.served_model_name,
                "app_key": scheduler.server_args.app_key,
            }
            self.kv_transfer_metrics = KVTransferMetricsCollector(labels, scheduler.server_args.metrics_reporters)
        else:
            self.kv_transfer_metrics = None

        # Decode cache hit rate tracking
        self.total_decode_tokens = 0
        self.total_decode_hit_tokens = 0
        self.scheduler_comm_device = scheduler_comm_device

    def _init_kv_manager(self) -> BaseKVManager:
        kv_args = KVArgs()
        kv_args.tp_rank = self.tp_rank
        kv_args.pp_rank = self.pp_rank
        kv_args.prefill_start_layer = self.token_to_kv_pool.start_layer
        kv_args.prefill_end_layer = self.token_to_kv_pool.end_layer
        assert kv_args.pp_rank == 0 and kv_args.prefill_start_layer == 0
        kv_data_ptrs, kv_data_lens, kv_item_lens = self.token_to_kv_pool.get_contiguous_buf_infos()
        kv_buffer_device_types = get_transfer_buffer_device_types(self.token_to_kv_pool, len(kv_data_ptrs))
        # [[layer0buf0, layer0buf1...], [layer1buf0, layer1buf1...], ...]
        offsets = self.token_to_kv_pool.get_layerwise_buf_info_offsets()
        target_layer_num = self.token_to_kv_pool.layer_num
        draft_layer_num = 0
        if self.draft_token_to_kv_pools:
            for draft_token_to_kv_pool in self.draft_token_to_kv_pools:
                draft_layer_num += draft_token_to_kv_pool.layer_num
                draft_kv_data_ptrs, draft_kv_data_lens, draft_kv_item_lens = draft_token_to_kv_pool.get_contiguous_buf_infos()
                draft_offsets = draft_token_to_kv_pool.get_layerwise_buf_info_offsets(len(kv_data_ptrs))
                kv_data_ptrs += draft_kv_data_ptrs
                kv_data_lens += draft_kv_data_lens
                kv_item_lens += draft_kv_item_lens
                offsets += draft_offsets

                kv_buffer_device_types += get_transfer_buffer_device_types(draft_token_to_kv_pool, len(draft_kv_data_ptrs))

        kv_args.kv_data_ptrs = kv_data_ptrs
        kv_args.kv_data_lens = kv_data_lens
        kv_args.kv_item_lens = kv_item_lens
        kv_args.offsets = offsets
        kv_args.target_layer_num = target_layer_num
        kv_args.draft_layer_num = draft_layer_num
        kv_args.draft_enable_mla_l1_5_cache = bool(
            self.draft_token_to_kv_pools and getattr(self.draft_token_to_kv_pools[0], "enable_mla_l1_5_cache", False)
        )
        kv_args.aux_data_ptrs, kv_args.aux_data_lens, kv_args.aux_item_lens = self.metadata_buffers.get_buf_infos()
        kv_args.ib_device = self.scheduler.server_args.disaggregation_ib_device
        kv_args.gpu_id = self.scheduler.gpu_id
        kv_args.buffer_device_types = kv_buffer_device_types

        if isinstance(self.token_to_kv_pool, HybridLinearKVPool):
            state_data_ptrs, state_data_lens, state_item_lens = self.token_to_kv_pool.get_state_buf_infos()
            kv_args.state_data_ptrs = state_data_ptrs
            kv_args.state_data_lens = state_data_lens
            kv_args.state_item_lens = state_item_lens

            kv_args.state_type = "mamba"
            # Get state dimension info for cross-TP slice transfer
            if hasattr(self.token_to_kv_pool, "get_state_dim_per_tensor"):
                kv_args.state_dim_per_tensor = self.token_to_kv_pool.get_state_dim_per_tensor()
        else:
            kv_args.state_data_ptrs = []
            kv_args.state_data_lens = []
            kv_args.state_item_lens = []
            kv_args.state_type = "none"
        kv_manager_class = get_kv_class(self.transfer_backend, KVClassType.MANAGER)
        kv_manager = kv_manager_class(kv_args, DisaggregationMode.DECODE, self.scheduler.server_args, self.is_mla_backend, self.draft_is_mla_backend)
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
                kv_receiver_class = get_kv_class(TransferBackend.FAKE, KVClassType.RECEIVER)
            else:
                kv_receiver_class = get_kv_class(self.transfer_backend, KVClassType.RECEIVER)
            kv_receiver = kv_receiver_class(
                mgr=self.kv_manager,
                bootstrap_addr=f"{req.bootstrap_host}:{req.bootstrap_port}",
                bootstrap_room=req.bootstrap_room,
            )

            if self.scheduler.attn_tp_rank == 0:
                req.add_latency(RequestStage.DECODE_PREPARE)
            self.queue.append(DecodeRequest(req=req, kv_receiver=kv_receiver, waiting_for_input=False))

    def extend(self, reqs: list[Req], is_retracted: bool = False) -> None:
        """Add a request to the pending queue."""
        for req in reqs:
            self.add(req, is_retracted=is_retracted)

    def resume_retracted_reqs(self) -> list[Req]:
        # TODO refactor scheduling part, reuse with unified engine logic as much as possible

        # allocate memory
        resumed_reqs = []
        indices_to_remove = set()

        for i, req in enumerate(self.retracted_queue):
            if req.to_abort:
                prepare_abort(
                    req,
                    "abort retracted req",
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                    err_type=AbortCode.TransferFailed,
                )
                self.scheduler.stream_output([req], req.return_logprob)
                self.kv_allocator.free_group_flush()
                indices_to_remove.add(i)
                continue
            if self.req_to_token_pool.available_size() <= 0:
                logger.debug(f"[resume_retracted_reqs] no available token pool: {self.req_to_token_pool.available_size()}")
                break

            prefix_len = self._match_prefix_and_lock(req, req.origin_input_ids)
            current_req_origin_tokens = len(req.origin_input_ids) + len(req.output_ids) - prefix_len
            remaining_reserved_pages = self._remaining_decode_reserved_pages(len(resumed_reqs), current_req_origin_tokens)
            if remaining_reserved_pages <= 0:
                logger.debug(
                    f"[resume_retracted_reqs] insufficient reserved budget "
                    f"for req {req.rid=}: "
                    f"remaining_reserved_pages={remaining_reserved_pages} "
                    f"available_pages={self.kv_allocator.available_size()} "
                    f"reserved_pages_per_req={self._reserved_pages_per_decode_req()} "
                    f"reserved_req_count="
                    f"{self._decode_reserved_req_count(len(resumed_reqs))} "
                    f"current_req_origin_tokens={current_req_origin_tokens} "
                    f"evictable_pages={self.tree_cache.evictable_size() if self.tree_cache is not None else 0}"
                )
                # Try eviction before breaking.
                if self.tree_cache is not None and self.tree_cache.evictable_size() > 0:
                    need_page_num = 1 - remaining_reserved_pages
                    self.tree_cache.evict(need_page_num)
                    if not self._has_decode_reserved_budget(len(resumed_reqs), current_req_origin_tokens):
                        logger.debug(f"[resume_retracted_reqs] still not enough space after eviction for req {req.rid=}, breaking")
                        break
                    remaining_reserved_pages = self._remaining_decode_reserved_pages(len(resumed_reqs), current_req_origin_tokens)
                else:
                    logger.debug(f"[resume_retracted_reqs] no evictable space for req {req.rid=}, breaking")
                    break

            saved_payload = getattr(req, "kv_cache_cpu", None)
            if isinstance(saved_payload, RankLocalCpuCopy):
                plan = self._plan_rank_alloc(req, saved_payload, alloced_len=req.prefix_len)
                if not self.kv_allocator.check_rank_alloc(
                    plan.rank_alloc_num_tokens,
                    plan.alloced_num_tokens,
                    plan.page2rank_new,
                ):
                    logger.debug(f"[resume_retracted_reqs] rank_alloc infeasible for {req.rid=}, breaking")
                    break
            logger.debug(
                f"[DecodePreallocQueue]before resumed req {req} "
                f"remaining_reserved_pages={remaining_reserved_pages} "
                f"required_tokens_for_request="
                f"{len(req.origin_input_ids) + len(req.output_ids) + CLIP_MAX_NEW_TOKEN} "
                f"prefix_len={prefix_len}"
            )
            resumed_reqs.append(req)
            indices_to_remove.add(i)
            req.is_retracted = False
            req.extend_input_len = len(req.fill_ids) - req.prefix_len
            kv_loc = self._pre_alloc(req, is_retracted=True)
            self.req_to_token_pool.alloced_lens[req.req_pool_idx] += kv_loc.shape[0] + req.prefix_len
            # load from cpu, release cpu copy
            req.load_kv_cache(self.req_to_token_pool, self.token_to_kv_pool)
            logger.debug(f"[DecodePreallocQueue]resumed req {req}")

        self.retracted_queue = [entry for i, entry in enumerate(self.retracted_queue) if i not in indices_to_remove]
        return resumed_reqs

    def process_waiters(self) -> None:
        # Expiry/cleanup must run even when retracted requests block admission.
        self._update_handshake_waiters()
        remaining = []
        for entry in self.queue:
            if isinstance(entry.req.finished_reason, FinishAbort) or entry.req.to_abort:
                self._finish_prealloc_request(entry)
            else:
                remaining.append(entry)
        self.queue = remaining

    def _finish_prealloc_request(self, entry: DecodeRequest) -> None:
        req = entry.req
        cancel = getattr(entry.kv_receiver, "cancel_before_transfer", None)
        if cancel is not None:
            cancel()
        if req.prefix_len > 0 and req.last_node is not None:
            self.tree_cache.dec_lock_ref(req.last_node)
        req.last_node = None
        if not isinstance(req.finished_reason, FinishAbort):
            prepare_abort(req, "Aborted before decode KV allocation")
        self.scheduler.stream_output([req], req.return_logprob)

    # While every queued request finished its handshake and only waits for a
    # KV slot, poll them once per this many calls: enough to notice a failure
    # (e.g. the prealloc deadline, a lost prefill) without an all-reduce on
    # every scheduler step. A call count, not a clock, so every rank joins
    # the same all-reduce.
    WAITING_POLL_EVERY = 64

    def _update_handshake_waiters(self, candidates=None) -> None:
        waiters = self.queue if candidates is None else candidates
        if not waiters:
            return

        if candidates is None and all(decode_req.waiting_for_input for decode_req in waiters):
            self._waiting_poll_calls += 1
            if self._waiting_poll_calls % self.WAITING_POLL_EVERY:
                return

        polls = poll_and_all_reduce([decode_req.kv_receiver for decode_req in waiters], self.gloo_group, self.scheduler_comm_device)

        for _i, (decode_req, poll) in enumerate(zip(waiters, polls)):
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
                    err_type=AbortCode.TransferFailed,
                )
            else:
                raise ValueError(f"Unexpected poll case: {poll}")

    def _ceil_tokens_with_page_size(self, tokens, page_size):
        return page_size * ((tokens + page_size - 1) // page_size)

    def _plan_rank_alloc(
        self,
        req: Req,
        saved_payload: RankLocalCpuCopy,
        alloced_len: int,
    ) -> _RankAllocPlan:
        page_size = self.kv_allocator.page_size
        prefix_len = req.prefix_len
        assert prefix_len % page_size == 0, f"prefix_len {prefix_len} must be page-aligned for rank_alloc resume"
        assert alloced_len % page_size == 0, f"alloced_len {alloced_len} must be page-aligned for rank_alloc resume"

        total_len = len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
        new_tokens_needed = total_len - prefix_len

        prefix_page_count = prefix_len // page_size
        new_page_count = (new_tokens_needed + page_size - 1) // page_size
        saved_num_pages = len(saved_payload.page2rank)
        assert prefix_page_count + new_page_count == saved_num_pages, (
            f"page count mismatch on resume: prefix={prefix_page_count} new={new_page_count} saved={saved_num_pages}"
        )

        return _RankAllocPlan(
            rank_alloc_num_tokens=new_page_count * page_size,
            page2rank_new=saved_payload.page2rank[prefix_page_count:],
            alloced_num_tokens=alloced_len,
        )

    def pop_preallocated(self) -> list[DecodeRequest]:
        """Pop requests from the queue for pre-allocation."""
        preallocated_reqs = []
        indices_to_remove = set()
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
            prefix_len = self._match_prefix_and_lock(req, req.origin_input_ids)
            current_req_origin_tokens = len(req.origin_input_ids) - prefix_len
            remaining_reserved_pages = self._remaining_decode_reserved_pages(len(preallocated_reqs), current_req_origin_tokens)
            if remaining_reserved_pages <= 0:
                logger.debug(
                    f"[DecodePreallocQueue][insufficient]"
                    f"req={req.rid} remaining_reserved_pages={remaining_reserved_pages} "
                    f"available_pages={self.kv_allocator.available_size()} "
                    f"reserved_pages_per_req={self._reserved_pages_per_decode_req()} "
                    f"reserved_req_count="
                    f"{self._decode_reserved_req_count(len(preallocated_reqs))} "
                    f"current_req_origin_tokens={current_req_origin_tokens} "
                    f"evictable_pages={self.tree_cache.evictable_size() if self.tree_cache is not None else 0} "
                    f"prealloc_queue={len(self.queue)} transfer_queue={len(self.transfer_queue.queue)} "
                    f"waiting_queue={len(self.scheduler.waiting_queue)} "
                    f"retracted_queue={len(self.retracted_queue)} "
                    f"running_reqs={len(self.scheduler.running_batch.reqs) if self.scheduler.running_batch else 0}"
                )
                # Try eviction before breaking.
                if self.tree_cache is not None and self.tree_cache.evictable_size() > 0:
                    need_page_num = 1 - remaining_reserved_pages
                    self.tree_cache.evict(need_page_num)
                    if not self._has_decode_reserved_budget(len(preallocated_reqs), current_req_origin_tokens):
                        break
                    remaining_reserved_pages = self._remaining_decode_reserved_pages(len(preallocated_reqs), current_req_origin_tokens)
                else:
                    break

            # Bypass the periodic throttle at the actual admission boundary.
            # All holding ranks must reject the same expired request.
            self._update_handshake_waiters([decode_req])
            if isinstance(req.finished_reason, FinishAbort) or req.to_abort:
                self._finish_prealloc_request(decode_req)
                indices_to_remove.add(i)
                continue

            kv_loc = self._pre_alloc(decode_req.req)
            self.req_to_token_pool.alloced_lens[decode_req.req.req_pool_idx] += kv_loc.shape[0] + decode_req.req.prefix_len

            logger.debug(
                f"[DecodePreallocQueue][pop] enough cache for req={decode_req.req.rid} "
                f"remaining_reserved_pages={remaining_reserved_pages} "
                f"required_tokens_for_request="
                f"{len(decode_req.req.origin_input_ids) + CLIP_MAX_NEW_TOKEN} "
                f"prefix_len={decode_req.req.prefix_len} "
                f"alloced_lens = {self.req_to_token_pool.alloced_lens[decode_req.req.req_pool_idx]}"
            )

            kv_indices = (
                self.req_to_token_pool.req_to_token[decode_req.req.req_pool_idx][decode_req.req.prefix_len : len(decode_req.req.origin_input_ids)]
                .cpu()
                .numpy()
                .astype(np.int64)
            )
            state_indices = None
            if isinstance(self.token_to_kv_pool, HybridLinearKVPool):
                # Mamba hybrid model: single mamba state index
                state_indices = [self.req_to_token_pool.req_index_to_mamba_index_mapping[decode_req.req.req_pool_idx].cpu().numpy()]

            decode_req.metadata_buffer_index = self.req_to_metadata_buffer_idx_allocator.alloc()
            assert decode_req.metadata_buffer_index is not None
            page_indices = kv_to_page_indices(kv_indices, self.token_to_kv_pool.page_size)
            decode_mla_l1_5_args = None
            if hasattr(self.token_to_kv_pool, "enable_mla_l1_5_cache") and self.token_to_kv_pool.enable_mla_l1_5_cache:
                page_meta = self.token_to_kv_pool.get_page_transfer_metadata(page_indices)
                decode_mla_l1_5_args = page_meta

            if self.kv_transfer_metrics:
                num_kv_indices = len(decode_req.req.origin_input_ids)
                page_size = self.token_to_kv_pool.page_size
                num_pages = kv_to_page_num(num_kv_indices, page_size)
                if isinstance(self.token_to_kv_pool, HybridLinearKVPool):
                    transfer_size_bytes = num_pages * self.token_to_kv_pool.full_page_size_bytes + self.token_to_kv_pool.mamba_size
                else:
                    transfer_size_bytes = num_pages * self.token_to_kv_pool.page_size_bytes
                self.kv_transfer_metrics.log_kv_transfer_size(transfer_size_bytes)

            # Synthetic warmup requests do not contribute to cache hit metrics.
            if req.bootstrap_host != FAKE_BOOTSTRAP_HOST:
                self.total_decode_tokens += len(req.origin_input_ids)
                self.total_decode_hit_tokens += req.prefix_len

            if req.bootstrap_host == FAKE_BOOTSTRAP_HOST and getattr(self.token_to_kv_pool, "dsa_kv_int8_packed", False):
                from sglang.srt.disaggregation.fake.conn import initialize_packed_fake_kv

                initialize_packed_fake_kv([self.token_to_kv_pool, *(self.draft_token_to_kv_pools or [])], page_indices)
                for buf in self.metadata_buffers.get_buf(decode_req.metadata_buffer_index):
                    buf.zero_()
                # The forward thread uses a separate stream. Publish FAKE Success
                # only after this request's synthetic history is ready to read.
                torch.get_device_module(self.device).current_stream().synchronize()

            decode_req.kv_receiver.init(
                page_indices,
                decode_req.metadata_buffer_index,
                decode_prefix_len=decode_req.req.prefix_len,
                mla_l1_5_args=decode_mla_l1_5_args,
                state_indices=state_indices,
            )
            if self.scheduler.attn_tp_rank == 0:
                decode_req.req.add_latency(RequestStage.DECODE_BOOTSTRAP)
            preallocated_reqs.append(decode_req)
            indices_to_remove.add(i)

        self.queue = [entry for i, entry in enumerate(self.queue) if i not in indices_to_remove]

        return preallocated_reqs

    def _reserved_pages_per_decode_req(self) -> int:
        return (CLIP_MAX_NEW_TOKEN + self.kv_allocator.page_size - 1) // self.kv_allocator.page_size

    def _decode_reserved_req_count(self, num_candidate_reqs: int) -> int:
        running_reqs = len(self.scheduler.running_batch.reqs) if self.scheduler.running_batch is not None else 0
        return running_reqs + num_candidate_reqs + 1

    def _current_req_origin_pages(self, current_req_origin_tokens: int) -> int:
        if current_req_origin_tokens <= 0:
            return 0
        page_size = self.kv_allocator.page_size
        return (current_req_origin_tokens + page_size - 1) // page_size

    def _remaining_decode_reserved_pages(self, num_candidate_reqs: int, current_req_origin_tokens: int = 0) -> int:
        return self.kv_allocator.available_size() - (
            self._decode_reserved_req_count(num_candidate_reqs) * self._reserved_pages_per_decode_req()
            + self._current_req_origin_pages(current_req_origin_tokens)
        )

    def _has_decode_reserved_budget(self, num_candidate_reqs: int, current_req_origin_tokens: int = 0) -> bool:
        return self._remaining_decode_reserved_pages(num_candidate_reqs, current_req_origin_tokens) > 0

    def get_and_reset_decode_cache_hit_tokens(self) -> tuple[int, int]:
        """Return (hit_tokens, total_tokens) admitted since the last call and reset."""
        hit_tokens, total_tokens = self.total_decode_hit_tokens, self.total_decode_tokens
        self.total_decode_tokens = 0
        self.total_decode_hit_tokens = 0
        return hit_tokens, total_tokens

    def get_and_reset_decode_cache_hit_rate(self) -> float:
        """Compatibility for the legacy scheduler stats mixin."""
        hit_tokens, total_tokens = self.get_and_reset_decode_cache_hit_tokens()
        return hit_tokens / total_tokens if total_tokens > 0 else 0.0

    def _match_prefix_and_lock(self, req: Req, input_ids: list[int]) -> tuple[torch.Tensor, int, object | None]:
        """Match prefix and lock the corresponding tree node

        Returns:
            Tuple of (matched_prefix_pages, prefix_len, matched_last_node)
        """
        self.tree_cache.dec_lock_ref(req.last_node)
        req.last_node = None
        match_result = self.tree_cache.match_prefix(input_ids)
        (matched_prefix_pages, prefix_len, matched_last_node) = (
            match_result.device_indices,
            match_result.device_prefix_length,
            match_result.last_device_node,
        )
        # Lock the matched prefix to prevent eviction
        if prefix_len > 0:
            self.tree_cache.inc_lock_ref(matched_last_node)

        req.prefix_len = prefix_len
        req.prefix_page_ids = matched_prefix_pages
        req.last_node = matched_last_node
        return prefix_len

    def _pre_alloc(self, req: Req, is_retracted: bool = False) -> torch.Tensor:
        """Pre-allocate memory for req_to_token and token_kv_pool.

        When ``is_retracted`` is True the req is being resumed from an L1.5 DSA
        rank-local offload payload. In that case the non-prefix allocation is
        dispatched to ``kv_allocator.rank_alloc`` using the page2rank snapshot
        saved at retract time, so that newly-allocated pages line up rank-wise
        with the originals. This removes the need for a cross-rank all_gather
        on the offload path. Everything else — prefix handling, req_to_page /
        req_to_token writes, metadata population — is shared with the normal
        prealloc path.
        """
        assert req.req_pool_idx is None, f"req_pool_idx shoule be None when pre_alloc, but get {req.req_pool_idx}"
        if isinstance(self.req_to_token_pool, HybridReqToTokenPool):
            req_pool_indices = self.req_to_token_pool.alloc(1, [req])
        else:
            req_pool_indices = self.req_to_token_pool.alloc(1)
        assert req_pool_indices is not None

        req.req_pool_idx = req_pool_indices[0]
        req.oe_init = False
        # A retracted request restores its full snapshot right after this.
        req.pd_kv_ready = is_retracted

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
            self.kv_allocator.req_to_page[req.req_pool_idx, :prefix_page_count] = matched_prefix_pages

        else:
            new_tokens_needed = total_len

        alloced_len = self.req_to_token_pool.alloced_lens[req.req_pool_idx].item()
        if prefix_len > alloced_len:
            alloced_len = prefix_len
        # Allocate memory for new tokens
        if new_tokens_needed > 0:
            from sglang.srt.mem_cache.memory_pool.host_memory_pool import RankLocalCpuCopy

            saved_payload = getattr(req, "kv_cache_cpu", None) if is_retracted else None
            use_rank_alloc = isinstance(saved_payload, RankLocalCpuCopy)

            if use_rank_alloc:
                plan = self._plan_rank_alloc(req, saved_payload, alloced_len=alloced_len)
                kv_loc = self.kv_allocator.rank_alloc(
                    req.req_pool_idx,
                    plan.rank_alloc_num_tokens,
                    plan.alloced_num_tokens,
                    plan.page2rank_new,
                )
                kv_loc = kv_loc[:new_tokens_needed]
            else:
                kv_loc = self.kv_allocator.alloc(
                    req.req_pool_idx,
                    new_tokens_needed,
                    alloced_len,
                )
                assert kv_loc is not None, (
                    f"Failed to allocate {new_tokens_needed} tokens for request "
                    f"{req.rid} {self.tree_cache.evictable_size()=} "
                    f"{self.tree_cache.protected_size()=} "
                    f"{self.kv_allocator.available_size()=}"
                )
            if self.kv_allocator.token_slot_refs is not None:
                self.kv_allocator.token_slot_refs[kv_loc] += 1

        self.req_to_token_pool.write((req.req_pool_idx, slice(0, total_len)), torch.cat([prefix_kv_locs, kv_loc]))
        # populate metadata
        req.fill_ids = req.origin_input_ids + req.output_ids
        req.extend_input_len = len(req.fill_ids) - req.prefix_len

        return kv_loc


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
        tree_cache: RadixCache,
        scheduler_comm_device,
    ):
        self.queue: list[DecodeRequest] = []
        self.gloo_group = gloo_group
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.metadata_buffers = metadata_buffers
        self.scheduler = scheduler
        self.tree_cache = tree_cache
        self.scheduler_comm_device = scheduler_comm_device

    def add(self, decode_req: DecodeRequest) -> None:
        self.queue.append(decode_req)

    def extend(self, decode_reqs: list[DecodeRequest]) -> None:
        self.queue.extend(decode_reqs)

    def pop_transferred(self) -> list[DecodeRequest]:
        if not self.queue:
            return []

        polls = poll_and_all_reduce([decode_req.kv_receiver for decode_req in self.queue], self.gloo_group, self.scheduler_comm_device)

        transferred_reqs = []
        indices_to_remove = set()
        for i, (decode_req, poll) in enumerate(zip(self.queue, polls)):
            # Decode-initiated abort: kick the abort and keep parking the request until poll()
            # reports Failed. Releasing the KV buffer now risks prefill
            # RDMA-writing into a slot that was already handed to another
            # request.
            if decode_req.req.to_abort and poll not in (KVPoll.Failed, KVPoll.Success):
                if not decode_req.abort_initiated:
                    logger.info(f"[decode-abort] trigger abort rid={decode_req.req.rid} room={decode_req.req.bootstrap_room} poll={poll}")
                    decode_req.kv_receiver.abort()
                    decode_req.abort_initiated = True
                continue

            if poll == KVPoll.Failed:
                error_message = (
                    f"Decode transfer failed for request rank={self.scheduler.tp_rank} {decode_req.req.rid=} {decode_req.req.bootstrap_room=}"
                )
                try:
                    decode_req.kv_receiver.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                logger.error(error_message)
                prepare_abort(
                    decode_req.req,
                    error_message,
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                    err_type=AbortCode.TransferFailed,
                )
                self.scheduler.stream_output([decode_req.req], decode_req.req.return_logprob)

                # Clean up KV cache to prevent memory leak
                self.scheduler.cache_finished_req(decode_req.req)
                self.tree_cache.token_to_kv_pool_allocator.free_group_flush()
                indices_to_remove.add(i)
                continue
            elif poll == KVPoll.Success:
                decode_req.req.pd_kv_ready = True
                idx = decode_req.metadata_buffer_index
                (
                    output_id,
                    output_token_logprobs_val,
                    output_token_logprobs_idx,
                    output_top_logprobs_val,
                    output_top_logprobs_idx,
                    cached_tokens,
                    dequeue_time,
                    abort_flags,
                ) = self.metadata_buffers.get_buf(idx)
                decode_req.req.cached_tokens = cached_tokens[0].item()
                decode_req.req.dequeue_time = dequeue_time[0].item()
                decode_req.req.output_ids.append(output_id[0].item())

                if decode_req.req.return_logprob:
                    decode_req.req.output_token_logprobs_val.append(output_token_logprobs_val[0].item())
                    decode_req.req.output_token_logprobs_idx.append(output_token_logprobs_idx[0].item())
                    decode_req.req.output_top_logprobs_val.append(output_top_logprobs_val[: decode_req.req.top_logprobs_num].tolist())
                    decode_req.req.output_top_logprobs_idx.append(output_top_logprobs_idx[: decode_req.req.top_logprobs_num].tolist())
                if abort_flags[0].item() == 1:
                    # Grammar compilation failed on prefill side, mark for abort
                    logger.warning(f"Grammar compilation failed for req {decode_req.req.rid}, aborting")
                    prepare_abort(
                        decode_req.req,
                        "Grammar compilation failed for the provided schema",
                        status_code=HTTPStatus.BAD_REQUEST,
                    )
                if abort_flags[0].item() == 1:
                    # Grammar compilation failed on prefill side, mark for abort
                    logger.warning(f"Grammar compilation failed for req {decode_req.req.rid}, aborting")
                    prepare_abort(
                        decode_req.req,
                        "Grammar compilation failed for the provided schema",
                        status_code=HTTPStatus.BAD_REQUEST,
                    )

                # ===== DEBUG: Print KV cache sequence after transfer =====
                # self._debug_print_kv_cache(decode_req.req)
                # =========================================================

                if hasattr(decode_req.kv_receiver, "clear"):
                    decode_req.kv_receiver.clear()
                    decode_req.kv_receiver = None  # Ensure receiver is properly cleaned up

                # check_finished handles normal completion (e.g. max_new_tokens == 1)
                # For aborted requests, finished() is already True from prepare_abort above
                decode_req.req.check_finished()
                if decode_req.req.finished():
                    self.scheduler.stream_output([decode_req.req], decode_req.req.return_logprob)
                    self.scheduler.cache_finished_req(decode_req.req)
                    self.tree_cache.token_to_kv_pool_allocator.free_group_flush()
                else:
                    if self.scheduler.server_args.ret_first_token_no_waiting:
                        self.scheduler.stream_output([decode_req.req], decode_req.req.return_logprob)
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
            if self.scheduler.attn_tp_rank == 0:
                self.queue[i].req.add_latency(RequestStage.DECODE_TRANSFERRED)
            self.req_to_metadata_buffer_idx_allocator.free(idx)

        self.queue = [entry for i, entry in enumerate(self.queue) if i not in indices_to_remove]

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
            if hasattr(token_to_kv_pool, "k_buffer"):
                # MHATokenToKVPool
                k_cache_layer0 = token_to_kv_pool.get_key_buffer(0)
            elif hasattr(token_to_kv_pool, "get_key_buffer"):
                # Try generic get_key_buffer method
                with contextlib.suppress(BaseException):
                    k_cache_layer0 = token_to_kv_pool.get_key_buffer(0)

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
                    prefix_k_avg = k_avg_per_pos[: req.prefix_len]
                    non_prefix_k_avg = k_avg_per_pos[req.prefix_len :]
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

        configure_gc_warning(0.5)
        gc.collect()
        freeze_gc("event_loop_normal_disagg_decode")
        logger.info(f"{gc.get_threshold()=}")
        gc.set_threshold(2500, 10, 5)
        logger.info(f"{gc.get_threshold()=} {gc.get_count()=}")

        while True:
            # EPD: poll the encode receiver for admitted/failed MM requests
            # before processing any new arrivals.  This is a no-op on non-EPD nodes.
            self.poll_epd_admissions()

            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            # polling and allocating kv cache
            self.process_decode_queue()

            # Check hierarchical cache events (write_through completion, load_back completion, etc.)
            if self.enable_hierarchical_cache:
                self.tick_cache()

            batch = self.get_next_disagg_decode_batch_to_run()
            self.cur_batch = batch

            if batch:
                # Generate fake extend output.
                if batch.forward_mode.is_extend():
                    # Note: Logprobs should be handled on the prefill engine.
                    self.stream_output(batch.reqs, any(req.return_logprob for req in batch.reqs))
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
                len(self.waiting_queue) + len(self.disagg_decode_transfer_queue.queue) + len(self.disagg_decode_prealloc_queue.queue) == 0
            ):
                # When server is idle, do self-check and re-init some states
                self.check_memory()
                self.new_token_ratio = self.init_new_token_ratio

            if batch is None:
                self.log_idle_stats()

            self.last_batch = batch

    def _multi_step_decode_launch(self: Scheduler, batch: ScheduleBatch):
        for i in range(self.server_args.num_continuous_decode_steps):
            # launch cur batch
            result = self.run_batch(batch)
            # put cur batch and into result queue for async post process
            self.result_queue.append((batch.copy(run_times=i), result))
            # update cur batch for next launch
            batch.input_ids = result.next_token_ids
            if i < self.server_args.num_continuous_decode_steps - 1:
                # NOTE: This is not correct when MTP enabled, so it should never be used
                batch.seq_lens = batch.seq_lens + 1
                batch.seq_lens_sum += len(batch.reqs)

    def _multi_step_decode_post_process(self: Scheduler):
        for _i in range(self.server_args.num_continuous_decode_steps):
            tmp_batch, tmp_result = self.result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)

    def _schedule_next_batch_to_lauch(self: Scheduler):
        recv_reqs = self.recv_requests()
        self.process_input_requests(recv_reqs)
        self.process_decode_queue()
        batch: ScheduleBatch = self.get_next_disagg_decode_batch_to_run()
        if batch and batch.forward_mode.is_extend():
            self.stream_output(batch.reqs, any(req.return_logprob for req in batch.reqs))
            self.last_batch = batch
            batch: ScheduleBatch = self.get_next_disagg_decode_batch_to_run(True)
        # Idle or Decode
        dp_batch = self.prepare_dp_attn_batch(batch)

        if batch and batch.forward_mode.is_decode():
            assert len(batch.reqs) > 0, f"{batch=} {dp_batch=}"
        return batch, dp_batch

    def multi_step_decode_overlap_loop(self: Scheduler, batch: ScheduleBatch):
        self._multi_step_decode_launch(batch)
        self._multi_step_decode_post_process()

    @torch.no_grad()
    def event_loop_overlap_disagg_multi_step_decode(self: Scheduler):
        # When dp size is 1, there's no cpu all-gather overhead, we don't need multi-step decode
        assert self.server_args.dp_size > 1, "multi-step decode only support dp > 1 for now"
        assert self.server_args.enable_dp_attention
        self.result_queue = deque()
        # When we get a newly built extend batch, we need a EXTEND batch to merge into
        # running batch, but we will generate an idle batch to launch. So we record a
        # last dp batch.
        self.last_batch = None
        self.last_dp_batch = None

        def _loop_process(step_func=None):
            # idle/busy time tracking (1-minute window)
            total_busy_time = 0.0
            total_idle_time = 0.0
            stats_window_seconds = 60.0
            loop_start = time.time()
            configure_gc_warning(0.05)
            gc.collect()
            freeze_gc("event_loop_overlap_disagg_multi_step_decode")
            logger.info(f"{gc.get_threshold()=}")
            gc.set_threshold(2500, 10, 5)
            logger.info(f"{gc.get_threshold()=} {gc.get_count()=}")

            while True:
                self.poll_epd_admissions()

                tm_start = time.time()
                batch, dp_batch = self._schedule_next_batch_to_lauch()
                if self.enable_hierarchical_cache:
                    self.tick_cache()

                self.cur_batch = batch
                if dp_batch:
                    if self.last_dp_batch:
                        if dp_batch:
                            # Overlapping
                            self.multi_step_decode_overlap_loop(dp_batch)
                            if step_func:
                                step_func()
                        else:
                            assert batch.forward_mode.is_extend()
                    else:
                        # Warmup
                        if dp_batch:
                            self._multi_step_decode_launch(dp_batch)
                            if step_func:
                                step_func()
                            tmp_batch, tmp_result = self.result_queue.popleft()
                            self.process_batch_result(tmp_batch, tmp_result)
                        else:
                            assert batch.forward_mode.is_extend()
                elif self.last_dp_batch:
                    # Cool down
                    for _ in range(self.server_args.num_continuous_decode_steps - 1):
                        tmp_batch, tmp_result = self.result_queue.popleft()
                        self.process_batch_result(tmp_batch, tmp_result)

                if batch and batch.forward_mode.is_extend():
                    self.last_batch = batch
                else:
                    self.last_batch = dp_batch
                self.last_dp_batch = dp_batch

                # check_memory condition: no real batch, no inflight queues
                _should_check_memory = (
                    dp_batch is None
                    and batch is None
                    and self.last_batch is None
                    and self.last_dp_batch is None
                    and (len(self.waiting_queue) + len(self.disagg_decode_transfer_queue.queue) + len(self.disagg_decode_prealloc_queue.queue) == 0)
                )
                if _should_check_memory:
                    # When the server is idle, do self-check and re-init some states
                    self.check_memory()
                    self.new_token_ratio = self.init_new_token_ratio

                # idle = no real req being processed (batch is None); fake idle dp_batch does not count
                _is_idle = batch is None
                if _is_idle:
                    total_idle_time += time.time() - tm_start
                else:
                    total_busy_time += time.time() - tm_start

                elapsed = time.time() - loop_start
                if elapsed >= stats_window_seconds:
                    self.decode_idle_ratio = total_idle_time / elapsed
                    self.decode_busy_ratio = total_busy_time / elapsed
                    total_idle_time = 0.0
                    total_busy_time = 0.0
                    loop_start = time.time()

        if self.npu_custom_profiler:
            with self.npu_custom_profiler as p:
                p.start()
                _loop_process(p.step)  # do npu process
                p.stop()
        else:
            _loop_process()

    @torch.no_grad()
    def event_loop_overlap_disagg_decode(self: Scheduler):
        self.result_queue = deque()
        self.last_batch: ScheduleBatch | None = None
        self.last_batch_in_queue = False  # last batch is modified in-place, so we need another variable to track if it's extend

        def _loop_process(step_func=None):
            configure_gc_warning(0.5)
            gc.collect()
            freeze_gc("event_loop_normal_disagg_decode")
            logger.info(f"{gc.get_threshold()=}")
            gc.set_threshold(2500, 10, 5)
            logger.info(f"{gc.get_threshold()=} {gc.get_count()=}")

            while True:
                # EPD: poll the encode receiver for admitted/failed MM requests
                # before processing any new arrivals.  This is a no-op on non-EPD nodes.
                self.poll_epd_admissions()

                recv_reqs = self.recv_requests()
                self.process_input_requests(recv_reqs)
                # polling and allocating kv cache
                self.process_decode_queue()

                # Check hierarchical cache events (write_through completion, load_back completion, etc.)
                if self.enable_hierarchical_cache:
                    self.tick_cache()

                batch = self.get_next_disagg_decode_batch_to_run()
                self.cur_batch = batch
                last_batch_in_queue = False

                if batch:
                    # Generate fake extend output.
                    if batch.forward_mode.is_extend():
                        # Note: Logprobs should be handled on the prefill engine.
                        self.stream_output(batch.reqs, any(req.return_logprob for req in batch.reqs))
                        self.last_batch = batch
                        batch = self.get_next_disagg_decode_batch_to_run(True)
                        self.cur_batch = batch
                    if self.server_args.enable_dp_attention:
                        self.prepare_dp_attn_batch(batch)
                    result = self.run_batch(batch)
                    self.result_queue.append((batch.copy(), result))
                    if step_func:
                        step_func()  # p.step()

                    if (self.last_batch is None) or (not self.last_batch_in_queue):
                        # Create a dummy first batch to start the pipeline for overlap schedule.
                        # It is now used for triggering the sampling_info_done event.
                        tmp_batch = ScheduleBatch(
                            reqs=None,
                            forward_mode=ForwardMode.DUMMY_FIRST,
                            next_batch_sampling_info=(
                                self.tp_worker.cur_sampling_info if self.draft_worker is None else self.draft_worker.cur_sampling_info
                            ),
                        )
                        self.set_next_batch_sampling_info_done(tmp_batch)
                    last_batch_in_queue = True

                elif self.server_args.enable_dp_attention:
                    batch, result = self._prepare_idle_batch_and_run(None, delay_process=True)
                    if batch:
                        if step_func:
                            step_func()
                        self.result_queue.append((batch.copy(), result))
                        last_batch_in_queue = True

                # Process the results of the previous batch but skip if the last batch is extend
                if self.last_batch and self.last_batch_in_queue:
                    tmp_batch, tmp_result = self.result_queue.popleft()
                    tmp_batch.next_batch_sampling_info = (
                        self.tp_worker.cur_sampling_info if self.draft_worker is None else self.draft_worker.cur_sampling_info
                    )
                    self.process_batch_result(tmp_batch, tmp_result)
                elif self.running_batch is not None and self.draft_worker:
                    # In overlap + MTP mode, correct reserve num tokens here to ensure
                    # sufficient allocation
                    for req in self.running_batch.reqs:
                        req.reserve_num_tokens = self.server_args.speculative_num_draft_tokens

                if batch is None and (
                    len(self.waiting_queue) + len(self.disagg_decode_transfer_queue.queue) + len(self.disagg_decode_prealloc_queue.queue) == 0
                ):
                    # When the server is idle, do self-check and re-init some states
                    self.check_memory()
                    self.new_token_ratio = self.init_new_token_ratio

                if batch is None:
                    self.log_idle_stats()

                self.last_batch = batch
                self.last_batch_in_queue = last_batch_in_queue

        if self.npu_custom_profiler:
            with self.npu_custom_profiler as p:
                p.start()
                _loop_process(p.step)  # do npu process
                p.stop()
        else:
            _loop_process()

    def get_next_disagg_decode_batch_to_run(self: Scheduler, only_merge=False) -> tuple[ScheduleBatch, bool] | None:
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

        new_prebuilt_batch = None if only_merge else self.get_new_prebuilt_batch()

        ret: ScheduleBatch | None = None
        if new_prebuilt_batch:
            ret = new_prebuilt_batch
        else:
            if self.running_batch is None or self.running_batch.is_empty():
                ret = None
            else:
                self.running_batch = self.update_running_batch(self.running_batch, self.last_batch)
                ret = self.running_batch if (self.running_batch is not None and not self.running_batch.is_empty()) else None
        self.update_oe_info(ret)
        return ret

    def get_new_prebuilt_batch(self: Scheduler) -> ScheduleBatch | None:
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
        can_run_list: list[Req] = []
        waiting_queue: list[Req] = []

        for i in range(len(self.waiting_queue)):
            req = self.waiting_queue[i]
            # we can only add at least `num_not_used_batch` new batch to the running queue
            if i < num_not_used_batch:
                can_run_list.append(req)
                logger.debug(f"[Scheduler][ConstrcutBatch] adding req {req} to running queue")
                if self.attn_tp_rank == 0:
                    req.add_latency(RequestStage.DECODE_WAITING)
                req.init_next_round_input()  # prealloc has already prefix matched, no need to match again here
            else:
                waiting_queue.append(req)

        self.waiting_queue = waiting_queue
        if len(can_run_list) == 0:
            return None

        # construct a schedule batch with those requests and mark as decode
        draft_token_num = self.server_args.speculative_num_draft_tokens if self.draft_worker else 0
        spec_num_steps = self.server_args.speculative_num_steps if self.draft_worker else 0
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
            self.server_args.num_continuous_decode_steps,
        )

        # construct fake completed prefill
        new_batch.prepare_for_prebuilt_extend()
        new_batch.process_prebuilt_extend(self.server_args, self.model_config)

        if self.draft_worker:
            self.draft_worker.renew_scaling_penalty(new_batch.req_pool_indices)
            self.speculative_sampling_info.reset(new_batch.req_pool_indices)

        return new_batch

    def process_decode_queue(self: Scheduler):
        self.disagg_decode_prealloc_queue.process_waiters()
        # try to resume retracted requests if there are enough space for another `num_reserved_decode_tokens` decode steps
        # 1. Retracted requests have the highest priority
        resumed_reqs = self.disagg_decode_prealloc_queue.resume_retracted_reqs()
        self.num_retracted_reqs -= len(resumed_reqs)
        self.waiting_queue.extend(resumed_reqs)
        if len(self.disagg_decode_prealloc_queue.retracted_queue) > 0:
            # if there are still retracted requests, we do not allocate new requests
            # but we allow reqs whose kv cache have already been allocated to run
            logger.debug(
                "[Scheduler][process_decode_queue] there are still "
                f"{len(self.disagg_decode_prealloc_queue.retracted_queue)} requests in retracted queue"
            )
        else:
            # 2. Newly arrived Reqs have the lowest priority
            req_conns = self.disagg_decode_prealloc_queue.pop_preallocated()
            self.disagg_decode_transfer_queue.extend(req_conns)

        # 3. Allocated reqs have the second highest priority
        alloc_reqs = self.disagg_decode_transfer_queue.pop_transferred()  # the requests which kv has arrived
        self.waiting_queue.extend(alloc_reqs)
