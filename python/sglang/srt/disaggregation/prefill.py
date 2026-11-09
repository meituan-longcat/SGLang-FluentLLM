"""
Life cycle of a request in the prefill server

1. Bootstrap Queue
    a. Initialize a sender for each request
    b. Use the queue to store requests whose bootstrap (handshake and preallocation) has not finished
    c. Poll senders to check bootstrap state
    d. Once bootstrap is complete, move request to Waiting Queue

2. Waiting Queue
    a. Use PrefillAdder to pop requests
    b. Run forward
    c. Add the request to Inflight Queue

3. Inflight Queue
    a. Poll (non-blocking) the sender of the request
    b. Once the transfer has finished, return the request
"""

from __future__ import annotations

import gc
import threading
import time
from collections import deque
from http import HTTPStatus
from typing import TYPE_CHECKING

import numpy as np
import torch  # type: ignore

from sglang.srt.disaggregation.base import BaseKVManager, KVArgs, KVPoll
from sglang.srt.disaggregation.kv_manager_queue_mixin import KVManagerQueueMixin, get_transfer_buffer_device_types
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
from sglang.srt.managers.req import (
    AbortCode,
    FinishLength,
    Req,
    RequestStage,
)
from sglang.srt.managers.scheduling.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import get_colorful_logger
from sglang.srt.utils.common import configure_gc_warning, freeze_gc

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from sglang.srt.layers.sampler.logits_processor import LogitsProcessorOutput
    from sglang.srt.managers.scheduling.schedule_batch import GenerationBatchResult
    from sglang.srt.managers.scheduling.scheduler import Scheduler
    from sglang.srt.mem_cache.memory_pool import KVCache
    from sglang.srt.mem_cache.memory_pool.kv_allocator import KVAllocator


logger = get_colorful_logger(__name__)


class PrefillBootstrapQueue(KVManagerQueueMixin):
    """
    Store the requests in bootstrapping
    """

    def __init__(
        self,
        token_to_kv_pool: KVCache,
        kv_allocator: KVAllocator,
        draft_token_to_kv_pools: list[KVCache | None],
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        metadata_buffers: MetadataBuffers,
        tp_rank: int,
        tp_size: int,
        bootstrap_port: int,
        gloo_group: ProcessGroup,
        transfer_backend: TransferBackend,
        scheduler: Scheduler,
        pp_rank: int,
        pp_size: int,
    ):
        self.token_to_kv_pool = token_to_kv_pool
        self.kv_allocator = kv_allocator
        self.draft_token_to_kv_pools = draft_token_to_kv_pools
        self.is_mla_backend = is_mla_backend(token_to_kv_pool)
        if self.draft_token_to_kv_pools is None:
            self.draft_is_mla_backend = self.is_mla_backend
        else:
            self.draft_is_mla_backend = is_mla_backend(draft_token_to_kv_pools[0])
        self.metadata_buffers = metadata_buffers
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.transfer_backend = transfer_backend
        self.scheduler = scheduler
        self.kv_manager = self._init_kv_manager()
        self.queue: list[Req] = []
        self.gloo_group = gloo_group
        self.bootstrap_port = bootstrap_port

    def store_prefill_results(self, idx: int, token_id: int):
        assert token_id >= 0, f"token_id: {token_id} is negative"
        output_id_buffer = self.metadata_buffers[0]
        output_id_buffer[idx] = token_id

    # TODO: use KVManagerQueueMixin method to avoid the repetition of prefill.py and decode.py
    def _init_kv_manager(self) -> BaseKVManager:
        kv_args = KVArgs()
        kv_args.tp_rank = self.tp_rank
        kv_args.pp_rank = self.pp_rank
        kv_args.prefill_start_layer = self.token_to_kv_pool.start_layer
        kv_args.prefill_end_layer = self.token_to_kv_pool.end_layer

        kv_data_ptrs, kv_data_lens, kv_item_lens = self.token_to_kv_pool.get_contiguous_buf_infos()
        kv_buffer_device_types = get_transfer_buffer_device_types(self.token_to_kv_pool, len(kv_data_ptrs))
        # [[layer0buf0, layer0buf1...], [layer1buf0, layer1buf1...], ...]
        offsets = self.token_to_kv_pool.get_layerwise_buf_info_offsets()
        target_layer_num = self.token_to_kv_pool.layer_num
        draft_layer_num = 0
        if self.draft_token_to_kv_pools is not None:
            for draft_token_to_kv_pool in self.draft_token_to_kv_pools:
                draft_layer_num += draft_token_to_kv_pool.layer_num
                draft_kv_data_ptrs, draft_kv_data_lens, draft_kv_item_lens = draft_token_to_kv_pool.get_contiguous_buf_infos()
                kv_buffer_device_types += get_transfer_buffer_device_types(draft_token_to_kv_pool, len(draft_kv_data_ptrs))
                draft_offsets = draft_token_to_kv_pool.get_layerwise_buf_info_offsets(len(kv_data_ptrs))
                kv_data_ptrs += draft_kv_data_ptrs
                kv_data_lens += draft_kv_data_lens
                kv_item_lens += draft_kv_item_lens
                offsets += draft_offsets

        kv_args.kv_data_ptrs = kv_data_ptrs
        kv_args.kv_data_lens = kv_data_lens
        kv_args.kv_item_lens = kv_item_lens
        kv_args.buffer_device_types = kv_buffer_device_types
        kv_args.target_layer_num = target_layer_num
        kv_args.draft_layer_num = draft_layer_num
        kv_args.draft_enable_mla_l1_5_cache = bool(
            self.draft_token_to_kv_pools and getattr(self.draft_token_to_kv_pools[0], "enable_mla_l1_5_cache", False)
        )
        kv_args.offsets = offsets

        # Define req -> input ids buffer
        kv_args.aux_data_ptrs, kv_args.aux_data_lens, kv_args.aux_item_lens = self.metadata_buffers.get_buf_infos()
        kv_args.ib_device = self.scheduler.server_args.disaggregation_ib_device
        kv_args.gpu_id = self.scheduler.gpu_id

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
        kv_manager = kv_manager_class(kv_args, DisaggregationMode.PREFILL, self.scheduler.server_args, self.is_mla_backend, self.draft_is_mla_backend)
        return kv_manager

    def add(self, req: Req) -> None:
        if req.bootstrap_host == FAKE_BOOTSTRAP_HOST:
            # Fake transfer for warmup reqs
            kv_sender_class = get_kv_class(TransferBackend.FAKE, KVClassType.SENDER)
        else:
            kv_sender_class = get_kv_class(self.transfer_backend, KVClassType.SENDER)
        req.disagg_kv_sender = kv_sender_class(
            mgr=self.kv_manager,
            bootstrap_addr=f"{req.bootstrap_host}:{self.bootstrap_port}",
            bootstrap_room=req.bootstrap_room,
        )
        self._process_req(req)
        if self.scheduler.global_rank == 0:
            req.add_latency(RequestStage.PREFILL_PREPARE)
        self.queue.append(req)

    def extend(self, reqs: list[Req]) -> None:
        for req in reqs:
            self.add(req)

    def _process_req(self, req: Req) -> None:
        """
        Set max_new_tokens = 1, so PrefillAdder memory estimation is accurate
        """
        req.sampling_params.max_new_tokens = 1

    def pop_bootstrapped(self) -> list[Req]:
        """
        pop the reqs which has finished bootstrapping
        """
        bootstrapped_reqs = []
        failed_reqs = []
        indices_to_remove = set()

        if len(self.queue) == 0:
            return []
        polls = poll_and_all_reduce([req.disagg_kv_sender for req in self.queue], self.gloo_group)

        for i, (req, poll) in enumerate(zip(self.queue, polls)):
            if poll == KVPoll.Bootstrapping and not req.to_abort:
                continue
            elif poll == KVPoll.Failed or req.to_abort:
                error_message = f"Prefill bootstrap failed for request rank={self.tp_rank} {req.rid=} {req.bootstrap_room=}"
                self.scheduler._release_l3_prefetch_ready(req, detach=True)
                try:
                    req.disagg_kv_sender.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                logger.error(error_message)
                prepare_abort(req, error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR, err_type=AbortCode.TransferFailed)
                self.scheduler.stream_output([req], req.return_logprob)
                indices_to_remove.add(i)
                failed_reqs.append(req)
                continue

            # KV.WaitingForInput
            # EPD deferred tokenization: origin_input_ids may be None if the
            # request was admitted but tokenization hasn't completed yet.
            # Skip this iteration; pop_bootstrapped will retry next cycle.
            if req.origin_input_ids is None:
                continue
            num_kv_indices = len(req.origin_input_ids)
            if self.req_to_metadata_buffer_idx_allocator.available_size() == 0:
                break

            req.metadata_buffer_index = self.req_to_metadata_buffer_idx_allocator.alloc()
            assert req.metadata_buffer_index is not None
            num_pages = kv_to_page_num(num_kv_indices, self.token_to_kv_pool.page_size)

            # Calculate suffix pages for sender initialization
            decode_prefix_len = 0
            if hasattr(req.disagg_kv_sender, "kv_mgr") and hasattr(req.disagg_kv_sender.kv_mgr, "receive_decode_prefix_info"):
                decode_prefix_len = req.disagg_kv_sender.kv_mgr.receive_decode_prefix_info(req.bootstrap_room)
            prefix_pages = decode_prefix_len // self.token_to_kv_pool.page_size

            req.disagg_kv_sender.init(num_pages - prefix_pages, req.metadata_buffer_index)
            if self.scheduler.global_rank == 0:
                req.add_latency(RequestStage.PREFILL_BOOTSTRAP)
            bootstrapped_reqs.append(req)
            indices_to_remove.add(i)

        self.queue = [entry for i, entry in enumerate(self.queue) if i not in indices_to_remove]

        return bootstrapped_reqs


class SchedulerDisaggregationPrefillMixin:
    """
    Mixin for Scheduler to handle disaggregation prefill
    """

    @torch.no_grad()
    def event_loop_normal_disagg_prefill(self: Scheduler):
        """A normal scheduler loop for prefill worker in disaggregation mode."""

        def _loop_process(step_func=None):

            configure_gc_warning(0.5)
            gc.collect()
            freeze_gc("event_loop_normal_disagg_prefill")
            logger.info(f"{gc.get_threshold()=}")
            gc.set_threshold(2500, 10, 5)
            logger.info(f"{gc.get_threshold()=} {gc.get_count()=}")

            while True:
                # EPD: poll the encode receiver for admitted/failed MM requests
                # before processing any new arrivals.  This is a no-op on non-EPD nodes.
                self.poll_epd_admissions()
                recv_reqs = self.recv_requests()
                self.process_input_requests(recv_reqs)
                # Check if the grammar is ready in the grammar queue
                if self.grammar_queue:
                    self.move_ready_grammar_requests()
                self.waiting_queue.extend(self.disagg_prefill_bootstrap_queue.pop_bootstrapped())
                if self.enable_hicache_storage:
                    self.process_prefetches()
                batch = self.get_new_batch_prefill()

                # Handle DP attention
                if self.server_args.enable_dp_attention:
                    batch = self.prepare_dp_attn_batch(batch)
                self.update_oe_info(batch)
                self.cur_batch = batch

                if batch:
                    if self.enable_layerwise_transfer and not batch.forward_mode.is_idle():
                        self.launch_send_async(batch)
                    result = self.run_batch(batch)
                    self.process_batch_result_disagg_prefill(batch, result)
                    if step_func:
                        step_func()  # p.step()

                # launch d2h AFTER cache_finished_req so write_backup is already in d2h_write_q_todo
                if self.enable_hierarchical_cache:
                    self.tree_cache.tick()

                if len(self.disagg_prefill_inflight_queue) > 0:
                    self.process_disagg_prefill_inflight_queue()

                if batch is None and len(self.disagg_prefill_inflight_queue) == 0:
                    self.check_memory()
                    self.new_token_ratio = self.init_new_token_ratio

                if batch is None:
                    self.log_idle_stats()

                self.last_batch = batch
                # HACK (byronhsu): reset the batch_is_full flag because we never enter update_running_batch which resets it
                # Otherwise, it hangs under high concurrency
                self.batch_is_full = False

        if self.npu_custom_profiler:
            with self.npu_custom_profiler as p:
                p.start()
                _loop_process(p.step)  # do npu process
                p.stop()
        else:
            _loop_process()

    @torch.no_grad()
    def event_loop_overlap_disagg_prefill(self: Scheduler):
        self.result_queue = deque()

        while True:
            # EPD: poll the encode receiver for admitted/failed MM requests
            # before processing any new arrivals.  This is a no-op on non-EPD nodes.
            self.poll_epd_admissions()
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            # Check if the grammar is ready in the grammar queue
            if self.grammar_queue:
                self.move_ready_grammar_requests()
            self.waiting_queue.extend(self.disagg_prefill_bootstrap_queue.pop_bootstrapped())
            if self.enable_hicache_storage:
                self.process_prefetches()
            batch = self.get_new_batch_prefill()

            # Handle DP attention
            if self.server_args.enable_dp_attention:
                batch = self.prepare_dp_attn_batch(batch)
            self.update_oe_info(batch)
            self.cur_batch = batch

            if batch:
                if self.enable_layerwise_transfer and not batch.forward_mode.is_idle():
                    self.launch_send_async(batch)
                result = self.run_batch(batch)
                self.result_queue.append((batch.copy(), result))

                if self.last_batch is None:
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

            if self.last_batch:
                tmp_batch, tmp_result = self.result_queue.popleft()
                self.process_batch_result_disagg_prefill(tmp_batch, tmp_result)

            # launch d2h AFTER process_batch_result_disagg_prefill so write_backup is already queued
            if self.enable_hierarchical_cache:
                self.tree_cache.tick()

            if len(self.disagg_prefill_inflight_queue) > 0:
                self.process_disagg_prefill_inflight_queue()

            if batch is None and len(self.disagg_prefill_inflight_queue) == 0:
                self.check_memory()
                self.new_token_ratio = self.init_new_token_ratio

            if batch is None:
                self.log_idle_stats()

            self.last_batch = batch
            # HACK (byronhsu): reset the batch_is_full flag because we never enter update_running_batch which resets it
            # Otherwise, it hangs under high concurrency
            self.batch_is_full = False

    def launch_send_async(self: Scheduler, batch: ScheduleBatch):
        with self.kv_transfer_manager.add_batch(is_idle=batch.forward_mode.is_idle()):
            output_buffer_indices: list[int] = []
            output_top_logprobs_indices: list[tuple[int, int]] = []
            output_token_logprobs_indices: list[tuple[int, int]] = []
            batch_abort_flags: list[int] = []
            cached_tokens = []
            for idx, req in enumerate(batch.reqs):
                last_chunk = self.chunked_req is None or req.rid != self.chunked_req.rid
                self.add_send_task(req, last_chunk=last_chunk, skip_layerwise_send=False)
                output_buffer_indices.append(req.metadata_buffer_index)
                batch_abort_flags.append(int(req.to_abort_grammar))
                cached_tokens.append(req.cached_tokens)
                if req.top_logprobs_num > 0:
                    output_token_logprobs_indices.append((idx, req.metadata_buffer_index))
                if req.token_ids_logprob is not None:
                    output_top_logprobs_indices.append((idx, req.metadata_buffer_index))

            abort_flags_tensor = torch.tensor(batch_abort_flags, dtype=torch.int32).to(self.device)
            cached_tokens_tensor = torch.tensor(cached_tokens, dtype=torch.int32).to(self.device)
            output_buffer_indices = torch.tensor(output_buffer_indices).to(self.device)

            def set_aux_fn(output_ids: torch.Tensor, logits_output: LogitsProcessorOutput):
                self.disagg_metadata_buffers.set_buf_by_batch(
                    output_ids=output_ids,
                    output_buffer_indices_d=output_buffer_indices,
                    logits_output=logits_output,
                    output_token_logprobs_indices=output_token_logprobs_indices,
                    output_top_logprobs_indices=output_top_logprobs_indices,
                    cached_tokens_d=cached_tokens_tensor,
                    abort_flags_d=abort_flags_tensor,
                )
                self.step_counter.record_aux()

            batch.disagg_set_aux_fn = set_aux_fn

    def process_batch_result_disagg_prefill(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
        launch_done: threading.Event | None = None,
    ) -> None:
        """
        Transfer kv for prefill completed requests and add it into disagg_prefill_infight_queue
        Adapted from process_batch_result_prefill
        """

        (
            logits_output,
            next_token_ids,
            extend_input_len_per_req,
            extend_logprob_start_len_per_req,
            bid,
        ) = (
            result.logits_output,
            result.next_token_ids,
            result.extend_input_len_per_req,
            result.extend_logprob_start_len_per_req,
            result.bid,
        )

        logprob_pt = 0
        # Transfer kv for prefill completed requests and add it into disagg_prefill_infight_queue
        if self.enable_overlap:
            # wait
            if self.draft_worker:
                (
                    logits_output,
                    next_token_ids,
                    _,
                    _,
                    _,
                    _,
                ) = self.draft_worker.resolve_batch_result(bid)
            else:
                logits_output, next_token_ids, _, _ = self.tp_worker.resolve_last_batch_result(launch_done)
        else:
            if batch.forward_mode.is_idle():
                return
            next_token_ids = result.next_token_ids.tolist()
            if batch.return_logprob:
                if logits_output.next_token_logprobs is not None:
                    logits_output.next_token_logprobs = logits_output.next_token_logprobs.tolist()
                if logits_output.input_token_logprobs is not None:
                    logits_output.input_token_logprobs = tuple(logits_output.input_token_logprobs.tolist())

        for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
            req: Req
            if req.first_chunk_forward_start_time is None:
                req.first_chunk_forward_start_time = req.last_tic

            end_idx = batch.prefill_batch_info[i].context_len
            if req.is_chunked <= 0:
                # There is no output_ids for prefill
                req.output_ids.append(next_token_id)
                self.tree_cache.cache_unfinished_req(req)  # update the tree and lock
                ttft = time.time() - req.created_time
                logger.info(f"== prefill done, rid: {req.rid}, input: {len(req.origin_input_ids)}, ttft: {ttft * 1000:.2f}ms")
                if self.global_rank == 0:
                    now = time.monotonic()
                    total_forward_time = now - req.first_chunk_forward_start_time
                    if getattr(self, "metrics_collector", None):
                        self.metrics_collector.observe_request_latency_seconds(RequestStage.PREFILL_FORWARD.value, total_forward_time)
                    req.last_tic = now
                self.disagg_prefill_inflight_queue.append(req)
                if req.return_logprob:
                    assert extend_logprob_start_len_per_req is not None
                    assert extend_input_len_per_req is not None
                    extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                    extend_input_len = extend_input_len_per_req[i]
                    num_input_logprobs = extend_input_len - extend_logprob_start_len
                    self.add_logprob_return_values(
                        i,
                        req,
                        logprob_pt,
                        next_token_ids,
                        num_input_logprobs,
                        logits_output,
                    )
                    logprob_pt += num_input_logprobs
                self.add_send_task(req, last_chunk=True)

                if req.grammar is not None:
                    try:
                        req.grammar.accept_token(next_token_id)
                    except Exception:
                        logger.error(
                            f"Grammar accept_token failed:\n"
                            f"  rid: {req.rid}\n"
                            f"  next_token_id: {next_token_id}\n"
                            f"  sampling_params: {req.sampling_params}\n"
                            f"  origin_input_text (last 500 chars): {req.origin_input_text[-500:]}\n"
                            f"  output_ids: {req.output_ids}\n"
                        )
                        req.to_abort_grammar = True
                        self.disagg_metadata_buffers.abort_flags[req.metadata_buffer_index][0] = 1
                    else:
                        req.grammar.finished = req.finished()
            else:
                # being chunked reqs' prefill is not finished
                req.is_chunked -= 1

                if req.return_logprob:
                    extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                    extend_input_len = extend_input_len_per_req[i]
                    if extend_logprob_start_len < extend_input_len:
                        # Update input logprobs.
                        num_input_logprobs = extend_input_len - extend_logprob_start_len
                        self.add_input_logprob_return_values(
                            i,
                            req,
                            logits_output,
                            logprob_pt,
                            num_input_logprobs,
                            last_prefill_chunk=False,
                        )
                        logprob_pt += num_input_logprobs

                self.tree_cache.cache_unfinished_req(req, end_idx)
                self.add_send_task(req, last_chunk=False, end_idx=end_idx)

        # We need to remove the sync in the following function for overlap schedule.
        self.set_next_batch_sampling_info_done(batch)

        self.log_prefill_mfu()

    def process_disagg_prefill_inflight_queue(self: Scheduler):
        """
        Poll the requests in the middle of transfer. If done, return the request.
        """
        if len(self.disagg_prefill_inflight_queue) == 0:
            return []

        done_reqs = []

        polls = poll_and_all_reduce(
            [req.disagg_kv_sender for req in self.disagg_prefill_inflight_queue],
            self.attn_tp_pp_cpu_group,
        )

        undone_reqs: list[Req] = []
        # Check .poll() for the reqs in disagg_prefill_inflight_queue. If Success, respond to the client and remove it from the queue
        for req, poll in zip(self.disagg_prefill_inflight_queue, polls):
            if poll in [KVPoll.WaitingForInput, KVPoll.Transferring]:
                undone_reqs.append(req)
            elif poll == KVPoll.Success:  # transfer done
                self.tree_cache.cache_finished_req(req)  # unlock the tree
                self.kv_allocator.free_group_flush()
                req.finished_reason = FinishLength(length=0)
                # FIXME: clean up req's data in transfer engine
                if hasattr(req.disagg_kv_sender, "clear"):
                    req.disagg_kv_sender.clear()
                done_reqs.append(req)
            elif poll == KVPoll.Failed or req.to_abort:
                error_message = f"Prefill transfer failed for request rank={self.tp_rank} {req.rid=} {req.bootstrap_room=} {self.pp_rank=}"
                try:
                    req.disagg_kv_sender.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                logger.warning(error_message)
                self.tree_cache.cache_finished_req(req)  # unlock the tree
                self.kv_allocator.free_group_flush()
                prepare_abort(req, error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR, err_type=AbortCode.TransferFailed)
                done_reqs.append(req)

        for req in done_reqs:
            req: Req
            if self.global_rank == 0:
                req.add_latency(RequestStage.PREFILL_TRANSFER_KV_CACHE)
            self.disagg_prefill_bootstrap_queue.req_to_metadata_buffer_idx_allocator.free(req.metadata_buffer_index)
            req.metadata_buffer_index = -1

        # Stream requests which have finished transfer
        if self.pp_group.is_last_rank:
            self.stream_output(
                done_reqs,
                any(req.return_logprob for req in done_reqs),
                None,
            )

        # Publish KV events after processing batch results
        self.kv_event_publisher.publish_kv_events()

        self.disagg_prefill_inflight_queue = undone_reqs

    def add_send_task(
        self: Scheduler,
        req: Req,
        last_chunk: bool = False,
        end_idx: int | None = None,
        skip_layerwise_send: bool = True,
    ) -> None:
        skip_send = self.enable_layerwise_transfer and skip_layerwise_send

        if not self.enable_layerwise_transfer and last_chunk:
            assert req.metadata_buffer_index >= 0
            self.disagg_metadata_buffers.set_buf(req)

        if not skip_send:
            logger.debug(f"[add_send_task] Calling send_kv_chunk for req={req.rid}")
            self.send_kv_chunk(req, last_chunk, end_idx)
        else:
            logger.debug(f"[add_send_task] Skipping send for req={req.rid} (layerwise transfer enabled)")

    def send_kv_chunk(
        self: Scheduler,
        req: Req,
        last_chunk: bool = False,
        end_idx: int | None = None,
    ) -> None:
        """
        Send a prefilled chunk to the decode server, skipping existing prefixes from decode side
        """
        # Add diagnostic log: record the start of KV block sending
        # send_start_time = time.time()
        logger.debug(f"[send_kv_chunk] START: Sending KV chunk for req={req.rid}, last_chunk={last_chunk}, end_idx={end_idx}")

        page_size = self.token_to_kv_pool.page_size
        start_idx = req.start_send_idx

        decode_prefix_len = 0
        # Ensure decode_prefix_len is properly synchronized from kv_sender
        if hasattr(req, "disagg_kv_sender") and hasattr(req.disagg_kv_sender, "decode_prefix_len"):
            decode_prefix_len = req.disagg_kv_sender.decode_prefix_len

        logger.debug(f"[send_kv_chunk] req={req.rid}, start_idx={start_idx}, decode_prefix_len={decode_prefix_len}")

        # Adjust start_idx to skip existing prefix from decode side
        effective_start_idx = max(start_idx, decode_prefix_len)

        # if end_idx is specified, use it as the end index of the kv chunk because in overlap schedule,
        # the resolved length is not the same as fill_ids's length
        end_idx = end_idx if end_idx is not None else min(len(req.fill_ids), len(req.origin_input_ids))

        if not last_chunk:
            # if not the last chunk and the last page is partial, delay the last partial page to the next send
            end_idx = end_idx - end_idx % page_size

        # Only send KV from effective_start_idx to end_idx
        if effective_start_idx >= end_idx:
            # Update to final position to ensure progress
            req.start_send_idx = end_idx
            effective_start_idx = end_idx

            # If this is the last chunk, directly mark sender as completed
            if last_chunk:
                logger.info(f"Marking sender as completed for request {req.rid} - all tokens already cached")
                # Directly set sender state to Success to avoid hanging in poll
                req.disagg_kv_sender.conclude_state = KVPoll.Success
                # IMPORTANT: Update manager status
                req.disagg_kv_sender.kv_mgr.update_status(req.bootstrap_room, KVPoll.Success)
                req.disagg_kv_sender.send(np.array([]))

        kv_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx, effective_start_idx:end_idx].cpu().numpy()
        req.start_send_idx = end_idx
        page_indices = kv_to_page_indices(kv_indices, page_size)

        state_indices = None
        if last_chunk:
            # Prepare extra pool indices for hybrid models
            if isinstance(self.token_to_kv_pool, HybridLinearKVPool):
                # Mamba hybrid model: send single mamba state index
                state_indices = [self.req_to_token_pool.req_index_to_mamba_index_mapping[req.req_pool_idx].cpu().numpy()]

        if len(page_indices) == 0:
            logger.info(
                f"Skip sending kv chunk for request {req.rid=} {req.bootstrap_room=} "
                f"because page_indices is empty, {effective_start_idx=} {end_idx=} {start_idx=}"
            )
            return
        if self.server_args.enable_mla_l1_5_cache:
            page_meta = self.token_to_kv_pool.get_page_transfer_metadata(page_indices)
            req.disagg_kv_sender.send(page_indices, 0, mla_l1_5_args=page_meta, state_indices=state_indices)
        else:
            req.disagg_kv_sender.send(page_indices, 0, state_indices=state_indices)
