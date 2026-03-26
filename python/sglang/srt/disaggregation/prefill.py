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

import threading
import time
from collections import deque
from http import HTTPStatus
from typing import TYPE_CHECKING, List, Optional, Tuple, Dict

import numpy as np
import torch # type: ignore

from sglang.srt.disaggregation.base import BaseKVManager, KVArgs, KVPoll
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
    kv_to_page_num,
    poll_and_all_reduce,
    prepare_abort,
)
from sglang.srt.utils import get_colorful_logger
from sglang.srt.managers.req import (
    ABORT_CODE,
    FINISH_LENGTH,
    Req,
    RequestStage,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardMode

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.managers.scheduler import GenerationBatchResult, Scheduler
    from sglang.srt.mem_cache.memory_pool import KVCache
    from sglang.srt.mem_cache.allocator import KVAllocator


logger = get_colorful_logger(__name__)


class PrefillBootstrapQueue:
    """
    Store the requests in bootstrapping
    """

    def __init__(
        self,
        token_to_kv_pool: KVCache,
        kv_allocator: KVAllocator,
        draft_token_to_kv_pool: Optional[KVCache],
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        metadata_buffers: MetadataBuffers,
        tp_rank: int,
        world_size: int,
        bootstrap_port: int,
        gloo_group: ProcessGroup,
        transfer_backend: TransferBackend,
        scheduler: Scheduler,
    ):
        self.token_to_kv_pool = token_to_kv_pool
        self.kv_allocator = kv_allocator
        self.draft_token_to_kv_pool = draft_token_to_kv_pool
        self.is_mla_backend = is_mla_backend(token_to_kv_pool)
        self.draft_is_mla_backend = is_mla_backend(draft_token_to_kv_pool)

        self.metadata_buffers = metadata_buffers
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.tp_rank = tp_rank
        self.tp_size = world_size
        self.transfer_backend = transfer_backend
        self.scheduler = scheduler
        self.kv_manager = self._init_kv_manager()
        self.queue: List[Req] = []
        self.gloo_group = gloo_group
        self.bootstrap_port = bootstrap_port

    def store_prefill_results(self, idx: int, token_id: int):
        assert token_id >= 0, f"token_id: {token_id} is negative"
        output_id_buffer = self.metadata_buffers[0]
        output_id_buffer[idx] = token_id

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
            # We should also transfer draft model kv cache. The indices are
            # always shared with a target model.
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
        kv_args.target_layer_num = target_layer_num
        kv_args.draft_layer_num = draft_layer_num
        kv_args.offsets = offsets

        # Define req -> input ids buffer
        kv_args.aux_data_ptrs, kv_args.aux_data_lens, kv_args.aux_item_lens, kv_args.other_output_offset_idx = (
            self.metadata_buffers.get_buf_infos()
        )
        kv_args.ib_device = self.scheduler.server_args.disaggregation_ib_device
        kv_args.gpu_id = self.scheduler.gpu_id
        kv_manager_class = get_kv_class(self.transfer_backend, KVClassType.MANAGER)
        kv_manager = kv_manager_class(
            kv_args,
            DisaggregationMode.PREFILL,
            self.scheduler.server_args,
            self.is_mla_backend,
            self.draft_is_mla_backend
        )
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

    def extend(self, reqs: List[Req]) -> None:
        for req in reqs:
            self.add(req)

    def _process_req(self, req: Req) -> None:
        """
        Set max_new_tokens = 1, so PrefillAdder memory estimation is accurate
        """
        req.sampling_params.max_new_tokens = 1

    def pop_bootstrapped(self) -> List[Req]:
        """pop the reqs which has finished bootstrapping"""
        bootstrapped_reqs = []
        indices_to_remove = set()
        if len(self.queue) == 0:
            return []
        polls = poll_and_all_reduce(
            [req.disagg_kv_sender for req in self.queue], self.gloo_group
        )

        for i, (req, poll) in enumerate(zip(self.queue, polls)):
            if poll == KVPoll.Bootstrapping:
                continue
            elif poll == KVPoll.Failed or req.to_abort:
                error_message = f"Prefill bootstrap failed for request rank={self.tp_rank} {req.rid=} {req.bootstrap_room=}"
                try:
                    req.disagg_kv_sender.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                logger.error(error_message)
                prepare_abort(
                    req, error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR, err_type = ABORT_CODE.TransferFailed
                )
                self.scheduler.stream_output([req], req.return_logprob)
                indices_to_remove.add(i)
                continue


            # KV.WaitingForInput
            num_kv_indices = len(req.origin_input_ids)
            if self.req_to_metadata_buffer_idx_allocator.available_size() == 0:
                break

            req.metadata_buffer_index = (
                self.req_to_metadata_buffer_idx_allocator.alloc()
            )
            assert req.metadata_buffer_index is not None
            num_pages = kv_to_page_num(num_kv_indices, self.token_to_kv_pool.page_size)

            # Calculate suffix pages for sender initialization
            decode_prefix_len = 0
            if hasattr(req.disagg_kv_sender, 'kv_mgr') and hasattr(req.disagg_kv_sender.kv_mgr, 'receive_decode_prefix_info'):
                decode_prefix_len = req.disagg_kv_sender.kv_mgr.receive_decode_prefix_info(req.bootstrap_room)
            prefix_pages = decode_prefix_len // self.token_to_kv_pool.page_size

            req.disagg_kv_sender.init(num_pages - prefix_pages, req.metadata_buffer_index)
            if self.scheduler.global_rank == 0:
                req.add_latency(RequestStage.PREFILL_BOOTSTRAP)
            bootstrapped_reqs.append(req)
            indices_to_remove.add(i)

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        return bootstrapped_reqs


class SchedulerDisaggregationPrefillMixin:
    """
    Mixin for Scheduler to handle disaggregation prefill
    """

    @torch.no_grad()
    def event_loop_normal_disagg_prefill(self: Scheduler):
        """A normal scheduler loop for prefill worker in disaggregation mode."""

        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            self.waiting_queue.extend(
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
            )
            self.process_prefill_chunk()
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

    @torch.no_grad()
    def event_loop_overlap_disagg_prefill(self: Scheduler):
        self.result_queue = deque()

        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            self.waiting_queue.extend(
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
            )
            self.process_prefill_chunk()
            batch = self.get_new_batch_prefill()

            # Handle DP attention
            if self.server_args.enable_dp_attention:
                batch = self.prepare_dp_attn_batch(batch)

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
                            self.tp_worker.cur_sampling_info
                            if self.draft_worker is None
                            else self.draft_worker.cur_sampling_info
                        ),
                    )
                    self.set_next_batch_sampling_info_done(tmp_batch)

            if self.last_batch:
                tmp_batch, tmp_result = self.result_queue.popleft()
                self.process_batch_result_disagg_prefill(tmp_batch, tmp_result)

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
            output_buffer_indices: List[int] = []
            logits_output_indices: List[int] = []
            output_top_logprobs_indices: List[Tuple[int, int]] = []
            output_token_logprobs_indices: List[Tuple[int, int]] = []
            for idx, req in enumerate(batch.reqs):
                last_chunk = self.chunked_req is None or req.rid != self.chunked_req.rid
                self.add_send_task(req, last_chunk=last_chunk, batch_complete=False)
                output_buffer_indices.append(req.metadata_buffer_index)
                #TODO @xiaobin check logits_output_indices last chunk
                logits_output_indices.append(idx)
                if req.top_logprobs_num > 0:
                    output_token_logprobs_indices.append(
                        (idx, req.metadata_buffer_index)
                    )
                if req.token_ids_logprob is not None:
                    output_top_logprobs_indices.append((idx, req.metadata_buffer_index))

            def set_aux_fn(
                output_ids: torch.Tensor, logits_output: LogitsProcessorOutput
            ):
                self.disagg_metadata_buffers.set_buf_by_batch(
                    output_ids=output_ids,
                    output_buffer_indices=output_buffer_indices,
                    logits_output_indices=logits_output_indices,
                    logits_output=logits_output,
                    output_token_logprobs_indices=output_token_logprobs_indices,
                    output_top_logprobs_indices=output_top_logprobs_indices,
                    cached_tokens=None,  # Don't set here, already set in add_send_task
                )
                self.step_counter.record_aux()

            batch.disagg_set_aux_fn = set_aux_fn

    def process_batch_result_disagg_prefill(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
        launch_done: Optional[threading.Event] = None,
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
                ) = self.draft_worker.resolve_batch_result(bid)
            else:
                logits_output, next_token_ids, _ = (
                    self.tp_worker.resolve_last_batch_result(launch_done)
                )
        else:
            if batch.forward_mode.is_idle():
                return
            next_token_ids = result.next_token_ids.tolist()
            if batch.return_logprob:
                if logits_output.next_token_logprobs is not None:
                    logits_output.next_token_logprobs = (
                        logits_output.next_token_logprobs.tolist()
                    )
                if logits_output.input_token_logprobs is not None:
                    logits_output.input_token_logprobs = tuple(
                        logits_output.input_token_logprobs.tolist()
                    )

        for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
            req: Req
            req.ongoing_batch_num = 0
            if req.first_chunk_forward_start_time is None:
                req.first_chunk_forward_start_time = req.last_tic

            if req.is_chunked <= 0:
                # There is no output_ids for prefill
                req.output_ids.append(next_token_id)
                self.tree_cache.cache_unfinished_req(req)  # update the tree and lock
                if self.global_rank == 0:
                    now = time.monotonic()
                    total_forward_time = now - req.first_chunk_forward_start_time
                    if getattr(self, "metrics_collector", None):
                        self.metrics_collector.observe_request_latency_seconds(
                            RequestStage.PREFILL_FORWARD.value, total_forward_time
                        )
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
                    req.grammar.accept_token(next_token_id)
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

                if self.enable_overlap:
                    self.add_send_task(req, last_chunk=False, end_idx=req.tmp_end_idx)

        # We need to remove the sync in the following function for overlap schedule.
        self.set_next_batch_sampling_info_done(batch)

    def process_disagg_prefill_inflight_queue(self: Scheduler) -> None:
        """
        Poll the requests in the middle of transfer. If done, return the request.
        """
        assert len(self.disagg_prefill_inflight_queue) > 0

        done_reqs = []

        polls = poll_and_all_reduce(
            [req.disagg_kv_sender for req in self.disagg_prefill_inflight_queue],
            self.attn_tp_cpu_group,
        )

        undone_reqs: List[Req] = []
        # Check .poll() for the reqs in disagg_prefill_inflight_queue. If Success, respond to the client and remove it from the queue
        for req, poll in zip(self.disagg_prefill_inflight_queue, polls):
            if poll in [KVPoll.WaitingForInput, KVPoll.Transferring]:
                undone_reqs.append(req)
            elif poll == KVPoll.Success:  # transfer done
                self.tree_cache.cache_finished_req(req)  # unlock the tree
                self.kv_allocator.free_group_end()
                req.finished_reason = FINISH_LENGTH(length=0)
                # FIXME: clean up req's data in transfer engine
                if hasattr(req.disagg_kv_sender, "clear"):
                    req.disagg_kv_sender.clear()
                done_reqs.append(req)
            elif poll == KVPoll.Failed or req.to_abort:
                error_message = f"Prefill transfer failed for request rank={self.tp_rank} {req.rid=} {req.bootstrap_room=}"
                try:
                    req.disagg_kv_sender.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                logger.warning(error_message)
                self.tree_cache.cache_finished_req(req)  # unlock the tree
                self.kv_allocator.free_group_end()
                prepare_abort(
                    req, error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR, err_type = ABORT_CODE.TransferFailed
                )
                done_reqs.append(req)

        for req in done_reqs:
            req: Req
            if self.global_rank == 0:
                req.add_latency(RequestStage.PREFILL_TRANSFER_KV_CACHE)
            self.disagg_prefill_bootstrap_queue.req_to_metadata_buffer_idx_allocator.free(
                req.metadata_buffer_index
            )
            req.metadata_buffer_index = -1

        # Stream requests which have finished transfer
        self.stream_output(
            done_reqs,
            any(req.return_logprob for req in done_reqs),
            None,
        )

        # Publish KV events after processing batch results
        self._publish_kv_events()

        self.disagg_prefill_inflight_queue = undone_reqs

    def process_prefill_chunk(self: Scheduler) -> None:
        if self.last_batch and self.last_batch.forward_mode.is_extend():
            if self.chunked_req:
                # Move the chunked request out of the batch so that we can merge
                # only finished requests to running_batch.
                self.last_batch.filter_batch(chunked_req_to_exclude=self.chunked_req)
                self.tree_cache.cache_unfinished_req(self.chunked_req)
                req = self.chunked_req
                # Set prefix_len and for init_next_round_input_chunk
                req.prefix_len = self.req_to_token_pool.alloced_lens[req.req_pool_idx].item()
                page_size = self.kv_allocator.page_size
                # Set prefix page ids
                req.prefix_page_ids = self.kv_allocator.req_to_page[
                    req.req_pool_idx, 0 : (req.prefix_len + page_size - 1) // page_size
                ]
                if self.enable_overlap:
                    # Delay KV transfer to process_batch_result_disagg_prefill when overlap is enabled to ensure results are resolved
                    self.chunked_req.tmp_end_idx = min(
                        len(self.chunked_req.fill_ids),
                        len(self.chunked_req.origin_input_ids),
                    )
                else:
                    self.add_send_task(self.chunked_req)
                self.batch_is_full = False

    def add_send_task(
        self: Scheduler,
        req: Req,
        last_chunk: bool = False,
        end_idx: Optional[int] = None,
        batch_complete: bool = True,
    ) -> None:
        skip_send = self.enable_layerwise_transfer and batch_complete
        
        # Set cached_tokens in metadata buffer BEFORE transfer starts
        # This is critical to ensure decode side receives the correct prefill cached_tokens value
        if req.metadata_buffer_index >= 0 and last_chunk:
            self.disagg_metadata_buffers.cached_tokens[req.metadata_buffer_index][0] = req.prefix_len
        if not self.enable_layerwise_transfer and last_chunk:
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
        end_idx: Optional[int] = None,
    ) -> None:
        """
        Send a prefilled chunk to the decode server, skipping existing prefixes from decode side
        """
        # Add diagnostic log: record the start of KV block sending
        send_start_time = time.time()
        logger.debug(f"[send_kv_chunk] START: Sending KV chunk for req={req.rid}, last_chunk={last_chunk}, end_idx={end_idx}")

        page_size = self.token_to_kv_pool.page_size
        start_idx = req.start_send_idx

        decode_prefix_len = 0
        # Ensure decode_prefix_len is properly synchronized from kv_sender
        if hasattr(req, 'disagg_kv_sender') and hasattr(req.disagg_kv_sender, 'decode_prefix_len'):
            decode_prefix_len = req.disagg_kv_sender.decode_prefix_len

        logger.debug(f"[send_kv_chunk] req={req.rid}, start_idx={start_idx}, decode_prefix_len={decode_prefix_len}")

        # Adjust start_idx to skip existing prefix from decode side
        effective_start_idx = max(start_idx, decode_prefix_len)

        # if end_idx is specified, use it as the end index of the kv chunk because in overlap schedule,
        # the resolved length is not the same as fill_ids's length
        end_idx = (
            end_idx
            if end_idx is not None
            else min(len(req.fill_ids), len(req.origin_input_ids))
        )

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

        kv_indices = (
            self.req_to_token_pool.req_to_token[req.req_pool_idx, effective_start_idx:end_idx]
            .cpu()
            .numpy()
        )
        req.start_send_idx = end_idx
        page_indices = kv_to_page_indices(kv_indices, page_size)
        if len(page_indices) == 0:
            logger.info(
                f"Skip sending kv chunk for request {req.rid=} {req.bootstrap_room=} because page_indices is empty, {effective_start_idx=} {end_idx=} {start_idx=}"
            )
            return
        req.disagg_kv_sender.send(page_indices)