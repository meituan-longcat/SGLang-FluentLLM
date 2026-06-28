from __future__ import annotations

import logging
import math
import queue
import time
from collections import deque
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.distributed
from tqdm import tqdm

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.utils import poll_and_all_reduce, prepare_abort, kv_to_page_num
from sglang.srt.distributed import get_attn_tp_group
from sglang.srt.distributed.parallel_state import P2PWork
from sglang.srt.managers.req import Req, ABORT_CODE, FINISH_LENGTH, RequestStage
from sglang.srt.managers.schedule_batch import ScheduleBatch, GenerationBatchResult
from sglang.srt.managers.utils import get_logprob_from_pp_outputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils import DynamicGradMode, broadcast_pyobj, point_to_point_pyobj, get_device_module, \
    get_colorful_logger

logger = get_colorful_logger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers import Scheduler

class PPOpId:
    RecvReq = 0
    BootstrapReq = 1
    TransferredReq = 2
    RunBatch = 3
    RunOver = 4

class SendWorkQueue:
    def __init__(self):
        self.uncommit_batch_op_size=0
        self.queue=deque()

    def append(self, op_id: PPOpId, work):
        self.queue.append([op_id, work])
        if op_id == PPOpId.RunOver:
            self.uncommit_batch_op_size = self.uncommit_batch_op_size + 1

    def commit(self, is_once=False):
        works=[]
        while self.queue:
            work_item=self.queue[0]
            works.append(self.queue.popleft()[1])
            if work_item[0]==PPOpId.RunOver:
                self.uncommit_batch_op_size = self.uncommit_batch_op_size - 1
                if is_once:
                    break
        for work in works:
            self.pp_commit_comm_work(work)

    def pp_commit_comm_work(self, work: List[P2PWork]) -> None:
        for p2p_work in work:
            p2p_work.work.wait()
        work.clear()

class SchedulerPPMixin:

    def pp_process_batch_result_disagg_prefill(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ) -> None:
        if not self.pp_group.is_last_rank:
            for i, req in enumerate(batch.reqs):
                req: Req
                req.ongoing_batch_num = 0
                if req.is_chunked <= 0:
                    req.output_ids.append(-1) # just for cache kv cache
                    self.tree_cache.cache_unfinished_req(req)  # update the tree and lock
                    self.disagg_prefill_inflight_queue.append(req)
                    self.add_send_task(req, last_chunk=True)
                    if req.grammar is not None:
                        req.grammar.finished=req.finished()
                else:
                    req.is_chunked -= 1
            # We need to remove the sync in the following function for overlap schedule.
            self.set_next_batch_sampling_info_done(batch)
        else:
            self.process_batch_result_disagg_prefill(batch, result)

    def pp_check_inflight(self: Scheduler):
        done_reqs_ids=[]
        polls=[]
        if len(self.disagg_prefill_inflight_queue)!=0:
            polls=poll_and_all_reduce(
                [req.disagg_kv_sender for req in self.disagg_prefill_inflight_queue],
                self.attn_tp_cpu_group,
            )
            for req, poll in zip(self.disagg_prefill_inflight_queue, polls):
                if poll==KVPoll.Success or poll==KVPoll.Failed or req.to_abort:
                    done_reqs_ids.append(req.rid)
        return done_reqs_ids, polls

    def pp_process_disagg_prefill_inflight_queue(self: Scheduler, done_reqs_ids, polls):
        if not self.pp_group.is_first_rank:
            # blocking to wait bootstrapped_reqs_ids+failed_reqs_ids okay
            pop_reqs_ids=set()
            while True:
                polls=poll_and_all_reduce(
                    [req.disagg_kv_sender for req in self.disagg_prefill_inflight_queue],
                    self.attn_tp_cpu_group,
                )
                if len(pop_reqs_ids)==len(done_reqs_ids):
                    break
                for i, (req, poll) in enumerate(zip(self.disagg_prefill_inflight_queue, polls)):
                    if poll in [KVPoll.WaitingForInput, KVPoll.Transferring]:
                        continue
                    if req.rid in done_reqs_ids:
                        pop_reqs_ids.add(req.rid)

        undone_reqs=[]
        success_reqs=[]
        failed_reqs=[]
        for req, poll in zip(self.disagg_prefill_inflight_queue, polls):
            if req.rid not in done_reqs_ids:
                undone_reqs.append(req)
            else:
                if poll==KVPoll.Success:
                    success_reqs.append(req)
                else:
                    failed_reqs.append(req)

        for req in success_reqs:
            self.tree_cache.cache_finished_req(req)  # unlock the tree
            self.kv_allocator.free_group_end()
            req.finished_reason=FINISH_LENGTH(length=0)
            # FIXME: clean up req's data in transfer engine
            if hasattr(req.disagg_kv_sender, "clear"):
                req.disagg_kv_sender.clear()

        for req in failed_reqs:
            error_message=f"Prefill transfer failed for request rank={self.tp_rank} {req.rid=} {req.bootstrap_room=} {self.pp_rank=}"
            try:
                req.disagg_kv_sender.failure_exception()
            except Exception as e:
                error_message+=f" with exception {e}"
            logger.warning(error_message)
            self.tree_cache.cache_finished_req(req)  # unlock the tree
            self.kv_allocator.free_group_end()
            prepare_abort(
                req, error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR, err_type=ABORT_CODE.TransferFailed
            )

        done_reqs=success_reqs+failed_reqs
        for req in done_reqs:
            # todo
            # if self.global_rank == 0:
            #     req.add_latency(RequestStage.PREFILL_TRANSFER_KV_CACHE)
            self.disagg_prefill_bootstrap_queue.req_to_metadata_buffer_idx_allocator.free(
                req.metadata_buffer_index
            )
            req.metadata_buffer_index = -1

        if self.pp_group.is_last_rank:
            # Stream requests which have finished transfer
            self.stream_output(
                done_reqs,
                any(req.return_logprob for req in done_reqs),
                None,
            )

        # todo
        # self._publish_kv_events()

        self.disagg_prefill_inflight_queue = undone_reqs

        if get_attn_tp_group().rank_in_group==0:
            logger.info(f"{done_reqs=} {undone_reqs=}")

    @DynamicGradMode()
    def event_loop_pp_disagg_prefill(self: Scheduler):
        # todo 记录时间指标

        self.send_work_queue=SendWorkQueue()
        def send_pp_op_to_next_stage(op):
            if not self.pp_group.is_last_rank:
                send_op_work=self._pp_send_pyobj_to_next_stage(
                    [op], async_send=True
                )
                self.send_work_queue.append(-1, send_op_work)

        def handle_recv_reqs_op(recv_reqs):
            recv_reqs_ids=[]
            for req in recv_reqs:
                recv_reqs_ids.append(getattr(req, "rid", None))
            if get_attn_tp_group().rank_in_group==0:
                logger.info(f"handle_recv_reqs_op: {recv_reqs_ids=}")
            if not self.pp_group.is_last_rank:
                send_req_work=self._pp_send_pyobj_to_next_stage(
                    recv_reqs, async_send=True
                )
                self.send_work_queue.append(PPOpId.RecvReq, send_req_work)
            self.process_input_requests(recv_reqs)

        def handle_bootstrapped_op(bootstrap_reqs_ids, failed_reqs_ids, polls):
            if get_attn_tp_group().rank_in_group==0:
                logger.info(f"handle_bootstrapped_op: {bootstrap_reqs_ids} {failed_reqs_ids}")
            if not self.pp_group.is_last_rank:
                send_bootstrapped_works=self._pp_send_pyobj_to_next_stage(
                    [bootstrap_reqs_ids, failed_reqs_ids], async_send=True
                )
                self.send_work_queue.append(PPOpId.BootstrapReq, send_bootstrapped_works)
            bootstrapped_reqs=self.disagg_prefill_bootstrap_queue.pp_pop_bootstrapped(bootstrap_reqs_ids, failed_reqs_ids, polls)
            self.waiting_queue.extend(bootstrapped_reqs)

        def handle_run_batch_op(step_func):
            self.process_prefill_chunk()
            batch=self.get_new_batch_prefill()

            if self.server_args.enable_dp_attention:
                batch=self.prepare_dp_attn_batch(batch)
            self.update_oe_info(batch)
            if batch:
                self.forward_ct+=1
            self.cur_batch=batch

            if batch:
                if get_attn_tp_group().rank_in_group==0:
                    req_ids=[req.rid for req in batch.reqs]
                    if len(req_ids)>0:
                        logger.info(f"{self.it=} {req_ids=}")
                if not self.pp_group.is_first_rank:
                    with torch.profiler.record_function(f"{self.it}_recv_pp_proxy_tensors"):
                        # todo should be async in device
                        pp_proxy_tensors=PPProxyTensors(
                            self.pp_group.recv_tensor_dict()
                        )
                        batch.pp_proxy_tensors=pp_proxy_tensors

                with torch.profiler.record_function(f"{self.it}_launch_batch"):
                    if self.enable_layerwise_transfer and not batch.forward_mode.is_idle():
                        self.launch_send_async(batch)
                    result=self.run_batch(batch)

                with torch.profiler.record_function(f"{self.it}_wait batch done"):
                    launch_event=torch.get_device_module(self.device).Event()
                    launch_event.record()
                    get_device_module().current_stream().wait_event(launch_event)
                    batch.pp_proxy_tensors=None

                if not self.pp_group.is_last_rank:
                    with torch.profiler.record_function(f"{self.it}_send_pp_proxy_tensors"):
                        send_proxy_work=self._pp_send_dict_to_next_stage(
                            result.logits_output.tensors,
                            async_send=True,
                        )
                        self.send_work_queue.append(PPOpId.RunBatch, send_proxy_work)

                with torch.profiler.record_function(f"{self.it}_process_batch_result"):
                    self.pp_process_batch_result_disagg_prefill(batch, result)

                if step_func and get_attn_tp_group().rank_in_group==0:
                    step_func()  # p.step()

                self.it=self.it+1

            self.send_work_queue.append(PPOpId.RunOver, [])
            self.last_batch=batch

            if self.send_work_queue.uncommit_batch_op_size >= 3:
                self.send_work_queue.commit(is_once=True)

            # if batch is None and len(self.disagg_prefill_inflight_queue)==0:
            #     self.new_token_ratio = self.init_new_token_ratio

            self.batch_is_full = False

        def handle_transferred_op(done_reqs_ids, polls):
            if get_attn_tp_group().rank_in_group==0:
                logger.info(f"handle_transferred_op: {done_reqs_ids}")
            if not self.pp_group.is_last_rank:
                send_transferred_work=self._pp_send_pyobj_to_next_stage(
                    done_reqs_ids, async_send=True
                )
                self.send_work_queue.append(PPOpId.TransferredReq, send_transferred_work)
            self.pp_process_disagg_prefill_inflight_queue(done_reqs_ids, polls)

        def _loop_process(step_func=None):
            self.it=0
            idle_timeout_seconds=20  # 20s force send run batch op
            last_batch_time=time.time()
            while True:
                if self.pp_group.is_first_rank:
                    # 1. recv reqs
                    recv_reqs=self.recv_requests()
                    if recv_reqs:
                        send_pp_op_to_next_stage(PPOpId.RecvReq)
                        handle_recv_reqs_op(recv_reqs)

                    # 2. bootstrap
                    if self.grammar_queue:
                        self.move_ready_grammar_requests()
                    bootstrap_reqs_ids, failed_reqs_ids, polls =self.disagg_prefill_bootstrap_queue.pp_check_bootstrapped()
                    if len(bootstrap_reqs_ids) > 0 or len(failed_reqs_ids) > 0:
                        send_pp_op_to_next_stage(PPOpId.BootstrapReq)
                        handle_bootstrapped_op(bootstrap_reqs_ids, failed_reqs_ids, polls)

                    # 3. run batch
                    current_time=time.time()
                    force_send=False
                    if current_time-last_batch_time>=idle_timeout_seconds:
                        force_send=True
                        last_batch_time=current_time

                    if self.chunked_req or len(self.waiting_queue)>0 or force_send:
                        send_pp_op_to_next_stage(PPOpId.RunBatch)
                        handle_run_batch_op(step_func)

                    # 4. transferred
                    done_reqs_ids, polls=self.pp_check_inflight()
                    if len(done_reqs_ids) > 0:
                        send_pp_op_to_next_stage(PPOpId.TransferredReq)
                        handle_transferred_op(done_reqs_ids, polls)
                else:
                    pp_op_id = self._pp_recv_pyobj_from_prev_stage()[0]
                    send_pp_op_to_next_stage(pp_op_id)
                    if pp_op_id == PPOpId.RecvReq:
                        recv_reqs=self._pp_recv_pyobj_from_prev_stage()
                        handle_recv_reqs_op(recv_reqs)
                    elif pp_op_id == PPOpId.BootstrapReq:
                        ids = self._pp_recv_pyobj_from_prev_stage()
                        bootstrap_reqs_ids, failed_reqs_ids = ids[0], ids[1]
                        handle_bootstrapped_op(bootstrap_reqs_ids, failed_reqs_ids, None)
                    elif pp_op_id == PPOpId.RunBatch:
                        handle_run_batch_op(step_func)
                    elif pp_op_id == PPOpId.TransferredReq:
                        done_reqs_ids=self._pp_recv_pyobj_from_prev_stage()
                        handle_transferred_op(done_reqs_ids, None)
                    else:
                        assert False, f"{pp_op_id=} is not a valid pp_op_id"

                if self.cur_batch is None and len(self.disagg_prefill_inflight_queue)==0 and len(
                    self.disagg_prefill_bootstrap_queue.queue)==0 and len(self.grammar_queue)==0:
                    self.check_memory()

        if self.npu_custom_profiler:
            with self.npu_custom_profiler as p:
                p.start()
                _loop_process(p.step)  # do npu process
                p.stop()
        else:
            _loop_process()

    def profile_and_init_predictor(self: Scheduler):
        """
        Profile prefill latency for dynamic chunk sizing.

        Only runs on PP0 (first rank), then broadcasts data to all ranks.
        All ranks fit coefficients using the same data.
        """
        seq_lens: List[int] = []
        latencies: List[float] = []

        if self.pp_group.is_first_rank:
            model_runner = self.tp_worker.model_runner
            model_config = model_runner.model_config
            input_ids_list = []
            for i in range(128):
                chunk_size = int(
                    self.chunked_prefill_size * 1.25
                    - i * (self.chunked_prefill_size * 1.25 // 128)
                )
                if chunk_size <= 0:
                    break
                input_ids = np.random.randint(
                    0, 10000, size=chunk_size, dtype=np.int64
                ).tolist()
                input_ids_list.append(input_ids)

            sampling_params = SamplingParams(
                temperature=0,
                max_new_tokens=1,
            )
            # Create and profile requests
            for i, input_ids in enumerate(
                tqdm(
                    input_ids_list,
                    desc="Profiling prefill latency for dynamic chunking",
                )
            ):
                req = Req(
                    rid=str(i),
                    origin_input_text="",
                    origin_input_ids=input_ids,
                    sampling_params=sampling_params,
                )
                req.fill_ids = req.origin_input_ids
                req.logprob_start_len = -1
                req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))

                # Prepare batch
                batch = ScheduleBatch.init_new(
                    [req],
                    self.req_to_token_pool,
                    self.token_to_kv_pool_allocator,
                    self.tree_cache,
                    self.model_config,
                    False,
                    self.spec_algorithm,
                )

                current_seq_len = len(req.fill_ids)

                if False:
                # if is_dp_attention_enabled():
                    # For profiling, we only have one request on PP0
                    # Set global_num_tokens to indicate this rank has tokens, others have 0
                    dp_size = get_attention_dp_size()
                    global_num_tokens = [0] * dp_size
                    dp_rank = get_attention_dp_rank()
                    global_num_tokens[dp_rank] = current_seq_len
                    batch.global_num_tokens = global_num_tokens
                    batch.global_num_tokens_for_logprob = global_num_tokens

                proxy_tensors = {
                    "hidden_states": torch.zeros(
                        (current_seq_len, model_config.hidden_size),
                        dtype=model_config.dtype,
                        device="cuda",
                    ),
                    "residual": torch.zeros(
                        (current_seq_len, model_config.hidden_size),
                        dtype=model_config.dtype,
                        device="cuda",
                    ),
                }

                pp_proxy = PPProxyTensors(proxy_tensors)

                # Measure latency with CUDA synchronization for accurate timing
                # Synchronize before starting timing to ensure clean measurement
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                start = time.perf_counter()
                batch.prepare_for_extend()
                model_worker_batch = batch.get_model_worker_batch()

                forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
                _ = model_runner.forward(
                    forward_batch=forward_batch, pp_proxy_tensors=pp_proxy
                )

                # Synchronize after forward to ensure GPU operations complete
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                latency_seconds = time.perf_counter() - start
                latency_ms = latency_seconds * 1e3  # Convert to milliseconds
                seq_lens.append(len(input_ids))
                latencies.append(latency_ms)

                # Release KV cache
                if req.req_pool_idx is not None:
                    kv_indices = self.req_to_token_pool.req_to_token[
                        req.req_pool_idx, : len(req.fill_ids)
                    ]
                    self.token_to_kv_pool_allocator.free(kv_indices)
                    self.req_to_token_pool.free(req)

            logger.info(
                f"[PP Dynamic Chunk] [PP0] Profiled {len(seq_lens)} samples: "
                f"seq_lens={seq_lens}, latencies_ms={latencies}"
            )

            if self.attn_tp_size > 1:
                data_to_sync_tp = [seq_lens, latencies]
                data_to_sync_tp = broadcast_pyobj(
                    data_to_sync_tp,
                    self.attn_tp_group.rank,
                    self.attn_tp_cpu_group,
                    src=self.attn_tp_group.ranks[0],
                )
                seq_lens, latencies = data_to_sync_tp

        # Broadcast data to all ranks
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            data_to_sync = [seq_lens, latencies]
            self.pp_group.broadcast_object_list(data_to_sync, src=0)
            seq_lens, latencies = data_to_sync

        # Quadratic model: f(l) = al^2 + bl + c
        self.length_predictor = ChunkSizePredictor()
        self.length_predictor.fit(seq_lens, latencies)
        self.length_predictor.set_target_latency(self.chunked_prefill_size)
        self.length_predictor.is_ready = True
        logger.info(
            f"[PP Dynamic Chunk] [PP{self.pp_rank}] Predictor ready (quadratic). "
            f"Target latency: {self.length_predictor.target_latency:.2f}ms"
        )

    def predict_next_chunk_size(self: Scheduler, history_len: int) -> Optional[int]:
        """
        Predict next chunk size dynamically based on current history length.

        Args:
            history_len: Current sequence length

        Returns:
            Predicted chunk size, or None to use default chunked_prefill_size
        """
        if (
            not self.enable_dynamic_chunking
            or self.length_predictor is None
            or not self.length_predictor.is_ready
        ):
            return None

        max_chunk_size = getattr(self, "max_prefill_tokens", None)
        predicted_size = self.length_predictor.predict_next_chunk_size(
            history_len=history_len,
            base_chunk_size=self.chunked_prefill_size,
            page_size=self.page_size,
            context_len=self.model_config.context_len,
            max_chunk_size=max_chunk_size,
        )

        if predicted_size is not None:
            logger.debug(
                f"[PP Dynamic Chunk] [PP{self.pp_rank}] Predicted chunk size: "
                f"{predicted_size} (history_len={history_len})"
            )

        return predicted_size

    def _pp_send_pyobj_to_next_stage(self: Scheduler, data, async_send: bool = False):
        p2p_work = []
        if self.attn_tp_rank == 0:
            dp_offset = self.attn_dp_rank * self.attn_tp_size
            p2p_work = point_to_point_pyobj(
                data,
                self.pp_rank * self.tp_size + dp_offset,
                self.world_group.cpu_group,
                self.pp_rank * self.tp_size + dp_offset,
                ((self.pp_rank + 1) % self.pp_size) * self.tp_size + dp_offset,
                async_send=async_send,
            )
        return p2p_work

    def _pp_recv_pyobj_from_prev_stage(self: Scheduler):
        if self.attn_tp_rank == 0:
            dp_offset = self.attn_dp_rank * self.attn_tp_size
            data = point_to_point_pyobj(
                [],
                self.pp_rank * self.tp_size + dp_offset,
                self.world_group.cpu_group,
                ((self.pp_rank - 1) % self.pp_size) * self.tp_size + dp_offset,
                self.pp_rank * self.tp_size + dp_offset,
            )
        else:
            data = None

        if self.attn_tp_size > 1:
            data = broadcast_pyobj(
                data,
                self.attn_tp_group.rank,
                self.attn_tp_cpu_group,
                src=self.attn_tp_group.ranks[0],
            )

        return data

    def _pp_prepare_tensor_dict(
        self: Scheduler, result: GenerationBatchResult, batch: ScheduleBatch
    ) -> Dict[str, torch.Tensor]:
        tensor_dict = {
            "next_token_ids": result.next_token_ids,
            # "accept_length": result.accept_length,
            # "new_verified_id": result.new_verified_id,
            # "token_list": result.token_list
        }

        if batch.return_logprob:
            logprob_dict = result.get_logprob_dict_from_result()
            tensor_dict = {
                **tensor_dict,
                **logprob_dict,
            }
        return tensor_dict

    def _pp_send_dict_to_next_stage(
        self: Scheduler,
        tensor_dict: Dict[str, torch.Tensor],
        async_send: bool = True,
    ):
        p2p_work = []
        if not self.pp_group.is_last_rank:
            p2p_work.extend(
                self.pp_group.send_tensor_dict(
                    tensor_dict=tensor_dict,
                    all_gather_group=None,
                    async_send=async_send,
                )
            )
        else:
            p2p_work.extend(
                self.pp_reverse_group.send_tensor_dict(
                    tensor_dict=tensor_dict,
                    all_gather_group=None,
                    async_send=async_send,
                )
            )
        return p2p_work

    def _pp_recv_dict_from_prev_stage(
        self: Scheduler,
    ) -> Dict[str, torch.Tensor]:
        if not self.pp_group.is_first_rank:
            res = self.pp_group.recv_tensor_dict(
                all_gather_group=None,
            )
        else:
            res=self.pp_reverse_group.recv_tensor_dict(
                all_gather_group=None,
            )
        return res

class ChunkSizePredictor:
    """
    Predictor for dynamic chunk size based on quadratic latency model.

    Models latency as: f(l) = a*l^2 + b*l + c
    Predicts next chunk size x such that: f(L+x) - f(L) = target_latency
    """

    def __init__(self):
        self.quadratic_coeff_a = 0.0
        self.linear_coeff_b = 0.0
        self.constant_coeff_c = 0.0
        self.target_latency: Optional[float] = None
        self.is_ready = False

    def fit(self, seq_lens: List[int], latencies: List[float]):
        """Fit quadratic coefficients f(l) = al^2 + bl + c from data points."""
        # Skip the first data point to reduce fitting bias, as the first run is slower without warmup
        L = np.array(seq_lens[1:], dtype=np.float64)
        T = np.array(latencies[1:], dtype=np.float64)

        if len(L) < 8:
            raise ValueError(
                f"Not enough data points for quadratic fitting ({len(L)} < 8). "
                "Need at least 8 samples with different sequence lengths."
            )

        # Build design matrix for f(l) = al^2 + bl + c
        X = np.column_stack([L * L, L, np.ones_like(L)])  # [l^2, l, 1]

        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(X, T, rcond=None)
            if len(coeffs) >= 3:
                fitted_a = float(coeffs[0])  # quadratic coefficient
                fitted_b = float(coeffs[1])  # linear coefficient
                fitted_c = float(coeffs[2])  # constant coefficient
            else:
                raise ValueError("Failed to fit coefficients: insufficient rank")
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Failed to fit f(l) = al^2 + bl + c: {e}")

        # Validate coefficients
        if fitted_a <= 0:
            raise ValueError(
                f"Fitted quadratic coefficient a={fitted_a:.2e} is not positive. "
                "Attention has O(n^2) complexity, so a must be positive. "
                "Check warmup data quality."
            )

        if fitted_b < 0:
            logger.warning(
                f"Fitted linear coefficient b={fitted_b:.2e} is negative. Setting b=0."
            )
            fitted_b = 0.0

        self.quadratic_coeff_a = fitted_a
        self.linear_coeff_b = fitted_b
        self.constant_coeff_c = fitted_c

        logger.info(
            f"[ChunkSizePredictor] Fitted coefficients: a={fitted_a:.2e}, "
            f"b={fitted_b:.2e}, c={fitted_c:.2e}"
        )

    def set_target_latency(self, base_chunk_size: int):
        """Set target latency based on base chunk size: target = f(base_chunk_size) - f(0)."""

        def f(l: float) -> float:
            """Total latency function: f(l) = al^2 + bl + c (or bl + c for linear)"""
            return (
                self.quadratic_coeff_a * l * l
                + self.linear_coeff_b * l
                + self.constant_coeff_c
            )

        self.target_latency = f(float(base_chunk_size)) - f(0.0)

        if self.target_latency <= 0:
            raise ValueError(
                f"Calculated target_latency={self.target_latency:.2f}ms is not positive. "
                "Check warmup data quality."
            )

        logger.info(
            f"[ChunkSizePredictor] Target latency: {self.target_latency:.2f}ms "
            f"(base_chunk_size={base_chunk_size})"
        )

    def predict_next_chunk_size(
        self,
        history_len: int,
        base_chunk_size: int,
        page_size: int,
        context_len: int,
        max_chunk_size: Optional[int] = None,
    ) -> Optional[int]:
        """
        Predict next chunk size x such that f(history_len + x) - f(history_len) = target_latency.

        Args:
            history_len: Current sequence length (L)
            base_chunk_size: Base chunk size
            page_size: Page size for alignment
            context_len: Maximum context length
            max_chunk_size: Maximum allowed chunk size (optional)

        Returns:
            Predicted chunk size, or None if prediction fails
        """
        if not self.is_ready or self.target_latency is None:
            return None

        # Handle quadratic model: f(l) = al^2 + bl + c
        if self.quadratic_coeff_a <= 0:
            return None

        # Solve f(L+x) - f(L) = T
        # where f(L) = a*L^2 + b*L + c
        # This expands to: ax^2 + (2aL+b)x - T = 0
        # A = a, B = 2aL + b, C = -T
        A = self.quadratic_coeff_a
        B = 2 * self.quadratic_coeff_a * history_len + self.linear_coeff_b
        C = -self.target_latency

        discriminant = B * B - 4 * A * C

        if discriminant < 0:
            logger.warning(
                f"Discriminant is negative ({discriminant:.2e}). "
                f"No real solution for chunk size. L={history_len}, T={self.target_latency:.2f}ms."
            )
            return None

        sqrt_discriminant = math.sqrt(discriminant)
        calculated_chunk_size_float = (-B + sqrt_discriminant) / (2 * A)

        if calculated_chunk_size_float <= 0:
            logger.warning(
                f"Calculated chunk size is non-positive ({calculated_chunk_size_float:.2f}). "
                f"L={history_len}, T={self.target_latency:.2f}ms."
            )
            return None

        # Use a smooth coefficient to reduce the abrupt decrease in chunk size
        # smooth_coeff = envs.SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR.get()
        smooth_coeff = 0.9
        smoothed_chunk_size = base_chunk_size + smooth_coeff * (
            calculated_chunk_size_float - base_chunk_size
        )
        # Make sure the dynamic chunk size is at least 1/4 of the base chunk size
        calculated_chunk_size = max(int(smoothed_chunk_size), base_chunk_size // 4)

        # Align to page_size (minimum alignment size is 64)
        alignment_size = max(page_size, 64)
        dynamic_chunk_size = (calculated_chunk_size // alignment_size) * alignment_size

        # Ensure aligned size is at least alignment_size
        if dynamic_chunk_size < alignment_size:
            dynamic_chunk_size = alignment_size

        # Apply constraints
        max_allowed = context_len - history_len - 100  # Leave 100 tokens margin
        if max_chunk_size is not None:
            max_allowed = min(max_allowed, max_chunk_size)
        dynamic_chunk_size = min(dynamic_chunk_size, max_allowed)

        # Align again after min operation
        dynamic_chunk_size = (dynamic_chunk_size // alignment_size) * alignment_size

        if dynamic_chunk_size < alignment_size:
            return None

        return dynamic_chunk_size
