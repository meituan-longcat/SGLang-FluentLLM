from __future__ import annotations

import gc
import signal
import threading
import time
from queue import Queue
from typing import TYPE_CHECKING

import psutil
import torch
import torch.distributed

from sglang.srt.distributed import get_attn_tp_group
from sglang.srt.distributed.parallel_state import P2PWork
from sglang.srt.managers.req import Req
from sglang.srt.managers.scheduling.schedule_batch import GenerationBatchResult, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.utils import (
    DynamicGradMode,
    configure_gc_warning,
    freeze_gc,
    get_colorful_logger,
    get_device_module,
    get_free_port,
    get_local_ip_by_remote,
    is_npu,
)
from sglang.srt.utils.pp_tracker import PPTracker
from sglang.utils import get_exception_traceback

logger = get_colorful_logger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers import Scheduler


class ForwardThread:
    def __init__(self, scheduler: Scheduler, device):
        self.scheduler = scheduler
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.forward_stream = get_device_module().Stream()
        self.parent_process = psutil.Process().parent()
        self.forward_thread = threading.Thread(
            target=self.forward_thread_func,
        )
        self.forward_thread.start()
        self.scheduler_stream = get_device_module().current_stream()
        if is_npu():
            torch.npu.set_stream_limit(self.scheduler_stream, 8, 16)
        self.device = device
        self.done_batch_ct = 0
        self.undone_batch_num = 0
        # Number of batches handed to the forward thread whose results have
        # not yet been consumed by the scheduler. Unlike undone_batch_num,
        # this count does not decrease when a result enters output_queue.
        self.unconsumed_batch_num = 0
        self.lock = threading.Lock()
        self.total_idle_time = 0.0
        self.total_active_time = 0.0
        self.pp_rank = self.scheduler.pp_rank
        self.pp_size = self.scheduler.pp_size
        self.pp_group = self.scheduler.pp_group
        self.send_proxy_works = []

    def run_batch_async(self, batch: ScheduleBatch):
        """Run a batch."""
        self.scheduler.profile_batch_predicate(batch)
        self.scheduler.profile_mfu(batch)
        if self.scheduler.enable_layerwise_transfer and not batch.forward_mode.is_idle():
            self.scheduler.launch_send_async(batch)
        self.scheduler_stream.synchronize()
        if batch.return_logprob:
            batch.extend_input_len_per_req = [req.extend_input_len for req in batch.reqs]
            batch.extend_logprob_start_len_per_req = [req.extend_logprob_start_len for req in batch.reqs]
        with self.lock:
            self.undone_batch_num += 1
            self.unconsumed_batch_num += 1
        self.input_queue.put(batch)

    def get_done_batch_ct(self):
        with self.lock:
            return self.done_batch_ct

    def get_undone_batch_num(self):
        with self.lock:
            return self.undone_batch_num

    def get_unconsumed_batch_num(self):
        with self.lock:
            return self.unconsumed_batch_num

    def get_input_queue_size(self):
        return self.input_queue.qsize()

    def get_output_queue_size(self):
        return self.output_queue.qsize()

    def pop_output_queue(self, n):
        results = []
        assert n <= self.get_output_queue_size()
        while len(results) < n:
            results.append(self.output_queue.get())
        with self.lock:
            assert n <= self.unconsumed_batch_num
            self.unconsumed_batch_num -= n
        return results

    def forward_thread_func(self):
        try:
            with get_device_module().stream(self.forward_stream):
                self.forward_thread_func_()
        except Exception:
            traceback = get_exception_traceback()
            logger.error(f"TpModelWorkerClient hit an exception: {traceback}")
            self.parent_process.send_signal(signal.SIGUSR1)

    def try_commit_op(self, commit_all):
        def commit_comm_work(work: list[P2PWork]) -> None:
            for p2p_work in work:
                p2p_work.work.wait()
            work.clear()

        min_num = 0 if commit_all else self.pp_size * 3
        while len(self.send_proxy_works) > min_num:
            commit_comm_work(self.send_proxy_works.pop(0))

        model = getattr(self.scheduler.tp_worker.model_runner.model, "model", None)
        if model is not None and getattr(model, "aux_pp_direct", None):
            aux_min_num = 0 if commit_all else self.pp_size * 3
            while len(model._aux_send_works) > aux_min_num:
                model._aux_send_works.pop(0)[0].wait()

    def forward_thread_func_(self):
        # idle/busy time tracking (1-minute window, same pattern as decode loop)
        stats_window_seconds = 60.0
        window_start = time.time()

        while True:
            t_loop_start = time.time()
            try:
                batch = self.input_queue.get(timeout=30.0)
                commit_all = False
            except Exception:
                batch = None
                commit_all = True
            t_after_get = time.time()

            # Accumulate idle time (time spent waiting for a batch)
            self.total_idle_time += t_after_get - t_loop_start

            self.try_commit_op(commit_all)
            self.cur_batch = batch
            if batch is not None:
                self.scheduler.forward_ct += 1

                # 1. recv
                t_recv_start = time.time()
                if not self.pp_group.is_first_rank:
                    pp_proxy_tensors = PPProxyTensors(self.pp_group.recv_tensor_dict())
                    batch.pp_proxy_tensors = pp_proxy_tensors
                t_recv_end = time.time()

                # 2. run's launch
                result = self.scheduler.run_batch_inner(batch)
                t_launch_end = time.time()

                # 3. run end
                self.forward_stream.synchronize()
                t_forward_end = time.time()

                # 4. send
                if not self.pp_group.is_last_rank:
                    send_proxy_work = self.scheduler._pp_send_dict_to_next_stage(
                        result.logits_output.tensors,
                        async_send=True,
                    )
                    result.logits_output = None
                    self.send_proxy_works.append(send_proxy_work)
                    # aux isend 须在 proxy send 之后 (同 dst FIFO)
                    model = self.scheduler.tp_worker.model_runner.model
                    model = getattr(model, "language_model", model).model
                    if getattr(model, "aux_pp_direct", None):
                        model.send_pending_aux_async()
                t_send_end = time.time()

                # Accumulate active time (time spent processing the batch)
                self.total_active_time += t_send_end - t_recv_start

                timings = {}
                timings["idle"] = (t_after_get - t_loop_start) * 1000
                timings["commit"] = (t_recv_start - t_after_get) * 1000
                timings["total"] = (t_send_end - t_recv_start) * 1000
                timings["recv"] = (t_recv_end - t_recv_start) * 1000
                timings["launch"] = (t_launch_end - t_recv_end) * 1000
                timings["forward"] = (t_forward_end - t_recv_end) * 1000
                timings["send"] = (t_send_end - t_forward_end) * 1000

                assert batch.next_batch_sampling_info is None
                batch.pp_proxy_tensors = None
                self.output_queue.put((batch.copy(), result, timings, self.done_batch_ct))
                with self.lock:
                    self.done_batch_ct += 1
                    self.undone_batch_num -= 1

            elapsed = time.time() - window_start
            if elapsed >= stats_window_seconds:
                self._update_prefill_idle_ratio(elapsed)
                window_start = time.time()

    def _update_prefill_idle_ratio(self, elapsed):
        """Compute prefill idle/busy ratio over the elapsed window and store on scheduler."""
        if elapsed > 0:
            self.scheduler.stats.prefill_idle_ratio = self.total_idle_time / elapsed
            self.scheduler.stats.prefill_busy_ratio = self.total_active_time / elapsed
        self.total_idle_time = 0.0
        self.total_active_time = 0.0


class SchedulerPPMixin:
    def pp_process_batch_result_disagg_prefill(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ) -> None:
        if not self.pp_group.is_last_rank:
            for i, req in enumerate(batch.reqs):
                req: Req
                end_idx = batch.prefill_batch_info[i].context_len
                if req.is_chunked <= 0:
                    req.output_ids.append(-1)  # just for cache kv cache
                    self.tree_cache.cache_unfinished_req(req)  # update the tree and lock
                    self.disagg_prefill_inflight_queue.append(req)
                    self.add_send_task(req, last_chunk=True)
                    if req.grammar is not None:
                        req.grammar.finished = req.finished()
                else:
                    req.is_chunked -= 1
                    self.tree_cache.cache_unfinished_req(req, num_token_id=end_idx)
                    self.add_send_task(req, last_chunk=False, end_idx=end_idx)

            # We need to remove the sync in the following function for overlap schedule.
            self.set_next_batch_sampling_info_done(batch)
        else:
            self.process_batch_result_disagg_prefill(batch, result)

    def create_pp_tracker(self: Scheduler) -> PPTracker:
        if self.pp_rank == 0:
            pp0_ip = get_local_ip_by_remote()
            pp0_port = get_free_port()
            pp_tracker = PPTracker(self.pp_rank, self.pp_size, pp0_ip, pp0_port)
            broadcast_data = f"{pp0_ip}:{pp0_port}"
            self.pp_group.broadcast_object(broadcast_data, src=0)
        else:
            broadcast_data = self.pp_group.broadcast_object(src=0)
            pp0_ip, pp0_port_str = broadcast_data.split(":")
            pp0_port = int(pp0_port_str)
            pp_tracker = PPTracker(self.pp_rank, self.pp_size, pp0_ip, pp0_port)
        return pp_tracker

    @DynamicGradMode()
    def event_loop_pp_disagg_prefill(self: Scheduler):
        forward_thread = ForwardThread(self, self.device)
        self.local_done_batch_ct = 0  # 该pp完成的batch数，不同pp不同
        self.global_done_batch_ct = 0  # 所有pp完成的batch数, 所有pp相同
        self.launch_batch_ct = 0
        admission_limit_logged = False
        pp_tracker = self.create_pp_tracker()

        def _loop_process(step_func=None):
            configure_gc_warning(0.05)
            gc.collect()
            freeze_gc("scheduler_pp_disagg_prefill")
            logger.info(f"{gc.get_threshold()=}")
            gc.set_threshold(2500, 10, 5)
            logger.info(f"{gc.get_threshold()=} {gc.get_count()=}")

            while True:
                self.schedule_forward_ct += 1
                # EPD admission must run in the PP-specific loop as well; PP
                # prefill does not use event_loop_*_disagg_prefill.
                self.poll_epd_admissions()
                recv_reqs = self.recv_requests()

                self.process_input_requests(recv_reqs)
                if self.grammar_queue:
                    self.move_ready_grammar_requests()

                self.waiting_queue.extend(self.disagg_prefill_bootstrap_queue.pop_bootstrapped())
                if self.enable_hicache_storage:
                    self.process_prefetches()

                run_batch_async()
                check_batch_result(step_func)

                if self.enable_hierarchical_cache:
                    self.tree_cache.tick()
                self.process_disagg_prefill_inflight_queue()

                if self.global_done_batch_ct == self.launch_batch_ct and len(self.disagg_prefill_inflight_queue) == 0:
                    self.check_memory()
                    self.new_token_ratio = self.init_new_token_ratio

                if self.global_done_batch_ct == self.launch_batch_ct:
                    self.log_idle_stats()

        def attn_tp_pp_reduce_min(value):
            if self.attn_tp_pp_size > 1:
                value_tensor = torch.tensor([value], dtype=torch.int, device="cpu")
                torch.distributed.all_reduce(
                    value_tensor,
                    op=torch.distributed.ReduceOp.MIN,
                    group=self.attn_tp_pp_cpu_group,
                )
                return value_tensor.item()
            return value

        def attn_tp_pp_reduce_max(value):
            if self.attn_tp_pp_size > 1:
                value_tensor = torch.tensor([value], dtype=torch.int, device="cpu")
                torch.distributed.all_reduce(
                    value_tensor,
                    op=torch.distributed.ReduceOp.MAX,
                    group=self.attn_tp_pp_cpu_group,
                )
                return value_tensor.item()
            return value

        def run_batch_async():
            nonlocal admission_limit_logged
            # Stop admission when any rank has too many launched batches whose
            # results have not yet been consumed. This includes both unfinished
            # forward work and completed results retained in output_queue.
            local_unconsumed_batch_num = forward_thread.get_unconsumed_batch_num()
            unconsumed_batch_num = attn_tp_pp_reduce_max(local_unconsumed_batch_num)
            unconsumed_batch_limit = self.pp_size * 3
            if unconsumed_batch_num >= unconsumed_batch_limit:
                if not admission_limit_logged and self.attn_tp_rank == 0:
                    logger.info(
                        "PP admission reached unconsumed batch limit: "
                        f"pp_rank={self.pp_rank}, "
                        f"local={local_unconsumed_batch_num}, "
                        f"global={unconsumed_batch_num}, "
                        f"limit={unconsumed_batch_limit}"
                    )
                admission_limit_logged = True
                # block too many batch，skip launch batch
                return
            batch = self.get_new_batch_prefill(log=False)
            self.update_oe_info(batch)
            if batch:
                if self.preparing_schedule_batch is None:
                    self.preparing_schedule_batch = batch
                else:
                    self.preparing_schedule_batch.merge_batch(batch, is_prefill=True)
            # 1. batch is full, launch
            # 2. batch is not full, get_undo_batch_ct() == 0 is not running, launch
            # 3. batch is not full, pp0 is running: just merge batch -> waiting enough reqs
            if self.preparing_adder and self.preparing_adder.is_full():
                launch_batch = 1
            else:
                undone_batch_num = attn_tp_pp_reduce_min(forward_thread.get_undone_batch_num())
                if undone_batch_num == 0:
                    launch_batch = 1
                else:
                    launch_batch = 0

            if launch_batch:
                if self.attn_tp_rank == 0 and self.preparing_adder and len(self.preparing_adder.batch_info) > 0:
                    self.log_prefill_stats(self.preparing_adder, 0, self.launch_batch_ct)
                batch = self.preparing_schedule_batch
                self.preparing_schedule_batch = None
                self.preparing_adder = None
            else:
                batch = None

            if self.server_args.enable_dp_attention:
                batch = self.prepare_dp_attn_batch(batch)
            if batch:
                self.launch_batch_ct += 1
                forward_thread.run_batch_async(batch)
            self.batch_is_full = False

        def check_batch_result(step_func):
            new_done_batch_ct = forward_thread.get_done_batch_ct()
            while self.local_done_batch_ct < new_done_batch_ct:
                if step_func and get_attn_tp_group().rank_in_group >= 0:
                    step_func()
                self.local_done_batch_ct += 1

            output_queue_size = forward_thread.get_output_queue_size()
            output_queue_size = attn_tp_pp_reduce_min(output_queue_size)
            batch_results = forward_thread.pop_output_queue(output_queue_size)
            for batch, result, timings, it in batch_results:
                if get_attn_tp_group().rank_in_group == 0:
                    pp_tracker.update_batch_metric(it, timings, batch.prefill_batch_info)
                    slm = self.stats.scheduler_latency_metrics
                    if slm is not None:
                        fwd_ms = timings.get("forward", 0.0) if isinstance(timings, dict) else 0.0
                        if fwd_ms > 0.0:
                            sum_kv = 0
                            if batch.prefill_batch_info:
                                sum_kv = sum(info.context_len for info in batch.prefill_batch_info)
                            slm.add_chunk_forward(fwd_ms, sum_kv)
                self.global_done_batch_ct += 1
                self.pp_process_batch_result_disagg_prefill(batch, result)

            if self.pp_rank == 0 and get_attn_tp_group().rank_in_group == 0:
                batch_metrics = pp_tracker.check_batch_metrics()
                for it in sorted(batch_metrics.keys()):
                    logger.info(f"BatchMetric {it=}: {batch_metrics[it]}")
            return new_done_batch_ct

        if self.npu_custom_profiler:
            with self.npu_custom_profiler as p:
                p.start()
                _loop_process(p.step)  # do npu process
                p.stop()
        else:
            _loop_process()

    def _pp_send_dict_to_next_stage(
        self: Scheduler,
        tensor_dict: dict[str, torch.Tensor],
        async_send: bool = True,
    ):
        assert not self.pp_group.is_last_rank
        p2p_works = self.pp_group.send_tensor_dict(
            tensor_dict=tensor_dict,
            all_gather_group=None,
            async_send=async_send,
        )
        return p2p_works

    def _pp_recv_dict_from_prev_stage(
        self: Scheduler,
    ) -> dict[str, torch.Tensor]:
        assert not self.pp_group.is_first_rank
        res = self.pp_group.recv_tensor_dict(
            all_gather_group=None,
        )
        return res
