import dataclasses
import math
import signal
import threading
from queue import Queue

import psutil
import torch
import triton
import triton.language as tl

from sglang.srt.env import ENV
from sglang.srt.managers.scheduling.schedule_batch import ModelWorkerBatch
from sglang.srt.utils import (
    get_colorful_logger,
    is_npu,
)
from sglang.utils import get_exception_traceback

logger = get_colorful_logger(__name__)

__is_npu__ = is_npu()


@triton.jit
def resolve_future_input_kernel(
    last_verified_ids_ptr,  # [bs]
    token_list_ptr,  # [bs, steps]
    future_last_map_ptr,  # [max_indices]
    future_tokens_ptr,  # [max_indices, steps]
    spec_steps,
    token_list_stride_0,
    future_tokens_stride_0,
):
    pid = tl.program_id(0)
    curr_last_id = tl.load(last_verified_ids_ptr + pid)
    # cur batch not resolved
    if curr_last_id < 0:
        neg_id = -curr_last_id
        index = neg_id if neg_id > 0 else 0
        future_last_id = tl.load(future_last_map_ptr + index)
        tl.store(last_verified_ids_ptr + pid, future_last_id)
        for i in range(spec_steps):
            curr_t_ptr = token_list_ptr + pid * token_list_stride_0 + i
            curr_f_ptr = future_tokens_ptr + index * future_tokens_stride_0 + i
            t_val = tl.load(curr_t_ptr)
            if t_val < 0:
                f_val = tl.load(curr_f_ptr)
                tl.store(curr_t_ptr, f_val)


@triton.jit
def copy_for_next_launch(
    future_last_verified_id_ptr,
    future_token_list_ptr,
    new_verified_id_ptr,
    token_list_ptr,
    future_token_ids_ct,
    spec_steps: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    new_verified_id = tl.load(new_verified_id_ptr + pid)
    tl.store(future_last_verified_id_ptr + future_token_ids_ct + pid + 1, new_verified_id)
    row_idx = future_token_ids_ct + pid + 1
    for i in range(spec_steps):
        t_val = tl.load(token_list_ptr + pid * spec_steps + i)
        offset = row_idx * spec_steps + i
        tl.store(future_token_list_ptr + offset, t_val)


def resolve_future_input(
    batch: ModelWorkerBatch,
    spec_steps: int,
    future_last_verifed_ids_map: torch.Tensor,
    future_token_list_map: torch.Tensor,
) -> None:
    if batch.spec_info is None:
        return
    if __is_npu__:
        indices = torch.clamp(-batch.spec_info.last_verified_ids, min=0)
        batch.spec_info.last_verified_ids[:] = torch.where(
            batch.spec_info.last_verified_ids < 0,
            future_last_verifed_ids_map[indices],
            batch.spec_info.last_verified_ids,
        )
        batch.spec_info.token_list[:] = torch.where(
            batch.spec_info.token_list < 0,
            future_token_list_map[indices],
            batch.spec_info.token_list,
        )
    else:
        bs = len(batch.seq_lens)
        token_list = batch.spec_info.token_list
        resolve_future_input_kernel[(bs,)](
            last_verified_ids_ptr=batch.spec_info.last_verified_ids,
            token_list_ptr=batch.spec_info.token_list,
            future_last_map_ptr=future_last_verifed_ids_map,
            future_tokens_ptr=future_token_list_map,
            spec_steps=spec_steps,
            token_list_stride_0=token_list.stride(0),
            future_tokens_stride_0=future_token_list_map.stride(0),
        )


class EagleWorkerOverlapped:
    def __init__(
        self,
        server_args,
        gpu_id,
        attn_tp_rank,
        moe_ep_rank,
        nccl_port,
        target_worker,
        global_rank,
    ):
        if server_args.is_multi_head_eagle:
            from sglang.srt.speculative.multi_head_eagle_worker import MultiHeadEAGLEWorker

            WorkerCls = MultiHeadEAGLEWorker
        elif server_args.speculative_algorithm == "DFLASH":
            from sglang.srt.speculative.dflash_worker import DFlashWorker

            WorkerCls = DFlashWorker
        else:
            from sglang.srt.speculative.eagle_worker import EAGLEWorker

            WorkerCls = EAGLEWorker
        self.worker = WorkerCls(
            server_args,
            gpu_id,
            attn_tp_rank,
            moe_ep_rank,
            nccl_port,
            target_worker,
            global_rank,
        )

        assert self.worker.server_args.speculative_eagle_topk == 1, "Currently only support topk == 1 and spec_steps == 1 in Overlap Scheduler"

        self.device = self.worker.device
        self.gpu_id = gpu_id
        torch.get_device_module(self.device).set_device(gpu_id)
        self.model_runner = self.worker.model_runner
        self.model_runner_list = self.worker.model_runner_list

        context_len = self.worker.model_runner.model_config.context_len
        chunk_size = self.worker.model_runner.server_args.chunked_prefill_size
        max_chunk_times = 1 if chunk_size == -1 else math.ceil(context_len / chunk_size)
        future_max_num_tokens = self.worker.max_running_requests * (max_chunk_times + 1)

        self.future_token_ids_limit = future_max_num_tokens
        future_map_size = self.future_token_ids_limit + self.worker.max_running_requests

        # 第一次 launch 的 batch 在 GPU 上执行时, 第二次调度的 batch 可能已经 launch
        # 而第三次的 launch 一定在第一次 launch 的 batch 执行完之后 (由 copy_done 保证)
        # 所以需要在 forward 线程上保留两次 launch 的 batch 数量的引用, 防止 torch 回收
        # model_worker_batch（生命周期短） 中的张量空间导致 illeagl memory access。
        self.num_continuous_decode_steps = self.worker.server_args.num_continuous_decode_steps
        self.keep_batch_reference_num = 2 * self.num_continuous_decode_steps

        # These two queues are used to communicate with main schedule loop
        self.input_queue = Queue()
        self.output_queue = Queue()

        self.future_token_ids_ct = 0
        self.spec_steps = self.worker.speculative_num_steps

        # The following are reserved draft results for the currently scheduled batch, used to construct input during verify.
        # For now only consider spec_steps = 1 case. D2D copy should happen on these variables.
        self.future_last_verified_ids: torch.Tensor = torch.zeros(
            (future_map_size,),
            dtype=torch.int32,
            device=self.worker.device,
        )
        # draft tokens
        self.future_token_list: torch.Tensor = torch.zeros(
            (future_map_size, self.spec_steps),  # bs, spec_steps
            device=self.worker.device,
            dtype=torch.int32,
        )

        self.forward_stream = torch.get_device_module(self.device).Stream()
        if __is_npu__ and ENV.npu_enable_graph:
            with torch.get_device_module(self.device).stream(self.forward_stream):
                self.worker.target_worker.model_runner.init_npu_graphs()
                self.worker.init_npu_graphs()
        self.parent_process = psutil.Process().parent()
        self.forward_thread = threading.Thread(
            target=self.forward_thread_func,
        )
        self.forward_thread.start()
        self.scheduler_stream: torch.Stream = torch.get_device_module(self.device).current_stream()
        if __is_npu__:
            torch.npu.set_stream_limit(self.scheduler_stream, 8, 16)
        if self.device == "cpu":
            self.scheduler_stream.synchronize = lambda: None  # No-op for CPU

    def register_hicache_layer_transfer_counter(self, counter):
        return self.worker.register_hicache_layer_transfer_counter(counter)

    def forward_thread_func(self):
        try:
            # device 绑定是线程局部的: 新线程不继承主线程的 set_device, 首次触发
            # LazySetDevice 会落到 device 0 (同节点多 DP rank 时表现为该卡 OOM)。
            torch.get_device_module(self.device).set_device(self.gpu_id)
            with torch.get_device_module(self.device).stream(self.forward_stream):
                self.forward_thread_func_()
        except Exception:
            traceback = get_exception_traceback()
            logger.error(f"TpModelWorkerClient hit an exception: {traceback}")
            self.parent_process.send_signal(signal.SIGUSR1)

    @torch.no_grad()
    def forward_thread_func_(self):
        batch_pt = 0
        batch_lists = [None] * self.keep_batch_reference_num
        while True:
            (batch, future_token_ids_ct) = self.input_queue.get()
            if not batch:
                break

            # Keep a reference of model_worker_batch by storing it into a list.
            # Otherwise, the tensor members of model_worker_batch will be released
            # by pytorch and cause CUDA illegal memory access errors.
            batch_lists[batch_pt % self.keep_batch_reference_num] = batch
            batch_pt += 1

            launch_done = threading.Event()
            copy_done = torch.get_device_module(self.worker.device).Event(enable_timing=True)

            # Resolve future spec info in the input
            resolve_future_input(
                batch=batch,
                spec_steps=self.spec_steps,
                future_last_verifed_ids_map=self.future_last_verified_ids,
                future_token_list_map=self.future_token_list,
            )
            # For multi-step decode, disable async vocab mask wait since the scheduler
            # doesn't compute it between steps. This makes the forward thread compute
            # vocab_mask inline (same as non-overlap mode).
            if self.num_continuous_decode_steps > 1 and batch.sampling_info:
                batch.sampling_info.sampling_info_done = None
            (
                logits_output,
                next_token_ids,
                accept_lengths,
                new_verified_id,
                token_list,
                grammar_terminated_list,
            ) = self.worker.forward_batch_speculative_generation(batch, launch_done)

            if __is_npu__:
                # NPU forwards a snapshot or an async grammar-update event.
                # The consumer waits for the event before reading the snapshot.
                grammar_collect_slot = -1
            else:
                # Grammar accept now runs as a host node on the shared grammar stream
                # for BOTH the graph verify path (SpecDecodeCudaGraphRunner.replay)
                # and the eager verify path (base_spec_worker._prepare_eager_grammar_
                # verify). Whichever ran records worker.last_grammar_slot; the forward
                # thread only carries that slot and the consumer collects the ring
                # snapshot after copy_done.synchronize().
                #
                # Under launch-ahead we MUST NOT collect the snapshot here: that would
                # read terminated_ring[slot] before the accept host node has run and
                # force an overlap-breaking sync. Collect on the consumer thread
                # instead (resolve_batch_result).
                grammar_terminated_list = None
                grammar_collect_slot = self.worker.last_grammar_slot

            # Trigger D2D copy, prepare for next kernel launch
            bs = len(batch.seq_lens)
            if not batch.forward_mode.is_idle():
                if __is_npu__:
                    self.future_last_verified_ids[future_token_ids_ct + 1 : future_token_ids_ct + bs + 1] = new_verified_id
                    self.future_token_list[future_token_ids_ct + 1 : future_token_ids_ct + bs + 1] = token_list
                else:
                    copy_for_next_launch[(bs,)](
                        future_last_verified_id_ptr=self.future_last_verified_ids,
                        future_token_list_ptr=self.future_token_list,
                        new_verified_id_ptr=new_verified_id,
                        token_list_ptr=token_list,
                        future_token_ids_ct=future_token_ids_ct,
                        spec_steps=self.spec_steps,
                    )
                if logits_output.hidden_states is not None:
                    logits_output.hidden_states = logits_output.hidden_states.to("cpu", non_blocking=True)
                next_token_ids = next_token_ids.to("cpu", non_blocking=True)
                accept_lengths_cpu = accept_lengths.to("cpu", non_blocking=True) if accept_lengths is not None else None
                # Transfer logprob tensors to CPU if computed
                if batch.return_logprob:
                    logits_output.transfer_logprobs_to_cpu()
            else:
                accept_lengths_cpu = None
            copy_done.record()

            self.output_queue.put(
                (
                    copy_done,
                    launch_done,
                    logits_output,
                    next_token_ids,
                    accept_lengths_cpu,
                    new_verified_id,
                    token_list,
                    batch.return_logprob,
                    grammar_terminated_list,
                    grammar_collect_slot,
                )
            )

    def forward_batch_speculative_generation(self, batch: ModelWorkerBatch, launch_done=None):
        # Create a new copy of sampling_info because it will be updated in-place by the scheduler for the next batch.
        sampling_info = batch.sampling_info
        # For MTP, ALL grammar operations (per-position vocab-mask DFS and the
        # permanent accept) happen on this forward thread. So we deliberately set
        # sampling_info_done=None: the forward thread must NOT wait for the
        # scheduler to compute a vocab mask (it builds its own and discards the
        # scheduler's), and the scheduler must NOT read the grammar at all. The
        # scheduler-side sites that call update_regex_vocab_mask / .set() are
        # guarded on `sampling_info_done is not None`, so a None event makes them
        # skip grammar entirely (no cross-thread read/write race on the matcher).
        batch.sampling_info = self.cur_sampling_info = dataclasses.replace(
            sampling_info,
            sampling_info_done=None,
            penalizer_orchestrator=None,
        )
        # A cuda stream sync here to avoid the cuda illegal memory access error.
        self.scheduler_stream.synchronize()
        self.input_queue.put((batch, self.future_token_ids_ct))

        # Allocate output future objects
        bs = len(batch.seq_lens)

        future_last_verified_ids = torch.arange(
            -(self.future_token_ids_ct + 1),
            -(self.future_token_ids_ct + 1 + bs),
            -1,
            dtype=torch.int32,
            device=self.worker.device,
        )
        future_tokens_list = [
            torch.arange(
                -(self.future_token_ids_ct + 1),
                -(self.future_token_ids_ct + 1 + bs),
                -1,
                dtype=torch.int32,
                device=self.worker.device,
            ).unsqueeze(1)
            for _ in range(self.spec_steps)
        ]
        future_tokens_list = torch.cat(future_tokens_list, dim=-1)

        self.future_token_ids_ct = (self.future_token_ids_ct + bs) % self.future_token_ids_limit
        grammar_terminated_list = None
        return (
            None,
            None,
            None,
            future_last_verified_ids,
            future_tokens_list,
            grammar_terminated_list,
        )

    def resolve_batch_result(self, bid: int):
        (
            copy_done,
            launch_done,
            logits_output,
            next_token_ids,
            accept_lengths_cpu,
            new_verified_id,
            token_list,
            return_logprob,
            grammar_terminated_list,
            grammar_collect_slot,
        ) = self.output_queue.get()
        copy_done.synchronize()
        # Collect the grammar termination snapshot HERE, after copy_done is
        # synchronized. For both the graph and eager verify paths, copy_done is
        # recorded after the grammar-stream join (graph end / eager
        # main_stream.wait_stream), so the accept host node's writes to
        # terminated_ring[slot] / error_ring[slot] are now visible. Doing it here
        # (not on the forward thread) keeps launch-ahead intact: the forward
        # thread never blocks on the grammar result.
        if grammar_collect_slot >= 0:
            ctx = self.worker.grammar_ctx
            accept_lens = accept_lengths_cpu.tolist() if accept_lengths_cpu is not None else []
            grammar_terminated_list = ctx.collect_terminated(grammar_collect_slot, accept_lens)
            error = ctx.error_ring[grammar_collect_slot]
            if bool(error.any().item()):
                bad = [i for i, e in enumerate(error.tolist()) if e]
                logger.error(f"[EAGLEWorkerOverlap] in-graph grammar accept failed for batch slots {bad}; their constrained output may be invalid.")
        # Wait here for current batch target kernel launch to complete
        launch_done.wait()
        if isinstance(grammar_terminated_list, threading.Event):
            assert __is_npu__, "Only NPU workers support async grammar update"
            # NPU: the grammar accept loop runs in the worker's grammar-update
            # thread. Wait here (on the scheduler thread, off the forward
            # critical path) until the materialized list is ready.
            grammar_terminated_list.wait()
            grammar_terminated_list = grammar_terminated_list.grammar_terminated_list
        if next_token_ids is not None:
            next_token_ids = next_token_ids.tolist()
        # Convert logprob tensors to list after GPU→CPU sync
        if return_logprob and logits_output is not None:
            logits_output.convert_logprobs_to_python()
        # Return to enter post-processing, after post-processing schedule next batch, at this time target GPU kernels have not finished running
        # can achieve overlap effect
        return (
            logits_output,
            next_token_ids,
            accept_lengths_cpu,
            new_verified_id,
            token_list,
            grammar_terminated_list,
        )

    def renew_scaling_penalty(self, req_pool_indices: torch.Tensor):
        self.worker.renew_scaling_penalty(req_pool_indices)

    def update_weights_from_tensor(self, recv_req):
        success, message = self.worker.update_weights_from_tensor(recv_req)
        return success, message
