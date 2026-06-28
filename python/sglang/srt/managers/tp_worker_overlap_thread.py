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
"""A tensor parallel worker."""

import dataclasses
from sglang.srt.utils import get_colorful_logger
import signal
import threading
from queue import Queue
from typing import Optional

import psutil
import torch
import math
from sglang.srt.env import ENV
from sglang.srt.managers.io_struct import (
    GetWeightsByNameReqInput,
    InitWeightsUpdateGroupReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    get_compiler_backend,
    is_npu
)
from sglang.utils import get_exception_traceback

logger = get_colorful_logger(__name__)


# @torch.compile(dynamic=True, backend=get_compiler_backend())
def resolve_future_token_ids(input_ids, future_token_ids_map):
    input_ids[:] = torch.where(
        input_ids < 0,
        future_token_ids_map[torch.clamp(-input_ids, min=0)],
        input_ids,
    )


class TpModelWorkerClient:
    """A tensor parallel model worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        attn_tp_rank: int,
        moe_ep_rank: int,
        global_rank: int,
        nccl_port: int,
    ):
        # Load the model
        self.worker = TpModelWorker(server_args, gpu_id, attn_tp_rank, moe_ep_rank, global_rank, nccl_port, enable_overlap=True)
        self.model_config = self.worker.model_config
        self.model_runner = self.worker.model_runner
        self.max_running_requests = self.worker.max_running_requests
        self.device = self.worker.device
        self.gpu_id = gpu_id

        # Init future mappings

        context_len = self.worker.model_runner.model_config.context_len
        chunk_size = self.worker.model_runner.server_args.chunked_prefill_size
        max_chunk_times = 1 if chunk_size == -1 else math.ceil(context_len / chunk_size)
        # Max value is chunk count of max_running Prefill requests plus max_running future indices reserved for decode batch
        future_max_num_tokens = self.max_running_requests * (max_chunk_times + 1)

        self.future_token_ids_ct = 0
        self.future_token_ids_limit = future_max_num_tokens

        # Upper bound for write and resolve is future_token_ids_limit + max_running_requests, see line 157 and line 215
        self.future_token_ids_map = torch.empty(
            (self.future_token_ids_limit + self.max_running_requests,), dtype=torch.int32, device=self.device
        )

        # Launch threads
        self.input_queue = Queue()
        self.output_queue = Queue()
        # self.forward_stream = self.worker.model_runner.forward_stream
        self.forward_stream = torch.get_device_module(self.device).Stream()
        if is_npu() and ENV.npu_enable_graph:
            with torch.get_device_module(self.device).stream(self.forward_stream):
                self.worker.model_runner.init_npu_graphs()
        self.parent_process = psutil.Process().parent()
        # 第一次 launch 的 batch 在 GPU 上执行时, 第二次调度的 batch 可能已经 launch
        # 而第三次的 launch 一定在第一次 launch 的 batch 执行完之后 (由 copy_done 保证)
        # 所以需要在 forward 线程上保留两次 launch 的 batch 数量的引用, 防止 torch 回收
        # model_worker_batch（生命周期短） 中的张量空间导致 illeagl memory access。
        self.num_continuous_decode_steps = server_args.num_continuous_decode_steps
        self.keep_batch_reference_num = 2 * self.num_continuous_decode_steps
        self.forward_thread = threading.Thread(
            target=self.forward_thread_func,
        )
        self.forward_thread.start()
        self.scheduler_stream = torch.get_device_module(self.device).current_stream()
        if is_npu():
            torch.npu.set_stream_limit(self.scheduler_stream, 8, 16)
        if self.device == "cpu":
            self.scheduler_stream.synchronize = lambda: None  # No-op for CPU
        self.use_over_embedding = getattr(self.worker.model_config, "use_over_embedding", False)
        if self.use_over_embedding:
            self.oe_ignore_tokens = torch.tensor(self.worker.model_config.oe_ignore_tokens, device=self.device)
        else:
            self.oe_ignore_tokens = None

    def get_worker_info(self):
        return self.worker.get_worker_info()

    def get_pad_input_ids_func(self):
        return self.worker.get_pad_input_ids_func()

    def get_tp_cpu_group(self):
        return self.worker.get_tp_cpu_group()

    def get_tp_group(self):
        return self.worker.get_tp_group()

    def get_attention_tp_cpu_group(self):
        return self.worker.get_attention_tp_cpu_group()

    def get_attention_tp_group(self):
        return self.worker.get_attention_tp_group()

    def get_tp_group(self):
        return self.worker.get_tp_group()

    def get_memory_pool(self):
        return (
            self.worker.model_runner.req_to_token_pool,
            self.worker.model_runner.token_to_kv_pool,
            self.worker.model_runner.kv_allocator,
        )

    def register_hicache_layer_transfer_counter(self, counter):
        return self.worker.register_hicache_layer_transfer_counter(counter)

    def set_hicache_consumer(self, consumer_index: int):
        return self.worker.set_hicache_consumer(consumer_index)

    def forward_thread_func(self):
        try:
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
            model_worker_batch, future_token_ids_ct = self.input_queue.get()
            if not model_worker_batch:
                break

            # Keep a reference of model_worker_batch by storing it into a list.
            # Otherwise, the tensor members of model_worker_batch will be released
            # by pytorch and cause CUDA illegal memory access errors.
            batch_lists[batch_pt % self.keep_batch_reference_num] = model_worker_batch
            batch_pt += 1

            # Create event
            copy_done = torch.get_device_module(self.device).Event()

            # Resolve future tokens in the input
            input_ids = model_worker_batch.input_ids
            resolve_future_token_ids(input_ids,
                                     self.future_token_ids_map)

            # Run forward
            logits_output, next_token_ids, next_token_multi_ids = self.worker.forward_batch_generation(
                model_worker_batch, model_worker_batch.launch_done
            )

            # Update the future token ids map
            bs = len(model_worker_batch.seq_lens)
            self.future_token_ids_map[
                future_token_ids_ct + 1 : future_token_ids_ct + bs + 1
            ] = next_token_ids

            # Copy results to the CPU
            if model_worker_batch.return_logprob:
                logits_output.next_token_logprobs = (
                    logits_output.next_token_logprobs.to("cpu", non_blocking=True)
                )
                if logits_output.input_token_logprobs is not None:
                    logits_output.input_token_logprobs = (
                        logits_output.input_token_logprobs.to("cpu", non_blocking=True)
                    )
            if logits_output.hidden_states is not None:
                logits_output.hidden_states = logits_output.hidden_states.to(
                    "cpu", non_blocking=True
                )
            self.worker.model_runner.req_to_token_pool.verified_lens[model_worker_batch.req_pool_indices] += model_worker_batch.new_tokens_to_compute
            next_token_ids = next_token_ids.to("cpu", non_blocking=True)
            if next_token_multi_ids is not None:
                next_token_multi_ids = next_token_multi_ids.to("cpu", non_blocking=True)
            copy_done.record()

            self.output_queue.put((copy_done, logits_output, next_token_ids, next_token_multi_ids))

    def resolve_last_batch_result(self, launch_done: Optional[threading.Event] = None):
        copy_done, logits_output, next_token_ids, next_token_multi_ids = self.output_queue.get()

        if launch_done is not None:
            launch_done.wait()
        copy_done.synchronize()

        if logits_output.next_token_logprobs is not None:
            logits_output.next_token_logprobs = (
                logits_output.next_token_logprobs.tolist()
            )
            if logits_output.input_token_logprobs is not None:
                logits_output.input_token_logprobs = tuple(
                    logits_output.input_token_logprobs.tolist()
                )
        next_token_ids = next_token_ids.tolist()
        return logits_output, next_token_ids, next_token_multi_ids

    def forward_batch_generation(self, model_worker_batch: ModelWorkerBatch):
        # Create a new copy of sampling_info because it will be updated in-place by the scheduler for the next batch.
        sampling_info = model_worker_batch.sampling_info
        sampling_info.update_penalties()
        model_worker_batch.sampling_info = self.cur_sampling_info = dataclasses.replace(
            sampling_info,
            sampling_info_done=threading.Event(),
            penalizer_orchestrator=None,
        )

        # A cuda stream sync here to avoid the cuda illegal memory access error.
        self.scheduler_stream.synchronize()

        # Push a new batch to the queue
        self.input_queue.put((model_worker_batch, self.future_token_ids_ct))

        # Allocate output future objects
        bs = len(model_worker_batch.seq_lens)
        future_next_token_ids = torch.arange(
            -(self.future_token_ids_ct + 1),
            -(self.future_token_ids_ct + 1 + bs),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self.future_token_ids_ct = (
            self.future_token_ids_ct + bs
        ) % self.future_token_ids_limit
        return None, future_next_token_ids, None

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        success, message = self.worker.update_weights_from_disk(recv_req)
        return success, message

    def init_weights_update_group(self, recv_req: InitWeightsUpdateGroupReqInput):
        success, message = self.worker.init_weights_update_group(recv_req)
        return success, message

    def update_weights_from_distributed(
        self, recv_req: UpdateWeightsFromDistributedReqInput
    ):
        success, message = self.worker.update_weights_from_distributed(recv_req)
        return success, message

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        success, message = self.worker.update_weights_from_tensor(recv_req)
        return success, message

    def get_weights_by_name(self, recv_req: GetWeightsByNameReqInput):
        return self.worker.get_weights_by_name(recv_req)

    def __delete__(self):
        self.input_queue.put((None, None))
        self.copy_queue.put((None, None, None))
