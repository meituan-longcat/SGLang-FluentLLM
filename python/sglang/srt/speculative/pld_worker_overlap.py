# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project

"""
PLD Worker with Overlap Scheduling
"""

import dataclasses
import signal
import threading
from queue import Queue
from typing import List

import psutil
import torch

from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.speculative.eagle_utils import EagleDraftOutput
from sglang.srt.speculative.pld_worker import PLDWorker
from sglang.srt.utils import get_colorful_logger
from sglang.utils import get_exception_traceback
try:
    from flashinfer import resolve_future_tensors
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to import resolve_future_tensors from flashinfer: {e}")
    raise


logger = get_colorful_logger(__name__)

def resolve_future_input(
    batch: ModelWorkerBatch,
    spec_steps: int,
    future_last_verified_ids_map: torch.Tensor,
    future_token_list_map: List[torch.Tensor],
) -> EagleDraftOutput:
    """
    Resolve future objects in PLD batch spec_info using CUDA kernel.
    """
    if batch.spec_info is None:
        return

    indices = torch.clamp(-batch.spec_info.last_verified_ids, min=0)

    batch.spec_info.last_verified_ids[:] = torch.where(
        batch.spec_info.last_verified_ids < 0,
        future_last_verified_ids_map[indices],
        batch.spec_info.last_verified_ids,
    )

    # Resolve tokens with single CUDA kernel call
    token_list = batch.spec_info.token_list
    if isinstance(token_list, torch.Tensor):
        # PD separation decode: token_list is [bs, spec_steps]
        stacked_tokens = token_list[:, :spec_steps].transpose(0, 1).unsqueeze(-1)
        stacked_future_tokens = torch.stack(future_token_list_map[:spec_steps])
        resolve_future_tensors(
            stacked_tokens,
            stacked_future_tokens,
            stacked_tokens,
            indices,
            spec_steps,
        )
        token_list[:, :spec_steps] = stacked_tokens.squeeze(-1).transpose(0, 1)
    else:
        stacked_tokens = torch.stack([token_list[i] for i in range(spec_steps)])
        stacked_future_tokens = torch.stack(future_token_list_map[:spec_steps])
        resolve_future_tensors(
            stacked_tokens,
            stacked_future_tokens,
            stacked_tokens,
            indices,
            spec_steps,
        )
        # return stacked_tokens
        for i in range(spec_steps):
            token_list[i][:] = stacked_tokens[i]


class PLDWorkerOverlapped:
    """
    PLD worker with overlap scheduling support.

    This class implements asynchronous processing for PLD (Prompt Lookup Decode)
    to enable CPU-GPU overlap, similar to EagleWorkerOverlapped but adapted
    for PLD's simpler n-gram based approach.
    """

    def __init__(
        self,
        server_args,
        gpu_id: int,
        attn_tp_rank: int,
        dp_rank: int,
        nccl_port: int,
        target_worker,
        global_rank: int,
    ):
        self.worker = PLDWorker(
            server_args,
            gpu_id,
            attn_tp_rank,
            dp_rank,
            nccl_port,
            target_worker,
            global_rank,
        )

        self.device = self.worker.device
        torch.get_device_module(self.device).set_device(gpu_id)
        self.server_args = server_args

        context_len = self.worker.target_worker.model_runner.model_config.context_len
        chunk_size = server_args.chunked_prefill_size or context_len
        max_chunk_times = (context_len + chunk_size - 1) // chunk_size
        future_max_num_tokens = self.worker.target_worker.max_running_requests * (max_chunk_times + 1)

        self.future_token_ids_limit = future_max_num_tokens
        future_map_size = self.future_token_ids_limit + self.worker.target_worker.max_running_requests

        self.input_queue = Queue()
        self.output_queue = Queue()

        self.future_token_ids_ct = 0
        self.draft_token_num = server_args.speculative_num_draft_tokens
        self.spec_steps = self.draft_token_num - 1

        self.future_last_verified_ids = torch.zeros(
            (future_map_size,),
            dtype=torch.int32,
            device=self.device,
        )

        self.future_token_list = [
            torch.zeros(
                (future_map_size, 1),
                device=self.device,
                dtype=torch.int32,
            )
            for _ in range(self.draft_token_num - 1)
        ]

        self.forward_stream = torch.get_device_module(self.device).Stream()
        self.parent_process = psutil.Process().parent()
        self.forward_thread = threading.Thread(
            target=self.forward_thread_func,
            daemon=True,
        )
        self.forward_thread.start()

        self.scheduler_stream = torch.get_device_module(self.device).current_stream()
        if self.device == "cpu":
            self.scheduler_stream.synchronize = lambda: None  # No-op for CPU

        self.cur_sampling_info = None

    def forward_thread_func(self):
        """Background thread function with exception handling."""
        try:
            with torch.get_device_module(self.device).stream(self.forward_stream):
                self.forward_thread_func_()
        except Exception:
            traceback = get_exception_traceback()
            logger.error(f"PLDWorkerOverlapped hit an exception: {traceback}")
            self.parent_process.send_signal(signal.SIGUSR1)

    @torch.no_grad()
    def forward_thread_func_(self):
        """Main background processing loop."""
        batch_pt = 0
        batch_lists = [None] * 2

        while True:
            queue_item = self.input_queue.get()
            if queue_item is None: 
                break

            batch, future_token_ids_ct = queue_item

            batch_lists[batch_pt % 2] = batch
            batch_pt += 1

            launch_done = threading.Event()
            copy_done = torch.get_device_module(self.device).Event()

            resolve_future_input(
                batch=batch,
                spec_steps=self.draft_token_num - 1,
                future_last_verified_ids_map=self.future_last_verified_ids,
                future_token_list_map=self.future_token_list,
            )

            (
                logits_output,
                next_token_ids,
                accept_lengths,
                new_verified_id,
                token_list,
            ) = self.worker.forward_batch_speculative_generation(batch, launch_done)

            bs = len(batch.seq_lens)
            if not batch.forward_mode.is_idle():
                actual_bs = new_verified_id.shape[0] if new_verified_id is not None else bs

                self.future_last_verified_ids[
                    future_token_ids_ct + 1 : future_token_ids_ct + actual_bs + 1
                ] = new_verified_id[:actual_bs]

                for i in range(len(token_list)):
                    self.future_token_list[i][
                        future_token_ids_ct + 1 : future_token_ids_ct + actual_bs + 1
                    ] = token_list[i][:actual_bs]

                if logits_output and logits_output.hidden_states is not None:
                    logits_output.hidden_states = logits_output.hidden_states.to(
                        "cpu", non_blocking=True
                    )
                if next_token_ids is not None:
                    next_token_ids = next_token_ids.to("cpu", non_blocking=True)
                if accept_lengths is not None:
                    accept_lengths_cpu = accept_lengths.to("cpu", non_blocking=True)
                else:
                    accept_lengths_cpu = None
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
                )
            )

    def forward_batch_speculative_generation(
        self, batch: ModelWorkerBatch, launch_done=None
    ):
        """
        Main entry point for speculative generation in overlap mode.

        This method immediately returns future objects while the actual
        computation happens in the background thread.
        """
        sampling_info = batch.sampling_info
        batch.sampling_info = self.cur_sampling_info = dataclasses.replace(
            sampling_info,
            sampling_info_done=threading.Event(),
            penalizer_orchestrator=None,
        )

        self.scheduler_stream.synchronize()

        self.input_queue.put((batch, self.future_token_ids_ct))

        bs = len(batch.seq_lens)

        future_last_verified_ids = torch.arange(
            -(self.future_token_ids_ct + 1),
            -(self.future_token_ids_ct + 1 + bs),
            -1,
            dtype=torch.int32,
            device=self.device,
        )

        future_token_list = [
            torch.arange(
                -(self.future_token_ids_ct + 1),
                -(self.future_token_ids_ct + 1 + bs),
                -1,
                dtype=torch.int32,
                device=self.device,
            ).unsqueeze(1)
            for _ in range(self.draft_token_num - 1)
        ]

        self.future_token_ids_ct = (
            self.future_token_ids_ct + bs
        ) % self.future_token_ids_limit

        return (
            None,  # logits_output (will be resolved later)
            None,  # next_token_ids (will be resolved later)
            None,  # accept_lengths (will be resolved later)
            future_last_verified_ids,
            future_token_list,
        )

    def resolve_batch_result(self, bid: int):
        """
        Resolve the actual computation results from the background thread.

        This method blocks until the background computation is complete
        and returns the actual results.
        """
        (
            copy_done,
            launch_done,
            logits_output,
            next_token_ids,
            accept_lengths_cpu,
            new_verified_id,
            token_list,
        ) = self.output_queue.get()

        copy_done.synchronize()
        launch_done.wait()

        if next_token_ids is not None:
            next_token_ids = next_token_ids.tolist()

        return (
            logits_output,
            next_token_ids,
            accept_lengths_cpu,
            new_verified_id,
            token_list,
        )

    def shutdown(self):
        """Gracefully shutdown the background thread."""
        self.input_queue.put(None)  # Shutdown signal
        if self.forward_thread.is_alive():
            self.forward_thread.join(timeout=5.0)
