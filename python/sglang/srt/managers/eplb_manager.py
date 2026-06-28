import logging
import time
import os
import threading
import queue
import numpy as np
from typing import TYPE_CHECKING, List, Any, Optional
from dataclasses import dataclass
from datetime import timedelta

import torch

from sglang.srt.managers.expert_distribution import (
    get_global_expert_distribution_recorder,
)
from sglang.srt.managers.expert_location import ExpertLocationMetadata
from sglang.srt.metrics.collector import EPLBMetricsCollector

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

from sglang.srt.utils import get_colorful_logger, is_npu, device_synchronize, get_device_module, get_910b_num_gpus_per_node
logger = get_colorful_logger(__name__)
_is_npu = is_npu()

class BackgroundWorker:
    def __init__(self, name, device_id, task_func, device, ep_rank=0):
        self.device = f"device:{device_id}"
        self.running = True
        self.taks_func = task_func
        self.name = name
        self.device_id = device_id
        self.ep_rank = ep_rank
        self.device = device
        
        # 1. 通信管道
        self.task_queue = queue.Queue(maxsize=1) # 主线程 -> 后台
        self.result_queue = queue.Queue(maxsize=1)  # 后台 -> 主线程
        self.finished_event = threading.Event()
        
        # 2. 启动常驻线程
        self.thread = threading.Thread(target=self._loop, daemon=True, name=f"{name}_{self.device}")
        self.thread.start()

    def _loop(self):
        get_device_module().set_device(self.device_id) 
        self.stream = torch.get_device_module(self.device).Stream()
        if _is_npu and get_910b_num_gpus_per_node() == 0:
            torch.npu.set_stream_limit(self.stream, 0, 4)
        with torch.get_device_module(self.device).stream(self.stream):
            while self.running:
                try:
                    # 阻塞等待任务，避免空转烧 CPU
                    # timeout 允许我们定期检查 self.running 状态
                    task_args = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                assert (not self.finished_event.is_set()), f"should not be finished"
                try:
                    args, kwargs = task_args
                    result = self.taks_func(*args, **kwargs)

                    try:
                        self.result_queue.put_nowait(result)
                    except queue.Full:
                        logger.error(f"[{self.name}] Result queue stuck! Something unexpected happened.")
                except Exception as e:
                    logger.warning(f"[Worker] Task {self.name} failed: {e}")
                    try:
                        self.result_queue.put_nowait(None)
                    except:
                        pass
                finally:
                    # 标记任务完成
                    self.finished_event.set()
                    self.task_queue.task_done()

    def submit_task(self, *args, **kwargs) -> bool:
        try:            
            self.task_queue.put_nowait((args, kwargs))
            return True
            
        except queue.Full:
            logger.warning(f"[{self.name}] Task queue full")
            return False
        
    def get_result(self, block: bool = False, timeout = 10.0):
        """
        获取任务结果
        
        Args:
            block: 是否阻塞等待
            timeout: 超时时间
        """
        try:
            ret = self.result_queue.get(block=block, timeout=timeout)
            self.finished_event.clear()
            return ret
        except queue.Empty:
            return None

    def is_finished(self):
        return self.finished_event.is_set()

    def stop(self):
        self.running = False
        self.thread.join()

class EPLBManager:
    def __init__(self, model_runner: "ModelRunner"):
        super().__init__()
        self._model_runner = model_runner
        self.ep_rank = model_runner.global_rank
        self.ep_size = model_runner.world_size
        self.host_group = torch.distributed.new_group(
            ranks=list(range(self.ep_size)),
            backend="gloo",
        )

        self._server_args = model_runner.server_args
        self._rebalance_layers_per_chunk = (
            self._server_args.eplb_rebalance_layers_per_chunk
        )
        self._rebalance_num_iterations = self._server_args.eplb_rebalance_num_iterations

        # Otherwise, the circular buffer will contain stale data. If the case is needed, it can be implemented.
        assert (
            self._server_args.eplb_rebalance_num_iterations
            >= self._server_args.expert_distribution_recorder_buffer_size
        ), "eplb_rebalance_num_iterations must be greater than expert_distribution_recorder_buffer_size"

        if not get_global_expert_distribution_recorder().recording:
            get_global_expert_distribution_recorder().start_record()

        logger.info(
            f"[EPLBManager] system started, will rebalance per {self._rebalance_num_iterations} iterations."
        )
        self.metrics_collector = EPLBMetricsCollector(
            labels={
                'model_name': self._server_args.served_model_name,
                'app_key': self._server_args.app_key,
            },
            metrics_reporters=self._server_args.metrics_reporters,
        )

        self._main_generator = self._entrypoint()
        self.dump_total_times = []
        self.compute_total_times = []
        self.transfer_total_times = []
        self.rebalance_total_times = []
        self._step_counter = 0
        self._extra_steps_to_sync = self._server_args.num_continuous_decode_steps * 200 + 1
        self.vote_buffer = torch.zeros(1, dtype=torch.int32, device="cpu")

        self.expert_metadata_computing_thread = BackgroundWorker(
            "ExpertMetadataComputing",
            self._model_runner.gpu_id,
            self.compute_new_expert_metadata,
            self._server_args.device,
            self._model_runner.global_rank,
        )
        self.sync_counter_thread = BackgroundWorker(
            "SyncCounter",
            self._model_runner.gpu_id,
            self.get_gloabl_fastest_steps,
            self._server_args.device,
            self._model_runner.global_rank,
        )

    def on_forward_pass_end(self):
        next(self._main_generator)
        self._step_counter += 1

    # can be more complex if needed
    def _entrypoint(self):
        while True:
            for _ in range(self._rebalance_num_iterations):
                yield

            yield from self.rebalance()

    def rebalance(self):
        logger.info("[EPLBManager] rebalance start")

        enable_timing = self._rebalance_layers_per_chunk is None

        if enable_timing:
            device_synchronize()
            time_start = time.time()

        dump_start_time = time.time()
        dumped_data = get_global_expert_distribution_recorder().dump_record(
            output_mode="object"
        )
        per_expert_metrics = dumped_data['per_expert_metrics']
        per_device_metrics = dumped_data['per_device_metrics']
        self.metrics_collector.log_expert_unbalancedness(per_expert_metrics)
        self.metrics_collector.log_gpu_unbalancedness(per_device_metrics)

        # GiniCoefficient
        logical_count = dumped_data["logical_count"].cpu()
        dump_time = time.time() - dump_start_time
        self.dump_total_times.append(dump_time)
        
        yield from self.async_compute_and_transfer(logical_count)

        msg = f"[EPLBManager] rebalance end"
        if enable_timing:
            device_synchronize()
            time_end = time.time()
            rebalance_time = time_end - time_start
            self.rebalance_total_times.append(rebalance_time)
            msg += f" time={rebalance_time:.3f}s avg time={np.mean(self.rebalance_total_times):.3f}s"
        logger.info(msg)

    def compute_new_expert_metadata(self, logical_count):
        compute_start_time = time.time()
        expert_location_metadata = ExpertLocationMetadata.init_by_eplb(
            self._server_args, self._model_runner.model_config, logical_count
        )
        compute_time = time.time() - compute_start_time
        self.compute_total_times.append(compute_time)
        logger.info(f"[EPLBManager] compute end time={compute_time:.3f}s avg time ={np.mean(self.compute_total_times):.3f}s")
        return expert_location_metadata
    
    def get_gloabl_fastest_steps(self):
        self.vote_buffer.fill_(self._step_counter)
        torch.distributed.all_reduce(
            self.vote_buffer, 
            op=torch.distributed.ReduceOp.MAX, 
            group=self.host_group
        )
        return self.vote_buffer.item() + self._extra_steps_to_sync

    def fake_rebalance(self, *args, **kwargs):
        if False:
            yield
        return

    def async_compute_and_transfer(self, logical_count):
        logger.info("[EPLBManager] async rebalance start")
        # 1. trigger an asynchronusly computing task new expert metadata in backgournd thread
        self.expert_metadata_computing_thread.submit_task(logical_count)
        # 2. yield before computing finished
        while not self.expert_metadata_computing_thread.is_finished():
            yield
        expert_location_metadata = self.expert_metadata_computing_thread.get_result(block=True)
        # 3. make sure all ranks reached same timepoint 
        self.sync_counter_thread.submit_task()
        logger.debug(f"[EPLBManager] start syncing rebalance timing")
        while not self.sync_counter_thread.is_finished():
            yield
        start_transfering_step = self.sync_counter_thread.get_result()
        while self._step_counter < start_transfering_step:
            yield
        logger.info(f"[EPLBManager] barrier at {self._step_counter}")
        torch.distributed.barrier(group=self.host_group)
        # 4. start transferring
        # 4.1    for each layer
        # 4.2    trigger weight2buffer, plan, and p2p comm in background thread
        # 4.3    yield before finish
        # 4.4    torch.allreduce to block scheduler thread
        # 4.5    buffer2weight by a blocking h2d
        device_synchronize()
        update_layer_ids_chunks = self._compute_update_layer_ids_chunks()
        transfer_start_time = time.time()
        for chunk_index, update_layer_ids in enumerate(update_layer_ids_chunks):
            if len(update_layer_ids_chunks) > 1:
                yield
            self._model_runner.update_expert_location(
                expert_location_metadata,
                update_layer_ids=update_layer_ids,
            )
        device_synchronize()
        transfer_time = time.time() - transfer_start_time
        self.transfer_total_times.append(transfer_time)
        logger.info(f"[EPLBManager] transfer end time={transfer_time:.3f}s avg time ={np.mean(self.transfer_total_times):.3f}s")

    def compute_and_transfer(self, logical_count):
        expert_location_metadata = self.compute_new_expert_metadata(logical_count)
        
        update_layer_ids_chunks = self._compute_update_layer_ids_chunks()
        transfer_start_time = time.time()
        for chunk_index, update_layer_ids in enumerate(update_layer_ids_chunks):
            if len(update_layer_ids_chunks) > 1:
                yield
            self._model_runner.update_expert_location(
                expert_location_metadata,
                update_layer_ids=update_layer_ids,
            )
        transfer_time = time.time() - transfer_start_time
        self.transfer_total_times.append(transfer_time)
        logger.info(f"[EPLBManager] transfer end time={transfer_time:.3f}s avg time ={np.mean(self.transfer_total_times):.3f}s")

    def _compute_update_layer_ids_chunks(self) -> List[List[int]]:
        all_layer_ids = sorted(
            list(self._model_runner.model.routed_experts_weights_of_layer.keys())
        )
        chunk_size = self._rebalance_layers_per_chunk or 1000000
        return list(_chunk_list(all_layer_ids, chunk_size=chunk_size))


def _chunk_list(items: List, chunk_size):
    for start_index in range(0, len(items), chunk_size):
        yield items[start_index : start_index + chunk_size]