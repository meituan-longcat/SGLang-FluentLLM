from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool.host_kv_allocator import HostKVAllocator
    from sglang.srt.mem_cache.memory_pool.kv_allocator import KVAllocator

from sglang.srt.mem_cache.hicache_operation import (
    D2HWrite,
    H2DLoad,
    L2PoolEntry,
)
from sglang.srt.utils import get_colorful_logger, get_device_module, is_npu

logger = get_colorful_logger(__name__)

device_module = get_device_module()


class L2CacheController:
    def __init__(
        self,
        device_kv_allocator: KVAllocator,
        pool_entries: list[L2PoolEntry],
        page_size: int,
        write_policy: str = "write_through_selective",
        io_backend: str = "",
    ):
        self.device_kv_allocator = device_kv_allocator
        self.pool_entries = pool_entries
        self.host_kv_pool = pool_entries[0].host_kv_pool
        self.device_kv_pool = pool_entries[0].device_kv_pool
        self.host_kv_allocator: HostKVAllocator = self.host_kv_pool.host_kv_allocator
        self.write_policy = write_policy
        self.page_size = page_size
        self.io_backend = io_backend

        self.device = pool_entries[0].device_kv_pool.device
        self.layer_num = pool_entries[0].layer_num
        self.layer_done_counter = pool_entries[0].layer_done_counter
        for entry in pool_entries:
            entry.device_kv_pool.register_layer_transfer_counter(entry.layer_done_counter)

        if write_policy not in [
            "write_through",
            "write_through_selective",
            "write_back",
        ]:
            raise ValueError(f"Invalid write policy: {write_policy}")

        self.d2h_write_q_todo: list[D2HWrite] = []
        self.d2h_write_q_ongoing: list[D2HWrite] = []
        self.h2d_load_q_todo: list[H2DLoad] = []
        self.h2d_load_q_ongoing: list[H2DLoad] = []

        self.stop_event = threading.Event()

        # Bulk D2H descriptor processing must not occupy the shared NPU
        # operator launch queue. Submit on the scheduler thread; device copy
        # and the existing producer/completion events remain asynchronous.
        from sglang.srt.env import global_server_args_dict

        self.write_stream = (
            device_module.SyncLaunchStream()
            if is_npu() and global_server_args_dict.get("disaggregation_mode") == "prefill"
            else device_module.Stream()
        )
        self.load_stream = device_module.Stream()

    def stop(self):
        self.stop_event.set()

    def join_and_restart(self):
        self.d2h_write_q_todo.clear()
        self.d2h_write_q_ongoing.clear()
        self.h2d_load_q_todo.clear()
        self.h2d_load_q_ongoing.clear()

        self.stop_event.clear()

    def reset(self):
        self.stop()
        self.join_and_restart()

    def pages_to_local_token_indices(
        self,
        device_page_indices: torch.Tensor,
        host_page_indices: torch.Tensor,
    ):
        device_page_indices, host_page_indices = self.pages_to_local_page_indices(device_page_indices, host_page_indices)
        device_token_indices = self.device_kv_pool.pages_to_tokens(device_page_indices)
        host_token_indices = self.host_kv_pool.pages_to_tokens(host_page_indices)
        return device_token_indices, host_token_indices

    def pages_to_local_page_indices(
        self,
        device_page_indices: torch.Tensor,
        host_page_indices: torch.Tensor,
    ):
        if hasattr(self.device_kv_pool, "global_pages_to_local_page_indices"):
            local_device_page_indices, _ = self.device_kv_pool.global_pages_to_local_page_indices(device_page_indices)
        else:
            local_device_page_indices = device_page_indices

        if hasattr(self.host_kv_pool, "global_pages_to_local_page_indices"):
            local_host_page_indices, _ = self.host_kv_pool.global_pages_to_local_page_indices(host_page_indices)
        else:
            local_host_page_indices = host_page_indices

        assert len(host_page_indices) == len(device_page_indices), f"{len(host_page_indices)=} != {len(device_page_indices)=}"
        return local_device_page_indices, local_host_page_indices

    def write(
        self,
        device_page_indices: torch.Tensor,
        host_page_indices: torch.Tensor,
        node,
    ) -> torch.Tensor | None:
        if not is_npu():
            device_token_indices, host_token_indices = self.pages_to_local_token_indices(device_page_indices, host_page_indices)
            self.d2h_write_q_todo.append(D2HWrite(host_token_indices, device_token_indices, node))
        else:
            # 保留未过滤的全局页号供异构 draft (不分片, 全局寻址) 使用。
            global_host_pages = host_page_indices
            global_device_pages = device_page_indices
            device_page_indices, host_page_indices = self.pages_to_local_page_indices(device_page_indices, host_page_indices)
            self.d2h_write_q_todo.append(
                D2HWrite(
                    host_page_indices,
                    device_page_indices.to("cpu"),
                    node,
                    global_host_indices=global_host_pages,
                    global_device_indices=global_device_pages.to("cpu"),
                )
            )
        self.launch_d2h()
        num_pages = len(device_page_indices)
        logger.debug(
            f"[D2H-Enqueue] node_id={node.id} pages={num_pages} tokens={num_pages * self.page_size} q_todo_size={len(self.d2h_write_q_todo)}"
        )

    def launch_d2h(self) -> None:
        with torch.profiler.record_function("launch_d2h"):
            if not self.d2h_write_q_todo:
                return
            d2h_write = D2HWrite.merge(self.d2h_write_q_todo)
            host_token_indices, device_token_indices = self.move_indices(d2h_write)
            self.d2h_write_q_todo.clear()
            start_event = device_module.Event()
            finish_event = device_module.Event()

            start_event.record()
            with device_module.stream(self.write_stream):
                start_event.wait(self.write_stream)
                for entry in self.pool_entries:
                    # 用 entry 自己的 host pool 与索引域: 异构 draft (DFLASH GQA) 有独立
                    # host pool 且不随 target 分片, 须用未过滤的全局页号。
                    if entry.use_global_pages and d2h_write.global_host_indices is not None:
                        h_idx = d2h_write.global_host_indices
                        d_idx = d2h_write.global_device_indices
                    else:
                        h_idx, d_idx = host_token_indices, device_token_indices
                    entry.host_kv_pool.d2h_with_offset(
                        h_idx.to(torch.int64),
                        d_idx.to(torch.int64),
                        entry.device_kv_pool,
                        entry.layer_offset,
                        entry.layer_num,
                        self.io_backend,
                    )
                finish_event.record()
                if host_token_indices.device.type != "cpu":
                    host_token_indices.record_stream(self.write_stream)
                if device_token_indices.device.type != "cpu":
                    device_token_indices.record_stream(self.write_stream)

            d2h_write.start_event = start_event
            d2h_write.finish_event = finish_event
            self.d2h_write_q_ongoing.append(d2h_write)

    def load(
        self,
        host_page_indices: torch.Tensor,
        device_page_indices: torch.Tensor,
        node,
    ) -> torch.Tensor | None:
        num_pages = len(host_page_indices)
        if not is_npu():
            device_token_indices, host_token_indices = self.pages_to_local_token_indices(device_page_indices, host_page_indices)
            self.h2d_load_q_todo.append(H2DLoad(host_token_indices, device_token_indices, node, is_page=False))
        else:
            # 保留未过滤的全局页号供异构 draft (不分片, 全局寻址) 使用。
            global_host_pages = host_page_indices
            global_device_pages = device_page_indices
            device_page_indices, host_page_indices = self.pages_to_local_page_indices(device_page_indices, host_page_indices)
            h2d = H2DLoad(
                host_page_indices,
                device_page_indices.to("cpu"),
                node,
                is_page=True,
                global_host_indices=global_host_pages,
                global_device_indices=global_device_pages.to("cpu"),
            )
            h2d.page_size = self.page_size
            self.h2d_load_q_todo.append(h2d)
        logger.debug(f"[H2D-Enqueue] node_id={node.id} pages={num_pages} tokens={num_pages * self.page_size} q_todo_size={len(self.h2d_load_q_todo)}")
        return device_page_indices

    def move_indices(self, op):
        host_token_indices, device_token_indices = op.host_token_indices, op.device_token_indices
        return self.host_kv_pool.move_indices(host_token_indices, device_token_indices)

    def launch_h2d(self, old_consumer_idx=-1) -> int:
        if len(self.h2d_load_q_todo) == 0:
            return -1

        with torch.profiler.record_function("launch_h2d"):
            target_entry = self.pool_entries[0]
            producer_id = target_entry.layer_done_counter.update_producer() if old_consumer_idx == -1 else old_consumer_idx
            target_event = target_entry.layer_done_counter.events[producer_id]

            draft_producer_ids = []
            draft_events = []
            for entry in self.pool_entries[1:]:
                draft_producer_id = entry.layer_done_counter.update_producer() if old_consumer_idx == -1 else old_consumer_idx
                draft_producer_ids.append(draft_producer_id)
                draft_events.append(entry.layer_done_counter.events[draft_producer_id])
                assert draft_producer_id == producer_id, "draft producer id mismatch"

            h2d_load = H2DLoad.merge(self.h2d_load_q_todo)
            num_nodes = len(h2d_load.nodes)
            total_tokens = num_nodes * self.page_size
            self.h2d_load_q_todo.clear()

            host_token_indices, device_token_indices = self.move_indices(h2d_load)
            logger.debug(f"[H2D-Launch] nodes={num_nodes} tokens={total_tokens}")
            target_event.start_event.record()

            timing_start = device_module.Event(enable_timing=True)
            timing_finish = device_module.Event(enable_timing=True)
            timing_start.record()

            with device_module.stream(self.load_stream):
                target_event.start_event.wait(self.load_stream)
                timing_start.wait(self.load_stream)
                host_idx = host_token_indices.to(torch.int64)
                dev_idx = device_token_indices.to(torch.int64)

                for local_layer in range(target_entry.layer_num):
                    if host_idx.numel() != 0:
                        target_entry.host_kv_pool.h2d_layerwise_with_offset(
                            host_idx,
                            dev_idx,
                            target_entry.device_kv_pool,
                            target_entry.layer_offset + local_layer,
                            local_layer,
                            self.io_backend,
                        )
                    target_event.complete(local_layer)
                    # if get_attn_tp_group().rank_in_group == 0:
                    #     logger.info(f"launch h2d layer {local_layer} done {target_event.load_events[local_layer].__hash__()}")

                for j, entry in enumerate(self.pool_entries[1:]):
                    # 用 entry 自己的 host pool 与索引域: 异构 draft (DFLASH GQA) 有独立
                    # host pool 且不随 target 分片, 须用未过滤的全局页号。
                    if entry.use_global_pages and h2d_load.global_host_indices is not None:
                        j_host_idx = h2d_load.global_host_indices.to(torch.int64)
                        j_dev_idx = h2d_load.global_device_indices.to(torch.int64)
                    else:
                        j_host_idx, j_dev_idx = host_idx, dev_idx
                    for local_layer in range(entry.layer_num):
                        if j_host_idx.numel() != 0:
                            entry.host_kv_pool.h2d_layerwise_with_offset(
                                j_host_idx,
                                j_dev_idx,
                                entry.device_kv_pool,
                                entry.layer_offset + local_layer,
                                local_layer,
                                self.io_backend,
                            )
                        draft_events[j].complete(local_layer)
                        # if get_attn_tp_group().rank_in_group==0:
                        #     logger.info(f"launch h2d draft layer {local_layer} done {draft_events[j].load_events[local_layer].__hash__()}")

                finish_event = device_module.Event()
                finish_event.record()
                timing_finish.record()

                if host_token_indices.device.type != "cpu":
                    host_token_indices.record_stream(self.load_stream)
                if device_token_indices.device.type != "cpu":
                    device_token_indices.record_stream(self.load_stream)

            h2d_load.start_event = target_event.start_event
            h2d_load.finish_event = finish_event
            h2d_load.timing_start_event = timing_start
            h2d_load.timing_finish_event = timing_finish
            self.h2d_load_q_ongoing.append(h2d_load)
        return producer_id
