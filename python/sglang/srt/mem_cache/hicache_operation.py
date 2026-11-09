from __future__ import annotations

import threading
from dataclasses import dataclass
from queue import Queue
from typing import TYPE_CHECKING

import torch

from sglang.srt.utils import get_colorful_logger, get_device_module

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import BaseTokenToKVPool
    from sglang.srt.mem_cache.memory_pool.host_memory_pool import HostTokenToKVPool

logger = get_colorful_logger(__name__)

device_module = get_device_module()


class LayerLoadingEvent:
    def __init__(self, num_layers: int):
        self._num_layers = num_layers
        self.load_events = [device_module.Event() for _ in range(num_layers)]
        self.start_event = device_module.Event()

    def complete(self, layer_index: int):
        assert 0 <= layer_index < self._num_layers
        self.load_events[layer_index].record()

    def wait(self, layer_index: int):
        device_module.current_stream().wait_event(self.load_events[layer_index])

    @property
    def finish_event(self):
        return self.load_events[-1]


class LayerDoneCounter:
    def __init__(self, num_layers: int, num_counters: int):
        self.num_counters = num_counters
        self.events = [LayerLoadingEvent(num_layers) for _ in range(self.num_counters)]
        self.producer_index = -1
        self.consumer_index = -1

    def update_producer(self):
        self.producer_index = (self.producer_index + 1) % self.num_counters
        assert self.events[self.producer_index].finish_event.query(), "Producer finish event should be ready before being reused."
        return self.producer_index

    def set_consumer(self, index: int):
        self.consumer_index = index

    def wait_until(self, threshold: int):
        if self.consumer_index < 0:
            return
        # if get_attn_tp_group().rank_in_group == 0:
        #     logger.info(f"wait until {threshold} {self.consumer_index} {self.events[self.consumer_index].load_events[threshold].__hash__()}")
        self.events[self.consumer_index].wait(threshold)


@dataclass
class L2PoolEntry:
    device_kv_pool: BaseTokenToKVPool
    host_kv_pool: HostTokenToKVPool
    layer_done_counter: LayerDoneCounter
    layer_num: int
    layer_offset: int = 0
    # True 表示该 entry 的 pool 不随 target 做 l1.5 分片 (如 DFLASH 的 GQA draft),
    # 须用未经过滤/重映射的全局页号寻址。
    use_global_pages: bool = False


class D2HWrite:
    def __init__(
        self,
        host_token_indices: torch.Tensor,
        device_token_indices: torch.Tensor,
        node,
        global_host_indices: torch.Tensor | None = None,
        global_device_indices: torch.Tensor | None = None,
    ):
        self.host_token_indices = host_token_indices
        self.device_token_indices = device_token_indices
        # 未经 l1.5 过滤/重映射的全局页号: target 的 pool 按 attn_tp 分片而异构 draft
        # (DFLASH GQA) 的 pool 不分片, 后者须用全局页号寻址。
        self.global_host_indices = global_host_indices
        self.global_device_indices = global_device_indices
        self.nodes = [node] if node is not None else []
        self.finish_event: device_module.Event | None = None

    @staticmethod
    def merge(writes: list[D2HWrite]) -> D2HWrite:
        assert len(writes) > 0
        if len(writes) == 1:
            return writes[0]

        host_token_indices = torch.cat([w.host_token_indices for w in writes])
        device_token_indices = torch.cat([w.device_token_indices for w in writes])
        has_global = all(w.global_host_indices is not None for w in writes)
        global_host = torch.cat([w.global_host_indices for w in writes]) if has_global else None
        global_device = torch.cat([w.global_device_indices for w in writes]) if has_global else None
        nodes = []
        for w in writes:
            nodes.extend(w.nodes)

        merged = D2HWrite(
            host_token_indices,
            device_token_indices,
            node=None,
            global_host_indices=global_host,
            global_device_indices=global_device,
        )
        merged.nodes = nodes
        return merged


class H2DLoad:
    def __init__(
        self,
        host_token_indices: torch.Tensor,
        device_token_indices: torch.Tensor,
        node,
        is_page: bool = False,
        global_host_indices: torch.Tensor | None = None,
        global_device_indices: torch.Tensor | None = None,
    ):
        self.host_token_indices = host_token_indices
        self.device_token_indices = device_token_indices
        # 同 D2HWrite: 异构 draft 用未过滤的全局页号。
        self.global_host_indices = global_host_indices
        self.global_device_indices = global_device_indices
        self.nodes = [node] if node is not None else []
        self.finish_event: device_module.Event | None = None
        self.timing_start_event: device_module.Event | None = None
        self.timing_finish_event: device_module.Event | None = None
        self._is_page = is_page
        self.page_size = 0

    @property
    def kv_tokens(self) -> int:
        base = self.host_token_indices
        if base is None:
            return 0
        n = int(base.numel())
        if self._is_page and self.page_size > 0:
            n *= self.page_size
        return n

    @staticmethod
    def merge(loads: list[H2DLoad]) -> H2DLoad:
        assert len(loads) > 0
        if len(loads) == 1:
            return loads[0]

        host_token_indices = torch.cat([load.host_token_indices for load in loads])
        device_token_indices = torch.cat([load.device_token_indices for load in loads])
        has_global = all(load.global_host_indices is not None for load in loads)
        global_host = torch.cat([load.global_host_indices for load in loads]) if has_global else None
        global_device = torch.cat([load.global_device_indices for load in loads]) if has_global else None
        nodes = []
        for load in loads:
            nodes.extend(load.nodes)

        merged = H2DLoad(
            host_token_indices,
            device_token_indices,
            node=None,
            is_page=loads[0]._is_page,
            global_host_indices=global_host,
            global_device_indices=global_device,
        )
        merged.page_size = loads[0].page_size
        merged.nodes = nodes
        return merged


class H2SWrite:
    def __init__(
        self,
        host_token_indices: torch.Tensor,
        hash_value: list[str],
        prefix_keys: list[str] | None,
        node,
        pp_prefix: str | None = None,
        layer_range: tuple | None = None,
    ):
        self.host_token_indices = host_token_indices
        self.hash_value = hash_value
        self.prefix_keys = prefix_keys
        self.node = node
        # Cross-PP write metadata: pp_prefix overrides the store's instance prefix
        # (decode writes pp{i}of{N}_ per i); layer_range slices to that rank's layers.
        self.pp_prefix = pp_prefix
        self.layer_range = layer_range
        self.completed_tokens = 0

        self.pending_pages = 0
        self.pending_lock = threading.Lock()
        self.ack_queue: Queue | None = None

    def setup_async_ack(self, ack_queue: Queue, total_pages: int):
        self.ack_queue = ack_queue
        self.pending_pages = total_pages

    def page_write_done(self):
        if self.ack_queue is None:
            return
        with self.pending_lock:
            self.pending_pages -= 1
            if self.pending_pages <= 0:
                self.ack_queue.put(self)

    def __lt__(self, other: H2SWrite):
        return id(self) < id(other)


class PrefetchStatus:
    SoftTimeout = -2
    QueryMiss = -1
    Prepare = 0
    WaitAllocate = 1
    Loading = 2
    IODone = 3
    Loaded = 4


class S2HLoad:
    def __init__(
        self,
        request_id: str,
        host_token_indices: torch.Tensor,
        token_ids: list[int],
        prefix_keys: list[str] | None,
        last_host_node,
        host_page_indices: torch.Tensor,
        last_hash,
    ):
        self.request_id = request_id
        self.host_token_indices = host_token_indices
        self.token_ids = token_ids
        self.prefix_keys = prefix_keys
        self.last_host_node = last_host_node
        self.host_page_indices = host_page_indices
        self.completed_tokens = 0
        self.processed_global_pages = 0
        # terminate is -1
        self.synced_completed_tokens: int | None = None
        self.hash_value: list[str] = []
        self.last_hash = last_hash

        self._lock = threading.Lock()
        self.page_mask: list[int] | None = None
        self.fetch_status = PrefetchStatus.Prepare
        self.page2rank = None
        self.hit_pages = None
        self.is_terminate = False
        self._query_result_below_threshold = False
        self._is_detached = False
        self.prefetch_deadline: float | None = None
        self.reserved_tokens = 0

        self.t_query_enq: float = 0.0
        self.t_query_start: float = 0.0
        self.t_query_storage_done: float = 0.0
        self.t_query_done: float = 0.0
        self.t_read_enq: float = 0.0
        self.t_read_start: float = 0.0
        self.t_io_done: float = 0.0
        self.t_finish_done: float = 0.0
        self._pf_query_recorded: bool = False
        self._pf_queue_recorded: bool = False

    def get_prefetch_status(self):
        with self._lock:
            return self.fetch_status

    def set_prefetch_deadline(self, deadline: float):
        with self._lock:
            self.prefetch_deadline = deadline

    def set_reserved_tokens(self, reserved_tokens: int):
        with self._lock:
            self.reserved_tokens = reserved_tokens

    def get_prefetch_status_for_sync(self, now: float):
        with self._lock:
            if not self._is_detached:
                if self.prefetch_deadline is not None and now >= self.prefetch_deadline:
                    return PrefetchStatus.SoftTimeout
                if self.is_terminate:
                    if self._query_result_below_threshold:
                        return PrefetchStatus.QueryMiss
                    return PrefetchStatus.SoftTimeout
            return self.fetch_status

    def get_synced_completed_tokens(self) -> int | None:
        with self._lock:
            return self.synced_completed_tokens

    def set_synced_completed_tokens(self, value: int):
        with self._lock:
            self.synced_completed_tokens = value
            self.fetch_status = PrefetchStatus.Loaded

    def set_io_done(self):
        with self._lock:
            self.fetch_status = PrefetchStatus.IODone

    def increment(self, num_tokens: int) -> bool:
        with self._lock:
            if self.is_terminate:
                return False
            self.completed_tokens += num_tokens
            return True

    def mark_terminate(self):
        with self._lock:
            self.is_terminate = True

    def mark_detached(self):
        with self._lock:
            self.is_terminate = True
            self._is_detached = True
            if self.fetch_status < PrefetchStatus.Loading:
                self.fetch_status = PrefetchStatus.IODone

    def is_terminated(self) -> bool:
        with self._lock:
            return self.is_terminate

    def is_detached(self) -> bool:
        with self._lock:
            return self._is_detached

    def set_query_result(
        self,
        hash_values,
        page_mask,
        page2rank,
        hit_pages,
        should_terminate: bool = False,
    ):
        with self._lock:
            if self._is_detached:
                return False
            self.hash_value = hash_values
            self.page_mask = page_mask
            self.page2rank = page2rank
            self.hit_pages = hit_pages
            self._query_result_below_threshold = should_terminate
            self.is_terminate = self.is_terminate or should_terminate
            self.fetch_status = PrefetchStatus.WaitAllocate
            return True

    def set_allocate_result(self, hit_pages, hit_tokens, allocated_pages, allocated_token_indices):
        with self._lock:
            self.hit_pages = hit_pages
            self.token_ids = self.token_ids[:hit_tokens]
            self.hash_value = self.hash_value[:hit_pages]
            self.page_mask = self.page_mask[:hit_pages]
            self.page2rank = self.page2rank[:hit_pages]
            self.host_page_indices = allocated_pages
            self.host_token_indices = allocated_token_indices
            self.fetch_status = PrefetchStatus.Loading

    def __lt__(self, other: S2HLoad):
        return id(self) < id(other)
