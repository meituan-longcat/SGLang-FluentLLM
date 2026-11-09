from __future__ import annotations

import dataclasses
import os
import time
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
)

import zmq

from sglang.srt.disaggregation.kv_events import (
    EventPublisherFactory,
    KVEventBatch,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import RadixCache
    from sglang.srt.observability.metrics.collector import SchedulerStats


@dataclasses.dataclass
class KvMetrics:
    request_active_slots: int = 0
    request_total_slots: int = 0
    kv_active_blocks: int = 0
    kv_total_blocks: int = 0
    num_requests_waiting: int = 0
    gpu_cache_usage_perc: float = 0.0
    data_parallel_rank: int = 0
    has_retracted: bool = False


@dataclass(kw_only=True, slots=True)
class SchedulerKvEventsPublisher:
    kv_events_config: str | None
    attn_tp_rank: int
    attn_dp_rank: int
    pp_rank: int
    dp_rank: int | None
    tree_cache: RadixCache
    send_metrics_from_scheduler: zmq.Socket | None
    max_running_requests: int
    max_total_num_tokens: int
    enable_kv_cache_events: bool = False
    kv_event_publisher: Any = None

    def __post_init__(self) -> None:
        self.init_kv_events(self.kv_events_config)

    def init_kv_events(self, kv_events_config: str | None):
        self.enable_kv_cache_events = bool(kv_events_config and self.attn_tp_rank == 0 and self.pp_rank == 0)

        if self.enable_kv_cache_events:
            self.kv_event_publisher = EventPublisherFactory.create(kv_events_config, self.attn_dp_rank)

    def emit_kv_metrics(self, stat: SchedulerStats):
        if self.send_metrics_from_scheduler is None:
            return

        if self.pp_rank != 0 or self.attn_tp_rank != 0:
            return

        kv_metrics = KvMetrics()
        kv_metrics.request_active_slots = stat.num_running_reqs
        kv_metrics.request_total_slots = self.max_running_requests
        kv_metrics.kv_active_blocks = int(stat.token_usage * self.max_total_num_tokens)
        kv_metrics.kv_total_blocks = self.max_total_num_tokens
        kv_metrics.num_requests_waiting = stat.num_queue_reqs
        kv_metrics.gpu_cache_usage_perc = stat.token_usage
        kv_metrics.data_parallel_rank = self.dp_rank if self.dp_rank is not None else 0
        mock_retract_file = os.environ.get("MOCK_RETRACT_FILE")
        if mock_retract_file:
            kv_metrics.has_retracted = os.path.exists(mock_retract_file)
        else:
            kv_metrics.has_retracted = stat.num_retracted_reqs > 0

        if not self.send_metrics_from_scheduler.closed:
            self.send_metrics_from_scheduler.send_pyobj(kv_metrics)

    def publish_kv_events(self):
        if self.pp_rank != 0 or self.attn_tp_rank != 0:
            return

        if not self.enable_kv_cache_events or not self.kv_event_publisher:
            return

        events = self.tree_cache.take_events()
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)
