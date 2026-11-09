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
"""Utilities for Prometheus Metrics Collection."""

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union

from sglang.srt.observability.metrics.utils import exponential_buckets
from sglang.srt.utils import get_bool_env_var, get_colorful_logger
from sglang.utils import get_cat_reporter

SGLANG_TEST_REQUEST_TIME_STATS = get_bool_env_var("SGLANG_TEST_REQUEST_TIME_STATS")
logger = get_colorful_logger(__name__)


class CacheHitMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_tokens = 0
        self.hit_tokens = 0
        self.l1_hit_tokens = 0
        self.l2_hit_tokens = 0
        self.l3_hit_tokens = 0

    def get(self):
        return self.total_tokens, self.hit_tokens, self.l1_hit_tokens, self.l2_hit_tokens, self.l3_hit_tokens

    def get_and_reset(self):
        total_tokens, hit_tokens, l1_hit_tokens, l2_hit_tokens, l3_hit_tokens = (
            self.total_tokens,
            self.hit_tokens,
            self.l1_hit_tokens,
            self.l2_hit_tokens,
            self.l3_hit_tokens,
        )
        self.reset()
        return total_tokens, hit_tokens, l1_hit_tokens, l2_hit_tokens, l3_hit_tokens

    def add(self, total_tokens, hit_tokens, l1_hit_tokens, l2_hit_tokens, l3_hit_tokens):
        self.total_tokens += total_tokens
        self.hit_tokens += hit_tokens
        self.l1_hit_tokens += l1_hit_tokens
        self.l2_hit_tokens += l2_hit_tokens
        self.l3_hit_tokens += l3_hit_tokens


class _AvgBucket:
    __slots__ = ("total_ms", "n")

    def __init__(self):
        self.total_ms = 0.0
        self.n = 0

    def add(self, ms: float):
        self.total_ms += ms
        self.n += 1

    def reset(self):
        self.total_ms = 0.0
        self.n = 0

    def avg(self):
        return (self.total_ms / self.n) if self.n > 0 else None


class SchedulerLatencyMetrics:
    """Rolling accumulator for L3 prefetch, forward, and L2 load latency.

    Each bucket carries its own (total_ms, n) so avg is computed only over
    requests / batches that actually went through that stage. Reset every
    60s alongside CacheHitMetrics.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.pf_query = _AvgBucket()
        self.pf_queue = _AvgBucket()
        self.pf_finish = _AvgBucket()
        self.chunk_forward = _AvgBucket()
        self.fwd_per_1kkv = _AvgBucket()
        self.l2_load = _AvgBucket()
        self.l2_load_per_1kkv = _AvgBucket()

    def add_pf_query(self, ms):
        self.pf_query.add(ms)

    def add_pf_queue(self, ms):
        self.pf_queue.add(ms)

    def add_pf_finish(self, ms):
        self.pf_finish.add(ms)

    def add_chunk_forward(self, ms, kv_tokens):
        self.chunk_forward.add(ms)
        if kv_tokens > 0:
            self.fwd_per_1kkv.add(ms * 1000.0 / kv_tokens)

    def add_l2_load(self, ms, kv_tokens):
        self.l2_load.add(ms)
        if kv_tokens > 0:
            self.l2_load_per_1kkv.add(ms * 1000.0 / kv_tokens)


@dataclass
class TimeStats:
    """
    Store the timestamps for each stage of a request.

    Unified: wait_queue -> forward -> completion
    Prefill: bootstrap_queue -> wait_queue -> forward -> transfer_queue -> completion
    Decode: prealloc_queue -> transfer_queue -> wait_queue -> forward -> completion
    """

    lb_entry_time: float = 0.0
    wait_queue_entry_time: float = 0.0
    forward_entry_time: float = 0.0
    completion_time: float = 0.0
    prefill_bootstrap_queue_entry_time: float = 0.0
    prefill_transfer_queue_entry_time: float = 0.0
    decode_prealloc_queue_entry_time: float = 0.0
    decode_transfer_queue_entry_time: float = 0.0

    class RequestType(Enum):
        UNIFIED = "unified"
        PREFILL = "prefill"
        DECODE = "decode"
        INVALID = "invalid"

    def get_queueing_time(self) -> float:
        return self.forward_entry_time - self.wait_queue_entry_time

    def __str__(self) -> str:
        # if unified
        _type = self.get_type()

        if _type == self.RequestType.UNIFIED:
            queue_duration = self.forward_entry_time - self.wait_queue_entry_time
            forward_duration = self.completion_time - self.forward_entry_time

            if SGLANG_TEST_REQUEST_TIME_STATS:
                assert queue_duration >= 0 and forward_duration >= 0, (
                    f"queue_duration={queue_duration} < 0 or forward_duration={forward_duration} < 0"
                )

            return (
                f"queue_duration={self.format_duration(queue_duration)}, "
                f"forward_duration={self.format_duration(forward_duration)}, "
                f"start_time={self.wait_queue_entry_time}"
            )
        elif _type == self.RequestType.PREFILL:
            bootstrap_duration = self.wait_queue_entry_time - self.prefill_bootstrap_queue_entry_time

            queue_duration = self.forward_entry_time - self.wait_queue_entry_time

            forward_duration = self.completion_time - self.forward_entry_time

            if SGLANG_TEST_REQUEST_TIME_STATS:
                assert bootstrap_duration >= 0 and queue_duration >= 0 and forward_duration >= 0, (
                    f"bootstrap_duration={bootstrap_duration} < 0 or queue_duration={queue_duration} < 0 or forward_duration={forward_duration} < 0"
                )
            return (
                f"bootstrap_duration={self.format_duration(bootstrap_duration)}, "
                f"queue_duration={self.format_duration(queue_duration)}, "
                f"forward_duration={self.format_duration(forward_duration)}, "
                f"start_time={self.prefill_bootstrap_queue_entry_time}"
            )
        # if decode
        elif _type == self.RequestType.DECODE:
            prealloc_duration = self.decode_transfer_queue_entry_time - self.decode_prealloc_queue_entry_time

            transfer_duration = self.wait_queue_entry_time - self.decode_transfer_queue_entry_time
            queue_duration = self.forward_entry_time - self.wait_queue_entry_time
            forward_duration = self.completion_time - self.forward_entry_time

            if SGLANG_TEST_REQUEST_TIME_STATS:
                assert prealloc_duration >= 0 and transfer_duration >= 0 and queue_duration >= 0 and forward_duration >= 0, (
                    f"prealloc_duration={prealloc_duration} < 0 or "
                    f"transfer_duration={transfer_duration} < 0 or "
                    f"queue_duration={queue_duration} < 0 or "
                    f"forward_duration={forward_duration} < 0"
                )

            return (
                f"prealloc_duration={self.format_duration(prealloc_duration)}, "
                f"transfer_duration={self.format_duration(transfer_duration)}, "
                f"queue_duration={self.format_duration(queue_duration)}, "
                f"forward_duration={self.format_duration(forward_duration)}, "
                f"start_time={self.decode_prealloc_queue_entry_time}"
            )
        else:
            return "Invalid Time Stats"

    def format_duration(self, duration: float) -> str:
        return f"{duration * 1e3:.2f}ms"

    def get_type(self) -> RequestType:
        """Determine the type of request based on timestamp values."""
        if (
            self.prefill_bootstrap_queue_entry_time == 0.0
            and self.prefill_transfer_queue_entry_time == 0.0
            and self.decode_prealloc_queue_entry_time == 0.0
            and self.decode_transfer_queue_entry_time == 0.0
        ):
            return self.RequestType.UNIFIED
        elif self.prefill_bootstrap_queue_entry_time > 0.0 and self.prefill_transfer_queue_entry_time > 0.0:
            return self.RequestType.PREFILL
        elif self.decode_prealloc_queue_entry_time > 0.0 and self.decode_transfer_queue_entry_time > 0.0 and self.wait_queue_entry_time > 0.0:
            return self.RequestType.DECODE
        else:
            return self.RequestType.INVALID


@dataclass
class SchedulerStats:
    num_running_reqs: int = 0
    running_bs_rate: float = 0.0
    num_used_tokens: int = 0
    num_free_tokens: int = 0
    token_usage: float = 0.0
    gen_throughput: float = 0.0
    num_queue_reqs: int = 0
    spec_accept_length: float = 0.0
    dequeue_time: float = 0.0  # prefill
    # Grammar
    num_grammar_queue_reqs: int = 0
    # PD disaggregation
    num_prefill_prealloc_queue_reqs: int = 0
    num_prefill_inflight_queue_reqs: int = 0
    num_decode_prealloc_queue_reqs: int = 0
    num_decode_transfer_queue_reqs: int = 0
    kv_transfer_speed_gb_s: float = 0.0
    kv_transfer_latency_ms: float = 0.0
    # Tokens admitted since the last report; total == 0 means no new sample.
    decode_cache_hit_tokens: int = 0
    decode_cache_total_tokens: int = 0
    decode_cache_hit_rate: float = 0.0

    # Retract
    total_retracted_reqs: int = 0
    num_retracted_reqs: int = 0
    num_retracted_queue_reqs: int = 0
    num_paused_reqs: int = 0

    # Prefill MFU
    prefill_mfu: float = 0.0
    prefill_tflops_per_sec: float = 0.0

    # Prefill chunk token count (tokens assembled per chunk in Prefill Server)
    prefill_chunk_tokens: int = 0

    # PP prefill loop utilization
    prefill_idle_ratio: float = 0.0  # fraction of loop time with no batch (idle)
    prefill_busy_ratio: float = 0.0  # fraction of loop time with a batch (busy)

    # Decode loop utilization
    decode_idle_ratio: float = 0.0  # fraction of loop time with no batch (idle)
    decode_busy_ratio: float = 0.0  # fraction of loop time with a batch (busy)

    # Device memory occupancy (leak probe, reported every N decode steps)
    mem_alloc_mb: float = 0.0  # torch caching allocator: bytes currently allocated
    mem_reserved_mb: float = 0.0  # torch caching allocator: bytes reserved (pool size)
    mem_phys_used_mb: float = 0.0  # driver-level physical occupancy (total - free)
    mem_non_torch_mb: float = 0.0  # phys_used - reserved (HCCL/workspace/EPLB/etc.)

    cache_hit_metrics: CacheHitMetrics = CacheHitMetrics()
    scheduler_latency_metrics: SchedulerLatencyMetrics = SchedulerLatencyMetrics()


class SchedulerMetricsCollector:
    def __init__(
        self,
        labels: dict[str, str],
        metrics_reporters: list[str],
        enable_multimodal: bool = False,
        language_only: bool = False,
    ) -> None:
        self.enable_prometheus = "prometheus" in metrics_reporters
        self.enable_cat = "cat" in metrics_reporters
        self.enable_multimodal_cat = self.enable_cat and enable_multimodal
        self.enable_epd_multimodal_cat = self.enable_multimodal_cat and language_only
        self.enable_llm_platform = "llm-platform" in metrics_reporters
        self.labels = labels

        if self.enable_prometheus:
            self._init_prometheus(labels)

        if self.enable_cat:
            self.cat_reporter = get_cat_reporter(labels.get("model_name", "DefalutModel"), labels.get("app_key", "DefaultAppKey"))

        if self.enable_llm_platform:
            from longcat.tracker.hooks.sglang import SglangMetricsCollector

            self.llm_platform_reporter = SglangMetricsCollector(process_name=labels.get("process_name", "DefalutProcess"))

        self.last_log_time = time.time()
        self.last_cache_hit_metric_update_time = time.time()

    def _init_prometheus(self, labels: dict[str, str]) -> None:
        # We need to import prometheus_client after setting the env variable `PROMETHEUS_MULTIPROC_DIR`
        from prometheus_client import Counter, Gauge, Histogram

        self.labels = labels

        self.num_running_reqs = Gauge(
            name="sglang:num_running_reqs",
            documentation="The number of running requests.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        self.running_bs_rate = Gauge(
            name="sglang:running_bs_rate",
            documentation="The percentage of running requests relative to the maximum running requests.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        self.num_used_tokens = Gauge(
            name="sglang:num_used_tokens",
            documentation="The number of used tokens.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        self.num_free_tokens = Gauge(
            name="sglang:num_free_tokens",
            documentation="The number of free token slots in KV cache.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        self.token_usage = Gauge(
            name="sglang:token_usage",
            documentation="The token usage.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        self.gen_throughput = Gauge(
            name="sglang:gen_throughput",
            documentation="The generation throughput (token/s).",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        self.num_queue_reqs = Gauge(
            name="sglang:num_queue_reqs",
            documentation="The number of requests in the waiting queue.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        self.cache_hit_rate = Gauge(
            name="sglang:cache_hit_rate",
            documentation="The prefix cache hit rate.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        self.spec_accept_length = Gauge(
            name="sglang:spec_accept_length",
            documentation="The average acceptance length of speculative decoding.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        # PD disaggregation
        self.num_prefill_prealloc_queue_reqs = Gauge(
            name="sglang:num_prefill_prealloc_queue_reqs",
            documentation="The number of requests in the prefill prealloc queue.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.num_prefill_inflight_queue_reqs = Gauge(
            name="sglang:num_prefill_inflight_queue_reqs",
            documentation="The number of requests in the prefill inflight queue.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.num_decode_prealloc_queue_reqs = Gauge(
            name="sglang:num_decode_prealloc_queue_reqs",
            documentation="The number of requests in the decode prealloc queue.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.num_decode_transfer_queue_reqs = Gauge(
            name="sglang:num_decode_transfer_queue_reqs",
            documentation="The number of requests in the decode transfer queue.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.num_bootstrap_failed_reqs = Counter(
            name="sglang:num_bootstrap_failed_reqs_total",
            documentation="The number of bootstrap failed requests.",
            labelnames=labels.keys(),
        )
        self.num_transfer_failed_reqs = Counter(
            name="sglang:num_transfer_failed_reqs_total",
            documentation="The number of transfer failed requests.",
            labelnames=labels.keys(),
        )
        self.kv_transfer_speed_gb_s = Gauge(
            name="sglang:kv_transfer_speed_gb_s",
            documentation="The transfer speed of the KV cache in GB/s.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.kv_transfer_latency_ms = Gauge(
            name="sglang:kv_transfer_latency_ms",
            documentation="The transfer latency of the KV cache in ms.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.decode_cache_hit_rate = Gauge(
            name="sglang:decode_cache_hit_rate",
            documentation="The decode cache hit rate (prefix_len / origin_input_ids).",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        # HiCache L1/L2/L3 metrics
        self.hicache_l1_hit_rate = Gauge(
            name="sglang:hicache_l1_hit_rate",
            documentation="The HiCache L1 (GPU) cache hit rate.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.hicache_l2_hit_rate = Gauge(
            name="sglang:hicache_l2_hit_rate",
            documentation="The HiCache L2 (Host/CPU) cache hit rate.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.hicache_l3_hit_rate = Gauge(
            name="sglang:hicache_l3_hit_rate",
            documentation="The HiCache L3 (Storage) cache hit rate.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        self.pf_query_ms = Gauge(
            name="sglang:pf_query_ms",
            documentation="Avg ms per prefetch spent in L3 storage query stage.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.pf_queue_ms = Gauge(
            name="sglang:pf_queue_ms",
            documentation="Avg ms per prefetch waiting in the prefetch_thread queue.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.pf_finish_ms = Gauge(
            name="sglang:pf_finish_ms",
            documentation="Avg ms per prefetch for rank sync + tree insert (io_done -> finalize).",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.pf_completed_reqs = Gauge(
            name="sglang:pf_completed_reqs",
            documentation="Number of prefetches that fully completed (finalize_prefetch) in the last window.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        # L3 prefetch per-stage latency histograms (seconds). Observed per-request
        # in observe_l3_prefetch_stages at prefetch finalize time; the stages sum
        # to l3_prefetch_e2e_s. Label set intentionally matches the tokenizer-side
        # request metrics (no process_name) so series identity is unchanged.
        pf_labels = {k: labels[k] for k in ("model_name", "app_key") if k in labels}
        self.pf_labels = pf_labels
        _pf_buckets = [
            0.001,
            0.002,
            0.005,
            0.01,
            0.02,
            0.03,
            0.05,
            0.08,
            0.1,
            0.15,
            0.2,
            0.3,
            0.5,
            1.0,
        ]
        self.histogram_pf_query_seconds = Histogram(
            name="sglang:pf_query_seconds",
            documentation="L3 prefetch query stage (t_query_done - t_query_enq), seconds.",
            labelnames=pf_labels.keys(),
            buckets=_pf_buckets,
        )
        self.histogram_pf_query_queue_seconds = Histogram(
            name="sglang:pf_query_queue_seconds",
            documentation="L3 prefetch query queue wait (t_query_start - t_query_enq), seconds.",
            labelnames=pf_labels.keys(),
            buckets=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
        )
        self.histogram_pf_storage_query_seconds = Histogram(
            name="sglang:pf_storage_query_seconds",
            documentation="L3 storage_hit_query duration, seconds.",
            labelnames=pf_labels.keys(),
            buckets=_pf_buckets,
        )
        self.histogram_pf_query_sync_seconds = Histogram(
            name="sglang:pf_query_sync_seconds",
            documentation="L3 query completion synchronization (storage query done -> query done), seconds.",
            labelnames=pf_labels.keys(),
            buckets=_pf_buckets,
        )
        self.histogram_pf_query_to_read_seconds = Histogram(
            name="sglang:pf_query_to_read_seconds",
            documentation="L3 prefetch query->read-enqueue gap (alloc/schedule), seconds.",
            labelnames=pf_labels.keys(),
            buckets=_pf_buckets,
        )
        self.histogram_pf_queue_seconds = Histogram(
            name="sglang:pf_queue_seconds",
            documentation="L3 prefetch queue stage (t_read_start - t_read_enq), seconds.",
            labelnames=pf_labels.keys(),
            buckets=[0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
        )
        self.histogram_pf_read_seconds = Histogram(
            name="sglang:pf_read_seconds",
            documentation="L3 prefetch read stage (t_io_done - t_read_start), seconds.",
            labelnames=pf_labels.keys(),
            buckets=[0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0],
        )
        self.histogram_pf_finish_seconds = Histogram(
            name="sglang:pf_finish_seconds",
            documentation="L3 prefetch finish stage (t_finish_done - t_io_done), seconds.",
            labelnames=pf_labels.keys(),
            buckets=[0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0],
        )

        self.chunk_forward_ms = Gauge(
            name="sglang:chunk_forward_ms",
            documentation="Avg forward time per prefill chunk batch.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.fwd_per_1kkv_ms = Gauge(
            name="sglang:fwd_per_1kkv_ms",
            documentation="Avg forward ms per 1000 KV context tokens per batch.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.l2_load_ms = Gauge(
            name="sglang:l2_load_ms",
            documentation="Avg device-side elapsed_time per H2D (L2->L1) transfer.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.l2_load_per_1kkv_ms = Gauge(
            name="sglang:l2_load_per_1kkv_ms",
            documentation="Avg L2 load ms per 1000 KV tokens transferred.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        # Retract
        self.total_retracted_reqs = Gauge(
            name="sglang:total_retracted_reqs",
            documentation="The total number of retracted requests due to kvcache full.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.num_retracted_reqs = Gauge(
            name="sglang:num_retracted_reqs",
            documentation="The number of retracted requests.",
            labelnames=labels.keys(),
        )
        self.num_retracted_queue_reqs = Gauge(
            name="sglang:num_retracted_queue_reqs",
            documentation="The number of requests in the retracted queue.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.num_paused_reqs = Gauge(
            name="sglang:num_paused_reqs",
            documentation="The number of paused requests by async weight sync.",
            labelnames=labels.keys(),
        )

        # Utilization
        self.utilization = Gauge(
            name="sglang:utilization",
            documentation="The utilization.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.max_running_requests_under_SLO = Gauge(
            name="sglang:max_running_requests_under_SLO",
            documentation="The maximum number of running requests under SLO.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        # Prefill chunk token count
        self.prefill_chunk_tokens = Gauge(
            name="sglang:prefill_chunk_tokens",
            documentation="The number of tokens assembled per chunk in Prefill Server.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        # PP prefill loop utilization
        self.prefill_idle_ratio = Gauge(
            name="sglang:prefill_idle_ratio",
            documentation="Fraction of PP prefill loop time with no batch (idle).",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.prefill_busy_ratio = Gauge(
            name="sglang:prefill_busy_ratio",
            documentation="Fraction of PP prefill loop time with a batch (busy).",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        # Decode loop utilization
        self.decode_idle_ratio = Gauge(
            name="sglang:decode_idle_ratio",
            documentation="Fraction of decode loop time with no batch (idle).",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.decode_busy_ratio = Gauge(
            name="sglang:decode_busy_ratio",
            documentation="Fraction of decode loop time with a batch (busy).",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        # Device memory occupancy (leak probe)
        self.mem_alloc_mb = Gauge(
            name="sglang:mem_alloc_mb",
            documentation="Torch caching allocator bytes currently allocated (MB).",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.mem_reserved_mb = Gauge(
            name="sglang:mem_reserved_mb",
            documentation="Torch caching allocator bytes reserved / pool size (MB).",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.mem_phys_used_mb = Gauge(
            name="sglang:mem_phys_used_mb",
            documentation="Driver-level physical memory occupancy, total - free (MB).",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.mem_non_torch_mb = Gauge(
            name="sglang:mem_non_torch_mb",
            documentation="Physical bytes held outside the torch caching allocator, phys_used - reserved (MB).",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        # Engine startup
        self.engine_startup_time = Gauge(
            name="sglang:engine_startup_time",
            documentation="The time taken for the engine to start up.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.engine_load_weights_time = Gauge(
            name="sglang:engine_load_weights_time",
            documentation="The time taken for the engine to load weights.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        # Additional queueing time histogram
        self.queue_time = Histogram(
            name="sglang:queue_time_s",
            documentation="Histogram of queueing time in seconds.",
            labelnames=labels.keys(),
            buckets=[
                0.0,
                0.1,
                0.2,
                0.5,
                1,
                2,
                3,
                4,
                5,
                10,
                15,
                20,
                30,
                40,
                50,
                60,
                70,
                80,
                90,
                100,
                200,
                300,
                400,
                500,
                600,
                700,
                800,
                900,
                1000,
                1200,
                1400,
                1600,
                1800,
                2000,
                2500,
                3000,
            ],
        )

        # Grammar metrics
        self.grammar_compilation_time = Histogram(
            name="sglang:grammar_compilation_time_seconds",
            documentation="Histogram of grammar compilation time in seconds.",
            labelnames=labels.keys(),
            buckets=[
                0.0,
                0.01,
                0.02,
                0.05,
                0.1,
                0.2,
                0.5,
                1,
                2,
                5,
                10,
                20,
                30,
                60,
                90,
                120,
                240,
            ],
        )
        self.num_grammar_cache_hit = Counter(
            name="sglang:num_grammar_cache_hit_total",
            documentation="Number of grammar cache hits.",
            labelnames=labels.keys(),
        )
        self.num_grammar_aborted = Counter(
            name="sglang:num_grammar_aborted_total",
            documentation="Number of grammar aborted requests.",
            labelnames=labels.keys(),
        )
        self.num_grammar_total = Counter(
            name="sglang:num_grammar_total",
            documentation="Number of the total grammar requests.",
            labelnames=labels.keys(),
        )
        self.grammar_schema_count = Histogram(
            name="sglang:grammar_schema_count",
            documentation="Histogram of grammar schema count.",
            labelnames=labels.keys(),
            buckets=[
                0,
                1,
                2,
                5,
                10,
                20,
                30,
                40,
                60,
                80,
                100,
                120,
                140,
                160,
                180,
                200,
                300,
                400,
                500,
                700,
                1000,
            ],
        )
        self.grammar_ebnf_size = Histogram(
            name="sglang:grammar_ebnf_size",
            documentation="Histogram of grammar EBNF size.",
            labelnames=labels.keys(),
            buckets=[
                0,
                50,
                100,
                200,
                300,
                500,
                1000,
                2000,
                3000,
                5000,
                10000,
                20000,
                30000,
                50000,
                100000,
            ],
        )

        tree_traversal_time_buckets = [
            0.0,
            0.01,
            0.02,
            0.05,
            0.1,
            0.2,
            0.5,
            1,
            2,
            5,
            10,
            15,
            30,
            60,
            90,
            120,
            240,
        ]
        self.grammar_tree_traversal_time_avg = Histogram(
            name="sglang:grammar_tree_traversal_time_avg",
            documentation="Histogram of average grammar tree traversal time in seconds.",
            labelnames=labels.keys(),
            buckets=tree_traversal_time_buckets,
        )
        self.grammar_tree_traversal_time_max = Histogram(
            name="sglang:grammar_tree_traversal_time_max",
            documentation="Histogram of max grammar tree traversal time in seconds.",
            labelnames=labels.keys(),
            buckets=tree_traversal_time_buckets,
        )

        self.request_latency_seconds = Histogram(
            name="sglang:request_latency_seconds",
            documentation="The latency of each stage of requests.",
            # captures latency in range [1ms - ~1191s]
            buckets=exponential_buckets(start=0.001, width=1.62, length=30),
            labelnames=list(labels.keys()) + ["stage"],
        )

    def _log_gauge(self, gauge, data: Union[int, float]) -> None:
        # Convenience function for logging to gauge.
        gauge.labels(**self.labels).set(data)

    def log_histogram(self, histogram, data: Union[int, float]) -> None:
        if self.enable_prometheus:
            histogram.labels(**self.labels).observe(data)

    def increment_bootstrap_failed_reqs(self) -> None:
        if self.enable_prometheus:
            self.num_bootstrap_failed_reqs.labels(**self.labels).inc(1)

    def increment_transfer_failed_reqs(self) -> None:
        if self.enable_prometheus:
            self.num_transfer_failed_reqs.labels(**self.labels).inc(1)

    def observe_request_latency_seconds(self, stage: str, latency: float) -> None:
        if self.enable_prometheus:
            labels_with_stage = {**self.labels, "stage": stage}
            self.request_latency_seconds.labels(**labels_with_stage).observe(latency)

        if self.enable_cat:
            # Log request latency to CAT with stage information
            self.cat_reporter.log_duration(f"RequestLatency_{stage}", latency)

    def observe_l3_prefetch_stages(
        self,
        query_queue_s: Optional[float],
        storage_query_s: Optional[float],
        query_sync_s: Optional[float],
        query_s: Optional[float],
        query_to_read_s: Optional[float],
        queue_s: Optional[float],
        read_s: Optional[float],
        finish_s: Optional[float],
    ):
        # Per-request L3 prefetch stage latencies, observed once per completed
        # prefetch at finalize time in the scheduler. Values < 0 mean some
        # timestamps were never stamped (abnormal path); skip them to avoid
        # polluting histogram sums.
        def _valid(v):
            return v is not None and v >= 0.0

        if self.enable_prometheus:
            for hist, value in (
                (self.histogram_pf_query_queue_seconds, query_queue_s),
                (self.histogram_pf_storage_query_seconds, storage_query_s),
                (self.histogram_pf_query_sync_seconds, query_sync_s),
                (self.histogram_pf_query_seconds, query_s),
                (self.histogram_pf_query_to_read_seconds, query_to_read_s),
                (self.histogram_pf_queue_seconds, queue_s),
                (self.histogram_pf_read_seconds, read_s),
                (self.histogram_pf_finish_seconds, finish_s),
            ):
                if _valid(value):
                    hist.labels(**self.pf_labels).observe(value)

        if self.enable_cat:
            for name, value in (
                ("Prefetch_QueryQueueMs", query_queue_s),
                ("Prefetch_StorageHitQueryMs", storage_query_s),
                ("Prefetch_QuerySyncMs", query_sync_s),
                ("Prefetch_QueryMs", query_s),
                ("Prefetch_QueryToReadMs", query_to_read_s),
                ("Prefetch_QueueMs", queue_s),
                ("Prefetch_ReadMs", read_s),
                ("Prefetch_FinishMs", finish_s),
            ):
                if _valid(value):
                    self.cat_reporter.log_duration(name, value)

    def _log_prometheus(self, stats: SchedulerStats) -> None:
        self._log_gauge(self.num_running_reqs, stats.num_running_reqs)
        self._log_gauge(self.running_bs_rate, stats.running_bs_rate)
        self._log_gauge(self.num_used_tokens, stats.num_used_tokens)
        self._log_gauge(self.num_free_tokens, stats.num_free_tokens)
        self._log_gauge(self.token_usage, stats.token_usage)
        self._log_gauge(self.gen_throughput, stats.gen_throughput)
        self._log_gauge(self.num_queue_reqs, stats.num_queue_reqs)

        # Report cumulative prefill hit rates, reset every 60 seconds
        now = time.time()
        if now - self.last_cache_hit_metric_update_time >= 60:
            total_tokens, hit_tokens, l1_hit_tokens, l2_hit_tokens, l3_hit_tokens = stats.cache_hit_metrics.get()
            if total_tokens > 0:
                self._log_gauge(self.cache_hit_rate, hit_tokens / total_tokens * 100)
                self._log_gauge(self.hicache_l1_hit_rate, l1_hit_tokens / total_tokens * 100)
                self._log_gauge(self.hicache_l2_hit_rate, l2_hit_tokens / total_tokens * 100)
                self._log_gauge(self.hicache_l3_hit_rate, l3_hit_tokens / total_tokens * 100)

            slm = stats.scheduler_latency_metrics
            for gauge, bucket in (
                (self.pf_query_ms, slm.pf_query),
                (self.pf_queue_ms, slm.pf_queue),
                (self.pf_finish_ms, slm.pf_finish),
                (self.chunk_forward_ms, slm.chunk_forward),
                (self.fwd_per_1kkv_ms, slm.fwd_per_1kkv),
                (self.l2_load_ms, slm.l2_load),
                (self.l2_load_per_1kkv_ms, slm.l2_load_per_1kkv),
            ):
                v = bucket.avg()
                if v is not None:
                    self._log_gauge(gauge, v)
            self._log_gauge(self.pf_completed_reqs, slm.pf_finish.n)

        self._log_gauge(self.spec_accept_length, stats.spec_accept_length)
        # PD disaggregation
        self._log_gauge(self.num_prefill_prealloc_queue_reqs, stats.num_prefill_prealloc_queue_reqs)
        self._log_gauge(self.num_prefill_inflight_queue_reqs, stats.num_prefill_inflight_queue_reqs)
        self._log_gauge(self.num_decode_prealloc_queue_reqs, stats.num_decode_prealloc_queue_reqs)
        self._log_gauge(self.num_decode_transfer_queue_reqs, stats.num_decode_transfer_queue_reqs)
        self._log_gauge(self.kv_transfer_speed_gb_s, stats.kv_transfer_speed_gb_s)
        self._log_gauge(self.kv_transfer_latency_ms, stats.kv_transfer_latency_ms)
        if stats.decode_cache_total_tokens > 0:
            self._log_gauge(self.decode_cache_hit_rate, stats.decode_cache_hit_rate)

        # Retract
        self._log_gauge(self.total_retracted_reqs, stats.total_retracted_reqs)
        self._log_gauge(self.num_retracted_reqs, stats.num_retracted_reqs)
        self._log_gauge(self.num_retracted_queue_reqs, stats.num_retracted_queue_reqs)
        self._log_gauge(self.num_paused_reqs, stats.num_paused_reqs)

        # Prefill chunk token count
        self._log_gauge(self.prefill_chunk_tokens, stats.prefill_chunk_tokens)

        # PP prefill loop utilization
        self._log_gauge(self.prefill_idle_ratio, stats.prefill_idle_ratio)
        self._log_gauge(self.prefill_busy_ratio, stats.prefill_busy_ratio)

        # Decode loop utilization
        self._log_gauge(self.decode_idle_ratio, stats.decode_idle_ratio)
        self._log_gauge(self.decode_busy_ratio, stats.decode_busy_ratio)

    def _log_cat(self, stats: SchedulerStats, is_prefill: bool = False) -> None:
        self.cat_reporter.log_count("RunningBS", stats.num_running_reqs)
        self.cat_reporter.log_count("RunningBSRate", stats.running_bs_rate)
        self.cat_reporter.log_count("NumUsedTokens", stats.num_used_tokens)
        self.cat_reporter.log_count("NumFreeTokens", stats.num_free_tokens)
        self.cat_reporter.log_count("TokenUsage", stats.token_usage)
        self.cat_reporter.log_count("TokenUsageRate", stats.token_usage * 100)
        self.cat_reporter.log_count("NumWaitingReqs", stats.num_queue_reqs)
        if is_prefill:
            # Report cumulative prefill hit rates, reset every 60 seconds
            now = time.time()
            if now - self.last_cache_hit_metric_update_time >= 60:
                total_tokens, hit_tokens, l1_hit_tokens, l2_hit_tokens, l3_hit_tokens = stats.cache_hit_metrics.get()
                self.cat_reporter.log_count("CumulativeTotalTokenNum", total_tokens)
                self.cat_reporter.log_count("CumulativeCacheTokenNum", hit_tokens)
                self.cat_reporter.log_count("CumulativeL1TokenNum", l1_hit_tokens)
                self.cat_reporter.log_count("CumulativeL2TokenNum", l2_hit_tokens)
                self.cat_reporter.log_count("CumulativeL3TokenNum", l3_hit_tokens)
                if total_tokens > 0:
                    self.cat_reporter.log_count("CumulativeCacheHitRate", hit_tokens / total_tokens * 100)
                    self.cat_reporter.log_count("CumulativeL1HitRate", l1_hit_tokens / total_tokens * 100)
                    self.cat_reporter.log_count("CumulativeL2HitRate", l2_hit_tokens / total_tokens * 100)
                    self.cat_reporter.log_count("CumulativeL3HitRate", l3_hit_tokens / total_tokens * 100)

                slm = stats.scheduler_latency_metrics
                for name, bucket in (
                    ("ChunkForward_AvgMs", slm.chunk_forward),
                    ("Fwd_Per_1kKv_Ms", slm.fwd_per_1kkv),
                    ("L2Load_AvgMs", slm.l2_load),
                    ("L2Load_Per_1kKv_Ms", slm.l2_load_per_1kkv),
                ):
                    v = bucket.avg()
                    if v is not None:
                        self.cat_reporter.log_count(name, v)

            if stats.dequeue_time > 0.0:
                self.cat_reporter.log_duration("DequeueTime", stats.dequeue_time)
            # grammar
            self.cat_reporter.log_count("GrammarQueueReqs", stats.num_grammar_queue_reqs)
            # Prefill chunk token count
            self.cat_reporter.log_count("PrefillChunkTokens", stats.prefill_chunk_tokens)
            # PP prefill loop utilization
            self.cat_reporter.log_count("PrefillIdleRatio", stats.prefill_idle_ratio * 100)
            self.cat_reporter.log_count("PrefillBusyRatio", stats.prefill_busy_ratio * 100)
            # PD disaggregation metrics
            self.cat_reporter.log_count("PrefillPreallocQueueReqs", stats.num_prefill_prealloc_queue_reqs)
            self.cat_reporter.log_count("PrefillInflightQueueReqs", stats.num_prefill_inflight_queue_reqs)
        else:
            # Decode loop utilization
            self.cat_reporter.log_count("DecodeIdleRatio", stats.decode_idle_ratio * 100)
            self.cat_reporter.log_count("DecodeBusyRatio", stats.decode_busy_ratio * 100)
            self.cat_reporter.log_count("GenThroughput", stats.gen_throughput)
            if stats.spec_accept_length != 0:
                self.cat_reporter.log_count("SpecAccpetLen", stats.spec_accept_length)
            # PD disaggregation metrics
            self.cat_reporter.log_count("DecodePreallocQueueReqs", stats.num_decode_prealloc_queue_reqs)
            self.cat_reporter.log_count("DecodeTransferQueueReqs", stats.num_decode_transfer_queue_reqs)
            if stats.decode_cache_total_tokens > 0:
                self.cat_reporter.log_count("DecodeCacheHitRate", stats.decode_cache_hit_rate * 100)
                self.cat_reporter.log_count("DecodeCacheTotalTokenNum", stats.decode_cache_total_tokens)
                self.cat_reporter.log_count("DecodeCacheHitTokenNum", stats.decode_cache_hit_tokens)

            #   9.22 SGLang open source community has not yet implemented actual recording of
            #   KVTransferSpeedGBs and other data, pending future cherry-pick
            #   self.cat_reporter.log_count("KVTransferSpeedGBs", stats.kv_transfer_speed_gb_s)
            #   self.cat_reporter.log_duration("KVTransferLatencyMs", stats.kv_transfer_latency_ms / 1000.0)  # Convert ms to seconds for duration
            # Retract metrics
            self.cat_reporter.log_count("TotalRetractedReqs", stats.total_retracted_reqs)
            self.cat_reporter.log_count("RetractedQueueReqs", stats.num_retracted_queue_reqs)
            #   self.cat_reporter.log_count("NumRetractedReqs", stats.num_retracted_reqs)
            #   self.cat_reporter.log_count("NumPausedReqs", stats.num_paused_reqs)

    def _log_llm_platform(self, stats: SchedulerStats, is_prefill: bool = False) -> None:
        if not is_prefill:
            self.llm_platform_reporter.metrics_decode(
                running_reqs=stats.num_running_reqs,
                pending_reqs=stats.num_queue_reqs,
                decode_used_tokens=stats.num_used_tokens,
                decode_token_usage=stats.token_usage,
                decode_throughput=stats.gen_throughput,
            )

    def log_stats(self, stats: SchedulerStats, is_prefill: bool = False) -> None:
        if self.enable_prometheus:
            self._log_prometheus(stats)

        if self.enable_cat:
            self._log_cat(stats, is_prefill)

        if self.enable_llm_platform:
            self._log_llm_platform(stats, is_prefill)

        self.last_log_time = time.time()
        if self.last_log_time - self.last_cache_hit_metric_update_time >= 60:
            stats.cache_hit_metrics.reset()
            stats.scheduler_latency_metrics.reset()
            self.last_cache_hit_metric_update_time = self.last_log_time

    def log_memory_stats(self, stats: SchedulerStats) -> None:
        """Report device memory occupancy (leak probe).

        Reported on its own cadence (every N decode steps), independent of the
        regular decode-stats interval. Values come from stats.mem_* which the
        scheduler fills in just before calling this.
        """
        if self.enable_prometheus:
            self._log_gauge(self.mem_alloc_mb, stats.mem_alloc_mb)
            self._log_gauge(self.mem_reserved_mb, stats.mem_reserved_mb)
            self._log_gauge(self.mem_phys_used_mb, stats.mem_phys_used_mb)
            self._log_gauge(self.mem_non_torch_mb, stats.mem_non_torch_mb)

        if self.enable_cat:
            self.cat_reporter.log_count("MemAllocMB", stats.mem_alloc_mb)
            self.cat_reporter.log_count("MemReservedMB", stats.mem_reserved_mb)
            self.cat_reporter.log_count("MemPhysUsedMB", stats.mem_phys_used_mb)
            self.cat_reporter.log_count("MemNonTorchMB", stats.mem_non_torch_mb)

    def observe_module_time_in_scheduler(self, step_time: float, name: str = "scheduler_module"):
        if self.enable_cat:
            self.cat_reporter.log_duration(name, step_time)

    def observe_grammar_degrade(self, _reason: str = "", _label: str = "") -> None:
        """Report one request degraded from grammar-constrained to unconstrained.

        Called once per request from the rank that owns the CAT series (see
        Scheduler.__init__); ``_reason``/``_label`` are already in the error log.
        """
        if self.enable_cat:
            self.cat_reporter.log_count("GrammarDegrade", 1)

    def observe_epd_poll_admission_success(self, poll_seconds: float, admission_seconds: float) -> None:
        """Report successful EPD admission with the minimum CAT writes.

        Either duration's CAT transaction count is the success count, so a
        separate poll_admission.success transaction would be redundant.
        """
        if not self.enable_epd_multimodal_cat:
            return
        self.cat_reporter.log_duration("poll_admission.poll", poll_seconds)
        self.cat_reporter.log_duration("poll_admission.admission", admission_seconds)

    def observe_epd_poll_admission_failed(self) -> None:
        if not self.enable_epd_multimodal_cat:
            return
        self.cat_reporter.log_count("poll_admission.failed", 1)

    def observe_mfu(self, mfu: float, tflops_per_sec: float):
        if self.enable_cat:
            self.cat_reporter.log_count("PrefillMFU", mfu * 100)
            self.cat_reporter.log_count("PrefillTFLOPsPerSec", tflops_per_sec)


class TokenizerMetricsCollector:
    def __init__(
        self,
        labels: dict[str, str],
        metrics_reporters: list[str],
        enable_multimodal: bool = False,
    ) -> None:
        self.enable_prometheus = "prometheus" in metrics_reporters
        self.enable_cat = "cat" in metrics_reporters
        self.enable_multimodal_cat = self.enable_cat and enable_multimodal
        self.enable_llm_platform = "llm-platform" in metrics_reporters

        if self.enable_prometheus:
            self._init_prometheus(labels)

        if self.enable_cat:
            self.cat_reporter = get_cat_reporter(labels.get("model_name", "DefalutModel"), labels.get("app_key", "DefaultAppKey"))

    def _init_prometheus(self, labels: dict[str, str]) -> None:
        # We need to import prometheus_client after setting the env variable `PROMETHEUS_MULTIPROC_DIR`
        from prometheus_client import Counter, Histogram

        self.labels = labels

        self.prompt_tokens_total = Counter(
            name="sglang:prompt_tokens_total",
            documentation="Number of prefill tokens processed.",
            labelnames=labels.keys(),
        )

        self.generation_tokens_total = Counter(
            name="sglang:generation_tokens_total",
            documentation="Number of generation tokens processed.",
            labelnames=labels.keys(),
        )

        self.num_requests_total = Counter(
            name="sglang:num_requests_total",
            documentation="Number of requests processed.",
            labelnames=labels.keys(),
        )

        self.histogram_time_to_first_token = Histogram(
            name="sglang:time_to_first_token_seconds",
            documentation="Histogram of time to first token in seconds.",
            labelnames=labels.keys(),
            buckets=[
                0.1,
                0.3,
                0.5,
                0.7,
                0.9,
                1,
                2,
                4,
                6,
                8,
                10,
                20,
                40,
                60,
                80,
                120,
                160,
            ],
        )

        self.histogram_time_per_output_token = Histogram(
            name="sglang:time_per_output_token_seconds",
            documentation="Histogram of time per output token in seconds.",
            labelnames=labels.keys(),
            buckets=[
                0.002,
                0.005,
                0.010,
                0.020,
                0.030,
                0.040,
                0.050,
                0.060,
                0.070,
                0.080,
                0.090,
                0.100,
                0.150,
                0.200,
                0.300,
                0.400,
                0.600,
                0.800,
                1.000,
                2.000,
            ],
        )

        self.histogram_inter_token_latency_seconds = Histogram(
            name="sglang:inter_token_latency_seconds",
            documentation="Histogram of inter-token latency in seconds.",
            labelnames=labels.keys(),
            buckets=[
                0.002,
                0.004,
                0.006,
                0.008,
                0.010,
                0.015,
                0.020,
                0.025,
                0.030,
                0.035,
                0.040,
                0.050,
                0.075,
                0.100,
                0.150,
                0.200,
                0.300,
                0.400,
                0.500,
                0.750,
                1.000,
                2.000,
            ],
        )

        self.histogram_e2e_request_latency = Histogram(
            name="sglang:e2e_request_latency_seconds",
            documentation="Histogram of End-to-end request latency in seconds",
            labelnames=labels.keys(),
            buckets=[
                0.1,
                0.2,
                0.4,
                0.8,
                1,
                2,
                5,
                10,
                20,
                40,
                60,
                80,
                100,
                150,
                200,
                250,
                300,
                350,
                500,
                1000,
            ],
        )

        # L3 prefetch per-stage latency histograms live in
        # SchedulerMetricsCollector (observed where the timestamps are produced);
        # only e2e/tokens/TTFT-pct are still reported here, see
        # observe_l3_prefetch_ttft.

    def _log_histogram(self, histogram, data: Union[int, float]) -> None:
        if self.enable_prometheus:
            histogram.labels(**self.labels).observe(data)

    def observe_one_finished_request(
        self,
        prompt_tokens: int,
        generation_tokens: int,
        e2e_latency: float,
        tokenized_duration: float,
        reasoning_tokens: int = 0,
        prompt_token_cat_aliases: tuple[str, ...] = (),
    ):
        if self.enable_prometheus:
            self.prompt_tokens_total.labels(**self.labels).inc(prompt_tokens)
            self.generation_tokens_total.labels(**self.labels).inc(generation_tokens)
            self.num_requests_total.labels(**self.labels).inc(1)
            self._log_histogram(self.histogram_e2e_request_latency, e2e_latency)
            if generation_tokens >= 1:
                self.histogram_time_per_output_token.labels(**self.labels).observe(e2e_latency / generation_tokens)

        if self.enable_cat:
            self.cat_reporter.log_count("PromptTokens", prompt_tokens)
            if self.enable_multimodal_cat:
                for alias in prompt_token_cat_aliases:
                    self.cat_reporter.log_count(alias, prompt_tokens)
            self.cat_reporter.log_count("CompletionTokens", generation_tokens)
            if reasoning_tokens:
                self.cat_reporter.log_count("ReasoningTokens", reasoning_tokens)
            self.cat_reporter.log_duration("RequestTime", e2e_latency)
            self.cat_reporter.log_duration("TokenizedTime", tokenized_duration)

    def observe_time_to_first_token(
        self,
        value: float,
        name: str = "TTFT",
        cat_aliases: tuple[str, ...] = (),
    ):
        if self.enable_prometheus:
            self.histogram_time_to_first_token.labels(**self.labels).observe(value)

        if self.enable_cat:
            self.cat_reporter.log_duration(name, value)
            if self.enable_multimodal_cat:
                for alias in cat_aliases:
                    self.cat_reporter.log_duration(alias, value)

    def observe_l3_prefetch_ttft(
        self,
        e2e_s: Optional[float],
        tokens: Optional[int],
        ttft_s: float,
    ):
        # Per-stage latencies are reported by the scheduler collector
        # (observe_l3_prefetch_stages); here we only join prefetch e2e with
        # TTFT, which is only known in the tokenizer process.
        if self.enable_cat:
            if e2e_s is not None and tokens is not None:
                self.cat_reporter.log_duration("L3PrefetchE2E", e2e_s)
                self.cat_reporter.log_count("L3PrefetchTokens", tokens)
                self.cat_reporter.log_count("L3PrefetchTTFTPct", 100.0 * e2e_s / ttft_s)

    def observe_inter_token_latency(
        self,
        internval: float,
        num_new_tokens: int,
        name: str = "TPOT",
        cat_aliases: tuple[str, ...] = (),
    ):
        adjusted_interval = internval / num_new_tokens

        if self.enable_prometheus:
            # A faster version of the Histogram::observe which observes multiple values at the same time.
            # reference: https://github.com/prometheus/client_python/blob/v0.21.1/prometheus_client/metrics.py#L639
            his = self.histogram_inter_token_latency_seconds.labels(**self.labels)
            his._sum.inc(internval)

            for i, bound in enumerate(his._upper_bounds):
                if adjusted_interval <= bound:
                    his._buckets[i].inc(num_new_tokens)
                    break

        if self.enable_cat:
            self.cat_reporter.log_duration(name, adjusted_interval)
            if self.enable_multimodal_cat:
                for alias in cat_aliases:
                    self.cat_reporter.log_duration(alias, adjusted_interval)

    def observe_accept_length(
        self,
        value: float,
        cat_aliases: tuple[str, ...] = (),
    ) -> None:
        """Report the per-request speculative acceptance length to CAT."""
        if self.enable_multimodal_cat:
            self.cat_reporter.log_count("acceptLength", value)
            for alias in cat_aliases:
                self.cat_reporter.log_count(alias, value)

    def observe_request_arrival(self, batch_size: int = 1):
        if self.enable_prometheus:
            pass

        if self.enable_cat:
            self.cat_reporter.log_count("RequestArrival", batch_size)


class EncoderMetricsCollector:
    """Metrics owned by Encoder TP rank 0.

    Prometheus updates happen inline and never perform network I/O. CAT samples
    are reported directly except for request-queue snapshots sampled by the
    server. CAT transaction counts are sample counts, not queue lengths.
    ``llm-platform`` is ignored until its schema has Encoder fields.
    """

    _QUEUES = ("request", "request_items", "vit_pending")
    _STATUSES = {"success", "timeout", "error"}
    _OUTPUT_FORMATS = {"rvq_indices", "vit_embedding"}
    _PREPROCESS_STAGES = {"download", "decode", "transform", "hash"}
    _PREPROCESS_BACKENDS = {"cpu", "device"}
    _PUSH_TYPES = {"push.full", "push.meta"}

    def __init__(
        self,
        labels: dict[str, str],
        metrics_reporters: Optional[list[str]],
        prometheus_registry=None,
    ) -> None:
        requested_reporters = list(dict.fromkeys(metrics_reporters or ["cat"]))
        self.enable_prometheus = "prometheus" in requested_reporters
        self.enable_cat = "cat" in requested_reporters
        self.labels = labels

        if "llm-platform" in requested_reporters:
            logger.warning("Encoder metrics do not support llm-platform yet; skipping it")

        self._last_reporter_warning_at = 0.0
        self._last_cat_queue_lengths: dict[str, int] = {}
        self._last_cat_queue_oldest_seconds: Optional[float] = None

        if self.enable_prometheus:
            self._init_prometheus(labels, prometheus_registry)

        self.cat_reporter = None
        if self.enable_cat:
            try:
                self.cat_reporter = get_cat_reporter(
                    labels.get("model_name") or "DefaultModel",
                    labels.get("app_key") or "DefaultAppKey",
                )
            except Exception:
                self.enable_cat = False
                logger.exception("Failed to initialize CAT reporter for Encoder metrics")

    @staticmethod
    def _metric_kwargs(prometheus_registry) -> dict:
        return {"registry": prometheus_registry} if prometheus_registry is not None else {}

    def _init_prometheus(self, labels, prometheus_registry) -> None:
        # Import only after the parent has set PROMETHEUS_MULTIPROC_DIR.
        from prometheus_client import Counter, Gauge, Histogram

        metric_kwargs = self._metric_kwargs(prometheus_registry)
        base_labels = tuple(labels.keys())
        preprocess_buckets = [
            0.001,
            0.005,
            0.01,
            0.05,
            0.1,
            0.25,
            0.5,
            1,
            2,
            5,
            10,
            20,
            30,
            60,
            120,
            300,
        ]
        vit_buckets = [
            0.001,
            0.002,
            0.005,
            0.01,
            0.02,
            0.05,
            0.1,
            0.2,
            0.5,
            1,
            2,
            5,
            10,
        ]
        push_buckets = [
            0.0001,
            0.0005,
            0.001,
            0.002,
            0.005,
            0.01,
            0.02,
            0.05,
            0.1,
            0.2,
            0.5,
            1,
            2,
            5,
        ]
        batch_buckets = [
            1,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
            32768,
        ]

        self.preprocess_duration = Histogram(
            name="fllm:encoder_preprocess_duration_seconds",
            documentation="Encoder media preprocessing duration in seconds.",
            labelnames=base_labels + ("modality", "status"),
            buckets=preprocess_buckets,
            **metric_kwargs,
        )
        self.preprocess_stage_duration = Histogram(
            name="fllm:encoder_preprocess_stage_duration_seconds",
            documentation=("Encoder media preprocessing stage duration in seconds."),
            labelnames=base_labels + ("modality", "stage", "status"),
            buckets=preprocess_buckets,
            **metric_kwargs,
        )
        self.vit_duration = Histogram(
            name="fllm:encoder_vit_duration_seconds",
            documentation="Encoder vision forward duration in seconds.",
            labelnames=base_labels
            + (
                "modality",
                "output_format",
                "status",
            ),
            buckets=vit_buckets,
            **metric_kwargs,
        )
        self.push_duration = Histogram(
            name="fllm:encoder_push_duration_seconds",
            documentation="Successful Encoder result push duration in seconds.",
            labelnames=base_labels + ("type",),
            buckets=push_buckets,
            **metric_kwargs,
        )
        self.api_calls = Counter(
            name="fllm:encoder_api_calls_total",
            documentation="Encoder API calls.",
            labelnames=base_labels + ("operation", "status"),
            **metric_kwargs,
        )
        self.cache_lookups = Counter(
            name="fllm:encoder_cache_lookups_total",
            documentation="Logical Encoder media cache lookups.",
            labelnames=base_labels,
            **metric_kwargs,
        )
        self.cache_hits = Counter(
            name="fllm:encoder_cache_hits_total",
            documentation="Logical Encoder media cache hits.",
            labelnames=base_labels,
            **metric_kwargs,
        )
        self.queue_length = Gauge(
            name="fllm:encoder_queue_length",
            documentation="Current Encoder queue length.",
            labelnames=base_labels + ("queue",),
            multiprocess_mode="mostrecent",
            **metric_kwargs,
        )
        self.request_queue_oldest_seconds = Gauge(
            name="fllm:encoder_request_queue_oldest_seconds",
            documentation="Age in seconds of the oldest request awaiting preprocessing.",
            labelnames=base_labels,
            multiprocess_mode="mostrecent",
            **metric_kwargs,
        )
        self.request_queue_oldest_seconds.labels(**self.labels).set(0)
        self.batch_tokens = Histogram(
            name="fllm:encoder_batch_tokens",
            documentation="Visual tokens in each Encoder scheduler batch.",
            labelnames=base_labels,
            buckets=batch_buckets,
            **metric_kwargs,
        )

        for queue in self._QUEUES:
            self.queue_length.labels(**self.labels, queue=queue).set(0)

    @classmethod
    def _status(cls, status: str) -> str:
        return status if status in cls._STATUSES else "error"

    @staticmethod
    def _modality(modality: str) -> str:
        value = str(modality).upper()
        return value if value in {"IMAGE", "VIDEO"} else "UNKNOWN"

    @classmethod
    def _preprocess_backend(cls, backend: str) -> str:
        backend = str(backend).lower()
        return backend if backend in cls._PREPROCESS_BACKENDS else "cpu"

    def record_api_call(self, operation: str, status: str) -> None:
        status = self._status(status)
        if self.enable_prometheus:
            self.api_calls.labels(**self.labels, operation=operation, status=status).inc()
        if self.enable_cat:
            cat_name = {
                "encode_async": "EncoderMetricsEncodeAsyncCalls",
                "register_delivery_addr_full": ("EncoderMetricsRegisterDeliveryAddrFullCalls"),
                "register_delivery_addr_meta": ("EncoderMetricsRegisterDeliveryAddrMetaCalls"),
            }.get(operation)
            if cat_name is not None:
                self._safe_cat_call("log_count", cat_name, 1)

    def observe_preprocess(
        self,
        modality: str,
        status: str,
        seconds: float,
        backend: str = "cpu",
    ) -> None:
        modality = self._modality(modality)
        status = self._status(status)
        backend = self._preprocess_backend(backend)
        seconds = max(0.0, float(seconds))
        if self.enable_prometheus:
            self.preprocess_duration.labels(**self.labels, modality=modality, status=status).observe(seconds)
        if self.enable_cat:
            suffix = "" if status == "success" else "Error"
            self._safe_cat_call(
                "log_duration",
                f"EncoderMetricsPreprocess{modality.title()}{suffix}",
                seconds,
            )

            # Preserve the aggregate series and add a backend-specific alias.
            self._safe_cat_call(
                "log_duration",
                f"EncoderMetricsPreprocess{modality.title()}{backend.title()}{suffix}",
                seconds,
            )

    def observe_preprocess_stage(
        self,
        modality: str,
        stage: str,
        status: str,
        seconds: float,
        backend: str = "cpu",
    ) -> None:
        modality = self._modality(modality)
        stage = stage if stage in self._PREPROCESS_STAGES else "unknown"
        backend = self._preprocess_backend(backend)
        status = self._status(status)
        seconds = max(0.0, float(seconds))
        if self.enable_prometheus:
            self.preprocess_stage_duration.labels(
                **self.labels,
                modality=modality,
                stage=stage,
                status=status,
            ).observe(seconds)
        if self.enable_cat:
            suffix = "" if status == "success" else "Error"
            self._safe_cat_call(
                "log_duration",
                f"EncoderMetricsPreprocess{modality.title()}{stage.title()}{suffix}",
                seconds,
            )

            # Preserve the aggregate stage series and add its backend alias.
            self._safe_cat_call(
                "log_duration",
                f"EncoderMetricsPreprocess{modality.title()}{backend.title()}{stage.title()}{suffix}",
                seconds,
            )

    def observe_vit(
        self,
        modality: str,
        output_format: str,
        status: str,
        seconds: float,
    ) -> None:
        modality = self._modality(modality)
        output_format = output_format if output_format in self._OUTPUT_FORMATS else "unknown"
        status = self._status(status)
        seconds = max(0.0, float(seconds))
        if self.enable_prometheus:
            self.vit_duration.labels(
                **self.labels,
                modality=modality,
                output_format=output_format,
                status=status,
            ).observe(seconds)
        if self.enable_cat:
            suffix = "" if status == "success" else "Error"
            self._safe_cat_call(
                "log_duration",
                f"EncoderMetricsVit{modality.title()}{suffix}",
                seconds,
            )

    def observe_push(self, push_type: str, seconds: float) -> None:
        push_type = push_type if push_type in self._PUSH_TYPES else "push.full"
        seconds = max(0.0, float(seconds))
        if self.enable_prometheus:
            self.push_duration.labels(**self.labels, type=push_type).observe(seconds)
        if self.enable_cat:
            type_name = "Meta" if push_type == "push.meta" else "Full"
            self._safe_cat_call(
                "log_duration",
                f"EncoderMetricsPush{type_name}",
                seconds,
            )

    def record_cache_lookup(self, hit: bool) -> None:
        if self.enable_prometheus:
            self.cache_lookups.labels(**self.labels).inc()
            if hit:
                self.cache_hits.labels(**self.labels).inc()
        if self.enable_cat:
            self._safe_cat_call(
                "log_success_rate",
                "EncoderMetricsCacheHitRate",
                hit,
            )

    def set_queue_length(self, queue: str, value: int) -> None:
        if queue not in self._QUEUES:
            raise ValueError(f"Unknown Encoder queue: {queue}")
        value = max(0, int(value))
        if self.enable_prometheus:
            self.queue_length.labels(**self.labels, queue=queue).set(value)
        if self.enable_cat and self._last_cat_queue_lengths.get(queue) != value:
            self._last_cat_queue_lengths[queue] = value
            cat_name = {
                "request": "EncoderMetricsRequestQueueLength",
                "request_items": "EncoderMetricsRequestQueueItems",
                "vit_pending": "EncoderMetricsVitPendingQueueLength",
            }[queue]
            self._safe_cat_call("log_count", cat_name, value)

    def observe_request_queue(self, requests: int, items: int, oldest_seconds: float) -> None:
        """Report a server-sampled queue snapshot; all values are instantaneous."""
        self.set_queue_length("request", requests)
        self.set_queue_length("request_items", items)
        oldest_seconds = max(0.0, float(oldest_seconds)) if requests else 0.0
        if self.enable_prometheus:
            self.request_queue_oldest_seconds.labels(**self.labels).set(oldest_seconds)
        if self.enable_cat and self._last_cat_queue_oldest_seconds != oldest_seconds:
            self._last_cat_queue_oldest_seconds = oldest_seconds
            # Numeric CAT samples use log_count, preserving the seconds unit.
            self._safe_cat_call("log_count", "EncoderMetricsRequestQueueOldestSeconds", oldest_seconds)

    def observe_batch_tokens(self, value: int) -> None:
        value = int(value)
        if value <= 0:
            return
        if self.enable_prometheus:
            self.batch_tokens.labels(**self.labels).observe(value)
        if self.enable_cat:
            self._safe_cat_call("log_count", "EncoderMetricsBatchTokens", value)

    def _safe_cat_call(self, method: str, name: str, value: float) -> None:
        if self.cat_reporter is None:
            return
        try:
            getattr(self.cat_reporter, method)(name, value)
        except Exception as error:
            now = time.monotonic()
            if now - self._last_reporter_warning_at >= 60:
                logger.warning("Encoder CAT metrics reporting failed: %s", error)
                self._last_reporter_warning_at = now


class ErrorMetricsCollector:
    def __init__(self, labels: dict[str, str], metrics_reporters: list[str]) -> None:
        self.enable_prometheus = "prometheus" in metrics_reporters
        self.enable_cat = "cat" in metrics_reporters
        self.enable_llm_platform = "llm-platform" in metrics_reporters

        if self.enable_cat:
            self.cat_reporter = get_cat_reporter(labels.get("model_name", "DefaultModel"), labels.get("app_key", "DefaultAppKey"))

    def record_error(self, error_message: str) -> None:
        # Currently only cat logging is supported
        if not self.enable_cat:
            return

        if "KVTransferError" in error_message and "due to timeout" in error_message.lower():
            formatted_error = "KVTransferTimeout"
        elif "KVTransferError" in error_message:
            match = re.search(r"remote_endpoint=([^,\)]+)", error_message)
            remote_dir = match.group(1) if match else "not found"
            formatted_error = f"KVTransferError, remote_dir={remote_dir}"
        else:
            formatted_error = "OtherError"

        self.cat_reporter.log_error(formatted_error)


class KVTransferMetricsCollector:
    def __init__(self, labels: dict[str, str], metrics_reporters: list[str]) -> None:
        self.enable_cat = "cat" in metrics_reporters

        if self.enable_cat:
            self.cat_reporter = get_cat_reporter(labels.get("model_name", "DefaultModel"), labels.get("app_key", "DefaultAppKey"))

    def log_kv_transfer_result(self, success: bool) -> None:
        """Record one terminal KV handoff as SUCCESS or EXCEPTION."""
        if not self.enable_cat:
            return

        self.cat_reporter.log_result("KVTransferResult", success)

    def log_kv_transfer_timeout(self) -> None:
        """Record a post-handshake timeout as an EXCEPTION transaction."""
        if not self.enable_cat:
            return

        self.cat_reporter.log_result("KVTransferTimeout", False)

    def log_kv_transfer_failed(self) -> None:
        """Record a non-timeout transfer failure as an EXCEPTION transaction."""
        if not self.enable_cat:
            return

        self.cat_reporter.log_result("KVTransferFailed", False)

    def log_kv_handshake_failed(self) -> None:
        """Record a handshake failure as an EXCEPTION transaction."""
        if not self.enable_cat:
            return

        self.cat_reporter.log_result("KVHandshakeFailed", False)

    def log_kv_transfer_size(self, transfer_size_bytes: int) -> None:
        if not self.enable_cat:
            return

        self.cat_reporter.log_count("KVTransferSizeBytes", transfer_size_bytes)

    def log_kv_transfer_time(self, transfer_time: float) -> None:
        if not self.enable_cat:
            return

        self.cat_reporter.log_duration("KVTransferTime", transfer_time)


class EPLBMetricsCollector:
    def __init__(self, labels: dict[str, str], metrics_reporters: list[str]) -> None:
        self.enable_cat = "cat" in metrics_reporters

        if self.enable_cat:
            self.cat_reporter = get_cat_reporter(labels.get("model_name", "DefaultModel"), labels.get("app_key", "DefaultAppKey"))

    def log_gpu_unbalancedness(self, per_device_metrics) -> None:
        if not self.enable_cat:
            return

        self.cat_reporter.log_duration("PerDeviceUnbalancedness", per_device_metrics * 100)

    def log_expert_unbalancedness(self, per_expert_metrics) -> None:
        if not self.enable_cat:
            return

        self.cat_reporter.log_duration("PerExpertUnbalancedness", per_expert_metrics * 100)


@dataclass
class StorageMetrics:
    prefetch_pgs: list[int] = field(default_factory=list)
    backup_pgs: list[int] = field(default_factory=list)
    prefetch_bandwidth: list[float] = field(default_factory=list)
    backup_bandwidth: list[float] = field(default_factory=list)


class StorageMetricsCollector:
    def __init__(
        self,
        labels: dict[str, str],
    ):
        from prometheus_client import Counter, Histogram

        self.labels = labels

        self.prefetched_tokens_total = Counter(
            name="sglang:prefetched_tokens_total",
            documentation="Number of prefetched prompt tokens.",
            labelnames=labels.keys(),
        )

        self.backuped_tokens_total = Counter(
            name="sglang:backuped_tokens_total",
            documentation="Number of backuped tokens.",
            labelnames=labels.keys(),
        )

        bucket_io = [
            1,
            5,
            10,
            50,
            100,
        ]

        bucket_bandwidth = [
            0.1,
            0.5,
            1,
            5,
            10,
            50,
            100,
        ]

        self.histogram_prefetch_pgs = Histogram(
            name="sglang:prefetch_pgs",
            documentation="Histogram of prefetch pages of batches.",
            labelnames=labels.keys(),
            buckets=bucket_io,
        )

        self.histogram_backup_pgs = Histogram(
            name="sglang:backup_pgs",
            documentation="Histogram of backup pages of batches.",
            labelnames=labels.keys(),
            buckets=bucket_io,
        )

        self.histogram_prefetch_bandwidth = Histogram(
            name="sglang:prefetch_bandwidth",
            documentation="Histogram of prefetch bandwidth in GB/s.",
            labelnames=labels.keys(),
            buckets=bucket_bandwidth,
        )

        self.histogram_backup_bandwidth = Histogram(
            name="sglang:backup_bandwidth",
            documentation="Histogram of backup bandwidth in GB/s.",
            labelnames=labels.keys(),
            buckets=bucket_bandwidth,
        )

    def log_prefetched_tokens(self, prefetched_tokens: int):
        if prefetched_tokens > 0:
            self.prefetched_tokens_total.labels(**self.labels).inc(prefetched_tokens)

    def log_backuped_tokens(self, backuped_tokens: int):
        if backuped_tokens > 0:
            self.backuped_tokens_total.labels(**self.labels).inc(backuped_tokens)

    def _log_histogram(self, histogram, data: Union[int, float]):
        histogram.labels(**self.labels).observe(data)

    def log_storage_metrics(self, storage_metrics: Optional[StorageMetrics] = None):
        if storage_metrics is None:
            return

        assert isinstance(storage_metrics, StorageMetrics)

        for v in storage_metrics.prefetch_pgs:
            self._log_histogram(self.histogram_prefetch_pgs, v)
        for v in storage_metrics.backup_pgs:
            self._log_histogram(self.histogram_backup_pgs, v)
        for v in storage_metrics.prefetch_bandwidth:
            self._log_histogram(self.histogram_prefetch_bandwidth, v)
        for v in storage_metrics.backup_bandwidth:
            self._log_histogram(self.histogram_backup_bandwidth, v)
