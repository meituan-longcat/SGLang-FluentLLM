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

import time
import re
from dataclasses import dataclass, field
from typing import Dict, Union, List
from sglang.utils import get_cat_reporter
from enum import Enum
from sglang.srt.metrics.utils import exponential_buckets
from sglang.srt.utils import get_bool_env_var

from typing import Dict, List, Optional, Union
SGLANG_TEST_REQUEST_TIME_STATS = get_bool_env_var("SGLANG_TEST_REQUEST_TIME_STATS")


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
                assert (
                    queue_duration >= 0 and forward_duration >= 0
                ), f"queue_duration={queue_duration} < 0 or forward_duration={forward_duration} < 0"

            return f"queue_duration={self.format_duration(queue_duration)}, forward_duration={self.format_duration(forward_duration)}, start_time={self.wait_queue_entry_time}"
        elif _type == self.RequestType.PREFILL:
            bootstrap_duration = (
                self.wait_queue_entry_time - self.prefill_bootstrap_queue_entry_time
            )

            queue_duration = self.forward_entry_time - self.wait_queue_entry_time

            forward_duration = self.completion_time - self.forward_entry_time

            if SGLANG_TEST_REQUEST_TIME_STATS:
                assert (
                    bootstrap_duration >= 0
                    and queue_duration >= 0
                    and forward_duration >= 0
                ), f"bootstrap_duration={bootstrap_duration} < 0 or queue_duration={queue_duration} < 0 or forward_duration={forward_duration} < 0"
            return f"bootstrap_duration={self.format_duration(bootstrap_duration)}, queue_duration={self.format_duration(queue_duration)}, forward_duration={self.format_duration(forward_duration)}, start_time={self.prefill_bootstrap_queue_entry_time}"
        # if decode
        elif _type == self.RequestType.DECODE:
            prealloc_duration = (
                self.decode_transfer_queue_entry_time
                - self.decode_prealloc_queue_entry_time
            )

            transfer_duration = (
                self.wait_queue_entry_time - self.decode_transfer_queue_entry_time
            )
            queue_duration = self.forward_entry_time - self.wait_queue_entry_time
            forward_duration = self.completion_time - self.forward_entry_time

            if SGLANG_TEST_REQUEST_TIME_STATS:
                assert (
                    prealloc_duration >= 0
                    and transfer_duration >= 0
                    and queue_duration >= 0
                    and forward_duration >= 0
                ), f"prealloc_duration={prealloc_duration} < 0 or transfer_duration={transfer_duration} < 0 or queue_duration={queue_duration} < 0 or forward_duration={forward_duration} < 0"

            return f"prealloc_duration={self.format_duration(prealloc_duration)}, transfer_duration={self.format_duration(transfer_duration)}, queue_duration={self.format_duration(queue_duration)}, forward_duration={self.format_duration(forward_duration)}, start_time={self.decode_prealloc_queue_entry_time}"
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
        elif (
            self.prefill_bootstrap_queue_entry_time > 0.0
            and self.prefill_transfer_queue_entry_time > 0.0
        ):
            return self.RequestType.PREFILL
        elif (
            self.decode_prealloc_queue_entry_time > 0.0
            and self.decode_transfer_queue_entry_time > 0.0
            and self.wait_queue_entry_time > 0.0
        ):
            return self.RequestType.DECODE
        else:
            return self.RequestType.INVALID


@dataclass
class SchedulerStats:
    num_running_reqs: int = 0
    num_used_tokens: int = 0
    token_usage: float = 0.0
    gen_throughput: float = 0.0
    num_queue_reqs: int = 0
    cache_hit_rate: float = 0.0
    spec_accept_length: float = 0.0
    dequeue_time: float = 0.0  # prefill
    # PD disaggregation
    num_prefill_prealloc_queue_reqs: int = 0
    num_prefill_inflight_queue_reqs: int = 0
    num_decode_prealloc_queue_reqs: int = 0
    num_decode_transfer_queue_reqs: int = 0
    kv_transfer_speed_gb_s: float = 0.0
    kv_transfer_latency_ms: float = 0.0
    decode_cache_hit_rate: float = 0.0  # decode cache hit rate
    
    # HiCache L1/L2/L3 hit rates
    hicache_l1_hit_rate: float = 0.0  # GPU cache hit rate
    hicache_l2_hit_rate: float = 0.0  # CPU/Host cache hit rate
    hicache_l3_hit_rate: float = 0.0  # Storage cache hit rate

    # Retract
    total_retracted_reqs: int = 0
    num_retracted_reqs: int = 0
    num_paused_reqs: int = 0


class SchedulerMetricsCollector:
    def __init__(self, labels: Dict[str, str], metrics_reporters: List[str]) -> None:
        self.enable_prometheus = 'prometheus' in metrics_reporters
        self.enable_cat = 'cat' in metrics_reporters
        self.enable_llm_platform = 'llm-platform' in metrics_reporters
        self.labels = labels

        if self.enable_prometheus:
            self._init_prometheus(labels)

        if self.enable_cat:
            self.cat_reporter = get_cat_reporter(labels.get('model_name', 'DefalutModel'),  labels.get('app_key', 'DefaultAppKey'))

        if self.enable_llm_platform:
            from longcat.tracker.hooks.sglang import SglangMetricsCollector
            self.llm_platform_reporter = SglangMetricsCollector(process_name=labels.get('process_name', 'DefalutProcess'))

        self.last_log_time = time.time()

    def _init_prometheus(self, labels: Dict[str, str]) -> None:
        # We need to import prometheus_client after setting the env variable `PROMETHEUS_MULTIPROC_DIR`
        from prometheus_client import Counter, Gauge, Histogram

        self.labels = labels

        self.num_running_reqs = Gauge(
            name="sglang:num_running_reqs",
            documentation="The number of running requests.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        self.num_used_tokens = Gauge(
            name="sglang:num_used_tokens",
            documentation="The number of used tokens.",
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

    def _log_prometheus(self, stats: SchedulerStats) -> None:
        self._log_gauge(self.num_running_reqs, stats.num_running_reqs)
        self._log_gauge(self.num_used_tokens, stats.num_used_tokens)
        self._log_gauge(self.token_usage, stats.token_usage)
        self._log_gauge(self.gen_throughput, stats.gen_throughput)
        self._log_gauge(self.num_queue_reqs, stats.num_queue_reqs)
        self._log_gauge(self.cache_hit_rate, stats.cache_hit_rate)
        self._log_gauge(self.spec_accept_length, stats.spec_accept_length)
        # PD disaggregation
        self._log_gauge(
            self.num_prefill_prealloc_queue_reqs, stats.num_prefill_prealloc_queue_reqs
        )
        self._log_gauge(
            self.num_prefill_inflight_queue_reqs, stats.num_prefill_inflight_queue_reqs
        )
        self._log_gauge(
            self.num_decode_prealloc_queue_reqs, stats.num_decode_prealloc_queue_reqs
        )
        self._log_gauge(
            self.num_decode_transfer_queue_reqs, stats.num_decode_transfer_queue_reqs
        )
        self._log_gauge(self.kv_transfer_speed_gb_s, stats.kv_transfer_speed_gb_s)
        self._log_gauge(self.kv_transfer_latency_ms, stats.kv_transfer_latency_ms)
        self._log_gauge(self.decode_cache_hit_rate, stats.decode_cache_hit_rate)
        self._log_gauge(self.hicache_l1_hit_rate, stats.hicache_l1_hit_rate)
        self._log_gauge(self.hicache_l2_hit_rate, stats.hicache_l2_hit_rate)
        self._log_gauge(self.hicache_l3_hit_rate, stats.hicache_l3_hit_rate)

        # Retract
        self._log_gauge(self.total_retracted_reqs, stats.total_retracted_reqs)
        self._log_gauge(self.num_retracted_reqs, stats.num_retracted_reqs)
        self._log_gauge(self.num_paused_reqs, stats.num_paused_reqs)

    def _log_cat(self, stats: SchedulerStats, is_prefill: bool = False) -> None:
        self.cat_reporter.log_count("RunningBS", stats.num_running_reqs)
        self.cat_reporter.log_count("NumUsedTokens", stats.num_used_tokens)
        self.cat_reporter.log_count("TokenUsage", stats.token_usage)
        self.cat_reporter.log_count("NumWaitingReqs", stats.num_queue_reqs)
        if is_prefill:
            self.cat_reporter.log_count("CacheHitRate", stats.cache_hit_rate * 100)
            if stats.dequeue_time > 0.0:
                self.cat_reporter.log_duration("DequeueTime", stats.dequeue_time)
            # PD disaggregation metrics
            self.cat_reporter.log_count("PrefillPreallocQueueReqs", stats.num_prefill_prealloc_queue_reqs)
            self.cat_reporter.log_count("PrefillInflightQueueReqs", stats.num_prefill_inflight_queue_reqs)
            # HiCache metrics
            self.cat_reporter.log_count("HiCacheL1HitRate", stats.hicache_l1_hit_rate * 100)
            self.cat_reporter.log_count("HiCacheL2HitRate", stats.hicache_l2_hit_rate * 100)
            self.cat_reporter.log_count("HiCacheL3HitRate", stats.hicache_l3_hit_rate * 100)
        else:
            self.cat_reporter.log_count("GenThroughput", stats.gen_throughput)
            if stats.spec_accept_length != 0:
                self.cat_reporter.log_count("SpecAccpetLen", stats.spec_accept_length)
            # PD disaggregation metrics
            self.cat_reporter.log_count("DecodePreallocQueueReqs", stats.num_decode_prealloc_queue_reqs)
            self.cat_reporter.log_count("DecodeTransferQueueReqs", stats.num_decode_transfer_queue_reqs)
            self.cat_reporter.log_count("DecodeCacheHitRate", stats.decode_cache_hit_rate * 100)

        #   9.22 SGLang open source community has not yet implemented actual recording of KVTransferSpeedGBs and other data, pending future cherry-pick
        #   self.cat_reporter.log_count("KVTransferSpeedGBs", stats.kv_transfer_speed_gb_s)
        #   self.cat_reporter.log_duration("KVTransferLatencyMs", stats.kv_transfer_latency_ms / 1000.0)  # Convert ms to seconds for duration
        # Retract metrics
            self.cat_reporter.log_count("TotalRetractedReqs", stats.total_retracted_reqs)
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

    def observe_module_time_in_scheduler(self, step_time: float, name: str="scheduler_module"):
        if self.enable_cat:
            self.cat_reporter.log_duration(name, step_time)


class TokenizerMetricsCollector:
    def __init__(self, labels: Dict[str, str], metrics_reporters: List[str]) -> None:
        self.enable_prometheus = 'prometheus' in metrics_reporters
        self.enable_cat = 'cat' in metrics_reporters
        self.enable_llm_platform = 'llm-platform' in metrics_reporters

        if self.enable_prometheus:
            self._init_prometheus(labels)

        if self.enable_cat:
            self.cat_reporter = get_cat_reporter(labels.get('model_name', 'DefalutModel'), labels.get('app_key', 'DefaultAppKey'))

    def _init_prometheus(self, labels: Dict[str, str]) -> None:
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

    def _log_histogram(self, histogram, data: Union[int, float]) -> None:
        if self.enable_prometheus:
            histogram.labels(**self.labels).observe(data)

    def observe_one_finished_request(
        self,
        prompt_tokens: int,
        generation_tokens: int,
        e2e_latency: float,
        tokenized_duration: float,
    ):
        if self.enable_prometheus:
            self.prompt_tokens_total.labels(**self.labels).inc(prompt_tokens)
            self.generation_tokens_total.labels(**self.labels).inc(generation_tokens)
            self.num_requests_total.labels(**self.labels).inc(1)
            self._log_histogram(self.histogram_e2e_request_latency, e2e_latency)
            if generation_tokens >= 1:
                self.histogram_time_per_output_token.labels(**self.labels).observe(
                    e2e_latency / generation_tokens
                )

        if self.enable_cat:
            self.cat_reporter.log_count("PromptTokens", prompt_tokens)
            self.cat_reporter.log_count("CompletionTokens", generation_tokens)
            self.cat_reporter.log_duration("RequestTime", e2e_latency)
            self.cat_reporter.log_duration("TokenizedTime", tokenized_duration)

    def observe_time_to_first_token(self, value: float, name: str = "TTFT"):
        if self.enable_prometheus:
            self.histogram_time_to_first_token.labels(**self.labels).observe(value)

        if self.enable_cat:
            self.cat_reporter.log_duration(name, value)

    def observe_inter_token_latency(self, internval: float, num_new_tokens: int, name: str="TPOT"):
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

    def observe_request_arrival(self, batch_size: int = 1):
        if self.enable_prometheus:
            pass

        if self.enable_cat:
            self.cat_reporter.log_count("RequestArrival", batch_size)

class ErrorMetricsCollector:
    def __init__(self, labels: Dict[str, str], metrics_reporters: List[str]) -> None:
        self.enable_prometheus = 'prometheus' in metrics_reporters
        self.enable_cat = 'cat' in metrics_reporters
        self.enable_llm_platform = 'llm-platform' in metrics_reporters

        if self.enable_cat:
            self.cat_reporter = get_cat_reporter(labels.get('model_name', 'DefaultModel'), labels.get('app_key', 'DefaultAppKey'))

    def record_error(self, error_message: str) -> None:
        # Currently only cat logging is supported
        if not self.enable_cat:
            return

        if "KVTransferError" in error_message:
            match = re.search(r'remote_endpoint=([^,\)]+)', error_message)
            remote_dir = match.group(1) if match else "not found"
            formatted_error = f"KVTransferError, remote_dir={remote_dir}"
        else:
            formatted_error = "OtherError"

        self.cat_reporter.log_error(formatted_error)

class KVTransferMetricsCollector:

    def __init__(self, labels: Dict[str, str], metrics_reporters: List[str]) -> None:
        self.enable_cat = 'cat' in metrics_reporters

        if self.enable_cat:
            self.cat_reporter = get_cat_reporter(
                labels.get('model_name', 'DefaultModel'),
                labels.get('app_key', 'DefaultAppKey')
            )

    def log_kv_transfer_timeout(self) -> None:
        if not self.enable_cat:
            return

        self.cat_reporter.log_count(f"KVTransferTimeout", 1)

    def log_kv_transfer_failed(self) -> None:
        if not self.enable_cat:
            return

        self.cat_reporter.log_count(f"KVTransferFailed", 1)

    def log_kv_transfer_size(self, transfer_size_bytes: int) -> None:
        if not self.enable_cat:
            return

        self.cat_reporter.log_count("KVTransferSizeBytes", transfer_size_bytes)

    def log_kv_transfer_time(self, transfer_time: float) -> None:
        if not self.enable_cat:
            return

        self.cat_reporter.log_duration("KVTransferTime", transfer_time)

class EPLBMetricsCollector:
    def __init__(self, labels: Dict[str, str], metrics_reporters: List[str]) -> None:
        self.enable_cat = 'cat' in metrics_reporters

        if self.enable_cat:
            self.cat_reporter = get_cat_reporter(
                labels.get('model_name', 'DefaultModel'),
                labels.get('app_key', 'DefaultAppKey')
            )

    def log_gpu_unbalancedness(self, per_device_metrics) -> None:
        if not self.enable_cat:
            return

        self.cat_reporter.log_duration(f"PerDeviceUnbalancedness", per_device_metrics * 100)

    def log_expert_unbalancedness(self, per_expert_metrics) -> None:
        if not self.enable_cat:
            return

        self.cat_reporter.log_duration(f"PerExpertUnbalancedness", per_expert_metrics * 100)

@dataclass
class StorageMetrics:
    prefetch_pgs: List[int] = field(default_factory=list)
    backup_pgs: List[int] = field(default_factory=list)
    prefetch_bandwidth: List[float] = field(default_factory=list)
    backup_bandwidth: List[float] = field(default_factory=list)


class StorageMetricsCollector:
    def __init__(
        self,
        labels: Dict[str, str],
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
