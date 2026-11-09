import logging
import time
import warnings
from typing import List, TYPE_CHECKING

import torch

from sglang.srt.managers.req import Req
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.utils import get_device_module, crash_on_warnings, check_memory_debug, get_bool_env_var, get_colorful_logger
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE

logger = get_colorful_logger(__name__)

RECORD_STEP_TIME = get_bool_env_var("SGLANG_RECORD_STEP_TIME")

if TYPE_CHECKING:
    from sglang.srt.managers.scheduling.scheduler import Scheduler as scheduler

class SchedulerStatsMixin:

    def log_prefill_stats(
        self: "scheduler",
        adder: PrefillAdder,
        running_bs: int,
        batch_it: int
    ):
        can_run_list = adder.can_run_list
        dequeue_durations = adder.dequeue_durations
        num_used_pages = self._get_num_used_pages()
        self._largest_prefill_len = max(
            self._largest_prefill_len, adder.log_input_tokens
        )
        dequeue_time = (
            (sum(dequeue_durations) / len(dequeue_durations))
            if len(dequeue_durations) > 0
            else 0.0
        )

        f = (
            f"Prefill batch. {batch_it}. "
            f"#new-seq: {len(can_run_list)}, "
            f"#new-token: {adder.log_input_tokens}, "
            f"#cached-token: {adder.log_hit_tokens}, "
            f"page usage: {num_used_pages / self.max_total_page_num:.2f}, "
            f"#running-req: {running_bs}, "
        )

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            f += f"#unbootstrapped-req: {len(self.disagg_prefill_bootstrap_queue.queue)}, "
            f += f"#queue-req: {len(self.waiting_queue)}, "
            f += f"#transferring-req: {len(self.disagg_prefill_inflight_queue)}, "
        else:
            f += f"#queue-req: {len(self.waiting_queue)}, "

        f += f"#dequeue-time: {dequeue_time * 1000:.2f}"

        logger.info(f)

        if self.enable_metrics:
            self.stats.num_running_reqs = running_bs
            self.stats.running_bs_rate = (
                running_bs / self.max_running_requests * 100
                if self.max_running_requests > 0
                else 0.0
            )
            self.stats.num_used_tokens = num_used_pages * self.server_args.page_size
            self.stats.num_free_tokens = (self.kv_allocator.available_size() + self.tree_cache.evictable_size()) * self.server_args.page_size
            self.stats.token_usage = round(num_used_pages * self.server_args.page_size / self.max_total_num_tokens, 2)
            self.stats.num_queue_reqs = len(self.waiting_queue)
            self.stats.dequeue_time = dequeue_time
            self.stats.gen_throughput = self.last_gen_throughput
            self.stats.prefill_chunk_tokens = adder.log_input_tokens
            self.stats.total_retracted_reqs = self.total_retracted_reqs
            self.stats.num_retracted_reqs = self.num_retracted_reqs
            self.stats.num_retracted_queue_reqs = 0
            self.stats.avg_request_queue_latency = 0.0
            self.stats.num_grammar_queue_reqs=len(self.grammar_queue)
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                self.stats.num_prefill_prealloc_queue_reqs = len(
                    self.disagg_prefill_bootstrap_queue.queue
                )
                self.stats.num_prefill_inflight_queue_reqs = len(
                    self.disagg_prefill_inflight_queue
                )
            if self.disaggregation_mode == DisaggregationMode.DECODE:
                self.stats.num_decode_prealloc_queue_reqs = len(
                    self.disagg_decode_prealloc_queue.queue
                )
                self.stats.num_decode_transfer_queue_reqs = len(
                    self.disagg_decode_transfer_queue.queue
                )
                self.stats.num_retracted_queue_reqs = len(
                    self.disagg_decode_prealloc_queue.retracted_queue
                )
            self.metrics_collector.log_stats(self.stats, is_prefill=True)

        # Publish KV cache events after prefill
        self.kv_event_publisher.publish_kv_events()
        self.kv_event_publisher.emit_kv_metrics(self.stats)

    def log_prefill_mfu(self: "scheduler"):
        if getattr(self, "mfu_runner", None) is None:
            return
        prefill_mfu_queue = getattr(self, "_prefill_mfu_queue", None)
        if not prefill_mfu_queue:
            return

        if len(prefill_mfu_queue) < 1:
            logger.warning("Prefill MFU queue is empty when logging MFU.")
            raise ValueError("Prefill MFU queue is empty when logging MFU.")

        launch_time, extend_lens, seq_lens = prefill_mfu_queue.popleft()
        elapsed = time.perf_counter() - launch_time
        if not extend_lens or elapsed <= 0:
            return

        result = self.mfu_runner.measure(
            num_tokens=extend_lens,
            seq_lens=seq_lens,
            elapsed_seconds=elapsed,
        )
        if not result:
            return

        tflops_per_sec = result["tflops_per_sec"]
        mfu = result["mfu"]
        logger.info(
            f"Prefill MFU. elapsed_ms={elapsed * 1000:.2f}, "
            f"tflops/s={tflops_per_sec:.2f}, mfu={mfu * 100:.2f}%"
        )

        if self.enable_metrics:
            self.stats.prefill_mfu = mfu
            self.stats.prefill_tflops_per_sec = tflops_per_sec
            self.metrics_collector.observe_mfu(mfu, tflops_per_sec)

    def log_decode_stats(self: "scheduler"):
        gap_latency = time.time() - self.last_decode_stats_tic
        self.last_decode_stats_tic = time.time()
        self.last_gen_throughput = self.num_generated_tokens / gap_latency
        self.num_generated_tokens = 0
        num_running_reqs = len(self.running_batch.reqs) if self.running_batch else 0
        seq_lens = [req.seqlen for req in self.running_batch.reqs] if self.running_batch else [0]
        avg_seq_len = sum(seq_lens) / len(seq_lens)
        num_used_pages = self._get_num_used_pages()

        if RECORD_STEP_TIME:
            self.step_time_dict[num_running_reqs].append(
                gap_latency / self.server_args.decode_log_interval
            )

        msg = (
            f"Decode batch: {self.forward_ct_decode},"
            f"#running-req: {num_running_reqs}, "
            f"#used page num: {num_used_pages}, "
            f"page usage: {num_used_pages / self.max_total_page_num:.2f}, "
            f"itl: {gap_latency}, "
        )

        if self.spec_algorithm.is_none():
            spec_accept_length = 0
        else:
            spec_accept_length = (
                self.spec_num_total_accepted_tokens / self.spec_num_total_forward_ct
            )
            self.cum_spec_accept_length += self.spec_num_total_accepted_tokens
            self.cum_spec_accept_count += self.spec_num_total_forward_ct
            self.spec_num_total_accepted_tokens = self.spec_num_total_forward_ct = 0
            msg += f"accept len: {spec_accept_length:.2f}, "

        if self.disaggregation_mode == DisaggregationMode.DECODE:
            msg += f"pre-allocated usage: {self.num_tokens_pre_allocated / self.max_total_num_tokens:.2f}, "
            msg += f"#retracted-req: {len(self.disagg_decode_prealloc_queue.retracted_queue)}, "
            # Calculate decode cache hit rate for display
            decode_cache_hit_rate = 0.0
            if self.disagg_decode_prealloc_queue.total_decode_tokens > 0:
                decode_cache_hit_rate = (
                    self.disagg_decode_prealloc_queue.total_decode_hit_tokens /
                    self.disagg_decode_prealloc_queue.total_decode_tokens
                )
            msg += f"decode cache hit rate: {decode_cache_hit_rate:.2%}, "
            msg += f"#prealloc-req: {len(self.disagg_decode_prealloc_queue.queue)}, "
            msg += f"#transfer-req: {len(self.disagg_decode_transfer_queue.queue)}, "

        msg += (
            f"avg_seq_len: {avg_seq_len}, "
            f"gen throughput (token/s): {self.last_gen_throughput:.2f}, "
            f"#queue-req: {len(self.waiting_queue)}"
        )

        logger.info(msg)
        if self.enable_metrics:
            self.stats.num_running_reqs = num_running_reqs
            self.stats.running_bs_rate = (
                num_running_reqs / self.max_running_requests * 100
                if self.max_running_requests > 0
                else 0.0
            )
            self.stats.num_used_tokens = num_used_pages * self.server_args.page_size
            self.stats.num_free_tokens = (self.kv_allocator.available_size() + self.tree_cache.evictable_size()) * self.server_args.page_size
            self.stats.token_usage = num_used_pages * self.server_args.page_size / self.max_total_num_tokens
            self.stats.gen_throughput = self.last_gen_throughput
            self.stats.num_queue_reqs = len(self.waiting_queue)
            self.stats.spec_accept_length = spec_accept_length
            self.stats.gen_throughput = self.last_gen_throughput
            self.stats.total_retracted_reqs = self.total_retracted_reqs
            self.stats.num_retracted_reqs = self.num_retracted_reqs
            self.num_retracted_reqs = 0
            self.stats.num_retracted_queue_reqs = 0
            self.stats.avg_request_queue_latency = 0.0
            self.stats.decode_idle_ratio = getattr(self, "decode_idle_ratio", 0.0)
            self.stats.decode_busy_ratio = getattr(self, "decode_busy_ratio", 0.0)
            self.stats.num_grammar_queue_reqs=len(self.grammar_queue)
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                self.stats.num_prefill_prealloc_queue_reqs = len(
                    self.disagg_prefill_bootstrap_queue.queue
                )
                self.stats.num_prefill_inflight_queue_reqs = len(
                    self.disagg_prefill_inflight_queue
                )
            if self.disaggregation_mode == DisaggregationMode.DECODE:
                self.stats.num_decode_prealloc_queue_reqs = len(
                    self.disagg_decode_prealloc_queue.queue
                )
                self.stats.num_decode_transfer_queue_reqs = len(
                    self.disagg_decode_transfer_queue.queue
                )
                self.stats.num_retracted_queue_reqs = len(
                    self.disagg_decode_prealloc_queue.retracted_queue
                )
                # Get decode cache hit rate
                self.stats.decode_cache_hit_rate = (
                    self.disagg_decode_prealloc_queue.get_and_reset_decode_cache_hit_rate()
                )
            self.metrics_collector.log_stats(self.stats)

        # Publish KV cache events after decode
        self.kv_event_publisher.publish_kv_events()
        self.kv_event_publisher.emit_kv_metrics(self.stats)

    def log_memory_stats(self: "scheduler"):
        """Probe device memory occupancy and report it via the metrics collector.
        """
        if not self.enable_metrics:
            return
        try:
            device_module = get_device_module()
            if device_module is None:
                return
            alloc = device_module.memory_allocated()
            reserved = device_module.memory_reserved()
            free, total = device_module.mem_get_info()
            phys_used = total - free
            non_torch = phys_used - reserved
            self.stats.mem_alloc_mb = alloc / 1048576
            self.stats.mem_reserved_mb = reserved / 1048576
            self.stats.mem_phys_used_mb = phys_used / 1048576
            self.stats.mem_non_torch_mb = non_torch / 1048576
            self.metrics_collector.log_memory_stats(self.stats)
        except Exception as e:
            logger.warning(f"log_memory_stats failed: {e}")

    def log_idle_stats(self: "scheduler"):
        if not self.enable_metrics or self.attn_tp_rank != 0:
            return

        if time.time() <= self.metrics_collector.last_log_time + 30:
            return

        num_used_pages = self._get_num_used_pages()
        num_running_reqs = len(self.running_batch.reqs) if self.running_batch else 0
        self.stats.num_running_reqs = num_running_reqs
        self.stats.running_bs_rate = (
            num_running_reqs / self.max_running_requests * 100
            if self.max_running_requests > 0
            else 0.0
        )
        self.stats.num_used_tokens = num_used_pages * self.server_args.page_size
        self.stats.num_free_tokens = (self.kv_allocator.available_size() + self.tree_cache.evictable_size()) * self.server_args.page_size
        self.stats.token_usage = self.stats.num_used_tokens / self.max_total_num_tokens
        self.stats.num_queue_reqs = len(self.waiting_queue)
        self.stats.gen_throughput = self.last_gen_throughput
        self.stats.total_retracted_reqs = self.total_retracted_reqs
        self.stats.num_retracted_reqs = self.num_retracted_reqs
        self.num_retracted_reqs = 0
        self.stats.num_retracted_queue_reqs = 0
        self.stats.avg_request_queue_latency = 0.0
        self.stats.spec_accept_length = 0
        self.stats.num_grammar_queue_reqs=len(self.grammar_queue)
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.stats.num_prefill_prealloc_queue_reqs = len(
                self.disagg_prefill_bootstrap_queue.queue
            )
            self.stats.num_prefill_inflight_queue_reqs = len(
                self.disagg_prefill_inflight_queue
            )
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            self.stats.num_decode_prealloc_queue_reqs = len(
                self.disagg_decode_prealloc_queue.queue
            )
            self.stats.num_decode_transfer_queue_reqs = len(
                self.disagg_decode_transfer_queue.queue
            )
            self.stats.num_retracted_queue_reqs = len(
                self.disagg_decode_prealloc_queue.retracted_queue
            )
        self.metrics_collector.log_stats(self.stats)
        self.kv_event_publisher.emit_kv_metrics(self.stats)

    def check_memory(self: "scheduler"):

        if GPU_MEMORY_TYPE_KV_CACHE in self.offload_tags:
            return

        if (
            self.decode_offload_manager is not None
            and self.decode_offload_manager.has_inflight()
        ):
            return

        # available_size is total num of free page id
        available_size = (
            self.kv_allocator.available_size() + self.tree_cache.evictable_size()
        )
        protected_size = self.tree_cache.protected_size()
        cache_memory_leak = available_size != (
            self.max_total_page_num
            if not self.enable_hierarchical_cache
            else self.max_total_page_num - protected_size
        )
        req_pool_leak = len(self.req_to_token_pool.free_slots) != self.req_to_token_pool.size - 1
        if cache_memory_leak:
            diff = available_size - self.max_total_page_num
            msg = (
                f"KV cache pool leak detected! diff={diff}\n"
                f"token_to_kv_pool avail {self.kv_allocator.available_size()}, tree_cache evictable_size {self.tree_cache.evictable_size()}\n"
                f"{protected_size=}, max_total_page_num={self.max_total_page_num}, max_total_num_tokens={self.max_total_num_tokens} \n"
            )
            free_slots = self.kv_allocator.free_slots
            unique_elems, counts = torch.unique(free_slots, return_counts=True)
            if free_slots.shape[0] != unique_elems.shape[0]:
                duplicates = unique_elems[counts > 1]
                warnings.warn(f"Dup Free! free_slots: {free_slots.shape[0]} unique: {unique_elems.shape[0]} {duplicates=}")
            warnings.warn(msg)
            if crash_on_warnings():
                raise ValueError(msg)

        if req_pool_leak:
            if len(self.req_to_token_pool.free_slots) < self.req_to_token_pool.size - 1:
                msg = (
                    "Memory pool leak detected!"
                    f"available_size={len(self.req_to_token_pool.free_slots)}, "
                    f"total_size={self.req_to_token_pool.size}\n"
                    f"free_slots={self.req_to_token_pool.free_slots}"
                )
                warnings.warn(msg)
                if crash_on_warnings():
                    raise ValueError(msg)
            else:
                # Duplicate free detected
                warnings.warn(f"Req Pool Duplicated free Detected! free_slots={self.req_to_token_pool.free_slots}")

        if self.enable_hierarchical_cache:
            host_allocator = self.tree_cache.l2_ctrl.host_kv_allocator
            host_free = host_allocator.available_size()
            host_total = host_allocator.global_page_num
            host_tree_pages = self.tree_cache._get_all_node_host_value()
            host_in_tree = len(host_tree_pages) if host_tree_pages is not None else 0
            l3_ctrl = self.tree_cache.l3_ctrl
            prefetch_occupied = 0
            if l3_ctrl is not None:
                prefetch_occupied = l3_ctrl.prefetch_tokens_occupied
            if prefetch_occupied == 0:
                offload_inflight = 0
                if self.decode_offload_manager is not None:
                    offload_inflight = self.decode_offload_manager.inflight_host_pages
                host_accounted = host_free + host_in_tree + offload_inflight
                if host_accounted != host_total:
                    diff = host_accounted - host_total
                    msg = (
                        f"Host KV cache pool leak detected! diff={diff}\n"
                        f"host_free={host_free}, host_in_tree={host_in_tree}, "
                        f"offload_inflight={offload_inflight}, host_total={host_total}\n"
                    )
                    warnings.warn(msg)
                    if crash_on_warnings():
                        raise ValueError(msg)
            host_free_slots = host_allocator.free_slots
            host_unique, host_counts = torch.unique(host_free_slots, return_counts=True)
            if host_free_slots.shape[0] != host_unique.shape[0]:
                duplicates = host_unique[host_counts > 1]
                warnings.warn(
                    f"Host Dup Free! free_slots: {host_free_slots.shape[0]} "
                    f"unique: {host_unique.shape[0]} {duplicates=}"
                )

        slots_refs = self.kv_allocator.token_slot_refs.cpu()
        radix_pages = self.tree_cache._get_all_node_value()
        if check_memory_debug() and radix_pages is not None:
            radix_pages = radix_pages.cpu()
            token_level_offsets = torch.arange(self.kv_allocator.page_size, dtype=torch.int32)
            radix_slots = (radix_pages[:, None] * self.kv_allocator.page_size + token_level_offsets).flatten()
            free_pages = self.kv_allocator.free_slots.cpu()
            free_token_slots = (free_pages[:, None] * self.kv_allocator.page_size + token_level_offsets).flatten()
            set1 = set(radix_slots.flatten().tolist())
            set2 = set(free_token_slots.flatten().tolist())
            # Positions in radix cache should not appear in free slots
            if bool(set1 & set2):
                warnings.warn(f"Found radix slots in free slots! {set1 & set2=}")

            # Slots not accepted are reused during speculation, here only check non-speculation case
            if not self.draft_worker:
                # Positions in radix cache should have ref count of 1
                if torch.any(slots_refs[radix_slots] != 1):
                    indices = torch.nonzero((slots_refs[radix_slots] != 1), as_tuple=False).squeeze()
                    warnings.warn(f"Radix slots ref not 1! {radix_slots[indices]=}  {slots_refs[radix_slots]=}")
                # Allocatable slots should have ref count of 0 at this time
                if torch.any(slots_refs[free_token_slots] != 0):
                    non_zeros_elems = (slots_refs[free_token_slots] != 0).sum()
                    non_zeros_slots = free_token_slots[slots_refs[free_token_slots] != 0]
                    warnings.warn(f"Non Radix slots ref is not 0! {non_zeros_elems=} {non_zeros_slots=}")

        # Check if verified_lens is 0 when server is idle (req_pool_idx 0 is used for padding)
        # This affects the position to write kv cache
        try:
            if len(self.req_to_token_pool.verified_lens) > 1:
                if torch.any(self.req_to_token_pool.verified_lens[1:] != 0):
                    warnings.warn(f"verified_lens has non-zero element! {self.req_to_token_pool.verified_lens}")
            elif len(self.req_to_token_pool.verified_lens) == 1:
                # If there's only one element, check if it's 0 (skip index 0 as it's padding)
                if self.req_to_token_pool.verified_lens[0] != 0:
                    warnings.warn(f"verified_lens has non-zero element! {self.req_to_token_pool.verified_lens}")
        except (RuntimeError, torch.AcceleratorError) as e:
            warnings.warn(f"Failed to check verified_lens: {e}")
