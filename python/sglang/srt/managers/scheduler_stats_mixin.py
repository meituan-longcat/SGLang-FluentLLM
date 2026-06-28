import logging
import time
import warnings
from typing import List

import torch

from sglang.srt.managers.req import Req
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.utils import crash_on_warnings, check_memory_debug, get_bool_env_var
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE

logger = logging.getLogger(__name__)

RECORD_STEP_TIME = get_bool_env_var("SGLANG_RECORD_STEP_TIME")


class SchedulerStatsMixin:

    def log_prefill_stats(
        self,
        adder: PrefillAdder,
        can_run_list: List[Req],
        running_bs: int,
        dequeue_durations: List[float],
    ):
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
            f"Prefill batch. "
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
            total_tokens = max(
                adder.log_input_tokens + adder.log_hit_tokens, 1
            )

            logger.debug(
                f"[log_prefill_stats] log_l1_hit_tokens={adder.log_l1_hit_tokens} log_l2_hit_tokens={adder.log_l2_hit_tokens} "
                f"log_l3_hit_tokens={adder.log_l3_hit_tokens} log_input_tokens={adder.log_input_tokens} total_tokens={total_tokens}"
            )

            if total_tokens > 0:
                cache_hit_rate = adder.log_hit_tokens / total_tokens
                self.stats.hicache_l1_hit_rate = adder.log_l1_hit_tokens / total_tokens
                self.stats.hicache_l2_hit_rate = adder.log_l2_hit_tokens / total_tokens
                self.stats.hicache_l3_hit_rate = adder.log_l3_hit_tokens / total_tokens
                # Note: With the fix in schedule_policy.py, cumulative_l1_hit_tokens now correctly
                # represents incremental L1 hits per prefill, not cumulative cache size
            else:
                cache_hit_rate = 0.0
                self.stats.hicache_l1_hit_rate = 0.0
                self.stats.hicache_l2_hit_rate = 0.0
                self.stats.hicache_l3_hit_rate = 0.0
            logger.debug(
                f"[log_prefill_metrics] cache_hit_rate={cache_hit_rate} {self.stats.hicache_l1_hit_rate=} {self.stats.hicache_l2_hit_rate=} {self.stats.hicache_l3_hit_rate=}"
            )

            self.stats.num_running_reqs = running_bs
            self.stats.num_used_tokens = num_used_pages * self.server_args.page_size
            self.stats.token_usage = round(num_used_pages * self.server_args.page_size / self.max_total_num_tokens, 2)
            self.stats.num_queue_reqs = len(self.waiting_queue)
            self.stats.cache_hit_rate = cache_hit_rate
            self.stats.dequeue_time = dequeue_time
            self.stats.gen_throughput = self.last_gen_throughput
            self.stats.total_retracted_reqs = self.total_retracted_reqs
            self.stats.avg_request_queue_latency = 0.0
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
            self.metrics_collector.log_stats(self.stats, is_prefill=True)

        # Publish KV cache events after prefill
        self._publish_kv_events()

    def log_decode_stats(self):
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
            f"Decode batch. "
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
            self.stats.num_used_tokens = num_used_pages * self.server_args.page_size
            self.stats.token_usage = num_used_pages * self.server_args.page_size / self.max_total_num_tokens
            self.stats.cache_hit_rate = 0.0
            self.stats.gen_throughput = self.last_gen_throughput
            self.stats.num_queue_reqs = len(self.waiting_queue)
            self.stats.spec_accept_length = spec_accept_length
            self.stats.gen_throughput = self.last_gen_throughput
            self.stats.total_retracted_reqs = self.total_retracted_reqs
            self.stats.avg_request_queue_latency = 0.0
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
                # Get decode cache hit rate
                self.stats.decode_cache_hit_rate = (
                    self.disagg_decode_prealloc_queue.get_and_reset_decode_cache_hit_rate()
                )
            self.metrics_collector.log_stats(self.stats)

        # Publish KV cache events after decode
        self._publish_kv_events()

    def log_idle_stats(self):
        if not self.enable_metrics or self.attn_tp_rank != 0:
            return

        if time.time() <= self.metrics_collector.last_log_time + 30:
            return

        num_used = self._get_num_used_pages()
        num_running_reqs = len(self.running_batch.reqs) if self.running_batch else 0
        self.stats.num_running_reqs = num_running_reqs
        self.stats.num_used_tokens = num_used
        self.stats.token_usage = num_used / self.max_total_num_tokens
        self.stats.num_queue_reqs = len(self.waiting_queue)
        self.stats.gen_throughput = self.last_gen_throughput
        self.stats.total_retracted_reqs = self.total_retracted_reqs
        self.stats.avg_request_queue_latency = 0.0
        self.stats.spec_accept_length = 0
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
        self.metrics_collector.log_stats(self.stats)

    def check_memory(self):

        if GPU_MEMORY_TYPE_KV_CACHE in self.offload_tags:
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
                logger.warning(f"Dup Free! free_slots: {free_slots.shape[0]} unique: {unique_elems.shape[0]} {duplicates=}")
            logger.warning(msg)
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
                logger.warning(msg)
                if crash_on_warnings():
                    raise ValueError(msg)
            else:
                # Duplicate free detected
                warnings.warn(f"Req Pool Duplicated free Detected! free_slots={self.req_to_token_pool.free_slots}")

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
