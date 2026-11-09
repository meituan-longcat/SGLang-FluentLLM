from __future__ import annotations

import os
import random
import time
from typing import TYPE_CHECKING

from sglang.srt.mem_cache.hicache_operation import H2SWrite
from sglang.srt.utils import get_colorful_logger

if TYPE_CHECKING:
    from sglang.srt.managers.req import Req
    from sglang.srt.mem_cache.radix_cache import TreeNode

logger = get_colorful_logger(__name__)


class L3MixinPrefetch:
    def prepare_prefetch(
        self,
        req: Req,
        last_host_node: TreeNode,
        new_input_tokens: list[int],
        last_hash: str | None = None,
        prefix_keys: list[str] | None = None,
    ) -> bool:
        prefetch_length = len(new_input_tokens) - (len(new_input_tokens) % self.page_size)
        if prefetch_length < self.prefetch_threshold:
            logger.debug(f"[L3-Prefetch-Skip] rid={req.rid} prefetch_length={prefetch_length} < threshold={self.prefetch_threshold}")
            return False

        if self.l3_ctrl.prefetch_rate_limited():
            logger.debug(f"[L3-Prefetch-Skip] rid={req.rid} rate_limited")
            return False

        if last_host_node is not None:
            last_host_node.protect_host()
        tokens = new_input_tokens[:prefetch_length]
        s2h_load = self.l3_ctrl.create_s2h_load(req.rid, tokens, last_hash, prefix_keys, last_host_node)

        req.s2h_load = s2h_load
        # query storage get hit_pages
        s2h_load.t_query_enq = time.monotonic()
        if getattr(self.l3_ctrl, "prefetch_stop_policy", "wait_complete") == "timeout":
            timeout = self.l3_ctrl.prefetch_timeout_base + prefetch_length // self.page_size * self.l3_ctrl.prefetch_timeout_per_page
            if bool(os.getenv("TEST_PREFETCH_RANDOM_TIMEOUT", "")):
                timeout = random.uniform(0, 0.2)  # noqa: S311
            else:
                timeout = min(
                    getattr(self.l3_ctrl, "prefetch_timeout_max", 30.0),
                    timeout,
                )
            s2h_load.set_prefetch_deadline(s2h_load.t_query_enq + timeout)
        self.l3_ctrl.enqueue_query(s2h_load)
        return True

    def start_prefetch(self, req: Req) -> bool:
        s2h_load = req.s2h_load
        hit_pages = s2h_load.hit_pages
        page2rank = s2h_load.page2rank
        min_hit_pages = max(1, self.prefetch_threshold // self.page_size)
        if hit_pages < min_hit_pages:
            s2h_load.mark_terminate()
            logger.debug(f"[L3-Prefetch-Skip] rid={req.rid} hit_pages={hit_pages} < min={min_hit_pages}")
            return True

        host_page_indices = self.alloc_host_pages(page2rank)
        if host_page_indices is None:
            s2h_load.mark_terminate()
            logger.debug(f"[L3-Prefetch-Skip] rid={req.rid} host_page alloc failed, pages={hit_pages}")
            return True

        # If partial alloc: truncate to actually allocated pages.
        # Reference pattern: hiradix_cache.py prefetch_from_storage falls back to
        # available_size - (available_size % page_size) when eviction is insufficient.
        allocated_pages = len(host_page_indices)
        if allocated_pages < hit_pages:
            logger.info(f"[L3-Prefetch-Partial] rid={req.rid} requested={hit_pages} allocated={allocated_pages}")
            hit_pages = allocated_pages
            if hit_pages < min_hit_pages:
                self.l2_ctrl.host_kv_allocator.free_pages(host_page_indices)
                s2h_load.mark_terminate()
                logger.debug(f"[L3-Prefetch-Skip] rid={req.rid} partial pages={hit_pages} < min={min_hit_pages}")
                return True
        host_token_indices = self.l2_ctrl.host_kv_pool.pages_to_tokens(host_page_indices)
        s2h_load.set_allocate_result(hit_pages, hit_pages * self.page_size, host_page_indices, host_token_indices)
        logger.info(f"[L3-Prefetch-Enqueue] rid={req.rid} pages={hit_pages} tokens={hit_pages * self.page_size}")
        # start prefetch
        s2h_load.t_read_enq = time.monotonic()
        reserved_tokens = hit_pages * self.page_size
        self.l3_ctrl.reserve_prefetch_tokens(reserved_tokens)
        s2h_load.set_reserved_tokens(reserved_tokens)
        self.l3_ctrl.s2h_load_q_todo.put(s2h_load)
        return True

    def finalize_prefetch(self, req: Req):
        s2h_load = req.s2h_load
        protected_host_node = s2h_load.last_host_node
        last_host_node = protected_host_node

        completed_tokens = s2h_load.get_synced_completed_tokens()
        if completed_tokens == -1:
            # abort
            if s2h_load.host_page_indices is not None:
                self.l2_ctrl.host_kv_allocator.free_pages(s2h_load.host_page_indices)
        else:
            if last_host_node is None:
                # No anchor in tree (host+device both evicted).
                # Insert at root so the prefetched data is usable.
                last_host_node = self.root_node
            host_page_indices = s2h_load.host_page_indices

            completed_pages = completed_tokens // self.page_size

            token_ids = s2h_load.token_ids[:completed_tokens]
            paged_token_ids = [tuple(token_ids[i * self.page_size : (i + 1) * self.page_size]) for i in range(completed_pages)]

            hash_value = s2h_load.hash_value[:completed_pages]
            matched_length = self.insert_host(
                last_host_node,
                key=paged_token_ids,
                host_value=host_page_indices[:completed_pages],
                hash_value=hash_value,
            )

            if matched_length > 0:
                self.l2_ctrl.host_kv_allocator.free_pages(host_page_indices[:matched_length])
            if completed_pages < len(host_page_indices):
                tail = host_page_indices[completed_pages:]
                self.l2_ctrl.host_kv_allocator.free_pages(tail)

            logger.info(f"[L3-Prefetch-Done] req={req.rid} completed_pages={completed_pages} inserted_pages={completed_pages - matched_length}")

            if self.enable_storage_metrics:
                self.storage_metrics_collector.log_prefetched_tokens((completed_pages - matched_length) * self.page_size)
            self.l3_ctrl.mark_prefetch_ready(req, completed_tokens)

        if protected_host_node is not None:
            protected_host_node.release_host()
        self.l3_ctrl.release_prefetch_tokens(s2h_load.reserved_tokens)
        s2h_load.t_finish_done = time.monotonic()
        req.s2h_load = None

    def check_l3_progress(
        self,
    ):
        l3 = self.l3_ctrl
        return [l3.h2s_write_q_done.qsize()]

    def process_l3_feedback(self, finish_counts):
        l3 = self.l3_ctrl
        self.process_done_writes(l3, finish_counts[0])

    def process_done_writes(self, l3, n_write: int):
        for _ in range(n_write):
            h2s_write: H2SWrite = l3.h2s_write_q_done.get()
            l3.complete_h2s_write()
            if h2s_write.node is not None:
                h2s_write.node.release_host()
            if self.enable_storage_metrics:
                self.storage_metrics_collector.log_backuped_tokens(h2s_write.completed_tokens)
