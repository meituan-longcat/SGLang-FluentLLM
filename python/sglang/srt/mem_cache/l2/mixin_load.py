from __future__ import annotations

from os import environ
from typing import TYPE_CHECKING

import torch

from sglang.srt.disaggregation.kv_events import StorageMedium
from sglang.srt.utils import get_colorful_logger, get_rank_page_nums

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import TreeNode

logger = get_colorful_logger(__name__)

test_load_failed = int(environ.get("TEST_L2_READ_FAILED", "0"))


class L2MixinLoad:
    def get_nodes_to_load(self, node: TreeNode):
        nodes = []
        while node.evicted:
            assert node.backuped, "No backup available on evicted nodes, should not happen"
            nodes.insert(0, node)
            node = node.parent
        return nodes, node

    def alloc_device_pages(self, page2rank: list[int]) -> torch.Tensor | None:
        device_kv_allocator = self.l2_ctrl.device_kv_allocator
        device_pages = device_kv_allocator.rank_alloc_pages(page2rank)
        if device_pages is None:
            self.evict_device_pages(page2rank)
            device_pages = device_kv_allocator.rank_alloc_pages(page2rank)
        if device_pages is None:
            return None
        return device_pages

    def load_back(self, node: TreeNode, mem_quota: int | None = None) -> tuple[torch.Tensor | None, bool]:
        """Load an evicted prefix and report a retryable allocation failure."""
        last_hit_node = node
        nodes_to_load, ancestor_node = self.get_nodes_to_load(node)

        delta = self.inc_lock_ref(ancestor_node)

        alloc_num_tokens = sum(len(n.key) for n in nodes_to_load) * self.page_size
        if alloc_num_tokens < self.load_back_threshold or (mem_quota is not None and alloc_num_tokens > mem_quota + delta):
            self.dec_lock_ref(ancestor_node)
            return None, False

        host_pages = torch.cat([n.host_value for n in nodes_to_load])
        page2rank = self.l2_ctrl.host_kv_pool.get_ranks_from_global_pages(host_pages)
        device_kv_allocator = self.l2_ctrl.device_kv_allocator
        rank_need_page_nums = get_rank_page_nums(page2rank)
        rank_free_before = device_kv_allocator.get_rank_free_page_nums()
        global test_load_failed
        if test_load_failed > 0:
            device_pages = None
        else:
            device_pages = self.alloc_device_pages(page2rank)

        if device_pages is None:
            rank_free_after = device_kv_allocator.get_rank_free_page_nums()
            rank_deficit = {rank: need - rank_free_after[rank] for rank, need in rank_need_page_nums.items() if need > rank_free_after[rank]}
            evictable_pages_during_alloc = self.evictable_size_
            protected_pages_during_alloc = self.protected_size_

        self.dec_lock_ref(ancestor_node)
        if device_pages is None:
            logger.info(
                f"Load back failed after eviction: pp_rank={self.pp_rank} "
                f"tp_rank={self.tp_rank} node={node.id} "
                f"pages={len(host_pages)} rank_need={rank_need_page_nums} "
                f"rank_free_before={rank_free_before} "
                f"rank_free_after={rank_free_after} "
                f"rank_deficit={rank_deficit} "
                f"evictable_pages_during_alloc={evictable_pages_during_alloc} "
                f"protected_pages_during_alloc={protected_pages_during_alloc}"
            )
            return None, True

        self.l2_ctrl.load(
            host_page_indices=host_pages,
            device_page_indices=device_pages,
            node=last_hit_node,
        )

        device_pages = device_pages.to(torch.int32).to(self.device)
        position = 0
        for node_to_load in nodes_to_load:
            node_to_load.value = device_pages[position : position + len(node_to_load.host_value)].clone()
            position += len(node_to_load.host_value)
            if self.enable_kv_cache_events:
                self._record_store_event(node_to_load, StorageMedium.GPU)

        self.evictable_size_ += len(device_pages)
        self.inc_lock_ref(last_hit_node)

        return device_pages, False

    def init_load_back(
        self,
        last_node: TreeNode,
        host_hit_length: int,
        mem_quota: int | None = None,
    ) -> tuple[torch.Tensor, TreeNode] | None:
        """Restore an L2 prefix, or return None when capacity requires a retry.

        A capacity failure leaves the requested node evicted and starts no H2D
        work, so the scheduler can retry this request after pages are released.
        An empty tensor in a successful result still means that loading was
        intentionally skipped, for example because the hit is below threshold.
        """
        _ = host_hit_length
        if last_node.evicted:
            loading_values, retryable_failure = self.load_back(last_node, mem_quota)
            if retryable_failure:
                return None
            if loading_values is not None:
                return loading_values, last_node
            # Loading was intentionally skipped; use the older device node.
            while last_node.evicted:
                last_node = last_node.parent

        return (
            torch.empty((0,), dtype=torch.int64, device=self.device),
            last_node,
        )

    def ready_to_load_host_cache(self, old_consumer_index=-1) -> int:
        return self.l2_ctrl.launch_h2d(old_consumer_index)

    def check_l2_loading_progress(self):
        finish_count = 0
        for load in self.l2_ctrl.h2d_load_q_ongoing:
            if not load.finish_event.query():
                break
            finish_count += 1
        return finish_count

    def process_l2_loads(self, finish_count):
        pf_metrics = self.scheduler_latency_metrics
        for load in self.l2_ctrl.h2d_load_q_ongoing[:finish_count]:
            load.finish_event.synchronize()
            if pf_metrics is not None and load.timing_start_event is not None and load.timing_finish_event is not None:
                try:
                    elapsed_ms = float(load.timing_start_event.elapsed_time(load.timing_finish_event))
                except Exception as e:
                    logger.warning(f"l2_load elapsed_time failed: {e}")
                    elapsed_ms = 0.0
                if elapsed_ms > 0.0:
                    pf_metrics.add_l2_load(elapsed_ms, load.kv_tokens)
            for node in load.nodes:
                self.dec_lock_ref(node)

        del self.l2_ctrl.h2d_load_q_ongoing[:finish_count]
