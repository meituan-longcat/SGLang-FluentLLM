from __future__ import annotations

import heapq
from typing import TYPE_CHECKING

from sglang.srt.disaggregation.kv_events import StorageMedium
from sglang.srt.utils import get_colorful_logger, get_rank_page_nums

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import TreeNode

logger = get_colorful_logger(__name__)


class L2MixinEvict:
    def collect_leaves_device(self):
        def _is_leaf(node: TreeNode) -> bool:
            if node.evicted:
                return False
            if node is self.root_node:
                return False
            if len(node.children) == 0:
                return True
            return all(child.evicted for child in node.children.values())

        ret_list = []
        stack = [self.root_node]
        while stack:
            cur_node = stack.pop()
            if _is_leaf(cur_node):
                ret_list.append(cur_node)
            else:
                for cur_child in cur_node.children.values():
                    if not cur_child.evicted:
                        stack.append(cur_child)
        return ret_list

    def evict_device_pages(self, page2rank: list | None, page_num: int = -1) -> int:
        leaves = self.collect_leaves_device()
        eviction_heap = [(self.eviction_strategy.get_priority(node), node) for node in leaves]
        heapq.heapify(eviction_heap)

        kv_allocator = self.l2_ctrl.device_kv_allocator
        if page2rank is not None:
            # for page rank's need
            rank_need_page_nums = get_rank_page_nums(page2rank)
            rank_free_page_nums = kv_allocator.get_rank_free_page_nums()
            rank_need_page_nums = {rank: max(0, need - rank_free_page_nums[rank]) for rank, need in rank_need_page_nums.items()}
            # group rank
            rank_evicted_page_nums = {rank: 0 for rank in range(kv_allocator.group_rank)}

            def is_enough(need_page_nums, evicted_page_nums):
                return all(evicted_page_nums[rank] >= needed for rank, needed in need_page_nums.items())

            def update_evicted_page_nums(evicted_page_nums, page_ids):
                # todo: maybe we need page2rank in tree node, page_ids in device
                ranks = kv_allocator.get_rank_from_global_page_id(page_ids).tolist()
                for rank in ranks:
                    evicted_page_nums[rank] += 1
        else:
            rank_need_page_nums = {-1: max(0, page_num)}
            rank_evicted_page_nums = {-1: 0}

            def is_enough(need_page_nums, evicted_page_nums):
                return evicted_page_nums[-1] >= need_page_nums[-1]

            def update_evicted_page_nums(evicted_page_nums, page_ids):
                evicted_page_nums[-1] += len(page_ids)

        write_back_nodes = []
        while not is_enough(rank_need_page_nums, rank_evicted_page_nums) and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)

            if x.lock_ref > 0:
                continue

            if not x.backuped:
                if self.l2_ctrl.write_policy == "write_back":
                    write_page_num = self.write_backup(x, write_back=True)
                    if write_page_num > 0:
                        write_back_nodes.append(x)
                        evicted_pages = x.value
                    else:
                        logger.warning(f"write_back failed for node {x.id}, evicting directly.")
                        evicted_pages = self.evict_regular(x)
                else:
                    evicted_pages = self.evict_regular(x)
            else:
                evicted_pages = self.evict_backuped(x)
            update_evicted_page_nums(rank_evicted_page_nums, evicted_pages)

            for child in x.parent.children.values():
                if child in write_back_nodes:
                    continue
                if not child.evicted:
                    break
            else:
                # all child is evicted(no device page), add parent
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

        if self.l2_ctrl.write_policy == "write_back":
            finish_count = len(self.l2_ctrl.d2h_write_q_ongoing)
            self.process_l2_writes(finish_count)
            for node in write_back_nodes:
                assert node.backuped
                self.evict_backuped(node)

        self.l2_ctrl.device_kv_allocator.free_group_flush()
        return sum(rank_evicted_page_nums.values())

    def evict_backuped(self, node: TreeNode) -> list[int]:
        released_pages = node.value
        self.l2_ctrl.device_kv_allocator.append_to_later_free(released_pages)
        num_evicted = len(released_pages)
        logger.debug(f"[L2-Evict-Backuped] node={node.id} pages={num_evicted} has_host={node.host_value is not None}")
        self.evictable_size_ -= num_evicted
        if self.enable_kv_cache_events:
            self._record_remove_event(node, StorageMedium.GPU)
        node.value = None
        return released_pages

    def evict_regular(self, node: TreeNode) -> list[int]:
        released_pages = node.value
        logger.debug(f"[L2-Evict-Regular] node={node.id} pages={len(released_pages)} backuped={node.backuped}")
        self.l2_ctrl.device_kv_allocator.append_to_later_free(released_pages)
        self._delete_leaf(node)
        return released_pages

    def evict_host_pages(self, page2rank: list[int]):
        leaves = self._collect_leaves()
        eviction_heap = [(self.eviction_strategy.get_priority(node), node) for node in leaves]
        heapq.heapify(eviction_heap)

        host_allocator = self.l2_ctrl.host_kv_allocator
        rank_need_page_nums = get_rank_page_nums(page2rank)
        rank_free_page_nums = host_allocator.get_rank_free_page_nums()
        rank_need_page_nums = {rank: max(0, need - rank_free_page_nums[rank]) for rank, need in rank_need_page_nums.items()}
        rank_evicted_page_nums = {rank: 0 for rank in range(host_allocator.group_rank)}

        def is_enough(need_page_nums, evicted_page_nums):
            return all(evicted_page_nums[rank] >= needed for rank, needed in need_page_nums.items())

        def update_evicted_page_nums(evicted_page_nums, page_ids):
            ranks = host_allocator.get_rank_from_global_page_id(page_ids).tolist()
            for rank in ranks:
                evicted_page_nums[rank] += 1

        while not is_enough(rank_need_page_nums, rank_evicted_page_nums) and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)
            if x is self.root_node:
                break
            if not x.evicted:
                continue
            if x.host_ref_counter > 0:
                continue

            evicted_pages = x.host_value
            host_allocator.free_pages(x.host_value)
            self._record_remove_event(x, StorageMedium.CPU)
            update_evicted_page_nums(rank_evicted_page_nums, evicted_pages)

            for k, v in x.parent.children.items():  # noqa: B007
                if v is x:
                    break
            del x.parent.children[k]

            if len(x.parent.children) == 0 and x.parent.evicted:
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))
