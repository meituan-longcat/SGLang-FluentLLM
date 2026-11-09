from __future__ import annotations

from os import environ
from typing import TYPE_CHECKING

import torch

from sglang.srt.disaggregation.kv_events import StorageMedium
from sglang.srt.utils import get_colorful_logger

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import TreeNode

logger = get_colorful_logger(__name__)

test_write_failed = int(environ.get("TEST_L2_WRITE_FAILED", "0"))


class L2MixinWrite:
    def get_nodes_to_write(self, node: TreeNode):
        nodes = []
        while not node.backuped:
            nodes.insert(0, node)
            node = node.parent
        return nodes

    def alloc_host_pages(self, page2rank) -> torch.Tensor | None:
        host_kv_allocator = self.l2_ctrl.host_kv_allocator
        host_pages = host_kv_allocator.rank_alloc_pages(page2rank)
        if host_pages is None:
            self.evict_host_pages(page2rank)
            host_pages = host_kv_allocator.rank_alloc_pages(page2rank)
        return host_pages

    def write_backup(self, node: TreeNode, write_back: bool = False) -> int:
        if not write_back:
            # may write failed before, now write again
            nodes_to_write = self.get_nodes_to_write(node)
        else:
            nodes_to_write = [node]

        device_pages = torch.cat([n.value for n in nodes_to_write])
        total_pages = len(device_pages)
        logger.debug(
            f"[L2-Backup] mode={'write_back' if write_back else 'write_through'} "
            f"node={node.id} nodes_to_write={len(nodes_to_write)} pages={total_pages}"
        )

        page2rank = self.l2_ctrl.device_kv_pool.get_ranks_from_global_pages(device_pages)
        global test_write_failed
        if test_write_failed > 0:
            test_write_failed += 1
            if test_write_failed % 4 != 0:
                logger.info(f"Write to host failed after eviction {node.id}")
                return 0
        host_pages = self.alloc_host_pages(page2rank)
        if host_pages is None:
            logger.info(f"Write to host failed after eviction {node.id}")
            return 0

        position = 0
        for node_to_write in nodes_to_write:
            node_to_write.host_value = host_pages[position : position + len(node_to_write.value)]
            position += len(node_to_write.value)

        self.l2_ctrl.write(
            device_page_indices=device_pages,
            host_page_indices=host_pages,
            node=node,
        )

        if not write_back:
            self.inc_lock_ref(node)

        return len(device_pages)

    def touch_tree_node(self, node: TreeNode, chunked: bool = False):
        with torch.profiler.record_function("touch_tree_node"):
            # or chunked todo which better? "update in time" should be better
            if self.l2_ctrl.write_policy == "write_back":
                return
            node.hit_count += 1

            if getattr(self, "decode_offload_bypass_l2", False):
                return
            if not node.backuped and node.hit_count >= self.write_through_threshold:
                logger.debug(f"[Touch-Trigger] node_id={node.id} hit_count={node.hit_count} write_through_threshold={self.write_through_threshold}")
                self.write_backup(node)

    def check_l2_writing_progress(self):
        finish_count = 0
        for w in self.l2_ctrl.d2h_write_q_ongoing:
            if w.finish_event is not None and not w.finish_event.query():
                # device is not ready, just failed
                break
            # device is ready
            if w.finish_event is not None:
                # device ready
                w.finish_event.synchronize()
                w.finish_event = None
            finish_count = finish_count + 1
        return finish_count

    def process_l2_writes(self, finish_count):
        while finish_count > 0:
            w = self.l2_ctrl.d2h_write_q_ongoing.pop(0)
            for node in w.nodes:
                self.dec_lock_ref(node)
                self._record_store_event(node, StorageMedium.CPU)
                self.add_h2s_write(node)
            finish_count -= 1
