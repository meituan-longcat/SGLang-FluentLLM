from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.srt.disaggregation.kv_events import AllBlocksCleared, BlockRemoved, BlockStored, StorageMedium
from sglang.srt.mem_cache.utils import compute_node_hash_values, hash_str_to_int64
from sglang.srt.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import TreeNode


class KVCacheEventMixin:
    def _record_store_event(self, node: TreeNode, storage_medium: StorageMedium = StorageMedium.GPU):
        if not self.enable_kv_cache_events or node.key is None:
            return

        if node.hash_value is None:
            node.hash_value = compute_node_hash_values(node, self.page_size)

        parent_block_hash = None
        if node.parent is not None and node.parent != self.root_node:
            if node.parent.hash_value is not None and len(node.parent.hash_value) > 0:
                parent_block_hash = hash_str_to_int64(node.parent.hash_value[-1])

        for i, page_tuple in enumerate(node.key):
            paged_token_ids = list(page_tuple)
            assert len(paged_token_ids) == self.page_size
            block_hash = hash_str_to_int64(node.hash_value[i])

            self.kv_event_queue.append(
                BlockStored(
                    block_hashes=[block_hash],
                    parent_block_hash=parent_block_hash,
                    token_ids=paged_token_ids,
                    block_size=len(paged_token_ids),
                    medium=storage_medium,
                )
            )
            parent_block_hash = block_hash

    def _record_remove_event(self, node: TreeNode, storage_medium: StorageMedium = StorageMedium.GPU):
        if not self.enable_kv_cache_events or node.key is None:
            return

        if node.hash_value is None:
            node.hash_value = compute_node_hash_values(node, self.page_size)

        for i in range(len(node.key)):
            assert len(node.key[i]) == self.page_size
            block_hash = hash_str_to_int64(node.hash_value[i])
            self.kv_event_queue.append(BlockRemoved(block_hashes=[block_hash], medium=storage_medium))

    def _record_all_cleared_event(self):
        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

    def take_events(self):
        if not self.enable_kv_cache_events:
            return []

        events = self.kv_event_queue
        self.kv_event_queue = []
        return events
