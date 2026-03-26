from __future__ import annotations

"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
The radix tree data structure for managing the KV cache.
"""

import heapq
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Callable, List, Optional, Any

from functools import lru_cache
import torch

import dataclasses
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, MatchResult
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool, BaseTokenToKVPool
from sglang.srt.mem_cache.allocator import KVAllocator
from sglang.srt.mem_cache.hicache_storage import get_hash_str
from sglang.srt.utils import get_colorful_logger
from sglang.srt.mem_cache.evict_policy import (
    EvictionStrategy,
    FIFOStrategy,
    FILOStrategy,
    LFUStrategy,
    LRUStrategy,
    MRUStrategy,
    PriorityStrategy,
)
logger = get_colorful_logger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.req import Req

# Import KV events classes if needed
try:
    from sglang.srt.disaggregation.kv_events import BlockStored, BlockRemoved, AllBlocksCleared
    KV_EVENTS_AVAILABLE = True
except ImportError:
    BlockStored = None
    BlockRemoved = None
    AllBlocksCleared = None
    KV_EVENTS_AVAILABLE = False
    logger.warning(
        "KV Events module not available. "
        "KV cache events will not be emitted even if enable_kv_cache_events=True."
    )

@dataclasses.dataclass
class CacheInitParams:
    disable: bool
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool_allocator: KVAllocator
    page_size: int

    token_to_kv_pool: BaseTokenToKVPool = None
    tp_cache_group: Optional[torch.distributed.ProcessGroup] = None
    eviction_policy: str = "lru"
    disable_finished_insert: bool = False

    enable_metrics: bool = False
    enable_kv_cache_events: bool = False

class TreeNode:
    counter = 0

    def __init__(self, id: Optional[int] = None, priority: int = 0):
        self.children = defaultdict(TreeNode)
        self.parent = None
        self.key = None
        self.value = None
        self.lock_ref = 0
        self.last_access_time = time.monotonic()
        self.creation_time = time.monotonic()

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

        
        ##### Hicache #####
        # host_value is token level, others are page level
        self.hit_count = 0
        # indicating the node is locked to protect from eviction
        # incremented when the node is referenced by a storage operation
        self.host_ref_counter = 0
        # store the host indices of KV cache
        self.host_value: Optional[torch.Tensor] = None
        # store hash values of each pages
        self.hash_value: Optional[List[str]] = None
        # priority for priority-aware eviction
        self.priority = priority


    def protect_host(self):
        """Protect the host value from eviction."""
        self.host_ref_counter += 1

    def release_host(self):
        """Release the host value, allowing it to be evicted."""
        if self.host_ref_counter > 0:
            self.host_ref_counter -= 1
        else:
            raise RuntimeError("Host reference counter is already zero.")

    def get_last_hash_value(self) -> Optional[str]:
        """Returns the hash value of the last page in this node."""
        if self.hash_value is None or len(self.hash_value) == 0:
            return None
        return self.hash_value[-1]

    @lru_cache(maxsize=1)
    def get_prefix_hash_values(self, node: TreeNode) -> List[str]:
        if node is None or node.hash_value is None:
            return []

        return node.get_prefix_hash_values(node.parent) + node.hash_value
    @property
    def evicted(self):
        return self.value is None

    @property
    def backuped(self):
        return self.host_value is not None

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time


def _key_match(key0: List, key1: List):
    i = 0
    for k0, k1 in zip(key0, key1):
        if k0 != k1:
            break
        i += 1
    return i


def compute_node_hash_values(node: TreeNode, page_size: int) -> List[str]:
    """Compute SHA256-based hash values for position-aware identification.

    Args:
        node: The TreeNode to compute hash values for
        page_size: The page size for chunking tokens

    Returns:
        List of SHA256 hex strings, one per page
    """
    hash_values = []

    # Get parent's last hash value if parent exists
    parent_hash = None
    if node.parent is not None and node.parent.hash_value is not None:
        # Check if parent is root by checking if it has empty key
        if len(node.parent.key) > 0 and len(node.parent.hash_value) > 0:
            parent_hash = node.parent.hash_value[-1]

    # Iterate through node's pages
    for start in range(0, len(node.key)):
        page_tokens = node.key[start]
        if not page_tokens:
            continue

        # Use SHA256-based chaining via get_hash_str
        hash_val = get_hash_str(page_tokens, prior_hash=parent_hash)
        hash_values.append(hash_val)
        parent_hash = hash_val

    return hash_values

class RadixCache(BasePrefixCache):
    def __init__(self,params: CacheInitParams):
        self.disable = params.disable
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.enable_kv_cache_events = params.enable_kv_cache_events
        self.page_size = (
            1
            if params.token_to_kv_pool_allocator is None
            else params.token_to_kv_pool_allocator.page_size
        )
        self.eviction_policy = params.eviction_policy.lower()
        self.key_match_fn = _key_match

        if self.eviction_policy == "lru":
            self.eviction_strategy: EvictionStrategy = LRUStrategy()
        elif self.eviction_policy == "lfu":
            self.eviction_strategy: EvictionStrategy = LFUStrategy()
        elif self.eviction_policy == "fifo":
            self.eviction_strategy: EvictionStrategy = FIFOStrategy()
        elif self.eviction_policy == "mru":
            self.eviction_strategy: EvictionStrategy = MRUStrategy()
        elif self.eviction_policy == "filo":
            self.eviction_strategy: EvictionStrategy = FILOStrategy()
        elif self.eviction_policy == "priority":
            self.eviction_strategy: EvictionStrategy = PriorityStrategy()
        else:
            raise ValueError(
                f"Unknown eviction policy: {self.eviction_policy}. Supported policies: 'lru', 'lfu', 'fifo', 'mru', 'filo', 'priority'."
            )
        # Initialize KV event queue
        self.kv_event_queue = []
        
        # Warn if KV events are enabled but module is not available
        if self.enable_kv_cache_events and not KV_EVENTS_AVAILABLE:
            logger.warning(
                "KV cache events are enabled (enable_kv_cache_events=True) "
                "but KV Events module is not available. Events will not be emitted."
            )
        
        self.reset()

    @classmethod
    def create_simulated(
        self,
        disable: bool = False,
        mock_allocator: Optional[Any] = None,
        page_size: int = 1,
        enable_kv_cache_events: bool = False,
    ) -> RadixCache:
        """Init a radix cache without memory pools for simulation purpose."""
        params = CacheInitParams(
            disable=disable,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator,
            page_size=page_size,
            enable_kv_cache_events=enable_kv_cache_events,
        )
        return RadixCache(params)

    ##### Public API #####

    def reset(self):
        self.root_node = TreeNode()
        self.root_node.key = []
        self.root_node.value = []
        self.root_node.host_value = []
        self.root_node.lock_ref = 1
        self.root_node.hash_value = []
        self.evictable_size_ = 0
        self.protected_size_ = 0

    def gen_paged_token_ids(self, full_page_num, key, page_size, req: Req):
        if req.input_extra_infos is not None and \
            isinstance(req.input_extra_infos, list) and \
                len(req.input_extra_infos) > 0 and \
                "paged_hash_ids" in req.input_extra_infos[0]:
            paged_hash_ids = req.input_extra_infos[0]["paged_hash_ids"]
            paged_token_ids = [
                tuple([paged_hash_ids[i]])
                for i in range(0, full_page_num)
            ]
        else:
            paged_token_ids = [
                tuple(key[i * page_size : (i + 1) * page_size])
                for i in range(0, full_page_num)
            ]
        return paged_token_ids
    
    def match_prefix(self, key: List, req: Req, **kwargs) -> MatchResult:
        """Find the matching prefix from the radix tree.
        Args:
            key: A list of token IDs to find a matching prefix.
        Returns:
            A tuple of a tensor of matching prefix token IDs and
            the last node that contains the prefix values. Note that
            this API can modify the internal state of the Radix tree.
            The last node create a new child if the prefix is shorter
            than the last node's value.
        """

        # Emit AllBlocksCleared event when cache is reset
        if self.enable_kv_cache_events and KV_EVENTS_AVAILABLE:
            self.kv_event_queue.append(AllBlocksCleared())
        if self.disable or len(key) == 0:
            return self._empty_match_result()

        page_size = self.token_to_kv_pool_allocator.page_size
        # Compatible with whether the incoming key is paged
        if not isinstance(key[0], tuple):
            full_page_num = len(key) // page_size
            paged_token_ids = self.gen_paged_token_ids(full_page_num, key, page_size, req)
        else:
            paged_token_ids = key
        if len(paged_token_ids) == 0:
            return self._empty_match_result()
        value = []
        last_node = [self.root_node]
        self._match_prefix_helper(self.root_node, paged_token_ids, value, last_node)
        if value and isinstance(value[0], list):
            flat_value = [e for arr in value for e in arr]
            value = torch.concat(flat_value)
        elif value:
            value = torch.concat(value)
        else:
            value = torch.tensor([], dtype=torch.int32)
        return MatchResult(
            device_indices=value,
            last_device_node=last_node[0],
            last_host_node=last_node[0],
            device_prefix_length=len(value) * page_size,
        )
    
    def insert(self, key: List, value=None):
        if self.disable:
            return 0

        if value is None:
            value = [x for x in key]
        return self._insert_helper(self.root_node, key, value)

    def cache_finished_req(self, req: Req, token_ids: Optional[List[int]] = None, delay_req_pool_release = False):
        """Cache request when it finishes."""
        if self.disable:
            alloced_len = self.req_to_token_pool.alloced_lens[req.req_pool_idx].item()
            self.token_to_kv_pool_allocator.free_req_cache(req.req_pool_idx, alloced_len)
            if not delay_req_pool_release:
                self.req_to_token_pool.free(req.req_pool_idx)
            return

        if token_ids is None:
            token_ids = (req.origin_input_ids + req.output_ids)[:-1]
        # Radix Cache takes one ref in memory pool
        req_pool_idx = req.req_pool_idx
        page_size = self.token_to_kv_pool_allocator.page_size
        seq_len = len(token_ids)
        # without last not full page
        full_page_num = seq_len // page_size
        paged_token_ids = self.gen_paged_token_ids(full_page_num, token_ids, page_size, req)
        page_ids = self.token_to_kv_pool_allocator.req_to_page[
            req_pool_idx, :full_page_num
        ].clone()
        logger.debug(f"insert_page_ids={page_ids}")
        _ = self.insert(paged_token_ids, page_ids)

        # After insert, the tree changed so the match prefix ids changed
        new_prefix_page_ids = self.match_prefix(paged_token_ids, req).device_indices
        # new_prefix_page_ids is cached in tree, free the diff part in page_ids

        if new_prefix_page_ids.numel() > 0:
            self.token_to_kv_pool_allocator.free_with_diff(
                new_prefix_page_ids, page_ids
            )
        self.token_to_kv_pool_allocator.free_extra_pages_not_cached(
            req_pool_idx,
            seq_len,
            self.req_to_token_pool.alloced_lens[req_pool_idx].item(),
        )
        if not delay_req_pool_release:
            self.req_to_token_pool.free(req.req_pool_idx)
            req.req_pool_idx = None
        # Remove req slot release the cache lock
        if req.last_node is not None:
            self.dec_lock_ref(req.last_node)
            req.last_node = None
        logger.debug(
            f"[cache_finished_req]\nold_prefix_page_ids={self.token_to_kv_pool_allocator.req_to_page[req_pool_idx, : (seq_len + page_size - 1) // page_size].tolist()}\n"
            f"new_prefix_page_ids={new_prefix_page_ids.tolist()}\n"
            f"len(token_ids)={len(token_ids)} \n"
            f"alloced_lens={self.req_to_token_pool.alloced_lens[req_pool_idx]}, {self.evictable_size_=}, {self.protected_size_=}, {req.last_node=}"
        )

    def cache_unfinished_req(self, req: Req, token_ids: Optional[List[int]] = None):
        """Cache request when it is unfinished."""
        
        if self.disable:
            return

        if token_ids is None:
            token_ids = req.fill_ids

        page_size = self.token_to_kv_pool_allocator.page_size
        req_pool_idx = req.req_pool_idx
        seq_len = len(token_ids)
        # without last not full page
        full_page_num = seq_len // page_size
        paged_token_ids = self.gen_paged_token_ids(full_page_num, token_ids, page_size, req)
        
        page_ids = self.token_to_kv_pool_allocator.req_to_page[
            req_pool_idx, :full_page_num
        ].clone()
        _ = self.insert(paged_token_ids, page_ids)
        # After insert, perform matching, use page_id in radix tree to replace the allocated
        # page before insert, release diff part, and write to req_to_token_pool
        match_result = self.match_prefix(paged_token_ids, req)
        (new_prefix_page_ids, new_last_node) = (match_result.device_indices, match_result.last_device_node)
        if new_prefix_page_ids.numel() > 0:
            diff = self.token_to_kv_pool_allocator.free_with_diff(
                new_prefix_page_ids, page_ids
            )
            diff_idxs = torch.nonzero(diff)
            self.token_to_kv_pool_allocator.req_to_page[
                req.req_pool_idx, diff_idxs.squeeze()
            ] = new_prefix_page_ids[diff]
            token_level_offsets = torch.arange(
                self.page_size, device=self.req_to_token_pool.device
            )
            indices_start_locs = diff_idxs * self.page_size
            diff_slots_indices = (
                indices_start_locs[:, None] + token_level_offsets
            ).flatten()
            new_slots_start_locs = new_prefix_page_ids[diff] * self.page_size
            new_slots = (new_slots_start_locs[:, None] + token_level_offsets).flatten()
            self.req_to_token_pool.req_to_token[
                req.req_pool_idx, diff_slots_indices
            ] = new_slots.to(torch.int32)
        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)
        req.last_node = new_last_node

    def pretty_print(self, start_str: str=""):
        logger.debug(self._print_helper(self.root_node, 0,start_str + f" #tokens: {self.total_size()} "))

    def total_size(self):
        return self._total_size_helper(self.root_node)

    def evict(self, num_tokens: int, evict_callback: Callable = None):
        if self.disable:
            return

        leaves = self._collect_leaves()
        heapq.heapify(leaves)

        num_evicted = 0
        while num_evicted < num_tokens and len(leaves):
            x = heapq.heappop(leaves)

            if x == self.root_node:
                break
            if x.lock_ref > 0:
                continue

            # use new free function
            self.token_to_kv_pool_allocator.append_to_later_free(x.value)
            num_evicted += len(x.value)
            self._delete_leaf(x)

            if len(x.parent.children) == 0:
                heapq.heappush(leaves, x.parent)

        # trigger, append tensor to free_slots
        self.token_to_kv_pool_allocator.free_group_end()
        return num_evicted

    def inc_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.value)
                self.protected_size_ += len(node.value)
                delta -= len(node.value)
            node.lock_ref += 1
            node = node.parent
        return delta

    def dec_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0
        
        if node is None:
            return 0
        
        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.value)
                self.protected_size_ -= len(node.value)
                delta += len(node.value)
            node.lock_ref -= 1
            node = node.parent
        return delta

    def evictable_size(self):
        return self.evictable_size_

    def protected_size(self):
        # protected size refers to the size of the cache that is locked
        return self.protected_size_

    ##### Internal Helper Functions #####

    def _empty_match_result(self):
        return MatchResult(
            device_indices=torch.tensor([], dtype=torch.int32),
            device_prefix_length=0,
            host_hit_length=0,
            last_device_node=self.root_node,
            last_host_node=self.root_node,
        )
    def _match_prefix_helper(
        self, node: TreeNode, key: List, value, last_node: TreeNode
    ):
        node.last_access_time = time.monotonic()
        if len(key) == 0:
            return

        if key[0] in node.children.keys():
            child = node.children[key[0]]
            prefix_len = _key_match(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                last_node[0] = new_node
            else:
                value.append(child.value)
                last_node[0] = child
                self._match_prefix_helper(child, key[prefix_len:], value, last_node)

    def _split_node(self, key, child: TreeNode, split_len: int):
        # new_node -> child
        new_node = TreeNode()
        new_node.children = {key[split_len]: child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len]
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]
        new_node.parent.children[key[0]] = new_node
        return new_node

    def _insert_helper(self, node: TreeNode, key: List, value):
        node.last_access_time = time.monotonic()
        if len(key) == 0:
            return 0

        if key[0] in node.children.keys():
            child = node.children[key[0]]
            prefix_len = _key_match(child.key, key)

            if prefix_len == len(child.key):
                if prefix_len == len(key):
                    return prefix_len
                else:
                    key = key[prefix_len:]
                    value = value[prefix_len:]
                    return prefix_len + self._insert_helper(child, key, value)

            new_node = self._split_node(child.key, child, prefix_len)
            return prefix_len + self._insert_helper(
                new_node, key[prefix_len:], value[prefix_len:]
            )

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            node.children[key[0]] = new_node
            self.evictable_size_ += len(value)
            
            # Emit BlockStored event when new KV blocks are inserted
            if self.enable_kv_cache_events:
                self._record_store_event(new_node)
        return 0

    def _print_helper(self, node: TreeNode, indent: int, print_str: str) -> str:
        for _, child in node.children.items():
            print_str += '\n'
            print_str += " " * indent
            print_str += f"{child=} key_len={len(child.key)} "
            print_str += f"value={child.value} {child.lock_ref=}"
            print_str = self._print_helper(child, indent=indent + 2, print_str=print_str)
        return print_str

    def _delete_leaf(self, node):
        for k, v in node.parent.children.items():
            if v == node:
                break
        del node.parent.children[k]
        self.evictable_size_ -= len(node.key)
        
        # Emit BlockRemoved event when KV blocks are removed
        if self.enable_kv_cache_events:
            self._record_remove_event(node)

    def _total_size_helper(self, node: TreeNode):
        if node.evicted:
            return 0
        x = len(node.value)
        for child in node.children.values():
            x += self._total_size_helper(child)
        return x

    def _collect_leaves(self):
        ret_list = []
        stack = [self.root_node]

        while stack:
            cur_node = stack.pop()
            if len(cur_node.children) == 0:
                ret_list.append(cur_node)
            else:
                stack.extend(cur_node.children.values())

        return ret_list

    def _get_all_node_value(self):
        ret_list = []
        stack = list(self.root_node.children.values())
        while stack:
            cur_node = stack.pop()
            if cur_node.value is not None:
                ret_list.append(cur_node.value)
            stack.extend(cur_node.children.values())
        if len(ret_list) == 0:
            return None
        return torch.concat(ret_list)

    def _record_store_event(self, node: TreeNode):
        """Record BlockStored event for a node."""
        if not self.enable_kv_cache_events or not KV_EVENTS_AVAILABLE or node.key is None:
            return
        
        # FluentLLM uses tuples as keys, where each tuple represents a page
        # node.key is a list of tuples: [(token1, token2, ...), (token3, token4, ...), ...]
        # Each tuple is already a page, so we iterate over them directly
        
        parent_block_hash = None
        if node.parent and node.parent != self.root_node and node.parent.key:
            # Use the last page (tuple) of the parent
            parent_block_hash = hash(node.parent.key[-1])
        
        # Each element in node.key is already a page (tuple of tokens)
        for page_tuple in node.key:
            if not page_tuple:
                continue
            
            # Convert tuple to list for token_ids
            token_ids = list(page_tuple)
            block_hash = hash(page_tuple)
            
            self.kv_event_queue.append(
                BlockStored(
                    block_hashes=[block_hash],
                    parent_block_hash=parent_block_hash,
                    token_ids=token_ids,
                    block_size=len(token_ids),
                    lora_id=None,
                )
            )
            # Chain next chunk to this one
            parent_block_hash = block_hash

    def _record_remove_event(self, node: TreeNode):
        """Record BlockRemoved event for a node."""
        if not self.enable_kv_cache_events or not KV_EVENTS_AVAILABLE or node.key is None:
            return
        
        # Create BlockRemoved event for each page
        for start in range(0, len(node.key), self.page_size):
            page_tokens = node.key[start : start + self.page_size]
            if not page_tokens:
                continue
            block_hash = hash(tuple(page_tokens))
            self.kv_event_queue.append(BlockRemoved(block_hashes=[block_hash]))

    def take_events(self):
        """Atomically takes all events and clears the queue.
        
        Returns:
            A list of KV cache events.
        """
        if not self.enable_kv_cache_events:
            return []
        
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events


if __name__ == "__main__":

    params = CacheInitParams(
        req_to_token_pool=None, token_to_kv_pool_allocator=None, disable=False
    )
    tree = RadixCache(params=params)
    tree.insert([(1, 2), (3, 4)], [torch.tensor([1, 2]), torch.tensor([3, 4])])
    tree.insert(
        [(1, 2), (3, 4), (5, 6)],
        [torch.tensor([1, 2]), torch.tensor([6, 5]), torch.tensor([7, 8])],
    )
    tree.pretty_print()
    print(tree.match_prefix([(1, 2), (3, 4)]))