from __future__ import annotations

import dataclasses
import heapq
import logging
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, NamedTuple

import torch

from sglang.srt.mem_cache.evict_policy import (
    EvictionStrategy,
    FIFOStrategy,
    FILOStrategy,
    LFUStrategy,
    LRUStrategy,
    MRUStrategy,
    PriorityStrategy,
)
from sglang.srt.mem_cache.kv_event_mixin import KVCacheEventMixin
from sglang.srt.mem_cache.memory_pool import BaseTokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.memory_pool.kv_allocator import KVAllocator
from sglang.srt.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.req import Req


class MatchResult(NamedTuple):
    device_indices: torch.Tensor = None
    last_device_node: Any = None
    device_prefix_length: int = 0
    last_host_node: Any = None
    host_hit_length: int = 0
    mamba_branching_seqlen: int | None = None


@dataclasses.dataclass
class HiCacheInitParams:
    disable: bool
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool_allocator: KVAllocator
    page_size: int
    pp_rank: int = None
    dp_rank: int = None
    attn_tp_rank: int = None

    token_to_kv_pool: BaseTokenToKVPool = None
    tp_pp_group: torch.distributed.ProcessGroup | None = None
    eviction_policy: str = "lru"
    disable_finished_insert: bool = False

    enable_metrics: bool = False
    enable_kv_cache_events: bool = False
    draft_token_to_kv_pools: list[BaseTokenToKVPool] | None = None


class TreeNode:
    counter = 0

    def __init__(self, id: int | None = None, priority: int = 0):
        self.children: dict[tuple[int], TreeNode] = defaultdict(TreeNode)
        self.parent = None
        self.key: list[tuple[int]] = None
        self.value: torch.Tensor | None = None  # int32 device
        self.lock_ref = 0
        self.last_access_time = time.monotonic()
        self.creation_time = time.monotonic()

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

        self.hit_count = 0
        self.host_ref_counter = 0
        self.host_value: torch.Tensor | None = None  # int64 host
        self.hash_value: list[str] | None = None
        self.priority = priority

    def print(self, indent: int = 0):
        prefix = "  " * indent
        # Node basic info
        key_len = len(self.key) if self.key else 0
        value_len = len(self.value) if self.value is not None else 0
        value_device = self.value.device if self.value is not None else None
        host_value_len = len(self.host_value) if self.host_value is not None else 0
        host_value_device = self.host_value.device if self.host_value is not None else None
        status = "not in device" if self.evicted else "in device"
        backup_status = "in host" if self.backuped else "not in host"

        message = (
            f"{prefix}[Node id={self.id}] "
            f"key_len={key_len}, value_len={value_len} {value_device}, "
            f"host_value_len={host_value_len} {host_value_device}, "
            f"lock_ref={self.lock_ref}, host_ref={self.host_ref_counter}, "
            f"hit_count={self.hit_count}, "
            f"status={status}, {backup_status}, "
            f"children={len(self.children)},"
            f"last_access_time={self.last_access_time}\n"
        )
        logger.info(message)

        for child in self.children.values():
            child.print(indent + 1)

    def protect_host(self):
        self.host_ref_counter += 1

    def release_host(self):
        if self.host_ref_counter > 0:
            self.host_ref_counter -= 1
        else:
            raise RuntimeError("Host reference counter is already zero.")

    def get_last_hash_value(self) -> str | None:
        if self.hash_value is None or len(self.hash_value) == 0:
            return None
        return self.hash_value[-1]

    def get_prefix_hash_values(self, node: TreeNode) -> list[str]:
        if node is None or node.hash_value is None:
            return []

        return node.get_prefix_hash_values(node.parent) + node.hash_value

    @property
    def evicted(self):
        return self.value is None

    @property
    def backuped(self):
        return self.host_value is not None

    def __lt__(self, other: TreeNode):
        return self.last_access_time < other.last_access_time


def _key_match(key0: list, key1: list):
    i = 0
    for k0, k1 in zip(key0, key1):
        if k0 != k1:
            break
        i += 1
    return i


class RadixCache(KVCacheEventMixin):
    def __init__(self, params: HiCacheInitParams):
        self.disable = params.disable
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.enable_kv_cache_events = params.enable_kv_cache_events
        self.page_size = 1 if params.token_to_kv_pool_allocator is None else params.token_to_kv_pool_allocator.page_size
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
            raise ValueError(f"Unknown eviction policy: {self.eviction_policy}. Supported policies: 'lru', 'lfu', 'fifo', 'mru', 'filo', 'priority'.")
        self.kv_event_queue = []

        self.reset()

    @classmethod
    def create_simulated(
        cls,
        disable: bool = False,
        mock_allocator: Any | None = None,
        page_size: int = 1,
        enable_kv_cache_events: bool = False,
    ) -> RadixCache:
        params = HiCacheInitParams(
            disable=disable,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator,
            page_size=page_size,
            enable_kv_cache_events=enable_kv_cache_events,
        )
        return RadixCache(params)

    def reset(self):
        self.root_node = TreeNode()
        self.root_node.key = []
        self.root_node.value = torch.empty(
            0, dtype=torch.int32, device=self.req_to_token_pool.device if self.req_to_token_pool is not None else "cpu"
        )
        self.root_node.host_value = torch.empty(0, dtype=torch.int32, device="cpu")
        self.root_node.lock_ref = 1
        self.root_node.hash_value = []
        self.evictable_size_ = 0
        self.protected_size_ = 0
        self._record_all_cleared_event()

    def match_prefix(self, key: list, **kwargs) -> MatchResult:
        if self.disable or len(key) == 0:
            return self._empty_match_result()

        page_size = self.token_to_kv_pool_allocator.page_size
        if not isinstance(key[0], tuple):
            full_page_num = len(key) // page_size
            paged_token_ids = [tuple(key[i * page_size : (i + 1) * page_size]) for i in range(0, full_page_num)]
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

    def insert(self, key: list, value=None):
        if self.disable:
            return 0

        if value is None:
            value = [x for x in key]
        return self._insert_helper(self.root_node, key, value)

    def cache_finished_req(self, req: Req, delay_req_pool_release=False):
        if self.disable:
            alloced_len = self.req_to_token_pool.alloced_lens[req.req_pool_idx].item()
            self.token_to_kv_pool_allocator.free_req_cache(req.req_pool_idx, alloced_len)
            if not delay_req_pool_release:
                self.req_to_token_pool.free(req.req_pool_idx)
            return

        # A failed PD transfer may retain only its already valid, locked prefix.
        # Never publish allocated suffix pages whose KV was not written.
        token_ids = (req.origin_input_ids + req.output_ids)[:-1] if req.pd_kv_ready else req.origin_input_ids[: req.prefix_len]

        req_pool_idx = req.req_pool_idx
        page_size = self.token_to_kv_pool_allocator.page_size
        seq_len = len(token_ids)
        full_page_num = seq_len // page_size
        paged_token_ids = [tuple(token_ids[i * page_size : (i + 1) * page_size]) for i in range(0, full_page_num)]
        page_ids = self.token_to_kv_pool_allocator.req_to_page[req_pool_idx, :full_page_num].clone()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"insert_page_ids={page_ids}")
        _ = self.insert(paged_token_ids, page_ids)

        new_prefix_page_ids = self.match_prefix(paged_token_ids).device_indices

        if new_prefix_page_ids.numel() > 0:
            self.token_to_kv_pool_allocator.free_with_diff(new_prefix_page_ids, page_ids)
        self.token_to_kv_pool_allocator.free_extra_pages_not_cached(
            req_pool_idx,
            seq_len,
            self.req_to_token_pool.alloced_lens[req_pool_idx].item(),
        )
        if not delay_req_pool_release:
            self.req_to_token_pool.free(req.req_pool_idx)
            req.req_pool_idx = None
        if req.last_node is not None:
            self.dec_lock_ref(req.last_node)
            req.last_node = None
        if logger.isEnabledFor(logging.DEBUG):
            old_prefix_page_ids = self.token_to_kv_pool_allocator.req_to_page[req_pool_idx, : (seq_len + page_size - 1) // page_size].tolist()
            logger.debug(
                f"[cache_finished_req]\nold_prefix_page_ids={old_prefix_page_ids}\n"
                f"new_prefix_page_ids={new_prefix_page_ids.tolist()}\n"
                f"len(token_ids)={len(token_ids)} \n"
                f"alloced_lens={self.req_to_token_pool.alloced_lens[req_pool_idx]}, "
                f"{self.evictable_size_=}, {self.protected_size_=}, {req.last_node=}"
            )

    def cache_unfinished_req(self, req: Req, num_token_id=None):

        if self.disable:
            return

        token_ids = req.fill_ids[:num_token_id] if num_token_id is not None else req.fill_ids

        req_pool_idx = req.req_pool_idx
        page_size = self.token_to_kv_pool_allocator.page_size
        seq_len = len(token_ids)
        full_page_num = seq_len // page_size
        paged_token_ids = [tuple(token_ids[i * page_size : (i + 1) * page_size]) for i in range(0, full_page_num)]
        page_ids = self.token_to_kv_pool_allocator.req_to_page[req_pool_idx, :full_page_num].clone()
        _ = self.insert(paged_token_ids, page_ids)

        match_result = self.match_prefix(paged_token_ids)
        new_last_node = match_result.last_device_node

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)
        req.last_node = new_last_node

    def pretty_print(self, start_str: str = ""):
        logger.debug(self._print_helper(self.root_node, 0, start_str + f" #tokens: {self.total_size()} "))

    def total_size(self):
        return self._total_size_helper(self.root_node)

    def evict(self, num_pages: int):
        if self.disable:
            return

        leaves = self._collect_leaves()
        heapq.heapify(leaves)

        num_evicted = 0
        while num_evicted < num_pages and len(leaves):
            x = heapq.heappop(leaves)

            if x == self.root_node:
                break
            if x.lock_ref > 0:
                continue

            self.token_to_kv_pool_allocator.append_to_later_free(x.value)
            num_evicted += len(x.value)
            self._delete_leaf(x)

            if len(x.parent.children) == 0:
                heapq.heappush(leaves, x.parent)

        self.token_to_kv_pool_allocator.free_group_flush()
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
        return self.protected_size_

    def _empty_match_result(self):
        return MatchResult(
            device_indices=torch.tensor([], dtype=torch.int32),
            device_prefix_length=0,
            host_hit_length=0,
            last_device_node=self.root_node,
            last_host_node=self.root_node,
        )

    def _match_prefix_helper(self, node: TreeNode, key: list, value, last_node: TreeNode):
        node.last_access_time = time.monotonic()
        if len(key) == 0:
            return

        child_key = key[0]
        while len(key) > 0 and child_key in node.children:
            child = node.children[child_key]
            child.last_access_time = time.monotonic()
            prefix_len = _key_match(child.key, key)

            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                last_node[0] = new_node
                return

            value.append(child.value)
            last_node[0] = child
            node = child
            key = key[prefix_len:]

            if len(key) > 0:
                child_key = key[0]

    def _split_node(self, key, child: TreeNode, split_len: int):
        new_node = TreeNode()
        new_node.children = {key[split_len]: child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len]
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]

        if child.backuped:
            new_node.host_value = child.host_value[:split_len]
            child.host_value = child.host_value[split_len:]
        if child.hash_value:
            new_node.hash_value = child.hash_value[:split_len]
            child.hash_value = child.hash_value[split_len:]

        new_node.parent.children[key[0]] = new_node
        return new_node

    def _insert_helper(self, node: TreeNode, key: list, value):
        matched_len = 0

        while True:
            node.last_access_time = time.monotonic()
            if len(key) == 0:
                return matched_len

            child = node.children.get(key[0])
            if child is None:
                new_node = TreeNode()
                new_node.parent = node
                new_node.key = key
                new_node.value = value
                node.children[key[0]] = new_node
                self.evictable_size_ += len(value)

                # Emit BlockStored event when new KV blocks are inserted
                if self.enable_kv_cache_events:
                    self._record_store_event(new_node)
                return matched_len

            prefix_len = _key_match(child.key, key)
            matched_len += prefix_len

            if prefix_len == len(child.key):
                if prefix_len == len(key):
                    return matched_len

                key = key[prefix_len:]
                value = value[prefix_len:]
                node = child
                continue

            node = self._split_node(child.key, child, prefix_len)
            key = key[prefix_len:]
            value = value[prefix_len:]

    def _print_helper(self, node: TreeNode, indent: int, print_str: str) -> str:
        for _, child in node.children.items():
            print_str += "\n"
            print_str += " " * indent
            print_str += f"{child=} key_len={len(child.key)} "
            print_str += f"value={child.value} {child.lock_ref=}"
            print_str = self._print_helper(child, indent=indent + 2, print_str=print_str)
        return print_str

    def _delete_leaf(self, node):
        for k, v in node.parent.children.items():  # noqa: B007
            if v == node:
                break
        del node.parent.children[k]
        self.evictable_size_ -= len(node.key)

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

    def _get_all_node_host_value(self):
        ret_list = []
        stack = list(self.root_node.children.values())
        while stack:
            cur_node = stack.pop()
            if cur_node.host_value is not None and len(cur_node.host_value) > 0:
                ret_list.append(cur_node.host_value)
            stack.extend(cur_node.children.values())
        if len(ret_list) == 0:
            return None
        return torch.concat(ret_list)


if __name__ == "__main__":
    params = HiCacheInitParams(req_to_token_pool=None, token_to_kv_pool_allocator=None, disable=False)
    tree = RadixCache(params=params)
    tree.insert([(1, 2), (3, 4)], [torch.tensor([1, 2]), torch.tensor([3, 4])])
    tree.insert(
        [(1, 2), (3, 4), (5, 6)],
        [torch.tensor([1, 2]), torch.tensor([6, 5]), torch.tensor([7, 8])],
    )
    tree.pretty_print()
    print(tree.match_prefix([(1, 2), (3, 4)]))
