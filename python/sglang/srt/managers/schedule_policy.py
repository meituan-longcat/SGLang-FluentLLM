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
"""Request scheduler policy"""

import os
import random
from collections import defaultdict
from contextlib import contextmanager
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Union

import torch

from sglang.srt.managers.req import Req
from sglang.srt.managers.scheduling.schedule_batch import ScheduleBatch, PrefillReqInfo
from sglang.srt.mem_cache.memory_pool.kv_allocator import KVAllocator
from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode
from sglang.srt.observability.metrics.collector import CacheHitMetrics
from sglang.srt.utils import get_colorful_logger
logger = get_colorful_logger(__name__)
from sglang.srt.env import global_server_args_dict

# Clip the estimation of max_new_tokens for the request whose max_new_tokens is very large.
# This can prevent the server from being too conservative.
# Note that this only clips the estimation in the scheduler but does not change the stop
# condition. The request can still generate tokens until it hits the unclipped max_new_tokens.
CLIP_MAX_NEW_TOKENS_ESTIMATION = int(
    os.environ.get("SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION", "4096")
)

# Threshold for in-batch prefix cache.
# If a request has a matched prefix length (against existing cache) less than this value,
# the scheduler runs the in-batch prefix caching check for this request.
# If we set it to -1, it means we disable in-batch prefix caching.
IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD = int(
    os.environ.get("IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD", "32")
)

# Threshold for in-batch prefix cache.
# If a request has a matched prefix length (within the waiting queue) larger than this value,
# the scheduler deprioritizes this request
IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD = int(
    os.environ.get("IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD", "32")
)


class CacheAwarePolicy(Enum):
    """Scheduling policies that are aware of the tree cache."""

    LPM = "lpm"  # longest prefix match
    DFS_WEIGHT = "dfs-weight"  # depth-first search weighting


class CacheAgnosticPolicy(Enum):
    """Scheduling policies that are not aware of the tree cache."""

    FCFS = "fcfs"  # first come first serve
    LOF = "lof"  # longest output first
    RANDOM = "random"


class SchedulePolicy:
    Policy = Union[CacheAwarePolicy, CacheAgnosticPolicy]

    def __init__(self, policy: str, tree_cache: RadixCache, enable_hierarchical_cache: bool):
        self.policy = self._validate_and_adjust_policy(policy, tree_cache)
        self.tree_cache = tree_cache
        self.enable_hierarchical_cache = enable_hierarchical_cache

        self.waiting_queue_radix_tree = RadixCache.create_simulated()

    def calc_priority(self, waiting_queue: List[Req]) -> bool:
        policy = self._determine_active_policy(waiting_queue)

        prefix_computed = False
        if isinstance(policy, CacheAwarePolicy):
            prefix_computed = True
            temporary_deprioritized = self._compute_prefix_matches(
                waiting_queue, policy
            )
            if policy == CacheAwarePolicy.LPM:
                SchedulePolicy._sort_by_longest_prefix(
                    waiting_queue, temporary_deprioritized
                )
            elif policy == CacheAwarePolicy.DFS_WEIGHT:
                SchedulePolicy._sort_by_dfs_weight(waiting_queue, self.tree_cache)
            else:
                raise ValueError(f"Unknown CacheAware Policy: {policy=}")
        else:
            if policy == CacheAgnosticPolicy.FCFS:
                pass
            elif policy == CacheAgnosticPolicy.LOF:
                SchedulePolicy._sort_by_longest_output(waiting_queue)
            elif policy == CacheAgnosticPolicy.RANDOM:
                SchedulePolicy._sort_randomly(waiting_queue)
            else:
                raise ValueError(f"Unknown CacheAgnostic Policy: {policy=}")

        return prefix_computed

    def _determine_active_policy(self, waiting_queue: List[Req]) -> Policy:
        if len(waiting_queue) > 128 and self.policy == CacheAwarePolicy.LPM:
            # Turn off the expensive prefix matching and sorting when the #queue is large.
            return CacheAgnosticPolicy.FCFS
        return self.policy

    def _validate_and_adjust_policy(
        self, policy: str, tree_cache
    ) -> Policy:
        """
        Validates the policy and adjusts it if necessary based on tree cache settings.
        """
        try:
            policy_enum = CacheAwarePolicy(policy)
            if tree_cache.disable:
                # If tree_cache is disabled, using CacheAgnosticPolicy policy
                return CacheAgnosticPolicy.FCFS
            return policy_enum
        except ValueError:
            try:
                return CacheAgnosticPolicy(policy)
            except ValueError:
                raise ValueError(f"Unknown schedule_policy: {policy=}")

    def _compute_prefix_matches(
        self, waiting_queue: List[Req], policy: CacheAwarePolicy
    ) -> Set[int]:
        """
        Computes and caches the matching prefixes for requests in the waiting queue,
            and handles in-batch prefix caching logic.
        """
        temporary_deprioritized: Set[int] = set()
        self.waiting_queue_radix_tree.reset()

        for r in waiting_queue:
            prefix_ids = r.adjust_max_prefix_ids()

            match_result = self.tree_cache.match_prefix(key=prefix_ids)

            (r.prefix_page_ids, r.prefix_len,r.last_node) = (
                match_result.device_indices, match_result.device_prefix_length, match_result.last_device_node
            )

            # NOTE(sang): This logic is for in-batch prefix caching;
            # If there are more than 1 request that have small matching prefix from
            # existing cache, but all those requests share the same prefix, we prefer
            # to schedule only one of them so that we can increase the cache hit rate.
            # We prefer to set IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD > 0 because too small
            # Use prefix_len (token count) instead of len(prefix_page_ids) (page count)
            if len(r.prefix_page_ids) <= IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD:
                match_result = self.waiting_queue_radix_tree.match_prefix(key=prefix_ids)

                in_batch_matching_prefixes = match_result.device_indices

                if (
                    len(in_batch_matching_prefixes)
                    >= IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD
                ):
                    temporary_deprioritized.add(r.rid)
                else:
                    # Insert with a dummy key
                    self.waiting_queue_radix_tree.insert(
                        prefix_ids, torch.empty(len(prefix_ids), dtype=torch.bool)
                    )
        return temporary_deprioritized

    @staticmethod
    def _sort_by_longest_prefix(
        waiting_queue: List[Req], temporary_deprioritized: Set[int]
    ) -> None:
        """Sorts the waiting queue based on the longest prefix match."""
        waiting_queue.sort(
            key=lambda r: (
                -len(r.prefix_page_ids)
                if r.rid not in temporary_deprioritized
                else float("inf")
            )
        )

    @staticmethod
    def _sort_by_dfs_weight(
        waiting_queue: List[Req], tree_cache: RadixCache
    ) -> None:
        """Sorts the waiting queue based on a depth-first search weighting."""
        last_node_to_reqs = defaultdict(list)
        for req in waiting_queue:
            last_node_to_reqs[req.last_node].append(req)

        node_to_weight = defaultdict(int)
        for node in last_node_to_reqs:
            node_to_weight[node] = len(last_node_to_reqs[node])
        SchedulePolicy._calc_weight(tree_cache.root_node, node_to_weight)

        waiting_queue.clear()
        SchedulePolicy._get_dfs_priority(
            tree_cache.root_node,
            node_to_weight,
            last_node_to_reqs,
            waiting_queue,
        )

    @staticmethod
    def _sort_by_longest_output(waiting_queue: List[Req]) -> None:
        """Sorts the waiting queue based on the longest output (max_new_tokens)."""
        waiting_queue.sort(key=lambda x: -x.sampling_params.max_new_tokens)

    @staticmethod
    def _sort_randomly(waiting_queue: List[Req]) -> None:
        """Shuffles the waiting queue randomly."""
        random.shuffle(waiting_queue)

    @staticmethod
    def _calc_weight(cur_node: TreeNode, node_to_weight: Dict[TreeNode, int]) -> None:
        for child in cur_node.children.values():
            SchedulePolicy._calc_weight(child, node_to_weight)
            node_to_weight[cur_node] += node_to_weight[child]

    @staticmethod
    def _get_dfs_priority(
        cur_node: TreeNode,
        node_to_priority: Dict[TreeNode, int],
        last_node_to_reqs: Dict[TreeNode, List[Req]],
        q: List,
    ) -> None:
        childs = [child for child in cur_node.children.values()]
        childs.sort(key=lambda x: -node_to_priority[x])
        for child in childs:
            SchedulePolicy._get_dfs_priority(
                child, node_to_priority, last_node_to_reqs, q
            )
        q.extend(last_node_to_reqs[cur_node])


class AddReqResult(Enum):
    CONTINUE = auto()  # Continue to add requests
    NO_TOKEN = auto()  # No token left or no req slot
    OTHER = auto()  # can not add any more

class PrefillAdder:
    def __init__(
        self,
        tree_cache: RadixCache,
        token_to_kv_pool_allocator: KVAllocator,
        chunk_size: int,
        total_reserved_input_tokens: int,
        total_reserved_kv_pages: int, #reserved kv pages
        max_request_num: int, # can_run_list<=max_request_num
        cache_hit_metrics: CacheHitMetrics,
    ):
        self.tree_cache = tree_cache
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.remain_input_tokens = chunk_size - total_reserved_input_tokens
        self.total_reserved_pages = total_reserved_kv_pages # remain_kv_tokens use property
        self.page_size = self.token_to_kv_pool_allocator.page_size
        self.max_request_num = max_request_num

        self.can_run_list = []
        self.chunked_req = None
        self.batch_info = []

        # Batch/chunk-level stats for Prefill batch log fields.
        self.log_hit_tokens = 0
        self.log_input_tokens = 0
        self.log_context_tokens = 0
        self.dequeue_durations=[]

        self.cache_hit_metrics = cache_hit_metrics

    @property
    def remain_kv_pages(self):
        return (
            self.token_to_kv_pool_allocator.available_size()
            + self.tree_cache.evictable_size()
        ) - self.total_reserved_pages

    def budget_state(self):
        if self.remain_input_tokens <= 0:
            return AddReqResult.OTHER

        if len(self.can_run_list) >= self.max_request_num:
            return AddReqResult.NO_TOKEN

        if self.remain_kv_pages <= 0:
            return AddReqResult.NO_TOKEN

        return AddReqResult.CONTINUE

    def is_full(self):
        # when kv cache or request slot is not enough, batch is not full, then we can wait for some time
        if self.remain_input_tokens<=0:
            return True

        if len(self.can_run_list)>=self.max_request_num:
            return True

        return False

    def _prefill_one_req(
        self, extend_input_len: int, max_new_tokens: int,
        req: Req,
        hit_len = 0, device_prefix_len: int = 0, host_hit_len: int = 0, storage_hit_len: int = 0,
    ):
        reserved_pages = (
            extend_input_len + max_new_tokens + self.page_size - 1
        ) // self.page_size + 1  # One draft page per request.
        self.total_reserved_pages += reserved_pages
        self.remain_input_tokens -= extend_input_len

        self.log_hit_tokens += hit_len
        self.log_input_tokens += extend_input_len
        self.log_context_tokens += (req.prefix_len + extend_input_len)

        self.cache_hit_metrics.add(extend_input_len + hit_len, hit_len, device_prefix_len, host_hit_len, storage_hit_len)
        self.batch_info.append(PrefillReqInfo(extend_input_len, req.prefix_len+extend_input_len, host_hit_len, storage_hit_len))

    def get_reserved_new_token_num(self, req: Req):
        if req.sampling_params.ignore_eos and global_server_args_dict["disaggregation_mode"]!="prefill":
            reserved_new_token_num=req.sampling_params.max_new_tokens
        else:
            reserved_new_token_num=min(
                req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS_ESTIMATION
            )
        return reserved_new_token_num

    def _add_one_req_inner(self, req: Req,
                           hit_len=0, device_prefix_len: int = 0, host_hit_len: int = 0, storage_hit_len: int = 0,):
        truncated = req.extend_input_len > self.remain_input_tokens
        if truncated:
            input_tokens=self.remain_input_tokens
            reserved_new_token_num=0
            req.chunk_split(input_tokens)
            assert self.chunked_req is None
            self.chunked_req=req
            logger.debug(f"[add_chunked_req] rid={req.rid}, len(fill_ids)={len(req.fill_ids)}, extend_input_len={req.extend_input_len}")
        else:
            input_tokens=req.extend_input_len
            reserved_new_token_num=self.get_reserved_new_token_num(req)

        self.can_run_list.append(req)
        self._prefill_one_req(
            input_tokens,
            reserved_new_token_num,
            req=req,
            hit_len=hit_len,
            device_prefix_len=device_prefix_len,
            host_hit_len=host_hit_len,
            storage_hit_len=storage_hit_len,
        )
        assert self.remain_kv_pages >= 0

    @contextmanager
    def _lock_node(self, last_node: TreeNode):
        try:
            self.tree_cache.inc_lock_ref(last_node)
            yield None
        finally:
            self.tree_cache.dec_lock_ref(last_node)

    def get_req_reserved_kv_pages(self, req: Req, host_hit_len=0):
        extend_input_len = req.extend_input_len - host_hit_len
        truncated = extend_input_len > self.remain_input_tokens
        if truncated:
            reserved_kv_tokens=self.remain_input_tokens + host_hit_len
        else:
            reserved_kv_tokens=extend_input_len + self.get_reserved_new_token_num(req)  + host_hit_len

        reserved_kv_pages = (
            reserved_kv_tokens + self.page_size - 1
        ) // self.page_size + 1
        return reserved_kv_pages

    def add_chunked_req(self, req: Req):
        if self.remain_input_tokens <= 0:
            return AddReqResult.OTHER

        if len(self.can_run_list) >= self.max_request_num:
            return AddReqResult.NO_TOKEN

        reserved_kv_pages = self.get_req_reserved_kv_pages(req)
        if reserved_kv_pages > self.remain_kv_pages:
            return AddReqResult.NO_TOKEN

        self._add_one_req_inner(req)
        return self.budget_state()

    def add_one_req(self, req: Req):
        if self.remain_input_tokens <= 0:
            return AddReqResult.OTHER

        if len(self.can_run_list) >= self.max_request_num:
            return AddReqResult.NO_TOKEN

        with self._lock_node(req.last_node):
            reserved_kv_pages = self.get_req_reserved_kv_pages(
                req, req.host_hit_length
            )
            if reserved_kv_pages>self.remain_kv_pages:
                return AddReqResult.NO_TOKEN

            # Calculate L1, L2, L3 hit lengths
            # L1: Original GPU cache (device_prefix_len, already in GPU)
            l1_hit_len = req.prefix_len
            prefix_len=req.prefix_len
            # L2: Host cache that will be loaded back (host native cache)
            if req.l2_cache_hit_len:
                l2_hit_len = min(req.l2_cache_hit_len, req.host_hit_length)
                l3_hit_len = req.host_hit_length - l2_hit_len
            else:
                l2_hit_len = 0
                l3_hit_len = req.host_hit_length

            if req.host_hit_length > 0:
                load_back_result = self.tree_cache.init_load_back(
                    req.last_host_node, req.host_hit_length
                )
                if load_back_result is None:
                    return AddReqResult.NO_TOKEN
                new_indices, req.last_node = load_back_result
                if len(new_indices) > 0:
                    req.prefix_page_ids = torch.cat([req.prefix_page_ids.to(new_indices.device), new_indices])
                    prefix_len = len(req.prefix_page_ids) * self.token_to_kv_pool_allocator.page_size
                    req.prefix_len += req.host_hit_length
                    assert prefix_len==req.prefix_len, (
                            f"prefix len should equal to page ids size "
                            f"rid={req.rid}, pages_len={prefix_len}, prefix_len={req.prefix_len}"
                        )

                    req.extend_input_len -= req.host_hit_length

                    if req.sampling_params.max_new_tokens > 0:
                        assert req.extend_input_len > 0, (
                            f"extend_input_len should be > 0 for generation requests. "
                            f"rid={req.rid}, extend_input_len={req.extend_input_len}, "
                            f"prefix_len={req.prefix_len}, input_len={len(req.fill_ids)}"
                        )
                else:
                    l2_hit_len = 0
                    l3_hit_len = 0

            self.tree_cache.inc_lock_ref(req.last_node)
            self._add_one_req_inner(req,
                                    hit_len=prefix_len,
                                    device_prefix_len=l1_hit_len,
                                    host_hit_len=l2_hit_len,
                                    storage_hit_len=l3_hit_len)

        return self.budget_state()
