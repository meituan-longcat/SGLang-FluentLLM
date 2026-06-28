from __future__ import annotations

import heapq
import json
import time
from typing import TYPE_CHECKING, List, Optional, Callable

import torch

from sglang.srt.managers.cache_controller import HiCacheController, PrefetchOperation
from sglang.srt.mem_cache.base_prefix_cache import MatchResult
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import (
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
)
from sglang.srt.mem_cache.radix_cache import (
    RadixCache,
    CacheInitParams,
    TreeNode,
    compute_node_hash_values,
)
from sglang.srt.metrics.collector import StorageMetricsCollector

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

from sglang.srt.utils import get_colorful_logger
logger = get_colorful_logger(__name__)


class HiRadixCache(RadixCache):

    def __init__(self, params: CacheInitParams, server_args: ServerArgs):
        if server_args.hicache_io_backend == "direct":
            # FIXME: move this logic into server_args parsing
            if server_args.hicache_mem_layout == "page_first":
                server_args.hicache_mem_layout = "page_first_direct"
                logger.warning(
                    "Page first layout is not supported with direct IO backend, switching to page first direct layout"
                )

        self.page_size = params.page_size
        self.token_to_kv_pool = params.token_to_kv_pool
        if isinstance(self.token_to_kv_pool, MHATokenToKVPool):
            self.token_to_kv_pool_host = MHATokenToKVPoolHost(
                self.token_to_kv_pool,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
            )
        elif isinstance(self.token_to_kv_pool, MLATokenToKVPool):
            self.token_to_kv_pool_host = MLATokenToKVPoolHost(
                self.token_to_kv_pool,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
            )
        else:
            raise ValueError(f"HiRadixCache only supports MHA and MLA yet")

        self.tp_group = params.tp_cache_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)
        self.enable_storage = server_args.hicache_storage_backend is not None
        # tmp stop storage metrics
        self.enable_storage_metrics = None #self.enable_storage and params.enable_metrics 

        (
            extra_config,
            prefetch_threshold,
            prefetch_timeout_base,
            prefetch_timeout_per_ki_token,
            hicache_storage_pass_prefix_keys,
        ) = self._parse_storage_backend_extra_config(
            server_args.hicache_storage_backend_extra_config
        )
        self.prefetch_threshold = prefetch_threshold
        self.prefetch_timeout_base = prefetch_timeout_base
        self.prefetch_timeout_per_page = (
            self.page_size / 1024 * prefetch_timeout_per_ki_token
        )
        self.hicache_storage_pass_prefix_keys = hicache_storage_pass_prefix_keys
        # TODO: support more timeout check functions
        self.is_prefetch_timeout = self._prefetch_timeout_check_linear_func
        self.prefetch_stop_policy = server_args.hicache_storage_prefetch_policy

        self.cache_controller = HiCacheController(
            params.token_to_kv_pool_allocator,
            self.token_to_kv_pool_host,
            self.token_to_kv_pool,
            self.page_size,
            self.tp_group,
            write_policy=server_args.hicache_write_policy,
            io_backend=server_args.hicache_io_backend,
            storage_backend=server_args.hicache_storage_backend,
            prefetch_threshold=self.prefetch_threshold,
            model_name=server_args.served_model_name,
            storage_backend_extra_config=extra_config,
            is_dp_attention_enabled=server_args.enable_dp_attention,
        )
        if self.enable_storage_metrics:
            # TODO: support pp
            labels = {
                "storage_backend": server_args.hicache_storage_backend,
                "tp_rank": self.cache_controller.tp_rank,
                "dp_rank": self.cache_controller.dp_rank,
            }
            self.storage_metrics_collector = StorageMetricsCollector(labels=labels)

        # record the nodes with ongoing write through
        self.ongoing_write_through = {}
        # record the node segments with ongoing load back
        self.ongoing_load_back = {}
        # record the ongoing prefetch requests
        self.ongoing_prefetch = {}
        self.ongoing_backup = {}
        # todo: dynamically adjust the threshold
        self.write_through_threshold = (
            1 if server_args.hicache_write_policy == "write_through" else 2
        )
        self.load_back_threshold = 10

        self.device = self.token_to_kv_pool.device
        self.is_decode = False 
        super().__init__(params=params)

    def _parse_storage_backend_extra_config(
        self, storage_backend_extra_config: Optional[str]
    ):
        """
        Parse storage backend extra config JSON and extract specific parameters.

        Args:
            storage_backend_extra_config: JSON string containing extra configuration

        Returns:
            tuple: (extra_config_dict, prefetch_threshold, prefetch_timeout_base, prefetch_timeout_per_ki_token, hicache_storage_pass_prefix_keys)
        """
        # Parse extra config JSON if provided
        extra_config = {}
        if storage_backend_extra_config:
            try:
                extra_config = json.loads(storage_backend_extra_config)
            except Exception as e:
                logger.error(f"Invalid backend extra config JSON: {e}")
                raise e

        prefetch_threshold = extra_config.pop("prefetch_threshold", 256)  # tokens
        prefetch_timeout_base = extra_config.pop("prefetch_timeout_base", 1)  # seconds
        prefetch_timeout_per_ki_token = extra_config.pop(
            "prefetch_timeout_per_ki_token", 0.25
        )  # seconds per 1024 tokens
        hicache_storage_pass_prefix_keys = extra_config.pop(
            "hicache_storage_pass_prefix_keys", False
        )

        if not isinstance(prefetch_threshold, int):
            raise ValueError(
                f"prefetch_threshold must be int, got {type(prefetch_threshold).__name__}"
            )
        if not isinstance(prefetch_timeout_base, (int, float)):
            raise ValueError(
                f"prefetch_timeout_base must be number, got {type(prefetch_timeout_base).__name__}"
            )
        if not isinstance(prefetch_timeout_per_ki_token, (int, float)):
            raise ValueError(
                f"prefetch_timeout_per_ki_token must be number, got {type(prefetch_timeout_per_ki_token).__name__}"
            )

        return (
            extra_config,
            prefetch_threshold,
            float(prefetch_timeout_base),
            float(prefetch_timeout_per_ki_token),
            hicache_storage_pass_prefix_keys,
        )

    def reset(self):
        TreeNode.counter = 0
        self.cache_controller.reset()
        self.token_to_kv_pool_host.clear()
        super().reset()

    def get_height(self, node: TreeNode):
        height = 0
        while node != self.root_node:
            node = node.parent
            height += 1
        return height

    def clear_storage_backend(self) -> bool:
        if self.enable_storage:
            try:
                # Check if the storage backend has a clear method (for nixl backends)
                if hasattr(self.cache_controller.storage_backend, "clear"):
                    self.cache_controller.storage_backend.clear()
                    logger.info(
                        "Hierarchical cache storage backend cleared successfully!"
                    )
                    return True
                else:
                    logger.warning(
                        f"Storage backend {type(self.cache_controller.storage_backend).__name__} does not support clear operation."
                    )
                    return False
            except Exception as e:
                logger.error(f"Failed to clear hierarchical cache storage backend: {e}")
                return False
        else:
            logger.warning("Hierarchical cache storage backend is not enabled.")
            return False

    def write_backup(self, node: TreeNode, write_back=False):
        """Backup node from device to host memory.
        
        Note: node.value is page indices, node.host_value will be token indices.
        """
        host_indices = self.cache_controller.write(
            device_indices=node.value,
            node_id=node.id,
        )
        if host_indices is None:
            self.evict_host(len(node.value) * self.page_size)  # evict_host needs token count
            host_indices = self.cache_controller.write(
                device_indices=node.value,
                node_id=node.id,
            )
        if host_indices is not None:
            node.host_value = host_indices  # host_indices is token indices
            assert len(node.host_value) > 0
            self.ongoing_write_through[node.id] = node
            if not write_back:
                # no need to lock nodes if write back
                self.inc_lock_ref(node)
        else:
            return 0

        return len(node.value)

    def write_backup_storage(self, node: TreeNode):
        prefix_keys = (
            node.get_prefix_hash_values(node.parent)
            if self.hicache_storage_pass_prefix_keys
            else None
        )

        logger.debug(f"start write storage {node.key=} {node.host_value=} {prefix_keys=}")
        operation_id = self.cache_controller.write_storage(
            node.host_value, node.key, node.hash_value, prefix_keys
        )
        self.ongoing_backup[operation_id] = node
        node.protect_host()

    def _inc_hit_count(self, node: TreeNode, chunked=False):
        # skip the hit count update for chunked requests
        if self.cache_controller.write_policy == "write_back" or chunked:
            return
        node.hit_count += 1

        if not node.backuped:
            if node.hit_count >= self.write_through_threshold:
                # write to host if the node is not backuped
                self.write_backup(node)

    def writing_check(self, write_back=False):
        if write_back:
            # blocking till all write back complete
            while len(self.ongoing_write_through) > 0:
                for _, finish_event, ack_list in self.cache_controller.ack_write_queue:
                    finish_event.synchronize()
                    for ack_id in ack_list:
                        del self.ongoing_write_through[ack_id]
                self.cache_controller.ack_write_queue.clear()
                assert len(self.ongoing_write_through) == 0
            return

        # NOTE: all ranks has the same ongoing_write_through, can skip sync if empty
        if len(self.ongoing_write_through) == 0:
            return

        finish_count = 0
        for _, finish_event, ack_list in self.cache_controller.ack_write_queue:
            if not finish_event.query():
                break
            finish_count += 1
        queue_size = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        if self.tp_world_size > 1:
            # synchronize TP workers to make the same update to radix cache
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )

        finish_count = int(queue_size.item())
        while finish_count > 0:
            _, finish_event, ack_list = self.cache_controller.ack_write_queue.pop(0)
            finish_event.synchronize()
            for ack_id in ack_list:
                backuped_node = self.ongoing_write_through.pop(ack_id)
                self.dec_lock_ref(backuped_node)
                if self.enable_storage:
                    self.write_backup_storage(backuped_node)
            finish_count -= 1

    def loading_check(self):
        finish_count = 0
        for _, finish_event, ack_list in self.cache_controller.ack_load_queue:
            if not finish_event.query():
                # the KV cache loading is still ongoing
                break
            finish_count += 1
            # no need to sync across TP workers as batch forwarding is synced
            for ack_id in ack_list:
                end_node = self.ongoing_load_back.pop(ack_id)
                self.dec_lock_ref(end_node)

        # ACK until all events are processed
        del self.cache_controller.ack_load_queue[:finish_count]

    def evictable_size(self):
        return self.evictable_size_

    def evict(self, num_pages: int, evict_callback: Callable = None):
        logger.debug(f"[evict] start {num_pages=} ")
        start_time = time.perf_counter()
        leaves = self._collect_leaves_device()
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        write_back_nodes = []
        while num_evicted < num_pages and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)

            if x.lock_ref > 0:
                continue

            if not x.backuped:
                if self.cache_controller.write_policy == "write_back":
                    # write to host if the node is not backuped
                    num_evicted += self.write_backup(x, write_back=True)
                    write_back_nodes.append(x)
                else:
                    num_evicted += self._evict_regular(x)
            else:
                num_evicted += self._evict_backuped(x)

            for child in x.parent.children.values():
                if child in write_back_nodes:
                    continue
                if not child.evicted:
                    break
            else:
                # all children are evicted or no children
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

        if self.cache_controller.write_policy == "write_back":
            self.writing_check(write_back=True)
            for node in write_back_nodes:
                assert node.backuped
                self._evict_backuped(node)

        self.cache_controller.mem_pool_device_allocator.free_group_end()
        logger.debug(f"[evict] end {num_pages=} {num_evicted}")
        return num_evicted

    def _evict_backuped(self, node: TreeNode):
        # evict a node already written to host
        self.cache_controller.mem_pool_device_allocator.append_to_later_free(node.value)
        num_evicted = len(node.value)
        self.evictable_size_ -= num_evicted
        node.value = None
        return num_evicted

    def _evict_regular(self, node: TreeNode):
        # evict a node not initiated write to host
        # Use delayed free mechanism (same as radix_cache.py)
        self.cache_controller.mem_pool_device_allocator.append_to_later_free(node.value)
        num_evicted = len(node.value)
        self._delete_leaf(node)
        return num_evicted

    def evict_host(self, num_tokens: int):
        """Evict host KV cache.
        
        Args:
            num_tokens: Number of tokens to evict
        
        Note: node.host_value is token indices.
        """
        leaves = self._collect_leaves()
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)
            if x == self.root_node:
                break
            # only evict the host value of evicted nodes
            if not x.evicted:
                continue

            # node is protected from eviction as it has ongoing prefetch or backup to storage
            if x.host_ref_counter > 0:
                continue

            num_evicted += self.cache_controller.evict_host(x.host_value)

            for k, v in x.parent.children.items():
                if v == x:
                    break
            del x.parent.children[k]

            if len(x.parent.children) == 0 and x.parent.evicted:
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

    def load_back(
        self, node: TreeNode, mem_quota: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        """Load KV cache from host to device.
        
        Args:
            node: The node to load back
            mem_quota: Memory quota in tokens
        
        Returns:
            Device page indices if successful, None otherwise
        """
        # todo: more loading policies

        start_time = time.perf_counter()
        last_hit_node = node
        nodes_to_load = []
        while node.evicted:
            assert (
                node.backuped
            ), "No backup available on evicted nodes, should not happen"
            nodes_to_load.insert(0, node)
            node = node.parent
        else:
            ancester_node = node

        # protect the ancestor nodes from eviction
        delta = self.inc_lock_ref(ancester_node)

        # load it all or not at all
        # host_value is token indices, concatenate all token indices
        host_indices = torch.cat([n.host_value for n in nodes_to_load])
        num_tokens = len(host_indices)
        num_pages = num_tokens // self.page_size
        if num_tokens < self.load_back_threshold or (
            num_tokens > mem_quota + delta if mem_quota is not None else False
        ):
            # skip loading back if the total size is too small or exceeding the memory quota
            self.dec_lock_ref(ancester_node)
            return None

        # Allocate device memory for loading
        # Use the last_hit_node's req_pool_idx if available, otherwise use 0
        req_pool_idx = last_hit_node.req_pool_idx if hasattr(last_hit_node, 'req_pool_idx') and last_hit_node.req_pool_idx is not None else 0
        # Get the already allocated length from req_to_token_pool
        alloced_len = self.req_to_token_pool.alloced_lens[req_pool_idx].item()
        device_token_indices = self.cache_controller.mem_pool_device_allocator.alloc(req_pool_idx, num_tokens, alloced_len)
        
        if device_token_indices is None:
            self.evict(num_pages)
            device_token_indices = self.cache_controller.mem_pool_device_allocator.alloc(req_pool_idx, num_tokens, alloced_len)
        
        self.dec_lock_ref(ancester_node)
        if device_token_indices is None:
            # no sufficient GPU memory to load back KV caches
            return None
        
        # Now call cache_controller.load with pre-allocated device memory
        device_indices = self.cache_controller.load(
            host_indices=host_indices, 
            device_token_indices=device_token_indices,
            node_id=last_hit_node.id
        )
        if device_indices is None:
            # This should not happen since we already allocated device_token_indices
            return None

        self.ongoing_load_back[last_hit_node.id] = last_hit_node
        offset = 0
        for node in nodes_to_load:
            # host_value is token indices, device_indices is page indices
            num_tokens_in_node = len(node.host_value)
            num_pages_in_node = num_tokens_in_node // self.page_size
            node.value = device_indices[offset : offset + num_pages_in_node]
            offset += num_pages_in_node
        self.evictable_size_ += len(device_indices)
        self.inc_lock_ref(last_hit_node)

        return device_indices.to(self.device)

    def init_load_back(
        self,
        last_node: TreeNode,
        host_hit_length: int,
        mem_quota: Optional[int] = None,
    ):
        """Initialize load back from host to device.
        
        Args:
            last_node: The last matched node
            host_hit_length: Number of tokens hit on host
            mem_quota: Memory quota in pages
        
        Returns:
            Tuple of (loaded_page_indices, last_node)
        """
        _ = host_hit_length  # unused, but kept for compatibility
        if last_node.evicted:
            loading_values = self.load_back(last_node, mem_quota)
            if loading_values is not None:
                logger.debug(
                    f"loading back {len(loading_values)} pages for node {last_node.id}"
                )
                return loading_values, last_node

            while last_node.evicted:
                last_node = last_node.parent

        return (
            torch.empty((0,), dtype=torch.int64, device=self.device),
            last_node,
        )

    def ready_to_load_host_cache(self) -> int:
        """
        Notify the cache controller to start the KV cache loading.
        Return the consumer index for the schedule batch manager to track.
        """
        return self.cache_controller.start_loading()

    def check_hicache_events(self):
        self.writing_check()
        self.loading_check()
        if self.enable_storage:
            self.drain_storage_control_queues()
        if self.enable_storage_metrics:
            self.storage_metrics_collector.log_storage_metrics(
                self.cache_controller.storage_backend.get_stats()
            )

    def drain_storage_control_queues(self):
        """
        Combine prefetch revoke, backup ack, and host mem release checks
        to minimize TP synchronization and Python overhead.
        """
        cc = self.cache_controller

        qsizes = torch.tensor(
            [
                cc.prefetch_revoke_queue.qsize(),
                cc.ack_backup_queue.qsize(),
                cc.host_mem_release_queue.qsize(),
            ],
            dtype=torch.int,
        )
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(
                qsizes, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
            )

        n_revoke, n_backup, n_release = map(int, qsizes.tolist())

        # process prefetch revokes
        for _ in range(n_revoke):
            req_id = cc.prefetch_revoke_queue.get()
            info = self.ongoing_prefetch.pop(req_id, None)
            if info is not None:
                last_host_node, token_ids, _, _ = info
                last_host_node.release_host()
                cc.prefetch_tokens_occupied -= len(token_ids)
            # else: the revoked operation already got terminated, nothing to do

        # process backup acks
        for _ in range(n_backup):
            operation = cc.ack_backup_queue.get()
            ack_id = operation.id
            entry = self.ongoing_backup.pop(ack_id, None)
            if entry is not None:
                entry.release_host()
            if self.enable_storage_metrics:
                self.storage_metrics_collector.log_backuped_tokens(
                    operation.completed_tokens
                )

        # release host memory
        # host_indices in queue are token indices
        host_indices_list = []
        for _ in range(n_release):
            host_indices_list.append(cc.host_mem_release_queue.get())
        if host_indices_list:
            host_token_indices = torch.cat(host_indices_list, dim=0)
            cc.mem_pool_host.free(host_token_indices)

    # Timeout is linearly increasing with the number of pages
    def _prefetch_timeout_check_linear_func(self, operation: PrefetchOperation):
        # If hash_value has not been computed in timeout_base seconds, terminate it.
        return (
            time.monotonic() - operation.start_time
            > self.prefetch_timeout_base
            + len(operation.hash_value) * self.prefetch_timeout_per_page
        )

    def can_terminate_prefetch(self, operation: PrefetchOperation):
        can_terminate = True

        if self.prefetch_stop_policy == "best_effort":
            return can_terminate

        if len(operation.hash_value) == 0:
            completed = False
        else:
            completed = (
                operation.completed_tokens == len(operation.hash_value) * self.page_size
            )

        if self.prefetch_stop_policy == "wait_complete":
            can_terminate = completed
        elif self.prefetch_stop_policy == "timeout":
            can_terminate = completed or self.is_prefetch_timeout(operation)
        else:
            # unknown prefetch stop policy, just return True
            return True

        operation_terminated = operation.is_terminated()
        if self.tp_world_size > 1:
            states = torch.tensor(
                [1 - int(can_terminate), int(operation_terminated)],
                dtype=torch.int,
            )
            torch.distributed.all_reduce(
                states,
                op=torch.distributed.ReduceOp.MAX,
                group=self.tp_group,
            )
            can_terminate = states[0].item() == 0
            operation_terminated = states[1].item() == 1
        # the operation should be terminated if it is already terminated on any TP worker
        # or it meets the termination condition on all TP workers
        can_terminate = can_terminate or operation_terminated
        return can_terminate

    def check_prefetch_progress(self, req_id: str) -> bool:
        """Check and finalize prefetch progress.
        
        Note: token_ids is stored as flat token sequence for storage backend,
        but internally we work with pages.
        """
        if req_id not in self.ongoing_prefetch:
            # there is no ongoing prefetch for this request or it has been revoked
            return True

        # todo: more policies for prefetch progress such as timeout
        # the current policy is to prefetch with best effort and terminate when queuing is over
        last_host_node, token_ids, host_indices, operation = self.ongoing_prefetch[
            req_id
        ]

        if operation.host_indices is None:
            # prefetch has not been issued due to insufficient host memory
            return True

        if not self.can_terminate_prefetch(operation):
            return False

        completed_tokens, hash_value = self.cache_controller.terminate_prefetch(
            operation
        )
        logger.debug(f"{req_id} Prefetch completed with {completed_tokens} tokens")

        min_completed_tokens = completed_tokens
        if self.tp_world_size > 1:
            # synchrnoize TP workers to make the same update to hiradix cache
            completed_tokens_tensor = torch.tensor(
                min_completed_tokens, dtype=torch.int
            )
            torch.distributed.all_reduce(
                completed_tokens_tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
            min_completed_tokens = completed_tokens_tensor.item()
        
        # Convert completed tokens to pages
        min_completed_pages = min_completed_tokens // self.page_size
        
        # Convert flat token sequence to paged format for radix tree insertion
        fetched_token_ids = token_ids[:min_completed_tokens]
        paged_token_ids = [
            tuple(fetched_token_ids[i * self.page_size : (i + 1) * self.page_size])
            for i in range(min_completed_pages)
        ]
        
        # host_indices is token indices, pass to _insert_helper_host
        matched_length = self._insert_helper_host(
            last_host_node,
            key=paged_token_ids,
            host_value=host_indices[:min_completed_tokens],
            hash_value=hash_value[:min_completed_pages],
        )

        # Free matched tokens: matched_length is in pages, convert to tokens
        matched_tokens = matched_length * self.page_size
        if matched_tokens > 0:
            self.cache_controller.mem_pool_host.free(host_indices[:matched_tokens])
        # Release all unmatched tokens (from matched_tokens to completed_tokens)
        if completed_tokens > matched_tokens:
            self.cache_controller.append_host_mem_release(
                host_indices[matched_tokens:completed_tokens]
            )
        last_host_node.release_host()
        del self.ongoing_prefetch[req_id]
        self.cache_controller.prefetch_tokens_occupied -= len(token_ids)

        if self.enable_storage_metrics:
            self.storage_metrics_collector.log_prefetched_tokens(
                (min_completed_pages - matched_length) * self.page_size
            )

        return True

    def match_prefix(self, key, **kwargs):
        if self.disable or len(key) == 0:
            return self._empty_match_result()

        # Compatible with whether the incoming key is paged
        if not isinstance(key[0], tuple):
            full_page_num = len(key) // self.page_size
            paged_token_ids = [
                tuple(key[i * self.page_size : (i + 1) * self.page_size])
                for i in range(0, full_page_num)
            ]
        else:
            paged_token_ids = key
        if len(paged_token_ids) == 0:
            return self._empty_match_result()

        value, last_node = self._match_prefix_helper(self.root_node, paged_token_ids)
        if value:
            value = torch.cat(value)
        else:
            value = torch.tensor([], dtype=torch.int32)

        # Calculate host hit length in tokens (host_value is token indices)
        host_hit_length = 0  # in tokens
        last_host_node = last_node
        while last_node.evicted:
            host_hit_length += len(last_node.host_value)  # host_value is token indices
            last_node = last_node.parent
        # Check if it's in decode phase
        
        if self.is_decode:
            # Decode phase: do not return L2/L3 hits
            return MatchResult(
                device_indices=value,
                last_device_node=last_node,
                last_host_node=last_node,
                host_hit_length=0,  # in tokens
                device_prefix_length=len(value) * self.page_size,
            )
        
        # Prefill phase: normally return L2/L3 hits
        while not last_host_node.backuped:
            last_host_node = last_host_node.parent

        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_host_node,
            host_hit_length=host_hit_length,  # in tokens
            device_prefix_length=len(value) * self.page_size,
        )

    def prefetch_from_storage(
        self,
        req_id: str,
        last_host_node: TreeNode,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
    ):
        # new_input_tokens is a flat token sequence, need to align to page size
        # and convert to page count for internal processing
        prefetch_length = len(new_input_tokens) - (
            len(new_input_tokens) % self.page_size
        )
        new_input_tokens = new_input_tokens[:prefetch_length]
        prefetch_page_count = prefetch_length // self.page_size
        
        logger.debug(f"[prefetch_from_storage] req_id={req_id} prefetch_length={prefetch_length} threshold={self.prefetch_threshold} enable_storage={self.enable_storage}")
        if (
            not self.enable_storage
            or prefetch_length < self.prefetch_threshold
            or self.cache_controller.prefetch_rate_limited()
        ):
            logger.debug(f"[prefetch_from_storage] early return: enable_storage={self.enable_storage} "
                         f"prefetch_length={prefetch_length} threshold={self.prefetch_threshold} "
                         f"{self.cache_controller.prefetch_capacity_limit=} {self.cache_controller.prefetch_tokens_occupied=}")
            return

        last_host_node.protect_host()
        # Allocate host memory: mem_pool_host.alloc needs token count
        num_tokens = prefetch_page_count * self.page_size
        host_indices = self.cache_controller.mem_pool_host.alloc(num_tokens)
        if host_indices is None:
            self.evict_host(num_tokens)
            host_indices = self.cache_controller.mem_pool_host.alloc(num_tokens)
        if host_indices is None:
            last_host_node.release_host()
            logger.debug(f"[prefetch_from_storage] no enough host memory")
            # no sufficient host memory for prefetch
            return
        # host_indices is token indices, pass to cache_controller
        operation = self.cache_controller.prefetch(
            req_id, host_indices, new_input_tokens, last_hash, prefix_keys
        )
        self.ongoing_prefetch[req_id] = (
            last_host_node,
            new_input_tokens,  # Keep flat token sequence for storage backend
            host_indices,  # token indices
            operation,
        )
        self.cache_controller.prefetch_tokens_occupied += prefetch_length

    def _insert_helper_host(
        self, node: TreeNode, key: List, host_value, hash_value
    ):
        """Insert host KV cache into radix tree.
        
        Args:
            node: Starting tree node
            key: list with paged token_ids (page granularity)
            host_value: Host token indices (token granularity)
            hash_value: Hash values (page granularity, one per page)
        
        Returns:
            matched_length: Number of pages that matched existing nodes
        """
        node.last_access_time = time.monotonic()
        if len(key) == 0:
            return 0

        child_key = key[0]

        matched_length = 0  # in pages
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(node.key, key)  # in pages
            key = key[prefix_len:]
            # host_value is in tokens, slice by tokens
            host_value = host_value[prefix_len * self.page_size:]
            hash_value = hash_value[prefix_len:]
            matched_length += prefix_len

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            if len(key):
                child_key = key[0]

        if len(key):
            new_node = TreeNode(priority=node.priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = None
            new_node.host_value = host_value  # token indices
            new_node.hash_value = hash_value
            node.children[child_key] = new_node
        return matched_length

    def _match_prefix_helper(self, node: TreeNode, key: List):
        node.last_access_time = time.monotonic()
        child_key = key[0]
        value = []

        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                if not new_node.evicted:
                    value.append(new_node.value)
                node = new_node
                break
            else:
                if not child.evicted:
                    value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = key[0]

        return value, node

    def _split_node(self, key: List, child: TreeNode, split_len: int):
        """Split a tree node at the given position.
        
        Args:
            key: The key being matched (paged tokens)
            child: The child node to split
            split_len: Split position in pages
        
        Returns:
            The new parent node created by splitting
        
        Note:
            - key, value, hash_value: page granularity
            - host_value: token granularity
        """
        # child node split into new_node -> child
        new_node = TreeNode(priority=child.priority)
        new_node.children = {key[split_len]: child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]  # in pages
        new_node.hit_count = child.hit_count

        # split value (in pages) and hash_value (in pages)
        if child.evicted:
            new_node.value = None
        else:
            new_node.value = child.value[:split_len]
            child.value = child.value[split_len:]
        
        # split host_value (in tokens, not pages!)
        if child.backuped:
            split_token_pos = split_len * self.page_size
            new_node.host_value = child.host_value[:split_token_pos]
            child.host_value = child.host_value[split_token_pos:]

        # split hash_value (only if it exists, i.e., when storage is enabled)
        if self.enable_storage:
            new_node.hash_value = child.hash_value[:split_len]
            child.hash_value = child.hash_value[split_len:]
        else:
            new_node.hash_value = None
            
        child.parent = new_node
        child.key = child.key[split_len:]
        new_node.parent.children[key[0]] = new_node
        return new_node

    def insert(
        self,
        key: List,
        value=None,
        chunked: bool = False,
        priority: int | None = None,
    ):
        if priority is None:
            priority = 0

        if len(key) == 0:
            return 0
        node = self.root_node
        child_key = key[0]
        total_prefix_length = 0

        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            node.priority = max(node.priority, priority)
            prefix_len = self.key_match_fn(node.key, key)

            if prefix_len == len(node.key):
                if node.evicted:
                    # change the reference if the node is evicted
                    # this often happens in the case of KV cache recomputation
                    node.value = value[:prefix_len]
                    self.evictable_size_ += len(node.value)
                else:
                    self._inc_hit_count(node, chunked)
                    total_prefix_length += prefix_len
            else:
                # partial match, split the node
                new_node = self._split_node(node.key, node, prefix_len)
                # shared-prefix node should also reflect max priority
                new_node.priority = max(new_node.priority, priority)
                if new_node.evicted:
                    new_node.value = value[:prefix_len]
                    self.evictable_size_ += len(new_node.value)
                else:
                    self._inc_hit_count(new_node, chunked)
                    total_prefix_length += prefix_len
                node = new_node

            key = key[prefix_len:]
            value = value[prefix_len:]

            if len(key):
                child_key = key[0]

        if len(key):
            new_node = TreeNode(priority=priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            node.children[child_key] = new_node
            self.evictable_size_ += len(value)

            # Compute hash_value if storage is enabled
            if self.enable_storage:
                new_node.hash_value = compute_node_hash_values(new_node, self.page_size)

            if self.cache_controller.write_policy != "write_back":
                self._inc_hit_count(new_node, chunked)
        return total_prefix_length

    def _collect_leaves_device(self):
        def is_leaf(node):
            if node.evicted:
                return False
            if node == self.root_node:
                return False
            if len(node.children) == 0:
                return True
            for child in node.children.values():
                if not child.evicted:
                    return False
            return True

        ret_list = []
        stack = [self.root_node]
        while stack:
            cur_node = stack.pop()
            if is_leaf(cur_node):
                ret_list.append(cur_node)
            else:
                for cur_child in cur_node.children.values():
                    if not cur_child.evicted:
                        stack.append(cur_child)
        return ret_list

    def release_aborted_request(self, rid: str):
        """Release resources for an aborted prefetch request."""
        if rid not in self.ongoing_prefetch:
            return

        last_host_node, token_ids, host_indices, operation = self.ongoing_prefetch[rid]
        if operation.host_indices is None:
            return

        completed_tokens, _ = self.cache_controller.terminate_prefetch(operation)
        if self.tp_world_size > 1:
            torch.distributed.barrier(group=self.tp_group)
        last_host_node.release_host()
        del self.ongoing_prefetch[rid]
        # host_indices is token-level, use completed_tokens not completed_pages
        self.cache_controller.append_host_mem_release(host_indices[:completed_tokens])
        self.cache_controller.prefetch_tokens_occupied -= len(token_ids)