from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

import torch

from sglang.srt.disaggregation.kv_events import StorageMedium
from sglang.srt.mem_cache.hicache_operation import L2PoolEntry, LayerDoneCounter
from sglang.srt.mem_cache.l2.controller import L2CacheController
from sglang.srt.mem_cache.l2.mixin_evict import L2MixinEvict
from sglang.srt.mem_cache.l2.mixin_load import L2MixinLoad
from sglang.srt.mem_cache.l2.mixin_write import L2MixinWrite
from sglang.srt.mem_cache.l3.controller import L3CacheController
from sglang.srt.mem_cache.l3.mixin_prefetch import L3MixinPrefetch
from sglang.srt.mem_cache.l3.mixin_write import L3MixinWrite
from sglang.srt.mem_cache.l3.status_monitor import parse_l3_status_config
from sglang.srt.mem_cache.memory_pool import create_host_token_to_kv_pool
from sglang.srt.mem_cache.radix_cache import (
    HiCacheInitParams,
    MatchResult,
    RadixCache,
    TreeNode,
)
from sglang.srt.mem_cache.utils import compute_node_hash_values
from sglang.srt.observability.metrics.collector import StorageMetricsCollector

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

from sglang.srt.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


class HiCacheEngine(L3MixinWrite, L2MixinWrite, L2MixinEvict, L2MixinLoad, L3MixinPrefetch, RadixCache):
    @staticmethod
    def _is_homogeneous_with_target(target_pool, draft_pool) -> bool:
        """draft 能否与 target 共用同一 host pool (靠 layer_offset 在层维度拼接)。

        条件是二者 host 侧每 token 的结构完全一致: MTP(NEXTN) 的 draft 与 target 同为
        MLA/DSA -> True; DFLASH 的 draft 是 GQA (k/v 双份 [H,D]) 而 target 是 DSA
        (kv_lora+rope+index_k), 每 token 字节数与布局均不同 -> False, 须各建 host pool。
        """
        from sglang.srt.mem_cache.memory_pool.memory_pool_l15_wrapper import (
            TokenToKVPoolL15Wrapper,
        )

        def unwrap(p):
            return p._inner if isinstance(p, TokenToKVPoolL15Wrapper) else p

        return type(unwrap(target_pool)) is type(unwrap(draft_pool))

    @staticmethod
    def _host_bytes_per_token(device_pool, extra_layer_num: int = 0) -> int:
        """按对应 Host pool 的 get_size_per_token 同款公式, 从 device pool 属性折算。"""
        from sglang.srt.mem_cache.memory_pool import (
            DSATokenToKVPool,
            MHATokenToKVPool,
            MLATokenToKVPool,
        )
        from sglang.srt.mem_cache.memory_pool.memory_pool_l15_wrapper import (
            TokenToKVPoolL15Wrapper,
        )

        p = device_pool._inner if isinstance(device_pool, TokenToKVPoolL15Wrapper) else device_pool
        layer_num = p.layer_num + extra_layer_num
        if isinstance(p, DSATokenToKVPool):
            kv = p.kv_cache_dim * p.store_dtype.itemsize * layer_num
            index_layers = sum(getattr(p, "indexer_layer_mask", (True,) * p.layer_num)) + extra_layer_num
            index = p.index_head_dim * p.index_k_with_scale_buffer_dtype.itemsize * index_layers
            return kv + index
        if isinstance(p, MLATokenToKVPool):
            return p.kv_cache_dim * p.store_dtype.itemsize * layer_num
        if isinstance(p, MHATokenToKVPool):
            return p.head_num * p.head_dim * layer_num * p.store_dtype.itemsize * 2
        raise ValueError(f"未知的 device pool 类型: {type(p)}")

    def _hetero_draft_budget_scale(self, extra_layer_num: int) -> float:
        """异构 draft 的 host 成本并入预算的收缩系数。

        每全局页的本 rank 成本 = target_bpt/group + Σdraft_bpt (target 按 l1.5 页
        分片, draft 按 head 分片, 每 rank 全页)。
        """
        group = getattr(self.token_to_kv_pool, "attn_tp_group_size", 1)
        target_bpt = self._host_bytes_per_token(self.token_to_kv_pool, extra_layer_num)
        draft_bpt = sum(self._host_bytes_per_token(p) for p in self._hetero_drafts)
        return target_bpt / (target_bpt + group * draft_bpt)

    def _check_hetero_draft_page_domain(self, kv_allocator) -> None:
        """校验异构 draft 用全局页号直接索引的两个不变式 (device / host 各一)。

        异构 draft (DFLASH GQA) 不随 target 做 l1.5 分片, 两侧都用未换算的全局页号寻址,
        因此各自的池必须覆盖对应发放方的整个值域。越界在 NPU 上不报错, 只静默掉 accept。
        """
        host_domain = self.token_to_kv_pool_host.global_page_num
        dev_domain = kv_allocator.size // self.page_size
        for dev_pool, host_pool in zip(self._hetero_drafts, self.draft_kv_pools_host):
            if host_pool.page_num < host_domain:
                raise RuntimeError(f"异构 draft host pool 页数 {host_pool.page_num} < target 全局 host 页域 {host_domain}: 全局 host 页号会越界")
            dev_pages = dev_pool.size // self.page_size
            if dev_pages < dev_domain:
                raise RuntimeError(
                    f"异构 draft device pool 页数 {dev_pages} < kv_allocator 页域 {dev_domain}: "
                    f"全局 device 页号会越界 (见 attn_initializer 的 slot_domain_size)"
                )
            logger.info(f"异构 draft 页域校验通过: host {host_pool.page_num}>={host_domain}, device {dev_pages}>={dev_domain}")

    def __init__(self, params: HiCacheInitParams, server_args: ServerArgs):
        self.page_size = params.page_size
        self.token_to_kv_pool = params.token_to_kv_pool
        self.scheduler_latency_metrics = None

        draft_pools = params.draft_token_to_kv_pools or []
        # 同构的 draft 拼进 target 的 host pool (extra_layer_num); 异构的各建一个。
        homo_drafts: list = []
        hetero_drafts: list = []
        for p in draft_pools:
            if self._is_homogeneous_with_target(self.token_to_kv_pool, p):
                homo_drafts.append(p)
            else:
                hetero_drafts.append(p)
        extra_layer_num = sum(p.layer_num for p in homo_drafts)
        self._homo_drafts = homo_drafts
        self._hetero_drafts = hetero_drafts

        hicache_size = server_args.hicache_size
        hicache_ratio = server_args.hicache_ratio
        if hetero_drafts:
            # 异构 draft 的 host 成本并入预算: 否则末 PP stage 占用远超 --hicache-size。
            scale = self._hetero_draft_budget_scale(extra_layer_num)
            hicache_size = hicache_size * scale
            hicache_ratio = hicache_ratio * scale
            logger.info(
                f"异构 draft host 成本并入预算: scale={scale:.4f}, "
                f"hicache_size {server_args.hicache_size} -> {hicache_size:.2f} GB, "
                f"hicache_ratio {server_args.hicache_ratio} -> {hicache_ratio:.2f}"
            )

        self.token_to_kv_pool_host = create_host_token_to_kv_pool(
            self.token_to_kv_pool,
            self.page_size,
            hicache_ratio=hicache_ratio,
            hicache_size=hicache_size,
            hicache_mem_layout=server_args.hicache_mem_layout,
            hicache_io_backend=server_args.hicache_io_backend,
            extra_layer_num=extra_layer_num,
        )
        # 异构 draft 的独立 host pool。token 数按 target 的全局页数对齐: 二者共用同一套
        # 全局 page 号 (由同一 allocator 发放), 各自按 GB 反算会因每 token 字节数不同而
        # 得到不同的 page_num, 页号域错位。
        draft_host_size_tokens = self.token_to_kv_pool_host.global_page_num * self.page_size
        self.draft_kv_pools_host = [
            create_host_token_to_kv_pool(
                p,
                self.page_size,
                hicache_ratio=hicache_ratio,
                hicache_size=hicache_size,
                hicache_mem_layout=server_args.hicache_mem_layout,
                hicache_io_backend=server_args.hicache_io_backend,
                host_size_tokens=draft_host_size_tokens,
            )
            for p in hetero_drafts
        ]
        self._check_hetero_draft_page_domain(params.token_to_kv_pool_allocator)

        self.tp_pp_group = params.tp_pp_group
        self.tp_pp_size = torch.distributed.get_world_size(group=self.tp_pp_group)
        self.tp_rank = params.attn_tp_rank
        self.tp_size = server_args.attn_tp_size
        self.dp_rank = params.dp_rank
        self.pp_rank = params.pp_rank
        self.pp_size = server_args.pp_size

        self.enable_storage = server_args.hicache_storage_backend is not None
        self.enable_storage_metrics = params.enable_metrics

        (
            extra_config,
            prefetch_threshold,
            prefetch_timeout_base,
            prefetch_timeout_per_ki_token,
            prefetch_timeout_max,
            hicache_storage_pass_prefix_keys,
            backup_inflight_high,
            backup_inflight_hard,
            backup_pacing_interval,
            l3_failure_threshold,
            l3_recovery_interval_sec,
        ) = self.parse_storage_backend_extra_config(
            server_args.hicache_storage_backend_extra_config,
            server_args.hicache_storage_backend,
        )
        self.prefetch_threshold = prefetch_threshold
        self.prefetch_timeout_base = prefetch_timeout_base
        self.prefetch_timeout_per_page = self.page_size / 1024 * prefetch_timeout_per_ki_token
        self.prefetch_timeout_max = prefetch_timeout_max
        self.hicache_storage_pass_prefix_keys = hicache_storage_pass_prefix_keys
        self.prefetch_stop_policy = server_args.hicache_storage_prefetch_policy
        self.backup_inflight_high = backup_inflight_high
        self.backup_inflight_hard = backup_inflight_hard
        self.backup_pacing_interval = backup_pacing_interval
        self.l3_failure_threshold = l3_failure_threshold
        self.l3_recovery_interval_sec = l3_recovery_interval_sec

        target_layer_num = self.token_to_kv_pool.layer_num
        target_entry = L2PoolEntry(
            device_kv_pool=self.token_to_kv_pool,
            host_kv_pool=self.token_to_kv_pool_host,
            layer_done_counter=LayerDoneCounter(target_layer_num, self.pp_size * 4),
            layer_num=target_layer_num,
            layer_offset=0,
        )
        pool_entries = [target_entry]

        # 同构 draft: 拼进 target 的 host pool, 用 layer_offset 定位。
        offset = target_layer_num
        for draft_device_pool in self._homo_drafts:
            pool_entries.append(
                L2PoolEntry(
                    device_kv_pool=draft_device_pool,
                    host_kv_pool=self.token_to_kv_pool_host,
                    layer_done_counter=LayerDoneCounter(draft_device_pool.layer_num, self.pp_size * 4),
                    layer_num=draft_device_pool.layer_num,
                    layer_offset=offset,
                )
            )
            offset += draft_device_pool.layer_num

        # 异构 draft (如 DFLASH 的 GQA): 各用独立 host pool, 层从 0 起算, 全局页号寻址。
        for draft_device_pool, draft_host_pool in zip(self._hetero_drafts, self.draft_kv_pools_host):
            pool_entries.append(
                L2PoolEntry(
                    device_kv_pool=draft_device_pool,
                    host_kv_pool=draft_host_pool,
                    layer_done_counter=LayerDoneCounter(draft_device_pool.layer_num, self.pp_size * 4),
                    layer_num=draft_device_pool.layer_num,
                    layer_offset=0,
                    use_global_pages=True,
                )
            )

        self.l2_ctrl = L2CacheController(
            params.token_to_kv_pool_allocator,
            pool_entries,
            self.page_size,
            write_policy=server_args.hicache_write_policy,
            io_backend=server_args.hicache_io_backend,
        )

        self.l3_ctrl: L3CacheController | None = None
        if self.enable_storage:
            from sglang.srt.mem_cache.l3.storage import HiCacheStorageConfig
            from sglang.srt.mem_cache.utils import get_hash_str

            is_mla_backend = self.token_to_kv_pool.is_mla_backend
            enable_l15 = getattr(self.token_to_kv_pool, "enable_mla_l1_5_cache", False)
            storage_config = HiCacheStorageConfig(
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                is_mla_model=is_mla_backend,
                is_page_first_layout=self.token_to_kv_pool_host.layout == "page_first",
                model_name=server_args.served_model_name,
                pp_rank=self.pp_rank,
                pp_size=self.pp_size,
                extra_config=extra_config,
                enable_l15=enable_l15,
            )
            backup_skip = is_mla_backend and self.tp_rank != 0 and not enable_l15
            self.l3_ctrl = L3CacheController(
                mem_pool_host=self.token_to_kv_pool_host,
                mem_pool_device=self.token_to_kv_pool,
                page_size=self.page_size,
                stop_event=self.l2_ctrl.stop_event,
                storage_backend_type=server_args.hicache_storage_backend,
                storage_config=storage_config,
                prefetch_threshold=max(self.prefetch_threshold, self.page_size),
                storage_batch_size=128,
                tp_world_size=self.tp_size,
                pp_world_size=self.pp_size,
                backup_skip=backup_skip,
                get_hash_str_fn=get_hash_str,
                prefetch_stop_policy=self.prefetch_stop_policy,
                prefetch_timeout_base=self.prefetch_timeout_base,
                prefetch_timeout_per_page=self.prefetch_timeout_per_page,
                prefetch_timeout_max=self.prefetch_timeout_max,
                backup_inflight_high=self.backup_inflight_high,
                backup_inflight_hard=self.backup_inflight_hard,
                backup_pacing_interval=self.backup_pacing_interval,
                l3_failure_threshold=self.l3_failure_threshold,
                l3_recovery_interval_sec=self.l3_recovery_interval_sec,
            )
            if enable_l15:
                from sglang.srt.mem_cache.l3.controller_l15_wrapper import L3CacheControllerL15Wrapper

                is_mooncake = server_args.hicache_storage_backend == "mooncake"
                self.l3_ctrl = L3CacheControllerL15Wrapper(self.l3_ctrl, is_mooncake=is_mooncake)
            self.l3_ctrl.restart_threads()

        if self.enable_storage_metrics:
            labels = {
                "storage_backend": server_args.hicache_storage_backend,
                "tp_rank": self.tp_rank,
                "dp_rank": self.dp_rank,
            }
            self.storage_metrics_collector = StorageMetricsCollector(labels=labels)

        self.write_through_threshold = 1 if server_args.hicache_write_policy == "write_through" else 2
        self.load_back_threshold = 10

        self.device = self.token_to_kv_pool.device
        self.is_decode = False
        self.prefill_kv_manager = None
        super().__init__(params=params)

    def get_node_prefix_keys(self, node: TreeNode) -> list[str] | None:
        prefix_keys = None
        if self.hicache_storage_pass_prefix_keys:
            prefix_keys = node.get_prefix_hash_values(node.parent)

        if getattr(self, "pp_size", 1) > 1:
            pp_prefix = f"pp{self.pp_rank}"
            if prefix_keys is None:
                return [pp_prefix]
            return [pp_prefix, *prefix_keys]

        return prefix_keys

    def parse_storage_backend_extra_config(
        self,
        storage_backend_extra_config: str | None,
        storage_backend_type: str | None = None,
    ):
        extra_config = {}
        if storage_backend_extra_config:
            try:
                extra_config = json.loads(storage_backend_extra_config)
            except Exception as e:
                logger.error(f"Invalid backend extra config JSON: {e}")
                raise e

        prefetch_threshold = extra_config.pop("prefetch_threshold", 256)
        prefetch_timeout_base = extra_config.pop("prefetch_timeout_base", 2.0)
        prefetch_timeout_per_ki_token = extra_config.pop("prefetch_timeout_per_ki_token", 0.1)
        prefetch_timeout_max = extra_config.pop("prefetch_timeout_max", 30.0)
        hicache_storage_pass_prefix_keys = extra_config.pop("hicache_storage_pass_prefix_keys", False)

        backup_inflight_high = extra_config.pop("backup_inflight_high", 0)
        backup_inflight_hard = extra_config.pop("backup_inflight_hard", 0)
        backup_pacing_interval = extra_config.pop("backup_pacing_interval", 0.005)
        l3_failure_threshold, l3_recovery_interval_sec = parse_l3_status_config(extra_config, storage_backend_type)

        if not isinstance(prefetch_threshold, int):
            raise ValueError(f"prefetch_threshold must be int, got {type(prefetch_threshold).__name__}")
        if not isinstance(prefetch_timeout_base, (int, float)):
            raise ValueError(f"prefetch_timeout_base must be number, got {type(prefetch_timeout_base).__name__}")
        if not isinstance(prefetch_timeout_per_ki_token, (int, float)):
            raise ValueError(f"prefetch_timeout_per_ki_token must be number, got {type(prefetch_timeout_per_ki_token).__name__}")
        if not isinstance(prefetch_timeout_max, (int, float)):
            raise ValueError(f"prefetch_timeout_max must be number, got {type(prefetch_timeout_max).__name__}")
        if prefetch_timeout_max <= 0:
            raise ValueError("prefetch_timeout_max must be greater than 0")
        if not isinstance(backup_inflight_high, int) or not isinstance(backup_inflight_hard, int):
            raise ValueError("backup_inflight_high/backup_inflight_hard must be int")
        if not isinstance(backup_pacing_interval, (int, float)):
            raise ValueError(f"backup_pacing_interval must be number, got {type(backup_pacing_interval).__name__}")

        if backup_inflight_high > 0 and backup_inflight_hard > 0 and backup_inflight_hard < backup_inflight_high:
            raise ValueError(f"backup_inflight_hard({backup_inflight_hard}) must be >= backup_inflight_high({backup_inflight_high})")

        return (
            extra_config,
            prefetch_threshold,
            float(prefetch_timeout_base),
            float(prefetch_timeout_per_ki_token),
            float(prefetch_timeout_max),
            hicache_storage_pass_prefix_keys,
            backup_inflight_high,
            backup_inflight_hard,
            float(backup_pacing_interval),
            l3_failure_threshold,
            float(l3_recovery_interval_sec),
        )

    def reset(self):
        TreeNode.counter = 0
        self.l2_ctrl.stop()
        if self.enable_storage:
            self.l3_ctrl.join_threads()
        self.l2_ctrl.join_and_restart()
        if self.enable_storage:
            self.l3_ctrl.restart_threads()
        self.token_to_kv_pool_host.clear()
        # 异构 draft 的独立 host pool 一并清空 (页号与 target 对齐, 残留会脏读)。
        for draft_host_pool in self.draft_kv_pools_host:
            draft_host_pool.clear()
        super().reset()

    def _get_height(self, node: TreeNode):
        height = 0
        while node != self.root_node:
            node = node.parent
            height += 1
        return height

    def clear_storage_backend(self) -> bool:
        if self.enable_storage:
            try:
                if hasattr(self.l3_ctrl.storage_backend, "clear"):
                    self.l3_ctrl.storage_backend.clear()
                    logger.info("Hierarchical cache storage backend cleared successfully!")
                    return True
                else:
                    logger.warning(f"Storage backend {type(self.l3_ctrl.storage_backend).__name__} does not support clear operation.")
                    return False
            except Exception as e:
                logger.error(f"Failed to clear hierarchical cache storage backend: {e}")
                return False
        else:
            logger.warning("Hierarchical cache storage backend is not enabled.")
            return False

    def evictable_size(self):
        return self.evictable_size_

    def tick(self):
        finish_counts = []
        finish_counts.append(self.check_l2_writing_progress())
        finish_counts.append(self.check_l2_loading_progress())
        if self.enable_storage:
            finish_counts.extend(self.check_l3_progress())

        if self.tp_pp_size > 1:
            finish_counts_tensor = torch.tensor(finish_counts, dtype=torch.int, device="cpu")
            torch.distributed.all_reduce(
                finish_counts_tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_pp_group,
            )
            finish_counts = finish_counts_tensor.tolist()
        self.process_l2_writes(finish_counts[0])
        self.process_l2_loads(finish_counts[1])
        if self.enable_storage:
            self.process_l3_feedback(finish_counts[2:])

        # update l3 metrics
        if self.enable_storage_metrics and self.enable_storage:
            self.storage_metrics_collector.log_storage_metrics(self.l3_ctrl.storage_backend.get_stats())

    def match_prefix(self, key, **kwargs):
        if self.disable or len(key) == 0:
            return self._empty_match_result()

        if not isinstance(key[0], tuple):
            full_page_num = len(key) // self.page_size
            paged_token_ids = [tuple(key[i * self.page_size : (i + 1) * self.page_size]) for i in range(0, full_page_num)]
        else:
            paged_token_ids = key
        if len(paged_token_ids) == 0:
            return self._empty_match_result()

        value, last_node = self._match_prefix_helper(self.root_node, paged_token_ids)
        value = torch.cat(value) if value else torch.tensor([], dtype=torch.int32)

        host_hit_length = 0
        last_host_node = last_node
        while last_node.evicted:
            host_hit_length += len(last_node.key) * self.page_size
            last_node = last_node.parent

        if self.is_decode:
            return MatchResult(
                device_indices=value,
                last_device_node=last_node,
                last_host_node=last_node,
                host_hit_length=0,
                device_prefix_length=len(value) * self.page_size,
            )

        while not last_host_node.backuped:
            last_host_node = last_host_node.parent

        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_host_node,
            host_hit_length=host_hit_length,
            device_prefix_length=len(value) * self.page_size,
        )

    def insert(
        self,
        key: list,
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

        while len(key) > 0 and child_key in node.children:
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            node.priority = max(node.priority, priority)
            prefix_len = self.key_match_fn(node.key, key)

            if prefix_len != len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                new_node.priority = max(new_node.priority, priority)
                node = new_node

            if node.evicted:
                # device page is evicted, but host page not — re-materialize on device
                node.value = value[:prefix_len]
                self.evictable_size_ += len(node.value)
                if self.enable_kv_cache_events:
                    self._record_store_event(node, StorageMedium.GPU)
            else:
                self.touch_tree_node(node, chunked)
                total_prefix_length += prefix_len

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

            # Needed for L3 storage and KV-event parent_block_hash chains.
            if self.enable_storage or self.enable_kv_cache_events:
                new_node.hash_value = compute_node_hash_values(new_node, self.page_size)

            if self.enable_kv_cache_events:
                self._record_store_event(new_node, StorageMedium.GPU)

            self.touch_tree_node(new_node, chunked)
        return total_prefix_length

    def insert_host(self, node: TreeNode, key: list, host_value, hash_value):
        node.last_access_time = time.monotonic()
        if len(key) == 0:
            return 0

        child_key = key[0]
        matched_length = 0
        while len(key) > 0 and child_key in node.children:
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(node.key, key)
            key = key[prefix_len:]
            host_value = host_value[prefix_len:]
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
            new_node.host_value = host_value
            new_node.hash_value = hash_value
            node.children[child_key] = new_node
        return matched_length

    def _match_prefix_helper(self, node: TreeNode, key: list):
        node.last_access_time = time.monotonic()
        child_key = key[0]
        value = []

        while len(key) > 0 and child_key in node.children:
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

    def _split_node(self, key: list, child: TreeNode, split_len: int):
        new_node = TreeNode(priority=child.priority)
        new_node.children = {key[split_len]: child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.hit_count = child.hit_count

        if child.evicted:
            new_node.value = None
        else:
            new_node.value = child.value[:split_len]
            child.value = child.value[split_len:]

        if child.backuped:
            new_node.host_value = child.host_value[:split_len]
            child.host_value = child.host_value[split_len:]

        if child.hash_value:
            new_node.hash_value = child.hash_value[:split_len]
            child.hash_value = child.hash_value[split_len:]
        else:
            new_node.hash_value = None

        child.parent = new_node
        child.key = child.key[split_len:]
        new_node.parent.children[key[0]] = new_node
        return new_node

    def evict(self, num_pages: int):
        return self.evict_device_pages(None, num_pages)
