from __future__ import annotations

import os
import random
import threading
import time
from dataclasses import dataclass
from queue import Empty, Queue
from typing import TYPE_CHECKING

import torch

from sglang.srt.mem_cache.hicache_operation import H2SWrite, S2HLoad
from sglang.srt.mem_cache.l3.status_monitor import L3StatusMonitor
from sglang.srt.mem_cache.l3.storage import (
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)
from sglang.srt.utils import get_colorful_logger

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import BaseTokenToKVPool
    from sglang.srt.mem_cache.memory_pool.host_memory_pool import HostTokenToKVPool

logger = get_colorful_logger(__name__)

DEFAULT_PREFETCH_SIZE = 4096
TEST_PREFETCH_RANDOM_TIMEOUT = bool(os.getenv("TEST_PREFETCH_RANDOM_TIMEOUT", ""))


def get_mooncake_query_threads() -> int:
    raw_value = os.getenv("MOONCAKE_QUERY_THREADS", "2")
    try:
        query_threads = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"MOONCAKE_QUERY_THREADS must be a positive integer, got {raw_value!r}") from exc
    if query_threads < 1:
        raise ValueError(f"MOONCAKE_QUERY_THREADS must be a positive integer, got {raw_value!r}")
    return query_threads


@dataclass(frozen=True)
class L3QueryTask:
    sequence_id: int
    s2h_load: S2HLoad


@dataclass
class L3QueryResult:
    hash_values: list[str]
    hit_pages: int
    page2rank: list[int] | None = None


@dataclass
class L3QueryCompletion:
    task: L3QueryTask
    result: L3QueryResult | None = None
    error: Exception | None = None


class L3CacheController:
    def __init__(
        self,
        mem_pool_host: HostTokenToKVPool,
        mem_pool_device: BaseTokenToKVPool,
        page_size: int,
        stop_event: threading.Event,
        storage_backend_type: str,
        storage_config: HiCacheStorageConfig,
        prefetch_threshold: int,
        storage_batch_size: int,
        tp_world_size: int,
        pp_world_size: int,
        backup_skip: bool,
        get_hash_str_fn,
        prefetch_stop_policy: str = "timeout",
        prefetch_timeout_base: float = 2.0,
        prefetch_timeout_per_page: float = 0.0,
        prefetch_timeout_max: float = 30.0,
        is_decode: bool = False,
        backup_inflight_high: int = 0,
        backup_inflight_hard: int = 0,
        backup_pacing_interval: float = 0.005,
        l3_failure_threshold: int = 1,
        l3_recovery_interval_sec: float = 30.0,
    ):
        self.mem_pool_host = mem_pool_host
        self.mem_pool_device = mem_pool_device
        self.page_size = page_size
        self.stop_event = stop_event
        self.storage_backend_type = storage_backend_type
        self.query_backend_worker_count = get_mooncake_query_threads() if storage_backend_type == "mooncake" else 1
        self.storage_config = storage_config
        self.backup_skip = backup_skip
        self.storage_batch_size = storage_batch_size
        self.tp_world_size = tp_world_size
        self.pp_world_size = pp_world_size
        self.get_hash_str = get_hash_str_fn
        self.enable_l15 = storage_config.enable_l15
        self.prefetch_stop_policy = prefetch_stop_policy
        self.prefetch_timeout_base = prefetch_timeout_base
        self.prefetch_timeout_per_page = prefetch_timeout_per_page
        self.prefetch_timeout_max = prefetch_timeout_max
        self.is_decode = is_decode

        self.backup_inflight_high = backup_inflight_high
        self.backup_inflight_hard = backup_inflight_hard
        self.backup_pacing_interval = backup_pacing_interval
        self.l3_status_monitor = (
            L3StatusMonitor(
                l3_failure_threshold,
                l3_recovery_interval_sec,
            )
            if storage_backend_type == "mooncake"
            else None
        )

        self.prefetch_threshold = prefetch_threshold
        self.prefetch_capacity_limit = int(0.8 * (mem_pool_host.size - mem_pool_device.size))
        self.prefetch_capacity_limit = max(self.prefetch_capacity_limit, DEFAULT_PREFETCH_SIZE)
        self.prefetch_tokens_occupied = 0
        self.prefetch_ready_tokens = 0

        if tp_world_size > 1:
            logger.info("[L3Controller] About to create prefetch_tp_group")
            from sglang.srt.distributed import get_l3_prefetch_tp_group, get_l3_query_tp_group

            self.prefetch_tp_group = get_l3_prefetch_tp_group()
            self.query_tp_group = get_l3_query_tp_group()
            logger.info("[L3Controller] prefetch_tp_group created")

        if self.pp_world_size > 1:
            logger.info("[L3Controller] About to create prefetch_pp_group")
            from sglang.srt.distributed import get_l3_prefetch_pp_group, get_l3_query_pp_group

            self.prefetch_pp_group = get_l3_prefetch_pp_group()
            self.query_pp_group = get_l3_query_pp_group()
            logger.info(f"[L3Controller] prefetch_pp_group created {pp_world_size=}")

        from sglang.srt.mem_cache.l3.backend_factory import StorageBackendFactory

        try:
            self.storage_backend = StorageBackendFactory.create_backend(storage_backend_type, storage_config, mem_pool_host)
        except ValueError as e:
            raise ValueError(f"Failed to create storage backend: {e}") from e

        self.storage_backend.register_mem_pool_host(mem_pool_host)

        self.page_get_func = self.generic_page_get
        self.page_set_func = self.generic_page_set

        if (storage_backend_type in ["hf3fs", "mooncake", "eic", "lmcache_local_disk", "lmcache_rust_raw_block"]) or (
            storage_backend_type == "dynamic" and bool((storage_config.extra_config or {}).get("interface_v1", 0))
        ):
            self.page_get_func = self.page_get_zero_copy
            self.page_set_func = self.page_set_zero_copy

        self.sync_tp_fn = self.sync_tp
        self.page_get_zero_copy_fn = self.page_get_zero_copy_default
        self.storage_hit_query_local_fn = self.storage_hit_query_local
        self.sync_storage_hit_query_fn = self.sync_storage_hit_query

        self.h2s_write_q_todo: Queue = Queue()
        self.h2s_write_q_done: Queue = Queue()
        self.h2s_write_inflight: int = 0
        self.s2h_load_q_todo: Queue = Queue()
        self.s2h_load_q_query: Queue = Queue()
        self.s2h_load_q_query_result: Queue = Queue()
        self._query_sequence_lock = threading.Lock()
        self._next_query_sequence = 0

    def join_threads(self):
        for worker in self.query_backend_threads:
            if worker.is_alive():
                worker.join()
        if self.query_thread.is_alive():
            self.query_thread.join()
        if self.prefetch_thread.is_alive():
            self.prefetch_thread.join()
        if self.backup_thread.is_alive():
            self.backup_thread.join()
        self.h2s_write_q_todo.queue.clear()
        self.h2s_write_q_done.queue.clear()
        self.h2s_write_inflight = 0
        self.s2h_load_q_todo.queue.clear()
        self.s2h_load_q_query.queue.clear()
        self.s2h_load_q_query_result.queue.clear()
        with self._query_sequence_lock:
            self._next_query_sequence = 0

    def restart_threads(self):
        self.query_backend_threads = [
            threading.Thread(
                target=self.query_backend_thread_func,
                name=f"l3-query-backend-{worker_id}",
                daemon=True,
            )
            for worker_id in range(self.query_backend_worker_count)
        ]
        self.query_thread = threading.Thread(
            target=self.query_thread_func,
            name="l3-query-coordinator",
            daemon=True,
        )
        self.prefetch_thread = threading.Thread(target=self.prefetch_thread_func, daemon=True)
        self.backup_thread = threading.Thread(target=self.backup_thread_func, daemon=True)
        if not self.is_decode:
            for worker in self.query_backend_threads:
                worker.start()
            self.query_thread.start()
            self.prefetch_thread.start()
        self.backup_thread.start()

    def reset(self):
        self.join_threads()
        self.restart_threads()

    def prefetch_rate_limited(self) -> bool:
        return self.prefetch_tokens_occupied + self.prefetch_ready_tokens >= self.prefetch_capacity_limit

    def mark_prefetch_ready(self, req, num_tokens: int):
        if num_tokens <= 0:
            return
        req.l3_prefetch_ready_tokens = num_tokens
        self.prefetch_ready_tokens += num_tokens

    def release_prefetch_ready(self, req):
        num_tokens = req.l3_prefetch_ready_tokens
        if num_tokens <= 0:
            return
        req.l3_prefetch_ready_tokens = 0
        if num_tokens > self.prefetch_ready_tokens:
            logger.warning(
                "[L3-Prefetch-Ready-Counter-Underflow] ready=%d release=%d; resetting to 0",
                self.prefetch_ready_tokens,
                num_tokens,
            )
            self.prefetch_ready_tokens = 0
            return
        self.prefetch_ready_tokens -= num_tokens

    def reserve_prefetch_tokens(self, num_tokens: int):
        self.prefetch_tokens_occupied += num_tokens

    def release_prefetch_tokens(self, num_tokens: int):
        if num_tokens > self.prefetch_tokens_occupied:
            logger.warning(
                "[L3-Prefetch-Counter-Underflow] occupied=%d release=%d; resetting to 0",
                self.prefetch_tokens_occupied,
                num_tokens,
            )
            self.prefetch_tokens_occupied = 0
            return
        self.prefetch_tokens_occupied -= num_tokens

    def enqueue_query(self, s2h_load: S2HLoad) -> int:
        with self._query_sequence_lock:
            sequence_id = self._next_query_sequence
            self._next_query_sequence += 1
        self.s2h_load_q_query.put(L3QueryTask(sequence_id, s2h_load))
        return sequence_id

    def query_backend_thread_func(self):
        logger.info("[Prefetch-Query-Backend-Thread] started")
        try:
            while not self.stop_event.is_set() or not self.s2h_load_q_query.empty():
                try:
                    task: L3QueryTask = self.s2h_load_q_query.get(block=True, timeout=0.1)
                except Empty:
                    continue

                s2h_load = task.s2h_load
                s2h_load.t_query_start = time.monotonic()
                try:
                    result = self.storage_hit_query_local_fn(
                        s2h_load.token_ids,
                        s2h_load.last_hash,
                        s2h_load.prefix_keys,
                    )
                except Exception as error:
                    completion = L3QueryCompletion(task=task, error=error)
                else:
                    s2h_load.t_query_storage_done = time.monotonic()
                    completion = L3QueryCompletion(task=task, result=result)
                self.s2h_load_q_query_result.put(completion)
        finally:
            self.s2h_load_q_query_result.put(None)

    def storage_hit_query_local(self, tokens, last_hash, prefix_keys):
        hash_values = []
        h = last_hash
        for i in range(0, len(tokens), self.page_size):
            h = self.get_hash_str(tokens[i : i + self.page_size], h)
            hash_values.append(h)
        extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)
        monitor = self.l3_status_monitor
        if monitor is None:
            hit_pages = self.storage_backend.batch_exists(hash_values, extra_info)
        else:
            token = monitor.begin_operation()
            if token is None:
                hit_pages = 0
                logger.warning("[L3-Status-Skip] Mooncake is unavailable")
            else:
                try:
                    hit_pages = self.storage_backend.batch_exists(hash_values, extra_info)
                    if not isinstance(hit_pages, int) or not (0 <= hit_pages <= len(hash_values)):
                        raise RuntimeError(f"invalid batch_exists result: {hit_pages!r}")
                except Exception:
                    unavailable = monitor.report_failure(token)
                    logger.exception(
                        "[L3-Storage-Query-Failed] falling back to cache miss%s",
                        "; Mooncake marked unavailable" if unavailable else "",
                    )
                    hit_pages = 0
                else:
                    if monitor.report_success(token):
                        logger.info("[L3-Status-Recovered] Mooncake query check succeeded")
        return L3QueryResult(hash_values, hit_pages)

    def assert_query_coordinator_thread(self):
        if threading.current_thread() is not getattr(self, "query_thread", None):
            raise RuntimeError("L3 query collectives must run on the query coordinator thread")

    def sync_storage_hit_query(self, result: L3QueryResult) -> L3QueryResult:
        self.assert_query_coordinator_thread()
        hit_pages_tensor = torch.tensor([result.hit_pages], dtype=torch.int)
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(
                hit_pages_tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=self.query_tp_group,
            )
        if self.pp_world_size > 1:
            torch.distributed.all_reduce(
                hit_pages_tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=self.query_pp_group,
            )
        hit_pages = int(hit_pages_tensor.item())
        page2rank = result.page2rank
        if page2rank is None:
            page2rank = [0 for _ in range(hit_pages)]
        return L3QueryResult(
            result.hash_values[:hit_pages],
            hit_pages,
            page2rank[:hit_pages],
        )

    def create_s2h_load(self, rid, tokens, last_hash, prefix_keys, last_host_node):
        s2h_load = S2HLoad(rid, None, tokens, prefix_keys, last_host_node, None, last_hash)
        return s2h_load

    def sync_completed_tokens(self, s2h_load):
        completed_tokens = self.sync_tp_fn(s2h_load)
        if self.pp_world_size > 1:
            completed_tensor = torch.tensor([completed_tokens], dtype=torch.int64)
            logger.debug("All Reduce Between pp group Start!")
            torch.distributed.all_reduce(
                completed_tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=self.prefetch_pp_group,
            )
            completed_tokens = completed_tensor.item()
            logger.debug(f"All Reduce Between pp group End! {completed_tokens=}")

        return completed_tokens

    def sync_tp(self, s2h_load: S2HLoad):
        completed_tokens = s2h_load.completed_tokens
        if self.tp_world_size > 1:
            logger.debug("All Reduce Between tp group Start!")
            completed_tensor = torch.tensor([completed_tokens], dtype=torch.int64)
            torch.distributed.all_reduce(
                completed_tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=self.prefetch_tp_group,
            )
            completed_tokens = completed_tensor.item()
            logger.debug(f"All Reduce Between tp group End! {completed_tokens=}")
        return completed_tokens

    def query_thread_func(self):
        if self.is_decode:
            return  # Decode worker: never prefetch
        logger.info("[Prefetch-Query-Coordinator-Thread] started")
        next_sequence_id = 0
        finished_workers = 0
        pending: dict[int, L3QueryCompletion] = {}
        while finished_workers < self.query_backend_worker_count:
            try:
                completion: L3QueryCompletion = self.s2h_load_q_query_result.get(block=True, timeout=0.1)
            except Empty:
                continue
            if completion is None:
                finished_workers += 1
                continue
            pending[completion.task.sequence_id] = completion

            while next_sequence_id in pending:
                completion = pending.pop(next_sequence_id)
                if completion.error is not None:
                    raise completion.error
                if completion.result is None:
                    raise RuntimeError("L3 query completed without a result")
                result = self.sync_storage_hit_query_fn(completion.result)
                s2h_load = completion.task.s2h_load
                s2h_load.t_query_done = time.monotonic()
                min_hit_pages = max(1, self.prefetch_threshold // self.page_size)
                s2h_load.set_query_result(
                    result.hash_values,
                    [-1] * result.hit_pages,
                    result.page2rank,
                    result.hit_pages,
                    should_terminate=result.hit_pages < min_hit_pages,
                )
                logger.debug(f"[S2H-Query] rid={s2h_load.request_id} query_pages={result.hit_pages}")
                next_sequence_id += 1

        if pending:
            raise RuntimeError("L3 query workers stopped with incomplete ordered results")

    def prefetch_thread_func(self):
        if self.is_decode:
            return  # Decode worker: never prefetch
        logger.info("[Prefetch-Thread] started")
        while not self.stop_event.is_set():
            try:
                s2h_load: S2HLoad = self.s2h_load_q_todo.get(block=True, timeout=0.1)
            except Empty:
                continue

            if s2h_load.is_detached():
                s2h_load.t_read_start = time.monotonic()
                s2h_load.t_io_done = s2h_load.t_read_start
                s2h_load.set_io_done()
                continue

            # Memory-safety contract: page_transfer must not return (including
            # failure paths) until every submitted Mooncake RDMA task is
            # terminal. TODO(mooncake-npu): enforce this contract in main.
            if self.l3_status_monitor is None:
                s2h_load.t_read_start = time.monotonic()
                logger.debug(f"[S2H-Start] rid={s2h_load.request_id} expected_pages={len(s2h_load.hash_value)}")
                self.page_transfer(s2h_load)
                s2h_load.t_io_done = time.monotonic()
                s2h_load.set_io_done()
            else:
                try:
                    s2h_load.t_read_start = time.monotonic()
                    logger.debug(f"[S2H-Start] rid={s2h_load.request_id} expected_pages={len(s2h_load.hash_value)}")
                    self.page_transfer(s2h_load)
                except Exception:
                    s2h_load.mark_terminate()
                    logger.exception(
                        "[S2H-Prefetch-Failed] rid=%s; continuing without L3",
                        s2h_load.request_id,
                    )
                finally:
                    s2h_load.t_io_done = time.monotonic()
                    s2h_load.set_io_done()

    def page_get_zero_copy(self, s2h_load: S2HLoad, hash_values, host_token_indices, extra_info=None):
        return self.page_get_zero_copy_fn(s2h_load, hash_values, host_token_indices, extra_info)

    def page_get_zero_copy_default(self, s2h_load, hash_values, host_token_indices, extra_info=None):
        results = self.storage_backend.batch_get_zero_copy(hash_values, host_token_indices, extra_info)
        inc = 0
        for i in range(len(hash_values)):
            if not results[i]:
                s2h_load.mark_terminate()
                logger.warning(f"Prefetch s2h_load {s2h_load.request_id} failed to retrieve page {hash_values[i]}.")
                break
            inc += self.page_size
        s2h_load.increment(inc)
        return True

    def generic_page_get(self, s2h_load, hash_values, host_token_indices, extra_info=None):
        dummy_page_dst = [self.mem_pool_host.get_dummy_flat_data_page() for _ in hash_values]
        page_data = self.storage_backend.batch_get(hash_values, dummy_page_dst)
        if page_data is None:
            return True
        for i in range(len(hash_values)):
            if page_data[i] is None:
                logger.warning(f"Prefetch s2h_load {s2h_load.request_id} failed to retrieve page {hash_values[i]}.")
                break
            self.mem_pool_host.set_from_flat_data_page(
                host_token_indices[i * self.page_size],
                page_data[i],
            )
            if not s2h_load.increment(self.page_size):
                break
        return True

    def page_transfer(self, s2h_load: S2HLoad):
        prefix_keys = s2h_load.prefix_keys
        total_pages = len(s2h_load.hash_value)
        if total_pages == 0:
            # dummy s2h_load from enqueue_dummy_s2h_load, nothing to prefetch
            return
        total_bytes = total_pages * self.page_size * self.mem_pool_host.get_size_per_token()
        logger.info(
            f"[L3-Prefetch-Read] rid={s2h_load.request_id} "
            f"pages={total_pages} tokens={total_pages * self.page_size} "
            f"bytes={total_bytes / 1024 / 1024:.1f}MB"
        )
        deadline = None
        if self.prefetch_stop_policy == "timeout":
            deadline = s2h_load.prefetch_deadline
            if deadline is None:
                timeout = self.prefetch_timeout_base + len(s2h_load.hash_value) * self.prefetch_timeout_per_page
                if TEST_PREFETCH_RANDOM_TIMEOUT:
                    timeout = random.uniform(0, 0.2)  # noqa: S311
                else:
                    timeout = min(self.prefetch_timeout_max, timeout)
                deadline = time.monotonic() + timeout
            logger.debug(f"prefetch_deadline={deadline}")

        for i in range(0, len(s2h_load.hash_value), self.storage_batch_size):
            if deadline is not None and time.monotonic() > deadline:
                s2h_load.mark_terminate()
                logger.warning(
                    f"[Prefetch-Timeout] rid={s2h_load.request_id} loaded_pages={i // self.storage_batch_size} total_pages={len(s2h_load.hash_value)}"
                )
                break

            batch_hashes = s2h_load.hash_value[i : i + self.storage_batch_size]
            batch_host_token_indices = s2h_load.host_token_indices[i * self.page_size : (i + len(batch_hashes)) * self.page_size]
            prev_completed_tokens = s2h_load.completed_tokens
            extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)
            monitor = self.l3_status_monitor
            token = None
            performed_io = True
            if monitor is None:
                self.page_get_func(
                    s2h_load,
                    batch_hashes,
                    batch_host_token_indices,
                    extra_info,
                )
            else:
                token = monitor.begin_operation()
                if token is None:
                    logger.warning(
                        "[L3-Status-Skip] rid=%s Mooncake is unavailable",
                        s2h_load.request_id,
                    )
                    s2h_load.mark_terminate()
                    break
                try:
                    performed_io = self.page_get_func(
                        s2h_load,
                        batch_hashes,
                        batch_host_token_indices,
                        extra_info,
                    )
                except Exception:
                    unavailable = monitor.report_failure(token)
                    logger.exception(
                        "[L3-Prefetch-Batch-Failed] rid=%s%s",
                        s2h_load.request_id,
                        "; Mooncake marked unavailable" if unavailable else "",
                    )
                    s2h_load.mark_terminate()
                    break
            s2h_load.processed_global_pages += len(batch_hashes)
            if not self.enable_l15:
                if s2h_load.completed_tokens != prev_completed_tokens + len(batch_hashes) * self.page_size:
                    if s2h_load.is_detached():
                        break
                    unavailable = monitor.report_failure(token) if monitor is not None else False
                    if unavailable:
                        logger.warning("[L3-Status-Unavailable] incomplete Mooncake prefetch batch")
                    s2h_load.mark_terminate()
                    break
            if monitor is not None:
                if performed_io is False:
                    monitor.cancel_check(token)
                elif monitor.report_success(token):
                    logger.info("[L3-Status-Recovered] Mooncake prefetch check succeeded")

            if prefix_keys and len(prefix_keys) > 0:
                prefix_keys += batch_hashes

        completed_pages = s2h_load.completed_tokens // self.page_size
        completed_bytes = s2h_load.completed_tokens * self.mem_pool_host.get_size_per_token()
        logger.info(
            f"[L3-Prefetch-Read-Done] rid={s2h_load.request_id} "
            f"expected={total_pages}p/{total_pages * self.page_size}t "
            f"actual={completed_pages}p/{s2h_load.completed_tokens}t "
            f"bytes={completed_bytes / 1024 / 1024:.1f}MB"
        )

    def is_backup_full(self) -> bool:
        return self.backup_inflight_hard > 0 and self.h2s_write_inflight >= self.backup_inflight_hard

    def complete_h2s_write(self):
        if self.h2s_write_inflight <= 0:
            logger.warning(
                "[L3-Backup-Counter-Underflow] inflight=%d; resetting to 0",
                self.h2s_write_inflight,
            )
            self.h2s_write_inflight = 0
            return
        self.h2s_write_inflight -= 1

    def enqueue_h2s_write(
        self,
        node,
        token_indices,
        page_hashes,
        prefix_keys,
        pp_prefix=None,
        layer_range=None,
    ) -> H2SWrite:
        h2s_write = H2SWrite(token_indices, page_hashes, prefix_keys, node, pp_prefix, layer_range)
        self.h2s_write_inflight += 1
        self.h2s_write_q_todo.put(h2s_write)
        return h2s_write

    def add_h2s_write(
        self,
        node,
        prefix_keys: list[str] | None = None,
        pp_prefix: str | None = None,
        layer_range: tuple | None = None,
    ) -> H2SWrite:
        host_token_indices = self.mem_pool_host.pages_to_tokens(node.host_value)
        num_pages = len(node.host_value)
        total_bytes = num_pages * self.page_size * self.mem_pool_host.get_size_per_token()
        logger.info(f"[L3-Backup-Enqueue] pages={num_pages} tokens={num_pages * self.page_size} bytes={total_bytes / 1024 / 1024:.1f}MB")
        logger.debug(f"[add_h2s_write] num_pages={num_pages} num_tokens={len(host_token_indices)}")
        page_hashes = node.hash_value if node.hash_value is not None else []
        return self.enqueue_h2s_write(node, host_token_indices, page_hashes, prefix_keys, pp_prefix, layer_range)

    def generic_page_set(self, hash_values, host_token_indices, extra_info=None, on_complete_callback=None) -> bool:
        data = [self.mem_pool_host.get_data_page(host_token_indices[i * self.page_size]) for i in range(len(hash_values))]
        return self.storage_backend.batch_set(hash_values, data)

    def page_set_zero_copy(self, hash_values, host_token_indices, extra_info=None, on_complete_callback=None) -> bool:
        results = self.storage_backend.batch_set_zero_copy(
            hash_values,
            host_token_indices,
            extra_info,
            on_complete_callback=on_complete_callback,
        )
        if self.l3_status_monitor is not None and len(results) != len(hash_values):
            raise RuntimeError(f"batch_set_zero_copy returned an unexpected result count: expected={len(hash_values)}, actual={len(results)}")
        success_count = sum(1 for result in results if result)
        logger.debug(
            "[L3-Backup-Batch-Result] pages=%s success=%s failed=%s sample_hashes=%s",
            len(hash_values),
            success_count,
            len(results) - success_count,
            [hash_value[:16] for hash_value in hash_values[: min(5, len(hash_values))]],
        )
        return all(results)

    def page_backup(self, h2s_write: H2SWrite):
        if self.l3_status_monitor is None:
            return self._page_backup_unmonitored(h2s_write)
        return self._page_backup_mooncake(h2s_write)

    def _page_backup_unmonitored(self, h2s_write: H2SWrite):
        """Preserve the existing backup behavior for non-Mooncake backends."""
        prefix_keys = h2s_write.prefix_keys
        total_pages = len(h2s_write.hash_value)
        if total_pages > 0:
            logger.debug(
                "[L3-Backup-Hash] first=%s... last=%s... pages=%s sample_hashes=%s",
                h2s_write.hash_value[0][:16],
                h2s_write.hash_value[-1][:16],
                total_pages,
                [hash_value[:16] for hash_value in h2s_write.hash_value[: min(5, total_pages)]],
            )

        if total_pages == 0:
            self.h2s_write_q_done.put(h2s_write)
            return

        total_bytes = total_pages * self.page_size * self.mem_pool_host.get_size_per_token()
        logger.debug(
            "[L3-Backup-Write] pages=%s tokens=%s bytes=%.1fMB prefix_count=%s prefix_sample=%s pp_rank=%s",
            total_pages,
            total_pages * self.page_size,
            total_bytes / 1024 / 1024,
            0 if prefix_keys is None else len(prefix_keys),
            [] if not prefix_keys else prefix_keys[: min(5, len(prefix_keys))],
            getattr(self.storage_config, "pp_rank", None),
        )

        def on_complete_callback(_key):
            return h2s_write.page_write_done()

        num_batches = (total_pages + self.storage_batch_size - 1) // self.storage_batch_size
        if self.storage_backend_type != "lmcache_local_disk":
            h2s_write.setup_async_ack(self.h2s_write_q_done, num_batches)
        else:
            h2s_write.setup_async_ack(self.h2s_write_q_done, total_pages)

        for i in range(0, total_pages, self.storage_batch_size):
            batch_hashes = h2s_write.hash_value[i : i + self.storage_batch_size]
            batch_host_token_indices = h2s_write.host_token_indices[i * self.page_size : (i + len(batch_hashes)) * self.page_size]
            current_prefix_keys = None if prefix_keys is None else list(prefix_keys)
            logger.debug(
                "[L3-Backup-Batch] batch=%s/%s pages=%s prefix_count=%s prefix_tail=%s hash_sample=%s host_token_range=[%s,%s]",
                i // self.storage_batch_size + 1,
                num_batches,
                len(batch_hashes),
                0 if current_prefix_keys is None else len(current_prefix_keys),
                [] if not current_prefix_keys else current_prefix_keys[-min(5, len(current_prefix_keys)) :],
                [hash_value[:16] for hash_value in batch_hashes[: min(5, len(batch_hashes))]],
                batch_host_token_indices[0].item() if len(batch_host_token_indices) > 0 else None,
                batch_host_token_indices[-1].item() if len(batch_host_token_indices) > 0 else None,
            )
            extra_info = HiCacheStorageExtraInfo(
                prefix_keys=current_prefix_keys,
                pp_prefix=h2s_write.pp_prefix,
                layer_range=h2s_write.layer_range,
            )
            self.page_set_func(batch_hashes, batch_host_token_indices, extra_info, on_complete_callback=on_complete_callback)

            if prefix_keys and len(prefix_keys) > 0:
                prefix_keys += batch_hashes
            h2s_write.completed_tokens += self.page_size * len(batch_hashes)

        completed_bytes = h2s_write.completed_tokens * self.mem_pool_host.get_size_per_token()
        logger.info(
            "[L3-Backup-Write-Done] submitted=%sp/%st bytes_total=%.1fMB async_batches=%s async_ack_pending=1",
            total_pages,
            total_pages * self.page_size,
            completed_bytes / 1024 / 1024,
            num_batches,
        )

    def _page_backup_mooncake(self, h2s_write: H2SWrite):
        prefix_keys = h2s_write.prefix_keys
        total_pages = len(h2s_write.hash_value)
        if total_pages > 0:
            logger.debug(
                "[L3-Backup-Hash] first=%s... last=%s... pages=%s sample_hashes=%s",
                h2s_write.hash_value[0][:16],
                h2s_write.hash_value[-1][:16],
                total_pages,
                [hash_value[:16] for hash_value in h2s_write.hash_value[: min(5, total_pages)]],
            )

        if total_pages == 0:
            self.h2s_write_q_done.put(h2s_write)
            return

        total_bytes = total_pages * self.page_size * self.mem_pool_host.get_size_per_token()
        logger.debug(
            "[L3-Backup-Write] pages=%s tokens=%s bytes=%.1fMB prefix_count=%s prefix_sample=%s pp_rank=%s",
            total_pages,
            total_pages * self.page_size,
            total_bytes / 1024 / 1024,
            0 if prefix_keys is None else len(prefix_keys),
            [] if not prefix_keys else prefix_keys[: min(5, len(prefix_keys))],
            getattr(self.storage_config, "pp_rank", None),
        )

        num_batches = (total_pages + self.storage_batch_size - 1) // self.storage_batch_size

        for i in range(0, total_pages, self.storage_batch_size):
            batch_hashes = h2s_write.hash_value[i : i + self.storage_batch_size]
            batch_host_token_indices = h2s_write.host_token_indices[i * self.page_size : (i + len(batch_hashes)) * self.page_size]
            current_prefix_keys = None if prefix_keys is None else list(prefix_keys)
            logger.debug(
                "[L3-Backup-Batch] batch=%s/%s pages=%s prefix_count=%s prefix_tail=%s hash_sample=%s host_token_range=[%s,%s]",
                i // self.storage_batch_size + 1,
                num_batches,
                len(batch_hashes),
                0 if current_prefix_keys is None else len(current_prefix_keys),
                [] if not current_prefix_keys else current_prefix_keys[-min(5, len(current_prefix_keys)) :],
                [hash_value[:16] for hash_value in batch_hashes[: min(5, len(batch_hashes))]],
                batch_host_token_indices[0].item() if len(batch_host_token_indices) > 0 else None,
                batch_host_token_indices[-1].item() if len(batch_host_token_indices) > 0 else None,
            )
            extra_info = HiCacheStorageExtraInfo(
                prefix_keys=current_prefix_keys,
                pp_prefix=h2s_write.pp_prefix,
                layer_range=h2s_write.layer_range,
            )
            monitor = self.l3_status_monitor
            token = monitor.begin_operation()
            if token is None:
                logger.warning("[L3-Status-Skip] Mooncake backup is unavailable")
                break

            try:
                success = self.page_set_func(
                    batch_hashes,
                    batch_host_token_indices,
                    extra_info,
                )
            except Exception:
                unavailable = monitor.report_failure(token)
                logger.exception(
                    "[L3-Backup-Batch-Failed]%s",
                    "; Mooncake marked unavailable" if unavailable else "",
                )
                break

            if not success:
                unavailable = monitor.report_failure(token)
                logger.warning(
                    "[L3-Backup-Batch-Failed] backend returned failure%s",
                    "; Mooncake marked unavailable" if unavailable else "",
                )
                break

            if monitor.report_success(token):
                logger.info("[L3-Status-Recovered] Mooncake backup check succeeded")

            # The complete ancestor prefix only needs one lease refresh per
            # backup task. Re-sending an ever-growing prefix on every storage
            # batch makes BatchExist work quadratic in the number of pages.
            # Each later batch still checks (and renews) its own keys in
            # batch_set_zero_copy before writing any missing objects.
            prefix_keys = None
            h2s_write.completed_tokens += self.page_size * len(batch_hashes)

        completed_bytes = h2s_write.completed_tokens * self.mem_pool_host.get_size_per_token()
        logger.info(
            "[L3-Backup-Write-Done] successful=%sp/%st requested=%sp/%st bytes_total=%.1fMB async_batches=%s async_ack_pending=1",
            h2s_write.completed_tokens // self.page_size,
            h2s_write.completed_tokens,
            total_pages,
            total_pages * self.page_size,
            completed_bytes / 1024 / 1024,
            num_batches,
        )
        self.h2s_write_q_done.put(h2s_write)

    def backup_thread_func(self):
        while not self.stop_event.is_set():
            try:
                h2s_write = self.h2s_write_q_todo.get(block=True, timeout=1)
                if h2s_write is None:
                    continue

                if self.backup_skip:
                    self.h2s_write_q_done.put(h2s_write)
                    continue

                if self.backup_inflight_high > 0:
                    _paced = 0
                    while self.s2h_load_q_todo.qsize() > self.backup_inflight_high and _paced < 10 and not self.stop_event.is_set():
                        time.sleep(self.backup_pacing_interval)
                        _paced += 1
                    if _paced > 0:
                        logger.info(
                            f"[L3-Backup-Pace] prefetch_q={self.s2h_load_q_todo.qsize()} >= high={self.backup_inflight_high}, paced {_paced} ticks"
                        )

                if self.l3_status_monitor is None:
                    self.page_backup(h2s_write)
                else:
                    try:
                        self.page_backup(h2s_write)
                    except Exception:
                        self.h2s_write_q_done.put(h2s_write)
                        logger.exception("[L3-Backup-Failed] worker survived unexpected error")

            except Empty:
                continue
