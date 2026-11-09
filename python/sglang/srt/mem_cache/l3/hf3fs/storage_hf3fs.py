from __future__ import annotations

import concurrent.futures
import ctypes
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from sglang.srt.mem_cache.l3.storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)

logger = logging.getLogger(__name__)

_IOV_CHUNK = 256


def _ptr_views(ptr_list: List[int], size_list: List[int]) -> List[memoryview]:
    """Wrap raw (ptr, size) pairs as zero-copy memoryviews.

    The memoryview keeps the ctypes buffer (and thus the mapping) alive; the
    caller just holds the returned list for the duration of the IO.
    """
    return [
        memoryview((ctypes.c_char * size).from_address(ptr))
        for ptr, size in zip(ptr_list, size_list)
    ]


def _consume(iovs: List[memoryview], n: int) -> List[memoryview]:
    """Drop the first n bytes from an iovec list (for partial readv/writev)."""
    out = []
    skipped = 0
    for v in iovs:
        if skipped < n:
            if skipped + len(v) <= n:
                skipped += len(v)
                continue
            v = v[n - skipped :]
            skipped = n
        out.append(v)
    return out


def _writev_all(fd: int, iovs: List[memoryview]) -> int:
    total = sum(len(v) for v in iovs)
    written = 0
    remaining = iovs
    while remaining:
        n = os.writev(fd, remaining[:_IOV_CHUNK])
        if n <= 0:
            break
        written += n
        remaining = _consume(remaining, n)
    return written


def _readv_all(fd: int, iovs: List[memoryview]) -> int:
    read = 0
    remaining = iovs
    while remaining:
        try:
            n = os.readv(fd, remaining[:_IOV_CHUNK])
        except OSError:
            break
        if n <= 0:  # EOF
            break
        read += n
        remaining = _consume(remaining, n)
    return read


@dataclass
class _TargetCtx:
    """Per pp-target context: directory + fixed page byte size."""

    pp_tag: str
    layer_range: Optional[Tuple[int, int]]
    dir_path: str
    bytes_per_page: int


class HiCacheHF3FS(HiCacheStorage):
    """Per-key-file HiCache storage backend (see module docstring)."""

    def __init__(self, storage_config: HiCacheStorageConfig = None):
        extra = (storage_config.extra_config or {}) if storage_config else {}

        self.file_path_prefix = (
            extra.get("file_path_prefix")
            or os.getenv("SGLANG_HICACHE_HF3FS_DIR")
            or "/data/hicache_hf3fs"
        )
        self.numjobs = int(extra.get("numjobs", 16))
        self.fsync = bool(extra.get("fsync", False))

        if storage_config is not None:
            self.tp_rank = storage_config.tp_rank
            self.pp_rank = getattr(storage_config, "pp_rank", 0)
            self.pp_size = getattr(storage_config, "pp_size", 1)
            self.is_mla_model = storage_config.is_mla_model
            model_name = storage_config.model_name or "unknown_model"
        else:
            self.tp_rank = 0
            self.pp_rank = 0
            self.pp_size = 1
            self.is_mla_model = False
            model_name = "unknown_model"
        self.model_name = "-".join(model_name.split("/"))

        # Same convention as MooncakeStore: prefill rank i's own namespace is
        # pp{i}of{N}_; decode (pp_size == 1) always overrides per-call via
        # extra_info.pp_prefix when it writes all N targets.
        self.pp_prefix = (
            f"pp{self.pp_rank}of{self.pp_size}_" if self.pp_size > 1 else ""
        )

        # kvp_rank comes from the host pool at register time (L1.5 sharding
        # rank); fall back to tp_rank which equals it for attn-tp pools.
        self.kvp_rank = self.tp_rank

        self.enable_storage_metrics = bool(
            getattr(storage_config, "enable_storage_metrics", False)
        )
        self.prefetch_pgs: List[int] = []
        self.backup_pgs: List[int] = []
        self.prefetch_bandwidth: List[float] = []
        self.backup_bandwidth: List[float] = []

        self._ctxs: Dict[Tuple[str, Optional[Tuple[int, int]]], _TargetCtx] = {}
        self._ctx_lock = threading.Lock()
        self._tmp_counter = 0

        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.numjobs,
            thread_name_prefix=f"HiCacheHF3FS-r{self.tp_rank}",
        )

        logger.info(
            f"[Rank {self.tp_rank}] HiCacheHF3FS initialized: "
            f"prefix={self.file_path_prefix} model={self.model_name} "
            f"pp_prefix='{self.pp_prefix}' numjobs={self.numjobs} fsync={self.fsync}. "
            f"NOTE: no capacity management; files accumulate until cleaned externally."
        )

    # ------------------------------------------------------------------
    # path / ctx helpers
    # ------------------------------------------------------------------

    def register_mem_pool_host(self, mem_pool_host):
        super().register_mem_pool_host(mem_pool_host)
        assert hasattr(mem_pool_host, "get_page_buffer_meta"), (
            "HiCacheHF3FS requires a host pool with get_page_buffer_meta "
            "(MLA/DSA host pools); MHA pools are not supported yet."
        )
        self.kvp_rank = getattr(mem_pool_host, "kvp_rank", self.tp_rank)
        self.page_size = mem_pool_host.page_size
        logger.info(
            f"[Rank {self.tp_rank}] HiCacheHF3FS registered host pool: "
            f"kvp_rank={self.kvp_rank} page_size={self.page_size} "
            f"layout={getattr(mem_pool_host, 'layout', '?')}"
        )

    def _resolve_pp_prefix(self, extra_info: Optional[HiCacheStorageExtraInfo]) -> str:
        if extra_info is not None and extra_info.pp_prefix:
            return extra_info.pp_prefix
        return self.pp_prefix

    def _get_ctx(
        self, pp_prefix: str, layer_range: Optional[Tuple[int, int]]
    ) -> _TargetCtx:
        pp_tag = pp_prefix.rstrip("_") or "ppall"
        key = (pp_tag, layer_range)
        ctx = self._ctxs.get(key)
        if ctx is None:
            with self._ctx_lock:
                ctx = self._ctxs.get(key)
                if ctx is None:
                    # Probe the fixed page size for this (target, layer_range)
                    # by asking the pool for one page's segment sizes. Page 0's
                    # pointers are only read, never dereferenced for content.
                    probe = torch.arange(self.page_size)
                    _, size_list = self.mem_pool_host.get_page_buffer_meta(
                        probe, layer_range
                    )
                    bytes_per_page = sum(size_list)
                    dir_path = os.path.join(
                        self.file_path_prefix,
                        self.model_name,
                        f"rank{self.kvp_rank}",
                        pp_tag,
                    )
                    os.makedirs(dir_path, exist_ok=True)
                    ctx = _TargetCtx(
                        pp_tag=pp_tag,
                        layer_range=layer_range,
                        dir_path=dir_path,
                        bytes_per_page=bytes_per_page,
                    )
                    self._ctxs[key] = ctx
                    logger.info(
                        f"[Rank {self.tp_rank}] HiCacheHF3FS target '{pp_tag}' "
                        f"layer_range={layer_range}: bytes_per_page="
                        f"{bytes_per_page} ({bytes_per_page / (1 << 20):.2f} MB) "
                        f"dir={dir_path}"
                    )
        return ctx

    def _key_to_path(self, ctx: _TargetCtx, key: str) -> str:
        return os.path.join(ctx.dir_path, key[:2], f"{key}.bin")

    # ------------------------------------------------------------------
    # single-page IO
    # ------------------------------------------------------------------

    def _write_one(
        self,
        ctx: _TargetCtx,
        key: str,
        host_token_indices: torch.Tensor,
        layer_range: Optional[Tuple[int, int]],
    ) -> bool:
        path = self._key_to_path(ctx, key)
        if os.path.exists(path):
            # Same KV content under the same hash; skip the rewrite.
            return True

        ptr_list, size_list = self.mem_pool_host.get_page_buffer_meta(
            host_token_indices, layer_range
        )
        views = _ptr_views(ptr_list, size_list)
        assert sum(size_list) == ctx.bytes_per_page, (
            f"page segment bytes {sum(size_list)} != ctx bytes_per_page "
            f"{ctx.bytes_per_page} for target {ctx.pp_tag} {layer_range}"
        )

        bucket = os.path.dirname(path)
        os.makedirs(bucket, exist_ok=True)
        with self._ctx_lock:
            self._tmp_counter += 1
            tmp_path = (
                f"{path}.tmp.{os.getpid()}.{threading.get_ident()}.{self._tmp_counter}"
            )

        fd = None
        try:
            fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            if _writev_all(fd, views) != ctx.bytes_per_page:
                logger.error(f"[Rank {self.tp_rank}] short write for key {key}")
                return False
            if self.fsync:
                os.fsync(fd)
            os.close(fd)
            fd = None
            os.rename(tmp_path, path)
            return True
        except OSError as e:
            logger.error(f"[Rank {self.tp_rank}] write key {key} failed: {e}")
            return False
        finally:
            if fd is not None:
                os.close(fd)
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _read_one(
        self,
        ctx: _TargetCtx,
        key: str,
        host_token_indices: torch.Tensor,
    ) -> bool:
        path = self._key_to_path(ctx, key)
        # Reads always cover the full local layer set: layer_range slicing is
        # a write-only path (cross-PP offload), never used on read.
        get_meta = getattr(self.mem_pool_host, "get_page_buffer_meta_for_restore", self.mem_pool_host.get_page_buffer_meta)
        ptr_list, size_list = get_meta(host_token_indices, None)
        views = _ptr_views(ptr_list, size_list)

        try:
            fd = os.open(path, os.O_RDONLY)
        except FileNotFoundError:
            return False
        except OSError as e:
            logger.error(f"[Rank {self.tp_rank}] open key {key} failed: {e}")
            return False
        try:
            if _readv_all(fd, views) != ctx.bytes_per_page:
                logger.warning(f"[Rank {self.tp_rank}] short read for key {key}")
                return False
            return True
        finally:
            os.close(fd)

    # ------------------------------------------------------------------
    # HiCacheStorage interface
    # ------------------------------------------------------------------

    def batch_get_zero_copy(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        if not keys:
            return []
        assert (
            extra_info is None or extra_info.layer_range is None
        ), "layer_range slicing is not supported on the read path"
        ctx = self._get_ctx(self._resolve_pp_prefix(extra_info), None)
        page_size = self.mem_pool_host.page_size
        assert len(host_indices) == len(keys) * page_size

        start = time.perf_counter()
        results = list(
            self.executor.map(
                lambda i: self._read_one(
                    ctx,
                    keys[i],
                    host_indices[i * page_size : (i + 1) * page_size],
                ),
                range(len(keys)),
            )
        )
        self._record_metric(
            is_backup=False,
            num_pages=sum(1 for r in results if r),
            elapsed=time.perf_counter() - start,
            bytes_per_page=ctx.bytes_per_page,
        )
        return results

    def batch_set_zero_copy(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
        on_complete_callback: Optional[Callable] = None,
    ) -> List[bool]:
        try:
            if not keys:
                return []
            layer_range = extra_info.layer_range if extra_info is not None else None
            ctx = self._get_ctx(self._resolve_pp_prefix(extra_info), layer_range)
            page_size = self.mem_pool_host.page_size
            assert len(host_indices) == len(keys) * page_size

            start = time.perf_counter()
            results = list(
                self.executor.map(
                    lambda i: self._write_one(
                        ctx,
                        keys[i],
                        host_indices[i * page_size : (i + 1) * page_size],
                        layer_range,
                    ),
                    range(len(keys)),
                )
            )
            self._record_metric(
                is_backup=True,
                num_pages=sum(1 for r in results if r),
                elapsed=time.perf_counter() - start,
                bytes_per_page=ctx.bytes_per_page,
            )
            return results
        finally:
            # The controller arms exactly one async ack per batch call
            # (num_batches for non-lmcache backends); never skip this or the
            # host pages stay protected forever.
            if on_complete_callback is not None:
                on_complete_callback(None)

    def exists(self, key: str) -> bool:
        ctx = self._get_ctx(self.pp_prefix, None)
        return os.path.exists(self._key_to_path(ctx, key))

    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        mask = self._batch_stat(keys, extra_info)
        for i, ok in enumerate(mask):
            if not ok:
                return i
        return len(keys)

    def batch_exists_mask(
        self,
        keys: List[str],
        kvp_rank: int,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[int]:
        mask = self._batch_stat(keys, extra_info)
        return [kvp_rank if ok else -1 for ok in mask]

    def _batch_stat(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo]
    ) -> List[bool]:
        if not keys:
            return []
        ctx = self._get_ctx(self._resolve_pp_prefix(extra_info), None)
        paths = [self._key_to_path(ctx, k) for k in keys]
        return list(self.executor.map(os.path.exists, paths))

    def clear(self) -> None:
        logger.warning(
            "HiCacheHF3FS.clear() is not supported; remove "
            f"{self.file_path_prefix}/{self.model_name} manually."
        )

    def close(self) -> None:
        self.executor.shutdown(wait=True)

    def get_stats(self):
        if not self.enable_storage_metrics:
            return None
        from sglang.srt.observability.metrics.collector import StorageMetrics

        m = StorageMetrics()
        m.prefetch_pgs.extend(self.prefetch_pgs)
        m.backup_pgs.extend(self.backup_pgs)
        m.prefetch_bandwidth.extend(self.prefetch_bandwidth)
        m.backup_bandwidth.extend(self.backup_bandwidth)
        self.prefetch_pgs.clear()
        self.backup_pgs.clear()
        self.prefetch_bandwidth.clear()
        self.backup_bandwidth.clear()
        return m

    def _record_metric(
        self, is_backup: bool, num_pages: int, elapsed: float, bytes_per_page: int
    ):
        if not self.enable_storage_metrics or num_pages == 0:
            return
        gb = num_pages * bytes_per_page / (1 << 30)
        bw = gb / max(elapsed, 1e-6)
        if is_backup:
            self.backup_pgs.append(num_pages)
            self.backup_bandwidth.append(bw)
        else:
            self.prefetch_pgs.append(num_pages)
            self.prefetch_bandwidth.append(bw)

    # ------------------------------------------------------------------
    # Deprecated tensor-based interface (kept to satisfy the ABC; the L3
    # controller routes hf3fs through the zero-copy methods above).
    # ------------------------------------------------------------------

    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        raise NotImplementedError("use batch_get_zero_copy")

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None] | int:
        raise NotImplementedError("use batch_get_zero_copy")

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        raise NotImplementedError("use batch_set_zero_copy")

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        raise NotImplementedError("use batch_set_zero_copy")
