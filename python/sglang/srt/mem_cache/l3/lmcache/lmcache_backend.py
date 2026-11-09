from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
from typing import Any, Callable

import torch

if not torch.cuda.is_available():
    try:
        import lmcache.non_cuda_equivalents as _nce  # type: ignore

        sys.modules.setdefault("lmcache.c_ops", _nce)
    except ImportError:
        pass

from sglang.srt.mem_cache.l3.storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)
from sglang.srt.mem_cache.memory_pool.host_memory_pool import HostTokenToKVPool

logger = logging.getLogger(__name__)


def start_background_loop() -> asyncio.AbstractEventLoop:
    loop = asyncio.new_event_loop()

    def _run():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    t = threading.Thread(target=_run, name="lmcache-backend-loop", daemon=True)
    t.start()
    return loop


def make_cache_engine_key(
    key: str,
    model_name: str,
    worker_id: int,
    dtype: torch.dtype,
    world_size: int = 1,
):
    try:
        from lmcache.utils import CacheEngineKey
    except ImportError as exc:
        raise ImportError("lmcache is not installed. Please install it with: pip install lmcache") from exc

    chunk_hash_int = int(key[:16], 16) if len(key) >= 16 else int(key, 16)
    if chunk_hash_int >= 2**63:
        chunk_hash_int -= 2**64
    return CacheEngineKey(
        model_name=model_name,
        world_size=world_size,
        worker_id=worker_id,
        chunk_hash=chunk_hash_int,
        dtype=dtype,
    )


def make_memory_obj(tensor: torch.Tensor, is_mla: bool = False):
    try:
        from lmcache.v1.memory_management import MemoryFormat, MemoryObjMetadata, TensorMemoryObj
    except ImportError as exc:
        raise ImportError("lmcache is not installed. Please install it with: pip install lmcache") from exc

    t = tensor.contiguous()
    fmt = MemoryFormat.KV_MLA_FMT if is_mla else MemoryFormat.KV_2LTD
    meta = MemoryObjMetadata(
        shape=t.shape,
        dtype=t.dtype,
        address=t.data_ptr(),
        phy_size=t.nelement() * t.element_size(),
        ref_count=1,
        fmt=fmt,
        cached_positions=None,
    )
    return TensorMemoryObj(raw_data=t, metadata=meta, parent_allocator=None)


def get_model_geometry(mem_pool_host: HostTokenToKVPool, is_mla: bool):
    kv_dtype = mem_pool_host.dtype
    chunk_size = mem_pool_host.page_size
    num_layers = mem_pool_host.layer_num

    if is_mla and hasattr(mem_pool_host, "kv_cache_dim"):
        num_kv_heads = 1
        head_dim = mem_pool_host.kv_cache_dim
    else:
        num_kv_heads = mem_pool_host.head_num
        head_dim = mem_pool_host.head_dim

    return num_layers, num_kv_heads, head_dim, chunk_size, kv_dtype


class LMCacheLocalDiskBackend(HiCacheStorage):
    def __init__(self, storage_config: HiCacheStorageConfig):
        extra = storage_config.extra_config or {}

        self.disk_path: str = extra.get("lmcache_local_disk", "")
        if not self.disk_path:
            raise ValueError("LMCacheLocalDiskBackend requires extra_config['lmcache_local_disk'] to be set.")
        self.max_disk_gb: float = float(extra.get("lmcache_max_disk_gb", 100.0))
        self.max_cpu_mb: float = float(extra.get("lmcache_max_cpu_mb", 512.0))
        self.extra = extra

        self.model_name: str = storage_config.model_name or "unknown_model"
        self.tp_rank: int = storage_config.tp_rank
        self.tp_size: int = storage_config.tp_size
        self.pp_rank: int = getattr(storage_config, "pp_rank", 0)
        self.pp_size: int = getattr(storage_config, "pp_size", 1)
        self.is_mla_backend: bool = storage_config.is_mla_model
        self.enable_l15 = getattr(storage_config, "enable_l15", False)
        self.store_only_on_first_rank = self.is_mla_backend and not self.enable_l15
        self.is_page_first: bool = storage_config.is_page_first_layout

        self._loop = start_background_loop()
        self.kv_dtype: torch.dtype | None = None
        self.cpu_backend = None
        self.disk_backend = None
        self.kv_rank = 0 if self.store_only_on_first_rank else self.tp_rank

    def register_mem_pool_host(self, mem_pool_host: HostTokenToKVPool):
        super().register_mem_pool_host(mem_pool_host)
        self.kv_dtype = mem_pool_host.dtype
        self.post_init(mem_pool_host)

    def post_init(self, mem_pool_host: HostTokenToKVPool):
        kv_shape, chunk_size = self.compute_kv_shapes(mem_pool_host)
        self.create_lmcache_backends(mem_pool_host, kv_shape, chunk_size)
        self.check_odirect_alignment(mem_pool_host, chunk_size)
        logger.info(
            "LMCacheLocalDiskBackend initialised: disk_path=%s max_disk_gb=%.1f tp_rank=%d chunk_size=%d",
            self.disk_path,
            self.max_disk_gb,
            self.tp_rank,
            chunk_size,
        )

    def compute_kv_shapes(self, mem_pool_host: HostTokenToKVPool):
        num_layers, num_kv_heads, _, chunk_size, kv_dtype = get_model_geometry(mem_pool_host, self.is_mla_backend)
        kv_size = 1 if self.is_mla_backend else 2
        storage_bytes_per_token = mem_pool_host.get_storage_size_per_token()
        storage_head_dim = storage_bytes_per_token // (kv_dtype.itemsize * num_layers * kv_size * num_kv_heads)
        kv_shape = (num_layers, kv_size, chunk_size, num_kv_heads, storage_head_dim)

        storage_last_dim = storage_bytes_per_token // (kv_dtype.itemsize * num_layers)
        if self.is_page_first:
            self._kv_shape = torch.Size((chunk_size, num_layers, num_kv_heads, storage_last_dim))
        else:
            self._kv_shape = torch.Size((num_layers, chunk_size, num_kv_heads, storage_last_dim))

        return kv_shape, chunk_size

    def create_lmcache_backends(self, mem_pool_host: HostTokenToKVPool, kv_shape, chunk_size: int):
        try:
            from lmcache.v1.config import LMCacheEngineConfig
            from lmcache.v1.metadata import LMCacheMetadata
            from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
            from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend
        except ImportError as exc:
            raise ImportError("lmcache is not installed. Please install it with: pip install lmcache") from exc

        kv_dtype = mem_pool_host.dtype

        lmcache_cfg = LMCacheEngineConfig(
            chunk_size=chunk_size,
            local_cpu=True,
            local_disk=self.disk_path,
            max_local_disk_size=self.max_disk_gb,
            max_local_cpu_size=self.max_cpu_mb / 1024.0,
            lmcache_instance_id=(
                f"{self.model_name}_pp{self.pp_rank}_tp{self.kv_rank}" if self.pp_size > 1 else f"{self.model_name}_rank{self.kv_rank}"
            ),
            extra_config=self.extra,
        )

        lmcache_meta = LMCacheMetadata(
            model_name=self.model_name,
            world_size=self.tp_size,
            local_world_size=self.tp_size,
            worker_id=self.tp_rank,
            local_worker_id=self.tp_rank,
            kv_dtype=kv_dtype,
            kv_shape=kv_shape,
            use_mla=self.is_mla_backend,
            chunk_size=chunk_size,
        )

        self.cpu_backend = LocalCPUBackend(
            config=lmcache_cfg,
            metadata=lmcache_meta,
            dst_device="cpu",
        )
        self.disk_backend = LocalDiskBackend(
            config=lmcache_cfg,
            loop=self._loop,
            local_cpu_backend=self.cpu_backend,
            dst_device="cpu",
        )

    def check_odirect_alignment(self, mem_pool_host: HostTokenToKVPool, chunk_size: int):
        storage_bytes_per_token = mem_pool_host.get_storage_size_per_token()
        chunk_bytes = storage_bytes_per_token * chunk_size
        fs_block_size = os.statvfs(self.disk_path).f_bsize
        logger.info(
            "[LMCacheLocalDiskBackend] KV Chunk Bytes Size=%d, file system Block Size=%d",
            chunk_bytes,
            fs_block_size,
        )

        if not self.extra.get("use_odirect", False):
            return

        if chunk_bytes % fs_block_size != 0:
            logger.warning(
                "O_DIRECT is enabled but chunk buffer size (%d bytes) is not "
                "aligned to filesystem block size (%d bytes). O_DIRECT will "
                "NOT be effective — LMCache will fall back to buffered I/O. "
                "To fix, adjust page_size so that chunk_bytes is a multiple of %d.",
                chunk_bytes,
                fs_block_size,
                fs_block_size,
            )
        else:
            logger.info("[LMCacheLocalDiskBackend] O_DIRECT alignment OK.")

    def _key(self, hash_str: str):
        key_model_name = f"{self.model_name}_pp{self.pp_rank}" if self.pp_size > 1 else self.model_name
        # C8's old copy path cast BF16 IndexK numerically to int8 (800 rather
        # than 928 B/token). Those objects are not a raw-byte cache format.
        storage_format = getattr(self.mem_pool_host, "storage_format_id", None)
        if storage_format:
            key_model_name += "_" + storage_format
        return make_cache_engine_key(hash_str, key_model_name, self.kv_rank, self.kv_dtype or torch.float16, self.tp_size)

    def _extract_tensor(self, host_indices: torch.Tensor) -> torch.Tensor:
        return self.mem_pool_host.get_data_for_storage(host_indices)

    def _write_tensor_back(self, tensor: torch.Tensor, host_indices: torch.Tensor) -> None:
        self.mem_pool_host.set_data_from_storage(host_indices, tensor)

    def batch_set_zero_copy(
        self,
        keys: list[str],
        host_indices: torch.Tensor,
        extra_info: HiCacheStorageExtraInfo | None = None,
        on_complete_callback: Callable | None = None,
    ) -> list[bool]:
        if not keys:
            return []

        logger.debug(f"BatchSet keys: {keys} host_indices: {host_indices}")
        page_size = self.mem_pool_host.page_size
        results: list[bool] = []

        for i, key_str in enumerate(keys):
            page_indices = host_indices[i * page_size : (i + 1) * page_size]
            lmcache_key = self._key(key_str)
            if self.disk_backend.contains(lmcache_key):
                results.append(True)
                if on_complete_callback is not None:
                    on_complete_callback(lmcache_key)
                continue

            if self.disk_backend.exists_in_put_tasks(lmcache_key):
                results.append(True)
                if on_complete_callback is not None:
                    on_complete_callback(lmcache_key)
                continue

            tensor = self._extract_tensor(page_indices)
            mem_obj = make_memory_obj(tensor, is_mla=self.is_mla_backend)
            self.disk_backend.submit_put_task(lmcache_key, mem_obj, on_complete_callback=on_complete_callback)
            results.append(True)

        return results

    def batch_get_zero_copy(
        self,
        keys: list[str],
        host_indices: torch.Tensor,
        extra_info: HiCacheStorageExtraInfo | None = None,
    ) -> list[bool]:
        if not keys:
            return []

        logger.debug(f"BatchGet keys: {keys} host_indices: {host_indices}")
        page_size = self.mem_pool_host.page_size
        results: list[bool] = []

        for i, key_str in enumerate(keys):
            page_indices = host_indices[i * page_size : (i + 1) * page_size]
            lmcache_key = self._key(key_str)

            if self.store_only_on_first_rank and self.tp_rank != 0:
                mem_obj = self.load_from_rank0_file(lmcache_key)
            else:
                if not self.disk_backend.contains(lmcache_key):
                    results.append(False)
                    continue
                mem_obj = self.disk_backend.get_blocking(lmcache_key)

            if mem_obj is None or mem_obj.tensor is None:
                results.append(False)
                continue

            self._write_tensor_back(mem_obj.tensor, page_indices)
            mem_obj.ref_count_down()
            results.append(True)

        return results

    def load_from_rank0_file(self, lmcache_key):
        from lmcache.v1.memory_management import MemoryFormat

        path = self.disk_backend._key_to_path(lmcache_key)
        if not os.path.isfile(path):
            return None
        fmt = MemoryFormat.KV_MLA_FMT if self.is_mla_backend else MemoryFormat.KV_2LTD
        return self.disk_backend.load_bytes_from_disk(lmcache_key, path, dtype=self.kv_dtype, shape=self._kv_shape, fmt=fmt)

    def get(
        self,
        key: str,
        target_location: Any | None = None,
        target_sizes: Any | None = None,
    ) -> torch.Tensor | None:
        lmcache_key = self._key(key)
        mem_obj = self.disk_backend.get_blocking(lmcache_key)
        if mem_obj is None:
            return None
        return mem_obj.tensor

    def batch_get(
        self,
        keys: list[str],
        target_locations: Any | None = None,
        target_sizes: Any | None = None,
    ) -> int:
        count = 0
        for key in keys:
            lmcache_key = self._key(key)
            if not self.disk_backend.contains(lmcache_key):
                break
            count += 1
        return count

    def set(
        self,
        key: str,
        value: Any | None = None,
        target_location: Any | None = None,
        target_sizes: Any | None = None,
    ) -> bool:
        if value is None:
            return False
        lmcache_key = self._key(key)
        mem_obj = make_memory_obj(value, is_mla=self.is_mla_backend) if isinstance(value, torch.Tensor) else value
        self.disk_backend.submit_put_task(lmcache_key, mem_obj)
        return True

    def batch_set(
        self,
        keys: list[str],
        values: Any | None = None,
        target_locations: Any | None = None,
        target_sizes: Any | None = None,
    ) -> bool:
        if values is None or len(keys) != len(values):
            return False
        return all(self.set(key, value) for key, value in zip(keys, values))

    def exists(self, key: str) -> bool:
        return self.disk_backend.contains(self._key(key))

    def batch_exists(self, keys: list[str], extra_info: HiCacheStorageExtraInfo | None = None) -> int:
        if self.store_only_on_first_rank and self.tp_rank != 0:
            return len(keys)
        return self._batch_exists_impl(keys, extra_info)

    def _batch_exists_impl(self, keys: list[str], extra_info: HiCacheStorageExtraInfo | None = None) -> int:
        for i, key in enumerate(keys):
            if not self.exists(key):
                return i
        return len(keys)

    def batch_exists_mask(self, keys: list[str], kvp_rank: int, extra_info: HiCacheStorageExtraInfo | None = None) -> list[int]:
        return [kvp_rank if self.exists(k) else -1 for k in keys]

    def clear(self) -> None:
        logger.warning("LMCacheLocalDiskBackend.clear() is not supported; manually remove the disk cache directory.")

    def close(self) -> None:
        self.disk_backend.close()
        self._loop.call_soon_threadsafe(self._loop.stop)


# FIXME: This is not ready for now!
class LMCacheRustRawBlockBackend(HiCacheStorage):
    def __init__(self, storage_config: HiCacheStorageConfig):
        extra = storage_config.extra_config or {}

        self.model_name: str = storage_config.model_name or "unknown_model"
        self.tp_rank: int = storage_config.tp_rank
        self.tp_size: int = storage_config.tp_size
        self.pp_rank: int = getattr(storage_config, "pp_rank", 0)
        self.pp_size: int = getattr(storage_config, "pp_size", 1)
        self.is_mla_backend: bool = storage_config.is_mla_model
        self.is_page_first: bool = storage_config.is_page_first_layout
        self.extra = extra

        self._loop = start_background_loop()
        self.kv_dtype: torch.dtype | None = None
        self.cpu_backend = None
        self._rust_backend = None

    def register_mem_pool_host(self, mem_pool_host: HostTokenToKVPool):
        if getattr(getattr(mem_pool_host, "device_pool", None), "dsa_kv_int8_packed", False):
            raise ValueError("LMCache Rust raw-block does not support packed DSA KV/IndexK; use local disk, Mooncake or HF3FS")
        super().register_mem_pool_host(mem_pool_host)
        self.kv_dtype = mem_pool_host.dtype

        try:
            from lmcache.v1.config import LMCacheEngineConfig
            from lmcache.v1.metadata import LMCacheMetadata
            from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
            from lmcache.v1.storage_backend.plugins.rust_raw_block_backend import (
                RustRawBlockBackend,
            )
        except ImportError as exc:
            raise ImportError(
                "lmcache (with Rust extension) is not installed. Please install lmcache and build the Rust raw-block extension."
            ) from exc

        num_layers, num_kv_heads, head_dim, chunk_size, kv_dtype = get_model_geometry(mem_pool_host, self.is_mla_backend)

        kv_size = 1 if self.is_mla_backend else 2
        kv_shape = (num_layers, kv_size, chunk_size, num_kv_heads, head_dim)

        max_cpu_mb: float = float(self.extra.get("lmcache_max_cpu_mb", 512.0))

        lmcache_cfg = LMCacheEngineConfig(
            chunk_size=chunk_size,
            local_cpu=True,
            max_local_cpu_size=max_cpu_mb / 1024.0,
            lmcache_instance_id=(
                f"{self.model_name}_pp{self.pp_rank}_tp{self.tp_rank}" if self.pp_size > 1 else f"{self.model_name}_tp{self.tp_rank}"
            ),
            extra_config=self.extra,
        )

        lmcache_meta = LMCacheMetadata(
            model_name=self.model_name,
            world_size=self.tp_size,
            local_world_size=self.tp_size,
            worker_id=self.tp_rank,
            local_worker_id=self.tp_rank,
            kv_dtype=kv_dtype,
            kv_shape=kv_shape,
            use_mla=self.is_mla_backend,
            chunk_size=chunk_size,
        )

        self.cpu_backend = LocalCPUBackend(
            config=lmcache_cfg,
            metadata=lmcache_meta,
            dst_device="cpu",
        )

        self._rust_backend = RustRawBlockBackend(
            config=lmcache_cfg,
            metadata=lmcache_meta,
            local_cpu_backend=self.cpu_backend,
            loop=self._loop,
            dst_device="cpu",
        )

        logger.info(
            "LMCacheRustRawBlockBackend initialised: tp_rank=%d tp_size=%d num_layers=%d chunk_size=%d",
            self.tp_rank,
            self.tp_size,
            num_layers,
            chunk_size,
        )

    def _key(self, hash_str: str):
        key_model_name = f"{self.model_name}_pp{self.pp_rank}" if self.pp_size > 1 else self.model_name
        return make_cache_engine_key(hash_str, key_model_name, self.tp_rank, self.kv_dtype or torch.float16, self.tp_size)

    def _extract_tensor(self, host_indices: torch.Tensor) -> torch.Tensor:
        kv_buf = self.mem_pool_host.buffers.kv_buffer
        if self.is_page_first:
            return kv_buf[host_indices].contiguous()
        else:
            return kv_buf[:, host_indices].contiguous()

    def _write_tensor_back(self, tensor: torch.Tensor, host_indices: torch.Tensor) -> None:
        kv_buf = self.mem_pool_host.buffers.kv_buffer
        if self.is_page_first:
            kv_buf[host_indices] = tensor
        else:
            kv_buf[:, host_indices] = tensor

    def batch_set_zero_copy(
        self,
        keys: list[str],
        host_indices: torch.Tensor,
        extra_info: HiCacheStorageExtraInfo | None = None,
    ) -> list[bool]:
        if not keys:
            return []

        page_size = self.mem_pool_host.page_size
        lmcache_keys = []
        mem_objs = []

        for i, key_str in enumerate(keys):
            page_indices = host_indices[i * page_size : (i + 1) * page_size]
            lmcache_key = self._key(key_str)

            if self._rust_backend.contains(lmcache_key):
                continue

            tensor = self._extract_tensor(page_indices)
            lmcache_keys.append(lmcache_key)
            mem_objs.append(make_memory_obj(tensor, is_mla=self.is_mla_backend))

        if lmcache_keys:
            self._rust_backend.batched_submit_put_task(lmcache_keys, mem_objs)

        return [True] * len(keys)

    def batch_get_zero_copy(
        self,
        keys: list[str],
        host_indices: torch.Tensor,
        extra_info: HiCacheStorageExtraInfo | None = None,
    ) -> list[bool]:
        if not keys:
            return []

        page_size = self.mem_pool_host.page_size
        lmcache_keys = [self._key(k) for k in keys]

        mem_objs = self._rust_backend.batched_get_blocking(lmcache_keys)

        results: list[bool] = []
        for i, mem_obj in enumerate(mem_objs):
            if mem_obj is None or mem_obj.tensor is None:
                results.append(False)
                continue
            page_indices = host_indices[i * page_size : (i + 1) * page_size]
            self._write_tensor_back(mem_obj.tensor, page_indices)
            results.append(True)

        return results

    def get(
        self,
        key: str,
        target_location: Any | None = None,
        target_sizes: Any | None = None,
    ) -> torch.Tensor | None:
        mem_obj = self._rust_backend.get_blocking(self._key(key))
        if mem_obj is None:
            return None
        return mem_obj.tensor

    def batch_get(
        self,
        keys: list[str],
        target_locations: Any | None = None,
        target_sizes: Any | None = None,
    ) -> int:
        lmcache_keys = [self._key(k) for k in keys]
        for i, k in enumerate(lmcache_keys):
            if not self._rust_backend.contains(k):
                return i
        return len(keys)

    def set(
        self,
        key: str,
        value: Any | None = None,
        target_location: Any | None = None,
        target_sizes: Any | None = None,
    ) -> bool:
        if value is None:
            return False
        lmcache_key = self._key(key)
        mem_obj = make_memory_obj(value, is_mla=self.is_mla_backend) if isinstance(value, torch.Tensor) else value
        self._rust_backend.batched_submit_put_task([lmcache_key], [mem_obj])
        return True

    def batch_set(
        self,
        keys: list[str],
        values: Any | None = None,
        target_locations: Any | None = None,
        target_sizes: Any | None = None,
    ) -> bool:
        if values is None or len(keys) != len(values):
            return False
        lmcache_keys = [self._key(k) for k in keys]
        mem_objs = [make_memory_obj(v, is_mla=self.is_mla_backend) if isinstance(v, torch.Tensor) else v for v in values]
        self._rust_backend.batched_submit_put_task(lmcache_keys, mem_objs)
        return True

    def exists(self, key: str) -> bool:
        return self._rust_backend.contains(self._key(key))

    def batch_exists(self, keys: list[str], extra_info: HiCacheStorageExtraInfo | None = None) -> int:
        for i, key in enumerate(keys):
            if not self.exists(key):
                return i
        return len(keys)

    def batch_exists_mask(self, keys: list[str], kvp_rank: int, extra_info: HiCacheStorageExtraInfo | None = None) -> list[int]:
        return [kvp_rank if self.exists(k) else -1 for k in keys]

    def clear(self) -> None:
        logger.warning("LMCacheRustRawBlockBackend.clear() is not supported; the device index will be invalidated on the next init.")

    def close(self) -> None:
        self._rust_backend.close()
        self._loop.call_soon_threadsafe(self._loop.stop)
