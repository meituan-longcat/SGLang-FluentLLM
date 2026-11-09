import json
import logging
import os
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional

import requests
import torch

from sglang.srt.env import envs
from sglang.srt.mem_cache.l3.storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)
from sglang.srt.mem_cache.memory_pool.host_memory_pool import HostTokenToKVPool

DEFAULT_LOCAL_BUFFER_SIZE = 16 * 1024 * 1024
SETUP_TIMEOUT = 600


def _get_l3_timeout_ms(name: str, default_ms: int) -> int:
    value = os.getenv(name, str(default_ms))
    try:
        timeout_ms = int(value)
    except ValueError:
        logger.warning(
            "Invalid %s=%r; disabling L3 timeout",
            name,
            value,
        )
        return 0
    if timeout_ms < 0:
        logger.warning(
            "Negative %s=%d; disabling L3 timeout",
            name,
            timeout_ms,
        )
        return 0
    return timeout_ms


MOONCAKE_RANK_BASE_PORT = int(os.getenv("MOONCAKE_RANK_BASE_PORT", "12002"))

logger = logging.getLogger(__name__)
L3_TIMEOUT_MS = _get_l3_timeout_ms("MOONCAKE_L3_TIMEOUT_MS", 2000)
# Keep the existing GET-only knob as a compatibility override. New
# deployments only need MOONCAKE_L3_TIMEOUT_MS.
L3_GET_TIMEOUT_MS = _get_l3_timeout_ms("MOONCAKE_L3_GET_TIMEOUT_MS", L3_TIMEOUT_MS)
L3_PUT_TIMEOUT_MS = L3_TIMEOUT_MS


def _parse_global_segment_size(value) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if s.endswith("gb"):
            num = s[:-2].strip()
            if not num:
                raise ValueError("Invalid global_segment_size: missing number before 'gb'")
            return int(num) * 1024 * 1024 * 1024
        return int(s)
    return int(value)


@dataclass
class MooncakeStoreConfig:
    local_hostname: str
    metadata_server: str
    global_segment_size: int
    protocol: str
    device_name: str
    master_server_address: str
    master_metrics_port: int
    check_server: bool

    @staticmethod
    def from_file() -> "MooncakeStoreConfig":
        if not envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.is_set():
            raise RuntimeError(f"Config file path not set. Please set {envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.name}")
        file_path = envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.value
        try:
            with open(file_path) as fin:
                config = json.load(fin)
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {file_path}: {str(e)}") from e

        if "master_server_address" not in config:
            raise ValueError("master_server_address is required in config file")

        return MooncakeStoreConfig(
            local_hostname=config.get("local_hostname", envs.MOONCAKE_LOCAL_HOSTNAME.default),
            metadata_server=config.get("metadata_server", envs.MOONCAKE_TE_META_DATA_SERVER.default),
            global_segment_size=_parse_global_segment_size(config.get("global_segment_size", envs.MOONCAKE_GLOBAL_SEGMENT_SIZE.default)),
            protocol=config.get("protocol", envs.MOONCAKE_PROTOCOL.default),
            device_name=config.get("device_name", envs.MOONCAKE_DEVICE.default),
            master_server_address=config.get("master_server_address"),
            master_metrics_port=config.get("master_metrics_port", envs.MOONCAKE_MASTER_METRICS_PORT.default),
            check_server=config.get("check_server", envs.MOONCAKE_CHECK_SERVER.default),
        )

    @staticmethod
    def load_from_env() -> "MooncakeStoreConfig":
        if not envs.MOONCAKE_MASTER.is_set():
            raise ValueError("The environment variable 'MOONCAKE_MASTER' is not set.")

        if envs.MOONCAKE_LOCAL_HOSTNAME.is_set():
            local_hostname = envs.MOONCAKE_LOCAL_HOSTNAME.value
        else:
            local_hostname = os.getenv("LOCAL_HOSTNAME", envs.MOONCAKE_LOCAL_HOSTNAME.default)

        return MooncakeStoreConfig(
            local_hostname=local_hostname,
            metadata_server=envs.MOONCAKE_TE_META_DATA_SERVER.value,
            global_segment_size=_parse_global_segment_size(envs.MOONCAKE_GLOBAL_SEGMENT_SIZE.value),
            protocol=envs.MOONCAKE_PROTOCOL.value,
            device_name=envs.MOONCAKE_DEVICE.value,
            master_server_address=envs.MOONCAKE_MASTER.value,
            master_metrics_port=envs.MOONCAKE_MASTER_METRICS_PORT.value,
            check_server=envs.MOONCAKE_CHECK_SERVER.value,
        )

    @staticmethod
    def load_from_extra_config(extra_config: dict) -> "MooncakeStoreConfig":
        if "master_server_address" not in extra_config:
            raise ValueError("master_server_address is required in extra_config")

        return MooncakeStoreConfig(
            local_hostname=extra_config.get("local_hostname", envs.MOONCAKE_LOCAL_HOSTNAME.default),
            metadata_server=extra_config.get("metadata_server", envs.MOONCAKE_TE_META_DATA_SERVER.default),
            global_segment_size=_parse_global_segment_size(extra_config.get("global_segment_size", envs.MOONCAKE_GLOBAL_SEGMENT_SIZE.default)),
            protocol=extra_config.get("protocol", envs.MOONCAKE_PROTOCOL.default),
            device_name=extra_config.get("device_name", envs.MOONCAKE_DEVICE.default),
            master_server_address=extra_config["master_server_address"],
            master_metrics_port=extra_config.get("master_metrics_port", envs.MOONCAKE_MASTER_METRICS_PORT.default),
            check_server=extra_config.get("check_server", envs.MOONCAKE_CHECK_SERVER.default),
        )


class MooncakeStore(HiCacheStorage):
    def __init__(self, storage_config: HiCacheStorageConfig = None):
        try:
            from mooncake.store import MooncakeDistributedStore
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://kvcache-ai.github.io/Mooncake/getting_started/build.html"
                "to run SGLang with MooncakeConnector."
            ) from e

        try:
            self.store = MooncakeDistributedStore()

            extra_config = getattr(storage_config, "extra_config", None) if storage_config else None
            if extra_config is not None and extra_config.get("master_server_address") is not None:
                self.config = MooncakeStoreConfig.load_from_extra_config(extra_config)
                logger.info("Mooncake Configuration loaded from extra_config successfully.")
            elif envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.is_set():
                self.config = MooncakeStoreConfig.from_file()
                logger.info("Mooncake Configuration loaded from file successfully.")
            else:
                self.config = MooncakeStoreConfig.load_from_env()
                logger.info("Mooncake Configuration loaded from env successfully.")

            self.extra_backend_tag = None
            if extra_config and "extra_backend_tag" in extra_config:
                self.extra_backend_tag = extra_config["extra_backend_tag"]
                logger.info(f"Using extra_backend_tag: {self.extra_backend_tag}")

            if self.config.check_server:
                self.check_server()

            device_name = self.config.device_name
            if device_name and device_name.strip().startswith("{"):
                try:
                    device_config = json.loads(device_name)
                    if storage_config and hasattr(storage_config, "tp_rank"):
                        tp_rank = storage_config.tp_rank
                        device_name = device_config.get(tp_rank, "")
                        if not device_name:
                            device_name = device_config.get(str(tp_rank), "")
                    else:
                        device_name = ""
                except (json.JSONDecodeError, AttributeError):
                    logger.warning(f"Failed to parse device_name as JSON: {device_name}")
                    device_name = ""

            self.config.local_hostname = self._concat_local_hostname_port(self.config.local_hostname)

            ret_code = self.store.setup(
                self.config.local_hostname,
                self.config.metadata_server,
                0,  # per_tp_global_segment_size,
                DEFAULT_LOCAL_BUFFER_SIZE,
                "rdma",  # self.config.protocol,
                device_name,
                self.config.master_server_address,
            )
            if ret_code:
                raise RuntimeError(f"Failed to setup Mooncake store, error code: {ret_code}")
            logger.info("Mooncake store setup successfully.")

            self.warmup()
            logger.info("Mooncake store warmup successfully.")

            if storage_config is not None:
                self.is_mla_backend = storage_config.is_mla_model
                self.local_rank = storage_config.tp_rank
                self.pp_rank = getattr(storage_config, "pp_rank", 0)
                self.pp_size = getattr(storage_config, "pp_size", 1)
            else:
                self.is_mla_backend = False
                self.local_rank = 0
                self.pp_rank = 0
                self.pp_size = 1

            logger.warning(
                "[L3-Timeout-Config] timeout_ms=%d scope=mooncake_store_batch_get pid=%d tp_rank=%d pp_rank=%d",
                L3_GET_TIMEOUT_MS,
                os.getpid(),
                self.local_rank,
                self.pp_rank,
            )
            logger.warning(
                "[L3-Timeout-Config] timeout_ms=%d scope=mooncake_store_batch_put pid=%d tp_rank=%d pp_rank=%d",
                L3_PUT_TIMEOUT_MS,
                os.getpid(),
                self.local_rank,
                self.pp_rank,
            )
            # pp_prefix is empty when pp_size == 1, so single-PP behavior is unchanged.
            # pp{rank}of{size}_ is the prefill-perspective namespace: a prefill rank
            # writes/reads its own i; decode (PP1) overrides per-call to write all i.
            self.pp_prefix = f"pp{self.pp_rank}of{self.pp_size}_" if self.pp_size > 1 else ""
            self.model_prefix = self._resolve_model_prefix(storage_config)

        except ValueError as e:
            logger.error("Configuration loading failed: %s", e)
            raise
        except Exception as exc:
            logger.error("An error occurred while loading the configuration: %s", exc)
            raise

    @staticmethod
    def _resolve_model_prefix(
        storage_config: Optional[HiCacheStorageConfig],
    ) -> str:
        # Resolve once during store initialization, not for every key operation.
        namespace = os.environ.get("L3_NAMESPACE")
        model_name = namespace
        source = "L3_NAMESPACE"
        if not model_name:
            model_name = os.environ.get("ARGS_MODEL_NAME")
            source = "ARGS_MODEL_NAME"
        if not model_name and storage_config is not None:
            model_name = storage_config.model_name
            source = "storage_config.model_name"
        if not model_name:
            raise ValueError("L3 model namespace is required: set L3_NAMESPACE or ARGS_MODEL_NAME")
        logger.info(
            "[L3-Key-Namespace] L3_NAMESPACE=%r namespace_applied=%s source=%s model_prefix=%r",
            namespace,
            bool(namespace),
            source,
            model_name,
        )
        return model_name

    def _concat_local_hostname_port(self, local_hostname: str) -> str:
        try:
            from sglang.srt.distributed import get_world_group

            global_rank = get_world_group().rank
        except Exception as e:
            logger.warning(
                "Mooncake: cannot resolve global rank (%s); leaving local_hostname=%s with engine-assigned random port.",
                e,
                local_hostname,
            )
            return local_hostname

        # Strip any existing port: keep only the host/IP. Guard IPv6 ("[::1]").
        host = local_hostname
        if host.startswith("["):
            # [ipv6]:port -> [ipv6]
            host = host[: host.rindex("]") + 1] if "]" in host else host
        elif host.count(":") == 1:
            host = host.split(":", 1)[0]

        port = MOONCAKE_RANK_BASE_PORT + global_rank
        pinned = f"{host}:{port}"
        logger.info(
            "Mooncake: pinned local_hostname=%s (global_rank=%s, base_port=%s)",
            pinned,
            global_rank,
            MOONCAKE_RANK_BASE_PORT,
        )
        return pinned

    def check_server(self):
        master_server_ip = self.config.master_server_address.split(":")[0]
        segments_url = f"http://{master_server_ip}:{self.config.master_metrics_port}/get_all_segments"
        start_time = time.perf_counter()

        check_result = False
        while time.perf_counter() - start_time < SETUP_TIMEOUT:
            try:
                check_segments_resp = requests.get(segments_url, timeout=3)
            except Exception:
                logger.info(
                    "waiting mooncake store server started, cost_time: %.2f seconds.",
                    time.perf_counter() - start_time,
                )
                time.sleep(3)
                continue

            if check_segments_resp.text == "":
                logger.info(
                    "waiting mooncake store server started, cost_time: %.2f seconds.",
                    time.perf_counter() - start_time,
                )
                time.sleep(3)
                continue

            logger.info("Mooncake store server started successfully.")
            check_result = True
            break

        if not check_result:
            logger.error("Launch mooncake store server timeout")
            raise ValueError("Launch mooncake store server timeout")

    def warmup(self):
        warmup_key = "sglang_mooncake_store_warmup_key" + uuid.uuid4().hex
        warmup_value = bytes(4 * 1024)
        put_result = self.store.put(warmup_key, warmup_value)
        if put_result != 0:
            logger.warning(
                f"Mooncake store warmup put failed with code {put_result}, skipping warmup (this is expected when global segment size is 0)"
            )
            return
        try:
            assert self.store.is_exist(warmup_key) == 1
            assert self.store.get(warmup_key) == warmup_value
        except Exception as e:
            logger.warning(f"Mooncake store warmup get/assert failed: {e}, skipping warmup (RDMA may not be fully configured)")
            return

    def _iter_registered_tensors(self, name: str, buffer: Any):
        if buffer is None:
            return
        if isinstance(buffer, torch.Tensor):
            yield name, buffer
            return
        if isinstance(buffer, Sequence) and not isinstance(buffer, (str, bytes)):
            for idx, item in enumerate(buffer):
                yield from self._iter_registered_tensors(f"{name}[{idx}]", item)
            return
        logger.warning(f"Mooncake store: skip unsupported buffer container for {name}: {type(buffer)}")

    def _iter_registration_targets(self, buffers: Any):
        chunk_fields = [
            "k_buffer_chunks",
            "v_buffer_chunks",
            "index_k_buffer_chunks",
            "index_k_storage_zero_chunks",
        ]
        has_chunk_backing = any(getattr(buffers, field, None) is not None for field in chunk_fields)
        if has_chunk_backing:
            for name in chunk_fields:
                for tensor_name, tensor in self._iter_registered_tensors(name, getattr(buffers, name, None)):
                    yield tensor_name, tensor
            return

        for name in ["kv_buffer", "k_buffer", "v_buffer", "index_k_buffer"]:
            for tensor_name, tensor in self._iter_registered_tensors(name, getattr(buffers, name, None)):
                yield tensor_name, tensor

    def register_mem_pool_host(self, mem_pool_host: HostTokenToKVPool):
        super().register_mem_pool_host(mem_pool_host)
        assert self.mem_pool_host.layout in [
            "layer_first",
            "page_first",
            "page_first_direct",
            "page_head",
        ], "mooncake store storage backend only support layer_first/page_first/page_first_direct/page_head layout"
        buffers = self.mem_pool_host.buffers
        buffer_names = []
        buffer_ptrs = []
        buffer_sizes = []
        for tensor_name, tensor in self._iter_registration_targets(buffers):
            buffer_names.append(tensor_name)
            buffer_ptrs.append(tensor.data_ptr())
            buffer_sizes.append(tensor.numel() * tensor.element_size())

        if not buffer_ptrs:
            logger.warning("Mooncake store: no host buffers found, skipping registration")
            return

        ret_code = self.store.register_buffers(buffer_ptrs, buffer_sizes)
        if ret_code:
            raise RuntimeError(f"Mooncake store: batch host buffer registration failed, count={len(buffer_ptrs)}, code={ret_code}")
        logger.info(
            "Mooncake store: registered %d host buffers with one metadata publish, total_size=%d, first=%s, last=%s",
            len(buffer_ptrs),
            sum(buffer_sizes),
            buffer_names[0],
            buffer_names[-1],
        )

    def _get_mha_buffer_meta(self, keys, indices):
        ptr_list, element_size_list = self.mem_pool_host.get_page_buffer_meta(indices)
        key_list = []
        for key_ in keys:
            key_list.append(f"{key_}_{self.local_rank}_k")
            key_list.append(f"{key_}_{self.local_rank}_v")
        assert len(key_list) == len(ptr_list)
        return key_list, ptr_list, element_size_list

    @staticmethod
    def _uses_multi_buffer(buffer_ptrs: list[Any]) -> bool:
        return bool(buffer_ptrs) and isinstance(buffer_ptrs[0], Sequence)

    def _resolve_pp_prefix(self, extra_info: Optional[HiCacheStorageExtraInfo]) -> str:
        """Per-call PP namespace override wins over the instance-fixed prefix.

        Lets decode (PP1) write N objects, each tagged pp{i}of{N}_, while prefill
        ranks keep writing/reading their own self.pp_prefix.
        """
        if extra_info is not None and extra_info.pp_prefix:
            return self.model_prefix + extra_info.pp_prefix
        return self.model_prefix + self.pp_prefix

    @staticmethod
    def _pack_multi_buffer_meta(key_strs: list[str], ptr_list: list[int], element_size_list: list[int]):
        if len(ptr_list) == len(key_strs):
            return ptr_list, element_size_list

        assert len(key_strs) > 0
        assert len(ptr_list) == len(element_size_list)
        assert len(ptr_list) % len(key_strs) == 0

        nbuf = len(ptr_list) // len(key_strs)
        return [ptr_list[i : i + nbuf] for i in range(0, len(ptr_list), nbuf)], [
            element_size_list[i : i + nbuf] for i in range(0, len(element_size_list), nbuf)
        ]

    def _get_mla_buffer_meta(self, keys, indices, layer_range=None, *, for_restore=False):
        get_meta = self.mem_pool_host.get_page_buffer_meta
        if for_restore:
            get_meta = getattr(self.mem_pool_host, "get_page_buffer_meta_for_restore", get_meta)
        ptr_list, element_size_list = get_meta(indices, layer_range)
        key_list = []
        for key_ in keys:
            key_list.append(f"{key_}_k")
        ptr_list, element_size_list = self._pack_multi_buffer_meta(key_list, ptr_list, element_size_list)
        assert len(key_list) == len(ptr_list)
        return key_list, ptr_list, element_size_list

    def _batch_preprocess(self, keys, host_indices, layer_range=None, *, for_restore=False):
        assert len(keys) > 0
        assert len(keys) == len(host_indices) // self.mem_pool_host.page_size
        if self.is_mla_backend:
            return self._get_mla_buffer_meta(keys, host_indices, layer_range, for_restore=for_restore)
        else:
            return self._get_mha_buffer_meta(keys, host_indices)

    def _batch_postprocess(self, results: list[int], is_set_operate=False):
        if self.is_mla_backend:
            return [k_res == 0 if is_set_operate else k_res > 0 for k_res in results]
        else:
            kv_pairs = zip(results[::2], results[1::2])
            return [((k_res == 0 and v_res == 0) if is_set_operate else (k_res > 0 and v_res > 0)) for k_res, v_res in kv_pairs]

    def _expand_exist_query_keys(self, keys: list[str]):
        """Expand logical page hashes to Mooncake's physical object keys."""
        if self.is_mla_backend:
            if keys and keys[0].endswith("_k"):
                return list(keys), 1
            return [f"{key}_k" for key in keys], 1

        if keys and (keys[0].endswith("_k") or keys[0].endswith("_v")):
            return list(keys), 2

        query_keys = []
        for key in keys:
            query_keys.append(f"{key}_{self.local_rank}_k")
            query_keys.append(f"{key}_{self.local_rank}_v")
        return query_keys, 2

    def batch_get_zero_copy(
        self,
        keys: list[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> list[bool]:
        pp_prefix = self._resolve_pp_prefix(extra_info)
        logger.debug(f"[batch get] {pp_prefix=}")
        if pp_prefix:
            keys = [f"{pp_prefix}{key}" for key in keys]
        if self.extra_backend_tag is not None:
            prefix = self.extra_backend_tag
            keys = [f"{prefix}_{key}" for key in keys]

        # Reads cover the full layer set: prefill reads its whole pp{i}of{N}_ object,
        # and decode never reads. layer_range slicing is a write-only path, so reject
        # it here instead of silently passing it down to a layout that can't slice.
        assert extra_info is None or extra_info.layer_range is None, "layer_range slicing is not supported on the read path"
        key_strs, buffer_ptrs, buffer_sizes = self._batch_preprocess(keys, host_indices, for_restore=True)
        get_results = self._get_batch_zero_copy_impl(key_strs, buffer_ptrs, buffer_sizes, L3_GET_TIMEOUT_MS)
        return self._batch_postprocess(get_results, is_set_operate=False)

    def batch_set_zero_copy(
        self,
        keys: list[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
        on_complete_callback: Optional[callable] = None,
    ) -> list[bool]:
        pp_prefix = self._resolve_pp_prefix(extra_info)
        if pp_prefix:
            keys = [f"{pp_prefix}{key}" for key in keys]
        if self.extra_backend_tag is not None:
            prefix = self.extra_backend_tag
            keys = [f"{prefix}_{key}" for key in keys]

        layer_range = extra_info.layer_range if extra_info is not None else None
        key_strs, buffer_ptrs, buffer_sizes = self._batch_preprocess(keys, host_indices, layer_range)
        # Backup also performs an existence query before writing. Include the
        # complete L1/L2 prefix so existing L3 copies are renewed. This uses
        # the legacy all-renew behavior rather than ordered-chain stopping:
        # any missing object in this batch is about to be written, making the
        # following objects reachable again.
        lease_prefix_keys = extra_info.prefix_keys if extra_info is not None and extra_info.prefix_keys else []
        # HiCache's PP write path historically prepends a routing marker such
        # as "pp0" to prefix_keys. It is not a page hash; the real Mooncake
        # namespace comes from extra_info.pp_prefix / self.pp_prefix.
        if (
            lease_prefix_keys
            and isinstance(lease_prefix_keys[0], str)
            and lease_prefix_keys[0].startswith("pp")
            and lease_prefix_keys[0][2:].isdigit()
        ):
            lease_prefix_keys = lease_prefix_keys[1:]
        if pp_prefix:
            lease_prefix_keys = [f"{pp_prefix}{key}" for key in lease_prefix_keys]
        if self.extra_backend_tag is not None:
            prefix = self.extra_backend_tag
            lease_prefix_keys = [f"{prefix}_{key}" for key in lease_prefix_keys]
        lease_prefix_key_strs, _ = self._expand_exist_query_keys(lease_prefix_keys)
        prefix_length = len(lease_prefix_key_strs)
        exist_result = self._batch_exist(lease_prefix_key_strs + key_strs)[prefix_length:]

        exist_count = sum(1 for result in exist_result if result == 1)
        logger.debug(
            "Mooncake batch_set_zero_copy start: total_keys=%s existing_keys=%s new_keys=%s sample_keys=%s",
            len(key_strs),
            exist_count,
            len(key_strs) - exist_count,
            key_strs[: min(5, len(key_strs))],
        )

        set_keys = []
        set_buffer_ptrs = []
        set_buffer_sizes = []
        set_indices = []
        set_results = [-1] * len(key_strs)
        skipped_existing_keys = []
        for i in range(len(key_strs)):
            if exist_result[i] != 1:
                set_keys.append(key_strs[i])
                set_buffer_ptrs.append(buffer_ptrs[i])
                set_buffer_sizes.append(buffer_sizes[i])
                set_indices.append(i)
            else:
                set_results[i] = 0
                if len(skipped_existing_keys) < 5:
                    skipped_existing_keys.append(key_strs[i])

        if skipped_existing_keys:
            logger.debug(
                "Mooncake batch_set_zero_copy skip existing keys: count=%s sample=%s",
                exist_count,
                skipped_existing_keys,
            )

        if len(set_keys) > 0:
            logger.debug(
                "Mooncake batch_set_zero_copy put new keys: count=%s sample=%s",
                len(set_keys),
                set_keys[: min(5, len(set_keys))],
            )
            put_results = self._put_batch_zero_copy_impl(set_keys, set_buffer_ptrs, set_buffer_sizes)
            put_success = sum(1 for result in put_results if result == 0)
            put_failures = [(set_keys[i], put_results[i]) for i in range(len(put_results)) if put_results[i] != 0]
            logger.debug(
                "Mooncake batch_set_zero_copy put done: success=%s failed=%s",
                put_success,
                len(put_results) - put_success,
            )
            if put_failures:
                logger.warning(
                    "Mooncake batch_set_zero_copy put failures sample=%s",
                    put_failures[:5],
                )
            for i in range(len(set_indices)):
                set_results[set_indices[i]] = put_results[i]
        else:
            logger.debug("Mooncake batch_set_zero_copy no new keys to put")

        post_results = self._batch_postprocess(set_results, is_set_operate=True)
        logger.info(
            "Mooncake batch_set_zero_copy finish: total_keys=%s success_keys=%s failed_keys=%s",
            len(post_results),
            sum(1 for result in post_results if result),
            sum(1 for result in post_results if not result),
        )

        # Call completion callback once for this batch (mooncake put is synchronous)
        if on_complete_callback is not None:
            on_complete_callback(None)

        return post_results

    def set(
        self,
        key,
        value: Optional[Any] = None,
        target_location: Optional[list[int]] = None,
        target_sizes: Optional[list[int]] = None,
    ) -> bool:
        assert target_location is not None and target_sizes is not None
        if self.is_mla_backend:
            query_keys = [f"{key}_k"]
            target_locations = [target_location]
            target_sizes_list = [target_sizes]
        else:
            query_keys = [f"{key}_{self.local_rank}_k", f"{key}_{self.local_rank}_v"]
            if isinstance(target_location, list) and len(target_location) >= 2:
                target_locations = [target_location[0], target_location[1]]
            else:
                target_locations = [target_location, target_location]

            if isinstance(target_sizes, list) and len(target_sizes) >= 2:
                target_sizes_list = [target_sizes[0], target_sizes[1]]
            else:
                target_sizes_list = [target_sizes, target_sizes]

        exist_result = self._batch_exist(query_keys)
        set_keys = []
        set_target_locations = []
        set_target_sizes = []
        for i in range(len(query_keys)):
            if exist_result[i] != 1:
                set_keys.append(query_keys[i])
                set_target_locations.append(target_locations[i])
                set_target_sizes.append(target_sizes_list[i])

        if len(set_keys) > 0:
            put_result = self._put_batch_zero_copy_impl(set_keys, set_target_locations, set_target_sizes)
            for result in put_result:
                if result != 0:
                    return False
        return True

    def batch_set(
        self,
        keys: list[str],
        values: Optional[list[torch.Tensor]] = None,
        target_locations: Optional[list[int]] = None,
        target_sizes: Optional[list[int]] = None,
    ) -> bool:
        assert target_locations is not None and target_sizes is not None
        assert len(keys) == len(target_locations) == len(target_sizes)

        if len(keys) == 0:
            return False

        for i in range(len(keys)):
            if keys[i] is None or target_locations[i] is None or target_sizes[i] is None:
                return False

        if self.is_mla_backend:
            if keys and keys[0].endswith("_k"):
                query_keys = keys
                expanded_target_locations = target_locations
                expanded_target_sizes = target_sizes
            else:
                query_keys = [f"{key}_k" for key in keys]
                expanded_target_locations = target_locations
                expanded_target_sizes = target_sizes
        else:
            if keys and (keys[0].endswith("_k") or keys[0].endswith("_v")):
                query_keys = keys
                expanded_target_locations = target_locations
                expanded_target_sizes = target_sizes
            else:
                query_keys = []
                expanded_target_locations = []
                expanded_target_sizes = []
                for i, key in enumerate(keys):
                    query_keys.append(f"{key}_{self.local_rank}_k")
                    query_keys.append(f"{key}_{self.local_rank}_v")
                    expanded_target_locations.append(target_locations[i * 2])
                    expanded_target_locations.append(target_locations[i * 2 + 1])
                    expanded_target_sizes.append(target_sizes[i * 2])
                    expanded_target_sizes.append(target_sizes[i * 2 + 1])

        exist_result = self._batch_exist(query_keys)
        set_keys = []
        set_target_locations = []
        set_target_sizes = []
        set_indices = []
        for i in range(len(query_keys)):
            if exist_result[i] != 1:
                set_keys.append(query_keys[i])
                set_target_locations.append(expanded_target_locations[i])
                set_target_sizes.append(expanded_target_sizes[i])
                set_indices.append(i)

        put_result = self._put_batch_zero_copy_impl(set_keys, set_target_locations, set_target_sizes)
        for i in range(len(set_indices)):
            if put_result[i] == 0:
                exist_result[set_indices[i]] = 1

        success_count = 0
        for i in range(len(query_keys)):
            if exist_result[i] == 0:
                break
            success_count += 1
        return success_count == len(query_keys)

    def get(
        self,
        key,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        assert target_location is not None and target_sizes is not None
        if self.is_mla_backend:
            query_keys = [f"{key}_k"]
            target_locations = [target_location]
            target_sizes_list = [target_sizes]
        else:
            query_keys = [f"{key}_{self.local_rank}_k", f"{key}_{self.local_rank}_v"]
            if isinstance(target_location, list) and len(target_location) >= 2:
                target_locations = [target_location[0], target_location[1]]
            else:
                target_locations = [target_location, target_location]

            if isinstance(target_sizes, list) and len(target_sizes) >= 2:
                target_sizes_list = [target_sizes[0], target_sizes[1]]
            else:
                target_sizes_list = [target_sizes, target_sizes]

        get_result = self._get_batch_zero_copy_impl(query_keys, target_locations, target_sizes_list)
        return all(result >= 0 for result in get_result)

    def batch_get(
        self,
        keys: list[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> int:
        assert len(keys) == len(target_locations) == len(target_sizes)
        if len(keys) == 0:
            return 0

        if self.is_mla_backend:
            if keys and keys[0].endswith("_k"):
                query_keys = keys
                key_multiplier = 1
            else:
                query_keys = [f"{key}_k" for key in keys]
                key_multiplier = 1
        else:
            if keys and (keys[0].endswith("_k") or keys[0].endswith("_v")):
                query_keys = keys
                key_multiplier = 2
            else:
                query_keys = []
                for key in keys:
                    query_keys.append(f"{key}_{self.local_rank}_k")
                    query_keys.append(f"{key}_{self.local_rank}_v")
                key_multiplier = 2

        if not self.is_mla_backend and not (keys and (keys[0].endswith("_k") or keys[0].endswith("_v"))):
            expanded_target_locations = []
            expanded_target_sizes = []
            for i in range(len(keys)):
                expanded_target_locations.append(target_locations[i * 2])
                expanded_target_locations.append(target_locations[i * 2 + 1])
                expanded_target_sizes.append(target_sizes[i * 2])
                expanded_target_sizes.append(target_sizes[i * 2 + 1])
            target_locations = expanded_target_locations
            target_sizes = expanded_target_sizes

        get_result = self._get_batch_zero_copy_impl(query_keys, target_locations, target_sizes)

        for i in range(len(query_keys)):
            if get_result[i] < 0:
                return i // key_multiplier
        return len(query_keys) // key_multiplier

    def exists(self, key) -> bool:
        if self.pp_prefix:
            key = f"{self.pp_prefix}{key}"
        if self.is_mla_backend:
            query_keys = [f"{key}_k"]
        else:
            query_keys = [f"{key}_{self.local_rank}_k", f"{key}_{self.local_rank}_v"]
        exist_result = self._batch_exist(query_keys)
        return all(result == 1 for result in exist_result)

    def batch_exists(self, keys, extra_info: Optional[HiCacheStorageExtraInfo] = None) -> int:
        pp_prefix = self._resolve_pp_prefix(extra_info)
        lease_prefix_keys = extra_info.prefix_keys if extra_info is not None and extra_info.prefix_keys else []
        keys = lease_prefix_keys + list(keys)
        logical_prefix_length = len(lease_prefix_keys)
        if pp_prefix:
            keys = [f"{pp_prefix}{key}" for key in keys]
        query_keys, key_multiplier = self._expand_exist_query_keys(keys)

        lease_prefix_length = logical_prefix_length * key_multiplier
        exist_result = self._batch_exist(
            query_keys,
            ordered_chain=True,
            lease_prefix_length=lease_prefix_length,
        )
        exist_result = exist_result[lease_prefix_length:]
        query_keys = query_keys[lease_prefix_length:]

        for i in range(len(query_keys)):
            if exist_result[i] != 1:
                result = i // key_multiplier
                return result
        result = len(query_keys) // key_multiplier
        return result

    def close(self):
        pass

    def clear(self) -> None:
        self.store.remove_all()

    def _put_batch_zero_copy_impl(self, key_strs: list[str], buffer_ptrs: list[int], buffer_sizes: list[int]) -> list[int]:
        if self._uses_multi_buffer(buffer_ptrs):
            if L3_PUT_TIMEOUT_MS > 0:
                return self.store.batch_put_from_multi_buffers(
                    key_strs,
                    buffer_ptrs,
                    buffer_sizes,
                    timeout_ms=L3_PUT_TIMEOUT_MS,
                )
            return self.store.batch_put_from_multi_buffers(key_strs, buffer_ptrs, buffer_sizes)
        if L3_PUT_TIMEOUT_MS > 0:
            return self.store.batch_put_from(
                key_strs,
                buffer_ptrs,
                buffer_sizes,
                timeout_ms=L3_PUT_TIMEOUT_MS,
            )
        return self.store.batch_put_from(key_strs, buffer_ptrs, buffer_sizes)

    def _get_batch_zero_copy_impl(
        self,
        key_strs: list[str],
        buffer_ptrs: list[int],
        buffer_sizes: list[int],
        timeout_ms: int = 0,
    ) -> list[int]:
        if self._uses_multi_buffer(buffer_ptrs):
            if timeout_ms > 0:
                return self.store.batch_get_into_multi_buffers(key_strs, buffer_ptrs, buffer_sizes, False, timeout_ms)
            return self.store.batch_get_into_multi_buffers(key_strs, buffer_ptrs, buffer_sizes)
        if timeout_ms > 0:
            return self.store.batch_get_into(key_strs, buffer_ptrs, buffer_sizes, timeout_ms)
        return self.store.batch_get_into(key_strs, buffer_ptrs, buffer_sizes)

    def _batch_exist(
        self,
        key_strs: list[str],
        ordered_chain: bool = False,
        lease_prefix_length: int = 0,
    ) -> list[int]:
        if ordered_chain or lease_prefix_length:
            results = self.store.batch_is_exist(key_strs, ordered_chain, lease_prefix_length)
        else:
            # Preserve the legacy call shape for ordinary existence checks and
            # rolling upgrades. Only the new policy requires the new binding.
            results = self.store.batch_is_exist(key_strs)
        if len(results) != len(key_strs):
            raise RuntimeError(f"Mooncake batch_is_exist returned an unexpected result count: expected={len(key_strs)}, actual={len(results)}")
        failed = [(key_strs[i], result) for i, result in enumerate(results) if result < 0]
        if failed:
            raise RuntimeError(f"Mooncake batch_is_exist failed: {failed[: min(5, len(failed))]}")
        return results
