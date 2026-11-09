import os
import tempfile
from typing import Any, Optional

import torch

from sglang.srt.mem_cache.l3.storage import HiCacheStorage, HiCacheStorageConfig
from sglang.srt.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


class SimpleFile(HiCacheStorage):
    def __init__(self, storage_config: HiCacheStorageConfig, file_path: Optional[str] = None):
        if file_path is None:
            file_path = os.path.join(tempfile.gettempdir(), "hicache")
        self.file_path = os.getenv("SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR", file_path)

        tp_rank, tp_size, model_name, is_mla_model = (
            storage_config.tp_rank,
            storage_config.tp_size,
            storage_config.model_name,
            storage_config.is_mla_model,
        )
        model_name = "-".join(model_name.split("/")) if model_name else ""
        if is_mla_model:
            self.config_suffix = f"_{model_name}"
        else:
            self.config_suffix = f"_{model_name}_{tp_rank}_{tp_size}"

        if not os.path.exists(self.file_path) and tp_rank == 0:
            os.makedirs(self.file_path)
            logger.info(f"Created SimpleFile storage directory at {self.file_path}")

    def _get_suffixed_key(self, key: str) -> str:
        return key + self.config_suffix

    def get(
        self,
        key: str,
        target_location: torch.Tensor,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        key = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        try:
            expected = target_location.numel() * target_location.element_size()
            with open(tensor_path, "rb", buffering=0) as f:
                buf = memoryview(target_location.view(torch.uint8).contiguous().numpy())
                if f.readinto(buf) != expected:
                    raise OSError(f"Short read for {key}")
            return target_location
        except FileNotFoundError:
            logger.warning(f"Failed to fetch {key} from SimpleFile storage.")
            return None

    def batch_get(
        self,
        keys: list[str],
        target_locations: list[torch.Tensor],
        target_sizes: Optional[Any] = None,
    ) -> list[torch.Tensor | None]:
        return [self.get(key, target_location) for key, target_location in zip(keys, target_locations or [None] * len(keys))]

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        if self.exists(key):
            logger.debug(f"Key {key} already exists. Skipped.")
            return True

        key = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        try:
            value.contiguous().view(dtype=torch.uint8).numpy().tofile(tensor_path)
            return True
        except Exception as e:
            logger.error(f"Failed to save tensor {key}: {e}")
            return False

    def batch_set(
        self,
        keys: list[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        return all(self.set(key, value) for key, value in zip(keys, values))

    def exists(self, key: str) -> bool:
        key = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        return os.path.exists(tensor_path)

    def clear(self) -> bool:
        try:
            for filename in os.listdir(self.file_path):
                file_path = os.path.join(self.file_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            logger.info("Cleared all entries in SimpleFile storage.")
            return True
        except Exception as e:
            logger.error(f"Failed to clear SimpleFile storage: {e}")
            return False
