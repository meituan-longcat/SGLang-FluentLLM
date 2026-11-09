from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

from sglang.srt.mem_cache.memory_pool.host_memory_pool import HostTokenToKVPool
from sglang.srt.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


@dataclass
class HiCacheStorageConfig:
    tp_rank: int
    tp_size: int
    is_mla_model: bool
    is_page_first_layout: bool
    model_name: Optional[str]
    pp_rank: int = 0
    pp_size: int = 1
    extra_config: Optional[dict] = None
    enable_l15: bool = False


@dataclass
class HiCacheStorageExtraInfo:
    prefix_keys: Optional[list[str]] = None
    extra_info: Optional[dict] = None
    pp_prefix: Optional[str] = None
    layer_range: Optional[tuple[int, int]] = None


class HiCacheStorage(ABC):
    def register_mem_pool_host(self, mem_pool_host: HostTokenToKVPool):
        self.mem_pool_host = mem_pool_host

    def batch_get_zero_copy(
        self,
        keys: list[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> list[bool]:
        pass

    def batch_set_zero_copy(
        self,
        keys: list[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
        on_complete_callback: Optional[Callable] = None,
    ) -> list[bool]:
        pass

    @abstractmethod
    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        pass

    @abstractmethod
    def batch_get(
        self,
        keys: list[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> list[torch.Tensor | None] | int:
        pass

    @abstractmethod
    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        pass

    @abstractmethod
    def batch_set(
        self,
        keys: list[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        pass

    def batch_exists(self, keys: list[str], extra_info: Optional[HiCacheStorageExtraInfo] = None) -> int:
        for i in range(len(keys)):
            if not self.exists(keys[i]):
                return i
        return len(keys)

    def batch_exists_mask(self, keys: list[str], kvp_rank: int, extra_info: Optional[HiCacheStorageExtraInfo] = None) -> list[int]:
        return [kvp_rank if self.exists(k) else -1 for k in keys]

    def clear(self) -> None:
        pass

    def get_stats(self):
        return None
