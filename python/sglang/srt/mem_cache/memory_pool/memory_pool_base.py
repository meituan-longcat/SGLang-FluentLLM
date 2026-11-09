from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from functools import wraps
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import torch

from sglang.srt.disaggregation.utils import PageTransferMetadata
from sglang.srt.env import global_server_args_dict
from sglang.srt.layers.attention.radix_attention import RadixAttention
from sglang.srt.utils import get_colorful_logger

if TYPE_CHECKING:
    from sglang.srt.mem_cache.hicache_operation import LayerDoneCounter

logger = get_colorful_logger(__name__)

GB = 1024 * 1024 * 1024


@dataclass
class ReqToTokenPoolInfo:
    verified_len: int
    alloced_len: int
    alloced_slots: torch.Tensor


class BaseTokenToKVPool:
    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        device: str,
        max_batch_size: int,
        max_context_len: int,
        page_size: int,
        rank: int,
        start_layer: int | None = None,
        end_layer: int | None = None,
    ):
        self.dtype = dtype
        self.rank = rank
        self.size = size
        self.page_size = page_size
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype
        self.device = device
        self.offload_chunk_page_num = 1024
        self.token_slot_refs = None

        self.max_batch_size = max_batch_size
        self.layer_transfer_counter = None
        self.start_layer = start_layer
        self.end_layer = end_layer
        logger.info(
            f"Initialized token to kv pool with size {size}, dtype {dtype}, device {device}, "
            f"page size {page_size}, rank {rank} {self.start_layer=} {self.end_layer=}"
        )

    def register_layer_transfer_counter(self, layer_transfer_counter: LayerDoneCounter):
        if str(self.device).split(":", 1)[0] == "npu" and global_server_args_dict.get("disaggregation_mode") == "decode":
            return
        self.layer_transfer_counter = layer_transfer_counter

    def wait_hicache_load(self, layer_id: int):
        layer_id = layer_id - self.start_layer
        if getattr(self, "layer_transfer_counter", None) is not None:
            self.layer_transfer_counter.wait_until(layer_id)

    def set_token_slot_refs(self, token_slot_refs: torch.Tensor | None):
        self.token_slot_refs = token_slot_refs

    @property
    def is_mla_backend(self) -> bool:
        return False

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def get_kv_buffer(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> None:
        raise NotImplementedError()

    def get_cpu_copy(self, page_indices: list) -> torch.Tensor:
        raise NotImplementedError()

    def load_cpu_copy(
        self,
        kv_cache_cpu: torch.Tensor,
        page_indices: list,
        prefix_page_count: int = 0,
    ) -> None:
        raise NotImplementedError()

    def get_contiguous_buf_infos(self):
        raise NotImplementedError()

    def get_layerwise_buf_info_offsets(self, start_idx=0):
        raise NotImplementedError()

    def get_page_transfer_metadata(self, page_indices: npt.NDArray[np.int64] | torch.Tensor) -> PageTransferMetadata:
        if isinstance(page_indices, torch.Tensor):
            page_indices = page_indices.detach().cpu().numpy()
        page_indices = np.asarray(page_indices, dtype=np.int64)
        return PageTransferMetadata(
            indices_are_local=False,
            page_transfer_mask=np.ones(page_indices.shape[0], dtype=np.bool_),
            page_local_indices=page_indices,
        )

    def select_local_pages_from_global_pages(self, global_page_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = torch.ones(len(global_page_indices), dtype=torch.bool, device=global_page_indices.device)
        return mask, global_page_indices

    def get_ranks_from_global_pages(self, global_page_indices: torch.Tensor) -> list:
        return [0] * len(global_page_indices)

    def pages_to_tokens(self, page_indices: torch.Tensor) -> torch.Tensor:
        return (page_indices.unsqueeze(1) * self.page_size + torch.arange(self.page_size, device=page_indices.device)).flatten()


class MemoryStateInt(IntEnum):
    IDLE = 0
    RESERVED = 1
    PROTECTED = 2
    SYNCED = 3
    BACKUP = 4


def synchronized(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with self.lock:
            return func(self, *args, **kwargs)

    return wrapper
