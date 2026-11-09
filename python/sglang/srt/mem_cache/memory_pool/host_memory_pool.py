from __future__ import annotations

import abc
from dataclasses import dataclass, field
from os import environ

import psutil
import torch

from sglang.srt.distributed import get_world_group
from sglang.srt.mem_cache.memory_pool.host_kv_allocator import HostKVAllocator
from sglang.srt.mem_cache.memory_pool.memory_pool_base import BaseTokenToKVPool
from sglang.srt.utils import KVPageViews, get_colorful_logger, is_cuda, is_npu, is_xpu

logger = get_colorful_logger(__name__)

_is_cuda = is_cuda()
_is_npu = is_npu()
_is_xpu = is_xpu()

if _is_npu or _is_xpu:
    transfer_kv_direct = None
    transfer_kv_all_layer = None
    transfer_kv_all_layer_direct_lf_pf = None
    transfer_kv_all_layer_lf_pf = None
    transfer_kv_all_layer_lf_ph = None
    transfer_kv_all_layer_mla = None
    transfer_kv_all_layer_mla_lf_pf = None
    transfer_kv_per_layer = None
    transfer_kv_per_layer_direct_pf_lf = None
    transfer_kv_per_layer_mla = None
    transfer_kv_per_layer_mla_pf_lf = None
    transfer_kv_per_layer_pf_lf = None
    transfer_kv_per_layer_ph_lf = None
else:
    from flashinfer import (
        transfer_kv_direct,
    )

TransferDirection = None
transfer_kv_dim_exchange = None
try:
    if _is_npu:
        from sgl_kernel_npu.kvcacheio import (  # noqa: F401
            TransferDirection,
            transfer_kv_dim_exchange,
        )
except ImportError as e:
    import logging

    logging.getLogger(__name__).warning(f"Failed to import TransferDirection from sgl_kernel_npu.kvcacheio: {e}")


def transfer_kv_direct_torch(
    src_layers,
    dst_layers,
    src_indices,
    dst_indices,
    page_size,
):
    for src, dst in zip(src_layers, dst_layers):
        gathered = src[src_indices.to(src.device)]
        dst[dst_indices.to(dst.device)] = gathered.to(dst.device, non_blocking=True)


if _is_npu or _is_xpu:
    transfer_kv_direct = transfer_kv_direct_torch

SUPPORT_PIN_MEMORY = not _is_npu


@dataclass
class RankLocalCpuCopy:
    page2rank: list[int]
    local_page_mask: torch.Tensor
    kv_local: list[list[list[torch.Tensor]]] = field(default_factory=list)


@dataclass
class HostKVBuffers:
    kv_buffer: torch.Tensor | None
    k_buffer: torch.Tensor | list[list[torch.Tensor]] | None = None
    v_buffer: torch.Tensor | list[list[torch.Tensor]] | None = None
    index_k_buffer: torch.Tensor | None = None
    data_refs: list[torch.Tensor] | None = None
    data_ptrs: torch.Tensor | None = None
    token_stride_size: int = 0
    layout_dim: int = 0
    k_data_refs: list[torch.Tensor] | None = None
    v_data_refs: list[torch.Tensor] | None = None
    k_data_ptrs: torch.Tensor | None = None
    v_data_ptrs: torch.Tensor | None = None
    k_buffer_page_slices: list[list[torch.Tensor]] | None = None
    v_buffer_page_slices: list[list[torch.Tensor]] | None = None
    index_k_buffer_page_slices: list[list[torch.Tensor]] | None = None
    k_buffer_chunks: list[list[torch.Tensor]] | None = None
    v_buffer_chunks: list[list[torch.Tensor]] | None = None
    index_k_buffer_chunks: list[list[torch.Tensor]] | None = None
    # Packed inactive IndexK's zero-copy PUT source. GET uses the separate
    # one-page placeholder as a write-only discard sink, never this zero page.
    index_k_storage_zero_chunks: list[list[torch.Tensor]] | None = None


@dataclass
class HostBufferSpec:
    size: int
    page_num: int
    page_size: int
    layer_num: int
    dtype: torch.dtype
    pin_memory: bool
    device: str
    kv_cache_dim: int = 0
    kv_lora_rank: int = 0
    qk_rope_head_dim: int = 0
    head_num: int = 0
    head_dim: int = 0
    index_head_dim: int = 0
    index_dtype: torch.dtype | None = None
    dsa_kv_int8_packed: bool = False


class HostKVBufferLayout(abc.ABC):
    @abc.abstractmethod
    def alloc_buffers(self, buf_spec: HostBufferSpec) -> HostKVBuffers:
        raise NotImplementedError


class HostTokenToKVPool(abc.ABC):
    def __init__(
        self,
        device_pool: BaseTokenToKVPool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool,
        host_size_tokens: int = 0,
    ):
        self.device_pool = device_pool
        self.page_size = page_size
        self.layout = layout
        self.pin_memory = pin_memory and SUPPORT_PIN_MEMORY

        self.dtype = device_pool.store_dtype
        self.size_per_token = self.get_size_per_token()
        self.fixed_size_bytes = self.get_fixed_size_bytes()
        if host_size_tokens > 0:
            # 直接指定 token 数: 异构 draft 的 host pool 须与 target 的页号域严格对齐,
            # 不能各自按 GB 反算 (每 token 字节数不同会算出不同的 page_num)。
            self.size = host_size_tokens
        elif host_size > 0:
            if host_size * 1e9 <= self.fixed_size_bytes:
                raise ValueError("Host budget cannot hold fixed cache allocations")
            self.size = int((host_size * 1e9 - self.fixed_size_bytes) // self.size_per_token)
        else:
            self.size = int(device_pool.size * host_to_device_ratio)
            if self.fixed_size_bytes:
                host_budget_bytes = self.size * self.size_per_token
                if host_budget_bytes <= self.fixed_size_bytes:
                    raise ValueError("Host budget cannot hold fixed cache allocations")
                self.size = int((host_budget_bytes - self.fixed_size_bytes) // self.size_per_token)
        if get_world_group().world_size > 1 and host_size_tokens <= 0:
            import torch.distributed as dist

            size_tensor = torch.tensor([self.size], dtype=torch.int64, device="cpu")
            dist.all_reduce(size_tensor, op=dist.ReduceOp.MIN, group=get_world_group().cpu_group)
            self.size = int(size_tensor.item())
        # Debug: limit L2 KV cache size via environment variable (in number of tokens)
        min_l2_kvsize = int(environ.get("MIN_L2_KVSIZE", "0"))
        if min_l2_kvsize > 0 and host_size_tokens <= 0:
            self.size = min(self.size, min_l2_kvsize)
            logger.info(f"MIN_L2_KVSIZE={min_l2_kvsize}, limiting host pool size to {self.size} tokens")
        self.page_num = self.size // self.page_size
        self.size = self.page_num * self.page_size

        self.enable_l15 = getattr(device_pool, "enable_mla_l1_5_cache", False)
        attn_tp_group_size = getattr(device_pool, "attn_tp_group_size", 1)
        self.global_page_num = self.page_num * attn_tp_group_size if self.enable_l15 else self.page_num
        self.group_mask = getattr(device_pool, "group_mask", 0)
        self.group_shift = getattr(device_pool, "group_shift", 0)
        self.kvp_rank = getattr(device_pool, "kvp_rank", 0)

        self.host_kv_allocator = HostKVAllocator(
            global_page_num=self.global_page_num,
            group_mask=self.group_mask,
        )

        logger.info(
            f"Host memory pool size: {self.size=} page_num={self.page_num} "
            f"global_page_num={self.global_page_num} size_per_token={self.size_per_token} "
            f"total={(self.size * self.size_per_token + self.fixed_size_bytes) / 1e9:.2f} GB "
            f"device_pool.size={device_pool.size}"
        )
        if self.size < device_pool.size:
            raise ValueError(f"Host memory pool size {self.size} is less than device pool size {device_pool.size}")

        self.sanity_check(host_size, host_to_device_ratio)

        buf_spec = self.make_buffer_spec()
        self.buf_layout: HostKVBufferLayout = self.make_buffer_layout()
        self.buffers: HostKVBuffers = self.buf_layout.alloc_buffers(buf_spec)

        from sglang.srt.env import global_server_args_dict

        if _is_npu and global_server_args_dict.get("disaggregation_mode") == "prefill":
            # Pool page layouts are fixed; build run boundaries outside the
            # request path instead of rescanning every page on each copy.
            for owner in (self.buffers, device_pool):
                for name in ("k_buffer_page_slices", "v_buffer_page_slices", "index_k_buffer_page_slices"):
                    layers = getattr(owner, name, None)
                    if isinstance(layers, (list, tuple)):
                        setattr(owner, name, [KVPageViews(layer) for layer in layers])

    def sanity_check(self, host_size: int, host_to_device_ratio: float) -> None:
        host_mem = psutil.virtual_memory()
        requested_bytes = self.size * self.size_per_token + self.fixed_size_bytes
        ten_gb = 10 * (1024**3)
        available_bytes = host_mem.available - ten_gb
        if requested_bytes > available_bytes:
            raise ValueError(
                f"Not enough host memory available. Requesting "
                f"{requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free. Please reduce the "
                f"size of the hierarchical cache."
            )
        else:
            logger.info(
                f"Allocating {requested_bytes / 1e9:.2f} GB host memory for hierarchical KV cache. "
                f"{host_size=} {self.size_per_token=} {host_to_device_ratio=} "
                f"{self.device_pool.size=} {host_mem.available=} "
                f"{self.page_num=} {self.global_page_num=}"
            )

    def get_fixed_size_bytes(self) -> int:
        return 0

    @abc.abstractmethod
    def get_size_per_token(self) -> int:
        raise NotImplementedError()

    @abc.abstractmethod
    def make_buffer_spec(self) -> HostBufferSpec:
        raise NotImplementedError()

    @abc.abstractmethod
    def make_buffer_layout(self) -> HostKVBufferLayout:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_dummy_flat_data_page(self) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        raise NotImplementedError()

    def get_data_for_storage(self, indices: torch.Tensor) -> torch.Tensor:
        kv_buf = self.buffers.kv_buffer
        if self.layout == "page_first":
            return kv_buf[indices].contiguous().clone()
        else:
            return kv_buf[:, indices].contiguous().clone()

    def set_data_from_storage(self, indices: torch.Tensor, tensor: torch.Tensor) -> None:
        kv_buf = self.buffers.kv_buffer
        if self.layout == "page_first":
            kv_buf[indices] = tensor
        else:
            kv_buf[:, indices] = tensor

    def get_storage_size_per_token(self) -> int:
        return self.get_size_per_token()

    def mirror_alloc(
        self,
        device_page_indices: torch.Tensor,
    ):
        return self.host_kv_allocator.alloc_pages(len(device_page_indices))

    def get_ranks_from_global_pages(self, global_host_indices: torch.Tensor) -> list:
        return [0 for _ in range(len(global_host_indices))]

    def pages_to_tokens(self, page_indices: torch.Tensor) -> torch.Tensor:
        return (page_indices.unsqueeze(1) * self.page_size + torch.arange(self.page_size, device=page_indices.device)).flatten()

    def tokens_to_pages(self, token_indices: torch.Tensor) -> torch.Tensor:
        return token_indices // self.page_size

    def clear(self):
        self.host_kv_allocator.clear()

    def d2h_with_offset(
        self,
        host_token_indices,
        device_token_indices,
        device_pool,
        host_layer_offset,
        num_layers,
        io_backend,
    ):
        if _is_npu:
            return self.d2h_with_offset_npu(
                host_token_indices,
                device_token_indices,
                device_pool,
                host_layer_offset,
                num_layers,
                io_backend,
            )
        else:
            return self.d2h_with_offset_gpu(
                host_token_indices,
                device_token_indices,
                device_pool,
                host_layer_offset,
                num_layers,
                io_backend,
            )

    @abc.abstractmethod
    def d2h_with_offset_npu(
        self,
        host_token_indices,
        device_token_indices,
        device_pool,
        host_layer_offset,
        num_layers,
        io_backend,
    ):
        raise NotImplementedError()

    @abc.abstractmethod
    def d2h_with_offset_gpu(
        self,
        host_token_indices,
        device_token_indices,
        device_pool,
        host_layer_offset,
        num_layers,
        io_backend,
    ):
        raise NotImplementedError()

    def h2d_layerwise_with_offset(
        self,
        host_token_indices,
        device_token_indices,
        device_pool,
        host_layer_id,
        device_layer_id,
        io_backend,
    ):
        if _is_npu:
            self.h2d_layerwise_with_offset_npu(
                host_token_indices,
                device_token_indices,
                device_pool,
                host_layer_id,
                device_layer_id,
                io_backend,
            )
        else:
            self.h2d_layerwise_with_offset_gpu(
                host_token_indices,
                device_token_indices,
                device_pool,
                host_layer_id,
                device_layer_id,
                io_backend,
            )

    @abc.abstractmethod
    def h2d_layerwise_with_offset_npu(
        self,
        host_token_indices,
        device_token_indices,
        device_pool,
        host_layer_id,
        device_layer_id,
        io_backend,
    ):
        raise NotImplementedError()

    @abc.abstractmethod
    def h2d_layerwise_with_offset_gpu(
        self,
        host_token_indices,
        device_token_indices,
        device_pool,
        host_layer_id,
        device_layer_id,
        io_backend,
    ):
        raise NotImplementedError()
