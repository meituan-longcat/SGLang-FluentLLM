from __future__ import annotations

import abc

try:
    from typing import override
except ImportError:
    from typing_extensions import override

import torch

from sglang.srt.env import ENV
from sglang.srt.mem_cache.memory_pool.host_memory_pool import (
    HostBufferSpec,
    HostKVBufferLayout,
    HostKVBuffers,
    HostTokenToKVPool,
)
from sglang.srt.mem_cache.memory_pool.memory_pool_mla import MLATokenToKVPool
from sglang.srt.utils import get_colorful_logger, get_device_module, is_npu, transfer_kv_layer_torch_npu

logger = get_colorful_logger(__name__)


def alloc_pinned_buffer(dims, dtype: torch.dtype, pin_memory: bool) -> torch.Tensor:
    buf = torch.empty(dims, dtype=dtype, device="cpu")
    if pin_memory:
        torch.cuda.cudart().cudaHostRegister(buf.data_ptr(), buf.numel() * buf.element_size(), 0)
    return buf


def make_data_ptrs(refs: list[torch.Tensor], device: str) -> torch.Tensor:
    return torch.tensor([x.data_ptr() for x in refs], dtype=torch.uint64, device=device)


class HostMLABufferLayout(HostKVBufferLayout):
    @abc.abstractmethod
    def alloc_buffers(self, spec: HostBufferSpec) -> HostKVBuffers:
        raise NotImplementedError

    @abc.abstractmethod
    def move_indices(
        self,
        pool,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_data_page(self, pool, index: int, flat: bool) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def get_dummy_flat_data_page(self, pool) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def set_from_flat_data_page(self, pool, index: int, data_page: torch.Tensor) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def get_page_buffer_meta(self, pool, indices, layer_range=None):
        raise NotImplementedError

    def d2h_with_offset_npu(
        self,
        pool,
        host_token_indices,
        device_token_indices,
        device_pool,
        host_layer_offset,
        num_layers,
    ):
        raise NotImplementedError("d2h_with_offset_npu not implemented for this layout")

    def d2h_with_offset_gpu(
        self,
        pool,
        host_token_indices,
        device_token_indices,
        device_pool,
        host_layer_offset,
        num_layers,
    ):
        raise NotImplementedError("d2h_with_offset_gpu not implemented for this layout")

    def h2d_layerwise_with_offset_npu(
        self,
        pool,
        host_token_indices,
        device_token_indices,
        device_pool,
        host_layer_id,
        device_layer_id,
    ):
        raise NotImplementedError("h2d_layerwise_with_offset_npu not implemented for this layout")

    def h2d_layerwise_with_offset_gpu(
        self,
        pool,
        host_token_indices,
        device_token_indices,
        device_pool,
        host_layer_id,
        device_layer_id,
    ):
        raise NotImplementedError("h2d_layerwise_with_offset_gpu not implemented for this layout")


class HostMLABufferLayoutLayerFirst(HostMLABufferLayout):
    def __init__(self, io_backend: str):
        self.io_backend = io_backend

    def alloc_buffers(self, spec: HostBufferSpec) -> HostKVBuffers:
        token_stride_size = spec.kv_cache_dim * spec.dtype.itemsize
        layout_dim = token_stride_size * spec.layer_num
        kv_buffer = alloc_pinned_buffer(
            (spec.layer_num, spec.size, 1, spec.kv_cache_dim),
            spec.dtype,
            spec.pin_memory,
        )
        data_refs = [kv_buffer[i] for i in range(spec.layer_num)]
        data_ptrs = make_data_ptrs(data_refs, spec.device)
        return HostKVBuffers(
            kv_buffer=kv_buffer,
            data_refs=data_refs,
            data_ptrs=data_ptrs,
            token_stride_size=token_stride_size,
            layout_dim=layout_dim,
        )

    def move_indices(self, pool, host_indices, device_indices):
        if self.io_backend == "kernel":
            if host_indices.device.type == "cpu":
                host_indices = host_indices.to(pool.device_pool.device, non_blocking=True)
            return host_indices, device_indices
        if self.io_backend == "direct":
            if is_npu():
                return host_indices, device_indices
            device_indices = device_indices.cpu()
            host_indices, idx = host_indices.sort()
            return host_indices, device_indices.index_select(0, idx)
        raise ValueError(f"Unsupported io_backend: {self.io_backend}")

    def get_data_page(self, pool, index, flat):
        data_page = pool.buffers.kv_buffer[:, index : index + pool.page_size, :, :]
        return data_page.flatten() if flat else data_page

    def get_dummy_flat_data_page(self, pool):
        return torch.zeros(
            (
                pool.layer_num,
                pool.page_size,
                1,
                pool.kv_lora_rank + pool.qk_rope_head_dim,
            ),
            dtype=pool.dtype,
            device="cpu",
            pin_memory=pool.pin_memory,
        ).flatten()

    def set_from_flat_data_page(self, pool, index, data_page):
        pool.buffers.kv_buffer[:, index : index + pool.page_size, :, :] = data_page.reshape(
            pool.layer_num,
            pool.page_size,
            1,
            pool.kv_lora_rank + pool.qk_rope_head_dim,
        )

    def get_page_buffer_meta(self, pool, indices, layer_range=None):
        assert len(indices) % pool.page_size == 0
        assert layer_range is None, "layer_range slicing only supported on NPU layer_first"
        base_ptr = pool.buffers.kv_buffer.data_ptr()
        kv_dim = pool.kv_lora_rank + pool.qk_rope_head_dim
        elem_bytes = pool.dtype.itemsize
        indices_list = indices.tolist()
        ptr_list = [
            base_ptr + indices_list[i] * kv_dim * elem_bytes + layer_id * pool.size * kv_dim * elem_bytes
            for i in range(0, len(indices_list), pool.page_size)
            for layer_id in range(pool.layer_num)
        ]
        return ptr_list, [elem_bytes * pool.page_size * kv_dim] * len(ptr_list)

    @override
    def d2h_with_offset_npu(
        self,
        pool,
        host_token_indices,
        device_token_indices,
        device_pool,
        host_layer_offset,
        num_layers,
    ):
        raise NotImplementedError

    @override
    def d2h_with_offset_gpu(self, pool, host_token_indices, device_token_indices, device_pool, host_layer_offset, num_layers):
        raise NotImplementedError("d2h_with_offset_gpu not implemented yet")

    @override
    def h2d_layerwise_with_offset_npu(self, pool, host_token_indices, device_token_indices, device_pool, host_layer_id, device_layer_id):
        raise NotImplementedError

    @override
    def h2d_layerwise_with_offset_gpu(self, pool, host_token_indices, device_token_indices, device_pool, host_layer_id, device_layer_id):
        raise NotImplementedError("h2d_layerwise_with_offset_gpu not implemented yet")


class HostMLABufferLayoutPageFirst(HostMLABufferLayout):
    def __init__(self, io_backend: str):
        self.io_backend = io_backend

    def alloc_buffers(self, spec: HostBufferSpec) -> HostKVBuffers:
        token_stride_size = spec.kv_cache_dim * spec.dtype.itemsize
        layout_dim = token_stride_size * spec.layer_num
        kv_buffer = alloc_pinned_buffer(
            (spec.size, spec.layer_num, 1, spec.kv_lora_rank + spec.qk_rope_head_dim),
            spec.dtype,
            spec.pin_memory,
        )
        return HostKVBuffers(
            kv_buffer=kv_buffer,
            token_stride_size=token_stride_size,
            layout_dim=layout_dim,
        )

    def move_indices(self, pool, host_indices, device_indices):
        if host_indices.device.type == "cpu":
            host_indices = host_indices.to(pool.device_pool.device, non_blocking=True)
        return host_indices, device_indices

    def get_data_page(self, pool, index, flat):
        data_page = pool.buffers.kv_buffer[index : index + pool.page_size, :, :, :]
        return data_page.flatten() if flat else data_page

    def get_dummy_flat_data_page(self, pool):
        return torch.zeros(
            (pool.layer_num, pool.page_size, 1, pool.kv_lora_rank + pool.qk_rope_head_dim),
            dtype=pool.dtype,
            device="cpu",
            pin_memory=pool.pin_memory,
        ).flatten()

    def set_from_flat_data_page(self, pool, index, data_page):
        pool.buffers.kv_buffer[index : index + pool.page_size, :, :, :] = data_page.reshape(
            pool.page_size, pool.layer_num, 1, pool.kv_lora_rank + pool.qk_rope_head_dim
        )

    def get_page_buffer_meta(self, pool, indices, layer_range=None):
        assert len(indices) % pool.page_size == 0
        assert layer_range is None, "layer_range slicing only supported on NPU layer_first"
        base_ptr = pool.buffers.kv_buffer.data_ptr()
        kv_dim = pool.kv_lora_rank + pool.qk_rope_head_dim
        elem_bytes = pool.dtype.itemsize
        indices_list = indices.tolist()
        ptr_list = [base_ptr + indices_list[i] * pool.layer_num * kv_dim * elem_bytes for i in range(0, len(indices_list), pool.page_size)]
        return ptr_list, [pool.layer_num * elem_bytes * pool.page_size * kv_dim] * len(ptr_list)


class HostMLABufferLayoutPageFirstSplitKV(HostMLABufferLayout):
    def __init__(self, io_backend: str):
        self.io_backend = io_backend

    def alloc_buffers(self, spec: HostBufferSpec) -> HostKVBuffers:
        token_stride_size = spec.kv_cache_dim * spec.dtype.itemsize
        layout_dim = token_stride_size * spec.layer_num
        base_dims = (spec.page_num, spec.layer_num, spec.page_size, 1)
        k_buffer = torch.empty((*base_dims, spec.kv_lora_rank), dtype=spec.dtype, device="cpu")
        v_buffer = torch.empty((*base_dims, spec.qk_rope_head_dim), dtype=spec.dtype, device="cpu")
        return HostKVBuffers(
            kv_buffer=None,
            k_buffer=k_buffer,
            v_buffer=v_buffer,
            token_stride_size=token_stride_size,
            layout_dim=layout_dim,
        )

    def move_indices(self, pool, host_indices, device_indices):
        return host_indices, device_indices.cpu()

    def get_data_page(self, pool, index, flat):
        raise NotImplementedError("get_data_page is not supported for page_first+split_kv layout")

    def get_dummy_flat_data_page(self, pool):
        return torch.zeros(
            (pool.layer_num, pool.page_size, 1, pool.kv_lora_rank + pool.qk_rope_head_dim),
            dtype=pool.dtype,
            device="cpu",
            pin_memory=pool.pin_memory,
        ).flatten()

    def set_from_flat_data_page(self, pool, index, data_page):
        raise NotImplementedError("set_from_flat_data_page is not supported for page_first+split_kv layout")

    def get_page_buffer_meta(self, pool, indices, layer_range=None):
        raise NotImplementedError("get_page_buffer_meta is not supported for page_first+split_kv layout")


class HostMLABufferLayoutLayerFirstSplitKV(HostMLABufferLayout):
    """Layer-first split KV layout: k_buffer/v_buffer shape [layer_num, page_num, page_size, 1, dim].
    Matches device pool's 5D buffer layout exactly, enabling direct use of transfer_kv_layer kernel.
    """

    def __init__(self, io_backend: str):
        self.io_backend = io_backend

    def alloc_buffers(self, spec: HostBufferSpec) -> HostKVBuffers:
        token_stride_size = spec.kv_cache_dim * spec.dtype.itemsize
        layout_dim = token_stride_size * spec.layer_num
        # Allocate CPU memory in chunks of 1024 pages per layer to avoid
        # large contiguous allocations and excessive fragmentation.
        # k_buffer/v_buffer: List[List[Tensor]], indexed as [layer][page],
        # each tensor shape: [page_size, 1, dim]
        CHUNK_SIZE = 1024
        k_buffer = []
        v_buffer = []
        k_buffer_chunks = []
        v_buffer_chunks = []
        for _ in range(spec.layer_num):
            layer_k = []
            layer_v = []
            layer_k_chunks = []
            layer_v_chunks = []
            for chunk_start in range(0, spec.page_num, CHUNK_SIZE):
                chunk_len = min(CHUNK_SIZE, spec.page_num - chunk_start)
                chunk_k = torch.empty(
                    (chunk_len, spec.page_size, 1, spec.kv_lora_rank),
                    dtype=spec.dtype,
                    device="cpu",
                )
                chunk_v = torch.empty(
                    (chunk_len, spec.page_size, 1, spec.qk_rope_head_dim),
                    dtype=spec.dtype,
                    device="cpu",
                )
                layer_k_chunks.append(chunk_k)
                layer_v_chunks.append(chunk_v)
                for p in range(chunk_len):
                    layer_k.append(chunk_k[p])
                    layer_v.append(chunk_v[p])
            k_buffer.append(layer_k)
            v_buffer.append(layer_v)
            k_buffer_chunks.append(layer_k_chunks)
            v_buffer_chunks.append(layer_v_chunks)
        get_device_module().current_stream().synchronize()
        logger.info(f"alloc_buffers {get_device_module().current_device()}")
        return HostKVBuffers(
            kv_buffer=None,
            k_buffer=k_buffer,
            v_buffer=v_buffer,
            token_stride_size=token_stride_size,
            layout_dim=layout_dim,
            k_buffer_page_slices=k_buffer,
            v_buffer_page_slices=v_buffer,
            k_buffer_chunks=k_buffer_chunks,
            v_buffer_chunks=v_buffer_chunks,
        )

    def move_indices(self, pool, host_indices, device_indices):
        return host_indices, device_indices

    def get_data_page(self, pool, index, flat):
        page_idx = index // pool.page_size
        # Gather [layer_num, page_size, 1, kv_lora_rank+qk_rope_head_dim]
        k_pages = [pool.buffers.k_buffer[layer_idx][page_idx] for layer_idx in range(pool.layer_num)]
        v_pages = [pool.buffers.v_buffer[layer_idx][page_idx] for layer_idx in range(pool.layer_num)]
        data_page = torch.cat(
            [
                torch.stack(k_pages),
                torch.stack(v_pages),
            ],
            dim=-1,
        )
        return data_page.flatten() if flat else data_page

    def get_dummy_flat_data_page(self, pool):
        return torch.zeros(
            (pool.layer_num, pool.page_size, 1, pool.kv_lora_rank + pool.qk_rope_head_dim),
            dtype=pool.dtype,
            device="cpu",
        ).flatten()

    def set_from_flat_data_page(self, pool, index, data_page):
        page_idx = index // pool.page_size
        reshaped = data_page.reshape(pool.layer_num, pool.page_size, 1, pool.kv_lora_rank + pool.qk_rope_head_dim)
        for layer_idx in range(pool.layer_num):
            pool.buffers.k_buffer[layer_idx][page_idx].copy_(reshaped[layer_idx, :, :, : pool.kv_lora_rank])
            pool.buffers.v_buffer[layer_idx][page_idx].copy_(reshaped[layer_idx, :, :, pool.kv_lora_rank :])

    def get_page_buffer_meta(self, pool, indices, layer_range=None):
        assert len(indices) % pool.page_size == 0
        start_layer, end_layer = layer_range if layer_range else (0, pool.layer_num)
        indices_list = indices.tolist()
        ptr_list = []
        size_list = []
        k_bytes = pool.page_size * pool.kv_lora_rank * pool.dtype.itemsize
        v_bytes = pool.page_size * pool.qk_rope_head_dim * pool.dtype.itemsize
        layer_page_count = len(pool.buffers.k_buffer[0]) if pool.buffers.k_buffer else 0
        for i in range(0, len(indices_list), pool.page_size):
            page_idx = indices_list[i] // pool.page_size
            if page_idx >= layer_page_count:
                logger.error(
                    "[layer_first_splitkv] page_idx=%s exceeds k_buffer length=%s "
                    "(token_idx=%s page_size=%s pool.page_num=%s pool.size=%s) "
                    "indices_sample=%s",
                    page_idx,
                    layer_page_count,
                    indices_list[i],
                    pool.page_size,
                    pool.page_num,
                    pool.size,
                    indices_list[: min(8, len(indices_list))],
                )
                raise IndexError(
                    f"layer_first splitkv page_idx={page_idx} out of range (k_buffer length={layer_page_count}); token_idx={indices_list[i]}"
                )
            for layer_id in range(start_layer, end_layer):
                ptr_list.append(pool.buffers.k_buffer[layer_id][page_idx].data_ptr())
                size_list.append(k_bytes)
                ptr_list.append(pool.buffers.v_buffer[layer_id][page_idx].data_ptr())
                size_list.append(v_bytes)
        return ptr_list, size_list

    @override
    def d2h_with_offset_npu(
        self,
        pool,
        host_token_indices,
        device_token_indices,
        device_pool,
        host_layer_offset,
        num_layers,
    ):
        host_page_indices = host_token_indices
        device_page_indices = device_token_indices

        # device k_buffer/v_buffer: [layer_num, page_num, page_size, 1, dim]
        # host k_buffer/v_buffer: [layer_num, page_num, page_size, 1, dim]
        # Slice layers for partial layer transfer
        device_k = device_pool.k_buffer_page_slices
        device_v = device_pool.v_buffer_page_slices
        host_k = pool.buffers.k_buffer_page_slices[host_layer_offset : host_layer_offset + num_layers]
        host_v = pool.buffers.v_buffer_page_slices[host_layer_offset : host_layer_offset + num_layers]

        transfer_kv_layer_torch_npu(
            device_k,
            host_k,
            device_v,
            host_v,
            None,
            None,
            device_page_indices,
            host_page_indices,
            2,  # direction=2 (D2H)
        )
        return None

    @override
    def d2h_with_offset_gpu(self, pool, host_token_indices, device_token_indices, device_pool, host_layer_offset, num_layers):
        raise NotImplementedError("d2h_with_offset_gpu not implemented for layer_first_split_kv")

    @override
    def h2d_layerwise_with_offset_npu(self, pool, host_token_indices, device_token_indices, device_pool, host_layer_id, device_layer_id):
        host_page_indices = host_token_indices
        device_page_indices = device_token_indices

        # Single layer: use 4D view [page_num, page_size, 1, dim]
        device_k_layer = device_pool.k_buffer_page_slices[device_layer_id]
        device_v_layer = device_pool.v_buffer_page_slices[device_layer_id]
        host_k_layer = pool.buffers.k_buffer_page_slices[host_layer_id]
        host_v_layer = pool.buffers.v_buffer_page_slices[host_layer_id]

        transfer_kv_layer_torch_npu(
            device_k_layer,
            host_k_layer,
            device_v_layer,
            host_v_layer,
            None,
            None,
            device_page_indices,
            host_page_indices,
            1,  # direction=1 (H2D)
        )

    @override
    def h2d_layerwise_with_offset_gpu(self, pool, host_token_indices, device_token_indices, device_pool, host_layer_id, device_layer_id):
        raise NotImplementedError("h2d_layerwise_with_offset_gpu not implemented for layer_first_split_kv")


class HostMLABufferLayoutPageFirstDirect(HostMLABufferLayout):
    def __init__(self, io_backend: str):
        self.io_backend = io_backend

    def alloc_buffers(self, spec: HostBufferSpec) -> HostKVBuffers:
        token_stride_size = spec.kv_cache_dim * spec.dtype.itemsize
        layout_dim = token_stride_size * spec.layer_num
        kv_buffer = alloc_pinned_buffer(
            (
                spec.page_num,
                spec.layer_num,
                spec.page_size,
                1,
                spec.kv_lora_rank + spec.qk_rope_head_dim,
            ),
            spec.dtype,
            spec.pin_memory,
        )
        data_refs = [kv_buffer[i] for i in range(spec.page_num)]
        data_ptrs = make_data_ptrs(data_refs, spec.device)
        return HostKVBuffers(
            kv_buffer=kv_buffer,
            data_refs=data_refs,
            data_ptrs=data_ptrs,
            token_stride_size=token_stride_size,
            layout_dim=layout_dim,
        )

    def move_indices(self, pool, host_indices, device_indices):
        return host_indices, device_indices.cpu()

    def get_data_page(self, pool, index, flat):
        real_index = index // pool.page_size
        data_page = pool.buffers.kv_buffer[real_index : real_index + 1, :, :, :, :]
        return data_page.flatten() if flat else data_page

    def get_dummy_flat_data_page(self, pool):
        return torch.zeros(
            (pool.layer_num, pool.page_size, 1, pool.kv_lora_rank + pool.qk_rope_head_dim),
            dtype=pool.dtype,
            device="cpu",
            pin_memory=pool.pin_memory,
        ).flatten()

    def set_from_flat_data_page(self, pool, index, data_page):
        real_index = index // pool.page_size
        pool.buffers.kv_buffer[real_index : real_index + 1, :, :, :, :] = data_page.reshape(
            1, pool.layer_num, pool.page_size, 1, pool.kv_lora_rank + pool.qk_rope_head_dim
        )

    def get_page_buffer_meta(self, pool, indices, layer_range=None):
        assert len(indices) % pool.page_size == 0
        assert layer_range is None, "layer_range slicing only supported on NPU layer_first"
        base_ptr = pool.buffers.kv_buffer.data_ptr()
        kv_dim = pool.kv_lora_rank + pool.qk_rope_head_dim
        elem_bytes = pool.dtype.itemsize
        indices_list = indices.tolist()
        ptr_list = [base_ptr + indices_list[i] * pool.layer_num * kv_dim * elem_bytes for i in range(0, len(indices_list), pool.page_size)]
        return ptr_list, [pool.layer_num * elem_bytes * pool.page_size * kv_dim] * len(ptr_list)


def make_mla_layout_npu(layout: str, io_backend: str) -> HostMLABufferLayout:
    if layout == "layer_first":
        if io_backend != "direct":
            raise ValueError(f"layer_first on NPU only supports 'direct' io_backend, got: {io_backend}")
        return HostMLABufferLayoutLayerFirstSplitKV(io_backend)
    if layout == "page_first":
        if io_backend != "kernel":
            raise ValueError(f"page_first on NPU only supports 'kernel' io_backend, got: {io_backend}")
        if not ENV.npu_enable_mla_split_kv_kr:
            raise ValueError("page_first on NPU requires npu_enable_mla_split_kv_kr")
        return HostMLABufferLayoutPageFirstSplitKV(io_backend)
    if layout == "page_first_direct":
        if io_backend != "direct":
            raise ValueError(f"page_first_direct on NPU only supports 'direct' io_backend, got: {io_backend}")
        return HostMLABufferLayoutPageFirstDirect(io_backend)
    raise ValueError(f"Unsupported layout on NPU: {layout!r}. Supported: 'layer_first', 'page_first', 'page_first_direct'.")


def make_mla_layout_gpu(layout: str, io_backend: str) -> HostMLABufferLayout:
    if layout == "layer_first":
        if io_backend not in ("kernel", "direct"):
            raise ValueError(f"layer_first on GPU supports 'kernel' or 'direct' io_backend, got: {io_backend}")
        return HostMLABufferLayoutLayerFirst(io_backend)
    if layout == "page_first":
        if io_backend != "kernel":
            raise ValueError(f"page_first on GPU only supports 'kernel' io_backend, got: {io_backend}")
        return HostMLABufferLayoutPageFirst(io_backend)
    if layout == "page_first_direct":
        if io_backend != "direct":
            raise ValueError(f"page_first_direct on GPU only supports 'direct' io_backend, got: {io_backend}")
        return HostMLABufferLayoutPageFirstDirect(io_backend)
    raise ValueError(f"Unsupported layout on GPU: {layout!r}. Supported: 'layer_first', 'page_first', 'page_first_direct'.")


def make_mla_layout(layout: str, io_backend: str) -> HostMLABufferLayout:
    if is_npu():
        return make_mla_layout_npu(layout, io_backend)
    else:
        return make_mla_layout_gpu(layout, io_backend)


class HostMLATokenToKVPool(HostTokenToKVPool):
    device_pool: MLATokenToKVPool
    buf_layout: HostMLABufferLayout

    def __init__(
        self,
        device_pool: MLATokenToKVPool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        io_backend: str = "kernel",
        extra_layer_num: int = 0,
    ):
        self.kv_lora_rank = device_pool.kv_lora_rank
        self.qk_rope_head_dim = device_pool.qk_rope_head_dim
        self.layer_num = device_pool.layer_num + extra_layer_num
        self.kv_cache_dim = device_pool.kv_cache_dim
        self.io_backend = io_backend

        super().__init__(
            device_pool,
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            pin_memory,
        )

    def get_size_per_token(self) -> int:
        return self.kv_cache_dim * self.dtype.itemsize * self.layer_num

    def get_ksize_per_token(self) -> int:
        return self.get_size_per_token()

    def make_buffer_spec(self) -> HostBufferSpec:
        return HostBufferSpec(
            size=self.size,
            page_num=self.page_num,
            page_size=self.page_size,
            layer_num=self.layer_num,
            dtype=self.dtype,
            pin_memory=self.pin_memory,
            device=self.device_pool.device,
            kv_cache_dim=self.kv_cache_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
        )

    def make_buffer_layout(self) -> HostMLABufferLayout:
        return make_mla_layout(self.layout, self.io_backend)

    @override
    def get_data_for_storage(self, indices: torch.Tensor) -> torch.Tensor:
        k_buffer = self.buffers.k_buffer
        if isinstance(k_buffer, list):
            # List[List[Tensor]] format (LayerFirstSplitKV)
            v_buffer = self.buffers.v_buffer
            page_indices = indices[:: self.page_size] // self.page_size
            num_pages = len(page_indices)
            kv_lora_rank = self.kv_lora_rank
            qk_rope_head_dim = self.qk_rope_head_dim
            total_dim = kv_lora_rank + qk_rope_head_dim

            # Pre-allocate output and copy page-by-page, avoiding repeated stack/cat
            # Output shape: [layer_num, num_pages * page_size, 1, total_dim] (token-level)
            out = torch.empty(
                (self.layer_num, num_pages * self.page_size, 1, total_dim),
                dtype=self.dtype,
                device="cpu",
            )
            for layer_idx in range(self.layer_num):
                for i, idx in enumerate(page_indices.tolist()):
                    s = i * self.page_size
                    e = s + self.page_size
                    out[layer_idx, s:e, :, :kv_lora_rank].copy_(k_buffer[layer_idx][idx])
                    out[layer_idx, s:e, :, kv_lora_rank:].copy_(v_buffer[layer_idx][idx])
            return out
        else:
            return super().get_data_for_storage(indices)

    @override
    def set_data_from_storage(self, indices: torch.Tensor, tensor: torch.Tensor) -> None:
        k_buffer = self.buffers.k_buffer
        if isinstance(k_buffer, list):
            # List[List[Tensor]] format (LayerFirstSplitKV)
            v_buffer = self.buffers.v_buffer
            page_indices = indices[:: self.page_size] // self.page_size

            kv_lora_rank = self.kv_lora_rank
            # tensor shape: [layer_num, num_pages * page_size, 1, total_dim] (token-level)
            for layer_idx in range(self.layer_num):
                for i, idx in enumerate(page_indices.tolist()):
                    s = i * self.page_size
                    e = s + self.page_size
                    k_buffer[layer_idx][idx].copy_(tensor[layer_idx, s:e, :, :kv_lora_rank])
                    v_buffer[layer_idx][idx].copy_(tensor[layer_idx, s:e, :, kv_lora_rank:])
        else:
            super().set_data_from_storage(indices, tensor)

    def move_indices(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.buf_layout.move_indices(self, host_indices, device_indices)

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        return self.buf_layout.get_data_page(self, index, flat)

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return self.buf_layout.get_dummy_flat_data_page(self)

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        self.buf_layout.set_from_flat_data_page(self, index, data_page)

    def get_page_buffer_meta(self, indices, layer_range=None):
        return self.buf_layout.get_page_buffer_meta(self, indices, layer_range)

    @override
    def d2h_with_offset_npu(self, host_token_indices, device_token_indices, device_pool, host_layer_offset, num_layers, io_backend):
        return self.buf_layout.d2h_with_offset_npu(
            self,
            host_token_indices,
            device_token_indices,
            device_pool,
            host_layer_offset,
            num_layers,
        )

    @override
    def d2h_with_offset_gpu(self, host_token_indices, device_token_indices, device_pool, host_layer_offset, num_layers, io_backend):
        return self.buf_layout.d2h_with_offset_gpu(
            self,
            host_token_indices,
            device_token_indices,
            device_pool,
            host_layer_offset,
            num_layers,
        )

    @override
    def h2d_layerwise_with_offset_npu(self, host_token_indices, device_token_indices, device_pool, host_layer_id, device_layer_id, io_backend):
        self.buf_layout.h2d_layerwise_with_offset_npu(
            self,
            host_token_indices,
            device_token_indices,
            device_pool,
            host_layer_id,
            device_layer_id,
        )

    @override
    def h2d_layerwise_with_offset_gpu(self, host_token_indices, device_token_indices, device_pool, host_layer_id, device_layer_id, io_backend):
        self.buf_layout.h2d_layerwise_with_offset_gpu(
            self,
            host_token_indices,
            device_token_indices,
            device_pool,
            host_layer_id,
            device_layer_id,
        )
