from __future__ import annotations

try:
    from typing import override
except ImportError:
    from typing_extensions import override

import torch

from sglang.jit_kernel.hicache import can_use_hicache_jit_kernel
from sglang.srt.mem_cache.memory_pool.host_memory_pool import (
    HostBufferSpec,
    HostKVBufferLayout,
    HostKVBuffers,
    HostTokenToKVPool,
    _is_cuda,
)
from sglang.srt.mem_cache.memory_pool.memory_pool_mha import MHATokenToKVPool
from sglang.srt.utils import (
    get_colorful_logger,
    is_npu,
    transfer_kv_layer_torch_npu,
)

__is_npu__ = is_npu()

logger = get_colorful_logger(__name__)


def alloc_mha_pinned_buffer(dims, dtype: torch.dtype, pin_memory: bool) -> torch.Tensor:
    buffer = torch.empty(dims, dtype=dtype, device="cpu")
    if pin_memory:
        torch.cuda.cudart().cudaHostRegister(buffer.data_ptr(), buffer.numel() * buffer.element_size(), 0)
    return buffer


def make_kv_data_ptrs(kv_buffer, layer_num: int, device: str):
    k_data_refs = [kv_buffer[0][i] for i in range(layer_num)]
    v_data_refs = [kv_buffer[1][i] for i in range(layer_num)]
    k_data_ptrs = torch.tensor([x.data_ptr() for x in k_data_refs], dtype=torch.uint64, device=device)
    v_data_ptrs = torch.tensor([x.data_ptr() for x in v_data_refs], dtype=torch.uint64, device=device)
    return k_data_refs, v_data_refs, k_data_ptrs, v_data_ptrs


class HostMHABufferLayoutLayerFirst(HostKVBufferLayout):
    def __init__(self, pin_memory: bool):
        self.pin_memory = pin_memory

    def alloc_buffers(self, spec: HostBufferSpec) -> HostKVBuffers:
        token_stride_size = spec.head_num * spec.head_dim * spec.dtype.itemsize
        layout_dim = token_stride_size * spec.layer_num
        kv_buffer = alloc_mha_pinned_buffer(
            (2, spec.layer_num, spec.size, spec.head_num, spec.head_dim),
            spec.dtype,
            self.pin_memory,
        )
        k_data_refs, v_data_refs, k_data_ptrs, v_data_ptrs = make_kv_data_ptrs(kv_buffer, spec.layer_num, spec.device)
        return HostKVBuffers(
            kv_buffer=kv_buffer,
            token_stride_size=token_stride_size,
            layout_dim=layout_dim,
            k_data_refs=k_data_refs,
            v_data_refs=v_data_refs,
            k_data_ptrs=k_data_ptrs,
            v_data_ptrs=v_data_ptrs,
        )


class HostMHABufferLayoutPageFirst(HostKVBufferLayout):
    def __init__(self, pin_memory: bool):
        self.pin_memory = pin_memory

    def alloc_buffers(self, spec: HostBufferSpec) -> HostKVBuffers:
        token_stride_size = spec.head_num * spec.head_dim * spec.dtype.itemsize
        layout_dim = token_stride_size * spec.layer_num
        kv_buffer = alloc_mha_pinned_buffer(
            (2, spec.size, spec.layer_num, spec.head_num, spec.head_dim),
            spec.dtype,
            self.pin_memory,
        )
        return HostKVBuffers(
            kv_buffer=kv_buffer,
            token_stride_size=token_stride_size,
            layout_dim=layout_dim,
        )


class HostMHABufferLayoutPageFirstDirect(HostKVBufferLayout):
    def __init__(self, pin_memory: bool):
        self.pin_memory = pin_memory

    def alloc_buffers(self, spec: HostBufferSpec) -> HostKVBuffers:
        token_stride_size = spec.head_num * spec.head_dim * spec.dtype.itemsize
        layout_dim = token_stride_size * spec.layer_num
        kv_buffer = alloc_mha_pinned_buffer(
            (2, spec.page_num, spec.layer_num, spec.page_size, spec.head_num, spec.head_dim),
            spec.dtype,
            self.pin_memory,
        )
        return HostKVBuffers(
            kv_buffer=kv_buffer,
            token_stride_size=token_stride_size,
            layout_dim=layout_dim,
        )


class HostMHABufferLayoutPageHead(HostKVBufferLayout):
    def __init__(self, pin_memory: bool):
        self.pin_memory = pin_memory

    def alloc_buffers(self, spec: HostBufferSpec) -> HostKVBuffers:
        token_stride_size = spec.head_num * spec.head_dim * spec.dtype.itemsize
        layout_dim = token_stride_size * spec.layer_num
        kv_buffer = alloc_mha_pinned_buffer(
            (2, spec.page_num, spec.head_num, spec.page_size, spec.layer_num, spec.head_dim),
            spec.dtype,
            self.pin_memory,
        )
        return HostKVBuffers(
            kv_buffer=kv_buffer,
            token_stride_size=token_stride_size,
            layout_dim=layout_dim,
        )


class HostMHABufferLayoutNpuNZ(HostKVBufferLayout):
    """NPU GQA/MHA 的 host 布局, 与 device buffer (NPUMHATokenToKVPool) 逐 page 同形。

    host 只暂存不参与计算, 同形使 D2H/H2D 可直接 _foreach_copy_ 零转换;
    按 chunk 分配避免单块大连续内存分配失败。
    """

    CHUNK_PAGES = 1024

    def __init__(self, pin_memory: bool, nd_layout: bool = False):
        self.pin_memory = pin_memory
        self.nd_layout = nd_layout

    def _page_dims(self, spec: HostBufferSpec):
        if self.nd_layout:
            return (spec.page_size, spec.head_num * spec.head_dim)
        return (spec.head_num * spec.head_dim // 16, spec.page_size, 16)

    def _alloc_page_slices(self, spec: HostBufferSpec):
        page_dims = self._page_dims(spec)
        page_slices, chunks = [], []
        for _ in range(spec.layer_num):
            layer_slices, layer_chunks = [], []
            for chunk_start in range(0, spec.page_num, self.CHUNK_PAGES):
                chunk_len = min(self.CHUNK_PAGES, spec.page_num - chunk_start)
                chunk = torch.empty(
                    (chunk_len, *page_dims),
                    dtype=spec.dtype,
                    device="cpu",
                )
                layer_chunks.append(chunk)
                for p in range(chunk_len):
                    layer_slices.append(chunk[p])
            page_slices.append(layer_slices)
            chunks.append(layer_chunks)
        return page_slices, chunks

    def alloc_buffers(self, spec: HostBufferSpec) -> HostKVBuffers:
        token_stride_size = spec.head_num * spec.head_dim * spec.dtype.itemsize
        k_slices, k_chunks = self._alloc_page_slices(spec)
        v_slices, v_chunks = self._alloc_page_slices(spec)
        return HostKVBuffers(
            kv_buffer=None,
            k_buffer=k_slices,
            v_buffer=v_slices,
            token_stride_size=token_stride_size,
            layout_dim=token_stride_size * spec.layer_num,
            k_buffer_page_slices=k_slices,
            v_buffer_page_slices=v_slices,
            k_buffer_chunks=k_chunks,
            v_buffer_chunks=v_chunks,
        )


def make_mha_buffer_layout(layout: str, pin_memory: bool, nd_layout: bool = False) -> HostKVBufferLayout:
    if __is_npu__:
        return HostMHABufferLayoutNpuNZ(pin_memory, nd_layout=nd_layout)
    if layout == "layer_first":
        return HostMHABufferLayoutLayerFirst(pin_memory)
    if layout == "page_first":
        return HostMHABufferLayoutPageFirst(pin_memory)
    if layout == "page_first_direct":
        return HostMHABufferLayoutPageFirstDirect(pin_memory)
    if layout == "page_head":
        return HostMHABufferLayoutPageHead(pin_memory)
    raise ValueError(f"Unsupported MHA layout: {layout!r}")


class HostMHATokenToKVPool(HostTokenToKVPool):
    device_pool: MHATokenToKVPool

    def __init__(
        self,
        device_pool: MHATokenToKVPool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        io_backend: str = "",
        extra_layer_num: int = 0,
        host_size_tokens: int = 0,
    ):
        self.head_num = device_pool.head_num
        self.head_dim = device_pool.head_dim
        # extra_layer_num: 与 draft 共用同一 host pool 时按层拼接 (要求同构)。
        self.layer_num = device_pool.layer_num + extra_layer_num

        super().__init__(
            device_pool,
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            pin_memory,
            host_size_tokens=host_size_tokens,
        )
        self.element_dim = self.device_pool.head_num * self.device_pool.head_dim
        self.can_use_jit = _is_cuda and can_use_hicache_jit_kernel(element_size=self.element_dim * self.dtype.itemsize)

    def get_size_per_token(self):
        return self.head_dim * self.head_num * self.layer_num * self.dtype.itemsize * 2

    def get_ksize_per_token(self):
        return self.get_size_per_token() // 2

    def make_buffer_spec(self) -> HostBufferSpec:
        return HostBufferSpec(
            size=self.size,
            page_num=self.page_num,
            page_size=self.page_size,
            layer_num=self.layer_num,
            dtype=self.dtype,
            pin_memory=self.pin_memory,
            device=self.device_pool.device,
            head_num=self.head_num,
            head_dim=self.head_dim,
        )

    def make_buffer_layout(self) -> HostKVBufferLayout:
        return make_mha_buffer_layout(
            self.layout,
            self.pin_memory,
            nd_layout=getattr(self.device_pool, "use_nd_layout", False),
        )

    @override
    def d2h_with_offset_npu(
        self,
        host_token_indices,
        device_token_indices,
        device_pool,
        host_layer_offset,
        num_layers,
        io_backend,
    ):
        # NPU 走 page 粒度: host/device 均为 NZ 同形, 索引即 page 号。
        host_k = self.buffers.k_buffer_page_slices[host_layer_offset : host_layer_offset + num_layers]
        host_v = self.buffers.v_buffer_page_slices[host_layer_offset : host_layer_offset + num_layers]
        transfer_kv_layer_torch_npu(
            device_pool.k_buffer_page_slices,
            host_k,
            device_pool.v_buffer_page_slices,
            host_v,
            None,
            None,
            device_token_indices,
            host_token_indices,
            2,  # D2H
        )
        return None

    @override
    def h2d_layerwise_with_offset_npu(
        self,
        host_token_indices,
        device_token_indices,
        device_pool,
        host_layer_id,
        device_layer_id,
        io_backend,
    ):
        transfer_kv_layer_torch_npu(
            device_pool.k_buffer_page_slices[device_layer_id],
            self.buffers.k_buffer_page_slices[host_layer_id],
            device_pool.v_buffer_page_slices[device_layer_id],
            self.buffers.v_buffer_page_slices[host_layer_id],
            None,
            None,
            device_token_indices,
            host_token_indices,
            1,  # H2D
        )

    @override
    def d2h_with_offset_gpu(
        self,
        host_token_indices,
        device_token_indices,
        device_pool,
        host_layer_offset,
        num_layers,
        io_backend,
    ):
        raise NotImplementedError("GPU MHA host pool 未实现 page_slices 传输; NPU NZ layout 走 d2h_with_offset_npu")

    @override
    def h2d_layerwise_with_offset_gpu(
        self,
        host_token_indices,
        device_token_indices,
        device_pool,
        host_layer_id,
        device_layer_id,
        io_backend,
    ):
        raise NotImplementedError("GPU MHA host pool 未实现 page_slices 传输; NPU NZ layout 走 h2d_layerwise_with_offset_npu")

    @override
    def get_data_for_storage(self, indices: torch.Tensor) -> torch.Tensor:
        if __is_npu__:
            # NPU NZ layout 无整块 kv_buffer (按 page slice 分块), L3 storage 接口未适配。
            raise NotImplementedError("L3 storage is not supported for the NPU NZ MHA host layout")
        kv_buf = self.buffers.kv_buffer
        if self.layout == "layer_first":
            return kv_buf[:, :, indices].contiguous()
        elif self.layout == "page_first":
            return kv_buf[:, indices].contiguous()
        else:
            raise ValueError(f"get_data_for_storage not supported for MHA layout: {self.layout}")

    @override
    def set_data_from_storage(self, indices: torch.Tensor, tensor: torch.Tensor) -> None:
        if __is_npu__:
            raise NotImplementedError("L3 storage is not supported for the NPU NZ MHA host layout")
        kv_buf = self.buffers.kv_buffer
        if self.layout == "layer_first":
            kv_buf[:, :, indices] = tensor
        elif self.layout == "page_first":
            kv_buf[:, indices] = tensor
        else:
            raise ValueError(f"set_data_from_storage not supported for MHA layout: {self.layout}")

    @property
    def k_buffer(self):
        return self.buffers.kv_buffer[0]

    @property
    def v_buffer(self):
        return self.buffers.kv_buffer[1]

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        if self.layout == "layer_first":
            data_page = self.buffers.kv_buffer[:, :, index : index + self.page_size, :, :]
        elif self.layout == "page_first":
            data_page = self.buffers.kv_buffer[:, index : index + self.page_size, :, :, :]
        elif self.layout in ["page_first_direct", "page_head"]:
            real_index = index // self.page_size
            data_page = self.buffers.kv_buffer[:, real_index : real_index + 1, :, :, :, :]
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        if flat:
            data_page = data_page.flatten()
        return data_page

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(
            (2, self.layer_num, self.page_size, self.head_num, self.head_dim),
            dtype=self.dtype,
            device="cpu",
            pin_memory=self.pin_memory,
        ).flatten()

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        if self.layout == "layer_first":
            self.buffers.kv_buffer[:, :, index : index + self.page_size, :, :] = data_page.reshape(
                2,
                self.layer_num,
                self.page_size,
                self.head_num,
                self.head_dim,
            )
        elif self.layout == "page_first":
            self.buffers.kv_buffer[:, index : index + self.page_size, :, :, :] = data_page.reshape(
                2, self.page_size, self.layer_num, self.head_num, self.head_dim
            )
        elif self.layout == "page_first_direct":
            real_index = index // self.page_size
            self.buffers.kv_buffer[:, real_index : real_index + 1, :, :, :, :] = data_page.reshape(
                2, 1, self.layer_num, self.page_size, self.head_num, self.head_dim
            )
        elif self.layout == "page_head":
            real_index = index // self.page_size
            self.buffers.kv_buffer[:, real_index : real_index + 1, :, :, :, :] = data_page.reshape(
                2, 1, self.head_num, self.page_size, self.layer_num, self.head_dim
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

    def get_page_buffer_meta(self, indices):
        assert len(indices) % self.page_size == 0
        ptr_list = []
        kv_buffer_data_ptr = self.buffers.kv_buffer.data_ptr()
        indices = indices.tolist()
        v_offset = self.layer_num * self.size * self.head_num * self.head_dim * self.dtype.itemsize

        if self.layout == "layer_first":
            for index in range(0, len(indices), self.page_size):
                for layer_id in range(self.layer_num):
                    k_ptr = (
                        kv_buffer_data_ptr
                        + indices[index] * self.head_num * self.head_dim * self.dtype.itemsize
                        + layer_id * self.size * self.head_num * self.head_dim * self.dtype.itemsize
                    )
                    v_ptr = k_ptr + v_offset
                    ptr_list.append(k_ptr)
                    ptr_list.append(v_ptr)
            element_size = self.dtype.itemsize * self.page_size * self.head_num * self.head_dim
            element_size_list = [element_size] * len(ptr_list)
        elif self.layout in ["page_first", "page_first_direct", "page_head"]:
            for index in range(0, len(indices), self.page_size):
                k_ptr = kv_buffer_data_ptr + indices[index] * self.layer_num * self.head_num * self.head_dim * self.dtype.itemsize
                v_ptr = k_ptr + v_offset
                ptr_list.append(k_ptr)
                ptr_list.append(v_ptr)
            element_size = self.layer_num * self.dtype.itemsize * self.page_size * self.head_num * self.head_dim
            element_size_list = [element_size] * len(ptr_list)
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        return ptr_list, element_size_list
