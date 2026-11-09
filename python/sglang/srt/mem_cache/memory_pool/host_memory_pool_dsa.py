from __future__ import annotations

try:
    from typing import override
except ImportError:
    from typing_extensions import override

import torch

from sglang.srt.mem_cache.memory_pool.host_memory_pool import (
    HostBufferSpec,
    HostKVBufferLayout,
    HostKVBuffers,
)
from sglang.srt.mem_cache.memory_pool.host_memory_pool_mla import (
    HostMLABufferLayout,
    HostMLABufferLayoutLayerFirstSplitKV,
    HostMLATokenToKVPool,
)
from sglang.srt.mem_cache.memory_pool.memory_pool_dsa import DSATokenToKVPool
from sglang.srt.utils import get_colorful_logger, is_npu, transfer_kv_layer_torch_npu

__is_npu__ = is_npu()
logger = get_colorful_logger(__name__)


class HostDSABufferLayout(HostKVBufferLayout):
    def __init__(self, base_layout: HostMLABufferLayout):
        self.base_layout = base_layout

    def alloc_buffers(self, spec: HostBufferSpec) -> HostKVBuffers:
        logger.info(f"HostDSABufferLayout.alloc_buffers: spec={spec}")
        assert spec.index_head_dim > 0
        if __is_npu__:
            assert isinstance(self.base_layout, HostMLABufferLayoutLayerFirstSplitKV)

        if spec.dsa_kv_int8_packed:
            # DSA KV int8 stores the device KV cache as one packed buffer per
            # layer. Mirror that on host instead of allocating split K/V
            # buffers, otherwise D2H sees device_v=None with a non-None host_v.
            buffers = self._alloc_packed_kv_buffers(spec)
        else:
            buffers = self.base_layout.alloc_buffers(spec)

        index_k_buffer_page_slices = None
        index_k_buffer_chunks = None
        index_k_storage_zero_chunks = [] if spec.dsa_kv_int8_packed else None
        index_dtype = spec.index_dtype or spec.dtype
        layer_mask = getattr(spec, "indexer_layer_mask", (True,) * spec.layer_num)
        if isinstance(self.base_layout, HostMLABufferLayoutLayerFirstSplitKV):
            # 5D layout: [layer_num, page_num, page_size, 1, index_head_dim]
            # Allocate in chunks of 4096 pages per layer to avoid large contiguous allocations.
            CHUNK_SIZE = 4096
            index_k_buffer_page_slices = []
            index_k_buffer_chunks = []
            for layer_idx in range(spec.layer_num):
                layer_pages = spec.page_num if layer_mask[layer_idx] else 1
                layer_slices = []
                layer_chunks = []
                for chunk_start in range(0, layer_pages, CHUNK_SIZE):
                    chunk_len = min(CHUNK_SIZE, layer_pages - chunk_start)
                    chunk = torch.empty(
                        (chunk_len, spec.page_size, 1, spec.index_head_dim),
                        dtype=index_dtype,
                        device="cpu",
                    )
                    if not layer_mask[layer_idx]:
                        chunk.zero_()
                    layer_chunks.append(chunk)
                    for p in range(chunk_len):
                        layer_slices.append(chunk[p])
                index_k_buffer_page_slices.append(layer_slices)
                index_k_buffer_chunks.append(layer_chunks)
                if index_k_storage_zero_chunks is not None:
                    # A persisted old/full-layout object may contain arbitrary
                    # inactive IndexK bytes. GET discards them into the ordinary
                    # placeholder; PUT reads this separate immutable zero page.
                    index_k_storage_zero_chunks.append([] if layer_mask[layer_idx] else [torch.zeros_like(layer_chunks[0])])
            index_k_buffer = index_k_buffer_page_slices
        elif buffers.k_buffer is not None:
            base_dims = (spec.page_num, spec.layer_num, spec.page_size, 1)
            index_k_buffer = torch.empty(
                (*base_dims, spec.index_head_dim),
                dtype=index_dtype,
                device="cpu",
            )
        else:
            index_k_buffer = torch.empty(
                (spec.layer_num, spec.size, 1, spec.index_head_dim),
                dtype=index_dtype,
                device="cpu",
            )
        buffers.index_k_buffer = index_k_buffer
        buffers.index_k_buffer_page_slices = index_k_buffer_page_slices
        buffers.index_k_buffer_chunks = index_k_buffer_chunks
        buffers.index_k_storage_zero_chunks = index_k_storage_zero_chunks
        return buffers

    def _alloc_packed_kv_buffers(self, spec: HostBufferSpec) -> HostKVBuffers:
        token_stride_size = spec.kv_cache_dim * torch.int8.itemsize
        layout_dim = token_stride_size * spec.layer_num
        chunk_size = 1024
        k_buffer = []
        k_buffer_chunks = []
        for _ in range(spec.layer_num):
            layer_pages = []
            layer_chunks = []
            for chunk_start in range(0, spec.page_num, chunk_size):
                chunk_len = min(chunk_size, spec.page_num - chunk_start)
                chunk = torch.empty(
                    (chunk_len, spec.page_size, 1, spec.kv_cache_dim),
                    dtype=torch.int8,
                    device="cpu",
                )
                layer_chunks.append(chunk)
                for page_idx in range(chunk_len):
                    layer_pages.append(chunk[page_idx])
            k_buffer.append(layer_pages)
            k_buffer_chunks.append(layer_chunks)
        return HostKVBuffers(
            kv_buffer=None,
            k_buffer=k_buffer,
            v_buffer=None,
            token_stride_size=token_stride_size,
            layout_dim=layout_dim,
            k_buffer_page_slices=k_buffer,
            v_buffer_page_slices=None,
            k_buffer_chunks=k_buffer_chunks,
        )

    def move_indices(self, pool, host_indices, device_indices):
        return self.base_layout.move_indices(pool, host_indices, device_indices)

    def get_data_page(self, pool, index: int, flat: bool):
        if getattr(pool.device_pool, "dsa_kv_int8_packed", False):
            start = int(index) // pool.page_size * pool.page_size
            page = pool.get_data_for_storage(torch.arange(start, start + pool.page_size, device="cpu"))
            return page.flatten() if flat else page
        return self.base_layout.get_data_page(pool, index, flat)

    def get_dummy_flat_data_page(self, pool):
        if getattr(pool.device_pool, "dsa_kv_int8_packed", False):
            return torch.zeros(pool.page_size * pool.get_storage_size_per_token(), dtype=torch.int8, device="cpu")
        return self.base_layout.get_dummy_flat_data_page(pool)

    def set_from_flat_data_page(self, pool, index: int, data_page: torch.Tensor) -> None:
        if getattr(pool.device_pool, "dsa_kv_int8_packed", False):
            start = int(index) // pool.page_size * pool.page_size
            pool.set_data_from_storage(torch.arange(start, start + pool.page_size, device="cpu"), data_page)
            return
        self.base_layout.set_from_flat_data_page(pool, index, data_page)

    def get_page_buffer_meta(self, pool, indices, layer_range=None, *, for_restore=False):
        if getattr(pool.device_pool, "dsa_kv_int8_packed", False):
            return self._get_packed_page_buffer_meta(pool, indices, layer_range, for_restore=for_restore)

        ptr_list, size_list = self.base_layout.get_page_buffer_meta(pool, indices, layer_range)
        if pool.buffers.index_k_buffer_page_slices is None:
            return ptr_list, size_list

        start_layer, end_layer = layer_range if layer_range else (0, pool.layer_num)
        sliced_layer_num = end_layer - start_layer

        assert len(indices) % pool.page_size == 0
        indices_list = indices.tolist()
        grouped_ptrs = []
        grouped_sizes = []
        kv_buffer_count = 1 if getattr(pool.device_pool, "dsa_kv_int8_packed", False) else 2
        stride = sliced_layer_num * kv_buffer_count
        index_bytes = pool.page_size * pool.device_pool.index_head_dim * pool.device_pool.index_k_with_scale_buffer_dtype.itemsize
        # Logical pool capacity; inactive IndexK layers use physical page 0 below.
        layer_page_count = pool.page_num

        for page_idx, token_idx in enumerate(range(0, len(indices_list), pool.page_size)):
            base = page_idx * stride
            host_page_idx = indices_list[token_idx] // pool.page_size
            if host_page_idx >= layer_page_count:
                logger.error(
                    "[DSA-sidecar] host_page_idx=%s exceeds index_k_buffer length=%s (token_idx=%s page_size=%s pool.page_num=%s)",
                    host_page_idx,
                    layer_page_count,
                    indices_list[token_idx],
                    pool.page_size,
                    pool.page_num,
                )
                raise IndexError(
                    f"DSA index_k_buffer host_page_idx={host_page_idx} out of range (length={layer_page_count}); token_idx={indices_list[token_idx]}"
                )
            for offset, layer_id in enumerate(range(start_layer, end_layer)):
                kv_base = base + offset * kv_buffer_count
                grouped_ptrs.append(ptr_list[kv_base])
                grouped_sizes.append(size_list[kv_base])
                if kv_buffer_count == 2:
                    grouped_ptrs.append(ptr_list[kv_base + 1])
                    grouped_sizes.append(size_list[kv_base + 1])
                grouped_ptrs.append(
                    pool.buffers.index_k_buffer_page_slices[layer_id][host_page_idx if pool.indexer_layer_mask[layer_id] else 0].data_ptr()
                )
                grouped_sizes.append(index_bytes)

        return grouped_ptrs, grouped_sizes

    @staticmethod
    def _packed_page_indices(pool, indices):
        values = indices.tolist()
        if indices.ndim != 1 or len(values) % pool.page_size:
            raise ValueError("Packed Host I/O requires complete one-dimensional pages")
        pages = []
        for start in range(0, len(values), pool.page_size):
            token = values[start]
            if token < 0 or token % pool.page_size or values[start : start + pool.page_size] != list(range(token, token + pool.page_size)):
                raise ValueError("Packed Host I/O requires aligned contiguous tokens in each page")
            page = token // pool.page_size
            if page >= pool.page_num:
                raise IndexError(f"Packed Host page {page} outside logical page count {pool.page_num}")
            pages.append(page)
        return pages

    def _get_packed_page_buffer_meta(self, pool, indices, layer_range=None, *, for_restore=False):
        if pool.buffers.index_k_buffer_page_slices is None:
            raise RuntimeError("DSA packed KV requires index_k_buffer_page_slices.")

        start_layer, end_layer = layer_range if layer_range else (0, pool.layer_num)
        if not 0 <= start_layer <= end_layer <= pool.layer_num:
            raise ValueError("Packed Host I/O layer range is outside the logical pool")
        grouped_ptrs = []
        grouped_sizes = []
        index_bytes = pool.page_size * pool.device_pool.index_head_dim * pool.device_pool.index_k_with_scale_buffer_dtype.itemsize
        for host_page_idx in self._packed_page_indices(pool, indices):
            for layer_id in range(start_layer, end_layer):
                kv_page = pool.buffers.k_buffer_page_slices[layer_id][host_page_idx]
                grouped_ptrs.append(kv_page.data_ptr())
                grouped_sizes.append(kv_page.nbytes)
                if pool.indexer_layer_mask[layer_id]:
                    index_page = pool.buffers.index_k_buffer_page_slices[layer_id][host_page_idx]
                elif for_restore:
                    # Write-only sink, with pool lifetime. Its bytes are never
                    # consumed by Indexer, restored to device or serialized.
                    index_page = pool.buffers.index_k_buffer_page_slices[layer_id][0]
                else:
                    index_page = pool.buffers.index_k_storage_zero_chunks[layer_id][0][0]
                if index_page.nbytes != index_bytes:
                    raise RuntimeError("Packed Host IndexK descriptor must cover exactly one physical page")
                grouped_ptrs.append(index_page.data_ptr())
                grouped_sizes.append(index_bytes)

        return grouped_ptrs, grouped_sizes


class HostDSATokenToKVPool(HostMLATokenToKVPool):
    device_pool: DSATokenToKVPool

    @property
    def storage_format_id(self):
        if not self.device_pool.dsa_kv_int8_packed:
            return None
        index_bytes = self.device_pool.index_head_dim * self.device_pool.index_k_with_scale_buffer_dtype.itemsize
        return f"dsa-c8-raw-v1-kv{self.kv_cache_dim}-ik{index_bytes}-p{self.page_size}"

    @property
    def indexer_layer_mask(self):
        target = tuple(getattr(self.device_pool, "indexer_layer_mask", (True,) * self.device_pool.layer_num))
        return target + (True,) * (self.layer_num - len(target))

    def get_fixed_size_bytes(self):
        inactive = len(self.indexer_layer_mask) - sum(self.indexer_layer_mask)
        # Packed inactive IndexK owns a zero PUT source and a distinct GET sink.
        pages = 2 if self.device_pool.dsa_kv_int8_packed else 1
        return pages * inactive * self.page_size * self.device_pool.index_head_dim * self.device_pool.index_k_with_scale_buffer_dtype.itemsize

    def get_size_per_token(self):
        base = super().get_size_per_token()
        assert self.device_pool.index_head_dim is not None
        index_bytes = self.device_pool.index_head_dim * self.device_pool.index_k_with_scale_buffer_dtype.itemsize * sum(self.indexer_layer_mask)
        return base + index_bytes

    def make_buffer_spec(self) -> HostBufferSpec:
        spec = super().make_buffer_spec()
        spec.index_head_dim = self.device_pool.index_head_dim
        spec.index_dtype = self.device_pool.index_k_with_scale_buffer_dtype
        spec.dsa_kv_int8_packed = self.device_pool.dsa_kv_int8_packed
        spec.indexer_layer_mask = self.indexer_layer_mask
        return spec

    def make_buffer_layout(self) -> HostKVBufferLayout:
        base_layout = super().make_buffer_layout()
        return HostDSABufferLayout(base_layout)

    def get_page_buffer_meta_for_restore(self, indices, layer_range=None):
        return self.buf_layout.get_page_buffer_meta(self, indices, layer_range, for_restore=True)

    def _packed_storage_geometry(self, indices):
        pages = HostDSABufferLayout._packed_page_indices(self, indices)
        index_bytes = self.device_pool.index_head_dim * self.device_pool.index_k_with_scale_buffer_dtype.itemsize
        return pages, self.kv_cache_dim + index_bytes

    def _get_packed_data_for_storage(self, indices):
        pages, row_bytes = self._packed_storage_geometry(indices)
        # Byte storage: one whole packed KV page followed by raw BF16 IndexK
        # bytes for each logical layer/page, matching zero-copy descriptors.
        out = torch.empty((self.layer_num, len(pages), self.page_size * row_bytes), dtype=torch.int8, device="cpu")
        kv_bytes = self.page_size * self.kv_cache_dim
        for layer in range(self.layer_num):
            for output_page, host_page in enumerate(pages):
                out[layer, output_page, :kv_bytes].copy_(self.buffers.k_buffer[layer][host_page].view(torch.int8).reshape(-1))
                tail = out[layer, output_page, kv_bytes:]
                if self.indexer_layer_mask[layer]:
                    tail.copy_(self.buffers.index_k_buffer[layer][host_page].view(torch.int8).reshape(-1))
                else:
                    tail.zero_()
        return out.view(self.layer_num, len(pages) * self.page_size, 1, row_bytes)

    def _set_packed_data_from_storage(self, indices, tensor):
        pages, row_bytes = self._packed_storage_geometry(indices)
        expected = self.layer_num * len(pages) * self.page_size * row_bytes
        if tensor.dtype != torch.int8 or tensor.numel() != expected:
            raise ValueError("Packed Host storage requires the complete raw-byte KV/IndexK payload")
        data = tensor.reshape(self.layer_num, len(pages), self.page_size * row_bytes)
        kv_bytes = self.page_size * self.kv_cache_dim
        for layer in range(self.layer_num):
            for input_page, host_page in enumerate(pages):
                self.buffers.k_buffer[layer][host_page].view(torch.int8).reshape(-1).copy_(data[layer, input_page, :kv_bytes])
                if self.indexer_layer_mask[layer]:
                    self.buffers.index_k_buffer[layer][host_page].view(torch.int8).reshape(-1).copy_(data[layer, input_page, kv_bytes:])

    @override
    def get_data_for_storage(self, indices: torch.Tensor) -> torch.Tensor:
        if self.device_pool.dsa_kv_int8_packed:
            return self._get_packed_data_for_storage(indices)
        ik_buf = self.buffers.index_k_buffer
        if isinstance(ik_buf, list):
            # List[List[Tensor]] format (LayerFirstSplitKV)
            k_buffer = self.buffers.k_buffer
            v_buffer = self.buffers.v_buffer
            page_indices = indices[:: self.page_size] // self.page_size
            num_pages = len(page_indices)
            index_head_dim = self.device_pool.index_head_dim
            kv_dim = self.kv_cache_dim
            total_dim = kv_dim + index_head_dim

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
                    if v_buffer is None:
                        out[layer_idx, s:e, :, :kv_dim].copy_(k_buffer[layer_idx][idx])
                    else:
                        out[layer_idx, s:e, :, : self.kv_lora_rank].copy_(k_buffer[layer_idx][idx])
                        out[
                            layer_idx,
                            s:e,
                            :,
                            self.kv_lora_rank : self.kv_lora_rank + self.qk_rope_head_dim,
                        ].copy_(v_buffer[layer_idx][idx])
                    if self.indexer_layer_mask[layer_idx]:
                        out[layer_idx, s:e, :, kv_dim:].copy_(ik_buf[layer_idx][idx])
                    else:
                        out[layer_idx, s:e, :, kv_dim:].zero_()
            return out
        else:
            # Tensor format (page_first / layer_first non-split)
            data = super().get_data_for_storage(indices)
            ik_data = ik_buf[:, indices]
            return torch.cat([data, ik_data], dim=-1).contiguous()

    @override
    def set_data_from_storage(self, indices: torch.Tensor, tensor: torch.Tensor) -> None:
        if self.device_pool.dsa_kv_int8_packed:
            self._set_packed_data_from_storage(indices, tensor)
            return
        ik_buf = self.buffers.index_k_buffer
        if isinstance(ik_buf, list):
            # List[List[Tensor]] format (LayerFirstSplitKV)
            k_buffer = self.buffers.k_buffer
            v_buffer = self.buffers.v_buffer
            page_indices = indices[:: self.page_size] // self.page_size

            kv_dim = self.kv_cache_dim
            # tensor shape: [layer_num, num_pages * page_size, 1, total_dim] (token-level)
            for layer_idx in range(self.layer_num):
                for i, idx in enumerate(page_indices.tolist()):
                    s = i * self.page_size
                    e = s + self.page_size
                    if v_buffer is None:
                        k_buffer[layer_idx][idx].copy_(tensor[layer_idx, s:e, :, :kv_dim])
                    else:
                        k_buffer[layer_idx][idx].copy_(tensor[layer_idx, s:e, :, : self.kv_lora_rank])
                        v_buffer[layer_idx][idx].copy_(
                            tensor[
                                layer_idx,
                                s:e,
                                :,
                                self.kv_lora_rank : self.kv_lora_rank + self.qk_rope_head_dim,
                            ]
                        )
                    if self.indexer_layer_mask[layer_idx]:
                        ik_buf[layer_idx][idx].copy_(tensor[layer_idx, s:e, :, kv_dim:])
        else:
            # Tensor format (page_first / layer_first non-split)
            kv_dim = self.buffers.kv_buffer.shape[-1]
            super().set_data_from_storage(indices, tensor[..., :kv_dim])
            ik_buf[:, indices] = tensor[..., kv_dim:]

    @override
    def get_storage_size_per_token(self) -> int:
        # L3 retains one IndexK slot for every logical layer.
        return (
            super().get_size_per_token()
            + self.layer_num * self.device_pool.index_head_dim * self.device_pool.index_k_with_scale_buffer_dtype.itemsize
        )

    def _check_indexer_transfer_layout(self, device_pool, host_layer, device_layer, layer_count):
        device_mask = tuple(getattr(device_pool, "indexer_layer_mask", (True,) * device_pool.layer_num))
        if (
            layer_count < 0
            or host_layer < 0
            or device_layer < 0
            or host_layer + layer_count > self.layer_num
            or device_layer + layer_count > len(device_mask)
            or self.indexer_layer_mask[host_layer : host_layer + layer_count] != device_mask[device_layer : device_layer + layer_count]
        ):
            raise ValueError("Host/device IndexK logical layer layout mismatch")

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
        self._check_indexer_transfer_layout(device_pool, host_layer_offset, 0, num_layers)
        # todo fix it npu
        host_page_indices = host_token_indices
        device_page_indices = device_token_indices

        device_k = device_pool.k_buffer_page_slices
        device_v = device_pool.v_buffer_page_slices
        device_ik = device_pool.index_k_buffer_page_slices
        host_k = self.buffers.k_buffer_page_slices[host_layer_offset : host_layer_offset + num_layers]
        host_v = (
            None
            if self.buffers.v_buffer_page_slices is None
            else self.buffers.v_buffer_page_slices[host_layer_offset : host_layer_offset + num_layers]
        )
        host_ik = self.buffers.index_k_buffer_page_slices[host_layer_offset : host_layer_offset + num_layers]

        # logger.info(
        #     f"d2h_with_offset_npu: device_page_indices={device_page_indices}, host_page_indices={host_page_indices}, "
        #     f"device_pages_num={len(device_k[0])}, host_pages_num={len(host_k[0])}, "
        #     f"max_device_idx={device_page_indices.max().item()}, max_host_idx={host_page_indices.max().item()}, "
        #     f"page_size={page_size}, host_layer_offset={host_layer_offset}, num_layers={num_layers}"
        # )

        transfer_kv_layer_torch_npu(
            device_k,
            host_k,
            device_v,
            host_v,
            device_ik,
            host_ik,
            device_page_indices,
            host_page_indices,
            2,  # D2H
            indexer_layer_mask=getattr(device_pool, "indexer_layer_mask", None),
        )
        return None

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
        # host_kv: (size, layer_count, 1, kv_cache_dim) — layer axis is dim 1
        # host_ik: (layer_count, size, 1, index_head_dim) — layer axis is dim 0
        host_kv = self.buffers.kv_buffer
        host_ik = self.buffers.index_k_buffer
        dev_kv_list = device_pool.kv_buffer
        dev_ik_list = device_pool.index_k_with_scale_buffer

        hti = host_token_indices.to(torch.int64).cpu()
        dti = device_token_indices.to(torch.int64).to(device_pool.device, non_blocking=True)

        total_entries = min(len(dev_kv_list), num_layers, host_kv.shape[1] - host_layer_offset)

        for layer_idx in range(total_entries):
            hl = host_layer_offset + layer_idx
            host_kv[hti, hl, 0, :].copy_(dev_kv_list[layer_idx][dti, :].squeeze(1))

        if host_ik is not None:
            host_ik_2d = host_ik.squeeze(2) if host_ik.dim() >= 3 else host_ik
            for layer_idx in range(total_entries):
                hl = host_layer_offset + layer_idx
                host_ik_2d[hl, hti, :].copy_(
                    dev_ik_list[layer_idx].reshape(-1, self.page_size, host_ik.shape[-1])[dti // self.page_size, dti % self.page_size, :]
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
        self._check_indexer_transfer_layout(device_pool, host_layer_id, device_layer_id, 1)
        # todo fix it npu
        host_page_indices = host_token_indices
        device_page_indices = device_token_indices

        # 4D: [page_num, page_size, 1, dim]
        device_k_layer = device_pool.k_buffer_page_slices[device_layer_id]
        device_v_layer = None if device_pool.v_buffer_page_slices is None else device_pool.v_buffer_page_slices[device_layer_id]
        device_ik_layer = device_pool.index_k_with_scale_buffer[device_layer_id]
        host_k_layer = self.buffers.k_buffer_page_slices[host_layer_id]
        host_v_layer = None if self.buffers.v_buffer_page_slices is None else self.buffers.v_buffer_page_slices[host_layer_id]
        host_ik_layer = self.buffers.index_k_buffer_page_slices[host_layer_id]
        if not self.indexer_layer_mask[host_layer_id]:
            device_ik_layer = host_ik_layer = None

        transfer_kv_layer_torch_npu(
            device_k_layer,
            host_k_layer,
            device_v_layer,
            host_v_layer,
            device_ik_layer,
            host_ik_layer,
            device_page_indices,
            host_page_indices,
            1,  # H2D
        )

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
        # host_kv: (size, layer_count, 1, kv_cache_dim) — layer axis is dim 1
        # host_ik: (layer_count, size, 1, index_head_dim) — layer axis is dim 0
        host_kv = self.buffers.kv_buffer
        host_ik = self.buffers.index_k_buffer
        dev_kv_list = device_pool.kv_buffer

        hti = host_token_indices.to(torch.int64).cpu()
        dti = device_token_indices.to(torch.int64).to(device_pool.device, non_blocking=True)

        dev_kv_list[device_layer_id][dti, :].copy_(host_kv[hti, host_layer_id, 0, :].unsqueeze(1))

        if host_ik is not None:
            host_ik_2d = host_ik.squeeze(2) if host_ik.dim() >= 3 else host_ik
            dev_ik_buf = device_pool.index_k_with_scale_buffer[device_layer_id]
            dev_ik_buf.reshape(-1, self.page_size, host_ik.shape[-1])[dti // self.page_size, dti % self.page_size, :].copy_(
                host_ik_2d[host_layer_id, hti, :]
            )
        return None
