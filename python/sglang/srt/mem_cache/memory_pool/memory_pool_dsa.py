from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.env import ENV
from sglang.srt.layers.attention.dsa import index_buf_accessor
from sglang.srt.mem_cache.dsa_indexer_layout import (
    allocate_indexer_pages,
    normalize_indexer_mask,
)
from sglang.srt.mem_cache.memory_pool.host_memory_pool import RankLocalCpuCopy
from sglang.srt.mem_cache.memory_pool.memory_pool_kernel import get_tensor_size_bytes
from sglang.srt.mem_cache.memory_pool.memory_pool_mla import (
    MLABufferLayout,
    MLABuffers,
    MLABufferSpec,
    MLATokenToKVPool,
    make_mla_buffer_layout,
)
from sglang.srt.utils import get_colorful_logger, get_page_slices_by_layer_and_page_idx, is_npu

if TYPE_CHECKING:
    # SparseCoordinator pulls in torch_npu. It appears here only as an
    # annotation, and this module is on the generic memory-pool import chain,
    # so loading it eagerly would break any non-NPU environment at import.
    from sglang.srt.mem_cache.kv_offload.sparse_coordinator import SparseCoordinator


logger = get_colorful_logger(__name__)
__is_npu__ = is_npu()

DSA_PACKED_KV_ROW_ALIGNMENT = 32


def get_dsa_packed_kv_cache_dim(
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    rope_dtype: torch.dtype,
    tile_size: int,
    packed_row_bytes: int | None = None,
) -> int:
    scale_num = (kv_lora_rank + tile_size - 1) // tile_size
    rope_bytes = qk_rope_head_dim * torch._utils._element_size(rope_dtype)
    scale_bytes = scale_num * torch._utils._element_size(torch.float32)
    payload_bytes = kv_lora_rank + rope_bytes + scale_bytes
    if packed_row_bytes is None:
        return payload_bytes

    aligned_row_bytes = (payload_bytes + DSA_PACKED_KV_ROW_ALIGNMENT - 1) // DSA_PACKED_KV_ROW_ALIGNMENT * DSA_PACKED_KV_ROW_ALIGNMENT
    if packed_row_bytes not in (payload_bytes, aligned_row_bytes):
        raise ValueError(
            "Packed DSA KV row bytes must equal the payload width or its "
            f"{DSA_PACKED_KV_ROW_ALIGNMENT}-byte aligned width; got "
            f"{packed_row_bytes}, expected {payload_bytes} or {aligned_row_bytes}."
        )
    return packed_row_bytes


@dataclass
class L15DSACpuCopy(RankLocalCpuCopy):
    index_local: list[list[list[torch.Tensor]]] = field(default_factory=list)


class DsaInt8PackedKVLayout(MLABufferLayout):
    requires_npu_page_copy = True
    allow_public_kv_setters = False

    def __init__(self, packed_row_bytes: int):
        self.packed_row_bytes = packed_row_bytes

    def raise_unsupported_error(self, func_name: str) -> None:
        raise RuntimeError(
            f"{func_name} must not write NPU DSA KV int8 cache. Packed C8 cache is written by torch.ops.custom.npu_mla_prolog_v3 mode 9 only."
        )

    def get_kv_cache_dim(
        self,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        model_dtype: torch.dtype,
    ) -> int:
        return get_dsa_packed_kv_cache_dim(
            kv_lora_rank,
            qk_rope_head_dim,
            model_dtype,
            DSATokenToKVPool.quant_block_size,
            self.packed_row_bytes,
        )

    def get_page_num(self, pool) -> int:
        page_num = (pool.size + pool.page_size) // pool.page_size
        page_bytes = pool.page_size * pool.kv_cache_dim
        alignment = 4096
        if (page_num * page_bytes) % alignment == 0:
            return page_num

        original_page_num = page_num
        while (page_num * page_bytes) % alignment != 0:
            page_num += 1
        logger.info(
            "Padding DSA KV int8 packed page_num from %s to %s for %s-byte registration alignment (page_bytes=%s).",
            original_page_num,
            page_num,
            alignment,
            page_bytes,
        )
        return page_num

    def get_page_size_bytes(self, pool) -> int:
        return pool.page_size * pool.layer_num * pool.kv_cache_dim

    def alloc_buffers(self, spec: MLABufferSpec) -> MLABuffers:
        from sglang.srt.mem_cache.kv_offload.budget import is_offloaded_attention

        kv_buffer = []
        buffer_device_types = []
        for local_layer in range(spec.layer_num):
            host = (
                spec.offload_cache and spec.offload_cache_storage == "host"
                and is_offloaded_attention(spec.start_layer + local_layer)
            )
            allocator = torch.zeros
            if host:
                import torch_npu

                allocator = torch_npu.empty_with_swapped_memory
            kv_buffer.append(allocator(
                (spec.page_num, spec.page_size, 1, spec.kv_cache_dim),
                dtype=torch.int8, device=spec.device,
            ).contiguous())
            if host:
                # Reserved dummy page, also read by graph padding requests.
                kv_buffer[-1][:1].zero_()
            buffer_device_types.append("cpu" if host else "npu")
        if "cpu" in buffer_device_types:
            logger.info(
                "Packed INT8 host backing: row_bytes=%s host_layers=%s host_bytes=%s route=cold_index_select",
                spec.kv_cache_dim, buffer_device_types.count("cpu"),
                sum(buf.nbytes for buf, kind in zip(kv_buffer, buffer_device_types) if kind == "cpu"),
            )
        return MLABuffers(
            kv_buffer=kv_buffer,
            buffer_device_types=buffer_device_types if "cpu" in buffer_device_types else None,
            k_buffer_page_slices=[[buf[i] for i in range(spec.page_num)] for buf in kv_buffer],
            v_buffer_page_slices=None,
            empty_kr_cache=torch.empty((0,), dtype=torch.int8, device=spec.device),
        )

    def get_log_buffer_info(self, pool):
        first_buffer = pool.kv_buffer[0]
        return first_buffer.device, first_buffer.shape

    def move_kv_cache_eager(self, pool, tgt_loc_flat, src_loc_flat) -> None:
        for buf in pool.kv_buffer:
            flat = buf.view(-1, 1, pool.kv_cache_dim)
            flat[tgt_loc_flat] = flat[src_loc_flat]

    def get_contiguous_buf_infos(self, pool):
        kv_data_ptrs = [pool.kv_buffer[i].data_ptr() for i in range(pool.layer_num)]
        kv_data_lens = [pool.kv_buffer[i].nbytes for i in range(pool.layer_num)]
        kv_item_lens = [pool.kv_buffer[i][0].nbytes for i in range(pool.layer_num)]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def get_layerwise_buf_info_offsets(self, pool, start_idx=0):
        return [[start_idx + layer_id] for layer_id in range(pool.layer_num)]

    def get_key_buffer(self, pool, layer_id: int):
        return pool.kv_buffer[layer_id]

    def get_key_split_contiguous(self, pool, layer_id: int, indices: torch.Tensor):
        raise RuntimeError("NPU DSA KV int8 packed cache is consumed by npu_kv_quant_sparse_flash_attention and cannot be split as bf16.")

    def get_value_buffer(self, pool, layer_id: int):
        return pool.kv_buffer[layer_id]

    def get_kv_buffer(self, pool, layer_id: int):
        return pool.kv_buffer[layer_id], pool.empty_kr_cache

    def npu_get_cpu_copy_layer(self, pool, layer_id: int, page_indices: torch.Tensor):
        kv_cpu = pool.kv_buffer[layer_id].view(-1, pool.page_size * pool.kv_cache_dim)[page_indices].to("cpu", non_blocking=True)
        return [kv_cpu]

    def npu_load_cpu_copy_layer(
        self,
        pool,
        layer_id: int,
        kv_cache_cpu,
        chunk_idx: int,
        chunk_page_mask_cpu: torch.Tensor,
        chunk_local_page_indices: torch.Tensor,
    ) -> None:
        kv_cpu = kv_cache_cpu[layer_id][chunk_idx][0][chunk_page_mask_cpu]
        if kv_cpu.shape[0] != len(chunk_local_page_indices):
            raise RuntimeError(f"DSA KV int8 CPU-load page count mismatch: kv={kv_cpu.shape[0]} target={len(chunk_local_page_indices)}.")
        pool.kv_buffer[layer_id].view(-1, pool.page_size * pool.kv_cache_dim)[chunk_local_page_indices] = kv_cpu.to(
            pool.kv_buffer[0].device, non_blocking=True
        )

    def get_cpu_copy_layer(self, pool, layer_id: int, page_indices: torch.Tensor):
        return self.npu_get_cpu_copy_layer(pool, layer_id, page_indices)

    def load_cpu_copy_layer(
        self,
        pool,
        layer_id: int,
        kv_cache_cpu,
        page_indices: torch.Tensor,
    ) -> None:
        kv_cpu = kv_cache_cpu[layer_id][0][0]
        assert kv_cpu.shape[0] == len(page_indices), f"kv_cpu.shape[0] {kv_cpu.shape[0]} != len(page_indices) {len(page_indices)}"
        kv = kv_cpu.to(pool.kv_buffer[0].device, non_blocking=True)
        pool.kv_buffer[layer_id].view(-1, pool.page_size * pool.kv_cache_dim)[page_indices] = kv


class DSATokenToKVPool(MLATokenToKVPool):
    quant_block_size = 128

    def __init__(
        self,
        size: int,
        model_dtype: torch.dtype,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        max_batch_size: int,
        max_context_len: int,
        page_size: int,
        rank: int,
        index_head_dim: int,
        index_dtype: torch.dtype,
        npu_dsa_kv_int8: bool = False,
        offload_cache: bool = False,
        sparse_coordinator: SparseCoordinator | None = None,
        npu_dsa_packed_kv_row_bytes: int = 656,
        start_layer: int | None = None,
        end_layer: int | None = None,
        indexer_layer_mask=None,
    ):
        self.indexer_layer_mask = normalize_indexer_mask(layer_num, indexer_layer_mask)
        self.has_indexer_placeholders = not all(self.indexer_layer_mask)
        placeholder_layout = (
            npu_dsa_kv_int8
            and page_size == 128
            and npu_dsa_packed_kv_row_bytes == 672
            and model_dtype == torch.bfloat16
            and kv_lora_rank == 512
            and qk_rope_head_dim == 64
        ) or (not npu_dsa_kv_int8 and ENV.npu_enable_mla_split_kv_kr)
        if self.has_indexer_placeholders and (not is_npu() or not placeholder_layout or index_dtype != torch.bfloat16):
            raise ValueError("Indexer placeholders require NPU BF16 IndexK with split BF16 or page128/row672 packed BF16-activation KV")
        self.npu_dsa_kv_int8 = npu_dsa_kv_int8
        self.npu_dsa_packed_kv_row_bytes = npu_dsa_packed_kv_row_bytes
        self.dsa_kv_int8_packed = npu_dsa_kv_int8
        self.packed_host_offload = bool(npu_dsa_kv_int8 and offload_cache)
        if self.dsa_kv_int8_packed and kv_lora_rank % self.quant_block_size != 0:
            raise ValueError(f"NPU DSA KV int8 requires kv_lora_rank to be divisible by tile_size={self.quant_block_size}; got {kv_lora_rank}.")
        super().__init__(
            size,
            model_dtype,
            dtype,
            None,
            kv_lora_rank,
            qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            max_batch_size,
            max_context_len,
            page_size,
            rank,
            use_dsa=True,
            npu_dsa_kv_int8=npu_dsa_kv_int8,
            offload_cache=offload_cache,
            start_layer=start_layer,
            end_layer=end_layer,
        )
        if self.dsa_kv_int8_packed:
            logger.info("DSA KV int8 packed C8 enabled: KV cache is int8 packed and indexer cache remains separate.")
        if index_dtype not in (torch.float8_e4m3fn, torch.bfloat16):
            raise ValueError(f"Unsupported DSA index dtype: {index_dtype}.")
        if index_head_dim != 128:
            raise ValueError(f"DSA index_head_dim must be 128, got {index_head_dim}.")
        if not is_npu():
            assert self.page_size == 64
        self.index_head_dim = index_head_dim
        self.index_dtype = index_dtype
        if index_dtype == torch.float8_e4m3fn:
            self.index_k_cache_dim = index_head_dim + index_head_dim // self.quant_block_size * 4
            self.index_k_with_scale_buffer_dtype = torch.uint8
        else:
            self.index_k_cache_dim = index_head_dim
            self.index_k_with_scale_buffer_dtype = torch.bfloat16

        self.sparse_coordinator = sparse_coordinator
        if self.offload_cache and self.sparse_coordinator:
            self.sparse_coordinator.create_selected_buffer(self.memory_saver_adapter)

        with self.memory_saver_adapter.region(tag=GPU_MEMORY_TYPE_KV_CACHE):
            self.create_index_k_buffer()

    def make_buffer_layout(self) -> MLABufferLayout:
        if self.dsa_kv_int8_packed:
            return DsaInt8PackedKVLayout(self.npu_dsa_packed_kv_row_bytes)
        return make_mla_buffer_layout(self.quant_method, self.use_dsa, self.dtype)

    def create_index_k_buffer(self):
        if self.has_indexer_placeholders:
            page_num = self.kv_page_num if self.dsa_kv_int8_packed else (self.size + self.page_size) // self.page_size
            (self.index_k_buffer, self.index_k_placeholder_buffer, self.index_k_with_scale_buffer) = allocate_indexer_pages(
                self.indexer_layer_mask,
                page_num,
                self.page_size,
                self.index_head_dim,
                dtype=self.index_k_with_scale_buffer_dtype,
                device=self.device,
            )
            self.index_k_buffer_page_slices = get_page_slices_by_layer_and_page_idx(self.index_k_with_scale_buffer)
            logger.info(
                "NPU IndexK active=%s logical=%s allocated_bytes=%s",
                sum(self.indexer_layer_mask),
                self.layer_num,
                sum(t.nbytes for t in self.index_k_with_scale_buffer),
            )
            return
        if is_npu():
            if ENV.npu_enable_mla_split_kv_kr or self.dsa_kv_int8_packed:
                page_num = self.kv_page_num if self.dsa_kv_int8_packed else (self.size + self.page_size) // self.page_size
                self.index_k_buffer = torch.zeros(
                    (self.layer_num, page_num, self.page_size, 1, self.index_head_dim),
                    dtype=self.index_k_with_scale_buffer_dtype,
                    device=self.device,
                ).contiguous()
                self.index_k_with_scale_buffer = [self.index_k_buffer[i] for i in range(self.layer_num)]
                self.index_k_buffer_page_slices = get_page_slices_by_layer_and_page_idx(self.index_k_buffer)
            else:
                self.index_k_with_scale_buffer = [
                    torch.empty(
                        (self.size + self.page_size, 1, self.index_k_cache_dim),
                        dtype=self.index_k_with_scale_buffer_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
        else:
            page_num = (self.size + self.page_size) // self.page_size
            self.index_k_buffer = torch.zeros(
                (self.layer_num, page_num, self.page_size, 1, self.index_k_cache_dim),
                dtype=self.index_k_with_scale_buffer_dtype,
                device=self.device,
            ).contiguous()
            self.index_k_with_scale_buffer = [self.index_k_buffer[i] for i in range(self.layer_num)]
            self.index_k_buffer_page_slices = get_page_slices_by_layer_and_page_idx(self.index_k_buffer)

    def get_contiguous_buf_infos(self):
        kv_data_ptrs, kv_data_lens, kv_item_lens = super().get_contiguous_buf_infos()
        index_data_ptrs, index_data_lens, index_item_lens = self.get_state_buf_infos()

        return (
            kv_data_ptrs + index_data_ptrs,
            kv_data_lens + index_data_lens,
            kv_item_lens + index_item_lens,
        )

    def get_buffer_device_types(self):
        kv_device_types = super().get_buffer_device_types()
        if kv_device_types is None:
            return None
        return kv_device_types + [buf.device.type for buf in self.index_k_with_scale_buffer]

    def get_layerwise_buf_info_offsets(self, start_idx=0):
        if self.dsa_kv_int8_packed:
            num_kv_buffers = self.layer_num
            offsets = [[start_idx + layer_id] for layer_id in range(self.layer_num)]
            index_offsets = [[start_idx + num_kv_buffers + layer_id] for layer_id in range(self.layer_num)]
            return [x + (y if self.indexer_layer_mask[i] else []) for i, (x, y) in enumerate(zip(offsets, index_offsets))]
        if ENV.npu_enable_mla_split_kv_kr and is_npu():
            num_kv_buffers = 2 * self.layer_num
            offsets = [[start_idx + layer_id * 2 + i for i in range(2)] for layer_id in range(self.layer_num)]
            index_offsets = [[start_idx + num_kv_buffers + layer_id] for layer_id in range(self.layer_num)]
            return [x + (y if self.indexer_layer_mask[i] else []) for i, (x, y) in enumerate(zip(offsets, index_offsets))]
        else:
            return [
                [start_idx + layer_id] + ([start_idx + self.layer_num + layer_id] if self.indexer_layer_mask[layer_id] else [])
                for layer_id in range(self.layer_num)
            ]

    def has_indexer_cache(self, layer_id: int) -> bool:
        local_id = layer_id - self.start_layer
        if not 0 <= local_id < self.layer_num:
            raise IndexError(f"Attention layer {layer_id} outside local IndexK layout")
        return self.indexer_layer_mask[local_id]

    def _check_indexer_layer(self, layer_id: int) -> int:
        if not self.has_indexer_cache(layer_id):
            raise ValueError(f"Attention layer {layer_id} has only an IndexK placeholder")
        return layer_id - self.start_layer

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        layer_id = self._check_indexer_layer(layer_id)
        if getattr(self, "layer_transfer_counter", None) is not None:
            self.layer_transfer_counter.wait_until(layer_id)
        return self.index_k_with_scale_buffer[layer_id]

    def get_index_k_continuous(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ):
        buf = self.get_index_k_with_scale_buffer(layer_id)
        buf = buf.view(-1, self.page_size * self.index_k_cache_dim)
        return index_buf_accessor.GetK.execute(self, buf, seq_len=seq_len, page_indices=page_indices)

    def get_index_k_scale_continuous(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ):
        buf = self.get_index_k_with_scale_buffer(layer_id)
        if self.index_dtype == torch.float8_e4m3fn:
            buf = buf.view(-1, self.page_size * self.index_k_cache_dim)
            return index_buf_accessor.GetS.execute(self, buf, seq_len=seq_len, page_indices=page_indices)
        else:
            return buf[page_indices]

    def get_index_k_by_token_idxs(
        self,
        layer_id: int,
        seq_len: int,
        token_idxs: torch.Tensor,
    ):
        buf = self.get_index_k_with_scale_buffer(layer_id)
        if self.index_dtype == torch.float8_e4m3fn:
            buf = buf.view(-1, self.page_size * self.index_k_cache_dim)
            return index_buf_accessor.get_k_and_s_by_token_idxs_torch(self, buf, token_idxs)
        else:
            buf = buf.view(-1, self.index_k_cache_dim)
            return buf[token_idxs]

    def get_index_k_flat_idxs_by_token_idxs(
        self,
        layer_id: int,
        seq_len: int,
        token_idxs: torch.Tensor,
    ):
        buf = self.get_index_k_with_scale_buffer(layer_id)
        if self.index_dtype == torch.float8_e4m3fn:
            buf = buf.view(-1, self.page_size * self.index_k_cache_dim)
            return index_buf_accessor.get_k_and_s_flat_idxs_by_token_idxs_torch(self, buf, token_idxs)
        else:
            raise NotImplementedError

    def set_index_k_and_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor = None,
    ) -> None:
        layer_id = self._check_indexer_layer(layer_id)
        if not is_npu():
            buf = self.index_k_with_scale_buffer[layer_id]
            if self.index_dtype == torch.float8_e4m3fn:
                # [page_num, page_size*dim]
                buf = buf.view(-1, self.page_size * self.index_k_cache_dim)
                index_buf_accessor.SetKAndS.execute(pool=self, buf=buf, loc=loc, index_k=index_k, index_k_scale=index_k_scale)
            else:
                buf = buf.view(-1, self.index_k_cache_dim)
                buf[loc] = index_k
        else:
            import torch_npu

            torch_npu.npu_scatter_nd_update_(
                self.index_k_with_scale_buffer[layer_id].view(-1, 1, self.index_head_dim),
                loc.view(-1, 1),
                index_k.view(-1, 1, self.index_head_dim),
            )

    def get_state_buf_infos(self):
        data_ptrs = [self.index_k_with_scale_buffer[i].data_ptr() for i in range(self.layer_num)]
        data_lens = [self.index_k_with_scale_buffer[i].nbytes for i in range(self.layer_num)]
        item_lens = [self.index_k_with_scale_buffer[i][0].nbytes for i in range(self.layer_num)]
        return data_ptrs, data_lens, item_lens

    def get_kv_size_bytes(self):
        kv_size_bytes = super().get_kv_size_bytes()
        if hasattr(self, "index_k_with_scale_buffer"):
            for index_k_cache in self.index_k_with_scale_buffer:
                kv_size_bytes += get_tensor_size_bytes(index_k_cache)
        return kv_size_bytes

    def get_cpu_copy(self, token_indices):
        if not is_npu():
            kv_cache_cpu = super().get_cpu_copy(token_indices)
            self.sync_device()
            index_cache_cpu = []
            page_indices = token_indices[:: self.page_size] // self.page_size
            page_chunk_size = max(1, self.offload_chunk_page_num // self.page_size)
            for layer_id in range(self.layer_num):
                index_cache_cpu.append([])
                if not __is_npu__:
                    for i in range(0, len(page_indices), page_chunk_size):
                        chunk_indices = page_indices[i : i + page_chunk_size]
                        index_cpu = (
                            self.index_k_with_scale_buffer[layer_id]
                            .view(-1, self.page_size * self.index_k_cache_dim)[chunk_indices]
                            .to("cpu", non_blocking=True)
                        )
                        index_cache_cpu[-1].append(index_cpu)
                else:
                    raise NotImplementedError
            self.sync_device()
            return kv_cache_cpu, index_cache_cpu
        else:
            page_indices = token_indices[:: self.page_size] // self.page_size
            return self.npu_get_cpu_copy(page_indices)

    def load_cpu_copy(self, kv_cache_cpu, token_indices, prefix_page_count=0):
        if not is_npu():
            assert len(kv_cache_cpu) == 2
            kv_cache_cpu, index_cache_cpu = kv_cache_cpu
            super().load_cpu_copy(kv_cache_cpu, token_indices, prefix_page_count)
            self.sync_device()
            page_indices = token_indices[:: self.page_size] // self.page_size
            page_chunk_size = max(1, self.offload_chunk_page_num // self.page_size)
            for layer_id in range(self.layer_num):
                for i in range(0, len(page_indices), page_chunk_size):
                    chunk_indices = page_indices[i : i + page_chunk_size]
                    index_cpu = index_cache_cpu[layer_id][i // page_chunk_size]
                    assert index_cpu.shape[0] == len(chunk_indices), (
                        f"index_cpu.shape[0] {index_cpu.shape[0]} != len(chunk_indices) {len(chunk_indices)}"
                    )
                    index_chunk = index_cpu.to(self.index_k_with_scale_buffer[0].device, non_blocking=True)
                    (self.index_k_with_scale_buffer[layer_id].view(-1, self.page_size * self.index_k_cache_dim)[chunk_indices]) = index_chunk
            self.sync_device()
        else:
            del prefix_page_count
            page_indices = token_indices[:: self.page_size] // self.page_size
            local_page_mask = torch.ones_like(page_indices, dtype=torch.bool)
            self.npu_load_cpu_copy(
                kv_cache_cpu,
                local_page_mask,
                local_page_mask.cpu(),
                page_indices,
                page_indices,
            )

    def npu_get_cpu_copy(self, local_page_indices: torch.Tensor):
        from sglang.srt.env import ENV

        if not (ENV.npu_enable_mla_split_kv_kr or self.dsa_kv_int8_packed):
            raise RuntimeError("DSATokenToKVPool.npu_get_cpu_copy requires split KV or DSA KV int8.")
        kv_cache_cpu = super().npu_get_cpu_copy(local_page_indices)
        index_cache_cpu = []
        for layer_id in range(self.layer_num):
            index_cache_cpu.append([])
            for i in range(0, len(local_page_indices), self.offload_chunk_page_num):
                chunk_page_indices = local_page_indices[i : i + self.offload_chunk_page_num]
                if not self.indexer_layer_mask[layer_id]:
                    # Preserve chunk/row alignment with KV without serializing
                    # a logical page that does not exist in this layout.
                    index_cache_cpu[-1].append([torch.empty((len(chunk_page_indices), 0), dtype=self.index_dtype, device="cpu")])
                    continue
                index_cpu = (
                    self.index_k_with_scale_buffer[layer_id]
                    .view(-1, self.page_size * self.index_head_dim)[chunk_page_indices]
                    .to("cpu", non_blocking=True)
                )
                index_cache_cpu[-1].append([index_cpu])
        self.sync_device()
        return kv_cache_cpu, index_cache_cpu

    def npu_load_cpu_copy(
        self,
        kv_cache_cpu,
        local_page_mask: torch.Tensor,
        local_page_mask_cpu: torch.Tensor,
        dense_local_page_indices: torch.Tensor,
        page_indices: torch.Tensor,
    ):
        kv_cache_only, index_cache_cpu = kv_cache_cpu
        super().npu_load_cpu_copy(
            kv_cache_only,
            local_page_mask,
            local_page_mask_cpu,
            dense_local_page_indices,
            page_indices,
        )
        self.sync_device()
        for layer_id in range(self.layer_num):
            if not self.indexer_layer_mask[layer_id]:
                continue
            for i in range(0, len(page_indices), self.offload_chunk_page_num):
                chunk_page_mask_cpu = local_page_mask_cpu[i : i + self.offload_chunk_page_num]
                if not chunk_page_mask_cpu.any():
                    continue
                chunk_page_mask = local_page_mask[i : i + self.offload_chunk_page_num]
                chunk_local_page_indices = dense_local_page_indices[i : i + self.offload_chunk_page_num][chunk_page_mask]
                index_cpu = index_cache_cpu[layer_id][i // self.offload_chunk_page_num][0][chunk_page_mask_cpu]
                assert index_cpu.shape[0] == len(chunk_local_page_indices)
                self.index_k_with_scale_buffer[layer_id].view(-1, self.page_size * self.index_head_dim)[chunk_local_page_indices] = index_cpu.to(
                    self.index_k_with_scale_buffer[0].device, non_blocking=True
                )
        self.sync_device()

    def npu_load_rank_local_cpu_copy(
        self,
        kv_local,
        index_local,
        target_local_pages: torch.Tensor,
        skip_prefix_local_pages: int = 0,
    ):
        if not (ENV.npu_enable_mla_split_kv_kr or self.dsa_kv_int8_packed):
            raise RuntimeError("DSA CPU offload requires split KV or DSA KV int8 packed cache.")

        total_target = int(target_local_pages.numel())
        self.sync_device()
        for layer_id in range(self.layer_num):
            layer_kv_chunks = kv_local[layer_id]
            layer_index_chunks = index_local[layer_id]
            skip_remaining = skip_prefix_local_pages
            write_offset = 0

            for kv_chunk, index_chunk in zip(layer_kv_chunks, layer_index_chunks):
                kv_cpu1 = kv_chunk[0]
                kv_cpu2 = None if self.dsa_kv_int8_packed else kv_chunk[1]
                index_cpu = index_chunk[0]
                chunk_page_num = kv_cpu1.shape[0]
                if self.dsa_kv_int8_packed:
                    assert index_cpu.shape[0] == chunk_page_num, f"inconsistent chunk sizes: kv={kv_cpu1.shape[0]} index={index_cpu.shape[0]}"
                else:
                    assert kv_cpu2.shape[0] == chunk_page_num and index_cpu.shape[0] == chunk_page_num, (
                        f"inconsistent chunk sizes: kv1={kv_cpu1.shape[0]} kv2={kv_cpu2.shape[0]} index={index_cpu.shape[0]}"
                    )
                if chunk_page_num == 0:
                    continue

                chunk_start = 0
                if skip_remaining > 0:
                    if skip_remaining >= chunk_page_num:
                        skip_remaining -= chunk_page_num
                        continue
                    chunk_start = skip_remaining
                    skip_remaining = 0

                kv_cpu1 = kv_cpu1[chunk_start:]
                if kv_cpu2 is not None:
                    kv_cpu2 = kv_cpu2[chunk_start:]
                index_cpu = index_cpu[chunk_start:]
                pages_in_this_chunk = kv_cpu1.shape[0]
                chunk_target = target_local_pages[write_offset : write_offset + pages_in_this_chunk]
                assert chunk_target.numel() == pages_in_this_chunk, (
                    f"target page count mismatch: target={chunk_target.numel()} payload={pages_in_this_chunk}"
                )
                write_offset += pages_in_this_chunk

                if self.dsa_kv_int8_packed:
                    self.kv_buffer[layer_id].view(-1, self.page_size * self.kv_cache_dim)[chunk_target] = kv_cpu1.to(
                        self.kv_buffer[0].device, non_blocking=True
                    )
                else:
                    self.kv_buffer[2 * layer_id].view(-1, self.page_size * self.kv_lora_rank)[chunk_target] = kv_cpu1.to(
                        self.kv_buffer[0].device, non_blocking=True
                    )
                    self.kv_buffer[2 * layer_id + 1].view(-1, self.page_size * self.qk_rope_head_dim)[chunk_target] = kv_cpu2.to(
                        self.kv_buffer[0].device, non_blocking=True
                    )
                if self.indexer_layer_mask[layer_id]:
                    self.index_k_with_scale_buffer[layer_id].view(-1, self.page_size * self.index_head_dim)[chunk_target] = index_cpu.to(
                        self.index_k_with_scale_buffer[0].device, non_blocking=True
                    )

            assert skip_remaining == 0, (
                f"failed to skip {skip_prefix_local_pages} prefix local pages; {skip_remaining} still remaining after all chunks"
            )
            assert write_offset == total_target, f"load_cpu_copy wrote {write_offset} local pages but target has {total_target}"
        self.sync_device()
