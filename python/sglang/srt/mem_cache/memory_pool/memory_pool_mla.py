from __future__ import annotations

import abc
import hashlib
from dataclasses import dataclass

import numpy as np
import torch
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.env import ENV, global_server_args_dict
from sglang.srt.mem_cache.kv_offload.budget import is_offloaded_attention
from sglang.srt.layers.attention.dsa.quant_k_cache import quantize_k_cache
from sglang.srt.layers.attention.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool.memory_pool_base import BaseTokenToKVPool
from sglang.srt.mem_cache.memory_pool.memory_pool_kernel import (
    _KV_COPY_NUM_WARPS_LARGE_TILE,
    _KV_COPY_NUM_WARPS_SMALL_TILE,
    _KV_COPY_STRIDE_THRESHOLD_LARGE,
    _KV_COPY_STRIDE_THRESHOLD_MEDIUM,
    _KV_COPY_TILE_SIZE_LARGE,
    _KV_COPY_TILE_SIZE_MEDIUM,
    _KV_COPY_TILE_SIZE_SMALL,
    GB,
    copy_all_layer_kv_cache_tiled,
    get_tensor_size_bytes,
    set_mla_kv_buffer_triton,
)
from sglang.srt.utils import get_colorful_logger, get_page_slices_by_layer_and_page_idx, is_npu, is_sm90_supported
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if not is_npu() and is_sm90_supported():
    from flash_mla_fp8 import dequantize_ckv_fused_indexed, quantize_and_cache_k
elif is_npu():
    import torch_npu

logger = get_colorful_logger(__name__)

__is_npu__ = is_npu()


@dataclass(frozen=True)
class MLABufferSpec:
    size: int
    page_num: int
    page_size: int
    layer_num: int
    dtype: torch.dtype
    store_dtype: torch.dtype
    model_dtype: torch.dtype
    device: str
    kv_cache_dim: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    nsa_kv_cache_store_fp8: bool = False
    offload_cache: bool = False
    start_layer: int = 0
    offload_cache_storage: str = "host"


@dataclass
class MLABuffers:
    kv_buffer: list
    buffer_device_types: list | None = None
    k_buffer_page_slices: list | None = None
    v_buffer_page_slices: list | None = None
    empty_kr_cache: torch.Tensor | None = None


class MLABufferLayout(abc.ABC):
    is_paged = False
    requires_npu_page_copy = False
    allow_public_kv_setters = True

    def raise_unsupported_error(self, func_name: str) -> None:
        raise RuntimeError(f"{func_name} is not supported for this KV layout.")

    @abc.abstractmethod
    def alloc_buffers(self, spec: MLABufferSpec) -> MLABuffers:
        raise NotImplementedError

    @abc.abstractmethod
    def get_kv_cache_dim(
        self,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        model_dtype: torch.dtype,
    ) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def get_page_num(self, pool) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def get_page_size_bytes(self, pool) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def get_log_buffer_info(self, pool):
        raise NotImplementedError

    @abc.abstractmethod
    def move_kv_cache_eager(
        self,
        pool,
        tgt_loc_flat: torch.Tensor,
        src_loc_flat: torch.Tensor,
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def get_contiguous_buf_infos(self, pool):
        raise NotImplementedError

    @abc.abstractmethod
    def get_layerwise_buf_info_offsets(self, pool, start_idx=0):
        raise NotImplementedError

    @abc.abstractmethod
    def get_key_buffer(self, pool, layer_id: int):
        raise NotImplementedError

    @abc.abstractmethod
    def get_key_split_contiguous(self, pool, layer_id: int, indices: torch.Tensor):
        raise NotImplementedError

    @abc.abstractmethod
    def get_value_buffer(self, pool, layer_id: int):
        raise NotImplementedError

    @abc.abstractmethod
    def get_kv_buffer(self, pool, layer_id: int):
        raise NotImplementedError

    @abc.abstractmethod
    def npu_get_cpu_copy_layer(self, pool, layer_id: int, page_indices: torch.Tensor):
        raise NotImplementedError

    @abc.abstractmethod
    def npu_load_cpu_copy_layer(
        self,
        pool,
        layer_id: int,
        kv_cache_cpu,
        chunk_idx: int,
        chunk_page_mask_cpu: torch.Tensor,
        chunk_local_page_indices: torch.Tensor,
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def get_cpu_copy_layer(self, pool, layer_id: int, page_indices: torch.Tensor):
        raise NotImplementedError

    @abc.abstractmethod
    def load_cpu_copy_layer(
        self,
        pool,
        layer_id: int,
        kv_cache_cpu,
        page_indices: torch.Tensor,
    ) -> None:
        raise NotImplementedError


class MLABufferLayoutBase(MLABufferLayout):
    def get_kv_cache_dim(
        self,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        model_dtype: torch.dtype,
    ) -> int:
        del model_dtype
        return kv_lora_rank + qk_rope_head_dim

    def get_page_num(self, pool) -> int:
        return (pool.size + pool.page_size) // pool.page_size

    def get_page_size_bytes(self, pool) -> int:
        return pool.page_size * pool.layer_num * pool.kv_cache_dim * torch._utils._element_size(pool.dtype)

    def get_log_buffer_info(self, pool):
        device_info = pool.kv_buffer[-1].device
        shape_info = pool.kv_buffer[0].shape
        return device_info, shape_info

    def move_kv_cache_eager(self, pool, tgt_loc_flat, src_loc_flat) -> None:
        for buf in pool.kv_buffer:
            buf[tgt_loc_flat] = buf[src_loc_flat]

    def get_contiguous_buf_infos(self, pool):
        enable_hicache = global_server_args_dict.get("enable_hierarchical_cache", False)
        kv_data_ptrs = [pool.kv_buffer[i].data_ptr() for i in range(pool.layer_num)]
        kv_data_lens = [pool.kv_buffer[i].nbytes for i in range(pool.layer_num)]
        kv_item_lens = [
            (pool.kv_buffer[i][0][0].nbytes if enable_hicache else pool.kv_buffer[i][0].nbytes) * pool.page_size for i in range(pool.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def get_layerwise_buf_info_offsets(self, pool, start_idx=0):
        return [[start_idx + layer_id] for layer_id in range(pool.layer_num)]

    def get_key_buffer(self, pool, layer_id: int):
        if pool.store_dtype != pool.dtype:
            return pool.kv_buffer[layer_id].view(pool.dtype)
        return pool.kv_buffer[layer_id]

    def get_key_split_contiguous(self, pool, layer_id: int, indices: torch.Tensor):
        if pool.store_dtype != pool.dtype:
            latent_cache = pool.kv_buffer[layer_id].view(pool.dtype)[indices].contiguous()
        else:
            latent_cache = pool.kv_buffer[layer_id][indices].contiguous()
        kv_a_normed, k_pe = latent_cache.split([pool.kv_lora_rank, pool.qk_rope_head_dim], dim=-1)
        return kv_a_normed, k_pe

    def get_value_buffer(self, pool, layer_id: int):
        if pool.store_dtype != pool.dtype:
            return pool.kv_buffer[layer_id][..., : pool.kv_lora_rank].view(pool.dtype)
        return pool.kv_buffer[layer_id][..., : pool.kv_lora_rank]

    def get_kv_buffer(self, pool, layer_id: int):
        return self.get_key_buffer(pool, layer_id), self.get_value_buffer(pool, layer_id)

    def npu_get_cpu_copy_layer(self, pool, layer_id: int, page_indices: torch.Tensor):
        kv_cpu = (
            pool.kv_buffer[layer_id].view(-1, pool.page_size * (pool.qk_rope_head_dim + pool.kv_lora_rank))[page_indices].to("cpu", non_blocking=True)
        )
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
        assert kv_cpu.shape[0] == len(chunk_local_page_indices)
        pool.kv_buffer[layer_id].view(-1, pool.page_size * (pool.kv_lora_rank + pool.qk_rope_head_dim))[chunk_local_page_indices] = kv_cpu.to(
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
        pool.kv_buffer[layer_id].view(-1, pool.page_size * (pool.kv_lora_rank + pool.qk_rope_head_dim))[page_indices] = kv


class MLADenseKVLayout(MLABufferLayoutBase):
    def alloc_buffers(self, spec: MLABufferSpec) -> MLABuffers:
        return MLABuffers(
            kv_buffer=[
                torch.ones(
                    (spec.size + spec.page_size, 1, spec.kv_cache_dim),
                    dtype=spec.store_dtype,
                    device=spec.device,
                )
                for _ in range(spec.layer_num)
            ]
        )


class MLASplitKVLayout(MLABufferLayoutBase):
    is_paged = True
    requires_npu_page_copy = True

    def get_page_size_bytes(self, pool) -> int:
        return pool.page_size * pool.layer_num * pool.kv_cache_dim * torch._utils._element_size(pool.dtype)

    def alloc_buffers(self, spec: MLABufferSpec) -> MLABuffers:
        assert not spec.nsa_kv_cache_store_fp8
        kv_buffer = []
        buffer_device_types = []
        for layer_idx in range(spec.layer_num):
            # Keep HBM padding and reserved pages initialized. Host-backed
            # layers must retain the provider's swapped-memory allocator.
            create_tensor_func = torch.zeros
            device_type = "npu"
            if spec.offload_cache and spec.offload_cache_storage == "host" and is_offloaded_attention(spec.start_layer + layer_idx):
                create_tensor_func = torch_npu.empty_with_swapped_memory
                device_type = "cpu"
            kv_buffer.append(
                create_tensor_func(
                    (spec.page_num, spec.page_size, 1, spec.kv_lora_rank),
                    dtype=spec.store_dtype,
                    device=spec.device,
                ).contiguous()
            )
            kv_buffer.append(
                create_tensor_func(
                    (spec.page_num, spec.page_size, 1, spec.qk_rope_head_dim),
                    dtype=spec.store_dtype,
                    device=spec.device,
                ).contiguous()
            )
            # Twice on purpose: this list is parallel to kv_buffer, one entry
            # per buffer and not per layer, and the loop just appended two
            # (nope, rope). Consumers rely on the lengths matching -
            # memory_pool_dsa zips the two lists, and the P/D registration
            # path falls back to one entry per kv_data_ptr.
            buffer_device_types.append(device_type)
            buffer_device_types.append(device_type)

        enable_hicache = global_server_args_dict.get("enable_hierarchical_cache", False)
        if enable_hicache:
            k_buffer = kv_buffer[::2]
            v_buffer = kv_buffer[1::2]
            return MLABuffers(
                kv_buffer=kv_buffer,
                buffer_device_types=buffer_device_types,
                k_buffer_page_slices=get_page_slices_by_layer_and_page_idx(k_buffer),
                v_buffer_page_slices=get_page_slices_by_layer_and_page_idx(v_buffer),
            )

        return MLABuffers(kv_buffer=kv_buffer,buffer_device_types=buffer_device_types)

    def get_log_buffer_info(self, pool):
        first_kv_buffer, first_kr_buffer, *_ = pool.kv_buffer
        device_info = first_kv_buffer.device
        shape_info = f"({first_kv_buffer.shape}, {first_kr_buffer.shape})"
        return device_info, shape_info

    def get_contiguous_buf_infos(self, pool):
        kv_data_ptrs = [pool.kv_buffer[i].data_ptr() for i in range(len(pool.kv_buffer))]
        kv_data_lens = [pool.kv_buffer[i].nbytes for i in range(len(pool.kv_buffer))]
        kv_item_lens = [pool.kv_buffer[i][0][0].nbytes * pool.page_size for i in range(len(pool.kv_buffer))]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def get_layerwise_buf_info_offsets(self, pool, start_idx=0):
        return [[start_idx + layer_id * 2 + i for i in range(2)] for layer_id in range(pool.layer_num)]

    def get_key_buffer(self, pool, layer_id: int):
        raise RuntimeError("get_key_buffer is not implemented when enable npu_enable_mla_split_kv_kr.")

    def get_key_split_contiguous(self, pool, layer_id: int, indices: torch.Tensor):
        raise RuntimeError("get_key_split_contiguous is not implemented when enable npu_enable_mla_split_kv_kr.")

    def get_value_buffer(self, pool, layer_id: int):
        raise RuntimeError("get_value_buffer is not implemented when enable npu_enable_mla_split_kv_kr.")

    def get_kv_buffer(self, pool, layer_id: int):
        return pool.kv_buffer[layer_id * 2], pool.kv_buffer[layer_id * 2 + 1]

    def npu_get_cpu_copy_layer(self, pool, layer_id: int, page_indices: torch.Tensor):
        kv_cpu1 = pool.kv_buffer[2 * layer_id].view(-1, pool.page_size * pool.kv_lora_rank)[page_indices].to("cpu", non_blocking=True)
        kv_cpu2 = pool.kv_buffer[2 * layer_id + 1].view(-1, pool.page_size * pool.qk_rope_head_dim)[page_indices].to("cpu", non_blocking=True)
        return [kv_cpu1, kv_cpu2]

    def npu_load_cpu_copy_layer(
        self,
        pool,
        layer_id: int,
        kv_cache_cpu,
        chunk_idx: int,
        chunk_page_mask_cpu: torch.Tensor,
        chunk_local_page_indices: torch.Tensor,
    ) -> None:
        kv_cpu1 = kv_cache_cpu[layer_id][chunk_idx][0][chunk_page_mask_cpu]
        kv_cpu2 = kv_cache_cpu[layer_id][chunk_idx][1][chunk_page_mask_cpu]
        assert kv_cpu1.shape[0] == len(chunk_local_page_indices) and kv_cpu2.shape[0] == len(chunk_local_page_indices)
        pool.kv_buffer[2 * layer_id].view(-1, pool.page_size * pool.kv_lora_rank)[chunk_local_page_indices] = kv_cpu1.to(
            pool.kv_buffer[0].device, non_blocking=True
        )
        pool.kv_buffer[2 * layer_id + 1].view(-1, pool.page_size * pool.qk_rope_head_dim)[chunk_local_page_indices] = kv_cpu2.to(
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
        kv_cpu1 = kv_cache_cpu[layer_id][0][0]
        kv_cpu2 = kv_cache_cpu[layer_id][0][1]
        assert kv_cpu1.shape[0] == len(page_indices) and kv_cpu2.shape[0] == len(page_indices), (
            f"kv_cpu1.shape[0] {kv_cpu1.shape[0]} or kv_cpu2.shape[0] {kv_cpu2.shape[0]} != len(page_indices) {len(page_indices)}"
        )
        kv1 = kv_cpu1.to(pool.kv_buffer[0].device, non_blocking=True)
        kv2 = kv_cpu2.to(pool.kv_buffer[0].device, non_blocking=True)
        pool.kv_buffer[2 * layer_id].view(-1, pool.page_size * pool.kv_lora_rank)[page_indices] = kv1
        pool.kv_buffer[2 * layer_id + 1].view(-1, pool.page_size * pool.qk_rope_head_dim)[page_indices] = kv2


class MLAPerTokenHeadLayout(MLABufferLayout):
    def get_kv_cache_dim(
        self,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        model_dtype: torch.dtype,
    ) -> int:
        del model_dtype
        return kv_lora_rank + qk_rope_head_dim

    def get_page_num(self, pool) -> int:
        return (pool.size + pool.page_size) // pool.page_size

    def get_page_size_bytes(self, pool) -> int:
        dim_size_bytes = (
            pool.kv_lora_rank * torch._utils._element_size(pool.dtype)
            + pool.qk_rope_head_dim * torch._utils._element_size(pool.model_dtype)
            + torch._utils._element_size(torch.float32)
        )
        return pool.page_size * pool.layer_num * dim_size_bytes

    def alloc_buffers(self, spec: MLABufferSpec) -> MLABuffers:
        if ENV.npu_enable_mla_split_kv_kr:
            raise RuntimeError("per_token_head is not implemented when enable npu_enable_mla_split_kv_kr.")
        return MLABuffers(
            kv_buffer=[
                (
                    torch.empty(
                        (spec.size + spec.page_size, 1, spec.kv_lora_rank),
                        dtype=spec.store_dtype,
                        device=spec.device,
                    ),
                    torch.empty(
                        (spec.size + spec.page_size, 1, 1),
                        dtype=torch.float32,
                        device=spec.device,
                    ),
                    torch.empty(
                        (spec.size + spec.page_size, 1, spec.qk_rope_head_dim),
                        dtype=spec.model_dtype,
                        device=spec.device,
                    ),
                )
                for _ in range(spec.layer_num)
            ]
        )

    def get_log_buffer_info(self, pool):
        first_buffer = pool.kv_buffer[0]
        device_info = first_buffer[0].device
        shape_info = f"({first_buffer[0].shape}, {first_buffer[1].shape}, {first_buffer[2].shape})"
        return device_info, shape_info

    def move_kv_cache_eager(self, pool, tgt_loc_flat, src_loc_flat) -> None:
        for layer_buffers in pool.kv_buffer:
            for buf in layer_buffers:
                buf[tgt_loc_flat] = buf[src_loc_flat]

    def get_contiguous_buf_infos(self, pool):
        if ENV.npu_enable_mla_split_kv_kr:
            raise RuntimeError("per_token_head is not implemented when enable npu_enable_mla_split_kv_kr.")
        kv_data_ptrs = [sub_tuple[i].data_ptr() for i in range(3) for sub_tuple in pool.kv_buffer]
        kv_data_lens = [sub_tuple[i].nbytes for i in range(3) for sub_tuple in pool.kv_buffer]
        kv_item_lens = [sub_tuple[i][0].nbytes * pool.page_size for i in range(3) for sub_tuple in pool.kv_buffer]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def get_layerwise_buf_info_offsets(self, pool, start_idx=0):
        return [[start_idx + i * pool.layer_num + layer_id for i in range(3)] for layer_id in range(pool.layer_num)]

    def get_key_buffer(self, pool, layer_id: int):
        return pool.kv_buffer[layer_id]

    def get_key_split_contiguous(self, pool, layer_id: int, indices: torch.Tensor):
        k_lora_cache, k_scale_cache, k_rope_cache = pool.kv_buffer[layer_id]
        if not is_npu() and is_sm90_supported():
            k_lora_deq, k_rope_deq = dequantize_ckv_fused_indexed(k_lora_cache.view(pool.dtype), k_rope_cache, k_scale_cache, indices)
            return k_lora_deq, k_rope_deq

        k_lora = k_lora_cache[indices].view(pool.dtype).float()
        k_scale = k_scale_cache[indices]
        k_rope = k_rope_cache[indices].float()
        k_lora_deq = (k_lora * k_scale).to(pool.model_dtype).contiguous()
        k_rope_deq = (k_rope * k_scale).to(pool.model_dtype).contiguous()
        return k_lora_deq, k_rope_deq

    def get_value_buffer(self, pool, layer_id: int):
        return pool.kv_buffer[layer_id][:2]

    def get_kv_buffer(self, pool, layer_id: int):
        return self.get_key_buffer(pool, layer_id), self.get_value_buffer(pool, layer_id)

    def npu_get_cpu_copy_layer(self, pool, layer_id: int, page_indices: torch.Tensor):
        raise RuntimeError("per_token_head NPU page copy is not supported.")

    def npu_load_cpu_copy_layer(
        self,
        pool,
        layer_id: int,
        kv_cache_cpu,
        chunk_idx: int,
        chunk_page_mask_cpu: torch.Tensor,
        chunk_local_page_indices: torch.Tensor,
    ) -> None:
        raise RuntimeError("per_token_head NPU page load is not supported.")

    def get_cpu_copy_layer(self, pool, layer_id: int, page_indices: torch.Tensor):
        raise RuntimeError("per_token_head NPU page copy is not supported.")

    def load_cpu_copy_layer(
        self,
        pool,
        layer_id: int,
        kv_cache_cpu,
        page_indices: torch.Tensor,
    ) -> None:
        raise RuntimeError("per_token_head NPU page load is not supported.")


class MLAFp8DSALayout(MLADenseKVLayout):
    def get_kv_cache_dim(
        self,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        model_dtype: torch.dtype,
    ) -> int:
        del kv_lora_rank, qk_rope_head_dim, model_dtype
        return 656


def make_mla_buffer_layout(
    quant_method: str,
    use_dsa: bool,
    dtype: torch.dtype,
) -> MLABufferLayout:
    if quant_method == "per_token_head":
        return MLAPerTokenHeadLayout()
    if ENV.npu_enable_mla_split_kv_kr:
        return MLASplitKVLayout()
    if use_dsa and dtype == torch.float8_e4m3fn:
        return MLAFp8DSALayout()
    return MLADenseKVLayout()


class MLATokenToKVPool(BaseTokenToKVPool):
    quant_block_size = 128

    @property
    def is_mla_backend(self) -> bool:
        return True

    def __init__(
        self,
        size: int,
        model_dtype: torch.dtype,
        dtype: torch.dtype,
        quant_method: str,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        max_batch_size: int,
        max_context_len: int,
        page_size: int,
        rank: int,
        use_dsa: bool = False,
        offload_cache: bool = False,
        enable_kv_cache_copy: bool = False,
        enable_alt_stream: bool = True,
        npu_dsa_kv_int8: bool = False,
        start_layer: int | None = None,
        end_layer: int | None = None,
    ):
        super().__init__(
            size,
            dtype,
            device,
            max_batch_size,
            max_context_len,
            page_size,
            rank,
            start_layer,
            end_layer,
        )
        self.model_dtype = model_dtype
        self.quant_method = quant_method
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.layer_num = layer_num
        self.use_dsa = use_dsa
        self.npu_dsa_kv_int8 = npu_dsa_kv_int8
        self.dsa_kv_int8_packed = npu_dsa_kv_int8
        if npu_dsa_kv_int8 and not use_dsa:
            raise ValueError("NPU DSA KV int8 packed cache is only valid for DSA KV pools.")
        if npu_dsa_kv_int8:
            self.store_dtype = torch.int8
        self.offload_cache = offload_cache
        self.nsa_kv_cache_store_fp8 = use_dsa and dtype == torch.float8_e4m3fn
        if ENV.npu_enable_mla_split_kv_kr and self.nsa_kv_cache_store_fp8 and not self.dsa_kv_int8_packed:
            raise AssertionError("nsa_kv_cache_store_fp8 is not supported with split KV layout.")
        self.buf_layout = self.make_buffer_layout()
        self.kv_cache_dim = self.buf_layout.get_kv_cache_dim(
            kv_lora_rank,
            qk_rope_head_dim,
            model_dtype,
        )
        self.page_size_bytes = self.get_page_size_bytes()
        self.kv_page_num = self.get_kv_page_num()
        self.enable_kv_cache_copy = enable_kv_cache_copy
        self.enable_alt_stream = enable_alt_stream

        self.memory_saver_adapter = memory_saver_adapter = TorchMemorySaverAdapter.create(enable=enable_memory_saver)

        with memory_saver_adapter.region(tag=GPU_MEMORY_TYPE_KV_CACHE):
            self.create_kv_buffer()

        self.post_init()

    def post_init(self):
        all_buffers = []
        if self.quant_method == "per_token_head":
            for layer_buffers in self.kv_buffer:
                all_buffers.extend(layer_buffers)
        else:
            all_buffers = self.kv_buffer

        if not self.buf_layout.is_paged:
            self.data_ptrs = torch.tensor(
                [buf.data_ptr() for buf in all_buffers],
                dtype=torch.uint64,
                device=self.device,
            )
            self.data_strides = torch.tensor(
                [np.prod(buf.shape[1:]) * buf.dtype.itemsize for buf in all_buffers],
                device=self.device,
            )

        self.device_module = torch.get_device_module(self.device)
        self.alt_stream = self.device_module.Stream() if torch.cuda.is_available() and self.enable_alt_stream else None

        if self.enable_kv_cache_copy:
            self.init_kv_copy_and_warmup()
        else:
            self._kv_copy_config = None

        kv_size = self.get_kv_size_bytes()
        logger.info(f"KV Cache is allocated. KV size: {kv_size / GB:.2f} GB.")

        device_info, shape_info = self.buf_layout.get_log_buffer_info(self) if self.layer_num else (self.device, ())
        logger.info(
            f"MLATokenToKVPool: {len(self.kv_buffer)=} device={device_info} shape={shape_info} "
            f"{self.size=} {self.page_size=} {self.kv_cache_dim=} {self.store_dtype=}"
        )

    def make_buffer_layout(self) -> MLABufferLayout:
        return make_mla_buffer_layout(self.quant_method, self.use_dsa, self.dtype)

    def make_buffer_spec(self) -> MLABufferSpec:
        return MLABufferSpec(
            size=self.size,
            page_num=self.kv_page_num,
            page_size=self.page_size,
            layer_num=self.layer_num,
            dtype=self.dtype,
            store_dtype=self.store_dtype,
            model_dtype=self.model_dtype,
            device=self.device,
            kv_cache_dim=self.kv_cache_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            nsa_kv_cache_store_fp8=self.nsa_kv_cache_store_fp8,
            offload_cache=self.offload_cache,
            start_layer=self.start_layer or 0,
            offload_cache_storage=global_server_args_dict.get("offload_cache_storage", "host"),
        )

    def get_kv_page_num(self):
        return self.buf_layout.get_page_num(self)

    def create_kv_buffer(self):
        buffers = self.buf_layout.alloc_buffers(self.make_buffer_spec())
        self.kv_buffer = buffers.kv_buffer
        self.buffer_device_types = buffers.buffer_device_types
        self.k_buffer_page_slices = buffers.k_buffer_page_slices
        self.v_buffer_page_slices = buffers.v_buffer_page_slices
        if buffers.empty_kr_cache is not None:
            self.empty_kr_cache = buffers.empty_kr_cache

    def get_page_size_bytes(self):
        return self.buf_layout.get_page_size_bytes(self)

    def init_kv_copy_and_warmup(self):
        if self.buf_layout.is_paged:
            raise NotImplementedError("mla_split_kv_kr模式KV buffer内存布局不同，暂未实现HiCache")

        stride_bytes = int(self.data_strides[0].item())
        if stride_bytes >= _KV_COPY_STRIDE_THRESHOLD_LARGE:
            bytes_per_tile = _KV_COPY_TILE_SIZE_LARGE
        elif stride_bytes >= _KV_COPY_STRIDE_THRESHOLD_MEDIUM:
            bytes_per_tile = _KV_COPY_TILE_SIZE_MEDIUM
        else:
            bytes_per_tile = _KV_COPY_TILE_SIZE_SMALL

        self._kv_copy_config = {
            "bytes_per_tile": bytes_per_tile,
            "byte_tiles": (stride_bytes + bytes_per_tile - 1) // bytes_per_tile,
            "num_warps": (_KV_COPY_NUM_WARPS_SMALL_TILE if bytes_per_tile <= _KV_COPY_TILE_SIZE_MEDIUM else _KV_COPY_NUM_WARPS_LARGE_TILE),
        }

        dummy_loc = torch.zeros(1, dtype=torch.int32, device=self.device)
        grid = (self.data_ptrs.numel(), self._kv_copy_config["byte_tiles"])

        copy_all_layer_kv_cache_tiled[grid](
            self.data_ptrs,
            self.data_strides,
            dummy_loc,
            dummy_loc,
            1,
            1,
            BYTES_PER_TILE=self._kv_copy_config["bytes_per_tile"],
            num_warps=self._kv_copy_config["num_warps"],
            num_stages=2,
        )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        if self.buf_layout.is_paged:
            raise NotImplementedError("mla_split_kv_kr模式KV buffer内存布局不同，暂未实现HiCache")
        if self._kv_copy_config is None:
            if tgt_loc.numel() == 0:
                return

            tgt_loc_flat = tgt_loc.view(-1).long()
            src_loc_flat = src_loc.view(-1).long()

            self.buf_layout.move_kv_cache_eager(self, tgt_loc_flat, src_loc_flat)
        else:
            grid = (self.data_ptrs.numel(), self._kv_copy_config["byte_tiles"])
            copy_all_layer_kv_cache_tiled[grid](
                self.data_ptrs,
                self.data_strides,
                tgt_loc,
                src_loc,
                tgt_loc.numel(),
                tgt_loc.numel(),
                BYTES_PER_TILE=self._kv_copy_config["bytes_per_tile"],
                num_warps=self._kv_copy_config["num_warps"],
                num_stages=2,
            )

    def get_kv_size_bytes(self):
        assert hasattr(self, "kv_buffer")
        kv_size_bytes = 0
        for kv_cache in self.kv_buffer:
            kv_size_bytes += get_tensor_size_bytes(kv_cache)
        return kv_size_bytes

    def get_contiguous_buf_infos(self):
        return self.buf_layout.get_contiguous_buf_infos(self)

    def get_buffer_device_types(self):
        """Registration locations, separate from the three-buffer-info API."""
        return None if self.buffer_device_types is None else list(self.buffer_device_types)

    def get_layerwise_buf_info_offsets(self, start_idx=0):
        return self.buf_layout.get_layerwise_buf_info_offsets(self, start_idx)

    def get_key_buffer(self, layer_id: int):
        layer_id = layer_id - self.start_layer
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id)
        return self.buf_layout.get_key_buffer(self, layer_id)

    def get_key_split_contiguous(self, layer_id: int, indices: torch.Tensor):
        layer_id = layer_id - self.start_layer
        return self.buf_layout.get_key_split_contiguous(self, layer_id, indices)

    def get_value_buffer(self, layer_id: int):
        layer_id = layer_id - self.start_layer
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id)
        return self.buf_layout.get_value_buffer(self, layer_id)

    def get_kv_buffer(self, layer_id: int):
        layer_id = layer_id - self.start_layer
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id)
        return self.buf_layout.get_kv_buffer(self, layer_id)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        if not self.buf_layout.allow_public_kv_setters:
            self.buf_layout.raise_unsupported_error("set_kv_buffer")
        if ENV.npu_enable_mla_split_kv_kr:
            raise RuntimeError("set_kv_buffer is not implemented when enable npu_enable_mla_split_kv_kr.")
        layer_id = layer.layer_id - self.start_layer
        if self.quant_method == "per_token_head":
            if not is_npu() and is_sm90_supported():
                quantize_and_cache_k(
                    key=cache_k.contiguous(),
                    k_lora_cache=self.kv_buffer[layer_id][0],
                    k_lora_scale_cache=self.kv_buffer[layer_id][1],
                    k_rope_cache=self.kv_buffer[layer_id][2],
                    indices=loc.to(torch.int32),
                    head_dim_v=self.kv_lora_rank,
                )
            else:
                k_lora = cache_k[..., : self.kv_lora_rank].float()
                k_rope = cache_k[..., self.kv_lora_rank :].float()
                scale = k_lora.abs().amax(dim=-1, keepdim=True).clamp(1e-26) / 448.0
                k_lora = (k_lora / scale).to(torch.float8_e4m3fn)
                k_rope = (k_rope / scale).to(self.model_dtype)
                self.kv_buffer[layer_id][0][loc] = k_lora.view(self.store_dtype)
                self.kv_buffer[layer_id][1][loc] = scale
                self.kv_buffer[layer_id][2][loc] = k_rope
        else:
            if cache_k.dtype != self.dtype:
                cache_k = cache_k.to(self.dtype)
            if self.store_dtype != self.dtype:
                self.kv_buffer[layer_id][loc] = cache_k.view(self.store_dtype)
            else:
                self.kv_buffer[layer_id][loc] = cache_k

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        if not self.buf_layout.allow_public_kv_setters:
            self.buf_layout.raise_unsupported_error("set_mla_kv_buffer")
        if ENV.npu_enable_mla_split_kv_kr:
            raise RuntimeError("set_mla_kv_buffer is not implemented when enable npu_enable_mla_split_kv_kr.")
        layer_id = layer.layer_id - self.start_layer
        if self.use_dsa and self.nsa_kv_cache_store_fp8:
            cache_k = torch.cat([cache_k_nope, cache_k_rope], dim=-1)
            cache_k = quantize_k_cache(cache_k.unsqueeze(1)).squeeze(1)
            cache_k = cache_k.view(self.store_dtype)
            enable_hicache = global_server_args_dict.get("enable_hierarchical_cache", False)
            if enable_hicache:
                buff = self.kv_buffer[layer_id]
                buff.view(-1, *buff.shape[2:])[loc] = cache_k
            else:
                self.kv_buffer[layer_id][loc] = cache_k
        elif self.quant_method == "per_token_head":
            if not is_npu() and is_sm90_supported():
                cache_k_combined = torch.cat([cache_k_nope, cache_k_rope], dim=-1)
                quantize_and_cache_k(
                    key=cache_k_combined.contiguous(),
                    k_lora_cache=self.kv_buffer[layer_id][0],
                    k_lora_scale_cache=self.kv_buffer[layer_id][1],
                    k_rope_cache=self.kv_buffer[layer_id][2],
                    indices=loc.to(torch.int32),
                    head_dim_v=self.kv_lora_rank,
                )
            else:
                k_lora = cache_k_nope.float()
                k_rope = cache_k_rope.float()
                scale = k_lora.abs().amax(dim=-1, keepdim=True).clamp(1e-26) / 448.0
                k_lora = (k_lora / scale).to(torch.float8_e4m3fn)
                k_rope = (k_rope / scale).to(self.model_dtype)
                self.kv_buffer[layer_id][0][loc] = k_lora.view(self.store_dtype)
                self.kv_buffer[layer_id][1][loc] = scale
                self.kv_buffer[layer_id][2][loc] = k_rope
        else:
            if cache_k_nope.dtype != self.dtype:
                cache_k_nope = cache_k_nope.to(self.dtype)
                cache_k_rope = cache_k_rope.to(self.dtype)
            if self.store_dtype != self.dtype:
                cache_k_nope = cache_k_nope.view(self.store_dtype)
                cache_k_rope = cache_k_rope.view(self.store_dtype)

            set_mla_kv_buffer_triton(self.kv_buffer[layer_id], loc, cache_k_nope, cache_k_rope)

    def sync_device(self):
        if not __is_npu__:
            torch.cuda.synchronize()
        else:
            torch.npu.synchronize()

    def get_md5(self, kv_cache_cpu):
        md5_hash = hashlib.md5(usedforsecurity=False)
        for layer_data in kv_cache_cpu:
            for chunk_data in layer_data:
                for tensor in chunk_data:
                    tensor_bytes = tensor.view(torch.int8).cpu().numpy().tobytes()
                    md5_hash.update(tensor_bytes)

        return md5_hash.hexdigest()

    def npu_get_cpu_copy(self, local_page_indices: torch.Tensor):
        if not self.buf_layout.requires_npu_page_copy:
            raise RuntimeError("MLATokenToKVPool.npu_get_cpu_copy requires page-based KV layout.")
        self.sync_device()
        kv_cache_cpu = []
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            for i in range(0, len(local_page_indices), self.offload_chunk_page_num):
                chunk_page_indices = local_page_indices[i : i + self.offload_chunk_page_num]
                kv_cache_cpu[-1].append(self.buf_layout.npu_get_cpu_copy_layer(self, layer_id, chunk_page_indices))
        self.sync_device()
        return kv_cache_cpu

    def npu_load_cpu_copy(
        self,
        kv_cache_cpu,
        local_page_mask: torch.Tensor,
        local_page_mask_cpu: torch.Tensor,
        dense_local_page_indices: torch.Tensor,
        page_indices: torch.Tensor,
    ):
        if not self.buf_layout.requires_npu_page_copy:
            raise RuntimeError("MLATokenToKVPool.npu_load_cpu_copy requires page-based KV layout.")
        self.sync_device()
        for layer_id in range(self.layer_num):
            for i in range(0, len(page_indices), self.offload_chunk_page_num):
                chunk_page_mask_cpu = local_page_mask_cpu[i : i + self.offload_chunk_page_num]
                if not chunk_page_mask_cpu.any():
                    continue
                chunk_page_mask = local_page_mask[i : i + self.offload_chunk_page_num]
                chunk_local_page_indices = dense_local_page_indices[i : i + self.offload_chunk_page_num][chunk_page_mask]
                self.buf_layout.npu_load_cpu_copy_layer(
                    self,
                    layer_id,
                    kv_cache_cpu,
                    i // self.offload_chunk_page_num,
                    chunk_page_mask_cpu,
                    chunk_local_page_indices,
                )
        self.sync_device()

    def get_cpu_copy(self, token_indices: list) -> torch.Tensor:
        self.sync_device()
        kv_cache_cpu = []
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            if not __is_npu__:
                for i in range(0, len(token_indices), self.offload_chunk_page_num):
                    chunk_indices = token_indices[i : i + self.offload_chunk_page_num]
                    if self.quant_method == "per_token_head":
                        if ENV.npu_enable_mla_split_kv_kr:
                            raise RuntimeError("per_token_head is not implemented when enable npu_enable_mla_split_kv_kr.")
                        kv_cache_cpu[-1].append([buffer[chunk_indices].to("cpu", non_blocking=True) for buffer in self.kv_buffer[layer_id]])
                    else:
                        kv_cpu = self.kv_buffer[layer_id][chunk_indices].to("cpu", non_blocking=True)
                        kv_cache_cpu[-1].append([kv_cpu])
            else:
                page_indices = token_indices[:: self.page_size] // self.page_size
                kv_cache_cpu[-1].append(self.buf_layout.get_cpu_copy_layer(self, layer_id, page_indices))
        self.sync_device()
        return kv_cache_cpu

    def load_cpu_copy(
        self,
        kv_cache_cpu: torch.Tensor,
        token_indices: list,
        prefix_page_count: int = 0,
    ) -> None:
        del prefix_page_count

        self.sync_device()
        for layer_id in range(self.layer_num):
            if not __is_npu__:
                for i in range(0, len(token_indices), self.offload_chunk_page_num):
                    chunk_indices = token_indices[i : i + self.offload_chunk_page_num]
                    if self.quant_method == "per_token_head":
                        if ENV.npu_enable_mla_split_kv_kr:
                            raise RuntimeError("per_token_head is not implemented when enable npu_enable_mla_split_kv_kr.")
                        for j in range(3):
                            t = kv_cache_cpu[layer_id][i // self.offload_chunk_page_num][j]
                            assert t.shape[0] == len(chunk_indices)
                            self.kv_buffer[layer_id][j][chunk_indices] = t.to(self.kv_buffer[0][0].device, non_blocking=True)
                    else:
                        if ENV.npu_enable_mla_split_kv_kr:
                            kv_cpu1 = kv_cache_cpu[layer_id][i // self.offload_chunk_page_num][0]
                            kv_cpu2 = kv_cache_cpu[layer_id][i // self.offload_chunk_page_num][1]
                            assert kv_cpu1.shape[0] == len(chunk_indices) and kv_cpu2.shape[0] == len(chunk_indices), (
                                f"kv_cpu1.shape[0] {kv_cpu1.shape[0]} or kv_cpu2.shape[0] {kv_cpu2.shape[0]} "
                                f"!= len(chunk_indices) {len(chunk_indices)}"
                            )
                            kv_chunk1 = kv_cpu1.to(self.kv_buffer[0].device, non_blocking=True)
                            kv_chunk2 = kv_cpu2.to(self.kv_buffer[0].device, non_blocking=True)
                            self.kv_buffer[2 * layer_id].view(-1, 1, self.kv_lora_rank)[chunk_indices] = kv_chunk1
                            self.kv_buffer[2 * layer_id + 1].view(-1, 1, self.qk_rope_head_dim)[chunk_indices] = kv_chunk2
                        else:
                            kv_cpu = kv_cache_cpu[layer_id][i // self.offload_chunk_page_num][0]
                            assert kv_cpu.shape[0] == len(chunk_indices), (
                                f"kv_cpu.shape[0] {kv_cpu.shape[0]} != len(chunk_indices) {len(chunk_indices)}"
                            )
                            kv_chunk = kv_cpu.to(self.kv_buffer[0].device, non_blocking=True)
                            self.kv_buffer[layer_id][chunk_indices] = kv_chunk
            else:
                page_indices = token_indices[:: self.page_size] // self.page_size
                self.buf_layout.load_cpu_copy_layer(self, layer_id, kv_cache_cpu, page_indices)

        self.sync_device()
