from __future__ import annotations

import numpy as np
import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache.memory_pool.memory_pool_kernel import GB
from sglang.srt.mem_cache.memory_pool.memory_pool_mha import MHATokenToKVPool
from sglang.srt.mem_cache.memory_pool.memory_pool_mla import MLATokenToKVPool
from sglang.srt.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


class NativeSparseMHATokenToKVPool(MHATokenToKVPool):
    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        compressed_block_stride: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        max_batch_size: int,
        max_context_len: int,
        page_size: int,
        rank: int,
    ):
        super().__init__(
            size=size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            max_batch_size=max_batch_size,
            max_context_len=max_context_len,
            page_size=page_size,
            rank=rank,
        )
        assert page_size % compressed_block_stride == 0, (
            f"The page size should be divisible by the kernel stride. Page size: {page_size}, compressed_block_stride: {compressed_block_stride}"
        )
        assert self.store_dtype == self.dtype
        self.compressed_block_stride = compressed_block_stride
        self.create_compressed_buffers()

        compressed_k_size, compressed_v_size = self.get_compressed_kv_size_bytes()
        logger.info(f"Compressed KV Cache is allocated. K size: {compressed_k_size / GB:.2f} GB, V size: {compressed_v_size / GB:.2f} GB.")

    def create_compressed_buffers(self):
        compress_block_size = (self.size + self.page_size) // self.compressed_block_stride
        with self.memory_saver_adapter.region(tag=GPU_MEMORY_TYPE_KV_CACHE):
            self.compressed_k_buffer = [
                torch.empty(
                    (compress_block_size, self.head_num, self.head_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]
            self.compressed_v_buffer = [
                torch.empty(
                    (compress_block_size, self.head_num, self.head_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]

    def clear_compressed_buffers(self):
        del self.compressed_k_buffer
        del self.compressed_v_buffer

    def get_compressed_kv_size_bytes(self):
        assert hasattr(self, "compressed_k_buffer")
        assert hasattr(self, "compressed_v_buffer")
        compressed_k_size_bytes = 0
        for compressed_k_cache in self.compressed_k_buffer:
            compressed_k_size_bytes += np.prod(compressed_k_cache.shape) * compressed_k_cache.dtype.itemsize
        compressed_v_size_bytes = 0
        for compressed_v_cache in self.compressed_v_buffer:
            compressed_v_size_bytes += np.prod(compressed_v_cache.shape) * compressed_v_cache.dtype.itemsize
        return compressed_k_size_bytes, compressed_v_size_bytes

    def get_compressed_key_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.compressed_k_buffer[layer_id].view(self.dtype)
        return self.compressed_k_buffer[layer_id]

    def get_compressed_value_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.compressed_v_buffer[layer_id].view(self.dtype)
        return self.compressed_v_buffer[layer_id]

    def get_compressed_kv_buffer(self, layer_id: int):
        return self.get_compressed_key_buffer(layer_id), self.get_compressed_value_buffer(layer_id)

    def get_contiguous_buf_infos(self):
        raise NotImplementedError

    def get_flat_data(self, indices):
        raise NotImplementedError

    def transfer(self, indices, flat_data):
        raise NotImplementedError


class NativeSparseMLATokenToKVPool(MLATokenToKVPool):
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
        compressed_block_stride: int,
    ):
        super().__init__(
            size=size,
            model_dtype=model_dtype,
            dtype=dtype,
            quant_method=quant_method,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            max_batch_size=max_batch_size,
            max_context_len=max_context_len,
            page_size=page_size,
            rank=rank,
        )
        assert page_size % compressed_block_stride == 0, (
            f"The page size should be divisible by the kernel stride. Page size: {page_size}, compressed_block_stride: {compressed_block_stride}"
        )
        assert self.store_dtype == self.dtype
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.compressed_block_stride = compressed_block_stride
        self.create_compressed_buffers()

        compressed_kv_size = self.get_compressed_kv_size_bytes()
        logger.info(f"Compressed KV Cache is allocated. latent KV size: {compressed_kv_size / GB:.2f} GB.")

    def create_compressed_buffers(self):
        compress_block_size = (self.size + self.page_size) // self.compressed_block_stride
        with self.memory_saver_adapter.region(tag=GPU_MEMORY_TYPE_KV_CACHE):
            self.compressed_kv_buffer = [
                torch.empty(
                    (compress_block_size, 1, self.kv_lora_rank + self.qk_rope_head_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]

    def clear_compressed_buffers(self):
        del self.compressed_kv_buffer

    def get_compressed_kv_size_bytes(self):
        assert hasattr(self, "compressed_kv_buffer")
        compressed_kv_size_bytes = 0
        for compressed_kv_cache in self.compressed_kv_buffer:
            compressed_kv_size_bytes += np.prod(compressed_kv_cache.shape) * compressed_kv_cache.dtype.itemsize
        return compressed_kv_size_bytes

    def get_compressed_key_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.compressed_kv_buffer[layer_id].view(self.dtype)
        return self.compressed_kv_buffer[layer_id]

    def get_compressed_value_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.compressed_kv_buffer[layer_id][..., : self.kv_lora_rank].view(self.dtype)
        return self.compressed_kv_buffer[layer_id][..., : self.kv_lora_rank]

    def get_compressed_kv_buffer(self, layer_id: int):
        return self.get_compressed_key_buffer(layer_id), self.get_compressed_value_buffer(layer_id)
