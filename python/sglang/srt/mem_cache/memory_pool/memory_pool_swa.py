from __future__ import annotations

import torch

from sglang.srt.layers.attention.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool.memory_pool_base import BaseTokenToKVPool
from sglang.srt.mem_cache.memory_pool.memory_pool_kernel import GB
from sglang.srt.mem_cache.memory_pool.memory_pool_mha import MHATokenToKVPool
from sglang.srt.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


class SWAKVPool(BaseTokenToKVPool):
    def __init__(
        self,
        size: int,
        size_swa: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        swa_attention_layer_ids: list[int],
        full_attention_layer_ids: list[int],
        enable_kvcache_transpose: bool,
        device: str,
    ):
        self.size = size
        self.size_swa = size_swa
        self.dtype = dtype
        self.device = device
        self.swa_layer_nums = len(swa_attention_layer_ids)
        self.full_layer_nums = len(full_attention_layer_ids)
        self.page_size = 1
        assert not enable_kvcache_transpose
        TokenToKVPoolClass = MHATokenToKVPool
        self.swa_kv_pool = TokenToKVPoolClass(
            size=size_swa,
            page_size=self.page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=self.swa_layer_nums,
            device=device,
            enable_memory_saver=False,
        )
        self.full_kv_pool = TokenToKVPoolClass(
            size=size,
            page_size=self.page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=self.full_layer_nums,
            device=device,
            enable_memory_saver=False,
        )
        self.layers_mapping: dict[int, tuple[int, bool]] = {}
        for full_attn_layer_id, global_layer_id in enumerate(full_attention_layer_ids):
            self.layers_mapping[global_layer_id] = (full_attn_layer_id, False)
        for swa_layer_id, global_layer_id in enumerate(swa_attention_layer_ids):
            self.layers_mapping[global_layer_id] = (swa_layer_id, True)
        self.full_to_swa_index_mapping: torch.Tensor | None = None

        k_size, v_size = self.get_kv_size_bytes()
        self.mem_usage = (k_size + v_size) / GB

    def get_kv_size_bytes(self):
        k_size, v_size = self.full_kv_pool.get_kv_size_bytes()
        k_size_swa, v_size_swa = self.swa_kv_pool.get_kv_size_bytes()
        return k_size + k_size_swa, v_size + v_size_swa

    def get_contiguous_buf_infos(self):
        full_kv_data_ptrs, full_kv_data_lens, full_kv_item_lens = self.full_kv_pool.get_contiguous_buf_infos()
        swa_kv_data_ptrs, swa_kv_data_lens, swa_kv_item_lens = self.swa_kv_pool.get_contiguous_buf_infos()

        kv_data_ptrs = full_kv_data_ptrs + swa_kv_data_ptrs
        kv_data_lens = full_kv_data_lens + swa_kv_data_lens
        kv_item_lens = full_kv_item_lens + swa_kv_item_lens

        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def get_key_buffer(self, layer_id: int):
        layer_id_pool, is_swa = self.layers_mapping[layer_id]
        if is_swa:
            return self.swa_kv_pool.get_key_buffer(layer_id_pool)
        else:
            return self.full_kv_pool.get_key_buffer(layer_id_pool)

    def get_value_buffer(self, layer_id: int):
        layer_id_pool, is_swa = self.layers_mapping[layer_id]
        if is_swa:
            return self.swa_kv_pool.get_value_buffer(layer_id_pool)
        else:
            return self.full_kv_pool.get_value_buffer(layer_id_pool)

    def get_kv_buffer(self, layer_id: int):
        layer_id_pool, is_swa = self.layers_mapping[layer_id]
        if is_swa:
            return self.swa_kv_pool.get_kv_buffer(layer_id_pool)
        else:
            return self.full_kv_pool.get_kv_buffer(layer_id_pool)

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        assert self.full_to_swa_index_mapping is not None
        return self.full_to_swa_index_mapping[kv_indices].to(torch.int32)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ):
        layer_id = layer.layer_id
        layer_id_pool, is_swa = self.layers_mapping[layer_id]
        if is_swa:
            if self.full_to_swa_index_mapping is not None:
                loc = self.translate_loc_from_full_to_swa(loc)
            self.swa_kv_pool.set_kv_buffer(
                None,
                loc,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override=layer_id_pool,
            )
        else:
            self.full_kv_pool.set_kv_buffer(
                None,
                loc,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override=layer_id_pool,
            )
