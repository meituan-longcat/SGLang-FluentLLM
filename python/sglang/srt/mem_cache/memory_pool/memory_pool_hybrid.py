from __future__ import annotations

from contextlib import contextmanager

import torch

from sglang.srt.layers.attention.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool.memory_pool_base import BaseTokenToKVPool
from sglang.srt.mem_cache.memory_pool.memory_pool_dsa import DSATokenToKVPool
from sglang.srt.mem_cache.memory_pool.memory_pool_kernel import GB
from sglang.srt.mem_cache.memory_pool.memory_pool_mha import MHATokenToKVPool
from sglang.srt.mem_cache.memory_pool.memory_pool_mla import MLATokenToKVPool
from sglang.srt.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


class HybridLinearKVPool(BaseTokenToKVPool):
    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        full_attention_layer_ids: list[int],
        enable_kvcache_transpose: bool,
        device: str,
        max_batch_size: int,
        max_context_len: int,
        page_size: int,
        rank: int,
        model_dtype: torch.dtype = torch.bfloat16,
        quant_method: str = "",
        use_mla: bool = False,
        kv_lora_rank: int = None,
        qk_rope_head_dim: int = None,
        use_dsa: bool = False,
        index_head_dim: int = None,
        index_dtype: torch.dtype = None,
        npu_dsa_kv_int8: bool = False,
        npu_dsa_packed_kv_row_bytes: int = 656,
    ):
        self.size = size
        self.dtype = dtype
        self.device = device
        self.full_layer_nums = len(full_attention_layer_ids)
        self.page_size = page_size if use_dsa else 1
        self.use_mla = use_mla
        self.use_dsa = use_dsa
        assert not enable_kvcache_transpose

        if use_dsa:
            self.full_kv_pool = DSATokenToKVPool(
                size=size,
                model_dtype=model_dtype,
                dtype=dtype,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                layer_num=self.full_layer_nums,
                device=device,
                enable_memory_saver=False,
                max_batch_size=max_batch_size,
                max_context_len=max_context_len,
                page_size=self.page_size,
                rank=rank,
                index_head_dim=index_head_dim,
                index_dtype=index_dtype,
                npu_dsa_kv_int8=npu_dsa_kv_int8,
                npu_dsa_packed_kv_row_bytes=npu_dsa_packed_kv_row_bytes,
                start_layer=0,
                end_layer=self.full_layer_nums,
            )
            self.index_dtype = self.full_kv_pool.index_dtype
            self.index_head_dim = self.full_kv_pool.index_head_dim
            self.kv_cache_dim = self.full_kv_pool.kv_cache_dim
        elif not use_mla:
            self.full_kv_pool = MHATokenToKVPool(
                size=size,
                page_size=self.page_size,
                dtype=dtype,
                head_num=head_num,
                head_dim=head_dim,
                layer_num=self.full_layer_nums,
                device=device,
                enable_memory_saver=False,
                max_batch_size=max_batch_size,
                max_context_len=max_context_len,
                rank=rank,
            )
        else:
            self.full_kv_pool = MLATokenToKVPool(
                size=size,
                page_size=self.page_size,
                dtype=dtype,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                layer_num=self.full_layer_nums,
                device=device,
                enable_memory_saver=False,
                max_batch_size=max_batch_size,
                max_context_len=max_context_len,
                rank=rank,
                model_dtype=model_dtype,
                quant_method=quant_method,
                start_layer=0,
                end_layer=self.full_layer_nums,
            )
        self.full_attention_layer_id_mapping = {id: i for i, id in enumerate(full_attention_layer_ids)}
        if use_mla or use_dsa:
            self.mem_usage = self.get_kv_size_bytes() / GB
        else:
            k_size, v_size = self.get_kv_size_bytes()
            self.mem_usage = (k_size + v_size) / GB

    def get_kv_size_bytes(self):
        return self.full_kv_pool.get_kv_size_bytes()

    @property
    def layer_num(self):
        return self.full_kv_pool.layer_num

    @property
    def is_mla_backend(self) -> bool:
        return self.use_mla or self.use_dsa

    def get_contiguous_buf_infos(self):
        return self.full_kv_pool.get_contiguous_buf_infos()

    def get_layerwise_buf_info_offsets(self, start_idx=0):
        return self.full_kv_pool.get_layerwise_buf_info_offsets(start_idx)

    def transfer_full_attention_id(self, layer_id: int):
        if layer_id not in self.full_attention_layer_id_mapping:
            raise ValueError(f"{layer_id=} not in full attention layers: {self.full_attention_layer_id_mapping.keys()}")
        return self.full_attention_layer_id_mapping[layer_id]

    def get_key_buffer(self, layer_id: int):
        layer_id = self.transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_key_buffer(layer_id)

    def get_key_split_contiguous(self, layer_id: int, indices: torch.Tensor):
        layer_id = self.transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_key_split_contiguous(layer_id, indices)

    def get_value_buffer(self, layer_id: int):
        layer_id = self.transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_value_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int):
        layer_id = self.transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_kv_buffer(layer_id)

    @contextmanager
    def transfer_id_context(self, layer: RadixAttention):

        @contextmanager
        def _patch_layer_id(layer):
            original_layer_id = layer.layer_id
            layer.layer_id = self.transfer_full_attention_id(layer.layer_id)
            try:
                yield
            finally:
                layer.layer_id = original_layer_id

        with _patch_layer_id(layer):
            yield

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ):
        layer_id = self.transfer_full_attention_id(layer.layer_id)
        if not self.use_mla and not self.use_dsa:
            self.full_kv_pool.set_kv_buffer(
                None,
                loc,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override=layer_id,
            )
        else:
            with self.transfer_id_context(layer):
                self.full_kv_pool.set_kv_buffer(
                    layer,
                    loc,
                    cache_k,
                    cache_v,
                )

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        assert self.use_mla, "set_mla_kv_buffer called when use_mla is False"
        with self.transfer_id_context(layer):
            self.full_kv_pool.set_mla_kv_buffer(layer, loc, cache_k_nope, cache_k_rope)

    def get_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        dst_dtype: torch.dtype | None = None,
    ):
        assert self.use_mla, "get_mla_kv_buffer called when use_mla is False"
        with self.transfer_id_context(layer):
            return self.full_kv_pool.get_mla_kv_buffer(layer, loc, dst_dtype)

    def get_index_k_with_scale_buffer(self, layer_id: int):
        assert self.use_dsa
        return self.full_kv_pool.get_index_k_with_scale_buffer(self.transfer_full_attention_id(layer_id))

    def get_index_k_continuous(self, layer_id: int, seq_len: int, page_indices: torch.Tensor):
        assert self.use_dsa
        return self.full_kv_pool.get_index_k_continuous(self.transfer_full_attention_id(layer_id), seq_len, page_indices)

    def get_index_k_scale_continuous(self, layer_id: int, seq_len: int, page_indices: torch.Tensor):
        assert self.use_dsa
        return self.full_kv_pool.get_index_k_scale_continuous(self.transfer_full_attention_id(layer_id), seq_len, page_indices)

    def get_index_k_flat_idxs_by_token_idxs(self, layer_id: int, seq_len: int, token_idxs: torch.Tensor):
        assert self.use_dsa
        return self.full_kv_pool.get_index_k_flat_idxs_by_token_idxs(self.transfer_full_attention_id(layer_id), seq_len, token_idxs)

    def get_index_k_by_token_idxs(self, layer_id: int, seq_len: int, token_idxs: torch.Tensor):
        assert self.use_dsa
        return self.full_kv_pool.get_index_k_by_token_idxs(self.transfer_full_attention_id(layer_id), seq_len, token_idxs)

    def set_index_k_and_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor = None,
    ) -> None:
        assert self.use_dsa
        self.full_kv_pool.set_index_k_and_scale_buffer(
            self.transfer_full_attention_id(layer_id),
            loc,
            index_k,
            index_k_scale,
        )
