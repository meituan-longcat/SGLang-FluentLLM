from __future__ import annotations

import torch

from sglang.srt.mem_cache.memory_pool.host_memory_pool import HostTokenToKVPool


class HostTokenToKVPoolL15Wrapper:
    def __init__(self, inner: HostTokenToKVPool):
        self._inner = inner
        self.enable_l15 = True
        self.group_mask = inner.group_mask
        self.group_shift = inner.group_shift
        self.kvp_rank = inner.kvp_rank
        self.page_size = inner.page_size
        self.attn_tp_group_size = inner.group_mask + 1

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def get_ranks_from_global_pages(self, global_pages: torch.Tensor) -> list:
        return (global_pages & self.group_mask).tolist()

    def pages_to_tokens(self, local_pages) -> torch.Tensor:
        return self._inner.pages_to_tokens(local_pages)

    def global_pages_to_local_page_indices(self, global_pages) -> torch.Tensor:
        local_page_mask = (global_pages & self.group_mask) == self.kvp_rank
        local_pages = global_pages[local_page_mask] >> self.group_shift
        return local_pages, local_page_mask

    def d2h_with_offset(
        self,
        host_token_indices,
        device_token_indices,
        device_pool,
        host_layer_offset,
        num_layers,
        io_backend,
    ):
        if host_token_indices.numel() == 0:
            return None
        return self._inner.d2h_with_offset(
            host_token_indices,
            device_token_indices,
            device_pool,
            host_layer_offset,
            num_layers,
            io_backend,
        )

    def h2d_layerwise_with_offset(
        self,
        host_token_indices,
        device_token_indices,
        device_pool,
        host_layer_id,
        device_layer_id,
        io_backend,
    ):
        self._inner.h2d_layerwise_with_offset(
            host_token_indices,
            device_token_indices,
            device_pool,
            host_layer_id,
            device_layer_id,
            io_backend,
        )

    def h2d_layerwise(self, host_token_indices, device_token_indices, layer_id, io_backend):
        self._inner.h2d_layerwise(host_token_indices, device_token_indices, layer_id, io_backend)
