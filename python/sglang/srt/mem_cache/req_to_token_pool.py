from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache.memory_pool.memory_pool_base import ReqToTokenPoolInfo
from sglang.srt.mem_cache.memory_pool.memory_pool_mamba import MambaPool
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    from sglang.srt.managers.req import Req


class ReqToTokenPool:
    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
    ):
        memory_saver_adapter = TorchMemorySaverAdapter.create(enable=enable_memory_saver)

        self.size = size
        self.max_context_len = max_context_len
        self.device = device
        with memory_saver_adapter.region(tag=GPU_MEMORY_TYPE_KV_CACHE):
            self.req_to_token = torch.zeros((size, max_context_len), dtype=torch.int32, device=device)
            self.verified_lens = torch.zeros(size, dtype=torch.int32, device=device)
            self.alloced_lens = torch.zeros(size, dtype=torch.int32, device=device)
        self.free_slots = list(range(size))[1:]

    def get_req_pool_info(self, req_pool_idx: int):
        alloced_len = self.alloced_lens[req_pool_idx].item()
        return ReqToTokenPoolInfo(
            self.verified_lens[req_pool_idx].item(),
            alloced_len,
            self.req_to_token[req_pool_idx, :alloced_len].clone(),
        )

    def set_req_pool_info(self, req_pool_idx: int, metadata: ReqToTokenPoolInfo):
        self.verified_lens[req_pool_idx] = metadata.verified_len
        self.alloced_lens[req_pool_idx] = metadata.alloced_len
        self.req_to_token[req_pool_idx, : metadata.alloced_len] = metadata.alloced_slots

    def write(self, indices, values):
        self.req_to_token[indices] = values

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, need_size: int) -> list[int]:
        if need_size > len(self.free_slots):
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        self.verified_lens[select_index] = 0
        self.alloced_lens[select_index] = 0

        return select_index

    def free(self, free_index: int | list[int]):
        if isinstance(free_index, (int,)):
            self.free_slots.append(free_index)
            self.verified_lens[free_index] = 0
            self.alloced_lens[free_index] = 0
        else:
            self.free_slots.extend(free_index)
            for index in free_index:
                self.verified_lens[index] = 0
                self.alloced_lens[index] = 0

    def clear(self):
        self.free_slots = list(range(self.size))[1:]
        self.verified_lens.zero_()
        self.alloced_lens.zero_()


class HybridReqToTokenPool(ReqToTokenPool):
    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        conv_dtype: torch.dtype,
        ssm_dtype: torch.dtype,
        mamba_layers: list[int],
        conv_state_shape: tuple[int, int],
        temporal_state_shape: tuple[int, int],
        speculative_num_draft_tokens: int,
    ):
        super().__init__(
            size=size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=enable_memory_saver,
        )

        self.mamba_pool = MambaPool(
            size,
            conv_dtype,
            ssm_dtype,
            len(mamba_layers),
            conv_state_shape,
            temporal_state_shape,
            device,
            speculative_num_draft_tokens,
        )
        self.mamba_map = {layer_id: i for i, layer_id in enumerate(mamba_layers)}

        self.device = device
        self.req_index_to_mamba_index_mapping: torch.Tensor = torch.empty(size, dtype=torch.int32, device=self.device)

        self.rid_to_mamba_index_mapping: dict[str, int] = {}
        self.mamba_index_to_rid_mapping: dict[int, str] = {}

    def alloc(self, need_size: int, reqs: list[Req] | None = None) -> list[int] | None:
        select_index = super().alloc(need_size)
        if select_index is None:
            return None

        req_index = select_index[:]
        mamba_index = []
        for i, req in enumerate(reqs):
            if req.req_pool_idx is not None:
                req_index.insert(i, req.req_pool_idx)
            rid = req.rid
            if rid in self.rid_to_mamba_index_mapping:
                mid = self.rid_to_mamba_index_mapping[rid]
            elif (mid := self.mamba_pool.alloc(1)) is not None:
                mid = mid[0]
                self.rid_to_mamba_index_mapping[rid] = mid
                self.mamba_index_to_rid_mapping[mid] = rid
            mamba_index.append(mid)
        assert len(req_index) == len(mamba_index), "Not enough space for mamba cache, try to increase --max-mamba-cache-size."
        self.req_index_to_mamba_index_mapping[req_index] = torch.tensor(mamba_index, dtype=torch.int32, device=self.device)
        return select_index

    def get_mamba_indices(self, req_indices: torch.Tensor) -> torch.Tensor:
        return self.req_index_to_mamba_index_mapping[req_indices]

    def get_mamba_params(self, layer_id: int):
        assert layer_id in self.mamba_map
        return self.mamba_pool.get_mamba_params(self.mamba_map[layer_id])

    def get_mamba_params_all_layers(self):
        return self.mamba_pool.get_mamba_params_all_layers()

    def mamba2_layer_cache(self, layer_id: int):
        assert layer_id in self.mamba_map
        return self.mamba_pool.mamba2_layer_cache(self.mamba_map[layer_id])

    def free(self, free_index: int | list[int], free_mamba_cache: bool = True):
        super().free(free_index)
        if free_mamba_cache:
            mamba_index = self.req_index_to_mamba_index_mapping[free_index]
            mamba_index_list = mamba_index.tolist()
            if isinstance(mamba_index_list, int):
                mamba_index_list = [mamba_index_list]
            self.mamba_pool.free(mamba_index_list)
            for mid in mamba_index_list:
                rid = self.mamba_index_to_rid_mapping[mid]
                self.mamba_index_to_rid_mapping.pop(mid)
                self.rid_to_mamba_index_mapping.pop(rid)

    def clear(self):
        super().clear()
        self.mamba_pool.clear()
