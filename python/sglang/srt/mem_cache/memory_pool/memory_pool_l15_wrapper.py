from __future__ import annotations

import logging
import math

import numpy as np
import numpy.typing as npt
import torch
import torch.distributed as dist

from sglang.srt.disaggregation.utils import PageTransferMetadata
from sglang.srt.distributed import get_attn_tp_group
from sglang.srt.env import ENV, global_server_args_dict
from sglang.srt.layers.attention.dp_attention import get_attention_tp_group
from sglang.srt.utils import is_npu

logger = logging.getLogger(__name__)

__is_npu__ = is_npu()


def global_page_to_local_page(global_page_id, reserved_page_num: int, group_shift: int):
    return ((global_page_id - reserved_page_num) >> group_shift) + reserved_page_num


class TokenToKVPoolL15Wrapper:
    def __init__(self, inner):
        self._inner = inner

        self.attn_tp_group_size = global_server_args_dict["attn_tp_size"]
        # KVP also uses this wrapper to expose page ownership. Report the
        # actual transfer layout instead of claiming L1.5 unconditionally.
        self.enable_mla_l1_5_cache = bool(
            global_server_args_dict["enable_mla_l1_5_cache"]
        )

        self.page_shift = int(math.log(self._inner.page_size, 2))
        assert self._inner.page_size == (1 << self.page_shift), f"page size is expected to be the power of 2, but {self._inner.page_size=}"
        self.page_mask = self._inner.page_size - 1
        self.group_shift = int(math.log(self.attn_tp_group_size, 2))
        assert self.attn_tp_group_size == (1 << self.group_shift), f"attn tp size is expected to be the power of 2, but {self.attn_tp_group_size=}"
        self.group_mask = self.attn_tp_group_size - 1
        self.kvp_rank = get_attn_tp_group().rank_in_group
        self.reserved_page_num = self.get_reserved_page_num()

    def __getattr__(self, name):
        return getattr(self._inner, name)

    @property
    def is_mla_backend(self) -> bool:
        return self._inner.is_mla_backend

    def get_reserved_page_num(self) -> int:
        if global_server_args_dict["kvp_size"] > 1:
            return 1 + self._inner.max_batch_size
        return 1

    def get_rank_from_global_page_id(self, global_page_id):
        return (global_page_id - self.reserved_page_num) & self.group_mask

    def calc_local_pages_mask(self, global_page_indices):
        reserved_page_num = self.get_reserved_page_num()
        return (global_page_indices < reserved_page_num) | (self.get_rank_from_global_page_id(global_page_indices) == self.kvp_rank)

    def masked_global_pages_to_local_pages(self, masked_global_page_indices):
        reserved_page_num = self.get_reserved_page_num()
        lib = torch if isinstance(masked_global_page_indices, torch.Tensor) else np
        return lib.where(
            masked_global_page_indices < reserved_page_num,
            0,
            global_page_to_local_page(masked_global_page_indices, reserved_page_num, self.group_shift),
        )

    def select_local_pages_from_global_pages(self, global_page_indices):
        local_pages_mask = self.calc_local_pages_mask(global_page_indices)
        masked_global_page_indices = global_page_indices[local_pages_mask]
        local_page_indices = self.masked_global_pages_to_local_pages(masked_global_page_indices)
        return local_pages_mask, local_page_indices

    def get_local_pages_num_from_global_pages(self, global_page_indices) -> int:
        local_pages_mask = self.calc_local_pages_mask(global_page_indices)
        return int(local_pages_mask.sum().item())

    def get_ranks_from_global_pages(self, global_page_indices) -> list:
        return self.get_rank_from_global_page_id(global_page_indices).tolist()

    def select_local_token_indices(self, global_token_indices: torch.Tensor) -> torch.Tensor:
        """从 global device token indices 中筛选出属于本 rank 的 local token indices。"""
        global_page_indices = global_token_indices >> self.page_shift
        offsets = global_token_indices & self.page_mask
        local_mask, local_pages = self.select_local_pages_from_global_pages(global_page_indices)
        local_token_indices = (local_pages << self.page_shift) | offsets[local_mask]
        return local_token_indices

    def pages_to_tokens(self, local_pages) -> torch.Tensor:
        return self._inner.pages_to_tokens(local_pages)

    def global_pages_to_local_page_indices(self, global_pages) -> torch.Tensor:
        # skip masked_global_pages_to_local_pages because we never alloc [0, reserved_page_num) pages for l2 write
        # todo never use for other purpose
        local_pages_mask = self.get_rank_from_global_page_id(global_pages) == self.kvp_rank
        local_pages = global_page_to_local_page(global_pages[local_pages_mask], self.reserved_page_num, self.group_shift)
        return local_pages, local_pages_mask

    def remap_tokens_for_kvp(self, global_token_indices: torch.Tensor, return_rank_counts=True, return_perm=True):
        if global_token_indices.numel() == 0:
            empty = torch.empty(0, dtype=global_token_indices.dtype, device=global_token_indices.device)
            pages_mask = torch.empty(0, dtype=torch.bool, device=global_token_indices.device)
            rank_counts = [0 for _ in range(self.attn_tp_group_size)]
            perm = torch.empty(0, dtype=torch.int64, device=global_token_indices.device)
            inv_perm = torch.empty(0, dtype=torch.int64, device=global_token_indices.device)
            return pages_mask, empty, rank_counts, perm, inv_perm

        global_page_indices = global_token_indices >> self.page_shift
        offsets = global_token_indices & self.page_mask

        pages_mask, local_pages = self.select_local_pages_from_global_pages(global_page_indices)
        local_offsets = offsets[pages_mask]
        local_token_indices = (local_pages << self.page_shift) | local_offsets

        dense_local = global_token_indices.clone()
        dense_local[pages_mask] = local_token_indices
        dense_local.masked_fill_(~pages_mask, 0)
        global_token_indices.copy_(dense_local)

        if return_rank_counts or return_perm:
            target_rank = self.get_rank_from_global_page_id(global_page_indices).to(torch.int64)
        rank_counts = None
        if return_rank_counts:
            rank_counts = torch.bincount(
                target_rank,
                minlength=self.attn_tp_group_size,
            ).to(torch.int32)
            rank_counts = rank_counts.tolist()
            if local_token_indices.shape[0] != rank_counts[self.kvp_rank]:
                logger.warning(
                    f"local token count {local_token_indices.shape[0]} does not match expected "
                    f"count {rank_counts[self.kvp_rank]} for rank {self.kvp_rank}"
                )
        perm = None
        inv_perm = None
        if return_perm:
            perm = torch.argsort(target_rank.to(torch.float32), stable=True)
            # perm_backup = torch.argsort(target_rank, stable=True)
            # logger.info(f"{target_rank=} {perm_backup=} {perm=}")
            # assert (perm_backup == perm).all(), "perm_backup != perm"
            inv_perm = torch.empty_like(perm)
            inv_perm[perm] = torch.arange(global_token_indices.numel(), device=global_token_indices.device)

        return pages_mask, local_token_indices, rank_counts, perm, inv_perm

    def remap_tokens_for_kvp_local(self, global_token_indices: torch.Tensor):
        pages_mask, local_token_indices, _, _, _ = self.remap_tokens_for_kvp(global_token_indices, return_rank_counts=False, return_perm=False)
        return pages_mask, local_token_indices

    def remap_tokens_for_kvp_inplace(self, global_token_indices: torch.Tensor) -> None:
        global_page_indices = global_token_indices >> self.page_shift
        offsets = global_token_indices & self.page_mask
        pages_mask = self.calc_local_pages_mask(global_page_indices)
        all_local_pages = self.masked_global_pages_to_local_pages(global_page_indices)
        local_token_indices = (all_local_pages << self.page_shift) | offsets
        dense_local = torch.where(pages_mask, local_token_indices, torch.zeros_like(global_token_indices))
        global_token_indices.copy_(dense_local)

    def remap_token_idxs_for_gather(
        self,
        indices: torch.Tensor,
    ):
        reserved_pages = self.reserved_page_num
        pages = indices >> self.page_shift
        offsets = indices & self.page_mask
        mask = (pages >= reserved_pages) & (((pages - reserved_pages) & self.group_mask) == self.kvp_rank)
        new_pages = ((pages - reserved_pages) >> self.group_shift) + reserved_pages
        local_indices = (new_pages << self.page_shift) | offsets
        target_rank = (pages - reserved_pages) & self.group_mask
        rank_counts = torch.bincount(
            target_rank,
            minlength=self.attn_tp_group_size,
        ).to(torch.int32)
        perm = torch.argsort(target_rank, stable=True)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(indices.numel(), device=indices.device)
        local_indices = local_indices[mask]
        return local_indices, rank_counts.tolist(), inv_perm

    def remap_page_table_to_local(
        self,
        global_table: torch.Tensor,
        seqlens: torch.Tensor,
    ):
        reserved_pages = self.reserved_page_num
        device = global_table.device
        bs, max_page_num = global_table.shape
        page_size = self._inner.page_size
        page_nums = (seqlens + page_size - 1) // page_size
        local_table = global_table.new_zeros(bs, max_page_num + 1)
        mask = (global_table >= reserved_pages) & (self.get_rank_from_global_page_id(global_table) == self.kvp_rank)
        lenmask = torch.arange(max_page_num, device=device) < page_nums[:, None]
        mask = mask & lenmask

        pos = torch.cumsum(mask.int(), dim=1) - 1
        safe_pos = torch.where(mask, pos, max_page_num)
        new_table = global_page_to_local_page(global_table, self.reserved_page_num, self.group_shift)
        safe_val = torch.where(mask, new_table, 0)
        local_table.scatter_(dim=1, index=safe_pos, src=safe_val)
        seqlens_64 = seqlens.to(torch.int64)
        last_page_pos = (seqlens_64 - 1) // page_size
        torch._assert_async(torch.all((last_page_pos >= 0) & (last_page_pos < max_page_num)), "last_page_pos must >=0 and < max_page_num")
        last_page_ids = global_table.gather(1, last_page_pos[:, None]).squeeze(1)
        last_page_is_local = (last_page_ids >= reserved_pages) & (self.get_rank_from_global_page_id(last_page_ids) == self.kvp_rank)

        local_page_num = mask.sum(1, dtype=torch.int32)
        # last_page_remain = (self.page_size-seqlens % self.page_size) % self.page_size
        last_page_remain = (-seqlens) % page_size
        local_seqlens = local_page_num * page_size - last_page_is_local.int() * last_page_remain

        return local_table[:, :-1], local_seqlens.clamp_min_(0)  # pad token may be negative len

    def remap_token_idxs_to_local_return_mask(
        self,
        global_indices: torch.Tensor,
        seqlens: torch.Tensor,
    ):
        page_size = self._inner.page_size
        reserved_pages = self.reserved_page_num
        bs, max_token_num = global_indices.shape
        local_indices = global_indices.new_zeros(bs, max_token_num + 1)
        page_ids = global_indices // page_size
        mask = (page_ids >= reserved_pages) & (self.get_rank_from_global_page_id(page_ids) == self.kvp_rank)
        # [bs, s]
        len_mask = torch.arange(max_token_num, device=seqlens.device)[None, :] < seqlens[:, None]
        mask = mask & len_mask
        pos = torch.cumsum(mask.int(), dim=1) - 1

        safe_pos = torch.where(mask, pos, max_token_num)
        new_pages = global_page_to_local_page(page_ids, self.reserved_page_num, self.group_shift)
        new_indices = new_pages * page_size + global_indices % page_size
        safe_val = torch.where(mask, new_indices, 0)

        local_indices.scatter_(dim=1, index=safe_pos, src=safe_val)
        global_valid_indices = global_indices.new_zeros(bs, max_token_num + 1)
        global_valid_indices.scatter_(dim=1, index=safe_pos, src=torch.where(mask, global_indices, 0))
        local_lens = mask.sum(dim=1, dtype=torch.int32)

        return local_indices[:, :-1], global_valid_indices[:, :-1], local_lens.clamp_min_(0), mask, safe_pos

    def empty_cache_shape(self):
        if hasattr(self._inner, "quant_method") and self._inner.quant_method == "per_token_head":
            raise NotImplementedError
        else:
            return (0, 1, self._inner.kv_cache_dim)

    def empty_attn_cache_shape(self):
        return (0, 1, self._inner.kv_cache_dim)

    def empty_attn_cache_tensor(self):
        return self._inner.kv_buffer[0].new_empty((0, 1, self._inner.kv_cache_dim))

    def empty_indexer_cache_shape(self):
        return (0, self._inner.index_head_dim)

    def empty_indexer_cache_tensor(self):
        return self._inner.index_k_with_scale_buffer[0].new_empty((0, self._inner.index_k_cache_dim))

    def get_page_transfer_metadata(self, page_indices: npt.NDArray[np.int64] | torch.Tensor):
        if isinstance(page_indices, torch.Tensor):
            page_indices = page_indices.detach().cpu().numpy()
        page_indices = np.asarray(page_indices, dtype=np.int64)
        page_transfer_mask, page_local_indices = self.select_local_pages_from_global_pages(page_indices)
        return PageTransferMetadata(
            indices_are_local=True,
            page_transfer_mask=page_transfer_mask,
            page_local_indices=page_local_indices,
        )

    def get_key_split_contiguous(self, layer_id: int, indices: torch.Tensor):
        mask, local_indices = self.remap_tokens_for_kvp_local(indices)
        if not mask.any():
            return None
        assert local_indices.max() < self._inner.size, "cannot access invalid location."
        return self._inner.get_key_split_contiguous(layer_id, local_indices)

    def get_key_contiguous(self, layer_id: int, indices: torch.Tensor) -> torch.Tensor:
        from sglang.srt.mem_cache.memory_pool.memory_pool_dsa import DSATokenToKVPool

        if isinstance(self._inner, DSATokenToKVPool):
            layer_id = layer_id - self._inner.start_layer
            mask, local_indices = self.remap_tokens_for_kvp_local(indices)
            if not mask.any():
                return self.empty_attn_cache_tensor()
            assert local_indices.max() < self._inner.size, "cannot access invalid location."
            latent_cache = self._inner.kv_buffer[layer_id][local_indices].contiguous()
            return latent_cache
        else:
            assert self._inner.quant_method != "per_token_head", "get_key_contiguous do not support get key contiguous"
            layer_id = layer_id - self._inner.start_layer
            try:
                if self._inner.store_dtype != self._inner.dtype:
                    latent_cache = self._inner.kv_buffer[layer_id].view(self._inner.dtype)[indices].contiguous()
                else:
                    latent_cache = self._inner.kv_buffer[layer_id][indices].contiguous()
            except Exception as e:
                logger.error(
                    f"Error in getting key contiguous for layer {layer_id} with indices shape "
                    f"{indices.shape} and max index {indices.max()}, kv_buffer shape "
                    f"{self._inner.kv_buffer[layer_id].shape}"
                )
                raise e
            return latent_cache

    def get_key_contiguous_by_local_loc(self, layer_id: int, local_indices: torch.Tensor) -> torch.Tensor:
        layer_id = layer_id - self._inner.start_layer
        latent_cache = self._inner.kv_buffer[layer_id][local_indices].contiguous()
        return latent_cache

    def get_index_k_continuous(self, layer_id: int, seq_len: int, page_indices: torch.Tensor):
        raise NotImplementedError("L1.5 not support page_indices get_index_k_continuous")

    def get_index_k_scale_continuous(self, layer_id: int, seq_len: int, page_indices: torch.Tensor):
        raise NotImplementedError

    def get_global_page_indices(self, token_indices: torch.Tensor) -> torch.Tensor:
        return token_indices[:: self._inner.page_size] >> self.page_shift

    def get_dense_local_page_mapping(self, page_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        local_page_mask, local_page_indices = self.select_local_pages_from_global_pages(page_indices)
        dense_local_page_indices = page_indices.clone()
        dense_local_page_indices[local_page_mask] = local_page_indices
        dense_local_page_indices.masked_fill_(~local_page_mask, 0)
        return local_page_mask, dense_local_page_indices

    def flatten_local_cpu_copy(self, kv_cache_cpu):
        assert self._inner.quant_method != "per_token_head", "TokenToKVPoolL15Wrapper CPU offload does not support kv cache quant path yet"

        flat_kv_cache_cpu = []
        for layer_id in range(self._inner.layer_num):
            layer_chunks = [chunk[0] for chunk in kv_cache_cpu[layer_id]]
            if len(layer_chunks) == 0:
                empty_shape = (0,) + tuple(self._inner.kv_buffer[layer_id].shape[1:])
                flat_layer = torch.empty(
                    empty_shape,
                    dtype=self._inner.kv_buffer[layer_id].dtype,
                    device="cpu",
                )
            else:
                flat_layer = torch.cat(layer_chunks, dim=0)
            flat_kv_cache_cpu.append(flat_layer)
        return flat_kv_cache_cpu

    def npu_get_cpu_copy(self, local_page_indices: torch.Tensor):
        return self._inner.npu_get_cpu_copy(local_page_indices)

    def npu_flatten_local_cpu_copy(self, kv_cache_cpu):
        assert self._inner.quant_method != "per_token_head", "TokenToKVPoolL15Wrapper CPU offload does not support kv cache quant path yet"

        flat_kv_cache_cpu = []
        for layer_id in range(self._inner.layer_num):
            layer_chunks = kv_cache_cpu[layer_id]
            if ENV.npu_enable_mla_split_kv_kr:
                flat_layer = []
                for tensor_idx in range(2):
                    tensor_chunks = [chunk[tensor_idx] for chunk in layer_chunks]
                    if len(tensor_chunks) == 0:
                        src_buf = self._inner.kv_buffer[layer_id * 2 + tensor_idx]
                        flat_tensor = torch.empty(
                            (0, src_buf.shape[-1] * self._inner.page_size),
                            dtype=src_buf.dtype,
                            device="cpu",
                        )
                    else:
                        flat_tensor = torch.cat(tensor_chunks, dim=0)
                    flat_layer.append(flat_tensor)
                flat_kv_cache_cpu.append(flat_layer)
            else:
                tensor_chunks = [chunk[0] for chunk in layer_chunks]
                if len(tensor_chunks) == 0:
                    flat_layer = torch.empty(
                        (0, self._inner.page_size * self._inner.kv_cache_dim),
                        dtype=self._inner.kv_buffer[layer_id].dtype,
                        device="cpu",
                    )
                else:
                    flat_layer = torch.cat(tensor_chunks, dim=0)
                flat_kv_cache_cpu.append([flat_layer])
        return flat_kv_cache_cpu

    def all_gather_cpu_copy(self, local_mask: torch.Tensor, kv_cache_cpu):
        assert self._inner.quant_method != "per_token_head", "TokenToKVPoolL15Wrapper CPU offload does not support kv cache quant path yet"

        if self.attn_tp_group_size == 1:
            return kv_cache_cpu

        gathered_payloads = [None for _ in range(self.attn_tp_group_size)]
        local_payload = (
            local_mask.detach().cpu(),
            self.flatten_local_cpu_copy(kv_cache_cpu),
        )
        dist.all_gather_object(
            gathered_payloads,
            local_payload,
            group=get_attention_tp_group().cpu_group,
        )

        num_tokens = local_mask.numel()
        global_kv_cache_cpu = []
        for layer_id in range(self._inner.layer_num):
            full_layer = torch.empty(
                (num_tokens,) + tuple(self._inner.kv_buffer[layer_id].shape[1:]),
                dtype=self._inner.kv_buffer[layer_id].dtype,
                device="cpu",
            )
            for rank_mask, rank_payload in gathered_payloads:
                if rank_mask.any():
                    full_layer[rank_mask] = rank_payload[layer_id]
            global_kv_cache_cpu.append([])
            for i in range(0, num_tokens, self._inner.offload_chunk_page_num):
                global_kv_cache_cpu[-1].append([full_layer[i : i + self._inner.offload_chunk_page_num]])

        return global_kv_cache_cpu

    def get_cpu_copy(self, token_indices):
        from sglang.srt.mem_cache.memory_pool.memory_pool_dsa import (
            DSATokenToKVPool,
            L15DSACpuCopy,
        )
        from sglang.srt.mem_cache.memory_pool.memory_pool_mla import MLATokenToKVPool

        if not isinstance(self._inner, MLATokenToKVPool):
            return self._inner.get_cpu_copy(token_indices)

        assert self._inner.quant_method != "per_token_head", "TokenToKVPoolL15Wrapper CPU offload does not support kv cache quant path yet"

        if __is_npu__:
            page_indices = self.get_global_page_indices(token_indices)
            local_page_mask, local_page_indices = self.select_local_pages_from_global_pages(page_indices)
            if isinstance(self._inner, DSATokenToKVPool):
                kv_local, index_local = self.npu_get_cpu_copy(local_page_indices)
                return L15DSACpuCopy(
                    page2rank=self.get_ranks_from_global_pages(page_indices),
                    local_page_mask=local_page_mask.detach().cpu(),
                    kv_local=kv_local,
                    index_local=index_local,
                )
            return self.npu_get_cpu_copy(local_page_indices)

        dense_local_indices = token_indices.clone()
        local_mask, _ = self.remap_tokens_for_kvp_local(dense_local_indices)
        local_indices = dense_local_indices[local_mask]
        kv_cache_cpu = self._inner.get_cpu_copy(local_indices)
        return self.all_gather_cpu_copy(local_mask, kv_cache_cpu)

    def npu_load_cpu_copy(
        self,
        kv_cache_cpu,
        local_page_mask,
        local_page_mask_cpu,
        dense_local_page_indices,
        page_indices,
    ):
        self._inner.npu_load_cpu_copy(
            kv_cache_cpu,
            local_page_mask,
            local_page_mask_cpu,
            dense_local_page_indices,
            page_indices,
        )

    def npu_load_dsa_rank_local_cpu_copy(self, saved_payload, token_indices, prefix_page_count: int):
        from sglang.srt.mem_cache.memory_pool.memory_pool_dsa import L15DSACpuCopy

        assert isinstance(saved_payload, L15DSACpuCopy), "TokenToKVPoolL15Wrapper expects an L15DSACpuCopy payload for DSA rank-local resume"
        assert ENV.npu_enable_mla_split_kv_kr, "DSA CPU offload requires npu_enable_mla_split_kv_kr"

        page_indices = self.get_global_page_indices(token_indices)
        new_local_page_mask, new_local_page_indices = self.select_local_pages_from_global_pages(page_indices)
        new_local_page_mask_cpu = new_local_page_mask.detach().cpu()
        saved_page_mask = saved_payload.local_page_mask

        assert new_local_page_mask_cpu.shape == saved_page_mask.shape, (
            f"page count mismatch on resume: new={new_local_page_mask_cpu.shape} saved={saved_page_mask.shape}"
        )
        assert torch.equal(
            new_local_page_mask_cpu[prefix_page_count:],
            saved_page_mask[prefix_page_count:],
        ), "rank alignment broken on resume: saved local page mask tail does not match the mask derived from newly allocated token indices"

        saved_prefix_local_count = int(saved_page_mask[:prefix_page_count].sum().item())
        new_prefix_local_count = int(new_local_page_mask_cpu[:prefix_page_count].sum().item())
        target_local_pages = new_local_page_indices[new_prefix_local_count:]
        self._inner.npu_load_rank_local_cpu_copy(
            saved_payload.kv_local,
            saved_payload.index_local,
            target_local_pages,
            skip_prefix_local_pages=saved_prefix_local_count,
        )

    def load_cpu_copy(self, kv_cache_cpu, token_indices, prefix_page_count: int = 0):
        from sglang.srt.mem_cache.memory_pool.memory_pool_dsa import L15DSACpuCopy
        from sglang.srt.mem_cache.memory_pool.memory_pool_mla import MLATokenToKVPool

        if not isinstance(self._inner, MLATokenToKVPool):
            return self._inner.load_cpu_copy(kv_cache_cpu, token_indices, prefix_page_count=prefix_page_count)

        assert self._inner.quant_method != "per_token_head", "TokenToKVPoolL15Wrapper CPU offload does not support kv cache quant path yet"

        if __is_npu__:
            if isinstance(kv_cache_cpu, L15DSACpuCopy):
                return self.npu_load_dsa_rank_local_cpu_copy(kv_cache_cpu, token_indices, prefix_page_count)
            del prefix_page_count
            page_indices = self.get_global_page_indices(token_indices)
            local_page_mask, dense_local_page_indices = self.get_dense_local_page_mapping(page_indices)
            local_page_mask_cpu = local_page_mask.cpu()
            return self.npu_load_cpu_copy(
                kv_cache_cpu,
                local_page_mask,
                local_page_mask_cpu,
                dense_local_page_indices,
                page_indices,
            )
        del prefix_page_count

        dense_local_indices = token_indices.clone()
        local_mask, _ = self.remap_tokens_for_kvp_local(dense_local_indices)
        local_mask_cpu = local_mask.cpu()

        self._inner.sync_device()
        for layer_id in range(self._inner.layer_num):
            for i in range(0, len(token_indices), self._inner.offload_chunk_page_num):
                chunk_mask_cpu = local_mask_cpu[i : i + self._inner.offload_chunk_page_num]
                if not chunk_mask_cpu.any():
                    continue
                chunk_mask = local_mask[i : i + self._inner.offload_chunk_page_num]
                chunk_local_indices = dense_local_indices[i : i + self._inner.offload_chunk_page_num][chunk_mask]
                kv_cpu = kv_cache_cpu[layer_id][i // self._inner.offload_chunk_page_num][0][chunk_mask_cpu]
                assert kv_cpu.shape[0] == len(chunk_local_indices), (
                    f"kv_cpu.shape[0] {kv_cpu.shape[0]} != len(chunk_local_indices) {len(chunk_local_indices)}"
                )
                kv_chunk = kv_cpu.to(self._inner.kv_buffer[0].device, non_blocking=True)
                self._inner.kv_buffer[layer_id][chunk_local_indices] = kv_chunk
        self._inner.sync_device()
