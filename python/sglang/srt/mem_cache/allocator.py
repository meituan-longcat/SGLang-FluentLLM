
from dataclasses import dataclass

import torch

from sglang.srt.utils import get_colorful_logger, is_npu
from sglang.srt.mem_cache.memory_pool import SWAKVPool
from sglang.srt.env import global_server_args_dict

logger = get_colorful_logger(__name__)


@dataclass
class PagesInfo:
    last_slot: int
    num_pages: int
    pages: torch.Tensor


class KVAllocator:
    """
    Only responsible for operating on token slots and block table metadata without holding the actual KV cache
    physical storage, separating slot and page management from KVCache physical storage. This allows the same
    metadata operations to work with different physical memory. (Mainly For spec decode)
    """

    def __init__(
        self,
        size: int,
        device: str,
        max_batch_size: int,
        max_context_len: int,
        page_size: int,
    ):
        self.free_slots = None
        self.token_slot_refs = None
        self.size = size
        self.is_not_in_free_group = True
        self.free_group = []
        self.page_size = page_size
        self.device = device
        self.last_slot = torch.ones(max_batch_size, dtype=torch.int32) * (page_size - 1)
        self.num_pages = torch.zeros(max_batch_size, dtype=torch.int32)
        self.req_to_page = torch.zeros(
            (max_batch_size, (max_context_len + page_size - 1) // page_size),
            dtype=torch.int32,
            device=device,
        )
        self.max_context_len = max_context_len
        self.max_page_num = (max_context_len + page_size - 1) // page_size
        self.max_batch_size = max_batch_size
        self.clear()

    def available_size(self):
        return len(self.free_slots)

    def available_pages(self):
        return self.available_size()


    def alloc(self, req_pool_index: int, need_size: int, alloced_len: int):
        page_offset = alloced_len % self.page_size
        page_num = (alloced_len + self.page_size - 1) // self.page_size
        last_page_remain = page_num * self.page_size - alloced_len
        last_page_id = self.req_to_page[req_pool_index, page_num - 1].item()
        # if last_page_remain is zero, kv_loc is Tensor([])
        kv_loc = (
            last_page_id * self.page_size
            + page_offset
            + torch.arange(0, min(last_page_remain, need_size), dtype=torch.int32)
        )
        if last_page_remain >= need_size:
            return kv_loc.to(self.device, non_blocking=True)

        remain_size = need_size - last_page_remain
        need_new_page_num = (remain_size + self.page_size - 1) // self.page_size
        if need_new_page_num > len(self.free_slots):
            # do not change self.seq_lens
            return None

        # Requested page range [384:512] exceeds max_page_num 504. alloced_len=49152, need_size=16384, page_num=384
        # Check if we have enough space in req_to_page tensor
        if page_num + need_new_page_num > self.max_page_num:
            logger.warning(
                f"Requested page range [{page_num}:{page_num + need_new_page_num}] "
                f"exceeds max_page_num {self.max_page_num}. "
                f"alloced_len={alloced_len}, need_size={need_size}, page_num={page_num}"
            )
            # Do not change self.seq_lens
            return None

        new_pages = self.free_slots[:need_new_page_num]
        self.free_slots = self.free_slots[need_new_page_num:]
        # update req_to_page
        self.req_to_page[req_pool_index, page_num : page_num + need_new_page_num] = (
            new_pages.to(self.device)
        )
        # construct kv_loc
        kv_loc1 = new_pages.unsqueeze(1) * self.page_size
        offsets = torch.arange(0, self.page_size, dtype=torch.int32)
        kv_loc1 = kv_loc1 + offsets
        kv_loc1 = kv_loc1.flatten()[:remain_size]
        final_kv_loc = torch.concat([kv_loc, kv_loc1]).to(self.device, non_blocking=True)
        return final_kv_loc

    def free_extra_pages_not_cached(self, req_pool_index: int, real_seq_len: int, alloced_len: int):
        full_page_num = real_seq_len // self.page_size
        alloced_page_num = (alloced_len + self.page_size - 1) // self.page_size
        page_num_to_free = alloced_page_num - full_page_num
        if page_num_to_free == 0:
            return
        page_ids_to_free = self.req_to_page[req_pool_index, full_page_num: full_page_num + page_num_to_free]
        self.need_to_free.append(page_ids_to_free.cpu())

    def free_req_cache(self, req_pool_index: int, alloced_len: int):
        """
        Without radix_cache, release all pages of the request
        """
        alloced_page_num = (alloced_len + self.page_size - 1) // self.page_size
        if alloced_page_num == 0:
            return
        page_ids_to_free = self.req_to_page[req_pool_index, :alloced_page_num]
        self.need_to_free.append(page_ids_to_free.cpu())

    def free_with_diff(self, new_prefix_page_ids, old_page_ids):
        # new kv are from radix tree, which is cached
        # so free the diff in old
        assert len(new_prefix_page_ids) == len(old_page_ids), (
            "[free with diff] new_prefix_page_ids and old_page_ids should have the same length"
        )
        diff = new_prefix_page_ids != old_page_ids
        if torch.any(diff):
            logger.debug(f"[DebugTrace] free_with_diff free page={old_page_ids[diff].tolist()}")
            self.need_to_free.append(old_page_ids[diff].cpu())
        else:
            logger.debug(f"[DebugTrace] free_with_diff: no pages to free, all pages are cached")
        return diff

    def append_to_later_free(self, page_ids: torch.Tensor):
        self.need_to_free.append(page_ids.cpu())

    def free(self, req_pool_index: int, indices=None):
        if self.is_not_in_free_group:
            num_pages = self.num_pages[req_pool_index]
            pages = self.req_to_page[req_pool_index, :num_pages].cpu()
            free_slots = [self.free_slots]
            for i in range(num_pages):
                page_index = pages[i]
                free_slots.append(
                    torch.arange(
                        page_index * self.page_size,
                        (page_index + 1) * self.page_size,
                        dtype=torch.int32,
                    )
                )
            self.free_slots = torch.concat(free_slots)
            self.num_pages[req_pool_index] = 0
            self.last_slot[req_pool_index] = self.page_size - 1
        else:
            self.free_group.append(req_pool_index)

    def free_group_begin(self):
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self):
        self.is_not_in_free_group = True
        if self.need_to_free:
            pages_need_to_free = torch.concat(self.need_to_free)
            logger.debug(f"[DebugTrace] free_group_end pages_need_to_free={pages_need_to_free.tolist()}")
            token_level_offsets = torch.arange(self.page_size)
            slots_to_free = (pages_need_to_free[:, None] * self.page_size + token_level_offsets).flatten().to(self.device)
            writted_positions = slots_to_free[self.token_slot_refs[slots_to_free] >= 1]
            self.token_slot_refs[writted_positions] += -1
            self.free_slots = torch.concat([self.free_slots] + self.need_to_free)
            self.need_to_free = []

    def clear(self):
        # Page 0 is used for padding
        # when npu + kvp_size>1, reserve self.max_batch_size kv block for current token
        reserved_block_num = 1
        if global_server_args_dict["kvp_size"] > 1:
            reserved_block_num = 1 + self.max_batch_size
        self.free_slots = torch.arange(
            reserved_block_num, self.size // self.page_size, dtype=torch.int32
        )
        if self.token_slot_refs is not None:
            self.token_slot_refs.zero_()
            self.req_to_page.zero_()
        else:
            self.token_slot_refs = torch.zeros(self.size, dtype=torch.int32, device=self.device)
            self.req_to_page = torch.zeros(
                (self.max_batch_size, self.max_page_num),
                dtype=torch.int32,
                device=self.device,
            )
        self.free_group = []
        self.need_to_free = []


class SWA_KVAllocator:
    """Allocator for SWA hybrid KV cache."""

    def __init__(
        self,
        size: int,
        size_swa: int,
        dtype: torch.dtype,
        device: str,
        kvcache: SWAKVPool,
    ):
        super().__init__(size, 1, dtype, device, kvcache)
        assert isinstance(kvcache, SWAKVPool)
        self._size_full = size
        self._size_swa = size_swa
        self.full_attn_allocator = KVAllocator(
            size,
            dtype,
            device,
            kvcache.full_kv_pool,
        )
        self.swa_attn_allocator = KVAllocator(
            size_swa,
            dtype,
            device,
            kvcache.swa_kv_pool,
        )
        self.full_to_swa_index_mapping = torch.empty(
            size + size_swa + 1,
            dtype=torch.int64,
            device=device,
        )
        self.clear()

        self._kvcache.full_to_swa_index_mapping = self.full_to_swa_index_mapping

    def available_size(self):
        raise NotImplementedError()

    def full_available_size(self):
        return self.full_attn_allocator.available_size()

    def swa_available_size(self):
        return self.swa_attn_allocator.available_size()

    @property
    def size_full(self):
        return self._size_full

    @property
    def size_swa(self):
        return self._size_swa

    def debug_print(self) -> str:
        msg = ""
        msg += f"#swa-available-size: {self.swa_attn_allocator.available_size()}, "
        msg += (
            f"#full-attn-available-size: {self.full_attn_allocator.available_size()}, "
        )
        return msg

    def get_kvcache(self):
        return self._kvcache

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        assert self.full_to_swa_index_mapping is not None
        return self.full_to_swa_index_mapping[kv_indices].to(torch.int32)

    def alloc(self, need_size: int):
        if need_size > self.full_attn_allocator.available_size():
            return None
        if need_size > self.swa_attn_allocator.available_size():
            return None

        alloc_full_indices = self.full_attn_allocator.alloc(need_size)
        alloc_swa_indices = self.swa_attn_allocator.alloc(need_size)
        self.full_to_swa_index_mapping[alloc_full_indices] = alloc_swa_indices
        return alloc_full_indices

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        if self.is_not_in_free_group:
            self.full_attn_allocator.free(free_index)
            self.free_swa(free_index)
        else:
            self.free_group.append(free_index)
        assert (
            self.full_attn_allocator.available_size() <= self.full_attn_allocator.size
        )
        assert self.swa_attn_allocator.available_size() <= self.swa_attn_allocator.size

    def free_swa(self, free_index: torch.Tensor):
        swa_indices = self.full_to_swa_index_mapping[free_index]
        swa_indices = swa_indices[swa_indices > 0]
        self.swa_attn_allocator.free(swa_indices)
        self.full_to_swa_index_mapping[free_index] = 0

    def backup_state(self):
        raise NotImplementedError

    def restore_state(self, state):
        raise NotImplementedError

    def clear(self):
        self.swa_attn_allocator.clear()
        self.full_attn_allocator.clear()
        self.full_to_swa_index_mapping.fill_(0)
        self.is_in_free_group = False
        self.free_group = []
