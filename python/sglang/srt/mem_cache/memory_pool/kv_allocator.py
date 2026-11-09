from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.env import global_server_args_dict
from sglang.srt.utils import check_memory_debug, get_colorful_logger, get_rank_page_nums

logger = get_colorful_logger(__name__)


@dataclass
class PagesInfo:
    last_slot: int
    num_pages: int
    pages: torch.Tensor


class KVAllocator:
    def __init__(
        self,
        size: int,
        device: str,
        max_batch_size: int,
        max_context_len: int,
        page_size: int,
    ):
        self.rank_free_slots = None
        self.token_slot_refs = torch.zeros(size, dtype=torch.int32, device=device) if check_memory_debug() else None
        self.size = size
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
        attn_tp_size = global_server_args_dict["attn_tp_size"]
        if global_server_args_dict["enable_mla_l1_5_cache"]:
            self.group_rank = attn_tp_size
            self.group_mask = attn_tp_size - 1
        else:
            self.group_rank = 1
            self.group_mask = 0
        self.cur_alloc_rank_idx = 0
        self.reserved_page_num = self.get_reserved_page_num()
        self.clear()

    def get_rank_from_global_page_id(self, global_page_id):
        return (global_page_id - self.reserved_page_num) & self.group_mask

    def clear(self):
        # A fresh pool must not inherit page-owner rotation from old requests.
        self.cur_alloc_rank_idx = 0
        self.rank_free_slots: list[list[int]] = [[] for _ in range(self.group_rank)]
        for page_id in range(self.reserved_page_num, self.size // self.page_size):
            rank = self.get_rank_from_global_page_id(page_id)
            self.rank_free_slots[rank].append(page_id)

        self.req_to_page.zero_()
        if self.token_slot_refs is not None:
            self.token_slot_refs.zero_()
        self.free_group = []

    @property
    def free_slots(self):
        free_slots = []
        for slots in self.rank_free_slots:
            free_slots.extend(slots)
        return torch.tensor(free_slots, dtype=torch.int32)

    def available_size(self):
        return sum(len(v) for v in self.rank_free_slots)

    def available_pages(self):
        return self.available_size()

    def get_reserved_page_num(self) -> int:
        if global_server_args_dict["kvp_size"] > 1:
            return 1 + self.max_batch_size
        return 1

    def get_rank_free_page_nums(self) -> list[int]:
        return [len(v) for v in self.rank_free_slots]

    def alloc_pages(self, need_page_num: int) -> torch.Tensor:
        if need_page_num > self.available_size():
            return None
        result = []
        while len(result) < need_page_num:
            cur_free_slots = self.rank_free_slots[self.cur_alloc_rank_idx]
            if cur_free_slots:
                result.append(cur_free_slots.pop(0))
            self.cur_alloc_rank_idx = (self.cur_alloc_rank_idx + 1) % self.group_rank
        return torch.tensor(result, dtype=torch.int64)

    def check_rank_pages_enough(self, page2rank: list):
        rank_need_page_nums = get_rank_page_nums(page2rank)
        rank_free_page_nums = self.get_rank_free_page_nums()
        return not any(rank_free_page_nums[rank] < rank_need_page_nums[rank] for rank in rank_need_page_nums)

    def rank_alloc_pages(self, page2rank: list) -> Optional[torch.Tensor]:
        if not self.check_rank_pages_enough(page2rank):
            return None

        allocated_pages = []
        for target_rank in page2rank:
            allocated_pages.append(self.rank_free_slots[target_rank].pop(0))
        return torch.tensor(allocated_pages, dtype=torch.int64)

    def free_pages(self, page_indices: torch.Tensor):
        if self.group_mask > 0:
            ranks = self.get_rank_from_global_page_id(page_indices).to("cpu", non_blocking=True)
            page_indices = page_indices.to("cpu", non_blocking=True)
            ranks = ranks.tolist()
            page_indices = page_indices.tolist()
            for rank, page in zip(ranks, page_indices):
                self.rank_free_slots[rank].append(page)
        else:
            page_indices = page_indices.tolist()
            for page in page_indices:
                self.rank_free_slots[0].append(page)

    def alloc(self, req_pool_index: int, need_size: int, alloced_len: int):
        page_offset = alloced_len % self.page_size
        page_num = (alloced_len + self.page_size - 1) // self.page_size
        last_page_remain = page_num * self.page_size - alloced_len
        last_page_id = self.req_to_page[req_pool_index, page_num - 1].item()
        kv_loc = last_page_id * self.page_size + page_offset + torch.arange(0, min(last_page_remain, need_size), dtype=torch.int32)
        if last_page_remain >= need_size:
            return kv_loc.to(self.device, non_blocking=True)

        remain_size = need_size - last_page_remain
        need_new_page_num = (remain_size + self.page_size - 1) // self.page_size

        if page_num + need_new_page_num > self.max_page_num:
            logger.warning(
                f"Requested page range [{page_num}:{page_num + need_new_page_num}] "
                f"exceeds max_page_num {self.max_page_num}. "
                f"alloced_len={alloced_len}, need_size={need_size}, page_num={page_num}"
            )
            return None

        new_pages = self.alloc_pages(need_new_page_num)
        if new_pages is None:
            return None

        self.req_to_page[req_pool_index, page_num : page_num + need_new_page_num] = new_pages.to(self.device)
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
        page_ids_to_free = self.req_to_page[req_pool_index, full_page_num : full_page_num + page_num_to_free]
        self.free_group.append(page_ids_to_free.cpu())

    def free_req_cache(self, req_pool_index: int, alloced_len: int):
        alloced_page_num = (alloced_len + self.page_size - 1) // self.page_size
        if alloced_page_num == 0:
            return
        page_ids_to_free = self.req_to_page[req_pool_index, :alloced_page_num]
        self.free_group.append(page_ids_to_free.cpu())

    def free_with_diff(self, new_prefix_page_ids, old_page_ids):
        assert len(new_prefix_page_ids) == len(old_page_ids), "[free with diff] new_prefix_page_ids and old_page_ids should have the same length"
        diff = new_prefix_page_ids != old_page_ids
        if torch.any(diff):
            logger.info(f"[DebugTrace] free_with_diff free page={old_page_ids[diff].tolist()}")
            self.free_group.append(old_page_ids[diff].cpu())
        else:
            logger.debug("[DebugTrace] free_with_diff: no pages to free, all pages are cached")
        return diff

    def append_to_later_free(self, page_ids: torch.Tensor):
        self.free_group.append(page_ids.cpu())

    def free_group_flush(self):
        if self.free_group:
            pages_need_to_free = torch.concat(self.free_group)
            logger.debug(f"[DebugTrace] free_group_flush pages_to_free={pages_need_to_free.tolist()}")
            if self.token_slot_refs is not None:
                token_level_offsets = torch.arange(self.page_size)
                slots_to_free = (pages_need_to_free[:, None] * self.page_size + token_level_offsets).flatten().to(self.device)
                writted_positions = slots_to_free[self.token_slot_refs[slots_to_free] >= 1]
                self.token_slot_refs[writted_positions] += -1
            self.free_pages(pages_need_to_free)
            self.free_group = []

    def check_rank_alloc(
        self,
        num_tokens: int,
        alloced_num_tokens: int,
        page2rank: list,
    ) -> bool:
        num_pages = num_tokens // self.page_size
        if num_pages == 0:
            return False

        assert len(page2rank) == num_pages, f"page2rank length {len(page2rank)} != num_pages {num_pages}"

        page_start = (alloced_num_tokens + self.page_size - 1) // self.page_size
        if page_start + num_pages > self.max_page_num:
            return False

        return self.check_rank_pages_enough(page2rank)

    def rank_alloc(self, req_pool_index, num_tokens, alloced_num_tokens, page2rank):
        assert alloced_num_tokens % self.page_size == 0, (
            f"rank_alloc requires page-aligned alloced_num_tokens, got {alloced_num_tokens} with page_size={self.page_size}"
        )
        assert num_tokens % self.page_size == 0, f"rank_alloc requires page-aligned num_tokens, got {num_tokens} with page_size={self.page_size}"

        if not self.check_rank_alloc(num_tokens, alloced_num_tokens, page2rank):
            return False

        allocated_pages = self.rank_alloc_pages(page2rank)
        assert allocated_pages is not None

        num_pages = num_tokens // self.page_size
        page_start = (alloced_num_tokens + self.page_size - 1) // self.page_size

        self.req_to_page[req_pool_index, page_start : page_start + num_pages] = allocated_pages.to(self.device)

        device_token_indices = (
            (allocated_pages.unsqueeze(1) * self.page_size + torch.arange(self.page_size, dtype=torch.int32))
            .flatten()[:num_tokens]
            .to(self.device, non_blocking=True)
        )

        return device_token_indices
