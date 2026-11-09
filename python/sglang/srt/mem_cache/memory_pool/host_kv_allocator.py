from __future__ import annotations

import torch

from sglang.srt.utils import get_colorful_logger, get_rank_page_nums

logger = get_colorful_logger(__name__)


class HostKVAllocator:
    def __init__(
        self,
        global_page_num: int,
        group_mask: int = 0,
    ):
        self.global_page_num = global_page_num
        self.group_mask = group_mask
        self.group_rank = group_mask + 1
        if group_mask:
            assert (group_mask + 1) & group_mask == 0, f"group_mask+1 {group_mask + 1} is not a power of 2"
        self.clear()
        logger.info(f"HostKVAllocator init: global_page_num={global_page_num} group_mask={group_mask}")

    @property
    def free_slots(self):
        free_slots = []
        for slots in self.rank_free_slots:
            free_slots.extend(slots)
        return torch.tensor(free_slots, dtype=torch.int64)

    def clear(self):
        self.rank_free_slots: list[list] = [[] for _ in range(self.group_rank)]
        for slot in range(self.global_page_num):
            rank = slot & self.group_mask
            self.rank_free_slots[rank].append(slot)
        self.cur_alloc_rank_idx = 0

    def available_size(self) -> int:
        return sum(len(v) for v in self.rank_free_slots)

    def get_rank_free_page_nums(self) -> list[int]:
        return [len(v) for v in self.rank_free_slots]

    def alloc_pages(self, num_pages: int) -> torch.Tensor | None:
        if num_pages > self.available_size():
            return None

        result = []
        while len(result) < num_pages:
            cur_free_slots = self.rank_free_slots[self.cur_alloc_rank_idx]
            if cur_free_slots:
                result.append(cur_free_slots.pop(0))
            self.cur_alloc_rank_idx = (self.cur_alloc_rank_idx + 1) % self.group_rank
        return torch.tensor(result, dtype=torch.int64)

    def rank_alloc_pages(self, page2rank: list) -> torch.Tensor | None:
        rank_need_page_nums = get_rank_page_nums(page2rank)
        for rank, need_page_num in rank_need_page_nums.items():
            if len(self.rank_free_slots[rank]) < need_page_num:
                return None

        result = []
        for target_rank in page2rank:
            result.append(self.rank_free_slots[target_rank].pop(0))
        return torch.tensor(result, dtype=torch.int64)

    def free_pages(self, page_indices: torch.Tensor) -> int:
        for slot in page_indices.tolist():
            self.rank_free_slots[slot & self.group_mask].append(slot)
        return len(page_indices)

    def get_rank_from_global_page_id(self, global_page_id):
        return global_page_id & self.group_mask
