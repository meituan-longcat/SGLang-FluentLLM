from enum import IntEnum, auto
from typing import Optional

import torch


class SpeculativeAlgorithm(IntEnum):
    NONE = auto()
    EAGLE = auto()
    EAGLE3 = auto()
    PLD = auto()
    DFLASH = auto()

    def is_none(self):
        return self == SpeculativeAlgorithm.NONE

    def is_eagle(self):
        return self == SpeculativeAlgorithm.EAGLE or self == SpeculativeAlgorithm.EAGLE3

    def is_eagle3(self):
        return self == SpeculativeAlgorithm.EAGLE3

    def is_pld(self):
        return self == SpeculativeAlgorithm.PLD

    def is_dflash(self):
        return self == SpeculativeAlgorithm.DFLASH

    @staticmethod
    def from_string(name: str):
        name_map = {
            "EAGLE": SpeculativeAlgorithm.EAGLE,
            "EAGLE3": SpeculativeAlgorithm.EAGLE3,
            "PLD": SpeculativeAlgorithm.PLD,
            "DFLASH": SpeculativeAlgorithm.DFLASH,
            None: SpeculativeAlgorithm.NONE,
        }
        if name is not None:
            name = name.upper()
        return name_map[name]


class SpecInfo:
    pass


class SpeculativeSamplingInfo:
    """Per-request draft sampling probabilities for speculative-sampling verification.

    The tensors are shared between the target worker and the draft worker(s):
    - probs:          [max_num_reqs, num_draft_tokens, top_k], the draft probability
                      of each candidate token. The first num_draft_tokens - 1 slots
                      are written by draft_sample; the last slot is a zero padding
                      that reads as q = 0 for the bonus position in verification.
    - probs_idx:      [max_num_reqs, num_draft_tokens, top_k], the vocab id of each
                      candidate token.
    - probs_selected: [max_num_reqs, num_draft_tokens - 1], the draft probability of
                      the actually sampled (selected) token at each draft step.
    """

    def __init__(
        self,
        probs: Optional[torch.Tensor] = None,
        probs_idx: Optional[torch.Tensor] = None,
        probs_selected: Optional[torch.Tensor] = None,
    ):
        self.probs = probs
        self.probs_idx = probs_idx
        self.probs_selected = probs_selected

    @classmethod
    def create(cls, max_num_reqs: int, num_draft_tokens: int, top_k: int, device: str):
        probs = torch.zeros([max_num_reqs, num_draft_tokens, top_k], dtype=torch.float32, device=device)
        # One-hot default on the real slots; the last slot stays zero as the padding.
        probs[:, :-1, 0] = 1.0
        probs_idx = torch.zeros([max_num_reqs, num_draft_tokens, top_k], dtype=torch.int32, device=device)
        probs_idx[:] = torch.arange(top_k, device=device, dtype=torch.int32)[None, None, :].expand(max_num_reqs, num_draft_tokens, -1)
        # Consistent with the default one-hot distribution: the selected token has prob 1.0.
        probs_selected = torch.ones([max_num_reqs, num_draft_tokens - 1], dtype=torch.float32, device=device)
        return cls(probs, probs_idx, probs_selected)

    def reset(self, req_pool_indices: torch.Tensor):
        """Reset the given requests to the default distribution
        (all probability mass on the first candidate, identity candidate ids)."""
        num_draft_tokens, top_k = self.probs.shape[1], self.probs.shape[2]
        self.probs_idx[req_pool_indices] = torch.arange(top_k, device=self.probs.device, dtype=torch.int32)[None, None, :].expand(
            len(req_pool_indices), num_draft_tokens, -1
        )
        if self.probs.device.type == "npu":
            # Reset complete rows: NPU view writeback can overwrite other slots.
            default_probs = self.probs.new_zeros((num_draft_tokens, top_k))
            default_probs[:-1, 0] = 1.0
            self.probs[req_pool_indices] = default_probs
        else:
            self.probs[req_pool_indices] = 0
            self.probs[req_pool_indices, :-1, 0] = 1.0
        self.probs_selected[req_pool_indices] = 1.0

    def update_draft(self, req_pool_indices, draft_step, probs, probs_idx, selected_index):
        if self.probs.device.type == "npu":
            # Tensor step indices avoid non-contiguous views on the NPU backend.
            steps = torch.full_like(req_pool_indices, draft_step)
            indices = (req_pool_indices, steps)
            self.probs.index_put_(indices, probs)
            self.probs_idx.index_put_(indices, probs_idx.to(self.probs_idx.dtype))
            self.probs_selected.index_put_(indices, probs.gather(1, selected_index).squeeze(1))
        else:
            self.probs[req_pool_indices, draft_step, :] = probs
            self.probs_idx[req_pool_indices, draft_step, :] = probs_idx.to(self.probs_idx.dtype)
            self.probs_selected[req_pool_indices, draft_step] = probs.gather(1, selected_index).squeeze(1)
