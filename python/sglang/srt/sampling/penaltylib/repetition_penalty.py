import weakref
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.scheduling.schedule_batch import ScheduleBatch
from sglang.srt.sampling.penaltylib.orchestrator import (
    _BatchedPenalizer,
)
from sglang.srt.utils import get_colorful_logger, get_compiler_backend, is_npu

logger = get_colorful_logger(__name__)

if is_npu():

    @torch.inference_mode()
    def apply_scaling_penalties(logits, scaling_penalties):
        logits[:] = torch.where(
            logits < 0,
            logits * scaling_penalties,
            logits / scaling_penalties,
        )
else:

    @torch.compile(dynamic=True, backend=get_compiler_backend())
    def apply_scaling_penalties(logits, scaling_penalties):
        logits[:] = torch.where(
            logits < 0,
            logits * scaling_penalties,
            logits / scaling_penalties,
        )


class BatchedRepetitionPenalizer(_BatchedPenalizer):
    """
    Repetition penalizer penalizes tokens based on their repetition in the input and output.
    """

    def __init__(self, batch, device, vocab_size):
        self._batch_ref = weakref.ref(batch)
        self.device = device
        self.vocab_size = vocab_size
        self._is_prepared = False

    @property
    def batch(self) -> "Optional[ScheduleBatch]":
        return self._batch_ref()

    @batch.setter
    def batch(self, value: "Optional[ScheduleBatch]"):
        if value is None:
            self._batch_ref = lambda: None
        else:
            self._batch_ref = weakref.ref(value)

    def reqs(self):
        return self.batch.reqs

    def _is_required(self) -> bool:
        return any(req.sampling_params.repetition_penalty != 1.0 for req in self.reqs())

    def _prepare(self):
        self.cumulated_repetition_penalties = torch.ones(
            (len(self.reqs()), self.vocab_size),
            dtype=torch.float32,
            device=self.device,
        )

        self.repetition_penalties = (
            torch.tensor(
                data=[req.sampling_params.repetition_penalty for req in self.reqs()],
                dtype=torch.float32,
                device=self.device,
            )
        ).unsqueeze_(1)

    def _cumulate_output_tokens(self, output_ids):
        try:
            self.cumulated_repetition_penalties.scatter_(
                dim=1,
                index=output_ids.unsqueeze(1),
                src=self.repetition_penalties,
            )
        except Exception as e:
            logger.info(f"repetiton penalty _cumulate_output_tokens failed due to Exception: {str(e)}")

    def _update_scaling_penalty(self, scaling_penalty: torch.Tensor) -> torch.Tensor:
        if not self._is_prepared:
            return
        # sacling_penalty should be a tensor filled with 1
        assert scaling_penalty is not None
        scaling_penalty.mul_(self.cumulated_repetition_penalties)

    @torch.inference_mode()
    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        try:
            apply_scaling_penalties(logits, self.cumulated_repetition_penalties)
        except Exception as e:
            logger.info(f"apply repetiton penalty failed due to Exception: {str(e)}")
        return logits

    def _filter(self, keep_indices: torch.Tensor):
        try:
            self.repetition_penalties = self.repetition_penalties[keep_indices]
            self.cumulated_repetition_penalties = self.cumulated_repetition_penalties[keep_indices]
        except Exception as e:
            logger.info(f"filter repetiton penalty failed due to Exception: {str(e)}")

    def _merge(self, their: "BatchedRepetitionPenalizer"):
        try:
            self.repetition_penalties = torch.cat([self.repetition_penalties, their.repetition_penalties], dim=0)
            self.cumulated_repetition_penalties = torch.cat(
                [
                    self.cumulated_repetition_penalties,
                    their.cumulated_repetition_penalties,
                ],
                dim=0,
            )
        except Exception as e:
            logger.info(f"merge repetiton penalty failed due to Exception: {str(e)}")
