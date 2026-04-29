import torch

from sglang.srt.sampling.penaltylib.orchestrator import (
    BatchedPenalizerOrchestrator,
    _BatchedPenalizer,
)
from sglang.srt.utils import get_compiler_backend
from sglang.srt.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


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

    def __init__(self, orchestrator: BatchedPenalizerOrchestrator):
        self.orchestrator = orchestrator
        self._is_prepared = False

    def _is_required(self) -> bool:
        return any(
            req.sampling_params.repetition_penalty != 1.0
            for req in self.orchestrator.reqs()
        )

    def _prepare(self):
        self.cumulated_repetition_penalties = torch.ones(
            (len(self.orchestrator.reqs()), self.orchestrator.vocab_size),
            dtype=torch.float32,
            device=self.orchestrator.device,
        )

        self.repetition_penalties = (
            torch.tensor(
                data=[
                    req.sampling_params.repetition_penalty
                    for req in self.orchestrator.reqs()
                ],
                dtype=torch.float32,
                device=self.orchestrator.device,
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
            logger.info(
                f"repetiton penalty _cumulate_output_tokens failed due to Exception: {str(e)}"
            )

    def _update_multiply_penalty(self, multiply_penalty: torch.Tensor) -> torch.Tensor:
        # multiply_penalty should be a tensor filled with 1
        assert multiply_penalty is not None
        if self._is_prepared:
            multiply_penalty.mul_(self.cumulated_repetition_penalties)

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        try:
            apply_scaling_penalties(logits, self.cumulated_repetition_penalties)
        except Exception as e:
            logger.info(f"apply repetiton penalty failed due to Exception: {str(e)}")
        return logits

    def _filter(self, keep_indices: torch.Tensor):
        try:
            self.repetition_penalties = self.repetition_penalties[keep_indices]
            self.cumulated_repetition_penalties = self.cumulated_repetition_penalties[
                keep_indices
            ]
        except Exception as e:
            logger.info(f"filter repetiton penalty failed due to Exception: {str(e)}")

    def _merge(self, their: "BatchedRepetitionPenalizer"):
        try:
            self.repetition_penalties = torch.cat(
                [self.repetition_penalties, their.repetition_penalties], dim=0
            )
            self.cumulated_repetition_penalties = torch.cat(
                [
                    self.cumulated_repetition_penalties,
                    their.cumulated_repetition_penalties,
                ],
                dim=0,
            )
        except Exception as e:
            logger.info(f"merge repetiton penalty failed due to Exception: {str(e)}")
