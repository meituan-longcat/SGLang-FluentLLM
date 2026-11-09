from __future__ import annotations

import dataclasses
import threading
import time
from typing import TYPE_CHECKING, Any, Callable

import torch

import sglang.srt.sampling.penaltylib as penaltylib
from sglang.srt.env import global_server_args_dict
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor
from sglang.srt.sampling.penaltylib.repetition_penalty import apply_scaling_penalties
from sglang.srt.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


if TYPE_CHECKING:
    from sglang.srt.managers.scheduling.schedule_batch import ScheduleBatch


@dataclasses.dataclass
class SamplingBatchInfo:
    # Basic batched sampling params
    temperatures: torch.Tensor
    top_ps: torch.Tensor
    top_ks: torch.Tensor
    min_ps: torch.Tensor

    # This will be a snapshot of repeat_penalties of reqs in ForwardBatch. Since the sampling of
    # spec-decoding is in the middle of a cudagraph and we need to cumulate multi output tokens,
    # the penalties in penalizer_orchestrator is difficult to use. So make it permanent like temperatures
    # and top_ks, and cumulate/apply repetition_penalties on forward_thread(for Spec-Decoding).
    # It's default value is 1.0 in samling_params.
    repetition_penalties: torch.Tensor

    # Whether all requests use greedy sampling
    is_all_greedy: bool

    # Whether any request needs min_p sampling
    need_min_p_sampling: bool

    # Masking tensors for grammar-guided structured outputs
    vocab_size: int
    grammars: list | None = None
    vocab_mask: torch.Tensor | None = None
    apply_mask_func: Callable[[torch.Tensor, torch.Tensor], None] | None = None
    # Per-request mask-budget overruns from the last update_regex_vocab_mask.
    # Only proposals: the sampler agrees on them across the attention-TP group
    # before degrading anything. None when the budget is disabled.
    grammar_mask_overrun: list[bool] | None = None

    # An event used for overlap schedule
    sampling_info_done: threading.Event | None = None

    # Penalizer
    penalizer_orchestrator: penaltylib.BatchedPenalizerOrchestrator | None = None
    linear_penalty: torch.Tensor = None
    scaling_penalty: torch.Tensor = None

    # Whether any request has custom logit processor
    has_custom_logit_processor: bool = False
    # Custom parameters
    custom_params: list[dict[str, Any] | None] | None = None
    # Custom logit processor
    custom_logit_processor: dict[int, tuple[CustomLogitProcessor, torch.Tensor]] | None = None

    # Use thinking_budget to truncate thinking
    thinking_budgets: torch.Tensor | None = None
    think_end_ids: torch.Tensor | None = None
    last_token_ids: torch.Tensor | None = None

    # Device
    device: str = "cuda"

    @classmethod
    def from_schedule_batch(cls, batch: ScheduleBatch, vocab_size: int):
        reqs = batch.reqs
        device = batch.device
        temperatures = (
            torch.tensor(
                [r.sampling_params.temperature for r in reqs],
                dtype=torch.float,
            )
            .view(-1, 1)
            .to(device, non_blocking=True)
        )
        top_ps = torch.tensor([r.sampling_params.top_p for r in reqs], dtype=torch.float).to(device, non_blocking=True)
        top_ks = torch.tensor([r.sampling_params.top_k for r in reqs], dtype=torch.int32).to(device, non_blocking=True)
        min_ps = torch.tensor([r.sampling_params.min_p for r in reqs], dtype=torch.float).to(device, non_blocking=True)
        repetition_penalties = torch.tensor(
            [r.sampling_params.repetition_penalty for r in reqs],
            dtype=torch.float32,
        ).to(device, non_blocking=True)

        if any(hasattr(r.tokenizer, "think_end_id") and r.sampling_params.thinking_budget is not None for r in reqs):
            thinking_budgets = torch.tensor(
                [
                    # When reasoning is on (this block only runs if some req has
                    # think_end_id), create_grammar_backend wraps every grammar in
                    # a ReasonerGrammarObject, which enforces thinking_budget itself
                    # via masking (only think_end_id survives once exhausted). So any
                    # req that has a grammar already owns its budget: -1 makes both
                    # apply_thinking_budgets and _spec skip this row, leaving the
                    # predicts-overwrite path only for grammar-less reqs.
                    -1 if r.grammar is not None else len(r.origin_input_ids) + (r.sampling_params.thinking_budget or r.sampling_params.max_new_tokens)
                    for r in reqs
                ],
                dtype=torch.int32,
            ).to(device, non_blocking=True)
            think_end_ids = torch.tensor(
                [[getattr(r.tokenizer, "think_end_id", -1)] for r in reqs],
                dtype=torch.int32,
            ).to(device, non_blocking=True)
            last_token_ids = torch.full_like(think_end_ids, -1)
        else:
            thinking_budgets = None
            think_end_ids = None
            last_token_ids = None

        # Check if any request has custom logit processor
        has_custom_logit_processor = (
            batch.enable_custom_logit_processor  # check the flag first.
            and any(r.custom_logit_processor for r in reqs)  # then check the requests.
        )

        if has_custom_logit_processor:
            # Merge the same type of custom logit processors together
            processor_dict = {}
            for i, r in enumerate(reqs):
                if r.custom_logit_processor is None:
                    continue
                processor_str = r.custom_logit_processor
                if processor_str not in processor_dict:
                    processor_dict[processor_str] = []
                processor_dict[processor_str].append(i)

            merged_custom_logit_processor = {
                hash(processor_str): (
                    # The deserialized custom logit processor object
                    CustomLogitProcessor.from_str(processor_str),
                    # The mask tensor for the requests that use this custom logit processor
                    torch.zeros(len(reqs), dtype=torch.bool).scatter_(0, torch.tensor(true_indices), True).to(device, non_blocking=True),
                )
                for processor_str, true_indices in processor_dict.items()
            }
            custom_params = [r.sampling_params.custom_params for r in reqs]
        else:
            merged_custom_logit_processor = None
            custom_params = None

        # Each penalizers will do nothing if they evaluate themselves as not required by looking at
        # the sampling_params of the requests (See {_is_required()} of each penalizers). So this
        # should not add hefty computation overhead other than simple checks.
        #
        # While we can choose not to even create the class instances if they are not required, this
        # could add additional complexity to the {ScheduleBatch} class, especially we need to
        # handle {filter_batch()} and {merge_batch()} cases as well.
        penalizer_orchestrator = penaltylib.BatchedPenalizerOrchestrator(
            vocab_size=vocab_size,
            batch=batch,
            penalizers={
                penaltylib.BatchedFrequencyPenalizer,
                penaltylib.BatchedMinNewTokensPenalizer,
                penaltylib.BatchedPresencePenalizer,
                penaltylib.BatchedRepetitionPenalizer,
            },
        )

        ret = cls(
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
            min_ps=min_ps,
            repetition_penalties=repetition_penalties,
            thinking_budgets=thinking_budgets,
            think_end_ids=think_end_ids,
            last_token_ids=last_token_ids,
            is_all_greedy=all(r.sampling_params.top_k <= 1 for r in reqs),
            need_min_p_sampling=any(r.sampling_params.min_p > 0 for r in reqs),
            vocab_size=vocab_size,
            penalizer_orchestrator=penalizer_orchestrator,
            has_custom_logit_processor=has_custom_logit_processor,
            custom_params=custom_params,
            custom_logit_processor=merged_custom_logit_processor,
            device=device,
        )
        return ret

    def __len__(self):
        return len(self.temperatures)

    def update_regex_vocab_mask(self):
        if not self.grammars:
            self.vocab_mask = None
            self.apply_mask_func = None
            return

        # Find a grammar from the list
        first_grammar = next(grammar for grammar in self.grammars if grammar)

        # TODO(lianmin): Maybe we can reuse the existing mask?
        self.vocab_mask = first_grammar.allocate_vocab_mask(
            vocab_size=self.vocab_size,
            batch_size=len(self.temperatures),
            device=self.device,
        )
        self.apply_mask_func = first_grammar.apply_vocab_mask  # force to use static method

        self.vocab_mask.fill_(-1)

        mask_budget_s = (global_server_args_dict.get("grammar_mask_timeout_ms") or 0.0) / 1000.0
        self.grammar_mask_overrun = [False] * len(self.grammars) if mask_budget_s > 0 else None

        # Apply the mask
        for i, grammar in enumerate(self.grammars):
            if grammar and not grammar.finished:
                if not grammar.is_terminated():
                    if mask_budget_s > 0 and getattr(grammar, "slow_hint", False):
                        # Recently slow schema: propose a degrade instead of
                        # paying a possibly multi-second fill (row stays allow-all).
                        self.grammar_mask_overrun[i] = True
                        continue
                    start = time.monotonic()
                    grammar.fill_vocab_mask(self.vocab_mask, i)
                    if mask_budget_s > 0 and time.monotonic() - start > mask_budget_s:
                        self.grammar_mask_overrun[i] = True
                else:
                    # Grammar already terminated — set all-pass bitmask so that
                    # subsequent multi-step decode forward passes can sample any
                    # token instead of being blocked by the default all-zero mask.
                    self.vocab_mask[i].fill_(-1)

        # Move the mask to the device if needed
        self.vocab_mask = first_grammar.move_vocab_mask(self.vocab_mask, self.device)

    def update_penalties(self):
        if self.penalizer_orchestrator.is_required:
            self.linear_penalty = torch.zeros(
                (len(self.temperatures), self.vocab_size),
                dtype=torch.float32,
                device=self.temperatures.device,
            )
            # NOTE: This is specialized for repetition penalty, since it's not a
            # linear verison penalty. I don't why we still need this, but add a
            # multiply version in case of more options in the future.
            self.scaling_penalty = torch.ones(
                (len(self.temperatures), self.vocab_size),
                dtype=torch.float32,
                device=self.temperatures.device,
            )
            self.penalizer_orchestrator.apply(self.linear_penalty)
            for penalizer in self.penalizer_orchestrator.penalizers.values():
                if isinstance(penalizer, penaltylib.BatchedRepetitionPenalizer) and penalizer.is_prepared():
                    penalizer._update_scaling_penalty(self.scaling_penalty)
        else:
            self.linear_penalty = None
            self.scaling_penalty = None

    def apply_thinking_budgets(self, seq_lens: torch.Tensor, next_token_ids: torch.Tensor):
        diff_lens = seq_lens - self.thinking_budgets
        bs, think_end_size = self.think_end_ids.size()
        torch.where(
            (self.thinking_budgets > 0) & (diff_lens >= 0) & (diff_lens < think_end_size),
            self.think_end_ids[
                torch.arange(bs, device=self.think_end_ids.device),
                diff_lens.clamp(min=0, max=think_end_size - 1),
            ],
            next_token_ids,
            out=next_token_ids,
        )

        self.last_token_ids[:] = torch.cat([self.last_token_ids[:, 1:], next_token_ids.unsqueeze(1)], dim=1)
        end_rows = (self.last_token_ids == self.think_end_ids).all(dim=1)
        self.thinking_budgets[(self.thinking_budgets > 0) & end_rows] = -1

        return next_token_ids

    def apply_thinking_budgets_spec(
        self,
        seq_lens: torch.Tensor,
        predicts: torch.Tensor,
        accept_length: torch.Tensor,
        draft_token_num: int,
    ) -> None:
        bs: int = self.thinking_budgets.shape[0]
        device = predicts.device
        seq_lens = seq_lens[:bs]
        acc = accept_length[:bs].to(torch.long)
        k = (self.thinking_budgets - seq_lens - 1).to(torch.long)
        end_ids = self.think_end_ids[:, 0].to(predicts.dtype)
        pred = predicts.view(-1, draft_token_num)[:bs]
        accepted = torch.arange(draft_token_num, device=device).unsqueeze(0) <= acc.unsqueeze(1)
        natural = (accepted & (pred == end_ids.unsqueeze(1))).any(dim=1)
        active = (self.thinking_budgets > 0) & (k >= 0) & (k <= acc) & ~natural
        rows = torch.arange(bs, device=device)
        flat_pos = rows * draft_token_num + k.clamp(min=0, max=draft_token_num - 1)
        predicts[flat_pos] = torch.where(active, end_ids, predicts[flat_pos])
        accept_length[:bs] = torch.where(active, k.to(accept_length.dtype), acc)
        self.thinking_budgets[:] = torch.where(active | natural, torch.full_like(self.thinking_budgets, -1), self.thinking_budgets)

    def apply_logits_bias(self, logits: torch.Tensor):
        if self.linear_penalty is not None:
            # Used in the overlap mode
            logits.add_(self.linear_penalty)

        if self.scaling_penalty is not None:
            # Used in the overlap mode
            apply_scaling_penalties(logits, self.scaling_penalty)

        if self.penalizer_orchestrator and self.penalizer_orchestrator.is_required:
            # Used in the non-overlap mode
            self.penalizer_orchestrator.apply(logits)

        if self.vocab_mask is not None:
            self.apply_mask_func(logits=logits, vocab_mask=self.vocab_mask)

    def filter_batch(self, keep_indices: list[int], keep_indices_device: torch.Tensor):
        if self.has_custom_logit_processor:
            self._filter_batch_custom_logit_processor(keep_indices, keep_indices_device)

        if self.penalizer_orchestrator and self.penalizer_orchestrator.is_required:
            # Used in the non-overlap mode
            self.penalizer_orchestrator.filter(keep_indices_device)

        for item in [
            "temperatures",
            "top_ps",
            "top_ks",
            "min_ps",
            "repetition_penalties",
            "thinking_budgets",
            "think_end_ids",
            "last_token_ids",
        ]:
            value = getattr(self, item, None)
            if value is not None:
                setattr(self, item, value[keep_indices_device])

    def _filter_batch_custom_logit_processor(self, keep_indices: list[int], keep_indices_device: torch.Tensor):
        """Filter the custom logit processor and custom params"""
        self.custom_logit_processor = {
            k: (p, mask[keep_indices_device])
            for k, (p, mask) in self.custom_logit_processor.items()
            if torch.any(mask[keep_indices_device])  # ignore the custom logit processor whose mask is all False
        }
        self.custom_params = [self.custom_params[i] for i in keep_indices]

        # If the custom logit processor is an empty dict, set the flag to False,
        # and set the custom logit processor and custom params to None.
        if len(self.custom_logit_processor) == 0:
            self.custom_logit_processor = None
            self.custom_params = None
            self.has_custom_logit_processor = False

    @staticmethod
    def merge_custom_logit_processor(
        lhs: dict[int, tuple[CustomLogitProcessor, torch.Tensor]] | None,
        rhs: dict[int, tuple[CustomLogitProcessor, torch.Tensor]] | None,
        bs1: int,
        bs2: int,
        device: str,
    ):
        if lhs is None and rhs is None:
            return None
        lhs, rhs = lhs or {}, rhs or {}

        keys = set(lhs.keys()).union(set(rhs.keys()))
        merged_dict = {}

        for k in keys:
            # Get the logit processor object
            processor = lhs[k][0] if k in lhs else rhs[k][0]
            # Get and merge the mask tensors from the two dicts
            left_mask = lhs[k][1] if k in lhs else torch.zeros(bs1, dtype=torch.bool, device=device)
            right_mask = rhs[k][1] if k in rhs else torch.zeros(bs2, dtype=torch.bool, device=device)
            merged_dict[k] = (processor, torch.cat([left_mask, right_mask]))

            assert merged_dict[k][1].shape[0] == bs1 + bs2, (
                f"The batch size of merged mask ({merged_dict[k][1].shape[0]}) does not match "
                f"the sum of the batch sizes of the two masks ({bs1 + bs2})"
                f"\n{left_mask=}\n{right_mask=}\n{bs1=}\n{bs2=}"
                f"\n{lhs=}\n{rhs=}"
            )

        return merged_dict

    def merge_batch(self, other: SamplingBatchInfo):
        self.penalizer_orchestrator.merge(other.penalizer_orchestrator)

        # Merge the custom logit processors and custom params lists
        if self.has_custom_logit_processor or other.has_custom_logit_processor:
            # Merge the custom logit processors
            self.custom_logit_processor = SamplingBatchInfo.merge_custom_logit_processor(
                self.custom_logit_processor,
                other.custom_logit_processor,
                len(self),
                len(other),
                self.device,
            )
            # Merge the custom params lists
            self.custom_params = self.custom_params or [None] * len(self)
            other.custom_params = other.custom_params or [None] * len(other)
            self.custom_params.extend(other.custom_params)

            # Set the flag to True if any of the two has custom logit processor
            self.has_custom_logit_processor = True

        # Note: becasue the __len()__ operator is defined on the temperatures tensor,
        # please make sure any merge operation with len(self) or len(other) is done before
        # the merge operation of the temperatures tensor below.
        self_batch_size = len(self)
        other_batch_size = len(other)
        for item in [
            "temperatures",
            "repetition_penalties",
            "top_ps",
            "top_ks",
            "min_ps",
        ]:
            self_val = getattr(self, item, None)
            other_val = getattr(other, item, None)
            if self_val is not None and other_val is not None:
                setattr(self, item, torch.concat([self_val, other_val]))

        if self.thinking_budgets is not None or other.thinking_budgets is not None:
            for item in ["thinking_budgets", "think_end_ids", "last_token_ids"]:
                self_val = getattr(self, item)
                other_val = getattr(other, item)
                template = self_val if self_val is not None else other_val
                if self_val is None:
                    self_val = torch.full(
                        (self_batch_size, *template.shape[1:]), -1,
                        dtype=template.dtype, device=template.device,
                    )
                if other_val is None:
                    other_val = torch.full(
                        (other_batch_size, *template.shape[1:]), -1,
                        dtype=template.dtype, device=template.device,
                    )
                setattr(self, item, torch.concat([self_val, other_val]))

        # A mixed batch must retain per-request sampling parameters.
        self.is_all_greedy &= other.is_all_greedy
        self.need_min_p_sampling |= other.need_min_p_sampling
