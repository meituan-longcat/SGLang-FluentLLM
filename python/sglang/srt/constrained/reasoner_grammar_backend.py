# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Reasoner grammar backend: wraps a grammar object so that a grammar
constraint (json_schema/regex/ebnf/...) is NOT applied during the thinking
phase, and thinking_budget is enforced by masking rather than by overwriting
sampled tokens.

Adapted from sglang's ReasonerGrammarObject. Device-agnostic: the bitmask,
DFS (traverse_tree) and apply machinery are shared by GPU and NPU. Unlike
open source, this fork's spec DFS calls ``allocate_token_bitmask.fill_(0)``
(block-all default, apply turns bit==0 into -inf), so the thinking phase must
*explicitly* fill the row with -1 (allow-all) instead of the open-source no-op
``return`` — on both GPU and NPU.
"""

from typing import Callable, Optional

import torch

from sglang.srt.constrained.base_grammar_backend import (
    BaseGrammarBackend,
    BaseGrammarObject,
)


def _think_end_bitmask_word(think_end_id: int) -> tuple[int, int]:
    """Return (word_index, signed_int32_value) that sets only think_end_id's
    bit inside an all-zero (block-all) bitmask row."""
    word = think_end_id // 32
    bit = think_end_id % 32
    value = 1 << bit
    if value >= (1 << 31):  # wrap to signed int32 (bit 31 -> negative)
        value -= 1 << 32
    return word, value


class ReasonerGrammarObject(BaseGrammarObject):
    """Wraps an inner grammar with a THINKING/GENERATION state machine.

    THINKING (tokens_in_think >= 0, tokens_after_end == -1):
      inner grammar is NOT consulted. accept_token does not advance it,
      fill_vocab_mask allows all tokens (or, once the thinking budget is
      exhausted, only think_end_id) so the model is forced to close thinking.
    GENERATION (tokens_after_end >= 0):
      inner grammar drives accept/fill/rollback as usual.

    A freshly constructed object defaults to GENERATION, so an object that is
    never configured behaves exactly like the plain inner grammar (coherent
    default, no reliance on a follow-up call). Call ``set_reasoning(True)`` to
    (re)start a reasoning request in THINKING.
    """

    def __init__(
        self,
        grammar: BaseGrammarObject,
        think_end_id: int,
        max_think_tokens: int = -1,
    ) -> None:
        self.grammar = grammar
        self.think_end_id = think_end_id
        self.max_think_tokens = max_think_tokens
        self._think_end_word, self._think_end_value = _think_end_bitmask_word(think_end_id)

        # State machine. Default to GENERATION so an object that never gets
        # set_reasoning() behaves as the plain inner grammar. set_reasoning(True)
        # flips a reasoning request into THINKING.
        self.tokens_in_think = -1
        self.tokens_after_end = 0

    def set_reasoning(self, reasoning: bool) -> None:
        if reasoning:
            self.tokens_in_think = 0
            self.tokens_after_end = -1
        else:
            self.tokens_in_think = -1
            self.tokens_after_end = 0

    def _is_thinking(self) -> bool:
        return self.tokens_in_think >= 0 and self.tokens_after_end == -1

    def _is_generation(self) -> bool:
        return self.tokens_after_end >= 0

    def _can_think_more(self) -> bool:
        return self.max_think_tokens < 0 or self.tokens_in_think < self.max_think_tokens

    def transfer_state(self, token: int) -> None:
        if self._is_thinking():
            if token == self.think_end_id:
                self.tokens_after_end = 0
            else:
                self.tokens_in_think += 1
        elif self._is_generation():
            self.tokens_after_end += 1

    def rollback_state(self) -> None:
        if self._is_thinking():
            if self.tokens_in_think > 0:
                self.tokens_in_think -= 1
        elif self._is_generation():
            if self.tokens_after_end == 0:
                self.tokens_after_end = -1
            elif self.tokens_after_end > 0:
                self.tokens_after_end -= 1

    # ---- grammar interface -------------------------------------------------

    def accept_token(self, token: int) -> None:
        if self._is_generation():
            self.grammar.accept_token(token)
        self.transfer_state(token)

    def is_terminated(self) -> bool:
        # During thinking the inner grammar has not advanced; a freshly
        # compiled matcher is not terminated, so delegating is safe and keeps
        # the DFS walking down the draft chain.
        return self.grammar.is_terminated()

    def rollback(self, k: int) -> None:
        steps_after = min(k, max(0, self.tokens_after_end))
        if steps_after > 0:
            self.grammar.rollback(steps_after)
        for _ in range(k):
            self.rollback_state()

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        if self._is_thinking():
            if self._can_think_more():
                # allow-all: every bit set (NPU bit==1 -> keep logit).
                vocab_mask[idx].fill_(-1)
            else:
                # budget exhausted: block everything, allow only think_end_id
                # so the model is forced to close the thinking phase.
                vocab_mask[idx].fill_(0)
                vocab_mask[idx, self._think_end_word] = self._think_end_value
            return
        self.grammar.fill_vocab_mask(vocab_mask, idx)

    @property
    def degraded(self) -> bool:
        return bool(getattr(self.grammar, "degraded", False))

    def degrade(self, reason: str) -> None:
        # Only the inner grammar is dropped; the thinking budget is still
        # enforced by fill_vocab_mask above.
        degrade_fn = getattr(self.grammar, "degrade", None)
        if degrade_fn is not None:
            degrade_fn(reason)

    @property
    def slow_hint(self) -> bool:
        # The inner grammar is only consulted after think_end.
        return self._is_generation() and bool(getattr(self.grammar, "slow_hint", False))

    def fill_draft_tree_bitmask(
        self,
        retrieve_next_token: torch.Tensor,
        retrieve_next_sibling: torch.Tensor,
        draft_tokens: torch.Tensor,
        token_bitmask: torch.Tensor,
        time_threshold_s: float = -1.0,
    ) -> bool:
        """Fill the draft-tree mask; in GENERATION delegate to the inner
        grammar's (native) traversal.

        During thinking the draft chain may cross think_end. Only the Python
        DFS advances THINKING -> GENERATION node by node, so the nodes after
        think_end get the inner grammar's mask (and, once the thinking budget
        is exhausted, only the first node is think_end-only). The DFS does not
        consult the inner grammar before think_end, so it is cheap.
        """
        if self._is_thinking():
            return super().fill_draft_tree_bitmask(
                retrieve_next_token,
                retrieve_next_sibling,
                draft_tokens,
                token_bitmask,
                time_threshold_s=time_threshold_s,
            )
        return self.grammar.fill_draft_tree_bitmask(
            retrieve_next_token,
            retrieve_next_sibling,
            draft_tokens,
            token_bitmask,
            time_threshold_s=time_threshold_s,
        )

    def allocate_vocab_mask(self, vocab_size: int, batch_size: int, device) -> torch.Tensor:
        return self.grammar.allocate_vocab_mask(vocab_size, batch_size, device)

    def move_vocab_mask(self, vocab_mask: torch.Tensor, device) -> torch.Tensor:
        return self.grammar.move_vocab_mask(vocab_mask, device)

    @property
    def apply_vocab_mask(self):
        return self.grammar.apply_vocab_mask

    @property
    def finished(self):
        return self.grammar.finished

    @finished.setter
    def finished(self, value):
        self.grammar.finished = value

    def try_jump_forward(self, tokenizer) -> Optional[tuple[list[int], str]]:
        return self.grammar.try_jump_forward(tokenizer)

    def jump_forward_str_state(self, helper) -> tuple[str, int]:
        return self.grammar.jump_forward_str_state(helper)

    def jump_and_retokenize(self, old_output_ids: list[int], new_output_ids: list[int], next_state: int):
        return self.grammar.jump_and_retokenize(old_output_ids, new_output_ids, next_state)

    def copy(self) -> "ReasonerGrammarObject":
        new_obj = ReasonerGrammarObject(self.grammar.copy(), self.think_end_id, self.max_think_tokens)
        new_obj.tokens_in_think = self.tokens_in_think
        new_obj.tokens_after_end = self.tokens_after_end
        return new_obj


class ReasonerGrammarBackend(BaseGrammarBackend):
    """Wraps an inner grammar backend, producing ReasonerGrammarObject values.

    Per-request thinking state (reasoning on/off, max_think_tokens) is applied
    by the scheduler after the grammar object is resolved, since the inner
    grammar is cached by grammar key only.
    """

    def __init__(self, grammar_backend: BaseGrammarBackend, think_end_id: int):
        super().__init__()
        self.grammar_backend = grammar_backend
        self.think_end_id = think_end_id

    def init_value_impl(self, key: tuple[str, str]) -> Optional[ReasonerGrammarObject]:
        inner = self.grammar_backend.init_value_impl(key)
        if inner is None:
            return None
        return ReasonerGrammarObject(inner, self.think_end_id)

    def set_degrade_reporter(
        self, reporter: Optional[Callable[[str, Optional[str]], None]]
    ) -> None:
        # The inner grammar owns the degraded state (ReasonerGrammarObject.degrade
        # delegates to it), so the reporter must reach the wrapped backend.
        self.grammar_backend.set_degrade_reporter(reporter)

    def reset(self):
        super().reset()
        self.grammar_backend.reset()

    def reset_vocab_masks(self, vocab_masks: torch.Tensor):
        self.grammar_backend.reset_vocab_masks(vocab_masks)
