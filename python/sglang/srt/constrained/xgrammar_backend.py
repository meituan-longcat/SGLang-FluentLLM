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
"""Constrained decoding with xgrammar backend."""

import json
import re
import threading
import time
from typing import Callable, Optional, Union

import torch
from xgrammar import (
    CompiledGrammar,
    GrammarCompiler,
    GrammarMatcher,
    StructuralTagItem,
    TokenizerInfo,
    allocate_token_bitmask,
    apply_token_bitmask_inplace,
)

from sglang.srt.constrained.base_grammar_backend import (
    BaseGrammarBackend,
    BaseGrammarObject,
)
from sglang.srt.utils import get_colorful_logger, is_npu

logger = get_colorful_logger(__name__)


# xgrammar 0.2.x has a latency regression for bounded repetitions with a large
# lower bound (e.g. ``U{900,}``): a single fill_next_token_bitmask can take tens
# of seconds, and traverse_draft_tree's time_threshold can only abort between
# nodes, not inside one fill. Detect such patterns at compile time and run the
# request unconstrained instead of letting the first fill stall the DP/EP group.
MAX_SAFE_REPETITION = 128
_REPETITION_RE = re.compile(r"\{(\d+)(?:,(\d*))?\}")


def _find_large_repetition(schema_str: str) -> Optional[str]:
    """Return a description if a JSON Schema pattern has a large repetition."""
    try:
        obj = json.loads(schema_str)
    except Exception:
        return None
    stack = [obj]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            pattern = node.get("pattern")
            if isinstance(pattern, str):
                for match in _REPETITION_RE.finditer(pattern):
                    lower = int(match.group(1))
                    if lower > MAX_SAFE_REPETITION:
                        return f"pattern repetition {match.group(0)} has lower bound {lower} > {MAX_SAFE_REPETITION}"
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return None


# A schema degraded on SLOW_KEY_STRIKES requests within SLOW_KEY_WINDOW_S is
# treated as slow for the next SLOW_KEY_WINDOW_S: new requests propose a
# degrade up front instead of each paying a possibly multi-second first fill.
# A single misjudged overrun (e.g. CPU contention) never disables a schema,
# and the slow mark expires.
SLOW_KEY_STRIKES = 3
SLOW_KEY_WINDOW_S = 600.0


class SlowGrammarKeys:
    """Per-backend record of schemas whose requests keep getting degraded.

    Only a hint: ``is_slow`` makes new requests *propose* a degrade, and the
    proposal still goes through the attention-TP consensus, so ranks whose
    records differ (e.g. a mark expiring at slightly different times) still
    make the same decision.
    """

    def __init__(self):
        self._lock = threading.Lock()
        # key -> (strikes in the current window, window start)
        self._strikes: dict[tuple[str, str], tuple[int, float]] = {}
        # key -> monotonic time until which the key is slow
        self._slow_until: dict[tuple[str, str], float] = {}

    def record_degrade(self, key: tuple[str, str], label: Optional[str]) -> None:
        now = time.monotonic()
        with self._lock:
            strikes, start = self._strikes.get(key, (0, now))
            if now - start > SLOW_KEY_WINDOW_S:
                strikes, start = 0, now
            strikes += 1
            if strikes < SLOW_KEY_STRIKES:
                self._strikes[key] = (strikes, start)
                return
            self._strikes.pop(key, None)
            self._slow_until[key] = now + SLOW_KEY_WINDOW_S
        logger.error(
            "Grammar schema marked slow for %.0fs after %d degraded requests; new requests propose a degrade: label=%s",
            SLOW_KEY_WINDOW_S,
            SLOW_KEY_STRIKES,
            label,
        )

    def is_slow(self, key: tuple[str, str]) -> bool:
        with self._lock:
            until = self._slow_until.get(key)
            if until is None:
                return False
            if time.monotonic() < until:
                return True
            del self._slow_until[key]
            return False

    def clear(self) -> None:
        with self._lock:
            self._strikes.clear()
            self._slow_until.clear()


class XGrammarGrammar(BaseGrammarObject):
    _bit_weights: Optional[torch.Tensor] = None

    def __init__(
        self,
        matcher: GrammarMatcher,
        vocab_size: int,
        ctx: CompiledGrammar,
        override_stop_tokens: Optional[Union[list[int], int]],
        grammar_label: Optional[str] = None,
        slow_keys: Optional[SlowGrammarKeys] = None,
        grammar_key: Optional[tuple[str, str]] = None,
        degrade_reporter: Optional[Callable[[str, Optional[str]], None]] = None,
    ) -> None:
        self.matcher = matcher
        self.vocab_size = vocab_size
        self.ctx = ctx
        self.override_stop_tokens = override_stop_tokens
        self.finished = False
        self.accepted_tokens = []
        self.grammar_label = grammar_label
        # Shared by every matcher of this backend (see SlowGrammarKeys).
        self._slow_keys = slow_keys
        self._grammar_key = grammar_key
        # Called once per request on the rank that reports metrics (see
        # SchedulerMetricsCollector.observe_grammar_degrade).
        self._degrade_reporter = degrade_reporter
        # The schema recently overran repeatedly: a budgeted mask fill skips
        # the work and proposes a degrade instead (see fill_draft_tree_bitmask).
        self.slow_hint = False
        # Once degraded, the request runs unconstrained for the rest of its
        # lifetime. We deliberately do NOT set `finished`/`is_terminated`,
        # since those would make the scheduler end the request.
        self.degraded = False
        self.degraded_reason: Optional[str] = None

    def degrade(self, reason: str) -> None:
        """Switch this request to unconstrained generation.

        A pathological grammar must not be allowed to block the forward thread
        (and therefore the whole DP/EP group). The request keeps running; only
        its structured-output guarantee is lost.

        The mask budget is wall-clock based, so it is never applied here
        directly: callers collect per-request overruns, agree on them across
        the attention-TP group (``degrade_grammars_by_consensus``) and only then
        call this, so every rank degrades the same requests at the same step.
        """
        if self.degraded:
            return
        self.degraded = True
        self.degraded_reason = reason
        self.slow_hint = False
        if self._slow_keys is not None and self._grammar_key is not None:
            self._slow_keys.record_degrade(self._grammar_key, self.grammar_label)
        logger.error(
            "Grammar mask degraded to unconstrained generation: reason=%s label=%s",
            reason,
            self.grammar_label,
        )
        if self._degrade_reporter is not None:
            try:
                self._degrade_reporter(reason, self.grammar_label)
            except Exception:
                # CAT must never block or fail the forward path.
                logger.warning("Failed to report GrammarDegrade metric", exc_info=True)

    def is_terminated(self):
        if self.degraded:
            return False
        return self.matcher.is_terminated()

    def accept_token(self, token: int):
        if self.degraded:
            return
        # The reasoner wrapper's draft-tree traversal yields scalar CPU tensors.
        # xgrammar's TVM FFI requires a Python integer at this boundary.
        token = int(token)
        if not self.is_terminated():
            accepted = self.matcher.accept_token(token)
            if not accepted:
                # log for debugging
                raise ValueError(
                    f"Tokens not accepted: {token}\nAccepted tokens: {self.accepted_tokens}\nTerminated: {self.matcher.is_terminated()}\n"
                )
        else:
            self.accepted_tokens.append(token)

    def rollback(self, k: int):
        if self.degraded:
            return
        self.matcher.rollback(k)
        self.accepted_tokens = self.accepted_tokens[:-k]

    def try_jump_forward(self, tokenizer) -> tuple[list[int], str]:
        if self.degraded:
            return None
        s = self.matcher.find_jump_forward_string()
        if s:
            return [], s
        return None

    def jump_forward_str_state(self, helper: tuple[list[int], str]) -> tuple[str, int]:
        _, data = helper
        return data, -1

    def jump_and_retokenize(self, old_output_ids: list[int], new_output_ids: list[int], next_state: int):
        if self.degraded:
            return
        k = 0
        for i, old_id in enumerate(old_output_ids):
            if old_id == new_output_ids[i]:
                k = i + 1
            else:
                break

        # rollback to the last token that is the same
        if k < len(old_output_ids):
            self.matcher.rollback(len(old_output_ids) - k)

        for i in range(k, len(new_output_ids)):
            assert self.matcher.accept_token(new_output_ids[i])

    def allocate_vocab_mask(self, vocab_size: int, batch_size: int, device) -> torch.Tensor:
        return allocate_token_bitmask(batch_size, vocab_size)

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        if self.degraded:
            vocab_mask[int(idx)].fill_(-1)
            return
        self.matcher.fill_next_token_bitmask(vocab_mask, int(idx))

    def fill_draft_tree_bitmask(
        self,
        retrieve_next_token: torch.Tensor,
        retrieve_next_sibling: torch.Tensor,
        draft_tokens: torch.Tensor,
        token_bitmask: torch.Tensor,
        time_threshold_s: float = -1.0,
    ) -> bool:
        # Native C++ DFS over the draft tree: one Python->C++ crossing per
        # request instead of ~4 per tree node. traverse_draft_tree writes the
        # visited rows but leaves untouched tail rows as-is, so we reset the
        # slice to disallow-all first to match the Python traverse_tree baseline
        # (unvisited nodes must be all-zero, not the caller's allow-all fill).
        # retrieve_next_token / retrieve_next_sibling / draft_tokens must be
        # int64 CPU tensors; token_bitmask int32 (xgrammar API contract).
        # Unlike the Python DFS, time_threshold lets the C++ traversal stop in
        # the middle of a pathological tree (between nodes, not inside a fill).
        # It is measured inside C++, so time spent re-acquiring the GIL after
        # the call does not count. Returns False on timeout (the rows are then
        # allow-all and the caller proposes a degrade); xgrammar restores the
        # matcher state either way.
        if self.degraded:
            token_bitmask.fill_(-1)
            return True
        if time_threshold_s > 0 and self.slow_hint:
            token_bitmask.fill_(-1)
            return False
        token_bitmask.fill_(0)
        completed = self.matcher.traverse_draft_tree(
            retrieve_next_token,
            retrieve_next_sibling,
            draft_tokens,
            token_bitmask,
            time_threshold=time_threshold_s,
        )
        if not completed:
            token_bitmask.fill_(-1)
        return completed

    @staticmethod
    def move_vocab_mask(vocab_mask: torch.Tensor, device) -> torch.Tensor:
        return vocab_mask.to(device, non_blocking=True)

    @staticmethod
    @torch.inference_mode()
    def apply_token_bitmask_inplace_npu(
        logits: torch.Tensor, vocab_mask: torch.Tensor
    ) -> None:
        # PP prefill sampling runs outside ModelRunner.forward's inference scope.
        # Its logits are inference tensors, so in-place masking needs its own scope.
        # Signed int32 AND preserves all 32 mask bits, including the sign bit.
        # Avoid expanding to int64 and dividing every vocabulary element.
        vocab_size = min(logits.shape[-1], vocab_mask.shape[-1] * 32)
        words = vocab_mask.contiguous().view(torch.int32)
        weights = XGrammarGrammar._bit_weights
        if weights is None or weights.device != vocab_mask.device:
            weights = torch.tensor(
                [1 << bit for bit in range(31)] + [-2147483648],
                dtype=torch.int32,
                device=vocab_mask.device,
            )
            XGrammarGrammar._bit_weights = weights
        bits = torch.bitwise_and(words.unsqueeze(-1), weights)
        bits = bits.reshape(*vocab_mask.shape[:-1], vocab_mask.shape[-1] * 32)
        logits.data[..., :vocab_size].masked_fill_(
            bits[..., :vocab_size] == 0, float("-inf")
        )

    @staticmethod
    def apply_vocab_mask(logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
        if is_npu():
            XGrammarGrammar.apply_token_bitmask_inplace_npu(logits, vocab_mask)
        else:
            apply_token_bitmask_inplace(logits, vocab_mask)

    def copy(self):
        matcher = GrammarMatcher(
            self.ctx,
            override_stop_tokens=self.override_stop_tokens,
        )
        grammar = XGrammarGrammar(
            matcher,
            self.vocab_size,
            self.ctx,
            self.override_stop_tokens,
            grammar_label=self.grammar_label,
            slow_keys=self._slow_keys,
            grammar_key=self._grammar_key,
            degrade_reporter=self._degrade_reporter,
        )
        # A cached template must never propagate a degraded state to new
        # requests. A slow schema only sets the hint: the degrade itself is
        # still agreed on across the attention-TP group at the first mask.
        if self._slow_keys is not None and self._grammar_key is not None:
            grammar.slow_hint = self._slow_keys.is_slow(self._grammar_key)
        return grammar


class XGrammarGrammarBackend(BaseGrammarBackend):
    def __init__(
        self,
        tokenizer,
        vocab_size: int,
    ):
        super().__init__()

        tokenizer_info = TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
        override_stop_tokens = None

        self.grammar_compiler = GrammarCompiler(tokenizer_info=tokenizer_info)
        self.vocab_size = vocab_size
        self.override_stop_tokens = override_stop_tokens
        # Shared by all matchers produced from this backend.
        self.slow_keys = SlowGrammarKeys()
        self.degrade_reporter: Optional[Callable[[str, Optional[str]], None]] = None

    def set_degrade_reporter(
        self, reporter: Optional[Callable[[str, Optional[str]], None]]
    ) -> None:
        self.degrade_reporter = reporter

    def init_value_impl(self, key: tuple[str, str]) -> XGrammarGrammar:

        key_type, key_string = key
        if key_type == "json" and key_string != "$$ANY$$":
            slow_reason = _find_large_repetition(key_string)
            if slow_reason is not None:
                logger.error(
                    "Refusing pathological json_schema at compile time (%s); "
                    "this request will be aborted (to_abort_grammar). key=%s",
                    slow_reason,
                    key,
                )
                return None
        if key_type == "json":
            try:
                if key_string == "$$ANY$$":
                    ctx = self.grammar_compiler.compile_builtin_json_grammar()
                else:
                    ctx = self.grammar_compiler.compile_json_schema(schema=key_string)
            except RuntimeError as e:
                logger.warning(f"Skip invalid json_schema: json_schema={key_string}, {e=}")
                return None
        elif key_type == "ebnf":
            try:
                ctx = self.grammar_compiler.compile_grammar(key_string)
            except RuntimeError as e:
                logger.warning(f"Skip invalid ebnf: ebnf={key_string}, {e=}")
                return None
        elif key_type == "regex":
            try:
                ctx = self.grammar_compiler.compile_regex(key_string)
            except RuntimeError as e:
                logger.warning(f"Skip invalid regex: regex={key_string}, {e=}")
                return None
        elif key_type == "structural_tag":
            try:
                structural_tag = json.loads(key_string)
                tags = [
                    StructuralTagItem(
                        begin=structure["begin"],
                        schema=json.dumps(structure["schema"]),
                        end=structure["end"],
                    )
                    for structure in structural_tag["structures"]
                ]
                ctx = self.grammar_compiler.compile_structural_tag(tags, structural_tag["triggers"])
            except RuntimeError as e:
                logger.warning(f"Skip invalid regex: regex={key_string}, {e=}")
                return None
        else:
            raise ValueError(f"Invalid key_type: {key_type}")

        matcher = GrammarMatcher(ctx)

        # Validate that the grammar can produce at least one allowed token.
        # A degenerate grammar (e.g. self-referencing $ref) may compile without
        # error but produce an all-zero bitmask, which crashes multinomial sampling.
        validation_mask = allocate_token_bitmask(1, self.vocab_size)
        matcher.fill_next_token_bitmask(validation_mask)
        if validation_mask[0].sum().item() == 0:
            logger.warning(f"Skip degenerate grammar: compiled successfully but initial bitmask allows no tokens. key={key}")
            return None

        grammar = XGrammarGrammar(
            matcher,
            self.vocab_size,
            ctx,
            self.override_stop_tokens,
            grammar_label=f"{key[0]}:{key[1][:120]}",
            slow_keys=self.slow_keys,
            grammar_key=key,
            degrade_reporter=self.degrade_reporter,
        )
        grammar.slow_hint = self.slow_keys.is_slow(key)
        return grammar

    def reset(self):
        if self.grammar_compiler:
            self.grammar_compiler.clear_cache()
        self.slow_keys.clear()

    def reset_vocab_masks(self, vocab_masks: torch.Tensor):
        torch.fill_(vocab_masks, -1)
