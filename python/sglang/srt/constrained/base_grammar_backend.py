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
"""The baseclass of a backend for grammar-guided constrained decoding."""

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

import torch

from sglang.srt.server_args import ServerArgs


class BaseGrammarObject:
    def fill_draft_tree_bitmask(
        self,
        retrieve_next_token: "torch.Tensor",  # type: ignore
        retrieve_next_sibling: "torch.Tensor",  # type: ignore
        draft_tokens: "torch.Tensor",  # type: ignore
        token_bitmask: "torch.Tensor",  # type: ignore
        time_threshold_s: float = -1.0,
    ) -> bool:
        """Fill ``token_bitmask`` (num_nodes x bitmask_size) for one request's
        draft tree. Default: the pure-Python DFS. Backends with a native
        implementation (xgrammar) override this. ``token_bitmask`` is reset to
        disallow-all before the allowed bits are set, so it is a drop-in for a
        direct ``traverse_tree`` call over the same slice.

        ``time_threshold_s`` > 0 is a best-effort budget checked between nodes.
        Returns False if it was exceeded (rows are then filled with allow-all
        bits), True otherwise."""
        deadline = time.monotonic() + time_threshold_s if time_threshold_s > 0 else None
        return traverse_tree(retrieve_next_token, retrieve_next_sibling, draft_tokens, self, token_bitmask, deadline=deadline)


class BaseGrammarBackend:
    def __init__(self):
        self.executor = ThreadPoolExecutor()
        self.cache: dict[tuple[str, str], Any] = {}

    def init_value_impl(self, key: tuple[str, str]) -> BaseGrammarObject:
        raise NotImplementedError()

    def get_cached_or_future_value(self, key: tuple[str, str]) -> tuple[Optional[BaseGrammarObject], bool]:
        # A cached value of None means the grammar is known to be invalid, so we
        # still treat it as a cache hit and skip recompilation/revalidation.
        if key in self.cache:
            value = self.cache[key]
            return (value.copy() if value is not None else None), True
        value = self.executor.submit(self.init_value_impl, key)
        return value, False

    def set_cache(self, key: tuple[str, str], value: BaseGrammarObject):
        self.cache[key] = value

    def set_degrade_reporter(
        self, reporter: Optional[Callable[[str, Optional[str]], None]]
    ) -> None:
        """Set a callback invoked once when a request's grammar degrades.

        Backends that do not support degradation keep the no-op default.
        """
        pass

    def reset(self):
        self.cache.clear()

    def reset_vocab_masks(self, vocab_masks: "torch.Tensor"):  # type: ignore
        raise NotImplementedError()


def create_grammar_backend(server_args: ServerArgs, tokenizer, vocab_size):
    if server_args.grammar_backend == "xgrammar":
        from sglang.srt.constrained.xgrammar_backend import XGrammarGrammarBackend

        grammar_backend = XGrammarGrammarBackend(tokenizer, vocab_size=vocab_size)
    else:
        raise ValueError(f"Invalid grammar backend: {server_args.grammar_backend}")

    # For reasoning models, wrap the backend so that grammar constraints are
    # suspended during the thinking phase and thinking_budget is enforced by
    # masking. This avoids injecting an ungrammared think_end token into the
    # verified stream (which crashes grammar.accept_token). The wrapper is
    # device-agnostic: the bitmask/DFS/apply machinery is shared by GPU and NPU.
    think_end_id = getattr(tokenizer, "think_end_id", None)
    if server_args.reasoning_parser and think_end_id is not None:
        from sglang.srt.constrained.reasoner_grammar_backend import (
            ReasonerGrammarBackend,
        )

        grammar_backend = ReasonerGrammarBackend(grammar_backend, think_end_id)

    return grammar_backend


def traverse_tree(
    retrieve_next_token: "torch.Tensor",  # type: ignore
    retrieve_next_sibling: "torch.Tensor",  # type: ignore
    draft_tokens: "torch.Tensor",  # type: ignore
    grammar: BaseGrammarObject,
    allocate_token_bitmask: "torch.Tensor",  # type: ignore
    deadline: Optional[float] = None,
) -> bool:
    """
    Traverse the tree constructed by the draft model to generate the logits mask.

    ``deadline`` is an absolute ``time.monotonic()`` value, checked between
    nodes. When it is reached the traversal stops, all rows are filled with
    allow-all bits, and False is returned. The DFS still unwinds through the
    normal rollback path so the grammar state is not left mid-traversal.
    """
    assert retrieve_next_token.shape == retrieve_next_sibling.shape == draft_tokens.shape, (
        f"retrieve_next_token={retrieve_next_token.shape}, retrieve_next_sibling={retrieve_next_sibling.shape}, draft_tokens={draft_tokens.shape}"
    )

    allocate_token_bitmask.fill_(0)
    aborted = False

    def dfs(
        curr: int,
        retrieve_next_token: "torch.Tensor",  # type: ignore
        retrieve_next_sibling: "torch.Tensor",  # type: ignore
        parent_pos: int,
    ):
        nonlocal aborted
        if aborted:
            return
        if deadline is not None and time.monotonic() >= deadline:
            aborted = True
            return
        if curr == 0:
            # the first token generated by the target model, and thus it is always
            # accepted from the previous iteration
            accepted = True
        else:
            parent_bitmask = allocate_token_bitmask[parent_pos]
            curr_token_id = draft_tokens[curr]
            # 32 boolean bitmask values are packed into 32-bit integers
            accepted = (parent_bitmask[curr_token_id // 32] & (1 << (curr_token_id % 32))) != 0

        if accepted:
            if curr != 0:
                # Accept the current token
                grammar.accept_token(draft_tokens[curr])
            if not grammar.is_terminated():
                # Generate the bitmask for the current token
                grammar.fill_vocab_mask(allocate_token_bitmask, curr)
                if retrieve_next_token[curr] != -1:
                    # Visit the child node
                    dfs(
                        retrieve_next_token[curr],
                        retrieve_next_token,
                        retrieve_next_sibling,
                        curr,
                    )

            if curr != 0:
                # Rollback the current token
                grammar.rollback(1)

        if retrieve_next_sibling[curr] != -1:
            # Visit the sibling node
            dfs(
                retrieve_next_sibling[curr],
                retrieve_next_token,
                retrieve_next_sibling,
                parent_pos,
            )

    dfs(0, retrieve_next_token, retrieve_next_sibling, -1)
    if aborted:
        # A partial mask is worse than no mask: allow every token for this
        # request and let the caller propose a degrade.
        allocate_token_bitmask.fill_(-1)
        return False
    return True


def agree_on_grammar_degrade(proposals: list[bool]) -> list[bool]:
    """OR per-request degrade proposals across the attention-TP group.

    Mask budgets are wall-clock based, so ranks of one attention-TP group can
    disagree on which request overran. They must still apply the same masks
    and accept the same tokens, otherwise spec verify diverges (different
    accept lengths / KV) or the request is aborted on only some ranks. Every
    rank of the group must call this with the same request list, from the
    thread that runs the forward pass.
    """
    from sglang.srt.layers.attention.dp_attention import (
        get_attention_tp_grammar_cpu_group,
    )

    group = get_attention_tp_grammar_cpu_group()
    if group is None or not proposals:
        return proposals
    flags = torch.tensor(proposals, dtype=torch.int32)
    torch.distributed.all_reduce(flags, op=torch.distributed.ReduceOp.MAX, group=group)
    return [bool(flag) for flag in flags.tolist()]


def degrade_grammars_by_consensus(grammars: list, proposals: list[bool], reason: str) -> list[bool]:
    """Agree on ``proposals`` (see ``agree_on_grammar_degrade``) and degrade
    the agreed grammars on this rank. Returns the agreed flags."""
    agreed = agree_on_grammar_degrade(proposals)
    for grammar, degrade in zip(grammars, agreed):
        if degrade and grammar is not None:
            degrade_fn = getattr(grammar, "degrade", None)
            if degrade_fn is not None:
                degrade_fn(reason)
    return agreed
