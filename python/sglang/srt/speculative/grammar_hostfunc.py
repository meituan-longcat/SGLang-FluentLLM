# Copyright 2024 SGLang Team
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
"""Grammar ops as CUDA-graph host nodes (overlapped, launch-ahead-safe).

The two CPU-side grammar operations of a MTP+grammar verify step --

  1. the DFS that turns the draft-token chain into a vocab bitmask
     (``BaseGrammarObject.fill_draft_tree_bitmask`` -- xgrammar's native
     ``GrammarMatcher.traverse_draft_tree``, or the Python ``traverse_tree``
     fallback), and
  2. the permanent accept that advances the live xgrammar matcher past the
     accepted tokens and snapshots ``is_terminated()`` per accepted token,

-- normally run *outside* the CUDA graph (DFS in ``preprocess_for_verify``
before ``replay()``; accept on the forward thread after ``replay()``), forcing
a CPU<->GPU sync around every replay.

This module hosts both as ``cudaLaunchHostFunc`` callbacks so they can be
*captured into* and *replayed by* the verify CUDA graph. The runner
(``SpecDecodeCudaGraphRunner.capture_one_batch_size``) forks them onto a
dedicated grammar stream via wait_stream fork/join, so the DFS overlaps the
target forward (``DFS||target_forward``) and the accept overlaps the draft steps
(``accept||draft``); sharing one grammar stream keeps DFS -> accept serialized on
the matcher.

A host callback runs on a CUDA-internal thread while holding the GIL and MUST
NOT call any CUDA API (a ``cudaLaunchHostFunc`` constraint). Its arguments are
bound at capture time and replayed verbatim, so everything mutable per-step must
live inside a single stable-identity container (``GrammarGraphCtx``) whose fields
are updated in place before each ``replay()``.

Under the overlap scheduler the forward thread launches replay(N+1) before the
consumer has read replay(N)'s grammar results (launch-ahead). So per-step state
that crosses the graph boundary is ring-buffered by ``slot``, and each host node
reads its own replay's slot from a device scalar copied D2H inside the graph
(``slot_host``) rather than closing over a capture-time Python int. See the
``GrammarGraphCtx.__init__`` comments for the per-buffer ring-vs-single rationale.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Host-func timeline capture
#
# The host callbacks run on a CUDA driver thread (see hostfunc/_binding.cpp),
# which torch.profiler's thread-local RecordFunction / NVTX hooks never reach --
# so record_function / nvtx annotations inside them are silently dropped and the
# thread gets no row on the Kineto trace. Instead we wall-clock each call with
# time.time_ns() (the same epoch clock Kineto's baseTimeNanoseconds uses) and,
# after the trace is exported, splice the spans in as synthetic events aligned
# to the exact same axis as every GPU kernel. That makes both questions legible:
#   * gap between the preceding kernel and the span start == dispatch/GIL delay
#   * span length == the callback's own execution time
# ---------------------------------------------------------------------------
_span_capture_enabled = False
# (label, start_ns, end_ns); appended under the GIL held by the running host
# callback, so no extra lock is needed.
_span_buffer: list[tuple[str, int, int]] = []


def enable_span_capture() -> None:
    """Start recording host-func spans and drop any spans from a prior run."""
    global _span_capture_enabled
    _span_buffer.clear()
    _span_capture_enabled = True


def disable_span_capture() -> None:
    global _span_capture_enabled
    _span_capture_enabled = False


def drain_spans() -> list[tuple[str, int, int]]:
    """Return the captured spans and clear the buffer."""
    spans = list(_span_buffer)
    _span_buffer.clear()
    return spans


# Base tid for the synthetic host-func lanes; picked far above any real thread
# id so it never collides with a scheduler / stream row.
_HOSTFUNC_LANE_BASE = 8_000_000


def inject_spans_into_trace(trace_path: str, spans: list[tuple[str, int, int]]) -> None:
    """Splice captured host-func spans into an exported Kineto chrome trace.

    Each span becomes an ``X`` (complete) event on a per-label lane, placed on
    the trace's own time axis via ``abs_ns = baseTimeNanoseconds + ts_us*1000``
    (the same epoch clock ``time.time_ns()`` reads). Runs in the same process as
    the callbacks, so ``os.getpid()`` matches the pid Kineto recorded.
    """
    if not spans:
        return

    import gzip
    import json
    import os

    open_fn = gzip.open if str(trace_path).endswith(".gz") else open
    with open_fn(trace_path, "rt") as f:
        data = json.load(f)

    base_ns = data.get("baseTimeNanoseconds", 0)
    events = data["traceEvents"]
    pid = os.getpid()

    lane_tids: dict = {}

    def lane_for(label: str) -> int:
        if label not in lane_tids:
            tid = _HOSTFUNC_LANE_BASE + len(lane_tids)
            lane_tids[label] = tid
            events.append(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": pid,
                    "tid": tid,
                    "args": {"name": f"hostfunc {label}"},
                }
            )
        return lane_tids[label]

    for label, t0, t1 in spans:
        events.append(
            {
                "name": label,
                "cat": "user_annotation",
                "ph": "X",
                "pid": pid,
                "tid": lane_for(label),
                "ts": (t0 - base_ns) / 1000.0,
                "dur": max(t1 - t0, 0) / 1000.0,
                "args": {"start_ns": t0, "end_ns": t1, "dur_us": (t1 - t0) / 1000.0},
            }
        )

    with open_fn(trace_path, "wt") as f:
        json.dump(data, f)

    logger.info(
        "[grammar hostfunc] injected %d host-func span(s) into %s",
        len(spans),
        trace_path,
    )


# Bit pattern meaning "all tokens allowed" in the xgrammar bitmask convention
# (a set bit == allowed). Matches ``XGrammarBackend.reset_vocab_masks`` which
# fills the device buffer with -1 for the no-constraint / padding case.
_ALLOW_ALL = -1


class GrammarGraphCtx:
    """Stable container shared by the DFS / accept host callbacks and replay().

    Field *identity* is fixed for the lifetime of the runner (so the host nodes
    captured into every per-bs graph keep referring to the same buffers); field
    *contents* are overwritten in place by ``replay()`` (the per-step inputs:
    ``grammars`` / ``raw_bs``) and by the captured graph (the device->host and
    host->device copies around the callbacks).

    Buffers are sized for ``max_bs * draft_token_num`` (the largest captured
    graph); a given replay only touches the ``raw_bs`` / ``raw_num_token``
    prefix. ``draft_token_num`` doubles as the MTP chain length
    (``speculative_num_steps + 1``): MTP draft tokens form a single linear path
    root -> t1 -> ... -> t(L-1) with no siblings.
    """

    def __init__(
        self,
        *,
        max_bs: int,
        draft_token_num: int,
        vocab_masks_device: torch.Tensor,
        ring_depth: int = 1,
    ):
        self.max_bs = max_bs
        self.draft_token_num = draft_token_num
        self.max_num_token = max_bs * draft_token_num
        self.ring_depth = ring_depth

        # --- Per-step state that the host nodes read/write, ring-buffered ---
        #
        # Under launch-ahead the forward thread launches replay(N+1) before the
        # consumer has read replay(N)'s grammar results, and before graph N's
        # host nodes have necessarily executed. A single shared slot would let
        # replay(N+1)'s ``set_step_inputs`` overwrite the grammars list / raw_bs
        # that graph N's host node still needs, and clobber graph N's termination
        # snapshot before the consumer reads it. So every piece of per-step state
        # that crosses the graph boundary lives in a ring indexed by ``slot``.
        #
        # The slot a given graph uses is chosen at *capture* time and replayed
        # verbatim, so it cannot be a Python int the host node closes over.
        # Instead replay() stages the slot into a device scalar, the captured
        # graph copies it D2H into ``slot_host``, and each host node reads
        # ``int(slot_host[0])`` at execution time -- picking the slot of the
        # replay it actually belongs to.
        self.raw_bs_ring: list[int] = [max_bs] * ring_depth
        # Length (in flat mask elements) that the DFS node must reset to allow-all
        # before writing per-grammar constraints. It is exactly the range the graph
        # / eager H2D copy reads back (capture_n = padded-bs * L for graph, raw_bs *
        # L for eager), NOT the whole buffer: max_bs is sized for the worst-case
        # decode bs per rank, but a given step only uses/H2D's its own capture_n, so
        # filling the whole buffer just wastes host time on rows that are never read.
        self.mask_fill_len_ring: list[int] = [self.max_num_token] * ring_depth
        self.grammars_ring: list[list[BaseGrammarObject | None]] = [[None] * max_bs for _ in range(ring_depth)]
        # Stream-staged slot index: device scalar -> captured D2H -> here.
        self.slot_host = torch.zeros((1,), dtype=torch.int32, pin_memory=True)

        # DFS input: draft tokens copied D2H *inside* the graph (captured copy),
        # so the host node -- ordered after it on the stream -- reads the draft
        # tokens of the current replay. Stream-ordered, so a single buffer is
        # safe (no ring needed). Pinned for async D2H. Kept int32 to match the
        # int32 device draft-token buffer so the captured D2H copy is a clean
        # same-dtype async copy; the host node casts the used prefix to the int64
        # that traverse_draft_tree requires (see _do_dfs).
        self.draft_token_host = torch.empty((self.max_num_token,), dtype=torch.int32, pin_memory=True)

        # DFS output: written on the host, then copied H2D into the device vocab
        # mask *inside* the graph (captured copy ordered after the DFS node).
        # Stream-ordered (host node -> H2D copy on the same stream), so a single
        # buffer is safe. Same shape/dtype as the device buffer it feeds. Pinned.
        self.vocab_masks_device = vocab_masks_device
        self.mask_host = torch.empty(
            tuple(vocab_masks_device.shape),
            dtype=vocab_masks_device.dtype,
            pin_memory=True,
        )

        # Accept inputs: copied D2H inside the graph, read by the accept node.
        # Stream-ordered (D2H copy -> accept node on the same stream), single
        # buffer safe. ``accepted_host`` is the gathered, per-request *compacted*
        # accepted token ids (output_ids = target_predict[accept_index]); the
        # accept walk reads accept_length[i] of them contiguously per request.
        self.accepted_host = torch.empty((self.max_num_token,), dtype=torch.int32, pin_memory=True)
        self.accept_length_host = torch.empty((self.max_bs,), dtype=torch.int32, pin_memory=True)

        # Accept outputs: written on the host by the accept node, read back on
        # the consumer thread after that replay's ``copy_done`` is synchronized.
        # These cross the graph boundary to a later consumer, so they ARE
        # ring-buffered. Pure CPU staging -- no graph copies touch these, so they
        # need not be pinned.
        # ``terminated_ring[slot]`` is flat and indexed the same way as the
        # accept walk (cumulative ``pt`` over accept lengths); collect_terminated
        # reshapes it into the per-request list-of-lists the post-processor
        # expects.
        self.terminated_ring = torch.zeros((ring_depth, self.max_num_token), dtype=torch.bool)
        # Per-request accept failure flag (xgrammar accept raised). The C ABI of
        # cudaLaunchHostFunc cannot propagate exceptions, so the callback records
        # the failure here and the consumer turns it into an abort.
        self.error_ring = torch.zeros((ring_depth, max_bs), dtype=torch.int32)

        # Constant retrieve arrays encoding the linear MTP chain for traverse_tree:
        # next_token[:, :-1] = arange(1, L) (last position has no child),
        # sibling is all -1 (a chain never branches). Built once.
        L = draft_token_num
        self.retrieve_next_token = torch.full((max_bs, L), -1, dtype=torch.long)
        if L > 1:
            self.retrieve_next_token[:, :-1] = torch.arange(1, L, dtype=torch.long)
        self.retrieve_next_sibling = torch.full((max_bs, L), -1, dtype=torch.long)

    def set_step_inputs(
        self,
        grammars: list[BaseGrammarObject | None],
        raw_bs: int,
        mask_fill_len: int,
        slot: int,
    ) -> None:
        """Overwrite the per-step inputs for ``slot`` in place (called before
        each replay, with ``slot`` cycling through the ring). ``mask_fill_len`` is
        the flat mask range the DFS node resets to allow-all -- the same range the
        H2D copy reads back this step (capture_n)."""
        self.raw_bs_ring[slot] = raw_bs
        self.mask_fill_len_ring[slot] = mask_fill_len
        # Refresh the grammars list in place; pad the tail with None so stale
        # references from a larger previous batch reusing this slot are not used.
        ring = self.grammars_ring[slot]
        n = len(grammars)
        for i in range(self.max_bs):
            ring[i] = grammars[i] if i < n else None
        self.error_ring[slot].zero_()

    def collect_terminated(self, slot: int, accept_lens: list[int]) -> list[list[bool]]:
        """Rebuild the per-request termination list-of-lists from the flat
        ``terminated_ring[slot]`` snapshot using ``accept_lens`` (mirrors the
        cumulative-``pt`` layout the accept callback wrote). ``accept_lens`` is
        passed explicitly by the consumer (from its own D2H'd accept_lengths)
        rather than read back from ``accept_length_host``, which is a single
        buffer the next replay may already have overwritten. Call only after the
        replay's ``copy_done`` has been synchronized."""
        terminated = self.terminated_ring[slot].tolist()
        out: list[list[bool]] = []
        pt = 0
        for n in accept_lens:
            out.append(terminated[pt : pt + n])
            pt += n
        return out


def _do_dfs(ctx: GrammarGraphCtx, slot: int) -> None:
    """Pure DFS core: build the vocab bitmask for ``slot`` into ``ctx.mask_host``.

    Mirrors ``generate_token_bitmask`` but writes into the persistent pinned
    buffer instead of allocating. Baseline is allow-all (-1) so requests with no
    grammar and padding slots impose no constraint; ``fill_draft_tree_bitmask``
    resets each grammar's own slice to disallow-all (0) and then sets the allowed
    bits.

    Shared by the graph wrapper (``dfs_hostfunc``, slot from ``slot_host``) and
    the eager wrapper (``dfs_hostfunc_eager``, slot bound as a closure arg).
    """
    raw_bs = ctx.raw_bs_ring[slot]
    grammars = ctx.grammars_ring[slot]

    # Allow-all baseline. Only reset the range that this step will H2D back
    # (capture_n): None-grammar requests and graph padding slots below capture_n
    # stay unconstrained, while rows beyond capture_n are never read, so filling
    # them would only waste host time (the buffer is sized for the worst-case bs).
    ctx.mask_host[: ctx.mask_fill_len_ring[slot]].fill_(_ALLOW_ALL)

    L = ctx.draft_token_num
    # traverse_draft_tree requires int64 draft tokens; the pinned host buffer is
    # int32 (dtype-matched to the device buffer for a clean captured D2H copy).
    # Cast the used prefix once per call (O(raw_bs*L), negligible vs the DFS) so
    # the per-request slices below are int64 without a per-request alloc.
    draft = ctx.draft_token_host[: raw_bs * L].to(torch.int64)
    for i in range(raw_bs):
        grammar = grammars[i]
        if grammar is None or grammar.finished or grammar.is_terminated():
            continue
        grammar.fill_draft_tree_bitmask(
            ctx.retrieve_next_token[i],
            ctx.retrieve_next_sibling[i],
            draft[i * L : (i + 1) * L],
            ctx.mask_host[i * L : (i + 1) * L],
        )


def _do_accept(ctx: GrammarGraphCtx, slot: int) -> None:
    """Pure accept core: permanently advance each live matcher past its accepted
    tokens and snapshot ``is_terminated()`` per accepted token into
    ``ctx.terminated_ring[slot]``.

    Byte-for-byte mirror of the out-of-graph forward-thread accept walk:
    cumulative ``pt`` over the accept lengths, accept only while the grammar is
    live, snapshot termination for every accepted slot. Exceptions are recorded
    in ``ctx.error_ring[slot]`` rather than raised (the host-callback C ABI
    cannot propagate them).

    Shared by the graph wrapper (``accept_hostfunc``, slot from ``slot_host``)
    and the eager wrapper (``accept_hostfunc_eager``, slot bound as a closure
    arg).
    """
    raw_bs = ctx.raw_bs_ring[slot]
    grammars = ctx.grammars_ring[slot]
    raw_num_token = raw_bs * ctx.draft_token_num
    token_ids = ctx.accepted_host[:raw_num_token].tolist()
    accept_lens = ctx.accept_length_host[:raw_bs].tolist()
    terminated = ctx.terminated_ring[slot]
    error = ctx.error_ring[slot]

    pt = 0
    for i in range(raw_bs):
        grammar = grammars[i]
        n = accept_lens[i]
        try:
            for j in range(n):
                if grammar is not None and not grammar.finished and not grammar.is_terminated():
                    grammar.accept_token(token_ids[pt + j])
                terminated[pt + j] = grammar.is_terminated() if grammar is not None else False
        except Exception as e:
            # Mark this request for abort; do not corrupt the rest of the
            # batch. The C ABI of cudaLaunchHostFunc cannot propagate
            # exceptions, so the consumer turns ``error_ring[slot]`` into an
            # abort.
            error[i] = 1
            logger.error(
                "[grammar accept] accept failed slot=%d req=%d raw_bs=%d accept_len=%d pt=%d token_ids=%s: %s",
                slot,
                i,
                raw_bs,
                n,
                pt,
                token_ids[pt : pt + n],
                e,
            )
        pt += n


def dfs_hostfunc(ctx: GrammarGraphCtx) -> None:
    """Graph host node wrapping ``_do_dfs``.

    The slot (which replay this node belongs to) is read from ``slot_host``,
    written by the captured D2H copy ordered before this node on the grammar
    stream -- so under launch-ahead each captured DFS node picks the grammars /
    raw_bs of its own replay, not whatever the latest ``set_step_inputs`` wrote.
    """
    # Wall-clock this callback so it can be spliced onto the profiler timeline;
    # see the module header for why in-callback record_function / NVTX can't.
    # start_ns is taken post-GIL (first line here), so the span length is the
    # callback's own execution time and any delay in *starting* it shows up as
    # the gap before the span.
    _t0 = time.time_ns()
    try:
        _do_dfs(ctx, int(ctx.slot_host[0]))
    finally:
        if _span_capture_enabled:
            _span_buffer.append(("grammar_dfs", _t0, time.time_ns()))


def accept_hostfunc(ctx: GrammarGraphCtx) -> None:
    """Graph host node wrapping ``_do_accept``.

    Slot is read from ``slot_host`` (see ``dfs_hostfunc``) so the snapshot lands
    in the ring slot the consumer will read for this replay.
    """
    # Wall-clock this callback for the profiler timeline; see dfs_hostfunc.
    _t0 = time.time_ns()
    try:
        _do_accept(ctx, int(ctx.slot_host[0]))
    finally:
        if _span_capture_enabled:
            _span_buffer.append(("grammar_accept", _t0, time.time_ns()))


def dfs_hostfunc_eager(ctx: GrammarGraphCtx, slot: int) -> None:
    """Eager host callback wrapping ``_do_dfs`` with ``slot`` bound at launch.

    Runs on the grammar stream in the eager (non-captured) verify path. Because
    ``launch_hostfunc`` is not capturing, the slot is passed directly as a bound
    argument -- no ``slot_host`` device-scalar read-back needed.
    """
    _t0 = time.time_ns()
    try:
        _do_dfs(ctx, slot)
    finally:
        if _span_capture_enabled:
            _span_buffer.append(("grammar_dfs_eager", _t0, time.time_ns()))


def accept_hostfunc_eager(ctx: GrammarGraphCtx, slot: int) -> None:
    """Eager host callback wrapping ``_do_accept`` with ``slot`` bound at launch.

    See ``dfs_hostfunc_eager`` -- the slot is a bound closure arg, not read from
    ``slot_host``.
    """
    _t0 = time.time_ns()
    try:
        _do_accept(ctx, slot)
    finally:
        if _span_capture_enabled:
            _span_buffer.append(("grammar_accept_eager", _t0, time.time_ns()))
