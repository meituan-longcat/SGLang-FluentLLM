"""EPD Prefill-side encode receive + admission control.

This module implements the Prefill Scheduler side of EPD:
1. ZMQ PULL-based MMResult receiving (phase 1, rvq_indices mode).
2. Rank-synced admission: all-reduce(MIN) ensures consumer-group-consistent
   readiness (Prefill PP0 attention-TP; Decode coordinator size 1).
3. Integration with the existing Scheduler event loop.

Phase 1 design (ZMQ transport, rvq_indices output):
- Each Prefill PP0 attention-TP rank runs an EpdPrefillReceiver with a ZMQ
  PULL socket; later PP stages consume canonical token descriptors instead.
- Encoder pushes MMResult (pickled) to each rank's PULL socket.
- The Scheduler event loop calls ``poll_admissions()`` each cycle to:
  a. Poll for new results (non-blocking).
  b. Check waiting requests for completion/timeout.
  c. All-reduce(MIN) for TP-consistent admission.
  d. Return (admitted, failed) request lists.

Phase 2 (Mooncake RDMA transport, vit_embedding output):
- Will use tokenspeed's MooncakeEmbeddingReceiver + row-sharding.
- Placeholder hooks are included for future integration.

Key alignment with TokenSpeed:
- ``EpdPrefillReceiver`` ≈ ``EpdPrefillAdmission`` (simplified for ZMQ).
- ``WaitingMMRequest`` (defined below) handles per-request state.
- This module adds the Scheduler integration layer.
"""

from __future__ import annotations

import logging
import time
from enum import IntEnum
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.disaggregation.encode.encode_data import (
    MMItemMeta,
    MMMeta,
    MMResult,
    MediaURL,
)

if TYPE_CHECKING:
    import torch.distributed as dist

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Prefill-side waiting state
# ═══════════════════════════════════════════════════════════════════════════


class WaitingMMRequestStatus(IntEnum):
    """State machine for waiting MM data.

    Encoding aligned with tokenspeed's ``{FAILED: 0, PENDING: 1, DONE: 2}``
    so that ``all_reduce(MIN)`` has correct semantics: any rank PENDING(1)
    → global PENDING(1); any rank FAILED(0) → global FAILED(0); only when
    all ranks are SUCCESS(2) does MIN stay 2.
    """

    FAILED = 0    # Encoder error or timeout
    PENDING = 1   # MM data not yet received
    SUCCESS = 2   # All TP ranks received, mm_inputs constructed


class WaitingMMRequest:
    """A request on the Prefill Scheduler side waiting for Encoder to push MM data.

    Each TP rank has its own EpdPrefillReceiver instance, independently receiving.

    Post-refactor (MMMeta rides on MMResult): the request no longer carries
    ``mm_meta`` at enqueue time — meta arrives alongside the ViT output in
    the combined ``MMResult`` (``item_index == -1``).  The request instead
    carries ``media_urls`` (extracted from the input text by the
    TokenizerManager) so that, after admission, the Scheduler can run
    tokenization using the received meta.

    Attributes:
        request_id: request identifier.
        media_urls: ``list[MediaURL]`` extracted from input text (for
            tokenization after admission, when mm_meta is available).
        status: current state.
        start_time: when the request started waiting.
        mm_meta: ``MMMeta`` extracted from the received combined MMResult
            (None until the combined result arrives).
        mm_results: received MMResult dict (item_index → MMResult).
    """

    def __init__(
        self,
        request_id: str,
        media_urls: list[MediaURL],
        timeout: float = 300.0,
        compute_to_logical: Optional[list[int]] = None,
    ):
        self.request_id = request_id
        self.media_urls = media_urls
        self.mm_meta: Optional[MMMeta] = None
        self.status = WaitingMMRequestStatus.PENDING
        self.start_time = time.monotonic()
        self.result_ready_ns: Optional[int] = None
        self.timeout = timeout
        self.compute_to_logical = list(compute_to_logical or [])
        # Results are normalized to one entry per global media index.  A
        # combined shard (item_index=-1) is split on receipt using its MMMeta.
        self.mm_results: dict[int, MMResult] = {}
        self._item_metas: dict[int, MMItemMeta] = {}
        expected_count = len(self.compute_to_logical) or len(media_urls)
        self._expected_item_indices = set(range(expected_count))
        self._vit_pending = False

    def _fail(self, reason: str) -> None:
        logger.error("EPD receive failed: rid=%s, reason=%s", self.request_id, reason)
        self.status = WaitingMMRequestStatus.FAILED

    def _merge_meta(self, mm_meta: MMMeta) -> bool:
        """Merge one Encoder shard's metadata by global media index."""
        if mm_meta.request_id != self.request_id:
            self._fail(
                f"metadata request_id mismatch: {mm_meta.request_id!r}"
            )
            return False

        shard_indices = [meta.item_index for meta in mm_meta.item_metas]
        if len(shard_indices) != len(set(shard_indices)):
            self._fail(f"duplicate media indices in shard: {shard_indices}")
            return False
        unexpected = set(shard_indices) - self._expected_item_indices
        if unexpected:
            self._fail(f"unexpected media indices: {sorted(unexpected)}")
            return False

        for meta in mm_meta.item_metas:
            existing = self._item_metas.get(meta.item_index)
            if existing is not None and existing != meta:
                self._fail(
                    f"conflicting metadata for media index: {meta.item_index}"
                )
                return False
            self._item_metas[meta.item_index] = meta
        self._vit_pending = self._vit_pending or mm_meta.vit_pending
        self.mm_meta = MMMeta(
            request_id=self.request_id,
            item_metas=[self._item_metas[i] for i in sorted(self._item_metas)],
            vit_pending=self._vit_pending,
        )
        return True

    def _maybe_mark_success(self, require_data: bool) -> None:
        if self.status == WaitingMMRequestStatus.FAILED:
            return
        meta_complete = set(self._item_metas) == self._expected_item_indices
        data_complete = set(self.mm_results) == self._expected_item_indices
        if (
            self.status != WaitingMMRequestStatus.SUCCESS
            and meta_complete
            and (data_complete or not require_data)
        ):
            self.status = WaitingMMRequestStatus.SUCCESS
            self.result_ready_ns = time.monotonic_ns()

    def on_result_received(self, result: MMResult) -> None:
        """Called when ZMQ PULL receives an MMResult (full, with ViT tensor).
        """
        if result.request_id != self.request_id:
            self._fail(f"result request_id mismatch: {result.request_id!r}")
            return
        if result.error:
            self._fail(result.error)
            return

        if result.item_index == -1:
            if result.mm_meta is None or result.data is None:
                self._fail("combined result is missing metadata or tensor data")
                return
            if not self._merge_meta(result.mm_meta):
                return

            token_counts = [meta.num_tokens for meta in result.mm_meta.item_metas]
            if sum(token_counts) != result.data.shape[0]:
                self._fail(
                    "combined tensor length does not match shard metadata: "
                    f"tensor={result.data.shape[0]}, meta={sum(token_counts)}"
                )
                return
            tensors = torch.split(result.data, token_counts, dim=0)
            for meta, tensor in zip(result.mm_meta.item_metas, tensors):
                self.mm_results[meta.item_index] = MMResult(
                    request_id=self.request_id,
                    item_index=meta.item_index,
                    rvq_indices=(
                        tensor.contiguous()
                        if result.rvq_indices is not None else None
                    ),
                    vit_embedding=(
                        tensor.contiguous()
                        if result.vit_embedding is not None else None
                    ),
                    num_tokens=tensor.shape[0],
                )
        else:
            if result.item_index not in self._expected_item_indices:
                self._fail(f"unexpected result media index: {result.item_index}")
                return
            if result.data is None:
                self._fail(f"media result is missing tensor data: {result.item_index}")
                return
            self.mm_results[result.item_index] = result
            if result.mm_meta is not None and not self._merge_meta(result.mm_meta):
                return

        self._maybe_mark_success(require_data=True)

    def on_meta_received(self, mm_meta: MMMeta, mark_success: bool = False) -> None:
        """Called when ZMQ PULL receives a lightweight MMMeta payload.
        Args:
            mm_meta: received meta payload.
            mark_success: if True, mark this request SUCCESS (meta-only mode,
                used by Decode nodes).  If False (Prefill nodes), the request
                still waits for the full MMResult carrying the ViT tensor.
        """
        if not self._merge_meta(mm_meta):
            return
        self._maybe_mark_success(require_data=not mark_success)

    def check_timeout(self) -> bool:
        """Check if the request has timed out."""
        if time.monotonic() - self.start_time > self.timeout:
            self.status = WaitingMMRequestStatus.FAILED
            return True
        return False

    @property
    def is_ready(self) -> bool:
        return self.status == WaitingMMRequestStatus.SUCCESS

    @property
    def is_failed(self) -> bool:
        return self.status == WaitingMMRequestStatus.FAILED

    def folded_result(self) -> tuple[MMMeta, dict[int, MMResult]]:
        """Fold temporal-group results back into prompt-level media items."""

        if self.mm_meta is None:
            raise RuntimeError("EPD request has no metadata")
        if not self.compute_to_logical:
            return self.mm_meta, self.mm_results
        from sglang.srt.disaggregation.encode.maas_frame_epd_adapter import (
            fold_maas_frame_results,
        )

        return fold_maas_frame_results(
            self.request_id,
            self.media_urls,
            self.compute_to_logical,
            self.mm_meta,
            self.mm_results,
        )


class EpdPrefillReceiver:
    """EPD Prefill-side receive + admission controller.

    Owns:
    - ZMQ PULL socket(s) to receive MMResult from Encoder.
    - Waiting queue for requests pending MM data.
    - Rank-synced admission logic (all_reduce MIN for TP consistency).

    This is the FluentLLM counterpart of TokenSpeed's EpdPrefillAdmission,
    simplified for phase 1 (ZMQ transport, no row-sharding, no Mooncake).

    Lifecycle:
    1. ``__init__``: bind ZMQ PULL socket.
    2. ``enqueue(request_id, mm_meta, mm_items)``: register a new waiting request.
    3. ``poll_admissions()``: called each Scheduler cycle; returns (admitted, failed) lists.
    4. ``close()``: cleanup ZMQ.

    Attributes:
        tp_rank: this rank's TP index.
        tp_size: total TP size.
        zmq_pull_port: ZMQ PULL port bound by this rank.
        timeout_s: per-request timeout waiting for MM data.
        tensor_device: destination device for full MMResult tensors.  Prefill
            PP0 passes its rank-local device; Decode meta-only mode leaves it unset.
    """

    def __init__(
        self,
        tp_rank: int,
        tp_size: int,
        zmq_pull_port: int,
        timeout_s: float = 300.0,
        tp_cpu_group: Optional["dist.ProcessGroup"] = None,
        meta_only_mode: bool = False,
        tensor_device: Optional[torch.device | str] = None,
    ):
        import zmq

        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.timeout_s = timeout_s
        self.tp_cpu_group = tp_cpu_group

        self.meta_only_mode = meta_only_mode
        self.tensor_device = tensor_device

        # ZMQ PULL socket: receive MMResult pushed by Encoder
        self._context = zmq.Context(1)
        self._socket = self._context.socket(zmq.PULL)
        self._socket.bind(f"tcp://*:{zmq_pull_port}")
        # High-water mark: buffer up to 1024 messages before backpressure
        self._socket.setsockopt(zmq.RCVHWM, 1024)

        # Waiting requests: request_id → WaitingMMRequest
        self._waiting: dict[str, WaitingMMRequest] = {}
        self._waiting_order: list[str] = []

        # Stats
        self._total_received = 0
        self._total_admitted = 0
        self._total_failed = 0

    def _move_result_to_device(self, result: MMResult) -> None:
        """Move a full result's CPU transport tensor to this receiver's device.

        Encoder serializes CPU tensors so pickle cannot restore every Prefill
        process onto the Encoder rank's NPU.  Each PP0 TP rank performs its own
        explicit H2D here before setting local readiness.
        """
        data = result.data
        if data is None or self.tensor_device is None:
            return

        target_device = torch.device(self.tensor_device)
        if data.device == target_device:
            return

        source_device = data.device
        moved = data.to(device=target_device)
        if result.rvq_indices is not None:
            result.rvq_indices = moved
        else:
            result.vit_embedding = moved
        logger.debug(
            "Moved EPD MMResult tensor: rank=%d, rid=%s, %s -> %s, bytes=%d",
            self.tp_rank,
            result.request_id,
            source_device,
            target_device,
            result.nbytes,
        )

    def enqueue(
        self,
        request_id: str,
        media_urls: list[MediaURL],
        compute_to_logical: Optional[list[int]] = None,
    ) -> None:
        """Register a new request waiting for MM data from Encoder.

        Post-refactor: the request carries ``media_urls`` (not ``MMMeta``);
        ``MMMeta`` arrives later on the combined ``MMResult``.

        Args:
            request_id: unique request identifier.
            media_urls: ``list[MediaURL]`` extracted from the input text
                (used for tokenization after admission, once mm_meta is
                available).
        """
        waiting = WaitingMMRequest(
            request_id,
            media_urls,
            timeout=self.timeout_s,
            compute_to_logical=compute_to_logical,
        )
        self._waiting[request_id] = waiting
        self._waiting_order.append(request_id)
        logger.info(
            "Enqueued EPD request: rid=%s, urls=%d, timeout=%.0fs",
            request_id,
            len(media_urls),
            self.timeout_s,
        )

    def _poll_results(self) -> int:
        """Non-blocking ZMQ PULL poll; dispatch received MMResult to waiting requests.

        Returns:
            Number of results received in this poll.
        """
        import pickle
        import zmq

        count = 0
        while True:
            try:
                raw = self._socket.recv(flags=zmq.NOBLOCK)
                result = pickle.loads(raw)
                # Mass D-side lightweight path: payload is a bare MMMeta
                # (no ViT tensor).  Dispatch to on_meta_received.
                if isinstance(result, MMMeta):
                    logger.info(
                        "[_poll_results rank=%d] received MMMeta: rid=%s, bytes=%d, waiting=%d, meta_only_mode=%s",
                        self.tp_rank, result.request_id, len(raw),
                        len(self._waiting), self.meta_only_mode,
                    )
                    waiting = self._waiting.get(result.request_id)
                    if waiting is not None:
                        waiting.on_meta_received(result, mark_success=self.meta_only_mode)
                        count += 1
                        logger.info(
                            "[_poll_results rank=%d] dispatched meta to waiting: rid=%s, status=%s",
                            self.tp_rank, result.request_id, waiting.status.name,
                        )
                    else:
                        logger.error(
                            "Discarded MMMeta for unknown request: rid=%s",
                            result.request_id,
                        )
                    continue
                if not isinstance(result, MMResult):
                    logger.warning(
                        "Unexpected ZMQ payload type: %s (expected MMResult or MMMeta)",
                        type(result).__name__,
                    )
                    continue
                logger.info(
                    "[_poll_results rank=%d] received MMResult: rid=%s, item_index=%d, bytes=%d, waiting=%d",
                    self.tp_rank, result.request_id, result.item_index, len(raw),
                    len(self._waiting),
                )
                waiting = self._waiting.get(result.request_id)
                if waiting is not None:
                    self._move_result_to_device(result)
                    waiting.on_result_received(result)
                    count += 1
                    meta_count = (
                        len(waiting.mm_meta.item_metas)
                        if waiting.mm_meta is not None else 0
                    )
                    logger.info(
                        "[_poll_results rank=%d] dispatched to waiting: rid=%s, status=%s, results=%d/%d",
                        self.tp_rank, result.request_id, waiting.status.name,
                        len(waiting.mm_results), meta_count,
                    )
                else:
                    # Result for unknown request (already admitted/failed/evicted)
                    logger.error(
                        "Discarded MMResult for unknown request: rid=%s",
                        result.request_id,
                    )
            except zmq.Again:
                break
            except Exception:
                logger.exception("Error receiving MMResult from ZMQ")
                break

        self._total_received += count
        return count

    def poll_admissions(self) -> tuple[list[str], list[str]]:
        """Poll + check + all_reduce → return (admitted_ids, failed_ids).

        Called by the Scheduler at the top of each event loop cycle.  This is
        the main entry point, aligned with sglang's ``_process_waiting_requests``
        and tokenspeed's ``EpdPrefillAdmission.drain``:

        1. Poll ZMQ for new results (non-blocking).
        2. If ``_waiting_order`` is empty, return early (rank-identical
           emptiness → safe to skip the collective, like tokenspeed's
           ``if not self._pending: return [], []``).
        3. Classify each waiting request as SUCCESS / PENDING / FAILED.
           Timeout is folded into the same all_reduce (aligned with tokenspeed):
           a locally-timed-out PENDING request is reclassified as FAILED(0)
           before the collective, so MIN propagates the failure to all ranks.
        4. ``all_reduce(MIN)`` on the per-request status **array**.  This is
           safe because ``_waiting_order`` is rank-identical (guaranteed by
           ``recv_requests``'s ``broadcast_pyobj``), so every rank's tensor has
           the same length.  MIN semantics with the FAILED=0/PENDING=1/SUCCESS=2
           encoding: any rank PENDING → global PENDING; any rank FAILED →
           global FAILED; only all-ranks SUCCESS stays SUCCESS.
        5. Admit SUCCESS, abort FAILED, keep PENDING for next cycle.

        Every frame with a non-empty ``_waiting_order`` performs the
        ``all_reduce`` — this is the same behavior as sglang and tokenspeed.
        The gloo CPU-group collective on a small int32 tensor is sub-ms and
        negligible compared to ViT computation (hundreds of ms).

        Returns:
            (admitted_ids, failed_ids): lists of request_id strings.
        """
        # 1. Poll for new results
        self._poll_results()

        # 2. Empty-list early return (rank-identical emptiness → safe skip).
        if not self._waiting_order:
            return [], []

        # 3. Compute local status per request.  Timeout is folded in here
        #    (aligned with tokenspeed): a PENDING request past the deadline is
        #    reclassified as FAILED(0) before the collective, so any rank's
        #    timeout propagates to all ranks via MIN.
        local_codes: list[int] = []
        for rid in self._waiting_order:
            waiting = self._waiting[rid]
            if waiting.is_ready:
                local_codes.append(int(WaitingMMRequestStatus.SUCCESS))
            elif waiting.is_failed or waiting.check_timeout():
                local_codes.append(int(WaitingMMRequestStatus.FAILED))
            else:
                local_codes.append(int(WaitingMMRequestStatus.PENDING))

        # 4. Admission-group sync: all_reduce(MIN) on the terminal status array.
        #    _waiting_order is rank-identical (recv_requests broadcast), so the
        #    tensor length matches across ranks.  MIN with FAILED=0/PENDING=1/
        #    SUCCESS=2 gives correct semantics: any rank not SUCCESS → not
        #    admitted; any rank FAILED → all abort.
        synced_codes = self._rank_sync_codes(local_codes)

        # 5. Classify based on the synced codes.
        admitted: list[str] = []
        failed: list[str] = []
        remaining_order: list[str] = []

        for rid, code in zip(self._waiting_order, synced_codes):
            if code == int(WaitingMMRequestStatus.SUCCESS):
                admitted.append(rid)
                # NOTE: do NOT pop from _waiting here — get_received_result()
                # needs the WaitingMMRequest to return mm_results after admit.
                # Cleanup happens in get_received_result() after the caller
                # has fetched the data.
            elif code == int(WaitingMMRequestStatus.FAILED):
                failed.append(rid)
                self._waiting.pop(rid, None)
            else:
                remaining_order.append(rid)

        self._waiting_order = remaining_order
        self._total_admitted += len(admitted)
        self._total_failed += len(failed)

        if admitted or failed:
            logger.info(
                "EPD drain: admitted=%d, failed=%d, pending=%d",
                len(admitted),
                len(failed),
                len(remaining_order),
            )

        return admitted, failed

    def _rank_sync_codes(self, codes: list[int]) -> list[int]:
        """All-reduce MIN across TP ranks for consistent admission decisions.

        Ensures that if ANY rank has not received MM data for a request,
        NO rank admits it (prevents inconsistent batching across TP).

        The ``codes`` list has the same length on every rank because
        ``_waiting_order`` is rank-identical — ``recv_requests`` broadcasts
        the request list via ``broadcast_pyobj``, so every TP rank sees the
        same EPD requests in the same order.

        Args:
            codes: per-request status codes (local to this rank).

        Returns:
            Synced codes (identical across all TP ranks).
        """
        if self.tp_size <= 1 or self.tp_cpu_group is None:
            return codes

        import torch.distributed as dist

        tensor = torch.tensor(codes, dtype=torch.int32, device="cpu")
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=self.tp_cpu_group)
        return tensor.tolist()

    def get_result_ready_ns(self, request_id: str) -> Optional[int]:
        """Return when this rank first completed the request's MM payload."""
        waiting = self._waiting.get(request_id)
        return waiting.result_ready_ns if waiting is not None else None

    def get_received_result(self, request_id: str) -> Optional[tuple[MMMeta, dict[int, MMResult]]]:
        """Get the (mm_meta, mm_results) for an admitted request.

        Post-refactor: ``mm_meta`` is extracted from the combined MMResult
        (it rode on the ZMQ payload).  Returns ``(mm_meta, mm_results)`` so
        the caller can run tokenization (using meta) and build mm_inputs
        (using the ViT tensor).

        Args:
            request_id: admitted request identifier.

        Returns:
            ``(mm_meta, mm_results)`` tuple, or None if not found.
        """
        waiting = self._waiting.pop(request_id, None)
        if waiting is not None:
            return waiting.folded_result()
        return None

    def cancel(self, request_id: str) -> None:
        """Cancel a waiting request (e.g., user abort).

        Args:
            request_id: request to cancel.
        """
        if request_id in self._waiting:
            del self._waiting[request_id]
            self._waiting_order = [
                rid for rid in self._waiting_order if rid != request_id
            ]

    def has_pending(self) -> bool:
        """Check if any requests are still waiting for MM data."""
        return len(self._waiting_order) > 0

    @property
    def num_pending(self) -> int:
        """Number of requests currently waiting for MM data."""
        return len(self._waiting_order)

    @property
    def stats(self) -> dict:
        """Receiver statistics."""
        return {
            "pending": self.num_pending,
            "total_received": self._total_received,
            "total_admitted": self._total_admitted,
            "total_failed": self._total_failed,
        }

    def close(self) -> None:
        """Cleanup ZMQ resources."""
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        if self._context is not None:
            self._context.term()
            self._context = None

    def __repr__(self) -> str:
        return (
            f"EpdPrefillReceiver(tp_rank={self.tp_rank}, tp_size={self.tp_size}, "
            f"pending={self.num_pending}, admitted={self._total_admitted}, "
            f"failed={self._total_failed})"
        )


def _is_decode_mode(server_args) -> bool:
    """True if this node is a Decode worker in PD-disaggregated mode."""
    from sglang.srt.disaggregation.utils import DisaggregationMode
    return server_args.disaggregation_mode == DisaggregationMode.DECODE.value


def build_epd_prefill_receiver(
    server_args,
    tp_rank: int,
    tp_size: int,
    zmq_pull_port: int,
    tp_cpu_group=None,
    tensor_device: Optional[torch.device | str] = None,
) -> Optional[EpdPrefillReceiver]:
    """Build the EPD Prefill receiver, or None for non-EPD nodes.


    Args:
        server_args: ServerArgs.
        tp_rank: this rank's TP index.
        tp_size: total TP size.
        zmq_pull_port: ZMQ PULL port for this rank.
        tp_cpu_group: TP CPU process group for all-reduce.
        tensor_device: rank-local target device for received full tensors.

    Returns:
        EpdPrefillReceiver instance, or None if not applicable.
    """
    # Only language_only nodes need the receiver (Encoder-separated LM nodes).
    if not server_args.language_only and not _is_decode_mode(server_args):
        return None

    return EpdPrefillReceiver(
        tp_rank=tp_rank,
        tp_size=tp_size,
        zmq_pull_port=zmq_pull_port,
        timeout_s=server_args.encoder_recv_timeout,
        tp_cpu_group=tp_cpu_group,
        meta_only_mode=_is_decode_mode(server_args),
        tensor_device=tensor_device,
    )
