"""Scheduler mixin for Encoder disaggregation (EPD).

Encapsulates all Scheduler-side logic for communicating with a remote
Encoder server.  This mixin is used by the LM Scheduler (both PD-disaggregated
Prefill nodes and non-PD-disaggregated LM nodes with ``language_only=True``).

Responsibilities:
1. **Receiver initialization** — ``init_epd_receiver`` creates full-result
   receivers only on Prefill PP0 attention-TP ranks and one meta receiver on
   Decode attention-TP0.  Other ranks are descriptor-only participants.
2. **Request routing** — ``route_epd_request`` intercepts requests flagged
   ``need_wait_for_mm_inputs=True``.  Consumer ranks stage them in the
   receiver; descriptor-only ranks keep the raw request in WAIT_DESCRIPTOR.
3. **Delivery-address registration** — ``_epd_register_delivery_addr`` is
   called asynchronously by ``route_epd_request`` to tell the Encoder where
   to push the ViT results.  This is the Scheduler's responsibility (not
   the TokenizerManager's) because the Scheduler owns the ZMQ PULL socket.
4. **Admission polling** — ``poll_epd_admissions`` is called at the top of
   every event loop cycle to poll for new MMResult/MMMeta.  Prefill readiness
   is reduced only inside PP0; PP0/TP0 or Decode/TP0 publishes one canonical
   token descriptor through Scheduler's ordered request broadcast.  Both
   roles tokenize admitted requests on a background worker and commit at most
   ``epd_max_admissions_per_tick`` descriptors per cycle.

Usage:
    class Scheduler(SchedulerEncoderDisaggMixin, ...):
        def __init__(self, ...):
            ...
            self.init_disaggregation()      # existing
            self.init_epd_receiver()        # new: EPD receiver setup

        def event_loop_normal(self):
            while True:
                self.poll_epd_admissions()  # new: EPD admission poll
                ...
                self.process_input_requests(recv_reqs)
                ...

        def _add_request_to_queue(self, req):
            if self.route_epd_request(req):  # new: EPD routing
                return
            ...  # normal queue logic
"""

from __future__ import annotations

import copy
import logging
import queue
import socket
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING

from sglang.srt.disaggregation.encode.admission_prepare import (
    EpdAdmissionPrepareWorker,
    EpdPrepareResult,
    pad_epd_input_ids,
    run_epd_mm_processor,
)
from sglang.srt.disaggregation.encode.encode_data import (
    CanonicalRequestTokenDescriptor,
    CanonicalRequestTokenStatus,
    MMMeta,
    MMResult,
    compute_packed_token_checksum,
    compute_token_checksum,
    pack_token_ids,
    packed_token_count,
    unpack_token_ids,
)
from sglang.srt.disaggregation.encode.encode_receiver import build_epd_prefill_receiver
from sglang.srt.disaggregation.utils import DisaggregationMode, prepare_abort
from sglang.srt.managers.req import AbortCode, Req
from sglang.srt.multimodal.multimodal_processor import get_mm_processor

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class _EpdAdmissionMetricState:
    register_sent_ns: int | None = None
    result_ready_ns: int | None = None


@dataclass
class _EpdPrefillInflight:
    """An admitted Prefill request waiting for its tokenization result.

    Every PP0 attention-TP rank appends entries in receiver admission order
    (rank-synchronized) and removes them only in the rank-synchronized commit
    step, so the FIFO is identical across ranks.  Only local readiness differs.
    """

    request_id: str
    req: Req | None
    result_tuple: tuple[MMMeta, dict[int, MMResult]] | None = None
    error_message: str | None = None
    # Tokenize on the Scheduler thread at commit time (no worker on this rank).
    inline: bool = False
    submitted: bool = False
    result: EpdPrepareResult | None = None


class SchedulerEncoderDisaggMixin:
    """Mixin that encapsulates all EPD (Encoder disaggregation) logic for the Scheduler.

    The host Scheduler class must:
    - Call ``init_epd_receiver()`` after ``init_disaggregation()`` in __init__.
    - Call ``poll_epd_admissions()`` at the top of each event loop cycle.
    - Call ``route_epd_request(req)`` at the top of ``_add_request_to_queue``
      (and the batched variant in ``_extend_requests_to_queue``).
    """

    # ────────────────────────────────────────────────────────────────────────
    # Initialization
    # ────────────────────────────────────────────────────────────────────────

    def _init_epd_admission_metrics(self) -> None:
        if hasattr(self, "_epd_poll_metrics_enabled"):
            return

        self._epd_poll_metrics_enabled: bool = (
            self.enable_metrics
            and self.metrics_collector.enable_epd_multimodal_cat
            and not self._is_decode_mode()
            and self.pp_rank == 0
            and self.attn_tp_rank == 0
        )
        if not self._epd_poll_metrics_enabled:
            return

        self._epd_admission_metric_states: dict[
            str, _EpdAdmissionMetricState
        ] = {}
        self._epd_admission_metric_lock = threading.Lock()

    def _start_epd_admission_metrics(self, request_id: str) -> None:
        if not self._epd_poll_metrics_enabled:
            return
        with self._epd_admission_metric_lock:
            self._epd_admission_metric_states.setdefault(
                request_id, _EpdAdmissionMetricState()
            )

    def _mark_epd_register_sent(self, request_id: str) -> None:
        if not self._epd_poll_metrics_enabled:
            return
        with self._epd_admission_metric_lock:
            state = self._epd_admission_metric_states.get(request_id)
            if state is not None and state.register_sent_ns is None:
                state.register_sent_ns = time.monotonic_ns()

    def _mark_epd_result_ready(
        self, request_id: str, result_ready_ns: int | None
    ) -> None:
        if not self._epd_poll_metrics_enabled or result_ready_ns is None:
            return
        with self._epd_admission_metric_lock:
            state = self._epd_admission_metric_states.get(request_id)
            if state is not None and state.result_ready_ns is None:
                state.result_ready_ns = result_ready_ns

    def _finish_epd_admission_success(self, request_id: str) -> None:
        if not self._epd_poll_metrics_enabled:
            return
        with self._epd_admission_metric_lock:
            state = self._epd_admission_metric_states.pop(request_id, None)
        if state is None:
            return
        admitted_ns = time.monotonic_ns()
        if (
            state.register_sent_ns is None
            or state.result_ready_ns is None
            or not state.register_sent_ns <= state.result_ready_ns <= admitted_ns
        ):
            logger.warning(
                "Skip incomplete EPD poll admission metrics: rid=%s, state=%s",
                request_id,
                state,
            )
            return

        poll_seconds = (
            state.result_ready_ns - state.register_sent_ns
        ) / 1_000_000_000
        admission_seconds = (
            admitted_ns - state.result_ready_ns
        ) / 1_000_000_000
        self.metrics_collector.observe_epd_poll_admission_success(
            poll_seconds, admission_seconds
        )
        logger.info(
            "EPD poll admission metrics: rid=%s, poll_ms=%.3f, admission_ms=%.3f",
            request_id,
            poll_seconds * 1000,
            admission_seconds * 1000,
        )

    def _finish_epd_admission_failed(self, request_id: str) -> None:
        if not self._epd_poll_metrics_enabled:
            return
        with self._epd_admission_metric_lock:
            state = self._epd_admission_metric_states.pop(request_id, None)
        if state is None:
            return
        self.metrics_collector.observe_epd_poll_admission_failed()
        logger.info("EPD poll admission failed: rid=%s", request_id)

    def init_epd_receiver(self) -> None:
        """Initialize the EPD Prefill receiver (if applicable).

        Called from Scheduler.__init__ after init_disaggregation().  Receiver
        ownership is deliberately smaller than descriptor ownership:

          - Prefill/null: PP0 attention-TP ranks receive full MMResult.
          - Decode: only attention-TP0 of the selected DP replica receives
            MMMeta.  The other TP ranks wait for its canonical descriptor.
          - PP > 0 never talks to Encoder and waits for PP0's descriptor.

        Pure-text requests (need_wait_for_mm_inputs=False) bypass the receiver
        entirely and go through the normal scheduling path.

        Idempotent: safe to call even if init_disaggregation already created
        the receiver (PD-prefill path).
        """
        self._init_epd_admission_metrics()
        if self.epd_prefill_receiver is not None:
            return  # already initialized by init_disaggregation (PD-prefill path)

        decode_mode = self._is_decode_mode()
        if not self.server_args.language_only and not decode_mode:
            return

        owns_receiver = (
            self.attn_tp_rank == 0
            if decode_mode
            else (self.pp_rank or 0) == 0
        )
        if not owns_receiver:
            logger.info(
                "EPD descriptor-only rank: pp_rank=%d, dp_rank=%d, "
                "attn_tp_rank=%d, mode=%s (no Encoder Receiver)",
                self.pp_rank or 0,
                self.dp_rank or 0,
                self.attn_tp_rank,
                self.server_args.disaggregation_mode,
            )
            return

        epd_zmq_pull_port = self._compute_epd_zmq_pull_port()
        # Decode has a single meta coordinator, hence no Encoder-readiness
        # collective.  Prefill readiness is PP0 stage-local attention TP and
        # must never include descriptor-only PP stages.
        admission_group = None if decode_mode else self.attn_tp_cpu_group
        admission_rank = 0 if decode_mode else self.attn_tp_rank
        admission_size = 1 if decode_mode else self.attn_tp_size
        tensor_device = None
        if not decode_mode:
            # Use gpu_id rather than TP rank: under DP/PP placement they are
            # not necessarily identical.  The receiver performs explicit H2D
            # after deserializing the device-neutral CPU payload.
            device_type = str(self.device).split(":", 1)[0]
            tensor_device = f"{device_type}:{self.gpu_id}"
        self.epd_prefill_receiver = build_epd_prefill_receiver(
            server_args=self.server_args,
            tp_rank=admission_rank,
            tp_size=admission_size,
            zmq_pull_port=epd_zmq_pull_port,
            tp_cpu_group=admission_group,
            tensor_device=tensor_device,
        )
        if self.epd_prefill_receiver is not None:
            logger.info(
                "EPD Receiver initialized: admission_rank=%d/%d, "
                "pp_rank=%d, dp_rank=%d, attn_tp_rank=%d, "
                "zmq_pull_port=%d, disaggregation_mode=%s",
                admission_rank,
                admission_size,
                self.pp_rank or 0,
                self.dp_rank or 0,
                self.attn_tp_rank,
                epd_zmq_pull_port,
                self.server_args.disaggregation_mode,
            )
            # Start a dedicated background thread for register_delivery_addr
            self._epd_register_queue: queue.Queue[str] = queue.Queue()
            self._epd_register_thread = threading.Thread(
                target=self._epd_register_worker,
                name="epd-register",
                daemon=True,
            )
            self._epd_register_thread.start()

            # EPD post-refactor: tokenization is deferred to admission (after
            # mm_meta arrives on the MMResult).  The Scheduler needs the
            # mm_processor to run process_mm_data_async.  Build it from the
            # model config + tokenizer; this mirrors TokenizerManager's init.
            self._epd_mm_processor = None
            try:
                self._epd_mm_processor = get_mm_processor(
                    self.model_config.hf_config,
                    self.server_args,
                    self.tokenizer,
                )
                logger.info(
                    "EPD mm_processor initialized on Scheduler: %s",
                    type(self._epd_mm_processor).__name__,
                )
            except Exception:
                logger.exception(
                    "Failed to initialize EPD mm_processor on Scheduler; "
                    "tokenization after admission will fail"
                )
            self._init_epd_prepare_worker()

    def _init_epd_prepare_worker(self) -> None:
        """Start this receiver rank's off-thread tokenization worker.

        Runs on the Decode coordinator and on every Prefill PP0 attention-TP
        rank.  On failure the rank keeps the inline path (Scheduler-thread
        tokenization), still bounded by the per-tick admission budget.
        """
        role = "decode" if self._is_decode_mode() else "prefill"
        if self.server_args.disable_epd_async_tokenization:
            logger.info(
                "EPD %s admission: inline tokenization, "
                "max_admissions_per_tick=%d",
                role,
                self._epd_admission_budget,
            )
            return
        try:
            # Private tokenizer copy: HF fast tokenizers are not safe to use
            # concurrently with the Scheduler thread's own instance.
            processor = get_mm_processor(
                self.model_config.hf_config,
                self.server_args,
                copy.deepcopy(self.tokenizer),
            )
        except Exception:
            logger.exception(
                "Failed to initialize EPD %s tokenization worker; "
                "falling back to inline tokenization",
                role,
            )
            return
        self._epd_prepare_worker = EpdAdmissionPrepareWorker(
            processor,
            name=f"epd-prepare-{role}-dp{self.dp_rank or 0}-tp{self.attn_tp_rank}",
        )
        logger.info(
            "EPD %s admission: async tokenization, "
            "max_admissions_per_tick=%d",
            role,
            self._epd_admission_budget,
        )

    def _compute_epd_zmq_pull_port(self) -> int:
        """Compute this rank's ZMQ PULL port.

        """
        dist_init_addr = self.server_args.dist_init_addr
        if dist_init_addr is None:
            # Fallback for single-node default: server port + ZMQ_TCP_PORT_DELTA (233)
            base = self.server_args.port + 233
        else:
            host, port_str = dist_init_addr.split(":")
            base = int(port_str)
        ranks_per_pp_stage = self.server_args.dp_size * self.attn_tp_size
        return (
            base
            + 200
            + (self.pp_rank or 0) * ranks_per_pp_stage
            + (self.dp_rank or 0) * self.attn_tp_size
            + self.attn_tp_rank
        )

    def _compute_epd_receive_url(self) -> str:
        """Compute this rank's ZMQ PULL receive URL for the Encoder to push to."""
        port = self._compute_epd_zmq_pull_port()
        try:
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
        except Exception:
            local_ip = "127.0.0.1"
        return f"tcp://{local_ip}:{port}"

    # ────────────────────────────────────────────────────────────────────────
    # Request routing
    # ────────────────────────────────────────────────────────────────────────

    def route_epd_request(self, req: Req) -> bool:
        """Intercept an EPD-pending request and stage it in the receiver.

        Called at the top of ``_add_request_to_queue``.  Post-refactor, the
        request carries ``epd_media_urls`` (not ``mm_meta``); ``mm_meta``
        arrives later on the ``MMResult``.  This method always records the Req
        in ``_epd_waiting_reqs``.  Ranks owning a Receiver additionally enqueue
        it there and asynchronously register with Encoder; all other ranks wait
        for the coordinator's canonical descriptor.

        Returns:
            True if the request was intercepted (caller must NOT enqueue it
            in the normal queue); False otherwise (caller proceeds normally).
        """
        if not req.need_wait_for_mm_inputs or req.epd_media_urls is None:
            return False
        if req.mm_inputs is None:
            error_message = (
                "EPD request is missing deferred multimodal inputs: "
                f"rid={req.rid}"
            )
            logger.error(error_message)
            self._abort_epd_req(req, error_message)
            return True

        self._epd_waiting_reqs[req.rid] = req
        self._start_epd_admission_metrics(req.rid)
        if self.epd_prefill_receiver is not None:
            self.epd_prefill_receiver.enqueue(
                req.rid,
                req.epd_media_urls,
                compute_to_logical=req.mm_inputs.epd_compute_to_logical,
            )

            # Hand off register_delivery_addr to the background worker thread.
            # The Scheduler thread has no asyncio event loop, so we must NOT use
            # asyncio.create_task here.  queue.put is non-blocking and keeps the
            # HTTP POST entirely off the scheduler thread.
            self._epd_register_queue.put(req.rid)
        else:
            logger.info(
                "EPD request waiting for canonical descriptor: rid=%s, "
                "pp_rank=%d, dp_rank=%d, attn_tp_rank=%d",
                req.rid,
                self.pp_rank or 0,
                self.dp_rank or 0,
                self.attn_tp_rank,
            )
        return True

    def route_epd_requests_batch(self, reqs: list[Req]) -> list[Req]:
        """Batched variant of ``route_epd_request`` for ``_extend_requests_to_queue``.

        Stages all EPD-pending requests in ``reqs`` and returns the remaining
        (non-EPD) requests that the caller should enqueue in the normal queue.

        Returns:
            List of requests NOT intercepted (caller enqueues these normally).
            If all requests were EPD-pending, returns an empty list.
        """
        normal_reqs: list[Req] = []
        for r in reqs:
            if not self.route_epd_request(r):
                normal_reqs.append(r)
        return normal_reqs


    def _epd_register_worker(self) -> None:
        """Background thread: drain the register queue and POST to the Encoder.

        Runs forever (daemon).  Each item is a request_id.  The worker computes
        the payload (tp_rank / receive_url / tp_size) and calls the synchronous
        ``_do_register_delivery_addr`` helper.  Failures are logged but never
        propagated — the Prefill-side timeout in ``EpdPrefillReceiver`` will
        eventually abort the request if the Encoder never pushes.
        """
        while True:
            request_id = self._epd_register_queue.get()
            if request_id is None:
                # Sentinel for shutdown.
                continue
            try:
                self._register_one_delivery_addr(request_id)
            except Exception:
                logger.exception(
                    "EPD register_delivery_addr worker error for rid=%s",
                    request_id,
                )

    def _register_one_delivery_addr(self, request_id: str) -> None:
        """Build payload and synchronously POST register_delivery_addr.

        Mass multi-encoder sharding: registers with **every** encoder in the
        resolved URL list (from ``mm_inputs.epd_encoder_urls`` or static
        ``server_args.encoder_urls`` fallback).  D-side registers with
        ``meta_only=True`` (encoder creates PushMetaJob, pushes only MMMeta).

        This runs in the background worker thread, so blocking is acceptable.
        """
        receive_url = self._compute_epd_receive_url()
        urls = self._resolve_encoder_urls(request_id)
        if not urls:
            logger.error(
                "EPD register_delivery_addr: no encoder_urls for rid=%s",
                request_id,
            )
            return

        meta_only = self._is_decode_mode()
        if meta_only:
            # Only the selected Decode replica's TP0 consumes MMMeta.
            delivery_rank = 0
            delivery_world_size = 1
        else:
            # Full results are consumed only by PP0's attention-TP ranks.
            delivery_rank = self.attn_tp_rank
            delivery_world_size = self.attn_tp_size
        for encoder_url in urls:
            logger.info(
                "EPD register_delivery_addr: rid=%s, encoder=%s, tp_rank=%d, "
                "receive_url=%s, meta_only=%s",
                request_id,
                encoder_url,
                delivery_rank,
                receive_url,
                meta_only,
            )
            payload = {
                "request_id": request_id,
                "tp_rank": delivery_rank,
                "receive_url": receive_url,
                "tp_size": delivery_world_size,
                "meta_only": meta_only,
            }
            self._mark_epd_register_sent(request_id)
            self._do_register_delivery_addr(encoder_url, payload)

    def _is_decode_mode(self) -> bool:

        return self.server_args.disaggregation_mode == DisaggregationMode.DECODE.value

    def _resolve_encoder_urls(self, request_id: str) -> list[str]:
        """Return the encoder URL list for this request.

        Priority:
        1. per-request dynamic list (``mm_inputs.epd_encoder_urls``, set by
           TM from EncoderRouter) — isomorphic to
           ``server_args.encoder_urls.split(",")``, zero transfer_info awareness.
        2. static ``server_args.encoder_urls`` (fallback for pre-Mass /
           transfer_info-missing requests).
        """
        req = self._epd_waiting_reqs.get(request_id)
        if req is not None:
            urls = getattr(req, "epd_encoder_urls", None)
            if urls:
                return list(urls)
        if not self.server_args.encoder_urls:
            return []
        return [u.strip() for u in self.server_args.encoder_urls.split(",") if u.strip()]

    def _resolve_encoder_url(self, request_id: str) -> str | None:
        """Pick a single Encoder URL for a request (deterministic hash).

        Retained for backward compatibility (single-encoder callers).  Mass
        multi-encoder path uses ``_resolve_encoder_urls`` instead.
        """
        urls = self._resolve_encoder_urls(request_id)
        if not urls:
            return None
        return urls[hash(request_id) % len(urls)]

    @staticmethod
    def _do_register_delivery_addr(encoder_url: str, payload: dict) -> None:
        """Synchronous HTTP POST to the Encoder's ``/register_delivery_addr``.

        ``encoder_url`` is the base URL of an Encoder HTTP server
        (e.g. ``http://10.0.0.1:9100``).  Runs in a worker thread via
        ``run_in_executor`` so it never blocks the Scheduler event loop.
        """
        import requests

        try:
            resp = requests.post(
                f"{encoder_url}/register_delivery_addr",
                json=payload,
                timeout=10,
            )
            if resp.status_code != 200:
                logger.warning(
                    "EPD register_delivery_addr failed: rid=%s, status=%d",
                    payload.get("request_id"),
                    resp.status_code,
                )
        except Exception:
            logger.exception(
                "EPD register_delivery_addr HTTP POST failed for rid=%s",
                payload.get("request_id"),
            )

    # ────────────────────────────────────────────────────────────────────────
    # Admission polling (called from the event loop)
    # ────────────────────────────────────────────────────────────────────────

    def poll_epd_admissions(self) -> None:
        """Poll the EpdPrefillReceiver and process admitted/failed requests.
        For admitted requests, consumer ranks prepare their local state and
        the coordinator queues one canonical descriptor.  Success and failure
        both return through Scheduler's ordered broadcast before any rank
        enters a normal scheduling queue.

        Both roles tokenize admitted requests off the Scheduler thread and
        commit a bounded number of descriptors per tick:
        ``_advance_epd_decode_admissions`` on the Decode coordinator and
        ``_advance_epd_prefill_admissions`` on Prefill PP0 attention-TP ranks.
        """
        if self.epd_prefill_receiver is None:
            return

        admitted_ids, failed_ids = self.epd_prefill_receiver.poll_admissions()

        if admitted_ids:
            for request_id in admitted_ids:
                self._mark_epd_result_ready(
                    request_id,
                    self.epd_prefill_receiver.get_result_ready_ns(request_id),
                )
            if self._is_decode_mode():
                self._epd_admitted_backlog.extend(admitted_ids)
            else:
                for request_id in admitted_ids:
                    self._submit_epd_prefill_prepare(request_id)
        if failed_ids:
            self._process_epd_failed(failed_ids)
        if self._is_decode_mode():
            self._advance_epd_decode_admissions()
        else:
            self._advance_epd_prefill_admissions()

    def _advance_epd_decode_admissions(self) -> None:
        """Advance Decode admissions under the per-tick budget.

        Runs only on the Decode coordinator (attention-TP0), the sole
        descriptor source of its DP replica.  With the worker, admitted
        requests are submitted immediately and at most ``budget`` finished
        results are committed; inline, at most ``budget`` backlog entries are
        tokenized.  Requests aborted meanwhile are dropped without a
        descriptor: every rank already removed them via the ordered AbortReq.
        """
        budget = self._epd_admission_budget
        worker = self._epd_prepare_worker
        if worker is not None and not worker.is_alive():
            self._fail_epd_prepare_worker(worker)
            worker = None

        if worker is None:
            batch: list[str] = []
            while self._epd_admitted_backlog and len(batch) < budget:
                request_id = self._epd_admitted_backlog.popleft()
                if request_id in self._epd_waiting_reqs:
                    batch.append(request_id)
            if batch:
                self._process_epd_admitted(batch)
            return

        while self._epd_admitted_backlog:
            self._submit_epd_prepare(worker, self._epd_admitted_backlog.popleft())
        for result in worker.drain(budget):
            self._commit_epd_prepare_result(result)

    def _submit_epd_prepare(
        self, worker: EpdAdmissionPrepareWorker, request_id: str
    ) -> None:
        """Hand one admitted Decode request to the tokenization worker."""
        req = self._epd_waiting_reqs.get(request_id)
        if req is None:
            # Aborted after admission; abort_request already cancelled the
            # receiver entry.
            return
        error_message = None
        result_tuple = None
        try:
            result_tuple = self.epd_prefill_receiver.get_received_result(request_id)
        except Exception as exc:
            logger.exception(
                "Unexpected EPD preparation failure for rid=%s", request_id
            )
            error_message = (
                f"Unexpected EPD preparation failure for rid={request_id}: {exc}"
            )
        if error_message is None and result_tuple is None:
            error_message = f"EPD admit: no received result for rid={request_id}"
        if error_message is None and result_tuple[0] is None:
            error_message = f"EPD admit: missing mm_meta for rid={request_id}"
        if error_message is not None:
            self._enqueue_epd_descriptor(
                request_id,
                status=CanonicalRequestTokenStatus.FAILED,
                error_message=error_message,
            )
            return

        self._epd_preparing[request_id] = req
        worker.submit(
            request_id,
            [mu.url for mu in req.epd_media_urls or []],
            req.epd_input_text or "",
            result_tuple[0].item_metas,
        )

    def _commit_epd_prepare_result(self, result: EpdPrepareResult) -> None:
        """Queue the canonical descriptor for one finished worker result."""
        request_id = result.request_id
        req = self._epd_preparing.pop(request_id, None)
        if req is None or self._epd_waiting_reqs.get(request_id) is not req:
            logger.info(
                "Drop EPD preparation result of aborted request: rid=%s",
                request_id,
            )
            return
        if result.error_message is not None:
            self._enqueue_epd_descriptor(
                request_id,
                status=CanonicalRequestTokenStatus.FAILED,
                error_message=result.error_message,
            )
            return
        logger.info(
            "EPD decode tokenization done: rid=%s, input_ids_len=%d, "
            "queue_ms=%.3f, prepare_ms=%.3f",
            request_id,
            result.num_tokens,
            result.queue_ms,
            result.prepare_ms,
        )
        self._enqueue_epd_descriptor(
            request_id,
            status=CanonicalRequestTokenStatus.TOKEN_READY,
            packed_input_ids=result.packed_input_ids,
            token_checksum=result.token_checksum,
        )

    def _fail_epd_prepare_worker(self, worker: EpdAdmissionPrepareWorker) -> None:
        """Settle a dead worker's requests and fall back to inline tokenization.

        Already finished results are still committed (outside the per-tick
        budget, since this happens at most once); the rest fail.
        """
        logger.error(
            "EPD decode tokenization worker exited with %d in-flight requests; "
            "falling back to inline tokenization",
            len(self._epd_preparing),
        )
        for result in worker.drain(len(self._epd_preparing)):
            self._commit_epd_prepare_result(result)
        for request_id, req in self._epd_preparing.items():
            if self._epd_waiting_reqs.get(request_id) is req:
                self._enqueue_epd_descriptor(
                    request_id,
                    status=CanonicalRequestTokenStatus.FAILED,
                    error_message=(
                        "EPD decode tokenization worker exited before "
                        f"rid={request_id} was prepared"
                    ),
                )
        self._epd_preparing.clear()
        self._epd_prepare_worker = None

    def _submit_epd_prefill_prepare(self, request_id: str) -> None:
        """Queue one admitted Prefill request for (off-thread) tokenization.

        Every PP0 attention-TP rank appends the same entry in the same order;
        a local failure only makes this rank report the entry ready early, and
        the failure then propagates through ``_sync_epd_prepare_result``.
        """
        req = self._epd_waiting_reqs.get(request_id)
        entry = _EpdPrefillInflight(request_id=request_id, req=req)
        self._epd_prefill_inflight.append(entry)
        self._epd_prefill_inflight_by_rid[request_id] = entry
        if req is None:
            return  # aborted; dropped at commit on every rank
        try:
            result_tuple = self.epd_prefill_receiver.get_received_result(request_id)
        except Exception as exc:
            logger.exception(
                "Unexpected EPD preparation failure for rid=%s", request_id
            )
            entry.error_message = (
                f"Unexpected EPD preparation failure for rid={request_id}: {exc}"
            )
            return
        entry.error_message = self._check_epd_received_result(
            request_id, result_tuple, meta_only=False
        )
        if entry.error_message is not None:
            return
        entry.result_tuple = result_tuple
        worker = self._epd_prepare_worker
        if worker is None:
            entry.inline = True
            return
        worker.submit(
            request_id,
            [mu.url for mu in req.epd_media_urls or []],
            req.epd_input_text or "",
            result_tuple[0].item_metas,
            meta_only=False,
        )
        entry.submitted = True

    def _advance_epd_prefill_admissions(self) -> None:
        """Commit the FIFO prefix that is ready on every PP0 attention-TP rank.

        At most ``budget`` entries per tick.  One MIN all-reduce over the
        readiness of the FIFO head makes every rank run the per-request
        ``_sync_epd_prepare_result`` for the same requests in the same order.
        It runs only while the FIFO is non-empty, which is rank-identical.
        """
        inflight = self._epd_prefill_inflight
        if not inflight:
            return
        self._collect_epd_prefill_results()
        count = min(self._epd_admission_budget, len(inflight))
        local_ready = [
            int(self._is_epd_prefill_entry_ready(inflight[i])) for i in range(count)
        ]
        for ready in self._sync_epd_ready_flags(local_ready):
            if not ready:
                break
            entry = inflight.popleft()
            del self._epd_prefill_inflight_by_rid[entry.request_id]
            self._commit_epd_prefill_entry(entry)

    def _collect_epd_prefill_results(self) -> None:
        """Attach finished worker results; fail local work of a dead worker."""
        worker = self._epd_prepare_worker
        if worker is None:
            return
        # Check liveness first: a dead worker has published all its results.
        alive = worker.is_alive()
        for result in worker.drain():
            entry = self._epd_prefill_inflight_by_rid.get(result.request_id)
            if entry is not None and entry.submitted and entry.result is None:
                entry.result = result
        if alive:
            return
        pending = [
            entry
            for entry in self._epd_prefill_inflight
            if entry.submitted and entry.result is None
        ]
        logger.error(
            "EPD prefill tokenization worker exited with %d in-flight requests; "
            "falling back to inline tokenization",
            len(pending),
        )
        for entry in pending:
            entry.error_message = (
                "EPD prefill tokenization worker exited before "
                f"rid={entry.request_id} was prepared"
            )
        self._epd_prepare_worker = None

    def _is_epd_prefill_entry_ready(self, entry: _EpdPrefillInflight) -> bool:
        if entry.req is None or self._epd_waiting_reqs.get(entry.request_id) is not entry.req:
            return True  # aborted: dropped at commit
        return (
            entry.error_message is not None
            or entry.inline
            or entry.result is not None
        )

    def _sync_epd_ready_flags(self, flags: list[int]) -> list[int]:
        """MIN-reduce per-entry readiness across PP0 attention-TP ranks."""
        if self.attn_tp_size <= 1 or not flags:
            return flags

        import torch
        import torch.distributed as dist

        tensor = torch.tensor(flags, dtype=torch.int32, device="cpu")
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=self.attn_tp_cpu_group)
        return tensor.tolist()

    def _commit_epd_prefill_entry(self, entry: _EpdPrefillInflight) -> None:
        """Bind features, verify canonical tokens across ranks, queue descriptor."""
        request_id = entry.request_id
        req = entry.req
        if req is None or self._epd_waiting_reqs.get(request_id) is not req:
            logger.info(
                "Drop EPD preparation result of aborted request: rid=%s",
                request_id,
            )
            return
        input_ids: list[int] = []
        error_message = entry.error_message
        if error_message is None:
            try:
                if entry.inline:
                    input_ids, error_message = self._prepare_epd_request(
                        req, entry.result_tuple, meta_only=False
                    )
                elif entry.result.error_message is not None:
                    error_message = entry.result.error_message
                else:
                    mm_meta, mm_results = entry.result_tuple
                    input_ids, error_message = self._bind_epd_request(
                        req,
                        mm_meta,
                        mm_results,
                        entry.result.mm_inputs,
                        entry.result.proc_output,
                        meta_only=False,
                    )
                    if self.attn_tp_rank == 0:
                        logger.info(
                            "EPD prefill tokenization done: rid=%s, "
                            "input_ids_len=%d, queue_ms=%.3f, prepare_ms=%.3f",
                            request_id,
                            len(input_ids),
                            entry.result.queue_ms,
                            entry.result.prepare_ms,
                        )
            except Exception as exc:
                logger.exception(
                    "Unexpected EPD preparation failure for rid=%s", request_id
                )
                input_ids = []
                error_message = (
                    f"Unexpected EPD preparation failure for rid={request_id}: {exc}"
                )
        self._publish_epd_prepare_result(request_id, input_ids, error_message)

    def _publish_epd_prepare_result(
        self,
        request_id: str,
        input_ids: list[int],
        error_message: str | None,
    ) -> None:
        """Agree on the canonical tokens across ranks, then queue a descriptor."""
        local_ok = error_message is None
        packed_input_ids = pack_token_ids(input_ids) if local_ok else b""
        token_checksum = (
            compute_packed_token_checksum(packed_input_ids) if local_ok else 0
        )
        prepare_ok, sync_error = self._sync_epd_prepare_result(
            local_ok, input_ids, token_checksum=token_checksum
        )
        if not prepare_ok:
            self._enqueue_epd_descriptor(
                request_id,
                status=CanonicalRequestTokenStatus.FAILED,
                error_message=sync_error or error_message or (
                    f"EPD canonical token preparation failed for rid={request_id}"
                ),
            )
            return
        self._enqueue_epd_descriptor(
            request_id,
            status=CanonicalRequestTokenStatus.TOKEN_READY,
            packed_input_ids=packed_input_ids,
            token_checksum=token_checksum,
        )

    def _process_epd_admitted(self, admitted_ids: list[str]) -> None:
        """Prepare rank-local MM state, then publish one canonical descriptor.

        Inline path used by the Decode coordinator when it has no tokenization
        worker (Prefill uses ``_advance_epd_prefill_admissions``).  No rank
        enters the normal queue until the descriptor returns through
        Scheduler's ordered request broadcast.
        """

        meta_only = self._is_decode_mode()
        for rid in admitted_ids:
            req = self._epd_waiting_reqs.get(rid)
            input_ids: list[int] = []
            error_message: str | None = None

            if req is None:
                error_message = f"EPD admit: missing waiting request for rid={rid}"
            else:
                try:
                    result_tuple = self.epd_prefill_receiver.get_received_result(rid)
                    if result_tuple is None:
                        error_message = f"EPD admit: no received result for rid={rid}"
                    else:
                        input_ids, error_message = self._prepare_epd_request(
                            req, result_tuple, meta_only=meta_only
                        )
                except Exception as exc:
                    logger.exception(
                        "Unexpected EPD preparation failure for rid=%s", rid
                    )
                    error_message = (
                        f"Unexpected EPD preparation failure for rid={rid}: {exc}"
                    )

            prepare_ok, sync_error = self._sync_epd_prepare_result(
                error_message is None, input_ids
            )
            if not prepare_ok:
                error_message = sync_error or error_message or (
                    f"EPD canonical token preparation failed for rid={rid}"
                )
                self._enqueue_epd_descriptor(
                    rid,
                    status=CanonicalRequestTokenStatus.FAILED,
                    error_message=error_message,
                )
                continue

            self._enqueue_epd_descriptor(
                rid,
                status=CanonicalRequestTokenStatus.TOKEN_READY,
                input_ids=input_ids,
            )

    def _prepare_epd_request(
        self,
        req: Req,
        result_tuple: tuple[MMMeta, dict[int, MMResult]],
        *,
        meta_only: bool,
    ) -> tuple[list[int], str | None]:
        """Build this receiver rank's request state without scheduling it."""

        import asyncio

        rid = req.rid
        error_message = self._check_epd_received_result(
            rid, result_tuple, meta_only=meta_only
        )
        if error_message is not None:
            return [], error_message
        mm_meta, mm_results = result_tuple

        media_urls = req.epd_media_urls or []
        image_data = [mu.url for mu in media_urls]
        input_text = req.epd_input_text or ""
        if self._epd_mm_processor is None:
            return [], f"EPD admission: mm_processor unavailable for rid={rid}"

        mm_inputs, proc_output, error_message = run_epd_mm_processor(
            self._epd_mm_processor,
            request_id=rid,
            image_data=image_data,
            input_text=input_text,
            item_metas=mm_meta.item_metas,
            run_coro=asyncio.run,
        )
        if error_message is not None:
            return [], error_message
        return self._bind_epd_request(
            req, mm_meta, mm_results, mm_inputs, proc_output, meta_only=meta_only
        )

    @staticmethod
    def _check_epd_received_result(
        request_id: str,
        result_tuple: tuple[MMMeta, dict[int, MMResult]] | None,
        *,
        meta_only: bool,
    ) -> str | None:
        if result_tuple is None:
            return f"EPD admit: no received result for rid={request_id}"
        mm_meta, mm_results = result_tuple
        if mm_meta is None:
            return f"EPD admit: missing mm_meta for rid={request_id}"
        if not meta_only and not mm_results:
            return f"EPD admit: missing mm_results for rid={request_id}"
        return None

    def _bind_epd_request(
        self,
        req: Req,
        mm_meta: MMMeta,
        mm_results: dict[int, MMResult],
        mm_inputs,
        proc_output,
        *,
        meta_only: bool,
    ) -> tuple[list[int], str | None]:
        """Attach received media results to tokenized inputs and pad the ids.

        Runs on the Scheduler thread: Prefill MMResult tensors already live on
        this rank's device.
        """

        import torch as _torch

        rid = req.rid
        mm_inputs.mm_meta = mm_meta
        mm_inputs.image_pad_len = [
            item.num_tokens for item in mm_meta.item_metas
        ]
        mm_inputs.num_image_tokens = sum(mm_inputs.image_pad_len)

        if not meta_only:
            # The receiver splits each Encoder shard into global media-indexed
            # results.  Consume those entries directly so arrival order and
            # Encoder shard boundaries cannot change item-to-feature mapping.
            missing_indices = [
                item_meta.item_index
                for item_meta in mm_meta.item_metas
                if item_meta.item_index not in mm_results
                or mm_results[item_meta.item_index].data is None
            ]
            if missing_indices:
                return [], (
                    "EPD admission: missing media results "
                    f"{missing_indices} for rid={rid}"
                )

            for item_meta in mm_meta.item_metas:
                if not 0 <= item_meta.item_index < len(mm_inputs.mm_items):
                    return [], (
                        "EPD admission: item index out of range "
                        f"rid={rid}, item_index={item_meta.item_index}, "
                        f"mm_items={len(mm_inputs.mm_items)}"
                    )
                embed = mm_results[item_meta.item_index].data
                item = mm_inputs.mm_items[item_meta.item_index]
                item.feature = embed.contiguous()
                item.hash = item_meta.hash
                item.pad_value = item_meta.pad_value
                t, h, w = item_meta.grid_thw
                item.model_specific_data["image_grid_thw"] = _torch.tensor(
                    [[t, h, w]], dtype=_torch.int64
                )

        input_ids = pad_epd_input_ids(mm_inputs, proc_output)
        self._set_epd_token_fields(req, input_ids)
        req.mm_inputs = None if meta_only else mm_inputs
        return input_ids, None

    def _sync_epd_prepare_result(
        self,
        local_ok: bool,
        input_ids: list[int],
        token_checksum: int | None = None,
    ) -> tuple[bool, str | None]:
        """Verify PP0 TP ranks prepared the same canonical token sequence.

        A single MIN reduction carries both the all-ranks success predicate
        and the min/max token signature.  ``MIN(-x) == -MAX(x)`` lets one
        reduction recover the maximum length/checksum without a second op.
        ``token_checksum`` may be precomputed by the caller for ``input_ids``.
        """

        if self._is_decode_mode() or self.attn_tp_size <= 1:
            return (
                local_ok,
                None if local_ok else "Decode meta coordinator preparation failed",
            )

        import torch
        import torch.distributed as dist

        if local_ok:
            token_length = len(input_ids)
            if token_checksum is None:
                token_checksum = compute_token_checksum(input_ids)
            packed = torch.tensor(
                [
                    1,
                    token_length,
                    token_checksum,
                    -token_length,
                    -token_checksum,
                ],
                dtype=torch.int64,
                device="cpu",
            )
        else:
            # Signature fields are ignored when the reduced all_ok field is 0.
            # Zeroes avoid sentinels and negation edge cases in int64.
            packed = torch.zeros(5, dtype=torch.int64, device="cpu")

        dist.all_reduce(
            packed, op=dist.ReduceOp.MIN, group=self.attn_tp_cpu_group
        )

        if packed[0].item() != 1:
            return False, "At least one PP0 attention-TP rank failed EPD preparation"
        min_signature = packed[1:3]
        max_signature = -packed[3:5]
        if not torch.equal(min_signature, max_signature):
            return False, (
                "PP0 attention-TP canonical token mismatch: "
                f"min={min_signature.tolist()}, max={max_signature.tolist()}"
            )
        return True, None

    def _enqueue_epd_descriptor(
        self,
        request_id: str,
        *,
        status: CanonicalRequestTokenStatus,
        input_ids: list[int] | tuple[int, ...] | None = None,
        error_message: str | None = None,
        packed_input_ids: bytes | None = None,
        token_checksum: int | None = None,
    ) -> None:
        """Enqueue a schema-2 descriptor on the sole Scheduler broadcast source rank.

        Pass either ``input_ids`` or an already packed ``packed_input_ids``
        with its ``token_checksum`` (e.g. computed off-thread).
        """

        if (self.pp_rank or 0) != 0 or self.attn_tp_rank != 0:
            return
        if status != CanonicalRequestTokenStatus.TOKEN_READY:
            packed_input_ids, token_checksum = b"", 0
        elif packed_input_ids is None:
            packed_input_ids = pack_token_ids(input_ids or ())
            token_checksum = compute_packed_token_checksum(packed_input_ids)
        elif token_checksum is None:
            token_checksum = compute_packed_token_checksum(packed_input_ids)
        self._epd_descriptor_generation += 1
        descriptor = CanonicalRequestTokenDescriptor(
            request_id=request_id,
            generation=self._epd_descriptor_generation,
            status=status,
            token_checksum=token_checksum,
            error_message=error_message,
            schema_version=2,
            packed_input_ids=packed_input_ids,
        )
        self._epd_descriptor_outbox.append(descriptor)
        logger.info(
            "EPD canonical descriptor queued: rid=%s, generation=%d, "
            "status=%s, input_ids_len=%d",
            request_id,
            descriptor.generation,
            descriptor.status.value,
            packed_token_count(packed_input_ids),
        )

    def drain_epd_descriptors(self) -> list[CanonicalRequestTokenDescriptor]:
        """Drain source-rank descriptors into Scheduler's ordered frame."""

        if (self.pp_rank or 0) != 0 or self.attn_tp_rank != 0:
            return []
        descriptors = list(self._epd_descriptor_outbox)
        self._epd_descriptor_outbox.clear()
        return descriptors

    def apply_epd_descriptor(
        self, descriptor: CanonicalRequestTokenDescriptor
    ) -> None:
        """Atomically commit a descriptor on every rank in this replica."""

        if descriptor.schema_version not in (1, 2):
            raise RuntimeError(
                "Unsupported EPD descriptor schema: "
                f"version={descriptor.schema_version}, rid={descriptor.request_id}"
            )
        expected_generation = self._epd_descriptor_last_applied + 1
        if descriptor.generation != expected_generation:
            raise RuntimeError(
                "EPD descriptor generation mismatch: "
                f"expected={expected_generation}, got={descriptor.generation}, "
                f"rid={descriptor.request_id}"
            )
        self._epd_descriptor_last_applied = descriptor.generation

        req = self._epd_waiting_reqs.pop(descriptor.request_id, None)
        if req is None:
            self._finish_epd_admission_failed(descriptor.request_id)
            raise RuntimeError(
                "EPD descriptor has no waiting request: "
                f"rid={descriptor.request_id}, generation={descriptor.generation}"
            )

        if descriptor.status != CanonicalRequestTokenStatus.TOKEN_READY:
            self._finish_epd_admission_failed(descriptor.request_id)
            self._abort_epd_req(
                req,
                descriptor.error_message
                or f"EPD canonicalization failed for rid={descriptor.request_id}",
            )
            return

        if descriptor.schema_version >= 2:
            # Verify the received bytes, then rebuild the list in one C call.
            actual_checksum = compute_packed_token_checksum(
                descriptor.packed_input_ids
            )
            input_ids = None
        else:
            input_ids = list(descriptor.input_ids)
            actual_checksum = compute_token_checksum(input_ids)
        if actual_checksum != descriptor.token_checksum:
            self._finish_epd_admission_failed(descriptor.request_id)
            raise RuntimeError(
                "EPD descriptor checksum mismatch: "
                f"rid={descriptor.request_id}, expected={descriptor.token_checksum}, "
                f"actual={actual_checksum}"
            )

        if input_ids is None:
            input_ids = unpack_token_ids(descriptor.packed_input_ids)
        self._set_epd_token_fields(req, input_ids)
        if self._is_decode_mode() or (self.pp_rank or 0) != 0:
            req.mm_inputs = None
        req.need_wait_for_mm_inputs = False
        req.mm_meta = None
        req.epd_media_urls = None
        req.epd_input_text = None
        req.epd_encoder_urls = None

        logger.info(
            "EPD canonical descriptor committed: rid=%s, generation=%d, "
            "input_ids_len=%d, pp_rank=%d, dp_rank=%d, attn_tp_rank=%d",
            descriptor.request_id,
            descriptor.generation,
            len(input_ids),
            self.pp_rank or 0,
            self.dp_rank or 0,
            self.attn_tp_rank,
        )
        self._finalize_request_after_tokenization(req)
        self._finish_epd_admission_success(descriptor.request_id)

    def _process_epd_failed(self, failed_ids: list[str]) -> None:
        """Publish Encoder failure through the same ordered descriptor path."""
        for rid in failed_ids:
            self._enqueue_epd_descriptor(
                rid,
                status=CanonicalRequestTokenStatus.FAILED,
                error_message=f"EPD encode failed or timed out for rid={rid}",
            )

    @staticmethod
    def _set_epd_token_fields(req: Req, input_ids: list[int]) -> None:
        """Share one canonical list across the Req token fields.

        ``Req.__init__`` aliases ``origin_input_ids``/``origin_input_ids_unpadded``
        the same way; ``fill_ids`` is always rebuilt as a new list before use.
        Copying a 680K-token list four times costs ~10 ms per rank.
        """
        req.origin_input_ids = input_ids
        req.origin_input_ids_unpadded = input_ids
        req.input_ids = input_ids
        req.fill_ids = input_ids

    def _abort_epd_req(self, req: Req, error_message: str) -> None:
        """Abort a failed EPD request and stream the error back to the client."""
        prepare_abort(
            req,
            error_message,
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            err_type=AbortCode.TransferFailed,
        )
        # Stream once: the last PP stage's attention-TP0 owns user output.
        if self.pp_group.is_last_rank and self.attn_tp_rank == 0:
            self.stream_output([req], req.return_logprob, None)
