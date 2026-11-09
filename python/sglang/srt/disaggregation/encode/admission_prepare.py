"""Off-thread deferred tokenization for EPD admission.

Building the canonical token sequence of an admitted EPD request re-runs the
multimodal processor over the full expanded prompt, which takes seconds for
prompts of hundreds of thousands of tokens.  On Decode every Scheduler tick is
a forward step that all DP replicas join through EP collectives; on Prefill
every PP0 attention-TP rank tokenizes and the other PP stages wait for its
descriptors.  Either way this must not run on the Scheduler thread.

- Decode (``meta_only``) jobs return the padded canonical ids packed for the
  descriptor (``pack_token_ids``) and their checksum.
- Prefill jobs return the CPU ``MultimodalInputs``; the Scheduler thread binds
  the received MMResult features (device tensors) and pads afterwards.

``EpdAdmissionPrepareWorker`` runs the processor on one daemon thread:

- It owns a dedicated processor/tokenizer.  HF fast tokenizers must not be
  shared with the Scheduler thread: ``__call__`` mutates truncation/padding
  state on the Rust backend while another thread may be encoding
  ("Already borrowed").  The encode itself goes through ``encode_batch``,
  which releases the GIL.
- It never touches ``Req`` or device tensors.  The Scheduler submits immutable
  jobs and drains results under its own per-tick admission budget.
- A single worker processes jobs FIFO; the processor is not known to be
  re-entrant.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Optional

from sglang.srt.disaggregation.encode.encode_data import (
    MMItemMeta,
    compute_packed_token_checksum,
    pack_token_ids,
)
from sglang.srt.managers.scheduling.multimodal_data import (
    MultimodalInputs,
    MultimodalProcessorOutput,
)

logger = logging.getLogger(__name__)


def pad_epd_input_ids(
    mm_inputs: MultimodalInputs, proc_output: MultimodalProcessorOutput
) -> list[int]:
    """Replace multimodal placeholder tokens in input_ids with pad_values."""
    from sglang.srt.multimodal.mm_utils import (
        MultiModalityDataPaddingPatternMultimodalTokens,
        MultiModalityDataPaddingPatternTokenPairs,
    )

    input_ids = proc_output.input_ids.tolist() \
        if hasattr(proc_output.input_ids, "tolist") else proc_output.input_ids
    if mm_inputs.mm_items and isinstance(input_ids, list):
        if (
            mm_inputs.im_start_id is not None
            or mm_inputs.im_end_id is not None
        ):
            data_token_pairs = []
            if (
                mm_inputs.im_start_id is not None
                and mm_inputs.im_end_id is not None
            ):
                data_token_pairs.append(
                    (mm_inputs.im_start_id, mm_inputs.im_end_id)
                )
            pattern = MultiModalityDataPaddingPatternTokenPairs(data_token_pairs)
        else:
            pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        input_ids = pattern.pad_input_tokens(input_ids, mm_inputs)
    return list(input_ids)


def run_epd_mm_processor(
    processor: Any,
    *,
    request_id: str,
    image_data: list[str],
    input_text: str,
    item_metas: list[MMItemMeta],
    run_coro: Callable[[Any], Any],
) -> tuple[
    Optional[MultimodalInputs], Optional[MultimodalProcessorOutput], Optional[str]
]:
    """Tokenize an EPD prompt from Encoder metadata.

    Returns ``(mm_inputs, proc_output, None)`` on success or
    ``(None, None, error_message)`` on failure.  ``run_coro`` drives the
    processor coroutine (``asyncio.run`` inline, a persistent loop off-thread).
    """
    try:
        proc_output = run_coro(
            processor.process_mm_data_async(
                image_data=image_data,
                audio_data=None,
                input_text=input_text,
                request_obj=None,
                epd_item_metas=item_metas,
            )
        )
    except Exception as exc:
        logger.exception("EPD admission tokenization failed for rid=%s", request_id)
        return None, None, f"EPD tokenization failed for rid={request_id}: {exc}"

    if proc_output is None or not proc_output.mm_items:
        return None, None, f"EPD admission: no mm_items for rid={request_id}"

    mm_inputs = MultimodalInputs.from_processor_output(proc_output)
    if len(mm_inputs.mm_items) != len(item_metas):
        return None, None, (
            "EPD admission: mm_items/mm_meta count mismatch "
            f"for rid={request_id}: mm_items={len(mm_inputs.mm_items)}, "
            f"item_metas={len(item_metas)}"
        )
    return mm_inputs, proc_output, None


@dataclass(frozen=True)
class EpdPrepareJob:
    request_id: str
    image_data: tuple[str, ...]
    input_text: str
    item_metas: tuple[MMItemMeta, ...]
    submitted_ns: int
    meta_only: bool = True


@dataclass(frozen=True)
class EpdPrepareResult:
    request_id: str
    # Decode (meta_only) jobs only: ``pack_token_ids`` payload and its checksum.
    packed_input_ids: bytes
    num_tokens: int
    token_checksum: int
    error_message: Optional[str]
    queue_ms: float
    prepare_ms: float
    # Prefill (non meta_only) jobs only.
    mm_inputs: Optional[MultimodalInputs] = None
    proc_output: Optional[MultimodalProcessorOutput] = None


class EpdAdmissionPrepareWorker:
    """Single background thread producing canonical EPD token sequences."""

    def __init__(self, processor: Any, name: str = "epd-admission-prepare"):
        self._processor = processor
        self._jobs: queue.SimpleQueue[Optional[EpdPrepareJob]] = queue.SimpleQueue()
        # deque append/popleft are atomic, so the worker can publish results
        # without a lock and the Scheduler can drain them non-blockingly.
        self._results: deque[EpdPrepareResult] = deque()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(
        self,
        request_id: str,
        image_data: list[str],
        input_text: str,
        item_metas: list[MMItemMeta],
        meta_only: bool = True,
    ) -> None:
        self._jobs.put(
            EpdPrepareJob(
                request_id=request_id,
                image_data=tuple(image_data),
                input_text=input_text,
                item_metas=tuple(item_metas),
                submitted_ns=time.monotonic_ns(),
                meta_only=meta_only,
            )
        )

    def drain(self, max_results: Optional[int] = None) -> list[EpdPrepareResult]:
        """Pop finished results in submission order (all if ``max_results`` is None)."""
        results: list[EpdPrepareResult] = []
        while max_results is None or len(results) < max_results:
            try:
                results.append(self._results.popleft())
            except IndexError:
                break
        return results

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def close(self, timeout: Optional[float] = None) -> None:
        self._jobs.put(None)
        self._thread.join(timeout)

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            while True:
                job = self._jobs.get()
                if job is None:
                    return
                self._results.append(self._prepare(job, loop))
        finally:
            loop.close()

    def _prepare(
        self, job: EpdPrepareJob, loop: asyncio.AbstractEventLoop
    ) -> EpdPrepareResult:
        start_ns = time.monotonic_ns()
        packed_input_ids = b""
        num_tokens = 0
        token_checksum = 0
        mm_inputs = proc_output = None
        try:
            mm_inputs, proc_output, error_message = run_epd_mm_processor(
                self._processor,
                request_id=job.request_id,
                image_data=list(job.image_data),
                input_text=job.input_text,
                item_metas=list(job.item_metas),
                run_coro=loop.run_until_complete,
            )
            if error_message is None and job.meta_only:
                input_ids = pad_epd_input_ids(mm_inputs, proc_output)
                packed_input_ids = pack_token_ids(input_ids)
                num_tokens = len(input_ids)
                token_checksum = compute_packed_token_checksum(packed_input_ids)
                mm_inputs = proc_output = None
        except Exception as exc:
            logger.exception(
                "Unexpected EPD admission preparation failure for rid=%s",
                job.request_id,
            )
            packed_input_ids = b""
            num_tokens = 0
            mm_inputs = proc_output = None
            error_message = (
                "Unexpected EPD preparation failure for "
                f"rid={job.request_id}: {exc}"
            )
        end_ns = time.monotonic_ns()
        return EpdPrepareResult(
            request_id=job.request_id,
            packed_input_ids=packed_input_ids,
            num_tokens=num_tokens,
            token_checksum=token_checksum,
            error_message=error_message,
            queue_ms=(start_ns - job.submitted_ns) / 1e6,
            prepare_ms=(end_ns - start_ns) / 1e6,
            mm_inputs=mm_inputs,
            proc_output=proc_output,
        )
