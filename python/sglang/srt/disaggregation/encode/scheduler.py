"""Encoder batching, media preprocessing, caches and push transports.

EncoderMediaCoordinator owns bounded request/media state on the event loop.
EncodeScheduler provides TP-aligned intake and media-task batching; only actual
full registrations allocate request-level result collectors. Meta-only requests retain
lightweight descriptors and reuse the same bounded media caches.
"""

from __future__ import annotations

import abc
import asyncio
import hashlib
import os
import threading
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import zmq

from sglang.srt.managers.scheduling.multimodal_data import Modality, MultimodalDataItem
from sglang.srt.utils import get_colorful_logger

from .encode_data import (
    MMItemMeta,
    MMMeta,
    MMResult,
    MediaURL,
    PendingEncodeItem,
)
from .device_worker import EncoderDeviceWorker
from .utils import Timer

if TYPE_CHECKING:
    from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult, MultiModalStaticCache
    from sglang.srt.mem_cache.tiered_embedding_cache import TieredEmbeddingCache
    from sglang.srt.observability.metrics.collector import EncoderMetricsCollector

    from .encoder_base import PreprocessedItem


logger = get_colorful_logger(__name__)

_FLUSH_CACHE_CONTROL_TAG = "__encoder_flush_cache__"


@dataclass(frozen=True)
class MediaURLCacheKey:
    """Versioned key for the URL memo that fronts the content-hash cache."""

    schema_version: int
    modality: Modality
    content_identity_digest: str
    source_version: str
    preprocess_signature: str
    output_format: str


@dataclass
class MediaURLCacheEntry:
    """Small URL-to-content-hash index entry; never stores media tensors."""

    modality: Modality
    item_hash: int
    mm_meta: MMItemMeta
    expires_at: float


def _clone_mm_item_meta(meta: MMItemMeta, item_index: int) -> MMItemMeta:
    """Clone cached metadata without leaking request-local mutable state."""
    return MMItemMeta(
        item_index=item_index,
        hash=meta.hash,
        pad_value=meta.pad_value,
        grid_thw=list(meta.grid_thw),
        num_tokens=meta.num_tokens,
        embedding_dim=meta.embedding_dim,
        video_timestamps=list(meta.video_timestamps),
    )


def _embedding_cache_key(item_hash: int, output_format: str) -> int:
    """Scope non-default ViT cache entries by output representation.

    Keep the established rvq_indices key unchanged for compatibility while
    preventing a vit_embedding request from reading RVQ data for the same
    preprocessed media hash.
    """
    if output_format == "rvq_indices":
        return item_hash
    digest = hashlib.sha256(
        f"{item_hash}:{output_format}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


# ═══════════════════════════════════════════════════════════════════════════
#  Push jobs — data push to Prefill TP ranks
# ═══════════════════════════════════════════════════════════════════════════


class MMPushJobBase(abc.ABC):
    """Abstract base class for data push jobs.

    Core design: receive_url (delivery addr) and data (ViT result) arrive
    asynchronously. Push is triggered only when both are ready.

    Phase 1: ZMQ PUSH (ZMQPushJob)
    Phase 2: Mooncake RDMA write (MooncakePushJob)
    """

    def __init__(
        self,
        request_id: str,
        tp_rank: int,
        metrics_collector: Optional["EncoderMetricsCollector"] = None,
        job_type: str = "full",
    ):
        self.request_id = request_id
        self.tp_rank = tp_rank
        self.metrics_collector = metrics_collector
        self.job_type = job_type
        self.receive_url: Optional[str] = None
        self.data: Optional[MMResult] = None
        self._url_event = asyncio.Event()
        self._data_event = asyncio.Event()
        # _done: push has been attempted (success or failure).
        self._done = False

    def set_receive_url(self, url: str) -> None:
        """Register delivery address (triggered by register_delivery_addr)."""
        self.receive_url = url
        self._url_event.set()

    def set_data(self, data: MMResult) -> None:
        """Publish a ready payload to this delivery."""
        self.data = data
        self._data_event.set()

    @property
    def ready(self) -> bool:
        """Both url and data are ready."""
        return self._url_event.is_set() and self._data_event.is_set()

    async def wait_ready(self, timeout: float = 120.0) -> bool:
        """Wait until both url and data are ready, with timeout."""
        try:
            await asyncio.wait_for(
                asyncio.gather(self._url_event.wait(), self._data_event.wait()),
                timeout=timeout,
            )
            return True
        except asyncio.TimeoutError:
            return False

    @abc.abstractmethod
    async def push(self) -> None:
        """Actually push data to the remote endpoint. Subclass implements."""
        ...

    async def execute(self, timeout: float = 120.0) -> bool:
        """Wait for ready then push. Returns True on success.
        """
        if not await self.wait_ready(timeout):
            self._done = True
            return False
        timer = Timer(
            f"push type={self.job_type} rid={self.request_id} "
            f"rank={self.tp_rank}"
        )
        try:
            with timer.phase("push"):
                await self.push()
            if self.metrics_collector is not None:
                self.metrics_collector.observe_push(
                    f"push.{self.job_type}",
                    timer.elapsed_ms("push") / 1000.0,
                )
            return True
        except Exception:
            logger.exception("Push failed for request=%s tp_rank=%d",
                             self.request_id, self.tp_rank)
            return False
        finally:
            self._done = True

    def clear(self) -> None:
        """Cleanup after sending."""
        self.data = None
        self.receive_url = None
        self._done = False


class ZMQPushJob(MMPushJobBase):
    """ZMQ PUSH-based data push (phase 1, rvq_indices mode).

    Encoder acts as ZMQ PUSH client, connects to Prefill TP Worker's
    PULL socket, sends pickled MMResult.
    """

    def __init__(
        self,
        request_id: str,
        tp_rank: int,
        zmq_context: zmq.Context,
        metrics_collector: Optional["EncoderMetricsCollector"] = None,
    ):
        super().__init__(
            request_id, tp_rank, metrics_collector=metrics_collector,
            job_type="full",
        )
        self._zmq_context = zmq_context
        self._socket: Optional[zmq.Socket] = None

    async def push(self) -> None:
        """ZMQ PUSH send MMResult."""
        import pickle

        if self._socket is None:
            self._socket = self._zmq_context.socket(zmq.PUSH)
            # Set high HWM to avoid message loss when PULL is slow to connect
            self._socket.setsockopt(zmq.SNDHWM, 10000)
            self._socket.connect(self.receive_url)
            logger.info(
                "[ZMQPushJob] connected to %s for tp_rank=%d",
                self.receive_url, self.tp_rank,
            )

        if self.data is None:
            raise RuntimeError("ZMQPushJob has no MMResult after readiness")
        tensor = self.data.data
        if tensor is not None and tensor.device.type != "cpu":
            raise RuntimeError(
                "ZMQPushJob requires a CPU transport tensor; "
                f"got device={tensor.device} for request={self.request_id}. "
                "Combine the full result on CPU before fan-out."
            )

        payload = pickle.dumps(self.data)
        logger.info(
            "[ZMQPushJob] sending payload len=%d to %s for tp_rank=%d",
            len(payload), self.receive_url, self.tp_rank,
        )
        # Preserve the original synchronous send; success means locally queued.
        self._socket.send(payload)
        logger.info(
            "[ZMQPushJob] sent successfully to tp_rank=%d", self.tp_rank,
        )

    def clear(self) -> None:
        super().clear()
        if self._socket is not None:
            # Allow the blocking send queued above to flush to the peer before
            # closing.  linger=0 can silently drop the payload during teardown.
            self._socket.close(linger=1000)
            self._socket = None


class ZMQPushMetaJob(MMPushJobBase):
    """ZMQ PUSH-based **metadata-only** push (Mass D-side, phase 1).

    Distinct from ``ZMQPushJob`` (which pushes full ``MMResult`` with ViT
    tensor after ViT completes):

    - Trigger: ``_on_preprocess_done`` (preprocess done, ``mm_meta`` ready)
      — does NOT wait for ViT forward.
    - Payload: pickled ``MMMeta`` (< 1KB: hash/pad_value/grid_thw/num_tokens),
      no ViT tensor.
    - Consumer: Decode-side ``mm_receiver`` (expands ``origin_input_ids``
      for correct kvcache pre-allocation without ViT tensor).

    Rendezvous: ``receive_url`` (from ``register_delivery_addr(meta_only=True)``)
    + ``mm_meta`` (from ``_on_preprocess_done``).  When both ready → push.
    """

    def __init__(
        self,
        request_id: str,
        tp_rank: int,
        zmq_context: zmq.Context,
        metrics_collector: Optional["EncoderMetricsCollector"] = None,
    ):
        super().__init__(
            request_id, tp_rank, metrics_collector=metrics_collector,
            job_type="meta",
        )
        self._zmq_context = zmq_context
        self._socket: Optional[zmq.Socket] = None

    def set_meta(self, mm_meta: "MMMeta") -> None:
        """Set the MMMeta payload (replaces ``set_data`` for the meta path)."""
        self.data = mm_meta  # store in self.data for wait_ready() to trigger
        self._data_event.set()

    async def push(self) -> None:
        """ZMQ PUSH send MMMeta only (no ViT tensor)."""
        import pickle

        if self._socket is None:
            self._socket = self._zmq_context.socket(zmq.PUSH)
            self._socket.setsockopt(zmq.SNDHWM, 10000)
            self._socket.connect(self.receive_url)
            logger.info(
                "[ZMQPushMetaJob] connected to %s for tp_rank=%d",
                self.receive_url, self.tp_rank,
            )

        payload = pickle.dumps(self.data)
        logger.info(
            "[ZMQPushMetaJob] sending meta payload len=%d to %s for tp_rank=%d",
            len(payload), self.receive_url, self.tp_rank,
        )
        self._socket.send(payload)
        logger.info(
            "[ZMQPushMetaJob] sent meta to tp_rank=%d", self.tp_rank,
        )

    def clear(self) -> None:
        super().clear()
        if self._socket is not None:
            # Allow the blocking send queued above to flush to the peer before
            # closing.  linger=0 can silently drop the payload during teardown.
            self._socket.close(linger=1000)
            self._socket = None


class MooncakePushJob(MMPushJobBase):
    """Mooncake RDMA-based data push (phase 2, vit_embedding mode).

    Encoder uses mooncake engine.transfer_sync for one-sided RDMA write
    directly into Prefill's GPU buffer_address.

    NOTE: Phase 2 placeholder — not fully implemented yet.
    """

    def __init__(self, request_id: str, tp_rank: int,
                 engine=None, session_id: str = "", buffer_address: int = 0,
                 metrics_collector: Optional["EncoderMetricsCollector"] = None):
        super().__init__(
            request_id, tp_rank, metrics_collector=metrics_collector,
            job_type="full",
        )
        self.engine = engine
        self.session_id = session_id
        self.buffer_address = buffer_address

    async def push(self) -> None:
        """RDMA write MMResult to Prefill GPU buffer."""
        # Phase 2: implement with mooncake transfer_sync
        raise NotImplementedError("MooncakePushJob.push() not yet implemented (phase 2)")


# ═══════════════════════════════════════════════════════════════════════════
#  EncodeScheduler — deterministic batcher + cache + ViT execution
# ═══════════════════════════════════════════════════════════════════════════


class EncodeScheduler:
    """Encoder-side scheduler: deterministic ViT batching + cache + execution.

    Owns TP intake, preprocessing, batching and device execution.
    Publishes media completion to the request coordinator on the event loop.

    Aggregates pending items from different requests into ViT batches.
    Sorted by (request_id, item_index) lexicographic order to guarantee
    all TP ranks form the **same batch** without runtime lock/broadcast
    coordination (TokenSpeed design).

    Greedy packing: iterate in sorted order, break when exceeding
    max_tokens_per_batch or max_items_per_batch. A single oversized item
    is still returned alone to guarantee progress.

    Attributes:
        max_tokens_per_batch: token budget per ViT batch.
        max_items_per_batch: item count cap per ViT batch.
        cache: TieredEmbeddingCache (GPU L1 + Host DRAM L2).
        model: the model object (LongcatFlashVL or similar).
    """

    def __init__(
        self,
        max_tokens_per_batch: int,
        max_items_per_batch: int,
        cache: "TieredEmbeddingCache",
        preprocessor,  # EncoderPreprocessor: model-agnostic URL→item+meta
        feature_extractor,  # EncoderFeature: model-agnostic ViT forward
        attn_tp_rank: int = 0,
        attn_tp_size: int = 1,
        tp_cpu_group=None,
        metrics_collector: Optional["EncoderMetricsCollector"] = None,
        max_drain_per_step: int = 10,
        max_video_drain_per_step: int = 1,
    ):
        if max_drain_per_step <= 0:
            raise ValueError("max_drain_per_step must be greater than zero")
        if max_video_drain_per_step <= 0:
            raise ValueError("max_video_drain_per_step must be greater than zero")
        self.max_tokens_per_batch = max_tokens_per_batch
        self.max_items_per_batch = max_items_per_batch
        self.max_drain_per_step = max_drain_per_step
        self.max_video_drain_per_step = max_video_drain_per_step
        self.cache = cache
        self.preprocessor = preprocessor          # model-agnostic preprocessing
        self.feature_extractor = feature_extractor  # model-agnostic ViT feature
        self.attn_tp_rank = attn_tp_rank
        self.attn_tp_size = attn_tp_size
        self.tp_cpu_group = tp_cpu_group  # gloo CPU group for broadcast_pyobj
        self.metrics_collector = metrics_collector
        self.device_worker = EncoderDeviceWorker(
            feature_extractor, tp_rank=attn_tp_rank
        )
        # Device for CUDA/NPU image/video transform (ENABLE_DEVICE_PREPROCESS=1).
        self._device: Optional[torch.device] = None
        # Single device-preprocess state shared by CUDA and NPU implementations.
        # The concrete preprocessor dispatches by device.type. CUDA helpers keep
        # TP=1 features device-resident; TP>1 canonicalizes them for gloo below.
        self._device_preprocess_active = (
            os.environ.get("ENABLE_DEVICE_PREPROCESS", "0") == "1"
        )
        # dict keyed by (request_id, item_index): idempotent add
        self._pending: dict[tuple[str, int], PendingEncodeItem] = {}
        # (request_id, item_index) → item awaiting ViT
        self._pending_items: dict[tuple[str, int], MultimodalDataItem] = {}
        # Internal media_task_id → output_format; removed after its forward.
        self._requests: dict[str, dict] = {}
        # Lightweight queue of (request_id, media_urls, output_format) filled by
        # enqueue_request (HTTP handler).  Drained at the top of step() by
        # rank 0 and broadcast to all TP ranks
        self._request_queue: list[tuple[str, list[MediaURL], str, tuple]] = []
        self._request_queue_items = 0
        self._request_queue_enqueued_at: dict[str, float] = {}
        self._request_intake_lock = threading.Lock()
        # Unified IMAGE/VIDEO URL memo.  This is a small, process-local index
        # into TieredEmbeddingCache; it never owns embeddings or pixel tensors.
        self._url_meta_cache: OrderedDict[
            MediaURLCacheKey, MediaURLCacheEntry
        ] = OrderedDict()
        self._url_meta_cache_max = int(
            os.environ.get("ENCODER_URL_MEMO_MAX_ENTRIES", "10240")
        )
        self._url_meta_cache_ttl = float(
            os.environ.get("ENCODER_URL_MEMO_TTL_SECONDS", "3600")
        )
        self._url_meta_cache_versioned_ttl = float(
            os.environ.get("ENCODER_URL_MEMO_VERSIONED_TTL_SECONDS", "86400")
        )
        # Rank 0 sets this flag through /flush_cache.  The next step broadcasts
        # it with the normal intake payload so every TP worker clears locally.
        self._flush_cache_requested = False
        from .media_coordinator import EncoderMediaCoordinator
        self.coordinator = EncoderMediaCoordinator(self)

    def shutdown(self) -> None:
        """Stop the persistent device worker after the step loop exits."""
        self.device_worker.shutdown()

    def _cache_get(self, item_hash: int, output_format: str):
        key = _embedding_cache_key(item_hash, output_format)
        if output_format == "rvq_indices" and hasattr(self.cache, "get_host"):
            return self.cache.get_host(key)
        return self.cache.get(key)

    def _cache_put(self, item_hash: int, output_format: str, result) -> bool:
        key = _embedding_cache_key(item_hash, output_format)
        if output_format == "rvq_indices" and hasattr(self.cache, "put_host"):
            return self.cache.put_host(key, result)
        return self.cache.put(key, result)

    def _clear_local_caches(self) -> None:
        """Clear reusable Encoder caches owned by this TP worker."""
        self.cache.clear()
        self._url_meta_cache.clear()

        # Also release unused blocks held by the device allocator.
        try:
            device_module = torch.get_device_module()
            empty_cache = getattr(device_module, "empty_cache", None)
            if empty_cache is not None:
                empty_cache()
        except Exception as error:
            logger.debug("Encoder allocator empty_cache skipped: %s", error)

    def request_cache_flush(self) -> None:
        """Ask the next scheduler step to flush every TP worker's cache."""
        if self.attn_tp_rank != 0:
            raise RuntimeError("Only Encoder TP rank 0 may request a cache flush")
        self._flush_cache_requested = True

    def _process_cache_flush(self, requested: bool) -> None:
        """Clear this TP worker's cache when the broadcast flag is set."""
        if not requested:
            return

        self._clear_local_caches()
        logger.info("Encoder cache flushed on rank %d", self.attn_tp_rank)

    def _make_url_cache_key(
        self, media_url: MediaURL, output_format: str,
    ) -> Optional[MediaURLCacheKey]:
        # Keep signed/raw URLs out of cache keys and diagnostics while making
        # no unsafe assumptions about which query parameters affect content.
        identity = str(media_url.content_identity or media_url.url)
        identity_digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        preprocess_signature = self.preprocessor.get_url_cache_signature(
            media_url.modality
        )
        if not preprocess_signature:
            return None
        return MediaURLCacheKey(
            schema_version=1,
            modality=media_url.modality,
            content_identity_digest=identity_digest,
            source_version=(
                str(media_url.source_version)
                if media_url.source_version is not None else ""
            ),
            preprocess_signature=preprocess_signature,
            output_format=output_format,
        )

    def _get_url_cache_entry(
        self, key: Optional[MediaURLCacheKey],
    ) -> Optional[MediaURLCacheEntry]:
        if key is None:
            return None
        entry = self._url_meta_cache.get(key)
        if entry is None:
            return None
        if entry.expires_at <= time.monotonic():
            self._url_meta_cache.pop(key, None)
            return None
        self._url_meta_cache.move_to_end(key)
        return entry

    def _put_url_cache_entry(
        self,
        key: Optional[MediaURLCacheKey],
        modality: Modality,
        item_hash: int,
        meta: MMItemMeta,
    ) -> None:
        if key is None:
            return
        ttl = (
            self._url_meta_cache_versioned_ttl
            if key.source_version
            else self._url_meta_cache_ttl
        )
        if self._url_meta_cache_max <= 0 or ttl <= 0:
            return
        self._url_meta_cache[key] = MediaURLCacheEntry(
            modality=modality,
            item_hash=item_hash,
            mm_meta=_clone_mm_item_meta(meta, meta.item_index),
            expires_at=time.monotonic() + ttl,
        )
        self._url_meta_cache.move_to_end(key)
        while len(self._url_meta_cache) > self._url_meta_cache_max:
            self._url_meta_cache.popitem(last=False)

    # ── Batching (deterministic, TP-lockstep) ──────────────────────────────

    def add(self, item: PendingEncodeItem) -> None:
        self._pending[item.key] = item
        if self.metrics_collector is not None:
            self.metrics_collector.set_queue_length(
                "vit_pending", len(self._pending)
            )

    def pending_size(self) -> int:
        return len(self._pending)

    def _ordered_pending(self) -> list[PendingEncodeItem]:
        """Sort by (request_id, item_index) → deterministic across TP ranks."""
        return [self._pending[k] for k in sorted(self._pending.keys())]

    def next_batch(self) -> list[PendingEncodeItem]:
        """Greedy packing: take the largest prefix within budget."""
        batch: list[PendingEncodeItem] = []
        used = 0
        for it in self._ordered_pending():
            if batch and (used + it.cost > self.max_tokens_per_batch
                         or len(batch) >= self.max_items_per_batch):
                break
            batch.append(it)
            used += it.cost
            del self._pending[it.key]
        if batch and self.metrics_collector is not None:
            self.metrics_collector.set_queue_length(
                "vit_pending", len(self._pending)
            )
            self.metrics_collector.observe_batch_tokens(used)
        return batch

    # ── Request intake (HTTP handler → lightweight queue) ──────────────────

    def request_queue_stats(self) -> dict:
        """Return waiting requests, compute items and oldest wait in seconds."""
        with self._request_intake_lock:
            oldest_seconds = 0.0
            if self._request_queue:
                oldest_id = self._request_queue[0][0]
                oldest_seconds = max(
                    0.0,
                    time.monotonic() - self._request_queue_enqueued_at[oldest_id],
                )
            return {
                "requests": len(self._request_queue),
                "items": self._request_queue_items,
                "oldest_seconds": oldest_seconds,
            }

    def _pop_request_queue_prefix_locked(self):
        # Finish materialized ViT work before downloading another request batch.
        if self._pending:
            return []
        request_count = 0
        video_request_count = 0
        for _, media_urls, _, _ in self._request_queue:
            if request_count >= self.max_drain_per_step:
                break
            # Mixed requests and MaaS temporal groups retain the VIDEO modality.
            if any(media_url.is_video for media_url in media_urls):
                if video_request_count >= self.max_video_drain_per_step:
                    break
                video_request_count += 1
            request_count += 1
        drained = self._request_queue[:request_count]
        del self._request_queue[:len(drained)]
        for request_id, media_urls, _, _ in drained:
            self._request_queue_items -= len(media_urls)
            self._request_queue_enqueued_at.pop(request_id)
        return drained

    def enqueue_request(
        self,
        request_id: str,
        media_urls: list[MediaURL],
        output_format: str = "rvq_indices",
        media_indices: Optional[list[int]] = None,
    ) -> None:
        """Bind a descriptor and eagerly enqueue media encoding.

        Retries are validated while the descriptor is resident. Downloads,
        cache resolution and TP synchronization run in ``step``; cold media
        execute ViT without waiting for meta/full registration.
        """
        indices = tuple(range(len(media_urls))) if media_indices is None else tuple(media_indices)
        self.coordinator.enqueue(request_id, media_urls, output_format, indices)

    # ── Step-boundary TP collective (drain + broadcast URLs) ───────────────

    def _drain_and_broadcast(
        self,
        flush_cache_requested: bool = False,
    ) -> tuple[list[tuple[str, list[MediaURL], str, tuple]], bool]:
        """Broadcast queued encode requests and cache-flush controls.

        Called at the top of ``step()`` on **all** ranks (wrapped in
        ``to_thread`` since ``broadcast_pyobj`` is a blocking gloo collective).
        Rank 0 drains a bounded FIFO prefix and broadcasts one list.  Flush
        controls use a tagged string so the idle path keeps the existing
        empty-list fast path and does not add another TP collective.

        Returns:
            ``(encode_requests, flush_cache_requested)``; both values are
            identical on all TP ranks.
        """
        if self.attn_tp_size <= 1 or self.tp_cpu_group is None:
            # TP=1: no broadcast needed; rank 0 (the only rank) drains locally.
            with self._request_intake_lock:
                drained = self._pop_request_queue_prefix_locked()
            return drained, flush_cache_requested

        from sglang.srt.utils import broadcast_pyobj

        if self.attn_tp_rank == 0:
            with self._request_intake_lock:
                payload = self._pop_request_queue_prefix_locked()
            if flush_cache_requested:
                payload.append(_FLUSH_CACHE_CONTROL_TAG)
            broadcast_pyobj(payload, rank=0, dist_group=self.tp_cpu_group, src=0)
        else:
            payload = broadcast_pyobj(
                None, rank=self.attn_tp_rank,
                dist_group=self.tp_cpu_group, src=0,
            )

        encode_requests = []
        flush_cache_requested = False
        for item in payload or []:
            if item == _FLUSH_CACHE_CONTROL_TAG:
                flush_cache_requested = True
            else:
                encode_requests.append(item)
        return encode_requests, flush_cache_requested

    # ── Preprocessing (URL → pixel_values → cache lookup → enqueue) ────────


    def _resolve_preprocess_device(self) -> None:
        """Validate the preprocessor and lazily resolve its accelerator."""
        if self.preprocessor is None:
            raise RuntimeError("EncodeScheduler preprocessor is not initialized")
        if self._device_preprocess_active and self._device is None:
            try:
                self._device = next(self.feature_extractor.parameters()).device
            except Exception as e:
                logger.warning(
                    "Device preprocessing enabled but failed to resolve device from "
                    "feature_extractor parameters; falling back to CPU preprocessing: %s",
                    e,
                )
                self._device_preprocess_active = False
            else:
                if self._device.type not in ("cuda", "npu"):
                    logger.warning(
                        "Device preprocessing requested on unsupported device %s; "
                        "falling back to CPU preprocessing",
                        self._device,
                    )
                    self._device_preprocess_active = False


    def _preprocess_media(
        self, request_id: str, media_url: MediaURL, item_index: int,
    ) -> PreprocessedItem:
        """Transform one image/video URL and record preprocessing metrics."""
        modality = media_url.modality
        if modality not in (Modality.IMAGE, Modality.VIDEO):
            raise NotImplementedError(
                f"Encoder _preprocess: modality {modality.name} "
                f"not yet supported (url={str(media_url.url)[:80]})"
            )
        timer = Timer(
            f"preprocess {modality.name.lower()} rid={request_id} idx={item_index}"
        )
        status = "success"
        backend = "device" if self._device_preprocess_active else "cpu"
        try:
            if modality is Modality.IMAGE:
                items = [self.preprocessor.process_image(
                    url=media_url.url, item_index=item_index,
                    device_preprocess=self._device_preprocess_active,
                    device=self._device, timer=timer,
                )]
            else:
                items = self.preprocessor.process_video(
                    url=media_url.url, item_index=item_index,
                    device_preprocess=self._device_preprocess_active,
                    device=self._device, timer=timer,
                )
            if len(items) != 1:
                raise ValueError(
                    "one media URL must produce exactly one Encoder "
                    f"item; got {len(items)} for index {item_index}"
                )
            return items[0]
        except Exception:
            status = "error"
            raise
        finally:
            timer.mark_total()
            if self.metrics_collector is not None:
                self.metrics_collector.observe_preprocess(
                    modality=modality.name, status=status,
                    seconds=timer.elapsed_ms("total") / 1000.0, backend=backend,
                )
                for stage in ("download", "decode", "transform", "hash"):
                    if timer.has_phase(stage):
                        self.metrics_collector.observe_preprocess_stage(
                            modality=modality.name, stage=stage, status=status,
                            seconds=timer.elapsed_ms(stage) / 1000.0,
                            backend=backend,
                        )
            logger.info(
                "Encoder preprocessor: rid=%s idx=%d modality=%s status=%s %s",
                request_id, item_index, modality.name, status, timer.format(),
            )

    def _enqueue_preprocess_misses(
        self, misses: list[tuple[str, str, int, int, MultimodalDataItem]],
    ) -> None:
        """Commit cache misses idempotently to the local ViT pending queue."""
        for request_id, _, item_index, cost, item in misses:
            key = (request_id, item_index)
            if key not in self._pending:
                self.add(PendingEncodeItem(
                    request_id=request_id, item_index=item_index, cost=cost,
                ))
                self._pending_items[key] = item

    def _sync_preprocess_misses(
        self, misses: list[tuple[str, str, int, int, MultimodalDataItem]],
    ) -> None:
        """Broadcast rank-0 misses and restore matching pending state on peers."""
        if self.attn_tp_size <= 1:
            return
        # Keep metadata small; send dense features through the CPU TP group.
        from .tp_feature_transport import broadcast_features

        if self.attn_tp_rank == 0:
            broadcast_features(misses, rank=0, group=self.tp_cpu_group)
            return

        received = broadcast_features(
            None, rank=self.attn_tp_rank, group=self.tp_cpu_group,
        )
        for entry in received:
            request_id, output_format, _, _, _ = entry
            # Keep CPU inputs until DeviceWorker moves them to this rank's device.
            # This thread pool has no rank-specific default accelerator device.
            self._enqueue_preprocess_misses([entry])
            # Peers need pending items and output format, not delivery metadata.
            self._requests.setdefault(request_id, {
                "output_format": output_format,
                "item_metas": [],
                "full_items": [],
                "has_miss": True,
            })

    async def step(self):
        s = self
        flush = s._flush_cache_requested
        s._flush_cache_requested = False
        requests, flush = await asyncio.to_thread(s._drain_and_broadcast, flush)
        s._process_cache_flush(flush)
        if requests:
            await self.coordinator.prepare(requests)
        batch = s.next_batch()
        groups = defaultdict(list)

        for pending in batch:
            item = s._pending_items.pop(pending.key)
            fmt = s._requests[pending.request_id]["output_format"]
            groups[(fmt, item.modality)].append((pending.request_id, item))

        for (fmt, modality), entries in groups.items():
            started = time.perf_counter()
            status = "error"
            forward_seconds = None
            try:
                from sglang.srt.managers.scheduling.multimodal_data import Modality

                items = [item for _, item in entries]
                counts = [s.feature_extractor.item_token_count(item) for item in items]
                feat_fn = (
                    s.feature_extractor.get_video_feature
                    if modality is Modality.VIDEO
                    else s.feature_extractor.get_image_feature
                )
                output = await s.device_worker.run_feature(
                    feat_fn,
                    items,
                    fmt,
                    expected_tokens=sum(counts),
                    request_id=entries[0][0],
                    modality=modality.name,
                )
                forward_seconds = time.perf_counter() - started
                if output.shape[0] != sum(counts):
                    raise RuntimeError("Encoder media output token count mismatch")
                for (task_id, _), embed in zip(entries, torch.split(output, counts)):
                    if s.attn_tp_rank == 0:
                        # A slice must not retain the entire batch storage in a cache.
                        self.coordinator.complete_media(task_id, embed.clone())
                self.coordinator.counters["vit_items"] += len(entries)
                status = "success"
            except Exception as error:
                # A peer may still be inside the model's collective. Continuing
                # with another media task after a TP forward failure is unsafe.
                if s.attn_tp_size > 1:
                    raise
                for task_id, _ in entries:
                    if s.attn_tp_rank == 0:
                        self.coordinator.complete_media(task_id, error=error)
                logger.exception("Encoder media execution failed")
            finally:
                elapsed = (
                    forward_seconds
                    if forward_seconds is not None
                    else time.perf_counter() - started
                )
                if s.metrics_collector is not None:
                    s.metrics_collector.observe_vit(modality.name, fmt, status, elapsed)
                logger.info(
                    "Encoder ViT: media_task=%s modality=%s output_format=%s status=%s items=%d vit_forward=%.1fms",
                    entries[0][0],
                    modality.name,
                    fmt,
                    status,
                    len(entries),
                    elapsed * 1000,
                )
                for task_id, _ in entries:
                    s._requests.pop(task_id, None)
        self.coordinator.log_snapshot()
        return len(batch)

    @staticmethod
    def _item_token_count(item: MultimodalDataItem) -> int:
        """Compute patch token count for an item (fallback).

        Prefer ``feature_extractor.item_token_count``; this static helper
        is kept for any call sites that don't have a feature_extractor ref.
        """
        grid_thw = item.image_grid_thw
        t, h, w = grid_thw[0].tolist()
        # merge_size=2 default for longcat
        return int(t * (h // 2) * (w // 2))
