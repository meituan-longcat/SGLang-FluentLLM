"""Eager media encoding with bounded request rendezvous and shared computation.

encode_async enqueues cache resolution and ViT execution without waiting for
any delivery registration. Ready results are reused; in-flight misses share
one media task. Full registration only creates a per-request delivery work.

All state belongs to the serving event loop. Only preprocessing, TP transport
and device execution are offloaded; their results are published on that loop.
Idle descriptors contain no tensor, work, callback or asyncio task.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from enum import Enum

import torch

from sglang.srt.utils import get_colorful_logger

from .encode_data import MMMeta, MMResult

logger = get_colorful_logger(__name__)


class EncoderCapacityError(RuntimeError):
    pass


class DeliveryStatus(str, Enum):
    WAITING = "waiting"
    SENDING = "sending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self):
        return self in (self.SUCCEEDED, self.FAILED, self.CANCELLED)


@dataclass
class Delivery:
    receive_url: str
    timeout: float
    status: DeliveryStatus = DeliveryStatus.WAITING
    job: object = None
    task: asyncio.Task | None = None


@dataclass
class FullResultContext:
    parts: dict = field(default_factory=dict)  # Media index -> per-occurrence result.
    combined: MMResult | None = None  # Assembled payload shared by full ranks.


@dataclass
class RequestState:
    """Join encode and delivery calls by rid; idle state retains no tensors."""

    rid: str
    media: tuple = ()
    indices: tuple = ()  # Original media indices, aligned with media.
    output_format: str = "rvq_indices"
    fingerprint: str | None = None  # Reject conflicting inputs for the same rid.
    metas: dict = field(default_factory=dict)  # Media index -> metadata.
    waiting: set = field(default_factory=set)  # Shared media task IDs.
    preparing: bool = False  # Queued or preprocessing; ViT is tracked by waiting.
    full_size: int | None = None  # Expected full-delivery ranks on the receiver.
    # (meta_only, rank) -> Delivery; terminal receipts prevent duplicate sends.
    deliveries: dict = field(default_factory=dict)
    full: FullResultContext | None = None  # Result ownership for full consumers only.
    deadline: asyncio.Task | None = None  # Delivery timeout, started on registration.
    closing: bool = False  # Prevent eviction while cancelled senders drain.
    error: str | None = None


@dataclass
class MediaTask:
    """Shared ViT computation started by encode, independent of delivery jobs.

    Input tensors live in the scheduler's pending items/current batch; this
    record tracks their content identity and request subscribers.
    """

    task_id: str
    content_key: tuple
    meta: object
    subscribers: set = field(default_factory=set)  # (rid, original media index).
    aliases: set = field(default_factory=set)  # Source keys pointing to this task.


class EncoderMediaCoordinator:
    """Own request and shared-media lifecycles on the serving event loop.

    Encode starts computation through the scheduler; registration controls delivery.
    """

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.states = {}  # rid -> RequestState.
        self.idle = OrderedDict()  # Evictable descriptor LRU, oldest first.
        # In-flight indexes only; completed results live in scheduler caches.
        self.media_tasks = {}  # Task ID -> MediaTask, shared across requests.
        self.by_source = {}  # URL cache key -> task ID, before preprocessing.
        self.by_content = {}  # (Content hash, output format) -> task ID.
        self.max_requests = self._limit("ENCODER_REQUEST_MAX_ENTRIES", 10240)
        self.max_media = self._limit("ENCODER_MEDIA_MAX_ITEMS", 512)
        self._closed = False
        self._serial = 0
        self._last_stats = 0.0
        self.counters = defaultdict(int)

    @staticmethod
    def _limit(name, default):
        value = int(os.environ.get(name, default))
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        return value

    @staticmethod
    def _full_pending(state):
        # Include expected receiver ranks that have not registered yet.
        return (
            state.full_size is not None
            and sum(
                not key[0] and delivery.status.terminal
                for key, delivery in state.deliveries.items()
            )
            < state.full_size
        )

    def _active(self, state):
        """Check whether unfinished work or cleanup prevents request eviction."""
        return bool(
            state.closing
            or state.preparing
            or state.waiting
            or any(not d.status.terminal for d in state.deliveries.values())
            or self._full_pending(state)
        )

    def _touch(self, state):
        """Update idle LRU eligibility and recency without releasing resources."""
        # Ignore callbacks for an old state after its rid has been reused.
        if self.states.get(state.rid) is not state:
            return

        self.idle.pop(state.rid, None)
        if not self._active(state):
            assert state.full is None and state.deadline is None
            assert all(
                d.job is None and d.task is None for d in state.deliveries.values()
            )
            self.idle[state.rid] = None

    def _reserve_request_slot(self):
        """Make room for a new rid by evicting idle states; reject if all are active."""
        while len(self.states) >= self.max_requests:
            if not self.idle:
                raise EncoderCapacityError("Encoder request capacity exhausted")
            victim, _ = self.idle.popitem(last=False)
            self.states.pop(victim)
            self.counters["descriptor_evictions"] += 1

    def _get(self, rid):
        if not rid or not isinstance(rid, str):
            raise ValueError("request_id must be a nonempty string")

        state = self.states.get(rid)
        if state is None:
            self._reserve_request_slot()
            state = RequestState(rid)
            self.states[rid] = state
        self._touch(state)
        return state

    def enqueue(self, rid, media, output_format, indices):
        """Schedule encoding now; consumers may register before or after it.

        The next scheduler step resolves ready/in-flight hits and batches cold
        media for ViT even if no meta or full consumer ever registers.
        """
        if self._closed:
            raise EncoderCapacityError("Encoder is shutting down")
        if output_format not in ("rvq_indices", "vit_embedding"):
            raise ValueError("unsupported Encoder output_format")

        if not media or len(media) != len(indices) or len(media) > self.max_media:
            raise ValueError("invalid or oversized Encoder media list")

        if any(type(i) is not int or i < 0 for i in indices) or len(
            set(indices)
        ) != len(indices):
            raise ValueError("media_indices must be unique nonnegative integers")

        payload = repr((tuple(media), output_format, tuple(indices))).encode()
        fingerprint = hashlib.sha256(payload).hexdigest()
        state = self._get(rid)
        if state.fingerprint is not None and state.fingerprint != fingerprint:
            raise ValueError("same request_id has conflicting Encoder input")

        if state.fingerprint is None:
            state.media, state.indices = tuple(media), tuple(indices)
            state.output_format, state.fingerprint = output_format, fingerprint

        if state.error is None and not state.preparing:
            self._schedule(state)
        self.counters["encode"] += 1

    def _schedule(self, state):
        """Queue preparation; the scheduler batches and executes ViT later."""
        if state.preparing or state.error is not None:
            return
        state.preparing = True
        self.idle.pop(state.rid, None)
        s = self.scheduler
        with s._request_intake_lock:
            s._request_queue.append(
                (state.rid, list(state.media), state.output_format, state.indices)
            )
            s._request_queue_items += len(state.media)
            s._request_queue_enqueued_at[state.rid] = time.monotonic()

    def _enqueue_media_miss(self, pp, output_format, misses):
        """Admit a cold media computation without a request-level full work.

        Called after both result-cache and in-flight lookups miss. Registration
        is deliberately absent: encode alone must prepare the result cache.
        """
        from .scheduler import _clone_mm_item_meta

        self._serial += 1
        task_id = f"__encoder_media_{self._serial:020d}"
        content = (pp.item.hash, output_format)
        task = MediaTask(task_id, content, _clone_mm_item_meta(pp.meta, 0))
        self.media_tasks[task_id] = task
        self.by_content[content] = task_id
        self.scheduler._requests[task_id] = {"output_format": output_format}
        misses.append((task_id, output_format, 0, pp.cost, pp.item))
        self.counters["cold"] += 1
        return task

    async def prepare(self, requests):
        """Resolve shared results and eagerly queue every uncached media item."""
        s = self.scheduler
        misses = []
        if s.attn_tp_rank == 0:
            s._resolve_preprocess_device()
            for rid, _, _, _ in requests:
                state = self.states.get(rid)
                if state is None:
                    continue
                if state.error is not None:
                    # A delivery can expire while this encode waits behind
                    # another forward. Drain its intake ownership as well.
                    state.preparing = False
                    self._touch(state)
                    continue
                try:
                    for mu, index in zip(state.media, state.indices):
                        source = s._make_url_cache_key(mu, state.output_format)
                        task_id = (
                            self.by_source.get(source) if source is not None else None
                        )
                        task = self.media_tasks.get(task_id)
                        memo = s._get_url_cache_entry(source)
                        if task is not None:
                            meta = task.meta
                            cached = None
                        elif memo is not None:
                            meta = memo.mm_meta
                            cached = s._cache_get(memo.item_hash, state.output_format)
                            # Meta remains useful even if the result was evicted.
                            self._bind_meta(state, index, meta)
                            self._advance_request(state)
                        else:
                            cached = None
                            meta = None
                        if task is None and cached is None:
                            # A full consumer can already own this result after LRU eviction.
                            full = state.full
                            if full is not None and index in full.parts:
                                continue
                            if len(self.media_tasks) >= self.max_media:
                                raise EncoderCapacityError(
                                    "Encoder media input capacity exhausted"
                                )
                            pp = await asyncio.to_thread(
                                s._preprocess_media, rid, mu, index
                            )
                            if state.error is not None or self._closed:
                                del pp
                                break
                            meta = pp.meta
                            cached = s._cache_get(pp.item.hash, state.output_format)
                            # Different sources may resolve to the same media content.
                            content = (pp.item.hash, state.output_format)
                            task = self.media_tasks.get(self.by_content.get(content))
                            if task is None and cached is None:
                                task = self._enqueue_media_miss(
                                    pp, state.output_format, misses
                                )
                            self.counters["preprocess"] += 1
                            s._put_url_cache_entry(
                                source, mu.modality, pp.item.hash, meta
                            )
                            # Do not keep a duplicate preprocessing result across await/idle.
                            del pp
                        self._bind_meta(state, index, meta)
                        if s.metrics_collector is not None:
                            s.metrics_collector.record_cache_lookup(
                                hit=cached is not None
                            )
                        if task is not None:
                            if source is not None:
                                self.by_source[source] = task.task_id
                                task.aliases.add(source)
                            # Track both task consumers and the request's dependencies.
                            task.subscribers.add((rid, index))
                            state.waiting.add(task.task_id)
                            self.counters["inflight_join"] += 1
                        elif cached is not None:
                            self._capture(state, index, cached.embedding)
                            self.counters["ready_hit"] += 1
                        self._advance_request(state)
                except Exception as error:
                    self._fail(state, error)
                finally:
                    state.preparing = False
                    self._advance_request(state)
                    self._touch(state)
            s._enqueue_preprocess_misses(misses)
        # All ranks receive the same media-task IDs, including when all are hits.
        await asyncio.to_thread(s._sync_preprocess_misses, misses)

    def _bind_meta(self, state, index, meta):
        from .scheduler import _clone_mm_item_meta

        previous = state.metas.get(index)
        if previous is not None and (
            previous.hash != meta.hash or previous.num_tokens != meta.num_tokens
        ):
            raise ValueError("media changed after Encoder metadata was published")
        state.metas[index] = _clone_mm_item_meta(meta, index)

    def _capture(self, state, index, tensor):
        """Retain a per-index result only for registered full consumers."""
        full = state.full

        # 已注册 full 消费者、尚未组装完整结果、当前media index 尚未收集、请求未失败才填充结果
        if (
            full is None
            or full.combined is not None
            or index in full.parts
            or state.error is not None
        ):
            return
        full.parts[index] = MMResult(
            request_id=state.rid,
            item_index=index,
            num_tokens=tensor.shape[0],
            **{state.output_format: tensor},
        )

    @staticmethod
    def _metadata_ready(state):
        return bool(state.fingerprint) and len(state.metas) == len(state.indices)

    @staticmethod
    def _metadata(state, pending):
        from .scheduler import _clone_mm_item_meta

        return MMMeta(
            state.rid,
            [_clone_mm_item_meta(state.metas[i], i) for i in sorted(state.indices)],
            pending,
        )

    def _publish_metadata_if_ready(self, state):
        """Prepare the meta payload once metadata is ready, without waiting for ViT."""
        delivery = state.deliveries.get((True, 0))

        # 已注册 meta 消费者、投递处于 WAITING、尚未设置数据、所有媒体的 metadata 已齐全 才会准备投递
        if (
            delivery is None
            or delivery.status is not DeliveryStatus.WAITING
            or delivery.job._data_event.is_set()
            or not self._metadata_ready(state)
        ):
            return
        # Cache misses also mark pending, even when no ViT task is running.
        pending = bool(state.waiting) or any(
            self.scheduler._cache_get(state.metas[i].hash, state.output_format) is None
            for i in state.indices
        )
        delivery.job.set_data(self._metadata(state, pending))

    def _advance_full_result(self, state):
        """Collect and assemble results, then fill waiting full deliveries."""
        full = state.full
        if full is None or not self._metadata_ready(state):
            return
        if full.combined is None:
            for index in state.indices:
                if index not in full.parts:
                    cached = self.scheduler._cache_get(
                        state.metas[index].hash, state.output_format
                    )
                    if cached is not None:
                        self._capture(state, index, cached.embedding)
            if len(full.parts) != len(state.indices):
                if not state.preparing and not state.waiting:
                    # Rebuild missing results after shared-cache eviction.
                    self._schedule(state)
                return
            # Assemble by media index; retain parts until the copy succeeds.
            data = torch.cat([full.parts[i].data for i in sorted(state.indices)]).to(
                "cpu"
            )
            full.combined = MMResult(
                request_id=state.rid,
                item_index=-1,
                num_tokens=data.shape[0],
                mm_meta=self._metadata(state, False),
                **{state.output_format: data},
            )
            full.parts.clear()
        # Full ranks share the assembled result, including late registrations.
        for (meta_only, _), delivery in state.deliveries.items():
            if (
                meta_only
                or delivery.status is not DeliveryStatus.WAITING
                or delivery.job._data_event.is_set()
            ):
                continue
            delivery.job.set_data(full.combined)

    def _start_ready_deliveries(self, state):
        for delivery in state.deliveries.values():
            if delivery.status is DeliveryStatus.WAITING and delivery.job.ready:
                delivery.status = DeliveryStatus.SENDING
                delivery.task = asyncio.create_task(self._send(state, delivery))

    def _advance_request(self, state):
        """Advance ready meta/full deliveries, then clean up completed stages."""
        if self.states.get(state.rid) is not state or state.closing:
            return
        if state.error is not None:
            self._fail(state, state.error)
            return
        try:
            self._publish_metadata_if_ready(state)
            self._advance_full_result(state)
        except Exception as error:
            self._fail(state, error)
            return
        self._start_ready_deliveries(state)
        self._settle(state)

    def _unsubscribe(self, state):
        """Detach this request while allowing shared ViT computation to finish."""
        for task_id in state.waiting:
            task = self.media_tasks.get(task_id)
            if task is not None:
                task.subscribers = {
                    key for key in task.subscribers if key[0] != state.rid
                }
        state.waiting.clear()

    def _fail(self, state, error):
        """Publish errors without re-entering result allocation or publication."""
        first_error = state.error is None
        state.error = str(error)[:2048]
        self._unsubscribe(state)
        result = MMResult(state.rid, -1, error=state.error)
        for delivery in state.deliveries.values():
            if delivery.status is DeliveryStatus.WAITING:
                # Replace any prepared payload before starting its send.
                delivery.job.set_data(result)
        if not any(
            not key[0] and d.status is DeliveryStatus.SENDING
            for key, d in state.deliveries.items()
        ):
            self._release_full_data(state)
        self._start_ready_deliveries(state)
        self._settle(state)
        if first_error:
            logger.warning(
                "Encoder request failed: rid=%s error=%s", state.rid, state.error
            )

    def _release_full_data(self, state):
        full = state.full
        if full is not None:
            full.parts.clear()
            full.combined = None
            state.full = None

    def _settle(self, state):
        """Release finished stages, then admit only lightweight state to idle."""
        if self.states.get(state.rid) is not state or state.closing:
            return
        if not self._full_pending(state):
            self._release_full_data(state)
            if all(d.status.terminal for d in state.deliveries.values()):
                if state.deadline is not None:
                    state.deadline.cancel()
                    state.deadline = None
        self._touch(state)

    async def register(self, addr, context, timeout):
        """Bind a consumer by rid; encode may arrive before or after registration."""
        from .scheduler import ZMQPushJob, ZMQPushMetaJob

        if self._closed:
            raise EncoderCapacityError("Encoder is shutting down")
        if not addr.receive_url:
            raise ValueError("receive_url must not be empty")
        if addr.meta_only:
            if addr.tp_rank != 0:
                raise ValueError("meta-only delivery requires tp_rank=0")
        elif not 0 < addr.tp_size <= 1024 or not 0 <= addr.tp_rank < addr.tp_size:
            raise ValueError("invalid full delivery topology")
        state = self._get(addr.request_id)
        key = (bool(addr.meta_only), addr.tp_rank)
        if not addr.meta_only and state.full_size not in (None, addr.tp_size):
            raise ValueError("inconsistent full tp_size")
        previous = state.deliveries.get(key)
        if previous is not None:
            if previous.receive_url != addr.receive_url:
                raise ValueError("conflicting delivery address for same request/rank")
            return True
        if state.closing:
            return False
        if not addr.meta_only:
            state.full_size = addr.tp_size
            if state.full is None and state.error is None:
                state.full = FullResultContext()
        job_type = ZMQPushMetaJob if addr.meta_only else ZMQPushJob
        job = job_type(
            state.rid, addr.tp_rank, context, self.scheduler.metrics_collector
        )
        job.set_receive_url(addr.receive_url)
        state.deliveries[key] = Delivery(addr.receive_url, timeout, job=job)
        self.idle.pop(state.rid, None)
        if state.deadline is None:
            # Additional ranks share the original deadline without extending it.
            state.deadline = asyncio.create_task(self._expire(state, timeout))
        self._advance_request(state)
        return True

    async def _send(self, state, delivery):
        job = delivery.job
        failure = None
        try:
            success = await job.execute(delivery.timeout)
            delivery.status = (
                DeliveryStatus.SUCCEEDED if success else DeliveryStatus.FAILED
            )
            if not success:
                failure = "Encoder delivery failed or timed out"
        except asyncio.CancelledError:
            delivery.status = DeliveryStatus.CANCELLED
            raise
        except Exception as error:
            delivery.status = DeliveryStatus.FAILED
            failure = error
        finally:
            try:
                # Keep the delivery receipt, but release its payload and transport.
                job.clear()
            finally:
                delivery.job = None
                delivery.task = None
                if failure is not None and not state.closing:
                    self._fail(state, failure)
                else:
                    self._settle(state)

    async def _close_request(self, state, error):
        """Pin the old state until cancelled senders release their references."""
        if self.states.get(state.rid) is not state:
            return
        if state.closing:
            if (
                state.deadline is not None
                and state.deadline is not asyncio.current_task()
            ):
                await asyncio.shield(state.deadline)
            return
        state.closing = True
        self.idle.pop(state.rid, None)
        state.error = str(error)[:2048]
        self._unsubscribe(state)
        deadline = state.deadline
        if deadline is not None and deadline is not asyncio.current_task():
            deadline.cancel()
            state.deadline = None
        tasks = []
        for delivery in state.deliveries.values():
            if delivery.task is not None:
                delivery.task.cancel()
                tasks.append(delivery.task)
            elif not delivery.status.terminal:
                delivery.status = DeliveryStatus.CANCELLED
                delivery.job.clear()
                delivery.job = None
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        # Also handle tasks cancelled before their first instruction: their
        # coroutine's finally block did not run, so ownership is cleared here.
        for delivery in state.deliveries.values():
            if delivery.job is not None:
                delivery.job.clear()
                delivery.job = None
            delivery.task = None
            if not delivery.status.terminal:
                delivery.status = DeliveryStatus.CANCELLED
        self._release_full_data(state)
        state.full_size = None
        state.deadline = None
        state.closing = False
        self._touch(state)

    async def _expire(self, state, timeout):
        try:
            await asyncio.sleep(timeout)
            await self._close_request(state, "Encoder delivery deadline exceeded")
        finally:
            if state.deadline is asyncio.current_task():
                state.deadline = None

    def complete_media(self, task_id, embedding=None, error=None):
        """Cache one media result and notify its dependent requests."""
        task = self.media_tasks.get(task_id)
        if task is None:
            return
        from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult

        if error is None:
            self.scheduler._cache_put(
                task.content_key[0],
                task.content_key[1],
                EmbeddingResult(embedding=embedding, mm_meta=task.meta),
            )
        # Publish cache before removing the in-flight identity.
        for alias in task.aliases:
            self.by_source.pop(alias, None)
        self.by_content.pop(task.content_key, None)
        self.media_tasks.pop(task_id)
        affected = {}  # rid -> state; one request may subscribe at several indices.
        for rid, index in task.subscribers:
            state = self.states.get(rid)
            if state is None:
                continue
            affected[rid] = state
            state.waiting.discard(task_id)
            try:
                if error is not None:
                    self._fail(state, error)
                else:
                    self._capture(state, index, embedding)
            except Exception as failure:
                self._fail(state, failure)
        # Capture all occurrences before advancing each request once.
        for state in affected.values():
            self._advance_request(state)
            self._touch(state)

    def log_snapshot(self):
        now = time.monotonic()
        if self.scheduler.attn_tp_rank == 0 and now - self._last_stats >= 10:
            self._last_stats = now
            logger.info("Encoder lifecycle: %s", self.snapshot())

    async def shutdown(self):
        """Close deliveries before the server destroys its ZMQ context."""
        self._closed = True
        for state in list(self.states.values()):
            await self._close_request(state, "Encoder is shutting down")
        self.states.clear()
        self.idle.clear()
        self.media_tasks.clear()
        self.by_source.clear()
        self.by_content.clear()
        self.scheduler._pending.clear()
        self.scheduler._pending_items.clear()
        self.scheduler._requests.clear()
        with self.scheduler._request_intake_lock:
            self.scheduler._request_queue.clear()
            self.scheduler._request_queue_items = 0
            self.scheduler._request_queue_enqueued_at.clear()

    def snapshot(self):
        return dict(
            requests=len(self.states),
            idle=len(self.idle),
            media_tasks=len(self.media_tasks),
            full_work=sum(s.full is not None for s in self.states.values()),
            meta_jobs=sum(
                key[0] and not d.status.terminal
                for s in self.states.values()
                for key, d in s.deliveries.items()
            ),
            **self.counters,
        )
