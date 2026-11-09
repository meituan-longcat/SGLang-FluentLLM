"""Encoder Server for EPD (Encode-Prefill-Decode) disaggregation.

The Encoder server runs the vision tower only (no LM/bridge) and serves
``encode_async`` / ``register_delivery_addr`` / ``health`` over **two**
pluggable transports that share the same core logic:

  - **HTTP** (default, always available) — JSON over FastAPI + uvicorn,
    used by the Prefill Scheduler's ``encoder_disagg_mixin`` and the
    TokenizerManager's ``_epd_encode_async``.
  - **gRPC** (optional) — enabled only when the generated proto stubs are
    importable; useful for cross-language / protobuf-typed callers.

Architecture:

    ┌──────────────────────────────────────────────────────────┐
    │                      EncoderServer                       │
    │   ┌────────────┐   ┌────────────┐   ┌────────────────┐   │
    │   │ HTTPServer │   │ gRPCServer │   │  EncodeScheduler│  │
    │   │ (FastAPI)  │   │  (optional)│   │  step loop task │  │
    │   └─────┬──────┘   └─────┬──────┘   └────────┬───────┘   │
    │         │                │                   │           │
    │         ▼                ▼                   │           │
    │   ┌─────────────────────────────┐            │           │
    │   │      EncoderBackend         │◄───────────┘           │
    │   │  (transport-agnostic core)  │                        │
    │   │  • encode_async             │                        │
    │   │  • register_delivery_addr   │                        │
    │   │  • flush_cache              │                        │
    │   │  • health                   │                        │
    │   └─────────────────────────────┘                        │
    └──────────────────────────────────────────────────────────┘

The HTTP server, gRPC server, and EncodeScheduler step loop all share a
single asyncio event loop.  Blocking operations (image preprocessing,
gloo ``broadcast_pyobj``, ViT forward) are offloaded via
a persistent single-thread Device Worker so the loop stays responsive.

Startup flow:
1. Load the model in encoder_only mode (vision_tower only).
2. Initialize TieredEmbeddingCache (GPU L1 + Host DRAM L2).
3. Initialize EncodeScheduler (cache + ViT exec + batching).
4. Wrap them in ``EncoderBackend`` (shared core logic).
5. Start FastAPI HTTP server (always) + gRPC server (if stubs available).
6. Run EncodeScheduler step loop as an asyncio task.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import TYPE_CHECKING, Optional

import zmq

from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_colorful_logger

from .encode_data import (
    DeliveryAddr,
    MediaURL,
)
from .media_coordinator import EncoderCapacityError
from .scheduler import (
    EncodeScheduler,
)

if TYPE_CHECKING:
    from fastapi import FastAPI
    from sglang.srt.observability.metrics.collector import EncoderMetricsCollector

logger = get_colorful_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  EncoderBackend — transport-agnostic core logic
# ═══════════════════════════════════════════════════════════════════════════


class EncoderBackend:
    """Transport-agnostic core logic shared by HTTP and gRPC servers.

    Holds the ``EncodeScheduler`` and its media coordinator and exposes four
    transport-agnostic operations:

      - ``encode_async(request_id, media_urls, output_format) -> bool``
      - ``register_delivery_addr(addr: DeliveryAddr) -> bool``
      - ``flush_cache() -> dict``
      - ``health() -> bool``

    All methods are coroutines and operate on plain Python dataclasses
(``DeliveryAddr``), so neither FastAPI/uvicorn nor grpc/proto
types leak into this class.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        scheduler: EncodeScheduler,
        zmq_context: zmq.Context,
        attn_tp_rank: int = 0,
        attn_tp_size: int = 1,
        metrics_collector: Optional["EncoderMetricsCollector"] = None,
    ):
        self.server_args = server_args
        self.scheduler = scheduler
        self._zmq_context = zmq_context
        self.attn_tp_rank = attn_tp_rank
        self.attn_tp_size = attn_tp_size
        self.metrics_collector = metrics_collector
        self._step_task: Optional[asyncio.Task] = None
        self._last_step_success_at: Optional[float] = None
        self._last_step_error: Optional[str] = None
        self._consecutive_step_errors = 0
        self._health_step_timeout = float(
            os.environ.get("ENCODER_HEALTH_STEP_TIMEOUT_SECONDS", "300")
        )

    async def encode_async(
        self,
        request_id: str,
        media_urls: list[MediaURL],
        output_format: str = "rvq_indices",
        media_indices: Optional[list[int]] = None,
    ) -> bool:
        """Trigger media encoding asynchronously, independently of registration.

        Bind the request and enqueue it for the continuously running ``step``
        loop. Cache hits reuse results, in-flight media share computation, and
        cold media execute ViT in an available batch without waiting for any
        ``register_delivery_addr`` call. Completion populates the bounded
        media cache even when no consumer has registered.

        Return immediately after enqueueing, not after ViT completes. Metadata
        may be pushed to registered meta consumers before ViT finishes; full
        consumers receive the result and metadata once both are available.

        Args:
            request_id: unique request identifier.
            media_urls: list of ``MediaURL`` (modality-tagged URLs) to encode.
            output_format: "rvq_indices" or "vit_embedding".
            media_indices: global indices of this batch's media in the
                request's ordered media list (for multi-encoder sharding).

        Returns:
            True if the request was accepted into the queue.
        """
        status = "success"
        try:
            self.scheduler.enqueue_request(
                request_id,
                media_urls,
                output_format,
                media_indices=media_indices,
            )
            return True
        except EncoderCapacityError:
            status = "error"
            logger.warning("Encoder encode rejected: capacity exhausted")
            return False
        except Exception:
            status = "error"
            raise
        finally:
            if self.metrics_collector is not None:
                self.metrics_collector.record_api_call(
                    "encode_async", status
                )

    async def register_delivery_addr(self, addr: DeliveryAddr) -> bool:
        operation = (
            "register_delivery_addr_meta"
            if addr.meta_only
            else "register_delivery_addr_full"
        )
        status = "success"
        try:
            return await self._register_delivery_addr(addr)
        except Exception:
            status = "error"
            raise
        finally:
            if self.metrics_collector is not None:
                self.metrics_collector.record_api_call(operation, status)

    async def _register_delivery_addr(self, addr: DeliveryAddr) -> bool:
        """Register a Prefill/Decode TP rank's ZMQ PULL address and trigger push.

        Mass: ``addr.meta_only=True`` (D-side) → triggers a ZMQPushMetaJob
        (push only MMMeta, not ViT tensor).  ``meta_only=False`` (P-side) →
        full ZMQPushJob (push MMResult with ViT tensor after ViT completes).
        """
        try:
            return await self.scheduler.coordinator.register(
                addr, self._zmq_context, self.server_args.encoder_register_timeout,
            )
        except EncoderCapacityError:
            logger.warning("Encoder delivery rejected: capacity exhausted")
            return False

    async def flush_cache(self) -> dict:
        """Schedule a TP-wide cache flush and return immediately.

        The caller guarantees that no request is running or arriving.  TP=1
        and TP>1 use the same fire-and-forget path; the next scheduler step
        broadcasts the flag and clears each worker's local cache.
        """
        if self.attn_tp_rank != 0:
            return {
                "ok": False,
                "message": "Cache flush is only available on Encoder TP rank 0.",
                "tp_size": self.attn_tp_size,
            }

        self.scheduler.request_cache_flush()
        return {
            "ok": True,
            "message": "Encoder cache flush scheduled.",
            "tp_size": self.attn_tp_size,
        }

    def attach_step_task(self, task: Optional[asyncio.Task]) -> None:
        self._step_task = task

    def record_step_success(self) -> None:
        self._last_step_success_at = time.monotonic()
        self._last_step_error = None
        self._consecutive_step_errors = 0

    def record_step_error(self, error: Exception) -> None:
        self._last_step_error = f"{type(error).__name__}: {error}"[:512]
        self._consecutive_step_errors += 1

    def health_status(self) -> dict:
        """Return model readiness, step-loop liveness and heartbeat details."""
        now = time.monotonic()
        preprocessor_ready = self.scheduler.preprocessor is not None
        feature_extractor_ready = self.scheduler.feature_extractor is not None
        step_loop_alive = (
            self._step_task is not None and not self._step_task.done()
        )
        heartbeat_age = (
            now - self._last_step_success_at
            if self._last_step_success_at is not None else None
        )
        heartbeat_fresh = (
            heartbeat_age is not None
            and heartbeat_age <= self._health_step_timeout
        )
        ok = bool(
            preprocessor_ready
            and feature_extractor_ready
            and step_loop_alive
            and heartbeat_fresh
        )
        request_queue = self.scheduler.request_queue_stats()
        return {
            "ok": ok,
            "preprocessor_ready": preprocessor_ready,
            "feature_extractor_ready": feature_extractor_ready,
            "step_loop_alive": step_loop_alive,
            "step_heartbeat_fresh": heartbeat_fresh,
            "last_step_success_age_s": (
                round(heartbeat_age, 3) if heartbeat_age is not None else None
            ),
            "consecutive_step_errors": self._consecutive_step_errors,
            "last_step_error": self._last_step_error,
            "request_queue_length": request_queue["requests"],
            "request_queue_items": request_queue["items"],
            "request_queue_oldest_seconds": round(request_queue["oldest_seconds"], 3),
        }

    def health(self) -> bool:
        return bool(self.health_status()["ok"])


# ═══════════════════════════════════════════════════════════════════════════
#  HTTP transport (FastAPI + uvicorn) — always available
# ═══════════════════════════════════════════════════════════════════════════


class EncoderHTTPServicer:
    """FastAPI request handlers wrapping ``EncoderBackend``.

    Each handler parses the JSON body into plain dataclasses (via Pydantic),
    delegates to ``EncoderBackend``, and serializes the result back to JSON.
    No core logic lives here — everything is delegated so HTTP and gRPC stay
    in sync.

    loop), so this handler is O(1) and never blocks the event loop.
    """

    def __init__(self, backend: EncoderBackend):
        self.backend = backend

    async def encode_async(self, request_id: str, media_urls: list[MediaURL],
                           output_format: str = "rvq_indices",
                           media_indices: Optional[list[int]] = None) -> dict:
        """Handle ``POST /encode_async``: enqueue URLs, return immediately.
        """
        logger.info(
            "[handle_encode_async] rid=%s, urls=%d, media_indices=%s, output_format=%s",
            request_id, len(media_urls), media_indices, output_format,
        )
        ok = await self.backend.encode_async(
            request_id, media_urls, output_format, media_indices=media_indices,
        )
        return {"ok": ok}

    async def register_delivery_addr(self, request_id: str, tp_rank: int,
                                      receive_url: str, tp_size: int = 1,
                                      meta_only: bool = False) -> dict:
        """Handle ``POST /register_delivery_addr``: register a Prefill/Decode TP rank.

        ``meta_only=True`` (D-side) → encoder creates PushMetaJob (push only
        MMMeta, not ViT tensor).  ``meta_only=False`` (P-side) → full MMResult.
        """
        addr = DeliveryAddr(
            request_id=request_id,
            tp_rank=tp_rank,
            receive_url=receive_url,
            tp_size=tp_size,
            meta_only=meta_only,
        )
        ok = await self.backend.register_delivery_addr(addr)
        return {"ok": ok}

    async def health(self) -> dict:
        """Handle ``GET /health``: model and step-loop readiness probe."""
        return self.backend.health_status()

    async def flush_cache(self) -> dict:
        """Handle ``GET/POST /flush_cache`` on all Encoder TP workers."""
        return await self.backend.flush_cache()

    # NOTE: ``_parse_items`` and ``_mm_meta_to_dict`` were removed — image
    # preprocessing now lives in ``EncodeScheduler._preprocess`` (runs inside
    # the ``step()`` loop on all TP ranks), and ``MMMeta`` is no longer
    # serialized over HTTP (it rides on ``MMResult`` via ZMQ).


# ═══════════════════════════════════════════════════════════════════════════
#  Pydantic request models (module-level so FastAPI/Pydantic can resolve them)
# ═══════════════════════════════════════════════════════════════════════════

# Models must be defined at module scope so that Pydantic's ForwardRef
# resolution can find them by name when FastAPI rebuilds the TypeAdapter at
# route registration time.  Defining them inside ``build_fastapi_app`` causes
# ``PydanticUserError: TypeAdapter is not fully defined``.
try:
    from pydantic import BaseModel

    class EncodeAsyncBody(BaseModel):
        request_id: str
        output_format: str = "rvq_indices"
        # list of {"modality": "IMAGE"|"VIDEO"|..., "url": str}; parsed into
        # MediaURL by the handler.  Kept as raw list[dict] here because
        # pydantic cannot easily validate ``MediaURL`` (a frozen dataclass
        # defined in another module) without extra config.
        media_urls: list[dict] = []
        # Mass multi-encoder sharding: global indices of this batch's media
        # in the request's ordered media list (for MMResult.item_index
        # alignment across encoders).  Optional; absent → 0..N-1.
        media_indices: Optional[list[int]] = None

    class DeliveryAddrBody(BaseModel):
        request_id: str
        tp_rank: int
        receive_url: str
        tp_size: int = 1
        # Mass D-side registration: True → encoder creates PushMetaJob
        # (push only MMMeta, not ViT tensor).  False (default, P-side) →
        # full MMResult (ViT tensor + meta).
        meta_only: bool = False
except ImportError:  # pydantic not available (e.g. minimal env)
    EncodeAsyncBody = None  # type: ignore[assignment,misc]
    DeliveryAddrBody = None  # type: ignore[assignment,misc]


def build_fastapi_app(backend: EncoderBackend) -> "FastAPI":
    """Build the FastAPI application wiring all Encoder HTTP routes.

    Routes:
      - ``POST /encode_async``            → ``EncoderHTTPServicer.encode_async``
      - ``POST /register_delivery_addr``  → ``EncoderHTTPServicer.register_delivery_addr``
      - ``GET|POST /flush_cache``         → ``EncoderHTTPServicer.flush_cache``
      - ``GET  /health``                   → ``EncoderHTTPServicer.health``
    """
    from fastapi import FastAPI, HTTPException

    servicer = EncoderHTTPServicer(backend)
    app = FastAPI(title="FluentLLM Encoder", docs_url="/docs")

    @app.post("/encode_async")
    async def _encode_async(body: EncodeAsyncBody) -> dict:
        from sglang.srt.managers.scheduling.multimodal_data import Modality

        media_urls = [
            MediaURL(
                modality=Modality.from_str(m.get("modality", "IMAGE")),
                url=m["url"],
                content_identity=m.get("content_identity"),
                source_version=m.get("source_version"),
            )
            for m in body.media_urls
        ]
        try:
            return await servicer.encode_async(
                body.request_id, media_urls, body.output_format,
                media_indices=body.media_indices,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/register_delivery_addr")
    async def _register_delivery_addr(body: DeliveryAddrBody) -> dict:
        try:
            return await servicer.register_delivery_addr(
                body.request_id, body.tp_rank, body.receive_url, body.tp_size,
                meta_only=body.meta_only,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/health")
    async def _health() -> dict:
        return await servicer.health()

    @app.api_route("/flush_cache", methods=["GET", "POST"])
    async def _flush_cache():
        from fastapi.responses import JSONResponse

        result = await servicer.flush_cache()
        return JSONResponse(content=result, status_code=200 if result["ok"] else 400)

    return app


# ═══════════════════════════════════════════════════════════════════════════
#  gRPC transport (optional) — enabled when proto stubs are importable
# ═══════════════════════════════════════════════════════════════════════════


class EncoderGrpcServicer:
    """gRPC servicer delegating to ``EncoderBackend``.

    Only used when the generated proto stubs (``encoder_pb2`` /
    ``encoder_pb2_grpc``) are importable.  Proto ↔ dataclass conversion
    lives here so the backend stays proto-free.
    """

    def __init__(self, backend: EncoderBackend):
        self.backend = backend

    async def EncodeAsync(self, request, context):
        from .proto import encoder_pb2
        from sglang.srt.managers.scheduling.multimodal_data import Modality

        # Post-refactor: only enqueue URLs; MMMeta rides on MMResult (ZMQ push).
        # gRPC proto items currently carry only ``url``; default modality to
        # IMAGE (proto can add a modality field later).
        media_urls = [MediaURL.image(item.url) for item in request.items]
        ok = await self.backend.encode_async(
            request.request_id, media_urls, request.output_format or "rvq_indices"
        )
        return encoder_pb2.RegisterResponse(ok=ok)

    async def RegisterDeliveryAddr(self, request, context):
        from .proto import encoder_pb2

        addr = DeliveryAddr(
            request_id=request.request_id,
            tp_rank=request.tp_rank,
            receive_url=request.receive_url,
            tp_size=request.tp_size,
        )
        ok = await self.backend.register_delivery_addr(addr)
        return encoder_pb2.RegisterResponse(ok=ok)

    async def Health(self, request, context):
        from .proto import encoder_pb2

        return encoder_pb2.HealthResponse(ok=self.backend.health())

    # NOTE: ``_parse_items`` and ``_build_mm_meta_response`` were removed —
    # gRPC now only enqueues URLs (like HTTP), and MMMeta rides on MMResult.


def _grpc_stubs_available() -> bool:
    """Return True if the generated encoder proto stubs are importable."""
    try:
        from .proto import encoder_pb2, encoder_pb2_grpc  # noqa: F401

        return True
    except ImportError:
        return False


# ═══════════════════════════════════════════════════════════════════════════
#  EncoderServer — top-level server orchestrator
# ═══════════════════════════════════════════════════════════════════════════


class EncoderServer:
    """Encoder server: HTTP (FastAPI+uvicorn, always) + gRPC (optional) + step loop.

    Holds the shared ``EncoderBackend`` and starts:
      - a FastAPI HTTP server (via uvicorn) on ``http_port`` (always),
      - a gRPC server on ``grpc_port`` (only if proto stubs are importable),
      - the EncodeScheduler step loop as an asyncio task.

    All three run on the **same** asyncio event loop.  uvicorn's
    ``Server.serve()`` is scheduled via ``asyncio.create_task`` so it
    shares the loop with the step loop and gRPC server, enabling direct
    in-memory access to the shared ``EncodeScheduler`` state.

    Startup flow:
      1. Load model (encoder_only=True, only vision_tower + quantizer)
      2. Initialize TieredEmbeddingCache (L1 GPU + L2 Host)
      3. Initialize EncodeScheduler (cache + ViT exec + batching)
      4. Wrap them in EncoderBackend (shared by HTTP & gRPC)
      5. Start FastAPI HTTP server + gRPC server (if stubs available)
      6. Start EncodeScheduler step loop (asyncio task)
    """

    def __init__(self, server_args: ServerArgs, model_runner):
        self.server_args = server_args
        self.model_runner = model_runner
        self.model = model_runner.model

        self.attn_tp_rank = model_runner.attn_tp_rank
        self.attn_tp_size = server_args.attn_tp_size
        # tp_cpu_group (gloo) for broadcast_pyobj; None when TP=1 (no sync needed).
        self.tp_cpu_group = None
        if self.attn_tp_size > 1:
            self.tp_cpu_group = model_runner.attention_tp_group.cpu_group

        logger.info(
            "EncoderServer distributed: attn_tp_rank=%d, attn_tp_size=%d, tp_cpu_group=%s",
            self.attn_tp_rank, self.attn_tp_size,
            "set" if self.tp_cpu_group is not None else "None",
        )

        # Shared ZMQ context for all push jobs
        self._zmq_context = zmq.Context(1)

        # Initialize multimodal preprocessor + feature extractor for
        # encoder-side preprocessing (download URL → image_transform /
        # video sampling → pixel_values + grid_thw + hash) via the
        # model-agnostic EncoderPreprocessor + EncoderFeature interfaces.
        self.preprocessor = None
        self.feature_extractor = None
        self._init_mm_processor()

        # Tiered ViT embedding cache
        from sglang.srt.mem_cache.tiered_embedding_cache import TieredEmbeddingCache

        self.cache = TieredEmbeddingCache(
            gpu_size_mb=server_args.encoder_cache_size_mb,
            host_size_mb=server_args.encoder_cache_host_size_mb,
            enable_l2=server_args.encoder_enable_tiered_cache,
        )

        # Encoder metrics are logical service metrics and must be emitted only
        # once, from the HTTP-owning TP rank 0 process.
        self.metrics_collector: Optional["EncoderMetricsCollector"] = None
        if server_args.enable_metrics and self.attn_tp_rank == 0:
            from sglang.srt.observability.metrics.collector import (
                EncoderMetricsCollector,
            )

            self.metrics_collector = EncoderMetricsCollector(
                labels={
                    "model_name": str(server_args.served_model_name),
                    "app_key": str(server_args.app_key or ""),
                    "process_name": (
                        f"Encoder-{server_args.encoder_http_port}-TP0"
                    ),
                },
                metrics_reporters=server_args.metrics_reporters,
            )

        # Scheduling and orchestration
        self.scheduler = EncodeScheduler(
            max_tokens_per_batch=server_args.encoder_max_tokens_per_batch,
            max_items_per_batch=server_args.encoder_max_items_per_batch,
            cache=self.cache,
            preprocessor=self.preprocessor,
            feature_extractor=self.feature_extractor,
            attn_tp_rank=self.attn_tp_rank,
            attn_tp_size=self.attn_tp_size,
            tp_cpu_group=self.tp_cpu_group,
            metrics_collector=self.metrics_collector,
            max_drain_per_step=server_args.encoder_max_drain_per_step,
            max_video_drain_per_step=server_args.encoder_max_video_drain_per_step,
        )

        # Transport-agnostic core (shared by HTTP & gRPC)
        self.backend = EncoderBackend(
            server_args=server_args,
            scheduler=self.scheduler,
            zmq_context=self._zmq_context,
            attn_tp_rank=self.attn_tp_rank,
            attn_tp_size=self.attn_tp_size,
            metrics_collector=self.metrics_collector,
        )

        # Runtime objects (started in serve())
        self._uvicorn_server: Optional["uvicorn.Server"] = None
        self._grpc_server = None
        self._step_task: Optional[asyncio.Task] = None
        self._queue_metrics_task: Optional[asyncio.Task] = None

    def _init_mm_processor(self) -> None:
        """Initialize the multimodal preprocessor for encoder-side preprocessing.

        Model-agnostic: this method never imports any model-specific
        processor class (no ``OmniImageProcessor`` / ``OmniVideoProcessor``
        / ``VideoProcessingPipeline`` /
        ``LongcatFlashVLEncoderPreprocessor``).  It only uses:

        1. ``model.get_encoder_preprocessor_class()`` — the model declares
           which ``EncoderPreprocessor`` subclass implements it.
        2. ``EncoderPreprocessor.from_hf_config(hf_config)`` — the
           interface's own classmethod factory builds the instance; the
           concrete subclass constructs all of its preprocessing
           dependencies internally from ``hf_config``.

        Sets:
        - ``self.preprocessor``: an ``EncoderPreprocessor`` instance (image +
          video preprocessing) built by the factory method.
        - ``self.feature_extractor``: the model itself (implements
          ``EncoderFeature`` for ViT forward).
        """
        from sglang.srt.disaggregation.encode.encoder_base import (
            EncoderFeature, EncoderPreprocessor,
        )

        try:
            feature_extractor: EncoderFeature = self.model
            if not hasattr(feature_extractor, "get_encoder_preprocessor_class"):
                raise RuntimeError(
                    "Encoder model does not implement "
                    "get_encoder_preprocessor_class(); cannot obtain a "
                    "preprocessor class.  Ensure the model class implements "
                    "the EncoderFeature interface."
                )
            preprocessor_cls = feature_extractor.get_encoder_preprocessor_class()
            if not isinstance(preprocessor_cls, type) or \
               not issubclass(preprocessor_cls, EncoderPreprocessor):
                raise RuntimeError(
                    f"get_encoder_preprocessor_class() returned "
                    f"{preprocessor_cls!r} which is not an EncoderPreprocessor "
                    f"subclass."
                )
            hf_config = self.model_runner.model_config.hf_config
            self.preprocessor = preprocessor_cls.from_hf_config(hf_config)
            self.feature_extractor = feature_extractor
            logger.info(
                "Encoder preprocessor (%s) + feature_extractor initialized "
                "via EncoderPreprocessor.from_hf_config() (model-agnostic).",
                preprocessor_cls.__name__,
            )
        except Exception:
            logger.exception(
                "Failed to initialize encoder preprocessor; preprocessing will fail"
            )

    async def serve(self, http_port: Optional[int], grpc_port: Optional[int] = None) -> None:
        """Start FastAPI HTTP server (+ gRPC if available) and the encode step loop.

        Args:
            http_port: port for the FastAPI HTTP server. When ``None``, HTTP
                server is skipped (non-rank-0 encoder TP workers).
            grpc_port: optional port for the gRPC server.  If ``None`` or
                if proto stubs are not importable, gRPC is skipped.
        """
        # 1. FastAPI HTTP server (only on rank 0; skipped for non-rank-0 workers)
        if http_port is not None:
            import uvicorn

            app = build_fastapi_app(self.backend)
            if (
                self.metrics_collector is not None
                and self.metrics_collector.enable_prometheus
            ):
                from sglang.srt.utils import add_prometheus_middleware

                add_prometheus_middleware(app)
            config = uvicorn.Config(
                app,
                host="0.0.0.0",
                port=http_port,
                loop="auto",        # reuse the current event loop (uvloop if installed)
                log_level="warning",
                access_log=False,
            )
            self._uvicorn_server = uvicorn.Server(config)
            # Schedule uvicorn on the current loop — do NOT let it install a
            # signal handler (we manage shutdown ourselves).
            self._uvicorn_server.install_signal_handlers = lambda: None
            asyncio.create_task(self._uvicorn_server.serve())
            logger.info("Encoder FastAPI HTTP server starting on port %d", http_port)
        else:
            logger.info("Skipping HTTP server (non-rank-0 encoder worker)")

        # 2. gRPC server (optional — only if stubs are importable)
        if grpc_port is not None and _grpc_stubs_available():
            try:
                import grpc
                from .proto import encoder_pb2_grpc

                self._grpc_server = grpc.aio.server()
                encoder_pb2_grpc.add_EncoderServicerServicer_to_server(
                    EncoderGrpcServicer(self.backend), self._grpc_server
                )
                self._grpc_server.add_insecure_port(f"[::]:{grpc_port}")
                await self._grpc_server.start()
                logger.info("Encoder gRPC server started on port %d", grpc_port)
            except Exception:
                logger.exception("Failed to start Encoder gRPC server; continuing HTTP-only")
                self._grpc_server = None
        elif grpc_port is not None:
            logger.info(
                "Encoder gRPC stubs not generated; skipping gRPC server. "
                "HTTP server on port %s is active.", http_port
            )

        # 3. Step loop (always — all TP ranks must run it so that ViT forward's
        # RowParallelLinear all_reduce is satisfied)
        self._step_task = asyncio.create_task(self._step_loop())
        self.backend.attach_step_task(self._step_task)
        if self.metrics_collector is not None:
            self._queue_metrics_task = asyncio.create_task(self._report_queue_metrics())

    async def _report_queue_metrics(self):
        # Sample on rank 0 even while preprocessing is waiting on media I/O.
        while True:
            try:
                snapshot = self.scheduler.request_queue_stats()
                self.metrics_collector.observe_request_queue(**snapshot)
            except Exception:
                logger.exception("Encoder queue metrics reporting failed")
            await asyncio.sleep(5.0)

    async def _step_loop(self):
        """Main step loop: drives EncodeScheduler.step() continuously.

        step() is async (ViT forward is offloaded to the persistent Device Worker),
        so the event loop can process HTTP requests (encode_async,
        register_delivery_addr) while ViT is computing.
        """
        while True:
            try:
                did = await self.scheduler.step()
                self.backend.record_step_success()
                if did == 0 and not self.scheduler.pending_size():
                    await asyncio.sleep(0.0005)  # yield when idle
            except Exception as e:
                try:
                    self.backend.record_step_error(e)
                    logger.critical(
                        "[Encoder fatal step_loop] Exiting worker after %s: %s",
                        type(e).__name__, e, exc_info=True,
                    )
                finally:
                    # Diagnostics must not prevent exit. The model/collective
                    # thread may be stuck, so do not wait for executor cleanup.
                    os._exit(1)

    async def shutdown(self):
        """Gracefully shutdown HTTP (uvicorn), gRPC, step loop, and ZMQ."""
        if self._queue_metrics_task is not None:
            self._queue_metrics_task.cancel()
            try:
                await self._queue_metrics_task
            except asyncio.CancelledError:
                pass
            self._queue_metrics_task = None
        # Stop step loop first to avoid new ViT forwards during teardown.
        if self._step_task is not None:
            self._step_task.cancel()
            try:
                await self._step_task
            except asyncio.CancelledError:
                pass
            self._step_task = None
            self.backend.attach_step_task(None)

        # No new device work can be submitted after the step loop exits.
        # Wait for an in-flight forward before tearing down model/device state.
        self.scheduler.shutdown()
        await self.scheduler.coordinator.shutdown()

        # Shutdown uvicorn (FastAPI HTTP).
        if self._uvicorn_server is not None:
            self._uvicorn_server.should_exit = True
            try:
                await self._uvicorn_server.shutdown()
            except Exception:
                logger.exception("Error during uvicorn shutdown")
            self._uvicorn_server = None

        if self._grpc_server is not None:
            await self._grpc_server.stop(grace=5)
            self._grpc_server = None

        self._zmq_context.term()
