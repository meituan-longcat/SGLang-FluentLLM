"""TokenizerManager multimodal mixin.

Encapsulates all multimodal-related logic for TokenizerManager:
  - mm_processor initialization
  - transport strategy initialization
  - _apply_mm_processing(): async image/audio preprocessing pipeline
  - wrap_mm_features(): called in _send_one_request()
  - EPD: _epd_encode_async() remote encode via HTTP + register_delivery_addr

Usage:
    class TokenizerManager(TokenizerCommunicatorMixin, TokenizerMultimodalMixin):
        def __init__(self, server_args, port_args):
            ...
            self.init_multimodal(server_args)
"""

from __future__ import annotations

import asyncio
import logging
import os
import reprlib
from typing import TYPE_CHECKING, List, Optional, Tuple

from sglang.srt.managers.scheduling.multimodal_data import (
    Modality,
    MultimodalInputs,
)
from sglang.srt.multimodal.mm_feature_transport import get_mm_transport_strategy
from sglang.srt.multimodal.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    MultiModalityDataPaddingPatternTokenPairs,
)
from sglang.srt.multimodal.multimodal_processor import get_mm_processor
from sglang.srt.server_args import ServerArgs

if TYPE_CHECKING:
    from sglang.srt.disaggregation.encode.encode_data import MediaURL
    from sglang.srt.managers.io_struct import GenerateReqInput

logger = logging.getLogger(__name__)


class TokenizerMultimodalMixin:
    """Mixin that encapsulates all multimodal-related logic for TokenizerManager.

    The main TokenizerManager class must call self.init_multimodal(server_args)
    after the tokenizer/processor fields are set up.
    """

    def init_multimodal(self, server_args: ServerArgs) -> None:
        """Initialize the multimodal processor and transport strategy."""
        self.mm_processor = None
        if not server_args.skip_tokenizer_init and self.model_config.is_multimodal:
            # tokenizer_executor_num_processes coexists with multimodal: the ProcessPool
            # handles text tokenization only, orthogonal to in-process image processing
            # (pool is created in TokenizerManager's multimodal branch). No raise here.
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            self.mm_processor = get_mm_processor(
                self.model_config.hf_config,
                server_args,
                self.processor,
            )
            assert self.mm_processor is not None, "Failed to initialize multimodal processor."
            logger.info(
                "Multimodal processor initialized: %s",
                type(self.mm_processor).__name__,
            )

        self._mm_image_start_token = getattr(
            self.mm_processor, "image_start_token", None
        )
        self._mm_video_start_token = getattr(
            self.mm_processor, "video_start_token", None
        )

        # Build the transport strategy. No processor injection needed — base_processor
        # always pre-hashes items, so strategy.wrap() can safely run after
        # from_processor_output() completes.
        self._mm_transport_strategy = get_mm_transport_strategy(server_args)

        self._epd_encoder_client = None
        self._epd_encoder_urls: list[str] = []
        self._epd_language_only = getattr(server_args, "language_only", False)
        if self._epd_language_only:
            self._init_epd_encoder_client(server_args)

    def _init_epd_encoder_client(self, server_args: ServerArgs) -> None:
        """Initialize the HTTP client to the remote Encoder server.

        The Encoder server exposes an HTTP API (``/encode_async`` and
        ``/register_delivery_addr``) via aiohttp.  We keep a list of encoder
        base URLs (e.g. ``http://10.0.0.1:9100``). Static routing uses a
        stable request-id offset and round-robins multiple media across URLs.
        """
        urls = []
        if server_args.encoder_urls:
            urls = [u.strip() for u in server_args.encoder_urls.split(",") if u.strip()]
        if not urls:
            logger.warning(
                "EPD language_only=True but no encoder_urls configured; "
                "remote encode will be disabled."
            )
            return
        self._epd_encoder_urls = urls
        logger.info(
            "EPD encoder client initialized: urls=%s, output_format=%s",
            self._epd_encoder_urls,
            server_args.encoder_output_format,
        )

    async def _apply_mm_processing(
        self,
        obj: "GenerateReqInput",
        input_text: str,
        input_ids: List[int],
        modality: Optional[Modality],
    ) -> Tuple[List[int], Optional[MultimodalInputs]]:
        """Run multimodal preprocessing: encode images/audio, replace placeholder tokens.

        In EPD (language_only) mode, this delegates to ``_apply_epd_mm_processing``
        Only sends image URLs to the remote Encoder (a lightweight HTTP notification) and
        defers tokenization to the Scheduler side

        In non-EPD mode, calls ``mm_processor.process_mm_data_async()`` locally
        and replaces placeholder tokens with per-item pad_values.
        """
        # EPD mode: remote encode, skip local process_mm_data_async
        if self._epd_language_only and modality is not None:
            return await self._apply_epd_mm_processing(obj, input_text, input_ids)

        mm_inputs = None
        try:
            proc_output = await self.mm_processor.process_mm_data_async(
                image_data=obj.image_data,
                audio_data=obj.audio_data,
                input_text=input_text,
                request_obj=obj,
            )
            if proc_output is not None:
                # Override input_ids with the expanded ids from processor
                if proc_output.input_ids is not None:
                    input_ids = proc_output.input_ids

                # from_processor_output calls item.set_pad_value() for every item,
                # so hash and pad_value are already available after this call.
                mm_inputs = MultimodalInputs.from_processor_output(proc_output)

                # Replace placeholder tokens in input_ids with pad_values
                # immediately here in the tokenizer process, before sending
                # to the scheduler via ZMQ. This guarantees that
                # req.origin_input_ids (and therefore fill_ids) already
                # carries pad_values when RadixCache does prefix matching.
                #
                # Two patterns are supported:
                #   - MultimodalTokens: Qwen-VL / KimiK25 style
                #   - TokenPairs:       LLaVA-style (im_start_id / im_end_id)
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
                        pattern = MultiModalityDataPaddingPatternTokenPairs(
                            data_token_pairs
                        )
                    else:
                        pattern = MultiModalityDataPaddingPatternMultimodalTokens()

                    input_ids = pattern.pad_input_tokens(input_ids, mm_inputs)
                    logger.debug(
                        "MM pad applied: %d items, rid=%s",
                        len(mm_inputs.mm_items), obj.rid,
                    )

                logger.debug(
                    "MM processing done: %d items, rid=%s",
                    len(mm_inputs.mm_items), obj.rid,
                )
        except Exception as e:
            logger.warning(
                "Multimodal processing failed for rid=%s: %s",
                obj.rid, e,
                exc_info=True,
            )
        return input_ids, mm_inputs

    def _get_mm_modality(
        self, obj: "GenerateReqInput", input_text: str = ""
    ) -> Optional[Modality]:
        """Detect the dominant raw request modality; VIDEO wins mixed input."""
        if obj.modality is not None:
            if not isinstance(obj.modality, Modality):
                raise ValueError(
                    "GenerateReqInput.modality must be a Modality "
                    "for a single request."
                )
            return obj.modality

        modalities = obj.modalities or ()
        if isinstance(modalities, str):
            modalities = (modalities,)
        modality_names = {
            (item if isinstance(item, str) else item.name)
            .strip()
            .upper()
            .replace("-", "_")
            for item in modalities
        }

        input_text = input_text if isinstance(input_text, str) else ""
        if (
            obj.video_data is not None
            or "VIDEO" in modality_names
            or bool(
                input_text
                and self._mm_video_start_token
                and self._mm_video_start_token in input_text
            )
        ):
            return Modality.VIDEO

        has_image_modality = any(
            name not in {"", "TEXT", "VIDEO", "AUDIO"}
            for name in modality_names
        )
        if (
            obj.image_data is not None
            or has_image_modality
            or bool(
                input_text
                and self._mm_image_start_token
                and self._mm_image_start_token in input_text
            )
        ):
            return Modality.IMAGE

        if obj.audio_data is not None or "AUDIO" in modality_names:
            return Modality.AUDIO
        return None

    @staticmethod
    def _resolve_mm_modality(
        mm_inputs: Optional[MultimodalInputs],
        fallback: Optional[Modality],
    ) -> Optional[Modality]:
        """Prefer the normalized processor or EPD modality over raw intent."""
        if mm_inputs is None:
            return fallback

        if mm_inputs.epd_media_urls:
            modalities = {item.modality for item in mm_inputs.epd_media_urls}
            if Modality.VIDEO in modalities:
                return Modality.VIDEO
            if Modality.IMAGE in modalities:
                return Modality.IMAGE
            if Modality.AUDIO in modalities:
                return Modality.AUDIO

        if mm_inputs.contains_video:
            return Modality.VIDEO
        if mm_inputs.contains_image:
            return Modality.IMAGE
        if mm_inputs.contains_audio:
            return Modality.AUDIO
        return fallback

    async def _apply_epd_mm_processing(
        self,
        obj: "GenerateReqInput",
        input_text: str,
        input_ids: List[int],
    ) -> Tuple[List[int], Optional[MultimodalInputs]]:
        """EPD mode (post-refactor): notify Encoder, defer tokenization.

        Mass dynamic routing (phase 1):
        1. Extracts image URLs from the input text (via ``_split_multimodal_chunk``).
        2. Parses ``obj.transfer_info`` → ``EncoderRouter`` (TM-internal).
        3. Groups ``media_urls`` by ``router.url_for_media(idx)`` and sends one
           ``encode_async`` per encoder in parallel (multi-encoder sharding).
           Phase 1: on failure, log + abort (no failover retry to candidates).
        4. Returns ``(input_ids, mm_inputs)`` where ``mm_inputs`` is a minimal
           placeholder carrying ``epd_media_urls`` / ``epd_input_text`` /
           ``epd_encoder_urls`` (a plain ``list[str]`` isomorphic to
           ``server_args.encoder_urls`` — Scheduler consumes it with zero
           ``transfer_info`` awareness).  Tokenization is deferred to the
           Scheduler admission.
        """
        # Step 1: Extract image/video URLs from text (pure regex, no I/O)
        text_list, mm_label_list = self.mm_processor._split_multimodal_chunk(input_text)
        from sglang.srt.disaggregation.encode.encode_data import MediaURL
        from sglang.srt.disaggregation.encode.encoder_router import EncoderRouter

        media_urls: list[MediaURL] = []
        for seg, label in zip(text_list, mm_label_list):
            if label == "image":
                media_urls.append(MediaURL.image(seg))
            elif label == "video":
                media_urls.append(MediaURL.video(seg))
            # 'text' segments are skipped — they carry no media URL.
        if not media_urls:
            logger.warning("EPD mode but no media URLs found in text for rid=%s", obj.rid)
            if input_ids is None:
                request_repr = reprlib.Repr()
                request_repr.maxstring = 512
                logger.warning(
                    "EPD invalid request: rid=%s, modality=%s, modalities=%s, "
                    "text=%s, image_data=%s, video_data=%s, audio_data=%s",
                    obj.rid, obj.modality, request_repr.repr(obj.modalities),
                    request_repr.repr(input_text), request_repr.repr(obj.image_data),
                    request_repr.repr(obj.video_data), request_repr.repr(obj.audio_data),
                )
                raise ValueError(
                    f"Invalid EPD multimodal request for rid={obj.rid}: "
                    "no image or video URLs found in text."
                )
            return input_ids, None

        # Step 2: Resolve Encoder assignments through the existing Router, then
        # adapt MaaS marker frames only when dynamic assignments are available.
        transfer_info = getattr(obj, "transfer_info", None)
        router = EncoderRouter.from_transfer_info(
            transfer_info,
            fallback_urls=self._epd_encoder_urls,
            request_id=obj.rid,
            media_count=len(media_urls),
        )

        frame_routing = None
        if router.has_dynamic_assignments:
            from sglang.srt.disaggregation.encode.maas_frame_epd_adapter import (
                build_maas_frame_routing,
            )

            frame_routing = build_maas_frame_routing(
                media_urls,
                self.mm_processor,
            )

        if frame_routing is None:
            compute_media_urls = media_urls
            owner_resource_indices = list(range(len(media_urls)))
            compute_to_logical = None
        else:
            compute_media_urls = list(frame_routing.compute_media_urls)
            owner_resource_indices = list(
                frame_routing.owner_resource_indices
            )
            compute_to_logical = (
                list(frame_routing.compute_to_logical)
                if frame_routing.compute_to_logical is not None
                else None
            )

        # Step 3: Group Encoder compute items by their upstream assignment.
        # encoder_url(str) → list[(media_index, MediaURL)]
        grouped: dict[str, list[tuple[int, MediaURL]]] = {}
        for idx, (media_url, owner_resource_index) in enumerate(
            zip(compute_media_urls, owner_resource_indices)
        ):
            url = router.url_for_media(owner_resource_index)
            if url is None:
                logger.error(
                    "EPD encode_async: no encoder for compute_index=%d rid=%s "
                    "(transfer_info=%s, fallback_urls=%d)",
                    idx, obj.rid, bool(transfer_info), len(self._epd_encoder_urls),
                )
                raise ValueError(
                    f"EPD encode_async failed for rid={obj.rid}: no encoder "
                    f"assigned for compute_index={idx}. Request aborted."
                )
            grouped.setdefault(url, []).append((idx, media_url))

        # Step 4: Parallel encode_async per encoder (multi-encoder sharding).
        tasks = []
        for url, items in grouped.items():
            subset = [media_url for _, media_url in items]
            media_indices = [idx for idx, _ in items]
            tasks.append(self._epd_encode_async(
                rid=obj.rid,
                media_urls=subset,
                media_indices=media_indices,
                output_format=self.server_args.encoder_output_format,
                encoder_url=url,
            ))
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Step 5: Failure handling — phase 1: log + abort, no failover retry.
        for (url, items), res in zip(grouped.items(), results):
            if isinstance(res, Exception):
                logger.error(
                    "EPD encode_async exception: rid=%s, encoder=%s, "
                    "media_indices=%s, err=%s: %s",
                    obj.rid, url, [idx for idx, _ in items],
                    type(res).__name__, res,
                )
                raise ValueError(
                    f"EPD encode_async failed for rid={obj.rid}: encoder {url} "
                    f"raised {type(res).__name__}. Request aborted (phase 1: "
                    f"no failover retry)."
                )
            if not res:
                logger.error(
                    "EPD encode_async rejected: rid=%s, encoder=%s, "
                    "media_indices=%s (timeout/error).",
                    obj.rid, url, [idx for idx, _ in items],
                )
                raise ValueError(
                    f"EPD encode_async failed for rid={obj.rid}: encoder {url} "
                    f"did not accept (timeout/error). Request aborted."
                )

        # Step 6: Build a minimal mm_inputs carrying deferred fields.
        mm_inputs = MultimodalInputs(mm_items=[])
        mm_inputs.epd_media_urls = media_urls
        mm_inputs.epd_input_text = input_text
        mm_inputs.epd_compute_to_logical = compute_to_logical
        # Register delivery addresses only with Encoders that actually received
        # a shard for this request.  Registering every routing candidate creates
        # orphan placeholder jobs and makes the receiver wait for nonexistent
        # results.
        mm_inputs.epd_encoder_urls = list(grouped.keys())

        logger.info(
            "EPD mm processing deferred: rid=%s, logical_media=%d, "
            "compute_items=%d, encoders=%d "
            "(tokenization → Scheduler admission)",
            obj.rid,
            len(media_urls),
            len(compute_media_urls),
            len(grouped),
        )
        return input_ids, mm_inputs

    def wrap_mm_features(self, tokenized_obj):
        """Wrap multimodal feature tensors before ZMQ send. Called in _send_one_request()."""
        return self._mm_transport_strategy.wrap(tokenized_obj)

    # ─────────────────────────────────────────────────────────────────────
    # EPD (Encoder-Prefill-Decode) remote encode methods
    # ─────────────────────────────────────────────────────────────────────

    async def _epd_encode_async(
        self,
        rid: str,
        media_urls: list[MediaURL],
        output_format: str = "rvq_indices",
        media_indices: Optional[list[int]] = None,
        encoder_url: Optional[str] = None,
    ) -> bool:
        """Notify the remote Encoder to encode the given media URLs.

        Args:
            rid: request id.
            media_urls: list of ``MediaURL`` (modality-tagged URLs) to encode.
            output_format: "rvq_indices" or "vit_embedding".
            media_indices: global indices of this batch's media in the
                request's ordered media list (for multi-encoder sharding
                alignment).  None/empty → encoder treats as 0..N-1.
            encoder_url: explicit encoder URL (from EncoderRouter).  If None,
                falls back to a stable request-id-derived selection for
                backward-compatible direct callers.

        Returns:
            True if the Encoder accepted the request, False on failure.
        """
        if encoder_url is None:
            if not self._epd_encoder_urls:
                logger.error("EPD encode_async called but no encoder_urls configured")
                return False
            from sglang.srt.disaggregation.encode.encoder_router import EncoderRouter
            fallback_router = EncoderRouter.from_transfer_info(
                None,
                fallback_urls=self._epd_encoder_urls,
                request_id=rid,
                media_count=1,
            )
            encoder_url = fallback_router.url_for_media(0)
            if encoder_url is None:
                return False

        # Serialize MediaURL, including optional stable identity/version used
        # by the Encoder's unified IMAGE/VIDEO URL memo.
        payload = {
            "request_id": rid,
            "media_urls": [
                {
                    "modality": mu.modality.name,
                    "url": mu.url,
                    **(
                        {"content_identity": mu.content_identity}
                        if mu.content_identity else {}
                    ),
                    **(
                        {"source_version": mu.source_version}
                        if mu.source_version else {}
                    ),
                }
                for mu in media_urls
            ],
            "output_format": output_format,
        }
        if media_indices:
            payload["media_indices"] = list(media_indices)

        logger.info(
            "EPD encode_async: rid=%s, encoder_url=%s, urls=%d, "
            "media_indices=%s, output_format=%s",
            rid, encoder_url, len(media_urls), media_indices, output_format,
        )

        try:
            resp = await asyncio.to_thread(
                self._post_encode_async, encoder_url, payload, self.server_args.encoder_encode_timeout
            )
            if resp is None:
                return False
            ok = bool(resp.get("ok", False))
            logger.info(
                "EPD encode_async accepted: rid=%s, ok=%s (MMMeta will arrive on MMResult)",
                rid, ok,
            )
            return ok
        except Exception:
            logger.exception("EPD encode_async HTTP POST failed for rid=%s", rid)
            return False

    @staticmethod
    def _post_encode_async(encoder_url: str, payload: dict, timeout: float = 5.0) -> Optional[dict]:
        """Synchronous HTTP POST to ``{encoder_url}/encode_async``.

        Runs in a worker thread via ``asyncio.to_thread`` so it never blocks
        the TokenizerManager event loop.
        Args:
            encoder_url: base URL of the Encoder server.
            payload: JSON body for /encode_async.
            timeout: HTTP timeout in seconds (from server_args.encoder_encode_timeout).
        """
        import requests

        try:
            resp = requests.post(
                f"{encoder_url}/encode_async",
                json=payload,
                timeout=timeout,
            )
            if resp.status_code != 200:
                logger.warning(
                    "EPD encode_async failed: rid=%s, status=%d",
                    payload.get("request_id"),
                    resp.status_code,
                )
                return None
            return resp.json()
        except Exception:
            logger.exception(
                "EPD encode_async HTTP POST failed for rid=%s",
                payload.get("request_id"),
            )
            return None
