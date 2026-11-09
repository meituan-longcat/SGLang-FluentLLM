"""Per-request Encoder router for Mass dynamic ``transfer_info``.

This module is the **only** place that parses the Mass ``transfer_info``
JSON structure (``encoders[]`` / ``media_candidates``).  It produces a
plain ``list[str]`` of encoder URLs that the rest of the LM side
(TokenManager + Scheduler) consumes isomorphically to the static
``server_args.encoder_urls`` — i.e. Scheduler has **zero** awareness of
``transfer_info``.

Lifecycle: ``EncoderRouter`` is a TM-internal transient object.  It is
constructed per request inside ``_apply_epd_mm_processing``, used to
group ``media_urls`` by encoder for parallel ``encode_async`` calls, then
discarded.  It is never pickled across process boundaries (only its
``resolved_urls()`` list[str] escapes to the Scheduler via
``mm_inputs.epd_encoder_urls``).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Optional

from sglang.srt.env import envs

logger = logging.getLogger(__name__)

# Read once at import: the encoder-port override never changes at runtime.
OVERRIDE_ENCODER_PORT: Optional[int] = (
    envs.SGLANG_OVERRIDE_ENCODER_PORT.get_set_value_or(None)
)


class EncoderRouter:
    """TM-internal per-request encoder router.

    Responsibilities:
    1. Parse ``transfer_info.encoders[]`` / ``media_candidates``.
    2. Provide ``media_index → encoder_url`` mapping (for grouping
       ``media_urls`` by encoder before parallel ``encode_async``).
    3. Provide failover advancement on ``media_candidates`` (phase 1:
       method exists but TM aborts on failure without retrying).
    4. Export the final effective encoder URL list (``resolved_urls``)
       for Scheduler consumption — a plain ``list[str]`` isomorphic to
       ``server_args.encoder_urls.split(",")``.
    """

    def __init__(
        self,
        primary: dict[int, str],
        candidates: dict[int, list[str]],
        fallback_urls: list[str],
        has_dynamic_assignments: bool = False,
    ):
        self._primary = primary              # media_index → "http://ip:port"
        self._candidates = candidates        # media_index → list[str] (incl primary)
        self._fallback_urls = fallback_urls  # server_args.encoder_urls fallback
        self._tried: dict[int, int] = {}     # media_index → next candidate idx
        self._has_dynamic_assignments = has_dynamic_assignments

    # ── construction ──────────────────────────────────────────────────────

    @classmethod
    def from_transfer_info(
        cls,
        transfer_info: Optional[dict[str, Any]],
        fallback_urls: list[str],
        *,
        request_id: Optional[str] = None,
        media_count: Optional[int] = None,
    ) -> "EncoderRouter":
        """Build an EncoderRouter from Mass ``transfer_info``.

        If ``transfer_info`` is missing/empty or contains no ``encoders``
        key, materialize a per-media routing table from ``fallback_urls``.
        The static table has the same shape as a Mass routing result, so the
        grouping, parallel ``encode_async`` calls, active-URL propagation,
        and delivery registration all use the common downstream path.

        Static routing starts at a stable request-id-derived offset and then
        round-robins media across Encoders. Unlike Python's process-local
        ``hash()``, the stable digest guarantees that Prefill and Decode map
        the same request to the same Encoder(s).
        """
        if not transfer_info or "encoders" not in transfer_info:
            return cls._from_static(
                fallback_urls,
                request_id=request_id,
                media_count=media_count,
            )

        primary: dict[int, str] = {}
        candidates: dict[int, list[str]] = {}

        encoders = transfer_info.get("encoders") or []
        for enc in encoders:
            ip = enc.get("ip")
            port = (
                enc.get("port")
                if OVERRIDE_ENCODER_PORT is None
                else OVERRIDE_ENCODER_PORT
            )
            if not ip or not port:
                logger.warning(
                    "EncoderRouter: skip encoder entry missing ip/port: %s", enc
                )
                continue
            url = f"http://{ip}:{port}"
            for idx in enc.get("media_indices", []):
                primary[int(idx)] = url

        raw_candidates = transfer_info.get("media_candidates") or {}
        for k, v in raw_candidates.items():
            cands: list[str] = []
            for x in v or []:
                ip = x.get("ip")
                port = (
                    x.get("port")
                    if OVERRIDE_ENCODER_PORT is None
                    else OVERRIDE_ENCODER_PORT
                )
                if ip and port:
                    cands.append(f"http://{ip}:{port}")
            if cands:
                candidates[int(k)] = cands

        # Validate candidates[i][0] == primary[i] (risk V2: mass bug)
        for idx, cands in candidates.items():
            if cands and idx in primary and cands[0] != primary[idx]:
                raise ValueError(
                    f"EncoderRouter: media_candidates[{idx}][0]={cands[0]} "
                    f"!= encoders primary[{idx}]={primary[idx]}. "
                    f"Mass returned inconsistent transfer_info, aborting."
                )

        logger.info(
            "EncoderRouter.from_transfer_info: primary=%s, candidates=%s, "
            "fallback=%d urls",
            primary, {k: len(v) for k, v in candidates.items()}, len(fallback_urls),
        )
        return cls(
            primary,
            candidates,
            fallback_urls,
            has_dynamic_assignments=bool(primary or candidates),
        )

    @classmethod
    def _from_static(
        cls,
        fallback_urls: list[str],
        *,
        request_id: Optional[str] = None,
        media_count: Optional[int] = None,
    ) -> "EncoderRouter":
        """Convert static URLs into the same per-media model used by Mass."""
        # Drop blanks and duplicates while preserving the configured order.
        urls = list(
            dict.fromkeys(url.strip() for url in fallback_urls if url.strip())
        )
        if not urls:
            return cls(primary={}, candidates={}, fallback_urls=[])

        count = max(0, int(media_count or 0))
        start = cls._stable_request_offset(request_id, len(urls))
        primary: dict[int, str] = {}
        candidates: dict[int, list[str]] = {}
        for media_index in range(count):
            primary_index = (start + media_index) % len(urls)
            ordered = urls[primary_index:] + urls[:primary_index]
            primary[media_index] = ordered[0]
            candidates[media_index] = ordered

        logger.info(
            "EncoderRouter._from_static: request_id=%s, media_count=%d, "
            "start=%d, primary=%s",
            request_id,
            count,
            start,
            primary,
        )
        return cls(primary=primary, candidates=candidates, fallback_urls=urls)

    @staticmethod
    def _stable_request_offset(request_id: Optional[str], encoder_count: int) -> int:
        """Return a process-independent Encoder offset for a request."""
        if encoder_count <= 1 or not request_id:
            return 0
        digest = hashlib.sha256(str(request_id).encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big") % encoder_count

    # ── routing ──────────────────────────────────────────────────────────

    @property
    def has_dynamic_assignments(self) -> bool:
        """Whether usable assignments came from request ``transfer_info``."""

        return self._has_dynamic_assignments

    def url_for_media(self, media_index: int) -> Optional[str]:
        """Current primary encoder URL for a media index.

        Considers failover advancement (``_tried``).  Returns None if no
        encoder is available (caller should abort the request).
        """
        if self._candidates:
            tried = self._tried.get(media_index, 0)
            cands = self._candidates.get(media_index, [])
            if tried < len(cands):
                return cands[tried]
        if self._primary:
            return self._primary.get(media_index)
        if self._fallback_urls:
            return self._fallback_urls[0]
        return None

    def advance_failover(self, media_index: int) -> None:
        """Advance to the next candidate for a media index.

        Phase 1 decision: TM aborts on failure without retrying, so this
        method is retained for API completeness / future phase 2 but not
        invoked by the TM failover path.
        """
        self._tried[media_index] = self._tried.get(media_index, 0) + 1
        cands = self._candidates.get(media_index, [])
        if self._tried[media_index] >= len(cands):
            logger.warning(
                "EncoderRouter: failover exhausted for media_index=%d "
                "(candidates=%d)",
                media_index, len(cands),
            )

    # ── export for Scheduler ──────────────────────────────────────────────

    def resolved_urls(self) -> list[str]:
        """Export the effective encoder URL list (deduplicated, order-stable).

        This is consumed by the Scheduler as a plain ``list[str]`` —
        isomorphic to ``server_args.encoder_urls.split(",")``.  Includes
        all primary + candidate URLs (so register_delivery_addr reaches
        every encoder that might push data back, including failover
        targets).  Static fallback mode returns ``fallback_urls`` as-is.
        """
        if not self._primary and not self._candidates:
            return list(self._fallback_urls)

        seen: dict[str, None] = {}
        for url in self._primary.values():
            if url not in seen:
                seen[url] = None
        for cands in self._candidates.values():
            for url in cands:
                if url not in seen:
                    seen[url] = None
        return list(seen.keys())
