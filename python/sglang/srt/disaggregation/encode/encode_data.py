"""Data structures for the EPD Encoder server.

This module contains only pure data classes shared between the Encoder side
(``scheduler.py``) and the Prefill side (``encode_receiver.py``):

- ``EncodeRequest`` / ``MMMeta`` / ``MMItemMeta``: request & metadata.
- ``PendingEncodeItem``: item awaiting ViT (Encoder side).
- ``MMResult``: ViT output pushed from Encoder to Prefill.
- ``DeliveryAddr``: Prefill TP rank's receive address registration.

Prefill-side runtime structures (``WaitingMMRequest`` etc.) live in
``encode_receiver.py``; Encoder-side runtime classes (``EncodeScheduler``,
push jobs) live in ``scheduler.py``; request/media ownership lives in
``media_coordinator.py``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import sys
from array import array
from enum import Enum
from typing import Optional

import torch

from sglang.srt.managers.scheduling.multimodal_data import Modality, MultimodalDataItem


class CanonicalRequestTokenStatus(str, Enum):
    """Terminal state of Scheduler-side EPD token canonicalization."""

    TOKEN_READY = "token_ready"
    FAILED = "failed"
    CANCELLED = "cancelled"


def compute_token_checksum(input_ids) -> int:
    """Return a stable signed-int64-safe checksum for canonical token IDs."""

    values = None
    if isinstance(input_ids, (list, tuple)):
        # C-level conversion: a per-token generator costs ~56 ms for 680K
        # tokens and runs on every rank's Scheduler thread.
        try:
            values = array("q", input_ids)
        except TypeError:
            values = None
    if values is None:
        values = array("q", (int(token_id) for token_id in input_ids))
    if sys.byteorder != "little":
        values.byteswap()
    digest = hashlib.blake2b(values, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) & ((1 << 63) - 1)


# Canonical ids fit int32 by design (vocab ids and signed-int32 multimodal pad
# values); int64 is the fallback for anything wider.
_PACK_TYPECODES = tuple(
    code for code, size in (("i", 4), ("q", 8)) if array(code).itemsize == size
)


def pack_token_ids(input_ids) -> bytes:
    """Compact canonical token encoding for descriptor transport.

    One typecode byte followed by little-endian values.  Unpickling one bytes
    object costs a memcpy, whereas a pickled tuple of 680K ints costs ~18 ms
    per receiving rank to rebuild.
    """
    for code in _PACK_TYPECODES:
        try:
            values = array(code, input_ids)
        except OverflowError:
            continue
        if sys.byteorder != "little":
            values.byteswap()
        return code.encode() + values.tobytes()
    raise OverflowError("token id does not fit in int64")


def unpack_token_ids(packed: bytes) -> list[int]:
    values = array(packed[:1].decode())
    values.frombytes(memoryview(packed)[1:])
    if sys.byteorder != "little":
        values.byteswap()
    return values.tolist()


def packed_token_count(packed: bytes) -> int:
    if not packed:
        return 0
    return (len(packed) - 1) // array(packed[:1].decode()).itemsize


def compute_packed_token_checksum(packed: bytes) -> int:
    """Checksum of a ``pack_token_ids`` payload (hashlib releases the GIL)."""
    digest = hashlib.blake2b(packed, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) & ((1 << 63) - 1)


@dataclasses.dataclass(frozen=True)
class CanonicalRequestTokenDescriptor:
    """Authoritative token state broadcast by an EPD Scheduler coordinator.

    Carried in the ordered Scheduler ``broadcast_pyobj`` frame.

    - ``schema_version=1``: ``input_ids`` tuple, ``token_checksum`` from
      ``compute_token_checksum``.
    - ``schema_version=2``: ``packed_input_ids`` from ``pack_token_ids``,
      ``token_checksum`` from ``compute_packed_token_checksum``.
    """

    request_id: str
    generation: int
    status: CanonicalRequestTokenStatus
    input_ids: tuple[int, ...] = ()
    token_checksum: int = 0
    error_message: Optional[str] = None
    schema_version: int = 1
    packed_input_ids: bytes = b""


# ═══════════════════════════════════════════════════════════════════════════
#  Media URL (modality-tagged URL carried from Prefill → Encoder)
# ═══════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass(frozen=True)
class MediaURL:
    """A single media URL tagged with its modality.
    Attributes:
        modality: one of ``Modality.IMAGE`` / ``Modality.VIDEO`` / ...
        url: the actual download URL (http(s)://..., file://..., data:...).
        content_identity: optional stable object/content identity supplied by
            the caller.  When absent, the Encoder conservatively uses the
            exact URL as the identity (no generic query stripping).
        source_version: optional immutable object version / ETag / revision.
            Versionless URLs use a short-lived best-effort URL memo entry.
    """
    modality: Modality
    url: str
    content_identity: Optional[str] = None
    source_version: Optional[str] = None

    @classmethod
    def image(cls, url: str) -> "MediaURL":
        return cls(modality=Modality.IMAGE, url=url)

    @classmethod
    def video(cls, url: str) -> "MediaURL":
        return cls(modality=Modality.VIDEO, url=url)

    @property
    def is_image(self) -> bool:
        return self.modality is Modality.IMAGE

    @property
    def is_video(self) -> bool:
        return self.modality is Modality.VIDEO


# ═══════════════════════════════════════════════════════════════════════════
#  Encoder-side request & metadata
# ═══════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass(frozen=True)
class EncodeRequest:
    """A multimodal request arriving at the Encoder.

    Attributes:
        request_id: globally unique request identifier (assigned by Prefill).
        items: all multimodal items for this request (after per-image splitting).
        output_format: "rvq_indices" (phase 1 default) or "vit_embedding" (phase 2).
        media_indices: global indices of this batch's media in the request's
            ordered media list (for multi-encoder sharding alignment).
    """
    request_id: str
    items: list[MultimodalDataItem]
    output_format: str = "rvq_indices"
    media_indices: tuple[int, ...] = ()


@dataclasses.dataclass
class MMMeta:
    """Metadata returned by encode_async to Prefill.

    Prefill uses this to:
    1. Set pad_value per item (for radix cache prefix matching).
    2. Get grid_thw (for MRoPE position encoding, token expansion).
    3. Know ViT output token count (for allocating receive buffers).

    Attributes:
        request_id: corresponding request ID.
        item_metas: per-item metadata, 1:1 with EncodeRequest.items.
        vit_pending: True if ViT still computing (cache-miss items exist).
    """
    request_id: str
    item_metas: list[MMItemMeta]
    vit_pending: bool = True

    def get_mm_extra_meta(self) -> dict:
        """Return a dict of per-item mm_extra_meta for mm_processor.get_mm_data.
        """
        grid_thw_list = [list(im.grid_thw) for im in self.item_metas]
        num_tokens_list = [im.num_tokens for im in self.item_metas]
        pad_values = [im.pad_value for im in self.item_metas]
        return {
            "grid_thw": grid_thw_list,
            "num_tokens": num_tokens_list,
            "pad_values": pad_values,
        }


@dataclasses.dataclass
class MMItemMeta:
    """Metadata for a single multimodal item.

    Attributes:
        item_index: item's index within the request.
        hash: content hash (computed from pixel_values, deterministic).
        pad_value: modality-tagged, content-hash-derived pad value.
        grid_thw: [T, H, W] patch grid dimensions.
        num_tokens: ViT output token count.
        embedding_dim: ViT output embedding dim (0 for rvq_indices mode).
        video_timestamps: temporal-group timestamp strings inserted into the
            original video prompt. Empty for non-video items or when the
            configured frame pattern has no timestamp text.
    """
    item_index: int
    hash: int
    pad_value: int
    grid_thw: list[int]       # [T, H, W]
    num_tokens: int
    embedding_dim: int = 0    # 0 for rvq_indices mode
    video_timestamps: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class PendingEncodeItem:
    """A vision item waiting for ViT encoding.

    Key design: items from the same request have different item_index,
    so they can be independently scheduled into different ViT batches.

    Attributes:
        request_id: request identifier.
        item_index: item's index within the request.
        cost: patch token count = T * H * W (for batch token budget).
    """
    request_id: str
    item_index: int
    cost: int

    @property
    def key(self) -> tuple[str, int]:
        return (self.request_id, self.item_index)


# ═══════════════════════════════════════════════════════════════════════════
#  ViT output & delivery address
# ═══════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass
class MMResult:
    """ViT computation output, pushed from Encoder to Prefill.

    ``error`` carries a terminal Encoder-side failure.  Error results never
    contain a tensor, but use the same transport envelope so both full-result
    and meta-only receivers can stop waiting immediately instead of timing out.
    """
    request_id: str
    item_index: int
    # Mutually exclusive; fill one based on output_format
    rvq_indices: Optional[torch.Tensor] = None    # [num_tokens, depth] int32
    vit_embedding: Optional[torch.Tensor] = None   # [num_tokens, hidden_dim] bf16
    num_tokens: int = 0

    mm_meta: Optional[MMMeta] = None
    error: Optional[str] = None

    @property
    def data(self) -> Optional[torch.Tensor]:
        """Get the actual data tensor."""
        return self.rvq_indices if self.rvq_indices is not None else self.vit_embedding

    @property
    def nbytes(self) -> int:
        t = self.data
        return t.numel() * t.element_size() if t is not None else 0


@dataclasses.dataclass
class DeliveryAddr:
    """Receiving address registered by an LM-side EPD consumer.

    Prefill registers PP0 attention-TP consumers.  Decode registers a single
    TP0 meta coordinator and relays canonical tokens inside its TP group.

    Attributes:
        meta_only: True for Decode-side registration — encoder creates a
            PushMetaJob that pushes only MMMeta (preprocess done, not waiting
            for ViT). False for Prefill-side — full MMResult (ViT tensor + meta).
    """
    request_id: str
    tp_rank: int
    receive_url: str        # e.g. "tcp://10.0.0.1:30001"
    tp_size: int
    meta_only: bool = False
