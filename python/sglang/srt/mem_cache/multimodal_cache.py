"""
ViT Embedding Cache for multimodal models.

Provides an LRU cache for vision encoder outputs, keyed by pixel_values hash.
When the same image appears multiple times (multi-turn conversations, repeated requests),
the ViT + Projector computation can be completely skipped by looking up the cached embedding.

Usage:
    # Initialize at server startup
    init_mm_embedding_cache(size_mb * 1024 * 1024)

    # In model forward pass:
    cached = embedding_cache.get_single(item.hash)
    if cached is not None:
        vision_embeds = cached.embedding  # Cache HIT: skip ViT
    else:
        vision_embeds = vision_tower(pixel_values)  # Cache MISS: run ViT
        embedding_cache.set(item.hash, EmbeddingResult(embedding=vision_embeds))
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, List, Optional

import torch


def _get_tensor_size(embedding: torch.Tensor) -> int:
    """Return size in bytes of a tensor."""
    return embedding.element_size() * embedding.numel()


@dataclass(kw_only=True)
class EmbeddingResult:
    """Container for a cached vision embedding.

    Attributes:
        embedding: the ViT output tensor.
        mm_meta: optional multimodal item metadata (e.g. ``MMItemMeta`` from
            ``sglang.srt.disaggregation.encode.encode_data``).  Stored as
            ``Any`` to keep ``mem_cache`` dependency-free from the EPD layer.
            The EPD Encoder populates this when writing a ViT result into the
            cache so that, on a cache hit in a later turn of a multi-turn
            conversation, the per-item metadata (pad_value / grid_thw /
            num_tokens) can be reused without re-running ``image_transform``
            or re-deriving pad_value.  ``None`` for non-EPD callers.
    """
    embedding: torch.Tensor
    mm_meta: Any = None


class MultiModalStaticCache:
    """
    Server-level LRU cache for multimodal (vision) embeddings.

    Implements an OrderedDict-based LRU eviction policy.
    When max_size is exceeded, the least recently used entry is evicted.

    Keys: integer hash of pixel_values (computed by hash_feature())
    Values: EmbeddingResult containing the embedding tensor

    Controlled by:
        - SGLANG_VLM_CACHE_SIZE_MB: maximum cache size in MB (default: 100)
        - init_mm_embedding_cache(size_bytes): called at Scheduler startup
    """

    def __init__(self, max_size: int):
        """
        Args:
            max_size: Maximum cache size in bytes. Set to 0 to disable caching.
        """
        self.max_size = max_size
        self.mm_cache: OrderedDict[int, EmbeddingResult] = OrderedDict()
        self.current_size = 0

    @staticmethod
    def combine_hashes(mm_hashes: List[int]) -> Optional[int]:
        """Combine multiple item hashes into a single combined hash."""
        if not mm_hashes:
            return None
        return hash(tuple(mm_hashes))

    def get(
        self, mm_hashes: List[int], combined_hash: Optional[int] = None
    ) -> Optional[EmbeddingResult]:
        """
        Get cached embedding by a list of item hashes (combined hash lookup).

        Args:
            mm_hashes: List of individual item hashes.
            combined_hash: Optional pre-computed combined hash.

        Returns:
            EmbeddingResult if found, None otherwise.
        """
        key = combined_hash if combined_hash is not None else self.combine_hashes(mm_hashes)
        if key is None:
            return None
        embedding = self.mm_cache.get(key)
        if embedding is not None:
            self.mm_cache.move_to_end(key)
        return embedding

    def get_single(self, mm_hash: int) -> Optional[EmbeddingResult]:
        """
        Get a single cached embedding by its hash (no combine_hashes).
        Args:
            mm_hash: Hash of a single multimodal item.

        Returns:
            EmbeddingResult if found (and moves to end as most-recently-used), None otherwise.
        """
        embedding = self.mm_cache.get(mm_hash)
        if embedding is not None:
            self.mm_cache.move_to_end(mm_hash)
        return embedding

    def set(
        self,
        mm_hash: int,
        embedding: EmbeddingResult,
    ) -> bool:
        """
        Store an embedding in the cache with LRU eviction.

        Args:
            mm_hash: Hash key for this embedding.
            embedding: EmbeddingResult to cache.

        Returns:
            True if successfully stored, False if the embedding is too large for cache.
        """
        if self.max_size <= 0:
            return False

        assert isinstance(embedding, EmbeddingResult), (
            f"Expected EmbeddingResult, got {type(embedding)}"
        )

        if mm_hash in self.mm_cache:
            self.mm_cache.move_to_end(mm_hash)
            return True

        data_size = _get_tensor_size(embedding.embedding)
        if data_size > self.max_size:
            return False  # Single item too large for cache

        # LRU eviction: remove oldest entries until we have space
        while self.current_size + data_size > self.max_size:
            if not self.mm_cache:
                return False
            lru_hash, lru_embedding = self.mm_cache.popitem(last=False)
            self.current_size -= _get_tensor_size(lru_embedding.embedding)

        self.mm_cache[mm_hash] = embedding
        self.current_size += data_size
        return True

    def has(self, mm_hash: int) -> bool:
        """Check if hash is in cache without updating LRU order."""
        return mm_hash in self.mm_cache

    def free(self, mm_hash: int) -> bool:
        """
        Remove a specific entry from the cache.

        Returns:
            True if found and removed, False if not found.
        """
        if mm_hash not in self.mm_cache:
            return False
        old_embedding = self.mm_cache.pop(mm_hash)
        self.current_size -= _get_tensor_size(old_embedding.embedding)
        return True

    def clear(self):
        """Remove all entries from the cache."""
        self.mm_cache.clear()
        self.current_size = 0

    def __len__(self) -> int:
        return len(self.mm_cache)

    def available_size(self) -> int:
        return len(self.mm_cache)

    def __repr__(self) -> str:
        return (
            f"MultiModalStaticCache(entries={len(self)}, "
            f"size={self.current_size / 1024 / 1024:.1f}MB / "
            f"{self.max_size / 1024 / 1024:.1f}MB)"
        )
