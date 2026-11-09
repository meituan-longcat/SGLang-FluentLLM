"""Two-tier vision-embedding cache for the EPD Encoder server: L1 GPU VRAM + L2 Host DRAM.

Design aligned with TokenSpeed's TieredEmbeddingCache (tokenspeed.runtime.cache.embedding_cache),
adapted for FluentLLM's conventions:

- Keys: int hash (same as MultiModalStaticCache, derived from pixel_values).
- Values: EmbeddingResult (from multimodal_cache.py) containing a single embedding tensor.
- L1 (GPU): OrderedDict-based LRU, byte-bounded. Stores device-resident EmbeddingResult.
- Host DRAM: Same LRU structure, stores CPU-resident RVQ results directly and,
  when ``enable_l2=True``, also catches L1 victims via the ``on_evict`` hook.
- Tiers are **exclusive**: a key lives in exactly one tier at any time.

Lifecycle:
- ``put(key, EmbeddingResult)`` → always lands in L1 (GPU), drops stale L2 copy.
- ``get(key)`` → L1 hit returns device tensor; L2 hit promotes to L1 (host→device copy).
- ``put_host/get_host`` → stores/reads CPU RVQ directly in Host DRAM without promotion.
- L1 eviction → demotes victim to L2 (device→host copy).
- L2 eviction → true drop (no L3 in phase 1).

Why not just use MultiModalStaticCache?
- MultiModalStaticCache is single-tier, always GPU-resident, sized by SGLANG_VLM_CACHE_SIZE_MB.
- The Encoder server needs *much* larger capacity (many unique images across requests).
- Host DRAM is ~10x cheaper per GB than VRAM; a two-tier cache captures cold images
  that would otherwise require re-running ViT.

Performance expectations:
- L1 hit: 0 ms (tensor already on device, ViT skipped).
- L2 hit: ~1-2 ms (host→device DMA for ~7 MB/image bf16 embedding).
- L1 miss + L2 miss: full ViT forward (~10-50 ms depending on resolution).
"""

from __future__ import annotations

import collections
import logging
from collections.abc import Callable, Hashable
from typing import Optional

import torch

from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult

logger = logging.getLogger(__name__)


def _get_tensor_nbytes(embedding: torch.Tensor) -> int:
    """Return size in bytes of a tensor."""
    return embedding.element_size() * embedding.numel()


def _embedding_result_nbytes(result: EmbeddingResult) -> int:
    """Return size in bytes of an EmbeddingResult's embedding tensor."""
    return _get_tensor_nbytes(result.embedding)


# ═══════════════════════════════════════════════════════════════════════════
#  Single-tier LRU cache (internal building block)
# ═══════════════════════════════════════════════════════════════════════════


class _LRUTier:
    """Bytes-bounded LRU cache tier, keyed by content hash.

    Values are ``EmbeddingResult`` objects; the tier itself is agnostic to
    device placement (GPU or CPU). The ``on_evict`` hook fires when an entry
    is dropped by capacity overflow, enabling demotion to a lower tier.

    This is the FluentLLM counterpart of TokenSpeed's ``EmbeddingCache``.
    """

    def __init__(
        self,
        capacity_bytes: int,
        on_evict: Callable[[Hashable, EmbeddingResult, int], None] | None = None,
    ):
        if capacity_bytes < 0:
            raise ValueError(f"capacity_bytes must be >= 0, got {capacity_bytes}")
        self.capacity_bytes = capacity_bytes
        self._on_evict = on_evict
        # key → (EmbeddingResult, nbytes); ordered by recency (oldest first)
        self._store: collections.OrderedDict[Hashable, tuple[EmbeddingResult, int]] = (
            collections.OrderedDict()
        )
        self._cur_bytes = 0
        self.hits = 0
        self.misses = 0

    def get(self, key: Hashable) -> Optional[EmbeddingResult]:
        """Lookup by key; returns EmbeddingResult on hit, None on miss.

        Hit: moves key to MRU position and increments hit counter.
        Miss: increments miss counter.
        """
        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
            return None
        self._store.move_to_end(key)
        self.hits += 1
        return entry[0]

    def put(self, key: Hashable, value: EmbeddingResult, nbytes: int) -> None:
        """Store an EmbeddingResult with known byte size.

        If the key already exists, it is replaced (in-place update, no on_evict).
        If the item is larger than the entire capacity, it is silently dropped.
        Otherwise, LRU eviction fires until there is room.
        """
        if nbytes < 0:
            raise ValueError(f"nbytes must be >= 0, got {nbytes}")
        if nbytes > self.capacity_bytes:
            return  # Item too large for this tier

        existing = self._store.pop(key, None)
        if existing is not None:
            self._cur_bytes -= existing[1]
        self._store[key] = (value, nbytes)
        self._cur_bytes += nbytes
        self._evict()

    def pop(self, key: Hashable) -> Optional[tuple[EmbeddingResult, int]]:
        """Remove key and return (EmbeddingResult, nbytes), or None if absent.

        Structural removal: does NOT count as hit/miss and does NOT fire on_evict.
        Used for L2→L1 promotion (caller takes ownership).
        """
        existing = self._store.pop(key, None)
        if existing is None:
            return None
        self._cur_bytes -= existing[1]
        return existing

    def _evict(self) -> None:
        """Evict LRU entries until within capacity."""
        while self._cur_bytes > self.capacity_bytes and self._store:
            key, (value, nbytes) = self._store.popitem(last=False)
            self._cur_bytes -= nbytes
            if self._on_evict is not None:
                self._on_evict(key, value, nbytes)

    def __contains__(self, key: Hashable) -> bool:
        return key in self._store

    def __len__(self) -> int:
        return len(self._store)

    @property
    def current_bytes(self) -> int:
        return self._cur_bytes


# ═══════════════════════════════════════════════════════════════════════════
#  Default device ↔ host copy functions
# ═══════════════════════════════════════════════════════════════════════════


def _embedding_to_host(result: EmbeddingResult) -> EmbeddingResult:
    """L1→L2 demote: copy embedding from GPU to host memory.

    Returns a new EmbeddingResult with .embedding.cpu() (pin_memory for
    faster subsequent .to(device) promotion).  ``mm_meta`` (if any) is
    carried over so multi-turn meta reuse survives tier demotion.
    """
    return EmbeddingResult(
        embedding=result.embedding.cpu().pin_memory(),
        mm_meta=result.mm_meta,
    )


def _embedding_to_device(result: EmbeddingResult, device) -> EmbeddingResult:
    """L2→L1 promote: copy host-resident embedding back to device.

    ``mm_meta`` (if any) is carried over so the promoted entry keeps the
    cached per-item metadata for multi-turn reuse.
    """
    return EmbeddingResult(
        embedding=result.embedding.to(device, non_blocking=True),
        mm_meta=result.mm_meta,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  TieredEmbeddingCache: GPU L1 + Host DRAM L2
# ═══════════════════════════════════════════════════════════════════════════


class TieredEmbeddingCache:
    """Two-tier vision-embedding cache for the EPD Encoder server.

    L1 (GPU VRAM): fast lookup, device-resident tensors. Sized by ``gpu_size_mb``.
    Host DRAM: stores host-only RVQ and, when ``enable_l2=True``, catches L1
    victims. Both uses share the ``host_size_mb`` byte budget.

    Exposes the same ``get`` / ``put`` / ``has`` surface as MultiModalStaticCache
    so EncodeWorker can use either interchangeably.

    Tier exclusivity: a key lives in exactly one tier at any time.
    - ``put`` always lands in L1, drops any stale L2 copy.
    - ``put_host`` always lands in Host DRAM and is never promoted by ``get_host``.
    - L1 eviction → demotes victim to L2 (device→host copy via ``to_host``).
    - L2 hit → promotes entry to L1 (host→device copy via ``to_device``),
      removing it from L2.
    - L2 eviction → true drop (no distributed L3 in phase 1).

    The device↔host copies are injectable for unit testing without a GPU.

    Args:
        gpu_size_mb: L1 GPU capacity in megabytes.
        host_size_mb: Host DRAM capacity in megabytes. Zero disables both
                      host-only RVQ caching and device L2 demotion.
        enable_l2: Whether device embeddings may demote/promote through the
                   host tier. Host-only RVQ access is independent of this flag.
        device: Target CUDA device for L2→L1 promotion (default: cuda:0).
        to_host: Injected device→host copy function.
        to_device: Injected host→device copy function.
    """

    def __init__(
        self,
        gpu_size_mb: int = 1024,
        host_size_mb: int = 8192,
        enable_l2: bool = True,
        *,
        device: Optional[str] = None,
        to_host: Optional[Callable[[EmbeddingResult], EmbeddingResult]] = None,
        to_device: Optional[Callable[[EmbeddingResult], EmbeddingResult]] = None,
    ):
        self._device = device or "cuda:0"
        l1_capacity = gpu_size_mb * 1024 * 1024  # MB → bytes

        # The host tier is always available when it has capacity. Device
        # embedding demotion/promotion remains controlled by enable_l2, while
        # compact RVQ indices use this tier directly and never enter L1.
        self._host_enabled = host_size_mb > 0
        self._enable_l2 = enable_l2 and self._host_enabled
        self.l2 = (
            _LRUTier(host_size_mb * 1024 * 1024)
            if self._host_enabled
            else None
        )
        if self._enable_l2:
            self._to_host = to_host or _embedding_to_host
            self._to_device = to_device or (
                lambda r: _embedding_to_device(r, self._device)
            )
            # L1 demotes evictions into L2
            self.l1 = _LRUTier(l1_capacity, on_evict=self._demote)
        else:
            self._to_host = None
            self._to_device = None
            # Single-tier mode: L1 evictions are true drops
            self.l1 = _LRUTier(l1_capacity)

        # Statistics
        self.l1_hits = 0
        self.l2_hits = 0
        self.misses = 0
        self.promotions = 0  # L2 → L1 promotions
        self.demotions = 0   # L1 → L2 demotions
        self.host_hits = 0
        self.host_misses = 0
        self.host_puts = 0

    # ─── Public API (compatible with MultiModalStaticCache) ──────────────

    def get(self, key: int) -> Optional[EmbeddingResult]:
        """Lookup by content hash.

        - L1 hit: return device-resident EmbeddingResult (0 ms ViT skip).
        - L2 hit: promote to L1 (host→device copy), return device result.
        - Miss: return None (caller must run ViT and ``put`` the result).
        """
        # L1 lookup
        result = self.l1.get(key)
        if result is not None:
            self.l1_hits += 1
            return result

        # L2 lookup (if enabled)
        if self._enable_l2:
            promoted = self.l2.pop(key)
            if promoted is not None:
                host_result, nbytes = promoted
                device_result = self._to_device(host_result)
                self.l2_hits += 1
                self.promotions += 1
                # Re-home as MRU in L1. May evict colder L1 entries (which
                # demote to L2); the just-promoted key is MRU so it is never
                # the victim.
                self.l1.put(key, device_result, nbytes)
                return device_result

        # Cache miss
        self.misses += 1
        return None

    def get_single(self, key: int) -> Optional[EmbeddingResult]:
        """Alias for ``get`` — compatible with MultiModalStaticCache API."""
        return self.get(key)

    def put(self, key: int, result: EmbeddingResult) -> bool:
        """Store an EmbeddingResult in L1 (GPU).

        Args:
            key: Content hash of the image.
            result: EmbeddingResult with device-resident embedding tensor.

        Returns:
            True if successfully stored, False if the item is too large for L1.
        """
        nbytes = _embedding_result_nbytes(result)
        if nbytes > self.l1.capacity_bytes:
            return False  # Item too large for L1

        # Drop stale L2 copy so tiers stay exclusive
        if self.l2 is not None:
            self.l2.pop(key)

        self.l1.put(key, result, nbytes)
        return True

    def get_host(self, key: int) -> Optional[EmbeddingResult]:
        """Return a host-only entry without promoting it to the device.

        RVQ indices are compact CPU/ZMQ transport data and have no Encoder-side
        device consumer. Keeping them in this shared host tier avoids both an
        unnecessary H2D on cache hit and cross-thread device storage lifetime.
        """
        if self.l2 is None:
            self.host_misses += 1
            return None
        result = self.l2.get(key)
        if result is None:
            self.host_misses += 1
            return None
        self.host_hits += 1
        return result

    def put_host(self, key: int, result: EmbeddingResult) -> bool:
        """Store a CPU-resident result directly in the shared host tier."""
        if result.embedding.device.type != "cpu":
            raise ValueError(
                "put_host requires a CPU-resident embedding, got "
                f"device={result.embedding.device}"
            )
        if self.l2 is None:
            return False
        nbytes = _embedding_result_nbytes(result)
        if nbytes > self.l2.capacity_bytes:
            return False

        # Preserve tier exclusivity if a key was previously stored in L1.
        self.l1.pop(key)
        self.l2.put(key, result, nbytes)
        if key not in self.l2:
            return False
        self.host_puts += 1
        return True

    def set(self, key: int, result: EmbeddingResult) -> bool:
        """Alias for ``put`` — compatible with MultiModalStaticCache API."""
        return self.put(key, result)

    def has(self, key: int) -> bool:
        """Check if key is in cache without updating LRU order."""
        if key in self.l1:
            return True
        if self.l2 is not None and key in self.l2:
            return True
        return False

    def free(self, key: int) -> bool:
        """Remove a specific entry from the cache.

        Returns True if found and removed, False if not found.
        """
        removed = self.l1.pop(key)
        if removed is not None:
            return True
        if self.l2 is not None:
            removed = self.l2.pop(key)
            if removed is not None:
                return True
        return False

    def clear(self) -> None:
        """Remove all entries from both tiers."""
        self.l1._store.clear()
        self.l1._cur_bytes = 0
        if self.l2 is not None:
            self.l2._store.clear()
            self.l2._cur_bytes = 0

    # ─── Demotion hook ──────────────────────────────────────────────────

    def _demote(self, key: Hashable, value: EmbeddingResult, nbytes: int) -> None:
        """L1 eviction hook: stash the victim in host DRAM instead of dropping it.

        Skips the wasted device→host copy for a victim that L2 would just drop
        (item larger than L2 capacity).
        """
        if not self._enable_l2:
            return
        if nbytes > self.l2.capacity_bytes:
            return
        host_value = self._to_host(value)
        self.l2.put(key, host_value, nbytes)
        if key in self.l2:
            self.demotions += 1

    # ─── Statistics ──────────────────────────────────────────────────────

    @property
    def hits(self) -> int:
        """Total hits across both tiers."""
        return self.l1_hits + self.l2_hits + self.host_hits

    @property
    def hit_rate(self) -> float:
        """Overall hit rate (L1 + L2) / (hits + misses)."""
        total = self.hits + self.misses + self.host_misses
        return self.hits / total if total > 0 else 0.0

    def __len__(self) -> int:
        """Total entries across both tiers."""
        if self.l2 is not None:
            return len(self.l1) + len(self.l2)
        return len(self.l1)

    def __contains__(self, key: Hashable) -> bool:
        return self.has(key)

    def available_size(self) -> int:
        """Number of entries (compatible with MultiModalStaticCache API)."""
        return len(self)

    def __repr__(self) -> str:
        l1_mb = self.l1.current_bytes / 1024 / 1024
        l1_cap_mb = self.l1.capacity_bytes / 1024 / 1024
        parts = [
            f"TieredEmbeddingCache(",
            f"L1={l1_mb:.1f}/{l1_cap_mb:.1f}MB({len(self.l1)} entries)",
        ]
        if self.l2 is not None:
            l2_mb = self.l2.current_bytes / 1024 / 1024
            l2_cap_mb = self.l2.capacity_bytes / 1024 / 1024
            parts.append(
                f", Host={l2_mb:.1f}/{l2_cap_mb:.1f}MB({len(self.l2)} entries)"
            )
        parts.append(
            f", hits=L1:{self.l1_hits}/L2:{self.l2_hits}/miss:{self.misses}"
            f"/host:{self.host_hits}/host_miss:{self.host_misses}"
            f", promo:{self.promotions}/demo:{self.demotions}"
            f"/host_put:{self.host_puts})"
        )
        return "".join(parts)
