"""
Multimodal utility functions for embedding, hashing, and token padding.

Key responsibilities:
1. hash_feature(): compute hash of pixel_values for RadixCache prefix matching
   and ViT Embedding Cache lookup.
2. Padding patterns: replace image/audio placeholder tokens with pad_values in input_ids.
3. Chunked prefill embedding: handle Scheduler-cut chunks that may slice through images.
4. embed_mm_inputs(): fuse vision and text embeddings.
5. general_mm_embed_routine(): high-level entry point called from model forward().
6. run_dp_sharded_mrope_vision_model(): DP-sharded vision model for Qwen2.5-VL / Qwen3-VL.

References:
    - sglang/python/sglang/srt/managers/mm_utils.py (upstream)
    - sglang/python/sglang/srt/multimodal/mm_utils.py (upstream)
"""

import copy
import hashlib
import itertools
import math
import os
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

from sglang.srt.managers.scheduling.multimodal_data import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.utils.common import flatten_nested_list as _flatten_nested_list
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult, MultiModalStaticCache
from sglang.srt.utils import get_colorful_logger, is_npu
from sglang.srt.multimodal.gpu_hash_ops import gpu_tensor_hash
from sglang.srt.layers.utils import CP_METADATA, cp_split_and_rebuild_data
from sglang.srt.distributed import get_attn_sp_token_slice_pos

logger = get_colorful_logger(__name__)

# Global ViT Embedding Cache instance (process-level singleton)
embedding_cache: Optional[MultiModalStaticCache] = None

DataEmbeddingFunc = Callable[[List[MultimodalDataItem]], torch.Tensor]


def init_mm_embedding_cache(max_size: int = 0):
    """
    Initialize the global ViT Embedding Cache.

    Called at Scheduler startup with size from SGLANG_VLM_CACHE_SIZE_MB env var.

    Args:
        max_size: Maximum cache size in bytes. 0 disables caching.
    """
    global embedding_cache
    embedding_cache = MultiModalStaticCache(max_size)
    if max_size > 0:
        logger.info(f"ViT Embedding Cache initialized: {max_size / 1024 / 1024:.0f}MB")


def clear_mm_embedding_cache() -> int:
    """Clear all entries from the global ViT embedding cache.

    Returns the number of entries that were dropped (0 if cache is uninitialized
    or empty). Called from Scheduler.flush_cache so /flush_cache also drops cached
    vision embeddings — required when verifying ViT precision, where a cache hit
    would skip the encoder and leave debug dumps empty.
    """
    global embedding_cache
    if embedding_cache is None:
        return 0
    n = len(embedding_cache)
    embedding_cache.clear()
    if n > 0:
        logger.info(f"ViT Embedding Cache cleared: dropped {n} entries")
    return n


# ============================================================
# Hash utilities
# ============================================================

def data_hash(data: bytes) -> int:
    """Compute SHA256-based hash of raw bytes."""
    hash_bytes = hashlib.sha256(data).digest()[:8]
    return int.from_bytes(hash_bytes, byteorder="big", signed=False)


def tensor_hash(tensor_or_list) -> int:
    """
    Hash a tensor or a list of tensors.
    GPU tensors use the fast Triton kernel; CPU tensors use SHA256.
    """
    if isinstance(tensor_or_list, list):
        flat = _flatten_nested_list(tensor_or_list)
        tensors = [
            x.flatten() if isinstance(x, torch.Tensor) else x for x in flat
        ]
        if any(isinstance(t, torch.Tensor) and t.is_cuda for t in tensors):
            gpu_tensors = [t for t in tensors if isinstance(t, torch.Tensor)]
            tensor = torch.cat(gpu_tensors)
            return gpu_tensor_hash(tensor.cuda())
        hasher = hashlib.sha256()
        for t in tensors:
            if isinstance(t, torch.Tensor):
                t = t.detach().contiguous()
                hasher.update(memoryview(t.view(torch.uint8).numpy()))
        hash_bytes = hasher.digest()[:8]
        return int.from_bytes(hash_bytes, byteorder="big", signed=False)

    tensor = tensor_or_list
    if isinstance(tensor, torch.Tensor):
        if tensor.is_cuda:
            return gpu_tensor_hash(tensor)
        tensor = tensor.detach().contiguous()
        hasher = hashlib.sha256()
        hasher.update(memoryview(tensor.view(torch.uint8).numpy()))
        hash_bytes = hasher.digest()[:8]
        return int.from_bytes(hash_bytes, byteorder="big", signed=False)

    raise TypeError(f"Unsupported type for tensor_hash: {type(tensor)}")


def hash_feature(f) -> int:
    """
    Compute a hash value for multimodal feature data.

    Supports:
    - torch.Tensor (GPU: Triton kernel, CPU: SHA256)
    - numpy.ndarray
    - list of tensors
    - bytes/raw data
    """
    if isinstance(f, list):
        if f and isinstance(f[0], torch.Tensor):
            return tensor_hash(f)
        flat = _flatten_nested_list(f)
        return data_hash(bytes(str(flat), "utf-8"))
    elif isinstance(f, np.ndarray):
        arr = np.ascontiguousarray(f)
        hasher = hashlib.sha256()
        hasher.update(memoryview(arr))
        hash_bytes = hasher.digest()[:8]
        return int.from_bytes(hash_bytes, byteorder="big", signed=False)
    elif isinstance(f, torch.Tensor):
        return tensor_hash(f)
    elif isinstance(f, bytes):
        return data_hash(f)
    return data_hash(bytes(str(f), "utf-8"))


# ============================================================
# Token padding patterns (pad_input_ids)
# ============================================================

class MultiModalityDataPaddingPattern:
    """
    Interface for replacing multimodal placeholder tokens in input_ids with pad_values.

    pad_values encode image content hash, enabling RadixCache prefix matching:
    same image → same pad_value → same prefix → KV cache hit.
    """

    @abstractmethod
    def pad_input_tokens(
        self, input_ids: List[int], mm_inputs: MultimodalInputs
    ) -> List[int]:
        """Replace multimodal placeholder tokens in input_ids with pad_values."""
        pass


class MultiModalityDataPaddingPatternTokenPairs(MultiModalityDataPaddingPattern):
    """
    Token-pairs pattern: multimodal data enclosed by start/end token pairs.
    Example: <image> [pad_value × N] </image>
    Used by LLaVA-style models.
    """

    def __init__(
        self,
        data_token_pairs: Optional[List[Tuple[int, int]]],
        data_start_token_ids: Optional[List[int]] = None,
    ) -> None:
        self.data_token_id_pairs = data_token_pairs
        self.data_start_token_ids = data_start_token_ids or [
            s for s, _e in data_token_pairs
        ]

    def pad_input_tokens(
        self, input_ids: List[int], mm_inputs: MultimodalInputs
    ) -> List[int]:
        pad_values = [item.pad_value for item in mm_inputs.mm_items]
        data_token_pairs = self.data_token_id_pairs
        mm_inputs.data_offsets = []

        if not data_token_pairs or data_token_pairs[0][0] is None:
            logger.warning_once(
                "No data_token_pairs provided, RadixAttention might be influenced."
            )
            return input_ids

        start_token_ids = {s for s, _e in data_token_pairs}
        end_token_ids = {e for _s, e in data_token_pairs}

        padded_ids = []
        last_idx = 0
        data_idx = -1

        start_indices = [i for i, x in enumerate(input_ids) if x in start_token_ids]
        end_indices = [i for i, x in enumerate(input_ids) if x in end_token_ids]

        if len(start_indices) != len(end_indices):
            return input_ids

        for start_idx, end_idx in zip(start_indices, end_indices):
            padded_ids.extend(input_ids[last_idx : start_idx + 1])
            if input_ids[start_idx] in self.data_start_token_ids:
                data_idx += 1
                mm_inputs.data_offsets.append(start_idx)
            if data_idx >= len(pad_values):
                data_idx = len(pad_values) - 1
            num_tokens = end_idx - start_idx - 1
            pad_value = pad_values[data_idx]
            padded_ids.extend([pad_value] * num_tokens)
            last_idx = end_idx

        padded_ids.extend(input_ids[last_idx:])
        assert len(input_ids) == len(padded_ids), "Length validation fails"
        return padded_ids


class MultiModalityDataPaddingPatternMultimodalTokens(MultiModalityDataPaddingPattern):
    """
    Multimodal-tokens pattern: data tokens are repetitions of a single token.
    Example: <|image_pad|> × N → [pad_value] × N
    Used by Qwen-VL and Kimi2.5.
    """

    def pad_input_tokens(
        self, input_ids: List[int], mm_inputs: MultimodalInputs
    ) -> List[int]:
        if not input_ids or not mm_inputs.mm_items:
            return input_ids

        # Pad with list slice assignment.  Converting a long id list to a
        # tensor and back holds the GIL for ~75 ms at 680K tokens.
        padded_ids = list(input_ids)
        items_by_modality = defaultdict(list)
        for item in mm_inputs.mm_items:
            items_by_modality[item.modality].append(item)

        token_id_map = {
            Modality.IMAGE: mm_inputs.im_token_id,
            Modality.AUDIO: mm_inputs.audio_token_id,
            Modality.VIDEO: mm_inputs.video_token_id,
        }

        for modality, items in items_by_modality.items():
            token_id = token_id_map.get(modality)
            if not items or token_id is None:
                continue
            for item in items:
                for offset in (item.offsets or []):
                    # Same clamping as tensor slicing; never extends the list.
                    start, stop, _ = slice(offset[0], offset[1] + 1).indices(
                        len(padded_ids)
                    )
                    if start < stop:
                        padded_ids[start:stop] = [int(item.pad_value)] * (stop - start)

        return padded_ids


# ============================================================
# Embedding utilities
# ============================================================

def _move_items_to_device(
    items: List[MultimodalDataItem], device: torch.device
) -> None:
    """Move item features to the target device (in-place, non-blocking)."""
    for item in items:
        if isinstance(item.feature, torch.Tensor) and item.feature.device != device:
            item.feature = item.feature.to(device, non_blocking=True)


@dataclass
class PerImageRequestInfo:
    """Metadata for a single request using the per-image encoding path."""

    req_idx: int
    items: List[MultimodalDataItem]
    items_offset: List[Tuple[int, int]]
    extend_prefix_len: int
    extend_seq_len: int
    overlapping: List[Tuple[int, MultimodalDataItem, int, int]] = field(
        default_factory=list
    )


def _batch_encode_per_image_misses(
    data_embedding_func: DataEmbeddingFunc,
    per_image_requests: List[PerImageRequestInfo],
    device: torch.device,
) -> Dict[int, torch.Tensor]:
    """
    Collect cache misses across ALL per-image requests, deduplicate by hash,
    encode in a single ViT call, and populate the cache.

    Returns:
        hash_to_embedding: mapping from item.hash to its full embedding tensor.
    """
    global embedding_cache

    unique_misses: Dict[int, Tuple[MultimodalDataItem, int]] = {}
    hash_to_embedding: Dict[int, torch.Tensor] = {}

    # Phase 1a: find overlapping items per request and collect cache misses
    for req_info in per_image_requests:
        chunk_start = req_info.extend_prefix_len
        chunk_end = chunk_start + req_info.extend_seq_len  # exclusive
        overlapping = []
        if req_info.extend_seq_len > 0:
            for idx, (item, (start, end)) in enumerate(
                zip(req_info.items, req_info.items_offset)
            ):
                if end >= chunk_start and start < chunk_end:
                    overlapping.append((idx, item, start, end))
        req_info.overlapping = overlapping

        for _idx, item, start, end in overlapping:
            if item.hash in hash_to_embedding:
                continue
            cached = embedding_cache.get_single(item.hash) if embedding_cache is not None else None
            if cached is not None:
                hash_to_embedding[item.hash] = cached.embedding
            elif item.hash not in unique_misses:
                token_count = end - start + 1
                unique_misses[item.hash] = (item, token_count)

    # Phase 1b: single ViT call for all unique cache misses
    if unique_misses:
        ordered_hashes = list(unique_misses.keys())
        miss_items = [unique_misses[h][0] for h in ordered_hashes]
        token_counts = [unique_misses[h][1] for h in ordered_hashes]

        _move_items_to_device(miss_items, device)
        all_miss_embedding = data_embedding_func(miss_items)
        all_miss_embedding = all_miss_embedding.reshape(
            -1, all_miss_embedding.shape[-1]
        )

        split_embeddings = torch.split(all_miss_embedding, token_counts, dim=0)
        for h, emb in zip(ordered_hashes, split_embeddings):
            if embedding_cache is not None and embedding_cache.max_size > 0:
                embedding_cache.set(h, EmbeddingResult(embedding=emb))
            # Keep a local ref (no extra GPU memory) so assembly never fails due to LRU eviction.
            hash_to_embedding[h] = emb

    return hash_to_embedding


def _assemble_per_image_chunk(
    overlapping: List[Tuple[int, MultimodalDataItem, int, int]],
    hash_to_embedding: Dict[int, torch.Tensor],
    extend_prefix_len: int,
    extend_seq_len: int,
) -> Optional[torch.Tensor]:
    """
    Assemble the chunk embedding for one request from pre-computed embeddings.
    All overlapping items must already have their embeddings in hash_to_embedding.
    """
    if not overlapping:
        return None

    chunk_start = extend_prefix_len
    chunk_end = extend_prefix_len + extend_seq_len  # exclusive

    chunk_slices = []
    for _idx, item, start, end in overlapping:
        emb = hash_to_embedding[item.hash]  # shape: (end - start + 1, hidden)
        overlap_start = max(start, chunk_start)
        overlap_end = min(end, chunk_end - 1)  # inclusive
        local_start = overlap_start - start
        local_end = overlap_end - start + 1  # exclusive for slicing
        chunk_slices.append(emb[local_start:local_end])

    return torch.cat(chunk_slices, dim=0)


def _get_chunked_embedding_full(
    data_embedding_func: DataEmbeddingFunc,
    embedding_items_per_req: List[MultimodalDataItem],
    items_offset: List[Tuple[int, int]],
    extend_prefix_len: int,
    extend_seq_len: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Fallback: encode all items at once, cache combined result, extract chunk.
    Used when items are bundled (not split per-image).
    """
    global embedding_cache

    item_hashes = [item.hash for item in embedding_items_per_req]
    combined_hash = MultiModalStaticCache.combine_hashes(item_hashes)
    embedding_per_req = None

    if embedding_cache is not None and embedding_cache.max_size > 0 and combined_hash is not None:
        result = embedding_cache.get([combined_hash])
        if result is not None:
            embedding_per_req = result.embedding

    if embedding_per_req is None:
        _move_items_to_device(embedding_items_per_req, device)
        embedding = data_embedding_func(embedding_items_per_req)
        embedding_per_req = embedding
        if embedding_cache is not None and embedding_cache.max_size > 0 and combined_hash is not None:
            embedding_cache.set(combined_hash, EmbeddingResult(embedding=embedding))

    embedding_per_req = embedding_per_req.reshape(-1, embedding_per_req.shape[-1])
    chunk_start = extend_prefix_len
    chunk_end = extend_prefix_len + extend_seq_len

    start_index, end_index = 0, 0
    for start, end in items_offset:
        if chunk_start >= start and chunk_start <= end:
            start_index += chunk_start - start
        elif chunk_start > end:
            start_index += end - start + 1
        if chunk_end - 1 >= start and chunk_end - 1 <= end:
            end_index += chunk_end - 1 - start + 1
        elif chunk_end - 1 > end:
            end_index += end - start + 1

    return embedding_per_req[start_index:end_index]


def _get_precomputed_embedding(
    items: List[MultimodalDataItem],
    items_size: List[int],
    prefix_length: List[int],
    extend_length: List[int],
    items_offset_list: List[List[Tuple[int, int]]],
) -> Optional[torch.Tensor]:
    """
    If all items have precomputed_embeddings, return their concatenation.
    Returns None if none have precomputed embeddings.
    """
    precomputed_embeddings = []
    max_iterations = min(len(items_size) - 1, len(prefix_length))

    for i in range(max_iterations):
        if items_size[i] == items_size[i + 1]:
            continue

        items_per_req = items[items_size[i] : items_size[i + 1]]
        extend_len = extend_length[i] if i < len(extend_length) else 0
        items_offset = items_offset_list[i]

        if any(item.precomputed_embeddings is None for item in items_per_req):
            chunk = None
        else:
            req_embeddings = torch.cat(
                [item.precomputed_embeddings for item in items_per_req]
            )
            req_embeddings = req_embeddings.reshape(-1, req_embeddings.shape[-1])
            chunk_start = prefix_length[i]
            chunk_end = chunk_start + extend_len
            s_idx, e_idx = 0, 0
            for start, end in items_offset:
                if chunk_start >= start and chunk_start <= end:
                    s_idx += chunk_start - start
                elif chunk_start > end:
                    s_idx += end - start + 1
                if chunk_end - 1 >= start and chunk_end - 1 <= end:
                    e_idx += chunk_end - 1 - start + 1
                elif chunk_end - 1 > end:
                    e_idx += end - start + 1
            chunk = req_embeddings[s_idx:e_idx]

        precomputed_embeddings.append(chunk)

    has_precomputed = any(f is not None for f in precomputed_embeddings)
    if not has_precomputed:
        return None
    if not all(f is not None for f in precomputed_embeddings):
        raise NotImplementedError("MM inputs where only some items are precomputed.")
    result = torch.cat(precomputed_embeddings)
    return result.reshape(-1, result.shape[-1])


def _get_chunked_prefill_embedding(
    data_embedding_func: DataEmbeddingFunc,
    embedding_items: List[MultimodalDataItem],
    items_size: List[int],
    prefix_length: List[int],
    extend_length: List[int],
    items_offset_list: List[List[Tuple[int, int]]],
    input_ids: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """
    Chunked prefill embedding: encode items across all requests and extract
    per-request chunks. Images from all requests are batched into a single
    ViT call for efficiency.

    Per-image requests (items with a single offset each) are batched together
    across the entire batch for a single ViT call, while full-path requests
    (EVS or multi-offset items) are handled per-request as before.
    """
    device = input_ids.device
    max_iterations = min(len(items_size) - 1, len(prefix_length))

    # Phase 0: classify requests into per-image vs full path
    per_image_requests: List[PerImageRequestInfo] = []  # batched ViT encoding
    full_path_requests: List[PerImageRequestInfo] = []  # per-request encoding

    for i in range(max_iterations):
        if items_size[i] == items_size[i + 1]:
            continue

        embedding_items_per_req = embedding_items[items_size[i] : items_size[i + 1]]
        items_offset = items_offset_list[i]
        assert items_offset is not None

        extend_prefix_len = prefix_length[i]
        extend_seq_len = extend_length[i] if i < len(extend_length) else 0

        # Skip if all items already fully prefilled
        if all(offset_end < prefix_length[i] for _, offset_end in items_offset):
            continue

        req_info = PerImageRequestInfo(
            req_idx=i,
            items=embedding_items_per_req,
            items_offset=items_offset,
            extend_prefix_len=extend_prefix_len,
            extend_seq_len=extend_seq_len,
        )

        # Per-image path: each item has exactly one offset (not multi-offset/EVS)
        is_per_image = all(len(item.offsets) == 1 for item in embedding_items_per_req)
        if is_per_image:
            per_image_requests.append(req_info)
        else:
            full_path_requests.append(req_info)

    # Phase 1: batch encode all per-image cache misses in ONE ViT call
    hash_to_embedding: Dict[int, torch.Tensor] = {}
    if per_image_requests:
        hash_to_embedding = _batch_encode_per_image_misses(
            data_embedding_func, per_image_requests, device
        )

    # Phase 2: assemble per-request chunks in original request order
    all_chunks: List[Tuple[int, torch.Tensor]] = []

    for req_info in per_image_requests:
        chunk = _assemble_per_image_chunk(
            req_info.overlapping,
            hash_to_embedding,
            req_info.extend_prefix_len,
            req_info.extend_seq_len,
        )
        if chunk is not None:
            all_chunks.append((req_info.req_idx, chunk))

    for req_info in full_path_requests:
        chunk_embedding = _get_chunked_embedding_full(
            data_embedding_func,
            req_info.items,
            req_info.items_offset,
            req_info.extend_prefix_len,
            req_info.extend_seq_len,
            device,
        )
        if chunk_embedding is not None:
            all_chunks.append((req_info.req_idx, chunk_embedding))

    # Sort by original request index to maintain correct output order
    all_chunks.sort(key=lambda x: x[0])
    embedding_list = [chunk for _, chunk in all_chunks]

    if not embedding_list:
        return None, input_ids
    return torch.cat(embedding_list, dim=0), input_ids


def _get_multimodal_mask(
    input_ids: torch.Tensor, placeholder_tensor: torch.Tensor
) -> torch.Tensor:
    """Build a boolean mask marking positions of multimodal tokens in input_ids."""
    return torch.isin(input_ids, placeholder_tensor).unsqueeze(-1)


def get_embedding_and_mask(
    data_embedding_func: DataEmbeddingFunc,
    embedding_items: List[MultimodalDataItem],
    placeholder_tensor: torch.Tensor,
    input_ids: torch.Tensor,
    items_size: List[int],
    prefix_length: List[int],
    extend_length: List[int],
    items_offset_list: List[List[Tuple[int, int]]],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
    """
    Generate multimodal embeddings and create a mask for their positions in input_ids.

    Returns:
        (embedding, mask, input_ids) where:
        - embedding: shape (num_mm_tokens, hidden_size) or None
        - mask: boolean tensor shape (seq_len, 1) marking mm token positions, or None
        - input_ids: possibly modified input_ids
    """
    # 1. Try precomputed embeddings first
    embedding = _get_precomputed_embedding(
        embedding_items, items_size, prefix_length, extend_length, items_offset_list
    )

    if embedding is None:
        embedding, input_ids = _get_chunked_prefill_embedding(
            data_embedding_func,
            embedding_items,
            items_size,
            prefix_length,
            extend_length,
            items_offset_list,
            input_ids,
        )
        if embedding is None:
            return None, None, input_ids

    # 2. Build mask
    special_multimodal_mask = _get_multimodal_mask(input_ids, placeholder_tensor)

    # 3. Adjust embedding length if there's a mismatch (chunked prefill edge case)
    num_mm_in_emb = embedding.shape[0]
    num_mm_in_ids = special_multimodal_mask.sum().item()

    if num_mm_in_ids != num_mm_in_emb:
        logger.warning(
            f"Multimodal token count mismatch: embedding={num_mm_in_emb}, "
            f"input_ids={num_mm_in_ids}. Adjusting."
        )
        if num_mm_in_ids < num_mm_in_emb:
            embedding = embedding[-num_mm_in_ids:, :]
        else:
            raise RuntimeError(
                f"Insufficient multimodal embedding: {num_mm_in_ids=} vs {num_mm_in_emb=}"
            )

    return embedding, special_multimodal_mask, input_ids


def embed_mm_inputs(
    mm_inputs_list: List[MultimodalInputs],
    extend_prefix_lens: List[int],
    extend_seq_lens: List[int],
    input_ids: torch.Tensor,
    input_embedding: nn.Embedding,
    multimodal_model: Optional[nn.Module] = None,
    data_embedding_func_mapping: Optional[Dict[Modality, DataEmbeddingFunc]] = None,
    forward_batch=None,
) -> Tuple[Optional[torch.Tensor], Dict]:
    """
    Embed multimodal inputs and integrate them with text token embeddings.

    Steps:
    1. For each modality: call get_{modality}_feature() → vision embeddings + mask
    2. Call embed_tokens(input_ids) → text embeddings
    3. In-place scatter vision embeddings into text positions (at mask indices)

    Returns:
        (input_embeds, other_info)
    """
    other_info = {}
    if mm_inputs_list is None:
        return None, other_info

    item_flatten_list = []
    for mm_inputs in mm_inputs_list:
        item_flatten_list += [item for item in mm_inputs.mm_items if item is not None]

    modalities, embeddings, masks = [], [], []

    for modality in Modality.all():
        items = [item for item in item_flatten_list if item.is_modality(modality)]
        if not items:
            continue

        # Find embedding function
        embedder = None
        if data_embedding_func_mapping is not None:
            embedder = data_embedding_func_mapping.get(modality)
        if embedder is None and multimodal_model is not None:
            modality_id = modality.name.lower()
            embedder = getattr(multimodal_model, f"get_{modality_id}_feature", None)

        if embedder is None:
            logger.warning(f"No embedding function found for modality {modality.name}")
            continue

        placeholder_tensor = torch.as_tensor(
            [item.pad_value for item in items],
            device=input_ids.device,
        )

        # Per-request sizes and offsets
        items_size = torch.zeros(len(mm_inputs_list) + 1, dtype=torch.long)
        items_offsets = []
        for i, mm_inputs in enumerate(mm_inputs_list):
            mm_items = [item for item in mm_inputs.mm_items if item.is_modality(modality)]
            items_size[i + 1] = len(mm_items)
            offsets = []
            for item in mm_items:
                offsets.extend(item.offsets or [])
            items_offsets.append(offsets)
        items_size = torch.cumsum(items_size, dim=0).tolist()

        embedding, mask, input_ids = get_embedding_and_mask(
            data_embedding_func=embedder,
            embedding_items=items,
            placeholder_tensor=placeholder_tensor,
            input_ids=input_ids,
            items_size=items_size,
            prefix_length=extend_prefix_lens,
            extend_length=extend_seq_lens,
            items_offset_list=items_offsets,
        )

        modalities.append(modality)
        embeddings.append(embedding)
        masks.append(mask)

    # Text token embedding
    # IMPORTANT: clamp pad_values (which may exceed vocab_size) back to valid range
    # before embedding lookup. The actual text values at mm positions will be
    # overwritten by vision embeddings anyway.
    vocab_size = input_embedding.num_embeddings
    input_ids.clamp_(min=0, max=vocab_size - 1)
    # Sequence-parallel (SP) and context-parallel (CP) both split the full sequence
    # across ranks, so input_embedding returns per-card *local*-token embeddings.
    # We remap the vision scatter into the local coordinate system via a
    # global->local position map g2l[p] = local index of global position p,
    # -1 if p is not held by this rank. Vision mask indices are global positions;
    # we drop the ones not held locally (g2l == -1) and scatter the rest into local
    # rows. Each vision token is owned by exactly one rank, so the union of per-rank
    # scatters reconstructs the full-sequence semantics.
    cp_active = bool(CP_METADATA)
    sp_active = (
        forward_batch is not None
        and getattr(forward_batch, "global_sp_num_tokens", None) is not None
    )
    g2l = None
    if cp_active:
        n_full = input_ids.shape[0]
        # local_pos[j] = global position of the j-th local token (CP zigzag split)
        local_pos = cp_split_and_rebuild_data(
            torch.arange(n_full, device=input_ids.device),
            CP_METADATA.value.split_list,
            CP_METADATA.value.zigzag_index,
        )
        g2l = torch.full((n_full,), -1, dtype=torch.long, device=input_ids.device)
        g2l[local_pos] = torch.arange(local_pos.shape[0], device=input_ids.device)
    elif sp_active:
        # SP: this rank holds a contiguous [slice_start:slice_end) of the full sequence
        n_full = input_ids.shape[0]
        slice_start, slice_end = get_attn_sp_token_slice_pos(forward_batch)
        g2l = torch.full((n_full,), -1, dtype=torch.long, device=input_ids.device)
        local_len = slice_end - slice_start
        g2l[slice_start:slice_end] = torch.arange(local_len, device=input_ids.device)

    # SP is currently populated by the NPU attention backend.  Keep its
    # enable_sp keyword on that path only: GPU FusedOverEmbedding accepts the
    # ForwardBatch but has no enable_sp argument, and GPU continues to use the
    # existing CP-aware embedding path.
    if forward_batch is not None:
        if is_npu():
            input_embeds = input_embedding(
                input_ids, forward_batch, enable_sp=sp_active
            )
        else:
            input_embeds = input_embedding(input_ids, forward_batch)
    else:
        input_embeds = input_embedding(input_ids)

    # Scatter vision embeddings into text embedding at mask positions
    for i, (modality, embedding, mask) in enumerate(zip(modalities, embeddings, masks)):
        if embedding is None or mask is None:
            continue
        indices = torch.where(mask.squeeze(dim=-1))[0]
        if g2l is not None:
            local_idx = g2l[indices]          # [num_vis], -1 = not local
            keep = local_idx >= 0
            input_embeds[local_idx[keep]] = embedding[keep].to(
                input_embeds.device, input_embeds.dtype
            )
        else:
            input_embeds[indices] = embedding.to(input_embeds.device, input_embeds.dtype)

    return input_embeds, other_info


def general_mm_embed_routine(
    input_ids: torch.Tensor,
    forward_batch,
    language_model: nn.Module,
    multimodal_model: Optional[nn.Module] = None,
    data_embedding_funcs: Optional[Dict[Modality, DataEmbeddingFunc]] = None,
    **kwargs,
) -> torch.Tensor:
    """
    High-level entry point for multimodal embedding fusion in model forward().

    Flow:
    - If Prefill with mm_inputs: run embed_mm_inputs() to fuse embeddings
    - Otherwise: simple embed_tokens(input_ids)
    - Forward through language_model with fused embeddings

    After embedding, pixel_values are offloaded to CPU to free GPU memory
    (for chunked prefill: subsequent chunks can reload from CPU if needed,
    or use ViT Embedding Cache to avoid recomputation).
    """
    assert hasattr(language_model, "get_input_embeddings"), (
        "language_model must implement get_input_embeddings()"
    )
    embed_tokens = language_model.get_input_embeddings()

    contains_mm = (
        forward_batch.mm_inputs is not None
        and any(m is not None for m in forward_batch.mm_inputs)
    )

    if not forward_batch.forward_mode.is_decode() and contains_mm:
        mm_inputs_list = [
            mm_input for mm_input in forward_batch.mm_inputs if mm_input is not None
        ]
        extend_prefix_lens = [
            prefix_len
            for i, prefix_len in enumerate(forward_batch.extend_prefix_lens_cpu)
            if forward_batch.mm_inputs[i] is not None
        ]
        extend_seq_lens = [
            seq_len
            for i, seq_len in enumerate(forward_batch.extend_seq_lens_cpu)
            if forward_batch.mm_inputs[i] is not None
        ]

        input_embeds, other_info = embed_mm_inputs(
            mm_inputs_list=mm_inputs_list,
            extend_prefix_lens=extend_prefix_lens,
            extend_seq_lens=extend_seq_lens,
            input_ids=input_ids,
            input_embedding=embed_tokens,
            multimodal_model=multimodal_model,
            data_embedding_func_mapping=data_embedding_funcs,
        )
        # Offload pixel_values to CPU after ViT runs.
        # For chunked prefill, the first chunk runs ViT and caches embeddings;
        # subsequent chunks look up the cache and skip pixel_values entirely.
        for mm_input_obj in mm_inputs_list:
            for mm_item in mm_input_obj.mm_items:
                if isinstance(mm_item.feature, torch.Tensor) and mm_item.feature.is_cuda:
                    mm_item.feature = mm_item.feature.to("cpu", non_blocking=True)

        # Clear references to free memory
        forward_batch.mm_inputs = None
    else:
        input_embeds = embed_tokens(input_ids)

    # Copy to pre-allocated buffer if available (for CUDA graph address stability)
    if forward_batch.input_embeds is not None:
        forward_batch.input_embeds.copy_(input_embeds)
        input_embeds = forward_batch.input_embeds

    hidden_states = language_model(
        input_ids=None,
        forward_batch=forward_batch,
        input_embeds=input_embeds,
        **kwargs,
    )
    return hidden_states


# ============================================================
# Item splitting utilities
# ============================================================

# ==============================================================================
# DP-sharded vision model utilities
# Ported from sglang/python/sglang/srt/multimodal/mm_utils.py
# ==============================================================================

def get_dp_encoder_lb_assignment(
    sizes: List[int],
    num_gpus: int = 2,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Generate load balancing assignment and metadata for distributing data across GPUs.

    The load is determined by the total image sizes, not the number of images.

    Args:
        sizes: The size of each image
        num_gpus: Number of GPUs to balance across

    Returns:
        shuffle_indices:   Indices to reorder data for balanced loading
        gpu_sample_counts: Number of samples assigned to each GPU
        grouped_sizes_per_gpu: Total size assigned to each GPU
    """
    n_samples = len(sizes)

    if n_samples == 0:
        return [], [0] * num_gpus, [0] * num_gpus

    gpu_assignments: List[List[int]] = [[] for _ in range(num_gpus)]
    gpu_loads = [0] * num_gpus

    large_to_small_indices = sorted(range(n_samples), key=lambda i: sizes[i], reverse=True)

    for idx in large_to_small_indices:
        min_gpu = min(range(num_gpus), key=lambda i: gpu_loads[i])
        gpu_assignments[min_gpu].append(idx)
        gpu_loads[min_gpu] += sizes[idx]

    shuffle_indices: List[int] = []
    gpu_sample_counts: List[int] = []
    for gpu_id in range(num_gpus):
        shuffle_indices.extend(gpu_assignments[gpu_id])
        gpu_sample_counts.append(len(gpu_assignments[gpu_id]))

    return (shuffle_indices, gpu_sample_counts, gpu_loads)


def run_dp_sharded_mrope_vision_model(
    vision_model: "torch.nn.Module",
    pixel_values: "torch.Tensor",
    grid_thw_list: list,
    *,
    rope_type: Literal["rope_3d", "rope_2d"],
) -> "torch.Tensor":
    """Run a vision model with data parallelism (DP) sharding.

    The function will shard the input image tensor on the first dimension and
    run the vision model.  Used for Qwen2.5-VL / Qwen3-VL with mrope.

    Args:
        vision_model: Vision model.
        pixel_values:  Image/Video input tensor.
        grid_thw_list: List of grid dimensions for each image.
        rope_type: "rope_3d" (Qwen2.5-VL) or "rope_2d" (Kimi-VL).

    Returns:
        torch.Tensor: Output image embeddings.
    """
    from sglang.srt.layers.attention.dp_attention import (
        get_attention_tp_group,
        get_attention_tp_rank,
        get_attention_tp_size,
    )

    tp_size = get_attention_tp_size()
    if tp_size == 1:
        return vision_model(pixel_values, grid_thw=torch.tensor(grid_thw_list))

    tp_rank_local = get_attention_tp_rank()

    patches_per_image = [math.prod(grid_thw) for grid_thw in grid_thw_list]
    cum_patches_per_image = [0, *itertools.accumulate(patches_per_image)]

    image_to_tp_rank, gpu_sample_counts, grouped_pixel_values_len = (
        get_dp_encoder_lb_assignment(patches_per_image, tp_size)
    )

    cum_gpu_sample_counts = [0, *itertools.accumulate(gpu_sample_counts)]

    image_idxs_local = image_to_tp_rank[
        cum_gpu_sample_counts[tp_rank_local] : cum_gpu_sample_counts[tp_rank_local + 1]
    ]

    if len(image_idxs_local) > 0:
        pixel_values_local = torch.cat(
            [
                pixel_values[cum_patches_per_image[i] : cum_patches_per_image[i + 1]]
                for i in image_idxs_local
            ]
        )
    else:
        pixel_values_local = torch.empty(
            (0, pixel_values.shape[1]),
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        )

    if rope_type == "rope_2d":
        embed_dim_reduction_factor = (
            vision_model.merge_kernel_size[0] * vision_model.merge_kernel_size[1]
        )
    else:
        embed_dim_reduction_factor = (
            vision_model.spatial_merge_size * vision_model.spatial_merge_size
        )

    max_len_per_rank = max(grouped_pixel_values_len) // embed_dim_reduction_factor
    local_grid_thw_list = [grid_thw_list[i] for i in image_idxs_local]

    if rope_type == "rope_2d":
        if pixel_values_local.shape[0] > 0:
            image_embeds_local = vision_model(
                pixel_values_local, torch.tensor(local_grid_thw_list)
            )
            if isinstance(image_embeds_local, list):
                image_embeds_local = torch.cat(image_embeds_local, dim=0)
        else:
            out_dim = getattr(vision_model.config, "hidden_size", None)
            image_embeds_local = torch.empty(
                (0, embed_dim_reduction_factor, out_dim),
                device=pixel_values.device,
                dtype=pixel_values.dtype,
            )
    else:
        if pixel_values_local.shape[0] > 0:
            image_embeds_local = vision_model(
                pixel_values_local, torch.tensor(local_grid_thw_list)
            )
        else:
            image_embeds_local = torch.empty(
                (0, vision_model.out_hidden_size),
                device=pixel_values.device,
                dtype=pixel_values.dtype,
            )

    current_len = image_embeds_local.shape[0]
    if current_len < max_len_per_rank:
        padding_size = max_len_per_rank - current_len
        if rope_type == "rope_2d":
            padding = torch.empty(
                (padding_size, image_embeds_local.shape[1], image_embeds_local.shape[2]),
                dtype=image_embeds_local.dtype,
                device=image_embeds_local.device,
            )
        else:
            padding = torch.empty(
                (padding_size, image_embeds_local.shape[1]),
                dtype=image_embeds_local.dtype,
                device=image_embeds_local.device,
            )
        image_embeds_local_padded = torch.cat([image_embeds_local, padding], dim=0)
    else:
        image_embeds_local_padded = image_embeds_local

    gathered_embeds = get_attention_tp_group().all_gather(
        image_embeds_local_padded, dim=0
    )

    rank_embeddings: List["torch.Tensor"] = []
    for rank in range(tp_size):
        start_idx = rank * max_len_per_rank
        end_idx = start_idx + (
            grouped_pixel_values_len[rank] // embed_dim_reduction_factor
        )
        rank_embeddings.append(gathered_embeds[start_idx:end_idx])

    patches_per_output_image = [
        (patch_size // embed_dim_reduction_factor) for patch_size in patches_per_image
    ]

    original_order_embeddings: List[Optional["torch.Tensor"]] = [None] * len(grid_thw_list)
    current_idx = 0
    for rank in range(tp_size):
        count = gpu_sample_counts[rank]
        if count > 0:
            rank_images = image_to_tp_rank[current_idx : current_idx + count]
            rank_embed = rank_embeddings[rank]
            embed_start = 0
            for img_idx in rank_images:
                img_patches = patches_per_output_image[img_idx]
                original_order_embeddings[img_idx] = rank_embed[
                    embed_start : embed_start + img_patches
                ]
                embed_start += img_patches
            current_idx += count

    out_embeddings = torch.cat(original_order_embeddings, dim=0)
    return out_embeddings


def get_new_expanded_mm_items(
    original_mm_items: List[MultimodalDataItem],
) -> List[MultimodalDataItem]:
    """
    Split bundled MultimodalDataItems (N images in one item) into per-image items.

    This is important for ViT Embedding Cache: each image needs its own hash
    so it can be independently cached and looked up.

    When processor returns all images in one item (pixel_values of shape [N×patches, C, P, P]),
    we split by image_grid_thw to get per-image pixel_values.
    """
    expanded = []
    for item in original_mm_items:
        is_bundled = item.offsets is not None and len(item.offsets) > 1

        if not is_bundled:
            expanded.append(item)
            continue

        num_items = len(item.offsets)

        if item.is_image():
            image_grid_thw = item.model_specific_data.get("image_grid_thw")
            if image_grid_thw is None or len(image_grid_thw) != num_items:
                # No grid info → simple split by feature dim-0
                feature = item.feature if item.feature is not None else item.precomputed_embeddings
                if (
                    feature is not None
                    and hasattr(feature, "shape")
                    and feature.shape[0] == num_items
                ):
                    for i in range(num_items):
                        new_item = copy.copy(item)
                        if item.feature is not None:
                            new_item.feature = item.feature[i : i + 1]
                        if item.precomputed_embeddings is not None:
                            new_item.precomputed_embeddings = item.precomputed_embeddings[i : i + 1]
                        new_item.offsets = [item.offsets[i]]
                        new_item.hash = None
                        expanded.append(new_item)
                else:
                    expanded.append(item)
                continue

            # Compute patch slices from grid_thw: T×H×W patches per image
            patches_per_item = []
            for grid in image_grid_thw:
                grid_tensor = torch.as_tensor(grid, dtype=torch.long)
                patches_per_item.append(int(torch.prod(grid_tensor).item()))

            cumulative = torch.cumsum(
                torch.tensor(patches_per_item, dtype=torch.long), dim=0
            )
            slice_indices = [0] + cumulative.tolist()

            feature_len = item.feature.shape[0] if item.feature is not None else None
            if feature_len is None and item.precomputed_embeddings is not None:
                feature_len = item.precomputed_embeddings.shape[0]

            if feature_len is None or slice_indices[-1] != feature_len:
                expanded.append(item)
                continue

            for i in range(num_items):
                start, end = slice_indices[i], slice_indices[i + 1]
                new_item = copy.copy(item)
                if item.feature is not None:
                    new_item.feature = item.feature[start:end]
                if item.precomputed_embeddings is not None:
                    new_item.precomputed_embeddings = item.precomputed_embeddings[start:end]
                new_item.offsets = [item.offsets[i]]
                # Slice model_specific_data accordingly
                new_data = {}
                for k, v in item.model_specific_data.items():
                    if isinstance(v, (list, tuple)) and len(v) == num_items:
                        new_data[k] = [v[i]]
                    elif (
                        isinstance(v, (torch.Tensor, np.ndarray))
                        and v.ndim > 0
                        and v.shape[0] == num_items
                    ):
                        new_data[k] = v[i : i + 1]
                    else:
                        new_data[k] = v
                new_item.model_specific_data = new_data
                new_item.hash = None
                expanded.append(new_item)
        else:
            # Video/audio: simple split by feature dim-0
            feature = item.feature if item.feature is not None else item.precomputed_embeddings
            if (
                feature is not None
                and hasattr(feature, "shape")
                and feature.shape[0] == num_items
            ):
                for i in range(num_items):
                    new_item = copy.copy(item)
                    if item.feature is not None:
                        new_item.feature = item.feature[i : i + 1]
                    if item.precomputed_embeddings is not None:
                        new_item.precomputed_embeddings = item.precomputed_embeddings[i : i + 1]
                    new_item.offsets = [item.offsets[i]]
                    new_item.hash = None
                    expanded.append(new_item)
            else:
                expanded.append(item)

    return expanded
