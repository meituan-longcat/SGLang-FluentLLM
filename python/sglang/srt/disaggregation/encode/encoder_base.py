"""Model-agnostic Encoder-side preprocessing & ViT feature interfaces.

This module defines two decoupled abstract interfaces used by the Encoder
Server (``EncodeScheduler``):

1. ``EncoderPreprocessor`` — pure preprocessing (URL → item + meta):
   download → transform → hash → MMItemMeta construction.
   Does **not** include ViT forward.

2. ``EncoderFeature`` — ViT feature computation (items → ViT output):
   ``get_image_feature`` / ``get_video_feature`` / ``item_token_count``.
   Implemented by the VL model main class directly (e.g.
   ``LongcatFlashVLForConditionalGeneration``) because ViT forward needs
   access to ``vision_tower`` / ``quantizer`` / ``fake_input`` which are
   model-internal state.

Both interfaces are model-agnostic: ``EncodeScheduler`` depends only on
these two ABCs, never on a concrete model class.  Future VL models
integrate with the Encoder Server by implementing both interfaces.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import List, Optional

import torch

from sglang.srt.managers.scheduling.multimodal_data import (
    Modality,
    MultimodalDataItem,
)

# Avoid a hard circular import at module import time: encode_data is in the
# same package and only needed for type hints inside method signatures.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.disaggregation.encode.encode_data import MMItemMeta
    from sglang.srt.disaggregation.encode.utils import Timer


@dataclass
class PreprocessedItem:
    """Result of preprocessing a single media item on the Encoder side.

    Encapsulates everything ViT forward needs + everything MMMeta needs.
    The Encoder ``_preprocess`` no longer constructs these itself; it
    delegates to ``EncoderPreprocessor`` which returns this struct.
    """

    item: MultimodalDataItem  # feature(pixel_values) + image_grid_thw + hash + pad_value
    meta: "MMItemMeta"  # item_index / hash / pad_value / grid_thw / num_tokens
    cost: int  # patch token count = T * H * W (for batching budget)
    is_cache_hit: bool = False  # True → ViT cache hit, skip ViT queue


class EncoderPreprocessor(abc.ABC):
    """Model-agnostic Encoder-side preprocessing interface (no ViT).

    Only responsible for the stateless transform "URL → item + meta":
    download → transform → hash → MMItemMeta construction.

    ViT feature computation is abstracted by ``EncoderFeature``; the two
    are decoupled with single responsibilities.

    Design principles:
    1. Methods are split by modality (process_image / process_video);
       Encoder ``_preprocess`` dispatches by ``MediaURL.modality``.
    2. Returns ``PreprocessedItem`` (item + meta + cost); the Encoder no
       longer constructs ``MultimodalDataItem`` / ``MMItemMeta`` / hash /
       pad_value itself.
    3. Cache lookup stays on the Encoder side (``EncodeScheduler`` owns
       ``TieredEmbeddingCache``) because the cache is an Encoder runtime
       resource, not model state.  ``EncoderPreprocessor`` only does the
       stateless "URL → item+meta" transform.
    4. Does **not** include vit_forward / item_token_count — those are
       provided by ``EncoderFeature`` (below), implemented directly by
       the VL model main class (e.g. LongcatFlashVLForConditionalGeneration).
    """

    def get_url_cache_signature(self, modality: Modality) -> Optional[str]:
        """Return a stable signature for URL-memo correctness.

        Concrete preprocessors must override this and include every model and
        preprocessing option that can change the generated item hash or
        metadata.  ``None`` disables URL memoization for unadapted models;
        falling back to only a class name would be unsafe across config changes.
        """
        return None

    @abc.abstractmethod
    def process_image(
        self,
        url: str,
        item_index: int,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        device_preprocess: bool = False,
        device: Optional[torch.device] = None,
        *,
        timer: "Timer",
    ) -> PreprocessedItem:
        """Process a single image URL: download → transform → hash → item+meta.

        Args:
            url: image URL / local path / data URI.
            item_index: global index of this item within the request.
            min_pixels / max_pixels: pixel budget override (None → model default).
            device_preprocess: True → select the CUDA/NPU device transform.
            device: target accelerator device (used when
                device_preprocess=True).
            timer: per-media timer. Concrete implementations record
                ``download``, ``decode``, ``transform`` and ``hash`` phases.

        Returns:
            PreprocessedItem with item(feature/grid_thw/hash/pad_value) +
            meta + cost.  is_cache_hit=False (cache lookup is done by
            EncodeScheduler).
        """
        ...

    @abc.abstractmethod
    def process_video(
        self,
        url: str,
        item_index: int,
        max_pixels: Optional[int] = None,
        device_preprocess: bool = False,
        device: Optional[torch.device] = None,
        *,
        timer: "Timer",
    ) -> List[PreprocessedItem]:
        """Process one video URL into model-native video item(s).

        The concrete preprocessor must preserve the original model processor's
        temporal semantics. Longcat groups adjacent natural frames according
        to temporal_patch_size and returns one whole-video item; it must not
        reinterpret natural frames as independent images.

        Args:
            url: video URL / local path / data URI / pre-extracted frame JSON.
            item_index: global index of this video in the request media list.
            max_pixels: per-frame pixel budget override.
            device_preprocess / device: same as process_image.
            timer: per-media timer using the same fixed preprocess
                phase names as ``process_image``.

        Returns:
            Model-native preprocessed items. For Longcat this is a one-element
            list whose item has grid_thw=[temporal_groups,H,W].
        """
        ...

    @classmethod
    def from_hf_config(cls, hf_config) -> "EncoderPreprocessor":
        """Factory method: build a preprocessor instance from ``hf_config``.

        This is the interface-level factory method — it belongs to
        ``EncoderPreprocessor`` (the thing being constructed), **not** to
        ``EncoderFeature`` (which only handles ViT computation).  Each
        concrete model-specific preprocessor overrides this classmethod to
        construct itself from ``hf_config``.

        The Encoder Server calls this on the concrete class obtained from
        ``model.get_encoder_preprocessor_class()`` — the model only
        *declares* which preprocessor class implements it, and the
        preprocessor class knows how to build itself.

        Args:
            hf_config: the HuggingFace config of the VL model.

        Returns:
            An ``EncoderPreprocessor`` instance wired with the model's
            preprocessing dependencies.
        """
        # Default implementation cannot construct an abstract instance; a
        # concrete subclass must override this.  Raising here surfaces a
        # clear error if a model forgets to implement it.
        raise NotImplementedError(
            f"{cls.__name__} must override EncoderPreprocessor.from_hf_config"
        )


class EncoderFeature(abc.ABC):
    """Model-agnostic Encoder-side ViT feature computation interface.

    Only responsible for the model-specific computation "items → ViT output".
    Preprocessing is abstracted by ``EncoderPreprocessor``; the two are
    decoupled with single responsibilities.

    Design principles:
    1. Split by modality: get_image_feature / get_video_feature; Encoder
       ``step()`` dispatches by item modality.
    2. Implemented directly by the VL model main class (e.g.
       LongcatFlashVLForConditionalGeneration) because ViT forward needs
       access to vision_tower / quantizer / fake_input which are model-
       internal state tied to the model lifecycle.
    3. Concrete implementations handle fake_input append / n_image_patches
       computation / vision_tower forward / quantizer.encode internally.
    """

    @abc.abstractmethod
    def get_image_feature(
        self,
        items: List[MultimodalDataItem],
        output_format: str = "rvq_indices",
    ) -> torch.Tensor:
        """Run ViT on a batch of image items, return rvq_indices or vit_embedding.

        Args:
            items: ViT inputs (each item's feature is pixel_values).
            output_format: "rvq_indices" or "vit_embedding".

        Returns:
            [num_tokens, depth] (rvq_indices) or [num_tokens, hidden] (vit_embedding).
        """
        ...

    @abc.abstractmethod
    def get_video_feature(
        self,
        items: List[MultimodalDataItem],
        output_format: str = "rvq_indices",
    ) -> torch.Tensor:
        """Run ViT on a batch of model-native video items, return rvq_indices or
        vit_embedding.

        The item layout must match the original joint-deployment video path.
        Args / Returns same as get_image_feature.
        """
        ...

    @abc.abstractmethod
    def item_token_count(self, item: MultimodalDataItem) -> int:
        """Compute the ViT output token count for a single item.

        Used for per_item_embeds split + cache put num_tokens.
        Image: T * (H//merge) * (W//merge).
        Video: T * (H//merge) * (W//merge), where T is temporal groups.
        """
        ...
