"""
Multimodal data structures for scheduling.

Directly re-uses sglang upstream implementation with minimal adaptations for fluentllm:
- `sglang.srt.managers.mm_utils` → `sglang.srt.multimodal.mm_utils`
- `sglang.srt.utils.flatten_nested_list` → from `sglang.srt.utils.common`
- `CudaIpcTensorTransportProxy` → gracefully skipped (not in fluentllm)

Public API:
    MM_PAD_SHIFT_VALUE          – base offset ensuring pad_values don't clash with text tokens
    _compute_pad_value()        – maps modality + content hash → a stable pad token id
    Modality                    – IMAGE / VIDEO / AUDIO enum
    MultimodalInputFormat       – NORMAL / PROCESSOR_OUTPUT / PRECOMPUTED_EMBEDDING enum
    MultimodalDataItem          – one image/video/audio with its feature tensor and hash
    MultimodalProcessorOutput   – raw output from model-specific processors
    MultimodalInputs            – processed, hashed items ready for scheduling
"""

from __future__ import annotations

import dataclasses
from enum import Enum, auto
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from sglang.srt.env import envs
from sglang.srt.utils.common import flatten_nested_list
from sglang.srt.utils.cuda_ipc_transport_utils import CudaIpcTensorTransportProxy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Constant used as the base offset for MM (multimodal) pad values.
# This ensures pad_values don't overlap with valid text token IDs.
MM_PAD_SHIFT_VALUE = 1_000_000
MM_PAD_INT32_MAX = (1 << 31) - 1
MM_PAD_MODALITY_COUNT = 3
MM_PAD_HASH_SLOTS = (
    MM_PAD_INT32_MAX - MM_PAD_SHIFT_VALUE + 1
) // MM_PAD_MODALITY_COUNT
MM_PAD_VALUE_MAX = (
    MM_PAD_SHIFT_VALUE + MM_PAD_HASH_SLOTS * MM_PAD_MODALITY_COUNT - 1
)


@lru_cache(maxsize=1)
def sanity_check_mm_pad_shift_value(vocab_size: int) -> None:
    if vocab_size > MM_PAD_SHIFT_VALUE:
        raise ValueError(
            f"Model vocab_size ({vocab_size}) exceeds MM_PAD_SHIFT_VALUE ({MM_PAD_SHIFT_VALUE}). "
            f"MM pad_values may overlap with valid token IDs. "
            f"Please increase MM_PAD_SHIFT_VALUE in multimodal_data.py."
        )

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Modality(Enum):
    IMAGE = auto()
    VIDEO = auto()
    AUDIO = auto()

    @staticmethod
    def from_str(modality_str: str) -> "Modality":
        try:
            return Modality[modality_str.upper()]
        except KeyError:
            raise ValueError(
                f"Invalid modality string: {modality_str}. Valid modalities are: {[m.name for m in Modality]}"
            )

    @staticmethod
    def all() -> List["Modality"]:
        return [Modality.IMAGE, Modality.VIDEO, Modality.AUDIO]


_MODALITY_PAD_TAG = {
    Modality.IMAGE: 0,
    Modality.VIDEO: 1,
    Modality.AUDIO: 2,
}


def _compute_pad_value(hash: int, modality: Modality) -> int:
    """Encode modality and content hash into one signed-int32 pad value.

    Each modality owns a disjoint, equally sized interval.  The target model
    and RadixCache keep this hash-derived value, while the MTP drafter can infer
    the modality from the interval and replace it with an in-vocabulary
    placeholder immediately before embedding lookup.
    """
    if not isinstance(modality, Modality):
        modality = Modality.from_str(str(modality))
    return (
        MM_PAD_SHIFT_VALUE
        + _MODALITY_PAD_TAG[modality] * MM_PAD_HASH_SLOTS
        + int(hash) % MM_PAD_HASH_SLOTS
    )


def is_mm_pad_value(input_ids: torch.Tensor) -> torch.Tensor:
    """Return a mask selecting every modality-tagged hash pad value."""
    return (input_ids >= MM_PAD_SHIFT_VALUE) & (input_ids <= MM_PAD_VALUE_MAX)


def is_mm_pad_value_for(
    input_ids: torch.Tensor, modality: Modality
) -> torch.Tensor:
    """Return a mask selecting hash pad values for one modality."""
    tag = _MODALITY_PAD_TAG[modality]
    lower = MM_PAD_SHIFT_VALUE + tag * MM_PAD_HASH_SLOTS
    upper = lower + MM_PAD_HASH_SLOTS
    return (input_ids >= lower) & (input_ids < upper)


def substitute_mm_pad_values_for_mtp(
    input_ids: torch.Tensor,
    placeholder_ids: Dict[Modality, int],
) -> torch.Tensor:
    """Replace tagged pad values on the drafter-local tensor only.

    The drafter tensor is cloned before replacement so the target input IDs and
    RadixCache keys remain content-hash based.  The implementation intentionally
    contains no device-to-host predicate or synchronization.
    """
    if not placeholder_ids:
        return input_ids

    output = input_ids.clone()
    for modality, placeholder_id in placeholder_ids.items():
        output.masked_fill_(
            is_mm_pad_value_for(input_ids, modality), int(placeholder_id)
        )
    return output


def resolve_mtp_mm_placeholder_ids(hf_config: Any) -> Dict[Modality, int]:
    """Resolve modality placeholder IDs once from the model configuration."""
    source_names = (
        None,
        "thinker_config",
        "text_config",
        "vision_config",
        "visual_config",
        "video_config",
        "audio_config",
    )
    candidate_names = {
        Modality.IMAGE: (
            "image_pad_token_id",
            "image_token_id",
            "image_placeholder_token_id",
        ),
        Modality.VIDEO: (
            "video_place_token_id",
            "video_pad_token_id",
            "video_token_id",
            "video_placeholder_token_id",
        ),
        Modality.AUDIO: (
            "audio_pad_token_id",
            "audio_token_id",
            "audio_placeholder_token_id",
        ),
    }

    shared_id = None
    for source_name in source_names:
        source = (
            hf_config
            if source_name is None
            else getattr(hf_config, source_name, None)
        )
        if source is not None:
            value = getattr(source, "media_placeholder_token_id", None)
            if value is not None:
                shared_id = int(value)
                break

    resolved: Dict[Modality, int] = {}
    for modality, names in candidate_names.items():
        for source_name in source_names:
            source = (
                hf_config
                if source_name is None
                else getattr(hf_config, source_name, None)
            )
            if source is None:
                continue
            for name in names:
                value = getattr(source, name, None)
                if value is not None:
                    resolved[modality] = int(value)
                    break
            if modality in resolved:
                break
        if modality not in resolved and shared_id is not None:
            resolved[modality] = shared_id
    return resolved


class MultimodalInputFormat(Enum):
    NORMAL = auto()
    PROCESSOR_OUTPUT = auto()
    PRECOMPUTED_EMBEDDING = auto()


# ---------------------------------------------------------------------------
# MultimodalDataItem
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class MultimodalDataItem:
    """
    One MultimodalDataItem represents a single multimodal input (one image, one video, or one audio).
    For example, if there are 3 images and 1 audio, there will be 4 MultimodalDataItems.

    Each item has its own hash and pad_value, enabling per-image RadixAttention caching.

    We put the common fields first and the model-specific fields in model_specific_data.
    """

    modality: Modality
    hash: int = None
    pad_value: int = None
    offsets: Optional[list] = None

    format: MultimodalInputFormat = MultimodalInputFormat.NORMAL

    # the raw features returned by processor, e.g. pixel_values or audio_features
    feature: Union[torch.Tensor, np.ndarray] = None
    # the precomputed embeddings, passed as final encoder embeddings
    # One and only one of the feature and precomputed_embeddings will be empty
    precomputed_embeddings: Optional[Union[torch.Tensor, np.ndarray]] = None

    # Model-specific data stored in a dictionary
    model_specific_data: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __getattr__(self, name: str):
        if (
            "model_specific_data" in self.__dict__
            and name in self.__dict__["model_specific_data"]
        ):
            return self.__dict__["model_specific_data"][name]
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

    def __setitem__(self, key: str, value: Any):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self.model_specific_data[key] = value

    def set(self, key: str, value: Any):
        self.__setitem__(key, value)

    @staticmethod
    def is_empty_list(l):
        if l is None:
            return True
        return len([item for item in flatten_nested_list(l) if item is not None]) == 0

    def set_pad_value(self):
        """
        Set the pad value after first hashing the data
        """
        if self.pad_value is not None:
            return

        from sglang.srt.multimodal.mm_utils import hash_feature

        if envs.SGLANG_MM_SKIP_COMPUTE_HASH.get():
            import uuid

            self.hash = uuid.uuid4().int
            self.pad_value = _compute_pad_value(self.hash, self.modality)
            return
        if self.hash is None:
            if self.feature is not None:
                hashed_feature = self.feature
            else:
                hashed_feature = self.precomputed_embeddings
            # Guard against accidentally hashing a CudaIpcTensorTransportProxy.
            # When CUDA IPC transport is enabled, base_processor pre-computes
            # item.hash on the real GPU tensor BEFORE replacing item.feature with
            # a CudaIpcTensorTransportProxy.  If, despite that, we still arrive
            # here with a proxy object, it means something has gone wrong
            # upstream – raise a clear error instead of silently computing a
            # meaningless/garbage hash from the proxy's internal state.
            if isinstance(hashed_feature, CudaIpcTensorTransportProxy):
                raise RuntimeError(
                    "set_pad_value() was called with item.hash=None but "
                    "item.feature is a CudaIpcTensorTransportProxy. "
                    "The hash must be pre-computed from the real tensor "
                    "before the feature is replaced by a proxy. "
                    "Check that base_processor pre-computes item.hash when "
                    "SGLANG_USE_CUDA_IPC_TRANSPORT is enabled."
                )
            self.hash = hash_feature(hashed_feature)
        assert self.hash is not None
        self.pad_value = _compute_pad_value(self.hash, self.modality)

    def is_modality(self, modality: Modality) -> bool:
        return self.modality == modality

    def is_audio(self):
        return self.modality == Modality.AUDIO

    def is_image(self):
        return self.modality == Modality.IMAGE

    def is_video(self):
        return self.modality == Modality.VIDEO

    def is_valid(self) -> bool:
        return self.is_image() or self.is_video() or self.is_audio()

    def is_precomputed_embedding(self):
        return self.format == MultimodalInputFormat.PRECOMPUTED_EMBEDDING

    def reconstruct(self):
        """
        Reconstruct GPU tensors from CudaIpcTensorTransportProxy objects.

        Called in Scheduler after broadcast_pyobj delivers the proxy objects to
        all TP ranks.  Each rank calls this independently, opening its own view
        of the CUDA IPC pool storage and performing a local GPU-to-GPU copy.

        After copying, the sync flag is incremented so the producer
        (TokenizerManager) can recycle the pool chunk once all ranks complete.
        """

        reconstruct_device = torch.cuda.current_device()

        if isinstance(self.feature, CudaIpcTensorTransportProxy):
            self.feature = self.feature.reconstruct_on_target_device(reconstruct_device)

        if isinstance(self.precomputed_embeddings, CudaIpcTensorTransportProxy):
            self.precomputed_embeddings = (
                self.precomputed_embeddings.reconstruct_on_target_device(reconstruct_device)
            )

        for key in list(self.model_specific_data.keys()):
            val = self.model_specific_data[key]
            if isinstance(val, CudaIpcTensorTransportProxy):
                self.model_specific_data[key] = val.reconstruct_on_target_device(
                    reconstruct_device
                )

    @staticmethod
    def from_dict(obj: dict) -> "MultimodalDataItem":
        kwargs = dict(obj)
        modality = kwargs.pop("modality")
        if isinstance(modality, str):
            modality = Modality[modality]
        return MultimodalDataItem(modality=modality, **kwargs)


# ---------------------------------------------------------------------------
# MultimodalProcessorOutput
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class MultimodalProcessorOutput:
    """Raw output from multimodal processors, before pad/hash computation.

    This is the typed replacement for the dict previously returned by
    ``BaseMultimodalProcessor.process_mm_data_async``.  Unlike
    ``MultimodalInputs``, items here do NOT carry pad_value or hash yet.
    """

    mm_items: List[MultimodalDataItem]
    input_ids: Optional[List[int]] = None

    # image
    im_token_id: Optional[int] = None
    im_start_id: Optional[int] = None
    im_end_id: Optional[int] = None
    slice_start_id: Optional[int] = None
    slice_end_id: Optional[int] = None

    # video
    video_token_id: Optional[int] = None

    # audio
    audio_token_id: Optional[int] = None
    audio_start_id: Optional[int] = None
    audio_end_id: Optional[int] = None

    # QWen2-VL related
    mrope_positions: Optional[torch.Tensor] = None
    mrope_position_delta: Optional[torch.Tensor] = None

    # for transformers-compatibility
    token_type_ids: Optional[torch.Tensor] = None

    @staticmethod
    def from_dict(d: dict) -> "MultimodalProcessorOutput":
        return MultimodalProcessorOutput(
            mm_items=d["mm_items"],
            input_ids=d.get("input_ids"),
            im_token_id=d.get("im_token_id"),
            im_start_id=d.get("im_start_id"),
            im_end_id=d.get("im_end_id"),
            slice_start_id=d.get("slice_start_id"),
            slice_end_id=d.get("slice_end_id"),
            video_token_id=d.get("video_token_id"),
            audio_token_id=d.get("audio_token_id"),
            audio_start_id=d.get("audio_start_id"),
            audio_end_id=d.get("audio_end_id"),
            mrope_positions=d.get("mrope_positions"),
            mrope_position_delta=d.get("mrope_position_delta"),
        )


# ---------------------------------------------------------------------------
# MultimodalInputs
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class MultimodalInputs:
    """The multimodal data related inputs."""

    # items of data
    mm_items: List[MultimodalDataItem]
    image_pad_len: Optional[list] = None
    num_image_tokens: Optional[int] = None

    # image
    im_token_id: Optional[int] = None
    im_start_id: Optional[int] = None
    im_end_id: Optional[int] = None
    slice_start_id: Optional[int] = None
    slice_end_id: Optional[int] = None

    # video
    video_token_id: Optional[int] = None

    # audio
    audio_token_id: Optional[int] = None
    audio_start_id: Optional[int] = None
    audio_end_id: Optional[int] = None

    # QWen2-VL related
    mrope_positions: Optional[torch.Tensor] = None
    mrope_position_delta: Optional[torch.Tensor] = None
    mrope_position_delta_repeated_cache: Optional[torch.Tensor] = None

    # EPD: mm_meta from remote encode_async (set by _apply_epd_mm_processing)
    mm_meta: Optional[object] = None

    epd_media_urls: Optional[list] = None
    epd_input_text: Optional[str] = None
    # Optional Encoder-compute-index -> prompt-media-index mapping. It is
    # populated only when one logical video is split into temporal groups.
    epd_compute_to_logical: Optional[list[int]] = None
    # EPD: resolved encoder URLs for this request (list[str], isomorphic to
    # server_args.encoder_urls.split(",")). Set by TM from EncoderRouter.
    # Scheduler consumes it as a plain list[str] — no transfer_info awareness.
    epd_encoder_urls: Optional[list] = None

    def release_features(self):
        """Release feature tensors to free GPU memory."""
        for item in self.mm_items:
            item.feature = None

    def reconstruct_features(self):
        """Reconstruct GPU tensors from CudaIpcTensorTransportProxy objects.

        Called in Scheduler.recv_requests() after broadcast_pyobj has delivered
        the proxy metadata to every TP rank.  Each rank independently opens the
        CUDA IPC handle and copies the tensor to its own device memory.
        """
        for item in self.mm_items:
            item.reconstruct()

    @property
    def contains_image(self) -> bool:
        return any(item.is_image() for item in self.mm_items)

    @property
    def contains_video(self) -> bool:
        return any(item.is_video() for item in self.mm_items)

    @property
    def contains_audio(self) -> bool:
        return any(item.is_audio() for item in self.mm_items)

    @staticmethod
    def from_processor_output(obj: "MultimodalProcessorOutput") -> "MultimodalInputs":
        mm_items = obj.mm_items

        ret = MultimodalInputs(
            mm_items=mm_items,
        )

        assert isinstance(ret.mm_items, list)
        ret.mm_items = [item for item in ret.mm_items if item.is_valid()]

        if envs.SGLANG_MM_BUFFER_SIZE_MB.get() > 0:
            # Multi-modal feature hashing optimization:
            # When SGLANG_MM_BUFFER_SIZE_MB > 0, we temporarily move feature tensors to GPU
            # for faster hash computation, while avoiding OOM issues.
            from sglang.srt.multimodal.mm_utils import (
                init_feature_buffer,
                is_feature_buffer_initialized,
                reset_buffer_offset,
                try_add_to_buffer,
            )

            device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
            if not is_feature_buffer_initialized():
                init_feature_buffer(device)
            reset_buffer_offset()
            for item in ret.mm_items:
                if item.feature is not None:
                    if isinstance(item.feature, torch.Tensor):
                        item.feature = try_add_to_buffer(item.feature)

        for item in ret.mm_items:
            item.set_pad_value()

        if envs.SGLANG_MM_BUFFER_SIZE_MB.get() > 0:
            for item in ret.mm_items:
                if item.feature is not None:
                    item.feature = item.feature.to("cpu", non_blocking=True)

        optional_args = [
            "mrope_positions",
            "mrope_position_delta",
            "im_token_id",
            "im_start_id",
            "im_end_id",
            "video_token_id",
            "slice_start_id",
            "slice_end_id",
            "audio_start_id",
            "audio_end_id",
            "audio_token_id",
        ]
        for arg in optional_args:
            val = getattr(obj, arg, None)
            if val is not None:
                setattr(ret, arg, val)

        return ret

    def merge(self, other: "MultimodalInputs"):
        """
        Merge image inputs when requests are being merged.
        """
        # args needed to be merged
        optional_args = [
            "mm_items",
            "image_pad_len",
        ]
        for arg in optional_args:
            self_arg = getattr(self, arg, None)
            if self_arg is not None:
                setattr(self, arg, self_arg + getattr(other, arg))

        mrope_positions = self.mrope_positions
        if mrope_positions is not None:
            if other.mrope_positions is None:
                self.mrope_positions = mrope_positions
            else:
                self.mrope_positions = torch.cat(
                    [self.mrope_positions, other.mrope_positions], dim=1
                )

        mrope_position_delta = self.mrope_position_delta
        if mrope_position_delta is not None:
            if other.mrope_position_delta is None:
                self.mrope_position_delta = mrope_position_delta
            else:
                self.mrope_position_delta = torch.cat(
                    [self.mrope_position_delta, other.mrope_position_delta], dim=0
                )

        for key, val in other.__dict__.items():
            if "_id" in key:
                # set token_ids
                if getattr(self, key, None) is None:
                    setattr(self, key, getattr(other, key, None))
        # other args would be kept intact
