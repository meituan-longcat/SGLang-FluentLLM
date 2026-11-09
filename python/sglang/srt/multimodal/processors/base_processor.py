"""
BaseMultimodalProcessor: Base class for all multimodal processors in fluentllm.

Key responsibilities:
1. Load multimodal data (images/videos/audios) from raw input (URL, base64, etc.)
2. Preprocess data using HuggingFace AutoProcessor
3. Build input_ids with expanded multimodal tokens (e.g. <image_pad> × N)
4. Return MultimodalProcessorOutput ready for hash computation and scheduling

Design:
- process_mm_data_async(): abstract method each model-specific processor implements
- load_mm_data(): loads raw media files using thread/process pools
- process_and_combine_mm_data(): calls HF processor and collects mm_items

References:
    - sglang/python/sglang/srt/multimodal/processors/base_processor.py (upstream)
"""

import concurrent.futures
import dataclasses
import multiprocessing as mp
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch

from sglang.srt.managers.scheduling.multimodal_data import (
    Modality,
    MultimodalDataItem,
    MultimodalInputFormat,
    MultimodalProcessorOutput,
)
from sglang.srt.env import envs
from sglang.srt.utils import get_colorful_logger
from sglang.srt.multimodal.mm_utils import hash_feature, get_new_expanded_mm_items

logger = get_colorful_logger(__name__)


# ---------------------------------------------------------------------------
# BaseMultiModalProcessorOutput: raw loaded multimodal data (before HF processing)
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class BaseMultiModalProcessorOutput:
    """
    Output of load_mm_data(): contains raw loaded media objects.
    This is fed into process_and_combine_mm_data() which calls the HF processor.
    """

    # Text prompt (may be modified after token expansion)
    input_text: str

    # Raw loaded images (PIL.Image or CUDA tensors from nvJPEG)
    images: Optional[List[Any]] = dataclasses.field(default_factory=list)

    # Raw loaded video frames (torch.Tensor or VideoDecoder objects)
    videos: Optional[List[Any]] = dataclasses.field(default_factory=list)

    # Raw loaded audio data (np.ndarray)
    audios: Optional[List[Any]] = dataclasses.field(default_factory=list)

    def organize_results(self) -> List[Tuple[Modality, Any]]:
        """Return list of (modality, data) pairs in original order."""
        return (
            [(Modality.IMAGE, data) for data in (self.images or [])]
            + [(Modality.VIDEO, data) for data in (self.videos or [])]
            + [(Modality.AUDIO, data) for data in (self.audios or [])]
        )


# ---------------------------------------------------------------------------
# MultimodalSpecialTokens: token string/id/regex for each modality
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MultimodalSpecialTokens:
    """
    Stores string representations, token IDs, and regexes for special tokens
    corresponding to each modality (image, video, audio).

    Call .build(processor) to automatically convert IDs to strings and compile regexes.
    """

    image_token: Optional[Union[str, List[str]]] = None
    video_token: Optional[Union[str, List[str]]] = None
    audio_token: Optional[Union[str, List[str]]] = None

    image_token_id: Optional[int] = None
    video_token_id: Optional[int] = None
    audio_token_id: Optional[int] = None

    image_token_regex: Optional[re.Pattern] = None
    video_token_regex: Optional[re.Pattern] = None
    audio_token_regex: Optional[re.Pattern] = None

    combined_regex: Optional[re.Pattern] = None

    def build(self, processor) -> "MultimodalSpecialTokens":
        """Convert IDs to strings, parse regexes, and build combined regex."""
        self.convert_to_strs(processor)
        self.parse_regex()
        self.get_combined_regex()
        return self

    @staticmethod
    def _convert_to_str(token: Union[str, int, None], processor) -> Optional[str]:
        if token is None:
            return None
        if isinstance(token, str):
            return token
        tokenizer = getattr(processor, "tokenizer", processor)
        return tokenizer.convert_ids_to_tokens([token])[0]

    def convert_to_strs(self, processor) -> None:
        if not self.image_token:
            self.image_token = self._convert_to_str(self.image_token_id, processor)
        if not self.video_token:
            self.video_token = self._convert_to_str(self.video_token_id, processor)
        if not self.audio_token:
            self.audio_token = self._convert_to_str(self.audio_token_id, processor)

    def get_modality_of_token(self, token: str) -> Optional[Modality]:
        """Return modality if token matches a special multimodal token or its regex."""
        direct_map = {
            self.image_token: Modality.IMAGE,
            self.video_token: Modality.VIDEO,
            self.audio_token: Modality.AUDIO,
        }
        modality = direct_map.get(token)
        if modality:
            return modality
        for regex, m in [
            (self.image_token_regex, Modality.IMAGE),
            (self.video_token_regex, Modality.VIDEO),
            (self.audio_token_regex, Modality.AUDIO),
        ]:
            if regex and regex.match(token):
                return m
        return None

    def get_token_id_by_modality(self, modality: Modality) -> Optional[int]:
        return {
            Modality.IMAGE: self.image_token_id,
            Modality.VIDEO: self.video_token_id,
            Modality.AUDIO: self.audio_token_id,
        }.get(modality)

    def parse_regex(self) -> None:
        if self.image_token_regex is None and self.image_token is not None:
            self.image_token_regex = re.compile(re.escape(self.image_token))
        if self.video_token_regex is None and self.video_token is not None:
            self.video_token_regex = re.compile(re.escape(self.video_token))
        if self.audio_token_regex is None and self.audio_token is not None:
            self.audio_token_regex = re.compile(re.escape(self.audio_token))

    def get_combined_regex(self) -> re.Pattern:
        """Compile a combined regex for splitting text at all MM special tokens."""
        if self.combined_regex:
            return self.combined_regex
        tokens = [
            self.image_token_regex,
            self.video_token_regex,
            self.audio_token_regex,
        ]
        patterns = []
        flags = 0
        for t in tokens:
            if t is not None:
                patterns.append(t.pattern)
                flags |= t.flags
        if not patterns:
            # No tokens → match nothing
            self.combined_regex = re.compile(r"(?!)")
        else:
            combined = "(" + "|".join(f"(?:{p})" for p in patterns) + ")"
            self.combined_regex = re.compile(combined, flags)
        return self.combined_regex


# ---------------------------------------------------------------------------
# BaseMultimodalProcessor: abstract base for model-specific processors
# ---------------------------------------------------------------------------


def _load_single_item_fn(
    cls,
    data,
    modality: Modality,
    frame_count_limit=None,
    audio_sample_rate: Optional[int] = None,
    discard_alpha_channel: bool = True,
):
    """
    Module-level helper for loading a single multimodal item.
    Needed because class methods can't be pickled for ProcessPoolExecutor.
    """
    return cls._load_single_item(
        data, modality, frame_count_limit, audio_sample_rate, discard_alpha_channel
    )


class BaseMultimodalProcessor(ABC):
    """
    Abstract base class for multimodal data processors.

    Subclasses must implement:
        - process_mm_data_async(): full async processing pipeline for a request

    Key attributes set by __init__:
        - _processor: HuggingFace processor (or compatible wrapper)
        - _tokenizer: tokenizer extracted from processor
        - io_executor: ThreadPoolExecutor for async I/O (loading media files)
        - cpu_executor: ProcessPoolExecutor for CPU-intensive image decoding

    Key feature attributes (model-specific, set by subclasses):
        - FEATURE_NAMES: list of tensor attribute names from HF processor that
          represent raw features (pixel_values, audio_features, etc.)
        - ATTR_NAME_TO_MODALITY: maps HF processor output attr names to Modality
    """

    models: List = []
    gpu_image_decode: bool = True  # Enable GPU JPEG decoding by default

    def __init__(
        self,
        hf_config,
        server_args,
        _processor,
        transport_mode=None,
        *args,
        **kwargs,
    ):
        self.hf_config = hf_config
        self._processor = _processor
        self.server_args = server_args
        self.transport_mode = transport_mode

        mm_process_config = getattr(server_args, "mm_process_config", {}) or {}
        self.image_config = mm_process_config.get("image", {})
        self.video_config = mm_process_config.get("video", {})
        self.audio_config = mm_process_config.get("audio", {})

        # Extract tokenizer from processor
        if hasattr(self._processor, "tokenizer"):
            self._tokenizer = self._processor.tokenizer
        else:
            self._tokenizer = self._processor

        # Approximate default token count per video frame
        self.NUM_TOKEN_PER_FRAME = 330

        # I/O and CPU thread/process pools
        self.io_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=int(os.environ.get("SGLANG_IO_WORKERS", 4))
        )
        self.cpu_executor = concurrent.futures.ProcessPoolExecutor(
            mp_context=mp.get_context("fork"),
            max_workers=int(os.environ.get("SGLANG_CPU_WORKERS", os.cpu_count())),
        )

        # Mapping: HF processor output attr name → Modality
        self.ATTR_NAME_TO_MODALITY: Dict[str, Modality] = {
            # Image attributes
            "pixel_values": Modality.IMAGE,
            "image_sizes": Modality.IMAGE,
            "image_grid_thw": Modality.IMAGE,
            "image_attention_mask": Modality.IMAGE,
            "image_emb_mask": Modality.IMAGE,
            "images_spatial_crop": Modality.IMAGE,
            "images_crop": Modality.IMAGE,
            "has_local_crops": Modality.IMAGE,
            "has_images": Modality.IMAGE,
            "tgt_size": Modality.IMAGE,
            "image_grid_hws": Modality.IMAGE,
            "aspect_ratio_ids": Modality.IMAGE,
            "aspect_ratio_mask": Modality.IMAGE,
            "num_patches": Modality.IMAGE,
            "patch_pixel_values": Modality.IMAGE,
            "block_sizes": Modality.IMAGE,
            "grid_thws": Modality.IMAGE,  # Kimi K2.5
            # Audio attributes
            "audio_features": Modality.AUDIO,
            "audio_feature_lens": Modality.AUDIO,
            "input_features": Modality.AUDIO,
            "input_features_mask": Modality.AUDIO,
            "audio_attention_mask": Modality.AUDIO,
            "feature_attention_mask": Modality.AUDIO,
            # Video attributes
            "pixel_values_videos": Modality.VIDEO,
            "second_per_grid_ts": Modality.VIDEO,
            "video_grid_thw": Modality.VIDEO,
        }

        # Names of raw feature tensors in HF processor output
        self.FEATURE_NAMES = [
            "pixel_values",
            "pixel_values_videos",
            "audio_features",
            "input_features",
        ]

        # No CUDA IPC pool here — owned by CudaIpcTransportStrategy in mm_feature_transport.py.

    # ------------------------------------------------------------------
    # Abstract method: subclasses implement the full processing pipeline
    # ------------------------------------------------------------------

    @abstractmethod
    async def process_mm_data_async(
        self,
        image_data,
        audio_data,
        input_text,
        request_obj,
        **kwargs,
    ) -> Optional[MultimodalProcessorOutput]:
        """
        Full async preprocessing pipeline for one request.

        Called by TokenizerManager before scheduling.

        Args:
            image_data: list of image URLs, bytes, base64, or dicts
            audio_data: list of audio data
            input_text: raw text prompt with multimodal placeholder tokens
            request_obj: GenerateReqInput carrying all request metadata

        Returns:
            MultimodalProcessorOutput with mm_items and input_ids
        """
        ...

    # ------------------------------------------------------------------
    # Optional override: compute M-RoPE positions
    # ------------------------------------------------------------------

    def compute_mrope_positions(
        self, input_ids: List[int], mm_items: List[MultimodalDataItem]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Compute M-RoPE positions from expanded input_ids and multimodal items.
        Override in QwenVL / similar models that use 3D RoPE.

        Returns:
            (mrope_positions, mrope_position_delta) or (None, None)
        """
        return None, None

    # ------------------------------------------------------------------
    # Loading utilities
    # ------------------------------------------------------------------

    @classmethod
    def _load_single_item(
        cls,
        data,
        modality: Modality,
        frame_count_limit=None,
        audio_sample_rate: Optional[int] = None,
        discard_alpha_channel: bool = True,
    ):
        """
        Load a single media item. If data is already a processed dict
        (format='processor_output' or 'precomputed_embedding'), return as-is.

        This is a classmethod so it can be submitted to ProcessPoolExecutor.
        """
        if isinstance(data, dict):
            fmt = data.get("format")
            if fmt in (
                MultimodalInputFormat.PROCESSOR_OUTPUT.name,
                MultimodalInputFormat.PRECOMPUTED_EMBEDDING.name,
                "processor_output",
                "precomputed_embedding",
            ):
                return data
        try:
            if modality == Modality.IMAGE:
                from sglang.srt.utils import load_image
                img, _ = load_image(data, cls.gpu_image_decode)
                if (
                    discard_alpha_channel
                    and not isinstance(img, torch.Tensor)
                    and img.mode != "RGB"
                ):
                    img = img.convert("RGB")
                return img
            elif modality == Modality.VIDEO:
                from sglang.srt.utils import load_video
                return load_video(data, frame_count_limit)
            elif modality == Modality.AUDIO:
                from sglang.srt.utils import load_audio
                return load_audio(data, audio_sample_rate)
        except Exception as e:
            raise RuntimeError(f"Error while loading {modality.name} data {data}: {e}")

    def _submit_mm_data_loading_tasks_simple(
        self,
        data_list: Optional[List],
        modality: Modality,
        audio_sample_rate: Optional[int],
        discard_alpha_channel: bool,
    ) -> List[Tuple[Modality, int, concurrent.futures.Future]]:
        """Submit I/O loading tasks for a single modality; returns list of futures."""
        futures: List[Tuple[Modality, int, concurrent.futures.Future]] = []
        if not data_list:
            return futures
        for idx, data in enumerate(data_list):
            future = self.io_executor.submit(
                self.__class__._load_single_item,
                data,
                modality,
                None,
                audio_sample_rate,
                discard_alpha_channel,
            )
            futures.append((modality, idx, future))
        return futures

    def submit_data_loading_tasks(
        self,
        text_parts: List[str],
        multimodal_tokens: MultimodalSpecialTokens,
        data_iterators: Dict[Modality, Iterator[Any]],
        discard_alpha_channel: bool = True,
        image_estimated_frames_iter: Optional[Iterator] = None,
        image_scaling_factor: float = 1.0,
        max_image_frames: int = 30,
        audio_sample_rate: Optional[int] = None,
    ) -> Tuple[List, List]:
        """
        Submit I/O tasks for multimodal data loading based on token order in text_parts.

        Returns (futures, task_info) in the same order as multimodal tokens appear.
        """
        futures = []
        task_info = []

        for text_part in text_parts:
            modality = multimodal_tokens.get_modality_of_token(text_part)
            if modality is None:
                continue

            data_iterator = data_iterators.get(modality)
            if data_iterator is None:
                raise ValueError(f"No data iterator for modality: {modality}")

            try:
                data = next(data_iterator)
            except StopIteration:
                logger.warning(
                    f"Mismatch: more '{modality.name}' tokens than data items provided."
                )
                return futures, task_info

            frame_count_limit = None
            if modality == Modality.IMAGE and image_estimated_frames_iter is not None:
                try:
                    estimated_frames = next(image_estimated_frames_iter)
                    frame_count_limit = max(
                        1, int(estimated_frames * image_scaling_factor)
                    )
                except StopIteration:
                    raise ValueError(
                        "Mismatch between image tokens and estimated frame counts."
                    )

            futures.append(
                self.io_executor.submit(
                    self.__class__._load_single_item,
                    data,
                    modality,
                    frame_count_limit,
                    audio_sample_rate,
                    discard_alpha_channel,
                )
            )
            task_info.append((modality, data, frame_count_limit))

        # Warn if remaining unused data items
        for modality, iterator in data_iterators.items():
            try:
                next(iterator)
                logger.warning(
                    f"More {modality.name.lower()} data items than tokens in the prompt."
                )
            except StopIteration:
                pass
            except Exception:
                pass

        return futures, task_info

    # ------------------------------------------------------------------
    # load_mm_data: dispatch to fast_load or legacy_load path
    # ------------------------------------------------------------------

    def load_mm_data(
        self,
        prompt: Union[str, List[int]],
        multimodal_tokens: MultimodalSpecialTokens,
        image_data: Optional[List] = None,
        video_data: Optional[List] = None,
        audio_data: Optional[List] = None,
        return_text: bool = True,
        discard_alpha_channel: bool = True,
        audio_sample_rate: Optional[int] = None,
    ) -> BaseMultiModalProcessorOutput:
        """
        Load all multimodal data for a request. Dispatches to fast path or
        legacy path based on token count alignment.
        """
        self._validate_mm_data(image_data, video_data, audio_data)

        multimodal_tokens_pattern = multimodal_tokens.get_combined_regex()
        if isinstance(prompt, list) and return_text:
            assert prompt and isinstance(prompt[0], int)
            prompt = self._tokenizer.decode(prompt)

        assert isinstance(prompt, str), f"Expected str prompt, got {type(prompt)}"
        text_parts = re.split(multimodal_tokens_pattern, prompt)

        # Count token occurrences vs actual data
        cnt = {Modality.IMAGE: 0, Modality.VIDEO: 0, Modality.AUDIO: 0}
        for part in text_parts:
            m = multimodal_tokens.get_modality_of_token(part)
            if m is not None:
                cnt[m] += 1

        n_image = len(image_data) if image_data else 0
        n_video = len(video_data) if video_data else 0
        n_audio = len(audio_data) if audio_data else 0

        # Use legacy path if counts don't align or skip_tokenizer_init is set
        use_legacy = (
            getattr(self.server_args, "skip_tokenizer_init", False)
            or cnt[Modality.IMAGE] != n_image
            or cnt[Modality.VIDEO] != n_video
            or cnt[Modality.AUDIO] != n_audio
            or getattr(self, "support_dynamic_frame_expansion", False)
        )

        if use_legacy:
            return self._legacy_load_mm_data(
                prompt=prompt,
                multimodal_tokens=multimodal_tokens,
                image_data=image_data,
                video_data=video_data,
                audio_data=audio_data,
                return_text=return_text,
                discard_alpha_channel=discard_alpha_channel,
                audio_sample_rate=audio_sample_rate,
            )

        return self._fast_load_mm_data(
            prompt=prompt,
            multimodal_tokens=multimodal_tokens,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            return_text=return_text,
            discard_alpha_channel=discard_alpha_channel,
            audio_sample_rate=audio_sample_rate,
        )

    def _fast_load_mm_data(
        self,
        prompt: str,
        multimodal_tokens: MultimodalSpecialTokens,
        image_data: Optional[List] = None,
        video_data: Optional[List] = None,
        audio_data: Optional[List] = None,
        return_text: bool = True,
        discard_alpha_channel: bool = True,
        audio_sample_rate: Optional[int] = None,
    ) -> BaseMultiModalProcessorOutput:
        """
        Fast path: submit all media loading tasks concurrently.
        Assumes token count == data item count (validated before calling).
        """
        if isinstance(prompt, list) and return_text:
            prompt = self._tokenizer.decode(prompt)

        futures: List[Tuple[Modality, int, concurrent.futures.Future]] = []
        for data_list, modality in [
            (image_data, Modality.IMAGE),
            (video_data, Modality.VIDEO),
            (audio_data, Modality.AUDIO),
        ]:
            futures.extend(
                self._submit_mm_data_loading_tasks_simple(
                    data_list, modality, audio_sample_rate, discard_alpha_channel
                )
            )

        images: List[Any] = [None] * len(image_data) if image_data else []
        videos: List[Any] = [None] * len(video_data) if video_data else []
        audios: List[Any] = [None] * len(audio_data) if audio_data else []

        for modality, idx, future in futures:
            try:
                result = future.result()
            except Exception as e:
                raise RuntimeError(
                    f"Error loading {modality.name} data at index {idx}: {e}"
                )
            if modality == Modality.IMAGE:
                images[idx] = result
            elif modality == Modality.VIDEO:
                videos[idx] = result
            elif modality == Modality.AUDIO:
                audios[idx] = result

        return BaseMultiModalProcessorOutput(
            images=images,
            videos=videos,
            audios=audios,
            input_text=prompt,
        )

    def _legacy_load_mm_data(
        self,
        prompt: str,
        multimodal_tokens: MultimodalSpecialTokens,
        image_data: Optional[List] = None,
        video_data: Optional[List] = None,
        audio_data: Optional[List] = None,
        return_text: bool = True,
        discard_alpha_channel: bool = True,
        audio_sample_rate: Optional[int] = None,
    ) -> BaseMultiModalProcessorOutput:
        """
        Legacy path: scan text_parts to match tokens with data in order.
        Handles mismatched counts (e.g. one video token → multiple frame images).
        """
        multimodal_tokens_pattern = multimodal_tokens.get_combined_regex()
        if isinstance(prompt, list) and return_text:
            prompt = self._tokenizer.decode(prompt)

        text_parts = re.split(multimodal_tokens_pattern, prompt)

        data_iterators: Dict[Modality, Iterator] = {}
        if multimodal_tokens.image_token and image_data:
            data_iterators[Modality.IMAGE] = iter(image_data)
        if multimodal_tokens.video_token and video_data:
            data_iterators[Modality.VIDEO] = iter(video_data)
        if multimodal_tokens.audio_token and audio_data:
            data_iterators[Modality.AUDIO] = iter(audio_data)

        futures, task_info = self.submit_data_loading_tasks(
            text_parts=text_parts,
            multimodal_tokens=multimodal_tokens,
            data_iterators=data_iterators,
            discard_alpha_channel=discard_alpha_channel,
            audio_sample_rate=audio_sample_rate,
        )
        task_info_iter = iter(task_info)
        futures_iter = iter(futures)

        images, videos, audios = [], [], []
        new_text_parts = []
        has_precomputed = False

        for text_part in text_parts:
            try:
                if multimodal_tokens_pattern.match(text_part):
                    modality, raw_data, _ = next(task_info_iter)
                    result = next(futures_iter).result()

                    is_pre, new_imgs, new_vids, new_auds = self._process_loaded_item(
                        modality, raw_data, result
                    )
                    has_precomputed |= is_pre
                    images.extend(new_imgs)
                    videos.extend(new_vids)
                    audios.extend(new_auds)

                    if modality == Modality.IMAGE:
                        if is_pre:
                            new_text_parts.append(text_part)
                        else:
                            count = len(new_imgs)
                            if count > 0:
                                new_text_parts += [multimodal_tokens.image_token] * count
                    elif modality == Modality.VIDEO:
                        mm_tok = text_part if is_pre else multimodal_tokens.video_token
                        new_text_parts += mm_tok
                    elif modality == Modality.AUDIO:
                        mm_tok = text_part if is_pre else multimodal_tokens.audio_token
                        new_text_parts += mm_tok
                else:
                    new_text_parts.append(text_part)
            except StopIteration:
                if has_precomputed:
                    new_text_parts.append(text_part)
                    continue
                raise RuntimeError("StopIteration while loading multimodal data.")
            except Exception as e:
                raise RuntimeError(f"Error loading multimodal data: {e}")

        return BaseMultiModalProcessorOutput(
            images=images,
            videos=videos,
            audios=audios,
            input_text="".join(new_text_parts),
        )

    def _process_loaded_item(self, modality, raw_data, result):
        """Process a loaded item; detect precomputed and categorize into images/videos/audios."""
        images, videos, audios = [], [], []
        is_precomputed = isinstance(raw_data, dict) and raw_data.get("format") in [
            MultimodalInputFormat.PROCESSOR_OUTPUT.name,
            MultimodalInputFormat.PRECOMPUTED_EMBEDDING.name,
            "processor_output",
            "precomputed_embedding",
        ]
        if modality == Modality.IMAGE:
            if is_precomputed:
                images.append(result)
            else:
                if isinstance(result, list):
                    images.extend(result)
                else:
                    images.append(result)
        elif modality == Modality.VIDEO:
            videos.append(result)
        elif modality == Modality.AUDIO:
            audios.append(result)
        return is_precomputed, images, videos, audios

    # ------------------------------------------------------------------
    # HF processor call
    # ------------------------------------------------------------------
    def process_mm_data(
        self, input_text: str, images=None, videos=None, audios=None, **kwargs
    ) -> dict:
        """
        Call the HuggingFace processor to get raw tensors (pixel_values, etc.).
        Applies GPU device placement when fast image processor is available.
        """
        from transformers import BaseImageProcessorFast

        if images:
            kwargs["images"] = images
            if self.image_config:
                kwargs.setdefault("images_kwargs", {}).update(self.image_config)
        if videos:
            kwargs["videos"] = videos
            if self.video_config:
                kwargs.setdefault("videos_kwargs", {}).update(self.video_config)
        if audios:
            kwargs["audios"] = audios
            if self.audio_config:
                kwargs.setdefault("audio_kwargs", {}).update(self.audio_config)

        processor = self._processor
        if (
            hasattr(processor, "image_processor")
            and isinstance(processor.image_processor, BaseImageProcessorFast)
            and not getattr(self.server_args, "disable_fast_image_processor", False)
        ):
            kwargs["device"] = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Using fast image processor on %s", kwargs["device"])

        result = processor.__call__(
            text=[input_text],
            padding=True,
            return_tensors="pt",
            **kwargs,
        )

        # Optionally move feature tensors to CPU to save GPU memory
        keep_on_device = getattr(self.server_args, "keep_mm_feature_on_device", False)
        if not keep_on_device:
            for feature_name in self.FEATURE_NAMES:
                if feature_name in result and isinstance(result[feature_name], torch.Tensor):
                    result[feature_name] = result[feature_name].to("cpu")

        return result

    # ------------------------------------------------------------------
    # Collecting mm_items from HF processor output
    # ------------------------------------------------------------------

    def collect_mm_items_from_processor_output(
        self, data_dict: dict, modality: Optional[Modality] = None
    ) -> List[MultimodalDataItem]:
        """
        Create MultimodalDataItem list from HF processor output dict.

        Looks up each key in ATTR_NAME_TO_MODALITY to determine which modality
        it belongs to, then assembles per-modality MultimodalDataItem objects.
        """
        items: Dict[Modality, MultimodalDataItem] = {}
        for attr_name, value in data_dict.items():
            if attr_name == "input_ids":
                continue

            current_modality = modality or self.ATTR_NAME_TO_MODALITY.get(attr_name)

            if attr_name == "precomputed_embeddings":
                modality_str = data_dict.get("modality")
                current_modality = Modality.IMAGE
                if modality_str:
                    try:
                        current_modality = Modality.from_str(modality_str)
                    except ValueError:
                        pass

            if current_modality is None:
                continue

            if current_modality not in items:
                items[current_modality] = MultimodalDataItem(modality=current_modality)

            # Rename raw feature attr to 'feature' for uniform access
            key = "feature" if attr_name in self.FEATURE_NAMES else attr_name
            items[current_modality].set(key, value)

        return list(items.values())

    def _process_and_collect_mm_items(
        self,
        input_text: str,
        images=None,
        audios=None,
        videos=None,
        **kwargs,
    ) -> Tuple[List[MultimodalDataItem], torch.Tensor, dict]:
        """Helper: call HF processor then collect mm_items."""
        ret = self.process_mm_data(
            input_text=input_text, images=images, audios=audios, videos=videos, **kwargs
        )
        input_ids = ret["input_ids"].flatten()
        collected_items = self.collect_mm_items_from_processor_output(ret)
        return collected_items, input_ids, ret

    def process_and_combine_mm_data(
        self,
        base_output: BaseMultiModalProcessorOutput,
        mm_tokens: MultimodalSpecialTokens,
        **kwargs,
    ) -> Tuple[List[MultimodalDataItem], torch.Tensor, Any]:
        """
        Full pipeline: HF process → collect mm_items → add offsets → split per-image.

        Returns:
            (mm_items, input_ids, ret)
            where ret is the raw HF processor output dict (or None for text-only).
        """
        all_loaded = base_output.organize_results()
        if not all_loaded:
            # Text-only request
            input_ids = self._tokenizer(
                base_output.input_text,
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids.flatten()
            return [], input_ids, {}

        dict_items, raw_images, raw_audios, raw_videos = [], [], [], []
        for m, item in all_loaded:
            if isinstance(item, dict):
                dict_items.append((m, item))
            elif m == Modality.IMAGE:
                raw_images.append(item)
            elif m == Modality.AUDIO:
                raw_audios.append(item)
            elif m == Modality.VIDEO:
                raw_videos.append(item)

        all_collected: List[MultimodalDataItem] = []
        input_ids = None
        ret = None

        if raw_images or raw_audios or raw_videos:
            collected, input_ids, ret = self._process_and_collect_mm_items(
                input_text=base_output.input_text,
                images=raw_images,
                audios=raw_audios,
                videos=raw_videos,
                **kwargs,
            )
            all_collected = collected

        # Handle precomputed / processor_output dicts
        for m, dict_item in dict_items:
            fmt = dict_item.get("format")
            if fmt == "processor_output":
                items = self.collect_mm_items_from_processor_output(dict_item)
                for item in items:
                    item.format = MultimodalInputFormat.PROCESSOR_OUTPUT
                all_collected.extend(items)
            elif fmt == "precomputed_embedding":
                feature = dict_item.pop("feature", None)
                all_collected.append(
                    MultimodalDataItem(
                        modality=m,
                        feature=feature,
                        format=MultimodalInputFormat.PRECOMPUTED_EMBEDDING,
                        model_specific_data=dict(dict_item),
                    )
                )

        if input_ids is None:
            input_ids = self._tokenizer(
                base_output.input_text,
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids.flatten()

        # Assign offsets to each mm_item
        for mm_item in all_collected:
            mm_token_id = mm_tokens.get_token_id_by_modality(mm_item.modality)
            if mm_token_id is None:
                raise ValueError(
                    f"No token_id found for modality: {mm_item.modality}"
                )
            mm_item.offsets = self.get_mm_items_offset(input_ids, mm_token_id)

        all_collected = get_new_expanded_mm_items(all_collected)

        # Always pre-compute item.hash on the real tensor. This guarantees that
        # set_pad_value() (called by from_processor_output) never needs to call
        # hash_feature() on a proxy or a moved-to-CPU tensor, regardless of
        # which transport strategy is active.
        for item in all_collected:
            if item.hash is None:
                raw = item.feature if item.feature is not None else item.precomputed_embeddings
                if raw is not None and isinstance(raw, torch.Tensor):
                    item.hash = hash_feature(raw)

        return all_collected, input_ids, ret

    # ------------------------------------------------------------------
    # Offset utilities
    # ------------------------------------------------------------------

    @staticmethod
    def get_mm_items_offset(
        input_ids: torch.Tensor, mm_token_id: int
    ) -> List[Tuple[int, int]]:
        """
        Find contiguous runs of mm_token_id in input_ids.

        Example:
            input_ids = [1, 3, 3, 3, 4, 3, 3]
            mm_token_id = 3
            → [(1, 3), (5, 6)]
        """
        mask = input_ids == mm_token_id

        # Prepend/append False to detect boundaries without circular wrap-around
        false_pad = mask.new_zeros(1)
        padded = torch.cat([false_pad, mask, false_pad])
        # A run starts where padded goes False→True (index i in padded → i-1 in mask)
        starts = (padded[1:] & ~padded[:-1]).nonzero(as_tuple=True)[0]
        # A run ends where padded goes True→False (index i in padded → i-1 in mask)
        ends = (~padded[1:] & padded[:-1]).nonzero(as_tuple=True)[0] - 1
        return list(zip(starts.tolist(), ends.tolist()))

    @staticmethod
    def get_mm_items_offset_by_pair(
        input_ids: torch.Tensor, mm_start_id: int, mm_end_id: int
    ) -> List[Tuple[int, int]]:
        """Find token ranges between start/end token pairs (exclusive of start/end)."""
        indices_start = (input_ids == mm_start_id).nonzero(as_tuple=True)[0] + 1
        indices_end = (input_ids == mm_end_id).nonzero(as_tuple=True)[0] - 1
        return list(zip(indices_start.tolist(), indices_end.tolist()))

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_one_modality(modality: Modality, data_list: Optional[List]) -> None:
        if data_list is None:
            return
        if not isinstance(data_list, list):
            raise TypeError(
                f"{modality.name} data must be a list or None, got {type(data_list)}"
            )
        formatted_indices = [
            i
            for i, item in enumerate(data_list)
            if isinstance(item, dict)
            and item.get("format") in {"processor_output", "precomputed_embedding"}
        ]
        if formatted_indices and len(data_list) != 1:
            raise ValueError(
                f"For {modality}, when providing processor_output/precomputed_embedding, "
                f"you must pass exactly one item; got {len(data_list)} items."
            )

    @staticmethod
    def _validate_mm_data(
        image_data: Optional[List] = None,
        video_data: Optional[List] = None,
        audio_data: Optional[List] = None,
    ) -> None:
        BaseMultimodalProcessor._validate_one_modality(Modality.IMAGE, image_data)
        BaseMultimodalProcessor._validate_one_modality(Modality.VIDEO, video_data)
        BaseMultimodalProcessor._validate_one_modality(Modality.AUDIO, audio_data)

    # ------------------------------------------------------------------
    # build_input_ids: build expanded token IDs from grid info
    # Useful for models that pass pre-computed embeddings (offline API).
    # ------------------------------------------------------------------

    @property
    def spatial_merge_size(self) -> int:
        return getattr(
            getattr(self.hf_config, "vision_config", None), "spatial_merge_size", 1
        )

    def build_input_ids(
        self,
        prompt: Union[str, List[int]],
        img_grid_thw=None,
        video_grid_thw=None,
        audio_seq_lens=None,
    ) -> Tuple[List[int], List[Tuple[int, int]], List[Modality]]:
        """
        Build input_ids from a prompt token list by expanding multimodal placeholders
        using the provided grid sizes.

        Args:
            prompt: str or list of token IDs
            img_grid_thw: list of image grid tensors (T, H, W)
            video_grid_thw: list of video grid tensors (T, H, W)
            audio_seq_lens: list of audio sequence lengths (int tensors)

        Returns:
            (input_ids, offsets, modality_list)
        """
        if not isinstance(prompt, list):
            prompt = self._tokenizer.encode(prompt)

        img_token_id = getattr(self, "IM_TOKEN_ID", None)
        video_token_id = getattr(self, "VIDEO_TOKEN_ID", None)
        audio_token_id = getattr(self, "audio_token_id", None)
        spatial_merge_size = self.spatial_merge_size

        input_ids: List[int] = []
        offsets: List[Tuple[int, int]] = []
        modality_list: List[Modality] = []

        vision_starts = []
        for i in range(len(prompt) - 1):
            if img_token_id is not None and prompt[i + 1] == img_token_id:
                vision_starts.append((i, Modality.IMAGE))
            elif video_token_id is not None and prompt[i + 1] == video_token_id:
                vision_starts.append((i, Modality.VIDEO))
            elif audio_token_id is not None and prompt[i + 1] == audio_token_id:
                vision_starts.append((i, Modality.AUDIO))

        modality_list = [m for _, m in vision_starts]

        img_idx = video_idx = audio_idx = 0
        cur_idx = 0

        for mm_start_idx, modality in vision_starts:
            if modality == Modality.IMAGE:
                mm_token_num = img_grid_thw[img_idx].prod() // (spatial_merge_size ** 2)
                mm_token_id = img_token_id
                img_idx += 1
            elif modality == Modality.VIDEO:
                mm_token_num = video_grid_thw[video_idx].prod() // (spatial_merge_size ** 2)
                mm_token_id = video_token_id
                video_idx += 1
            elif modality == Modality.AUDIO:
                mm_token_num = int(audio_seq_lens[audio_idx].item())
                mm_token_id = audio_token_id
                audio_idx += 1
            else:
                raise ValueError(f"Invalid modality: {modality}")

            assert cur_idx <= mm_start_idx
            input_ids.extend(prompt[cur_idx : mm_start_idx + 1])
            mm_offset_start = len(input_ids)
            input_ids.extend([mm_token_id] * mm_token_num)
            cur_idx = mm_start_idx + 2  # skip one past the placeholder
            offsets.append((mm_offset_start, len(input_ids) - 1))

        input_ids.extend(prompt[cur_idx:])
        return input_ids, offsets, modality_list

    def get_mm_data(
        self, prompt, embeddings: Dict[Modality, torch.Tensor], **kwargs
    ) -> MultimodalProcessorOutput:
        """
        Build MultimodalProcessorOutput from pre-computed embeddings (offline API path).
        Called when users pass embeddings directly instead of raw images.

        Args:
            prompt: str or list of token IDs
            embeddings: dict mapping Modality → embedding tensor
        """
        img_grid_thw = kwargs.get("img_grid_thw")
        video_grid_thw = kwargs.get("video_grid_thw")
        audio_feature_lens = kwargs.get("audio_feature_lens")

        input_ids, offsets, modality_list = self.build_input_ids(
            prompt,
            img_grid_thw=img_grid_thw,
            video_grid_thw=video_grid_thw,
            audio_seq_lens=audio_feature_lens,
        )

        mm_items = []
        consumed_per_modality: Dict[Modality, int] = {}

        for modality, offset in zip(modality_list, offsets):
            num_tokens = offset[1] - offset[0] + 1
            start = consumed_per_modality.get(modality, 0)
            emb_slice = embeddings[modality][start : start + num_tokens]
            consumed_per_modality[modality] = start + num_tokens
            mm_items.append(
                MultimodalDataItem(
                    modality=modality,
                    offsets=[offset],
                    precomputed_embeddings=emb_slice,
                )
            )

        im_start_id = getattr(self, "IM_START_TOKEN_ID", None)
        im_end_id = getattr(self, "IM_END_TOKEN_ID", None)
        im_token_id = getattr(self, "IM_TOKEN_ID", None)
        video_token_id = getattr(self, "VIDEO_TOKEN_ID", None)
        audio_token_id = getattr(self, "audio_token_id", None)

        return MultimodalProcessorOutput(
            input_ids=input_ids,
            mm_items=mm_items,
            im_start_id=im_start_id,
            im_end_id=im_end_id,
            im_token_id=im_token_id,
            video_token_id=video_token_id,
            audio_token_id=audio_token_id,
        )
