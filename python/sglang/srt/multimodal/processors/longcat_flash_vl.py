"""
Longcat Flash VL Multimodal Processor for fluentllm.

Supports: LongcatFlashVLForConditionalGeneration

Key design:
- Images are embedded as <img_start> image_url <img_end>
- Videos are embedded as <video_start> video_path_json <video_end>
- Token pattern for image: <img_start> + <img_pad> × N + <img_end>
- Token pattern for video: <video_start> + <video_place> × N + <video_end>
- No M-RoPE; uses standard 1D RoPE (from FLASHForCausalLM)
- Patchify uses merge-group-first ordering (spatial_merge_size grouping)

The VQ quantization happens in the model's vision tower forward pass,
so the processor only needs to produce pixel_values and grid_thw.
"""

import hashlib
import math
import os
import re
import time
from array import array
from io import BytesIO
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
import requests as _requests
import torch
from PIL import Image

from sglang.srt.managers.scheduling.multimodal_data import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
    _compute_pad_value,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.multimodal.processors.maas_video_frames import (
    is_maas_video_frame_sequence,
    parse_maas_video_frame_sequence,
)
from sglang.srt.disaggregation.encode.encoder_base import (
    EncoderPreprocessor, PreprocessedItem,
)
from sglang.srt.disaggregation.encode.encode_data import MMItemMeta
from sglang.srt.multimodal.mm_utils import hash_feature

from sglang.srt.utils.common import get_image_bytes, load_image
from sglang.srt.utils import get_colorful_logger,is_npu

if TYPE_CHECKING:
    from sglang.srt.disaggregation.encode.encode_data import MMItemMeta
    from sglang.srt.disaggregation.encode.utils import Timer

logger = get_colorful_logger(__name__)


# ============================================================
# Constants
# ============================================================

IMAGE_FACTOR = 28  # patch_size * merge_size = 14 * 2
MIN_PIXELS = 256 * 14 * 14
MAX_PIXELS = 16384 * 14 * 14
MAX_RATIO = 200

# Video pixel budget defaults (per-frame, not per-video)
VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28    # default per-frame pixel cap
VIDEO_MAX_FRAMES = 32               # max sampled frames
VIDEO_DECODE_WAY = "1fps"           # "1fps" or "key"

IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]

# Default token IDs (overridden by config when available).
# Image token IDs
IMAGE_START_TOKEN_ID = 223   # <img_start>
IMAGE_END_TOKEN_ID = 222     # <img_end>
IMAGE_PAD_TOKEN_ID = 221     # <img_pad>
# Video token IDs
VIDEO_START_TOKEN_ID = 216   # <video_start>
VIDEO_END_TOKEN_ID = 215     # <video_end>
VIDEO_PLACE_TOKEN_ID = 214   # <video_palce>  NOTE: tokenizer has a typo (palce vs place)

IMAGE_START_TOKEN = "<longcat_img_start>"
IMAGE_END_TOKEN = "<longcat_img_end>"
IMAGE_PAD_TOKEN = "<longcat_img_pad>"
VIDEO_START_TOKEN = "<longcat_video_start>"
VIDEO_END_TOKEN = "<longcat_video_end>"
# NOTE: tokenizer stores this token as "<video_palce>" (typo), but the string used in
# prompts/config is still "<video_place>". _get_token_str decodes the configured
# ID to the actual tokenizer string "<video_palce>", so regex matching works.
VIDEO_PLACE_TOKEN = "<longcat_video_place>"


def _get_device_normalization_tensors(
    processor,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return cached ``[1,C,1,1]`` mean/std tensors for one device/dtype.

    The cache belongs to the processor instance, so processors with different
    normalization constants never share tensors. At worst two concurrent
    first requests may allocate the same small pair; subsequent reads reuse
    the cached tensors.
    """
    device = torch.device(device)
    key = (device.type, device.index, dtype)
    cache = getattr(processor, "_device_norm_tensor_cache", None)
    if cache is None:
        cache = {}
        processor._device_norm_tensor_cache = cache
    tensors = cache.get(key)
    if tensors is None:
        mean = torch.tensor(
            processor.image_mean, device=device, dtype=dtype
        ).view(1, -1, 1, 1)
        std = torch.tensor(
            processor.image_std, device=device, dtype=dtype
        ).view(1, -1, 1, 1)
        tensors = (mean, std)
        cache[key] = tensors
    return tensors


# ============================================================
# Image preprocessing utilities
# ============================================================


def _token_ids_view(input_ids: List[int]) -> torch.Tensor:
    """Zero-copy int64 tensor over token ids for vectorized offset search.

    ``torch.as_tensor(list)`` converts element by element while holding the
    GIL (~60 ms at 680K tokens); ``array("q", ...)`` is a C-level copy.
    """
    if not input_ids:
        return torch.empty(0, dtype=torch.int64)
    return torch.frombuffer(array("q", input_ids), dtype=torch.int64)


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> Tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, "
            f"got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def split_text(text, match_regex):
    """Split text by regex matches, returning segments and match flags.
    """
    matches = list(re.finditer(match_regex, text))
    # 初始化结果列表
    result = []
    match_flag_list = []
    # 上一个匹配的结束位置
    last_end = 0
    # 遍历所有匹配项
    for match in matches:
        # 添加匹配项之前的部分
        if text[last_end:match.start()]:
            result.append(text[last_end:match.start()])
            match_flag_list.append(False)
        # 添加匹配项
        result.append(match.group(0))
        match_flag_list.append(True)
        # 更新上一个匹配的结束位置
        last_end = match.end()
    # 添加最后一个匹配项之后的部分
    if text[last_end:]:
        result.append(text[last_end:])
        match_flag_list.append(False)
    return result, match_flag_list


# ============================================================
# OmniImageProcessor
# ============================================================
class OmniImageProcessor:
    """Image processor.

    Pipeline:
    1. PIL Image -> uint8 HWC numpy
    2. PIL BICUBIC resize (uint8 domain, via transformers.image_transforms.resize)
    3. Transpose to CHW, normalize in float64, then cast final patches to
       contiguous float32 at the helper boundary
    4. Tile for temporal_patch_size
    5. Reshape with spatial_merge_size grouping + transpose(0,3,6,4,7,2,1,5,8)
    6. Flatten to [N_patches, patch_dim]
    """

    def __init__(self, patch_size=14, temporal_patch_size=2, spatial_merge_size=2,
                 min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
                 image_mean=None, image_std=None):
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size
        self.merge_size = spatial_merge_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.image_mean = image_mean or IMAGE_MEAN
        self.image_std = image_std or IMAGE_STD

    def image_transform(self, image_input, return_mm_data=True, fix_res: int = -1,
                        min_pixels: Optional[int] = None, max_pixels: Optional[int] = None):
        """Transform image to patches.

        Args:
            image_input: PIL Image, file path, URL, data: URI, raw base64
                string, or bytes. Loading reuses sglang.srt.utils.load_image
                (the same loader used by load_mm_data) for consistent handling
                across all input formats.
            min_pixels: per-call override of pixel lower bound (falls back to self.min_pixels)
            max_pixels: per-call override of pixel upper bound (falls back to self.max_pixels)

        Returns:
            flatten_patches: [grid_t * grid_h * grid_w, patch_dim] numpy float32
            image_org_size: (h, w) original size
            grid_thw: (grid_t, grid_h, grid_w) tuple
        """
        from transformers.image_transforms import resize
        from transformers.image_utils import PILImageResampling

        eff_min_pixels = min_pixels if min_pixels is not None else self.min_pixels
        eff_max_pixels = max_pixels if max_pixels is not None else self.max_pixels

        image = None
        if return_mm_data:
            image, _ = load_image(image_input, gpu_image_decode=False)

        image = image.convert("RGB")
        image = np.array(image)
        image_org_size = image.shape[:2]

        # resize
        resized_height, resized_width = smart_resize(
            image_org_size[0], image_org_size[1],
            factor=self.patch_size * self.spatial_merge_size,
            min_pixels=eff_min_pixels,
            max_pixels=eff_max_pixels,
        )
        output_size = (resized_height, resized_width)

        # 使用 resize 函数将图像调整到 output_size 大小。PILImageResampling.BICUBIC 指定使用双三次插值法
        image = resize(image, output_size, PILImageResampling.BICUBIC)
        img = image.transpose(2, 0, 1)
        # Preserve the established float64 arithmetic, then enforce the
        # Encoder's float32 output contract once at the helper boundary.
        image = (img / 255.0 - np.array(self.image_mean)[:, None, None]) / np.array(self.image_std)[:, None, None]
        # 处理成patch
        patches = image[np.newaxis, :]
        if patches.shape[0] == 1:
            patches = np.tile(patches, (self.temporal_patch_size, 1, 1, 1))
        channel = patches.shape[1]
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size

        patches = patches.reshape(
            grid_t,
            self.temporal_patch_size,
            channel,
            grid_h // self.spatial_merge_size,
            self.spatial_merge_size,
            self.patch_size,
            grid_w // self.spatial_merge_size,
            self.spatial_merge_size,
            self.patch_size,
        )
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w, channel * self.temporal_patch_size * self.patch_size * self.patch_size
        )

        flatten_patches = np.ascontiguousarray(flatten_patches, dtype=np.float32)
        return flatten_patches, image_org_size, (grid_t, grid_h, grid_w)

    # ── GPU image transform (encoder pre-compute optimization) ───────────
    @staticmethod
    def _pil_to_chw_tensor(image: "Image.Image") -> torch.Tensor:
        """PIL.Image -> CPU **uint8** (C,H,W) tensor (values 0..255).

        Single-copy chain: PIL pixels -> one writable numpy array -> zero-copy
        torch view.  The uint8→fp32 expansion (1080p: 6.2MB → 24.9MB, the
        dominant cost of the old float() here) is deferred to the GPU — see
        gpu_image_transform's h2d phase — where it runs as a kernel, and the
        H2D upload itself shrinks 4× (uint8).
        """
        if image.mode != "RGB":
            image = image.convert("RGB")  # PIL convert copies unconditionally
        arr = np.array(image)  # HWC uint8, one writable copy (was asarray+arr.copy: two)
        return torch.from_numpy(arr).permute(2, 0, 1)  # uint8 view, zero-copy

    def gpu_image_transform(
        self,
        image_input,
        device: torch.device,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int, int]]:
        """GPU image transform — Resize+Normalize+Patchify on device.

        Mirrors ``OmniImageProcessor.image_transform`` but runs the heavy stages on
        GPU. Decode stays on CPU (PIL via ``load_image(gpu_image_decode=False)``)
        so the only numerical delta vs the CPU path is the bicubic resize kernel
        (PIL vs torch) — keeping the delta source singular makes the equivalence
        gate meaningful. nvJPEG is *not* used (its decode delta ~0.9 blows the
        gate).

        Unlike Ascend NPU (aclnnInplaceCopy 8-dim limit -> EZ1001), CUDA has no
        tensor-rank limit, so the 9-D patchify intermediate runs on device — NO
        D2H. The helper returns a contiguous float32 CUDA tensor, so the caller
        does not add a D2H/H2D round-trip.

        Returns:
            flatten_patches: [grid_t*grid_h*grid_w, patch_dim] float32 CUDA tensor
            image_org_size: (h, w) original size
            grid_thw: (grid_t, grid_h, grid_w) tuple
        """
        # Lazy import: encoder/__init__ cascades into encoder.server →
        # multimodal_processor → this module — a top-level import would close an
        # import cycle.  At call time the package is long fully initialized.
        from sglang.srt.disaggregation.encode.utils import Timer

        # Per-stage timing.  GPU stages are async-launched: without
        # GPU_TRANSFORM_PROF_SYNC=1 their phases measure CPU-side launch cost only
        # (the kernel tail lands on the caller's next sync point, e.g. a .cpu()
        # downstream).  Set GPU_TRANSFORM_PROF_SYNC=1 to synchronize at each GPU
        # phase boundary for accurate kernel timings (serializes the pipeline —
        # debug only).  GPU_TRANSFORM_PROF=1 elevates the summary log to INFO.
        prof_sync = os.environ.get("GPU_TRANSFORM_PROF_SYNC", "0") == "1"
        prof_info = (
            os.environ.get("GPU_TRANSFORM_PROF", "0") == "1" or prof_sync
        )
        timer = Timer("gpu_image_transform")

        def _maybe_sync():
            if prof_sync:
                torch.cuda.synchronize(device)

        eff_min_pixels = min_pixels if min_pixels is not None else self.min_pixels
        eff_max_pixels = max_pixels if max_pixels is not None else self.max_pixels

        # load + convertRGB (CPU, same as CPU path).  PIL Image.open is LAZY
        # (header-only): the actual zlib/libjpeg decode fires on first pixel
        # access.  Force it here so this phase reflects the true decode cost
        # (1080p PNG ~18ms) instead of letting it leak into the to_chw phase.
        with timer.phase("load_decode"):
            image, _ = load_image(image_input, gpu_image_decode=False)
            if isinstance(image, Image.Image):
                image.load()
        image_org_size = image.size[::-1]  # (H, W) from PIL (W,H); size is mode-agnostic

        with timer.phase("to_chw"):
            chw = self._pil_to_chw_tensor(image)  # (C,H,W) uint8 on CPU (converts RGB)

        # smart resize (CPU, pure integer — must match CPU path bit-for-bit)
        with timer.phase("smart_resize"):
            resized_h, resized_w = smart_resize(
                image_org_size[0],
                image_org_size[1],
                factor=self.patch_size * self.spatial_merge_size,
                min_pixels=eff_min_pixels,
                max_pixels=eff_max_pixels,
            )

        # uint8 H2D (¼ the bytes of fp32) then GPU-side float cast — pageable H2D
        # is effectively synchronous, so this phase ≈ real upload+cast time.
        # uint8→fp32 is exact, so numerics match the old CPU-float path bit-for-bit.
        with timer.phase("h2d"):
            t = chw.unsqueeze(0).to(device).float()  # (1,C,H,W) fp32 on device

        # resize (BICUBIC, antialias) on GPU
        with timer.phase("resize_gpu"):
            if (resized_h, resized_w) != (chw.shape[1], chw.shape[2]):
                t = torch.nn.functional.interpolate(
                    t, size=(resized_h, resized_w), mode="bicubic",
                    align_corners=False, antialias=True,
                )
                t = t.clamp(0, 255)  # bicubic overshoot — PIL never exceeds 0..255
            _maybe_sync()

        # normalize (mean/std) on GPU
        with timer.phase("normalize_gpu"):
            mean, std = _get_device_normalization_tensors(
                self, device, t.dtype
            )
            t = (t / 255.0 - mean) / std  # (1,C,H,W)
            _maybe_sync()

        # tile + patchify (reshape/permute) on GPU
        with timer.phase("patchify_gpu"):
            tps = self.temporal_patch_size
            if t.shape[0] == 1:
                t = t.repeat(tps, 1, 1, 1)  # (tps,C,H,W)
            c = t.shape[1]
            grid_t = t.shape[0] // tps
            grid_h = resized_h // self.patch_size
            grid_w = resized_w // self.patch_size
            merge = self.spatial_merge_size
            ps = self.patch_size
            patches = t.reshape(
                grid_t, tps, c,
                grid_h // merge, merge, ps,
                grid_w // merge, merge, ps,
            ).permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
            flatten_patches = patches.reshape(
                grid_t * grid_h * grid_w,
                c * tps * ps * ps,
            ).contiguous()
            _maybe_sync()

        timer.mark_total()
        (logger.info if prof_info else logger.debug)(
            "gpu_image_transform: org=%dx%d resized=%dx%d patches=%s out=%.1fMB %s",
            image_org_size[1], image_org_size[0], resized_w, resized_h,
            tuple(flatten_patches.shape),
            flatten_patches.nelement() * flatten_patches.element_size() / 1e6,
            timer.format(),
        )

        return flatten_patches, image_org_size, (grid_t, grid_h, grid_w)


# ============================================================
# Video decoding helpers
# ============================================================

def read_video(image_path, max_frame_number, decode_way):
    """Decode video frames."""
    import io as _io
    from decord import VideoReader, cpu

    if decode_way == '1fps':
        try:
            vr = VideoReader(image_path, ctx=cpu(0), num_threads=1)
            total_frame_num = len(vr)
            fps = round(vr.get_avg_fps())
            frame_idx = [i for i in range(0, len(vr), fps)]
            frames = vr.get_batch(frame_idx).asnumpy()
            cnt = len(frames)
            frame_times = range(cnt)
        except Exception as e:
            print('error is', e)
            return None
    elif decode_way == 'key':
        try:
            import av
            with av.open(image_path) as container:
                stream = container.streams.video[0]
                stream.codec_context.skip_frame = 'NONKEY'
                frames = []
                frame_times = []
                fps = int(stream.average_rate)
                cnt = 0
                for frame in container.decode(stream):
                    image = np.array(frame.to_image())
                    frames.append(image)
                    frame_time = int(frame.time)
                    frame_times.append(frame_time)
                    cnt += 1
        except Exception as e:
            print('error is', e)
            return None
    if frames is None or len(frames) == 0:
        return None
    if len(frames) > max_frame_number and max_frame_number > 0:
        indices = np.linspace(0, len(frames) - 1, max_frame_number, dtype=int)
        frames = frames[indices]
        frame_times = [frame_times[i] for i in indices]
    return frames, frame_times


# ============================================================
# OmniVideoProcessor
# ============================================================

class OmniVideoProcessor:
    """Video processor."""

    # Dynamic pixel budget thresholds
    # self.frame_tokens = [64, 128, 256, 512, 1024, 2048]
    # border for inference: [512, 256, 128, 64, 32, 16]
    frame_tokens = [64, 128, 256, 512, 1024, 2048]

    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
        min_pixels: int = VIDEO_MIN_PIXELS,
        max_pixels: int = VIDEO_MAX_PIXELS,
        image_mean=None,
        image_std=None,
        dynamic_frame_pixel: bool = False,
        split_video: bool = False,
    ):
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size
        self.merge_size = spatial_merge_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        # image_mean/std stored on a mock config-like attribute (processor_omni accesses self.config.image_mean)
        # We store directly on self for direct access
        self.image_mean = image_mean or IMAGE_MEAN
        self.image_std = image_std or IMAGE_STD
        self.dynamic_frame_pixel = dynamic_frame_pixel
        self.split_video = split_video

    def extracted_images_transform(
        self, path_list, return_mm_data=True, train_max_length=None,
        training=False, max_pixels=None,
    ):
        """Transform a list of frame paths/PIL images into video patches.
        Uses inference branch of dynamic_frame_pixel (training=False).
        """
        from transformers.image_transforms import resize
        from transformers.image_utils import PILImageResampling

        extracted_images = []
        if max_pixels is not None:
            effective_max_pixels = max_pixels
        elif self.dynamic_frame_pixel:
            if training:
                border = []
                for frame_token in self.frame_tokens:
                    border.append(train_max_length // frame_token)
            else:
                border = [512, 256, 128, 64, 32, 16]
            video_frame_num = len(path_list)
            effective_max_pixels = self.max_pixels
            for idx in range(1, len(border)):
                if video_frame_num > border[idx]:
                    effective_max_pixels = self.frame_tokens[idx - 1] * 14 * 14 * 4
                    break
            if video_frame_num <= border[-1]:
                effective_max_pixels = self.frame_tokens[-1] * 14 * 14 * 4
        else:
            effective_max_pixels = self.max_pixels

        for strseq in path_list:
            image = None
            if isinstance(strseq, str):
                if return_mm_data:
                    max_attempts = 3
                    attempts = 0
                    while attempts < max_attempts:
                        try:
                            image = Image.open(strseq).convert("RGB")
                            break
                        except Exception as e:
                            attempts += 1
                            if attempts == max_attempts:
                                raise
                            time.sleep(0.1)
            elif isinstance(strseq, Image.Image):
                image = strseq.convert("RGB")
            else:
                image = Image.open(BytesIO(strseq)).convert("RGB")

            image = np.array(image.convert("RGB"))
            image_org_size = image.shape[:2]

            resized_height, resized_width = smart_resize(
                image_org_size[0], image_org_size[1],
                factor=self.patch_size * self.spatial_merge_size,
                min_pixels=self.min_pixels,
                max_pixels=effective_max_pixels,
            )
            output_size = (resized_height, resized_width)
            image = resize(image, output_size, PILImageResampling.BICUBIC)
            img = image.transpose(2, 0, 1)
            image = (img / 255.0 - np.array(self.image_mean)[:, None, None]) / np.array(self.image_std)[:, None, None]
            extracted_images.append(image)

        # 处理成patch
        patches = np.asarray(extracted_images)

        # Check that videos have `num_frames` divisible by `temporal_patch_size`
        if patches.shape[0] % self.temporal_patch_size != 0:
            repeats = np.tile(patches[-1:], (self.temporal_patch_size - 1, 1, 1, 1))
            patches = np.concatenate((patches, repeats), axis=0)

        channel = patches.shape[1]
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size
        patches = patches.reshape(
            grid_t,
            self.temporal_patch_size,
            channel,
            grid_h // self.spatial_merge_size,
            self.spatial_merge_size,
            self.patch_size,
            grid_w // self.spatial_merge_size,
            self.spatial_merge_size,
            self.patch_size,
        )
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w, channel * self.temporal_patch_size * self.patch_size * self.patch_size
        )

        flatten_patches = np.ascontiguousarray(flatten_patches, dtype=np.float32)
        return flatten_patches, image_org_size, (grid_t, grid_h, grid_w)

    def gpu_video_transform(
        self,
        frame_inputs,
        device: torch.device,
        max_pixels: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int, int]]:
        """Resize, normalize, temporal-pad and patchify video frames on CUDA.

        Decode remains on CPU and produces one writable uint8 HWC allocation
        per frame. A normal fixed-resolution video is packed into one uint8
        batch, uploaded once, and processed entirely on CUDA. Unlike the NPU
        helper, CUDA supports the 9-D patchify intermediate, so the returned
        contiguous float32 tensor remains device-resident for the ViT caller.
        """
        from sglang.srt.disaggregation.encode.utils import Timer

        prof_sync = os.environ.get("GPU_TRANSFORM_PROF_SYNC", "0") == "1"
        prof_info = (
            os.environ.get("GPU_TRANSFORM_PROF", "0") == "1" or prof_sync
        )
        timer = Timer("gpu_video_transform")

        def _maybe_sync():
            if prof_sync:
                torch.cuda.synchronize(device)

        with timer.phase("load_decode"):
            images = []
            for frame_input in frame_inputs:
                if isinstance(frame_input, str):
                    max_attempts = 3
                    for attempt in range(max_attempts):
                        try:
                            image = Image.open(frame_input)
                            image.load()
                            break
                        except Exception:
                            if attempt + 1 == max_attempts:
                                raise
                            time.sleep(0.1)
                elif isinstance(frame_input, Image.Image):
                    image = frame_input
                    image.load()
                else:
                    image = Image.open(BytesIO(frame_input))
                    image.load()
                images.append(
                    image if image.mode == "RGB" else image.convert("RGB")
                )
        if not images:
            raise ValueError("video contains no sampled frames")

        if max_pixels is not None:
            effective_max_pixels = max_pixels
        elif self.dynamic_frame_pixel:
            border = [512, 256, 128, 64, 32, 16]
            effective_max_pixels = self.max_pixels
            for index in range(1, len(border)):
                if len(images) > border[index]:
                    effective_max_pixels = (
                        self.frame_tokens[index - 1] * 14 * 14 * 4
                    )
                    break
            if len(images) <= border[-1]:
                effective_max_pixels = self.frame_tokens[-1] * 14 * 14 * 4
        else:
            effective_max_pixels = self.max_pixels

        original_sizes = [image.size[::-1] for image in images]
        with timer.phase("smart_resize"):
            resized_sizes = [
                smart_resize(
                    height,
                    width,
                    factor=self.patch_size * self.spatial_merge_size,
                    min_pixels=self.min_pixels,
                    max_pixels=effective_max_pixels,
                )
                for height, width in original_sizes
            ]
        if len(set(resized_sizes)) != 1:
            raise ValueError(
                "video frames resolve to different resized shapes: "
                f"{sorted(set(resized_sizes))}"
            )
        resized_h, resized_w = resized_sizes[0]

        # Keep decoded pixels uint8 until H2D. Fixed-resolution videos use one
        # allocation and one H2D; unusual variable-resolution inputs fall back
        # to per-frame upload while preserving the same output contract.
        with timer.phase("to_chw"):
            if len(set(original_sizes)) == 1:
                height, width = original_sizes[0]
                batch_hwc = np.empty(
                    (len(images), height, width, 3), dtype=np.uint8
                )
                for index, image in enumerate(images):
                    batch_hwc[index] = np.asarray(image, dtype=np.uint8)
                chw_batches = [
                    torch.from_numpy(batch_hwc).permute(0, 3, 1, 2)
                ]
            else:
                chw_batches = [
                    OmniImageProcessor._pil_to_chw_tensor(image).unsqueeze(0)
                    for image in images
                ]

        resized_batches = []
        for chw_batch in chw_batches:
            with timer.phase("h2d"):
                tensor = chw_batch.to(device).float()
            with timer.phase("resize_gpu"):
                if tuple(chw_batch.shape[-2:]) != (resized_h, resized_w):
                    tensor = torch.nn.functional.interpolate(
                        tensor,
                        size=(resized_h, resized_w),
                        mode="bicubic",
                        align_corners=False,
                        antialias=True,
                    ).clamp(0, 255)
                _maybe_sync()
            resized_batches.append(tensor)
        tensor = (
            resized_batches[0]
            if len(resized_batches) == 1
            else torch.cat(resized_batches, dim=0)
        )

        with timer.phase("normalize_gpu"):
            mean, std = _get_device_normalization_tensors(
                self, device, tensor.dtype
            )
            tensor = (tensor / 255.0 - mean) / std
            _maybe_sync()

        with timer.phase("patchify_gpu"):
            temporal_patch_size = self.temporal_patch_size
            remainder = tensor.shape[0] % temporal_patch_size
            if remainder:
                tensor = torch.cat(
                    [
                        tensor,
                        tensor[-1:].repeat(
                            temporal_patch_size - remainder, 1, 1, 1
                        ),
                    ],
                    dim=0,
                )
            channel = tensor.shape[1]
            grid_t = tensor.shape[0] // temporal_patch_size
            grid_h = resized_h // self.patch_size
            grid_w = resized_w // self.patch_size
            merge = self.spatial_merge_size
            patch_size = self.patch_size
            patches = tensor.reshape(
                grid_t,
                temporal_patch_size,
                channel,
                grid_h // merge,
                merge,
                patch_size,
                grid_w // merge,
                merge,
                patch_size,
            ).permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
            flatten_patches = patches.reshape(
                grid_t * grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            ).contiguous()
            _maybe_sync()

        timer.mark_total()
        (logger.info if prof_info else logger.debug)(
            "gpu_video_transform: frames=%d org=%dx%d resized=%dx%d "
            "patches=%s out=%.1fMB %s",
            len(images), original_sizes[-1][1], original_sizes[-1][0],
            resized_w, resized_h, tuple(flatten_patches.shape),
            flatten_patches.nelement() * flatten_patches.element_size() / 1e6,
            timer.format(),
        )
        return flatten_patches, original_sizes[-1], (
            grid_t, grid_h, grid_w
        )


# ============================================================
# VideoProcessingPipeline
#   High-level pipeline: raw video source → token string + feature tensor
#
# Relationship with OmniVideoProcessor:
#   OmniVideoProcessor  — low-level frame feature extractor
#                         (resize → normalize → patchify → numpy array)
#   VideoProcessingPipeline — high-level pipeline that sits on top:
#                         (URL/file/base64 I/O → decode frames → build
#                          frame-sequence token string → call
#                          OmniVideoProcessor → return token_str + feat)
# ============================================================

class VideoProcessingPipeline:
    """High-level video processing pipeline for LongcatFlashVLImageProcessor.

    Handles everything from raw video input to model-ready token strings
    and feature tensors.  ``LongcatFlashVLImageProcessor`` delegates all
    video logic here so that it can stay focused on SGLang integration
    (tokenisation, offset assignment, mm_items construction).

    Contrast with ``OmniVideoProcessor`` (the low-level collaborator):
      OmniVideoProcessor        — frame-level feature extractor:
                                  PIL Image list → patchified numpy array
      VideoProcessingPipeline   — end-to-end pipeline:
                                  video URL/path/base64
                                    → download / read bytes
                                    → decode & sample frames
                                    → serialise frame-sequence string
                                    → OmniVideoProcessor.extracted_images_transform
                                    → (feat, patch_nums, grid_thw, token_str)

    Constructor arguments:
      frame_processor     OmniVideoProcessor instance (low-level extractor)
      image_start_token   string token for <img_start>
      image_end_token     string token for <img_end>
      video_start_token   string token for <video_start>
      video_end_token     string token for <video_end>
      video_place_token   string token for <video_place>
      frame_pattern       per-frame serialisation template
      videoframe_start_tag / videoframe_end_tag  pre-extracted frame tags
      video_max_frames    int
      video_decode_way    "1fps" or "key"
    """

    def __init__(
        self,
        frame_processor: "OmniVideoProcessor",
        image_start_token: str,
        image_end_token: str,
        video_start_token: str,
        video_end_token: str,
        video_place_token: str,
        frame_pattern: str,
        videoframe_start_tag: str,
        videoframe_end_tag: str,
        video_max_frames: int,
        video_decode_way: str,
    ):
        # frame_processor: low-level OmniVideoProcessor (frame feature extractor)
        self.video_processor = frame_processor
        self.image_start_token = image_start_token
        self.image_end_token = image_end_token
        self.video_start_token = video_start_token
        self.video_end_token = video_end_token
        self.video_place_token = video_place_token
        self.frame_pattern = frame_pattern
        self.videoframe_start_tag = videoframe_start_tag
        self.videoframe_end_tag = videoframe_end_tag
        self.video_max_frames = video_max_frames
        self.video_decode_way = video_decode_way

        # ------------------------------------------------------------------
        # Disk cache configuration for URL videos
        #
        # VIDEO_URL_CACHE_DIR (env):
        #   When non-empty, downloaded videos AND their decoded frame directories
        #   are stored here persistently (keyed by md5(url)), allowing the same
        #   video URL to be reused across requests / evaluation runs without
        #   re-downloading.
        #   Default: "" (empty) → fall back to a per-request tempfile.mkdtemp().
        #
        # VIDEO_TEMP_VIDEO_CLEANUP (env):
        #   Set to "1" / "true" / "yes" to delete per-request temp directories
        #   after frames have been consumed.
        #   Has no effect when VIDEO_URL_CACHE_DIR is set (persistent cache is
        #   never cleaned up).
        # ------------------------------------------------------------------
        self.video_url_cache_dir: str = os.environ.get("VIDEO_URL_CACHE_DIR", "").strip()
        _cleanup_env = os.environ.get("VIDEO_TEMP_VIDEO_CLEANUP", "1").strip().lower()
        self.video_temp_cleanup: bool = _cleanup_env in ("1", "true", "yes")

    # ------------------------------------------------------------------
    # Video I/O helpers
    # ------------------------------------------------------------------

    def _get_video_bytes(self, video_info: str) -> Optional[bytes]:
        """Load video bytes from URL / local path / base64 data URI.

        Supports three input forms (matching image loading logic):
          1. HTTP/HTTPS URL:  "https://example.com/video.mp4"
          2. Local file path: "/path/to/video.mp4"
          3. Data URI base64: "data:video/mp4;base64,..."

        Returns raw video bytes, or None on failure.
        """
        if video_info.startswith("http://") or video_info.startswith("https://"):
            try:
                # 使用 requests.get 而非 urllib.request.urlopen：
                #   1. requests 库在内部自动对 URL 做 percent-encoding，
                #      可正确处理含空格（如 "Information Synopsis"）的 URL；
                #   2. urllib.request.urlopen 要求 URL 不含控制字符，
                #      对含空格的原始 URL 会抛出 "URL can't contain control characters"；
                #   3. verify=False 等效于原先 ssl_ctx.verify_mode=CERT_NONE，
                #      跳过服务端证书校验；
                #   4. 与 fluentllm 内部 get_image_bytes (srt/utils/common.py) 保持一致。
                timeout = int(os.environ.get("REQUEST_TIMEOUT", "30"))
                response = _requests.get(video_info, timeout=timeout, verify=False)
                response.raise_for_status()
                return response.content
            except Exception as e:
                logger.warning("Failed to download video URL %s: %s", video_info, e)
                return None
        elif os.path.isfile(video_info):
            with open(video_info, "rb") as f:
                return f.read()
        elif video_info.startswith('data:'):
            import base64 as _b64
            _b64_data = video_info.split(',', 1)[1] if ',' in video_info else video_info
            return _b64.b64decode(_b64_data)
        else:
            logger.warning(
                "_get_video_bytes: unrecognised video_info type, "
                "treating as local path: %r", video_info
            )
            if os.path.exists(video_info):
                with open(video_info, "rb") as f:
                    return f.read()
            return None

    # ------------------------------------------------------------------
    # Backend A: disk-based frame decoding
    # ------------------------------------------------------------------

    def _decode_video_bytes_to_frames_disk(
        self,
        mm_obj_byte: bytes,
        frame_path: str,
        decode_way: str,
        max_frame_number: int,
    ) -> Tuple[List[str], List[int]]:
        """Decode video bytes to JPEG frame files on disk.

        Args:
            mm_obj_byte:      Raw video bytes.
            frame_path:       Absolute path to the directory where JPEG frame
                              files will be written (created if absent).
            decode_way:       "1fps" or "key".
            max_frame_number: Maximum number of sampled frames (≤0 means no limit).

        Returns:
            (sampled_frame_paths, sampled_frame_times) where *sampled_frame_paths*
            are absolute JPEG file paths and *sampled_frame_times* are integer
            timestamps (seconds) corresponding to each path.

        Raises:
            ValueError: if *decode_way* is invalid.
            RuntimeError: if frame decoding fails.
        """
        import io as _io
        import cv2

        if decode_way not in ('1fps', 'key'):
            raise ValueError(f"Invalid decode_way: {decode_way!r}")

        _cache_hit = os.path.exists(frame_path) and len(os.listdir(frame_path)) > 0
        if not _cache_hit:
            os.makedirs(frame_path, exist_ok=True)
            result = read_video(_io.BytesIO(mm_obj_byte), max_frame_number=-1, decode_way=decode_way)
            if result is None:
                raise RuntimeError("read_video returned None")
            frames, frame_times = result
            for idx, frame in enumerate(frames):
                out_path = os.path.join(frame_path, f"{frame_times[idx]}.jpg")
                cv2.imwrite(out_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        # Collect and sort cached frame timestamps
        all_times = sorted(
            int(fn.replace('.jpg', ''))
            for fn in os.listdir(frame_path)
            if fn.endswith('.jpg')
        )
        frame_number = len(all_times)

        # Uniform sampling with tps alignment
        sample_num = min(frame_number, max_frame_number) if max_frame_number > 0 else frame_number
        sample_num = floor_by_factor(sample_num, self.video_processor.temporal_patch_size)
        if sample_num == 0:
            sample_num = min(frame_number, max_frame_number) if max_frame_number > 0 else frame_number

        indices = np.linspace(0, frame_number - 1, sample_num, dtype=int)
        sampled_paths = [os.path.join(frame_path, f"{all_times[i]}.jpg") for i in indices]
        sampled_times = [all_times[i] for i in indices]

        return sampled_paths, sampled_times

    # ------------------------------------------------------------------
    # Backend B: in-memory frame decoding (SGLang-style, default)
    # ------------------------------------------------------------------

    def _decode_video_bytes_to_frames_memory(
        self,
        mm_obj_byte: bytes,
        decode_way: str,
        max_frame_number: int,
    ) -> Tuple[List[Image.Image], List[int]]:
        """Decode video bytes to in-memory PIL Images (no disk I/O).

        Follows SGLang's convention of keeping media data in memory as PIL
        Images so that ``extracted_images_transform`` can consume them
        directly via its ``isinstance(strseq, Image.Image)`` branch.

        Returns:
            (sampled_pil_frames, sampled_frame_times) where *sampled_pil_frames*
            are RGB PIL Images and *sampled_frame_times* are integer timestamps.
        """
        import io as _io

        if decode_way not in ('1fps', 'key'):
            raise ValueError(f"Invalid decode_way: {decode_way!r}")

        result = read_video(_io.BytesIO(mm_obj_byte), max_frame_number=-1, decode_way=decode_way)
        if result is None:
            raise RuntimeError("read_video returned None")
        frames, frame_times = result
        frame_times = list(frame_times)

        # Uniform sampling with tps alignment
        frame_number = len(frames)
        sample_num = min(frame_number, max_frame_number) if max_frame_number > 0 else frame_number
        sample_num = floor_by_factor(sample_num, self.video_processor.temporal_patch_size)
        if sample_num == 0:
            sample_num = min(frame_number, max_frame_number) if max_frame_number > 0 else frame_number

        indices = np.linspace(0, frame_number - 1, sample_num, dtype=int)
        sampled_pil = [Image.fromarray(frames[i]) for i in indices]
        sampled_times = [frame_times[i] for i in indices]

        return sampled_pil, sampled_times

    # ------------------------------------------------------------------
    # Frame dispatch: memory (default) or disk backend
    # ------------------------------------------------------------------

    def _split_video_to_frames(
        self,
        video_info,
        max_frame_number=-1,
        decode_way="1fps",
        *,
        timer: "Timer",
    ):
        """Decode a video source into a frame-sequence string for the model.

        Two internal backends are available — selected by the
        ``use_disk_frame_cache`` attribute (default False):

        - **memory backend** (default): frames stay as PIL Images in RAM;
          no disk writes.  Follows SGLang convention.
        - **disk backend** (legacy): frames are written to JPEG files on disk.
        """
        try:
            import ujson as _ujson
        except ImportError:
            import json as _ujson

        use_disk = getattr(self, 'use_disk_frame_cache', False)
        if use_disk:
            return self._split_video_to_frames_disk(
                video_info, max_frame_number=max_frame_number,
                decode_way=decode_way, timer=timer,
            )
        return self._split_video_to_frames_memory(
            video_info, max_frame_number=max_frame_number,
            decode_way=decode_way, timer=timer,
        )

    def _split_video_to_frames_memory(
        self,
        video_info,
        max_frame_number=-1,
        decode_way="1fps",
        *,
        timer: "Timer",
    ):
        """Memory-backend: decode video bytes to PIL Images in RAM (no disk writes).

        Frame serialisation protocol:
        - Decoded PIL Images are stored on ``self._current_pil_frames``.
        - Each frame token in *replace_str* embeds ``{"_pil_idx": N}`` so that
          _get_video_frame can retrieve the N-th PIL Image without any file I/O.
        """
        try:
            import ujson as _ujson
        except ImportError:
            import json as _ujson

        with timer.phase("download"):
            mm_obj_byte = self._get_video_bytes(video_info)
        if mm_obj_byte is None:
            return ""

        try:
            with timer.phase("decode"):
                sampled_pil, sampled_times = self._decode_video_bytes_to_frames_memory(
                    mm_obj_byte, decode_way=decode_way,
                    max_frame_number=max_frame_number,
                )
        except Exception as e:
            logger.warning("_split_video_to_frames_memory: decode error: %s", e)
            return ""

        # Store PIL frames for retrieval in _get_video_frame via _pil_idx marker
        self._current_pil_frames = sampled_pil

        replace_str = ""
        for frame_idx, frame_time in enumerate(sampled_times):
            frame_str = self.frame_pattern.format(frame_idx) if '{}' in self.frame_pattern else self.frame_pattern
            frame_str = frame_str.replace('<TIMEIDX>', "at second[" + str(frame_time) + "]")
            frame_str = frame_str.replace('<TIMESTAMP>', time.strftime("%H:%M:%S", time.gmtime(frame_time)))
            # Embed a lightweight index marker; _get_video_frame resolves it via
            # self._current_pil_frames to avoid any disk round-trip.
            frame_str = frame_str.replace(
                '<frame>',
                f'{self.image_start_token}{_ujson.dumps({"_pil_idx": frame_idx})}{self.image_end_token}'
            )
            replace_str += frame_str

        return replace_str

    def _split_video_to_frames_disk(
        self,
        video_info,
        max_frame_number=-1,
        decode_way="1fps",
        *,
        timer: "Timer",
    ):
        """Disk-backend: decode video bytes, write JPEG frames to disk.

        Frame directory naming policy:

        **Local file**
            frame_path = ``<video_stem>`` + ``_frames|_keyframes``
            placed alongside the source file.

        **URL / base64 — persistent cache mode** (``VIDEO_URL_CACHE_DIR`` is set)
            - Video file  : ``<cache_dir>/<md5(url)>.mp4``
            - Frame dir   : ``<cache_dir>/<md5(url)>_frames``
            - Neither file nor directory is ever deleted; subsequent requests
              for the same URL hit the frame-dir cache and skip both download
              and decode.

        **URL / base64 — temp mode** (``VIDEO_URL_CACHE_DIR`` is empty, default)
            - A ``tempfile.mkdtemp()`` workspace is created per request.
            - By default the workspace is **kept** (not deleted) after use.
            - Set ``VIDEO_TEMP_VIDEO_CLEANUP=1`` to delete it at the start of
              the next call (after frames have been consumed).
        """
        import hashlib as _hashlib
        import shutil as _shutil
        import tempfile as _tempfile
        try:
            import ujson as _ujson
        except ImportError:
            import json as _ujson

        if decode_way not in ('1fps', 'key'):
            raise ValueError(f"Invalid decode_way: {decode_way!r}")

        frame_suffix = '_frames' if decode_way == '1fps' else '_keyframes'
        tmp_workspace = None          # only set for the temp-mode path
        mm_obj_byte: Optional[bytes] = None
        downloaded_raw: Optional[bytes] = None

        def _timed_get_video_bytes(source):
            with timer.phase("download"):
                return self._get_video_bytes(source)

        if os.path.isfile(video_info):
            # ── Local file ─────────────────────────────────────────────────
            # No download needed; derive frame_path from the source stem.
            stem = video_info[:video_info.rfind('.')] if '.' in video_info else video_info
            frame_path = stem + frame_suffix
            mm_obj_byte = _timed_get_video_bytes(video_info)

        elif (video_info.startswith("http://") or video_info.startswith("https://")):
            # ── URL video ──────────────────────────────────────────────────
            url = video_info
            md5_name = _hashlib.md5(url.encode('utf-8')).hexdigest()

            cache_dir = self.video_url_cache_dir
            if cache_dir:
                # Persistent cache: store video file + frame dir under cache_dir.
                os.makedirs(cache_dir, exist_ok=True)
                video_path = os.path.join(cache_dir, md5_name + ".mp4")
                frame_path = os.path.join(cache_dir, md5_name + frame_suffix)

                # Download only if the video file is not already cached.
                if not os.path.isfile(video_path):
                    logger.info("Downloading video URL to cache: %s → %s", url, video_path)
                    downloaded_raw = _timed_get_video_bytes(url)
                    if downloaded_raw is None:
                        return ""
                    try:
                        with open(video_path, "wb") as _f:
                            _f.write(downloaded_raw)
                    except Exception as e:
                        logger.warning("Failed to write video cache file %s: %s", video_path, e)
                        return ""

                # If frame dir already exists and is non-empty, skip decode entirely.
                if os.path.isdir(frame_path) and os.listdir(frame_path):
                    logger.info("Frame cache hit for URL (md5=%s): %s", md5_name, frame_path)
                    # Read bytes only for the _decode call's non-cache path; since
                    # _decode_video_bytes_to_frames_disk checks _cache_hit itself,
                    # passing an empty bytes is harmless — it will be skipped.
                    mm_obj_byte = b""
                elif downloaded_raw is not None:
                    # Reuse the just-downloaded bytes.  Reading the cache file
                    # again would both add I/O and overwrite the real download
                    # phase in the per-media Timer.
                    mm_obj_byte = downloaded_raw
                else:
                    with timer.phase("download"):
                        mm_obj_byte = open(video_path, "rb").read()
                # tmp_workspace remains None → no cleanup scheduled.

            else:
                # Temp mode: per-request directory, cleaned up after use.
                tmp_workspace = _tempfile.mkdtemp(prefix='longcat_video_')
                frame_path = os.path.join(tmp_workspace, 'frames' + frame_suffix)
                mm_obj_byte = _timed_get_video_bytes(url)

        else:
            # ── base64 or unrecognised (treat as bytes source) ─────────────
            tmp_workspace = _tempfile.mkdtemp(prefix='longcat_video_')
            frame_path = os.path.join(tmp_workspace, 'frames' + frame_suffix)
            mm_obj_byte = _timed_get_video_bytes(video_info)

        # Clean up temp dirs from *previous* requests (frames already consumed).
        # Only runs when VIDEO_TEMP_VIDEO_CLEANUP=1; default is to keep temp dirs.
        if self.video_temp_cleanup:
            pending = getattr(self, '_pending_cleanup_dirs', [])
            for d in pending:
                if os.path.exists(d):
                    _shutil.rmtree(d, ignore_errors=True)
        self._pending_cleanup_dirs = []

        if mm_obj_byte is None:
            return ""

        try:
            with timer.phase("decode"):
                sampled_paths, sampled_times = self._decode_video_bytes_to_frames_disk(
                    mm_obj_byte, frame_path=frame_path,
                    decode_way=decode_way, max_frame_number=max_frame_number,
                )
        except Exception as e:
            logger.warning("_split_video_to_frames_disk: decode error: %s", e)
            if tmp_workspace and os.path.exists(tmp_workspace):
                _shutil.rmtree(tmp_workspace, ignore_errors=True)
            return ""

        # Schedule temp workspace cleanup for after _get_video_frame has read the frames.
        # Persistent-cache paths never schedule cleanup (tmp_workspace is None).
        # Temp-mode cleanup only runs when VIDEO_TEMP_VIDEO_CLEANUP=1.
        if tmp_workspace and self.video_temp_cleanup:
            self._pending_cleanup_dirs = [tmp_workspace]

        replace_str = ""
        for frame_idx, (fpath, frame_time) in enumerate(zip(sampled_paths, sampled_times)):
            frame_dict = {"local": fpath}
            frame_str = self.frame_pattern.format(frame_idx) if '{}' in self.frame_pattern else self.frame_pattern
            frame_str = frame_str.replace('<TIMEIDX>', "at second[" + str(frame_time) + "]")
            frame_str = frame_str.replace('<TIMESTAMP>', time.strftime("%H:%M:%S", time.gmtime(frame_time)))
            frame_str = frame_str.replace('<frame>', f'{self.image_start_token}{_ujson.dumps(frame_dict)}{self.image_end_token}')
            replace_str += frame_str

        return replace_str

    # ------------------------------------------------------------------
    # Frame sampling / time extraction
    # ------------------------------------------------------------------

    def sample_frame(self, frames_str, max_frame=32):
        """Sample frames from pre-extracted frame string, keeping tps alignment.
        """
        def uniform_sample(lst, num_samples):
            frame_number = len(lst)
            max_frame_number = num_samples
            sample_num = min(frame_number, max_frame_number)
            sample_num = floor_by_factor(sample_num, self.video_processor.temporal_patch_size)
            if sample_num == 0:
                sample_num = min(frame_number, max_frame_number)
            interval = len(lst) / sample_num
            return [lst[int(i * interval)] for i in range(sample_num)]

        p = rf'({re.escape(self.image_start_token)}.*?{re.escape(self.image_end_token)})'
        frames_str_split = re.split(p, frames_str)

        frame_idxs = [idx for idx in range(len(frames_str_split))
                      if self.image_start_token in frames_str_split[idx]]

        sampled_idxs = uniform_sample(frame_idxs, max_frame)
        sample_frame_idxs = set(sampled_idxs)

        paths = []
        timestamps = []
        timestamp_pattern = re.compile(r'at second\[[^\]]+\]')

        ans_components = []
        for idx, item in enumerate(frames_str_split):
            if idx in sample_frame_idxs:
                ans_components.append(item)
                if idx in sample_frame_idxs and (idx + 1) < len(frames_str_split):
                    next_item = frames_str_split[idx + 1]
                    ts_match = timestamp_pattern.search(next_item)
                    if ts_match:
                        timestamps.append(ts_match.group())
                    else:
                        timestamps.append("")
            elif self.image_start_token not in item:
                ans_components.append(item)

        ans = ''.join(ans_components)
        start_tag = re.escape(self.image_start_token)
        end_tag = re.escape(self.image_end_token)
        pattern = rf'{start_tag}(.*?){end_tag}'
        paths = re.findall(pattern, ans, re.DOTALL)

        if len(paths) != len(timestamps):
            diff = len(paths) - len(timestamps)
            timestamps.extend([""] * diff)

        return paths, timestamps

    def _get_video_time(self, video_frame_infos):
        """Extract 'at second[X]' timestamps from frame-string.
        """
        try:
            if isinstance(video_frame_infos, str):
                pattern = r'at second\[.*?\]'
                matches = re.findall(pattern, video_frame_infos)
                if len(matches) == 0:
                    return []
                video_time = []
                for match in matches:
                    video_time.append(match)
                return video_time
        except Exception as e:
            logger.warning("_get_video_time: error: %s, info: %s", str(e)[:256], str(video_frame_infos)[:256])
            raise
        return []

    def _get_video_frame(
        self,
        video_frame_infos,
        max_pixels=None,
        device_preprocess=False,
        device=None,
        *,
        timer: "Timer",
    ):
        """Load frame data and run extracted_images_transform.

        Supports two input forms:
        - str: JSON markers embedded in the frame-sequence string.  Each
          ``{"_pil_idx": N}`` marker (memory backend) resolves to the N-th
          PIL Image stored on ``self._current_pil_frames``; ``{"local": path}``
          / ``{"path": path}`` markers (disk backend) resolve to file paths.
        - list: a pre-built list of file paths or PIL Images (pre-extracted).

        Returns (video_feat, patch_nums, grid_thw) or None.
        """
        try:
            import ujson as _ujson
        except ImportError:
            import json as _ujson
        try:
            if isinstance(video_frame_infos, str):
                pattern = r'\{.*?\}'
                matches = re.findall(pattern, video_frame_infos)
                if len(matches) == 0:
                    return None
                video_sampled_frames = []
                pil_frames = getattr(self, '_current_pil_frames', None)
                for match in matches:
                    video_frame_info = _ujson.loads(match)
                    if '_pil_idx' in video_frame_info:
                        # Memory-backend path: retrieve PIL Image by index
                        idx = video_frame_info['_pil_idx']
                        if pil_frames is None or idx >= len(pil_frames):
                            raise ValueError(
                                f"_pil_idx={idx} out of range "
                                f"(self._current_pil_frames has "
                                f"{len(pil_frames) if pil_frames else 0} frames)"
                            )
                        video_sampled_frames.append(pil_frames[idx])
                    elif 'local' in video_frame_info:
                        video_sampled_frames.append(video_frame_info['local'])
                    elif 'path' in video_frame_info and os.path.exists(video_frame_info['path']):
                        video_sampled_frames.append(video_frame_info['path'])
                    else:
                        raise ValueError(f"cannot resolve frame info: {video_frame_info}")
            elif isinstance(video_frame_infos, list):
                video_sampled_frames = video_frame_infos
            else:
                raise ValueError(f"unsupported video_frame_infos type: {type(video_frame_infos)}")

            with timer.phase("transform"):
                if (
                    device_preprocess
                    and device is not None
                    and device.type == "npu"
                ):
                    from sglang.srt.multimodal.processors.longcat_flash_vl_npu_transform import (
                        npu_video_transform,
                    )

                    video_feat, org_size, video_list = npu_video_transform(
                        video_sampled_frames, self.video_processor, device,
                        max_pixels=max_pixels,
                    )
                elif (
                    device_preprocess
                    and device is not None
                    and device.type == "cuda"
                ):
                    video_feat, org_size, video_list = (
                        self.video_processor.gpu_video_transform(
                            video_sampled_frames,
                            device=device,
                            max_pixels=max_pixels,
                        )
                    )
                else:
                    video_feat, org_size, video_list = self.video_processor.extracted_images_transform(
                        video_sampled_frames, training=False, max_pixels=max_pixels
                    )
            merge_length = self.video_processor.merge_size ** 2
            patch_nums = int(np.array(video_list).prod() // merge_length)

            if org_size[0] * org_size[1] > 16 ** 2:
                return video_feat, patch_nums, video_list
            else:
                logger.warning(
                    "_get_video_frame: video too small: org_size=%s", org_size
                )
                return None
        except Exception as e:
            logger.warning(
                "_get_video_frame: error: %s", str(e)[:256]
            )
            raise

    def _get_video_frame_str(self, video_info, *, timer: "Timer"):
        """Parse video_info and return (frames_str_or_list, frame_timestamps)."""
        is_maas_input = is_maas_video_frame_sequence(video_info)
        try:
            if is_maas_input:
                refs = parse_maas_video_frame_sequence(
                    video_info,
                    max_frames=self.video_max_frames,
                )
                with timer.phase("download"):
                    frames = [
                        load_image(ref.url, gpu_image_decode=False)[0]
                        for ref in refs
                    ]
                return frames, [ref.timestamp_text for ref in refs]
            if self.videoframe_start_tag in video_info:
                # Pre-extracted frame format: replace frame tags with image tags
                frames_str = video_info
                frames_str = frames_str.replace(self.videoframe_start_tag, self.image_start_token).replace(
                    self.videoframe_end_tag, self.image_end_token)
                return self.sample_frame(frames_str, max_frame=self.video_max_frames)
            # Dispatch to _split_video_to_frames; it handles URL / local path / base64
            frames_str = self._split_video_to_frames(
                video_info,
                max_frame_number=self.video_max_frames,
                decode_way=self.video_decode_way,
                timer=timer,
            )
            return frames_str, None
        except Exception as e:
            if is_maas_input:
                logger.warning("MaaS pre-extracted video parsing failed: %s", str(e))
            else:
                logger.warning("_get_video_frame_str: error: %s, info: %s", str(e), str(video_info)[:256])
            raise

    def _replace_video_frame(
        self,
        video_frame_text,
        frame_timestamps,
        max_pixels=None,
        device_preprocess=False,
        device=None,
        *,
        timer: "Timer",
    ):
        """Process frame list/string and build video token string.
        Returns (video_feat, patch_nums, grid_thw, token_str) or None.
        """
        if isinstance(video_frame_text, str):
            video_frame_info = re.sub(
                re.compile(re.escape(self.image_start_token) + "|" + re.escape(self.image_end_token)), '',
                video_frame_text)
            if "second" in video_frame_info:
                time_info = self._get_video_time(video_frame_info)
            else:
                time_info = []
        elif isinstance(video_frame_text, list):
            video_frame_info = video_frame_text
            time_info = frame_timestamps
        else:
            raise ValueError("unsupported video_frame_text type")

        result = self._get_video_frame(
            video_frame_info,
            max_pixels=max_pixels,
            device_preprocess=device_preprocess,
            device=device,
            timer=timer,
        )
        if result is None:
            return None

        video_feat, patch_nums, video_grid = result
        grid_t, grid_h, grid_w = video_grid
        if time_info:
            self.last_video_timestamps = [
                time_info[idx * self.video_processor.temporal_patch_size]
                for idx in range(grid_t)
            ]
        else:
            self.last_video_timestamps = []

        # Build token string
        if self.video_processor.split_video:
            if time_info is not None and len(time_info) > 0:
                video_frame_str = "".join([
                    self.image_start_token
                    + self.video_place_token * (patch_nums // grid_t)
                    + self.image_end_token
                    + time_info[idx * self.video_processor.temporal_patch_size]
                    for idx in range(grid_t)
                ])
            else:
                video_frame_str = "".join([
                    self.image_start_token + self.video_place_token * (patch_nums // grid_t) + self.image_end_token
                    for _ in range(grid_t)
                ])
        else:
            video_frame_str = self.image_start_token + self.video_place_token * patch_nums + self.image_end_token

        # Wrap with video_start/end tags
        token_str = self.video_start_token + video_frame_str + self.video_end_token
        return video_feat, patch_nums, video_grid, token_str

    def process_video(
        self,
        video_info: str,
        max_pixels: Optional[int] = None,
        device_preprocess: bool = False,
        device: Optional[torch.device] = None,
        *,
        timer: "Timer",
    ) -> Optional[
        Tuple[Union[np.ndarray, torch.Tensor], int, Tuple[int, int, int], str]
    ]:
        """Decode video and return (feat, patch_nums, grid_thw, token_str).
        """
        try:
            video_frames, frame_timestamps = self._get_video_frame_str(
                video_info, timer=timer,
            )
            if isinstance(video_frames, str):
                # Pass the raw frame-sequence string; _replace_video_frame will
                # wrap the final token_str with video_start/end itself.
                frame_str = video_frames
            elif isinstance(video_frames, list):
                frame_str = video_frames
            else:
                logger.warning("_get_video_frame_str returned empty for: %s", video_info[:128])
                return None
            result = self._replace_video_frame(
                frame_str, frame_timestamps, max_pixels=max_pixels,
                timer=timer,
                device_preprocess=device_preprocess,
                device=device,
            )
            return result
        except Exception as e:
            logger.warning("Process video error: %s, info: %s", str(e)[:256], video_info[:128])
            return None


# ============================================================
# Longcat Flash VL SGLang Processor
# ============================================================


class LongcatFlashVLImageProcessor(BaseMultimodalProcessor):
    """
    Multimodal processor for Longcat Flash VL.

    Supported modalities:
    - Image: <img_start> image_url <img_end> → <img_start> + <img_pad> × N + <img_end>
    - Video: <video_start> video_json <video_end> → <video_start> + <video_place> × N + <video_end>

    """

    models = ["LongcatFlashVLForConditionalGeneration"]
    gpu_image_decode = True

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        vis_cfg = hf_config.vision_config
        self.patch_size = getattr(vis_cfg, "patch_size", 14)
        self.temporal_patch_size = getattr(vis_cfg, "temporal_patch_size", 2)
        self.merge_size = getattr(vis_cfg, "spatial_merge_size", getattr(vis_cfg, "merge_size", 2))

        self.min_pixels = getattr(vis_cfg, "min_pixels", MIN_PIXELS)
        self.max_pixels = getattr(vis_cfg, "max_pixels", MAX_PIXELS)
        self.image_mean = getattr(vis_cfg, "image_mean", IMAGE_MEAN)
        self.image_std = getattr(vis_cfg, "image_std", IMAGE_STD)

        # Image token IDs
        self.image_start_token_id = getattr(vis_cfg, "image_start_token_id", IMAGE_START_TOKEN_ID)
        self.image_end_token_id = getattr(vis_cfg, "image_end_token_id", IMAGE_END_TOKEN_ID)
        self.image_pad_token_id = getattr(vis_cfg, "image_pad_token_id", IMAGE_PAD_TOKEN_ID)

        self.image_start_token = self._get_token_str(self.image_start_token_id, IMAGE_START_TOKEN)
        self.image_end_token = self._get_token_str(self.image_end_token_id, IMAGE_END_TOKEN)
        self.image_pad_token = self._get_token_str(self.image_pad_token_id, IMAGE_PAD_TOKEN)

        # Video token IDs — read from hf_config.video_config if present
        vid_cfg = getattr(hf_config, "video_config", None)
        self.video_start_token_id = getattr(vid_cfg, "video_start_token_id", VIDEO_START_TOKEN_ID) if vid_cfg else VIDEO_START_TOKEN_ID
        self.video_end_token_id = getattr(vid_cfg, "video_end_token_id", VIDEO_END_TOKEN_ID) if vid_cfg else VIDEO_END_TOKEN_ID
        self.video_place_token_id = getattr(vid_cfg, "video_place_token_id", VIDEO_PLACE_TOKEN_ID) if vid_cfg else VIDEO_PLACE_TOKEN_ID

        self.video_start_token = self._get_token_str(self.video_start_token_id, VIDEO_START_TOKEN)
        self.video_end_token = self._get_token_str(self.video_end_token_id, VIDEO_END_TOKEN)
        self.video_place_token = self._get_token_str(self.video_place_token_id, VIDEO_PLACE_TOKEN)

        # Video decode / sampling config
        # Note: config.json video_config uses "max_pixels"/"min_pixels" (no "video_" prefix)
        self.video_max_frames = getattr(vid_cfg, "max_frame_num", VIDEO_MAX_FRAMES) if vid_cfg else VIDEO_MAX_FRAMES
        self.video_decode_way = getattr(vid_cfg, "decode_way", VIDEO_DECODE_WAY) if vid_cfg else VIDEO_DECODE_WAY
        # Use "max_pixels"/"min_pixels" (matching config.json field names), fall back to constants
        self.video_max_pixels = getattr(vid_cfg, "max_pixels", VIDEO_MAX_PIXELS) if vid_cfg else VIDEO_MAX_PIXELS
        self.video_min_pixels = getattr(vid_cfg, "min_pixels", VIDEO_MIN_PIXELS) if vid_cfg else VIDEO_MIN_PIXELS
        dynamic_frame_pixel = getattr(vid_cfg, "dynamic_frame_pixel", False) if vid_cfg else False

        # Environment variable overrides — allow evaluation platforms to configure video params
        # without modifying config.json or checkpoint files.
        # Priority: env var > config.json > code default constant.
        #   VIDEO_MAX_FRAMES : max number of sampled frames (int), e.g. export VIDEO_MAX_FRAMES=128
        #   VIDEO_MAX_PIXELS : per-frame pixel upper bound (int), e.g. export VIDEO_MAX_PIXELS=150528
        #   VIDEO_MIN_PIXELS : per-frame pixel lower bound (int), e.g. export VIDEO_MIN_PIXELS=78400
        if "VIDEO_MAX_FRAMES" in os.environ:
            self.video_max_frames = int(os.environ["VIDEO_MAX_FRAMES"])
            logger.info(f"[video] VIDEO_MAX_FRAMES override from env: {self.video_max_frames}")
        if "VIDEO_MAX_PIXELS" in os.environ:
            self.video_max_pixels = int(os.environ["VIDEO_MAX_PIXELS"])
            logger.info(f"[video] VIDEO_MAX_PIXELS override from env: {self.video_max_pixels}")
        if "VIDEO_MIN_PIXELS" in os.environ:
            self.video_min_pixels = int(os.environ["VIDEO_MIN_PIXELS"])
            logger.info(f"[video] VIDEO_MIN_PIXELS override from env: {self.video_min_pixels}")

        # Video frame handling attributes
        # videoframe_start_tag / videoframe_end_tag: used when video is pre-extracted to frames format
        self.videoframe_start_tag = '<videoframe_start_omni>'
        self.videoframe_end_tag = '<videoframe_end_omni>'
        # frame_pattern: controls how each frame is serialized
        # config.json sets frame_pattern = "<frame><TIMEIDX>"; default fallback is the simpler "<frame>"
        self.frame_pattern = getattr(vid_cfg, 'frame_pattern', '<frame>') if vid_cfg else '<frame>'

        pad_re = re.escape(self.image_pad_token)
        start_re = re.escape(self.image_start_token)
        end_re = re.escape(self.image_end_token)

        # TODO(video): Register video_token / video_token_id in MultimodalSpecialTokens
        # once the base class fast-path (_fast_load_mm_data) is used for video routing.
        # Currently, process_mm_data_async performs its own text splitting via
        # _split_multimodal_chunk, so mm_tokens only needs image_token for the
        # backup path (process_mm_data_async_backup) which uses load_mm_data.
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.image_pad_token,
            image_token_id=self.image_pad_token_id,
            image_token_regex=re.compile(rf"{start_re}(?:{pad_re})+{end_re}"),
        ).build(_processor)

        # OmniImageProcessor for precision-identical image preprocessing
        self.visual_processor = OmniImageProcessor(
            patch_size=self.patch_size,
            temporal_patch_size=self.temporal_patch_size,
            spatial_merge_size=self.merge_size,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            image_mean=self.image_mean,
            image_std=self.image_std,
        )

        # OmniVideoProcessor for video preprocessing
        # split_video: from config.json video_config.split_video (True → per-frame token groups)
        self.video_processor = OmniVideoProcessor(
            patch_size=self.patch_size,
            temporal_patch_size=self.temporal_patch_size,
            spatial_merge_size=self.merge_size,
            min_pixels=self.video_min_pixels,
            max_pixels=self.video_max_pixels,
            image_mean=self.image_mean,
            image_std=self.image_std,
            dynamic_frame_pixel=dynamic_frame_pixel,
            split_video=getattr(vid_cfg, 'split_video', True) if vid_cfg else True,
        )

        # VideoProcessingPipeline — high-level pipeline wrapping OmniVideoProcessor.
        # LongcatFlashVLImageProcessor delegates all video logic to this pipeline so
        # that this class stays focused on SGLang integration (tokenisation, mm_items).
        self._video_handler = VideoProcessingPipeline(
            frame_processor=self.video_processor,
            image_start_token=self.image_start_token,
            image_end_token=self.image_end_token,
            video_start_token=self.video_start_token,
            video_end_token=self.video_end_token,
            video_place_token=self.video_place_token,
            frame_pattern=self.frame_pattern,
            videoframe_start_tag=self.videoframe_start_tag,
            videoframe_end_tag=self.videoframe_end_tag,
            video_max_frames=self.video_max_frames,
            video_decode_way=self.video_decode_way,
        )

        # Regex for matching <img_start>...<img_end> in input text
        self._image_match_regex = re.compile(
            re.escape(self.image_start_token) + '.*?' + re.escape(self.image_end_token), re.S
        )
        self._image_drop_regex = re.compile(
            re.escape(self.image_start_token) + "|" + re.escape(self.image_end_token), re.S
        )

        # Regex for matching <video_start>...<video_end> in input text
        self._video_match_regex = re.compile(
            re.escape(self.video_start_token) + '.*?' + re.escape(self.video_end_token), re.S
        )
        self._video_drop_regex = re.compile(
            re.escape(self.video_start_token) + "|" + re.escape(self.video_end_token), re.S
        )

        # Block markers for _split_multimodal_chunk, in alternation priority order.
        self._mm_block_markers = (
            ("image", self.image_start_token, self.image_end_token),
            ("video", self.video_start_token, self.video_end_token),
        )

    def _get_token_str(self, token_id: int, fallback: str) -> str:
        """Convert token_id to string via tokenizer, with fallback."""
        try:
            token_str = self._tokenizer.convert_ids_to_tokens(token_id)
            if token_str is not None:
                return token_str
        except Exception:
            pass
        return fallback

    # ============================================================
    # Text/image/video splitting (split_multimodal_chunk)
    # ============================================================

    def _encode_input_ids(self, text: str) -> List[int]:
        """Same ids as ``self._tokenizer(text, add_special_tokens=True)``.

        Calls the Rust backend's ``encode_batch_fast`` (no offset tracking)
        when the HF wrapper would not change the result: no truncation or
        padding configured on the backend and the same special-token
        splitting.  Freeing the offsets-tracking ``Encoding`` built through
        the wrapper holds the GIL for ~38 ms at 680K tokens (~13 ms here).
        """
        backend = getattr(self._tokenizer, "_tokenizer", None)
        encode_fast = getattr(backend, "encode_batch_fast", None)
        if (
            encode_fast is not None
            and backend.truncation is None
            and backend.padding is None
            and bool(getattr(backend, "encode_special_tokens", False))
            == bool(getattr(self._tokenizer, "split_special_tokens", False))
        ):
            return encode_fast([text], add_special_tokens=True)[0].ids
        return self._tokenizer(
            text, add_special_tokens=True, return_attention_mask=False
        ).input_ids

    def _iter_mm_blocks(self, text: str):
        """Yield ``(start, end, label)`` of image/video blocks, leftmost first.

        Same matches as ``finditer`` over ``(?P<image>S.*?E)|(?P<video>S.*?E)``
        with ``re.S``, but built on ``str.find``: the regex engine scans the
        whole prompt in one GIL-holding call (~34 ms for 1.8M chars), which
        stalls the Scheduler when EPD admission tokenizes off-thread.
        """
        exhausted = set()
        pos = 0
        while True:
            best = None
            for label, start_token, end_token in self._mm_block_markers:
                if label in exhausted:
                    continue
                start = text.find(start_token, pos)
                end = -1 if start < 0 else text.find(end_token, start + len(start_token))
                if end < 0:
                    # No later start of this kind can find a closing token.
                    exhausted.add(label)
                    continue
                # Ties keep the earlier marker, like regex alternation order.
                if best is None or start < best[0]:
                    best = (start, end + len(end_token), label)
            if best is None:
                return
            yield best
            pos = best[1]

    def _split_multimodal_chunk(self, text: str) -> Tuple[List[str], List[str]]:
        """Split text into text/image/video segments.

        Single-pass split over image blocks (<img_start>…<img_end>) and video
        blocks (<video_start>…<video_end>), see ``_iter_mm_blocks``.

        Returns:
            text_list:    list of segment strings (image/video content stripped of tags).
            mm_label_list: corresponding labels — 'image', 'video', or 'text'.
        """
        text_list, mm_label_list = [], []
        last_end = 0
        for start, end, label in self._iter_mm_blocks(text):
            # Text before this match
            if start > last_end:
                text_list.append(text[last_end:start])
                mm_label_list.append('text')
            drop_regex = self._image_drop_regex if label == 'image' else self._video_drop_regex
            text_list.append(re.sub(drop_regex, '', text[start:end]))
            mm_label_list.append(label)
            last_end = end
        # Trailing text after the last match
        if last_end < len(text):
            text_list.append(text[last_end:])
            mm_label_list.append('text')
        return text_list, mm_label_list

    # ============================================================
    # Image processing
    # ============================================================

    def _get_image(self, image_info: str, min_pixels: Optional[int] = None,
                   max_pixels: Optional[int] = None) -> Optional[Tuple[np.ndarray, int, Tuple[int, int, int]]]:
        """Load and preprocess a single image from path/URL.

        Returns:
            (flatten_patches, patch_nums, grid_thw) or None on failure
        """
        try:
            image_feat, org_size, grid_thw = self.visual_processor.image_transform(
                image_info, min_pixels=min_pixels, max_pixels=max_pixels
            )
            merge_length = self.visual_processor.merge_size ** 2
            patch_nums = int(np.array(grid_thw).prod() // merge_length)
            if org_size[0] * org_size[1] > 16 ** 2:  # 极端小的图过滤
                return image_feat, patch_nums, grid_thw
            else:
                logger.warning(f"Image too small: {org_size}, info: {image_info}")
                return None
        except Exception as e:
            logger.warning(f"Get image error: {str(e)[:256]}, info: {image_info}")
            return None

    def _replace_image(self, image_info: str, min_pixels: Optional[int] = None,
                       max_pixels: Optional[int] = None) -> Optional[Tuple[np.ndarray, int, Tuple[int, int, int], str]]:
        """Process an image segment and return features + token string.

        Returns:
            (image_feat, patch_nums, grid_thw, token_str) or None on failure
        """
        result = self._get_image(image_info, min_pixels=min_pixels, max_pixels=max_pixels)
        if result is None:
            return None
        image_feat, patch_nums, grid_thw = result
        token_str = self.image_start_token + self.image_pad_token * patch_nums + self.image_end_token
        return image_feat, patch_nums, grid_thw, token_str

    # ============================================================
    # Video processing — delegated to VideoProcessingPipeline
    # ============================================================

    def _process_video(
        self,
        video_info: str,
        max_pixels: Optional[int] = None,
    ) -> Optional[Tuple[np.ndarray, int, Tuple[int, int, int], str]]:
        """Decode video and return (feat, patch_nums, grid_thw, token_str).
        Delegates to self._video_handler.process_video().
        """
        from sglang.srt.disaggregation.encode.utils import Timer

        timer = Timer("multimodal process video")
        return self._video_handler.process_video(
            video_info, max_pixels=max_pixels, timer=timer,
        )

    # ============================================================
    # Main async processing entry point
    # ============================================================

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes, Dict]],
        audio_data,
        input_text,
        request_obj,
        *args,
        epd_item_metas: Optional[List["MMItemMeta"]] = None,
        **kwargs,
    ) -> MultimodalProcessorOutput:
        """Process multimodal data (images + videos) embedded in text.

        Supported embed formats in input_text:
          Image: <img_start> image_url_or_path <img_end>
          Video: <video_start> {"local":"/path/to/video.mp4"} <video_end>
                 <video_start> {"url":"https://..."} <video_end>
                 <video_start> {"base64":"..."} <video_end>
                 <video_start> ["/frame_0.jpg", "/frame_1.jpg", ...] <video_end>  (pre-extracted)

        Pipeline:
        1. _split_multimodal_chunk: split text into text/image/video segments
        2. For image: OmniImageProcessor.image_transform → <img_start><img_pad>×N<img_end>
           For video: decode frames → OmniVideoProcessor.video_transform → <video_start><video_place>×N<video_end>
        3. Reassemble text, tokenize
        4. Build MultimodalDataItems (Modality.IMAGE for image, Modality.VIDEO for video)

        EPD mode (epd_item_metas is not None):
          When epd_item_metas is provided (a list of MMItemMeta from the remote Encoder),
          image download and image_transform are skipped.  Instead, num_tokens/hash/
          pad_value/grid_thw from each MMItemMeta are used to construct token_str and
          MultimodalDataItem directly.  This allows the LM-side processor to produce
          correct input_ids (with proper pad_token counts) and pad_values for RadixCache
          without ever touching pixel_values.

          The caller (TokenizerMultimodalMixin._apply_epd_mm_processing) is responsible
          for calling encode_async first and passing the returned item_metas here.
        """
        from sglang.srt.multimodal.mm_utils import get_new_expanded_mm_items, hash_feature

        # Per-request pixel budget override
        req_min_pixels = getattr(request_obj, "min_pixels", None)
        req_max_pixels = getattr(request_obj, "max_pixels", None)

        # Step 1: Split text into image/video/text segments
        text_list, mm_label_list = self._split_multimodal_chunk(input_text)

        # Step 2: Process each segment
        expanded_text_parts = []
        # each entry: (feat, patch_nums, grid_thw, modality, token_id_for_offset, url)
        mm_features: List[
            Tuple[
                Optional[np.ndarray],
                int,
                Tuple[int, int, int],
                Modality,
                int,
                Union[str, bytes, Dict],
            ]
        ] = []

        # EPD mode: index into epd_item_metas for each media segment.
        # metas are flat — image and video segments consume metas in order
        # of appearance in the text (image segments first as in the original
        # code, but now video segments also draw from the same flat list).
        epd_meta_idx = 0
        epd_metas: List[MMItemMeta] = []
        if epd_item_metas is not None:
            epd_metas = list(epd_item_metas)

        # GenerateReqInput allows a single string as well as a list. Empty image
        # blocks are placeholders whose source is supplied through image_data.
        if isinstance(image_data, str):
            image_data = [image_data]
        img_data_idx = 0

        for text_seg, label in zip(text_list, mm_label_list):
            if label == 'image':
                if not text_seg and image_data and img_data_idx < len(image_data):
                    image_info = image_data[img_data_idx]
                    img_data_idx += 1
                else:
                    image_info = text_seg

                if epd_item_metas is not None:
                    # EPD mode: use remote meta, skip image download
                    if epd_meta_idx < len(epd_metas):
                        meta = epd_metas[epd_meta_idx]
                        epd_meta_idx += 1
                        patch_nums = meta.num_tokens
                        grid_thw = tuple(meta.grid_thw) if meta.grid_thw else (1, 1, 1)
                        token_str = self.image_start_token + self.image_pad_token * patch_nums + self.image_end_token
                        # feat is None in EPD mode (no pixel_values on LM side)
                        mm_features.append((None, patch_nums, grid_thw, Modality.IMAGE, self.image_pad_token_id, image_info))
                        expanded_text_parts.append(token_str)
                    else:
                        logger.warning(
                            "EPD mode: fewer item_metas than image segments; "
                            "leaving segment as raw text: %s", text_seg[:80]
                        )
                        expanded_text_parts.append(text_seg)
                else:
                    result = self._replace_image(
                        image_info, min_pixels=req_min_pixels, max_pixels=req_max_pixels
                    )
                    if result is not None:
                        feat, patch_nums, grid_thw, token_str = result
                        expanded_text_parts.append(token_str)
                        mm_features.append((feat, patch_nums, grid_thw, Modality.IMAGE, self.image_pad_token_id, image_info))
                    else:
                        expanded_text_parts.append(text_seg)

            elif label == 'video':
                if epd_item_metas is not None:
                    # EPD mode: use remote meta, skip video download/decode.
                    # Build token_str from meta.num_tokens + grid_thw, mirroring
                    # VideoProcessingPipeline._replace_video_frame's split_video
                    # logic so the LM-side input_ids carry the right pad counts.
                    if epd_meta_idx < len(epd_metas):
                        meta = epd_metas[epd_meta_idx]
                        epd_meta_idx += 1
                        patch_nums = meta.num_tokens
                        grid_thw = tuple(meta.grid_thw) if meta.grid_thw else (1, 1, 1)
                        grid_t = grid_thw[0]
                        if self.video_processor.split_video and grid_t > 0:
                            per_frame = patch_nums // grid_t if grid_t else patch_nums
                            timestamps = list(meta.video_timestamps or [])
                            video_frame_str = "".join(
                                self.image_start_token
                                + self.video_place_token * per_frame
                                + self.image_end_token
                                + (timestamps[idx] if idx < len(timestamps) else "")
                                for idx in range(grid_t)
                            )
                        else:
                            video_frame_str = (
                                self.image_start_token
                                + self.video_place_token * patch_nums
                                + self.image_end_token
                            )
                        token_str = self.video_start_token + video_frame_str + self.video_end_token
                        mm_features.append((None, patch_nums, grid_thw, Modality.VIDEO, self.video_place_token_id, text_seg))
                        expanded_text_parts.append(token_str)
                    else:
                        logger.warning(
                            "EPD mode: fewer item_metas than video segments; "
                            "leaving segment as raw text: %s", text_seg[:80]
                        )
                        expanded_text_parts.append(text_seg)
                else:
                    result = self._process_video(
                        text_seg, max_pixels=req_max_pixels
                    )
                    if result is not None:
                        feat, patch_nums, grid_thw, token_str = result
                        expanded_text_parts.append(token_str)
                        mm_features.append((feat, patch_nums, grid_thw, Modality.VIDEO, self.video_place_token_id, text_seg))
                    else:
                        expanded_text_parts.append(text_seg)

            else:
                expanded_text_parts.append(text_seg)

        # Step 3: Reassemble & tokenize.  Keep ids as a list: tensor round
        # trips of long prompts hold the GIL, which stalls the Scheduler when
        # EPD admission tokenizes on a background thread.
        expanded_text = "".join(expanded_text_parts)
        input_ids = self._encode_input_ids(expanded_text)

        if not mm_features:
            return MultimodalProcessorOutput(
                input_ids=input_ids,
                mm_items=[],
                im_token_id=self.image_pad_token_id,
            )

        # Step 4: Build mm_items — group by token_id for offset lookup
        # Collect per-token_id offsets once (avoid re-scanning input_ids)
        token_ids_needed = {tid for _, _, _, _, tid, _ in mm_features}
        input_ids_view = _token_ids_view(input_ids)
        offsets_by_token_id: Dict[int, List] = {
            tid: self.get_mm_items_offset(input_ids_view, tid)
            for tid in token_ids_needed
        }
        # Counters for sequential offset assignment per token_id
        offset_idx: Dict[int, int] = {tid: 0 for tid in token_ids_needed}

        mm_items = []
        epd_meta_idx = 0  # re-index for item construction in EPD mode
        for feat, patch_nums, grid_thw, modality, pad_token_id, url in mm_features:
            if feat is None:
                # EPD mode: no pixel_values, use remote meta for hash/pad_value/grid_thw
                if epd_meta_idx < len(epd_metas):
                    meta = epd_metas[epd_meta_idx]
                    epd_meta_idx += 1
                else:
                    meta = None

                item = MultimodalDataItem(modality=modality)
                grid_thw_tensor = torch.tensor([grid_thw], dtype=torch.int64)
                item.model_specific_data["image_grid_thw"] = grid_thw_tensor
                item.model_specific_data["url"] = url
                if meta is not None:
                    item.hash = meta.hash
                    item.pad_value = meta.pad_value
            else:
                pixel_values = torch.from_numpy(feat.astype(np.float32))
                if not is_npu():
                    pixel_values = pixel_values.cuda()
                grid_thw_tensor = torch.tensor([grid_thw], dtype=torch.int64, device=pixel_values.device)

                item = MultimodalDataItem(modality=modality, feature=pixel_values)
                # Use "image_grid_thw" key for both modalities (model forward pass expects this)
                item.model_specific_data["image_grid_thw"] = grid_thw_tensor
                # Save original URL for EPD encoder-side re-download (language_only mode)
                item.model_specific_data["url"] = url

            ofs_list = offsets_by_token_id[pad_token_id]
            idx = offset_idx[pad_token_id]

            if modality == Modality.VIDEO and self.video_processor.split_video:
                # split_video=True: _process_video emits grid_t separate token groups,
                # each formatted as <img_start><video_place>×(N/T)<img_end>.
                # get_mm_items_offset() returns one (start, end) run per group, so there
                # are grid_t discontinuous runs for this single video item.  Collect all
                # grid_t runs so that mm_utils' full_path (len(offsets) > 1) correctly
                # iterates over every group when scattering embeddings.
                grid_t = grid_thw[0]
                needed = grid_t
                if idx + needed <= len(ofs_list):
                    item.offsets = ofs_list[idx : idx + needed]
                    offset_idx[pad_token_id] += needed
                    logger.debug(
                        "VIDEO offset assignment: split_video=True, grid_t=%d, "
                        "runs=[%d..%d], patch_nums=%d",
                        grid_t, idx, idx + needed - 1, patch_nums,
                    )
                else:
                    logger.warning(
                        "VIDEO item at run=%d has insufficient token offsets: "
                        "need %d runs for grid_t=%d, got %d total (token_id=%d)",
                        idx, needed, grid_t, len(ofs_list), pad_token_id,
                    )
                    item.offsets = ofs_list[idx:]
                    offset_idx[pad_token_id] = len(ofs_list)
            else:
                # Image or video with split_video=False: single contiguous token run per item.
                if idx < len(ofs_list):
                    item.offsets = [ofs_list[idx]]
                    offset_idx[pad_token_id] += 1
                    logger.debug(
                        "%s offset assignment: run=%d, offsets=%s, patch_nums=%d",
                        modality.name, idx, item.offsets, patch_nums,
                    )
                else:
                    logger.warning(
                        "%s item at run=%d has no matching token offset "
                        "(total_runs=%d, token_id=%d)",
                        modality.name, idx, len(ofs_list), pad_token_id,
                    )
                    item.offsets = []

            mm_items.append(item)

        mm_items = get_new_expanded_mm_items(mm_items)

        for item in mm_items:
            if item.hash is None:
                raw = item.feature if item.feature is not None else item.precomputed_embeddings
                if raw is not None and isinstance(raw, torch.Tensor):
                    item.hash = hash_feature(raw)

        return MultimodalProcessorOutput(
            input_ids=input_ids,
            mm_items=mm_items,
            im_token_id=self.image_pad_token_id,
            # video_token_id is required by MultiModalityDataPaddingPatternMultimodalTokens
            # so that pad_input_tokens() can replace video_place_token positions in input_ids
            # with item.pad_value.  Without this, _get_multimodal_mask() finds 0 mm tokens
            # in input_ids and the subsequent embedding scatter crashes with shape mismatch.
            video_token_id=self.video_place_token_id,
        )

    # ============================================================
    # Backup: process_mm_data_async via load_mm_data (OpenAI multimodal format)
    # ============================================================

    async def process_mm_data_async_backup(
        self,
        image_data: List[Union[str, bytes, Dict]],
        audio_data,
        input_text,
        request_obj,
        *args,
        **kwargs,
    ) -> MultimodalProcessorOutput:
        # TODO(video): Video support is not implemented in this path.  Videos passed via the
        # OpenAI ``video_url`` field would need to be routed through _process_video /
        # OmniVideoProcessor and produce Modality.VIDEO items, analogous to what
        # process_mm_data_async does.  Currently all video requests must use the primary
        # process_mm_data_async path (embedding references in input_text via
        # <video_start>…<video_end> tags).
        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
        )

        images = base_output.images or []
        if not images:
            input_ids = self._tokenizer(
                base_output.input_text,
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids.flatten()
            return MultimodalProcessorOutput(
                input_ids=input_ids.tolist(),
                mm_items=[],
                im_token_id=self.image_pad_token_id,
            )

        resized_dims = []
        token_counts = []
        factor = self.patch_size * self.merge_size  # 28

        # Per-request pixel budget override (falls back to defaults).
        # Validation is handled upstream in serving_chat.
        req_min_pixels = getattr(request_obj, "min_pixels", None)
        req_max_pixels = getattr(request_obj, "max_pixels", None)
        eff_min_pixels = req_min_pixels if req_min_pixels is not None else self.min_pixels
        eff_max_pixels = req_max_pixels if req_max_pixels is not None else self.max_pixels

        for image in images:
            if isinstance(image, torch.Tensor):
                h, w = image.shape[1], image.shape[2]
            else:
                w, h = image.size

            target_h, target_w = smart_resize(
                h, w, factor=factor,
                min_pixels=eff_min_pixels,
                max_pixels=eff_max_pixels,
            )
            resized_dims.append((target_h, target_w))

            grid_t = 1
            grid_h = target_h // self.patch_size
            grid_w = target_w // self.patch_size
            num_tokens = grid_t * (grid_h // self.merge_size) * (grid_w // self.merge_size)
            token_counts.append(num_tokens)

        # Expand placeholder tokens in text: <img_start><img_pad>×N<img_end>
        image_token_str = self.mm_tokens.image_token
        parts = base_output.input_text.split(image_token_str)
        result_text = parts[0]
        for i, part in enumerate(parts[1:]):
            if i < len(token_counts):
                expanded = (
                    self.image_start_token
                    + self.image_pad_token * token_counts[i]
                    + self.image_end_token
                )
                result_text += expanded + part
            else:
                result_text += image_token_str + part

        input_ids = self._tokenizer(
            result_text,
            return_tensors="pt",
            add_special_tokens=True,
        ).input_ids.flatten()

        # Preprocess images via OmniImageProcessor
        all_patches = []
        all_grids = []
        for image, (target_h, target_w) in zip(images, resized_dims):
            if isinstance(image, torch.Tensor):
                pil_image = Image.fromarray(image.permute(1, 2, 0).cpu().numpy().astype(np.uint8))
            else:
                pil_image = image
            feat, _, grid = self.visual_processor.image_transform(
                pil_image, min_pixels=eff_min_pixels, max_pixels=eff_max_pixels
            )
            all_patches.append(torch.from_numpy(feat.astype(np.float32)).cuda())
            all_grids.append(grid)

        pixel_values = torch.cat(all_patches, dim=0)

        # Build mm_items
        mm_items = []
        patch_offset = 0
        for i, (grid_thw, num_tokens) in enumerate(zip(all_grids, token_counts)):
            grid_t, grid_h, grid_w = grid_thw
            num_patches = grid_t * grid_h * grid_w
            item_pixels = pixel_values[patch_offset:patch_offset + num_patches]
            patch_offset += num_patches

            grid_thw_tensor = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.int64, device=item_pixels.device)
            item = MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=item_pixels,
            )
            item.model_specific_data["image_grid_thw"] = grid_thw_tensor
            mm_items.append(item)

        # Assign offsets: each item gets only its own offset segment
        all_offsets = self.get_mm_items_offset(input_ids, self.image_pad_token_id)
        for i, item in enumerate(mm_items):
            if i < len(all_offsets):
                item.offsets = [all_offsets[i]]
            else:
                logger.warning(f"Image {i} has no matching offset in input_ids (total offsets={len(all_offsets)})")
                item.offsets = []

        from sglang.srt.multimodal.mm_utils import get_new_expanded_mm_items, hash_feature
        mm_items = get_new_expanded_mm_items(mm_items)

        for item in mm_items:
            if item.hash is None:
                raw = item.feature if item.feature is not None else item.precomputed_embeddings
                if raw is not None and isinstance(raw, torch.Tensor):
                    item.hash = hash_feature(raw)

        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=mm_items,
            im_token_id=self.image_pad_token_id,
        )

    # ============================================================
    # Offline embedding path
    # ============================================================

    def get_mm_data(self, prompt, embeddings: Dict, **kwargs) -> MultimodalProcessorOutput:
        """Offline embedding path: build input_ids from grid info and assign embeddings.

        TODO(video): Video is not supported in this path.  To add support, accept
        ``video_grid_thw`` from kwargs, expand video placeholder tokens (self.video_place_token_id)
        analogously to images, slice embeddings[Modality.VIDEO], and construct Modality.VIDEO
        items.  Also set ``video_token_id=self.video_place_token_id`` in the returned
        MultimodalProcessorOutput so that pad_input_tokens() handles the video items correctly.
        See BaseMultimodalProcessor.get_mm_data for a reference implementation.
        """
        img_grid_thw = kwargs.get("img_grid_thw")
        if img_grid_thw is None:
            raise ValueError("img_grid_thw is required for LongcatFlashVL get_mm_data")

        if not isinstance(prompt, list):
            prompt = self._tokenizer.encode(prompt)

        image_token_id = self.image_pad_token_id
        merge_length = self.merge_size ** 2
        image_token_counts = [
            int(np.array(grid.tolist()).prod() // merge_length)
            for grid in img_grid_thw
        ]

        input_ids = []
        offsets = []
        img_idx = 0

        for token in prompt:
            if token != image_token_id:
                input_ids.append(token)
                continue

            if img_idx >= len(image_token_counts):
                raise ValueError(
                    "Number of image placeholders exceeds img_grid_thw entries."
                )

            num_tokens = image_token_counts[img_idx]
            start = len(input_ids)
            input_ids.extend([image_token_id] * num_tokens)
            offsets.append((start, len(input_ids) - 1))
            img_idx += 1

        if img_idx != len(image_token_counts):
            raise ValueError(
                "Number of image placeholders does not match img_grid_thw entries."
            )

        image_embeddings = embeddings[Modality.IMAGE]
        mm_items = []
        consumed = 0
        for start, end in offsets:
            num_tokens = end - start + 1
            emb_slice = image_embeddings[consumed:consumed + num_tokens]
            consumed += num_tokens
            mm_items.append(
                MultimodalDataItem(
                    modality=Modality.IMAGE,
                    offsets=[(start, end)],
                    precomputed_embeddings=emb_slice,
                )
            )

        return MultimodalProcessorOutput(
            input_ids=input_ids,
            mm_items=mm_items,
            im_token_id=image_token_id,
        )


# Encoder-side preprocessor (EPD): implements the EncoderPreprocessor
# interface for LongcatFlashVL.

class LongcatFlashVLEncoderPreprocessor(EncoderPreprocessor):
    """LongcatFlashVL Encoder-side preprocessor (implements EncoderPreprocessor).

    Only responsible for the stateless transform "URL → item + meta":
    download → transform → hash → MMItemMeta construction.
    ViT feature computation (get_image_feature / get_video_feature) is
    implemented by the model main class (EncoderFeature interface).

    Reuses the OmniImageProcessor / OmniVideoProcessor /
    VideoProcessingPipeline defined above so that preprocessing logic is
    written only once.

    """

    @classmethod
    def from_hf_config(cls, hf_config) -> "LongcatFlashVLEncoderPreprocessor":
        """EncoderPreprocessor factory method: build a preprocessor from
        ``hf_config``.

        Constructs OmniImageProcessor / OmniVideoProcessor /
        VideoProcessingPipeline internally from ``hf_config`` so the Encoder
        Server stays model-agnostic.
        """
        vis_cfg = hf_config.vision_config
        vid_cfg = getattr(hf_config, "video_config", None)

        patch_size = getattr(vis_cfg, "patch_size", 14)
        temporal_patch_size = getattr(vis_cfg, "temporal_patch_size", 2)
        merge_size = getattr(
            vis_cfg, "spatial_merge_size", getattr(vis_cfg, "merge_size", 2),
        )
        image_min_pixels = getattr(vis_cfg, "min_pixels", 31360)
        image_max_pixels = getattr(vis_cfg, "max_pixels", 1254400)
        image_mean = getattr(vis_cfg, "image_mean", IMAGE_MEAN)
        image_std = getattr(vis_cfg, "image_std", IMAGE_STD)

        self = cls.__new__(cls)
        self.visual_processor = OmniImageProcessor(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            spatial_merge_size=merge_size,
            min_pixels=image_min_pixels,
            max_pixels=image_max_pixels,
            image_mean=image_mean,
            image_std=image_std,
        )

        self.video_processor = OmniVideoProcessor(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            spatial_merge_size=merge_size,
            min_pixels=getattr(vid_cfg, "min_pixels", 128 * 28 * 28) if vid_cfg else 128 * 28 * 28,
            max_pixels=getattr(vid_cfg, "max_pixels", 768 * 28 * 28) if vid_cfg else 768 * 28 * 28,
            image_mean=image_mean,
            image_std=image_std,
            dynamic_frame_pixel=getattr(vid_cfg, "dynamic_frame_pixel", False) if vid_cfg else False,
            split_video=getattr(vid_cfg, "split_video", True) if vid_cfg else True,
        )

        video_max_frames = (
            getattr(vid_cfg, "max_frame_num", VIDEO_MAX_FRAMES)
            if vid_cfg else VIDEO_MAX_FRAMES
        )
        if "VIDEO_MAX_FRAMES" in os.environ:
            video_max_frames = int(os.environ["VIDEO_MAX_FRAMES"])
        if "VIDEO_MAX_PIXELS" in os.environ:
            self.video_processor.max_pixels = int(os.environ["VIDEO_MAX_PIXELS"])
        if "VIDEO_MIN_PIXELS" in os.environ:
            self.video_processor.min_pixels = int(os.environ["VIDEO_MIN_PIXELS"])

        self._video_handler = VideoProcessingPipeline(
            frame_processor=self.video_processor,
            image_start_token="<img_start>",
            image_end_token="<img_end>",
            video_start_token="<video_start>",
            video_end_token="<video_end>",
            video_place_token="<video_place>",
            frame_pattern=getattr(vid_cfg, "frame_pattern", "<frame>") if vid_cfg else "<frame>",
            videoframe_start_tag="<videoframe_start_omni>",
            videoframe_end_tag="<videoframe_end_omni>",
            video_max_frames=video_max_frames,
            video_decode_way=getattr(vid_cfg, "decode_way", "1fps") if vid_cfg else "1fps",
        )
        model_revision = (
            getattr(hf_config, "_commit_hash", None)
            or getattr(hf_config, "_name_or_path", None)
            or type(hf_config).__name__
        )
        common_signature = (
            "longcat_flash_vl_url_cache_v1",
            str(model_revision),
            int(patch_size),
            int(temporal_patch_size),
            int(merge_size),
            tuple(float(v) for v in image_mean),
            tuple(float(v) for v in image_std),
        )
        self._url_cache_signatures = {
            Modality.IMAGE: repr(common_signature + (
                "image",
                int(self.visual_processor.min_pixels),
                int(self.visual_processor.max_pixels),
            )),
            Modality.VIDEO: repr(common_signature + (
                "video",
                int(self.video_processor.min_pixels),
                int(self.video_processor.max_pixels),
                bool(self.video_processor.dynamic_frame_pixel),
                bool(self.video_processor.split_video),
                int(video_max_frames),
                str(self._video_handler.video_decode_way),
                str(self._video_handler.frame_pattern),
            )),
        }
        return self

    def get_url_cache_signature(self, modality: Modality) -> Optional[str]:
        signature = self._url_cache_signatures.get(modality)
        if signature is None:
            return super().get_url_cache_signature(modality)
        return hashlib.sha256(signature.encode("utf-8")).hexdigest()

    def process_image(self, url, item_index, min_pixels=None, max_pixels=None,
                      device_preprocess=False, device=None, *, timer: "Timer"):
        """Build one device-neutral image item for the Encoder scheduler.

        The implementation is selected from ``device.type``. Each helper owns
        its output contract: CPU and current Ascend hybrid helpers return
        contiguous CPU float32, while a resident CUDA helper returns contiguous device float32. Scheduler TP transport performs a
        conditional D2H only when it is actually required.
        """
        # Split source I/O from Pillow decode.  The transform implementations
        # accept a loaded PIL image, so they do not download/decode it again.
        with timer.phase("download"):
            if isinstance(url, Image.Image):
                image_bytes = None
                image = url
            elif isinstance(url, bytes):
                image_bytes = url
                image = None
            elif isinstance(url, str) and url.startswith("file://"):
                from urllib.parse import unquote, urlparse

                with open(unquote(urlparse(url).path), "rb") as image_file:
                    image_bytes = image_file.read()
                image = None
            elif isinstance(url, str) and os.path.isfile(url):
                with open(url, "rb") as image_file:
                    image_bytes = image_file.read()
                image = None
            else:
                image_bytes = get_image_bytes(url)
                image = None

        with timer.phase("decode"):
            if image is None:
                image = Image.open(BytesIO(image_bytes))
            image.load()

        visual = self.visual_processor
        with timer.phase("transform"):
            if device_preprocess and device is not None and device.type == "npu":
                from sglang.srt.multimodal.processors.longcat_flash_vl_npu_transform import (
                    npu_image_transform,
                )

                image_feat, _, grid_thw = npu_image_transform(
                    image,
                    processor=visual,
                    device=device,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                )
                pixel_values = image_feat
            elif device_preprocess and device is not None and device.type == "cuda":
                image_feat, _, grid_thw = visual.gpu_image_transform(
                    image,
                    device=device,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                )
                pixel_values = image_feat
            else:
                image_feat, _, grid_thw = visual.image_transform(
                    image, min_pixels=min_pixels, max_pixels=max_pixels,
                )
                pixel_values = torch.from_numpy(image_feat).contiguous()

        grid_thw_tensor = torch.tensor([grid_thw], dtype=torch.int64)
        item = MultimodalDataItem(modality=Modality.IMAGE, feature=pixel_values)
        item.model_specific_data["image_grid_thw"] = grid_thw_tensor
        with timer.phase("hash"):
            # CUDA hashing returns a Python int through ``Tensor.item()``. This
            # is the readiness barrier between async device preprocessing here
            # and ViT execution on EncoderDeviceWorker's persistent thread.
            item.hash = hash_feature(pixel_values)
        item.pad_value = _compute_pad_value(item.hash, item.modality)

        t, h, w = grid_thw
        merge_size = visual.merge_size
        num_tokens = int(t * (h // merge_size) * (w // merge_size))
        meta = MMItemMeta(
            item_index=item_index,
            hash=item.hash,
            pad_value=item.pad_value,
            grid_thw=[int(t), int(h), int(w)],
            num_tokens=num_tokens,
        )
        timer.mark_total()
        return PreprocessedItem(
            item=item, meta=meta, cost=num_tokens, is_cache_hit=False,
        )

    def process_video(self, url, item_index, max_pixels=None,
                      device_preprocess=False, device=None, *, timer: "Timer"):
        """Build one whole-video item preserving LongCat temporal layout."""
        result = self._video_handler.process_video(
            url, max_pixels=max_pixels,
            device_preprocess=device_preprocess,
            device=device,
            timer=timer,
        )
        if result is None:
            timer.mark_total()
            return []
        feat, _, grid_thw, _ = result
        t, h, w = grid_thw
        merge_size = self.video_processor.merge_size

        if isinstance(feat, torch.Tensor):
            # Device helpers own dtype/device/contiguity; avoid a redundant
            # to(dtype).contiguous().cpu() allocation in the caller.
            feat_tensor = feat
        else:
            feat_tensor = torch.from_numpy(feat).contiguous()
        grid_thw_tensor = torch.tensor([grid_thw], dtype=torch.int64)
        item = MultimodalDataItem(modality=Modality.VIDEO, feature=feat_tensor)
        item.model_specific_data["image_grid_thw"] = grid_thw_tensor
        with timer.phase("hash"):
            # See process_image: GPU hash materialization synchronizes the
            # producer stream before this item can be queued for ViT.
            item.hash = hash_feature(feat_tensor)
        item.pad_value = _compute_pad_value(item.hash, item.modality)

        num_tokens = int(t * (h // merge_size) * (w // merge_size))
        meta = MMItemMeta(
            item_index=item_index,
            hash=item.hash,
            pad_value=item.pad_value,
            grid_thw=[int(t), int(h), int(w)],
            num_tokens=num_tokens,
            video_timestamps=list(
                getattr(self._video_handler, "last_video_timestamps", []) or []
            ),
        )
        timer.mark_total()
        return [PreprocessedItem(
            item=item, meta=meta, cost=num_tokens, is_cache_hit=False,
        )]
