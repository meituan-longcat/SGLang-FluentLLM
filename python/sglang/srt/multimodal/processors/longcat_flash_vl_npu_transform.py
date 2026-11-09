"""Ascend device-side image and video transforms for Longcat Flash VL.

Decode and smart-resize shape selection stay on CPU. Resize and normalize run
on NPU. Patchify currently returns to CPU because the 9-D materialization hits
the Ascend ``aclnnInplaceCopy`` rank limit; removing that round-trip belongs to
the NPU-resident P2 work.
"""

from typing import Optional, Sequence, Tuple

import numpy as np
import torch


# PIL BICUBIC and torch bicubic use different kernels. This tolerance covers
# the measured normalized delta and must still be guarded by end-to-end tests.
_NPU_CPU_MAX_ABS_DELTA = 0.35


def _load_rgb_image(image_input):
    """Load one input as a fully decoded RGB PIL image."""
    from PIL import Image
    from sglang.srt.utils.common import load_image

    if isinstance(image_input, Image.Image):
        image = image_input
    else:
        image, _ = load_image(image_input)
    image.load()
    return image if image.mode == "RGB" else image.convert("RGB")


def _pil_to_chw_tensor(image: "Image.Image") -> torch.Tensor:
    """PIL RGB -> zero-copy CHW view over one writable uint8 HWC array."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    array = np.array(image)  # one writable HWC allocation
    return torch.from_numpy(array).permute(2, 0, 1)


def _resize_normalize_batch(
    chw: torch.Tensor,
    device: torch.device,
    resized_h: int,
    resized_w: int,
    processor,
) -> torch.Tensor:
    """Upload uint8 NCHW, then resize and normalize as float32 on device."""
    from sglang.srt.multimodal.processors.longcat_flash_vl import (
        _get_device_normalization_tensors,
    )

    tensor = chw.to(device).float()
    if (resized_h, resized_w) != (chw.shape[-2], chw.shape[-1]):
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(resized_h, resized_w),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        tensor = tensor.clamp(0, 255)
    mean, std = _get_device_normalization_tensors(
        processor, device, tensor.dtype
    )
    return (tensor / 255.0 - mean) / std


def _patchify_on_cpu(
    tensor: torch.Tensor,
    processor,
    resized_h: int,
    resized_w: int,
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """Apply the model's temporal/spatial patch order on a CPU float32 tensor."""
    tensor = tensor.to(device="cpu", dtype=torch.float32).contiguous()
    temporal_patch_size = processor.temporal_patch_size
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
    grid_h = resized_h // processor.patch_size
    grid_w = resized_w // processor.patch_size
    merge = processor.spatial_merge_size
    patch_size = processor.patch_size
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
    return flatten_patches, (grid_t, grid_h, grid_w)


def npu_image_transform(
    image_input,
    processor,
    device: torch.device,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int, int]]:
    """Resize/normalize one image on NPU and return contiguous CPU float32 patches."""
    from sglang.srt.multimodal.processors.longcat_flash_vl import smart_resize

    image = _load_rgb_image(image_input)
    image_org_size = image.size[::-1]
    chw = _pil_to_chw_tensor(image)
    resized_h, resized_w = smart_resize(
        image_org_size[0],
        image_org_size[1],
        factor=processor.patch_size * processor.spatial_merge_size,
        min_pixels=(processor.min_pixels if min_pixels is None else min_pixels),
        max_pixels=(processor.max_pixels if max_pixels is None else max_pixels),
    )
    tensor = _resize_normalize_batch(
        chw.unsqueeze(0), device, resized_h, resized_w, processor
    )
    flatten_patches, grid_thw = _patchify_on_cpu(
        tensor, processor, resized_h, resized_w
    )
    return flatten_patches, image_org_size, grid_thw


def _effective_video_max_pixels(
    processor,
    num_frames: int,
    max_pixels: Optional[int],
) -> int:
    """Match OmniVideoProcessor's inference-time dynamic pixel budget."""
    if max_pixels is not None:
        return max_pixels
    if not processor.dynamic_frame_pixel:
        return processor.max_pixels

    border = [512, 256, 128, 64, 32, 16]
    effective_max_pixels = processor.max_pixels
    for index in range(1, len(border)):
        if num_frames > border[index]:
            effective_max_pixels = (
                processor.frame_tokens[index - 1] * 14 * 14 * 4
            )
            break
    if num_frames <= border[-1]:
        effective_max_pixels = processor.frame_tokens[-1] * 14 * 14 * 4
    return effective_max_pixels


def npu_video_transform(
    frame_inputs: Sequence,
    processor,
    device: torch.device,
    max_pixels: Optional[int] = None,
) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int, int]]:
    """Batch video Resize/Normalize on NPU, preserving temporal patch ordering.

    The decoded frames remain uint8 until H2D. A normal fixed-resolution video
    uses one H2D and one batched interpolate call. Variable-size frame inputs
    are handled per frame and must still resolve to one common output size, as
    required by the original NumPy stack contract.
    """
    from sglang.srt.multimodal.processors.longcat_flash_vl import smart_resize

    images = [_load_rgb_image(frame) for frame in frame_inputs]
    if not images:
        raise ValueError("video contains no sampled frames")

    effective_max_pixels = _effective_video_max_pixels(
        processor, len(images), max_pixels
    )
    original_sizes = [image.size[::-1] for image in images]
    resized_sizes = [
        smart_resize(
            height,
            width,
            factor=processor.patch_size * processor.spatial_merge_size,
            min_pixels=processor.min_pixels,
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

    if len(set(original_sizes)) == 1:
        height, width = original_sizes[0]
        batch_hwc = np.empty(
            (len(images), height, width, 3), dtype=np.uint8
        )
        for index, image in enumerate(images):
            batch_hwc[index] = np.asarray(image, dtype=np.uint8)
        chw_batch = torch.from_numpy(batch_hwc).permute(0, 3, 1, 2)
        tensor = _resize_normalize_batch(
            chw_batch, device, resized_h, resized_w, processor
        )
    else:
        tensors = []
        for image in images:
            chw = _pil_to_chw_tensor(image).unsqueeze(0)
            tensors.append(
                _resize_normalize_batch(
                    chw, device, resized_h, resized_w, processor
                )
            )
        tensor = torch.cat(tensors, dim=0)

    flatten_patches, grid_thw = _patchify_on_cpu(
        tensor, processor, resized_h, resized_w
    )
    return flatten_patches, original_sizes[-1], grid_thw


def assert_npu_cpu_equivalent(
    image_input,
    processor,
    device: torch.device,
    max_abs_delta: float = _NPU_CPU_MAX_ABS_DELTA,
) -> None:
    """Assert that the NPU image path stays numerically close to the CPU path."""
    cpu_patches, _, cpu_grid = processor.image_transform(image_input)
    npu_patches, _, npu_grid = npu_image_transform(
        image_input, processor, device
    )

    assert cpu_grid == npu_grid, f"grid mismatch: cpu={cpu_grid} npu={npu_grid}"
    cpu_tensor = torch.as_tensor(cpu_patches)
    assert cpu_tensor.shape == npu_patches.shape, (
        f"shape mismatch: cpu={cpu_tensor.shape} npu={npu_patches.shape}"
    )
    delta = (cpu_tensor - npu_patches).abs().max().item()
    assert delta <= max_abs_delta, (
        f"NPU vs CPU transform delta {delta:.4f} exceeds tolerance "
        f"{max_abs_delta:.4f} (check bicubic antialias/clamp)"
    )
