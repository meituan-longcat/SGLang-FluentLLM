"""Adapt MaaS video-frame inputs and Encoder results at the EPD boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from sglang.srt.disaggregation.encode.encode_data import (
    MMItemMeta,
    MMMeta,
    MMResult,
    MediaURL,
)
from sglang.srt.managers.scheduling.multimodal_data import _compute_pad_value
from sglang.srt.multimodal.mm_utils import hash_feature
from sglang.srt.multimodal.processors.maas_video_frames import (
    is_maas_video_frame_sequence,
    parse_maas_video_frame_sequence,
    serialize_maas_video_frame_sequence,
)


@dataclass(frozen=True)
class MaaSFrameRoutingResult:
    """Prepared MaaS frame inputs for the existing EPD Encoder interface.

    ``owner_resource_indices`` and ``compute_media_urls`` are parallel arrays.
    Each owner index is resolved by ``EncoderRouter``; positions in
    ``compute_media_urls`` become the dense ``media_indices`` sent to Encoder.

    For ``image + three-frame video + image`` with temporal patch size 2,
    owner indices are ``(0, 1, 3, 4)`` and compute-to-logical mapping is
    ``(0, 1, 1, 2)``. The padded copy of frame 2 keeps resource index 3.
    """

    compute_media_urls: tuple[MediaURL, ...]
    owner_resource_indices: tuple[int, ...]
    compute_to_logical: Optional[tuple[int, ...]]


def build_maas_frame_routing(
    prompt_media_urls: Sequence[MediaURL],
    mm_processor: object,
) -> Optional[MaaSFrameRoutingResult]:
    """Build temporal-group inputs, or return ``None`` when no marker exists."""

    if not any(
        media_url.is_video and is_maas_video_frame_sequence(media_url.url)
        for media_url in prompt_media_urls
    ):
        return None

    video_processor = getattr(mm_processor, "video_processor", None)
    temporal_patch_size = int(
        getattr(mm_processor, "temporal_patch_size", 0)
        or getattr(video_processor, "temporal_patch_size", 0)
    )
    max_frames = int(getattr(mm_processor, "video_max_frames", 0))
    if temporal_patch_size <= 0:
        raise ValueError(
            "MaaS frame routing requires a positive temporal_patch_size"
        )

    compute_media_urls: list[MediaURL] = []
    owner_resource_indices: list[int] = []
    compute_to_logical: list[int] = []
    next_resource_index = 0

    for logical_index, media_url in enumerate(prompt_media_urls):
        is_marker_video = media_url.is_video and is_maas_video_frame_sequence(
            media_url.url
        )
        if not is_marker_video:
            compute_media_urls.append(media_url)
            owner_resource_indices.append(next_resource_index)
            compute_to_logical.append(logical_index)
            next_resource_index += 1
            continue

        frame_refs = parse_maas_video_frame_sequence(
            media_url.url,
            max_frames=max_frames,
        )
        indexed_frames = [
            (frame_ref, next_resource_index + index)
            for index, frame_ref in enumerate(frame_refs)
        ]
        next_resource_index += len(frame_refs)

        remainder = len(indexed_frames) % temporal_patch_size
        if remainder:
            indexed_frames.extend(
                [indexed_frames[-1]] * (temporal_patch_size - remainder)
            )

        for start in range(0, len(indexed_frames), temporal_patch_size):
            group = indexed_frames[start : start + temporal_patch_size]
            compute_media_urls.append(
                MediaURL.video(
                    serialize_maas_video_frame_sequence(
                        [frame_ref for frame_ref, _ in group]
                    )
                )
            )
            # A temporal group is an indivisible model input. MaaS assigns an
            # endpoint per raw frame, so the first real frame owns the group.
            owner_resource_indices.append(group[0][1])
            compute_to_logical.append(logical_index)

    requires_result_fold = len(compute_media_urls) != len(prompt_media_urls)
    if requires_result_fold and bool(
        getattr(video_processor, "dynamic_frame_pixel", False)
    ):
        raise ValueError(
            "EPD MaaS frame routing does not support dynamic_frame_pixel=True"
        )

    return MaaSFrameRoutingResult(
        compute_media_urls=tuple(compute_media_urls),
        owner_resource_indices=tuple(owner_resource_indices),
        compute_to_logical=(
            tuple(compute_to_logical) if requires_result_fold else None
        ),
    )


def fold_maas_frame_results(
    request_id: str,
    media_urls: Sequence[MediaURL],
    compute_to_logical: Sequence[int],
    mm_meta: MMMeta,
    mm_results: dict[int, MMResult],
) -> tuple[MMMeta, dict[int, MMResult]]:
    """Fold complete temporal-group outputs into prompt-level media items.

    Results are ordered by dense compute index, not network arrival. Missing,
    inconsistent, or mixed-format group outputs fail the entire request rather
    than returning a partial video. The aggregate hash is deterministic across
    Prefill and Decode, but intentionally differs from the hash of a whole-video
    preprocessing tensor.
    """

    logical_indices = list(compute_to_logical)
    for logical_index in logical_indices:
        if (
            not isinstance(logical_index, int)
            or isinstance(logical_index, bool)
            or logical_index < 0
            or logical_index >= len(media_urls)
        ):
            raise ValueError("EPD logical media index is out of range")
    if set(logical_indices) != set(range(len(media_urls))):
        raise ValueError(
            "EPD logical media indices do not match original media list"
        )

    meta_by_compute = {meta.item_index: meta for meta in mm_meta.item_metas}
    expected = set(range(len(logical_indices)))
    if set(meta_by_compute) != expected:
        raise ValueError("EPD temporal-group metadata is incomplete")

    result_mode: Optional[str] = None
    if mm_results:
        if set(mm_results) != expected:
            raise ValueError("EPD temporal-group results are incomplete")
        for compute_index in range(len(logical_indices)):
            result = mm_results[compute_index]
            meta = meta_by_compute[compute_index]
            if result.error:
                raise ValueError(result.error)
            if result.item_index != compute_index:
                raise ValueError("EPD result item index does not match its key")
            has_rvq = result.rvq_indices is not None
            has_embedding = result.vit_embedding is not None
            if has_rvq == has_embedding:
                raise ValueError(
                    "EPD result must contain exactly one output tensor"
                )
            mode = "rvq" if has_rvq else "embedding"
            if result_mode is None:
                result_mode = mode
            elif result_mode != mode:
                raise ValueError("EPD temporal groups use different output formats")
            if (
                result.data.shape[0] != result.num_tokens
                or result.num_tokens != meta.num_tokens
            ):
                raise ValueError(
                    "EPD result tensor length does not match metadata"
                )

    grouped: dict[int, list[int]] = {}
    for compute_index, logical_index in enumerate(logical_indices):
        grouped.setdefault(logical_index, []).append(compute_index)

    logical_metas: list[MMItemMeta] = []
    logical_results: dict[int, MMResult] = {}
    for logical_index, media_url in enumerate(media_urls):
        compute_indices = grouped[logical_index]
        metas = [meta_by_compute[index] for index in compute_indices]
        if len(compute_indices) == 1:
            meta = metas[0]
            logical_metas.append(
                MMItemMeta(
                    item_index=logical_index,
                    hash=meta.hash,
                    pad_value=meta.pad_value,
                    grid_thw=list(meta.grid_thw),
                    num_tokens=meta.num_tokens,
                    embedding_dim=meta.embedding_dim,
                    video_timestamps=list(meta.video_timestamps),
                )
            )
            result = mm_results.get(compute_indices[0])
            if result is not None:
                logical_results[logical_index] = MMResult(
                    request_id=request_id,
                    item_index=logical_index,
                    rvq_indices=result.rvq_indices,
                    vit_embedding=result.vit_embedding,
                    num_tokens=result.num_tokens,
                )
            continue

        if not media_url.is_video:
            raise ValueError(
                f"logical image {logical_index} has multiple compute items"
            )
        if any(
            len(meta.grid_thw) != 3 or int(meta.grid_thw[0]) != 1
            for meta in metas
        ):
            raise ValueError("each temporal group must produce grid_t=1")
        if any(
            len(meta.video_timestamps) != int(meta.grid_thw[0])
            for meta in metas
        ):
            raise ValueError(
                "each temporal group must provide one timestamp per grid_t"
            )
        if len({meta.embedding_dim for meta in metas}) != 1:
            raise ValueError(
                f"logical media {logical_index} has inconsistent embedding dimensions"
            )
        spatial_shapes = {
            (int(meta.grid_thw[1]), int(meta.grid_thw[2])) for meta in metas
        }
        if len(spatial_shapes) != 1:
            raise ValueError(
                f"logical media {logical_index} has inconsistent group grids: "
                f"{sorted(spatial_shapes)}"
            )

        grid_h, grid_w = next(iter(spatial_shapes))
        # Decode has group metadata but no full result tensor, so it cannot
        # reproduce the whole-video feature hash. Hash the ordered group identity
        # with a versioned domain tag to derive the same logical hash on P and D.
        hash_material = "|".join(
            ["epd-frame-fold-v1"]
            + [
                f"{meta.hash}:{meta.grid_thw}:{meta.num_tokens}:"
                f"{meta.video_timestamps}"
                for meta in metas
            ]
        ).encode("utf-8")
        logical_hash = hash_feature(hash_material)
        logical_metas.append(
            MMItemMeta(
                item_index=logical_index,
                hash=logical_hash,
                pad_value=_compute_pad_value(logical_hash, media_url.modality),
                grid_thw=[len(metas), grid_h, grid_w],
                num_tokens=sum(int(meta.num_tokens) for meta in metas),
                embedding_dim=metas[0].embedding_dim,
                video_timestamps=[
                    timestamp
                    for meta in metas
                    for timestamp in meta.video_timestamps
                ],
            )
        )

        if mm_results:
            data = torch.cat(
                [mm_results[index].data for index in compute_indices], dim=0
            ).contiguous()
            logical_results[logical_index] = MMResult(
                request_id=request_id,
                item_index=logical_index,
                rvq_indices=data if result_mode == "rvq" else None,
                vit_embedding=data if result_mode == "embedding" else None,
                num_tokens=data.shape[0],
            )

    return (
        MMMeta(
            request_id=request_id,
            item_metas=logical_metas,
            vit_pending=mm_meta.vit_pending,
        ),
        logical_results,
    )
