"""MaaS pre-extracted video-frame transport format."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import List, Sequence


MAAS_FRAME_START_TAG = "<img_token_start>"
MAAS_FRAME_END_TAG = "<img_token_end>"
MAAS_TIMESTAMP_PREFIX = "at second["

_TIMESTAMP_PATTERN = re.compile(r"[0-9]+(?:\.[0-9]+)?")


class MaaSVideoFrameError(ValueError):
    """Invalid MaaS pre-extracted video-frame input."""


@dataclass(frozen=True)
class MaaSVideoFrameRef:
    """One ordered frame source carried in ``video_url.url``."""

    url: str
    timestamp_text: str


def is_maas_video_frame_sequence(video_info: object) -> bool:
    """Return whether input contains a MaaS transport marker."""

    return isinstance(video_info, str) and (
        MAAS_FRAME_START_TAG in video_info
        or MAAS_FRAME_END_TAG in video_info
    )


def parse_maas_video_frame_sequence(
    video_info: str,
    *,
    max_frames: int,
) -> List[MaaSVideoFrameRef]:
    """Parse ordered ``frame source + timestamp`` entries.

    Whitespace is accepted before, after, and between complete entries. Frame
    sources are intentionally left to FluentLLM's existing image loader.
    """

    if not isinstance(video_info, str):
        raise MaaSVideoFrameError("MaaS video frame sequence must be a string")
    if max_frames <= 0:
        raise MaaSVideoFrameError(
            "MaaS video requires a positive configured frame-count limit"
        )

    refs: List[MaaSVideoFrameRef] = []
    cursor = 0
    previous_timestamp = Decimal("-Infinity")

    while cursor < len(video_info):
        while cursor < len(video_info) and video_info[cursor].isspace():
            cursor += 1
        if cursor == len(video_info):
            break
        if not video_info.startswith(MAAS_FRAME_START_TAG, cursor):
            raise MaaSVideoFrameError(
                f"expected {MAAS_FRAME_START_TAG!r} at offset {cursor}"
            )
        cursor += len(MAAS_FRAME_START_TAG)

        frame_end = video_info.find(MAAS_FRAME_END_TAG, cursor)
        if frame_end < 0:
            raise MaaSVideoFrameError(f"missing {MAAS_FRAME_END_TAG!r}")
        frame_url = video_info[cursor:frame_end]
        if not frame_url:
            raise MaaSVideoFrameError("MaaS frame source is empty")
        cursor = frame_end + len(MAAS_FRAME_END_TAG)

        if not video_info.startswith(MAAS_TIMESTAMP_PREFIX, cursor):
            raise MaaSVideoFrameError(
                f"expected timestamp immediately after {MAAS_FRAME_END_TAG!r}"
            )
        timestamp_start = cursor + len(MAAS_TIMESTAMP_PREFIX)
        timestamp_end = video_info.find("]", timestamp_start)
        if timestamp_end < 0:
            raise MaaSVideoFrameError("MaaS frame timestamp is missing closing ']'")

        timestamp_value = video_info[timestamp_start:timestamp_end]
        if _TIMESTAMP_PATTERN.fullmatch(timestamp_value) is None:
            raise MaaSVideoFrameError(
                "MaaS frame timestamp must use non-negative decimal notation"
            )
        timestamp_decimal = Decimal(timestamp_value)
        if timestamp_decimal < previous_timestamp:
            raise MaaSVideoFrameError("MaaS frame timestamps must be non-decreasing")

        refs.append(
            MaaSVideoFrameRef(
                url=frame_url,
                timestamp_text=video_info[cursor : timestamp_end + 1],
            )
        )
        if len(refs) > max_frames:
            raise MaaSVideoFrameError(
                f"MaaS frame count exceeds model limit: {len(refs)} > {max_frames}"
            )

        previous_timestamp = timestamp_decimal
        cursor = timestamp_end + 1

    if not refs:
        raise MaaSVideoFrameError("MaaS video frame sequence is empty")
    return refs


def serialize_maas_video_frame_sequence(
    frame_refs: Sequence[MaaSVideoFrameRef],
) -> str:
    """Serialize one internal temporal group for the existing VIDEO payload."""

    return "".join(
        MAAS_FRAME_START_TAG
        + frame_ref.url
        + MAAS_FRAME_END_TAG
        + frame_ref.timestamp_text
        for frame_ref in frame_refs
    )
