"""PyAV episode-slice video decode, deform-strip splitting, and RGB wire-resize.

REDESIGN_PLAN.md §4. Prep and deploy must decode bit-identically from the 224x224 wire
image onward (invariant 3) -- ``squash_to_wire`` is shared by ``convert.py`` and
``serve_zenoh.py``.
"""
from __future__ import annotations

import logging
from typing import Iterator

import av
import cv2
import numpy as np

from origami.constants import (
    DEFORM_N_COLS,
    DEFORM_N_ROWS,
    DEFORM_STRIP_SHAPE,
    DEFORM_TILE_SHAPE,
    RAW_STRIP_SHAPE,
    RAW_TILE_SHAPE,
    WIRE_IMAGE_SIZE,
)

logger = logging.getLogger(__name__)


class DecodeError(RuntimeError):
    pass


def decode_episode_stream(
    video_path: str,
    from_ts: float,
    n_frames: int,
    fmt: str = "rgb24",
) -> Iterator[np.ndarray]:
    """Decode exactly ``n_frames`` frames starting at ``from_ts`` (seconds).

    §4.1: seek lands on the preceding keyframe, so pre-roll (dropping frames until we reach
    the requested timestamp) is mandatory. Raises ``DecodeError`` if fewer than ``n_frames``
    frames are available -- the caller (§4.4) is responsible for clamping episode length
    before calling this, never this function padding silently.

    True streaming: yields each frame's ndarray as it's decoded, never materializing a whole
    episode's frames as a Python list (a 4000+-frame, 480x480 episode across 4 video streams
    held simultaneously that way is 10+GB and OOMs -- exactly what §5.4's bounded-memory
    design is meant to avoid elsewhere). ``probe_available_frames`` (cheap: counts frames
    without converting any to ndarray) decides seek-vs-linear-scan *before* any pixel data is
    decoded, so a PTS discontinuity never means frames already handed to the caller need to be
    thrown away or duplicated.
    """
    if probe_available_frames(video_path, from_ts, n_frames) >= n_frames:
        yield from _stream_from_seek(video_path, from_ts, n_frames, fmt)
        return

    logger.warning(
        "decode_episode_stream: seek path can't supply %d frames for %s @ %.3fs; "
        "falling back to a linear scan from t=0",
        n_frames, video_path, from_ts,
    )
    if _count_linear_scan(video_path, from_ts, n_frames) < n_frames:
        raise DecodeError(
            f"{video_path}: linear-scan fallback can't supply {n_frames} frames from "
            f"t={from_ts:.3f}s"
        )
    yield from _stream_linear_scan(video_path, from_ts, n_frames, fmt)


def _stream_from_seek(video_path, from_ts, n_frames, fmt) -> Iterator[np.ndarray]:
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        time_base = float(stream.time_base)
        pre_roll_cutoff = from_ts - 0.5 / float(stream.average_rate or 30)
        container.seek(int(from_ts / time_base), stream=stream)

        n = 0
        for frame in container.decode(stream):
            pts_s = frame.pts * time_base
            if pts_s < pre_roll_cutoff:
                continue
            yield frame.to_ndarray(format=fmt)
            n += 1
            if n >= n_frames:
                break
    finally:
        container.close()


def _stream_linear_scan(video_path, from_ts, n_frames, fmt) -> Iterator[np.ndarray]:
    """Fallback: scan from t=0 and yield the first ``n_frames`` whose pts >= from_ts."""
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        n = 0
        for frame in container.decode(stream):
            pts_s = frame.pts * float(stream.time_base)
            if pts_s < from_ts:
                continue
            yield frame.to_ndarray(format=fmt)
            n += 1
            if n >= n_frames:
                break
    finally:
        container.close()


def _count_linear_scan(video_path, from_ts, max_frames) -> int:
    """Cheap (no ndarray decode) frame count for the linear-scan path, mirroring
    ``probe_available_frames``'s seek-path counting."""
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        n = 0
        for frame in container.decode(stream):
            pts_s = frame.pts * float(stream.time_base)
            if pts_s < from_ts:
                continue
            n += 1
            if n >= max_frames:
                break
        return n
    finally:
        container.close()


def probe_available_frames(video_path: str, from_ts: float, max_frames: int) -> int:
    """Count frames actually available from ``from_ts`` onward, capped at ``max_frames``,
    without raising -- used by the caller (§4.4) to reconcile episode length across streams
    before requesting an exact-count decode."""
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        time_base = float(stream.time_base)
        pre_roll_cutoff = from_ts - 0.5 / float(stream.average_rate or 30)
        container.seek(int(from_ts / time_base), stream=stream)
        n = 0
        for frame in container.decode(stream):
            pts_s = frame.pts * time_base
            if pts_s < pre_roll_cutoff:
                continue
            n += 1
            if n >= max_frames:
                break
        return n
    finally:
        container.close()


def split_deform_strip(y: np.ndarray) -> np.ndarray:
    """(480,1200) uint8 luma -> (10,240,240) uint8, [L thumb..pinky, R thumb..pinky]."""
    assert y.shape == DEFORM_STRIP_SHAPE, y.shape
    h, w = DEFORM_TILE_SHAPE
    return y.reshape(DEFORM_N_ROWS, h, DEFORM_N_COLS, w).transpose(0, 2, 1, 3).reshape(10, h, w)


def split_raw_strip(y: np.ndarray) -> np.ndarray:
    """(480,1600) uint8 luma -> (10,240,320) uint8, same finger order as split_deform_strip."""
    assert y.shape == RAW_STRIP_SHAPE, y.shape
    h, w = RAW_TILE_SHAPE
    return y.reshape(DEFORM_N_ROWS, h, DEFORM_N_COLS, w).transpose(0, 2, 1, 3).reshape(10, h, w)


def squash_to_wire(rgb480: np.ndarray) -> np.ndarray:
    """(480,480,3) -> (224,224,3), deploy-identical (D7). INTER_AREA for the 480->224
    decimation -- correct for downsampling, sharpness-only difference vs the organizer's
    1920x1536->224 squash, not a geometry difference."""
    h, w = WIRE_IMAGE_SIZE
    return cv2.resize(rgb480, (w, h), interpolation=cv2.INTER_AREA)


def reconcile_episode_length(length_from_meta: int, frames_decoded_per_stream: list[int]) -> int:
    """§4.4: N = min(length_from_meta, every stream's decoded frame count)."""
    return min([length_from_meta, *frames_decoded_per_stream])
