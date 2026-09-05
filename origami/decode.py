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
    to_ts: float,
    n_frames: int,
    fmt: str = "rgb24",
) -> Iterator[np.ndarray]:
    """Decode exactly ``n_frames`` frames starting at ``from_ts`` (seconds).

    §4.1: seek lands on the preceding keyframe, so pre-roll (dropping frames until we reach
    the requested timestamp) is mandatory. Raises ``DecodeError`` if fewer than ``n_frames``
    frames are available -- the caller (§4.4) is responsible for clamping episode length
    before calling this, never this function padding silently.

    Buffers into a list before yielding (rather than streaming directly) so a PTS
    discontinuity on the seek path can fall back to a full linear-scan retry without ever
    having already handed the caller a short/duplicate sequence.
    """
    frames = _decode_from_seek(video_path, from_ts, n_frames, fmt)
    if len(frames) < n_frames:
        logger.warning(
            "decode_episode_stream: seek path only found %d/%d frames for %s @ %.3fs; "
            "falling back to a linear scan from t=0",
            len(frames), n_frames, video_path, from_ts,
        )
        frames = _decode_linear_scan(video_path, from_ts, n_frames, fmt)
    yield from frames


def _decode_from_seek(video_path, from_ts, n_frames, fmt) -> list[np.ndarray]:
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        time_base = float(stream.time_base)
        pre_roll_cutoff = from_ts - 0.5 / float(stream.average_rate or 30)
        container.seek(int(from_ts / time_base), stream=stream)

        frames = []
        for frame in container.decode(stream):
            pts_s = frame.pts * time_base
            if pts_s < pre_roll_cutoff:
                continue
            frames.append(frame.to_ndarray(format=fmt))
            if len(frames) >= n_frames:
                break
        return frames
    finally:
        container.close()


def _decode_linear_scan(video_path, from_ts, n_frames, fmt) -> list[np.ndarray]:
    """Fallback: scan from t=0 and collect the first ``n_frames`` whose pts >= from_ts."""
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        time_base = float(stream.time_base)
        frames = []
        for frame in container.decode(stream):
            pts_s = frame.pts * time_base
            if pts_s < from_ts:
                continue
            frames.append(frame.to_ndarray(format=fmt))
            if len(frames) >= n_frames:
                break
        if len(frames) < n_frames:
            raise DecodeError(
                f"{video_path}: linear-scan fallback only found {len(frames)}/{n_frames} frames "
                f"from t={from_ts:.3f}s"
            )
        return frames
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
