"""Real-data observation sources for the offline replay, in wire format.

Both sources yield `(frame, observation, gt_future_abs, gt_frame_hz)` where
`observation` is a complete `origami-zenoh-v1` observation dict
(`docs/robot_io_spec.md` dtypes and shapes) and `gt_future_abs` is
`[>= 1, 65]` -- the teleoperator's absolute commands from that frame onward, at
30 Hz -- so a replay can score a chunk *and* stitch a 30 Hz ground-truth
stream.

  LeRobotSource   a season straight from the Hugging Face release
                  (`<season>/lerobot3.0`), decoded with ffmpeg.  This is the
                  same data the organizer's `real_observation_source.py`
                  replays, and the closest thing to the live robot available
                  offline: all four cameras, torque, tactile, deform grid.

  FlatSource      an origami-flat split (`trex_origami.prepare` output).  No
                  `head_right` stream exists there, so it is filled with
                  `head_left` (the checkpoint ignores it anyway) and torque is
                  zero-filled, as the organizer does for unavailable sources.
                  Useful because the held-out val split is already on disk in
                  this format.
"""
from __future__ import annotations

import io
import json
import os
from typing import Iterator, List, Optional, Tuple

import numpy as np
import PIL.Image
import pyarrow.parquet as pq

from .anchoring import build_anchor, spec_from_meta, to_absolute
from .lerobot_v3 import LeRobotV3Season, wire_observations
from .seasons import ACTION_DIM, DATASET_TASK_STRING

Sample = Tuple[int, dict, np.ndarray, float]


class LeRobotSource:
    def __init__(self, root: str, include_raw: bool = False):
        self.season = LeRobotV3Season(root)
        self.include_raw = include_raw
        self.fps = self.season.fps
        self.name = os.path.basename(os.path.dirname(self.season.root.rstrip("/")))

    def episode_indices(self) -> List[int]:
        return [ep.episode_index for ep in self.season.episodes]

    def episode_length(self, episode_index: int) -> int:
        return self.season.episode(episode_index).length

    def episode(self, episode_index: int, every: int, max_requests: int = 0,
                start: int = 0, decode_batch: int = 64) -> Iterator[Sample]:
        ep = self.season.episode(episode_index)
        offsets = list(range(start, ep.length, max(1, every)))
        if max_requests:
            offsets = offsets[:max_requests]
        for i in range(0, len(offsets), decode_batch):
            for o, obs, future in wire_observations(self.season, ep, offsets[i:i + decode_batch],
                                                    include_raw=self.include_raw):
                yield o, obs, future.astype(np.float64), self.fps

    def gt_actions(self, episode_index: int) -> np.ndarray:
        return self.season.arrays(self.season.episode(episode_index))["action"].astype(np.float64)


class FlatSource:
    def __init__(self, root: str):
        self.root = root
        with open(os.path.join(root, "meta", "dataset.json")) as handle:
            self.meta = json.load(handle)
        self.episodes = self.meta["episodes"]
        self.stride = int(self.meta.get("config", {}).get("sample_stride", 1))
        self.chunk = int(self.meta.get("config", {}).get("action_chunk", 25))
        self.spec = spec_from_meta(self.meta)
        self.fps = 30.0
        self.name = os.path.basename(root.rstrip("/"))

    def episode_indices(self) -> List[int]:
        return list(range(len(self.episodes)))

    def episode_length(self, episode_index: int) -> int:
        return int(self.episodes[episode_index]["source_frames"])

    def _table(self, episode_index: int):
        return pq.read_table(os.path.join(self.root, self.episodes[episode_index]["file"]))

    def gt_actions(self, episode_index: int) -> np.ndarray:
        """30 Hz absolute command stream reassembled from the rows' chunks."""
        table = self._table(episode_index)
        state = np.asarray(table["state"].to_pylist(), dtype=np.float64)
        prev = np.asarray(table["prev_command"].to_pylist(), dtype=np.float64)
        rel = np.asarray(table["action_chunk"].to_pylist(), dtype=np.float64).reshape(
            len(state), self.chunk, ACTION_DIM)
        absolute = to_absolute(rel, build_anchor(state, prev, self.spec))   # [R, T, D]
        if self.stride > self.chunk:
            raise ValueError("sample stride exceeds the chunk; no dense GT stream")
        return np.concatenate([absolute[r, :self.stride] for r in range(len(state))])

    def episode(self, episode_index: int, every: int, max_requests: int = 0,
                start: int = 0) -> Iterator[Sample]:
        """`every` is in source frames; rounded up to a multiple of the stride."""
        table = self._table(episode_index)
        n_rows = table.num_rows
        state = np.asarray(table["state"].to_pylist(), dtype=np.float32)
        prev = np.asarray(table["prev_command"].to_pylist(), dtype=np.float64)
        rel = np.asarray(table["action_chunk"].to_pylist(), dtype=np.float64).reshape(
            n_rows, self.chunk, ACTION_DIM)
        absolute = to_absolute(rel, build_anchor(state.astype(np.float64), prev, self.spec))
        hist = np.asarray(table["tacf6_hist"].to_pylist(), dtype=np.float32).reshape(
            n_rows, -1, 60)
        prompt = self.episodes[episode_index].get("instruction") or DATASET_TASK_STRING
        row_every = max(1, int(np.ceil(every / self.stride)))
        rows = list(range(start // self.stride, n_rows, row_every))
        if max_requests:
            rows = rows[:max_requests]
        gt_stream = self.gt_actions(episode_index)
        for r in rows:
            head = _jpeg_rgb(table["head"][r].as_py())
            deform_gray = np.asarray(PIL.Image.open(io.BytesIO(table["deform"][r].as_py()))
                                     .convert("L"))
            deform_rgb = np.ascontiguousarray(np.repeat(deform_gray[..., None], 3, axis=2))
            obs = {
                "observation/image/head_left": head,
                "observation/image/head_right": head.copy(),
                "observation/image/wrist_left": _jpeg_rgb(table["wrist_left"][r].as_py()),
                "observation/image/wrist_right": _jpeg_rgb(table["wrist_right"][r].as_py()),
                "observation/state": state[r].astype(np.float32),
                "observation/state/joint_torque": np.zeros(ACTION_DIM, dtype=np.float32),
                "observation/tactile": hist[r, -1].astype(np.float32),
                "observation/image/tactile_deform": deform_rgb,
                "prompt": prompt,
            }
            frame = r * self.stride
            yield frame, obs, gt_stream[frame:], self.fps


def _jpeg_rgb(blob: bytes, size: int = 224) -> np.ndarray:
    img = PIL.Image.open(io.BytesIO(blob)).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), PIL.Image.BILINEAR)
    return np.ascontiguousarray(np.asarray(img, dtype=np.uint8))


def make_source(kind: str, root: str, include_raw: bool = False):
    if kind == "lerobot":
        return LeRobotSource(root, include_raw=include_raw)
    if kind == "flat":
        return FlatSource(root)
    raise ValueError(kind)
