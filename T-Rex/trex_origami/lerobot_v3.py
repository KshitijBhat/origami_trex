"""Minimal reader for a Robotic Origami Challenge season in LeRobot v3.0 layout.

The organizer's `real_observation_source.py` replays the same data through
`LeRobotDataset`, which pulls in the whole `lerobot` package and decodes every
stream through PyAV one frame at a time.  This module reads the parquet tables
with pyarrow and decodes exactly the frames a replay asks for with one ffmpeg
call per stream (libdav1d handles the AV1 mp4s), which is all the offline
replay needs.

Layout (see `meta/info.json`):

    <season>/lerobot3.0/
      meta/info.json                 fps, path templates, features
      meta/episodes/chunk-*/file-*.parquet
                                     episode_index, length, dataset_from_index,
                                     dataset_to_index, videos/<key>/{chunk_index,
                                     file_index, from_timestamp, to_timestamp}
      meta/tasks.parquet             task strings
      data/chunk-*/file-*.parquet    observation.state, action,
                                     observation.state.joint_torque,
                                     observation.tactile, timestamp, frame_index, ...
      videos/<key>/chunk-*/file-*.mp4   one file holds one or more episodes
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq

CAMERA_KEYS = ("observation.images.head_left", "observation.images.head_right",
               "observation.images.wrist_left", "observation.images.wrist_right")
DEFORM_KEY = "observation.images.tactile_deform"
RAW_KEY = "observation.images.tactile_raw"
DATA_COLUMNS = ("observation.state", "action", "observation.state.joint_torque",
                "observation.tactile", "timestamp", "frame_index", "task_index")


@dataclass
class Episode:
    episode_index: int
    length: int
    row_from: int
    row_to: int
    task: str
    video: Dict[str, Tuple[int, int, float]]      # key -> (chunk, file, from_timestamp)


class LeRobotV3Season:
    def __init__(self, root: str):
        """`root` is the `lerobot3.0/` directory (or the season dir holding it)."""
        if not os.path.exists(os.path.join(root, "meta", "info.json")):
            nested = os.path.join(root, "lerobot3.0")
            if os.path.exists(os.path.join(nested, "meta", "info.json")):
                root = nested
            else:
                raise FileNotFoundError(f"no lerobot3.0 tree at {root}")
        self.root = root
        with open(os.path.join(root, "meta", "info.json")) as handle:
            self.info = json.load(handle)
        self.fps = float(self.info.get("fps", 30))
        self.video_path = self.info["video_path"]
        self.features = self.info["features"]
        self.video_keys = [k for k, v in self.features.items() if v.get("dtype") == "video"
                           and glob.glob(os.path.join(root, "videos", k, "**", "*.mp4"),
                                         recursive=True)]
        tasks = pq.read_table(os.path.join(root, "meta", "tasks.parquet")).to_pylist()
        # tasks.parquet is indexed by task string with a `task_index` column in
        # some exports and by position in others; handle both.
        self.tasks: Dict[int, str] = {}
        for i, row in enumerate(tasks):
            idx = int(row.get("task_index", i))
            text = row.get("task") or row.get("__index_level_0__") or next(
                (v for v in row.values() if isinstance(v, str)), "")
            self.tasks[idx] = text
        self.episodes = self._read_episodes()
        self._data: Optional[Dict[str, np.ndarray]] = None

    # ── metadata ──────────────────────────────────────────────────────────────
    def _read_episodes(self) -> List[Episode]:
        files = sorted(glob.glob(os.path.join(self.root, "meta", "episodes", "**", "*.parquet"),
                                 recursive=True))
        out: List[Episode] = []
        for path in files:
            table = pq.read_table(path)
            cols = set(table.schema.names)
            rows = table.select([c for c in table.schema.names
                                 if not c.startswith("stats/")]).to_pylist()
            for row in rows:
                video = {}
                for key in self.video_keys:
                    ck, fk, tk = (f"videos/{key}/chunk_index", f"videos/{key}/file_index",
                                  f"videos/{key}/from_timestamp")
                    if {ck, fk, tk} <= cols and row[ck] is not None:
                        video[key] = (int(row[ck]), int(row[fk]), float(row[tk]))
                tasks = row.get("tasks")
                task = tasks[0] if isinstance(tasks, list) and tasks else ""
                out.append(Episode(int(row["episode_index"]), int(row["length"]),
                                   int(row["dataset_from_index"]), int(row["dataset_to_index"]),
                                   str(task), video))
        out.sort(key=lambda e: e.episode_index)
        return out

    def episode(self, episode_index: int) -> Episode:
        for ep in self.episodes:
            if ep.episode_index == episode_index:
                return ep
        raise KeyError(f"episode {episode_index} not in {self.root}")

    # ── tables ────────────────────────────────────────────────────────────────
    @property
    def data(self) -> Dict[str, np.ndarray]:
        if self._data is None:
            files = sorted(glob.glob(os.path.join(self.root, "data", "**", "*.parquet"),
                                     recursive=True))
            parts: Dict[str, list] = {c: [] for c in DATA_COLUMNS}
            for path in files:
                table = pq.read_table(path, columns=[c for c in DATA_COLUMNS
                                                     if c in pq.ParquetFile(path).schema_arrow.names])
                for c in DATA_COLUMNS:
                    if c in table.schema.names:
                        col = table[c].to_pylist()
                        parts[c].append(np.asarray(col, dtype=np.float32
                                                   if c not in ("frame_index", "task_index")
                                                   else np.int64))
            self._data = {c: np.concatenate(v) for c, v in parts.items() if v}
        return self._data

    def arrays(self, ep: Episode) -> Dict[str, np.ndarray]:
        d = self.data
        return {c: d[c][ep.row_from:ep.row_to] for c in d}

    def task_text(self, ep: Episode) -> str:
        if ep.task:
            return ep.task
        idx = int(self.arrays(ep)["task_index"][0]) if "task_index" in self.data else 0
        return self.tasks.get(idx, "")

    # ── video ─────────────────────────────────────────────────────────────────
    def decode(self, ep: Episode, key: str, offsets: Sequence[int],
               size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Frames `offsets` (0 = first frame of the episode) of stream `key`.

        Returns uint8 [N, H, W, 3] RGB.  `size=(W, H)` resizes inside ffmpeg
        (bilinear, no letterbox), matching what the organizer does to the live
        cameras.  Uses an accurate `-ss` to the episode's first frame so the
        `select` expression is in episode-relative frame numbers.
        """
        if key not in ep.video:
            raise KeyError(f"{key} not available for episode {ep.episode_index}")
        chunk, file_idx, from_ts = ep.video[key]
        path = os.path.join(self.root, self.video_path.format(
            video_key=key, chunk_index=chunk, file_index=file_idx))
        feat = self.features[key]
        h, w = int(feat["shape"][0]), int(feat["shape"][1])
        offsets = sorted(int(o) for o in offsets)
        if not offsets:
            return np.zeros((0, h, w, 3), dtype=np.uint8)
        expr = "+".join(f"eq(n\\,{o})" for o in offsets)
        vf = f"select='{expr}'"
        out_w, out_h = (w, h) if size is None else (int(size[0]), int(size[1]))
        if size is not None:
            vf += f",scale={out_w}:{out_h}:flags=bilinear"
        cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-accurate_seek",
               "-ss", f"{from_ts:.6f}", "-i", path, "-vf", vf, "-fps_mode", "passthrough",
               "-frames:v", str(len(offsets)), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed on {path}: {proc.stderr.decode(errors='replace')[-800:]}")
        frame_bytes = out_w * out_h * 3
        n = len(proc.stdout) // frame_bytes
        if n != len(offsets):
            raise RuntimeError(f"{key}: asked for {len(offsets)} frames, decoded {n} "
                               f"(episode {ep.episode_index}, offsets up to {offsets[-1]})")
        return np.frombuffer(proc.stdout[:n * frame_bytes], dtype=np.uint8).reshape(
            n, out_h, out_w, 3)


def wire_observations(season: LeRobotV3Season, ep: Episode, offsets: Sequence[int],
                      prompt: Optional[str] = None, include_raw: bool = False):
    """Build `origami-zenoh-v1` observations for the given episode frames.

    Yields (offset, observation, gt_actions[T=all remaining frames, 65]) with
    the wire dtypes/shapes of `docs/robot_io_spec.md`.  Cameras are squashed to
    224x224 exactly as the organizer squashes the live 1920x1536 frames; the
    deform (and optional raw) tactile grids pass through at native size.
    """
    arrays = season.arrays(ep)
    offsets = [int(o) for o in offsets]
    cams = {key: season.decode(ep, key, offsets, size=(224, 224)) for key in CAMERA_KEYS}
    deform = season.decode(ep, DEFORM_KEY, offsets)
    raw = season.decode(ep, RAW_KEY, offsets) if include_raw and RAW_KEY in ep.video else None
    prompt = season.task_text(ep) if prompt is None else prompt
    torque = arrays.get("observation.state.joint_torque")
    for i, o in enumerate(offsets):
        obs = {
            "observation/image/head_left": np.ascontiguousarray(cams[CAMERA_KEYS[0]][i]),
            "observation/image/head_right": np.ascontiguousarray(cams[CAMERA_KEYS[1]][i]),
            "observation/image/wrist_left": np.ascontiguousarray(cams[CAMERA_KEYS[2]][i]),
            "observation/image/wrist_right": np.ascontiguousarray(cams[CAMERA_KEYS[3]][i]),
            "observation/state": arrays["observation.state"][o].astype(np.float32),
            "observation/state/joint_torque": (torque[o] if torque is not None
                                               else np.zeros(65)).astype(np.float32),
            "observation/tactile": arrays["observation.tactile"][o].astype(np.float32),
            "observation/image/tactile_deform": np.ascontiguousarray(deform[i]),
            "prompt": prompt,
        }
        if raw is not None:
            obs["observation/image/tactile_raw"] = np.ascontiguousarray(raw[i])
        yield o, obs, arrays["action"][o:]
