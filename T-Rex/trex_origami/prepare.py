"""Convert Robotic Origami Challenge seasons into the "origami-flat" format.

Why a bespoke format instead of feeding LeRobot v3.0 to `TRexLeRobotDataset`:
the origami release ships one concatenated mp4 per camera per *season* and a
single 1200x480 mp4 holding all ten fingertip deform maps, so a LeRobot-style
random-access sample costs four video seeks into AV1 streams and forces the
sample rate to equal the source 30 Hz.  Baking the action chunk and the 30 Hz F6
window into each row and storing frames as JPEG blobs turns one training sample
into one parquet row read plus four JPEG decodes, and decouples the sample rate
from the action rate.

Output layout (read by `qwen_vla.origami_dataset.OrigamiDataset`):

    <out_root>/
      meta/dataset.json     config + episode index
      meta/norm_stats.json  q01/q99 normalisation (written by stats.py)
      data/<season>/ep<NNN>.parquet

Row schema, at source frame `t` of an episode of length `N` starting at `s`:

    state         [65]        observation.state[t]
    action_chunk  [16*65]     action[min(t+k, s+N-1)]                 k = 0..15
    action_abs    [65]        action[t]  (== action_chunk[0]; kept for convenience)
    phase         float       (t - s) / (N - 1)
    head/wrist_left/wrist_right  JPEG bytes, 224x224 RGB
    deform        JPEG bytes, 1200x480 (grayscale content, 2x5 grid of 240x240)
    tacf6_hist    [16*10*6]   tactile[clip(t-15+i, s, t)]             i = 0..15

The chunk is **all-absolute** on all 65 dims -- no delta, no anchoring, matching
the raw dataset's own native representation (`meta/modality.json` already marks
the whole 65-D block absolute) and the competition's wire contract (65-D
absolute-radian joints; `observation.state.tcp` is identically zero, so there's
no eef pose to make a delta representation meaningful for anyway). This
deliberately drops the delta-from-state approach a prior attempt used (and the
hybrid per-dim anchoring that followed it) -- see PLAN_fresh_branch.md for why.
The F6 history stays at the native 30 Hz regardless of `sample_stride`, because
the embedded VQ-VAE was trained on 30 Hz windows.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .seasons import (
    ACTION_DIM,
    INSTRUCTION,
    SRC_FPS,
    VIDEO_KEYS,
    VIDEO_KEY_TO_COLUMN,
    select_seasons,
)

logger = logging.getLogger(__name__)

JPEG_SOI = b"\xff\xd8\xff"
F6_FINGERS = 10
F6_PER_FINGER = 6


# ── configuration ─────────────────────────────────────────────────────────────
@dataclass
class PrepConfig:
    sample_stride: int = 5        # emit every Nth source frame (5 -> 6 Hz samples)
    chunk_stride: int = 1         # spacing *inside* the action chunk, in source frames
    action_chunk: int = 16        # matches original T-Rex's own action_chunk, not the kit's action_horizon
    action_dim: int = ACTION_DIM  # 65
    vqvae_window: int = 16        # F6 history length, at the native 30 Hz
    image_size: int = 224         # square, matching the 224x224 the wire delivers
    rgb_quality: int = 3          # ffmpeg -q:v for the three RGB cameras (2..31, lower=better)
    deform_quality: int = 4       # ffmpeg -q:v for the deform strip
    instruction: str = INSTRUCTION
    n_phases: int = 6             # the target figure is a 6-fold plane
    phase_mode: str = "none"      # "progress" bakes "(fold k of 6)" into the prompt
    row_group_size: int = 64      # BlockShuffleSampler shuffles at this granularity


# ── source metadata ───────────────────────────────────────────────────────────
@dataclass
class EpisodeSpec:
    season: str
    episode_index: int
    length: int
    row_from: int                       # index into the season's concatenated data parquet
    row_to: int                         # end-exclusive
    video: Dict[str, Tuple[int, int, int]]   # video_key -> (chunk_index, file_index, start_frame)


def _season_root(src_root: str, season: str) -> str:
    """Accept either `<src>/<season>/lerobot3.0` or a direct lerobot3.0 path."""
    direct = os.path.join(src_root, season, "lerobot3.0")
    if os.path.isdir(direct):
        return direct
    if os.path.isdir(os.path.join(src_root, "meta")):
        return src_root
    raise FileNotFoundError(f"no lerobot3.0 tree for {season} under {src_root}")


def read_episode_specs(root: str, season: str) -> List[EpisodeSpec]:
    """Episode boundaries + the per-stream video file map.

    Video packing is non-uniform across streams (head_* hold ~2 episodes per
    file, tactile_deform can hold a whole season, wrist_* hold one), so the
    (chunk, file, start-frame) triple has to come from the episodes table rather
    than being inferred from filenames.
    """
    files = sorted(glob.glob(os.path.join(root, "meta", "episodes", "**", "*.parquet"),
                             recursive=True))
    if not files:
        raise FileNotFoundError(f"{season}: no meta/episodes/**/*.parquet under {root}")

    specs: List[EpisodeSpec] = []
    for path in files:
        table = pq.read_table(path)
        cols = set(table.schema.names)
        for i in range(table.num_rows):
            video = {}
            for key in VIDEO_KEYS:
                ck, fk = f"videos/{key}/chunk_index", f"videos/{key}/file_index"
                tk = f"videos/{key}/from_timestamp"
                if not {ck, fk, tk} <= cols:
                    raise KeyError(f"{season}: episodes table lacks columns for {key}")
                start_frame = int(round(float(table[tk][i].as_py()) * SRC_FPS))
                video[key] = (int(table[ck][i].as_py()), int(table[fk][i].as_py()), start_frame)
            specs.append(EpisodeSpec(
                season=season,
                episode_index=int(table["episode_index"][i].as_py()),
                length=int(table["length"][i].as_py()),
                row_from=int(table["dataset_from_index"][i].as_py()),
                row_to=int(table["dataset_to_index"][i].as_py()),
                video=video,
            ))
    specs.sort(key=lambda s: s.episode_index)
    for spec in specs:
        if spec.row_to - spec.row_from != spec.length:
            raise ValueError(
                f"{season} ep{spec.episode_index}: length {spec.length} disagrees with "
                f"row range {spec.row_from}:{spec.row_to}")
    return specs


def read_season_arrays(root: str, season: str) -> Dict[str, np.ndarray]:
    """Concatenated state / action / tactile for the whole season, in row order."""
    files = sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"{season}: no data/**/*.parquet under {root}")
    wanted = ["observation.state", "action", "observation.tactile"]
    parts = defaultdict(list)
    for path in files:
        table = pq.read_table(path, columns=wanted)
        for key in wanted:
            parts[key].append(np.asarray(table[key].to_pylist(), dtype=np.float32))
    out = {key: np.concatenate(parts[key], axis=0) for key in wanted}
    if out["observation.state"].shape[1] != ACTION_DIM:
        raise ValueError(f"{season}: state is {out['observation.state'].shape}, expected [:,65]")
    if out["observation.tactile"].shape[1] != F6_FINGERS * F6_PER_FINGER:
        raise ValueError(f"{season}: tactile is {out['observation.tactile'].shape}, expected [:,60]")
    return out


# ── video decoding ────────────────────────────────────────────────────────────
def _select_expr(ranges: Sequence[Tuple[int, int, int]]) -> str:
    """ffmpeg `select` predicate picking every `stride`-th frame of each range.

    `ranges` is a list of (start_frame, end_frame_inclusive, stride).  Terms are
    summed because `select` treats any non-zero value as "keep", and the ranges
    are disjoint so at most one term fires per frame.
    """
    terms = []
    for start, end, stride in ranges:
        if stride == 1:
            terms.append(f"(gte(n\\,{start})*lte(n\\,{end}))")
        else:
            terms.append(f"(gte(n\\,{start})*lte(n\\,{end})*not(mod(n-{start}\\,{stride})))")
    return "+".join(terms)


def _split_jpegs(blob: bytes, expected: int, context: str) -> List[bytes]:
    """Split ffmpeg's concatenated MJPEG stream into individual JPEGs.

    Scanning for SOI is safe here: inside a JPEG's entropy-coded data every 0xFF
    is byte-stuffed as FF 00, and ffmpeg's mjpeg encoder writes no embedded
    thumbnails, so FF D8 FF cannot legitimately occur mid-image.  The count
    assertion below is the guard in case that ever stops holding.
    """
    offsets = []
    pos = blob.find(JPEG_SOI)
    while pos != -1:
        offsets.append(pos)
        pos = blob.find(JPEG_SOI, pos + 3)
    if len(offsets) != expected:
        raise RuntimeError(
            f"{context}: decoded {len(offsets)} JPEGs but expected {expected} "
            f"({len(blob)} bytes) — frame indexing is off")
    offsets.append(len(blob))
    return [blob[offsets[i]:offsets[i + 1]] for i in range(expected)]


def decode_frames(
    video_path: str,
    ranges: Sequence[Tuple[int, int, int]],
    expected: int,
    *,
    scale: Optional[int],
    quality: int,
) -> List[bytes]:
    """One ffmpeg pass over `video_path`, returning JPEG bytes for the selected frames.

    ffmpeg does the select, the resize and the JPEG encode, so raw frames never
    cross into Python — the whole conversion is bounded by video decode.  The
    release ships h264 (not AV1): measured on this box, 480x480 RGB decodes at
    ~1950 fps and the 1200x480 deform strip at ~3200 fps.  `accel.install()`
    reroutes the RGB streams through NVDEC for ~2.6x; see that module for why
    the deform strip stays here.
    """
    vf = f"select='{_select_expr(ranges)}'"
    if scale:
        vf += f",scale={scale}:{scale}:flags=lanczos"
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error",
        "-i", video_path,
        "-vf", vf,
        "-vsync", "0",
        "-f", "image2pipe", "-c:v", "mjpeg", "-q:v", str(quality),
        "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed on {video_path}: {proc.stderr.decode('utf-8', 'replace')[:2000]}")
    return _split_jpegs(proc.stdout, expected, os.path.basename(video_path))


def _video_path(root: str, key: str, chunk_index: int, file_index: int) -> str:
    return os.path.join(root, "videos", key,
                        f"chunk-{chunk_index:03d}", f"file-{file_index:03d}.mp4")


class FrameSource:
    """Lazily decodes video files and hands out per-episode JPEG blobs.

    Two constraints pull in opposite directions.  Decoding a file more than once
    is wasteful (a season's deform stream can hold every episode, and it is the
    most expensive to decode at 1200x480), but buffering every episode's blobs
    before writing anything costs gigabytes on a large season.

    Episodes are contiguous within a video file, so processing them in order
    lets us decode each file exactly once and drop it as soon as no remaining
    episode needs it.  Peak memory is then ~4 files (one per stream) rather than
    the whole season.
    """

    def __init__(self, root: str, specs: Sequence[EpisodeSpec],
                 sample_offsets: Dict[int, np.ndarray], cfg: PrepConfig):
        self.root = root
        self.cfg = cfg
        self.offsets = sample_offsets
        self.specs = list(specs)
        self.by_index = {spec.episode_index: spec for spec in self.specs}

        # file -> episodes living in it, in episode order
        self.members: Dict[Tuple[str, int, int], List[int]] = defaultdict(list)
        for spec in self.specs:
            for key in VIDEO_KEYS:
                chunk_index, file_index, _ = spec.video[key]
                self.members[(key, chunk_index, file_index)].append(spec.episode_index)

        self._cache: Dict[Tuple[str, int, int], Dict[int, List[bytes]]] = {}
        self._pending: Dict[Tuple[str, int, int], set] = {
            file_id: set(eps) for file_id, eps in self.members.items()}
        self.decoded_files = 0

    def _decode_file(self, file_id: Tuple[str, int, int]) -> Dict[int, List[bytes]]:
        key, chunk_index, file_index = file_id
        episodes = sorted(self.members[file_id],
                          key=lambda e: self.by_index[e].video[key][2])
        ranges, counts = [], []
        for episode in episodes:
            spec = self.by_index[episode]
            start = spec.video[key][2]
            offsets = self.offsets[episode]
            # Offsets are a uniform stride grid anchored at the episode start,
            # so one (start, end, stride) term reproduces them exactly.
            ranges.append((start + int(offsets[0]), start + int(offsets[-1]),
                           self.cfg.sample_stride))
            counts.append(len(offsets))

        is_deform = key.endswith("tactile_deform")
        blobs = decode_frames(
            _video_path(self.root, key, chunk_index, file_index),
            ranges, sum(counts),
            scale=None if is_deform else self.cfg.image_size,
            quality=self.cfg.deform_quality if is_deform else self.cfg.rgb_quality,
        )
        self.decoded_files += 1
        out, pos = {}, 0
        for episode, count in zip(episodes, counts):
            out[episode] = blobs[pos:pos + count]
            pos += count
        return out

    def take(self, episode_index: int) -> Dict[str, List[bytes]]:
        """All four streams for one episode; frees any file it was the last user of."""
        spec = self.by_index[episode_index]
        frames = {}
        for key in VIDEO_KEYS:
            chunk_index, file_index, _ = spec.video[key]
            file_id = (key, chunk_index, file_index)
            if file_id not in self._cache:
                self._cache[file_id] = self._decode_file(file_id)
            frames[VIDEO_KEY_TO_COLUMN[key]] = self._cache[file_id][episode_index]
            self._pending[file_id].discard(episode_index)
            if not self._pending[file_id]:
                del self._cache[file_id]
        return frames


# ── row assembly ──────────────────────────────────────────────────────────────
def build_episode_rows(
    spec: EpisodeSpec,
    arrays: Dict[str, np.ndarray],
    offsets: np.ndarray,
    frames: Dict[str, List[bytes]],
    cfg: PrepConfig,
) -> Dict[str, object]:
    state_all = arrays["observation.state"][spec.row_from:spec.row_to]
    action_all = arrays["action"][spec.row_from:spec.row_to]
    tactile_all = arrays["observation.tactile"][spec.row_from:spec.row_to]
    n = spec.length
    last = n - 1

    # Chunk targets: action[t + k*chunk_stride], clamped at the episode end so
    # the tail of an episode degrades to "hold the final commanded pose".
    # All-absolute: no delta, no anchor -- the raw commanded joint values are
    # the target, exactly as stored in the dataset's own `action` column.
    k_offsets = np.arange(cfg.action_chunk, dtype=np.int64) * cfg.chunk_stride
    chunk_idx = np.clip(offsets[:, None] + k_offsets[None, :], 0, last)      # [M, 16]
    chunks = action_all[chunk_idx]                                          # [M, 16, 65]

    # F6 history stays on the native 30 Hz grid (the VQ-VAE's training rate),
    # left-padded by repeating the episode's first frame.
    w_offsets = np.arange(cfg.vqvae_window, dtype=np.int64) - (cfg.vqvae_window - 1)
    hist_idx = np.clip(offsets[:, None] + w_offsets[None, :], 0, last)       # [M, 16]
    hist = tactile_all[hist_idx]                                             # [M, 16, 60]

    phase = (offsets.astype(np.float32) / max(1, last))

    # Keep the numeric columns as numpy right up to the Arrow boundary.
    # `[row.tolist() for row in ...]` on a [M, 25, 65] chunk materialises ~3.5M
    # Python floats per episode, which dominated peak RSS.
    rows = {
        "state": state_all[offsets],
        "action_chunk": chunks.reshape(len(offsets), -1),
        "action_abs": action_all[offsets],
        "phase": phase,
        "tacf6_hist": hist.reshape(len(offsets), -1),
    }
    for column in ("head", "wrist_left", "wrist_right", "deform"):
        blobs = frames[column]
        if len(blobs) != len(offsets):
            raise RuntimeError(
                f"{spec.season} ep{spec.episode_index}: {column} has {len(blobs)} frames "
                f"for {len(offsets)} samples")
        rows[column] = blobs
    return rows


_SCHEMA = pa.schema([
    ("state", pa.list_(pa.float32())),
    ("action_chunk", pa.list_(pa.float32())),
    ("action_abs", pa.list_(pa.float32())),
    ("phase", pa.float32()),
    ("tacf6_hist", pa.list_(pa.float32())),
    ("head", pa.binary()),
    ("wrist_left", pa.binary()),
    ("wrist_right", pa.binary()),
    ("deform", pa.binary()),
])


def _list_column(values: np.ndarray) -> pa.Array:
    """[M, D] float32 -> Arrow list<float32> without going through Python floats."""
    n_rows, width = values.shape
    offsets = pa.array(np.arange(n_rows + 1, dtype=np.int32) * width, type=pa.int32())
    flat = pa.array(np.ascontiguousarray(values, dtype=np.float32).reshape(-1),
                    type=pa.float32())
    return pa.ListArray.from_arrays(offsets, flat)


def write_episode_parquet(path: str, rows: Dict[str, object], cfg: PrepConfig) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    columns = {}
    for name in _SCHEMA.names:
        value = rows[name]
        if name == "phase":
            columns[name] = pa.array(np.asarray(value, dtype=np.float32), type=pa.float32())
        elif isinstance(value, np.ndarray):
            columns[name] = _list_column(value)
        else:
            columns[name] = pa.array(value, type=pa.binary())
    table = pa.table(columns, schema=_SCHEMA)
    # Write-then-rename: `prepare_season` treats an existing file as already
    # converted, so a run killed mid-write must not leave a truncated parquet
    # that the next run would silently accept.
    tmp = path + ".tmp"
    pq.write_table(table, tmp, compression="zstd", compression_level=3,
                   row_group_size=cfg.row_group_size, use_dictionary=False)
    os.replace(tmp, path)
    return table.num_rows


# ── per-season driver ─────────────────────────────────────────────────────────
def prepare_season(season: str, src_root: str, out_root: str, cfg: PrepConfig,
                   overwrite: bool = False) -> List[dict]:
    """Convert one season; returns its episode entries for `meta/dataset.json`."""
    root = _season_root(src_root, season)
    specs = read_episode_specs(root, season)

    entries, pending = [], []
    for spec in specs:
        rel = os.path.join("data", season, f"ep{spec.episode_index:03d}.parquet")
        dest = os.path.join(out_root, rel)
        offsets = np.arange(0, spec.length, cfg.sample_stride, dtype=np.int64)
        if len(offsets) == 0:
            continue
        entry = {
            "file": rel,
            "season": season,
            "episode_index": spec.episode_index,
            "source_frames": spec.length,
            "n_samples": int(len(offsets)),
            "instruction": cfg.instruction,
            "n_phases": cfg.n_phases,
        }
        entries.append(entry)
        if overwrite or not os.path.exists(dest):
            pending.append((spec, offsets, dest, entry))

    if not pending:
        logger.info("[prep] %s: already converted (%d episodes)", season, len(entries))
        return entries

    arrays = read_season_arrays(root, season)
    total_rows = arrays["observation.state"].shape[0]
    if specs[-1].row_to != total_rows:
        raise ValueError(
            f"{season}: episodes cover {specs[-1].row_to} rows but data has {total_rows}")

    source = FrameSource(root, [p[0] for p in pending],
                         {p[0].episode_index: p[1] for p in pending}, cfg)
    for spec, offsets, dest, entry in pending:
        rows = build_episode_rows(spec, arrays, offsets,
                                  source.take(spec.episode_index), cfg)
        written = write_episode_parquet(dest, rows, cfg)
        del rows
        if written != entry["n_samples"]:
            raise RuntimeError(f"{season} ep{spec.episode_index}: wrote {written} rows, "
                               f"expected {entry['n_samples']}")
    logger.info("[prep] %s: %d episodes, %d samples (%d video files decoded)",
                season, len(entries), sum(e["n_samples"] for e in entries),
                source.decoded_files)
    return entries


def season_already_done(out_root: str, season: str) -> bool:
    """True if a previous run already converted every episode of this season.

    Without this, a rerun re-downloads each finished season just to discover its
    parquets exist -- 30 s and ~1 GB of traffic apiece, which across 126 seasons
    is most of a day.  The episode list comes from the checkpointed
    meta/dataset.json, since the season's own metadata is exactly what we are
    trying to avoid fetching.
    """
    index = os.path.join(out_root, "meta", "dataset.json")
    if not os.path.exists(index):
        return False
    try:
        with open(index) as handle:
            entries = json.load(handle).get("episodes", [])
    except (json.JSONDecodeError, OSError):
        return False
    files = [e["file"] for e in entries if e.get("season") == season]
    return bool(files) and all(os.path.exists(os.path.join(out_root, f)) for f in files)


def write_dataset_meta(out_root: str, entries: List[dict], cfg: PrepConfig) -> str:
    os.makedirs(os.path.join(out_root, "meta"), exist_ok=True)
    path = os.path.join(out_root, "meta", "dataset.json")
    payload = {
        "config": asdict(cfg),
        "n_episodes": len(entries),
        "n_samples": sum(e["n_samples"] for e in entries),
        "n_seasons": len({e["season"] for e in entries}),
        "episodes": sorted(entries, key=lambda e: (e["season"], e["episode_index"])),
    }
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=1)
    return path


def _merge_entries(out_root: str, new_entries: List[dict]) -> List[dict]:
    """Union new entries with whatever a previous (partial) run recorded."""
    path = os.path.join(out_root, "meta", "dataset.json")
    merged = {}
    if os.path.exists(path):
        with open(path) as handle:
            for entry in json.load(handle).get("episodes", []):
                merged[entry["file"]] = entry
    for entry in new_entries:
        merged[entry["file"]] = entry
    return list(merged.values())


# ── CLI ───────────────────────────────────────────────────────────────────────
#: Some seasons failed but the split on disk is usable; callers should carry on
#: with stats/verify and surface the gap at the end rather than aborting.
EXIT_PARTIAL = 3


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert Robotic Origami Challenge seasons to origami-flat.")
    parser.add_argument("--out-root", required=True,
                        help="destination root (one per split)")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--revision", choices=["main", "competition-paper-set"],
                        default="main", help="which HF dataset revision's season split to use")
    parser.add_argument("--limit", type=int, default=0,
                        help="use only the first N seasons of the split (0 = all)")
    parser.add_argument("--seasons", nargs="*", default=None,
                        help="explicit season names, overriding --split/--limit")
    parser.add_argument("--cache-root", default="",
                        help="where seasons are downloaded to (default <out-root>/../_src)")
    parser.add_argument("--src-root", default="",
                        help="use already-downloaded seasons here instead of the hub")
    parser.add_argument("--keep-source", action="store_true",
                        help="do not delete a season after converting it")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sample-stride", type=int, default=PrepConfig.sample_stride)
    parser.add_argument("--chunk-stride", type=int, default=PrepConfig.chunk_stride)
    parser.add_argument("--action-chunk", type=int, default=PrepConfig.action_chunk)
    parser.add_argument("--vqvae-window", type=int, default=PrepConfig.vqvae_window)
    parser.add_argument("--image-size", type=int, default=PrepConfig.image_size)
    parser.add_argument("--rgb-quality", type=int, default=PrepConfig.rgb_quality)
    parser.add_argument("--deform-quality", type=int, default=PrepConfig.deform_quality)
    parser.add_argument("--phase-mode", choices=["none", "progress"], default="none")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", "") or None)
    parser.add_argument("--stats", action="store_true",
                        help="also compute meta/norm_stats.json when done")
    parser.add_argument("--stats-subsample", type=int, default=4)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")

    cfg = PrepConfig(
        sample_stride=args.sample_stride,
        chunk_stride=args.chunk_stride,
        action_chunk=args.action_chunk,
        vqvae_window=args.vqvae_window,
        image_size=args.image_size,
        rgb_quality=args.rgb_quality,
        deform_quality=args.deform_quality,
        phase_mode=args.phase_mode,
    )

    seasons = args.seasons if args.seasons else select_seasons(args.split, args.limit, args.revision)
    cache_root = args.cache_root or os.path.join(os.path.dirname(os.path.abspath(args.out_root)),
                                                 "_src")
    os.makedirs(args.out_root, exist_ok=True)

    logger.info("[prep] %s: %d seasons -> %s (stride %d, chunk %d, %dpx)",
                args.split, len(seasons), args.out_root,
                cfg.sample_stride, cfg.action_chunk, cfg.image_size)

    all_entries: List[dict] = []
    failures: List[Tuple[str, str]] = []
    started = time.time()
    for i, season in enumerate(seasons, 1):
        if not args.overwrite and season_already_done(args.out_root, season):
            all_entries += [e for e in _merge_entries(args.out_root, [])
                            if e["season"] == season]
            logger.info("[prep] %d/%d %s: already converted, not fetching",
                        i, len(seasons), season)
            continue
        try:
            if args.src_root:
                src_root, downloaded = args.src_root, False
            else:
                from .fetch import download_season, drop_season, have_season
                downloaded = not have_season(cache_root, season)
                download_season(season, cache_root, token=args.hf_token)
                src_root = cache_root
            entries = prepare_season(season, src_root, args.out_root, cfg,
                                     overwrite=args.overwrite)
            all_entries += entries
            if downloaded and not args.keep_source and not args.src_root:
                from .fetch import drop_season
                drop_season(cache_root, season)
        except Exception as exc:                      # keep going; report at the end
            logger.error("[prep] %s FAILED: %s", season, exc)
            failures.append((season, str(exc)))
            # Most failures are a truncated download; drop it so a rerun
            # refetches rather than hitting the same missing file forever.
            if not args.src_root and not args.keep_source:
                from .fetch import drop_season
                drop_season(cache_root, season)
            continue
        # Checkpoint the index after every season so an interrupted sweep still
        # leaves a usable dataset.
        write_dataset_meta(args.out_root, _merge_entries(args.out_root, all_entries), cfg)
        done = sum(e["n_samples"] for e in all_entries)
        logger.info("[prep] %d/%d seasons | %d samples | %.1f min elapsed",
                    i, len(seasons), done, (time.time() - started) / 60)

    entries = _merge_entries(args.out_root, all_entries)
    path = write_dataset_meta(args.out_root, entries, cfg)
    logger.info("[prep] wrote %s: %d episodes / %d samples / %d seasons",
                path, len(entries), sum(e["n_samples"] for e in entries),
                len({e["season"] for e in entries}))

    if args.stats:
        from .stats import compute_and_write
        compute_and_write(args.out_root, subsample=args.stats_subsample)

    if failures:
        logger.error("[prep] %d season(s) failed:", len(failures))
        for season, message in failures:
            logger.error("         %s: %s", season, message)
        # Per-season failures are already non-fatal to the sweep, so make them
        # non-fatal to the *pipeline* too.  Exiting 1 here meant a run that lost
        # 2 of 101 seasons to a flaky download also lost the val split, the
        # stats fit and both verifies, because the caller runs under `set -e`.
        # EXIT_PARTIAL says "output is written and usable, just incomplete";
        # 1 stays reserved for "nothing came out, do not proceed".
        return EXIT_PARTIAL if entries else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
