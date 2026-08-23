"""q01/q99 normalisation stats for an origami-flat dataset.

Mirrors `utils/lerobot_common.py:165-232` but for the 65-D joint layout.  The
one detail that must not drift: **action stats are per-(step, dim)**, shape
[action_chunk, 65], because `OrigamiDataset._normalize` broadcasts them against
a [B, chunk, 65] tensor.  A [65] action block would broadcast silently and
mis-scale every step of the chunk, which is the single easiest way to get a
model that trains to a plausible-looking loss and predicts nonsense.

Emitted schema (single top-level key; the loader takes `next(iter(...))`):

    {"origami": {
        "action":         {mean,std,min,max,q01,q99,mask}   each [chunk, 65]
        "state":          {...}                             each [65]
        "tactile_f6":     {...}                             each [60]
        "tracking_error": {mean,std,mean_abs,mask}           each [65]
        "num_transitions": int, "num_trajectories": int}}
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import List, Optional, Sequence

import numpy as np
import pyarrow.parquet as pq

from .seasons import ACTION_DIM

logger = logging.getLogger(__name__)

STATS_KEY = "origami"
F6_DIM = 60
NORM_STATS_FILENAME = os.path.join("meta", "norm_stats.json")


# Joints whose q99-q01 spread is below this are treated as frozen and left
# un-normalized.  `_normalize` is `where(mask, scaled, raw)`, so a masked-off dim
# passes through untouched instead of having its float noise stretched to the
# full [-1, 1] range by the `1e-8` guard in the denominator.  Motor j0/j1 are
# exactly this case: they hold to ~1e-4 rad for a whole season.
# Units differ per block, so the thresholds do too.  1e-3 rad is 0.057 deg: a
# joint that moves less than that across the entire training set is frozen (the
# torso's lower_body_joint_1/2 hold to ~4e-4 rad for a whole season).  Tactile
# torque channels legitimately live at 1e-3 N*m, so they get a far lower bar.
MIN_NORM_RANGE_JOINT = 1e-3
MIN_NORM_RANGE_TACTILE = 1e-5


def calculate_stats(data: np.ndarray, mask: Optional[List[bool]] = None,
                    min_range: float = MIN_NORM_RANGE_JOINT) -> dict:
    """Per-dim summary over axis 0, keeping every trailing axis.

    When `mask` is None it is derived from the data: dims with a degenerate
    q01..q99 spread are masked off.  For a per-(step, dim) action block the mask
    is still one flag per *dim* (that is what the loader broadcasts), so a dim is
    kept only if it is non-degenerate at every step of the chunk.
    """
    q01 = np.quantile(data, 0.01, axis=0)
    q99 = np.quantile(data, 0.99, axis=0)
    if mask is None:
        spread = q99 - q01
        while spread.ndim > 1:              # [chunk, dim] -> [dim]
            spread = spread.min(axis=0)
        mask = (spread > min_range).tolist()
    return {
        "mean": np.mean(data, axis=0).tolist(),
        "std": np.std(data, axis=0).tolist(),
        "max": np.max(data, axis=0).tolist(),
        "min": np.min(data, axis=0).tolist(),
        "q01": q01.tolist(),
        "q99": q99.tolist(),
        "mask": mask,
    }


class NormStatsAccumulator:
    """Collects action chunks / states / tactile frames across a whole split."""

    def __init__(self, action_chunk: int, action_dim: int = ACTION_DIM):
        self.action_chunk = action_chunk
        self.action_dim = action_dim
        self._actions: List[np.ndarray] = []      # each [chunk, 65]
        self._states: List[np.ndarray] = []       # each [65]
        self._tactile: List[np.ndarray] = []      # each [60]
        self._tracking: List[np.ndarray] = []     # each [65]
        self.num_trajectories = 0

    def add_episode(self, chunks: np.ndarray, states: np.ndarray,
                    tactile: np.ndarray, abs_actions: np.ndarray) -> None:
        self.num_trajectories += 1
        self._actions.append(chunks)
        self._states.append(states)
        self._tactile.append(tactile)
        # "Tracking error" here is the joint-space analogue of T-Rex's eef
        # version: how far the commanded target sits from the measured state.
        # Only used to parameterise optional state-noise augmentation.
        self._tracking.append(abs_actions - states)

    def assemble(self) -> dict:
        if not self._states:
            raise RuntimeError("no samples accumulated — is the dataset empty?")
        actions = np.concatenate(self._actions, axis=0)       # [M, chunk, 65]
        states = np.concatenate(self._states, axis=0)         # [M, 65]
        tactile = np.concatenate(self._tactile, axis=0)       # [M, 60]
        tracking = np.concatenate(self._tracking, axis=0)     # [M, 65]

        if actions.shape[1:] != (self.action_chunk, self.action_dim):
            raise ValueError(f"action stats source is {actions.shape}, expected "
                             f"[M, {self.action_chunk}, {self.action_dim}]")

        return {STATS_KEY: {
            "action": calculate_stats(actions),
            "state": calculate_stats(states),
            "tactile_f6": calculate_stats(tactile, min_range=MIN_NORM_RANGE_TACTILE),
            "tracking_error": {
                "mean": np.mean(tracking, axis=0).tolist(),
                "std": np.std(tracking, axis=0).tolist(),
                "mean_abs": np.mean(np.abs(tracking), axis=0).tolist(),
                "mask": [True] * self.action_dim,
            },
            "num_transitions": int(states.shape[0]),
            "num_trajectories": int(self.num_trajectories),
            "action_chunk": int(self.action_chunk),
            "action_dim": int(self.action_dim),
        }}


def compute_stats(root: str, subsample: int = 4) -> dict:
    """Scan every episode parquet in `root` (numeric columns only) and accumulate.

    `subsample` keeps every Nth row.  Quantiles over a uniform subsample of
    hundreds of thousands of frames are indistinguishable from the full scan,
    and it keeps peak memory to a few hundred MB rather than tens of GB.
    """
    with open(os.path.join(root, "meta", "dataset.json")) as handle:
        meta = json.load(handle)
    cfg = meta.get("config", {})
    action_chunk = int(cfg.get("action_chunk", 25))
    action_dim = int(cfg.get("action_dim", ACTION_DIM))
    window = int(cfg.get("vqvae_window", 16))

    acc = NormStatsAccumulator(action_chunk, action_dim)
    episodes = meta["episodes"]
    for i, entry in enumerate(episodes, 1):
        table = pq.read_table(
            os.path.join(root, entry["file"]),
            columns=["state", "action_chunk", "action_abs", "tacf6_hist"])
        step = max(1, subsample)
        states = np.asarray(table["state"].to_pylist(), dtype=np.float32)[::step]
        chunks = np.asarray(table["action_chunk"].to_pylist(),
                            dtype=np.float32)[::step].reshape(-1, action_chunk, action_dim)
        abs_actions = np.asarray(table["action_abs"].to_pylist(), dtype=np.float32)[::step]
        # The current tactile frame is the last entry of the history window.
        hist = np.asarray(table["tacf6_hist"].to_pylist(),
                          dtype=np.float32)[::step].reshape(-1, window, F6_DIM)
        acc.add_episode(chunks, states, hist[:, -1, :], abs_actions)
        if i % 25 == 0 or i == len(episodes):
            logger.info("[stats] %d/%d episodes", i, len(episodes))
    return acc.assemble()


def write_stats(root: str, stats: dict) -> str:
    path = os.path.join(root, NORM_STATS_FILENAME)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(stats, handle)
    return path


def compute_and_write(root: str, subsample: int = 4) -> str:
    stats = compute_stats(root, subsample=subsample)
    path = write_stats(root, stats)
    block = stats[STATS_KEY]
    logger.info("[stats] %s | %d transitions / %d trajectories | action q01 %s",
                path, block["num_transitions"], block["num_trajectories"],
                np.shape(block["action"]["q01"]))
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compute origami-flat norm stats.")
    parser.add_argument("--root", required=True, help="split root holding meta/dataset.json")
    parser.add_argument("--subsample", type=int, default=4)
    parser.add_argument("--copy-to", nargs="*", default=[],
                        help="also write the same stats into these roots (e.g. the val split)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    stats = compute_stats(args.root, subsample=args.subsample)
    write_stats(args.root, stats)
    # Val must normalise with the *train* statistics, or its loss is not
    # comparable to the training loss.
    for other in args.copy_to:
        write_stats(other, stats)
        logger.info("[stats] copied train stats into %s", other)
    block = stats[STATS_KEY]
    logger.info("[stats] action q01 shape %s | state q01 shape %s | tactile q01 shape %s",
                np.shape(block["action"]["q01"]), np.shape(block["state"]["q01"]),
                np.shape(block["tactile_f6"]["q01"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
