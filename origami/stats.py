"""Bounded-memory replacement for ``lerobot_common.NormStatsAccumulator``.

REDESIGN_PLAN.md §5.4: the upstream accumulator keeps every frame in RAM (~47 GB for the
action block alone at full corpus scale). ``StreamingNormStats`` is API-compatible
(``add_frame``, ``add_episode_tracking``, ``assemble``, ``write``) but keeps only:
  * exact float64 running (n, sum, sumsq, min, max) per element -- mean/std/min/max are exact;
  * a bounded uniform reservoir (Vitter Algorithm R) per block -- q01/q99 are approximate,
    bounded by gate G8 (< 1% relative error per non-degenerate dim vs the exact upstream
    accumulator on one season).
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np

from utils.lerobot_common import (
    ACTION_CHUNK,
    ACTION_DIM,
    F6_DIM,
    STATS_KEY,
    TRACKING_ERROR_DIM,
    compute_bimanual_tracking_error,
)

DEGENERATE_EPS_ARM = 1e-6     # state / action
DEGENERATE_EPS_TACTILE = 1e-3  # tactile_f6
TRACKING_ERROR_WARMUP_FRAMES = 30  # §5.4: skip the hand-init transient at episode start


class _ExactRunning:
    """Exact streaming (n, sum, sumsq, min, max) over arrays of a fixed trailing shape."""

    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape
        self.n = 0
        self.sum = np.zeros(shape, dtype=np.float64)
        self.sumsq = np.zeros(shape, dtype=np.float64)
        self.min = np.full(shape, np.inf, dtype=np.float64)
        self.max = np.full(shape, -np.inf, dtype=np.float64)

    def add(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        assert x.shape == self.shape, (x.shape, self.shape)
        self.n += 1
        self.sum += x
        self.sumsq += x * x
        self.min = np.minimum(self.min, x)
        self.max = np.maximum(self.max, x)

    def merge(self, other: "_ExactRunning") -> None:
        assert self.shape == other.shape
        self.n += other.n
        self.sum += other.sum
        self.sumsq += other.sumsq
        self.min = np.minimum(self.min, other.min)
        self.max = np.maximum(self.max, other.max)

    def mean(self) -> np.ndarray:
        return self.sum / max(self.n, 1)

    def std(self) -> np.ndarray:
        mean = self.mean()
        var = np.maximum(self.sumsq / max(self.n, 1) - mean * mean, 0.0)
        return np.sqrt(var)


class _Reservoir:
    """Uniform reservoir sample (Vitter Algorithm R) of full-shape arrays."""

    def __init__(self, capacity: int, shape: tuple[int, ...], rng: np.random.Generator):
        self.capacity = capacity
        self.shape = shape
        self.rng = rng
        self.n_seen = 0
        self.items: list[np.ndarray] = []

    def add(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32)
        if len(self.items) < self.capacity:
            self.items.append(x)
        else:
            j = self.rng.integers(0, self.n_seen + 1)
            if j < self.capacity:
                self.items[j] = x
        self.n_seen += 1

    def merge(self, other: "_Reservoir") -> None:
        """Approximate merge: weighted subsample of the union, each item's inclusion weight
        proportional to the real population size it represents (n_seen / len(items)) in its
        own shard -- an unbiased-in-expectation approximation, not an exact Algorithm-R chain."""
        if other.n_seen == 0:
            return
        if self.n_seen == 0:
            self.n_seen = other.n_seen
            self.items = list(other.items)
            return
        pool = self.items + other.items
        w_self = self.n_seen / max(len(self.items), 1)
        w_other = other.n_seen / max(len(other.items), 1)
        weights = np.array([w_self] * len(self.items) + [w_other] * len(other.items), dtype=np.float64)
        weights = weights / weights.sum()
        keep = min(self.capacity, len(pool))
        idx = self.rng.choice(len(pool), size=keep, replace=False, p=weights)
        self.items = [pool[i] for i in idx]
        self.n_seen += other.n_seen

    def array(self) -> np.ndarray:
        if not self.items:
            return np.zeros((0, *self.shape), dtype=np.float32)
        return np.stack(self.items, axis=0)


def _calculate_stats(exact: _ExactRunning, reservoir: _Reservoir, mask: list[bool] | None = None) -> dict:
    sample = reservoir.array()
    if sample.shape[0] == 0:
        q01 = np.zeros(exact.shape)
        q99 = np.zeros(exact.shape)
    else:
        q01 = np.quantile(sample, 0.01, axis=0)
        q99 = np.quantile(sample, 0.99, axis=0)
    if mask is None:
        mask = [True] * exact.shape[-1]
    return {
        "mean": exact.mean().tolist(),
        "std": exact.std().tolist(),
        "max": np.where(np.isfinite(exact.max), exact.max, 0.0).tolist(),
        "min": np.where(np.isfinite(exact.min), exact.min, 0.0).tolist(),
        "q01": q01.tolist(),
        "q99": q99.tolist(),
        "mask": mask,
    }


def _degenerate_mask(q01: np.ndarray, q99: np.ndarray, eps: float) -> list[bool]:
    return ((q99 - q01) > eps).tolist()


class StreamingNormStats:
    def __init__(self, reservoir: int = 200_000, seed: int = 0):
        self.reservoir_size = reservoir
        self.seed = seed
        rng = np.random.default_rng(seed)
        self._rng = rng
        self._action = _ExactRunning((ACTION_CHUNK, ACTION_DIM))
        self._action_res = _Reservoir(reservoir, (ACTION_CHUNK, ACTION_DIM), rng)
        self._state = _ExactRunning((ACTION_DIM,))
        self._state_res = _Reservoir(reservoir, (ACTION_DIM,), rng)
        self._tac = _ExactRunning((F6_DIM,))
        self._tac_res = _Reservoir(reservoir, (F6_DIM,), rng)
        self._track = _ExactRunning((TRACKING_ERROR_DIM,))
        self._track_res = _Reservoir(reservoir, (TRACKING_ERROR_DIM,), rng)
        self._track_full = _ExactRunning((TRACKING_ERROR_DIM,))
        self._track_full_res = _Reservoir(reservoir, (TRACKING_ERROR_DIM,), rng)
        self.num_traj = 0

    # ── same call signature as NormStatsAccumulator ─────────────────────────
    def add_frame(self, action_chunk, state, tactile_f6=None) -> None:
        action_chunk = np.asarray(action_chunk, dtype=np.float32)
        state = np.asarray(state, dtype=np.float32)
        self._action.add(action_chunk)
        self._action_res.add(action_chunk)
        self._state.add(state)
        self._state_res.add(state)
        if tactile_f6 is not None:
            tac = np.asarray(tactile_f6, dtype=np.float32).reshape(-1)
            self._tac.add(tac)
            self._tac_res.add(tac)

    def add_episode_tracking(self, states_62: np.ndarray, abs_targets_62: np.ndarray) -> None:
        self.num_traj += 1
        n = len(states_62)
        for t in range(1, n):
            err = compute_bimanual_tracking_error(states_62[t], abs_targets_62[t - 1])
            self._track_full.add(err)
            self._track_full_res.add(err)
            if t >= TRACKING_ERROR_WARMUP_FRAMES:
                self._track.add(err)
                self._track_res.add(err)

    def add_tracking_errors(self, errors_56: np.ndarray) -> None:
        for e in np.asarray(errors_56, dtype=np.float32):
            self._track.add(e)
            self._track_res.add(e)
            self._track_full.add(e)
            self._track_full_res.add(e)

    # ── shard merging ────────────────────────────────────────────────────────
    def merge(self, other: "StreamingNormStats") -> None:
        self._action.merge(other._action)
        self._action_res.merge(other._action_res)
        self._state.merge(other._state)
        self._state_res.merge(other._state_res)
        self._tac.merge(other._tac)
        self._tac_res.merge(other._tac_res)
        self._track.merge(other._track)
        self._track_res.merge(other._track_res)
        self._track_full.merge(other._track_full)
        self._track_full_res.merge(other._track_full_res)
        self.num_traj += other.num_traj

    # ── output ───────────────────────────────────────────────────────────────
    def assemble(self) -> dict:
        action_stats = _calculate_stats(self._action, self._action_res)
        state_stats = _calculate_stats(self._state, self._state_res)
        action_mask = _degenerate_mask(
            np.array(action_stats["q01"]), np.array(action_stats["q99"]), DEGENERATE_EPS_ARM
        )
        state_mask = _degenerate_mask(
            np.array(state_stats["q01"]), np.array(state_stats["q99"]), DEGENERATE_EPS_ARM
        )
        action_stats["mask"] = action_mask
        state_stats["mask"] = state_mask

        block = {
            "action": action_stats,
            "state": state_stats,
            "num_transitions": int(self._state.n),
            "num_trajectories": int(self.num_traj),
        }
        if self._tac.n > 0:
            tac_stats = _calculate_stats(self._tac, self._tac_res)
            tac_stats["mask"] = _degenerate_mask(
                np.array(tac_stats["q01"]), np.array(tac_stats["q99"]), DEGENERATE_EPS_TACTILE
            )
            block["tactile_f6"] = tac_stats
        if self._track.n > 0:
            # mean_abs needs |x|, which _ExactRunning (signed sum) can't give; approximate from
            # the reservoir sample, consistent with how q01/q99 are already approximated.
            sample = self._track_res.array()
            mean_abs = (
                np.mean(np.abs(sample), axis=0).tolist()
                if sample.shape[0] > 0
                else [0.0] * TRACKING_ERROR_DIM
            )
            block["tracking_error"] = {
                "mean": self._track.mean().tolist(),
                "std": self._track.std().tolist(),
                "mean_abs": mean_abs,
                "mask": [True] * TRACKING_ERROR_DIM,
            }
        if self._track_full.n > 0:
            sample_full = self._track_full_res.array()
            block["tracking_error_full"] = {
                "mean": self._track_full.mean().tolist(),
                "std": self._track_full.std().tolist(),
                "mean_abs": (
                    np.mean(np.abs(sample_full), axis=0).tolist()
                    if sample_full.shape[0] > 0
                    else [0.0] * TRACKING_ERROR_DIM
                ),
                "mask": [True] * TRACKING_ERROR_DIM,
            }
        return {STATS_KEY: block}

    def write(self, dataset_root: str) -> str:
        out = self.assemble()
        path = os.path.join(dataset_root, "meta", "trex_norm_stats.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        return path

    # ── shard checkpointing ──────────────────────────────────────────────────
    def dump(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "StreamingNormStats":
        with open(path, "rb") as f:
            return pickle.load(f)
