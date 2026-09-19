"""OrigamiDataset — Robotic Origami Challenge data in T-Rex's batch contract.

Drop-in for `SftDataset` / `TRexLeRobotDataset`: `collate_fn` returns the exact
same keys the cascaded-flow training loop in `scripts/train.py` consumes, so the
model and the loss are untouched.

Why a third loader instead of the LeRobot path
----------------------------------------------
The origami release ships one *concatenated* mp4 per camera per season and one
1200x480 mp4 holding all ten fingertip deform maps.  Going through LeRobot v3.0
would mean (a) re-encoding 10 separate per-finger videos, (b) 13 random-access
video seeks per training sample, and (c) keeping the dataset at the source
30 Hz because the F6 history and the action chunk are pulled with
`delta_timestamps`.  The "origami-flat" format prepared by `trex_origami/`
instead bakes the chunk and the 30 Hz F6 window into each row and stores the
frames as JPEG blobs, which decouples the *sample* rate from the *action* rate
and turns one sample into one parquet row read + 4 JPEG decodes.

Action target
-------------
All 65 dims are absolute joint radians -- no delta, no anchoring. `action` is
exactly the dataset's own `action_chunk`; a handful of dims (frozen/near-static
joints) get held at the measured state instead of predicted, per
`frozen_action_dims`/`clamp_frozen_absolute` below, driven by an explicit dim
list rather than an anchor spec.

Batch contract (identical to TRexLeRobotDataset.collate_fn):
    input_ids, attention_mask, pixel_values, image_grid_thw, n_slow_images,
    noisy_actions, target, timesteps, norm_actions, tactile_f6s,
    tactile_deforms, tactile_f6s_delayed, tactile_deforms_delayed,
    tactile_codes, tactile_f6_history, time_r, eps_r, state_raw, torque_raw,
    flare_pixel_values, flare_grid_thw
"""
from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

# The origami robot is commanded in 65-D joint space, not T-Rex's eef-62:
# `observation.state.tcp` is identically zero in the release, and the
# competition wire contract is float32[25, 65] absolute radians.  Resuming a
# 62-dim checkpoint therefore re-initialises x_embedder / final_layer /
# final_layer_tactile / state_embedder (train.py drops shape-mismatched keys);
# everything else — the MoT backbone, the tactile expert, the VQ-VAE and the
# deform encoder — transfers unchanged.
ACTION_DIM = 65
ACTION_CHUNK = 25          # == the kit's action_horizon (0.83 s at 30 Hz)
N_FINGERS = 10
F6_PER_FINGER = 6
F6_DIM = 60
DEFORM_TILE = 240
DEFORM_COLS = 5
DEFORM_ROWS = 2

# (name, start, end) of the 65-D joint groups, end-exclusive.  Mirrors
# `participant_local_evaluator/contract.py:88-94`.
JOINT_GROUPS = (
    ("left_arm", 0, 7),
    ("left_hand", 7, 29),
    ("right_arm", 29, 36),
    ("right_hand", 36, 58),
    ("motor", 58, 65),
)


def _normalize(values, mask, vmin, vmax):
    return np.where(mask, np.clip(2 * (values - vmin) / (vmax - vmin + 1e-8) - 1, -1, 1),
                    values)


def denormalize(norm_values, mask, vmin, vmax):
    """Inverse of `_normalize` (shared with the evaluator)."""
    return np.where(mask, (norm_values + 1) / 2 * (vmax - vmin + 1e-8) + vmin,
                    norm_values)


def frozen_action_dims(mask) -> np.ndarray:
    """Indices of the action dims the norm-stats declared frozen.

    `trex_origami.stats.calculate_stats` masks off any dim whose q01..q99 spread
    is below `MIN_NORM_RANGE_JOINT`, and `_normalize` then passes those dims
    through un-scaled.  The flow head still emits *something* on them -- with a
    target of ~0 and a unit-variance noise prior, its job there is to cancel its
    own input noise -- so what reaches the wire is whatever it failed to cancel,
    in raw radians rather than in [-1, 1].  On the pilot split that is the torso
    `lower_body_joint_1/2`: constant to ~4e-4 rad in the data, +-0.3 rad out of
    the policy, and 8% of the checkpoint's total per-joint MAE.

    Derived from the mask rather than hard-coded, so a split where the torso
    *does* move simply reports no frozen dims and nothing is clamped.
    """
    return np.where(~np.asarray(mask, dtype=bool))[0]


def clamp_frozen_absolute(absolute, mask, state):
    """Hold the measured position on the frozen dims of an *absolute* chunk.

    `absolute` is [..., T, D] raw radians, `state` is [..., D]. Commanding
    `state[j]` for the whole chunk is what the teleoperator did on those dims
    for every frame of every season.
    """
    dims = frozen_action_dims(mask)
    if dims.size == 0:
        return absolute
    out = np.array(absolute, copy=True)
    out[..., dims] = np.asarray(state)[..., None, dims]
    return out


def split_deform_strip(arr: np.ndarray) -> np.ndarray:
    """[480, 1200] -> [10, 240, 240], ordered left thumb..little then right."""
    t = arr.reshape(DEFORM_ROWS, DEFORM_TILE, DEFORM_COLS, DEFORM_TILE)
    return t.transpose(0, 2, 1, 3).reshape(N_FINGERS, DEFORM_TILE, DEFORM_TILE)


def add_joint_state_noise(state, te_mean, te_std, action_dim):
    """Joint-space state-noise augmentation.

    T-Rex's `add_tracking_error_noise` treats state[3:9] / state[34:40] as a 6-D
    rotation and perturbs it through Rodrigues; every slot here is a joint
    angle, so that would scramble them.  Perturb additively per dim instead,
    with the per-dim tracking-error statistics the prep pipeline measured
    (`action[t] - state[t]`), so the noise has the magnitude of the real
    servo lag the policy will see at deployment.
    """
    noisy = np.asarray(state, dtype=np.float32).copy()
    d = min(action_dim, noisy.shape[0], len(te_mean), len(te_std))
    noisy[:d] += np.random.normal(te_mean[:d], te_std[:d]).astype(np.float32)
    return noisy


class BlockShuffleSampler(torch.utils.data.Sampler):
    """Shuffle at parquet row-group granularity, then shuffle inside a pool.

    Random access into a JPEG-blob parquet costs one row-group read (~64 rows).
    Drawing a window of `pool_groups` row groups from *different* episodes and
    emitting a permutation of their rows gives near-perfect cache hits while
    still mixing episodes within every batch.  `pool_groups=32` with
    `row_group=64` shuffles inside a 2048-sample window spanning 32 sites.
    """

    def __init__(self, dataset, pool_groups: int = 32, seed: int = 0):
        self.ds = dataset
        self.pool_groups = max(1, pool_groups)
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.ds)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + 1000 * self.epoch)
        groups = list(self.ds.row_groups)          # [(ep, rg, [global idx ...])]
        rng.shuffle(groups)
        for i in range(0, len(groups), self.pool_groups):
            pool = [g for grp in groups[i:i + self.pool_groups] for g in grp[2]]
            rng.shuffle(pool)
            yield from pool


class OrigamiDataset(torch.utils.data.Dataset):
    def __init__(self, config, processor, accelerator, root: Optional[str] = None,
                 _stats: Optional[dict] = None, _quiet: bool = False):
        import pyarrow.parquet as pq
        self.pq = pq
        self.config = config
        self.processor = processor
        self.accelerator = accelerator
        self.root = root or config.origami_root

        with open(os.path.join(self.root, "meta", "dataset.json")) as f:
            self.meta = json.load(f)
        self.episodes = self.meta["episodes"]
        max_episodes = int(getattr(config, "max_episodes", 0) or 0)
        if max_episodes > 0:
            self.episodes = self.episodes[:max_episodes]
        cfg = self.meta.get("config", {})
        self.sample_stride = int(cfg.get("sample_stride", 1))
        self.chunk_stride = int(cfg.get("chunk_stride", 1))
        self.src_fps = 30.0
        self.sample_fps = self.src_fps / max(1, self.sample_stride)

        # ── flags (mirror SftDataset / TRexLeRobotDataset) ──
        g = lambda k, d=0: getattr(config, k, d)
        self.image_size = tuple(config.image_size) if g("image_size", None) else None
        self.use_flare = bool(g("use_flare", 0))
        self.flare_weight = float(g("flare_loss_weight", 0.5))
        self.n_flare_steps = int(g("n_flare_steps", 0)) if self.use_flare else 0
        self.flare_stride = int(g("flare_frame_stride", 1))
        self.bake_flare = self.use_flare and self.flare_weight > 0 and self.n_flare_steps > 0
        self.use_tactile_vec = bool(g("use_tactile_vec", 0))
        self.use_tactile_deform = bool(g("use_tactile_deform", 0))
        self.use_tactile_vqvae = bool(g("use_tactile_vqvae", 0))
        self.use_robot_state = bool(g("use_robot_state", 0))
        self.use_torque = bool(g("use_torque", 0))
        self.vqvae_window = int(g("vqvae_window", 16))
        # Cross-timestep memory (dev/memory): past same-episode rows, split
        # along the same slow(head+text)/fast(wrist+action+tactile) boundary
        # `collate_fn` already uses for the live row (see split_slow_fast_embeds
        # in modeling_vla.py). Empty list / 0 = off, byte-identical to current
        # behavior.
        #
        # Slow (context) memory uses an EXPONENTIAL lookback in *seconds*, not
        # a fixed row count: task-level context is useful much further back
        # than it's useful densely, so a few log-spaced samples (default
        # curr + 0.25/0.5/1/5s ago) cover a 5s reach for the same token cost
        # as 4 consecutive rows would (which at 30Hz only reaches ~0.13s
        # back). Converted to row offsets via self.sample_fps (already
        # computed above). Fast (wrist+action+tactile) memory stays a short
        # *linear* window instead -- it's meant to catch a near-immediate
        # reflex (e.g. a slip), where reaching seconds back is dead weight.
        # Accepts either an actual list of floats (tests, and any caller that
        # already has one) or a CLI-style comma-separated string (train.py's
        # --memory_slow_seconds, always a plain str from argparse) -- parsed
        # here once so every caller can pass whichever form it naturally has.
        _mss = g("memory_slow_seconds", [])
        if isinstance(_mss, str):
            _mss = [float(s) for s in _mss.split(",") if s.strip()]
        self.memory_slow_seconds = list(_mss or [])
        # Fixed ABSOLUTE jitter (seconds), not a percentage of each target --
        # at inference the memory buffer is timestamped and each target is
        # matched to its nearest real tick (ticks aren't perfectly uniform),
        # so the real achieved lookback error is bounded by roughly half the
        # inter-tick interval, which doesn't scale with how far back the
        # target is. Training samples dt_k = nominal_k +/- jitter so the
        # model isn't trained on unrealistically exact offsets. 0 = off
        # (exact offsets every time).
        self.memory_slow_jitter_sec = float(g("memory_slow_jitter_sec", 0.0))
        self.memory_fast = int(g("memory_fast", 0))
        self.action_dim = int(g("action_dim", ACTION_DIM))
        self.action_chunk = int(g("action_chunk", ACTION_CHUNK))
        self.state_noise_mode = str(g("state_noise_mode", "none"))
        self.phase_mode = str(g("phase_mode", "") or cfg.get("phase_mode", "none"))
        self.n_phases = int(cfg.get("n_phases", 6))
        self.instruction = cfg.get("instruction", "")
        # An explicit --instruction on the trainer wins over whatever prep baked
        # into meta/dataset.json.  The parquet rows carry no text, so the prompt
        # can be changed without re-prepping a single episode.
        self.instruction_override = str(g("instruction", "") or "")
        self.effective_instruction = (
            self.instruction_override
            or (self.episodes[0].get("instruction") if self.episodes else "")
            or self.instruction)

        # The rows were written with a fixed chunk/dim/window; a mismatch here
        # would only surface much later as a reshape error inside __getitem__,
        # so fail loudly with the actual numbers instead.
        for name, want, got in (("action_chunk", self.action_chunk, cfg.get("action_chunk")),
                                ("action_dim", self.action_dim, cfg.get("action_dim")),
                                ("vqvae_window", self.vqvae_window, cfg.get("vqvae_window"))):
            if got is not None and int(got) != want:
                raise ValueError(
                    f"{self.root} was prepared with {name}={got} but training was launched "
                    f"with --{name} {want}. Re-prepare the data or fix the flag.")

        # ── normalisation ──
        self.stats_data = _stats if _stats is not None else self._load_stats()
        block = self.stats_data[next(iter(self.stats_data))]
        arr = lambda k, s: np.array(block[k][s], dtype=np.float32)
        self.action_mask = np.array(block["action"]["mask"])
        self.action_min, self.action_max = arr("action", "q01"), arr("action", "q99")
        self.frozen_dims = frozen_action_dims(self.action_mask)
        self.state_mask = np.array(block["state"]["mask"])
        self.state_min, self.state_max = arr("state", "q01"), arr("state", "q99")
        self.has_torque = "torque" in block
        if self.use_torque:
            if not self.has_torque:
                raise ValueError(
                    f"{self.root} norm_stats.json has no 'torque' block but "
                    f"--use_torque was set. Recompute stats with a torque "
                    f"entry (trex_origami/stats.py).")
            self.torque_mask = np.array(block["torque"]["mask"])
            self.torque_min, self.torque_max = arr("torque", "q01"), arr("torque", "q99")
        self.has_tactile = "tactile_f6" in block
        if self.has_tactile:
            self.tacf6_mask = np.array(block["tactile_f6"]["mask"])
            self.tacf6_min, self.tacf6_max = arr("tactile_f6", "q01"), arr("tactile_f6", "q99")
        te = block.get("tracking_error", {})
        self.te_mean = np.array(te.get("mean", np.zeros(self.action_dim)), dtype=np.float32)
        self.te_std = np.array(te.get("std", np.zeros(self.action_dim)), dtype=np.float32)

        # ── sample index + row-group map ──
        self.index: List[tuple] = []            # global idx -> (ep_i, local row)
        self.row_groups: List[tuple] = []       # (ep_i, rg, [global idx ...])
        self.ep_rows: List[int] = []
        self._files: Dict[int, object] = {}     # per-process ParquetFile cache
        self._cache: Dict[tuple, object] = {}
        self._cache_order: List[tuple] = []
        self.cache_groups = int(g("origami_cache_groups", 8))
        self._build_index()

        if not _quiet:
            accelerator.print(
                f"[origami] {self.root}: {len(self.index)} samples / "
                f"{len(self.episodes)} episodes / "
                f"{len({e['season'] for e in self.episodes})} seasons | "
                f"sample {self.sample_fps:.1f} Hz, chunk horizon "
                f"{self.action_chunk * self.chunk_stride / self.src_fps:.2f} s | "
                f"flare={'on' if self.bake_flare else 'off'}")
            accelerator.print(
                f"[origami] action target: all-absolute, {self.action_dim} dims | "
                f"frozen dims (held at measured state): {self.frozen_dims.tolist()}")

    # ── index ──────────────────────────────────────────────────────────────
    def _load_stats(self):
        with open(os.path.join(self.root, "meta", "norm_stats.json")) as f:
            return json.load(f)

    def _build_index(self):
        for ei, ep in enumerate(self.episodes):
            path = os.path.join(self.root, ep["file"])
            pf = self.pq.ParquetFile(path)
            n_rows = pf.metadata.num_rows
            base = len(self.index)
            self.ep_rows.append(n_rows)
            off = 0
            for rg in range(pf.metadata.num_row_groups):
                k = pf.metadata.row_group(rg).num_rows
                self.row_groups.append((ei, rg, list(range(base + off, base + off + k))))
                off += k
            self.index += [(ei, r) for r in range(n_rows)]
            pf.close() if hasattr(pf, "close") else None

    def make_sampler(self, seed: int = 0):
        mode = getattr(self.config, "origami_sampler", "block")
        if mode == "random":
            return None
        return BlockShuffleSampler(
            self, pool_groups=int(getattr(self.config, "origami_pool_groups", 32)),
            seed=seed)

    def __len__(self):
        return len(self.index)

    # ── row access (per-worker handles + row-group LRU) ────────────────────
    def _file(self, ei: int):
        key = (os.getpid(), ei)
        f = self._files.get(key)
        if f is None:
            f = self.pq.ParquetFile(os.path.join(self.root, self.episodes[ei]["file"]))
            self._files[key] = f
        return f

    def _row_group_of(self, ei: int, row: int):
        pf = self._file(ei)
        acc = 0
        for rg in range(pf.metadata.num_row_groups):
            k = pf.metadata.row_group(rg).num_rows
            if row < acc + k:
                return rg, row - acc
            acc += k
        raise IndexError(f"row {row} out of range for episode {ei}")

    def _read_row(self, ei: int, row: int) -> dict:
        rg, off = self._row_group_of(ei, row)
        key = (os.getpid(), ei, rg)
        tbl = self._cache.get(key)
        if tbl is None:
            tbl = self._file(ei).read_row_group(rg)
            self._cache[key] = tbl
            self._cache_order.append(key)
            while len(self._cache_order) > self.cache_groups:
                self._cache.pop(self._cache_order.pop(0), None)
        return {k: tbl[k][off] for k in tbl.column_names}

    # ── decode helpers ─────────────────────────────────────────────────────
    def _pil(self, blob) -> PIL.Image.Image:
        import io
        img = PIL.Image.open(io.BytesIO(blob.as_py())).convert("RGB")
        if self.image_size is not None and img.size != self.image_size:
            img = img.resize(self.image_size, PIL.Image.LANCZOS)
        return img

    def _deform(self, blob) -> np.ndarray:
        import io
        img = PIL.Image.open(io.BytesIO(blob.as_py())).convert("L")
        return split_deform_strip(np.asarray(img, dtype=np.float32) / 255.0)

    def _task_text(self, ep: dict, phase: float) -> str:
        base = self.instruction_override or ep.get("instruction") or self.instruction
        mode = self.phase_mode or ep.get("phase_mode", "none")
        if mode != "progress":
            return base
        n = int(ep.get("n_phases", self.n_phases))
        return f"{base} (fold {min(n, 1 + int(phase * n))} of {n})"

    def __getitem__(self, idx):
        ei, row = self.index[idx]
        ep = self.episodes[ei]
        r = self._read_row(ei, row)
        n_rows = self.ep_rows[ei]

        state = np.asarray(r["state"].as_py(), dtype=np.float32)
        chunk = np.asarray(r["action_chunk"].as_py(),
                           dtype=np.float32).reshape(self.action_chunk, self.action_dim)

        action_abs = np.asarray(r["action_abs"].as_py(), dtype=np.float32)
        item = {
            "state": state,
            "action": chunk,
            "action_abs": action_abs,
            # No augmentation exists under all-absolute, so both of these are
            # plain aliases -- kept only so collate_fn's eval-only extras don't
            # KeyError; real semantics land when the eval scripts get rewritten.
            "action_eval": chunk,
            "prev_command": action_abs,
            "task": self._task_text(ep, float(r["phase"].as_py())),
            "head": self._pil(r["head"]),
            "wrist_left": self._pil(r["wrist_left"]),
            "wrist_right": self._pil(r["wrist_right"]),
        }
        if self.use_torque:
            item["torque"] = np.asarray(r["torque"].as_py(), dtype=np.float32)
        if self.use_tactile_vec or self.use_tactile_vqvae:
            item["tacf6_hist"] = np.asarray(
                r["tacf6_hist"].as_py(), dtype=np.float32
            ).reshape(self.vqvae_window, N_FINGERS, F6_PER_FINGER)
        if self.use_tactile_deform:
            item["deform"] = self._deform(r["deform"])
        if self.bake_flare:
            # Future head frames live in the same parquet: FLARE step k is the
            # sample `k * flare_stride` rows ahead (clamped at the episode end,
            # which is what the LeRobot loader's `_is_pad` fallback does too).
            flare = []
            for k in range(self.n_flare_steps):
                nxt = min(row + (k + 1) * self.flare_stride, n_rows - 1)
                rr = r if nxt == row else self._read_row(ei, nxt)
                flare.append(self._pil(rr["head"]))
            item["flare"] = flare
        if self.memory_slow_seconds:
            # Past rows' slow/latent content only (head image + task text),
            # sampled at EXPONENTIAL lookback in seconds (not a fixed row
            # count) -- e.g. curr + 0.25/0.5/1/5s ago, converted to row
            # offsets via self.sample_fps. Left-padded at episode start by
            # clamping to row 0, same convention tacf6_hist's own history
            # window already uses. Sorted furthest-first so the result is
            # oldest -> newest, matching item["flare"]'s ordering convention,
            # regardless of what order the seconds were given in config.
            mem_slow = []
            for dt_nominal in sorted(self.memory_slow_seconds, reverse=True):
                # Fixed absolute jitter (not scaled by dt_nominal) -- matches
                # the nearest-timestamp snap error at inference, which is
                # bounded by ~half the inter-tick interval regardless of how
                # far back the target is. Clamp at 0: jitter must not push a
                # near target negative ("seconds back" can't be negative).
                dt = dt_nominal
                if self.memory_slow_jitter_sec > 0:
                    dt = max(0.0, dt_nominal + random.uniform(
                        -self.memory_slow_jitter_sec, self.memory_slow_jitter_sec))
                offset = max(1, round(dt * self.sample_fps))
                prow = max(row - offset, 0)
                # Real elapsed time of the row actually fetched, NOT the
                # pre-clamp jittered dt -- episode-start padding can clamp
                # prow to 0, at which point the true gap (row - prow) may be
                # much smaller than dt (e.g. requesting 5s back only 1s into
                # an episode). The model-side position shift must reflect
                # what the content actually is, not the unclamped target, or
                # position label and content would disagree.
                dt_actual = (row - prow) / self.sample_fps
                rr = r if prow == row else self._read_row(ei, prow)
                mem_slow.append({
                    "head": self._pil(rr["head"]),
                    "task": self._task_text(ep, float(rr["phase"].as_py())),
                    "dt_actual": dt_actual,
                })
            item["memory_slow"] = mem_slow
        if self.memory_fast > 0:
            # Past rows' fast/action-expert content (wrist cameras + action +
            # tactile) -- these three travel together as one contiguous
            # segment at the live row too (see split_slow_fast_embeds).
            mem_fast = []
            for k in range(self.memory_fast, 0, -1):
                prow = max(row - k, 0)
                rr = r if prow == row else self._read_row(ei, prow)
                entry = {
                    "wrist_left": self._pil(rr["wrist_left"]),
                    "wrist_right": self._pil(rr["wrist_right"]),
                    "action_abs": np.asarray(rr["action_abs"].as_py(), dtype=np.float32),
                }
                if self.use_tactile_vec or self.use_tactile_vqvae:
                    entry["tacf6_hist"] = np.asarray(
                        rr["tacf6_hist"].as_py(), dtype=np.float32
                    ).reshape(self.vqvae_window, N_FINGERS, F6_PER_FINGER)
                mem_fast.append(entry)
            item["memory_fast"] = mem_fast
        return item

    # ── val split ──────────────────────────────────────────────────────────
    def _as_val(self, val):
        """Turn a dataset into a validation view.

        No-op under all-absolute (no target augmentation exists to disable) --
        kept as the hook point should a val-specific override ever be needed.
        """
        return val

    def create_val_split(self, val_ratio=0.05, seed=42):
        """Prefer a separate root (held-out seasons).  Falls back to an
        episode-level split of this root so `--val_ratio` keeps working."""
        val_root = getattr(self.config, "origami_val_root", "") or ""
        if val_root:
            val = self._as_val(OrigamiDataset(
                self.config, self.processor, self.accelerator,
                root=val_root, _stats=self.stats_data, _quiet=True))
            self.accelerator.print(
                f"[origami] val root {val_root}: {len(val)} samples / "
                f"{len(val.episodes)} episodes / "
                f"{len({e['season'] for e in val.episodes})} held-out seasons")
            return val
        import copy
        n_ep = len(self.episodes)
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n_ep)
        n_val = max(1, int(n_ep * val_ratio))
        val_eps = sorted(perm[:n_val].tolist())
        train_eps = sorted(perm[n_val:].tolist())
        val = copy.copy(self)
        val.episodes = [self.episodes[i] for i in val_eps]
        val._files, val._cache, val._cache_order = {}, {}, []
        val.index, val.row_groups, val.ep_rows = [], [], []
        val._build_index()
        self.episodes = [self.episodes[i] for i in train_eps]
        self._files, self._cache, self._cache_order = {}, {}, []
        self.index, self.row_groups, self.ep_rows = [], [], []
        self._build_index()
        self.accelerator.print(f"[origami] episode split: "
                               f"{len(train_eps)} train / {len(val_eps)} val episodes")
        return self._as_val(val)

    # ── collate (parity with TRexLeRobotDataset.collate_fn) ────────────────
    def collate_fn(self, batch: List[Dict]) -> Dict:
        B = len(batch)

        actions = np.stack([b["action"] for b in batch], axis=0)          # [B,16,62]
        norm_actions = torch.tensor(
            _normalize(actions, self.action_mask, self.action_min, self.action_max),
            dtype=torch.bfloat16)
        beta = torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0))
        time = (beta.sample((B,)) * 0.999 + 0.001).to(torch.bfloat16)
        t_ = time[:, None, None]
        noise = torch.randn_like(norm_actions)
        x_t = t_ * noise + (1 - t_) * norm_actions
        u_t = noise - norm_actions
        time_r = (beta.sample((B,)) * 0.999 + 0.001).to(torch.bfloat16)
        eps_r = torch.randn_like(norm_actions)

        norm_tacf6 = None
        f6_hist_t = None
        if "tacf6_hist" in batch[0]:
            hist = np.stack([b["tacf6_hist"] for b in batch], axis=0)     # [B,W,10,6]
            if self.use_tactile_vqvae:
                f6_hist_t = torch.tensor(hist, dtype=torch.float32)       # model normalises
            if self.use_tactile_vec:
                cur = hist[:, -1].reshape(B, -1)
                norm_tacf6 = torch.tensor(
                    _normalize(cur, self.tacf6_mask, self.tacf6_min,
                               self.tacf6_max).reshape(B, -1, 6),
                    dtype=torch.bfloat16)

        deforms = None
        if "deform" in batch[0]:
            d = np.stack([b["deform"] for b in batch], axis=0)            # [B,10,240,240]
            deforms = torch.tensor(d, dtype=torch.float32).unsqueeze(2)   # [B,10,1,240,240]

        state_raw = None
        if self.use_robot_state:
            sl = []
            for b in batch:
                s = b["state"]
                if self.state_noise_mode == "joint":
                    s = add_joint_state_noise(s, self.te_mean, self.te_std, self.action_dim)
                sl.append(torch.tensor(
                    _normalize(s, self.state_mask, self.state_min, self.state_max),
                    dtype=torch.bfloat16))
            state_raw = torch.stack(sl)

        torque_raw = None
        if self.use_torque:
            tl = [torch.tensor(
                    _normalize(b["torque"], self.torque_mask, self.torque_min, self.torque_max),
                    dtype=torch.bfloat16)
                  for b in batch]
            torque_raw = torch.stack(tl)

        # ── images -> Qwen processor: slow = head, fast = [wrist_r, wrist_l] ──
        all_ids, all_pv, all_thw = [], [], []
        n_slow_images = 1
        for b in batch:
            pil_slow = [b["head"]]
            pil_fast = [b["wrist_right"], b["wrist_left"]]
            content = [{"type": "image"} for _ in pil_slow]
            content.append({"type": "text", "text": b.get("task", "")})
            content += [{"type": "image"} for _ in pil_fast]
            text = self.processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False,
                add_generation_prompt=True)
            inp = self.processor(text=text, images=pil_slow + pil_fast,
                                 return_tensors="pt", padding=False)
            all_ids.append(inp.input_ids[0])
            if getattr(inp, "pixel_values", None) is not None:
                all_pv.append(inp.pixel_values)
                all_thw.append(inp.image_grid_thw)

        # ── memory (dev/memory): K per-timestep sequences, each processed
        # through the same apply_chat_template + processor path as the live
        # row above, since each memory row needs its own input_ids to be
        # forwarded separately (one KV-populating forward per past row, per
        # the plan) rather than just embedded as raw pixels the way flare is.
        memory_slow_out = None
        if self.memory_slow_seconds and "memory_slow" in batch[0]:
            memory_slow_out = []
            for k in range(len(self.memory_slow_seconds)):
                k_ids, k_pv, k_thw, k_dt = [], [], [], []
                for b in batch:
                    m = b["memory_slow"][k]
                    k_dt.append(m["dt_actual"])
                    content = [{"type": "image"}, {"type": "text", "text": m.get("task", "")}]
                    text = self.processor.apply_chat_template(
                        [{"role": "user", "content": content}], tokenize=False,
                        add_generation_prompt=True)
                    inp = self.processor(text=text, images=[m["head"]],
                                         return_tensors="pt", padding=False)
                    k_ids.append(inp.input_ids[0])
                    if getattr(inp, "pixel_values", None) is not None:
                        k_pv.append(inp.pixel_values)
                        k_thw.append(inp.image_grid_thw)
                pad_id_k = self.processor.tokenizer.pad_token_id or 0
                max_len_k = max(i.shape[0] for i in k_ids)
                ids_k, ams_k = [], []
                for i in k_ids:
                    pad = max_len_k - i.shape[0]
                    ids_k.append(F.pad(i, (pad, 0), value=pad_id_k))
                    a = torch.ones(max_len_k, dtype=torch.long)
                    if pad > 0:
                        a[:pad] = 0
                    ams_k.append(a)
                memory_slow_out.append({
                    "input_ids": torch.stack(ids_k),
                    "attention_mask": torch.stack(ams_k),
                    "pixel_values": torch.cat(k_pv, dim=0) if k_pv else None,
                    "image_grid_thw": torch.cat(k_thw, dim=0) if k_thw else None,
                    # Real elapsed seconds for the row actually fetched
                    # (post-jitter, post-episode-clamp) -- the model-side
                    # position-id shift must use this, not the nominal
                    # memory_slow_seconds target, since clamping at episode
                    # start can make them disagree.
                    "dt_actual": torch.tensor(k_dt, dtype=torch.float32),
                })

        memory_fast_out = None
        if self.memory_fast > 0 and "memory_fast" in batch[0]:
            memory_fast_out = []
            for k in range(self.memory_fast):
                k_ids, k_pv, k_thw = [], [], []
                for b in batch:
                    m = b["memory_fast"][k]
                    pil_fast_k = [m["wrist_right"], m["wrist_left"]]
                    content = [{"type": "image"} for _ in pil_fast_k]
                    text = self.processor.apply_chat_template(
                        [{"role": "user", "content": content}], tokenize=False,
                        add_generation_prompt=True)
                    inp = self.processor(text=text, images=pil_fast_k,
                                         return_tensors="pt", padding=False)
                    k_ids.append(inp.input_ids[0])
                    if getattr(inp, "pixel_values", None) is not None:
                        k_pv.append(inp.pixel_values)
                        k_thw.append(inp.image_grid_thw)
                pad_id_k = self.processor.tokenizer.pad_token_id or 0
                max_len_k = max(i.shape[0] for i in k_ids)
                ids_k, ams_k = [], []
                for i in k_ids:
                    pad = max_len_k - i.shape[0]
                    ids_k.append(F.pad(i, (pad, 0), value=pad_id_k))
                    a = torch.ones(max_len_k, dtype=torch.long)
                    if pad > 0:
                        a[:pad] = 0
                    ams_k.append(a)
                # Normalized the same way the live tick's flow target
                # (norm_actions, above) is -- x_embedder only ever sees
                # normalized action-space values (the flow variable x_t is
                # literally t*noise + (1-t)*norm_actions), so an unnormalized
                # past action here would be a real train/inference input-
                # distribution mismatch, not just a style inconsistency.
                action_raw = np.stack(
                    [b["memory_fast"][k]["action_abs"] for b in batch], axis=0)
                action_k = torch.tensor(
                    _normalize(action_raw, self.action_mask,
                              self.action_min, self.action_max),
                    dtype=torch.float32)
                entry = {
                    "input_ids": torch.stack(ids_k),
                    "attention_mask": torch.stack(ams_k),
                    "pixel_values": torch.cat(k_pv, dim=0) if k_pv else None,
                    "image_grid_thw": torch.cat(k_thw, dim=0) if k_thw else None,
                    "action_abs": action_k,
                }
                if "tacf6_hist" in batch[0]["memory_fast"][k]:
                    entry["tacf6_hist"] = torch.tensor(
                        np.stack([b["memory_fast"][k]["tacf6_hist"] for b in batch], axis=0),
                        dtype=torch.float32)
                memory_fast_out.append(entry)

        flare_pv = flare_thw = None
        if self.bake_flare and "flare" in batch[0]:
            flare_pil = [img for b in batch for img in b["flare"]]
            finp = self.processor.image_processor(flare_pil, return_tensors="pt")
            flare_pv = finp.pixel_values.to(torch.bfloat16)
            flare_thw = finp.image_grid_thw

        pad_id = self.processor.tokenizer.pad_token_id or 0
        max_len = max(i.shape[0] for i in all_ids)
        ids, ams = [], []
        for i in all_ids:
            pad = max_len - i.shape[0]
            ids.append(F.pad(i, (pad, 0), value=pad_id))
            a = torch.ones(max_len, dtype=torch.long)
            if pad > 0:
                a[:pad] = 0
            ams.append(a)

        return {
            "input_ids": torch.stack(ids),
            "attention_mask": torch.stack(ams),
            "pixel_values": torch.cat(all_pv, dim=0) if all_pv else None,
            "image_grid_thw": torch.cat(all_thw, dim=0) if all_thw else None,
            "n_slow_images": n_slow_images,
            "noisy_actions": x_t,
            "target": u_t,
            "timesteps": time,
            "norm_actions": norm_actions,
            "tactile_f6s": norm_tacf6,
            "tactile_deforms": deforms,
            "tactile_f6s_delayed": norm_tacf6,
            "tactile_deforms_delayed": deforms,
            "tactile_codes": None,
            "tactile_f6_history": f6_hist_t,
            "time_r": time_r,
            "eps_r": eps_r,
            "state_raw": state_raw,
            "torque_raw": torque_raw,
            "flare_pixel_values": flare_pv,
            "flare_grid_thw": flare_thw,
            "memory_slow": memory_slow_out,
            "memory_fast": memory_fast_out,
            # extras used only by the offline evaluator (ignored by train.py)
            "eval_state": torch.tensor(np.stack([b["state"] for b in batch]),
                                       dtype=torch.float32),
            # The stored target -- same as "action" (no augmentation exists),
            # kept as a separate key for the evaluator.
            "eval_action_raw": torch.tensor(
                np.stack([b["action_eval"] for b in batch]), dtype=torch.float32),
            "eval_prev_command": torch.tensor(
                np.stack([b["prev_command"] for b in batch]), dtype=torch.float32),
            "eval_contact": torch.tensor(
                np.stack([_contact_flag(b) for b in batch]), dtype=torch.bool)
            if "tacf6_hist" in batch[0] else None,
        }


def _contact_flag(item, thresh: float = 1.0) -> bool:
    """True if any fingertip carries >`thresh` N at the sample frame."""
    h = item.get("tacf6_hist")
    if h is None:
        return False
    f = np.linalg.norm(h[-1][:, :3], axis=-1)
    return bool((f > thresh).any())
