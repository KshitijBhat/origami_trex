"""origami vs T-Rex-midtrain distribution-shift report. REDESIGN_PLAN.md §11.9 step 2 / §12 step 10b.

Produces one markdown report covering:
  * chunk-delta magnitude per action block (L_trans3/L_rot6d6/L_hand22/R_trans3/R_rot6d6/R_hand22)
  * F6 force/torque magnitude per finger
  * deform-tile occupancy per finger (fraction of pixels showing real contact signal)
  * **G15** — frozen VQ-VAE codebook usage when T-Rex's own ``tacf6_vqvae_{min,max,mask}``
    buffers are applied to origami F6 windows (§11.8): usage histogram, codes used, normalized
    entropy, clamp-saturation fraction
  * frozen-ViT feature statistics: per-channel mean/std of pooled patch tokens for the origami
    head camera, and (when a comparison root is supplied) the same for a sample of
    ``zekaiwang/trex_dataset`` plus the cosine distance between the two dataset means

The first three sections only need the merged dataset root (`meta/trex_norm_stats.json` +
the deform videos) — no checkpoint. G15 and the ViT section need the T-Rex midtrain checkpoint
directory (``training_args.json`` + ``config.json`` + ``model.pt`` + ``processor/``, the format
``scripts/test.py::model_load`` expects) — pass ``--checkpoint``.

This is a read-only diagnostic: it never mutates the checkpoint or the dataset root. The
prescribed fix (§11.8 fix ladder step 1, re-fitting the buffers) lives in ``refit_vqvae_stats.py``
(step 10c), gated on this script's G15 verdict.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# T-Rex on sys.path (mirrors origami/__init__.py).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_TREX_DIR = _REPO_ROOT / "T-Rex"
if str(_TREX_DIR) not in sys.path:
    sys.path.insert(0, str(_TREX_DIR))

from utils.lerobot_common import DEFORM_KEYS, F6_PER_FINGER, N_FINGERS_PER_HAND  # noqa: E402

FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
FINGER_LABELS = tuple(f"L-{n}" for n in FINGER_NAMES) + tuple(f"R-{n}" for n in FINGER_NAMES)
assert len(FINGER_LABELS) == len(DEFORM_KEYS) == 10

# [name, action-dim slice] — REDESIGN_PLAN.md §6 / T-Rex's build_action_chunk layout:
# [dl(3+6), a_l_hnd(22), dr(3+6), a_r_hnd(22)] = 62.
ACTION_BLOCKS = (
    ("L_trans3", slice(0, 3)),
    ("L_rot6d6", slice(3, 9)),
    ("L_hand22", slice(9, 31)),
    ("R_trans3", slice(31, 34)),
    ("R_rot6d6", slice(34, 40)),
    ("R_hand22", slice(40, 62)),
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Chunk-delta magnitude per block (from meta/trex_norm_stats.json — no rescan)
# ─────────────────────────────────────────────────────────────────────────────

def _load_stats_block(root: str) -> dict:
    with open(os.path.join(root, "meta", "trex_norm_stats.json")) as f:
        stats = json.load(f)
    return stats[next(iter(stats))]


def chunk_delta_magnitude(root: str) -> dict:
    """RMS-per-dim magnitude of the baked delta chunk, per block, at chunk steps k=0 and k=-1.

    RMS = sqrt(mean^2 + std^2) per dim (exact given E[X^2] = Var + Mean^2), averaged over the
    dims in the block. Uses the accumulator's exact mean/std (§5.4), not the reservoir
    quantiles, so this is exact -- not subject to G8's approximation bound.
    """
    block = _load_stats_block(root)
    mean = np.array(block["action"]["mean"], dtype=np.float64)   # [16, 62]
    std = np.array(block["action"]["std"], dtype=np.float64)     # [16, 62]
    chunk = mean.shape[0]
    rms = np.sqrt(mean ** 2 + std ** 2)                          # [16, 62]
    out = {}
    for k in (0, chunk - 1):
        out[f"k={k}"] = {name: float(np.mean(rms[k, sl])) for name, sl in ACTION_BLOCKS}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. F6 magnitude per finger (from meta/trex_norm_stats.json)
# ─────────────────────────────────────────────────────────────────────────────

def f6_magnitude_per_finger(root: str) -> dict:
    """Mean |force/torque| RMS per finger, split into the 3 force and 3 torque channels.

    Assumes T-Rex's per-finger F6 channel order is [Fx, Fy, Fz, Tx, Ty, Tz] (§1.2) -- the
    magnitude computed here doesn't depend on that ordering (it's per-finger over all 6), but
    the force/torque sub-split does.
    """
    block = _load_stats_block(root)
    mean = np.array(block["tactile_f6"]["mean"], dtype=np.float64).reshape(N_FINGERS_PER_HAND * 2, F6_PER_FINGER)
    std = np.array(block["tactile_f6"]["std"], dtype=np.float64).reshape(N_FINGERS_PER_HAND * 2, F6_PER_FINGER)
    mask = np.array(block["tactile_f6"]["mask"], dtype=bool).reshape(N_FINGERS_PER_HAND * 2, F6_PER_FINGER)
    rms = np.sqrt(mean ** 2 + std ** 2)
    out = {}
    for i, label in enumerate(FINGER_LABELS):
        out[label] = {
            "force_rms": float(np.mean(rms[i, 0:3])),
            "torque_rms": float(np.mean(rms[i, 3:6])),
            "n_masked_degenerate": int((~mask[i]).sum()),
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. Deform-tile occupancy per finger (real PyAV decode of a small frame sample)
# ─────────────────────────────────────────────────────────────────────────────

def _seek_sample_frames(files: list[str], n_samples: int, fmt: str, seed: int = 0) -> list[np.ndarray]:
    """Sample up to `n_samples` frames (1 random frame each from up to `n_samples` distinct
    files, more files sampled with repeats if there are fewer files than n_samples).

    Uses `container.seek()` to the target frame's approximate timestamp rather than
    sequentially decoding from frame 0 -- with these videos' GOP=2 encoding (§5.2) a seek
    lands within ~1 frame, so this is O(1) per sample instead of O(episode length). Decoding
    every file from frame 0 to a random target (the naive approach) was the actual bottleneck
    found when this was first tried against the real fixture: ~4 minutes for 20 samples/finger
    across 180 real episodes, because a uniform-random target can sit near the end of a
    ~7 700-frame episode.
    """
    import av

    if not files:
        raise FileNotFoundError("no video files given to sample from")
    rng = np.random.RandomState(seed)
    if len(files) >= n_samples:
        chosen = list(rng.choice(files, size=n_samples, replace=False))
    else:
        chosen = [files[i % len(files)] for i in range(n_samples)]
    frames = []
    for path in chosen:
        container = av.open(path)
        stream = container.streams.video[0]
        n_avail = stream.frames or 0
        if n_avail <= 0:
            container.close()
            continue
        target = int(rng.randint(0, n_avail))
        fps = float(stream.average_rate)
        ts = int(target / fps / stream.time_base)
        container.seek(ts, stream=stream, backward=True, any_frame=False)
        got = None
        for f in container.decode(stream):
            if f.pts is None or f.pts >= ts:
                got = f
                break
        if got is not None:
            frames.append(got.to_ndarray(format=fmt))
        container.close()
    return frames


def _sample_deform_frames(root: str, key: str, n_samples: int, seed: int = 0) -> list[np.ndarray]:
    files = sorted(glob.glob(os.path.join(root, "videos", key, "chunk-*", "*.mp4")))
    if not files:
        raise FileNotFoundError(f"no deform videos found for {key} under {root}")
    return _seek_sample_frames(files, n_samples, fmt="gray", seed=seed)


def deform_occupancy(root: str, n_samples: int = 40, active_thresh: int = 8) -> dict:
    """Fraction of pixels per finger that deviate from the tile's own median by more than
    ``active_thresh`` gray levels -- a cheap proxy for "this tile is showing a real contact
    deformation" vs a flat/degenerate sensor. Reports the mean occupancy fraction over the
    sampled frames, per finger.
    """
    out = {}
    for key, label in zip(DEFORM_KEYS, FINGER_LABELS):
        frames = _sample_deform_frames(root, key, n_samples)
        if not frames:
            out[label] = None
            continue
        fracs = []
        for frame in frames:
            med = np.median(frame)
            fracs.append(float(np.mean(np.abs(frame.astype(np.int16) - med) > active_thresh)))
        out[label] = {"mean_occupancy": float(np.mean(fracs)), "n_frames": len(frames)}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. G15 — frozen VQ-VAE codebook health under T-Rex's own tacf6_vqvae_{min,max,mask}
# ─────────────────────────────────────────────────────────────────────────────

def _load_embedded_vqvae(checkpoint_dir: str):
    """Load only the ``tactile_vqvae.*`` submodule + ``tacf6_vqvae_{min,max,mask}`` buffers
    from ``model.pt`` (mmap'd — never materializes the ~8.5 GB full checkpoint) plus the
    ``vqvae_config`` dict from ``training_args.json``. Returns (vqvae_module, min, max, mask).
    """
    import torch

    from tactile_vqvae.models.tactile_vqvae import TactileVQVAE, TactileVQVAEConfig

    with open(os.path.join(checkpoint_dir, "training_args.json")) as f:
        ta = json.load(f)
    vqvae_config = ta["vqvae_config"]

    sd = torch.load(os.path.join(checkpoint_dir, "model.pt"), map_location="cpu",
                     mmap=True, weights_only=True)
    prefix = "tactile_vqvae."
    vq_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    vq = TactileVQVAE(TactileVQVAEConfig.from_dict(vqvae_config))
    missing, unexpected = vq.load_state_dict(vq_sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    vq.eval()
    for p in vq.parameters():
        p.requires_grad = False

    tacf6_min = sd["tacf6_vqvae_min"].clone().float()
    tacf6_max = sd["tacf6_vqvae_max"].clone().float()
    tacf6_mask = sd["tacf6_vqvae_mask"].clone().bool()
    return vq, tacf6_min, tacf6_max, tacf6_mask


def _f6_windows_from_root(root: str, window: int, max_windows: int, seed: int = 0) -> np.ndarray:
    """Build [N, window, 10, 6] raw F6 windows by sliding over each episode's parquet column,
    padding the start of each episode by repeating its first frame (matches the loader's
    intent -- §7.5's `_f6_offsets` clips to `[-(W-1)/fps .. 0]` -- closely enough for a
    ≥200k-window statistical diagnostic; exact boundary handling doesn't matter at this scale).
    """
    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(root, "data", "chunk-*", "*.parquet")))
    rng = np.random.RandomState(seed)
    rng.shuffle(files)
    windows = []
    for path in files:
        table = pq.read_table(path, columns=["observation.tactile_f6"])
        col = table.column("observation.tactile_f6").to_pylist()
        f6 = np.array(col, dtype=np.float32)          # [T, 10, 6]
        if f6.shape[0] == 0:
            continue
        pad = np.repeat(f6[:1], window - 1, axis=0)
        padded = np.concatenate([pad, f6], axis=0)      # [T + window - 1, 10, 6]
        n = f6.shape[0]
        idx = np.arange(window)[None, :] + np.arange(n)[:, None]   # [n, window]
        windows.append(padded[idx])                     # [n, window, 10, 6]
        if sum(w.shape[0] for w in windows) >= max_windows:
            break
    out = np.concatenate(windows, axis=0)
    if out.shape[0] > max_windows:
        sel = rng.choice(out.shape[0], size=max_windows, replace=False)
        out = out[sel]
    return out


def vqvae_code_entropy(root: str, checkpoint_dir: str, n_windows: int = 200_000,
                        batch_size: int = 4096, seed: int = 0) -> dict:
    """**G15**: run T-Rex's frozen tacf6-VQ-VAE (min-max normalize with the CHECKPOINT's own
    buffers, hard-clamp to [-1,1], encode) over ≥ n_windows real origami F6 windows.

    Reports, per hand (the checkpoint's granularity is "finger", so per (hand, finger)):
      * codebook usage histogram (top codes)
      * number of distinct codes used / codebook size
      * normalized entropy H / log(K)  (1.0 = uniform use, 0 = single-code collapse)
      * clamp-saturation fraction: fraction of normalized values that hit exactly -1 or +1
    """
    import torch

    vq, tacf6_min, tacf6_max, tacf6_mask = _load_embedded_vqvae(checkpoint_dir)
    window = vq.cfg.window
    windows = _f6_windows_from_root(root, window=window, max_windows=n_windows, seed=seed)
    n = windows.shape[0]

    denom = (tacf6_max - tacf6_min) + 1e-8
    codebook_size = vq.cfg.codebook_size
    n_hands = 2
    n_fingers = vq.cfg.n_fingers if vq.cfg.granularity == "finger" else 1
    codes_per_slot = np.zeros((n_hands, n_fingers, n), dtype=np.int64) if vq.cfg.granularity == "finger" \
        else np.zeros((n_hands, n), dtype=np.int64)
    sat_count = 0
    total_count = 0

    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = torch.from_numpy(windows[start:start + batch_size]).float()   # [b, W, 10, 6]
            b = batch.shape[0]
            flat = batch.reshape(b, window, 60)
            raw_norm = 2.0 * (flat - tacf6_min) / denom - 1.0
            clamped = torch.clamp(raw_norm, -1.0, 1.0)
            sat_count += int(((raw_norm.abs() > 1.0) & tacf6_mask).sum().item())
            total_count += int(tacf6_mask.sum().item()) * b * window
            normed = torch.where(tacf6_mask, clamped, flat).reshape(b, window, 10, 6)
            for h in range(n_hands):
                wh = normed[:, :, h * 5:(h + 1) * 5, :]
                idx = vq.encode(wh).cpu().numpy()
                if vq.cfg.granularity == "finger":
                    codes_per_slot[h, :, start:start + b] = idx.T if idx.ndim == 2 else idx.reshape(1, b)
                else:
                    codes_per_slot[h, start:start + b] = idx.reshape(-1)

    out = {"n_windows": n, "codebook_size": codebook_size,
           "clamp_saturation_fraction": sat_count / max(total_count, 1),
           "slots": {}}
    slot_iter = (
        [(f"{('L','R')[h]}-{FINGER_NAMES[fi]}", codes_per_slot[h, fi]) for h in range(2) for fi in range(n_fingers)]
        if vq.cfg.granularity == "finger"
        else [(("L", "R")[h], codes_per_slot[h]) for h in range(2)]
    )
    for name, codes in slot_iter:
        counts = np.bincount(codes, minlength=codebook_size).astype(np.float64)
        p = counts / counts.sum()
        p_nz = p[p > 0]
        entropy = float(-(p_nz * np.log(p_nz)).sum())
        out["slots"][name] = {
            "n_codes_used": int((counts > 0).sum()),
            "normalized_entropy": entropy / np.log(codebook_size),
            "top5_code_frac": [float(x) for x in np.sort(p)[::-1][:5]],
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 5. Frozen-ViT feature statistics + cross-dataset cosine distance
# ─────────────────────────────────────────────────────────────────────────────

def _load_visual_tower(checkpoint_dir: str, device: str = "cpu"):
    import torch
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    with open(os.path.join(checkpoint_dir, "config.json")) as f:
        full_cfg = json.load(f)
    vis_cfg = Qwen3VLVisionConfig(**{k: v for k, v in full_cfg["vision_config"].items()
                                     if k != "model_type"})
    visual = Qwen3VLVisionModel(vis_cfg)

    sd = torch.load(os.path.join(checkpoint_dir, "model.pt"), map_location="cpu",
                     mmap=True, weights_only=True)
    prefix = "visual."
    vis_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    missing, unexpected = visual.load_state_dict(vis_sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    visual.eval().to(device=device, dtype=torch.bfloat16)
    for p in visual.parameters():
        p.requires_grad = False
    return visual


def _pooled_vit_features(visual, processor, images: list, device: str = "cpu") -> np.ndarray:
    """[N, hidden] mean-pooled patch-token feature per image (one image at a time — simplest
    correct way to avoid tracking per-image token offsets in the flattened multi-image output).
    """
    import torch

    feats = []
    with torch.no_grad():
        for img in images:
            inp = processor.image_processor([img], return_tensors="pt")
            pixel_values = inp.pixel_values.to(device=device, dtype=torch.bfloat16)
            grid_thw = inp.image_grid_thw.to(device=device)
            out = visual(pixel_values, grid_thw=grid_thw)
            if hasattr(out, "last_hidden_state"):
                hidden = out.last_hidden_state
            elif isinstance(out, (tuple, list)):
                hidden = out[0]
            else:
                hidden = out
            feats.append(hidden.float().mean(dim=0).cpu().numpy())
    return np.stack(feats, axis=0)


def _sample_pil_frames(video_paths: list[str], n_samples: int, seed: int = 0):
    from PIL import Image

    frames = _seek_sample_frames(video_paths, n_samples, fmt="rgb24", seed=seed)
    return [Image.fromarray(f) for f in frames]


def vit_feature_shift(checkpoint_dir: str, origami_root: str,
                       trex_root: Optional[str] = None, n_samples: int = 40,
                       origami_head_key: str = "observation.images.head",
                       trex_head_key: str = "observation.images.head_left",
                       device: str = "cpu") -> dict:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        os.path.join(checkpoint_dir, "processor"), trust_remote_code=True)
    visual = _load_visual_tower(checkpoint_dir, device=device)

    origami_videos = sorted(glob.glob(
        os.path.join(origami_root, "videos", origami_head_key, "chunk-*", "*.mp4")))
    origami_imgs = _sample_pil_frames(origami_videos, n_samples)
    origami_feats = _pooled_vit_features(visual, processor, origami_imgs, device=device)

    out = {
        "origami": {
            "n_samples": len(origami_imgs),
            "channel_mean_abs_avg": float(np.mean(np.abs(origami_feats.mean(axis=0)))),
            "channel_std_avg": float(np.mean(origami_feats.std(axis=0))),
        }
    }
    if trex_root is not None:
        trex_videos = sorted(glob.glob(
            os.path.join(trex_root, "videos", trex_head_key, "chunk-*", "*.mp4")))
        trex_imgs = _sample_pil_frames(trex_videos, n_samples)
        trex_feats = _pooled_vit_features(visual, processor, trex_imgs, device=device)
        out["trex_midtrain"] = {
            "n_samples": len(trex_imgs),
            "channel_mean_abs_avg": float(np.mean(np.abs(trex_feats.mean(axis=0)))),
            "channel_std_avg": float(np.mean(trex_feats.std(axis=0))),
        }
        mu_o = origami_feats.mean(axis=0)
        mu_t = trex_feats.mean(axis=0)
        cos = float(np.dot(mu_o, mu_t) / (np.linalg.norm(mu_o) * np.linalg.norm(mu_t) + 1e-12))
        out["cosine_distance_between_means"] = 1.0 - cos
    else:
        out["trex_midtrain"] = None
        out["cosine_distance_between_means"] = None
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def render_markdown(report: dict) -> str:
    lines = ["# origami vs T-Rex-midtrain distribution-shift report", ""]

    lines += ["## Chunk-delta magnitude per block (RMS, real units)", ""]
    lines += ["| chunk step | " + " | ".join(name for name, _ in ACTION_BLOCKS) + " |",
              "|---" * (len(ACTION_BLOCKS) + 1) + "|"]
    for k, blocks in report["chunk_delta_magnitude"].items():
        lines.append("| " + k + " | " + " | ".join(f"{blocks[n]:.4g}" for n, _ in ACTION_BLOCKS) + " |")
    lines.append("")

    lines += ["## F6 magnitude per finger (RMS, N / N·m)", ""]
    lines += ["| finger | force_rms | torque_rms | degenerate dims |", "|---|---|---|---|"]
    for label, v in report["f6_magnitude_per_finger"].items():
        lines.append(f"| {label} | {v['force_rms']:.4g} | {v['torque_rms']:.4g} | {v['n_masked_degenerate']} |")
    lines.append("")

    lines += ["## Deform-tile occupancy per finger", ""]
    lines += ["| finger | mean occupancy | n frames |", "|---|---|---|"]
    for label, v in report["deform_occupancy"].items():
        if v is None:
            lines.append(f"| {label} | n/a | 0 |")
        else:
            lines.append(f"| {label} | {v['mean_occupancy']:.4g} | {v['n_frames']} |")
    lines.append("")

    if report.get("g15") is not None:
        g15 = report["g15"]
        lines += ["## G15 — frozen VQ-VAE codebook health on origami F6 windows", "",
                  f"n_windows={g15['n_windows']}, codebook_size={g15['codebook_size']}, "
                  f"clamp_saturation_fraction={g15['clamp_saturation_fraction']:.4g}", ""]
        lines += ["| slot | codes used | normalized entropy | top-5 code frac |", "|---|---|---|---|"]
        for name, v in g15["slots"].items():
            top5 = ", ".join(f"{x:.3f}" for x in v["top5_code_frac"])
            lines.append(f"| {name} | {v['n_codes_used']}/{g15['codebook_size']} | "
                        f"{v['normalized_entropy']:.4g} | {top5} |")
        lines.append("")
        verdict = "PASS" if g15["clamp_saturation_fraction"] < 0.5 and \
            min(v["normalized_entropy"] for v in g15["slots"].values()) > 0.1 else "FAIL"
        lines.append(f"**G15 verdict: {verdict}** (heuristic: <50% clamp saturation and every "
                     "slot's normalized entropy > 0.1 -- i.e. not a near-constant code; see "
                     "REDESIGN_PLAN.md §11.8 for the fix ladder if this fails)")
        lines.append("")

    if report.get("vit_feature_shift") is not None:
        v = report["vit_feature_shift"]
        lines += ["## Frozen-ViT pooled-patch-token feature statistics", ""]
        lines.append(f"origami: n={v['origami']['n_samples']}, "
                     f"mean|channel mean|={v['origami']['channel_mean_abs_avg']:.4g}, "
                     f"avg channel std={v['origami']['channel_std_avg']:.4g}")
        if v["trex_midtrain"] is not None:
            lines.append(f"trex_midtrain sample: n={v['trex_midtrain']['n_samples']}, "
                         f"mean|channel mean|={v['trex_midtrain']['channel_mean_abs_avg']:.4g}, "
                         f"avg channel std={v['trex_midtrain']['channel_std_avg']:.4g}")
            lines.append(f"cosine distance between dataset means: "
                         f"{v['cosine_distance_between_means']:.4g}")
            lines.append("*Caveat: ViT activations are known to have a handful of very "
                         "large-magnitude \"massive activation\" channels shared across "
                         "essentially all inputs (a documented transformer phenomenon); those "
                         "can dominate a raw cosine similarity between mean feature vectors "
                         "and mask real per-channel distribution shift even when one exists. "
                         "A near-zero distance here should not be read as \"no visual shift\" "
                         "on its own -- cross-check against the per-channel mean/std reported "
                         "above, and REDESIGN_PLAN.md §11.9's qualitative shift table (FOV, "
                         "wrist fisheye vs rectilinear, photometry) which this coarse "
                         "statistic doesn't capture.*")
        else:
            lines.append("(no --trex-dataset-root given — cross-dataset comparison skipped)")
        lines.append("")

    return "\n".join(lines)


def run(root: str, checkpoint: Optional[str], trex_dataset_root: Optional[str],
        n_windows: int, n_deform_samples: int, n_vit_samples: int, device: str) -> dict:
    report = {
        "chunk_delta_magnitude": chunk_delta_magnitude(root),
        "f6_magnitude_per_finger": f6_magnitude_per_finger(root),
        "deform_occupancy": deform_occupancy(root, n_samples=n_deform_samples),
        "g15": None,
        "vit_feature_shift": None,
    }
    if checkpoint is not None:
        report["g15"] = vqvae_code_entropy(root, checkpoint, n_windows=n_windows)
        report["vit_feature_shift"] = vit_feature_shift(
            checkpoint, root, trex_root=trex_dataset_root, n_samples=n_vit_samples, device=device)
    else:
        logger.warning("--checkpoint not given: skipping G15 and the ViT feature section "
                       "(both need the midtrain checkpoint's model.pt).")
    return report


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, help="Merged origami dataset root (post prepare.py).")
    p.add_argument("--checkpoint", default=None,
                   help="T-Rex midtrain checkpoint dir (training_args.json/config.json/model.pt/"
                        "processor/). Needed for G15 and the ViT section.")
    p.add_argument("--trex-dataset-root", default=None,
                   help="Local LeRobot-v3 root of a zekaiwang/trex_dataset sample, for the "
                        "cross-dataset ViT feature comparison.")
    p.add_argument("--n-windows", type=int, default=200_000, help="F6 windows for G15.")
    p.add_argument("--n-deform-samples", type=int, default=40, help="Frames per finger for occupancy.")
    p.add_argument("--n-vit-samples", type=int, default=40, help="Head-camera frames per dataset for ViT stats.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default=None, help="Markdown report path (default: stdout).")
    p.add_argument("--json-output", default=None, help="Also dump the raw report dict as JSON.")
    args = p.parse_args(argv)

    report = run(args.root, args.checkpoint, args.trex_dataset_root,
                 args.n_windows, args.n_deform_samples, args.n_vit_samples, args.device)
    md = render_markdown(report)
    if args.output:
        with open(args.output, "w") as f:
            f.write(md)
        logger.info("wrote %s", args.output)
    else:
        print(md)
    if args.json_output:
        with open(args.json_output, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
