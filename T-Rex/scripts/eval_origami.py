"""Offline evaluation of a fine-tuned T-Rex policy on held-out origami seasons.

Runs the model exactly as the deployed policy would -- the cascaded slow tick
(`forward_flow_action_partial`) followed by the tactile fast tick
(`tactile_flow_continue`) -- over held-out seasons, and reports how far the
predicted 25x65 chunk lands from the teleoperator's.

What it reports and why:

  MAE / RMSE            in radians and degrees, overall and per joint group.
  MSE                   flattened over horizon x dims, which is the definition
                        `check_zenoh_policy.py --obs-type dataset` uses, so the
                        number is directly comparable to the pi0.5 baseline
                        recorded under `dataset_checks/`.
  horizon curve         error at each chunk step k = 0..24; a policy that is
                        accurate at k=0 and diverges by k=24 fails differently
                        from one that is uniformly biased.
  contact split         metrics restricted to frames where a fingertip carries
                        real force.  This is the only place the tactile expert
                        can help, so a whole-dataset average hides its effect.
  tactile ablation      the same metrics from `forward_flow_action_full`, which
                        runs the action expert alone over the full flow.  The
                        gap is what the tactile expert buys.
  naive baselines       hold-current-state (zero delta) and repeat-current-
                        command.  A policy that cannot beat these has learned
                        nothing, and on a 0.83 s horizon they are not weak.
                        These are the floor to compare against -- the released
                        midtrain checkpoint cannot serve as a zero-shot baseline
                        because its action head is 62-D eef and this task is
                        65-D joint, so it has no way to emit a valid action.
  safety                the local Shadow evaluator's own checks -- URDF position
                        limits, per-group step jumps and velocity at 30 Hz --
                        so a submission-blocking trajectory shows up here.

Usage:
    python scripts/eval_origami.py \
        --checkpoint_path /content/outputs/.../checkpoint-1-8000 \
        --origami_root    /content/data/origami_flat/val \
        --out_dir         /content/eval/finetuned
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from qwen_vla import extend_position_ids_for_flare, split_slow_fast_embeds
from qwen_vla.origami_dataset import JOINT_GROUPS, OrigamiDataset, denormalize
from trex_origami.seasons import JOINT_NAMES

RAD2DEG = 180.0 / math.pi

# The local Shadow evaluator's per-group step-jump budget
# (`participant_local_evaluator/trajectory.py:18-24`).
GROUP_JUMP_LIMITS = {
    "left_arm": 0.1,
    "left_hand": math.radians(60),
    "right_arm": 0.1,
    "right_hand": math.radians(60),
    "motor": 0.06,
}
POSITION_TOLERANCE_RAD = math.radians(2)   # same tolerance the evaluator allows
CONTROL_HZ = 30.0


# ── URDF-based safety checking ────────────────────────────────────────────────
class SafetyChecker:
    """Replicates the Shadow evaluator's trajectory checks on a predicted chunk.

    Position limits and velocity limits come from the URDF; step jumps from the
    kit's per-group table.  Step 0 is compared against the current measured
    state, exactly as `TrajectoryValidator.validate` seeds `previous = current`.

    These numbers are only meaningful against a reference: the URDF's finger
    limits are tighter than the hardware actually allows, so the *teleoperated
    demonstrations themselves* trip the position check on ~3% of values (six
    DIP/MCP joints reach ~1.57 rad against a 1.396 rad URDF bound).  The
    evaluator therefore scores the ground-truth chunk too and reports it as the
    `teleop_gt` row; only the gap to that row says anything about the policy.
    """

    def __init__(self, urdf_path: str):
        self.limits = self._load_limits(urdf_path)
        self.jump = np.array(
            [GROUP_JUMP_LIMITS[_group_of(i)] for i in range(len(JOINT_NAMES))],
            dtype=np.float64)

    @staticmethod
    def _load_limits(urdf_path: str) -> np.ndarray:
        root = ET.parse(urdf_path).getroot()
        by_name = {}
        for joint in root.iter("joint"):
            limit = joint.find("limit")
            if limit is None:
                continue
            by_name[joint.get("name")] = (
                float(limit.get("lower", "-inf")),
                float(limit.get("upper", "inf")),
                float(limit.get("velocity", "inf")),
            )
        missing = [n for n in JOINT_NAMES if n not in by_name]
        if missing:
            raise KeyError(f"URDF {urdf_path} lacks limits for {missing[:5]} "
                           f"({len(missing)} joints) — is this the north_poc2_2 URDF?")
        return np.array([by_name[n] for n in JOINT_NAMES], dtype=np.float64)

    def check(self, state: np.ndarray, chunk_abs: np.ndarray) -> Dict[str, np.ndarray]:
        """state [65], chunk_abs [T, 65] -> per-violation-type counts over [T, 65]."""
        lower, upper, vel = self.limits[:, 0], self.limits[:, 1], self.limits[:, 2]
        previous = np.concatenate([state[None, :], chunk_abs[:-1]], axis=0)
        delta = np.abs(chunk_abs - previous)
        return {
            "lower_limit": chunk_abs < (lower - POSITION_TOLERANCE_RAD),
            "upper_limit": chunk_abs > (upper + POSITION_TOLERANCE_RAD),
            "step_jump": delta > self.jump,
            "velocity": (delta * CONTROL_HZ) > vel,
        }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _group_of(dim: int) -> str:
    for name, lo, hi in JOINT_GROUPS:
        if lo <= dim < hi:
            return name
    raise IndexError(dim)


# ── metric accumulation ───────────────────────────────────────────────────────
class ErrorAccumulator:
    """Sufficient statistics for every metric, kept as [T, D] sums.

    Everything reported (overall, per joint, per group, per horizon step,
    contact split) is a reduction of these, so one pass over the data serves
    all of them and nothing needs the full prediction tensor in memory.
    """

    def __init__(self, horizon: int, dim: int):
        self.horizon, self.dim = horizon, dim
        self.abs = np.zeros((horizon, dim), dtype=np.float64)
        self.sq = np.zeros((horizon, dim), dtype=np.float64)
        self.n = 0
        self.abs_contact = np.zeros((horizon, dim), dtype=np.float64)
        self.sq_contact = np.zeros((horizon, dim), dtype=np.float64)
        self.n_contact = 0
        self.violations = defaultdict(int)
        self.n_chunks_with_violation = 0
        self.throughput: List[float] = []      # amortised seconds/sample, batched

    def add(self, error: np.ndarray, contact: Optional[np.ndarray] = None) -> None:
        """error [B, T, D] in radians."""
        a, s = np.abs(error), error ** 2
        self.abs += a.sum(axis=0)
        self.sq += s.sum(axis=0)
        self.n += error.shape[0]
        if contact is not None and contact.any():
            self.abs_contact += a[contact].sum(axis=0)
            self.sq_contact += s[contact].sum(axis=0)
            self.n_contact += int(contact.sum())

    def add_safety(self, flags: Dict[str, np.ndarray]) -> None:
        any_flag = False
        for name, mask in flags.items():
            hits = int(mask.sum())
            self.violations[name] += hits
            any_flag |= hits > 0
        self.n_chunks_with_violation += int(any_flag)

    # ── reductions ────────────────────────────────────────────────────────────
    def _summary(self, abs_sum: np.ndarray, sq_sum: np.ndarray, n: int) -> dict:
        if n == 0:
            return {}
        mae = abs_sum / n
        mse = sq_sum / n
        return {
            "n_samples": int(n),
            "mae_rad": float(mae.mean()),
            "mae_deg": float(mae.mean() * RAD2DEG),
            "rmse_rad": float(math.sqrt(mse.mean())),
            "rmse_deg": float(math.sqrt(mse.mean()) * RAD2DEG),
            # Flattened over horizon x dims — the definition check_zenoh_policy.py
            # uses, so this is comparable to the kit's pi0.5 numbers.
            "mse": float(mse.mean()),
            "per_group": {
                name: {
                    "mae_deg": float(mae[:, lo:hi].mean() * RAD2DEG),
                    "rmse_deg": float(math.sqrt(mse[:, lo:hi].mean()) * RAD2DEG),
                }
                for name, lo, hi in JOINT_GROUPS
            },
            "per_horizon_step_mae_deg": (mae.mean(axis=1) * RAD2DEG).tolist(),
        }

    def report(self) -> dict:
        out = {"overall": self._summary(self.abs, self.sq, self.n)}
        if self.n_contact:
            out["contact"] = self._summary(self.abs_contact, self.sq_contact, self.n_contact)
            no_abs = self.abs - self.abs_contact
            no_sq = self.sq - self.sq_contact
            n_free = self.n - self.n_contact
            if n_free > 0:
                out["no_contact"] = self._summary(no_abs, no_sq, n_free)
            out["contact_fraction"] = self.n_contact / max(1, self.n)
        if self.n:
            # Report per-value rates alongside the per-chunk flag.  The URDF's
            # finger limits are tight enough that ~90% of *teleoperated* chunks
            # contain at least one flagged value, so the binary rate saturates
            # and only the per-value fraction separates a good policy from a bad
            # one.  Denominator is every (sample, step, joint) triple checked.
            checked = self.n * self.horizon * self.dim
            out["safety"] = {
                "violations": {k: int(v) for k, v in sorted(self.violations.items())},
                "violation_rate_per_value": {
                    k: v / checked for k, v in sorted(self.violations.items())},
                "chunks_with_any_violation": int(self.n_chunks_with_violation),
                "chunk_violation_rate": self.n_chunks_with_violation / self.n,
            }
        if self.throughput:
            per = np.array(self.throughput)
            out["batched_ms_per_sample"] = float(per.mean() * 1000)
        return out

    def per_joint_mae_deg(self) -> np.ndarray:
        return (self.abs / max(1, self.n)).mean(axis=0) * RAD2DEG


# ── model plumbing ────────────────────────────────────────────────────────────
def build_embeds(model, batch, device):
    """Slow/fast embedding split, position ids and state token for one batch.

    Mirrors `train.py:run_validation` so the evaluation sees exactly the
    sequence layout the model was trained on.
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    pixel_values = batch.get("pixel_values")
    grid_thw = batch.get("image_grid_thw")
    if pixel_values is not None:
        pixel_values = pixel_values.to(device, dtype=torch.bfloat16)
    if grid_thw is not None:
        grid_thw = grid_thw.to(device)

    inputs_embeds = model.prepare_inputs_embeds(
        input_ids=input_ids, pixel_values=pixel_values, image_grid_thw=grid_thw)

    batch_size = inputs_embeds.shape[0]
    n_slow = batch["n_slow_images"]
    if grid_thw is not None and grid_thw.shape[0] > n_slow:
        merge = getattr(model.visual, "spatial_merge_size", 2)
        per_sample = grid_thw.shape[0] // batch_size
        n_slow_tokens = sum(int(g[0] * (g[1] // merge) * (g[2] // merge))
                            for g in grid_thw[:per_sample][:n_slow])
        slow_embeds, fast_embeds = split_slow_fast_embeds(
            inputs_embeds, input_ids, model.image_token_id, n_slow_tokens)
    else:
        slow_embeds, fast_embeds = inputs_embeds, inputs_embeds[:, :0]

    position_ids, _ = model.get_rope_index(
        input_ids=input_ids, image_grid_thw=grid_thw, attention_mask=attention_mask)
    position_ids = position_ids[:, :, :slow_embeds.shape[1]]

    if model.n_flare_tokens > 0:
        flare = model.flare_queries.expand(batch_size, -1, -1).to(
            device=slow_embeds.device, dtype=slow_embeds.dtype)
        slow_embeds = torch.cat([slow_embeds, flare], dim=1)
        position_ids = extend_position_ids_for_flare(position_ids, model.n_flare_tokens)

    state_embeds = None
    if model.use_robot_state and batch.get("state_raw") is not None:
        state_vec = batch["state_raw"].to(slow_embeds.device, dtype=slow_embeds.dtype)
        state_embeds = model.state_embedder(state_vec).unsqueeze(1)

    return slow_embeds, position_ids, fast_embeds, state_embeds, attention_mask


def _tactile_inputs(batch, device):
    def move(key, dtype):
        value = batch.get(key)
        return None if value is None else value.to(device, dtype=dtype)
    return {
        "tactile_f6": move("tactile_f6s", torch.bfloat16),
        "tactile_deform": move("tactile_deforms", torch.bfloat16),
        "tactile_f6_history": move("tactile_f6_history", torch.float32),
    }


@torch.no_grad()
def predict(model, batch, device, mode: str, total_steps: int, split_step: int,
            generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Normalised action chunk [B, T, D] from one of the two inference paths."""
    slow, pos, fast, state, mask = build_embeds(model, batch, device)
    batch_size = slow.shape[0]
    noise = torch.randn(batch_size, model.action_chunk, model.action_dim,
                        dtype=torch.bfloat16, device=device, generator=generator)

    if mode == "blind":
        # Action expert alone over the full tau in [0, 1] — the paper's clean
        # "without tactile expert" ablation.
        return model.forward_flow_action_full(
            inputs_embeds=slow, position_ids=pos, attention_mask=mask,
            noise=noise, state_embeds=state, fast_embeds=fast,
            num_steps=total_steps)

    x_split, cached_kv, n_action, tau_split = model.forward_flow_action_partial(
        inputs_embeds=slow, position_ids=pos, attention_mask=mask,
        noise=noise, state_embeds=state, fast_embeds=fast,
        num_steps_total=total_steps, split_step=split_step, refresh_clean_kv=True)
    return model.tactile_flow_continue(
        cached_kv=cached_kv, latent_position_ids=pos, n_action_in_cache=n_action,
        x_split=x_split, tau_split=tau_split, attention_mask=mask,
        num_steps_total=total_steps, split_step=split_step,
        **_tactile_inputs(batch, device))


# ── dataset wiring ────────────────────────────────────────────────────────────
class _Printer:
    """Stands in for the accelerate `Accelerator` the dataset only uses to print."""
    @staticmethod
    def print(*args, **kwargs):
        print(*args, **kwargs)


def dataset_config_from_checkpoint(checkpoint: str, overrides: dict) -> SimpleNamespace:
    """Rebuild the loader flags the checkpoint was trained with.

    Evaluating with different flags than training silently changes the input
    the model sees (e.g. dropping the deform tokens shortens the tactile
    sequence), so these come from `training_args.json` rather than the CLI.
    """
    path = os.path.join(checkpoint, "training_args.json")
    saved = {}
    if os.path.exists(path):
        with open(path) as handle:
            saved = json.load(handle)
    else:
        print(f"WARNING: no training_args.json in {checkpoint}; falling back to defaults")

    config = SimpleNamespace(
        action_dim=int(saved.get("action_dim", 65)),
        action_chunk=int(saved.get("action_chunk", 25)),
        image_size=saved.get("image_size", [224, 224]),
        use_robot_state=int(saved.get("use_robot_state", 1)),
        use_tactile_vec=int(saved.get("use_tactile_vec", 1)),
        use_tactile_deform=int(saved.get("use_tactile_deform", 1)),
        use_tactile_vqvae=int(saved.get("use_tactile_vqvae", 1)),
        vqvae_window=int(saved.get("vqvae_window", 16)),
        # Never augment at eval, and never bake FLARE future frames: they cost
        # extra parquet reads and the loss they serve is not computed here.
        state_noise_mode="none",
        use_flare=0,
        flare_loss_weight=0.0,
        n_flare_steps=0,
        flare_frame_stride=int(saved.get("flare_frame_stride", 4)),
        phase_mode=saved.get("phase_mode", "") or "",
        origami_sampler="random",
        origami_cache_groups=8,
        origami_val_root="",
        origami_root="",
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


# ── plots ─────────────────────────────────────────────────────────────────────
def write_plots(out_dir: str, reports: Dict[str, dict], per_joint: Dict[str, np.ndarray],
                traces: Optional[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # `teleop_gt` exists only to give the safety numbers a reference floor; its
    # error is zero by construction and would just be a flat line here.
    reports = {k: v for k, v in reports.items() if k != "teleop_gt"}
    per_joint = {k: v for k, v in per_joint.items() if k != "teleop_gt"}

    # 1. error vs horizon step
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, report in reports.items():
        curve = report.get("overall", {}).get("per_horizon_step_mae_deg")
        if curve:
            ax.plot(range(len(curve)), curve, marker="o", ms=3, label=name)
    ax.set_xlabel("chunk step k (30 Hz)")
    ax.set_ylabel("MAE (deg)")
    ax.set_title("Prediction error across the 0.83 s action horizon")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "horizon_error.png"), dpi=150)
    plt.close(fig)

    # 2. per-group bars
    groups = [g[0] for g in JOINT_GROUPS]
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(reports)
    width = 0.8 / max(1, len(names))
    for i, name in enumerate(names):
        per_group = reports[name].get("overall", {}).get("per_group", {})
        values = [per_group.get(g, {}).get("mae_deg", 0.0) for g in groups]
        ax.bar(np.arange(len(groups)) + i * width, values, width, label=name)
    ax.set_xticks(np.arange(len(groups)) + 0.4 - width / 2)
    ax.set_xticklabels(groups)
    ax.set_ylabel("MAE (deg)")
    ax.set_title("Error by joint group")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "group_error.png"), dpi=150)
    plt.close(fig)

    # 3. per-joint bars, coloured by group
    palette = {g: c for g, c in zip(groups, ["C0", "C1", "C2", "C3", "C4"])}
    for name, values in per_joint.items():
        fig, ax = plt.subplots(figsize=(16, 5))
        colors = [palette[_group_of(i)] for i in range(len(values))]
        ax.bar(np.arange(len(values)), values, color=colors)
        ax.set_xticks(np.arange(len(values)))
        ax.set_xticklabels(JOINT_NAMES, rotation=90, fontsize=6)
        ax.set_ylabel("MAE (deg)")
        ax.set_title(f"Per-joint error — {name}")
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"joint_error_{name}.png"), dpi=150)
        plt.close(fig)

    # 4. predicted vs ground-truth traces on a contiguous stretch
    if traces:
        picks = [0, 3, 7, 29, 36, 58]        # one joint from each group, plus both arms
        fig, axes = plt.subplots(len(picks), 1, figsize=(13, 2.1 * len(picks)), sharex=True)
        for ax, dim in zip(np.atleast_1d(axes), picks):
            ax.plot(traces["gt"][:, dim], label="teleop", lw=1.4)
            ax.plot(traces["pred"][:, dim], label="policy", lw=1.0, alpha=0.85)
            ax.set_ylabel(JOINT_NAMES[dim], fontsize=7)
            ax.grid(alpha=0.3)
        np.atleast_1d(axes)[0].legend(loc="upper right", fontsize=8)
        np.atleast_1d(axes)[0].set_title(
            f"Held-out episode ({traces.get('episode', '?')}, {traces.get('mode', '')}): "
            f"teleop vs predicted absolute joint target at chunk step 0")
        np.atleast_1d(axes)[-1].set_xlabel("sample index within the episode")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "episode_trace.png"), dpi=150)
        plt.close(fig)


def write_per_joint_csv(path: str, per_joint: Dict[str, np.ndarray]) -> None:
    names = list(per_joint)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "joint", "group"] + [f"mae_deg_{n}" for n in names])
        for i, joint in enumerate(JOINT_NAMES):
            writer.writerow([i, joint, _group_of(i)]
                            + [f"{per_joint[n][i]:.4f}" for n in names])


def compare_npz(path: str, horizon: int, dim: int) -> Optional[dict]:
    """Score the kit's saved pi0.5 replay under our metric definitions."""
    data = np.load(path)
    if "predicted_actions" not in data or "gt_actions" not in data:
        print(f"WARNING: {path} lacks predicted_actions/gt_actions; skipping")
        return None
    pred, gt = data["predicted_actions"], data["gt_actions"]
    if pred.shape != gt.shape:
        print(f"WARNING: {path} shape mismatch {pred.shape} vs {gt.shape}; skipping")
        return None
    acc = ErrorAccumulator(pred.shape[1], pred.shape[2])
    acc.add((pred - gt).astype(np.float64))
    report = acc.report()
    report["source"] = os.path.basename(path)
    report["note"] = ("baseline replay from the inference kit; same metric "
                      "definitions, but its own sampling of frames")
    return report


# ── main ──────────────────────────────────────────────────────────────────────
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--origami_root", required=True, help="held-out split root")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--base_model_path", default="")
    parser.add_argument("--stats_path", default="",
                        help="defaults to <checkpoint>/stats_data.json")
    parser.add_argument("--dataset_name", default="")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_eval_samples", type=int, default=2000,
                        help="evenly spaced across the split (0 = all)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--modes", nargs="+", default=["cascaded", "blind"],
                        choices=["cascaded", "blind"])
    parser.add_argument("--no_baselines", action="store_true")
    parser.add_argument("--trace_samples", type=int, default=300,
                        help="contiguous samples from the first held-out episode, "
                             "for the predicted-vs-teleop trace plot")
    parser.add_argument("--latency_samples", type=int, default=20,
                        help="batch-1 timed requests, for slow/fast tick latency")
    parser.add_argument("--urdf",
                        default="/home/kshitij/origami_trex/north_poc2_2_urdf_usd/"
                                "north_poc2_2_v3_1.urdf",
                        help="URDF used for the Shadow-evaluator safety checks")
    parser.add_argument("--compare_npz", nargs="*", default=[],
                        help="baseline .npz files from check_zenoh_policy.py")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cascaded_total_steps", type=int, default=10)
    parser.add_argument("--cascaded_split_step", type=int, default=6)
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # ── model ─────────────────────────────────────────────────────────────────
    # Load scripts/test.py by path rather than `import test`: CPython ships its
    # own top-level `test` package, and which one wins depends on sys.path
    # order.  Reusing test.model_load keeps checkpoint reconstruction (config
    # rebuild, embedded VQ-VAE, fp32 tactile buffers) identical to serving.
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "trex_test_server", os.path.join(_SCRIPT_DIR, "test.py"))
    _test = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_test)
    model_load = _test.model_load

    load_args = SimpleNamespace(
        checkpoint_path=args.checkpoint_path,
        base_model_path=args.base_model_path,
        stats_path=args.stats_path,
        dataset_name=args.dataset_name,
        action_dim=65, action_chunk=25,
        use_robot_state=1, use_tactile_vec=1, use_tactile_deform=1,
        tactile_intermediate_size=0,
        n_flare_tokens_per_frame=0, n_flare_steps=0,
        use_tactile_code=0, vqvae_codebook_size=64,
        use_tactile_vqvae=0, vqvae_config=None,
        cascaded_total_steps=args.cascaded_total_steps,
        cascaded_split_step=args.cascaded_split_step,
    )
    training_args_path = os.path.join(args.checkpoint_path, "training_args.json")
    if os.path.exists(training_args_path):
        with open(training_args_path) as handle:
            saved = json.load(handle)
        load_args.action_dim = int(saved.get("action_dim", 65))
        load_args.action_chunk = int(saved.get("action_chunk", 25))
        load_args.use_robot_state = int(saved.get("use_robot_state", 1))
        load_args.use_tactile_deform = int(saved.get("use_tactile_deform", 1))
        load_args.use_tactile_vec = int(saved.get("use_tactile_vec", 1))

    model, processor, _ = model_load(load_args)
    model = model.to(device).eval()
    total_steps = load_args.cascaded_total_steps
    split_step = load_args.cascaded_split_step
    print(f"cascaded flow: {total_steps} steps, split at {split_step} "
          f"(tau_split={1 - split_step / total_steps:.2f})")

    # ── data ──────────────────────────────────────────────────────────────────
    config = dataset_config_from_checkpoint(args.checkpoint_path, {})
    dataset = OrigamiDataset(config, processor, _Printer(), root=args.origami_root)
    horizon, dim = dataset.action_chunk, dataset.action_dim
    if (horizon, dim) != (model.action_chunk, model.action_dim):
        raise ValueError(f"data is [{horizon}, {dim}] but the model is "
                         f"[{model.action_chunk}, {model.action_dim}]")

    if args.num_eval_samples and args.num_eval_samples < len(dataset):
        # Evenly spaced rather than a prefix, so every held-out episode and every
        # phase of the fold contributes.
        indices = np.linspace(0, len(dataset) - 1, args.num_eval_samples).astype(int)
        eval_set = Subset(dataset, np.unique(indices).tolist())
    else:
        eval_set = dataset
    loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=dataset.collate_fn,
                        pin_memory=True)
    print(f"evaluating {len(eval_set)} of {len(dataset)} samples "
          f"({len(dataset.episodes)} episodes, "
          f"{len({e['season'] for e in dataset.episodes})} held-out seasons)")

    action_mask = dataset.action_mask
    action_min, action_max = dataset.action_min, dataset.action_max

    accumulators = {mode: ErrorAccumulator(horizon, dim) for mode in args.modes}
    if not args.no_baselines:
        accumulators["hold_state"] = ErrorAccumulator(horizon, dim)
        accumulators["repeat_command"] = ErrorAccumulator(horizon, dim)

    safety = None
    if os.path.exists(args.urdf):
        safety = SafetyChecker(args.urdf)
        # Reference row: the same checks applied to the teleoperator's own
        # chunk. Its error is zero by construction; only its safety counts matter.
        accumulators["teleop_gt"] = ErrorAccumulator(horizon, dim)
    else:
        print(f"WARNING: URDF not found at {args.urdf}; skipping safety checks")

    # ── evaluation loop ───────────────────────────────────────────────────────
    generator = torch.Generator(device=device).manual_seed(args.seed)
    started = time.time()
    for step, batch in enumerate(loader):
        gt_delta = batch["eval_action_raw"].numpy().astype(np.float64)   # [B, T, D]
        state = batch["eval_state"].numpy().astype(np.float64)           # [B, D]
        contact = batch.get("eval_contact")
        contact = None if contact is None else contact.numpy()

        for mode in args.modes:
            began = time.time()
            normalised = predict(model, batch, device, mode, total_steps, split_step,
                                 generator=generator)
            _sync(device)
            elapsed = (time.time() - began) / max(1, normalised.shape[0])

            pred_delta = denormalize(normalised.float().cpu().numpy().astype(np.float64),
                                     action_mask, action_min, action_max)
            accumulator = accumulators[mode]
            accumulator.add(pred_delta - gt_delta, contact)
            # Amortised throughput, not request latency: batching hides the
            # per-call cost the robot actually waits on. `--latency_samples`
            # measures that separately at batch size 1.
            accumulator.throughput += [elapsed] * normalised.shape[0]

            if safety is not None:
                # The wire contract is absolute radians, so safety is judged on
                # state + delta, not on the delta itself.
                for b in range(pred_delta.shape[0]):
                    accumulator.add_safety(
                        safety.check(state[b], state[b] + pred_delta[b]))

        if safety is not None:
            reference = accumulators["teleop_gt"]
            reference.add(np.zeros_like(gt_delta))
            for b in range(gt_delta.shape[0]):
                reference.add_safety(safety.check(state[b], state[b] + gt_delta[b]))

        if not args.no_baselines:
            # Hold the current measured state for the whole chunk.
            accumulators["hold_state"].add(np.zeros_like(gt_delta) - gt_delta, contact)
            # Repeat the target already commanded at t (gt_delta[:, 0] by
            # construction) for every step of the chunk.  Uses the current
            # command, which the robot does know, so it is a fair reference.
            repeat = np.repeat(gt_delta[:, :1, :], horizon, axis=1)
            accumulators["repeat_command"].add(repeat - gt_delta, contact)

        if step % 20 == 0:
            done = (step + 1) * args.batch_size
            print(f"  {done}/{len(eval_set)} samples | {time.time() - started:.0f}s")

    # ── contiguous trace over one held-out episode ────────────────────────────
    # Must be its own pass: the metric subset is strided across the whole split,
    # so consecutive entries there come from different episodes and would plot
    # as noise rather than a trajectory.
    traces = None
    if args.trace_samples > 0 and args.modes:
        n_trace = min(args.trace_samples, dataset.ep_rows[0])
        trace_loader = DataLoader(
            Subset(dataset, list(range(n_trace))), batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers,
            collate_fn=dataset.collate_fn)
        preds, gts = [], []
        for batch in trace_loader:
            gt_delta = batch["eval_action_raw"].numpy().astype(np.float64)
            state = batch["eval_state"].numpy().astype(np.float64)
            normalised = predict(model, batch, device, args.modes[0],
                                 total_steps, split_step, generator=generator)
            pred_delta = denormalize(normalised.float().cpu().numpy().astype(np.float64),
                                     action_mask, action_min, action_max)
            preds.append(state + pred_delta[:, 0])
            gts.append(state + gt_delta[:, 0])
        traces = {"pred": np.concatenate(preds), "gt": np.concatenate(gts),
                  "episode": dataset.episodes[0]["file"], "mode": args.modes[0]}
        print(f"trace: {n_trace} contiguous samples from {traces['episode']}")

    # ── request latency at batch size 1 ───────────────────────────────────────
    # This is the number the robot waits on, and what the kit's ~540 ms pi0.5
    # figure measures. The slow/fast split matters: only the fast tick reruns
    # within an action chunk, so it bounds the tactile refresh rate.
    # Kept outside the accumulators: this pass always times the cascaded
    # slow+fast path, so attaching it to a scored mode would claim a fast-tick
    # cost the tactile-blind path never pays.
    latency: Dict[str, List[float]] = defaultdict(list)
    if args.latency_samples > 0:
        single = DataLoader(
            Subset(dataset, np.linspace(0, len(dataset) - 1,
                                        args.latency_samples).astype(int).tolist()),
            batch_size=1, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn)
        for batch in single:
            slow, pos, fast, state_emb, mask = build_embeds(model, batch, device)
            noise = torch.randn(1, model.action_chunk, model.action_dim,
                                dtype=torch.bfloat16, device=device, generator=generator)
            _sync(device)
            t0 = time.time()
            x_split, cached_kv, n_action, tau_split = model.forward_flow_action_partial(
                inputs_embeds=slow, position_ids=pos, attention_mask=mask, noise=noise,
                state_embeds=state_emb, fast_embeds=fast,
                num_steps_total=total_steps, split_step=split_step,
                refresh_clean_kv=True)
            _sync(device)
            t1 = time.time()
            model.tactile_flow_continue(
                cached_kv=cached_kv, latent_position_ids=pos,
                n_action_in_cache=n_action, x_split=x_split, tau_split=tau_split,
                attention_mask=mask,
                num_steps_total=total_steps, split_step=split_step,
                **_tactile_inputs(batch, device))
            _sync(device)
            t2 = time.time()
            latency["slow_tick"].append(t1 - t0)
            latency["fast_tick"].append(t2 - t1)
            latency["total"].append(t2 - t0)

    # ── report ────────────────────────────────────────────────────────────────
    reports = {name: acc.report() for name, acc in accumulators.items()}
    per_joint = {name: acc.per_joint_mae_deg() for name, acc in accumulators.items()}

    baselines = [r for r in (compare_npz(p, horizon, dim) for p in args.compare_npz) if r]
    payload = {
        "checkpoint": args.checkpoint_path,
        "data_root": args.origami_root,
        "n_eval_samples": len(eval_set),
        "n_episodes": len(dataset.episodes),
        "n_seasons": len({e["season"] for e in dataset.episodes}),
        "action_chunk": horizon,
        "action_dim": dim,
        "cascaded_total_steps": total_steps,
        "cascaded_split_step": split_step,
        "results": reports,
        "request_latency_ms": {
            phase: {
                "mean": float(np.mean(v) * 1000),
                "p50": float(np.percentile(v, 50) * 1000),
                "p95": float(np.percentile(v, 95) * 1000),
            } for phase, v in latency.items() if v
        },
        "external_baselines": baselines,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as handle:
        json.dump(payload, handle, indent=2)
    write_per_joint_csv(os.path.join(args.out_dir, "per_joint.csv"), per_joint)

    write_plots(args.out_dir, reports, per_joint, traces)

    # ── console summary ───────────────────────────────────────────────────────
    def _unsafe(report):
        rates = report.get("safety", {}).get("violation_rate_per_value")
        return None if rates is None else 100 * sum(rates.values())

    print(f"\n{'model':<16} {'MAE(deg)':>10} {'RMSE(deg)':>10} {'MSE':>10} "
          f"{'MAE contact':>12} {'unsafe%':>9}")
    print("-" * 74)
    for name, report in reports.items():
        if name == "teleop_gt":
            continue
        overall = report.get("overall", {})
        contact = report.get("contact", {})
        unsafe = _unsafe(report)
        print(f"{name:<16} {overall.get('mae_deg', float('nan')):>10.3f} "
              f"{overall.get('rmse_deg', float('nan')):>10.3f} "
              f"{overall.get('mse', float('nan')):>10.5f} "
              f"{contact.get('mae_deg', float('nan')):>12.3f} "
              f"{'' if unsafe is None else f'{unsafe:>8.3f}%'}")
    if "teleop_gt" in reports:
        rate = _unsafe(reports["teleop_gt"]) or 0.0
        print(f"{'teleop_gt':<16} {'-':>10} {'-':>10} {'-':>10} {'-':>12} "
              f"{rate:>8.3f}%   <- reference floor: the demonstrations' own rate, "
              f"since the URDF's finger limits are tighter than the hardware")
    for baseline in baselines:
        overall = baseline["overall"]
        print(f"{baseline['source'][:16]:<16} {overall['mae_deg']:>10.3f} "
              f"{overall['rmse_deg']:>10.3f} {overall['mse']:>10.5f}"
              f"{'':>12} {'  (kit replay)':>8}")
    n_plots = len([f for f in os.listdir(args.out_dir) if f.endswith(".png")])
    if latency["total"]:
        print(f"\nrequest latency at batch 1 (ms): "
              f"slow {np.mean(latency['slow_tick']) * 1000:.0f}  "
              f"fast {np.mean(latency['fast_tick']) * 1000:.0f}  "
              f"total {np.mean(latency['total']) * 1000:.0f}   "
              f"(kit's pi0.5 replay: ~540 ms/infer)")
    print(f"\nwrote {args.out_dir}/metrics.json, per_joint.csv and {n_plots} plots")
    return 0


if __name__ == "__main__":
    sys.exit(main())
