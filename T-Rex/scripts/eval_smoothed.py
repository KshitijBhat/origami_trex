"""Inference-time smoothing for the T-Rex origami policy, evaluated offline.

`eval_diagnostics.py` showed that ~34% of the checkpoint's MSE is *sampling
variance* -- draw-to-draw spread of the flow head, not learned bias -- and that
the predicted chunks are ~80x jerkier than the teleoperator's.  All of that is
removable at inference, without touching the weights.  This script implements
and scores the three deployment-side fixes:

  mean-of-K draws      K flow integrations from independent noise, averaged.
                       Batched: the vision/language prefix is embedded once and
                       repeated K times, so the cost is K flow decodes sharing
                       one prefix, not K full forward passes.  Variance falls
                       as 1/K, so K=4 already removes ~75% of the removable MSE.

  temporal ensembling  ACT-style (Zhao et al. 2023): during a receding-horizon
                       rollout, every past chunk that still covers the current
                       frame votes on it, with exponential weights over chunk
                       age.  Softens the seam jump at each replan.  How much
                       overlap exists depends on the data's sample stride: at
                       the pilot's stride 5 up to 5 chunks overlap; at the full
                       tier's stride 20 only the first 5 frames after a replan
                       get a second vote.

  safety projection    Sequential clamp of the absolute command trajectory to
                       the Shadow evaluator's own feasible set -- URDF position
                       limits (within the evaluator's 2 deg tolerance) and
                       per-step travel bounded by min(group jump limit,
                       velocity limit / 30 Hz).  Zero flagged step-jump /
                       velocity / position violations by construction, at a
                       measured (small) MAE cost, instead of the current
                       chunk_violation_rate of 1.0.

Two passes:

  1. chunk pass    strided samples across the whole split, like eval_origami.
                   Scores every --modes x --draws x {raw, safe} cell.  One set
                   of max(draws) flow draws serves every K (mean over the first
                   K), so the GPU cost is max(draws), not sum(draws).
  2. rollout pass  contiguous rows of a few held-out episodes, replanning at
                   every row exactly as the deployed policy would.  This is the
                   only pass where temporal ensembling is defined (it needs
                   overlapping chunks), and where smoothness is measured on the
                   30 Hz *executed* stream rather than within single chunks.

All numbers remain open-loop chunk-prediction metrics: they rank inference
configurations, they are not a folding success rate.

Usage:
    python scripts/eval_smoothed.py \
        --checkpoint_path /content/outputs/.../checkpoint-2-8000 \
        --origami_root    /content/data/origami_flat/full/val \
        --out_dir         /content/eval/smoothed \
        --urdf            /content/urdf/north_poc2_2_v3_1.urdf
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from collections import defaultdict
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from qwen_vla.origami_dataset import (JOINT_GROUPS, OrigamiDataset,
                                      clamp_frozen_actions, denormalize,
                                      frozen_action_dims)
from trex_origami.seasons import JOINT_NAMES

RAD2DEG = 180.0 / math.pi


def _load_eval_origami():
    """Reuse eval_origami's plumbing (model load path, embeds, accumulator,
    safety checker) without duplicating it.  By path, not by name: `scripts/`
    is not a package."""
    spec = importlib.util.spec_from_file_location(
        "trex_eval_origami", os.path.join(_SCRIPT_DIR, "eval_origami.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EV = None   # set in main(); module-level so helpers can reach the shared code


# ── smoothness helpers (same definitions as eval_diagnostics) ─────────────────
def _jerk_deg_sum(chunk: np.ndarray) -> float:
    """Sum over samples of mean |2nd difference| along the trajectory (deg).

    For a chunk of deltas from a constant state this equals the absolute
    trajectory's own 2nd difference, i.e. commanded jerk at 30 Hz."""
    if chunk.shape[-2] < 3:
        return 0.0
    d2 = chunk[..., 2:, :] - 2 * chunk[..., 1:-1, :] + chunk[..., :-2, :]
    return float(np.abs(d2).mean(axis=(-2, -1)).sum() * RAD2DEG)


def _step_deg_sum(chunk: np.ndarray) -> float:
    """Sum over samples of mean |1st difference| along the trajectory (deg)."""
    if chunk.shape[-2] < 2:
        return 0.0
    return float(np.abs(np.diff(chunk, axis=-2)).mean(axis=(-2, -1)).sum() * RAD2DEG)


class Smoothness:
    """Per-configuration jerk/step accumulator over chunks or streams."""

    def __init__(self):
        self.jerk = 0.0
        self.step = 0.0
        self.n = 0

    def add(self, traj: np.ndarray, n: Optional[int] = None) -> None:
        """traj [..., T, D]; n = number of trajectories (default: leading dims)."""
        self.jerk += _jerk_deg_sum(traj)
        self.step += _step_deg_sum(traj)
        self.n += int(np.prod(traj.shape[:-2])) if n is None else n

    def report(self) -> dict:
        n = max(1, self.n)
        return {"mean_abs_2nd_diff_deg": self.jerk / n,
                "mean_abs_1st_diff_deg": self.step / n}


# ── safety projection ─────────────────────────────────────────────────────────
class SafetyProjector:
    """Project an absolute command trajectory into the evaluator's feasible set.

    Sequential: each step moves from the previous *projected* command toward
    the (position-clipped) target, with per-step travel bounded by
    `rate_margin * min(group jump limit, velocity limit / 30 Hz)`.  Because the
    violation checks are strict inequalities against exactly these bounds, the
    output cannot flag a step-jump or velocity violation, and with position
    clipping on it cannot flag a position violation either (unless the seed
    state itself starts farther outside the limits than one step can recover,
    which the teleop data never does).

    `position_mode`:
      tol   clip into [lower - tolerance, upper + tolerance] (the evaluator
            allows 2 deg past the URDF).  Default: deviates least from the
            teleoperator, who exceeds the URDF finger limits on ~3% of values.
      urdf  clip into [lower, upper] exactly.
      off   no position clipping, rate limiting only.
    """

    def __init__(self, checker, position_mode: str = "tol",
                 rate_margin: float = 0.999):
        lower = checker.limits[:, 0].copy()
        upper = checker.limits[:, 1].copy()
        vel = checker.limits[:, 2]
        tol = None
        if position_mode == "tol":
            tol = rate_margin * EV.POSITION_TOLERANCE_RAD
            self.lo, self.hi = lower - tol, upper + tol
        elif position_mode == "urdf":
            self.lo, self.hi = lower, upper
        else:
            self.lo = np.full_like(lower, -np.inf)
            self.hi = np.full_like(upper, np.inf)
        self.max_step = rate_margin * np.minimum(
            checker.jump, vel / EV.CONTROL_HZ)

    def project(self, seed_abs: np.ndarray, traj_abs: np.ndarray) -> np.ndarray:
        """seed_abs [..., D] (current measured state), traj_abs [..., T, D]."""
        prev = np.array(seed_abs, dtype=np.float64, copy=True)
        out = np.empty_like(traj_abs, dtype=np.float64)
        for k in range(traj_abs.shape[-2]):
            target = np.clip(traj_abs[..., k, :], self.lo, self.hi)
            prev = prev + np.clip(target - prev, -self.max_step, self.max_step)
            out[..., k, :] = prev
        return out


# ── batched mean-of-K prediction ──────────────────────────────────────────────
def _flow_on_embeds(model, slow, pos, fast, state, mask, tact,
                    mode: str, total_steps: int, split_step: int,
                    noise: torch.Tensor) -> torch.Tensor:
    """One flow integration on already-built embeddings (either inference path)."""
    if mode == "blind":
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
        num_steps_total=total_steps, split_step=split_step, **tact)


def _rep(t: Optional[torch.Tensor], k: int, dim: int = 0) -> Optional[torch.Tensor]:
    return None if t is None else t.repeat_interleave(k, dim=dim)


@torch.no_grad()
def predict_draws(model, batch, device, mode: str, total_steps: int,
                  split_step: int, k: int,
                  generator: Optional[torch.Generator] = None,
                  max_flow_batch: int = 64) -> torch.Tensor:
    """K independent flow draws per sample, batched: normalised [B, K, T, D].

    The expensive vision/language prefix is embedded once per sample and
    repeated K times; only the flow decode runs K-wide.  `max_flow_batch`
    bounds the flow batch (B*K) so K=8 at batch 8 does not OOM a 40 GB card.
    The first draw is bit-identical to what a K=1 call with the same generator
    state would produce, so mean-of-K for every K' <= K is a prefix average of
    the same tensor.
    """
    slow, pos, fast, state, mask = EV.build_embeds(model, batch, device)
    B = slow.shape[0]
    tact = (EV._tactile_inputs(batch, device) if mode == "cascaded"
            else {})                                    # unused on the blind path
    noise = torch.randn(B * k, model.action_chunk, model.action_dim,
                        dtype=torch.bfloat16, device=device, generator=generator)

    slow_r, fast_r = _rep(slow, k), _rep(fast, k)
    pos_r = _rep(pos, k, dim=1)                         # position_ids are [3, B, L]
    mask_r, state_r = _rep(mask, k), _rep(state, k)
    tact_r = {key: _rep(value, k) for key, value in tact.items()}

    outs = []
    for s in range(0, B * k, max_flow_batch):
        sl = slice(s, min(s + max_flow_batch, B * k))
        outs.append(_flow_on_embeds(
            model, slow_r[sl], pos_r[:, sl], fast_r[sl],
            None if state_r is None else state_r[sl], mask_r[sl],
            {key: (None if value is None else value[sl])
             for key, value in tact_r.items()},
            mode, total_steps, split_step, noise[sl]))
    # repeat_interleave is sample-major, so the reshape groups draws per sample.
    return torch.cat(outs, dim=0).view(B, k, model.action_chunk, model.action_dim)


# ── temporal ensembling ───────────────────────────────────────────────────────
def ensemble_stream(plans_abs: np.ndarray, gap: int, m: float) -> np.ndarray:
    """ACT-style temporal ensemble of overlapping absolute plans.

    plans_abs [N, T, D]: the absolute chunk predicted at each of N consecutive
    rows, one replan per row, `gap` chunk steps executed between replans.
    Frame j after replan r is covered by plan r-a at chunk step j + a*gap for
    every age a with j + a*gap < T; those votes are blended with weights
    exp(-m * a).  m = 0 is a uniform average; m > 0 favours newer plans;
    m < 0 favours older ones (the ACT paper's exp(-m*i) with i=0 oldest).

    Returns the executed stream [N * gap, D].
    """
    n_plans, horizon, dim = plans_abs.shape
    out = np.empty((n_plans * gap, dim), dtype=np.float64)
    for r in range(n_plans):
        for j in range(gap):
            votes, weights = [], []
            for age in range(horizon):                  # ages until out of chunk
                rr, kk = r - age, j + age * gap
                if rr < 0 or kk >= horizon:
                    break
                votes.append(plans_abs[rr, kk])
                weights.append(math.exp(-m * age))
            w = np.asarray(weights)
            out[r * gap + j] = (np.stack(votes) * (w / w.sum())[:, None]).sum(axis=0)
    return out


# ── rollout stream statistics ─────────────────────────────────────────────────
class StreamStats:
    """Error / smoothness / safety sums for one executed-stream variant."""

    def __init__(self, dim: int):
        self.abs = np.zeros(dim, dtype=np.float64)
        self.sq = np.zeros(dim, dtype=np.float64)
        self.n_frames = 0
        self.smooth = Smoothness()
        self.violations = defaultdict(int)
        self.n_checked = 0

    def add(self, stream: np.ndarray, gt_stream: np.ndarray,
            seed_abs: np.ndarray, checker) -> None:
        err = stream - gt_stream
        self.abs += np.abs(err).sum(axis=0)
        self.sq += (err ** 2).sum(axis=0)
        self.n_frames += stream.shape[0]
        self.smooth.add(stream, n=1)
        if checker is not None:
            for name, mask in checker.check(seed_abs, stream).items():
                self.violations[name] += int(mask.sum())
            self.n_checked += stream.size

    def report(self) -> dict:
        n = max(1, self.n_frames)
        mae = self.abs / n
        mse = self.sq / n
        out = {
            "n_frames": int(self.n_frames),
            "mae_deg": float(mae.mean() * RAD2DEG),
            "rmse_deg": float(math.sqrt(mse.mean()) * RAD2DEG),
            "per_group_mae_deg": {
                name: float(mae[lo:hi].mean() * RAD2DEG)
                for name, lo, hi in JOINT_GROUPS},
            "smoothness": self.smooth.report(),
        }
        if self.n_checked:
            out["safety"] = {
                "violations": {k: int(v) for k, v in sorted(self.violations.items())},
                "violation_rate_per_value": {
                    k: v / self.n_checked for k, v in sorted(self.violations.items())},
            }
        return out


# ── plots ─────────────────────────────────────────────────────────────────────
def write_plots(out_dir: str, chunk_reports: Dict[str, dict],
                smooth_reports: Dict[str, dict], modes: List[str],
                draws: List[int], rollout: Optional[dict],
                traces: List[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def overall(name: str, key: str) -> float:
        return chunk_reports.get(name, {}).get("overall", {}).get(key, float("nan"))

    # 1. MAE vs number of draws, raw vs safety-projected, with the naive floors.
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, mode in enumerate(modes):
        ax.plot(draws, [overall(f"{mode}_k{k}", "mae_deg") for k in draws],
                marker="o", color=f"C{i}", label=f"{mode} (raw)")
        ax.plot(draws, [overall(f"{mode}_k{k}_safe", "mae_deg") for k in draws],
                marker="s", ls="--", color=f"C{i}", label=f"{mode} (safety-projected)")
    for name, color in (("hold_state", "gray"), ("repeat_command", "black")):
        value = overall(name, "mae_deg")
        if math.isfinite(value):
            ax.axhline(value, color=color, lw=1, ls=":", alpha=0.8)
            ax.annotate(name, (draws[-1], value), fontsize=8, color=color,
                        va="bottom", ha="right")
    ax.set_xscale("log", base=2)
    ax.set_xticks(draws)
    ax.set_xticklabels([str(k) for k in draws])
    ax.set_xlabel("flow draws averaged (K)")
    ax.set_ylabel("MAE (deg)")
    ax.set_title("Mean-of-K flow draws: error vs K")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "mae_vs_draws.png"), dpi=150)
    plt.close(fig)

    # 2. within-chunk jerk vs draws, against the teleoperator's own jerk.
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, mode in enumerate(modes):
        ax.plot(draws,
                [smooth_reports.get(f"{mode}_k{k}", {}).get(
                    "mean_abs_2nd_diff_deg", float("nan")) for k in draws],
                marker="o", color=f"C{i}", label=f"{mode} (raw)")
        ax.plot(draws,
                [smooth_reports.get(f"{mode}_k{k}_safe", {}).get(
                    "mean_abs_2nd_diff_deg", float("nan")) for k in draws],
                marker="s", ls="--", color=f"C{i}", label=f"{mode} (safety-projected)")
    teleop = smooth_reports.get("teleop_gt", {}).get("mean_abs_2nd_diff_deg")
    if teleop:
        ax.axhline(teleop, color="black", lw=1, ls=":")
        ax.annotate("teleop_gt", (draws[0], teleop), fontsize=8, va="bottom")
    ax.set_xscale("log", base=2)
    ax.set_xticks(draws)
    ax.set_xticklabels([str(k) for k in draws])
    ax.set_yscale("log")
    ax.set_xlabel("flow draws averaged (K)")
    ax.set_ylabel("mean |2nd difference| over the chunk (deg, log)")
    ax.set_title("Commanded-trajectory jerk vs K")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "smoothness_vs_draws.png"), dpi=150)
    plt.close(fig)

    # 3. horizon curves across K for the first mode.
    fig, ax = plt.subplots(figsize=(9, 5))
    for k in draws:
        curve = chunk_reports.get(f"{modes[0]}_k{k}", {}).get(
            "overall", {}).get("per_horizon_step_mae_deg")
        if curve:
            ax.plot(range(len(curve)), curve, marker="o", ms=3, label=f"K={k}")
    for name in ("hold_state", "repeat_command"):
        curve = chunk_reports.get(name, {}).get("overall", {}).get(
            "per_horizon_step_mae_deg")
        if curve:
            ax.plot(range(len(curve)), curve, ls=":", lw=1.2, label=name)
    ax.set_xlabel("chunk step k (30 Hz)")
    ax.set_ylabel("MAE (deg)")
    ax.set_title(f"Error across the horizon vs draws averaged — {modes[0]}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "horizon_error_by_k.png"), dpi=150)
    plt.close(fig)

    # 4. rollout: executed-stream MAE and jerk per variant.
    if rollout and rollout.get("variants"):
        names = list(rollout["variants"])
        mae = [rollout["variants"][n]["mae_deg"] for n in names]
        jerk = [rollout["variants"][n]["smoothness"]["mean_abs_2nd_diff_deg"]
                for n in names]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        ax1.bar(range(len(names)), mae, color="C0")
        ax1.set_ylabel("MAE over executed stream (deg)")
        ax1.set_title("Receding-horizon rollout: error")
        ax2.bar(range(len(names)), jerk, color="C1")
        ax2.set_yscale("log")
        ax2.set_ylabel("mean |2nd diff| of executed stream (deg, log)")
        ax2.set_title("Receding-horizon rollout: jerk")
        for ax in (ax1, ax2):
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
            ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "rollout_bars.png"), dpi=150)
        plt.close(fig)

    # 5. executed-stream traces on the joints where the wobble lives.
    picks = [3, 7, 14, 36, 43]      # arm 4, thumb CMC_FE + index PIP, both hands
    for i, trace in enumerate(traces):
        fig, axes = plt.subplots(len(picks), 1, figsize=(13, 2.1 * len(picks)),
                                 sharex=True)
        for ax, dim in zip(np.atleast_1d(axes), picks):
            ax.plot(trace["gt"][:, dim], label="teleop", lw=1.4)
            ax.plot(trace["single"][:, dim], label="single draw (deployed today)",
                    lw=0.9, alpha=0.8)
            ax.plot(trace["smoothed"][:, dim],
                    label="mean-of-K + ensemble + safety", lw=1.1, alpha=0.9)
            ax.set_ylabel(JOINT_NAMES[dim], fontsize=7)
            ax.grid(alpha=0.3)
        np.atleast_1d(axes)[0].legend(loc="upper right", fontsize=8)
        np.atleast_1d(axes)[0].set_title(
            f"[{i + 1}/{len(traces)}] {trace['episode']} — executed 30 Hz command "
            f"stream, replan every {trace['gap']} steps", fontsize=9)
        np.atleast_1d(axes)[-1].set_xlabel("frame (30 Hz)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"rollout_trace_{i:02d}.png"), dpi=140)
        plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────
def main(argv: Optional[Sequence[str]] = None) -> int:
    global EV
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--origami_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--base_model_path", default="")
    parser.add_argument("--stats_path", default="")
    parser.add_argument("--dataset_name", default="")
    parser.add_argument("--modes", nargs="+", default=["blind", "cascaded"],
                        choices=["cascaded", "blind"])
    parser.add_argument("--draws", type=int, nargs="+", default=[1, 2, 4, 8],
                        help="K values to score; GPU cost is max(draws), the "
                             "smaller K are prefix averages of the same draws")
    parser.add_argument("--num_eval_samples", type=int, default=1000,
                        help="evenly spaced across the split (0 = all)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_flow_batch", type=int, default=64,
                        help="upper bound on the batch*K flow batch; lower it "
                             "if the K-wide decode OOMs")
    parser.add_argument("--rollout_mode", default="blind",
                        choices=["cascaded", "blind"],
                        help="inference path for the receding-horizon pass "
                             "(blind: the cascade currently subtracts value)")
    parser.add_argument("--rollout_draws", type=int, default=8)
    parser.add_argument("--rollout_episodes", type=int, default=6)
    parser.add_argument("--rollout_rows", type=int, default=120,
                        help="contiguous rows per rolled-out episode")
    parser.add_argument("--ensemble_m", type=float, default=0.0,
                        help="exp(-m*age) chunk weights; 0 = uniform, >0 favours "
                             "newer chunks, <0 favours older (ACT's convention)")
    parser.add_argument("--clamp_positions", choices=["tol", "urdf", "off"],
                        default="tol",
                        help="position bound for the safety projection: within "
                             "the evaluator's 2 deg tolerance (tol), the raw "
                             "URDF limits (urdf), or rate limiting only (off)")
    parser.add_argument("--rate_margin", type=float, default=0.999,
                        help="fraction of the step/velocity limit the projection "
                             "may use (the checks are strict inequalities)")
    parser.add_argument("--latency_samples", type=int, default=10,
                        help="batch-1 timed requests per (mode, K); 0 to skip")
    parser.add_argument("--urdf",
                        default="/home/kshitij/origami_trex/north_poc2_2_urdf_usd/"
                                "north_poc2_2_v3_1.urdf")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clamp_frozen", choices=["auto", "off"], default="auto")
    parser.add_argument("--cascaded_total_steps", type=int, default=10)
    parser.add_argument("--cascaded_split_step", type=int, default=6)
    args = parser.parse_args(argv)

    draws = sorted(set(args.draws))
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    EV = _load_eval_origami()

    # ── model (identical reconstruction path to eval_origami) ─────────────────
    spec = importlib.util.spec_from_file_location(
        "trex_test_server", os.path.join(_SCRIPT_DIR, "test.py"))
    test_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_mod)

    load_args = SimpleNamespace(
        checkpoint_path=args.checkpoint_path, base_model_path=args.base_model_path,
        stats_path=args.stats_path, dataset_name=args.dataset_name,
        action_dim=65, action_chunk=25,
        use_robot_state=1, use_tactile_vec=1, use_tactile_deform=1,
        tactile_intermediate_size=0, n_flare_tokens_per_frame=0, n_flare_steps=0,
        use_tactile_code=0, vqvae_codebook_size=64, use_tactile_vqvae=0,
        vqvae_config=None,
        cascaded_total_steps=args.cascaded_total_steps,
        cascaded_split_step=args.cascaded_split_step)
    training_args_path = os.path.join(args.checkpoint_path, "training_args.json")
    if os.path.exists(training_args_path):
        with open(training_args_path) as handle:
            saved = json.load(handle)
        for key in ("action_dim", "action_chunk", "use_robot_state",
                    "use_tactile_deform", "use_tactile_vec"):
            if key in saved:
                setattr(load_args, key, int(saved[key]))

    model, processor, _ = test_mod.model_load(load_args)
    model = model.to(device).eval()
    total_steps, split_step = args.cascaded_total_steps, args.cascaded_split_step

    # ── data ──────────────────────────────────────────────────────────────────
    config = EV.dataset_config_from_checkpoint(args.checkpoint_path, {})
    dataset = OrigamiDataset(config, processor, EV._Printer(), root=args.origami_root)
    horizon, dim = dataset.action_chunk, dataset.action_dim
    action_mask = dataset.action_mask
    action_min, action_max = dataset.action_min, dataset.action_max
    frozen = frozen_action_dims(action_mask)
    clamp = args.clamp_frozen == "auto" and frozen.size > 0
    print(f"frozen action dims {frozen.tolist()} -> "
          f"{'clamped' if clamp else 'not clamped'}")

    checker = projector = None
    if os.path.exists(args.urdf):
        checker = EV.SafetyChecker(args.urdf)
        projector = SafetyProjector(checker, args.clamp_positions, args.rate_margin)
    else:
        print(f"WARNING: URDF not found at {args.urdf}; safety checks and the "
              f"safety projection are disabled")

    def denorm(normalised: torch.Tensor) -> np.ndarray:
        pred = denormalize(normalised.float().cpu().numpy().astype(np.float64),
                           action_mask, action_min, action_max)
        return clamp_frozen_actions(pred, action_mask) if clamp else pred

    # ═══════════════════ pass 1: chunk-level, strided ═════════════════════════
    if args.num_eval_samples and args.num_eval_samples < len(dataset):
        indices = np.linspace(0, len(dataset) - 1, args.num_eval_samples).astype(int)
        eval_set = Subset(dataset, np.unique(indices).tolist())
    else:
        eval_set = dataset
    loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=dataset.collate_fn,
                        pin_memory=True)
    print(f"chunk pass: {len(eval_set)} samples, modes {args.modes}, "
          f"K in {draws} (drawing {max(draws)}x per mode)")

    accs: Dict[str, "EV.ErrorAccumulator"] = {}
    smooths: Dict[str, Smoothness] = {}

    def acc(name: str):
        if name not in accs:
            accs[name] = EV.ErrorAccumulator(horizon, dim)
            smooths[name] = Smoothness()
        return accs[name]

    generator = torch.Generator(device=device).manual_seed(args.seed)
    started = time.time()
    k_chunk = max(draws)
    for step, batch in enumerate(loader):
        gt_delta = batch["eval_action_raw"].numpy().astype(np.float64)
        state = batch["eval_state"].numpy().astype(np.float64)
        contact = batch.get("eval_contact")
        contact = None if contact is None else contact.numpy()

        for mode in args.modes:
            # Same seed for both modes: paired noise, so the cascaded-vs-blind
            # gap is the tactile expert's effect, not sampling variance.
            generator.manual_seed(args.seed * 1_000_003 + step)
            pred_all = denorm(predict_draws(
                model, batch, device, mode, total_steps, split_step,
                k_chunk, generator=generator, max_flow_batch=args.max_flow_batch))

            for k in draws:
                pred = pred_all[:, :k].mean(axis=1)
                name = f"{mode}_k{k}"
                acc(name).add(pred - gt_delta, contact)
                smooths[name].add(pred)
                if projector is not None:
                    safe = projector.project(state, state + pred) - state
                    acc(f"{name}_safe").add(safe - gt_delta, contact)
                    smooths[f"{name}_safe"].add(safe)
                    for b in range(pred.shape[0]):
                        accs[name].add_safety(
                            checker.check(state[b], state[b] + pred[b]))
                        accs[f"{name}_safe"].add_safety(
                            checker.check(state[b], state[b] + safe[b]))

        acc("hold_state").add(np.zeros_like(gt_delta) - gt_delta, contact)
        acc("repeat_command").add(
            np.repeat(gt_delta[:, :1, :], horizon, axis=1) - gt_delta, contact)
        acc("teleop_gt").add(np.zeros_like(gt_delta))
        smooths["teleop_gt"].add(gt_delta)
        if checker is not None:
            for b in range(gt_delta.shape[0]):
                accs["teleop_gt"].add_safety(
                    checker.check(state[b], state[b] + gt_delta[b]))

        if step % 10 == 0:
            print(f"  {(step + 1) * args.batch_size}/{len(eval_set)} samples | "
                  f"{time.time() - started:.0f}s")

    chunk_reports = {name: a.report() for name, a in accs.items()}
    smooth_reports = {name: s.report() for name, s in smooths.items()}
    for name, report in chunk_reports.items():
        report["smoothness"] = smooth_reports.get(name, {})

    # ═══════════════════ pass 2: receding-horizon rollout ═════════════════════
    rollout: Optional[dict] = None
    traces: List[dict] = []
    gap_frames = dataset.sample_stride
    chunk_stride = max(1, dataset.chunk_stride)
    if gap_frames % chunk_stride or gap_frames // chunk_stride > horizon:
        print(f"WARNING: sample_stride={gap_frames} / chunk_stride={chunk_stride} "
              f"does not fit inside the {horizon}-step chunk; skipping the "
              f"rollout pass (re-prepare val with a smaller --sample-stride)")
    elif args.rollout_episodes > 0 and args.rollout_rows > 1:
        gap = gap_frames // chunk_stride
        max_overlap = 1 + (horizon - 1) // gap
        print(f"rollout pass: mode={args.rollout_mode}, K={args.rollout_draws}, "
              f"replan every {gap} chunk steps ({gap / 30:.2f}s), up to "
              f"{max_overlap} overlapping chunks per frame, ensemble m={args.ensemble_m}")

        starts, offset = [], 0
        for n_rows in dataset.ep_rows:
            starts.append(offset)
            offset += n_rows
        n_eps = min(args.rollout_episodes, len(dataset.ep_rows))
        picked = sorted(set(np.linspace(0, len(dataset.ep_rows) - 1,
                                        n_eps).astype(int).tolist()))

        variant_names = ["single_draw", f"mean_of_{args.rollout_draws}",
                         f"mean_of_{args.rollout_draws}_ens",
                         f"mean_of_{args.rollout_draws}_ens_safe",
                         "hold_state", "teleop_gt"]
        streams = {name: StreamStats(dim) for name in variant_names}

        for ep_i in picked:
            n_rows = min(args.rollout_rows, dataset.ep_rows[ep_i])
            ep_loader = DataLoader(
                Subset(dataset, list(range(starts[ep_i], starts[ep_i] + n_rows))),
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, collate_fn=dataset.collate_fn)
            ep_pred, ep_gt, ep_state = [], [], []
            for bstep, batch in enumerate(ep_loader):
                generator.manual_seed(args.seed * 1_000_003
                                      + ep_i * 10_007 + bstep * 101)
                ep_pred.append(denorm(predict_draws(
                    model, batch, device, args.rollout_mode, total_steps,
                    split_step, args.rollout_draws, generator=generator,
                    max_flow_batch=args.max_flow_batch)))
                ep_gt.append(batch["eval_action_raw"].numpy().astype(np.float64))
                ep_state.append(batch["eval_state"].numpy().astype(np.float64))
            pred = np.concatenate(ep_pred)              # [R, K, T, D] deltas
            gt = np.concatenate(ep_gt)                  # [R, T, D]
            state = np.concatenate(ep_state)            # [R, D]

            plans_single = state[:, None, :] + pred[:, 0]
            plans_mean = state[:, None, :] + pred.mean(axis=1)
            gt_stream = np.concatenate(
                [state[r] + gt[r, :gap] for r in range(gt.shape[0])])
            stitched = {
                "single_draw": np.concatenate(
                    [plans_single[r, :gap] for r in range(gt.shape[0])]),
                f"mean_of_{args.rollout_draws}": np.concatenate(
                    [plans_mean[r, :gap] for r in range(gt.shape[0])]),
                f"mean_of_{args.rollout_draws}_ens": ensemble_stream(
                    plans_mean, gap, args.ensemble_m),
                "hold_state": np.repeat(state, gap, axis=0),
                "teleop_gt": gt_stream,
            }
            if projector is not None:
                stitched[f"mean_of_{args.rollout_draws}_ens_safe"] = \
                    projector.project(
                        state[0], stitched[f"mean_of_{args.rollout_draws}_ens"])

            for name, stream in stitched.items():
                streams[name].add(stream, gt_stream, state[0], checker)
            if len(traces) < 3:
                traces.append({
                    "episode": os.path.basename(dataset.episodes[ep_i]["file"]),
                    "gap": gap, "gt": gt_stream,
                    "single": stitched["single_draw"],
                    "smoothed": stitched.get(
                        f"mean_of_{args.rollout_draws}_ens_safe",
                        stitched[f"mean_of_{args.rollout_draws}_ens"]),
                })
            print(f"  ep {ep_i:3d} "
                  f"{os.path.basename(dataset.episodes[ep_i]['file'])}: "
                  f"{gt.shape[0]} rows -> {gt_stream.shape[0]} executed frames")

        rollout = {
            "mode": args.rollout_mode,
            "draws": args.rollout_draws,
            "gap_steps": gap,
            "gap_ms": 1000.0 * gap_frames / 30.0,
            "max_overlapping_chunks": max_overlap,
            "ensemble_m": args.ensemble_m,
            "episodes": len(picked),
            "variants": {name: s.report() for name, s in streams.items()
                         if s.n_frames},
        }

    # ═══════════════════ pass 3: batch-1 latency per (mode, K) ════════════════
    latency: Dict[str, dict] = {}
    if args.latency_samples > 0:
        single = DataLoader(
            Subset(dataset, np.linspace(0, len(dataset) - 1,
                                        args.latency_samples).astype(int).tolist()),
            batch_size=1, shuffle=False, num_workers=0,
            collate_fn=dataset.collate_fn)
        times: Dict[tuple, List[float]] = defaultdict(list)
        for batch in single:
            for mode in args.modes:
                for k in draws:
                    EV._sync(device)
                    t0 = time.time()
                    predict_draws(model, batch, device, mode, total_steps,
                                  split_step, k, generator=generator,
                                  max_flow_batch=args.max_flow_batch)
                    EV._sync(device)
                    times[(mode, k)].append(time.time() - t0)
        latency = {
            f"{mode}_k{k}": {
                "mean_ms": float(np.mean(v) * 1000),
                "p50_ms": float(np.percentile(v, 50) * 1000),
                "p95_ms": float(np.percentile(v, 95) * 1000),
            } for (mode, k), v in times.items()}

    # ── report ────────────────────────────────────────────────────────────────
    payload = {
        "checkpoint": args.checkpoint_path,
        "data_root": args.origami_root,
        "n_eval_samples": len(eval_set),
        "modes": args.modes,
        "draws": draws,
        "action_chunk": horizon,
        "action_dim": dim,
        "cascaded_total_steps": total_steps,
        "cascaded_split_step": split_step,
        "clamp_frozen": bool(clamp),
        "safety_projection": {
            "position_mode": args.clamp_positions,
            "rate_margin": args.rate_margin,
            "enabled": projector is not None,
        },
        "chunk_pass": chunk_reports,
        "rollout": rollout,
        "request_latency_ms_batch1": latency,
    }
    with open(os.path.join(args.out_dir, "metrics_smoothed.json"), "w") as handle:
        json.dump(payload, handle, indent=2)

    write_plots(args.out_dir, chunk_reports, smooth_reports, args.modes, draws,
                rollout, traces)

    # ── console summary ───────────────────────────────────────────────────────
    def _unsafe(report):
        rates = report.get("safety", {}).get("violation_rate_per_value")
        return None if rates is None else 100 * sum(rates.values())

    print(f"\nchunk pass ({len(eval_set)} samples)")
    print(f"{'config':<22} {'MAE(deg)':>9} {'RMSE(deg)':>10} {'MAE contact':>12} "
          f"{'jerk(deg)':>10} {'unsafe%':>9}")
    print("-" * 78)
    order = ([f"{m}_k{k}{s}" for m in args.modes for k in draws
              for s in ("", "_safe")] + ["hold_state", "repeat_command", "teleop_gt"])
    for name in order:
        if name not in chunk_reports:
            continue
        overall = chunk_reports[name].get("overall", {})
        contact = chunk_reports[name].get("contact", {})
        jerk = smooth_reports.get(name, {}).get("mean_abs_2nd_diff_deg", float("nan"))
        unsafe = _unsafe(chunk_reports[name])
        print(f"{name:<22} {overall.get('mae_deg', float('nan')):>9.3f} "
              f"{overall.get('rmse_deg', float('nan')):>10.3f} "
              f"{contact.get('mae_deg', float('nan')):>12.3f} "
              f"{jerk:>10.3f} "
              f"{'' if unsafe is None else f'{unsafe:>8.3f}%'}")

    if rollout:
        print(f"\nrollout pass (mode={rollout['mode']}, replan every "
              f"{rollout['gap_steps']} steps, {rollout['episodes']} episodes)")
        print(f"{'variant':<26} {'MAE(deg)':>9} {'jerk(deg)':>10} {'unsafe%':>9}")
        print("-" * 58)
        for name, report in rollout["variants"].items():
            unsafe = _unsafe(report)
            print(f"{name:<26} {report['mae_deg']:>9.3f} "
                  f"{report['smoothness']['mean_abs_2nd_diff_deg']:>10.3f} "
                  f"{'' if unsafe is None else f'{unsafe:>8.3f}%'}")

    if latency:
        print("\nbatch-1 request latency (ms, full slow tick incl. embeds):")
        for name, entry in latency.items():
            print(f"  {name:<16} mean {entry['mean_ms']:7.1f}  "
                  f"p50 {entry['p50_ms']:7.1f}  p95 {entry['p95_ms']:7.1f}")

    n_plots = len([f for f in os.listdir(args.out_dir) if f.endswith(".png")])
    print(f"\nwrote {args.out_dir}/metrics_smoothed.json and {n_plots} plots")
    return 0


if __name__ == "__main__":
    sys.exit(main())
