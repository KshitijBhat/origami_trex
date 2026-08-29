"""Bias/variance and self-consistency diagnostics for an origami T-Rex policy.

`eval_origami.py` answers "how far off is the policy".  This answers "off *how*",
because the three plausible causes want three different fixes:

  sampling noise    A flow policy returns a *draw* from the conditional, but MAE
                    rewards the conditional *mean*.  If the draw-to-draw spread
                    is large, most of the reported error is removable at
                    inference by averaging K draws -- no retraining, no data.
                    `mean_of_k` measures exactly that, and the decomposition
                    below says how much is left once it is gone.

  offset vs shape   Predicting the current *pose* well and the coming *motion*
                    badly looks identical to the reverse under a chunk-averaged
                    MAE.  Every row is therefore also reported `anchored`:
                    each predictor's own step-0 error is subtracted from all 25
                    steps, so the anchored number is purely "does it know where
                    the arm is going", with the constant offset removed.
                    This is also what makes `repeat_command` an honest
                    comparison -- unanchored it is handed `action[t]` and so
                    scores exactly 0 at k=0, which flatters it against every row
                    that has to predict k=0.  Anchored, that advantage is gone.
                    (Anchored `repeat_command` and anchored `hold_state` are
                    identical by construction: both predict a constant chunk, so
                    they differ *only* by that step-0 offset.  Seeing them agree
                    is the check that the anchoring is doing what it claims.)

  jitter            A chunk with the right mean and high-frequency fuzz on top
                    scores the same MAE as a smooth chunk that is slightly
                    biased, but only one of them trips the evaluator's velocity
                    and step-jump budgets.  The smoothness block reports mean
                    |2nd difference| along the chunk against the
                    teleoperator's, which is the ratio that predicts those
                    violations.

Baselines
---------
`hold_state`, `repeat_command` and `oracle_prev_command` are the same three
constant-chunk floors `eval_origami.py` reports, all in absolute radians.
`repeat_command` is handed `action[t]`, which is the very thing the policy is
being asked to produce, so it is a floor that cannot actually be reached;
`oracle_prev_command` repeats `action[t-1]`, which a deployed policy genuinely
does have (it sent it), and under the hybrid prep it is also precisely the arms'
own anchor -- so it is what "predict zero" now buys, and the bar the policy has
to clear to have earned anything.

Everything is scored after reconstructing absolute radians through the dataset's
declared anchoring rule, so these numbers are comparable across preps that
anchor differently.

Evaluation runs over *contiguous* stretches of held-out episodes rather than the
strided subset `eval_origami.py` scores, because the smoothness and per-episode
trace numbers are only meaningful within an episode.

Usage:
    python scripts/eval_diagnostics.py \\
        --checkpoint_path /content/outputs/.../checkpoint-1-12000 \\
        --origami_root    /content/data/origami_flat/pilot/val \\
        --out_dir         /content/eval/diagnostics \\
        --n_samples 8 --episodes 10
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from trex_origami.anchoring import build_anchor, describe as describe_anchor, to_absolute
from qwen_vla.origami_dataset import (JOINT_GROUPS, OrigamiDataset,
                                      clamp_frozen_absolute, denormalize,
                                      frozen_action_dims)
from trex_origami.seasons import JOINT_NAMES

RAD2DEG = 180.0 / math.pi


def _load_eval_origami():
    """Reuse eval_origami's model/data plumbing without duplicating it.

    By path rather than by name for the same reason eval_origami loads test.py
    that way: `scripts/` is not a package, and importing by name is at the mercy
    of sys.path order.
    """
    spec = importlib.util.spec_from_file_location(
        "trex_eval_origami", os.path.join(_SCRIPT_DIR, "eval_origami.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── metric helpers ────────────────────────────────────────────────────────────
def _anchor(chunk: np.ndarray) -> np.ndarray:
    """Re-express a chunk as motion relative to its own step 0.

    Removes the constant offset every predictor carries, so what is left is the
    trajectory *shape*.  A predictor handed the true step 0 (`repeat_command`)
    loses its advantage here, and one that nails the current pose but not the
    coming motion stops looking good.
    """
    return chunk - chunk[:, :1, :]


def _jerk_deg(chunk: np.ndarray) -> float:
    """Mean |2nd difference| along the chunk, in degrees.

    The anchor is constant over a chunk, so the 2nd difference is the same in
    target space and in absolute radians -- this measures the commanded
    trajectory's own smoothness, not the offset.
    """
    if chunk.shape[1] < 3:
        return 0.0
    d2 = chunk[:, 2:] - 2 * chunk[:, 1:-1] + chunk[:, :-2]
    return float(np.abs(d2).mean() * RAD2DEG)


def _step_deg(chunk: np.ndarray) -> float:
    """Mean |1st difference| along the chunk, in degrees — per-step travel."""
    if chunk.shape[1] < 2:
        return 0.0
    return float(np.abs(np.diff(chunk, axis=1)).mean() * RAD2DEG)


class Row:
    """Accumulates every reported statistic for one predictor, as [T, D] sums."""

    def __init__(self, horizon: int, dim: int):
        self.abs = np.zeros((horizon, dim))
        self.sq = np.zeros((horizon, dim))
        self.abs_anchored = np.zeros((horizon, dim))
        self.sq_anchored = np.zeros((horizon, dim))
        self.n = 0
        self.jerk_sum = 0.0
        self.step_sum = 0.0
        self.n_chunks = 0

    def add(self, pred: np.ndarray, gt: np.ndarray) -> None:
        """pred/gt are [B, T, D] delta chunks in radians."""
        err = pred - gt
        self.abs += np.abs(err).sum(axis=0)
        self.sq += (err ** 2).sum(axis=0)
        err_a = _anchor(pred) - _anchor(gt)
        self.abs_anchored += np.abs(err_a).sum(axis=0)
        self.sq_anchored += (err_a ** 2).sum(axis=0)
        self.n += pred.shape[0]
        self.jerk_sum += _jerk_deg(pred) * pred.shape[0]
        self.step_sum += _step_deg(pred) * pred.shape[0]
        self.n_chunks += pred.shape[0]

    def report(self) -> dict:
        n = max(1, self.n)
        mae, mse = self.abs / n, self.sq / n
        mae_a, mse_a = self.abs_anchored / n, self.sq_anchored / n
        return {
            "n_samples": int(self.n),
            "mae_deg": float(mae.mean() * RAD2DEG),
            "rmse_deg": float(math.sqrt(mse.mean()) * RAD2DEG),
            "mse": float(mse.mean()),
            "anchored_mae_deg": float(mae_a.mean() * RAD2DEG),
            "anchored_rmse_deg": float(math.sqrt(mse_a.mean()) * RAD2DEG),
            "per_horizon_step_mae_deg": (mae.mean(axis=1) * RAD2DEG).tolist(),
            "anchored_per_horizon_step_mae_deg":
                (mae_a.mean(axis=1) * RAD2DEG).tolist(),
            "per_group_mae_deg": {
                name: float(mae[:, lo:hi].mean() * RAD2DEG)
                for name, lo, hi in JOINT_GROUPS},
            "smoothness": {
                "mean_abs_2nd_diff_deg": self.jerk_sum / max(1, self.n_chunks),
                "mean_abs_1st_diff_deg": self.step_sum / max(1, self.n_chunks),
            },
        }

    def per_joint_mae_deg(self) -> np.ndarray:
        return (self.abs / max(1, self.n)).mean(axis=0) * RAD2DEG

    def per_joint_anchored_mae_deg(self) -> np.ndarray:
        return (self.abs_anchored / max(1, self.n)).mean(axis=0) * RAD2DEG


class Variance:
    """Draw-to-draw spread of the policy, and the bias/variance split it implies.

    For K iid draws x_i of a predictor with mean mu and per-element variance V,
    against target y:

        E[(x_i - y)^2] = V + (mu - y)^2          <- what a single draw scores
        E[(xbar - y)^2] = V/K + (mu - y)^2       <- what mean-of-K scores

    so the unbiased sample variance s^2 estimates V, and (mu - y)^2 -- the part
    no amount of averaging removes -- is `mse_mean_of_k - s^2 / K`.  Reporting
    all three says how much of the headline MAE is a retraining problem and how
    much is a sampling-budget problem.
    """

    def __init__(self, k: int):
        self.k = k
        self.var_sum = 0.0          # sum of unbiased per-element sample variance
        self.n_elements = 0

    def add(self, draws: np.ndarray) -> None:
        """draws is [K, B, T, D]."""
        if draws.shape[0] < 2:
            return
        var = draws.var(axis=0, ddof=1)
        self.var_sum += float(var.sum())
        self.n_elements += int(var.size)

    def report(self, mse_single: float, mse_mean_of_k: float) -> dict:
        if not self.n_elements:
            return {}
        variance = self.var_sum / self.n_elements
        bias_sq = max(0.0, mse_mean_of_k - variance / self.k)
        return {
            "k": self.k,
            "sampling_variance_rad2": variance,
            "sampling_std_deg": float(math.sqrt(variance) * RAD2DEG),
            "bias_rmse_deg": float(math.sqrt(bias_sq) * RAD2DEG),
            "mse_single_draw": mse_single,
            "mse_mean_of_k": mse_mean_of_k,
            # What averaging infinitely many draws would score: the floor that
            # inference-time sampling cannot get below.
            "mse_infinite_k": bias_sq,
            "removable_fraction_of_mse": (
                (mse_single - bias_sq) / mse_single if mse_single > 0 else 0.0),
        }


# ── main ──────────────────────────────────────────────────────────────────────
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--origami_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--base_model_path", default="")
    parser.add_argument("--stats_path", default="")
    parser.add_argument("--dataset_name", default="")
    parser.add_argument("--mode", default="cascaded", choices=["cascaded", "blind"],
                        help="which inference path to diagnose")
    parser.add_argument("--n_samples", type=int, default=8,
                        help="flow draws per observation; >1 enables the "
                             "mean-of-K row and the bias/variance split")
    parser.add_argument("--episodes", type=int, default=10,
                        help="held-out episodes to draw contiguous rows from")
    parser.add_argument("--rows_per_episode", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--clamp_frozen", choices=["auto", "off"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cascaded_total_steps", type=int, default=10)
    parser.add_argument("--cascaded_split_step", type=int, default=6)
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    ev = _load_eval_origami()

    # ── model ─────────────────────────────────────────────────────────────────
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

    # ── data ──────────────────────────────────────────────────────────────────
    config = ev.dataset_config_from_checkpoint(args.checkpoint_path, {})
    dataset = OrigamiDataset(config, processor, ev._Printer(), root=args.origami_root)
    horizon, dim = dataset.action_chunk, dataset.action_dim

    action_mask = dataset.action_mask
    action_min, action_max = dataset.action_min, dataset.action_max
    frozen = frozen_action_dims(action_mask)
    clamp = args.clamp_frozen == "auto" and frozen.size > 0
    print(f"frozen action dims {frozen.tolist()} -> "
          f"{'held at state[j]' if clamp else 'not clamped'}")
    anchor_spec = dataset.anchor_spec
    print(f"action anchoring: {describe_anchor(anchor_spec)}")
    ev.check_anchor_consistency(args.checkpoint_path, dataset)

    # Episodes are laid out contiguously in the global index (`_build_index`
    # appends every row of episode e before episode e+1), so an episode's rows
    # are a slice.  Contiguity is what makes the per-episode traces a trajectory
    # rather than a scatter of unrelated frames.
    starts, offset = [], 0
    for n_rows in dataset.ep_rows:
        starts.append(offset)
        offset += n_rows
    n_eps = min(args.episodes, len(dataset.ep_rows))
    picked = np.linspace(0, len(dataset.ep_rows) - 1, n_eps).astype(int)
    picked = sorted(set(picked.tolist()))
    print(f"diagnosing {len(picked)} of {len(dataset.ep_rows)} held-out episodes, "
          f"<= {args.rows_per_episode} contiguous rows each, "
          f"{args.n_samples} flow draw(s) per observation")

    rows: Dict[str, Row] = {}

    def row(name: str) -> Row:
        if name not in rows:
            rows[name] = Row(horizon, dim)
        return rows[name]

    variance = Variance(args.n_samples)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    teleop_jerk_sum = teleop_step_sum = 0.0
    n_chunks_total = 0
    episode_traces: List[dict] = []

    for ep_i in picked:
        n_rows = min(args.rows_per_episode, dataset.ep_rows[ep_i])
        indices = list(range(starts[ep_i], starts[ep_i] + n_rows))
        loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            collate_fn=dataset.collate_fn)

        ep_gt, ep_pred_single, ep_pred_mean = [], [], []
        for step, batch in enumerate(loader):
            state = batch["eval_state"].numpy().astype(np.float64)
            prev_command = batch["eval_prev_command"].numpy().astype(np.float64)
            # Score in absolute radians so every row below means the same thing
            # regardless of what the prep anchored each dim to.
            anchor = build_anchor(state, prev_command, anchor_spec)
            gt_abs = to_absolute(
                batch["eval_action_raw"].numpy().astype(np.float64), anchor)

            draws = []
            for k in range(args.n_samples):
                # Same observation, different flow noise: the spread across k is
                # the sampling variance and nothing else.  Seeded per (episode,
                # batch, draw) so the whole run is reproducible.
                generator.manual_seed(args.seed * 1_000_003
                                      + ep_i * 10_007 + step * 101 + k)
                normalised = ev.predict(model, batch, device, args.mode,
                                        args.cascaded_total_steps,
                                        args.cascaded_split_step,
                                        generator=generator)
                pred = to_absolute(
                    denormalize(normalised.float().cpu().numpy().astype(np.float64),
                                action_mask, action_min, action_max), anchor)
                if clamp:
                    pred = clamp_frozen_absolute(pred, action_mask, state)
                draws.append(pred)
            draws = np.stack(draws)                       # [K, B, T, D] absolute

            single, mean_k = draws[0], draws.mean(axis=0)
            row(args.mode).add(single, gt_abs)
            row(f"{args.mode}_mean_of_{args.n_samples}").add(mean_k, gt_abs)
            variance.add(draws)

            const = lambda pose: np.repeat(np.asarray(pose)[:, None, :], horizon, axis=1)
            row("hold_state").add(const(state), gt_abs)
            row("repeat_command").add(const(gt_abs[:, 0]), gt_abs)
            # The action commanded one row ago -- what a deployed policy has
            # actually sent, and (under hybrid anchoring) the arms' own anchor.
            # Unlike `repeat_command` it is not handed action[t], the very thing
            # being predicted.  It comes straight off the row now rather than
            # being reconstructed from the previous row, so a strided subset is
            # no longer a problem for it.
            row("oracle_prev_command").add(const(prev_command), gt_abs)

            teleop_jerk_sum += _jerk_deg(gt_abs) * gt_abs.shape[0]
            teleop_step_sum += _step_deg(gt_abs) * gt_abs.shape[0]
            n_chunks_total += gt_abs.shape[0]

            ep_gt.append(gt_abs)
            ep_pred_single.append(single)
            ep_pred_mean.append(mean_k)

        gt_ep = np.concatenate(ep_gt)                     # [R, T, D] absolute

        episode_traces.append({
            "episode": dataset.episodes[ep_i]["file"],
            "season": dataset.episodes[ep_i]["season"],
            "gt": gt_ep[:, 0, :],
            "pred": np.concatenate(ep_pred_single)[:, 0, :],
            "pred_mean": np.concatenate(ep_pred_mean)[:, 0, :],
        })
        print(f"  ep {ep_i:3d} {os.path.basename(dataset.episodes[ep_i]['file'])}: "
              f"{gt_ep.shape[0]} rows")

    # ── report ────────────────────────────────────────────────────────────────
    reports = {name: r.report() for name, r in rows.items()}
    mean_name = f"{args.mode}_mean_of_{args.n_samples}"
    reports["teleop_gt"] = {
        "smoothness": {
            "mean_abs_2nd_diff_deg": teleop_jerk_sum / max(1, n_chunks_total),
            "mean_abs_1st_diff_deg": teleop_step_sum / max(1, n_chunks_total),
        }
    }

    decomposition = variance.report(reports[args.mode]["mse"],
                                    reports[mean_name]["mse"]) \
        if mean_name in reports else {}

    # Per-joint ratio against hold_state: the only view that says, joint by
    # joint, whether the policy has learned anything there at all.  A ratio >= 1
    # means predicting "the robot does not move" would have been better.
    hold = rows["hold_state"].per_joint_mae_deg()
    model_pj = rows[args.mode].per_joint_mae_deg()
    mean_pj = rows[mean_name].per_joint_mae_deg() if mean_name in rows else model_pj
    hold_a = rows["hold_state"].per_joint_anchored_mae_deg()
    model_a = rows[args.mode].per_joint_anchored_mae_deg()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(hold > 1e-9, model_pj / np.maximum(hold, 1e-12), np.inf)
        ratio_mean = np.where(hold > 1e-9, mean_pj / np.maximum(hold, 1e-12), np.inf)
        ratio_anchored = np.where(hold_a > 1e-9, model_a / np.maximum(hold_a, 1e-12),
                                  np.inf)

    csv_path = os.path.join(args.out_dir, "per_joint_ratio.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "joint", "group", "mae_deg_model",
                         "mae_deg_mean_of_k", "mae_deg_hold_state",
                         "ratio_model_over_hold", "ratio_mean_of_k_over_hold",
                         "ratio_anchored", "beats_hold_state"])
        for i, joint in enumerate(JOINT_NAMES):
            writer.writerow([
                i, joint, ev._group_of(i), f"{model_pj[i]:.4f}", f"{mean_pj[i]:.4f}",
                f"{hold[i]:.4f}",
                "inf" if not np.isfinite(ratio[i]) else f"{ratio[i]:.3f}",
                "inf" if not np.isfinite(ratio_mean[i]) else f"{ratio_mean[i]:.3f}",
                "inf" if not np.isfinite(ratio_anchored[i]) else f"{ratio_anchored[i]:.3f}",
                int(np.isfinite(ratio[i]) and ratio[i] < 1.0)])

    n_beat = int(np.sum(np.isfinite(ratio) & (ratio < 1.0)))
    n_beat_mean = int(np.sum(np.isfinite(ratio_mean) & (ratio_mean < 1.0)))
    payload = {
        "checkpoint": args.checkpoint_path,
        "data_root": args.origami_root,
        "mode": args.mode,
        "n_flow_draws": args.n_samples,
        "n_episodes": len(picked),
        "rows_per_episode": args.rows_per_episode,
        "frozen_action_dims": frozen.tolist(),
        "clamp_frozen": bool(clamp),
        "action_anchor": list(anchor_spec),
        "results": reports,
        "bias_variance": decomposition,
        "per_joint_vs_hold_state": {
            "n_joints": len(JOINT_NAMES),
            "n_beating_hold_state": n_beat,
            "n_beating_hold_state_mean_of_k": n_beat_mean,
            "joints_beating_hold_state": [
                JOINT_NAMES[i] for i in range(len(JOINT_NAMES))
                if np.isfinite(ratio[i]) and ratio[i] < 1.0],
            "median_ratio": float(np.median(ratio[np.isfinite(ratio)])),
        },
    }
    with open(os.path.join(args.out_dir, "diagnostics.json"), "w") as handle:
        json.dump(payload, handle, indent=2)

    write_plots(args.out_dir, reports, args.mode, mean_name, ratio,
                ratio_anchored, episode_traces, decomposition)

    # ── console summary ───────────────────────────────────────────────────────
    print(f"\n{'row':<26} {'MAE(deg)':>9} {'anchored':>9} {'RMSE(deg)':>10} "
          f"{'|d2|(deg)':>10}")
    print("-" * 68)
    for name, report in reports.items():
        if name == "teleop_gt":
            continue
        print(f"{name:<26} {report['mae_deg']:>9.3f} "
              f"{report['anchored_mae_deg']:>9.3f} {report['rmse_deg']:>10.3f} "
              f"{report['smoothness']['mean_abs_2nd_diff_deg']:>10.4f}")
    tele = reports["teleop_gt"]["smoothness"]
    print(f"{'teleop_gt':<26} {'-':>9} {'-':>9} {'-':>10} "
          f"{tele['mean_abs_2nd_diff_deg']:>10.4f}   <- reference")
    jerk_ratio = (reports[args.mode]["smoothness"]["mean_abs_2nd_diff_deg"]
                  / max(1e-9, tele["mean_abs_2nd_diff_deg"]))
    print(f"\nsmoothness: policy chunks are {jerk_ratio:.1f}x as jerky as the "
          f"teleoperator's (mean |2nd difference| over the chunk)")

    if decomposition:
        print(f"\nbias/variance over K={decomposition['k']} flow draws:")
        print(f"  single draw        RMSE {math.sqrt(decomposition['mse_single_draw']) * RAD2DEG:7.3f} deg")
        print(f"  mean of K          RMSE {math.sqrt(decomposition['mse_mean_of_k']) * RAD2DEG:7.3f} deg")
        print(f"  irreducible (bias) RMSE {decomposition['bias_rmse_deg']:7.3f} deg  "
              f"<- what K -> inf converges to")
        print(f"  sampling noise      std {decomposition['sampling_std_deg']:7.3f} deg")
        print(f"  {100 * decomposition['removable_fraction_of_mse']:.0f}% of single-draw MSE "
              f"is sampling noise, removable at inference by averaging draws")

    print(f"\nper-joint vs hold_state: model beats it on {n_beat}/{len(JOINT_NAMES)} "
          f"joints (mean-of-K: {n_beat_mean}/{len(JOINT_NAMES)}), "
          f"median ratio {payload['per_joint_vs_hold_state']['median_ratio']:.2f}")
    print(f"wrote {args.out_dir}/diagnostics.json, per_joint_ratio.csv and plots")
    return 0


def write_plots(out_dir: str, reports: Dict[str, dict], mode: str, mean_name: str,
                ratio: np.ndarray, ratio_anchored: np.ndarray,
                traces: List[dict], decomposition: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scored = {k: v for k, v in reports.items() if k != "teleop_gt"}

    # 1. horizon curves, raw and anchored side by side.  The gap between the two
    #    panels is the constant-offset component: a row that moves a lot between
    #    them was being scored mostly on its offset, not on its motion.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    # Cycled dashes and shrinking widths, because anchoring makes curves
    # *coincide*: anchored hold_state and anchored repeat_command are equal by
    # construction, and a solid line drawn over a solid line just looks like one
    # row silently went missing.
    styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    for i, (name, report) in enumerate(scored.items()):
        kw = dict(ls=styles[i % len(styles)], lw=2.4 - 0.3 * i, marker="o", ms=3,
                  label=name)
        axes[0].plot(report["per_horizon_step_mae_deg"], **kw)
        axes[1].plot(report["anchored_per_horizon_step_mae_deg"], **kw)
    axes[0].set_title("as reported")
    axes[1].set_title("anchored (each row's own step-0 error removed)")
    for ax in axes:
        ax.set_xlabel("chunk step k (30 Hz)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("MAE (deg)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Horizon error: offset included vs. shape only")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "horizon_anchored.png"), dpi=150)
    plt.close(fig)

    # 2. bias/variance split
    if decomposition:
        fig, ax = plt.subplots(figsize=(7, 5))
        single = math.sqrt(decomposition["mse_single_draw"]) * RAD2DEG
        mean_k = math.sqrt(decomposition["mse_mean_of_k"]) * RAD2DEG
        bias = decomposition["bias_rmse_deg"]
        bars = ax.bar(["single draw", f"mean of {decomposition['k']}", "K -> inf\n(bias)"],
                      [single, mean_k, bias], color=["C3", "C0", "C2"])
        for bar, value in zip(bars, [single, mean_k, bias]):
            ax.text(bar.get_x() + bar.get_width() / 2, value,
                    f"{value:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("RMSE (deg)")
        ax.set_title(f"How much of the error is removable by averaging draws\n"
                     f"({100 * decomposition['removable_fraction_of_mse']:.0f}% of "
                     f"single-draw MSE is sampling noise)")
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "bias_variance.png"), dpi=150)
        plt.close(fig)

    # 3. per-joint ratio against hold_state.  The 1.0 line is the whole point:
    #    everything above it is a joint where predicting "no motion" was better.
    fig, ax = plt.subplots(figsize=(16, 5))
    finite = np.isfinite(ratio)
    capped = np.where(finite, ratio, np.nanmax(ratio[finite]) * 1.15 if finite.any() else 1)
    colors = ["C2" if finite[i] and ratio[i] < 1 else "C3" for i in range(len(ratio))]
    ax.bar(np.arange(len(capped)), capped, color=colors)
    ax.axhline(1.0, color="black", lw=1.2, ls="--")
    ax.text(0.5, 1.03, "worse than predicting no motion", fontsize=8, va="bottom")
    for i in np.where(~finite)[0]:
        ax.text(i, capped[i], "inf", ha="center", va="bottom", fontsize=6, rotation=90)
    ax.set_xticks(np.arange(len(capped)))
    ax.set_xticklabels(JOINT_NAMES, rotation=90, fontsize=6)
    ax.set_ylabel("MAE(model) / MAE(hold_state)")
    ax.set_title(f"Per-joint skill vs. the do-nothing baseline — {mode} "
                 f"(green = model wins)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "per_joint_ratio.png"), dpi=150)
    plt.close(fig)

    # 4. smoothness against the teleoperator
    fig, ax = plt.subplots(figsize=(9, 5))
    names = [n for n in reports if reports[n].get("smoothness")]
    values = [reports[n]["smoothness"]["mean_abs_2nd_diff_deg"] for n in names]
    colors = ["C7" if n == "teleop_gt" else "C0" for n in names]
    ax.bar(names, values, color=colors)
    ax.set_ylabel("mean |2nd difference| over the chunk (deg)")
    ax.set_title("Commanded-trajectory smoothness\n"
                 "(constant-chunk baselines are 0 by construction; "
                 "the comparison that matters is policy vs. teleop_gt)")
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "smoothness.png"), dpi=150)
    plt.close(fig)

    # 5. per-episode joint traces, single draw vs mean-of-K vs teleop
    picks = [0, 3, 7, 29, 36, 58]
    for i, trace in enumerate(traces):
        fig, axes = plt.subplots(len(picks), 1, figsize=(13, 2.0 * len(picks)),
                                 sharex=True)
        for ax, d in zip(np.atleast_1d(axes), picks):
            ax.plot(trace["gt"][:, d], label="teleop", lw=1.5, color="C0")
            ax.plot(trace["pred"][:, d], label="policy (1 draw)", lw=0.9,
                    alpha=0.8, color="C1")
            ax.plot(trace["pred_mean"][:, d], label=f"policy ({mean_name})",
                    lw=1.1, alpha=0.9, color="C2")
            ax.set_ylabel(JOINT_NAMES[d], fontsize=7)
            ax.grid(alpha=0.3)
        np.atleast_1d(axes)[0].legend(loc="upper right", fontsize=7)
        np.atleast_1d(axes)[0].set_title(
            f"{trace['episode']} ({trace['season']}) — absolute joint target at "
            f"chunk step 0", fontsize=9)
        np.atleast_1d(axes)[-1].set_xlabel("sample index within the episode")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"trace_ep{i:02d}.png"), dpi=140)
        plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
