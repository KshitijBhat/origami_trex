"""Deployment-robustness sweep for a fine-tuned T-Rex origami policy.

`eval_origami.py` scores one prediction per observation, averaged uniformly over
all 25 chunk steps.  The competition harness does receding-horizon control: it
consumes only a *prefix* of the chunk, replans from a fresh observation, and the
observation it hands you may be stale.  Neither of those shows up in a uniform
25-step average.  This script measures what does:

  executed-prefix error   MAE restricted to steps 0..h-1, for every plausible
                          execution length h.  If the harness executes 5 steps
                          and replans, steps 5..24 are never commanded and the
                          uniform average is scoring predictions the robot
                          throws away.  This table is the open-loop-length
                          curve: it is what "how long can I run open loop"
                          actually means for a fixed 25-step chunk.

  receding-horizon stitch predictions at every row of a contiguous stretch are
                          cached once, then re-assembled into the trajectory
                          the robot would have executed for each replan period
                          R.  Two numbers fall out: the error of the stitched
                          trajectory, and the *seam jump* -- how far the new
                          chunk's step 0 sits from the old chunk's step R.  The
                          teleoperator's seam is zero by construction, so any
                          seam jump is the policy disagreeing with its own past
                          self, which is exactly the discontinuity that trips
                          the evaluator's per-group step-jump budget.

  observation staleness   the same observation fed d rows late, per modality
                          (visual / tactile / state / all), plus zero-order
                          hold at period k, which is what a camera or policy
                          running slower than the control loop looks like.

  tactile history         the 16-frame F6 window subsampled-and-held (a
                          lower-rate tactile sensor), and frozen to the current
                          frame repeated 16x.  The frozen variant answers
                          whether the window carries any information at all --
                          the origami fine-tune trains at delay_k=0
                          (`train.py:470`), so nothing in post-training teaches
                          the model to tolerate a degraded window.

  modality dropout        tactile zeroed, wrist cameras blanked.  Bounds the
                          worst case if a sensor drops out mid-episode.

Every configuration sees the *same* samples and the *same* flow-matching noise
per sample, so a reported degradation is the perturbation and not a different
noise draw.  Every configuration is also reported against the `repeat_command`
floor on the identical subset: a perturbation that costs less than the gap to
that floor is not a finding.

Granularity, and its limit
--------------------------
One dataset row is `sample_stride` source frames.  Delays and replan periods are
therefore quantized to that: 167 ms on the stride-5 pilot split, 667 ms on the
stride-20 full split.  Run this on a stride-5 (or finer) split -- on stride 20
the only measurable replan period is 20 frames and sub-second delays are
invisible.  Measuring the 33-133 ms jitter regime needs a val split re-prepared
with `--sample-stride 1`.

What this cannot measure
------------------------
The observation at the replan boundary is the *teleoperator's*, not the one the
robot's own actions would have produced.  The stitching pass is therefore
teacher-forced: it measures prefix accuracy and self-consistency, not
compounding closed-loop error.  True closed-loop divergence needs the simulator.

Usage:
    python scripts/eval_robustness.py \
        --checkpoint_path /content/outputs/.../checkpoint-1-8000 \
        --origami_root    /content/data/origami_flat/pilot/val \
        --out_dir         /content/eval/robustness
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

import numpy as np
import PIL.Image
import torch
from torch.utils.data import DataLoader, Subset

from qwen_vla.origami_dataset import (JOINT_GROUPS, OrigamiDataset, _contact_flag,
                                      clamp_frozen_absolute, denormalize,
                                      frozen_action_dims)
from trex_origami.anchoring import build_anchor, describe as describe_anchor, to_absolute
from trex_origami.seasons import JOINT_NAMES

RAD2DEG = 180.0 / math.pi


def _load_eval_module():
    """Import `eval_origami.py` by path.

    Same reason it loads `test.py` by path: `scripts/` is not a package, and the
    metric definitions, safety checker and embedding plumbing must be shared
    with the main evaluator or the two scripts' numbers stop being comparable.
    """
    spec = importlib.util.spec_from_file_location(
        "trex_eval_origami", os.path.join(_SCRIPT_DIR, "eval_origami.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EVAL = _load_eval_module()
ErrorAccumulator = _EVAL.ErrorAccumulator
SafetyChecker = _EVAL.SafetyChecker
GROUP_JUMP_LIMITS = _EVAL.GROUP_JUMP_LIMITS
build_embeds = _EVAL.build_embeds
_tactile_inputs = _EVAL._tactile_inputs
_group_of = _EVAL._group_of
_sync = _EVAL._sync


# ── perturbation specs ────────────────────────────────────────────────────────
@dataclass
class PerturbSpec:
    """One deployment condition, as a transform on the observation only.

    Nothing here touches the target chunk: the label always comes from the true
    current frame, because the robot is still judged against what it should have
    done *now* even when its inputs are stale.
    """
    label: str
    delay_rows: int = 0
    modalities: Tuple[str, ...] = ()          # subset of vision/tactile/state
    hold_rows: int = 0                        # zero-order hold period, in rows
    hist_stride: int = 1                      # subsample-and-hold the F6 window
    freeze_hist: bool = False                 # window := current frame x W
    drop_tactile: bool = False
    blank_wrists: bool = False
    note: str = ""

    @property
    def is_nominal(self) -> bool:
        return not (self.delay_rows or self.hold_rows or self.hist_stride > 1
                    or self.freeze_hist or self.drop_tactile or self.blank_wrists)


def default_configs() -> Dict[str, PerturbSpec]:
    """The sweep. Names are stable so runs across checkpoints can be diffed."""
    specs = [
        PerturbSpec("nominal", note="reference; identical to eval_origami cascaded"),
        # staleness, one modality at a time -- the per-modality split is the
        # actionable part: if vision dominates, the fix is a faster camera path;
        # if tactile dominates, it is delay augmentation on the tactile expert.
        PerturbSpec("vis_delay_1", delay_rows=1, modalities=("vision",)),
        PerturbSpec("vis_delay_2", delay_rows=2, modalities=("vision",)),
        PerturbSpec("vis_delay_3", delay_rows=3, modalities=("vision",)),
        PerturbSpec("tac_delay_1", delay_rows=1, modalities=("tactile",)),
        PerturbSpec("tac_delay_2", delay_rows=2, modalities=("tactile",)),
        PerturbSpec("tac_delay_3", delay_rows=3, modalities=("tactile",)),
        PerturbSpec("state_delay_1", delay_rows=1, modalities=("state",)),
        PerturbSpec("all_delay_1", delay_rows=1,
                    modalities=("vision", "tactile", "state")),
        PerturbSpec("all_delay_2", delay_rows=2,
                    modalities=("vision", "tactile", "state")),
        # zero-order hold: the whole observation is refreshed only every k rows,
        # so the average staleness is (k-1)/2 rows and the worst case is k-1.
        # This is a slow camera / slow policy, not a fixed latency.
        PerturbSpec("hold_obs_2", hold_rows=2,
                    modalities=("vision", "tactile", "state"),
                    note="observation refreshed every 2nd row"),
        PerturbSpec("hold_obs_4", hold_rows=4,
                    modalities=("vision", "tactile", "state"),
                    note="observation refreshed every 4th row"),
        # tactile window structure
        PerturbSpec("tac_hist_stride_2", hist_stride=2),
        PerturbSpec("tac_hist_stride_4", hist_stride=4),
        PerturbSpec("tac_hist_frozen", freeze_hist=True,
                    note="current F6 frame repeated across the window"),
        # sensor loss
        PerturbSpec("tac_drop", drop_tactile=True),
        PerturbSpec("wrist_blank", blank_wrists=True),
    ]
    return {s.label: s for s in specs}


# ── perturbed view of the dataset ─────────────────────────────────────────────
class PerturbedView(torch.utils.data.Dataset):
    """Applies a `PerturbSpec` to items of an `OrigamiDataset`.

    A delayed observation is a real donor row, not synthesised: rows of one
    episode are contiguous in the global index (`_build_index`), so the sample
    `d` rows back is `idx - d` as long as it stays inside the episode.  The
    caller restricts the evaluated indices so that always holds, which also
    keeps every configuration on the same sample set.
    """
    VISION = ("head", "wrist_left", "wrist_right")
    TACTILE = ("tacf6_hist", "deform")
    STATE = ("state",)
    _BY_MODALITY = {"vision": VISION, "tactile": TACTILE, "state": STATE}

    def __init__(self, base: OrigamiDataset, spec: PerturbSpec):
        self.base, self.spec = base, spec

    def __len__(self):
        return len(self.base)

    def _delay_for(self, idx: int) -> int:
        _, row = self.base.index[idx]
        spec = self.spec
        d = (row % spec.hold_rows) if spec.hold_rows > 1 else spec.delay_rows
        return min(d, row)

    def __getitem__(self, idx):
        item = dict(self.base[idx])
        # Pin the contact flag to the *true* current frame before anything is
        # perturbed, so the contact split stays the same set of frames across
        # configs. Otherwise `tac_drop` would report "no contact anywhere" and
        # `tac_delay` would slice a different subset than `nominal`.
        item["_contact"] = _contact_flag(item)
        item["_idx"] = idx
        spec = self.spec

        d = self._delay_for(idx)
        if d > 0 and spec.modalities:
            donor = self.base[idx - d]
            for modality in spec.modalities:
                for key in self._BY_MODALITY[modality]:
                    if key in donor and key in item:
                        item[key] = donor[key]

        hist = item.get("tacf6_hist")
        if hist is not None and spec.hist_stride > 1:
            # Zero-order hold backwards from the most recent frame, so the
            # current reading survives and only the window's temporal
            # resolution degrades.
            w = hist.shape[0]
            take = [w - 1 - ((w - 1 - i) // spec.hist_stride) * spec.hist_stride
                    for i in range(w)]
            item["tacf6_hist"] = hist[take]
        elif hist is not None and spec.freeze_hist:
            item["tacf6_hist"] = np.repeat(hist[-1:], hist.shape[0], axis=0)

        if spec.drop_tactile:
            if item.get("tacf6_hist") is not None:
                item["tacf6_hist"] = np.zeros_like(item["tacf6_hist"])
            if item.get("deform") is not None:
                item["deform"] = np.zeros_like(item["deform"])

        if spec.blank_wrists:
            for key in ("wrist_left", "wrist_right"):
                if key in item:
                    item[key] = PIL.Image.new("RGB", item[key].size, (128, 128, 128))
        return item

    def collate_fn(self, batch):
        out = self.base.collate_fn(batch)
        out["_idx"] = torch.tensor([b["_idx"] for b in batch], dtype=torch.long)
        out["eval_contact"] = torch.tensor([b["_contact"] for b in batch],
                                           dtype=torch.bool)
        return out


def eligible_indices(dataset: OrigamiDataset, min_row: int,
                     n_samples: int) -> List[int]:
    """Rows at least `min_row` into their episode, evenly spaced across the split.

    Clamping a delay at an episode start would quietly make that sample
    *undelayed*, biasing every delay config optimistic by the fraction of
    samples near a boundary.  Dropping those rows instead costs a little data
    and keeps all configs on one honest, shared subset.
    """
    ok = [i for i, (_, row) in enumerate(dataset.index) if row >= min_row]
    if not ok:
        raise ValueError(f"no rows at least {min_row} into an episode; "
                         f"episodes are too short for this delay sweep")
    if n_samples and n_samples < len(ok):
        picks = np.linspace(0, len(ok) - 1, n_samples).astype(int)
        ok = [ok[i] for i in np.unique(picks)]
    return ok


# ── paired noise + prediction ─────────────────────────────────────────────────
def paired_noise(indices: torch.Tensor, chunk: int, dim: int, seed: int,
                 device: torch.device) -> torch.Tensor:
    """Flow-matching noise keyed to the *sample*, not the call order.

    `eval_origami.py` draws from one running generator, so two configurations
    scoring the same sample get different noise and their difference mixes the
    perturbation with sampling variance.  Seeding per sample index makes every
    configuration's prediction for a given observation differ only by the
    perturbation.  Generated on CPU so the values do not depend on the device.
    """
    generator = torch.Generator()
    out = torch.empty(len(indices), chunk, dim, dtype=torch.float32)
    for i, idx in enumerate(indices.tolist()):
        generator.manual_seed((seed * 1_000_003 + idx) % (2 ** 63 - 1))
        out[i] = torch.randn(chunk, dim, generator=generator)
    return out.to(device=device, dtype=torch.bfloat16)


@torch.no_grad()
def predict_cascaded(model, batch, device, total_steps: int, split_step: int,
                     noise: torch.Tensor) -> torch.Tensor:
    """The deployed path (slow tick then tactile fast tick) with supplied noise."""
    slow, pos, fast, state, mask = build_embeds(model, batch, device)
    x_split, cached_kv, n_action, tau_split = model.forward_flow_action_partial(
        inputs_embeds=slow, position_ids=pos, attention_mask=mask,
        noise=noise, state_embeds=state, fast_embeds=fast,
        num_steps_total=total_steps, split_step=split_step, refresh_clean_kv=True)
    return model.tactile_flow_continue(
        cached_kv=cached_kv, latent_position_ids=pos, n_action_in_cache=n_action,
        x_split=x_split, tau_split=tau_split, attention_mask=mask,
        num_steps_total=total_steps, split_step=split_step,
        **_tactile_inputs(batch, device))


# ── metrics ───────────────────────────────────────────────────────────────────
def prefix_curve(acc: ErrorAccumulator, horizon_steps: Sequence[int]) -> Dict[str, float]:
    """MAE (deg) over executed steps 0..h-1, for each h.

    This is a reduction of the accumulator's [T, D] absolute-error sums, so it
    needs no extra pass and is exactly consistent with the reported MAE (h =
    horizon reproduces it).
    """
    if acc.n == 0:
        return {}
    mae = acc.abs / acc.n
    return {str(h): float(mae[:h].mean() * RAD2DEG) for h in horizon_steps}


def run_config(model, dataset, spec: PerturbSpec, indices: List[int], args,
               device, safety: Optional[SafetyChecker],
               floors: Optional[Dict[str, ErrorAccumulator]] = None
               ) -> ErrorAccumulator:
    view = PerturbedView(dataset, spec)
    loader = DataLoader(Subset(view, indices), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers,
                        collate_fn=view.collate_fn, pin_memory=True)
    acc = ErrorAccumulator(dataset.action_chunk, dataset.action_dim)
    horizon = dataset.action_chunk

    for batch in loader:
        state = batch["eval_state"].numpy().astype(np.float64)
        prev_command = batch["eval_prev_command"].numpy().astype(np.float64)
        contact = batch["eval_contact"].numpy()
        # Absolute radians throughout, reconstructed through the dataset's own
        # anchoring rule.  The perturbations here move observations, never the
        # anchor, so the anchor is the un-perturbed one either way.
        anchor = build_anchor(state, prev_command, dataset.anchor_spec)
        gt_abs = to_absolute(
            batch["eval_action_raw"].numpy().astype(np.float64), anchor)
        noise = paired_noise(batch["_idx"], horizon, dataset.action_dim,
                             args.seed, device)

        normalised = predict_cascaded(model, batch, device, args.cascaded_total_steps,
                                      args.cascaded_split_step, noise)
        _sync(device)
        # Same clamp the eval and serve paths apply, so a degradation reported
        # here is the perturbation and not the torso's un-normalised noise.
        pred_abs = clamp_frozen_absolute(
            to_absolute(denormalize(
                normalised.float().cpu().numpy().astype(np.float64),
                dataset.action_mask, dataset.action_min, dataset.action_max),
                anchor),
            dataset.action_mask, state)
        acc.add(pred_abs - gt_abs, contact)
        if safety is not None:
            for b in range(pred_abs.shape[0]):
                acc.add_safety(safety.check(state[b], pred_abs[b]))

        if floors is not None:
            const = lambda pose: np.repeat(np.asarray(pose)[:, None, :], horizon, axis=1)
            floors["hold_state"].add(const(state) - gt_abs, contact)
            floors["repeat_command"].add(const(gt_abs[:, 0]) - gt_abs, contact)
            floors["oracle_prev_command"].add(const(prev_command) - gt_abs, contact)
    return acc


# ── receding-horizon stitching ────────────────────────────────────────────────
@torch.no_grad()
def receding_horizon(model, dataset, args, device) -> Optional[dict]:
    """Cache predictions at every row of a contiguous stretch, then re-assemble.

    Predicting once per row and re-assembling offline gives every replan period
    from one pass -- a separate pass per period would re-predict the same rows.
    A replan every R rows means the robot executes R*sample_stride/chunk_stride
    chunk steps before the next observation arrives, so only periods that divide
    into whole chunk steps and fit inside the horizon are measurable.
    """
    stride = dataset.sample_stride
    chunk_stride = max(1, dataset.chunk_stride)
    horizon = dataset.action_chunk
    n_rows = min(args.stitch_rows, dataset.ep_rows[0])
    if n_rows < 4:
        print("WARNING: first episode too short for the stitching pass; skipping")
        return None

    view = PerturbedView(dataset, PerturbSpec("nominal"))
    loader = DataLoader(Subset(view, list(range(n_rows))), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers,
                        collate_fn=view.collate_fn)
    preds, gts, states = [], [], []
    for batch in loader:
        state_b = batch["eval_state"].numpy().astype(np.float64)
        anchor = build_anchor(
            state_b, batch["eval_prev_command"].numpy().astype(np.float64),
            dataset.anchor_spec)
        noise = paired_noise(batch["_idx"], horizon, dataset.action_dim,
                             args.seed, device)
        normalised = predict_cascaded(model, batch, device, args.cascaded_total_steps,
                                      args.cascaded_split_step, noise)
        preds.append(clamp_frozen_absolute(
            to_absolute(denormalize(
                normalised.float().cpu().numpy().astype(np.float64),
                dataset.action_mask, dataset.action_min, dataset.action_max),
                anchor),
            dataset.action_mask, state_b))
        gts.append(to_absolute(
            batch["eval_action_raw"].numpy().astype(np.float64), anchor))
        states.append(state_b)
    pred = np.concatenate(preds)      # [N, T, D] absolute radians
    gt = np.concatenate(gts)
    state = np.concatenate(states)    # [N, D]

    jump_limit = np.array([GROUP_JUMP_LIMITS[_group_of(i)]
                           for i in range(dataset.action_dim)])
    out = {"n_rows": int(pred.shape[0]),
           "episode": dataset.episodes[0]["file"],
           "sample_stride": stride,
           "chunk_stride": chunk_stride,
           "by_replan_period": {}}

    r = 1
    while True:
        gap_frames = r * stride
        if gap_frames % chunk_stride or gap_frames // chunk_stride > horizon:
            break
        gap = gap_frames // chunk_stride            # executed chunk steps
        starts = list(range(0, pred.shape[0] - 1, r))
        # stitched executed trajectory vs the teleoperator's over the same frames
        exec_err = np.concatenate([pred[p, :gap] - gt[p, :gap] for p in starts])
        entry = {
            "replan_rows": r,
            "gap_frames": int(gap_frames),
            "gap_ms": 1000.0 * gap_frames / 30.0,
            "executed_steps_per_chunk": int(gap),
            "stitched_mae_deg": float(np.abs(exec_err).mean() * RAD2DEG),
            "stitched_rmse_deg": float(math.sqrt((exec_err ** 2).mean()) * RAD2DEG),
        }
        if gap < horizon:
            # The old chunk's step `gap` and the new chunk's step 0 are the same
            # frame. Teleop agrees with itself there exactly, so whatever the
            # policy shows is its own inconsistency across replans -- and it
            # lands as a single-step jump on the wire.
            pairs = [(p, p + r) for p in starts if p + r < pred.shape[0]]
            seam = np.stack([pred[q, 0] - pred[p, gap]
                             for p, q in pairs]) if pairs else np.zeros((0, dataset.action_dim))
            if len(seam):
                entry["seam"] = {
                    "n_seams": int(len(seam)),
                    "mean_abs_deg": float(np.abs(seam).mean() * RAD2DEG),
                    "p95_abs_deg": float(np.percentile(np.abs(seam), 95) * RAD2DEG),
                    "max_abs_deg": float(np.abs(seam).max() * RAD2DEG),
                    "step_jump_violation_rate": float(
                        (np.abs(seam) > jump_limit).mean()),
                    "seams_with_any_violation": float(
                        (np.abs(seam) > jump_limit).any(axis=1).mean()),
                    "per_group_mean_abs_deg": {
                        name: float(np.abs(seam[:, lo:hi]).mean() * RAD2DEG)
                        for name, lo, hi in JOINT_GROUPS},
                }
        out["by_replan_period"][str(r)] = entry
        r += 1

    if not out["by_replan_period"]:
        print(f"WARNING: sample_stride={stride} with chunk_stride={chunk_stride} "
              f"admits no replan period inside a {horizon}-step chunk; "
              f"re-prepare val with a smaller --sample-stride")
        return None
    return out


# ── plots ─────────────────────────────────────────────────────────────────────
def write_plots(out_dir: str, results: Dict[str, dict], floors: Dict[str, dict],
                stitch: Optional[dict], prefix_steps: Sequence[int],
                stride: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nominal = results["nominal"]["overall"]["mae_deg"]
    floor = floors.get("repeat_command", {}).get("overall", {}).get("mae_deg")
    ms_per_row = 1000.0 * stride / 30.0

    # 1. degradation vs staleness, split by modality -- the slope here is the
    # per-modality latency budget, which is the number that decides where to
    # spend engineering effort.
    fig, ax = plt.subplots(figsize=(8, 5))
    families = {"vision": "vis_delay", "tactile": "tac_delay", "state": "state_delay",
                "all": "all_delay"}
    for name, prefix in families.items():
        xs, ys = [0.0], [nominal]
        for label, report in results.items():
            if label.startswith(prefix + "_"):
                xs.append(int(label.rsplit("_", 1)[1]) * ms_per_row)
                ys.append(report["overall"]["mae_deg"])
        if len(xs) > 1:
            order = np.argsort(xs)
            ax.plot(np.array(xs)[order], np.array(ys)[order],
                    marker="o", ms=4, label=name)
    if floor is not None:
        ax.axhline(floor, ls="--", color="gray", lw=1,
                   label=f"repeat_command floor ({floor:.2f} deg)")
    ax.axhline(nominal, ls=":", color="black", lw=1, label="nominal")
    ax.set_xlabel(f"observation staleness (ms; 1 row = {ms_per_row:.0f} ms)")
    ax.set_ylabel("MAE (deg)")
    ax.set_title("Sensitivity to observation staleness, by modality")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "delay_sensitivity.png"), dpi=150)
    plt.close(fig)

    # 2. executed-prefix curves -- error as a function of how much of the chunk
    # the harness actually commands.
    fig, ax = plt.subplots(figsize=(9, 5))
    show = [k for k in ("nominal", "all_delay_1", "all_delay_2", "hold_obs_4",
                        "tac_hist_frozen", "tac_drop", "wrist_blank")
            if k in results]
    for label in show:
        curve = results[label].get("executed_prefix_mae_deg", {})
        if curve:
            hs = sorted(int(h) for h in curve)
            ax.plot(hs, [curve[str(h)] for h in hs], marker="o", ms=3, label=label)
    for label, report in floors.items():
        curve = report.get("executed_prefix_mae_deg", {})
        if curve:
            hs = sorted(int(h) for h in curve)
            ax.plot(hs, [curve[str(h)] for h in hs], ls="--", lw=1, label=label)
    ax.set_xlabel("executed prefix length h (chunk steps commanded before replan)")
    ax.set_ylabel("MAE over steps 0..h-1 (deg)")
    ax.set_title("Error vs. open-loop execution length")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "executed_prefix.png"), dpi=150)
    plt.close(fig)

    # 3. every configuration as a bar, against the nominal and floor lines
    fig, ax = plt.subplots(figsize=(12, 5))
    labels = [k for k in results if k != "nominal"]
    values = [results[k]["overall"]["mae_deg"] for k in labels]
    order = np.argsort(values)[::-1]
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    ax.bar(np.arange(len(labels)), values, color="C0")
    ax.axhline(nominal, ls=":", color="black", lw=1.2,
               label=f"nominal ({nominal:.2f} deg)")
    if floor is not None:
        ax.axhline(floor, ls="--", color="gray", lw=1.2,
                   label=f"repeat_command floor ({floor:.2f} deg)")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MAE (deg)")
    ax.set_title("Every deployment condition, ranked")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "condition_ranking.png"), dpi=150)
    plt.close(fig)

    # 4. receding horizon: stitched error and seam jump vs replan period
    if stitch:
        entries = [stitch["by_replan_period"][k]
                   for k in sorted(stitch["by_replan_period"], key=int)]
        gaps = [e["gap_ms"] for e in entries]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        ax1.plot(gaps, [e["stitched_mae_deg"] for e in entries], marker="o", ms=4)
        ax1.set_xlabel("replan period (ms)")
        ax1.set_ylabel("MAE over executed trajectory (deg)")
        ax1.set_title("Stitched receding-horizon error")
        ax1.grid(alpha=0.3)
        seams = [(e["gap_ms"], e["seam"]) for e in entries if "seam" in e]
        if seams:
            ax2.plot([g for g, _ in seams], [s["mean_abs_deg"] for _, s in seams],
                     marker="o", ms=4, label="mean")
            ax2.plot([g for g, _ in seams], [s["p95_abs_deg"] for _, s in seams],
                     marker="s", ms=4, label="p95")
            ax2.set_xlabel("replan period (ms)")
            ax2.set_ylabel("seam jump (deg)")
            ax2.set_title("Chunk-to-chunk disagreement at the replan seam\n"
                          "(teleop is exactly 0 here)")
            ax2.grid(alpha=0.3)
            ax2.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "receding_horizon.png"), dpi=150)
        plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--origami_root", required=True,
                        help="held-out split; use a stride-5 or finer one")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--base_model_path", default="")
    parser.add_argument("--stats_path", default="")
    parser.add_argument("--dataset_name", default="")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--robustness_samples", type=int, default=400,
                        help="shared across every condition; each condition is "
                             "a full forward pass, so this multiplies")
    parser.add_argument("--configs", nargs="*", default=[],
                        help="subset of condition names (default: all)")
    parser.add_argument("--stitch_rows", type=int, default=240,
                        help="contiguous rows of the first held-out episode for "
                             "the receding-horizon pass (0 to skip)")
    parser.add_argument("--urdf",
                        default="/home/kshitij/origami_trex/north_poc2_2_urdf_usd/"
                                "north_poc2_2_v3_1.urdf")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cascaded_total_steps", type=int, default=10)
    parser.add_argument("--cascaded_split_step", type=int, default=6)
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # ── model (same reconstruction path as serving) ────────────────────────────
    spec = importlib.util.spec_from_file_location(
        "trex_test_server", os.path.join(_SCRIPT_DIR, "test.py"))
    test_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_mod)

    from types import SimpleNamespace
    load_args = SimpleNamespace(
        checkpoint_path=args.checkpoint_path, base_model_path=args.base_model_path,
        stats_path=args.stats_path, dataset_name=args.dataset_name,
        action_dim=65, action_chunk=25,
        use_robot_state=1, use_tactile_vec=1, use_tactile_deform=1,
        tactile_intermediate_size=0, n_flare_tokens_per_frame=0, n_flare_steps=0,
        use_tactile_code=0, vqvae_codebook_size=64,
        use_tactile_vqvae=0, vqvae_config=None,
        cascaded_total_steps=args.cascaded_total_steps,
        cascaded_split_step=args.cascaded_split_step)
    saved_path = os.path.join(args.checkpoint_path, "training_args.json")
    saved = {}
    if os.path.exists(saved_path):
        with open(saved_path) as handle:
            saved = json.load(handle)
        for key in ("action_dim", "action_chunk", "use_robot_state",
                    "use_tactile_deform", "use_tactile_vec"):
            if key in saved:
                setattr(load_args, key, int(saved[key]))
    model, processor, _ = test_mod.model_load(load_args)
    model = model.to(device).eval()

    # ── data ──────────────────────────────────────────────────────────────────
    config = _EVAL.dataset_config_from_checkpoint(args.checkpoint_path, {})
    dataset = OrigamiDataset(config, processor, _EVAL._Printer(),
                             root=args.origami_root)
    if not dataset.use_tactile_vqvae:
        print("NOTE: use_tactile_vqvae=0 for this checkpoint, so only the *last* "
              "frame of the F6 window reaches the model (origami_dataset.py:390). "
              "tac_hist_stride_* and tac_hist_frozen are no-ops here and their "
              "numbers should be read as a sanity check, not a result.")
    frozen = frozen_action_dims(dataset.action_mask)
    if frozen.size:
        print(f"frozen action dims {frozen.tolist()} -> held at state[j], "
              f"matching eval_origami.py and the serve path")
    print(f"action anchoring: {describe_anchor(dataset.anchor_spec)}")
    _EVAL.check_anchor_consistency(args.checkpoint_path, dataset)

    ms_per_row = 1000.0 * dataset.sample_stride / 30.0
    print(f"sample_stride={dataset.sample_stride} -> one row of delay is "
          f"{ms_per_row:.0f} ms; delays and replan periods are quantized to that")
    if dataset.sample_stride > 5:
        print("WARNING: this split is too coarse for a meaningful delay sweep. "
              "Use the stride-5 pilot val, or re-prepare with --sample-stride 1.")

    all_configs = default_configs()
    chosen = args.configs or list(all_configs)
    unknown = [c for c in chosen if c not in all_configs]
    if unknown:
        raise SystemExit(f"unknown configs {unknown}; "
                         f"available: {sorted(all_configs)}")
    if "nominal" not in chosen:
        chosen = ["nominal"] + chosen           # every delta is relative to it
    specs = [all_configs[c] for c in chosen]

    max_delay = max([s.delay_rows for s in specs]
                    + [s.hold_rows - 1 for s in specs if s.hold_rows > 1] + [0])
    indices = eligible_indices(dataset, max_delay, args.robustness_samples)
    print(f"{len(indices)} samples per condition x {len(specs)} conditions "
          f"(rows >= {max_delay} into their episode, so no delay is silently "
          f"clamped at an episode boundary)")

    safety = None
    if os.path.exists(args.urdf):
        safety = SafetyChecker(args.urdf)
    else:
        print(f"WARNING: URDF not found at {args.urdf}; skipping safety checks")

    horizon = dataset.action_chunk
    prefix_steps = [h for h in (1, 2, 3, 5, 8, 10, 15, 20, 25) if h <= horizon]

    # ── sweep ─────────────────────────────────────────────────────────────────
    floor_accs = {name: ErrorAccumulator(horizon, dataset.action_dim)
                  for name in ("hold_state", "repeat_command", "oracle_prev_command")}
    results: Dict[str, dict] = {}
    for i, spec_i in enumerate(specs):
        print(f"[{i + 1}/{len(specs)}] {spec_i.label}")
        acc = run_config(model, dataset, spec_i, indices, args, device, safety,
                         floors=floor_accs if spec_i.is_nominal else None)
        report = acc.report()
        report["executed_prefix_mae_deg"] = prefix_curve(acc, prefix_steps)
        report["spec"] = dict(spec_i.__dict__)
        report["spec"]["modalities"] = list(spec_i.modalities)
        results[spec_i.label] = report

    floors = {}
    for name, acc in floor_accs.items():
        if acc.n:
            floors[name] = acc.report()
            floors[name]["executed_prefix_mae_deg"] = prefix_curve(acc, prefix_steps)

    nominal_mae = results["nominal"]["overall"]["mae_deg"]
    for label, report in results.items():
        report["delta_vs_nominal_deg"] = report["overall"]["mae_deg"] - nominal_mae
        curve = report["executed_prefix_mae_deg"]
        base = results["nominal"]["executed_prefix_mae_deg"]
        report["delta_vs_nominal_prefix_deg"] = {
            h: curve[h] - base[h] for h in curve}

    # ── receding horizon ──────────────────────────────────────────────────────
    stitch = receding_horizon(model, dataset, args, device) if args.stitch_rows else None

    # ── write ─────────────────────────────────────────────────────────────────
    payload = {
        "checkpoint": args.checkpoint_path,
        "data_root": args.origami_root,
        "sample_stride": dataset.sample_stride,
        "chunk_stride": dataset.chunk_stride,
        "ms_per_row_of_delay": ms_per_row,
        "n_samples_per_condition": len(indices),
        "action_chunk": horizon,
        "action_dim": dataset.action_dim,
        "use_tactile_vqvae": int(bool(dataset.use_tactile_vqvae)),
        "cascaded_total_steps": args.cascaded_total_steps,
        "cascaded_split_step": args.cascaded_split_step,
        "action_anchor": list(dataset.anchor_spec),
        "prefix_steps": prefix_steps,
        "conditions": results,
        "floors": floors,
        "receding_horizon": stitch,
    }
    with open(os.path.join(args.out_dir, "robustness.json"), "w") as handle:
        json.dump(payload, handle, indent=2)
    write_plots(args.out_dir, results, floors, stitch, prefix_steps,
                dataset.sample_stride)

    # ── console ───────────────────────────────────────────────────────────────
    def unsafe(report):
        rates = report.get("safety", {}).get("violation_rate_per_value")
        return float("nan") if rates is None else 100 * sum(rates.values())

    print(f"\n{'condition':<20} {'MAE(deg)':>9} {'d(nom)':>8} {'MAE@h=5':>9} "
          f"{'d(nom)':>8} {'MAE@h=25':>9} {'contact':>9} {'unsafe%':>8}")
    print("-" * 88)
    order = sorted(results, key=lambda k: results[k]["overall"]["mae_deg"])
    for label in order:
        report = results[label]
        overall = report["overall"]
        pre = report["executed_prefix_mae_deg"]
        print(f"{label:<20} {overall['mae_deg']:>9.3f} "
              f"{report['delta_vs_nominal_deg']:>+8.3f} "
              f"{pre.get('5', float('nan')):>9.3f} "
              f"{report['delta_vs_nominal_prefix_deg'].get('5', float('nan')):>+8.3f} "
              f"{pre.get(str(horizon), float('nan')):>9.3f} "
              f"{report.get('contact', {}).get('mae_deg', float('nan')):>9.3f} "
              f"{unsafe(report):>8.3f}")
    for name, report in floors.items():
        print(f"{name:<20} {report['overall']['mae_deg']:>9.3f} "
              f"{'':>8} {report['executed_prefix_mae_deg'].get('5', float('nan')):>9.3f} "
              f"{'':>8} {report['executed_prefix_mae_deg'].get(str(horizon), float('nan')):>9.3f}"
              f"{'':>10} {'  <- floor':>8}")

    if stitch:
        print(f"\nreceding horizon over {stitch['n_rows']} contiguous rows of "
              f"{stitch['episode']}")
        print(f"{'replan':>8} {'executed':>9} {'stitched':>10} {'seam mean':>10} "
              f"{'seam p95':>9} {'seam jump':>10}")
        print(f"{'(ms)':>8} {'steps':>9} {'MAE(deg)':>10} {'(deg)':>10} "
              f"{'(deg)':>9} {'viol.rate':>10}")
        print("-" * 62)
        for key in sorted(stitch["by_replan_period"], key=int):
            e = stitch["by_replan_period"][key]
            s = e.get("seam", {})
            print(f"{e['gap_ms']:>8.0f} {e['executed_steps_per_chunk']:>9d} "
                  f"{e['stitched_mae_deg']:>10.3f} "
                  f"{s.get('mean_abs_deg', float('nan')):>10.3f} "
                  f"{s.get('p95_abs_deg', float('nan')):>9.3f} "
                  f"{s.get('step_jump_violation_rate', float('nan')):>10.5f}")
        print("teleop's own seam is exactly 0, so all of the above is the policy "
              "disagreeing with its own previous chunk.")

    n_plots = len([f for f in os.listdir(args.out_dir) if f.endswith(".png")])
    print(f"\nwrote {args.out_dir}/robustness.json and {n_plots} plots")
    return 0


if __name__ == "__main__":
    sys.exit(main())
