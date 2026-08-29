"""Pre-launch checks for a long origami post-training run.

A full-tier run is ~16 h/epoch.  Every failure this script catches is one that
does not surface until hours in, or -- worse -- does not surface at all and
quietly produces a checkpoint calibrated to the wrong normalisation.

Checks:

  split containment   The pilot's val seasons must still be inside the full
                      run's val split, or the full run's numbers cannot be
                      compared against the pilot checkpoint at all.  Guaranteed
                      by `select_seasons` truncating a fixed list by prefix, but
                      it is one edited list away from silently not holding, and
                      the cost of finding out afterwards is the whole run.

  anchoring           What the 65 outputs are measured from, per dim.  Train
                      and val must agree, the parquets must actually carry the
                      columns the rule needs, and a warm start must not put a
                      head trained under one rule on data prepared under
                      another -- that failure has no shape mismatch to catch it
                      and shows up only as a constant bias in every metric.

  frozen dims         Which action dims the refit stats declared frozen.  These
                      are the dims eval/serve hold at the measured state, so if
                      the torso turns out to move in one of the 91 seasons the
                      pilot never saw, the clamp must not be applied and this is
                      where that shows up.

  resume calibration  The action head's output scale is defined by the q01/q99
                      the model trained against.  Warm-starting the full run
                      from the pilot checkpoint while the full split refits its
                      own stats silently mis-scales every action dim.  Two cases
                      are fine and are not flagged: resuming *the same run*
                      after a pre-emption, and cold-starting from the midtrain
                      checkpoint, whose stats are eef-62 -- a different action
                      space means the head is re-initialised on load, so there
                      is no old normalisation to carry over.

  budget              samples -> micro-steps -> wall clock, so the checkpoint
                      cadence and the session count are chosen from a number
                      rather than from optimism.

Usage:
    python -m trex_origami.preflight \\
        --train-root /content/data/origami_flat/full/train \\
        --val-root   /content/data/origami_flat/full/val \\
        --resume-checkpoint /content/assets/trex_midtrain \\
        --batch-size 16 --grad-accum 4 --epochs 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional, Sequence

import numpy as np

from .anchoring import (ANCHOR_PREV_COMMAND, describe as describe_anchor,
                        masks as anchor_masks, mode_of, spec_from_meta)
from .seasons import JOINT_NAMES, select_seasons

# `stats.MIN_NORM_RANGE_JOINT`, repeated so this stays runnable against a stats
# file written by a different version of the prep.
MIN_NORM_RANGE_JOINT = 1e-3

# Steady-state seconds per micro-step measured on the pilot run (A100-40GB,
# batch 16, gradient checkpointing on, 3 images/sample at 224px).  Only used for
# the budget estimate; override with --sec-per-step on other hardware.
PILOT_SEC_PER_MICRO_STEP = 2.25


class Problems:
    def __init__(self) -> None:
        self.fatal: List[str] = []
        self.warn: List[str] = []

    def fail(self, message: str) -> None:
        self.fatal.append(message)
        print(f"  FAIL  {message}")

    def warning(self, message: str) -> None:
        self.warn.append(message)
        print(f"  WARN  {message}")

    @staticmethod
    def ok(message: str) -> None:
        print(f"  ok    {message}")


def _load_meta(root: str) -> Optional[dict]:
    path = os.path.join(root, "meta", "dataset.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def _seasons_of(root: str) -> Optional[List[str]]:
    meta = _load_meta(root)
    if meta is None:
        return None
    return sorted({e["season"] for e in meta["episodes"]})


def _action_block(path: str) -> Optional[dict]:
    """The `action` stats block from a norm_stats.json / stats_data.json."""
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        raw = json.load(handle)
    if not raw:
        return None
    block = raw[next(iter(raw))]
    return block.get("action")


# ── checks ────────────────────────────────────────────────────────────────────
def check_split_containment(val_root: str, pilot_val_limit: int,
                            problems: Problems) -> None:
    print("\n[1] split containment")
    seasons = _seasons_of(val_root)
    if seasons is None:
        problems.fail(f"{val_root} has no meta/dataset.json — not prepared yet")
        return
    pilot = select_seasons("val", pilot_val_limit)
    missing = [s for s in pilot if s not in seasons]
    if missing:
        problems.fail(
            f"val split is missing {len(missing)} of the pilot's {len(pilot)} val "
            f"seasons: {missing}. The full run's metrics would not be comparable "
            f"to the pilot checkpoint on the same held-out data.")
    else:
        problems.ok(f"all {len(pilot)} pilot val seasons present in the "
                    f"{len(seasons)}-season val split")
        for s in pilot:
            print(f"          {s}")

    train_seasons = _seasons_of(os.path.join(os.path.dirname(val_root), "train"))
    if train_seasons:
        leak = sorted(set(train_seasons) & set(seasons))
        if leak:
            problems.fail(f"{len(leak)} season(s) in BOTH train and val: {leak[:3]}")
        else:
            problems.ok(f"no season appears in both splits "
                        f"({len(train_seasons)} train / {len(seasons)} val)")


def check_anchoring(train_root: str, val_root: str, resume_checkpoint: str,
                    resume_full_state: bool, problems: Problems) -> None:
    print("\n[2] action anchoring")
    train_meta = _load_meta(train_root)
    if train_meta is None:
        problems.fail(f"{train_root} has no meta/dataset.json — not prepared yet")
        return
    spec = spec_from_meta(train_meta)
    problems.ok(f"train: {describe_anchor(spec)}")

    val_meta = _load_meta(val_root)
    if val_meta is not None:
        val_spec = spec_from_meta(val_meta)
        if tuple(val_spec) != tuple(spec):
            problems.fail(
                f"val is anchored '{mode_of(val_spec)}' but train is "
                f"'{mode_of(spec)}'. The validation loss would be measured "
                f"against targets in a different space than the model emits. "
                f"Re-prepare both splits with the same --anchor-mode.")
        else:
            problems.ok("val split declares the same anchoring")

    # The rule is only as good as the column it needs.  A `hybrid` dataset whose
    # parquets predate the `prev_command` column would raise inside the
    # dataloader -- but hours later, on a worker, mid-epoch.
    if anchor_masks(spec)[ANCHOR_PREV_COMMAND].any():
        import pyarrow.parquet as pq
        entry = train_meta["episodes"][0]
        path = os.path.join(train_root, entry["file"])
        if os.path.exists(path):
            names = set(pq.ParquetFile(path).schema_arrow.names)
            if "prev_command" not in names:
                problems.fail(
                    f"{entry['file']} has no `prev_command` column but the split "
                    f"anchors the arms to it — this data was written by an older "
                    f"trex_origami. Re-prepare it.")
            else:
                problems.ok("parquets carry the `prev_command` column the rule needs")

    # A checkpoint carries the anchoring its head was trained to emit.
    if not resume_checkpoint or resume_full_state:
        return
    path = os.path.join(resume_checkpoint, "training_args.json")
    if not os.path.exists(path):
        return
    with open(path) as handle:
        saved = json.load(handle)
    ckpt_anchor = saved.get("action_anchor")
    if not ckpt_anchor:
        # Pre-anchoring checkpoints and the released midtrain checkpoint (eef-62)
        # both land here; the action head is re-initialised on a dim change
        # anyway, and `check_resume` covers the same-action-space case.
        return
    if tuple(ckpt_anchor) != tuple(spec):
        problems.fail(
            f"{resume_checkpoint} was trained with anchoring "
            f"'{mode_of(ckpt_anchor)}' but this data is '{mode_of(spec)}'. Its "
            f"action head emits numbers measured from something else, and nothing "
            f"downstream would raise — every arm joint would simply be off by the "
            f"anchor difference. Cold-start from trex_midtrain instead.")
    else:
        problems.ok(f"resume checkpoint's anchoring matches the data "
                    f"('{mode_of(spec)}')")


def check_frozen_dims(train_root: str, problems: Problems) -> np.ndarray:
    print("\n[3] frozen action dims (held at the measured state at inference)")
    block = _action_block(os.path.join(train_root, "meta", "norm_stats.json"))
    if block is None:
        problems.fail(f"{train_root} has no meta/norm_stats.json — run "
                      f"`python -m trex_origami.stats --root {train_root}`")
        return np.array([], dtype=int)

    mask = np.array(block["mask"], dtype=bool)
    spread = np.max(np.array(block["q99"]) - np.array(block["q01"]), axis=0)
    frozen = np.where(~mask)[0]

    if frozen.size == 0:
        problems.warning(
            "no frozen dims in these stats. The pilot froze lower_body_joint_1/2; "
            "if they move in the wider split that is a real behaviour change — "
            "the inference clamp will correctly do nothing, but the head now has "
            "to learn those dims.")
    else:
        problems.ok(f"{frozen.size} dim(s) frozen -> commanded at state[j] at inference")
        for i in frozen:
            print(f"          dim {i:2d}  {JOINT_NAMES[i]:<22} "
                  f"spread {spread[i]:.2e} rad ({np.degrees(spread[i]):.4f} deg)")

    near = [i for i in np.where(mask)[0] if spread[i] < 10 * MIN_NORM_RANGE_JOINT]
    for i in near:
        problems.warning(
            f"dim {i} {JOINT_NAMES[i]} is NOT frozen but its spread is only "
            f"{spread[i]:.2e} rad — within 10x of the threshold, so the clamp "
            f"decision for this joint is one season away from flipping.")
    return frozen


def check_resume(resume_checkpoint: str, train_root: str, resume_full_state: bool,
                 problems: Problems) -> None:
    print("\n[4] resume calibration")
    if not resume_checkpoint:
        problems.warning("no --resume-checkpoint given; skipping")
        return
    if resume_full_state:
        problems.ok("--resume-full-state: continuing the same run after a "
                    "pre-emption, so the stats are the ones it trained on")
        return

    ckpt_block = _action_block(os.path.join(resume_checkpoint, "stats_data.json"))
    if ckpt_block is None:
        problems.ok(f"{resume_checkpoint} carries no action stats — nothing the "
                    f"action head could be mis-calibrated against")
        return

    data_block = _action_block(os.path.join(train_root, "meta", "norm_stats.json"))
    if data_block is None:
        problems.fail(f"{train_root} has no norm_stats.json to compare against")
        return

    ckpt_q01, ckpt_q99 = np.array(ckpt_block["q01"]), np.array(ckpt_block["q99"])
    data_q01, data_q99 = np.array(data_block["q01"]), np.array(data_block["q99"])

    # A different action *dim* is the signature of the intended cold start: the
    # released midtrain checkpoint ships its own stats_data.json fitted on
    # T-Rex's eef-62 / chunk-16 corpus, and train.py drops the shape-mismatched
    # x_embedder / final_layer / final_layer_tactile / state_embedder keys, so
    # the head is re-initialised and calibrated against *these* stats.  There is
    # no old normalisation left to mis-scale.
    if ckpt_q01.shape[-1] != data_q01.shape[-1]:
        problems.ok(
            f"checkpoint's action stats are {tuple(ckpt_q01.shape)} (chunk x dim) vs "
            f"the data's {tuple(data_q01.shape)} — a different action space, so the "
            f"action head is re-initialised on load. This is the intended cold start "
            f"from the midtrain checkpoint")
        return

    scale_ckpt = np.maximum(ckpt_q99 - ckpt_q01, 1e-12)
    scale_data = np.maximum(data_q99 - data_q01, 1e-12)
    cell = "chunk step {}"
    if ckpt_q01.shape != data_q01.shape:
        # Same action space, different chunk length.  No weight depends on chunk
        # length, so the head *does* transfer and its calibration still matters;
        # compare the per-dim envelope instead of cell-wise.
        problems.warning(
            f"checkpoint chunk length {ckpt_q01.shape[0]} != the data's "
            f"{data_q01.shape[0]}; comparing the per-dim envelope, not cell-wise")
        scale_ckpt = scale_ckpt.max(axis=0, keepdims=True)
        scale_data = scale_data.max(axis=0, keepdims=True)
        cell = "per-dim envelope, {}"
    ratio = scale_data / scale_ckpt
    # Worst dim by |log ratio| so a 2x shrink is ranked as badly as a 2x stretch.
    # Report the ratio *at that cell*, not the global max: the extremes usually
    # sit on different dims, and quoting max(ratio) next to argmax(|log ratio|)
    # can print "1.00x" for a dim that actually halved.
    where = np.unravel_index(int(np.argmax(np.abs(np.log(ratio)))), ratio.shape)
    if float(np.max(np.abs(np.log(ratio)))) < np.log(1.02):
        problems.ok("resume checkpoint's action stats match the training data "
                    "(within 2% on every dim) — warm start is calibrated")
        return

    dim = int(where[-1])
    problems.fail(
        f"resume checkpoint was calibrated to DIFFERENT action stats: worst dim is "
        f"{dim} ({JOINT_NAMES[dim]}), rescaled by {float(ratio[where]):.3f}x "
        f"({cell.format(int(where[0]))}). "
        f"Its action head emits in the old normalisation, so every predicted delta "
        f"would be mis-scaled. Cold-start from trex_midtrain instead, or re-fit "
        f"stats to match. Override with ALLOW_STATS_MISMATCH=1 if this is "
        f"deliberate and you are watching the first 500 steps.")


def check_budget(train_root: str, batch_size: int, grad_accum: int, epochs: int,
                 sec_per_step: float, save_steps: int, problems: Problems) -> None:
    print("\n[5] budget")
    meta = _load_meta(train_root)
    if meta is None:
        problems.warning(f"{train_root} not prepared; skipping budget estimate")
        return
    n_samples = int(meta.get("n_samples")
                    or sum(int(e["n_samples"]) for e in meta["episodes"]))
    if not n_samples:
        problems.warning("dataset.json has no sample counts; "
                         "read the sample count off the trainer's startup line")
        return
    stride = int(meta.get("config", {}).get("sample_stride", 0))
    print(f"          {meta.get('n_seasons', '?')} seasons / "
          f"{meta.get('n_episodes', len(meta['episodes']))} episodes"
          + (f" / sample stride {stride} ({30.0 / stride:.1f} Hz)" if stride else ""))

    micro_per_epoch = -(-n_samples // batch_size)
    hours_per_epoch = micro_per_epoch * sec_per_step / 3600
    print(f"          {n_samples:,} samples / batch {batch_size} = "
          f"{micro_per_epoch:,} micro-steps per epoch")
    print(f"          {micro_per_epoch // grad_accum:,} optimizer steps per epoch "
          f"(grad accum {grad_accum}, effective batch {batch_size * grad_accum})")
    print(f"          ~{hours_per_epoch:.1f} h/epoch at {sec_per_step:.2f} s/micro-step "
          f"-> ~{hours_per_epoch * epochs:.1f} h for {epochs} epochs")
    ckpts = micro_per_epoch // max(1, save_steps)
    print(f"          --save_steps {save_steps} -> ~{ckpts} checkpoints/epoch, "
          f"one every ~{save_steps * sec_per_step / 3600:.1f} h")
    if hours_per_epoch > 10:
        problems.warning(
            f"a {hours_per_epoch:.0f} h epoch will not fit in one Colab session. "
            f"Mid-epoch resume must work, or every pre-emption replays the head of "
            f"the epoch while the LR schedule keeps advancing.")
    if ckpts > 20:
        problems.warning(f"~{ckpts} checkpoints/epoch is a lot of I/O with "
                         f"--save_optimizer_state 1; consider a larger --save_steps")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--resume-full-state", action="store_true",
                        help="set when continuing the same run after a pre-emption")
    parser.add_argument("--pilot-val-limit", type=int, default=3,
                        help="how many val seasons the pilot tier used")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--save-steps", type=int, default=5000)
    parser.add_argument("--sec-per-step", type=float, default=PILOT_SEC_PER_MICRO_STEP)
    args = parser.parse_args(argv)

    print(f"pre-flight: train={args.train_root}\n            val={args.val_root}")
    problems = Problems()
    check_split_containment(args.val_root, args.pilot_val_limit, problems)
    check_anchoring(args.train_root, args.val_root, args.resume_checkpoint,
                    args.resume_full_state, problems)
    check_frozen_dims(args.train_root, problems)
    check_resume(args.resume_checkpoint, args.train_root,
                 args.resume_full_state, problems)
    check_budget(args.train_root, args.batch_size, args.grad_accum, args.epochs,
                 args.sec_per_step, args.save_steps, problems)

    print()
    if problems.fatal:
        print(f"pre-flight FAILED: {len(problems.fatal)} blocking problem(s), "
              f"{len(problems.warn)} warning(s)")
        for message in problems.fatal:
            print(f"  - {message}")
        return 1
    print(f"pre-flight passed with {len(problems.warn)} warning(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
