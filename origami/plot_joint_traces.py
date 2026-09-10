"""Plot predicted vs. ground-truth hand-joint-angle traces for one held-out-season
(val) episode. Companion to ``eval_offline.py`` (step 10b) -- same checkpoint load path
and the same real cascaded ``CascadedServer`` inference, but walked in temporal order
over a single episode instead of i.i.d-sampled frames, so the result is a trace you can
look at rather than an aggregate metric.

Only the ``*_hand22`` action blocks are plotted: per ``eval_offline.py``'s
``ACTION_BLOCKS``/``hold_position_prediction``, those 22+22 dims are the model's
*absolute* target joint angle per hand (radians, HAND_ORDER, REDESIGN_PLAN.md §1.1) --
unlike ``*_trans3``/``*_rot6d6`` (EEF pose deltas), they are directly "joint angles".

For each sampled frame we call ``CascadedServer.predict("slow_and_fast", ...)`` fresh
(independent per call -- see ``scripts/test.py::CascadedServer.predict``, "slow_and_fast"
runs slow+fast in one shot off the payload's own dense tactile window, so no cross-frame
server state leaks between samples) and take chunk step k=0, the model's immediate next-
action prediction, against the dataset's real ``action[0]`` ground truth.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TREX_DIR = _REPO_ROOT / "T-Rex"
if str(_TREX_DIR) not in sys.path:
    sys.path.insert(0, str(_TREX_DIR))

import origami.trex_patch as _trex_patch  # noqa: E402

_trex_patch.apply()

from utils.lerobot_common import KEY_ACTION  # noqa: E402
from origami.constants import HAND_ORDER  # noqa: E402
from origami.eval_offline import load_model_and_stats, open_eval_dataset, run_config  # noqa: E402

L_HAND_SLICE = slice(9, 31)
R_HAND_SLICE = slice(40, 62)


def episode_frame_range(root: str, episode_index: int) -> tuple[int, int]:
    import pandas as pd

    ep_files = sorted((Path(root) / "meta" / "episodes").rglob("*.parquet"))
    for f in ep_files:
        df = pd.read_parquet(f)
        row = df[df["episode_index"] == episode_index]
        if len(row):
            r = row.iloc[0]
            return int(r["dataset_from_index"]), int(r["dataset_to_index"])
    raise ValueError(f"episode {episode_index} not found under {root}/meta/episodes")


def collect_traces(root: str, checkpoint: str, episode_index: int, stride: int,
                    max_frames: int, config: str = "cascaded") -> dict:
    args, model, processor, statistic = load_model_and_stats(checkpoint)
    vqvae_window = int(args.vqvae_config["window"]) if getattr(args, "use_tactile_vqvae", 0) else 16
    ds = open_eval_dataset(root, vqvae_window)

    lo, hi = episode_frame_range(root, episode_index)
    idx = np.arange(lo, hi, stride)
    if max_frames and len(idx) > max_frames:
        idx = idx[:max_frames]
    logger.info("episode %d: frames [%d, %d), sampling %d frames (stride=%d)",
                episode_index, lo, hi, len(idx), stride)

    from scripts.test import CascadedServer
    server = CascadedServer(args, model, processor, statistic)

    pred_L, pred_R, gt_L, gt_R, t = [], [], [], [], []
    for i in idx:
        item = ds[int(i)]
        action = run_config(config, server, item, bool(args.use_robot_state),
                             int(args.action_chunk))  # [action_chunk, 62]
        gt = np.asarray(item[KEY_ACTION], dtype=np.float32)      # [action_chunk, 62]
        pred_L.append(action[0, L_HAND_SLICE])
        pred_R.append(action[0, R_HAND_SLICE])
        gt_L.append(gt[0, L_HAND_SLICE])
        gt_R.append(gt[0, R_HAND_SLICE])
        t.append((int(i) - lo) / 30.0)

    return {
        "t": np.asarray(t),
        "pred_L": np.stack(pred_L), "pred_R": np.stack(pred_R),
        "gt_L": np.stack(gt_L), "gt_R": np.stack(gt_R),
        "episode_index": episode_index, "config": config, "checkpoint": checkpoint,
    }


def plot_hand(t, pred, gt, hand_label: str, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(HAND_ORDER)
    ncols = 4
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.4 * nrows), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    for j, name in enumerate(HAND_ORDER):
        ax = axes[j]
        ax.plot(t, gt[:, j], color="tab:blue", label="ground truth", linewidth=1.5)
        ax.plot(t, pred[:, j], color="tab:orange", label="prediction", linewidth=1.2,
                 linestyle="--")
        mae = float(np.mean(np.abs(pred[:, j] - gt[:, j])))
        ax.set_title(f"{name}  (MAE={mae:.3f})", fontsize=9)
        ax.tick_params(labelsize=7)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.supxlabel("time (s)")
    fig.supylabel("joint angle (rad)")
    fig.suptitle(f"{hand_label} hand -- predicted vs. ground-truth joint angles")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info("wrote %s", out_path)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, help="Held-out (val) merged origami dataset root.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episode", type=int, default=0, help="episode_index within --root.")
    p.add_argument("--stride", type=int, default=15, help="Frames between sampled points.")
    p.add_argument("--max-frames", type=int, default=200)
    p.add_argument("--config", default="cascaded",
                    choices=("cascaded", "disable_tactile", "tactile_zeroed", "hold_position"))
    p.add_argument("--output-dir", default=".")
    args = p.parse_args(argv)

    traces = collect_traces(args.root, args.checkpoint, args.episode, args.stride,
                             args.max_frames, args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_hand(traces["t"], traces["pred_L"], traces["gt_L"], "Left",
              out_dir / f"ep{args.episode}_{args.config}_left_hand.png")
    plot_hand(traces["t"], traces["pred_R"], traces["gt_R"], "Right",
              out_dir / f"ep{args.episode}_{args.config}_right_hand.png")

    np.savez(out_dir / f"ep{args.episode}_{args.config}_traces.npz",
              t=traces["t"], pred_L=traces["pred_L"], pred_R=traces["pred_R"],
              gt_L=traces["gt_L"], gt_R=traces["gt_R"])
    logger.info("wrote %s", out_dir / f"ep{args.episode}_{args.config}_traces.npz")


if __name__ == "__main__":
    main()
