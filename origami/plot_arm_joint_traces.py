"""Plot predicted vs. ground-truth ARM joint angles (after IK), in real joint space --
not eef62 space. Companion to ``plot_joint_traces.py`` (hand22, which IS already joint
space) and a stopgap for the not-yet-built ``retarget.py`` (REDESIGN_PLAN.md §12 step 13 /
eval_offline.py's §8.1-C): this script does the IK itself using ``origami.kinematics
.OrigamiKinematics`` (the same Pink IK ``solve_ik`` used at prep/deploy time, §3.1).

Why IK is needed at all: the dataset only stores the eef62 representation (absolute or
chunk-delta translation + 6D rotation per arm, HAND_ORDER for hands) -- REAL arm joint
angles were never baked into ``observation.state``/``action``. So both traces here are
themselves computed, not read off disk:

  * ground truth arm7  = solve_ik(target = the dataset's own ``action_abs`` EEF pose)
  * predicted arm7     = solve_ik(target = base_pose (from ``observation.state``) composed
                          with the model's predicted chunk-k=0 delta9, via
                          ``kinematics.delta9_to_matrix`` -- the inverse of the delta-base
                          encoding ``build_action_chunk`` used to build the training target)

Both IK solves warm-start from their own trace's previous solution (falling back to the
kinematics model's ``default_qpos`` on the first frame of the episode) -- an redundant
7-DOF arm has a 1-DOF pose-preserving null space, so the "ground truth" arm7 here is one
representative joint solution consistent with the true EEF pose, not necessarily the
teleoperator's actual joint angles at capture time.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TREX_DIR = _REPO_ROOT / "T-Rex"
if str(_TREX_DIR) not in sys.path:
    sys.path.insert(0, str(_TREX_DIR))

import origami.trex_patch as _trex_patch  # noqa: E402

_trex_patch.apply()

from utils.lerobot_common import KEY_ACTION_ABS, KEY_STATE  # noqa: E402
from origami.eval_offline import load_model_and_stats, open_eval_dataset, run_config  # noqa: E402
from origami.kinematics import (  # noqa: E402
    ARM_JOINT_NAMES_14,
    LockedConfig,
    OrigamiKinematics,
    delta9_to_matrix,
    rot6d_to_matrix,
)
from origami.plot_joint_traces import episode_frame_range  # noqa: E402

L_ARM_JOINT_NAMES = ARM_JOINT_NAMES_14[:7]
R_ARM_JOINT_NAMES = ARM_JOINT_NAMES_14[7:]

L_POSE9_SLICE = slice(0, 9)
R_POSE9_SLICE = slice(31, 40)


def load_locked_config(root: str) -> LockedConfig:
    """The exact lock used at prep time (REDESIGN_PLAN.md §3.2/§3.3), so IK targets the
    same reduced 14-DOF model the dataset's own EEF poses were generated against --
    ``meta/origami_prep.json``'s ``locked_config`` (validated against ``training_args
    .json``'s ``locked_config.digest`` by ``load_model_and_stats``'s caller if needed)."""
    import json
    d = json.load(open(Path(root) / "meta" / "origami_prep.json"))["locked_config"]
    return LockedConfig(
        lower_body=np.asarray(d["lower_body"], dtype=np.float64),
        neck=np.asarray(d["neck"], dtype=np.float64),
        left_hand=np.asarray(d["left_hand"], dtype=np.float64),
        right_hand=np.asarray(d["right_hand"], dtype=np.float64),
    )


def pose9d_to_matrix(pose9: np.ndarray) -> np.ndarray:
    """trans3(3) + rot6d(6) absolute pose -> 4x4 homogeneous matrix."""
    out = np.eye(4)
    out[:3, :3] = rot6d_to_matrix(pose9[3:9])
    out[:3, 3] = pose9[0:3]
    return out


def collect_arm_traces(root: str, checkpoint: str, episode_index: int, stride: int,
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

    kin = OrigamiKinematics(locked=load_locked_config(root))
    warm_gt_l, warm_gt_r = kin._disassemble(kin.default_qpos)
    warm_pred_l, warm_pred_r = kin._disassemble(kin.default_qpos)

    from scripts.test import CascadedServer
    server = CascadedServer(args, model, processor, statistic)

    gt_L, gt_R, pred_L, pred_R, t = [], [], [], [], []
    for i in idx:
        item = ds[int(i)]
        state = np.asarray(item[KEY_STATE], dtype=np.float64)
        action_abs = np.asarray(item[KEY_ACTION_ABS], dtype=np.float64)
        action = run_config(config, server, item, bool(args.use_robot_state),
                             int(args.action_chunk)).astype(np.float64)  # [chunk, 62]

        gt_target_l = pin.SE3(pose9d_to_matrix(action_abs[L_POSE9_SLICE]))
        gt_target_r = pin.SE3(pose9d_to_matrix(action_abs[R_POSE9_SLICE]))
        warm_gt_l, warm_gt_r = kin.solve_ik(gt_target_l, gt_target_r, warm_gt_l, warm_gt_r)

        base_l = pose9d_to_matrix(state[L_POSE9_SLICE])
        base_r = pose9d_to_matrix(state[R_POSE9_SLICE])
        pred_target_l = pin.SE3(delta9_to_matrix(action[0, L_POSE9_SLICE], base_l))
        pred_target_r = pin.SE3(delta9_to_matrix(action[0, R_POSE9_SLICE], base_r))
        warm_pred_l, warm_pred_r = kin.solve_ik(pred_target_l, pred_target_r,
                                                 warm_pred_l, warm_pred_r)

        gt_L.append(warm_gt_l.copy()); gt_R.append(warm_gt_r.copy())
        pred_L.append(warm_pred_l.copy()); pred_R.append(warm_pred_r.copy())
        t.append((int(i) - lo) / 30.0)

    return {
        "t": np.asarray(t),
        "gt_L": np.stack(gt_L), "gt_R": np.stack(gt_R),
        "pred_L": np.stack(pred_L), "pred_R": np.stack(pred_R),
        "episode_index": episode_index, "config": config,
    }


def plot_arm(t, pred, gt, joint_names, arm_label: str, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(joint_names)
    ncols = 4
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.6 * nrows), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    for j, name in enumerate(joint_names):
        ax = axes[j]
        ax.plot(t, gt[:, j], color="tab:blue", label="ground truth (IK)", linewidth=1.5)
        ax.plot(t, pred[:, j], color="tab:orange", label="prediction (IK)", linewidth=1.2,
                 linestyle="--")
        mae = float(np.mean(np.abs(pred[:, j] - gt[:, j])))
        ax.set_title(f"{name}  (MAE={mae:.3f})", fontsize=9)
        ax.tick_params(labelsize=7)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.supxlabel("time (s)")
    fig.supylabel("joint angle (rad)")
    fig.suptitle(f"{arm_label} arm -- predicted vs. ground-truth joint angles (post-IK, joint space)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info("wrote %s", out_path)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, help="Held-out (val) merged origami dataset root.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--stride", type=int, default=15)
    p.add_argument("--max-frames", type=int, default=200)
    p.add_argument("--config", default="cascaded",
                    choices=("cascaded", "disable_tactile", "tactile_zeroed", "hold_position"))
    p.add_argument("--output-dir", default=".")
    args = p.parse_args(argv)

    traces = collect_arm_traces(args.root, args.checkpoint, args.episode, args.stride,
                                 args.max_frames, args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_arm(traces["t"], traces["pred_L"], traces["gt_L"], L_ARM_JOINT_NAMES, "Left",
              out_dir / f"ep{args.episode}_{args.config}_left_arm_joints.png")
    plot_arm(traces["t"], traces["pred_R"], traces["gt_R"], R_ARM_JOINT_NAMES, "Right",
              out_dir / f"ep{args.episode}_{args.config}_right_arm_joints.png")

    np.savez(out_dir / f"ep{args.episode}_{args.config}_arm_traces.npz",
              t=traces["t"], pred_L=traces["pred_L"], pred_R=traces["pred_R"],
              gt_L=traces["gt_L"], gt_R=traces["gt_R"])
    logger.info("wrote %s", out_dir / f"ep{args.episode}_{args.config}_arm_traces.npz")


if __name__ == "__main__":
    main()
