"""Held-out-season / zero-shot offline evaluation. REDESIGN_PLAN.md §8.1, §12 step 10b/14.

``--zero-shot`` points this at the **untrained-for-origami** T-Rex midtrain checkpoint
(§11.9 step 1) and measures the pretrain -> origami shift end-to-end: real cascaded
slow/fast inference (the exact deploy-time path, driven through ``scripts.test.CascadedServer``
rather than reimplemented) on real held-out origami frames, denormalized through the
checkpoint's OWN normalization stats (that's the honest zero-shot test -- the model was never
told about origami's action scale), compared to the real un-normalized ``action`` ground truth.

Implements §8.1-A (action-space accuracy: MSE/MAE/normalized-variance-share per chunk step k,
per block) and a version of §8.1-D (tactile-expert contribution) across four configs:
  * ``cascaded``           -- the deployed path (action expert + tactile expert)
  * ``disable_tactile``    -- action-expert-only ablation (``forward_flow_action_full``)
  * ``tactile_zeroed``     -- cascaded path, but F6/deform inputs zeroed at the fast tick
  * ``hold_position``      -- trivial baseline, no model call: repeat ``observation.state``
    (zero delta9, hand target = current hand state), constant across every chunk step k

§8.1-B (EEF mm/deg via FK), -C (joint-space via retarget.py), -E (rollout drift) and -F
(smoothness) all need ``retarget.py`` (step 13) and are deliberately NOT implemented here --
that's step 14's job (see REDESIGN_PLAN.md §12's build order). This module only produces what
step 10b needs: the zero-shot floor for §8.1-A/D and the "beats hold-position" sanity check
§11.9 asks for before step 11's pilot train.

Real forward passes through the checkpoint only work after ``origami.trex_patch.apply()`` --
that call also carries three NEW patches (3/4/5) found while building this module: the
installed transformers version (5.16.x) has drifted from T-Rex's own pin (4.57.0.dev0) in the
vision-tower output shape, the rotary-embedding config schema, and ``get_rope_index``'s
signature. Without them, `model_load` raises before a single real image goes through the
model. See ``origami/trex_patch.py``'s module docstring for the full story on each.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TREX_DIR = _REPO_ROOT / "T-Rex"
if str(_TREX_DIR) not in sys.path:
    sys.path.insert(0, str(_TREX_DIR))

import origami.trex_patch as _trex_patch  # noqa: E402

_trex_patch.apply()

from utils.lerobot_common import DEFORM_KEYS, KEY_ACTION, KEY_STATE  # noqa: E402
from origami.constants import INSTRUCTION  # noqa: E402
from origami.kinematics import matrix_to_rot6d  # noqa: E402
from origami.diagnose_shift import ACTION_BLOCKS  # noqa: E402

CONFIGS = ("cascaded", "disable_tactile", "tactile_zeroed", "hold_position")


# ─────────────────────────────────────────────────────────────────────────────
# Model / dataset setup
# ─────────────────────────────────────────────────────────────────────────────

def build_eval_args(checkpoint_dir: str) -> SimpleNamespace:
    """Everything `scripts.test.model_load`/`CascadedServer` need, sourced from the
    checkpoint's own `training_args.json` -- the zero-shot config IS whatever the checkpoint
    was trained with, by definition (we're not adapting anything)."""
    with open(Path(checkpoint_dir) / "training_args.json") as f:
        ta = json.load(f)
    args = SimpleNamespace(**ta)
    args.checkpoint_path = checkpoint_dir
    args.base_model_path = ""
    args.stats_path = ""
    args.dataset_name = ""
    args.cuda = "0"
    args.image_size = None
    args.disable_tactile = 0
    return args


def load_model_and_stats(checkpoint_dir: str):
    from scripts.test import model_load

    args = build_eval_args(checkpoint_dir)
    model, processor, statistic = model_load(args)
    return args, model, processor, statistic


def open_eval_dataset(root: str, vqvae_window: int):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    repo_id = Path(root.rstrip("/")).name
    delta_timestamps = {"observation.tactile_f6": [(i - (vqvae_window - 1)) / 30.0
                                                    for i in range(vqvae_window)]}
    return LeRobotDataset(repo_id, root=root, delta_timestamps=delta_timestamps)


def sample_indices(n_total: int, n_samples: int, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.choice(n_total, size=min(n_samples, n_total), replace=False)


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame payload + the 4 configs
# ─────────────────────────────────────────────────────────────────────────────

def _tensor_to_pil(t) -> "PIL.Image.Image":
    from PIL import Image
    arr = (t.clamp(0, 1).float() * 255.0).to("cpu").numpy().astype(np.uint8)
    return Image.fromarray(arr.transpose(1, 2, 0))


def _pil_to_png_bytes(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def build_payload(item: dict, use_robot_state: bool) -> dict:
    deform = np.stack([item[k][0].numpy() for k in DEFORM_KEYS], axis=0)   # [10,H,W] in [0,1]
    payload = {
        "image_head": _pil_to_png_bytes(_tensor_to_pil(item["observation.images.head"])),
        "image_wrist_right": _pil_to_png_bytes(_tensor_to_pil(item["observation.images.wrist_right"])),
        "image_wrist_left": _pil_to_png_bytes(_tensor_to_pil(item["observation.images.wrist_left"])),
        "task_description": INSTRUCTION,
        "tactile_f6": item["observation.tactile_f6"].numpy(),   # [W,10,6] dense window
        "tactile_deform": deform,
        "state_fast": item[KEY_STATE].numpy() if use_robot_state else None,
    }
    return payload


_IDENTITY_ROT6D = matrix_to_rot6d(np.eye(3)).astype(np.float32)


def hold_position_prediction(state62: np.ndarray, action_chunk: int) -> np.ndarray:
    """Baseline: repeat `observation/state` -- zero delta9 (stay at the current pose) and the
    current hand joints as the hand target, held constant across every chunk step k."""
    row = np.empty(62, dtype=np.float32)
    for name, sl in ACTION_BLOCKS:
        if name.endswith("trans3"):
            row[sl] = 0.0
        elif name.endswith("rot6d6"):
            row[sl] = _IDENTITY_ROT6D
        else:  # hand22 -- absolute target, hold = current
            row[sl] = state62[sl]
    return np.tile(row[None, :], (action_chunk, 1))


def run_config(config: str, server, item: dict, use_robot_state: bool, action_chunk: int) -> np.ndarray:
    if config == "hold_position":
        return hold_position_prediction(item[KEY_STATE].numpy(), action_chunk)

    payload = build_payload(item, use_robot_state)
    if config == "tactile_zeroed":
        payload["tactile_f6"] = np.zeros_like(payload["tactile_f6"])
        payload["tactile_deform"] = np.zeros_like(payload["tactile_deform"])
    result = server.predict("slow_and_fast", payload)
    return np.asarray(result["actions"], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def action_space_accuracy(pred: np.ndarray, gt: np.ndarray) -> dict:
    """§8.1-A: per chunk step k, per block -- MSE, MAE, Var(pred)/Var(gt).

    pred, gt : [N, action_chunk, 62], real physical units.
    """
    N, K, _ = gt.shape
    out = {}
    for k in range(K):
        p_k, g_k = pred[:, k, :], gt[:, k, :]
        block_metrics = {}
        for name, sl in ACTION_BLOCKS:
            p_b, g_b = p_k[:, sl], g_k[:, sl]
            err = p_b - g_b
            var_gt = float(np.var(g_b))
            var_pred = float(np.var(p_b))
            block_metrics[name] = {
                "mse": float(np.mean(err ** 2)),
                "mae": float(np.mean(np.abs(err))),
                "variance_share": (var_pred / var_gt) if var_gt > 1e-12 else float("nan"),
            }
        out[f"k={k}"] = block_metrics
    return out


def render_markdown(report: dict) -> str:
    lines = ["# eval_offline zero-shot report", "",
              f"n_samples={report['n_samples']}, checkpoint={report['checkpoint']}, "
              f"root={report['root']}", ""]
    for config in CONFIGS:
        if config not in report["configs"]:
            continue
        lines += [f"## {config}", ""]
        acc = report["configs"][config]["action_space_accuracy"]
        for k_label in ("k=0", f"k={report['action_chunk'] - 1}"):
            if k_label not in acc:
                continue
            lines += [f"### {k_label}", "",
                      "| block | mse | mae | variance_share |", "|---|---|---|---|"]
            for name, _ in ACTION_BLOCKS:
                m = acc[k_label][name]
                lines.append(f"| {name} | {m['mse']:.4g} | {m['mae']:.4g} | {m['variance_share']:.4g} |")
            lines.append("")
    if "hold_position" in report["configs"] and "cascaded" in report["configs"]:
        cas_mae = np.mean([report["configs"]["cascaded"]["action_space_accuracy"]["k=0"][n]["mae"]
                           for n, _ in ACTION_BLOCKS])
        hold_mae = np.mean([report["configs"]["hold_position"]["action_space_accuracy"]["k=0"][n]["mae"]
                            for n, _ in ACTION_BLOCKS])
        verdict = "BEATS" if cas_mae < hold_mae else "DOES NOT BEAT"
        lines += [f"**Zero-shot floor check (§11.9 step 1): cascaded k=0 mean MAE={cas_mae:.4g} "
                 f"{verdict} hold-position k=0 mean MAE={hold_mae:.4g}.**", ""]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────

def run(root: str, checkpoint: str, n_samples: int, configs: list[str], seed: int) -> dict:
    from scripts.test import CascadedServer

    args, model, processor, statistic = load_model_and_stats(checkpoint)
    vqvae_window = int(args.vqvae_config["window"]) if getattr(args, "use_tactile_vqvae", 0) else 16
    ds = open_eval_dataset(root, vqvae_window)
    idx = sample_indices(len(ds), n_samples, seed=seed)
    logger.info("evaluating %d/%d frames from %s", len(idx), len(ds), root)

    servers = {}
    if "cascaded" in configs or "tactile_zeroed" in configs:
        servers["cascaded"] = CascadedServer(args, model, processor, statistic)
    if "disable_tactile" in configs:
        dt_args = SimpleNamespace(**vars(args))
        dt_args.disable_tactile = 1
        servers["disable_tactile"] = CascadedServer(dt_args, model, processor, statistic)

    preds = {c: [] for c in configs}
    gts = []
    for i in idx:
        item = ds[int(i)]
        gts.append(np.asarray(item[KEY_ACTION], dtype=np.float32))
        for config in configs:
            server = servers.get("disable_tactile" if config == "disable_tactile" else "cascaded")
            preds[config].append(run_config(config, server, item, bool(args.use_robot_state),
                                             int(args.action_chunk)))

    gt = np.stack(gts, axis=0)
    report = {
        "n_samples": len(idx), "checkpoint": checkpoint, "root": root,
        "action_chunk": int(args.action_chunk), "configs": {},
    }
    for config in configs:
        pred = np.stack(preds[config], axis=0)
        report["configs"][config] = {"action_space_accuracy": action_space_accuracy(pred, gt)}
    return report


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, help="Held-out (val) merged origami dataset root.")
    p.add_argument("--checkpoint", required=True,
                   help="Checkpoint dir (training_args.json/config.json/model.pt/processor/). "
                        "--zero-shot points this at the untrained T-Rex midtrain checkpoint.")
    p.add_argument("--zero-shot", action="store_true",
                   help="Documents intent (§11.9 step 1); behavior is identical either way -- "
                        "this script always evaluates whatever checkpoint --checkpoint names.")
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=CONFIGS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None, help="Markdown report path (default: stdout).")
    p.add_argument("--json-output", default=None)
    args = p.parse_args(argv)

    report = run(args.root, args.checkpoint, args.n_samples, args.configs, args.seed)
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
