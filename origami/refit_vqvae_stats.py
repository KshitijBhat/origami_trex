"""Re-fit ``tacf6_vqvae_{min,max,mask}`` to origami's own F6 quantiles. REDESIGN_PLAN.md
§11.8 fix ladder item 1, §12 step 10c.

T-Rex's frozen tactile VQ-VAE min-max normalizes the raw F6 history window with buffers
carried over from *its own* pretraining corpus, then hard-clamps to [-1, 1]
(``modeling_vla.py::encode_tactile_f6_history``). ``diagnose_shift.py::vqvae_code_entropy``
(G15) measures whether that scale is wrong for origami's forces (§1.2: p99 |F| ≈ 42 N on some
fingers, ≈ 0 on the dead ring/pinky ones) -- a saturated or near-constant code collapses the
tactile expert's contribution with no error raised.

This is the cheapest rung of the fix ladder: overwrite only the three normalization buffers
with origami's own q01/q99 (and the degenerate-dim mask) from ``meta/trex_norm_stats.json``,
leaving the encoder weights and codebook untouched -- no retraining. Never edits the
checkpoint in place; always writes a new checkpoint directory with
``training_args.json["vqvae_stats_source"] = "origami"`` recorded.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def load_origami_f6_stats(dataset_root: str) -> dict:
    """``tactile_f6`` q01/q99/mask (each ``[60]``) from a prep root's
    ``meta/trex_norm_stats.json`` -- REDESIGN_PLAN.md §5.4's ``STATS_KEY`` block, same shape
    and flattening (10 fingers x 6) the model's ``tacf6_vqvae_*`` buffers expect."""
    with open(Path(dataset_root) / "meta" / "trex_norm_stats.json") as f:
        stats_raw = json.load(f)
    block = next(iter(stats_raw.values()))["tactile_f6"]
    q01 = np.asarray(block["q01"], dtype=np.float32)
    q99 = np.asarray(block["q99"], dtype=np.float32)
    mask = np.asarray(block["mask"], dtype=bool)
    assert q01.shape == q99.shape == mask.shape == (60,), (q01.shape, q99.shape, mask.shape)
    return {"tacf6_min": q01, "tacf6_max": q99, "tacf6_mask": mask}


def refit(checkpoint_dir: str, dataset_root: str, out_dir: str) -> None:
    import torch

    checkpoint_dir = Path(checkpoint_dir)
    out_dir = Path(out_dir)
    if out_dir.resolve() == checkpoint_dir.resolve():
        raise ValueError("--out-dir must differ from --checkpoint (never edit in place)")

    stats = load_origami_f6_stats(dataset_root)
    logger.info(
        "origami tactile_f6 stats: %d/60 dims masked in, min-range=%.4g max-range=%.4g",
        int(stats["tacf6_mask"].sum()),
        float((stats["tacf6_max"] - stats["tacf6_min"])[stats["tacf6_mask"]].min()),
        float((stats["tacf6_max"] - stats["tacf6_min"])[stats["tacf6_mask"]].max()),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "training_args.json", "stats_data.json", "processor"):
        src = checkpoint_dir / name
        dst = out_dir / name
        if not src.exists():
            continue
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    with open(checkpoint_dir / "training_args.json") as f:
        ta = json.load(f)
    assert bool(ta.get("use_tactile_vqvae", 0)), "checkpoint has use_tactile_vqvae=0; nothing to refit"

    logger.info("loading model.pt (this reads the full checkpoint once)...")
    sd = torch.load(checkpoint_dir / "model.pt", map_location="cpu")
    for key in ("tacf6_vqvae_min", "tacf6_vqvae_max", "tacf6_vqvae_mask"):
        assert key in sd, f"{key} not found in model.pt -- is use_tactile_vqvae really on?"

    sd["tacf6_vqvae_min"] = torch.from_numpy(stats["tacf6_min"]).clone()
    sd["tacf6_vqvae_max"] = torch.from_numpy(stats["tacf6_max"]).clone()
    sd["tacf6_vqvae_mask"] = torch.from_numpy(stats["tacf6_mask"]).clone()

    logger.info("writing refit model.pt to %s ...", out_dir / "model.pt")
    torch.save(sd, out_dir / "model.pt")

    ta["vqvae_stats_source"] = "origami"
    ta["vqvae_stats_source_root"] = str(dataset_root)
    with open(out_dir / "training_args.json", "w") as f:
        json.dump(ta, f, indent=2)
    logger.info("done: %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="source checkpoint directory")
    parser.add_argument(
        "--dataset-root", required=True,
        help="a prep root (e.g. the train root) whose meta/trex_norm_stats.json supplies "
             "origami's own tactile_f6 q01/q99/mask",
    )
    parser.add_argument("--out-dir", required=True, help="new checkpoint directory to write")
    parser.add_argument(
        "--diagnose", action="store_true",
        help="also run diagnose_shift.vqvae_code_entropy (G15) before and after, on "
             "--dataset-root, and print the verdict change",
    )
    parser.add_argument("--n-windows", type=int, default=200_000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    before = None
    if args.diagnose:
        from origami.diagnose_shift import vqvae_code_entropy

        logger.info("G15 before refit...")
        before = vqvae_code_entropy(args.dataset_root, args.checkpoint, n_windows=args.n_windows)
        logger.info("G15 before: clamp_saturation=%.4g, min_entropy=%.4g",
                    before["clamp_saturation_fraction"],
                    min(v["normalized_entropy"] for v in before["slots"].values()))

    refit(args.checkpoint, args.dataset_root, args.out_dir)

    if args.diagnose:
        from origami.diagnose_shift import vqvae_code_entropy

        logger.info("G15 after refit...")
        after = vqvae_code_entropy(args.dataset_root, args.out_dir, n_windows=args.n_windows)
        logger.info("G15 after: clamp_saturation=%.4g, min_entropy=%.4g",
                    after["clamp_saturation_fraction"],
                    min(v["normalized_entropy"] for v in after["slots"].values()))
        report = {"before": before, "after": after}
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
