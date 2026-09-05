"""CLI: §10 gates checkable against a real prepared/merged dataset root. REDESIGN_PLAN.md §10.

Gates already covered by fast, fixture-scale unit tests -- G0, G1a, G1b, G2, G2b, G3, G3b,
G8, G8b -- live in ``origami/tests/`` and are not reimplemented here ("every gate is a test
... or a verify.py subcommand", §10); ``verify.py unit`` just re-runs them via pytest. This
module adds the checks that need a real prepared corpus at production scale rather than the
small local fixture, and is invoked as ``verify.py all --root <merged_root>`` from
``prepare.py``'s phase 2 (§5.5).

Not yet implementable here: **G1c** (checkpoint/server-side digest -- no checkpoint/server
exist yet), **G7/G7b** (loader/forward-pass parity -- needs ``delayed_lerobot_dataset.py``,
step 9), **G9-G17** (train/deploy gates -- need steps 9/13/14). ``verify.py list`` reports
this.

**G6/G4 here are deliberately weaker than a stream-and-delete-time check.** By the time
``verify.py`` runs against a merged root, the original raw per-season videos have already
been dropped (§5.1 stream-and-delete) -- there is no "true source luma" or "true source
joint angles" left to compare against out here. So:
  * G6 becomes a **self-consistency** check: `gray_to_3ch` (§5.2) replicates one luma channel
    into 3; if the lossless round trip is intact, all 3 decoded channels of every deform
    frame must be *exactly* equal. This does not re-verify losslessness against the original
    source (that's `test_convert.py`'s job, checked against real source video while it's
    still on disk during a single season's conversion) -- it verifies the merge/storage step
    didn't introduce a channel-wise divergence.
  * G4 becomes a **pose-reconstruction** check, extending `test_convert.py`'s single-frame
    version to a sampled pass over the whole root: for each sampled frame, rebuild the left/
    right FK matrices from the stored 9-D `observation.state`/`action_abs`, feed them back
    through the *same* `build_action_chunk` used at conversion (with the hand-joint targets
    read out of the stored chunk's own k=0 step, since raw `action65` is also gone), and
    compare the recomputed chunk against the stored one to a tight tolerance (rot6d/FK
    reconstruction is not bit-exact, so "exact" here means "matches to 1e-5", not literal
    bitwise identity -- a real full-season bitwise check is `test_convert.py`'s job, run
    while the season's raw data is still present).
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

from origami.constants import REPO_ROOT
from origami.kinematics import rot6d_to_matrix
from origami.splits import parse_splits, validate_splits
from utils.lerobot_common import ACTION_CHUNK, DEFORM_KEYS, F6_DIM, build_action_chunk

logger = logging.getLogger(__name__)

UNIT_GATE_TEST_FILES = [
    "test_upstream_drift.py",  # G0
    "test_kinematics.py",  # G1a, G1b, G2, G2b, G3, G3b
    "test_stats.py",  # G8, G8b
]

# Gates that need infra not yet built (steps 9/13/14) -- see module docstring.
NOT_YET_IMPLEMENTABLE = {
    "G1c": "needs a checkpoint + a running serve_zenoh server to compare digests against",
    "G7": "needs delayed_lerobot_dataset.py (step 9)",
    "G7b": "needs a tiny stub Qwen3VLVLAModel forward pass (step 9)",
    "G9a": "needs train_origami.py (step 9)",
    "G9b": "needs a midtrain checkpoint (step 10c/11)",
    "G10": "needs serve_zenoh.py (step 13)",
    "G11": "needs serve_zenoh.py + eval_shadow.py (step 13/14)",
    "G12": "needs serve_zenoh.py (step 13)",
    "G13": "needs bench_latency.py (step 13)",
    "G14": "needs policy.py + serve_zenoh.py (step 13)",
    "G15": "needs a trained VQ-VAE (step 11+)",
    "G16": "needs a real checkpoint's preprocessor_config.json (step 9/11)",
    "G17": "needs delayed_lerobot_dataset.py + serve_zenoh.py (step 9/13)",
}


def run_unit_gates() -> bool:
    files = [str(REPO_ROOT / "origami" / "tests" / f) for f in UNIT_GATE_TEST_FILES]
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", *files])
    return result.returncode == 0


def verify_g18(root: Path, split: str, extra_seasons: str = "none") -> bool:
    prep = json.loads((root / "meta" / "origami_prep.json").read_text())
    got = set(prep["seasons"])

    splits = parse_splits()
    validate_splits(splits)
    expected_core = set(splits[split])

    if extra_seasons == "train" and split == "train":
        missing = expected_core - got
        extra = got - expected_core  # extras are allowed to be present
        ok = not missing
    else:
        missing = expected_core - got
        extra = got - expected_core
        ok = not missing and not extra

    if not ok:
        logger.error("G18 FAIL: missing=%s unexpected_extra=%s", sorted(missing), sorted(extra))
    else:
        logger.info("G18 PASS: %d seasons match the '%s' split (extra_seasons=%s)", len(got), split, extra_seasons)
    return ok


def verify_g8(root: Path) -> bool:
    stats = json.loads((root / "meta" / "trex_norm_stats.json").read_text())
    block = next(iter(stats.values()))
    ok = True

    for key, shape in (("action", (ACTION_CHUNK, 62)), ("state", (62,))):
        arr = np.array(block[key]["q01"])
        if arr.shape != shape:
            logger.error("G8 FAIL: %s q01 shape %s != %s", key, arr.shape, shape)
            ok = False

    if "tactile_f6" in block:
        mask = block["tactile_f6"]["mask"]
        if len(mask) != F6_DIM:
            logger.error("G8 FAIL: tactile_f6 mask length %d != %d", len(mask), F6_DIM)
            ok = False
        if all(mask):
            logger.error("G8b FAIL: tactile mask is all-True (no degenerate channel detected)")
            ok = False

    if ok:
        logger.info("G8/G8b PASS: %d transitions, %d trajectories", block["num_transitions"], block["num_trajectories"])
    return ok


def _sample_frame_indices(n_total: int, n_sample: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(n_total, size=min(n_sample, n_total), replace=False)


def verify_g6(root: Path, n_sample: int = 2000) -> bool:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("origami/eef62_verify", root=str(root))
    idxs = _sample_frame_indices(len(ds), n_sample)

    n_bad = 0
    for i in idxs:
        item = ds[int(i)]
        for key in DEFORM_KEYS:
            frame = item[key].numpy()  # (3, H, W)
            if not np.array_equal(frame[0], frame[1]) or not np.array_equal(frame[1], frame[2]):
                n_bad += 1

    ok = n_bad == 0
    if ok:
        logger.info("G6 PASS: %d sampled frames x %d deform tiles, all 3 channels equal", len(idxs), len(DEFORM_KEYS))
    else:
        logger.error("G6 FAIL: %d channel-inequality violations across %d sampled frames", n_bad, len(idxs))
    return ok


def verify_g4(root: Path, n_sample: int = 500, atol: float = 1e-5, pass_rate_threshold: float = 0.995) -> bool:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("origami/eef62_verify", root=str(root))
    idxs = _sample_frame_indices(len(ds), n_sample)

    def _pose9_to_matrix(v9):
        m = np.eye(4)
        m[:3, 3] = v9[0:3]
        m[:3, :3] = rot6d_to_matrix(v9[3:9])
        return m

    n_ok = 0
    n_checked = 0
    for i in idxs:
        i = int(i)
        item = ds[i]

        state = item["observation.state"].numpy().astype(np.float64)
        action_abs = item["action_abs"].numpy().astype(np.float64)
        chunk0 = item["action"][0].numpy().astype(np.float64)

        # 62-D layout (§1.4): [0:9]=left pose9, [9:31]=left hand(22), [31:40]=right pose9,
        # [40:62]=right hand(22).
        S_l = _pose9_to_matrix(state[0:9])[None]
        S_r = _pose9_to_matrix(state[31:40])[None]
        A_l = _pose9_to_matrix(action_abs[0:9])[None]
        A_r = _pose9_to_matrix(action_abs[31:40])[None]
        left_hand_target = chunk0[9:31][None].astype(np.float32)
        right_hand_target = chunk0[40:62][None].astype(np.float32)

        n_checked += 1
        # min_len=1 forces build_action_chunk's clamp to fut=0 for every k, so every row of
        # the recomputed chunk equals its own k=0 step -- only chunk[0] is a meaningful
        # comparison against the stored chunk's k=0 step (this reconstructs the
        # delta9(state_pose, action_abs_pose) invariant §6 states, not the multi-step clamp,
        # which needs the full episode and is exactly what test_convert.py's G4 test covers).
        recon_chunk0 = build_action_chunk(S_l, A_l, left_hand_target, S_r, A_r, right_hand_target, 0, 1)[0]
        if np.allclose(recon_chunk0, chunk0, atol=atol):
            n_ok += 1

    pass_rate = n_ok / max(n_checked, 1)
    ok = pass_rate >= pass_rate_threshold
    level = logger.info if ok else logger.error
    level("G4 %s: %d/%d sampled frames reconstruct within atol=%s (%.2f%%, threshold %.2f%%)",
          "PASS" if ok else "FAIL", n_ok, n_checked, atol, 100 * pass_rate, 100 * pass_rate_threshold)
    return ok


def cmd_list(_args) -> None:
    print("Fixture-scale gates (run via `verify.py unit`, defined in origami/tests/):")
    for f in UNIT_GATE_TEST_FILES:
        print(f"  {f}")
    print("\nRoot-scale gates (run via `verify.py all --root ...`): G18, G8/G8b, G6, G4")
    print("\nNot yet implementable (see module docstring for why):")
    for gate, why in NOT_YET_IMPLEMENTABLE.items():
        print(f"  {gate}: {why}")


def cmd_unit(_args) -> None:
    sys.exit(0 if run_unit_gates() else 1)


def run_all_gates(
    root: Path, split: str, extra_seasons: str = "none",
    g6_sample: int = 2000, g4_sample: int = 500,
) -> dict[str, bool]:
    """G18, G8/G8b, G6, G4 against a prepared/merged root. Shared by ``verify.py all`` and
    ``prepare.py``'s phase 2 (§5.5: "merge_shards(...) then verify.py --root out_root")."""
    root = Path(root)
    return {
        "G18": verify_g18(root, split, extra_seasons),
        "G8/G8b": verify_g8(root),
        "G6": verify_g6(root, g6_sample),
        "G4": verify_g4(root, g4_sample),
    }


def cmd_all(args) -> None:
    results = run_all_gates(Path(args.root), args.split, args.extra_seasons, args.g6_sample, args.g4_sample)
    print(json.dumps(results, indent=2))
    sys.exit(0 if all(results.values()) else 1)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)

    p_list = sub.add_parser("list", help="list all gates and their status")
    p_list.set_defaults(func=cmd_list)

    p_unit = sub.add_parser("unit", help="run the fixture-scale gates (G0, G1*, G2*, G3*, G8*)")
    p_unit.set_defaults(func=cmd_unit)

    p_all = sub.add_parser("all", help="run root-scale gates (G18, G8/G8b, G6, G4) against a prepared root")
    p_all.add_argument("--root", required=True)
    p_all.add_argument("--split", choices=["train", "val"], default="train")
    p_all.add_argument("--extra-seasons", choices=["none", "train"], default="none")
    p_all.add_argument("--g6-sample", type=int, default=2000)
    p_all.add_argument("--g4-sample", type=int, default=500)
    p_all.set_defaults(func=cmd_all)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
