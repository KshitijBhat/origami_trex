"""Torch-free sanity checks for a prepared origami-flat dataset.

Runs on the prep box before anything is uploaded, so a malformed dataset is
caught here rather than 20 minutes into an A100 session.  It re-implements the
handful of transforms `OrigamiDataset` applies (deform tiling, min-max
normalisation) instead of importing them, so no torch is needed.

    python -m trex_origami.verify --root <split root> [--src-root <raw seasons>]

Checks:
  1. every episode parquet exists, has the declared row count and schema
  2. norm-stats ranks: action [chunk, 65], state [65], tactile_f6 [60]
  3. normalised actions/states land inside [-1, 1]
  4. JPEGs decode at the expected size; the deform strip splits into a 2x5 grid
     of 240x240 tiles (what DeformEncoder's hardcoded 128*15*15 requires)
  5. deform tile occupancy is consistent with the tactile force vector -- i.e.
     the image grid and the 60-D signal agree on which finger is in contact
  6. anchoring round-trip: reconstructing `anchor + action_chunk` reproduces the
     wire contract, and step 0 of the reconstruction equals `action_abs` on
     every dim -- including the absolute dims, which must land there *without*
     anyone adding the state
  7. with --src-root, one sample's chunk/state/history is re-derived straight
     from the raw season parquet under the declared anchoring rule
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import random
import sys
from typing import List, Optional, Sequence

import numpy as np
import pyarrow.parquet as pq

from .anchoring import (ANCHOR_ABSOLUTE, ANCHOR_PREV_COMMAND, build_anchor,
                        describe as describe_anchor, masks as anchor_masks,
                        spec_from_meta, to_absolute)
from .seasons import ACTION_DIM, FINGER_NAMES, JOINT_NAMES

logger = logging.getLogger(__name__)

DEFORM_TILE = 240
DEFORM_ROWS, DEFORM_COLS = 2, 5
N_FINGERS = 10
F6_PER_FINGER = 6
# A fingertip counts as "in contact" above this force, and needs to be in
# contact this often before its deform/force correlation means anything.
# Mirrors `stats.MIN_NORM_RANGE_JOINT`; imported by value so verify stays
# runnable against a stats file produced by any version of the prep.
MIN_NORM_RANGE_JOINT = 1e-3
CONTACT_THRESHOLD_N = 0.5
MIN_CONTACT_FRACTION = 0.05
COLUMNS = ("state", "action_chunk", "action_chunk_abs", "action_abs",
           "prev_command", "phase", "tacf6_hist", "head", "wrist_left",
           "wrist_right", "deform")
# Columns added after splits were already prepared in the wild.  A dataset
# missing them still verifies -- the checks that need them are skipped -- so
# `verify` runs against an older dataset instead of failing on schema.
OPTIONAL_COLUMNS = ("prev_command", "action_chunk_abs")
LEGACY_COLUMNS = tuple(c for c in COLUMNS if c not in OPTIONAL_COLUMNS)


def split_deform_strip(arr: np.ndarray) -> np.ndarray:
    """[480, 1200] -> [10, 240, 240]; row 0 = left hand, row 1 = right hand.

    Identical to `qwen_vla.origami_dataset.split_deform_strip`; duplicated here
    only to keep this module importable without torch.
    """
    tiles = arr.reshape(DEFORM_ROWS, DEFORM_TILE, DEFORM_COLS, DEFORM_TILE)
    return tiles.transpose(0, 2, 1, 3).reshape(N_FINGERS, DEFORM_TILE, DEFORM_TILE)


def normalize(values, mask, vmin, vmax):
    return np.where(mask, np.clip(2 * (values - vmin) / (vmax - vmin + 1e-8) - 1, -1, 1),
                    values)


class Failure(AssertionError):
    pass


def _check(condition: bool, message: str, problems: List[str]) -> bool:
    if not condition:
        problems.append(message)
    return condition


def verify(root: str, n_samples: int = 8, seed: int = 0,
           src_root: str = "", montage_path: str = "") -> List[str]:
    problems: List[str] = []
    rng = random.Random(seed)

    with open(os.path.join(root, "meta", "dataset.json")) as handle:
        meta = json.load(handle)
    cfg = meta["config"]
    chunk, dim = int(cfg["action_chunk"]), int(cfg["action_dim"])
    window = int(cfg["vqvae_window"])
    episodes = meta["episodes"]
    spec = spec_from_meta(meta)
    sel = anchor_masks(spec)
    has_prev = bool(sel[ANCHOR_PREV_COMMAND].any())

    logger.info("[verify] %s", root)
    logger.info("[verify] %d episodes / %d samples / %d seasons | stride %d, chunk %d, %dpx",
                len(episodes), meta["n_samples"], meta["n_seasons"],
                cfg["sample_stride"], chunk, cfg["image_size"])
    logger.info("[verify] anchoring %s", describe_anchor(spec))

    # ── 1. episode files ──────────────────────────────────────────────────────
    total = 0
    for entry in episodes:
        path = os.path.join(root, entry["file"])
        if not _check(os.path.exists(path), f"missing {entry['file']}", problems):
            continue
        pf = pq.ParquetFile(path)
        _check(pf.metadata.num_rows == entry["n_samples"],
               f"{entry['file']}: {pf.metadata.num_rows} rows, meta says {entry['n_samples']}",
               problems)
        names = set(pf.schema_arrow.names)
        _check(set(LEGACY_COLUMNS) <= names <= set(COLUMNS),
               f"{entry['file']}: columns {pf.schema_arrow.names}", problems)
        _check(not has_prev or "prev_command" in names,
               f"{entry['file']}: anchors arm dims to the previous command but has "
               f"no `prev_command` column — prepared by an older trex_origami",
               problems)
        total += pf.metadata.num_rows
    _check(total == meta["n_samples"],
           f"row total {total} != meta n_samples {meta['n_samples']}", problems)
    logger.info("[verify] episode files OK (%d rows)", total)

    # ── 2. norm-stats ranks ───────────────────────────────────────────────────
    stats_path = os.path.join(root, "meta", "norm_stats.json")
    block = None
    if _check(os.path.exists(stats_path), "meta/norm_stats.json missing", problems):
        with open(stats_path) as handle:
            stats = json.load(handle)
        _check(len(stats) == 1, f"norm_stats has {len(stats)} top-level keys, expected 1",
               problems)
        block = stats[next(iter(stats))]
        for key, want in (("action", (chunk, dim)), ("state", (dim,)),
                          ("tactile_f6", (N_FINGERS * F6_PER_FINGER,))):
            for field in ("q01", "q99"):
                got = np.shape(block[key][field])
                _check(got == want,
                       f"norm_stats.{key}.{field} has shape {got}, expected {want} "
                       f"(a wrong action rank mis-broadcasts silently)", problems)
        # Masked-off dims are normalisation *passthrough*, which is also what
        # the serve/eval path clamps to delta 0 (`test.py:_clamp_frozen`).  Log
        # the names and the actual spread, not just indices: whether the torso
        # is still frozen once all 101 train seasons are in is exactly the
        # question that decides whether clamping them is safe.
        mask = np.array(block["action"]["mask"], dtype=bool)
        spread = np.max(np.array(block["action"]["q99"])
                        - np.array(block["action"]["q01"]), axis=0)   # [chunk,dim] -> [dim]
        off = np.where(~mask)[0]
        logger.info("[verify] norm-stats ranks OK | %d action dim(s) masked off "
                    "(frozen -> held at the measured state at inference)", len(off))
        for i in off:
            logger.info("[verify]     dim %2d %-22s max q99-q01 spread %.2e rad "
                        "(%.4f deg)", i, JOINT_NAMES[i], spread[i],
                        float(np.degrees(spread[i])))
        # The near-misses matter too: a dim that only just cleared the threshold
        # is one season away from flipping, and the clamp decision would flip
        # with it.
        near = [i for i in np.where(mask)[0] if spread[i] < 10 * MIN_NORM_RANGE_JOINT]
        for i in near:
            logger.info("[verify]     dim %2d %-22s NOT masked but spread is only "
                        "%.2e rad — within 10x of the frozen threshold",
                        i, JOINT_NAMES[i], spread[i])

    # ── 3-5. sample-level checks ──────────────────────────────────────────────
    from PIL import Image

    picks = [rng.choice(episodes) for _ in range(n_samples)]
    tiles_for_montage = None
    for entry in picks:
        table = pq.read_table(os.path.join(root, entry["file"]))
        i = rng.randrange(table.num_rows)

        state = np.asarray(table["state"][i].as_py(), dtype=np.float32)
        chunk_arr = np.asarray(table["action_chunk"][i].as_py(), dtype=np.float32)
        hist = np.asarray(table["tacf6_hist"][i].as_py(), dtype=np.float32)
        action_abs = np.asarray(table["action_abs"][i].as_py(), dtype=np.float32)
        prev_command = (np.asarray(table["prev_command"][i].as_py(), dtype=np.float32)
                        if "prev_command" in table.column_names else action_abs)
        future_abs = (np.asarray(table["action_chunk_abs"][i].as_py(), dtype=np.float32)
                      if "action_chunk_abs" in table.column_names else None)

        # ── 6. anchoring round-trip ──────────────────────────────────────────
        # `anchor + stored target` must be the wire contract, and its step 0 is
        # the command at t by construction.  Checking it here (rather than only
        # under --src-root) is what makes a mis-declared `action_anchor` a
        # prep-time failure instead of a silent 0.7 deg bias in every eval.
        anchor = build_anchor(state, prev_command, spec)
        recon0 = to_absolute(chunk_arr.reshape(chunk, dim), anchor)[0]
        _check(np.allclose(recon0, action_abs, atol=1e-5),
               f"{entry['file']}[{i}]: anchor + action_chunk[0] does not reproduce "
               f"action_abs (max |diff| {np.abs(recon0 - action_abs).max():.2e} rad) "
               f"— the declared anchoring rule does not match the stored targets",
               problems)
        # The absolute dims must be readable straight off the wire.  If someone
        # re-introduces a state offset there, this is where it shows up.
        abs_dims = sel[ANCHOR_ABSOLUTE]
        if abs_dims.any():
            _check(np.allclose(chunk_arr.reshape(chunk, dim)[0][abs_dims],
                               action_abs[abs_dims], atol=1e-5),
                   f"{entry['file']}[{i}]: absolute-anchored dims do not equal "
                   f"action_abs at step 0 — they carry an offset they should not",
                   problems)
        _check(state.shape == (dim,), f"state shape {state.shape}", problems)
        _check(chunk_arr.size == chunk * dim, f"action_chunk size {chunk_arr.size}", problems)
        if future_abs is not None:
            # The raw future commands are stored, not derived, so this catches a
            # mismatch between the two action columns directly rather than
            # relying on the reconstruction above.
            _check(future_abs.size == chunk * dim,
                   f"action_chunk_abs size {future_abs.size}", problems)
            if future_abs.size == chunk * dim:
                recon = to_absolute(chunk_arr.reshape(chunk, dim), anchor)
                _check(np.allclose(recon, future_abs.reshape(chunk, dim), atol=1e-5),
                       f"{entry['file']}[{i}]: anchor + action_chunk does not reproduce "
                       f"action_chunk_abs (max |diff| "
                       f"{np.abs(recon - future_abs.reshape(chunk, dim)).max():.2e} rad)",
                       problems)
        _check(hist.size == window * N_FINGERS * F6_PER_FINGER,
               f"tacf6_hist size {hist.size}", problems)
        _check(np.isfinite(state).all() and np.isfinite(chunk_arr).all()
               and np.isfinite(hist).all(), f"{entry['file']}[{i}]: non-finite values", problems)
        phase = float(table["phase"][i].as_py())
        _check(0.0 <= phase <= 1.0, f"phase {phase} outside [0,1]", problems)

        if block is not None:
            na = normalize(chunk_arr.reshape(chunk, dim), np.array(block["action"]["mask"]),
                           np.array(block["action"]["q01"], dtype=np.float32),
                           np.array(block["action"]["q99"], dtype=np.float32))
            ns = normalize(state, np.array(block["state"]["mask"]),
                           np.array(block["state"]["q01"], dtype=np.float32),
                           np.array(block["state"]["q99"], dtype=np.float32))
            # Masked-off dims pass through raw, so only check the normalised ones.
            am = np.array(block["action"]["mask"])
            sm = np.array(block["state"]["mask"])
            _check(np.abs(na[:, am]).max() <= 1.0 + 1e-6,
                   f"normalised action out of range: {np.abs(na[:, am]).max()}", problems)
            _check(np.abs(ns[sm]).max() <= 1.0 + 1e-6,
                   f"normalised state out of range: {np.abs(ns[sm]).max()}", problems)

        want_rgb = (cfg["image_size"], cfg["image_size"])
        for column in ("head", "wrist_left", "wrist_right"):
            img = Image.open(io.BytesIO(table[column][i].as_py()))
            _check(img.size == want_rgb, f"{column} is {img.size}, expected {want_rgb}",
                   problems)

        deform = Image.open(io.BytesIO(table["deform"][i].as_py())).convert("L")
        _check(deform.size == (DEFORM_COLS * DEFORM_TILE, DEFORM_ROWS * DEFORM_TILE),
               f"deform is {deform.size}, expected (1200, 480)", problems)
        tiles = split_deform_strip(np.asarray(deform, dtype=np.float32) / 255.0)
        _check(tiles.shape == (N_FINGERS, DEFORM_TILE, DEFORM_TILE),
               f"deform tiles {tiles.shape}", problems)
        if tiles_for_montage is None and tiles.max() > 0.02:
            tiles_for_montage = tiles

    logger.info("[verify] sampled %d rows OK", n_samples)

    problems += _verify_deform_ordering(root, meta, seed=seed)

    if montage_path and tiles_for_montage is not None:
        strip = np.concatenate([
            np.concatenate(list(tiles_for_montage[r * DEFORM_COLS:(r + 1) * DEFORM_COLS]), axis=1)
            for r in range(DEFORM_ROWS)], axis=0)
        Image.fromarray((strip * 255).astype(np.uint8)).save(montage_path)
        logger.info("[verify] wrote deform montage (%s) to %s",
                    " | ".join(FINGER_NAMES), montage_path)

    # ── 6. optional round-trip against the raw season ─────────────────────────
    if src_root:
        problems += _verify_against_source(root, meta, src_root)

    return problems


def _verify_deform_ordering(root: str, meta: dict, n_rows: int = 150,
                            seed: int = 0) -> List[str]:
    """Confirm the 2x5 deform grid indexes the same fingers as `observation.tactile`.

    A transposed or row-swapped grid would feed the deform encoder one finger's
    image alongside another finger's wrench, and nothing downstream would
    complain -- the shapes are identical either way.  The check that actually
    catches it: correlate per-finger contact-force magnitude against per-finger
    tile brightness over many frames and require the 10x10 matrix to be
    diagonal.  On a correct dataset the diagonal sits near 0.9 and the
    off-diagonal near 0.
    """
    from PIL import Image

    problems: List[str] = []
    cfg = meta["config"]
    window = int(cfg["vqvae_window"])
    rng = random.Random(seed + 1)
    episodes = meta["episodes"]

    forces, inks = [], []
    per_episode = max(1, n_rows // min(len(episodes), 12))
    for entry in rng.sample(episodes, min(len(episodes), 12)):
        table = pq.read_table(os.path.join(root, entry["file"]),
                              columns=["tacf6_hist", "deform"])
        step = max(1, table.num_rows // per_episode)
        for i in range(0, table.num_rows, step):
            hist = np.asarray(table["tacf6_hist"][i].as_py(),
                              dtype=np.float32).reshape(window, N_FINGERS, F6_PER_FINGER)
            forces.append(np.linalg.norm(hist[-1, :, :3], axis=-1))
            gray = np.asarray(
                Image.open(io.BytesIO(table["deform"][i].as_py())).convert("L"),
                dtype=np.float32) / 255.0
            inks.append(split_deform_strip(gray).mean(axis=(1, 2)))

    force = np.asarray(forces)
    ink = np.asarray(inks)
    if len(force) < 20 or ink.max() <= 0:
        logger.warning("[verify] too few contact frames to check deform ordering")
        return problems

    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.nan_to_num(np.array([
            [np.corrcoef(force[:, a], ink[:, b])[0, 1] for b in range(N_FINGERS)]
            for a in range(N_FINGERS)]))

    # Only fingers that actually make contact carry a testable signal.  Folding
    # a plane is a thumb-and-index task: the outer three fingers touch nothing
    # for whole episodes, so their force column is near-constant and correlates
    # with everything and nothing.  Testing them would flag sample noise as a
    # mis-indexed grid.
    contact_frac = (force > CONTACT_THRESHOLD_N).mean(axis=0)
    testable = (contact_frac >= MIN_CONTACT_FRACTION) & (ink.std(axis=0) > 0)
    idx = np.where(testable)[0]

    mean_force = force.mean(axis=0)
    logger.info("[verify] mean contact force per finger (N): %s",
                ", ".join(f"{FINGER_NAMES[k]}={mean_force[k]:.2f}" for k in range(N_FINGERS)))
    if len(idx) < 2:
        logger.warning("[verify] only %d finger(s) make contact in the sampled frames; "
                       "cannot test deform/force alignment", len(idx))
        return problems

    sub = corr[np.ix_(idx, idx)]
    diag = float(np.mean(np.diag(sub)))
    offdiag = float((sub.sum() - np.trace(sub)) / max(1, len(idx) * (len(idx) - 1)))
    logger.info("[verify] deform/force alignment over %d frames on %d contact finger(s) "
                "(%s): diag %.3f, off-diag %.3f", len(force), len(idx),
                ", ".join(FINGER_NAMES[k] for k in idx), diag, offdiag)

    # Each testable finger's tile must correlate best with its own wrench,
    # compared against every channel — not just the testable ones, so a swap
    # with an idle finger is still caught.
    mismatched = [(FINGER_NAMES[k], FINGER_NAMES[int(corr[k].argmax())])
                  for k in idx if int(corr[k].argmax()) != k]
    _check(not mismatched,
           f"deform tile order does not match the force vector: "
           f"{', '.join(f'{a} best-matches {b}' for a, b in mismatched)} "
           f"— the 2x5 grid is mis-indexed", problems)
    _check(diag > offdiag + 0.3,
           f"deform/force correlation is not diagonal over the contact fingers "
           f"(diag {diag:.3f} vs off-diag {offdiag:.3f})", problems)
    return problems


def _verify_against_source(root: str, meta: dict, src_root: str) -> List[str]:
    """Re-derive one episode's numeric columns from the raw season and compare."""
    from .prepare import read_episode_specs, read_season_arrays, _season_root

    problems: List[str] = []
    cfg = meta["config"]
    chunk, dim = int(cfg["action_chunk"]), int(cfg["action_dim"])
    stride, cstride = int(cfg["sample_stride"]), int(cfg["chunk_stride"])
    window = int(cfg["vqvae_window"])

    spec = spec_from_meta(meta)
    entry = meta["episodes"][0]
    season = entry["season"]
    try:
        season_root = _season_root(src_root, season)
    except FileNotFoundError:
        logger.warning("[verify] %s not present under %s — skipping round-trip",
                       season, src_root)
        return problems

    specs = {s.episode_index: s for s in read_episode_specs(season_root, season)}
    arrays = read_season_arrays(season_root, season)
    ep_spec = specs[entry["episode_index"]]
    table = pq.read_table(os.path.join(root, entry["file"]))

    state_all = arrays["observation.state"][ep_spec.row_from:ep_spec.row_to]
    action_all = arrays["action"][ep_spec.row_from:ep_spec.row_to]
    tactile_all = arrays["observation.tactile"][ep_spec.row_from:ep_spec.row_to]
    last = ep_spec.length - 1

    # `prepare` anchors row 0 of an episode to the command at t itself.
    want_prev = np.empty_like(action_all)
    want_prev[0] = action_all[0]
    want_prev[1:] = action_all[:-1]

    for i in (0, table.num_rows // 2, table.num_rows - 1):
        t = i * stride
        idx = np.clip(t + np.arange(chunk) * cstride, 0, last)
        want_chunk = action_all[idx] - build_anchor(state_all[t], want_prev[t], spec)
        hidx = np.clip(t + np.arange(window) - (window - 1), 0, last)
        want_hist = tactile_all[hidx]

        got_state = np.asarray(table["state"][i].as_py(), dtype=np.float32)
        got_chunk = np.asarray(table["action_chunk"][i].as_py(),
                               dtype=np.float32).reshape(chunk, dim)
        got_abs = np.asarray(table["action_abs"][i].as_py(), dtype=np.float32)
        got_hist = np.asarray(table["tacf6_hist"][i].as_py(),
                              dtype=np.float32).reshape(window, -1)

        _check(np.array_equal(got_state, state_all[t]), f"state mismatch at row {i}", problems)
        _check(np.array_equal(got_abs, action_all[t]), f"action_abs mismatch at row {i}", problems)
        _check(np.allclose(got_chunk, want_chunk, atol=1e-6),
               f"action_chunk mismatch at row {i} (re-derived under "
               f"{describe_anchor(spec)})", problems)
        _check(np.allclose(got_hist, want_hist, atol=1e-6),
               f"tacf6_hist mismatch at row {i}", problems)
        if "prev_command" in table.column_names:
            got_prev = np.asarray(table["prev_command"][i].as_py(), dtype=np.float32)
            _check(np.array_equal(got_prev, want_prev[t]),
                   f"prev_command mismatch at row {i}", problems)
    logger.info("[verify] round-trip against raw %s ep%d: exact under %s",
                season, ep_spec.episode_index, describe_anchor(spec))
    return problems


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Verify a prepared origami-flat dataset.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--src-root", default="",
                        help="raw season tree, to re-derive one episode exactly")
    parser.add_argument("--montage", default="",
                        help="write a 2x5 deform montage PNG here")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    problems = verify(args.root, n_samples=args.samples, seed=args.seed,
                      src_root=args.src_root, montage_path=args.montage)
    if problems:
        logger.error("[verify] %d PROBLEM(S):", len(problems))
        for message in problems:
            logger.error("           %s", message)
        return 1
    logger.info("[verify] all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
