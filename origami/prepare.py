"""CLI: fetch -> convert -> merge. REDESIGN_PLAN.md §5.5.

Concurrency note (simplification, documented): §5.5 describes a ``ProcessPoolExecutor``
over seasons with a separate ``ThreadPoolExecutor`` prefetching downloads so at most
``--disk-budget`` seasons are resident on disk at once. This implements the same *resource
bound* (a multiprocessing semaphore of size ``disk-budget``, acquired around
download+convert+drop inside each worker) with one pool instead of two layered pools --
simpler to get right, same guarantee (never more than ``disk-budget`` seasons' raw video on
disk simultaneously), at the cost of not overlapping "downloading season N+1" with
"converting season N" within the disk budget. Fine for a first real run; revisit only if
profiling on the actual prep server shows the pipeline is download-bound.
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from origami.constants import INSTRUCTION, URDF_PATH
from origami.convert import ConvertConfig, convert_season
from origami.fetch import (
    download_season,
    download_season_meta_and_data,
    drop_season,
    have_season,
    list_hub_seasons,
    season_has_lerobot3_data,
    validate_token,
)
from origami.kinematics import LockedConfig, OrigamiKinematics, arm_default_from_state_median, urdf_sha256
from origami.merge import merge_shards
from origami.splits import parse_splits, validate_splits
from origami.verify import run_all_gates

logger = logging.getLogger(__name__)

BYTES_PER_GB = 1024**3
DISK_PER_SEASON_GB = 3.5  # §5.6: peak on-disk size of one season's raw download
OUTPUT_PER_SEASON_GB = 0.85  # §5.6: converted-shard size per season

# The season the plan cites as the local fixture, hub-absent by construction (§1.1a) --
# folding it into a split would contaminate the held-out protocol.
FIXTURE_SEASON_NAME = "season_POC22061_2026_07_23_10_20_33_train"


@dataclass
class PrepConfig:
    """Recorded in ``meta/origami_prep.json`` at the merged root; changing any of these
    fields invalidates an existing root (§5.5) -- refuse to write into it.

    Every field here must either be a real CLI-configurable knob that's actually threaded
    into ``ConvertConfig`` (``instruction``, ``image_size``, ``deform_size``), or a genuine
    fixed constant enforced elsewhere (``deform_codec`` is hardcoded in
    ``convert.py``'s ``LOSSLESS_DEFORM_ENCODER``; ``frame_stride`` is fixed by D4) -- never a
    field that looks configurable but silently isn't, which made this comparison vacuous.
    """

    instruction: str
    extra_seasons: str = "none"
    image_size: tuple[int, int] = (224, 224)
    deform_size: tuple[int, int] = (240, 240)
    deform_codec: str = "lossless_h264"  # fixed: convert.py's LOSSLESS_DEFORM_ENCODER (§5.2)
    frame_stride: int = 1  # fixed: D4 (settled, not an ablation knob)

    def as_dict(self) -> dict:
        return {
            "instruction": self.instruction,
            "extra_seasons": self.extra_seasons,
            "image_size": list(self.image_size),
            "deform_size": list(self.deform_size),
            "deform_codec": self.deform_codec,
            "frame_stride": self.frame_stride,
        }


def resolve_season_list(split: str, extra_seasons: str, token: str) -> list[str]:
    """G18: parse+validate the split lists, assert every season resolves on the hub, assert
    the fixture season is absent from both, fold in the opt-in extras (train only)."""
    splits = parse_splits()
    validate_splits(splits)

    for name in (splits["train"], splits["val"]):
        assert FIXTURE_SEASON_NAME not in name, (
            f"{FIXTURE_SEASON_NAME} (the fixture season) must not appear in either split list"
        )

    hub_seasons = list_hub_seasons(token)
    seasons = list(splits[split])
    missing = [s for s in seasons if s not in hub_seasons]
    assert not missing, f"season(s) in the '{split}' split don't resolve on the hub: {missing}"

    if extra_seasons == "train" and split == "train":
        extras = sorted(hub_seasons - set(splits["train"]) - set(splits["val"]))
        seasons = seasons + extras
    return seasons


def check_disk_budget(out_root: Path, cache_root: Path, n_seasons: int, disk_budget: int) -> None:
    output_estimate_gb = n_seasons * OUTPUT_PER_SEASON_GB
    required_gb = output_estimate_gb + disk_budget * DISK_PER_SEASON_GB
    for path in (out_root, cache_root):
        path.mkdir(parents=True, exist_ok=True)
        free_gb = shutil.disk_usage(path).free / BYTES_PER_GB
        assert free_gb >= required_gb, (
            f"{path}: only {free_gb:.1f} GB free, need >= {required_gb:.1f} GB "
            f"({output_estimate_gb:.1f} GB estimated output + "
            f"{disk_budget * DISK_PER_SEASON_GB:.1f} GB disk-budget headroom)"
        )


def phase0_locked_config(
    seasons: list[str], cache_root: Path, token: str
) -> tuple[LockedConfig, np.ndarray, np.ndarray, int, list[str]]:
    """§5.5 phase 0: meta+data only (no video) per season, for the exact frame total and to
    compute ``LockedConfig`` + arm medians over the whole split -- must run before phase 1.

    §3.3 is explicit that this is computed **once over the training split, then frozen** --
    callers must only ever pass train seasons here, never whatever ``--split`` is currently
    being prepared (§3.2: the baked ``action`` representation is invariant to the lock value,
    but the absolute 62-D ``state``/``action_abs`` and their q01/q99 stats are not -- a val
    run computing its own median would silently live in a different frame than train's).

    Some hub seasons were never backfilled to lerobot3.0 (meta/videos present, no ``data/``
    at all) -- skip those here rather than crashing on a missing parquet dir; they're also
    skipped in phase 1 (``run_phase1``), so they never enter the merge. Returns the actually-
    used season subset alongside the stats, for an accurate ``seasons_used`` record.
    """
    all_states = []
    total_frames = 0
    seasons_used = []
    for season in seasons:
        if not season_has_lerobot3_data(season, token):
            logger.warning(
                "phase0: season %s has no lerobot3.0/data on the hub (never backfilled) -- skipping", season,
            )
            continue
        local_dir = download_season_meta_and_data(season, cache_root, token)
        df = pd.read_parquet(Path(local_dir) / "lerobot3.0" / "data", columns=["observation.state"])
        states = np.stack(df["observation.state"].to_numpy())
        all_states.append(states)
        total_frames += len(states)
        seasons_used.append(season)
        drop_season(season, cache_root)
    all_states = np.concatenate(all_states, axis=0)
    locked = LockedConfig.from_state_median(all_states)
    left_default, right_default = arm_default_from_state_median(all_states)
    return locked, left_default, right_default, total_frames, seasons_used


def write_locked_config(
    path: Path, locked: LockedConfig, default_q_left7: np.ndarray, default_q_right7: np.ndarray,
    total_frames: int, seasons_used: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "lower_body": locked.lower_body.tolist(), "neck": locked.neck.tolist(),
        "left_hand": locked.left_hand.tolist(), "right_hand": locked.right_hand.tolist(),
        "default_q_left7": np.asarray(default_q_left7).tolist(),
        "default_q_right7": np.asarray(default_q_right7).tolist(),
        "digest": locked.digest(),
        "total_frames": total_frames,
        "n_seasons_used": len(seasons_used),
        "seasons_used": seasons_used,
    }, indent=2))


def load_locked_config(path: Path) -> tuple[LockedConfig, np.ndarray, np.ndarray, int]:
    data = json.loads(path.read_text())
    locked = LockedConfig(
        lower_body=np.array(data["lower_body"]), neck=np.array(data["neck"]),
        left_hand=np.array(data["left_hand"]), right_hand=np.array(data["right_hand"]),
    )
    assert locked.digest() == data["digest"], f"{path}: locked_config.json's own digest field is stale/corrupt"
    return locked, np.array(data["default_q_left7"]), np.array(data["default_q_right7"]), data["total_frames"]


def _process_one_season(
    season: str,
    cache_root: str,
    shard_root: str,
    urdf_path: str,
    locked_dict: dict,
    default_q_left7: list,
    default_q_right7: list,
    cfg_dict: dict,
    token: str,
    semaphore,
) -> dict:
    """Runs in a worker process: acquire the disk-budget semaphore, download (if not already
    present), convert, write manifest fields, release the semaphore, drop the raw season."""
    semaphore.acquire()
    try:
        t0 = time.time()
        if not have_season(season, Path(cache_root)):
            download_season(season, Path(cache_root), token)

        locked = LockedConfig(
            lower_body=np.array(locked_dict["lower_body"]), neck=np.array(locked_dict["neck"]),
            left_hand=np.array(locked_dict["left_hand"]), right_hand=np.array(locked_dict["right_hand"]),
        )
        kin = OrigamiKinematics(
            urdf_path=Path(urdf_path), locked=locked,
            default_q_left7=np.array(default_q_left7), default_q_right7=np.array(default_q_right7),
        )
        cfg = ConvertConfig(
            instruction=cfg_dict["instruction"],
            image_size=tuple(cfg_dict["image_size"]),
            deform_size=tuple(cfg_dict["deform_size"]),
        )
        result = convert_season(Path(cache_root) / season, Path(shard_root), kin, cfg)

        manifest_entry = {
            "season": season, "n_episodes": result.n_episodes, "n_frames": result.n_frames,
            "truncated": result.truncated, "rejected_short_episodes": result.rejected_short_episodes,
            "elapsed_s": time.time() - t0, "locked_digest": locked.digest(),
        }
    finally:
        drop_season(season, Path(cache_root))
        semaphore.release()
    return manifest_entry


def _read_manifest(manifest_path: Path) -> dict[str, dict]:
    """Single source of truth for phase-1 resumability and results -- keyed by season, last
    entry wins (a retried season's new entry supersedes its earlier ``failed`` one)."""
    entries: dict[str, dict] = {}
    if not manifest_path.exists():
        return entries
    for line in manifest_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            logger.warning(
                "%s: skipping a malformed trailing line (likely a crash mid-write); "
                "that season will be retried", manifest_path,
            )
            continue
        entries[entry["season"]] = entry
    return entries


def run_phase1(
    seasons: list[str],
    cache_root: Path,
    shard_root: Path,
    urdf_path: Path,
    locked: LockedConfig,
    default_q_left7: np.ndarray,
    default_q_right7: np.ndarray,
    cfg: ConvertConfig,
    token: str,
    workers: int,
    disk_budget: int,
) -> tuple[list[Path], dict[str, dict]]:
    """Convert every season not already recorded ``"status": "done"`` in ``manifest.jsonl``,
    ``workers``-way parallel, at most ``disk_budget`` seasons resident on disk at once.

    Returns ``(shard_roots, manifest_entries)`` -- both derived from the same parsed
    manifest, never a separate DONE-marker file (a season whose manifest line was written
    but whose marker-touch never ran, e.g. a crash in between, must not disagree with itself
    about being done). A season whose conversion raises gets a durable ``"status": "failed"``
    manifest entry (§5.1: "mark the season failed and continue") instead of only a log line,
    so a re-run retries exactly the failed/missing seasons and the failure is auditable
    without needing to have kept stdout.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    shard_root = Path(shard_root)
    shard_root.mkdir(parents=True, exist_ok=True)
    manifest_path = shard_root / "manifest.jsonl"

    all_entries = _read_manifest(manifest_path)
    done_seasons = {s for s, e in all_entries.items() if e.get("status") in ("done", "skipped")}
    todo = [s for s in seasons if s not in done_seasons]

    unbackfilled = [s for s in todo if not season_has_lerobot3_data(s, token)]
    if unbackfilled:
        logger.warning(
            "phase1: %d season(s) have no lerobot3.0/data on the hub (never backfilled) -- "
            "skipping: %s", len(unbackfilled), unbackfilled,
        )
        with open(manifest_path, "a") as mf:
            for season in unbackfilled:
                entry = {"season": season, "status": "skipped", "reason": "no lerobot3.0/data on hub"}
                mf.write(json.dumps(entry) + "\n")
                all_entries[season] = entry
        todo = [s for s in todo if s not in unbackfilled]

    logger.info("phase1: %d/%d seasons already done, %d to convert", len(done_seasons), len(seasons), len(todo))

    locked_dict = {
        "lower_body": locked.lower_body.tolist(), "neck": locked.neck.tolist(),
        "left_hand": locked.left_hand.tolist(), "right_hand": locked.right_hand.tolist(),
    }
    cfg_dict = {"instruction": cfg.instruction, "image_size": list(cfg.image_size), "deform_size": list(cfg.deform_size)}

    if todo:
        manager = multiprocessing.Manager()
        semaphore = manager.Semaphore(disk_budget)
        with ProcessPoolExecutor(max_workers=workers) as pool, open(manifest_path, "a") as mf:
            futures = {
                pool.submit(
                    _process_one_season, season, str(cache_root), str(shard_root), str(urdf_path),
                    locked_dict, default_q_left7.tolist(), default_q_right7.tolist(), cfg_dict,
                    token, semaphore,
                ): season
                for season in todo
            }
            for fut in as_completed(futures):
                season = futures[fut]
                try:
                    entry = fut.result()
                    entry["status"] = "done"
                except Exception as e:
                    logger.exception("phase1: season %s failed", season)
                    entry = {"season": season, "status": "failed", "error": repr(e)}
                mf.write(json.dumps(entry) + "\n")
                mf.flush()
                all_entries[season] = entry

    failed = [s for s in seasons if all_entries.get(s, {}).get("status") == "failed"]
    if failed:
        logger.warning("phase1: %d season(s) failed and are excluded from the merge: %s", len(failed), failed)

    shard_roots = [shard_root / s for s in seasons if all_entries.get(s, {}).get("status") == "done"]
    return shard_roots, all_entries


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--disk-budget", type=int, default=3)
    parser.add_argument("--urdf", type=Path, default=URDF_PATH)
    parser.add_argument("--extra-seasons", choices=["none", "train"], default="none")
    parser.add_argument("--instruction", default=INSTRUCTION)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--image-size", type=int, nargs=2, default=[224, 224], metavar=("H", "W"))
    parser.add_argument("--deform-size", type=int, nargs=2, default=[240, 240], metavar=("H", "W"))
    parser.add_argument(
        "--phase", choices=["locked", "all"], default="all",
        help="'locked': only compute/refresh the shared train-split LockedConfig "
             "(--locked-config-path) and exit. 'all' (default): locked (if needed) + convert "
             "+ merge + verify.",
    )
    parser.add_argument(
        "--locked-config-path", type=Path, default=None,
        help="Shared locked_config.json (§3.3: computed once over train, frozen, reused for "
             "val). Defaults to <cache-root>/locked_config.json. Must already exist when "
             "--split val (val never computes its own).",
    )
    parser.add_argument(
        "--limit-seasons", type=int, default=None,
        help="Only prepare the first N seasons of the resolved split -- for a quick local "
             "smoke test, not a real prep run. Use a separate --out-root from any real run.",
    )
    args = parser.parse_args(argv)

    assert args.hf_token, "HF_TOKEN required: pass --hf-token or set the HF_TOKEN env var"
    validate_token(args.hf_token)

    out_root = Path(args.out_root)
    cache_root = Path(args.cache_root)
    image_size, deform_size = tuple(args.image_size), tuple(args.deform_size)
    prep_cfg = PrepConfig(
        instruction=args.instruction, extra_seasons=args.extra_seasons,
        image_size=image_size, deform_size=deform_size,
    )

    prep_meta_path = out_root / "meta" / "origami_prep.json"
    if prep_meta_path.exists():
        existing = json.loads(prep_meta_path.read_text())
        assert existing.get("prep_config") == prep_cfg.as_dict(), (
            f"{out_root} was prepared with a different config {existing.get('prep_config')} "
            f"than requested {prep_cfg.as_dict()} -- refusing to write into it (§5.5)"
        )

    seasons = resolve_season_list(args.split, args.extra_seasons, args.hf_token)
    if args.limit_seasons is not None:
        seasons = seasons[: args.limit_seasons]
        logger.info("--limit-seasons %d: restricting to %s", args.limit_seasons, seasons)
    check_disk_budget(out_root, cache_root, len(seasons), args.disk_budget)

    # §3.3/§3.2: LockedConfig is computed once over the TRAIN split and frozen -- never
    # recomputed per --split. Load the shared artifact if it exists; only a train run may
    # create it fresh (a val run with none yet is a usage error, not something to silently
    # compute over val's own seasons).
    locked_config_path = args.locked_config_path or (cache_root / "locked_config.json")
    if locked_config_path.exists():
        locked, default_left, default_right, locked_total_frames = load_locked_config(locked_config_path)
        logger.info(
            "phase 0: loaded existing locked config from %s (%d frames, digest %s)",
            locked_config_path, locked_total_frames, locked.digest(),
        )
    else:
        assert args.split == "train", (
            f"{locked_config_path} doesn't exist yet -- LockedConfig (§3.3) must be computed "
            f"once over the TRAIN split before preparing '{args.split}'. Run with "
            f"--split train first (add --phase locked to only run phase 0)."
        )
        logger.info("phase 0: computing locked config over %d train seasons", len(seasons))
        locked, default_left, default_right, locked_total_frames, seasons_used = phase0_locked_config(
            seasons, cache_root, args.hf_token)
        write_locked_config(
            locked_config_path, locked, default_left, default_right, locked_total_frames, seasons_used)
        logger.info(
            "phase 0 done: %d total frames, locked digest %s, written to %s",
            locked_total_frames, locked.digest(), locked_config_path,
        )

    if args.phase == "locked":
        logger.info("--phase locked: stopping after phase 0.")
        return

    logger.info("phase 1: converting %d seasons (workers=%d, disk_budget=%d)", len(seasons), args.workers, args.disk_budget)
    cfg = ConvertConfig(instruction=args.instruction, image_size=image_size, deform_size=deform_size)
    shard_roots, manifest_entries = run_phase1(
        seasons, cache_root, out_root.parent / f"{out_root.name}_shards", args.urdf,
        locked, default_left, default_right, cfg, args.hf_token, args.workers, args.disk_budget,
    )
    # §1.1a: replace the corpus-size estimate with the exact sum phase 1 actually measured
    # (each season's SeasonResult.n_frames), for whichever split this run is -- not the
    # train-only total_frames stored inside locked_config.json.
    exact_total_frames = sum(
        manifest_entries[s]["n_frames"] for s in seasons if manifest_entries.get(s, {}).get("status") == "done"
    )
    logger.info("phase 1 done: %d exact frames across %d converted seasons", exact_total_frames, len(shard_roots))

    logger.info("phase 2: merging %d shards", len(shard_roots))
    merge_shards(shard_roots, out_root)

    logger.info("phase 2: verifying merged root (G18, G8/G8b, G6, G4)")
    gate_results = run_all_gates(out_root, args.split, args.extra_seasons)
    logger.info("verify: %s", gate_results)
    assert all(gate_results.values()), f"post-merge verification failed: {gate_results}"

    prep_meta_path.parent.mkdir(parents=True, exist_ok=True)
    prep_meta = json.loads(prep_meta_path.read_text()) if prep_meta_path.exists() else {}
    prep_meta["prep_config"] = prep_cfg.as_dict()
    prep_meta["instruction"] = args.instruction
    prep_meta["urdf_sha256"] = urdf_sha256(args.urdf)
    prep_meta["locked_config"] = {
        "lower_body": locked.lower_body.tolist(), "neck": locked.neck.tolist(),
        "left_hand": locked.left_hand.tolist(), "right_hand": locked.right_hand.tolist(),
        "digest": locked.digest(),
    }
    prep_meta["total_frames"] = exact_total_frames
    prep_meta_path.write_text(json.dumps(prep_meta, indent=2))
    logger.info("done: %s", out_root)


if __name__ == "__main__":
    main()
