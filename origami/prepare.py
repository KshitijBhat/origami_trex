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
    validate_token,
)
from origami.kinematics import LockedConfig, OrigamiKinematics, arm_default_from_state_median
from origami.merge import merge_shards
from origami.splits import parse_splits, validate_splits

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
    fields invalidates an existing root (§5.5) -- refuse to write into it."""

    image_size: tuple[int, int] = (224, 224)
    deform_size: tuple[int, int] = (240, 240)
    deform_codec: str = "lossless_h264"
    frame_stride: int = 1
    extra_seasons: str = "none"

    def as_dict(self) -> dict:
        return {
            "image_size": list(self.image_size),
            "deform_size": list(self.deform_size),
            "deform_codec": self.deform_codec,
            "frame_stride": self.frame_stride,
            "extra_seasons": self.extra_seasons,
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
) -> tuple[LockedConfig, np.ndarray, np.ndarray, int]:
    """§5.5 phase 0: meta+data only (no video) per season, for the exact frame total and to
    compute ``LockedConfig`` + arm medians over the whole split -- must run before phase 1."""
    all_states = []
    total_frames = 0
    for season in seasons:
        local_dir = download_season_meta_and_data(season, cache_root, token)
        df = pd.read_parquet(Path(local_dir) / "lerobot3.0" / "data", columns=["observation.state"])
        states = np.stack(df["observation.state"].to_numpy())
        all_states.append(states)
        total_frames += len(states)
        drop_season(season, cache_root)
    all_states = np.concatenate(all_states, axis=0)
    locked = LockedConfig.from_state_median(all_states)
    left_default, right_default = arm_default_from_state_median(all_states)
    return locked, left_default, right_default, total_frames


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
) -> list[Path]:
    """Convert every season not already marked DONE, ``workers``-way parallel, at most
    ``disk_budget`` seasons resident on disk at once. Returns the shard roots to merge."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    shard_root = Path(shard_root)
    shard_root.mkdir(parents=True, exist_ok=True)
    manifest_path = shard_root / "manifest.jsonl"

    done_seasons = set()
    if manifest_path.exists():
        for line in manifest_path.read_text().splitlines():
            done_seasons.add(json.loads(line)["season"])

    todo = [s for s in seasons if s not in done_seasons]
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
                except Exception:
                    logger.exception("phase1: season %s failed", season)
                    continue
                mf.write(json.dumps(entry) + "\n")
                mf.flush()
                (shard_root / season / "DONE").touch()

    return [shard_root / s for s in seasons if (shard_root / s / "DONE").exists()]


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
    args = parser.parse_args(argv)

    assert args.hf_token, "HF_TOKEN required: pass --hf-token or set the HF_TOKEN env var"
    validate_token(args.hf_token)

    out_root = Path(args.out_root)
    cache_root = Path(args.cache_root)
    prep_cfg = PrepConfig(extra_seasons=args.extra_seasons)

    prep_meta_path = out_root / "meta" / "origami_prep.json"
    if prep_meta_path.exists():
        existing = json.loads(prep_meta_path.read_text())
        assert existing.get("prep_config") == prep_cfg.as_dict(), (
            f"{out_root} was prepared with a different config {existing.get('prep_config')} "
            f"than requested {prep_cfg.as_dict()} -- refusing to write into it (§5.5)"
        )

    seasons = resolve_season_list(args.split, args.extra_seasons, args.hf_token)
    check_disk_budget(out_root, cache_root, len(seasons), args.disk_budget)

    logger.info("phase 0: locked config over %d seasons", len(seasons))
    locked, default_left, default_right, total_frames = phase0_locked_config(seasons, cache_root, args.hf_token)
    logger.info("phase 0 done: %d total frames, locked digest %s", total_frames, locked.digest())

    logger.info("phase 1: converting %d seasons (workers=%d, disk_budget=%d)", len(seasons), args.workers, args.disk_budget)
    cfg = ConvertConfig(instruction=args.instruction)
    shard_roots = run_phase1(
        seasons, cache_root, out_root.parent / f"{out_root.name}_shards", args.urdf,
        locked, default_left, default_right, cfg, args.hf_token, args.workers, args.disk_budget,
    )

    logger.info("phase 2: merging %d shards", len(shard_roots))
    merge_shards(shard_roots, out_root)

    prep_meta_path.parent.mkdir(parents=True, exist_ok=True)
    prep_meta = json.loads(prep_meta_path.read_text()) if prep_meta_path.exists() else {}
    prep_meta["prep_config"] = prep_cfg.as_dict()
    prep_meta_path.write_text(json.dumps(prep_meta, indent=2))
    logger.info("done: %s", out_root)


if __name__ == "__main__":
    main()
