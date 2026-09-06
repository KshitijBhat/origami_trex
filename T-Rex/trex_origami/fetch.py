"""Per-season download from the Hugging Face hub, with the fat streams excluded.

A full season is ~4.6 GB in `lerobot3.0`, but `tactile_raw` alone is 3.6 GB of
that (73%) and T-Rex has no encoder for it; `head_right` is another 234 MB the
slow expert never sees.  Fetching only what we convert brings a season down to
~1.0 GB, which is what makes a 126-season sweep practical on a home connection.

The intended usage is stream-and-delete: fetch one season, convert it, drop the
source.  Peak disk then stays at `output-so-far + one season` instead of the
~300 GB the full source would need.
"""
from __future__ import annotations

import logging
import os
import shutil
from typing import List, Optional

from .seasons import HF_REPO_ID, VIDEO_KEYS

logger = logging.getLogger(__name__)


def season_allow_patterns(season: str) -> List[str]:
    """glob patterns covering exactly the files `prepare` reads."""
    base = f"{season}/lerobot3.0"
    patterns = [f"{base}/meta/**", f"{base}/data/**"]
    patterns += [f"{base}/videos/{key}/**" for key in VIDEO_KEYS]
    return patterns


def local_season_dir(cache_root: str, season: str) -> str:
    return os.path.join(cache_root, season)


def have_season(cache_root: str, season: str) -> bool:
    """True if every file `prepare` reads is actually present.

    Checks for real files, not just directories: an interrupted download leaves
    the tree in place with the mp4s missing, and a directory-only check would
    happily hand that to the converter.
    """
    import glob

    root = os.path.join(local_season_dir(cache_root, season), "lerobot3.0")
    if not os.path.exists(os.path.join(root, "meta", "info.json")):
        return False
    if not glob.glob(os.path.join(root, "meta", "episodes", "**", "*.parquet"), recursive=True):
        return False
    if not glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True):
        return False
    return all(glob.glob(os.path.join(root, "videos", key, "**", "*.mp4"), recursive=True)
               for key in VIDEO_KEYS)


def download_season(
    season: str,
    cache_root: str,
    repo_id: str = HF_REPO_ID,
    token: Optional[str] = None,
    max_workers: int = 8,
    revision: Optional[str] = None,
) -> str:
    """Download one season into `cache_root/<season>` and return that path.

    Uses `local_dir` (not the HF blob cache) so `drop_season` actually reclaims
    the space — a symlinked cache would leave the blobs behind.

    `revision` is the HF git ref to pull from (e.g. the `competition-paper-set`
    branch, which freezes a season set that has since partly rotated off
    `main`). Left `None`, `snapshot_download` resolves it to the repo's
    default branch, same as before this parameter existed.
    """
    from huggingface_hub import snapshot_download

    target = local_season_dir(cache_root, season)
    os.makedirs(cache_root, exist_ok=True)
    # Always call snapshot_download even when the season looks complete: it is a
    # cheap metadata round-trip when nothing is missing, and it repairs a
    # partial tree left behind by an interrupted run instead of letting the
    # converter fail on an absent mp4 halfway through.
    logger.info("[fetch] %s %s (revision=%s)", "verifying" if have_season(cache_root, season)
                else "downloading", season, revision or "default")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        allow_patterns=season_allow_patterns(season),
        local_dir=cache_root,
        token=token,
        max_workers=max_workers,
    )
    if not have_season(cache_root, season):
        raise RuntimeError(
            f"{season}: download finished but the expected lerobot3.0 tree is "
            f"missing under {target} — check the season name against the hub "
            f"revision {revision or 'default'}.")
    return target


def drop_season(cache_root: str, season: str) -> None:
    """Delete a downloaded season (called after a successful conversion)."""
    target = local_season_dir(cache_root, season)
    if os.path.isdir(target):
        shutil.rmtree(target, ignore_errors=True)
        logger.info("[fetch] dropped %s", season)


def list_hub_seasons(repo_id: str = HF_REPO_ID, token: Optional[str] = None,
                      revision: Optional[str] = None) -> List[str]:
    """Season directory names actually present on the hub.

    Useful to reconcile `seasons.py` (written from dataset.md) against the live
    repo before starting a long sweep. Pass the same `revision` the sweep will
    fetch from -- a season list pulled from the default branch can differ from
    what a named revision (e.g. a frozen competition-set branch) still has.
    """
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo_id, repo_type="dataset", token=token,
                                     revision=revision)
    return sorted({f.split("/", 1)[0] for f in files if f.startswith("season_")})
