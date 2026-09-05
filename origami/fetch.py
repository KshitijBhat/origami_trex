"""Per-season HF stream-and-delete download. REDESIGN_PLAN.md §5.1.

The dataset repo is gated: anonymous reads return HTTP 401. ``validate_token`` must be
called (and pass) before any season download is attempted -- never discover a bad token at
season 1 of 126.
"""
from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import HfHubHTTPError

logger = logging.getLogger(__name__)

HF_REPO_ID = "SharpaIT/Robotic_Origami_Challenge"
NEEDED_VIDEO_KEYS = (
    "observation.images.head_left",
    "observation.images.wrist_left",
    "observation.images.wrist_right",
    "observation.images.tactile_deform",
)
RAW_VIDEO_KEY = "observation.images.tactile_raw"

MAX_ATTEMPTS = 3
BACKOFF_BASE_S = 2.0


def validate_token(token: str) -> None:
    """Authenticate against the gated repo before phase 0 starts (§5.1)."""
    HfApi().dataset_info(HF_REPO_ID, token=token)


def season_allow_patterns(season: str, include_raw: bool = False) -> list[str]:
    patterns = [
        f"{season}/lerobot3.0/meta/**",
        f"{season}/lerobot3.0/data/**",
    ]
    for key in NEEDED_VIDEO_KEYS:
        patterns.append(f"{season}/lerobot3.0/videos/{key}/**")
    if include_raw:
        patterns.append(f"{season}/lerobot3.0/videos/{RAW_VIDEO_KEY}/**")
    patterns.append(f"{season}/lerobot3.0/.backfill_complete")
    return patterns


def season_meta_data_only_patterns(season: str) -> list[str]:
    """Phase 0 (§5.5): meta + data only, no video -- for the exact frame count + LockedConfig."""
    return [f"{season}/lerobot3.0/meta/**", f"{season}/lerobot3.0/data/**"]


def have_season(season: str, cache_root: Path, include_raw: bool = False) -> bool:
    """§5.1: an interrupted download leaves the tree with no media -- glob for the real
    artifacts (parquet/mp4), and require the .backfill_complete marker (§1.1a note: several
    of the hub's non-split "extras" seasons have meta-only uploads with no data/videos at
    all -- the marker's presence is what distinguishes a genuinely complete season)."""
    root = Path(cache_root) / season / "lerobot3.0"
    if not (root / ".backfill_complete").exists():
        return False
    if not list((root / "data").rglob("*.parquet")):
        return False
    for key in NEEDED_VIDEO_KEYS:
        if not list((root / "videos" / key).rglob("*.mp4")):
            return False
    if include_raw and not list((root / "videos" / RAW_VIDEO_KEY).rglob("*.mp4")):
        return False
    return True


def download_season(
    season: str, cache_root: Path, token: str, include_raw: bool = False
) -> str:
    """snapshot_download with exponential backoff, per §5.1."""
    local_dir = Path(cache_root) / season
    patterns = [p[len(season) + 1:] for p in season_allow_patterns(season, include_raw)]
    last_exc = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            snapshot_download(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                allow_patterns=patterns,
                local_dir=str(local_dir),
                max_workers=8,
                token=token,
            )
            return str(local_dir)
        except HfHubHTTPError as e:
            last_exc = e
            if attempt < MAX_ATTEMPTS:
                sleep_s = BACKOFF_BASE_S * (2 ** (attempt - 1))
                logger.warning(
                    "download_season(%s) attempt %d/%d failed: %s; retrying in %.1fs",
                    season, attempt, MAX_ATTEMPTS, e, sleep_s,
                )
                time.sleep(sleep_s)
    raise RuntimeError(f"download_season({season}) failed after {MAX_ATTEMPTS} attempts") from last_exc


def download_season_meta_and_data(season: str, cache_root: Path, token: str) -> str:
    """Phase 0 (§5.5): fetch only meta+data (~66MB, no video) for one season."""
    local_dir = Path(cache_root) / season
    patterns = [p[len(season) + 1:] for p in season_meta_data_only_patterns(season)]
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        allow_patterns=patterns,
        local_dir=str(local_dir),
        max_workers=8,
        token=token,
    )
    return str(local_dir)


def drop_season(season: str, cache_root: Path) -> None:
    """Stream-and-delete: free disk once a season has been converted."""
    path = Path(cache_root) / season
    if path.exists():
        shutil.rmtree(path)


def list_hub_seasons(token: str) -> set[str]:
    """All season directories on the hub, via the (public) tree listing."""
    api = HfApi()
    entries = api.list_repo_tree(HF_REPO_ID, repo_type="dataset", token=token)
    seasons = set()
    for e in entries:
        name = e.path.split("/")[0]
        if name.startswith("season_"):
            seasons.add(name)
    return seasons
