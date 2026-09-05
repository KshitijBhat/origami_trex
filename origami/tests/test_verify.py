"""Root-scale gate checks — REDESIGN_PLAN.md §10 / §12 step 8.

Reuses the same 2-shard merge fixture pattern as ``test_merge.py`` (real FK, real
decode/encode, real merge) so G8/G8b, G6, G4 run against real converted+merged data.
"""
from __future__ import annotations

import pytest

from origami import verify
from origami.constants import INSTRUCTION, REPO_ROOT
from origami.convert import ConvertConfig, convert_season
from origami.kinematics import LockedConfig, OrigamiKinematics
from origami.merge import merge_shards

FIXTURE_SEASON = REPO_ROOT / "season_POC22061_2026_05_23_19_21_25_train"
N_FRAMES = 72


@pytest.fixture(scope="module")
def merged_root(tmp_path_factory):
    pytest.importorskip("pyarrow")
    kin = OrigamiKinematics(locked=LockedConfig.zeros())
    cfg = ConvertConfig(instruction=INSTRUCTION)
    shard_roots = []
    for i in range(2):
        out = tmp_path_factory.mktemp(f"verify_shard_{i}")
        convert_season(FIXTURE_SEASON, out, kin, cfg, max_episodes=1, max_frames_per_episode=N_FRAMES)
        shard_roots.append(out / FIXTURE_SEASON.name)
    out_root = tmp_path_factory.mktemp("verify_merged")
    merge_shards(shard_roots, out_root)
    return out_root


def test_g8_passes_on_real_merged_stats(merged_root):
    assert verify.verify_g8(merged_root) is True


def test_g6_passes_on_real_merged_deform_videos(merged_root):
    assert verify.verify_g6(merged_root, n_sample=50) is True


def test_g4_passes_on_real_merged_frames(merged_root):
    assert verify.verify_g4(merged_root, n_sample=50) is True


def test_g18_matches_when_seasons_equal_split(merged_root, monkeypatch):
    monkeypatch.setattr(
        verify, "parse_splits",
        lambda: {"train": [FIXTURE_SEASON.name], "val": []},
    )
    monkeypatch.setattr(verify, "validate_splits", lambda splits: None)
    assert verify.verify_g18(merged_root, split="train") is True


def test_g18_fails_when_seasons_dont_match_split(merged_root, monkeypatch):
    monkeypatch.setattr(
        verify, "parse_splits",
        lambda: {"train": ["some_other_season"], "val": []},
    )
    monkeypatch.setattr(verify, "validate_splits", lambda splits: None)
    assert verify.verify_g18(merged_root, split="train") is False
