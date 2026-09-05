"""Merge (§5.3) round-trip test — REDESIGN_PLAN.md §12 step 7.

Builds two small shards from the real fixture season (truncated via
``max_frames_per_episode`` so each shard converts in seconds) and merges them, checking that
episode/frame indices are renumbered into one global sequence and the merged root is readable
by the real ``LeRobotDataset``.
"""
from __future__ import annotations

import json

import pytest

from origami.constants import INSTRUCTION, REPO_ROOT
from origami.convert import ConvertConfig, convert_season
from origami.kinematics import LockedConfig, OrigamiKinematics
from origami.merge import merge_shards

FIXTURE_SEASON = REPO_ROOT / "season_POC22061_2026_05_23_19_21_25_train"
N_FRAMES = 72  # > MIN_EPISODE_LENGTH (64)


@pytest.fixture(scope="module")
def two_shards(tmp_path_factory):
    pytest.importorskip("pyarrow")
    kin = OrigamiKinematics(locked=LockedConfig.zeros())
    cfg = ConvertConfig(instruction=INSTRUCTION)
    roots = []
    for i in range(2):
        out = tmp_path_factory.mktemp(f"shard_{i}")
        convert_season(FIXTURE_SEASON, out, kin, cfg, max_episodes=1, max_frames_per_episode=N_FRAMES)
        roots.append(out / FIXTURE_SEASON.name)
    return roots


@pytest.fixture(scope="module")
def merged(tmp_path_factory, two_shards):
    out_root = tmp_path_factory.mktemp("merged")
    merge_shards(two_shards, out_root)
    return out_root


def test_merge_renumbers_episodes_and_frames(merged):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("origami/eef62_merge_test", root=str(merged))
    assert len(ds) == 2 * N_FRAMES

    last_of_first = ds[N_FRAMES - 1]
    first_of_second = ds[N_FRAMES]
    assert int(last_of_first["episode_index"]) == 0
    assert int(first_of_second["episode_index"]) == 1
    assert int(first_of_second["frame_index"]) == 0
    assert int(first_of_second["index"]) == N_FRAMES


def test_merge_combines_norm_stats(merged):
    stats = json.loads((merged / "meta" / "trex_norm_stats.json").read_text())
    block = next(iter(stats.values()))
    assert block["num_transitions"] == 2 * N_FRAMES
    assert block["num_trajectories"] == 2


def test_merge_combines_origami_prep(merged):
    prep = json.loads((merged / "meta" / "origami_prep.json").read_text())
    assert len(prep["seasons"]) == 2
    assert len(prep["truncated"]) == 2
    assert prep["locked_digest"] == LockedConfig.zeros().digest()


def test_merge_rejects_mismatched_locked_config(tmp_path_factory, two_shards):
    """§3.2: shards converted under different LockedConfig values must not be silently merged."""
    import shutil

    shard_a, shard_b = two_shards
    tampered = tmp_path_factory.mktemp("tampered_shard_b") / shard_b.name
    shutil.copytree(shard_b, tampered)
    prep_path = tampered / "meta" / "origami_prep.json"
    prep = json.loads(prep_path.read_text())
    prep["locked_digest"] = "deadbeef"
    prep_path.write_text(json.dumps(prep))

    out_root = tmp_path_factory.mktemp("merged_bad")
    with pytest.raises(ValueError, match="different LockedConfig digests"):
        merge_shards([shard_a, tampered], out_root)
