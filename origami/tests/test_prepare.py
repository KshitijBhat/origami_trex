"""prepare.py unit tests — REDESIGN_PLAN.md §5.5 / §12 step 7.

CPU-only, network-free: everything that would need the real hub (``list_hub_seasons``,
``download_season*``) or real multiprocessing conversion work is monkeypatched. The
``run_phase1``/``ProcessPoolExecutor`` conversion path itself is exercised for real by
``test_merge.py`` (via ``convert_season`` + ``merge_shards`` directly) and step 6/7's manual
smoke runs (PROGRESS.md) -- this file covers the CLI-level logic that sits around that:
G18 season-list resolution, the disk-budget guard, and phase-1 resumability bookkeeping.
"""
from __future__ import annotations

import json

import pytest

import origami.prepare as prepare
from origami.prepare import FIXTURE_SEASON_NAME, PrepConfig, check_disk_budget, resolve_season_list


def test_prep_config_as_dict_is_stable():
    a = PrepConfig()
    b = PrepConfig()
    assert a.as_dict() == b.as_dict()
    c = PrepConfig(frame_stride=2)
    assert c.as_dict() != a.as_dict()


# ── G18: season-list resolution ─────────────────────────────────────────────
def _fake_splits():
    return {"train": ["season_a", "season_b"], "val": ["season_c"]}


def test_resolve_season_list_happy_path(monkeypatch):
    monkeypatch.setattr(prepare, "parse_splits", _fake_splits)
    monkeypatch.setattr(prepare, "validate_splits", lambda splits: None)
    monkeypatch.setattr(prepare, "list_hub_seasons", lambda token: {"season_a", "season_b", "season_c"})

    seasons = resolve_season_list("train", "none", token="fake")
    assert seasons == ["season_a", "season_b"]


def test_resolve_season_list_missing_on_hub_raises(monkeypatch):
    monkeypatch.setattr(prepare, "parse_splits", _fake_splits)
    monkeypatch.setattr(prepare, "validate_splits", lambda splits: None)
    monkeypatch.setattr(prepare, "list_hub_seasons", lambda token: {"season_a"})  # season_b missing

    with pytest.raises(AssertionError, match="don't resolve on the hub"):
        resolve_season_list("train", "none", token="fake")


def test_resolve_season_list_rejects_fixture_in_split(monkeypatch):
    def splits_with_fixture():
        return {"train": ["season_a", FIXTURE_SEASON_NAME], "val": ["season_c"]}

    monkeypatch.setattr(prepare, "parse_splits", splits_with_fixture)
    monkeypatch.setattr(prepare, "validate_splits", lambda splits: None)
    monkeypatch.setattr(prepare, "list_hub_seasons", lambda token: {"season_a", "season_c", FIXTURE_SEASON_NAME})

    with pytest.raises(AssertionError, match="fixture season"):
        resolve_season_list("train", "none", token="fake")


def test_resolve_season_list_extras_train_only(monkeypatch):
    monkeypatch.setattr(prepare, "parse_splits", _fake_splits)
    monkeypatch.setattr(prepare, "validate_splits", lambda splits: None)
    monkeypatch.setattr(
        prepare, "list_hub_seasons",
        lambda token: {"season_a", "season_b", "season_c", "season_extra1", "season_extra2"},
    )

    train_seasons = resolve_season_list("train", "train", token="fake")
    assert set(train_seasons) == {"season_a", "season_b", "season_extra1", "season_extra2"}

    val_seasons = resolve_season_list("val", "train", token="fake")  # extras only fold into train
    assert val_seasons == ["season_c"]


# ── disk budget guard ────────────────────────────────────────────────────────
class _FakeUsage:
    def __init__(self, free_bytes):
        self.free = free_bytes


def test_check_disk_budget_passes_with_enough_space(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare.shutil, "disk_usage", lambda path: _FakeUsage(10_000 * prepare.BYTES_PER_GB))
    check_disk_budget(tmp_path / "out", tmp_path / "cache", n_seasons=126, disk_budget=3)


def test_check_disk_budget_raises_with_too_little_space(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare.shutil, "disk_usage", lambda path: _FakeUsage(1 * prepare.BYTES_PER_GB))
    with pytest.raises(AssertionError, match="only.*GB free"):
        check_disk_budget(tmp_path / "out", tmp_path / "cache", n_seasons=126, disk_budget=3)


# ── phase-1 resumability ─────────────────────────────────────────────────────
def test_run_phase1_skips_already_done_seasons(tmp_path):
    """Pre-populate DONE markers + manifest for every season; run_phase1 must not submit any
    conversion work and must still return all shard roots."""
    import numpy as np

    shard_root = tmp_path / "shards"
    seasons = ["season_x", "season_y"]
    for s in seasons:
        (shard_root / s).mkdir(parents=True)
        (shard_root / s / "DONE").touch()
    manifest_path = shard_root / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(json.dumps({"season": s, "n_episodes": 1, "n_frames": 10}) for s in seasons) + "\n"
    )

    from origami.kinematics import LockedConfig
    from origami.convert import ConvertConfig

    roots = prepare.run_phase1(
        seasons, cache_root=tmp_path / "cache", shard_root=shard_root, urdf_path=tmp_path / "fake.urdf",
        locked=LockedConfig.zeros(), default_q_left7=np.zeros(7), default_q_right7=np.zeros(7),
        cfg=ConvertConfig(instruction="x"), token="fake", workers=1, disk_budget=1,
    )
    assert set(roots) == {shard_root / s for s in seasons}
