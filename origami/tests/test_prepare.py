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

import numpy as np
import pytest

import origami.prepare as prepare
from origami.kinematics import LockedConfig
from origami.prepare import FIXTURE_SEASON_NAME, PrepConfig, check_disk_budget, resolve_season_list


def test_prep_config_as_dict_is_stable():
    a = PrepConfig(instruction="fold it")
    b = PrepConfig(instruction="fold it")
    assert a.as_dict() == b.as_dict()
    c = PrepConfig(instruction="fold it", frame_stride=2)
    assert c.as_dict() != a.as_dict()


def test_prep_config_instruction_and_image_size_are_compared():
    """§5.5 invalidation: fields that actually vary must be in the compared dict -- catches
    the bug where --instruction varied but wasn't part of as_dict() at all."""
    base = PrepConfig(instruction="fold it")
    assert PrepConfig(instruction="fold it differently").as_dict() != base.as_dict()
    assert PrepConfig(instruction="fold it", image_size=(384, 384)).as_dict() != base.as_dict()


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


# ── phase-1 resumability: manifest.jsonl is the ONLY source of truth ─────────
def test_run_phase1_skips_already_done_seasons(tmp_path):
    """Pre-populate manifest.jsonl (status=done) for every season, no DONE-marker files at
    all -- run_phase1 must not submit any conversion work and must still return all shard
    roots + the parsed manifest entries, derived purely from the manifest."""
    import numpy as np

    shard_root = tmp_path / "shards"
    shard_root.mkdir(parents=True)
    seasons = ["season_x", "season_y"]
    manifest_path = shard_root / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(
            json.dumps({"season": s, "n_episodes": 1, "n_frames": 10, "status": "done"}) for s in seasons
        ) + "\n"
    )

    from origami.kinematics import LockedConfig
    from origami.convert import ConvertConfig

    roots, entries = prepare.run_phase1(
        seasons, cache_root=tmp_path / "cache", shard_root=shard_root, urdf_path=tmp_path / "fake.urdf",
        locked=LockedConfig.zeros(), default_q_left7=np.zeros(7), default_q_right7=np.zeros(7),
        cfg=ConvertConfig(instruction="x"), token="fake", workers=1, disk_budget=1,
    )
    assert set(roots) == {shard_root / s for s in seasons}
    assert all(entries[s]["status"] == "done" for s in seasons)


def test_run_phase1_retries_failed_seasons_not_done_ones(tmp_path):
    """A season recorded as 'failed' must NOT be treated as done -- it must stay in `todo`
    (this only checks the skip-list bookkeeping, not a real retry-and-succeed conversion,
    which needs real multiprocessing/network per the module docstring above)."""
    shard_root = tmp_path / "shards"
    shard_root.mkdir(parents=True)
    manifest_path = shard_root / "manifest.jsonl"
    manifest_path.write_text(json.dumps({"season": "season_x", "status": "failed", "error": "boom"}) + "\n")

    entries = prepare._read_manifest(manifest_path)
    done = {s for s, e in entries.items() if e.get("status") == "done"}
    assert "season_x" not in done
    assert entries["season_x"]["status"] == "failed"


def test_read_manifest_ignores_malformed_trailing_line(tmp_path):
    """A crash mid-write can leave a truncated last line -- must not raise, and that season
    must not count as done (single source of truth: no separate marker to fall back on)."""
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps({"season": "season_x", "status": "done", "n_frames": 5}) + "\n"
        '{"season": "season_y", "status": "do'  # truncated
    )
    entries = prepare._read_manifest(manifest_path)
    assert entries["season_x"]["status"] == "done"
    assert "season_y" not in entries


# ── locked-config persistence (§3.2/§3.3): computed once over train, frozen, shared ──────
def test_write_load_locked_config_round_trips(tmp_path):
    locked = LockedConfig(
        lower_body=np.array([0.1, 0.2, 0.3, 0.4, 0.5]), neck=np.array([0.6, 0.7]),
        left_hand=np.zeros(22), right_hand=np.zeros(22),
    )
    path = tmp_path / "locked_config.json"
    prepare.write_locked_config(path, locked, np.arange(7.0), np.arange(7.0) + 1, total_frames=12345,
                                 seasons_used=["season_a", "season_b"])

    loaded, default_left, default_right, total_frames = prepare.load_locked_config(path)
    assert loaded.digest() == locked.digest()
    np.testing.assert_array_equal(default_left, np.arange(7.0))
    np.testing.assert_array_equal(default_right, np.arange(7.0) + 1)
    assert total_frames == 12345

    data = json.loads(path.read_text())
    assert data["seasons_used"] == ["season_a", "season_b"]
    assert data["n_seasons_used"] == 2


def test_load_locked_config_rejects_corrupted_digest(tmp_path):
    locked = LockedConfig.zeros()
    path = tmp_path / "locked_config.json"
    prepare.write_locked_config(path, locked, np.zeros(7), np.zeros(7), total_frames=1, seasons_used=["s"])

    data = json.loads(path.read_text())
    data["lower_body"] = [9.9, 9.9, 9.9, 9.9, 9.9]  # tamper without updating the digest field
    path.write_text(json.dumps(data))

    with pytest.raises(AssertionError, match="stale/corrupt"):
        prepare.load_locked_config(path)


def test_main_refuses_to_bootstrap_locked_config_from_val_split(tmp_path, monkeypatch):
    """§3.2/§3.3: LockedConfig must come from train. A val run with no locked_config.json
    yet must refuse, not silently compute one over val's own seasons."""
    monkeypatch.setattr(prepare, "validate_token", lambda token: None)
    monkeypatch.setattr(prepare, "resolve_season_list", lambda split, extra, token: ["season_c"])
    monkeypatch.setattr(prepare, "check_disk_budget", lambda *a, **k: None)

    out_root = tmp_path / "out_val"
    cache_root = tmp_path / "cache"
    with pytest.raises(AssertionError, match="must be computed once over the TRAIN split"):
        prepare.main([
            "--split", "val",
            "--out-root", str(out_root),
            "--cache-root", str(cache_root),
            "--hf-token", "fake",
        ])
