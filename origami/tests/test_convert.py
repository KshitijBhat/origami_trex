"""Gate G4/G4b + writer/loader round-trip — REDESIGN_PLAN.md §5.2 / §12 step 6.

Runs the real ``convert_season`` pipeline (real FK, real PyAV decode/encode, real
``LeRobotDataset`` writer) end-to-end against the local fixture season, truncated to a
handful of frames via ``max_frames_per_episode`` so the test stays fast and memory-bounded
(a full ~8000-frame episode takes minutes and, before the streaming-decode fix in
``decode.py``, OOM-killed a 15GB dev machine -- see PROGRESS.md).
"""
from __future__ import annotations

import shutil

import numpy as np
import pytest

from origami.constants import INSTRUCTION, REPO_ROOT
from origami.convert import ConvertConfig, convert_season
from origami.kinematics import LockedConfig, OrigamiKinematics, delta9_to_matrix, matrix_to_rot6d
from utils.lerobot_common import DEFORM_KEYS

FIXTURE_SEASON = REPO_ROOT / "season_POC22061_2026_05_23_19_21_25_train"
N_FRAMES = 96  # > MIN_EPISODE_LENGTH (64), small enough to run in seconds


@pytest.fixture(scope="module")
def kin() -> OrigamiKinematics:
    return OrigamiKinematics(locked=LockedConfig.zeros())


@pytest.fixture(scope="module")
def converted(tmp_path_factory, kin):
    pytest.importorskip("pyarrow")
    out_root = tmp_path_factory.mktemp("convert_smoke")
    cfg = ConvertConfig(instruction=INSTRUCTION)
    result = convert_season(
        FIXTURE_SEASON, out_root, kin, cfg, max_episodes=1, max_frames_per_episode=N_FRAMES,
    )
    return out_root, result


def test_convert_season_writes_one_episode(converted):
    _, result = converted
    assert result.n_episodes == 1
    assert result.n_frames == N_FRAMES
    assert result.rejected_short_episodes == []


def test_writer_output_is_readable_by_lerobot_dataset(converted):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out_root, result = converted
    season_root = out_root / FIXTURE_SEASON.name
    ds = LeRobotDataset("origami/eef62_test", root=str(season_root))
    assert len(ds) == N_FRAMES

    item = ds[0]
    assert tuple(item["observation.images.head"].shape) == (3, 224, 224)
    assert tuple(item["observation.images.wrist_left"].shape) == (3, 224, 224)
    assert tuple(item["observation.images.wrist_right"].shape) == (3, 224, 224)
    assert tuple(item["observation.state"].shape) == (62,)
    assert tuple(item["action"].shape) == (16, 62)
    assert tuple(item["action_abs"].shape) == (62,)
    assert tuple(item["observation.tactile_f6"].shape) == (10, 6)
    for key in DEFORM_KEYS:
        assert tuple(item[key].shape) == (3, 240, 240)


def test_trex_norm_stats_written(converted):
    import json

    out_root, _ = converted
    season_root = out_root / FIXTURE_SEASON.name
    stats = json.loads((season_root / "meta" / "trex_norm_stats.json").read_text())
    block = next(iter(stats.values()))
    assert block["num_transitions"] == N_FRAMES
    assert block["num_trajectories"] == 1
    assert np.array(block["action"]["mean"]).shape == (16, 62)


# ── G4/G4b: chunk-base delta9 reconstructs the same absolute FK target the converter used ──
def test_g4_chunk_reconstructs_abs_target(converted, kin):
    """For frame i, step k=0 of the action chunk is delta9(state_pose_i, action_pose_i) for
    each arm. Reconstructing the absolute target pose from that delta9 + the frame's own
    ``observation.state`` FK pose must equal the frame's ``action_abs`` FK pose (§6)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out_root, _ = converted
    season_root = out_root / FIXTURE_SEASON.name
    ds = LeRobotDataset("origami/eef62_test", root=str(season_root))

    for i in (0, N_FRAMES // 2, N_FRAMES - 1):
        item = ds[i]
        state = item["observation.state"].numpy().astype(np.float64)
        action_abs = item["action_abs"].numpy().astype(np.float64)
        chunk0 = item["action"][0].numpy().astype(np.float64)

        # left arm: state pose = state[0:9] (trans + 2 rot cols), delta9 = chunk0[0:9]
        state_pose_l = np.eye(4)
        state_pose_l[:3, 3] = state[0:3]
        state_pose_l[:3, :3] = np.column_stack(
            [state[3:6], state[6:9], np.cross(state[3:6], state[6:9])]
        )
        recon_l = delta9_to_matrix(chunk0[0:9], state_pose_l)

        abs_pose_l = np.eye(4)
        abs_pose_l[:3, 3] = action_abs[0:3]
        abs_pose_l[:3, :3] = np.column_stack(
            [action_abs[3:6], action_abs[6:9], np.cross(action_abs[3:6], action_abs[6:9])]
        )

        np.testing.assert_allclose(recon_l[:3, 3], abs_pose_l[:3, 3], atol=1e-4)
        np.testing.assert_allclose(
            matrix_to_rot6d(recon_l[:3, :3]), matrix_to_rot6d(abs_pose_l[:3, :3]), atol=1e-4
        )


def test_deform_video_round_trips_losslessly(converted):
    """G6: the lossless deform encoder must decode back to the exact source luma."""
    from origami.decode import decode_episode_stream, split_deform_strip
    import pandas as pd

    out_root, _ = converted
    lerobot_root = FIXTURE_SEASON / "lerobot3.0"
    ep_row = pd.read_parquet(lerobot_root / "meta" / "episodes").sort_values("episode_index").iloc[0]
    chunk_i = int(ep_row["videos/observation.images.tactile_deform/chunk_index"])
    file_i = int(ep_row["videos/observation.images.tactile_deform/file_index"])
    from_ts = float(ep_row["videos/observation.images.tactile_deform/from_timestamp"])
    src_path = (
        lerobot_root / "videos" / "observation.images.tactile_deform"
        / f"chunk-{chunk_i:03d}" / f"file-{file_i:03d}.mp4"
    )
    src_frame = next(decode_episode_stream(str(src_path), from_ts, 1, fmt="gray"))
    src_tiles = split_deform_strip(src_frame)

    season_root = out_root / FIXTURE_SEASON.name
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset("origami/eef62_test", root=str(season_root))
    item = ds[0]
    for k, key in enumerate(DEFORM_KEYS):
        decoded = (item[key].numpy() * 255.0).round().astype(np.uint8)[0]  # 1 of 3 replicated channels
        np.testing.assert_array_equal(decoded, src_tiles[k])
