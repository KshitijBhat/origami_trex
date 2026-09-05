"""Season -> T-Rex eef-62 LeRobot shard. REDESIGN_PLAN.md §5.2.

Two LeRobot writers per season (§5.2 codec note, discovered during implementation): LeRobot's
``LeRobotDataset.create`` takes exactly ONE ``rgb_encoder`` for the whole dataset -- there is no
per-video-key override in the public API. Since deform maps need lossless encoding (physically
meaningful uint8 depth) while the 3 RGB cameras should keep LeRobot's default lossy codec (§5.6
storage budget), we write them as two separate shards with one-episode-per-file settings (so
their chunk/file numbering lines up 1:1) and splice the deform shard's video files + episode
metadata into the camera shard after both finalize, then discard the deform shard directory.
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from origami.decode import (
    decode_episode_stream,
    probe_available_frames,
    reconcile_episode_length,
    split_deform_strip,
    squash_to_wire,
)
from origami.kinematics import OrigamiKinematics
from origami.stats import StreamingNormStats
from utils.lerobot_common import (
    DEFORM_KEYS,
    KEY_ACTION,
    KEY_ACTION_ABS,
    KEY_HEAD,
    KEY_STATE,
    KEY_TACF6,
    KEY_WRIST_L,
    KEY_WRIST_R,
    N_FINGERS_PER_HAND,
    build_action_chunk,
    build_trex_features,
    pose_matrix_to_9d,
)

logger = logging.getLogger(__name__)

VIDEO_SRC_KEYS = {
    "head": "observation.images.head_left",
    "wrist_left": "observation.images.wrist_left",
    "wrist_right": "observation.images.wrist_right",
    "deform": "observation.images.tactile_deform",
}

MIN_EPISODE_LENGTH = 64  # §4.4: reject if N < one chunk + one F6 window

LOSSLESS_DEFORM_ENCODER = RGBEncoderConfig(vcodec="h264", crf=0, pix_fmt="yuv444p")

# Small enough that each episode lands in its own data/video file (§5.2).
ONE_EPISODE_PER_FILE_MB = dict(data_files_size_in_mb=64, video_files_size_in_mb=128)


def gray_to_3ch(a: np.ndarray) -> np.ndarray:
    """[H,W] uint8 -> [H,W,3] uint8 (replicated), matching convert_inlab_to_lerobot._gray_to_3ch."""
    a = np.clip(a, 0, 255).astype(np.uint8)
    return np.repeat(a[:, :, None], 3, axis=2)


@dataclass
class ConvertConfig:
    instruction: str
    image_size: tuple[int, int] = (224, 224)      # (H, W), D7
    deform_size: tuple[int, int] = (240, 240)


@dataclass
class SeasonResult:
    season: str
    n_episodes: int = 0
    n_frames: int = 0
    truncated: list[tuple[int, int, int]] = field(default_factory=list)  # (ep_idx, length, N)
    rejected_short_episodes: list[int] = field(default_factory=list)


def _video_path(season_root: Path, video_key: str, chunk_index: int, file_index: int) -> Path:
    return season_root / "videos" / video_key / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"


def convert_season(
    season_dir: str | Path,
    out_root: str | Path,
    kin: OrigamiKinematics,
    cfg: ConvertConfig,
    max_episodes: int | None = None,
    max_frames_per_episode: int | None = None,
) -> SeasonResult:
    """``max_episodes``/``max_frames_per_episode`` (test/debug only): convert only the first N
    episodes / first M frames of each episode, to keep smoke tests fast and memory-bounded."""
    season_dir = Path(season_dir)
    season = season_dir.name
    lerobot_root = season_dir / "lerobot3.0"

    info = json.loads((lerobot_root / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v3.0"

    episodes_meta = pd.read_parquet(lerobot_root / "meta" / "episodes")
    if max_episodes is not None:
        episodes_meta = episodes_meta.sort_values("episode_index").iloc[:max_episodes]
    tasks = pd.read_parquet(lerobot_root / "meta" / "tasks.parquet")
    assert len(tasks) == 1 and tasks.index[0] == "north ces task", tasks

    data_df = pd.read_parquet(
        lerobot_root / "data",
        columns=["observation.state", "action", "observation.tactile"],
    )

    head_shape = (3, *cfg.image_size)
    features = build_trex_features(
        head_shape=head_shape, include_wrist=True, wrist_shape=head_shape,
        include_tactile=True, deform_shape=(3, *cfg.deform_size), include_action_abs=True,
    )
    cam_features = {k: v for k, v in features.items() if k not in DEFORM_KEYS}
    deform_features = {k: v for k, v in features.items() if k in DEFORM_KEYS}

    cam_root = Path(out_root) / f"_shard_{season}_cam"
    deform_root = Path(out_root) / f"_shard_{season}_deform"
    for p in (cam_root, deform_root):
        if p.exists():
            shutil.rmtree(p)

    ds = LeRobotDataset.create(
        repo_id=f"origami/eef62_{season}_cam", fps=info["fps"], features=cam_features,
        root=cam_root, robot_type="north_poc2_2", use_videos=True, **ONE_EPISODE_PER_FILE_MB,
    )
    ds_deform = LeRobotDataset.create(
        repo_id=f"origami/eef62_{season}_deform", fps=info["fps"], features=deform_features,
        root=deform_root, robot_type="north_poc2_2", use_videos=True,
        rgb_encoder=LOSSLESS_DEFORM_ENCODER, **ONE_EPISODE_PER_FILE_MB,
    )

    acc = StreamingNormStats()
    result = SeasonResult(season=season)

    for _, ep_row in episodes_meta.sort_values("episode_index").iterrows():
        ep_idx = int(ep_row["episode_index"])
        length = int(ep_row["length"])
        from_i, to_i = int(ep_row["dataset_from_index"]), int(ep_row["dataset_to_index"])

        slice_df = data_df.iloc[from_i:to_i]
        state65 = np.stack(slice_df["observation.state"].to_numpy()).astype(np.float64)
        action65 = np.stack(slice_df["action"].to_numpy()).astype(np.float64)
        tactile60 = np.stack(slice_df["observation.tactile"].to_numpy()).astype(np.float32)

        video_paths = {}
        from_ts = {}
        for view, src_key in VIDEO_SRC_KEYS.items():
            chunk_i = int(ep_row[f"videos/{src_key}/chunk_index"])
            file_i = int(ep_row[f"videos/{src_key}/file_index"])
            video_paths[view] = _video_path(lerobot_root, src_key, chunk_i, file_i)
            from_ts[view] = float(ep_row[f"videos/{src_key}/from_timestamp"])

        available = [
            probe_available_frames(str(video_paths[v]), from_ts[v], length) for v in VIDEO_SRC_KEYS
        ]
        N = reconcile_episode_length(length, available + [len(state65), len(action65)])
        if max_frames_per_episode is not None:
            N = min(N, max_frames_per_episode)
        if N < length:
            result.truncated.append((ep_idx, length, N))
        if N < MIN_EPISODE_LENGTH:
            logger.warning("season %s episode %d: N=%d < %d, rejecting", season, ep_idx, N, MIN_EPISODE_LENGTH)
            result.rejected_short_episodes.append(ep_idx)
            continue

        state65, action65, tactile60 = state65[:N], action65[:N], tactile60[:N]

        head_iter = decode_episode_stream(str(video_paths["head"]), from_ts["head"], N, fmt="rgb24")
        wl_iter = decode_episode_stream(str(video_paths["wrist_left"]), from_ts["wrist_left"], N, fmt="rgb24")
        wr_iter = decode_episode_stream(str(video_paths["wrist_right"]), from_ts["wrist_right"], N, fmt="rgb24")
        deform_iter = decode_episode_stream(str(video_paths["deform"]), from_ts["deform"], N, fmt="gray")

        S_l, S_r = kin.fk_matrices_batch(state65[:, 0:7], state65[:, 29:36])
        A_l, A_r = kin.fk_matrices_batch(action65[:, 0:7], action65[:, 29:36])
        states = np.concatenate(
            [pose_matrix_to_9d(S_l), state65[:, 7:29], pose_matrix_to_9d(S_r), state65[:, 36:58]], axis=1
        )
        abs_targets = np.concatenate(
            [pose_matrix_to_9d(A_l), action65[:, 7:29], pose_matrix_to_9d(A_r), action65[:, 36:58]], axis=1
        )

        for i in range(N):
            chunk = build_action_chunk(
                S_l, A_l, action65[:, 7:29].astype(np.float32),
                S_r, A_r, action65[:, 36:58].astype(np.float32),
                i, N,
            )
            deform_frame = next(deform_iter)
            deform10 = split_deform_strip(deform_frame)

            cam_frame = {
                "task": cfg.instruction,
                KEY_HEAD: squash_to_wire(next(head_iter)),
                KEY_WRIST_R: squash_to_wire(next(wr_iter)),
                KEY_WRIST_L: squash_to_wire(next(wl_iter)),
                KEY_STATE: states[i].astype(np.float32),
                KEY_ACTION: chunk,
                KEY_ACTION_ABS: abs_targets[i].astype(np.float32),
                KEY_TACF6: tactile60[i].reshape(10, 6).astype(np.float32),
            }
            ds.add_frame(cam_frame)

            deform_frame_dict = {"task": cfg.instruction}
            for k in range(10):
                deform_frame_dict[DEFORM_KEYS[k]] = gray_to_3ch(deform10[k])
            ds_deform.add_frame(deform_frame_dict)

            acc.add_frame(chunk, states[i], tactile60[i].reshape(10, 6))

        ds.save_episode()
        ds_deform.save_episode()
        acc.add_episode_tracking(states, abs_targets)
        result.n_episodes += 1
        result.n_frames += N

    ds.finalize()
    ds_deform.finalize()

    _splice_deform_shard(cam_root, deform_root)
    shutil.rmtree(deform_root)

    final_root = Path(out_root) / season
    if final_root.exists():
        shutil.rmtree(final_root)
    cam_root.rename(final_root)

    acc.write(str(final_root))
    acc.dump(final_root / "meta" / "trex_norm_stats.pkl")  # shard checkpointing, §5.3/§5.4

    prep_meta = {
        "season": season,
        "locked_digest": kin.locked.digest(),
        "truncated": result.truncated,
        "rejected_short_episodes": result.rejected_short_episodes,
    }
    (final_root / "meta" / "origami_prep.json").write_text(json.dumps(prep_meta, indent=2))

    return result


def _splice_deform_shard(cam_root: Path, deform_root: Path) -> None:
    """Merge the deform-only shard's video files + episode metadata into the cam shard.

    Valid because both shards were built with identical episode ordering and
    one-episode-per-file settings, so their (chunk_index, file_index) sequences line up 1:1.
    """
    from utils.lerobot_common import DEFORM_KEYS

    cam_info_path = cam_root / "meta" / "info.json"
    deform_info_path = deform_root / "meta" / "info.json"
    cam_info = json.loads(cam_info_path.read_text())
    deform_info = json.loads(deform_info_path.read_text())
    cam_info["features"].update(deform_info["features"])
    cam_info_path.write_text(json.dumps(cam_info, indent=2))

    for key in DEFORM_KEYS:
        src = deform_root / "videos" / key
        dst = cam_root / "videos" / key
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    cam_ep_dir = cam_root / "meta" / "episodes"
    deform_ep_dir = deform_root / "meta" / "episodes"
    cam_ep = pd.read_parquet(cam_ep_dir)
    deform_ep = pd.read_parquet(deform_ep_dir)

    deform_cols = ["episode_index"] + [
        c for c in deform_ep.columns if any(c.startswith(f"videos/{k}/") for k in DEFORM_KEYS)
    ]
    merged = cam_ep.merge(deform_ep[deform_cols], on="episode_index", how="left", validate="one_to_one")
    assert len(merged) == len(cam_ep)

    shutil.rmtree(cam_ep_dir)
    (cam_ep_dir / "chunk-000").mkdir(parents=True)
    merged.to_parquet(cam_ep_dir / "chunk-000" / "file-000.parquet", index=False)
