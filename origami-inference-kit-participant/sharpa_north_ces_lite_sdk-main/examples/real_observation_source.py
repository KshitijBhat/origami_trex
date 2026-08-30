"""
real_observation_source.py

Drop-in replacement data source for `make_synthetic_observation()` in the
origami-zenoh-v1 black-box validator.

Reads actual frames from a LeRobot v3.0 dataset instead of generating
synthetic pixels, so real captured episodes can be replayed through the same
protocol path used against a live policy server.

This module also exposes metadata and ground-truth actions corresponding to
the most recently returned observation so the validator can compare policy
predictions against the actual dataset actions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "real_observation_source.py needs the `lerobot` package installed "
        "in the validator's environment (same one origami_dataset.py uses)."
    ) from exc


# ---------------------------------------------------------------------------
# Wire / dataset dimensions
# ---------------------------------------------------------------------------

IMAGE_SHAPE = (224, 224, 3)
TACTILE_DEFORM_SHAPE = (480, 1200, 3)
TACTILE_RAW_SHAPE = (480, 1600, 3)

STATE_DIM = 65
ACTION_DIM = 65
TACTILE_DIM = 60


# ---------------------------------------------------------------------------
# Dataset feature -> wire observation mappings
# ---------------------------------------------------------------------------

_RESIZED_CAM_KEYS = {
    "observation.images.head_left": "observation/image/head_left",
    "observation.images.head_right": "observation/image/head_right",
    "observation.images.wrist_left": "observation/image/wrist_left",
    "observation.images.wrist_right": "observation/image/wrist_right",
}

_PASSTHROUGH_IMAGE_KEYS = {
    "observation.images.tactile_deform": "observation/image/tactile_deform",
    "observation.images.tactile_raw": "observation/image/tactile_raw",
}


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _float_chw_to_uint8_hwc(frame) -> np.ndarray:
    """
    LeRobotDataset video frames decode as float32 CHW in [0, 1].

    The wire protocol wants uint8 HWC in [0, 255].
    """
    arr = frame.detach().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return arr


def _resize_uint8_hwc(
    image: np.ndarray,
    target_hw: tuple[int, int],
) -> np.ndarray:
    """
    Plain resize without letterboxing/cropping.

    Uses cv2 if available, otherwise PIL.
    """
    try:
        import cv2

        return cv2.resize(
            image,
            (target_hw[1], target_hw[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    except ImportError:
        from PIL import Image

        pil = Image.fromarray(image)
        pil = pil.resize(
            (target_hw[1], target_hw[0]),
            Image.BILINEAR,
        )
        return np.array(pil)


# ---------------------------------------------------------------------------
# Dataset source
# ---------------------------------------------------------------------------

class RealObservationSource:
    """
    Wrap a LeRobot v3.0 dataset and provide one real observation per call.

    In addition to the wire observation, this class tracks:
        - episode index
        - dataset row index
        - frame position within the episode
        - dataset timestamp in seconds
        - season name
        - GT actions corresponding to the latest observation
    """

    def __init__(
        self,
        dataset_root: str | Path,
        drop_tactile_raw_every_n: int,
        episode_index: int = 0,
        prompt_override: Optional[str] = None,
    ):
        """
        Args:
            dataset_root:
                Path to the `lerobot3.0/` directory.

            drop_tactile_raw_every_n:
                If set to N, every Nth observation omits
                observation/image/tactile_raw.

            episode_index:
                Episode to replay.

            prompt_override:
                Optional prompt replacing the dataset task text.
        """
        self.dataset_root = Path(dataset_root)

        self.ds = LeRobotDataset(
            repo_id=None,
            root=self.dataset_root,
            video_backend="pyav",
        )

        self.prompt_override = prompt_override
        self.drop_tactile_raw_every_n = drop_tactile_raw_every_n

        self._call_count = 0

        self.episode_index = episode_index

        self._frame_indices = self._episode_frame_indices(
            episode_index
        )

        if not self._frame_indices:
            raise ValueError(
                f"episode {episode_index} has no frames in this dataset"
            )

        self._cursor = 0

        # Position within self._frame_indices corresponding to the
        # most recently returned observation.
        self._last_frame_pos: Optional[int] = None

        # Metadata for the most recently returned observation.
        self.last_dataset_row_index: Optional[int] = None
        self.last_frame_index: Optional[int] = None
        self.last_timestamp_s: Optional[float] = None

        # Dataset / season metadata.
        self.season_name = self._infer_season_name()

        self._episode_prompt = (
            self.prompt_override
            or self._lookup_task_text(self._frame_indices[0])
        )

    # ------------------------------------------------------------------ #

    def _infer_season_name(self) -> str:
        """
        Infer the season name from the dataset path.

        Example:
            /foo/season_3/lerobot3.0
                -> season_3

        If no directory containing 'season' is found, fall back to the
        parent directory name.
        """
        parts = list(self.dataset_root.parts)

        for part in reversed(parts):
            if part.lower().startswith("season"):
                return part

        parent_name = self.dataset_root.parent.name
        if parent_name:
            return parent_name

        return self.dataset_root.name

    # ------------------------------------------------------------------ #

    def _episode_frame_indices(
        self,
        episode_index: int,
    ) -> list[int]:
        """All dataset row indices belonging to one episode, in order."""
        table = self.ds.hf_dataset

        return [
            i
            for i in range(len(table))
            if int(table[i]["episode_index"]) == episode_index
        ]

    # ------------------------------------------------------------------ #

    def _lookup_task_text(self, frame_idx: int) -> str:
        row = self.ds.hf_dataset[frame_idx]
        task_index = int(row["task_index"])

        try:
            return self.ds.meta.tasks[task_index]
        except Exception:
            tasks_path = self.dataset_root / "meta" / "tasks.jsonl"

            if tasks_path.exists():
                with open(tasks_path, encoding="utf-8") as f:
                    for line in f:
                        entry = json.loads(line)

                        if entry.get("task_index") == task_index:
                            return entry.get(
                                "task",
                                "origami task",
                            )

            return "origami task"

    # ------------------------------------------------------------------ #

    def reset_episode(
        self,
        episode_index: Optional[int] = None,
    ) -> None:
        """
        Reset replay position and metadata.

        Call alongside the server's reset operation.
        """
        if episode_index is not None:
            self.episode_index = episode_index

            self._frame_indices = self._episode_frame_indices(
                episode_index
            )

            if not self._frame_indices:
                raise ValueError(
                    f"episode {episode_index} has no frames"
                )

            self._episode_prompt = (
                self.prompt_override
                or self._lookup_task_text(
                    self._frame_indices[0]
                )
            )

        self._cursor = 0
        self._call_count = 0

        self._last_frame_pos = None
        self.last_dataset_row_index = None
        self.last_frame_index = None
        self.last_timestamp_s = None

    # ------------------------------------------------------------------ #

    def has_next(self) -> bool:
        return self._cursor < len(self._frame_indices)

    # ------------------------------------------------------------------ #

    def assert_requests_validity(
        self,
        requests: int,
        frame_stride: int = 1,
    ) -> None:
        """
        Validate that the requested number of observations exists.

        This check is intentionally retained for observation replay itself.

        GT action horizons are handled separately by
        get_ground_truth_actions(), which pads beyond the episode with the
        final available GT action.
        """
        if requests < 1:
            raise ValueError(
                f"requests must be >= 1, got {requests}"
            )

        if frame_stride < 1:
            raise ValueError(
                f"frame_stride must be >= 1, got {frame_stride}"
            )

        frames_needed = (
            (requests - 1) * frame_stride + 1
        )

        frames_available = (
            len(self._frame_indices) - self._cursor
        )

        if frames_needed > frames_available:
            raise ValueError(
                f"requests={requests} at frame_stride={frame_stride} "
                f"needs {frames_needed} more frame(s), but episode "
                f"{self.episode_index} only has {frames_available} "
                f"remaining from the current cursor "
                f"(episode length={len(self._frame_indices)}). "
                f"Reduce --dataset-requests, reduce --frame-stride, "
                f"or point --episode-index at a longer episode."
            )

    # ------------------------------------------------------------------ #

    def get_last_frame_info(self) -> dict[str, Any]:
        """
        Metadata for the frame returned by the most recent
        next_observation() call.

        This is NOT included in the wire observation.
        """
        return {
            "season_name": self.season_name,
            "episode_index": self.episode_index,
            "dataset_row_index": self.last_dataset_row_index,
            "frame_index": self.last_frame_index,
            "timestamp_s": self.last_timestamp_s,
        }

    # ------------------------------------------------------------------ #

    def get_ground_truth_actions(
        self,
        horizon: int,
    ) -> np.ndarray:
        """
        Return GT actions beginning at the exact dataset frame used for the
        most recent observation.

        Returns:
            float32 ndarray of shape (horizon, ACTION_DIM)

        If horizon extends beyond the end of the episode, the final
        available GT action is repeated instead of raising an error.
        """
        if horizon < 1:
            raise ValueError(
                f"horizon must be >= 1, got {horizon}"
            )

        if self._last_frame_pos is None:
            raise RuntimeError(
                "get_ground_truth_actions() must be called after "
                "next_observation()"
            )

        gt_actions: list[np.ndarray] = []

        final_pos = len(self._frame_indices) - 1

        for offset in range(horizon):
            position = self._last_frame_pos + offset

            # ----------------------------------------------------------
            # If the requested horizon extends beyond the episode,
            # repeat the last available action.
            # ----------------------------------------------------------
            if position > final_pos:
                position = final_pos

            dataset_row_index = self._frame_indices[position]

            sample = self.ds[dataset_row_index]

            action = (
                sample["action"]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            if action.ndim != 1:
                action = action.reshape(-1)

            if action.shape[0] < ACTION_DIM:
                raise ValueError(
                    f"dataset action has dimension {action.shape[0]}, "
                    f"but expected at least {ACTION_DIM}"
                )

            gt_actions.append(
                np.ascontiguousarray(
                    action[:ACTION_DIM]
                )
            )

        return np.stack(
            gt_actions,
            axis=0,
        )

    # ------------------------------------------------------------------ #

    def next_observation(
        self,
        frame_stride: int,
    ) -> dict[str, Any]:
        """
        Return one observation matching the wire contract.

        frame_stride:
            Number of dataset frames to advance after this call.
        """
        stride = frame_stride

        assert (
            stride is not None and stride >= 0
        ), "frame_stride must be >= 0"

        if not self.has_next():
            raise StopIteration(
                f"episode {self.episode_index} exhausted "
                f"({len(self._frame_indices)} frames replayed) -- "
                f"call reset_episode(next_episode_index) or reduce "
                f"--dataset-requests"
            )

        # --------------------------------------------------------------
        # IMPORTANT:
        # Save the position BEFORE advancing the cursor.
        #
        # This ensures GT actions correspond to the actual observation
        # even when frame_stride > 1.
        # --------------------------------------------------------------
        frame_pos = self._cursor
        frame_idx = self._frame_indices[frame_pos]

        self._last_frame_pos = frame_pos
        self.last_dataset_row_index = frame_idx
        self.last_frame_index = frame_pos

        # --------------------------------------------------------------
        # Read dataset timestamp.
        # --------------------------------------------------------------
        sample = self.ds[frame_idx]

        timestamp = None

        if "timestamp" in sample:
            timestamp = sample["timestamp"]
        elif "observation.timestamp" in sample:
            timestamp = sample["observation.timestamp"]

        if timestamp is not None:
            try:
                if hasattr(timestamp, "item"):
                    timestamp = timestamp.item()

                self.last_timestamp_s = float(timestamp)
            except (
                TypeError,
                ValueError,
                OverflowError,
            ):
                self.last_timestamp_s = None
        else:
            self.last_timestamp_s = None

        # Advance AFTER saving the current position.
        self._cursor += stride
        self._call_count += 1

        observation: dict[str, Any] = {}

        # --------------------------------------------------------------
        # Cameras
        # --------------------------------------------------------------

        for ds_key, wire_key in _RESIZED_CAM_KEYS.items():
            img = _float_chw_to_uint8_hwc(
                sample[ds_key]
            )

            img = _resize_uint8_hwc(
                img,
                IMAGE_SHAPE[:2],
            )

            observation[wire_key] = np.ascontiguousarray(
                img
            )

        # --------------------------------------------------------------
        # Tactile images
        # --------------------------------------------------------------

        for ds_key, wire_key in _PASSTHROUGH_IMAGE_KEYS.items():
            if ds_key not in sample:
                continue

            img = _float_chw_to_uint8_hwc(
                sample[ds_key]
            )

            expected_shape = (
                TACTILE_DEFORM_SHAPE
                if "deform" in ds_key
                else TACTILE_RAW_SHAPE
            )

            if img.shape != expected_shape:
                img = _resize_uint8_hwc(
                    img,
                    expected_shape[:2],
                )

            observation[wire_key] = np.ascontiguousarray(
                img
            )

        # --------------------------------------------------------------
        # State
        # --------------------------------------------------------------

        state = (
            sample["observation.state"]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        observation["observation/state"] = (
            np.ascontiguousarray(
                state[:STATE_DIM]
            )
        )

        # --------------------------------------------------------------
        # Joint torque
        # --------------------------------------------------------------

        if "observation.state.joint_torque" in sample:
            torque = (
                sample["observation.state.joint_torque"]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        else:
            torque = np.zeros(
                STATE_DIM,
                dtype=np.float32,
            )

        observation["observation/state/joint_torque"] = (
            np.ascontiguousarray(
                torque[:STATE_DIM]
            )
        )

        # --------------------------------------------------------------
        # Tactile vector
        # --------------------------------------------------------------

        tactile = (
            sample["observation.tactile"]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        observation["observation/tactile"] = (
            np.ascontiguousarray(
                tactile[:TACTILE_DIM]
            )
        )

        # --------------------------------------------------------------
        # Prompt
        # --------------------------------------------------------------

        observation["prompt"] = self._episode_prompt

        # --------------------------------------------------------------
        # Optional tactile_raw removal
        # --------------------------------------------------------------

        if (
            self.drop_tactile_raw_every_n
            and self._call_count
            % self.drop_tactile_raw_every_n
            == 0
        ):
            observation.pop(
                "observation/image/tactile_raw",
                None,
            )

        return observation