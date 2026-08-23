"""Public Origami tensor and joint contract used by the local evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

ACTION_DIM = 65
IMAGE_SHAPE = (224, 224, 3)
CAMERA_IMAGE_KEYS = (
    "observation/image/head_left",
    "observation/image/head_right",
    "observation/image/wrist_left",
    "observation/image/wrist_right",
)
TACTILE_DEFORM_SHAPE = (480, 1200, 3)
TACTILE_RAW_SHAPE = (480, 1600, 3)
REQUIRED_IMAGE_SPECS = {
    **{key: IMAGE_SHAPE for key in CAMERA_IMAGE_KEYS},
    "observation/image/tactile_deform": TACTILE_DEFORM_SHAPE,
}
OPTIONAL_IMAGE_SPECS = {
    "observation/image/tactile_raw": TACTILE_RAW_SHAPE,
}
IMAGE_KEYS = tuple(REQUIRED_IMAGE_SPECS)
VECTOR_SPECS = {
    "observation/state": (ACTION_DIM,),
    "observation/state/joint_torque": (ACTION_DIM,),
    "observation/tactile": (60,),
}
OBSERVATION_FIELD_METADATA = {
    **{
        key: {"dtype": "uint8", "shape": list(shape), "required": True}
        for key, shape in REQUIRED_IMAGE_SPECS.items()
    },
    **{
        key: {"dtype": "uint8", "shape": list(shape), "required": False}
        for key, shape in OPTIONAL_IMAGE_SPECS.items()
    },
    **{
        key: {"dtype": "float32", "shape": list(shape), "required": True}
        for key, shape in VECTOR_SPECS.items()
    },
    "prompt": {"dtype": "str", "shape": [], "required": True},
}
PROTOCOL_VERSION = "origami-v1"
ZENOH_PROTOCOL_VERSION = "origami-zenoh-v1"


def _hand_joint_names(side: str) -> tuple[str, ...]:
    return (
        f"{side}_thumb_CMC_FE",
        f"{side}_thumb_CMC_AA",
        f"{side}_thumb_MCP_FE",
        f"{side}_thumb_MCP_AA",
        f"{side}_thumb_IP",
        f"{side}_index_MCP_FE",
        f"{side}_index_MCP_AA",
        f"{side}_index_PIP",
        f"{side}_index_DIP",
        f"{side}_middle_MCP_FE",
        f"{side}_middle_MCP_AA",
        f"{side}_middle_PIP",
        f"{side}_middle_DIP",
        f"{side}_ring_MCP_FE",
        f"{side}_ring_MCP_AA",
        f"{side}_ring_PIP",
        f"{side}_ring_DIP",
        f"{side}_pinky_CMC",
        f"{side}_pinky_MCP_FE",
        f"{side}_pinky_MCP_AA",
        f"{side}_pinky_PIP",
        f"{side}_pinky_DIP",
    )


JOINT_NAMES = (
    tuple(f"left_arm_joint_{index}" for index in range(1, 8))
    + _hand_joint_names("left")
    + tuple(f"right_arm_joint_{index}" for index in range(1, 8))
    + _hand_joint_names("right")
    + tuple(f"lower_body_joint_{index}" for index in range(1, 6))
    + ("neck_joint_1", "neck_joint_2")
)

JOINT_GROUPS = (
    ("left_arm", 0, 7),
    ("left_hand", 7, 29),
    ("right_arm", 29, 36),
    ("right_hand", 36, 58),
    ("motor", 58, 65),
)

if len(JOINT_NAMES) != ACTION_DIM or len(set(JOINT_NAMES)) != ACTION_DIM:
    raise RuntimeError("joint contract must contain 65 unique names")


@dataclass(frozen=True)
class PolicyMetadata:
    action_horizon: int
    action_dim: int = ACTION_DIM
    action_type: str = "absolute_joint_position"
    action_units: str = "radians"
    protocol_version: str = PROTOCOL_VERSION

    @classmethod
    def parse(cls, value: Mapping[str, Any]) -> "PolicyMetadata":
        expected = {
            "protocol_version": (str, PROTOCOL_VERSION),
            "action_dim": (int, ACTION_DIM),
            "action_type": (str, "absolute_joint_position"),
            "action_units": (str, "radians"),
        }
        for key, (kind, required) in expected.items():
            actual = value.get(key)
            if type(actual) is not kind or actual != required:
                raise ValueError(f"metadata[{key!r}] must be {required!r}, got {actual!r}")
        horizon = value.get("action_horizon")
        if type(horizon) is not int or not 1 <= horizon <= 1024:
            raise ValueError("metadata['action_horizon'] must be an integer in [1,1024]")
        names = value.get("joint_names")
        if not isinstance(names, (list, tuple)) or tuple(names) != JOINT_NAMES:
            raise ValueError("metadata['joint_names'] must match the public 65-joint order")
        return cls(action_horizon=horizon)

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "action_dim": self.action_dim,
            "action_horizon": self.action_horizon,
            "action_type": self.action_type,
            "action_units": self.action_units,
            "joint_names": list(JOINT_NAMES),
        }


def validate_actions(value: Any, metadata: PolicyMetadata) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"actions must decode to ndarray, got {type(value).__name__}")
    if value.dtype != np.dtype(np.float32):
        raise ValueError(f"actions dtype must be float32, got {value.dtype}")
    expected = (metadata.action_horizon, ACTION_DIM)
    if value.shape != expected:
        raise ValueError(f"actions shape must be {expected}, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("actions contain NaN or Inf")
    return np.ascontiguousarray(value)
