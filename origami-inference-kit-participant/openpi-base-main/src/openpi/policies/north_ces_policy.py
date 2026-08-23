"""North CES policy transforms for OpenPI training and inference."""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


JOINT_CONFIG = {
    "left_arm": 7,
    "left_hand": 22,
    "right_arm": 7,
    "right_hand": 22,
    "motor": 7,
}
TOTAL_DOF = sum(JOINT_CONFIG.values())


def make_north_ces_example() -> dict:
    return {
        "observation/image/head_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/image/head_right": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/image/wrist_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/image/wrist_right": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.rand(TOTAL_DOF).astype(np.float32),
        "observation/state/joint_torque": np.zeros(TOTAL_DOF, dtype=np.float32),
        "observation/tactile": np.zeros(60, dtype=np.float32),
        "observation/image/tactile_deform": np.zeros((480, 1200, 3), dtype=np.uint8),
        "observation/image/tactile_raw": np.zeros((480, 1600, 3), dtype=np.uint8),
        "prompt": "fold the plane",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class NorthCESInputs(transforms.DataTransformFn):
    """Map North CES observations to the image/state layout expected by pi0/pi05."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        head_left = _parse_image(data["observation/image/head_left"])
        wrist_left = _parse_image(data["observation/image/wrist_left"])
        wrist_right = _parse_image(data["observation/image/wrist_right"])

        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {
                "base_0_rgb": head_left,
                "left_wrist_0_rgb": wrist_left,
                "right_wrist_0_rgb": wrist_right,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class NorthCESOutputs(transforms.DataTransformFn):
    """Extract the physical North CES action dimensions from model outputs."""

    def __call__(self, data: dict) -> dict:
        return {
            "actions": np.asarray(
                data["actions"][:, :TOTAL_DOF],
                dtype=np.float32,
            )
        }
