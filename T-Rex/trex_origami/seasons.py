"""Season inventory, the train/val split, and the 65-D joint contract.

The split is season-level (never episode-level): demonstrations recorded in the
same session share lighting, paper batch and operator drift, so mixing them
across train/val would leak.  The lists below are the split recorded in
`dataset.md` (seed 0, 20% val) — 101 train / 25 val out of the 126 usable
seasons.

The joint contract is copied verbatim from the competition's normative spec
(`origami-inference-kit-participant/docs/robot_io_spec.md` §2, mirrored in
`sharpa_north_ces_lite_sdk-main/examples/policy_server_template.py:48-92`).  The
dataset's `observation.state` / `action` use exactly this order, and so must
anything we emit.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

HF_REPO_ID = "SharpaIT/Robotic_Origami_Challenge"

# The task string in every season's `meta/tasks.parquet`, and the organizer's
# `default_prompt` in the serving entrypoint.  Train on the same text we will be
# prompted with at evaluation.
INSTRUCTION = "north ces task"

# Source frame rate of every stream (RGB, tactile video and the parquet rows).
SRC_FPS = 30

# Which of the six video streams we actually fetch.  `tactile_raw` is 73% of a
# season's bytes and T-Rex has no encoder for it; `head_right` is unused because
# the checkpoint's slow expert takes a single image.
VIDEO_KEYS = (
    "observation.images.head_left",
    "observation.images.wrist_left",
    "observation.images.wrist_right",
    "observation.images.tactile_deform",
)

# Column name in our flat parquet for each fetched stream.
VIDEO_KEY_TO_COLUMN = {
    "observation.images.head_left": "head",
    "observation.images.wrist_left": "wrist_left",
    "observation.images.wrist_right": "wrist_right",
    "observation.images.tactile_deform": "deform",
}


# ── 65-D joint contract ───────────────────────────────────────────────────────
def _hand_joint_names(side: str) -> Tuple[str, ...]:
    return (
        f"{side}_thumb_CMC_FE", f"{side}_thumb_CMC_AA", f"{side}_thumb_MCP_FE",
        f"{side}_thumb_MCP_AA", f"{side}_thumb_IP",
        f"{side}_index_MCP_FE", f"{side}_index_MCP_AA", f"{side}_index_PIP",
        f"{side}_index_DIP",
        f"{side}_middle_MCP_FE", f"{side}_middle_MCP_AA", f"{side}_middle_PIP",
        f"{side}_middle_DIP",
        f"{side}_ring_MCP_FE", f"{side}_ring_MCP_AA", f"{side}_ring_PIP",
        f"{side}_ring_DIP",
        f"{side}_pinky_CMC", f"{side}_pinky_MCP_FE", f"{side}_pinky_MCP_AA",
        f"{side}_pinky_PIP", f"{side}_pinky_DIP",
    )


JOINT_NAMES: Tuple[str, ...] = (
    tuple(f"left_arm_joint_{i}" for i in range(1, 8))
    + _hand_joint_names("left")
    + tuple(f"right_arm_joint_{i}" for i in range(1, 8))
    + _hand_joint_names("right")
    + ("lower_body_joint_1", "lower_body_joint_2", "lower_body_joint_3",
       "lower_body_joint_4", "lower_body_joint_5", "neck_joint_1", "neck_joint_2")
)
assert len(JOINT_NAMES) == 65 and len(set(JOINT_NAMES)) == 65

# (name, start, end) — end-exclusive, matching `participant_local_evaluator/contract.py`.
JOINT_GROUPS: Tuple[Tuple[str, int, int], ...] = (
    ("left_arm", 0, 7),
    ("left_hand", 7, 29),
    ("right_arm", 29, 36),
    ("right_hand", 36, 58),
    ("motor", 58, 65),
)

ACTION_DIM = 65

# Fingertip order of `observation.tactile` (10 x 6) and of the 2x5 deform grid:
# row 0 = left hand, row 1 = right hand; columns thumb..little.
FINGER_NAMES = tuple(
    f"{side}_{finger}"
    for side in ("left", "right")
    for finger in ("thumb", "index", "middle", "ring", "little")
)


# ── the split (dataset.md "My Split", seed 0, val_fraction 0.2) ───────────────
TRAIN_SEASONS: List[str] = [
    "season_POC22032_2026_05_14_19_21_01_train",
    "season_POC22032_2026_05_14_20_40_58_train",
    "season_POC22032_2026_05_14_21_08_06_train",
    "season_POC22032_2026_05_15_16_43_23_train",
    "season_POC22061_2026_05_18_19_25_41_train",
    "season_POC22061_2026_05_19_13_40_31_train",
    "season_POC22061_2026_05_19_15_37_17_train",
    "season_POC22061_2026_05_19_19_08_43_train",
    "season_POC22061_2026_05_19_21_17_23_train",
    "season_POC22061_2026_05_20_10_23_55_train",
    "season_POC22061_2026_05_20_14_02_17_train",
    "season_POC22061_2026_05_20_17_13_24_train",
    "season_POC22061_2026_05_20_19_24_50_train",
    "season_POC22061_2026_05_23_10_50_01_train",
    "season_POC22061_2026_05_23_13_39_47_train",
    "season_POC22061_2026_05_24_10_23_04_train",
    "season_POC22061_2026_05_24_13_41_09_train",
    "season_POC22061_2026_05_24_19_33_29_train",
    "season_POC22061_2026_05_25_13_50_44_train",
    "season_POC22061_2026_05_25_16_02_56_train",
    "season_POC22061_2026_05_26_10_13_09_train",
    "season_POC22061_2026_05_26_13_55_16_train",
    "season_POC22061_2026_05_26_19_13_22_train",
    "season_POC22061_2026_05_26_20_26_01_train",
    "season_POC22061_2026_05_27_13_36_59_train",
    "season_POC22061_2026_05_27_15_57_42_train",
    "season_POC22061_2026_05_27_19_13_53_train",
    "season_POC22061_2026_05_28_10_34_44_train",
    "season_POC22061_2026_05_28_19_19_16_train",
    "season_POC22061_2026_05_28_20_12_14_train",
    "season_POC22061_2026_05_29_13_40_14_train",
    "season_POC22061_2026_05_29_19_14_17_train",
    "season_POC22061_2026_05_30_10_12_51_train",
    "season_POC22061_2026_05_30_16_15_19_train",
    "season_POC22061_2026_06_24_13_45_15_train",
    "season_POC22061_2026_06_25_13_34_16_train",
    "season_POC22061_2026_06_25_15_41_29_train",
    "season_POC22061_2026_06_26_13_37_24_train",
    "season_POC22061_2026_06_26_19_05_35_train",
    "season_POC22061_2026_06_27_10_08_03_train",
    "season_POC22061_2026_06_27_15_56_17_train",
    "season_POC22061_2026_06_28_10_10_11_train",
    "season_POC22061_2026_06_28_15_55_53_train",
    "season_POC22061_2026_06_28_19_26_39_train",
    "season_POC22061_2026_06_29_10_12_41_train",
    "season_POC22061_2026_06_29_13_34_29_train",
    "season_POC22061_2026_06_29_14_27_15_train",
    "season_POC22061_2026_06_29_15_46_06_train",
    "season_POC22061_2026_06_29_16_37_08_train",
    "season_POC22061_2026_06_30_13_40_03_train",
    "season_POC22061_2026_07_01_10_15_18_train",
    "season_POC22061_2026_07_01_13_35_43_train",
    "season_POC22061_2026_07_01_15_48_25_train",
    "season_POC22061_2026_07_01_19_29_05_train",
    "season_POC22061_2026_07_02_14_23_26_train",
    "season_POC22061_2026_07_02_16_04_24_train",
    "season_POC22061_2026_07_02_19_10_24_train",
    "season_POC22061_2026_07_04_10_09_32_train",
    "season_POC22061_2026_07_04_16_13_34_train",
    "season_POC22061_2026_07_04_20_38_34_train",
    "season_POC22061_2026_07_05_10_11_10_train",
    "season_POC22061_2026_07_05_13_37_13_train",
    "season_POC22061_2026_07_06_11_05_16_train",
    "season_POC22061_2026_07_07_10_52_58_train",
    "season_POC22061_2026_07_07_13_37_02_train",
    "season_POC22061_2026_07_07_19_15_51_train",
    "season_POC22061_2026_07_08_10_10_59_train",
    "season_POC22061_2026_07_08_11_10_11_train",
    "season_POC22061_2026_07_08_13_37_45_train",
    "season_POC22061_2026_07_08_16_01_38_train",
    "season_POC22061_2026_07_08_17_17_40_train",
    "season_POC22061_2026_07_08_19_05_30_train",
    "season_POC22061_2026_07_09_10_08_39_train",
    "season_POC22061_2026_07_09_13_37_31_train",
    "season_POC22061_2026_07_09_16_23_46_train",
    "season_POC22061_2026_07_09_19_08_15_train",
    "season_POC22061_2026_07_10_10_04_17_train",
    "season_POC22061_2026_07_10_11_05_01_train",
    "season_POC22061_2026_07_10_19_07_06_train",
    "season_POC22061_2026_07_11_10_07_34_train",
    "season_POC22061_2026_07_12_10_06_51_train",
    "season_POC22061_2026_07_12_13_50_31_train",
    "season_POC22061_2026_07_12_17_04_27_train",
    "season_POC22061_2026_07_13_14_31_37_train",
    "season_POC22061_2026_07_13_14_54_37_train",
    "season_POC22061_2026_07_14_10_09_42_train",
    "season_POC22061_2026_07_14_11_01_49_train",
    "season_POC22061_2026_07_14_13_36_02_train",
    "season_POC22061_2026_07_14_15_06_58_train",
    "season_POC22061_2026_07_14_16_42_30_train",
    "season_POC22061_2026_07_14_19_16_15_train",
    "season_POC22061_2026_07_14_20_09_08_train",
    "season_POC22061_2026_07_15_10_13_37_train",
    "season_POC22061_2026_07_15_13_58_14_train",
    "season_POC22061_2026_07_15_16_41_55_train",
    "season_POC22061_2026_07_15_19_11_36_train",
    "season_POC22061_2026_07_17_10_23_21_train",
    "season_POC22061_2026_07_17_13_35_14_train",
    "season_POC22061_2026_07_17_15_59_04_train",
    "season_POC22061_2026_07_17_19_08_00_train",
    "season_POC22061_2026_07_18_10_03_25_train",
]

VAL_SEASONS: List[str] = [
    "season_POC22061_2026_05_19_10_18_58_train",
    "season_POC22061_2026_05_20_16_07_05_train",
    "season_POC22061_2026_05_23_15_56_33_train",
    "season_POC22061_2026_05_27_10_39_38_train",
    "season_POC22061_2026_05_28_13_42_37_train",
    "season_POC22061_2026_05_28_15_51_13_train",
    "season_POC22061_2026_05_29_10_19_22_train",
    "season_POC22061_2026_05_29_15_58_16_train",
    "season_POC22061_2026_06_25_20_02_02_train",
    "season_POC22061_2026_06_27_13_48_20_train",
    "season_POC22061_2026_06_27_19_13_22_train",
    "season_POC22061_2026_06_28_13_47_21_train",
    "season_POC22061_2026_06_29_19_18_53_train",
    "season_POC22061_2026_06_30_10_16_20_train",
    "season_POC22061_2026_06_30_19_27_21_train",
    "season_POC22061_2026_06_30_19_45_01_train",
    "season_POC22061_2026_07_04_13_39_40_train",
    "season_POC22061_2026_07_05_16_23_24_train",
    "season_POC22061_2026_07_10_15_21_08_train",
    "season_POC22061_2026_07_10_17_05_30_train",
    "season_POC22061_2026_07_11_13_34_49_train",
    "season_POC22061_2026_07_13_19_20_33_train",
    "season_POC22061_2026_07_13_19_56_06_train",
    "season_POC22061_2026_07_13_20_50_09_train",
    "season_POC22061_2026_07_14_15_43_22_train",
]

assert len(TRAIN_SEASONS) == 101, len(TRAIN_SEASONS)
assert len(VAL_SEASONS) == 25, len(VAL_SEASONS)
assert not (set(TRAIN_SEASONS) & set(VAL_SEASONS)), "train/val overlap"


def select_seasons(split: str, limit: int = 0) -> List[str]:
    """Seasons for `split`, optionally truncated to the first `limit`.

    Truncation is deterministic (list order, which is chronological) rather than
    random so a pilot run and its later full rerun share a prefix — already
    converted seasons are then skipped instead of redone.
    """
    if split == "train":
        seasons = list(TRAIN_SEASONS)
    elif split == "val":
        seasons = list(VAL_SEASONS)
    else:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    return seasons[:limit] if limit > 0 else seasons


def group_of(dim: int) -> str:
    """Name of the joint group owning 65-D index `dim`."""
    for name, lo, hi in JOINT_GROUPS:
        if lo <= dim < hi:
            return name
    raise IndexError(dim)
