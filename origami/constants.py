"""65-D joint layout, URDF joint names, dataset keys, and the language instruction.

Single source of truth for everything that isn't pose math or the T-Rex feature schema
(those live in ``utils.lerobot_common``, imported verbatim per REDESIGN_PLAN.md §1.4).
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
URDF_PATH = REPO_ROOT / "north_poc2_2_urdf_usd" / "north_poc2_2_v3_1.urdf"
URDF_MESH_DIR = URDF_PATH.parent

# ── HAND_ORDER (22) — REDESIGN_PLAN.md §1.1, identical to T-Rex's SHARPA_HAND_JOINT_ORDER ──
HAND_ORDER: tuple[str, ...] = (
    "thumb_CMC_FE", "thumb_CMC_AA", "thumb_MCP_FE", "thumb_MCP_AA", "thumb_IP",
    "index_MCP_FE", "index_MCP_AA", "index_PIP", "index_DIP",
    "middle_MCP_FE", "middle_MCP_AA", "middle_PIP", "middle_DIP",
    "ring_MCP_FE", "ring_MCP_AA", "ring_PIP", "ring_DIP",
    "pinky_CMC", "pinky_MCP_FE", "pinky_MCP_AA", "pinky_PIP", "pinky_DIP",
)
assert len(HAND_ORDER) == 22

ARM_JOINT_SUFFIXES: tuple[str, ...] = tuple(f"joint_{i}" for i in range(1, 8))  # 1..7
LOWER_BODY_JOINT_SUFFIXES: tuple[str, ...] = tuple(f"joint_{i}" for i in range(1, 6))  # 1..5
NECK_JOINT_SUFFIXES: tuple[str, ...] = ("joint_1", "joint_2")

# ── JOINT_NAMES_65 in dataset-index order — REDESIGN_PLAN.md §1.1 table ──
JOINT_NAMES_65: tuple[str, ...] = (
    tuple(f"left_arm_{s}" for s in ARM_JOINT_SUFFIXES)                    # 0:7
    + tuple(f"left_{h}" for h in HAND_ORDER)                              # 7:29
    + tuple(f"right_arm_{s}" for s in ARM_JOINT_SUFFIXES)                 # 29:36
    + tuple(f"right_{h}" for h in HAND_ORDER)                             # 36:58
    + tuple(f"lower_body_{s}" for s in LOWER_BODY_JOINT_SUFFIXES)         # 58:63
    + tuple(f"neck_{s}" for s in NECK_JOINT_SUFFIXES)                     # 63:65
)
assert len(JOINT_NAMES_65) == 65

# Slices into the 65-D state/action vectors.
SLICE_LEFT_ARM = slice(0, 7)
SLICE_LEFT_HAND = slice(7, 29)
SLICE_RIGHT_ARM = slice(29, 36)
SLICE_RIGHT_HAND = slice(36, 58)
SLICE_LOWER_BODY = slice(58, 63)
SLICE_NECK = slice(63, 65)

EEF_FRAMES: dict[str, str] = {"left": "left_hand_base_link", "right": "right_hand_base_link"}

# ── Arm joint limits (rad) and velocity limit (rad/s) — REDESIGN_PLAN.md §1.3 ──
ARM_LOWER = {
    "left": (-1.5359, -3.6652, -3.1067, -1.0472, -3.1067, -0.9599, -0.6981),
    "right": (-1.5359, -1.7453, -3.1067, -1.0472, -3.1067, -0.9599, -1.5708),
}
ARM_UPPER = {
    "left": (4.6775, 0.5236, 3.1067, 2.5307, 3.1067, 0.9599, 0.6981),
    "right": (4.6775, 0.5236, 3.1067, 2.5307, 3.1067, 0.9599, 1.5708),
}
ARM_VELOCITY_LIMIT = 2.6179  # rad/s, every arm joint

# ── Deform tiles — REDESIGN_PLAN.md §1.2 ──
DEFORM_STRIP_SHAPE = (480, 1200)   # (H, W) of observation.images.tactile_deform luma
DEFORM_TILE_SHAPE = (240, 240)
DEFORM_N_ROWS = 2   # row 0 = left, row 1 = right
DEFORM_N_COLS = 5   # thumb, index, middle, ring, pinky
RAW_STRIP_SHAPE = (480, 1600)
RAW_TILE_SHAPE = (240, 320)

# ── RGB wire format — REDESIGN_PLAN.md §4.3 / §1.5 ──
WIRE_IMAGE_SIZE = (224, 224)  # (H, W)
CAMERA_NATIVE_SIZE = (480, 480)  # (H, W) of observation.images.{head_left,wrist_left,wrist_right}

# ── Language instruction — REDESIGN_PLAN.md §5.7 ──
INSTRUCTION = (
    "Use both hands to fold the paper into an airplane on the table, alternating hands "
    "to crease the corners inward to the center and fold the wings down."
)
WIRE_PROMPT_REFERENCE = "fold the plane"   # organizer's reference prompt; mismatch logging at deploy
INSTRUCTION_SHORT = "fold the plane"       # short-instruction ablation

# ── Season-split gate thresholds — REDESIGN_PLAN.md §1.1a / G18 ──
EXPECTED_NUM_TRAIN_SEASONS = 101
EXPECTED_NUM_VAL_SEASONS = 25
