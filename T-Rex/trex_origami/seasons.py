"""Season inventory, the train/val split, and the 65-D joint contract.

The split is season-level (never episode-level): demonstrations recorded in the
same session share lighting, paper batch and operator drift, so mixing them
across train/val would leak. Season lists live in `splits_<revision>.json`
(loaded, not hardcoded) -- one file per HF dataset revision, since `main` and
`competition-paper-set` are different season sets with no overlap. `main`'s
split is the pre-existing one recorded in `dataset.md` (seed 0, 20% val — 101
train / 25 val out of 126 usable seasons); `competition-paper-set` has no
pre-existing split, so one was generated the same way (see that JSON file's own
`source` field).

The joint contract is copied verbatim from the competition's normative spec
(`origami-inference-kit-participant/docs/robot_io_spec.md` §2, mirrored in
`sharpa_north_ces_lite_sdk-main/examples/policy_server_template.py:48-92`).  The
dataset's `observation.state` / `action` use exactly this order, and so must
anything we emit.
"""
from __future__ import annotations

import json
import os
from typing import List, Sequence, Tuple

HF_REPO_ID = "SharpaIT/Robotic_Origami_Challenge"

# The prompt every sample is trained with.  It is *not* the dataset's own task
# string -- `meta/tasks.parquet` says "north ces task" in every season, which is
# also the organizer's `default_prompt` (openpi config.py's north_ces entries)
# and the string the T-Rex midtrain data's "I am T-Rex." plays the same role as:
# an opaque label carrying no information about the task.
#
# The prompt is constant across the dataset either way, so it can add nothing
# per-sample; what it *can* do is pick which of Qwen3-VL's pretrained
# visual-language priors the shared prefix lands on.  "north ces task" grounds on
# nothing; naming the deformable object and the target figure grounds on the
# things actually in the head camera.  Kept short deliberately -- these tokens
# are prepended to all ~410k samples, and length costs sequence budget on every
# forward pass.
#
# Because we diverge from the organizer's default_prompt, serving MUST ignore the
# `prompt` field the robot sends (it "may be empty", and the kit's own example
# says "fold the plane") and inject this string instead; `scripts/test.py` reads
# it back from the checkpoint's training_args.json to do exactly that.
INSTRUCTION = "fold the paper into a paper airplane"

#: What the raw LeRobot data and the organizer's reference config call the task.
#: Kept for provenance and for anything that needs to match the source dataset.
DATASET_TASK_STRING = "north ces task"

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


# ── seasons come from splits_<revision>.json, not hardcoded here ──────────────
# Two HF dataset revisions exist, kept as separate files (different season sets,
# no overlap): `splits_main.json` (SharpaIT/Robotic_Origami_Challenge@main, 143
# seasons on the hub, 126 in the documented split -- the 17 extras are opt-in
# and not included here) and `splits_competition_paper_set.json`
# (@competition-paper-set, 38 seasons, no pre-existing split was published for
# this revision so one was generated here with the same seeded-random 80/20
# convention as main's -- see that file's own `source` field).
_SPLITS_DIR = os.path.dirname(os.path.abspath(__file__))
_REVISION_FILES = {
    "main": "splits_main.json",
    "competition-paper-set": "splits_competition_paper_set.json",
}


def _load_split(revision: str) -> dict:
    fname = _REVISION_FILES.get(revision)
    if fname is None:
        raise ValueError(
            f"unknown revision {revision!r}; expected one of {sorted(_REVISION_FILES)}")
    with open(os.path.join(_SPLITS_DIR, fname)) as f:
        d = json.load(f)
    assert len(d["train_seasons"]) == d["num_train"], revision
    assert len(d["val_seasons"]) == d["num_val"], revision
    assert not (set(d["train_seasons"]) & set(d["val_seasons"])), \
        f"{revision}: train/val overlap"
    return d


def select_seasons(split: str, limit: int = 0, revision: str = "main") -> List[str]:
    """Seasons for `split` on `revision`, optionally truncated to the first `limit`.

    Truncation is deterministic (list order, which is chronological) rather than
    random so a pilot run and its later full rerun share a prefix -- already
    converted seasons are then skipped instead of redone.
    """
    d = _load_split(revision)
    if split == "train":
        seasons = list(d["train_seasons"])
    elif split == "val":
        seasons = list(d["val_seasons"])
    else:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    return seasons[:limit] if limit > 0 else seasons


def group_of(dim: int) -> str:
    """Name of the joint group owning 65-D index `dim`."""
    for name, lo, hi in JOINT_GROUPS:
        if lo <= dim < hi:
            return name
    raise IndexError(dim)
