"""trex_origami — Robotic Origami Challenge → T-Rex "origami-flat" data prep.

Converts Sharpa's `SharpaIT/Robotic_Origami_Challenge` LeRobot v3.0 seasons into
the compact parquet + JPEG-blob format consumed by
`qwen_vla.origami_dataset.OrigamiDataset`.

Deliberately torch-free (pyarrow + PIL + ffmpeg only) so the whole pipeline runs
on a CPU box while training happens elsewhere.

Entry points:
    python -m trex_origami.prepare --help
    python -m trex_origami.verify  --help
"""

from .seasons import (
    HF_REPO_ID,
    INSTRUCTION,
    JOINT_NAMES,
    JOINT_GROUPS,
    select_seasons,
)

__all__ = [
    "HF_REPO_ID",
    "INSTRUCTION",
    "JOINT_NAMES",
    "JOINT_GROUPS",
    "select_seasons",
]
