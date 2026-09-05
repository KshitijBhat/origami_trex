"""Parse the train/val season lists out of ``dataset.md``'s ``## My Split`` block.

REDESIGN_PLAN.md §1.1a / gate G18: never hard-code the season lists or their counts —
always regenerate ``origami/splits.json`` from ``dataset.md`` with this module, so a
future dataset-card update is caught rather than silently stale.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from origami.constants import EXPECTED_NUM_TRAIN_SEASONS, EXPECTED_NUM_VAL_SEASONS

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_MD_PATH = REPO_ROOT / "dataset.md"
SPLITS_JSON_PATH = REPO_ROOT / "origami" / "splits.json"


def parse_splits(dataset_md_path: Path = DATASET_MD_PATH) -> dict:
    """Extract ``{"train": [...], "val": [...]}`` from the ``## My Split`` fenced JSON block."""
    text = dataset_md_path.read_text()
    heading_idx = text.find("## My Split")
    if heading_idx == -1:
        raise ValueError(f"'## My Split' heading not found in {dataset_md_path}")
    tail = text[heading_idx:]
    match = re.search(r"```json\s*(\{.*?\})\s*```", tail, re.DOTALL)
    if match is None:
        raise ValueError(f"No fenced json block found after '## My Split' in {dataset_md_path}")
    block = json.loads(match.group(1))
    train = block["train_seasons"]
    val = block["val_seasons"]
    return {"train": train, "val": val}


def validate_splits(splits: dict) -> None:
    train, val = splits["train"], splits["val"]
    assert len(train) == EXPECTED_NUM_TRAIN_SEASONS, (
        f"expected {EXPECTED_NUM_TRAIN_SEASONS} train seasons, got {len(train)}"
    )
    assert len(val) == EXPECTED_NUM_VAL_SEASONS, (
        f"expected {EXPECTED_NUM_VAL_SEASONS} val seasons, got {len(val)}"
    )
    assert len(set(train)) == len(train), "duplicate season(s) in train list"
    assert len(set(val)) == len(val), "duplicate season(s) in val list"
    overlap = set(train) & set(val)
    assert not overlap, f"train/val overlap: {overlap}"


def write_splits_json(out_path: Path = SPLITS_JSON_PATH) -> dict:
    splits = parse_splits()
    validate_splits(splits)
    out_path.write_text(json.dumps(splits, indent=2) + "\n")
    return splits


def load_splits(path: Path = SPLITS_JSON_PATH) -> dict:
    return json.loads(path.read_text())


if __name__ == "__main__":
    splits = write_splits_json()
    print(f"wrote {SPLITS_JSON_PATH}: {len(splits['train'])} train, {len(splits['val'])} val")
