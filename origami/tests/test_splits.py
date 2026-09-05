"""Gate G18 (list-parsing portion) — REDESIGN_PLAN.md §1.1a / §12 step 3.

Network-free: only checks the parsed lists' internal integrity and the fixture's absence.
Hub resolution of every listed season is checked by ``prepare.py`` phase 0 (needs network),
not here.
"""
import json

from origami.splits import (
    DATASET_MD_PATH,
    SPLITS_JSON_PATH,
    load_splits,
    parse_splits,
    validate_splits,
)

FIXTURE_SEASON = "season_POC22061_2026_05_23_19_21_25_train"


def test_splits_json_matches_fresh_parse_of_dataset_md():
    """splits.json must never drift from dataset.md -- regenerate, don't hand-edit."""
    fresh = parse_splits(DATASET_MD_PATH)
    committed = load_splits(SPLITS_JSON_PATH)
    assert fresh == committed


def test_splits_counts_and_disjointness():
    splits = load_splits(SPLITS_JSON_PATH)
    validate_splits(splits)  # asserts len==101/25, no duplicates, no overlap


def test_fixture_season_absent_from_both_splits():
    splits = load_splits(SPLITS_JSON_PATH)
    assert FIXTURE_SEASON not in splits["train"]
    assert FIXTURE_SEASON not in splits["val"]


def test_all_season_names_well_formed():
    splits = load_splits(SPLITS_JSON_PATH)
    for season in splits["train"] + splits["val"]:
        assert season.startswith("season_POC2"), season
        assert season.endswith("_train"), season
