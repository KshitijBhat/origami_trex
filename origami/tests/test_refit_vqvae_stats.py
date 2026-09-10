"""Tests for origami/refit_vqvae_stats.py. REDESIGN_PLAN.md §11.8 fix ladder item 1, G15."""
from __future__ import annotations

import json

import numpy as np
import pytest

from origami.constants import REPO_ROOT
from origami.refit_vqvae_stats import load_origami_f6_stats, refit

DATA_ROOT = REPO_ROOT / "data"


def test_refit_refuses_to_edit_in_place(tmp_path):
    with pytest.raises(ValueError, match="differ"):
        refit(str(tmp_path), str(tmp_path), str(tmp_path))


@pytest.mark.skipif(
    not (DATA_ROOT / "meta" / "trex_norm_stats.json").is_file(),
    reason="needs a real prep root's meta/trex_norm_stats.json locally",
)
def test_load_origami_f6_stats_shapes_and_mask():
    stats = load_origami_f6_stats(str(DATA_ROOT))
    assert stats["tacf6_min"].shape == (60,)
    assert stats["tacf6_max"].shape == (60,)
    assert stats["tacf6_mask"].shape == (60,)
    assert stats["tacf6_mask"].dtype == bool
    assert np.all(stats["tacf6_max"][stats["tacf6_mask"]] >= stats["tacf6_min"][stats["tacf6_mask"]])
