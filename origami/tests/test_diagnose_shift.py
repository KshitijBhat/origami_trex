"""diagnose_shift.py -- REDESIGN_PLAN.md §11.9 step 2 / §12 step 10b.

Uses `data_trex_origami/eef62_train_smalltest/` -- a real merged (post-prepare.py) root, unlike
the raw per-season fixture the steps 1-8 tests use, per this session's explicit instruction to
exercise the eval/diagnose scripts against it. The checkpoint-free sections (chunk-delta
magnitude, F6 magnitude, deform occupancy) always run. G15 and the ViT-feature section need the
~8.5 GB T-Rex midtrain checkpoint (`checkpoints/midtrain/`, gitignored, never committed) and
are skipped when it isn't present on disk -- they were verified for real in this session (see
PROGRESS.md) but aren't something CI or a fresh checkout can be expected to have.
"""
from __future__ import annotations

import pytest

from origami.constants import REPO_ROOT
from origami.diagnose_shift import (
    ACTION_BLOCKS,
    FINGER_LABELS,
    chunk_delta_magnitude,
    deform_occupancy,
    f6_magnitude_per_finger,
    render_markdown,
    run,
    vqvae_code_entropy,
)

FIXTURE_ROOT = REPO_ROOT / "data_trex_origami" / "eef62_train_smalltest"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "midtrain"

pytestmark = pytest.mark.skipif(not FIXTURE_ROOT.is_dir(), reason="smalltest merged root not present")


def test_chunk_delta_magnitude_shape_and_sanity():
    out = chunk_delta_magnitude(str(FIXTURE_ROOT))
    assert set(out.keys()) == {"k=0", "k=15"}
    for blocks in out.values():
        assert set(blocks.keys()) == {name for name, _ in ACTION_BLOCKS}
        assert all(v >= 0 for v in blocks.values())
    # translation deltas are small (meters, near-static folding motion); hand-joint deltas
    # (radians, active fingers) are the dominant motion -- sanity-checks the block slicing
    # matches build_action_chunk's real layout, not just that it runs.
    assert out["k=0"]["L_trans3"] < out["k=0"]["L_hand22"]
    assert out["k=0"]["R_trans3"] < out["k=0"]["R_hand22"]


def test_f6_magnitude_per_finger_thumb_index_dominate():
    out = f6_magnitude_per_finger(str(FIXTURE_ROOT))
    assert set(out.keys()) == set(FINGER_LABELS)
    # §1.2's known pattern: thumb/index carry real force, the other three fingers are ~dead.
    for side in ("L", "R"):
        assert out[f"{side}-thumb"]["force_rms"] > out[f"{side}-middle"]["force_rms"]
        assert out[f"{side}-index"]["force_rms"] > out[f"{side}-ring"]["force_rms"]


def test_deform_occupancy_thumb_index_more_active_than_middle_ring_pinky():
    out = deform_occupancy(str(FIXTURE_ROOT), n_samples=8)
    assert set(out.keys()) == set(FINGER_LABELS)
    for v in out.values():
        assert v is not None and v["n_frames"] > 0
        assert 0.0 <= v["mean_occupancy"] <= 1.0
    for side in ("L", "R"):
        assert out[f"{side}-thumb"]["mean_occupancy"] >= out[f"{side}-middle"]["mean_occupancy"]


def test_render_markdown_runs_without_checkpoint():
    report = {
        "chunk_delta_magnitude": chunk_delta_magnitude(str(FIXTURE_ROOT)),
        "f6_magnitude_per_finger": f6_magnitude_per_finger(str(FIXTURE_ROOT)),
        "deform_occupancy": deform_occupancy(str(FIXTURE_ROOT), n_samples=4),
        "g15": None,
        "vit_feature_shift": None,
    }
    md = render_markdown(report)
    assert "Chunk-delta magnitude" in md
    assert "G15" not in md  # section omitted entirely when g15 is None


@pytest.mark.skipif(not CHECKPOINT_DIR.is_dir(), reason="midtrain checkpoint not downloaded locally")
def test_g15_vqvae_code_entropy_matches_known_dead_fingers():
    """**G15**. Verified against the real midtrain checkpoint + real fixture data in this
    session (see PROGRESS.md): T-Rex's own tacf6_vqvae buffers collapse L-ring/L-pinky to a
    single code on origami F6 -- the exact failure §11.8 predicts."""
    out = vqvae_code_entropy(str(FIXTURE_ROOT), str(CHECKPOINT_DIR), n_windows=5000)
    assert out["n_windows"] == 5000
    assert set(out["slots"].keys()) == {f"{s}-{n}" for s in ("L", "R")
                                        for n in ("thumb", "index", "middle", "ring", "pinky")}
    # The known-dead fingers collapse toward a single dominant code.
    assert out["slots"]["L-ring"]["top5_code_frac"][0] > 0.9
    assert out["slots"]["L-pinky"]["top5_code_frac"][0] > 0.9
    # The live fingers use meaningfully more of the codebook.
    assert out["slots"]["L-thumb"]["n_codes_used"] > out["slots"]["L-ring"]["n_codes_used"]


@pytest.mark.skipif(not CHECKPOINT_DIR.is_dir(), reason="midtrain checkpoint not downloaded locally")
def test_run_end_to_end_with_checkpoint_smoke():
    report = run(str(FIXTURE_ROOT), str(CHECKPOINT_DIR), trex_dataset_root=None,
                 n_windows=2000, n_deform_samples=4, n_vit_samples=3, device="cpu")
    assert report["g15"] is not None
    assert report["vit_feature_shift"] is not None
    assert report["vit_feature_shift"]["origami"]["n_samples"] == 3
    md = render_markdown(report)
    assert "G15" in md
