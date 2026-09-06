"""eval_offline.py -- REDESIGN_PLAN.md §8.1 / §12 step 10b.

Same fixture split as test_diagnose_shift.py: the checkpoint-free pieces (hold-position
baseline, the §8.1-A metric arithmetic) always run; the real cascaded-inference path needs the
~8.5 GB midtrain checkpoint (gitignored, not committed) and is skipped when absent -- it was
verified end-to-end against the real checkpoint + real fixture data in this session (see
PROGRESS.md, including the 5 trex_patch.py compatibility patches this actually required).
"""
from __future__ import annotations

import numpy as np
import pytest

from origami.constants import REPO_ROOT
from origami.diagnose_shift import ACTION_BLOCKS
from origami.eval_offline import (
    action_space_accuracy,
    hold_position_prediction,
    render_markdown,
    run,
)

FIXTURE_ROOT = REPO_ROOT / "data_trex_origami" / "eef62_train_smalltest"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "midtrain"


def test_hold_position_prediction_zero_delta_and_holds_hand():
    rng = np.random.RandomState(0)
    state62 = rng.uniform(-1, 1, size=62).astype(np.float32)
    pred = hold_position_prediction(state62, action_chunk=16)
    assert pred.shape == (16, 62)
    # Every chunk step is identical (constant prediction over the horizon).
    assert np.all(pred == pred[0])
    for name, sl in ACTION_BLOCKS:
        if name.endswith("trans3"):
            assert np.all(pred[0, sl] == 0.0)
        elif name.endswith("hand22"):
            assert np.allclose(pred[0, sl], state62[sl])
        else:  # rot6d6 -- identity rotation, first two columns of I: [1,0,0, 0,1,0]
            assert np.allclose(pred[0, sl], [1, 0, 0, 0, 1, 0])


def test_action_space_accuracy_zero_error_gives_zero_mse_mae():
    gt = np.random.RandomState(1).randn(5, 16, 62).astype(np.float32)
    out = action_space_accuracy(gt.copy(), gt)
    for k_label, blocks in out.items():
        for name, m in blocks.items():
            assert m["mse"] == pytest.approx(0.0, abs=1e-6)
            assert m["mae"] == pytest.approx(0.0, abs=1e-6)
            assert m["variance_share"] == pytest.approx(1.0, abs=1e-4)


def test_action_space_accuracy_detects_known_offset():
    gt = np.zeros((4, 16, 62), dtype=np.float32)
    pred = np.full((4, 16, 62), 2.0, dtype=np.float32)
    out = action_space_accuracy(pred, gt)
    m = out["k=0"]["L_trans3"]
    assert m["mse"] == pytest.approx(4.0)
    assert m["mae"] == pytest.approx(2.0)


def test_render_markdown_reports_hold_position_verdict():
    report = {
        "n_samples": 2, "checkpoint": "x", "root": "y", "action_chunk": 16,
        "configs": {
            "hold_position": {"action_space_accuracy": action_space_accuracy(
                np.zeros((2, 16, 62), dtype=np.float32), np.ones((2, 16, 62), dtype=np.float32))},
            "cascaded": {"action_space_accuracy": action_space_accuracy(
                np.full((2, 16, 62), 5.0, dtype=np.float32), np.ones((2, 16, 62), dtype=np.float32))},
        },
    }
    md = render_markdown(report)
    assert "Zero-shot floor check" in md
    assert "DOES NOT BEAT" in md  # cascaded (err=4) is worse than hold_position (err=1) here


@pytest.mark.skipif(not (FIXTURE_ROOT.is_dir() and CHECKPOINT_DIR.is_dir()),
                    reason="smalltest root and/or midtrain checkpoint not present locally")
def test_run_end_to_end_smoke_against_real_checkpoint():
    """Real cascaded slow/fast inference through the real midtrain checkpoint on real fixture
    frames -- the exact scenario that surfaced trex_patch.py's patches 3/4/5."""
    report = run(str(FIXTURE_ROOT), str(CHECKPOINT_DIR), n_samples=2,
                 configs=["cascaded", "hold_position"], seed=0)
    assert report["n_samples"] == 2
    assert set(report["configs"].keys()) == {"cascaded", "hold_position"}
    acc = report["configs"]["cascaded"]["action_space_accuracy"]
    assert set(acc.keys()) == {f"k={k}" for k in range(16)}
    for name, _ in ACTION_BLOCKS:
        assert acc["k=0"][name]["mse"] >= 0
