"""Tests for origami/policy.py. REDESIGN_PLAN.md §9.4, §13, §12 step 13.

Pure-logic tests (``resolve_instruction``/``resolve_locked_config``/``_view_size_wh``) run
everywhere, CPU-only, no checkpoint. The real ``Policy`` construction + cascaded inference
smoke test is ``skipif``-gated on a real checkpoint being present locally, matching
``test_eval_offline.py``'s convention.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from origami.constants import INSTRUCTION, REPO_ROOT
from origami.kinematics import LockedConfig
from origami.policy import (
    REQUIRED_ACTION_CHUNK,
    REQUIRED_ACTION_DIM,
    Policy,
    _view_size_wh,
    resolve_instruction,
    resolve_locked_config,
)

CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "sept9_ckpt"
LOCKED_CONFIG_SOURCE = REPO_ROOT / "data" / "meta" / "origami_prep.json"


def test_resolve_instruction_uses_recorded_value():
    assert resolve_instruction({"instruction": "fold it"}) == "fold it"


def test_resolve_instruction_falls_back_when_null():
    assert resolve_instruction({"instruction": None}) == INSTRUCTION


def test_resolve_instruction_falls_back_when_missing():
    assert resolve_instruction({}) == INSTRUCTION


def test_resolve_locked_config_direct_from_training_args():
    lc = LockedConfig.zeros()
    ta = {
        "locked_config": {
            "lower_body": lc.lower_body.tolist(), "neck": lc.neck.tolist(),
            "left_hand": lc.left_hand.tolist(), "right_hand": lc.right_hand.tolist(),
            "digest": lc.digest(),
        }
    }
    out = resolve_locked_config(ta, locked_config_source=None)
    assert out.digest() == lc.digest()


def test_resolve_locked_config_raises_without_source_when_digest_only():
    ta = {"locked_config": {"digest": "deadbeef"}}
    with pytest.raises(ValueError, match="locked-config-source"):
        resolve_locked_config(ta, locked_config_source=None)


def test_resolve_locked_config_digest_mismatch_raises(tmp_path):
    lc = LockedConfig.zeros()
    source = tmp_path / "origami_prep.json"
    source.write_text(json.dumps({
        "locked_config": {
            "lower_body": lc.lower_body.tolist(), "neck": lc.neck.tolist(),
            "left_hand": lc.left_hand.tolist(), "right_hand": lc.right_hand.tolist(),
            "digest": lc.digest(),
        }
    }))
    ta = {"locked_config": {"digest": "not-the-real-digest"}}
    with pytest.raises(ValueError, match="digest mismatch"):
        resolve_locked_config(ta, locked_config_source=source)


@pytest.mark.skipif(not LOCKED_CONFIG_SOURCE.is_file(), reason="needs a real prep root locally")
def test_resolve_locked_config_recovers_from_real_prep_root():
    ta = json.loads((CHECKPOINT_DIR / "training_args.json").read_text()) \
        if (CHECKPOINT_DIR / "training_args.json").is_file() \
        else {"locked_config": json.loads(LOCKED_CONFIG_SOURCE.read_text())["locked_config"]}
    out = resolve_locked_config(ta, locked_config_source=LOCKED_CONFIG_SOURCE)
    expected_digest = json.loads(LOCKED_CONFIG_SOURCE.read_text())["locked_config"]["digest"]
    assert out.digest() == expected_digest


def test_view_size_wh_shared_scalar():
    assert _view_size_wh([384, 288], "head") == (384, 288)


def test_view_size_wh_per_view_dict():
    image_size = {"shared": [224, 224], "head": [384, 384], "wrist_left": [224, 224]}
    assert _view_size_wh(image_size, "head") == (384, 384)
    assert _view_size_wh(image_size, "wrist_right") == (224, 224)  # falls back to "shared"


@pytest.mark.skipif(not CHECKPOINT_DIR.is_dir(), reason="needs the real checkpoint locally")
def test_policy_slow_and_fast_real_checkpoint():
    policy = Policy(
        str(CHECKPOINT_DIR), cuda="0", locked_config_source=str(LOCKED_CONFIG_SOURCE),
    )
    assert policy.args.action_dim == REQUIRED_ACTION_DIM
    assert policy.args.action_chunk == REQUIRED_ACTION_CHUNK

    from PIL import Image

    images = {
        "head": Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)),
        "wrist_left": Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)),
        "wrist_right": Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)),
    }
    f6_window = np.zeros((policy.vqvae_window, 10, 6), dtype=np.float32)
    deform = np.zeros((10, 240, 240), dtype=np.float32)
    chunk = policy.slow_and_fast(images, f6_window, deform, state62=None)
    assert chunk.shape == (REQUIRED_ACTION_CHUNK, REQUIRED_ACTION_DIM)
    assert chunk.dtype == np.float32
    assert np.isfinite(chunk).all()

    chunk2 = policy.fast(f6_window, deform)
    assert chunk2.shape == (REQUIRED_ACTION_CHUNK, REQUIRED_ACTION_DIM)
    assert np.isfinite(chunk2).all()

    policy.reset()
    with pytest.raises(RuntimeError):
        policy.fast(f6_window, deform)  # no slow tick since reset()
