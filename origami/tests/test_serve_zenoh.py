"""Tests for origami/serve_zenoh.py. REDESIGN_PLAN.md §9.4, §12 step 13. Gates G1c, G14.

Pure-logic tests run everywhere; the real ``TeamPolicy`` construction + inference test is
``skipif``-gated on a real checkpoint + prep root, matching ``test_eval_offline.py``'s
convention -- this is the one test that actually exercises the whole deploy stack (Policy +
OrigamiKinematics + Retargeter) together, over real cascaded inference.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from origami.constants import JOINT_NAMES_65, REPO_ROOT

CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "sept9_ckpt"
LOCKED_CONFIG_SOURCE = REPO_ROOT / "data" / "meta" / "origami_prep.json"


def test_joint_names_65_matches_sdk_template():
    from origami.serve_zenoh import SDK_JOINT_NAMES

    assert tuple(JOINT_NAMES_65) == tuple(SDK_JOINT_NAMES)


def test_zero_observation_matches_wire_contract_shapes():
    from origami.serve_zenoh import zero_observation

    obs = zero_observation()
    assert obs["observation/image/head_left"].shape == (224, 224, 3)
    assert obs["observation/image/tactile_deform"].shape == (480, 1200, 3)
    assert obs["observation/state"].shape == (65,)
    assert obs["observation/tactile"].shape == (60,)
    assert isinstance(obs["prompt"], str)


def test_state62_from_state65_shape_and_finiteness():
    from origami.kinematics import LockedConfig, OrigamiKinematics
    from origami.constants import URDF_PATH
    from origami.serve_zenoh import state62_from_state65

    kin = OrigamiKinematics(URDF_PATH, LockedConfig.zeros())
    state65 = np.zeros(65, dtype=np.float32)
    state62 = state62_from_state65(kin, state65)
    assert state62.shape == (62,)
    assert state62.dtype == np.float32
    assert np.isfinite(state62).all()
    # Hand blocks pass through untouched.
    assert np.allclose(state62[9:31], state65[7:29])
    assert np.allclose(state62[40:62], state65[36:58])


def test_slow_every_times_horizon_must_equal_16():
    from origami.serve_zenoh import TeamPolicy

    with pytest.raises(ValueError, match="slow_every"):
        TeamPolicy(action_horizon=5, slow_every=4, checkpoint_path="/nonexistent")


@pytest.mark.skipif(
    not (CHECKPOINT_DIR.is_dir() and LOCKED_CONFIG_SOURCE.is_file()),
    reason="needs the real checkpoint + a real prep root's meta/origami_prep.json locally",
)
def test_team_policy_real_checkpoint_infer_shapes_and_safety():
    from origami.serve_zenoh import TeamPolicy, zero_observation

    tp = TeamPolicy(
        action_horizon=4,
        checkpoint_path=str(CHECKPOINT_DIR),
        locked_config_source=str(LOCKED_CONFIG_SOURCE),
        cuda="0",
    )
    # G1c: the checkpoint's recorded digest and the kinematics module built from the
    # recovered LockedConfig must agree.
    assert tp.policy.locked_config.digest() == tp.kin.locked.digest()

    for _ in range(3):
        actions = tp.infer(zero_observation())
        assert actions.shape == (4, 65)
        assert actions.dtype == np.float32
        assert np.isfinite(actions).all()
        lower65, upper65 = tp.kin.full_joint_limits_65()
        assert np.all(actions >= lower65[None, :] - 1e-4)
        assert np.all(actions <= upper65[None, :] + 1e-4)

    tp.reset()
    assert tp.retarget.prev_cmd is None
    assert tp.calls == 0
