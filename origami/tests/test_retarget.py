"""Tests for origami/retarget.py. REDESIGN_PLAN.md §9.2/§9.3, §12 step 13.

CPU-only, no checkpoint needed -- ``Retargeter`` only depends on ``OrigamiKinematics``
(pinocchio + pink), never on the T-Rex model.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from origami.constants import REPO_ROOT, URDF_PATH
from origami.kinematics import LockedConfig, OrigamiKinematics, matrix_to_rot6d
from origami.retarget import RetargetConfig, Retargeter, aggregate_chunks

DATA_PREP_JSON = REPO_ROOT / "data" / "meta" / "origami_prep.json"

_IDENTITY_ROT6D = matrix_to_rot6d(np.eye(3)).astype(np.float64)


def _locked_config() -> LockedConfig:
    if DATA_PREP_JSON.is_file():
        lc = json.loads(DATA_PREP_JSON.read_text())["locked_config"]
        return LockedConfig(
            lower_body=np.asarray(lc["lower_body"], dtype=np.float64),
            neck=np.asarray(lc["neck"], dtype=np.float64),
            left_hand=np.asarray(lc["left_hand"], dtype=np.float64),
            right_hand=np.asarray(lc["right_hand"], dtype=np.float64),
        )
    return LockedConfig.zeros()


@pytest.fixture(scope="module")
def kin() -> OrigamiKinematics:
    return OrigamiKinematics(URDF_PATH, _locked_config())


def _zero_action62() -> np.ndarray:
    a = np.zeros(62, dtype=np.float64)
    a[3:9] = _IDENTITY_ROT6D
    a[34:40] = _IDENTITY_ROT6D
    return a


def test_aggregate_chunks_single_pred():
    chunk = np.arange(16 * 62, dtype=np.float64).reshape(16, 62)
    out = aggregate_chunks([(0, chunk)], current_global_step=3, k=0.0)
    assert np.allclose(out, chunk[3])


def test_aggregate_chunks_none_when_uncovered():
    chunk = np.zeros((16, 62))
    assert aggregate_chunks([(0, chunk)], current_global_step=99, k=0.0) is None


def test_aggregate_chunks_uniform_average_at_k0():
    a = np.full((16, 62), 1.0)
    b = np.full((16, 62), 3.0)
    out = aggregate_chunks([(0, a), (0, b)], current_global_step=0, k=0.0)
    assert np.allclose(out, 2.0)  # uniform average of 1 and 3


def test_aggregate_chunks_newest_dominates_at_high_k():
    a = np.full((16, 62), 1.0)
    b = np.full((16, 62), 3.0)
    out = aggregate_chunks([(0, a), (0, b)], current_global_step=0, k=50.0)
    assert np.allclose(out, 3.0, atol=1e-6)  # newest (b, appended last) dominates


def test_retargeter_step_zero_action_holds_pose(kin):
    r = Retargeter(kin)
    state65 = np.zeros(65, dtype=np.float64)
    r.set_anchor(state65)
    cmd = r.step(_zero_action62(), state65[58:65])
    assert cmd.shape == (65,)
    assert cmd.dtype == np.float32
    counters = r.counters()
    assert counters["n_nan"] == 0
    assert counters["n_ik_failed"] == 0


def test_retargeter_motor_block_held_at_current_observation(kin):
    r = Retargeter(kin, RetargetConfig(max_joint_vel=100.0))  # disable rate limiting for this check
    state65 = np.zeros(65, dtype=np.float64)
    r.set_anchor(state65)
    motor7 = np.array([0.1, -0.2, 0.3, -0.4, 0.5, 0.05, -0.05])
    cmd = r.step(_zero_action62(), motor7)
    assert np.allclose(cmd[58:65], motor7, atol=1e-6)


def test_retargeter_degenerate_rot6d_counts_as_ik_failure_and_holds_warm_start(kin):
    """A degenerate rot6d (zero vector) makes ``rot6d_to_matrix`` divide by zero -> NaN target
    pose; ``step()`` catches this as an IK failure (not a finite-check hit, since the
    resulting command falls back to the last *valid* warm-start joints, which are finite)."""
    r = Retargeter(kin)
    state65 = np.zeros(65, dtype=np.float64)
    r.set_anchor(state65)
    prev = r.step(_zero_action62(), state65[58:65])

    bad_action = _zero_action62()
    bad_action[3:9] = 0.0  # degenerate rot6d -> rot6d_to_matrix divides by zero -> NaN
    cmd = r.step(bad_action, state65[58:65])
    assert np.isfinite(cmd).all()
    assert r.counters()["n_ik_failed"] == 1
    # arms held at the same warm-start joints as the previous (successful) step
    assert np.allclose(cmd[0:7], prev[0:7], atol=1e-5)
    assert np.allclose(cmd[29:36], prev[29:36], atol=1e-5)


def test_retargeter_safety_finite_check_holds_prev_cmd(kin):
    """Directly exercises ``_safety``'s own finite check (§9.3 item 1) with a raw command
    that is itself NaN -- the path a genuinely NaN model/IK output would take."""
    r = Retargeter(kin)
    state65 = np.zeros(65, dtype=np.float64)
    r.set_anchor(state65)
    prev = r.step(_zero_action62(), state65[58:65]).astype(np.float64)

    raw_cmd = np.full(65, np.nan)
    cmd = r._safety(raw_cmd, state65[58:65])
    assert np.array_equal(cmd, prev.astype(np.float32))
    assert r.counters()["n_nan"] == 1


def test_retargeter_joint_limits_are_enforced(kin):
    r = Retargeter(kin, RetargetConfig(max_joint_vel=1000.0))  # effectively disable rate limit
    state65 = np.zeros(65, dtype=np.float64)
    r.set_anchor(state65)
    action = _zero_action62()
    action[9:31] = 1000.0   # hand22 targets far outside any plausible joint range
    action[40:62] = -1000.0
    cmd = r.step(action, state65[58:65])
    lower65, upper65 = kin.full_joint_limits_65()
    assert np.all(cmd >= lower65 - 1e-6)
    assert np.all(cmd <= upper65 + 1e-6)
    assert r.counters()["n_limit_clipped"] >= 1


def test_retargeter_rate_limit_bounds_step_size(kin):
    max_vel = 0.3
    command_hz = 30
    r = Retargeter(kin, RetargetConfig(max_joint_vel=max_vel, command_hz=command_hz))
    state65 = np.zeros(65, dtype=np.float64)
    r.set_anchor(state65)
    action = _zero_action62()
    action[9:31] = 2.0    # a big instantaneous hand jump
    cmd = r.step(action, state65[58:65])
    max_step = max_vel / command_hz
    assert np.all(np.abs(cmd[9:31] - state65[9:31]) <= max_step + 1e-6)
    assert r.counters()["n_rate_clipped"] >= 1


def test_retarget_config_defaults():
    cfg = RetargetConfig()
    assert cfg.max_joint_vel == 0.3
    assert cfg.command_hz == 30
    assert cfg.check_collisions is False
