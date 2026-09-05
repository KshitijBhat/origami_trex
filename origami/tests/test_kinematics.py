"""Gates G1a, G1b, G2, G2b, G3, G3b — REDESIGN_PLAN.md §3 / §12 step 3.

CPU-only, network-free. Uses the local fixture season's real ``observation.state`` for the
FK/IK round-trip (G2/G2b) so the sampled joint angles are dataset-realistic, not synthetic.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pinocchio as pin
import pytest

from origami.constants import JOINT_NAMES_65, REPO_ROOT
from origami.kinematics import (
    ARM_JOINT_NAMES_14,
    LockedConfig,
    OrigamiKinematics,
    assert_urdf_joint_set,
    delta9_to_matrix,
    matrix_to_rot6d,
    rot6d_to_matrix,
)

FIXTURE_PARQUET = (
    REPO_ROOT
    / "season_POC22061_2026_05_23_19_21_25_train"
    / "lerobot3.0"
    / "data"
    / "chunk-000"
    / "file-000.parquet"
)

RNG = np.random.default_rng(0)


@pytest.fixture(scope="module")
def kin() -> OrigamiKinematics:
    return OrigamiKinematics()


@pytest.fixture(scope="module")
def fixture_states() -> np.ndarray:
    pytest.importorskip("pyarrow")
    import pandas as pd

    df = pd.read_parquet(FIXTURE_PARQUET, columns=["observation.state"])
    states = np.stack(df["observation.state"].to_numpy())
    assert states.shape[1] == 65
    return states


# ── G1a ──────────────────────────────────────────────────────────────────────
def test_g1a_urdf_joint_set_matches_65_mapped_names():
    from origami.constants import URDF_PATH

    full_model = pin.buildModelFromUrdf(str(URDF_PATH))
    assert_urdf_joint_set(full_model)
    assert set(full_model.names[1:]) == set(JOINT_NAMES_65)
    assert len(JOINT_NAMES_65) == 65


# ── G1b ──────────────────────────────────────────────────────────────────────
def test_g1b_reduced_model_shape(kin):
    assert kin.model.nq == kin.model.nv == 14
    assert set(kin.model.names[1:]) == set(ARM_JOINT_NAMES_14)


# ── G2 / G2b ──────────────────────────────────────────────────────────────────
# NOTE on thresholds: REDESIGN_PLAN.md states max|dq|<1e-3 rad for G2. We measured (see
# PROGRESS.md step 3 notes) that this is mathematically unachievable for ANY correct IK
# solver on this arm: it is 7-DOF solving a 6-DOF pose (a genuine 1-DOF self-motion null
# space), and warm-start noise is injected on all 7 raw joints, so the noise's null-space
# component is uncorrectable by definition -- retuning task weights across a wide sweep
# (smoothness cost 1..200) left max|dq| unchanged (~0.01-0.04 rad for std=0.02 noise),
# confirming convergence to the true constrained optimum, not under-iteration. We therefore
# gate EEF-pose reconstruction (position/orientation -- the part that IS well-defined and
# that solve_ik_tight recovers to ~1e-4 deg / ~1e-5 m) at the plan's thresholds, and report
# (not gate) max|dq| against a bound that only rules out amplification of the injected noise.
POS_THRESHOLD_M = 1e-4
ROT_THRESHOLD_DEG = 0.01


def _fk_ik_round_trip(kin, q_left7, q_right7, warm_noise_std):
    poses = kin.fk(q_left7, q_right7)
    warm_left = q_left7 + RNG.normal(0, warm_noise_std, 7)
    warm_right = q_right7 + RNG.normal(0, warm_noise_std, 7)
    warm_left, warm_right = kin.clip_arm(warm_left, warm_right)

    sol_left, sol_right = kin.solve_ik_tight(poses["left"], poses["right"], warm_left, warm_right)

    max_dq = max(np.abs(sol_left - q_left7).max(), np.abs(sol_right - q_right7).max())
    sol_poses = kin.fk(sol_left, sol_right)
    pos_err_m = max(
        np.linalg.norm(sol_poses["left"].translation - poses["left"].translation),
        np.linalg.norm(sol_poses["right"].translation - poses["right"].translation),
    )
    rot_err_rad = max(
        np.linalg.norm(pin.log3(poses["left"].rotation.T @ sol_poses["left"].rotation)),
        np.linalg.norm(pin.log3(poses["right"].rotation.T @ sol_poses["right"].rotation)),
    )
    return max_dq, pos_err_m, np.degrees(rot_err_rad)


def test_g2_fk_ik_pose_round_trip_small_warm_perturbation(kin, fixture_states):
    n = 10_000
    idx = RNG.choice(len(fixture_states), size=n, replace=False)
    sample = fixture_states[idx]
    warm_noise_std = 0.02

    n_pose_pass = 0
    max_dq_seen = 0.0
    for row in sample:
        q_left7, q_right7 = row[0:7], row[29:36]
        max_dq, pos_err_m, rot_err_deg = _fk_ik_round_trip(kin, q_left7, q_right7, warm_noise_std)
        max_dq_seen = max(max_dq_seen, max_dq)
        if pos_err_m < POS_THRESHOLD_M and rot_err_deg < ROT_THRESHOLD_DEG:
            n_pose_pass += 1
    assert n_pose_pass == n, f"{n_pose_pass}/{n} passed the pose-reconstruction thresholds"
    # max|dq| must not exceed the injected warm-start noise by more than a small multiple --
    # i.e. IK must not itself amplify the (uncorrectable, null-space) noise component.
    assert max_dq_seen < 8 * warm_noise_std, max_dq_seen


def test_g2b_fk_ik_pose_round_trip_perturbed_warm_start(kin, fixture_states):
    n = 10_000
    idx = RNG.choice(len(fixture_states), size=n, replace=False)
    sample = fixture_states[idx]
    warm_noise_std = 0.1

    n_pose_pass = 0
    failures = []
    max_dq_seen = 0.0
    for row in sample:
        q_left7, q_right7 = row[0:7], row[29:36]
        max_dq, pos_err_m, rot_err_deg = _fk_ik_round_trip(kin, q_left7, q_right7, warm_noise_std)
        max_dq_seen = max(max_dq_seen, max_dq)
        if pos_err_m < POS_THRESHOLD_M and rot_err_deg < ROT_THRESHOLD_DEG:
            n_pose_pass += 1
        else:
            failures.append((max_dq, pos_err_m, rot_err_deg))
    pass_rate = n_pose_pass / n
    assert pass_rate >= 0.995, f"pose pass rate {pass_rate:.4f} < 99.5% ({len(failures)} failures)"
    assert max_dq_seen < 8 * warm_noise_std, max_dq_seen


# ── G3 ───────────────────────────────────────────────────────────────────────
def test_g3_delta9_rot6d_reconstructs_target_pose(kin, fixture_states):
    from utils.lerobot_common import compute_chunk_delta_pose

    n = 2_000
    idx = RNG.choice(len(fixture_states) - 1, size=n, replace=False)
    max_err = 0.0
    for i in idx:
        q_left7 = fixture_states[i, 0:7]
        q_left7_next = fixture_states[i + 1, 0:7]
        base = kin.fk_matrices(q_left7, fixture_states[i, 29:36])[0]
        target = kin.fk_matrices(q_left7_next, fixture_states[i + 1, 29:36])[0]

        delta9 = compute_chunk_delta_pose(base, target)
        recon = delta9_to_matrix(delta9, base)
        err = np.abs(recon - target).max()
        max_err = max(max_err, err)
    assert max_err < 1e-9, max_err


# ── G3b ──────────────────────────────────────────────────────────────────────
def _random_rotation() -> np.ndarray:
    q = RNG.normal(size=4)
    q /= np.linalg.norm(q)
    return pin.Quaternion(q[0], q[1], q[2], q[3]).matrix()


def test_g3b_rot6d_round_trip_random_so3():
    max_err = 0.0
    for _ in range(10_000):
        R = _random_rotation()
        rot6d = matrix_to_rot6d(R)
        R_recon = rot6d_to_matrix(rot6d)
        max_err = max(max_err, np.abs(R_recon - R).max())
    assert max_err < 1e-12, max_err


# ── LockedConfig ─────────────────────────────────────────────────────────────
def test_locked_config_digest_stable_and_sensitive():
    a = LockedConfig.zeros()
    b = LockedConfig.zeros()
    assert a.digest() == b.digest()

    c = LockedConfig(
        lower_body=np.ones(5) * 1e-5, neck=np.zeros(2),
        left_hand=np.zeros(22), right_hand=np.zeros(22),
    )
    assert c.digest() != a.digest()


def test_locked_config_digest_is_dtype_invariant(fixture_states):
    """A JSON round trip (prepare.py's locked_config.json, §5.5) always reconstructs
    float64 arrays via ``np.array(python_floats)``, but ``LockedConfig.from_state_median``
    on real parquet data comes out float32 -- the digest must agree across both, or the
    exact scenario this guards against (write float32-derived config, load it back as
    float64, compare digests) spuriously fails every time. This regressed once already:
    hashing raw ``.tobytes()`` after ``np.round(x, 6)`` is NOT dtype-invariant, because
    float32's coarser grid can round to a bit-different float64 value than the same decimal
    number stored natively in float64."""
    locked_f32 = LockedConfig.from_state_median(fixture_states.astype(np.float32))
    assert locked_f32.lower_body.dtype == np.float32

    roundtripped = LockedConfig(
        lower_body=np.array(locked_f32.lower_body.tolist()),
        neck=np.array(locked_f32.neck.tolist()),
        left_hand=np.array(locked_f32.left_hand.tolist()),
        right_hand=np.array(locked_f32.right_hand.tolist()),
    )
    assert roundtripped.lower_body.dtype == np.float64
    assert roundtripped.digest() == locked_f32.digest()


def test_locked_config_from_state_median(fixture_states):
    locked = LockedConfig.from_state_median(fixture_states)
    assert locked.lower_body.shape == (5,)
    assert locked.neck.shape == (2,)
