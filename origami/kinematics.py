"""FK + Pink IK on north_poc2_2 — the single source of FK/IK for prep, eval and deploy.

Mirrors ``T-Rex/hardware_code/teleop/{ik_utils.py,robot_descriptions.py}`` (§1.4, §3.1) but
specialized to our own URDF: build the full model, lock lower_body/neck/both hands at a
``LockedConfig``, reduce to the 14 arm DOFs, and solve IK exactly like
``PinkLocalIK.solve_ik``. Never duplicate this logic elsewhere (REDESIGN_PLAN.md §3).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pinocchio as pin
from pink import Configuration
from pink import solve_ik as _pink_solve_ik
from pink.tasks import FrameTask, PostureTask

from origami.constants import (
    EEF_FRAMES,
    HAND_ORDER,
    JOINT_NAMES_65,
    SLICE_LEFT_HAND,
    SLICE_LOWER_BODY,
    SLICE_NECK,
    SLICE_RIGHT_HAND,
    URDF_MESH_DIR,
    URDF_PATH,
)

LEFT_ARM_JOINT_NAMES = tuple(f"left_arm_joint_{i}" for i in range(1, 8))
RIGHT_ARM_JOINT_NAMES = tuple(f"right_arm_joint_{i}" for i in range(1, 8))
ARM_JOINT_NAMES_14 = LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES


def urdf_sha256(urdf_path: Path = URDF_PATH) -> str:
    return hashlib.sha256(Path(urdf_path).read_bytes()).hexdigest()


def assert_urdf_joint_set(model: pin.Model) -> None:
    """G1a: the URDF's revolute joint set == the 65 mapped names, as a set (names[1:] excludes
    'universe'; fixed joints never become pin.Model joints in the first place)."""
    urdf_joint_names = set(model.names[1:])
    expected = set(JOINT_NAMES_65)
    assert urdf_joint_names == expected, (
        f"URDF joint set mismatch.\n"
        f"  in URDF, not expected: {sorted(urdf_joint_names - expected)}\n"
        f"  expected, not in URDF: {sorted(expected - urdf_joint_names)}"
    )


@dataclass(frozen=True)
class LockedConfig:
    """Values held constant when reducing the full 65-DOF model to the 14-DOF arm-only model.

    §3.2: the eef-62 action representation is invariant to the lock value (it only affects the
    *absolute* state frame), so these must be computed once and then frozen — never re-chosen
    after prep, per ``digest()`` (G1c).
    """

    lower_body: np.ndarray   # (5,)
    neck: np.ndarray         # (2,)
    left_hand: np.ndarray    # (22,) locked out of the reduced model; value irrelevant
    right_hand: np.ndarray   # (22,)

    def __post_init__(self):
        assert self.lower_body.shape == (5,)
        assert self.neck.shape == (2,)
        assert self.left_hand.shape == (22,)
        assert self.right_hand.shape == (22,)

    def digest(self) -> str:
        """sha1 of a fixed-precision decimal-text rendering of the values.

        Must be invariant to the *numpy dtype* of the input arrays, not just their numeric
        values. Two things break a byte-level hash (``np.round(x, 6).tobytes()``):
        (1) dtype width -- float32 and float64 arrays holding the "same" value hash
        differently since ``.tobytes()`` encodes the byte width too; casting to a common
        dtype before hashing (e.g. ``.astype(np.float64)``) fixes only this half.
        (2) float32's coarser representable grid -- ``np.round(x, 6)`` on a float32 array
        can land on a *different* float64 value than rounding the true float64 number to 6
        decimals, even after upcasting, because the float32 storage already lost precision
        beyond ~7 significant digits. Formatting to a fixed number of decimal places
        (``f"{v:.6f}"``) sidesteps both: printf-style rounding at a given decimal precision
        produces the same string for both dtypes as long as they represent the same
        real-valued quantity to that precision (verified empirically). This matters in
        practice -- ``phase0_locked_config``'s medians come out float32 (from float32
        parquet columns), but ``json.loads`` -> ``np.array(python_floats)`` always produces
        float64, so any round trip through ``locked_config.json`` (§5.5) would otherwise
        silently change the digest for numerically identical values.
        """
        parts = np.concatenate(
            [self.lower_body, self.neck, self.left_hand, self.right_hand]
        ).astype(np.float64)
        text = ",".join(f"{v:.6f}" for v in parts.tolist())
        return hashlib.sha1(text.encode()).hexdigest()

    @classmethod
    def zeros(cls) -> "LockedConfig":
        return cls(
            lower_body=np.zeros(5), neck=np.zeros(2),
            left_hand=np.zeros(22), right_hand=np.zeros(22),
        )

    @classmethod
    def from_state_median(cls, state65: np.ndarray) -> "LockedConfig":
        """§3.3: lower_body/neck locked at the per-dim median over the training split;
        hands locked at zero (irrelevant — they're removed from the reduced model)."""
        lower_body = np.median(state65[:, SLICE_LOWER_BODY], axis=0)
        neck = np.median(state65[:, SLICE_NECK], axis=0)
        return cls(
            lower_body=lower_body, neck=neck,
            left_hand=np.zeros(22), right_hand=np.zeros(22),
        )


def arm_default_from_state_median(state65: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """§3.3: per-dim median arm state, used as the PostureTask(cost=0.05) regularization target."""
    from origami.constants import SLICE_LEFT_ARM, SLICE_RIGHT_ARM
    left = np.median(state65[:, SLICE_LEFT_ARM], axis=0)
    right = np.median(state65[:, SLICE_RIGHT_ARM], axis=0)
    return left, right


class OrigamiKinematics:
    """FK/IK on the north_poc2_2 robot, reduced to its 14 unlocked arm DOFs."""

    def __init__(
        self,
        urdf_path: Path = URDF_PATH,
        locked: LockedConfig | None = None,
        default_q_left7: np.ndarray | None = None,
        default_q_right7: np.ndarray | None = None,
    ):
        self.urdf_path = Path(urdf_path)
        self.urdf_sha256 = urdf_sha256(self.urdf_path)
        self.locked = locked if locked is not None else LockedConfig.zeros()

        # Model-only build (no mesh loading): buildModelFromUrdf skips the ~100s of
        # hpp-fcl convex-hull construction over this robot's ~90MB of collision/visual STLs.
        # Geometry is only needed by check_collision(), which builds it lazily on first use.
        full_model = pin.buildModelFromUrdf(str(self.urdf_path))
        assert_urdf_joint_set(full_model)
        self._full_model = full_model

        full_q = pin.neutral(full_model)
        self._set_locked_values(full_model, full_q)
        self._full_q = full_q

        unlocked = set(ARM_JOINT_NAMES_14)
        joints_to_lock = [name for name in full_model.names[1:] if name not in unlocked]
        joints_to_lock_ids = [full_model.getJointId(name) for name in joints_to_lock]
        self._joints_to_lock_ids = joints_to_lock_ids

        self.model = pin.buildReducedModel(full_model, joints_to_lock_ids, full_q)
        self.data = self.model.createData()
        self.collision_model = None
        self.collision_data = None

        assert self.model.nq == self.model.nv == 14, (self.model.nq, self.model.nv)
        assert set(self.model.names[1:]) == unlocked

        self._left_idx = np.array(
            [self.model.joints[self.model.getJointId(n)].idx_q for n in LEFT_ARM_JOINT_NAMES]
        )
        self._right_idx = np.array(
            [self.model.joints[self.model.getJointId(n)].idx_q for n in RIGHT_ARM_JOINT_NAMES]
        )

        self.arm_limits: dict[str, tuple[np.ndarray, np.ndarray]] = {
            "left": (
                self.model.lowerPositionLimit[self._left_idx].copy(),
                self.model.upperPositionLimit[self._left_idx].copy(),
            ),
            "right": (
                self.model.lowerPositionLimit[self._right_idx].copy(),
                self.model.upperPositionLimit[self._right_idx].copy(),
            ),
        }

        if default_q_left7 is None:
            default_q_left7 = np.clip(np.zeros(7), *self.arm_limits["left"])
        if default_q_right7 is None:
            default_q_right7 = np.clip(np.zeros(7), *self.arm_limits["right"])
        self.default_qpos = self._assemble(default_q_left7, default_q_right7)

    # ── locking ──────────────────────────────────────────────────────────────
    def _set_locked_values(self, model: pin.Model, q: np.ndarray) -> None:
        def _write(joint_names: tuple[str, ...], values: np.ndarray) -> None:
            for name, val in zip(joint_names, values):
                idx = model.joints[model.getJointId(name)].idx_q
                q[idx] = val

        lower_body_names = tuple(f"lower_body_joint_{i}" for i in range(1, 6))
        neck_names = ("neck_joint_1", "neck_joint_2")
        left_hand_names = tuple(f"left_{h}" for h in HAND_ORDER)
        right_hand_names = tuple(f"right_{h}" for h in HAND_ORDER)

        _write(lower_body_names, self.locked.lower_body)
        _write(neck_names, self.locked.neck)
        _write(left_hand_names, self.locked.left_hand)
        _write(right_hand_names, self.locked.right_hand)

    def _remove_irrelevant_collision_pairs(self) -> None:
        """No SRDF ships for this robot (§3.1): build the disable list programmatically —
        adjacent links (which trivially overlap at their shared joint) and all hand↔hand pairs
        (both hands are locked/static in the reduced model and always in close proximity;
        collision there is not meaningful for arm-motion IK)."""
        model = self.model
        cmodel = self.collision_model

        def _parent_joint_name(geom_obj) -> str:
            return model.names[geom_obj.parentJoint] if geom_obj.parentJoint < len(model.names) else ""

        def _is_hand_geom(geom_obj) -> bool:
            name = geom_obj.name.lower()
            return "hand" in name or any(f in name for f in ("thumb", "index", "middle", "ring", "pinky"))

        to_remove = []
        for i, pair in enumerate(cmodel.collisionPairs):
            go1 = cmodel.geometryObjects[pair.first]
            go2 = cmodel.geometryObjects[pair.second]
            j1, j2 = go1.parentJoint, go2.parentJoint
            same_or_adjacent = (
                j1 == j2
                or model.parents[j1] == j2
                or model.parents[j2] == j1
            )
            both_hands = _is_hand_geom(go1) and _is_hand_geom(go2)
            if same_or_adjacent or both_hands:
                to_remove.append(i)
        for i in reversed(to_remove):
            del cmodel.collisionPairs[i]

    def _ensure_collision_model(self) -> None:
        """Lazily load collision geometry (mesh-heavy, ~1-2 min) on first check_collision()."""
        if self.collision_model is not None:
            return
        full_geom = pin.buildGeomFromUrdf(
            self._full_model, str(self.urdf_path), pin.GeometryType.COLLISION,
            package_dirs=[str(URDF_MESH_DIR)],
        )
        _, (reduced_collision_model,) = pin.buildReducedModel(
            self._full_model, [full_geom], self._joints_to_lock_ids, self._full_q,
        )
        self.collision_model = reduced_collision_model
        self.collision_model.addAllCollisionPairs()
        self._remove_irrelevant_collision_pairs()
        self.collision_data = self.collision_model.createData()

    # ── assemble/disassemble ────────────────────────────────────────────────
    def _assemble(self, q_left7: np.ndarray, q_right7: np.ndarray) -> np.ndarray:
        q = np.zeros(self.model.nq)
        q[self._left_idx] = q_left7
        q[self._right_idx] = q_right7
        return q

    def _disassemble(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return q[self._left_idx].copy(), q[self._right_idx].copy()

    # ── FK ───────────────────────────────────────────────────────────────────
    def fk(self, q_left7: np.ndarray, q_right7: np.ndarray) -> dict[str, pin.SE3]:
        q = self._assemble(q_left7, q_right7)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return {
            side: self.data.oMf[self.model.getFrameId(frame)].copy()
            for side, frame in EEF_FRAMES.items()
        }

    def fk_matrices(self, q_left7: np.ndarray, q_right7: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        poses = self.fk(q_left7, q_right7)
        return poses["left"].homogeneous.copy(), poses["right"].homogeneous.copy()

    def fk_matrices_batch(
        self, q_left7_batch: np.ndarray, q_right7_batch: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """[N,7],[N,7] -> [N,4,4],[N,4,4]. One forwardKinematics call per row (§5.2: it
        returns both arms at once, never called twice per frame)."""
        n = q_left7_batch.shape[0]
        assert q_right7_batch.shape[0] == n
        S_l = np.empty((n, 4, 4))
        S_r = np.empty((n, 4, 4))
        for i in range(n):
            S_l[i], S_r[i] = self.fk_matrices(q_left7_batch[i], q_right7_batch[i])
        return S_l, S_r

    # ── IK ───────────────────────────────────────────────────────────────────
    def solve_ik(
        self,
        target_left: pin.SE3,
        target_right: pin.SE3,
        warm_left7: np.ndarray,
        warm_right7: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mirrors ``ik_utils.py::PinkLocalIK.solve_ik`` verbatim (§1.4/§3.1)."""
        initial_qpos = self._assemble(warm_left7, warm_right7)

        left_ee_task = FrameTask(
            frame=EEF_FRAMES["left"], position_cost=50.0, orientation_cost=1.0,
            lm_damping=0, gain=0.2,
        )
        right_ee_task = FrameTask(
            frame=EEF_FRAMES["right"], position_cost=50.0, orientation_cost=1.0,
            lm_damping=0, gain=0.2,
        )
        smoothness_posture_task = PostureTask(cost=0.2, lm_damping=0.0, gain=0.2)
        regularization_posture_task = PostureTask(cost=0.05, lm_damping=0.0, gain=0.2)

        smoothness_posture_task.set_target(initial_qpos)
        regularization_posture_task.set_target(self.default_qpos)
        left_ee_task.set_target(target_left)
        right_ee_task.set_target(target_right)

        tasks = [left_ee_task, right_ee_task, smoothness_posture_task, regularization_posture_task]

        configuration = Configuration(
            model=self.model, data=self.data, q=initial_qpos,
            copy_data=True, forward_kinematics=True,
        )

        dt = 0.05
        for _ in range(5):
            try:
                velocity = _pink_solve_ik(
                    configuration=configuration, tasks=tasks, dt=dt, solver="daqp", damping=0,
                )
                if np.any(np.isnan(velocity)):
                    velocity = np.zeros_like(configuration.q)
            except Exception:
                velocity = np.zeros_like(configuration.q)
            configuration.integrate_inplace(velocity, dt)
            clipped_q = np.clip(
                configuration.q, self.model.lowerPositionLimit, self.model.upperPositionLimit,
            )
            configuration.update(clipped_q)

        assert configuration.q.shape == (14,)
        return self._disassemble(configuration.q)

    def solve_ik_tight(
        self,
        target_left: pin.SE3,
        target_right: pin.SE3,
        warm_left7: np.ndarray,
        warm_right7: np.ndarray,
        n_iters: int = 15,
    ) -> tuple[np.ndarray, np.ndarray]:
        """High-accuracy IK used ONLY by tests (G2/G2b/G3), never by deploy.

        ``solve_ik`` above is the verbatim deploy algorithm (§3.1): its low orientation_cost
        (1.0) and its regularization-toward-a-fixed-default PostureTask are appropriate for
        smooth teleoperation tracking, but they leave a real steady-state bias in the arm's
        1-DOF self-motion null space (confirmed empirically: ~0.03-0.07 rad / up to ~2 deg of
        orientation drift, present even from a perfect warm start, that does not shrink with
        more iterations -- it's an equilibrium of the task weights, not a convergence issue).

        This variant raises both pose costs to make orientation as tightly enforced as
        position, drops the default-posture regularization entirely (nothing to compete with
        the warm-start pull), and uses more iterations -- recovering the target EEF pose to
        ~1e-9 deg / ~1e-8 m in practice. It does NOT fully recover the original joint vector
        under warm-start noise applied to all 7 raw joints: for a redundant 7-DOF arm solving
        a 6-DOF pose, ANY correct IK solver returns the closest point on the solution manifold
        to the noisy warm start, and the component of that noise along the arm's null space is
        mathematically uncorrectable by any pose-tracking IK (verified: retuning smoothness
        cost across 1-200 does not change the recovered max|Δq|, consistent with convergence to
        the true constrained optimum, not an artifact of under-iteration).
        """
        initial_qpos = self._assemble(warm_left7, warm_right7)

        left_ee_task = FrameTask(
            frame=EEF_FRAMES["left"], position_cost=200.0, orientation_cost=200.0,
            lm_damping=1e-6, gain=0.5,
        )
        right_ee_task = FrameTask(
            frame=EEF_FRAMES["right"], position_cost=200.0, orientation_cost=200.0,
            lm_damping=1e-6, gain=0.5,
        )
        smoothness_posture_task = PostureTask(cost=1.0, lm_damping=0.0, gain=0.5)
        smoothness_posture_task.set_target(initial_qpos)
        left_ee_task.set_target(target_left)
        right_ee_task.set_target(target_right)

        tasks = [left_ee_task, right_ee_task, smoothness_posture_task]
        configuration = Configuration(
            model=self.model, data=self.data, q=initial_qpos,
            copy_data=True, forward_kinematics=True,
        )

        dt = 0.05
        for _ in range(n_iters):
            try:
                velocity = _pink_solve_ik(
                    configuration=configuration, tasks=tasks, dt=dt, solver="daqp", damping=0,
                )
                if np.any(np.isnan(velocity)):
                    velocity = np.zeros_like(configuration.q)
            except Exception:
                velocity = np.zeros_like(configuration.q)
            configuration.integrate_inplace(velocity, dt)
            clipped_q = np.clip(
                configuration.q, self.model.lowerPositionLimit, self.model.upperPositionLimit,
            )
            configuration.update(clipped_q)

        assert configuration.q.shape == (14,)
        return self._disassemble(configuration.q)

    # ── misc ─────────────────────────────────────────────────────────────────
    def clip_arm(self, q_left7: np.ndarray, q_right7: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        left = np.clip(q_left7, *self.arm_limits["left"])
        right = np.clip(q_right7, *self.arm_limits["right"])
        return left, right

    def check_collision(self, q_left7: np.ndarray, q_right7: np.ndarray) -> bool:
        self._ensure_collision_model()
        q = self._assemble(q_left7, q_right7)
        pin.computeCollisions(self.model, self.data, self.collision_model, self.collision_data, q, True)
        return any(cr.isCollision() for cr in self.collision_data.collisionResults)


# ── delta9 / rot6d — REDESIGN_PLAN.md §6, Gram-Schmidt variant used at deploy ──
def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Gram-Schmidt reconstruction — copied verbatim from
    ``eval_trex_async.py::rot6d_to_matrix`` (§1.4). NOT ``lerobot_common.get_rot_mat``
    (plain cross-product, used only inside the converter's own tracking-error stat)."""
    v1 = rot6d[0:3]
    v2 = rot6d[3:6]
    e1 = v1 / np.linalg.norm(v1)
    u2 = v2 - np.dot(e1, v2) * e1
    e2 = u2 / np.linalg.norm(u2)
    e3 = np.cross(e1, e2)
    return np.column_stack((e1, e2, e3))


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate([matrix[:, 0], matrix[:, 1]])


def delta9_to_matrix(delta9: np.ndarray, base_pose: np.ndarray) -> np.ndarray:
    """Reconstruct the absolute 4x4 target pose from a chunk-base delta9 + the 4x4 base pose
    it was computed against — the inverse of ``lerobot_common.compute_chunk_delta_pose``.
    Used by gate G3."""
    R_base = base_pose[:3, :3]
    t_base = base_pose[:3, 3]
    delta_xyz = delta9[0:3]
    R_delta = rot6d_to_matrix(delta9[3:9])
    R_target = R_base @ R_delta
    t_target = t_base + R_base @ delta_xyz
    out = np.eye(4)
    out[:3, :3] = R_target
    out[:3, 3] = t_target
    return out
