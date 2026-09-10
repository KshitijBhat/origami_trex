"""eef-62 chunk -> 65-D absolute joints via differential IK. REDESIGN_PLAN.md §9.2/§9.3.

``Retargeter`` mirrors ``T-Rex/hardware_code/eval/eval_trex_async.py``'s chunk-execution loop
(anchor captured once per chunk at `:992`, per-row delta applied from `:1113`) but drives
``origami.kinematics.OrigamiKinematics`` (our own URDF/IK) instead of the Dexmate model.
``aggregate_chunks`` is copied verbatim from that file (`:75`) per REDESIGN_PLAN.md §13 --
its module has heavy hardware imports at load time, so it is copied with citation rather than
imported.
"""
from __future__ import annotations

import logging

import numpy as np
import pinocchio as pin

from origami.constants import SLICE_LOWER_BODY, SLICE_NECK
from origami.kinematics import OrigamiKinematics, rot6d_to_matrix

logger = logging.getLogger(__name__)

DEFAULT_MAX_JOINT_VEL = 0.3  # rad/s -- matches hardware_code/teleop/arm_hand_control.py
DEFAULT_COMMAND_HZ = 30


def aggregate_chunks(chunk_buffer, current_global_step, k):
    """Exp-weighted average of all chunk predictions covering ``current_global_step``.

    Copied verbatim from ``T-Rex/hardware_code/eval/eval_trex_async.py::aggregate_chunks``
    (`:75`) -- REDESIGN_PLAN.md §13.

    chunk_buffer : list of (start_step, np.ndarray[CHUNK, action_dim]) tuples,
                   ordered oldest -> newest by append time.
    Newest entries get the highest weight; weight decays as exp(-k * age).
    Returns a 1-D action vector or None if no chunk covers this step.
    """
    preds = []
    for start, chunk in chunk_buffer:
        rel = current_global_step - start
        if 0 <= rel < len(chunk):
            preds.append(chunk[rel])
    if not preds:
        return None
    if len(preds) == 1:
        return preds[0]
    preds = np.stack(preds)  # [N, action_dim]
    weights = np.exp(-k * np.arange(len(preds))[::-1])
    weights = weights / weights.sum()
    return (preds * weights[:, None]).sum(axis=0)


class RetargetConfig:
    def __init__(
        self,
        max_joint_vel: float = DEFAULT_MAX_JOINT_VEL,
        command_hz: int = DEFAULT_COMMAND_HZ,
        check_collisions: bool = False,
    ):
        self.max_joint_vel = max_joint_vel
        self.command_hz = command_hz
        self.check_collisions = check_collisions


class Retargeter:
    """eef-62 chunk row -> 65-D absolute joint command, with mandatory safety filtering.

    One instance holds all per-episode state (chunk-base anchor pose, IK warm start, previous
    command) needed to convert successive ``action62`` rows into robot commands; call
    ``set_anchor`` once per slow tick (per REDESIGN_PLAN.md §9.1) and ``step`` once per chunk
    row.
    """

    def __init__(self, kin: OrigamiKinematics, cfg: RetargetConfig | None = None):
        self.kin = kin
        self.cfg = cfg if cfg is not None else RetargetConfig()
        lower65, upper65 = kin.full_joint_limits_65()
        self._lower65 = lower65
        self._upper65 = upper65
        self._max_step = self.cfg.max_joint_vel / self.cfg.command_hz

        self.base_l: np.ndarray | None = None
        self.base_r: np.ndarray | None = None
        self.warm_l: np.ndarray | None = None
        self.warm_r: np.ndarray | None = None
        self.prev_cmd: np.ndarray | None = None

        self.n_nan = 0
        self.n_ik_failed = 0
        self.n_limit_clipped = 0
        self.n_rate_clipped = 0
        self.n_collision_blocked = 0

    def set_anchor(self, state65: np.ndarray) -> None:
        """Capture the chunk-base anchor pose from the current arm joints (§9.1: at the slow
        tick), and (re-)seed the IK warm start and the safety filter's previous command from
        the observed state. Call once per slow tick, before the chunk's ``step`` calls."""
        state65 = np.asarray(state65, dtype=np.float64)
        q_left7 = state65[0:7]
        q_right7 = state65[29:36]
        self.base_l, self.base_r = self.kin.fk_matrices(q_left7, q_right7)
        self.warm_l, self.warm_r = q_left7.copy(), q_right7.copy()
        if self.prev_cmd is None:
            self.prev_cmd = state65.copy()

    def step(self, action62: np.ndarray, motor7: np.ndarray) -> np.ndarray:
        """One eef-62 chunk row -> a safety-filtered 65-D absolute joint command.

        Mirrors ``eval_trex_async.py:1113``. ``motor7`` is ``observation/state[58:65]`` of the
        *current* call (§9.3 item 4: the motor block is held at the observed value, never
        predicted, never stale -- D5).
        """
        assert self.base_l is not None, "set_anchor() must be called before step()"
        action62 = np.asarray(action62, dtype=np.float64)

        dpos_l, drot_l, hand_l = action62[0:3], action62[3:9], action62[9:31]
        dpos_r, drot_r, hand_r = action62[31:34], action62[34:40], action62[40:62]

        ik_ok = True
        try:
            t_l = self.base_l[:3, 3] + self.base_l[:3, :3] @ dpos_l
            R_l = self.base_l[:3, :3] @ rot6d_to_matrix(drot_l)
            t_r = self.base_r[:3, 3] + self.base_r[:3, :3] @ dpos_r
            R_r = self.base_r[:3, :3] @ rot6d_to_matrix(drot_r)
            if not (np.isfinite(t_l).all() and np.isfinite(R_l).all()
                    and np.isfinite(t_r).all() and np.isfinite(R_r).all()):
                raise FloatingPointError("non-finite target pose")
            target_l = pin.SE3(R_l, t_l)
            target_r = pin.SE3(R_r, t_r)
            q_l, q_r = self.kin.solve_ik(target_l, target_r, self.warm_l, self.warm_r)
        except Exception:
            logger.exception("IK failed; holding previous arm command")
            ik_ok = False
            q_l, q_r = self.warm_l, self.warm_r

        if not ik_ok:
            self.n_ik_failed += 1
        else:
            self.warm_l, self.warm_r = q_l, q_r

        raw_cmd = np.concatenate([q_l, hand_l, q_r, hand_r, np.asarray(motor7, dtype=np.float64)])
        cmd = self._safety(raw_cmd, np.asarray(motor7, dtype=np.float64))
        return cmd

    # ── §9.3 -- mandatory, in this order ────────────────────────────────────
    def _safety(self, raw_cmd: np.ndarray, motor7_now: np.ndarray) -> np.ndarray:
        assert self.prev_cmd is not None
        cmd = raw_cmd

        # 1. Finite check.
        if not np.isfinite(cmd).all():
            self.n_nan += 1
            self.prev_cmd = self.prev_cmd  # unchanged
            return self.prev_cmd.astype(np.float32)

        # 2. Joint limits.
        clipped = np.clip(cmd, self._lower65, self._upper65)
        if not np.allclose(clipped, cmd):
            self.n_limit_clipped += 1
        cmd = clipped

        # 3. Rate limit, clipped toward prev_cmd.
        delta = cmd - self.prev_cmd
        max_step = self._max_step
        over = np.abs(delta) > max_step
        if over.any():
            self.n_rate_clipped += 1
        delta = np.clip(delta, -max_step, max_step)
        cmd = self.prev_cmd + delta

        # 4. Motor block held at the current observed value -- never predicted, never stale.
        cmd[SLICE_LOWER_BODY] = motor7_now[0:5]
        cmd[SLICE_NECK] = motor7_now[5:7]

        # 5. Optional collision gate.
        if self.cfg.check_collisions:
            q_left7, q_right7 = cmd[0:7], cmd[29:36]
            if self.kin.check_collision(q_left7, q_right7):
                self.n_collision_blocked += 1
                cmd = self.prev_cmd.copy()
                cmd[SLICE_LOWER_BODY] = motor7_now[0:5]
                cmd[SLICE_NECK] = motor7_now[5:7]

        # 6.
        self.prev_cmd = cmd.copy()
        return cmd.astype(np.float32)

    def counters(self) -> dict:
        return {
            "n_nan": self.n_nan,
            "n_ik_failed": self.n_ik_failed,
            "n_limit_clipped": self.n_limit_clipped,
            "n_rate_clipped": self.n_rate_clipped,
            "n_collision_blocked": self.n_collision_blocked,
        }
