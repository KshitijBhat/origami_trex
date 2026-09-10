"""origami-zenoh-v1 ``TeamPolicy`` + server. REDESIGN_PLAN.md §9.4, §12 step 13.

Loads the SDK's ``examples/policy_server_template.py`` from the vendored competition SDK
directly off disk (it is not an installed package) and keeps its ``OrigamiZenohServer`` +
msgpack codec byte-identical -- only ``TeamPolicy`` is replaced, per §9.4.

``TeamPolicy.infer`` implements the cadence in §9.1: ``action_horizon T`` rows per call, a slow
tick (re-encode vision, refresh the cascaded KV cache, one tactile continuation) every
``slow_every`` calls, a fast-only tick (cached KV, fresh tactile) otherwise, with
``slow_every * T == 16`` so one grand 16-row chunk spans exactly one slow-tick group. Chunk
rows are converted to safety-filtered absolute joints via ``retarget.Retargeter``, temporally
aggregated across overlapping chunk predictions via ``retarget.aggregate_chunks`` (`eef62`
chunks refresh at in-chunk offsets ``{0, 4, 8, 12}``, mirroring
``hardware_code/config/default.yaml``'s ``refine_offsets``).
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TREX_DIR = _REPO_ROOT / "T-Rex"
if str(_TREX_DIR) not in sys.path:
    sys.path.insert(0, str(_TREX_DIR))

_SDK_TEMPLATE_PATH = (
    _REPO_ROOT
    / "origami-inference-kit-participant"
    / "sharpa_north_ces_lite_sdk-main"
    / "examples"
    / "policy_server_template.py"
)


def _load_sdk_template():
    spec = importlib.util.spec_from_file_location(
        "_origami_sdk_policy_server_template", _SDK_TEMPLATE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_sdk = _load_sdk_template()
OrigamiZenohServer = _sdk.OrigamiZenohServer
SDK_JOINT_NAMES = _sdk.JOINT_NAMES

import origami.trex_patch as _trex_patch  # noqa: E402

_trex_patch.apply()

from utils.lerobot_common import pose_matrix_to_9d  # noqa: E402
from origami.constants import JOINT_NAMES_65, URDF_PATH, WIRE_PROMPT_REFERENCE  # noqa: E402
from origami.decode import split_deform_strip  # noqa: E402
from origami.kinematics import OrigamiKinematics  # noqa: E402
from origami.policy import Policy  # noqa: E402
from origami.retarget import RetargetConfig, Retargeter, aggregate_chunks  # noqa: E402

assert tuple(JOINT_NAMES_65) == tuple(SDK_JOINT_NAMES), (
    "origami.constants.JOINT_NAMES_65 must match the SDK template's JOINT_NAMES exactly "
    "(same 65-name order the wire contract's absolute-joint action assumes)"
)

_ZERO_IMAGE = np.zeros((224, 224, 3), dtype=np.uint8)
_ZERO_DEFORM = np.zeros((480, 1200, 3), dtype=np.uint8)


def zero_observation(prompt: str = WIRE_PROMPT_REFERENCE) -> dict:
    """A zero-filled but contract-valid observation, used for the warm-up inference call
    (§9.4 item 6: assert chunk shape (16, 62) equivalent -- here (T, 65))."""
    return {
        "observation/image/head_left": _ZERO_IMAGE,
        "observation/image/head_right": _ZERO_IMAGE,
        "observation/image/wrist_left": _ZERO_IMAGE,
        "observation/image/wrist_right": _ZERO_IMAGE,
        "observation/image/tactile_deform": _ZERO_DEFORM,
        "observation/state": np.zeros(65, dtype=np.float32),
        "observation/state/joint_torque": np.zeros(65, dtype=np.float32),
        "observation/tactile": np.zeros(60, dtype=np.float32),
        "prompt": prompt,
    }


def state62_from_state65(kin: OrigamiKinematics, state65: np.ndarray) -> np.ndarray:
    """§6/§9.4: ``[FK9(state65[0:7]), state65[7:29], FK9(state65[29:36]), state65[36:58]]``,
    using ``lerobot_common.pose_matrix_to_9d`` (imported verbatim, never reimplemented) on the
    same ``OrigamiKinematics`` FK the rest of the project uses. Returned *raw* -- the caller
    (``Policy._run_slow``) normalizes it with the checkpoint's own state q01/q99."""
    T_l, T_r = kin.fk_matrices(state65[0:7], state65[29:36])
    pose_l = pose_matrix_to_9d(T_l[None])[0]
    pose_r = pose_matrix_to_9d(T_r[None])[0]
    return np.concatenate([pose_l, state65[7:29], pose_r, state65[36:58]]).astype(np.float32)


class TeamPolicy:
    """origami-zenoh-v1 ``TeamPolicy`` backed by the eef-62 T-Rex checkpoint.

    See REDESIGN_PLAN.md §9.4 for the exact per-call algorithm this implements.
    """

    def __init__(
        self,
        action_horizon: int,
        checkpoint_path: str | None = None,
        urdf_path: str | None = None,
        locked_config_source: str | None = None,
        cuda: str = "0",
        disable_tactile: int = 0,
        slow_every: int = 4,
        temporal_agg_k: float = 0.0,
        check_collisions: bool = False,
        max_joint_vel: float = 0.3,
        command_hz: int = 30,
        use_wire_prompt: bool = False,
    ) -> None:
        checkpoint_path = checkpoint_path or os.environ.get("ORIGAMI_CHECKPOINT_PATH")
        if not checkpoint_path:
            raise ValueError(
                "checkpoint_path is required (pass --checkpoint-path or set "
                "ORIGAMI_CHECKPOINT_PATH)"
            )
        urdf_path = Path(urdf_path) if urdf_path else URDF_PATH
        locked_config_source = locked_config_source or os.environ.get(
            "ORIGAMI_LOCKED_CONFIG_SOURCE"
        )

        if slow_every < 1 or action_horizon < 1 or slow_every * action_horizon != 16:
            raise ValueError(
                f"slow_every * action_horizon must == 16 (§9.1); got "
                f"slow_every={slow_every}, action_horizon={action_horizon}"
            )
        self.action_horizon = action_horizon
        self.slow_every = slow_every
        self.temporal_agg_k = temporal_agg_k
        self.use_wire_prompt = use_wire_prompt
        self._warned_prompt_mismatch = False

        self.policy = Policy(
            checkpoint_path,
            cuda=cuda,
            disable_tactile=disable_tactile,
            locked_config_source=locked_config_source,
        )
        self.kin = OrigamiKinematics(urdf_path, self.policy.locked_config)
        self.retarget = Retargeter(
            self.kin,
            RetargetConfig(
                max_joint_vel=max_joint_vel, command_hz=command_hz,
                check_collisions=check_collisions,
            ),
        )

        self.f6_hist: deque = deque(maxlen=self.policy.vqvae_window)
        self.chunk_buf: list = []
        self.calls = 0

        self.reset()
        chunk = self.infer(zero_observation())
        assert chunk.shape == (self.action_horizon, 65), chunk.shape
        assert chunk.dtype == np.float32
        logger.info(
            "TeamPolicy warm-up OK: action_horizon=%d slow_every=%d instruction=%r "
            "locked_digest=%s",
            self.action_horizon, self.slow_every, self.policy.instruction,
            self.policy.locked_config.digest(),
        )

    def reset(self) -> None:
        self.policy.reset()
        self.chunk_buf = []
        self.calls = 0
        self.f6_hist.clear()
        for _ in range(self.policy.vqvae_window):
            self.f6_hist.append(np.zeros((10, 6), dtype=np.float32))
        self.retarget.base_l = None
        self.retarget.base_r = None
        self.retarget.warm_l = None
        self.retarget.warm_r = None
        self.retarget.prev_cmd = None

    def infer(self, obs: dict) -> np.ndarray:
        from PIL import Image

        state65 = np.asarray(obs["observation/state"], dtype=np.float32)
        f6 = np.asarray(obs["observation/tactile"], dtype=np.float32).reshape(10, 6)
        deform_luma = np.asarray(obs["observation/image/tactile_deform"])[:, :, 0]
        deform = split_deform_strip(deform_luma).astype(np.float32) / 255.0
        self.f6_hist.append(f6)

        prompt = obs.get("prompt", "")
        if self.use_wire_prompt:
            instruction = prompt
        else:
            instruction = self.policy.instruction
            if (
                prompt and prompt != WIRE_PROMPT_REFERENCE
                and prompt != instruction and not self._warned_prompt_mismatch
            ):
                logger.warning(
                    "obs['prompt']=%r differs from the reference %r and from the trained "
                    "instruction; ignoring it (language tokenization is participant-internal, "
                    "per robot_io_spec.md). Pass --use-wire-prompt to override.",
                    prompt, WIRE_PROMPT_REFERENCE,
                )
                self._warned_prompt_mismatch = True

        slot = self.calls % self.slow_every
        if slot == 0:
            images = {
                "head": Image.fromarray(obs["observation/image/head_left"]),
                "wrist_left": Image.fromarray(obs["observation/image/wrist_left"]),
                "wrist_right": Image.fromarray(obs["observation/image/wrist_right"]),
            }
            state62 = (
                state62_from_state65(self.kin, state65) if self.policy.use_robot_state else None
            )
            f6_window = np.stack(self.f6_hist, axis=0)
            chunk = self.policy.slow_and_fast(
                images, f6_window, deform, state62=state62, instruction=instruction,
            )
            self.chunk_buf = [(0, chunk)]
            self.retarget.set_anchor(state65.astype(np.float64))
        else:
            f6_window = np.stack(self.f6_hist, axis=0)
            chunk = self.policy.fast(f6_window, deform)
            self.chunk_buf.append((0, chunk))

        motor7 = state65[58:65]
        rows = []
        for j in range(self.action_horizon):
            step = slot * self.action_horizon + j
            agg = aggregate_chunks(self.chunk_buf, step, k=self.temporal_agg_k)
            if agg is None:
                agg = chunk[min(step, chunk.shape[0] - 1)]
            rows.append(self.retarget.step(agg, motor7))
        self.calls += 1
        return np.ascontiguousarray(np.stack(rows), dtype=np.float32)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=os.environ.get("ORIGAMI_ZENOH_ENDPOINT"))
    parser.add_argument("--session-id", default=os.environ.get("ORIGAMI_SESSION_ID"))
    parser.add_argument(
        "--action-horizon", type=int,
        default=int(os.environ.get("ORIGAMI_ACTION_HORIZON", "4")),
    )
    parser.add_argument("--slow-every", type=int, default=int(os.environ.get("ORIGAMI_SLOW_EVERY", "4")))
    parser.add_argument(
        "--execution-mode", choices=("sync", "async"),
        default=os.environ.get("EXECUTION_MODE", "async"),
    )
    parser.add_argument(
        "--checkpoint-path", default=os.environ.get("ORIGAMI_CHECKPOINT_PATH"),
    )
    parser.add_argument("--urdf-path", default=os.environ.get("ORIGAMI_URDF_PATH"))
    parser.add_argument(
        "--locked-config-source", default=os.environ.get("ORIGAMI_LOCKED_CONFIG_SOURCE"),
    )
    parser.add_argument("--cuda", default=os.environ.get("ORIGAMI_CUDA", "0"))
    parser.add_argument("--disable-tactile", type=int, default=0)
    parser.add_argument("--temporal-agg-k", type=float, default=0.0)
    parser.add_argument("--check-collisions", action="store_true")
    parser.add_argument("--max-joint-vel", type=float, default=0.3)
    parser.add_argument("--command-hz", type=int, default=30)
    parser.add_argument("--use-wire-prompt", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if not args.endpoint:
        raise SystemExit("--endpoint or ORIGAMI_ZENOH_ENDPOINT is required")
    if not args.session_id:
        raise SystemExit("--session-id or ORIGAMI_SESSION_ID is required")
    policy = TeamPolicy(
        args.action_horizon,
        checkpoint_path=args.checkpoint_path,
        urdf_path=args.urdf_path,
        locked_config_source=args.locked_config_source,
        cuda=args.cuda,
        disable_tactile=args.disable_tactile,
        slow_every=args.slow_every,
        temporal_agg_k=args.temporal_agg_k,
        check_collisions=args.check_collisions,
        max_joint_vel=args.max_joint_vel,
        command_hz=args.command_hz,
        use_wire_prompt=args.use_wire_prompt,
    )
    server = OrigamiZenohServer(
        policy,
        endpoint=args.endpoint,
        session_id=args.session_id,
        action_horizon=args.action_horizon,
        execution_mode=args.execution_mode,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    raise SystemExit(main())
