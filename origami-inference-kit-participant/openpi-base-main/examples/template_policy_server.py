#!/usr/bin/env python3
"""Legacy local WebSocket helper for adapting an OpenPI policy.

Wrap your own trained model in ``MyPolicy.infer(obs) -> {"actions": np.ndarray}``
and serve it over WebSocket for local OpenPI debugging. Competition submission
images must use ``scripts/serve_policy_zenoh.py`` and ``origami-zenoh-v1``.

Observation you receive (dict) -- see ../docs/robot_io_spec.md. The runtime always
sends the full contract; pick whichever fields your model needs:
    obs["observation/image/head_left"]      uint8  HxWx3 RGB   head stereo, left eye
    obs["observation/image/head_right"]     uint8  HxWx3 RGB   head stereo, right eye
    obs["observation/image/wrist_left"]     uint8  HxWx3 RGB
    obs["observation/image/wrist_right"]    uint8  HxWx3 RGB
    obs["observation/state"]                float32 (65,)  joint angles (rad)
    obs["observation/state/joint_torque"]   float32 (65,)  joint torques, same order
    obs["observation/tactile"]              float32 (60,)  10 fingertips x 6-DoF wrench
    obs["observation/image/tactile_deform"] uint8 (480, 1200, 3)  2x5 fingertip grid
    obs["observation/image/tactile_raw"]    uint8 (480, 1600, 3)  2x5 grid, may be omitted
    obs["prompt"]                           str

Fields the robot cannot source right now arrive zero-filled rather than missing,
so the observation schema is stable across frames.

Action you must return (dict):
    {"actions": np.ndarray of shape (T, 65), float32, ABSOLUTE joint angles (rad)}
    where T is your action horizon. T is declared in the server ``metadata`` below
    and may be any value >= 1.

Joint order of the 65-dim state/action vector (see docs/robot_io_spec.md):
    [ left_arm(7) | left_hand(22) | right_arm(7) | right_hand(22) | motor(7) ]

Run:
    python examples/template_policy_server.py --port 8000
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
from openpi_client import base_policy as _base_policy

from openpi.serving import websocket_policy_server

# 65 = left_arm 7 + left_hand 22 + right_arm 7 + right_hand 22 + motor 7
ACTION_DIM = 65
# Number of future steps returned per inference. Declare it in server metadata.
# Choose any T >= 1 (the pi05 example uses 25).
ACTION_HORIZON = 25


class MyPolicy(_base_policy.BasePolicy):
    """Replace the body of ``infer`` with your own model."""

    def __init__(self) -> None:
        # TODO: load your trained model / weights here.
        pass

    def infer(self, obs: dict) -> dict:
        # Inputs (see module docstring / docs/robot_io_spec.md):
        head_left = np.asarray(obs["observation/image/head_left"])      # uint8 HxWx3
        wrist_left = np.asarray(obs["observation/image/wrist_left"])    # uint8 HxWx3
        wrist_right = np.asarray(obs["observation/image/wrist_right"])  # uint8 HxWx3
        state = np.asarray(obs["observation/state"], dtype=np.float32)  # (65,)
        prompt = obs.get("prompt", "")
        del head_left, wrist_left, wrist_right, prompt  # placeholder uses only state

        # TODO: run your model here and return absolute joint-angle targets (rad),
        # shape (ACTION_HORIZON, 65), ordered per docs/robot_io_spec.md.
        #
        # Placeholder below is a SAFE no-op: it holds the current joint state for
        # every step, so the robot does not move. Replace it with real predictions.
        actions = np.repeat(state[None, :ACTION_DIM], ACTION_HORIZON, axis=0).astype(np.float32)

        return {"actions": actions}

    def reset(self) -> None:
        # Optional: reset any per-episode state of your model here.
        pass


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    policy = MyPolicy()
    metadata = {
        "policy": "template",
        "protocol_version": "origami-v1",
        "action_dim": ACTION_DIM,
        "action_horizon": ACTION_HORIZON,
        "action_type": "absolute_joint_position",
        "action_units": "radians",
    }
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
    )
    logging.info("serving template policy on %s:%d (action_horizon=%d)", args.host, args.port, ACTION_HORIZON)
    server.serve_forever()


if __name__ == "__main__":
    main()
