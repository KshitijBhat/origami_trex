from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

MODULE_PATH = Path(__file__).parents[1] / "examples" / "policy_server_template.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


server_module = load_module("policy_server_template", MODULE_PATH)


def request(operation: str, **extra) -> dict:
    return {
        "protocol_version": "origami-zenoh-v1",
        "operation": operation,
        "request_id": f"{operation}-request",
        "session_id": "test-session",
        **extra,
    }


def observation() -> dict:
    return {
        "observation/image/head_left": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/image/head_right": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/image/wrist_left": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/image/wrist_right": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/state": np.linspace(-0.5, 0.5, 65, dtype=np.float32),
        "observation/state/joint_torque": np.zeros(65, dtype=np.float32),
        "observation/tactile": np.zeros(60, dtype=np.float32),
        "observation/image/tactile_deform": np.zeros(
            (480, 1200, 3), dtype=np.uint8
        ),
        "observation/image/tactile_raw": np.zeros(
            (480, 1600, 3), dtype=np.uint8
        ),
        "prompt": "fold the paper",
    }


class PolicyServerTemplateTests(unittest.TestCase):
    def setUp(self) -> None:
        policy = server_module.TeamPolicy(action_horizon=4)
        self.server = server_module.OrigamiZenohServer(
            policy,
            endpoint="tcp/127.0.0.1:7447",
            session_id="test-session",
            action_horizon=4,
        )

    def test_metadata_reset_and_infer_contract(self) -> None:
        metadata = self.server.process("metadata", request("metadata"))
        self.assertEqual(metadata["metadata"]["action_horizon"], 4)
        self.assertEqual(tuple(metadata["metadata"]["joint_names"]), server_module.JOINT_NAMES)

        reset = self.server.process("reset", request("reset"))
        self.assertIs(reset["ok"], True)

        obs = observation()
        result = self.server.process("infer", request("infer", observation=obs))
        self.assertEqual(result["actions"].dtype, np.dtype(np.float32))
        self.assertEqual(result["actions"].shape, (4, 65))
        np.testing.assert_array_equal(
            result["actions"],
            np.repeat(obs["observation/state"][None, :], 4, axis=0),
        )

    def test_codec_round_trip_and_validation(self) -> None:
        value = {"observation": observation()}
        decoded = server_module.unpack_payload(server_module.pack_payload(value))
        np.testing.assert_array_equal(
            decoded["observation"]["observation/state"],
            value["observation"]["observation/state"],
        )

        bad = observation()
        bad["observation/state"] = bad["observation/state"].astype(np.float64)
        with self.assertRaisesRegex(ValueError, "float32"):
            self.server.process("infer", request("infer", observation=bad))


if __name__ == "__main__":
    unittest.main()
