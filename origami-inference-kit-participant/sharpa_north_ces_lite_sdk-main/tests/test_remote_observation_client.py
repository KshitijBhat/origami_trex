from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

sys.dont_write_bytecode = True

MODULE_PATH = Path(__file__).parents[1] / "examples" / "remote_observation_client.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


remote = load_module("remote_observation_client", MODULE_PATH)


def observation() -> dict:
    return {
        **{
            key: np.full(shape, index, dtype=np.uint8)
            for index, (key, shape) in enumerate(
                {
                    **remote.REQUIRED_IMAGE_SPECS,
                    **remote.OPTIONAL_IMAGE_SPECS,
                }.items(),
                1,
            )
        },
        "observation/state": np.linspace(
            -0.5,
            0.5,
            remote.STATE_DIM,
            dtype=np.float32,
        ),
        "observation/state/joint_torque": np.zeros(65, dtype=np.float32),
        "observation/tactile": np.zeros(60, dtype=np.float32),
        "prompt": "fold the paper",
    }


def metadata() -> dict:
    return {
        "protocol_version": "origami-v1",
        "observation_schema": "policy-infer-input",
        "observation_fields": remote.OBSERVATION_FIELD_METADATA,
        "joint_names": list(remote.JOINT_NAMES),
    }


class FakeTransport:
    def __init__(self, mutate=None):
        self.mutate = mutate
        self.requests = []
        self.closed = False

    def query(self, *, key, payload, timeout_s):
        request = remote.unpack_payload(payload)
        self.requests.append((key, request, timeout_s))
        response = {
            "protocol_version": remote.REMOTE_PROTOCOL_VERSION,
            "operation": remote.OBSERVATION_OPERATION,
            "request_id": request["request_id"],
            "session_id": request["session_id"],
            "observation": observation(),
            "observation_timestamp": 1000.0,
            "metadata": metadata(),
        }
        if self.mutate is not None:
            self.mutate(response)
        return remote.pack_payload(response)

    def close(self):
        self.closed = True


class RemoteObservationClientTests(unittest.TestCase):
    def make_client(self, transport=None, **kwargs):
        return remote.RemoteObservationClient(
            "tcp/public.example:7448",
            session_id="team-a",
            token="secret-token",
            timeout_s=2.5,
            max_observation_age_s=5.0,
            transport=transport or FakeTransport(),
            wall_clock=lambda: 1001.0,
            **kwargs,
        )

    def test_get_observation_is_direct_policy_input(self):
        transport = FakeTransport()
        client = self.make_client(transport)

        result = client.get_observation()

        self.assertEqual(
            set(result),
            {
                *remote.REQUIRED_IMAGE_SPECS,
                *remote.OPTIONAL_IMAGE_SPECS,
                *remote.VECTOR_SPECS,
                "prompt",
            },
        )
        for key, shape in {
            **remote.REQUIRED_IMAGE_SPECS,
            **remote.OPTIONAL_IMAGE_SPECS,
        }.items():
            self.assertEqual(result[key].dtype, np.dtype(np.uint8))
            self.assertEqual(result[key].shape, shape)
            self.assertTrue(result[key].flags.c_contiguous)
        self.assertEqual(result["observation/state"].dtype, np.dtype(np.float32))
        self.assertEqual(result["observation/state"].shape, (65,))
        self.assertTrue(np.isfinite(result["observation/state"]).all())
        self.assertEqual(result["observation/state/joint_torque"].shape, (65,))
        self.assertEqual(result["observation/tactile"].shape, (60,))
        self.assertEqual(result["prompt"], "fold the paper")
        self.assertEqual(client.last_observation_timestamp, 1000.0)
        self.assertEqual(tuple(client.last_metadata["joint_names"]), remote.JOINT_NAMES)

    def test_optional_tactile_raw_may_be_absent(self):
        transport = FakeTransport(
            lambda response: response["observation"].pop(
                "observation/image/tactile_raw"
            )
        )
        result = self.make_client(transport).get_observation()
        self.assertNotIn("observation/image/tactile_raw", result)

    def test_request_uses_team_session_key_token_and_unique_ids(self):
        transport = FakeTransport()
        client = self.make_client(transport)
        client.get_observation()
        client.get_observation()

        self.assertEqual(
            [entry[0] for entry in transport.requests],
            [
                "origami-remote-v1/team-a/observation",
                "origami-remote-v1/team-a/observation",
            ],
        )
        ids = [entry[1]["request_id"] for entry in transport.requests]
        self.assertEqual(len(ids), len(set(ids)))
        for _, request, timeout in transport.requests:
            self.assertEqual(
                request,
                {
                    "protocol_version": "origami-remote-v1",
                    "operation": "observation",
                    "request_id": request["request_id"],
                    "session_id": "team-a",
                    "token": "secret-token",
                },
            )
            self.assertEqual(timeout, 2.5)

    def test_rejects_wrong_envelope_and_service_error(self):
        wrong = FakeTransport(lambda response: response.update(request_id="wrong"))
        with self.assertRaisesRegex(remote.RemoteObservationError, "request_id"):
            self.make_client(wrong).get_observation()

        def set_error(response):
            response["error"] = {
                "code": "RATE_LIMITED",
                "message": "retry later",
                "retryable": True,
            }

        with self.assertRaisesRegex(remote.RemoteObservationError, "RATE_LIMITED"):
            self.make_client(FakeTransport(set_error)).get_observation()

    def test_strictly_rejects_bad_shapes_dtypes_nonfinite_and_joint_order(self):
        cases = {
            "head_left": lambda response: response["observation"].update(
                {remote.CAMERA_IMAGE_KEYS[0]: np.zeros((1, 1, 3), dtype=np.uint8)}
            ),
            "float32": lambda response: response["observation"].update(
                {"observation/state": np.zeros(65, dtype=np.float64)}
            ),
            "NaN": lambda response: response["observation"]["observation/state"].__setitem__(
                0, np.nan
            ),
            "joint_names": lambda response: response["metadata"].update(
                joint_names=list(reversed(remote.JOINT_NAMES))
            ),
        }
        for expected, mutate in cases.items():
            with self.subTest(expected=expected), self.assertRaisesRegex(
                remote.RemoteObservationError,
                expected,
            ):
                self.make_client(FakeTransport(mutate)).get_observation()

    def test_rejects_stale_or_invalid_timestamp(self):
        stale = FakeTransport(
            lambda response: response.update(observation_timestamp=990.0)
        )
        with self.assertRaisesRegex(remote.RemoteObservationError, "stale"):
            self.make_client(stale).get_observation()

        invalid = FakeTransport(
            lambda response: response.update(observation_timestamp="1000")
        )
        with self.assertRaisesRegex(remote.RemoteObservationError, "Unix timestamp"):
            self.make_client(invalid).get_observation()

    def test_tls_requires_ca_and_complete_mtls_pair(self):
        with self.assertRaisesRegex(ValueError, "root_ca"):
            remote.RemoteObservationClient(
                "tls/public.example:7448",
                session_id="team-a",
                token="token",
                transport=FakeTransport(),
            )
        with self.assertRaisesRegex(ValueError, "supplied together"):
            remote.RemoteObservationClient(
                "tcp/public.example:7448",
                session_id="team-a",
                token="token",
                tls_client_certificate="client.pem",
                transport=FakeTransport(),
            )

    def test_close_is_idempotent_and_source_has_no_robot_or_publish_api(self):
        transport = FakeTransport()
        client = self.make_client(transport)
        client.close()
        client.close()
        self.assertTrue(transport.closed)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            client.get_observation()

        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("sharpa_north_ces_lite", source)
        self.assertNotIn("publish_single_action", source)
        self.assertNotIn("/observe/", source)
        self.assertNotIn("/action/left", source)

if __name__ == "__main__":
    unittest.main()
