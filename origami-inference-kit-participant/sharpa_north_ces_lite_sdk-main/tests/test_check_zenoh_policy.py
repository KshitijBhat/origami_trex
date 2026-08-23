from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
import unittest

import msgpack
import numpy as np

sys.dont_write_bytecode = True


MODULE_PATH = Path(__file__).parents[1] / "examples" / "check_zenoh_policy.py"
ROBOT_IO_DOC_PATH = Path(__file__).parents[2] / "docs" / "robot_io_spec.md"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


validator = load_module("check_zenoh_policy", MODULE_PATH)


def metadata(horizon: int = 4) -> dict:
    return {
        "protocol_version": "origami-v1",
        "action_dim": 65,
        "action_horizon": horizon,
        "action_type": "absolute_joint_position",
        "action_units": "radians",
        "joint_names": validator.JOINT_NAMES,
    }


class FakeZBytes:
    def __init__(self, value: bytes):
        self.value = value

    def to_bytes(self) -> bytes:
        return self.value


class FakeSample:
    def __init__(self, payload: bytes):
        self.payload = FakeZBytes(payload)


class FakeReply:
    def __init__(self, payload: bytes):
        self.ok = FakeSample(payload)
        self.err = None


class FakeSession:
    def __init__(self, *, duplicate: bool = False, wrong_request_id: bool = False):
        self.requests: list[tuple[str, dict, float, object]] = []
        self.duplicate = duplicate
        self.wrong_request_id = wrong_request_id

    def get(self, key, *, payload, timeout, consolidation):
        request = validator.unpack_payload(payload)
        self.requests.append((key, request, timeout, consolidation))
        response = {
            "protocol_version": validator.ZENOH_PROTOCOL_VERSION,
            "operation": request["operation"],
            "request_id": "wrong" if self.wrong_request_id else request["request_id"],
            "session_id": request["session_id"],
        }
        if request["operation"] == "metadata":
            response["metadata"] = metadata()
        elif request["operation"] == "reset":
            response["ok"] = True
        elif request["operation"] == "infer":
            response["actions"] = np.zeros((4, 65), dtype=np.float32)
        reply = FakeReply(validator.pack_payload(response))
        return [reply, reply] if self.duplicate else [reply]


class CodecTests(unittest.TestCase):
    def test_round_trip_protocol_arrays(self) -> None:
        value = {
            "image": np.arange(24, dtype=np.uint8).reshape(2, 4, 3),
            "state": np.linspace(-1, 1, 65, dtype=np.float32),
        }

        decoded = validator.unpack_payload(validator.pack_payload(value))

        np.testing.assert_array_equal(decoded["image"], value["image"])
        np.testing.assert_array_equal(decoded["state"], value["state"])
        self.assertEqual(decoded["image"].dtype, np.dtype(np.uint8))
        self.assertEqual(decoded["state"].dtype, np.dtype(np.float32))

    def test_object_arrays_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validator.pack_payload(np.array([object()], dtype=object))

    def test_malformed_array_byte_length_is_rejected(self) -> None:
        malformed = {
            b"__ndarray__": True,
            b"data": b"\x00",
            b"dtype": "<f4",
            b"shape": [1],
        }
        payload = msgpack.packb(malformed, use_bin_type=True)

        with self.assertRaisesRegex(validator.ValidationError, "byte length mismatch"):
            validator.unpack_payload(payload)


class ContractTests(unittest.TestCase):
    def test_synthetic_observation_contract(self) -> None:
        observation = validator.make_synthetic_observation()

        self.assertEqual(
            set(observation),
            {
                "observation/image/head_left",
                "observation/image/head_right",
                "observation/image/wrist_left",
                "observation/image/wrist_right",
                "observation/state",
                "observation/state/joint_torque",
                "observation/tactile",
                "observation/image/tactile_deform",
                "observation/image/tactile_raw",
                "prompt",
            },
        )
        for key in (
            "observation/image/head_left",
            "observation/image/head_right",
            "observation/image/wrist_left",
            "observation/image/wrist_right",
        ):
            self.assertEqual(observation[key].shape, (224, 224, 3))
            self.assertEqual(observation[key].dtype, np.dtype(np.uint8))
            self.assertTrue(observation[key].flags.c_contiguous)
        self.assertEqual(observation["observation/state"].shape, (65,))
        self.assertEqual(observation["observation/state"].dtype, np.dtype(np.float32))
        self.assertEqual(observation["observation/state/joint_torque"].shape, (65,))
        self.assertEqual(observation["observation/tactile"].shape, (60,))
        self.assertEqual(
            observation["observation/image/tactile_deform"].shape,
            (480, 1200, 3),
        )
        self.assertEqual(
            observation["observation/image/tactile_raw"].shape,
            (480, 1600, 3),
        )
        self.assertIsInstance(observation["prompt"], str)

    def test_metadata_and_actions_must_share_horizon(self) -> None:
        parsed_metadata = validator.validate_metadata(
            {"metadata": metadata()},
            expected_horizon=4,
        )
        actions = np.zeros((4, 65), dtype=np.float32)

        self.assertIs(
            validator.validate_infer({"actions": actions}, parsed_metadata["action_horizon"]),
            actions,
        )

        with self.assertRaisesRegex(validator.ValidationError, "actions shape"):
            validator.validate_infer({"actions": actions[:3]}, parsed_metadata["action_horizon"])

    def test_metadata_requires_semantic_version_and_all_joint_names(self) -> None:
        wrong_version = metadata()
        wrong_version["protocol_version"] = "origami-zenoh-v1"
        with self.assertRaisesRegex(validator.ValidationError, "protocol_version"):
            validator.validate_metadata({"metadata": wrong_version})

        missing_names = metadata()
        del missing_names["joint_names"]
        with self.assertRaisesRegex(validator.ValidationError, "joint_names"):
            validator.validate_metadata({"metadata": missing_names})

    def test_non_finite_actions_are_rejected(self) -> None:
        actions = np.zeros((2, 65), dtype=np.float32)
        actions[0, 0] = np.nan

        with self.assertRaisesRegex(validator.ValidationError, "NaN or Inf"):
            validator.validate_infer({"actions": actions}, 2)

    def test_application_error_is_rejected(self) -> None:
        with self.assertRaisesRegex(validator.ValidationError, "NOT_READY"):
            validator.validate_infer(
                {
                    "error": {
                        "code": "NOT_READY",
                        "message": "loading",
                        "retryable": True,
                    }
                },
                2,
            )

    def test_endpoint_and_session_validation(self) -> None:
        for endpoint in ("tcp/127.0.0.1:7447", "tcp/origami-router:7447"):
            with self.subTest(endpoint=endpoint):
                self.assertEqual(validator.validate_endpoint(endpoint), endpoint)
        self.assertEqual(validator.validate_session_id("opaque/session:value"), "opaque/session:value")

        for endpoint in ("udp/127.0.0.1:7447", "tcp/:7447", "tcp/127.0.0.1:0"):
            with self.subTest(endpoint=endpoint), self.assertRaises(validator.ValidationError):
                validator.validate_endpoint(endpoint)
        with self.assertRaises(validator.ValidationError):
            validator.validate_session_id("")

    def test_fixed_keys_unique_request_ids_and_envelopes(self) -> None:
        session = FakeSession()
        first = validator.query_once(session, "metadata", "session-a", 2.5)
        second = validator.query_once(session, "metadata", "session-a", 2.5)
        reset = validator.query_once(session, "reset", "session-a", 2.5)
        infer = validator.query_once(
            session,
            "infer",
            "session-a",
            2.5,
            observation=validator.make_synthetic_observation(),
        )

        self.assertEqual([entry[0] for entry in session.requests], [
            "origami-zenoh-v1/metadata",
            "origami-zenoh-v1/metadata",
            "origami-zenoh-v1/reset",
            "origami-zenoh-v1/infer",
        ])
        request_ids = [entry[1]["request_id"] for entry in session.requests]
        self.assertEqual(len(request_ids), len(set(request_ids)))
        for _, request, _, _ in session.requests:
            self.assertEqual(request["protocol_version"], "origami-zenoh-v1")
            self.assertEqual(request["session_id"], "session-a")
        self.assertNotIn("observation", session.requests[0][1])
        self.assertNotIn("observation", session.requests[2][1])
        self.assertIn("observation", session.requests[3][1])
        self.assertEqual(first["request_id"], request_ids[0])
        self.assertEqual(second["request_id"], request_ids[1])
        self.assertEqual(reset["request_id"], request_ids[2])
        self.assertEqual(infer["request_id"], request_ids[3])

    def test_rejects_duplicate_and_mismatched_envelope_replies(self) -> None:
        with self.assertRaisesRegex(validator.ValidationError, "exactly one reply"):
            validator.query_once(FakeSession(duplicate=True), "metadata", "session-a", 1.0)
        with self.assertRaisesRegex(validator.ValidationError, "request_id"):
            validator.query_once(
                FakeSession(wrong_request_id=True),
                "metadata",
                "session-a",
                1.0,
            )

    def test_strictly_validates_all_reply_envelope_fields(self) -> None:
        expected = {
            "protocol_version": "origami-zenoh-v1",
            "operation": "infer",
            "request_id": "request-a",
            "session_id": "session-a",
        }
        for field in expected:
            bad_reply = {**expected, field: "wrong"}
            with self.subTest(field=field), self.assertRaisesRegex(
                validator.ValidationError,
                field,
            ):
                validator.validate_reply_envelope(
                    bad_reply,
                    operation="infer",
                    request_id="request-a",
                    session_id="session-a",
                )

    def test_documented_joint_names_match_validator(self) -> None:
        document = ROBOT_IO_DOC_PATH.read_text(encoding="utf-8")
        names_section = document.split(
            "The exact `metadata[\"joint_names\"]`, state order, and action-column order are:",
            1,
        )[1]
        literal = names_section.split("```python", 1)[1].split("```", 1)[0]
        self.assertEqual(tuple(ast.literal_eval(literal)), validator.JOINT_NAMES)


if __name__ == "__main__":
    unittest.main()
