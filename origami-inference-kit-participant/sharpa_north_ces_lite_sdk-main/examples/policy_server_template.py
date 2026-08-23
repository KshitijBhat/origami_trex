#!/usr/bin/env python3
"""Framework-neutral origami-zenoh-v1 policy server template.

Replace ``TeamPolicy`` with the team's model adapter. The public server boundary
must remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import threading
import time
import uuid
from collections.abc import Mapping
from typing import Any

import msgpack
import numpy as np
import zenoh

TRANSPORT_VERSION = "origami-zenoh-v1"
SEMANTIC_VERSION = "origami-v1"
ACTION_DIM = 65
IMAGE_SHAPE = (224, 224, 3)
REQUIRED_IMAGE_SPECS = {
    "observation/image/head_left": IMAGE_SHAPE,
    "observation/image/head_right": IMAGE_SHAPE,
    "observation/image/wrist_left": IMAGE_SHAPE,
    "observation/image/wrist_right": IMAGE_SHAPE,
    "observation/image/tactile_deform": (480, 1200, 3),
}
OPTIONAL_IMAGE_SPECS = {
    "observation/image/tactile_raw": (480, 1600, 3),
}
VECTOR_SPECS = {
    "observation/state": (ACTION_DIM,),
    "observation/state/joint_torque": (ACTION_DIM,),
    "observation/tactile": (60,),
}
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024


def _hand_joint_names(side: str) -> tuple[str, ...]:
    return (
        f"{side}_thumb_CMC_FE",
        f"{side}_thumb_CMC_AA",
        f"{side}_thumb_MCP_FE",
        f"{side}_thumb_MCP_AA",
        f"{side}_thumb_IP",
        f"{side}_index_MCP_FE",
        f"{side}_index_MCP_AA",
        f"{side}_index_PIP",
        f"{side}_index_DIP",
        f"{side}_middle_MCP_FE",
        f"{side}_middle_MCP_AA",
        f"{side}_middle_PIP",
        f"{side}_middle_DIP",
        f"{side}_ring_MCP_FE",
        f"{side}_ring_MCP_AA",
        f"{side}_ring_PIP",
        f"{side}_ring_DIP",
        f"{side}_pinky_CMC",
        f"{side}_pinky_MCP_FE",
        f"{side}_pinky_MCP_AA",
        f"{side}_pinky_PIP",
        f"{side}_pinky_DIP",
    )


JOINT_NAMES = (
    tuple(f"left_arm_joint_{index}" for index in range(1, 8))
    + _hand_joint_names("left")
    + tuple(f"right_arm_joint_{index}" for index in range(1, 8))
    + _hand_joint_names("right")
    + (
        "lower_body_joint_1",
        "lower_body_joint_2",
        "lower_body_joint_3",
        "lower_body_joint_4",
        "lower_body_joint_5",
        "neck_joint_1",
        "neck_joint_2",
    )
)

if len(JOINT_NAMES) != ACTION_DIM or len(set(JOINT_NAMES)) != ACTION_DIM:
    raise RuntimeError("joint contract must contain 65 unique names")


def _pack_numpy(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"O", "V", "c"}:
            raise ValueError(f"unsupported numpy dtype: {value.dtype}")
        array = np.ascontiguousarray(value)
        return {
            b"__ndarray__": True,
            b"data": array.tobytes(),
            b"dtype": array.dtype.str,
            b"shape": array.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _mapping_value(value: Mapping[Any, Any], key: str) -> Any:
    return value[key] if key in value else value.get(key.encode())


def _unpack_numpy(value: dict[Any, Any]) -> Any:
    if _mapping_value(value, "__ndarray__") is True:
        data = _mapping_value(value, "data")
        shape = _mapping_value(value, "shape")
        dtype = np.dtype(_mapping_value(value, "dtype"))
        if (
            not isinstance(data, bytes)
            or not isinstance(shape, (list, tuple))
            or len(shape) > 8
            or dtype.kind in {"O", "V", "c"}
            or dtype.hasobject
        ):
            raise ValueError("invalid numpy array payload")
        normalized_shape = tuple(int(dimension) for dimension in shape)
        if any(dimension < 0 for dimension in normalized_shape):
            raise ValueError("invalid numpy array shape")
        expected_size = math.prod(normalized_shape) * dtype.itemsize
        if expected_size > MAX_PAYLOAD_BYTES or len(data) != expected_size:
            raise ValueError("numpy array payload size does not match shape")
        return np.frombuffer(data, dtype=dtype).reshape(normalized_shape)
    if _mapping_value(value, "__npgeneric__") is True:
        return np.dtype(_mapping_value(value, "dtype")).type(
            _mapping_value(value, "data")
        )
    return value


def pack_payload(value: Any) -> bytes:
    payload = msgpack.packb(value, default=_pack_numpy, use_bin_type=True)
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("response exceeds 64 MiB")
    return payload


def unpack_payload(value: Any) -> Any:
    payload = value.to_bytes() if hasattr(value, "to_bytes") else bytes(value)
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("request exceeds 64 MiB")
    return msgpack.unpackb(
        payload,
        object_hook=_unpack_numpy,
        raw=False,
        strict_map_key=False,
        max_bin_len=MAX_PAYLOAD_BYTES,
        max_array_len=1_000_000,
        max_map_len=10_000,
        max_str_len=1_000_000,
    )


class TeamPolicy:
    """Replace this class with the team's model and adapter."""

    def __init__(self, action_horizon: int) -> None:
        self.action_horizon = action_horizon
        # TODO: load model, checkpoint, normalization and tokenizer here.

    def reset(self) -> None:
        # TODO: clear all episode-scoped model state here.
        pass

    def infer(self, observation: dict[str, Any]) -> np.ndarray:
        # TODO: preprocess the public observation, run the model, denormalize,
        # convert to absolute radians and return exactly float32[T, 65].
        #
        # This placeholder holds the current position. It exists only so teams can
        # verify their container/protocol wiring before integrating a real model.
        state = observation["observation/state"]
        return np.repeat(state[None, :], self.action_horizon, axis=0).astype(
            np.float32
        )


class OrigamiZenohServer:
    def __init__(
        self,
        policy: TeamPolicy,
        *,
        endpoint: str,
        session_id: str,
        action_horizon: int,
    ) -> None:
        if action_horizon < 1 or action_horizon > 1024:
            raise ValueError("action_horizon must be in [1, 1024]")
        self.policy = policy
        self.endpoint = endpoint
        self.session_id = session_id
        self.action_horizon = action_horizon
        self._policy_lock = threading.Lock()
        self._stop = threading.Event()
        self._session: Any | None = None
        self._queryables: list[Any] = []
        self.metadata = {
            "protocol_version": SEMANTIC_VERSION,
            "action_dim": ACTION_DIM,
            "action_horizon": action_horizon,
            "action_type": "absolute_joint_position",
            "action_units": "radians",
            "joint_names": JOINT_NAMES,
        }

    def serve_forever(self) -> None:
        config = zenoh.Config()
        config.insert_json5("mode", json.dumps("client"))
        config.insert_json5("connect/endpoints", json.dumps([self.endpoint]))
        config.insert_json5("scouting/multicast/enabled", "false")
        config.insert_json5("transport/shared_memory/enabled", "false")
        self._session = zenoh.open(config)
        self._queryables = [
            self._session.declare_queryable(
                f"{TRANSPORT_VERSION}/{operation}",
                self._handle_query,
                complete=True,
            )
            for operation in ("metadata", "reset", "infer")
        ]
        signal.signal(signal.SIGTERM, lambda *_: self._stop.set())
        signal.signal(signal.SIGINT, lambda *_: self._stop.set())
        logging.info(
            "READY transport=%s endpoint=%s horizon=%d",
            TRANSPORT_VERSION,
            self.endpoint,
            self.action_horizon,
        )
        self._stop.wait()
        for queryable in self._queryables:
            queryable.undeclare()
        self._session.close()

    def _handle_query(self, query: Any) -> None:
        operation = str(query.key_expr).rsplit("/", 1)[-1]
        request: Any = None
        try:
            request = unpack_payload(query.payload)
            response = self.process(operation, request)
        except Exception as exc:  # noqa: BLE001 - sanitized protocol error
            error_id = uuid.uuid4().hex
            logging.error(
                "request failed operation=%s error_id=%s type=%s",
                operation,
                error_id,
                type(exc).__name__,
            )
            public_message = f"request failed; error_id={error_id}"
            if not isinstance(request, Mapping):
                query.reply_err(
                    pack_payload(
                        {
                            "error": {
                                "code": "INVALID_REQUEST",
                                "message": public_message,
                                "retryable": False,
                            }
                        }
                    ),
                    encoding="application/msgpack",
                )
                return
            response = self._envelope(operation, request)
            response["error"] = {
                "code": (
                    "INFERENCE_FAILED" if operation == "infer" else "INVALID_REQUEST"
                ),
                "message": public_message,
                "retryable": False,
            }
        query.reply(
            str(query.key_expr),
            pack_payload(response),
            encoding="application/msgpack",
        )

    def process(self, operation: str, request: Any) -> dict[str, Any]:
        if not isinstance(request, Mapping):
            raise ValueError("request must be a MessagePack map")
        response = self._envelope(operation, request)
        if request.get("protocol_version") != TRANSPORT_VERSION:
            raise ValueError("invalid protocol_version")
        if request.get("operation") != operation:
            raise ValueError("operation does not match queryable key")
        if request.get("session_id") != self.session_id:
            raise ValueError("session_id does not match assigned session")
        if not isinstance(request.get("request_id"), str) or not request["request_id"]:
            raise ValueError("request_id must be a non-empty string")

        if operation == "metadata":
            response["metadata"] = self.metadata
            return response
        if operation == "reset":
            with self._policy_lock:
                self.policy.reset()
            response["ok"] = True
            return response
        if operation != "infer":
            raise ValueError(f"unsupported operation: {operation}")

        observation = request.get("observation")
        self._validate_observation(observation)
        started = time.monotonic()
        with self._policy_lock:
            actions = self.policy.infer(dict(observation))
        actions = np.asarray(actions)
        expected_shape = (self.action_horizon, ACTION_DIM)
        if actions.dtype != np.float32 or actions.shape != expected_shape:
            raise ValueError(
                f"policy actions must be float32{expected_shape}, "
                f"got {actions.dtype}{actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("policy actions contain NaN or Inf")
        response["actions"] = np.ascontiguousarray(actions)
        response["server_timing"] = {
            "infer_ms": (time.monotonic() - started) * 1000.0
        }
        return response

    def _envelope(
        self,
        operation: str,
        request: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "protocol_version": TRANSPORT_VERSION,
            "operation": operation,
            "request_id": request.get("request_id"),
            "session_id": self.session_id,
        }

    @staticmethod
    def _validate_observation(observation: Any) -> None:
        if not isinstance(observation, Mapping):
            raise ValueError("infer request must contain an observation map")
        required = {*REQUIRED_IMAGE_SPECS, *VECTOR_SPECS, "prompt"}
        allowed = required | set(OPTIONAL_IMAGE_SPECS)
        if not required.issubset(observation) or not set(observation).issubset(allowed):
            raise ValueError("observation keys do not match the public full schema")
        for key, shape in {**REQUIRED_IMAGE_SPECS, **OPTIONAL_IMAGE_SPECS}.items():
            if key not in observation:
                continue
            image = observation.get(key)
            if (
                not isinstance(image, np.ndarray)
                or image.dtype != np.uint8
                or image.shape != shape
            ):
                raise ValueError(f"{key} must be uint8{shape}")
        for key, shape in VECTOR_SPECS.items():
            vector = observation.get(key)
            if (
                not isinstance(vector, np.ndarray)
                or vector.dtype != np.float32
                or vector.shape != shape
                or not np.isfinite(vector).all()
            ):
                raise ValueError(f"{key} must be finite float32{shape}")
        if not isinstance(observation.get("prompt"), str):
            raise ValueError("prompt must be a string")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("ORIGAMI_ZENOH_ENDPOINT"),
    )
    parser.add_argument(
        "--session-id",
        default=os.environ.get("ORIGAMI_SESSION_ID"),
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=int(os.environ.get("ORIGAMI_ACTION_HORIZON", "25")),
    )
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    if not args.endpoint:
        raise SystemExit("--endpoint or ORIGAMI_ZENOH_ENDPOINT is required")
    if not args.session_id:
        raise SystemExit("--session-id or ORIGAMI_SESSION_ID is required")
    policy = TeamPolicy(args.action_horizon)
    server = OrigamiZenohServer(
        policy,
        endpoint=args.endpoint,
        session_id=args.session_id,
        action_horizon=args.action_horizon,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    raise SystemExit(main())
