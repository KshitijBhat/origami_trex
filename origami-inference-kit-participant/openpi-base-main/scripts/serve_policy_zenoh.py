#!/usr/bin/env python3
"""Serve an OpenPI checkpoint through the public origami-zenoh-v1 contract."""

from __future__ import annotations

import dataclasses
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
import tyro
import zenoh

import serve_policy as _base
from openpi.training import config as _config

TRANSPORT_VERSION = "origami-zenoh-v1"
SEMANTIC_VERSION = "origami-v1"
ACTION_DIM = 65
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
REQUIRED_IMAGE_SPECS = {
    "observation/image/head_left": (224, 224, 3),
    "observation/image/head_right": (224, 224, 3),
    "observation/image/wrist_left": (224, 224, 3),
    "observation/image/wrist_right": (224, 224, 3),
    "observation/image/tactile_deform": (480, 1200, 3),
}
OPTIONAL_IMAGE_SPECS = {
    "observation/image/tactile_raw": (480, 1600, 3),
}
VECTOR_SPECS = {
    "observation/state": (65,),
    "observation/state/joint_torque": (65,),
    "observation/tactile": (60,),
}


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
    + tuple(f"lower_body_joint_{index}" for index in range(1, 6))
    + ("neck_joint_1", "neck_joint_2")
)

if len(JOINT_NAMES) != ACTION_DIM or len(set(JOINT_NAMES)) != ACTION_DIM:
    raise RuntimeError("joint contract must contain 65 unique names")


@dataclasses.dataclass
class Args(_base.Args):
    zenoh_endpoint: str | None = None
    session_id: str | None = None


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


class OpenPIZenohServer:
    def __init__(
        self,
        policy: Any,
        *,
        endpoint: str,
        session_id: str,
        action_horizon: int,
    ) -> None:
        self.policy = policy
        self.endpoint = endpoint
        self.session_id = session_id
        self.action_horizon = int(action_horizon)
        self._policy_lock = threading.Lock()
        self._stop = threading.Event()
        self._session: Any | None = None
        self._queryables: list[Any] = []
        self.metadata = {
            "protocol_version": SEMANTIC_VERSION,
            "action_dim": ACTION_DIM,
            "action_type": "absolute_joint_position",
            "action_units": "radians",
            "action_horizon": self.action_horizon,
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
            "READY transport=%s endpoint=%s session=%s horizon=%d",
            TRANSPORT_VERSION,
            self.endpoint,
            self.session_id,
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
            response = self._process(operation, request)
        except Exception as exc:  # noqa: BLE001 - sanitized wire error
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

    def _process(self, operation: str, request: Any) -> dict[str, Any]:
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
            result = self.policy.infer(dict(observation))
        if not isinstance(result, Mapping) or "actions" not in result:
            raise ValueError("policy result must contain actions")
        actions = np.ascontiguousarray(result["actions"], dtype=np.float32)
        expected_shape = (self.action_horizon, ACTION_DIM)
        if actions.shape != expected_shape:
            raise ValueError(
                f"policy actions must have shape {expected_shape}, got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("policy actions contain NaN or Inf")
        response["actions"] = actions
        response["policy_timing"] = result.get("policy_timing", {})
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
            image = observation[key]
            if (
                not isinstance(image, np.ndarray)
                or image.dtype != np.uint8
                or image.shape != shape
            ):
                raise ValueError(f"{key} must be uint8{shape}")
        for key, shape in VECTOR_SPECS.items():
            vector = observation[key]
            if (
                not isinstance(vector, np.ndarray)
                or vector.dtype != np.float32
                or vector.shape != shape
                or not np.isfinite(vector).all()
            ):
                raise ValueError(f"{key} must be finite float32{shape}")
        if not isinstance(observation.get("prompt"), str):
            raise ValueError("prompt must be a string")


def main(args: Args) -> None:
    endpoint = args.zenoh_endpoint or os.environ.get("ORIGAMI_ZENOH_ENDPOINT")
    session_id = args.session_id or os.environ.get("ORIGAMI_SESSION_ID")
    if not endpoint:
        raise ValueError(
            "--zenoh-endpoint or ORIGAMI_ZENOH_ENDPOINT is required"
        )
    if not session_id:
        raise ValueError("--session-id or ORIGAMI_SESSION_ID is required")
    if not isinstance(args.policy, _base.Checkpoint):
        raise ValueError("Zenoh submission server requires policy:checkpoint")

    policy = _base.create_policy(args)
    action_horizon = _config.get_config(args.policy.config).model.action_horizon
    server = OpenPIZenohServer(
        policy,
        endpoint=endpoint,
        session_id=session_id,
        action_horizon=action_horizon,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    main(tyro.cli(Args))
