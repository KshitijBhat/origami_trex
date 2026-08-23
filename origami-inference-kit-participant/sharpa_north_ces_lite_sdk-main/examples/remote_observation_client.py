#!/usr/bin/env python3
"""Public read-only client for policy-shaped observations over Zenoh."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import uuid
from collections.abc import Iterable, Mapping
from typing import Any

import msgpack
import numpy as np

REMOTE_PROTOCOL_VERSION = "origami-remote-v1"
SEMANTIC_PROTOCOL_VERSION = "origami-v1"
OBSERVATION_OPERATION = "observation"
STATE_DIM = 65
IMAGE_SHAPE = (224, 224, 3)
TACTILE_DEFORM_SHAPE = (480, 1200, 3)
TACTILE_RAW_SHAPE = (480, 1600, 3)
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
CAMERA_IMAGE_KEYS = (
    "observation/image/head_left",
    "observation/image/head_right",
    "observation/image/wrist_left",
    "observation/image/wrist_right",
)
REQUIRED_IMAGE_SPECS = {
    **{key: IMAGE_SHAPE for key in CAMERA_IMAGE_KEYS},
    "observation/image/tactile_deform": TACTILE_DEFORM_SHAPE,
}
OPTIONAL_IMAGE_SPECS = {
    "observation/image/tactile_raw": TACTILE_RAW_SHAPE,
}
IMAGE_KEYS = tuple(REQUIRED_IMAGE_SPECS)
VECTOR_SPECS = {
    "observation/state": (STATE_DIM,),
    "observation/state/joint_torque": (STATE_DIM,),
    "observation/tactile": (60,),
}
OBSERVATION_FIELD_METADATA = {
    **{
        key: {"dtype": "uint8", "shape": list(shape), "required": True}
        for key, shape in REQUIRED_IMAGE_SPECS.items()
    },
    **{
        key: {"dtype": "uint8", "shape": list(shape), "required": False}
        for key, shape in OPTIONAL_IMAGE_SPECS.items()
    },
    **{
        key: {"dtype": "float32", "shape": list(shape), "required": True}
        for key, shape in VECTOR_SPECS.items()
    },
    "prompt": {"dtype": "str", "shape": [], "required": True},
}
_SESSION_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


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

if len(JOINT_NAMES) != STATE_DIM or len(set(JOINT_NAMES)) != STATE_DIM:
    raise RuntimeError("joint contract must contain 65 unique joint names")


class RemoteObservationError(RuntimeError):
    """The remote observation service or payload violated its contract."""


class RemoteObservationClient:
    """Fetch observations that can be passed directly to a competition policy."""

    def __init__(
        self,
        endpoint: str,
        *,
        session_id: str,
        token: str,
        timeout_s: float = 10.0,
        max_observation_age_s: float = 5.0,
        tls_root_ca_certificate: str | None = None,
        tls_client_certificate: str | None = None,
        tls_client_private_key: str | None = None,
        session: Any | None = None,
        transport: Any | None = None,
        wall_clock: Any = time.time,
    ) -> None:
        if session is not None and transport is not None:
            raise ValueError("session and transport are mutually exclusive")
        _validate_endpoint(endpoint)
        if not isinstance(session_id, str) or _SESSION_ID_PATTERN.fullmatch(session_id) is None:
            raise ValueError("session_id must contain only A-Z, a-z, 0-9, dot, underscore, or dash")
        if not isinstance(token, str) or not 1 <= len(token) <= 4096:
            raise ValueError("token must be a string of 1..4096 characters")
        if not math.isfinite(float(timeout_s)) or float(timeout_s) <= 0:
            raise ValueError("timeout_s must be finite and positive")
        if (
            not math.isfinite(float(max_observation_age_s))
            or float(max_observation_age_s) <= 0
        ):
            raise ValueError("max_observation_age_s must be finite and positive")
        if endpoint.startswith("tls/") and not tls_root_ca_certificate:
            raise ValueError("tls_root_ca_certificate is required for a tls/ endpoint")
        if bool(tls_client_certificate) != bool(tls_client_private_key):
            raise ValueError(
                "tls_client_certificate and tls_client_private_key must be supplied together"
            )

        self.endpoint = endpoint
        self.session_id = session_id
        self.key = f"{REMOTE_PROTOCOL_VERSION}/{session_id}/{OBSERVATION_OPERATION}"
        self.timeout_s = float(timeout_s)
        self.max_observation_age_s = float(max_observation_age_s)
        self.last_metadata: dict[str, Any] | None = None
        self.last_observation_timestamp: float | None = None
        self._token = token
        self._wall_clock = wall_clock
        self._closed = False
        self._transport = transport or _ZenohTransport(
            endpoint,
            session=session,
            tls_root_ca_certificate=tls_root_ca_certificate,
            tls_client_certificate=tls_client_certificate,
            tls_client_private_key=tls_client_private_key,
        )

    def get_observation(self) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("RemoteObservationClient is closed")
        request_id = uuid.uuid4().hex
        request = {
            "protocol_version": REMOTE_PROTOCOL_VERSION,
            "operation": OBSERVATION_OPERATION,
            "request_id": request_id,
            "session_id": self.session_id,
            "token": self._token,
        }
        replies = self._transport.query(
            key=self.key,
            payload=pack_payload(request),
            timeout_s=self.timeout_s,
        )
        payloads = _materialize_payloads(replies)
        if not payloads:
            raise TimeoutError(f"no observation reply within {self.timeout_s:g}s")
        if len(payloads) != 1:
            raise RemoteObservationError(
                f"expected exactly one observation reply, got {len(payloads)}"
            )
        response = unpack_payload(payloads[0])
        _validate_envelope(response, request_id=request_id, session_id=self.session_id)
        error = response.get("error")
        if error is not None:
            if not isinstance(error, Mapping):
                raise RemoteObservationError("service returned a malformed error")
            code = error.get("code", "UNKNOWN")
            message = error.get("message", "")
            raise RemoteObservationError(f"{code}: {message}")

        metadata = _validate_metadata(response.get("metadata"))
        timestamp = _validate_timestamp(
            response.get("observation_timestamp"),
            now=float(self._wall_clock()),
            max_age_s=self.max_observation_age_s,
        )
        observation = _validate_observation(response.get("observation"))
        self.last_metadata = metadata
        self.last_observation_timestamp = timestamp
        return observation

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self._transport, "close", None)
        if close is not None:
            close()

    def __enter__(self) -> RemoteObservationClient:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def pack_payload(value: Any) -> bytes:
    payload = msgpack.packb(value, default=_pack_numpy, use_bin_type=True)
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise RemoteObservationError("encoded payload exceeds 64 MiB")
    return payload


def unpack_payload(value: bytes | bytearray | memoryview) -> Any:
    payload = bytes(value)
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise RemoteObservationError("reply payload exceeds 64 MiB")
    try:
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
    except RemoteObservationError:
        raise
    except Exception as error:
        raise RemoteObservationError(f"invalid MessagePack reply: {error}") from error


def _validate_endpoint(endpoint: str) -> None:
    if not isinstance(endpoint, str) or not endpoint.startswith(("tcp/", "tls/")):
        raise ValueError("endpoint must use tcp/<host>:<port> or tls/<host>:<port>")
    address = endpoint.split("/", 1)[1]
    host, separator, port_text = address.rpartition(":")
    if not separator or not host or any(character.isspace() for character in host):
        raise ValueError("endpoint must contain a non-empty host and port")
    try:
        port = int(port_text)
    except ValueError as error:
        raise ValueError("endpoint port must be an integer") from error
    if not 1 <= port <= 65535:
        raise ValueError("endpoint port must be in 1..65535")


def _validate_envelope(value: Any, *, request_id: str, session_id: str) -> None:
    if not isinstance(value, Mapping):
        raise RemoteObservationError("observation reply must be an object")
    expected = {
        "protocol_version": REMOTE_PROTOCOL_VERSION,
        "operation": OBSERVATION_OPERATION,
        "request_id": request_id,
        "session_id": session_id,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise RemoteObservationError(
                f"reply {key} must be {expected_value!r}, got {value.get(key)!r}"
            )


def _validate_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RemoteObservationError("reply metadata must be an object")
    expected = {
        "protocol_version": SEMANTIC_PROTOCOL_VERSION,
        "observation_schema": "policy-infer-input",
        "observation_fields": OBSERVATION_FIELD_METADATA,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise RemoteObservationError(
                f"metadata[{key!r}] must be {expected_value!r}, got {value.get(key)!r}"
            )
    names = value.get("joint_names")
    if not isinstance(names, (list, tuple)) or tuple(names) != JOINT_NAMES:
        raise RemoteObservationError(
            "metadata['joint_names'] must match the 65-joint protocol order"
        )
    return dict(value)


def _validate_timestamp(value: Any, *, now: float, max_age_s: float) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise RemoteObservationError("observation_timestamp must be a finite Unix timestamp")
    timestamp = float(value)
    age = now - timestamp
    if age < -max_age_s:
        raise RemoteObservationError(
            f"observation_timestamp is too far in the future: offset={-age:.3f}s"
        )
    if age > max_age_s:
        raise RemoteObservationError(f"observation is stale: age={age:.3f}s")
    return timestamp


def _validate_observation(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RemoteObservationError("reply observation must be an object")
    required_keys = {*REQUIRED_IMAGE_SPECS, *VECTOR_SPECS, "prompt"}
    allowed_keys = required_keys | set(OPTIONAL_IMAGE_SPECS)
    if not required_keys.issubset(value) or not set(value).issubset(allowed_keys):
        missing = sorted(required_keys - set(value))
        extra = sorted(set(value) - allowed_keys)
        raise RemoteObservationError(
            f"observation keys mismatch; missing={missing}, extra={extra}"
        )
    result: dict[str, Any] = {}
    for key, shape in {**REQUIRED_IMAGE_SPECS, **OPTIONAL_IMAGE_SPECS}.items():
        if key not in value:
            continue
        image = value[key]
        if not isinstance(image, np.ndarray):
            raise RemoteObservationError(f"{key} must decode to a NumPy array")
        if image.dtype != np.dtype(np.uint8) or image.shape != shape:
            raise RemoteObservationError(
                f"{key} must be uint8{list(shape)}, got {image.dtype}{list(image.shape)}"
            )
        result[key] = np.ascontiguousarray(image)
    for key, shape in VECTOR_SPECS.items():
        vector = value[key]
        if not isinstance(vector, np.ndarray):
            raise RemoteObservationError(f"{key} must decode to a NumPy array")
        if vector.dtype != np.dtype(np.float32) or vector.shape != shape:
            raise RemoteObservationError(
                f"{key} must be float32{list(shape)}, "
                f"got {vector.dtype}{list(vector.shape)}"
            )
        if not np.isfinite(vector).all():
            raise RemoteObservationError(f"{key} contains NaN or Inf")
        result[key] = np.ascontiguousarray(vector)
    prompt = value["prompt"]
    if not isinstance(prompt, str):
        raise RemoteObservationError("prompt must be a string")
    result["prompt"] = prompt
    return result


class _ZenohTransport:
    def __init__(
        self,
        endpoint: str,
        *,
        session: Any | None,
        tls_root_ca_certificate: str | None,
        tls_client_certificate: str | None,
        tls_client_private_key: str | None,
    ) -> None:
        self._owns_session = session is None
        self._zenoh = None
        if session is None:
            try:
                import zenoh
            except ImportError as error:
                raise RuntimeError("eclipse-zenoh 1.9 is required") from error
            config = zenoh.Config()
            config.insert_json5("mode", json.dumps("client"))
            config.insert_json5("connect/endpoints", json.dumps([endpoint]))
            config.insert_json5("scouting/multicast/enabled", "false")
            config.insert_json5("transport/shared_memory/enabled", "false")
            if endpoint.startswith("tls/"):
                tls = {"root_ca_certificate": tls_root_ca_certificate}
                if tls_client_certificate is not None:
                    tls.update(
                        {
                            "enable_mtls": True,
                            "connect_certificate": tls_client_certificate,
                            "connect_private_key": tls_client_private_key,
                        }
                    )
                config.insert_json5("transport/link/tls", json.dumps(tls))
            session = zenoh.open(config)
            self._zenoh = zenoh
        self._session = session

    def query(self, *, key: str, payload: bytes, timeout_s: float) -> list[bytes]:
        options: dict[str, Any] = {"payload": payload, "timeout": timeout_s}
        if self._zenoh is not None:
            options["consolidation"] = self._zenoh.ConsolidationMode.NONE
        replies = self._session.get(key, **options)
        return [_reply_payload(reply) for reply in replies]

    def close(self) -> None:
        if self._owns_session:
            self._session.close()


def _materialize_payloads(replies: Any) -> list[bytes]:
    if isinstance(replies, (bytes, bytearray, memoryview)):
        return [bytes(replies)]
    if not isinstance(replies, Iterable):
        raise TypeError("transport.query() must return a payload or iterable")
    return [_reply_payload(reply) for reply in replies]


def _reply_payload(reply: Any) -> bytes:
    if isinstance(reply, (bytes, bytearray, memoryview)):
        return bytes(reply)
    error = getattr(reply, "err", None)
    if error is not None:
        raise RemoteObservationError("Zenoh returned an error reply")
    sample = getattr(reply, "ok", None)
    if sample is None:
        raise RemoteObservationError("Zenoh reply contains no sample")
    payload = getattr(sample, "payload", sample)
    to_bytes = getattr(payload, "to_bytes", None)
    if to_bytes is None:
        raise RemoteObservationError("Zenoh reply payload is not bytes")
    return bytes(to_bytes())


def _pack_numpy(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"O", "V", "c"}:
            raise ValueError(f"unsupported NumPy dtype: {value.dtype}")
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
    raise TypeError(f"cannot encode {type(value).__name__}")


def _unpack_numpy(value: dict[Any, Any]) -> Any:
    def field(name: str) -> Any:
        return value[name] if name in value else value.get(name.encode("ascii"))

    if field("__ndarray__") is True:
        data = field("data")
        shape = field("shape")
        try:
            dtype = np.dtype(field("dtype"))
        except (TypeError, ValueError) as error:
            raise RemoteObservationError("invalid NumPy dtype") from error
        if dtype.kind in {"O", "V", "c"} or dtype.hasobject:
            raise RemoteObservationError(f"unsafe NumPy dtype: {dtype}")
        if not isinstance(data, bytes) or not isinstance(shape, (list, tuple)):
            raise RemoteObservationError("invalid NumPy array payload")
        if any(type(dimension) is not int or dimension < 0 for dimension in shape):
            raise RemoteObservationError("invalid NumPy array shape")
        normalized_shape = tuple(shape)
        expected_size = math.prod(normalized_shape) * dtype.itemsize
        if expected_size > MAX_PAYLOAD_BYTES or len(data) != expected_size:
            raise RemoteObservationError("NumPy array payload size mismatch")
        return np.frombuffer(data, dtype=dtype).reshape(normalized_shape)
    if field("__npgeneric__") is True:
        return np.dtype(field("dtype")).type(field("data"))
    return value


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("ORIGAMI_REMOTE_ENDPOINT"),
        help="assigned tcp/... or tls/... endpoint (ORIGAMI_REMOTE_ENDPOINT)",
    )
    parser.add_argument(
        "--session-id",
        default=os.environ.get("ORIGAMI_REMOTE_SESSION_ID"),
        help="assigned session ID (ORIGAMI_REMOTE_SESSION_ID)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("ORIGAMI_REMOTE_TOKEN"),
        help="per-team access token (ORIGAMI_REMOTE_TOKEN)",
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--max-observation-age", type=float, default=5.0)
    parser.add_argument(
        "--tls-root-ca-certificate",
        default=os.environ.get("ORIGAMI_REMOTE_TLS_CA"),
    )
    parser.add_argument(
        "--tls-client-certificate",
        default=os.environ.get("ORIGAMI_REMOTE_TLS_CERT"),
    )
    parser.add_argument(
        "--tls-client-private-key",
        default=os.environ.get("ORIGAMI_REMOTE_TLS_KEY"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if not args.endpoint:
        parser.error("--endpoint or ORIGAMI_REMOTE_ENDPOINT is required")
    if not args.session_id:
        parser.error("--session-id or ORIGAMI_REMOTE_SESSION_ID is required")
    if not args.token:
        parser.error("--token or ORIGAMI_REMOTE_TOKEN is required")
    try:
        with RemoteObservationClient(
            args.endpoint,
            session_id=args.session_id,
            token=args.token,
            timeout_s=args.timeout,
            max_observation_age_s=args.max_observation_age,
            tls_root_ca_certificate=args.tls_root_ca_certificate,
            tls_client_certificate=args.tls_client_certificate,
            tls_client_private_key=args.tls_client_private_key,
        ) as client:
            observation = client.get_observation()
    except (RemoteObservationError, RuntimeError, TimeoutError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(
        "PASS: received policy-compatible observation "
        f"state={observation['observation/state'].shape} "
        f"cameras={[observation[key].shape for key in CAMERA_IMAGE_KEYS]} "
        f"tactile_deform={observation['observation/image/tactile_deform'].shape} "
        f"tactile_raw={getattr(observation.get('observation/image/tactile_raw'), 'shape', None)} "
        f"timestamp={client.last_observation_timestamp:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
