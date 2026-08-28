#!/usr/bin/env python3
"""Black-box validator for the public origami-zenoh-v1 policy protocol.

The validator opens a Zenoh client session, sends deterministic synthetic
observations, and validates metadata, reset, and inference replies. It has no
North SDK dependency, reads no robot data, and publishes no actions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import uuid
from collections.abc import Mapping
from typing import Any

import msgpack
import numpy as np


ZENOH_PROTOCOL_VERSION = "origami-zenoh-v1"
SEMANTIC_PROTOCOL_VERSION = "origami-v1"
METADATA_KEY = f"{ZENOH_PROTOCOL_VERSION}/metadata"
RESET_KEY = f"{ZENOH_PROTOCOL_VERSION}/reset"
INFER_KEY = f"{ZENOH_PROTOCOL_VERSION}/infer"
OPERATION_KEYS = {
    "metadata": METADATA_KEY,
    "reset": RESET_KEY,
    "infer": INFER_KEY,
}
ACTION_DIM = 65
IMAGE_SHAPE = (224, 224, 3)
TACTILE_DEFORM_SHAPE = (480, 1200, 3)
TACTILE_RAW_SHAPE = (480, 1600, 3)
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024

LEFT_ARM_JOINT_NAMES = tuple(f"left_arm_joint_{index}" for index in range(1, 8))
RIGHT_ARM_JOINT_NAMES = tuple(f"right_arm_joint_{index}" for index in range(1, 8))


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


LEFT_HAND_JOINT_NAMES = _hand_joint_names("left")
RIGHT_HAND_JOINT_NAMES = _hand_joint_names("right")
MOTOR_JOINT_NAMES = (
    "lower_body_joint_1",
    "lower_body_joint_2",
    "lower_body_joint_3",
    "lower_body_joint_4",
    "lower_body_joint_5",
    "neck_joint_1",
    "neck_joint_2",
)
JOINT_NAMES = (
    LEFT_ARM_JOINT_NAMES
    + LEFT_HAND_JOINT_NAMES
    + RIGHT_ARM_JOINT_NAMES
    + RIGHT_HAND_JOINT_NAMES
    + MOTOR_JOINT_NAMES
)

if len(JOINT_NAMES) != ACTION_DIM or len(set(JOINT_NAMES)) != ACTION_DIM:
    raise RuntimeError("joint contract must contain 65 unique joint names")


class ValidationError(RuntimeError):
    """A public protocol contract violation."""


def _mapping_value(mapping: dict[Any, Any], key: str) -> Any:
    if key in mapping:
        return mapping[key]
    return mapping.get(key.encode("ascii"))


def _pack_array(value: Any) -> Any:
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"unsupported NumPy dtype: {value.dtype}")
    if isinstance(value, np.ndarray):
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
    raise TypeError(f"cannot MessagePack-encode {type(value).__name__}")


def _parse_dtype(raw_dtype: Any) -> np.dtype:
    if isinstance(raw_dtype, bytes):
        raw_dtype = raw_dtype.decode("ascii")
    if not isinstance(raw_dtype, str):
        raise ValidationError("NumPy dtype descriptor must be a string")
    try:
        dtype = np.dtype(raw_dtype)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"invalid NumPy dtype descriptor: {raw_dtype!r}") from exc
    if dtype.kind in ("V", "O", "c") or dtype.hasobject:
        raise ValidationError(f"unsafe/unsupported NumPy dtype: {dtype}")
    return dtype


def _unpack_array(value: dict[Any, Any]) -> Any:
    if _mapping_value(value, "__ndarray__") is True:
        data = _mapping_value(value, "data")
        shape = _mapping_value(value, "shape")
        dtype = _parse_dtype(_mapping_value(value, "dtype"))
        if not isinstance(data, bytes):
            raise ValidationError("NumPy array data must be MessagePack binary")
        if not isinstance(shape, (list, tuple)) or len(shape) > 8:
            raise ValidationError("NumPy array shape must contain at most 8 dimensions")
        if any(not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in shape):
            raise ValidationError(f"invalid NumPy array shape: {shape!r}")
        item_count = math.prod(shape)
        expected_bytes = item_count * dtype.itemsize
        if expected_bytes > MAX_PAYLOAD_BYTES:
            raise ValidationError("decoded NumPy array exceeds validator size limit")
        if len(data) != expected_bytes:
            raise ValidationError(
                f"NumPy array byte length mismatch: expected {expected_bytes}, got {len(data)}"
            )
        return np.frombuffer(data, dtype=dtype).reshape(tuple(shape))

    if _mapping_value(value, "__npgeneric__") is True:
        dtype = _parse_dtype(_mapping_value(value, "dtype"))
        data = _mapping_value(value, "data")
        try:
            return dtype.type(data)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValidationError(f"invalid NumPy scalar for dtype {dtype}") from exc

    return value


def pack_payload(value: Any) -> bytes:
    """Encode a protocol payload using the safe repository msgpack-numpy codec."""
    packed = msgpack.packb(value, default=_pack_array, use_bin_type=True)
    if len(packed) > MAX_PAYLOAD_BYTES:
        raise ValidationError("encoded payload exceeds validator size limit")
    return packed


def unpack_payload(payload: bytes) -> Any:
    """Decode a bounded protocol payload without pickle/object arrays."""
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValidationError("reply payload exceeds validator size limit")
    try:
        return msgpack.unpackb(
            payload,
            object_hook=_unpack_array,
            raw=False,
            strict_map_key=False,
            max_bin_len=MAX_PAYLOAD_BYTES,
            max_array_len=1_000_000,
            max_map_len=10_000,
            max_str_len=1_000_000,
        )
    except ValidationError:
        raise
    except (msgpack.UnpackException, ValueError, TypeError) as exc:
        raise ValidationError(f"invalid MessagePack reply: {exc}") from exc


def validate_endpoint(endpoint: str) -> str:
    """Validate tcp/<IP-or-DNS-hostname>:<port> without resolving it."""
    if not isinstance(endpoint, str) or not endpoint.startswith("tcp/"):
        raise ValidationError("endpoint must use tcp/<host>:<port>")
    address = endpoint[4:]
    if address.startswith("["):
        closing = address.find("]")
        if closing < 0 or closing + 1 >= len(address) or address[closing + 1] != ":":
            raise ValidationError("invalid bracketed IPv6 endpoint")
        host, port_text = address[1:closing], address[closing + 2 :]
    else:
        host, separator, port_text = address.rpartition(":")
        if not separator:
            raise ValidationError("endpoint must include a TCP port")
    if not host or any(character.isspace() for character in host) or "/" in host:
        raise ValidationError("endpoint host must be a non-empty IP address or DNS hostname")
    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValidationError("endpoint port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise ValidationError("endpoint port must be in 1..65535")
    return endpoint


def validate_session_id(session_id: str) -> str:
    if not isinstance(session_id, str) or not session_id:
        raise ValidationError("session ID must be a non-empty string")
    return session_id


def make_synthetic_observation() -> dict[str, Any]:
    """Build deterministic protocol-correct data without reading a robot."""
    rows = np.arange(IMAGE_SHAPE[0], dtype=np.uint8)[:, None]
    cols = np.arange(IMAGE_SHAPE[1], dtype=np.uint8)[None, :]
    base = np.empty(IMAGE_SHAPE, dtype=np.uint8)
    base[..., 0] = rows
    base[..., 1] = cols
    base[..., 2] = rows ^ cols
    return {
        "observation/image/head_left": np.ascontiguousarray(base),
        "observation/image/head_right": np.ascontiguousarray(np.roll(base, 11, axis=0)),
        "observation/image/wrist_left": np.ascontiguousarray(np.roll(base, 17, axis=1)),
        "observation/image/wrist_right": np.ascontiguousarray(np.flip(base, axis=1)),
        "observation/state": np.linspace(-0.25, 0.25, ACTION_DIM, dtype=np.float32),
        "observation/state/joint_torque": np.zeros(ACTION_DIM, dtype=np.float32),
        "observation/tactile": np.zeros(60, dtype=np.float32),
        "observation/image/tactile_deform": np.zeros(
            TACTILE_DEFORM_SHAPE, dtype=np.uint8
        ),
        "observation/image/tactile_raw": np.zeros(TACTILE_RAW_SHAPE, dtype=np.uint8),
        "prompt": "origami synthetic protocol check",
    }


def _check_error_envelope(reply: dict[str, Any], operation: str) -> None:
    if "error" not in reply:
        return
    error = reply["error"]
    if isinstance(error, dict):
        code = error.get("code", "UNKNOWN")
        message = error.get("message", "")
        raise ValidationError(f"{operation} returned {code}: {message}")
    raise ValidationError(f"{operation} returned malformed error envelope")


def validate_reply_envelope(
    reply: Any,
    *,
    operation: str,
    request_id: str,
    session_id: str,
) -> dict[str, Any]:
    if not isinstance(reply, dict):
        raise ValidationError(f"reply for {operation!r} must be a map")
    expected = {
        "protocol_version": ZENOH_PROTOCOL_VERSION,
        "operation": operation,
        "request_id": request_id,
        "session_id": session_id,
    }
    for key, value in expected.items():
        if reply.get(key) != value:
            raise ValidationError(
                f"{operation} reply {key!r} must echo {value!r}, got {reply.get(key)!r}"
            )
    _check_error_envelope(reply, operation)
    return reply


def validate_metadata(reply: Any, expected_horizon: int | None = None) -> dict[str, Any]:
    if not isinstance(reply, dict):
        raise ValidationError(f"metadata reply must be a map, got {type(reply).__name__}")
    metadata = reply.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValidationError("metadata reply must contain a metadata object")
    expected = {
        "protocol_version": (str, SEMANTIC_PROTOCOL_VERSION),
        "action_dim": (int, ACTION_DIM),
        "action_type": (str, "absolute_joint_position"),
        "action_units": (str, "radians"),
    }
    for key, (value_type, expected_value) in expected.items():
        actual = metadata.get(key)
        if type(actual) is not value_type or actual != expected_value:
            raise ValidationError(
                f"metadata[{key!r}] must be {expected_value!r}, got {actual!r}"
            )
    horizon = metadata.get("action_horizon")
    if type(horizon) is not int or not 1 <= horizon <= 1024:
        raise ValidationError("metadata['action_horizon'] must be an integer in [1,1024]")
    joint_names = metadata.get("joint_names")
    if not isinstance(joint_names, (list, tuple)) or tuple(joint_names) != JOINT_NAMES:
        raise ValidationError("metadata['joint_names'] must match the 65-joint protocol order")
    if expected_horizon is not None and horizon != expected_horizon:
        raise ValidationError(
            f"metadata action_horizon must be {expected_horizon}, got {horizon}"
        )
    return dict(metadata)


def validate_reset(reply: Any) -> None:
    if not isinstance(reply, dict):
        raise ValidationError(f"reset reply must be a map, got {type(reply).__name__}")
    _check_error_envelope(reply, "reset")
    if reply.get("ok") is not True:
        raise ValidationError(f"reset reply must contain {{'ok': True}}, got {reply!r}")


def validate_infer(reply: Any, horizon: int) -> np.ndarray:
    if not isinstance(reply, dict):
        raise ValidationError(f"infer reply must be a map, got {type(reply).__name__}")
    _check_error_envelope(reply, "infer")
    if "actions" not in reply:
        raise ValidationError("infer reply is missing 'actions'")
    actions = reply["actions"]
    if not isinstance(actions, np.ndarray):
        raise ValidationError(f"actions must decode to ndarray, got {type(actions).__name__}")
    if actions.dtype != np.dtype(np.float32):
        raise ValidationError(f"actions dtype must be float32, got {actions.dtype}")
    if actions.shape != (horizon, ACTION_DIM):
        raise ValidationError(
            f"actions shape must be ({horizon}, {ACTION_DIM}), got {actions.shape}"
        )
    if not np.isfinite(actions).all():
        raise ValidationError("actions contain NaN or Inf")
    return actions


def open_zenoh_session(endpoint: str) -> Any:
    """Open a direct client-only Zenoh session (imported lazily for unit tests)."""
    try:
        import zenoh
    except ImportError as exc:
        raise ValidationError(
            "eclipse-zenoh is required; install the SDK project dependencies"
        ) from exc

    config = zenoh.Config()
    config.insert_json5("mode", json.dumps("client"))
    config.insert_json5("connect/endpoints", json.dumps([endpoint]))
    config.insert_json5("scouting/multicast/enabled", "false")
    config.insert_json5("transport/shared_memory/enabled", "false")
    zenoh.init_log_from_env_or("error")
    return zenoh.open(config)


def query_once(
    session: Any,
    operation: str,
    session_id: str,
    timeout: float,
    **body: Any,
) -> dict[str, Any]:
    """Send one enveloped query and validate its single correlated reply."""
    import zenoh

    request_id = uuid.uuid4().hex
    request = {
        "protocol_version": ZENOH_PROTOCOL_VERSION,
        "operation": operation,
        "request_id": request_id,
        "session_id": session_id,
        **body,
    }
    try:
        key = OPERATION_KEYS[operation]
    except KeyError as exc:
        raise ValidationError(f"unsupported operation: {operation!r}") from exc
    replies = session.get(
        key,
        payload=pack_payload(request),
        timeout=timeout,
        consolidation=zenoh.ConsolidationMode.NONE,
    )
    successful: list[Any] = []
    transport_errors: list[str] = []
    for reply in replies:
        sample = reply.ok
        if sample is not None:
            successful.append(unpack_payload(sample.payload.to_bytes()))
            continue
        error = reply.err
        if error is not None:
            try:
                detail = error.payload.to_string()
            except Exception:
                detail = "<non-text Zenoh error>"
            transport_errors.append(detail)
            continue
        raise ValidationError(f"{key} reply contains neither a sample nor an error")

    if transport_errors:
        raise ValidationError(f"{key} returned Zenoh error: {'; '.join(transport_errors)}")
    if len(successful) != 1:
        raise ValidationError(f"{key} must return exactly one reply, got {len(successful)}")
    return validate_reply_envelope(
        successful[0],
        operation=operation,
        request_id=request_id,
        session_id=session_id,
    )


def run_validation(
    endpoint: str,
    session_id: str,
    timeout: float,
    requests: int,
    expected_horizon: int | None,
) -> None:
    endpoint = validate_endpoint(endpoint)
    session_id = validate_session_id(session_id)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValidationError("timeout must be a finite number > 0")
    if requests < 1:
        raise ValidationError("requests must be >= 1")
    if expected_horizon is not None and not 1 <= expected_horizon <= 1024:
        raise ValidationError("expected horizon must be in [1,1024]")

    session = open_zenoh_session(endpoint)
    try:
        metadata = validate_metadata(
            query_once(session, "metadata", session_id, timeout),
            expected_horizon,
        )
        horizon = metadata["action_horizon"]
        print(
            "metadata: PASS "
            f"(transport={ZENOH_PROTOCOL_VERSION}, semantic={SEMANTIC_PROTOCOL_VERSION}, "
            f"horizon={horizon}, dim={ACTION_DIM})"
        )

        validate_reset(query_once(session, "reset", session_id, timeout))
        print("reset: PASS")

        observation = make_synthetic_observation()
        latencies_ms: list[float] = []
        for index in range(requests):
            request_observation = dict(observation)
            if index == requests - 1:
                request_observation.pop("observation/image/tactile_raw")
            started = time.monotonic()
            reply = query_once(
                session,
                "infer",
                session_id,
                timeout,
                observation=request_observation,
            )
            latency_ms = (time.monotonic() - started) * 1000.0
            validate_infer(reply, horizon)
            latencies_ms.append(latency_ms)
            raw_mode = "without optional tactile_raw" if index == requests - 1 else "full"
            print(
                f"infer {index + 1}/{requests}: PASS "
                f"({latency_ms:.1f} ms, {raw_mode})"
            )
    finally:
        session.close()

    print(f"PASS: policy is compatible with {ZENOH_PROTOCOL_VERSION}")
    print(
        f"latency: requests={requests} "
        f"median={statistics.median(latencies_ms):.1f} ms max={max(latencies_ms):.1f} ms"
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("ORIGAMI_ZENOH_ENDPOINT"),
        help="Zenoh router endpoint tcp/<IP-or-DNS-host>:<port> (or ORIGAMI_ZENOH_ENDPOINT)",
    )
    parser.add_argument(
        "--session-id",
        default=os.environ.get("ORIGAMI_SESSION_ID"),
        help="assigned session ID (or ORIGAMI_SESSION_ID)",
    )
    parser.add_argument("--timeout", type=float, default=10.0, help="per-query timeout in seconds")
    parser.add_argument("--requests", type=int, default=3, help="number of infer queries")
    parser.add_argument(
        "--expected-horizon",
        "--expected-action-horizon",
        dest="expected_horizon",
        type=int,
        default=None,
        help="require this exact metadata/action horizon",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if not args.endpoint:
        parser.error("--endpoint or ORIGAMI_ZENOH_ENDPOINT is required")
    if not args.session_id:
        parser.error("--session-id or ORIGAMI_SESSION_ID is required")
    try:
        run_validation(
            endpoint=args.endpoint,
            session_id=args.session_id,
            timeout=args.timeout,
            requests=args.requests,
            expected_horizon=args.expected_horizon,
        )
    except (ValidationError, TimeoutError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
