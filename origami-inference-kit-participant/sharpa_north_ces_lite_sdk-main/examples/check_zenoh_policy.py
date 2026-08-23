#!/usr/bin/env python3

"""
Black-box validator for the public origami-zenoh-v1 policy protocol.

The validator can run synthetic and/or real LeRobot dataset observations.

For dataset validation it saves:

    episode_<episode>_queries_<N>.npz
    episode_<episode>_queries_<N>.json

The NPZ contains:
    predicted_actions
    gt_actions

The JSON contains the complete validation record, including:
    - season name
    - episode index
    - number of queries
    - action horizon / dimension
    - dataset information
    - frame indices
    - dataset row indices
    - timestamps in seconds
    - inference times
    - per-query MSE
    - full-run MSE
    - policy metadata
    - latency statistics
    - printed log lines
    - output file names
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

from datetime import datetime
from pathlib import Path
# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

ZENOH_PROTOCOL_VERSION = "origami-zenoh-v1"
SEMANTIC_PROTOCOL_VERSION = "origami-v1"

METADATA_KEY = (
    f"{ZENOH_PROTOCOL_VERSION}/metadata"
)

RESET_KEY = (
    f"{ZENOH_PROTOCOL_VERSION}/reset"
)

INFER_KEY = (
    f"{ZENOH_PROTOCOL_VERSION}/infer"
)

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


# ---------------------------------------------------------------------------
# Joint names
# ---------------------------------------------------------------------------

LEFT_ARM_JOINT_NAMES = tuple(
    f"left_arm_joint_{index}"
    for index in range(1, 8)
)

RIGHT_ARM_JOINT_NAMES = tuple(
    f"right_arm_joint_{index}"
    for index in range(1, 8)
)


def _hand_joint_names(
    side: str,
) -> tuple[str, ...]:
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


LEFT_HAND_JOINT_NAMES = _hand_joint_names(
    "left"
)

RIGHT_HAND_JOINT_NAMES = _hand_joint_names(
    "right"
)

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

if (
    len(JOINT_NAMES) != ACTION_DIM
    or len(set(JOINT_NAMES)) != ACTION_DIM
):
    raise RuntimeError(
        "joint contract must contain 65 unique joint names"
    )


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ValidationError(RuntimeError):
    """A public protocol contract violation."""


# ---------------------------------------------------------------------------
# MessagePack helpers
# ---------------------------------------------------------------------------

def _mapping_value(
    mapping: dict[Any, Any],
    key: str,
) -> Any:
    if key in mapping:
        return mapping[key]

    return mapping.get(
        key.encode("ascii")
    )


def _pack_array(value: Any) -> Any:
    if (
        isinstance(value, (np.ndarray, np.generic))
        and value.dtype.kind in ("V", "O", "c")
    ):
        raise ValueError(
            f"unsupported NumPy dtype: {value.dtype}"
        )

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

    raise TypeError(
        f"cannot MessagePack-encode "
        f"{type(value).__name__}"
    )


def _parse_dtype(raw_dtype: Any) -> np.dtype:
    if isinstance(raw_dtype, bytes):
        raw_dtype = raw_dtype.decode(
            "ascii"
        )

    if not isinstance(raw_dtype, str):
        raise ValidationError(
            "NumPy dtype descriptor must be a string"
        )

    try:
        dtype = np.dtype(raw_dtype)
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValidationError(
            f"invalid NumPy dtype descriptor: "
            f"{raw_dtype!r}"
        ) from exc

    if (
        dtype.kind in ("V", "O", "c")
        or dtype.hasobject
    ):
        raise ValidationError(
            f"unsafe/unsupported NumPy dtype: {dtype}"
        )

    return dtype


def _unpack_array(
    value: dict[Any, Any],
) -> Any:
    if _mapping_value(
        value,
        "__ndarray__",
    ) is True:
        data = _mapping_value(
            value,
            "data",
        )

        shape = _mapping_value(
            value,
            "shape",
        )

        dtype = _parse_dtype(
            _mapping_value(
                value,
                "dtype",
            )
        )

        if not isinstance(data, bytes):
            raise ValidationError(
                "NumPy array data must be "
                "MessagePack binary"
            )

        if (
            not isinstance(
                shape,
                (list, tuple),
            )
            or len(shape) > 8
        ):
            raise ValidationError(
                "NumPy array shape must contain "
                "at most 8 dimensions"
            )

        if any(
            not isinstance(dim, int)
            or isinstance(dim, bool)
            or dim < 0
            for dim in shape
        ):
            raise ValidationError(
                f"invalid NumPy array shape: {shape!r}"
            )

        item_count = math.prod(shape)

        expected_bytes = (
            item_count * dtype.itemsize
        )

        if expected_bytes > MAX_PAYLOAD_BYTES:
            raise ValidationError(
                "decoded NumPy array exceeds "
                "validator size limit"
            )

        if len(data) != expected_bytes:
            raise ValidationError(
                f"NumPy array byte length mismatch: "
                f"expected {expected_bytes}, "
                f"got {len(data)}"
            )

        return np.frombuffer(
            data,
            dtype=dtype,
        ).reshape(tuple(shape))

    if _mapping_value(
        value,
        "__npgeneric__",
    ) is True:
        dtype = _parse_dtype(
            _mapping_value(
                value,
                "dtype",
            )
        )

        data = _mapping_value(
            value,
            "data",
        )

        try:
            return dtype.type(data)
        except (
            TypeError,
            ValueError,
            OverflowError,
        ) as exc:
            raise ValidationError(
                f"invalid NumPy scalar for "
                f"dtype {dtype}"
            ) from exc

    return value


def pack_payload(
    value: Any,
) -> bytes:
    packed = msgpack.packb(
        value,
        default=_pack_array,
        use_bin_type=True,
    )

    if len(packed) > MAX_PAYLOAD_BYTES:
        raise ValidationError(
            "encoded payload exceeds validator size limit"
        )

    return packed


def unpack_payload(
    payload: bytes,
) -> Any:
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValidationError(
            "reply payload exceeds validator size limit"
        )

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

    except (
        msgpack.UnpackException,
        ValueError,
        TypeError,
    ) as exc:
        raise ValidationError(
            f"invalid MessagePack reply: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_endpoint(
    endpoint: str,
) -> str:
    if (
        not isinstance(endpoint, str)
        or not endpoint.startswith("tcp/")
    ):
        raise ValidationError(
            "endpoint must use tcp/<host>:<port>"
        )

    address = endpoint[4:]

    if address.startswith("["):
        closing = address.find("]")

        if (
            closing < 0
            or closing + 1 >= len(address)
            or address[closing + 1] != ":"
        ):
            raise ValidationError(
                "invalid bracketed IPv6 endpoint"
            )

        host = address[1:closing]
        port_text = address[
            closing + 2:
        ]

    else:
        host, separator, port_text = (
            address.rpartition(":")
        )

        if not separator:
            raise ValidationError(
                "endpoint must include a TCP port"
            )

    if (
        not host
        or any(
            character.isspace()
            for character in host
        )
        or "/" in host
    ):
        raise ValidationError(
            "endpoint host must be a non-empty "
            "IP address or DNS hostname"
        )

    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValidationError(
            "endpoint port must be an integer"
        ) from exc

    if not 1 <= port <= 65535:
        raise ValidationError(
            "endpoint port must be in 1..65535"
        )

    return endpoint


def validate_session_id(
    session_id: str,
) -> str:
    if (
        not isinstance(session_id, str)
        or not session_id
    ):
        raise ValidationError(
            "session ID must be a non-empty string"
        )

    return session_id


def make_synthetic_observation() -> dict[str, Any]:
    """Build deterministic protocol-correct data."""

    rows = np.arange(
        IMAGE_SHAPE[0],
        dtype=np.uint8,
    )[:, None]

    cols = np.arange(
        IMAGE_SHAPE[1],
        dtype=np.uint8,
    )[None, :]

    base = np.empty(
        IMAGE_SHAPE,
        dtype=np.uint8,
    )

    base[..., 0] = rows
    base[..., 1] = cols
    base[..., 2] = rows ^ cols

    return {
        "observation/image/head_left":
            np.ascontiguousarray(base),

        "observation/image/head_right":
            np.ascontiguousarray(
                np.roll(
                    base,
                    11,
                    axis=0,
                )
            ),

        "observation/image/wrist_left":
            np.ascontiguousarray(
                np.roll(
                    base,
                    17,
                    axis=1,
                )
            ),

        "observation/image/wrist_right":
            np.ascontiguousarray(
                np.flip(
                    base,
                    axis=1,
                )
            ),

        "observation/state":
            np.linspace(
                -0.25,
                0.25,
                ACTION_DIM,
                dtype=np.float32,
            ),

        "observation/state/joint_torque":
            np.zeros(
                ACTION_DIM,
                dtype=np.float32,
            ),

        "observation/tactile":
            np.zeros(
                60,
                dtype=np.float32,
            ),

        "observation/image/tactile_deform":
            np.zeros(
                TACTILE_DEFORM_SHAPE,
                dtype=np.uint8,
            ),

        "observation/image/tactile_raw":
            np.zeros(
                TACTILE_RAW_SHAPE,
                dtype=np.uint8,
            ),

        "prompt":
            "origami synthetic protocol check",
    }


def _check_error_envelope(
    reply: dict[str, Any],
    operation: str,
) -> None:
    if "error" not in reply:
        return

    error = reply["error"]

    if isinstance(error, dict):
        code = error.get(
            "code",
            "UNKNOWN",
        )

        message = error.get(
            "message",
            "",
        )

        raise ValidationError(
            f"{operation} returned "
            f"{code}: {message}"
        )

    raise ValidationError(
        f"{operation} returned malformed "
        "error envelope"
    )


def validate_reply_envelope(
    reply: Any,
    *,
    operation: str,
    request_id: str,
    session_id: str,
) -> dict[str, Any]:
    if not isinstance(reply, dict):
        raise ValidationError(
            f"reply for {operation!r} must be a map"
        )

    expected = {
        "protocol_version":
            ZENOH_PROTOCOL_VERSION,

        "operation":
            operation,

        "request_id":
            request_id,

        "session_id":
            session_id,
    }

    for key, value in expected.items():
        if reply.get(key) != value:
            raise ValidationError(
                f"{operation} reply {key!r} "
                f"must echo {value!r}, "
                f"got {reply.get(key)!r}"
            )

    _check_error_envelope(
        reply,
        operation,
    )

    return reply


def validate_metadata(
    reply: Any,
    expected_horizon: int | None = None,
) -> dict[str, Any]:
    if not isinstance(reply, dict):
        raise ValidationError(
            f"metadata reply must be a map, "
            f"got {type(reply).__name__}"
        )

    metadata = reply.get(
        "metadata"
    )

    if not isinstance(
        metadata,
        Mapping,
    ):
        raise ValidationError(
            "metadata reply must contain "
            "a metadata object"
        )

    expected = {
        "protocol_version":
            (str, SEMANTIC_PROTOCOL_VERSION),

        "action_dim":
            (int, ACTION_DIM),

        "action_type":
            (str, "absolute_joint_position"),

        "action_units":
            (str, "radians"),
    }

    for key, (
        value_type,
        expected_value,
    ) in expected.items():

        actual = metadata.get(key)

        if (
            type(actual) is not value_type
            or actual != expected_value
        ):
            raise ValidationError(
                f"metadata[{key!r}] must be "
                f"{expected_value!r}, "
                f"got {actual!r}"
            )

    horizon = metadata.get(
        "action_horizon"
    )

    if (
        type(horizon) is not int
        or not 1 <= horizon <= 1024
    ):
        raise ValidationError(
            "metadata['action_horizon'] "
            "must be an integer in [1,1024]"
        )

    joint_names = metadata.get(
        "joint_names"
    )

    if (
        not isinstance(
            joint_names,
            (list, tuple),
        )
        or tuple(joint_names)
        != JOINT_NAMES
    ):
        raise ValidationError(
            "metadata['joint_names'] must match "
            "the 65-joint protocol order"
        )

    if (
        expected_horizon is not None
        and horizon != expected_horizon
    ):
        raise ValidationError(
            "metadata action_horizon must be "
            f"{expected_horizon}, "
            f"got {horizon}"
        )

    return dict(metadata)


def validate_reset(
    reply: Any,
) -> None:
    if not isinstance(reply, dict):
        raise ValidationError(
            f"reset reply must be a map, "
            f"got {type(reply).__name__}"
        )

    _check_error_envelope(
        reply,
        "reset",
    )

    if reply.get("ok") is not True:
        raise ValidationError(
            "reset reply must contain "
            "{'ok': True}, "
            f"got {reply!r}"
        )


def validate_infer(
    reply: Any,
    horizon: int,
) -> np.ndarray:
    if not isinstance(reply, dict):
        raise ValidationError(
            f"infer reply must be a map, "
            f"got {type(reply).__name__}"
        )

    _check_error_envelope(
        reply,
        "infer",
    )

    if "actions" not in reply:
        raise ValidationError(
            "infer reply is missing 'actions'"
        )

    actions = reply["actions"]

    if not isinstance(
        actions,
        np.ndarray,
    ):
        raise ValidationError(
            "actions must decode to ndarray, "
            f"got {type(actions).__name__}"
        )

    if actions.dtype != np.dtype(
        np.float32
    ):
        raise ValidationError(
            "actions dtype must be float32, "
            f"got {actions.dtype}"
        )

    if actions.shape != (
        horizon,
        ACTION_DIM,
    ):
        raise ValidationError(
            "actions shape must be "
            f"({horizon}, {ACTION_DIM}), "
            f"got {actions.shape}"
        )

    if not np.isfinite(
        actions
    ).all():
        raise ValidationError(
            "actions contain NaN or Inf"
        )

    return actions


# ---------------------------------------------------------------------------
# Zenoh
# ---------------------------------------------------------------------------

def open_zenoh_session(
    endpoint: str,
) -> Any:
    """Open a direct client-only Zenoh session."""

    try:
        import zenoh
    except ImportError as exc:
        raise ValidationError(
            "eclipse-zenoh is required; "
            "install the SDK project dependencies"
        ) from exc

    config = zenoh.Config()

    config.insert_json5(
        "mode",
        json.dumps("client"),
    )

    config.insert_json5(
        "connect/endpoints",
        json.dumps([endpoint]),
    )

    config.insert_json5(
        "scouting/multicast/enabled",
        "false",
    )

    config.insert_json5(
        "transport/shared_memory/enabled",
        "false",
    )

    zenoh.init_log_from_env_or(
        "error"
    )

    return zenoh.open(config)


def query_once(
    session: Any,
    operation: str,
    session_id: str,
    timeout: float,
    **body: Any,
) -> dict[str, Any]:
    """Send one enveloped Zenoh query."""

    import zenoh

    request_id = uuid.uuid4().hex

    request = {
        "protocol_version":
            ZENOH_PROTOCOL_VERSION,

        "operation":
            operation,

        "request_id":
            request_id,

        "session_id":
            session_id,

        **body,
    }

    try:
        key = OPERATION_KEYS[
            operation
        ]
    except KeyError as exc:
        raise ValidationError(
            f"unsupported operation: "
            f"{operation!r}"
        ) from exc

    replies = session.get(
        key,
        payload=pack_payload(
            request
        ),
        timeout=timeout,
        consolidation=(
            zenoh.ConsolidationMode.NONE
        ),
    )

    successful: list[Any] = []
    transport_errors: list[str] = []

    for reply in replies:
        sample = reply.ok

        if sample is not None:
            successful.append(
                unpack_payload(
                    sample.payload.to_bytes()
                )
            )
            continue

        error = reply.err

        if error is not None:
            try:
                detail = (
                    error.payload.to_string()
                )
            except Exception:
                detail = (
                    "<non-text Zenoh error>"
                )

            transport_errors.append(
                detail
            )
            continue

        raise ValidationError(
            f"{key} reply contains neither "
            "a sample nor an error"
        )

    if transport_errors:
        raise ValidationError(
            f"{key} returned Zenoh error: "
            f"{'; '.join(transport_errors)}"
        )

    if len(successful) != 1:
        raise ValidationError(
            f"{key} must return exactly "
            f"one reply, got {len(successful)}"
        )

    return validate_reply_envelope(
        successful[0],
        operation=operation,
        request_id=request_id,
        session_id=session_id,
    )


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _json_safe(
    value: Any,
) -> Any:
    """
    Convert common NumPy / Python values into JSON-safe values.

    Used for policy metadata and logging information.
    """
    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if isinstance(
        value,
        dict,
    ):
        return {
            str(k): _json_safe(v)
            for k, v in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            _json_safe(v)
            for v in value
        ]

    if isinstance(
        value,
        (str, int, float, bool),
    ) or value is None:
        return value

    return str(value)


def _mse(
    predicted: np.ndarray,
    target: np.ndarray,
) -> float:
    """
    MSE after flattening all dimensions.

    For one query:
        (horizon, 65) -> (horizon * 65,)

    For the complete run:
        (T, horizon, 65) -> (T * horizon * 65,)
    """
    predicted_flat = np.asarray(
        predicted
    ).reshape(-1)

    target_flat = np.asarray(
        target
    ).reshape(-1)

    if predicted_flat.shape != target_flat.shape:
        raise ValueError(
            "Cannot calculate MSE: flattened "
            f"shapes differ: "
            f"{predicted_flat.shape} vs "
            f"{target_flat.shape}"
        )

    return float(
        np.mean(
            (
                predicted_flat
                - target_flat
            ) ** 2
        )
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def run_validation(
    endpoint: str,
    session_id: str,
    timeout: float,
    synth_requests: int,
    expected_horizon: int | None,
    dataset_requests: int,
    obs_type: str,
    dataset_root: str | None,
    episode_index: int,
    frame_stride: int,
    out_dir: str | None,
) -> None:

    endpoint = validate_endpoint(
        endpoint
    )

    session_id = validate_session_id(
        session_id
    )

    if (
        not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValidationError(
            "timeout must be a finite number > 0"
        )

    if synth_requests < 1:
        raise ValidationError(
            "requests must be >= 1"
        )

    if (
        expected_horizon is not None
        and not 1 <= expected_horizon <= 1024
    ):
        raise ValidationError(
            "expected horizon must be in [1,1024]"
        )

    if obs_type not in (
        "synthetic",
        "dataset",
        "both",
    ):
        raise ValidationError(
            "obs_type must be one of "
            "'synthetic', 'dataset', 'both'"
        )

    run_synthetic = (
        obs_type
        in ("synthetic", "both")
    )

    run_dataset = (
        obs_type
        in ("dataset", "both")
    )

    # -----------------------------------------------------------------------
    # Run logging
    # -----------------------------------------------------------------------

    log_lines: list[str] = []

    def log(message: str) -> None:
        print(message)
        log_lines.append(message)

    # -----------------------------------------------------------------------
    # Open session
    # -----------------------------------------------------------------------

    session = open_zenoh_session(
        endpoint
    )

    # -----------------------------------------------------------------------
    # Runtime collections
    # -----------------------------------------------------------------------

    synth_latencies_ms: list[float] = []

    dataset_latencies_ms: list[float] = []

    dataset_actions: list[np.ndarray] = []

    dataset_gt_actions: list[np.ndarray] = []

    dataset_frame_indices: list[int] = []

    dataset_row_indices: list[int] = []

    dataset_timestamps_s: list[
        float | None
    ] = []

    dataset_individual_losses: list[
        float
    ] = []

    dataset_raw_modes: list[str] = []

    dataset_query_status: list[str] = []

    metadata: dict[str, Any] = {}

    source = None

    try:
        # ===================================================================
        # Metadata
        # ===================================================================

        metadata = validate_metadata(
            query_once(
                session,
                "metadata",
                session_id,
                timeout,
            ),
            expected_horizon,
        )

        horizon = metadata[
            "action_horizon"
        ]

        log(
            "metadata: PASS "
            f"(transport={ZENOH_PROTOCOL_VERSION}, "
            f"semantic={SEMANTIC_PROTOCOL_VERSION}, "
            f"horizon={horizon}, "
            f"dim={ACTION_DIM})"
        )

        log("Building Dataset")

        # ===================================================================
        # Dataset source
        # ===================================================================

        if run_dataset:
            if dataset_root is None:
                raise ValidationError(
                    "dataset_root is required "
                    "for dataset validation"
                )

            from real_observation_source import (
                RealObservationSource,
            )

            source = RealObservationSource(
                dataset_root=dataset_root,
                drop_tactile_raw_every_n=(
                    dataset_requests
                ),
                episode_index=episode_index,
            )

            source.assert_requests_validity(
                dataset_requests,
                frame_stride=frame_stride,
            )

        # ===================================================================
        # Synthetic reset
        # ===================================================================

        validate_reset(
            query_once(
                session,
                "reset",
                session_id,
                timeout,
            )
        )

        log("reset: PASS")

        # ===================================================================
        # Synthetic validation
        # ===================================================================

        if run_synthetic:
            observation = (
                make_synthetic_observation()
            )

            for index in range(
                synth_requests
            ):
                request_observation = dict(
                    observation
                )

                if (
                    index
                    == synth_requests - 1
                ):
                    request_observation.pop(
                        "observation/image/tactile_raw",
                        None,
                    )

                started = time.monotonic()

                reply = query_once(
                    session,
                    "infer",
                    session_id,
                    timeout,
                    observation=(
                        request_observation
                    ),
                )

                latency_ms = (
                    time.monotonic()
                    - started
                ) * 1000.0

                validate_infer(
                    reply,
                    horizon,
                )

                synth_latencies_ms.append(
                    latency_ms
                )

                raw_mode = (
                    "without optional tactile_raw"
                    if index
                    == synth_requests - 1
                    else "full"
                )

                log(
                    f"infer Synthetic frames "
                    f"{index + 1}/"
                    f"{synth_requests}: PASS "
                    f"({latency_ms:.1f} ms, "
                    f"{raw_mode})"
                )

        # ===================================================================
        # Dataset validation
        # ===================================================================

        if run_dataset:
            validate_reset(
                query_once(
                    session,
                    "reset",
                    session_id,
                    timeout,
                )
            )

            source.reset_episode()

            log(
                "reset: PASS (dataset)"
            )

            for index in range(
                dataset_requests
            ):
                # -----------------------------------------------------------
                # Observation
                # -----------------------------------------------------------

                request_observation = (
                    source.next_observation(
                        frame_stride=frame_stride
                    )
                )

                frame_info = (
                    source.get_last_frame_info()
                )

                # -----------------------------------------------------------
                # Policy inference
                # -----------------------------------------------------------

                started = time.monotonic()

                reply = query_once(
                    session,
                    "infer",
                    session_id,
                    timeout,
                    observation=(
                        request_observation
                    ),
                )

                latency_ms = (
                    time.monotonic()
                    - started
                ) * 1000.0

                actions = validate_infer(
                    reply,
                    horizon,
                )

                # -----------------------------------------------------------
                # Ground truth
                #
                # If the horizon runs past the episode, the source repeats
                # the final GT action.
                # -----------------------------------------------------------

                gt_actions = (
                    source.get_ground_truth_actions(
                        horizon
                    )
                )

                # -----------------------------------------------------------
                # Store arrays
                # -----------------------------------------------------------

                dataset_actions.append(
                    actions
                )

                dataset_gt_actions.append(
                    gt_actions
                )

                dataset_latencies_ms.append(
                    latency_ms
                )

                # -----------------------------------------------------------
                # Per-query loss
                # -----------------------------------------------------------

                individual_loss = _mse(
                    actions,
                    gt_actions,
                )

                dataset_individual_losses.append(
                    individual_loss
                )

                # -----------------------------------------------------------
                # Metadata
                # -----------------------------------------------------------

                dataset_frame_indices.append(
                    int(
                        frame_info[
                            "frame_index"
                        ]
                    )
                )

                dataset_row_indices.append(
                    int(
                        frame_info[
                            "dataset_row_index"
                        ]
                    )
                )

                dataset_timestamps_s.append(
                    (
                        None
                        if frame_info[
                            "timestamp_s"
                        ] is None
                        else float(
                            frame_info[
                                "timestamp_s"
                            ]
                        )
                    )
                )

                raw_mode = (
                    "without optional tactile_raw"
                    if (
                        "observation/image/"
                        "tactile_raw"
                        not in request_observation
                    )
                    else "full"
                )

                dataset_raw_modes.append(
                    raw_mode
                )

                dataset_query_status.append(
                    "PASS"
                )

                # -----------------------------------------------------------
                # Console / JSON log
                # -----------------------------------------------------------

                timestamp = (
                    frame_info[
                        "timestamp_s"
                    ]
                )

                timestamp_text = (
                    "None"
                    if timestamp is None
                    else f"{timestamp:.3f}s"
                )

                log(
                    f"infer real-dataset "
                    f"{index + 1}/"
                    f"{dataset_requests}: PASS "
                    f"({latency_ms:.1f} ms, "
                    f"{raw_mode}, "
                    f"ep={frame_info['episode_index']} "
                    f"frame={frame_info['frame_index']} "
                    f"row={frame_info['dataset_row_index']} "
                    f"t={timestamp_text} "
                    f"loss={individual_loss:.8f})"
                )

    finally:
        session.close()

    # =========================================================================
    # Overall protocol result
    # =========================================================================

    log(
        f"PASS: policy is compatible with "
        f"{ZENOH_PROTOCOL_VERSION}"
    )

    log(
        f"obs_type: {obs_type}"
    )

    # =========================================================================
    # Latency summaries
    # =========================================================================

    synthetic_latency_summary = None

    if synth_latencies_ms:
        synthetic_latency_summary = {
            "requests": len(
                synth_latencies_ms
            ),
            "median_ms": float(
                statistics.median(
                    synth_latencies_ms
                )
            ),
            "max_ms": float(
                max(
                    synth_latencies_ms
                )
            ),
            "min_ms": float(
                min(
                    synth_latencies_ms
                )
            ),
            "mean_ms": float(
                np.mean(
                    synth_latencies_ms
                )
            ),
        }

        log(
            f"latency (synthetic): "
            f"requests="
            f"{len(synth_latencies_ms)} "
            f"median="
            f"{statistics.median(synth_latencies_ms):.1f} ms "
            f"max="
            f"{max(synth_latencies_ms):.1f} ms"
        )

    dataset_latency_summary = None

    if dataset_latencies_ms:
        dataset_latency_summary = {
            "requests": len(
                dataset_latencies_ms
            ),
            "median_ms": float(
                statistics.median(
                    dataset_latencies_ms
                )
            ),
            "max_ms": float(
                max(
                    dataset_latencies_ms
                )
            ),
            "min_ms": float(
                min(
                    dataset_latencies_ms
                )
            ),
            "mean_ms": float(
                np.mean(
                    dataset_latencies_ms
                )
            ),
        }

        log(
            f"latency (real-dataset): "
            f"requests="
            f"{len(dataset_latencies_ms)} "
            f"median="
            f"{statistics.median(dataset_latencies_ms):.1f} ms "
            f"max="
            f"{max(dataset_latencies_ms):.1f} ms"
        )

    # =========================================================================
    # Save dataset results
    # =========================================================================

    if (
        out_dir
        and dataset_actions
        and dataset_gt_actions
    ):
        predicted_actions = np.stack(
            dataset_actions,
            axis=0,
        )

        gt_actions = np.stack(
            dataset_gt_actions,
            axis=0,
        )

        # ---------------------------------------------------------------------
        # Individual losses are already calculated above.
        # ---------------------------------------------------------------------

        individual_losses = [
            float(x)
            for x in dataset_individual_losses
        ]

        # ---------------------------------------------------------------------
        # Full-run loss
        #
        # (T, horizon, 65)
        #          ->
        # (T * horizon * 65,)
        # ---------------------------------------------------------------------

        full_run_loss = _mse(
            predicted_actions,
            gt_actions,
        )

        # ---------------------------------------------------------------------
        # Determine output directory.
        #
        # --save-actions may be:
        #     /path/results/foo.npy
        # or
        #     /path/results/foo
        #
        # The actual generated names are controlled here.
        # ---------------------------------------------------------------------
        os.makedirs(
            out_dir,
            exist_ok=True,
        )

        season_name = (
            source.season_name
            if source is not None
            else "notnamed"
        )
        
        path_details  = 'policy_name' + f"_{season_name}" + f"_{datetime.now().strftime('%m-%d-%H-%M')}"
        save_dir = out_dir + '/' + path_details
        os.makedirs(
            save_dir,
            exist_ok=True,
        )
        
        base_name = (
            f"episode_{episode_index}"
            f"_queries_{dataset_requests}"
            f"_{path_details}"
        )

        actions_path = os.path.join(
            save_dir,
            f"{base_name}.npz",
        )

        json_path = os.path.join(
            save_dir,
            f"{base_name}.json",
        )

        # ---------------------------------------------------------------------
        # Save arrays
        # ---------------------------------------------------------------------

        np.savez(
            actions_path,
            predicted_actions=(
                predicted_actions
            ),
            gt_actions=(
                gt_actions
            ),
        )
        
        for start in range(0, 65, 10):
            dims = list(range(start, min(start + 10, 65)))

            plot_actions_vs_ground_truth(
                predicted_actions,
                gt_actions,
                os.path.join(
                    save_dir,
                    f"actions_vs_predicted_dims_{start:02d}_{dims[-1]:02d}.png",
                ),
                action_dims=dims,
            )

        # ---------------------------------------------------------------------
        # Per-query records
        # ---------------------------------------------------------------------

        query_results = []

        for i in range(
            len(dataset_actions)
        ):
            query_results.append(
                {
                    "query_index": int(i),

                    "status": (
                        dataset_query_status[i]
                    ),

                    "season_name": (
                        season_name
                    ),

                    "episode_index": int(
                        episode_index
                    ),

                    "frame_index": int(
                        dataset_frame_indices[i]
                    ),

                    "dataset_row_index": int(
                        dataset_row_indices[i]
                    ),

                    "timestamp_s": (
                        None
                        if dataset_timestamps_s[i]
                        is None
                        else float(
                            dataset_timestamps_s[i]
                        )
                    ),

                    "inference_time_ms": float(
                        dataset_latencies_ms[i]
                    ),

                    "raw_mode": (
                        dataset_raw_modes[i]
                    ),

                    "loss_mse": float(
                        dataset_individual_losses[i]
                    ),
                }
            )

        # ---------------------------------------------------------------------
        # Full JSON record
        # ---------------------------------------------------------------------

        results = {
            "run": {
                "status": "PASS",

                "season_name": season_name,

                "episode_index": int(
                    episode_index
                ),

                "num_queries": int(
                    dataset_requests
                ),

                "action_horizon": int(
                    horizon
                ),

                "action_dim": int(
                    ACTION_DIM
                ),

                "obs_type": obs_type,

                "dataset_root": (
                    os.path.abspath(
                        dataset_root
                    )
                    if dataset_root
                    else None
                ),

                "frame_stride": int(
                    frame_stride
                ),

                "endpoint": endpoint,

                "session_id": session_id,

                "timeout_s": float(
                    timeout
                ),

                "protocol_version": (
                    ZENOH_PROTOCOL_VERSION
                ),

                "semantic_protocol_version": (
                    SEMANTIC_PROTOCOL_VERSION
                ),

                "actions_file": (
                    os.path.basename(
                        actions_path
                    )
                ),
            },

            # -----------------------------------------------------------------
            # Full policy metadata returned by the server.
            # -----------------------------------------------------------------

            "policy_metadata": _json_safe(
                metadata
            ),

            # -----------------------------------------------------------------
            # Losses
            # -----------------------------------------------------------------

            "loss": {
                "metric":
                    "mean_squared_error",

                "flattening":
                    "flatten horizon and action dimensions",

                "per_query": individual_losses,

                "full_run": float(
                    full_run_loss
                ),

                "num_queries": len(
                    individual_losses
                ),

                "last_query": (
                    float(
                        individual_losses[-1]
                    )
                    if individual_losses
                    else None
                ),
            },

            # -----------------------------------------------------------------
            # Timing
            # -----------------------------------------------------------------

            "timing": {
                "per_query_inference_time_ms": [
                    float(x)
                    for x in dataset_latencies_ms
                ],

                "mean_ms": (
                    float(
                        np.mean(
                            dataset_latencies_ms
                        )
                    )
                    if dataset_latencies_ms
                    else None
                ),

                "median_ms": (
                    float(
                        np.median(
                            dataset_latencies_ms
                        )
                    )
                    if dataset_latencies_ms
                    else None
                ),

                "min_ms": (
                    float(
                        np.min(
                            dataset_latencies_ms
                        )
                    )
                    if dataset_latencies_ms
                    else None
                ),

                "max_ms": (
                    float(
                        np.max(
                            dataset_latencies_ms
                        )
                    )
                    if dataset_latencies_ms
                    else None
                ),

                "synthetic": (
                    synthetic_latency_summary
                ),

                "real_dataset": (
                    dataset_latency_summary
                ),
            },

            # -----------------------------------------------------------------
            # Every dataset query
            # -----------------------------------------------------------------

            "queries": query_results,

            # -----------------------------------------------------------------
            # Dataset-level arrays / information
            # -----------------------------------------------------------------

            "dataset": {
                "season_name": season_name,

                "episode_index": int(
                    episode_index
                ),

                "num_queries": int(
                    dataset_requests
                ),

                "frame_stride": int(
                    frame_stride
                ),

                "frame_indices": [
                    int(x)
                    for x in dataset_frame_indices
                ],

                "dataset_row_indices": [
                    int(x)
                    for x in dataset_row_indices
                ],

                "timestamps_s": [
                    (
                        None
                        if x is None
                        else float(x)
                    )
                    for x in dataset_timestamps_s
                ],

                "raw_modes": list(
                    dataset_raw_modes
                ),
            },

            # -----------------------------------------------------------------
            # Useful summary
            # -----------------------------------------------------------------

            "summary": {
                "status": "PASS",

                "season_name": season_name,

                "episode_index": int(
                    episode_index
                ),

                "num_queries": int(
                    dataset_requests
                ),

                "predicted_actions_shape": (
                    list(
                        predicted_actions.shape
                    )
                ),

                "gt_actions_shape": (
                    list(
                        gt_actions.shape
                    )
                ),

                "full_run_loss_mse": float(
                    full_run_loss
                ),

                "last_query_loss_mse": (
                    float(
                        individual_losses[-1]
                    )
                    if individual_losses
                    else None
                ),

                "first_timestamp_s": (
                    dataset_timestamps_s[0]
                    if dataset_timestamps_s
                    else None
                ),

                "last_timestamp_s": (
                    dataset_timestamps_s[-1]
                    if dataset_timestamps_s
                    else None
                ),

                "episode_duration_s": (
                    (
                        dataset_timestamps_s[-1]
                        - dataset_timestamps_s[0]
                    )
                    if (
                        dataset_timestamps_s
                        and dataset_timestamps_s[0]
                        is not None
                        and dataset_timestamps_s[-1]
                        is not None
                    )
                    else None
                ),
            },

            # -----------------------------------------------------------------
            # Exact console messages generated by this run.
            # -----------------------------------------------------------------

            "printed_log": list(
                log_lines
            ),

            # -----------------------------------------------------------------
            # Output files
            # -----------------------------------------------------------------

            "output": {
                "actions_file": os.path.abspath(
                    actions_path
                ),

                "json_file": os.path.abspath(
                    json_path
                ),
            },
        }

        # ---------------------------------------------------------------------
        # Save JSON
        # ---------------------------------------------------------------------

        with open(
            json_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                results,
                f,
                indent=2,
                allow_nan=False,
            )

        # ---------------------------------------------------------------------
        # Final console summary
        # ---------------------------------------------------------------------

        log("")
        log("=" * 70)
        log("DATASET RESULTS SAVED")
        log("=" * 70)

        log(
            f"season               : "
            f"{season_name}"
        )

        log(
            f"episode              : "
            f"{episode_index}"
        )

        log(
            f"queries              : "
            f"{dataset_requests}"
        )

        log(
            f"predicted shape      : "
            f"{predicted_actions.shape}"
        )

        log(
            f"GT shape             : "
            f"{gt_actions.shape}"
        )

        log(
            f"full-run MSE         : "
            f"{full_run_loss:.8f}"
        )

        if individual_losses:
            log(
                f"last-query MSE      : "
                f"{individual_losses[-1]:.8f}"
            )

        if dataset_latencies_ms:
            log(
                f"mean inference      : "
                f"{np.mean(dataset_latencies_ms):.2f} ms"
            )

            log(
                f"median inference    : "
                f"{np.median(dataset_latencies_ms):.2f} ms"
            )

            log(
                f"min inference       : "
                f"{np.min(dataset_latencies_ms):.2f} ms"
            )

            log(
                f"max inference       : "
                f"{np.max(dataset_latencies_ms):.2f} ms"
            )

        if dataset_timestamps_s:
            first_t = dataset_timestamps_s[0]
            last_t = dataset_timestamps_s[-1]

            if (
                first_t is not None
                and last_t is not None
            ):
                log(
                    f"episode time        : "
                    f"{first_t:.3f}s -> "
                    f"{last_t:.3f}s "
                    f"(duration="
                    f"{last_t - first_t:.3f}s)"
                )

        log(
            f"actions file        : "
            f"{actions_path}"
        )

        log(
            f"results file        : "
            f"{json_path}"
        )

        log("=" * 70)

        # ---------------------------------------------------------------------
        # Re-write JSON after the final summary logging so the JSON contains
        # those final printed lines too.
        # ---------------------------------------------------------------------

        results["printed_log"] = list(
            log_lines
        )

        with open(
            json_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                results,
                f,
                indent=2,
                allow_nan=False,
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argument_parser(
) -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--endpoint",
        default=os.environ.get(
            "ORIGAMI_ZENOH_ENDPOINT"
        ),
        help=(
            "Zenoh router endpoint "
            "tcp/<IP-or-DNS-host>:<port> "
            "(or ORIGAMI_ZENOH_ENDPOINT)"
        ),
    )

    parser.add_argument(
        "--session-id",
        default=os.environ.get(
            "ORIGAMI_SESSION_ID"
        ),
        help=(
            "assigned session ID "
            "(or ORIGAMI_SESSION_ID)"
        ),
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="per-query timeout in seconds",
    )

    parser.add_argument(
        "--requests",
        type=int,
        default=3,
        help="number of synthetic infer queries",
    )

    parser.add_argument(
        "--expected-horizon",
        "--expected-action-horizon",
        dest="expected_horizon",
        type=int,
        default=None,
        help="require this exact metadata/action horizon",
    )

    parser.add_argument(
        "--obs-type",
        type=str,
        default=None,
        required=True,
        help=(
            "one of ['synthetic', "
            "'dataset', 'both']"
        ),
    )

    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help=(
            "path to .../lerobot3.0 "
            "for real dataset replay"
        ),
    )

    parser.add_argument(
        "--dataset-requests",
        type=int,
        default=None,
        help=(
            "number of real dataset infer queries"
        ),
    )

    parser.add_argument(
        "--episode-index",
        type=int,
        default=None,
        help="dataset episode to replay",
    )

    parser.add_argument(
        "--frame-stride",
        type=int,
        default=None,
        help=(
            "dataset frames to advance per "
            "infer() call; use action horizon "
            "for non-overlapping chunks"
        ),
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help=(
            "output path/directory hint. "
            "The validator generates "
            "episode_<episode>_queries_<N>.npz "
            "and .json in its directory."
        ),
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    argv: list[str] | None = None,
) -> int:

    parser = build_argument_parser()

    args = parser.parse_args(
        argv
    )

    if not args.endpoint:
        parser.error(
            "--endpoint or "
            "ORIGAMI_ZENOH_ENDPOINT is required"
        )

    if not args.session_id:
        parser.error(
            "--session-id or "
            "ORIGAMI_SESSION_ID is required"
        )

    if args.obs_type in (
        "dataset",
        "both",
    ):
        if not args.dataset_root:
            parser.error(
                "--dataset-root is required "
                "when --obs-type is "
                "'dataset' or 'both'"
            )

        if args.episode_index is None:
            parser.error(
                "--episode-index is required "
                "when --obs-type is "
                "'dataset' or 'both'"
            )

        if args.frame_stride is None:
            parser.error(
                "--frame-stride is required "
                "when --obs-type is "
                "'dataset' or 'both'"
            )

        if (
            args.dataset_requests is None
            or args.dataset_requests < 1
        ):
            parser.error(
                "--dataset-requests must be "
                ">= 1 when --obs-type is "
                "'dataset' or 'both'"
            )

        if args.out_dir is None:
            parser.error(
                "--save-actions is required "
                "when --obs-type is "
                "'dataset' or 'both'"
            )
        os.makedirs(
            args.out_dir,
            exist_ok=True,
        )

    try:
        run_validation(
            endpoint=args.endpoint,
            session_id=args.session_id,
            timeout=args.timeout,
            synth_requests=args.requests,
            expected_horizon=(
                args.expected_horizon
            ),
            obs_type=args.obs_type,
            dataset_requests=(
                args.dataset_requests
                if args.dataset_requests
                is not None
                else 0
            ),
            dataset_root=args.dataset_root,
            episode_index=(
                args.episode_index
                if args.episode_index
                is not None
                else 0
            ),
            frame_stride=(
                args.frame_stride
                if args.frame_stride
                is not None
                else 1
            ),
            out_dir=args.out_dir,
        )

    except (
        ValidationError,
        TimeoutError,
        OSError,
    ) as exc:

        print(
            f"FAIL: {exc}",
            file=sys.stderr,
        )

        return 1

    return 0

def plot_actions_vs_ground_truth(
    predicted_actions,
    gt_actions,
    output_path,
    action_dims=None,
):
    """
    Plot ground-truth vs predicted actions for selected action dimensions.

    Parameters
    ----------
    predicted_actions : np.ndarray
        Shape: (num_queries, horizon, action_dim)
    gt_actions : np.ndarray
        Shape: (num_queries, horizon, action_dim)

    output_path : str or Path
        Where to save the PNG.

    action_dims : list[int] or None
        Action dimensions to plot.
        If None, plots a useful default set.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    predicted_actions = np.asarray(predicted_actions)
    gt_actions = np.asarray(gt_actions)

    if predicted_actions.shape != gt_actions.shape:
        raise ValueError(
            f"Shape mismatch: predicted={predicted_actions.shape}, "
            f"gt={gt_actions.shape}"
        )

    if predicted_actions.ndim != 3:
        raise ValueError(
            "Expected actions with shape "
            "(num_queries, horizon, action_dim)"
        )

    num_queries, horizon, action_dim = predicted_actions.shape

    # Flatten query/horizon so the plot follows the actual episode timeline.
    pred_flat = predicted_actions.reshape(
        num_queries * horizon,
        action_dim,
    )

    gt_flat = gt_actions.reshape(
        num_queries * horizon,
        action_dim,
    )

    if action_dims is None:
        # Pick the first few dimensions by default.
        action_dims = list(range(min(6, action_dim)))

    # Validate requested dimensions.
    for dim in action_dims:
        if dim < 0 or dim >= action_dim:
            raise ValueError(
                f"Invalid action dimension {dim}. "
                f"Valid range: 0..{action_dim - 1}"
            )

    n_plots = len(action_dims)

    fig, axes = plt.subplots(
        n_plots,
        1,
        figsize=(14, 3.2 * n_plots),
        sharex=True,
    )

    # When there is only one subplot, matplotlib doesn't return a list.
    if n_plots == 1:
        axes = [axes]

    x = np.arange(num_queries * horizon)

    for ax, dim in zip(axes, action_dims):
        ax.plot(
            x,
            gt_flat[:, dim],
            label="Ground truth",
            linewidth=1.5,
        )

        ax.plot(
            x,
            pred_flat[:, dim],
            label="Predicted",
            linewidth=1.2,
            linestyle="--",
        )

        ax.set_ylabel(f"Action dim {dim}")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel(
        "Action timestep across evaluated queries"
    )

    fig.suptitle(
        "Ground Truth vs Predicted Actions",
        fontsize=14,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)
    
    
if __name__ == "__main__":
    raise SystemExit(
        main()
    )