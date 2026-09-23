#!/usr/bin/env python3
"""`origami-zenoh-v1` policy server for the T-Rex origami checkpoint (all-absolute).

This is the kit's `policy_server_template.py` (codec, envelope, the three
queryables, observation validation, shutdown) with `TeamPolicy` replaced by
`trex_origami.policy.TRexOrigamiPolicy`.  The transport code is copied rather
than imported so the submission image needs nothing from the kit at runtime.

    ORIGAMI_ZENOH_ENDPOINT=tcp/<router>:7447 ORIGAMI_SESSION_ID=<id> \\
    python scripts/serve_origami_zenoh.py --checkpoint_path /opt/policy/checkpoint

Local test (see `docs/competition_participant_complete_guide.md` §13):

    zenohd -l tcp/127.0.0.1:17447 &
    ORIGAMI_ZENOH_ENDPOINT=tcp/127.0.0.1:17447 ORIGAMI_SESSION_ID=local-contract-test \\
        python scripts/serve_origami_zenoh.py --checkpoint_path ... &
    python <kit>/examples/check_zenoh_policy.py --endpoint tcp/127.0.0.1:17447 \\
        --session-id local-contract-test --timeout 180 --requests 3 --expected-horizon 25
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import sys
import threading
import time
import uuid
from collections.abc import Mapping
from typing import Any

import msgpack
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from trex_origami.policy import (TRexOrigamiPolicy, add_policy_arguments,  # noqa: E402
                                 config_from_args)
from trex_origami.seasons import JOINT_NAMES  # noqa: E402

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
OPTIONAL_IMAGE_SPECS = {"observation/image/tactile_raw": (480, 1600, 3)}
VECTOR_SPECS = {
    "observation/state": (ACTION_DIM,),
    "observation/state/joint_torque": (ACTION_DIM,),
    "observation/tactile": (60,),
}
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024


# ── codec (verbatim from the kit template) ────────────────────────────────────
def _pack_numpy(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"O", "V", "c"}:
            raise ValueError(f"unsupported numpy dtype: {value.dtype}")
        array = np.ascontiguousarray(value)
        return {b"__ndarray__": True, b"data": array.tobytes(),
                b"dtype": array.dtype.str, b"shape": array.shape}
    if isinstance(value, np.generic):
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _mapping_value(value: Mapping[Any, Any], key: str) -> Any:
    return value[key] if key in value else value.get(key.encode())


def _unpack_numpy(value: dict[Any, Any]) -> Any:
    if _mapping_value(value, "__ndarray__") is True:
        data = _mapping_value(value, "data")
        shape = _mapping_value(value, "shape")
        dtype = np.dtype(_mapping_value(value, "dtype"))
        if (not isinstance(data, bytes) or not isinstance(shape, (list, tuple))
                or len(shape) > 8 or dtype.kind in {"O", "V", "c"} or dtype.hasobject):
            raise ValueError("invalid numpy array payload")
        normalized_shape = tuple(int(d) for d in shape)
        if any(d < 0 for d in normalized_shape):
            raise ValueError("invalid numpy array shape")
        expected_size = math.prod(normalized_shape) * dtype.itemsize
        if expected_size > MAX_PAYLOAD_BYTES or len(data) != expected_size:
            raise ValueError("numpy array payload size does not match shape")
        return np.frombuffer(data, dtype=dtype).reshape(normalized_shape)
    if _mapping_value(value, "__npgeneric__") is True:
        return np.dtype(_mapping_value(value, "dtype")).type(_mapping_value(value, "data"))
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
    return msgpack.unpackb(payload, object_hook=_unpack_numpy, raw=False,
                           strict_map_key=False, max_bin_len=MAX_PAYLOAD_BYTES,
                           max_array_len=1_000_000, max_map_len=10_000,
                           max_str_len=1_000_000)


def validate_observation(observation: Any) -> None:
    """The template's schema check; also used by the offline replay."""
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
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.shape != shape:
            raise ValueError(f"{key} must be uint8{shape}")
    for key, shape in VECTOR_SPECS.items():
        vector = observation.get(key)
        if (not isinstance(vector, np.ndarray) or vector.dtype != np.float32
                or vector.shape != shape or not np.isfinite(vector).all()):
            raise ValueError(f"{key} must be finite float32{shape}")
    if not isinstance(observation.get("prompt"), str):
        raise ValueError("prompt must be a string")


# ── server ────────────────────────────────────────────────────────────────────
class OrigamiZenohServer:
    def __init__(self, policy: TRexOrigamiPolicy, *, endpoint: str, session_id: str,
                 execution_mode: str = "async") -> None:
        if execution_mode not in {"sync", "async"}:
            raise ValueError("execution_mode must be 'sync' or 'async'")
        self.policy = policy
        self.endpoint = endpoint
        self.session_id = session_id
        self.action_horizon = int(policy.action_chunk)
        self.execution_mode = execution_mode
        self._stop = threading.Event()
        self._session: Any | None = None
        self._queryables: list[Any] = []
        self.metadata = {
            "protocol_version": SEMANTIC_VERSION,
            "action_dim": ACTION_DIM,
            "action_horizon": self.action_horizon,
            "action_type": "absolute_joint_position",
            "action_units": "radians",
            "joint_names": list(JOINT_NAMES),
            "execution_mode": execution_mode,
            "inference_kit": "origami-inference-kit-async",
        }

    def serve_forever(self) -> None:
        import zenoh
        config = zenoh.Config()
        config.insert_json5("mode", json.dumps("client"))
        config.insert_json5("connect/endpoints", json.dumps([self.endpoint]))
        config.insert_json5("scouting/multicast/enabled", "false")
        config.insert_json5("transport/shared_memory/enabled", "false")
        self._session = zenoh.open(config)
        self._queryables = [
            self._session.declare_queryable(f"{TRANSPORT_VERSION}/{op}", self._handle_query,
                                            complete=True)
            for op in ("metadata", "reset", "infer")]
        signal.signal(signal.SIGTERM, lambda *_: self._stop.set())
        signal.signal(signal.SIGINT, lambda *_: self._stop.set())
        logging.info("READY transport=%s endpoint=%s horizon=%d mode=%s",
                     TRANSPORT_VERSION, self.endpoint, self.action_horizon,
                     self.execution_mode)
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
            logging.error("request failed operation=%s error_id=%s type=%s msg=%s",
                          operation, error_id, type(exc).__name__, exc)
            public_message = f"request failed; error_id={error_id}"
            if not isinstance(request, Mapping):
                query.reply_err(pack_payload({"error": {"code": "INVALID_REQUEST",
                                                        "message": public_message,
                                                        "retryable": False}}),
                                encoding="application/msgpack")
                return
            response = self._envelope(operation, request)
            response["error"] = {
                "code": "INFERENCE_FAILED" if operation == "infer" else "INVALID_REQUEST",
                "message": public_message, "retryable": False}
        query.reply(str(query.key_expr), pack_payload(response), encoding="application/msgpack")

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
            self.policy.reset()
            response["ok"] = True
            return response
        if operation != "infer":
            raise ValueError(f"unsupported operation: {operation}")

        observation = request.get("observation")
        validate_observation(observation)
        started = time.monotonic()
        actions = np.asarray(self.policy.infer(dict(observation)))
        expected_shape = (self.action_horizon, ACTION_DIM)
        if actions.dtype != np.float32 or actions.shape != expected_shape:
            raise ValueError(f"policy actions must be float32{expected_shape}, "
                             f"got {actions.dtype}{actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("policy actions contain NaN or Inf")
        response["actions"] = np.ascontiguousarray(actions)
        infer_ms = (time.monotonic() - started) * 1000.0
        response["server_timing"] = {"infer_ms": infer_ms}
        logging.info("infer #%d %.0f ms (%s)", self.policy.n_infer, infer_ms,
                     " ".join(f"{k}={v:.0f}" for k, v in self.policy.last_timing.items()))
        return response

    def _envelope(self, operation: str, request: Mapping[str, Any]) -> dict[str, Any]:
        return {"protocol_version": TRANSPORT_VERSION, "operation": operation,
                "request_id": request.get("request_id"), "session_id": self.session_id}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoint", default=os.environ.get("ORIGAMI_ZENOH_ENDPOINT"))
    parser.add_argument("--session-id", default=os.environ.get("ORIGAMI_SESSION_ID"))
    parser.add_argument("--execution-mode", choices=("sync", "async"),
                        default=os.environ.get("EXECUTION_MODE", "async"))
    add_policy_arguments(parser)
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    if not args.endpoint:
        raise SystemExit("--endpoint or ORIGAMI_ZENOH_ENDPOINT is required")
    if not args.session_id:
        raise SystemExit("--session-id or ORIGAMI_SESSION_ID is required")
    if not args.checkpoint_path:
        raise SystemExit("--checkpoint_path or TREX_CKPT_PATH is required")
    policy = TRexOrigamiPolicy(config_from_args(args))
    server = OrigamiZenohServer(policy, endpoint=args.endpoint, session_id=args.session_id,
                                execution_mode=args.execution_mode)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
