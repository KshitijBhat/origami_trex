"""Authenticated read-only client for ``origami-remote-v1`` observations."""

from __future__ import annotations

import json
import math
import re
import time
import uuid
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from .codec import pack_payload, unpack_payload
from .contract import (
    JOINT_NAMES,
    OBSERVATION_FIELD_METADATA,
    OPTIONAL_IMAGE_SPECS,
    PROTOCOL_VERSION,
    REQUIRED_IMAGE_SPECS,
    VECTOR_SPECS,
)

REMOTE_PROTOCOL_VERSION = "origami-remote-v1"
_SESSION_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


class RemoteObservationClient:
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
        transport: Any | None = None,
        wall_clock: Any = time.time,
    ) -> None:
        _validate_endpoint(endpoint)
        if not isinstance(session_id, str) or _SESSION_ID_PATTERN.fullmatch(session_id) is None:
            raise ValueError("session_id contains unsafe characters")
        if not isinstance(token, str) or not 1 <= len(token) <= 4096:
            raise ValueError("token must contain 1..4096 characters")
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
            raise ValueError("mTLS certificate and private key must be supplied together")
        self.endpoint = endpoint
        self.session_id = session_id
        self.timeout_s = float(timeout_s)
        self.max_observation_age_s = float(max_observation_age_s)
        self.last_metadata: dict[str, Any] | None = None
        self.last_observation_timestamp: float | None = None
        self._token = token
        self._clock = wall_clock
        self._closed = False
        self._transport = transport or _ZenohRemoteTransport(
            endpoint,
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
            "operation": "observation",
            "request_id": request_id,
            "session_id": self.session_id,
            "token": self._token,
        }
        replies = self._transport.query(
            key=f"{REMOTE_PROTOCOL_VERSION}/{self.session_id}/observation",
            payload=pack_payload(request),
            timeout_s=self.timeout_s,
        )
        payloads = _materialize(replies)
        if len(payloads) != 1:
            raise RuntimeError(f"expected exactly one observation reply, got {len(payloads)}")
        response = unpack_payload(payloads[0])
        if not isinstance(response, Mapping):
            raise ValueError("observation reply must be an object")
        expected = {
            "protocol_version": REMOTE_PROTOCOL_VERSION,
            "operation": "observation",
            "request_id": request_id,
            "session_id": self.session_id,
        }
        for field, value in expected.items():
            if response.get(field) != value:
                raise ValueError(f"observation reply {field} does not match the request")
        if "error" in response:
            error = response["error"]
            if isinstance(error, Mapping):
                raise RuntimeError(
                    f"{error.get('code', 'UNKNOWN')}: {error.get('message', '')}"
                )
            raise RuntimeError("remote service returned a malformed error")
        metadata = _validate_metadata(response.get("metadata"))
        timestamp = _validate_timestamp(
            response.get("observation_timestamp"),
            now=float(self._clock()),
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


class _ZenohRemoteTransport:
    def __init__(
        self,
        endpoint: str,
        *,
        tls_root_ca_certificate: str | None,
        tls_client_certificate: str | None,
        tls_client_private_key: str | None,
    ) -> None:
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
            tls: dict[str, Any] = {"root_ca_certificate": tls_root_ca_certificate}
            if tls_client_certificate is not None:
                tls.update(
                    {
                        "enable_mtls": True,
                        "connect_certificate": tls_client_certificate,
                        "connect_private_key": tls_client_private_key,
                    }
                )
            config.insert_json5("transport/link/tls", json.dumps(tls))
        self._zenoh = zenoh
        self._session = zenoh.open(config)

    def query(self, *, key: str, payload: bytes, timeout_s: float) -> list[bytes]:
        replies = self._session.get(
            key,
            payload=payload,
            timeout=timeout_s,
            consolidation=self._zenoh.ConsolidationMode.NONE,
        )
        return [_reply_bytes(reply) for reply in replies]

    def close(self) -> None:
        self._session.close()


def _validate_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("remote metadata must be an object")
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "observation_schema": "policy-infer-input",
        "observation_fields": OBSERVATION_FIELD_METADATA,
    }
    for field, required in expected.items():
        if value.get(field) != required:
            raise ValueError(f"remote metadata[{field!r}] must be {required!r}")
    names = value.get("joint_names")
    if not isinstance(names, (list, tuple)) or tuple(names) != JOINT_NAMES:
        raise ValueError("remote metadata joint_names do not match the 65-joint order")
    return dict(value)


def _validate_timestamp(value: Any, *, now: float, max_age_s: float) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError("observation_timestamp must be finite")
    timestamp = float(value)
    age = now - timestamp
    if age < -max_age_s or age > max_age_s:
        raise ValueError(f"remote observation timestamp is outside freshness window: {age:.3f}s")
    return timestamp


def _validate_observation(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("remote observation must be an object")
    required = {*REQUIRED_IMAGE_SPECS, *VECTOR_SPECS, "prompt"}
    allowed = required | set(OPTIONAL_IMAGE_SPECS)
    if not required.issubset(value) or not set(value).issubset(allowed):
        raise ValueError("remote observation keys do not match the infer schema")
    result: dict[str, Any] = {}
    for key, shape in {**REQUIRED_IMAGE_SPECS, **OPTIONAL_IMAGE_SPECS}.items():
        if key not in value:
            continue
        image = value[key]
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.dtype(np.uint8)
            or image.shape != shape
        ):
            raise ValueError(f"{key} must be uint8{shape}")
        result[key] = np.ascontiguousarray(image)
    for key, shape in VECTOR_SPECS.items():
        vector = value[key]
        if (
            not isinstance(vector, np.ndarray)
            or vector.dtype != np.dtype(np.float32)
            or vector.shape != shape
            or not np.isfinite(vector).all()
        ):
            raise ValueError(f"{key} must be finite float32{shape}")
        result[key] = np.ascontiguousarray(vector)
    if not isinstance(value["prompt"], str):
        raise ValueError("prompt must be a string")
    result["prompt"] = value["prompt"]
    return result


def _materialize(replies: Any) -> list[bytes]:
    if isinstance(replies, (bytes, bytearray, memoryview)):
        return [bytes(replies)]
    if not isinstance(replies, Iterable):
        raise TypeError("remote transport must return a payload or iterable")
    return [_reply_bytes(reply) for reply in replies]


def _reply_bytes(reply: Any) -> bytes:
    if isinstance(reply, (bytes, bytearray, memoryview)):
        return bytes(reply)
    if getattr(reply, "err", None) is not None:
        raise RuntimeError("Zenoh returned a remote error reply")
    sample = getattr(reply, "ok", None)
    if sample is None:
        raise RuntimeError("Zenoh reply contains no sample")
    payload = getattr(sample, "payload", sample)
    to_bytes = getattr(payload, "to_bytes", None)
    if to_bytes is None:
        raise TypeError("Zenoh reply payload is not bytes")
    return bytes(to_bytes())


def _validate_endpoint(endpoint: str) -> None:
    if not isinstance(endpoint, str) or not endpoint.startswith(("tcp/", "tls/")):
        raise ValueError("endpoint must use tcp/<host>:<port> or tls/<host>:<port>")
    host, separator, port_text = endpoint.split("/", 1)[1].rpartition(":")
    if not separator or not host or any(character.isspace() for character in host):
        raise ValueError("endpoint must contain a non-empty host and port")
    try:
        port = int(port_text)
    except ValueError as error:
        raise ValueError("endpoint port must be an integer") from error
    if not 1 <= port <= 65535:
        raise ValueError("endpoint port must be in [1,65535]")
