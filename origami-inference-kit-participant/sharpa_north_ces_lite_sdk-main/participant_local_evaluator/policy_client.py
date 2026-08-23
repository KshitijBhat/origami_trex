"""Strict client for a local ``origami-zenoh-v1`` policy container."""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Iterable, Mapping
from typing import Any

from .codec import pack_payload, unpack_payload
from .contract import PolicyMetadata, ZENOH_PROTOCOL_VERSION, validate_actions

_OPERATIONS = frozenset({"metadata", "reset", "infer"})
_ENVELOPE_KEYS = frozenset(
    {"protocol_version", "operation", "request_id", "session_id"}
)


class ZenohPolicyClient:
    def __init__(
        self,
        endpoint: str,
        *,
        session_id: str,
        timeout_s: float = 180.0,
        session: Any | None = None,
        transport: Any | None = None,
    ) -> None:
        if session is not None and transport is not None:
            raise ValueError("session and transport are mutually exclusive")
        if not isinstance(endpoint, str) or not endpoint.startswith(
            ("tcp/", "unixsock-stream/")
        ):
            raise ValueError("local policy endpoint must use tcp/ or unixsock-stream/")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.endpoint = endpoint
        self.session_id = session_id
        self.timeout_s = float(timeout_s)
        self._transport = transport or _ZenohTransport(endpoint, session=session)
        self._lock = threading.Lock()
        self._closed = False
        try:
            reply = self._query("metadata")
            raw = reply.get("metadata")
            if not isinstance(raw, Mapping):
                raise ValueError("metadata reply must contain a metadata object")
            self.metadata = PolicyMetadata.parse(raw)
            self.raw_metadata = dict(raw)
        except BaseException:
            self.close()
            raise

    def reset(self) -> None:
        reply = self._query("reset")
        if reply.get("ok") is not True:
            raise RuntimeError("policy reset reply must contain {'ok': True}")

    def infer(self, observation: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        if not isinstance(observation, Mapping):
            raise TypeError("observation must be an object")
        reply = self._query("infer", observation=dict(observation))
        actions = validate_actions(reply.get("actions"), self.metadata)
        metrics = {
            key: value
            for key, value in reply.items()
            if key not in _ENVELOPE_KEYS and key != "actions"
        }
        return actions, metrics

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            close = getattr(self._transport, "close", None)
            if close is not None:
                close()

    def _query(self, operation: str, **body: Any) -> dict[str, Any]:
        if operation not in _OPERATIONS:
            raise ValueError(f"unsupported policy operation: {operation!r}")
        request_id = uuid.uuid4().hex
        request = {
            "protocol_version": ZENOH_PROTOCOL_VERSION,
            "operation": operation,
            "request_id": request_id,
            "session_id": self.session_id,
            **body,
        }
        key = f"{ZENOH_PROTOCOL_VERSION}/{operation}"
        with self._lock:
            if self._closed:
                raise RuntimeError("ZenohPolicyClient is closed")
            replies = self._transport.query(
                key=key,
                payload=pack_payload(request),
                timeout_s=self.timeout_s,
            )
            payloads = _materialize(replies)
        if len(payloads) != 1:
            raise RuntimeError(f"{key} must return exactly one reply, got {len(payloads)}")
        reply = unpack_payload(payloads[0])
        if not isinstance(reply, dict):
            raise ValueError(f"{key} reply must be an object")
        expected = {
            "protocol_version": ZENOH_PROTOCOL_VERSION,
            "operation": operation,
            "request_id": request_id,
            "session_id": self.session_id,
        }
        for field, value in expected.items():
            if reply.get(field) != value:
                raise ValueError(
                    f"{operation} reply {field!r} must be {value!r}, got {reply.get(field)!r}"
                )
        if "error" in reply:
            error = reply["error"]
            if isinstance(error, Mapping):
                raise RuntimeError(
                    f"policy {operation} failed: {error.get('code', 'UNKNOWN')}: "
                    f"{error.get('message', '')}"
                )
            raise RuntimeError(f"policy {operation} returned a malformed error")
        return reply


class _ZenohTransport:
    def __init__(self, endpoint: str, *, session: Any | None) -> None:
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
            session = zenoh.open(config)
            self._zenoh = zenoh
        self._session = session

    def query(self, *, key: str, payload: bytes, timeout_s: float) -> list[bytes]:
        options: dict[str, Any] = {
            "payload": payload,
            "timeout": timeout_s,
        }
        if self._zenoh is not None:
            options["consolidation"] = self._zenoh.ConsolidationMode.NONE
        replies = self._session.get(key, **options)
        return [_reply_bytes(reply) for reply in replies]

    def close(self) -> None:
        if self._owns_session:
            self._session.close()


def open_router_session(listen_endpoint: str) -> Any:
    if not isinstance(listen_endpoint, str) or not listen_endpoint.startswith(
        "unixsock-stream/"
    ):
        raise ValueError("listen_endpoint must use unixsock-stream/")
    try:
        import zenoh
    except ImportError as error:
        raise RuntimeError("eclipse-zenoh 1.9 is required") from error
    config = zenoh.Config()
    config.insert_json5("mode", json.dumps("router"))
    config.insert_json5("listen/endpoints", json.dumps([listen_endpoint]))
    config.insert_json5("scouting/multicast/enabled", "false")
    config.insert_json5("transport/shared_memory/enabled", "false")
    return zenoh.open(config)


def _materialize(replies: Any) -> list[bytes]:
    if isinstance(replies, (bytes, bytearray, memoryview)):
        return [bytes(replies)]
    if not isinstance(replies, Iterable):
        raise TypeError("transport.query() must return a payload or iterable")
    return [_reply_bytes(reply) for reply in replies]


def _reply_bytes(reply: Any) -> bytes:
    if isinstance(reply, (bytes, bytearray, memoryview)):
        return bytes(reply)
    error = getattr(reply, "err", None)
    if error is not None:
        raise RuntimeError("Zenoh returned an error reply")
    sample = getattr(reply, "ok", None)
    if sample is None:
        raise RuntimeError("Zenoh reply contains no sample")
    payload = getattr(sample, "payload", sample)
    to_bytes = getattr(payload, "to_bytes", None)
    if to_bytes is None:
        raise TypeError("Zenoh reply payload is not bytes")
    return bytes(to_bytes())
