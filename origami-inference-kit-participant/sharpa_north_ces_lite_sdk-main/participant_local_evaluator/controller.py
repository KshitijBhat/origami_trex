"""Stateful orchestration for remote observation -> local policy -> Shadow result."""

from __future__ import annotations

import math
import hashlib
import os
import pathlib
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from typing import Any

import cv2
import numpy as np

from .contract import (
    ACTION_DIM,
    JOINT_NAMES,
    OPTIONAL_IMAGE_SPECS,
    REQUIRED_IMAGE_SPECS,
    VECTOR_SPECS,
)
from .docker_runtime import DockerRuntime
from .policy_client import ZenohPolicyClient, open_router_session
from .remote_client import RemoteObservationClient
from .trajectory import TrajectoryValidator

RemoteFactory = Callable[..., Any]
PolicyFactory = Callable[..., Any]


class LocalEvaluatorController:
    """Own local-only credentials and the Shadow evaluation lifecycle."""

    def __init__(
        self,
        runtime: DockerRuntime,
        trajectory_validator: TrajectoryValidator,
        *,
        remote_factory: RemoteFactory = RemoteObservationClient,
        policy_factory: PolicyFactory = ZenohPolicyClient,
        router_session_factory: Callable[[str], Any] = open_router_session,
        policy_timeout_s: float = 180.0,
        startup_timeout_s: float = 900.0,
        upload_dir: str | pathlib.Path = "/tmp/origami-participant-uploads",
        max_upload_bytes: int = 100 * 1024 * 1024 * 1024,
    ) -> None:
        self.runtime = runtime
        self.trajectory_validator = trajectory_validator
        self.remote_factory = remote_factory
        self.policy_factory = policy_factory
        self.router_session_factory = router_session_factory
        self.policy_timeout_s = float(policy_timeout_s)
        self.startup_timeout_s = float(startup_timeout_s)
        self.upload_dir = pathlib.Path(upload_dir).resolve()
        self.max_upload_bytes = int(max_upload_bytes)
        if self.max_upload_bytes < 1:
            raise ValueError("max_upload_bytes must be positive")
        self._lock = threading.RLock()
        self._archive_state_lock = threading.Lock()
        self._archive_job: dict[str, Any] | None = None
        self._remote: Any | None = None
        self._policy: Any | None = None
        self._host_zenoh_session: Any | None = None
        self._remote_endpoint: str | None = None
        self._remote_session: str | None = None
        self._remote_token: str | None = None
        self._policy_session: str | None = None
        self._image: str | None = None
        self._observation: dict[str, Any] | None = None
        self._trajectory: dict[str, Any] | None = None
        self._last_error: str | None = None
        self._last_operation = "idle"
        self._events: list[str] = []

    def connect_remote(
        self,
        *,
        endpoint: str,
        session_id: str,
        token: str,
        tls_root_ca_certificate: str | None = None,
        tls_client_certificate: str | None = None,
        tls_client_private_key: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self._set_operation("connecting remote observation")
            old = self._remote
            client = None
            try:
                client = self.remote_factory(
                    endpoint,
                    session_id=session_id,
                    token=token,
                    tls_root_ca_certificate=tls_root_ca_certificate or None,
                    tls_client_certificate=tls_client_certificate or None,
                    tls_client_private_key=tls_client_private_key or None,
                )
                observation = client.get_observation()
                self._validate_observation(observation)
            except Exception as error:
                if client is not None:
                    client.close()
                self._fail(error, token=token)
                raise RuntimeError(self._last_error) from None
            if old is not None:
                old.close()
            self._remote = client
            self._remote_endpoint = endpoint
            self._remote_session = session_id
            self._remote_token = token
            self._observation = observation
            self._last_error = None
            self._set_operation("remote observation connected")
            return self.remote_status()

    def load_archive(self, archive_path: str, sha256: str) -> dict[str, Any]:
        job = self.start_archive_load(archive_path, sha256)
        return self.wait_archive_load(job["job_id"])

    def start_archive_load(self, archive_path: str, sha256: str) -> dict[str, Any]:
        archive = archive_path.strip()
        if not archive:
            raise ValueError("archive_path must not be empty")
        expected = sha256.strip().lower()
        if len(expected) != 64 or any(
            character not in "0123456789abcdef" for character in expected
        ):
            raise ValueError("sha256 must contain exactly 64 hex characters")
        return self._start_archive_load_job(
            archive,
            expected,
            cleanup_path=None,
        )

    def archive_load_status(self, job_id: str) -> dict[str, Any]:
        with self._archive_state_lock:
            if self._archive_job is None or self._archive_job["job_id"] != job_id:
                raise ValueError("unknown archive load job")
            return dict(self._archive_job)

    def wait_archive_load(self, job_id: str, *, timeout_s: float = 1800.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() <= deadline:
            job = self.archive_load_status(job_id)
            if job["status"] == "completed":
                assert job["result"] is not None
                return dict(job["result"])
            if job["status"] == "failed":
                raise RuntimeError(job.get("error") or "archive load failed")
            time.sleep(0.5)
        raise TimeoutError(f"archive load job {job_id} timed out after {timeout_s}s")

    def upload_archive(
        self,
        stream: Any,
        *,
        filename: str,
        content_length: int,
        expected_sha256: str | None = None,
    ) -> dict[str, Any]:
        safe_name = pathlib.PurePath(filename).name
        if safe_name != filename or not safe_name.endswith(".tar.zst"):
            raise ValueError("selected file must have a safe .tar.zst filename")
        if not 1 <= content_length <= self.max_upload_bytes:
            raise ValueError(
                f"upload size must be in [1,{self.max_upload_bytes}] bytes"
            )
        expected = (expected_sha256 or "").strip().lower()
        if expected and (
            len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise ValueError("expected SHA-256 must contain exactly 64 hex characters")

        self._set_operation(f"uploading local archive {safe_name}")
        self.upload_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.upload_dir, 0o700)
        upload_id = uuid.uuid4().hex
        partial = self.upload_dir / f".{upload_id}.partial"
        archive = self.upload_dir / f"{upload_id}-{safe_name}"
        digest = hashlib.sha256()
        remaining = content_length
        try:
            with partial.open("xb") as target:
                while remaining:
                    chunk = stream.read(min(8 * 1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("upload ended before Content-Length bytes")
                    target.write(chunk)
                    digest.update(chunk)
                    remaining -= len(chunk)
            actual = digest.hexdigest()
            if expected and actual != expected:
                raise ValueError(
                    f"archive SHA-256 mismatch: expected {expected}, got {actual}"
                )
            partial.replace(archive)
        except Exception as error:
            partial.unlink(missing_ok=True)
            archive.unlink(missing_ok=True)
            self._fail(error)
            raise RuntimeError(self._last_error) from None

        self._last_error = None
        self._set_operation("upload complete; loading archive into local Docker")
        job = self._start_archive_load_job(
            str(archive),
            actual,
            cleanup_path=archive,
            extra={
                "uploaded_filename": safe_name,
                "upload_size_bytes": content_length,
                "sha256": actual,
            },
        )
        return {
            **job,
            "uploaded_filename": safe_name,
            "upload_size_bytes": content_length,
            "sha256": actual,
        }

    def start_policy(self, image: str) -> dict[str, Any]:
        with self._lock:
            self._set_operation("starting isolated policy")
            self._close_policy()
            session_id = f"local-shadow-{uuid.uuid4().hex[:12]}"
            host_session = None
            try:
                gateway_endpoint, router_connect_endpoint = (
                    self.runtime.prepare_gateway_ipc()
                )
                host_session = self.router_session_factory(gateway_endpoint)
                self.runtime.grant_gateway_ipc_access()
                container = self.runtime.start(
                    image,
                    session_id=session_id,
                    gateway_endpoint=router_connect_endpoint,
                )
                deadline = time.monotonic() + self.startup_timeout_s
                last_error: Exception | None = None
                policy = None
                while time.monotonic() <= deadline:
                    try:
                        policy = self.policy_factory(
                            gateway_endpoint,
                            session_id=session_id,
                            timeout_s=min(self.policy_timeout_s, 10.0),
                            session=host_session,
                        )
                        # Keep readiness probes short, then allow the first model
                        # inference enough time for JAX compilation/warm-up.
                        if hasattr(policy, "timeout_s"):
                            policy.timeout_s = self.policy_timeout_s
                        break
                    except Exception as error:
                        last_error = error
                        if not self.runtime.status().running:
                            break
                        time.sleep(1)
                if policy is None:
                    raise TimeoutError(
                        f"policy metadata readiness timed out: {last_error or 'no reply'}"
                    )
                policy.reset()
            except Exception as error:
                self.runtime.stop()
                if host_session is not None:
                    host_session.close()
                self._fail(error)
                raise RuntimeError(self._last_error) from None
            self._policy = policy
            self._host_zenoh_session = host_session
            self._policy_session = session_id
            self._image = image
            self._trajectory = None
            self._last_error = None
            self._set_operation("policy ready")
            return {
                "container": container.to_dict()
                if hasattr(container, "to_dict")
                else _jsonable(container),
                "session_id": session_id,
                "endpoint": gateway_endpoint,
                "metadata": self._metadata_dict(policy),
            }

    def stop_policy(self) -> dict[str, bool]:
        with self._lock:
            self._set_operation("stopping policy")
            self._close_policy()
            self.runtime.stop()
            self._policy_session = None
            self._image = None
            self._trajectory = None
            self._last_error = None
            self._set_operation("policy stopped")
            return {"ok": True}

    def reset_policy(self) -> dict[str, bool]:
        with self._lock:
            policy = self._require_policy()
            self._set_operation("resetting policy")
            try:
                policy.reset()
            except Exception as error:
                self._fail(error)
                raise RuntimeError(self._last_error) from None
            self._trajectory = None
            self._last_error = None
            self._set_operation("policy reset")
            return {"ok": True}

    def shadow(self, *, preview_steps: int = 100, control_hz: float = 30.0) -> dict[str, Any]:
        if type(preview_steps) is not int or not 1 <= preview_steps <= 100:
            raise ValueError("preview_steps must be an integer in [1,100]")
        if not math.isfinite(float(control_hz)) or not 0 < float(control_hz) <= 240:
            raise ValueError("control_hz must be finite and in (0,240]")
        with self._lock:
            remote = self._require_remote()
            policy = self._require_policy()
            self._trajectory = None
            self._set_operation("running shadow inference")
            started = time.monotonic()
            try:
                observation = self._get_remote_observation_with_retry(remote)
                self._validate_observation(observation)
                current = observation["observation/state"].copy()
                rollout_observation = dict(observation)
                chunks: list[np.ndarray] = []
                chunk_latencies = []
                chunk_metrics = []
                remaining = preview_steps
                while remaining:
                    chunk_started = time.monotonic()
                    actions, metrics = policy.infer(rollout_observation)
                    chunk_latencies.append((time.monotonic() - chunk_started) * 1000)
                    actions = self._validate_chunk(actions, policy)
                    take = min(remaining, actions.shape[0])
                    chunks.append(np.ascontiguousarray(actions[:take]))
                    chunk_metrics.append(_jsonable(metrics))
                    remaining -= take
                    rollout_observation["observation/state"] = np.ascontiguousarray(
                        actions[take - 1],
                        dtype=np.float32,
                    )
                prediction = np.ascontiguousarray(
                    np.concatenate(chunks, axis=0),
                    dtype=np.float32,
                )
                validation = self.trajectory_validator.validate(
                    current,
                    prediction,
                    control_hz=control_hz,
                )
            except Exception as error:
                self._fail(error)
                raise RuntimeError(self._last_error) from None
            result = {
                "shadow_only": True,
                "preview_steps": int(prediction.shape[0]),
                "chunk_count": len(chunks),
                "chunk_latency_ms": [round(value, 1) for value in chunk_latencies],
                "latency_ms": round((time.monotonic() - started) * 1000, 1),
                "current_state": current.tolist(),
                "prediction": prediction.tolist(),
                "action_shape": list(prediction.shape),
                "action_min": float(prediction.min()),
                "action_max": float(prediction.max()),
                "metrics": chunk_metrics,
                "validation": validation,
                "compatible": bool(validation["compatible"]),
                "metadata": self._metadata_dict(policy),
                "remote_observation_timestamp": getattr(
                    remote,
                    "last_observation_timestamp",
                    None,
                ),
            }
            self._observation = observation
            self._trajectory = result
            self._last_error = None
            self._set_operation(f"shadow complete ({prediction.shape[0]} steps)")
            return result

    @staticmethod
    def _get_remote_observation_with_retry(remote: Any) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                return remote.get_observation()
            except Exception as error:
                last_error = error
                message = str(error)
                retryable = any(
                    marker in message
                    for marker in (
                        "Zenoh returned an error reply",
                        "RATE_LIMITED",
                        "BUSY",
                        "OBSERVATION_UNAVAILABLE",
                        "no observation reply",
                    )
                )
                if not retryable or attempt == 2:
                    raise
                # A Zenoh route can remain transiently unavailable immediately
                # after a router or policy restart; short immediate retries repeat
                # the same transport error before route discovery settles.
                time.sleep(1.0)
        assert last_error is not None
        raise last_error

    def status(self) -> dict[str, Any]:
        with self._lock:
            try:
                container = self.runtime.status()
                container_value = (
                    container.to_dict()
                    if hasattr(container, "to_dict")
                    else _jsonable(container)
                )
            except Exception as error:
                container_value = {"present": False, "running": False, "error": str(error)}
            return {
                "shadow_only": True,
                "read_only_remote": True,
                "last_operation": self._last_operation,
                "last_error": self._last_error,
                "remote": self.remote_status(),
                "policy": {
                    "connected": self._policy is not None,
                    "image": self._image,
                    "session_id": self._policy_session,
                    "metadata": (
                        self._metadata_dict(self._policy)
                        if self._policy is not None
                        else None
                    ),
                },
                "container": container_value,
                "observation_available": self._observation is not None,
                "trajectory_available": self._trajectory is not None,
                "archive_job": self._public_archive_job(),
                "events": list(self._events[-100:]),
            }

    def _public_archive_job(self) -> dict[str, Any] | None:
        with self._archive_state_lock:
            if self._archive_job is None:
                return None
            return {
                "job_id": self._archive_job["job_id"],
                "status": self._archive_job["status"],
                "uploaded_filename": self._archive_job.get("uploaded_filename"),
                "upload_size_bytes": self._archive_job.get("upload_size_bytes"),
                "sha256": self._archive_job.get("sha256"),
                "error": self._archive_job.get("error"),
            }

    def _start_archive_load_job(
        self,
        archive_path: str,
        expected_sha256: str,
        *,
        cleanup_path: pathlib.Path | None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._archive_state_lock:
            if self._archive_job is not None and self._archive_job["status"] == "running":
                raise RuntimeError("another archive load is already running")
            job_id = uuid.uuid4().hex
            self._archive_job = {
                "job_id": job_id,
                "status": "running",
                "archive_path": archive_path,
                "result": None,
                "error": None,
                **(extra or {}),
            }

        def worker() -> None:
            try:
                self._set_operation("loading image archive")
                result = self.runtime.load_archive(
                    archive_path,
                    expected_sha256=expected_sha256,
                )
                public_result = dict(result)
                if extra:
                    public_result.update(extra)
                with self._archive_state_lock:
                    assert self._archive_job is not None
                    self._archive_job["status"] = "completed"
                    self._archive_job["result"] = public_result
                self._last_error = None
                self._set_operation("image archive loaded")
            except Exception as error:
                with self._archive_state_lock:
                    assert self._archive_job is not None
                    self._archive_job["status"] = "failed"
                    self._archive_job["error"] = str(error)
                self._fail(error)
            finally:
                if cleanup_path is not None:
                    cleanup_path.unlink(missing_ok=True)

        threading.Thread(
            target=worker,
            name=f"participant-archive-load-{job_id[:8]}",
            daemon=True,
        ).start()
        return {"job_id": job_id, "status": "running"}

    def remote_status(self) -> dict[str, Any]:
        return {
            "connected": self._remote is not None,
            "endpoint": self._remote_endpoint,
            "session_id": self._remote_session,
            "token_configured": self._remote_token is not None,
        }

    def observation(self) -> dict[str, Any]:
        with self._lock:
            if self._observation is None:
                raise RuntimeError("no remote observation is available")
            return {
                "state": self._observation["observation/state"].tolist(),
                "prompt": self._observation["prompt"],
                "joint_names": list(JOINT_NAMES),
            }

    def trajectory(self) -> dict[str, Any]:
        with self._lock:
            if self._trajectory is None:
                raise RuntimeError("no Shadow trajectory is available")
            return self._trajectory

    def image_jpeg(self, key: str) -> bytes:
        with self._lock:
            if self._observation is None:
                raise RuntimeError("no remote observation is available")
            image = self._observation[key]
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise RuntimeError("could not encode observation image")
        return encoded.tobytes()

    def logs(self) -> dict[str, Any]:
        with self._lock:
            return {
                "events": list(self._events[-200:]),
                "container": self.runtime.logs() if self.runtime.status().present else "",
            }

    def close(self) -> None:
        with self._lock:
            self._close_policy()
            self.runtime.stop()
            if self._remote is not None:
                self._remote.close()
                self._remote = None
            self._remote_token = None

    def _close_policy(self) -> None:
        if self._policy is not None:
            self._policy.close()
            self._policy = None
        if self._host_zenoh_session is not None:
            self._host_zenoh_session.close()
            self._host_zenoh_session = None

    def _require_remote(self) -> Any:
        if self._remote is None:
            raise RuntimeError("configure the remote observation connection first")
        return self._remote

    def _require_policy(self) -> Any:
        if self._policy is None:
            raise RuntimeError("start a compatible local policy image first")
        return self._policy

    def _fail(self, error: Exception, *, token: str | None = None) -> None:
        message = str(error)
        for secret in (token, self._remote_token):
            if secret:
                message = message.replace(secret, "<redacted>")
        self._last_error = message
        self._set_operation("operation failed")
        self._events.append(f"ERROR {message}")

    def _set_operation(self, value: str) -> None:
        self._last_operation = value
        self._events.append(value)
        del self._events[:-200]

    @staticmethod
    def _validate_observation(observation: Any) -> None:
        if not isinstance(observation, Mapping):
            raise ValueError("remote observation must be an object")
        required = {*REQUIRED_IMAGE_SPECS, *VECTOR_SPECS, "prompt"}
        allowed = required | set(OPTIONAL_IMAGE_SPECS)
        if not required.issubset(observation) or not set(observation).issubset(allowed):
            raise ValueError("remote observation keys do not match the public infer schema")
        for key, shape in {**REQUIRED_IMAGE_SPECS, **OPTIONAL_IMAGE_SPECS}.items():
            if key not in observation:
                continue
            image = observation[key]
            if (
                not isinstance(image, np.ndarray)
                or image.dtype != np.dtype(np.uint8)
                or image.shape != shape
            ):
                raise ValueError(f"{key} must be uint8{shape}")
        for key, shape in VECTOR_SPECS.items():
            value = observation[key]
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != np.dtype(np.float32)
                or value.shape != shape
                or not np.isfinite(value).all()
            ):
                raise ValueError(f"{key} must be finite float32{shape}")
        if not isinstance(observation["prompt"], str):
            raise ValueError("prompt must be a string")

    @staticmethod
    def _validate_chunk(actions: Any, policy: Any) -> np.ndarray:
        if not isinstance(actions, np.ndarray):
            raise ValueError("policy actions must decode to a NumPy array")
        if (
            actions.dtype != np.dtype(np.float32)
            or actions.ndim != 2
            or actions.shape[0] < 1
            or actions.shape[1] != ACTION_DIM
        ):
            raise ValueError(
                f"policy actions must be float32[T,65] with T>=1, got "
                f"{actions.dtype}{list(actions.shape)}"
            )
        horizon = getattr(getattr(policy, "metadata", None), "action_horizon", None)
        if horizon is not None and actions.shape[0] != horizon:
            raise ValueError(
                f"policy actions horizon {actions.shape[0]} does not match metadata {horizon}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("policy actions contain NaN or Inf")
        return np.ascontiguousarray(actions)

    @staticmethod
    def _metadata_dict(policy: Any) -> dict[str, Any] | None:
        metadata = getattr(policy, "metadata", None)
        if metadata is None:
            return None
        if hasattr(metadata, "as_dict"):
            return metadata.as_dict()
        if isinstance(metadata, Mapping):
            return _jsonable(metadata)
        return _jsonable(vars(metadata))


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return value
