"""Hardened local Docker runner for one untrusted participant policy."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import secrets
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

DEFAULT_ROUTER_IMAGE = (
    "eclipse/zenoh@sha256:"
    "157965d71e0bfd0a044d76a985ff0e5c306ad3968929168fb9678cd2a7fec23f"
)
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


@dataclass(frozen=True)
class ContainerStatus:
    present: bool
    running: bool
    health: str
    image: str | None
    image_id: str | None
    started_at: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DockerRuntime:
    """Create an internal network, trusted router, and sandboxed policy."""

    def __init__(
        self,
        *,
        container_name: str = "origami-local-policy",
        router_name: str = "origami-local-router",
        network_name: str = "origami-local-shadow",
        router_image: str = DEFAULT_ROUTER_IMAGE,
        router_port: int = 7447,
        memory: str = "32g",
        cpus: str = "8",
        shm_size: str = "8g",
        tmpfs_size: str = "4g",
        pids_limit: int = 512,
        gpus: str | None = "all",
        ipc_dir: str | pathlib.Path | None = None,
        runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        for label, value in (
            ("container", container_name),
            ("router", router_name),
            ("network", network_name),
        ):
            if _SAFE_NAME.fullmatch(value) is None:
                raise ValueError(f"invalid {label} name: {value!r}")
        if container_name == router_name:
            raise ValueError("container and router names must differ")
        if not 1 <= router_port <= 65535:
            raise ValueError("router_port must be in [1,65535]")
        for label, value in (
            ("router_image", router_image),
            ("memory", memory),
            ("cpus", cpus),
            ("shm_size", shm_size),
            ("tmpfs_size", tmpfs_size),
        ):
            _nonblank(value, label)
        if gpus is not None:
            _nonblank(gpus, "gpus")
        if pids_limit < 1:
            raise ValueError("pids_limit must be positive")
        self.container_name = container_name
        self.router_name = router_name
        self.network_name = network_name
        self.router_image = router_image
        self.router_port = int(router_port)
        self.memory = memory
        self.cpus = cpus
        self.shm_size = shm_size
        self.tmpfs_size = tmpfs_size
        self.pids_limit = int(pids_limit)
        self.gpus = gpus
        self.ipc_dir = pathlib.Path(
            ipc_dir
            or (
                pathlib.Path(tempfile.gettempdir())
                / f"{container_name}-zenoh-ipc-{secrets.token_hex(8)}"
            )
        ).resolve()
        self._runner = runner or subprocess.run

    @property
    def host_endpoint(self) -> str:
        return f"unixsock-stream/{self.ipc_dir / 'gateway.sock'}"

    def prepare_gateway_ipc(self) -> tuple[str, str]:
        if self.ipc_dir.is_symlink():
            raise RuntimeError("local evaluator IPC path must not be a symlink")
        self.ipc_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not self.ipc_dir.is_dir():
            raise RuntimeError("local evaluator IPC path must be a directory")
        os.chmod(self.ipc_dir, 0o700)
        socket_path = self.ipc_dir / "gateway.sock"
        for stale_path in (socket_path, pathlib.Path(f"{socket_path}.lock")):
            stale_path.unlink(missing_ok=True)
        return (
            f"unixsock-stream/{socket_path}",
            "unixsock-stream//origami-ipc/gateway.sock",
        )

    def grant_gateway_ipc_access(self) -> None:
        socket_path = self.ipc_dir / "gateway.sock"
        if not socket_path.exists():
            raise RuntimeError("trusted local evaluator IPC socket was not created")
        if socket_path.is_symlink():
            raise RuntimeError("trusted local evaluator IPC socket must not be a symlink")
        os.chmod(socket_path, 0o600)

    def start(
        self,
        image: str,
        *,
        session_id: str,
        gateway_endpoint: str,
    ) -> ContainerStatus:
        _nonblank(image, "image")
        if _SAFE_NAME.fullmatch(session_id) is None:
            raise ValueError("session_id contains unsafe characters")
        if not isinstance(gateway_endpoint, str) or not gateway_endpoint.startswith(
            "unixsock-stream/"
        ):
            raise ValueError("gateway_endpoint must use unixsock-stream/")
        self.stop()
        try:
            self._docker(
                [
                    "network",
                    "create",
                    "--internal",
                    "--label",
                    "org.origami.role=participant-local-shadow",
                    "--label",
                    f"org.origami.session={session_id}",
                    self.network_name,
                ]
            )
            self._docker(self._router_command(session_id, gateway_endpoint))
            self._docker(self._policy_command(image, session_id))
        except Exception:
            self.stop()
            raise
        status = self.status()
        if not status.running:
            logs = self.logs()
            self.stop()
            raise RuntimeError(f"policy container exited during startup\n{logs}")
        return status

    def stop(self) -> None:
        for name in (self.container_name, self.router_name):
            self._docker(["rm", "--force", name], check=False)
        self._docker(["network", "rm", self.network_name], check=False)

    def status(self) -> ContainerStatus:
        completed = self._docker(
            ["container", "inspect", self.container_name],
            check=False,
        )
        if completed.returncode:
            return ContainerStatus(False, False, "absent", None, None, None)
        try:
            value = json.loads(completed.stdout)[0]
        except (IndexError, TypeError, json.JSONDecodeError) as error:
            raise RuntimeError("docker inspect returned malformed JSON") from error
        state = value.get("State", {})
        return ContainerStatus(
            present=True,
            running=bool(state.get("Running")),
            health=str(state.get("Health", {}).get("Status", "none")),
            image=value.get("Config", {}).get("Image"),
            image_id=value.get("Image"),
            started_at=state.get("StartedAt"),
        )

    def logs(self, *, tail: int = 300) -> str:
        completed = self._docker(
            ["logs", "--tail", str(max(1, min(int(tail), 5000))), self.container_name],
            check=False,
        )
        return (completed.stdout + completed.stderr)[-100_000:]

    def load_archive(
        self,
        archive_path: str | pathlib.Path,
        *,
        expected_sha256: str | None,
    ) -> dict[str, Any]:
        archive = pathlib.Path(archive_path).expanduser().resolve()
        if not archive.is_file():
            raise ValueError(f"archive does not exist: {archive}")
        if not archive.name.endswith(".tar.zst"):
            raise ValueError("archive path must end with .tar.zst")
        if not expected_sha256:
            raise ValueError("expected SHA-256 is required when loading an archive")
        expected = expected_sha256.strip().lower()
        if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError("expected SHA-256 must contain exactly 64 hex characters")

        digest = hashlib.sha256()
        with archive.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected:
            raise ValueError(f"archive SHA-256 mismatch: expected {expected}, got {actual}")

        integrity = self._run(["zstd", "-t", str(archive)], timeout=1800, check=False)
        if integrity.returncode:
            raise RuntimeError(f"zstd integrity check failed: {integrity.stderr.strip()}")
        decompressor = subprocess.Popen(
            ["zstd", "-dc", str(archive)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert decompressor.stdout is not None
        try:
            loaded = subprocess.run(
                ["docker", "load"],
                stdin=decompressor.stdout,
                capture_output=True,
                text=True,
                timeout=1800,
                check=False,
            )
        finally:
            decompressor.stdout.close()
        stderr = (
            decompressor.stderr.read().decode(errors="replace")
            if decompressor.stderr is not None
            else ""
        )
        decompressor_status = decompressor.wait(timeout=30)
        if decompressor_status:
            raise RuntimeError(f"zstd decompression failed: {stderr.strip()}")
        if loaded.returncode:
            raise RuntimeError(f"docker load failed: {loaded.stderr.strip()}")
        output = (loaded.stdout + loaded.stderr).strip()
        images = [
            line.split(":", 1)[1].strip()
            for line in output.splitlines()
            if line.startswith("Loaded image:")
        ]
        return {
            "archive": str(archive),
            "sha256": actual,
            "images": images,
            "output": output[-10_000:],
        }

    def _router_command(self, session_id: str, gateway_endpoint: str) -> list[str]:
        return [
            "run",
            "--detach",
            "--name",
            self.router_name,
            "--network",
            self.network_name,
            "--network-alias",
            self.router_name,
            "--label",
            "org.origami.role=trusted-local-router",
            "--label",
            f"org.origami.session={session_id}",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges=true",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,size=64m",
            "--tmpfs",
            "/run:rw,noexec,nosuid,nodev,size=16m",
            "--mount",
            f"type=bind,src={self.ipc_dir},dst=/origami-ipc",
            "--pids-limit",
            "128",
            "--memory",
            "512m",
            "--cpus",
            "1",
            "--restart",
            "no",
            self.router_image,
            "-l",
            f"tcp/0.0.0.0:{self.router_port}",
            "--no-multicast-scouting",
            "--cfg",
            "transport/shared_memory/enabled:false",
            "-e",
            gateway_endpoint,
        ]

    def _policy_command(self, image: str, session_id: str) -> list[str]:
        command = [
            "run",
            "--detach",
            "--name",
            self.container_name,
            "--network",
            self.network_name,
            "--label",
            "org.origami.role=participant-policy-shadow",
            "--label",
            f"org.origami.session={session_id}",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges=true",
            "--user",
            "65532:65532",
            "--tmpfs",
            f"/tmp:rw,noexec,nosuid,nodev,size={self.tmpfs_size}",
            "--tmpfs",
            "/run:rw,noexec,nosuid,nodev,size=64m",
            "--pids-limit",
            str(self.pids_limit),
            "--memory",
            self.memory,
            "--cpus",
            self.cpus,
            "--shm-size",
            self.shm_size,
            "--restart",
            "no",
            "--env",
            f"ORIGAMI_ZENOH_ENDPOINT=tcp/{self.router_name}:{self.router_port}",
            "--env",
            f"ORIGAMI_SESSION_ID={session_id}",
        ]
        if self.gpus is not None:
            command.extend(["--gpus", self.gpus])
        command.append(image)
        return command

    def _docker(
        self,
        arguments: Sequence[str],
        *,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        return self._run(["docker", *arguments], timeout=1200, check=check)

    def _run(
        self,
        command: Sequence[str],
        *,
        timeout: float,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        completed = self._runner(
            list(command),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if check and completed.returncode:
            raise RuntimeError(
                f"command failed ({completed.returncode}): {' '.join(command)}\n"
                f"{completed.stderr.strip()}"
            )
        return completed


def _nonblank(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value.startswith("-")
        or any(character.isspace() for character in value)
    ):
        raise ValueError(f"{label} must be non-empty, option-safe, and contain no whitespace")
