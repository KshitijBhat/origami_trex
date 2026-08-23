"""Loopback HTTP server for the participant local Shadow evaluator."""

from __future__ import annotations

import json
import mimetypes
import pathlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from .contract import CAMERA_IMAGE_KEYS
from .controller import LocalEvaluatorController

IMAGE_ROUTES = {
    "/api/image/head-left": CAMERA_IMAGE_KEYS[0],
    "/api/image/head-right": CAMERA_IMAGE_KEYS[1],
    "/api/image/left-wrist": CAMERA_IMAGE_KEYS[2],
    "/api/image/right-wrist": CAMERA_IMAGE_KEYS[3],
}
POST_ROUTES = frozenset(
    {
        "/api/remote/connect",
        "/api/submission/upload",
        "/api/submission/load",
        "/api/submission/start",
        "/api/submission/stop",
        "/api/policy/reset",
        "/api/policy/shadow",
    }
)
STATIC_SUFFIXES = {".html", ".js", ".css", ".svg"}
ROBOT_ASSET_SUFFIXES = {
    ".urdf",
    ".stl",
    ".dae",
    ".obj",
    ".mtl",
    ".png",
    ".jpg",
    ".jpeg",
}


def make_handler(
    controller: LocalEvaluatorController,
    *,
    static_root: pathlib.Path,
    robot_assets_root: pathlib.Path,
) -> type[BaseHTTPRequestHandler]:
    static_root = static_root.resolve()
    robot_assets_root = robot_assets_root.resolve()

    class LocalEvaluatorHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            try:
                if path in {"/", "/index.html"}:
                    self._send_file(static_root, "index.html", STATIC_SUFFIXES)
                elif path == "/api/status":
                    self._send_json(HTTPStatus.OK, controller.status())
                elif path == "/api/observation":
                    self._send_json(HTTPStatus.OK, controller.observation())
                elif path == "/api/trajectory":
                    self._send_json(HTTPStatus.OK, controller.trajectory())
                elif path == "/api/robot/config":
                    self._send_json(
                        HTTPStatus.OK,
                        controller.trajectory_validator.robot_config(),
                    )
                elif path == "/api/logs":
                    self._send_json(HTTPStatus.OK, controller.logs())
                elif path == "/api/submission/load/status":
                    query = parse_qs(urlsplit(self.path).query)
                    job_id = str((query.get("job_id") or [""])[0])
                    self._send_json(
                        HTTPStatus.OK,
                        controller.archive_load_status(job_id),
                    )
                elif path in IMAGE_ROUTES:
                    self._send(
                        HTTPStatus.OK,
                        controller.image_jpeg(IMAGE_ROUTES[path]),
                        "image/jpeg",
                    )
                elif path.startswith("/static/"):
                    self._send_file(
                        static_root,
                        path.removeprefix("/static/"),
                        STATIC_SUFFIXES,
                    )
                elif path.startswith("/robot-assets/"):
                    self._send_file(
                        robot_assets_root,
                        path.removeprefix("/robot-assets/"),
                        ROBOT_ASSET_SUFFIXES,
                    )
                else:
                    self._send_json(
                        HTTPStatus.NOT_FOUND,
                        {"ok": False, "error": "not found"},
                    )
            except Exception as error:
                self._send_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {"ok": False, "error": str(error)},
                )

        def do_POST(self) -> None:
            path = urlsplit(self.path).path
            if path not in POST_ROUTES:
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"ok": False, "error": "POST route is not allowed"},
                )
                return
            try:
                if path == "/api/submission/upload":
                    result = self._read_archive_upload()
                    self._send_json(HTTPStatus.OK, result)
                    return
                payload = self._read_json()
                if path == "/api/remote/connect":
                    _require_only(
                        payload,
                        {
                            "endpoint",
                            "session_id",
                            "token",
                            "tls_ca",
                            "tls_certificate",
                            "tls_private_key",
                        },
                    )
                    result = controller.connect_remote(
                        endpoint=_required_string(payload, "endpoint"),
                        session_id=_required_string(payload, "session_id"),
                        token=_secret_string(payload, "token"),
                        tls_root_ca_certificate=_optional_string(payload, "tls_ca"),
                        tls_client_certificate=_optional_string(payload, "tls_certificate"),
                        tls_client_private_key=_optional_string(payload, "tls_private_key"),
                    )
                elif path == "/api/submission/load":
                    _require_only(payload, {"archive_path", "sha256"})
                    result = controller.start_archive_load(
                        _required_string(payload, "archive_path"),
                        _required_string(payload, "sha256"),
                    )
                elif path == "/api/submission/start":
                    _require_only(payload, {"image"})
                    result = controller.start_policy(_required_string(payload, "image"))
                elif path == "/api/submission/stop":
                    _require_empty(payload)
                    result = controller.stop_policy()
                elif path == "/api/policy/reset":
                    _require_empty(payload)
                    result = controller.reset_policy()
                else:
                    _require_only(payload, {"preview_steps", "control_hz"})
                    result = controller.shadow(
                        preview_steps=_integer(payload, "preview_steps", 100),
                        control_hz=_number(payload, "control_hz", 30.0),
                    )
                self._send_json(HTTPStatus.OK, result)
            except Exception as error:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"ok": False, "error": str(error)},
                )

        def _read_archive_upload(self) -> dict[str, Any]:
            content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip()
            if content_type != "application/octet-stream":
                raise ValueError(
                    "archive upload requires Content-Type: application/octet-stream"
                )
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as error:
                raise ValueError("invalid Content-Length") from error
            filename = unquote(self.headers.get("X-Origami-Filename", ""))
            expected_sha256 = self.headers.get("X-Origami-Sha256", "")
            return controller.upload_archive(
                self.rfile,
                filename=filename,
                content_length=length,
                expected_sha256=expected_sha256,
            )

        def do_PUT(self) -> None:
            self._method_not_allowed()

        def do_PATCH(self) -> None:
            self._method_not_allowed()

        def do_DELETE(self) -> None:
            self._method_not_allowed()

        def _method_not_allowed(self) -> None:
            self._send_json(
                HTTPStatus.METHOD_NOT_ALLOWED,
                {"ok": False, "error": "method not allowed"},
            )

        def _read_json(self) -> dict[str, Any]:
            content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip()
            if content_type != "application/json":
                raise ValueError("POST requires Content-Type: application/json")
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as error:
                raise ValueError("invalid Content-Length") from error
            if not 0 <= length <= 64 * 1024:
                raise ValueError("request body exceeds 64 KiB")
            value = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(value, dict):
                raise ValueError("request body must be a JSON object")
            return value

        def _send_file(
            self,
            root: pathlib.Path,
            relative_url: str,
            allowed_suffixes: set[str],
        ) -> None:
            relative = pathlib.PurePosixPath(unquote(relative_url))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("invalid asset path")
            target = (root / pathlib.Path(*relative.parts)).resolve()
            try:
                target.relative_to(root)
            except ValueError as error:
                raise ValueError("asset path escapes configured root") from error
            if target.suffix.lower() not in allowed_suffixes or not target.is_file():
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"ok": False, "error": "asset not found"},
                )
                return
            content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            self._send(HTTPStatus.OK, target.read_bytes(), content_type)

        def _send_json(self, status: HTTPStatus, value: Any) -> None:
            self._send(
                status,
                json.dumps(
                    value,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode(),
                "application/json; charset=utf-8",
            )

        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            try:
                self.send_response(status.value)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("X-Frame-Options", "DENY")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'self'; script-src 'self'; style-src 'self'; "
                    "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'",
                )
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                return

        def log_message(self, format: str, *args: Any) -> None:
            return

    return LocalEvaluatorHandler


def create_server(
    controller: LocalEvaluatorController,
    *,
    host: str = "127.0.0.1",
    port: int = 7861,
    static_root: pathlib.Path | None = None,
    robot_assets_root: pathlib.Path | None = None,
) -> ThreadingHTTPServer:
    package_root = pathlib.Path(__file__).resolve().parent
    handler = make_handler(
        controller,
        static_root=(static_root or package_root / "static"),
        robot_assets_root=(
            robot_assets_root or controller.trajectory_validator.assets_root
        ),
    )
    return ThreadingHTTPServer((host, port), handler)


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _optional_string(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _secret_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _integer(payload: dict[str, Any], key: str, default: int) -> int:
    value = payload.get(key, default)
    if type(value) is not int:
        raise ValueError(f"{key} must be an integer")
    return value


def _number(payload: dict[str, Any], key: str, default: float) -> float:
    value = payload.get(key, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be a number")
    return float(value)


def _require_empty(payload: dict[str, Any]) -> None:
    if payload:
        raise ValueError("this endpoint does not accept fields")


def _require_only(payload: dict[str, Any], allowed: set[str]) -> None:
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"unknown fields: {sorted(unknown)}")
