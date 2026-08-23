from __future__ import annotations

import json
import hashlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

sys.dont_write_bytecode = True
SDK_ROOT = Path(__file__).parents[1]
if str(SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(SDK_ROOT))

from participant_local_evaluator.contract import (  # noqa: E402
    JOINT_NAMES,
    OPTIONAL_IMAGE_SPECS,
    PolicyMetadata,
    REQUIRED_IMAGE_SPECS,
    VECTOR_SPECS,
)
from participant_local_evaluator import __main__ as evaluator_main  # noqa: E402
from participant_local_evaluator.controller import LocalEvaluatorController  # noqa: E402
from participant_local_evaluator.docker_runtime import (  # noqa: E402
    ContainerStatus,
    DockerRuntime,
)
from participant_local_evaluator.trajectory import TrajectoryValidator  # noqa: E402
from participant_local_evaluator.web import POST_ROUTES, create_server  # noqa: E402


def observation() -> dict:
    return {
        **{
            key: np.full(shape, index * 10, dtype=np.uint8)
            for index, (key, shape) in enumerate(
                {**REQUIRED_IMAGE_SPECS, **OPTIONAL_IMAGE_SPECS}.items()
            )
        },
        **{
            key: np.zeros(shape, dtype=np.float32)
            for key, shape in VECTOR_SPECS.items()
        },
        "prompt": "fold the paper",
    }


class FakeRemote:
    def __init__(self, endpoint, *, session_id, token, fail=False, **kwargs):
        self.endpoint = endpoint
        self.session_id = session_id
        self.token = token
        self.fail = fail
        self.closed = False
        self.calls = 0
        self.last_observation_timestamp = None

    def get_observation(self):
        if self.fail:
            raise RuntimeError(f"credential rejected: {self.token}")
        self.calls += 1
        self.last_observation_timestamp = time.time()
        return observation()

    def close(self):
        self.closed = True


class FakePolicy:
    def __init__(self, endpoint, *, session_id, timeout_s, session=None):
        self.endpoint = endpoint
        self.session_id = session_id
        self.timeout_s = timeout_s
        self.session = session
        self.metadata = PolicyMetadata(action_horizon=25)
        self.infer_calls = 0
        self.reset_calls = 0
        self.closed = False

    def reset(self):
        self.reset_calls += 1

    def infer(self, value):
        self.infer_calls += 1
        state = value["observation/state"]
        increments = np.arange(1, 26, dtype=np.float32)[:, None] * np.float32(0.001)
        return np.ascontiguousarray(state[None, :] + increments), {"chunk": self.infer_calls}

    def close(self):
        self.closed = True


class FakeRuntime:
    host_endpoint = "unixsock-stream//tmp/fake-gateway.sock"

    def __init__(self):
        self.running = False
        self.image = None
        self.session_id = None
        self.stops = 0

    def prepare_gateway_ipc(self):
        return (
            "unixsock-stream//tmp/fake-gateway.sock",
            "unixsock-stream//origami-ipc/gateway.sock",
        )

    def grant_gateway_ipc_access(self):
        return None

    def start(self, image, *, session_id, gateway_endpoint):
        self.running = True
        self.image = image
        self.session_id = session_id
        self.gateway_endpoint = gateway_endpoint
        return self.status()

    def stop(self):
        self.running = False
        self.stops += 1

    def status(self):
        return ContainerStatus(
            present=self.running,
            running=self.running,
            health="none",
            image=self.image,
            image_id="sha256:fake" if self.running else None,
            started_at="now" if self.running else None,
        )

    def logs(self, *, tail=300):
        return "fake policy logs"

    def load_archive(self, archive_path, *, expected_sha256):
        return {
            "archive": archive_path,
            "sha256": expected_sha256,
            "images": ["team:test"],
            "output": "Loaded image: team:test",
        }


class FakeSession:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def make_controller(root: Path, *, remote_factory=FakeRemote):
    return LocalEvaluatorController(
        FakeRuntime(),
        TrajectoryValidator(root),
        remote_factory=remote_factory,
        policy_factory=FakePolicy,
        router_session_factory=lambda endpoint: FakeSession(),
        startup_timeout_s=0.1,
    )


class ControllerTests(unittest.TestCase):
    def test_transient_remote_zenoh_error_is_retried(self):
        class TransientRemote:
            def __init__(self):
                self.calls = 0

            def get_observation(self):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("Zenoh returned an error reply")
                return observation()

        remote = TransientRemote()
        result = LocalEvaluatorController._get_remote_observation_with_retry(remote)
        self.assertEqual(result["observation/state"].shape, (65,))
        self.assertEqual(remote.calls, 2)

    def test_browser_archive_upload_streams_hashes_loads_and_deletes_temp_file(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = LocalEvaluatorController(
                FakeRuntime(),
                TrajectoryValidator(directory),
                remote_factory=FakeRemote,
                policy_factory=FakePolicy,
                router_session_factory=lambda endpoint: FakeSession(),
                upload_dir=Path(directory) / "uploads",
            )
            payload = b"small fake zstd archive for streaming test"
            expected = hashlib.sha256(payload).hexdigest()
            job = controller.upload_archive(
                io.BytesIO(payload),
                filename="team-submission.tar.zst",
                content_length=len(payload),
                expected_sha256=expected,
            )
            self.assertEqual(job["status"], "running")
            self.assertEqual(job["sha256"], expected)
            result = controller.wait_archive_load(job["job_id"], timeout_s=5.0)
            self.assertEqual(result["sha256"], expected)
            self.assertEqual(result["images"], ["team:test"])
            self.assertEqual(list((Path(directory) / "uploads").iterdir()), [])

    def test_multi_chunk_shadow_is_exactly_100_steps_and_local_only(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = make_controller(Path(directory))
            controller.connect_remote(
                endpoint="tcp/example.test:7448",
                session_id="team-a",
                token="top-secret",
            )
            controller.start_policy("team:test")
            result = controller.shadow(preview_steps=100)

            self.assertEqual(result["action_shape"], [100, 65])
            self.assertEqual(result["chunk_count"], 4)
            self.assertEqual(len(result["prediction"]), 100)
            self.assertEqual(result["validation"]["validation_level"], "shape-finite")
            self.assertTrue(result["compatible"])
            self.assertEqual(controller._policy.infer_calls, 4)
            self.assertEqual(result["prediction"][25][0], np.float32(0.026))
            self.assertNotIn("top-secret", json.dumps(controller.status()))
            self.assertNotIn("token", result)
            controller.close()

    def test_non_float32_policy_output_is_rejected(self):
        class BadPolicy(FakePolicy):
            def infer(self, value):
                return np.zeros((25, 65), dtype=np.float64), {}

        with tempfile.TemporaryDirectory() as directory:
            controller = LocalEvaluatorController(
                FakeRuntime(),
                TrajectoryValidator(directory),
                remote_factory=FakeRemote,
                policy_factory=BadPolicy,
                router_session_factory=lambda endpoint: FakeSession(),
                startup_timeout_s=0.1,
            )
            controller.connect_remote(
                endpoint="tcp/example.test:7448",
                session_id="team-a",
                token="secret",
            )
            controller.start_policy("team:test")
            with self.assertRaisesRegex(RuntimeError, r"float32\[T,65\]"):
                controller.shadow()

    def test_remote_errors_redact_token_from_state_and_exception(self):
        def failing_factory(*args, **kwargs):
            return FakeRemote(*args, fail=True, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            controller = make_controller(Path(directory), remote_factory=failing_factory)
            with self.assertRaises(RuntimeError) as raised:
                controller.connect_remote(
                    endpoint="tcp/example.test:7448",
                    session_id="team-a",
                    token="never-print-me",
                )
            self.assertNotIn("never-print-me", str(raised.exception))
            self.assertNotIn("never-print-me", json.dumps(controller.status()))
            self.assertIn("<redacted>", str(raised.exception))


class DockerRuntimeTests(unittest.TestCase):
    def test_ipc_path_is_randomized_and_owner_only(self):
        runtime = DockerRuntime(
            runner=lambda command, **kwargs: subprocess.CompletedProcess(
                command, 0, "", ""
            )
        )
        first_path = runtime.ipc_dir
        other = DockerRuntime(
            runner=lambda command, **kwargs: subprocess.CompletedProcess(
                command, 0, "", ""
            )
        )
        self.assertNotEqual(first_path, other.ipc_dir)
        runtime.prepare_gateway_ipc()
        self.assertEqual(first_path.stat().st_mode & 0o777, 0o700)
        socket_path = first_path / "gateway.sock"
        socket_path.touch()
        runtime.grant_gateway_ipc_access()
        self.assertEqual(socket_path.stat().st_mode & 0o777, 0o600)
        socket_path.unlink()
        first_path.rmdir()

    def test_commands_enforce_internal_read_only_nonroot_sandbox(self):
        commands = []
        running = {"value": False}

        def runner(command, **kwargs):
            commands.append(command)
            if command[:3] == ["docker", "run", "--detach"] and "origami-local-policy" in command:
                running["value"] = True
            if command[:3] == ["docker", "container", "inspect"]:
                if not running["value"]:
                    return subprocess.CompletedProcess(command, 1, "", "absent")
                payload = json.dumps([{
                    "State": {"Running": True, "StartedAt": "now"},
                    "Config": {"Image": "team:test"},
                    "Image": "sha256:fake",
                }])
                return subprocess.CompletedProcess(command, 0, payload, "")
            return subprocess.CompletedProcess(command, 0, "", "")

        with tempfile.TemporaryDirectory() as ipc_directory:
            runtime = DockerRuntime(runner=runner, ipc_dir=ipc_directory)
            runtime.start(
                "team:test",
                session_id="local-shadow-test",
                gateway_endpoint="unixsock-stream//origami-ipc/gateway.sock",
            )

        network = next(command for command in commands if command[:3] == ["docker", "network", "create"])
        self.assertIn("--internal", network)
        policy = next(
            command
            for command in commands
            if command[:3] == ["docker", "run", "--detach"]
            and "origami-local-policy" in command
        )
        for required in (
            "--read-only",
            "--cap-drop",
            "ALL",
            "no-new-privileges=true",
            "--user",
            "65532:65532",
            "--pids-limit",
            "--memory",
            "--cpus",
            "--tmpfs",
        ):
            self.assertIn(required, policy)
        environments = [
            policy[index + 1]
            for index, item in enumerate(policy[:-1])
            if item == "--env"
        ]
        self.assertEqual(len(environments), 2)
        self.assertTrue(any(item.startswith("ORIGAMI_ZENOH_ENDPOINT=") for item in environments))
        self.assertTrue(any(item.startswith("ORIGAMI_SESSION_ID=") for item in environments))
        self.assertFalse(any("REMOTE" in item or "TOKEN" in item for item in environments))
        self.assertNotIn("--privileged", policy)
        router = next(
            command
            for command in commands
            if command[:3] == ["docker", "run", "--detach"]
            and "origami-local-router" in command
        )
        self.assertNotIn("--publish", router)
        self.assertIn("--mount", router)
        self.assertIn("unixsock-stream//origami-ipc/gateway.sock", router)

    def test_image_cannot_be_parsed_as_a_docker_option(self):
        runtime = DockerRuntime(
            runner=lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "", "")
        )
        with self.assertRaisesRegex(ValueError, "option-safe"):
            runtime.start(
                "--privileged",
                session_id="local-shadow-test",
                gateway_endpoint="unixsock-stream//origami-ipc/gateway.sock",
            )


class HttpTests(unittest.TestCase):
    def test_cli_rejects_non_loopback_binding(self):
        with self.assertRaises(SystemExit):
            evaluator_main.main(["--host", "0.0.0.0"])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.controller = make_controller(Path(self.temp.name))
        self.server = create_server(
            self.controller,
            host="127.0.0.1",
            port=0,
            robot_assets_root=Path(self.temp.name),
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.controller.close()
        self.temp.cleanup()

    def post(self, path, value):
        return urlopen(
            Request(
                self.base + path,
                data=json.dumps(value).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            ),
            timeout=2,
        )

    def test_api_connect_start_reset_shadow_and_token_never_echoes(self):
        secret = "browser-one-shot-secret"
        with self.post("/api/remote/connect", {
            "endpoint": "tcp/example.test:7448",
            "session_id": "team-web",
            "token": secret,
        }) as response:
            self.assertNotIn(secret.encode(), response.read())
        archive_bytes = b"browser selected archive"
        with urlopen(
            Request(
                self.base + "/api/submission/upload",
                data=archive_bytes,
                headers={
                    "Content-Type": "application/octet-stream",
                    "X-Origami-Filename": "team-web.tar.zst",
                    "X-Origami-Sha256": hashlib.sha256(archive_bytes).hexdigest(),
                },
                method="POST",
            ),
            timeout=2,
        ) as response:
            upload = json.load(response)
        self.assertEqual(upload["status"], "running")
        deadline = time.monotonic() + 5.0
        while time.monotonic() <= deadline:
            with urlopen(
                self.base + f"/api/submission/load/status?job_id={upload['job_id']}",
                timeout=2,
            ) as response:
                job = json.load(response)
            if job["status"] == "completed":
                upload = job["result"]
                break
            if job["status"] == "failed":
                self.fail(job.get("error") or "archive load failed")
            time.sleep(0.05)
        else:
            self.fail("archive load did not complete")
        self.assertEqual(upload["images"], ["team:test"])
        with self.post("/api/submission/start", {"image": "team:test"}):
            pass
        with self.post("/api/policy/reset", {}):
            pass
        with self.post("/api/policy/shadow", {"preview_steps": 100}) as response:
            result = json.load(response)
        self.assertEqual(result["action_shape"], [100, 65])

        with urlopen(self.base + "/api/status", timeout=2) as response:
            status_body = response.read()
        self.assertNotIn(secret.encode(), status_body)
        self.assertNotIn(b'"token"', status_body)
        with urlopen(self.base + "/api/observation", timeout=2) as response:
            self.assertEqual(len(json.load(response)["state"]), 65)
        with urlopen(self.base + "/api/image/head-left", timeout=2) as response:
            self.assertTrue(response.read().startswith(b"\xff\xd8"))

    def test_only_explicit_shadow_post_routes_exist_and_assets_cannot_escape(self):
        self.assertEqual(POST_ROUTES, {
            "/api/remote/connect",
            "/api/submission/upload",
            "/api/submission/load",
            "/api/submission/start",
            "/api/submission/stop",
            "/api/policy/reset",
            "/api/policy/shadow",
        })
        for path in ("/api/action", "/api/publish", "/api/execute"):
            with self.subTest(path=path), self.assertRaises(HTTPError) as raised:
                self.post(path, {})
            self.assertEqual(raised.exception.code, 404)
        with self.assertRaises(HTTPError) as raised:
            urlopen(self.base + "/robot-assets/%2e%2e/secret", timeout=2)
        self.assertIn(raised.exception.code, (404, 503))

    def test_page_has_required_controls_and_no_execution_controls(self):
        with urlopen(self.base + "/", timeout=2) as response:
            html = response.read().decode()
        self.assertIn("Participant Local Shadow Evaluation", html)
        self.assertIn("Run Shadow Inference", html)
        self.assertIn("North URDF", html)
        self.assertIn("Complete URDF + STL Mesh", html)
        self.assertIn('type="module"', html)
        self.assertIn('type="file"', html)
        self.assertIn("Select and Load Submission Archive", html)
        self.assertIn("Four Real RGB Observations", html)
        self.assertNotIn("Start Real Robot", html)
        self.assertNotIn("/api/action", html)
        with urlopen(self.base + "/static/vendor/URDFLoader.js", timeout=2) as response:
            self.assertIn(b"class URDFLoader", response.read())


class UrdfValidationTests(unittest.TestCase):
    def test_parses_official_joint_limits_when_available(self):
        with tempfile.TemporaryDirectory() as directory:
            urdf_dir = Path(directory) / "urdf"
            urdf_dir.mkdir()
            joints = "\n".join(
                f'<joint name="{name}" type="revolute"><parent link="p{index}"/>'
                f'<child link="c{index}"/><limit lower="-1" upper="1" velocity="2"/></joint>'
                for index, name in enumerate(JOINT_NAMES)
            )
            (urdf_dir / "north_poc2_2_with_hand_description.urdf").write_text(
                f"<robot name=\"north\">{joints}</robot>",
                encoding="utf-8",
            )
            validator = TrajectoryValidator(directory)
            self.assertTrue(validator.has_urdf_limits)
            actions = np.zeros((1, 65), dtype=np.float32)
            actions[0, 0] = np.float32(1.5)
            report = validator.validate(
                np.zeros(65, dtype=np.float32),
                actions,
                control_hz=30,
            )
            self.assertEqual(report["validation_level"], "shape-finite-urdf-position-velocity")
            self.assertFalse(report["compatible"])
            self.assertIn("upper_limit", {item["type"] for item in report["violations"]})


if __name__ == "__main__":
    unittest.main()
