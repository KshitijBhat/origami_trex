"""SDK Shadow / wire-contract check. REDESIGN_PLAN.md §8.2, §12 step 14. Gates G10, G11.

Drives ``serve_zenoh.py`` (started as a real subprocess, real GPU, real checkpoint) over a
real Zenoh session, exactly the wire protocol the competition uses -- no in-process shortcuts.
Two checks:

1. **Protocol conformance (G10).** Delegates entirely to the SDK's own
   ``examples/check_zenoh_policy.py::run_validation`` -- synthetic observations, metadata/
   reset/infer envelope + shape/dtype validation. We do not reimplement any of that.
2. **Shadow replay (G11).** Replays real recorded episodes through the real wire protocol
   using the SDK's ``examples/real_observation_source.py::RealObservationSource`` (built for
   exactly this -- see its sibling ``check_zenoh_policy_real_episodes.py``), scored for
   URDF limit/velocity/jump violations by the SDK's own
   ``participant_local_evaluator/trajectory.py::TrajectoryValidator`` -- the same checker the
   organizer's Shadow evaluator uses. G11 requires zero violations and zero NaN/IK-failure.

No Docker, no organizer router: a local Zenoh router isn't a participant-visible surface (the
wire protocol only cares that the policy is a Zenoh client reachable at some endpoint), so this
script opens a lightweight in-process Zenoh peer session as the router (``mode: peer``,
listening on loopback, multicast disabled) instead of running the SDK's hardened
Docker-in-Docker sandbox (``participant_local_evaluator/docker_runtime.py``) -- that sandbox
also requires the finished submission image (step 15) and is exercised separately by the SDK's
own local-evaluator web UI, not by this gate.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SDK_ROOT = _REPO_ROOT / "origami-inference-kit-participant" / "sharpa_north_ces_lite_sdk-main"
_EXAMPLES_DIR = _SDK_ROOT / "examples"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_sdk_modules():
    # real_observation_source imports `from contract import ACTION_DIM` (bare, sibling-relative)
    # and check_zenoh_policy_real_episodes imports `from check_zenoh_policy import ...` the same
    # way -- both assume `examples/` (and, transitively, `participant_local_evaluator/`) are on
    # sys.path, exactly as the SDK's own scripts do when run directly.
    if str(_EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(_EXAMPLES_DIR))
    if str(_SDK_ROOT) not in sys.path:
        sys.path.insert(0, str(_SDK_ROOT))
    check_zenoh_policy = _load_module("check_zenoh_policy", _EXAMPLES_DIR / "check_zenoh_policy.py")
    real_observation_source = _load_module(
        "real_observation_source", _EXAMPLES_DIR / "real_observation_source.py"
    )
    # `participant_local_evaluator` is a real package (its submodules use relative imports),
    # unlike the two standalone example scripts above -- import it normally now that its
    # parent is on sys.path, rather than loading trajectory.py in isolation (which would
    # re-trigger the package's own `__init__.py` and its relative imports out of context).
    import participant_local_evaluator.trajectory as trajectory

    return check_zenoh_policy, real_observation_source, trajectory


class LocalZenohRouter:
    """A lightweight in-process Zenoh peer session acting as the router for this check.

    A Zenoh ``client``-mode session (what ``policy_server_template.py::OrigamiZenohServer``
    and the SDK's own validators both open, per the wire protocol) only needs *some* reachable
    peer/router to connect to -- it doesn't distinguish a dedicated router process from a peer
    session with routing enabled. Avoids Docker and the organizer's hardened unix-socket
    sandbox, which step 14 doesn't need (that sandbox also requires the finished submission
    image, step 15).
    """

    def __init__(self, port: int = 0):
        import zenoh

        self.port = port or _free_tcp_port()
        config = zenoh.Config()
        config.insert_json5("mode", json.dumps("peer"))
        config.insert_json5("listen/endpoints", json.dumps([f"tcp/127.0.0.1:{self.port}"]))
        config.insert_json5("scouting/multicast/enabled", "false")
        self.session = zenoh.open(config)

    @property
    def endpoint(self) -> str:
        return f"tcp/127.0.0.1:{self.port}"

    def close(self) -> None:
        self.session.close()


def _free_tcp_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_policy_server(
    checkpoint_path: str,
    endpoint: str,
    session_id: str,
    action_horizon: int,
    locked_config_source: str | None,
    urdf_path: str | None,
    cuda: str,
    extra_args: list[str] | None,
    startup_timeout: float,
) -> subprocess.Popen:
    """Launches ``python -m origami.serve_zenoh`` as a real subprocess."""
    cmd = [
        sys.executable, "-m", "origami.serve_zenoh",
        "--endpoint", endpoint,
        "--session-id", session_id,
        "--action-horizon", str(action_horizon),
        "--checkpoint-path", checkpoint_path,
        "--cuda", cuda,
    ]
    if locked_config_source:
        cmd += ["--locked-config-source", locked_config_source]
    if urdf_path:
        cmd += ["--urdf-path", urdf_path]
    cmd += extra_args or []
    logger.info("launching: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd, cwd=str(_REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    lines: list[str] = []
    deadline = time.monotonic() + startup_timeout
    ready = False
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            lines.append(proc.stdout.read())
            raise RuntimeError(
                f"serve_zenoh exited during startup (code={proc.returncode}):\n" + "".join(lines)
            )
        line = proc.stdout.readline()
        if line:
            lines.append(line)
            sys.stdout.write(line)
            if "TeamPolicy warm-up OK" in line:
                ready = True
                break
        else:
            time.sleep(0.1)
    if not ready:
        proc.terminate()
        raise RuntimeError(f"serve_zenoh did not become ready within {startup_timeout}s:\n" + "".join(lines))
    return proc


def run_protocol_conformance(endpoint: str, session_id: str, timeout: float, requests: int) -> dict:
    """G10: SDK's own ``check_zenoh_policy.py::run_validation``, unmodified."""
    check_zenoh_policy, _, _ = _load_sdk_modules()
    try:
        check_zenoh_policy.run_validation(
            endpoint=endpoint, session_id=session_id, timeout=timeout,
            requests=requests, expected_horizon=None,
        )
        return {"pass": True}
    except check_zenoh_policy.ValidationError as exc:
        return {"pass": False, "error": str(exc)}


def _write_urdf_assets_dir(urdf_path: Path) -> Path:
    tmp_dir = Path(tempfile.mkdtemp(prefix="origami_shadow_urdf_"))
    urdf_dir = tmp_dir / "urdf"
    urdf_dir.mkdir(parents=True)
    dest = urdf_dir / "robot.urdf"
    dest.write_bytes(urdf_path.read_bytes())
    return tmp_dir


def run_shadow_replay(
    endpoint: str,
    session_id: str,
    season_root: str,
    urdf_path: str,
    n_episodes: int,
    frames_per_episode: int,
    frame_stride: int,
    timeout: float,
) -> dict:
    """G11: replay real episodes through the real wire protocol, score with the SDK's own
    URDF limit/jump/velocity checker. Returns zero violations / zero NaN / zero IK-failure iff
    the gate passes."""
    check_zenoh_policy, real_observation_source, trajectory = _load_sdk_modules()

    assets_dir = _write_urdf_assets_dir(Path(urdf_path))
    validator = trajectory.TrajectoryValidator(assets_dir, urdf_relative_path="urdf/robot.urdf")
    if not validator.has_urdf_limits:
        logger.warning("TrajectoryValidator could not load URDF limits: %s", validator.load_error)

    season_root = Path(season_root)
    lerobot_dir = season_root if (season_root / "meta").is_dir() else season_root / "lerobot3.0"

    session = check_zenoh_policy.open_zenoh_session(endpoint)
    n_requests = 0
    n_nan_or_inf = 0
    all_violations: list[dict] = []
    latencies_ms: list[float] = []
    try:
        metadata = check_zenoh_policy.validate_metadata(
            check_zenoh_policy.query_once(session, "metadata", session_id, timeout), None,
        )
        horizon = metadata["action_horizon"]

        for episode_index in range(n_episodes):
            source = real_observation_source.RealObservationSource(
                dataset_root=lerobot_dir, drop_tactile_raw_every_n=0, episode_index=episode_index,
            )
            try:
                source.assert_requests_validity(frames_per_episode, frame_stride)
            except ValueError as exc:
                logger.info("episode %d: SKIP (%s)", episode_index, exc)
                continue

            check_zenoh_policy.validate_reset(
                check_zenoh_policy.query_once(session, "reset", session_id, timeout)
            )

            n_frames = 0
            while source.has_next() and n_frames < frames_per_episode:
                observation = source.next_observation(frame_stride)
                current_state = np.asarray(observation["observation/state"], dtype=np.float32)
                started = time.monotonic()
                reply = check_zenoh_policy.query_once(
                    session, "infer", session_id, timeout, observation=observation,
                )
                latencies_ms.append((time.monotonic() - started) * 1000.0)
                actions = check_zenoh_policy.validate_infer(reply, horizon)
                n_requests += 1
                if not np.isfinite(actions).all():
                    n_nan_or_inf += 1

                report = validator.validate(current_state, actions, control_hz=30.0)
                for v in report.get("violations", []):
                    v = dict(v)
                    v["episode_index"] = episode_index
                    v["frame_index"] = n_frames
                    all_violations.append(v)
                n_frames += 1
    finally:
        session.close()

    return {
        "n_requests": n_requests,
        "n_nan_or_inf": n_nan_or_inf,
        "n_violations": len(all_violations),
        "violations": all_violations[:50],
        "latency_ms": {
            "p50": float(np.percentile(latencies_ms, 50)) if latencies_ms else None,
            "p99": float(np.percentile(latencies_ms, 99)) if latencies_ms else None,
        },
        "urdf_limits_loaded": validator.has_urdf_limits,
        "pass": n_requests > 0 and n_nan_or_inf == 0 and len(all_violations) == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--locked-config-source", default=None)
    parser.add_argument("--urdf-path", default=None)
    parser.add_argument("--season-root", required=True, help="a season_*/ or season_*/lerobot3.0 dir")
    parser.add_argument("--action-horizon", type=int, default=4)
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--frames-per-episode", type=int, default=20)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--conformance-requests", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--serve-arg", action="append", default=[],
        help="extra raw args passed through to `python -m origami.serve_zenoh` (repeatable)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from origami.constants import URDF_PATH

    urdf_path = args.urdf_path or str(URDF_PATH)
    session_id = uuid.uuid4().hex

    router = LocalZenohRouter()
    logger.info("local Zenoh router listening at %s", router.endpoint)
    server_proc = None
    try:
        server_proc = start_policy_server(
            args.checkpoint, router.endpoint, session_id, args.action_horizon,
            args.locked_config_source, urdf_path, args.cuda, args.serve_arg,
            args.startup_timeout,
        )

        conformance = run_protocol_conformance(
            router.endpoint, session_id, args.timeout, args.conformance_requests,
        )
        logger.info("G10 protocol conformance: %s", "PASS" if conformance["pass"] else "FAIL")

        shadow = run_shadow_replay(
            router.endpoint, session_id, args.season_root, urdf_path,
            args.n_episodes, args.frames_per_episode, args.frame_stride, args.timeout,
        )
        logger.info(
            "G11 shadow replay: %s (n_requests=%d, n_violations=%d, n_nan=%d)",
            "PASS" if shadow["pass"] else "FAIL",
            shadow["n_requests"], shadow["n_violations"], shadow["n_nan_or_inf"],
        )
    finally:
        if server_proc is not None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server_proc.kill()
        router.close()

    report = {"G10_conformance": conformance, "G11_shadow_replay": shadow}
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))

    ok = conformance["pass"] and shadow["pass"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
