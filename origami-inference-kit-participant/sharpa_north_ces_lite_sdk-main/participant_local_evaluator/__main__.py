"""Run the participant-side local Shadow image evaluator."""

from __future__ import annotations

import argparse
import os
import pathlib

from .controller import LocalEvaluatorController
from .docker_runtime import DEFAULT_ROUTER_IMAGE, DockerRuntime
from .trajectory import DEFAULT_URDF_RELATIVE_PATH, TrajectoryValidator
from .web import create_server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument(
        "--robot-assets-dir",
        type=pathlib.Path,
        default=(
            pathlib.Path(os.environ["ORIGAMI_ROBOT_ASSETS_DIR"])
            if os.environ.get("ORIGAMI_ROBOT_ASSETS_DIR")
            else None
        ),
        help="official North asset directory containing urdf/ and meshes/",
    )
    parser.add_argument(
        "--urdf-relative-path",
        default=DEFAULT_URDF_RELATIVE_PATH,
    )
    parser.add_argument("--router-image", default=DEFAULT_ROUTER_IMAGE)
    parser.add_argument("--policy-timeout", type=float, default=180.0)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--memory", default="32g")
    parser.add_argument("--cpus", default="8")
    parser.add_argument("--shm-size", default="8g")
    parser.add_argument("--tmpfs-size", default="4g")
    parser.add_argument("--pids-limit", type=int, default=512)
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="do not pass --gpus all to the policy container",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("--host must be a loopback address; remote binding is not supported")
    if args.robot_assets_dir is None:
        parser.error("--robot-assets-dir or ORIGAMI_ROBOT_ASSETS_DIR is required")
    validator = TrajectoryValidator(
        args.robot_assets_dir,
        urdf_relative_path=args.urdf_relative_path,
    )
    runtime = DockerRuntime(
        router_image=args.router_image,
        memory=args.memory,
        cpus=args.cpus,
        shm_size=args.shm_size,
        tmpfs_size=args.tmpfs_size,
        pids_limit=args.pids_limit,
        gpus=None if args.no_gpu else "all",
    )
    controller = LocalEvaluatorController(
        runtime,
        validator,
        policy_timeout_s=args.policy_timeout,
        startup_timeout_s=args.startup_timeout,
    )
    server = create_server(
        controller,
        host=args.host,
        port=args.port,
        robot_assets_root=validator.assets_root,
    )
    print(
        f"participant local Shadow evaluator: http://{args.host}:{args.port}",
        flush=True,
    )
    if validator.load_error:
        print(
            f"compatibility notice: URDF checks unavailable: {validator.load_error}",
            flush=True,
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        controller.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
