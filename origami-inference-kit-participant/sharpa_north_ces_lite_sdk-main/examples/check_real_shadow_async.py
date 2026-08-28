#!/usr/bin/env python3
"""Exercise async temporal aggregation with the local Shadow API.

The Shadow API snapshots the real robot observation and runs the currently
loaded policy image. This checker sends every resulting action chunk only to an
in-process sink. It never calls a live/start endpoint or a North action API.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request

import numpy as np

from openpi_origami_async import AsyncTimeAggregationInferencer


ACTION_KEY = "/action/all/joint_position"


def request_json(url: str, payload: dict | None = None, timeout: float = 240.0):
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"} if body is not None else {},
        method="POST" if body is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


class RealObservationClockMemorySink:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.published: list[dict[str, np.ndarray]] = []
        self._last_timestamp = 0
        self._lock = threading.Lock()

    def wait_for_observation(self, *, timeout, after_ts):
        deadline = time.monotonic() + (timeout or 1.0)
        while time.monotonic() < deadline:
            status = request_json(f"{self.base_url}/api/status", timeout=5.0)
            robot = status.get("robot") or {}
            evaluation = status.get("evaluation") or {}
            if evaluation.get("running"):
                raise RuntimeError("refusing dry-run while a live episode is running")
            if robot.get("connected") and float(robot.get("age_ms", 1e9)) < 1000.0:
                self._last_timestamp += 1
                return {
                    "into_buffer_ts": self._last_timestamp,
                    "robot_age_ms": robot.get("age_ms"),
                }
            time.sleep(0.02)
        return None

    def publish_single_action(self, action):
        with self._lock:
            self.published.append(
                {key: np.asarray(value).copy() for key, value in action.items()}
            )

    def clear_action_and_history(self):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:7860")
    parser.add_argument("--prompt", default="fold a paper airplane")
    parser.add_argument("--inferences", type=int, default=5)
    parser.add_argument("--inference-hz", type=float, default=6.0)
    parser.add_argument("--control-hz", type=float, default=30.0)
    args = parser.parse_args()
    if args.inferences < 1:
        parser.error("inferences must be positive")

    environment = RealObservationClockMemorySink(args.base_url)
    latencies_ms: list[float] = []

    def policy(_observation):
        started = time.monotonic()
        result = request_json(
            f"{args.base_url.rstrip('/')}/api/policy/shadow",
            {
                "prompt": args.prompt,
                "preview_steps": 50,
                "control_hz": args.control_hz,
            },
        )
        latency_ms = (time.monotonic() - started) * 1000.0
        latencies_ms.append(latency_ms)
        if result.get("compatible") is not True:
            raise RuntimeError(f"Shadow prediction is incompatible: {result.get('validation')}")
        actions = np.ascontiguousarray(result["prediction"], dtype=np.float32)
        if actions.shape != (50, 65) or not np.isfinite(actions).all():
            raise RuntimeError(f"invalid real policy output: {actions.dtype}{actions.shape}")
        return {ACTION_KEY: actions}

    runner = AsyncTimeAggregationInferencer(
        environment,
        policy,
        inference_hz=args.inference_hz,
        control_hz=args.control_hz,
        compensation_steps="auto",
        send_actions=True,
        require_new_obs=True,
        ta_agg_n=4,
        ta_exp_k=0.01,
        ta_max_chunks=16,
        ta_hold_last=True,
        log_interval_s=0.0,
    )
    result = runner.run(max_steps=args.inferences)
    status = request_json(f"{args.base_url.rstrip('/')}/api/status", timeout=5.0)
    evaluation = status.get("evaluation") or {}
    if evaluation.get("running") or evaluation.get("frames") != 0:
        raise RuntimeError("live action state changed during a Shadow-only dry-run")
    if not environment.published:
        raise RuntimeError("the in-process control sink received no fused steps")
    print(
        "PASS: real-observation real-policy async dry-run "
        f"inferences={result['steps']} memory_steps={len(environment.published)} "
        f"latency_ms={[round(value, 1) for value in latencies_ms]} "
        "robot_action_frames=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
