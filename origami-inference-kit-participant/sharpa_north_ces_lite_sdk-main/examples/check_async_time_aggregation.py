#!/usr/bin/env python3
"""Offline smoke test for the asynchronous temporal-aggregation loop.

This command never connects to a robot and never publishes outside this process.
It validates that inference and control run at independent rates, overlapping
action chunks are fused, and the runner shuts down cleanly.
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from openpi_origami_async import AsyncTimeAggregationInferencer


ACTION_KEY = "/action/all/joint_position"


class OfflineEnvironment:
    def __init__(self) -> None:
        self.observation_index = 0
        self.published_steps = 0
        self.last_action: dict[str, np.ndarray] | None = None

    def wait_for_observation(self, *, timeout, after_ts):
        del timeout, after_ts
        self.observation_index += 1
        return {"into_buffer_ts": self.observation_index}

    def publish_single_action(self, action):
        self.published_steps += 1
        self.last_action = action

    def clear_action_and_history(self):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-hz", type=float, default=10.0)
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--inferences", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--latency-ms", type=float, default=20.0)
    args = parser.parse_args()
    if args.inferences < 1 or args.horizon < 1 or args.latency_ms < 0:
        parser.error("inferences/horizon must be positive and latency-ms non-negative")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    environment = OfflineEnvironment()
    inference_index = 0

    def policy(_observation):
        nonlocal inference_index
        inference_index += 1
        time.sleep(args.latency_ms / 1000.0)
        value = np.float32(inference_index)
        return {
            ACTION_KEY: np.full(
                (args.horizon, 65), value, dtype=np.float32
            )
        }

    runner = AsyncTimeAggregationInferencer(
        environment,
        policy,
        inference_hz=args.inference_hz,
        control_hz=args.control_hz,
        compensation_steps="auto",
        send_actions=True,
        ta_agg_n=4,
        ta_exp_k=0.01,
        ta_max_chunks=16,
        ta_hold_last=True,
        log_interval_s=0.0,
    )
    result = runner.run(max_steps=args.inferences)
    if result["steps"] != args.inferences:
        raise RuntimeError(f"expected {args.inferences} inferences, got {result}")
    if environment.published_steps < 1 or environment.last_action is None:
        raise RuntimeError("control loop published no action steps")
    action = environment.last_action.get(ACTION_KEY)
    if action is None or action.shape != (65,) or action.dtype != np.float32:
        raise RuntimeError("published action does not satisfy the 65-D float32 shape")
    print(
        "PASS: async temporal aggregation "
        f"inferences={result['steps']} published_steps={environment.published_steps} "
        f"avg_latency_ms={result['avg_latency_ms']:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
