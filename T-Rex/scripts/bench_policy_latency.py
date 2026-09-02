#!/usr/bin/env python3
"""Batch-1 request latency of `TRexOrigamiPolicy` per inference configuration.

Times the full `infer()` call -- the number the Gateway sees -- on a real
observation, for every (mode, steps, K) requested, with the per-phase breakdown
(prep / embed / slow flow / fast flow / post).  Run it on an otherwise idle GPU.

    python scripts/bench_policy_latency.py --checkpoint_path ... \
        --flat_root /workspace/data/origami_trex/origami_flat/full/val
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from trex_origami.policy import (TRexOrigamiPolicy, add_policy_arguments,  # noqa: E402
                                 config_from_args, synthetic_observation)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_policy_arguments(parser)
    parser.add_argument("--flat_root", default="",
                        help="origami-flat split to take a real observation from")
    parser.add_argument("--configs", nargs="+",
                        default=["cascaded:10:6:1", "cascaded:10:6:8", "cascaded:5:3:1",
                                 "cascaded:5:3:8", "cascaded:5:3:16", "blind:10:0:8",
                                 "blind:5:0:8"],
                        help="mode:total:split:K")
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    if not args.checkpoint_path:
        parser.error("--checkpoint_path is required")
    policy = TRexOrigamiPolicy(config_from_args(args))
    obs = synthetic_observation()
    if args.flat_root:
        from trex_origami.replay_sources import FlatSource
        src = FlatSource(args.flat_root)
        _, obs, _, _ = next(iter(src.episode(src.episode_indices()[0], every=1, max_requests=1)))
    results = {}
    for item in args.configs:
        mode, total, split, k = item.split(":")
        policy.cfg.mode, policy.cfg.total_steps = mode, int(total)
        policy.cfg.split_step, policy.cfg.n_draws = int(split), int(k)
        policy.reset()
        policy.infer(obs)                                   # warm this config
        totals, parts = [], {}
        for _ in range(args.samples):
            t0 = time.time()
            policy.infer(obs)
            totals.append(1000 * (time.time() - t0))
            for key, value in policy.last_timing.items():
                parts.setdefault(key, []).append(value)
        results[item] = {
            "mean_ms": float(np.mean(totals)), "p50_ms": float(np.percentile(totals, 50)),
            "p95_ms": float(np.percentile(totals, 95)),
            "breakdown_mean_ms": {key: float(np.mean(v)) for key, v in parts.items()},
        }
        print(f"{item:<20} mean {results[item]['mean_ms']:7.1f}  p50 {results[item]['p50_ms']:7.1f}"
              f"  p95 {results[item]['p95_ms']:7.1f}   "
              + " ".join(f"{key}={v:.0f}" for key, v in results[item]["breakdown_mean_ms"].items()))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(results, handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
