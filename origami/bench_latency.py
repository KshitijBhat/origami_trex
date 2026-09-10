"""Slow/fast/dataloader latency -> feasible ``action_horizon``. REDESIGN_PLAN.md §8.3, §12 step 13.

Must run before choosing ``--action-horizon``/``--slow-every`` for ``serve_zenoh.py`` (G13):
in sync execution mode the gateway blocks for the whole ``infer()`` call, so
``slow_p99 < action_horizon / 30 s`` must hold, or the horizon must grow (§9.1's ladder,
``T in {4, 8, 16}``).
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _percentiles(samples_s: list) -> dict:
    arr = np.asarray(samples_s, dtype=np.float64) * 1000.0  # -> ms
    return {
        "n": len(arr),
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "max_ms": float(np.max(arr)),
    }


def feasible_action_horizon(slow_p99_s: float, command_hz: int = 30) -> int:
    """§8.3: ``T_min = ceil(slow_p99_s * command_hz)`` in sync mode, rounded up to the next
    value in the §9.1 ladder ``{4, 8, 16}`` (``slow_every * T == 16`` must hold)."""
    t_min = int(np.ceil(slow_p99_s * command_hz))
    for t in (4, 8, 16):
        if t >= t_min:
            return t
    return 16


def bench_slow_and_fast(policy, n_calls: int, warmup: int = 3) -> dict:
    from origami.serve_zenoh import state62_from_state65, zero_observation
    from origami.kinematics import LockedConfig, OrigamiKinematics
    from origami.constants import URDF_PATH

    kin = OrigamiKinematics(URDF_PATH, policy.locked_config)
    obs = zero_observation()
    state65 = np.asarray(obs["observation/state"], dtype=np.float32)
    from PIL import Image
    images = {
        "head": Image.fromarray(obs["observation/image/head_left"]),
        "wrist_left": Image.fromarray(obs["observation/image/wrist_left"]),
        "wrist_right": Image.fromarray(obs["observation/image/wrist_right"]),
    }
    f6_window = np.zeros((policy.vqvae_window, 10, 6), dtype=np.float32)
    deform = np.zeros((10, 240, 240), dtype=np.float32)
    state62 = state62_from_state65(kin, state65) if policy.use_robot_state else None

    slow_times, fast_times = [], []
    for i in range(warmup + n_calls):
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        policy.slow_and_fast(images, f6_window, deform, state62=state62)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt_slow = time.perf_counter() - t0

        t0 = time.perf_counter()
        policy.fast(f6_window, deform)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt_fast = time.perf_counter() - t0

        if i >= warmup:
            slow_times.append(dt_slow)
            fast_times.append(dt_fast)

    return {"slow": _percentiles(slow_times), "fast": _percentiles(fast_times)}


def bench_dataloader(root: str, vqvae_window: int, n_batches: int, num_workers: int, batch_size: int) -> dict:
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    delta_timestamps = {
        "observation.tactile_f6": [(i - (vqvae_window - 1)) / 30.0 for i in range(vqvae_window)]
    }
    repo_id = Path(root.rstrip("/")).name
    ds = LeRobotDataset(repo_id, root=root, delta_timestamps=delta_timestamps)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers, shuffle=True,
    )
    times = []
    it = iter(loader)
    for i in range(n_batches + 2):
        t0 = time.perf_counter()
        next(it)
        dt = time.perf_counter() - t0
        if i >= 2:  # drop 2 warmup batches (worker spin-up)
            times.append(dt)
    samples_per_s_per_worker = [
        batch_size / t / max(num_workers, 1) for t in times
    ]
    return {
        "batch_latency": _percentiles(times),
        "samples_per_s_per_worker_mean": float(np.mean(samples_per_s_per_worker)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("slow", "fast", "slow_and_fast", "dataloader"), required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--locked-config-source", default=None)
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--n-calls", type=int, default=30)
    parser.add_argument("--command-hz", type=int, default=30)
    parser.add_argument("--root", default=None, help="dataloader mode: LeRobot dataset root")
    parser.add_argument("--n-batches", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--out", default=None, help="write JSON report here")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.mode == "dataloader":
        if not args.root:
            parser.error("--root is required for --mode dataloader")
        report = {
            "mode": "dataloader",
            **bench_dataloader(args.root, 16, args.n_batches, args.num_workers, args.batch_size),
        }
    else:
        if not args.checkpoint:
            parser.error("--checkpoint is required for slow/fast/slow_and_fast modes")
        from origami.policy import Policy

        policy = Policy(
            args.checkpoint, cuda=args.cuda, locked_config_source=args.locked_config_source,
        )
        result = bench_slow_and_fast(policy, args.n_calls)
        slow_p99_s = result["slow"]["p99_ms"] / 1000.0
        t_feasible = feasible_action_horizon(slow_p99_s, command_hz=args.command_hz)
        report = {
            "mode": args.mode,
            "checkpoint": args.checkpoint,
            **result,
            "feasible_action_horizon_sync": t_feasible,
            "slow_p99_budget_check": {
                "T=4": result["slow"]["p99_ms"] < 4 / args.command_hz * 1000.0,
                "T=8": result["slow"]["p99_ms"] < 8 / args.command_hz * 1000.0,
                "T=16": result["slow"]["p99_ms"] < 16 / args.command_hz * 1000.0,
            },
        }

    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
