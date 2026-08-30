#!/usr/bin/env python3
"""Real-episode validator for the origami-zenoh-v1 policy protocol.

Companion to check_zenoh_policy.py (protocol conformance, synthetic data
only) and eval_origami.py (real data, but calls the model directly and
never touches the Zenoh server). This is the missing third leg: real
recorded episodes, replayed through the actual running container over the
real wire protocol, scored against the real teleop ground-truth actions.

Reuses check_zenoh_policy.py's low-level protocol helpers (session open,
msgpack pack/unpack, reply-envelope/metadata/reset/infer validation) rather
than duplicating them, so both scripts can only ever agree on what "a valid
reply" means.
"""

from __future__ import annotations

import argparse
import json
import os
import datetime as _dt
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from check_zenoh_policy import (  # noqa: E402
    ACTION_DIM,
    SEMANTIC_PROTOCOL_VERSION,
    ZENOH_PROTOCOL_VERSION,
    ValidationError,
    open_zenoh_session,
    query_once,
    validate_endpoint,
    validate_infer,
    validate_metadata,
    validate_reset,
    validate_session_id,
)
from real_observation_source import RealObservationSource  # noqa: E402


def _joint_error_rad(actions: np.ndarray, ground_truth: np.ndarray) -> tuple[float, float]:
    """MAE/MSE in radians, flattened over horizon x dims -- same definition
    eval_origami.py uses, so numbers are directly comparable to it."""
    diff = actions.astype(np.float64) - ground_truth.astype(np.float64)
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    return mae, mse


def run_slot(
    session: Any,
    session_id: str,
    timeout: float,
    data_root: Path,
    slot_name: str,
    slot_config: dict[str, Any],
    n_episodes: int,
    frames_per_episode: int,
    frame_stride: int,
    horizon: int,
    records: list[dict[str, Any]],
) -> None:
    episode_indices = slot_config["episode_indices"][:n_episodes]
    season_root = data_root / slot_config["season"]
    if not season_root.exists():
        raise ValidationError(
            f"{slot_name}: season root does not exist: {season_root} "
            f"-- check --data-root and manifest.json's season field"
        )

    for episode_index in episode_indices:
        source = RealObservationSource(
            dataset_root=season_root,
            drop_tactile_raw_every_n=0,
            episode_index=episode_index,
        )
        try:
            source.assert_requests_validity(frames_per_episode, frame_stride)
        except ValueError as exc:
            print(f"{slot_name} ep={episode_index}: SKIP ({exc})")
            continue

        validate_reset(query_once(session, "reset", session_id, timeout))

        n_frames = 0
        while source.has_next() and n_frames < frames_per_episode:
            observation = source.next_observation(frame_stride)
            started = time.monotonic()
            reply = query_once(session, "infer", session_id, timeout, observation=observation)
            latency_ms = (time.monotonic() - started) * 1000.0
            actions = validate_infer(reply, horizon)
            ground_truth = source.get_ground_truth_actions(horizon)
            mae_rad, mse = _joint_error_rad(actions, ground_truth)
            frame_info = source.get_last_frame_info()

            record = {
                "slot": slot_name,
                "episode_index": episode_index,
                **frame_info,
                "latency_ms": latency_ms,
                "mae_rad": mae_rad,
                "mse": mse,
            }
            records.append(record)
            print(
                f"{slot_name} ep={episode_index} frame={frame_info['frame_index']}: PASS "
                f"({latency_ms:.1f} ms, mae={mae_rad:.4f} rad, mse={mse:.6f})"
            )
            n_frames += 1


class _Tee:
    """Writes to multiple streams at once -- used to mirror this script's own
    stdout/stderr into a per-run log file without touching any print() call
    site."""

    def __init__(self, *streams: Any) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def capture_container_logs(container: str, out_path: str) -> None:
    """Best-effort: dump the container's ENTIRE docker logs (since process
    start, not just this run's time window) to out_path. Deliberately not
    windowed by --since/--until: the policy container is long-lived across
    many validation runs, and the parts outside any one run's window --
    startup, model load, torch.compile warmup, READY -- are exactly the
    context you need when a run's numbers look off. Never raises -- a
    validation run's PASS/FAIL must not depend on whether this side channel
    worked."""
    try:
        with open(out_path, "w") as handle:
            subprocess.run(
                ["docker", "logs", container],
                stdout=handle, stderr=subprocess.STDOUT, check=False, timeout=30,
            )
        print(f"wrote {out_path}")
    except Exception as exc:  # noqa: BLE001 -- deliberately broad, see docstring
        print(f"warning: could not capture container logs for {container!r}: {exc}")


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for slot_name in ("train_slot", "val_slot"):
        slot_records = [r for r in records if r["slot"] == slot_name]
        if not slot_records:
            continue
        latencies = [r["latency_ms"] for r in slot_records]
        maes = [r["mae_rad"] for r in slot_records]
        mses = [r["mse"] for r in slot_records]
        summary[slot_name] = {
            "n_requests": len(slot_records),
            "n_episodes": len({r["episode_index"] for r in slot_records}),
            "latency_ms": {
                "mean": statistics.mean(latencies),
                "p50": statistics.median(latencies),
                "p95": float(np.percentile(latencies, 95)),
                "max": max(latencies),
            },
            "mae_rad_mean": statistics.mean(maes),
            "mse_mean": statistics.mean(mses),
        }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=os.environ.get("ORIGAMI_ZENOH_ENDPOINT"))
    parser.add_argument("--session-id", default=os.environ.get("ORIGAMI_SESSION_ID"))
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--manifest", required=True,
                        help="path to container_check_dataset/manifest.json")
    parser.add_argument("--data-root", required=True,
                        help="directory the manifest's season paths are relative to")
    parser.add_argument("--train-episodes", type=int, default=None,
                        help="how many of train_slot's episode_indices to replay "
                             "(default: all listed in the manifest)")
    parser.add_argument("--val-episodes", type=int, default=None,
                        help="how many of val_slot's episode_indices to replay "
                             "(default: all listed in the manifest)")
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="dataset frames to advance per infer() call")
    parser.add_argument("--frames-per-episode", type=int, default=10,
                        help="how many frames to replay per episode before "
                             "moving to the next one")
    parser.add_argument("--out-dir", default="container_validations",
                        help="parent directory for all validation runs -- "
                             "each run gets its own timestamped subfolder "
                             "under here, not written directly into this dir")
    parser.add_argument("--run-name", default=None,
                        help="subfolder name for this run's outputs, under "
                             "--out-dir (default: UTC timestamp)")
    parser.add_argument("--policy-container", default="origami-contract-policy",
                        help="name of the running policy container, used to "
                             "capture its docker logs for this run's duration "
                             "(default: origami-contract-policy)")
    parser.add_argument("--no-capture-container-logs", action="store_true",
                        help="skip capturing the policy container's docker "
                             "logs for this run (on by default)")
    args = parser.parse_args(argv)

    run_name = args.run_name or _dt.datetime.now(_dt.timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"writing this run's outputs to {run_dir}/")

    validator_log_path = os.path.join(run_dir, "validator.log")
    log_file = open(validator_log_path, "w")
    real_stdout, real_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(real_stdout, log_file)
    sys.stderr = _Tee(real_stderr, log_file)

    try:
        with open(args.manifest) as handle:
            manifest = json.load(handle)
        data_root = Path(args.data_root)

        endpoint = validate_endpoint(args.endpoint)
        session_id = validate_session_id(args.session_id)
        session = open_zenoh_session(endpoint)
        records: list[dict[str, Any]] = []
        try:
            metadata = validate_metadata(
                query_once(session, "metadata", session_id, args.timeout), None)
            horizon = metadata["action_horizon"]
            print(
                "metadata: PASS "
                f"(transport={ZENOH_PROTOCOL_VERSION}, semantic={SEMANTIC_PROTOCOL_VERSION}, "
                f"horizon={horizon}, dim={ACTION_DIM})"
            )

            for slot_name, requested_n in (
                ("train_slot", args.train_episodes),
                ("val_slot", args.val_episodes),
            ):
                slot_config = manifest[slot_name]
                available = len(slot_config["episode_indices"])
                n_episodes = available if requested_n is None else max(1, min(requested_n, available))
                print(f"--- {slot_name}: {n_episodes}/{available} episode(s), "
                      f"season={slot_config['season']} "
                      f"({slot_config.get('split_membership', 'unknown')}) ---")
                run_slot(session, session_id, args.timeout, data_root, slot_name,
                        slot_config, n_episodes, args.frames_per_episode,
                        args.frame_stride, horizon, records)
        finally:
            session.close()

        if not records:
            print("FAIL: no requests were sent")
            return 1

        summary = summarize(records)
        print("\nPASS: real-episode replay completed")
        for slot_name, stats in summary.items():
            lat = stats["latency_ms"]
            print(
                f"{slot_name}: requests={stats['n_requests']} episodes={stats['n_episodes']} "
                f"latency mean={lat['mean']:.1f}ms p50={lat['p50']:.1f}ms "
                f"p95={lat['p95']:.1f}ms max={lat['max']:.1f}ms "
                f"mae={stats['mae_rad_mean']:.4f}rad mse={stats['mse_mean']:.6f}"
            )

        out_path = os.path.join(run_dir, "real_episode_metrics.json")
        with open(out_path, "w") as handle:
            json.dump({
                "run_name": run_name,
                "manifest": os.path.abspath(args.manifest),
                "data_root": os.path.abspath(args.data_root),
                "train_episodes_requested": args.train_episodes,
                "val_episodes_requested": args.val_episodes,
                "frame_stride": args.frame_stride,
                "frames_per_episode": args.frames_per_episode,
                "summary": summary,
                "records": records,
            }, handle, indent=2, default=str)
        print(f"\nwrote {out_path}")
        return 0
    finally:
        if not args.no_capture_container_logs:
            capture_container_logs(args.policy_container, os.path.join(run_dir, "container.log"))
        sys.stdout, sys.stderr = real_stdout, real_stderr
        log_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
