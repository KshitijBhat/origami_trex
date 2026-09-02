#!/usr/bin/env python3
"""Replay real Robotic Origami Challenge episodes through the deployment adapter.

Feeds `TRexOrigamiPolicy.infer` the *exact* `origami-zenoh-v1` observation the
Gateway would send (validated with the server's own schema check), built from
real recorded data, and scores what comes back three ways:

  chunk           MAE / MSE of the returned float32[25, 65] against the
                  teleoperator's next 25 commands -- the definition the kit's
                  `check_zenoh_policy_real_episodes.py` uses, so the number is
                  comparable to the organizer's pi0.5 baseline.
  latency         wall-clock per request with the adapter's phase breakdown.
  gateway stream  the kit's async `TemporalEnsembler` replayed with the
                  *measured* latency of every request (or `--latency_frames`
                  to ask "what if"): a chunk computed from frame f is pushed
                  f + latency frames later aligned to f, and the next request
                  is issued only once the previous one has returned, exactly
                  like `AsyncTimeAggregationInferencer`.  The 30 Hz executed
                  stream is scored against the teleoperator's command stream.

Sources (`--source`):
  lerobot   a season from the HF release (`<season>/lerobot3.0`), all six
            wire fields real, decoded with ffmpeg.
  flat      an origami-flat split (val is already on disk in this format);
            `head_right` is filled with `head_left`, torque zero-filled.

    python scripts/replay_origami_real.py --checkpoint_path /workspace/outputs/checkpoint-2-7000 \
        --source lerobot --root /workspace/data/origami_trex/raw/<season>/lerobot3.0 \
        --episodes 0 5 --every 5 --max_seconds 40 --out_dir /workspace/eval/.../replay_raw
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
for p in (_PROJECT_DIR, _SCRIPT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from trex_origami.policy import (TRexOrigamiPolicy, add_policy_arguments,  # noqa: E402
                                 config_from_args)
from trex_origami.replay_sources import make_source  # noqa: E402
from trex_origami.seasons import JOINT_GROUPS, JOINT_NAMES  # noqa: E402
from serve_origami_zenoh import (TRANSPORT_VERSION, pack_payload,  # noqa: E402
                                 unpack_payload, validate_observation)

RAD2DEG = 180.0 / math.pi
KIT_ASYNC = os.environ.get(
    "ORIGAMI_KIT_ASYNC",
    "/workspace/origami_trex/origami-inference-kit-participant/"
    "sharpa_north_ces_lite_sdk-main/examples/openpi_origami_async.py")


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def group_mae(err: np.ndarray) -> Dict[str, float]:
    mae = np.abs(err).reshape(-1, err.shape[-1]).mean(axis=0)
    return {name: float(mae[lo:hi].mean() * RAD2DEG) for name, lo, hi in JOINT_GROUPS}


def jerk_deg(stream: np.ndarray) -> float:
    if stream.shape[0] < 3:
        return 0.0
    d2 = stream[2:] - 2 * stream[1:-1] + stream[:-2]
    return float(np.abs(d2).mean() * RAD2DEG)


def gateway_stream(chunks: List[dict], gt: np.ndarray, agg_n: int, exp_k: float,
                   ensembler_cls, hold_pose: np.ndarray, start: int = 0):
    """Replay the kit's TemporalEnsembler; returns (executed, covered, gt_used).

    `gt` is the 30 Hz command stream from episode frame `start` on, and chunk
    frames are absolute episode frames, so everything is shifted by `start`.
    """
    n = min(gt.shape[0], max(c["frame"] - start + c["latency"] for c in chunks) + 25)
    ens = ensembler_cls(max_chunks=16, agg_n=agg_n, exp_k=exp_k, hold_last=True)
    by_arrival = defaultdict(list)
    for c in chunks:
        by_arrival[c["frame"] - start + c["latency"]].append(c)
    executed = np.empty((n, gt.shape[1]), dtype=np.float64)
    covered = np.zeros(n, dtype=bool)
    for f in range(n):
        ens.set_current_step(f)
        for c in by_arrival.get(f, ()):
            ens.push_chunk({"a": c["actions"]}, offset_steps=c["latency"])
        # Covered = some chunk really spans this frame; hold_last repeats are stale.
        covered[f] = any(c.start_step <= f < c.end_step_exclusive for c in ens._chunks)
        step = ens.pop_step(f)
        executed[f] = hold_pose if step is None else step["a"]
    return executed, covered, gt[:n]


class ZenohPolicyClient:
    """Drive a *running* policy container over `origami-zenoh-v1` instead of an
    in-process adapter -- the same reset/infer surface, so the replay scores
    the exact image that will be submitted.  Envelope/codec mirror the kit's
    `check_zenoh_policy.py`."""

    def __init__(self, endpoint: str, session_id: str, timeout: float = 60.0):
        import uuid
        import zenoh
        self._uuid, self._zenoh = uuid, zenoh
        config = zenoh.Config()
        config.insert_json5("mode", json.dumps("client"))
        config.insert_json5("connect/endpoints", json.dumps([endpoint]))
        config.insert_json5("scouting/multicast/enabled", "false")
        config.insert_json5("transport/shared_memory/enabled", "false")
        self.session = zenoh.open(config)
        self.session_id, self.timeout = session_id, timeout
        meta = self.query("metadata")["metadata"]
        self.action_chunk = int(meta["action_horizon"])
        self.metadata_ = meta
        self.last_timing: Dict[str, float] = {}
        self.n_infer = 0

    def query(self, operation: str, **body):
        request_id = self._uuid.uuid4().hex
        request = {"protocol_version": TRANSPORT_VERSION, "operation": operation,
                   "request_id": request_id, "session_id": self.session_id, **body}
        replies = self.session.get(f"{TRANSPORT_VERSION}/{operation}",
                                   payload=pack_payload(request), timeout=self.timeout,
                                   consolidation=self._zenoh.ConsolidationMode.NONE)
        out = []
        for reply in replies:
            if reply.ok is not None:
                out.append(unpack_payload(reply.ok.payload.to_bytes()))
            elif reply.err is not None:
                raise RuntimeError(f"{operation}: zenoh error {reply.err.payload.to_string()}")
        if len(out) != 1:
            raise RuntimeError(f"{operation}: expected one reply, got {len(out)}")
        reply = out[0]
        if reply.get("request_id") != request_id or "error" in reply:
            raise RuntimeError(f"{operation}: bad reply {reply.get('error') or reply}")
        return reply

    def metadata(self):
        return {"policy": {"remote": True, **{k: v for k, v in self.metadata_.items()
                                              if k != "joint_names"}}}

    def reset(self) -> None:
        self.query("reset")

    def infer(self, observation, now=None) -> np.ndarray:
        reply = self.query("infer", observation=observation)
        self.last_timing = {"server_infer_ms": float(
            (reply.get("server_timing") or {}).get("infer_ms", float("nan")))}
        self.n_infer += 1
        return np.asarray(reply["actions"], dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_policy_arguments(parser)
    parser.add_argument("--source", choices=["lerobot", "flat"], required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--episodes", type=int, nargs="*", default=[],
                        help="episode indices (default: first)")
    parser.add_argument("--every", type=int, default=5,
                        help="candidate request spacing in source frames; with "
                             "--cadence measured a candidate is skipped while the "
                             "previous request is still in flight")
    parser.add_argument("--cadence", choices=["measured", "fixed"], default="measured")
    parser.add_argument("--latency_frames", type=int, default=-1,
                        help=">= 0: pretend every request took this many frames "
                             "(gateway replay only); -1 = use the measured latency")
    parser.add_argument("--start_seconds", type=float, default=0.0)
    parser.add_argument("--max_seconds", type=float, default=40.0,
                        help="replay this much of each episode")
    parser.add_argument("--agg_n", type=int, default=4)
    parser.add_argument("--exp_k", type=float, default=0.01)
    parser.add_argument("--include_raw", action="store_true",
                        help="also send observation/image/tactile_raw (lerobot source)")
    parser.add_argument("--urdf", default="",
                        help="optional URDF for the Shadow-evaluator safety checks")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--endpoint", default="",
                        help="Zenoh router of a *running* policy container (e.g. "
                             "tcp/127.0.0.1:17447); the local model is not loaded")
    parser.add_argument("--session-id", default=os.environ.get("ORIGAMI_SESSION_ID", ""))
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    kit = _load("openpi_origami_async", KIT_ASYNC)
    checker = None
    if args.urdf and os.path.exists(args.urdf):
        ev = _load("trex_eval_origami", os.path.join(_SCRIPT_DIR, "eval_origami.py"))
        checker = ev.SafetyChecker(args.urdf)

    source = make_source(args.source, args.root, include_raw=args.include_raw)
    episodes = args.episodes or source.episode_indices()[:1]
    if args.endpoint:
        if not args.session_id:
            raise SystemExit("--session-id (or ORIGAMI_SESSION_ID) is required with --endpoint")
        policy = ZenohPolicyClient(args.endpoint, args.session_id, args.timeout)
        print(f"driving remote policy at {args.endpoint}: {policy.metadata_}")
    else:
        if not args.checkpoint_path:
            raise SystemExit("--checkpoint_path is required without --endpoint")
        policy = TRexOrigamiPolicy(config_from_args(args))
    horizon = policy.action_chunk
    fps = source.fps

    report = {"source": args.source, "root": args.root, "policy": policy.metadata()["policy"],
              "cadence": args.cadence, "latency_frames": args.latency_frames,
              "episodes": {}}
    all_lat, all_chunk_err, all_stream_err = [], [], []

    for ep in episodes:
        policy.reset()
        gt_all = source.gt_actions(ep)
        start = int(args.start_seconds * fps)
        max_requests = 0
        n_frames_budget = int(args.max_seconds * fps)
        chunks: List[dict] = []
        busy_until = -1
        timings: List[dict] = []
        print(f"\nepisode {ep} ({source.episode_length(ep)} frames): replaying "
              f"{args.max_seconds:.0f}s from {args.start_seconds:.0f}s, every {args.every} "
              f"frames, cadence={args.cadence}")
        t_ep = time.time()
        for frame, obs, future, _ in source.episode(ep, args.every, max_requests, start=start):
            if frame - start >= n_frames_budget:
                break
            if args.cadence == "measured" and frame < busy_until:
                continue
            validate_observation(obs)
            t0 = time.time()
            actions = policy.infer(obs, now=frame / fps)
            wall = time.time() - t0
            lat_frames = (args.latency_frames if args.latency_frames >= 0
                          else int(round(wall * fps)))
            busy_until = frame + max(1, lat_frames)
            gt_chunk = future[:horizon]
            if gt_chunk.shape[0] < horizon:           # hold the final command
                gt_chunk = np.concatenate([gt_chunk, np.repeat(gt_chunk[-1:],
                                                              horizon - gt_chunk.shape[0], 0)])
            err = actions.astype(np.float64) - gt_chunk
            chunks.append({"frame": frame, "latency": lat_frames, "wall_ms": 1000 * wall,
                           "actions": actions,
                           "state": np.asarray(obs["observation/state"], dtype=np.float64),
                           "mae_rad": float(np.abs(err).mean()),
                           "mse": float((err ** 2).mean()), "groups": group_mae(err),
                           "timing": dict(policy.last_timing)})
            all_chunk_err.append(err)
            all_lat.append(1000 * wall)
            timings.append(dict(policy.last_timing))
            if len(chunks) % 10 == 1:
                print(f"  f={frame:6d} t={frame / fps:6.1f}s  {1000 * wall:6.0f} ms  "
                      f"mae={np.abs(err).mean() * RAD2DEG:.3f} deg  "
                      f"({' '.join(f'{k}={v:.0f}' for k, v in policy.last_timing.items())})")
        if not chunks:
            print("  no requests issued")
            continue

        # Before the first chunk arrives the robot holds where it started.
        executed, covered, gt_used = gateway_stream(
            chunks, gt_all[start:], args.agg_n, args.exp_k, kit.TemporalEnsembler,
            hold_pose=chunks[0]["state"], start=start)
        np.savez_compressed(
            os.path.join(args.out_dir, f"replay_{args.source}_ep{ep:03d}_chunks.npz"),
            frame=np.array([c["frame"] for c in chunks]),
            latency=np.array([c["latency"] for c in chunks]),
            wall_ms=np.array([c["wall_ms"] for c in chunks]),
            actions=np.stack([c["actions"] for c in chunks]),
            state=np.stack([c["state"] for c in chunks]),
            gt_stream=gt_all, start=start)
        # Score only frames after the first chunk could have arrived; before
        # that the Gateway has nothing to execute whatever the policy does.
        first = chunks[0]["frame"] - start + chunks[0]["latency"]
        s_err = executed[first:] - gt_used[first:]
        all_stream_err.append(s_err)
        stream = {
            "frames_scored": int(s_err.shape[0]),
            "mae_deg": float(np.abs(s_err).mean() * RAD2DEG),
            "per_group_mae_deg": group_mae(s_err),
            "uncovered_fraction": float(1.0 - covered[first:].mean()) if s_err.shape[0] else 1.0,
            "jerk_deg": jerk_deg(executed[first:]),
            "teleop_jerk_deg": jerk_deg(gt_used[first:]),
        }
        if checker is not None and s_err.shape[0]:
            flags = checker.check(executed[first], executed[first + 1:])
            stream["safety_violation_rate_per_value"] = {
                k: float(v.mean()) for k, v in flags.items()}
        lat = np.array([c["wall_ms"] for c in chunks])
        ep_report = {
            "n_requests": len(chunks),
            "replay_wall_s": time.time() - t_ep,
            "latency_ms": {"mean": float(lat.mean()), "p50": float(np.percentile(lat, 50)),
                           "p95": float(np.percentile(lat, 95)),
                           "frames_mean": float(np.mean([c["latency"] for c in chunks]))},
            "timing_breakdown_ms": {k: float(np.mean([t[k] for t in timings if k in t]))
                                    for k in timings[0]},
            "chunk": {
                "mae_rad": float(np.mean([c["mae_rad"] for c in chunks])),
                "mae_deg": float(np.mean([c["mae_rad"] for c in chunks]) * RAD2DEG),
                "mse": float(np.mean([c["mse"] for c in chunks])),
                "per_group_mae_deg": {g: float(np.mean([c["groups"][g] for c in chunks]))
                                      for g in chunks[0]["groups"]},
            },
            "gateway_stream": stream,
            "requests": [{k: v for k, v in c.items() if k not in ("actions", "timing", "state")}
                         for c in chunks],
        }
        report["episodes"][str(ep)] = ep_report
        print(f"  {len(chunks)} requests | latency mean {lat.mean():.0f} ms p95 "
              f"{np.percentile(lat, 95):.0f} ms | chunk MAE {ep_report['chunk']['mae_deg']:.3f} deg "
              f"MSE {ep_report['chunk']['mse']:.6f} | executed-stream MAE {stream['mae_deg']:.3f} deg "
              f"(uncovered {stream['uncovered_fraction']:.2f}, jerk {stream['jerk_deg']:.3f} vs "
              f"teleop {stream['teleop_jerk_deg']:.3f})")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            picks = [3, 7, 14, 36, 43]
            fig, axes = plt.subplots(len(picks), 1, figsize=(13, 2.1 * len(picks)), sharex=True)
            t_axis = (np.arange(gt_used.shape[0]) + start) / fps
            for ax, d in zip(axes, picks):
                ax.plot(t_axis, gt_used[:, d], label="teleop command", lw=1.3)
                ax.plot(t_axis, executed[:, d], label="executed (gateway replay)", lw=1.0)
                for c in chunks[::max(1, len(chunks) // 12)]:
                    tt = (np.arange(horizon) + c["frame"]) / fps
                    ax.plot(tt, c["actions"][:, d], color="gray", lw=0.6, alpha=0.5)
                ax.set_ylabel(JOINT_NAMES[d], fontsize=7)
                ax.grid(alpha=0.3)
            axes[0].legend(fontsize=8, loc="upper right")
            axes[0].set_title(f"{source.name} episode {ep}: chunks (gray) and executed stream",
                              fontsize=9)
            axes[-1].set_xlabel("time (s)")
            fig.tight_layout()
            fig.savefig(os.path.join(args.out_dir, f"replay_{args.source}_ep{ep:03d}.png"), dpi=130)
            plt.close(fig)
        except Exception as exc:  # noqa: BLE001 - plotting is best-effort
            print(f"  (plot skipped: {exc})")

    if all_lat:
        chunk_err = np.concatenate(all_chunk_err)
        stream_err = np.concatenate(all_stream_err) if all_stream_err else np.zeros((0, 65))
        report["overall"] = {
            "n_requests": len(all_lat),
            "latency_ms": {"mean": float(np.mean(all_lat)), "p50": float(np.percentile(all_lat, 50)),
                           "p95": float(np.percentile(all_lat, 95))},
            "chunk_mae_deg": float(np.abs(chunk_err).mean() * RAD2DEG),
            "chunk_mse": float((chunk_err ** 2).mean()),
            "chunk_per_group_mae_deg": group_mae(chunk_err),
            "stream_mae_deg": float(np.abs(stream_err).mean() * RAD2DEG) if stream_err.size else None,
            "stream_per_group_mae_deg": group_mae(stream_err) if stream_err.size else None,
        }
        print(f"\noverall: {len(all_lat)} requests, latency mean {np.mean(all_lat):.0f} ms, "
              f"chunk MAE {report['overall']['chunk_mae_deg']:.3f} deg, MSE "
              f"{report['overall']['chunk_mse']:.6f}, executed-stream MAE "
              f"{report['overall']['stream_mae_deg']}")
    with open(os.path.join(args.out_dir, f"replay_{args.source}.json"), "w") as handle:
        json.dump(report, handle, indent=2, default=float)
    print(f"wrote {args.out_dir}/replay_{args.source}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
