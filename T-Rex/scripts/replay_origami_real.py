#!/usr/bin/env python3
"""Replay real Robotic Origami Challenge episodes through the deployment adapter.

Two sources:
  lerobot   a season from the HF release (`<season>/lerobot3.0`), all six
            streams decoded with ffmpeg -- the same data the organizer's
            real_observation_source.py replays, closest to the live robot
            available offline.
  flat      an origami-flat split (`trex_origami.prepare` output).

Drives either a running `serve_origami_zenoh.py` (via --endpoint, no torch
needed in this process beyond satisfying trex_origami.policy's own imports)
or an in-process `TRexOrigamiPolicy` (no --endpoint, loads the checkpoint
directly). Scores each returned chunk's MAE/MSE against the teleoperator's
absolute-radian ground truth, per joint group and overall.

    python scripts/replay_origami_real.py \\
        --source lerobot --root /workspace/raw/<season>/lerobot3.0 \\
        --episodes 0 1 --every 5 --max_seconds 40 \\
        --endpoint tcp/127.0.0.1:17447 --session-id test-session \\
        --out_dir /workspace/eval/replay
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

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

RAD2DEG = 180.0 / np.pi


def group_mae(err: np.ndarray) -> Dict[str, float]:
    mae = np.abs(err).mean(axis=tuple(range(err.ndim - 1)))  # per-dim, over all leading axes
    return {name: float(mae[lo:hi].mean() * RAD2DEG) for name, lo, hi in JOINT_GROUPS}


class ZenohPolicyClient:
    """Pure client: talks to a running serve_origami_zenoh.py over Zenoh.

    No torch/model in this process -- only trex_origami.policy's *import*
    (for JOINT_NAMES/etc via the shared modules) requires torch to be
    importable, never executed on this path.
    """

    def __init__(self, endpoint: str, session_id: str, timeout: float = 60.0):
        import zenoh
        self.session_id = session_id
        self.timeout = timeout
        config = zenoh.Config()
        config.insert_json5("mode", json.dumps("client"))
        config.insert_json5("connect/endpoints", json.dumps([endpoint]))
        config.insert_json5("scouting/multicast/enabled", "false")
        config.insert_json5("transport/shared_memory/enabled", "false")
        self.session = zenoh.open(config)
        self.metadata_ = self._call("metadata", {})["metadata"]
        self.action_chunk = self.metadata_["action_horizon"]
        self.action_dim = self.metadata_["action_dim"]

    def _call(self, operation: str, extra: dict) -> dict:
        import uuid
        request = {"protocol_version": TRANSPORT_VERSION, "operation": operation,
                   "request_id": uuid.uuid4().hex, "session_id": self.session_id, **extra}
        replies = list(self.session.get(f"{TRANSPORT_VERSION}/{operation}",
                                        payload=pack_payload(request),
                                        timeout=self.timeout))
        if not replies:
            raise RuntimeError(f"{operation}: no reply within {self.timeout}s")
        reply = replies[0]
        if hasattr(reply, "err") and reply.err is not None:
            raise RuntimeError(f"{operation}: {unpack_payload(reply.err.payload)}")
        response = unpack_payload(reply.ok.payload)
        if response.get("error"):
            raise RuntimeError(f"{operation}: {response['error']}")
        return response

    def reset(self) -> None:
        self._call("reset", {})

    def infer(self, observation: dict) -> np.ndarray:
        response = self._call("infer", {"observation": observation})
        return np.asarray(response["actions"])


class LocalPolicyClient:
    """In-process TRexOrigamiPolicy -- needs the real checkpoint + torch/GPU."""

    def __init__(self, args):
        self.policy = TRexOrigamiPolicy(config_from_args(args))
        self.action_chunk = self.policy.action_chunk
        self.action_dim = self.policy.action_dim

    def reset(self) -> None:
        self.policy.reset()

    def infer(self, observation: dict) -> np.ndarray:
        return self.policy.infer(observation)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["lerobot", "flat"], required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--episodes", type=int, nargs="*", default=[],
                        help="episode indices (default: first)")
    parser.add_argument("--every", type=int, default=5,
                        help="request every N source frames")
    parser.add_argument("--start_seconds", type=float, default=0.0)
    parser.add_argument("--max_seconds", type=float, default=40.0,
                        help="replay this much of each episode")
    parser.add_argument("--include_raw", action="store_true")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--endpoint", default="",
                        help="Zenoh router of a *running* policy container (e.g. "
                             "tcp/127.0.0.1:17447); the local model is not loaded")
    parser.add_argument("--session-id", default=os.environ.get("ORIGAMI_SESSION_ID", ""))
    parser.add_argument("--timeout", type=float, default=60.0)
    add_policy_arguments(parser)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

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
        policy = LocalPolicyClient(args)

    fps = getattr(source, "fps", 30.0)
    max_requests = int(args.max_seconds * fps / max(1, args.every)) if args.max_seconds else 0
    start = int(args.start_seconds * fps)

    report = {"source": args.source, "root": args.root, "episodes": {}}
    all_chunk_err: List[np.ndarray] = []
    all_chunks: List[dict] = []
    for ep in episodes:
        length = source.episode_length(ep)
        print(f"\nepisode {ep} ({length} frames): replaying {args.max_seconds}s from "
              f"{args.start_seconds}s, every {args.every} frames")
        policy.reset()
        chunks = []
        n = 0
        for frame, obs, future, ep_fps in source.episode(ep, args.every, max_requests, start=start):
            validate_observation(obs)
            t0 = time.time()
            actions = policy.infer(obs)
            ms = 1000 * (time.time() - t0)
            k = min(actions.shape[0], future.shape[0])
            err = actions[:k].astype(np.float64) - future[:k]
            mae = float(np.abs(err).mean() * RAD2DEG)
            mse = float((err ** 2).mean())
            chunks.append({"frame": frame, "latency_ms": ms, "mae_deg": mae,
                           "mse": mse, "groups": group_mae(err)})
            all_chunk_err.append(err)
            n += 1
            if n % 50 == 1:
                print(f"  f={frame:>6} t={frame / ep_fps:6.1f}s {ms:7.0f} ms "
                      f"mae={mae:.3f} deg")
        if not chunks:
            print(f"  episode {ep}: no requests generated, skipping")
            continue
        lat = np.array([c["latency_ms"] for c in chunks])
        ep_mae_arr = np.array([c["mae_deg"] for c in chunks])
        ep_mse_arr = np.array([c["mse"] for c in chunks])
        print(f"  {len(chunks)} requests | latency mean {lat.mean():.0f} ms p95 "
              f"{np.percentile(lat, 95):.0f} ms | chunk MAE mean {ep_mae_arr.mean():.3f} "
              f"median {np.median(ep_mae_arr):.3f} deg | MSE mean {ep_mse_arr.mean():.6f} "
              f"median {np.median(ep_mse_arr):.6f}")
        report["episodes"][str(ep)] = {
            "n_requests": len(chunks), "latency_ms_mean": float(lat.mean()),
            "latency_ms_p95": float(np.percentile(lat, 95)),
            "chunk_mae_deg_mean": float(ep_mae_arr.mean()),
            "chunk_mae_deg_median": float(np.median(ep_mae_arr)),
            "chunk_mse_mean": float(ep_mse_arr.mean()),
            "chunk_mse_median": float(np.median(ep_mse_arr)),
            "per_group_mae_deg": {g: float(np.mean([c["groups"][g] for c in chunks]))
                                  for g in chunks[0]["groups"]},
        }
        all_chunks.extend(chunks)

    if all_chunk_err:
        cat = np.concatenate([e.reshape(-1, e.shape[-1]) for e in all_chunk_err], axis=0)
        all_mae = np.array([c["mae_deg"] for c in all_chunks])
        all_mse = np.array([c["mse"] for c in all_chunks])
        report["overall"] = {
            "n_requests": sum(v["n_requests"] for v in report["episodes"].values()),
            "chunk_mae_deg_mean": float(all_mae.mean()),
            "chunk_mae_deg_median": float(np.median(all_mae)),
            "chunk_mse_mean": float(all_mse.mean()),
            "chunk_mse_median": float(np.median(all_mse)),
            "per_group_mae_deg": group_mae(cat),
        }
        o = report["overall"]
        print(f"\noverall: {o['n_requests']} requests, chunk MAE mean {o['chunk_mae_deg_mean']:.3f} "
              f"median {o['chunk_mae_deg_median']:.3f} deg, MSE mean {o['chunk_mse_mean']:.6f} "
              f"median {o['chunk_mse_median']:.6f}")
        print(f"per-group MAE (deg): " +
              " ".join(f"{g}={v:.3f}" for g, v in o["per_group_mae_deg"].items()))

    out_path = os.path.join(args.out_dir, f"replay_{args.source}.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
