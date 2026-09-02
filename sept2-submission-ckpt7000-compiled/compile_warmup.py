#!/usr/bin/env python3
"""Build-time only: populate the torch.compile (Inductor + Triton) cache.

Run inside a *writable* container built from this image with an empty
`compile-cache/` (runbook section 2).  Constructing `TRexOrigamiPolicy` with
`compile=True` already compiles the three flow entry points and runs the
warm-up patterns (random content, the public validator's all-zero-tactile
gradient pattern, two differently-valued random observations -- each of
which produced its own graph on the previous submission); this script does
that construction with the image's own TREX_* configuration, then hammers a
few more varied observations and re-times a repeat to prove every graph is a
cache hit.  Everything torch.compile writes lands in TORCHINDUCTOR_CACHE_DIR /
TRITON_CACHE_DIR (under /app/compile-cache per the Dockerfile), which is then
`docker cp`'d out and baked into the final image.

The cache is keyed to: checkpoint weights, `qwen_vla`/`tactile_vqvae`/
`trex_origami.policy` model code, torch/Triton versions, GPU architecture,
and the TREX_* inference settings (steps and K change the traced graphs).
Change any of those -> regenerate.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, "/app")

from trex_origami.policy import (TRexOrigamiPolicy, add_policy_arguments,  # noqa: E402
                                 config_from_args, synthetic_observation,
                                 validator_observation, varied_observation)


def _du(path: str) -> str:
    try:
        return subprocess.run(["du", "-sh", path], capture_output=True, text=True,
                              check=False).stdout.strip()
    except OSError:
        return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_policy_arguments(parser)
    parser.add_argument("--extra_varied", type=int, default=4)
    args = parser.parse_args()
    if not args.compile:
        print("[compile_warmup] TREX_COMPILE/--compile is off; nothing to bake", flush=True)
    print(f"[compile_warmup] TORCHINDUCTOR_CACHE_DIR={os.environ.get('TORCHINDUCTOR_CACHE_DIR')} "
          f"TRITON_CACHE_DIR={os.environ.get('TRITON_CACHE_DIR')}", flush=True)
    t0 = time.time()
    policy = TRexOrigamiPolicy(config_from_args(args))     # __init__ runs the warm-up patterns
    print(f"[compile_warmup] policy constructed + warmed in {time.time() - t0:.0f} s", flush=True)

    print(f"[compile_warmup] exercising {args.extra_varied} more varied observations "
          f"(each in a fresh episode)...", flush=True)
    for seed in range(10, 10 + args.extra_varied):
        policy.reset()
        t = time.time()
        policy.infer(varied_observation(seed))
        policy.infer(varied_observation(seed + 1000))
        print(f"[compile_warmup]   seed {seed}: two calls {1000 * (time.time() - t):.0f} ms "
              f"(second: {policy.last_timing.get('total_ms', float('nan')):.0f} ms)", flush=True)

    print("[compile_warmup] steady-state check (must be cache hits, no compile):", flush=True)
    for name, obs in (("validator", validator_observation()), ("random", synthetic_observation(3)),
                      ("varied", varied_observation(99))):
        policy.reset()
        policy.infer(obs)
        times = []
        for _ in range(3):
            t = time.time()
            policy.infer(obs)
            times.append(1000 * (time.time() - t))
        print(f"[compile_warmup]   {name:<10} {np.mean(times):7.1f} ms mean over 3 "
              f"(breakdown {' '.join(f'{k}={v:.0f}' for k, v in policy.last_timing.items())})",
              flush=True)

    for key in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        path = os.environ.get(key)
        if path:
            print(f"[compile_warmup] {key}: {_du(path)}", flush=True)
    print(f"[compile_warmup] done in {time.time() - t0:.0f} s -- compiled cache written to "
          f"TORCHINDUCTOR_CACHE_DIR / TRITON_CACHE_DIR", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
