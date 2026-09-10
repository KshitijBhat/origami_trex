"""Populate the baked torch.compile cache for the origami-zenoh-v1 submission image.

Run inside a writable (non-read-only) throwaway container built from the same Dockerfile
with an empty `compile-cache/`, per `origami/docker/Dockerfile`'s comments and the submission
runbook: constructs `Policy(..., compile=True)` (which arms `torch.compile` on
`forward_flow_action_partial`/`tactile_flow_continue`, see `origami/policy.py`), runs a few
warm-up call patterns to trigger compilation once per shape, then times repeat calls to
confirm steady state before the populated cache is `docker cp`'d out and baked into the final
image. Never touches real dataset files -- shapes are fixed for a given checkpoint
(`image_size`/`action_chunk`/`vqvae_window` are all constant, batch=1 always), so synthetic
random content exercises the same graphs real observations would without needing the
(deliberately small, code-only) Docker build context to carry any season data.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import numpy as np

logger = logging.getLogger(__name__)


def _random_images(rng: np.random.Generator) -> dict:
    from PIL import Image

    def rand_img() -> "Image.Image":
        return Image.fromarray(rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8))

    return {"head": rand_img(), "wrist_left": rand_img(), "wrist_right": rand_img()}


def _zero_pattern(policy, kin) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray | None]:
    from origami.serve_zenoh import state62_from_state65, zero_observation

    obs = zero_observation()
    from PIL import Image

    images = {
        "head": Image.fromarray(obs["observation/image/head_left"]),
        "wrist_left": Image.fromarray(obs["observation/image/wrist_left"]),
        "wrist_right": Image.fromarray(obs["observation/image/wrist_right"]),
    }
    f6_window = np.zeros((policy.vqvae_window, 10, 6), dtype=np.float32)
    deform = np.zeros((10, 240, 240), dtype=np.float32)
    state65 = np.asarray(obs["observation/state"], dtype=np.float32)
    state62 = state62_from_state65(kin, state65) if policy.use_robot_state else None
    return images, f6_window, deform, state62


def _varied_pattern(
    policy, kin, rng: np.random.Generator
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray | None]:
    from origami.serve_zenoh import state62_from_state65

    images = _random_images(rng)
    f6_window = rng.uniform(-1.0, 1.0, size=(policy.vqvae_window, 10, 6)).astype(np.float32)
    deform = rng.uniform(0.0, 1.0, size=(10, 240, 240)).astype(np.float32)
    state62 = None
    if policy.use_robot_state:
        state65 = np.zeros(65, dtype=np.float32)
        state65[0:7] = rng.uniform(-0.2, 0.2, size=7)
        state65[29:36] = rng.uniform(-0.2, 0.2, size=7)
        state62 = state62_from_state65(kin, state65)
    return images, f6_window, deform, state62


def _time_call(policy, images, f6_window, deform, state62) -> tuple[float, float]:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    policy.slow_and_fast(images, f6_window, deform, state62=state62)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_slow = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    policy.fast(f6_window, deform)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_fast = (time.perf_counter() - t0) * 1000.0
    return t_slow, t_fast


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=os.environ.get("ORIGAMI_CHECKPOINT_PATH"))
    parser.add_argument(
        "--locked-config-source", default=os.environ.get("ORIGAMI_LOCKED_CONFIG_SOURCE"),
    )
    parser.add_argument("--urdf-path", default=os.environ.get("ORIGAMI_URDF_PATH"))
    parser.add_argument("--cuda", default=os.environ.get("ORIGAMI_CUDA", "0"))
    parser.add_argument("--n-steady-calls", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.checkpoint:
        parser.error("--checkpoint or ORIGAMI_CHECKPOINT_PATH is required")

    from origami.constants import URDF_PATH
    from origami.kinematics import OrigamiKinematics
    from origami.policy import Policy

    t_construct0 = time.time()
    policy = Policy(
        args.checkpoint, cuda=args.cuda, locked_config_source=args.locked_config_source,
        compile=True,
    )
    kin = OrigamiKinematics(
        args.urdf_path if args.urdf_path else URDF_PATH, policy.locked_config,
    )
    logger.info(
        "[compile_warmup] policy constructed in %.1fs (torch.compile armed)",
        time.time() - t_construct0,
    )

    rng = np.random.default_rng(args.seed)
    patterns = [("zero", _zero_pattern(policy, kin))]
    for i in range(2):
        patterns.append((f"varied_{i + 1}", _varied_pattern(policy, kin, rng)))

    for name, (images, f6_window, deform, state62) in patterns:
        t0 = time.time()
        t_slow, t_fast = _time_call(policy, images, f6_window, deform, state62)
        logger.info(
            "[compile_warmup] warm-up pattern %s: first call slow=%.1fms fast=%.1fms "
            "(wall %.1fs, includes any compilation)",
            name, t_slow, t_fast, time.time() - t0,
        )

    logger.info("[compile_warmup] steady-state check (%d repeats per pattern)", args.n_steady_calls)
    for name, (images, f6_window, deform, state62) in patterns:
        slow_times, fast_times = [], []
        for _ in range(args.n_steady_calls):
            t_slow, t_fast = _time_call(policy, images, f6_window, deform, state62)
            slow_times.append(t_slow)
            fast_times.append(t_fast)
        logger.info(
            "[compile_warmup]   %-10s slow mean=%.1fms max=%.1fms | fast mean=%.1fms max=%.1fms",
            name, float(np.mean(slow_times)), float(np.max(slow_times)),
            float(np.mean(fast_times)), float(np.max(fast_times)),
        )

    logger.info("[compile_warmup] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
