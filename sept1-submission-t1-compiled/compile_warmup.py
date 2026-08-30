"""Build-time only: trigger torch.compile for the two flow functions and
persist the resulting Triton/Inductor cache into the image (TORCHINDUCTOR_CACHE_DIR
/ TRITON_CACHE_DIR, both under /app/compile-cache per the Dockerfile). Not part
of the runtime image entrypoint -- run once during image build, while the
filesystem is still writable and executable, so the read-only + noexec /tmp
runtime environment only ever needs to *load* pre-existing compiled kernels,
never write or execute new ones.

As of TeamPolicy.__init__ doing its own startup warmup (see
trex_policy_server.py's _warmup(), added per participant_zenoh_submission.md's
"load and warm the model during startup" guidance), simply constructing a
TeamPolicy already triggers both compiles as a side effect -- this script's
job is just to do that construction at build time and verify a second call
reuses the compiled graphs, using the exact same synthetic-observation shapes
the runtime warmup uses (imported from trex_policy_server, not duplicated, so
the two can never drift out of shape-sync).
"""
import logging

import numpy as np

from trex_policy_server import ACTION_DIM, FIXED_PROMPT, TeamPolicy, make_warmup_observation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

ACTION_HORIZON = 25


def make_varied_observation(seed: int) -> dict:
    """Same shapes/dtypes as make_warmup_observation(), but different concrete
    values -- real traffic never repeats the exact same synthetic tensor
    make_warmup_observation() produces, and a real (differently-valued) call
    was observed to trigger a second, distinct compiled graph beyond the one
    the single-observation warmup produces (confirmed: the live server, after
    a clean warmup, still hit a fresh ~80s InductorError compile attempt on
    its first real query, which crashes under the runtime's --read-only
    profile). Exercising a second, differently-valued observation here, at
    build time while the filesystem is still writable, bakes that second
    graph into the cache too instead of leaving it to be discovered live."""
    rng = np.random.default_rng(seed)
    return {
        "observation/image/head_left": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/image/head_right": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/image/wrist_left": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/image/wrist_right": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/state": rng.uniform(-0.25, 0.25, ACTION_DIM).astype(np.float32),
        "observation/state/joint_torque": np.zeros(ACTION_DIM, dtype=np.float32),
        "observation/tactile": rng.uniform(0, 1, 60).astype(np.float32),
        "observation/image/tactile_deform": rng.integers(0, 255, (480, 1200, 3), dtype=np.uint8),
        "prompt": FIXED_PROMPT,  # value is irrelevant -- overridden at inference
    }


def make_zero_tactile_observation() -> dict:
    """Matches the public validator's (check_zenoh_policy.py)
    make_synthetic_observation() tactile/image content: gradient-pattern
    images, but all-zero observation/tactile + observation/image/tactile_deform
    + joint_torque. Confirmed directly (via an isolated in-process test, not
    guesswork) that this specific all-zero tactile pattern -- distinct from
    both the fixed warmup observation and randomly-varied ones -- triggers yet
    a THIRD distinct compiled graph (31 fresh compile_worker subprocesses
    observed). Baking this exact case in here is what lets the public
    validator's first query hit a cache hit instead of a fresh ~80-100s
    compile that crashes under the runtime's --read-only profile. (Its prompt
    value doesn't need to match anything anymore -- see FIXED_PROMPT.)"""
    rows = np.arange(224, dtype=np.uint8)[:, None]
    cols = np.arange(224, dtype=np.uint8)[None, :]
    base = np.empty((224, 224, 3), dtype=np.uint8)
    base[..., 0] = rows
    base[..., 1] = cols
    base[..., 2] = rows ^ cols
    return {
        "observation/image/head_left": np.ascontiguousarray(base),
        "observation/image/head_right": np.ascontiguousarray(np.roll(base, 11, axis=0)),
        "observation/image/wrist_left": np.ascontiguousarray(np.roll(base, 17, axis=1)),
        "observation/image/wrist_right": np.ascontiguousarray(np.flip(base, axis=1)),
        "observation/state": np.linspace(-0.25, 0.25, ACTION_DIM, dtype=np.float32),
        "observation/state/joint_torque": np.zeros(ACTION_DIM, dtype=np.float32),
        "observation/tactile": np.zeros(60, dtype=np.float32),
        "observation/image/tactile_deform": np.zeros((480, 1200, 3), dtype=np.uint8),
        "prompt": FIXED_PROMPT,  # value is irrelevant -- overridden at inference
    }


def main():
    print("[compile_warmup] constructing TeamPolicy (triggers its own startup "
          "warmup -- compiles both flow functions as a side effect of __init__)...",
          flush=True)
    policy = TeamPolicy(action_horizon=ACTION_HORIZON)

    print("[compile_warmup] verifying a fast-tick call reuses the already-"
          "compiled tactile_flow_continue graph...", flush=True)
    obs = make_warmup_observation()
    policy.infer(obs)   # due_for_refresh (post-construction reset left tick=0)
    policy.infer(obs)   # fast-only tick: should be fast, confirms reuse

    print("[compile_warmup] exercising varied-content observations (2 full "
          "episodes) to bake in any graph a differently-valued real call "
          "needs, not just the single fixed synthetic one above...", flush=True)
    for seed in (1, 2):
        policy.reset()
        policy.infer(make_varied_observation(seed))
        policy.infer(make_varied_observation(seed + 100))

    print("[compile_warmup] exercising the exact all-zero-tactile pattern the "
          "public validator sends, to bake in its distinct graph too...", flush=True)
    policy.reset()
    policy.infer(make_zero_tactile_observation())
    policy.infer(make_zero_tactile_observation())

    print("[compile_warmup] done -- compiled cache written to "
          "TORCHINDUCTOR_CACHE_DIR / TRITON_CACHE_DIR", flush=True)


if __name__ == "__main__":
    main()
