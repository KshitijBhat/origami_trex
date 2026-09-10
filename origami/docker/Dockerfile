# Submission image for the origami-zenoh-v1 policy service. REDESIGN_PLAN.md §9.5, §12 step 15.
#
# Build context MUST be the repo root (`origami_trex/`), so this Dockerfile can COPY the
# checkpoint, URDF, T-Rex submodule subset, and origami/ package. Example:
#   docker build -f origami/docker/Dockerfile -t origami-submission:latest .
#
# Contains: torch/transformers/pin/pink/qpsolvers[daqp]/eclipse-zenoh/msgpack (pinned in
# requirements.txt to the exact versions verified end-to-end against the real trained
# checkpoint -- see requirements.txt's header comment for why these differ from the plan's
# original speculative pin), the checkpoint (model.pt, processor/, config.json,
# training_args.json, stats_data.json), the URDF + meshes, and origami/ itself.
# No secrets, no dataset, no .git (see .dockerignore).
#
# Entrypoint: `python -m origami.serve_zenoh`, no required CLI args -- everything comes from
# environment variables (ORIGAMI_ZENOH_ENDPOINT / ORIGAMI_SESSION_ID are the contractual ones
# the organizer sets per docs/participant_zenoh_submission.md §2; ORIGAMI_CHECKPOINT_PATH /
# ORIGAMI_LOCKED_CONFIG_SOURCE / ORIGAMI_ACTION_HORIZON / ORIGAMI_SLOW_EVERY point at this
# image's own baked-in paths and this project's own §9.1 cadence choice).

FROM nvidia/cuda:12.6.0-runtime-ubuntu24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

# ── Python deps (cache-friendly: before any code/checkpoint COPY) ──────────────────────────
COPY origami/docker/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ── T-Rex: only the subset origami/ actually imports at deploy time (§13's import list) --
# NOT hardware_code/ (252MB, heavy hardware SDK imports, never imported by serve_zenoh -- its
# 3 functions are copied into origami/retarget.py and origami/kinematics.py, with citations,
# specifically to avoid this dependency) and NOT dataset_quickstart/ (354MB, prep-only). ──
COPY T-Rex/__init__.py /app/T-Rex/__init__.py
COPY T-Rex/utils /app/T-Rex/utils
COPY T-Rex/qwen_vla /app/T-Rex/qwen_vla
COPY T-Rex/tactile_vqvae /app/T-Rex/tactile_vqvae
COPY T-Rex/scripts /app/T-Rex/scripts

# ── Official robot asset (URDF + meshes) ────────────────────────────────────────────────────
COPY north_poc2_2_urdf_usd /app/north_poc2_2_urdf_usd

# ── origami/ itself, plus the one SDK file it loads at import time: the policy-server
# template whose OrigamiZenohServer + msgpack codec origami/serve_zenoh.py keeps
# byte-identical (§9.4) -- resolved relative to the repo root, so the path below must match. --
COPY origami /app/origami
COPY origami-inference-kit-participant/sharpa_north_ces_lite_sdk-main/examples/policy_server_template.py \
     /app/origami-inference-kit-participant/sharpa_north_ces_lite_sdk-main/examples/policy_server_template.py

# ── The trained checkpoint. Build-arg so CI can swap checkpoints without editing this file. --
ARG CHECKPOINT_DIR=checkpoints/sept9_ckpt
COPY ${CHECKPOINT_DIR} /app/checkpoint

# ── locked_config source (§3.2/G1c): only needed while a checkpoint's own
# training_args.json["locked_config"] carries just a digest (see origami/policy.py's
# resolve_locked_config docstring for the real gap this recovers). Optional -- omit the
# build-arg (leave it pointing at a nonexistent path) once a checkpoint records the full
# LockedConfig itself. ──
ARG LOCKED_CONFIG_SOURCE=data/meta/origami_prep.json
COPY ${LOCKED_CONFIG_SOURCE} /app/locked_config_source.json

ENV ORIGAMI_CHECKPOINT_PATH=/app/checkpoint \
    ORIGAMI_URDF_PATH=/app/north_poc2_2_urdf_usd/north_poc2_2_v3_1.urdf \
    ORIGAMI_LOCKED_CONFIG_SOURCE=/app/locked_config_source.json \
    ORIGAMI_ACTION_HORIZON=4 \
    ORIGAMI_SLOW_EVERY=4 \
    EXECUTION_MODE=async \
    ORIGAMI_CUDA=0

ENTRYPOINT ["python", "-m", "origami.serve_zenoh"]
