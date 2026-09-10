# syntax=docker/dockerfile:1
# Submission image for the origami-zenoh-v1 policy service. REDESIGN_PLAN.md §9.5, §12 step 15.
#
# Build context MUST be the repo root (`origami_trex/`), so this Dockerfile can COPY the URDF,
# T-Rex submodule subset, and origami/ package. The checkpoint is supplied as a separate named
# build context (BuildKit additional context, hence the `# syntax` line above) rather than a
# path inside the main context -- this repo root lives at /workspace/origami_trex while the
# checkpoint lives at /workspace/checkpoints/<name>, a different directory tree entirely.
# Example:
#   docker build -f origami/docker/Dockerfile \
#     --build-context checkpoint=/workspace/checkpoints/sept9_ckpt \
#     -t origami-submission:precompile .
#
# Contains: torch/transformers/pin/pink/qpsolvers[daqp]/eclipse-zenoh/msgpack (pinned in
# requirements.txt to the exact versions verified end-to-end against the real trained
# checkpoint -- see requirements.txt's header comment), the checkpoint (model.pt, processor/,
# config.json, training_args.json, stats_data.json), the URDF + meshes, origami/ itself, and a
# baked torch.compile cache (see section below -- empty on the first build; populated by
# origami/compile_warmup.py in a throwaway writable container, then baked into the final
# build; see the submission runbook). No secrets, no dataset, no .git (see .dockerignore).
#
# Entrypoint: origami/docker/entrypoint.sh -> `python -m origami.serve_zenoh`, no required CLI
# args -- everything comes from environment variables (ORIGAMI_ZENOH_ENDPOINT /
# ORIGAMI_SESSION_ID are the contractual ones the organizer sets per
# docs/participant_zenoh_submission.md §2; ORIGAMI_CHECKPOINT_PATH / ORIGAMI_LOCKED_CONFIG_SOURCE
# / ORIGAMI_ACTION_HORIZON / ORIGAMI_SLOW_EVERY / ORIGAMI_COMPILE point at this image's own
# baked-in paths and this project's own §9.1 cadence choice). Runs as a non-root user with a
# read-only rootfs at evaluation time (docs/container_submission.md §7) -- see the runbook for
# the exact `docker run` hardening flags.

FROM nvidia/cuda:12.6.0-runtime-ubuntu24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/tmp/origami-home \
    USER=origami-policy \
    LOGNAME=origami-policy \
    XDG_CACHE_HOME=/tmp/origami-cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev python3-pip \
        libgl1 libglib2.0-0 libgomp1 gcc g++ \
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
COPY --chown=65532:65532 --chmod=755 T-Rex/__init__.py /app/T-Rex/__init__.py
COPY --chown=65532:65532 --chmod=755 T-Rex/utils /app/T-Rex/utils
COPY --chown=65532:65532 --chmod=755 T-Rex/qwen_vla /app/T-Rex/qwen_vla
COPY --chown=65532:65532 --chmod=755 T-Rex/tactile_vqvae /app/T-Rex/tactile_vqvae
COPY --chown=65532:65532 --chmod=755 T-Rex/scripts /app/T-Rex/scripts

# ── Official robot asset (URDF + meshes) ────────────────────────────────────────────────────
COPY --chown=65532:65532 --chmod=755 north_poc2_2_urdf_usd /app/north_poc2_2_urdf_usd

# ── origami/ itself, plus the one SDK file it loads at import time: the policy-server
# template whose OrigamiZenohServer + msgpack codec origami/serve_zenoh.py keeps
# byte-identical (§9.4) -- resolved relative to the repo root, so the path below must match. --
COPY --chown=65532:65532 --chmod=755 origami /app/origami
COPY --chown=65532:65532 --chmod=755 \
     origami-inference-kit-participant/sharpa_north_ces_lite_sdk-main/examples/policy_server_template.py \
     /app/origami-inference-kit-participant/sharpa_north_ces_lite_sdk-main/examples/policy_server_template.py

# ── The trained checkpoint, from the `checkpoint` named build context (see the header
# comment) so CI can swap checkpoints via `--build-context checkpoint=<path>` without editing
# this file. --chown/--chmod on the COPY itself: a later `chmod -R` would copy the ~16GB
# model.pt up into a fresh layer just to touch its metadata. ──
COPY --from=checkpoint --chown=65532:65532 --chmod=755 . /app/checkpoint

# ── locked_config source (§3.2/G1c): only needed while a checkpoint's own
# training_args.json["locked_config"] carries just a digest (sept9_ckpt's case -- see
# origami/policy.py's resolve_locked_config docstring for the real gap this recovers).
# Staged at origami/docker/locked_config_source.json (a copy of the val-root's own
# meta/origami_prep.json, matching sept9_ckpt's recorded digest 0bb3b0aa...) rather than
# under data/ -- data/ is .dockerignore'd on purpose (no dataset in the image). ──
ARG LOCKED_CONFIG_SOURCE=origami/docker/locked_config_source.json
COPY --chown=65532:65532 --chmod=644 ${LOCKED_CONFIG_SOURCE} /app/locked_config_source.json

COPY --chown=65532:65532 --chmod=755 origami/docker/entrypoint.sh /app/entrypoint.sh
COPY --chown=65532:65532 --chmod=644 origami/docker/licenses/ /licenses/

# ── Inference configuration -- read by origami.serve_zenoh.build_argument_parser. steps/
# split_step come from the checkpoint's own training_args.json (not overridable here); the
# cadence (T=4, slow_every=4) and torch.compile toggle are this deploy config's own choice and
# ARE part of the compile-cache key (changing ORIGAMI_COMPILE changes whether the compiled
# graphs are ever exercised) -- see the runbook section 2/4. ──
ENV ORIGAMI_CHECKPOINT_PATH=/app/checkpoint \
    ORIGAMI_URDF_PATH=/app/north_poc2_2_urdf_usd/north_poc2_2_v3_1.urdf \
    ORIGAMI_LOCKED_CONFIG_SOURCE=/app/locked_config_source.json \
    ORIGAMI_ACTION_HORIZON=4 \
    ORIGAMI_SLOW_EVERY=4 \
    EXECUTION_MODE=async \
    ORIGAMI_CUDA=0 \
    ORIGAMI_COMPILE=1 \
    TORCHINDUCTOR_CACHE_DIR=/app/compile-cache/torchinductor \
    TRITON_CACHE_DIR=/app/compile-cache/triton \
    TORCH_HOME=/tmp/origami-cache/torch

# ── Pre-baked torch.compile cache: generated by running origami/compile_warmup.py in a
# writable `--gpus all` container built from this same Dockerfile with an EMPTY
# origami/docker/compile-cache/, then `docker cp`'d out and rebuilt (runbook section 4). The
# runtime's read-only rootfs + noexec /tmp can only *load* kernels that already sit in this
# layer -- an empty directory builds fine (first call then compiles live, slowly, and would
# crash under --read-only), so never ship it empty. ──
COPY --chown=65532:65532 --chmod=755 origami/docker/compile-cache /app/compile-cache

USER 65532:65532

HEALTHCHECK NONE

ENTRYPOINT ["/app/entrypoint.sh"]
