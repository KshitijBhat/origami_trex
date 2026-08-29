FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG EXECUTION_MODE=async

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp/origami-home \
    XDG_CACHE_HOME=/tmp/origami-cache \
    HF_HOME=/tmp/origami-cache/hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TORCH_HOME=/tmp/origami-cache/torch \
    TORCHINDUCTOR_CACHE_DIR=/tmp/origami-cache/torchinductor \
    TRITON_CACHE_DIR=/tmp/origami-cache/triton \
    EXECUTION_MODE=${EXECUTION_MODE}

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip ca-certificates \
    libgl1 libglib2.0-0 libgomp1 g++ gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch/torchvision from the cu128 index first: no --extra-index-url, or pip
# would pick a higher cu13x wheel from PyPI and silently defeat the pin.
# cu128 (not cu124) is required for sm_120 (Blackwell/RTX 50-series) kernel
# support -- confirmed via a real forward pass on an RTX 5090; see
# origami_trex/blackwell_torch_versions.md for the isolated-venv verification.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --default-timeout=100 --retries=5 \
    --index-url "${TORCH_INDEX_URL}" torch==2.11.0+cu128 torchvision==0.26.0+cu128

COPY requirements-inference.lock /tmp/requirements-inference.lock
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -r /tmp/requirements-inference.lock \
    && rm /tmp/requirements-inference.lock

COPY qwen_vla /app/qwen_vla
COPY tactile_vqvae /app/tactile_vqvae
COPY trex_policy_server.py /app/trex_policy_server.py
COPY entrypoint.sh /app/entrypoint.sh

# Checkpoint set: a training run's self-contained output dir (model.pt,
# config.json, training_args.json, stats_data.json, processor/). Swap this
# COPY to point at the final chosen checkpoint before building.
COPY checkpoints/model /app/checkpoints/model

ENV TREX_CKPT_PATH=/app/checkpoints/model \
    ORIGAMI_ACTION_HORIZON=25 \
    PYTHONPATH=/app

RUN chmod 0755 /app/entrypoint.sh \
    && chmod -R a+rX /app \
    && chown -R 65532:65532 /app

COPY licenses/ /licenses/
RUN chmod -R a+rX /licenses

USER 65532:65532

HEALTHCHECK NONE

ENTRYPOINT ["/app/entrypoint.sh"]
