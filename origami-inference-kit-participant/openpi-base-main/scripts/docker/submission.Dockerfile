# Upstream/local WebSocket runtime image.
#
# Origami competition submissions must use
# submission-zenoh-bundled.Dockerfile for the final image.
#
# The image intentionally contains no North robot transport, Zenoh configuration,
# protobuf definitions, or private SDK. During development a checkpoint is mounted
# read-only at /opt/policy/checkpoint. Teams must bundle their checkpoint in the
# final image they submit (see submission-bundled.Dockerfile).

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04@sha256:0bb88834d973ca1b450fcc2a05333c6fe45510bee289912a5391274c351c4a4d

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_PROJECT_ENVIRONMENT=/.venv \
    OPENPI_DATA_HOME=/opt/openpi-cache \
    PYTHONPATH=/app/src:/app/packages/openpi-client/src \
    PYTHONUNBUFFERED=1 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Prepared and checksum-verified by scripts/build_and_test_docker.sh. Keeping the
# archive in the context avoids proxy differences between host and Docker builds.
COPY .docker/uv-x86_64-unknown-linux-gnu.tar.gz /tmp/uv.tar.gz
RUN echo "942e29ff6769b096c7c99e9c3b1c08276869667d0d5d6121852dd9b9d875b3f3  /tmp/uv.tar.gz" \
       | sha256sum --check - \
    && tar -xzf /tmp/uv.tar.gz --strip-components=1 -C /usr/local/bin \
    && rm /tmp/uv.tar.gz

WORKDIR /app

# Install only serving dependencies. CUDA 12.4 and cuDNN 9.1 come from the
# NVIDIA base image, so JAX uses its local-CUDA plugin instead of downloading a
# second CUDA stack. PyTorch remains optional for PyTorch-based entries.
COPY requirements.submission.txt ./
RUN uv venv --python 3.11.9 "$UV_PROJECT_ENVIRONMENT" \
    && uv pip install --python "$UV_PROJECT_ENVIRONMENT" \
       --index-strategy unsafe-best-match \
       -r requirements.submission.txt

# JAX compiles kernels at startup; the runtime CUDA image does not contain
# ptxas/nvlink, so add only the pinned compiler-tools package.
RUN apt-get update \
    && apt-get install -y --no-install-recommends cuda-nvcc-12-4=12.4.131-1 \
    && rm -rf /var/lib/apt/lists/*

# OpenPI otherwise downloads this tokenizer from GCS during policy startup.
COPY .docker/paligemma_tokenizer.model /tmp/paligemma_tokenizer.model
RUN echo "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6  /tmp/paligemma_tokenizer.model" \
       | sha256sum --check - \
    && mkdir -p /opt/openpi-cache/big_vision \
    && mv /tmp/paligemma_tokenizer.model /opt/openpi-cache/big_vision/paligemma_tokenizer.model \
    && chmod 0777 /opt/openpi-cache /opt/openpi-cache/big_vision \
    && chmod 0666 /opt/openpi-cache/big_vision/paligemma_tokenizer.model

# The projects are imported directly through PYTHONPATH, so no build backend or
# editable-install metadata is needed in the runtime image.
COPY src src
COPY packages/openpi-client/src packages/openpi-client/src
COPY scripts/serve_policy.py scripts/serve_policy.py
COPY scripts/docker/submission_entrypoint.sh scripts/docker/submission_entrypoint.sh

RUN useradd --create-home --uid 10001 policy \
    && mkdir -p /opt/policy/checkpoint \
    && chown -R policy:policy /opt/policy

USER policy
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=180s --retries=6 \
    CMD ["/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).read()"]

ENTRYPOINT ["/app/scripts/docker/submission_entrypoint.sh"]
