# Self-contained OpenPI submission image for origami-zenoh-v1.
#
# Required named build contexts:
#   checkpoint      OpenPI checkpoint root containing params/ and assets/
#   python_packages site-packages root containing eclipse-zenoh 1.9

ARG RUNTIME_IMAGE=origami-openpi-runtime:dev
FROM ${RUNTIME_IMAGE}

USER root

# Add the public Zenoh Python binding to the existing OpenPI virtual environment.
COPY --from=python_packages /zenoh /.venv/lib/python3.11/site-packages/zenoh
COPY --from=python_packages /eclipse_zenoh-1.9.0.dist-info \
  /.venv/lib/python3.11/site-packages/eclipse_zenoh-1.9.0.dist-info

# Refresh the participant-visible OpenPI source and formal Zenoh server.
COPY src /app/src
COPY packages/openpi-client/src /app/packages/openpi-client/src
COPY scripts/serve_policy.py /app/scripts/serve_policy.py
COPY scripts/serve_policy_zenoh.py /app/scripts/serve_policy_zenoh.py
COPY scripts/docker/submission_zenoh_entrypoint.sh \
  /app/scripts/docker/submission_zenoh_entrypoint.sh

COPY --from=checkpoint / /opt/policy/checkpoint/

RUN chmod 0755 /app/scripts/docker/submission_zenoh_entrypoint.sh \
  && chown -R policy:policy /opt/policy/checkpoint

ENV CHECKPOINT_DIR=/opt/policy/checkpoint \
    HOME=/tmp/origami-home \
    XDG_CACHE_HOME=/tmp/origami-cache \
    JAX_COMPILATION_CACHE_DIR=/tmp/origami-jax-cache

USER policy
HEALTHCHECK NONE
ENTRYPOINT ["/app/scripts/docker/submission_zenoh_entrypoint.sh"]
