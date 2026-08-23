# Upstream/local WebSocket bundled image.
#
# Origami competition submissions must use
# submission-zenoh-bundled.Dockerfile instead.

ARG RUNTIME_IMAGE=origami-openpi-runtime:latest
FROM ${RUNTIME_IMAGE}

ARG POLICY_CONFIG=pi05_origami_fold_plane_v2_v0715
ENV POLICY_CONFIG=${POLICY_CONFIG}

USER root
COPY --chown=policy:policy submission_checkpoint/ /opt/policy/checkpoint/
USER policy
