#!/usr/bin/env bash
# Example: serve a pi05 origami policy over websocket.
#
# This starts an OpenPI-style policy server on :8000 that a websocket client
# connects to. It shows how a pi05 policy is served; it does
# NOT ship any weights. Provide your own trained checkpoint dir.
#
# The checkpoint dir must contain:
#   <ckpt>/params/                       (JAX)  OR  <ckpt>/model.safetensors (PyTorch)
#   <ckpt>/assets/<asset_id>/norm_stats.json
# where <asset_id> matches the config (see src/openpi/training/config.py).
set -euo pipefail

OP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-pi05_origami_fold_plane}"   # train config name (src/openpi/training/config.py)
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"             # REQUIRED: path to your trained checkpoint dir
PORT="${PORT:-8000}"
PROMPT="${PROMPT:-north ces task}"

if [ -z "$CHECKPOINT_DIR" ]; then
  echo "[serve][ERROR] set CHECKPOINT_DIR to your trained checkpoint dir" >&2
  echo "  it must contain params/ (or model.safetensors) and assets/<asset_id>/norm_stats.json" >&2
  exit 1
fi

cd "$OP_DIR"
exec python scripts/serve_policy.py \
  --port="$PORT" --default-prompt="$PROMPT" \
  policy:checkpoint \
  --policy.config="$CONFIG" \
  --policy.dir="$CHECKPOINT_DIR"
