#!/usr/bin/env bash
set -euo pipefail

if [ -z "${ORIGAMI_ZENOH_ENDPOINT:-}" ]; then
    echo "ERROR: ORIGAMI_ZENOH_ENDPOINT is required" >&2
    exit 1
fi
if [ -z "${ORIGAMI_SESSION_ID:-}" ]; then
    echo "ERROR: ORIGAMI_SESSION_ID is required" >&2
    exit 1
fi

: "${TREX_CKPT_PATH:=/app/checkpoints/model}"
: "${ORIGAMI_ACTION_HORIZON:=25}"

if [ ! -f "/app/qwen_vla/modeling_vla.py" ]; then
    echo "ERROR: /app/qwen_vla/modeling_vla.py not found" >&2
    exit 1
fi

for f in model.pt config.json training_args.json stats_data.json; do
    if [ ! -s "${TREX_CKPT_PATH}/${f}" ]; then
        echo "ERROR: ${TREX_CKPT_PATH}/${f} not found (or empty)" >&2
        exit 1
    fi
done
if [ ! -d "${TREX_CKPT_PATH}/processor" ] || [ -z "$(ls -A "${TREX_CKPT_PATH}/processor" 2>/dev/null)" ]; then
    echo "ERROR: ${TREX_CKPT_PATH}/processor/ not found or empty" >&2
    exit 1
fi

export HOME="${HOME:-/tmp/origami-home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/origami-cache}"
mkdir -p "$HOME" "$XDG_CACHE_HOME"

echo "=============================================="
echo "T-Rex Policy Server"
echo "=============================================="
echo "  Action horizon: ${ORIGAMI_ACTION_HORIZON}"
echo "  Checkpoint:     ${TREX_CKPT_PATH}"
echo "  Execution mode: ${EXECUTION_MODE:-async}"
echo "=============================================="

exec python3 /app/trex_policy_server.py
