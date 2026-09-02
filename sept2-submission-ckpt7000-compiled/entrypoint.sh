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

for f in /app/serve_origami_zenoh.py /app/trex_origami/policy.py /app/qwen_vla/modeling_vla.py \
         /app/trex_origami/joint_limits.json; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found" >&2
        exit 1
    fi
done
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
if [ "${TREX_COMPILE:-0}" = "1" ] && [ -z "$(ls -A "${TORCHINDUCTOR_CACHE_DIR:-/app/compile-cache/torchinductor}" 2>/dev/null)" ]; then
    echo "WARNING: TREX_COMPILE=1 but ${TORCHINDUCTOR_CACHE_DIR:-/app/compile-cache/torchinductor} is empty;" >&2
    echo "         the first call will compile live (minutes) and FAIL on a read-only rootfs." >&2
    echo "         This is only acceptable for the compile-cache generation container." >&2
fi

export HOME="${HOME:-/tmp/origami-home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/origami-cache}"
mkdir -p "$HOME" "$XDG_CACHE_HOME"

echo "=============================================="
echo "T-Rex Origami Policy Server (origami-zenoh-v1)"
echo "=============================================="
echo "  Checkpoint:     ${TREX_CKPT_PATH}"
echo "  Execution mode: ${EXECUTION_MODE:-async}"
echo "  Flow:           ${TREX_MODE:-cascaded} steps=${TREX_TOTAL_STEPS:-10}/${TREX_SPLIT_STEP:-6} K=${TREX_N_DRAWS:-8}"
echo "  Anchor/safety:  ${TREX_ANCHOR_SOURCE:-state_offset} / ${TREX_SAFETY:-tol}"
echo "  torch.compile:  ${TREX_COMPILE:-0}  (cache ${TORCHINDUCTOR_CACHE_DIR:-<default>})"
echo "=============================================="

# All policy flags default from the TREX_* environment (see
# trex_origami.policy.add_policy_arguments); endpoint/session from ORIGAMI_*.
exec python3 /app/serve_origami_zenoh.py
