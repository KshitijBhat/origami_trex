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

: "${ORIGAMI_CHECKPOINT_PATH:=/app/checkpoint}"

for f in /app/origami/serve_zenoh.py /app/origami/policy.py /app/origami/retarget.py \
         /app/T-Rex/qwen_vla/modeling_vla.py "${ORIGAMI_URDF_PATH:-/app/north_poc2_2_urdf_usd/north_poc2_2_v3_1.urdf}"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found" >&2
        exit 1
    fi
done
for f in model.pt config.json training_args.json stats_data.json; do
    if [ ! -s "${ORIGAMI_CHECKPOINT_PATH}/${f}" ]; then
        echo "ERROR: ${ORIGAMI_CHECKPOINT_PATH}/${f} not found (or empty)" >&2
        exit 1
    fi
done
if [ ! -d "${ORIGAMI_CHECKPOINT_PATH}/processor" ] || [ -z "$(ls -A "${ORIGAMI_CHECKPOINT_PATH}/processor" 2>/dev/null)" ]; then
    echo "ERROR: ${ORIGAMI_CHECKPOINT_PATH}/processor/ not found or empty" >&2
    exit 1
fi
if [ "${ORIGAMI_COMPILE:-0}" = "1" ] && [ -z "$(ls -A "${TORCHINDUCTOR_CACHE_DIR:-/app/compile-cache/torchinductor}" 2>/dev/null)" ]; then
    echo "WARNING: ORIGAMI_COMPILE=1 but ${TORCHINDUCTOR_CACHE_DIR:-/app/compile-cache/torchinductor} is empty;" >&2
    echo "         the first call will compile live (minutes) and FAIL on a read-only rootfs." >&2
    echo "         This is only acceptable for the compile-cache generation container." >&2
fi

export HOME="${HOME:-/tmp/origami-home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/origami-cache}"
mkdir -p "$HOME" "$XDG_CACHE_HOME"

echo "=============================================="
echo "origami-zenoh-v1 policy server (sept9_ckpt)"
echo "=============================================="
echo "  Checkpoint:     ${ORIGAMI_CHECKPOINT_PATH}"
echo "  Execution mode: ${EXECUTION_MODE:-async}"
echo "  Action horizon: ${ORIGAMI_ACTION_HORIZON:-4}  slow_every=${ORIGAMI_SLOW_EVERY:-4}"
echo "  torch.compile:  ${ORIGAMI_COMPILE:-0}  (cache ${TORCHINDUCTOR_CACHE_DIR:-<default>})"
echo "=============================================="

# All policy flags default from the ORIGAMI_* environment (see
# origami.serve_zenoh.build_argument_parser); endpoint/session are the two organizer-supplied
# contractual values (container_submission.md §2).
exec python -m origami.serve_zenoh
