#!/usr/bin/env bash
# Vendors exactly the T-Rex source the serving path needs into this
# submission folder, traced from serve_origami_zenoh.py's imports.  The Docker
# build context cannot reach outside this directory, and the image must not
# depend on the training repo (accelerate, deepspeed, datasets, pyarrow, pyzmq).
#
#   ./scripts/sync_submission_files.sh            # sync, report what changed
#   ./scripts/sync_submission_files.sh --check    # exit 1 if stale, no writes
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMISSION_DIR="$(dirname "$SCRIPT_DIR")"
SRC="${TREX_SRC:-/workspace/origami_trex/T-Rex}"

# qwen_vla/__init__.py imports modeling_vla + modeling_qwen3vl_mot, which import
# diffusion.py and DeformAE.py.  origami_dataset.py is needed by
# trex_origami/policy.py for denormalize / clamp_frozen_absolute /
# split_deform_strip (its pyarrow import is lazy, inside the Dataset class).
QWEN_VLA_FILES=(__init__.py modeling_vla.py modeling_qwen3vl_mot.py diffusion.py DeformAE.py
                origami_dataset.py)

# modeling_vla.py imports tactile_vqvae.models.tactile_vqvae (use_tactile_vqvae=1).
TACTILE_VQVAE_FILES=(__init__.py models/__init__.py models/encoder.py models/decoder.py
                     models/quantizer.py models/tactile_vqvae.py)

# The deployment adapter and what it imports.  prepare/fetch/verify/stats/
# lerobot_v3/replay_sources are prep- and replay-only and deliberately excluded.
TREX_ORIGAMI_FILES=(__init__.py anchoring.py seasons.py loading.py policy.py joint_limits.json)

# scripts/<name> -> <submission>/<name>
TOP_FILES=(serve_origami_zenoh.py)

CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

[ -d "$SRC" ] || { echo "[fail] source not found: $SRC (set TREX_SRC)" >&2; exit 1; }

stale=0
report() { printf '  %-40s %s\n' "$1" "$2"; }

sync_file() {
    local rel="$1" src_root="$2" dst_root="$3" dst_rel="${4:-$1}"
    local s="$src_root/$rel" d="$dst_root/$dst_rel"
    [ -f "$s" ] || { echo "[fail] missing in source: $s" >&2; exit 1; }
    mkdir -p "$(dirname "$d")"
    if [ ! -f "$d" ] || ! diff -q "$s" "$d" >/dev/null 2>&1; then
        stale=1
        report "$dst_rel" "STALE"
        [ "$CHECK_ONLY" = "1" ] || cp "$s" "$d"
    else
        report "$dst_rel" "in sync"
    fi
}

echo "qwen_vla/"
for f in "${QWEN_VLA_FILES[@]}"; do sync_file "$f" "$SRC/qwen_vla" "$SUBMISSION_DIR/qwen_vla"; done
echo "tactile_vqvae/"
for f in "${TACTILE_VQVAE_FILES[@]}"; do sync_file "$f" "$SRC/tactile_vqvae" "$SUBMISSION_DIR/tactile_vqvae"; done
echo "trex_origami/"
for f in "${TREX_ORIGAMI_FILES[@]}"; do sync_file "$f" "$SRC/trex_origami" "$SUBMISSION_DIR/trex_origami"; done
echo "top level"
for f in "${TOP_FILES[@]}"; do sync_file "$f" "$SRC/scripts" "$SUBMISSION_DIR" "$f"; done

if [ "$CHECK_ONLY" = "1" ]; then
    if [ "$stale" = "1" ]; then
        echo "[fail] vendored source is STALE -- run ./scripts/sync_submission_files.sh" >&2
        exit 1
    fi
    echo "[ ok ] vendored source matches $SRC"
else
    [ "$stale" = "1" ] && echo "[ ok ] synced from $SRC" || echo "[ ok ] already up to date"
fi
