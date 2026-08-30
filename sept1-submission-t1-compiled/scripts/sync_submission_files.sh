#!/usr/bin/env bash
# Vendors exactly the T-Rex source the inference path needs into this
# submission folder, traced from trex_policy_server.py's imports. The Docker
# build context cannot reach outside this directory, and the image must not
# depend on the training repo (accelerate, deepspeed, datasets, pyarrow).
#
#   ./scripts/sync_submission_files.sh            # sync, report what changed
#   ./scripts/sync_submission_files.sh --check    # exit 1 if stale, no writes
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMISSION_DIR="$(dirname "$SCRIPT_DIR")"
SRC="${TREX_SRC:-$(dirname "$(dirname "$SUBMISSION_DIR")")/origami_trex/T-Rex}"

# qwen_vla/__init__.py only imports modeling_vla + modeling_qwen3vl_mot; those
# import diffusion.py and DeformAE.py. attention_capture.py, origami_dataset.py,
# lerobot_dataset.py are training/eval-only and deliberately excluded.
QWEN_VLA_FILES=(__init__.py modeling_vla.py modeling_qwen3vl_mot.py diffusion.py DeformAE.py)

# modeling_vla.py imports tactile_vqvae.models.tactile_vqvae when
# use_tactile_vqvae=True (this checkpoint's setting). data/, eval.py,
# extract_codes.py, scripts/, train.py are training-only.
TACTILE_VQVAE_FILES=(__init__.py models/__init__.py models/encoder.py models/decoder.py models/quantizer.py models/tactile_vqvae.py)

CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

[ -d "$SRC" ] || { echo "[fail] source not found: $SRC (set TREX_SRC)" >&2; exit 1; }

stale=0
report() { printf '  %-40s %s\n' "$1" "$2"; }

sync_file() {
    local rel="$1" src_root="$2" dst_root="$3"
    local s="$src_root/$rel" d="$dst_root/$rel"
    [ -f "$s" ] || { echo "[fail] missing in source: $s" >&2; exit 1; }
    mkdir -p "$(dirname "$d")"
    if [ ! -f "$d" ] || ! diff -q "$s" "$d" >/dev/null 2>&1; then
        stale=1
        report "$rel" "STALE"
        [ "$CHECK_ONLY" = "1" ] || cp "$s" "$d"
    else
        report "$rel" "in sync"
    fi
}

echo "qwen_vla/"
for f in "${QWEN_VLA_FILES[@]}"; do
    sync_file "$f" "$SRC/qwen_vla" "$SUBMISSION_DIR/qwen_vla"
done

echo "tactile_vqvae/"
for f in "${TACTILE_VQVAE_FILES[@]}"; do
    sync_file "$f" "$SRC/tactile_vqvae" "$SUBMISSION_DIR/tactile_vqvae"
done

if [ "$CHECK_ONLY" = "1" ]; then
    if [ "$stale" = "1" ]; then
        echo "[fail] vendored source is STALE -- run ./scripts/sync_submission_files.sh" >&2
        exit 1
    fi
    echo "[ ok ] vendored source matches $SRC"
else
    [ "$stale" = "1" ] && echo "[ ok ] synced from $SRC" || echo "[ ok ] already up to date"
fi
