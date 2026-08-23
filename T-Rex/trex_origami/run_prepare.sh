#!/usr/bin/env bash
# Prepare the Robotic Origami Challenge data for T-Rex post-training.
#
# Streams seasons from the HF hub one at a time, converts each to origami-flat,
# and deletes the source, so peak disk stays at "output so far + one season"
# (~1 GB) instead of the ~300 GB the raw release would need.
#
# Usage:
#   bash trex_origami/run_prepare.sh pilot      # 10 train + 3 val seasons, stride 5
#   bash trex_origami/run_prepare.sh full       # 101 train + 25 val seasons, stride 20
#   bash trex_origami/run_prepare.sh dense      # 30 train + 8 val seasons, stride 5
#
# Re-running is safe and resumable: already-converted episodes are skipped.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ── edit these for your machine ───────────────────────────────────────────────
OUT_ROOT="${OUT_ROOT:-/home/kshitij/origami_trex/data/origami_flat}"
CACHE_ROOT="${CACHE_ROOT:-/home/kshitij/origami_trex/data/_src}"
# export HF_TOKEN=hf_...   # only needed if the dataset repo is gated

TIER="${1:-pilot}"
case "${TIER}" in
  pilot) TRAIN_LIMIT=10;  VAL_LIMIT=3;  STRIDE=5  ;;
  dense) TRAIN_LIMIT=30;  VAL_LIMIT=8;  STRIDE=5  ;;
  full)  TRAIN_LIMIT=0;   VAL_LIMIT=0;  STRIDE=20 ;;   # 0 = every season in the split
  *) echo "unknown tier '${TIER}' (expected pilot | dense | full)" >&2; exit 2 ;;
esac

TRAIN_ROOT="${OUT_ROOT}/${TIER}/train"
VAL_ROOT="${OUT_ROOT}/${TIER}/val"

echo ">>> tier=${TIER}  stride=${STRIDE}  train_limit=${TRAIN_LIMIT}  val_limit=${VAL_LIMIT}"
echo ">>> out=${OUT_ROOT}/${TIER}  cache=${CACHE_ROOT}"

COMMON=(--cache-root "${CACHE_ROOT}" --sample-stride "${STRIDE}"
        --action-chunk 25 --chunk-stride 1 --image-size 224
        --vqvae-window 16 --phase-mode none)

python3 -m trex_origami.prepare --split train --limit "${TRAIN_LIMIT}" \
        --out-root "${TRAIN_ROOT}" "${COMMON[@]}"

python3 -m trex_origami.prepare --split val --limit "${VAL_LIMIT}" \
        --out-root "${VAL_ROOT}" "${COMMON[@]}"

# Normalisation is fit on train only and copied to val, so the two splits scale
# identically and their losses stay comparable.
python3 -m trex_origami.stats --root "${TRAIN_ROOT}" --subsample 4 --copy-to "${VAL_ROOT}"

python3 -m trex_origami.verify --root "${TRAIN_ROOT}" --samples 16 \
        --montage "${OUT_ROOT}/${TIER}/deform_montage.png"
python3 -m trex_origami.verify --root "${VAL_ROOT}" --samples 8

echo
echo ">>> done. train=${TRAIN_ROOT}  val=${VAL_ROOT}"
du -sh "${TRAIN_ROOT}" "${VAL_ROOT}"
echo ">>> upload with:"
echo "    huggingface-cli upload <your-user>/origami_flat_${TIER} ${OUT_ROOT}/${TIER} . --repo-type dataset --private"
