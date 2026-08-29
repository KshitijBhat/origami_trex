#!/usr/bin/env bash
# Prepare the Robotic Origami Challenge data for T-Rex post-training.
#
# Streams seasons from the HF hub one at a time, converts each to origami-flat,
# and deletes the source, so peak disk stays at "output so far + one season"
# (~1 GB) instead of the ~300 GB the raw release would need.
#
# The `lerobot3.0` exports are mostly AV1 with some h264 seasons, and this script
# decodes everything on the CPU, where AV1 costs roughly 2-4x more per frame.
# `run_prepare_fast.sh` overlaps the downloads and puts the RGB streams on NVDEC
# where the GPU supports the codec, and is what you want for the full tier.
#
# Usage:
#   bash trex_origami/run_prepare.sh pilot      # 10 train + 3 val seasons, stride 5
#   bash trex_origami/run_prepare.sh full       # 101 train + 25 val seasons, stride 20
#   bash trex_origami/run_prepare.sh dense      # 30 train + 8 val seasons, stride 5
#   ANCHOR_MODE=state PHASE_MODE=none bash ... full   # the attempt-2 prep
#
# Re-running is safe and resumable: already-converted episodes are skipped.
# Changing ANCHOR_MODE, PHASE_MODE or the stride is not -- the prep refuses to
# mix two contracts in one root, so use a fresh OUT_ROOT or --overwrite.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ── edit these for your machine ───────────────────────────────────────────────
OUT_ROOT="${OUT_ROOT:-/home/kshitij/origami_trex/data/origami_flat}"
CACHE_ROOT="${CACHE_ROOT:-/home/kshitij/origami_trex/data/_src}"
# export HF_TOKEN=hf_...   # only needed if the dataset repo is gated

# See trex_origami/anchoring.py and run_prepare_fast.sh for what these mean.
ANCHOR_MODE="${ANCHOR_MODE:-hybrid}"
PHASE_MODE="${PHASE_MODE:-none}"

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
echo ">>> anchor=${ANCHOR_MODE}  phase=${PHASE_MODE}"
echo ">>> out=${OUT_ROOT}/${TIER}  cache=${CACHE_ROOT}"

COMMON=(--cache-root "${CACHE_ROOT}" --sample-stride "${STRIDE}"
        --action-chunk 25 --chunk-stride 1 --image-size 224
        --vqvae-window 16 --phase-mode "${PHASE_MODE}"
        --anchor-mode "${ANCHOR_MODE}")

# A season that fails to download is not a reason to skip everything after it.
# prepare exits 3 ("wrote a usable split, some seasons missing"), and we carry
# on so the other split, the stats fit and the verifies still run -- then
# re-raise at the very end.  A harder failure (bad flag, no output at all)
# still aborts immediately.
PARTIAL=""
prepare_split() {                        # prepare_split <split> <limit> <out-root>
  local split="$1" limit="$2" out="$3" rc=0
  python3 -m trex_origami.prepare --split "${split}" --limit "${limit}" \
          --out-root "${out}" "${COMMON[@]}" || rc=$?
  case "${rc}" in
    0) ;;
    3) echo ">>> WARNING: ${split} split is incomplete -- some seasons failed (see above)" >&2
       PARTIAL="${PARTIAL} ${split}" ;;
    *) echo ">>> ${split} split failed hard (exit ${rc}), stopping" >&2; exit "${rc}" ;;
  esac
}

prepare_split train "${TRAIN_LIMIT}" "${TRAIN_ROOT}"
prepare_split val   "${VAL_LIMIT}"   "${VAL_ROOT}"

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

if [[ -n "${PARTIAL}" ]]; then
  echo >&2
  echo ">>> incomplete split(s):${PARTIAL} -- rerun this script to refetch only" >&2
  echo "    the failed seasons; converted ones are skipped." >&2
  exit 3
fi
