#!/usr/bin/env bash
# Prepare the Robotic Origami Challenge data, overlapping fetch with convert and
# decoding the RGB cameras on the GPU.  Drop-in replacement for run_prepare.sh.
#
# Two changes over the serial script:
#   * downloads run on a thread pool while conversions run on a process pool, so
#     the network and the CPU/GPU are busy at the same time instead of taking
#     turns;
#   * the three 480x480 RGB streams decode through NVDEC (`h264_cuvid -resize`),
#     which on this box runs at ~5100 fps against ~1950 fps on the CPU.  The
#     1200x480 deform strip stays on the CPU, where it is measurably faster --
#     it is not downscaled, so reading full frames back off the GPU costs more
#     than NVDEC saves.
#
# Peak disk is bounded by --disk-budget seasons (~1 GB each) plus the output.
#
# Usage:
#   bash trex_origami/run_prepare_fast.sh pilot
#   bash trex_origami/run_prepare_fast.sh full
#   CONVERTERS=3 DOWNLOADERS=4 bash trex_origami/run_prepare_fast.sh full
#   ORIGAMI_GPU=0 bash trex_origami/run_prepare_fast.sh full   # byte-exact CPU path
#
# Re-running is safe and resumable: already-converted seasons are skipped
# without being refetched, so this can pick up after run_prepare.sh.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ── edit these for your machine ───────────────────────────────────────────────
OUT_ROOT="${OUT_ROOT:-/home/kshitij/origami_trex/data/origami_flat}"
CACHE_ROOT="${CACHE_ROOT:-/home/kshitij/origami_trex/data/_src}"
# export HF_TOKEN=hf_...   # only needed if the dataset repo is gated

# Converters are processes doing ffmpeg + parquet; 2 is enough to keep both the
# GPU (rgb) and the CPU (deform) busy, and more mostly contends for the same
# 16 threads.  Downloaders are socket-bound, so they can outnumber them.
DOWNLOADERS="${DOWNLOADERS:-3}"
CONVERTERS="${CONVERTERS:-2}"
DISK_BUDGET="${DISK_BUDGET:-4}"

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
echo ">>> downloaders=${DOWNLOADERS}  converters=${CONVERTERS}  disk_budget=${DISK_BUDGET} seasons"

COMMON=(--cache-root "${CACHE_ROOT}" --sample-stride "${STRIDE}"
        --action-chunk 25 --chunk-stride 1 --image-size 224
        --vqvae-window 16 --phase-mode none
        --downloaders "${DOWNLOADERS}" --converters "${CONVERTERS}"
        --disk-budget "${DISK_BUDGET}")

# A season that fails to download is not a reason to skip everything after it.
# prepare exits 3 ("wrote a usable split, some seasons missing"), and we carry
# on so the other split, the stats fit and the verifies still run -- then
# re-raise at the very end.  A harder failure (bad flag, no output at all)
# still aborts immediately.
PARTIAL=""
prepare_split() {                        # prepare_split <split> <limit> <out-root>
  local split="$1" limit="$2" out="$3" rc=0
  python3 -m trex_origami.prepare_fast --split "${split}" --limit "${limit}" \
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
