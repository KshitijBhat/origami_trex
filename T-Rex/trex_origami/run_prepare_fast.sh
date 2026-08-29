#!/usr/bin/env bash
# Prepare the Robotic Origami Challenge data, overlapping fetch with convert and
# decoding the RGB cameras on the GPU.  Drop-in replacement for run_prepare.sh.
#
# Two changes over the serial script:
#   * downloads run on a thread pool while conversions run on a process pool, so
#     the network and the CPU/GPU are busy at the same time instead of taking
#     turns;
#   * the three 480x480 RGB streams decode through NVDEC, with the decoder
#     chosen per file.  The 1200x480 deform strip stays on the CPU, where it is
#     measurably faster -- it is not downscaled, so reading full frames back off
#     the GPU costs more than NVDEC saves.
#
# Codecs: the `lerobot3.0` exports this pipeline reads are *mostly AV1*, with
# some h264 seasons.  That matters twice.  AV1 costs roughly 2-4x more CPU per
# frame through libdav1d, so the deform strip -- which never leaves the CPU --
# sets the pace and CONVERTERS is the knob that keeps the cores busy.  And NVDEC
# only decodes AV1 from Ampere on, so an A100/V100/T4 GPU-decodes the h264
# seasons and falls back for the rest; `trex_origami.accel` probes each codec
# once and prints the verdict on startup rather than warning per file.
#
# Peak disk is bounded by --disk-budget seasons (~1 GB each) plus the output.
#
# Usage:
#   bash trex_origami/run_prepare_fast.sh pilot
#   bash trex_origami/run_prepare_fast.sh full
#   CONVERTERS=3 DOWNLOADERS=4 bash trex_origami/run_prepare_fast.sh full
#   ORIGAMI_GPU=0 bash trex_origami/run_prepare_fast.sh full   # byte-exact CPU path
#   ANCHOR_MODE=state PHASE_MODE=none bash ... full            # the attempt-2 prep
#
# Re-running is safe and resumable: already-converted seasons are skipped
# without being refetched, so this can pick up after run_prepare.sh.  Changing
# ANCHOR_MODE, PHASE_MODE or the stride is *not* resumable -- the prep refuses
# to mix two contracts in one root, so use a fresh OUT_ROOT or --overwrite.
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

# ── action anchoring & prompt ─────────────────────────────────────────────────
# `hybrid` is the attempt-3 target space: the 14 arm dims are deltas from the
# previous command, the 44 hand dims and the 7 motor dims are absolute joint
# angles.  `state` reproduces the attempt-2 all-delta-from-state prep.  See
# trex_origami/anchoring.py for why.
#
# `progress` appends "(fold k of 6)" to the prompt.  The fraction is taken from
# the sample's position in its episode, which needs the episode length -- so at
# deployment it has to be approximated from elapsed time against the training
# median (`median_episode_frames` in meta/dataset.json; `scripts/test.py
# --phase_mode progress --phase_episode_seconds` does exactly that).  Set
# PHASE_MODE=none to train on the bare "north ces task" string instead.
ANCHOR_MODE="${ANCHOR_MODE:-hybrid}"
PHASE_MODE="${PHASE_MODE:-none}"

TIER="${1:-pilot}"
# VAL_FINE_LIMIT val seasons are prepared a second time at stride 5.  The
# robustness sweep and the temporal-ensemble rollout both quantize to one
# dataset row, so at the full tier's stride 20 they can only resolve 667 ms and
# chunks barely overlap -- neither measures what it claims.  Preparing a small
# stride-5 slice of the *same* val seasons is cheaper than keeping the pilot
# tier around for it.
case "${TIER}" in
  pilot) TRAIN_LIMIT=10;  VAL_LIMIT=3;  STRIDE=5;  VAL_FINE_LIMIT=0 ;;
  dense) TRAIN_LIMIT=30;  VAL_LIMIT=8;  STRIDE=5;  VAL_FINE_LIMIT=0 ;;
  full)  TRAIN_LIMIT=0;   VAL_LIMIT=0;  STRIDE=20; VAL_FINE_LIMIT=5 ;;  # 0 = whole split
  *) echo "unknown tier '${TIER}' (expected pilot | dense | full)" >&2; exit 2 ;;
esac
VAL_FINE_LIMIT="${VAL_FINE_LIMIT_OVERRIDE:-${VAL_FINE_LIMIT}}"

TRAIN_ROOT="${OUT_ROOT}/${TIER}/train"
VAL_ROOT="${OUT_ROOT}/${TIER}/val"
VAL_FINE_ROOT="${OUT_ROOT}/${TIER}/val_stride5"

echo ">>> tier=${TIER}  stride=${STRIDE}  train_limit=${TRAIN_LIMIT}  val_limit=${VAL_LIMIT}"
echo ">>> anchor=${ANCHOR_MODE}  phase=${PHASE_MODE}"
echo ">>> out=${OUT_ROOT}/${TIER}  cache=${CACHE_ROOT}"
echo ">>> downloaders=${DOWNLOADERS}  converters=${CONVERTERS}  disk_budget=${DISK_BUDGET} seasons"

COMMON=(--cache-root "${CACHE_ROOT}" --sample-stride "${STRIDE}"
        --action-chunk 25 --chunk-stride 1 --image-size 224
        --vqvae-window 16 --phase-mode "${PHASE_MODE}"
        --anchor-mode "${ANCHOR_MODE}"
        --downloaders "${DOWNLOADERS}" --converters "${CONVERTERS}"
        --disk-budget "${DISK_BUDGET}")

# A season that fails to download is not a reason to skip everything after it.
# prepare exits 3 ("wrote a usable split, some seasons missing"), and we carry
# on so the other split, the stats fit and the verifies still run -- then
# re-raise at the very end.  A harder failure (bad flag, no output at all)
# still aborts immediately.
PARTIAL=""
prepare_split() {            # prepare_split <split> <limit> <out-root> [extra...]
  local split="$1" limit="$2" out="$3" rc=0
  shift 3
  python3 -m trex_origami.prepare_fast --split "${split}" --limit "${limit}" \
          --out-root "${out}" "${COMMON[@]}" "$@" || rc=$?
  case "${rc}" in
    0) ;;
    3) echo ">>> WARNING: ${split} split is incomplete -- some seasons failed (see above)" >&2
       PARTIAL="${PARTIAL} ${split}" ;;
    *) echo ">>> ${split} split failed hard (exit ${rc}), stopping" >&2; exit "${rc}" ;;
  esac
}

prepare_split train "${TRAIN_LIMIT}" "${TRAIN_ROOT}"
prepare_split val   "${VAL_LIMIT}"   "${VAL_ROOT}"

COPY_TO=("${VAL_ROOT}")
if [[ "${VAL_FINE_LIMIT}" -gt 0 ]]; then
  echo ">>> stride-5 val slice: first ${VAL_FINE_LIMIT} val season(s) -> ${VAL_FINE_ROOT}"
  prepare_split val "${VAL_FINE_LIMIT}" "${VAL_FINE_ROOT}" --sample-stride 5
  COPY_TO+=("${VAL_FINE_ROOT}")
fi

# Normalisation is fit on train only and copied to every val root, so the splits
# scale identically and their losses stay comparable.
python3 -m trex_origami.stats --root "${TRAIN_ROOT}" --subsample 4 \
        --copy-to "${COPY_TO[@]}"

python3 -m trex_origami.verify --root "${TRAIN_ROOT}" --samples 16 \
        --montage "${OUT_ROOT}/${TIER}/deform_montage.png"
python3 -m trex_origami.verify --root "${VAL_ROOT}" --samples 8
# `if`, not `[[ ... ]] &&`: under `set -e` a false test at the end of a compound
# is a non-zero exit and would abort the script right before the summary.
if [[ "${VAL_FINE_LIMIT}" -gt 0 ]]; then
  python3 -m trex_origami.verify --root "${VAL_FINE_ROOT}" --samples 8
fi

echo
echo ">>> done. train=${TRAIN_ROOT}  val=${VAL_ROOT}"
du -sh "${TRAIN_ROOT}" "${VAL_ROOT}"
if [[ "${VAL_FINE_LIMIT}" -gt 0 ]]; then
  du -sh "${VAL_FINE_ROOT}"
  echo ">>> stride-5 val slice for robustness / rollout evals: ${VAL_FINE_ROOT}"
fi
echo ">>> upload with:"
echo "    huggingface-cli upload <your-user>/origami_flat_${TIER} ${OUT_ROOT}/${TIER} . --repo-type dataset --private"

if [[ -n "${PARTIAL}" ]]; then
  echo >&2
  echo ">>> incomplete split(s):${PARTIAL} -- rerun this script to refetch only" >&2
  echo "    the failed seasons; converted ones are skipped." >&2
  exit 3
fi
