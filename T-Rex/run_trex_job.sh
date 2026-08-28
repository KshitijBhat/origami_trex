#!/bin/bash
set -euo pipefail

# --- Raise file descriptor limit (same reason as vitacformer++'s run_ori_job.sh:
#     many DataLoader workers) ---
ulimit -n 65536

# --- Environment setup (must be self-contained for phd run) ---
export HOME=${HOME:-/home/sr5/sairaj.loke}

# Activate the venv created by scripts/setup_server_env.sh
source "${HOME}/other/venv_trex/bin/activate"

# Working directory -- must be T-Rex for PROJECT_ROOT/relative paths to work
cd "${HOME}/other/origami_trex/T-Rex"

export PROJECT_ROOT="${HOME}/other/origami_trex/T-Rex"
export ASSET_ROOT="${ASSET_ROOT:-${HOME}/other/trex_assets}"

# ============================================================================
# --- Configurable dataset source (env var override, same pattern as
#     vitacformer++'s DATASET_SRC) ---
# ============================================================================
DATA_ROOT_SRC="${DATA_ROOT_SRC:-${HOME}/other/new_data/origami_flat/full}"

# ============================================================================
# --- Scratch storage: copy dataset to local NVMe (same as run_ori_job.sh --
#     the origami-flat dataset is JPEG blobs, same NFS-spike concern applies)
# ============================================================================
USE_SCRATCH="${USE_SCRATCH:-1}"
DATA_ROOT_DST=""
JOB_ID="${B2_JOB_ID:-$$}"

if [ "$USE_SCRATCH" = "1" ]; then
    for scratch_dir in /scratch_h4 /scratch_h8 /scratch_a4 /scratch_a8; do
        if [ -d "$scratch_dir" ]; then
            DATA_ROOT_DST="${scratch_dir}/${USER}/${JOB_ID}/origami_flat"
            break
        fi
    done

    if [ -n "$DATA_ROOT_DST" ]; then
        echo "[Scratch] Found scratch at $(dirname "$DATA_ROOT_DST")"
        echo "[Scratch] Copying dataset from $DATA_ROOT_SRC to $DATA_ROOT_DST ..."
        mkdir -p "$DATA_ROOT_DST"
        cp -r "$DATA_ROOT_SRC"/* "$DATA_ROOT_DST/"
        export DATA_ROOT="$DATA_ROOT_DST"
        echo "[Scratch] Dataset copied. Training from local NVMe."

        cleanup_scratch() {
            echo "[Scratch] Cleaning up $DATA_ROOT_DST ..."
            du -h -d 2 "$DATA_ROOT_DST"
            rm -rf "$DATA_ROOT_DST"
        }
        trap cleanup_scratch EXIT
    else
        echo "[Scratch] No scratch dir found. Training from NFS (home storage)."
        export DATA_ROOT="$DATA_ROOT_SRC"
    fi
else
    echo "[Scratch] USE_SCRATCH=0 -- skipping scratch copy. Training from NFS."
    export DATA_ROOT="$DATA_ROOT_SRC"
fi
# ============================================================================

# Persistent, NOT scratch/ephemeral -- train_origami.sh's OUTPUT_DIR guard
# refuses to start a long run pointed at ephemeral storage.
export OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/other/trex_outputs}"
export WANDB_MODE="${WANDB_MODE:-offline}"

# --- Print config summary ---
echo "============================================"
echo "NUM_GPUS        : ${1:-1}"
echo "TRAIN_BSZ       : ${TRAIN_BSZ:-8} (per GPU)"
echo "GRAD_ACCUM      : ${GRAD_ACCUM:-4}"
echo "LR              : ${LR:-5e-5}  <-- scale this yourself if you raise NUM_GPUS/TRAIN_BSZ"
echo "N_EPOCHS        : ${N_EPOCHS:-2}"
echo "DATA_ROOT_SRC   : $DATA_ROOT_SRC"
echo "OUTPUT_DIR      : $OUTPUT_DIR"
echo "============================================"

# --- Run training ---
# Usage: bash run_trex_job.sh [NUM_GPUS]
# Default: 1 GPU (train_origami.sh's own default -- this model + the pilot
# tier was only ever validated single-GPU; see cmd_trex.sh before assuming
# a 4-8 GPU run just works without retuning LR/batch).
NUM_GPUS="${1:-1}"
if [[ ! "$NUM_GPUS" =~ ^(1|2|4|8)$ ]]; then
    echo "ERROR: NUM_GPUS must be one of: 1, 2, 4, 8. Got: $NUM_GPUS"
    exit 1
fi

NUM_GPUS="$NUM_GPUS" bash scripts/train_origami.sh

# ============================================================================
# Example submission commands -- see cmd_trex.sh for the full matrix.
# ============================================================================
# EXPERIMENT_NAME=aug28_pilot_4gpu \
# phd run -ng 4 -p shr_gpu -GR H100 -l %J.log sh run_trex_job.sh 4
