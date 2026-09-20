#!/bin/bash
set -euo pipefail

# --- Raise file descriptor limit (same reason as vitacformer++'s run_ori_job.sh:
#     many DataLoader workers) ---
# ulimit -n 65536
# Set the soft limit to whatever the current maximum allowed hard limit is
ulimit -n $(ulimit -Hn)


# --- NCCL workarounds for H100 cluster P2P/SHM init failures ---
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
# export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
# export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

export NCCL_DEBUG=INFO


# --- Environment setup (must be self-contained for phd run) ---
export HOME=${HOME:-/home/sr5/sairaj.loke}

# Activate the venv created by scripts/setup_server_env.sh
source "${HOME}/other/.venv_trex/bin/activate"

# Working directory -- must be T-Rex for PROJECT_ROOT/relative paths to work
cd "${HOME}/other/ori/ori/origami_trex_sept20/T-Rex"

export PROJECT_ROOT="${HOME}/other/ori/ori/origami_trex_sept20/T-Rex"
export ASSET_ROOT="${ASSET_ROOT:-${HOME}/other/ori/ori/origami_trex_sept7/trex_assets}"

# ============================================================================
# --- Configurable dataset source (env var override, same pattern as
#     vitacformer++'s DATASET_SRC) ---
# ============================================================================
DATA_ROOT_SRC="${DATA_ROOT_SRC:-${HOME}/svb/SVB/external_folders/new_svb/otherdata/sept16_data/all_abs_data}"
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
export OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/other/ori/ori/origami_trex_sept20/trex_outputs_sept16/trex_outputs}"
export WANDB_MODE="${WANDB_MODE:-offline}"

# --- Budget & schedule: this script is the single source of truth for these
#     five -- train_origami.sh has no fallback of its own for them anymore
#     (it errors loudly via `set -u` if one arrives unset), so there is only
#     one place left that can disagree with what gets echoed below. ---
export TRAIN_BSZ="${TRAIN_BSZ:-16}"
export GRAD_ACCUM="${GRAD_ACCUM:-2}"
export LR="${LR:-3e-4}"
export N_EPOCHS="${N_EPOCHS:-10}"
export FREEZE_LATENT_EXPT="${FREEZE_LATENT_EXPT:-0}"

# --- Print config summary ---
echo "============================================"
echo "TRAIN_SCRIPT    : ${TRAIN_SCRIPT:-scripts/train_origami.sh}"
echo "NUM_GPUS        : ${1:-1}"
echo "TRAIN_BSZ       : ${TRAIN_BSZ} (per GPU)"
echo "GRAD_ACCUM      : ${GRAD_ACCUM}"
echo "LR              : ${LR}  <-- scale this yourself if you raise NUM_GPUS/TRAIN_BSZ"
echo "N_EPOCHS        : ${N_EPOCHS}"
echo "MAX_EPISODES    : ${MAX_EPISODES:-0}"
echo "MAX_EPISODE_SEC : ${MAX_EPISODE_SECONDS:-0}"
echo "MASK_FROZEN_LOSS: ${MASK_FROZEN_LOSS:-1}"
echo "FREEZE_LATENT_EXPT : ${FREEZE_LATENT_EXPT}"
echo "USE_ROBOT_STATE : ${USE_ROBOT_STATE:-0}"
echo "MEMORY_SLOW_SECONDS : ${MEMORY_SLOW_SECONDS:-<off>}"
echo "MEMORY_FAST     : ${MEMORY_FAST:-0}"
echo "USE_TORQUE      : ${USE_TORQUE:-0}"
echo "SMOKE_NEW_FEATURES : ${SMOKE_NEW_FEATURES:-0}  <-- 1 forces memory+torque on for a wiring check"
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

# Which recipe script actually runs. Defaults to scripts/train_origami.sh --
# on the memory-n-torque branch this is the same all-absolute recipe the
# archive's scripts/run_all_abs_ori.sh described, plus MEMORY_*/USE_TORQUE
# wired in (both opt-in, off by default). Point this at
# scripts/run_all_abs_ori.sh instead only if that file actually exists in
# your checkout -- it did not exist in the archive this was built from.
TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/train_origami.sh}"

# Whole environment (TRAIN_BSZ/GRAD_ACCUM/LR/N_EPOCHS/FREEZE_LATENT_EXPT
# exported above, plus MEMORY_SLOW_SECONDS/MEMORY_FAST/USE_TORQUE/
# SMOKE_NEW_FEATURES if the caller set them) passes through to whatever
# TRAIN_SCRIPT runs -- nothing extra needed here to thread flags in;
# train_origami.sh reads them directly.
NUM_GPUS="$NUM_GPUS" MAX_EPISODE_SECONDS="${MAX_EPISODE_SECONDS:-0}" bash "$TRAIN_SCRIPT"

# ============================================================================
# Example submission commands -- see cmd_trex.sh for the full matrix.
# ============================================================================
# EXPERIMENT_NAME=aug28_pilot_4gpu \
# phd run -ng 4 -p shr_gpu -GR H100 -l %J.log sh run_trex_job.sh 4

# --- memory-n-torque wiring check before a real job: run this first ---
# EXPERIMENT_NAME=dbg_memtorque_smoke SKIP_PREFLIGHT=1 SMOKE=1 SMOKE_NEW_FEATURES=1 \
#   phd run -ng 1 -p shr_gpu -GR H100 -l %J.log sh run_trex_job.sh 1

# --- real run, memory + torque both on ---
# EXPERIMENT_NAME=memtorque_run1 SKIP_PREFLIGHT=1 \
# MASK_FROZEN_LOSS=1 MAX_EPISODES=0 \
# MEMORY_SLOW_SECONDS=0.25,0.5,1,5 MEMORY_FAST=4 USE_TORQUE=1 FLARE_LOSS_WEIGHT=0.5 \
# TRAIN_BSZ=16 GRAD_ACCUM=2 LR=3e-4 N_EPOCHS=3 \
#   phd run -ng 4 -p shr_gpu -GR H100 -l %J.log sh run_trex_job.sh 4
