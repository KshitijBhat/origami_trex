#!/usr/bin/env bash
# All-absolute origami training launcher, self-contained for a fresh box
# (no PhD-cluster job-submission harness -- combines what run_trex_job.sh
# and train_origami.sh used to split between them).
#
#   NUM_GPUS=1 SMOKE=1 bash scripts/run_all_abs_ori.sh   # few steps, proves the wiring
#   NUM_GPUS=1 bash scripts/run_all_abs_ori.sh            # real run
#   NUM_GPUS=2 bash scripts/run_all_abs_ori.sh            # multi-GPU on one node
#
# Everything below can be overridden from the environment, e.g.
#   NUM_GPUS=1 TRAIN_BSZ=4 GRAD_ACCUM=8 MAX_EPISODES=1 bash scripts/run_all_abs_ori.sh
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace}"
ASSET_ROOT="${ASSET_ROOT:-/workspace/assets}"
DATA_ROOT="${DATA_ROOT:-/workspace/data/dummy}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs}"

cd "${PROJECT_ROOT}/scripts"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ORIGIN_MODEL_PATH="${ORIGIN_MODEL_PATH:-${ASSET_ROOT}/Qwen3-VL-2B-Instruct}"
ORIGAMI_ROOT="${ORIGAMI_ROOT:-${DATA_ROOT}/train}"
ORIGAMI_VAL_ROOT="${ORIGAMI_VAL_ROOT:-${DATA_ROOT}/val}"

if [ -z "${RESUME_CHECKPOINT:-}" ]; then
    RESUME_CHECKPOINT="$(dirname "$(find "${ASSET_ROOT}/trex_midtrain" -name model.pt | head -1)")"
fi
if [ ! -f "${RESUME_CHECKPOINT}/model.pt" ]; then
    echo "no model.pt under RESUME_CHECKPOINT=${RESUME_CHECKPOINT}" >&2
    exit 1
fi

EXPERIMENT_NAME="${EXPERIMENT_NAME:-trex_all_abs_ori}"
RUN_NAME="${RUN_NAME:-${EXPERIMENT_NAME}_$(date +%m%d_%H%M)}"

NUM_GPUS="${NUM_GPUS:?NUM_GPUS must be set explicitly, e.g. NUM_GPUS=1 bash scripts/run_all_abs_ori.sh}"

# ── action space ──────────────────────────────────────────────────────────────
# 65-D absolute-radian joints (this branch's own wire contract -- the dataset
# has no eef poses). Chunk 16, NOT the earlier attempt's 25: chunk_stride in
# prepare.py already spaces the 16 action_chunk steps at native ~33ms
# regardless of sample_stride, so 16 covers the same real-time horizon the
# T-Rex Action Expert operates at (paper: ~5Hz) without over-allocating tokens.
ACTION_DIM=65
ACTION_CHUNK=16

# ── budget & schedule ─────────────────────────────────────────────────────────
TRAIN_BSZ="${TRAIN_BSZ:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-1.5e-4}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0}"
N_EPOCHS="${N_EPOCHS:-3}"

# ── loss masking over frozen/degenerate action dims (new this branch) ────────
MASK_FROZEN_LOSS="${MASK_FROZEN_LOSS:-1}"

# ── quick-check dataset cap (new this branch; 0 = use every episode) ────────
MAX_EPISODES="${MAX_EPISODES:-0}"

EXTRA_ARGS=()
if [ "${SMOKE:-0}" = "1" ]; then
    echo ">>> SMOKE RUN: 5 steps, no checkpoint"
    EXTRA_ARGS+=(--max_steps 5 --save_steps 0 --save_freq 100000 --val_freq 0)
else
    EXTRA_ARGS+=(--save_steps "${SAVE_STEPS:-5000}" --save_optimizer_state 1
                 --val_freq "${VAL_FREQ:-1000}" --max_val_batches 30)
fi
if [ "${RESUME:-0}" = "1" ]; then
    LATEST="$(ls -dt "${OUTPUT_DIR}/${EXPERIMENT_NAME}"/*/checkpoint-* 2>/dev/null | head -1 || true)"
    if [ -z "${LATEST}" ]; then
        echo "RESUME=1 but no checkpoint under ${OUTPUT_DIR}/${EXPERIMENT_NAME}" >&2
        exit 1
    fi
    echo ">>> resuming full state from ${LATEST}"
    RESUME_CHECKPOINT="${LATEST}"
    RUN_NAME="$(basename "$(dirname "${LATEST}")")"
    EXTRA_ARGS+=(--resume_full_state 1)
fi

# Expected startup warnings, benign:
#   "DeformEncoder checkpoint not found at " -- the midtrain model.pt already
#     carries deform_encoder.* weights, no --deform_encoder_ckpt needed.
#   shape-mismatch drops for x_embedder / final_layer / final_layer_tactile /
#     state_embedder -- the 62->65 action-head change plus 25->16 action_chunk.
#     Anything *else* being dropped means the checkpoint does not match this
#     architecture.

echo ">>> data   : ${ORIGAMI_ROOT}  (val ${ORIGAMI_VAL_ROOT})"
echo ">>> resume : ${RESUME_CHECKPOINT}"
echo ">>> output : ${OUTPUT_DIR}/${EXPERIMENT_NAME}/${RUN_NAME}"
echo ">>> batch  : ${TRAIN_BSZ} x ${GRAD_ACCUM} x ${NUM_GPUS} gpu(s) = $((TRAIN_BSZ * GRAD_ACCUM * NUM_GPUS))  lr=${LR}"
echo ">>> mask_frozen_loss=${MASK_FROZEN_LOSS}  max_episodes=${MAX_EPISODES}"

accelerate launch \
    --num_processes "${NUM_GPUS}" --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
    train.py \
    --model_path "${ORIGIN_MODEL_PATH}" \
    --data_format origami \
    --origami_root "${ORIGAMI_ROOT}" \
    --origami_val_root "${ORIGAMI_VAL_ROOT}" \
    --origami_sampler block --origami_pool_groups 32 --origami_cache_groups 8 \
    --max_episodes "${MAX_EPISODES}" \
    --output_dir "${OUTPUT_DIR}" --log_dir "${OUTPUT_DIR}" \
    --experiment_name "${EXPERIMENT_NAME}" --run_name "${RUN_NAME}" \
    --n_epochs "${N_EPOCHS}" --save_freq 1 --max_ckpts 3 \
    --action_dim ${ACTION_DIM} --action_chunk ${ACTION_CHUNK} \
    --image_size 224 224 \
    --train_bsz_per_gpu "${TRAIN_BSZ}" --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --learning_rate "${LR}" --min_lr_ratio "${MIN_LR_RATIO}" --warmup_rates 0.03 \
    --weight_decay 0 --max_grad_norm 1.0 \
    --optim adamw8bit --gradient_checkpointing 1 \
    --freeze_latent_expert 1 --train_latent_last_n "${TRAIN_LATENT_LAST_N:-4}" \
    --num_workers "${NUM_WORKERS:-8}" \
    --use_robot_state 0 \
    --use_tactile_vec 1 --use_tactile_deform 1 --use_tactile_vqvae 1 \
    --state_noise_mode joint \
    --mask_frozen_loss "${MASK_FROZEN_LOSS}" \
    --tactile_intermediate_size 1536 \
    --training_stage 2 \
    --cascaded_total_steps 10 --cascaded_split_step 6 \
    --cascaded_tactile_dropout 0.1 --cascaded_loss_weight 1.0 \
    --use_flare 1 --n_flare_tokens_per_frame 4 --n_flare_steps 8 \
    --flare_loss_weight "${FLARE_LOSS_WEIGHT:-0.0}" --flare_frame_stride 4 --flare_layer_index -1 \
    --resume_checkpoint "${RESUME_CHECKPOINT}" --resume_source midtrain \
    --seed 42 \
    "${EXTRA_ARGS[@]}"

echo ">>> run finished."
