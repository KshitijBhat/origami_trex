#!/usr/bin/env bash
# T-Rex post-training on the Robotic Origami Challenge fold-plane data.
# Single A100 (40 GB), no DeepSpeed.
#
#   bash scripts/train_origami.sh              # train
#   RESUME=1 bash scripts/train_origami.sh     # continue after a Colab pre-emption
#   SMOKE=1  bash scripts/train_origami.sh     # 5 steps, proves the wiring
#
# Everything below can be overridden from the environment, e.g.
#   TRAIN_BSZ=4 GRAD_ACCUM=8 bash scripts/train_origami.sh
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/content/T-Rex}"
ASSET_ROOT="${ASSET_ROOT:-/content/assets}"
DATA_ROOT="${DATA_ROOT:-/content/data/origami_flat}"
OUTPUT_DIR="${OUTPUT_DIR:-/content/outputs}"

cd "${PROJECT_ROOT}/scripts"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false
# The MoT backbone allocates in bursty shapes; expandable segments keeps a long
# run from fragmenting itself into an OOM after a few thousand steps.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ORIGIN_MODEL_PATH="${ORIGIN_MODEL_PATH:-${ASSET_ROOT}/Qwen3-VL-2B-Instruct}"
ORIGAMI_ROOT="${ORIGAMI_ROOT:-${DATA_ROOT}/train}"
ORIGAMI_VAL_ROOT="${ORIGAMI_VAL_ROOT:-${DATA_ROOT}/val}"

# Resolve the midtrain checkpoint (the directory holding model.pt).
if [ -z "${RESUME_CHECKPOINT:-}" ]; then
    RESUME_CHECKPOINT="$(dirname "$(find "${ASSET_ROOT}/trex_midtrain" -name model.pt | head -1)")"
fi
if [ ! -f "${RESUME_CHECKPOINT}/model.pt" ]; then
    echo "no model.pt under RESUME_CHECKPOINT=${RESUME_CHECKPOINT}; run scripts/colab_setup.sh" >&2
    exit 1
fi

EXPERIMENT_NAME="${EXPERIMENT_NAME:-trex_origami_fold_plane}"
RUN_NAME="${RUN_NAME:-${EXPERIMENT_NAME}_$(date +%m%d_%H%M)}"

# ── action space ──────────────────────────────────────────────────────────────
# 65-D absolute-radian joints is the competition's wire contract, and the
# dataset has no eef poses (observation.state.tcp is identically zero), so
# T-Rex's native eef-62 is unavailable.  Chunk 25 == the kit's action_horizon;
# no weight depends on chunk length, only the number of action tokens.
ACTION_DIM=65
ACTION_CHUNK=25

# ── budget ────────────────────────────────────────────────────────────────────
# The paper trains at effective batch 128 (16 x 8 GPUs) with LR 1e-4.  At
# effective 32 on one GPU, LR is scaled down accordingly.
# Rough A100-40GB planning numbers for the `pilot` tier (~160k samples at
# stride 5): ~20k micro-steps per epoch at batch 8, so roughly 2-2.5 h/epoch.
# Two epochs fits a single session with room for the eval pass.
TRAIN_BSZ="${TRAIN_BSZ:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-5e-5}"
N_EPOCHS="${N_EPOCHS:-2}"

EXTRA_ARGS=()
if [ "${SMOKE:-0}" = "1" ]; then
    echo ">>> SMOKE RUN: 5 steps, no checkpoint"
    EXTRA_ARGS+=(--max_steps 5 --save_steps 0 --save_freq 100000 --val_freq 0)
else
    EXTRA_ARGS+=(--save_steps "${SAVE_STEPS:-2000}" --save_optimizer_state 1
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

# Expected startup warnings, both benign:
#   "DeformEncoder checkpoint not found at " — no --deform_encoder_ckpt is passed
#      because the midtrain model.pt already carries deform_encoder.* weights.
#   shape-mismatch drops for x_embedder / final_layer / final_layer_tactile /
#      state_embedder — the 62->65 action-head change. Anything *else* being
#      dropped means the checkpoint does not match this architecture.

echo ">>> data   : ${ORIGAMI_ROOT}  (val ${ORIGAMI_VAL_ROOT})"
echo ">>> resume : ${RESUME_CHECKPOINT}"
echo ">>> output : ${OUTPUT_DIR}/${EXPERIMENT_NAME}/${RUN_NAME}"
echo ">>> batch  : ${TRAIN_BSZ} x ${GRAD_ACCUM} = $((TRAIN_BSZ * GRAD_ACCUM))  lr=${LR}"

accelerate launch \
    --num_processes 1 --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
    train.py \
    --model_path "${ORIGIN_MODEL_PATH}" \
    --data_format origami \
    --origami_root "${ORIGAMI_ROOT}" \
    --origami_val_root "${ORIGAMI_VAL_ROOT}" \
    --origami_sampler block --origami_pool_groups 32 --origami_cache_groups 8 \
    --output_dir "${OUTPUT_DIR}" --log_dir "${OUTPUT_DIR}" \
    --experiment_name "${EXPERIMENT_NAME}" --run_name "${RUN_NAME}" \
    --n_epochs "${N_EPOCHS}" --save_freq 1 --max_ckpts 3 \
    --action_dim ${ACTION_DIM} --action_chunk ${ACTION_CHUNK} \
    --image_size 224 224 \
    --train_bsz_per_gpu "${TRAIN_BSZ}" --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --learning_rate "${LR}" --min_lr_ratio 0.05 --warmup_rates 0.03 \
    --weight_decay 0 --max_grad_norm 1.0 \
    --optim adamw8bit --gradient_checkpointing 1 \
    --freeze_latent_expert 1 --train_latent_last_n "${TRAIN_LATENT_LAST_N:-4}" \
    --num_workers "${NUM_WORKERS:-8}" \
    --use_robot_state 1 \
    --use_tactile_vec 1 --use_tactile_deform 1 --use_tactile_vqvae 1 \
    --state_noise_mode joint \
    --tactile_intermediate_size 1536 \
    --training_stage 2 \
    --cascaded_total_steps 10 --cascaded_split_step 6 \
    --cascaded_tactile_dropout 0.1 --cascaded_loss_weight 1.0 \
    --use_flare 1 --n_flare_tokens_per_frame 4 --n_flare_steps 8 \
    --flare_loss_weight "${FLARE_LOSS_WEIGHT:-0.0}" --flare_frame_stride 4 --flare_layer_index -1 \
    --resume_checkpoint "${RESUME_CHECKPOINT}" --resume_source midtrain \
    --seed 42 \
    "${EXTRA_ARGS[@]}"

echo ">>> post-training finished."
