#!/usr/bin/env bash
# T-Rex post-training on the Robotic Origami Challenge fold-plane data.
# memory-n-torque branch: adds cross-timestep memory (dev/memory) and
# joint-torque (feature/joint-torque) on top of the archive's (sept7)
# all-absolute recipe -- both are opt-in, default OFF, byte-identical
# behavior to the archive's recipe if you don't touch MEMORY_*/USE_TORQUE.
#
#   NUM_GPUS=1 bash scripts/train_origami.sh                    # train
#   NUM_GPUS=1 RESUME=1 bash scripts/train_origami.sh           # continue after pre-emption
#   NUM_GPUS=1 SMOKE=1  bash scripts/train_origami.sh           # 5 steps, proves the wiring
#   NUM_GPUS=1 SMOKE=1 SMOKE_NEW_FEATURES=1 bash scripts/train_origami.sh
#       # 5 steps with memory + torque BOTH forced on, regardless of
#       # MEMORY_SLOW_SECONDS/MEMORY_FAST/USE_TORQUE -- the one command to
#       # run before trusting a real phd job submission that the branch's
#       # new edits actually work end-to-end on whatever cluster this lands
#       # on, not just on the rented box they were smoke-tested on this
#       # session.
#
# NUM_GPUS has no default (see below) -- always pass it explicitly.
#
# Everything below can be overridden from the environment, e.g.
#   NUM_GPUS=1 TRAIN_BSZ=4 GRAD_ACCUM=8 bash scripts/train_origami.sh
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
# TF32 for fp32 ops (VQ-VAE, deform encoder) -- free precision/speed tradeoff
# on Ampere+ tensor cores, no-op (ignored) on older cards.
export TORCH_ALLOW_TF32=1

# No --instruction flag here on purpose: train.py resolves args.instruction
# from the dataset's own effective_instruction (meta/dataset.json's baked-in
# prep-time string) and records whatever that resolves to into
# training_args.json itself. Pass --instruction "..." yourself only if you
# deliberately want to override the dataset's own prompt for an ablation.

ORIGIN_MODEL_PATH="${ORIGIN_MODEL_PATH:-${ASSET_ROOT}/Qwen3-VL-2B-Instruct}"
ORIGAMI_ROOT="${ORIGAMI_ROOT:-${DATA_ROOT}/train}"
ORIGAMI_VAL_ROOT="${ORIGAMI_VAL_ROOT:-${DATA_ROOT}/val}"

# Resolve the midtrain checkpoint (the directory holding model.pt).
if [ -z "${RESUME_CHECKPOINT:-}" ]; then
    RESUME_CHECKPOINT="$(dirname "$(find "${ASSET_ROOT}/trex_midtrain_mecka23k_ucb100_vqvae_epoch6" -name model.pt | head -1)")"
fi
if [ ! -f "${RESUME_CHECKPOINT}/model.pt" ]; then
    echo "no model.pt under RESUME_CHECKPOINT=${RESUME_CHECKPOINT}; run scripts/colab_setup.sh" >&2
    exit 1
fi

# A checkpoint you cannot resume from is the same as no checkpoint. Refuse to
# start a long run pointed at ephemeral storage unless the caller says it is
# deliberate.
case "${OUTPUT_DIR}" in
    /content/drive/*|/mnt/*|"${HOME}"/*) ;;
    *)
        if [ "${ALLOW_EPHEMERAL_OUTPUT:-0}" != "1" ] && [ "${SMOKE:-0}" != "1" ]; then
            echo "OUTPUT_DIR=${OUTPUT_DIR} is not on persistent storage." >&2
            echo "A pre-emption would take the checkpoints with it. Point it at" >&2
            echo "  /content/drive/MyDrive/... , or set ALLOW_EPHEMERAL_OUTPUT=1." >&2
            exit 1
        fi
        echo ">>> WARNING: OUTPUT_DIR=${OUTPUT_DIR} is ephemeral (ALLOW_EPHEMERAL_OUTPUT=1)"
        ;;
esac

EXPERIMENT_NAME="${EXPERIMENT_NAME:-trex_origami_fold_plane}"
RUN_NAME="${RUN_NAME:-${EXPERIMENT_NAME}_$(date +%m%d_%H%M)}"

# No default: a silently-defaulted-to-1 process count is how a single-GPU
# launch ships unnoticed to an 8-GPU node. Caller must say how many
# (run_trex_job.sh already does this via its own NUM_GPUS arg).
NUM_GPUS="${NUM_GPUS:?NUM_GPUS must be set explicitly, e.g. NUM_GPUS=4 bash scripts/train_origami.sh}"

# ── action space ──────────────────────────────────────────────────────────────
# 65-D absolute-radian joints, all-absolute target (no anchor/delta -- the
# competition data has no eef poses, observation.state.tcp is identically
# zero). Chunk 16 matches how the real dataset (e.g.
# drakedrake/origami_preprocessed_stride_1) is actually prepared -- confirm
# against your own DATA_ROOT's meta/dataset.json "action_chunk" before
# changing this; a mismatch is a fail-loud error from origami_dataset.py,
# not a silent one.
ACTION_DIM=65
ACTION_CHUNK="${ACTION_CHUNK:-16}"

# ── budget & schedule ─────────────────────────────────────────────────────────
# TRAIN_BSZ / GRAD_ACCUM / LR / N_EPOCHS have no default here on purpose --
# run_trex_job.sh is the single source of truth for these (exported there
# before it calls this script). Set -u makes an unset one a loud failure
# instead of silently falling back to a number that could disagree with
# whatever run_trex_job.sh just printed in its config summary. Running this
# script directly (not via run_trex_job.sh) means you must export all four
# yourself.
: "${TRAIN_BSZ:?TRAIN_BSZ must be set (run_trex_job.sh sets this -- see its budget & schedule exports)}"
: "${GRAD_ACCUM:?GRAD_ACCUM must be set}"
: "${LR:?LR must be set}"
: "${N_EPOCHS:?N_EPOCHS must be set}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0}"

# ── loss masking over frozen/degenerate action dims ───────────────────────────
MASK_FROZEN_LOSS="${MASK_FROZEN_LOSS:-1}"

# ── quick-check dataset cap (0 = use every episode) ──────────────────────────
MAX_EPISODES="${MAX_EPISODES:-0}"

# ── episode duration cap (0 = full episode; 120 = first 120s per episode) ───
MAX_EPISODE_SECONDS="${MAX_EPISODE_SECONDS:-0}"

# ── latent-expert freeze ──────────────────────────────────────────────────────
# 1 = freeze the ~1.4B vision-language backbone except the top
# TRAIN_LATENT_LAST_N layers (single-GPU/rented-box budget). 0 = full
# fine-tune (needs meaningfully more memory + a real multi-GPU job -- see
# the dbg6 example in cmd_trex.sh, which paired FREEZE_LATENT_EXPT=0 with
# resuming from an already-partially-trained checkpoint, not cold-starting
# from midtrain).
# No default -- see the TRAIN_BSZ/GRAD_ACCUM/LR/N_EPOCHS note above, same
# reasoning: run_trex_job.sh is the single source of truth.
: "${FREEZE_LATENT_EXPT:?FREEZE_LATENT_EXPT must be set}"
USE_ROBOT_STATE="${USE_ROBOT_STATE:-0}"   # off by default -- never on during midtrain
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"

# ── cross-timestep memory (dev/memory) ────────────────────────────────────────
# Empty/0 = off, byte-identical to no memory at all. MEMORY_SLOW_SECONDS is a
# comma-separated list of real-second lookback targets (exponential spacing
# recommended, e.g. "0.25,0.5,1,5"); MEMORY_FAST is a linear window row-count
# (e.g. 4). See qwen_vla/MEMORY_DESIGN.md for the full design.
MEMORY_SLOW_SECONDS="${MEMORY_SLOW_SECONDS:-}"
MEMORY_FAST="${MEMORY_FAST:-0}"
MEMORY_ROPE_STRIDE_SLOW="${MEMORY_ROPE_STRIDE_SLOW:-32.0}"
MEMORY_ROPE_STRIDE_FAST="${MEMORY_ROPE_STRIDE_FAST:-8.0}"

# ── joint torque (feature/joint-torque) ───────────────────────────────────────
# 0/1. Mirrors USE_ROBOT_STATE's exact pattern -- see qwen_vla/origami_dataset.py
# and qwen_vla/modeling_vla.py's torque_embedder. Needs a norm_stats.json with
# a "torque" block (trex_origami/stats.py) -- fails loudly at dataset
# construction if you turn this on against stats computed before that column
# existed.
USE_TORQUE="${USE_TORQUE:-0}"

# One-shot override for a wiring smoke check: force memory + torque both on
# for this run only, regardless of the flags above. Doesn't touch
# MASK_FROZEN_LOSS/FREEZE_LATENT_EXPT/etc -- purely for confirming the new
# code paths execute on whatever machine this script lands on.
if [ "${SMOKE_NEW_FEATURES:-0}" = "1" ]; then
    # MEMORY_FAST already defaulted to "0" above (a real, non-empty value --
    # ":-" can't distinguish "user left it unset" from "already 0" the way
    # it can for MEMORY_SLOW_SECONDS's empty-string default), so force it
    # explicitly rather than relying on ":-" a second time.
    [ -z "${MEMORY_SLOW_SECONDS}" ] && MEMORY_SLOW_SECONDS="0.25,0.5,1,5"
    [ "${MEMORY_FAST}" = "0" ] && MEMORY_FAST=4
    USE_TORQUE=1
    echo ">>> SMOKE_NEW_FEATURES=1: forcing memory_slow_seconds=${MEMORY_SLOW_SECONDS} memory_fast=${MEMORY_FAST} use_torque=1"
fi

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

# Memory/torque flags, appended only when actually in use -- keeps the
# printed command line clean and matches origami_dataset.py's own no-op
# convention (empty/0 = feature doesn't exist for this run at all).
if [ -n "${MEMORY_SLOW_SECONDS}" ] || [ "${MEMORY_FAST}" != "0" ]; then
    EXTRA_ARGS+=(--memory_slow_seconds "${MEMORY_SLOW_SECONDS}" --memory_fast "${MEMORY_FAST}"
                 --memory_rope_stride_slow "${MEMORY_ROPE_STRIDE_SLOW}"
                 --memory_rope_stride_fast "${MEMORY_ROPE_STRIDE_FAST}")
fi
if [ "${USE_TORQUE}" = "1" ]; then
    EXTRA_ARGS+=(--use_torque 1)
fi

# Expected startup warnings, both benign:
#   "DeformEncoder checkpoint not found at " — no --deform_encoder_ckpt is passed
#      because the midtrain model.pt already carries deform_encoder.* weights.
#   shape-mismatch drops for x_embedder / final_layer / final_layer_tactile /
#      state_embedder / torque_embedder — the 62->65 action-head change, and
#      torque_embedder specifically is ALWAYS a fresh-init "missing" key since
#      no released checkpoint has ever had it. Anything *else* being dropped
#      means the checkpoint does not match this architecture.

# ── pre-flight ────────────────────────────────────────────────────────────────
if [ "${SKIP_PREFLIGHT:-0}" != "1" ] && [ "${SMOKE:-0}" != "1" ]; then
    PREFLIGHT_ARGS=(--train-root "${ORIGAMI_ROOT}" --val-root "${ORIGAMI_VAL_ROOT}"
                    --resume-checkpoint "${RESUME_CHECKPOINT}"
                    --batch-size "${TRAIN_BSZ}" --grad-accum "${GRAD_ACCUM}"
                    --epochs "${N_EPOCHS}" --save-steps "${SAVE_STEPS:-5000}")
    [ "${RESUME:-0}" = "1" ] && PREFLIGHT_ARGS+=(--resume-full-state)
    if ! (cd "${PROJECT_ROOT}" && python3 -m trex_origami.preflight "${PREFLIGHT_ARGS[@]}"); then
        if [ "${ALLOW_STATS_MISMATCH:-0}" = "1" ]; then
            echo ">>> pre-flight failed but ALLOW_STATS_MISMATCH=1 — continuing anyway"
        else
            echo "pre-flight failed; fix the above or set SKIP_PREFLIGHT=1 to override" >&2
            exit 1
        fi
    fi
fi

echo ">>> data   : ${ORIGAMI_ROOT}  (val ${ORIGAMI_VAL_ROOT})"
echo ">>> resume : ${RESUME_CHECKPOINT}"
echo ">>> output : ${OUTPUT_DIR}/${EXPERIMENT_NAME}/${RUN_NAME}"
echo ">>> batch  : ${TRAIN_BSZ} x ${GRAD_ACCUM} x ${NUM_GPUS} gpu(s) = $((TRAIN_BSZ * GRAD_ACCUM * NUM_GPUS))  lr=${LR}"
echo ">>> sched  : cosine -> ${MIN_LR_RATIO} x peak, warmup 3%, ${N_EPOCHS} epochs"
echo ">>> mask_frozen_loss=${MASK_FROZEN_LOSS}  max_episodes=${MAX_EPISODES}  max_episode_seconds=${MAX_EPISODE_SECONDS}  freeze_latent_expert=${FREEZE_LATENT_EXPT}  use_robot_state=${USE_ROBOT_STATE}"
echo ">>> memory : slow_seconds=[${MEMORY_SLOW_SECONDS}]  fast=${MEMORY_FAST}  rope_stride slow/fast=${MEMORY_ROPE_STRIDE_SLOW}/${MEMORY_ROPE_STRIDE_FAST}"
echo ">>> torque : use_torque=${USE_TORQUE}"
# prompt is printed by train.py itself once the dataset resolves it

# --- Background GPU memory logger (polls every 10s, zero GPU overhead) ---
GPU_LOG="${OUTPUT_DIR}/gpu_mem_${EXPERIMENT_NAME}.log"
mkdir -p "${OUTPUT_DIR}"
( while true; do
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
               --format=csv,noheader,nounits | \
    while IFS=',' read -r idx used total util; do
        echo "$(date '+%H:%M:%S') gpu${idx} mem ${used}/${total}MiB util ${util}%"
    done >> "${GPU_LOG}"
    sleep 10
done ) &
GPU_LOGGER_PID=$!
trap "kill ${GPU_LOGGER_PID} 2>/dev/null" EXIT

accelerate launch \
    --num_processes "${NUM_GPUS}" --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
    train.py \
    --model_path "${ORIGIN_MODEL_PATH}" \
    --data_format origami \
    --origami_root "${ORIGAMI_ROOT}" \
    --origami_val_root "${ORIGAMI_VAL_ROOT}" \
    --origami_sampler block --origami_pool_groups 32 --origami_cache_groups "${CACHE_GROUPS:-16}" \
    --max_episodes "${MAX_EPISODES}" \
    --max_episode_seconds "${MAX_EPISODE_SECONDS}" \
    --mask_frozen_loss "${MASK_FROZEN_LOSS}" \
    --output_dir "${OUTPUT_DIR}" --log_dir "${OUTPUT_DIR}" \
    --experiment_name "${EXPERIMENT_NAME}" --run_name "${RUN_NAME}" \
    --n_epochs "${N_EPOCHS}" --save_freq 1 --max_ckpts "${MAX_CKPTS:-3}" \
    --action_dim ${ACTION_DIM} --action_chunk ${ACTION_CHUNK} \
    --image_size 224 224 \
    --train_bsz_per_gpu "${TRAIN_BSZ}" --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --learning_rate "${LR}" --min_lr_ratio "${MIN_LR_RATIO}" --warmup_rates 0.03 \
    --weight_decay 0 --max_grad_norm 1.0 \
    --optim adamw8bit --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
    --freeze_latent_expert "${FREEZE_LATENT_EXPT}" --train_latent_last_n "${TRAIN_LATENT_LAST_N:-4}" \
    --num_workers "${NUM_WORKERS:-8}" \
    --use_robot_state "${USE_ROBOT_STATE}" \
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
