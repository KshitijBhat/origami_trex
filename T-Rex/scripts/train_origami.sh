#!/usr/bin/env bash
# T-Rex post-training on the Robotic Origami Challenge fold-plane data.
# Single A100 (40 GB), no DeepSpeed.
#
#   bash scripts/train_origami.sh              # train
#   RESUME=1 bash scripts/train_origami.sh     # continue after a Colab pre-emption
#   SMOKE=1  bash scripts/train_origami.sh     # 5 steps, proves the wiring
#
# Attempt-3 launch (see REIMPLEMENTATION_PLAN.md), on hybrid-anchored data:
#   LR=1.5e-4 N_EPOCHS=3 TRAIN_BSZ=128 GRAD_ACCUM=4 bash scripts/train_origami.sh
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

# No --instruction flag here on purpose: train.py resolves args.instruction
# from the dataset's own effective_instruction (meta/dataset.json's baked-in
# prep-time string, e.g. DATASET_TASK_STRING = "north ces task") and records
# whatever that resolves to into training_args.json itself (train.py:1194-1196,
# 707). Sourcing a default here too previously meant this script's own
# `from trex_origami import INSTRUCTION` (= "fold the paper into a paper
# airplane") could silently outrank the dataset's real prompt -- and per
# DEPLOY.md that phrasing is the one that measurably performs *worse* than the
# trained "north ces task" string. Pass --instruction "..." yourself only if
# you deliberately want to override the dataset's own prompt for an ablation.

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

# A checkpoint you cannot resume from is the same as no checkpoint.  On Colab
# /content is wiped when the VM is reclaimed, which is exactly the event the
# checkpoints exist for, so refuse to start a long run pointed there unless the
# caller says it is deliberate.
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

# ── action space ──────────────────────────────────────────────────────────────
# 65-D absolute-radian joints is the competition's wire contract, and the
# dataset has no eef poses (observation.state.tcp is identically zero), so
# T-Rex's native eef-62 is unavailable.  Chunk 25 == the kit's action_horizon;
# no weight depends on chunk length, only the number of action tokens.
ACTION_DIM=65
ACTION_CHUNK=25

# ── budget & schedule ─────────────────────────────────────────────────────────
# The authors' own post-train launcher (scripts/train.sh) runs LR 1e-4 at
# effective batch 128 with `--min_lr_ratio 0`, no warmup, for 100 epochs; the
# paper's Table 4 is 1e-4 at 384.  Attempt 2 ran 5e-5 at effective 512 -- about
# 8x below that recipe -- and its loss was flat by ~2k optimizer steps while
# still losing to both naive baselines.  The default here is 1.5e-4, between the
# paper-faithful 1e-4 and the 2e-4 that sqrt-scaling from 1e-4@128 to eff. 512
# would give.  Higher is tolerable because the latent expert is mostly frozen;
# if the first 200 steps look unstable, drop to LR=1e-4.
#
# `--min_lr_ratio 0` matches the authors: a single half-cosine decaying to zero,
# not to 5% of peak.  Warmup stays at 3% (a deliberate deviation -- we cold-start
# five head tensors for the 62->65 change and they do not).
#
# Rough A100-40GB planning numbers for the `pilot` tier (~160k samples at
# stride 5): ~20k micro-steps per epoch at batch 8, so roughly 2-2.5 h/epoch.
TRAIN_BSZ="${TRAIN_BSZ:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-1.5e-4}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0}"
N_EPOCHS="${N_EPOCHS:-3}"

# ── anchor augmentation ───────────────────────────────────────────────────────
# Only bites on a dataset whose arms are anchored to the previous command
# (`--anchor-mode hybrid`); a no-op on the old all-delta-from-state prep.  The
# anchor is a textbook causal-confusion shortcut -- and the open-loop MAE metric
# rewards copying it -- so training perturbs it with the measured tracking-error
# statistics and re-anchors ANCHOR_DROPOUT of samples to state[t], which is also
# what keeps the policy usable once its own command stream has drifted.
ANCHOR_NOISE_MODE="${ANCHOR_NOISE_MODE:-tracking}"
ANCHOR_DROPOUT="${ANCHOR_DROPOUT:-0.15}"

EXTRA_ARGS=()
if [ "${SMOKE:-0}" = "1" ]; then
    echo ">>> SMOKE RUN: 5 steps, no checkpoint"
    EXTRA_ARGS+=(--max_steps 5 --save_steps 0 --save_freq 100000 --val_freq 0)
else
    # 5000 micro-steps is ~3 h at the pilot's 2.25 s/step: frequent enough that a
    # pre-emption costs at most one checkpoint's worth of work, rare enough that
    # the ~10 GB each (weights + adamw8bit state, --save_optimizer_state 1) does
    # not spend the session on I/O.  At --save_steps 2000 a full-tier epoch
    # writes ~13 of them.
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

# Expected startup warnings, both benign:
#   "DeformEncoder checkpoint not found at " — no --deform_encoder_ckpt is passed
#      because the midtrain model.pt already carries deform_encoder.* weights.
#   shape-mismatch drops for x_embedder / final_layer / final_layer_tactile /
#      state_embedder — the 62->65 action-head change. Anything *else* being
#      dropped means the checkpoint does not match this architecture.

# ── pre-flight ────────────────────────────────────────────────────────────────
# Every check here fails in seconds; the alternative is finding out hours in, or
# not at all.  The stats-calibration check is the one that matters most: the
# full split refits its own q01/q99, so warm-starting from the *pilot*
# checkpoint would leave the action head emitting in the old normalisation and
# silently mis-scale every predicted delta.  Cold-starting from midtrain and
# resuming the same run after a pre-emption both pass.
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
echo ">>> batch  : ${TRAIN_BSZ} x ${GRAD_ACCUM} = $((TRAIN_BSZ * GRAD_ACCUM))  lr=${LR}"
echo ">>> sched  : cosine -> ${MIN_LR_RATIO} x peak, warmup 3%, ${N_EPOCHS} epochs"
echo ">>> anchor : noise=${ANCHOR_NOISE_MODE} dropout=${ANCHOR_DROPOUT}"
# prompt is printed by train.py itself once the dataset resolves it (see the
# note above -- this script no longer sources or overrides it)

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
    --learning_rate "${LR}" --min_lr_ratio "${MIN_LR_RATIO}" --warmup_rates 0.03 \
    --weight_decay 0 --max_grad_norm 1.0 \
    --optim adamw8bit --gradient_checkpointing 1 \
    --freeze_latent_expert 1 --train_latent_last_n "${TRAIN_LATENT_LAST_N:-4}" \
    --num_workers "${NUM_WORKERS:-8}" \
    --use_robot_state 1 \
    --use_tactile_vec 1 --use_tactile_deform 1 --use_tactile_vqvae 1 \
    --state_noise_mode joint \
    --anchor_noise_mode "${ANCHOR_NOISE_MODE}" --anchor_dropout "${ANCHOR_DROPOUT}" \
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
