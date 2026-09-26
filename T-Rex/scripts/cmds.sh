#!/bin/bash
# Reference commands for running memory-dropout training on the phd cluster.
# Not meant to be executed directly -- copy the block you need.
#
# Env-var flow: phd run's leading VAR=val assignments -> run_trex_job.sh
# (no filtering, passes everything through -- see its own comment at the
# TRAIN_SCRIPT invocation) -> run_all_abs_ori.sh (reads MEMORY_* with
# ${VAR:-default}, appends --memory_* flags to EXTRA_ARGS) -> train.py.
# Verified this session: --memory_slow_dropout/--memory_fast_dropout were
# missing from run_all_abs_ori.sh's EXTRA_ARGS entirely (a real "set the
# env var, nothing happens" bug) -- now wired the same way every other
# memory flag already is.

# ── Full production run, memory + dropout enabled ───────────────────────────
EXPERIMENT_NAME=all_abs_ori_memory_dropout_run1 \
TRAIN_SCRIPT=scripts/run_all_abs_ori.sh \
DATA_ROOT_SRC=${HOME}/other/new_data/competition \
MASK_FROZEN_LOSS=1 MAX_EPISODES=0 \
TRAIN_BSZ=8 GRAD_ACCUM=4 LR=1.5e-4 N_EPOCHS=3 \
MEMORY_SLOW_SECONDS=0.25,0.5,1.0,5.0 MEMORY_FAST=4 \
MEMORY_SLOW_DROPOUT=0.2 MEMORY_FAST_DROPOUT=0.2 \
  phd run -ng 2 -p shr_gpu -GR H100 -l %J.log sh run_trex_job.sh 2

# ── Same run, resuming from a memory-less checkpoint (recommended -- see
#    this session's discussion: memory adds zero new parameters, so a
#    memory-less checkpoint's weights are directly reusable; warm-starting
#    from one avoids relearning general task competence from scratch) ──────
EXPERIMENT_NAME=all_abs_ori_memory_dropout_run1 \
TRAIN_SCRIPT=scripts/run_all_abs_ori.sh \
DATA_ROOT_SRC=${HOME}/other/new_data/competition \
RESUME_CHECKPOINT=<path to a memory-less checkpoint model.pt> \
MASK_FROZEN_LOSS=1 MAX_EPISODES=0 \
TRAIN_BSZ=8 GRAD_ACCUM=4 LR=1.5e-4 N_EPOCHS=3 \
MEMORY_SLOW_SECONDS=0.25,0.5,1.0,5.0 MEMORY_FAST=4 \
MEMORY_SLOW_DROPOUT=0.2 MEMORY_FAST_DROPOUT=0.2 \
  phd run -ng 2 -p shr_gpu -GR H100 -l %J.log sh run_trex_job.sh 2

# ── Small-GPU smoke test (verified working this session on a single rented
#    24GB card -- adamw8bit + gradient_checkpointing are required to fit;
#    plain adamw OOMs even at train_bsz_per_gpu=2 on 24GB) ──────────────────
NUM_GPUS=1 \
PROJECT_ROOT=/workspace/T-Rex DATA_ROOT=/workspace/data OUTPUT_DIR=/workspace/outputs \
ORIGIN_MODEL_PATH=Qwen/Qwen3-VL-2B-Instruct \
MEMORY_SLOW_SECONDS=0.25,0.5,1.0,5.0 MEMORY_FAST=4 \
MEMORY_SLOW_DROPOUT=0.2 MEMORY_FAST_DROPOUT=0.2 \
SMOKE=1 \
bash scripts/run_all_abs_ori.sh

# ── Direct train.py invocation (no phd, no wrapper scripts -- what this
#    session actually ran end-to-end, real 15-step run, real data, real
#    Qwen3-VL-2B backbone; confirmed no crash, finite loss, checkpoint
#    saved) ────────────────────────────────────────────────────────────────
accelerate launch --num_processes 1 --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
    train.py \
    --model_path Qwen/Qwen3-VL-2B-Instruct \
    --data_format origami --origami_root <root>/train --origami_val_root <root>/val \
    --origami_sampler block --origami_pool_groups 32 --origami_cache_groups 8 --max_episodes 2 \
    --output_dir <out> --log_dir <out> --experiment_name smoke --run_name r1 \
    --n_epochs 1 --save_freq 100000 --max_ckpts 1 \
    --action_dim 65 --action_chunk 16 --image_size 224 224 \
    --train_bsz_per_gpu 2 --gradient_accumulation_steps 1 \
    --learning_rate 1.5e-4 --min_lr_ratio 0.1 --warmup_rates 0.05 \
    --weight_decay 0 --max_grad_norm 1.0 --optim adamw8bit --gradient_checkpointing 1 \
    --freeze_latent_expert 1 --train_latent_last_n 4 --num_workers 4 \
    --use_robot_state 0 --use_tactile_vec 1 --use_tactile_deform 1 \
    --state_noise_mode joint --mask_frozen_loss 1 --tactile_intermediate_size 1536 \
    --training_stage 2 --cascaded_total_steps 10 --cascaded_split_step 6 \
    --cascaded_tactile_dropout 0.1 --cascaded_loss_weight 1.0 \
    --use_flare 0 --seed 42 \
    --max_steps 15 --save_steps 0 --val_freq 0 \
    --memory_slow_seconds 0.25,0.5,1.0,5.0 --memory_fast 2 \
    --memory_rope_stride_slow 32.0 --memory_rope_stride_fast 8.0 \
    --memory_slow_dropout 0.2 --memory_fast_dropout 0.2

# ── Serving (once a memory+dropout-trained checkpoint exists) ───────────────
# Dropout is train-only (OrigamiDataset-side, never reaches serving), so
# nothing new is needed here beyond the existing memory flags:
python scripts/test.py --checkpoint_path <ckpt> \
    --memory_slow_seconds 0.25,0.5,1.0,5.0 --memory_fast 4 \
    --memory_rope_stride_slow 32.0 --memory_rope_stride_fast 8.0
