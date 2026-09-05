#!/bin/bash
# REDESIGN_PLAN.md §7.3 -- verbatim recipe (deviations from T-Rex's scripts/train.sh are
# tabulated there, along with the step-budget calibration). Fill in the environment
# variables below before running.
set -euo pipefail

IMAGE_SIZE="384 384"          # from the §4.5-A probe

accelerate launch --config_file T-Rex/config/sft_qwen.yaml --num_processes 1 \
  origami/train_origami.py \
  --model_path ${ORIGIN_MODEL_PATH} \
  --data_format lerobot --lerobot_root ${LEROBOT_ROOT} \
  --lerobot_val_root ${LEROBOT_VAL_ROOT} \
  --action_dim 62 --action_chunk 16 --image_size ${IMAGE_SIZE} \
  --use_robot_state 1 \
  --use_tactile_vec 1 --use_tactile_deform 1 --use_tactile_vqvae 1 \
  --deform_encoder_ckpt ${DEFORM_ENCODER_PATH} \
  --tactile_intermediate_size 1536 --training_stage 2 --tactile_loss_weight 1.0 \
  --cascaded_total_steps 10 --cascaded_split_step 6 \
  --cascaded_tactile_dropout 0.1 --cascaded_loss_weight 1.0 \
  --tactile_delay_offsets 0 4 8 12 --tactile_delay_scope f6 \
  --use_flare 1 --n_flare_tokens_per_frame 4 --n_flare_steps 8 \
  --flare_loss_weight 0.5 --flare_frame_stride 4 --flare_layer_index -1 \
  --resume_checkpoint ${RESUME_CHECKPOINT} --resume_source midtrain \
  --learning_rate 1e-4 --min_lr_ratio 0 --weight_decay 0.01 --max_grad_norm 1.0 \
  --warmup_rates 0.03 \
  --train_bsz_per_gpu 2 --gradient_accumulation_steps 64 \
  --gradient_checkpointing 1 --optim adamw8bit --num_workers 8 \
  --n_epochs 1 --max_steps 40000 --save_steps 2000 --save_optimizer_state 1 \
  --val_freq 1000 --max_val_batches 30 --seed 42
