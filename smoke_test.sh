#!/bin/bash
# TRAINING.md §5 step 1 -- smoke test on A100 40GB (tier: adamw8bit, bsz2 x accum64, grad ckpt on)
set -euo pipefail
cd /workspace/origami_trex
source .venv/bin/activate

export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch --config_file origami/config/single_gpu_bf16.yaml --num_processes 1 \
  -m origami.train_origami \
  --model_path checkpoints/qwen3vl_base \
  --resume_checkpoint checkpoints/midtrain --resume_source midtrain \
  --data_format lerobot \
  --lerobot_root data_trex_origami/eef62_train \
  --lerobot_val_root data_trex_origami/eef62_val \
  --action_dim 62 --action_chunk 16 --image_size 384 384 \
  --use_robot_state 1 \
  --use_tactile_vec 1 --use_tactile_deform 1 --use_tactile_vqvae 1 \
  --tactile_intermediate_size 1536 --training_stage 2 --tactile_loss_weight 1.0 \
  --cascaded_total_steps 10 --cascaded_split_step 6 \
  --cascaded_tactile_dropout 0.1 --cascaded_loss_weight 1.0 \
  --tactile_delay_offsets 0 4 8 12 --tactile_delay_scope f6 \
  --use_flare 1 --n_flare_tokens_per_frame 4 --n_flare_steps 8 \
  --flare_loss_weight 0.5 --flare_frame_stride 4 --flare_layer_index -1 \
  --learning_rate 1e-4 --min_lr_ratio 0 --weight_decay 0.01 --max_grad_norm 1.0 \
  --warmup_rates 0.03 \
  --train_bsz_per_gpu 1 --gradient_accumulation_steps 128 \
  --gradient_checkpointing 1 --optim paged_adamw8bit --num_workers 8 \
  --n_epochs 1 --max_steps 5 --save_steps 0 --save_optimizer_state 0 \
  --val_freq 500 --max_val_batches 30 --seed 42 \
  --output_dir ./outputs/smoke_test --log_dir ./logs
