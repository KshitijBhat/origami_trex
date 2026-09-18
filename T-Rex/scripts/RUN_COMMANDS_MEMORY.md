# dev/memory run commands

Memory flags (`MEMORY_SLOW_SECONDS`, `MEMORY_FAST`, `MEMORY_SLOW_JITTER_SEC`,
`MEMORY_ROPE_STRIDE_SLOW`, `MEMORY_ROPE_STRIDE_FAST`) default off everywhere
below -- omit them for byte-identical current behavior. See
`qwen_vla/MEMORY_DESIGN.md` for design detail and GPU verification results.

## Production launch (`phd` cluster)

Same pattern as `run_trex_job.sh`'s own header example, `TRAIN_SCRIPT` still
`scripts/run_all_abs_ori.sh`, just with the two memory env vars added:

```bash
EXPERIMENT_NAME=all_abs_ori_memory_run1 \
TRAIN_SCRIPT=scripts/run_all_abs_ori.sh \
DATA_ROOT_SRC=${HOME}/other/new_data/competition \
MASK_FROZEN_LOSS=1 MAX_EPISODES=0 \
TRAIN_BSZ=8 GRAD_ACCUM=4 LR=1.5e-4 N_EPOCHS=3 \
MEMORY_SLOW_SECONDS=0.25,0.5,1.0,5.0 MEMORY_FAST=4 \
  phd run -ng 2 -p shr_gpu -GR H100 -l %J.log sh run_trex_job.sh 2
```

Goes through `run_trex_job.sh` -> `run_all_abs_ori.sh` -> `train.py`,
auto-discovering the real midtrain checkpoint under `${ASSET_ROOT}/trex_midtrain`
and using the full dataset. Everything else stays at `run_all_abs_ori.sh`'s
hardcoded production defaults (`use_tactile_vqvae=1`, `cascaded_*`, etc.).

## Direct launcher (no cluster, e.g. a rented box)

```bash
NUM_GPUS=1 \
PROJECT_ROOT=/workspace/T-Rex DATA_ROOT=/workspace/data OUTPUT_DIR=/workspace/outputs \
ORIGIN_MODEL_PATH=Qwen/Qwen3-VL-2B-Instruct \
RESUME_CHECKPOINT=<path with model.pt> \
MEMORY_SLOW_SECONDS=0.25,0.5,1.0,5.0 MEMORY_FAST=4 \
bash scripts/run_all_abs_ori.sh
```

`SMOKE=1` for a 5-step no-checkpoint sanity pass first.

## Bare `train.py` smoke test (no resume checkpoint needed)

What this session actually used on the sandbox GPU box (no real midtrain
checkpoint available there) -- fresh-inits from the base model instead of
resuming, useful for a quick "does it crash / does loss move" check:

```bash
accelerate launch --num_processes 1 --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
    train.py \
    --model_path Qwen/Qwen3-VL-2B-Instruct \
    --data_format origami --origami_root <root>/train --origami_val_root <root>/val \
    --origami_sampler block --origami_pool_groups 32 --origami_cache_groups 8 --max_episodes 2 \
    --output_dir <out> --log_dir <out> --experiment_name smoke --run_name r1 \
    --n_epochs 1 --save_freq 100000 --max_ckpts 1 \
    --action_dim 65 --action_chunk 16 --image_size 224 224 \
    --train_bsz_per_gpu 4 --gradient_accumulation_steps 1 \
    --learning_rate 1.5e-4 --min_lr_ratio 0.1 --warmup_rates 0.05 \
    --weight_decay 0 --max_grad_norm 1.0 --optim adamw --gradient_checkpointing 1 \
    --freeze_latent_expert 1 --train_latent_last_n 4 --num_workers 4 \
    --use_robot_state 0 --use_tactile_vec 1 --use_tactile_deform 1 \
    --use_tactile_vqvae 1 --vqvae_ckpt <vqvae ckpt> \
    --state_noise_mode joint --mask_frozen_loss 1 --tactile_intermediate_size 1536 \
    --training_stage 2 --cascaded_total_steps 10 --cascaded_split_step 6 \
    --cascaded_tactile_dropout 0.1 --cascaded_loss_weight 1.0 \
    --use_flare 1 --n_flare_tokens_per_frame 4 --n_flare_steps 8 \
    --flare_loss_weight 0.0 --flare_frame_stride 4 --flare_layer_index -1 --seed 42 \
    --max_steps 150 --save_steps 0 --val_freq 0 \
    --memory_slow_seconds '0.25,0.5,1.0,5.0' --memory_fast 2 \
    --memory_rope_stride_slow 32.0 --memory_rope_stride_fast 8.0
```

Real 150-step run with this command (2 real episodes, `drakedrake/origami_
preprocessed_stride_1`, `Qwen/Qwen3-VL-2B-Instruct`) showed loss falling
2.14 -> 1.23 total, 1.09 -> 0.23 on the action-expert term. Full log:
`qwen_vla/dev_memory_run_logs/train_loss_trend.log`; the other logs in that
directory are the earlier crash-freedom/regression smoke tests from Parts
B-D (see `MEMORY_DESIGN.md`'s Progress sections for which is which).

## Serving (once a memory-trained checkpoint exists)

```bash
python scripts/test.py --checkpoint_path <ckpt> \
    --memory_slow_seconds 0.25,0.5,1.0,5.0 --memory_fast 4 \
    --memory_rope_stride_slow 32.0 --memory_rope_stride_fast 8.0
```

Must match what the checkpoint actually trained with (`training_args.json`
records it, but `test.py` does not yet auto-read it back -- pass explicitly).
