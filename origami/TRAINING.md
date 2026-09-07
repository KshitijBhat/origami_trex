# Training on a rented A100 / H100

This is a practical runbook for REDESIGN_PLAN.md §12 steps 11 (pilot) and 12 (the real run),
written after step 10 finished for real: both `data_trex_origami/eef62_train/` (101 seasons,
8 379 271 frames) and `data_trex_origami/eef62_val/` (25 seasons, 2 120 726 frames) exist,
verified (`verify.py all` -- G18/G8/G8b/G6/G4 pass on both). See `PROGRESS.md` for how that
data was produced and what's been verified so far; this file is only about the training run
itself. Every command below was checked against the real checkpoint/data on a 16 GB dev GPU up
to the point that GPU's memory runs out (see "Why not just copy `train_origami.sh` verbatim"
below) -- nothing here is guessed.

## 0. What you need on the new box

* An A100 (40 GB or 80 GB) or H100 (80 GB), single GPU. Everything below assumes one GPU;
  see "Multi-GPU" at the end if you have more.
* This repo, at the commit that has this file.
* The two prepared data roots: `data_trex_origami/eef62_train/` (~115 GB) and
  `data_trex_origami/eef62_val/` (~29 GB). Copy them over (`rsync -av --info=progress2`,
  or your cloud provider's object storage) rather than re-running `prepare.py` -- that needs
  a `HF_TOKEN` for the gated source dataset and re-downloads/re-encodes ~250+ GB for no reason
  if you already have the output.
* ~150 GB free disk for the data roots above, plus ~20 GB for the two downloaded
  checkpoints below, plus headroom for training-run checkpoints (each saved checkpoint is
  roughly one model-sized `model.pt`, ~8.5 GB bf16, times however many `--max_ckpts` you keep).

## 1. Environment

```bash
cd origami_trex
uv sync --extra dev          # same lockfile as this dev box; pyproject.toml pins pin/pink/
                              # qpsolvers[daqp]/av/lerobot/etc -- do not hand-roll this list
source .venv/bin/activate    # or prefix commands with `uv run`
```

**`deepspeed` and `bitsandbytes` are not installed and not in `pyproject.toml` -- this is
deliberate, not an oversight (see PROGRESS.md's "Step 10 confirmed complete..." note).**
`T-Rex/config/sft_qwen.yaml` (upstream's own accelerate config, referenced by
`origami/train_origami.sh`) is built for **8 GPUs with DeepSpeed ZeRO-2**; a single rented
A100/H100 gets no partitioning benefit from ZeRO at world_size 1, so pulling in DeepSpeed just
adds a heavy, sometimes-finicky-to-build dependency for nothing. Use the plain config added
this session instead:

```
origami/config/single_gpu_bf16.yaml    # distributed_type: 'NO', mixed_precision: bf16, num_processes: 1
```

This has been run for real (`accelerate launch --config_file origami/config/single_gpu_bf16.yaml
--num_processes 1 -m origami.train_origami ...`) against the real 101-season root and the real
checkpoints below -- it launches cleanly and reaches model construction + dataset open + the
first `accelerator.prepare()` call. (On this session's 16 GB dev GPU that call then hits
`CUDA out of memory`, which is exactly why this file exists -- see §4.)

Only install `bitsandbytes` if you're on a 40 GB card and want `--optim adamw8bit` (§4):
```bash
uv pip install bitsandbytes
```
Do **not** install `deepspeed` unless you're deliberately setting up multi-GPU (§7).

**Weights & Biases.** `train_origami.py` always calls `wandb.init(...)`. Either:
```bash
wandb login                       # if you want real experiment tracking
# or, to skip it entirely:
export WANDB_MODE=disabled
```

## 2. Get the two checkpoints

Both are public on the Hub -- no `HF_TOKEN` needed for these (unlike the gated source dataset
`prepare.py` reads from).

```bash
python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('Qwen/Qwen3-VL-2B-Instruct', local_dir='checkpoints/qwen3vl_base'))
print(snapshot_download('miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6', local_dir='checkpoints/midtrain'))
"
```
`checkpoints/qwen3vl_base` (~4 GB, safetensors) is `--model_path` below -- the frozen-ViT base
model whose vision tower and tokenizer/processor `Qwen3VLVLAModel.from_pretrained_qwen3vl` reads.
`checkpoints/midtrain` (~8.5 GB, `model.pt` + `training_args.json` + `stats_data.json` +
`processor/`) is `--resume_checkpoint` -- T-Rex's own midtrain checkpoint, the warm start §7.5
argues for. **Don't swap these two** -- `--model_path` must be a plain HF model directory with
`config.json` at its root (`checkpoints/qwen3vl_base` is, straight from `snapshot_download`;
`checkpoints/midtrain` is NOT -- its `config.json` sits next to a `model.pt` state-dict blob,
not safetensors, and `AutoProcessor.from_pretrained` on it fails looking for a
`preprocessor_config.json` that's one level down in `processor/`). This split is intentional in
`train_origami.py`: `--model_path` supplies the frozen base weights + processor,
`--resume_checkpoint` supplies the midtrain-trained MoT/action/tactile-expert weights that get
loaded over it (`Resumed: missing=4, unexpected=0` in the log is expected -- see PROGRESS.md's
note on the 4 unexplained missing keys, not blocking).

`--deform_encoder_ckpt` (`sharpa_wave_deform_encoder.pth` in the plan) is not needed here: it's
only read if you're training the deform encoder from scratch, and `checkpoints/midtrain/model.pt`
already carries trained deform-encoder weights that `--resume_checkpoint` restores. Leave the
flag unset (`load_deform_encoder_weights` prints a harmless `Warning: DeformEncoder checkpoint
not found at` and no-ops, then the resume step overwrites those weights anyway -- confirmed in
this session's logs).

## 3. Before spending real GPU-hours: check the tactile VQ-VAE (recommended, not required)

`origami/PROGRESS.md`'s G15 gate **FAILED** against this exact midtrain checkpoint: its
`tacf6_vqvae_{min,max,mask}` buffers were fit on T-Rex's own force ranges, and origami's forces
mostly fall in a narrow sliver of that range -- 6 of the tactile expert's 10 finger channels
collapse to 1-2 codes (near-zero information), and the zero-shot eval shows the cascaded/
tactile-enabled path performing **identically** to `--disable_tactile` and `--tactile_zeroed`,
i.e. the tactile expert currently contributes nothing. Training will still run and the loss will
still go down (the action-expert path is unaffected), but you'll be paying for a tactile expert
that isn't learning anything useful until this is addressed. Two options, in order of effort:

1. **Cheapest: re-fit just the buffers.** `origami/refit_vqvae_stats.py` (§11.8 fix #1,
   REDESIGN_PLAN.md §12 step 10c) is **not yet built** -- this is genuinely the next thing to
   implement per the plan, not a training-time flag. It would overwrite
   `tacf6_vqvae_{min,max,mask}` in a copy of the midtrain checkpoint with origami's own
   q01/q99 (already computed, in `data_trex_origami/eef62_train/meta/trex_norm_stats.json`) and
   the degenerate-channel mask, leaving the encoder weights/codebook untouched -- cheap, no
   retraining of the VQ-VAE itself.
2. **More work: retrain the VQ-VAE on origami F6 data** (§11.8 fix #2) -- not started, bigger
   lift.
3. **Or just proceed without it**: `--use_tactile_vqvae 0` (keep `--use_tactile_vec 1
   --use_tactile_deform 1`) trains without the temporal tactile-code path at all, avoiding the
   dead weight rather than fixing it. Cheapest of all, but gives up the delay-curriculum
   ingredient §7.5-B calls "the ingredient worth importing" from midtrain.

None of this blocks running a pilot -- it's a quality/priority note, not a gate to implement
before touching the GPU. If you want to fix it first, come back to this file after that's done;
the training commands below are unaffected either way except for the `--use_tactile_vqvae` flag.

## 4. Memory tiers and the actual command

**Why not just copy `origami/train_origami.sh` verbatim.** That script already targets a
*single* GPU (`--num_processes 1`, `train_bsz_per_gpu 2 × grad_accum 64` for effective batch
128 -- §7.3's own "2 × 1" bsz×GPUs row) -- the GPU count isn't the issue. The **actual and only**
reason you can't run it as written is that it launches via `--config_file
T-Rex/config/sft_qwen.yaml`, and that config's `distributed_type: DEEPSPEED` gets used
regardless of `--num_processes`, needing the `deepspeed` package this project deliberately
doesn't install (§1) -- swap in `origami/config/single_gpu_bf16.yaml` (§4's command below) and
everything else about that script's flags carries over unchanged, including its
`train_bsz_per_gpu`/`grad_accum` split, which is one valid choice from the tier table below.
The model itself (`Qwen3VLVLAModel`, MoT with separate action/tactile expert weight copies) is
**4255.7M total params, 3844.7M trainable**, confirmed by loading it for real this session.
At fp32 master weights (standard mixed-precision training keeps these in fp32 and only
autocasts forward/backward to bf16 -- confirmed by reading T-Rex's own upstream `midtrain.py`,
which does exactly this, not full-bf16 weights) that's **~17 GB for parameters alone**, before
optimizer state or activations -- confirmed empirically: this is exactly what OOM'd on this
session's 16 GB card, at the very first `accelerator.prepare()` call, before a single training
step. Plan for that floor on whatever card you use.

Rough budget per GPU tier (parameters ~17 GB fixed; optimizer state and activations are what
scale with these choices -- **these are estimates, not measurements on real A100/H100
hardware**; validate with a short run before committing to the full step count):

| tier | `--optim` | `--train_bsz_per_gpu` | `--gradient_accumulation_steps` | `--gradient_checkpointing` | notes |
|---|---|---|---|---|---|
| **A100 40 GB** | `adamw8bit` | 2 | 64 | 1 | `uv pip install bitsandbytes` first (§1). Plain `adamw`'s fp32 optimizer state (~34 GB for 2 moments × 4255.7M params × 4 B) will not fit alongside the ~17 GB of weights on 40 GB. |
| **A100/H100 80 GB** | `adamw` (simplest) or `adamw8bit` (more headroom) | 4-8 | 32-16 | 1 (start here; try `0` only after a pilot succeeds, for speed) | 17 GB weights + ~34 GB plain-AdamW state + gradients + activations is tight on 80 GB with a bigger batch -- start conservative, watch `nvidia-smi`, scale up. |

Whatever `--train_bsz_per_gpu × --gradient_accumulation_steps` you pick, **keep the product at
128** to match the recipe's effective batch size (and hence its `--learning_rate 1e-4` /
`--warmup_rates 0.03` tuning).

```bash
export WANDB_MODE=disabled   # or `wandb login` first -- see §1
export HF_HUB_OFFLINE=1      # everything needed is already local (§2); skip Hub lookups
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduces OOM from fragmentation

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
  --train_bsz_per_gpu <TIER_BSZ> --gradient_accumulation_steps <TIER_ACCUM> \
  --gradient_checkpointing 1 --optim <TIER_OPTIM> --num_workers 8 \
  --n_epochs 1 --max_steps <STEPS> --save_steps <SAVE_EVERY> --save_optimizer_state 1 \
  --val_freq 500 --max_val_batches 30 --seed 42 \
  --output_dir ./outputs/origami_run1 --log_dir ./logs
```
Fill in `<TIER_BSZ>` / `<TIER_ACCUM>` / `<TIER_OPTIM>` from the table above, and `<STEPS>` /
`<SAVE_EVERY>` from §5 below. `--lerobot_val_root` uses the real held-out-season val root
(§7.2 item 5) instead of a random within-root episode split -- always pass it now that it
exists.

## 5. Run order: smoke test → pilot → full

Don't jump straight to 40 000 steps. In order:

1. **Smoke test (a few minutes, not a real gate result -- just "does it run on this box").**
   Same command as §4 with `--max_steps 5 --save_steps 0 --save_optimizer_state 0`. Watch
   `nvidia-smi` during this run; if you OOM, drop `--train_bsz_per_gpu` (down to 1) before
   raising `--gradient_accumulation_steps` to compensate, or move up a memory tier.
2. **Pilot -- REDESIGN_PLAN.md §12 step 11, 2000 steps.** `--max_steps 2000 --save_steps 500`.
   This is where §11.1's open question gets answered: run `eval_offline.py`'s §8.1-A
   (normalized variance share per action block) on the resulting checkpoint against
   `data_trex_origami/eef62_val` and compare to `eval_offline.py --zero-shot`'s already-recorded
   floor (`reports/eval_offline_report.md` -- cascaded does NOT beat hold-position on hand
   blocks, zero-shot). The pilot should visibly close that gap on at least the EEF-pose blocks
   before you commit to the full run.
3. **Full run -- step 12, 40 000 steps.** `--max_steps 40000 --save_steps 2000`. At 128
   effective batch this is ≈5.1M samples, about 2× T-Rex's own entire midtrain corpus (§7.3's
   own calibration table) -- expect this to take a while; budget wall-clock from your pilot's
   measured steps/sec, not from step count alone.

## 6. Resuming an interrupted run

Add `--resume_checkpoint ./outputs/origami_run1/checkpoint-<epoch>-<step> --resume_source
midtrain --resume_full_state 1` (yes, `--resume_checkpoint` gets repointed at your own
in-progress run's checkpoint dir, not the original midtrain one, once you have one). This
session fixed two real bugs in this exact path -- both verified against real `accelerate`
machinery in `origami/tests/test_checkpoint_resume.py` (CPU-only, so you can re-run that test
locally without a GPU to convince yourself before trusting a real resume):
* the LR scheduler now actually resumes at the same learning rate and continues the same
  cosine curve, instead of silently restarting warmup from zero;
* `global_step` now actually resumes from where you left off, instead of resetting to 0 (which
  used to throw off `--save_steps` cadence and step-indexed logging on every resume).

Requires the checkpoint you're resuming from to have been saved with `--save_optimizer_state 1`
(on by default in §4's command) -- that's what writes the `state/` subdirectory
`--resume_full_state` reads.

## 7. Multi-GPU (if you rent more than one)

Everything above assumes 1 GPU. With N GPUs, either:
* Simplest: `accelerate launch --config_file origami/config/single_gpu_bf16.yaml
  --num_processes N ...` (bump `num_processes`; `distributed_type: 'NO'` still works for
  plain data-parallel via `torch.distributed`, accelerate handles the launch). Divide
  `--gradient_accumulation_steps` by N to keep effective batch 128 (`EpisodeGroupedSampler`
  already shards by rank correctly -- fixed and tested this session).
* Or install `deepspeed` and use (a copy of) `T-Rex/config/sft_qwen.yaml` with
  `num_processes` set to your GPU count -- ZeRO-2 now has something to shard across. Not
  tested this session (no multi-GPU hardware available); the plain path above is the
  lower-risk default.

## 8. What's NOT covered here

Deploy (retarget/serve_zenoh/policy), the docker submission image, and `eval_offline.py`'s
full metric suite (§8.1-B..F) are separate, later steps (REDESIGN_PLAN.md §12 steps 13-14-15)
-- out of scope for "get a training run going." See `PROGRESS.md`'s build-order table for
what's left overall.
