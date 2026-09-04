#!/bin/bash
# Reference commands for phd run job submissions -- T-Rex post-training on the
# origami-flat data. Not meant to be run as one script (each block is a
# separate job submission) -- copy/paste the block you want. Analogous to
# vitacformer++'s cmd++.sh.
#
# Before any of these:
#   1. bash scripts/setup_server_env.sh   (BACKBONE_ZIP=... -- see
#      notebooks/download_trex_backbones.ipynb; this server has no HF hub
#      access, so the base model + midtrain checkpoint come from that zip,
#      not a download)
#   2. Make sure DATA_ROOT_SRC below actually points at a prepared
#      origami-flat dataset (trex_origami/run_prepare.sh pilot|dense|full,
#      run elsewhere and rsync'd here -- prep also needs hub access)
#
# GPU: this node type maxes out at 4 GPUs, for EITHER H100 or A100 (-GR
# selects which). All blocks below default to H100; swap -GR A100 to use the
# other pool. Nothing else changes -- run_trex_job.sh/train_origami.sh don't
# care which GPU type they land on, only how many (NUM_GPUS, the trailing
# arg to run_trex_job.sh, capped at 4 here).

# 1. Pilot, single GPU (matches the Colab-validated config exactly -- same
#    TRAIN_BSZ=8/GRAD_ACCUM=4/LR=5e-5 the checkpoint-1-12000 eval used)
EXPERIMENT_NAME=aug28_pilot_1gpu \
DATA_ROOT_SRC=$HOME/other/new_data/origami_flat/pilot \
phd run -ng 1 -p shr_gpu -GR H100 -l %J.log sh run_trex_job.sh 1

# 1b. Same, on an A100 node instead (only -GR changes)
EXPERIMENT_NAME=aug28_pilot_1gpu_a100 \
DATA_ROOT_SRC=$HOME/other/new_data/origami_flat/pilot \
phd run -ng 1 -p shr_gpu -GR A100 -l %J.log sh run_trex_job.sh 1

# 2. Pilot, 4 GPU (the max this node type gives you) -- NOTE: LR is NOT
#    auto-scaled (train_origami.sh says so on startup). The paper's own
#    reference is effective batch 128 (16/gpu x 8 gpu) at LR 1e-4;
#    TRAIN_BSZ x GRAD_ACCUM x NUM_GPUS here would be 8x4x4=128 at the
#    still-single-GPU-tuned LR=5e-5 -- consider raising LR toward the
#    paper's 1e-4 if you use this, and watch the val act/tac loss curve for
#    the first ~500 steps before trusting it.
EXPERIMENT_NAME=aug28_pilot_4gpu \
DATA_ROOT_SRC=$HOME/other/new_data/origami_flat/pilot \
phd run -ng 4 -p shr_gpu -GR H100 -l %J.log sh run_trex_job.sh 4

# 3. Full dataset (101 train / 25 val seasons), 4 GPU
EXPERIMENT_NAME=aug28_full_4gpu \
DATA_ROOT_SRC=$HOME/other/new_data/origami_flat/full \
N_EPOCHS=2 \
phd run -ng 4 -p shr_gpu -GR H100 -l %J.log sh run_trex_job.sh 4

# 4. Full dataset, resume after a preemption/crash (same RESUME=1 convention
#    train_origami.sh already implements for Colab)
EXPERIMENT_NAME=aug28_full_4gpu \
DATA_ROOT_SRC=$HOME/other/new_data/origami_flat/full \
RESUME=1 \
phd run -ng 4 -p shr_gpu -GR H100 -l %J.log sh run_trex_job.sh 4

# 5. Smoke test only (5 steps, proves the wiring on THIS machine before
#    burning a real allocation -- run this first, always). Also exercises
#    the pre-training validation + attention-capture pass added to train.py,
#    since that runs at global_step==0 regardless of --max_steps.
EXPERIMENT_NAME=aug28_smoke \
DATA_ROOT_SRC=$HOME/other/new_data/origami_flat/pilot \
SMOKE=1 USE_SCRATCH=0 \
phd run -ng 1 -p shr_gpu -GR H100 -l %J.log sh run_trex_job.sh 1

# 5b. Same idea, but against a tiny 2-train/1-val-season dataset (a handful
#    of episodes total) instead of the real pilot tier -- for a from-scratch
#    box where the real dataset isn't staged yet, or to sanity-check the data
#    pipeline itself in isolation from pilot-tier size/timing. Deliberately a
#    DIFFERENT directory name (origami_flat_dummy, not origami_flat) so it
#    can never collide with or be mistaken for a real tier already on this
#    box. Unzip the dummy dataset here first:
#      mkdir -p ~/other/new_data/origami_flat_dummy
#      unzip -q origami_flat_dummy.zip -d ~/other/new_data/origami_flat_dummy
#      # -> ~/other/new_data/origami_flat_dummy/origami_flat/full/{train,val}
EXPERIMENT_NAME=smoke_dummy \
DATA_ROOT_SRC=$HOME/other/new_data/origami_flat_dummy/origami_flat/full \
SMOKE=1 USE_SCRATCH=0 \
phd run -ng 1 -p shr_gpu -GR H100 -l %J.log sh run_trex_job.sh 1

# 6. Same as #3, but capture attention maps on every validation call instead
#    of every 5th (CAPTURE_ATTN_EVERY_N_VAL is threaded through
#    train_origami.sh -> train.py's --capture_attn_every_n_val)
EXPERIMENT_NAME=aug28_full_4gpu_denseviz \
DATA_ROOT_SRC=$HOME/other/new_data/origami_flat/full \
CAPTURE_ATTN_EVERY_N_VAL=1 \
phd run -ng 4 -p shr_gpu -GR H100 -l %J.log sh run_trex_job.sh 4

# 7. Try the README's #1 lever for the underperforming checkpoint-1-12000
#    result: re-prep with fold-phase conditioning in the task prompt (this
#    changes trex_origami's prep step, not the training job -- run on a CPU
#    box per trex_origami/README.md, then point DATA_ROOT_SRC at the result):
#    bash trex_origami/run_prepare.sh pilot --phase_mode progress

# 8. Ablation: no robot-state input (USE_ROBOT_STATE=0 -> --use_robot_state 0).
#    Data is unchanged -- this only removes the state vector from the model's
#    input, it does not need a different DATA_ROOT_SRC. Smoke it first.
EXPERIMENT_NAME=sep_ablation_no_state \
DATA_ROOT_SRC=$HOME/other/new_data/origami_flat/pilot \
USE_ROBOT_STATE=0 SMOKE=1 USE_SCRATCH=0 \
phd run -ng 1 -p shr_gpu -GR H100 -l %J.log sh run_trex_job.sh 1

# 9. Causal-confusion mitigation knobs for hybrid anchoring (arms anchored to
#    prev_command): ANCHOR_NOISE_MODE=tracking perturbs the anchor with
#    measured tracking-error stats, ANCHOR_DROPOUT re-anchors that fraction of
#    samples to state[t] instead of prev_command[t]. No-ops on data prepared
#    with --anchor-mode state (no dims are prev_command-anchored there).
EXPERIMENT_NAME=sep_anchor_noise \
DATA_ROOT_SRC=$HOME/other/new_data/origami_flat/full \
ANCHOR_NOISE_MODE=tracking ANCHOR_DROPOUT=0.1 \
phd run -ng 4 -p shr_gpu -GR H100 -l %J.log sh run_trex_job.sh 4
