# Training pipeline check (V100)

`run_all_abs_ori.sh` is the self-contained launcher for the all-absolute
branch — combines what `run_trex_job.sh` + `train_origami.sh` used to split
between them, no PhD-cluster harness needed.

```
NUM_GPUS=1 SMOKE=1 bash scripts/run_all_abs_ori.sh   # 5 steps, proves the wiring
NUM_GPUS=1 bash scripts/run_all_abs_ori.sh            # real run
NUM_GPUS=2 bash scripts/run_all_abs_ori.sh            # multi-GPU, one node
```

Fixes vs. the old `train_origami.sh`: `ACTION_CHUNK` 25 → 16, `use_robot_state`
1 → 0 (never on during midtrain), anchor flags dropped (unused post
all-absolute), `--mask_frozen_loss`/`--max_episodes` wired in.

## Verified: 2x Tesla V100-SXM2-32GB

Ran `NUM_GPUS=1 SMOKE=1` against the dummy dataset
(`drakedrake/ori-trex-competition`) + midtrain checkpoint resume. Result:
**all 5 steps completed, checkpoint saved, no errors.**

| | |
|---|---|
| peak memory | ~14 GiB / 32 GiB |
| per-step time | ~160–210s (Volta has no BF16 tensor cores — perf-only, not a bug) |
| loss at step 5 | act≈1.39, tac≈1.42, total≈2.81 — no NaNs |
| resume | `missing=5, unexpected=0` (expected: the 62→65 dim / 25→16 chunk head resize) |

An earlier 16 GB card (RTX 4080S) OOM'd inside `bitsandbytes.AdamW8bit`'s
one-time optimizer-state allocation, which only fires *after* backward
completes — static floor is ≈17.9 GiB (bf16 weights + trainable grads + 8-bit
optimizer state) regardless of freeze schedule. 32 GiB clears it with room to
spare.

### Verified: 2x V100-SXM2-32GB, `NUM_GPUS=2`

Same dataset/checkpoint, `accelerate launch --num_processes 2` (effective
batch 8×4×2=64). **All 5 steps completed, checkpoint saved, no errors** —
both GPUs actively used (~22.7 GiB each), no NCCL/distributed init issues.

| | |
|---|---|
| peak memory (per GPU) | ~22.7 GiB / 32 GiB |
| per-step time | ~143–183s (vs. ~160–210s single-GPU, at 2x the effective batch) |
| loss at step 5 | act≈1.38, tac≈1.41, total≈2.79 — matches the single-GPU run |

Per-step wall time is roughly flat going 1→2 GPUs, but each step now covers
2x the samples (data-parallel), so throughput (samples/sec) is close to 2x —
the expected scaling for a data-parallel run that isn't yet communication- or
CPU-dataloader-bound at this batch size.

### Env gotchas hit on a fresh box

- `torch==2.14.0`'s default wheel is a **CUDA 13.0** build, which dropped
  kernel support for Volta (sm_70) entirely. Reinstall against cu126:
  `pip install torch==2.14.0 torchvision --index-url https://download.pytorch.org/whl/cu126`
- `bitsandbytes` isn't in `requirements.txt` — install separately
  (`pip install bitsandbytes==0.50.2`).
- `requirements.txt`'s `open3d==0.18.0` pin has no Python 3.12 wheel and isn't
  imported anywhere in this pipeline — drop that line rather than chase it.

Not yet run: a full (non-smoke) training run over the full competition split.

## Running this via `phd` (PhD cluster)

`run_trex_job.sh` hardcoded `scripts/train_origami.sh` (the stale chunk25/
anchor-era recipe). It now reads a `TRAIN_SCRIPT` env var instead, defaulting
to the old script for backward compat — point it at `run_all_abs_ori.sh` to
get everything verified above. `run_trex_job.sh` still does its own thing
(NFS→scratch copy of `DATA_ROOT_SRC`, `PROJECT_ROOT`/`ASSET_ROOT`/`OUTPUT_DIR`
exports); `run_all_abs_ori.sh` just reads whatever it exports, same as
`train_origami.sh` did.

```bash
# smoke check, 1 GPU, on-cluster
TRAIN_SCRIPT=scripts/run_all_abs_ori.sh SMOKE=1 \
  phd run -ng 1 -p shr_gpu -GR H100 -l %J.log sh run_trex_job.sh 1

# real run, 2 GPUs, with this session's overrides
EXPERIMENT_NAME=all_abs_ori_run1 \
TRAIN_SCRIPT=scripts/run_all_abs_ori.sh \
DATA_ROOT_SRC=${HOME}/other/new_data/competition \
MASK_FROZEN_LOSS=1 MAX_EPISODES=0 \
TRAIN_BSZ=8 GRAD_ACCUM=4 LR=1.5e-4 N_EPOCHS=3 \
  phd run -ng 2 -p shr_gpu -GR H100 -l %J.log sh run_trex_job.sh 2
```

Any `run_all_abs_ori.sh` env var (`TRAIN_BSZ`, `GRAD_ACCUM`, `LR`,
`MIN_LR_RATIO`, `N_EPOCHS`, `MASK_FROZEN_LOSS`, `MAX_EPISODES`,
`TRAIN_LATENT_LAST_N`, `SAVE_STEPS`, `VAL_FREQ`, `RESUME=1`, etc.) can be set
the same way, exported before the `phd run` invocation — `run_trex_job.sh`
passes its whole environment through to whatever `TRAIN_SCRIPT` it launches.
`DATA_ROOT_SRC` must point at a directory with `train/`+`val/` subfolders in
this branch's schema (e.g. a synced copy of the `competition-paper-set` pull
from `drakedrake/ori-trex-competition`), not the old anchor-era layout.
