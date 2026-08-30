# trex_origami — Robotic Origami Challenge → T-Rex

Data prep for post-training [T-Rex](../README.md) on Sharpa's
[Robotic Origami Challenge](https://huggingface.co/datasets/SharpaIT/Robotic_Origami_Challenge)
fold-plane demonstrations (IROS 2026).

Torch-free (pyarrow + PIL + ffmpeg), so the whole pipeline runs on a CPU box while
training happens on a GPU elsewhere.

## Why the data has to be reshaped at all

T-Rex's shipped loaders assume its own recording format. Three mismatches force a
conversion rather than a config change:

| | T-Rex | Robotic Origami | Consequence |
|---|---|---|---|
| Action space | eef-62: `[L_arm_delta 9 \| L_hand 22 \| R_arm_delta 9 \| R_hand 22]` | 65-D joint, and `observation.state.tcp` is **identically zero** | No end-effector poses exist, so we train in 65-D joint space and re-init the four action-head tensors |
| Horizon | chunk 16 | the kit's `action_horizon` is **25** | Chunk length touches no weights, so 25 is free |
| Deform maps | 10 separate per-finger videos | one 1200×480 strip, 2×5 grid | Split at load; the 240×240 cell size is already what `DeformEncoder` requires |

What *does* transfer untouched: the MoT backbone, the tactile expert, the embedded
tactile VQ-VAE and the deform encoder. Both robots use Sharpa Wave hands, so the
22-DoF-per-hand block and the 10×6 fingertip wrench layout line up exactly.

Going through LeRobot v3.0 instead was rejected because the release ships one
concatenated mp4 per camera *per season*: a random-access sample would cost four
AV1 seeks and pin the sample rate to the source 30 Hz.

## Output format ("origami-flat")

```
<root>/
  meta/dataset.json     config + episode index
  meta/norm_stats.json  q01/q99 normalisation
  data/<season>/ep<NNN>.parquet
```

One parquet row = one training sample = one row-group read + four JPEG decodes.
At source frame `t` of an episode of length `N`:

| column | shape | contents |
|---|---|---|
| `state` | 65 | `observation.state[t]` |
| `action_chunk` | 25×65 | `action[min(t+k, N-1)] - anchor[t]`, k = 0..24 |
| `action_chunk_abs` | 25×65 | `action[min(t+k, N-1)]`, k = 0..24 — the same targets unanchored |
| `action_abs` | 65 | `action[t]` |
| `prev_command` | 65 | `action[t-1]` (`action[t]` at an episode start) |
| `phase` | — | episode progress in [0, 1] |
| `tacf6_hist` | 16×10×6 | `tactile[clip(t-15+i, 0, t)]`, **at the native 30 Hz** |
| `head`, `wrist_left`, `wrist_right` | — | JPEG, 224×224 RGB |
| `deform` | — | JPEG, 1200×480 grayscale |

### What the 65 numbers are measured from

`anchor[t]` is **per dimension**, defined in `anchoring.py` and recorded in
`meta/dataset.json` as a 65-entry `action_anchor` list. Under the default
`--anchor-mode hybrid`:

| dims | group | anchor | target is |
|---|---|---|---|
| 0–6, 29–35 | arms (14) | `prev_command[t]` | delta from the previous command |
| 7–28, 36–57 | hands (44) | 0 | the absolute joint angle |
| 58–64 | torso/neck motor (7) | 0 | the absolute joint angle |

This matches the paper's own per-group anchoring (§5.1, and the released
`…deltabase_axis_eef…` stats): arms relative, fingers absolute. The first
fine-tune used `--anchor-mode state` — every dim a delta from `observation.state[t]`
— which cost it two ways. The constant part of every target became the
command-minus-state tracking offset (~0.74°), which is not observable from any
model input because it lives in the *command* history, so `hold_state` was
unbeatable by construction; and q01/q99 over near-static delta dims spanned little
more than that jitter, so normalisation stretched it to the full [-1, 1] range and
the flow head's residual sampling noise denormalised back into degrees of finger
wobble. `--anchor-mode state` still reproduces that prep exactly, and a dataset
written before this existed declares no rule and is read as `state`.

Absolute radians — the wire contract — come back the same way everywhere:

```python
from trex_origami.anchoring import build_anchor, spec_from_meta, to_absolute
spec   = spec_from_meta(meta)                          # from meta/dataset.json
anchor = build_anchor(state, prev_command, spec)       # [..., 65]
absolute = to_absolute(predicted_chunk, anchor)        # [..., 25, 65]
```

At deployment `prev_command` is the policy's own last emitted command, which it
always has; `scripts/test.py` tracks it and serves absolute radians directly,
because the rule is per-dim and no client can do the reconstruction for us.

Because that anchor is a nuisance shortcut the policy could simply copy (de Haan
et al., causal confusion) — and open-loop MAE *rewards* copying it — it enters only
additively at the output, never as a network input, and training perturbs it:
`--anchor_noise_mode tracking` adds per-dim noise with the measured tracking-error
statistics, and `--anchor_dropout 0.15` re-anchors a fraction of samples to
`state[t]` so the policy stays usable once its own command stream has drifted.
The acceptance metric is correspondingly the **motion** metric (MAE with each
chunk's own step 0 removed), not raw MAE.

Two other deliberate choices:

- **The F6 history stays at 30 Hz** regardless of `sample_stride`, because the
  embedded VQ-VAE was trained on 30 Hz windows.
- **Frozen dims are held at `state[j]`, not zeroed.** `stats.py` masks off any dim
  whose q01..q99 spread is degenerate (the torso's `lower_body_joint_1/2`), and
  `_normalize` passes those through un-scaled, so the flow head's residual noise
  reaches the wire in raw radians. Eval and serving clamp them — but on the
  *absolute* chunk, since under hybrid anchoring those dims are absolute and
  zeroing one would command 0 rad, a full-travel move.

## Usage

```bash
bash trex_origami/run_prepare_fast.sh pilot   # 10 train + 3 val seasons, stride 5 (6 Hz)
bash trex_origami/run_prepare_fast.sh dense   # 30 train + 8 val seasons, stride 5
bash trex_origami/run_prepare_fast.sh full    # 101 train + 25 val seasons, stride 20
```

`run_prepare_fast.sh` overlaps the downloads with the conversions and puts the RGB
streams on NVDEC; `run_prepare.sh` is the serial, CPU-only equivalent and produces
the same bytes. Both default to `ANCHOR_MODE=hybrid PHASE_MODE=progress`; set
`ANCHOR_MODE=state PHASE_MODE=none` to reproduce the first fine-tune's prep.

The `full` tier writes **three** roots: `train` and `val` at stride 20, plus
`val_stride5` — the first 5 val seasons re-prepared at stride 5. One dataset row is
one delay quantum and one replan period, so at stride 20 `eval_robustness.py` can
only resolve 667 ms and consecutive chunks overlap by 5 of 25 steps; the delay sweep
and the temporal-ensemble rollout both need the finer root to measure what they claim.

Seasons stream from the hub one at a time and are deleted after conversion, so peak
disk is `output-so-far + disk_budget seasons` rather than the ~300 GB the raw release
needs. Only `head_left`, `wrist_left`, `wrist_right` and `tactile_deform` are fetched
— skipping `tactile_raw` alone drops 73% of the bytes.

Reruns are resumable, but only for the *same* contract: changing `--anchor-mode`,
`--phase-mode`, the stride or the chunk shape is refused against an existing root,
because already-converted seasons would be skipped and the split would end up
half in one target space and half in the other with a `dataset.json` claiming one
of them. Use a fresh `--out-root` or `--overwrite`.

**Codecs.** The `lerobot3.0` exports are *mostly AV1*, with some h264 seasons
(the dataset card: "lerobot3.0 is primarily AV1, lerobotv2.1 is primarily H.264").
AV1 costs roughly 2–4× more CPU per frame through libdav1d, and NVDEC only decodes
AV1 from Ampere on — so on an A100/V100/T4 the h264 seasons take the GPU and the
AV1 ones fall back to the CPU. `accel.py` probes each codec end-to-end once at
startup, prints which decoders are live, and retires a decoder for the rest of the
process the first time a real file fails on it, so a GPU that cannot do AV1 costs
one failed decode rather than one per file. The deform strip always stays on the
CPU: it is not downscaled, so reading full 1200×480 frames back off the GPU costs
more than NVDEC saves. When most seasons land on the CPU, `CONVERTERS` is the knob
that matters.

Individual stages:

```bash
python -m trex_origami.prepare --split train --limit 10 --out-root <dir> \
       --sample-stride 5 --anchor-mode hybrid --phase-mode progress
python -m trex_origami.stats   --root <train dir> --copy-to <val dir> <val_stride5 dir>
python -m trex_origami.verify  --root <train dir> --src-root <raw seasons> --montage m.png
```

`stats` fits normalisation on **train only** and copies it to every val root, so the
splits scale identically and their losses stay comparable.

## Verification

`verify.py` is the gate before anything is uploaded:

- episode files match the declared row counts and schema
- norm-stats ranks: `action` is **[25, 65]** — per *(step, dim)*. A `[65]` action block
  broadcasts silently and mis-scales every step of the chunk, which is the easiest way
  to get a model that trains to a plausible loss and predicts nonsense.
- normalised values land in [-1, 1]
- the deform strip splits into 2×5 tiles of 240×240
- **deform/force alignment**: per-finger tile brightness is correlated against per-finger
  contact force over ~150 frames and the 10×10 matrix must be diagonal. A transposed
  grid would pair one finger's image with another's wrench and nothing downstream
  would complain, since the shapes are identical either way. On correct data the
  diagonal sits near 0.9 against an off-diagonal near 0.05.
- **anchoring round-trip**: `anchor + action_chunk[0]` must reproduce `action_abs` on
  every dim, and the dims declared absolute must equal `action_abs` at step 0 *without*
  anyone adding the state. A mis-declared `action_anchor` fails here rather than
  surfacing as a constant bias in every metric days later.
- with `--src-root`, one episode's numeric columns are re-derived from the raw season
  under the declared anchoring rule and compared exactly

Then check the batch contract the trainer actually consumes:

```bash
python scripts/smoke_test_origami.py --root <train dir>
```

## Fine-tuning on Colab (single A100)

Prep runs on CPU, training runs on one A100-40GB. The two are deliberately
decoupled: nothing in this section touches the hub or ffmpeg, so a pre-empted
session costs GPU minutes, not a re-download.

### 0. Get the code and data onto Drive

A Colab session starts empty, and the pilot dataset is ~6 GB of JPEG blobs — too
slow to re-fetch per session. Zip both once from the prep box:

```bash
cd /home/kshitij/origami_trex
zip -qr T-Rex.zip T-Rex -x '*/.git/*' '*/__pycache__/*'
cd data && zip -qr ../trex_data_pilot.zip origami_flat/pilot && cd ..
# upload T-Rex.zip, trex_data_pilot.zip and the URDF (128 KB) to
# Drive:/MyDrive/iros2026/
```

The URDF (`north_poc2_2_urdf_usd/north_poc2_2_v3_1.urdf`) is only needed for the
eval safety checks, but bring it: those are the same limit/velocity checks that
block a competition submission, and without the file `eval_origami.py` silently
skips them.

Alternative to Drive — push the prepared split to a private HF dataset repo and
let `colab_setup.sh` pull it (`DATA_REPO=<user>/origami_flat_pilot`). Faster on a
fresh session, but you pay the upload once and the repo counts against your HF
storage.

### 1. Runtime

**Runtime → Change runtime type → A100 GPU**, and **High-RAM** if offered. Check
what you actually got before spending anything else:

```python
!nvidia-smi --query-gpu=name,memory.total --format=csv
```

A 40 GB A100 is the target. A V100/L4 will not hold this model at the default
batch; see the OOM ladder in §6. Colab hands out T4s freely, so verify.

### 2. Unpack

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
!unzip -q /content/drive/MyDrive/iros2026/trex_data_pilot.zip -d /content/
!unzip -q /content/drive/MyDrive/iros2026/T-Rex.zip -d /content/
!cp /content/drive/MyDrive/iros2026/north_poc2_2_v3_1.urdf /content/
```

That lands the tree at `/content/code/T-Rex` and the data at
`/content/data/origami_flat/pilot/{train,val}`. Both differ from the script
defaults (`/content/T-Rex`, `/content/data/origami_flat`), so **every cell below
sets `PROJECT_ROOT` and `DATA_ROOT` explicitly**. Getting this wrong surfaces as
a `FileNotFoundError` on `meta/dataset.json`.

### 3. Environment and weights

```python
%env PROJECT_ROOT=/content/code/T-Rex
!bash /content/code/T-Rex/scripts/colab_setup.sh
```

Installs deps with `uv`, then downloads `Qwen/Qwen3-VL-2B-Instruct` (the
architecture + processor) and `miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6`
into `/content/assets`. Roughly 10 GB and 5–10 min on a good session. Every step
is skipped if its output already exists, so re-run it freely after a
pre-emption.

Two things to read in its output rather than skim:

- `use_tactile_vqvae: 1 | vqvae_config: present` — if `vqvae_config` says
  `MISSING`, the embedded tactile tokenizer did not come down and
  `--resume_source midtrain` will hand you an untrained tactile expert. That is
  the ablation the paper measures at 65% → 45% success; stop and re-download.
- `action_dim: 62 | action_chunk: 16` — expected. We train 65/25, and §5
  explains which tensors that re-initialises.

If the torch install forces a version change, Colab needs **Runtime → Restart
session** before the first import; re-run the setup cell afterwards (it will
no-op through the downloads).

**Post-train from the midtrain checkpoint, never the pretrain one.** The paper's
ablation: full recipe 65% average success, pretrain-only 45%, midtrain-only 34%,
neither 18%.

### 4. Smoke test, twice

```python
%%bash
cd /content/code/T-Rex
export PYTHONPATH=/content/code/T-Rex
python scripts/smoke_test_origami.py --root /content/data/origami_flat/pilot/train
```

CPU-only, ~30 s: checks all 23 batch keys and their shapes against the
configured chunk/dim, the flow-matching identity, the [-1, 1] normalisation
bound, that the F6 history arrives un-normalised, and that the sampler emits a
true permutation. This is the cheap gate — a shape bug found here costs seconds,
the same bug found by the trainer costs a model load.

Then five real optimiser steps on the GPU:

```python
%%bash
export PROJECT_ROOT=/content/code/T-Rex
export DATA_ROOT=/content/data/origami_flat/pilot
export OUTPUT_DIR=/content/outputs
SMOKE=1 bash /content/code/T-Rex/scripts/train_origami.sh
```

This proves the model loads, fits in memory, and that the loss is finite before
you commit hours. Note the reported `it/s` — §6 turns it into a wall-clock
estimate.

### 5. Train

```python
%%bash
export PROJECT_ROOT=/content/code/T-Rex
export DATA_ROOT=/content/data/origami_flat/pilot
export OUTPUT_DIR=/content/outputs
export WANDB_MODE=offline
bash /content/code/T-Rex/scripts/train_origami.sh 2>&1 | tee -a /content/train.log
```

Defaults, all overridable from the environment:

| | value | why |
|---|---|---|
| `TRAIN_BSZ` × `GRAD_ACCUM` | 8 × 4 = 32 | the paper's Table 4 is effective 384 (16/device × 24 H100) |
| `LR` | 1.5e-4 | the authors' own post-train launcher is 1e-4 at effective 128 and Table 4 is 1e-4 at 384. The first fine-tune ran 5e-5 at effective 512 — ~8× under-scaled — and its loss was flat by ~2k optimizer steps. 1.5e-4 sits between the paper-faithful 1e-4 and the 2e-4 sqrt-scaling gives; drop to 1e-4 if the first 200 steps look unstable |
| `MIN_LR_RATIO` | 0 | single half-cosine to zero, as in the authors' `scripts/train.sh`. Warmup stays at 3% — a deliberate deviation from their 0, because we cold-start five head tensors for the 62→65 change and they do not |
| `N_EPOCHS` | 3 | the authors post-train for 100; two was short even before the LR problem |
| `ANCHOR_NOISE_MODE` / `ANCHOR_DROPOUT` | `tracking` / 0.15 | keep the policy from simply copying the previous-command anchor, and keep it usable once its own command stream has drifted. Both touch only the dims anchored to the previous command, so they are no-ops on a `--anchor-mode state` dataset |
| `--image_size` | 224 224 | 49 vision tokens/image × 3 cameras |
| `--freeze_latent_expert 1`, `TRAIN_LATENT_LAST_N=4` | | the latent expert only encodes RGB+language; the action and tactile experts are what adapt. Freezing all but the top 4 layers is what makes 40 GB work |
| `--optim adamw8bit` | | 2 B/param of optimiser state instead of 4, and better behaved than AdamW's bf16 moments |
| `FLARE_LOSS_WEIGHT` | 0.0 | with the latent expert frozen the FLARE loss can only train `flare_proj`, yet costs 8 extra head-frame decodes and a ViT pass per sample. The 32 query tokens stay in the sequence, so the action expert still sees what the checkpoint trained it on |

Expected startup lines, in order:

```
[vqvae] embedded VQ-VAE auto-detected from resume checkpoint
Skipped N keys with shape mismatch (e.g. x_embedder.mlp.fc1.weight)
Resumed: missing=..., unexpected=...
Tactile expert weights kept from resumed midtrain checkpoint.
Latent expert frozen (... last 4 of 28 layers trainable).
Gradient checkpointing enabled on the MoT decoder layers.
Model: <total>M total, <trainable>M trainable
Optimizer: bitsandbytes AdamW8bit
[origami] /content/data/.../train: 161045 samples / 97 episodes / 10 seasons | ...
```

The skipped keys must all belong to **four modules only** — `x_embedder`,
`final_layer`, `final_layer_tactile`, `state_embedder` (5–6 tensors, since the
in/out projections and their biases count separately, and `state_embedder` is
absent from the checkpoint entirely if it was midtrained with
`use_robot_state 0`). Those are the 62→65 action-space change and are expected.
Anything else in that list means the checkpoint does not match this
architecture, and you are quietly training a partly random model — print the
full list before continuing. The `DeformEncoder checkpoint not found at`
warning is benign: no `--deform_encoder_ckpt` is passed because the midtrain
`model.pt` already carries `deform_encoder.*`.

Then per-step postfix `loss / act / tac / fut / lr`, and every `VAL_FREQ` steps:

```
  [Val step=1000] act=0.xxxxxx tac=0.xxxxxx
```

`act` and `tac` are flow-matching **velocity** MSE, not action error — they are
comparable across runs but say nothing directly about radians. Watch that `tac`
is finite and tracking `act`; a `tac` that stays flat while `act` falls means the
tactile expert is not learning, and the §7 ablation will show no gap.

**`--save_steps` and `--val_freq` count dataloader batches, not optimiser
steps.** At 8 × 4 the default `--save_steps 2000` is every 500 optimiser steps.

### 6. Budget, and what to cut first

Planning numbers for the `pilot` tier (161 k samples, stride 5): ~20 k
micro-steps per epoch at batch 8, ≈2–2.5 h/epoch. The smoke run's `it/s` is the
real number — multiply it out before starting a 2-epoch run against a session
limit.

Checkpoints are the full model, so ~8 GB each for `model.pt` plus ~4 GB of
`state/` when `--save_optimizer_state 1`. `--max_ckpts 3` therefore wants ~35 GB
of local disk. Check with `!df -h /content` and `!du -sh /content/outputs/*/*/*`,
and copy only the checkpoint you intend to keep to Drive — a 12 GB Drive write
mid-run is slower than the training step it blocks.

If it OOMs, in this order (each roughly halves activation memory or optimiser
state, in increasing cost to quality):

1. `TRAIN_BSZ=4 GRAD_ACCUM=8` — same effective batch, half the activations.
2. `TRAIN_LATENT_LAST_N=0` — freeze the latent expert completely.
3. `NUM_WORKERS=4` — worker processes each hold a row-group cache.
4. `TRAIN_BSZ=2 GRAD_ACCUM=16`.

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is already exported by the
launcher; the MoT backbone allocates in bursty shapes and without it a long run
can fragment into an OOM after a few thousand steps.

### 7. Resume after a pre-emption

Colab will kill the session. Checkpoints carry optimiser, scheduler, RNG and
step counter, so resume is exact rather than a fresh LR schedule:

```python
%%bash
export PROJECT_ROOT=/content/code/T-Rex
export DATA_ROOT=/content/data/origami_flat/pilot
export OUTPUT_DIR=/content/outputs
RESUME=1 bash /content/code/T-Rex/scripts/train_origami.sh
```

`RESUME=1` picks the newest `${OUTPUT_DIR}/${EXPERIMENT_NAME}/*/checkpoint-*`,
reuses its run name, and adds `--resume_full_state 1`. It expects
`--save_optimizer_state 1` to have been on (it is, unless `SMOKE=1`).

If `/content/outputs` was on ephemeral disk and is gone, the run restarts from
the midtrain checkpoint. Either keep `OUTPUT_DIR` on Drive from the start and
accept the write cost, or `cp -r` the latest checkpoint to Drive at the end of
each session.

### 8. Evaluate

```python
%%bash
export PYTHONPATH=/content/code/T-Rex
cd /content/code/T-Rex
CKPT=$(ls -dt /content/outputs/*/*/checkpoint-* | head -1)
python scripts/eval_origami.py \
  --checkpoint_path "${CKPT}" \
  --origami_root /content/data/origami_flat/pilot/val \
  --out_dir /content/eval/$(basename "${CKPT}") \
  --urdf /content/north_poc2_2_v3_1.urdf \
  --num_eval_samples 2000 --batch_size 8 \
  --modes cascaded blind
```

Runs the deployed policy path — slow tick `forward_flow_action_partial`, then
tactile fast tick `tactile_flow_continue` — over held-out **seasons** (never
episodes from a training session; same paper, same lighting, same paper batch).
Reports MAE/RMSE in radians and degrees per joint group, the per-step horizon
curve over k = 0..24, a contact-only split, latency, and the URDF/velocity
safety checks.

Everything is scored in **absolute radians**, reconstructed through the
dataset's declared anchoring rule, and the script hard-exits if the checkpoint
was trained under a different one. That keeps the baselines meaning the same
thing across preps that anchor differently.

Read four numbers first:

0. **`motion`** — MAE with each chunk's own step-0 value removed, i.e. the shape
   of the coming movement rather than the pose it starts from. This is the
   acceptance metric: with the arms anchored to the previous command, step 0 is
   nearly free, so a policy that emits a constant chunk equal to its anchor
   scores a respectable raw MAE having predicted nothing. All three constant
   baselines collapse to one identical `motion` score — the static floor — and
   beating it is the only evidence of motion skill.
1. **vs the naive baselines** (`hold_state`, `oracle_prev_command`,
   `repeat_command`). On a 0.83 s horizon these are not weak. A policy that does
   not beat them has learned nothing, and they are the only floor available — the
   released midtrain checkpoint cannot serve as a zero-shot baseline, because its
   62-D eef action head cannot emit a valid 65-D joint action for this task.
   `oracle_prev_command` is the one to watch: it repeats what the policy last
   sent, which under hybrid anchoring is precisely the arms' own anchor, so it is
   what "predict zero" now buys.
2. **`cascaded` vs `blind`** on the *contact* split. `blind` is
   `forward_flow_action_full`, the action expert alone. The gap on contact frames
   is what the tactile expert buys; averaged over the whole split it is diluted,
   since only thumb and index ever load up.
3. **the horizon curve**. Accurate at k=0 and diverging by k=24 is a different
   failure from uniformly biased, and only the first is fixed by more data.

These are all **open-loop chunk-prediction** metrics. Actions change the world
and a recorded episode cannot react, so none of them is a folding success rate;
they rank checkpoints and localise failures. Success comes from the challenge's
real-world lab or an Isaac Sim rollout.

**Inference-time smoothing** (`scripts/eval_smoothed.py`) scores the three
deployment-side fixes that need no retraining: batched mean-of-K flow draws
(sampling variance falls as 1/K — K=4 removes ~75% of the removable MSE),
ACT-style temporal ensembling over overlapping chunks in a receding-horizon
rollout, and a safety projection that clamps the absolute command stream to
the Shadow evaluator's own position/step/velocity limits so the violation rate
goes to zero by construction. Chunks overlap by `25 − sample_stride` steps, so
run the rollout pass on a stride-5 split for the full ensembling effect.

### 9. If the first run underperforms

In expected-value order:

1. **`--phase-mode progress` at prep time** — now the default. A 6-minute,
   6-fold task conditioned on one static sentence is badly under-specified, and
   the dataset's only task string is literally `north ces task`; appending the
   fold index gives the policy a phase signal. Note the deployment caveat: the
   baked fraction is `(t − s) / (N − 1)`, which needs the episode's *total*
   length, and the robot only knows how long it has been folding. The prep
   therefore records `median_episode_frames` in `meta/dataset.json`, and
   `scripts/test.py --phase_mode progress --phase_episode_seconds <median/30>`
   reproduces the same prompt online from elapsed wall-clock. It degrades
   gracefully — a slow attempt just reaches the later folds late — but the
   train/deploy signals are not identical, so `PHASE_MODE=none` remains the
   conservative choice.
2. **More seasons before more epochs** — `run_prepare.sh dense`. Ten seasons is
   ten lighting/paper/operator conditions; the split is by season precisely
   because episodes within one are near-duplicates.
3. **`--sample_stride`** — stride 5 samples at 6 Hz. Denser sampling buys
   correlated samples, not new information.
4. **`--cascaded_split_step`** (τ_split = 0.4 at 6/10). The paper reports a
   peaked optimum and says to re-tune it when the chunk length or control rate
   changes — ours changed from 16 to 25.

## Notes on the source data

- Video packing is **non-uniform**: `head_*` hold ~2 episodes per file, `tactile_deform`
  can hold a whole season, `wrist_*` hold one. Episode→file mapping must come from
  `meta/episodes/**.parquet` (`file_index` + `from_timestamp`), never from filenames.
- `observation.state.tcp` and `joint_torque[58:65]` are identically zero.
- Only thumb and index actually make contact during folding (~1.6 N left, ~5.6 N right);
  middle/ring/little stay under 0.4 N.
- `lower_body_joint_1/2` move less than 5e-4 rad across a season, so `stats.py` masks
  them out of normalisation rather than stretching that noise to full scale.
- The teleoperated actions themselves exceed the URDF's finger position limits on ~3%
  of values, so the URDF is conservative relative to the hardware — see the `teleop_gt`
  reference row in `scripts/eval_origami.py`.
