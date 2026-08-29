# T-Rex origami: why results are poor, and what to change

Grounded in `checkpoint-1-5500`'s `metrics.json` / `robustness.json` and the
training code. Numbers below are measured, not estimated.

## 0. Blocking issue first

`checkpoint-0-5/training_state.json` says `{"epoch": 0, "global_step": 5}` —
that is the **5-step smoke-test checkpoint**, not a trained model. Do not ship
or eval it. Its weights file is also named `model-001.pt`; `train.py` only ever
writes `model.pt`, so rename before use.

## 1. The core result: the policy loses to doing nothing

| row | MAE (deg) | vs hold_state |
|---|---|---|
| cascaded | 2.259 | **+40% worse** |
| blind | 2.118 | +31% worse |
| hold_state (zero delta) | 1.612 | — |
| repeat_command (oracle) | 0.704 | -56% |

This holds at **every** horizon step (k=0: 1.039 vs 1.003; k=24: 3.320 vs 2.153).
The action head predicts a delta from `observation/state`; predicting exactly
zero would score better. At k=0 the model's error (1.039°) equals the signal
magnitude itself (hold_state MAE = mean |true delta| = 1.003°) — i.e. **~zero
skill**, not "slightly imperfect skill". Everything below follows from this.

## 2. Ranked changes

### A. Train on more data, for longer (highest expected value)
`stats_data.json`: `num_trajectories: 97`, `num_transitions: 40299` — this is
the **pilot tier only** (~20% of seasons). The full tier is 101 train seasons.
`train_origami.sh` defaults to `N_EPOCHS=2` at `LR=5e-5`, effective batch 32
(8×4×1 GPU) vs the paper's reference effective batch 128 at LR 1e-4.
- Move to the full split; raise epochs.
- On multi-GPU, scale LR toward 1e-4 (`train_origami.sh` explicitly does *not*
  auto-scale it — see its own note).

### B. Unfreeze more of the latent expert
`--freeze_latent_expert 1 --train_latent_last_n 4` trains only the top 4 of 28
layers. The task is a large visual-domain shift from Qwen3-VL pretraining
(top-down fold plane, fixed rig). Try `train_latent_last_n 8-12`, budget
permitting.

### C. Fix the FLARE dead weight
`train_origami.sh` sets `--use_flare 1` with `--flare_loss_weight 0.0`. So 32
flare query tokens sit in the latent sequence (~31% of it) trained toward
nothing. Either enable the loss (paper default 0.5) or set `--use_flare 0` and
reclaim the tokens. Currently it is the worst of both.

### D. Add a smoothness penalty — the clearest actionable defect
Safety violation rates, model vs the teleoperator's own chunks:

| check | cascaded | teleop_gt | ratio |
|---|---|---|---|
| step_jump | 0.0232 | 0.00018 | **130x** |
| velocity | 0.0291 | 0.00030 | **98x** |
| upper_limit | 0.0283 | 0.0269 | 1.05x |

`upper_limit` matching teleop confirms it is URDF conservatism, not the model.
But step_jump/velocity being ~100x worse is genuine, and MAE cannot see it. Add
a 2nd-difference (jerk) penalty to the training loss, and/or temporal smoothing
at serving.

### E. Tactile currently contributes nothing — diagnose before investing
From `robustness.json` `delta_vs_nominal_deg`:

| perturbation | delta (deg) |
|---|---|
| blank both wrist cameras | **+0.251** |
| vision stale 3 rows (500ms) | +0.029 |
| **drop tactile entirely** | +0.0018 |
| tactile stale 1-3 rows | ~-0.00003 (noise) |
| tactile history frozen/subsampled | ~0.00003 (noise) |

You can delete, freeze, or delay the tactile signal and the model does not
notice. Consistent with `blind` beating `cascaded`. Before tuning the cascade,
check whether `loss_tac` is actually decreasing during training — if it is flat,
the tactile expert never learned, and `--cascaded_split_step` tuning is
premature.

### F. Replan often at serving (free, no retraining)
`robustness.json` `by_replan_period`, stitched MAE / mean seam jump / % seams
tripping a safety check:

| replan gap | stitched MAE | seam jump | bad seams |
|---|---|---|---|
| 167 ms | 1.258 | 1.47 | 67% |
| 333 ms | 1.581 | 2.04 | 81% |
| 500 ms | 1.832 | 2.65 | 90% |
| 667 ms | 2.090 | 3.06 | 98% |

Execute as few steps per chunk as the latency budget allows.

## 3. Vision resolution — verified, and *not* the main problem

Measured by running the checkpoint's own processor on a 224x224 input:

```
patch_size 16, merge_size 2, min_pixels(shortest_edge) 65536
224x224 (50176 px) < min_pixels  ->  smart_resize UPSCALES to 256x256
  -> 16x16 = 256 patches -> 8x8 = 64 LLM tokens per image
  -> 3 images (head, wrist_r, wrist_l) = 192 image tokens
```

Two consequences:
- The 224->256 upscale is pure interpolation: **no information gained**, ~30%
  more vision compute than needed. Train and inference both do it, so there is
  no train/test mismatch — it is a latency cost, not an accuracy bug.
- 64 tokens/image is coarse (8x8 spatial cells). The original T-Rex config
  (`test.sh`, `image_size 384 288`) yields ~108 tokens/image — we run at ~60% of
  that spatial budget.

**But 224x224 is the wire contract** (`robot_io_spec.md`), so it is the hard
information ceiling; prepping training data at higher resolution would only
create a train/deploy mismatch. Given (1) the model currently loses to
do-nothing and (2) blanking wrists costs 0.25 deg while every tactile ablation
costs ~0, resolution is not the binding constraint. Revisit only after A-D.

## 4. Not worth changing yet
- `cascaded_split_step` (6/10): tuning the tactile expert's share is pointless
  while tactile is a measured no-op (E).
- URDF `upper_limit` violations: matches teleop_gt, so it is the URDF's finger
  bounds being tighter than the hardware, not a policy defect.
