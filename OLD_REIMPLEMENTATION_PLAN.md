# Reimplementation Plan — T-Rex Origami Fine-tune, Attempt 3

Consolidated plan from the checkpoint-2-8000 post-mortem (2026-08-28/29). One full
fine-tune budget remains, so every training-side change ships in a single run;
everything else is prep-time (CPU) or inference-time (no retraining).

## Status (2026-08-29)

Sections 1-5 are **implemented in the tree**; nothing here has been *run* yet.
What landed, and where:

| item | where |
|---|---|
| 3a/3b hybrid anchoring + previous-command anchor | `trex_origami/anchoring.py` (new, single source of truth), `prepare.py` (`--anchor-mode`, new `prev_command` column, `action_anchor` in `meta/dataset.json`) |
| 3b anchor noise + anchor dropout | `origami_dataset.reanchor_chunk`, `--anchor_noise_mode` / `--anchor_dropout` |
| 3c phase conditioning | `--phase-mode progress` is now the prep default; `median_episode_frames` recorded so serving can reproduce the prompt from elapsed time |
| 3d stats + verify gates | anchoring round-trip in `verify.py`, anchoring check in `preflight.py`, prep refuses to mix two contracts in one root |
| 2/4 LR + schedule | `train_origami.sh`: LR 1.5e-4, `--min_lr_ratio 0`, 3 epochs, warmup 3% kept |
| 5 per-dim anchoring in every consumer | `eval_origami.py`, `eval_diagnostics.py`, `eval_robustness.py`, `eval_smoothed.py`, `test.py` all reconstruct absolute radians through the declared rule and hard-exit on a checkpoint/data mismatch |
| 5 motion-only metric | `ErrorAccumulator` `motion` block (step-0 removed) in `eval_origami` and `eval_smoothed`; the three constant baselines collapse onto one static floor there |
| AV1/h264 mix | `accel.py` probes NVDEC per codec and retires a decoder that fails on a real file; `prepare_fast.py` logs the verdict once instead of warning per file |
| stride-5 val slice (§6.2) | `run_prepare_fast.sh full` also writes `full/val_stride5` |

Two deviations from the text below, both deliberate:

- **The frozen-dim clamp is expressed on the absolute chunk**, not as "zero the
  delta". Under hybrid anchoring a frozen dim is an *absolute* dim, so zeroing it
  would command 0 rad — a full-travel move — instead of holding. On the legacy
  all-delta prep the two are the same operation, so nothing changes there.
- **§5's motion-only metric is defined as step-0 removal**, not `pred_abs - anchor`
  vs `gt_abs - anchor`: subtracting the same anchor from both sides is algebraically
  identical to raw MAE and would measure nothing. Removing each chunk's *own* step 0
  is what actually denies credit for copying the anchor.

Open, and flagged rather than silently resolved: `--phase-mode progress` bakes
`(t - s) / (N - 1)`, which needs the episode's total length — a number the robot
does not have. `scripts/test.py --phase_mode progress --phase_episode_seconds`
approximates it from elapsed wall-clock against the training median, which
degrades gracefully but is not the identical signal. `PHASE_MODE=none` remains
the conservative choice.

## 0. Why the last run underperformed — the evidence

Checkpoint `trex_origami_full_0827_2005/checkpoint-2-8000` (99 seasons, 410k samples,
2 epochs, eff. batch 512, LR 5e-5), evaluated on 25 held-out seasons:

| finding | evidence |
|---|---|
| Policy loses to both naive floors at **every** horizon step | blind MAE 2.07°, cascaded 2.28° vs hold_state 1.42°, repeat_command 0.79° (`checkpoint-2-8000/metrics.json`); only 2/65 joints beat hold_state |
| ~34% of MSE is **flow sampling variance**, not bias | mean-of-8 draws: MAE 2.50°→1.64°, chunk jerk 3.65°→1.30° (teleop: 0.05°) (`diagnostics/diagnostics.json`) |
| The finger wobble **is** that variance | traces: arms track teleop closely; thumb CMC_FE / index PIP oscillate around it in single draws, settle under mean-of-8. Worst joints = contact fingers (thumb CMC_FE 6.4°, 2.9× hold_state) |
| Tactile cascade **subtracts** value | cascaded +0.21° worse than blind in every group incl. arms; tactile input delayed 1–3 rows changes MAE by 0.0002° (vision: 0.05–0.13°) — the tactile pathway is ignored |
| Every chunk violates safety checks | `chunk_violation_rate = 1.0`; velocity 3.1%, step-jump 2.5% of values (teleop: 0.05%) — submission-blocking |
| More training of the same recipe won't help | epoch-1-5000 (pilot) ≈ epoch-2-8000 (full) ≈ same MAE; loss plateaued by ~2k steps |
| The model structurally **cannot** beat hold_state | targets are `command[t+k] − state[t]`; the ~0.74° command−state tracking offset lives only in command history, which is not an input |
| LR was ~8× below the authors' post-train recipe | we ran 5e-5 @ eff. 512; the shipped `scripts/train.sh` post-train runs **1e-4 @ eff. 128**, cosine→0, no warmup (paper Table 4: 1e-4 @ 384) |
| Action space diverges from the paper on the hands | paper §5.1 + `...deltabase_axis_eef...` JSON: arms = relative deltas, **fingers = absolute joint angles**. Our prep made all 65 dims deltas-from-state — the midtrain trunk never saw delta finger targets, and delta targets inherit tracking-error noise that q01/q99 normalization then amplifies |

Changes below, in dependency order. Items 1–2 are already implemented; 3–5 are the plan.

---

## 1. Inference-time smoothing — DONE (`T-Rex/scripts/eval_smoothed.py`)

No retraining; applies to the *current* checkpoint and every future one.

1. **Mean-of-K flow draws, batched** (`predict_draws`): embed the vision/language
   prefix once, `repeat_interleave` K×, one K-wide flow decode
   (`--max_flow_batch 64` bounds memory). Variance ∝ 1/K: K=4 removes ~75% of the
   removable MSE, K=8 measured 2.50°→1.64°. Averaging is done post-denormalize
   (affine, commutes), pre-frozen-clamp.
2. **ACT-style temporal ensembling** (`ensemble_stream`): receding-horizon rollout
   where every past chunk covering the current frame votes, weights `exp(−m·age)`
   (`--ensemble_m`, 0 = uniform). Full effect needs chunk overlap: run the rollout
   pass on a **stride-5** split (chunks overlap 25−stride steps; at stride 20 only
   the first 5 frames/replan get a second vote).
3. **Safety projection** (`SafetyProjector`): sequential clamp of the absolute
   command stream to the Shadow evaluator's own limits — positions within URDF ±2°
   tolerance (`--clamp_positions tol`), per-step travel ≤
   `0.999·min(jump limit, velocity/30 Hz)`. Violations → 0 by construction; MAE
   cost measured by the script (expected ~nothing).
4. **Deployment mode**: run `blind` (skip the tactile fast tick) until the cascade
   stops hurting; sweep `--cascaded_split_step` on the frozen checkpoint (free) —
   the paper says τ_split has a peaked optimum and must be re-tuned when chunk
   length changes (16→25).

Notebook: smoothing cells added to `trex_colab.ipynb` §4 (pilot) and §8 (full +
pilot-rollout rerun). First action item: **run these on checkpoint-2-8000** to get
the smoothed baseline before the retrain.

## 2. LR & schedule fix — decided (apply to the next run)

Source: paper Table 4 + the authors' own post-train launcher
(`T-Rex/scripts/train.sh`: LR 1e-4, eff. batch 128, `--min_lr_ratio 0`, warmup 0,
AdamW, wd 0, clip 1.0, 100 epochs; scheduler = linear warmup + single half-cosine
over total optimizer steps, `train.py:105`).

At our eff. batch 512 (`TRAIN_BSZ=128 GRAD_ACCUM=4`):

- **Peak LR 1.5e-4** (bounds: 1e-4 paper-faithful floor, 2e-4 sqrt-scaled from
  1e-4@128). Higher end is tolerable — the latent expert is mostly frozen.
- **`--min_lr_ratio 0`** (edit `train_origami.sh`; currently 0.05).
- **Keep `--warmup_rates 0.03`** — deliberate deviation from the paper's 0 warmup,
  because we cold-start 5 head tensors (`x_embedder`, `final_layer`,
  `final_layer_tactile`, `state_embedder`) for the 62→65 change; the authors don't.
- **`N_EPOCHS=3`** (~2,400 optimizer steps; preflight budgeted ~6 h). The authors
  post-train for 100 epochs — 2 epochs was short even before the LR issue.
- Unchanged (already paper-matched): AdamW(8bit), weight decay 0, grad clip 1.0, bf16.
- Resume: scheduler is checkpointed (`register_for_checkpointing`) — a resume must
  continue the cosine, not restart warmup (regression already fixed once; keep the test).

Launch: `LR=1.5e-4 N_EPOCHS=3 TRAIN_BSZ=128 GRAD_ACCUM=4 bash scripts/train_origami.sh`

## 3. Data prep changes (CPU-only, before the run)

### 3a. Hybrid action space — match the paper's per-group anchoring

Change `trex_origami/prepare.py` target construction:

- **Arms (dims 0–6, 29–35; 14 joints): keep deltas** — but re-anchor to the
  **previous command** `action_abs[t−1]` instead of `state[t]` (see 3b). Arms were
  delta-trained in the original too; this part transfers.
- **Hands (44 dims) + neck/torso motor (7 dims): absolute joint angles**, exactly
  as the original trained the midtrain checkpoint. Rationale: (i) the trunk's
  features were shaped for absolute finger poses; (ii) absolute targets are the
  clean teleop command, free of the ~0.7° tracking-error anchor noise that
  delta-from-state bakes into every sample; (iii) q01/q99 over absolute angles
  spans real poses, so residual flow noise denormalizes to far fewer radians than
  on near-static delta dims (the frozen `lower_body` passthrough pathology, milder,
  on every quiet finger).
- Record the per-dim anchoring in `meta/dataset.json` (e.g.
  `action_anchor: ["prev_command"×14 …, "absolute"×51]`) so loaders/eval/serving
  read it instead of hard-coding.

### 3b. Previous-command anchoring for the delta dims (causal-confusion aware)

- Target for arm dims: `action[t+k] − action_abs[t−1]` (row 0: fall back to
  `state[t]`, which equals the command at episode start). At deployment the anchor
  is the policy's own last emitted command — always available.
- Effect: the "predict zero" floor moves from hold_state (1.42°) to
  repeat_command-level (0.79°), and the unobservable tracking offset leaves the target.
- **de Haan et al. mitigations** (the anchor is a textbook nuisance shortcut, and
  our open-loop MAE metric *rewards* copying it):
  - anchor enters **only additively at the output** (retargeting), not as a network
    input the model can attend to;
  - train-time noise on the anchor using the existing `tracking_error` stats
    machinery (`te_mean/te_std`), plus anchor dropout (fall back to `state[t]`
    anchor for ~10–20% of samples) so the policy stays robust to its own
    deployment-time command drift (DAgger-style compounding);
  - **motion-only guard metric** (see §5): a policy that only copies the anchor
    must show zero motion skill there.

### 3c. Phase conditioning

Prep with `--phase_mode progress`: appends "(fold k of 6)" to the static
`north ces task` string. Code path already exists (`origami_dataset.py:_task_text`);
it is currently off. Deployment-legal (elapsed time is known). README's own top
untested lever.

### 3d. Stats & verification

- Refit `norm_stats.json` on the new targets (train-only, copied to val); the
  action ranks stay `[25, 65]`. The preflight stats-calibration check will hard-fail
  any stale-stats warm start — that is intended; the new run cold-starts from
  midtrain again.
- Extend `verify.py`: re-derive one episode's targets from raw per anchoring rule;
  assert absolute dims round-trip to the wire contract without adding state.

## 4. Training config for the one run

Everything from §2 plus:

- **Keep `--use_robot_state 1` with `--state_noise_mode joint`.** Note: the
  authors' post-train uses `use_robot_state 0` (vision+tactile only) — a deliberate
  starvation of the state shortcut. We keep state because our absolute-dim targets
  no longer need it for anchoring and the noise injection guards the shortcut; if
  the motion-only metric later shows state-copying, flipping to 0 is the documented
  fallback.
- **Tactile cascade**: leave training flags as-is (`cascaded_tactile_dropout 0.1`,
  `split 6/10`); the fix is measured at eval (τ_split sweep, blind-vs-cascaded on
  the contact split). If `tac` loss again tracks `act` with zero delay sensitivity,
  the tactile expert isn't getting gradient signal that matters — investigate before
  spending another run on it, don't assume.
- Unchanged: freeze latent expert except last 4 layers, FLARE loss 0.0 (frozen
  latent expert ⇒ only `flare_proj` would train), image 224², chunk 25 / dim 65,
  seed 42, `--save_optimizer_state 1`.

## 5. Eval & serving changes required by §3

- **Per-dim anchoring in every consumer**: `eval_origami.py`, `eval_smoothed.py`,
  `eval_diagnostics.py`, `eval_robustness.py`, `test.py` (serving) currently assume
  delta-from-state everywhere. Reconstruction becomes
  `abs = anchor_per_dim + pred` with anchor ∈ {prev command (arms), 0/absolute
  (hands, motor)}. Read the rule from `meta/dataset.json`.
- **Baselines stay defined in absolute space** so numbers remain comparable across
  preps: hold_state = command chunk ≡ `state[t]`; repeat_command = ≡ `action_abs[t]`.
- **New motion-only metric** (the anti-shortcut guard): MAE of
  `(pred_abs − anchor)` vs `(gt_abs − anchor)` per group + its horizon curve.
  Acceptance is judged on this, not raw MAE, after re-anchoring.
- Keep the smoothing stack (§1) in the deployment path: mean-of-K (K=4 default,
  latency-checked), temporal ensembling at the real replan rate, safety projection
  as the final stage before the wire.

## 6. Execution order & acceptance criteria

1. Run `eval_smoothed.py` on checkpoint-2-8000 → smoothed baseline numbers.
2. Re-prep `full` tier with §3 (hybrid anchoring, phase_mode progress, stats refit,
   verify gates). Also re-prep a stride-5 val slice for rollout/robustness evals.
3. Update eval/serving anchoring (§5) + smoke tests (`smoke_test_origami.py` must
   assert the new flow-matching identity per anchoring group).
4. One fine-tune: §2 + §4 (`LR=1.5e-4 N_EPOCHS=3`, eff. 512, cosine→0, warmup 3%).
5. Full eval ladder: eval_origami → diagnostics → robustness → smoothed; pick
   blind/cascaded + τ_split + K on val.

Bars to clear, in order (on the motion-aware metrics):
1. beat **hold_state** overall and on >50% of joints (was: 2/65);
2. beat **oracle_prev_command** (anchored 0.90°);
3. close toward **repeat_command**'s floor with nonzero motion skill;
4. executed-stream jerk within ~2× teleop (0.05°) after smoothing; safety
   violations 0 post-projection;
5. cascaded ≥ blind on the contact split (else ship blind).

## 7. Risks

| risk | mitigation |
|---|---|
| Anchor copying inflates MAE gains (causal confusion) | motion-only metric is the acceptance gate; anchor noise + dropout at train time |
| Own-command drift compounds at deployment | anchor noise from `te_std`; safety projection bounds excursions; closed-loop sanity in sim before submission |
| Absolute hand dims regress vs deltas | per_joint CSV comparison vs 2-8000 on the same val; hands are where 2-8000 was worst, so the bar is low |
| Stale stats / mixed anchoring silently mis-scales | preflight stats check (existing) + new verify.py round-trip gate |
| 1.5e-4 destabilizes fresh heads | 3% warmup, grad clip 1.0; watch first 200 steps' loss — fallback 1e-4 |
| Mean-of-K blows the tick budget | latency pass in eval_smoothed per (mode, K); drop to K=4 or K=2 |
