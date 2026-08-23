# T-Rex: Key Implementation Details for Fine-Tuning on New Data

Source: *T-Rex: Tactile-Reactive Dexterous Manipulation* (arXiv:2606.17055v1)

This document distills the paper's architecture, data, and training-recipe details that matter most
if you want to **post-train / fine-tune T-Rex on a new task or new dataset**, rather than train it
from scratch.

---

## 1. What You're Fine-Tuning: Model Overview

T-Rex is a **Mixture-of-Transformer-Experts (MoT)** with three experts sharing one backbone via
multimodal shared attention:

| Expert | Role | Frequency | Backbone | Params |
|---|---|---|---|---|
| **Latent Expert** | Predicts future visual latents from RGB + language (temporal grounding) | — | Qwen3VL-2B | 1.41B |
| **Action Expert** | Low-frequency flow-matching action denoising (coarse plan) | ~5 Hz | Qwen3VL-2B | 1.41B |
| **Tactile Expert** | High-frequency residual refinement using tactile tokens, reuses cached KV from action expert | ~20 Hz | Lightweight FFN (1536 intermediate size) | 0.62B |

Action representation: **62-dim action**, **chunk length Ta = 16**, flow matching (not diffusion),
conditional vector field `v_θ(x_τ, τ | c_t)`.

**Key idea for fine-tuning:** the tactile expert only ever denoises the *tail* of the flow-matching
trajectory (τ ∈ [0, τ_split]) and never touches raw vision — it reuses a cached KV context from the
action expert. This is what makes the tactile stream cheap and asynchronous at inference. If you
fine-tune on new tasks/objects, you are mostly adapting the **action expert + tactile expert jointly**
on task-specific demos, not retraining the latent/vision backbone from scratch.

---

## 2. The Three-Stage Training Recipe (relevant for fine-tuning strategy)

1. **Large-scale human egocentric pre-training** (22,889 h of human video) — trains latent + action
   experts only, no tactile expert. Provides broad visuomotor priors. *(You likely start from this
   checkpoint / the T-Rex mid-trained checkpoint rather than redoing this stage.)*
2. **Tactile-grounded robot mid-training** (100 h T-Rex Dataset, 22 motor primitives, 200+ objects) —
   adapts the action expert to robot embodiment/actions and trains the tactile expert on high-frequency
   denoising. This is the stage that gives the model its "zero-shot" contact-rich competence.
3. **Skill-specific post-training (= fine-tuning)** — **~100 task-specific demonstrations**, fine-tunes
   the full model on the new task while *preserving* tactile-reactive behavior learned in stage 2.

**Practical implication:** Skipping stage 2 (mid-training) and fine-tuning directly on ~100 task demos
from a pretrained-only checkpoint gives much worse results (see ablation, Table 3): average success
drops from 65% (full recipe) to 45% (pretrain + no mid-training) to 34% (no pretrain, mid-training only)
to 18% (neither). **Always fine-tune from the tactile-grounded mid-trained checkpoint, not from the
raw human-video-pretrained checkpoint.**

Data efficiency (Fig. 5): with mid-training, ~10-20 task demos already reach performance that
without-mid-training baselines need 100-200 demos to reach. So **new-task fine-tuning can use as few
as 10-50 demonstrations** if starting from the mid-trained checkpoint.

---

## 3. Data Requirements for New-Task Fine-Tuning

Each demonstration episode must be a **time-aligned bundle** at 30 Hz containing:
- 3x monocular RGB streams (1 head camera + 2 wrist cameras), 640×360
- Bimanual proprioception (arm joint pos/vel + hand joint states)
- SE(3) end-effector poses (both wrists)
- **Per-fingertip tactile: deformation depth map (1-channel) + 6-axis net wrench**, per finger
- Natural language instruction (single imperative sentence)

If your new task uses a different robot/gripper, note the paper's action space:
- Bimanual arms: **relative end-effector delta control**
- Fingers: **absolute joint control**
- 22-DoF dexterous hands per side in their setup

**Data recipe advice from the paper (Section 3):** rather than collecting narrow task-specific demos,
structure data around **verb-noun (motor-primitive × object) combinations** — this is what let 100h of
data generalize broadly. If you're building your own mid-training-style dataset, prioritize diverse
elementary motor primitives (grasp, twist, insert, wipe, squeeze, etc.) over exhaustive per-task
coverage — this was more data-efficient than an equal-sized task-specific dataset (Fig. 6).

If actually just fine-tuning (post-training) on one target task: **~100 demonstrations** with
randomized object pose/position and scene distractors (0-5 distractor objects) is the paper's regime.

---

## 4. Core Training Hyperparameters (from Table 4)

```
Optimizer:            AdamW
Peak LR:              1e-4
Min LR:               0
LR schedule:          Cosine with min LR
Weight decay:         0
Warmup ratio:         0
Gradient clipping:    1.0
Precision:            bf16
Deepspeed:            ZeRO Stage 1
Per-device batch size:16
Grad accumulation:    1
GPUs:                 24x H100 (scale down proportionally; adjust LR/batch accordingly)
```

Flow-matching specifics:
```
Action Expert:
  Training timestep sampling: Beta(1.5, 1.0) over (0, 1]
  Num inference timesteps:    6 (paper text says N=10 total Euler steps, split Kslow=6 / Kfast=4)
Tactile Expert:
  Training timestep sampling: τ_tac = τ_split * τ̃, τ̃ ~ Beta(1.5, 1.0) over (0, τ_split]
  Num inference timesteps:    4
Split point:                  τ_split = 0.4 (empirically optimal — see Fig. 4 ablation)
Loss weights:                 λ_tac = 1.0, λ_future = 0.5
```

**If you fine-tune with a different action chunk length or robot, you'll need to re-tune τ_split** —
the paper shows performance is a peaked function of the split step (Kslow ∈ {1,2,4,6,8}), with too
small a split under-informing the tactile expert of visuomotor priors, and too large a split leaving
the tactile expert too little capacity to incorporate tactile feedback.

---

## 5. Loss Function (what to reproduce if reimplementing the training loop)

Both experts regress the **same flow-matching target** but see different context and different
τ sub-intervals:

```
x_τ = (1-τ) * A_demo + τ * ε,      v* = ε - A_demo

L_act = E[ || f_act_θ(x_τact, τ_act; c_vl) - v* ||^2 ]              # full (0,1], vision+language context
L_tac = E[ || f_tac_θ(x_τtac, τ_tac; c_tac, KV_τsplit) - v* ||^2 ]  # (0, τ_split], tactile context + detached KV cache
L_future = ...  # auxiliary future-visual-latent prediction loss (latent expert)

L_total = L_act + λ_tac * L_tac + λ_future * L_future
```

Important detail: **KV_τsplit is extracted from a detached (no_grad) slow-stream pass** — this prevents
gradients from the tactile loss flowing back into the action expert's forward pass, keeping the two
experts' training somewhat decoupled while sharing the target.

**Delay augmentation:** during training, sample a discrete offset δ ~ Uniform{0, 4, 8, 12} to
desynchronize the tactile stream's frame index relative to the vision/language stream. This matches
the real inference-time staleness (tactile stream runs async at higher frequency than vision) and is
necessary — without it the tactile expert overfits to perfectly synced modalities and won't transfer
to the actual asynchronous deployment loop.

---

## 6. Tactile Encoder Details (needed if adapting to a different tactile sensor)

Two parallel tactile pathways, concatenated into token `z_t^τ`:

1. **Per-finger VQ-VAE force encoder**
   - Input: 6D force vector history over T=16 frames per fingertip
   - 1D temporal CNN, 2 strided downsampling blocks → temporal mean pool → 256-d embedding
   - Vector-quantized against a codebook of size K=64, updated via EMA with re-seeding of
     under-used codes (prevents codebook collapse)
   - Loss: magnitude-weighted MSE (penalizes reconstruction error more during high-force contact)
   - **Convolutional weights shared across all 5 fingers**, with per-finger identity embeddings
     injected before encoding (parameter efficient, scales across digits)

2. **Deformation map encoder**
   - ResNet-18 backbone, modified stem for single-channel input, only first 3 residual stages kept
   - Each stage followed by a 3×3 conv projecting to 128 channels, flattened + linearly projected
   - **Pretrained via self-supervised convolutional autoencoder, then frozen during policy training**
     (this encoder is NOT fine-tuned jointly with the policy)

If your new tactile sensor has a different modality (e.g., GelSight images instead of force+deformation),
you will need to retrain/replace this encoder — the paper's own baselines (Tactile-VLA) show that
naively swapping tactile representations without adapting the encoder degrades performance.

---

## 7. Inference / Deployment Details Relevant to Fine-Tuned Model Serving

- Asynchronous two-thread design: slow stream (action expert, ~5 Hz) computes an intermediate
  denoised state and caches `KV_τsplit`; fast stream (tactile expert, ~20 Hz) is triggered at chunk
  offsets `{0, 4, 8, 12}` within each 16-step action chunk, reusing the cached KV and never
  re-running the vision tower.
- Thread safety: single-threaded request socket + execution lock — the fast tick cannot start until
  the slow tick has fully committed its cache.
- Low-level robot control loop runs independently at 300 Hz, consuming the async-updated action chunks.
- This architecture is what gives the large compute savings — **if fine-tuning changes the tactile
  expert's size/latency, re-validate the ~4x frequency ratio (5 Hz action / 20 Hz tactile) still holds
  for your control loop.**

---

## 8. Practical Checklist for Fine-Tuning on a New Task

1. Start from the **tactile-grounded mid-trained checkpoint** (not human-video-pretrained-only).
2. Collect ~10-100 teleoperated demos of the new task with synchronized RGB + tactile (force +
   deformation) + proprioception + language instruction, at 30 Hz.
3. Include object pose/position randomization and a handful of scene distractors to avoid overfitting
   to a fixed layout.
4. Fine-tune both action expert and tactile expert jointly using the cascaded flow-matching loss
   (Section 4.2 / Eq. 6-7), keeping delay augmentation (δ ~ Uniform{0,4,8,12}) active.
5. Keep τ_split = 0.4 as a starting point; sweep if your action chunk length or control frequencies
   differ substantially from the paper's (Ta=16, 5 Hz / 20 Hz).
6. Do NOT fine-tune the deformation-map ResNet encoder — keep it frozen (it's meant to be a stable,
   pretrained geometric feature extractor).
7. Use AdamW, LR 1e-4 cosine decay, bf16, ZeRO-1, as a default starting hyperparameter set; scale
   batch size/LR to your available GPU count.
8. Validate with the paper's evaluation style: progress-based or additive sub-step rubrics over
   ~16 trials with randomized object placement, rather than single binary success/fail.

---

## 9. Known Limitations to Keep in Mind

- Long-horizon tasks with tight tolerances are still hard for pure imitation learning (paper suggests
  RL / online refinement as future work).
- Model is bottlenecked by tactile hardware quality: sensor distortion, calibration drift, and lack of
  whole-palm sensing (only fingertip sensors). Fine-tuning cannot fix upstream sensor noise.
- Common failure modes observed (Appendix H): object collision from imprecise visual alignment,
  slipping during fine in-hand manipulation, imprecise placement (BC distribution shift), multi-finger
  unintended contact, excessive force on deformable objects, and sliding misalignment in tasks
  requiring fine temporal tactile conditioning. These are useful categories to check against when
  evaluating your fine-tuned model's failures.