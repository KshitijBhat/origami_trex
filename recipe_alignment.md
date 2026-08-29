# Aligning origami_trex to T-Rex's weights + intended post-train recipe

Sources: `/home/sai/Desktop/ORI/T-Rex` (untouched, `git remote` verified `ZhuoyangLiu2005/T-Rex`),
`/home/sai/Desktop/ORI/origami_trex` (current), and the paper (arXiv 2606.17055v1).
Not implemented — reference for one-by-one implementation.

Tags: **[CHANGE]** deviates from reference, fixable · **[FORCED]** competition contract,
cannot change · **[OK]** already matches, no action needed.

## A. Vision input

| # | | reference (`T-Rex`) | origami_trex | |
|---|---|---|---|---|
| A1 | camera platform | 640×360, ZED head + 2 ZED wrist (paper App. G) | 480×480 square (different robot rig) | platform difference, not portable |
| A2 | **head crop** | `CROP_BOX_SLOW=(0,300,140,540)` applied before resize (`utils/gen_json_..._parallel.py:23`) | mechanism built (`--head_crop_box "top,bottom,left,right"`, `origami_dataset.py:_pil`, train-time, off by default), box still unmeasured — see A2 note below | **[CHANGE — mechanism landed, box TBD]** |
| A3 | resize target | 384×288 → ~108 tok/view (`scripts/test.sh:35`) | 224×224 → 64 tok/view (wire contract) | **[FORCED]**, ceiling not fixable |
| A4 | view order | slow=[head], fast=[wrist_r, wrist_l] (`gen_json...:209`) | slow=[head], fast=[wrist_right, wrist_left] (`qwen_vla/origami_dataset.py:450`) | **[OK]** |
| A5 | head_right | not fetched — "slow expert takes a single image" | not fetched (`trex_origami/seasons.py:31`) | **[OK]** |

A2 is the one real lever here: with only 64 tokens/view, cropping to the workspace before resizing (as reference does) spends the fixed token budget on the fold region instead of background. Can be done wire-side inside `TeamPolicy.infer()` and prep-side in `trex_origami/prepare.py`'s `decode_frames`, as long as both match.

A2 note: implemented as a **train-time** crop instead — `--head_crop_box "top,bottom,left,right"` (`scripts/train.py`, threaded through `train_origami.sh`'s `HEAD_CROP_BOX` env var), applied in `OrigamiDataset._pil` before the vision processor sees the head frame, no resize back up (the processor's own `smart_resize` upscales once, whatever size it's handed — resizing the crop ourselves first would just be a second, redundant lossy resample). Chosen over prep-time (`prepare.py`/the notebook's `APPLY_HEAD_CROP` hook, which also exists, off) because 224×224 is a hard ceiling on both train and serve (`robot_io_spec.md`: the live wire feed and the archival dataset are both direct, non-aspect-preserving squashes of the same native 1920×1536 sensor — no extra resolution sits in a not-yet-resized frame to lose by cropping post-hoc), so a train-time crop against the already-prepared data is free to iterate on and not a lesser version of doing it earlier. Off by default (empty string = full frame) until a real box is measured from actual frames — same guard as the notebook's `HEAD_CROP_BOX = None`. **Not yet mirrored in `TeamPolicy.infer()`** (`ori/aug31_submission_origami_trex_v1/trex_policy_server.py`) — required before this could ever be used at eval/competition time, since serving must apply the identical box to the live 224×224 frame.

## B. Action / state representation

| # | | reference | origami_trex | |
|---|---|---|---|---|
| B1 | **action semantics** | body-frame SE(3) delta (arms, 18-D) + **absolute** (hands, 44-D) — paper §5.1: *"relative end-effector delta control for the bimanual arms, and absolute joint control for the fingers"* | uniform elementwise delta, all 65 dims (`trex_origami/prepare.py:344`) | **[CHANGE]** — highest impact, see §3 |
| B2 | `use_robot_state` | `0` (`scripts/train.sh`, `test.sh` — confirmed both, not just train) | `1`, and genuinely wired into the forward pass (`train.py:656-676`: `state_embedder(state_vec)` concatenated into `full_embeds`), not a dead flag | **[CHANGE]** — real, not cosmetic; running both as separate experiments |
| B3 | `action_chunk` | 16 | 25 | **[CHANGE, not forced]** — corrected: `action_horizon` is participant-declared metadata in `[1, 1024]` (`participant_zenoh_submission.md:148`), not a kit-fixed value; 25 was just this team's choice. `--action-chunk` now env-overridable in `run_prepare_fast.sh`/`train_origami.sh` (`ACTION_CHUNK=16 ...`). Changing it needs a real re-prep (baked into data at prep time) + matching `TeamPolicy(action_chunk=...)`/declared `action_horizon` at serving |
| B4 | `action_dim` | 62 (18+44) | 65 | **[FORCED]** = kit's joint contract |
| B5 | Num Inference Timesteps | action=6, tactile=4 (paper Table 4) | `cascaded_split_step=6`, remaining=4 | **[OK]** — exact match |
| B6 | τ_split | 0.4 (paper Fig. 4, "an intermediate split achieves the best performance") | 0.4 | **[OK]** |

## C. Training recipe

| # | | reference | origami_trex (`train_origami.sh`) | |
|---|---|---|---|---|
| C1 | **latent expert freeze** | none — all 28 layers trainable (Table 4 lists full 1.41B latent expert under "Training Configurations", no freeze/LoRA mentioned anywhere in paper) | `--freeze_latent_expert 1 --train_latent_last_n 4` (4/28 trainable) | **[CHANGE]** — biggest lever |
| C2 | peak LR | 1×10⁻⁴ (Table 4) | 5×10⁻⁵ | **[CHANGE]** |
| C3 | LR warmup | **ratio 0** (Table 4) | `--warmup_rates 0.03` | **[CHANGE]**, minor |
| C4 | weight decay | 0 | 0 | **[OK]** |
| C5 | scheduler | cosine to min LR 0 | (check current script) | verify matches |
| C6 | optimizer / precision | AdamW / bf16 | adamw8bit / bf16 | close; 8-bit is a memory tradeoff, not a correctness issue |
| C7 | batch × GPUs | 16/device × 24 H100 (=384, but this is pretrain/midtrain scale — paper doesn't disclose a separate post-train batch) | 8×4×1=32, single GPU | **[CHANGE]** — scale with available GPUs, and *actually* scale LR when you do (`train_origami.sh` explicitly does not auto-scale) |
| C8 | epochs | not disclosed for post-train specifically (checked: absent from Table 4 and from every "our method" mention — only baselines get explicit epoch counts) | 2 (Colab-session-budget choice, not paper-derived) | **[CHANGE]** — `checkpoint-0-5` is `global_step=5`, barely started regardless of what the "right" number is |
| C9 | `flare_loss_weight` | 0.5 | 0.0 | **[CHANGE]** |
| C10 | resume point | midtrain checkpoint ("start here to fine-tune on your own task") | `--resume_source midtrain` | **[OK]** |
| C11 | LoRA / adapters | none found anywhere in paper — full fine-tune | full fine-tune (modulo C1) | **[OK]** once C1 is fixed |

## D. Data sampling / prep

| # | | reference | origami_trex | |
|---|---|---|---|---|
| D1 | frame rate | `FRAME_STRIDE=1` — every recorded frame (30 Hz) | `sample_stride` 5 (pilot, 6 Hz) / 20 (full, 1.5 Hz) | **[CHANGE]**, real compute/storage tradeoff — not free |
| D2 | tactile history | 16 frames @ native 30 Hz, feeds VQ-VAE | 16 @ native 30 Hz regardless of `sample_stride` (`prepare.py:32`) | **[OK]**, and not independently choosable either way — `F6Encoder.forward()`/`F6PerFingerEncoder.forward()` (`tactile_vqvae/models/encoder.py`) hard-`raise` if `T != self.window`, and `train.py:780/883` overwrites whatever `--vqvae_window` is passed with `vqvae_config["window"]` from the loaded (frozen) checkpoint. Window is a property of which VQ-VAE checkpoint you load, not a training flag; changing it means training a new VQ-VAE. `in_channels=30` (5 fingers × 6 F6 dims) is baked into the same encoder's first `Conv1d` even more rigidly — a different finger count or per-finger dim can't even load the pretrained weights, shape mismatch, not just a runtime assert |
| D3 | deform encoding | video codec only | video codec **+ JPEG q4** on top (`prepare.py` `deform_quality=4`) | unverified hypothesis — cheap to test (re-prep a slice at q2, diff) |
| D4 | stats mask | hardcoded `[True]*dim` (`lerobot_common.py`) | derived from spread, catches frozen dims (`trex_origami/stats.py:53`) | **[OK]** — origami's is strictly better here |

## E. Not fixable — platform/contract differences, don't spend time here

65-D action / 65 unique joint names, no EEF pose in the dataset (no IK path available), 224×224 resolution ceiling, 480×480 square camera vs reference's 640×360 4:3 rig. These are the actual task, not porting bugs.

## Priority order for implementation

1. **C1** (unfreeze latent expert) — single biggest lever, free on multi-GPU.
2. **C8 + C7 + C2** together (epochs, batch, LR) — these three are coupled; scaling batch without epochs/LR wastes the extra GPUs.
3. **B1** (action representation) — highest-impact but touches `prepare.py`'s `build_episode_rows`, `stats.py`, and `TeamPolicy.infer()`'s un-mapping. Do after C1/C7/C8 land, so the comparison isn't confounded by two changes landing on the same eval.
4. **C9** (flare weight) — one-line change, do anytime.
5. **A2** (head crop) — cheap, needs matching prep + serving change.
6. **C3** (warmup) — trivial, low expected impact.
7. **D1** (denser sampling) — only after 1-4 show the recipe itself is sound; it's the most expensive to test (full re-prep).
8. **D3** (deform quality) — diagnostic only, do if tactile still looks dead after B1.

Run `eval_diagnostics.py`'s bias/variance split before and after each change — it's the only way to tell "genuinely learned more" from "sampling noise moved."

## F. Target spec: what one training example should contain, if B1 is adopted

```
observation:
  head, wrist_left, wrist_right   — JPEG, 224x224 RGB (crop-then-resize if A2 adopted)
  deform                          — JPEG, 1200x480 (2x5 grid of 240x240)
  state                           — [65] float32, absolute radians
  tacf6_hist                      — [16,10,6] float32, native 30Hz window

target (per horizon step k=0..24), split by dim group:
  arm dims   (left_arm, right_arm, motor: 0-7, 29-36, 58-65)
      -> delta from state[t]:  action[t+k] - state[t]
      -> normalized per-(step,dim) q01/q99, mask derived (catches lower_body_joint_1/2)
  hand dims  (left_hand, right_hand: 7-29, 36-58)
      -> absolute:             action[t+k]
      -> normalized per-(step,dim) q01/q99 of the ABSOLUTE distribution
         (a new stats block, separate from the arm-delta one — do not reuse
         action_min/max across both halves, they are different quantities now)

at inference (TeamPolicy.infer / eval_origami.py's predict()):
  arm/motor dims  -> denormalize, then + state[t]   (unchanged from today)
  hand dims       -> denormalize directly, no + state
```

This is exactly reference's split (18 delta + 44 absolute), applied to origami's 65-dim
layout by joint group instead of by arm/hand index ranges. The stats file (`stats_data.json`)
gains a second per-(step,dim) block (`action_abs_hands` or similar) rather than replacing
`action` outright, so `_clamp_frozen`/frozen-dim detection keeps working unmodified on the
arm-delta half.
