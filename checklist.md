# T-Rex → origami-inference-kit integration checklist

Open items to confirm before/while implementing `TeamPolicy` in
`policy_server_template.py`, plus broader training/architecture items worth
tracking. Nothing here is implemented yet — tracking only.

1. [ ] **Confirm IK or not, in the configs.** The original T-Rex hardware config
   (`hardware_code/config/default.yaml`) uses a delta-EEF-pose + differential-IK
   control mode. For now, **assume whatever trex_origami checkpoint we have does
   NOT need this** — `train_origami.sh` sets `ACTION_DIM=65` with direct
   absolute-joint-radian output, and the origami dataset has no EEF poses
   (`observation.state.tcp` is all zero), so IK should be irrelevant to this
   checkpoint. Revisit and confirm directly against the checkpoint's actual
   config/training_args before relying on this.

2. [ ] **Flare loss weight was never actually fine-tuned on this checkpoint.**
   `train_origami.sh` sets `--flare_loss_weight "${FLARE_LOSS_WEIGHT:-0.0}"` —
   zero by default. The 32 flare query tokens exist and participate in joint
   attention, but `flare_proj`'s future-frame-prediction head was never pushed
   toward anything meaningful during origami fine-tuning. Don't read anything
   into flare-token attention/behavior on this checkpoint as if it were a
   trained signal — it's architecture only, not learned. Revisit if a future
   run actually sets `FLARE_LOSS_WEIGHT` > 0.

3. [ ] **Implement stage-aware prompting — needs real fold-boundary labels first.**
   `phase_mode="progress"` already exists (`origami_dataset.py:329`,
   `f"(fold {k} of {n})"`) but `phase = offset / last` (`prepare.py:352`) is
   pure linear time-position within the episode — it assumes all 6 folds take
   equal wall-clock time, which is very likely false. Enabling the flag as-is
   would inject a confidently-wrong stage claim into the prompt some fraction
   of the time, worse than the current generic prompt. Blocked on: labeling
   real fold-transition timestamps in the dataset (manual or semi-automated)
   before `phase` means anything. Generic descriptive prompt (replacing
   `INSTRUCTION = "north ces task"` in `trex_origami/seasons.py:24` with real
   natural language, e.g. "fold the paper into an origami airplane") can land
   now, independently — it's a `meta/dataset.json` field edit, not a re-prep,
   since `_task_text` reads `ep.get("instruction")` first (`origami_dataset.py:325`).

4. [ ] **PLANNED for 2026-08-30.** **Implement history/state-aware attention — genuine architectural gap, not a flag.**
   Confirmed neither T-Rex nor origami_trex uses image history anywhere:
   `origami_dataset.py.__getitem__` reads exactly one frame per camera per row,
   and upstream's own `gen_json_tac_deltabase_eef_bimanual_parallel.py` does
   the same (single-index frame extraction). The two things that look like
   temporal context aren't: FLARE tokens are *future* frames used only as an
   auxiliary loss target (see item 2 — currently unused anyway), and the
   cached slow-tick KV reused across fast ticks is compute-reuse of the *same*
   single frame's embedding, not multiple distinct past frames. With no visual
   motion signal and (per item 2 / the robustness sweep) no working tactile
   signal either, the model has no mechanism to sense ongoing dynamic events
   (slip, settling) mid-chunk — only the current static frame + state. Needs
   new model code (a temporal cross-attention module over N past frames,
   analogous to vitacformer++'s own `image_history` design from earlier this
   project), not a config change.

   Motivating discussion (2026-08-29): the single-frame-only observation means
   the model can't tell "which fold stage am I at" from anything but the
   current frame's visual appearance — no history, and neither `observation/state`
   (robot joint pose, not paper state) nor `phase` (linear time position, not a
   real fold-boundary label, see item 3) substitutes. This gets worse, not
   independent, if the head crop (`--head_crop_box`, see `recipe_alignment.md`
   A2) removes part of the paper's visible extent — worth rechecking the crop
   against early-episode (pre-fold, largest paper footprint) frames specifically,
   not just the hand/arm-spread frames already checked, before this lands.
