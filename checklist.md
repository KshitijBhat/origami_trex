# T-Rex → origami-inference-kit integration checklist

Open items to confirm before/while implementing `TeamPolicy` in
`policy_server_template.py`. Nothing here is implemented yet — tracking only.

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
