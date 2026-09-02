# sept2-submission-ckpt7000-compiled

Build context and runbook for the `checkpoint-2-7000` submission image
(`orvizkar/origami-policy-t1:sept2-ckpt7000-compiled`), built on the RTX 4090
host. Start with
`docker_image_generation_and_validation_compiled_ckpt7000.md`.

## Contents

- `Dockerfile`, `.dockerignore`, `entrypoint.sh` -- the build recipe. Inference
  settings are `ENV TREX_*` in the Dockerfile (5/3 steps, K=8, `state_offset`
  anchor, `tol` safety, compile on) and are part of the compile-cache key.
- `serve_origami_zenoh.py` -- Zenoh policy server (kit template + the T-Rex
  adapter). Vendored from `T-Rex/scripts/`.
- `trex_origami/` -- the deployment adapter (`policy.py`), checkpoint loader
  (`loading.py`), anchoring rule, joint contract, and the URDF-derived
  `joint_limits.json`. Vendored subset of `T-Rex/trex_origami/`.
- `qwen_vla/`, `tactile_vqvae/` -- model code, vendored subset.
- `compile_warmup.py` -- build-time cache generation.
- `requirements-inference.lock` -- pinned inference-only Python deps
  (torch/torchvision come from the Dockerfile's index pin).
- `scripts/sync_submission_files.sh` -- re-vendors the three source trees from
  `/workspace/origami_trex/T-Rex` (`--check` to verify without writing).
- `checkpoint_meta/` -- the checkpoint's small files for reference
  (`training_state.json`: epoch 2, global_step 7000). `model.pt` (8.5 GB) is
  not duplicated here; it is copied into the build context as
  `checkpoints/model/model.pt` on the build host.
- `licenses/THIRD_PARTY_LICENSES.txt` -- **interim placeholder carried over
  from sept1**; proper notices still need writing before the final image.
- `compile-cache/` -- empty here; populated on the build host (runbook §2).
- `logs/` -- put `docker_build.log`, container start-up log, validator and
  replay outputs here after the build.

## Not in the image

`scripts/replay_origami_real.py`, `trex_origami/lerobot_v3.py`,
`trex_origami/replay_sources.py`, `scripts/eval_deploy.py` (evaluation /
validation tooling, run from the T-Rex checkout against the container), the
URDF asset package, the participant kit.
