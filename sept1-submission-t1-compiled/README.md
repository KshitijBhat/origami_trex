# sept1-submission-t1-compiled

Everything needed to reproduce and verify the `orvizkar/origami-policy-t1:sept1-compiled`
Docker Hub image, minus the two large binary artifacts that are already baked
into the image itself (listed below) rather than duplicated here.

## What's in this folder

- `Dockerfile`, `.dockerignore`, `entrypoint.sh` -- the build recipe.
- `trex_policy_server.py` -- the Zenoh policy server. Includes this
  submission's logging additions over `aug31_submission_origami_trex_v1`:
  episode-reset events, `request_id`-correlated metadata/reset queries,
  per-tick GPU memory (`mem_alloc_mb`/`mem_reserved_mb`), and an
  optional-observation-key inventory (`tactile_raw` presence) logged once
  per episode refresh.
- `compile_warmup.py` -- generates the baked `torch.compile` cache (see
  `compile_cache_recipe.md` for the exact regeneration steps).
- `requirements-inference.lock` -- pinned inference-only Python deps.
- `qwen_vla/`, `tactile_vqvae/` -- vendored T-Rex source, traced from
  `trex_policy_server.py`'s imports (see `scripts/sync_submission_files.sh`
  for exactly which files and why).
- `licenses/` -- third-party license notices.
- `checkpoint_meta/` -- the checkpoint's small config/stats/tokenizer files
  (`config.json`, `training_args.json`, `stats_data.json`,
  `training_state.json`, `processor/`). **Not included:** `model.pt`
  (8,514,017,322 bytes) -- baked into the image at `/app/checkpoints/model/model.pt`,
  omitted here on disk-space grounds (local disk had ~5.6GB free at build
  time). The checkpoint used is epoch-2 (`checkpoint-2-8000`); see
  `checkpoint_meta/training_state.json` for the exact step count.
- `logs/` -- see below.
- `compile_cache_recipe.md`, `real_episode_validator_plan.md` -- design docs
  for the compile-cache bake and the real-episode Zenoh validator.

## `logs/`

- `docker_build.log` -- full build output for the `sept1-compiled` image
  (built on the RTX 4090 host, `64.64.224.51`, from this exact folder's
  contents plus the checkpoint/compile-cache noted above).
- `container_startup.log` -- `docker logs origami-contract-policy` from
  process start through `READY`, captured after recreating the container
  from the `sept1-compiled` image (model load, `torch.compile` warmup,
  the two baked warmup patterns, `READY`).
- `real_episode_validation/` -- one full run of
  `check_zenoh_policy_real_episodes.py` (1 train-slot + 1 val-slot episode,
  10 frames each, against real recorded teleop episodes from the
  `competition-paper-set` HF revision) against this exact running container:
  `real_episode_metrics.json` (PASS/FAIL summary + per-frame MAE/MSE in
  radians), `validator.log` (the validator's own stdout), `container.log`
  (the full container log for that container's lifetime, captured by the
  validator itself).

## Image

Built and validated on the RTX 4090 host (`64.64.224.51`) from this folder's
contents. Hardening profile the container is run with: `--read-only` rootfs,
`--cap-drop ALL`, `--security-opt no-new-privileges=true`, tmpfs-only
`/run` and `/tmp`, `--pids-limit 512`, GPU passthrough via `--gpus all`.
Note: the container's rootfs is read-only, so code changes must go through a
rebuild + recreate -- `docker cp` into the running container silently does
not persist (confirmed the hard way: a `trex_policy_server.py` patch applied
via `docker cp` + restart did not survive, only a full image rebuild did).

Pushed to Docker Hub as `orvizkar/origami-policy-t1:sept1-compiled`
(digest `sha256:2516298c4e013fc2fdb182ae075d66bce4d0b514eb6d15a18d8615454be9720b`,
image ID `sha256:69ee0390e9bc34a309556805140e4dded0bb58445fbbbcb25b7064a01405f43c`).
See `manifest.json` for the full submission record.

**Not yet final**: `licenses/THIRD_PARTY_LICENSES.txt` is an interim
placeholder carried over from an earlier submission. Once proper license
notices are written, the image must be rebuilt (new `image_id`/digest) and
re-pushed before this is submission-ready -- update `manifest.json`
accordingly at that point.
