# Docker image generation and validation, compiled policy

End-to-end runbook for what was actually done to go from checkpoint + code
to a validated, pushed, archived submission on the RTX 4090 host
(`64.64.224.51`, build context `/root/aug31_submission_origami_trex_v1`,
image `orvizkar/origami-policy-t1:sept1-compiled`). Every command and number
below is what actually ran today, not a idealized/projected version of it.
See `compile_cache_recipe.md` for the compile-cache-specific deep dive this
doc summarizes inline (step 2).

## 0. Host prerequisites (one-time, per fresh host)

- **GPU passthrough**: `nvidia-container-toolkit` must actually be
  installed, not just referenced in `/etc/docker/daemon.json`. If
  `docker run --gpus all ...` fails with `could not select device driver ""
  with capabilities: [[gpu]]`, install it from NVIDIA's apt repo, then
  `nvidia-ctk runtime configure --runtime=docker` and
  `systemctl restart docker` (this restarts the whole daemon -- anything
  else running in Docker, e.g. `origami-contract-router`, goes down and
  needs a manual restart afterward).
- **`uv`** for the SDK's Python environment (the validator scripts'
  dependencies) -- do not ad-hoc `pip install` into system Python:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  cd sharpa_north_ces_lite_sdk-main
  uv sync --frozen --no-install-project
  uv pip install lerobot   # not in the lock file, needed for real_observation_source.py
  ```

## 1. Get the checkpoint into the build context

`checkpoints/model/` needs: `model.pt`, `config.json`, `training_args.json`,
`stats_data.json`, `training_state.json`, `processor/` (9 files). Do **not**
include the `state/` subdirectory some checkpoint zips ship with
(`model.safetensors` duplicate + `optimizer.bin` + `random_states_0.pkl`) --
that's training-resume state, not needed for inference, and it roughly
doubles the size for nothing (~13GB wasted on this checkpoint).

## 2. Build the image

Two paths, depending on whether the compile-cache needs regenerating (full
detail in `compile_cache_recipe.md`):

**Cheap path** (only `trex_policy_server.py` orchestration code changed,
checkpoint/model code/GPU architecture unchanged -- this is what actually
produced `sept1-compiled` from `aug31-compiled` today):
```bash
cd /root/aug31_submission_origami_trex_v1
docker build -t orvizkar/origami-policy-t1:sept1-compiled .
```
BuildKit reuses the existing `compile-cache/` and `checkpoints/model/`
layers by content hash -- the 8.5GB checkpoint COPY still only takes ~20-25s
even on a "changed" build, because its bytes didn't actually change.

**Full regeneration path** (new checkpoint, new GPU architecture, or
`qwen_vla`/`tactile_vqvae` model code changed) -- see
`compile_cache_recipe.md`'s 6-step loop: build with an empty cache, run a
writable throwaway container, run `compile_warmup.py` inside it to populate
the cache, `docker cp` the cache out, rebuild the final image with the now-
populated cache baked in.

## 3. Recreate the hardened container

**The running container's rootfs is read-only** (`--read-only`, part of the
hardening profile below). This has one sharp edge: `docker cp`-patching a
file into a running instance does **not** persist -- confirmed directly
today (copied an updated `trex_policy_server.py` in, `docker exec ... grep`
afterward showed the old code still there). Any code change requires a
full image rebuild (step 2) and then removing + recreating the container,
never a live patch.

```bash
docker rm -f origami-contract-policy 2>/dev/null
docker run -d --name origami-contract-policy \
  --network origami-contract-test --gpus all \
  --read-only --cap-drop ALL --security-opt no-new-privileges=true \
  --tmpfs /run:rw,noexec,nosuid,nodev,size=64m \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=4g \
  --shm-size 8g --memory 34359738368 --memory-swap 68719476736 --cpus 8 \
  --pids-limit 512 --ipc private --cgroupns private \
  -e ORIGAMI_ZENOH_ENDPOINT=tcp/origami-contract-router:7447 \
  -e ORIGAMI_SESSION_ID=test-session \
  orvizkar/origami-policy-t1:sept1-compiled
```

The container's non-root user (`65532:65532`) comes from the Dockerfile's
`USER` instruction, not a run flag. Recreating the container from an
**unchanged** image (no rebuild) is also a legitimate, separate action --
did this today purely to get a clean process for a validation run, since
the previous long-running instance already had tick counters/cached state
from earlier requests.

### Expected startup sequence (from real `container.log`)

```
startup flags ...
model.pt load_state_dict missing=0 unexpected=0      # ~35-45s after start
model_load checkpoint=... use_robot_state=1 ...      # ~10s later
TeamPolicy ready device=cuda ...
warmup starting -- compiling tactile_flow_continue and forward_flow_action_partial ...
policy reset episode_ticks_completed=0
infer tick=0 observation_optional_keys=[]
[Graph break from Tensor.item() warning -- see "expected warnings" below]
slow_pass tick=0 vision_ms=... slow_ms=...            # the expensive first-touch compile
fast_pass tick=0 fast_ms=... actions_shape=(25, 65)
infer tick=0 refresh=True total_ms=... mem_alloc_mb=... mem_reserved_mb=...
[repeats once more for the zero-tactile warmup pattern, much faster -- cache hit]
warmup complete elapsed_ms=...
READY transport=origami-zenoh-v1 endpoint=... horizon=25
```

### Expected warnings (benign, not errors)

```
W... torch/_dynamo/variables/tensor.py:1379] Graph break from `Tensor.item()`, ...
    File "/app/qwen_vla/modeling_vla.py", line 767, in forward_flow_action_partial
        tau_split = float(time.item())
```
`tau_split = float(time.item())` converts the flow-matching timestep tensor
to a plain Python float for the next stage's `tau_split: float` argument.
Dynamo can't keep a `.item()` sync inside one symbolic graph, so it splits
the trace there and warns about it. This is a one-time cost per fresh
process (not per inference tick) and is already priced into the
`slow_pass`/warmup timings below -- it is not a bug and does not indicate a
problem.

### Timing, by host (measured, not estimated)

| | RTX 5090 (`aug31-compiled`) | RTX 4090 (`sept1-compiled`, this host) |
|---|---|---|
| First-touch compile (`slow_pass`, fixed pattern) | ~65-160s | ~157-159s |
| Total `warmup complete elapsed_ms` | ~160s | ~197-200s |
| Steady-state `fast_pass` latency | ~30ms median | ~85-90ms |

RTX 4090 is consistently ~3x slower for steady-state inference and somewhat
slower for the one-time compile -- weaker hardware, not a regression.
Re-measure on any new host rather than trusting these as universal.

## 4. Validate

Two complementary checks -- run both, they cover different things:

```bash
# (a) Synthetic protocol-conformance check -- no real data, fast, catches
#     wire-format/schema bugs
python3 examples/check_zenoh_policy.py \
  --endpoint tcp/127.0.0.1:17447 --session-id test-session \
  --timeout 30 --requests 10 --expected-horizon 25

# (b) Real recorded episodes replayed through the actual running container,
#     scored against real ground-truth teleop actions
cd /root/sharpa_north_ces_lite_sdk-main
uv run --no-sync python examples/check_zenoh_policy_real_episodes.py \
  --endpoint tcp/127.0.0.1:17447 --session-id test-session --timeout 30 \
  --manifest /root/container_check_manifest.json \
  --data-root /root/competition_paper_set_data \
  --train-episodes 1 --val-episodes 1 \
  --out-dir /root/container_validations
```

(b) auto-creates a timestamped subfolder under `--out-dir` and writes, with
no extra wiring needed:
- `real_episode_metrics.json` -- PASS/FAIL summary, per-frame MAE/MSE (rad)
- `validator.log` -- the validator's own stdout, mirrored to this file
- `container.log` -- the policy container's **entire** lifetime log (not
  windowed to just this run -- deliberately, so startup/model-load/warmup
  context is always in the same file as the run's request/reply lines),
  captured via a plain `docker logs <container>` at the end of the run

Expected result on this checkpoint (epoch 2, `checkpoint-2-8000`): PASS on
both `train_slot` and `val_slot`, MAE around 0.06-0.10 rad, MSE around
0.007-0.04, decreasing over the 10 frames of a slot as the fast-path KV
cache "locks in" on the episode (first frame of each slot is the expensive
`slow_pass` + highest error; subsequent frames are `fast_pass`-only and
error trends down).

## 5. Generate the `.tar.zst` archive

```bash
mkdir -p /root/sept1-submission
cd /root/sept1-submission
docker save orvizkar/origami-policy-t1:sept1-compiled \
  | zstd -T0 -3 -o orvizkar-origami-policy-t1-sept1-compiled.tar.zst.partial
mv orvizkar-origami-policy-t1-sept1-compiled.tar.zst.partial \
   orvizkar-origami-policy-t1-sept1-compiled.tar.zst

zstd -t orvizkar-origami-policy-t1-sept1-compiled.tar.zst   # integrity check
sha256sum orvizkar-origami-policy-t1-sept1-compiled.tar.zst \
  | tee orvizkar-origami-policy-t1-sept1-compiled.tar.zst.sha256
```
`docker save`, not `docker export` -- `export` discards image config,
Entrypoint, and layer metadata. Today's result: 21GB image -> 13.1GB
`.tar.zst` (`-3`, multi-threaded via `-T0`); took a few minutes on this
host. Write to a `.partial` name and `mv` at the end so a truncated/failed
run never looks like a complete archive.

## 6. Push to Docker Hub

```bash
docker login -u <username>   # interactive -- enter password/token at the prompt
docker push orvizkar/origami-policy-t1:sept1-compiled
```
Same repo (`orvizkar/origami-policy-t1`) across tags -- pushing a new tag
does not create a new repo. Most base-image/dependency layers were already
present in the registry (either from a prior push, or Docker Hub's
cross-repo blob matching for widely-used public base layers), so only the
checkpoint layer (~8.5GB) and the compile-cache layer (~132MB) actually
transferred; everything else showed `Layer already exists`. Once piped
through anything (or logged to a file), `docker push`'s output loses its
live byte-percentage progress bars and only prints on state transitions
(`Preparing` -> `Pushed`) -- to estimate progress without that, sample
`/sys/class/net/<iface>/statistics/tx_bytes` over a few seconds for actual
live throughput instead of guessing from silence.

## 7. Write `manifest.json`

Two acceptable shapes (per `participant_zenoh_submission.md` section 10:
"the OCI/Docker image and immutable digest **or** archive checksum are
supplied" -- either is sufficient on its own):

**Registry-based** (no archive needed):
```json
{
  "submission_format": "origami-oci-registry-v1",
  "team_id": "orvizkar",
  "image": "orvizkar/origami-policy-t1:sept1-compiled",
  "image_id": "sha256:...",
  "image_digest": "orvizkar/origami-policy-t1@sha256:...",
  "image_size_bytes": 21028462657,
  "protocol": "origami-zenoh-v1",
  "action_dim": 65,
  "action_horizon": 25
}
```

**Archive-based** (`competition_participant_complete_guide.md` section
16.2's documented schema -- use this when shipping the `.tar.zst`):
```json
{
  "submission_format": "origami-oci-archive-v1",
  "team_id": "orvizkar",
  "image": "orvizkar/origami-policy-t1:sept1-compiled",
  "image_id": "sha256:...",
  "archive": "orvizkar-origami-policy-t1-sept1-compiled.tar.zst",
  "archive_size_bytes": 13114207592,
  "archive_sha256": "...",
  "protocol": "origami-zenoh-v1",
  "action_dim": 65,
  "action_horizon": 25
}
```
Get `image_id` via `docker image inspect --format '{{.Id}}' <image>`, and
`image_digest`/`RepoDigests` via
`docker image inspect --format '{{json .RepoDigests}}' <image>` (only
populated after a push). If asked to regenerate a checksum in a manifest,
recompute it fresh (`sha256sum`) rather than trust a value from memory or
an earlier file -- confirmed clean today, but that's a verify-every-time
habit, not a one-off.

## Without a compiled policy (no `torch.compile`)

If a policy never calls `torch.compile` at all, most of this doc collapses:

- **No compile-cache step at all** -- skip section 2's "full regeneration"
  path entirely; there's no `compile-cache/` directory, no
  `TORCHINDUCTOR_CACHE_DIR`/`TRITON_CACHE_DIR`, nothing to bake or copy out
  of a throwaway container.
- **No warmup-triggered slow first compile** -- container startup is just
  checkpoint load + a few sanity-check forward passes; expect `READY`
  within seconds of `model_load`, not ~160-200s later.
  Correspondingly, no `slow_pass ... slow_ms=1xxxxx` outlier and no
  `Graph break from Tensor.item()` warning (that warning is specifically a
  `torch.compile`/Dynamo artifact).
  - Note: this codebase's `slow_pass`/`fast_pass` split refers to the
    cascaded flow-matching architecture (Qwen3-VL vision encoding vs. the
    tactile continuation step), not to compilation -- that split still
    exists without `torch.compile`; only the *compile-driven* first-touch
    latency spike disappears.
- **Everything from section 3 onward is unchanged**: same hardening flags,
  same read-only-rootfs gotcha for code updates, same two validators, same
  archive/push/manifest steps. Compilation only affects what happens
  between `model_load` and `READY`, and steady-state per-tick latency
  (uncompiled eager-mode kernels are slower per call, with no warmup cost
  paid up front -- compiled trades a large one-time cost for lower
  steady-state latency).

## Gotchas encountered today (worth re-reading before repeating this)

- Piping a long-running remote command through `tail` (`cmd | tail -N`)
  buffers ALL of that command's output until it exits -- looks like total
  silence for the command's whole duration, not a stall. Use `grep`
  (doesn't buffer the same way), redirect to a file and poll it, or a
  separate terminal instead.
- `docker restart` on a `--read-only` container can itself error
  (`container rootfs is marked read-only`) even though the container was
  already running fine -- `docker stop` + `docker start` as two separate
  commands worked where `docker restart` did not.
- Restarting the Docker daemon (e.g. after installing
  `nvidia-container-toolkit`) kills every running container, including
  unrelated ones (`origami-contract-router` went down as a side effect) --
  expect to manually restart those afterward.
