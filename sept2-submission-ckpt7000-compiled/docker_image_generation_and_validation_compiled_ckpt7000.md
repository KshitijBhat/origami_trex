# Compiled submission image for `checkpoint-2-7000` (RTX 4090 host)

Runbook for turning `checkpoint-2-7000` plus the *deployment adapter* version
of the T-Rex origami code (`trex_origami/policy.py`, `serve_origami_zenoh.py`)
into a validated, compile-cache-baked `origami-zenoh-v1` image, archive and
manifest. It follows the same shape as the sept1 runbook
(`../sept1-submission-t1-compiled/docker_image_generation_and_validation_for_compile_without_compiled_policy.md`)
and only spells out what is different. Every command is meant to be run on
the RTX 4090 host unless it says "on the A100 box" (this machine,
`/workspace`).

## 0. What changed since the sept1 image (read before building)

| | sept1 (`trex_policy_server.py`) | this image (`serve_origami_zenoh.py` + `trex_origami/policy.py`) |
|---|---|---|
| checkpoint | `checkpoint-2-8000`, all-delta-from-state | `checkpoint-2-7000`, **hybrid anchoring** (arms = delta from the previous command, hands/motor absolute) |
| per request | slow pass every 25th request, fast-only in between | full slow + fast pass **every** request (the arm anchor and the vision must be fresh) |
| flow schedule | trained 10/6 | **5/3** (same tau_split = 0.4; identical MAE on val, half the latency) |
| draws | 1 | **K = 8** flow draws averaged (batched on one prefix; free in latency) |
| anchor | state | **state + mean tracking offset** (`state_offset`) |
| safety | none | sequential projection into the Shadow evaluator's limits (`joint_limits.json`, no URDF in the image) |
| prompt | fixed `north ces task` | same string, read from the dataset default (checkpoint records no `instruction`) |
| tactile history | rolling deque of received wrenches | time-resampled 30 Hz window (hold-previous); no measurable effect either way |
| compile | `forward_flow_action_partial`, `tactile_flow_continue` | the same two plus `forward_flow_action_full`, `mode="default"`, `automatic_dynamic_shapes=False` |
| config | env `TREX_FULL_REFRESH_EVERY` | env `TREX_MODE/TREX_TOTAL_STEPS/TREX_SPLIT_STEP/TREX_N_DRAWS/TREX_ANCHOR_SOURCE/TREX_SAFETY/TREX_COMPILE` (baked as Dockerfile `ENV`) |

Why these settings: `/workspace/eval/full/checkpoint-2-7000/deploy/metrics_deploy.json`
(and `trex_origami/DEPLOY.md` in the T-Rex repo). The numbers that matter for
the build: on the A100, eager `cascaded 5/3 K=8` takes ~370 ms per request
(11 frames at 30 Hz), `10/6` ~630 ms (19 frames). The Gateway aligns a
chunk to the observation it was computed from, so with a 25-step chunk
anything above ~20 frames of latency executes mostly stale repeats and
anything above 25 executes nothing. **Latency is the reason to compile.**

**Compile-cache key.** The baked cache is valid only for the exact tuple
(checkpoint weights, `qwen_vla/` + `tactile_vqvae/` + `trex_origami/policy.py`
code, torch + Triton versions, GPU architecture, `TREX_TOTAL_STEPS`,
`TREX_SPLIT_STEP`, `TREX_N_DRAWS`, `TREX_MODE`, prompt). Unlike sept1, the
inference *settings* are part of the key: the flow step count changes the
unrolled graph and K changes every batch dimension. Anything in
`serve_origami_zenoh.py` outside the policy call (logging, envelope) is not.

## 1. Files to bring to the 4090 host

From this box (`/workspace`) to the 4090 build context, e.g.
`/root/sept2_submission_ckpt7000`:

```bash
# on the A100 box -- refresh the vendored source first, it must match the repo
/workspace/inference/sept2-submission-ckpt7000-compiled/scripts/sync_submission_files.sh --check

# build context (small: code, Dockerfile, entrypoint, warm-up, lock, licenses)
rsync -av --exclude logs --exclude compile-cache --exclude checkpoint_meta \
    /workspace/inference/sept2-submission-ckpt7000-compiled/ \
    root@<4090-host>:/root/sept2_submission_ckpt7000/

# checkpoint WITHOUT the 13GB training-resume state/ directory
rsync -av --exclude state \
    /workspace/outputs/checkpoint-2-7000/ \
    root@<4090-host>:/root/sept2_submission_ckpt7000/checkpoints/model/

# validators: the participant kit (sync + real-episode) and one raw val season
rsync -av /workspace/origami_trex/origami-inference-kit-participant/ root@<4090-host>:/root/origami-inference-kit-participant/
rsync -av /workspace/data/origami_trex/raw/season_POC22061_2026_05_19_10_18_58_train/ \
    root@<4090-host>:/root/raw/season_POC22061_2026_05_19_10_18_58_train/     # 3.1 GB, no tactile_raw
# the replay harness + its reader live in the T-Rex repo, not in the image
rsync -av /workspace/origami_trex/T-Rex/ root@<4090-host>:/root/T-Rex/
```

The build context must contain exactly:

```text
Dockerfile  .dockerignore  entrypoint.sh  compile_warmup.py  requirements-inference.lock
serve_origami_zenoh.py
qwen_vla/        (__init__ modeling_vla modeling_qwen3vl_mot diffusion DeformAE origami_dataset)
tactile_vqvae/   (__init__ models/{__init__,encoder,decoder,quantizer,tactile_vqvae})
trex_origami/    (__init__ anchoring seasons loading policy joint_limits.json)
licenses/THIRD_PARTY_LICENSES.txt
checkpoints/model/{model.pt,config.json,training_args.json,stats_data.json,training_state.json,processor/}
compile-cache/   (empty on the first build; populated in section 2)
```

`checkpoints/model/model.pt` is 8.5 GB; `training_state.json` should say
`"global_step": 7000`.

Host prerequisites are the same as sept1: `nvidia-container-toolkit`
actually installed (`docker run --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`
works), `uv` for the kit's venv (`cd sharpa_north_ces_lite_sdk-main && uv sync --frozen --no-install-project`),
`zstd`. The raw-season replay additionally needs `ffmpeg` with an AV1 decoder
(`ffmpeg -decoders | grep -i av1` should list `libdav1d`) and `pyarrow` in the
venv you run the replay from (`uv pip install pyarrow`); it does not need
`lerobot`.

## 2. Build with a baked compile cache (full regeneration)

This checkpoint, the adapter code, and the inference settings are all new
relative to sept1, so the cheap "reuse `compile-cache/`" path does not apply
the first time. Six steps, identical in shape to sept1's recipe:

```bash
cd /root/sept2_submission_ckpt7000
mkdir -p compile-cache

# 1. image with an EMPTY cache (code + checkpoint only)
docker build -t origami-trex/policy-7000:precompile .

# 2. writable throwaway container (no --read-only), entrypoint overridden
docker rm -f compile-cache-gen 2>/dev/null
docker run -d --name compile-cache-gen --gpus all \
  --entrypoint sleep origami-trex/policy-7000:precompile infinity

# 3. populate the cache -- constructs TRexOrigamiPolicy with TREX_COMPILE=1 (the
#    image ENV), runs the warm-up patterns (random / validator all-zero-tactile /
#    varied x2) and extra varied episodes, then re-times repeats
docker exec -d compile-cache-gen bash -c 'python3 /app/compile_warmup.py > /tmp/compile_warmup.log 2>&1'
#    poll until "[compile_warmup] done" (sept1 on the 4090: ~12-15 min from cold;
#    this adapter compiles three functions and a K=8 batch -- budget 15-25 min):
docker exec compile-cache-gen tail -f /tmp/compile_warmup.log

# 4. copy the populated cache out into the build context
rm -rf compile-cache && docker cp compile-cache-gen:/app/compile-cache ./compile-cache
du -sh compile-cache          # sept1 was ~130 MB; expect the same order of magnitude

# 5. final image with the cache baked in
docker build -t origami-trex/policy-7000:localtest .
docker tag origami-trex/policy-7000:localtest orvizkar/origami-policy-t1:sept2-ckpt7000-compiled

# 6. clean up
docker rm -f compile-cache-gen
```

What the warm-up log must show before you trust the cache:

```text
[policy] torch.compile('default') armed for forward_flow_action_partial / tactile_flow_continue / forward_flow_action_full; inductor cache /app/compile-cache/torchinductor
[policy] warm-up pattern random: first+second call <long> ms, second call timing prep_ms=.. embed_ms=.. slow_ms=.. fast_ms=.. total_ms=<steady>
[policy] warm-up pattern validator_zero_tactile: ...
[policy] warm-up pattern varied_1: ...
[policy] warm-up pattern varied_2: ...
[policy] warm-up complete in <N> s
[compile_warmup]   validator   <steady> ms mean over 3 ...
[compile_warmup]   random      <steady> ms mean over 3 ...
[compile_warmup]   varied      <steady> ms mean over 3 ...
[compile_warmup] done in <N> s
```

The three "steady-state check" lines must all be in the same ballpark and
far below the first-call times. If one of them is still seconds long a
pattern is recompiling; add it to `_warmup()` in `trex_origami/policy.py`,
re-sync, and redo steps 1-5. Expected benign noise: the
`Graph break from Tensor.item()` warning at `tau_split = float(time.item())`
in `forward_flow_action_partial` (same as sept1).

Section 4 of the local A100 test of this exact vendored tree
(`/workspace/eval/full/checkpoint-2-7000/deploy/compile_local/log.txt`) is
the reference for what "good" looks like; re-measure on the 4090, the sept1
numbers there were ~3x the 5090's.

### torch version fallback

The Dockerfile pins `torch==2.11.0+cu128` (what the sept1 4090 image used for
this model code). The eval/training venv on the A100 box runs
`torch 2.6.0+cu124`, which is where the compile path was tested locally. If
the 2.11 build fails to compile or produces different actions, rebuild with

```bash
docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
  --build-arg TORCH_SPEC="torch==2.6.0+cu124 torchvision==0.21.0+cu124" ...
```

and regenerate the cache (the torch/Triton version is part of the key).

### Cheap rebuild (when it applies)

Only when nothing in the cache key changed -- e.g. a log line in
`serve_origami_zenoh.py` or `entrypoint.sh`: `docker build -t <tag> .` reuses
`compile-cache/` and the checkpoint layer by content hash. Changing any
`TREX_*` ENV, the checkpoint, or any file under `qwen_vla/`,
`tactile_vqvae/`, `trex_origami/` -> full regeneration.

## 3. Run the hardened container

Same profile as sept1 (rootfs read-only, so a `docker cp` patch never
persists -- rebuild + recreate for every change):

```bash
docker network create origami-contract-test 2>/dev/null || true
docker rm -f origami-contract-router 2>/dev/null
docker run -d --name origami-contract-router --network origami-contract-test \
  -p 127.0.0.1:17447:7447 eclipse/zenoh:1.9.0 --listen tcp/0.0.0.0:7447

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
  orvizkar/origami-policy-t1:sept2-ckpt7000-compiled

docker logs -f origami-contract-policy
```

Expected start-up sequence:

```text
T-Rex Origami Policy Server (origami-zenoh-v1)
  Flow: cascaded steps=5/3 K=8   Anchor/safety: state_offset / tol   torch.compile: 1
Checkpoint loaded: missing=0, unexpected=0
[serve] frozen action dims [58, 59] -> held at state[j] ...
[serve] action anchoring hybrid: left_arm[0:7]=prev_command ... motor[58:65]=absolute
[policy] hybrid: ...; prompt 'north ces task'; mode=cascaded steps=5/3 K=8 anchor=state_offset safety=tol compile=True
[policy] torch.compile('default') armed for ...
[policy] warm-up pattern random: first+second call <cache-hit, a few s> ...
[policy] warm-up pattern validator_zero_tactile: ...
[policy] warm-up pattern varied_1: ...
[policy] warm-up pattern varied_2: ...
[policy] warm-up complete in <N> s
READY transport=origami-zenoh-v1 endpoint=tcp/origami-contract-router:7447 horizon=25 mode=async
```

With the cache baked, "warm-up complete" should take well under a minute
(loading the kernels, not compiling them). If it takes minutes, the cache
missed: the image was built with a different key than the cache was
generated with (typically a changed `TREX_*` value or code file) -- and the
container would then die on the first Triton write under `--read-only`.
`entrypoint.sh` prints a `WARNING` if `TREX_COMPILE=1` and the inductor cache
directory is empty.

## 4. Validate

### (a) Synthetic protocol check

```bash
cd /root/origami-inference-kit-participant/sharpa_north_ces_lite_sdk-main
uv run --no-sync python examples/check_zenoh_policy.py \
  --endpoint tcp/127.0.0.1:17447 --session-id test-session \
  --timeout 30 --requests 10 --expected-horizon 25
```

Expected: `metadata: PASS`, `reset: PASS`, 10x `infer: PASS`, and a latency
line. On the A100 the same server (eager, shared GPU) passed at 612-944 ms;
the whole point of this image is to bring that down on the 4090 -- record
the median from this line in the manifest.

### (b) Real recorded episodes through the running container

Uses the repo's replay harness in Zenoh-client mode: real observations from
the raw HF season (all four cameras, torque, tactile, deform) are sent to
the container as `infer` queries, scored against the teleoperator's
commands, and the Gateway's async temporal aggregation is replayed with the
measured round-trip latency.

```bash
cd /root/T-Rex
uv run --project /root/origami-inference-kit-participant/sharpa_north_ces_lite_sdk-main --no-sync \
  python scripts/replay_origami_real.py \
  --endpoint tcp/127.0.0.1:17447 --session-id test-session --timeout 60 \
  --source lerobot --root /root/raw/season_POC22061_2026_05_19_10_18_58_train/lerobot3.0 \
  --episodes 0 3 6 9 --every 1 --start_seconds 20 --max_seconds 40 \
  --out_dir /root/container_validations/sept2-ckpt7000
```

(`--every 1` decodes every frame so the next request is issued at the true
measured cadence; use `--every 5` for a faster smoke run.) It writes
`replay_lerobot.json` plus one trace PNG per episode. Reference values from
the same episodes on the A100 (eager 5/3 K=8, in-process, idle GPU) are in
`/workspace/eval/full/checkpoint-2-7000/deploy/replay_raw_cas5_k8_offset/replay_lerobot.json`;
the container must match its chunk MAE / MSE within run-to-run noise (the
flow draws are stochastic) and should beat its latency. Also keep the
`check_zenoh_policy_real_episodes.py` run from sept1 if the manifest +
lerobot venv are still on the host; both are protocol-level and independent
of this code.

### (c) Restart / multi-episode / read-only checks

As in sept1: `docker stop` + `docker start` (not `restart`), a second
`reset` + `infer` cycle, and confirm nothing was written outside `/tmp`
(`docker diff origami-contract-policy` should list only tmpfs paths).

## 5. Archive, push, manifest

```bash
mkdir -p /root/sept2-submission && cd /root/sept2-submission
IMAGE=orvizkar/origami-policy-t1:sept2-ckpt7000-compiled
ARCHIVE=orvizkar-origami-policy-t1-sept2-ckpt7000-compiled.tar.zst
docker save "$IMAGE" | zstd -T0 -3 -o "$ARCHIVE.partial" && mv "$ARCHIVE.partial" "$ARCHIVE"
zstd -t "$ARCHIVE"
sha256sum "$ARCHIVE" | tee "$ARCHIVE.sha256"

docker login -u orvizkar
docker push "$IMAGE"
docker image inspect --format '{{.Id}}' "$IMAGE"
docker image inspect --format '{{json .RepoDigests}}' "$IMAGE"
docker image inspect --format '{{.Size}}' "$IMAGE"
```

`manifest.json` (both shapes accepted; fill every value from the commands
above, never from memory):

```json
{
  "submission_format": "origami-oci-registry-v1",
  "team_id": "orvizkar",
  "image": "orvizkar/origami-policy-t1:sept2-ckpt7000-compiled",
  "image_id": "sha256:<docker image inspect .Id>",
  "image_digest": "orvizkar/origami-policy-t1@sha256:<RepoDigests>",
  "image_size_bytes": <.Size>,
  "archive": "orvizkar-origami-policy-t1-sept2-ckpt7000-compiled.tar.zst",
  "archive_size_bytes": <stat -c %s>,
  "archive_sha256": "<sha256sum>",
  "protocol": "origami-zenoh-v1",
  "action_dim": 65,
  "action_horizon": 25,
  "execution_mode": "async",
  "checkpoint": "checkpoint-2-7000 (epoch 2, global_step 7000)",
  "inference": {"mode": "cascaded", "total_steps": 5, "split_step": 3, "n_draws": 8,
                "anchor_source": "state_offset", "safety": "tol", "prompt": "north ces task",
                "torch_compile": "default, baked inductor+triton cache"},
  "measured_on_build_host": {"gpu": "RTX 4090", "validator_median_ms": <from 4a>,
                             "real_replay_chunk_mae_deg": <from 4b>},
  "built_on": "RTX 4090 host, <date>"
}
```

Resources to state in the submission: GPU with >= 16 GB (the model is ~5 GB
in bf16 plus the K=8 flow batch; sept1 ran within the 32 GB `--memory`
limit above), `--shm-size 8g`, 8 CPUs.

## 6. Gotchas specific to this version

- **Do not change `TREX_*` at `docker run` time.** They are baked as `ENV`
  so the runtime graphs match the cache. An `-e TREX_N_DRAWS=4` on the run
  command would compile fresh and crash under `--read-only`.
- **`head_right`, `joint_torque`, `tactile_raw` are accepted and ignored**;
  `prompt` is replaced by `north ces task`. Do not "fix" this -- the model
  never saw them, and a different prompt changes the token length (a new
  graph) and measurably hurts (+0.055 deg MAE on val for the newer trainer
  default string).
- **Anchor source.** `self` (the policy's own last chunk, sept1's `auto`)
  drifts without a state correction and was 3x worse in the open-loop
  replay; `state_offset` is the default for a reason. Revisit only with
  closed-loop evidence from the robot.
- **`joint_limits.json`** is a 65-row table exported from the organizer's
  URDF (`north_poc2_2_v3_1.urdf`). The URDF/mesh package itself must not be
  in the image; the table is what the safety projection needs.
- The sept1 gotchas still hold: `cmd | tail` buffers until exit, `docker
  restart` can fail on a read-only container (stop + start), restarting the
  Docker daemon kills the router container too.
