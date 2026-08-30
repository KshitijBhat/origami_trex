# Compile-cache regeneration recipe

Run from the submission build context on the remote GPU host (this session:
`/root/aug31_submission_origami_trex_v1` on the RTX 4090 host,
`64.64.224.51`; the directory name is a leftover from an earlier checkpoint
and doesn't need to match the image tag -- e.g. it currently produces
`orvizkar/origami-policy-t1:sept1-compiled`, not `:aug31-compiled`).

## Which path applies

The baked `compile-cache/` is keyed to (checkpoint weights + model/graph
code + GPU architecture) -- **not** to `trex_policy_server.py`'s
orchestration code (logging, Zenoh handling, request validation, etc.).
Pick the cheap path unless one of the three things above actually changed.

| Changed | Path |
|---|---|
| Only `trex_policy_server.py` orchestration (logging, protocol handling, anything outside the `TeamPolicy` model-forward methods) | **Cheap rebuild**, below |
| `qwen_vla/`, `tactile_vqvae/`, the checkpoint itself, or moved to a different GPU architecture | **Full regeneration**, below |

## Cheap rebuild (compile-cache unchanged)

Just rebuild -- Docker's layer cache reuses the existing `compile-cache/`
and `checkpoints/model/` directories in the build context as-is (BuildKit
matches COPY layers by content hash, so this is fast: ~20-25s for the
8.5GB checkpoint layer, ~2s for the 132MB compile-cache layer, when neither
actually changed).

```bash
cd /root/aug31_submission_origami_trex_v1
docker build -t orvizkar/origami-policy-t1:<new-tag> .
```

This is exactly what produced `sept1-compiled` from `aug31-compiled` this
session -- only `trex_policy_server.py`'s logging additions changed (episode
reset events, per-tick GPU memory, optional-key inventory); the
`compile-cache/` directory was untouched and reused byte-for-byte.

## Full regeneration (6-step loop)

Needed when the checkpoint, `qwen_vla`/`tactile_vqvae` model code, or GPU
architecture changes -- the previously baked cache would either miss (slow
first-touch compile at runtime instead of at bake time) or, worse, silently
load a stale/incompatible cache.

```bash
# 1. Build a base image with an EMPTY compile-cache/ (just code + checkpoint, no cache yet)
mkdir -p /root/aug31_submission_origami_trex_v1/compile-cache
cd /root/aug31_submission_origami_trex_v1
docker build -t origami-trex/policy:precompile .

# 2. Run it WITHOUT --read-only (writable), override entrypoint to just sleep
docker rm -f compile-cache-gen 2>/dev/null
docker run -d --name compile-cache-gen --gpus all \
  --entrypoint sleep origami-trex/policy:precompile infinity

# 3. Inside that writable container, run compile_warmup.py -- this constructs
#    TeamPolicy (triggers its own __init__ warmup: fixed + zero-tactile
#    patterns), then ALSO exercises the varied-content pattern. Every
#    torch.compile call here writes real .so/.py cache files to
#    TORCHINDUCTOR_CACHE_DIR / TRITON_CACHE_DIR (both under /app/compile-cache
#    per the Dockerfile's ENV), since the container isn't read-only.
docker exec -d compile-cache-gen bash -c \
  'python3 /app/compile_warmup.py > /tmp/compile_warmup.log 2>&1'
# ... poll: docker exec compile-cache-gen cat /tmp/compile_warmup.log
#     until "[compile_warmup] done" appears
#     (RTX 5090: ~9-10 min from a cold cache; RTX 4090: noticeably longer,
#     see timing notes below -- budget more like 12-15 min on weaker GPUs)

# 4. Copy the now-populated cache OUT of the container, replacing the empty
#    one in the build context
rm -rf /root/aug31_submission_origami_trex_v1/compile-cache
docker cp compile-cache-gen:/app/compile-cache \
  /root/aug31_submission_origami_trex_v1/compile-cache

# 5. Rebuild the FINAL image -- this time the Dockerfile's
#    `COPY compile-cache /app/compile-cache` bakes in the real, populated
#    cache instead of an empty placeholder
cd /root/aug31_submission_origami_trex_v1
docker build -t origami-trex/policy:localtest .
docker tag origami-trex/policy:localtest orvizkar/origami-policy-t1:<new-tag>

# 6. Clean up the throwaway writable container
docker rm -f compile-cache-gen
```

## Recreate the hardened container + validate

Whichever path was used above, the running container needs to be recreated
from the new image -- **`docker cp`-patching a running container does not
work here and will not be picked up**: this container runs with
`--read-only` rootfs as part of its hardening profile, and a file copied in
via `docker cp` does not persist (confirmed the hard way this session -- a
`trex_policy_server.py` patch applied via `docker cp` + `docker restart`
silently reverted; `docker exec ... grep` on the running container showed
the old code was still there). Any code or cache change requires a full
image rebuild followed by removing and recreating the container, not a
patch-in-place.

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
  orvizkar/origami-policy-t1:<new-tag>

# wait for "warmup complete elapsed_ms=..." and "READY" in:
docker logs -f origami-contract-policy
```

The container's own user (`65532:65532`) comes from the Dockerfile's `USER`
instruction -- no need to pass `--user` explicitly.

### Validate: two complementary checks

```bash
# 1. Synthetic protocol-conformance check (no real data, from the SDK dir)
python3 examples/check_zenoh_policy.py \
  --endpoint tcp/127.0.0.1:17447 --session-id test-session \
  --timeout 30 --requests 10 --expected-horizon 25

# 2. Real recorded episodes replayed through the actual running container
#    (needs the uv-managed venv -- see sharpa_north_ces_lite_sdk-main/README.md)
cd /root/sharpa_north_ces_lite_sdk-main
uv run --no-sync python examples/check_zenoh_policy_real_episodes.py \
  --endpoint tcp/127.0.0.1:17447 --session-id test-session --timeout 30 \
  --manifest /root/container_check_manifest.json \
  --data-root /root/competition_paper_set_data \
  --train-episodes 1 --val-episodes 1 \
  --out-dir /root/container_validations
```

The real-episode validator auto-creates a timestamped subfolder under
`--out-dir` and writes `real_episode_metrics.json`, `validator.log` (its own
stdout), and `container.log` (the policy container's full lifetime log,
captured automatically) into it -- nothing else to wire up.

## Timing, by host (measured, not estimated)

| | RTX 5090 (old host, `aug31-compiled`) | RTX 4090 (this host, `sept1-compiled`) |
|---|---|---|
| First-touch compile (`slow_pass`, fixed pattern) | ~65-160s | ~157-159s |
| Total `warmup complete elapsed_ms` | ~160s | ~197-200s |
| Steady-state `fast_pass` latency | ~30ms median | ~85-90ms |

The RTX 4090 numbers are consistently ~3x the RTX 5090's for steady-state
inference, and moderately slower for the one-time compile -- weaker
hardware, not a regression. Numbers above are from real `container.log`
output, not projections; re-measure on any new host rather than trusting
these as universal constants.

## Notes on what's baked into the cache

`compile_warmup.py` exercises three distinct observation patterns (each is
its own compiled graph -- torch.compile needs a fresh, expensive compile the
first time any process touches a pattern it hasn't seen before):

1. **fixed** -- `make_warmup_observation()` (imported from
   `trex_policy_server.py`, not duplicated)
2. **varied** -- `make_varied_observation(seed)`, random content, build-time
   only (not re-exercised by the runtime's own `_warmup()`)
3. **zero-tactile** -- `make_zero_tactile_observation()` / matching
   `make_zero_tactile_warmup_observation()` in `trex_policy_server.py` --
   matches the public validator's exact tactile/image content pattern
   (all-zero tactile/torque/deform + gradient images)

The runtime server's own `_warmup()` (inside `TeamPolicy.__init__`, before
`READY`) exercises patterns 1 and 3 -- covering the validator's first live
query before it ever arrives.

`FIXED_PROMPT` (`trex_policy_server.py`) is set to `"north ces task"`,
matching the actual training-time value (`origami_trex` main branch's
`trex_origami/seasons.py::INSTRUCTION`), not an arbitrary string -- every
real query's `prompt` field is ignored and replaced with this value, both to
avoid a fourth compiled-graph axis and to avoid a train/inference prompt
mismatch. Do not change this for an already-trained checkpoint without
retraining -- see the prompt-swap discussion in `checklist.md` item 3.

## TODO

The checkpoint path (`checkpoints/model` in the Dockerfile `COPY`) and the
image tag are still hand-edited per submission, not parameterized. Worth a
build arg or env var if this recipe gets run for more than one checkpoint
going forward.
