# Compile-cache regeneration recipe

The 6-step loop used every time `compile-cache/` needs to be regenerated
(e.g. after any change to `trex_policy_server.py`, `compile_warmup.py`, or
the checkpoint). Run from `/root/aug31_submission_origami_trex_v1` on the
remote GPU host.

**Status as of last run**: paused mid-step 3 (`compile-cache-gen` container
was running `compile_warmup.py` when told to hold off). Not resumed yet.

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
#    patterns), then ALSO exercises the varied-content patterns. Every
#    torch.compile call here writes real .so/.py cache files to
#    TORCHINDUCTOR_CACHE_DIR / TRITON_CACHE_DIR (both under /app/compile-cache
#    per the Dockerfile's ENV), since the container isn't read-only.
docker exec -d compile-cache-gen bash -c \
  'python3 /app/compile_warmup.py > /tmp/compile_warmup.log 2>&1'
# ... poll: docker exec compile-cache-gen cat /tmp/compile_warmup.log
#     until "[compile_warmup] done" appears (took ~9-10 min from a cold cache)

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
docker tag origami-trex/policy:localtest orvizkar/origami-policy-t1:aug31-compiled

# 6. Clean up the throwaway writable container
docker rm -f compile-cache-gen
```

## After step 6: validate

Restart the real container under the full hardening profile and run the
public validator.

```bash
docker rm -f origami-contract-policy 2>/dev/null
docker run -d --name origami-contract-policy \
  --network origami-contract-test --gpus all \
  --read-only --cap-drop ALL --security-opt no-new-privileges=true \
  --user 65532:65532 \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=4g \
  --tmpfs /run:rw,noexec,nosuid,nodev,size=64m \
  --shm-size 8g --memory 32g --cpus 8 --pids-limit 512 \
  -e ORIGAMI_ZENOH_ENDPOINT=tcp/origami-contract-router:7447 \
  -e ORIGAMI_SESSION_ID=test-session \
  orvizkar/origami-policy-t1:aug31-compiled

# wait for "warmup complete elapsed_ms=..." and "READY" in:
docker logs -f origami-contract-policy

# then, from wherever check_zenoh_policy.py lives:
python3 check_zenoh_policy.py \
  --endpoint tcp/127.0.0.1:17447 --session-id test-session \
  --timeout 30 --requests 10 --expected-horizon 25
```

Expect `warmup complete` around ~160s (two patterns now: fixed synthetic +
zero-tactile), then `PASS` on every `infer` call, median latency ~30ms.

## Notes on what's baked into the cache

`compile_warmup.py` currently exercises three distinct observation patterns
(each is its own compiled graph -- torch.compile needs a fresh ~80-290s
compile the first time any process touches a pattern it hasn't seen before):

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
mismatch.
