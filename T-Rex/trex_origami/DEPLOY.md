# Deploying the T-Rex origami policy (`origami-zenoh-v1`)

Everything the competition Gateway talks to lives in three files, all built on
one adapter so the offline replay and the live server cannot drift apart:

| file | role |
|---|---|
| `trex_origami/policy.py` | `TRexOrigamiPolicy` -- `reset()` / `infer(observation)` on the kit's full public observation; returns `float32[25, 65]` absolute radians |
| `scripts/serve_origami_zenoh.py` | the kit's `policy_server_template.py` with the adapter plugged in (codec, envelope, `metadata` / `reset` / `infer` queryables) |
| `scripts/replay_origami_real.py` | replays real episodes (HF LeRobot v3.0 season or origami-flat split) through the adapter, scores chunks, latency and the Gateway's async temporal aggregation |
| `scripts/eval_deploy.py` | offline sweep of inference configurations (flow steps, K, prompt, anchor source, safety, request latency) on held-out val |
| `scripts/bench_policy_latency.py` | batch-1 request latency per configuration with a phase breakdown |
| `trex_origami/joint_limits.json` | URDF position/velocity limits + kit step-jump budgets, exported so the image needs no URDF asset |

## Recommended configuration

See "Findings" below for the numbers behind each choice.

```bash
python scripts/serve_origami_zenoh.py \
    --checkpoint_path /opt/policy/checkpoint \
    --mode cascaded --total_steps 5 --split_step 3 --n_draws 8 \
    --anchor_source state_offset --safety tol --execution-mode async
```

These are the defaults of `PolicyConfig` except `--total_steps 5 --split_step 3`
(the trained schedule is 10/6; 5/3 keeps tau_split = 0.4 at half the latency).

## What `infer` does with the observation

* `observation/image/head_left` -> slow-expert image; `[wrist_right, wrist_left]`
  -> fast images (the training order). `head_right`, `joint_torque` and
  `tactile_raw` are accepted and ignored.
* `prompt` is ignored; the trained instruction is injected (`north ces task`
  for `checkpoint-2-7000`, which recorded no `--instruction`).
* `observation/tactile` (60) -> normalised `tacf6` token, plus a 16-frame 30 Hz
  history for the embedded VQ-VAE, resampled (hold-previous) from the wrenches
  seen since `reset`. `observation/image/tactile_deform` -> grayscale, split
  into the 10 fingertip tiles.
* K flow draws share one vision/language prefix; their mean is denormalised,
  the 14 arm dims are added to the anchor (`state + mean tracking offset` by
  default), the frozen torso dims are held at the measured state, and the
  chunk is projected into the Shadow evaluator's feasible set.

## Testing without the robot

Protocol (needs `zenohd`, e.g. the standalone 1.9.0 binary in `/workspace/tools`):

```bash
/workspace/tools/zenohd -l tcp/127.0.0.1:17447 &
ORIGAMI_ZENOH_ENDPOINT=tcp/127.0.0.1:17447 ORIGAMI_SESSION_ID=local-contract-test \
    python scripts/serve_origami_zenoh.py --checkpoint_path /workspace/outputs/checkpoint-2-7000 \
    --total_steps 5 --split_step 3 &
cd /workspace/origami_trex/origami-inference-kit-participant/sharpa_north_ces_lite_sdk-main
python examples/check_zenoh_policy.py --endpoint tcp/127.0.0.1:17447 \
    --session-id local-contract-test --timeout 180 --requests 5 --expected-horizon 25
```

Real data (a season from `SharpaIT/Robotic_Origami_Challenge`, gated -- needs
`HF_TOKEN`; `tactile_raw` can be skipped, it is 73% of the bytes):

```bash
python scripts/replay_origami_real.py --checkpoint_path /workspace/outputs/checkpoint-2-7000 \
    --source lerobot --root /workspace/data/origami_trex/raw/<season>/lerobot3.0 \
    --episodes 0 3 --every 5 --max_seconds 40 --total_steps 5 --split_step 3 \
    --urdf /workspace/urdf/north_poc2_2_v3_1.urdf --out_dir /workspace/eval/.../replay_raw
```

`--cadence measured` (default) issues the next request only when the previous
one has returned, and the Gateway replay uses each request's real latency, so
the executed-stream numbers reflect *this* GPU. `--latency_frames N` asks what a
faster/slower GPU would do.

## Tactile history vs. request rate

The VQ-VAE window is 16 consecutive frames at the native 30 Hz (0.53 s), fixed
by training regardless of the sample stride. The wire delivers one wrench per
`infer`, i.e. at the request rate (2-3 Hz here), so the window cannot be
rebuilt at 30 Hz. `eval_robustness` on this checkpoint: freezing the history or
striding it 2x/4x changes MAE by 0.000 deg, dropping tactile entirely by
0.002 deg. The adapter resamples what it has (hold-previous) and tiles the
current frame when it has nothing else; neither choice is measurable.

## Findings (checkpoint-2-7000, held-out val seasons)

Source: `/workspace/eval/full/checkpoint-2-7000/deploy/metrics_deploy.json`
(`scripts/eval_deploy.py`, 1000 strided val samples for the chunk pass, six
`val_stride5` episodes x 150 rows for the Gateway replay) and
`latency_bench.json` (`scripts/bench_policy_latency.py`, idle A100-40GB).
All errors are open-loop MAE in degrees against the teleoperator's commands.

**Latency (eager, batch 1, A100).** cascaded 10/6: ~630 ms (19 frames);
cascaded 5/3: ~370 ms (11 frames); blind 5: ~300 ms. K=1 vs K=8 is within
noise (the K draws share the prefix). Two-thirds of the time is the flow
steps themselves (~60 ms per Euler step), not the vision prefix (~40 ms).

**Chunk pass (K=8, oracle anchor, raw):**

| config | MAE | motion | arms | hands | jerk |
|---|---|---|---|---|---|
| cascaded 10/6 | 1.567 | 1.152 | 1.460 | 1.815 | 0.74 |
| cascaded 5/3 | 1.562 | 1.114 | 1.455 | 1.808 | 0.78 |
| blind 10 | 1.815 | 1.777 | 1.539 | 2.158 | 2.39 |
| hold_state | 1.402 | 0.780 | 2.365 | 1.302 | 0 |
| repeat_command | 0.780 | 0.780 | 1.330 | 0.717 | 0 |

* Flow steps: 5/3 equals 10/6 (K=1: 2.081 vs 2.137) -- take the latency.
* Draws: K=1 -> 4 -> 8 = 2.08 -> 1.65 -> 1.56 (cascaded 5/3). K=16 costs
  +130 ms and was not scored.
* Prompt: the trained "north ces task" beats "fold the paper into a paper
  airplane" by 0.055 (1.567 vs 1.622 at K=8).
* Anchor (cascaded 5/3 K=8): oracle previous command 1.562 / arms 1.455;
  state + mean tracking offset 1.682 / arms 2.012; raw state 1.743 / arms
  2.296. The offset recovers about half of what the oracle has over the
  state; `state` is barely better than holding the arms still.
* Safety projection: +0.03 MAE, flagged values 0.86% -> 0.00%.
* Hands are where the model loses to hold_state (1.81 vs 1.30); arms are
  where it wins, and only with a good anchor. This is model quality, not an
  inference setting.

**Gateway replay (K=8, safe, kit aggregation agg_n=4 exp_k=0.01):**

| latency (frames) | 0 | 5 | 10 | 12 | 15 | 20 | 25 |
|---|---|---|---|---|---|---|---|
| cascaded 5/3, state_offset | 1.84 | 2.06 | 2.29 | 2.45 | 2.59 | 2.93 | none executed |
| cascaded 10/6, state_offset | 1.86 | 2.08 | 2.32 | 2.48 | 2.61 | 2.94 | none executed |
| cascaded 5/3, oracle | 1.77 | 2.00 | 2.24 | 2.41 | 2.55 | 2.90 | none executed |
| cascaded 5/3, self | 5.17 | 5.65 | 6.20 | 5.71 | 5.80 | 5.53 | none executed |
| cascaded 5/3, K=1 | 2.05 | 2.28 | 2.83 | 3.14 | 3.30 | 3.57 | none executed |
| hold_state (5-frame refresh) | 0.84 | | | | | | |

* The Gateway aligns each chunk to its observation frame and the next
  inference starts when the previous returns, so a chunk arriving L frames
  late covers 25 - L future frames while the replan cycle is ~L frames.
  Steady-state stale (hold-last) fraction: 0 up to 10 frames, ~7% at 11,
  13% at 12, 33% at 15, 75% at 20, 100% from 25. That is why 5/3 (11-12
  frames eager) is the choice and why compiling to well under 10 frames is
  the next win; 10/6 at 19-20 frames is mostly stale.
* At the operating latencies the replan cycle is at least as long as the
  usable chunk, so consecutive chunks barely overlap and the Gateway's
  temporal aggregation reduces to "latest chunk"; smoothing comes from
  K-averaging, not from the ensembler.
* `self` anchoring (the policy's own last chunk, as `scripts/test.py` does
  by default) drifts without state feedback and is 3x worse here; the
  open-loop replay overstates this, but there is no evidence for it either.
* Safety projection costs ~0.01 on the stream and cuts flagged values
  roughly 3x (seam jumps between chunks remain).
