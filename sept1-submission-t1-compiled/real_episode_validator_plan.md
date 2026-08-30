# Plan: `check_zenoh_policy_real_episodes.py`

Additive validator that replays real recorded episodes through the actual
running container (Zenoh protocol), as a third leg alongside the two
existing checks:

| Tool | Uses the server? | Uses real data? |
|---|---|---|
| `check_zenoh_policy.py` (deployed SDK) | yes | no (synthetic only) |
| `eval_origami.py` | no (calls the model directly) | yes |
| `check_zenoh_policy_real_episodes.py` (this) | yes | yes |

## File

New file, alongside the deployed `check_zenoh_policy.py` on remote
(`/root/sharpa_north_ces_lite_sdk-main/examples/`). Doesn't modify or import
from the existing script beyond copying its low-level Zenoh helpers (session
open, `pack_payload`/`unpack_payload`, error-envelope check) so both stay
independent and one can't break the other.

## Data source: ported `real_observation_source.py`

Self-contained (~230 lines, only depends on `numpy` + `lerobot` +
`cv2`/`PIL`), ported from the `my-origami-inference-kit-participant` SDK
fork's `examples/real_observation_source.py`. Provides `RealObservationSource`:

- wraps one `LeRobotDataset` (raw LeRobot v3.0 format: `meta/`, `data/`,
  `videos/` -- NOT the "origami-flat" parquet format `trex_origami.prepare`
  produces for training)
- `next_observation(frame_stride)` -> one wire-protocol-shaped observation
  dict, advancing a cursor through one episode
- `get_ground_truth_actions(horizon)` -> the real teleop action chunk
  starting at the frame just returned, padding with the final action past
  episode end
- `reset_episode(episode_index)` -> jump to a different episode, resetting
  cursor/metadata

No changes needed to this file beyond copying it as-is.

## New dependency: `lerobot` pip package

Not installed anywhere we currently run things (not in
`requirements-inference.lock`, not baked into the submission image -- this
script is a test client, never ships). Needs a venv or throwaway container
with `lerobot` + `av`/`opencv` installed, separate from the policy image.
Candidate: a small venv alongside `sharpa_north_ces_lite_sdk-main/.venv` on
remote, or install straight into that same venv if there's no conflict.

## Data: `container_check_dataset/manifest.json`

Already built (see `/media/sai/CRUZER_BLA/ori/container_check_dataset/` on
the local desktop, to be rsynced to remote). Structure:

```json
{
  "train_slot": {"season": "<name>/lerobot3.0", "episode_indices": [0,1,2,3,4]},
  "val_slot":   {"season": "<name>/lerobot3.0", "episode_indices": [9,10,11,12,13]}
}
```

`season` paths are relative to a `--data-root` CLI arg (different on local
vs remote -- nothing about the manifest itself needs to change between
machines, only where `--data-root` points). Right now both slots point at
the same real TRAIN-split season (`season_POC22061_2026_07_09_16_23_46_train`,
14 episodes total) with non-overlapping episode ranges -- val_slot is an
explicit placeholder until a genuine `VAL_SEASONS` entry is downloaded on
remote with full streams (see manifest's own `next_step` field).

## CLI surface

```
check_zenoh_policy_real_episodes.py
  --endpoint ...  --session-id ...  --timeout ...        # same as check_zenoh_policy.py
  --manifest container_check_dataset/manifest.json
  --data-root /path/to/dataset/parent
  --train-episodes N     # default: all of train_slot.episode_indices, min 1
  --val-episodes M       # default: all of val_slot.episode_indices, min 1
  --frame-stride 1       # passed straight to RealObservationSource.next_observation()
  --frames-per-episode K # how many frames to replay per episode before moving to the next
```

`--train-episodes`/`--val-episodes` slice `episode_indices[:N]` /
`[:M]` from the manifest -- explicit requirement: a run might use only 1
episode from each slot instead of all 5, without touching the dataset or
manifest.

## What it does, per episode

1. `reset` (Zenoh call) + `source.reset_episode(episode_index)`
2. loop `--frames-per-episode` times (or until `source.has_next()` is false):
   - `obs = source.next_observation(frame_stride)`
   - send `infer` with `obs`, time the round trip
   - `gt = source.get_ground_truth_actions(action_horizon)`
   - compute per-request MAE/MSE (radians, matching `eval_origami.py`'s
     metric definitions for direct comparability) between the reply's
     actions and `gt`
3. aggregate per slot (train/val) and overall: latency mean/p50/p95/max
   (matching the existing validator's latency summary format) plus
   MAE/MSE mean and per-episode breakdown

## Output

Prints a PASS/FAIL summary line per episode (same style as
`check_zenoh_policy.py`'s `infer N/M: PASS (... ms)` lines) plus a final
`metrics.json`-style file (reuse that name/shape where the fields overlap
with `eval_origami.py`'s output, so results are diffable against the
offline baseline) written to `--out-dir`.

## Open items before coding starts

1. Confirm `--frames-per-episode` default -- a quick spot check (e.g. 10
   frames/episode) or full-episode replay (some episodes run into the
   thousands of frames at 30 Hz, which would take a long time serialized
   through one Zenoh session).
2. Confirm the val_slot placeholder is acceptable to ship this first version
   against, given the genuine val season download is still pending on
   remote.
3. Decide where to install `lerobot` on remote before this can actually run.
