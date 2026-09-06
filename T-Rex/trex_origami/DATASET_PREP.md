# Dataset prep summary

Source: `SharpaIT/Robotic_Origami_Challenge` (HF dataset repo).
Output: `drakedrake/ori-trex-competition` (`train/` + `val/`).

## What this build is

The **competition-paper-set** revision: 35 train seasons / 3 val seasons
(576 / 34 episodes, 522,715 / 34,090 samples). Built from the dedicated
`competition-paper-set` branch on the source repo, not `main` — 21 of these
38 seasons had rotated off `main` by the time we ran this, so `fetch.py` now
passes the HF `revision` through explicitly instead of always pulling the
default branch (see `fetch.py::download_season`/`list_hub_seasons`).

The `main`-revision split (`splits_main.json`, 116 train / 10 val) is the
larger pool for the eventual full run; not (re)built in this pass.

## Config

| | |
|---|---|
| sample rate | 6 Hz (`sample_stride=5` on the native 30 Hz stream) |
| action chunk | 16 steps, spaced at native ~33ms (`chunk_stride=1`) — independent of sample_stride |
| action_dim | 65 (all-absolute joint radians; no eef pose in this dataset) |
| image size | 224px, JPEG-encoded in-parquet |
| instruction | "Use both hands to fold the paper into an airplane on the table, alternating hands to crease the corners inwards and fold the wings" |
| phase_mode | none |

Full field-by-field reference: `prep_config_reference.json`.

## Row schema

`state[65]`, `torque[65]`, `action_chunk[16*65]`, `action_abs[65]`, `phase`,
`frame_index` (episode-relative), `season_frame_index` (season-relative),
`tacf6_hist[16*60]`, plus JPEG blobs for `head`/`wrist_left`/`wrist_right`/`deform`.

`frozen_dims` (fixed/degenerate action dims) are recorded per-episode; the
training side can optionally mask them out of the action loss via
`train.py --mask_frozen_loss 1`.

## Known gotcha

`season_already_done()` checks `meta/dataset.json`, not disk — an interrupted
prep run can leave parquet files on disk one step ahead of what's recorded.
The `--seasons <name> --overwrite` combo forces a clean re-convert of one
season if that's ever in doubt.
