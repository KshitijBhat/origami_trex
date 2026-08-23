#!/usr/bin/env bash
filepath="examples/check_zenoh_policy.py"
srcpath="~/ori/submission-0/scripts/run.sh"

source $srcpath



# dataset only 
expected_horizon=25
datasetroot="season_POC22061_2026_07_09_16_23_46_train/lerobot3.0"

uv run --no-sync python $filepath \
  --endpoint tcp/127.0.0.1:17447 \
  --session-id "$SESSION" \
  --timeout 180 \
  --expected-horizon $expected_horizon \
  --obs-type dataset \
  --dataset-root $datasetroot \
  --episode-index 0 \
  --dataset-requests 3 \
  --frame-stride $expected_horizon \
  --out-dir dataset_checks


# #requests is sythetic requests
# uv run --no-sync python $filepath \
#   --endpoint tcp/127.0.0.1:17447 \
#   --session-id "$SESSION" \
#   --timeout 180 \
#   --requests 3 \
#   --expected-horizon 25 \
#   --obs-type both \
#   --dataset-root "Robotic_Origami_Challenge/season_POC22032_2026_05_14_19_21_01_train/lerobot3.0" \
#   --episode-index 0 \
#   --dataset-requests 10 \
#   --frame-stride 25 \
#   --save-actions ./actions_ep0.npy

# dummy data only 
# uv run --no-sync python $filepath \
#   --endpoint tcp/127.0.0.1:17447 \
#   --session-id "$SESSION" \
#   --timeout 180 \
#   --requests 3 \
#   --expected-horizon 25 \
#   --obs-type synthetic 