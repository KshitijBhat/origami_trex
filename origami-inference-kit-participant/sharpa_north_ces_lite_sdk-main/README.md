# sharpa_north_ces_lite_sdk

Public participant tools for the Origami inference kit, including the formal
Zenoh server template and validator, read-only observation/evaluation clients,
and a transport-independent async temporal-aggregation runtime.

For the observation/action contract, data format, and wire protocol, see the
top-level `docs/` in this kit.

The private robot transport/runtime is intentionally not included or required.

## Install and test

```bash
uv sync --frozen --no-install-project
uv run --no-sync python -m unittest discover -s tests -v
uv run --no-sync python examples/check_async_time_aggregation.py
```

## Layout

```
examples/
  policy_server_template.py      # framework-independent production server
  check_zenoh_policy.py          # public protocol validator
  remote_observation_client.py   # public read-only observation client
  openpi_origami_async.py        # self-contained async runtime (ensembler + loop)
  check_async_time_aggregation.py# offline async smoke test
participant_local_evaluator/     # read-only Shadow/URDF evaluator
tests/
```

## Local async loop (temporal aggregation)

An asynchronous alternative to the sync loop, using rtac1-style temporal
ensembling: an inference thread (`inference_hz`) pushes each action chunk into a
scheduler with delay-compensating `offset_steps`, and a publish thread
(`control_hz`) emits one *fused* step per tick (per-step ensembling over recent
overlapping chunks). Inference jitter never stalls publishing, and overlapping
predictions are blended for a smoother stream.

The async runtime is self-contained in `examples/openpi_origami_async.py` and is
defined against a small environment interface, so it contains no private robot
transport. The supplied checker uses only an in-process observation source and
action sink.

```bash
uv run --no-sync python examples/check_async_time_aggregation.py
```

Key knobs (see `--help`): `--inference-hz`, `--control-hz`,
`--compensation-steps` (`auto` or fixed offset), and temporal-ensembling params
`--ta-agg-n` / `--ta-exp-k` / `--ta-max-chunks` / `--ta-smooth-alpha`
(`--no-ta-hold-last` to disable the hold-last fallback).
