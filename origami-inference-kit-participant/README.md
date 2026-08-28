# Origami Inference Kit Async

This repository is the complete public development kit for Origami competition
teams, plus the asynchronous Gateway integration. Use it to:

- Build a self-contained Docker/OCI inference image implementing `origami-zenoh-v1`;
- Choose `sync` or `async` Gateway execution at image build time;
- Validate the image with synthetic observations and an offline async smoke test;
- Retrieve read-only physical-robot observations during a reserved time slot;
- Run read-only Shadow/URDF tests locally;
- Export an immutable image archive and checksum.

Start with [`PARTICIPANT_GUIDE.md`](PARTICIPANT_GUIDE.md).
Changes in this tested release are summarized in [`RELEASE_NOTES.md`](RELEASE_NOTES.md).

## Public contents

```text
PARTICIPANT_GUIDE.md
docs/
  competition_participant_complete_guide.md
  participant_zenoh_submission.md
  robot_io_spec.md
  container_submission.md
  remote_participant_development.md

openpi-base-main/                   # Complete OpenPI/pi inference reference
  scripts/serve_policy_zenoh.py
  scripts/docker/submission-zenoh-bundled.Dockerfile

sharpa_north_ces_lite_sdk-main/
  examples/
    policy_server_template.py
    check_zenoh_policy.py
    remote_observation_client.py
    openpi_origami_async.py
    check_async_time_aggregation.py
  participant_local_evaluator/
  tests/
```

The kit includes OpenPI model code, sync-compatible request/reply serving,
asynchronous temporal aggregation, participant Docker templates, Zenoh protocol
validation and the local evaluator. Teams still supply their own checkpoint,
normalization assets, tokenizer and runtime dependencies.

## Quick start

```bash
cd sharpa_north_ces_lite_sdk-main
uv sync --frozen --no-install-project
uv run --no-sync python -m unittest discover -s tests -v
uv run --no-sync python examples/check_async_time_aggregation.py
```

These checks do not require robot hardware or the private North SDK.

## Build sync or async

Build from the repository root. For an asynchronous image:

```bash
docker build \
  --build-arg RUNTIME_IMAGE=<compatible-openpi-runtime-image> \
  --build-arg EXECUTION_MODE=async \
  --build-context checkpoint=/absolute/path/to/checkpoint \
  --build-context python_packages=/absolute/path/to/python/site-packages \
  -f openpi-base-main/scripts/docker/submission-zenoh-bundled.Dockerfile \
  -t team-name/origami-policy:submission \
  openpi-base-main
```

Use `EXECUTION_MODE=sync` for synchronous execution; omitting it defaults to
`async`. The server advertises `execution_mode` in metadata, and the organizer
Gateway selects the corresponding strategy automatically. It also advertises
`inference_kit=origami-inference-kit-async`, while the OCI image carries the
matching source-kit label.

The checkpoint must contain `params/` or `model.safetensors` and
`assets/**/norm_stats.json`. The Python package context must contain the public
Eclipse Zenoh 1.9 package.

## Public tensor contract

The image receives four `uint8[224,224,3]` RGB cameras, `float32[65]` joint
state and torque, `float32[60]` tactile values, tactile images, and a string
prompt. It returns:

```python
{"actions": float32[T, 65]}
```

Actions must be finite absolute joint-position targets in radians in the fixed
order defined by `docs/robot_io_spec.md`.

## Public/private boundary

This repository intentionally contains no `Sharpa.py`, `NorthClient`, private
`sharpa_north_ces_lite` transport implementation, robot IP/topic, or action
publisher. The production image contains only the policy/OpenPI source, public
Zenoh query/reply endpoint, dependencies and the team's model assets. Organizer
Gateway code remains outside the participant image.

## License

OpenPI-derived code is Apache-2.0; see `openpi-base-main/LICENSE`. Vendored and
runtime dependency notices are in `THIRD_PARTY_LICENSES.md`.
