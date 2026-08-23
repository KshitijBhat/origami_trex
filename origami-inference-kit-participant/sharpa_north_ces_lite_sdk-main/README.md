# Origami Participant SDK

This directory contains only public tools for competition teams:

```text
examples/
  policy_server_template.py     # Production origami-zenoh-v1 server template
  check_zenoh_policy.py         # Synthetic black-box validator
  remote_observation_client.py  # Public, read-only observation client

participant_local_evaluator/    # Local image Shadow/URDF evaluator
scripts/docker/                 # Policy template, validator, and remote-client images
tests/                          # Public, self-contained tests
```

## Installation and testing

```bash
uv sync --frozen --no-install-project
uv run --no-sync python -m unittest discover -s tests -v
```

## Implementing the image service

Copy and modify:

```text
examples/policy_server_template.py
```

Teams need to replace only the model loading logic, `reset()`, and `infer()` in `TeamPolicy`.
Do not modify the public queryables, envelope, metadata, observation/action validation, or
65-dimensional joint order.

At runtime, the production image reads:

```text
ORIGAMI_ZENOH_ENDPOINT
ORIGAMI_SESSION_ID
```

## Synthetic validator

After starting the local Zenoh router and image, run:

```bash
uv run --no-sync python examples/check_zenoh_policy.py \
  --endpoint tcp/127.0.0.1:17447 \
  --session-id local-contract-test \
  --requests 3 \
  --expected-horizon 25
```

## Public, read-only observations

After reserving a time slot, use the endpoint, session, token, and TLS CA sent separately by the organizer:

```bash
uv run --no-sync python examples/remote_observation_client.py
```

This client only reads observations identical to the production `infer` input. It does not provide actions or robot control.

## Docker builds

Build commands for the framework-neutral policy template, validator, and remote
observation client are documented in `scripts/README.md`. OpenPI-specific
submission Dockerfiles are under `../openpi-base-main/scripts/docker/`.

For the complete workflow, see `PARTICIPANT_GUIDE.md` in the repository root.
