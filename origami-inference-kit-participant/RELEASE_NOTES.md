# Origami Inference Kit Async — 2026-08-26

This is the participant-facing update of the Origami inference kit.

## What changed

- `execution_mode=async` is now the default when metadata or the Docker build
  argument is omitted;
- `execution_mode=sync` remains supported;
- Added asynchronous inference/control decoupling with temporal aggregation;
- Added the offline `examples/check_async_time_aggregation.py` smoke test;
- Updated the OpenPI Zenoh server and bundled submission Docker image;
- Added tests for async metadata, defaults, Docker propagation, and aggregation;
- Retained the complete OpenPI/pi model reference, public tensor contract,
  Docker templates, protocol validator, and read-only local evaluator.

## Verified

On 2026-08-26:

```text
43 participant-kit unit tests: PASS
offline async temporal aggregation smoke test: PASS
Gateway sync real-robot test: PASS (organizer environment)
Gateway async real-robot test: PASS (organizer environment)
```

The real-robot statements describe organizer-side validation; the participant
package itself has no robot-control transport.

## Public/private boundary

The release contains no `Sharpa.py`, private North SDK/client, robot IP/topic,
action publisher, checkpoint, credential, token, certificate, or organizer
Gateway source. Teams provide their own checkpoint and runtime assets.

## Start here

Read `README.md`, then `PARTICIPANT_GUIDE.md`. Run:

```bash
cd sharpa_north_ces_lite_sdk-main
uv sync --frozen --no-install-project
uv run --no-sync python -m unittest discover -s tests -v
uv run --no-sync python examples/check_async_time_aggregation.py
```
