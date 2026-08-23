# Origami Inference Kit

This repository is the public development kit for Origami competition teams. Use it to:

- Build a self-contained Docker/OCI inference image that implements `origami-zenoh-v1`;
- Validate the image protocol with synthetic observations;
- Retrieve observations from the physical robot through a public, read-only interface during a reserved time slot;
- Run read-only Shadow/URDF image tests on the team's local machine;
- Export a `.tar.zst` image archive and its SHA-256 checksum.

Start with [`PARTICIPANT_GUIDE.md`](PARTICIPANT_GUIDE.md).

## Public contents

```text
PARTICIPANT_GUIDE.md
docs/
  competition_participant_complete_guide.md
  participant_zenoh_submission.md
  robot_io_spec.md
  container_submission.md
  remote_participant_development.md

openpi-base-main/                   # OpenPI inference/model reference source

sharpa_north_ces_lite_sdk-main/
  examples/
    policy_server_template.py
    check_zenoh_policy.py
    remote_observation_client.py
  participant_local_evaluator/
  tests/
```

Key components:

- `policy_server_template.py`: Framework-independent production Zenoh server template;
- `check_zenoh_policy.py`: Public black-box validator that does not connect to the robot;
- `remote_observation_client.py`: Public, read-only observation client for use after reserving a time slot;
- `participant_local_evaluator`: Sends real, read-only observations to the final image locally and visualizes
  the predicted trajectory using the URDF.
- `openpi-base-main`: OpenPI inference and model-adapter reference source. Teams using OpenPI must still
  package their own checkpoint and runtime assets.

## Quick start

```bash
cd sharpa_north_ces_lite_sdk-main
uv sync --frozen --no-install-project
uv run --no-sync python -m unittest discover -s tests -v
```

Before your reserved time slot, you can build the image and run the synthetic validator. After reservation, the organizer will send the following values separately:

```text
ORIGAMI_REMOTE_ENDPOINT=tls/<public-host>:<port>
ORIGAMI_REMOTE_SESSION_ID=<assigned-team-session>
ORIGAMI_REMOTE_TOKEN=<assigned-team-secret>
ORIGAMI_REMOTE_TLS_CA=/path/to/organizer-ca.pem
```

Never write these credentials to Git, source code, a Dockerfile, an image layer, or logs.

## Public tensor contract

The image receives:

```python
{
    "observation/image/head_left":      uint8[224, 224, 3],
    "observation/image/head_right":     uint8[224, 224, 3],
    "observation/image/wrist_left":     uint8[224, 224, 3],
    "observation/image/wrist_right":    uint8[224, 224, 3],
    "observation/state":                float32[65],
    "observation/state/joint_torque":   float32[65],
    "observation/tactile":              float32[60],
    "observation/image/tactile_deform": uint8[480, 1200, 3],
    "observation/image/tactile_raw":    uint8[480, 1600, 3],  # optional
    "prompt":                           str,
}
```

The image returns:

```python
{"actions": float32[T, 65]}
```

Actions must be finite absolute joint-position targets in radians and must use the fixed
65-dimensional order defined in `docs/robot_io_spec.md`.

## Security boundaries

- The public observation interface is read-only and cannot send actions;
- The production image may connect only to the isolated Zenoh router injected by the organizer;
- The image must not contain the North SDK, robot IP addresses/topics, an action publisher, or public-development credentials;
- The organizer is responsible for robot I/O, action safety checks, Shadow execution, and authorized Live execution.

## License

The code in this repository is licensed under the Apache License 2.0; see `LICENSE`.
Licenses for vendored frontend dependencies and runtime dependencies are listed in `THIRD_PARTY_LICENSES.md`.
