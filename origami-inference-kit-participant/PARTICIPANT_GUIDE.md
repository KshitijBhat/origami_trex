# Origami Participant Guide

This document is the primary entry point for teams receiving this repository. Each team must complete two tasks:

1. Build a self-contained Docker/OCI inference image that implements the standard protocol;
2. During a reserved development time slot, retrieve observations from the physical robot through
   the organizer's public, read-only interface and validate both the local inference adapter and the final image.

Public observations are provided only for development validation and cannot be used to control the
robot. The submitted production image also has no direct access to the robot; the organizer is
responsible for observations, action safety checks, and execution.

## 1. What you can do before and after reservation

### Before reservation

Without a public IP address, session, or token, you can:

- Implement the model adapter;
- Build a self-contained image;
- Run the public black-box validator with synthetic observations;
- Check metadata, reset, infer, input/output shapes and dtypes, and the action horizon;
- Export a `.tar.zst` image archive and calculate its SHA-256 checksum.

### After reservation

The organizer sends the following values separately for each team and reserved time slot:

```text
ORIGAMI_REMOTE_ENDPOINT=tls/<public-host>:<port>
ORIGAMI_REMOTE_SESSION_ID=<assigned-team-session>
ORIGAMI_REMOTE_TOKEN=<assigned-team-secret>
ORIGAMI_REMOTE_TLS_CA=/path/to/organizer-ca.pem
```

If mTLS is enabled, the organizer will also send a team-specific client certificate and private key.

Teams that need the local URDF Shadow evaluator will also receive the official North asset package,
containing the URDF and meshes, through the organizer's distribution channel. This asset package is
for local visualization only and must not be included in the production submission image. Not having
the asset package does not prevent use of the synthetic validator or development of the image protocol.

These values are temporary development credentials:

- Do not write them to Git, a Dockerfile, an image layer, source-code defaults, or logs;
- Do not include them in the production submission image;
- Do not forward them to other teams;
- After the reservation ends, delete them as instructed by the organizer or wait for them to expire.

During official evaluation, the organizer injects only a separate set of isolated environment variables into the image:

```text
ORIGAMI_ZENOH_ENDPOINT=tcp/<isolated-router>:7447
ORIGAMI_SESSION_ID=<evaluation-session>
```

The production image must not depend on any `ORIGAMI_REMOTE_*` variable.

## 2. Installing the public development environment

```bash
cd /path/to/origami-inference-kit/sharpa_north_ces_lite_sdk-main
uv sync --frozen --no-install-project
uv run --no-sync python -m unittest discover -s tests -v
```

Protocol and observation adaptation do not require the North SDK or access to the robot's private network.

## 3. Standard inference input

The `observation` returned by the public client is identical to the `observation` in the
`origami-zenoh-v1/infer` request received by the production image:

```python
{
    "observation/image/head_left":       uint8[224, 224, 3],
    "observation/image/head_right":      uint8[224, 224, 3],
    "observation/image/wrist_left":      uint8[224, 224, 3],
    "observation/image/wrist_right":     uint8[224, 224, 3],
    "observation/state":                 float32[65],
    "observation/state/joint_torque":    float32[65],
    "observation/tactile":               float32[60],
    "observation/image/tactile_deform":  uint8[480, 1200, 3],
    "observation/image/tactile_raw":     uint8[480, 1600, 3],  # optional
    "prompt":                            str,
}
```

The four camera images are HWC, RGB, C-contiguous, with values in the range `0..255`. The organizer
directly stretches the native `1920x1536` camera frames to `224x224`; this operation does not preserve
the aspect ratio and adds no padding or letterboxing, matching the square transformation used for the
training data. Teams must not add black bars based on the original camera aspect ratio.

`observation/state` contains 65-dimensional absolute joint angles in radians. See
`docs/robot_io_spec.md` for the exact order. That document also defines joint torque, tactile force,
and the tactile image grid. The image handler must accept every field, even if the model uses only a subset.

The local development workflow and image service should reuse the same adapter:

```python
class TeamPolicyAdapter:
    def reset(self) -> None:
        ...

    def infer(self, observation: dict) -> dict:
        model_input = self.preprocess(observation)
        model_output = self.model(model_input)
        actions = self.to_absolute_radian_actions(model_output)
        return {"actions": actions}
```

Do not maintain separate preprocessing, normalization, or joint mapping implementations for public
testing and the image service.

## 4. Production image output requirements

A successful inference response must contain:

```python
{
    "actions": np.ndarray((T, 65), dtype=np.float32)
}
```

Requirements:

- `T` must match the fixed `action_horizon` in the metadata;
- Every value must be finite; NaN and Inf are not allowed;
- Each row must be an absolute joint-position target;
- Values must be in radians;
- The order of the 65 columns must exactly match `observation/state` and the metadata `joint_names`;
- Any normalized, delta, velocity, tokenized, or padded actions used internally by the model must be
  denormalized and semantically converted inside the image.

For example, if OpenPI internally outputs 72 dimensions, the image must map them to and return the 65
physical dimensions. Padding must not cross the protocol boundary.

## 5. Building the production image

The image must be self-contained and include:

- Model code and a fixed entrypoint;
- Checkpoints/weights;
- Normalization statistics;
- Tokenizer/vocabulary;
- Preprocessing, postprocessing, and the 65-dimensional joint mapping;
- Zenoh, the MessagePack codec, and all runtime dependencies;
- The CUDA/framework runtime;
- Third-party licenses and notices.

At runtime, no team source code, checkpoint, configuration, or host Python environment will be
mounted, and no internet downloads will be available. The image must run with a read-only root
filesystem, a non-root user, dropped capabilities, and an isolated network.

After startup, the image connects to `ORIGAMI_ZENOH_ENDPOINT` in Zenoh client mode and declares only:

```text
origami-zenoh-v1/metadata
origami-zenoh-v1/reset
origami-zenoh-v1/infer
```

The image does not need to `EXPOSE` a port and must not use HTTP `/healthz`. It also must not contain
the North SDK, robot topics, an action publisher, a remote observation token, an SSH key, or cloud credentials.

For the production server template, generic Dockerfile, and entrypoint instructions, see:

- `sharpa_north_ces_lite_sdk-main/examples/policy_server_template.py`;
- Sections 9-10 of `docs/competition_participant_complete_guide.md`;
- `openpi-base-main/` for the OpenPI inference/model reference implementation.

OpenPI teams should use
`openpi-base-main/scripts/docker/submission-zenoh-bundled.Dockerfile`; its exact
named-context build command is documented in
`sharpa_north_ces_lite_sdk-main/scripts/README.md`.

## 6. Running synthetic black-box validation before reservation

First, follow Section 13 of `docs/competition_participant_complete_guide.md` to start the local Zenoh
router and final image. Then run:

```bash
cd sharpa_north_ces_lite_sdk-main
uv run --no-sync python examples/check_zenoh_policy.py \
  --endpoint tcp/127.0.0.1:17447 \
  --session-id local-contract-test \
  --timeout 180 \
  --requests 3 \
  --expected-horizon 25
```

Replace `25` with the image's actual fixed horizon.

A lightweight mock can use the default 10-second timeout. For a final image that loads GPU weights or
performs JIT/XLA compilation, increase the timeout to 180 seconds or the competition's published value,
as appropriate for the actual cold-start time. Record both cold-start and steady-state latency.

The validator checks:

- Metadata, all 65 joint names, and the horizon;
- Reset;
- A complete synthetic observation, including all four RGB images, torque, and tactile data;
- The request/reply envelope and unique request ID;
- `finite float32[T,65]`;
- Repeated requests and query latency.

`PASS` confirms that the image is protocol-compatible, but it does not confirm model performance on
the task. You must still reserve access to physical-robot observations for validation.

## 7. Reading public physical-robot observations after reservation

Store the CA file sent by the organizer in a location readable only by the current user. The endpoint,
session, and CA path may be set as environment variables. Enter the token through a hidden prompt to
keep it out of shell history:

```bash
export ORIGAMI_REMOTE_ENDPOINT='tls/<public-host>:<port>'
export ORIGAMI_REMOTE_SESSION_ID='<assigned-team-session>'
export ORIGAMI_REMOTE_TLS_CA='/absolute/path/to/organizer-ca.pem'
read -rsp 'ORIGAMI remote token: ' ORIGAMI_REMOTE_TOKEN
export ORIGAMI_REMOTE_TOKEN
printf '\n'
```

Replace the placeholders with the actual values sent by the organizer, then check connectivity:

```bash
cd sharpa_north_ces_lite_sdk-main
uv run --no-sync python examples/remote_observation_client.py
```

Successful output includes:

```text
PASS: received policy-compatible observation
state=(65,)
cameras=[(224, 224, 3), (224, 224, 3), (224, 224, 3), (224, 224, 3)]
tactile_deform=(480, 1200, 3)
```

Validate the standard adapter directly in Python:

```python
import numpy as np

from examples.remote_observation_client import RemoteObservationClient

with RemoteObservationClient(
    endpoint=endpoint,
    session_id=session_id,
    token=token,
    tls_root_ca_certificate=ca_path,
) as client:
    observation = client.get_observation()
    result = adapter.infer(observation)

actions = np.asarray(result["actions"])
assert actions.dtype == np.float32
assert actions.ndim == 2 and actions.shape[1] == 65
assert np.isfinite(actions).all()
```

The public interface never sends local predictions to the robot. It provides no action, prediction,
or robot-control API.

## 8. Testing the final image with physical-robot observations

The local Shadow evaluator in the public package can send the same public observation to the final image:

```bash
cd sharpa_north_ces_lite_sdk-main
uv run --no-sync python -m participant_local_evaluator \
  --robot-assets-dir /absolute/path/to/organizer-north-assets
```

Open the following URL in a browser:

```text
http://127.0.0.1:7861
```

On the page:

1. Enter the endpoint, session, token, and TLS CA sent by the organizer;
2. Enter the local final image tag, or the `.tar.zst` path and SHA-256 checksum;
3. Start the image and execute reset;
4. Run Shadow to inspect images, state, actions, compatibility checks, and the URDF prediction trajectory.

This platform visualizes predictions locally only and never sends actions to the physical robot. The
public token is also never injected into the team's image.

## 9. Exporting the submission package

```bash
IMAGE='team-name/origami-policy:submission'
ARCHIVE='team-name-origami-policy.tar'

docker save -o "$ARCHIVE" "$IMAGE"
zstd -T0 -19 "$ARCHIVE"
sha256sum "${ARCHIVE}.zst" > "${ARCHIVE}.zst.sha256"
zstd -t "${ARCHIVE}.zst"
```

Follow the competition announcement for submission contents and naming rules. At minimum, retain:

- The `.tar.zst` image archive;
- Its SHA-256 checksum;
- The image ID/tag;
- The action horizon;
- GPU/CPU/RAM/shared-memory requirements;
- Third-party licenses and notices.

Before submission, run `docker load` on a clean machine, restart the image, and pass the public validator again.

## 10. Final checklist

- [ ] The local adapter can process `RemoteObservationClient.get_observation()` directly;
- [ ] All four camera images are received as HWC RGB `uint8[224,224,3]` with no additional letterboxing;
- [ ] State, joint torque, tactile data, and tactile image grid shapes and dtypes are correct;
- [ ] State is finite `float32[65]` in radians;
- [ ] Actions are finite `float32[T,65]` absolute values in radians;
- [ ] All three queryables—metadata, reset, and infer—pass the validator;
- [ ] The image contains the code, weights, normalization statistics, tokenizer, and all dependencies;
- [ ] The image does not depend on host mounts or runtime downloads;
- [ ] The image runs in a read-only, non-root, offline environment;
- [ ] The image contains no `ORIGAMI_REMOTE_*` credentials, certificate private keys, or robot communication code;
- [ ] The `.tar.zst` archive integrity and SHA-256 checksum have been verified.

## 11. Authoritative documentation

- Complete workflow: `docs/competition_participant_complete_guide.md`
- Production Zenoh wire protocol: `docs/participant_zenoh_submission.md`
- Observation/action tensor contract: `docs/robot_io_spec.md`
- Container submission requirements: `docs/container_submission.md`
- Public, read-only interface: `docs/remote_participant_development.md`

This public package does not include the organizer's physical-robot deployment, the robot's private
network, an action publisher, historical chat records, or legacy WebSocket service code.
