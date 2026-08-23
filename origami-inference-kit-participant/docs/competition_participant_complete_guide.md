# Origami Complete Participant Guide to Development, Inference Interfaces, and Image Submission

This is an end-to-end guide for participants, covering everything from reading real-robot
observations to submitting a complete Docker/OCI image. Its goal is to help teams complete
the following workflow:

```text
Public-network read-only observation
  -> Local inference code
  -> Input/output adapter validation
  -> Self-contained Docker image
  -> origami-zenoh-v1 black-box validation
  -> .tar.zst + SHA-256 submission
  -> Organizer-run Shadow/URDF evaluation
  -> Live real-robot evaluation only after organizer authorization
```

In this guide, "must" denotes a protocol or submission requirement; "recommended" denotes
an engineering practice that can significantly reduce failures during evaluation.

Authoritative related files:

- `docs/participant_zenoh_submission.md`: the production-image Zenoh wire protocol;
- `docs/robot_io_spec.md`: images, state, actions, and the 65-joint ordering;
- `docs/container_submission.md`: container runtime and security constraints;
- `docs/remote_participant_development.md`: the public-network read-only development interface;
- `sharpa_north_ces_lite_sdk-main/examples/check_zenoh_policy.py`: the public black-box validator;
- `sharpa_north_ces_lite_sdk-main/examples/remote_observation_client.py`: a template for reading
  real-robot observations;
- `sharpa_north_ces_lite_sdk-main/examples/policy_server_template.py`: a framework-independent
  production Zenoh server template.

---

## 1. Understand the Two Independent Interfaces First

The system has two Zenoh protocols for different purposes. They share the same inference
observation schema, but differ in network direction, identity, and permissions.

### 1.1 Participant development: `origami-remote-v1`

Purpose: participants obtain real-robot observations read-only through the organizer's
public-network Relay and run their models on their own development machines.

The organizer provides:

```text
ORIGAMI_REMOTE_ENDPOINT=tls/<public-host>:<port>
ORIGAMI_REMOTE_SESSION_ID=<team-session>
ORIGAMI_REMOTE_TOKEN=<team-secret>
ORIGAMI_REMOTE_TLS_CA=/path/to/organizer-ca.pem
```

These public-network parameters are not stored in the public repository. The organizer
sends them separately based on each team's reserved time slot and may revoke or rotate
them after the reservation ends. They are not required to build the image or run the
synthetic validator before the reservation.

Read-only queryable:

```text
origami-remote-v1/{session_id}/observation
```

This interface provides no action, prediction, robot topic, or robot SDK. Predictions
generated locally by participants are not sent back to the robot through this interface.

### 1.2 Production submission image: `origami-zenoh-v1`

Purpose: the organizer starts the participant image and sends an observation into it as
an `infer` request; the image returns a predicted action chunk.

The organizer injects only the following into the container:

```text
ORIGAMI_ZENOH_ENDPOINT=tcp/<isolated-router>:7447
ORIGAMI_SESSION_ID=<opaque-evaluation-session>
```

The image must declare, and may depend only on, the following fixed queryables:

```text
origami-zenoh-v1/metadata
origami-zenoh-v1/reset
origami-zenoh-v1/infer
```

The production image must not connect to the public-network development Relay and must
never package `ORIGAMI_REMOTE_TOKEN`. Development Relay credentials are unrelated to the
production image runtime environment.

### 1.3 Why inputs can be unified across both interfaces

The organizer's public-network read-only service and production evaluation service use
the same observation construction rules:

```text
North raw observation
  -> build_policy_observation(...)
  -> {
       four uint8[224,224,3] camera images,
       state / torque / tactile,
       tactile deform/raw image grids,
       prompt
     }
```

Participants should therefore make the local adapter and in-image adapter share the same
entry function instead of maintaining two preprocessing implementations.

---

## 2. Obtain the Development Package and Prepare the Python Environment

Enter the public SDK project:

```bash
cd /path/to/origami-inference-kit/sharpa_north_ces_lite_sdk-main
```

Using the project's lockfile to create the environment is recommended:

```bash
uv sync --frozen --no-install-project
```

Run the public tests:

```bash
uv run --no-sync python -m unittest discover -s tests -v
```

The robot SDK is not required when working only on protocol adaptation. The core packages
required by the public templates are:

```text
eclipse-zenoh
msgpack
numpy
```

Participants should pin versions independently for dependencies such as their model
framework, CUDA, PyTorch/JAX, and tokenizer, and ultimately include them in the submission
image.

---

## 3. Use Public-Network Real-Robot Observations

### 3.1 Command-line connectivity check

Do not write the token to Git, scripts, a Dockerfile, or shell history. Temporary
environment variables are recommended:

```bash
export ORIGAMI_REMOTE_ENDPOINT='tls/<public-host>:<port>'
export ORIGAMI_REMOTE_SESSION_ID='<assigned-team-session>'
export ORIGAMI_REMOTE_TLS_CA='/secure/origami-ca.pem'
read -rsp 'ORIGAMI remote token: ' ORIGAMI_REMOTE_TOKEN
export ORIGAMI_REMOTE_TOKEN
printf '\n'

uv run --no-sync python examples/remote_observation_client.py
```

If mTLS is enabled:

```bash
export ORIGAMI_REMOTE_TLS_CERT='/secure/team-a-cert.pem'
export ORIGAMI_REMOTE_TLS_KEY='/secure/team-a-key.pem'
```

Successful output should include:

```text
PASS: received policy-compatible observation
state=(65,)
cameras=[(224, 224, 3), (224, 224, 3), (224, 224, 3), (224, 224, 3)]
tactile_deform=(480, 1200, 3)
```

`tcp/` is suitable only for the same machine or a closed test network. Public-network
access must use `tls/` and a trusted CA.

### 3.2 Python template

```python
from examples.remote_observation_client import RemoteObservationClient

with RemoteObservationClient(
    endpoint="tls/<public-host>:<port>",
    session_id="<assigned-team-session>",
    token="<assigned-secret>",
    tls_root_ca_certificate="/secure/origami-ca.pem",
) as client:
    observation = client.get_observation()
    result = policy.infer(observation)
```

Notes:

- `policy` is the participant's own local adapter;
- `get_observation()` strictly checks the protocol, timestamp, joint names, shape, dtype,
  and finiteness;
- do not rename public keys after reading them and then use different entry points in the
  development and image versions;
- do not include `RemoteObservationClient` in the production submission image;
- do not attempt to connect local predictions to the robot.

### 3.3 Participant-local Shadow image evaluation platform

The public kit provides a standalone `participant_local_evaluator` package. On the
participant's machine, it performs:

```text
Public-network real read-only observation
  -> Locally isolated Docker network + Zenoh router
  -> Production submission image metadata/reset/infer
  -> finite float32[T,65] validation
  -> Multi-chunk open-loop rollout of up to 100 steps
  -> North URDF Current/Predicted timeline
```

It does not require the North SDK, contains no real-robot command publisher, and provides
no real-robot execution API or button.

Extract the North asset package published by the organizer locally, then start:

```bash
cd sharpa_north_ces_lite_sdk-main
uv run --no-sync python -m participant_local_evaluator \
  --robot-assets-dir /absolute/path/to/north-assets
```

Open:

```text
http://127.0.0.1:7861
```

On the web page, enter:

- the public-network endpoint, session, token, and TLS CA provided by the organizer;
- the tag of an already loaded local Docker image; or
- the absolute path to a local `.tar.zst` and the SHA-256 that it must match.

The platform injects only `ORIGAMI_ZENOH_ENDPOINT` and `ORIGAMI_SESSION_ID` into the team
image. The remote token is retained only in local backend memory: it is not passed into
the image, written to disk, or echoed in status, errors, or logs. The browser clears the
token field after submission and does not connect directly to Zenoh.

The container uses an internal network, read-only rootfs, non-root UID/GID, cap-drop,
no-new-privileges, tmpfs, and CPU/RAM/PID limits. Each Shadow run obtains a fresh real
observation frame. If the policy horizon is shorter than the requested number of steps,
subsequent chunks locally update only `observation/state` using the final state of the
previous chunk; the same image frame and prompt remain unchanged.

Output is always checked as strict finite `float32[T,65]`. When the official URDF can be
fully parsed, position, jump, and velocity are checked as well; otherwise, the page
clearly indicates `shape-finite` only. The URDF/mesh service confines paths to
`--robot-assets-dir` and rejects traversal and symlinks that escape it.

See `docs/remote_participant_development.md` for detailed instructions, security
boundaries, and known limitations.

---

## 4. Exact Inference Input Contract

`observation` uses the complete fixed schema:

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

Only `observation/image/tactile_raw` may be omitted. A model may choose not to consume
certain fields, but the public handler must accept the complete object.

### 4.1 Four RGB images

Each image must satisfy:

```text
dtype:  uint8
shape:  (224, 224, 3)
layout: HWC
channel order: RGB
value range: 0..255
wire memory: C-contiguous
```

The organizer directly stretches native `1920x1536` camera frames to `224x224`. This
conversion does not preserve the aspect ratio and adds no padding or letterboxing,
matching the square conversion used for training data. Participants must not add black
borders again based on the original camera aspect ratio.

Do not use any of the following directly as public input:

- OpenCV BGR;
- CHW;
- `float32` in the range `0..1` or `-1..1`;
- JPEG/PNG bytes;
- `(1,224,224,3)` with a batch dimension;
- the original camera resolution;
- undeclared camera streams.

If the model internally requires CHW, batching, or normalization, convert it inside the
adapter:

```python
def preprocess_rgb(image: np.ndarray) -> np.ndarray:
    assert image.dtype == np.uint8
    assert image.shape == (224, 224, 3)
    value = image.astype(np.float32) / 255.0
    return np.transpose(value, (2, 0, 1))[None, ...]
```

Do not modify the public observation itself; construct a new model tensor.

### 4.2 65-dimensional state

It must satisfy:

```text
dtype: float32
shape: (65,)
units: radians
meaning: current absolute joint angles
all values: finite
```

The grouped slices are fixed as follows:

```python
left_arm  = state[0:7]    # 7
left_hand = state[7:29]   # 22
right_arm = state[29:36]  # 7
right_hand = state[36:58] # 22
motor = state[58:65]      # 7
```

See `docs/robot_io_spec.md` for the exact 65 joint names and their order. `JOINT_NAMES` in
the public template contains the same authoritative ordering.

Do not:

- swap left and right;
- sort names;
- remove apparently stationary motor/head/torso dimensions;
- add model padding;
- use degrees;
- replace true radians with normalized state;
- use NaN to represent missing joints.

### 4.3 Torque and tactile

- `observation/state/joint_torque` is `float32[65]` and uses the same joint ordering as state;
- `observation/tactile` contains 10 fingertips x a 6-dimensional wrench, for a total of
  `float32[60]`;
- the deform grid is `uint8[480,1200,3]`;
- the raw tactile grid is optional `uint8[480,1600,3]`.

See `docs/robot_io_spec.md` for detailed slices, fingertip ordering, and zero-padding rules.

### 4.4 Prompt

`prompt` is always a Python `str` and may be empty. Language tokenization, default prompts,
embeddings, and caching are all internal to the model. If the model caches the prompt,
episode-scoped state must be cleared on `reset`.

---

## 5. Recommended Model Adapter Boundary

The local development environment and image service should call the same function:

```python
class TeamPolicyAdapter:
    def __init__(self, model, normalization):
        self.model = model
        self.normalization = normalization

    def reset(self) -> None:
        # Clear history, RNN state, temporal ensemble, cached prompt, and so on
        ...

    def infer(self, observation: dict) -> dict:
        model_input = self.preprocess(observation)
        model_output = self.model(model_input)
        actions = self.to_public_actions(model_output)
        return {"actions": actions}

    def preprocess(self, observation: dict):
        # public RGB/state/prompt -> framework tensors
        ...

    def to_public_actions(self, model_output) -> np.ndarray:
        # denormalize + semantic conversion + explicit joint mapping
        actions = ...
        actions = np.ascontiguousarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 65:
            raise ValueError(f"expected [T,65], got {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("actions contain NaN or Inf")
        return actions
```

Local development:

```python
observation = remote_client.get_observation()
result = adapter.infer(observation)
```

In-image Zenoh handler:

```python
result = adapter.infer(request["observation"])
```

This ensures that both entry points are identical in preprocessing, normalization, joint
mapping, and output conversion.

---

## 6. Exact Inference Output Contract

In a successful production `infer` reply, `actions` must be:

```text
dtype: float32
shape: (T, 65)
T: the fixed action_horizon declared in metadata
units: radians
meaning: absolute joint-position targets
values: all finite
column order: exactly the same as observation/state
```

For example, with `T=25`:

```python
actions = np.asarray(result["actions"])
assert actions.dtype == np.float32
assert actions.shape == (25, 65)
assert np.isfinite(actions).all()
```

### 6.1 The most common semantic errors

The following are noncompliant even if the shape is correct:

- outputting delta actions;
- outputting velocity or torque;
- outputting degrees;
- outputting `[-1,1]` in the training normalization space;
- outputting token IDs;
- outputting 72-dimensional padded actions;
- removing motor dimensions and zero-padding without semantic mapping;
- using an action column order inconsistent with state.

The following must be completed inside the image:

```text
model-native output
  -> decode/sample
  -> denormalize
  -> required delta/velocity-to-absolute-radian conversion
  -> explicit model-joint -> protocol-joint mapping
  -> finite float32[T,65]
```

If the model is internally 72-dimensional, it may be sliced with `[:, :65]` only after
confirming that the first 65 dimensions already match the protocol ordering column by
column. Otherwise, an explicit mapping must be created from joint names; blind slicing is
not permitted.

### 6.2 Horizon

`T` is fixed for the lifetime of the process:

```python
metadata["action_horizon"] == actions.shape[0]
```

The organizer may execute only the first few steps before running inference again, or may
perform an open-loop rollout over multiple chunks in Shadow to produce 100 future steps.
A participant policy must not assume that the entire chunk is executed; the next
observation is the authoritative current state.

---

## 7. Production Image Zenoh Protocol

### 7.1 Session configuration

The image reads:

```python
endpoint = os.environ["ORIGAMI_ZENOH_ENDPOINT"]
session_id = os.environ["ORIGAMI_SESSION_ID"]
```

Zenoh must:

- use client mode;
- connect to the injected endpoint;
- not depend on multicast scouting;
- not depend on shared-memory transport;
- not hard-code an IP address, port, or session;
- not declare any other publisher, subscriber, or wildcard queryable.

Reference configuration:

```python
config = zenoh.Config()
config.insert_json5("mode", '"client"')
config.insert_json5("connect/endpoints", json.dumps([endpoint]))
config.insert_json5("scouting/multicast/enabled", "false")
config.insert_json5("transport/shared_memory/enabled", "false")
session = zenoh.open(config)
```

### 7.2 Envelope

Each request contains:

```python
{
    "protocol_version": "origami-zenoh-v1",
    "operation": "metadata" | "reset" | "infer",
    "request_id": unique_string,
    "session_id": injected_session_id,
}
```

The reply must echo all four correlation fields unchanged. It must not generate a new
request ID or reuse a reply from an old request for a new observation.

### 7.3 `metadata`

A successful reply contains at least:

```python
{
    # envelope...
    "metadata": {
        "protocol_version": "origami-v1",
        "action_dim": 65,
        "action_type": "absolute_joint_position",
        "action_units": "radians",
        "action_horizon": T,
        "joint_names": JOINT_NAMES,
    },
}
```

Note that the outer transport version is `origami-zenoh-v1`, while the inner semantic
version is `origami-v1`. They are distinct.

### 7.4 `reset`

The organizer calls this before each episode. Before replying, the image must clear:

- the observation/history buffer;
- RNN/recurrent state;
- the temporal ensemble;
- the action queue;
- the prompt cache;
- episode random state.

Successful reply:

```python
{
    # envelope...
    "ok": True,
}
```

### 7.5 `infer`

Request:

```python
{
    # envelope...
    "observation": {
        # All observation fields from Section 4
    },
}
```

Reply:

```python
{
    # envelope...
    "actions": np.ndarray(shape=(T, 65), dtype=np.float32),
    # Optional: metrics containing no secrets
}
```

Each query must receive exactly one reply. Actions must not be sent asynchronously through
a publisher.

### 7.6 MessagePack NumPy

Array encoding:

```python
{
    b"__ndarray__": True,
    b"data": array.tobytes(order="C"),
    b"dtype": array.dtype.str,
    b"shape": array.shape,
}
```

Requirements:

- payload no larger than 64 MiB;
- no pickle;
- no object, structured, or complex dtype;
- validate shape, dtype, and byte length before and after decoding;
- use C-contiguous arrays for images and state/actions.

Refer directly to:

```text
sharpa_north_ces_lite_sdk-main/examples/policy_server_template.py
```

This file includes the codec, envelope, queryables, strict input/output validation, a
serialized inference lock, and SIGTERM cleanup. Participants need only replace model
loading, reset, and infer in `TeamPolicy`.

### 7.7 Structured errors

Reply on failure:

```python
{
    # envelope...
    "error": {
        "code": "INVALID_REQUEST" | "NOT_READY" | "INFERENCE_FAILED" | "OVERLOADED",
        "message": "non-secret detail",
        "retryable": False | True,
    },
}
```

Do not:

- hang until timeout;
- return stale actions;
- conceal errors with fake all-zero actions;
- print tokens, checkpoint secrets, or complete observations in error messages or logs.

---

## 8. Required Submission Image Contents

The final image must be self-contained and include:

- model code and architecture;
- checkpoints/weights;
- normalization statistics;
- tokenizer/vocabulary;
- preprocessing/postprocessing;
- the 65-dimensional joint mapping;
- the Zenoh and MessagePack codec;
- a fixed entrypoint;
- CUDA/runtime/shared libraries;
- Python packages;
- third-party licenses/notices.

At runtime, the organizer will not:

- mount a checkpoint;
- mount participant source code;
- provide host Python site-packages;
- provide model configuration;
- provide internet downloads;
- provide a robot SDK;
- expose the Docker socket or host network.

If the image cannot start after host source code and checkpoint mounts are removed, it is
not a complete submission image.

---

## 9. Recommended Project Structure

```text
team-submission/
  Dockerfile
  requirements.lock
  entrypoint.sh
  policy_server.py
  team_policy/
    adapter.py
    model.py
    preprocessing.py
  assets/
    checkpoint-or-weights/
    norm_stats.json
    tokenizer/
  licenses/
  .dockerignore
```

`.dockerignore` should exclude at least:

```text
.git
.venv
__pycache__
*.pyc
tests/output
datasets
wandb
logs
secrets
*.pem
*.key
```

Do not place remote Relay tokens, SSH keys, cloud credentials, or dataset credentials in
the build context.

---

## 10. Generic Dockerfile Template

The following is only a structural template. CUDA and model-framework versions must match
the team's actual code:

```dockerfile
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp/team-home \
    XDG_CACHE_HOME=/tmp/team-cache

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.lock /app/requirements.lock
RUN python3 -m pip install --no-cache-dir -r /app/requirements.lock

COPY team_policy/ /app/team_policy/
COPY policy_server.py /app/policy_server.py
COPY entrypoint.sh /app/entrypoint.sh
COPY assets/ /opt/team-policy/assets/
COPY licenses/ /licenses/

RUN chmod 0755 /app/entrypoint.sh \
    && chmod -R a+rX /app /opt/team-policy /licenses

USER 65532:65532

HEALTHCHECK NONE
ENTRYPOINT ["/app/entrypoint.sh"]
```

Example fixed entrypoint:

```bash
#!/usr/bin/env bash
set -euo pipefail

: "${ORIGAMI_ZENOH_ENDPOINT:?ORIGAMI_ZENOH_ENDPOINT is required}"
: "${ORIGAMI_SESSION_ID:?ORIGAMI_SESSION_ID is required}"

export HOME="${HOME:-/tmp/team-home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/team-cache}"
mkdir -p "$HOME" "$XDG_CACHE_HOME"

exec python3 /app/policy_server.py
```

### 10.1 Dockerfile considerations

- use a fixed base-image digest to avoid build-day drift;
- pin Python/CUDA/framework versions;
- place weights in an image layer and do not depend on a bind mount;
- ensure all model files are readable by the runtime UID;
- the root filesystem is read-only during evaluation;
- caches, the JAX compilation cache, and temporary files may be written only to `/tmp` or
  `/run`;
- do not require `--privileged` or host IPC/PID/network;
- do not start SSH, Jupyter, a development Dashboard, or any other listener;
- `EXPOSE` is unnecessary because the container initiates the connection to the Zenoh
  router;
- do not use an HTTP health endpoint; the metadata query is the readiness check;
- the entrypoint must `exec` the final process so that SIGTERM is propagated correctly;
- do not print the complete environment from the entrypoint;
- network access may be available during the build stage, but the production runtime must
  be treated as offline.

### 10.2 Large checkpoints

Using a BuildKit named context is recommended to avoid copying large weights into the Git
workspace:

```bash
docker build \
  --build-context checkpoint=/absolute/path/to/checkpoint \
  -t team-name/origami-policy:submission \
  .
```

Dockerfile:

```dockerfile
COPY --from=checkpoint / /opt/team-policy/checkpoint/
```

Note that the named context still writes the checkpoint into the final image; it only
keeps it out of the normal source context. After building, verify that the final image
actually contains every required file.

---

## 11. Framework-Independent Server Template

The public template is located at:

```text
sharpa_north_ces_lite_sdk-main/examples/policy_server_template.py
```

The template implements the production codec, envelope, metadata/reset/infer queryables,
input/output validation, and process-shutdown handling. The default hold-position output
in `TeamPolicy` is only for validating protocol integration; it is not a task policy.
Participants must replace model loading, checkpoint handling, preprocessing, reset, and
inference, and copy all code and assets into their own image.

OpenPI, ACT, Diffusion Policy, or any other framework may be used. This package includes
`openpi-base-main/` as an OpenPI inference and model-adapter reference. Participants must
still provide and package their own model checkpoint, normalization assets, tokenizer,
runtime dependencies, and any team-specific configuration.

The OpenPI reference's formal Zenoh image can be built from the repository root:

```bash
docker build \
  --build-arg RUNTIME_IMAGE=<compatible-openpi-runtime-image> \
  --build-context checkpoint=/absolute/path/to/checkpoint \
  --build-context python_packages=/absolute/path/to/python/site-packages \
  -f openpi-base-main/scripts/docker/submission-zenoh-bundled.Dockerfile \
  -t team-name/origami-openpi:submission \
  openpi-base-main
```

The Python package context must contain Eclipse Zenoh 1.9. The checkpoint context
must contain model parameters and `assets/**/norm_stats.json`.

---

## 12. Build and Static Checks

Build:

```bash
IMAGE='team-name/origami-policy:submission'
docker build --pull --tag "$IMAGE" .
```

Inspect the image:

```bash
docker image inspect "$IMAGE"
docker image inspect --format '{{.Id}}' "$IMAGE"
docker image inspect --format '{{json .Config.Entrypoint}}' "$IMAGE"
docker image inspect --format '{{json .Config.Cmd}}' "$IMAGE"
docker history --no-trunc "$IMAGE"
```

Key checks:

- the Entrypoint is nonempty and does not depend on extra arguments;
- no secrets are written into `ENV`;
- no datasets, Git history, or development caches were copied accidentally;
- weights, norm stats, and the tokenizer are all in the image;
- the image architecture matches the evaluation machine;
- the GPU framework can recognize the target GPU/CUDA driver.

Quickly verify that missing environment variables cause a fast failure:

```bash
docker run --rm "$IMAGE"
```

It should explicitly report a missing `ORIGAMI_ZENOH_ENDPOINT` or `ORIGAMI_SESSION_ID`
rather than hanging silently.

---

## 13. Start the Router, Image, and Black-Box Validator Locally

The following procedure simulates the production Zenoh boundary. Use the router image
pinned in the public documentation:

```bash
ROUTER_IMAGE='eclipse/zenoh@sha256:157965d71e0bfd0a044d76a985ff0e5c306ad3968929168fb9678cd2a7fec23f'
IMAGE='team-name/origami-policy:submission'
SESSION='local-contract-test'

docker network create origami-contract-test

docker run -d --name origami-contract-router \
  --network origami-contract-test \
  -p 127.0.0.1:17447:7447 \
  "$ROUTER_IMAGE" \
  -l tcp/0.0.0.0:7447 \
  --no-multicast-scouting \
  --cfg 'transport/shared_memory/enabled:false'
```

Start the team image under conditions similar to the production sandbox:

```bash
docker run -d --name origami-contract-policy \
  --network origami-contract-test \
  --gpus all \
  --read-only \
  --cap-drop ALL \
  --security-opt no-new-privileges=true \
  --user 65532:65532 \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=4g \
  --tmpfs /run:rw,noexec,nosuid,nodev,size=64m \
  --shm-size 8g \
  --memory 32g \
  --cpus 8 \
  --pids-limit 512 \
  -e ORIGAMI_ZENOH_ENDPOINT=tcp/origami-contract-router:7447 \
  -e ORIGAMI_SESSION_ID="$SESSION" \
  "$IMAGE"
```

View startup logs:

```bash
docker logs -f origami-contract-policy
```

Run the public validator in another terminal:

```bash
cd sharpa_north_ces_lite_sdk-main
uv run --no-sync python examples/check_zenoh_policy.py \
  --endpoint tcp/127.0.0.1:17447 \
  --session-id "$SESSION" \
  --timeout 180 \
  --requests 3 \
  --expected-horizon 25
```

Replace `25` with the image's actual fixed horizon.

The validator checks:

- metadata and all 65 joint names;
- reset;
- a complete synthetic observation: four RGB images, state/torque, tactile, and prompt;
- the envelope and unique request ID;
- exactly one reply;
- `float32[T,65]`;
- finiteness;
- query latency.

Clean up:

```bash
docker rm -f origami-contract-policy origami-contract-router
docker network rm origami-contract-test
```

Passing the synthetic validator demonstrates only interface compatibility, not model task
quality or action safety.

---

## 14. Validate the Adapter with Real Read-Only Observations

Before the production build, validate the local adapter with the public-network template:

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
assert actions.ndim == 2
assert actions.shape[1] == 65
assert np.isfinite(actions).all()
```

Then ensure that the in-image `policy_server.py` imports and uses the same adapter. Do not
copy a slightly different preprocessing implementation into the server file.

A correct validation result using a real observation should satisfy:

```text
input cameras: 4 x uint8[224,224,3]
input state: float32[65]
input torque/tactile: float32[65], float32[60]
input tactile images: uint8[480,1200,3], optional uint8[480,1600,3]
output actions: finite float32[25,65]
```

---

## 15. Offline and Restart Tests

Before the production submission, perform at least:

### 15.1 No host mounts

Ensure that `docker run` has none of the following:

```text
-v source-code:...
-v checkpoint:...
-v ~/.cache:...
--mount type=bind,...
```

Only tmpfs/shm provided by the runner is permitted.

### 15.2 No internet access

Run on a network that can access only the test router. The first inference must not
download a tokenizer, weights, JAX/PyTorch assets, or remote code.

### 15.3 Read-only root

Testing with `--read-only` is mandatory. Explicitly redirect all required cache paths to
`/tmp`. Common OpenPI/JAX variables:

```text
HOME=/tmp/team-home
XDG_CACHE_HOME=/tmp/team-cache
JAX_COMPILATION_CACHE_DIR=/tmp/team-jax-cache
```

### 15.4 Non-root

Test with the production UID/GID:

```text
--user 65532:65532
```

Verify that the checkpoint, Python packages, entrypoint, and licenses are all readable.

### 15.5 Multiple episodes

Test:

```text
metadata
reset
infer x N
reset
infer x N
container restart
metadata
reset
infer
```

Confirm that no history/action/prompt from the previous episode remains after reset.

### 15.6 First compilation

The first inference with JAX/XLA, Torch compile, or TensorRT may be significantly slower.
Record:

- cold-start model loading time;
- first-infer time;
- steady-state infer median/max;
- peak GPU memory usage;
- peak RAM and `/tmp` usage.

Do not assume that the organizer will perform an undeclared warmup before official timing.

---

## 16. Export the `.tar.zst` Submission Package

If the competition website accepts a compressed OCI/Docker archive, `docker save` must be
used, not `docker export`. `docker export` discards image configuration, the Entrypoint,
and layer metadata.

```bash
IMAGE='team-name/origami-policy:submission'
ARCHIVE='team-name-origami-policy-submission.tar.zst'

docker save "$IMAGE" \
  | zstd -T0 -3 -o "${ARCHIVE}.partial"
mv "${ARCHIVE}.partial" "$ARCHIVE"
```

Check the archive and SHA-256:

```bash
zstd -t "$ARCHIVE"
sha256sum "$ARCHIVE" | tee "${ARCHIVE}.sha256"
```

Record the following:

```bash
docker image inspect --format '{{.Id}}' "$IMAGE"
docker image inspect --format '{{json .RepoDigests}}' "$IMAGE"
```

If there is no registry digest, submit at least the image ID, archive SHA-256, and original
tag.

### 16.1 Reload and validate from the archive

In a clean Docker environment or on another machine:

```bash
zstd -dc "$ARCHIVE" | docker load
docker image inspect "$IMAGE"
```

Then repeat the black-box test from Section 13. Only a passing reloaded archive proves that
the competition website's download workflow is viable.

### 16.2 Recommended manifest

```json
{
  "submission_format": "origami-oci-archive-v1",
  "team_id": "team-a",
  "image": "team-name/origami-policy:submission",
  "image_id": "sha256:...",
  "archive": "team-name-origami-policy-submission.tar.zst",
  "archive_size_bytes": 123456789,
  "archive_sha256": "...64 hex...",
  "protocol": "origami-zenoh-v1",
  "action_dim": 65,
  "action_horizon": 25
}
```

The manifest must not contain tokens, certificate private keys, or secrets in internal
paths.

---

## 17. Production Evaluation Boundary

The organizer will validate the archive, start the image in an isolated network and
restricted container, and then perform metadata, reset, real-observation inference, and
Shadow safety checks in sequence. By default, no actions are sent to the real robot; Live
evaluation requires separate organizer authorization. A participant image cannot request
that isolation, Shadow, safety validation, or organizer confirmation be skipped.

---

## 18. Framework-Specific Considerations

### 18.1 OpenPI

- the checkpoint's `params/` or safetensors must be included in the image;
- `assets/**/norm_stats.json` must be included in the image;
- the tokenizer and other remote assets must be downloaded in advance;
- a common internal output dimension is 72, which must be converted to the protocol's 65;
- model configuration should be fixed in the image entrypoint rather than requiring the
  runner to inject it;
- write the JAX cache to `/tmp`;
- the first infer may trigger XLA compilation.

### 18.2 ACT

- confirm that the chunk contains absolute radians, not normalized/delta values;
- the temporal ensemble is episode state and must be cleared on reset;
- the horizon must remain consistent with metadata;
- joint normalization must be inverted before returning.

### 18.3 Diffusion Policy

- sampling noise, the scheduler, and history must all remain inside the image;
- output denormalization must be completed;
- there must be a deterministic rule for whether random state is reset;
- complete the entire chunk within the timeout; placeholder data must not be returned first.

### 18.4 Custom models

The model may internally use any fields and tensor layout, but the public adapter's inputs
and outputs must not change. Do not require the organizer's runtime environment to
understand the team's model name, checkpoint format, or normalization method.

---

## 19. Common Errors and Troubleshooting

### `metadata` timeout

Check:

- whether the entrypoint is running;
- whether the endpoint uses `ORIGAMI_ZENOH_ENDPOINT`;
- whether Zenoh is in client mode;
- whether the wrong network interface was disabled;
- whether the queryable key is exactly fixed;
- whether model loading is stuck or out of memory.

### `session_id does not match`

Do not generate, rewrite, or cache the session ID. Use the current request/injected
session in the reply.

### `joint_names must match`

Use the exact ordering from `robot_io_spec.md`. Do not sort it, remove motor joints, or use
an ordering obtained by traversing the URDF.

### `image must be uint8(224,224,3)`

Check BGR/RGB, CHW/HWC, batching, float normalization, and JPEG bytes.

### `state must be finite float32[65]`

Do not allow NumPy to default to `float64`; do not pass a framework tensor; do not use NaN
padding.

### `actions shape must be (T,65)`

Check model padding, horizon, the batch dimension, and joint mapping.

### The URDF has predictions but does not move

Interface success does not imply sufficient action magnitude. Check:

- whether normalized output is being returned;
- whether delta is being treated as absolute;
- whether all values are close to the current state;
- whether playback is in Predicted mode;
- whether the trajectory is flagged by URDF limit/jump/velocity checks.

### Read-only filesystem error

Move caches and temporary output to `/tmp`; do not write to `/app`, the checkpoint
directory, or `/root` at runtime.

### Permission denied

Reproduce with `--user 65532:65532`; correct file permissions in image layers rather than
requiring the organizer to use root.

### Runtime download failure

Build the checkpoint, tokenizer, norm stats, and hub cache into the image, and test on an
offline network.

### GPU/CUDA error

Record the base image, CUDA runtime, framework wheel, and target driver requirements. Do
not test only under conditions where the development machine already has the host
libraries/cache.

---

## 20. Final Pre-Submission Checklist

Interface:

- [ ] Declare only the three fixed metadata/reset/infer queryables;
- [ ] The outer protocol is `origami-zenoh-v1`;
- [ ] The metadata semantic protocol is `origami-v1`;
- [ ] Every request/reply envelope is strictly correlated;
- [ ] Reset clears episode state;
- [ ] Infer replies exactly once.

Input:

- [ ] Four HWC RGB `uint8[224,224,3]` images;
- [ ] Correct shape/dtype for joint torque, the tactile vector, and tactile image grids;
- [ ] Finite `float32[65]` radians;
- [ ] Prompt is a `str`;
- [ ] Local RemoteObservationClient input goes directly into the same adapter.

Output:

- [ ] Finite `float32[T,65]`;
- [ ] T is consistent with metadata;
- [ ] Absolute joint positions;
- [ ] Radians;
- [ ] All 65 columns are ordered exactly as state/joint_names;
- [ ] Denormalization and padding/layout conversion are complete.

Image:

- [ ] Fixed Entrypoint;
- [ ] Includes code, weights, norm stats, tokenizer, dependencies, and licenses;
- [ ] Does not mount host files;
- [ ] Performs no runtime downloads;
- [ ] Runs with a read-only root;
- [ ] Runs as UID/GID 65532;
- [ ] Runs with cap-drop/no-new-privileges;
- [ ] Writes caches only to `/tmp` and `/run`;
- [ ] Exits on SIGTERM;
- [ ] Contains no robot SDK, topic, action publisher, or development Relay token.

Testing and submission:

- [ ] The public validator passes;
- [ ] Multiple episodes/resets pass;
- [ ] Cold-start and steady-state latency are recorded;
- [ ] The `.tar.zst` is generated with `docker save`;
- [ ] `zstd -t` passes;
- [ ] The SHA-256 is recorded;
- [ ] The validator passes again after `docker load` on a clean machine;
- [ ] Submit the immutable digest or image ID, archive checksum, resource requirements, and
  licenses.

Only after all items above are complete does the image meet the basic requirements for
entry into the organizer's Shadow/URDF evaluation.
