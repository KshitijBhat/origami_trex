# Public Read-Only Real-Robot Observation Development Interface

This interface lets participating teams read real-robot observations through a
public Zenoh endpoint provided by the organizer during development and validate
inference code on their own machines. It uses the same observation object as the
`infer` input received by the final submitted image.

This is not a robot control interface. The public service declares only one
queryable and provides no action, prediction, robot topic, North SDK, or
publishing access.

## Two Independent Workflows

### 1. Organizer's local image evaluation

After receiving a participant image, the organizer runs it in an isolated
container. The organizer's environment supplies observations:

- Shadow: displays the predicted trajectory on the URDF without sending
  real-robot actions;
- Live: enabled only by the organizer under controlled conditions and executed
  after safety checks.

The participant image follows `origami-zenoh-v1` and declares the three
`metadata`, `reset`, and `infer` queryables. See
`participant_zenoh_submission.md`.

### 2. Public read-only development for participants

The organizer assigns each team:

```text
ORIGAMI_REMOTE_ENDPOINT=tls/<public-host>:<port>
ORIGAMI_REMOTE_SESSION_ID=<opaque-team-session>
ORIGAMI_REMOTE_TOKEN=<per-team-secret>
ORIGAMI_REMOTE_TLS_CA=/path/to/organizer-ca.pem
```

Public endpoints and credentials are not included in the public code package.
The organizer sends them separately for each team's reserved development slot
and may revoke or rotate them after the reservation ends. Before receiving
credentials, participants can still build their images and use the synthetic
validator. After receiving credentials, they can validate real-robot
observations and the final image.

Participants obtain observations through `origami-remote-v1`. This interface
cannot control the robot and does not replace the formal image submission. Both
workflows share exactly the same inference input schema.

## Minimal Usage

After installing the SDK project's frozen dependencies:

```python
from examples.remote_observation_client import RemoteObservationClient

client = RemoteObservationClient(
    "tls/<public-host>:<port>",
    session_id="<assigned-team-session>",
    token="<assigned-team-secret>",
    tls_root_ca_certificate="/secure/origami-ca.pem",
)

obs = client.get_observation()
result = policy.infer(obs)
```

`obs` can be passed unchanged to the participant's inference code. The template
does not import the robot SDK and contains no robot addresses, internal topics,
or action code.

Using a context manager is recommended:

```python
with RemoteObservationClient(
    endpoint,
    session_id=session_id,
    token=token,
    tls_root_ca_certificate=ca_path,
) as client:
    while developing:
        observation = client.get_observation()
        prediction = policy.infer(observation)
        visualize_or_record_locally(prediction)
```

`prediction` exists only on the participant's machine. The remote template does
not send it back to the robot.

## Command-Line Connectivity Check

Do not put the token in shell history. Pass it through environment variables:

```bash
cd sharpa_north_ces_lite_sdk-main

export ORIGAMI_REMOTE_ENDPOINT='tls/<public-host>:<port>'
export ORIGAMI_REMOTE_SESSION_ID='<assigned-team-session>'
export ORIGAMI_REMOTE_TLS_CA='/secure/origami-ca.pem'
read -rsp 'ORIGAMI remote token: ' ORIGAMI_REMOTE_TOKEN
export ORIGAMI_REMOTE_TOKEN
printf '\n'

uv sync --frozen --no-install-project
uv run --no-sync python examples/remote_observation_client.py
```

## Participant Local Shadow Image Evaluator

`participant_local_evaluator` connects public read-only observations to the
participant's formal submission image on the participant's machine. It does not
depend on the North SDK and cannot send real-robot commands:

```text
origami-remote-v1 (public, read-only)
  -> local Python backend
  -> isolated Docker network + trusted Zenoh router
  -> participant image origami-zenoh-v1 metadata/reset/infer
  -> local validation and North URDF player for the next 100 steps
```

First extract the official North asset package supplied by the organizer onto
the participant's machine. The directory must contain at least:

```text
<north-assets>/
  urdf/north_poc2_2_with_hand_description.urdf
  meshes/...
```

Start the evaluator:

```bash
cd sharpa_north_ces_lite_sdk-main
uv sync --frozen --no-install-project
uv run --no-sync python -m participant_local_evaluator \
  --robot-assets-dir /absolute/path/to/north-assets
```

By default, it listens only at:

```text
http://127.0.0.1:7861
```

Web interface workflow:

1. Enter the public endpoint, session, token, and TLS CA path, then click
   "Connect and Read One Frame";
2. Select the competition submission `.tar.zst` file to stream it to the local
   backend, or enter an absolute local path and the required matching SHA-256.
   The image tag is filled automatically after loading;
3. Start the image. The evaluator creates an internal Docker network, a trusted
   Zenoh router, and the team container;
4. After Reset, run Shadow. The evaluator obtains a fresh real observation and
   calls the image's `infer` locally;
5. Review the four RGB images, current 65-dimensional state, complete Three.js
   URDF/STL Current/Predicted 3D model, compatibility, and timeline.

If a model's horizon is shorter than 100, the evaluator performs multiple
open-loop chunks. Each new chunk uses the final step of the previous chunk as
the local `observation/state`, while images and the prompt remain from the same
remote snapshot, until at most the requested 100 steps are produced. This
process exists only on the participant's machine.

### Local container boundary

The team container uses:

- an internal Docker network, with access only to the trusted router on that
  network;
- `--read-only`, `--cap-drop ALL`, and `no-new-privileges`;
- `--user 65532:65532`;
- `/tmp` and `/run` tmpfs mounts;
- CPU, RAM, shared-memory, and PID limits;
- only `ORIGAMI_ZENOH_ENDPOINT` and `ORIGAMI_SESSION_ID` injected.

The public endpoint, session, and token are never passed into the team image.
The token is stored only in the local Python backend's memory. HTTP status,
errors, logs, and responses never echo the token, and the page clears the token
input immediately after submission. The browser does not run a Zenoh client.

The evaluator's POST routes are limited to remote connection, archive streaming
upload/local-path loading, image start/stop, policy reset, and Shadow inference.
There is no route for real-robot execution or command publication. URDF and mesh
files are served only by extension from beneath `--robot-assets-dir`; absolute
paths, `..`, and symlink escapes are rejected.

### Validation levels

The evaluator always strictly checks that output is finite `float32[T,65]` and
that each chunk horizon matches metadata. If all 65 contract joints in the
official URDF provide parsable `lower`, `upper`, and `velocity` values, it also
checks position, jumps between adjacent steps, and velocity calculated from the
page's Hz setting. If assets are missing or the URDF is incomplete, the page
explicitly shows the downgraded `shape-finite` level instead of presenting it as
full URDF compatibility.

If the organizer enables mTLS, also set:

```bash
export ORIGAMI_REMOTE_TLS_CERT='/secure/team-a-cert.pem'
export ORIGAMI_REMOTE_TLS_KEY='/secure/team-a-key.pem'
```

`tcp/127.0.0.1:<port>` may be used for local closed-network debugging. Formal
public deployment must use `tls/`, a trusted CA, and a separate token for each
team. The organizer may additionally require mTLS.

## Wire Contract

Each team has exactly one read-only key:

```text
origami-remote-v1/{session_id}/observation
```

Requests use MessagePack with the repository's msgpack-numpy encoding:

```python
{
    "protocol_version": "origami-remote-v1",
    "operation": "observation",
    "request_id": unique_string,
    "session_id": assigned_session_id,
    "token": assigned_token,
}
```

A successful reply echoes four correlation fields:

```python
{
    "protocol_version": "origami-remote-v1",
    "operation": "observation",
    "request_id": request_id,
    "session_id": session_id,
    "observation_timestamp": unix_seconds,
    "metadata": {
        "protocol_version": "origami-v1",
        "observation_schema": "policy-infer-input",
        "observation_fields": {
            "<field-key>": {
                "dtype": "uint8" | "float32" | "str",
                "shape": [...],
                "required": bool,
            },
            # Exact entries match docs/robot_io_spec.md.
        },
        "joint_names": [...exact 65 names...],
    },
    "observation": {
        "observation/image/head_left": uint8[224, 224, 3],
        "observation/image/head_right": uint8[224, 224, 3],
        "observation/image/wrist_left": uint8[224, 224, 3],
        "observation/image/wrist_right": uint8[224, 224, 3],
        "observation/state": float32[65],
        "observation/state/joint_torque": float32[65],
        "observation/tactile": float32[60],
        "observation/image/tactile_deform": uint8[480, 1200, 3],
        "observation/image/tactile_raw": uint8[480, 1600, 3],  # optional
        "prompt": str,
    },
}
```

This `observation` is field-for-field identical to the `observation` field in
the `origami-zenoh-v1/infer` request received by the formal image.

The organizer stretches each of the four RGB images directly from its native
`1920x1536` resolution to `224x224`, without preserving aspect ratio or adding
padding/letterboxing. Participants receive the final public observation and
must not add black bars based on the original camera aspect ratio.

The template strictly validates:

- the reply envelope and unique request ID;
- the shape and dtype of the four RGB and tactile deform/raw images;
- the shape, dtype, and absence of NaN/Inf in state, joint torque, and the
  tactile vector;
- the semantic protocol and exact joint order;
- the observation Unix timestamp and freshness;
- the 64 MiB payload limit and safe NumPy dtypes.

## Errors and Rate Limits

Application errors use a structured reply:

```python
{
    # correlation envelope...
    "error": {
        "code": "UNAUTHORIZED" | "RATE_LIMITED" | "BUSY"
                | "OBSERVATION_UNAVAILABLE" | "INVALID_REQUEST",
        "message": "...",
        "retryable": True | False,
    },
}
```

`RATE_LIMITED`, `BUSY`, and transient `OBSERVATION_UNAVAILABLE` errors may be
retried with backoff. Do not bypass rate limits by increasing request
concurrency. The interface intentionally permits only one concurrent request so
public clients cannot interfere with local real-robot evaluation.

## Security Boundary

- Tokens must be unique per team, revocable, and rotatable, and must never be
  written into images, Git, or logs;
- The organizer relay reads the latest observation only after authentication
  and rate-limit checks pass;
- The relay calls only a read-only snapshot and neither holds nor calls an
  action publisher;
- The organizer exposes only the TLS relay, not internal robot services or the
  evaluation administration interface;
- Production deployments should retain access auditing, connection limits, and
  bandwidth limits before and after the relay;
- Receiving observations does not grant participants control of the real robot.

The only read implementation participants need is
`sharpa_north_ces_lite_sdk-main/examples/remote_observation_client.py`. The
organizer maintains server deployment and robot integration.
