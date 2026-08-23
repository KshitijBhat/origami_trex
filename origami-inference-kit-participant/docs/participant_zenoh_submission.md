# Participant submission protocol (`origami-zenoh-v1`)

This is the normative public interface for Origami competition submissions.
The submitted container is a policy service; it is not a robot controller.
For a Chinese end-to-end workflow covering remote development, model adapters,
Docker builds, validation, archives, and submission checks, start with
`competition_participant_complete_guide.md`; this file remains the normative
wire-protocol authority.

The organizer gives each running submission:

- one Zenoh router endpoint in the form `tcp/<ip-or-DNS-hostname>:<port>`; and
- one opaque session ID.

The only official transport is Zenoh query/reply using protocol version
`origami-zenoh-v1`. This document and `robot_io_spec.md` are the authoritative
competition submission contract.

## 1. Trust boundary

The image must contain everything needed to load and run the policy:

- model architecture and code;
- checkpoint/weights;
- preprocessing and postprocessing;
- normalization statistics, tokenizers, and other runtime assets;
- a fixed entrypoint that starts the Zenoh policy service.

The participant receives no North SDK, robot credentials, robot topic names, or
action publisher. The container must not discover, read, subscribe to, or
publish robot data. It receives observations only as `infer` query payloads and
returns predictions only as query replies. The organizer-owned runtime is
responsible for robot I/O, validation, safety filtering, scheduling, and
execution.

## 2. Zenoh session

The service must open a Zenoh session and connect to the endpoint supplied by
the runner. Multicast discovery must not be required. The container environment
variables are exactly:

```text
ORIGAMI_ZENOH_ENDPOINT=tcp/<ip-or-DNS-hostname>:<port>
ORIGAMI_SESSION_ID=<opaque-session-id>
```

Use Zenoh `client` mode, disable multicast scouting, and disable shared-memory
transport. The router is in another container namespace, so shared-memory
transport is neither available nor part of the contract.

The entrypoint must read these names. Treat `ORIGAMI_SESSION_ID` as an opaque,
case-sensitive string carried in every request and reply; it is not part of a
Zenoh key. Do not transform it or derive robot addresses or topic names from it.

Every team receives a separate router. Declare exactly these fixed queryables:

```text
origami-zenoh-v1/metadata
origami-zenoh-v1/reset
origami-zenoh-v1/infer
```

Do not prepend/append the session ID, team name, or another namespace. Do not
declare a wildcard queryable, publisher, or subscriber. There must be one active
owner for each queryable and exactly one reply for every query.

## 3. Wire encoding

Every request and reply payload is a MessagePack map encoded with the
repository's safe `msgpack-numpy` convention:

```python
{
    b"__ndarray__": True,
    b"data": array.tobytes(order="C"),
    b"dtype": array.dtype.str,
    b"shape": array.shape,
}
```

NumPy scalars use `__npgeneric__`, `data`, and `dtype` in the same convention.
Map field names in the protocol schemas below are UTF-8 strings. Arrays must be
C-contiguous on the wire. Object, structured, and complex dtypes are forbidden;
the decoder must never fall back to pickle.

## 4. Required request/reply envelope

Every request is a MessagePack map containing all four envelope fields:

```python
{
    "protocol_version": "origami-zenoh-v1",
    "operation": "metadata" | "reset" | "infer",
    "request_id": unique_string,
    "session_id": ORIGAMI_SESSION_ID,
    # operation-specific request fields follow
}
```

The organizer generates a new unique `request_id` for every query. The queryable
must return all four envelope fields with their values copied unchanged from
that request:

```python
{
    "protocol_version": "origami-zenoh-v1",
    "operation": request["operation"],
    "request_id": request["request_id"],
    "session_id": request["session_id"],
    # operation-specific reply fields follow
}
```

A missing, changed, stale, or incorrectly typed correlation field fails the
request. Do not generate a replacement request/session ID in the reply.

`origami-zenoh-v1` is the Zenoh transport/envelope version. The nested policy
metadata intentionally advertises semantic policy protocol `origami-v1`; these
two values are different and both are required.

## 5. Queryables

### `metadata`

Key and request:

```python
# key: origami-zenoh-v1/metadata
{
    "protocol_version": "origami-zenoh-v1",
    "operation": "metadata",
    "request_id": request_id,
    "session_id": session_id,
}
```

Required reply:

```python
{
    "protocol_version": "origami-zenoh-v1",
    "operation": "metadata",
    "request_id": request_id,
    "session_id": session_id,
    "metadata": {
        "protocol_version": "origami-v1",
        "action_dim": 65,
        "action_horizon": T,  # integer in [1, 1024]
        "action_type": "absolute_joint_position",
        "action_units": "radians",
        "joint_names": [...],  # exact 65 names from robot_io_spec.md
    },
}
```

`action_horizon` is fixed for the life of the process and must match every
successful `infer` reply. `joint_names` is mandatory and must exactly equal the
ordered 65-name list in `robot_io_spec.md`; omission, reordering, duplication,
or model-internal padded names fail validation. Extra metadata fields are
allowed only if they do not change required fields.

### `reset`

The organizer calls `reset` before the first inference of every episode.

Key and request:

```python
# key: origami-zenoh-v1/reset
{
    "protocol_version": "origami-zenoh-v1",
    "operation": "reset",
    "request_id": request_id,
    "session_id": session_id,
}
```

Successful reply:

```python
{
    "protocol_version": "origami-zenoh-v1",
    "operation": "reset",
    "request_id": request_id,
    "session_id": session_id,
    "ok": True,
}
```

Before replying, clear all episode-scoped state: observation history, action
queues, recurrent state, temporal-ensemble buffers, cached prompts, and
episode-specific random state as appropriate. `reset` must be idempotent. It
does not move the robot and must not run inference.

The organizer will not issue `infer` concurrently with `reset` for the same
session. A service should either serialize stateful inference requests or reject
unsupported concurrency explicitly.

### `infer`

Key and request:

```python
{
    # key: origami-zenoh-v1/infer
    "protocol_version": "origami-zenoh-v1",
    "operation": "infer",
    "request_id": request_id,
    "session_id": session_id,
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

The required successful reply is:

```python
{
    "protocol_version": "origami-zenoh-v1",
    "operation": "infer",
    "request_id": request_id,
    "session_id": session_id,
    "actions": float32[T, 65],
}
```

Images are RGB in HWC order. The complete field semantics, optional raw tactile
grid, and state/action column order are defined in `robot_io_spec.md`. Every
action value must be finite and is an absolute target
in radians, not a delta, velocity, normalized value, token, or motor command.
Do not return padded model dimensions.

The organizer performs receding-horizon control and may consume only a prefix of
the chunk. A later observation is authoritative; the service must not assume
that all previously returned rows were executed.

## 6. Timing and lifecycle

A normal episode follows this sequence:

```text
container start
  -> connect to supplied Zenoh endpoint
  -> declare metadata/reset/infer queryables
  -> metadata query
  -> reset query and successful ACK
  -> infer query -> action reply
  -> infer query -> action reply ...
  -> reset before the next episode
```

The organizer sets startup, query, episode, and overall evaluation deadlines.
Each query must finish within the supplied query timeout. A timeout, dropped
Zenoh session, malformed response, NaN/Inf, wrong shape/dtype, or inconsistent
horizon fails that request; the organizer will not execute the affected chunk.
Do not rely on unlimited first-request compilation time. Load and warm the model
during startup where possible.

The service must tolerate reconnecting to the supplied router after a transient
transport interruption, but must not silently replay an old inference reply.
It must stop accepting work on process termination and exit promptly when sent
`SIGTERM`.

## 7. Errors

For an application-level failure, send a normal MessagePack reply:

```python
{
    "protocol_version": "origami-zenoh-v1",
    "operation": request["operation"],
    "request_id": request["request_id"],
    "session_id": request["session_id"],
    "error": {
        "code": "INVALID_REQUEST",
        "message": "human-readable, non-secret detail",
        "retryable": False,
    }
}
```

Stable recommended codes are:

- `INVALID_REQUEST`: missing key, wrong type, dtype, or shape;
- `NOT_READY`: model is loading or temporarily unavailable;
- `INFERENCE_FAILED`: model execution failed;
- `OVERLOADED`: bounded request capacity is exhausted;
- `INTERNAL`: an unexpected non-sensitive failure.

Do not include stack traces, credentials, host paths, model download tokens, or
private data in replies. A service must not substitute zeros, stale actions, or
a different horizon after an error. Zenoh transport-level error replies are
also treated as request failures.

The four envelope fields are mandatory even on application errors.

## 8. Model adapter boundaries

The public contract is model-agnostic:

- **ACT:** assemble any history inside the container, denormalize outputs, and
  return the fixed `float32[T,65]` absolute-radian chunk. Temporal aggregation
  may be internal, but cannot change the wire schema.
- **Diffusion Policy:** perform sampling and denormalization internally. Budget
  all denoising steps inside the query deadline. Return deterministic dtype and
  shape even if sampling is stochastic.
- **OpenPI:** map whichever full-observation fields the model consumes into its
  internal naming, ignore unused documented fields, remove action padding (for
  example 72 to 65), and convert to absolute radians before replying.
- **Other policies:** may use any internal architecture, history, tokenizer, or
  action representation, provided the boundary exactly matches this protocol.

The organizer does not provide a model config, checkpoint mount, normalization
file, tokenizer cache, framework-specific client, or internet access.

## 9. Validate without a robot

Install the public Python dependencies, start the submitted container against a
test Zenoh router, and run:

```bash
cd sharpa_north_ces_lite_sdk-main
python examples/check_zenoh_policy.py \
  --endpoint tcp/127.0.0.1:7447 \
  --session-id test-session \
  --timeout 10 \
  --requests 3 \
  --expected-horizon 25
```

The validator sends only deterministic synthetic observations. It does not
import the North SDK, read hardware, know robot topics, or publish actions. A
pass proves wire-contract compatibility only; it does not prove model quality,
real-observation preprocessing equivalence, latency under organizer load, or
robot safety.

## 10. Submission checklist

- The OCI/Docker image and immutable digest or archive checksum are supplied.
- The image starts with no source/checkpoint bind mounts and no internet access.
- Endpoint and session ID are runtime configuration, not baked into the image.
- Only the three fixed `origami-zenoh-v1/*` queryables are declared.
- Every request/reply has a correlated transport envelope; metadata is nested,
  uses semantic protocol `origami-v1`, and contains all 65 joint names.
- Metadata, reset, and repeated synthetic inference checks pass.
- Replies are finite `float32[T,65]` absolute radians with a fixed horizon.
- No North SDK, robot credentials/topics, action publisher, SSH key, or secret
  is present in any image layer.
- License notices for redistributed model code, weights, and assets are included.
