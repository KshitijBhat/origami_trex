# Container submission and deployment contract

Teams submit a self-contained OCI/Docker policy image. The normative public
protocol is Zenoh-only `origami-zenoh-v1`, defined in
`participant_zenoh_submission.md`. The image is queried by the organizer-owned
runtime and never communicates with the robot directly.

## 1. Self-contained image

The final image must include:

- model architecture, inference code, and fixed entrypoint;
- weights/checkpoint;
- preprocessing and output conversion;
- normalization statistics, tokenizer, vocabulary, and runtime assets;
- Zenoh client/queryable implementation for `origami-zenoh-v1`;
- all shared libraries and Python packages not guaranteed by the OCI runtime;
- applicable third-party license notices.

At evaluation time the organizer does not mount team source code, checkpoints,
configs, caches, credentials, or host Python packages. It does not generate
team-specific OpenPI/ACT/Diffusion configs. The image must start from its
declared entrypoint and reconstruct its policy without framework knowledge in
the runner.

Builds may download dependencies, subject to organizer build rules. Runtime is
offline except for the isolated Zenoh network. Test a clean image with an empty
host cache to catch undeclared assets and accidental downloads.

## 2. Runtime configuration

The organizer supplies an isolated Zenoh router endpoint and opaque session ID.
The container environment names are exactly:

```text
ORIGAMI_ZENOH_ENDPOINT=tcp/<ip-or-Docker-DNS-hostname>:<port>
ORIGAMI_SESSION_ID=<opaque-session-id>
```

The entrypoint must fail fast and non-zero if either value is missing or
malformed. Do not hard-code development endpoints, robot addresses, credentials,
or session IDs. `ORIGAMI_SESSION_ID` is carried in protocol envelopes, not in
the queryable key.

No inbound container port is required. The policy opens an outbound Zenoh
`client` session to the supplied endpoint and declares:

```text
origami-zenoh-v1/metadata
origami-zenoh-v1/reset
origami-zenoh-v1/infer
```

Multicast discovery must not be required. The policy must not declare any
publisher, subscriber, wildcard queryable, robot key expression, or queryable
outside these fixed keys. Each team has an isolated router, so no team/session
prefix is added to the keys.

The organizer may pass standard OCI controls such as:

```bash
docker run --rm --gpus all \
  --network origami-eval \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=1g \
  --shm-size 8g \
  --memory 32g \
  --cpus 8 \
  -e ORIGAMI_ZENOH_ENDPOINT=tcp/origami-router:7447 \
  -e ORIGAMI_SESSION_ID=team-episode-worker-0 \
  team-name/origami-policy@sha256:<digest>
```

Exact GPU, memory, CPU, shared-memory, writable-temp, and timeout limits are
published separately by the organizer. A submission must not require
`--privileged`, host networking, host PID/IPC namespaces, Docker socket access,
extra Linux capabilities, USB devices, or host filesystem mounts.

## 3. Startup and readiness

The image process should:

1. validate runtime configuration;
2. initialize logging without printing secrets;
3. load the model and all assets;
4. open the supplied Zenoh session in client mode;
5. declare all three queryables;
6. serve `metadata`, accept `reset`, and answer `infer`.

The organizer considers the service ready when the fixed
`origami-zenoh-v1/metadata` query succeeds with a correlated reply envelope and
valid nested `metadata`. There is no official HTTP health endpoint.

Prefer loading and compilation during startup. If queryables are declared before
the model is ready, `metadata` must still be valid and `infer` must reply with a
structured `NOT_READY` error until readiness; it must not hang or return dummy
actions. Startup and every query have hard deadlines.

On `SIGTERM`, stop receiving new work, release queryables/session resources, and
exit promptly. Store only disposable data under documented writable paths.

## 4. Episode and request timing

The runner performs:

```text
metadata -> reset -> infer -> infer -> ... -> reset -> ...
```

`reset` occurs before each episode and must clear all episode state before
acknowledgement. Every `infer` receives one complete observation and returns one
complete action chunk. The organizer may consume only a prefix and then replan;
the policy may not publish, schedule, or assume execution of its result.

Every request contains `protocol_version`, `operation`, a unique `request_id`,
and `session_id`; `infer` additionally contains `observation`. Every reply
echoes the same four envelope values. The metadata reply places semantic
protocol `origami-v1`, dimension, horizon, absolute-radian semantics, and all 65
ordered joint names inside its `metadata` object.

Do not queue unbounded requests. If internal concurrency is unsupported, process
queries serially or return `OVERLOADED`. A response arriving after the query
deadline is discarded. A transport reconnect must not replay a cached action
for a new observation.

## 5. Validation before submission

Run a test Zenoh router, start the final image with only the two public runtime
values, then use the black-box validator:

```bash
cd sharpa_north_ces_lite_sdk-main
python examples/check_zenoh_policy.py \
  --endpoint tcp/127.0.0.1:7447 \
  --session-id test-session \
  --timeout 10 \
  --requests 3 \
  --expected-horizon 25
```

The validator checks:

- metadata protocol, action semantics, dimension, and horizon;
- all 65 ordered metadata joint names;
- reset acknowledgement;
- repeated inference with the complete synthetic observation: four camera
  images, state/torque, tactile force and image grids, and prompt;
- unique request IDs, strict reply correlation, and exactly one reply per query;
- `float32[T,65]`, finite values, and metadata/response horizon consistency;
- per-request latency against the configured query timeout.

It does not import the North SDK, read a robot or training dataset, subscribe to
robot topics, or publish actions. Passing it is necessary but does not validate
model quality or deployment safety.

Also test the exact immutable image with:

- no checkpoint/source mounts;
- no internet route except the isolated Zenoh router;
- the published resource limits and startup timeout;
- a read-only root filesystem and bounded writable temporary storage;
- repeated episodes and process restart;
- invalid inputs and model failures without stale/dummy action fallback.

## 6. Framework packaging boundaries

ACT, Diffusion Policy, OpenPI, and custom frameworks are all allowed. Their
internal history, transforms, action padding, normalization, sampling, and
tokenization stay inside the image. The wire boundary is always the full
observation in `robot_io_spec.md` and `float32[T,65]` absolute-radian actions.

An OpenPI submission must bake its chosen config and checkpoint into the image,
expose Zenoh queryables at its entrypoint, and trim any padded action dimension
to exactly 65.

## 7. Deployment security

- Submit an immutable registry digest or OCI archive plus SHA-256 checksum; do
  not rely on a mutable `latest` tag.
- Run as a non-root user and drop all capabilities where the framework permits.
- Never embed North SDK code, robot keys/topics, action publishers, cloud
  credentials, SSH keys, package-registry tokens, or dataset secrets.
- Keep logs on stdout/stderr, bounded by the runner. Do not log observations,
  prompts, action arrays, environment values, or model secrets by default.
- Do not start shells, SSH servers, dashboards, notebooks, telemetry exporters,
  or unrelated network listeners.
- Use MessagePack only with the safe NumPy codec; pickle and object-array
  deserialization are forbidden.
- Treat all query payloads as untrusted: enforce key, dtype, shape, size, and
  finite-value checks before model execution.

The organizer independently enforces network isolation, resource limits,
timeouts, action validation, joint/velocity limits, watchdogs, episode
authorization, and emergency stops. Participant code must not attempt to replace
or bypass that boundary.

## 8. Deliverables

Provide:

1. image reference and immutable digest, or OCI archive and checksum;
2. declared GPU/CPU/RAM/shared-memory requirements within competition limits;
3. expected startup time and fixed action horizon;
4. third-party/model licenses and notices;
5. optional non-secret operator notes for the fixed entrypoint.

Do not submit a raw checkpoint, host setup script, model-specific runner patch,
North communication package, or organizer-runtime modification as a substitute
for the self-contained image.
