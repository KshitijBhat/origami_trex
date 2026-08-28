# [Announcement] Origami Inference Kit Async Update (2026-08-26)

Dear participating teams,

We have updated the Origami Inference Kit to simplify model integration, container packaging, and real-robot observation testing. The main updates are listed below.

## 1. Standardized Image Build and Submission

We added Docker/OCI image specifications and reference templates that support:

- Packaging model code, checkpoints, normalization statistics, tokenizers, and runtime dependencies into a self-contained image;
- Using the unified `origami-zenoh-v1` inference interface;
- Testing metadata, reset, and infer locally;
- Running black-box protocol validation with the public validator;
- Exporting a `.tar.zst` submission archive;
- Testing under read-only, non-root conditions without host filesystem mounts.

The public development kit does not include model checkpoints. Each team must package its own weights and related assets into the final submission image.

## 2. Async Execution Is Now the Default

The submission protocol now supports an optional metadata field:

```json
{"execution_mode": "async"}
```

- `async` is the default when the field or Docker build argument is omitted;
- `sync` remains supported for teams that require the original synchronous execution path;
- The asynchronous Gateway path decouples inference from the 30 Hz control loop and temporally aggregates overlapping action chunks;
- The public kit includes an offline async smoke test that requires no robot hardware or private SDK.

For the bundled OpenPI image, select the mode with `--build-arg EXECUTION_MODE=async|sync`.

## 3. Public Read-Only Real-Robot Observations

A public remote-development interface has been added.

After reserving a testing slot, each team will separately receive:

```text
Public endpoint
Session ID
Team token
TLS CA
```

Teams can use the official client to retrieve real-robot observations and pass them directly to their local inference adapter.

This interface is strictly read-only. It does not provide action publishing, robot control, internal topics, the North SDK, or access to the organizer's internal network. Predictions generated locally are never sent back to the robot through this interface.

## 4. Local Image Shadow Testing

A local Shadow Evaluator has been added.

Teams can access it on their own machine at:

```text
http://127.0.0.1:7861
```

The evaluator supports:

1. Connecting to the public read-only observation service;
2. Loading a local Docker image or `.tar.zst` submission archive;
3. Calling the image's metadata, reset, and infer interfaces;
4. Viewing real RGB observations and robot state;
5. Playing predicted trajectories in the North URDF;
6. Checking action shape, dtype, NaN, and Inf;
7. Checking joint limits, action jumps, and velocity.

Shadow mode never sends actions to the physical robot.

## 5. Complete Observation, Tactile, and Image Protocol

The formal inference input now includes:

- Head-left RGB;
- Head-right RGB;
- Left-wrist RGB;
- Right-wrist RGB;
- 65-dimensional joint state;
- 65-dimensional joint torque;
- 60-dimensional fingertip force/torque;
- `uint8[480,1200,3]` tactile deform grid;
- Optional `uint8[480,1600,3]` tactile raw grid;
- Task prompt.

TCP state is not provided.

The native RGB camera resolution is `1920x1536`. The organizer directly resizes each frame to `224x224` without preserving the aspect ratio and without adding padding, letterboxing, or black borders. This matches the square-image conversion used by the training data.

A model may consume only a subset of these fields, but the submitted image's inference handler must accept the complete observation.

For field names, shapes, dtypes, tactile ordering, and zero-filling rules, see:

```text
docs/robot_io_spec.md
```

## 6. Formal Zenoh Inference Protocol

Each submitted image must implement:

```text
origami-zenoh-v1/metadata
origami-zenoh-v1/reset
origami-zenoh-v1/infer
```

The inference output must be:

```text
finite float32[T,65]
```

Requirements:

- `T` must match the action horizon declared in metadata;
- The 65 columns must follow the published joint order exactly;
- Actions must be absolute joint-position targets;
- Units must be radians;
- NaN and Inf are not allowed;
- Normalized, delta, velocity, tokenized, or padded actions must not cross the public interface.

## 7. OpenPI and General Development Tools

The development kit now includes:

- A framework-neutral Zenoh policy server template;
- OpenPI inference and model-adapter reference code;
- An OpenPI Zenoh submission Dockerfile;
- A transport-independent async temporal-aggregation example and smoke test;
- A public black-box validator;
- A public read-only observation client;
- A local Shadow Evaluator.

Teams may use OpenPI, ACT, Diffusion Policy, or a custom framework. Each team remains responsible for packaging its model code, checkpoint, normalization assets, tokenizer, runtime dependencies, and all applicable third-party licenses and notices.

## 8. FAQ, Documentation, and Development Package

The following documentation has been added or expanded:

- Participant quick-start guide;
- Complete development and image-submission guide;
- Zenoh request/reply protocol;
- Observation/action tensor contract;
- Docker sandbox and resource requirements;
- Public remote-development instructions;
- FAQ, troubleshooting, and common errors;
- OpenPI integration and image-build instructions.

Please start with:

```text
PARTICIPANT_GUIDE.md
```

Release package:

```text
origami-inference-kit-async-20260826.zip
```

The public endpoint, session ID, token, and TLS CA are not included in the package. They will be distributed separately according to each team's reserved testing slot.

Please use the latest development kit to complete model integration, image packaging, and local validation.

Note: A `PASS` result from the public validator confirms protocol compatibility only. It does not guarantee task performance, action safety, or acceptance for final real-robot evaluation.
