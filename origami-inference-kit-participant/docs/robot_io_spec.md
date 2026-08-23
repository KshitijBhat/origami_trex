# Robot I/O specification

This document is the normative tensor contract for the public
`origami-zenoh-v1` policy interface. Transport, queryable names, metadata, reset,
and errors are specified in `participant_zenoh_submission.md`.

The participant policy receives an already assembled observation and returns an
action chunk. It receives no North SDK, native robot topics, control mode,
publisher, or action-execution API.

## 1. Inference observation

The `observation` object in every `infer` request contains:

| key | dtype/type | exact shape | meaning |
|---|---|---|---|
| `observation/image/head_left` | `uint8` | `(224, 224, 3)` | head stereo left-eye RGB |
| `observation/image/head_right` | `uint8` | `(224, 224, 3)` | head stereo right-eye RGB |
| `observation/image/wrist_left` | `uint8` | `(224, 224, 3)` | left-wrist RGB |
| `observation/image/wrist_right` | `uint8` | `(224, 224, 3)` | right-wrist RGB |
| `observation/state` | `float32` | `(65,)` | current absolute joint angles, radians |
| `observation/state/joint_torque` | `float32` | `(65,)` | joint torque, same layout as state |
| `observation/tactile` | `float32` | `(60,)` | 10 fingertip force/torque vectors, §3 |
| `observation/image/tactile_deform` | `uint8` | `(480, 1200, 3)` | 2×5 fingertip deform grid, §3 |
| `observation/image/tactile_raw` | `uint8` | `(480, 1600, 3)` | optional 2×5 raw tactile grid, §3 |
| `prompt` | UTF-8 string | scalar | task instruction |

All fields except `observation/image/tactile_raw` are required. The raw tactile
grid may be omitted when the organizer disables that high-bandwidth stream.
Participant code must tolerate its absence. Other undocumented fields are not
part of this protocol version and may be rejected.

### RGB camera preprocessing

- Camera images are HWC, RGB, C-contiguous, `uint8`, range `0..255`.
- The native camera frame is `1920x1536` (width × height).
- The organizer directly squashes it to `224x224`; aspect ratio is deliberately
  not preserved and no padding/letterbox is added.
- This matches the training-data conversion, which directly squashed native
  frames to square images. Participants must not restore the native aspect ratio
  or add black bars before model inference.
- The public policy never receives BGR, CHW, JPEG/PNG bytes, normalized floats,
  a batch dimension, or native-resolution RGB frames.

### Stable schema and unavailable sources

Required fields keep fixed shapes and dtypes. If a physical source is unavailable
for a run, the organizer provides a zero-filled value rather than changing the
schema. In particular:

- the motor torque block has no source and is zero-filled;
- unavailable tactile sensors produce zero-filled force/image cells.

Selecting which fields the model consumes is a participant decision, but the
server boundary must accept the full observation.

## 2. State, torque and action layout (65 DoF)

`observation/state`, `observation/state/joint_torque`, and every row of returned
`actions` use this exact concatenation:

| slice (Python, end-exclusive) | part | dimension |
|---|---|---:|
| `0:7` | left arm | 7 |
| `7:29` | left dexterous hand | 22 |
| `29:36` | right arm | 7 |
| `36:58` | right dexterous hand | 22 |
| `58:65` | head / torso motor group | 7 |

`observation/state` and `actions` contain absolute joint angles in radians.
`observation/state/joint_torque` uses the same indices but contains torque; its
`58:65` motor block is always zero.

The exact `metadata["joint_names"]`, state order, and action-column order are:

```python
[
    "left_arm_joint_1",
    "left_arm_joint_2",
    "left_arm_joint_3",
    "left_arm_joint_4",
    "left_arm_joint_5",
    "left_arm_joint_6",
    "left_arm_joint_7",
    "left_thumb_CMC_FE",
    "left_thumb_CMC_AA",
    "left_thumb_MCP_FE",
    "left_thumb_MCP_AA",
    "left_thumb_IP",
    "left_index_MCP_FE",
    "left_index_MCP_AA",
    "left_index_PIP",
    "left_index_DIP",
    "left_middle_MCP_FE",
    "left_middle_MCP_AA",
    "left_middle_PIP",
    "left_middle_DIP",
    "left_ring_MCP_FE",
    "left_ring_MCP_AA",
    "left_ring_PIP",
    "left_ring_DIP",
    "left_pinky_CMC",
    "left_pinky_MCP_FE",
    "left_pinky_MCP_AA",
    "left_pinky_PIP",
    "left_pinky_DIP",
    "right_arm_joint_1",
    "right_arm_joint_2",
    "right_arm_joint_3",
    "right_arm_joint_4",
    "right_arm_joint_5",
    "right_arm_joint_6",
    "right_arm_joint_7",
    "right_thumb_CMC_FE",
    "right_thumb_CMC_AA",
    "right_thumb_MCP_FE",
    "right_thumb_MCP_AA",
    "right_thumb_IP",
    "right_index_MCP_FE",
    "right_index_MCP_AA",
    "right_index_PIP",
    "right_index_DIP",
    "right_middle_MCP_FE",
    "right_middle_MCP_AA",
    "right_middle_PIP",
    "right_middle_DIP",
    "right_ring_MCP_FE",
    "right_ring_MCP_AA",
    "right_ring_PIP",
    "right_ring_DIP",
    "right_pinky_CMC",
    "right_pinky_MCP_FE",
    "right_pinky_MCP_AA",
    "right_pinky_PIP",
    "right_pinky_DIP",
    "lower_body_joint_1",
    "lower_body_joint_2",
    "lower_body_joint_3",
    "lower_body_joint_4",
    "lower_body_joint_5",
    "neck_joint_1",
    "neck_joint_2",
]
```

The list has exactly 65 unique entries. Do not sort, mirror, swap sides, remove
stationary joints, or expose model-internal padded names.

## 3. Tactile layout

The fixed fingertip order is:

```text
left_thumb, left_index, left_middle, left_ring, left_little,
right_thumb, right_index, right_middle, right_ring, right_little
```

`observation/tactile` concatenates one six-dimensional wrench
`[fx, fy, fz, tx, ty, tz]` per finger, producing `(60,)`.

The image grids use two rows and five columns:

- top row: left-hand fingers in the order above;
- bottom row: right-hand fingers;
- deform cell: `240x240`, total grid `(480,1200,3)`;
- raw cell: `240x320`, total grid `(480,1600,3)`.

The raw grid is optional. The deform grid and force vector are required.

## 4. Inference reply

A successful reply is:

```python
{
    "protocol_version": "origami-zenoh-v1",
    "operation": "infer",
    "request_id": request_id,
    "session_id": session_id,
    "actions": np.ndarray(shape=(T, 65), dtype=np.float32),
}
```

Requirements:

- `T` is fixed for the process and equals metadata `action_horizon`;
- all values are finite;
- each row is an absolute joint-position target in radians;
- columns use §2 exactly.

Actions are not velocity, torque, delta, normalized values, tokens, or per-part
maps. A model with a padded internal action dimension must denormalize and map it
to exactly the physical 65 columns before replying.

## 5. MessagePack NumPy representation

Arrays use:

```python
{
    b"__ndarray__": True,
    b"data": value.tobytes(order="C"),
    b"dtype": value.dtype.str,
    b"shape": value.shape,
}
```

Object, structured and complex arrays are forbidden. Pickle is forbidden.
Implementations must validate shape and byte length before model execution.

## 6. Model adaptation

```text
full public observation
  -> participant preprocessing / model
  -> participant denormalization and 65-column mapping
  -> public absolute-radian action chunk
  -> organizer validation and execution boundary
```

History, image normalization, modality selection, language tokenization,
diffusion sampling and temporal ensembling are participant-internal. Any
episode-scoped state must be cleared by `reset`.

## 7. Decoded example

```python
request = {
    "protocol_version": "origami-zenoh-v1",
    "operation": "infer",
    "request_id": request_id,
    "session_id": session_id,
    "observation": {
        "observation/image/head_left": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/image/head_right": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/image/wrist_left": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/image/wrist_right": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/state": np.zeros((65,), dtype=np.float32),
        "observation/state/joint_torque": np.zeros((65,), dtype=np.float32),
        "observation/tactile": np.zeros((60,), dtype=np.float32),
        "observation/image/tactile_deform": np.zeros((480, 1200, 3), dtype=np.uint8),
        "observation/image/tactile_raw": np.zeros((480, 1600, 3), dtype=np.uint8),
        "prompt": "fold the plane",
    },
}
```

The model may ignore fields it does not use, but its public handler must accept
the complete object and must not require undocumented robot-side data.
