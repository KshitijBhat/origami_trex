# Origami × T-Rex — Complete Redesign Plan (eef-62)

**Target:** convert the Robotic Origami Challenge dataset into the **exact** format
`T-Rex/qwen_vla/lerobot_dataset.py::TRexLeRobotDataset` consumes, post-train with
upstream T-Rex's recipe unchanged, and deploy through the competition's
`origami-zenoh-v1` contract with differential IK.

**Status:** plan only. No code has been written. Implementation target: Sonnet 5.

---

## 0. Decisions already made (do not re-litigate)

| # | Decision | Value |
|---|---|---|
| D1 | Data format | **LeRobot v3.0 only** (`--data_format lerobot`). No JSON path. |
| D2 | Code location | **New top-level `origami/` package. `T-Rex/` stays byte-identical to upstream.** |
| D3 | Data scope | **All 51 seasons, stream-and-delete** from HF. |
| D4 | Action space | **eef-62**: `[L_eef_delta9, L_hand_abs22, R_eef_delta9, R_hand_abs22]`, `action_chunk=16`, `FRAME_STRIDE=1`. |
| D5 | Torso/neck (`motor`, dims 58:65) | **Not predicted.** Held at the observed value at deploy. |
| D6 | "raw tactile" | Interpreted as **raw 60-D wrench + raw deform maps fed to the on-the-fly VQ-VAE** (T-Rex's default, no pre-baked codes). `observation.images.tactile_raw` is an *optional* extra stream, off by default — see §11.3. |
| D7 | Images | **No crop.** Full square FOV, 480×480 → 224×224 (byte-identical to the deploy wire), then `--image_size 384 384` in both training and deploy. See §4.3. |

---

## 1. Established facts (verified against the actual data — cite these, don't re-derive)

### 1.1 Source dataset (`season_*/lerobot3.0/`)

Verified on `season_POC22061_2026_07_23_10_20_33_train`.

* `codebase_version: v3.0`, `robot_type: north_ces`, `fps: 30`, 22 episodes, 95 533 frames.
* Episode lengths 4 038–5 154 frames (135–172 s). Whole dataset: 51 seasons / 682 episodes / 4 763 267 frames.
* `data/chunk-000/file-000.parquet` holds the **whole season** in one row group. Columns:
  `observation.state[65] f32`, `observation.state.joint_torque[65] f32`,
  `observation.state.tcp[24] f32`, `action[65] f32`, `observation.tactile[60] f32`,
  `timestamp f32`, `frame_index i64`, `episode_index i64`, `index i64`, `task_index i64`.
* **`observation.state.tcp` is identically zero** (verified: `abs().max(axis=0)` is all-zero
  over all 95 533 rows). There is **no** EEF data in the release — FK is mandatory.
* `observation.state` / `action` layout (65 D, absolute radians):

  | slice | group | URDF joint names |
  |---|---|---|
  | `0:7` | left arm | `left_arm_joint_1..7` |
  | `7:29` | left hand | `left_<HAND_ORDER>` (22) |
  | `29:36` | right arm | `right_arm_joint_1..7` |
  | `36:58` | right hand | `right_<HAND_ORDER>` (22) |
  | `58:63` | lower body | `lower_body_joint_1..5` |
  | `63:65` | neck | `neck_joint_1..2` |

  `HAND_ORDER` (22, identical in the URDF, in `participant_local_evaluator/contract.py::_hand_joint_names`,
  and in `T-Rex/hardware_code/teleop/robot_descriptions.py::SHARPA_HAND_JOINT_ORDER`):
  ```
  thumb_CMC_FE, thumb_CMC_AA, thumb_MCP_FE, thumb_MCP_AA, thumb_IP,
  index_MCP_FE, index_MCP_AA, index_PIP, index_DIP,
  middle_MCP_FE, middle_MCP_AA, middle_PIP, middle_DIP,
  ring_MCP_FE, ring_MCP_AA, ring_PIP, ring_DIP,
  pinky_CMC, pinky_MCP_FE, pinky_MCP_AA, pinky_PIP, pinky_DIP
  ```
* **`action` is a look-ahead command.** `argmin_k mean|action[t] − state[t+k]|` over episode 0:
  arms **k≈5–6** (0.17–0.20 s), hands **k≈2**, motor **k≈4** but with error 4e-4 rad
  (i.e. the motor block is effectively an echo of the state). So `action` ↔ *target*,
  `state` ↔ *current* — exactly T-Rex's `*_target_*` / `*_current_*` split.
* Video streams (all `h264`, `yuv420p`, 30 fps, no audio) — per-season sizes measured:

  | key | shape | size/season | used? |
  |---|---|---|---|
  | `observation.images.head_left` | 480×480×3 | 854 MB | **yes** → `observation.images.head` |
  | `observation.images.head_right` | 480×480×3 | 844 MB | no (T-Rex has one slow cam) |
  | `observation.images.wrist_left` | 480×480×3 | 957 MB | **yes** |
  | `observation.images.wrist_right` | 480×480×3 | 1.1 GB | **yes** |
  | `observation.images.tactile_deform` | 480×1200×3 | 80 MB | **yes** → split to 10 |
  | `observation.images.tactile_raw` | 480×1600×3 | 3.0 GB | optional (§11.3) |

  Fetch budget = head_left + wrist_l + wrist_r + tactile_deform + data + meta ≈ **3.1 GB/season**
  (this season; AV1 seasons are smaller, ~1 GB). Total download 50–160 GB, stream-and-delete.
* **Video files are chunked *per key*, not per episode.** In this season `head_left` spans 6
  files, `tactile_deform` 1 file (all 95 533 frames), `tactile_raw` 22 files (one per episode).
  `meta/episodes/chunk-000/file-000.parquet` gives, **per episode and per video key**:
  `videos/<key>/chunk_index`, `videos/<key>/file_index`,
  `videos/<key>/from_timestamp`, `videos/<key>/to_timestamp`.
  It also gives `length`, `tasks`, `data/{chunk,file}_index`, `dataset_from_index`, `dataset_to_index`.
  **Any decoder must use this table; never assume file-per-episode.**
* Single task string: `"north ces task"` (`meta/tasks.parquet`). **Not usable as a language
  instruction** — it names the rig, not the task. Overridden at conversion time; see §5.7.
* `meta/modality.json` declares state/action as one absolute 65-D `joints` block, no rotations.

### 1.2 Tactile — verified layout

* `observation.tactile` is `[10 fingers × 6]` flattened, order
  `left_{thumb,index,middle,ring,little}` then `right_{...}`, each `[fx,fy,fz,tx,ty,tz]`.
  This is **exactly** T-Rex's `tactile_f6` order (left 5 then right 5).
* `tactile_deform` 480×1200 = **2 rows × 5 cols of 240×240**.
  **Row 0 = LEFT hand, row 1 = RIGHT hand; columns = thumb, index, middle, ring, pinky.**
  Verified empirically at frame 2205 of episode 0: forces
  `[10.9, .1, 0, 0, 0 | 34.7, 42.9, 0, 0, 0]` N ↔ tile means
  `row0 = [6.3, 0, 0, 0, 0]`, `row1 = [9.1, 18.7, 0, 0, 0]`. Perfect correspondence.
  Split expression (identical to the old `split_deform_strip`):
  ```python
  arr.reshape(2, 240, 5, 240).transpose(0, 2, 1, 3).reshape(10, 240, 240)
  ```
* All three tactile channels of a decoded deform frame are equal (it is a grayscale signal
  carried in `yuv420p`) → **decode the luma plane directly** (`frame.to_ndarray(format="gray")`
  in PyAV, or `ffmpeg -pix_fmt gray`) to avoid YUV→RGB→gray rounding.
* `tactile_raw` 480×1600 = 2×5 of **240×320**, mean ≈ 125 (raw sensor images).
* **Dead channels.** Mean |F| per finger over episode 0:
  `L = [4.44, 1.61, 0.57, 0.009, 0.065]`, `R = [6.47, 9.52, 0.73, 0.002, 0.003]` N.
  Left ring/pinky and right ring/pinky carry essentially no signal, right middle is weak.
  → q99−q01 ≈ 0 for those dims; see §5.4 for the masking requirement.

### 1.3 Robot model — `north_poc2_2_urdf_usd/north_poc2_2_v3_1.urdf`

**This URDF is complete** (an earlier read wrongly concluded it was hands-only; the
SolidWorks exporter puts `name=` on its own line so a single-line grep misses it).

* `robot name="north_poc2_2_with_double_continental_grip_description"`, 105 joints,
  **exactly 65 revolute** = 5 lower_body + 7 left_arm + 22 left_hand + 7 right_arm
  + 22 right_hand + 2 neck. Names match §1.1 one-for-one.
* Chain: `root_link` → `lower_body_base_link` → `lower_body_link_1..5` → **`torso_base_link`**
  → `{left_arm_base_link → left_arm_link_1..7 → left_hand_base_link → left_hand_C_MC → fingers}`,
  `{right_arm_...}`, `{neck_base_link → neck_link_1..2 → head_base_link}`.
* **EEF frames: `left_hand_base_link` / `right_hand_base_link`** (fixed children of
  `*_arm_link_7`; the direct analogue of T-Rex's `L_ee`/`R_ee` wrist-mount frames).
* Per-arm joint limits (rad), from the URDF — needed for IK clipping and deploy safety:
  ```
  left_arm  lower = [-1.5359, -3.6652, -3.1067, -1.0472, -3.1067, -0.9599, -0.6981]
            upper = [ 4.6775,  0.5236,  3.1067,  2.5307,  3.1067,  0.9599,  0.6981]
  right_arm lower = [-1.5359, -1.7453, -3.1067, -1.0472, -3.1067, -0.9599, -1.5708]
            upper = [ 4.6775,  0.5236,  3.1067,  2.5307,  3.1067,  0.9599,  1.5708]
  velocity limit  = 2.6179 rad/s on every arm joint
  ```
* Sanity-checked with an independent numpy FK on real data (`state[0]` of episode 0):
  `left_hand_base_link = (0.362, 0.194, 1.049) m`, `right = (0.362, −0.193, 1.050) m`,
  `head_base_link = (0.002, 0.000, 1.459) m`. Symmetric, in front, ~1.05 m high. ✅
* Workspace over episode 0 (`left_hand_base_link`): x ∈ [0.312, 0.405], y ∈ [0.066, 0.263],
  z ∈ [0.908, 1.049] m. Right: x ∈ [0.25, 0.45], y ∈ [−0.252, −0.021], z ∈ [0.917, 1.050].
* **Per-frame EEF translation step: mean 0.39 mm, p99 2.04 mm** at 30 Hz.
  → over a 16-frame chunk the arm moves ~6 mm typical. **This is the single most important
  modelling fact in this plan** — see §11.1.
* `lower_body_joint_3` varies by 0.19 rad within a season; joints 4–5 by ≤0.08; neck_2 by 0.28.
  See §3.2 for why this does not matter.

### 1.4 T-Rex contract (upstream, pinned at `T-Rex@f88e10c`)

Everything below is what the converter must produce, cited by file:line.

* **Feature schema** — `T-Rex/utils/lerobot_common.py:128` `build_trex_features(...)`:
  ```
  observation.images.head            video (3, H, W)
  observation.images.wrist_right     video (3, H, W)
  observation.images.wrist_left      video (3, H, W)
  observation.state                  float32 (62,)
  action                             float32 (16, 62)     ← baked delta-base chunk
  action_abs                         float32 (62,)
  observation.tactile_f6             float32 (10, 6)
  observation.tactile_deform.l0..l4  video (3, 240, 240)  ← left thumb..pinky
  observation.tactile_deform.r0..r4  video (3, 240, 240)  ← right thumb..pinky
  ```
  Key constants: `ACTION_DIM=62`, `ACTION_CHUNK=16`, `FRAME_STRIDE=1`, `STATS_KEY="rlbench"`
  (`lerobot_common.py:25-45`).
* **Pose math** — `lerobot_common.py:50-125`. Reuse **verbatim by import**, do not re-implement:
  `pose_matrix_to_9d`, `get_rot_mat`, `compute_chunk_delta_pose`,
  `compute_tracking_error_axis_angle`, `compute_bimanual_tracking_error`, `build_action_chunk`.
* **Norm stats sidecar** — `meta/trex_norm_stats.json`, schema
  `{"rlbench": {"action": {...}, "state": {...}, "tactile_f6": {...}, "tracking_error": {...},
  "num_transitions": int, "num_trajectories": int}}`, each stat block carrying
  `mean/std/max/min/q01/q99/mask`. `action` stats are **per-(step, dim), shape [16, 62]**
  (`lerobot_common.py:222-226`). `tracking_error` carries `mean/std/mean_abs/mask` at 56 D.
* **Loader** — `T-Rex/qwen_vla/lerobot_dataset.py:75`. Reads `meta/info.json` for `fps` and
  feature presence, builds `delta_timestamps`:
  * `observation.images.head`: `[0, 1·s/fps, 2·s/fps, …, n_flare_steps·s/fps]` where
    `s = flare_frame_stride` (`:133`)
  * `observation.tactile_f6`: `[-(W-1)/fps … 0]`, `W = vqvae_window = 16` (`:139`)
  * `collate_fn` (`:183`) normalizes with q01/q99, does the flow-matching noising
    (Beta(1.5, 1.0) time), maps `slow = head[0]`, `fast = [wrist_right, wrist_left]`,
    left-pads `input_ids`, and returns 21 keys.
  * `observation.state` is only read when `use_robot_state=1`; `action_abs` is **never read**
    by the loader (it only feeds the tracking-error stats at conversion time).
* **Batch contract** returned by `collate_fn` (shapes for B, chunk=16, dim=62):
  ```
  input_ids [B,L] i64            attention_mask [B,L] i64
  pixel_values [Σpatches,C]      image_grid_thw [3B,3]
  n_slow_images = 1              noisy_actions [B,16,62] bf16
  target [B,16,62] bf16          timesteps [B] bf16
  norm_actions [B,16,62] bf16    tactile_f6s [B,10,6] bf16
  tactile_deforms [B,10,1,240,240] f32
  tactile_f6s_delayed = tactile_f6s      tactile_deforms_delayed = tactile_deforms
  tactile_codes = None           tactile_f6_history [B,16,10,6] f32
  time_r [B] bf16                eps_r [B,16,62] bf16
  state_raw [B,62] bf16 | None   flare_pixel_values [B·S patches,C] bf16
  flare_grid_thw [B·S,3]
  ```
* **Model input shape constraints** (`T-Rex/qwen_vla/modeling_vla.py`):
  * `tacf6_embedder = ActionEmbedder(6, H)` (`:105`) → `tactile_f6` must be `[B, 10, 6]`.
  * `deform_proj = ActionEmbedder(28800, H)  # 128·15·15` (`:109`) → the deform encoder
    (`DeformAE.py::DeformEncoder`, ResNet-18 stem + layer1..3, total stride 16) requires
    **exactly 240×240** input. 240/16 = 15. **Any other deform resolution breaks the head.**
  * `_embed_tactile_observations` (`:469`) takes `tactile_deform [B, n_fingers, C, H, W]`,
    flattens to `[B·n, C, H, W]`. `C = 1`.
* **Stages.** `main` ships post-train + inference only; `pretrain.py` / `midtrain.py` /
  `prepare_midtrain_merged.py` / `convert_egodex_to_lerobot.py` live on the
  **`full-pipeline`** branch (pinned `b23eafe`, fetched in §2.1). **The ViT is frozen in all
  three stages** (`pretrain.py:946`, `midtrain.py:1427`, `train.py:900`). `midtrain.py` is a
  superset of `train.py` and also supports `--data_format lerobot` (`:1455-1459`) — see §7.5.
* **Official post-train recipe** — `T-Rex/scripts/train.sh`:
  ```
  action_dim 62   action_chunk 16   image_size 384 288
  use_robot_state 0   use_tactile_vec 1   use_tactile_deform 1   use_tactile_vqvae 1
  tactile_intermediate_size 1536   training_stage 2   tactile_loss_weight 1.0
  cascaded_total_steps 10   cascaded_split_step 6
  cascaded_tactile_dropout 0.1   cascaded_loss_weight 1.0
  use_flare 1   n_flare_tokens_per_frame 4   n_flare_steps 8
  flare_loss_weight 0.5   flare_frame_stride 4   flare_layer_index -1
  lr 1e-4   min_lr_ratio 0   weight_decay 0   max_grad_norm 1.0
  train_bsz_per_gpu 16 × 8 GPUs   grad_accum 1   n_epochs 100   save_freq 50
  val_ratio 0.05   val_freq 500   max_val_batches 30
  resume_source midtrain   accelerate config config/sft_qwen.yaml (DeepSpeed ZeRO-2, bf16)
  RESUME_CHECKPOINT = miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6
  ORIGIN_MODEL_PATH = Qwen3-VL-2B-Instruct
  DEFORM_ENCODER_PATH = sharpa_wave_deform_encoder.pth
  ```
* **Inference server** — `T-Rex/scripts/test.py`. `model_load` (`:153`) restores
  `tactile_intermediate_size, n_flare_tokens_per_frame, n_flare_steps, use_tactile_code,
  vqvae_codebook_size, use_tactile_vqvae, vqvae_config, cascaded_total_steps,
  cascaded_split_step` from `training_args.json`, and q01/q99 from `stats_data.json`.
  `CascadedServer` (`:330`) holds the slow-tick snapshot (`cached_kv`, `x_split`, `tau_split`,
  `position_ids`, `n_action_in_cache`) and a 16-frame rolling F6 buffer (`:407`).
  `_run_slow` (`:486`) → `forward_flow_action_partial(refresh_clean_kv=True)`;
  `_run_fast` (`:609`) → `tactile_flow_continue` → denormalized chunk.
* **Deploy reference client** — `T-Rex/hardware_code/eval/eval_trex_async.py`.
  `rot6d_to_matrix` (`:283`, Gram–Schmidt), `matrix_to_rot6d` (`:295`),
  `get_current_pose` (`:446`, FK → 62-D state), `aggregate_chunks` (`:75`, exp-weighted ACT).
  Chunk loop (`:1006-1150`): `target_pos = initial_pos + initial_R @ delta_pos_local`,
  `target_R = initial_R @ rot6d_to_matrix(delta_rot6d_local)`, then
  `solve_ik` warm-started from the previous row. Cadence
  `chunk_size 16, execute_steps_per_chunk 16, refine_offsets [4,8,12]` at
  `command_hz 30, arm_action_hz 300` ⇒ **slow 1.875 Hz, fast 7.5 Hz**.
* **IK** — `T-Rex/hardware_code/teleop/ik_utils.py:23` `PinkLocalIK`.
  `FrameTask(position_cost=50.0, orientation_cost=1.0, lm_damping=0, gain=0.2)` per arm,
  `PostureTask(cost=0.2, gain=0.2)` → previous q (smoothness),
  `PostureTask(cost=0.05, gain=0.2)` → default q (regularization),
  `dt=0.05`, **5 iterations**, `solver="daqp"`, `damping=0`, clip to model limits each iteration.
  Model built by `robot_descriptions.py::build_reduced_bimanual_robot` — torso + head + wheels
  **locked**, `nq == 14`.

### 1.5 Deploy contract (`origami-zenoh-v1`)

From `origami-inference-kit-participant/docs/robot_io_spec.md` and
`sharpa_north_ces_lite_sdk-main/participant_local_evaluator/contract.py`.

* One RPC: `TeamPolicy.infer(observation) -> np.float32[T, 65]`, `T == metadata["action_horizon"]`,
  fixed for the process, `1 ≤ T ≤ 1024`. Plus `reset()`.
* Metadata must be exactly:
  `protocol_version="origami-v1"`, `action_dim=65`, `action_type="absolute_joint_position"`,
  `action_units="radians"`, `joint_names=JOINT_NAMES` (the 65 names, §1.1 order).
* Observation keys (msgpack ndarrays):
  ```
  observation/image/head_left      uint8 (224,224,3)   required
  observation/image/head_right     uint8 (224,224,3)   required
  observation/image/wrist_left     uint8 (224,224,3)   required
  observation/image/wrist_right    uint8 (224,224,3)   required
  observation/image/tactile_deform uint8 (480,1200,3)  required
  observation/image/tactile_raw    uint8 (480,1600,3)  optional (may be absent)
  observation/state                float32 (65,)       required, absolute radians
  observation/state/joint_torque   float32 (65,)       required (zero-filled in practice)
  observation/tactile              float32 (60,)       required
  prompt                           str                 required
  ```
  The organizer's reference value is `"fold the plane"`
  (`openpi-base-main/src/openpi/policies/north_ces_policy.py:33`). Not contractually pinned —
  the spec only requires a string. See §5.7 for how we handle it.
* **Camera preprocessing, quoted:** *"The native camera frame is 1920×1536. The organizer
  directly squashes it to 224×224; aspect ratio is deliberately not preserved and no
  padding/letterbox is added. This matches the training-data conversion, which directly
  squashed native frames to square images. Participants must not restore the native aspect
  ratio or add black bars before model inference."*
  → **This is why D7 says no crop.** The dataset's 480×480 and the wire's 224×224 are the
  same squashed FOV.
* Actions "are not velocity, torque, delta, normalized values, tokens, or per-part maps."
  All values finite. Column layout exactly §1.1.
* `execution_mode`: `async` (default; the gateway keeps executing the previous chunk while
  inference runs) or `sync`. Submission is a Docker image.
* Unavailable sensors are **zero-filled, not omitted** (except `tactile_raw`, which may be absent).

---

## 2. Repo layout after the redesign

```
origami_trex/
├── T-Rex/                          git submodule, pinned f88e10c, ZERO local diff
├── north_poc2_2_urdf_usd/          official robot asset (already present, complete)
├── origami-inference-kit-participant/   competition SDK (already present)
├── dataset.md                      dataset card + the 101/25 season split
├── REDESIGN_PLAN.md                this file
├── origami/
│   ├── __init__.py
│   ├── constants.py                65-D layout, joint names, keys, split lists
│   ├── kinematics.py               FK + Pink IK on north_poc2_2
│   ├── decode.py                   PyAV episode-slice decoding, luma, deform split
│   ├── fetch.py                    per-season HF stream-and-delete
│   ├── stats.py                    bounded-memory q01/q99 accumulator
│   ├── convert.py                  season → T-Rex eef-62 LeRobot shard
│   ├── merge.py                    shard concatenation into one root
│   ├── prepare.py                  CLI orchestrating fetch→convert→merge
│   ├── verify.py                   all §10 gates
│   ├── trex_patch.py               2 monkeypatches to T-Rex (grad-ckpt KV)
│   ├── train_origami.py            vendored full-pipeline:midtrain.py + single-GPU knobs
│   ├── delayed_lerobot_dataset.py  TRexLeRobotDataset + tactile delay curriculum (§7.5-B)
│   ├── train_origami.sh            the recipe
│   ├── policy.py                   T-Rex checkpoint → eef-62 chunk (slow/fast)
│   ├── retarget.py                 eef-62 chunk → 65-D absolute joints via IK
│   ├── serve_zenoh.py              origami-zenoh-v1 TeamPolicy + server
│   ├── eval_offline.py             held-out-season metrics (+ --zero-shot)
│   ├── eval_shadow.py              SDK Shadow / wire-contract check
│   ├── diagnose_shift.py           pretrain↔origami shift report (§11.9)
│   ├── refit_vqvae_stats.py        re-fit tacf6_vqvae_{min,max,mask} (§11.8)
│   ├── bench_latency.py            slow/fast latency → feasible action_horizon
│   ├── docker/Dockerfile           submission image
│   └── tests/                      pytest, CPU-only, no GPU, no network
└── pyproject.toml                  origami package deps (uv)
```

### 2.1 Deletions / git hygiene (step 0 of implementation)

1. `T-Rex/` is currently a fresh clone **with its own `.git`**. Make it a submodule:
   ```bash
   cd /home/kshitij/origami_trex
   git rm -r --cached T-Rex                 # drop the 1000+ old blobs from the index
   rm -rf T-Rex
   git submodule add https://github.com/ZhuoyangLiu2005/T-Rex.git T-Rex
   git -C T-Rex checkout f88e10c
   git -C T-Rex fetch origin full-pipeline      # pinned at b23eafe -- source for §7.5
   git add .gitmodules T-Rex
   ```
   A submodule pins one commit, so `full-pipeline` is *fetched but not checked out*;
   `origami/train_origami.py` is vendored from it once
   (`git -C T-Rex show origin/full-pipeline:scripts/midtrain.py`) and
   `origami/tests/test_upstream_drift.py` hashes `b23eafe`'s copy so drift is detected.
   (If a submodule is unwanted, instead `rm -rf T-Rex/.git` and commit the pristine files —
   but then add `origami/tests/test_upstream_drift.py`, §10 gate G0.)
2. Delete: `T-Rex_modified/` (the previous fork — keep on a branch, not in the tree),
   `OLD_REIMPLEMENTATION_PLAN.md`, `prepare_full.log`, `train_pilot.log`,
   `trex_colab.ipynb` (rewritten in §8.4), `reimplementation/`, `delta_run/`.
3. `.gitignore` additions: `data/`, `out/`, `*.log`, `wandb/`, `outputs/`, `.pytest_cache/`.
4. Keep `season_POC22061_2026_07_23_10_20_33_train/` as the fixture for tests
   (it is the only local season; add it to `.gitignore` if not already tracked).

---

## 3. `origami/kinematics.py`

The most safety-critical module. It must be **the single source of FK/IK** for prep,
offline eval, and deploy — never duplicated.

### 3.1 API

```python
HAND_ORDER: tuple[str, ...]          # 22 names, §1.1
ARM_ORDER:  tuple[str, ...]          # ("arm_joint_1", ..., "arm_joint_7")
JOINT_NAMES_65: tuple[str, ...]      # the 65 URDF names in dataset-index order
EEF_FRAMES = {"left": "left_hand_base_link", "right": "right_hand_base_link"}

@dataclass(frozen=True)
class LockedConfig:
    lower_body: np.ndarray   # (5,)
    neck:       np.ndarray   # (2,)
    left_hand:  np.ndarray   # (22,)  -- irrelevant to arm FK, locked for a clean 14-DOF model
    right_hand: np.ndarray   # (22,)
    def digest(self) -> str  # sha1 of the rounded values; recorded in prep metadata

class OrigamiKinematics:
    def __init__(self, urdf_path: str, locked: LockedConfig): ...
    # 14-DOF reduced model: everything except left_arm_joint_1..7 + right_arm_joint_1..7 locked.
    # asserts self.robot.nq == self.robot.nv == 14
    # asserts set(model.names[1:]) == {"left_arm_joint_1".."right_arm_joint_7"}

    def fk(self, q_left7, q_right7) -> dict[str, pin.SE3]        # {"left": SE3, "right": SE3}
    def fk_matrices(self, q_left7, q_right7) -> tuple[np.ndarray, np.ndarray]   # two 4×4
    def solve_ik(self, target_left: pin.SE3, target_right: pin.SE3,
                 warm_left7, warm_right7) -> tuple[np.ndarray, np.ndarray]
    def clip_arm(self, q_left7, q_right7) -> tuple[np.ndarray, np.ndarray]
    arm_limits: dict[str, tuple[np.ndarray, np.ndarray]]         # from the URDF
```

* `__init__` builds the model exactly like `robot_descriptions.py::build_reduced_bimanual_robot`:
  `RobotWrapper.BuildFromURDF(urdf_path, [mesh_dir])`; assemble a full `pin.neutral(model)`,
  write the locked values in at the right `idx_q`, `pin.buildReducedModel(model,
  [collision_model, visual_model], joints_to_lock_ids, full_qpos)`.
  Collision pairs: `addAllCollisionPairs()` then remove within-hand and between-hand pairs.
  **No SRDF is shipped for this robot** — instead build the disable list programmatically
  (adjacent links in the URDF tree + all hand↔hand pairs) and expose
  `check_collision(q_left7, q_right7) -> bool` for deploy gating. Do not use collisions inside
  the Pink `Configuration` (upstream comments say it slows Pink down).
* `fk`: `pin.forwardKinematics` → `pin.updateFramePlacements` → `data.oMf[getFrameId(frame)].copy()`.
* `solve_ik`: copy `ik_utils.py::PinkLocalIK.solve_ik` (§1.4) **verbatim**, substituting
  `"L_ee"/"R_ee"` → `EEF_FRAMES`, and `default_qpos` → the `left/right_arm_default_joint_pos`
  from §3.3.

### 3.2 Why locking `lower_body`/`neck` at a constant is exact, not an approximation

The arms hang off `torso_base_link`, which is downstream of `lower_body_joint_1..5` only.
Therefore

```
T_root_eef(q_lb, q_arm) = T_root_torso(q_lb) · T_torso_eef(q_arm)
```

and the whole action representation is built from

```
delta9(A, B) = [ A_R^T (B_t − A_t),  (A_R^T B_R)[:,0],  (A_R^T B_R)[:,1] ]
```

which is **invariant** under a common left-multiplied rigid transform `X`:
`delta9(X·A, X·B) = delta9(A, B)`. Both `A` and `B` come from the same locked
`T_root_torso`, so the action is independent of the lock value. The *absolute* 9-D state
(`observation.state[0:9]`) is not invariant, but as long as prep and deploy use the **same**
`LockedConfig`, it is consistent — and the q01/q99 stats are computed on that same frame.

**Requirement:** `LockedConfig` is written into `meta/origami_prep.json` at conversion time
and loaded (not re-chosen) by `eval_offline.py` and `serve_zenoh.py`.
`OrigamiKinematics.__init__` must refuse a `LockedConfig` whose `digest()` disagrees with the
one recorded in the dataset/checkpoint (gate G1c).

### 3.3 `LockedConfig` values

Computed once over the training split and frozen:

```
lower_body = per-dim median of state[:, 58:63]   over all training frames
neck       = per-dim median of state[:, 63:65]
left_hand  = zeros(22)      # hand joints are locked-out of the reduced model; value irrelevant
right_hand = zeros(22)
```
Arm posture-regularization target (`default_qpos`, the `cost=0.05` PostureTask):
```
left_arm_default  = per-dim median of state[:, 0:7]
right_arm_default = per-dim median of state[:, 29:36]
```
Emit these into `meta/origami_prep.json` together with `urdf_sha256`.

For the local fixture season the medians are approximately
`lower_body ≈ [0.563, −1.107, 0.55, 0.0, 0.0]`, `neck ≈ [0.0, −0.60]` — the implementation
must compute them, not hard-code them.

---

## 4. `origami/decode.py`

### 4.1 Episode-slice decoding

```python
def decode_episode_stream(
    video_path: str, from_ts: float, to_ts: float, n_frames: int,
    fmt: str,                      # "rgb24" | "gray"
) -> Iterator[np.ndarray]
```
* Open with PyAV. Seek: `container.seek(int(from_ts / stream.time_base), stream=stream)`,
  then **drop frames until `frame.pts * time_base >= from_ts − 0.5/fps`**, then yield exactly
  `n_frames`. Seeking lands on the preceding keyframe, so pre-roll decoding is mandatory.
* If the stream yields fewer than `n_frames`, raise — do **not** silently pad. The caller
  clamps the episode to `min_len` across streams and logs it (§4.4).
* `fmt="gray"` for `tactile_deform` (luma plane, §1.2), `"rgb24"` for the RGB cameras.
* Robustness: wrap in a retry that falls back to a linear scan from t=0 if the seek path
  produces a PTS discontinuity. Log every fallback (they are slow).

### 4.2 Deform splitting

```python
def split_deform_strip(y: np.ndarray) -> np.ndarray:
    """(480, 1200) uint8 luma -> (10, 240, 240) uint8, [L thumb..pinky, R thumb..pinky]."""
    assert y.shape == (480, 1200)
    return y.reshape(2, 240, 5, 240).transpose(0, 2, 1, 3).reshape(10, 240, 240)
```
Also `split_raw_strip` for the optional 480×1600 → `(10, 240, 320)`.

### 4.3 RGB resize (D7)

```python
def squash_to_wire(rgb480: np.ndarray) -> np.ndarray:
    """(480,480,3) uint8 RGB -> (224,224,3) uint8, deploy-identical."""
    return cv2.resize(rgb480, (224, 224), interpolation=cv2.INTER_AREA)
```
* Prep stores 224×224. The loader's `_img_to_pil` then upsamples to `--image_size 384 384`
  with `PIL.Image.LANCZOS`; `serve_zenoh.py` applies **the same** LANCZOS 224→384 step. The
  two pipelines are then bit-identical from the wire onward.
* `INTER_AREA` for the 480→224 downscale (correct for decimation). Document that the
  organizer's own 1920×1536→224 squash may use a different filter; this only affects
  training-vs-deploy sharpness, not geometry, and 224 is the common bottleneck.
* `--image_size` is square (not upstream's `384 288`) because origami frames are square:
  `384 288` would introduce a 1.33× horizontal stretch that upstream's 4:3 crops did not
  have. Size selection is §4.5.

### 4.4 Episode length reconciliation

For each episode take `N = min(length_from_meta, frames_decoded_per_stream…)`. Log any
episode where `N < length` and record `(season, episode, length, N)` in
`meta/origami_prep.json["truncated"]`. Reject the episode entirely if `N < 64`
(too short for one chunk + one F6 window).

### 4.5 Vision token budget and view allocation

**The resolution gap versus T-Rex is real and cannot be closed at deploy.**

| | T-Rex head | Origami head |
|---|---|---|
| native | 640×360 | 1920×1536 |
| preprocessing | crop `(0,300,140,540)` → 400×300 | squash → 480×480, then 224×224 |
| fed to the ViT | 384×288 | see below |
| effective scale | **≈ 1 : 1** (400 → 392 up) | **8.6× linear downsample** |

T-Rex's head view is a tight, near-lossless workspace crop. Ours is the whole room at
1/8.6 scale, and `robot_io_spec.md` forbids recovering it ("the organizer directly squashes
it to 224×224 … participants must not restore the native aspect ratio"). Measured on the
fixture: the 15×15 cm sheet spans roughly 60×40 px of the 480² frame → **≈ 28×19 px at 224²**.
**Crease geometry is not present in the head view.** The usable fine-grained signal is in the
two wrist views (the paper fills a large fraction of each fisheye) and in tactile. Design
accordingly.

**A. Probe the processor factor before assuming any token count.**
`smart_resize` (`transformers/models/qwen2_vl/image_processing_qwen2_vl.py`) rounds each
side to `factor = image_processor.patch_size * image_processor.merge_size`, taken from the
checkpoint's `preprocessor_config.json` — **not** from the vision config. `Qwen3VLVisionConfig`
defaults are `patch_size=16, spatial_merge_size=2` (factor 32) but the `Qwen2VLImageProcessor`
class default is `14 × 2` (factor 28). Tokens per image = `(h/f) · (w/f)`:

| `--image_size` | f = 32 | f = 28 |
|---|---|---|
| `384 288` (upstream) | 12×9 = **108** | 14×10 = **140** |
| `384 384` | 12×12 = **144** | 14×14 = **196** |
| `336 336` | — (rounds to 320 → 100) | 12×12 = **144** |
| `320 320` | 10×10 = **100** | 11×11 = **121** |
| `224 224` | 7×7 = **49** | 8×8 = **64** |

`origami/tests/test_processor_probe.py` prints
`processor.image_processor.patch_size * merge_size` and the resolved grid for each candidate,
then **picks the square size whose token count is closest to upstream's `384 288`** so the
midtrain ViT operates near its trained point. Under f=32 that is `384 384` (144 vs 108);
under f=28 it is `336 336` (144 vs 140). Record the chosen value in `training_args.json`
(gate G16). Note this also scales the FLARE cost: `n_flare_steps 8` pushes 8 extra head
frames per sample through the no-grad ViT, so tokens/image multiplies by 9 for the head.

**B. Per-view `image_size` — reallocate tokens toward the wrists.**
`collate_fn` (`lerobot_dataset.py:203-217`) calls the processor **once** with
`pil_slow + pil_fast`, and Qwen3-VL computes an independent grid per image. So different
sizes per view need no architecture change — only that `_img_to_pil` take a per-key size.
Add `--image_size_head W H` / `--image_size_wrist W H` (both defaulting to `--image_size`)
in the vendored loader subclass, and mirror them in `serve_zenoh.py`. Recommended first
ablation: head at the token-matched size, **wrists one step larger**, since that is where
the crease signal is. Constraint to respect: `split_slow_fast_embeds` derives
`n_slow_img_tokens` from the first `n_slow` grid rows (`train.py:1050-1060`), so the slow/fast
boundary stays correct automatically as long as the slow images come first in the list.

**C. Optional workspace crop — applied identically on both sides.**
The top ~35% of the head frame is dark background (measured row-mean ≈ 8/255) and the bottom
band is the robot's own chassis. Cropping adds no pixels but reallocates ViT tokens onto the
paper and removes distractors. Rules, non-negotiable:
* the crop is applied **after** the 480→224 squash, so `serve_zenoh.py` reproduces it exactly
  from the 224 wire image (cropping the 480 source would give training more real detail than
  deploy can ever supply — a silent train/deploy break);
* the crop is **square**, so we are not "restoring the native aspect ratio";
* the relative box is recorded in `meta/origami_prep.json` and `training_args.json`, and
  gate G12 compares the two paths pixel-exactly.
Expose `--head_crop_rel y0 y1 x0 x1` in *normalized* coordinates (default `0 1 0 1` = off).
A reasonable first candidate from the fixture is `0.25 1.0 0.0 1.0` on the head — but it is
not square, so either pad the shorter axis by widening the row span or accept a 1×1.33 box
and set `--image_size_head` to match its aspect. **Treat this as an ablation, not the
default**: the plan's default is no crop (D7), and the crop's value must be shown on the
§8.1-A/C metrics before adoption.

**D. `head_right` is unused by default.** T-Rex has exactly one slow camera, and the
organizer's own reference policy also consumes only `head_left`, `wrist_left`, `wrist_right`
(`openpi-base-main/src/openpi/policies/north_ces_policy.py:52-70`) — independent
confirmation of this view choice. The rig does provide a stereo head pair, and
`split_slow_fast_embeds` supports `n_slow_images = 2`, so adding `head_right` as a second
slow image is a legitimate ablation (stereo depth cues for paper/crease geometry) at the cost
of +1 ViT pass and tokens the midtrain checkpoint never saw. Requires refetching
`head_right` (+844 MB/season) — so decide **before** the full prep run, or accept a refetch.

---

## 5. Conversion

### 5.1 `origami/fetch.py`

```python
HF_REPO_ID = "SharpaIT/Robotic_Origami_Challenge"
NEEDED_VIDEO_KEYS = ("observation.images.head_left",
                     "observation.images.wrist_left",
                     "observation.images.wrist_right",
                     "observation.images.tactile_deform")   # + tactile_raw if --include-raw

def season_allow_patterns(season, include_raw=False) -> list[str]:
    base = f"{season}/lerobot3.0"
    return [f"{base}/meta/**", f"{base}/data/**"] + [f"{base}/videos/{k}/**" for k in keys]

def download_season(season, cache_root, token=None, include_raw=False) -> str
def have_season(season, cache_root, include_raw=False) -> bool   # checks real files, not dirs
def drop_season(season, cache_root) -> None
```
* `huggingface_hub.snapshot_download(repo_id, repo_type="dataset", allow_patterns=...,
  local_dir=..., max_workers=8)`.
* `have_season` must glob for actual `*.parquet` / `*.mp4` files — an interrupted download
  leaves the directory tree with no media.
* Retry with exponential backoff on `HfHubHTTPError`; 3 attempts, then mark the season failed
  and continue.

### 5.2 `origami/convert.py` — per-season shard

```
convert_season(season_dir, out_root, kin, cfg) -> SeasonResult
```
Steps, per episode `e` (in `episode_index` order):

1. Read the episode row from `meta/episodes/**.parquet` → `length`, `dataset_from_index`,
   `dataset_to_index`, and per-key `(chunk_index, file_index, from_timestamp)`.
   `tasks[0]` is read only to assert it is the expected `"north ces task"`; the written
   instruction is `cfg.instruction` (§5.7).
2. Slice the season parquet rows `[dataset_from_index : dataset_to_index]` →
   `state65 (N,65) f64`, `action65 (N,65) f64`, `tactile60 (N,60) f32`.
   **Cast joints to float64 before FK.** FK/IK in float32 loses ~1e-4 rad of round-trip
   accuracy, which is comparable to the 0.39 mm/frame signal.
3. Open the 4 video streams for this episode (§4.1) — one thread each, feeding bounded queues.
4. `N = min(...)` (§4.4).
5. FK, vectorized over the episode:
   ```python
   S_l = [kin.fk_matrices(state65[i,0:7],  state65[i,29:36])[0] for i in range(N)]   # (N,4,4)
   S_r = [                                                  [1]                   ]
   A_l = [kin.fk_matrices(action65[i,0:7], action65[i,29:36])[0] for i in range(N)]
   A_r = [                                                  [1]                   ]
   ```
   (One `forwardKinematics` call per frame returns both arms — do **not** call it twice.)
   Then
   ```python
   s_l_9d = pose_matrix_to_9d(S_l);  s_r_9d = pose_matrix_to_9d(S_r)
   a_l_9d = pose_matrix_to_9d(A_l);  a_r_9d = pose_matrix_to_9d(A_r)
   s_l_hnd = state65[:, 7:29];   s_r_hnd = state65[:, 36:58]
   a_l_hnd = action65[:, 7:29];  a_r_hnd = action65[:, 36:58]
   states      = np.concatenate([s_l_9d, s_l_hnd, s_r_9d, s_r_hnd], axis=1)   # (N,62)
   abs_targets = np.concatenate([a_l_9d, a_l_hnd, a_r_9d, a_r_hnd], axis=1)   # (N,62)
   ```
6. Per frame `i ∈ [0, N)`:
   ```python
   chunk = build_action_chunk(S_l, A_l, a_l_hnd, S_r, A_r, a_r_hnd, i, N)   # (16,62) f32
   tgt   = min(i + FRAME_STRIDE - 1, N - 1)     # == i for FRAME_STRIDE=1
   tac   = tactile60[i].reshape(10, 6).astype(np.float32)
   deform10 = split_deform_strip(next(deform_iter))       # (10,240,240) uint8
   frame = {
       "task": cfg.instruction,          # §5.7 -- NOT the dataset's "north ces task"
       "observation.images.head":        squash_to_wire(next(head_iter)),
       "observation.images.wrist_right": squash_to_wire(next(wr_iter)),
       "observation.images.wrist_left":  squash_to_wire(next(wl_iter)),
       "observation.state":              states[i].astype(np.float32),
       "action":                         chunk,
       "action_abs":                     abs_targets[tgt].astype(np.float32),
       "observation.tactile_f6":         tac,
       **{DEFORM_KEYS[k]: gray_to_3ch(deform10[k]) for k in range(10)},
   }
   ds.add_frame(frame); acc.add_frame(chunk, states[i], tac)
   ```
   `build_action_chunk` is **imported from `T-Rex/utils/lerobot_common.py:100`**, not
   re-implemented. Its clamp is `fut = min(min(i + k*FRAME_STRIDE, N-1) + FRAME_STRIDE - 1, N-1)`
   which for `FRAME_STRIDE=1` is `min(i+k, N-1)` — i.e. **the chunk clamps at the episode end
   and never crosses into the next episode**, because `N` is this episode's length.
   `gray_to_3ch(a) = np.repeat(a[:, :, None], 3, axis=2)` (matches
   `convert_inlab_to_lerobot.py::_gray_to_3ch`).
7. `ds.save_episode()`; `acc.add_episode_tracking(states[:N], abs_targets[:N])`.

After all episodes: `ds.finalize()`, write the shard's stat state (`acc.dump(path)`).

**Writer construction:**
```python
features = build_trex_features(
    head_shape=(3, 224, 224), include_wrist=True, wrist_shape=(3, 224, 224),
    include_tactile=True, deform_shape=(3, 240, 240), include_action_abs=True)
ds = LeRobotDataset.create(repo_id=f"origami/eef62_{season}", fps=30, features=features,
                           root=shard_root, robot_type="north_poc2_2", use_videos=True)
```
Set `chunks_size` / `data_files_size_in_mb` / `video_files_size_in_mb` **small enough that
each episode ends up in its own data/video file** (e.g. `data_files_size_in_mb=64`,
`video_files_size_in_mb=128`). This makes §5.3 a pure file-copy-and-renumber.

**Video encoding.** The deform videos carry physically meaningful uint8 depth. Encode them
**losslessly** (`libx264 -qp 0` / `-crf 0`, or `ffv1`) so the decoded channel-0 equals the
source luma exactly. Expose `--deform_codec {lossless_h264, ffv1, crf18}`; default
`lossless_h264`; gate G6 checks the round-trip. RGB cameras use LeRobot's default codec.

### 5.3 `origami/merge.py`

**First: probe for `lerobot.datasets.aggregate.aggregate_datasets` (or
`lerobot.datasets.utils.aggregate_datasets`) in the pinned version. If it exists, use it and
delete the hand-rolled merger.** Only if it does not:

`merge_shards(shard_roots: list[str], out_root: str)`:
1. Read each shard's `meta/info.json`; assert identical `features`, `fps`, `codebase_version`.
2. Copy `data/**` and `videos/**` into `out_root` with `(chunk_index, file_index)` renumbered
   into a single global sequence per key; keep a `map[(shard, key, chunk, file)] → (chunk, file)`.
3. Rewrite `meta/episodes/**.parquet`: renumber `episode_index`, recompute
   `dataset_from_index` / `dataset_to_index` cumulatively, remap every
   `data/{chunk,file}_index` and `videos/<key>/{chunk,file}_index`.
   **`from_timestamp`/`to_timestamp` are per-file and stay unchanged** when files are copied
   whole — which is why §5.2 forces one episode per file.
4. Rewrite the per-row `index` column in the copied data parquets to be globally increasing
   (LeRobot uses it as the flat frame id). `episode_index` in the data parquet must be
   renumbered too.
5. Merge `meta/tasks.parquet` (single task → trivial) and rewrite `meta/stats.json` (LeRobot's
   own min/max/mean/std) by weighted combination, or regenerate it with LeRobot's helper.
6. Merge the per-shard stat states → write `meta/trex_norm_stats.json` (§5.4) and
   `meta/origami_prep.json`.
7. Rewrite `meta/info.json`: `total_episodes`, `total_frames`, `total_tasks`, `splits`.

Two output roots are produced: `origami_eef62_train/` (101 seasons) and
`origami_eef62_val/` (25 seasons), per the split in `dataset.md`. `--origami_val_root`
does not exist in the LeRobot loader, so the val root is passed via a separate
`TRexLeRobotDataset` instantiation (§7.2).

### 5.4 `origami/stats.py` — bounded-memory q01/q99

**`T-Rex/utils/lerobot_common.py::NormStatsAccumulator` keeps every frame in RAM:
4 763 267 × 16 × 62 × 4 B = 18.9 GB for the action block alone. It will OOM.**
Replace it with an API-compatible accumulator:

```python
class StreamingNormStats:
    """Exact mean/std/min/max over all frames; reservoir q01/q99."""
    def __init__(self, reservoir: int = 200_000, seed: int = 0)
    def add_frame(self, action_chunk, state, tactile_f6=None)     # same signature
    def add_episode_tracking(self, states_62, abs_targets_62)
    def merge(self, other) -> None                                # for shard merging
    def assemble(self) -> dict                                    # same schema as upstream
    def write(self, dataset_root) -> str                          # meta/trex_norm_stats.json
    def dump(self, path) / classmethod load(path)                 # shard checkpointing
```
* **mean/std/min/max:** exact, streaming (`n`, `Σx`, `Σx²`, `min`, `max` per element;
  shapes `[16,62]` for action, `[62]` for state, `[60]` for tactile, `[56]` for tracking error).
  Use float64 accumulators.
* **q01/q99:** uniform reservoir sample (Vitter's Algorithm R) of `reservoir` frames per block.
  Memory: action `200 000 × 16 × 62 × 4 B = 794 MB`; state 50 MB; tactile 48 MB; tracking 45 MB.
  Cap the action reservoir at 100 000 if RAM is tight (397 MB) — gate G8 bounds the error.
  `merge()` combines two reservoirs by weighted subsampling proportional to `n`.
* **Degenerate-dim masking.** After assembling, for each block set
  `mask[j] = (q99[j] - q01[j]) > eps` with `eps = 1e-6` for `state`/`action` and `1e-3`
  (N or N·m) for `tactile_f6`. Dims with `mask=False` pass through `_normalize` unchanged
  (`lerobot_dataset.py:71` uses `np.where(mask, ...)`), which is exactly what we want for the
  dead ring/pinky tactile channels (§1.2). Record the masked-out dim indices and names in
  `meta/origami_prep.json` and print them — **an all-`True` tactile mask on this dataset is a bug.**
* **Tracking-error robustness.** The first frames of an episode show a >1 rad state↔action gap
  on `thumb_CMC_FE` (state 0.100 vs action 1.139) — a hand-init transient. Compute
  `tracking_error` over `t ∈ [30, N)` (skip the first second) **and** report the
  full-episode version under `tracking_error_full` for comparison. Only `mean`/`std` are used,
  and only when `--use_robot_state 1`.
* Keep `STATS_KEY = "rlbench"` so the file is byte-schema-identical to upstream. (`test.py:206`
  falls back to `next(iter(stats_raw))`, and `lerobot_dataset.py:107` always takes the first
  key, so the name is free — but there is no reason to diverge.)

### 5.5 `origami/prepare.py` — orchestration

```
python -m origami.prepare \
  --split train --out-root /mnt/big/origami/eef62_train \
  --cache-root /mnt/big/origami/_src --workers 3 --disk-budget 3 \
  --urdf north_poc2_2_urdf_usd/north_poc2_2_v3_1.urdf \
  --locked-config out/locked_config.json
```
* Phase 0 (`--phase locked`): scan the training seasons' `data/*.parquet` **only** (66 MB/season,
  no video) to compute the `LockedConfig` medians and the arm posture defaults; write
  `locked_config.json`. Must run before any conversion.
* Phase 1: `ProcessPoolExecutor(workers)` over seasons; a `ThreadPoolExecutor` prefetches
  downloads so at most `disk-budget` seasons are on disk at once. Each worker:
  `download → convert_season → acc.dump → drop_season`. A season already present in
  `out_root/_shards/<season>/` with a `DONE` marker is skipped.
* Phase 2: `merge.merge_shards(...)`; then `verify.py --root out_root` (gates G4–G8).
* Resumability: per-season `DONE` markers + a `manifest.jsonl` recording
  `(season, n_episodes, n_frames, truncated, elapsed_s, sha of LockedConfig)`.
  Changing the `LockedConfig`, `--image-size`, `--deform-codec` or the chunk parameters
  **invalidates the whole root** — `prepare.py` must refuse to write into a root whose
  `meta/origami_prep.json` records different values.

### 5.6 Storage and time budget (state these in the CLI's opening banner)

Per season, measured on the fixture (95 533 frames):

| artifact | size |
|---|---|
| `action[16,62]` f32 | 3 968 B/frame → **379 MB** |
| `state[62]` + `action_abs[62]` + `tactile_f6[10,6]` f32 | 736 B/frame → 70 MB |
| head + 2 wrists @ 224² | ~250 MB |
| 10 deform videos @ 240², lossless | ~150 MB |
| **total** | **≈ 0.85 GB/season → ≈ 43 GB for 51 seasons** |

Source download 50–160 GB (stream-and-delete, peak ≈ `disk-budget × 3 GB`).
**Local free disk is 23 GB — `--out-root` and `--cache-root` must point at external storage.**
`prepare.py` must `statvfs` both paths and refuse to start if free space
< `output_estimate + disk_budget × 3.5 GB`.

**Optional `--action-layout abs`** (documented, not default): store only `action_abs[62]`
and derive the `[16,62]` chunk inside a `TRexLeRobotDataset` subclass using
`delta_timestamps["action_abs"] = [k/fps for k in range(16)]` plus
`compute_chunk_delta_pose(state_pose, target_pose)` in `collate_fn`. 16× smaller
(43 GB → 8 GB) and produces **numerically identical tensors** — gate G4b proves it.
Default remains the baked layout because it is the exact upstream schema.

### 5.7 Language instruction

The dataset's stored task string is `"north ces task"` — it names the rig, not the task, and
carries no information the vision tower does not already have. Upstream T-Rex's own defaults
are no better (`"I am T-Rex."` in `gen_json_bimanual.sh:17` and
`convert_inlab_to_lerobot.sh:16`), though its `convert_inlab_to_lerobot.py:48`
`DEFAULT_INSTRUCTION` shows the intended style: one long, concrete, sequential sentence.

**The organizer's reference prompt for this task is `"fold the plane"`**
(`openpi-base-main/src/openpi/policies/north_ces_policy.py:33`, in
`make_north_ces_example()`). That is what the gateway is expected to send in
`observation["prompt"]`, but it is not contractually pinned — the spec only requires that
`prompt` be a string.

`origami/constants.py`:

```python
# The instruction written into every training frame and re-supplied verbatim at
# deploy.  Long-form and step-ordered, matching the style of T-Rex's own
# post-train instructions (utils/convert_inlab_to_lerobot.py:48) and the 6-fold
# sequence documented in dataset.md.
INSTRUCTION = (
    "Fold the square sheet of paper into a traditional paper airplane: crease it in half "
    "along the centre line and unfold, fold the two top corners down to meet the centre "
    "crease, fold the resulting angled edges in to the centre line again, then fold the "
    "body in half along the crease and fold one wing down on each side."
)

# The organizer's reference prompt (north_ces_policy.py:33).  Kept so the deploy
# server can log a mismatch against whatever the gateway actually sends.
WIRE_PROMPT_REFERENCE = "fold the plane"

# Short alias, for the --instruction-style short ablation.
INSTRUCTION_SHORT = "fold the plane"
```

* `convert.py` takes `--instruction` (default `INSTRUCTION`) and writes it as the LeRobot
  `task` for every frame. It lands in `meta/tasks.parquet`, the loader surfaces it as
  `x["task"]`, and `TRexLeRobotDataset.collate_fn` (`lerobot_dataset.py:203`) puts it in the
  Qwen chat template **between** the slow image and the two fast images — exactly where
  upstream puts it.
* The instruction is recorded verbatim in `meta/origami_prep.json` and copied into
  `training_args.json` by `train_origami.py` (§7.2 delta 4).
* **At deploy, `serve_zenoh.py` uses the instruction from `training_args.json` and ignores
  `observation["prompt"]`** — a single-task policy trained on one string must be fed that
  same string, and the contract explicitly makes this ours to decide: *"language
  tokenization is participant-internal."* Log a one-time warning when the received prompt
  differs from `WIRE_PROMPT_REFERENCE`, and expose `--use-wire-prompt` to pass it through
  for robustness testing. This replaces the `obs["prompt"]` read in §9.4.
* Because the dataset is single-task, the instruction is a constant, not a label. Its only
  real jobs are (a) to sit in-distribution for the midtrain checkpoint's language prior and
  (b) to be identical between training and deploy. Gate **G14** enforces (b).
* Do **not** append per-fold progress hints ("fold 3 of 6"). The fold index is unknown at
  deploy, so any such conditioning is unusable and would train the policy to depend on it.

---

## 6. The exact tensor definitions (single source of truth)

```
FK9(q7_side) : 7 joint angles -> 9-D
    T = kin.fk_matrices(q_left7, q_right7)[side]        # 4x4, torso-locked frame
    FK9 = [ T[:3,3], T[:3,0], T[:3,1] ]                 # trans, R col1, R col2

state[62]      = [ FK9(state65[0:7]),  state65[7:29],
                   FK9(state65[29:36]), state65[36:58] ]
action_abs[62] = [ FK9(action65[0:7]), action65[7:29],
                   FK9(action65[29:36]), action65[36:58] ]

delta9(A4x4, B4x4) = [ A_R^T (B_t - A_t), (A_R^T B_R)[:,0], (A_R^T B_R)[:,1] ]

action[k, :]   = [ delta9( FK4(state65[i,0:7]),   FK4(action65[f,0:7])  ), action65[f, 7:29],
                   delta9( FK4(state65[i,29:36]), FK4(action65[f,29:36])), action65[f, 36:58] ]
                 where f = min(i + k, N - 1),  k = 0..15
                 (base pose is at frame i for ALL k — chunk-base, not frame-to-frame)

tactile_f6[10,6] = observation.tactile[i].reshape(10, 6)

tracking_error[56] = compute_bimanual_tracking_error(state[t], action_abs[t-1])
                     = per arm: [ R_s^T(t_a - t_s), Rodrigues(R_s^T R_a), hand_a - hand_s ]
```

> **NEW_DESIGN.md §Step 3 says "the change in position between sequential frames".
> That is wrong.** The upstream code
> (`gen_json_tac_deltabase_eef_bimanual_parallel.py:210-235`, mirrored in
> `lerobot_common.py:100-125`) uses a **single chunk-base pose at frame `i` for all 16 steps**,
> and the target comes from the **`action` (commanded) joints**, not the next `state`.
> `eval_trex_async.py:1006-1129` confirms it at deploy: `initial_*_ee_pose` is captured once
> per chunk and every row's delta is applied to that same anchor. Implement the code, not the doc.

---

## 7. Training

### 7.1 `origami/trex_patch.py`

Two monkeypatches, applied by importing this module before `train`/`test` code.
Both are needed only because we enable gradient checkpointing on one GPU; both are no-ops
otherwise, and each ships with a unit test proving equivalence.

1. **`Qwen3VLAttentionMoT.forward` cached-prefix read.**
   Upstream (`modeling_qwen3vl_mot.py`, the `if past_key_value is not None:` branch) always
   calls `past_key_value.update(...)`, which **appends**. `update()` is not idempotent, so
   under gradient checkpointing the backward recompute of the cascaded tactile step appends
   this step's K/V a second time and the KV length outgrows the causal mask built for
   `past_len + seq_len`. Patch: when `torch.is_grad_enabled()`, read the stored prefix
   without appending and concatenate:
   ```python
   def read_cached_kv(cache, layer_idx):
       layers = getattr(cache, "layers", None)
       if isinstance(layers, list):
           if layer_idx < len(layers):
               l = layers[layer_idx]
               k = getattr(l, "keys", None)
               if k is not None and k.numel() > 0: return k, l.values
           return None
       k = getattr(cache, "key_cache", None)
       if k is not None and layer_idx < len(k) and k[layer_idx] is not None:
           return k[layer_idx], cache.value_cache[layer_idx]
       return None
   ```
   Nothing downstream consumes the appended entries during training, so the read-only concat
   is equivalent and safe to replay.
2. **Drop `attention_mask` on cached-prefix decoder calls.** Upstream already omits it in the
   `past_kv is not None` branches of `forward_flow_action_full` /
   `forward_flow_action_partial` (only `modeling_vla.py:613` and `:706`, the `past_kv is None`
   branches, pass it). Verify at import time that the omission holds in the pinned commit;
   if a future upstream adds it back, patch it out (a `[B, L_slow]` mask against an
   action-only `inputs_embeds` is a length mismatch).

`trex_patch.apply()` must be idempotent and must assert the pinned upstream source hashes
before patching (fail loudly rather than patch a changed file).

### 7.2 `origami/train_origami.py`

A **vendored copy of `T-Rex@full-pipeline:scripts/midtrain.py`**, not `main:scripts/train.py`
(T-Rex stays pristine per D2). See §7.5 for why. `midtrain.py` is a strict **superset** of
`train.py`: identical model, identical cascaded-flow loss, identical `--data_format lerobot`
→ `TRexLeRobotDataset` dispatch (`midtrain.py:1455-1459`), plus seven extra flags
(`--tactile_delay_offsets`, `--frame_stride`, `--num_workers`, `--n_fingers`,
`--tactile_code_per_finger`, `--tactile_f6_stats_ckpt`, `--vqvae_codes_h5_name`).
`comm` on the two argument lists shows **zero** flags present in `train.py` but absent from
`midtrain.py`, so nothing in §7.3's recipe is lost.

Header listing exactly these deltas and nothing else:

1. `import origami.trex_patch as _p; _p.apply()` at the top.
2. `_world_size()` helper replacing bare `dist.get_world_size()`
   (`midtrain.py:1106` and `:1312`) so a non-distributed launch works; guard the
   `dist.all_reduce` calls with `dist.is_initialized()`; guard the
   `accelerator.state.deepspeed_plugin` writes with a `None` check.
3. New flags: `--save_steps`, `--max_steps`, `--save_optimizer_state`,
   `--resume_full_state`, `--gradient_checkpointing`, `--freeze_latent_expert`,
   `--train_latent_last_n`, `--optim {adamw,adamw8bit}`, `--action_loss_weight`
   (§7.5-C). `--num_workers` and `--frame_stride` already exist upstream — one fewer
   delta than the `train.py` route.
4. `save_checkpoint` additionally writes `training_state.json`
   (`epoch, global_step, learning_rate, warmup_rates, min_lr_ratio`), `state/` via
   `accelerator.save_state`, and adds `instruction` (§5.7), `image_size`, `vqvae_window`, `locked_config`,
   `origami_prep` to `training_args.json`. **`locked_config` in the checkpoint is what
   `serve_zenoh.py` loads** — this is the mechanism behind gate G1c.
5. `--lerobot_val_root`: build a second `TRexLeRobotDataset` on the held-out-season root
   instead of `create_val_split`'s within-root episode split. (The 101/25 season split in
   `dataset.md` is the correct held-out protocol; a random episode split leaks a session.)
6. `train_flare = use_flare and flare_loss_weight > 0` so `--flare_loss_weight 0` keeps the K
   latent query tokens (sequence-shape parity with the resumed checkpoint) while skipping the
   extra no-grad ViT pass. Needed when the latent expert is frozen.
7. `--max_steps` truncates `num_training_steps` before building the LR schedule
   (otherwise the cosine schedule is computed for a 100-epoch run and barely decays).

8. `--lerobot_delay_offsets` plumbing into the loader subclass of §7.5-B (the LeRobot path
   ignores `--tactile_delay_offsets` as shipped).

Add `origami/tests/test_train_vendor.py` printing
`git -C T-Rex show origin/full-pipeline:scripts/midtrain.py | diff - origami/train_origami.py`
and asserting the changed hunks touch only the regions above (match on the marker comments
`# ORIGAMI-DELTA:`).

### 7.3 `origami/train_origami.sh` — the recipe

Everything from §1.4 unchanged, plus the single-GPU adaptation. Effective batch is held at
upstream's **128**.

```bash
ORIGIN_MODEL_PATH=<Qwen3-VL-2B-Instruct>
RESUME_CHECKPOINT=<miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6>
RESUME_SOURCE=midtrain              # keeps the tactile expert
DEFORM_ENCODER_PATH=<sharpa_wave_deform_encoder.pth>
LEROBOT_ROOT=<.../origami_eef62_train>
LEROBOT_VAL_ROOT=<.../origami_eef62_val>
IMAGE_SIZE="384 384"           # from the §4.5-A probe: token-matched square size

accelerate launch --config_file T-Rex/config/sft_qwen.yaml --num_processes 1 \
  origami/train_origami.py \
  --model_path        ${ORIGIN_MODEL_PATH} \
  --data_format lerobot --lerobot_root ${LEROBOT_ROOT} \
  --lerobot_val_root  ${LEROBOT_VAL_ROOT} \
  --action_dim 62 --action_chunk 16 \
  --image_size ${IMAGE_SIZE} \
  --use_robot_state 1 \
  --use_tactile_vec 1 --use_tactile_deform 1 --use_tactile_vqvae 1 \
  --deform_encoder_ckpt ${DEFORM_ENCODER_PATH} \
  --tactile_intermediate_size 1536 --training_stage 2 --tactile_loss_weight 1.0 \
  --cascaded_total_steps 10 --cascaded_split_step 6 \
  --cascaded_tactile_dropout 0.1 --cascaded_loss_weight 1.0 \
  --tactile_delay_offsets 0 4 8 12 --tactile_delay_scope f6 \
  --use_flare 1 --n_flare_tokens_per_frame 4 --n_flare_steps 8 \
  --flare_loss_weight 0.5 --flare_frame_stride 4 --flare_layer_index -1 \
  --resume_checkpoint ${RESUME_CHECKPOINT} --resume_source ${RESUME_SOURCE} \
  --learning_rate 1e-4 --min_lr_ratio 0 --weight_decay 0.01 --max_grad_norm 1.0 \
  --warmup_rates 0.03 \
  --train_bsz_per_gpu 2 --gradient_accumulation_steps 64 \
  --gradient_checkpointing 1 --optim adamw8bit --num_workers 8 \
  --n_epochs 1 --max_steps 40000 --save_steps 2000 --save_optimizer_state 1 \
  --val_freq 1000 --max_val_batches 30 \
  --seed 42
```

Deviations from `T-Rex/scripts/train.sh`, each with its reason:

| flag | upstream | here | why |
|---|---|---|---|
| `image_size` | `384 288` | square, **probed** (§4.5-A: `384 384` at factor 32, `336 336` at factor 28) | square source, no 1.33× stretch; token count matched to upstream so the frozen ViT operates near its trained point |
| `image_size_head` / `image_size_wrist` | — | ablation | reallocate tokens to the wrist views, where the crease signal actually lives (§4.5-B) |
| `use_robot_state` | `0` | **`1`** | fingers are largely self-occluded in the head view, and unlike the 65-D joint design the eef-9d+22 state layout makes T-Rex's `add_tracking_error_noise` (axis-angle on `state[3:9]`/`state[34:40]`) **valid**. This is the main representational win of eef-62. Keep `0` as the ablation arm. |
| `train_bsz_per_gpu` × GPUs | 16 × 8 | 2 × 1 | 40 GB budget |
| `gradient_accumulation_steps` | 1 | 64 | keeps effective batch = 128 |
| `gradient_checkpointing` | — | 1 | needed at 40 GB; requires §7.1 patch 1 |
| `optim` | adamw | adamw8bit | bf16 params make torch AdamW hold bf16 moments; blockwise 8-bit is smaller and better conditioned |
| `n_epochs`/`max_steps` | 100 / — | 1 / 40 000 | one epoch over 4.76 M frames at batch 128 is 37 k steps. Upstream's "100 epochs" is for a few-hundred-episode in-lab set. |
| `save_steps` | — | 2 000 | epoch-only checkpointing would never fire |
| `warmup_rates` | 0 (post-train) | **0.03** (midtrain value, §7.5-A) | 1 200-step warmup; the re-initialised `x_embedder`/`final_layer*` heads spike the loss otherwise. **Note:** with `action_dim=62` matching the midtrain checkpoint, `x_embedder`, `final_layer`, `final_layer_tactile` and `state_embedder` all load cleanly — the previous 65-D design dropped them. Confirm `missing == 0` in the resume log (gate G9b). |
| val split | `--val_ratio 0.05` (random samples) | held-out seasons | a random split leaks the session |

Escalation ladder if 2×64 still OOMs: `--freeze_latent_expert 1 --train_latent_last_n 8`
(≈45% less optimizer memory, ~25% faster step) with `--flare_loss_weight 0`
(the flare loss would only train `flare_proj` once the latent expert is frozen).

### 7.4 Data-loading throughput

`TRexLeRobotDataset` does **13 random-access video seeks per sample** (head with 9
`delta_timestamps` offsets, 2 wrists, 10 deform) — decode is the bottleneck, not the GPU.
Mitigations, in order of preference:
1. `--num_workers 8`, `prefetch_factor 4`, `persistent_workers` (already in §7.2's vendored copy).
2. `--flare_frame_stride 4` with `n_flare_steps 8` reaches +32 frames; LeRobot fetches these
   from the **same** head video file, so the seeks are near-local. Keep it.
3. If throughput is still < 4 samples/s/worker, add a `--sample_stride` option to the
   *sampler* (not the data): a `torch.utils.data.Sampler` emitting a strided,
   block-shuffled index so consecutive samples in a worker hit the same video GOP. Do **not**
   subsample at conversion time — the F6 window and the chunk both need 30 Hz neighbours.
4. Measure first: `origami/bench_latency.py --mode dataloader` reports samples/s and the
   per-key decode breakdown. Only then optimise.

### 7.5 Re-running midtrain on origami data (the vision-shift question)

**First, the correction that determines the whole answer: T-Rex never trains the vision
tower.** `visual*` parameters are set `requires_grad = False` in **all three** stages —
`pretrain.py:946`, `midtrain.py:1427`, `train.py:900` (post-train). The frozen ViT is
`Qwen3-VL-2B`'s internet-pretrained encoder and no T-Rex stage touches it. What T-Rex calls
the *latent expert* is the MoT's vision-language copy of the decoder layers, and that **is
already fully trainable in post-train**.

Consequence: **an origami midtrain stage unlocks no parameter that post-train does not
already train.** Its value is entirely in the *objective and schedule*, not in reaching new
weights. That said, the ingredients it adds are worth having, and one of them is a direct
match for our deployment protocol.

#### A. What `midtrain.sh` actually does differently

Everything structural is identical: `action_dim 62`, `action_chunk 16`,
`image_size 384 288`, `cascaded_total_steps 10 / split_step 6 / tactile_dropout 0.1 /
loss_weight 1.0`, flare `4 / 8 / 0.5 / stride 4 / layer −1`,
`tactile_intermediate_size 1536`, `training_stage 2`, `lr 1e-4`, `bsz 16 × 8`.
The differences are:

| | midtrain | post-train |
|---|---|---|
| **`--tactile_delay_offsets`** | **`0 4 8 12`** | absent (delay ≡ 0) |
| `weight_decay` | **0.01** | 0 |
| `warmup_rates` | **0.03** | 0 |
| `n_epochs` / `save_freq` | 10 / 2 | 100 / 50 |
| `val_ratio` | 0.02 | 0.05 |
| `--tactile_f6_stats_ckpt` | supplied | absent |
| data | merged multi-source root (`prepare_midtrain_merged.py`) | one task |

**Our run is midtrain-shaped, not post-train-shaped.** 4.76 M frames / 44 h is the same
order as T-Rex's own midtrain corpus, and upstream's "100 epochs over a few hundred
episodes" post-train framing does not transfer. Adopt midtrain's `weight_decay 0.01` and
`warmup_rates 0.03` (§7.3 already used 0.02 warmup for a different reason; use 0.03 for
upstream parity).

#### B. The one ingredient genuinely worth importing: the tactile delay curriculum

`midtrain.py:766-779` samples `delay_k ~ Uniform({0, 4, 8, 12})` per item and feeds the
tactile expert observations from frame `t + delay_k` while the action chunk stays anchored at
`t`. The docstring is explicit: *"simulates the inference pattern: slow chunk anchored at
t=0, tactile fast tick fired at t+k."*

**Those are literally our deploy offsets.** §9.1 fires fast ticks at in-chunk offsets
`{0, 4, 8, 12}` and `hardware_code/config/default.yaml` sets `refine_offsets [4, 8, 12]`.
Post-training with `delay_k ≡ 0` trains the tactile expert on a condition it will never see
at deploy — every fast tick after the chunk start reads tactile that is 4/8/12 frames ahead
of the anchor. This is a **train/deploy mismatch in the tactile channel**, and it is the
strongest argument for the midtrain path.

`midtrain.py` reaches the LeRobot loader (`:1455-1459` → `TRexLeRobotDataset`), but
`TRexLeRobotDataset.collate_fn` hard-wires `tactile_f6s_delayed = norm_tacf6` and
`tactile_deforms_delayed = deforms_tensor` (`lerobot_dataset.py:283-284`) — **the delay is not
implemented on the LeRobot path.** So it needs a loader subclass,
`origami/delayed_lerobot_dataset.py`:

* **F6 (cheap).** Extend `_f6_offsets` from `[-(W-1)/fps … 0]` to
  `[-(W-1)/fps … +max(delay)/fps]`, i.e. 16 + 12 = 28 offsets. `observation.tactile_f6` is a
  **numeric parquet feature, not a video**, so extra `delta_timestamps` on it cost a parquet
  read, effectively free. Then `tactile_f6s_delayed = window[:, W-1+k]` and
  `tactile_f6_history = window[:, k : k+W]` (the VQ-VAE window must also end at the delayed
  frame — `midtrain.py:798-800` does exactly this).
* **Deform (expensive).** Needs `delta_timestamps` `[0, k/fps]` on all 10 deform keys →
  **20 video seeks per sample instead of 10**, on top of §7.4's existing decode bottleneck.
* Therefore expose `--tactile_delay_scope {none, f6, both}`, default **`f6`**: take the
  delay on the F6/VQ-VAE channel (which is where the temporal signal lives and where the
  cost is zero) and keep the deform maps at the anchor frame. Measure `both` against `f6`
  on §8.1-D before paying 2× decode. Record the choice in `training_args.json`; gate **G17**
  asserts the deployed fast-tick offsets equal the trained `delay_offsets`.

#### C. Is there scope for a *vision-only* adaptation stage?

Yes, but be clear about what it can and cannot reach.

* **What it can do — FLARE-only warmup (Phase 0).** The FLARE loss is self-supervised: the
  latent expert's K query tokens predict the frozen ViT's features of head frames
  `+{1..8}·stride` ahead, cosine loss (`train.py:1119-1180`). **It needs no action labels.**
  Running a short stage with `--flare_loss_weight 1.0 --cascaded_loss_weight 0
  --action_loss_weight 0.1` lets the latent expert absorb origami's visual statistics
  (dark, wide-FOV, fisheye wrists, deforming paper) *before* the action loss starts pulling
  it. This is the closest thing to "adapting the vision path on origami data" that stays
  inside T-Rex's design. Requires the `--action_loss_weight` flag from §7.2 delta 3, since
  upstream hardcodes `loss = loss_act + w_tac·loss_tac + w_flare·loss_flare`. Cheap: no
  tactile expert, no Euler steps. **Recommended as an ablation, 1–2 k steps.**
* **What it cannot do — adapt the ViT.** The visual shift (fisheye wrists, 8.6× downsample,
  dark low-contrast scene) lands on frozen weights. No T-Rex stage fixes this, and neither
  would an origami midtrain. FLARE only teaches the latent expert to *predict* origami-ViT
  features; the features themselves do not move.
* **If you want to move them, that is a deliberate deviation.** Unfreezing the ViT (say the
  last N blocks at `0.1 × lr`) has three coupled consequences that must be handled together:
  1. **FLARE targets break.** They are produced by the same `self.visual` under
     `torch.no_grad()` (`train.py:1152-1155`). If the ViT trains, the target moves with the
     predictor and the cosine loss admits a trivial solution. Either disable FLARE
     (`--flare_loss_weight 0`, keeping the K tokens for sequence parity — §7.2 delta 6) or
     hold a frozen ViT copy for targets (+1.3 GB and a second forward).
  2. **Activation memory** for 3 images with grad, on top of gradient checkpointing on a
     40 GB card. Likely forces `train_bsz_per_gpu 1`.
  3. **Forgetting risk** on a frozen-elsewhere pipeline whose downstream MoT was tuned
     against these exact features.
  Verdict: **park it.** Run it only if §11.9's zero-shot + `diagnose_shift.py` ViT-feature
  statistics show the origami features are genuinely degenerate (e.g. collapsed variance or
  large cosine distance from the T-Rex feature mean). Otherwise the 44 h of in-domain MoT
  training is the better use of the same compute.
* **Rejected: adapters between the ViT and the MoT.** Would change the graph and break
  checkpoint compatibility with the released midtrain weights — contrary to D2's "as close
  as possible".

#### D. Co-training with T-Rex's own dataset

`prepare_midtrain_merged.py` exists because midtrain co-trains over a merged multi-source
root, and T-Rex's public ~50 h corpus (`zekaiwang/trex_dataset`, LeRobot v3.0, same Sharpa
Wave hands and tactile) could be converted to eef-62 with *their* `dexmate-urdf` FK and
merged with ours. Honest assessment: this is an **anti-forgetting regularizer, not a
shift mitigation** — mixing in T-Rex data teaches the model nothing about origami's cameras.
It would roughly double prep cost and require a second kinematics model. **Defer**; revisit
only if step 12 shows the pretrained features being washed out (val curves improving on
origami while §8.1-D's tactile contribution collapses).

#### E. Recommended staging

| phase | steps | config | purpose |
|---|---|---|---|
| **0** (ablation) | 1–2 k | FLARE-only: `--flare_loss_weight 1.0 --cascaded_loss_weight 0 --action_loss_weight 0.1 --freeze_latent_expert 0` | latent expert absorbs origami visual statistics |
| **A** | 2 k | `--freeze_latent_expert 1 --train_latent_last_n 0 --flare_loss_weight 0` | action/tactile heads settle onto origami action statistics (§11.9-3) |
| **B** | 40 k | full §7.3 recipe with midtrain hyperparameters + `--tactile_delay_offsets 0 4 8 12` | the real run |

Phases chain through `--resume_checkpoint` + `--resume_full_state`. Phase 0 and A are each
~5% of the phase-B budget, so both are cheap to try and cheap to skip. **Only phase B is
required**; 0 and A are ablations to be justified by §8.1 metrics against the §11.9
zero-shot floor.

---

## 8. Evaluation

### 8.1 `origami/eval_offline.py`

Loads a checkpoint through **exactly** `test.py::model_load`'s logic (import it; do not
reimplement) and a `TRexLeRobotDataset` on the val root, then reports the metrics below.

`--zero-shot` points it at the **untrained midtrain checkpoint** and skips nothing else: at
`action_dim=62` every head loads, so this measures the pretrain→origami shift end-to-end and
establishes the floor every trained checkpoint must beat (§11.9 mitigation 1). Always report
it alongside two trivial baselines: **hold-position** (repeat `observation/state`, which is
what the SDK's placeholder `TeamPolicy` does) and **`--disable_tactile`**.

**A. Action-space accuracy** (normalized and raw), per chunk step `k = 0..15`, per block:
`L_trans(3) / L_rot6d(6) / L_hand(22) / R_trans(3) / R_rot6d(6) / R_hand(22)`.
Metrics: MSE, MAE, and **normalized variance share** (`Var(pred_block)/Var(gt_block)`) —
this last one is the §11.1 diagnostic.

**B. EEF-space error.** Reconstruct the target SE3 from the predicted delta9 and compare to
the ground-truth target SE3: translation error in **mm**, rotation error in **degrees**
(`Rodrigues` angle of `R_pred^T R_gt`). Report per k. Reference scale: the ground-truth
per-frame EEF step is 0.39 mm mean / 2.04 mm p99, so a 1 mm error at k=15 is already large.

**C. Joint-space error — the metric that matters.** Run the full deploy retarget
(`origami/retarget.py`, §9.2) on the predicted chunk and compare the resulting 65-D absolute
joint vector to the dataset's `action65`. Report per-group RMS in rad and the fraction of
steps where IK failed or clipped. **This is the competition's actual objective.**

**D. Tactile-expert contribution.** Run three configurations on the same batches:
1. cascaded (slow at τ_split + tactile continue) — the deployed path;
2. `--disable_tactile 1` → `forward_flow_action_full` on the action expert alone;
3. cascaded with the tactile inputs zeroed (the dropout condition).
Report the metric deltas, split by contact / no-contact using the F6 flag
(`|F| > 1 N` on any finger, cf. `origami_dataset.py::_contact_flag` in the old tree).

**E. Rollout drift.** For 10 held-out episodes, two modes:
* *anchored*: at each chunk start take the anchor pose from the ground-truth state (what
  happens on a well-tracking robot);
* *closed-loop*: integrate the IK output forward and re-anchor from the integrated joints.
Plot EEF trajectory vs ground truth and report terminal drift.

**F. Smoothness.** Per-step joint velocity and acceleration of the retargeted commands vs the
demonstrations; flag any step exceeding the URDF velocity limit (2.6179 rad/s) or the
deploy rate limit (§9.3).

Outputs: `metrics.json`, per-block CSVs, and PNGs (per-k error curves, parity plots, episode
traces, contact-conditioned bars). No new plotting framework — matplotlib, one figure per file.

### 8.2 `origami/eval_shadow.py`

Drives `origami-inference-kit-participant/sharpa_north_ces_lite_sdk-main`'s
`participant_local_evaluator` in Shadow mode against a locally running `serve_zenoh.py`,
using the SDK's bundled `season_POC22061_2026_07_09_16_23_46_train`. Asserts:
metadata validation passes, every reply is finite `float32[T,65]`, and the URDF
limit/jump/velocity checks report no violations. Also run the SDK's
`examples/check_zenoh_policy.py` and `tests/test_check_zenoh_policy.py`.

### 8.3 `origami/bench_latency.py`

Modes: `slow`, `fast`, `slow_and_fast`, `dataloader`. Reports p50/p95/p99 latency and prints
the **maximum feasible `action_horizon`** given `slow_p99` at 30 Hz execution
(`T_min = ceil(slow_p99_s × 30)` for sync mode; async mode tolerates
`slow_p99 < T/30 + one chunk`). Must be run before choosing `--action_horizon` (§9.1).

### 8.4 Notebook

Rewrite `trex_colab.ipynb` as `origami/colab.ipynb`: install cell (uv, pinned deps, lerobot,
pin/pink/daqp), asset/checkpoint download, a pre-flight cell running gates G1–G3 and G7,
`SMOKE=1 train_origami.sh`, the real run with Drive-backed `--output_dir`, then
`eval_offline.py`. No notebook-only logic — every cell shells out to `origami/*` scripts.

---

## 9. Deploy

### 9.1 Cadence design

The wire gives one `infer` per call returning a fixed `[T, 65]`. To reproduce T-Rex's
slow-1.875 Hz / fast-7.5 Hz protocol:

* **`action_horizon T = 4`.** The gateway consumes 4 steps (0.133 s at 30 Hz) then calls again.
* Keep a call counter. `n % 4 == 0` → **slow + fast** (re-encode vision, cache KV at τ_split,
  then one tactile continue). Otherwise → **fast only** (reuse the cached KV with fresh tactile).
* The 16-step chunk is therefore refreshed at in-chunk offsets `{0, 4, 8, 12}` — **identical to
  `eval_trex_async.py`'s `refine_offsets [4, 8, 12]` plus the chunk-start fast tick.**
* Return rows `[4·(n mod 4) : 4·(n mod 4) + 4]` of the (aggregated) chunk.
* Chunk-start anchor: the arm joints from `observation/state` **at the slow tick**, held for
  all 16 rows. Matches `eval_trex_async.py:992-1005`.
* ACT temporal aggregation: append each refreshed chunk to a buffer and use
  `aggregate_chunks(buffer, global_step, k=temporal_agg_k)` — **import it from
  `T-Rex/hardware_code/eval/eval_trex_async.py:75`**. Default `temporal_agg_k = 0.0`
  (uniform average), matching `hardware_code/config/default.yaml`.

`T = 4` is a **hypothesis contingent on measured latency** (§8.3). If `slow_p99 > 130 ms` in
sync mode, fall back to `T = 8` (slow every 2 calls, refine offsets `{0, 8}`) or `T = 16`
(slow every call, no fast ticks — degrades to the tactile-blind cadence).
`--action_horizon` and `--slow_every` are CLI flags; `slow_every × T == 16` is asserted.

### 9.2 `origami/retarget.py`

```python
class Retargeter:
    def __init__(self, kin: OrigamiKinematics, cfg: RetargetConfig)
    def set_anchor(self, state65: np.ndarray) -> None
        # q_l = state65[0:7], q_r = state65[29:36]
        # self.base_l, self.base_r = kin.fk_matrices(q_l, q_r)
        # self.warm_l, self.warm_r = q_l.copy(), q_r.copy()
        # self.prev_cmd = state65.copy()
    def step(self, action62: np.ndarray, motor7: np.ndarray) -> np.ndarray   # -> (65,)
```
`step` (mirrors `eval_trex_async.py:1113-1145`):
```python
dpos_l, drot_l, hand_l = action62[0:3],  action62[3:9],  action62[9:31]
dpos_r, drot_r, hand_r = action62[31:34], action62[34:40], action62[40:62]
t_l = self.base_l[:3,3] + self.base_l[:3,:3] @ dpos_l
R_l = self.base_l[:3,:3] @ rot6d_to_matrix(drot_l)      # Gram-Schmidt, eval_trex_async.py:283
# same for right
q_l, q_r = self.kin.solve_ik(pin.SE3(R_l, t_l), pin.SE3(R_r, t_r), self.warm_l, self.warm_r)
self.warm_l, self.warm_r = q_l, q_r                      # warm-start the next row
cmd = np.concatenate([q_l, hand_l, q_r, hand_r, motor7]).astype(np.float32)
return self._safety(cmd)
```
`rot6d_to_matrix` is imported from `eval_trex_async.py:283` (Gram–Schmidt orthonormalisation),
**not** `lerobot_common.py::get_rot_mat` (which is a raw `column_stack` + cross product and
does not renormalise). The training targets are exact rotation columns so the two agree on
clean data, but the policy's output is only approximately orthonormal — use Gram–Schmidt.

### 9.3 `_safety` — mandatory, in this order

1. **Finite check.** Any NaN/Inf → return `self.prev_cmd` and increment a counter.
2. **Joint limits.** Clip arms to the URDF `[lower, upper]` (§1.3); clip hands to the URDF
   hand limits; clip the motor block to its URDF limits.
3. **Rate limit.** `|cmd - prev_cmd| ≤ max_step` with
   `max_step = max_joint_vel / 30` and `max_joint_vel = 0.3 rad/s` default
   (matches `arm_hand_control.py`'s `max_edge_joint_step = 0.3 / command_hz`). Clip
   element-wise toward `prev_cmd`.
4. **Motor block held.** `cmd[58:65] = motor7` where `motor7 = observation/state[58:65]`
   of the current call — never a stale value, never a predicted value (D5).
5. **Optional collision gate** (`--check-collisions`): if `kin.check_collision(q_l, q_r)`,
   fall back to `prev_cmd`. Off by default (adds ~1 ms/row).
6. `self.prev_cmd = cmd`; return `float32`.

Log a per-episode summary of clip/limit/NaN/IK-failure counts; a nonzero NaN or
IK-failure count in Shadow is a **blocking** finding.

### 9.4 `origami/serve_zenoh.py`

Start from `sharpa_north_ces_lite_sdk-main/examples/policy_server_template.py` and keep
`OrigamiZenohServer` and the msgpack codec **byte-identical** — only `TeamPolicy` changes.

```python
class TeamPolicy:
    def __init__(self, action_horizon, checkpoint_path, urdf_path, **cfg):
        # 1. test.py::model_load(args) -> model, processor, statistic (q01/q99 + masks)
        # 2. training_args.json -> instruction (§5.7), image_size,
        #    cascaded_{total,split}_step, vqvae_window,
        #    locked_config, action_dim (assert 62), action_chunk (assert 16)
        # 3. OrigamiKinematics(urdf_path, LockedConfig.from_dict(ta["locked_config"]))
        #    -- asserts digest match (gate G1c)
        # 4. CascadedServer(args, model, processor, statistic)  [imported from test.py:330]
        # 5. Retargeter(kin, cfg)
        # 6. warm-up: one slow_and_fast on zeros; assert output shape (16, 62)

    def reset(self):
        # clear CascadedServer.cached_kv/x_split/f6_buffer, chunk buffer, call counter,
        # Retargeter anchor + warm start + prev_cmd

    def infer(self, obs) -> np.ndarray:      # float32 (T, 65)
```
`infer` body:
```python
state65 = np.asarray(obs["observation/state"], np.float32)
f6      = np.asarray(obs["observation/tactile"], np.float32).reshape(10, 6)
deform  = split_deform_strip(obs["observation/image/tactile_deform"][:, :, 0])   # (10,240,240)
deform  = deform.astype(np.float32) / 255.0
self.f6_hist.append(f6)                       # deque(maxlen=vqvae_window), left-pad on reset

slot = self.calls % self.slow_every
if slot == 0:
    head = pil(obs["observation/image/head_left"]).resize(self.image_size, LANCZOS)
    wr   = pil(obs["observation/image/wrist_right"]).resize(self.image_size, LANCZOS)
    wl   = pil(obs["observation/image/wrist_left"]).resize(self.image_size, LANCZOS)
    payload = {"mode": "slow_and_fast", "task_description": self.instruction,   # §5.7
               "image_head": jpeg(head), "image_wrist_right": jpeg(wr),
               "image_wrist_left": jpeg(wl),
               "tactile_f6": np.stack(self.f6_hist),      # [W,10,6] dense window
               "tactile_deform": deform,
               "state_fast": self.norm_state(state65)}    # only if use_robot_state
    chunk = self.server.predict("slow_and_fast", payload)["actions"]   # (16,62) denormalized
    self.chunk_buf = [(0, np.asarray(chunk))]
    self.retarget.set_anchor(state65)
else:
    payload = {"mode": "fast", "tactile_f6": np.stack(self.f6_hist),
               "tactile_deform": deform}
    chunk = self.server.predict("fast", payload)["actions"]
    self.chunk_buf.append((0, np.asarray(chunk)))

rows = []
for j in range(self.T):
    step = slot * self.T + j
    a62 = aggregate_chunks(self.chunk_buf, step, k=self.temporal_agg_k)
    rows.append(self.retarget.step(a62, state65[58:65]))
self.calls += 1
return np.ascontiguousarray(np.stack(rows), dtype=np.float32)
```
Notes:
* **The images must go through the same 224→`image_size` LANCZOS resize as training**
  (§4.3). Do not resize from any other source resolution.
* `CascadedServer.predict` decodes `image_head` etc. from JPEG bytes. Pass the encoded bytes
  (as `eval_trex_async.py` does) so the JPEG round-trip matches the reference client, **or**
  refactor `predict` to accept PIL objects. Prefer the latter (`origami/policy.py` wraps
  `CascadedServer` and calls `_run_slow`/`_run_fast` directly with PILs) — it removes a
  ~5 ms JPEG encode+decode per call and one lossy step. Keep the ZMQ path only for parity tests.
* `state_fast` must be **normalized** by the same q01/q99 the model was trained with —
  `test.py::_run_slow` does the normalization itself given `statistic["state_*"]`, so pass
  the **raw** 62-D state and let it normalize. Build the 62-D state with
  `[FK9(state65[0:7]), state65[7:29], FK9(state65[29:36]), state65[36:58]]` using the same
  `OrigamiKinematics`.
* `--disable_tactile 1` is available for the ablation and as a fallback if the tactile stream
  is zero-filled by the organizer.

### 9.5 `origami/docker/Dockerfile`

Multi-stage, CUDA runtime base. Must contain: torch 2.6.0 cu124, transformers 4.57.3,
`pin`/`pink`/`qpsolvers[daqp]`, `eclipse-zenoh==1.9.*`, `msgpack`, the checkpoint
(`model.pt`, `processor/`, `config.json`, `training_args.json`, `stats_data.json`), the URDF +
meshes, and `origami/`. Entrypoint `python -m origami.serve_zenoh` with no required args.
`ENV EXECUTION_MODE=async`. No secrets in `ENV`, no dataset, no `.git`.

---

## 10. Verification gates

Every gate is a test in `origami/tests/` or a `verify.py` subcommand, with the stated
threshold. **Do not proceed past a failing gate.**

| id | gate | threshold |
|---|---|---|
| **G0** | Upstream drift: sha256 of the 8 T-Rex files we import/vendor matches the pinned commit | exact |
| **G1a** | URDF joint set == the 65 mapped names, in dataset-index order | exact |
| **G1b** | Reduced model `nq == nv == 14`; joint names == `{left,right}_arm_joint_1..7` | exact |
| **G1c** | `LockedConfig.digest()` in the dataset meta == the checkpoint's == the server's | exact |
| **G2** | **FK/IK round-trip**: for 10 000 random dataset frames, `solve_ik(fk(q), warm=q+N(0,0.02))` recovers `q` | max abs Δq < 1e-3 rad **and** EEF pos err < 0.1 mm, rot err < 0.01° |
| **G2b** | IK from a perturbed warm start (`+N(0, 0.1)`) | ≥ 99.5% of frames meet G2; log the failures' configs |
| **G3** | `delta9` → `rot6d_to_matrix` → reconstruct target pose | max err < 1e-9 (exact math) |
| **G3b** | `matrix_to_rot6d`/`rot6d_to_matrix` round-trip on 10 000 random SO(3) | < 1e-12 |
| **G4** | **Chunk parity**: converter output vs a direct call to `lerobot_common.build_action_chunk` for one full season | bitwise identical for `action`, `state`, `action_abs`, `tactile_f6` |
| **G4b** | `--action-layout abs` derived chunk vs baked chunk | max abs diff < 1e-6 |
| **G5** | **Deform tile mapping**: over 2 000 frames where exactly one finger has \|F\| > 5 N, that finger's tile has the max mean intensity | ≥ 95% |
| **G6** | **Deform video round-trip**: decoded channel-0 vs source luma | exact (lossless codec) or PSNR > 45 dB |
| **G7** | **Loader parity**: `TRexLeRobotDataset` on the converted root; `collate_fn` returns all 21 keys with the §1.4 shapes and dtypes; `tactile_deforms` is `[B,10,1,240,240]`; `tactile_f6_history` is `[B,16,10,6]`; `norm_actions ∈ [-1,1]` on masked dims | exact shapes; ≥ 99.9% of masked-dim values in range |
| **G7b** | A forward pass of `Qwen3VLVLAModel` on that batch (CPU, tiny config) produces finite `loss_act` and `loss_tac` | finite |
| **G8** | Reservoir q01/q99 vs exact q01/q99 on one season | max relative error < 1% per dim (on non-degenerate dims) |
| **G8b** | Tactile mask is **not** all-`True`; the masked-out set includes L-ring, L-pinky, R-ring, R-pinky | assert |
| **G9a** | Smoke train: 5 steps, bsz 1, checkpoint written, `--resume_full_state` resumes at the same step | finite losses |
| **G9b** | Resume log from the midtrain checkpoint | `len(missing) == 0` **and** `len(skipped_shape_mismatch) == 0` — with `action_dim=62` every head must load |
| **G10** | `serve_zenoh` passes the SDK's `check_zenoh_policy.py` + metadata validation | pass |
| **G11** | Shadow replay on the SDK's bundled season: no NaN, no IK failure, no URDF limit/jump/velocity violation | zero violations |
| **G12** | Prep↔deploy image parity: the 62-D state and 384×384 tensors produced from a dataset frame equal those produced by `serve_zenoh` from the same frame downscaled to 224 | state max diff < 1e-5; pixels exact |
| **G13** | Latency: measured `slow_p99` admits the chosen `action_horizon` (§8.3) | `slow_p99 < T/30 s` (sync) |
| **G17** | Delay-curriculum parity: `training_args.json["tactile_delay_offsets"]` == the in-chunk fast-tick offsets `serve_zenoh` actually fires (§9.1), and the delayed-F6 tensor at `delay_k` equals `f6_window[:, W-1+k]` | exact |
| **G15** | Frozen VQ-VAE codebook health on ≥ 200 k origami F6 windows (§11.8) | ≥ 25% of the codebook used, normalized entropy ≥ 0.5, clamp-saturation ≤ 5% |
| **G16** | Token budget: probed `patch_size × merge_size`, the chosen `--image_size*`, and the resolved per-view grid are recorded in `training_args.json` and reproduced byte-identically by `serve_zenoh` | token count within 1 grid cell of upstream's `384 288`; grids equal |
| **G14** | Instruction identity: `meta/origami_prep.json["instruction"]` == `training_args.json["instruction"]` == the string `serve_zenoh` feeds `task_description`, and the tokenized `input_ids` from a dataset sample equal those from the deploy path for the same frame | exact string match; `input_ids` bitwise equal |

Tests must be **CPU-only and network-free**, using the local fixture season
`season_POC22061_2026_07_23_10_20_33_train` and a tiny stub Qwen config for G7b.

---

## 11. Risks, ranked, with mitigations

Ranked by expected impact: **§11.1 (action SNR) > §11.8 (VQ-VAE collapse) > §11.9 (pretrain
shift) > §11.2 (latency) > the rest.** §11.8 and §11.9 are numbered last only because they
were added after the first draft — do not read the numbering as the ranking.

### 11.1 The arm barely moves — eef-62 may be a worse parameterization than 65-D joints

Measured: the EEF translates **0.39 mm per frame on average, 2.04 mm at p99**. Over a
16-frame chunk that is ~6 mm. The 22-D finger blocks, by contrast, sweep 0–1.5 rad. So
9 of the 62 dims per arm carry very little variance, and after per-dim q01/q99 normalization
they will be *amplified* — including their noise, which comes from FK on a 30 Hz encoder
signal.

Mitigations, all in the plan:
* Per-(step, dim) q01/q99 normalization is already T-Rex's scheme (`[16,62]` stats), so each
  chunk step gets its own scale — a k=1 delta and a k=15 delta are not normalized together.
* §8.1-A reports the **normalized variance share** per block. If the arm blocks are
  dominated by FK quantization noise, that shows up as `Var(pred)/Var(gt) ≪ 1` at small k.
* Keep the 65-D joint-space pipeline as an explicit comparison arm (it is what the previous
  attempt built). §8.1-C's joint-space RMS is directly comparable between the two.
* If the arm blocks are noise-dominated, the fix inside eef-62 is a longer effective horizon:
  raise `FRAME_STRIDE` to 2 or 3 at conversion time (chunk then spans 32–48 source frames,
  1.1–1.6 s) so the deltas are 3× larger relative to the noise floor. `FRAME_STRIDE` is
  already a `lerobot_common` constant and `build_action_chunk` honours it; expose it as a
  conversion flag and record it in `meta/origami_prep.json`. **Deploy must then execute one
  chunk row every `FRAME_STRIDE` robot steps** — this changes §9.1's cadence arithmetic, so
  gate the flag behind an explicit `--frame-stride` and assert consistency at load.

**This is the single biggest open question about the whole redesign. Run §8.1-A on a
2000-step pilot checkpoint before committing to the full 40 k-step run.**

### 11.2 Deploy latency

A 2 B Qwen3-VL + ViT slow tick, 6 Euler steps on the action expert, then 4 on the tactile
expert. `T = 4` gives a 133 ms budget in sync mode. Mitigations: `execution_mode=async`
(default), bf16 + SDPA, `--action_horizon` escalation ladder (§9.1), and a hard requirement
to run §8.3 before submission (gate G13).

### 11.3 `tactile_raw` cannot feed T-Rex's deform head

`deform_proj` is `ActionEmbedder(28800)` = 128·15·15, i.e. **240×240 only**. `tactile_raw`
cells are 240×**320** → 128·15·20 = 38 400. Options if raw is wanted later:
(a) centre-crop each 240×320 cell to 240×240 (loses 25% of the width);
(b) resize 240×320 → 240×240 (anisotropic);
(c) add a second `raw_proj` head — a new module, so the midtrain checkpoint would not cover it.
Plan: ship `--include-tactile-raw` in the converter (stores 10 extra `(3,240,320)` videos
under `observation.tactile_raw.{l,r}{0..4}`, **not** in `build_trex_features` and **not**
read by the loader), default **off** because it triples the download (3.0 GB/season) and
nothing consumes it. Revisit only after §11.1 is settled.

### 11.4 LeRobot API drift

`LeRobotDataset.create/add_frame/save_episode/finalize` signatures, whether non-`observation.images.*`
keys are accepted as `dtype: "video"` (`DEFORM_KEYS` are `observation.tactile_deform.*`), and
whether `aggregate_datasets` exists all vary by version. **Implementation step 1 is
`origami/tests/test_lerobot_probe.py`**: create a 3-frame dataset with the full
`build_trex_features` schema, write it, reopen it with `delta_timestamps`, and assert every
key round-trips. Pin the exact lerobot commit in `pyproject.toml` and record it in
`meta/origami_prep.json`. If the `observation.images.` prefix turns out to be required,
report it before changing key names — renaming breaks `DEFORM_KEYS` parity and would force a
loader override.

### 11.5 Storage / bandwidth

43 GB output + 50–160 GB streamed source, against 23 GB local free. `prepare.py` must
`statvfs`-check and refuse to start (§5.6). Recommend an external disk for `--out-root` and
`--cache-root`, or run prep on the same machine that will train.

### 11.6 Hand state↔action transient

`state[7] = 0.100` vs `action[7] = 1.139` rad at episode start (hand init). Only affects
`tracking_error` (used by `add_tracking_error_noise` when `use_robot_state=1`). Mitigated by
skipping the first 30 frames per episode for the tracking-error stats and reporting both
variants (§5.4). If `te_std` on the hand block is still > 0.1 rad, run with
`--use_robot_state 0` for the first pilot.

### 11.7 Gradient checkpointing × cascaded KV cache

The non-idempotent `cache.update()` under checkpoint recompute (§7.1 patch 1). Covered by a
unit test that runs the cascaded tactile step twice with `torch.is_grad_enabled()` true and
asserts the KV length is unchanged.

### 11.8 Frozen tactile VQ-VAE codebook collapse — **high priority, cheap to check and fix**

`Qwen3VLVLAModel.encode_tactile_f6_history` (`modeling_vla.py:354-361`) min-max normalizes the
raw F6 window with **T-Rex's own** `tacf6_vqvae_{min,max,mask}` buffers and **hard-clamps to
[−1, 1]**:
```python
denom  = (self.tacf6_vqvae_max - self.tacf6_vqvae_min) + 1e-8
normed = torch.clamp(2.0 * (flat - self.tacf6_vqvae_min) / denom - 1.0, -1.0, 1.0)
normed = torch.where(self.tacf6_vqvae_mask, normed, flat)
```
Origami's force distribution is very unlike a pick-and-place corpus: **p99 |F| reaches ≈ 42 N
on the right index and ≈ 32 N on the right thumb, while 4 of 10 fingers are effectively dead**
(mean |F| 0.002–0.065 N; §1.2). Two failure modes follow:
* if T-Rex's range is *narrower*, the origami window saturates at ±1 → the encoder sees a
  square wave → few distinct codes;
* if it is *wider*, origami forces occupy a sliver near −1 → the encoder sees near-constant
  input → **one** code.

Either way the tactile expert receives a near-constant token, the cascaded mechanism silently
degrades to the action-expert baseline, and **nothing raises an error**. This is the most
likely way for this project to "work" while the entire tactile contribution is zero.

**Upstream knows about this failure mode.** `midtrain.py:322-325` prints:
*"WARN: no tactile_f6 stats in manifests AND no VQ-VAE ckpt provided — using default [-1, 1]
which WILL saturate real F6 values."* — and exposes `--tactile_f6_stats_ckpt` specifically to
avoid it.

**Two separate normalizations must not be confused:**
* the **per-frame F6 vector** fed to `tacf6_embedder` is normalized by the loader from
  `meta/trex_norm_stats.json` → **origami q01/q99, correct by construction** (§5.4);
* the **VQ-VAE history window** is normalized *inside the model* by
  `tacf6_vqvae_{min,max,mask}`, which come from the **T-Rex checkpoint's** `blob["stats"]`
  (`modeling_vla.py:323-325`) → **wrong scale, and this is the live risk.**
Fixing the first does not fix the second.

Diagnosis (gate **G15**): run `encode_tactile_f6_history` over ≥ 200 k origami windows and
report, per hand (or per finger, for a `granularity="finger"` checkpoint), the **codebook
usage histogram, the number of codes used, and the normalized entropy**
`H / log(codebook_size)`. Also report the fraction of normalized values that hit the clamp.
Thresholds: **≥ 25% of the codebook used and normalized entropy ≥ 0.5**, clamp-saturation
fraction ≤ 5%.

Fix ladder, cheapest first:
1. **Re-fit the normalization buffers only.** `tacf6_vqvae_min/max/mask` are plain registered
   buffers (`modeling_vla.py:139-144`). Overwrite them with origami's own q01/q99 and the
   degenerate-dim mask from `meta/trex_norm_stats.json` (§5.4) — the encoder weights and
   codebook are untouched, so this costs nothing and often suffices. Add
   `origami/refit_vqvae_stats.py` writing a new checkpoint; record
   `vqvae_stats_source: "origami"` in `training_args.json`.
2. **Retrain the VQ-VAE on origami F6.** T-Rex ships `tactile_vqvae/train.py` +
   `config/vqvae_f6.yaml` for exactly this (window 16, `in_channels 30` = 5 fingers × 6).
   Feed it the converted `observation.tactile_f6` windows, then
   `utils/merge_vqvae_into_ckpt.py --vla_ckpt … --vqvae_ckpt … --output …` bakes it in and
   sets `use_tactile_vqvae=1` + `vqvae_config` so both `train.py` and `test.py` auto-detect
   it. Consider `granularity: finger` so the 6 live fingers get their own codes instead of
   being averaged with 4 dead ones.
3. If the codebook still collapses, drop to `--use_tactile_vqvae 0 --use_tactile_vec 1
   --use_tactile_deform 1` (raw F6 vector + deform maps, no temporal codes) and treat the
   temporal-tactile channel as future work. Measure against §8.1-D.

**Note that the deform encoder is not at risk**: `sharpa_wave_deform_encoder.pth` was trained
self-supervised on Sharpa Wave deform maps and the origami rig uses the *same* sensor with the
*same* 240×240 cells. It transfers as-is.

### 11.9 Pretrain → origami distribution shift

Two distinct shifts that need **opposite** treatments — do not conflate them.

**(a) train ↔ deploy: ≈ zero.** Same rig, same cameras, same organizer squash, same Sharpa
Wave hands and tactile sensors. There is no sim2real or cross-camera gap. **Therefore domain
randomization is the wrong tool here** — heavy augmentation would only blur the single
distribution we know exactly, and §10's G12/G14 gates exist to keep the two paths identical.
The genuine residual is *cross-season* variation (lighting, time of day, paper batch) plus
the organizer's eval-day rig, which the 101/25 held-out-**season** split measures directly.
Justified augmentation is therefore **light photometric only** — brightness/contrast/gamma
jitter at ±10%, no geometric transforms, no crops, no colour shifts. Gate it behind
`--photometric_aug 0.0` (default off) and adopt only if held-out-season val improves.

**(b) T-Rex midtrain → origami: large.** Measured axes:

| axis | T-Rex midtrain | origami |
|---|---|---|
| head camera | rectilinear, tight 400×300 workspace crop | wide FOV squashed to square, ~1/8.6 scale |
| wrist cameras | rectilinear 640×360 | **fisheye**, circular vignette, 480×480 |
| photometry | varied objects/lighting | dark, low-contrast, near-monochrome gray-on-black |
| arm motion | coarse pick / transfer primitives | **0.39 mm/frame** mean, fingers do the work |
| horizon | short primitives | 135–172 s, 4 038–5 154 frames, 6 sequential folds |
| language | 22 distinct primitive instructions | **one constant string** (§5.7) |
| embodiment | Dexmate Vega-1 arms | Sharpa North POC2.2 arms (both 7-DOF, different geometry) |
| hands + tactile | Sharpa Wave | **Sharpa Wave — identical** |
| corpus | ~50 h, 5 400 trajectories | **44 h, 682 episodes, 4.76 M frames** |

What this does and does not imply:
* The **action-scale** part of the shift (row 4) is largely neutralized by q01/q99
  normalization: chunk deltas are mapped to [−1, 1] using **origami's own** quantiles, so the
  action head sees the same numerical range it was trained on. §11.1's concern is about
  **SNR inside that range**, not scale mismatch with T-Rex.
* The **visual** part (rows 1–3) lands on a **frozen** ViT (`train.py` sets
  `requires_grad=False` for every `visual*` parameter), so it cannot be fine-tuned away —
  all adaptation must happen in the MoT layers. That is an argument for **not** freezing the
  latent expert.
* The **strongest available mitigation is that we have 44 h of in-domain data**, comparable to
  T-Rex's own midtrain corpus. This is domain-specific training warm-started from a related
  checkpoint, not a few-shot fine-tune. Upstream's "100 epochs on a few hundred episodes"
  framing does not transfer; §7.3's 1 epoch / 40 k steps is the right shape.

Ordered mitigations, all cheap relative to the 40 k-step run:

1. **Measure the shift end-to-end first: zero-shot evaluation.** At `action_dim=62` every head
   in the midtrain checkpoint loads (gate G9b), so the checkpoint can be evaluated on origami
   val **before any training**. `eval_offline.py --zero-shot` reports the §8.1-A/B/C metrics.
   This is one number that quantifies the whole shift and sets the floor any training must
   beat. **Run it before step 11.** If zero-shot joint-space RMS is no better than the
   `--disable_tactile` or hold-position baselines, the transfer value is near zero and the
   pretrained init is worth less than the plan assumes.
2. **`origami/diagnose_shift.py`** — side-by-side histograms/statistics, origami vs a sample of
   T-Rex's public dataset (`zekaiwang/trex_dataset`, LeRobot v3.0):
   chunk-delta magnitude per block, F6 magnitude per finger, **VQ-VAE code usage entropy**
   (§11.8 G15), deform-map occupancy fraction, and frozen-ViT feature statistics (per-channel
   mean/std of the pooled patch tokens, plus the cosine distance between the two datasets'
   mean feature). Output one markdown report. Run before step 11.
3. **Staged unfreeze.** Phase A: 2 000 steps with `--freeze_latent_expert 1
   --train_latent_last_n 0 --flare_loss_weight 0` — lets `x_embedder` / `final_layer*` /
   `state_embedder` and the tactile expert settle onto origami's action statistics before the
   backbone moves. Phase B: resume with everything trainable and the full recipe. Cheaper than
   risking 44 h of in-domain gradient washing out the pretrained features in the first few
   hundred steps, and it reuses `--resume_full_state`.
4. **Discriminative LR** (ablation): a third param group at `0.1 × lr` for
   `model.layers.*` (the transferred backbone) and `1.0 × lr` for the heads and the tactile
   expert. Upstream uses a single LR; this is a documented deviation, off by default.
5. **Do not re-initialize the loading heads.** With `action_dim=62` they load cleanly, and
   discarding them throws away the only action prior we have. G9b asserts `missing == 0`;
   if it ever reports skipped keys, that is a bug in the recipe, not something to accept.

---

## 12. Implementation order

Each step ends at a named gate. Do not start step *n+1* before step *n*'s gate passes.

| step | deliverable | gate |
|---|---|---|
| 1 | Repo hygiene: T-Rex submodule pinned, deletions, `.gitignore`, `pyproject.toml` (uv), venv with `pin`, `pink`, `qpsolvers[daqp]`, `av`, `lerobot@<pin>`, `huggingface_hub`, `pandas` | G0 |
| 2 | `origami/tests/test_lerobot_probe.py` (LeRobot write/read probe) + `test_processor_probe.py` (§4.5-A token factor) | §11.4 resolved, G16 size chosen |
| 3 | `origami/constants.py` + `origami/kinematics.py` | G1a, G1b, G2, G2b, G3, G3b |
| 4 | `origami/decode.py` | G5, G6 (on the fixture) |
| 5 | `origami/stats.py` | G8 (vs the exact accumulator on the fixture) |
| 6 | `origami/convert.py` — single-season shard | G4, G4b, G8b |
| 7 | `origami/fetch.py` + `origami/merge.py` + `origami/prepare.py` | merged 2-season root opens in `TRexLeRobotDataset` |
| 8 | `origami/verify.py` — all gates as CLI subcommands | G7, G7b, G12 |
| 9 | `origami/trex_patch.py` + `origami/train_origami.py` (vendored from `full-pipeline:midtrain.py`, §7.2) + `origami/delayed_lerobot_dataset.py` (§7.5-B) + `.sh` | G9a, G9b, G17, §11.7 test |
| 10 | Full prep run (51 seasons, both splits) | manifest complete, no truncation > 1% |
| **10b** | **`origami/diagnose_shift.py` + `eval_offline.py --zero-shot`** (§11.9 mitigations 1–2) — no training | **G15**; zero-shot floor recorded |
| **10c** | `origami/refit_vqvae_stats.py` if G15 fails; else skip (§11.8 fix ladder) | G15 passes |
| 11 | Pilot train: staged-unfreeze phase A 2 000 steps (§11.9-3) + `eval_offline.py` §8.1-A/D | **§11.1 decision point**; beats the 10b floor |
| **11b** | Optional phase 0 (FLARE-only) / phase A (frozen latent) ablations, §7.5-E | each beats the 10b floor, else skip |
| 12 | Phase B: full train 40 000 steps with the delay curriculum | val loss curve, checkpoints on Drive |
| 13 | `origami/policy.py` + `retarget.py` + `serve_zenoh.py` + `bench_latency.py` | G10, G13, G14 |
| 14 | `eval_offline.py` §8.1-B..F + `eval_shadow.py` | G11 |
| 15 | `origami/docker/` submission image | SDK container checks pass |

---

## 13. Quick reference — what to import from T-Rex (never reimplement)

```python
from utils.lerobot_common import (              # T-Rex/utils/lerobot_common.py
    ACTION_DIM, ACTION_CHUNK, FRAME_STRIDE, N_FINGERS, N_FINGERS_PER_HAND,
    F6_PER_FINGER, F6_DIM, TRACKING_ERROR_DIM, STATS_KEY,
    KEY_HEAD, KEY_WRIST_R, KEY_WRIST_L, KEY_STATE, KEY_ACTION, KEY_ACTION_ABS,
    KEY_TACF6, DEFORM_KEYS,
    pose_matrix_to_9d, get_rot_mat, compute_chunk_delta_pose,
    compute_tracking_error_axis_angle, compute_bimanual_tracking_error,
    build_action_chunk, build_trex_features, calculate_stats, load_norm_stats,
)
from qwen_vla.lerobot_dataset import TRexLeRobotDataset, add_tracking_error_noise
from qwen_vla import Qwen3VLVLAModel, extend_position_ids_for_flare, split_slow_fast_embeds
from scripts.test import model_load, CascadedServer, _encode_tactile_f6, _encode_tactile_deform
# T-Rex/hardware_code/eval/eval_trex_async.py (import by path; the module has heavy
# hardware imports at top level -- copy these three pure functions into origami/retarget.py
# with a source citation instead):
#   rot6d_to_matrix (:283), matrix_to_rot6d (:295), aggregate_chunks (:75)
# T-Rex/hardware_code/teleop/ik_utils.py::PinkLocalIK.solve_ik (:58) -- the task weights and
# solver loop are copied verbatim into origami/kinematics.py (the upstream class hard-codes
# the Dexmate model and L_ee/R_ee frame names).
```
`PYTHONPATH` must contain `T-Rex/` (all its `.sh` scripts do this; `origami/__init__.py`
should do it too, guarded, so `import utils.lerobot_common` works).
