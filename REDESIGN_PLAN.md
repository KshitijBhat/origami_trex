# Origami × T-Rex — Redesign Plan (eef-62)

Convert the Robotic Origami Challenge dataset into the **exact** format
`T-Rex/qwen_vla/lerobot_dataset.py::TRexLeRobotDataset` consumes, train with upstream T-Rex's
recipe, and deploy through the competition's `origami-zenoh-v1` contract with differential IK.

**Status:** plan only, no code written. Implementation target: Sonnet 5.

**Citation style:** `path::symbol` is authoritative (line numbers drift). Line numbers appear
only where verified against `T-Rex@f88e10c` / `T-Rex@b23eafe` (full-pipeline).

---

## 0. Decisions (settled — do not re-litigate)

| # | Decision |
|---|---|
| D1 | **LeRobot v3.0 only** (`--data_format lerobot`). No JSON path. |
| D2 | **New top-level `origami/` package. `T-Rex/` stays byte-identical to upstream.** |
| D3 | **All 126 split seasons** (101 train / 25 val), stream-and-delete. 143 exist on the hub; the 17 extras are opt-in. Prep runs on a **large-RAM/large-disk server**, not the dev laptop. |
| D4 | **eef-62** action: `[L_delta9, L_hand22, R_delta9, R_hand22]`, `action_chunk=16`, `FRAME_STRIDE=1`. |
| D5 | **`motor` dims 58:65 are never predicted** — held at the observed value at deploy. |
| D6 | "raw tactile" = raw 60-D wrench + raw deform maps into the on-the-fly VQ-VAE (T-Rex's default). `tactile_raw` video is optional, off by default (§11.3). |
| D7 | **No crop.** Full square FOV, 480×480 → 224×224 (deploy-identical), then upsampled to a **probed** square `--image_size` (§4.5-A) in both training and deploy. |

---

## 1. Established facts (verified — cite these, don't re-derive)

### 1.1 Source dataset (`season_*/lerobot3.0/`)

Verified on the local fixture `season_POC22061_2026_07_23_10_20_33_train`.

* `codebase_version: v3.0`, `robot_type: north_ces`, `fps: 30`, 22 episodes, 95 533 frames.
* Episode lengths 4 038–5 154 frames (135–172 s), mean 4 343.
* `data/chunk-000/file-000.parquet` holds the **whole season** in one row group:
  `observation.state[65] f32`, `observation.state.joint_torque[65] f32`,
  `observation.state.tcp[24] f32`, `action[65] f32`, `observation.tactile[60] f32`,
  `timestamp f32`, `frame_index/episode_index/index/task_index i64`.
* **`observation.state.tcp` is identically zero** (`abs().max(axis=0)` all-zero over 95 533
  rows). There is no EEF data in the release — **FK is mandatory**.
* `observation.state` / `action` layout (65 D, absolute radians):

  | slice | group | URDF joint names |
  |---|---|---|
  | `0:7` | left arm | `left_arm_joint_1..7` |
  | `7:29` | left hand | `left_<HAND_ORDER>` (22) |
  | `29:36` | right arm | `right_arm_joint_1..7` |
  | `36:58` | right hand | `right_<HAND_ORDER>` (22) |
  | `58:63` | lower body | `lower_body_joint_1..5` |
  | `63:65` | neck | `neck_joint_1..2` |

  `HAND_ORDER` (22) — identical in the URDF, in
  `participant_local_evaluator/contract.py::_hand_joint_names`, and in
  `T-Rex/hardware_code/teleop/robot_descriptions.py::SHARPA_HAND_JOINT_ORDER`:
  ```
  thumb_CMC_FE, thumb_CMC_AA, thumb_MCP_FE, thumb_MCP_AA, thumb_IP,
  index_MCP_FE, index_MCP_AA, index_PIP, index_DIP,
  middle_MCP_FE, middle_MCP_AA, middle_PIP, middle_DIP,
  ring_MCP_FE, ring_MCP_AA, ring_PIP, ring_DIP,
  pinky_CMC, pinky_MCP_FE, pinky_MCP_AA, pinky_PIP, pinky_DIP
  ```
* **`action` is a look-ahead command.** `argmin_k mean|action[t] − state[t+k]|` over episode 0:
  arms k≈5–6 (0.17–0.20 s), hands k≈2, motor error 4e-4 rad (an echo of the state).
  So `action` ↔ *target*, `state` ↔ *current* — exactly T-Rex's `*_target_*` / `*_current_*`.
* Video streams (all `h264`, `yuv420p`, 30 fps, no audio), sizes measured on the fixture:

  | key | shape | size/season | used |
  |---|---|---|---|
  | `observation.images.head_left` | 480×480×3 | 854 MB | **yes** → `observation.images.head` |
  | `observation.images.head_right` | 480×480×3 | 844 MB | no (§4.5-D) |
  | `observation.images.wrist_left` | 480×480×3 | 957 MB | **yes** |
  | `observation.images.wrist_right` | 480×480×3 | 1.1 GB | **yes** |
  | `observation.images.tactile_deform` | 480×1200×3 | 80 MB | **yes** → split to 10 |
  | `observation.images.tactile_raw` | 480×1600×3 | 3.0 GB | optional (§11.3) |

  Fetched set (4 videos + data + meta) ≈ **3.1 GB/season** here; AV1 seasons are ~1 GB.
* **Video files are chunked *per key*, not per episode.** In the fixture `head_left` spans 6
  files, `tactile_deform` 1 file (all 95 533 frames), `tactile_raw` 22.
  `meta/episodes/**.parquet` gives per episode **and per video key**:
  `videos/<key>/{chunk_index,file_index,from_timestamp,to_timestamp}`, plus `length`, `tasks`,
  `data/{chunk,file}_index`, `dataset_from_index`, `dataset_to_index`.
  **Any decoder must use this table; never assume file-per-episode.**
* Single task string `"north ces task"` (`meta/tasks.parquet`) — names the rig, not the task.
  Overridden at conversion (§5.7).
* `meta/modality.json` declares state/action as one absolute 65-D `joints` block.

### 1.1a Season inventory — the dataset card's header is stale

`dataset.md`'s statistics table (51 seasons / 682 episodes / 4 763 267 frames) describes the
**original release** and must not be used for budgeting. Verified against the hub tree API and
`dataset.md`'s own `## My Split` block:

| set | count | note |
|---|---|---|
| season dirs on the hub | **143** | each has `lerobot3.0/` and `lerobotv2.1/` |
| `## My Split` train / val | **101 / 25** | list lengths match declared counts; zero overlap |
| split total | **126** | all 126 resolve on the hub |
| on hub, outside the split | **17** | added after the split was generated |
| local fixture (`…07_23_10_20_33…`) | — | **not on the hub, in neither split list** |

Hub date range 2026-05-14 → 2026-07-18; prefixes `POC22032` ×4, `POC22061` ×139.

**Frames.** Per-season `info.json` is not readable anonymously (repo **gated**, §5.1), so the
total is estimated from two anchors that agree: the fixture (95 533 frames/season) and the
stale card (4 763 267 / 51 = 93 397). Take **≈ 94 k frames/season**. The two anchors disagree
sharply on *episodes* per season (22 vs 13.4), so **frames is the reliable quantity**.

| scope | seasons | est. frames | est. hours @30 fps |
|---|---|---|---|
| split train | 101 | **≈ 9.5 M** | ≈ 88 h |
| split val | 25 | **≈ 2.4 M** | ≈ 22 h |
| **split total** | **126** | **≈ 11.8 M** | **≈ 109 h** |

`prepare.py` phase 0 must **replace this estimate with the exact sum** of per-season
`info.json total_frames` and print it.

**Season-list requirements — gate G18:**
* parse the train/val lists from `dataset.md`'s `## My Split` block into
  `origami/splits.json`; **never hard-code counts**;
* assert `len(train)==101`, `len(val)==25`, `train ∩ val == ∅`;
* assert every listed season resolves on the hub **before** the run starts;
* the **fixture season is in neither split** and must be asserted absent — folding it in would
  contaminate the held-out protocol;
* the 17 extras are opt-in via `--extra-seasons {none,train}` (default `none`). Adding them to
  **train only** keeps val frozen; adding them to val invalidates comparisons. Record the
  choice in `meta/origami_prep.json`.

### 1.2 Tactile — verified layout

* `observation.tactile` = `[10 fingers × 6]` flattened, order
  `left_{thumb,index,middle,ring,little}` then `right_{...}`, each `[fx,fy,fz,tx,ty,tz]`.
  **Exactly** T-Rex's `tactile_f6` order.
* `tactile_deform` 480×1200 = **2 rows × 5 cols of 240×240**;
  **row 0 = LEFT, row 1 = RIGHT; columns = thumb, index, middle, ring, pinky.**
  Verified at fixture frame 2205: forces `[10.9,.1,0,0,0 | 34.7,42.9,0,0,0]` N ↔ tile means
  `row0=[6.3,0,0,0,0]`, `row1=[9.1,18.7,0,0,0]`.
* All 3 channels of a decoded deform frame are equal (grayscale carried in `yuv420p`) →
  **decode the luma plane directly** (`frame.to_ndarray(format="gray")`), never YUV→RGB→gray.
* `tactile_raw` 480×1600 = 2×5 of **240×320**, mean ≈ 125.
* **Dead channels.** Mean |F| per finger over episode 0:
  `L = [4.44, 1.61, 0.57, 0.009, 0.065]`, `R = [6.47, 9.52, 0.73, 0.002, 0.003]` N.
  Both ring/pinky pairs carry no signal → `q99−q01 ≈ 0`; see §5.4 masking.

### 1.3 Robot model — `north_poc2_2_urdf_usd/north_poc2_2_v3_1.urdf`

The URDF is **complete**. (The SolidWorks exporter puts `name=` on its own line, so a
`grep '<joint name='` wrongly suggests the arms are missing. **Parse the XML.**)

* `robot name="north_poc2_2_with_double_continental_grip_description"`, 105 joints,
  **exactly 65 revolute** = 5 lower_body + 7 left_arm + 22 left_hand + 7 right_arm
  + 22 right_hand + 2 neck, matching §1.1 one-for-one.
* Chain: `root_link` → `lower_body_base_link` → `lower_body_link_1..5` → **`torso_base_link`**
  → `{left_arm_base_link → left_arm_link_1..7 → left_hand_base_link → left_hand_C_MC → fingers}`,
  `{right_arm_…}`, `{neck_base_link → neck_link_1..2 → head_base_link}`.
* **EEF frames: `left_hand_base_link` / `right_hand_base_link`** (fixed children of
  `*_arm_link_7`; the analogue of T-Rex's `L_ee`/`R_ee`).
* Arm joint limits (rad), for IK clipping and deploy safety:
  ```
  left  lower [-1.5359,-3.6652,-3.1067,-1.0472,-3.1067,-0.9599,-0.6981]
        upper [ 4.6775, 0.5236, 3.1067, 2.5307, 3.1067, 0.9599, 0.6981]
  right lower [-1.5359,-1.7453,-3.1067,-1.0472,-3.1067,-0.9599,-1.5708]
        upper [ 4.6775, 0.5236, 3.1067, 2.5307, 3.1067, 0.9599, 1.5708]
  velocity limit 2.6179 rad/s on every arm joint
  ```
* Sanity-checked with an independent numpy FK on fixture `state[0]`:
  `left_hand_base_link = (0.362, 0.194, 1.049) m`, `right = (0.362, −0.193, 1.050)`,
  `head_base_link = (0.002, 0.000, 1.459)`. Symmetric, in front, ~1.05 m high.
* Workspace over episode 0 (left): x ∈ [0.312, 0.405], y ∈ [0.066, 0.263], z ∈ [0.908, 1.049] m.
* **Per-frame EEF translation step: mean 0.39 mm, p99 2.04 mm** at 30 Hz → ~6 mm over a
  16-frame chunk. **The single most important modelling fact here** — §11.1.
* `lower_body_joint_3` varies ~0.19 rad within a season; joints 4–5 ≤0.08; neck_2 ~0.28.
  §3.2 explains why this does not matter.

### 1.4 T-Rex contract (`T-Rex@f88e10c`)

* **Feature schema** — `utils/lerobot_common.py::build_trex_features`:
  ```
  observation.images.head            video (3,H,W)
  observation.images.wrist_right     video (3,H,W)
  observation.images.wrist_left      video (3,H,W)
  observation.state                  float32 (62,)
  action                             float32 (16,62)   ← baked delta-base chunk
  action_abs                         float32 (62,)
  observation.tactile_f6             float32 (10,6)
  observation.tactile_deform.l0..l4  video (3,240,240) ← left thumb..pinky
  observation.tactile_deform.r0..r4  video (3,240,240) ← right thumb..pinky
  ```
  Constants: `ACTION_DIM=62`, `ACTION_CHUNK=16`, `FRAME_STRIDE=1`, `STATS_KEY="rlbench"`.
* **Pose math** — `lerobot_common.py`: `pose_matrix_to_9d`, `get_rot_mat`,
  `compute_chunk_delta_pose`, `compute_tracking_error_axis_angle`,
  `compute_bimanual_tracking_error`, `build_action_chunk`. **Import verbatim; never reimplement.**
* **Norm-stats sidecar** `meta/trex_norm_stats.json`:
  `{"rlbench": {"action", "state", "tactile_f6", "tracking_error", "num_transitions",
  "num_trajectories"}}`; each stat block has `mean/std/max/min/q01/q99/mask`.
  `action` stats are **per-(step,dim), shape [16,62]** (`NormStatsAccumulator.assemble`,
  `lerobot_common.py:213`). `tracking_error` has `mean/std/mean_abs/mask` at 56 D.
* **Loader** — `qwen_vla/lerobot_dataset.py::TRexLeRobotDataset` (`:75`). Builds
  `delta_timestamps`: head `[0, s/fps, …, n_flare_steps·s/fps]` with `s=flare_frame_stride`
  (`_head_offsets`, `:133`); `observation.tactile_f6` `[-(W-1)/fps … 0]`, `W=vqvae_window=16`
  (`_f6_offsets`, `:139`). Stats block taken as `next(iter(self.stats_data))` (`:109`), so the
  top-level key name is free. `collate_fn` (`:183`) normalizes with q01/q99, does flow-matching
  noising (Beta(1.5,1.0)), maps slow=`head[0]` / fast=`[wrist_right, wrist_left]`, applies the
  chat template (`:248`), left-pads `input_ids`, returns 21 keys.
  `observation.state` is read only when `use_robot_state=1`; **`action_abs` is never read by
  the loader** (it only feeds tracking-error stats at conversion time).
  `tactile_f6s_delayed = norm_tacf6` is **hard-wired to delay 0** (`:301`) — see §7.5-B.
* **Batch contract** from `collate_fn` (B, chunk 16, dim 62):
  ```
  input_ids [B,L] i64          attention_mask [B,L] i64
  pixel_values [Σpatch,C]      image_grid_thw [3B,3]     n_slow_images = 1
  noisy_actions/target/norm_actions/eps_r [B,16,62] bf16  timesteps/time_r [B] bf16
  tactile_f6s (+_delayed) [B,10,6] bf16
  tactile_deforms (+_delayed) [B,10,1,240,240] f32
  tactile_codes = None         tactile_f6_history [B,16,10,6] f32
  state_raw [B,62] bf16|None   flare_pixel_values [B·S,C] bf16   flare_grid_thw [B·S,3]
  ```
* **Model shape constraints** (`qwen_vla/modeling_vla.py`):
  * `tacf6_embedder = ActionEmbedder(6, H)` (`:105`) → `tactile_f6` must be `[B,10,6]`.
  * `deform_proj = ActionEmbedder(28800, H)` = 128·15·15 (`:109`) → `DeformAE.py::DeformEncoder`
    (ResNet-18 stem + layer1..3, total stride 16) requires **exactly 240×240**. 240/16 = 15.
    **Any other deform size breaks the head.**
  * `_embed_tactile_observations` (`:469`) takes `tactile_deform [B,n_fingers,C,H,W]`, `C=1`.
* **Stages.** `main` ships post-train + inference only. `pretrain.py` / `midtrain.py` /
  `prepare_midtrain_merged.py` live on **`full-pipeline`** (pinned `b23eafe`). **The ViT is
  frozen in all three stages** — `pretrain.py:946`, `midtrain.py:1427`, `train.py:860`.
  `midtrain.py` is a superset of `train.py` and also supports `--data_format lerobot`
  (`:1457`) — §7.5.
* **Official post-train recipe** — `scripts/train.sh`:
  ```
  action_dim 62  action_chunk 16  image_size 384 288
  use_robot_state 0  use_tactile_vec 1  use_tactile_deform 1  use_tactile_vqvae 1
  tactile_intermediate_size 1536  training_stage 2  tactile_loss_weight 1.0
  cascaded_total_steps 10  cascaded_split_step 6
  cascaded_tactile_dropout 0.1  cascaded_loss_weight 1.0
  use_flare 1  n_flare_tokens_per_frame 4  n_flare_steps 8
  flare_loss_weight 0.5  flare_frame_stride 4  flare_layer_index -1
  lr 1e-4  min_lr_ratio 0  weight_decay 0  max_grad_norm 1.0
  train_bsz_per_gpu 16 × 8 GPUs  grad_accum 1  n_epochs 100  save_freq 50
  val_ratio 0.05  val_freq 500  max_val_batches 30  resume_source midtrain
  config/sft_qwen.yaml (DeepSpeed ZeRO-2, bf16)
  RESUME_CHECKPOINT  miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6
  ORIGIN_MODEL_PATH  Qwen3-VL-2B-Instruct
  DEFORM_ENCODER     sharpa_wave_deform_encoder.pth
  ```
* **Inference server** — `scripts/test.py`. `model_load` (`:153`) restores
  `tactile_intermediate_size, n_flare_*, use_tactile_code, vqvae_codebook_size,
  use_tactile_vqvae, vqvae_config, cascaded_{total,split}_step` from `training_args.json`, and
  q01/q99 from `stats_data.json` (falls back to `next(iter(stats_raw))`, `:263`).
  `CascadedServer` (`:330`) holds the slow-tick snapshot (`cached_kv`, `x_split`, `tau_split`,
  `position_ids`, `n_action_in_cache`) + a 16-frame rolling F6 buffer (`:407`).
  `_run_slow` (`:486`) → `forward_flow_action_partial(refresh_clean_kv=True)`;
  `_run_fast` (`:609`) → `tactile_flow_continue` → denormalized chunk.
* **Deploy reference client** — `hardware_code/eval/eval_trex_async.py`:
  `aggregate_chunks` (`:75`), `rot6d_to_matrix` (`:283`, Gram–Schmidt), `matrix_to_rot6d`
  (`:295`), `get_current_pose` (`:446`). Chunk anchor captured once at `:992`; per-row delta
  applied from `:1113`. Cadence `chunk_size 16, execute_steps_per_chunk 16,
  refine_offsets [4,8,12]` at `command_hz 30` ⇒ **slow 1.875 Hz, fast 7.5 Hz**.
* **IK** — `hardware_code/teleop/ik_utils.py::PinkLocalIK.solve_ik` (`:58`):
  `FrameTask(position_cost=50.0, orientation_cost=1.0, lm_damping=0, gain=0.2)` per arm,
  `PostureTask(cost=0.2, gain=0.2)` → previous q, `PostureTask(cost=0.05, gain=0.2)` → default q,
  `dt=0.05`, **5 iterations**, `solver="daqp"`, `damping=0`, clip to limits each iteration.
  Model from `robot_descriptions.py::build_reduced_bimanual_robot` — torso/head/wheels locked,
  `nq == 14`.

### 1.5 Deploy contract (`origami-zenoh-v1`)

From `origami-inference-kit-participant/docs/robot_io_spec.md` and
`sharpa_north_ces_lite_sdk-main/participant_local_evaluator/contract.py`.

* One RPC: `TeamPolicy.infer(obs) -> np.float32[T, 65]`, `T == metadata["action_horizon"]`,
  fixed for the process, `1 ≤ T ≤ 1024`. Plus `reset()`.
* Metadata exactly: `protocol_version="origami-v1"`, `action_dim=65`,
  `action_type="absolute_joint_position"`, `action_units="radians"`,
  `joint_names=JOINT_NAMES` (the 65 names in §1.1 order).
* Observation keys (msgpack ndarrays):
  ```
  observation/image/head_left       uint8 (224,224,3)   required
  observation/image/head_right      uint8 (224,224,3)   required
  observation/image/wrist_left      uint8 (224,224,3)   required
  observation/image/wrist_right     uint8 (224,224,3)   required
  observation/image/tactile_deform  uint8 (480,1200,3)  required
  observation/image/tactile_raw     uint8 (480,1600,3)  optional (may be absent)
  observation/state                 float32 (65,)       required, absolute radians
  observation/state/joint_torque    float32 (65,)       required (zero-filled in practice)
  observation/tactile               float32 (60,)       required
  prompt                            str                 required
  ```
  The organizer's reference prompt is `"fold the plane"`
  (`openpi-base-main/src/openpi/policies/north_ces_policy.py:33`); not contractually pinned —
  the spec only requires a string. §5.7.
* **Camera preprocessing, quoted:** *"The native camera frame is 1920×1536. The organizer
  directly squashes it to 224×224; aspect ratio is deliberately not preserved and no
  padding/letterbox is added. … Participants must not restore the native aspect ratio or add
  black bars before model inference."* → the basis for D7.
* Actions "are not velocity, torque, delta, normalized values, tokens, or per-part maps"; all
  finite; columns exactly §1.1.
* `execution_mode`: `async` (default — the gateway keeps executing the previous chunk while
  inference runs) or `sync`. Submission is a Docker image.
* Unavailable sensors are **zero-filled, not omitted** (except `tactile_raw`, which may be absent).

---

## 2. Repo layout

```
origami_trex/
├── T-Rex/                          submodule, pinned f88e10c, ZERO local diff
├── north_poc2_2_urdf_usd/          official robot asset (complete)
├── origami-inference-kit-participant/   competition SDK
├── dataset.md                      dataset card + the 101/25 season split
├── REDESIGN_PLAN.md
├── origami/
│   ├── constants.py                65-D layout, joint names, keys, INSTRUCTION
│   ├── splits.json                 train/val season lists parsed from dataset.md
│   ├── kinematics.py               FK + Pink IK on north_poc2_2
│   ├── decode.py                   PyAV episode-slice decode, luma, deform split
│   ├── fetch.py                    per-season HF stream-and-delete (gated: HF_TOKEN)
│   ├── stats.py                    bounded-memory q01/q99 accumulator
│   ├── convert.py                  season → T-Rex eef-62 LeRobot shard
│   ├── merge.py                    shard concatenation into one root
│   ├── prepare.py                  CLI: fetch → convert → merge
│   ├── verify.py                   all §10 gates as subcommands
│   ├── trex_patch.py               2 monkeypatches (grad-ckpt KV)
│   ├── train_origami.py            vendored full-pipeline:midtrain.py + single-GPU knobs
│   ├── delayed_lerobot_dataset.py  loader + tactile delay curriculum (§7.5-B)
│   ├── train_origami.sh            the recipe
│   ├── policy.py                   checkpoint → eef-62 chunk (slow/fast)
│   ├── retarget.py                 eef-62 chunk → 65-D absolute joints via IK
│   ├── serve_zenoh.py              origami-zenoh-v1 TeamPolicy + server
│   ├── eval_offline.py             held-out-season metrics (+ --zero-shot)
│   ├── eval_shadow.py              SDK Shadow / wire-contract check
│   ├── diagnose_shift.py           pretrain↔origami shift report (§11.9)
│   ├── refit_vqvae_stats.py        re-fit tacf6_vqvae_{min,max,mask} (§11.8)
│   ├── bench_latency.py            slow/fast latency → feasible action_horizon
│   ├── docker/Dockerfile           submission image
│   └── tests/                      pytest, CPU-only, no GPU, no network
└── pyproject.toml                  uv deps
```

### 2.1 Git hygiene (step 1)

```bash
git rm -r --cached T-Rex && rm -rf T-Rex
git submodule add https://github.com/ZhuoyangLiu2005/T-Rex.git T-Rex
git -C T-Rex checkout f88e10c
git -C T-Rex fetch origin full-pipeline      # b23eafe — source for §7.2 vendoring
git add .gitmodules T-Rex
```
A submodule pins one commit, so `full-pipeline` is fetched but not checked out; read it with
`git -C T-Rex show origin/full-pipeline:scripts/midtrain.py`.
`origami/tests/test_upstream_drift.py` (gate G0) hashes every upstream file we import or
vendor, on both refs.

Delete: `T-Rex_modified/`, `OLD_REIMPLEMENTATION_PLAN.md`, `NEW_DESIGN.md`,
`prepare_full.log`, `train_pilot.log`, `trex_colab.ipynb` (rewritten §8.4), `reimplementation/`,
`delta_run/`. `.gitignore`: `data/ out/ *.log wandb/ outputs/ .pytest_cache/`.
Keep the fixture season locally for tests.

---

## 3. `origami/kinematics.py`

The **single source of FK/IK** for prep, eval and deploy. Never duplicated.

### 3.1 API

```python
HAND_ORDER: tuple[str, ...]        # 22 names, §1.1
JOINT_NAMES_65: tuple[str, ...]    # 65 URDF names in dataset-index order
EEF_FRAMES = {"left": "left_hand_base_link", "right": "right_hand_base_link"}

@dataclass(frozen=True)
class LockedConfig:
    lower_body: np.ndarray   # (5,)
    neck:       np.ndarray   # (2,)
    left_hand:  np.ndarray   # (22,)  locked out of the reduced model; value irrelevant
    right_hand: np.ndarray   # (22,)
    def digest(self) -> str  # sha1 of rounded values; recorded in prep metadata

class OrigamiKinematics:
    def __init__(self, urdf_path, locked: LockedConfig)
    def fk(self, q_left7, q_right7) -> dict[str, pin.SE3]
    def fk_matrices(self, q_left7, q_right7) -> tuple[np.ndarray, np.ndarray]   # two 4×4
    def solve_ik(self, target_left, target_right, warm_left7, warm_right7)
    def clip_arm(self, q_left7, q_right7)
    def check_collision(self, q_left7, q_right7) -> bool
    arm_limits: dict[str, tuple[np.ndarray, np.ndarray]]   # from the URDF
```

* `__init__` mirrors `robot_descriptions.py::build_reduced_bimanual_robot`:
  `RobotWrapper.BuildFromURDF(urdf_path, [mesh_dir])`; build a full `pin.neutral(model)`, write
  the locked values at their `idx_q`, then `pin.buildReducedModel(...)` locking everything
  except `left_arm_joint_1..7` + `right_arm_joint_1..7`.
  Assert `nq == nv == 14` and the joint-name set (G1b).
* **No SRDF ships for this robot.** Build the collision-pair disable list programmatically
  (adjacent links in the URDF tree + all hand↔hand pairs). Do **not** put collisions inside the
  Pink `Configuration` — upstream notes it slows Pink down; expose `check_collision` separately.
* `fk`: `pin.forwardKinematics` → `pin.updateFramePlacements` → `data.oMf[getFrameId(f)].copy()`.
* `solve_ik`: copy `ik_utils.py::PinkLocalIK.solve_ik` (§1.4) verbatim, substituting
  `L_ee`/`R_ee` → `EEF_FRAMES` and `default_qpos` → §3.3's arm medians.

### 3.2 Why locking `lower_body`/`neck` at a constant is exact

Arms hang off `torso_base_link`, downstream of `lower_body_joint_1..5` only, so
`T_root_eef(q_lb, q_arm) = T_root_torso(q_lb) · T_torso_eef(q_arm)`. The action representation
is built from
`delta9(A,B) = [A_R^T (B_t − A_t), (A_R^T B_R)[:,0], (A_R^T B_R)[:,1]]`, which is **invariant**
under a common left-multiplied rigid transform: `delta9(X·A, X·B) = delta9(A,B)`. Both poses
come from the same locked `T_root_torso`, so **the action is independent of the lock value**.
The *absolute* 9-D state is not invariant, but is consistent as long as prep and deploy use the
same `LockedConfig` — and the q01/q99 stats are computed in that same frame.

`LockedConfig` is written to `meta/origami_prep.json` at conversion and **loaded, not
re-chosen**, by `eval_offline.py` and `serve_zenoh.py`. `__init__` refuses a config whose
`digest()` disagrees with the one recorded in the dataset/checkpoint (G1c).

### 3.3 `LockedConfig` values

Computed once over the training split in phase 0, then frozen:
```
lower_body = per-dim median of state[:, 58:63]
neck       = per-dim median of state[:, 63:65]
left_hand  = zeros(22);  right_hand = zeros(22)      # locked out; value irrelevant
left_arm_default  = per-dim median of state[:, 0:7]   # PostureTask(cost=0.05) target
right_arm_default = per-dim median of state[:, 29:36]
```
Emit with `urdf_sha256` into `meta/origami_prep.json`. **Compute, never hard-code.**

---

## 4. `origami/decode.py`

### 4.1 Episode-slice decoding

```python
def decode_episode_stream(video_path, from_ts, to_ts, n_frames, fmt) -> Iterator[np.ndarray]
```
* PyAV. `container.seek(int(from_ts / stream.time_base), stream=stream)`, then **drop frames
  until `frame.pts * time_base >= from_ts − 0.5/fps`**, then yield exactly `n_frames`. Seek
  lands on the preceding keyframe, so pre-roll is mandatory.
* Fewer than `n_frames` → raise; do **not** silently pad. The caller clamps (§4.4).
* `fmt="gray"` for `tactile_deform` (§1.2), `"rgb24"` for RGB cameras.
* Retry path: fall back to a linear scan from t=0 on a PTS discontinuity; log every fallback.

### 4.2 Deform splitting

```python
def split_deform_strip(y: np.ndarray) -> np.ndarray:
    """(480,1200) uint8 luma -> (10,240,240) uint8, [L thumb..pinky, R thumb..pinky]."""
    assert y.shape == (480, 1200)
    return y.reshape(2, 240, 5, 240).transpose(0, 2, 1, 3).reshape(10, 240, 240)
```
Also `split_raw_strip` for the optional 480×1600 → `(10,240,320)`.

### 4.3 RGB resize (D7)

```python
def squash_to_wire(rgb480):   # (480,480,3) -> (224,224,3), deploy-identical
    return cv2.resize(rgb480, (224, 224), interpolation=cv2.INTER_AREA)
```
Prep stores 224×224. The loader's `_img_to_pil` upsamples to `--image_size` with LANCZOS, and
`serve_zenoh.py` applies **the same** step from the 224 wire image — the two pipelines are then
identical from the wire onward (G12). `INTER_AREA` is correct for the 480→224 decimation; the
organizer's own 1920×1536→224 squash may use a different filter, which affects sharpness only,
not geometry, and 224 is the common bottleneck either way.

### 4.4 Episode length reconciliation

`N = min(length_from_meta, frames_decoded_per_stream…)`. Log any `N < length` into
`meta/origami_prep.json["truncated"]` as `(season, episode, length, N)`. Reject the episode if
`N < 64` (too short for one chunk + one F6 window).

### 4.5 Vision token budget and view allocation

**The resolution gap versus T-Rex is real and cannot be closed at deploy.**

| | T-Rex head | Origami head |
|---|---|---|
| native | 640×360 | 1920×1536 |
| preprocessing | crop `(0,300,140,540)` → 400×300 | squash → 480×480 → 224×224 |
| effective scale | **≈ 1 : 1** | **8.6× linear downsample** |

Measured on the fixture, the 15×15 cm sheet spans ~60×40 px of the 480² frame → **≈ 28×19 px
at 224²**. **Crease geometry is not present in the head view.** The fine-grained signal lives
in the two wrist views (the paper fills a large fraction of each fisheye) and in tactile.

**A. Probe the processor factor — do not assume a token count.**
`smart_resize` rounds each side to `factor = image_processor.patch_size *
image_processor.merge_size`, read from the checkpoint's `preprocessor_config.json` — **not**
from the vision config. `Qwen3VLVisionConfig` defaults to `16 × 2` (factor 32) but the
`Qwen2VLImageProcessor` class default is `14 × 2` (factor 28). Tokens = `(h/f)·(w/f)`:

| `--image_size` | f = 32 | f = 28 |
|---|---|---|
| `384 288` (upstream) | 12×9 = **108** | 14×10 = **140** |
| `384 384` | 12×12 = **144** | 14×14 = **196** |
| `336 336` | rounds to 320 → 100 | 12×12 = **144** |
| `224 224` | 7×7 = 49 | 8×8 = 64 |

`origami/tests/test_processor_probe.py` prints the factor and resolved grid per candidate, then
**picks the square size whose token count is closest to upstream's `384 288`** so the midtrain
ViT operates near its trained point: `384 384` at f=32, `336 336` at f=28. Record it in
`training_args.json` (G16). This also scales FLARE: `n_flare_steps 8` pushes 8 extra head
frames per sample through the no-grad ViT (`train.py:1143`), so head cost ×9.

**B. Per-view `image_size` (ablation).** `collate_fn` calls the processor **once** with
`pil_slow + pil_fast` and Qwen3-VL grids each image independently, so per-view sizes need no
architecture change — only that `_img_to_pil` take a per-key size. Add `--image_size_head` /
`--image_size_wrist` (defaulting to `--image_size`) in the loader subclass and mirror them in
`serve_zenoh.py`. First ablation: head token-matched, **wrists one step larger**.
`split_slow_fast_embeds` derives `n_slow_img_tokens` from the first `n_slow` grid rows
(`train.py:962`), so the boundary stays correct as long as slow images come first.

**C. Optional workspace crop (ablation, default off).** The top ~35% of the head frame is dark
background (row-mean ≈ 8/255) and the bottom band is the robot's chassis. Cropping adds no
pixels but reallocates tokens onto the paper. Rules: applied **after** the 480→224 squash (so
deploy reproduces it from the wire image — cropping the 480 source would give training real
detail deploy cannot supply); **square** (so we are not "restoring the native aspect ratio");
box recorded in `meta/origami_prep.json` + `training_args.json` and checked by G12.
`--head_crop_rel y0 y1 x0 x1` in normalized coords, default `0 1 0 1` = off. Adopt only if
§8.1-A/C improve.

**D. `head_right` unused by default.** T-Rex has one slow camera, and the organizer's own
reference policy also consumes only `head_left`/`wrist_left`/`wrist_right`
(`north_ces_policy.py::NorthCESInputs`, `:47-70`). `split_slow_fast_embeds` does support
`n_slow_images = 2`, so stereo is a legitimate ablation at +1 ViT pass — but it needs
`head_right` refetched (+844 MB/season), so **decide before the full prep run**.

---

## 5. Conversion

### 5.1 `origami/fetch.py`

```python
HF_REPO_ID = "SharpaIT/Robotic_Origami_Challenge"
NEEDED_VIDEO_KEYS = ("observation.images.head_left", "observation.images.wrist_left",
                     "observation.images.wrist_right", "observation.images.tactile_deform")

def season_allow_patterns(season, include_raw=False) -> list[str]   # meta/**, data/**, videos/<key>/**
def download_season(season, cache_root, token, include_raw=False) -> str
def have_season(season, cache_root, include_raw=False) -> bool
def drop_season(season, cache_root) -> None
```
* **The repo is gated.** Anonymous file reads return
  `HTTP 401 "Access to dataset SharpaIT/Robotic_Origami_Challenge is restricted"` (the *tree*
  listing is public; `resolve` is not). `prepare.py` must require `HF_TOKEN` / `--hf-token` and
  validate it with one authenticated `HfApi().dataset_info(...)` **before** phase 0. Do not
  discover this at season 1 of 126.
* `snapshot_download(repo_id, repo_type="dataset", allow_patterns=…, local_dir=…,
  max_workers=8, token=token)`.
* Season enumeration = `HfApi().list_repo_tree(...)` ∩ the parsed split lists (G18).
* `have_season` globs for real `*.parquet` / `*.mp4` — an interrupted download leaves the tree
  with no media.
* Exponential backoff on `HfHubHTTPError`, 3 attempts, then mark the season failed and continue.

### 5.2 `origami/convert.py` — per-season shard

`convert_season(season_dir, out_root, kin, cfg) -> SeasonResult`, per episode in
`episode_index` order:

1. Read the episode row from `meta/episodes/**.parquet` → `length`, `dataset_from_index`,
   `dataset_to_index`, per-key `(chunk_index, file_index, from_timestamp)`. Assert
   `tasks[0] == "north ces task"`; the written instruction is `cfg.instruction` (§5.7).
2. Slice season parquet `[dataset_from_index : dataset_to_index]` → `state65 (N,65) f64`,
   `action65 (N,65) f64`, `tactile60 (N,60) f32`. **Cast joints to float64 before FK** —
   float32 loses ~1e-4 rad of round-trip accuracy, comparable to the 0.39 mm/frame signal.
3. Open the 4 video streams for this episode (§4.1), one thread each, bounded queues.
4. `N = min(...)` (§4.4).
5. FK per frame — **one `forwardKinematics` call returns both arms; do not call it twice**:
   ```python
   S_l[i], S_r[i] = kin.fk_matrices(state65[i, 0:7],  state65[i, 29:36])
   A_l[i], A_r[i] = kin.fk_matrices(action65[i, 0:7], action65[i, 29:36])
   states      = concat([pose_matrix_to_9d(S_l), state65[:, 7:29],
                         pose_matrix_to_9d(S_r), state65[:, 36:58]], axis=1)   # (N,62)
   abs_targets = concat([pose_matrix_to_9d(A_l), action65[:, 7:29],
                         pose_matrix_to_9d(A_r), action65[:, 36:58]], axis=1)  # (N,62)
   ```
6. Per frame `i ∈ [0, N)`:
   ```python
   chunk = build_action_chunk(S_l, A_l, action65[:, 7:29],
                              S_r, A_r, action65[:, 36:58], i, N)      # (16,62) f32
   tgt   = min(i + FRAME_STRIDE - 1, N - 1)                            # == i at stride 1
   frame = {
       "task": cfg.instruction,                       # §5.7 — NOT "north ces task"
       "observation.images.head":        squash_to_wire(next(head_iter)),
       "observation.images.wrist_right": squash_to_wire(next(wr_iter)),
       "observation.images.wrist_left":  squash_to_wire(next(wl_iter)),
       "observation.state":              states[i].astype(np.float32),
       "action":                         chunk,
       "action_abs":                     abs_targets[tgt].astype(np.float32),
       "observation.tactile_f6":         tactile60[i].reshape(10, 6).astype(np.float32),
       **{DEFORM_KEYS[k]: gray_to_3ch(deform10[k]) for k in range(10)},
   }
   ds.add_frame(frame); acc.add_frame(chunk, states[i], tac)
   ```
   `build_action_chunk` is **imported** from `lerobot_common.py`. Its clamp
   `fut = min(min(i + k·stride, N-1) + stride - 1, N-1)` reduces to `min(i+k, N-1)` at stride 1,
   so the chunk **clamps at the episode end and never crosses episodes** (`N` is this episode's
   length). `gray_to_3ch(a) = np.repeat(a[:, :, None], 3, axis=2)`, matching
   `convert_inlab_to_lerobot.py::_gray_to_3ch`.
7. `ds.save_episode()`; `acc.add_episode_tracking(states[:N], abs_targets[:N])`.

After all episodes: `ds.finalize()`, `acc.dump(path)`.

**Writer:**
```python
features = build_trex_features(head_shape=(3,224,224), include_wrist=True,
                               wrist_shape=(3,224,224), include_tactile=True,
                               deform_shape=(3,240,240), include_action_abs=True)
ds = LeRobotDataset.create(repo_id=f"origami/eef62_{season}", fps=30, features=features,
                           root=shard_root, robot_type="north_poc2_2", use_videos=True)
```
Set `data_files_size_in_mb` / `video_files_size_in_mb` small enough that **each episode lands
in its own data/video file** (e.g. 64 / 128 MB) — this makes §5.3 a pure copy-and-renumber.

**Video encoding.** Deform maps carry physically meaningful uint8 depth → encode **losslessly**
(`libx264 -qp 0` or `ffv1`) so decoded channel-0 equals the source luma exactly.
`--deform_codec {lossless_h264, ffv1, crf18}`, default `lossless_h264`; G6 checks the
round-trip. RGB cameras use LeRobot's default codec.

### 5.3 `origami/merge.py`

**First probe for `lerobot.datasets.aggregate.aggregate_datasets` (or in
`lerobot.datasets.utils`) in the pinned version. If it exists, use it and delete the
hand-rolled merger.** Otherwise `merge_shards(shard_roots, out_root)`:

1. Assert identical `features`, `fps`, `codebase_version` across shards.
2. Copy `data/**` and `videos/**` with `(chunk_index, file_index)` renumbered into one global
   sequence per key; keep the mapping.
3. Rewrite `meta/episodes/**.parquet`: renumber `episode_index`, recompute
   `dataset_from_index`/`dataset_to_index` cumulatively, remap every `data/` and `videos/<key>/`
   chunk+file index. **`from_timestamp`/`to_timestamp` stay unchanged** because files are copied
   whole — which is why §5.2 forces one episode per file.
4. Rewrite the data parquets' `index` column to be globally increasing and renumber their
   `episode_index`.
5. Merge `meta/tasks.parquet` (single task) and regenerate `meta/stats.json`.
6. Merge per-shard stat states → `meta/trex_norm_stats.json` (§5.4) + `meta/origami_prep.json`.
7. Rewrite `meta/info.json`: `total_episodes`, `total_frames`, `total_tasks`, `splits`.

Two roots are produced: `origami_eef62_train/` (101 seasons) and `origami_eef62_val/` (25).

### 5.4 `origami/stats.py` — bounded-memory q01/q99

**`lerobot_common.py::NormStatsAccumulator` keeps every frame in RAM: ≈ 11.8 M × 16 × 62 × 4 B
= 47 GB for the action block alone.** Replace with an API-compatible accumulator:

```python
class StreamingNormStats:
    def __init__(self, reservoir: int = 200_000, seed: int = 0)
    def add_frame(self, action_chunk, state, tactile_f6=None)   # same signature as upstream
    def add_episode_tracking(self, states_62, abs_targets_62)
    def merge(self, other) -> None                              # for shard merging
    def assemble(self) -> dict                                  # upstream schema
    def write(self, dataset_root) -> str                        # meta/trex_norm_stats.json
    def dump(self, path); @classmethod load(cls, path)          # shard checkpointing
```
* **mean/std/min/max:** exact and streaming (`n`, `Σx`, `Σx²`, `min`, `max` per element;
  float64 accumulators). Shapes `[16,62]`, `[62]`, `[60]`, `[56]`.
* **q01/q99:** uniform reservoir (Vitter Algorithm R) per block. Action reservoir at 200 k =
  `200 000 × 16 × 62 × 4 B = 794 MB`; state 50 MB; tactile 48 MB; tracking 45 MB. Drop to 100 k
  (397 MB) if RAM is tight — G8 bounds the error. `merge()` combines reservoirs by weighted
  subsampling proportional to `n`.
* **Degenerate-dim masking.** Set `mask[j] = (q99[j] − q01[j]) > eps`, `eps = 1e-6` for
  state/action and `1e-3` for `tactile_f6`. Masked-off dims pass through `_normalize` unchanged
  (`lerobot_dataset.py:71` uses `np.where(mask, ...)`), which is what we want for the dead
  ring/pinky channels (§1.2). Record the masked dims in `meta/origami_prep.json` and print them
  — **an all-`True` tactile mask is a bug** (G8b).
* **Tracking-error robustness.** Episode starts show a >1 rad state↔action gap on
  `thumb_CMC_FE` (state 0.100 vs action 1.139) — a hand-init transient. Compute
  `tracking_error` over `t ∈ [30, N)` and also emit `tracking_error_full`. Only `mean`/`std` are
  consumed, and only when `--use_robot_state 1`.
* Keep `STATS_KEY = "rlbench"` — the loader takes `next(iter(...))` so the name is free, but
  there is no reason to diverge.

### 5.5 `origami/prepare.py`

```
python -m origami.prepare --split train \
  --out-root /mnt/big/origami/eef62_train --cache-root /mnt/big/origami/_src \
  --workers 3 --disk-budget 3 --urdf north_poc2_2_urdf_usd/north_poc2_2_v3_1.urdf
```
* **Phase 0** (`--phase locked`): validate `HF_TOKEN`; resolve the season list (G18); fetch only
  `meta/**` + `data/**` per season (~66 MB, no video) to compute the exact frame total and the
  `LockedConfig` + arm medians (§3.3); write `locked_config.json`. **Must run first.**
* **Phase 1:** `ProcessPoolExecutor(workers)` over seasons, with a `ThreadPoolExecutor`
  prefetching downloads so at most `disk-budget` seasons are on disk. Each worker:
  `download → convert_season → acc.dump → drop_season`. Seasons with a `DONE` marker are
  skipped.
* **Phase 2:** `merge_shards(...)` then `verify.py --root out_root`.
* Resumability: per-season `DONE` markers + `manifest.jsonl` with
  `(season, n_episodes, n_frames, truncated, elapsed_s, locked_digest)`.
  Changing `LockedConfig`, `--image-size`, `--deform-codec`, `--frame-stride` or the chunk
  parameters **invalidates the root** — refuse to write into a root whose
  `meta/origami_prep.json` disagrees.

### 5.6 Storage and time budget (print in the CLI banner)

Per season, measured on the fixture (95 533 frames):

| artifact | size |
|---|---|
| `action[16,62]` f32 | 3 968 B/frame → **379 MB** |
| `state` + `action_abs` + `tactile_f6` f32 | 736 B/frame → 70 MB |
| head + 2 wrists @ 224² | ~250 MB |
| 10 deform videos @ 240², lossless | ~150 MB |
| **total** | **≈ 0.85 GB/season → ≈ 107 GB for 126 seasons** |

Source download **126–390 GB** (1–3.1 GB/season by codec), stream-and-delete, peak on disk
≈ `disk-budget × 3.5 GB`. **The dev laptop has 23 GB free — prep must run on the large server.**
`prepare.py` must `statvfs` both paths and refuse to start below
`output_estimate + disk_budget × 3.5 GB`.

**Optional `--action-layout abs`** (not default): store only `action_abs[62]` and derive the
`[16,62]` chunk in a loader subclass from `delta_timestamps["action_abs"] = [k/fps for k in
range(16)]` + `compute_chunk_delta_pose(state_pose, target_pose)`. 16× smaller on the action
block (**107 → ≈ 62 GB**) and **numerically identical** — G4b proves it. Default stays baked
because that is the exact upstream schema.

### 5.7 Language instruction

The stored task string names the rig, not the task. Upstream's own defaults are no better
(`"I am T-Rex."` in `gen_json_bimanual.sh:17`, `convert_inlab_to_lerobot.sh:16`), but
`convert_inlab_to_lerobot.py::DEFAULT_INSTRUCTION` shows the intended style: one long, concrete
sentence. The organizer's reference prompt is `"fold the plane"`.

```python
# origami/constants.py
INSTRUCTION = (
    "Use both hands to fold the paper into an airplane on the table, alternating hands to crease the corners inward to the center and fold the wings down."
)
WIRE_PROMPT_REFERENCE = "fold the plane"   # for mismatch logging at deploy
INSTRUCTION_SHORT     = "fold the plane"   # short-instruction ablation
```
* `convert.py --instruction` (default `INSTRUCTION`) writes it as the LeRobot `task` for every
  frame; the loader surfaces it as `x["task"]` and the chat template puts it **between** the
  slow image and the two fast images — exactly where upstream puts it.
* Recorded in `meta/origami_prep.json` and copied into `training_args.json` (§7.2 delta 4).
* **At deploy, `serve_zenoh.py` uses the instruction from `training_args.json` and ignores
  `obs["prompt"]`** — a single-task policy must be fed the string it was trained on, and the
  contract makes this ours: *"language tokenization is participant-internal."* Log a one-time
  warning on mismatch with `WIRE_PROMPT_REFERENCE`; `--use-wire-prompt` passes it through for
  robustness testing. Gate G14 enforces identity.
* **No per-fold progress hints** ("fold 3 of 6") — the fold index is unknown at deploy, so such
  conditioning is unusable and would teach the policy to depend on it.

---

## 6. Exact tensor definitions (single source of truth)

```
FK9(q7_side) : T = kin.fk_matrices(q_left7, q_right7)[side]     # 4×4, torso-locked frame
               FK9 = [ T[:3,3], T[:3,0], T[:3,1] ]              # trans, R col1, R col2

state[62]      = [ FK9(state65[0:7]),  state65[7:29],  FK9(state65[29:36]),  state65[36:58] ]
action_abs[62] = [ FK9(action65[0:7]), action65[7:29], FK9(action65[29:36]), action65[36:58] ]

delta9(A,B)    = [ A_R^T (B_t − A_t), (A_R^T B_R)[:,0], (A_R^T B_R)[:,1] ]

action[k, :]   = [ delta9(FK4(state65[i,0:7]),   FK4(action65[f,0:7])),   action65[f, 7:29],
                   delta9(FK4(state65[i,29:36]), FK4(action65[f,29:36])), action65[f, 36:58] ]
                 f = min(i + k, N − 1),  k = 0..15
                 base pose is frame i for ALL k — chunk-base, not frame-to-frame

tactile_f6[10,6]   = observation.tactile[i].reshape(10, 6)
tracking_error[56] = compute_bimanual_tracking_error(state[t], action_abs[t−1])
```

> **`NEW_DESIGN.md` says the deltas are "the change in position between sequential frames".
> That is wrong.** `gen_json_tac_deltabase_eef_bimanual_parallel.py:228-243` (mirrored in
> `lerobot_common.py::build_action_chunk`) uses **one chunk-base pose at frame `i` for all 16
> steps**, with targets from the **commanded `action` joints**, not the next `state`.
> `eval_trex_async.py:992` confirms it at deploy — the anchor is captured once per chunk.
> Implement the code, not the doc.

---

## 7. Training

### 7.1 `origami/trex_patch.py`

Two monkeypatches, needed only because we enable gradient checkpointing on one GPU; both are
no-ops otherwise, each with a unit test proving equivalence. `apply()` must be idempotent and
assert the pinned upstream source hashes before patching.

1. **`Qwen3VLAttentionMoT` cached-prefix read.** Upstream always calls
   `past_key_value.update(...)`, which **appends**. `update()` is not idempotent, so under
   gradient checkpointing the backward recompute of the cascaded tactile step appends the K/V a
   second time and the KV length outgrows the causal mask built for `past_len + seq_len`.
   Patch: when `torch.is_grad_enabled()`, read the stored prefix without appending and concat:
   ```python
   def read_cached_kv(cache, layer_idx):
       layers = getattr(cache, "layers", None)
       if isinstance(layers, list):
           if layer_idx < len(layers):
               l = layers[layer_idx]; k = getattr(l, "keys", None)
               if k is not None and k.numel() > 0: return k, l.values
           return None
       k = getattr(cache, "key_cache", None)
       if k is not None and layer_idx < len(k) and k[layer_idx] is not None:
           return k[layer_idx], cache.value_cache[layer_idx]
       return None
   ```
   Nothing downstream consumes the appended entries during training, so the read-only concat is
   equivalent and safe to replay.
2. **Assert `attention_mask` stays off cached-prefix decoder calls.** Upstream already omits it
   in the `past_kv is not None` branches of `forward_flow_action_{full,partial}` (only
   `modeling_vla.py:613` and `:706`, the `past_kv is None` branches, pass it). Verify at import;
   patch it out if a future upstream adds it back — a `[B, L_slow]` mask against an action-only
   `inputs_embeds` is a length mismatch.

### 7.2 `origami/train_origami.py`

A **vendored copy of `T-Rex@full-pipeline:scripts/midtrain.py`**, not `main:scripts/train.py`
(§7.5 explains why). `midtrain.py` is a strict **superset**: identical model, identical
cascaded-flow loss, identical `--data_format lerobot → TRexLeRobotDataset` dispatch (`:1457`),
plus `--tactile_delay_offsets`, `--frame_stride`, `--num_workers`, `--n_fingers`,
`--tactile_code_per_finger`, `--tactile_f6_stats_ckpt`, `--vqvae_codes_h5_name`. Comparing the
two argument lists shows **zero** flags in `train.py` absent from `midtrain.py`.

Deltas, each marked `# ORIGAMI-DELTA:` in the source, and nothing else:

1. `import origami.trex_patch as _p; _p.apply()` at the top.
2. `_world_size()` helper replacing bare `dist.get_world_size()` (`midtrain.py:1106`, `:1312`)
   so a non-distributed launch works; guard `dist.all_reduce` with `dist.is_initialized()`;
   guard the `accelerator.state.deepspeed_plugin` writes with a `None` check.
3. New flags: `--save_steps`, `--max_steps`, `--save_optimizer_state`, `--resume_full_state`,
   `--gradient_checkpointing`, `--freeze_latent_expert`, `--train_latent_last_n`,
   `--optim {adamw,adamw8bit}`, `--action_loss_weight` (§7.5-C).
   (`--num_workers` and `--frame_stride` already exist upstream.)
4. `save_checkpoint` also writes `training_state.json`
   (`epoch, global_step, learning_rate, warmup_rates, min_lr_ratio`) and `state/` via
   `accelerator.save_state`, and adds `instruction`, `image_size` (+ per-view), `vqvae_window`,
   `locked_config`, `tactile_delay_offsets`, `origami_prep` to `training_args.json`.
   **`locked_config` in the checkpoint is what `serve_zenoh.py` loads** (G1c).
5. `--lerobot_val_root`: a second `TRexLeRobotDataset` on the held-out-season root instead of
   `create_val_split`'s within-root episode split (a random episode split leaks a session).
6. `train_flare = use_flare and flare_loss_weight > 0`, so `--flare_loss_weight 0` keeps the K
   latent query tokens (sequence-shape parity with the resumed checkpoint) while skipping the
   extra no-grad ViT pass. Needed when the latent expert is frozen.
7. `--max_steps` truncates `num_training_steps` **before** the LR schedule is built (otherwise
   cosine is computed for the full epoch count and barely decays).
8. `--tactile_delay_scope` plumbing into the §7.5-B loader subclass (the LeRobot path ignores
   `--tactile_delay_offsets` as shipped).

`origami/tests/test_train_vendor.py` diffs the vendored file against
`git -C T-Rex show origin/full-pipeline:scripts/midtrain.py` and asserts every changed hunk
carries an `# ORIGAMI-DELTA:` marker.

### 7.3 `origami/train_origami.sh`

Effective batch held at upstream's **128**.

```bash
IMAGE_SIZE="384 384"          # from the §4.5-A probe

accelerate launch --config_file T-Rex/config/sft_qwen.yaml --num_processes 1 \
  origami/train_origami.py \
  --model_path ${ORIGIN_MODEL_PATH} \
  --data_format lerobot --lerobot_root ${LEROBOT_ROOT} \
  --lerobot_val_root ${LEROBOT_VAL_ROOT} \
  --action_dim 62 --action_chunk 16 --image_size ${IMAGE_SIZE} \
  --use_robot_state 1 \
  --use_tactile_vec 1 --use_tactile_deform 1 --use_tactile_vqvae 1 \
  --deform_encoder_ckpt ${DEFORM_ENCODER_PATH} \
  --tactile_intermediate_size 1536 --training_stage 2 --tactile_loss_weight 1.0 \
  --cascaded_total_steps 10 --cascaded_split_step 6 \
  --cascaded_tactile_dropout 0.1 --cascaded_loss_weight 1.0 \
  --tactile_delay_offsets 0 4 8 12 --tactile_delay_scope f6 \
  --use_flare 1 --n_flare_tokens_per_frame 4 --n_flare_steps 8 \
  --flare_loss_weight 0.5 --flare_frame_stride 4 --flare_layer_index -1 \
  --resume_checkpoint ${RESUME_CHECKPOINT} --resume_source midtrain \
  --learning_rate 1e-4 --min_lr_ratio 0 --weight_decay 0.01 --max_grad_norm 1.0 \
  --warmup_rates 0.03 \
  --train_bsz_per_gpu 2 --gradient_accumulation_steps 64 \
  --gradient_checkpointing 1 --optim adamw8bit --num_workers 8 \
  --n_epochs 1 --max_steps 40000 --save_steps 2000 --save_optimizer_state 1 \
  --val_freq 1000 --max_val_batches 30 --seed 42
```

Deviations from `scripts/train.sh`:

| flag | upstream | here | why |
|---|---|---|---|
| `image_size` | `384 288` | square, probed (§4.5-A) | square source; token count matched to upstream |
| `use_robot_state` | `0` | **`1`** | fingers are self-occluded in the head view, and the eef-9d+22 layout makes `add_tracking_error_noise` (axis-angle on `state[3:9]`/`[34:40]`) **valid** — the main representational win of eef-62. Keep `0` as the ablation arm. |
| bsz × GPUs | 16 × 8 | 2 × 1 | 40 GB budget |
| `gradient_accumulation_steps` | 1 | 64 | keeps effective batch 128 |
| `gradient_checkpointing` | — | 1 | needed at 40 GB; requires §7.1 patch 1 |
| `optim` | adamw | adamw8bit | bf16 params make torch AdamW hold bf16 moments |
| `weight_decay` / `warmup_rates` | 0 / 0 | **0.01 / 0.03** | midtrain values (§7.5-A) — our run is midtrain-shaped |
| `n_epochs` / `max_steps` | 100 / — | 1 / 40 000 | see calibration below |
| `save_steps` | — | 2 000 | epoch-only checkpointing would never fire |
| val split | `val_ratio 0.05` (random) | held-out seasons | a random split leaks a session |

**Step-budget calibration.** T-Rex's released midtrain checkpoint is `checkpoint-5-19464`
(epoch 6, `global_step` 19 464). `global_step` increments once per optimizer step and midtrain
ran 16 × 8 GPUs × grad_accum 1 = 128 samples/step ⇒ **≈ 2.5 M samples for their entire midtrain
stage**. Our 40 000 × 128 = **5.1 M samples, about 2× T-Rex's whole midtrain run** — generous,
not truncated. The binding constraint is wall-clock and §7.4 decode throughput, not epochs.

| target | steps @ 128 | fraction of 1 epoch (9.5 M) |
|---|---|---|
| pilot | 2 000 | 0.03 |
| **first full run** | **40 000** | **0.54** |
| one epoch | 74 000 | 1.00 |
| two epochs | 148 000 | 2.00 |

If 2×64 still OOMs: `--freeze_latent_expert 1 --train_latent_last_n 8` with
`--flare_loss_weight 0` (the flare loss would only train `flare_proj` once the latent expert is
frozen).

**Resume expectation.** With `action_dim=62` matching the midtrain checkpoint, `x_embedder`,
`final_layer`, `final_layer_tactile` and `state_embedder` all load cleanly. Confirm
`missing == 0` and no shape-mismatch skips in the resume log (G9b).

### 7.4 Data-loading throughput

`TRexLeRobotDataset` does **13 random-access video seeks per sample** (head with 9
`delta_timestamps` offsets, 2 wrists, 10 deform) — decode is the bottleneck, not the GPU.
1. `--num_workers 8`, `prefetch_factor 4`, `persistent_workers`.
2. Keep `--flare_frame_stride 4` with `n_flare_steps 8`: the 9 head offsets come from the
   **same** video file, so the seeks are near-local.
3. If still < 4 samples/s/worker, add a strided block-shuffled **Sampler** so consecutive
   samples in a worker hit the same GOP. Do **not** subsample at conversion time — the F6
   window and the chunk both need 30 Hz neighbours.
4. Measure first: `bench_latency.py --mode dataloader`.

### 7.5 Why midtrain, and the vision-shift question

**T-Rex never trains the vision tower.** `visual*` is `requires_grad = False` in all three
stages (`pretrain.py:946`, `midtrain.py:1427`, `train.py:860`). What T-Rex calls the *latent
expert* is the MoT's vision-language copy of the decoder layers, and that **is already fully
trainable in post-train**. So an origami midtrain stage **unlocks no parameter post-train does
not already train** — its value is in the objective and schedule.

**A. What midtrain does differently.** Everything structural is identical (`action_dim 62`,
`action_chunk 16`, cascaded 10/6/0.1/1.0, flare 4/8/0.5/stride 4/layer −1,
`tactile_intermediate_size 1536`, lr 1e-4, bsz 16×8). Differences:

| | midtrain | post-train |
|---|---|---|
| **`--tactile_delay_offsets`** | **`0 4 8 12`** | absent (delay ≡ 0) |
| `weight_decay` / `warmup_rates` | 0.01 / 0.03 | 0 / 0 |
| `n_epochs` / `val_ratio` | 10 / 0.02 | 100 / 0.05 |
| data | merged multi-source root | one task |

≈ 109 h of origami is over 2× T-Rex's own midtrain corpus, so the midtrain hyperparameters are
the better-matched ones; upstream's "100 epochs over a few hundred episodes" does not transfer.

**B. The tactile delay curriculum — the ingredient worth importing.**
`midtrain.py:769` samples `delay_k ~ U{0,4,8,12}` per item and feeds the tactile expert
observations from frame `t + delay_k` while the action chunk stays anchored at `t`; the
docstring says *"simulates the inference pattern: slow chunk anchored at t=0, tactile fast tick
fired at t+k."* **Those are literally our deploy offsets** (§9.1 fires at in-chunk `{0,4,8,12}`;
`hardware_code/config/default.yaml` sets `refine_offsets [4,8,12]`). Post-training with
`delay_k ≡ 0` trains the tactile expert on a condition it never sees at deploy.

The LeRobot path does **not** implement the delay (`lerobot_dataset.py:301` hard-wires
`tactile_f6s_delayed = norm_tacf6`), so `origami/delayed_lerobot_dataset.py` must:
* **F6 (free).** Extend `_f6_offsets` from `[-(W-1)/fps … 0]` to `[-(W-1)/fps … +max(delay)/fps]`
  (16 + 12 = 28 offsets). `observation.tactile_f6` is a **numeric parquet feature, not a video**,
  so extra `delta_timestamps` cost a parquet read. Then `tactile_f6s_delayed = window[:, W-1+k]`
  and `tactile_f6_history = window[:, k : k+W]` — the VQ-VAE window must also **end at the
  delayed frame**, as `midtrain.py` does.
* **Deform (expensive).** Needs `[0, k/fps]` on all 10 deform keys → **20 video seeks per sample
  instead of 10**, on top of §7.4's bottleneck.
* `--tactile_delay_scope {none, f6, both}`, default **`f6`**: delay the F6/VQ-VAE channel (where
  the temporal signal lives, at zero cost) and keep deform at the anchor frame. Measure `both`
  on §8.1-D before paying 2× decode. Gate **G17** asserts trained offsets == deployed offsets.

**C. A vision-only adaptation stage — what it can and cannot reach.**
* **Can: FLARE-only warmup (Phase 0).** FLARE is self-supervised — the latent expert's K query
  tokens predict the frozen ViT's features of head frames `+{1..8}·stride` ahead, cosine loss
  (`train.py:1119`). **No action labels needed.** A short stage at `--flare_loss_weight 1.0
  --cascaded_loss_weight 0 --action_loss_weight 0.1` lets the latent expert absorb origami's
  visual statistics before the action loss pulls on it. Needs the `--action_loss_weight` flag
  (upstream hardcodes `loss = loss_act + w_tac·loss_tac + w_flare·loss_flare`). Cheap; 1–2 k
  steps; **ablation.**
* **Cannot: adapt the ViT.** The visual shift lands on frozen weights. FLARE only teaches the
  latent expert to *predict* origami-ViT features; the features do not move.
* **Unfreezing the ViT is a deliberate deviation** with three coupled consequences: (1) FLARE
  targets come from the same `self.visual` under `no_grad` (`train.py:1143`), so a training ViT
  makes the target chase the predictor and the cosine loss admits a trivial solution — disable
  FLARE or hold a frozen target copy (+1.3 GB, second forward); (2) activation memory for 3
  images with grad on top of gradient checkpointing likely forces bsz 1; (3) forgetting risk on
  a downstream MoT tuned against those features. **Park it** unless `diagnose_shift.py`'s
  ViT-feature statistics show the origami features are genuinely degenerate.
* **Rejected: ViT→MoT adapters** — changes the graph and breaks checkpoint compatibility.

**D. Co-training with T-Rex's own dataset.** `zekaiwang/trex_dataset` (~50 h, LeRobot v3.0, same
hands and tactile) could be converted to eef-62 with *their* `dexmate-urdf` FK and merged.
Honest framing: an **anti-forgetting regularizer, not a shift mitigation** — it teaches nothing
about origami's cameras, and costs a second kinematics model plus double prep. **Defer.**

**E. Staging.**

| phase | steps | config | purpose |
|---|---|---|---|
| **0** (ablation) | 1–2 k | `--flare_loss_weight 1.0 --cascaded_loss_weight 0 --action_loss_weight 0.1` | latent expert absorbs origami visual statistics |
| **A** (ablation) | 2 k | `--freeze_latent_expert 1 --train_latent_last_n 0 --flare_loss_weight 0` | action/tactile heads settle onto origami action statistics |
| **B** (required) | 40 k | full §7.3 recipe | the real run |

Phases chain via `--resume_checkpoint` + `--resume_full_state`. 0 and A are ~5% of B each —
cheap to try, cheap to skip; justify them against the §11.9 zero-shot floor.

---

## 8. Evaluation

### 8.1 `origami/eval_offline.py`

Loads a checkpoint through `test.py::model_load` (import it) and a `TRexLeRobotDataset` on the
val root. `--zero-shot` points it at the **untrained midtrain checkpoint** — at `action_dim=62`
every head loads, so this measures the pretrain→origami shift end-to-end and sets the floor
every trained checkpoint must beat. Always report against two trivial baselines:
**hold-position** (repeat `observation/state`, what the SDK's placeholder policy does) and
**`--disable_tactile`**.

* **A. Action-space accuracy**, per chunk step `k = 0..15`, per block
  (`L_trans3 / L_rot6d6 / L_hand22 / R_trans3 / R_rot6d6 / R_hand22`): MSE, MAE, and
  **normalized variance share** `Var(pred)/Var(gt)` — the §11.1 diagnostic.
* **B. EEF-space error.** Reconstruct the target SE3 from the predicted delta9: translation
  error in **mm**, rotation in **degrees** (`Rodrigues` angle of `R_pred^T R_gt`), per k.
  Scale reference: ground-truth per-frame step is 0.39 mm mean / 2.04 mm p99.
* **C. Joint-space error — the metric that matters.** Run the full deploy retarget
  (`retarget.py`) on the predicted chunk and compare to the dataset's `action65`. Per-group RMS
  in rad, plus the fraction of steps where IK failed or clipped. **This is the competition's
  actual objective.**
* **D. Tactile-expert contribution.** Three configs on the same batches: cascaded (deployed
  path); `--disable_tactile 1` (`forward_flow_action_full`, action expert alone); cascaded with
  tactile zeroed. Report deltas split by contact / no-contact (`|F| > 1 N` on any finger).
* **E. Rollout drift.** 10 held-out episodes, *anchored* (anchor from ground-truth state) and
  *closed-loop* (integrate the IK output and re-anchor). Plot EEF trajectory vs ground truth;
  report terminal drift.
* **F. Smoothness.** Per-step joint velocity/acceleration vs the demonstrations; flag steps
  exceeding the URDF velocity limit (2.6179 rad/s) or the §9.3 rate limit.

Outputs `metrics.json`, per-block CSVs, matplotlib PNGs (one figure per file).

### 8.2 `origami/eval_shadow.py`

Drives the SDK's `participant_local_evaluator` in Shadow mode against a locally running
`serve_zenoh.py`, using the SDK's bundled `season_POC22061_2026_07_09_16_23_46_train`. Asserts
metadata validation passes, every reply is finite `float32[T,65]`, and the URDF
limit/jump/velocity checks report no violations. Also runs the SDK's
`examples/check_zenoh_policy.py`.

### 8.3 `origami/bench_latency.py`

Modes `slow | fast | slow_and_fast | dataloader`. Reports p50/p95/p99 and the **maximum
feasible `action_horizon`** given `slow_p99` at 30 Hz (`T_min = ceil(slow_p99_s × 30)` in sync
mode). Must run before choosing `--action_horizon` (§9.1, G13).

### 8.4 Notebook

`origami/colab.ipynb`: install cell (uv, pinned deps, lerobot, pin/pink/daqp), asset+checkpoint
download, a pre-flight cell running G1–G3 and G7, smoke train, the real run with a Drive-backed
`--output_dir`, then `eval_offline.py`. Every cell shells out to `origami/*` — no notebook-only
logic.

---

## 9. Deploy

### 9.1 Cadence

The wire gives one `infer` per call returning a fixed `[T, 65]`. To reproduce T-Rex's
slow-1.875 Hz / fast-7.5 Hz protocol:

* **`action_horizon T = 4`** — the gateway consumes 4 steps (0.133 s at 30 Hz), then calls again.
* Call counter: `n % 4 == 0` → **slow + fast** (re-encode vision, cache KV at τ_split, one
  tactile continue); otherwise → **fast only** (cached KV + fresh tactile).
* The 16-step chunk is refreshed at in-chunk offsets `{0, 4, 8, 12}` — identical to
  `eval_trex_async.py`'s `refine_offsets [4,8,12]` plus the chunk-start tick.
* Return rows `[4·(n mod 4) : 4·(n mod 4) + 4]` of the aggregated chunk.
* Anchor: arm joints from `observation/state` **at the slow tick**, held for all 16 rows.
* ACT temporal aggregation via `aggregate_chunks(buffer, global_step, k=temporal_agg_k)`,
  default `temporal_agg_k = 0.0` (uniform), matching `hardware_code/config/default.yaml`.

`T = 4` is **contingent on measured latency** (§8.3). If `slow_p99 > 130 ms` in sync mode, fall
back to `T = 8` (slow every 2 calls, offsets `{0,8}`) or `T = 16` (slow every call, no fast
ticks). `--action_horizon` and `--slow_every` are flags; `slow_every × T == 16` is asserted.

### 9.2 `origami/retarget.py`

```python
class Retargeter:
    def __init__(self, kin: OrigamiKinematics, cfg)
    def set_anchor(self, state65):     # base_l/base_r = kin.fk_matrices(state65[0:7], state65[29:36])
                                       # warm_l/warm_r = those joints;  prev_cmd = state65
    def step(self, action62, motor7) -> np.ndarray   # (65,)
```
`step` (mirrors `eval_trex_async.py:1113`):
```python
dpos_l, drot_l, hand_l = action62[0:3],   action62[3:9],   action62[9:31]
dpos_r, drot_r, hand_r = action62[31:34], action62[34:40], action62[40:62]
t_l = base_l[:3,3] + base_l[:3,:3] @ dpos_l
R_l = base_l[:3,:3] @ rot6d_to_matrix(drot_l)          # Gram-Schmidt
# … same for right …
q_l, q_r = kin.solve_ik(pin.SE3(R_l, t_l), pin.SE3(R_r, t_r), warm_l, warm_r)
warm_l, warm_r = q_l, q_r                              # warm-start the next row
return _safety(np.concatenate([q_l, hand_l, q_r, hand_r, motor7]).astype(np.float32))
```
Use `eval_trex_async.py::rot6d_to_matrix` (Gram–Schmidt), **not**
`lerobot_common.py::get_rot_mat` (raw `column_stack` + cross, no renormalisation). They agree on
clean data, but the policy's output is only approximately orthonormal.

### 9.3 `_safety` — mandatory, in this order

1. **Finite check** — NaN/Inf → return `prev_cmd`, increment a counter.
2. **Joint limits** — clip arms/hands/motor to the URDF ranges (§1.3).
3. **Rate limit** — `|cmd − prev_cmd| ≤ max_joint_vel / 30`, default `max_joint_vel = 0.3 rad/s`
   (matches `arm_hand_control.py`'s `max_edge_joint_step = 0.3 / command_hz`); clip toward
   `prev_cmd`.
4. **Motor block held** — `cmd[58:65] = observation/state[58:65]` of the **current** call; never
   stale, never predicted (D5).
5. **Optional collision gate** (`--check-collisions`, off by default, ~1 ms/row) → `prev_cmd`.
6. `prev_cmd = cmd`; return float32.

Log per-episode clip/limit/NaN/IK-failure counts. **A nonzero NaN or IK-failure count in Shadow
is blocking.**

### 9.4 `origami/serve_zenoh.py`

Start from `sharpa_north_ces_lite_sdk-main/examples/policy_server_template.py` and keep
`OrigamiZenohServer` + the msgpack codec **byte-identical** — only `TeamPolicy` changes.

```python
class TeamPolicy:
    def __init__(self, action_horizon, checkpoint_path, urdf_path, **cfg):
        # 1. test.py::model_load(args) -> model, processor, statistic (q01/q99 + masks)
        # 2. training_args.json -> instruction, image_size (+per-view), cascaded_{total,split}_step,
        #    vqvae_window, tactile_delay_offsets, locked_config, assert action_dim==62, chunk==16
        # 3. OrigamiKinematics(urdf_path, LockedConfig.from_dict(ta["locked_config"]))  → G1c
        # 4. CascadedServer(args, model, processor, statistic)
        # 5. Retargeter(kin, cfg)
        # 6. warm-up: one slow_and_fast on zeros; assert chunk shape (16, 62)
    def reset(self):   # clear cached_kv/x_split/f6_buffer, chunk buffer, call counter,
                       # Retargeter anchor/warm-start/prev_cmd
    def infer(self, obs) -> np.ndarray:   # float32 (T, 65)
```
`infer`:
```python
state65 = np.asarray(obs["observation/state"], np.float32)
f6      = np.asarray(obs["observation/tactile"], np.float32).reshape(10, 6)
deform  = split_deform_strip(obs["observation/image/tactile_deform"][:, :, 0]) / 255.0
self.f6_hist.append(f6)                    # deque(maxlen=vqvae_window), left-padded on reset

slot = self.calls % self.slow_every
if slot == 0:
    imgs = {k: pil(obs[f"observation/image/{k}"]).resize(size_for(k), LANCZOS)
            for k in ("head_left", "wrist_right", "wrist_left")}
    chunk = self.policy.slow_and_fast(imgs, self.instruction, np.stack(self.f6_hist),
                                      deform, self.state62(state65))    # (16,62) denormalized
    self.chunk_buf = [(0, chunk)]
    self.retarget.set_anchor(state65)
else:
    chunk = self.policy.fast(np.stack(self.f6_hist), deform)
    self.chunk_buf.append((0, chunk))

rows = [self.retarget.step(aggregate_chunks(self.chunk_buf, slot * self.T + j,
                                            k=self.temporal_agg_k), state65[58:65])
        for j in range(self.T)]
self.calls += 1
return np.ascontiguousarray(np.stack(rows), dtype=np.float32)
```
* Images must go through the **same 224→`image_size` LANCZOS resize as training** (§4.3).
* `origami/policy.py` wraps `CascadedServer` and calls `_run_slow`/`_run_fast` **directly with
  PIL objects**, avoiding the ~5 ms JPEG encode+decode the ZMQ path imposes. Keep the ZMQ path
  only for parity tests.
* `state62(state65)` = `[FK9(state65[0:7]), state65[7:29], FK9(state65[29:36]), state65[36:58]]`
  from the same `OrigamiKinematics`. Pass it **raw** — `_run_slow` normalizes it with
  `statistic["state_*"]`.
* `--disable_tactile 1` is available as an ablation and as a fallback if the organizer
  zero-fills the tactile stream.

### 9.5 `origami/docker/Dockerfile`

Multi-stage, CUDA runtime base. Contains torch 2.6.0 cu124, transformers 4.57.3,
`pin`/`pink`/`qpsolvers[daqp]`, `eclipse-zenoh==1.9.*`, `msgpack`, the checkpoint
(`model.pt`, `processor/`, `config.json`, `training_args.json`, `stats_data.json`), the URDF +
meshes, and `origami/`. Entrypoint `python -m origami.serve_zenoh` with no required args.
`ENV EXECUTION_MODE=async`. No secrets, no dataset, no `.git`.

---

## 10. Verification gates

Every gate is a test in `origami/tests/` or a `verify.py` subcommand. **Do not proceed past a
failing gate; do not weaken a threshold.**

| id | gate | threshold |
|---|---|---|
| **G0** | Upstream drift: sha256 of every T-Rex file we import or vendor matches the pinned refs (`f88e10c`, `b23eafe`) | exact |
| **G1a** | URDF joint set == the 65 mapped names in dataset-index order | exact |
| **G1b** | Reduced model `nq == nv == 14`; names == `{left,right}_arm_joint_1..7` | exact |
| **G1c** | `LockedConfig.digest()` in dataset meta == checkpoint == server | exact |
| **G2** | FK/IK round-trip on 10 000 dataset frames, `solve_ik(fk(q), warm=q+N(0,0.02))` | max \|Δq\| < 1e-3 rad; EEF pos < 0.1 mm; rot < 0.01° |
| **G2b** | Same from a perturbed warm start `+N(0,0.1)` | ≥ 99.5% meet G2; log failures |
| **G3** | `delta9` → `rot6d_to_matrix` → reconstruct target pose | max err < 1e-9 |
| **G3b** | `matrix_to_rot6d`/`rot6d_to_matrix` round-trip on 10 000 random SO(3) | < 1e-12 |
| **G4** | Chunk parity: converter output vs a direct `build_action_chunk` call, one full season | bitwise identical for `action`, `state`, `action_abs`, `tactile_f6` |
| **G4b** | `--action-layout abs` derived chunk vs baked chunk | max diff < 1e-6 |
| **G5** | Deform tile mapping: 2 000 frames with exactly one finger at \|F\| > 5 N → that tile has max mean intensity | ≥ 95% |
| **G6** | Deform video round-trip: decoded channel-0 vs source luma | exact (lossless) or PSNR > 45 dB |
| **G7** | Loader parity: `collate_fn` returns all 21 keys with §1.4 shapes/dtypes; `tactile_deforms [B,10,1,240,240]`; `tactile_f6_history [B,16,10,6]`; `norm_actions ∈ [−1,1]` on masked dims | exact shapes; ≥ 99.9% in range |
| **G7b** | Forward pass of `Qwen3VLVLAModel` on that batch (CPU, tiny stub config) | finite `loss_act`, `loss_tac` |
| **G8** | Reservoir q01/q99 vs exact, one season | max relative error < 1% per non-degenerate dim |
| **G8b** | Tactile mask is **not** all-`True`; masked set includes L-ring, L-pinky, R-ring, R-pinky | assert |
| **G9a** | Smoke train: 5 steps bsz 1, checkpoint written, `--resume_full_state` resumes at the same step | finite losses |
| **G9b** | Resume from the midtrain checkpoint | `missing == 0` **and** zero shape-mismatch skips |
| **G10** | `serve_zenoh` passes the SDK's `check_zenoh_policy.py` + metadata validation | pass |
| **G11** | Shadow replay on the SDK's bundled season | zero NaN / IK-failure / limit / jump / velocity violations |
| **G12** | Prep↔deploy parity: 62-D state and image tensors from a dataset frame == those from `serve_zenoh` on the same frame at 224 | state diff < 1e-5; pixels exact |
| **G13** | Latency: measured `slow_p99` admits the chosen `action_horizon` | `slow_p99 < T/30 s` (sync) |
| **G14** | Instruction identity across `origami_prep.json` / `training_args.json` / the string fed to `task_description`; `input_ids` from a dataset sample == from the deploy path | exact string; `input_ids` bitwise equal |
| **G15** | Frozen VQ-VAE codebook health on ≥ 200 k origami F6 windows (§11.8) | ≥ 25% of codebook used; normalized entropy ≥ 0.5; clamp-saturation ≤ 5% |
| **G16** | Token budget: probed `patch_size × merge_size`, chosen `--image_size*`, resolved per-view grid recorded in `training_args.json` and reproduced by `serve_zenoh` | tokens within 1 grid cell of upstream's `384 288`; grids equal |
| **G17** | Delay-curriculum parity: `training_args.json["tactile_delay_offsets"]` == the offsets `serve_zenoh` fires; delayed-F6 tensor at `delay_k` == `f6_window[:, W-1+k]` | exact |
| **G18** | Season-set integrity (§1.1a) | exact |

Tests are **CPU-only and network-free**, using the fixture season and a tiny stub Qwen config
for G7b.

---

## 11. Risks

Ranked by expected impact: **11.1 > 11.8 > 11.9 > 11.2 > the rest.** (Numbering reflects
drafting order, not priority.)

### 11.1 The arm barely moves — eef-62 may be worse than 65-D joints

The EEF translates **0.39 mm/frame mean, 2.04 mm p99** (~6 mm per 16-frame chunk) while the
22-D finger blocks sweep 0–1.5 rad. So 9 of every 31 dims carry little variance and, after
per-dim q01/q99 normalization, their FK quantization noise is amplified along with the signal.

* Per-(step,dim) q01/q99 (`[16,62]` stats) already gives each chunk step its own scale.
* §8.1-A reports **normalized variance share**; noise-dominated arm blocks show
  `Var(pred)/Var(gt) ≪ 1` at small k.
* The 65-D joint pipeline stays available as a comparison arm; §8.1-C's joint-space RMS is
  directly comparable.
* If noise-dominated, the in-representation fix is a longer effective horizon: raise
  `FRAME_STRIDE` to 2–3 at conversion (chunk spans 32–48 frames, 1.1–1.6 s) so deltas grow
  relative to the noise floor. `build_action_chunk` already honours it. **Deploy must then
  execute one chunk row every `FRAME_STRIDE` robot steps**, changing §9.1's arithmetic — gate it
  behind `--frame-stride` and assert consistency at load.

**Run §8.1-A on the 2 000-step pilot before committing to the 40 k run.**

### 11.2 Deploy latency

A 2 B Qwen3-VL + ViT slow tick, 6 Euler steps on the action expert, 4 on the tactile expert.
`T = 4` gives a 133 ms sync-mode budget. Mitigations: `execution_mode=async` (default),
bf16 + SDPA, the §9.1 horizon ladder, and G13 before submission.

### 11.3 `tactile_raw` cannot feed the deform head

`deform_proj` is `ActionEmbedder(28800)` = 128·15·15, i.e. **240×240 only**; raw cells are
240×**320** → 128·15·20 = 38 400. Options if wanted later: centre-crop to 240×240 (loses 25% of
the width), anisotropic resize, or a new `raw_proj` head (not covered by the midtrain
checkpoint). Ship `--include-tactile-raw` storing 10 extra `(3,240,320)` videos under
`observation.tactile_raw.{l,r}{0..4}` — **not** in `build_trex_features`, **not** read by the
loader — default **off** (it triples the download and nothing consumes it).

### 11.4 LeRobot API drift

`LeRobotDataset.create/add_frame/save_episode/finalize` signatures, whether
non-`observation.images.*` keys are accepted as `dtype: "video"` (the 10 deform keys are
`observation.tactile_deform.*`), and whether `aggregate_datasets` exists all vary by version.
**Step 2 is `origami/tests/test_lerobot_probe.py`**: create a 3-frame dataset with the full
`build_trex_features` schema, reopen with `delta_timestamps`, assert every key round-trips. Pin
the lerobot commit in `pyproject.toml` and record it in `meta/origami_prep.json`. If the
`observation.images.` prefix turns out to be required, **report before renaming** — renaming
breaks `DEFORM_KEYS` parity and forces a loader override.

### 11.5 Storage / bandwidth

≈ 107 GB output + 126–390 GB streamed against 23 GB free on the dev laptop. `prepare.py`
`statvfs`-checks and refuses to start (§5.6). Prep runs on the large server.

### 11.6 Hand state↔action transient

`state[7] = 0.100` vs `action[7] = 1.139` rad at episode start. Affects only `tracking_error`
(used by `add_tracking_error_noise` when `use_robot_state=1`). Mitigated by skipping the first
30 frames per episode (§5.4). If `te_std` on the hand block still exceeds 0.1 rad, run the first
pilot with `--use_robot_state 0`.

### 11.7 Gradient checkpointing × cascaded KV cache

Non-idempotent `cache.update()` under checkpoint recompute (§7.1 patch 1). Unit test: run the
cascaded tactile step twice with `torch.is_grad_enabled()` true and assert the KV length is
unchanged.

### 11.8 Frozen VQ-VAE codebook collapse — high priority, cheap to check and fix

`modeling_vla.py::encode_tactile_f6_history` (`:354-361`) min-max normalizes the raw F6 window
with **T-Rex's own** `tacf6_vqvae_{min,max,mask}` buffers and **hard-clamps to [−1,1]**:
```python
denom  = (self.tacf6_vqvae_max - self.tacf6_vqvae_min) + 1e-8
normed = torch.clamp(2.0 * (flat - self.tacf6_vqvae_min) / denom - 1.0, -1.0, 1.0)
normed = torch.where(self.tacf6_vqvae_mask, normed, flat)
```
Origami's forces are unlike a pick-and-place corpus: **p99 |F| ≈ 42 N on the right index, ≈ 32 N
on the right thumb, while 4 of 10 fingers are dead** (§1.2). If T-Rex's range is narrower the
window saturates at ±1 (square wave → few codes); if wider, origami forces occupy a sliver near
−1 (near-constant → one code). Either way the tactile expert gets a near-constant token, the
cascade degrades to the action-expert baseline, and **nothing raises an error** — the most
likely way for this project to "work" with zero tactile contribution.

**Upstream knows this failure mode:** `midtrain.py:323` warns *"using default [-1, 1] which WILL
saturate real F6 values"* and exposes `--tactile_f6_stats_ckpt` to avoid it.

**Two normalizations, not one.** The per-frame F6 vector into `tacf6_embedder` is normalized by
the **loader** from `meta/trex_norm_stats.json` → origami q01/q99, correct by construction. The
**VQ-VAE history window** is normalized *inside the model* by buffers loaded from the T-Rex
checkpoint (`modeling_vla.py:323-325`) → wrong scale. **Fixing the first does not fix the second.**

Diagnosis (**G15**): run `encode_tactile_f6_history` over ≥ 200 k origami windows; report the
codebook usage histogram, number of codes used, normalized entropy `H / log(K)`, and the
clamp-saturation fraction.

Fix ladder:
1. **Re-fit the buffers only.** `tacf6_vqvae_min/max/mask` are plain registered buffers
   (`modeling_vla.py:139-144`). Overwrite with origami q01/q99 + the degenerate mask from
   `meta/trex_norm_stats.json`; encoder weights and codebook untouched. `refit_vqvae_stats.py`
   writes a new checkpoint and records `vqvae_stats_source: "origami"`.
2. **Retrain the VQ-VAE on origami F6** — `tactile_vqvae/train.py` + `config/vqvae_f6.yaml`
   (window 16, `in_channels 30`), then `utils/merge_vqvae_into_ckpt.py` bakes it in and sets
   `use_tactile_vqvae=1` + `vqvae_config` for auto-detect. Consider `granularity: finger` so the
   6 live fingers get their own codes instead of being averaged with 4 dead ones.
3. Fall back to `--use_tactile_vqvae 0 --use_tactile_vec 1 --use_tactile_deform 1` and treat
   temporal tactile as future work. Measure against §8.1-D.

**The deform encoder is not at risk** — `sharpa_wave_deform_encoder.pth` was trained on the same
Sharpa Wave sensor with the same 240×240 cells; it transfers as-is.

### 11.9 Pretrain → origami distribution shift

Two shifts needing **opposite** treatments.

**(a) train ↔ deploy: ≈ zero.** Same rig, cameras, organizer squash, hands and tactile. No
sim2real gap. **Domain randomization is the wrong tool** — it would blur the one distribution we
know exactly, and G12/G14 exist to keep the paths identical. The genuine residual is
*cross-season* variation plus the organizer's eval-day rig, which the 101/25 held-out-**season**
split measures. Justified augmentation is **light photometric only** (brightness/contrast/gamma
±10%, no geometric transforms), behind `--photometric_aug 0.0` (default off), adopted only if
held-out-season val improves.

**(b) T-Rex midtrain → origami: large.**

| axis | T-Rex midtrain | origami |
|---|---|---|
| head camera | rectilinear, tight 400×300 crop | wide FOV squashed square, ~1/8.6 scale |
| wrist cameras | rectilinear 640×360 | **fisheye**, circular vignette, 480×480 |
| photometry | varied objects/lighting | dark, low-contrast, gray-on-black |
| arm motion | coarse pick/transfer primitives | **0.39 mm/frame**; fingers do the work |
| horizon | short primitives | 135–172 s, 6 sequential folds |
| language | 22 distinct instructions | **one constant string** |
| embodiment | Dexmate Vega-1 arms | Sharpa North POC2.2 arms |
| hands + tactile | Sharpa Wave | **Sharpa Wave — identical** |
| corpus | ~50 h | **≈ 109 h, ≈ 11.8 M frames** |

* The **action-scale** part is largely neutralized by q01/q99 normalization — deltas map to
  [−1,1] using origami's own quantiles. §11.1 is about SNR *inside* that range, not scale.
* The **visual** part lands on a **frozen** ViT, so it cannot be fine-tuned away — an argument
  for **not** freezing the latent expert.
* The strongest mitigation is that **we have ≈ 109 h of in-domain data**, over 2× T-Rex's own
  midtrain corpus. This is domain-specific training warm-started from a related checkpoint, not
  a few-shot fine-tune.

Ordered mitigations, all cheap relative to the 40 k run:
1. **Zero-shot evaluation first.** `eval_offline.py --zero-shot` on the untrained midtrain
   checkpoint quantifies the whole shift in one number and sets the floor. **Run before step 11.**
   If it is no better than hold-position, the pretrained init is worth less than assumed.
2. **`diagnose_shift.py`** — origami vs a sample of `zekaiwang/trex_dataset`: chunk-delta
   magnitude per block, F6 magnitude per finger, VQ-VAE code entropy (G15), deform occupancy,
   and frozen-ViT feature statistics (per-channel mean/std of pooled patch tokens + cosine
   distance between dataset means). One markdown report.
3. **Staged unfreeze** — §7.5-E phase A before phase B.
4. **Discriminative LR** (ablation, off by default): `0.1 × lr` for `model.layers.*`, `1.0 × lr`
   for heads and the tactile expert.
5. **Do not re-initialize the loading heads.** At `action_dim=62` they load cleanly; discarding
   them throws away the only action prior we have. G9b asserts this.

---

## 12. Implementation order

Each step ends at named gates. Do not start step *n+1* until step *n*'s gates pass.

| step | deliverable | gate |
|---|---|---|
| 1 | Repo hygiene: submodule pinned + `full-pipeline` fetched, deletions, `.gitignore`, `pyproject.toml` (uv) with `pin`, `pink`, `qpsolvers[daqp]`, `av`, `lerobot@<pin>`, `huggingface_hub`, `pandas` | G0 |
| 2 | `tests/test_lerobot_probe.py` + `tests/test_processor_probe.py` | §11.4 resolved; §4.5-A size chosen |
| 3 | `constants.py` + `splits.json` + `kinematics.py` | G1a, G1b, G2, G2b, G3, G3b, G18 (list parsing) |
| 4 | `decode.py` | G5, G6 |
| 5 | `stats.py` | G8 |
| 6 | `convert.py` — single-season shard | G4, G4b, G8b |
| 7 | `fetch.py` + `merge.py` + `prepare.py` | G18 (hub resolution); merged 2-season root opens in `TRexLeRobotDataset` |
| 8 | `verify.py` — every gate as a subcommand | G7, G7b, G12 |
| 9 | `trex_patch.py` + `train_origami.py` + `delayed_lerobot_dataset.py` + `.sh` | G9a, G9b, G17, §11.7 test |
| 10 | **Full prep run on the large server** (126 seasons, both splits) | manifest complete; truncation < 1% |
| 10b | `diagnose_shift.py` + `eval_offline.py --zero-shot` — no training | **G15**; zero-shot floor recorded |
| 10c | `refit_vqvae_stats.py` if G15 failed | G15 passes |
| 11 | Pilot train 2 000 steps + §8.1-A/D | **§11.1 decision point**; beats the 10b floor |
| 11b | Optional phase 0 / phase A ablations (§7.5-E) | each beats the floor, else skip |
| 12 | Phase B: 40 000 steps with the delay curriculum | val curve; checkpoints preserved |
| 13 | `policy.py` + `retarget.py` + `serve_zenoh.py` + `bench_latency.py` | G10, G13, G14, G16 |
| 14 | `eval_offline.py` §8.1-B..F + `eval_shadow.py` | G11 |
| 15 | `docker/` submission image | SDK container checks pass |

Steps 10–12 are **runs**, executed by the user; build them launchable and resumable.

---

## 13. What to import from T-Rex (never reimplement)

```python
from utils.lerobot_common import (          # T-Rex/utils/lerobot_common.py
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
```
`hardware_code/eval/eval_trex_async.py` has heavy hardware imports at module level — **copy**
`rot6d_to_matrix` (`:283`), `matrix_to_rot6d` (`:295`) and `aggregate_chunks` (`:75`) into
`origami/retarget.py` with a source citation instead of importing.
`hardware_code/teleop/ik_utils.py::PinkLocalIK.solve_ik` (`:58`) is likewise **copied** into
`origami/kinematics.py` (the upstream class hard-codes the Dexmate model and `L_ee`/`R_ee`).

`origami/__init__.py` must put `T-Rex/` on `PYTHONPATH` (guarded) so `import
utils.lerobot_common` resolves — every upstream `.sh` does the same.
