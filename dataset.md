---
license: cc-by-4.0
---

<p align="center">
  <img src="https://huggingface.co/datasets/SharpaIT/Robotic_Origami_Challenge/resolve/main/assets/fold_plane_overview.gif" alt="Robotic Origami Challenge fold-plane overview" width="100%">
</p>

<h1 align="center">Robotic Origami Challenge: Fold Plane Demonstrations</h1>

<p align="center">
  <b>Real-world LeRobot demonstrations for dexterous paper-airplane folding.</b>
</p>

<p align="center">
  <a href="https://www.sharpa.com/"><img src="https://img.shields.io/badge/Website-sharpa.com-555555?style=flat-square" alt="Website"></a>
  <a href="https://github.com/sharpa-robotics"><img src="https://img.shields.io/badge/GitHub-Sharpa-24292f?style=flat-square&logo=github&logoColor=white" alt="GitHub"></a>
  <a href="https://www.linkedin.com/company/sharpa-robotics"><img src="https://img.shields.io/badge/LinkedIn-Sharpa-0a66c2?style=flat-square&logo=linkedin&logoColor=white" alt="LinkedIn"></a>
  <a href="https://www.youtube.com/@sharpa-robotics"><img src="https://img.shields.io/badge/YouTube-Sharpa-ff0000?style=flat-square&logo=youtube&logoColor=white" alt="YouTube"></a>
  <a href="https://x.com/SharpaRobotics"><img src="https://img.shields.io/badge/X-%40SharpaRobotics-000000?style=flat-square&logo=x&logoColor=white" alt="X"></a>
  <a href="https://robotic-origami-challenge.github.io/"><img src="https://img.shields.io/badge/Project-Robotic%20Origami%20Challenge-0077b6?style=flat-square" alt="Project Page"></a>
  <a href="https://huggingface.co/datasets/SharpaIT/Robotic_Origami_Challenge"><img src="https://img.shields.io/badge/Dataset-Hugging%20Face-f3b000?style=flat-square" alt="Dataset"></a>
</p>

## Overview

**Robotic Origami Challenge: Fold Plane Demonstrations** is a real-world teleoperation dataset for folding a paper airplane with a bimanual dexterous robot system. It is released by **Sharpa** in a **LeRobot-compatible** format for the Robotic Origami Challenge community.

Origami is a demanding benchmark for embodied AI: paper is thin, deformable, easy to occlude, and highly sensitive to contact timing and crease quality. A successful policy must coordinate two arms, dexterous hands, tactile feedback, and multi-view vision over a long sequential horizon.

The dataset is designed for imitation learning, visuomotor policy learning, visual-tactile representation learning, long-horizon action modeling, and policy development for the [Robotic Origami Challenge](https://robotic-origami-challenge.github.io/).

<table>
  <tr>
    <td><b>Task</b><br>Traditional paper-airplane folding</td>
    <td><b>Format</b><br>LeRobot v3.0 / v2.1</td>
    <td><b>Scale</b><br>51 seasons / 682 episodes</td>
    <td><b>Frequency</b><br>30 FPS</td>
  </tr>
  <tr>
    <td><b>Video</b><br>6 synchronized streams</td>
    <td><b>State / Action</b><br>65D joint space</td>
    <td><b>Tactile</b><br>10-fingertip 6-axis signals + tactile video</td>
    <td><b>Use</b><br>Training and policy development</td>
  </tr>
</table>

## Target Figure

The target figure for the challenge is a traditional paper airplane, or *kami hikoki*. Every team folds the same figure.

| Target Figure Item | Specification |
| --- | --- |
| Figure | Traditional Japanese paper airplane |
| Paper size | 15 x 15 cm |
| Paper weight | >= 60 gsm |
| Fold count | 6 folds |
| Attempt limit | 10 minutes |
| Success criterion | The judge panel declares whether the robot folded a recognizable airplane |
| Ranking criterion | Successful attempts are ranked by folding time |
| Judge | Origami Grand Master from the Nippon Origami Association |

The task is judged on crease accuracy, structural fidelity, symmetry, and paper integrity. Whether the resulting plane flies is not part of the official score, but the folded structure should remain recognizable as the target airplane.

## Challenge Context

The Robotic Origami Challenge is an IROS 2026 competition in Pittsburgh focused on dexterous robotic origami. It benchmarks fine-motor, sequential paper manipulation against an origami standard curated by expert artists.

<table>
  <tr>
    <th>Resource</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><b>Teleoperation data</b></td>
    <td>500+ demonstrations collected on the target folding sequence, together with fold-sequence diagrams and reference folds.</td>
  </tr>
  <tr>
    <td><b>Simulation</b></td>
    <td>NVIDIA Isaac Sim based environment with thin-shell paper physics, plastic creasing, fold memory, partner robotic-hand digital twins, and the same scoring rubric as the live event.</td>
  </tr>
  <tr>
    <td><b>Real-world evaluation</b></td>
    <td>A remote lab where participants upload policies and run them on the same bimanual arms and Sharpa Hands rig used for the challenge.</td>
  </tr>
</table>

The remote evaluation setup lowers the hardware barrier for participants: teams can train in simulation, use the demonstration data, upload policies, and evaluate on a standardized real-world system.

## Example Views

The demonstrations include synchronized head, wrist, and tactile video streams. Each preview below uses a 3-minute window from the same `lerobotv2.1` episode, played at 10x speed. GIF previews render directly in Markdown; click any preview to open the MP4 version. Tactile previews preserve their full wide-frame layout.

<table>
  <tr>
    <td width="50%">
      <b>Head Left</b><br>
      <a href="https://huggingface.co/datasets/SharpaIT/Robotic_Origami_Challenge/resolve/main/assets/example_views/head_left.mp4">
        <img src="https://huggingface.co/datasets/SharpaIT/Robotic_Origami_Challenge/resolve/main/assets/example_views/head_left.gif" alt="Head Left" width="100%">
      </a>
    </td>
    <td width="50%">
      <b>Head Right</b><br>
      <a href="https://huggingface.co/datasets/SharpaIT/Robotic_Origami_Challenge/resolve/main/assets/example_views/head_right.mp4">
        <img src="https://huggingface.co/datasets/SharpaIT/Robotic_Origami_Challenge/resolve/main/assets/example_views/head_right.gif" alt="Head Right" width="100%">
      </a>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <b>Wrist Left</b><br>
      <a href="https://huggingface.co/datasets/SharpaIT/Robotic_Origami_Challenge/resolve/main/assets/example_views/wrist_left.mp4">
        <img src="https://huggingface.co/datasets/SharpaIT/Robotic_Origami_Challenge/resolve/main/assets/example_views/wrist_left.gif" alt="Wrist Left" width="100%">
      </a>
    </td>
    <td width="50%">
      <b>Wrist Right</b><br>
      <a href="https://huggingface.co/datasets/SharpaIT/Robotic_Origami_Challenge/resolve/main/assets/example_views/wrist_right.mp4">
        <img src="https://huggingface.co/datasets/SharpaIT/Robotic_Origami_Challenge/resolve/main/assets/example_views/wrist_right.gif" alt="Wrist Right" width="100%">
      </a>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <b>Tactile Deformation</b><br>
      <a href="https://huggingface.co/datasets/SharpaIT/Robotic_Origami_Challenge/resolve/main/assets/example_views/tactile_deform.mp4">
        <img src="https://huggingface.co/datasets/SharpaIT/Robotic_Origami_Challenge/resolve/main/assets/example_views/tactile_deform.gif" alt="Tactile Deformation" width="100%">
      </a>
    </td>
    <td width="50%">
      <b>Raw Tactile</b><br>
      <a href="https://huggingface.co/datasets/SharpaIT/Robotic_Origami_Challenge/resolve/main/assets/example_views/tactile_raw.mp4">
        <img src="https://huggingface.co/datasets/SharpaIT/Robotic_Origami_Challenge/resolve/main/assets/example_views/tactile_raw.gif" alt="Raw Tactile" width="100%">
      </a>
    </td>
  </tr>
</table>

## Dataset Capabilities

| Capability | Dataset Support |
| --- | --- |
| Long-horizon imitation learning | Real-world demonstrations for the official paper-airplane folding sequence |
| Multi-view visuomotor policies | Synchronized head-camera and wrist-camera observations |
| Visual-tactile learning | High-resolution tactile videos, synchronized raw tactile camera views, and 10-fingertip 6-axis tactile signals |
| Joint-space control | 65D synchronized state and action for two arms, two dexterous hands, and torso/motor-related joints |
| LeRobot ecosystem | Full `lerobot3.0` coverage across all seasons, with `lerobotv2.1` available for many seasons |

## Dataset Statistics

| Item | Value |
| --- | ---: |
| Total collection seasons | 51 |
| `lerobot3.0` seasons | 51 |
| `lerobot3.0` episodes | 682 |
| `lerobot3.0` frames | 4,763,267 |
| `lerobotv2.1` seasons | 41 |
| `lerobotv2.1` episodes | 555 |
| `lerobotv2.1` frames | 3,893,595 |
| FPS | 30 |
| Video streams | 6 |
| State/action dimension | 65 |

For new users, we recommend starting from `lerobot3.0`, since it covers all 51 seasons in this release.

## Get Started

### Download The Dataset

Make sure Git LFS is installed before cloning from Hugging Face.

```bash
git lfs install
git clone https://huggingface.co/datasets/SharpaIT/Robotic_Origami_Challenge
```

If you want to clone only metadata first and fetch large files later:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/SharpaIT/Robotic_Origami_Challenge
```

If you only want a specific season, use sparse checkout:

```bash
git init Robotic_Origami_Challenge
cd Robotic_Origami_Challenge
git remote add origin https://huggingface.co/datasets/SharpaIT/Robotic_Origami_Challenge
git sparse-checkout init
git sparse-checkout set season_POC22032_2026_05_14_19_21_01_train README.md
git pull origin main
```

### Quick Inspection

Each season contains one or two LeRobot exports. Inspect `meta/info.json` first to understand the exact schema and file templates.

```python
import json
from pathlib import Path

dataset_root = Path("Robotic_Origami_Challenge")
episode_root = dataset_root / "season_POC22032_2026_05_14_19_21_01_train" / "lerobot3.0"

with open(episode_root / "meta" / "info.json", "r") as f:
    info = json.load(f)

print(info["total_episodes"])
print(info["total_frames"])
print(info["features"].keys())
```

## Dataset Structure

The dataset is organized by collection season. Each season may contain a `lerobot3.0` export and, when available, a `lerobotv2.1` export.

```text
Robotic_Origami_Challenge/
├── README.md
├── season_POC22032_2026_05_14_19_21_01_train/
│   ├── lerobot3.0/
│   │   ├── meta/
│   │   │   ├── info.json
│   │   │   ├── modality.json
│   │   │   ├── episodes.jsonl
│   │   │   └── tasks.jsonl
│   │   ├── data/
│   │   │   └── chunk-000/
│   │   └── videos/
│   │       ├── observation.images.head_left/
│   │       ├── observation.images.head_right/
│   │       ├── observation.images.wrist_left/
│   │       ├── observation.images.wrist_right/
│   │       ├── observation.images.tactile_deform/
│   │       └── observation.images.tactile_raw/
│   └── lerobotv2.1/
│       ├── meta/
│       ├── data/
│       └── videos/
└── season_.../
```

### LeRobot Storage Layout

| Part | Description |
| --- | --- |
| `meta/` | Dataset metadata, feature schema, task metadata, and path templates |
| `data/` | Episode frame data stored as Apache Parquet files |
| `videos/` | Per-camera MP4 videos |

The most important metadata file is `meta/info.json`. It defines `total_episodes`, `total_frames`, `fps`, `splits`, `data_path`, `video_path`, and `features`.

### File Path Templates

LeRobot v3.0 uses templates similar to:

```text
data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet
videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4
```

LeRobot v2.1 uses templates similar to:

```text
data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet
videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4
```

## Features Schema

### Main Feature Groups

| Feature | Type | Shape | Description |
| --- | --- | ---: | --- |
| `observation.state` | float32 | 65 | Joint-space robot state |
| `action` | float32 | 65 | Joint-space action |
| `observation.state.joint_torque` | float32 | 65 | Joint torque signal |
| `observation.tactile` | float32 | 60 | 10-fingertip 6-axis force/torque tactile signal |
| `observation.images.*` | video | varies | Multi-view visual observations |
| `timestamp` | float32 | 1 | Timestamp |
| `frame_index` | int64 | 1 | Frame index within an episode |
| `episode_index` | int64 | 1 | Episode index |
| `task_index` | int64 | 1 | Task index |

### Proprioceptive State

The 65D `observation.state` and `action` vectors are ordered as follows:

| Range | Names | Meaning |
| --- | --- | --- |
| 0 - 6 | `left_arm_j0` to `left_arm_j6` | Left arm joints |
| 7 - 28 | `left_hand_j0` to `left_hand_j21` | Left dexterous hand joints |
| 29 - 35 | `right_arm_j0` to `right_arm_j6` | Right arm joints |
| 36 - 57 | `right_hand_j0` to `right_hand_j21` | Right dexterous hand joints |
| 58 - 64 | `motor_j0` to `motor_j6` | Torso / motor-related joints |

### Video Streams

The complete visual observation set contains six camera streams:

| Feature key | Description | Shape |
| --- | --- | --- |
| `observation.images.head_left` | Left head camera | 480 x 480 x 3 |
| `observation.images.head_right` | Right head camera | 480 x 480 x 3 |
| `observation.images.wrist_left` | Left wrist camera | 480 x 480 x 3 |
| `observation.images.wrist_right` | Right wrist camera | 480 x 480 x 3 |
| `observation.images.tactile_deform` | Tactile deformation video | 480 x 1200 x 3 |
| `observation.images.tactile_raw` | Raw tactile video | 480 x 1600 x 3 |

Video files are MP4 without audio. Codec may differ across exports and seasons. `lerobot3.0` is primarily AV1, while `lerobotv2.1` is primarily H.264.

## Tactile Modality

The dataset includes tactile observations as both compact numeric signals and high-resolution video streams. These modalities are synchronized with the robot state, action, and visual camera streams at 30 FPS, making them suitable for contact-rich policy learning and visual-tactile representation learning.

| Tactile feature | Type | Shape | Description |
| --- | --- | ---: | --- |
| `observation.tactile` | float32 | 60 | Per-frame tactile force/torque signal: 10 fingertips x 6 axes |
| `observation.images.tactile_deform` | video | 480 x 1200 x 3 | Deformation-oriented tactile video stream that visualizes contact-induced surface changes |
| `observation.images.tactile_raw` | video | 480 x 1600 x 3 | Raw tactile camera stream preserving the full tactile sensor image layout |

The `observation.tactile` vector provides a compact force/torque representation for each frame. It contains 10 fingertip groups: left and right `thumb`, `index`, `middle`, `ring`, and `little`. Each fingertip contributes six values, ordered as `fx`, `fy`, `fz`, `tx`, `ty`, and `tz`, for a total of 60 dimensions. This makes it suitable for models that consume structured numeric tactile feedback alongside proprioception and action labels. The two tactile video streams provide complementary image-based tactile observations: `tactile_deform` emphasizes deformation patterns caused by contact, while `tactile_raw` preserves the raw tactile image for users who want to build their own visual-tactile preprocessing or representation learning pipeline.

For downstream experiments, users can start with `observation.tactile` as a lightweight contact signal, then add one or both tactile video streams when the model architecture can handle the additional spatial resolution and bandwidth. The per-season `meta/info.json` records the exact feature schema and channel names.

## Usage Recommendations

This release is provided as training data. The official Robotic Origami Challenge evaluation is performed through the competition's real-world evaluation setup rather than through a fixed validation split in this repository.

For local experiments, users may split by season to avoid mixing demonstrations from the same collection session across train and evaluation sets.

For policy learning, a typical setup is:

- Visual observations: one or more `observation.images.*` streams
- Proprioception: `observation.state`
- Optional tactile inputs: `observation.tactile`, `observation.images.tactile_deform`, and/or `observation.images.tactile_raw`
- Supervision target: `action`

For multi-view policies, start with:

```text
observation.images.head_left
observation.images.head_right
observation.images.wrist_left
observation.images.wrist_right
```

Then add tactile video streams if your model can use high-resolution tactile observations:

```text
observation.images.tactile_deform
observation.images.tactile_raw
```

## Dataset Notes

- The dataset is released by season, not as a single flattened LeRobot root.
- `lerobot3.0` has full season coverage in this release.
- Some seasons also include `lerobotv2.1` for compatibility with older pipelines.
- `motor_j0` to `motor_j6` are the torso/motor-related dimensions in `observation.state` and `action`.
- The task is contact-rich and deformable: policies should expect paper motion, occlusions, hand-paper contacts, and fine-grained crease formation.


## My Split

```json
{
  "num_train": 101,
  "num_val": 25,
  "val_fraction": 0.2,
  "split": "random",
  "seed": 0,
  "train_seasons": [
    "season_POC22032_2026_05_14_19_21_01_train",
    "season_POC22032_2026_05_14_20_40_58_train",
    "season_POC22032_2026_05_14_21_08_06_train",
    "season_POC22032_2026_05_15_16_43_23_train",
    "season_POC22061_2026_05_18_19_25_41_train",
    "season_POC22061_2026_05_19_13_40_31_train",
    "season_POC22061_2026_05_19_15_37_17_train",
    "season_POC22061_2026_05_19_19_08_43_train",
    "season_POC22061_2026_05_19_21_17_23_train",
    "season_POC22061_2026_05_20_10_23_55_train",
    "season_POC22061_2026_05_20_14_02_17_train",
    "season_POC22061_2026_05_20_17_13_24_train",
    "season_POC22061_2026_05_20_19_24_50_train",
    "season_POC22061_2026_05_23_10_50_01_train",
    "season_POC22061_2026_05_23_13_39_47_train",
    "season_POC22061_2026_05_24_10_23_04_train",
    "season_POC22061_2026_05_24_13_41_09_train",
    "season_POC22061_2026_05_24_19_33_29_train",
    "season_POC22061_2026_05_25_13_50_44_train",
    "season_POC22061_2026_05_25_16_02_56_train",
    "season_POC22061_2026_05_26_10_13_09_train",
    "season_POC22061_2026_05_26_13_55_16_train",
    "season_POC22061_2026_05_26_19_13_22_train",
    "season_POC22061_2026_05_26_20_26_01_train",
    "season_POC22061_2026_05_27_13_36_59_train",
    "season_POC22061_2026_05_27_15_57_42_train",
    "season_POC22061_2026_05_27_19_13_53_train",
    "season_POC22061_2026_05_28_10_34_44_train",
    "season_POC22061_2026_05_28_19_19_16_train",
    "season_POC22061_2026_05_28_20_12_14_train",
    "season_POC22061_2026_05_29_13_40_14_train",
    "season_POC22061_2026_05_29_19_14_17_train",
    "season_POC22061_2026_05_30_10_12_51_train",
    "season_POC22061_2026_05_30_16_15_19_train",
    "season_POC22061_2026_06_24_13_45_15_train",
    "season_POC22061_2026_06_25_13_34_16_train",
    "season_POC22061_2026_06_25_15_41_29_train",
    "season_POC22061_2026_06_26_13_37_24_train",
    "season_POC22061_2026_06_26_19_05_35_train",
    "season_POC22061_2026_06_27_10_08_03_train",
    "season_POC22061_2026_06_27_15_56_17_train",
    "season_POC22061_2026_06_28_10_10_11_train",
    "season_POC22061_2026_06_28_15_55_53_train",
    "season_POC22061_2026_06_28_19_26_39_train",
    "season_POC22061_2026_06_29_10_12_41_train",
    "season_POC22061_2026_06_29_13_34_29_train",
    "season_POC22061_2026_06_29_14_27_15_train",
    "season_POC22061_2026_06_29_15_46_06_train",
    "season_POC22061_2026_06_29_16_37_08_train",
    "season_POC22061_2026_06_30_13_40_03_train",
    "season_POC22061_2026_07_01_10_15_18_train",
    "season_POC22061_2026_07_01_13_35_43_train",
    "season_POC22061_2026_07_01_15_48_25_train",
    "season_POC22061_2026_07_01_19_29_05_train",
    "season_POC22061_2026_07_02_14_23_26_train",
    "season_POC22061_2026_07_02_16_04_24_train",
    "season_POC22061_2026_07_02_19_10_24_train",
    "season_POC22061_2026_07_04_10_09_32_train",
    "season_POC22061_2026_07_04_16_13_34_train",
    "season_POC22061_2026_07_04_20_38_34_train",
    "season_POC22061_2026_07_05_10_11_10_train",
    "season_POC22061_2026_07_05_13_37_13_train",
    "season_POC22061_2026_07_06_11_05_16_train",
    "season_POC22061_2026_07_07_10_52_58_train",
    "season_POC22061_2026_07_07_13_37_02_train",
    "season_POC22061_2026_07_07_19_15_51_train",
    "season_POC22061_2026_07_08_10_10_59_train",
    "season_POC22061_2026_07_08_11_10_11_train",
    "season_POC22061_2026_07_08_13_37_45_train",
    "season_POC22061_2026_07_08_16_01_38_train",
    "season_POC22061_2026_07_08_17_17_40_train",
    "season_POC22061_2026_07_08_19_05_30_train",
    "season_POC22061_2026_07_09_10_08_39_train",
    "season_POC22061_2026_07_09_13_37_31_train",
    "season_POC22061_2026_07_09_16_23_46_train",
    "season_POC22061_2026_07_09_19_08_15_train",
    "season_POC22061_2026_07_10_10_04_17_train",
    "season_POC22061_2026_07_10_11_05_01_train",
    "season_POC22061_2026_07_10_19_07_06_train",
    "season_POC22061_2026_07_11_10_07_34_train",
    "season_POC22061_2026_07_12_10_06_51_train",
    "season_POC22061_2026_07_12_13_50_31_train",
    "season_POC22061_2026_07_12_17_04_27_train",
    "season_POC22061_2026_07_13_14_31_37_train",
    "season_POC22061_2026_07_13_14_54_37_train",
    "season_POC22061_2026_07_14_10_09_42_train",
    "season_POC22061_2026_07_14_11_01_49_train",
    "season_POC22061_2026_07_14_13_36_02_train",
    "season_POC22061_2026_07_14_15_06_58_train",
    "season_POC22061_2026_07_14_16_42_30_train",
    "season_POC22061_2026_07_14_19_16_15_train",
    "season_POC22061_2026_07_14_20_09_08_train",
    "season_POC22061_2026_07_15_10_13_37_train",
    "season_POC22061_2026_07_15_13_58_14_train",
    "season_POC22061_2026_07_15_16_41_55_train",
    "season_POC22061_2026_07_15_19_11_36_train",
    "season_POC22061_2026_07_17_10_23_21_train",
    "season_POC22061_2026_07_17_13_35_14_train",
    "season_POC22061_2026_07_17_15_59_04_train",
    "season_POC22061_2026_07_17_19_08_00_train",
    "season_POC22061_2026_07_18_10_03_25_train"
  ],
  "val_seasons": [
    "season_POC22061_2026_05_19_10_18_58_train",
    "season_POC22061_2026_05_20_16_07_05_train",
    "season_POC22061_2026_05_23_15_56_33_train",
    "season_POC22061_2026_05_27_10_39_38_train",
    "season_POC22061_2026_05_28_13_42_37_train",
    "season_POC22061_2026_05_28_15_51_13_train",
    "season_POC22061_2026_05_29_10_19_22_train",
    "season_POC22061_2026_05_29_15_58_16_train",
    "season_POC22061_2026_06_25_20_02_02_train",
    "season_POC22061_2026_06_27_13_48_20_train",
    "season_POC22061_2026_06_27_19_13_22_train",
    "season_POC22061_2026_06_28_13_47_21_train",
    "season_POC22061_2026_06_29_19_18_53_train",
    "season_POC22061_2026_06_30_10_16_20_train",
    "season_POC22061_2026_06_30_19_27_21_train",
    "season_POC22061_2026_06_30_19_45_01_train",
    "season_POC22061_2026_07_04_13_39_40_train",
    "season_POC22061_2026_07_05_16_23_24_train",
    "season_POC22061_2026_07_10_15_21_08_train",
    "season_POC22061_2026_07_10_17_05_30_train",
    "season_POC22061_2026_07_11_13_34_49_train",
    "season_POC22061_2026_07_13_19_20_33_train",
    "season_POC22061_2026_07_13_19_56_06_train",
    "season_POC22061_2026_07_13_20_50_09_train",
    "season_POC22061_2026_07_14_15_43_22_train"
  ]
}
```

## License and Terms

This dataset is released under the [Creative Commons Attribution 4.0 International License (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/). You may use, share, and adapt the dataset, including for commercial purposes, provided that you give appropriate attribution.

If you use the dataset for Robotic Origami Challenge participation, please also follow the official competition rules and evaluation protocol.

## Citation

If this dataset contributes to your research, please cite or acknowledge the dataset and the Robotic Origami Challenge.

```bibtex
@misc{robotic_origami_challenge_fold_plane_2026,
  title        = {Robotic Origami Challenge Fold Plane LeRobot Dataset},
  howpublished = {\url{https://huggingface.co/datasets/SharpaIT/Robotic_Origami_Challenge}},
  year         = {2026}
}

@misc{robotic_origami_challenge_2026,
  title        = {The Robotic Origami Challenge},
  howpublished = {\url{https://robotic-origami-challenge.github.io/}},
  year         = {2026}
}
```