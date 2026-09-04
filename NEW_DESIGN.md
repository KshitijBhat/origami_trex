To train the **T-Rex policy** on your **Robotic Origami Challenge** dataset, you need to map the **65D joint-space** dataset down to the **62D end-effector (EEF) + absolute finger** configuration that T-Rex expects. 

Below is a detailed breakdown of how to handle this dataset conversion, the official tooling provided in the repository to automate it, and how inference and Inverse Kinematics (IK) are executed during real-time rollouts.

---

### Converting the 65D Joint Space Dataset to 62D EEF Space

The 65-dimensional Origami Challenge dataset consists entirely of joint angles. T-Rex, on the other hand, operates on a **62D action space** consisting of **relative end-effector (EEF) deltas** for the arms and **absolute joint angles** for the fingers. 

The conversion pipeline involves four main steps:

#### Step 1: Discard the Torso/Motor Joint Dimensions
*   **Origami Space (65D)**: Contains 7 extra dimensions (`motor_j0` to `motor_j6`, indices 58 to 64) corresponding to the torso and auxiliary joints.
*   **T-Rex Space (62D)**: T-Rex holds the wheels, torso, and head joints fixed during training and execution. 
*   **Action**: Slice your vectors to ignore these final 7 dimensions, keeping only the 58 joint dimensions corresponding to the left arm (7), left hand (22), right arm (7), and right hand (22).

#### Step 2: Compute Forward Kinematics (FK) for the Arms
*   Using the robot’s URDF model, pass the 7D joint positions of the left and right arms through a **Forward Kinematics (FK)** solver. 
*   This computes the absolute 3D position \\([x, y, z]\\) and 3D rotation matrix \\(R\\) of each wrist's end-effector (EEF) frame at every timestep \\(t\\).

#### Step 3: Compute Relative Delta-Poses and 6D Rotations
Rather than outputting absolute cartesian coordinates, the T-Rex policy outputs relative delta commands:
*   **Translation Deltas**: Calculate the change in position between sequential frames: 
    \\[\Delta p = [dx, dy, dz]\\]
*   **Continuous 6D Rotations**: To prevent the training singularities and mathematical discontinuities common to Euler angles or quaternions, represent the rotation delta using a **continuous 6D rotation** representation. This is done by extracting the first two columns of the relative rotation delta matrix:
    \\[R_{\text{delta}} = R_t^T R_{t+1}\\]
    and flattening them into a 6-dimensional vector. 
*   This transforms the 7D joint commands of each arm into a **9D relative EEF action** (3 translation + 6 rotation).

#### Step 4: Concatenate with Absolute Hand Actions
*   Keep the **22D finger joint positions** for the left and right dexterous hands strictly in absolute space.
*   Combine the processed elements into the final target vector: 
    \\[\text{Left EEF (9D)} + \text{Left Fingers (22D)} + \text{Right EEF (9D)} + \text{Right Fingers (22D)} = \text{62D Action Space}\\]

---

### Using the Official T-Rex Script Tooling

You do not have to write this conversion code from scratch. The official T-Rex repository provides scripts to process and format your raw task directories directly:

1.  **For JSON-based training (Default)**:
    Run the bimanual generation script `utils/gen_json_tac_deltabase_eef_bimanual_parallel.py`. It automatically calculates the forward kinematics, computes the relative delta-poses (eef-62), and outputs a clean training JSON:
    ```bash
    python utils/gen_json_tac_deltabase_eef_bimanual_parallel.py \
        --data_roots /path/to/origami_raw_episodes \
        --img_save_root /path/to/training_data/images \
        --json_save_root /path/to/training_data/json \
        --task_name robotic_origami_challenge
    ```
2.  **For LeRobot v3.0 training (Opt-in)**:
    You can use the helper script `utils/convert_inlab_to_lerobot.sh` to package your raw episodes directly into a **LeRobot v3.0** directory. It parses the joint inputs, builds the 62D end-effector actions, and writes a normalized sidecar statistics file (`meta/trex_norm_stats.json`).

---

### Inference and the Inverse Kinematics (IK) Solver

#### How T-Rex Does Inference
At deployment, T-Rex runs an **asynchronous slow-fast cascaded protocol**:
*   **The Slow Tick (~5 Hz)**: The heavy visual tower and Qwen-VLA backbone process the camera observations, predicting future visual frames and caching the vision-language key-value embeddings (`KV_split`).
*   **The Fast Tick (~20 Hz)**: A lightweight tactile expert reuses the cached vision context and consumes high-frequency tactile forces/deformation sequences to predict residual refinements within the action chunk, bypassing the vision network entirely.

#### Do we need to solve Inverse Kinematics during inference?
**Yes, you must solve Inverse Kinematics (IK) during inference rollouts.**

Because the T-Rex policy outputs 62D actions—where the arms are commanded in 9D relative end-effector delta space—the low-level controllers of your bimanual arms cannot execute these cartesian deltas directly. 

To close this loop, the deployment setup incorporates a real-time solver pipeline:
1.  **Policy Output**: The model outputs the 62-dimensional command (containing the 9D arm Cartesian delta targets).
2.  **Differential IK Solver**: The inference rollout client routes the relative arm targets through **differential inverse kinematics** powered by the **Pink** library (which leverages **Pinocchio** and **CasADi** for high-speed, rigid-body dynamics optimization).
3.  **Low-Level Cascade Control**: The calculated joint commands are smoothed with a low-pass filter and pushed asynchronously to the robot's hardware/simulator cascade PID controller operating at a high-frequency **300 Hz loop**.

This decoupled setup ensures that the model can quickly output agile, cartesian-reactive touches, while the underlying mathematical solver translates those paths into stable physical motor joint targets.

---

### How T-Rex Uses Tactile Data

The T-Rex model uses a **spatial-temporal tactile encoding** architecture to capture how physical touch develops over time and across surfaces. It processes tactile inputs as two separate signals:

*   **Temporal Force Dynamics**: It measures 6D force and torque variations across the robot’s fingers. To filter out sensor noise and drift, a sliding window of **16 frames of raw 6D force history** is passed through a per-finger **VQ-VAE**. The VQ-VAE tokenizes these temporal changes into discrete tokens using a learned codebook (\\(K=64\\)). At the same time, the current, raw 6D force vector is projected directly into the network to make sure there is no delay in sensing instantaneous contact.
*   **Spatial Contact Geometry**: It monitors local skin displacement fields on each fingertip using **spatial deformation maps**. A lightweight convolutional network—adapted from the first three stages of a **ResNet-18** autoencoder—compresses these maps into geometry-aware features (capturing fine-grained details like slip, shear, and edge alignment). This encoder is pre-trained in a self-supervised way and kept frozen during policy training.

These temporal force tokens and spatial deformation embeddings are concatenated into a single tactile token sequence \\(z_t^\tau\\).

---

### Synchronisation and Expected Frequencies

**Yes, T-Rex expects synchronized raw tactile streams.** It is designed around two control loops running at different speeds to handle these streams:

*   **The Slow Visuomotor Loop (~5 Hz)**: The heavy Vision-Language-Action (VLA) backbone processes visual camera frames and language commands to produce a coarse action plan, caching its visual-language key-value embeddings (\\(KV_{\tau_{\text{split}}}\\)).
*   **The Fast Tactile Loop (~20 Hz)**: The lightweight tactile expert runs at a much higher rate. It re-fires asynchronously at offsets \\(\{0, 4, 8, 12\}\\) within each 16-frame control window. By reusing the frozen, cached visual context and focusing only on the high-frequency real-time tactile tokens, it can apply rapid physical adjustments directly to the robot's movements without slowing down to re-run the heavy vision tower.

The raw telemetry from the hardware is recorded at **30 Hz** (matching the 30 FPS visual stream), which is then interpolated and tracked by the physical robot's internal low-level cascade PID controller running at **300 Hz**.

---

### Tactile Data in the Robotic Origami Challenge Dataset

**Yes, the Sharpa Robotic Origami Challenge dataset provides this tactile data.** Because it uses the same physical hardware rig (a Dexmate Vega-1 robot with two Sharpa Wave hands), it records the same raw physical touch signals synchronized at 30 FPS. 

The Origami Challenge dataset includes:
1.  **`observation.tactile` (60D float32 vector)**: Consists of the 6-axis net force/torque wrench \\((F_x, F_y, F_z, M_x, M_y, M_z)\\) recorded across all 10 fingertips (thumb through pinky on both hands).
2.  **`observation.images.tactile_deform` (video)**: A wide-frame, multi-view composite video (\\(480 \times 1200 \times 3\\)) visualizing physical surface deformation patterns across the fingers.
3.  **`observation.images.tactile_raw` (video)**: The raw wide-frame composite camera stream (\\(480 \times 1600 \times 3\\)) from the tactile sensors.

---

### Required Transformations & Dataset Preprocessing

While the physical signals are identical, their formatting and structure in the Origami Challenge dataset differ from what the T-Rex training pipeline expects. To train the T-Rex policy on this data, you must apply three main transformations:

#### A. Feature Key Mapping
The raw Origami force key must be renamed and reshaped to match the T-Rex training schema:
*   In the Origami dataset, the force telemetry is a flat vector named **`observation.tactile`** (\\(60\\)).
*   T-Rex expects this under the key **`observation.tactile_force`** (for raw flat arrays) or reshaped into **`observation.tactile_f6`** (\\(10 \times 6\\), representing the 10 fingers and 6 wrench dimensions). The training model then reads a sliding historical window to construct a \\([B, 16, 10, 6]\\) tensor for on-the-fly VQ-VAE tokenization.

#### B. Slicing and Cropping Tactile Videos
*   **Origami Dataset**: Combines the tactile feeds of all fingers into single, wide composite video streams (`tactile_deform` at \\(480 \times 1200\\) and `tactile_raw` at \\(480 \times 1600\\)).
*   **T-Rex Policy**: Expects **10 separate, individual video files** (one per finger) under keys like `observation.images.tactile_left_deform_thumb` through `observation.images.tactile_right_deform_pinky` (at a cropped \\(240 \times 240\\) or \\(240 \times 320\\) resolution).
*   **Transformation**: You must write a script to crop and divide the wide multi-view composite videos into 10 separate fingertip video sequences.

#### C. Grayscale Luma (Y) Plane Extraction
*   The raw and deformation tactile videos are saved using lossless H.264 compression. Because these pixel values represent exact, physically meaningful depth measurements, standard color-space conversions will distort the data. 
*   **Transformation**: During video decoding, you must extract only the **luma (Y) plane** directly (e.g., using `gray` decoding format in PyAV or ffmpeg) to recover the original exact uint8 sensor signals without compression artifact distortion.

#### D. Using the Automated Scripts
To simplify this process, the T-Rex repository includes the helper script **`utils/convert_inlab_to_lerobot.sh`**. Running this script on your Origami Challenge directories automatically splits the composite tactile video frames, maps the force keys, and builds a standard LeRobot v3.0 directory with a unified statistics file (`meta/trex_norm_stats.json`) that is fully compatible with T-Rex training.

---