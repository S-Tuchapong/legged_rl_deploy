# G1 full-body beam depth distillation

Deployment of `model_7000.pt` (7000 updates), with all 29 joints
controlled. `config.yaml` follows the Go2 depth deployment contract, but uses the
G1 training observation, action, camera and recurrent-state interfaces.
This document covers the legacy `policy.onnx` / `config.yaml` pair. For the
V1–V4 simulation configs and tasks, use [README_SIM_VARIANTS.md](README_SIM_VARIANTS.md).

Original training parameters are retained in `training_params/`; raw TensorBoard
logs and smoke-run checkpoints are excluded from version control.

## Model contract

- Policy: 50 Hz; low-level loop: 200 Hz.
- Proprioception: 5978 values, 61 frames in term-major order: angular velocity,
  projected gravity, velocity commands, gait phase, joint position, joint velocity,
  previous clipped raw action. History starts by repeating the first observation.
- Depth: `[1,1,240,424]`, **optical-axis metres**, finite and nonnegative.
  The export maps nonfinite/nonpositive values (including zero/no-hit) to 6 m,
  clips to [0,6] m, and normalizes with `z / 6` inside ONNX.
  Feed raw metres to this wrapper, not already-normalized depth.
  RealSense already reports axial depth: do not apply a ray cosine correction.
  A MuJoCo ray-range image must be converted to axial depth upstream.
- The ONNX graph consumes depth on alternate policy calls (25 Hz), holds its visual
  features between those calls, and supplies freshness and age (0 / 0.02 seconds).
  These are scheduled holds, as in training; ROS arrival timestamps are not model inputs.
- Runtime states: 64 terrain GRU values, 64 cached visual features and a hold toggle.
  The runner zeroes all three on policy reset. The 61-frame proprioception GRU is
  recomputed from the input window each call, as in training.
- Commands: forward/backward ±0.8 m/s, zero lateral and yaw command. Gait period:
  0.9 seconds; activity threshold: zero.
- Soft joint limits clip raw actions before PD target scaling; the clipped actions
  feed the next observation. No Go2 ±3 action clip is used.

`training_core.py` is preserved byte-for-byte from G1Beam commit `80c0826`, path
`src/tasks/distillation/fullbody_depth/core.py`. Its SHA-256 matches the original
checkpoint's recorded `depth_memory_fullbody_course_ground_v3_2gpu512/core.py` hash:
`277cdcaaeaa5d3e539033c8e24da4f5c3eaebc6d1979bd2df964bae93d4db29f`.
`depth_health.py` is its companion from the same commit. This checkpoint was
trained with 0.3–3 m preprocessing. The current export intentionally uses the
requested **0–6 m deployment override**. The checkpoint and preserved training
source are unchanged. Numerical export parity does not establish walking quality
under the changed preprocessing. V1–V4 already use the 0–6 m depth contract.
No training repository or teacher checkpoint is required to run the exported ONNX.

## Camera setup

Training camera pose relative to `torso_link`: position `[0.12, 0, 0.42]`, MuJoCo
quaternion (wxyz) `[0.60483387, 0.36630041, -0.36630041, -0.60483387]`, vertical
FOV `58.310943190845826` degrees, image 424×240. Match the physical camera geometry.

`D435i.yaml` adapts the template's 848×480×60 stream to decimation factor 2.
The preprocessor's intrinsics are scaled from the Go2 template's calibrated camera;
**replace them with your G1 camera's actual post-decimation CameraInfo values**.
The strict calibration check rejects mismatches. Compare the actual FOV to training
as well; merely changing the expected calibration does not correct image geometry.
The template's hardware filters are retained; training used no synthetic corruption.

## Build and run

Run from `/home/pro/unitree_ws`. Rebuild for the configurable gait threshold:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
colcon build --packages-select legged_rl_deploy
source install/setup.bash
```

In separate terminals with the same ROS setup (replace `eth0` with your interface):

```bash
ros2 launch realsense2_camera rs_launch.py \
  config_file:="'src/legged_rl_deploy/policies/g1/beam/g1_beam_depth_distillation/D435i.yaml'"

ros2 run legged_rl_deploy depth_image_preprocessor_node.py --ros-args \
  --params-file src/legged_rl_deploy/policies/g1/beam/g1_beam_depth_distillation/depth_image_preprocessor.yaml

ros2 topic hz /unitree_g1_beam_depth/depth_m

./src/legged_rl_deploy/scripts/run.sh eth0 humble \
  ros2 run legged_rl_deploy legged_rl_deploy_node eth0 \
  src/legged_rl_deploy/policies/g1/beam/g1_beam_depth_distillation/config.yaml
```

## Reproduce the export

Use a Python environment containing `torch`, `numpy`, `onnx`, and `onnxruntime`:

```bash
python src/legged_rl_deploy/policies/g1/beam/g1_beam_depth_distillation/export_policy.py
```

The script strictly loads the checkpoint, verifies the recorded source hash,
exports a fixed batch-size-1 graph and compares ONNX with the original training
Student supplied with the requested 0–6 m preprocessing over 80 recurrent steps,
including holds, resets, invalid values and clipping endpoints. `validation.json`
records tensor shapes, numerical error, and both the original training range and
the deployment override. It replaces adaptive average pooling
with mathematically equivalent fixed spatial slices supported by ONNX.

Validation covers CPU numerical parity and deployment configuration. No physical
robot, camera stream or closed-loop simulation has been run for this policy.

## MuJoCo first (no RealSense driver needed)

All four workspace folders provide these VS Code tasks:

1. `G1 Beam Sim: Mujoco + Depth` — starts G1 in `scene_bridge.xml` with the
   training camera attached to its torso, and the simulation depth preprocessor.
2. `G1 Beam Sim: Check Depth Rate` — verify `/g1_sim/beam_depth/normalized` publishes.
3. `G1 Beam Sim: Policy` — select a V1–V4 `config_sim.yaml` on loopback (`lo`).

Stop previously started hardware Camera/Depth/Policy tasks before this sequence.
Simulation tasks always use `lo` and do not launch `realsense2_camera`. The physical
joystick remains enabled; no upper-body lock is applied. The simulator requires a
working graphical display even with `--no-viewer`, because depth uses OpenGL.

`mujoco.yaml` and its adjacent `camera_profiles.yaml` specify the training camera
pose and centered 424×240 intrinsics. The simulator publishes axial Z16 depth on
`/g1_sim/camera/depth/image_rect_raw`. The current variant tasks use
`depth_image_preprocessor_variants_sim.yaml` to resize to 212×120 and normalize
once to [0,1] using the 0–6 m range. The root `config_sim.yaml` selects V2;
it has a different model interface from the legacy hardware `config.yaml`.
`depth_image_preprocessor_sim.yaml` remains the raw-metre pipeline for legacy
wrappers that normalize internally.
The existing bridge is a smoke-test scene, not an exact copy of the training course.

After source changes, rebuild both packages from the workspace root:

```bash
colcon build --packages-select unitree_mujoco legged_rl_deploy
```

Simulation smoke validation: both packages rebuilt, bridge XML loaded, camera
quaternion checked against training, and 10 rendered/preprocessed 424×240 `32FC1`
frames received on an isolated ROS domain. CameraInfo validation passed. This
checks the sensor pipeline; it does not establish closed-loop walking performance.

## Compare against training terrain

Stop the previous MuJoCo/depth/policy tasks, then run:

1. **G1 Beam Sim: Training Terrain + Depth**; choose level 0 first.
2. **G1 Beam Sim: Check Depth Rate**.
3. **G1 Beam Sim: Policy**; select the same V1–V4 variant for each terrain.

The new scenes are `unitree_mujoco/unitree_robots/g1/scene_depth_training_level_0.xml`
through `scene_depth_training_level_9.xml`. They reproduce one Phase II training
lane at the playback height: 4 m beam, 0.8 m square platforms, solid boxes down to
continuous ground, and a 0.55 m deck-to-ground distance. Width follows the exact
training schedule `0.60 - level * (0.60 - 0.18) / 9` metres. Levels 0, 6 and 9
are 0.60, 0.32 and 0.18 m wide respectively.

The whole course is translated +0.4 m in X and -0.55 m in Z relative to the
training lane, keeping the existing robot spawn on the start platform with the
deck at Z=0. Training terrain group 4 is enabled in both the viewer and depth
renderer. The old bridge and its launch task remain available for comparison.

Use the same command speed, startup procedure, spring-band state and number of
attempts on each scene. Begin with the wide level 0, then progress to narrower
levels. Compare fall location and time as well as successful crossings. A change
in outcomes indicates terrain sensitivity; it alone does not prove the failure's
cause. These are deterministic single-lane geometry tests, not a full recreation
of training: no randomized height, lane curriculum, randomized spawn, training
robot dynamics or automatic endpoint command logic is added. Existing simulator
robot model, initial joints, control and policy remain unchanged.

## Live normalized depth preview

The simulation tasks show a grayscale depth preview below the right-hand viewer
controls. They subscribe to `/g1_sim/beam_depth/normalized`, already processed as
`clip(z, 0, 6) / 6` with invalid/no-hit pixels mapped to 1. Preview near/far limits
are [0,1], so the display does not normalize the input a second time. Black is 0,
white is 1. The image is top-left oriented and keeps its aspect ratio.
`STALE` indicates no frame has arrived for more than 0.5 seconds.

The preview shows the latest input frame, not synchronized model-internal
features. Settings are under `depth_camera.preview` in the simulator YAMLs.
