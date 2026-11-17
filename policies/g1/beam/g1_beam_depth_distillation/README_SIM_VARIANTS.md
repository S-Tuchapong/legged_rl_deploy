# V1–V4 ROS / MuJoCo deployment

The raw V1–V4 ONNX cores are supported by the updated deployment runner. Each
variant directory contains `config_sim.yaml`. The top-level `config_sim.yaml`
is also configured for V2. The old `policy.onnx` uses a different interface;
changing only its model path is not sufficient.

## Start from VS Code

Use the tasks belonging to the **legged_rl_deploy** workspace folder:

1. Stop earlier simulation/depth/policy tasks so there is only one publisher
   for each stream and one controller for the simulator.
2. Run **G1 Beam Sim: Training Terrain + Depth**, initially choosing level 0.
3. Run **G1 Beam Sim: Check Depth Rate**. Expect approximately 50 Hz on
   `/g1_sim/beam_depth/normalized`.
4. Run **G1 Beam Sim: Policy**, and select **V1**, **V2**, **V3**, or **V4**.
5. Use the usual controller sequence: `L2+A` to stand, wait for the stand
   transition, then `START` to activate the policy. Forward speed follows the
   joystick command, clipped to ±0.8 m/s; lateral and yaw commands are zero.

For midpoint-width tests, choose scenes `scene_depth_training_level_0p5.xml`
through `scene_depth_training_level_8p5.xml`. They are halfway between the
original integer levels, with full width `0.60 - level * 0.42 / 9` metres.
The nine scenes are installed and included in all workspace terrain selectors.
Level 8.5 is 20.333 cm, distinct from the custom 20.000 cm scene.
The standalone evaluator accepts `--levels 0.5 1.5 ... 8.5`.

For a 20 cm beam test, choose `scene_depth_training_width_20cm.xml` in the
terrain selector. This custom width lies between levels 8 (22.7 cm) and 9
(18 cm); beam length, platforms and deck height match those levels.
The simulator resolves scene names in the **installed** `unitree_mujoco` share
directory. After adding a source scene, install it there before launching:

```bash
# From /home/pro/unitree_ws; uses the existing build, no recompilation.
cmake --install build/unitree_mujoco
```

The 20 cm scene has already been installed in this workspace.

Stop the policy task before switching variants. The simulation and common
normalized-depth preprocessor can remain running. Simulator spring support is
still controlled by the simulator configuration/viewer; turn it off for
unsupported walking comparisons. These tasks do not automatically reproduce the
standalone benchmark's randomized resets or fixed 0.6 m/s command.

## Build

From `/home/pro/unitree_ws`:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
MAKEFLAGS=-j2 colcon build --packages-select legged_rl_deploy
source install/setup.bash
```

## Direct launch

From `/home/pro/unitree_ws`, in separate terminals:

```bash
src/legged_rl_deploy/scripts/run.sh lo humble ros2 run unitree_mujoco unitree_mujoco \
  --config src/legged_rl_deploy/policies/g1/beam/g1_beam_depth_distillation/mujoco_training.yaml \
  --scene scene_depth_training_level_0.xml --depth-camera
```

```bash
src/legged_rl_deploy/scripts/run.sh lo humble ros2 run legged_rl_deploy depth_image_preprocessor_node.py \
  --ros-args --params-file src/legged_rl_deploy/policies/g1/beam/g1_beam_depth_distillation/depth_image_preprocessor_variants_sim.yaml
```

```bash
src/legged_rl_deploy/scripts/run.sh lo humble ros2 run legged_rl_deploy legged_rl_deploy_node lo \
  src/legged_rl_deploy/policies/g1/beam/g1_beam_depth_distillation/V2/config_sim.yaml
```

Replace `V2` in the last command to select another variant. The shared depth
preprocessor is the same for all four. The original
`depth_image_preprocessor_sim.yaml` remains available for metre-valued legacy
models. The hardware `config.yaml` is not migrated by this change.

## Model and timestamp contract

| Input | Shape | YAML source |
|---|---|---|
| `history` | `[1,5978]` | `observations` |
| `depth` | `[1,1,120,212]` | `external.depth` |
| `fresh` | `[1]` | `image_fresh.depth` |
| `age` | `[1]` | `image_age.depth` |
| `hidden` | `[1,H]` | `state.hidden` |
| `reset` | `[1]`, `dtype: bool` | `policy_reset` |

H is 64 for V1/V4 and 128 for V2/V3. Outputs are `action_raw` to `actions`
and `hidden_next` to `state.hidden`. The old visual-cache/hold states are absent.
The 50 Hz policy uses the existing joint ordering, PD gains, action scaling and
clipping, H61 term-major observation history, and 0.9 s gait phase.

The preprocessor converts axial metres to clean normalized depth: invalid and
nonpositive values become 6 m, then clip to [0,6], nearest-neighbor resize from
424×240 to 212×120, then divide by 6. Capture headers are preserved. Its output
is `/g1_sim/beam_depth/normalized`, range [0,1]. MuJoCo preview maps that topic
with near=0/far=1. No stochastic V4 training corruption is applied at deployment.

For each inference, pixels and capture timestamp are copied under the same lock.
`fresh` is 1 on the first consumed frame after reset or a changed timestamp, and
0 when reusing the same frame. `age` is current ROS time minus capture time in
seconds. Frames are marked consumed only after successful inference. Timing
inputs reject missing/future/backward capture timestamps and capture ages beyond
`timeout_sec`; arrival timeout is checked as well. Camera and controller must
use the same ROS clock. Reset the policy after resetting a simulation clock.

`policy_reset` is true on the first successful inference after reset, including
after the startup model self-check. Recurrent buffers, observation history and
image-consumption tracking reset together. Float32 remains the default tensor
dtype; boolean input tensors are converted by the ORT/Torch backends. Model
outputs and recurrent-state buffers remain float32.

## Validation

The optional `BUILD_POLICY_CONTRACT_TESTS=ON` build provides:

- `temporal_contract`: legacy inference, observation history, clipping, phase,
  and reset regression checks.
- `beam_variant_contract`: all four real ONNX graphs compared with Python ORT
  reference outputs, including recurrent carry and reset; PolicySlot execution
  using ROS image messages; same-frame metadata, held frames, reset, and rejected
  stale/future/backward/missing timestamps.

`tests/generate_beam_reference.py` regenerates the numerical fixture using Python
ONNX Runtime and records the source model hashes. Tests require `WORKSPACE` to
point to the workspace and a sourced ROS environment. Use a separate ROS domain
for tests that publish synthetic images.

Verified on this workspace: both CTest suites pass; the live MuJoCo camera and
preprocessor deliver approximately 50 Hz; 20 paired raw/normalized frames match
the expected pixels and preserve capture timestamps. V2 entered
`HIGH_CONTROLLER` with valid output and no fault for a 3-second supported,
zero-command smoke test. The test injected button presses into relayed simulated
G1 LowState remote bytes (G1 reads that embedded remote, not the Go2 wireless
controller topic). This verifies integration, not unsupported walking performance.

The earlier 400-trial benchmark used a standalone Python evaluator. Its success
rates are not results from the ROS deployment path.
