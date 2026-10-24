# GRU61Delay — TemporalV1 T01 policy interface

Model moved from `../unitree_rl_mjlab_fix_upper/GRU61Delay.onnx`; a relative compatibility symlink remains there to avoid breaking existing model-path references. The old YAML was NOT corrected and remains incompatible (47 inputs); use this folder's YAML.

SHA256: `5bcef0dbf59e10d2cc8421a8042c805e0609c26d21266e2bb0b4bcec964a0611` (913910 bytes).

## Interface
- ONNX: one `obs` input, 2867 floats; one `actions` output, 12 floats. GRU is windowed, with no external hidden-state inputs.
- Observation normalization is embedded: graph begins `Sub(obs, obs_normalizer._mean)` then `Div`. Do NOT normalize again in deployment.
- 61 frames per term, oldest first, repeat-first warmup; every policy reset clears history, previous action and phase. Terms: body angular velocity (3), projected gravity (3), effective command (3), phase (2), relative joint position (12), joint velocity (12), previous clipped raw action (12). No base linear velocity.
- Existing runtime history/AppliedActionClip support is reused. New opt-in `gait_phase_temporal_v1` uses training's fixed 0.9 s cycle, [sin,cos], phase zero at reset, frozen while inactive, resumed (not reset) on motion. Existing `gait_phase_2` and `gait_phase_narrow_v5` were inspected but do not match these semantics.
- Commands clip to forward 0–0.8 m/s, lateral/yaw zero; planar norm + absolute yaw <=0.1 becomes exactly zero before observation/phase.
- Actions: clip raw output using training soft-limit-derived bounds, store that as last_action, then exact training scale and default-pose offset. Joint order: left hip pitch/roll/yaw, knee, ankle pitch/roll, then right in the same order. Source: G1Beam `src/assets/robots/unitree_g1/xmls/g1.xml`, factor 0.9, saved GRU61Delay env.yaml; matches local robot joint IDs 0–11.
- No injected sensor noise, random delay or encoder bias on hardware. Actual calibrated joint readings are expected. Delay training does not justify adding deployment latency.

## Preserved hardware settings / limitations
Kp/stiffness, Kd/damping, llc_config_file, ll_dt, policy_dt, action joint IDs and offsets are unchanged from original `unitree_rl_mjlab_fix_upper/config.yaml`. No Ki existed. Original YAML had no fixed_joints; none were invented/copied. Low-level/FixStand upper-body behavior is unchanged and must be reviewed separately before hardware use. Dynamics and robot calibration may differ from simulation; interface checks are NOT hardware safety certification. Model training provenance/checkpoint iteration is not encoded reliably in this supplied file.

## Offline checks (no controller or hardware)
Build runtime and test after sourcing the existing ROS/workspace environment:

```bash
cmake -S /home/pro/unitree_ws/src/legged_rl_deploy -B /home/pro/unitree_ws/build/legged_rl_deploy -DBUILD_POLICY_CONTRACT_TESTS=ON
cmake --build /home/pro/unitree_ws/build/legged_rl_deploy --target temporal_contract_test policy_contract_test -j2
WORKSPACE=/home/pro/unitree_ws ROS_LOCALHOST_ONLY=1 LD_LIBRARY_PATH=/home/pro/unitree_ws/build/legged_rl_deploy:$LD_LIBRARY_PATH ctest --test-dir /home/pro/unitree_ws/build/legged_rl_deploy -R '^(temporal_contract|policy_contract)$' --output-on-failure
```

Passed: real ONNX inference/reset parity; all 2867 packed entries over 420 synthetic ticks; deadband boundary, stop/restart phase, action clipping/scale/offset/last_action, reset/decimation; pre-existing narrow-v5 and legacy regression tests. Tests create a localhost-only ROS node for runtime dependencies, never construct a hardware adapter or launch a controller. Build tree updated, no controller install/launch or training changes performed.
