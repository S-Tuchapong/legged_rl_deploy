# Bidirectional full-body TemporalV1 policies

`config.yaml` selects `GRU61Delay_bidirect_fullbody.onnx` and controls all 29 G1
joints (legs, waist, arms and wrists). The supplied TCN model has the same tensor
shapes and joint/action metadata; change `policy.model_path` to select it.

## Interface

- Input `obs`: [1, 5978]; output `actions`: [1, 29].
- Observation terms: body angular velocity (3), projected gravity (3), velocity
  command (3), gait phase (2), relative joint position (29), joint velocity (29),
  previous clipped raw action (29). Each term has 61 frames, oldest first, with
  repeat-first warmup: 98 x 61 = 5978 values, packed term-major.
- Joint order matches `legged_base/config/g1_29dof-config.yaml` (IDs 0–28).
- Default pose comes from ONNX metadata: hip pitch -0.1, knee 0.3, ankle pitch
  -0.2; all other joints zero. Observation offsets subtract this pose.
- Gains and action scales match exported metadata to its three-decimal precision.
  Full-precision leg constants are retained; ankle-roll scale is 0.12, waist,
  shoulder and elbow scales are 0.1, and wrist scales are 0.05.
- Raw actions are clipped before scaling/offset and before storage as last_action.
  Bounds are derived from the local G1Beam `g1.xml` joint ranges using the 0.9
  soft-limit factor and `(soft_limit - default_position) / action_scale`.
- Policy period is 0.02 s; low-level period is 0.005 s.

The existing command ranges, observation preprocessing and TemporalV1 phase/history
semantics are retained. These settings and the soft-limit convention are not fully
specified by the ONNX metadata; exact training equivalence requires the matching
training configuration. The copied 12-joint README previously in this directory
did not describe these models.

## Validation

Offline checks confirmed both ONNX tensor shapes and joint/action metadata,
29-element vectors, observation dimensions, inverse pose offsets, and all clipping
bounds. Both models produced finite 29-element outputs for a synthetic standing
observation using the bundled ONNX Runtime. No simulation or hardware trial was
performed. Full-body simulation requires the simulator's upper-body lock disabled.
