# Pre-recovery ONNX policy exports

These are single-step recurrent policy cores, not complete robot controllers.
Batch size is fixed at 1, opset 17, CPU float32 inputs (reset is bool).

Inputs:
- history: [1,5978], raw (not pre-normalized) term-major H61 proprioception. Embedded checkpoint normalizer is applied internally. Terms in order: base_ang_vel (3), projected_gravity (3), command (3), phase (2), joint_pos (29), joint_vel (29), actions (29). Each term stores its 61-frame history; match training observation construction/order/scaling exactly. V3 also retains this interface, although its actor uses the current frame after normalization.
- depth: [1,1,120,212], optical-axis depth normalized to [0,1] with 0–6 metre bounds; invalid/no-hit pixels must be 1. If input sensor reports ray distance, convert to axial depth first; do not apply that conversion again to axial depth. Camera calibration, mounting and image orientation must match training.
- fresh: [1] float32, 1 when the delivered depth frame changes, otherwise 0.
- age: [1] float32, elapsed seconds since the delivered frame was captured.
- hidden: [1,H] float32; H=64 for V1/V4, H=128 for V2/V3.
- reset: [1] bool; True on the current observation at episode/controller reset, False otherwise.

Outputs:
- action_raw: [1,29], raw policy action, NOT joint position targets or torque.
- hidden_next: [1,H], carry into hidden on the next control step.

Run at 50 Hz. Initialize hidden to zeros and reset=True on startup. Maintain the H61 observation history externally with exactly the training reset/history convention. Maintain previous action consistently with training. Depth caching, sensor timestamps, image preprocessing, action clipping, joint ordering, action scale/offset, PD gains and safety checks are NOT included in ONNX. These must be implemented in the deployment controller according to the training environment and robot configuration. Do not send action_raw directly to hardware actuators.

V4 has the same clean inference interface; training depth randomization is not exported as stochastic preprocessing.

Export validation: see [manifest.json](manifest.json) and the per-variant manifests. ONNX checker and ONNX ReferenceEvaluator numerical parity are checked against the saved PyTorch model over recurrent steps and reset cases. Those export checks are separate from the later ONNX Runtime and ROS integration tests documented in [README_SIM_VARIANTS.md](README_SIM_VARIANTS.md). They are not TensorRT or hardware validation. Confirm target-runtime compatibility and simulated closed-loop behavior before any robot deployment.

Source checkpoints remain unchanged. Export code is export_before_recovery_onnx.py and depends on the G1Beam checkout. No teacher file is needed by the exported ONNX graphs.
