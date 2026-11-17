# G1 beam V1–V4 evaluation — 17 November 2026

Linked CSVs, plots, videos, trajectories, and reproduction scripts are local generated artifacts and are not included in this documentation-only commit.

400 trials: four models × ten terrain levels × ten seeds (0–9). Fixed body-frame command **(0.6, 0, 0)** m/s and rad/s throughout.

## Success rates

Each cell is **near / onto**, in percent (10 trials per cell). Higher levels have narrower beams.

| Level | Beam width (m) | V1 | V2 | V3 | V4 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.600 | 100% / 100% | 100% / 100% | 100% / 100% | 0% / 0% |
| 1 | 0.553 | 100% / 100% | 100% / 100% | 100% / 100% | 0% / 0% |
| 2 | 0.507 | 100% / 100% | 100% / 100% | 100% / 100% | 0% / 0% |
| 3 | 0.460 | 100% / 100% | 100% / 100% | 100% / 100% | 0% / 0% |
| 4 | 0.413 | 100% / 100% | 100% / 100% | 100% / 100% | 0% / 0% |
| 5 | 0.367 | 100% / 100% | 100% / 100% | 100% / 100% | 0% / 0% |
| 6 | 0.320 | 100% / 100% | 100% / 100% | 100% / 100% | 0% / 0% |
| 7 | 0.273 | 40% / 10% | 100% / 100% | 80% / 70% | 0% / 0% |
| 8 | 0.227 | 40% / 0% | 100% / 100% | 0% / 0% | 0% / 0% |
| 9 | 0.180 | 0% / 0% | 0% / 0% | 0% / 0% | 0% / 0% |

## Selected videos

Selection uses **near** success only. Each video replays a saved scored trajectory; there are no additional attempts or cherry-picked reruns. The lowest successful seed at the selected level is used. A category without a qualifying level has no video.

| Model | Hardest level with 10/10 near | Hardest level with any near success |
|---|---|---|
| V1 | [Level 6, seed 0](videos/V1_hardest_100pct_L6_seed0.mp4) (10/10 near) | [Level 8, seed 0](videos/V1_hardest_any_success_L8_seed0.mp4) (4/10 near) |
| V2 | [Level 8, seed 0](videos/V2_hardest_100pct_L8_seed0.mp4) (10/10 near) | [Level 8, seed 0](videos/V2_hardest_any_success_L8_seed0.mp4) (10/10 near) |
| V3 | [Level 6, seed 0](videos/V3_hardest_100pct_L6_seed0.mp4) (10/10 near) | [Level 7, seed 0](videos/V3_hardest_any_success_L7_seed0.mp4) (8/10 near) |
| V4 | None | None |

## Protocol and interpretation

- **Near:** pelvis reaches x ≥ 4.10 m, within 0.30 m of the far platform’s leading edge (x=4.40 m); |y| ≤ 0.40 m, pelvis z > 0.45 m, tilt < 60°, and a foot has contacted the course in the preceding 0.12 s. No prior fall or ground contact.
- **Onto:** near has been reached, pelvis lies over the far platform (x=4.40–5.20 m, |y|≤0.40 m), both feet have contacted it, and the same upright/support conditions hold. This measures arrival, not sustained standing or a dwell interval.
- Criteria are latched: reaching near and subsequently falling still counts for near, but not onto. A trial stops on onto success, any ground contact, pelvis z<0.40 m, tilt>60°, |y|>1 m, or 15 s.
- The same seeds produce the same platform-only random start x for every model and level. Start x is uniform over [−0.290, 0.264] m; y=0, yaw=0, nominal joints, zero initial velocities, pelvis z=0.793 m. No settling controller or spring support.
- Uses the workspace `unitree_mujoco` G1 robot and the ten existing training-terrain XMLs: 4 m beam, 0.8 m square end platforms, 0.55 m deck height above ground. This is a standalone MuJoCo policy test, **not** a test of the ROS deployment state machine, communication latency, or physical hardware.
- 50 Hz ONNX inference, 200 Hz PD, 1 kHz physics. Training action scales/offsets and configured PD gains; actuator torque limits remain active. H61 term-major histories are backfilled with the first frame; recurrent state resets to zero.
- All models receive clean 212×120 optical-axis depth at 50 Hz, clipped to 0–6 m and normalized to [0,1], with fresh=1 and age=0. Training camera pose/FOV and geometry groups 0/1/2/4 are used. No observation noise, random sensor delays, depth corruption, or dynamics randomization. V4 was trained with depth randomization but is evaluated with the same clean input as the others.
- Rates are empirical results from ten start-position seeds per cell. “100%” means 10/10 in this sample, not a guarantee on new seeds or real hardware. Width-dependent rates need not be monotonic.

## Reproducibility and validation

- The four seed-0 smoke trajectories reproduced bit-for-bit in the main run. The four training parameter sets have identical observation/action/reset definitions. ONNX input dimensions and joint ordering were checked before rollout.
- All 400 result keys and matching trajectories were validated; all states are finite, all timestamps increase, all onto successes also satisfy near, and every model SHA-256 matches its supplied manifest.
- [Per-trial CSV](trials.csv), [summary CSV](summary.csv), [summary JSON](summary.json), [video selection](video_selection.json), and `trials/` retain all scored outcomes and trajectories.
- `source/` contains the evaluator snapshot, controller config snapshot, and installed Python versions. Per-variant protocol JSONs and logs are in this folder. Each trial records the ONNX, robot XML, and terrain XML SHA-256.

Re-run from the `legged_rl_deploy` repository root (Python requires mujoco, onnxruntime, numpy, PyYAML, imageio, imageio-ffmpeg, Pillow):

```bash
for v in V1 V2 V3 V4; do
  MUJOCO_GL=egl /tmp/beam_eval_venv/bin/python scripts/eval_g1_beam_variants.py \
    --output experiments/beam_v1_v4_20261117/reproduction --variant "$v"
done
/tmp/beam_eval_venv/bin/python scripts/eval_g1_beam_variants.py \
  --output experiments/beam_v1_v4_20261117/reproduction --summarize --videos
```
