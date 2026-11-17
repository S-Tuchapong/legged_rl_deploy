# G1 beam midpoint-width experiment

Linked CSVs, plots, videos, trajectories, and reproduction scripts are local generated artifacts and are not included in this documentation-only commit.

**360 new trials:** four models × nine midpoint widths × ten seeds (0–9), fixed command **(0.6, 0, 0)**. Only beam width changes.

## Midpoint results

Cells show **near / onto successes out of 10**. Fractional levels are exactly halfway between adjacent original widths.

| Midpoint level | Width (cm) | V1 | V2 | V3 | V4 |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 57.667 | 10 / 10 | 10 / 10 | 10 / 10 | 0 / 0 |
| 1.5 | 53.000 | 10 / 10 | 10 / 10 | 10 / 10 | 0 / 0 |
| 2.5 | 48.333 | 10 / 10 | 10 / 10 | 10 / 10 | 0 / 0 |
| 3.5 | 43.667 | 10 / 10 | 10 / 10 | 10 / 10 | 0 / 0 |
| 4.5 | 39.000 | 10 / 10 | 10 / 10 | 10 / 10 | 0 / 0 |
| 5.5 | 34.333 | 10 / 10 | 10 / 10 | 10 / 10 | 0 / 0 |
| 6.5 | 29.667 | 10 / 8 | 10 / 10 | 10 / 10 | 0 / 0 |
| 7.5 | 25.000 | 0 / 0 | 10 / 10 | 5 / 4 | 0 / 0 |
| 8.5 | 20.333 | 0 / 0 | 7 / 7 | 0 / 0 | 0 / 0 |

![Near-success rates versus width](width_sensitivity.png)

[Enlarged view of 18–34 cm widths](width_sensitivity_narrow.png). Original integer levels and the previous 20 cm study are reused, not rerun. The new level 8.5 is **20.333 cm**, which is distinct from the earlier **20.000 cm** case. All three studies together contain 800 trials.

## What this says about overfitting

The available local training implementation explicitly sets each row’s difficulty to `row/(layout_rows−1)` in `CourseTerrainCfg.function`; Phase II then computes the beam width from that value. The supplied checkpoint parameters specify 10 rows and full-width bounds 0.60–0.18 m. This supports examining the midpoint widths as interpolation between that discrete width grid. Copies of the relevant local training source are in `source/`; this is not an independent reconstruction of every checkpoint’s training run.

A useful descriptive signal is a midpoint doing worse than **both** neighboring widths, rather than simply getting worse as the beam narrows. We also check matched seeds that succeed at both endpoints but fail at the midpoint. These are exploratory comparisons, not corrected significance tests.

**Near: 1 of 36 model/midpoint cells fall below both adjacent endpoint success counts.**

| Model | Midpoint | Wider / midpoint / narrower successes | Seeds succeeding at both endpoints but failing at midpoint |
|---|---:|---:|---|
| V1 | 7.5 (25.000 cm) | 4 / 0 / 4 | 2 |

**Onto: 0 of 36 model/midpoint cells fall below both adjacent endpoint success counts.**

A drop at an intermediate width would indicate local width sensitivity and motivate further investigation; it does **not** identify its cause as overfitting. Nonlinear foot placement/contact dynamics, the deployment robot’s differences from training, and variation across only ten start positions can also produce nonmonotonic success rates. Conversely, smooth performance on these midpoints would not rule out overfitting to other environment properties. V4’s original all-zero results provide no successful endpoint baseline for this diagnostic.

For a stronger follow-up, use additional held-out reset seeds at any observed dips and compare with policies trained using continuously randomized beam widths. Do not infer a sharp width limit or an overfitting diagnosis from this ten-seed sweep alone.

## Videos

Video categories are evaluated over the **nine new midpoint terrains** only, using near success. The first qualifying seed is replayed from its saved scored trajectory; no extra attempts are used.

- V1, hardest_100pct: [level 6.5, seed 0](videos/V1_hardest_100pct_L6.5_seed0.mp4).
- V1, hardest_any_success: [level 6.5, seed 0](videos/V1_hardest_any_success_L6.5_seed0.mp4).
- V2, hardest_100pct: [level 7.5, seed 0](videos/V2_hardest_100pct_L7.5_seed0.mp4).
- V2, hardest_any_success: [level 8.5, seed 0](videos/V2_hardest_any_success_L8.5_seed0.mp4).
- V3, hardest_100pct: [level 6.5, seed 0](videos/V3_hardest_100pct_L6.5_seed0.mp4).
- V3, hardest_any_success: [level 7.5, seed 0](videos/V3_hardest_any_success_L7.5_seed0.mp4).
- V4, hardest_100pct: no qualifying level.
- V4, hardest_any_success: no qualifying level.

## Protocol and validation

- Same standalone MuJoCo 3.5.0 evaluator as the original benchmark; these are not ROS deployment walking results. Raw ONNX models, workspace robot, action scales/clipping, PD gains, camera, clean depth, and initial-position seeds match the original experiment.
- Near: pelvis x ≥ 4.10 m, |y| ≤ 0.40 m, z > 0.45 m, tilt < 60°, recent course foot contact, and no prior fall/ground contact. Onto: pelvis over the far platform and both feet have contacted it under the same upright/support conditions. Near remains recorded if the robot falls before onto.
- Policy/depth 50 Hz, PD 200 Hz, physics 1 kHz; 15 s timeout. No elastic support, observation noise, stochastic depth corruption, or dynamics randomization. Seeds vary start x on the start platform, with y/yaw/initial velocities zero.
- Validated all 360 keys, finite trajectories, increasing timestamps, matching model/robot hashes and start positions, and onto-implies-near. All nine installed scenes also load with the ROS simulator’s MuJoCo 3.3.6; only beam width differs from level 0.
- [Summary CSV](summary.csv), [all trials](trials.csv), [adjacent comparisons](adjacent_comparisons.csv), [combined 800-trial summary](combined_summary.csv), [study plan](study_plan.json), and [protocol](protocol_V2.json). Saved trajectories are in `trials/`.
- Exportable plot: [PNG](width_sensitivity.png) and [PDF](width_sensitivity.pdf). Bars are pointwise Wilson intervals for each ten-seed proportion; they are not intervals for paired differences or a multiple-comparison test.

Reproduce from the repository root:

```bash
for v in V1 V2 V3 V4; do
  /tmp/beam_eval_venv/bin/python scripts/eval_g1_beam_variants.py \
    --output experiments/beam_midpoints_reproduction --variant "$v" \
    --levels 0.5 1.5 2.5 3.5 4.5 5.5 6.5 7.5 8.5
done
/tmp/beam_eval_venv/bin/python scripts/eval_g1_beam_variants.py \
  --output experiments/beam_midpoints_reproduction \
  --levels 0.5 1.5 2.5 3.5 4.5 5.5 6.5 7.5 8.5 --summarize --videos
```
