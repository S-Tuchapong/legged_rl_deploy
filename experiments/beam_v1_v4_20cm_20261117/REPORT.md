# G1 beam: 20 cm evaluation

Linked CSVs, plots, videos, trajectories, and reproduction scripts are local generated artifacts and are not included in this documentation-only commit.

40 trials: V1–V4, seeds 0–9 per model. Fixed forward command **0.6 m/s**, zero lateral/yaw command.

Uses the same standalone MuJoCo 3.5.0 evaluator, clean depth, nominal controller settings, model files, robot model, start-position seeds and success criteria as the original 400-trial benchmark. This is not a ROS walking benchmark.

| Model | Near far platform | Onto far platform |
|---|---:|---:|
| V1 | 0/10 (0%) | 0/10 (0%) |
| V2 | 3/10 (30%) | 3/10 (30%) |
| V3 | 0/10 (0%) | 0/10 (0%) |
| V4 | 0/10 (0%) | 0/10 (0%) |

## Comparison with adjacent widths

Cells show near / onto success counts out of 10. Existing 22.7 cm and 18 cm results are reused from the original benchmark.

| Model | 22.7 cm (level 8) | 20 cm (new) | 18 cm (level 9) |
|---|---:|---:|---:|
| V1 | 4 / 0 | 0 / 0 | 0 / 0 |
| V2 | 10 / 10 | 3 / 3 | 0 / 0 |
| V3 | 0 / 0 | 0 / 0 | 0 / 0 |
| V4 | 0 / 0 | 0 / 0 | 0 / 0 |

## Videos

Selection uses near success only, as requested. For this fixed-width follow-up, a 100% category exists only at 10/10 near; an any-success category exists at >0/10 near. Each qualifying case uses the lowest successful seed and replays that exact scored trajectory. No additional attempts were used.

- V1, hardest_100pct: no qualifying run/category.
- V1, hardest_any_success: no qualifying run/category.
- V2, hardest_100pct: no qualifying run/category.
- V2, hardest_any_success: [seed 2](videos/V2_hardest_any_success_L20cm_seed2.mp4).
- V3, hardest_100pct: no qualifying run/category.
- V3, hardest_any_success: no qualifying run/category.
- V4, hardest_100pct: no qualifying run/category.
- V4, hardest_any_success: no qualifying run/category.

## Scoring and reproducibility

- Near: pelvis x ≥ 4.10 m (within 0.30 m of the platform edge), |y| ≤ 0.40 m, z > 0.45 m, tilt < 60°, recent course foot contact, and no earlier fall or ground contact.
- Onto: pelvis over the far platform and both feet have contacted it, with upright/support conditions satisfied. Near success remains recorded if the robot subsequently falls.
- Timeout 15 seconds. No elastic support, observation noise, sensor delay or dynamics randomization. Seeds randomize the same platform-only start x as the original study.
- All 40 trial keys, finite saved trajectories, increasing timestamps, model/robot hashes, matched initial positions, and the onto-implies-near condition were verified.
- Rates are empirical over 10 seeds. Differences do not establish guaranteed performance on new seeds or hardware.

[Summary CSV](summary.csv) · [Per-trial CSV](trials.csv) · [Protocol](protocol_V2.json)

Reproduction, from the repository root:

```bash
for v in V1 V2 V3 V4; do
  /tmp/beam_eval_venv/bin/python scripts/eval_g1_beam_variants.py \
    --output experiments/beam_20cm_reproduction --variant "$v" --levels 20cm
done
/tmp/beam_eval_venv/bin/python scripts/eval_g1_beam_variants.py \
  --output experiments/beam_20cm_reproduction --levels 20cm --summarize --videos
```
