# Sobol-128 finalists: uniform budget-1.0 deployment audit

**Date:** 2026-08-30
**Result:** four exports produced; zero plans pass all deployment gates; zero plans copied into the
runnable station directory.

This is the directly comparable audit requested after the pilot's 0.5 rad clip was found to be an
RL residual budget rather than an appropriate open-loop trajectory limit. Each finalist was
re-exported from its saved pilot scene and selected operating cell with a per-joint clip of exactly
1.0 rad.

## Gates

A plan is runnable only if it passes all of the following:

1. export through `scripts/real_v1_export_plan.py`, with `budget_rad: 1.0` recorded in metadata;
2. both the three-set-point chord and the 50 Hz CSV trajectory maintain at least 5 mm simulated
   finger-to-finger clearance at eight interpolation substeps;
3. `manta_hand.plan.HandPlan` reports no measured-rail or servo-command violation;
4. repeated 4,800-step / 9.6-second open-loop holds retain and align the tool (`cos >= 0.7`).

## Results

| design | chord / CSV clearance | driver gate | held | mean final cos | aligned + held | verdict |
|---|---:|---|---:|---:|---:|---|
| `sv1_u0060` | 9.9 / 9.9 mm | **fail:** middle yaw 3.88 deg past cap | 10/10 | 0.418 | 0/10 | reject b100 |
| `sv1_u0100` | 9.8 / 9.8 mm | pass | 0/10 | 0.000 | 0/10 | reject b100 |
| `sv1_w0099` | 8.2 / 8.2 mm | pass | 7/10 | 0.400 | 0/10 | reject b100 |
| `sv1_w0116` | 11.6 / 11.8 mm | **fail:** middle yaw 6.27 deg past cap | 5/10 | 0.345 | 0/10 | reject b100 |

All four clear the modeled finger geometry. That necessary gate does not rescue them: two are not
commandable under the hardware's conservative +/-70 degree yaw cap, and none produces a repeated
aligned hold at this shared budget.

The result does not invalidate the separately tuned exports `sv1_u0060_b75` and `sv1_u0100_b70`
already present in `docs/experiments/20260829-real_v1_deploy/deploy`. It shows that **budget 1.0 is
not a universal deployment setting**. Those lower-budget plans remain untested on hardware and
retain their existing warnings.

## Artifacts

- `*_plan.json`, `*_traj.csv`, `*_poses.txt`, and `*_build.txt`: all four budget-1.0 exports.
- [`clearance.json`](clearance.json): full chord/CSV traces.
- [`hold_4800_reps10.json`](hold_4800_reps10.json): 60 repeat rollouts, including the four
  finalists and the two saved anchors.
- [`validation.json`](validation.json): compact gate results and final dispositions.

`scripts/real_v1_export_plan.py` now accepts `--tag` and writes `meta.budget_rad`, so future budget
exports no longer depend on manual renaming to preserve their provenance.
