# Run 13 — Power drill pivot-to-down (corrected) + contact_map sweep

## Why this run exists

Run 8 was the last full **morphology sweep** on the power drill. Run 12 was
the last **trajectory** experiment on the short-proximal variant. Both
specified the "pivot to down" wrist motion incompletely:

- **run8** drove only `pivot-delta-ry 1.20`. `rx` and `rz` were untouched.
- **run12** drove only `pivot-delta-rz 1.5708`. `rx` and `ry` were untouched.

The `open_flat` keyframe of `scene_power_drill_short_proximal.xml` starts the
wrist at `ctrl = (px, py, pz, rx, ry, rz) = (0.052, -0.038, -0.051, 1.6, 1.4,
-0.031)`. Pivot deltas are *added* to the initial ctrl per
[`src/morphohand/optimization/phase1_common.py:387-389`](../src/morphohand/optimization/phase1_common.py#L387-L389),
so to drive the wrist to `(rx, ry, rz) = (0, 0, π/2)` the three deltas must
be set:

| axis | initial | target | delta (run13)        |
|------|---------|--------|----------------------|
| rx   |  1.6    |  0     | **-1.6**             |
| ry   |  1.4    |  0     | **-1.4**             |
| rz   | -0.031  |  1.5708| **+1.6018**          |

For comparison:

| run | scope                  | delta_rx | delta_ry | delta_rz | resulting (rx, ry, rz)         |
|-----|------------------------|----------|----------|----------|--------------------------------|
| 8   | morphology sweep       | 0        | 1.20     | 0        | (1.6, 2.6, -0.031)             |
| 12  | trajectory only        | 0        | 0        | 1.5708   | (1.6, 1.4,  1.540)             |
| 13  | morphology sweep + CM  | -1.6     | -1.4     | 1.6018   | (0.0, 0.0,  1.5708)            |

## What this run includes

Two morphology sweeps over the **same** (corrected) pivot trajectory,
differing only in objective shape:

1. **`run13_drill_short_proximal_pivot_to_down_baseline`** — run8-style
   objective, no contact-target augmentation.
2. **`run13_drill_short_proximal_pivot_to_down_contact_map`** — same
   geometry/timing, plus the `contact_map` method
   ([`assets/contact_targets/power_drill_short_proximal.yaml`](../assets/contact_targets/power_drill_short_proximal.yaml))
   with `reward=10.0` / `distance_penalty=20.0`. These weights match what
   [`scripts/compare_methods.py:199-202`](compare_methods.py#L199-L202) and
   [`scripts/eval_suite.py:236-245`](eval_suite.py#L236-L245) use for the
   `contact_map` method, so the morphology sweep results are
   apples-to-apples with the existing benchmark harness.

Each variant gets its own foundational pass (3 seeds at `open_flat`) so
the morphology sweep adapts from a control vector that was itself trained
under that variant's objective.

## Improvements / checks since run 8

| improvement                       | wired? | how                                                                                                                                                                                                                                       |
|----------------------------------|--------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Frozen-scene protocol            | yes    | `phase1_optimize_grasp.py` freezes via `freeze_scene_for_eval` before constructing the evaluator; `run6_combined_multitask.py` writes a rigid scene per candidate via `write_rigid_scene_with_object_size` (same underlying baker).      |
| Contact-target patches           | new    | new CLI args `--contact-targets-yaml`, `--objective-weight-contact-target-reward`, `--objective-weight-contact-target-distance-penalty` on both `phase1_optimize_grasp.py` and `run6_combined_multitask.py`, wired into `Phase1GraspEvaluator(contact_target_set=...)`. |
| Correct pivot spec               | yes    | all three of `pivot-delta-rx / ry / rz` set; see table above.                                                                                                                                                                              |
| Side-by-side variant comparison  | yes    | `scripts/run6_analysis.py` invoked over both run dirs after the sweeps complete.                                                                                                                                                          |

## How to launch

```bash
bash scripts/run13_power_drill_pivot_to_down.sh
```

Environment knobs (all optional):

- `SAMPLES` (default `800`) — morphologies per sweep.
- `SEED` (default `13`) — RNG for morphology sampling.
- `RUN_FOUNDATIONAL` / `RUN_BASELINE` / `RUN_CONTACT_MAP` — set to `0` to
  skip a phase (e.g. resume after a partial run).
- `PIVOT_STEPS` / `PIVOT_RAMP_STEPS` — pivot phase length / ramp.
- `CT_REWARD` / `CT_DIST_PENALTY` — contact-target weights.

## Outputs

```
results/phase1/run13_power_drill_pivot_to_down_foundational/
    baseline/open_flat/seed_{0,1,2}/run_*/summary.json
    contact_map/open_flat/seed_{0,1,2}/run_*/summary.json
results/phase1/run13_drill_short_proximal_pivot_to_down_baseline/
    summary.json
    all_candidates_multitask.csv
    top5_videos/...
results/phase1/run13_drill_short_proximal_pivot_to_down_contact_map/
    summary.json   (contact_targets block records the YAML + weights)
    all_candidates_multitask.csv
    top5_videos/...
results/phase1/run13_drill_short_proximal_pivot_to_down_baseline/analysis/
    (run6_analysis.py output spanning both run dirs)
```
