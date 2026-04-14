# 4-14 Experiments: Warp Throughput vs Metric Fidelity

Date: 2026-04-14

## Goal

1. Validate whether reduced warp->MuJoCo sync frequency changes grasp outcome for a fixed foundational pose.
2. Compare `mjwarp` and `comfree-warp` on the same fixed-control foundational grasp.
3. Select a production configuration for the long cube+prism multiscene run.

## FP-only backend sweep

Setup:

- Scene/keyframe/control source:
  - `results/phase1/run_20260413_phaseA_fp_conditioned/foundational_diffmjx_mvp/summary.json`
- Evaluated with `keyframe=foundational` and fixed 9D finger control.
- Output:
  - `results/phase1/run_20260414_run4_prism/fp_backend_interval_sweep/fp_backend_interval_sweep.csv`
  - `results/phase1/run_20260414_run4_prism/fp_backend_interval_sweep/fp_backend_interval_sweep_summary.json`

### Results table

| backend | N (sync/sample) | eval seconds | score | lift | contacts | contact persistence | min finger persistence |
|---|---:|---:|---:|---:|---:|---:|---:|
| mujoco | 1 | 0.034 | 7.325236 | 0.050146 | 6.0 | 1.0 | 1.0 |
| mjwarp | 1 | 2.912 | 7.325102 | 0.050143 | 6.0 | 1.0 | 1.0 |
| mjwarp | 4 | 1.782 | 7.325100 | 0.050143 | 6.0 | 1.0 | 1.0 |
| mjwarp | 8 | 1.597 | 7.325023 | 0.050140 | 6.0 | 1.0 | 1.0 |
| comfree-warp | 1 | 2.638 | 1084.906559 | 104.660309 | 3.0 | 1.0 | 0.0 |
| comfree-warp | 4 | 1.567 | 1084.906559 | 104.660309 | 3.0 | 1.0 | 0.0 |
| comfree-warp | 8 | 1.342 | 1084.906559 | 104.660309 | 3.0 | 1.0 | 0.0 |

### Interpretation

- `mjwarp` preserves the foundational grasp metrics across `N=1,4,8` with only tiny numerical drift.
- `mjwarp` at `N=4` gives a strong speedup vs `N=1` while keeping metrics effectively unchanged.
- `comfree-warp` in this scene/configuration produced physically implausible score/lift magnitudes, so it is not selected for quality-preserving production runs.

## Selected production configuration

Chosen for long multiscene run:

- backend: `mjwarp`
- speed mode: `balanced`
- metric collection: `sampled`
- intervals: `backend_sync_interval=4`, `metric_sample_interval=4`

Rationale:

1. Best quality-preserving speedup from the FP sweep.
2. Minimal metric degradation for foundational grasp checks.
3. Avoids unstable scaling seen with `comfree-warp` in this workload.

## Re-queued long cube+prisms run

Queued tag:

- `run_20260414_run4_pollard_multiscene_500_mjwarp_balanced_n4`

Command profile:

- 500 samples
- cube + prism multiscene constraints from run3
- top-k refinement enabled
- `mjwarp` + balanced sampled speed mode

## Notes on fidelity

Reducing sync/sample interval does not change backend integration itself in this open-loop Phase 1 setup; it changes how often metrics are observed from MuJoCo state. This is why core final metrics stayed effectively constant for `mjwarp` while runtime dropped.

## Next steps plan

1. Let the long run finish and collect per-scene summaries.
2. Re-evaluate top candidates from the long run with `speed_mode=accurate` (`N=1`) for final ranking sanity check.
3. Generate/report GIFs only from the final rechecked top candidates.
4. If `comfree-warp` is revisited, do targeted parameter tuning (`stiffness`, `damping`) on a small calibration set before large-scale sweeps.
5. Add an automated two-pass mode in `phase1_pollard_multiscene.py`: fast search pass + accurate rerank pass.