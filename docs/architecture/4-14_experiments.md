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

## Warp/GPU Backend Conclusion

After investigating mjwarp and comfree-warp throughput, the conclusion is:

- For current Phase 1 single-candidate evaluation loops, **CPU MuJoCo is the right default**.
- mjwarp/GPU only wins when effective batch per kernel launch is large (B x T >> 1000) and host sync is rare.
- The current host-driven loop with per-morphology evaluation keeps batch size small, so CPU remains faster.
- **Decision**: Stay on CPU MuJoCo for Phase 1. Only revisit GPU after building a batched evaluator where 100s-1000s of morphologies stay resident on GPU for full rollout.

## Run 5: Foundational Pose Adaptation Impact Study

Date: 2026-04-10

### Motivation

Run 3 used fixed foundational poses (FPs) found by CEM on the base morphology. As sampled morphologies diverge from base, these fixed FPs become suboptimal. Full CEM per morphology is too expensive (~51s each), so we need lightweight FP adaptation strategies.

The core question: **can cheap per-morphology FP refinement recover feasibility for morphologies that fail with fixed FPs?**

### Bugs Fixed Before Run 5

Three bugs were found in the untested `--fp-adaptation` code:

1. **Random morphology ordering defeated interval refresh**: Candidates near each other in iteration order were unrelated in 9D morphology space. Added `--morph-sort distance` to sort by L2 distance from base before evaluation.
2. **Adaptation path skipped multi-FP selection**: When `fp_adaptation != "none"`, only the adapted ctrl was evaluated. For cube with 3 loaded FPs, this could perform worse than baseline. Fixed to evaluate adapted ctrl AND all original FPs, picking the best.
3. **`sparse-per-morph` always started from fixed baseline**: Changed to use `interval_ctrl` (which may have been updated by prior interval refresh) instead of the immutable `baseline_ctrl`.

### Strategies Tested

Six sub-runs with identical seed, samples, and feasibility gates (same 500 morphologies):

| Sub-run | Strategy | Description |
|---------|----------|-------------|
| 5a | `none` (baseline) | Fixed FPs from run 3, evaluate all loaded poses per morphology, pick best feasible |
| 5b | `interval-initial-fp` i=50 | Every 50 morphologies (sorted by distance), re-run CEM (12 iter, 24 pop) warm-started from current best FP. Morphologies between triggers reuse the updated FP. |
| 5c | `interval-initial-fp` i=25 | Same as 5b but refresh every 25 morphologies (2x the CEM budget) |
| 5d | `interval-open` i=50 | Every 50 morphologies, re-run CEM from scratch (no warm-start). Tests whether warm-starting matters. |
| 5e | `sparse-per-morph` | For every morphology, run 1 CEM iteration with population 5 (try 5 random perturbations of current FP, keep best). Cheapest per-morphology method. |
| 5f | `local-perturbation` | New mode: for every morphology, systematically try +/- delta on each of 9 control dimensions (19 evaluations). More structured than random sampling. |

All adaptation modes also evaluate original FPs and pick the overall best (Bug 2 fix).

### Results

#### Feasibility (out of 500 per scene)

| Run | Mode | Cube | Prism1 | Prism2 | Prism3 | Total |
|-----|------|------|--------|--------|--------|-------|
| 5a | baseline | 420 | 317 | 293 | 412 | 1442 |
| 5b | interval-fp i=50 | 434 | 387 | 443 | 489 | 1753 |
| 5c | interval-fp i=25 | 462 | 399 | 422 | 477 | 1760 |
| 5d | interval-open i=50 | 422 | 323 | 305 | 415 | 1465 |
| 5e | sparse-5 | 431 | 436 | 412 | 479 | 1758 |
| 5f | local-perturb | 454 | 436 | 436 | 474 | 1800 |

#### Feasibility Recovery (infeasible in 5a that became feasible)

| Scene | 5b | 5c | 5d | 5e | 5f |
|-------|----|----|----|----|-----|
| cube | 71/80 | 73/80 | 68/80 | 11/80 | 34/80 |
| prism1 | 142/183 | 142/183 | 113/183 | 119/183 | 119/183 |
| prism2 | 192/207 | 175/207 | 124/207 | 119/207 | 143/207 |
| prism3 | 86/88 | 85/88 | 79/88 | 67/88 | 62/88 |

#### Mean Score Improvement vs Baseline (all 500 candidates)

| Scene | 5b | 5c | 5d | 5e | 5f |
|-------|----|----|----|----|-----|
| cube | +0.147 | +0.401 | +0.176 | +0.180 | +0.450 |
| prism1 | +0.801 | +0.993 | +0.737 | +1.360 | +1.453 |
| prism2 | +1.801 | +1.588 | +1.003 | +1.373 | +1.701 |
| prism3 | +1.006 | +0.831 | +0.405 | +0.845 | +0.856 |

#### Cost-Effectiveness

| Run | Mode | Total Feasible | Gain | Total Time (s) | Extra Time (s) | Gain/sec |
|-----|------|----------------|------|-----------------|-----------------|----------|
| 5a | baseline | 1442 | -- | 57.7 | -- | -- |
| 5b | interval-fp i=50 | 1753 | +311 | 300.9 | +243.2 | 1.28 |
| 5c | interval-fp i=25 | 1760 | +318 | 503.1 | +445.4 | 0.71 |
| 5d | interval-open i=50 | 1465 | +23 | 222.8 | +165.1 | 0.14 |
| 5e | sparse-5 | 1758 | +316 | 280.4 | +222.7 | **1.42** |
| 5f | local-perturb | 1800 | **+358** | 796.9 | +739.2 | 0.48 |

#### Adapted FP Win Rate

| Mode | Adapted Pose Chosen |
|------|-------------------|
| 5b (interval-fp i=50) | 52.7% |
| 5c (interval-fp i=25) | 52.9% |
| 5d (interval-open i=50) | 19.6% |
| 5e (sparse-5) | 85.5% |
| 5f (local-perturb) | 89.1% |

### Key Findings

1. **FP adaptation dramatically helps**: Every mode except interval-open beats baseline by 300+ feasible candidates (+22% improvement). The biggest impact is on prism2, the hardest scene (293 -> 443 with 5b).

2. **`interval-open` is nearly useless** (+23 gain): Starting CEM from scratch with a small budget produces poor FPs. The Bug 2 fix (also evaluate originals) prevents regression, but adaptation rarely wins (19.6%).

3. **Warm-starting matters**: 5b (warm-start) vs 5d (from-scratch) with same interval and budget: +311 vs +23 feasibility gain. The current FP is a much better CEM starting point than zero.

4. **Diminishing returns for interval frequency**: 5b (i=50) vs 5c (i=25) get nearly identical gains (+311 vs +318) but 5c costs 2x more. i=50 is sufficient.

5. **Per-morphology methods win on adapted-FP rate** (85-89%) but interval methods win on feasibility recovery for hard cases. Interval CEM can make larger jumps in control space; per-morphology methods only explore locally.

6. **Best cost-effectiveness**: `sparse-per-morph` (5e) at 1.42 gain/sec. Best absolute: `local-perturbation` (5f) at +358 total.

### Recommendation

For production morphology sweeps, use **`sparse-per-morph` with 5 samples** as the default. It provides the best cost-effectiveness (+316 feasible for only +223s extra). If maximum feasibility is needed and runtime is less constrained, `local-perturbation` adds +42 more at 3x cost.

### Commands

Shared flags for all sub-runs:

```
--seed 0 --samples 500 --backend mujoco \
--top-k-gifs 5 --refine-top-k --refine-pool-size 15 --refine-iterations 10 \
--cube-foundational-run-dir results/phase1/run_20260413_run3_cube_foundational_cem \
--prism-foundational-run-dir results/phase1/run_20260413_run3_prism_foundational_cem \
--cube-min-contacts 4 --prism-min-contacts 4 \
--cube-min-finger-contact-persistence 0.55 --prism-min-finger-contact-persistence 0.45 \
--cube-max-finger-yaw-drift 0.30 --prism-max-finger-yaw-drift 0.40 \
--cube-max-mean-tip-distance 0.012 --prism-max-mean-tip-distance 0.02 \
--lift-ramp-steps 100 \
--objective-weight-min-finger-persistence 2.4 \
--objective-weight-finger-persistence-imbalance-penalty 1.2 \
--objective-weight-finger-yaw-drift-penalty 1.0 \
--objective-weight-finger-flex-drift-penalty 0.5
```

Sub-run-specific flags:

- **5a**: `--fp-adaptation none --tag run5a_baseline`
- **5b**: `--fp-adaptation interval-initial-fp --fp-refresh-interval 50 --morph-sort distance --tag run5b_interval_fp_i50`
- **5c**: `--fp-adaptation interval-initial-fp --fp-refresh-interval 25 --morph-sort distance --tag run5c_interval_fp_i25`
- **5d**: `--fp-adaptation interval-open --fp-refresh-interval 50 --morph-sort distance --tag run5d_interval_open_i50`
- **5e**: `--fp-adaptation sparse-per-morph --fp-adapt-iterations 1 --fp-adapt-population 5 --fp-adapt-sigma-init 0.06 --tag run5e_sparse_5`
- **5f**: `--fp-adaptation local-perturbation --fp-adapt-sigma-init 0.06 --tag run5f_local_perturb`

### Artifacts

All sub-runs under `results/phase1/`:

- [run5a_baseline](results/phase1/run5a_baseline)
- [run5b_interval_fp_i50](results/phase1/run5b_interval_fp_i50)
- [run5c_interval_fp_i25](results/phase1/run5c_interval_fp_i25)
- [run5d_interval_open_i50](results/phase1/run5d_interval_open_i50)
- [run5e_sparse_5](results/phase1/run5e_sparse_5)
- [run5f_local_perturb](results/phase1/run5f_local_perturb)

## 2026-04-15 Evaluation Object-Set Expansion

Date: 2026-04-15

### Added/updated object XMLs

- `assets/objects/prism.xml`
  - 22.5 x 67.5 x 18 mm prism, mass 0.2 kg.
- `assets/objects/screwdriver_medium.xml`
  - 25 mm diameter, 100 mm length cylinder, mass 0.3 kg.
- `assets/objects/screwdriver_small.xml`
  - 8 mm diameter, 80 mm length cylinder, mass 0.075 kg.
- `assets/objects/power_drill.xml`
  - YCB drill mesh source retained; corrected relative mesh path.
  - Total mass 1.5 kg.
- `assets/objects/human_calf.xml`
  - Reworked to a connected kinematic chain:
    - tapered calf approximation (110 mm -> 70 mm over 300 mm),
    - ankle cylinder (70 mm x 70 mm),
    - foot box with rear heel extension,
    - hinge pivot 1.0 m from proximal calf side.

### Added scene XMLs

- `assets/mjcf/scene_screwdriver_medium.xml`
  - keyframes:
    - `open_flat`: near-contact pickup pose for horizontal tool.
    - `open_vertical`: near-contact wielding pose for vertical tool.
    - `open_90vertical`: near-contact pickup pose for 90-degree vertical orientation.
- `assets/mjcf/scene_screwdriver_small.xml`
  - keyframes:
    - `open_flat`
    - `open_vertical`
- `assets/mjcf/scene_power_drill.xml`
  - keyframes:
    - `open_flat`: index-forward support + thumb/middle wrapping posture.
    - `open_vertical`: vertical wielding-oriented pre-grasp posture.
  - palm plate thickness increased (`0.001` -> `0.003`) to improve palm-side contact stability studies.
- `assets/mjcf/scene_human_calf.xml`
  - keyframes:
    - `open_under_ankle`: under-ankle power-grasp initialization.
    - `open_lifted`: same grasp family with elevated calf hinge angle.
  - includes explicit calf hinge joint (`calf_rotate`) to represent trajectory-like lift behavior.

`assets/mjcf/scene_prism.xml` remains the prism baseline scene and already matches the requested prism dimensions and mass profile.

### Validation

Ran MuJoCo model-load smoke checks with `uv run python` across:

- all five object XMLs above, and
- `scene_prism.xml`, `scene_screwdriver_medium.xml`, `scene_screwdriver_small.xml`, `scene_power_drill.xml`, `scene_human_calf.xml`.

Result: all listed models load successfully with no XML or compile-time MuJoCo errors.

### Notes and current limitations

- Screwdriver and drill scenes include separate flat and vertical pre-grasp keyframes but do not yet implement a continuous in-scene transition controller between those poses.
- The calf scene models hinge-based lifting kinematics; full controller synthesis for dynamic anti-slip support (thumb-over / fingers-under through motion) is deferred to evaluation scripts.
- The drill still uses the raw YCB mesh, so practical grasp quality may benefit from future convex decomposition or a dual-cylinder proxy for faster contact tuning.

## Run 6: Standalone Pose Morphology Sweep (Screwdriver Medium)

Date: 2026-04-20

### Goal

Evaluate morphology robustness on `scene_screwdriver_medium.xml` for three standalone keyframes (no transition trajectory), using two FP adaptation modes:

- `sparse-per-morph` (sparse5)
- `interval-initial-fp` with `--fp-refresh-interval 50` and morphology-distance sorting

### Pipeline updates used by Run 6

1. Corrected keyframe controls for `open_flat`, `open_vertical`, and `open_90vertical` in `scene_screwdriver_medium.xml`.
2. Kept evaluator object-body compatibility by using object body name `cube` in the screwdriver scene.
3. Extended object extent inference in `src/morphohand/optimization/phase1_common.py` so distance proxies work for non-box geoms (cylinder/capsule/sphere/mesh fallback).
4. Added Run 6 utilities:
   - `scripts/run6_screwdriver_multikey_sampling.py`
   - `scripts/run6_analysis.py`

### Foundational pose search (quality grasps)

Per keyframe CEM was run with 2 seeds (`16` iterations, `48` population):

| Keyframe | Best seed score | Lift | Contacts |
|---|---:|---:|---:|
| `open_flat` | 5.7028 | 0.0496 | 3 |
| `open_vertical` | 6.9068 | 0.0501 | 6 |
| `open_90vertical` | 7.0911 | 0.0696 | 5 |

Artifacts: `results/phase1/run6_foundational/<keyframe>/seed_{0,1}/summary.json`

### Morphology sweep setup

- Samples per keyframe: `240`
- Keyframes: `open_flat`, `open_vertical`, `open_90vertical`
- Feasibility gates:
  - `mean_tip_distance <= 0.022`
  - `cube_tip_contacts >= 2`
- Morphology ordering: `--morph-sort distance`
- Modes:
  - `run6_sparse5`: `--fp-adaptation sparse-per-morph`
  - `run6_interval50`: `--fp-adaptation interval-initial-fp --fp-refresh-interval 50`

### Run 6 results

| Mode | Keyframe | Total | Feasible | Feasible rate | Mean score | Max score |
|---|---|---:|---:|---:|---:|---:|
| sparse-per-morph | open_flat | 240 | 213 | 0.887 | 4.3686 | 6.2963 |
| sparse-per-morph | open_vertical | 240 | 230 | 0.958 | 4.0126 | 7.3066 |
| sparse-per-morph | open_90vertical | 240 | 224 | 0.933 | 4.1554 | 7.4742 |
| interval-initial-fp | open_flat | 240 | 227 | 0.946 | 4.7423 | 7.2212 |
| interval-initial-fp | open_vertical | 240 | 223 | 0.929 | 3.5406 | 7.3075 |
| interval-initial-fp | open_90vertical | 240 | 213 | 0.887 | 3.3777 | 7.8153 |

### Interpretation

1. Both adaptation modes produce high standalone-pose feasibility across all three keyframes.
2. `interval-initial-fp` is more compute-efficient in adaptation frequency (5 adaptations per keyframe vs 240), while preserving competitive feasible counts.
3. `sparse-per-morph` gives stronger consistency on `open_vertical` and `open_90vertical` feasible-rate metrics in this run.
4. `open_90vertical` reaches the highest observed max score (7.8153 with interval mode), indicating high-quality but narrower high-performing regions.

### Analysis outputs

Run 6 includes:

- t-SNE embeddings over 9D morphology vectors (color by score)
- 2D thumb feature heat maps (`thumb_x`, `thumb_y`) for:
  - `cube_xy_drift`
  - `finger_flex_drift`
- 3D surface plots for the same two metrics

Artifacts:

- `results/phase1/run6_sparse5/analysis/`
- `results/phase1/run6_interval50/analysis/`
- `results/phase1/run6_sparse5/analysis/run6_analysis_summary.md`