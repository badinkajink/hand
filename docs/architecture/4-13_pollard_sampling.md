# 4-13 Pollard Sampling (Large-Scale)

Date: 2026-04-13

## Goal

Run a larger Pollard-style morphology experiment with 500 sampled hands, evaluate across cube and prism scenes, add foundational grasp keyframes to generated scenes, and export top-5 grasp GIFs per object type.

## Key Correction Before Large Run

Prism foundational poses from MJX-autodiff were weak. We first generated stronger prism foundational controls using CEM:

- Source run: [results/phase1/run_20260413_prism_y_sweep_cem](results/phase1/run_20260413_prism_y_sweep_cem)
- Prism foundational scores:
  - y=0.0250: 3.1085
  - y=0.0300: 3.1254
  - y=0.0350: 3.0793
- All three CEM prism foundational runs reached 3 contacts with ~0.055 lift.

## Commands Used

CEM prism foundational sweep:

```bash
MUJOCO_GL=egl bash -lc 'for y in 0.0250 0.0300 0.0350; do
  tag="y$(echo "$y" | tr "." "d")"
  uv run python scripts/phase1_optimize_grasp.py \
    --scene-xml "assets/mjcf/generated/scene_prism_x0.0200_y${y}_z0.0200.xml" \
    --optimizer cem \
    --iterations 20 \
    --population 36 \
    --elite-fraction 0.2 \
    --sigma-init 0.2 \
    --seed 0 \
    --skip-gif \
    --output-dir results/phase1/run_20260413_prism_y_sweep_cem \
    --tag "$tag"
done'
```

Large multi-scene Pollard run (500 samples + top-5 GIFs per scene):

```bash
MUJOCO_GL=egl uv run python scripts/phase1_pollard_multiscene.py \
  --samples 500 \
  --top-k-gifs 5 \
  --prism-foundational-run-dir results/phase1/run_20260413_prism_y_sweep_cem \
  --tag run_20260413_pollard_multiscene_500_cem
```

## Artifacts

Main run directory:

- [results/phase1/run_20260413_pollard_multiscene_500_cem](results/phase1/run_20260413_pollard_multiscene_500_cem)

Global summary:

- [results/phase1/run_20260413_pollard_multiscene_500_cem/summary.json](results/phase1/run_20260413_pollard_multiscene_500_cem/summary.json)

Per-scene CSVs and plots:

- Cube:
  - [results/phase1/run_20260413_pollard_multiscene_500_cem/cube/all_candidates.csv](results/phase1/run_20260413_pollard_multiscene_500_cem/cube/all_candidates.csv)
  - [results/phase1/run_20260413_pollard_multiscene_500_cem/cube/feasible_candidates.csv](results/phase1/run_20260413_pollard_multiscene_500_cem/cube/feasible_candidates.csv)
  - [results/phase1/run_20260413_pollard_multiscene_500_cem/cube/pareto_front.csv](results/phase1/run_20260413_pollard_multiscene_500_cem/cube/pareto_front.csv)
  - [results/phase1/run_20260413_pollard_multiscene_500_cem/cube/top5_with_gifs.csv](results/phase1/run_20260413_pollard_multiscene_500_cem/cube/top5_with_gifs.csv)
- Prism1 (y=0.0250):
  - [results/phase1/run_20260413_pollard_multiscene_500_cem/prism1/all_candidates.csv](results/phase1/run_20260413_pollard_multiscene_500_cem/prism1/all_candidates.csv)
  - [results/phase1/run_20260413_pollard_multiscene_500_cem/prism1/feasible_candidates.csv](results/phase1/run_20260413_pollard_multiscene_500_cem/prism1/feasible_candidates.csv)
  - [results/phase1/run_20260413_pollard_multiscene_500_cem/prism1/pareto_front.csv](results/phase1/run_20260413_pollard_multiscene_500_cem/prism1/pareto_front.csv)
  - [results/phase1/run_20260413_pollard_multiscene_500_cem/prism1/top5_with_gifs.csv](results/phase1/run_20260413_pollard_multiscene_500_cem/prism1/top5_with_gifs.csv)
- Prism2 (y=0.0300):
  - [results/phase1/run_20260413_pollard_multiscene_500_cem/prism2/all_candidates.csv](results/phase1/run_20260413_pollard_multiscene_500_cem/prism2/all_candidates.csv)
  - [results/phase1/run_20260413_pollard_multiscene_500_cem/prism2/feasible_candidates.csv](results/phase1/run_20260413_pollard_multiscene_500_cem/prism2/feasible_candidates.csv)
  - [results/phase1/run_20260413_pollard_multiscene_500_cem/prism2/pareto_front.csv](results/phase1/run_20260413_pollard_multiscene_500_cem/prism2/pareto_front.csv)
  - [results/phase1/run_20260413_pollard_multiscene_500_cem/prism2/top5_with_gifs.csv](results/phase1/run_20260413_pollard_multiscene_500_cem/prism2/top5_with_gifs.csv)
- Prism3 (y=0.0350):
  - [results/phase1/run_20260413_pollard_multiscene_500_cem/prism3/all_candidates.csv](results/phase1/run_20260413_pollard_multiscene_500_cem/prism3/all_candidates.csv)
  - [results/phase1/run_20260413_pollard_multiscene_500_cem/prism3/feasible_candidates.csv](results/phase1/run_20260413_pollard_multiscene_500_cem/prism3/feasible_candidates.csv)
  - [results/phase1/run_20260413_pollard_multiscene_500_cem/prism3/pareto_front.csv](results/phase1/run_20260413_pollard_multiscene_500_cem/prism3/pareto_front.csv)
  - [results/phase1/run_20260413_pollard_multiscene_500_cem/prism3/top5_with_gifs.csv](results/phase1/run_20260413_pollard_multiscene_500_cem/prism3/top5_with_gifs.csv)

## Foundational Keyframes Added

Each generated scene XML now includes a keyframe named foundational representing the settled grasp pose using the selected foundational finger controls.

Example generated scene set:

- [results/phase1/run_20260413_pollard_multiscene_500_cem/prism2/generated_mjcf](results/phase1/run_20260413_pollard_multiscene_500_cem/prism2/generated_mjcf)

## Top-5 GIFs Per Scene

- Cube GIF folder:
  - [results/phase1/run_20260413_pollard_multiscene_500_cem/cube/top_gifs](results/phase1/run_20260413_pollard_multiscene_500_cem/cube/top_gifs)
- Prism1 GIF folder:
  - [results/phase1/run_20260413_pollard_multiscene_500_cem/prism1/top_gifs](results/phase1/run_20260413_pollard_multiscene_500_cem/prism1/top_gifs)
- Prism2 GIF folder:
  - [results/phase1/run_20260413_pollard_multiscene_500_cem/prism2/top_gifs](results/phase1/run_20260413_pollard_multiscene_500_cem/prism2/top_gifs)
- Prism3 GIF folder:
  - [results/phase1/run_20260413_pollard_multiscene_500_cem/prism3/top_gifs](results/phase1/run_20260413_pollard_multiscene_500_cem/prism3/top_gifs)

## Results Snapshot

Feasibility rates and Pareto sizes:

- Cube: 324/500 feasible (64.8%), Pareto size 25
- Prism1: 493/500 feasible (98.6%), Pareto size 40
- Prism2: 500/500 feasible (100.0%), Pareto size 45
- Prism3: 500/500 feasible (100.0%), Pareto size 42

Top-ranked morphology/grasp per scene (from top5_with_gifs.csv):

- Cube rank 1: candidate 91, score 5.2406, lift 0.1160, contacts 3
- Prism1 rank 1: candidate 264, score 3.3081, lift 0.0609, contacts 3
- Prism2 rank 1: candidate 114, score 3.8539, lift 0.0765, contacts 3
- Prism3 rank 1: candidate 369, score 3.1184, lift 0.0558, contacts 3

## Analysis

- The CEM-first correction was necessary and effective: prism foundational poses became high-quality and yielded very high feasibility in the subsequent Pollard sweep.
- The same morphology pool can robustly cover all prism variants under this setup.
- Cube remains the hardest of the four scenes under the current strict threshold (mean tip distance <= 0.012 and contacts >= 2), which is expected from requiring compatibility with two cube foundational controls.
- Stronger candidates exist beyond the original baseline, especially for cube and prism2.

## Recommended Follow-Up

1. Evaluate cross-scene transfer of top candidates (for example, top 10 from each scene evaluated on all 4 scenes).
2. Add a stability filter on cube velocity for top-k promotion before hardware-oriented ranking.
3. Build a combined multi-object Pareto front using objective aggregation across cube + prisms.

## Run 2 (Stability + Better Foundations)

### Why a Run 2

Observed issue from Run 1: many top GIFs still slipped during lift despite good static metrics. Two likely causes:

1. Contact model was too weak (friction/contact geometry not robust enough).
2. Foundational controls were transferred from base scenes and not re-solved per selected morphology for final GIF ranking.

### Run 2 Method Changes

1. Physical/contact upgrades in [assets/mjcf/scene.xml](assets/mjcf/scene.xml):
   - Higher default friction and tighter contact solver settings (`solref`, `solimp`).
   - Higher cube geom friction.
   - Extra fingertip contact geoms (small side spheres) on thumb/index/middle tips.
2. Stability-aware objective in [src/morphohand/optimization/phase1_grasp.py](src/morphohand/optimization/phase1_grasp.py):
   - Added penalties for XY drift and drop-from-peak height.
   - Added reward for contact persistence through lift/hold.
   - Exported metrics:
     - `cube_xy_drift`
     - `cube_z_drop_from_peak`
     - `contact_persistence`
3. Better foundational search (large budget + multi-seed CEM):
   - Cube: 3 seeds, iterations 36, population 64.
   - Prism1/2/3: 3 seeds each, iterations 36, population 64.
   - Best seed per prism variant selected automatically in the Pollard runner.
4. Morphology-specific control correction for top results:
   - Added top-k local CEM refinement in [scripts/phase1_pollard_multiscene.py](scripts/phase1_pollard_multiscene.py).
   - Top ranking/GIF generation now uses refined controls (`refined_for_topk=True`).

### Run 2 Commands

Generate Run 2 prism scenes from updated physics:

```bash
uv run python scripts/generate_prism_scene.py --base-scene-xml assets/mjcf/scene.xml --size-x 0.02 --size-y 0.025 --size-z 0.02 --output-scene-xml assets/mjcf/generated/scene_prism_run2_x0.0200_y0.0250_z0.0200.xml
uv run python scripts/generate_prism_scene.py --base-scene-xml assets/mjcf/scene.xml --size-x 0.02 --size-y 0.030 --size-z 0.02 --output-scene-xml assets/mjcf/generated/scene_prism_run2_x0.0200_y0.0300_z0.0200.xml
uv run python scripts/generate_prism_scene.py --base-scene-xml assets/mjcf/scene.xml --size-x 0.02 --size-y 0.035 --size-z 0.02 --output-scene-xml assets/mjcf/generated/scene_prism_run2_x0.0200_y0.0350_z0.0200.xml
```

Cube foundational CEM (multi-seed):

```bash
MUJOCO_GL=egl bash -lc 'for seed in 0 1 2; do
  uv run python scripts/phase1_optimize_grasp.py \
    --scene-xml assets/mjcf/scene.xml \
    --optimizer cem \
    --iterations 36 \
    --population 64 \
    --elite-fraction 0.2 \
    --sigma-init 0.2 \
    --seed "$seed" \
    --skip-gif \
    --output-dir results/phase1/run_20260413_run2_cube_foundational_cem \
    --tag "cube_s${seed}"
done'
```

Prism foundational CEM (multi-seed):

```bash
MUJOCO_GL=egl bash -lc 'for y in 0.0250 0.0300 0.0350; do
  for seed in 0 1 2; do
    ytag="y$(echo "$y" | tr "." "d")"
    uv run python scripts/phase1_optimize_grasp.py \
      --scene-xml "assets/mjcf/generated/scene_prism_run2_x0.0200_y${y}_z0.0200.xml" \
      --optimizer cem \
      --iterations 36 \
      --population 64 \
      --elite-fraction 0.2 \
      --sigma-init 0.2 \
      --seed "$seed" \
      --skip-gif \
      --output-dir results/phase1/run_20260413_run2_prism_foundational_cem \
      --tag "${ytag}_s${seed}"
  done
done'
```

Run 2 refined 500-sample Pollard:

```bash
MUJOCO_GL=egl uv run python scripts/phase1_pollard_multiscene.py \
  --samples 500 \
  --top-k-gifs 5 \
  --refine-top-k \
  --refine-pool-size 15 \
  --refine-iterations 8 \
  --refine-population 24 \
  --cube-foundational-run-dir results/phase1/run_20260413_run2_cube_foundational_cem \
  --prism-foundational-run-dir results/phase1/run_20260413_run2_prism_foundational_cem \
  --cube-min-contacts 4 \
  --prism-min-contacts 4 \
  --cube-max-mean-tip-distance 0.012 \
  --prism-max-mean-tip-distance 0.02 \
  --tag run_20260413_pollard_multiscene_500_run2_refined
```

### Run 2 Artifacts

- Final run: [results/phase1/run_20260413_pollard_multiscene_500_run2_refined](results/phase1/run_20260413_pollard_multiscene_500_run2_refined)
- Summary: [results/phase1/run_20260413_pollard_multiscene_500_run2_refined/summary.json](results/phase1/run_20260413_pollard_multiscene_500_run2_refined/summary.json)
- Cube top-5: [results/phase1/run_20260413_pollard_multiscene_500_run2_refined/cube/top5_with_gifs.csv](results/phase1/run_20260413_pollard_multiscene_500_run2_refined/cube/top5_with_gifs.csv)
- Prism3 top-5: [results/phase1/run_20260413_pollard_multiscene_500_run2_refined/prism3/top5_with_gifs.csv](results/phase1/run_20260413_pollard_multiscene_500_run2_refined/prism3/top5_with_gifs.csv)

### Run 2 Results Snapshot

Feasibility and Pareto sizes:

- Cube: 491/500 feasible (98.2%), Pareto size 48
- Prism1: 491/500 feasible (98.2%), Pareto size 48
- Prism2: 485/500 feasible (97.0%), Pareto size 44
- Prism3: 496/500 feasible (99.2%), Pareto size 63

Top-5 rows now include stability metrics and refinement flag:

- `cube_xy_drift`
- `cube_z_drop_from_peak`
- `contact_persistence`
- `refined_for_topk`

Notable pattern in top rows: `contact_persistence` is commonly 1.0 and `cube_z_drop_from_peak` is commonly 0.0 after refinement, indicating much more stable lift-and-hold behavior than Run 1.

### On the Force-Closure Question

Current pipeline does **not** explicitly solve a static force-closure optimization (no wrench-cone QP / closure certificate yet). Run 2 still optimizes a dynamic proxy objective with stronger stability terms and better contact modeling.

What changed in response to this concern:

1. Contact model made richer and stickier (friction + extra tip geoms).
2. Stability terms added directly into scoring.
3. Foundational controls now searched with larger CEM budgets and multiple seeds.
4. Top-ranked morphologies are re-solved locally (short CEM) per morphology before GIF generation.

This is a practical intermediate step toward force-closure-aware optimization, but not a full closure solver yet.

## Run 3 (Three-Finger Stability and Twist Suppression)

### Motivation

Run 2 improved gross stability, but qualitative review still showed post-lift finger yaw/flex drift in many scenes, especially outside prism1. Run 3 targeted this directly.

### Run 3 Method Changes

1. Evaluator upgrades in [src/morphohand/optimization/phase1_grasp.py](src/morphohand/optimization/phase1_grasp.py):
   - Lift ramp in dynamics and rollout (`lift_ramp_steps`) instead of abrupt palm raise.
   - Per-finger contact tracking across dynamic phase:
     - `thumb_contact_persistence`
     - `index_contact_persistence`
     - `middle_contact_persistence`
     - `min_finger_contact_persistence`
     - `all_finger_contact_persistence`
     - `finger_persistence_imbalance`
   - Finger-joint twist/flex drift metrics from end-of-settle to end-of-hold:
     - `finger_yaw_drift`
     - `finger_flex_drift`
   - Objective terms added:
     - reward on `min_finger_contact_persistence`
     - penalty on persistence imbalance
     - penalties on yaw/flex drift
2. Foundational and multiscene script knobs:
   - [scripts/phase1_optimize_grasp.py](scripts/phase1_optimize_grasp.py) now exposes all Run 3 evaluator weights.
   - [scripts/phase1_pollard_multiscene.py](scripts/phase1_pollard_multiscene.py) adds strict feasibility gates:
     - min per-finger persistence threshold
     - max finger-yaw drift threshold
3. Top-k local refinement retained:
   - Top candidates are re-solved per morphology before GIF rendering (`refined_for_topk=True`).

### Run 3 Commands

Generate Run 3 prism scenes:

```bash
uv run python scripts/generate_prism_scene.py --base-scene-xml assets/mjcf/scene.xml --size-x 0.02 --size-y 0.025 --size-z 0.02 --output-scene-xml assets/mjcf/generated/scene_prism_run3_x0.0200_y0.0250_z0.0200.xml
uv run python scripts/generate_prism_scene.py --base-scene-xml assets/mjcf/scene.xml --size-x 0.02 --size-y 0.030 --size-z 0.02 --output-scene-xml assets/mjcf/generated/scene_prism_run3_x0.0200_y0.0300_z0.0200.xml
uv run python scripts/generate_prism_scene.py --base-scene-xml assets/mjcf/scene.xml --size-x 0.02 --size-y 0.035 --size-z 0.02 --output-scene-xml assets/mjcf/generated/scene_prism_run3_x0.0200_y0.0350_z0.0200.xml
```

Cube foundational CEM (Run 3 objective):

```bash
MUJOCO_GL=egl bash -lc 'for seed in 0 1 2; do
  uv run python scripts/phase1_optimize_grasp.py \
    --scene-xml assets/mjcf/scene.xml \
    --optimizer cem \
    --iterations 40 \
    --population 72 \
    --elite-fraction 0.2 \
    --sigma-init 0.2 \
    --seed "$seed" \
    --lift-ramp-steps 100 \
    --objective-weight-min-finger-persistence 2.4 \
    --objective-weight-finger-persistence-imbalance-penalty 1.2 \
    --objective-weight-finger-yaw-drift-penalty 1.0 \
    --objective-weight-finger-flex-drift-penalty 0.5 \
    --skip-gif \
    --output-dir results/phase1/run_20260413_run3_cube_foundational_cem \
    --tag "cube_s${seed}"
done'
```

Prism foundational CEM (Run 3 objective):

```bash
MUJOCO_GL=egl bash -lc 'for y in 0.0250 0.0300 0.0350; do
  for seed in 0 1 2; do
    ytag="y$(echo "$y" | tr "." "d")"
    uv run python scripts/phase1_optimize_grasp.py \
      --scene-xml "assets/mjcf/generated/scene_prism_run3_x0.0200_y${y}_z0.0200.xml" \
      --optimizer cem \
      --iterations 40 \
      --population 72 \
      --elite-fraction 0.2 \
      --sigma-init 0.2 \
      --seed "$seed" \
      --lift-ramp-steps 100 \
      --objective-weight-min-finger-persistence 2.4 \
      --objective-weight-finger-persistence-imbalance-penalty 1.2 \
      --objective-weight-finger-yaw-drift-penalty 1.0 \
      --objective-weight-finger-flex-drift-penalty 0.5 \
      --skip-gif \
      --output-dir results/phase1/run_20260413_run3_prism_foundational_cem \
      --tag "${ytag}_s${seed}"
  done
done'
```

Run 3 full multiscene sampling (500) with strict gates and top-k refinement:

```bash
MUJOCO_GL=egl uv run python scripts/phase1_pollard_multiscene.py \
  --samples 500 \
  --top-k-gifs 5 \
  --refine-top-k \
  --refine-pool-size 15 \
  --refine-iterations 10 \
  --refine-population 28 \
  --cube-foundational-run-dir results/phase1/run_20260413_run3_cube_foundational_cem \
  --prism-foundational-run-dir results/phase1/run_20260413_run3_prism_foundational_cem \
  --cube-min-contacts 4 \
  --prism-min-contacts 4 \
  --cube-min-finger-contact-persistence 0.55 \
  --prism-min-finger-contact-persistence 0.45 \
  --cube-max-finger-yaw-drift 0.30 \
  --prism-max-finger-yaw-drift 0.40 \
  --cube-max-mean-tip-distance 0.012 \
  --prism-max-mean-tip-distance 0.02 \
  --lift-ramp-steps 100 \
  --objective-weight-min-finger-persistence 2.4 \
  --objective-weight-finger-persistence-imbalance-penalty 1.2 \
  --objective-weight-finger-yaw-drift-penalty 1.0 \
  --objective-weight-finger-flex-drift-penalty 0.5 \
  --tag run_20260413_pollard_multiscene_500_run3
```

### Run 3 Artifacts

- Main run: [results/phase1/run_20260413_pollard_multiscene_500_run3](results/phase1/run_20260413_pollard_multiscene_500_run3)
- Summary: [results/phase1/run_20260413_pollard_multiscene_500_run3/summary.json](results/phase1/run_20260413_pollard_multiscene_500_run3/summary.json)
- Cube top-5: [results/phase1/run_20260413_pollard_multiscene_500_run3/cube/top5_with_gifs.csv](results/phase1/run_20260413_pollard_multiscene_500_run3/cube/top5_with_gifs.csv)
- Prism1 top-5: [results/phase1/run_20260413_pollard_multiscene_500_run3/prism1/top5_with_gifs.csv](results/phase1/run_20260413_pollard_multiscene_500_run3/prism1/top5_with_gifs.csv)
- Prism2 top-5: [results/phase1/run_20260413_pollard_multiscene_500_run3/prism2/top5_with_gifs.csv](results/phase1/run_20260413_pollard_multiscene_500_run3/prism2/top5_with_gifs.csv)
- Prism3 top-5: [results/phase1/run_20260413_pollard_multiscene_500_run3/prism3/top5_with_gifs.csv](results/phase1/run_20260413_pollard_multiscene_500_run3/prism3/top5_with_gifs.csv)

### Run 3 Results Snapshot

Feasible counts (strict gates):

- Cube: 420/500 (84.0%), Pareto size 62
- Prism1: 317/500 (63.4%), Pareto size 37
- Prism2: 293/500 (58.6%), Pareto size 38
- Prism3: 412/500 (82.4%), Pareto size 41

Comparison to Run 2 refined feasibility (same 500 samples):

- Cube: 491 -> 420
- Prism1: 491 -> 317
- Prism2: 485 -> 293
- Prism3: 496 -> 412

This drop is expected because Run 3 intentionally enforced stricter three-finger persistence and yaw-drift requirements.

Top-5 aggregate metrics by scene (Run 3):

- Cube top-5 average:
  - score 7.991, lift 0.0538, contacts 8.0
  - `min_finger_contact_persistence` 0.9994
  - `all_finger_contact_persistence` 0.9994
  - `finger_yaw_drift` 0.1230
- Prism1 top-5 average:
  - score 7.690, lift 0.0496, contacts 7.0
  - `min_finger_contact_persistence` 1.0000
  - `all_finger_contact_persistence` 1.0000
  - `finger_yaw_drift` 0.0108
- Prism2 top-5 average:
  - score 7.455, lift 0.0535, contacts 6.4
  - `min_finger_contact_persistence` 1.0000
  - `all_finger_contact_persistence` 1.0000
  - `finger_yaw_drift` 0.0538
- Prism3 top-5 average:
  - score 7.423, lift 0.0517, contacts 6.2
  - `min_finger_contact_persistence` 1.0000
  - `all_finger_contact_persistence` 1.0000
  - `finger_yaw_drift` 0.0248

### Run 3 Analysis

1. The new optimization pressure is doing the intended thing:
   - Top-ranked grasps strongly maintain all three-finger contact through lift/hold.
   - Most top rows show very low drop-from-peak and high persistence metrics.
2. The stricter gates significantly filter out unstable or imbalanced grasps:
   - Feasibility reduced relative to Run 2, but retained candidates are higher quality under the new stability definition.
3. Prism1 remains the cleanest qualitatively and quantitatively for low yaw drift.
4. Cube still shows higher residual yaw drift than prism1, suggesting next tuning should target cube-specific anti-twist weighting or actuator stiffness for yaw channels.

### Run 3 Next Tweaks (if needed)

1. Increase yaw-drift penalty for cube only and re-run cube lane quickly.
2. Add direct per-finger actuator delta penalty between settle and lift phases.
3. Add explicit force-closure proxy term (wrench-balance surrogate) in evaluator to complement persistence metrics.
