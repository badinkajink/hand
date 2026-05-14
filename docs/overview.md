# MorphoHand Project Snapshot

Last updated: 2026-05-01

This page is a consolidated snapshot of the repository structure, key modules and scripts, and a day-by-day work log distilled from docs/architecture/*.md. It is intended to help you continue iteration without needing a coding agent.

## Project Structure (What Lives Where)

- assets/
  - mjcf/
    - hand.xml: canonical 3-finger hand template
    - scene.xml: base scene with palm pose actuators
    - scene_*.xml: object-specific scenes with keyframes
    - generated/: generated morphology scenes (when used)
  - objects/: object-only XML definitions (prism, screwdriver, drill, calf)
- docs/
  - index.md: documentation entrypoint
  - architecture/: detailed logs and architecture notes
  - roadmap/, simulators/: roadmap and backend notes
- scripts/
  - phase1_optimize_grasp.py: single-scene inner-loop optimizer and reports
  - phase1_pollard_multiscene.py: cube + prism morphology sweep
  - run6_combined_multitask.py: screwdriver multitask sweep
  - run6_analysis.py: embeddings and feature-metric plots
  - generate_*.py: object/scene generators
- src/morphohand/
  - optimization/: Phase 1 evaluator and optimizer lanes
  - sampling/: morphology sampling, FP loading, feasibility gates, scene emission
  - tools/: morphology XML utilities
  - backends/: backend protocol and placeholders
- results/
  - phase1/: run outputs, summaries, plots, gifs

## Core Code Map (Modules and Responsibilities)

### Optimization (inner loop)

- src/morphohand/optimization/phase1_common.py
  - Phase1EvalConfig: all metric weights and dynamics settings
  - Phase1GraspEvaluator: evaluate(), evaluate_trajectory(), rollout(), render_rollout()
  - Metrics: contact persistence, drift, lift, velocity, etc.

- src/morphohand/optimization/phase1_strategy_cem.py
  - CEM for constant controls: optimize_finger_controls()
  - CEM for trajectories: optimize_finger_control_trajectory()

- src/morphohand/optimization/phase1_strategy_mjx_autodiff.py
  - Autodiff lane using MJX surrogate

- src/morphohand/optimization/phase1_strategy_diffmjx.py
  - DiffMJX MVP lane with smooth contact proxies

- src/morphohand/optimization/phase1_trajectory.py
  - TrajectoryInterpolator and helper for piecewise-linear finger trajectories

### Sampling and morphology

- src/morphohand/sampling/morphology.py
  - Morphology bounds, sampling, distance, and name suffix helpers

- src/morphohand/sampling/foundational.py
  - Load best foundational controls from summary.json files

- src/morphohand/sampling/feasibility.py
  - FeasibilityCriteria gates and Pareto front helpers

- src/morphohand/sampling/scene.py
  - Write generated scenes, bake morphology, add foundational keyframes

- src/morphohand/sampling/adapt.py
  - Foundational pose adaptation modes (interval, sparse, local)

- src/morphohand/sampling/io.py
  - CSV writers and standard scatter plots

### XML morphology utilities

- src/morphohand/tools/morphology_xml.py
  - MorphologyValues
  - create_rigid_morphology_xml(), create_rigid_hand_and_scene_xmls()
  - apply_morphology_to_qpos(), extract_morphology_from_qpos()
  - rebase_asset_file_paths() for generated-scene stability

### Scripts (entrypoints)

- scripts/phase1_optimize_grasp.py
  - Main inner-loop CLI, writes summary.json, report.md, plots, gifs
  - Supports frozen scene generation and trajectory controls

- scripts/phase1_pollard_multiscene.py
  - Pollard-style sampling across cube + prism scenes
  - Feasibility gates, optional FP adaptation, top-k refinement

- scripts/run6_combined_multitask.py
  - Multi-keyframe evaluation for screwdriver with combined adaptation

- scripts/run6_analysis.py
  - Embeddings and thumb-x/y heatmaps for run6 outputs

## Day-by-Day Log (Consolidated)

### 2026-04-13
- Phase 1 baselines and MJX-autodiff runs were documented.
- CEM runs on a generated scene outperformed MJX-autodiff under current settings.
- Pollard-style multiscene sweep (500 samples) was run using CEM foundational poses for cube + prism.
- Run 2 and Run 3 refinements added stability terms, contact persistence metrics, and top-k refinement.
- FP adaptation strategies were evaluated; sparse-per-morph and interval-initial-fp improved feasibility the most.

### 2026-04-14
- Warp backend fidelity was evaluated against MuJoCo.
- mjwarp preserved metrics with reduced sync intervals; comfree-warp produced implausible results.
- Decision: keep CPU MuJoCo as default for Phase 1 until a batched GPU evaluator exists.

### 2026-04-15
- Object and scene set expanded: prism, screwdriver medium/small, power drill, human calf.
- Scene keyframes added for screwdriver/drill/calf.
- Model-load smoke checks passed for new assets and scenes.

### 2026-04-20
- Run 6 combined multi-keyframe screwdriver sweep completed.
- Results showed strong pose sensitivity: open_flat was easiest; open_90vertical hardest.
- run6_analysis tools were used to summarize feasibility and embeddings.

### 2026-04-24
- Power drill run8 recovery: generated scenes failed to load meshes due to path rebasing.
- Fix applied in sampling/scene.py via rebase_asset_file_paths().
- Run relaunched; mesh load issues resolved, some MuJoCo instability warnings remained.

### 2026-04-30
- Trajectory-based finger control optimization added and integrated.
- Frozen-scene generation (baked morphology) standardized in phase1_optimize_grasp.py.
- Pivot axis corrected to use rz for drill reorientation.
- Objective and evaluator refactoring reduced duplicated code paths.
- Open issue: contact count is ambiguous for multi-sphere drill proxy; per-finger persistence is the better metric.

## What To Touch To Continue Iteration

### If you want to adjust the grasp objective or metrics
- src/morphohand/optimization/phase1_common.py
  - Edit Phase1EvalConfig defaults and _compute_score_and_metrics()
- scripts/phase1_optimize_grasp.py
  - Expose new CLI flags or weight defaults
- src/morphohand/sampling/feasibility.py
  - Update gating logic for feasibility thresholds

### If you want to add or update scenes/objects
- assets/objects/*.xml for object geometry and mass
- assets/mjcf/scene_*.xml for task scenes and keyframes
- scripts/generate_prism_scene.py or other generators for variants
- src/morphohand/sampling/scene.py when new scene emission logic is needed

### If you want to change morphology sampling
- src/morphohand/sampling/morphology.py
  - Bounds, perturbation, sampling strategy
- scripts/phase1_pollard_multiscene.py
  - Sample counts, gates, and refinement policies

### If you want to change foundational pose adaptation
- src/morphohand/sampling/adapt.py
  - Adaptation modes and local-perturb strategy
- scripts/phase1_pollard_multiscene.py or scripts/run6_combined_multitask.py
  - Adaptation knobs and intervals

### If you want to extend trajectory optimization
- src/morphohand/optimization/phase1_trajectory.py
  - Interpolation logic
- src/morphohand/optimization/phase1_strategy_cem.py
  - Trajectory optimizer logic and sampling
- src/morphohand/optimization/phase1_common.py
  - Trajectory rollout and evaluation hooks

### If you want to experiment with backends
- src/morphohand/backends/*
  - Backend protocol and placeholders
- scripts/phase1_optimize_grasp.py
  - Backend CLI and runtime controls

## Practical Run Entry Points

- Foundational pose search:
  - scripts/phase1_optimize_grasp.py
- Morphology sweep (cube + prism):
  - scripts/phase1_pollard_multiscene.py
- Multi-keyframe screwdriver sweep:
  - scripts/run6_combined_multitask.py
- Run analysis and plots:
  - scripts/run6_analysis.py

## Results Artifacts to Inspect

- results/phase1/<run_tag>/summary.json
- results/phase1/<run_tag>/report.md
- results/phase1/<run_tag>/all_candidates.csv
- results/phase1/<run_tag>/all_candidates_multitask.csv
- results/phase1/<run_tag>/all_task_results.csv
- results/phase1/<run_tag>/top5_gifs/

## Known Open Issues

- Contact count is ambiguous for multi-sphere drill proxies; rely on per-finger persistence.
- Drill reorientation can induce cube axis tilt and XY drift; weights need rebalancing when targeting stable pivots.
- Backend GPU throughput is limited by host-driven evaluation; batched evaluator is a prerequisite.
