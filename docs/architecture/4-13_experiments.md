# 4-13 Experiments Log

Date: 2026-04-13

Reference commit for reproducibility: `2ea2bf2`

## Scope

This log documents:

- What was run for Phase 1 inner-loop experiments.
- How to replay runs.
- What each metric means.
- Why `phase1_generated_scene` and `sanity_phase1` differ.
- Why MJX-autodiff underperformed relative to CEM in current settings.
- Concrete next-step plans (Pollard-style lane and DiffMJX direction).
- Dependency hardening to avoid breakage across `uv run` invocations.

## Environment Snapshot

Machine: desktop with NVIDIA GPU (RTX 4070 Ti SUPER)

Validated runtime imports:

- `warp`: available
- `mujoco_warp`: available
- `comfree_warp`: available
- `jax`: available

Current default `uv run` backend after dependency hardening:

- JAX backend: CPU (`jax==0.4.38`)

Note on CUDA usage:

- MJX can run on CPU or GPU.
- GPU support requires a CUDA-enabled `jaxlib` (and matching plugin stack for current JAX versions).
- Earlier failures were due to CPU-only JAX, missing optional `warp` deps, and mixed-version plugin leftovers.

## Runs Completed

### A) Historical run: `run_20260410_163959`

Directory: `results/phase1/run_20260410_163959`

Subruns:

1. `phase1_generated_scene`
2. `sanity_phase1`

Both subruns used:

- Scene: `assets/mjcf/generated/scene_tp0d0000p0d0200p0d0000_ip0d0100n0d0123p0d0000_mp0d0100p0d0153p0d0000.xml`
- Optimizer family: CEM (`optimize_finger_controls`)

They differ by optimizer budget:

- `phase1_generated_scene`: `iterations=20`, `population=36`
- `sanity_phase1`: `iterations=6`, `population=18`

Observed outcomes:

- `phase1_generated_scene` best score: `3.420744`, lift `0.064198`, contacts `3.0`
- `sanity_phase1` best score: `2.958089`, lift `0.051363`, contacts `3.0`

Interpretation:

- These are not two different methods; both are CEM.
- Better behavior in `phase1_generated_scene` is expected from higher search budget (more samples and iterations), which improves chance of finding a better local optimum.

### B) Prism y-sweep rerun with MJX-autodiff + GIF

Directory: `results/phase1/run_20260413_prism_y_sweep_mjx_autodiff`

Subruns:

- `y0d0250`
- `y0d0300`
- `y0d0350`

All include full artifacts:

- `best_rollout.gif`
- `summary.json`
- `optimization_trace.csv`
- plots and rollout arrays

Results:

- `y=0.0250`: score `1.147062`, lift `0.022655`, contacts `1.0`
- `y=0.0300`: score `2.402859`, lift `0.046855`, contacts `2.0`
- `y=0.0350`: score `1.411974`, lift `0.029578`, contacts `1.0`

## Commands (Replay)

### 1) Single-run replay template

```bash
MUJOCO_GL=egl uv run python scripts/phase1_optimize_grasp.py \
  --scene-xml assets/mjcf/generated/scene_prism_x0.0200_y0.0300_z0.0200.xml \
  --optimizer mjx-autodiff \
  --iterations 12 \
  --seed 0 \
  --output-dir results/phase1/run_YYYYMMDD_prism_y_sweep_mjx_autodiff \
  --tag y0d0300
```

### 2) Full prism y-sweep replay (with GIF rendering)

```bash
for y in 0.0250 0.0300 0.0350; do
  tag="y$(echo "$y" | tr '.' 'd')"
  MUJOCO_GL=egl uv run python scripts/phase1_optimize_grasp.py \
    --scene-xml "assets/mjcf/generated/scene_prism_x0.0200_y${y}_z0.0200.xml" \
    --optimizer mjx-autodiff \
    --iterations 12 \
    --seed 0 \
    --output-dir results/phase1/run_YYYYMMDD_prism_y_sweep_mjx_autodiff \
    --tag "$tag"
done
```

### 3) CEM replay template

```bash
MUJOCO_GL=egl uv run python scripts/phase1_optimize_grasp.py \
  --scene-xml assets/mjcf/generated/scene_tp0d0000p0d0200p0d0000_ip0d0100n0d0123p0d0000_mp0d0100p0d0153p0d0000.xml \
  --optimizer cem \
  --iterations 20 \
  --population 36 \
  --seed 0 \
  --output-dir results/phase1/run_YYYYMMDD_163959 \
  --tag phase1_generated_scene
```

## Metric Definitions (Phase 1)

The current evaluator objective is:

$$
\text{score} =
w_{lift} \cdot \text{cube\_lift}
- w_{dist} \cdot \text{mean\_tip\_distance}
+ w_{contact} \cdot \text{cube\_tip\_contacts}
- w_{vel} \cdot \|v_{cube}\|
$$

Where defaults are:

- `w_lift = 35.0`
- `w_dist = 2.0`
- `w_contact = 0.4`
- `w_vel = 0.15`

Key reported metrics:

- `mean_tip_distance`: average fingertip-to-cube distance proxy after settle.
- `cube_tip_contacts`: count of fingertip contact pairs with cube.
- `cube_z_before_lift`: cube height before palm lift command.
- `cube_z_peak`: maximum cube height during lift phase.
- `cube_z_after_hold`: cube height after hold phase.
- `cube_lift = cube_z_peak - cube_z_before_lift`.
- `cube_vel_norm`: norm of cube generalized velocity (stability proxy).

## Why MJX-autodiff looked worse

Current likely causes:

1. Surrogate mismatch:
- Autodiff path optimizes a smooth proxy (distance + sigmoid contact proxy + L2 penalty), not exact contact events in the final MuJoCo score.

2. Contact gradient quality:
- Hard-contact dynamics are poorly conditioned for gradients; the surrogate may not provide reliable guidance toward stable force closure.

3. Optimization schedule:
- Current default iterations (`12`) and learning-rate/clip schedule can be too short or unstable for this landscape.

4. Compilation/runtime overhead:
- First-run JAX/XLA compile is expensive; this inflates wall time relative to CEM for short experiments.

## CEM vs MJX-autodiff in current code

Current methods exposed by `--optimizer`:

1. `cem`
- Cross-Entropy Method.
- Population sampling, elite selection, update of mean/sigma.
- Robust in non-smooth objectives; currently strongest baseline.

2. `mjx-autodiff`
- Gradient ascent on MJX surrogate objective.
- Uses JAX autodiff and clipping.
- Faster asymptotically only if gradients are high quality and compile/runtime overhead is amortized.

## Next-Step Plans

### Plan A: Pollard-style baseline lane (recommended immediate)

Goal: de-risk progress with feasibility-first sampling and post-hoc evaluation.

1. Add feasibility prefilter before expensive optimization:
- quick collision checks,
- initial closure/contact plausibility,
- simple support/lift proxy.

2. Sample morphology candidates with diversity threshold.

3. Run short CEM inner loops on feasible candidates only.

4. Compute Pareto fronts across:
- lift,
- contact quality,
- stability/velocity proxy,
- control effort.

5. Cluster accepted designs to identify morphology families.

Success criteria:

- better throughput per useful candidate,
- stable trade-off front trends across object subsets.

### Plan B: Improve differentiable lane (DiffMJX-style direction)

Goal: recover useful gradients for contact-rich grasping.

1. Implement contact-from-distance (CFD) style surrogate branch in backward path.

2. Run gradient diagnostics:
- finite-difference agreement checks on small scenes,
- per-term gradient scale logging.

3. Add curriculum:
- start with distance-heavy weights,
- increase contact/lift emphasis over iterations.

4. Add hybrid optimizer:
- few gradient steps for warm-start,
- CEM refinement for final robust grasp.

Success criteria:

- gradient lane beats CEM on wall-time-to-threshold or reaches comparable quality with fewer rollouts.

## Timing Instrumentation Added (for future runs)

`phase1_optimize_grasp.py` now logs timing in `summary.json` and `report.md`:

- `optimization_wall_time_seconds`
- `mean_iteration_seconds`
- `rollout_wall_time_seconds`
- `gif_render_wall_time_seconds`
- `total_run_wall_time_seconds`

`optimization_trace.csv` now also includes `iteration_seconds`.

## Dependency Hardening Across `uv run`

For the separate GPU experiment lane setup, see: `docs/architecture/gpu_mjx_usage.md`.

To avoid missing MJX module and version drift, dependencies were tightened in `pyproject.toml`:

- `mujoco>=3.6.0,<3.7`
- `mujoco-mjx>=3.6.0,<3.7`
- `jax>=0.4.30,<0.5`
- `jaxlib>=0.4.30,<0.5`

Rationale:

- `mujoco_warp` currently expects MuJoCo 3.6+.
- `mujoco.mjx` availability is guaranteed by explicit `mujoco-mjx` dependency.
- `jax/jaxlib` are kept in a Python-3.10-compatible range so `uv run` resolves cleanly against project `requires-python`.

Important practical note:

- This default hardened setup prioritizes `uv run` reproducibility and currently uses CPU JAX.
- If GPU JAX is desired, install a matching CUDA-enabled JAX stack as a separate lane and avoid mixing incompatible plugin versions in the same environment.

If you want strict separation (recommended), keep two environments:

- Core phase1/MJX env.
- MJWarp/ComFree backend env.

This avoids cross-backend dependency churn while preserving reproducibility.
