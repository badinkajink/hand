# MorphoHand Simulation Stack (Scaffold)

This repository is now structured to support a staged build-out of the morphology optimization pipeline described in the proposal in `hand/main.tex`.

## Proposal Alignment (What this scaffold assumes)

- Research objective: compare morphology families over object distributions, not just a single best hand.
- Optimization structure: bi-level optimization.
  - Outer loop: MAP-Elites over morphology.
  - Inner loop: gradient-based grasp synthesis.
- Updated DOF split (from `yaw_inner_loop.md`):
  - Outer loop morphology variables: per finger `(x, y, l)`.
  - Inner loop control variables: per finger `(yaw, mcp, pip)`.
- Backend strategy from `simulator_backends.md`:
  - `mjx-native` and `diffmjx-lite` for differentiable inner loop.
  - `mjwarp` and `comfree-warp` for high-throughput outer loop evaluation.

## Repository Layout

```text
assets/
  mjcf/
    hand.xml                  # 9 actuator model (3 fingers x yaw/mcp/pip)
    scene.xml                 # 15 actuator model (hand + 6-DoF palm pose)

docs/
  index.md
  architecture/
  simulators/
  roadmap/

scripts/
  smoke_test_models.py        # validates actuator counts and basic model loading

src/morphohand/
  backends/
    base.py                   # common backend protocol
    factory.py                # backend registry + constructor
    mjx_native.py             # MJX autodiff backend skeleton
    diffmjx.py                # DiffMJX-compatible backend skeleton
    mjwarp_backend.py         # MJWarp backend skeleton
    comfree_backend.py        # ComFree backend skeleton
  optimization/
    inner_loop.py             # gradient-based grasp optimizer skeleton
    outer_loop.py             # MAP-Elites skeleton

tests/
  test_model_layout.py        # static XML validation tests
```

## Environment and Build (uv)

### 1) Create environment

```bash
uv venv --python 3.10
source .venv/bin/activate
```

Note: `uv sync` can pick a different interpreter if multiple Python versions are available. If ROS Humble integration is expected later, stay on Python 3.10 for compatibility.

### 2) Install project

```bash
uv sync --extra dev
```

### 3) Optional backend extras

```bash
# Warp prerequisites for mjwarp/comfree integration
uv sync --extra mjwarp --extra comfree
```

### 4) Install backend repos (editable local checkouts)

The Python package names and install paths for `mujoco_warp` and `comfree_warp` can vary by upstream state. Keep these in sibling directories and install with `uv pip`:

```bash
./scripts/setup_backends.sh

# Install if/when setup metadata is available in those repos
uv pip install -e external/mujoco_warp
uv pip install -e external/comfree_warp
```

## Quick Start

Run the baseline model smoke test:

```bash
uv run python scripts/smoke_test_models.py
```

Run Phase 1 inner-loop grasp synthesis on a generated rigid scene:

```bash
uv run python scripts/phase1_optimize_grasp.py \
  --scene-xml assets/mjcf/generated/scene_tp0d0000p0d0200p0d0000_ip0d0100n0d0123p0d0000_mp0d0100p0d0153p0d0000.xml
```

This creates run artifacts under `results/phase1/<run_tag>/` including optimization trace CSV,
plots, trajectory arrays, and an animated best-rollout GIF.

Run tests:

```bash
./scripts/test.sh
```

Generate a rigid morphology XML (no morph joints, only control joints remain):

```bash
uv run python scripts/generate_morphology_xml.py \
  --base-hand-xml assets/mjcf/hand.xml \
  --base-scene-xml assets/mjcf/scene.xml \
  --thumb 0.01 0.00 0.01 \
  --index 0.00 0.01 0.005 \
  --middle -0.01 0.00 0.00
```

Launch the Tkinter morphology editor (sliders + qpos paste + save):

```bash
uv run python scripts/morphology_gui.py
```

Expected output includes:

- `hand.xml` actuator count: `9`
- `scene.xml` actuator count: `15`

## Modeling Baseline (Implemented)

Current MJCF baseline is intentionally simple and differentiability-friendly:

- Three fingers: `thumb`, `index`, `middle`.
- Per finger joints (6):
  - Morphology: `x` slide, `y` slide, `len` slide.
  - Control: `yaw` hinge, `mcp` hinge, `pip` hinge.
- Geometry: capsule links + sphere fingertip, cube object on tabletop.
- `scene.xml` adds palm pose joints (`px, py, pz, rx, ry, rz`) for manual canonical placements.

This supports your requested workflow: manually define 3-5 canonical palm poses per object class, then optimize morphology+grasp on top of fixed coarse setup.

## Evaluation Scene Set (Current)

Scene assets now include object-specific evaluation templates with near-contact open keyframes:

- `assets/mjcf/scene_prism.xml`
- `assets/mjcf/scene_screwdriver_medium.xml` (`open_flat`, `open_vertical`, `open_90vertical`)
- `assets/mjcf/scene_screwdriver_small.xml` (`open_flat`, `open_vertical`)
- `assets/mjcf/scene_power_drill.xml` (`open_flat`, `open_vertical`)
- `assets/mjcf/scene_human_calf.xml` (`open_under_ankle`, `open_lifted`)

Corresponding object XMLs are under `assets/objects/`:

- `prism.xml`
- `screwdriver_medium.xml`
- `screwdriver_small.xml`
- `power_drill.xml`
- `human_calf.xml`

## Run 6: Screwdriver Standalone-Pose Sweep

Run 6 evaluates medium screwdriver keyframes independently (no transition), using two foundational-pose adaptation modes across a few hundred sampled morphologies.

### 1) Foundational pose search per keyframe

```bash
for kf in open_flat open_vertical open_90vertical; do
  for seed in 0 1; do
    uv run python scripts/phase1_optimize_grasp.py \
      --scene-xml assets/mjcf/scene_screwdriver_medium.xml \
      --keyframe "$kf" \
      --iterations 16 \
      --population 48 \
      --elite-fraction 0.2 \
      --sigma-init 0.18 \
      --seed "$seed" \
      --skip-gif \
      --output-dir "results/phase1/run6_foundational/$kf" \
      --tag "seed_${seed}"
  done
done
```

### 2) Morphology sweep (sparse5)

```bash
uv run python scripts/run6_screwdriver_multikey_sampling.py \
  --scene-xml assets/mjcf/scene_screwdriver_medium.xml \
  --foundational-root results/phase1/run6_foundational \
  --samples 240 \
  --seed 6 \
  --fp-adaptation sparse-per-morph \
  --morph-sort distance \
  --max-mean-tip-distance 0.022 \
  --min-contacts 2 \
  --output-dir results/phase1 \
  --tag run6_sparse5
```

### 3) Morphology sweep (interval refresh 50)

```bash
uv run python scripts/run6_screwdriver_multikey_sampling.py \
  --scene-xml assets/mjcf/scene_screwdriver_medium.xml \
  --foundational-root results/phase1/run6_foundational \
  --samples 240 \
  --seed 6 \
  --fp-adaptation interval-initial-fp \
  --fp-refresh-interval 50 \
  --morph-sort distance \
  --max-mean-tip-distance 0.022 \
  --min-contacts 2 \
  --output-dir results/phase1 \
  --tag run6_interval50
```

### 4) Embedding + feature analysis

```bash
uv run python scripts/run6_analysis.py \
  --run-dirs results/phase1/run6_sparse5 results/phase1/run6_interval50 \
  --embedding tsne \
  --metrics cube_xy_drift finger_flex_drift
```

Primary outputs:

- `results/phase1/run6_sparse5/`
- `results/phase1/run6_interval50/`
- `results/phase1/run6_sparse5/analysis/run6_analysis_summary.md`

## Backend Plan

### Backend contract

All backends implement `PhysicsBackend` in `src/morphohand/backends/base.py`:

- `load_model(xml_path)`
- `reset(seed)`
- `step(control)`
- `rollout(controls)`
- `metrics()`
- `supports_autodiff` flag

### Initial backend targets

1. `mjx-native`
- Primary differentiable backend for inner loop.
- JAX-based gradients through kinematics and soft-contact dynamics.

2. `diffmjx-lite`
- Extension of native MJX with placeholders for:
  - smooth collision branch blending,
  - CFD (contacts-from-distance) in backward pass,
  - optional adaptive integration hooks.
- Start with smooth collision + CFD path only.

3. `mjwarp`
- Throughput-oriented outer loop evaluator.
- Used when gradients are not required.

4. `comfree-warp`
- Throughput + analytical contact candidate.
- Run as alternative outer-loop evaluator; compare rank correlation vs hardware later.

## Optimization Plan (detailed, living architecture)

### Phase 0: Foundation (now)

- [x] uv project scaffold.
- [x] backend abstraction and registry.
- [x] baseline MJCF hand and scene.
- [x] XML + smoke tests.
- [x] docs scaffold with MkDocs.

### Phase 1: Inner-loop baseline (MJX)

- [x] Implement a real inner-loop grasp synthesis runner over `(yaw, mcp, pip)`.
- [x] Add practical grasp objective proxy (distance + contacts + lift + stability).
- [x] Add optimizer, trace logging, plots, and rollout visualization artifact generation.
- [ ] Add MJX-native autodiff path and compare against CEM baseline.

### Phase 2: Outer-loop baseline (MAP-Elites)

- [ ] Genome = `(x_i, y_i, l_i)` for each finger.
- [ ] Descriptor choices: spread, reach, opposition quality.
- [ ] Archive update rule + novelty handling.
- [ ] Evaluate each morphology with inner-loop best response.

### Phase 3: Multi-backend execution

- [ ] Run outer loop with `mjwarp` and `comfree-warp`.
- [ ] Keep inner loop on `mjx-native` by default.
- [ ] Add fallback policy: if `comfree` unstable, retry with `mjwarp`.
- [ ] Compare ranking consistency across backends.

### Phase 4: DiffMJX track

- [ ] Implement `diffmjx-lite` branch in code (not a forked package yet).
- [ ] Toggle CFD in backward pass.
- [ ] Add finite-difference gradient sanity checks.
- [ ] Compare `mjx-native` vs `diffmjx-lite` inner-loop convergence.

### Phase 5: CTR-inspired options (optional)

- [ ] Add implementation note and API hooks for projected gradient / cone-feasibility filters.
- [ ] Keep this as optional regularizer, not required for baseline results.

## Why one package (not many envs) initially

The scaffold keeps one uv environment and plugin backends by optional dependencies + runtime imports. This avoids duplicated code and keeps experiments comparable.

If backend dependency conflicts become real, split into backend-specific uv lockfiles:

- `uv.lock` (core + mjx)
- `uv.mjwarp.lock`
- `uv.comfree.lock`

Only split when needed.

## Documentation

Docs are in `docs/` and wired for MkDocs. The README remains the high-level execution plan; docs hold detailed architecture and backend notes.

Build docs locally:

```bash
uv run mkdocs serve
```

## Immediate Next Tasks

1. Implement real MJX stepping in `MJXNativeBackend`.
2. Add first inner-loop objective for cube grasp stability.
3. Wire minimal MAP-Elites loop over `(x, y, l)`.
4. Add canonical scene pose config file and evaluator.
