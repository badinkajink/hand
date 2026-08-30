# MorphoHand

MorphoHand is a simulation-to-real co-design stack for a three-finger, reconfigurable hand.
It now contains both the simulation/RL research system and the live `real_v1` hardware path:
six morphology gantry axes, nine independent finger servos, validated sim-to-hardware plans,
a CB1 control service, a workstation/browser UI, and hardware experiment logging.

The key design split is:

- 9D morphology parameters: per finger `(x, y, len)`
- 9D control parameters: per finger `(yaw, mcp, pip)`

That split is reflected directly in the package layout and in the current evaluation scripts.

## What Is Implemented

- `src/morphohand/sampling/`: morphology sampling, foundational pose loading, feasibility
  gating, scene XML patching, frozen-scene baking, and CSV/plot helpers.
- `src/morphohand/optimization/phase1_common.py`: shared Phase 1 evaluator and objective.
- `src/morphohand/optimization/phase1_strategy_cem.py`: CEM grasp optimizer.
- `src/morphohand/optimization/phase1_strategy_synergy_cem.py`: CEM in a low-dim
  (eigengrasp) coefficient subspace — see [docs/eigengrasp.md](docs/eigengrasp.md).
- `src/morphohand/optimization/phase1_strategy_mjx_autodiff.py`: MJX autodiff lane.
- `src/morphohand/optimization/phase1_strategy_diffmjx.py`: DiffMJX-style MVP lane.
- `src/morphohand/optimization/contact_targets.py`: hand-authorable contact patches as
  an objective term — see [docs/contact_targets.md](docs/contact_targets.md).
- `src/morphohand/optimization/force_closure.py`: continuous Ferrari-Canny / DFC grasp
  quality energy — see [docs/force_closure.md](docs/force_closure.md).
- `src/morphohand/optimization/eigengrasp.py`: PCA basis fit on historical grasps,
  hand-designed fallback basis.
- `src/morphohand/tools/morphology_xml.py`: morphology encoding, parsing, and XML generation.
- `assets/mjcf/`: base hand/scene templates plus object-specific scenes.
- `src/morphohand/driver/manta/`: current Manta M8P + CB1 hardware firmware and host package.
- `src/morphohand/driver/manta/host/manta_hand/plan.py`: verified `real_v1` frame, envelope,
  and sim-joint-to-servo plan conversion.
- `src/morphohand/driver/manta/host/manta_hand/web.py`: token-protected control service and UI.
- `assets/contact_targets/`: per-(scene, keyframe) target patch specs (YAML).
- `scripts/`: runnable Phase 1 sweeps, analysis tools, and the cross-object
  [eval suite](docs/eval_suite.md).

The outer-loop MAP-Elites modules are still skeletons; the working system today is the
sampling-and-evaluation pipeline around Phase 1.

### Grasp specification methods at a glance

Three orthogonal approaches, all opt-in via config knobs (default off):

| Method | Toggle | When to use |
|---|---|---|
| Eigengrasp / synergy CEM | call `optimize_finger_controls_synergy` with a fitted basis | sample-efficient search when you have historical runs |
| Contact-target patches | `objective_weight_contact_target_reward` / `_distance_penalty` | object-specific intent: "thumb here, index there" |
| Force-closure energy | `objective_weight_force_closure` | grasp-quality signal beyond contact count |

Full overview and empirical comparison: [docs/grasp_methods.md](docs/grasp_methods.md).

### Eval suite

Cross-object benchmark harness at [scripts/eval_suite.py](scripts/eval_suite.py) — 8
benchmarks × N methods × seeds, with oracle re-scoring under the baseline objective,
auto-generated GIFs per `(benchmark, method)`, and per-benchmark markdown reports.
See [docs/eval_suite.md](docs/eval_suite.md) for adding new benchmarks / methods.

## Repository Layout

```text
assets/
  mjcf/
    hand.xml                  # 3-finger hand template
    scene.xml                 # base scene with palm pose actuators
    scene_*.xml               # object-specific evaluation scenes
  objects/                    # object XMLs used by the scenes

docs/
  index.md
  architecture/
  simulators/
  roadmap/

scripts/
  phase1_optimize_grasp.py    # single-scene Phase 1 optimizer
  phase1_pollard_multiscene.py# cube + prism morphology sampling sweep
  run6_combined_multitask.py  # screwdriver multi-keyframe combined run
  run6_analysis.py            # embeddings and feature-metric plots
  smoke_test_models.py        # sanity checks for the MJCF models

src/morphohand/
  sampling/                   # morphology, FP, feasibility, scene, IO
  optimization/               # Phase 1 evaluator and strategy lanes
  tools/                      # morphology XML helpers
  backends/                   # backend protocol and adapter shells
  driver/manta/               # current hardware: STM32 firmware + CB1 Python host
```

## Real hand control

The deployable hardware method today is the morphology-specific CEM grasp and buffered
open-loop reorientation. The learned A/B policies require object pose/velocity and other
observations the servo-only prototype does not have, so the UI marks closed-loop RL unavailable
instead of silently filling missing observations.

Run the complete control station safely against a mock hand:

```bash
PYTHONPATH=src/morphohand/driver/manta/host \
  python -m manta_hand.web --mock --host 127.0.0.1 --port 8765
```

Then open `http://127.0.0.1:8765`. CB1 installation, the once-per-session homing flow,
telemetry-rate benchmark, real launch command, API, logs, and recovery behavior are documented
in [docs/hardware_control_station.md](docs/hardware_control_station.md). The static UI may be
served by the CB1, but it executes in the workstation browser; the guide also shows how to serve
those files locally while using the CB1 only as the hardware API. Hardware electrical,
StallGuard, and servo calibration details remain in
[src/morphohand/driver/manta/docs/](src/morphohand/driver/manta/docs/).

## Environment

The repo is configured for Python 3.10.

```bash
uv venv --python 3.10
source .venv/bin/activate
uv sync --extra dev
```

Optional backend extras:

```bash
uv sync --extra mjwarp --extra comfree
```

If you use the local Warp checkouts under `external/`, install them with the helper script
and then editable `uv pip` installs when the upstream metadata is available.

## Quick Start

Smoke-test the model layout:

```bash
uv run python scripts/smoke_test_models.py
```

Run the default Phase 1 grasp optimizer on a single scene/keyframe:

```bash
uv run python scripts/phase1_optimize_grasp.py \
  --scene-xml assets/mjcf/baseline/scenes/scene_screwdriver_medium.xml \
  --keyframe open_flat \
  --optimizer cem
```

Run the current morphology sweep used in the docs and results folders:

```bash
uv run python scripts/phase1_pollard_multiscene.py \
  --samples 500 \
  --fp-adaptation sparse-per-morph
```

Run the combined screwdriver multi-keyframe experiment:

```bash
uv run python scripts/run6_combined_multitask.py \
  --samples 1000 \
  --keyframes open_flat open_vertical open_90vertical
```

Analyze a run:

```bash
uv run python scripts/run6_analysis.py \
  --run-dirs results/phase1/run6_combined_1000 \
  --embedding tsne \
  --metrics cube_xy_drift finger_flex_drift
```

## Current Benchmark Composition

The repository currently centers on these scene/object pairs:

- `assets/mjcf/baseline/scenes/scene_prism.xml`
- `assets/mjcf/baseline/scenes/scene_screwdriver_medium.xml`
- `assets/mjcf/baseline/scenes/scene_screwdriver_small.xml`
- `assets/mjcf/baseline/scenes/scene_power_drill.xml`
- `assets/mjcf/baseline/scenes/scene_human_calf.xml`

The medium screwdriver scene is the most developed benchmark. It uses three keyframes:

- `open_flat`
- `open_vertical`
- `open_90vertical`

## Current Results

Foundational pose search on the medium screwdriver scene produced the following best
scores:

- `open_flat`: `5.702777`
- `open_vertical`: `6.906849`
- `open_90vertical`: `7.091068`

The combined screwdriver multitask run then confirmed that the pipeline can score all
three keyframes per morphology in one pass. In the current 1000-sample sweep, feasible
rates were:

- `open_flat`: `0.994`
- `open_vertical`: `0.884`
- `open_90vertical`: `0.696`

This is the main signal the docs and paper now track: pose choice matters, and the same
morphology distribution behaves differently across the three screwdriver orientations.

## Artifacts

Phase 1 runs write their outputs under `results/phase1/<run_tag>/`.
Common artifacts include:

- `summary.json`
- `all_candidates.csv` or `all_candidates_multitask.csv`
- `all_task_results.csv`
- `rolling_efficiency.csv`
- `objective_trace.png`
- `grasp_metrics_trace.png`
- `top5_gifs/`

## Grasp Run Protocol

**Every grasp experiment must run against a frozen scene XML.** The base scenes under
`assets/mjcf/` still carry morphology DOFs as joints, which drift during the rollout
and silently invalidate any cross-method comparison. Full rationale, the canonical
helper (`morphohand.sampling.scene.freeze_scene_for_eval`), DOF-count sanity check,
and a before/after numbers table for the eval suite are in
[docs/frozen_scene_protocol.md](docs/frozen_scene_protocol.md).

The eval-suite harnesses (`scripts/eval_suite.py`, `scripts/compare_methods.py`) and
`scripts/phase1_optimize_grasp.py` enforce this automatically. New scripts that
construct a `Phase1GraspEvaluator` should freeze first.

The frozen scene artifact should capture:

- the exact scene XML used for the run,
- the keyframe name,
- the frozen morphology setting or frozen morphology scene variant,
- the tuned `open_flat` qpos/qctrl pair when a manual pregrasp is being carried forward,
- the pivot convention used for the lift-and-tilt pass,
- and whether all three fingertips actually made contact.

For the power-drill runs in particular:

- the short-proximal scene is the current manual-tuning target,
- the drill should tilt from the forward pose toward facing down, not the reverse direction,
- and a result with only six contacts usually means the middle finger never engaged.

## Notes on Backend Support

The evaluator supports `mujoco`, `mjwarp`, and `comfree-warp` runtime backends.
MuJoCo is the default and most reliable option for the current Phase 1 loop.
The Warp-backed paths remain optional and are best treated as throughput experiments.

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

There are two doc surfaces, serving different purposes:

**1. Reference docs (`docs/`) — MkDocs.** Detailed architecture, backend notes, and the full RL
research log (`docs/rl/`). The README stays the high-level execution plan; `docs/` holds the depth.

```bash
uv run mkdocs serve          # live-reload server at http://localhost:8000
```

**2. Project site (`webpaper/`) — Typst → static HTML.** A research-paper-grade, media-rich
write-up (morphology/grasp optimization, RL manipulation + policy switching, hardware validation).
Authored in Typst so it carries paper-quality math *and* embedded video. Requires `typst` on PATH.

```bash
webpaper/build.sh                                 # compiles src/*.typ -> webpaper/build/*.html
python3 -m http.server -d webpaper/build 8080     # then open http://localhost:8080
```

See [webpaper/README.md](webpaper/README.md) for authoring (adding a page, the math→SVG show rule,
and Typst-HTML export gotchas).

**RL research state.** The canonical living handoff for the in-hand reorientation work is
[RESEARCH_STATE.md](RESEARCH_STATE.md) at the repo root (self-contained problem statement + current
state + open problems); the full chronological log is [docs/rl/reorientation.md](docs/rl/reorientation.md).

## Immediate Next Tasks

1. Implement real MJX stepping in `MJXNativeBackend`.
2. Add first inner-loop objective for cube grasp stability.
3. Wire minimal MAP-Elites loop over `(x, y, l)`.
4. Add canonical scene pose config file and evaluator.
