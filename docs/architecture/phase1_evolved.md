# Phase 1 Evolved Plan (MJX/DiffMJX Pivot)

Date: 2026-04-13

## Why this pivot

Pollard-style feasibility sampling delivered strong candidates, but now we want a cleaner answer to:

1. Can MJX gradients work reliably if we condition on a good foundational pose?
2. Is a short hybrid stage (gradient then sampling) enough before a full DiffMJX investment?
3. What diagnostics do we need before implementing a full differentiable outer loop?

## Plan Overview

### Phase A: Foundational-pose-conditioned MJX benchmark

Goal:
- Compare MJX-autodiff from `open` vs MJX-autodiff from `foundational` keyframe on matched scenes and budgets.

Success criteria:
- Foundational-start consistently improves objective and stability metrics (especially contact persistence and drift/drop).
- Runtime remains comparable to open-start MJX.

Deliverables:
- Per-scene A/B run directories (`open_mjx`, `foundational_mjx`).
- Comparison table with score, lift, contacts, persistence, drift/drop, and runtime.

### Phase B: Hybrid optimizer lane (MJX -> short CEM polish)

Goal:
- Use MJX as a warm-start generator, then run short CEM refinement from that control vector.

Success criteria:
- Hybrid outperforms pure MJX in score/stability with modest extra runtime.
- Hybrid approaches or exceeds long CEM quality at lower wall time.

Deliverables:
- Script option to run two-stage optimization.
- Ablation: MJX only vs MJX->CEM vs CEM only.

### Phase C: DiffMJX readiness + MVP

Goal:
- Add gradient diagnostics first, then implement minimal differentiable outer-loop update.

Success criteria:
- Stable gradients on representative scenes/morphologies.
- Verified finite-difference agreement on key objective terms.

Deliverables:
- Gradient check report.
- Small DiffMJX MVP experiment with fixed seeds and reproducible config.

## Phase A Execution (Started)

### Scene used for first A/B run

- `results/phase1/run_20260413_pollard_multiscene_500_run3/cube/generated_mjcf/scene_cube_tp0d0000p0d0200p0d0000_ip0d0100n0d0123p0d0000_mp0d0100p0d0153p0d0000.xml`

This scene includes a `foundational` keyframe.

### Command executed

```bash
MUJOCO_GL=egl bash -lc 'SCENE="results/phase1/run_20260413_pollard_multiscene_500_run3/cube/generated_mjcf/scene_cube_tp0d0000p0d0200p0d0000_ip0d0100n0d0123p0d0000_mp0d0100p0d0153p0d0000.xml"; OUT="results/phase1/run_20260413_phaseA_fp_conditioned"; uv run python scripts/phase1_optimize_grasp.py --scene-xml "$SCENE" --keyframe open --optimizer mjx-autodiff --iterations 18 --learning-rate 0.035 --grad-clip-norm 5.0 --skip-gif --seed 0 --output-dir "$OUT" --tag open_mjx; uv run python scripts/phase1_optimize_grasp.py --scene-xml "$SCENE" --keyframe foundational --optimizer mjx-autodiff --iterations 18 --learning-rate 0.035 --grad-clip-norm 5.0 --skip-gif --seed 0 --output-dir "$OUT" --tag foundational_mjx'
```

### Initial result (A/B)

- Open-start MJX:
  - score: 1.6070
  - lift: 0.04877
  - contacts: 2
  - contact persistence: 0.45
  - min finger persistence: 0.00
  - xy drift: 0.02065
  - drop-from-peak: 0.05308
  - optimization wall time: 208.85s
- Foundational-start MJX:
  - score: 6.9056
  - lift: 0.04957
  - contacts: 6
  - contact persistence: 1.00
  - min finger persistence: 1.00
  - xy drift: 0.00027
  - drop-from-peak: 0.00004
  - optimization wall time: 207.13s

Interpretation:
- First Phase A result strongly supports FP-conditioned starts: much higher quality with similar runtime.

Artifacts:
- `results/phase1/run_20260413_phaseA_fp_conditioned/open_mjx/summary.json`
- `results/phase1/run_20260413_phaseA_fp_conditioned/foundational_mjx/summary.json`

### Next Phase A batch

1. Repeat the same A/B on prism1/prism2/prism3 representative generated scenes.
2. Run seeds {0,1,2} per scene for variance.
3. Aggregate into one Phase A comparison CSV and add to this doc.

## DiffMJX MVP Follow-Up (Completed)

After Phase A MJX A/B, DiffMJX MVP was executed on the same cube scene with GPU JAX active in the project `uv` environment.

Environment check:

- `jax==0.4.38`
- `jax.default_backend() == gpu`
- `jax.devices() == [CudaDevice(id=0)]`

Commands run (GIF enabled):

```bash
MUJOCO_GL=egl JAX_PLATFORM_NAME=gpu uv run python scripts/phase1_optimize_grasp.py \
  --scene-xml results/phase1/run_20260413_pollard_multiscene_500_run3/cube/generated_mjcf/scene_cube_tp0d0000p0d0200p0d0000_ip0d0100n0d0123p0d0000_mp0d0100p0d0153p0d0000.xml \
  --keyframe foundational \
  --optimizer diffmjx-mvp \
  --iterations 12 \
  --learning-rate 0.03 \
  --grad-clip-norm 5.0 \
  --score-weight-contact-proxy 0.8 \
  --score-weight-ctrl-l2 0.015 \
  --diffmjx-eval-interval 2 \
  --seed 0 \
  --output-dir results/phase1/run_20260413_phaseA_fp_conditioned \
  --tag foundational_diffmjx_mvp

MUJOCO_GL=egl JAX_PLATFORM_NAME=gpu uv run python scripts/phase1_optimize_grasp.py \
  --scene-xml results/phase1/run_20260413_pollard_multiscene_500_run3/cube/generated_mjcf/scene_cube_tp0d0000p0d0200p0d0000_ip0d0100n0d0123p0d0000_mp0d0100p0d0153p0d0000.xml \
  --keyframe open \
  --optimizer diffmjx-mvp \
  --iterations 12 \
  --learning-rate 0.03 \
  --grad-clip-norm 5.0 \
  --score-weight-contact-proxy 0.8 \
  --score-weight-ctrl-l2 0.015 \
  --diffmjx-eval-interval 2 \
  --seed 0 \
  --output-dir results/phase1/run_20260413_phaseA_fp_conditioned \
  --tag open_diffmjx_mvp
```

DiffMJX results snapshot:

- Foundational-start DiffMJX MVP:
  - score: 6.9273
  - lift: 0.05015
  - contacts: 6
  - artifacts: `results/phase1/run_20260413_phaseA_fp_conditioned/foundational_diffmjx_mvp/`
- Open-start DiffMJX MVP:
  - score: 0.3504
  - lift: 0.00017
  - contacts: 1
  - artifacts: `results/phase1/run_20260413_phaseA_fp_conditioned/open_diffmjx_mvp/`

Implementation note:

- The DiffMJX strategy now falls back to forward-mode gradients (`jax.jacfwd`) when reverse-mode fails for dynamic-loop primitives on a given JAX/MJX build, keeping the run functional on GPU.
