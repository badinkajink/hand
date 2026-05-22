# Reinforcement Learning Pipeline

## Why

Phase 1 grasp synthesis via CEM has hit a quality ceiling on the harder
objects in `run18_final` (drill, small flat screwdriver) and is generally
inefficient. The next move is a **learning-based policy** that matches —
and ideally exceeds — the per-task CEM scores on the *easier* end of the
run18 distribution first, providing a foothold for later cross-morphology
transfer and in-hand manipulation work.

## MVP scope

| Aspect | Choice | Why |
|---|---|---|
| Task | Cube pickup | Easiest object in run18; CEM hits +29.45 on the top morph |
| Morphology | `candidate_id=0` of run18_final | Cube score +29.45, persistence 1.0, lift 49.4 mm — known-good basin |
| Algorithm | On-policy PPO | Standard for Isaac Lab–style manipulation; via `rsl-rl-lib` |
| Simulator | mjlab + MuJoCo Warp | GPU-parallelised, ~313× faster manipulation vs MJX |
| Warm start | Opt2Skill-style trajectory tracking | Reuse CEM rollout as reference; reward = task + tracking |
| Docs | This mkdocs-material section | One toolchain with the rest of the repo |

## What is in scope

- Single `(task, morphology, algorithm)` MVP.
- Reference-trajectory-augmented reward (Opt2Skill).
- mjlab manager-API env, RSL-RL PPO trainer.
- uv-unified environment with `[gpu]` and `[rl]` extras.

## What is out of scope (do **not** expand without a new conversation)

- Cross-morphology transfer / Pollard-style fine-tuning across the run18 set.
- In-hand reorientation, pivot, regrasping.
- Multi-task RL (drill, screwdrivers). Drill is also blocked by the keyframe
  issue documented in the CEM phase results.
- Asymmetric actor-critic, DAgger/AWAC, RL-from-CEM distillation.

## Reading order

1. [Architecture](architecture.md) — modules, observations, rewards, the
   reference-trajectory loader.
2. [Training](training.md) — env setup with `uv`, launch command,
   hyperparameter knobs, troubleshooting.
3. [Results](results.md) — MVP numbers vs CEM baseline (filled in after
   first successful run).

## Open question we explicitly punted

**Palm: scripted vs learned.** The plan recommended scripting the palm 6
DOFs and learning only the 9 finger ctrls, on the argument that the palm
is just a settle→lift profile and isolating credit-assignment to the
fingers is easier. In the current implementation the palm is exposed as
learned with a *small per-dim scale* (residual policy), because mjlab's
`ActionTerm` API made the fully-scripted path awkward and the residual
form is one config flip away from either extreme. Revisit once cube
training converges and we have a measurable answer.
