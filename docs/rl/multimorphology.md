# Multi-Morphology Adaptation

How do we get a single grasp policy to work across the hand-morphology
candidates in `run18_final` (2000 cube candidates with varying finger
attachment points + link lengths)? This page documents the research
review, the chosen first experiment, and the open follow-ups.

## Problem

- The "morphology" is a 9-dim vector per candidate: `(thumb_x, thumb_y,
  thumb_len, index_x, index_y, index_len, middle_x, middle_y, middle_len)`
  — fingertip attachment offsets on the palm + a length scalar per finger.
- All candidates use the **same kinematic structure** (3 fingers × 3
  joints + 6-DOF palm). Only attachment points + lengths change.
  This is "intra-embodiment" morphology variation, not the
  cross-embodiment Leap↔Allegro↔Shadow problem.
- The current policy (`cube_stable_v1/model_1400`) was trained on
  exactly one morphology (`candidate_id=0`), with the `LerpFinger`
  setpoint pinned to candidate-0's CEM `best_finger_ctrl`.
- Two parts of the policy stack are morphology-dependent:
  1. The **`LerpFinger` setpoint** (`target_ctrl`) — currently CEM-optimized
     per morphology. Wrong setpoint for a new morphology = wrong grip
     pose; the policy's residual can't recover.
  2. The **policy network** — implicitly assumes the kinematics it was
     trained on. Residuals it outputs may or may not be useful on a
     different geometry.

## What the literature does

| Paper (year) | Approach | Lesson for us |
|---|---|---|
| [Pollard / Li (2007)](https://www.ri.cmu.edu/pub_files/pub4/li_ying_2007_1/li_ying_2007_1.pdf) — *Data-Driven Grasp Synthesis Using Shape Matching* | Database of human grasps; shape matching from object → grasp candidate. Per-hand re-execution. | Not directly applicable (we're not doing shape DB lookup), but the *eval-against-database* pattern matches our "test on K morphs" idea. |
| [Emergent Hand Morphology (2020)](https://arxiv.org/pdf/2012.12209) | Jointly optimize morphology + control via RL. | Confirms morphology + policy co-design works in principle. Tangential for us — we already have a fixed morphology set. |
| [UniMorphGrasp (2026)](https://arxiv.org/pdf/2602.00915) | Diffusion model conditioned on a **morphology embedding** + an **eigengrasp set** derived from the hand's morphology description. Cross-embodiment grasp synthesis. | Directly proposes morphology embedding + eigengrasp compression — the exact alternatives we mentioned. |
| [DexFormer (2026)](https://arxiv.org/pdf/2602.08278) | History-conditioned transformer infers morphology + dynamics on the fly from temporal obs context. Zero-shot to Leap, Allegro, Rapid. | "Infer morphology from obs history" — *no explicit conditioning vector needed*. We'd need a recurrent / transformer policy though. |
| [DexGrasp-Zero (2026)](https://arxiv.org/html/2603.16806) | Morphology-aligned policy for **zero-shot** cross-embodiment grasping. 82 % success on unseen objects across 3 robot platforms. | Demonstrates that one policy can serve multiple embodiments without per-hand finetune. Their setup is much more diverse than ours though (different fingers, different DOFs). |
| [House of Dextra (2025)](https://arxiv.org/pdf/2512.03743) | Pre-train a morphology-conditioned cross-embodied policy + per-design finetune. | Validates the "pretrain + finetune" pattern as a baseline. |

## Five paths we could take

| Path | What it does | Cost (tonight) | Risk |
|---|---|---|---|
| **A. Eval-and-finetune (chosen)** | Train on one morphology, eval on K others, finetune those below threshold. | Low — uses existing scripts. | Doesn't scale beyond the training distribution. |
| **B. Train-time morphology DR** | At reset, randomly perturb hand morphology within a range. Policy learns robustness. | Medium — mjlab needs per-env scene swap. | Unclear if mjlab supports per-env entity geometry. |
| **C. Explicit 9-dim morphology obs** | Concatenate the 9-dim morphology vector to the actor's observation. | Low. | Network may not learn to use it; high-dim conditioning often gets ignored. |
| **D. Compressed (eigengrasp-style) morphology embedding** | PCA / VAE over the 9-dim morphology → low-dim embedding → policy obs. | Medium — needs fitting + integration. | The 9-dim space is already small; compression may not help. |
| **E. History-conditioned policy (DexFormer-style)** | Recurrent / transformer policy infers morphology from obs window. | High — rewrites the actor architecture, retrains from scratch. | Highest payoff if it works, but biggest scope change. |

## Decided first experiment (tonight)

**Path A — Eval-and-finetune.** Reasons:

1. Cheapest path to a real degradation curve, which we need before
   choosing between B–E.
2. The threshold for "needs finetune" is the actual deliverable —
   without measured degradation we'd be guessing.
3. Reuses our entire existing stack (rl_eval_object.py,
   rl_train_cube.py + `--init-actor-checkpoint`).

### Pipeline

```
[ candidate_id=0 (trained policy)  ─────────────────────────┐
                                                            │
[ pick K morphologies at varying d(M_i, M_0) ]              │
   │                                                        │
   ├── for each M_i:                                        │
   │       if no foundational CEM run:                      │
   │           rerun CEM (scripts/phase1_optimize_grasp.py) │   (slow, GPU)
   │       eval cube_stable_v1/model_1400 on M_i:           │   (fast, GPU)
   │           record success rate, contact_min, drift      │
   │                                                        │
   ├── threshold: success_rate_6cm < 80 % => finetune       │
   │                                                        │
   └── finetune cube_stable_v1/model_1400 on each below     │
       threshold (250-iter run, --init-actor-checkpoint)    │   (slow, GPU)
```

### Choice of K candidates

`results/phase1/run18_final/cube/all_candidates.csv` ranks all 2000 by
CEM score. The "varying distance" axis is the 9-dim Euclidean distance
from `candidate_id=0`'s morphology vector. Pick:

| Rank in candidates | What it tests |
|---|---|
| `candidate_id=0` (training morphology) | Sanity baseline; should hit 100 %. |
| Nearest neighbor (d=ε) | Near-zero generalization gap; should hit ~100 %. |
| Median-distance (d=p50) | Typical case; the actual generalization question. |
| Far (d=p95) | Edge of the distribution; expected to need finetune. |

### Caveat — the `LerpFinger` setpoint is also morphology-dependent

Each `M_i` needs its own CEM `best_finger_ctrl` for the `LerpFinger`
setpoint to be a valid open-loop grip pose. If we skip the CEM rerun
and reuse `M_0`'s setpoint, we're measuring "*policy* generalization
under a *wrong* open-loop schedule" — which underestimates true
policy generalization. The cleanest experiment runs CEM per morphology.

For tonight's batch, we run CEM on the 3 picked morphologies *that
don't already have a foundational run* (typical: all of them except
`candidate_id=0`). This is the bulk of the wall-clock cost.

### Success metric

Eval per morphology (deterministic, 64 envs, cube DR x ±20mm, y ±5mm,
yaw ±0.52rad — same as cube_stable_v1):

- **`lift_success_6cm`** — pass bar at ≥ 80 %.
- `contact_min_hold` — must be ≥ 0.9 for a "stable grasp".
- `cube_xy_drift_hold_mean_m` — should be < 5 mm.

If `lift_success_6cm < 80 %`, the morphology needs finetune. Finetune
budget per morphology: 250 iters (~12 min wall-clock) starting from
`cube_stable_v1/model_1400` actor weights.

## Where this stops + what's next (paths B–E)

The eval-and-finetune baseline is necessary but not the endgame: if
every new morphology needs a finetune, the cost is linear in
morphology count.

**Next experiments to run (separate sessions):**

- **Path B (morphology DR)**: most promising for our intra-embodiment
  case. The 9-dim morphology space is small enough that uniform
  perturbation could give zero-shot coverage. Needs mjlab per-env
  scene-swap support — to be investigated.
- **Path C → D (morphology obs)**: lowest-effort follow-up. Add the
  9-dim vector to the actor obs at train time. If it doesn't help,
  try Path D with a 2-3 dim PCA embedding.
- **Path E (DexFormer-style)**: only if A-D plateau. Largest scope
  change; expected to dominate the others if it works.

## Sources

- [Data-Driven Grasp Synthesis Using Shape Matching and Task-Based Pruning](https://www.ri.cmu.edu/pub_files/pub4/li_ying_2007_1/li_ying_2007_1.pdf)
- [Emergent Hand Morphology and Control from Optimizing Robust Grasps of Diverse Objects](https://arxiv.org/pdf/2012.12209)
- [UniMorphGrasp: Diffusion Model with Morphology-Awareness for Cross-Embodiment Dexterous Grasp Generation](https://arxiv.org/pdf/2602.00915)
- [DexFormer: Cross-Embodied Dexterous Manipulation via History-Conditioned Transformer](https://arxiv.org/pdf/2602.08278)
- [DexGrasp-Zero: A Morphology-Aligned Policy for Zero-Shot Cross-Embodiment Dexterous Grasping](https://arxiv.org/pdf/2603.16806)
- [House of Dextra: Cross-embodied Co-design for Dexterous Hands](https://arxiv.org/pdf/2512.03743)
