# Morphology optimization for a low-force, smooth, in-hand reorient — plan (2026-06-22)

The RL side has reached a **structural** ceiling: no reward lever removes the excess grip
force, because the cause is the hand's geometry, not the policy. This document lays out how
to optimize the morphology to get a smooth, low-force, balanced grasp+reorient, using the
existing RL policies and the existing parametric-morphology infrastructure — and an honest
read on the "Shape Your Body" value-gradient approach the user raised.

## 1. Why morphology now — the RL ceiling is structural (the evidence)

The whole grip-force effort converged on one conclusion: **the excess force is a structural
grip defect, not a reward-tunable quantity.**

- **Force floors ~6.6 N** for a fingertip grip of this rod, regardless of penalty (b32→b34);
  the "B3 is gentle at 3 N" premise was a **phantom** (B3/B4 actually grip 7–9 N).
- **Relaxing verticality** (gentleB) bought smoother + slightly lower force (6.8→5.3 N,
  ang-jerk 74→57) — good, but the grip stayed lopsided.
- **The per-finger probe** (this session) found the grip is a **degenerate pinch**: the
  **thumb is idle (~1.6–1.8 N)** while index+middle clamp ~8 N each. All three touch, but the
  load is on two fingers.
- **The spread penalty** (`grip_force_spread`, built this session to reward a balanced
  tripod) **barely moved it**: thumb 1.6→1.8 N, index 8.0→7.1 N — the thumb stayed idle.

  | policy | thumb | index | middle | force | ang-jerk | held-cos |
  |---|---|---|---|---|---|---|
  | gentleB | 1.6 N | 8.0 N | 6.4 N | 5.3 N | 57 | 0.64 |
  | + spread penalty | 1.8 N | 7.1 N | 6.3 N | 5.1 N | 66 | 0.60 |

  **The policy cannot recruit the thumb into a load-bearing opposition** — its placement
  doesn't let it oppose the other two against this object without dropping it. That is a
  **geometry** problem.
- **Contact-hardening** (this session) showed the soft contact is **functionally
  load-bearing** — even mild stiffening broke Policy A's grasp entirely (object never left the
  floor). So the visible penetration is a symptom of a marginal grip, again **structural**.

→ **The remaining lever is the hand geometry.** And the two structural defects map cleanly
onto morphology knobs: (i) **recruit the thumb** (reposition it for true opposition → balanced,
lower-peak grip), and (ii) **enable seating** (let the object reach the palm so the palm bears
load and the fingers relax to ~3 N — the documented true-low-force path; today the object hangs
7–8 cm below the palm, `palm_brace_force` has fired in **0** of all runs).

## 2. What we already have (a large head start)

A developed parametric-morphology / sampling-based-grasp system from prior work:

- **`src/morphohand/sampling/morphology.py`** — the design vector: **9 params** = per-finger
  (thumb/index/middle) base position **(x, y)** + **length**, with bounds + clipping + samplers
  (`MorphologyValues`, `MorphologyBounds`, `sample_morphologies`, `clip_morphology`).
- **`scripts/generate_morphology_xml.py`** — design vector → scene XML.
- **`scripts/phase1_optimize_grasp.py`** + `multimorph_run_pipeline.py` /
  `multimorph_pick_candidates.py` — CEM grasp optimization + multi-morphology pipeline.
- **`src/morphohand/optimization/phase1_common.py`** — the optimization backbone.
- Today's **per-finger / force / jerk / seating diagnostics**
  (`rl_demo_handoff_continuous.py`, `scripts/probe_grip_balance.py`) — the design-quality
  readout.

So we do **not** build parametric morphology from scratch; we wire the existing generator to a
task-quality score and search it.

## 3. The design objective (matches the user's priorities)

A scalar score from a **rollout** of A+B on a candidate morphology (ground-truth — see §4 on
why rollout, not value gradient):

```
score =  w_hold   · 1[min-z > 0.05]           # must not drop
       − w_force  · mean_fingertip_force        # LOW force  (primary)
       − w_spread · (max_finger − min_finger)   # BALANCED   (recruit the thumb)
       − w_jerk   · ang_jerk                     # SMOOTH     (primary)
       + w_seat   · palm_force                   # SEATING bonus (the true low-force unlock)
       + w_cos    · held_cos                     # reorient "enough" (de-prioritized)
```

Weights follow the user's stated order: smooth + low-force + balanced ≫ verticality.

## 4. The "Shape Your Body" (VGDS) approach — honest fit for our task

**Method.** Train one **morphology-conditioned universal** policy + critic (URMA) across ~50
robots; **freeze**; optimize a new design by gradient-ascending the critic's value
**∇_design V(s, design)** inside a trust region, over 1100+ design params, ~1–2 min/design.
Demonstrated on **legged locomotion only**.

**Why it's appealing:** design search becomes cheap (gradients, no per-design retrain).

**Why it is hard for *our* task (the user's instinct, made precise):**
1. **The value gradient does not exist for us yet.** ∇_design V requires a critic that is a
   *function of* morphology and *generalizes across* morphologies. Ours is trained on one fixed
   hand; the observation contains **no design parameters**, so there is nothing to differentiate.
   We would first have to build a morphology-conditioned universal policy/critic.
2. **Brittleness makes the surrogate untrustworthy.** Our reorienter memorizes a **razor-thin
   basin** around one grip (B3/B4 crater held-cos 0.98 → 0.26 under a sub-cm perturbation,
   measured earlier). A universal policy would have to reorient across a *distribution* of hands
   — but we can barely train **one**. The critic's value on an unseen morphology would be
   **fantasy** (the policy fails off-distribution), so the gradient points at imagined, not real,
   performance.
3. **Contact-rich ≠ locomotion.** A small geometry change flips the contact mode (we just saw
   mild contact-stiffening break A's grasp entirely). Value/sim gradients through stiff,
   intermittent contact are notoriously noisy. The **same caveat sinks naive
   differentiable-simulation** morphology gradients (MuJoCo-Warp is differentiable, but contact
   gradients here would be unreliable).

**Net.** Borrow VGDS's *core idea* — use the RL policy/value as a cheap design surrogate — but
**not** its zero-shot value-gradient mechanism. Our design space is **9 params, not 1100**, so we
don't need gradient efficiency; the binding constraint is making design **evaluation reliable
despite brittleness**. That points to rollout-based scoring (ground truth), with VGDS-style value
guidance only as a *guarded accelerator* once a robust critic exists (Stage 2).

## 5. The plan (staged, brittleness-aware)

**Stage 0 — Objective + evaluator harness (cheap; do first).** A single function:
`design vector (9) → generate_morphology_xml → build scene → roll out A+B → §3 scalar score`,
reusing the session's per-finger/force/jerk/seating diagnostics. This is the reusable unit
every later stage calls.

**Stage 1 — DE-RISK with hand-picked morphologies (the recommended FIRST move; no new infra).**
The make-or-break question is whether morphology even *moves our metrics*. Test 3–4 hypotheses by
**retraining B** (~40 min each — ground-truth, immune to the brittleness that breaks zero-shot
transfer) on a fixed candidate:
  - **(a) Thumb true-opposition** — move `thumb_x/thumb_y` so the thumb genuinely opposes the
    index/middle line against the object. *Target: thumb becomes load-bearing, force balances,
    peak drops.* (Directly attacks the structural cause the spread penalty couldn't.)
  - **(b) Seating via lengths** — shorten index/middle and/or lengthen the thumb (or all three)
    so the gripped object rides **higher**, toward the palm. *Target: `palm_force > 0`.*
  - **(c) Closer palm / longer fingers** — bring the object to the palm so it seats and the
    fingers relax. *Target: fingertip force → ~3 N.*
  - **(d) [extension, needs new params] larger / higher-friction fingertips** — a bigger contact
    patch or stickier pad holds at lower normal force. (Adds fingertip radius/friction to the
    9-param vector.)
This validates that morphology is the lever, shows *which* params matter, and validates the
score — **without** the DR/surrogate machinery. Coordinate-descent by hand before automating.

**Stage 2 — Automated search (only if Stage 1 shows morphology helps).**
- Small space ⇒ **CEM / Bayesian-opt over the 9-param vector**, scored by Stage-0 rollouts; the
  existing phase1 CEM infra is the backbone.
- **Brittleness fix (our DR analog of VGDS's universal policy):** train A+B with **morphology
  domain randomization** over a *narrow* range so one policy can evaluate a neighborhood of
  designs without per-design retraining; widen as it holds. If DR fails (brittleness wins), fall
  back to **per-design retrain** — slow but reliable, and a 9-D space is small enough to afford it.
- **VGDS-inspired accelerator (optional, last):** once a DR-robust critic is trustworthy over the
  range, use its value as a cheap pre-**ranking** (finite-difference or gradient), then **verify
  the top-k by full rollout**. This buys the paper's speed where the critic is reliable, guarded
  against contact-brittleness.

**Stage 3 — Finalize.** Retrain A+B from scratch on the winning morphology (full quality), then
the deferred **hard-contact sim-to-real pass** (now feasible — we're rebuilding the lineage anyway).

## 6. Honest assessment + recommended first move

- This is a multi-week effort; the make-or-break unknown is whether a **balanced, seated** grip is
  even reachable in the 9-param space, or whether it needs fingertip/palm-geometry extensions (d).
- **RECOMMENDED FIRST EXPERIMENT — Stage 1(a):** reposition the thumb for true opposition, retrain
  B, measure per-finger balance/force. It is the cheapest test of "does fixing the thumb geometry
  recruit it and lower the force," it directly targets the structural cause the spread penalty
  could not, and it reuses everything we have. ~1 hour. If the thumb engages and the index/middle
  8 N drops, morphology is confirmed as the answer and we proceed to Stage 1(b/c) seating, then
  Stage 2 automation.

**One-line summary:** the grip defect is structural (idle thumb, no seating), so the lever is the
9-param finger geometry; evaluate designs by *rollout* (brittleness makes VGDS's zero-shot value
gradient unreliable), starting with a hand-picked thumb-opposition retrain before automating the
search.
