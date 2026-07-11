# Morphology-conditioned policy — mjwarp per-env geometry feasibility spike

**Date:** 2026-07-11 (idle-tick CPU spike, decision-tree step 8a).
**Question:** can ONE policy train across per-world randomized morphologies (conditioned on the
9-vector), so evaluating a design no longer requires optimizing a fresh policy for it? This is
the structural fix to the draw-noise bottleneck the probe suite identified (single-draw cos
sd 0.3–0.5, gate-invisible; 2-draw means still sd ≈ 0.32).

**Verdict: FEASIBLE — the sim stack already supports it natively. No mjwarp fork needed; the
work is project-side data plumbing (est. 2–4 days).**

## Evidence (vendored `external/mujoco_warp` @ 36fc8be + installed mjlab)

1. **mjwarp model fields are batchable by design.** Every geometry-relevant Model field is
   declared with a leading `*` dim (`types.py`: `body_pos: array("*", "nbody", vec3)`,
   `geom_size: array("*", "ngeom", vec3)`, same for `geom_pos`, `site_pos`, `body_ipos`,
   `body_iquat`, `body_mass`, `body_inertia`, `qpos0`, `jnt_pos`, `jnt_axis`, …) and **every
   kernel reads them as `field[worldid % field.shape[0], id]`** — verified in kinematics
   (`smooth.py:108,161-162,202,223`), CoM/inertia (`smooth.py:473,525-526`), collision
   broadphase (`bvh.py:197` — which already tiles `geom_size` to nworld for rendering at
   `bvh.py:303`), narrowphase (`collision_core.py:102,108`, `collision_convex.py:120`),
   sites/sensors, and render. Shape (1, n, …) broadcasts; shape (nworld, n, …) is per-world.

2. **mjlab exposes the whole mechanism.** `Simulation.expand_model_fields(fields)`
   (`mjlab/sim/sim.py:248`) expands named fields to nworld, clears the torch bridge cache,
   recreates the sensor context, and **recaptures the CUDA graphs** (the one real footgun —
   stale graphs silently read old arrays — is already handled). Plus
   `get_default_field(field)` (cached originals) and `recompute_constants(RecomputeLevel)`
   (post-DR derived constants, wired into the event manager at `event_manager.py:329`).
   This is mjlab's domain-randomization path; per-world morphology is exactly a startup-mode
   DR term over placement/size fields.

## What the project actually has to build

1. **Same-topology check (prerequisite, trivial):** all our designs come from
   `generate_morphology_xml.py` on one template — same tree, nbody/ngeom/njnt identical;
   only pos/size/inertial values differ. Assert this once.
2. **Per-design field scatter:** compile each design's frozen XML to an `MjModel` on CPU
   (cheap, ~ms each), `np.stack` the differing fields
   (`body_pos, body_ipos, body_iquat, body_mass, body_inertia, geom_pos, geom_size,
   site_pos, qpos0`), call `expand_model_fields(...)` at env startup, write the stacked
   values through the model bridge, then `recompute_constants` at the strongest level once.
   Compiling per-design models makes MuJoCo do the inertia-from-geometry bookkeeping — no
   hand-rolled mass math.
3. **Per-world reset keyframe:** each design needs its own `open_ik` keyframe
   (`retarget_keyframe_ik.py` is CPU-cheap, run for all designs up front); reset writes
   per-world qpos from a (nworld, nq) table instead of one keyframe. Data-side only.
4. **Policy conditioning:** append the normalized 9-vector (or delta-to-m05) to the obs.
5. **Sampling schedule:** resample designs per rollout bucket (fixed per world within a
   rollout; CUDA graph is fine since values change, not shapes/addresses — same contract DR
   already relies on).

## Open risks (small, checkable in day 1)

- Live-A reset: the frozen Policy A that drives B's pre-onset lift is per-design today. A
  conditioned B needs either a conditioned A first (train A across morphologies the same
  way — likely the right order) or per-world A checkpoints (memory-heavy, avoid).
- `contact pair`/exclude tables are geometry-independent for us (same topology) — should be
  unaffected, verify no `pair_*` field depends on size.
- Watch one thing empirically: whether PPO across mixed morphologies destabilizes the lift
  curriculum (start with a narrow box around m05, widen).

## Why this matters (tie-back)

The probe suite concluded the policy DRAW is THE landscape bottleneck: per-design
from-scratch training is a noisy evaluator (P4 is paying 2 full draws/design to average it
down). A single conditioned policy amortizes optimization across ALL designs — evaluation
becomes a forward rollout sweep over the 9-box (minutes, deterministic), and the landscape
map stops being optimizer-noise-limited. P4's replicated map doubles as the
validation set for the conditioned policy's per-design predictions.
