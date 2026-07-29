# Force-closure energy as a continuous contact-quality term

## What this is

A continuous, geometry-aware replacement for the plain
`objective_weight_contact * contact_count` term in the Phase-1 objective.
Contact *count* is a coarse signal: three tip-object pairs from a poor grasp
score the same as three pairs from a textbook 120° wrap. This branch adds a
force-closure quality score derived from the resolved contact set:

- **normal balance** — penalize `|| Σ_i n_i ||` (opposed normals cancel out;
  non-cancelling normals can't form a static hold).
- **wrench spread** — reward the geometric mean of singular values of the
  contact wrench matrix; rewards non-degenerate wrench-space coverage.
- **Q1 distance** — the Ferrari-Canny / DexGraspNet DFC residual:
  distance from origin to the convex hull of the friction-cone-discretized
  wrenches. Zero means force closure is achievable from the current contacts.

Reference: Liu et al., *Synthesizing Diverse and Physically Stable Grasps
with Arbitrary Hand Structures using Differentiable Force Closure Estimator*
(2021), used by DexGraspNet.

## Files

- `src/morphohand/optimization/force_closure.py` — `ContactWrench`,
  `extract_finger_contacts` (from `mujoco.MjData`), Q1 distance solver
  (Frank-Wolfe on the simplex over cone-discretized wrenches), spread,
  balance, and a combined `force_closure_metrics` entry point.
- `src/morphohand/optimization/phase1_common.py` — evaluator now samples
  FC at the end of the settle phase (when CEM has placed the fingers).
  Five new config knobs:
  - `objective_weight_force_closure` (default 0.0 = backwards-compatible)
  - `force_closure_friction_mu` (default 0.5)
  - `force_closure_cone_edges` (default 4)
  - `force_closure_weight_balance` (default 0.5)
  - `force_closure_weight_q1` (default 1.0)
- `tests/test_force_closure.py` — math + integration unit tests.

## Combined score

```
fc_score = wrench_spread
         - weight_balance * normal_balance
         - weight_q1 * q1_distance
```

This is added to the objective with `objective_weight_force_closure`. Set to
0 (default) to recover the original behaviour; set to ~2-5 to make CEM
prefer wrap-style grasps over single-finger pokes when contact counts tie.

## How to use

```python
from morphohand.optimization.phase1_common import Phase1GraspEvaluator, Phase1EvalConfig

cfg = Phase1EvalConfig(
    objective_weight_force_closure=2.0,
    force_closure_friction_mu=0.5,
    force_closure_cone_edges=4,
)
ev = Phase1GraspEvaluator(
    scene_xml="assets/mjcf/baseline/scenes/scene_power_drill_short_proximal.xml",
    keyframe="open_flat",
    cfg=cfg,
)
score, metrics = ev.evaluate(finger_ctrl)
# metrics["fc_q1_distance"], metrics["fc_wrench_spread"], etc.
```

## What to check experimentally

- Confirm that on the existing failure case "only six contacts (middle
  finger never engages)" mentioned in the README, `fc_fingers_engaged`
  reports 2 not 3, and the FC score drops accordingly — i.e. it correctly
  diagnoses the failure mode.
- The expected qualitative effect of adding this term: CEM solutions
  prefer to spread tips around the object (Q1 → 0, balance → 0) rather than
  pile all three tips on one face, even when both options yield the same
  contact count.
- Knob to scan: `force_closure_weight_q1` (1 → 3) controls how aggressively
  CEM is pushed toward an actually-closeable wrench space.

## Limitations

- Not differentiable through MuJoCo's contact resolution. Suitable as a
  better CEM cost; not as a JAX gradient signal. For the gradient version
  see Liu et al.'s SDF-based DFC formulation, or the more recent GraspQP
  (CoRL 2025) which formulates FC as a differentiable QP.
- Sampled once at end-of-settle. A per-phase or time-averaged version is
  straightforward to add but doubles the contact extraction cost.
