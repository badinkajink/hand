# Open-loop carry: the mass envelope, and a success criterion that scores a dropped shaft

2026-08-30. `scripts/probe_real_v1_carry.py --object-mass`, design `rv05_manual_stored`
(the built hand), published carry parameters `--linear-anchor --angle-deg -90 --axis-k 0.25
--turn-steps 550 --budget 0.5`, 3 repeats per mass.

The appendix named object mass as the parameter most likely to invalidate a hardware claim —
*"the simulation cylinder mass is 24.5 g and doubling that mass causes the present open-loop
solution to fail"* — and until now the probe had no way to vary it. It does now.

**Baseline reproduces exactly** before anything was swept: axis height 8.2 mm, peak 0.996,
final 0.996, object z 0.1130, 3 contacts, 12.02 N. Identical to the published row.

## The envelope

| mass | peak | final | object z | contacts | kept |
|---:|---:|---:|---:|---:|---|
| 24.5 g (nominal) | 0.996 | 0.996 | 0.113 | 3 | 3/3 |
| 35 g | 0.975 | 0.975 | 0.111 | 3 | 3/3 |
| 50 g | 0.984 | 0.980 | 0.107 | 3 | 3/3 |
| 60 g | 0.996 | 0.965 | 0.104 | **2** | 3/3 |
| 65 g | 0.997 | 0.995 | 0.101 | **2** | 3/3 |
| 70 g | 1.000 | 0.999 | 0.050 | **0** | **0/3** |
| 75 g | 0.999 | 0.000 | 0.012 | 0 | 0/3 |
| 100–150 g | 0.000 | 0.000 | 0.012 | 0 | 0/3 |

**The carry tolerates ~2.7x nominal, failing between 65 and 70 g — not at 2x.** 50 g holds at
0.980 on three contacts. The appendix's stated failure point was wrong in the conservative
direction, which is worth correcting because it is the number a reviewer would use to judge
whether the hardware claim is reachable.

Degradation is not graceful, and it announces itself one step early: **three contacts become
two at 60 g**, before any loss of the object. That is the same degenerate-grip signature the
trajectory-health scorecard exists to catch, so 60 g is the honest edge of the envelope even
though 65 g still scores 3/3.

## The 70 g row is why a height threshold cannot be the success criterion

At 70 g the shaft reads **final alignment 0.999 at z = 0.0503 m with zero contacts**. It has
been dropped, landed on its butt, and is standing upright on the table. A vertical shaft on a
table is perfectly aligned with the vertical.

The appendix's stated rule was *"kept when hold-phase minimum object height exceeds 0.05 m"*.
0.0503 > 0.05. **That rule scores a total failure as a near-perfect reorientation**, by 0.3 mm.

`probe_real_v1_carry.py` already gets this right (`ok = nh >= 1 and min_z_hold > start_z −
0.02`: contact AND height relative to the post-lift height, over the whole hold phase, not the
last step) — its own comment says a shaft standing on the table also reads cos 1.0. The
implementation was correct and the paper's description of it was not. Now fixed in
`paper/appendix.tex`.

This is the third independent appearance of the same trap in one day:

* the observation ablation: `peak_cos` 0.54–0.97 in every *failed* condition, measuring the
  shaft rotating on its way to the floor;
* the 75 g row: peak 0.999, final 0.000;
* this row: final 0.999 *and* a passing height, defeated only by the contact count.

Peak alignment is not a metric. Height alone is not a metric. **Held alignment, on rollouts
still in contact, relative to the post-lift height.**

## What is NOT shown here

**Whether more grip extends the envelope is untested.** Sweeping `--squeeze` at 0.004 and 0.006
returned byte-identical numbers at every mass, which is the signature of a parameter that did
nothing rather than a parameter that did not matter: on the `--morph-run` path the grip comes
from the CEM `best_finger_ctrl`, and `squeeze` is only consumed by `_grip_from_fit` on the
`--scene` path. The test had no teeth. Recorded as untested, not as negative — and it is worth
running properly, because the bench already found that clamp force and reorientation trade
directly, so a grip that buys mass headroom may cost the turn.

## SUPERSEDED for deployment purposes (same day)

The cliff below is real for `rv05_manual` at `axis_k 0.25` with the CEM grip. It does **not**
describe the deployed plan. Sweeping mass through g12's own exported CSV
(`20260830-deploy_renders/`) gives a flat 0.779 -> 0.790 from 24 g to 85 g with no cliff,
because g12 clamps 10 mm of pad squeeze where this sweep used the CEM grasp. Mass tolerance
belongs to the plan, not to the hand. Sweep the plan you intend to run.

The measured objects -- a 24 g bench cylinder and a 65 g screwdriver -- are both comfortably
inside g12's range.

## The objects, measured (2026-08-30)

The bench cylinder is **24 g** and the intended screwdriver is **65 g**. Against the deployed
g12 plan both are inside the range with no cliff; against this rv05_manual carry, 65 g is the
last passing mass. Deploy on g12.

## Reproduce

    uv run --extra rl python scripts/probe_real_v1_carry.py \
      --morph-run results/phase1/real_v1/rv05_manual_stored \
      --linear-anchor --angle-deg -90 --axis-k 0.25 --turn-steps 550 --budget 0.5 \
      --object-mass 0.050 --repeats 3

`--object-mass` sets kilograms, `--mass-scale` multiplies. Both scale `body_inertia` with
`body_mass`, since scaling mass alone would leave the shaft rotationally heavy and quietly
change the dynamics the sweep is trying to measure. The override is applied at every model
load — grip fit, IK scratch model, live rollout — because a fit that picks a grip for one
object and a carry that runs it on another is a silent parity bug.
