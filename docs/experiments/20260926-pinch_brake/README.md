# Pinch brake on real_v1: runbook and handoff (2026-09-26)

Status file: `logs/20260926-pinch_brake.status` (written by the driver once it exists).
Background Claude session **06fb23c2** ("pinch-brake"): started 2026-09-26 09:45 MDT with `claude --bg --resume
cd85fabe-aef2-4ee8-b595-97b09954e3ca`; list with `claude agents`, attach with `claude attach <id>`.

## Why

Wang, Oh, Pollard, "Relative Contact Velocity-Controlled Hand-Object Mechanism for Dexterous Tool
Manipulation", arXiv 2609.25619 (22 Sep 2026). Their screwdriver load = pinch hard, lift, reduce
the pinch force so the pinch acts as a force-modulated revolute joint and gravity rotates the tool
about the pinch axis, then a pusher finger aligns it, then a tripod regrasp. On real_v1 that is the
2026-09-16 pinch-and-swing (user: PLAN Z, rejected as ballistic) plus a brake and a pusher.

Done already (commit beeaed47): `scripts/real_v1_pusher_reach.py` ->
`docs/experiments/20260926-hom_pusher_reach/pusher_reach.json`. The middle (only free finger)
sits at full extension in all six swing grasps: 1.5-2.5 mm of a 10 mm request downward, 9.9 mm
upward; IK residual 6-11 mm at 10 deg about the pinch axis, 47-91 mm at 90 deg; its normal force
aids the gravity swing. The pinch-axis angles in that JSON (25-73 deg to the shaft) are NOT
reliable: thumb/index pads sit 6-11 mm inside the shaft in the rigid anchor pose, so the witness
points are ill-defined. Measure the pinch axis from settled contacts in the dynamic sim.

## The brake prediction (corrected)

The chat reply of 2026-09-26 wrote cos(theta_max) = mu_t / (2 mu d). That has a factor-2 error
(both pads brake). With per-pad normal force N, torsional coefficient mu_t (MuJoCo condim 4:
torque = mu_t x N) and the CG at perpendicular distance d from a horizontal pinch axis:

* quasi-static balance, decoupled cone: m g d cos(phi) = 2 mu_t N, N >= N_min = m g / (2 mu)
  -> end angle cos(phi_max) = mu_t / (mu d)
* MuJoCo's elliptic cone couples torsion with the tangential load the pads carry (the weight):
  tau_max per pad = mu_t sqrt(N^2 - (f_t/mu)^2) with f_t = m g / 2, so
  cos(phi) = 2 mu_t sqrt(N^2 - N_min^2) / (m g d), and phi -> 90 deg as N -> N_min (then linear slip).
* phi = elevation of the tool axis toward vertical (tip down, signed cos = R[2,2] of the tool body).
* This holds only if the pinch axis is perpendicular to the shaft; an oblique axis at angle beta
  caps the elevation reachable by pure rotation at beta.

## Experiment 1 (the queue)

Chain machinery: `probe_real_v1_chain.chain()` via `real_v1_chain_hands` (`prepare`, `plant_scene`,
`BASE`), `turn_ctrl` hook (called every 10 sim steps in the turn phase, returns {joint: ctrl}),
`step_hook` for the per-step tracer, `turn_relief` dict for the middle release
({"middle": 0.02} = middle 20 mm out; negative = pushed in). Torsional friction via
`real_v1_turn_probe.contact_variant(scene, None, None, None, torsional=mu_t)` on top of the plant
scene. Cached swing grasps in `assets/mjcf/experimental/20260904-chain_bothsets/`
(D8 `sv1_w0099_b100_y180_sq10`, D6 ta20 `sv1_u0308_b050_ta20_y180_sq10`, D7 ta20
`rv05_manual_b85_ta20_y180_sq10`, D2 ta30 `sv1_w2360_b075_ta30_y180_sq10`,
D4 ta30 `g12_b095_ta30_y180_sq10`). 09-16 swing command: `--plant cal --angles 0 --pinch middle:20
--yaws 180 --squeeze-mm 10`.

Modes per cell: `step` (09-16 control: middle released, pinch unchanged), `ramp` (middle released,
then thumb+index relieved radially 0 -> r_end over T s, open loop), `servo` (thumb+index relief
driven by the tool angle, as the bench tag would give it: phi_ref ramps at a set rate to ~85 deg,
PI on the angle error with rate limit and a slip guard; then optionally re-close the middle on the
shaft where it can reach and hold). Sweep: hands D8/D6ta20/D2ta30 (d spans ~10-36 mm), plants
`cal` and `calfast` (brackets the unresolved finger damping), mu_t 0/0.001/0.003/0.01, ramps
1/3 s, seeds (spawn jitter 0.5 mm). Record per 50 Hz sample: phi, per-pad N / tangential /
torsional, relief, tool angular rate, pinch-point slide, pinch axis and d from contacts.
Falsifier: the tool jumps > 50 deg in one slip at every ramp rate and mu_t, i.e. no rate control.

Compute rules (CLAUDE.md): `~/.claude/bin/resguard.sh status` before launch, measure one cell's RSS
with `/usr/bin/time -v`, size jobs from it, wrap every job in `resguard.sh run --mem --cpu`, fsync
each row, OMP/OPENBLAS threads 1. Another session's decks 02-05 batch shares the box: leave it alone.
Commit ONLY this study's paths (`git commit -m ... -- <paths>`): the working tree carries the user's
unrelated edits (.gitignore, docs/hardware_control_station.md, scripts/real_v1_tracker_service.py).

## Deliverable

Builder `scripts/pinch_brake_page.py` + `scripts/pinch_brake_page.template.html` (copy the newest
builder/template pair), page `docs/experiments/20260926-pinch_brake/20260926-pinch_brake.html`,
row in `docs/experiments/INDEX.md`, artifact published and its URL written here and in INDEX.md,
films of representative cells (step control vs best servo cell per hand).
