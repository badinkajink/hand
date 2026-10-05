# Contact comparison bed: task protocol (runbook)

Shared definitions for every bed script, so rows from different scripts and simulators compare.
Results go in this folder as one JSONL file per task, one fsynced line per case. The bed scores
task-level agreement between contact models, consistency of each model with itself across step
size and resolution, agreement with analytic limits, and cost. Physical fidelity and cost carry
equal weight. No model is ground truth: Drake and Newton hydroelastic are pressure-field references,
the analytic laws are idealizations, and the real fingertip (printed TPU) is not modelled by any of them.

## Rig

The two-pad pinch of the 10-01 study (`scripts/hom_contact_rig.py`, `make_rig(spec, d_cg, gravity)`):
two real_v1 fingertip spheres (R 10.55 mm, 20 g) on force-controlled rails along world x squeeze the
real_v1 screwdriver (cylinder r 12.5 mm, length 100 mm, 24.5 g), axis along world y, mu 1.
Pinch force N is per pad. New tasks extend the rig in new scripts (`scripts/contact_bed_<task>.py`)
by subclassing or wrapping `MjRig` / `DrakeRig`; do not edit `hom_contact_rig.py`.

## Models (row field `model`, rig spec in `rig_spec`)

| model | rig spec | notes |
|---|---|---|
| `mj_point3` | `mj:point3:ir100` | MuJoCo, one contact per pad, condim 3 |
| `mj_point4s` | `mj:point4s:fit0.000996044x0.2498:ir100` | condim 4, mu_t rescheduled each step |
| `mj_pads1` | `mj:spheres:s1:rs0.75:ir100:tr0.02` | 1 mm sphere pads (primary MuJoCo model) |
| `mj_pads2`, `mj_pads05` | 2 mm and 0.5 mm pads as in `20261001-hom_contact_patch/torsion.jsonl` (0.5 mm needs tr0.03) | resolution check |
| `drake_hydro` | `drake:hydro:E1e7:r1:rt0.01` | Drake hydroelastic, SAP |
| `newton_hydro` | Newton hydroelastic, native mapping, kh_pad = E/R = 9.5e8 N/m^3, tool 100x, 0.5 mm voxels, impratio 100, reduction on; two-value soft fallback solref written before `add_mjcf` (`scripts/newton_pinch_probe.py`) | variant `newton_hydro_unreduced` |

Physics steps: every model at 1 and 5 ms. Consistency set for `mj_pads1`: 0.5, 1, 2, 5 ms; and
`mj_pads2` / `mj_pads05` at 1 ms. Pinch forces per task below.

## Tasks

**T1 pull** (done, `contact_bed_pull.py`, `pull_slip.jsonl`): gravity off; settle 0.4 s at N; axial force
on the tool at 2 N/s until gross slip (10 mm/s) + 80 ms; onset by the creep-law method in that
script; hold at 50 % of onset 1 s for creep. N = 0.5, 1, 3.

**T2 twist** (`twist_slip.jsonl`): gravity off; settle 0.4 s; torque about the pinch axis (world x)
on the tool ramped at tau_law(N) per second, tau_law(N) = 2 mu N c N^(1/4), c = 0.996e-3 m N^-1/4,
until spin > 30 deg/s + 80 ms; onset by the same creep-law method as T1 on the angular speed.
Report tau_onset, rbar_onset = tau_onset / (2 mu N), pre-onset rotation (deg), creep (deg/s) during a
1 s hold at 50 % of tau_onset, and the kinetic torque at a driven 1 rad/s spin (`exp_torsion`).
Also the ratio rbar(3 N) / rbar(0.5 N): 1.565 for a Winkler / hydroelastic law, 1.817 for Hertz.
N = 0.5, 1, 3.

**T3 roll** (`roll.jsonl`): gravity off; settle at N; the +x pad translates along world z at
v = 10 mm/s for 1.0 s (also 50 mm/s for 0.2 s) while both pads stay force-controlled at N along x;
the -x pad has no z motion. No-slip kinematics: tool centre z = s/2 and rotation about y
theta = s / (2 r_tool) for pad travel s. Report rolling ratio rho = theta r_tool / (s/2), slip
distance at each contact (integral of tangential relative speed at the contact point, mm), tool
drift along y and rotation about x and z, normal force range, success (both pads in contact,
tool within 5 mm of the no-slip path). N = 0.5, 1, 3.

**T4 shake** (`shake.jsonl`): gravity along -y (tool axis vertical); settle at N; an inertial force
F(t) = m A w^2 sin(w t) along y on the tool (equivalent to shaking the pads with amplitude A),
f = 5 Hz, peak acceleration a_pk = 0.5, 1, 2, 4 g, 2 s. Rigid Coulomb slips when
m (g + a_pk) > 2 mu N. Report net drift per cycle (mm), peak relative displacement, drop
(drift > 10 mm), and the rigid-Coulomb prediction. N = 0.2, 0.5.

**T5 brake** (`brake.jsonl`): `exp_brake` of the rig: gravity on, CG offset d = 15 mm, pinch held at
6 N then lowered geometrically to 0.2 N over 4 s. Report swing end angle, time to 80 deg, peak
swing speed, and whether the tool stays pinched.

**T6 creep** (`creep.jsonl`): MuJoCo models only. Creep of T1 and T2 holds and T4 drift at a
sub-threshold acceleration with noslip_iterations 0 / 10 and impratio 100 / 1000. Report the
change in creep and every other T1/T2/T4 metric, and cost.

**T7 stability map** (`stability.jsonl`): `mj_pads1` hold (gravity on, N = 1) with d0 set directly
over 0.2-0.95 and tc = 5, 10, 20 ms; largest step that holds 1 s without ejection, against the
prediction dt_max = d0 tc for many spheres in contact (collective damping rate 2 d' / (d0 tc),
d' = n d0 / (n d0 + 1 - d0), explicit in velocity).

## Row fields (all tasks)

`task`, `model`, `rig_spec`, `N`, `dt_ms`, task parameters, metrics, `us_per_step_median` (wall
time of the physics step alone), `status` (`complete`, `ejected`, `failed`), `film` (relative path
or null), `script`, `git_rev`. Units in field names (`_mm`, `_deg`, `_mm_s`, `_Nm`).

## Films

For each task, film N = 1 N (T4: N = 0.5, a_pk = 2 g) at dt 1 ms for every model: close-up of the
pinch with pads drawn translucent and contacts coloured by force or pressure (`FilmRig` in
`hom_contact_rig.py`; Drake and Newton states are copied into a MuJoCo model for drawing),
labelled with model, simulated time and playback speed. Save `media/<task>_<model>.mp4`
(H.264, CRF 27, 25 fps, 480x360 per model) and a side-by-side tile `media/<task>_models.mp4`
with a JPEG poster.
