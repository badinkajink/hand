# Tall indenter for edge and flank contacts on the UR5 cell

The tall indenter is a one-piece PLA print: a 5 mm sphere 160 mm above the 2x4 on a neck of 3 mm radius that widens at
15 deg into a 60 mm column. Its homing seat stands on a strut 55 mm under the sphere. In the reach map on mount v7 and
cb3t's registration, a press reaches every face point (104 of 104, against 83 on the present indenter) and 107 of 124
edge points (against 89). Twists to 20 deg reach 104 face, 106 edge and 66 flank points, against 16, 53 and 15. Flank
presses rise from 60 to 88 of 492 points, 1.47 times, short of the doubling that the specification asks for. Of the 404
flank points still out of reach, 364 fail on joints 5 and 6 at every turn, whatever the indenter. Printed solid, the
column gives 0.004 to 0.005 mm/N laterally at the sphere's centre, against 0.0009 mm/N for the present indenter.

Specification: `20261026-tall_indenter_spec.md`. Builder: `scripts/tall_indenter.py`, from the numbers in
`ur5_cb3_cell.TALL`, which the cell's clearance checks and `ur5_tip_calibration_plan.py --indenter tall` read too.

![The part](media/20261026-tall_indenter_part.png)

*Figure 1. The tall indenter in its frame I: x_I along the tool axis toward the tip at homing, z_I up from the 2x4's top.
Red: the three raised flats under the mount's palmar face. Blue: the pad's contact point at homing.*

## Terms and metrics

- Regions of the tip, from the outward normal n of an FE surface point in the pad frame P
  (`ur5_tip_calibration_r24_pass3.region`): tip end if n_z > 0.7, else flank if |n_y| > 0.7, else face if n_x > 0.9,
  else edge if n_x > 0.5. The FE points are the 812 of `20261020-b_v2_surface_basis.npz`; 808 have a region.
- A press reaches a point when some turn about the vertical from -20 to +40 deg (steps of 5) gives the rest pose (1.5 mm
  off) and the deep pose (1 mm past contact) an IK solution near cb3t's homing branch with joint 5 within 34 to 98 deg,
  joint 6 within -65 to +50 deg (the HEX-E's cable), the cable port 15 mm from the arm, and these clearances to the
  indenter, the 2x4, the table, the rod fixture and the robot: mount 3.0, finger 0.8, HEX-E 6.0, robot links 8.0 and pad
  0.3 mm (the pad away from its contact, against the indenter without its sphere).
- A slide reaches a reached point when the deep pose turned 2.3 deg either way about four horizontal axes through the
  sphere's centre passes the same checks. A twist reaches it when five turns in a row (20 deg) pass.
- Lateral compliance: the sphere centre's displacement per newton of side load, from an Euler-Bernoulli cantilever of
  the column's profile on a rigid base, $\delta/F = \int_0^H (H-z)^2 / (E\,I(z))\,dz$, with E = 2.0 and 2.5 GPa.

## The part

| Feature | Value |
|---|---|
| Sphere | R 5.0 mm, centre 160 mm above the 2x4's top |
| Neck | R 3.0 mm where it meets the sphere, 4 mm under its centre, widening at 15 deg |
| Column | R 30 mm from the base to 55 mm up, then the 15 deg cone to the neck; a 6 mm fillet cone at the base |
| Base plate | 80 x 115 x 8 mm, four 4.6 mm holes counterbored 9 mm, 3 mm deep |
| Homing seat | mouth plane 55 mm under the sphere's centre, the key 44.36 mm toward the robot and 40 mm to +y_I |
| Volume | 391.7 cm³ (486 g of PLA if solid); the STEP reads back as one valid solid of the same volume |

Sideways at the sphere's top, 10 N bends the neck to 4.2 MPa where it meets the sphere, and less below, where it widens.
Printed PLA breaks across its layers at about 15 to 30 MPa, so the neck should snap at 35 to 70 N, 3.5 to 7 times the
program's abort force.

## Homing seat

The mount's socket (8 x 18 mm at its mouth, drafted 20 deg, in the column's palmar face) is unchanged. The present
indenters' key seats on its two long faces: their contact lines leave the roll about the tool axis free, and the
2026-10-22 seatings scattered ±0.58 deg in roll. The ends are 0.15 mm short, so the key also moves ±0.15 mm along
the tool axis.

On the tall indenter the mount's palmar face rests on three raised flats, 4 mm square and 0.6 mm proud of the seat's
floor. They sit at (y_M, z_M) = (-16.5, 36), (9.0, 36) and (-3.5, 66) mm on the palmar face, 25.5 mm apart across the
tool axis and 30 mm along it. The flats set the depth, the roll and the pitch. The key keeps the present key's shape
and stands 0.05 mm inside the socket on all four faces at the flats' plane. It sets the place along and across the tool
axis and the turn about the socket's axis, within ±0.05 mm. Walls 2.4 mm thick and 4 mm high run along both sides of
the mount's column at 0.4 mm and guide it onto the flats. The wall on the -y side stops at z_M 59.5, where the clamp
plate begins. Along the column's narrow part the walls end at z_M 45.0, 0.45 mm before it widens (until 2026-10-27 they
ended at 45.5 and overlapped the fork by 0.05 mm). The seat was built from the v7 fork's STL: its palmar face is flat at x_M 17.0 from z_M 28 to 73.

At homing on the tall indenter, cb3t's branch puts the joints at (61.0, -82.1, 117.0, -35.4, 63.4, 2.1) deg. The robot,
the HEX-E, the finger and the pad then clear the indenter and the cell by 38.1, 17.8, 6.6 and 14.6 mm.

## Reach map

| Region (points) | Present: press / slide / twist | Tall: press / slide / twist |
|---|---|---|
| Face (104) | 83 / 82 / 16 | 104 / 104 / 104 |
| Edge (124) | 89 / 85 / 53 | 107 / 104 / 106 |
| Flank (492) | 60 / 45 / 15 | 88 / 64 / 66 |
| Tip end (88) | 0 / 0 / 0 | 0 / 0 / 0 |

![Reach on the tip](media/20261026-tall_indenter_reach.png)

*Figure 2. The FE surface points by their angle about the pad's long axis and their place along it, marked by which
indenter reaches them.*

Every point that the present indenter reaches, the tall one reaches too. By side, the flanks below y_P = -1.5 mm rise
from 19 to 24 points and those above +1.5 mm from 41 to 64. The sphere at 100, 130, 160 and 200 mm gives the same
counts (on mount v3's parts), so the height is not the limit in that range. The present indenter loses its other face
and edge points to clearance: the mount, the finger or the pad comes within its margin of the indenter.

A flank needs a roll of 55 to 90 deg about the tool axis, and the IK puts that roll on joint 6: 82 to 87 deg for flank
points near 90 deg. With joint
6 allowed ±95 deg, flank presses reach 115 points, slides 110 and twists 115. The remaining 377 fail on joint 5 and on
the finger's cover and cap coming within 0.8 mm of the indenter.

## Stiffness

| Print | Lateral, E 2.0 GPa (mm/N) | Lateral, E 2.5 GPa (mm/N) |
|---|---|---|
| Solid (100 % infill) | 0.0050 | 0.0040 |
| 6 walls (2.7 mm), 25 % gyroid | 0.0075 | 0.0060 |
| Present indenter, solid | 0.0009 | 0.0007 |

The axial compliance is 0.0002 mm/N. Presses run along the column, so they load it axially. Of the lateral compliance,
76 % comes from the top 60 mm. Both prints meet acceptance test 2 on paper (at most 0.01 mm/N over the present
indenter). Only the solid print meets the specification's 0.005 mm/N.

## Bill of materials and print

- The indenter, `cad/20261026-tall_indenter.stl` (STEP beside it), PLA as the present sphere, the same printer, layer
  height and orientation: base on the bed, sphere up. No supports: the strut's faces lean at most 31 deg from vertical,
  and only the sphere's underside, where it meets the neck, overhangs more (53 deg).
- Four wood screws, 4 x 30 mm, pan head. Steel is fine: they stand more than 100 mm from the sphere, and the 50 mm rule of
  the specification keeps steel away from the sphere only.

## Fitting it to the 2x4

1. Take the sphere fixture off and put the tall one where it stood, centred across the 2x4, the strut toward the robot.
2. Drive the four screws snug in a cross pattern, then tighten them.
3. Home after fastening: in teach mode, seat the socket on the key with the palmar side down, press it down onto the
   three flats, end teach mode and record the seating (`record_pose.py`). Fastening after homing would move the base
   under a recorded home; the seating made after fastening has nothing to lose.
4. Build the plan with `--indenter tall` and pass the recording with `--home-from`. Built on the lab's placement
   (`plan/20261026-plan_tall_H160_check.json`), the plan has 153 actions. Five gravity stations fall outside joint
   5's window and pivots 6 to 8 come within 30 deg of the wrist's singularity, so the builder drops them. No travel
   path is short of clearance. The present sphere's plan dropped six actions on the same placement.

## Tests left for the robot

These are acceptance tests 2 to 4 of the specification, with their pass marks.

- Pass 1 on the tall indenter fits within 0.05 mm RMS (cb3t on the present sphere: 0.034 mm).
- Its fitted series compliance exceeds the present indenter's by at most 0.01 mm/N.
- With ID 3's board streaming, the tip within 1 mm of the neck and the column changes no chip by more than 1 uT.

## Draft request for the lab's queue

To go into `../20261022-ur5_tip_calibration/QUEUE.md` once the indenter is printed and on the 2x4:

> R28 (draft). The tall indenter in place of the sphere fixture (this folder's README, "Fitting it to the 2x4"). Record
> three seatings on its key and pass the first with `--home-from`. Run pass 1, then presses and slides at the 67 points
> that only the tall indenter reaches (Figure 2, red) and twists to 20 deg at the face and edge points. Pass 1 within
> 0.05 mm RMS and a series compliance within 0.01 mm/N of cb3t's admit the data; otherwise the seat or the column is
> reprinted.

## Future work

- Flank coverage: rerun `scripts/tall_indenter.py reach --kind tall --j6=-95,95` after the HEX-E's cable is rerouted for a
  wider joint 6; at ±95 deg the flank presses reach 115 of 492. A count under 100 with the reroute in place falsifies the
  cable as the limit.
- A seat check on the robot: ten seatings by hand, each recorded; their spread in roll under 0.15 deg and along the
  tool axis under 0.05 mm confirms the flats and the key's fit.

## Files

- `cad/20261026-tall_indenter.step`, `cad/20261026-tall_indenter.stl`: the part, frame I in mm.
- `20261026-tall_indenter_parts.json`: the resolved geometry, volume and the STEP's read-back.
- `20261026-tall_indenter_stiffness.json`: the compliance figures.
- `data/20261026-tall_indenter_reach_*.json`: the reach maps; `data/20261026-tall_indenter_homing_H160.json`: the homing pose.
- `plan/20261026-plan_tall_H160_check.json`: a plan built with `--indenter tall` on the lab's placement, to check the builder.
