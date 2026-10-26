# Tall slender indenter for contacts at large incidence angles on the UR5 cell

Specification for a design agent (owner, 2026-10-26: "maybe we need to build a taller indenter that allows the
robot+fingertip mount as configured to try out contacts, presses, rotations, shears, at larger/more extreme incident
angles. currently, since we are so low to the table/obstacles, this is hard"). Written on the workstation from the
calibration data of v1 ID 1 and ID 2; the runbook of the cell is `docs/experiments/20261022-ur5_tip_calibration/lab/README.md`.

## Outcome wanted

The fingertip's flanks are calibrated as densely as its face. On ID 2, 74 to 79 % of the flanks' FE surface points lie
more than 1.5 mm from any press, against 5 to 14 % of the face's
(`docs/experiments/20261022-ur5_tip_calibration/data/20261025-152920-cb2t/density_curve.json`). Location error follows
that distance: 0.36 mm at 1.5 N and over where the nearest press is 1 mm away, 2.0 mm where it is 2 to 2.5 mm away. The
owner sees the contact direction go wrong at large angles in manual tests. The new indenter lets the robot press, slide,
roll and twist the flanks and edges with the program and plans as they are.

Terms. The incidence angle of a contact is the angle between the tip's outward surface normal at the contact and the
pad's central face normal: about 0 deg on the face, 45 deg on the edges, 90 deg on the flanks (the regions of
`ur5_tip_calibration_r24_pass3.region`). The FE surface points are those of
`docs/experiments/20261020-fingertip_contact_estimation/data/20261020-b_v2_surface_basis.npz`. A margin is the
clearance distance reported by `scripts/ur5_tip_calibration_check.py --clearance`.

## Why the present indenter limits the angle

- The sphere (R 5 mm, centre 41 mm above the indenter's base) sits on a cone that widens to a column of radius 10 mm,
  twice the sphere's radius (`scripts/ur5_cb3_cell.py`, `indenter_sdf`). A finger tilted far from vertical meets the
  cone or the column before the flank meets the sphere.
- The base stands on a 2x4 whose top is at z_B 70.8 mm; the table is at 32.8 mm. With the sphere's centre 41 mm above
  the 2x4, a tool tilted towards horizontal brings the HEX-E and the wrist down onto the 2x4. The pad is about 211 mm
  from the flange (HEX-E face 128.36 mm, then 82.5 mm to the pad on mount v7).
- The homing post and its key stand 49 mm tall, 32 to 57 mm from the sphere's axis.
- The program keeps joint 5 within 30 to 100 deg; plans keep 34 to 98 deg, and joint 6 within -65 to +50 deg, where the
  HEX-E's cable stays connected (cb3t lost the sensor at +54.5 deg on 10-26).

## Requirements

1. Contact element: a sphere of radius 5.0 mm, of the same material and finish as the present printed sphere, so that
   friction in slides matches earlier sessions. Roundness within 0.05 mm.
2. Neck: below the sphere, a neck of radius at most 3 mm for at least 30 mm, so that the finger reaches contacts from
   the sphere's top to 30 deg below its equator without coming within 1 mm of the neck.
3. Height: the sphere's centre at least 150 mm above the 2x4's top. Choose the height from the reach map (test 1);
   a column that can be set to two or three heights is welcome.
4. Stiffness: at most 0.005 mm/N at the sphere's centre in any direction, which is 1/40 of the mount's 0.2 mm/N. An
   aluminium extrusion column (for example 30 x 30 mm) and an aluminium or brass neck meet this on paper.
5. No ferromagnetic material within 50 mm of the sphere's centre: aluminium, brass, titanium or polymer parts, with
   brass or nylon screws near the top. Steel screws at the base are allowed. A steel part near the tip moves the board's
   field and trips the viewer's foreign-field rejection.
6. Placement: the fixture stands where the sphere fixture stands, on the same 2x4 and clamping, and the two are swapped
   between sessions. It carries the same homing post and key as the present fixture, at a height the robot reaches, so
   that the program's homing (`--home-from`) works unchanged.
7. Rigid mounting: a contact of the arm at the program's abort force of 10 N, anywhere on the column, must not move the
   base.
8. The program and the plan builders need no change beyond the indenter's geometry: presses still run along the
   indenter's vertical axis.

## Acceptance tests

1. Reach map. For the present indenter and the new one, count the FE surface points by region (face, edges, flanks) and
   side (y below -1.5 mm, between, above 1.5 mm) that a press reaches. A point counts when some turn about the vertical
   in -20 to +40 deg (step 5) keeps joint 5 within 34 to 98 deg and joint 6 within -65 to +50 deg (the simulator's IK,
   `ur5_tip_calibration_r24_pass3.joints`, on cb3t's registration `lab/20261026-134522-cb3t/registration.json`) and
   the margins exceed mount 3.0, finger 0.8 and HEX-E 6.0 mm, with the indenter's own body as an obstacle. The new
   indenter must reach at least twice the present share of flank points and every face and edge point the present one
   reaches. Report the share for slides (turns of 2.3 deg) and twists (to 20 deg) as well.
2. Stiffness. On the robot, the series compliance fitted in a registration (pass 1) on the new indenter exceeds the
   present indenter's by at most 0.01 mm/N.
3. Field. With ID 3's board streaming, the tip brought within 1 mm of the neck and the column without contact changes
   no chip's reading by more than 1 uT.
4. Registration. Pass 1 on the new indenter fits within 0.05 mm rms (cb3t on the present sphere: 0.034 mm).

## Deliverables

- CAD in `docs/experiments/20261026-tall_indenter/`: STEP for each part (the owner edits in Inventor) and STL for printed
  parts, with a bill of materials.
- The new indenter in `scripts/ur5_cb3_cell.py` as an `indenter_sdf` kind, with the sphere's centre and radius for the
  plan's geometry block, used by `ur5_tip_calibration_check.py` and the builders.
- The reach map of test 1 as a dated HTML result page built from data, as in the repository's CLAUDE.md, with a figure
  of the reachable points on the tip for each indenter.
- A draft request for the lab's queue (`docs/experiments/20261022-ur5_tip_calibration/QUEUE.md`): the swap, the homing,
  pass 1, and a pass of presses and slides at the newly reached points.

## Choices left to the owner

- A variant with the sphere on a horizontal arm, pressed sideways with the tool upright. It needs a review of the
  program, which presses along the indenter's z axis, and is worth building only if the vertical post fails test 1.
- Whether the new fixture replaces the present sphere for later tips, or both stay in use.
