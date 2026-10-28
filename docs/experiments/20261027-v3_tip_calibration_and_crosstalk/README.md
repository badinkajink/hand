# Calibration and crosstalk plan for the four-magnet v3 tips on the UR5 cell

This plan takes one four-magnet tip, v3, through every action that v2 ID 1 ran on the sphere fixture, the rod and the tall
indenter. It then measures two more v3 tips on the fingertip holder. Three results come out of it. The first is v3's
location, force and direction errors against v2 ID 1's at the same targets and forces. The second is the field that a v3
neighbour adds at the robot tip's chips at known poses, against the dipole model. The third is the robot tip's contact
estimate in a pinch between two v3 tips. The robot time is about 18 h over two to three days. Nothing runs before the
owner's answers to the decisions at the end.

## Terms and metrics

- v3 is a fingertip pad 05R4 (`assets/cad_v2/fingertip_prints/`) with four N52 magnets of 3 x 2 mm. They sit in a 5.0 mm
  square centred over the board's centre chip U1, at the five-magnet pad's depth of 7.5 mm. Alike poles lie on the
  square's diagonals, so the four have no net moment. IDs are given as the tips go on: v3 ID 1 on the robot, v3 ID 2 and
  ID 3 on the holder.
- The robot tip is the tip on mount v7, whose board streams. A held tip is a pad unit (pad, cassette 3T, top cover TW15H3)
  without a board on one of the holder's two stations (`../20261027-fingertip_holder/README.md`).
- Location error is the distance along the surface between the estimated and the registered contact point (mm). Angle error
  is the angle between the estimated and the measured force (deg). Force error is the norm of their difference (N). Each is
  a median by region (face, edge, flank) at 1.5 N and over and under 1.5 N. The estimates come from a GP of the tip's own
  presses in five folds grouped by action, as in `scripts/ur5_tip_calibration_direction.py`.
- Coverage is the share of a region's FE surface points within 1.5 mm of a press.
- W, g and s are the holder's pose parameters. W is the gap between the two apexes with the pads facing. g is the gap
  between the flanks side by side, with the apexes level. s shifts the robot tip toward its own tip end. A twist turns the
  robot tip about the vertical through its apex.
- The neighbour field at a pose is the change in the robot tip's 15 chip components when it moves from a reference 40 mm
  above to the pose. The same change measured with the stations empty is subtracted from it. It is given in uT, as the norm
  over the 15 components.
- The model residual is the norm of the measured minus the modelled neighbour field at poses left out of the fit. It is
  given in uT and as a share of the measured field.

## What the plan measures

v2 ID 1's own errors are the reference. At 1.5 N and over, its GP places contacts within 0.23 mm on the face, 0.32 mm on
the edges and 0.49 mm on the flanks (`../20261022-ur5_tip_calibration/data/20261027-143529-tallv2id1/r29_tall_vs_sphere.json`,
`sphere_alone`). Its angle errors there are 2.5, 3.3 and 4.9 deg. The readout model resolved 0.029 N at the contact with
five magnets. For an earlier, wider four-magnet layout it resolved 0.051 N (`../20261020-ur5_calibration_mount_v2/`); it
has no figure for 05R4's square. Stages 1 to 4 run v2 ID 1's plans unchanged on v3's registration, so the comparison uses
the same targets, force ladders and density.

The model of the crosstalk budget (`scripts/fingertip_crosstalk.py`) gives the field of a neighbour at the robot tip's
chips, at twist 0 and s 0:

| Pose | Four N52 (v3), uT | Five N52 (v2), uT |
|---|---|---|
| Facing, W 0.5 mm | 33.7 | 483 |
| Facing, W 2 mm | 24.2 | 406 |
| Facing, W 5 mm | 13.1 | 295 |
| Facing, W 10 mm | 5.3 | 185 |
| Facing, W 20 mm | 1.2 | 86 |
| Side by side, g 13 mm | 38.9 | 231 |
| Side by side, g 20 mm | 8.2 | 102 |
| Side by side, g 25 mm | 3.4 | 65 |

A press of 0.4 to 0.7 N changes v2 ID 1's board by 23 uT. The other station's v3 tip adds at most 0.31 uT, so both
stations take a tip for the whole run. Side by side and touching, four magnets still give 2,928 uT in the model. The
holder reaches side by side from g 13 mm only, because v7's fork meets the station block below that.

The board's noise sets which poses are informative. In the preflight of `lab/20261027-143529-tallv2id1` it was 1.6 to
3.0 uT per component and record, with a median of 1.9 uT. The chips convert about 140 times a second, so a 2 s dwell
averages about 280 conversions per chip. If the noise averages down as white noise, that gives 0.11 uT per component.
The neighbour field takes the difference of four such means, which gives 0.23 uT per component and 0.9 uT as a norm over
15. Facing poses to W 10 mm and side poses to g 20 mm then give five times the noise or more. The farther poses dwell
6 s, and the repeated poses measure the noise that is left.

## Stages

In v2 ID 1's sessions a press took 37 s and a slide 185 s, in eight directions at two amplitudes. A twist took 144 s, a
roll 182 s and a touch-off 46 to 97 s.

| Stage | Fixture | Actions | Robot time | Owner present for |
|---|---|---|---|---|
| 0 | none | v3 ID 1's pad unit on the finger's board on mount v7 | none | the change of pads |
| 1 | sphere | pass 1: preflight, homing, 34 touch-offs, registration | 0.7 h | the pole check |
| 2 | sphere | v2 ID 1's sphere plans: 212 presses with the hard presses to 8 N, 100 slides, 20 twists, 6 rolls, the final references | 8.5 h | the actions past +45 deg |
| 3 | rod | v2 ID 1's rod passes 1 and 2 | 1.9 h | none |
| 4 | tall | pass 1; R29's targets (60 presses with the references, 50 slides, 12 twists); six points pressed on both indenters | 4.5 h | the actions past +45 deg |
| 5 | holder | homing; field map with the stations empty; v3 ID 2 and ID 3 mounted; field map; pinch presses | 2 h | the homing and the change of tips |

In stage 1, each chip's field at rest and at 5 N is compared with its range at the board's gain, as for v2 ID 1. The
poles are then fitted at rest. The checkerboard (+1, -1, -1, +1) in the slots' order is expected. Its exact reverse can be
undone in software. Any other pattern stops the run for the owner.

Stage 4 adds six points that are pressed on both the sphere fixture and the tall indenter in one session. They are the
face's centre (index 788), two edge points and three flank points that both reach. On v2 ID 1, the tall indenter's forces
had a shear part along the finger of 0.08 to 0.13 N per newton of normal force
(`../20261022-ur5_tip_calibration/data/20261027-143529-tallv2id1/r29_tall_vs_sphere.json`). On the sphere fixture it was
-0.02 to +0.02 N per newton. The shared points measure that difference on v3 in one session.

## Field map and pinch presses on the holder

The poses:

- Facing, on the +y station: nine gaps W (0.5, 1, 2, 3, 5, 10, 15, 20 and 30 mm) at five twists (-20 to +20 deg in steps
  of 10 deg). All 45 poses pass the holder's reach check. Twelve more, at W 1, 3 and 10 mm and twist 0, move the robot
  apex 3 mm across or along the held tip. Their reach is still to be checked.
- Side by side, on the -y station: the 31 poses that pass the holder's reach check, at g 13 to 25 mm and s 0, 2 and 5 mm.

Each pose is a no-contact action. The robot tip dwells 1.5 s at the reference 40 mm above the pose, with the pose's
orientation. It then dwells 2 s at the pose and 1.5 s at the reference again. At W 15 mm and over and g 20 mm and over it
dwells 6 s at the pose. The action aborts as a contact if the HEX-E reads over 0.1 N at the pose. Each held tip's place is
known from the CAD to about ±0.2 mm. So a touch-off of the robot apex on the held apex at 0.1 N sets W = 0 before the
facing poses.

The empty run comes first, with both stations bare. The owner then mounts v3 ID 2 on the +y station (facing up) and v3
ID 3 under the -y station (facing down), and the same poses run again. Each run repeats its first 20 poses at its end,
which gives the drift. A run takes about 20 min.

The pinch presses then put the robot apex on the held apex on the +y station at 0.25, 0.5, 1, 2 and 3 N. Eight actions
shift the robot apex by ±2 and ±4 mm across and along, and four turn it by -20, -10, +10 and +20 deg. The apex press
repeats first, in the middle and last. That is 16 actions and 80 holds. Under the press the held pad's magnets move with
its surface, so the neighbour field changes with force. The difference between the apex press and the field at W 0.5 mm
gives that change.

## Analysis on the workstation

- The neighbour field is fitted with each held tip's six pose parameters and its magnets' strength, in five folds by pose.
  The residual at held-out poses is reported in uT and as a share of the field. At W 5 mm and under, a residual over 10 %
  of the field (3.4 uT at W 0.5 mm) means the model cannot be subtracted to the board's noise.
- In the pinch, the robot tip's contact is estimated by the GP of its sphere presses in three ways. The board change is
  taken raw, with the modelled field at the commanded pose subtracted, and with the held tip's pose fitted from the board.
  The truth is the contact of the two FE surfaces at the registered poses. The errors are compared with the robot tip's
  sphere presses at the same levels.
- v3 is compared with v2 ID 1 by the errors by region and force band, the coverage and the board change per newton at the
  face's centre. The rod and the tall indenter's transfers are compared as in `scripts/ur5_tip_calibration_rod_tips.py`
  and `scripts/ur5_tip_calibration_tall_r29.py`.

## Code needed before the stages

1. Before stage 1's pole check: `fingertip_wrench_model.calibrate` searches the magnet orders of five magnets. It needs
   05R4's layout, the square of `fingertip_crosstalk.four_row`; the FE row `b_v2_four` is the earlier 6 x 7 mm layout.
   `scripts/ur5_tip_calibration_analysis.py` needs a `--case` flag.
2. Before stage 4: R29's targets on v3's tall registration (`scripts/ur5_tip_calibration_tall_pass.py --session <v3's
   tall pass 1>`), with the six shared points added to both indenters' plans.
3. Before stage 5:
   - a holder plan builder for the poses above, with the clearances of `ur5_cb3_cell` (`TIPHOLD`, `TIP_F`). The holder's
     reach check had the rod fixture in the cell; it is rerun with the tall indenter on its second 2x4 as well;
   - a `hover` action in `src/morphohand/driver/reskin5x/host/ur5_tip_calibration.py`: reference, pose and reference, with
     the 0.1 N contact guard;
   - a `tip` indenter kind for the pinch presses, with the approach along the held apex's normal and the held pad's
     surface for the clearance. Both new kinds run first in the lab's simulator;
   - `scripts/fingertip_crosstalk.py` subcommands for the map's fit and the pinch's estimates.

## Decisions for the owner

1. IDs and places: v3 ID 1 on the robot; ID 2 on the +y station, facing up; ID 3 under the -y station, facing down.
2. The pads: 05R4's outer shape is B00's, as on ID 1 to 3 and v2 ID 1, since the plans' targets come from its FE surface.
   The poles went in as each pause named them. The glue sits only on the slots' floors, as the note of 10-26 asked.
3. Order: stages 1 to 4 first (about 15.6 h), then the holder. The holder's field map needs only pass 1, so it could run
   first. The pinch's analysis waits for the calibration either way.
4. Slides: v2 ID 1's 100 slides are 5.1 h of stage 2, and R29's 50 are 2.6 h of stage 4. All of them keep the comparison
   whole; one amplitude halves the time.
5. A five-magnet tip on the +y station for about 30 min. Its 483 uT at W 0.5 mm tests the model at 1 % (4.8 uT). At v3's
   34 uT, 1 % is 0.34 uT, about the noise.
6. v2 ID 1 first finishes R29: 77 actions in about 3.2 h, among them 7 slides past +45 deg.
