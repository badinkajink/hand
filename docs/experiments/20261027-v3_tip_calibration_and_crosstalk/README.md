# Calibration and crosstalk plan for the four-magnet v3 tips on the UR5 cell

This plan measures the four-magnet tip v3 ID 1 on the robot. It runs in three parts. First comes the crosstalk map on the
fingertip holder, where two more v3 tips with their own boards stand at the two stations. Next comes v2 ID 1's suite on
the sphere fixture, the rod and the tall indenter, cut from 15.0 h of robot time to 9.1 h. Last comes a pass on the top
edge, which the trimmed cap opens. The map takes about 1 h with its touch-offs. Its plan is built (`lab/20261027-plan_holder_lab_map_v3id1.json`)
and waits on the owner's homing on the holder: the owner stopped R29 at 19:00 on 2026-10-27 to put v3 ID 1 on. Before
the suite, the lab program needs the speed changes listed under Code.

## Terms and metrics

- v3 is a pad 05R4 (`assets/cad_v2/fingertip_prints/`) with four N52 magnets of 3 x 2 mm in a 5.0 mm square centred
  over the board's centre chip U1, 7.5 mm under the apex. Alike poles lie on the square's diagonals, so the four have no
  net moment. v3 ID 1 is the tip on the robot, read by the ESP32-C3 (`--board`). The two held tips stream through the
  QT Py's mux (`--held-board`, `held.bin`); they get IDs when they go on the robot.
- A held tip is a whole fingertip (pad, cassette 3T, top cover TW15H3 trimmed, board) screwed to one of the holder's two
  stations (`../20261027-fingertip_holder/README.md`). Both stand on their blocks' top faces, facing up; the +y tip is
  turned 180 deg about the vertical, its connector toward +x_I (owner, 2026-10-27 19:50 and 19:59).
- W is the gap between the two apexes with the pads facing, on the +y station. g is the gap between the flanks side by
  side, with the apexes level, on the -y station; s shifts the robot tip toward its own end. A twist turns the robot tip
  about the vertical through its apex.
- A hover is the program's no-contact action (85ddc425). The robot pad's apex is held at a point in I with its normal
  down. It dwells there between two dwells at a reference straight above, and the HEX-E guards it at 0.5 N.
- The neighbour field at a pose is the change of a board's 15 chip components between the reference and the pose (uT,
  norm over the 15). It is read on the robot's board for the held tips' magnets, and on the held boards for the robot's.
- Location error is the distance along the surface between the estimated and the registered contact point (mm). Angle
  error is the angle between the estimated and the measured force (deg). Force error is the norm of their difference
  (N), or its magnitude as a share of the force. All are medians by region, under five folds grouped by action.
- The direction GP is a GP from the direction of the board change, d/|d|, to the contact point and to the force per unit
  change; its force is that times |d|. The board GP is the GP from d to the point and the force used so far.

## Crosstalk map on the holder

The held tips' field at the robot's chips, from the model of `scripts/fingertip_crosstalk.py`, is in the table below. The
other station's tip adds at most 0.31 uT. The magnets' pull between the tips is at most 1.0 mN at these poses
(`fingertip_crosstalk.py pull`), against the HEX-E's noise of 50 mN in x and y and 240 mN in z.

| Pose | Four N52 (v3), uT | Five N52 (v2), uT |
|---|---|---|
| Facing, W 0.5 mm | 33.7 | 483 |
| Facing, W 5 mm | 13.1 | 295 |
| Facing, W 20 mm | 1.2 | 86 |
| Side by side, g 13 mm | 38.9 | 231 |
| Side by side, g 25 mm | 3.4 | 65 |

The board's noise is 1.9 uT per component and record (median of the preflight of 143529-tallv2id1). Averaged over a 2 s
dwell at the chips' 140 conversions a second, a neighbour field carries about 0.6 uT as a norm over 15 components. Facing
to W 10 mm and side by side to g 20 mm read at five times that or more. The farther poses dwell 6 s, and the held tips'
boards read the robot's field in the same dwells.

The plan (`scripts/ur5_tip_calibration_holder_plan.py`) starts with three touch-offs of the robot apex on the +y held
apex. Their onsets give the held apex's height, which the CAD knows only to the boards' 0.6 mm of FR-4 (the owner
confirmed it) and the screws' 0.2 mm of play. The owner's three seatings on the holder (19:13 to 19:19) put it on the
tall indenter's 2x4, about 100 mm behind the low sphere and the rod; in the holder's frame the rod stands at x -104,
y -28.5 mm, in front of the -y station. With all four fixtures on the cell, 64 of the plan's 89 hovers fail, all on the
rod: wrist 3 comes within 0.7 to 2.6 mm of it at the touch-offs and facing W 0 to 3 mm, and side by side the HEX-E, the
adapter and wrists 2 and 3 run into it. The low sphere keeps wrist 3 7.4 to 8.3 mm away at the seatings.

The owner then moved the rod about 96 mm along -x_B instead of taking it off, both held tips turned out to face up, and
the holder was reseated (holder_seated_5). The plan for that cell, `lab/20261027-plan_holder_lab_map_seat5_v3id1.json`,
has the three touch-offs, 49 facing hovers over the +y tip and 37 over the -y tip at W 2 mm and over, and 20 repeats:
about 43 min, wrist 3 17.2 mm and the HEX-E 22.9 mm at least. With no tip under a block there is no side-by-side map.
Along every move the paths keep joint 5 39 deg or more from its singular angles, the elbow 81 deg or more from straight
and the wrist 518 mm or more from joint 1's axis (`scripts/ur5_tip_calibration_holder_paths.py`). The earlier
plans follow.

With the rod fixture off, `lab/20261027-plan_holder_lab_map_norod_v3id1.json` (on holder_seated_2) keeps 49 facing hovers
(W 0.5 to 30 mm at twists -20 to +10 deg, the twelve offsets, W 45 mm) and 32 side-by-side hovers (g 13 to 25 mm), with
the first 20 repeated: about 41 min. Twist +20 deg fails on joint 5 (30.5 deg). Its least clearances are wrist 2
20.95, mount 3.49, finger 3.48, pad 0.58 (at W 0.5 mm) and HEX-E 21.6 mm. With the rod in place, 25 facing hovers and 1
side-by-side hover keep every margin.

The three seatings tilt the holder's frame 4.3 to 4.9 deg from level and put its base 6.5 mm above the other 2x4s' tops.
If the holder leans that much, the plan stands. If it sits level, the fork rests 4.6 deg off in the seat, and the held
apexes, about 100 mm from the seat, would lie up to 7 to 8 mm from where the plan puts them; the touch-offs show that as a
wrong height or a miss. The hovers are rebuilt with `--z-offset` from the touch-offs and the homed registration.

A run with the stations empty would remove the background of the cell and the robot from the robot board's changes. The
held tips went on before the map, so the far poses (W 45 mm; g 40 mm failed the mount's margin) and the repeats measure
the background and the drift instead. The held boards do not move, so their background is constant.

Pinch presses on the +y held apex follow the map as their own plan: 0.25 to 3 N, robot targets 2 and 4 mm off its apex
across and along, twists of -20 to +20 deg, the apex press first, in the middle and last.

## The suite, cut to 9.1 h

The suite-value analysis (`scripts/ur5_tip_calibration_suite_value.py`, `../20261022-ur5_tip_calibration/data/20261027-suite_value_v2id1.json`)
timed v2 ID 1's actions and measured what each part adds to the held-out errors. v2 ID 1's actions took 15.0 h. Program
changes that leave the data as they are save 5.0 h:

| Change | Saves |
|---|---|
| Slider 1.0 for travel, the fast approach and the retreat, with the press and turn speeds kept | 2.2 h |
| Turn waypoints every 1.4 deg in place of 0.5 deg, turn_v 0.016 or more | 2.5 h |
| Slide holds of 0.2 s in place of 0.4 s | 0.2 h |
| Travel lift 30 mm | 0.1 h |

Cutting the 23 turn-speed variants and the rod's 24 rate presses, and thinning the sphere's rate set to 6 presses and the
tall twists to 4, saves 0.9 h more. The commanded turn speeds ran at 0.23 to 0.36 deg/s, against the 0.07 to 6.9 deg/s
the plans assumed, so the turn-rate variants had compared little. Every press target, force level, hard press, slide and
touch-off stays.

| Stage | As v2 ID 1 ran it | Cut |
|---|---|---|
| Sphere pass 1 | 0.65 h | 0.49 h |
| Sphere | 8.44 h | 4.89 h |
| Rod | 1.87 h | 1.10 h |
| Tall | 4.06 h | 2.66 h |
| Total | 15.0 h | 9.1 h |

The slides are worth their time. Without them the board GP places slide contacts at 0.96 mm and their shear's direction
at 15.4 deg; with them, at 0.43 mm and 6.6 deg. Halving them in any way (directions, amplitudes or count) raises some
slide median by 15 to 20 %. The 42 actions past joint 6 +45 deg take 0.6 h and run with the owner present.

The hard presses matter to the board GP: without them its location error at 3 N and over rises 48 %. The direction GP
needs them much less. Trained at 2 N and under, it places presses of 2 to 8 N within 0.25 to 0.31 mm and 5 to 6 % of the
force on actions it never saw. The board GP misses them by 1.0 to 3.8 mm and 45 to 87 %
(`scripts/ur5_tip_calibration_hard_extrapolation.py`). It also does better in range: on v2 ID 1's sphere session it gives
0.17, 0.23 and 0.23 mm at 1.5 N and over on the face, edges and flanks, against 0.23, 0.32 and 0.49 mm. On the tall
indenter's new points it does no better than the board GP. The hard presses stay in the cut suite as test data.

Whether the suite needs both spheres is open. On v2 ID 1 the tall indenter's forces had a shear part along the finger of
0.08 to 0.13 N per newton, against -0.02 to +0.02 on the sphere fixture, and its presses erred twice as much at the same
distance to the nearest training press (0.82 against 0.36 mm). The sphere check that would have separated the finger
from the indenter did not run before v2 ID 1 came off. On v3 ID 1, stage 4 presses six points on both indenters in one
session. If they agree, the tall indenter alone can carry v3's face, edges and flanks, and the sphere stage goes.

## The top edge and the flanks at the end

With the cap trimmed flush at x_F -0.10, the tall indenter reaches 14 of the 88 FE points of the tip's end (none
before), 9 of them at 1 mm spacing, and 19 to 24 more edge points (`scripts/ur5_tip_calibration_trimmed_cap_reach.py`).
It reaches one more flank point: 346 of the 402 flank points it misses fail on joints 5 and 6 at every turn. The four M2
cheese heads stand 1.3 mm proud of the trimmed face and cost 11 to 14 flank points. Low heads or countersunk screws
would recover them. This pass runs last, the next morning, with joint 6 to +58 deg while the owner is present; it needs
its own force levels for the tip region.

## Code

Done:
- the four-magnet rest fit (fe04ef46): `fingertip_wrench_model.fe_row("four_05R4")`, and `--case four_05R4` in
  `scripts/ur5_tip_calibration_analysis.py`;
- the hover action and the held boards' stream in the lab program (85ddc425);
- the cell with the held boards, the trimmed cap and the screw heads (5049f2c5); the map's plan (d86bdebd).

Still needed:
- the map rebuilt after the touch-offs, and the pinch plan;
- in the lab program, for the suite's speed: the travel at slider 1.0 with the press and turn speeds kept, step_deg
  as a plan field, the turn acceleration scaled with the slider, and the clearance checks rerun for the 30 mm lift;
- the cut suite's plan files on v3 ID 1's registration, with the six shared points;
- the direction GP as the analysis's default predictor, and the crosstalk map's fit (`fingertip_crosstalk.py`).

## Answers and open questions

The owner answered on 2026-10-27. The tip on the robot is v3 ID 1, read by the ESP32-C3. All pads have B00's shape. The
owner tells the lab when to stop and when to change fixtures. The five-magnet check of the model at 1 % is tabled.

Open:
1. Which tips have the trimmed cap: v3 ID 1 only, or the held tips too (the model trims all three).
2. The rod fixture off the cell for the map, or the 26 hovers that clear it.
3. The holder's 4.6 deg tilt: the holder or the seat (a level on the holder's base settles it).
4. Low-profile or countersunk M2 screws beside the pad, for the 11 to 14 flank points.
5. One indenter or two, after the six shared points.
