# Calibration and crosstalk plan for the four-magnet v3 tips on the UR5 cell

This plan measures the four-magnet tip v3 ID 1 on the robot. It runs in three parts. First comes the crosstalk map on the
fingertip holder, where two more v3 tips with their own boards stand at the two stations. Next comes v2 ID 1's suite on
the sphere fixture, the rod and the tall indenter, cut from 15.0 h of robot time to 9.1 h. Last comes a pass on the top
edge, which the trimmed cap opens. The map ran on 2026-10-27 from 21:21 to 22:52. The suite ran from 22:52 until the HEX-E
came off its DAQ at 02:54 on 10-28, in sphere stage 2 after its grid and 40 of its 84 slides. On 10-28 the HEX-E streamed
again from 09:33. The check after its reseat and the sphere twists ran from 09:41 to 10:25 (18 of 18 ok), rod pass 1
from 10:27 to 11:46 (registration 0.042 mm rms) and tall pass 1 from 11:49 to 12:21 (0.045 mm rms). Rod stage 2 and
the tall plans ran by 17:41. At about 18:40 the owner put v3 ID 2 on the robot. Its suite of 488 actions on the sphere,
its top edge, the tall indenter and the rod ran from 18:44 to 08:13 on 10-29, all ok, with no HEX-E time-out
(`../20261022-ur5_tip_calibration/QUEUE.md`, R31 at 08:19 on 10-29). The cell's current state, the runs done and what
waits on the owner are in `docs/experiments/20261022-ur5_tip_calibration/lab/STATE.md`, the one place kept current.

## Runs left for v3 ID 1, and the side-by-side map

Three plans need v3 ID 1 back on the robot: the top edge, the sphere morning plan and the 43 slides left of sphere
stage 2. Their files and needs are listed in `lab/STATE.md`.

The rod fixture stays on the cell (owner, 10-29 11:37: "i am not going to move the rod. figure something else out."),
so the side-by-side map runs with both held tips facing up. The robot's pad faces down beside the -y held tip, its apex
17 mm under the held apex, so that the two magnet squares stand level; their centres lie 8.5 mm under each apex.
Turned over about its long axis, the four-magnet square maps onto itself, because alike poles lie on its diagonals. The
held tip's field at the robot's chips is then the one a held tip facing down would give. The plan
`lab/20261029-plan_holder_lab_side_up_v3id2.json` (`scripts/ur5_tip_calibration_holder_plan.py --side-up`, on v3 ID 2's
holder seed) keeps 14 poses and their repeats, about 12 min: g 14 to 25 mm at a twist of -10 deg, g 25 mm at -20 deg
and a far pose at g 40 mm. At twists 0 and +10 deg the robot's links come up to 7 mm into the rod's fixture. Its path
check flags nothing, with the mount 3.19 mm and the finger 2.59 mm from everything. Any other tip on the robot needs its
own seed and a recheck. The pad-down plan without the rod (`lab/20261028-plan_holder_lab_side_norod30_v3id1.json`) is
off.

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
the holder was reseated (holder_seated_5). The plan for that cell, `lab/superseded/20261027-plan_holder_lab_map_seat5_v3id1.json`,
has the three touch-offs, 49 facing hovers over the +y tip and 37 over the -y tip at W 2 mm and over, and 20 repeats:
about 43 min, wrist 3 17.2 mm and the HEX-E 22.9 mm at least. With no tip under a block there is no side-by-side map.
Along every move the paths keep joint 5 39 deg or more from its singular angles, the elbow 81 deg or more from straight
and the wrist 518 mm or more from joint 1's axis (`scripts/ur5_tip_calibration_holder_paths.py`). The earlier
plans follow.

With the rod fixture off, `lab/superseded/20261027-plan_holder_lab_map_norod_v3id1.json` (on holder_seated_2) keeps 49 facing hovers
(W 0.5 to 30 mm at twists -20 to +10 deg, the twelve offsets, W 45 mm) and 32 side-by-side hovers (g 13 to 25 mm), with
the first 20 repeated: about 41 min. Twist +20 deg fails on joint 5 (30.5 deg). Its least clearances are wrist 2
20.95, mount 3.49, finger 3.48, pad 0.58 (at W 0.5 mm) and HEX-E 21.6 mm. With the rod in place, 25 facing hovers and 1
side-by-side hover keep every margin.

The seatings tilt the holder's frame 4.3 to 4.9 deg from level and put its base 5 to 6.5 mm above the 2x4's top. The
fixtures registered by touch-offs stand within 0.7 to 1.8 deg of level, and the owner finds the plywood level. The
program computes the holder's frame from the homing pose, so homing again keeps the error. Three touch-offs found no
contact 1.2 mm past the expected +y apex (200536-holdv3id1). Five touch-offs reaching 6 mm past it (202426-holdapexv3id1)
found the apex 2.86 mm lower and 2.50 mm toward -y_I, in the program's frame (fit rms 0.024 mm). The touches 3 mm along x
came only 0.04 and 0.07 mm deeper than the one at the apex, so the two pads meet flat along x, and the fit keeps x as
planned (`--flat-x`). A level frame through the measured apex puts the holder's base at z_B 71.35 mm. The tall indenter's
base on the same 2x4 stands at 71.2 mm. That frame is `lab/20261027-holder-apex-plusy/registration.json`
(`scripts/ur5_tip_calibration_holder_apex_fit.py --out`). It puts the -y apex 7.13 mm below the plan's; a holder tilted as
seated would put it 4.27 mm higher.

The plan builder places every apex and hover by that frame (`--true-frame`) and writes them in the program's frame. Its
checks put the holder there. The pad keeps the program's orientation, 4.3 deg from the holder's vertical.
`lab/superseded/20261027-plan_holder_lab_map_apex_v3id1.json` starts with five touch-offs on the -y apex (`--touch-station -y`).
They start 9.5 mm above the level frame's prediction and reach 3 mm past it, so they meet the apex under either
hypothesis. Then come 49 facing hovers over the +y tip, with the owner's three +y watched hovers first. Next are 21 hovers
over the -y tip at W 10 mm and over, which clear it under either hypothesis, and 20 repeats: about 45 min. The path
checker stops each touch-off at its first contact, as the program does. With the holder level, none of the 95 actions is
flagged: joint 5 stays 9.0 deg inside its window, the wrist 519 mm from joint 1's axis, the mount 5.3, the finger 4.1 and
the HEX-E 24.5 mm from everything. With the holder tilted as seated, none is flagged either: the touch-offs meet the apex 5.1 to 5.5 mm into their approach in place of 9.4 to 9.9 mm, and the mount keeps 5.6 mm and the finger 4.2 mm. The close hovers over the -y tip follow after its fit, as a second plan.

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
screws beside the pad have button heads (owner, 20:2x), 3.5 mm across and domed, 1.3 mm proud of the trimmed face. They
cost 4 flank points with the finger's margin at 0.8 mm and 12 at 1.5 mm; flat 3.8 mm cheese heads, as the model had
them before, cost 11 and 13 to 14. This pass runs last, the next morning, with the owner present.

`scripts/ur5_tip_calibration_top_edge_pass.py` builds it on the tall indenter's registration. Its targets are the
points of the tip's end, the edges and the flanks that no press target of v3 ID 1's plans comes within 1.0 mm of,
thinned to 1.0 mm. The tip's end takes the flanks' levels, 0.25 to 2 N. On 143529's registration (v2 ID 1's tall pass 1,
standing in until v3 ID 1's), the pass has 27 actions and about 0.64 h of robot time
(`lab/superseded/20261028-plan_tall_lab_top_edge_v3id1_on143529.json`):

| Actions | Tip's end | Edges | Flanks |
|---|---|---|---|
| Presses | 9 | 4 | 0 |
| Slides at 1 N | 8 | 3 | 0 |

Three reference presses at the face's centre complete it. No flank point beyond the plans' own is reached, even with
joint 6 allowed to +58 deg. Joint 6 stays at 42.6 deg or less. Pressing the tip's end points the finger down at the
indenter, so the arm stands stretched. The program's lifted travel at the plans' 60 mm brought the elbow within 3.7 deg
of straight. The pass therefore sets travel.lift_mm to 20 mm, and its path check flags nothing: the elbow stays 26.3 deg
or more from straight, the finger clears by 1.52 mm and the mount by 4.32 mm. The plan is rebuilt on v3 ID 1's own tall
pass 1.

On 10-28 the owner stopped the tall indenter's pass at 18:05, near the joint limits, and the top edge moved to the low
sphere (`scripts/ur5_tip_calibration_sphere_top_edge.py`). On v3 ID 2 the sphere's pass ran 12 of 12 from 19:21 to
19:30. It pressed the tip's end nine times at 0.25 to 2 N and the edge once to 3 N, between two reference presses at
the face centre. The end gives 30.2 µT/N at 0.5 to 3.5 N, between the face's 22.9 and the edges' 41.4
(`data/20261029-gain_by_region_sphere.json`). A direction GP trained on the grid alone places the end's presses 2.98 mm
off at the median. With the pass's other presses in training it places them within 0.46 mm, and within 0.23 mm at 2 to
3.5 N (`data/20261029-top_edge_gp_v3id2.json`).

## v3 ID 2 against v3 ID 1

v3 ID 2 ran the night suite of 10-28 to 10-29. On each fixture its presses are compared with v3 ID 1's on the actions
both tips ran (`scripts/ur5_tip_calibration_tip_vs_tip.py`, `data/20261029-tip_vs_tip_*`). Each tip is scored on its own
holds with the direction GP under five folds grouped by action. Location is the median error across the contact normal,
force the median relative error of the force's magnitude, and gain the median |d_board| / f_n at 1 to 2 N.

| Fixture | Actions | Location (mm), ID 2 / ID 1 | Force, ID 2 / ID 1 | Gain (µT/N), ID 2 / ID 1 |
|---|---|---|---|---|
| Sphere | 116 | 0.29 / 0.38 | 5.1 / 5.2 % | 28.2 / 28.8 |
| Rod | 75 | 0.36 / 0.36 | 6.1 / 5.5 % | 23.6 / 21.6 |
| Tall indenter | 57 | 1.06 / 0.95 | 9.5 / 14.9 % | 22.2 / 26.0 |

On the tall indenter both boards put the contact 0.75 and 0.98 mm along -y of the registration at 3 N and over. Its
flanks give 30.7 µT/N on ID 2 and 45.5 on ID 1 at 0.5 to 3.5 N; the sphere's flanks give 57.9 and 60.2
(`data/20261029-gain_by_region_*`).

Against the five-magnet v2 ID 1, ID 2 repeats ID 1's result. At 1 to 2 N the five magnets give 1.87, 1.85 and 1.83
times ID 2's signal per newton on the sphere, the tall indenter and the rod. The noise-equivalent force is 62, 83 and 74
mN with four magnets, against 35, 47 and 42 mN with five (`data/20261029-four_vs_five_*_v3id2.json`).

## Motions for the calibration

The indenter plate v2d stays off the cell until the calibration's motions are final (owner, 10-29 11:37). What each motion adds on
both tips is on the page `docs/experiments/20261029-calibration_motions/20261029-calibration_motions_v3.html`
(`scripts/ur5_tip_calibration_motion_value.py`, `data/20261029-motion_value.json`). Every indenter's presses, the
tip's end and the slides must be in a tip's training data. Each indenter left out of training is placed 0.57 to 1.80 mm
off, against 0.25 to 1.05 mm with its own presses, and the rod's forces then read 14 % low. Presses over 3.5 N left out
are still placed within 0.19 to 0.27 mm. One tip's calibration places the other tip's presses 1.02 and 1.30 mm off,
against 0.51 and 0.38 mm from their own, so each tip runs its own suite.

At the six shared points the tall indenter and the low sphere differ for the same registered contact. The registered
points lie 0.06 and 0.15 mm apart and the tool's orientation 2 to 3 deg apart. Yet the board's changes point 8.7 and
12.9 deg apart, the tall indenter gives 0.83 and 0.82 times the gain, and its shear per newton is twice the sphere's.
The cause is open. Plate v2d's only 5 mm sphere stands on a 160 mm column, so the comparison is to be repeated there
before the low sphere goes.

## Inverted rig: a probe on the HEX-E

The owner asked at 13:0x on 10-29 for an inverted rig, in which a probe on the HEX-E presses tips held still on the
plate, and approved building it at 15:2x ("let's shore up the plan and get it ready to print tomorrow"). The design,
its reach and its plans are on the page `docs/experiments/20261029-inverted_calibration_rig/20261029-inverted_rig.html`
(`scripts/ur5_inverted_rig.py`); the print, hardware, assembly and first session are in that folder's `README.md`. The
probe base takes the fork's place on mount v7's adapter disc with four M3 screws and keeps v7's homing socket. Its tools
are 5 and 2 mm spheres, the half rod and two flat 14 mm faces, one turned 30 deg. Plate v2e keeps v2d's keys and rod
and a 5 mm sphere, and its back row has three seats that hold the tips pad up with their ends toward the robot.

The hold was chosen at 16:00. Four of six holds reach 97 of 201 pad points (1.5 mm apart) with the 5 mm sphere; the
cassette's walls, which every hold brings along, set the count. Pad up, a face press loads the seat's column along its
axis (0.2 um/N at the apex, beam estimate as printed), where a tip facing the robot with its end up bends it (13 um/N).
Over all 808 points either sphere reaches 452 (finger on the robot 286): all 104 face and 124 edge points, 184 of 492
flank points and 40 of 88 on the tip's end. A plan of about 275 actions replays clean on each of the three seats
(joint 5 at most 96.8 deg, joint 6 within +-16.3 deg). The print is two P1S plates: the probe base and five tools
(6 h 13 min, 178 g) and the plate (15 h 37 min, 524 g). The lab program needs a probe mode before the rig can run.

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

## Results feed for the doc-writing session

The owner's doc-writing session (2026-10-27, 20:50) writes the crosstalk pages from what the workstation's session
pushes overnight. Each update is one JSON line appended to `feed.jsonl` in this folder: `time` (from `date`), `kind`
(session, registration, plan, analysis, figure), `summary` (the outcome with its numbers), `files` (repository paths:
data under `data/`, figures under `media/`, scripts) and `commit`. Lines are only appended. Commits that add crosstalk
results start with "Crosstalk:". Every push goes to `main`; data and figures are LFS files.

## Answers and open questions

The owner answered on 2026-10-27. The tip on the robot is v3 ID 1, read by the ESP32-C3. All pads have B00's shape. The
owner tells the lab when to stop and when to change fixtures. The five-magnet check of the model at 1 % is tabled. All
three v3 tips have the trimmed cap, as the model has them. The rod fixture stays on the cell at its new place. The plywood
is level across the table, and the holder's tilt is its seating's (the apex search above). The screws beside the pad are
M2 button heads.

Whether the suite needs both indenters is not the owner's to answer: the six shared points decide it (the suite's last
paragraph). Open for the owner: whether the program's faster travel (slider 1.0, 5.0 h of the 15.0 h) runs without
anyone at the robot.
