# Identification of the SR2 hand at 7.5 V: requests between the workstation and the lab computer

Owner, 2026-11-17 18:01: "i thought we queued up additional experiments to better understand real2sim transfer,
calibration of the various components of our system prior to running a larger scale GPU study of morphology
optimization, co-design, contact models? can you hash out / figure out ... what the high priority experiments are, and
coordinate with the ur5 session to fill out our understanding of the system as-is?"

Only the workstation writes this file. The lab session answers in its commit messages and in `NOTES.md` in this folder,
as in `docs/experiments/20261022-ur5_tip_calibration/QUEUE.md`. Each request's status changes here when the push that
answers it arrives.

## Why these measurements, in this order

On the blind twists of 2026-11-16 the simulation's mean tracking error against the hand was 0.248 with the landscapes'
plant. It fell to 0.152 with the identified servo model (no-backdrive gears, HOM's gain 0.55 N m/rad, torque budget
0.84 N m) and to 0.060 with soft tangential contact (impratio 1). With point contact in place of the pads it was 0.061.
So the servo model and the tangential contact each accounted for about half of the error, and the pad geometry for
none of it.

On 2026-11-17 the owner re-zeroed every servo (16:1x) and raised their supply from 5.0-5.3 V to 7.5 V. The joint zeros
and the servo parameters of the 11-06 identification therefore no longer describe the hand. The 11-06 gain was never
measured: it is the datasheet's stall torque scaled to the supply, because gravity loads stayed below the joints'
friction. The hand model also misplaced the index and middle fingertips by 3-7 mm (DIP tags, 11-16). Today the g13 plan
sent the index and middle MCPs to -15.0 and -13.7 deg, and they stopped against the mechanism at -11.4 and -10.0 deg.

Geometry comes first, because every later fit reads the joint angles and the tip positions. The servos follow, in free
air and then against a force sensor. Contact and the blind validation trials need the owner at the bench.

## Rules for unattended runs

- `lab/hom_setup.py diag` before and after each series: rail voltage, temperatures and status flags of the nine servos.
  Pause a series when a servo reaches 55 C (their limit is 70 C) and resume below 40 C.
- Free-air moves one joint at a time, with the clearance checks of 11-06
  (`scripts/fingertip/sr2_plan_tolerance_check.py`) at the layout used.
- Every press against a surface stops at 3 N on the HEX-E, after a tare with the finger clear.
- Commit and push each request's data and a one-line summary in the commit message, as for the bench trials.

## Requests to the lab

- S1 (open, 18:0x on 11-17). Geometry after the re-zeroing.
  (a) Each joint's reachable range inside the station's limits: at low speed, step the target past the plan's extreme
  in 0.5 deg steps and record the angle where the read-back stops following. Output, per joint: that angle against the
  model's limit and the station's limit.
  (b) The DIP tags 24-26 at 8-10 poses per finger, spread over each joint's range (`lab/fingertip_tag_check.py poses`,
  `move`, `measure`, both cameras). Output, per pose: the tag centre's error (mm) and its normal's error (deg) against the
  model at the measured joints. The workstation fits each joint's zero offset and each finger mount's position from
  these (W2).
- S2 (open, 18:0x on 11-17). Free-air servo identification at 7.5 V: the chain of 11-06
  (`docs/experiments/20261104-sr2_hand_crosstalk/lab/make_sysid_plan.py`, `sysid_chain.sh`, `sysid_tables.py`; about
  50 min), then the registers (`scs0009_registers.py`, read only). Output as on 11-06: `dwells.tsv` and `holds_c.tsv`.
  The workstation compares the dead band, the drive per step of error and the friction bands with 11-06 (W2).
- S3 (open, 18:0x on 11-17; unattended if the lab judges it safe, otherwise with the owner). Torque scale at 7.5 V. The
  arm sets one fingertip 1-2 mm from a rigid surface (the table, or a block fixed on it), along the direction in which
  the chosen joint moves the tip. The joint's target then steps 0.5 deg at a time past contact, 2 s per step, until 3 N
  on the HEX-E or an overload flag, and steps back the same way. Record the HEX-E, the joint's read-back and load, and
  the finger's board. One joint per finger: thumb MCP, index MCP and middle MCP, then the PIPs if time allows. Output:
  force against target offset. The workstation converts it to torque against offset (the gain in N m/rad), the dead
  band under load and the load at the overload trip, and compares each board's normal force with the HEX-E (W2).
- S4 (with the owner, next session). Contact on the hand, three trials each.
  (a) Pick-up of the press cylinder and of the taped sphere, then a slow opening of the pinch over 4-6 s above a catch
  box, the tags and the boards recording. The object's angle about the pinch line against the normal force gives the
  torque arm of the contact patch; the moment it slides gives the friction coefficient.
  (b) A plate covered in the sphere's grip tape on the table: the arm drags one fingertip 5 mm at 0.5, 1 and 2 N, the
  HEX-E recording. Output: the friction coefficient and the tangential stiffness before slip.
- S5 (after S1-S3 and W1, with predictions committed first). Blind validation: the g13 twist from the re-planned open
  pose, three trials with the tips off and three with them on, alternating; the press at lat0 twice; the pick-up hang
  twice. The g13 twists run unattended tonight, after S2, S1 and the board sweep: the owner told the lab at 18:0x to run
  HOM twist repeats overnight ("we can be a little aggressive with trials"; lab message of 18:1x). Start from the W1 plan;
  the predictions are committed (W1).

## Workstation

- W1 (done, 18:2x on 11-17). Re-plan g13's open pose within the measured MCP range, check in simulation that HOM's
  close from it reproduces the grasp, and ship the plan. The plan is
  `docs/experiments/20261115-hom_bench_trials/plans/20261117-film_x105_y70_twist_gr_g13_replan_plan.json`. Its open pose
  puts the index and middle MCPs 0.5 deg inside the stops (-10.9, -9.5 deg), turns their yaws 1 deg toward zero and opens
  their PIPs 4 deg; the thumb is as planned. The pads sit 5.2 and 7.4 mm off the sphere on the model, against 3.0 and 5.1
  for mcpstop. Of 17 start poses simulated (52 checked for clearance), HOM's close from this one ended nearest the plan's grasp among those with the index
  pad at least 5 mm off: within 3.5 deg on every joint at 5 and 7.5 V (mcpstop 2.3-4.4 deg). The open9 pose closed the
  index PIP 16 deg straighter in simulation, as on the hand. The HOM server keeps loading
  `20261116-film_x105_y70_twist_gr_plan.json`. Selection: `data/20261117-g13_open_pose_selection.json`. Blind predictions
  for both starts (pads soft, point soft, pads stiff; 5 and 7.5 V; contacts to HOM off and on; seeds 0-2):
  `data/20261117-g13_twist_predictions.json`. Script: `sim/20261117-g13_open_pose_replan.py`.
- W2. Fits from S1-S3: joint zero offsets and mount positions; the servo's gain, dead band, friction and overload load
  at 7.5 V; a plant string for the HOM scripts at 7.5 V. One table per fit, against 11-06.
- W3 (started 18:3x on 11-17). Today's four g13 twists replayed in closed loop with pads soft, point soft and pads stiff, each at its supply, budget and start: `data/20261117-g13_twists_hand_vs_contact_models.json`. Pads and point contact agreed within 0.02 in tracking and 1 deg in twist on all four. Soft contact matched the 5 V tips-off trial (hand 0.322 and 33.6 deg; pads 0.305 and 35.2, point 0.320 and 34.2); stiff contact gave 0.146 and 16.9. At 7.5 V the hand (0.439, 41.4 deg, 10.5 mm drift) fell between soft (0.369, 48.5 deg, 21.2 mm) and stiff (0.256, 26.5 deg, 8.6 mm). With the tips on, the hand drifted 19.8 and 20.4 mm; the simulation with exact contacts drifted as with none (12.7 and 21.2 mm).
- W3 (18:4x). Blind predictions for the overnight g0 repeats at (70, 110) from the mcpstop pose (the lab's choice: g13's index and middle collide at 7.5 V): `data/20261117-g0_x70y110_twist_predictions.json`. At 7.5 V with the tips off: pads soft 0.406, 43.2 deg, rise 9.3 mm; point soft 0.437, 48.2 deg, 9.9 mm; pads stiff 0.362, 38.2 deg, 1.3 mm. The hand's 18:21 trial, which ran before them: 0.453, 48.8 deg, rise 2.7 mm. Finger gaps (`data/20261117-g13_finger_gaps.json`): in both layouts HOM's targets overlap the index and middle by about 10 mm. The simulated fingers stay 13-17 mm apart (g13) and 27 mm (g0); on the hand at 7.5 V g13's came within 1.9-7.8 mm and g0's within 17.1 mm. The CPU stem scenes let fingers touch each other; the GPU scenes of the landscapes do not (owner, 18:2x).
- W3, next. Predictions of S5 on the fitted plant. After S4: the friction coefficient, the tangential softness and the
  torsion, and the same predictions with the pads, point contact and point contact with torsion (condim 4).
- W4. The three-size landscapes of twist, roll and press with the pads and with point contact (about 6-8 GPU-hours),
  on the plant of W2-W3. Each rollout also records the least index-middle and thumb-finger gaps of the joint states and of HOM's targets (the three spheres per finger of hom_setup.finger_pts; no contact solve), and layouts whose fingers pass within a margin are flagged (lab request, 18:2x: "put it in W4, or flag layouts and grasps whose fingers pass within a margin of each other").
