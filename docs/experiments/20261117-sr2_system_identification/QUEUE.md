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
- W3 (18:5x). Finger gaps over the 22 best layouts of the 11-16 re-score (owner, 18:2x: "is there a similarly performing layout ... that dont have such close proximity"): `data/20261117-layout_screen_finger_gaps.json`. HOM's targets overlap the index and middle by 7-20 mm in every layout. Among ground-clear grasps the best tracking was 0.580 at (110, 80) g1, with the simulated fingers 5.5 mm apart; (105, 60) g9 tracked 0.563 at 18.6 mm, (100, 70) g9 0.537 at 17.0 mm, (100, 60) g15 0.516 at 22.0 mm. g13's layout (105, 70) gave 0.553 at 10.7 mm with its g6. Single seed, assumed 7.5 V plant; the hand's fingers came about 10 mm closer than the simulation's.
- W1b (18:4x, lab request 18:3x). The lat0 press at (70, 110): `docs/experiments/20261115-hom_bench_trials/plans/20261117-press_x70_y110_lat0_replan_plan.json`. The index MCP 0.5 deg inside its stop and the three PIPs 4 deg more open put the pads 3.5, 6.1 and 9.4 mm off the cylinder (plan 1.4, 8.1 and 7.2, the index at most 3.7 at its stop). On the stiff-contact scene HOM's close ends within 5.0 deg of the plan's grasp and the press tracks 0.12-0.13 (plan 0.09-0.14); on the soft-contact scene the cylinder drops at the lift from every start. The 18:30 chatter: the middle MCP and PIP targets swung 4.4 and 4.2 deg rms at 4 Hz with the cylinder's tracked x 1.3 mm, no clipped period, correlation -0.66 with its tracked z; with exact state the simulation's targets swing 0.13-0.22 deg. `data/20261117-press_open_pose_and_chatter.json`.
- W3 (19:0x, owner via the lab 18:5x: "does not have force closure at the very onset, produces some skew"). `data/20261117-press_grasp_closure.json`. In simulation the lat0 grasp has force closure at the onset: the thumb's and the index's normals 176-179 deg apart, the pinch line 4-7 deg off them (11-13 from the re-planned pose); epsilon 0.12-0.16 with three fingers at mu 1 and 0.02-0.04 at mu 0.4, 0.05-0.07 for the pinch alone at mu 1; friction can resist 34-87 mN m about the cylinder's axis against about 0.03 applied, and the cylinder turns less than 1 deg by the lift. On the hand (18:30, 18:44, 18:49) it turned 10-14 deg about its axis by 0.8 s and 21-27 deg by the lift, starting at 0.3-0.5 s, before any board read 0.3 N, with 4-5 tags in view at 18:30. The thumb's board read 0.3 N first, 0.14-0.40 s before the index's in two of the three. The skew comes from the close, not from the planned grasp.
- W1c (19:1x, owner via the lab 19:0x: HOM's fingertip feedback faithful to the paper before anything else). The paper (arXiv 2609.25619 v2, v3) runs the framework at 500 Hz, writes Eq. (1) in the measured GCFs {L} (in contact, the points closest to the hydroelastic centre of pressure) and puts the impedance at {C1}; it gives no numeric gains and no force threshold. Server flags, off by default (hom.py, hom_server.py; e0e14dcf1): `--filter-step-s 0.002` (Eqs. 4-6 at the paper's rate inside each 20 ms period), `--contact-off N` (contact hysteresis below the floor; 0.10 with --tips on), `--eq1-frames filtered` (a test, not the paper). On the calibrated stem scene every variant held and jittered alike (0.70-0.82 deg rms at 0.6-0.9 Hz; no 4-5 Hz chatter in simulation); the hysteresis cut GCF source switches 3-4x. On the server's own scene the g0 twist held 1/6 with the filter step against 4/6 without. `data/20261117-faithful_filter_check.json`. Suggested on the hand: an A/B of each flag in the overnight g0 repeats.
- W5 (owner, 18:5x via the lab: "the torque of the shaft/base reduces as the effective lever length decreases ... are our sphere pads contact model more predictive?"). The pick-up's lift and slip simulated with the pads and with point contact against the hand's pick-ups of 18:05 (thumb-middle, end tilt 45 deg) and 18:38 (thumb-index, 26.6 deg), slip turn at 2.5-2.85 deg of squeeze at 7.5 V.
- W6 (owner, 18:5x: "should we try to run annealed sampling or predictive sampling?"). Annealed and predictive sampling against HOM in simulation on the bench layouts (scripts/hom/inner_loop_mpc.py), after W2-W3.
- W2, S2 (19:4x). The free-air identification at 7.5 V (S2, 19:06-19:36 at the roll layout; supply 7.4-7.7 V, 20-27 C, no flags) fitted by `scripts/hardware/sr2_servo_sysid_fit.py --dir data/s2` (blocks identical to 11-06's; the supply from supply_diag.txt, no register read): `data/s2/20261117-sr2_servo_7v5_fit.json`. The controller's dead band is 3 encoder steps (0.88 deg) against 2 (0.59 deg) at 5.1 V, and its drive slope 14.55 per mille per step against 14.6; only 22 dwells lay outside the band, against 203. Yaw steps of 20-40 deg fell short by 0.20-0.85 deg on average, against 0.75-3.05 deg at 5.1 V. The torque scale is still the datasheet's, scaled to 7.53 V: stall 0.283 N m, 0.81 N m/rad; S3 measures it. The friction bands used gravity at 11-06's arm pose, while tonight the hand hung 11.6 deg from vertical at another layout; the yaw bands (6-18 mN m) are the size of the gravity torques (up to 16.7 mN m), so they wait for the fit at tonight's pose.
- W2, S1(b) (19:5x). The thumb's DIP tag at 9 poses (joints spanning 39, 64 and 70 deg of yaw, MCP and PIP), fitted by `sim/20261117-s1_zero_fit.py` (the tag model of lab/fingertip_tag_check.fk): `data/s1/20261117-thumb_zero_fit.json`. As built, the tag centre misses by 5.96 mm RMS (mean +5.1 mm in x, +2.4 in y in P). Joint zeros alone leave 3.72 mm (leave-one-out 3.87); the mount's position alone leaves 2.05 mm (2.17) with the mount 5.1 mm along x and 2.4 mm along y; both leave 1.59 mm (1.83) with the yaw zero at +2.8 +- 0.3 deg and the mount at (+7.2, -1.2) mm. The MCP and PIP zeros trade off (their signs flip between models), so these poses fix the thumb's mount and yaw zero only. The middle's tag was not detected at any of its 9 poses; the index waits for the sphere to be moved off the table's centre.
- W3, next. Predictions of S5 on the fitted plant. After S4: the friction coefficient, the tangential softness and the
  torsion, and the same predictions with the pads, point contact and point contact with torsion (condim 4).
- W4. The three-size landscapes of twist, roll and press with the pads and with point contact (about 6-8 GPU-hours),
  on the plant of W2-W3. Each rollout also records the least index-middle and thumb-finger gaps of the joint states and of HOM's targets (the three spheres per finger of hom_setup.finger_pts; no contact solve), and layouts whose fingers pass within a margin are flagged (lab request, 18:2x: "put it in W4, or flag layouts and grasps whose fingers pass within a margin of each other").
