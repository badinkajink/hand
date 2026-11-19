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

- S1 (18:0x on 11-17; (a) done 21:31, W2 23:3x; (b) the thumb only, W2 19:5x). Geometry after the re-zeroing.
  (a) Each joint's reachable range inside the station's limits: at low speed, step the target past the plan's extreme
  in 0.5 deg steps and record the angle where the read-back stops following. Output, per joint: that angle against the
  model's limit and the station's limit.
  (b) The DIP tags 24-26 at 8-10 poses per finger, spread over each joint's range (`lab/fingertip_tag_check.py poses`,
  `move`, `measure`, both cameras). Output, per pose: the tag centre's error (mm) and its normal's error (deg) against the
  model at the measured joints. The workstation fits each joint's zero offset and each finger mount's position from
  these (W2).
- S2 (18:0x on 11-17; done 19:36 without the registers, W2 19:4x). Free-air servo identification at 7.5 V: the chain of 11-06
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
  Prediction for the index and middle yaws, first in the owner's order (W2, 11-18 12:0x; forces corrected 15:3x): with
  the pad on the block, the reading advances about 4 deg past first contact by the time the target is 14-15 deg past
  the reading, as in the night's squeezes. Press in the direction the yaws moved in the squeezes, toward the sphere.
  With the finger straight (1.53 mm of pad per deg of yaw, an 88 mm lever) the force there is about 0.6 N if the gear
  passes 0.4 of the servo's torque (the clutch plant at kp 0.525 N m/rad), and 1.5 N if it passes all of it. The
  index board read 0.6-1.6 N in the squeezes. The 1.0 and 2.4 N given at 12:0x assumed a gain of about 0.8 N m/rad,
  the datasheet's stall scaled to 7.5 V (0.81). Per newton on the HEX-E the reading advances 6.6 deg with the give of
  0.76 N m/rad; a rigid chain advances only by the pad's compression, under 1 deg. The block pins the pad, so the yaw
  presses need no DIP tag. The finger stays straight: its tag then faces away from both cameras (lab, 15:2x), and its
  lever is close to the twist's (1.44 mm per deg). On the step back the force shows whether the load can backdrive the
  gear (W3, the owner's alternative). In the clutch plant the force stays flat and the reading stays put until the
  target passes the reading; a backdrivable gear lets both fall before that.
  Geometry at today's arm pose (lab, 15:2x; `s3_press.py block`, 28db7ea9c). At zero the yaws move the pads along y_P,
  1.53 mm per deg. The MCPs and PIPs move them along x_P, 1.17 and 0.63 mm per deg, toward the palm's centre and then
  upward as they curl. With the palm level no flexion presses a pad onto the block's top. The MCP and PIP presses need
  a vertical land on the palm's-centre side of the finger (+x_P for the index and middle, -x_P for the thumb). The arm
  can turn the hand about the vertical until a side land faces the finger, or tilt it; the owner chooses at the bench.
  The pad centres hang 29 mm (index), 31 mm (thumb) and 38 mm (middle) above the table in the model, about 4 mm lower
  with W2's offset. A yaw press needs a land at the pad's height: the arm lowers the hand, with the other fingers clear
  of the table, or a spacer raises the block.
  The block as built (owner, 15:3x-15:4x; lab 15:42, 6ff03f252) is a 20 mm 80/20 post about 128 mm tall, clamped to
  the table through a second piece along it. Two 45 x 45 x 1 mm plates are glued on opposite faces. Tag 0 is on one
  (edge 39.85 mm, centre 108.2 mm above the table); grip tape is on the other, 22 mm behind the tag's face. The plates
  face +-x_P, 2.3 deg off square. The post flexes, and `s3_press.py --track-block` follows its tag through a press.
  The flexions press the plates head-on: the index and middle into the taped plate, the thumb into the tag's. A yaw
  slides its pad along a plate, so it loads its chain only through the tape's friction, with an MCP holding the pad on
  (the lab's proposal; the owner decides). Prediction for an MCP press head-on, on the 7.5 V replay plant (kp 0.825 N
  m/rad, at most 0.27 N m; 67 mm from the MCP axis to the pad): if the gear passes 0.4 of the torque, the force rises
  0.086 N per deg of target offset, to 1.5 N at the tool's 18 deg limit. If it passes all of it, the force rises 0.215
  N per deg and reaches the 3 N stop at about 14 deg. A rigid chain's reading stays at first contact, apart from the
  pad's compression. With the MCP press first, its slope sets the hold for the yaws. The hold's friction must exceed
  the yaw's force at 18 deg of offset: 0.75 N if the gear passes 0.4 of the torque, 1.9 N if it passes all of it.
  Slip shows on the HEX-E as a tangential force that stops rising while the reading jumps. The yaw's reading also
  advances by the pad's shear, which S4(b) measures on the same taped plate.
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
- W1c, addendum (20:2x). The sub-steps cost the server 12.5-12.8 ms of each 20 ms period on the hand (p95 16-25, max 22-53; 2.7-2.9 ms without), so tonight's fstep arms ran about 10 ms later; the five one-period records after their trials hold corvus's last state arriving after the slower server had closed the trial. `--filter-exact` (FilterParams.exact) integrates Eq. (5)'s correction exactly over the period, (1 - exp(-k dt)) of the error per period: in sil it costs 2.56 ms (2.61 without flags, 12.52 with the sub-steps), and in simulation it reproduces the sub-steps (stem scene within 0.005 in w_rmse_ratio, same jitter; server scene held 1/6 with either). Use it in place of --filter-step-s. `data/20261117-faithful_filter_exact.json`.
- W3, A/B (20:2x). The night's g0 repeats at (70, 110), scored by the lab (`docs/experiments/20261115-hom_bench_trials/data/20261117-night_repeats_tracking.json`). Tips off without flags tracked 0.479, 0.445 and 0.417 (mean 0.447, twist at the end 44-52 deg); of the blind predictions point contact with soft tangential contact came nearest (0.437, 48.2 deg), pads soft 0.406 (43.2), pads stiff 0.362 (38.2). Tips off with the sub-steps: 0.473, 0.427, 0.434. Tips on without flags: 0.468 (drift guard at 6.5 s), 0.622, 0.202 (dropped); with the sub-steps and --contact-off 0.10: 0.421, 0.506, both full. Three trials an arm and 10 ms of added latency in the flagged arms: no difference is established.
- W3, why the twist tracks about 0.45 (21:5x, owner 21:4x). On the hand, in every tips-off g0 trial, the index yaw, the middle yaw and the thumb MCP sat 16 deg from their targets over the whole task and moved 2-4 deg: 16 deg is the torque budget's cap on one servo's offset (19.3 deg at saturation, times the budget's 0.8). The budget cut HOM's targets in 100 % of task periods, and the index PIP sat at its +1.78 deg clip in 46-67 % of them. In simulation the same joints stall 25-30 deg short, and lifting the budget or the clip does not help (point soft 0.437 -> 0.438 and 0.394-0.406; pads 0.406 -> 0.427 and 0.365-0.398): the yaw servos lack the torque to carry the fingertips round the sphere under this grasp's forces. The best trial (0.622, tips on) is the one where the middle yaw and thumb MCP moved (13.5 and 24.3 deg). Trajectories: the blind twist predictions follow the hand's mean twist trace (5 trials, spread 3.6 deg RMS) within 5.3 deg RMS (point soft) and 5.9 (pads soft); pads stiff 8.8. The soft settings put the sphere's rise at 9-10 mm against the hand's 2.7. With MuJoCo's noslip the rise is right (1.1-1.2 mm) and the 11-16 hang holds (pads stiff + noslip: 42 -> 36 deg in 10 s, the hand 41-43 for 60 s; point contact swings free), but the twist turns 35-37 deg (9.1-10.1 deg RMS). No friction setting of these fits rotation, rise and hang together. `data/20261117-twist_limit_and_trajectories.json`.
- W2, S1(a) (23:3x). The joint ranges at 7.5 V (lab, 20:53-21:03, and five ends again at 21:28-21:31 with the stall test; `data/s1/20261117-s1a_*`) against the model's limits: `data/s1/20261117-s1a_ranges_vs_limits.json`, by `sim/20261117-s1a_ranges_vs_limits.py`. The model's limits (`hand_object_scale_kin.limits`, the joint ranges of HOM's scenes and of the planners) equal the station's calibrated ranges at all 18 ends. Two ends stopped more than 1 deg inside them, both MCP lows. The index MCP stalled at -10.3 deg and the middle MCP at -8.5 deg, against -15.0 in the model. The stops move with the push (duty: the servo's present load, in % of full PWM duty). At 12-18 % tonight the index read -10.3 to -10.8 deg and the middle -8.5 to -8.8; at 21-22 % in the afternoon they read -11.4 and -10.0. The other four low ends reached the station's limits within 1.5 deg (the thumb MCP -10.8 of -12.3). The yaws were swept to +-45 deg, the MCP highs to 58 deg and the PIP highs to 80 deg, the owner's caps. Those ends followed within 0.3-2.5 deg, and the middle yaw's low end within 4.0 deg at 22.5 % duty, the most of any end. Its reading reached -41.0 deg; the lab's table 9 gives -23.7 deg, the last reading within 2 deg of its target. The 6.7-40 deg between the caps and the model's limits were not tested. A stall needs 2 deg of lag, so a stop within 2 deg of a sweep's end goes unseen. The thumb MCP's reading did not move over the last 1.3 deg of its sweep. At four ends the server clips HOM's targets 4.7 deg inside the model's limits (servo_limits' safe range). These are the index PIP's low end (+1.78 against -2.93 deg) and the index MCP's high end (60.04 against 64.75). The others are the middle MCP's high end (65.02 against 69.73) and the middle PIP's (72.34 against 77.05).
- W2, own-pose GP (23:3x). The boards' free-air sweep (lab, 20:10-20:53) moved all nine joints to 300 random poses, with 27 zero poses between them. It ran at the roll layout (85, 170); the trials ran at (70, 110). It was fitted as on 11-04/05 by `sim/20261117-own_pose_sweep_fit.py`, with 294 still poses per board: rows `data/sweep/20261117-own_pose_rows_s3.npz`, numbers `data/sweep/20261117-own_pose_sweep_fit.json`. The new GP is `docs/experiments/20261115-hom_bench_trials/data/20261117-tip_contacts/20261117-own_pose_gp_sweep_x85y170_7v5.npz`; the trials still load the 11-04/05 GP (`tip_contacts.OWN_POSE`). Each board changed by 62-67 uT (median) with its finger's pose, against 66-69 on 11-04/05. A zero pose predicted from its two neighbours missed by 3.6-3.7 uT. The new GP's five-fold error is 5.4, 4.7 and 4.7 uT (thumb, index, middle; 0.17-0.19 N), against 6.1, 5.8 and 5.4 uT for the 11-04/05 fit. The trials' GP misses the new rows by 11.0, 15.3 and 14.1 uT (0.39-0.54 N). One offset per joint, fitted to the new rows, brings it to 6.3, 7.2 and 8.4 uT. On 11-05 its error at this layout, left out of the fit, was 8.1, 8.4 and 6.6 uT. The offsets reach 5.5 deg (the index yaw) and 5.1 deg (the middle MCP); the owner re-zeroed the servos at 16:1x.
- W3, A/B 2 (23:4x). The second A/B of the g0 twist at (70, 110) ran 21:06-21:25 with `--filter-exact` in place of the 2 ms sub-steps: 9 of 12 cycles before the lights went off. The scores are the lab's (`docs/experiments/20261115-hom_bench_trials/data/20261117-night_repeats_tracking.json`). All nine trials ran to the schedule's end. Tips off as before tracked 0.404, 0.424 and 0.428 (twist at the end 43-45 deg), against 0.479, 0.445 and 0.417 in the first A/B. The blind predictions at 7.5 V were 0.437 with point contact (48.2 deg), 0.406 with soft pads (43.2) and 0.362 with stiff pads (38.2). Tips off with `--filter-exact` tracked 0.428 and 0.424 (47 and 46 deg); the first A/B's sub-steps gave 0.473, 0.427 and 0.434. Tips on as before tracked 0.397 and 0.335 (37 and 31 deg), both full runs. In the first A/B this arm gave 0.468 and 0.202, both stopped by the drift guard (at 6.5 and 5.4 s), and 0.622. With the contacts given to HOM in simulation, the predictions were 0.437 (point), 0.397 (soft pads) and 0.346 (stiff pads). Tips on with `--filter-exact --contact-off 0.10` tracked 0.310 and 0.311 (34 and 40 deg), both full. The first A/B's sub-steps with the same hysteresis gave 0.421 and 0.506, at about 10 ms more latency (W1c addendum). Within the second block the four tips-on trials tracked 0.310-0.397 and the five tips-off trials 0.404-0.428. Over both blocks the drift guard stopped none of the four tips-on trials with the paper's filter and the hysteresis, and two of five without. The server records (`data/20261117-ab2_server_timing.json`, by `sim/20261117-ab2_server_timing.py`) give the compute time per 20 ms period. The `--filter-exact` arms took 3.02-3.23 ms (median; 95th percentile 3.4-3.7 ms), the unflagged arms of the same block 2.99-3.20 ms. Both are about 0.3 ms over the first block's unflagged arms (2.69-2.93 ms); the sub-steps took 12.5-12.8 ms. The second block has no one-period fragment record; the first block's five each followed a sub-step trial.
- W3, replays (11-18 10:1x; owner 09:1x: every trial reported with the simulation's prediction and replay, per contact
  model). The night's 20 scored g0 trials were replayed open loop on the predictions' calibrated plant. Each replay
  started from the hand's start and applied the hand's recorded targets 12 ms after their state; the table went at the
  arm's lift. Contact models: soft pads, soft point contact, stiff pads, and stiff pads with noslip. Plants: 5 V, 7.5 V
  (the 5 V gains and torque limit times 1.5, as the predictions assumed), and 7.5 V with the index and middle yaws at
  the other joints' 0.825 N m/rad. A simulated closed-loop record, replayed on its own plant with no delay, came back
  identical (`sim/20261118-night_trials_replay_check.py`). A delay of 0 or 30 ms in place of the 12 ms moved the
  tips-off replays' twist by 0.3 deg at most (soft and stiff pads). Of the 240 replays 239 ran; one stopped on MuJoCo's
  constraint arena at 431 contacts. Data: `data/20261118-night_trials_replay.json` (per trial, model and plant, with
  traces), by `sim/20261118-night_trials_replay.py` and `sim/20261118-night_trials_replay_summary.py`. With the tips
  off (11 trials, 7.5 V plant):

  | | tracking | twist at the end (deg) | rise (mm) |
  |---|---|---|---|
  | hand, mean of 11 | 0.435 | 49.6 | 3.0 |
  | point soft: blind prediction, replay | 0.437, 0.351 | 48.2, 37.1 | 9.9, 9.9 |
  | pads soft: blind prediction, replay | 0.406, 0.347 | 43.2, 36.7 | 9.3, 10.0 |
  | pads stiff: blind prediction, replay | 0.362, 0.255 | 38.2, 30.7 | 1.3, 0.8 |
  | pads stiff + noslip: closed loop after the trials, replay | 0.354, 0.251 | 37.1, 30.0 | 1.2, 0.8 |

  Driven by the hand's own targets, the simulated sphere turned 12-20 deg less than the hand's in every contact model.
  The contact model set the rise: about 10 mm with soft contact and 0.8 mm with stiff, against the hand's 3.0 mm. In
  free air (0-0.6 s) the replayed joints were within 0.87 deg RMS of the hand's on the 5 V plant, 0.76 deg on the
  7.5 V plant and 0.69 deg with the stronger yaws. From 2.0 s the joints' RMS grew to 2.4-3.2 deg. In every model the
  simulated index yaw moved 3.0-4.8 deg less toward its target than the hand's, and the middle yaw 1.8-4.7 deg less.
  The stronger yaws closed 0.7 deg of the index's gap and 1.8 deg of the middle's. They added 1-3 deg of twist with
  soft contact and none with stiff, and raised the soft rise to 12 mm. The hand's sphere turned in jumps. In a trial
  its largest rise of twist within 0.2 s averaged 24 deg, and 3.8 windows of 0.2 s rose more than 8 deg. In the
  replays the largest rise averaged 7-10 deg, with 0.2-1.7 such windows. With the tips on (9 trials) the hand tracked
  0.397, turned the sphere 46.1 deg and raised it 4.3 mm. The blind predictions with the contacts given to HOM were
  0.437, 47.8 deg and 9.8 mm with point contact, 0.397, 42.5 and 9.3 with soft pads, and 0.346, 36.8 and 1.2 with
  stiff pads. The replays turned the sphere 20-22 deg on the 7.5 V plant and 19-48 deg with the stronger yaws. Two
  tips-on trials were not reproduced at all. The hand's best, 19:57 (0.622, 64.8 deg), turned 0-13 deg in every
  replay. The replays of 20:08, which the drift guard stopped on the hand at 5.4 s, spun the sphere 52-115 deg with
  soft contact.
- W2, S1(b) with the arm lowered (11-18 12:0x). The lab's rows of 11:12-11:41 (`data/s1/20261118-*_rows.jsonl`, all
  three fingers, both cameras) were fitted by `sim/20261118-s1b_tag_fit.py`: `data/s1/20261118-s1b_tag_fit.json`. The
  tag model is the one of 19:5x; the leave-one-out error leaves out both runs of a commanded pose. The middle (9 poses,
  twice; yaw -13 to 29 deg, MCP 52-57, PIP 74-75) missed by 6.59 mm RMS as built, its normal by 12.4 deg. A PIP zero of
  +12.3 deg with the mount 6.0 mm toward -x and 0.8 mm toward -y leaves 0.81 mm (0.87 left out) and 0.8 deg. An MCP
  zero in its place leaves 3.3 mm. The index (7 poses, twice; yaw -28 to 29, MCP 48-58, PIP 69-80) missed by 9.03 mm
  and 9.1 deg. Its mount alone, moved (+8.5, -2.5) mm, leaves 1.81 mm and the 9 deg. Zeros and mount together leave
  1.67 mm (1.93 left out), with an MCP zero of -10.0 deg and a PIP zero of +19.0; a PIP zero alone leaves 3.7 mm. The
  thumb (13 poses at its home and stepped mounts, 27 runs; yaw -30 to 29, MCP -7 to 23, PIP 0-30) missed by 8.67 mm,
  and every model of zeros and mount leaves 4.4-6.0 mm. Its tag sits 3.5-5.5 mm below the model at every pose. In last
  night's rows it sat 0.7 mm above, and the same fit left 1.59 mm. The mount fitted at each gantry place moved (6.98,
  1.52) and (6.79, 1.95) mm, so the 12 mm step came out as commanded. A height offset shared by the three fingers
  explains most of the rest (12:3x). Fitted together, with each finger's zeros and mount, the tags sit 3.86 mm (+-0.20)
  below the model. The index then fits to 0.55 mm (0.64 left out) and the middle to 0.49 mm (0.54). Their readings
  fall short of the tags' flexion by about 9 and 12 deg, split between the MCP and the PIP differently in each model;
  the mounts move 4.8 mm (index) toward +x and 4.2 mm (middle) toward -x. The thumb still misses by 2.90 mm. With its
  tag moved 4.7 mm along the finger, away from the PIP axis, it fits to 1.85 mm; last night's rows give -0.4 mm for the
  same shift. On the lab's camera-only heights the cameras agree within 1-2 mm across the two days, so the 3.9 mm more likely
  comes from the arm's pose, read with the arm off at its new low pose, or from P's frame there. To check on the hand:
  the thumb's DIP sticker and DIP link. As the lab notes, these poses cannot separate zero offsets from pots that read
  low near the top of their range.
- W2, the pads on the sphere at the night's grasps (11-18 12:0x). For the 20 scored g0 twists, the replay plant was
  set to the hand's measured joints and the tracked sphere, as built (`sim/20261118-grasp_pad_gap.py`,
  `data/20261118-grasp_pad_gap.json`). The gap is the least distance from a pad's surface to the sphere's (negative:
  inside), its median over each window. In the close (0.60-0.80 s) the index and middle yaws stop at about 20 deg, 2.4-3.2
  deg short of their targets. There the gaps are +0.8 mm for the thumb (median of 20; -0.4 to +1.8), -0.3 mm for the
  index (-1.1 to +0.3) and -0.6 mm for the middle (-2.5 to +0.2). The model meets the sphere where the hand's fingers
  stop. In the squeeze (1.30-1.58 s) the yaw targets go 14-15 deg past the readings. The index and middle readings
  advance 4.1 deg (2.3-4.7) and 3.5 deg (2.9-3.8), and the thumb pushes the sphere 1.3-3.1 mm toward them. The model's
  pads then sit 7.3 mm inside the sphere for the index (5.5-9.0) and 5.6 mm for the middle (4.5-7.7); the thumb's sit
  1.3 mm inside (0.2-2.6). On the hand the pads touch the sphere, compressed by about a millimetre. The boards read 2.6 N
  (thumb) and 0.8 N (index), and nothing on the middle in 18 of 20 trials. The index and middle yaw chains therefore give
  about 6 mm at the pad under the squeeze, about 4 deg of reading. The rigid model has no such give: in the replays'
  task windows its index and middle yaws moved 3.0-4.8 and 1.8-4.7 deg less than the hand's. At this grasp +5 deg of
  yaw changes the index's or middle's gap by 7.2 mm, and +5 deg of MCP or PIP by 0.2-1.4 mm. With the S1(b) zeros the
  close gaps become +1.3 to +1.7 mm, and the squeeze gaps -5.2 and -3.2 mm.
- W3, the yaw chains' give in the replays (11-18 12:4x). The night's 11 tips-off g0 twists were replayed as in "W3,
  replays", with a spring between the index's and middle's yaw servo shafts and their links
  (`sim/20261118-yaw_give_replay.py`; summary `sim/20261118-yaw_give_summary.py`: `data/20261118-yaw_give_replay.json`).
  The shaft is the yaw's reading. It stops where the gear's forward output (0.4 of the servo's torque) meets the
  spring, and the spring cannot drive it back. A stiffness of 0.76 N m/rad gives the night's 4 deg on the 7.5 V plant;
  1.5 N m/rad gives 2 deg. With 0.76 the yaw readings follow the hand's: the grasp's joint RMS falls from 2.2-2.3 to
  1.0-1.2 deg, and the task's yaw shortfall from 3.7-4.1 deg to -1.1 to +0.2 deg. With stiff pads the sphere turns 45.6
  deg by the end (rigid 30.8, hand 49.6), within 3 deg of the hand in 8 of 11 trials. It rises 1.6 mm (hand 3.0). Its
  twist rises more than 8 deg within 0.2 s 4.3 times per trial (rigid 1.6, hand 3.8), at most 13.9 deg (hand 23.9).
  The turn comes about 1 s later than the hand's, so the twist trace's RMS falls only from 16.9 to 15.8 deg and the
  tracking from 0.25 to 0.32 (hand 0.43). With soft pads the give changes little: twist 35.3-37.7 deg, rise 7.3-8.4 mm.
  The stiffness comes from the grasp, not from the twist; S3 measures it on the hand.
- W3, the owner's alternative: the load backdrives the yaw servos (11-18 13:4x). The owner asked whether the long lever
  from the contact to the yaw servo lets the load backdrive the servo. In the night's squeezes the index and middle
  readings moved 4.1 and 3.5 deg toward their targets. A backdriven servo's reading would retreat with its finger, since
  the potentiometer turns with the output shaft, and the model at the readings would stay on the sphere. Its pads sit
  5.6-7.3 mm inside the sphere instead, and the thumb pushed the sphere 1.3-3.1 mm toward those fingers: the shafts
  advanced and the pads were pushed back. The pads sit about 82 mm from the yaw axis (7.2 mm of gap per 5 deg). At that
  lever 0.8 N on the index pad is 0.066 N m about its yaw, and 4 deg of give is 5.8 mm at the pad. The 11 tips-off
  twists were replayed with the alternative (`sim/20261118-yaw_give_replay.py --backdrive`; rows
  `logs/20261118-night_replay/yaw_backdrive.jsonl`; table rows "pads backdrive" and "pads_stiff backdrive" in
  `data/20261118-yaw_give_replay.json`). There is no spring, and a load that opposes either yaw servo and exceeds its
  torque pushes it back as a plain position servo (kp 0.525 N m/rad). The readings stayed 3.7-4.3 deg short of the
  hand's in the task (rigid 3.7-4.1). The sphere turned 30.7 deg with stiff pads and 36.7 deg with soft pads (rigid 30.8
  and 36.7, hand 49.6). In the squeeze the targets sit 14-15 deg past the readings, so each servo pushes with 0.13 N m,
  and a load must exceed that to push it back.
- W3, S3's index presses against the replay plant (11-18 16:1x; lab 5b6131538, `data/s3/`, NOTES "2026-11-18,
  afternoon"). The give of 0.76 N m/rad is refuted. With the MCP holding the pad on the taped plate, the index yaw's
  reading advanced 0.9 deg while the tangential force rose to 0.69 N; the give predicted 4.6 deg. That is 1.3 deg per N
  with the pad's shear included, a chain of about 4 N m/rad or stiffer. The servos push harder than the replay plant.
  The index MCP's force rose 0.4 N per deg of target offset, 1.5 N m/rad at its 67 mm lever. That is 4.7 times the
  clutch plant's 0.086 N per deg and 1.9 times its 0.215 with all of the torque. The yaw gave 0.69 N at 3.6 deg of
  offset toward the sphere (0.96 N m/rad, 1.8 times kp 0.525) and 0.50 N m/rad the other way. The gears hold: on both
  step backs the force stayed until the target passed the reading. The MCP's force also stayed at 2.7 N after its
  overload flag cut the drive. Its force stopped at 1.75 N while the offset grew from 4.8 to 7.2 deg, then jumped to
  2.55 N with a 1.5 deg jump of the reading, and stayed near 2.5 N (0.17 N m). Its reading advanced 5.5 deg past first
  contact, and the PIP was pushed back 1.2 deg. The pad slid on the tape at about 0.5 of the normal force; the scenes
  use 1.0 for the tape. If the yaw's slope holds to the twist's 14-15 deg of offset, the index pushed the sphere with
  about 2.8 N, where its board read 0.8 N. At 1.3 deg per N that gives 3.6 deg of advance, near the night's 4.1 deg.
  The head-on yaw press with the wrist turn (owner's OK, 16:0x) tests this. Prediction for the index yaw, straight:
  0.19 N per deg of target offset, about 2.8 N at 14-15 deg unless the servo stops near the MCP's 0.17 N m (1.9 N),
  and 1.3 deg of advance per N. On the step back the force stays until the target passes the reading.
- W2, the index board against the HEX-E in S3 (11-18 16:4x; `sim/20261118-s3_board_vs_hexe.py`,
  `data/s3/20261118-board_vs_hexe.json`). Each step's HEX-E force in P (x and y, the mean over the dwell's last second)
  was set beside the index board's estimate over the same second, from the splitter's log, with the estimator's gates
  off. In the MCP press, head on, the trials' estimator (the board GP with the own-pose term) read 0.95 N where the
  HEX-E read 2.51 N (median of 45 steps over 1.5 N), and 0.31 N with no contact. Its line is 0.31 N + 0.26 times the
  HEX-E's force (RMS 0.05 N). Without the own-pose term the board reads 1.22 N at 2.51 N (0.28 N + 0.38 times). The
  physical model alone reads 1.38 N, with the force pointing the wrong way (its x_P slope 0.07). In the yaw press the
  trials' estimator read 0.32 N + 0.38 times the normal force and 0.60 times the tangential force (RMS 0.06-0.08 N).
  On the taped plate the board thus reads 0.26-0.38 of the normal force. By these lines the night's 0.8 N on the
  index was 1.3-1.9 N, against the 2.8 N that the yaw's slope gives at 14-15 deg of offset. The lines come from one
  pad and two contact points on a flat plate; the sphere's contacts and the other boards need their own.
- W3, S3's head-on yaw presses against the prediction (11-18 17:3x; lab 8c439dbe6, `data/s3/*_yaw_head_on.jsonl`,
  NOTES "16:46-17:18"; `sim/20261118-s3_yaw_head_on_fits.py`: `data/s3/20261118-yaw_head_on_fits.json`). The error
  below is the servo's target minus its reading; the forces are the HEX-E's in x_P and y_P, at 88 mm from the yaw
  axis. The slope agreed with the prediction, and the force did not. Past contact the force rose 0.12-0.34 N per deg
  of error (the lab's median 0.22; predicted 0.19). The yaws stopped at 1.07-2.27 N (0.09-0.20 N m), where 2.8 N at
  14-15 deg was predicted. The index and middle gave 0.82 N at 5 deg of error and 1.5 N at 10 deg (medians of
  eleven presses), largest 1.64 N (0.144 N m). The thumb gave 1.16 N at 5 deg, largest 2.18 N (0.19 N m). The night's
  targets move the index's yaw toward -1 and the middle's toward +1. In those directions the index stopped at
  1.07-1.53 N and the middle at 1.15-1.47 N: 0.094-0.135 N m, against the 7.5 V plant's forward limit of 0.108 N m.
  The force rose only when the reading moved. Between moves the duty climbed about 50 per deg of error, and the force
  stayed flat. The reading then moved 0.3-2 deg at once, 5-11 times per press: the gear train sticks. Per N the reading
  advanced 2.0-3.6 deg on the index and middle and 0.7-1.0 deg on the thumb. About the yaw axis that is 1.4-2.5 N m/rad
  (median 2.04) and 5-7 N m/rad. The predicted 1.3 deg per N came from the tangential press, with the pad's face on the
  tape. On the -1 side the DIP link's flat side met the plate, so this compliance lies in the finger, not in the pad.
  The step backs held the force until the target came within -0.15 to +1.2 deg of the reading, as predicted. In the
  night's squeeze the yaw targets ran 14-15 deg past the readings, so each yaw pushed at its limit. Over 2.04 N m/rad,
  0.13-0.144 N m bends the chain 3.7-4.0 deg, and the night's readings ran 3.5-4.1 deg past the pads' contact. The
  morning's give had the right bend: its servo was 2.7 times too weak and its spring 2.7 times too soft.
- W3, the night's twists on a plant set to S3 (11-18 17:3x; `sim/20261118-s3_plant_replay.py`, rows
  `logs/20261118-night_replay/s3_plant.jsonl`, table `data/20261118-s3_plant_replay.json` by
  `sim/20261118-yaw_give_summary.py`). The clutch plant keeps its hold rule and loses its forward friction (c = 0).
  Each servo pushes kp_j (target - q) up to tau_j. The index's and middle's yaws take 0.83 N m/rad up to 0.144 N m,
  and the thumb's yaw 1.17 N m/rad up to 0.19 N m. Every MCP and PIP takes 1.6 N m/rad up to 0.17 N m (the index's MCP
  press). Two dampings set the free air: "s3" a top speed of 9.8 rad/s, as the 7.5 V plant's, and "s3tc" the 7v5y
  plant's time constant of 33 ms. Each ran the 11 tips-off twists with soft pads, stiff pads and point contact, at
  friction 1.0 and 0.5 (132 replays). Stronger servos do not close the twist deficit. At friction 1.0 and with "s3tc",
  the sphere turned 35.9 deg on soft pads (rigid 36.7), 31.6 deg on stiff pads (30.8) and 35.7 deg with point contact
  (37.1). The hand turned it 49.6 deg. Friction 0.5 adds 3-4 deg: 40.6, 33.6 and 40.2 deg. On the soft contacts the
  task's yaw shortfall falls from 3.7 deg to 1.5-2.2 deg at friction 1.0, and to -1.0 to +1.6 deg at 0.5. The sphere
  then rises 17-23 mm (rigid 10, hand 3.0). On the stiff pads the rise stays at 0.7-1.7 mm and the yaws stay 2.2-3.5
  deg short. "s3tc" keeps the free-air joint RMS at 0.76 deg; "s3" raises it to 0.91 deg. The grasp's joint RMS stays
  at 2.3 deg in every variant. Stiff pads leave no room for the finger's 4 deg of bend, and soft contacts make room by
  lifting the sphere.
- W3, prediction for the S3 plant with the measured give (11-18 17:3x, written before the run). The plant is "s3tc"
  with a spring of 2.0 N m/rad between each of the index's and middle's yaw shafts and links (`--kc 2.0`). On stiff
  pads the yaw readings run 3.6-4 deg past the pads' contact, as on the hand. The grasp's joint RMS falls from 2.3 to
  about 1.1 deg. The sphere turns 40-46 deg (the 0.76 give on the 7.5 V plant: 45.6) and rises under 2 mm. On soft
  pads the readings overrun the hand's by 2-4 deg, and the rise stays at 17-23 mm.
- W3, the S3 plant with the measured give, result (11-18 17:4x; 66 replays in `logs/20261118-night_replay/s3_plant.jsonl`,
  table rows "<contact> s3tc mu<friction> kc2" in `data/20261118-s3_plant_replay.json`). The readings came out as
  predicted; the twist did not. On stiff pads at friction 1.0 the yaws ran 0.8 deg past the hand's readings (3.2 deg
  short without the give). The grasp's joint RMS fell from 2.3 to 1.8 deg, where 1.1 was predicted. The sphere turned
  31.7 deg (predicted 40-46; without the give 31.6; hand 49.6) and rose 0.8 mm. At friction 0.5 it turned 33.3 deg. On
  soft pads the yaws ran 1.7-2.2 deg past the hand's, the sphere turned 35.5 deg and it rose 16 mm. In every S3 variant
  the thumb's MCP stopped 4.3-4.8 deg short of the hand's in the grasp; on stiff pads it stayed 5.9-6.2 deg short in the
  task. That chain bends as well: in S3 the index MCP's reading ran 5.5 deg past first contact at 2.55 N. The servos'
  strength and the yaw chains' bend leave the twist deficit where it was; friction 0.5 recovers 2-4 deg of its 18 deg.
  On the hand the twist rose more than 8 deg within 0.2 s 3.8 times per trial, at most 23.9 deg (means over the 11
  trials). In the S3 replays it did so 0.1-1.6 times, at most 6.8-11.4 deg. Two mechanisms of the hand can store a turn
  and release it at once. S3 showed one: the gears stick, and the force rose only when the reading jumped. The other is
  a static friction at the pads above their sliding friction. MuJoCo's friction has one coefficient for both, and the
  replay plant's gears do not stick. S4(b), the pad dragged across the taped plate, measures the pad's static and
  sliding friction.
- W2, the boards in S3's head-on yaw presses (11-18 17:4x; `sim/20261118-s3_board_vs_hexe.py` now runs every S3 press;
  `data/s3/20261118-board_vs_hexe.json`). In all fourteen presses of the index's, middle's and thumb's tip sides, no
  board followed the force. The GP estimators' lines rise -0.06 to +0.03 N per N of the HEX-E's |F_xy|. At the largest
  HEX-E force (1.06-2.27 N) the boards read 0.07-0.38 N, about what they read before contact. The physical model rises
  at most 0.26 N per N, with an RMS of 0.06-0.22 N. The DIP link is 3 mm wider than the pad on each side, so these
  presses loaded the link and not the pad: a board does not see a load on its tip's side. In the night's twists the
  middle's board read nothing in 18 of 20 trials. A contact on the middle tip's side would read the same. A press of
  the middle's pad face would show whether its board reads a face load. The index's two face presses give the lines
  of the 16:4x entry again.
- W3, films of the night's twists beside the replays (11-18 18:4x; `sim/20261118-night_trial_film.py`;
  `media/20261118-night_twist_194243_0344_hand_vs_replay.mp4` and `..._1918_...`). Both tracking cameras filmed every
  trial of the night at 960 x 540 and 30 fps (`hom_trial.py --video-dir`, on corvus in
  `logs/20261116-hom_trials/20261117-night_repeats*/`). The 22 films of the tips-off trials are copied to
  `logs/20261117-night_trial_films/`. The script draws the replay from each camera's calibrated pose beside the
  hand's frames, with each object's x and y axes and the two twist traces. The server's record and corvus's share the
  trial's start to 0.3 ms. In trial 19:42:43 the hand's sphere turned about 30 deg between 5.0 and 5.3 s. The tag
  under the sphere blurs in one frame, and the fingertips' housings move little: the sphere slipped in the grasp. The
  replay (S3 plant, 2.0 N m/rad bend, stiff pads, friction 1.0) starts to turn at 5.1 s and adds 27 deg by 6.5 s
  without a slip. It ends at 34.6 deg, against the hand's 50.9.
- W3, next (11-18). Fit the index and middle yaw gain, the give of their yaw chains (a spring between the servo's shaft
  and the link, W2 12:0x), the sliding friction and the contact's impratio to the replays'
  joints and object poses, the second block held out (CMA-ES as `scripts/hardware/sr2_hom_replay.py fit`, on the clutch
  plant; about 30 min on 6 cores). S3 then checks the fitted gain and give, and S4(b) the friction. Then predictions of S5 on the
  fitted plant. After S4: the friction coefficient, the tangential softness and the torsion, and the same predictions
  with the pads, point contact and point contact with torsion (condim 4).
- W4. The three-size landscapes of twist, roll and press with the pads and with point contact (about 6-8 GPU-hours),
  on the plant of W2-W3. Each rollout also records the least index-middle and thumb-finger gaps of the joint states and of HOM's targets (the three spheres per finger of hom_setup.finger_pts; no contact solve), and layouts whose fingers pass within a margin are flagged (lab request, 18:2x: "put it in W4, or flag layouts and grasps whose fingers pass within a margin of each other").

## Owner's notes of 2026-11-18 09:0x

- The middle finger's DIP tag is probably visible only with the middle finger curled and the thumb moved well out of
  the way along its local x axis (owner). Use that pose for S1(b)'s middle.
- S3 needs a rigid block fixed to the table, its top and one side face flat; the HEX-E reads the force. Order: index
  and middle yaws first (pushing the side face; they stall in the twist), then one MCP and one PIP (pushing the top).
- S4(b) uses a plate covered in the sphere's grip tape on the same block. Drag 5 mm at 1 mm/s each way at 0.5, 1 and
  2 N, twice each.
- S3's block (owner, 09:55): a length of 80/20 extrusion clamped to the table, with an AprilTag glued to one side so
  that the camera finds it. Press on a flat land beside a T-slot, never across the slot. For S4(b), stick the grip
  tape on a land, or on a flat plate bolted on with T-nuts, and drag along the slot. Measure the tag's offset to the
  pressed faces once with calipers, or touch each face off at 0.2 N on the HEX-E. The tag's ID must differ from the
  object's tags and the DIP tags 24-26.
- S3's block tag (owner, 15:0x): tag36h11 id 0 on the back of the 80/20, for localization. No map of the cell uses
  id 0 (the hand's tag plate 6, the objects' blocks 10-23 and 32-43, the DIP tags 24-26). The tag test plate of 11-07
  has tag36h11 ids 0-4 and stays out of view; the old real_v1 bench tracker (`src/sr2hand/bench/tags.py`) also reads
  id 0, as its cylinder. The block's map is `sr2hand.tracking.tags.plate_map("tag36h11", edge, [0], [(0, 0, 0)])`,
  with `edge` the printed black square measured with calipers. The faces' offsets from the tag, or the touch-offs at
  0.2 N, are still needed.
- The press chatter: the owner attributes it to the skewed cylinder in the misaligned grasp giving the middle finger
  poor contact. The records agree: the pushing finger's targets chattered in every press on both days (3.3-9.0 deg rms
  at 3-10 Hz), both days skewed the cylinder in the close (17-22 deg, 18-27 deg), and the simulation's cylinder, which
  does not skew, does not chatter. The twists' targets with the tips off jittered 0.6-1.1 deg (one 2.1); with the tips
  on 1.3-4.1 deg at 1.4-3.5 Hz.
