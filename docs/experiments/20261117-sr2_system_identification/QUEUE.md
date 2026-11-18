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
  twice.

## Workstation

- W1. Re-plan g13's open pose within the measured MCP range, check in simulation that HOM's close from it reproduces
  the grasp, and ship the plan.
- W2. Fits from S1-S3: joint zero offsets and mount positions; the servo's gain, dead band, friction and overload load
  at 7.5 V; a plant string for the HOM scripts at 7.5 V. One table per fit, against 11-06.
- W3. Predictions of S5 on the fitted plant. After S4: the friction coefficient, the tangential softness and the
  torsion, and the same predictions with the pads, point contact and point contact with torsion (condim 4).
- W4. The three-size landscapes of twist, roll and press with the pads and with point contact (about 6-8 GPU-hours),
  on the plant of W2-W3.
