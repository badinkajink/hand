# Notes from the lab session (corvus) on the identification queue

The lab session (hand-18 on corvus) answers QUEUE.md here and in its commit messages. Times are corvus's, MST.

## 2026-11-17, evening

The owner left at about 19:10: "headed home now. good luck with the remaining trials. as always, be careful! and make
sure that our documentation / summary of the things we worked on, with a clear motivation for evreything described,
gets worked on in a cohesive, readable fashion. good idea to get the magnetic fields response recalibrated too with this
new sysid".

### S2, free-air servo identification at 7.5 V (started 19:06)

- **Layout.** The blocks of 11-06 (`20261104-sr2_hand_crosstalk/lab/make_sysid_plan.py`) unchanged, but run at the roll
  press plan's layout (85, 170): thumb (-36.3, 2.4), index (36.3, 81.6), middle (36.3, -81.6) mm in the palm frame.
  11-06 ran at the gantries' home layout. Reaching home needs a full homing, which grinds every axis into its hard stop,
  and the owner asked on 11-05 not to home. At (70, 110), where tonight's trials ran, the CAD check of the same blocks
  found c40 colliding (-5.2 mm at a pose, -6.3 mm in the move boxes) and every other block under the 12 mm margin.
  At (85, 170) the blocks keep 34-51 mm in the move boxes; c40 keeps 15.3 mm at its poses and 10.0 mm in the boxes
  (thumb yaw -40 with the MCP at 40). The runner's measured-pose gate (`--min-clearance 5`) checks the live mounts.
- **Why it matters for the fits.** Gravity loads depend on the finger's orientation, not on where its gantry sits, so
  the joint data compare with 11-06's. The hand hangs tilted 11.6 deg from vertical (the last putdown's levelling);
  gravity loads therefore differ from a level hand by about 2 %.
- **Files.** Plans and the check: `logs/20261117-sr2_sysid_7v5/plan/` (the refused (70, 110) set in
  `plan_x70y110_refused/`), `lab/make_sysid_plan_at.py` in the crosstalk folder. Runs: `logs/20261117-sr2_sysid_7v5/<block>/`
  (`events.jsonl`, `telemetry.jsonl`), tables in `logs/20261117-sr2_sysid_7v5/tables/` (`dwells.tsv`, `holds_c.tsv`), the
  chain's log `chain.out`. The chain is `lab/s2_chain.sh` here: the flags of `sysid_chain.sh` plus `--max-temp 55`,
  `hom_setup.py diag` before and after each block, and a wait for every servo under 40 C when one reaches 50 C. I
  commit the tables and the runs' small files when the chain ends.
- **The fingertip boards** record throughout in the splitter's file
  `logs/20261117-three_finger_mux/20261117-180517-mux.bin` on corvus (three boards, mux channels 0, 1 and 3, corvus's
  clock; it grows by about 2.4 MB a minute and is not in git). Every free-air series tonight lands in that file, with the
  station's joints in each run's `telemetry.jsonl`. This is the data for the boards' own-pose recalibration the owner
  asked for; the file moves to the workstation by rsync when the night ends.
- **Registers.** Not read: reading them needs the station stopped, and a restart with corvus's key failed once
  (11-16). They wait for a session with the owner.

### Clashes with the queue

- S5 says "with the owner"; the owner told the lab at 18:0x to run unattended HOM twist repeats tonight ("we can be a
  little aggressive with trials"). They run last tonight, on g0 at (70, 110) from
  `plans/20261117-film_x70_y110_twist_gr_mcpstop_plan.json`, against the blind predictions of c5aec6c15.
- S2 runs at (85, 170), not at the home layout (above).

### The night's results (19:36-20:5x)

- **S2** done, all six blocks "plan done" (19:06-19:36), in `data/s2/` (614de9b46). Servos 20-27 C, no flags.
- **S1(b)** (19:36-19:40), `data/s1/rows.jsonl` (057c9a1df): the thumb's tag at 9 poses, |tag - model| median 5.78 mm,
  fitted by the workstation as a mount offset (`data/s1/20261117-thumb_zero_fit.json`). The middle's tag was not
  detected at any of its 9 poses. The index's needs the standard pose (palm z 234 mm, level) with the sphere moved
  off the table's centre: 17-23 candidate poses there on the model.
- **Twist repeats** (19:42-20:08), g0 at (70, 110), 11 of 12 cycles over four arms; records in the bench study's
  `runs/` and `runs_corvus/`, scored in its `data/20261117-night_repeats_tracking.json` (adc0083bc). Tips off as before
  0.447 (3 full runs), with `--filter-step-s 0.002` 0.445 (3 full); tips on as before 0.431 (1 of 3 full: two drift
  guards, the sphere fell at 20:08), with the filter and `--contact-off 0.10` 0.463 (2 of 2 full). The sub-steps added
  about 10 ms of server latency a period, so the arms do not separate; the next A/B uses `--filter-exact`.
- **Putdown failures and their fixes.** 19:49: the levelling turn of 28.6 deg outran the putdown's fixed 12 s move
  limit. 20:02: the levelling's fixed 2 mm slide guard stopped an 18.9 deg turn twice. Both spheres were put down by
  hand-run putdowns and released upright. `repeat_cycle.sh` now caps the turn at 20 deg and falls back to a putdown
  without levelling.
- **The boards' random-pose sweep** (from 20:10, 327 steps, at (85, 170)) and **S1(a)** (after it) record into
  `logs/20261117-sr2_boards_sweep/s3/` and `logs/20261117-sr2_s1a_ranges/`; their summaries follow when they end.
- **Rig at the end:** the sphere lies on its side 227 mm from the table's centre (at B (77, -402) mm), the cylinder
  stands 18 cm from it; the hand at the roll layout, fingers straight, torque on.

### The end of the night (20:53-21:33)

- **The boards' sweep** ran all 327 poses ("plan done", 20:10-20:53): events and plan in `data/sweep/`, telemetry
  (37 MB) in `logs/20261117-sr2_boards_sweep/s3/` on corvus and the workstation. The splitter stopped at 21:3x; its file
  `logs/20261117-three_finger_mux/20261117-180517-mux.bin` (495 MB, 18:05-21:3x) is on both machines, same path: S2,
  S1(a), the sweep and every trial of the evening are in it.
- **S1(a)** (20:53-21:03, again for five ends at 21:28): `data/s1/20261117-s1a_*`. With the stall test every end was
  reached except the two low MCPs: the index's stalls at -10.3 deg, the middle's at -8.5 deg, pushed at about 2 deg of
  lag; the afternoon's -11.4 and -10.0 deg came at 21-22 % duty. The first run's lag rule had stopped the index's and
  middle's yaw and the middle's MCP early on friction lag (S2 shows them at -40 and 61 deg).
- **The second A/B** with `--filter-exact` (21:05-21:25): 9 of 12 cycles, records in the bench study's `runs/` and
  `runs_corvus/`, scored with the first in its `data/20261117-night_repeats_tracking.json`. Per arm over both blocks:
  tips off as before 0.433 (6 full), 2 ms sub-steps 0.445 (3 full), exact 0.426 (2 full); tips on as before 0.405 (3
  of 5 full, two drift-outs), sub-steps with hysteresis 0.463 (2 of 2), exact with hysteresis 0.310 (2 of 2).
- **The lab lights went off at about 21:25** (frame luma 14 and 5). Cycle 9's putdown lost the tracker with the
  sphere in the hand; it was set down on the HEX-E (40.1 mm, 2.38 N), released to the open pose, the hand raised 70 mm
  and the fingers straightened, all without vision. The sphere stands somewhere under the hand at the table's centre.
- **The rig:** the hand at the roll layout, palm about z 300 mm, fingers straight, torque on, servos 22-36 C with no
  flags. The workstation's HOM servers 47051, 47052, 47057-47060 are stopped. The cylinder stands 18 cm off the centre.

## 2026-11-18, morning: S1(b) with the arm lowered (11:12-11:41)

At 09:4x the owner wrote: "the middle finger tag should be visible from at least one camera now, when it is curled into
the tag-visible joint config. if you yaw the thumb finger toward +x and also step the thumb +x, it should be visible by
both". The arm still read running at the night's pose then. The owner lowered it 43 mm at 11:12:24-27 (TCP z 305 to
262 mm), switched it off at 11:12:30 and moved the sphere aside ("ok done im headed out for 2 hours"). The arm's
read-only stream still gives its pose while it is off (robot mode 3), so the tracker places the frames as before. The
watcher's log is `logs/20261118-sr2_tags/arm_watch.jsonl`.

- **Why the poses changed.** The palm's origin is now 120 mm above the table (z_B 123 mm; 298 mm at 19:36 last night),
  level with both cameras (z_B 96 and 106 mm). The index's and middle's tags face +x at zero, away from both cameras,
  and a curl turns them downward. Last night's middle poses now view at 50-100 deg. Curled further (MCP 58, PIP 75-80),
  a tag faces -x and both cameras see it. MCP 58 and PIP 80 are the ends S1(a) reached last night, below the MCP pots'
  dead zone (62-68 deg). Lowering the palm brought the tags into frame without blocking them.
- **The tool.** `lab/s1b_tags.py` takes the mounts from the station (so a moved gantry is modelled where it is), chooses
  poses for given `--fingers`, keeps only poses both cameras see with `--both`, measures the sphere and the cylinder
  with the tracker (`--avoid-map MAP:R`, 60 mm round each, 20 mm more clear along each pose's path from zero), moves one
  gantry (`mount`), and refuses a run if the arm, a mount or an object moved since the choice. The rows now record
  which cameras saw the tag.
- **The thumb's step.** The thumb's yaw axis lies along x, so a positive yaw swings the finger toward +y and turns its tag
  between the two cameras. On the model, stepping the thumb's gantry +12 mm (to x -24.3, inside the station's -20 limit)
  raised the poses both cameras see within 40 deg from 3 to 15 (within 50 deg from 60 to 71).
- **Runs** (`logs/20261118-sr2_s1b_tags/<set>/`; copies and the summary in `data/s1/20261118-*`, from `lab/s1b_summary.py`):

| set | mount (station, mm) | runs seen | cameras | tag minus model, median (mm) | normal off the model | repeat |
|---|---|---|---|---|---|---|
| middle | (36.3, -81.6) | 18 of 18 | both, every sample | (-1.6, -1.5, +5.7), 6.4 | 12.2 deg, +12.2 about the flexion axis | 0.29 mm |
| index | (36.3, 81.6) | 14 of 18 | both (628), 0344 alone (15) | (+8.5, -2.5, +0.2), 9.0 | 9.2 deg, +9.2 about the flexion axis | 0.27 mm |
| thumb | (-24.3, 2.4) | 18 of 18 | both (821), 0344 alone (20) | (+7.1, +3.0, -4.1), 8.7 | 1.9 deg | 0.37 mm |
| thumb_home | (-36.3, 2.4) | 9 of 9 | both (818), 0344 alone (18) | (+6.8, +3.5, -4.0), 8.6 | 1.1 deg | one run |

  The two index poses never seen, twice each, are the two at yaw -25 to -30 with the least curl ((-25, 58, 65) and
  (-30, 43, 80)). The model put them at 47-49 deg from camera 1918. The thumb's gantry step came out within 0.1-0.9 mm of
  the 12 mm commanded (its offsets at the two mounts agree to that).
- **What the normals say.** Assuming the stickers lie flat on the links as modelled, the index's and middle's DIP links
  sit 9 and 12 deg more flexed than the station's joint readings put them. The turn is about the flexion axis, under 1 deg
  out of the plane. Every pose where those tags are visible is at MCP 47-58 and PIP 65-80. The thumb's normal agrees
  within 2 deg at MCP 23 or less and PIP 30 or less. These poses cannot separate joint zero offsets from pots that read
  low near the top of their range; tag views at less curl would. The workstation's W2 fit has the rows with the measured
  joints and both mounts of the thumb.
- **The boards** recorded every pose in `logs/20261118-three_finger_mux/20261118-112222-mux.bin` (46 MB, 11:22-11:41,
  corvus). Its "dropped by outlet" counts are the live outlets nobody read; the file holds every record.
- **Rig after.** Every finger at zero, the thumb back at (-36.3, 2.4), torque on, servos 19-26 C, no flags. The arm
  is off at the owner's lowered pose (palm origin z_B 123 mm). The sphere is 200 mm from the palm toward +y; the cylinder
  is 190 mm away toward -y.

## 2026-11-18, afternoon: S3 on the owner's block, the index (15:28-16:06)

- **The block** (owner, 15:3x-15:4x): a 20 x 20 mm 80/20 post about 128 mm tall on the table, with an L base clamped to
  it. Two 45 x 45 x 1 mm plates are glued on opposite faces, their tops about 128 mm up. Tag36h11 id 0 is on one plate,
  and grip tape (0.75-1 mm) covers the other, whose face is 22 mm behind the tag's. Clamped (15:42), both cameras
  triangulate the tag's edge at 39.85 mm (sd 0.01). The tag sits 108.2 mm above the table, tilted -0.2 deg. The plates
  face +-x_P to 2.1 deg. The owner allows a 90 deg wrist turn if the yaw needs it.
- **Approach** (`lab/s3_approach.py`, plan `data/s3/20261118-155238_approach_index.json`, 15:52:51-15:53:26): a pure
  translation, up 140.7 mm, across 85.6 mm, down 64.2 mm. On the way down the least clearance was 8.6 mm, and the pad
  stopped 4.6 mm off the plate (`data/s3/20261118-155336_block_at_press_pose.json`). The HEX-E changed at most 1.5 N.
- **Index MCP head on** (`data/s3/20261118-155403_index_mcp_press.jsonl`; raw and boards in
  `logs/20261118-sr2_s3_press/`, `logs/20261118-three_finger_mux/20261118-155402-mux.bin`, both machines). The force
  along x_P rose about 0.4 N per deg of target offset to 1.75 N at 4.8 deg. It held there while the MCP stalled, then
  jumped to 2.55 N with a 1.5 deg jump of the reading, and stayed at about 2.5 N. The MCP flagged overload (status 32)
  at load -630 after about 20 s at -540 to -630. On the step back the force stayed flat until the target came within
  about 2 deg of the reading. Contact came at a reading of about 3 deg, and the reading advanced 5.5 deg past it. The
  PIP reading was pushed back 1.2 deg, and the post moved 0.065 mm per N.
- **Index yaw with the MCP holding 2 N** (`data/s3/20261118-160047_index_yaw_mcp_hold.jsonl`). The hold reached
  2.01 N at an MCP offset of 5.1 deg. While the pad stuck, the yaw's tangential force (y_P) rose to 0.69 N with 0.9 deg
  of reading advance; W3's 0.76 N m/rad give predicts 4.6 deg there. Then it stick-slipped at 0.86-1.05 N against
  1.9 N normal (friction about 0.5) over 14 deg of reading, about 21 mm along the tape, to within about 1.5 mm of the
  plate's edge. On the step back the force (0.88-0.93 N) and the reading stayed put until the target passed the
  reading. Reversed, the yaw stalled at -8.8 deg with offset -9 deg and -0.90 N: 0.08 N m, kp 0.525 x 9 deg. It
  returned to zero when the MCP unloaded.
- **Drift.** The HEX-E's z drifts about 0.4 N a minute after an arm move; x and y held to 0.02 N. The stops now read
  the force's x_P and y_P only.
- **Rig after.** The index is at zero, beside the plate at the press pose. The yaw servo is at 47 C and the MCP at
  39 C; the others are 20-22 C, with no flags. The arm runs at the press pose, and the splitter records.

## 2026-11-18, late afternoon: head-on yaw presses with the hand turned (16:17-16:45)

- **Go.** The owner gave the go for the wrist turn at 16:2x ("you have the go ahead on the wrist turn, im here to
  monitor"). At 16:45 they left ("try to get at much yaw calibration done now though").
- **Turned approach** (`lab/s3_approach.py plan --press yaw`): the hand turns -92 deg about z_P, so the index's and
  middle's -y yaw drives the tip's side into the taped plate. The level leg stopped at 3.08 N after 44.9 mm. That was
  the hand's cable bundle dragging on the table; with a 5 N guard the leg finished.
- **Collision, 16:25.** The descent stopped at 3.13 N, 7.6 mm above its target: the index's DIP link had landed on the
  ~3 mm lip on top of the block (owner). The planner modelled each finger as hom_setup's three balls (tip pad 11 mm,
  tip end 6 mm, PIP 8 mm) and left out two things:
  - the DIP link's side, which by the PIP reaches 23 mm from the finger's axis on -y and 17 mm on +y (cad_v3
    collision spheres); with the hand turned, that side faced the block;
  - the lip.
  A lift then caught at 6 N. The hand moved 10 mm horizontally away from the plate (1.1 N) and rose 30 mm, free. The
  owner checked the DIP link: fine.
- **Fix.** The planner now checks every finger's cad_v3 collision spheres (mount, yaw, MCP and DIP link) plus the
  three balls, and the lip as a 3 mm box over the block's top. A part squarely in front of the pressed face may end
  at the gap; a part over an edge needs 5 mm. Pre-move joint checks use the median of five readings, after a single
  1.5 deg spike on the thumb's yaw: the owner was moving the thumb. The new model flags the collided plan (-2.6 mm with
  the noise). The -y side needs the tip 13 mm higher, pressing the plate's upper part, and a 6 mm gap.
- **Results** (`data/s3/*_head_on.jsonl`; the yaw from its reading at contact):

| Press | Slope before the stall | Stall, then jump | Overload flag (status 32) | Step back |
|---|---|---|---|---|
| index 16:33 | 0.21 N per deg of offset | 1.0 N, then 1.5 N with a 1.8 deg jump | offset 12.4 deg, load -600 | flat to within 1 deg |
| index 16:37 | 0.25 N per deg | 1.05 N, no jump | offset 12.5 deg, load -660 | flat to within 0.3 deg |
| middle 16:42 | 0.2 N per deg | 1.1 N, then 1.74 N with a 2 deg jump | offset 13.4 deg, load -675 | flat to within 0.9 deg |

- **Reading.** Past first contact the reading advanced 2.4-3.5 deg per N, counting the jumps. The contact force had
  a downward part about 0.8 times the horizontal one: friction on the tape as the yaw's arc lifts the tip, plus the
  HEX-E's drift. hand-4a predicted 0.19 N per deg, and 2.8 N at 14-15 deg unless the servo stops near the MCP's
  0.17 N m. The slope agrees, but the yaw's overload trips at 1.0-1.7 N, 0.09-0.15 N m at 88 mm.

## 2026-11-18, 16:46-17:18: the yaw series, unattended (owner away)

- **Presses.** Fourteen head-on yaw presses, all with the hand turned -92 deg and the tip's side 13 mm above the
  tag's height (`lab/s3_yaw_summary.py` -> `data/s3/20261118-yaw_head_on_summary.json`):
  - index and middle toward -y into the taped plate, and toward +y into the tag plate;
  - the thumb both ways;
  - slow (0.5 deg every 2 s) and fast (2 deg every 1 s).
- **Results.**
  - Contact came once the target ran 0.6-2.3 deg past the reading.
  - The force then rose 0.12-0.34 N per deg of target offset (median about 0.22).
  - It stalled, often jumped with a 1.5-2 deg jump of the reading, and peaked at 1.07-2.27 N at 7.6-14.3 deg of
    offset: 0.09-0.20 N m at 88 mm. The +y presses into the tag plate reached the most (index 1.9-2.15 N, thumb
    2.18-2.27 N).
  - Slow runs ended at the yaw's overload flag (status 32, loads 600-795); fast runs at the tool's 900-load limit,
    with forces like the slow runs'.
  - On the step back the force held until the target came within -0.15 to +1.2 deg of the reading (the clutch
    plant's hold rule).
  - hand-4a's 2.8 N at 14-15 deg did not appear: the yaws saturate near 0.1-0.2 N m.
- **Moves.** Each finger's approach was planned with the CAD spheres and moved without contact; the guards changed at
  most 1.8 N. The tag-plate presses hide the tag, so they ran without block tracking, and the hand rose 64 mm before
  each new plan. The thumb's +y side met the tag plate after about 1.5 deg of yaw, about 3 mm sooner than the model
  said; the 5 mm margins took it.
- **Parked** at 17:18: palm origin 152 mm straight above the session's start, the wrist back within 0.06 deg of its
  first orientation, every finger at zero, torque on. The splitter's file `20261118-155402-mux.bin` holds all of
  today's presses.

## 2026-11-18, 18:34-18:54: the middle's pad-face press, the pads' angle presses and S4(b) (owner's go, 18:3x)

The owner, 18:3x: "ok go ahead on these plans".

- **Middle MCP into the taped plate** (`data/s3/20261118-183614_middle_mcp_press.jsonl`, pad face, slow).
  - Contact came within 1 deg of MCP, about 5 mm sooner than the model said.
  - The force rose 0.47 N per deg of offset to 2.0 N, then jumped to 2.63 N with a 1.5 deg jump of the reading, and
    stopped at the 3 N cap.
  - On the step back it held 3.07 N until the target came within about 1 deg of the reading.
  - The middle's board recorded it (the splitter's new file from 18:36).
- **Angle presses for the boards** (quick, 1 deg per s to 2 N, `*_pip_prebend.jsonl`), each pad pressed face on
  with the PIP pre-bent 15 and 30 deg (0 deg: the slow presses), so that the contact moves along the pad.
  - The index met the plate where the model said, within about 1-2 mm.
  - The middle met it 7 mm (PIP 15) and 9 mm (PIP 30) sooner. The middle's model error grows with its flexion, as
    S1(b)'s tag normals (12 deg) suggested; the planner's 10 and 14 mm gaps took it.
  - The planner aims the pad's 11 mm ball (`--lead ball`) when the PIP is bent. Approach joints are now the median
    of five readings, after a 2 deg single-sample jump of the thumb's PIP.
- **S4(b)** (`lab/s4b_drag.py`; `data/s3/20261118-18*_index_drag*`, with the raw HEX-E, TCP and joints).
  - Method: the index's MCP held the pad on the tape at 0.5, 1 and 2 N, and the arm dragged the hand 5 mm up and
    down at 1 mm/s, twice each way.
  - The drag runs along the HEX-E's z, which drifts about 0.4 N a minute, so the friction comes from half the
    difference between the up and down legs' plateaus.
  - At 1 N: about 0.4 N against 0.9 N normal (about 0.45). At 2 N: about 0.83 N against 2.1-2.9 N (about 0.3-0.4).
  - The normal force rose on the downward legs, to 3.49 N at most in the 2 N run. That is over the queue's 3 N rule,
    because the legs' guard (4 N) counts change from each leg's start. The servos kept no flags.
- **Parked** at 18:54: palm origin 231 mm above the session's start, unturned, every finger near zero, torque on, no
  flags.

## Night of 2026-11-18 (19:17-21:42): the calibration suite, unattended

- **Go and plan.** The owner gave the go by relay through hand-4a (owner, 19:1x: "yea you have the go ahead to run
  tonight's session"). hand-4a's plan:
  1. pad friction for all three pads at 0.5-3 N, drag speeds 0.2 and 2 mm/s, rests of 1, 10 and 60 s;
  2. the three PIPs and the thumb's MCP, three slow presses each;
  3. the thumb's board, an MCP face press and angle presses at PIP 15 and 30 deg.
- **Changes** (hand-4a agreed):
  - No 3 N hold: the queue's 3 N rule. Drag legs stop at 3 N absolute (x_P, y_P).
  - The thumb's friction ran on the tag plate: its pad faces the tag's side, and the taped plate needs a 180 deg wrist
    turn (the owner OK'd 90).
  - Runs were grouped by finger.
- **Scripts** are in `lab/night/`; logs in `logs/20261118-sr2_night/`; boards in
  `logs/20261118-three_finger_mux/20261118-191744-mux.bin` (344 MB). All of it is rsynced to the workstation.
- **Friction** (`lab/s4b_drag.py --protocol`; `data/s3/20261118-*_protocol.jsonl`, raw in logs). Index (19:18-19:43)
  and middle (20:03-20:28) on the tape, thumb (20:46-21:11) on the tag plate. Each held 0.5, 1 and 2 N; for each speed
  and rest, 5 mm up, then rest, then 5 mm down, along z. Each rest's raw segment gives the HEX-E's z drift for the leg
  after it. No early stops. The MCPs reached 33-34 C.
- **Index PIP**, three presses (`*_index_pip_press.jsonl`): contact at a PIP reading near 10 deg. The force reached
  2.6-2.95 N when the 35 deg target travel ran out; the reading advanced about 15 deg past contact, since the MCP
  behind gives too.
- **Middle PIP**, three presses (`*_middle_pip_press.jsonl`): each reached the 3 N stop at a target of 31.5 deg
  (readings 23.1-24.6 deg, offsets 6.9-8.4 deg).
- **Thumb MCP**, three presses on the tag plate (`*_thumb_mcp_press.jsonl`):
  1. stopped when the MCP reached 55 C, at 2.6 N, offset 12.9 deg, load -705;
  2. reached the 3 N stop at offset 10.1 deg;
  3. overload flag (status 32) at 2.26 N, offset 14.3 deg.
- **Thumb PIP**, three presses (`*_thumb_pip_press.jsonl`): 3.07 N and 3.00 N at the stop; the third hit the overload
  flag at 2.38 N, offset 13.6 deg.
- **Rule breach.** The scripts waited 60 s between presses, so the thumb's second and third MCP presses started above
  the queue's 40 C resume limit. The 55 C stop held. Next time the scripts wait under 40 C.
- **Not run: the thumb's angle presses** (PIP 15 and 30). The lab lights went off by 21:40 (frames' mean luma 16,
  against 137 at 20:02, as at 21:25 the night before). The planner refused without the tag, and I did not approach the
  block blind. Phase C first stopped at 21:39 for the same reason: the tag was not seen after the 80 mm rise.
- **Parked** at 21:42: palm origin z_B 356.7 over the session's start, unturned; all fingers within 3 deg of zero; no
  flags; servos 21-32 C; torque on; arm running (mode 7). Splitter stopped.

## 2026-11-19, 12:12-13:54: the tips' PLA sides on the tape

- **Go.** The owner gave the go to the workstation session at about 12:00 ("we can go for the pla on tape tests"),
  relayed by hand-4a. The test as proposed on 11-18:
  - the hand turned -92 deg, as in the head-on yaw presses;
  - the index's and middle's yaws hold their tips' sides on the taped plate at about 0.5, 1 and 2 N;
  - the arm drags 5 mm up and down, and the friction is read as in S4(b).
  hand-4a added phase A's rests and speeds (1, 10, 60 s; 0.2 and 2 mm/s), which ran. Two suggestions wait for the
  owner's word: a 20 mm/s leg (on a 5 mm leg it needs about 0.2 m/s^2 against the drags' 0.02) and the thumb pad on
  the tape.
- **Where the side meets the plate** (`lab/s3_yaw_contact.py`, the cad_v3 spheres).
  - The DIP link's side is flat, 10.5 mm from the finger's axis, from 4 to 32 mm above the link's end. Above that it
    steps out to 17-23 mm.
  - At 11-18's plan (gap 6, dz 13) the flat side met the plate almost flush, its top 0.1 mm from the lip. Part of the
    force could then have gone to the lip.
  - At gap 14 and dz 15 the yaw meets the tape about 9 deg further on. The link's lower end touches first, 15 mm below
    the plate's top, and the flat side's top stays 0.6-1 mm off the face. After the 5 mm drag up the contact is 10 mm
    below the top.
  - The index's force began 1.2 deg of yaw sooner than the model said (a reading of -5.3 to -5.6 deg, against -6.7): the
    face sits about 2 mm closer than modelled. The middle's 0.5 N hold reading (-7.9 deg) matched its model (-7.8).
- **The yaw hold** (`lab/s4b_drag.py --joint yaw --sign -1 --hold-load 500 --clutch 2`). The yaw steps until the force
  reaches N or its load reaches 500, and keeps the force reached. The target then steps back until it lies 2 deg past
  the reading. The gears kept the force while the load fell from 480-510 to 135-150, so the yaw never neared its
  overload flag (600 and more on 11-18).
  - Index: 0.47 N, then 0.97 N (the hold reached the load cap at 0.94 N), then 1.0 N. The third hold also stopped at the
    cap, so the "2 N" level repeats the 1 N one. The yaw ran 24-40 C and settled at 36 C at load 135-150.
  - Middle: 0.50 N, 0.99 N (load 405) and 1.31 N (the cap). The yaw ran 23-36 C.
  - During the index's 1 N clutch a force along the plate's width (x_P) grew from 0.1 to 0.5 N and stayed through the
    legs.
- **Free-air controls.** The same legs with the finger off the plate: before and after each finger's protocol, and at
  the night's pad poses, unturned, for phases A and B. The cable bundle biases every drag. For the index's pose, hand-4a's
  estimator gives 0.08-0.13 N along the drag at 0.2 mm/s and 0.16-0.22 N at 2 mm/s. On the up legs it also gives
  -0.2 to -0.39 N along y_P, the side presses' normal. At 0.5-1 N of normal force that is a large part of the reading;
  hand-4a subtracts the controls leg by leg.
- **Result** (hand-4a: 38097ea33, f30587ed0; `data/s3/20261119-pla_side_friction.json`). With the controls
  subtracted:
  - the index's PLA side slid at 0.43-0.54 of the normal force at 0.8-1.1 N (median 0.47): 0.48 at 0.2 mm/s and 0.46
    at 2 mm/s;
  - the middle's slid at 0.72-0.87 at 0.9-1.35 N (median 0.78);
  - uncorrected, the same legs read 0.61-1.12 (index) and 0.97-1.86 (middle);
  - neither side stuck and slipped.
  The middle/index ratio, 1.6, is the pads' ratio of 11-18. By the plans all four contacts lie on the same patch of tape
  (aimed at l = 0, z_B 116-117). In the turned hand the middle's x_P model error, about 5 mm, lies along the width, so
  its side touched within about 5 mm of the index's spot.
- **Records** (`data/s3/`; raws over 2 MB in `logs/20261119-sr2_s3_press/`, rsynced):
  - index: 20261119-122906 (control), 123219 (loaded), 130711 (control); approach 122743;
  - middle: 131125 (control), 131412 (loaded), 134049 (control); approach 131019;
  - unturned controls: 134601 (the index at 191629's pose), 135018 (the middle at 200220's pose).
  - Boards: `logs/20261119-three_finger_mux/20261119-122901-mux.bin` (203 MB, rsynced).
  - Scripts: `lab/day/`.
- **Stops and slips.**
  - The first descent, 149 mm from the park, stopped on its 3 N guard at 105 mm with nothing near the hand. Two frames
    and a steady HEX-E showed no contact: the cable loop's pull changes as the hand lowers. Re-planned, the last 61 mm
    finished at 2.65 N.
  - My waiter for the end of the index's protocol matched its own shell: the next s4b_drag command in the same line
    held the pattern (lesson 9 of CLAUDE.md). The index's second control started 8 min late. Nothing moved meanwhile.
- **Parked** at 13:53: palm origin z_B 356.8 over the session's start, unturned. Fingers within 4.4 deg of zero, no
  flags, servos 19-23 C, torque on, arm running (mode 7). Splitter stopped.

## 2026-11-19, 14:21-16:0x: the boards against the HEX-E, faster drags, and the friction's drift

- **Go.** The owner to the workstation session, 14:0x-14:19, relayed by hand-4a: "can we also run some magnetic
  fingertip calibrations? what else can we do with this before we take out this mount?", then "you have my go ahead on
  whatever runs". Scripts in `lab/day/`; records and raws in `data/s3/` (all under 2 MB); boards in
  `logs/20261119-three_finger_mux/20261119-142714-mux.bin`.
- **Angle presses for the boards** (`day/20261119_board_presses.sh`). Each pad face on with its PIP pre-bent 15 and
  30 deg, so the contact moves along the pad; the MCP pressed, each press with its own approach.
  - The index's first press (PIP 15, 1 deg steps) stuck at 1.95 N for six steps while the MCP's load built from 285 to
    585. Then the gear slipped and the force jumped to 3.48 N, over the 3 N rule.
  - The rest stopped at 2.6 N or at the MCP's load 450, in 0.5 deg steps every 0.5 s: index PIP 30 at the load cap with
    1.99 N; middle PIP 15 and 30 at 2.96 and 2.62 N; thumb PIP 15 and 30 (on the tag plate, last night's unrun pair)
    at 2.62 and 2.65 N.
  - The index's MCP stalls near 2 N in both pre-bent presses, so it cannot reach 3 N on this rig without such a jump.
- **Shear across the finger** (`day/20261119_shear_across.sh`): each pad head on at PIP 0, the MCP holding 0 (free
  air), 1 and 2 N, the arm dragging +-5 mm along the plate's width at 1 mm/s. hand-4a (31ce8b2f2): the pads stayed put
  and the fingers gave as springs, 0.15-0.28 N/mm with 1-2 deg of yaw. That is known shear at a fixed contact for the
  boards, and no friction.
- **20 mm/s** (`day/20261119_fast_legs.sh`, `_more.sh`; `s4b_drag.py --acc 0.5`, which reaches 20 mm/s within
  0.4 mm). The index pad at phase A's pose, with free-air controls at the same speeds.
  - At 1 N, three pairs each at 2 and 20 mm/s.
  - The 2 N level stopped on its first 2 mm/s down leg: the normal force rose from 2.2 to 3.27 N and the 3 N guard caught
    it at 3.03 N.
  - An extra run held 1.5 N for five pairs at 20 mm/s.
  - hand-4a (9c4e7e6d4): the pad slid at 0.63 at 2 mm/s and 0.72 at 20 mm/s, against 0.40 at the same pose last night.
- **A-B-A** (`day/20261119_aba.sh`, hand-4a's question): index, middle, index pad at 1 N and 2 mm/s, each with a
  free-air control. The index slid at 0.71 and 0.59, the middle at 0.82 between them (hand-4a, 4e8c7f559). Within one
  session the middle grips 1.2-1.4 times the index. Most of the 1.8-2 times of 11-18 was the index's pad rising from
  0.40 (11-18 19:4x) to 0.6-0.7 today.
- **Thumb pad on the tape: not run.** The pad faces +x_P, and the taped face needs a 180 deg turn of the wrist. The
  owner's rule is no big joint-6 turns (11-16), with 90 OK'd on 11-18; a strip of the tape on the tag plate would do
  instead. Both are with the owner.
- **The boards' free-air sweep at the trials' layout** (from 15:12). Gantries to (70, 110) (`hom_setup.py layout`,
  filmed). Then 327 steps of `20261104-sr2_hand_crosstalk/lab/run_plan.py` at the park, the motors kept homed. Plan
  `data/sweep/20261119-plan_random_x70y110.json`: make_random_at.py, seed 4, 300 random poses at least 12 mm finger to
  finger at the station's mounts (-28.82, 2.40; 28.82, 51.60; 28.82, -51.60). make_random_at.py takes the station's
  mount coordinates, not a plan's d6_P_mm; the first plan, built from d6_P_mm, overstated the clearances by up to 6 mm
  and is kept as `logs/20261119-sr2_boards_sweep/20261119-plan_random_x70y110_WRONG_MOUNTS.json`. Events and
  telemetry in `logs/20261119-sr2_boards_sweep/s1/`.
  - "plan done" at 15:55: all 327 steps, no safety stop, every pose reached
    (`data/sweep/20261119-sweep_x70y110_events.jsonl`; the 38 MB telemetry stays in logs, rsynced).
  - The measured clearance at the measured joints and the station's mounts was 11.7 mm at least (median 21.1). The
    settle error had a median of 1.4 deg; 11 poses ended over 3 deg, at most 4.0.
  - The boards recorded throughout (`20261119-142714-mux.bin`, 215 MB, rsynced).
- **End, 15:57.** The gantries are back at the roll layout (85, 170), filmed like the move out
  (`logs/20261119-sr2_boards_sweep/*_layout_*.mp4`). The hand is parked unturned over the session's start, all
  fingers within 0.6 deg of zero, no flags, servos 21-26 C, torque on, arm running (mode 7). Splitter stopped.

## 2026-11-19, 16:36-17:45: what the fingertip boards read, and why "half the HEX-E"

The owner, 16:3x: "the contact area vs force sensing is really disconcerting. is tehre any way to estimate the size of
the contact patch and thus get better performance. can we try to learn that from our existing calibrations? ... you have
my go ahead to execute plans to further our understanding of the existing models"; 16:4x: "you can calibrate more on the
post, you can also run experiments on the sphere".

- **The calibrations already vary the contact's size** (`sim/20261119-board_patch_size_calibration.py`,
  `data/s3/20261119-board_patch_size_calibration.json`). The v3 tips were pressed with R 2 and R 5 mm spheres and an
  R 12.5 mm rod (UR5 tip mode, 10-28 to 10-30; inverted rig, 11-02 to 11-04). Matched to the R 5 sphere at the same point
  and force:
  - the R 2 sphere reads 1.03-1.05 times its field per newton and the rod 0.93-1.09 (tips 2, 4, 5, 6);
  - the depth per newton follows Hertz: 1.26-1.33 for the R 2 sphere, 0.68-0.91 for the rod;
  - the field pattern differs little (cosine 0.96-0.996 against 0.995-0.999 repeat to repeat).
  The boards measure force, not patch size. The four magnets sit 7.5 mm under the apex, and the patches are 1-2 mm.
- **Arm presses** (`lab/arm_press.py`, new; `day/20261119_arm_press_plate.sh`, `_pip.sh`). The arm pushes one pad
  straight into a plate at 0.2 mm/s to 2.5 N, holds 2 s and backs off, while the finger's servos hold. Three reps per set.
  - The joints still give 0.6-1.5 deg in extension (servo loads 60-100). The finger yields about 0.45 N/mm at the tip,
    so 2.5 N takes 3-5 mm past contact (`--travel 8`).
  - The free-air check at the park (16:55) and the HEX-E's 100 Hz samples (+-0.2 N) led to a 0.1 s mean for the stops
    and 0.4 s of blanking after a move starts.
  - At 17:11 my own `s3_press.py check` opened a second HEX-E client, which reconfigured the DAQ and took the stream.
    The middle's third press stopped in its air leg, untouched. **Never open a second HEX-E client while one runs.**
- **Raw field per newton on the hand** (`sim/20261119-arm_press_boards.py`, `data/s3/20261119-arm_press_boards.json`).
  This is the board's 15-component change against the HEX-E's force along the press, matched to the tip's calibration
  presses anywhere on the pad by the cosine of the patterns:

  | press (2.5 N) | uT/N | best cosine | calibration there | GP estimator | pattern-matched |
  |---|---|---|---|---|---|
  | index, PIP 0, taped plate | 6.2-7.0 | 0.36-0.44 | none matches | 0.20-0.28 | 0.24-0.26 |
  | middle, PIP 0, taped plate | 7.2-7.5 | 0.55-0.61 | none matches | 0.30 | 0.16-0.18 |
  | thumb, PIP 0, tag plate | 10.2-12.4 | 0.75-0.84 | 23-30 at z 15-18 | 0.23-0.34 | 0.25-0.33 |
  | index, PIP 10 | 18.7-19.1 | 0.96-0.97 | 28 at z 19 | 0.43-0.45 | 0.63-0.64 |
  | index, PIP 20 | 24.6-25.7 | 0.986-0.990 | 28 at z 19 | 0.60-0.63 | 0.83-0.87 |

  - The pad frame's z runs from the proximal mount face (0) to the distal end (21.4); the apex is at 10.9.
  - With the PIP bent 20 deg, the plate meets the distal part of the pad. There the boards read 0.88-0.92 of tip 3's
    calibration, and their pattern matches it at 0.99. So the boards still hold their calibration, and the HEX-E is not
    2x off.
  - With the finger straight, the plate meets the pad's flat proximal end, and no calibrated contact gives that field.
    The proximal half of the 11/22 pad turns only 12 deg over 11 mm, so a few degrees of lean slide the contact
    there: the hand is tilted 2.7 deg and the joints give 1-1.5 deg. Tip 3's UR5 sphere presses start at z 4.3 mm.
  - Summing the calibration's point fields over a flat-plate patch (Winkler,
    `sim/20261119-board_patch_superposition.py`, with tip 3's rig rod presses down to z 0.6) reproduces PIP 20 at
    cosine 0.993 and 0.92-0.94 of the field. No patch on the pad reproduces PIP 0 (at most 0.51).
  - So at PIP 0 something besides the pad's surface carries part of the load, or the pad deforms in a way the calibration
    never sampled. The DIP link's body just proximal of the pad is the first candidate. The cameras cannot see the
    taped side.
- **The 11-18 presses agree** (`sim/20261119-board_field_per_newton_hand.py`, `data/s3/20261119-board_field_per_newton_hand.json`).
  Raw field per HEX-E newton is 0.37-0.68 of the calibration at the GP's point, and best-match cosines are 0.70-0.97.
  In a joint-driven press the finger's own motion moves the board too: 1.7 uT per degree of MCP in free air, before
  contact. The trials' own-pose term takes off part of the contact signal with it; without it, the GP's slopes were
  0.64-0.86 on the bent-PIP presses against 0.44-0.56 with it. In the step back, the servo's load fell 200-400 at
  constant force and the field moved 1-3 uT, so the motor current matters little.
- **The estimator** (`sim/20261119-board_force_pattern_match.py`, `data/s3/20261119-board_force_pattern_match.json`).
  - Take the calibration point whose field per newton is most nearly parallel to the measured field; the least-squares
    force there reads 0.83-0.87 of the HEX-E at PIP 20, where the trials' GP reads 0.60-0.63.
  - The GP itself loses a quarter of the force, and adds 0.15-0.3 N at zero force.
  - The best cosine works as a trust flag: 0.99 when the force is right, under 0.7 when no calibrated contact explains
    the field.
- The sphere is not an indenter on this rig. A curled finger's pad faces the palm (up), so no pad can press down on the
  sphere's top. Sideways, the stem would tip it. The owner will not set up a fixed sphere tonight (17:2x, relayed).
- At 17:30 the hand parked over the start and the post was cleared for removal (the owner's question, relayed by the
  coordinating session). It still stood in the cameras' frames at 17:34.
