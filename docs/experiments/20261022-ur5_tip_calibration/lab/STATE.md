# UR5 cell: current state

Updated 2026-11-03 08:31 by the lab. From the tip swap on, the lab session updates this file at every session
start and end. This file gives the present state of the cell. `../QUEUE.md` and `NOTES.md` are the history. Every
request to the owner must come from this file.

## On the robot

| Item | Now | Registration (folder here) |
|---|---|---|
| Probe | probe base with the 5 mm sphere (sphere5) on mount v7's adapter disc on the HEX-E (owner, 11-02 13:0x); no tip on the robot | none yet |
| Strip | the seat strip on one 2x4 clamped to the table (owner, 11-02); 192 mm from the place the 10-30 plans assume | tonight's sessions: the frame through both keys' socket points (`20261102-strip_frame_two_keys.json`), by `--frame`, with `--home-from poses.jsonl:strip_key1_seated_4`; each seat's grid aims by its own pass 1 fit. For later plans: the workstation's frame from the three pads' fits (`../../20261030-inverted_rig_seat_block/data/20261102-strip_frame_pads.json`, 9cd8b375: the two-key frame turned 3.44 deg about y_A and moved (2.48, 0.10, -3.70) mm; the pads within 0.14 to 0.22 mm on average, the rest the strip's twist) |
| Held tips | seat 1 (y_A -80, the right end seen from the robot) v3 ID 6 on mux channel 3; seat 2 v3 ID 5 on channel 2; seat 3 (left) v3 ID 4 on channel 0 (owner, 11-02 13:0x); all three stream at 138 records/s | seat 1: `20261102-tip_calibration/145010-s1v6b` (pass 1, 0.055 mm rms; its pad 1.68 mm and 4.1 deg from the CAD in the two-key frame); seat 2: `182710-s2v5` (0.051 mm rms; 1.21 mm, 3.5 deg); seat 3: `220337-s3v4` (0.051 mm rms; 0.81 mm, 2.9 deg); the overnight runs resume these sessions |
| Old fixtures | the low sphere, rod, tall indenter and holder: off the table as the strip's README asks; the camera frame of 13:05 shows only the strip, its tips and the mux board on the 2x4 | as before 10-30 |

`cell.json` holds the same registrations for the path checks and the renders.

The HEX-E's link fails about once per 6 to 9 h of running: 7 losses since 10-25 (`../QUEUE.md` R31, 17:42 on 10-28).
An eighth came on 11-02 between 13:03 and 13:20, while the owner moved the arm by hand to the press pose (joint 6 to
+178 deg); by 13:27 it streamed again after a restart, untared (Fz 629 N, status 0x40000), as after the loss of 10-22.
A loss aborts three actions and stops the session, and the robot then waits for a reseat. The cable is clamped 3 to 5
cm from the connector with slack. The fix is a new cable.

## Rules in force

- Every robot run needs the owner's go in the lab's session.
- Freedrive (`teach_hold.py`) never ends or restarts on its own while the owner may hold the arm: extend it (touch
  `logs/ur5_lab_teach_extend`) or stop it on the owner's word, and ask before switching it either way. On 11-02 at
  13:21:06 its 15 min cap ended it under the owner's hands and the robot e-stopped (nothing damaged).
- Joint 6 goes to +45 deg at most without the owner, and to +60 deg with the owner present.
- No gravity sweeps.
- travel.speed is 1.0. The tall plans lift 20 mm between actions.
- Before it runs, every plan is path-checked on the current registrations and `cell.json` with
  `scripts/ur5_tip_calibration_holder_paths.py` (margins: robot 8 mm, mount 3.0 mm or 2.5 mm for touch-offs, finger
  0.8 mm, HEX-E 6 mm), and the lab simulates it.
- No force control, contact motion or safety threshold is set from a borrowed calibration that was not validated.
- The workstation reads corvus over SSH and never writes there; plans reach corvus by git.

## v3 ID 3's suite: done at 14:49 on 10-30

The workstation built the suite at 12:00 on 10-29 (62d93424, d6795072). The owner's go came in the lab's session at
13:4x for block 1 ("lets a go mario"), and at 13:5x for the rest: "1. top edge -- lets go for it. add more density if
anything 2. side map -- sure 3. yea send it".

| Step | Plan | Actions | State |
|---|---|---|---|
| block 1 | `20261029-plan_sphere_lab_pass1_night_v3id3.json` | 18 | Done 13:45 to 13:59: 18 of 18, 0.044 mm rms (`20261029-134508-spherev3id3`). |
| top edge | `20261029-plan_sphere_lab_top_edge_v3id3.json` | 26 | Done 14:50 to 15:11: 26 of 26. 24 presses (ID 2 had 10) and two references. |
| side map | `20261029-plan_holder_lab_side_up_v3id3.json` | 24 | Done 15:13 to 15:29: 24 of 24 (`20261029-151334-sidev3id3`). |
| block 3 | `20261029-plan_tall_lab_pass1_night_v3id3.json` | 19 | Done 15:30 to 15:48: 19 of 19, 0.055 mm rms (`20261029-153034-tallv3id3`). |
| block 2 | `20261029-plan_sphere_lab_stage2_night_v3id3.json` | 211 | Done 16:34 to 23:07: 211 of 211, no abort, no early contact, 6.55 h (resumed on block 1's folder). |
| block 5 | `20261029-plan_rod_lab_pass1_night_v3id3.json` | 26 | Done 23:08 to 23:31: 21 of 21 touch-offs and 5 pivot presses, 0.037 mm rms (`20261029-230820-rodv3id3`, a seed of block 1's tool and 031559's rod). |
| block 4 | `20261029-plan_tall_lab_stage2_night_v3id3.json`; its last 55 in the order of `20261030-plan_tall_lab_stage2_night_wrist2_v3id3.json` | 101 | Done 23:33 on 10-29 to 13:19:51 on 10-30: 101 of 101 ok, no early contact. HEX-E losses stopped it at 00:08:00 (46 ok) and at 11:03:04, on the night order's first travel (home into tall-edge-626). After the owner's reseat with more cable slack (11:09), the last 55 ran in wrist2's order from 11:15:24, with no abort and no HEX-E time-out. |
| block 6 | `20261029-plan_rod_lab_stage2_night_v3id3.json` | 100 | Done 13:20 to 14:49 on 10-30: 100 of 100 ok, no abort, no early contact, no HEX-E time-out (`logs/20261030-132039-tipcal-rodv3id3.log`, resumed on block 5's folder). |

- **ID 3's suite is done.** The owner said no calibration runs tonight. The next runs use the inverted rig on the
  low seat strip (ac021c3dd), with the probe mode (below). Everything comes off the table after this suite. The arm stands at rod-final-ref's rest over the rod (joints 32.9, -61.1, 97.5, -33.1, 36.4, -6.8 deg).
- **The probe mode is in the program and the simulator, with touch-offs on the tip's side walls** (NOTES, "Probe mode
  for the inverted rig: written and simulated" and "Probe mode: the tip's side walls").
  - `scripts/ur5_probe_plan.py PLAN --pass1-standoff 3.5 --walls WALLS --walls-check CHECK` turns the workstation's
    seat plans into program plans. The simulated ones are here: `20261030-plan_probe_seat1_walls_v3id1.json`,
    `20261030-plan_probe_seat2_walls_v3id3.json` and `20261030-plan_probe_seat3_walls_v3id2.json` (the owner's
    seats and mux channels 3, 1 and 0).
  - A session runs `--stages preflight,homing,pad,register,wall,register,grid`, with the mount seeded from a tip-mode
    registration.
  - The workstation checked pass 1 at 3.5 mm and the walls at z_F -99.75 (8ee8556e): 25 moves per seat, none flagged.
  - Simulated whole on all three seats: every action done, no abort. The walls find the seat's turn about the tip's
    long axis, which the pad's touches barely see. Presses then stand 0.02 to 0.03 mm rms off along the normal and
    0.05 to 0.07 mm across, where seat 1 without walls stood 0.165 mm across.
  - The tip mode is unchanged: HEAD and the new code print the same simulated session line for line.
- **The HEX-E's losses on 10-30** came at 00:08 and 11:03 (NOTES), both on arriving at tall-edge-626's pose, where all
  five tall losses came. The sensor came back by itself at 01:37 and held at that pose all morning. The owner reseated
  the connector with more cable slack at 11:09. Since then the arm has reached that pose twice (11:18:58 and
  12:42:24), and block 4 ran its last 55 with no time-out.
- The sphere check of 11:11 (`20261030-111133-chkv3id3`) put the three touch-offs within 0.08 mm of block 1's at the
  same nodes (scratchpad `chk_change.py`), so the reseat did not move the tool.
- The rest ran through scratchpad `resume_id3.sh`: the check, block 4's 55, then block 6, each gated as before.
- Each block starts only after the run before it exits 0 with no ABORT line, and after the simulations that gate it
  pass. Block 2's 4 chunks run on block 1's registration, block 3 on seat 1, and block 5 on its seed. Blocks 4 and 6
  run on their pass's registration, checked on every 5th sample. Then come the home path check and go home. If a step
  fails, the chain stops and the arm waits where the last run left it.
- The 58 touch-offs of blocks 1, 3 and 5 approach at v 0.0005, which is 0.15 mm/s at the slider.
- rod-ref-a stays in block 6 unless block 6's simulation puts the mount under 3.0 mm on it.

## Last night: v3 ID 2's night suite, done

Built by `scripts/ur5_tip_calibration_next_tip_suite.py` and path-checked on v3 ID 1's registrations. About 11.7 h.
Done at 08:13:17: blocks 1 to 6 and the sphere's top edge, 488 actions, all ok, with no abort, no early contact and no HEX-E time-out. They took 12.6 h of robot time; block 2 ran 0.5 h over its estimate. The blocks ran in the order 3, 2, 5, 4, 6 (`NOTES.md`, from "Block 1"). The owner's go came in the lab's session at 18:3x; the owner left at about 19:25 ("worst case i can come back at ~9:30 to reseat").

| Block | Plan | Actions | Hours | How it runs |
|---|---|---|---|---|
| 1 | `20261028-plan_sphere_lab_pass1_night_v3id2.json` | 18 | 0.5 | A seed folder from `scripts/ur5_tip_calibration_seed_registration.py`, with --tool and --fixture both `20261027-225213-spherev3id1/registration.json` (homing pose kept). Stages preflight,pad,register; no homing. |
| 2 | `20261028-plan_sphere_lab_stage2_night_v3id2.json` | 211 | 6.1 | Resumed on block 1's registration. |
| 3 | `20261028-plan_tall_lab_pass1_night_v3id2.json` | 19 | 0.5 | A new session on block 1's tool, as 114908 ran on 225213's. |
| 4 | `20261028-plan_tall_lab_stage2_night_v3id2.json` | 102 | 2.8 | Resumed on block 3's registration. |
| 5 | `20261028-plan_rod_lab_pass1_night_v3id2.json` | 26 | 0.4 | A seed folder with --tool block 1's registration and --fixture `20261028-102701-rodv3id1/registration.json`. Stages preflight,pad,pivot,register. |
| 6 | `20261028-plan_rod_lab_stage2_night_v3id2.json` | 100 | 1.4 | Resumed on block 5's registration. |
| top edge | `20261028-plan_sphere_lab_top_edge_v3id2.json` | 12 | 0.3 | After block 1, with the owner present; resumed on block 1's folder, stage grid. Ten presses on the tip's end and edges and a reference press before and after, built by `scripts/ur5_tip_calibration_sphere_top_edge.py` at the sphere's 60 mm lift. At a 20 mm lift, the joint move into the first top press swept the fork 3.9 mm into the sphere's model. Presses with the cap under 0.8 mm from the sphere are left out. |

After a HEX-E reseat, `20261028-plan_sphere_lab_check2_v3id1.json` repeats three sphere touch-offs; their onsets show
whether the tool moved.

## v3 ID 2: runs done

| Runs | Session folder | Time | Result |
|---|---|---|---|
| Sphere pass 1 (block 1), top edge, stage 2 (block 2) | `20261028-184415-spherev3id2` | 10-28 18:44 to 19:09; 19:21 to 19:30; 20:40 to 10-29 03:15 | 0.044 mm rms; 12 of 12; 211 of 211 |
| Tall pass 1 (block 3), stage 2 (block 4) | `20261028-200508-tallv3id2` | 20:05 to 20:39; 10-29 03:59 to 06:45 | 0.073 mm rms; 19 of 19; 102 of 102 |
| Rod pass 1 (block 5), stage 2 (block 6) | `20261029-031559-rodv3id2` | 10-29 03:16 to 03:57; 06:46 to 08:13 | 0.023 mm rms; 21 of 21 and 5 pivots; 100 of 100 |

## v3 ID 1: runs done

| Runs | Session folder | Time | Result |
|---|---|---|---|
| Holder touch-offs, apex search, holder map, close map | `20261027-200536-holdv3id1`, `-202426-holdapexv3id1`, `-212131-holdmapv3id1`, `-222640-holdclosev3id1` | 10-27, 20:05 to 22:4x | the held apexes, `20261027-holder-apex-both` |
| Sphere pass 1, stage 2, twists | `20261027-225213-spherev3id1` | 10-27 22:52 to 10-28 02:54 (HEX-E loss); 09:36 to 10:25 | 0.049 mm rms; 160 grid presses and 40 slides; 12 twists, 2 rolls |
| Rod pass 1, stage 2, morning6 | `20261028-102701-rodv3id1` | 10:27 to 13:52 | 0.042 mm rms; 94 of 94; 6 of 6 |
| Tall pass 1, stage 2, morning plan | `20261028-114908-tallv3id1` | 11:49 to 17:41 | 0.045 mm rms; 102 (44, then 58 after the 14:32 loss); 18 of 18 |

## v3 ID 1: not run

These need v3 ID 1 back on the robot.

| Plan | Actions | Needs |
|---|---|---|
| `20261028-plan_tall_lab_top_edge_ts1_v3id1.json` | 34 | the owner present; its first 8 run from 18:03:40 with the owner watching (`20261028-plan_tall_lab_top_edge_first8_v3id1.json`), so that the next tip can run it too |
| `20261028-plan_sphere_lab_morning_unattended_ts1_v3id1.json` | 43 | the owner present (joint 6 to +59 deg) |
| `20261028-plan_sphere_lab_stage2_resume3_v3id1.json` | 43 slides | |

The side map without the rod (`20261028-plan_holder_lab_side_norod30_v3id1.json`) is off, because the rod stays on.

## Answered by the owner on 10-29

The owner answered at about 11:37, in the workstation's session, which passed the answers on at 11:49.

- Touch-offs: "we can approach at 0.15mm/s sure, or faster if you want". The next tip's plans run the touch-offs' slow
  approach at v 0.0005 (0.15 mm/s at the slider), as the presses run. At that speed the onset moved 0.000 mm (SE
  0.006) in block 2's presses, and a touch-off takes about 50 s instead of 81 to 108 s (`NOTES.md`, "Faster
  touch-offs"). The workstation advised against going faster: at 0.58 mm/s the onsets came 0.026 mm early (SE 0.017),
  and that saves only 10 to 17 s more per touch-off.
- The side map: "i am not going to move the rod. figure something else out." The workstation built one with both held
  tips pad-up ("Ready" above). The robot's pad faces down beside the -y held tip, with its apex 17 mm under the held
  apex. That puts the two magnet squares level and 15 to 25 mm apart, plus a far pose at 40 mm. The robot stands
  about 25 mm higher than in the old poses, and it clears the rod.
- Indenter plate v2d waits: "im not sure i plan to put on the indenter plate v2d until we have 100% finalized all the
  motions we need for effective calibration". The workstation is writing the list of motions calibration needs. For
  each it gives the fixture and what the motion adds to the held-out errors on ID 1 and ID 2.

## Waiting on the owner

- The owner's go of 15:0x on 11-02 covers the remaining runs ("you have my go ahead to run the remaining runs"). More
  runs on twists and wrenches, and at other contact speeds, are covered too, if the data shows that more data still
  pays.
- Done: seat 1 (v3 ID 6), 14:50 to 18:27, pass 1 20 of 20 and the grid 294 of 294 with no abort (`20261102-145010-s1v6b/`).
- Done: seat 2 (v3 ID 5), 18:27 to 22:03, pass 1 20 of 20 (0.051 mm rms) and the grid 294 of 294 with no abort
  (`20261102-182710-s2v5/`). Its pad stands 1.21 mm and 3.5 deg from its CAD, much as seat 1's (1.68 mm, 4.1 deg).
- Done: seat 3 (v3 ID 4), 22:03 to 01:43 on 11-03, pass 1 20 of 20 (0.051 mm rms) and the grid 295 of 295 with no abort
  (`20261102-220337-s3v4/`). All three seats' main runs: 883 grid actions and 60 touch-offs, not one aborted.
- Done overnight: speed3, speed2, speed1 (01:43 to 02:47), 48 of 48 each with no abort.
- Done overnight: night1 on seat 1 (02:47 to 04:41), 232 of 232 with no abort.
- Done overnight: night2 on seat 2 (04:41 to 06:36), 231 of 231 with no abort.
- Done overnight: night3 on seat 3 (06:36 to 08:31), 232 of 232 with no abort. So the three seats' main runs, speed
  sets and night obliques all ran without one abort: 60 touch-offs, 883 grid actions, 144 speed presses and 695
  night presses.
- Running since 08:31: fast3, the fallback (the grid's 263 presses at 0.6 mm/s) on seat 3's session, to about 10:20,
  printout `logs/20261103-083111-tipcal-fast3.log`; then fast2 and fast1 (to about 14:00). To stop after the run in
  progress: kill the chain's PID 2658916. To stop now: Ctrl-C in tmux `tipcal` stops the robot (stopL) and closes the
  files.
- Armed (16:10): `logs/20261102-chain_seats23.sh` (copy here, PID 2658916, its log `logs/20261102-chain_seats23.log`)
  started seat 2 (v3 ID 5) at 18:27, to about 22:05; then seat 3 (v3 ID 4), about 22:05 to 01:45
  (`20261102-plan_probe_table_seat2_v3id5.json`, `..._seat3_v3id4.json`: safe_mm 15, hubs 40 mm over the apex). Both
  passed the chained simulations and the workstation's replays. A run stopped by a HEX-E loss is resumed once
  `hexe_probe.py` finds the sensor streaming; anything else stops the chain. To stop the chain: kill its PID (a running
  program goes on).
- Overnight, after seat 3 (owner's go, if more data pays; the learning curves say about 13 to 17 % less direction error
  per doubling): `logs/20261102-overnight.txt` (copy here), each run resuming its seat's session (stages
  preflight,grid) and only with its marker `logs/20261102-ok-NAME`; the list stops at the first run without one. In
  order, each seam joining neighbouring seats:
  - speed3, speed2, speed1 (`..._speed_seat*`: 12 points a seat at 0.05, 0.15, 0.6 and 1.2 mm/s), about 0.4 h each;
  - night1, night2, night3 (`..._night_seat*`: the workstation's night subset, 231 to 232 flank, face and edge
    obliques at 10/20/30 deg with levels from 0.25 N; ids end in -n), about 2 h each;
  - fast3, fast2, fast1 (`..._fast_seat*`: the grid's 264 presses at 0.6 mm/s), about 1.8 h each, into the morning.
  The workstation replayed the whole list chained from seat 3's end (0 of 1,521 travels flagged; three night presses
  capped at 0.75 mm, 83150a9a). No twists: torsion has no skill on v3 IDs 2 and 3.
- Owner, 14:4x: the travels go too high ("you don't need to go this high at all"). Seats 2 and 3 and the overnight
  plans back off 15 mm (was 45) and go by a hub 40 mm over the apex (was 90): the sphere stays 46 to 80 mm over the
  strip between seats. The owner also asked for one shared set of pose helpers (quaternions) in place of the copies in
  each script: later, not while runs depend on the program.
- The program returns a twist st + 0.5 mm from its press's end, which leaves the sphere on the pad when the press ran to
  its full depth (the workstation, 15:3x): fix later (return to 0.5 mm behind the rest); tonight's extra runs have no
  twists.
- Joint 6 near 0 in the press configuration (owner, 13:5x).
- A new HEX-E cable.

## Where things are

- History: `../QUEUE.md` (R31 covers v3 ID 1) and `NOTES.md` (the lab's log).
- Results: `docs/experiments/20261027-v3_tip_calibration_and_crosstalk/` (README, `data/`, `feed.jsonl`) and the
  fingertip calibration page.
- Plan files that are no longer current, and their path checks: `superseded/`, under the same names. A plan that ran
  also has a copy in its session folder.
