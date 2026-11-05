# UR5 cell: current state

Updated 2026-11-04 20:4x by the lab. **The calibration is done** (the owner, 16:4x on 11-04; below, under "Waiting on the owner"). **The SR2 hand hangs on the UR5, and the arm is powered off** (18:57:27; the brakes hold it). The crosstalk sweep of `../../20261104-sr2_hand_crosstalk/` ran from 18:57 to 20:34. It stopped with the hand at zero and its motors disabled, because the index's fingertip board stopped answering at 19:43:52, probably its cable pulled by a curl. **For the owner:** check the index's fingertip cable at both ends before anything moves. From the tip swap on, the lab session updates this file at every session
start and end. This file gives the present state of the cell. `../QUEUE.md` and `NOTES.md` are the history. Every
request to the owner must come from this file.

## On the robot

| Item | Now | Registration (folder here) |
|---|---|---|
| SR2 hand | on the flange since 18:3x on 11-04 (owner). The arm was powered off from the pendant at 18:57:27; as the brakes engaged the TCP moved 0.2 mm. Its fingertips: v3 ID 1 on the thumb (mux channel 3), v3 ID 3 on the index (channel 1) and v3 ID 2 on the middle finger (channel 0), on the C6's mux (0x71) plugged into corvus. The lab has not checked the probe base or the strip since | `../../20261104-sr2_hand_crosstalk/README.md` |
| Probe | probe base with the rod tool (R 12.5 mm, its axis along y_M) on mount v7's adapter disc on the HEX-E (owner, 11-03 16:5x; set right at 17:2x after it sat a quarter turn off its square boss); before it the 2 mm sphere from 11-03 10:5x and the 5 mm sphere from 11-02 13:0x; no tip on the robot | the workstation's joint fit on the night's 0.5 N sweeps (bb18d7c65): c_M (-0.988, 0, 106.713) mm in M, turned +0.43 deg about z_M (sd 1.04) and -0.89 deg across (sd 0.50); each rod session's pass 1 fits it too |
| Strip | the seat strip on one 2x4 clamped to the table (owner, 11-02); 192 mm from the place the 10-30 plans assume | tonight's sessions: the frame through both keys' socket points (`20261102-strip_frame_two_keys.json`), by `--frame`, with `--home-from poses.jsonl:strip_key1_seated_4`; each seat's grid aims by its own pass 1 fit. For later plans: the workstation's frame from the three pads' fits (`../../20261030-inverted_rig_seat_block/data/20261102-strip_frame_pads.json`, 9cd8b375: the two-key frame turned 3.44 deg about y_A and moved (2.48, 0.10, -3.70) mm; the pads within 0.14 to 0.22 mm on average, the rest the strip's twist) |
| Held tips | **coming off for the SR2 hand (the owner, 16:2x on 11-04).** Until then: seat 1 (y_A -80, the right end seen from the robot) v3 ID 3 on mux channel 3 since 14:3x on 11-04 (the owner; v3 ID 6 sat there from 11-02 13:0x); seat 2 v3 ID 5 on channel 2; seat 3 (left) v3 ID 4 on channel 0 (owner, 11-02 13:0x); all three stream at 151 records/s (14:40 on 11-04; 138 on 11-02) | seat 1's ID 3: `20261104-tip_calibration/145215-id3pts11_seed` (pass 1, 0.060 mm rms; its pad 0.71 mm and 1.55 deg from ID 6's). Seat 1's ID 6: `20261102-tip_calibration/145010-s1v6b` (pass 1, 0.055 mm rms; its pad 1.68 mm and 4.1 deg from the CAD in the two-key frame); seat 2: `182710-s2v5` (0.051 mm rms; 1.21 mm, 3.5 deg); seat 3: `220337-s3v4` (0.051 mm rms; 0.81 mm, 2.9 deg). The rod's sessions of 11-03 have lab folders `20261103-214753-rod13`, `20261103-172647-rod21` and `20261103-202017-rod32` (seats 1, 2 and 3) |
| Old fixtures | the low sphere, rod, tall indenter and holder: off the table as the strip's README asks; the camera frame of 13:05 shows only the strip, its tips and the mux board on the 2x4 | as before 10-30 |

`cell.json` holds the same registrations for the path checks and the renders.

The HEX-E's link fails about once per 6 to 9 h of running: 7 losses since 10-25 (`../QUEUE.md` R31, 17:42 on 10-28).
An eighth came on 11-02 between 13:03 and 13:20, while the owner moved the arm by hand to the press pose (joint 6 to
+178 deg); by 13:27 it streamed again after a restart, untared (Fz 629 N, status 0x40000), as after the loss of 10-22.
A loss aborts three actions and stops the session, and the robot then waits for a reseat. The cable is clamped 3 to 5
cm from the connector with slack. The fix is a new cable.

## Rules in force

- **Never freedrive or teach mode while the SR2 hand is on the UR5** (owner, 18:3x on 11-04): the arm drops and crashes under the hand's weight.
- Every robot run needs the owner's go. Since 15:4x on 11-03 ("relays count yes"), a go that the workstation relays
  with the owner's words and time counts the same as one typed in the lab's session.
- Simulate new or changed program code. A plan that changes only numbers on paths already checked (speeds, or depth
  caps the workstation's path check covers) runs on that path check alone (owner, 14:2x on 11-03).
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
- Done: fast3 on seat 3 (08:31 to 10:07), 252 presses with no abort. Its other 11 ids (`-v060`) were speed3's on the
  same session, so `--resume` skipped them; speed3 has those 11 at the same speed.
- Stopped: fast2 on seat 2 at 10:16:40 after 25 of 264 (SIGINT by PID between presses, the program's Ctrl-C), with
  fast1, because the owner (relayed by the workstation, 10:1x) means to put the 2 mm sphere on after fast3. The chain
  then stopped itself.
- The owner put the 2 mm sphere on and gave the go at 10:5x in the lab's session: "ok the 2mm radius sphere is on
  there. the arm is oevr the seats. you can go ahead". The arm stood over seat 2, moved a little by hand (joints 54.2,
  -79.1, 105.9, -114.1, -86.6, -2.0 deg). From those joints, the workstation's replay code (dry-check --real, its start
  carried in by a scratch wrapper) flags nothing in seat 1's first travel by the hub. Its least clearances are those of
  the final descent into seat 1, as in the workstation's replay from the hub.
- **Stopped at 11:25:04: seat 1's 2 mm sphere run (s1sph2) and its chain**, after a bad pass 1 fit (NOTES, "s1sph2: a
  touch on the tip's body and a bad fit").
  - Now the arm stands over seat 1 (joints 38.3, -70.1, 97.3, -103.1, -67.0, -7.6 deg) with the tool clear of the pad
    (the HEX-E at its rest reading), sphere2 on, and no program running.
  - pad-tip-349-o35-a270 met the tip's body 4.7 mm before the pad. The replay gave its own moves 0.76 mm to the body,
    and the pad stood about 0.7 mm nearer than the frame put it. register took it in (0.758 mm rms); the 19 others fit
    to 0.051 mm.
  - On that fit the grid ran 23 actions: 10 contacts before the slow approach, 8 presses with no level, and two travels
    aborted at 8.8 and 8.4 N on v3 ID 6.
  - **The owner should look at v3 ID 6 (seat 1) before it is pressed again.**
  - The fix is in the program (ada1ee4b). register leaves out a touch over 0.5 mm off the pad (at most 3) and stops
    before the grid over 0.15 mm rms. A run stops at its 3rd early contact or its 3rd press with no level.
  - The plans: seats 1 and 2 less pad-tip-349-o35-a270 (`20261103-plan_probe_table_sphere2_seat{1,2}_v3id{6,5}_no349.json`,
    df6d5b4f), seat 3 as before.
  - In the simulator on the fixed program: 260, 269 and 271 of 260, 269 and 271, with no early contact, miss or abort.
    Seat 3 prints line for line as before the fix.
  - Replayed from the robot's joints (the lab, and the workstation with `--start-q`): none of 858 travels flagged, and
    only this morning's three own-press flags.
  - The owner, 12:1x in the lab's session: "currently, the 2mm radius sphere is not damaged or contacting anything. it
    is almost grazing the flank of the the fingertip. you have the goahead to contnue, although ive asked the lab
    session to see if we can speed up these fingertip calibrations on these passes". Under s1sph2's good 19-touch fit
    the sphere stood 0.6 mm from the pad; backing along the stem opens the gap steadily.
  - **Now (from 12:20:20): `logs/20261103-chain_sphere2b.sh`** (copy here; PID 2823533; its log
    `logs/20261103-chain_sphere2b.log`). Seat 1 runs as s1sph2b, a new session on the _no349 plan, then seats 2 and 3.
    Seat 1 runs to about 14:40.
  - The speed-up, for seats 2 and 3 and the rod: quick plans (`..._quick.json`) with travel.probe.retract_mm 6 (was
    12) and travel.probe.in_v 0.02 (the last move in at 6 mm/s, was 3). They save about 4 s a press, about 20 min a
    seat. They are in the simulator and the workstation's --retract 6 replay. The presses are unchanged, and so is the
    return to the rest (0.6 mm/s): its rest window t_rest1 sets the analysis's zeros.
  - Both passed. The workstation's replays at --retract 6 flagged none of 858 and 587 travels (6e53dc57). In the
    simulator the quick plans of seats 2 and 3 ran 269 of 269 and 271 of 271, with no early contact, miss or abort. The
    program reads in_v and back_v from 6d0a05ef.
  - From 12:58:17, `logs/20261103-chain_sphere2c.sh` waited to run seats 2 and 3 on the quick plans. It was stopped by
    its PID at 13:52 for the owner's go on the deeper flank presses.
  - The owner, 13:5x in the lab's session: "yes please go", to the deeper flank presses (the workstation's neck rule,
    840f40da). It was relayed by the workstation as "we can go on it. will we be done by 6?".
  - From 13:52:41, `logs/20261103-chain_sphere2d.sh` waited to run seats 2 and 3 on `..._quick_neck.json`, with 170 and
    169 presses deeper (sims l_sph2_2 and l_sph2_3 clean). It was stopped by its PID at 14:17.
  - The owner, 13:5x in the lab's session: "Yes go on the faster presses too".
  - **Done: seat 1 (s1sph2b), 12:20 to 14:48:02.** Pass 1 19 of 19 (0.060 mm rms), the grid 260 of 260, no early
    contact, miss or abort (`20261103-122020-s1sph2b/`). 40 presses stopped at their depth below their top level.
  - **Done: seat 2 (s2sph2, v3 ID 5) on `..._seat2_v3id5_quick_neck_fast.json`, 14:48:07 to 16:11:13.** Pass 1 19 of
    19 (0.053 mm rms), the grid 269 of 269, no early contact, miss or abort; 29 presses stopped short of their top level.
    The HEX-E gave 497,626 samples with none timed out. Copy: `20261103-144808-s2sph2/`.
    - **The speed slider went from 0.30 to about 0.50 between 15:42:11 and 15:42:25**, from probe-edge-639-o20-a0 on,
      unseen until 16:2x. The 1 mm fast approach took 0.85 s, not 1.42 s, and the press about 1.0 mm/s, not 0.6.
    - So the last 146 presses stopped about 0.45 mm deeper at each level (the reference press reached its 0.5 N hold at
      2.60 mm, against 2.15), and the 3 N presses peaked at 5 to 7.8 N, against 3.7 to 4.6 N before.
    - The force against travel at the reference point did not change, and the sweep test's touches found the pad within
      0.07 mm of 14:48: the tip and the sphere did not move. No program sets the slider mid-run, so it was likely moved
      at the pendant (the owner asked). The sweep test's start set it back to 0.30 at 16:11.
    - The program now reads the slider before every action and stops if it is off the plan's (logs/sweepwork, in the
      simulator; it goes in before the next robot run).
  - **The sweep test on seat 2 (s2sweep), 16:11:15 to 16:17:29: stopped at the owner's word** ("if you pause now i can
    check the seating"), after 10 of its 19 pass 1 touches. chain_sweep2b was stopped by its PID first, then the program
    by SIGINT to its PID, right after a touch's return to its rest. The arm stands over seat 2: joints (55.4, -80.1,
    109.2, -121.5, -87.9, 0.2) deg, TCP (-198.0, -487.0, 238.4) mm, the HEX-E at its rest reading. Nothing runs.
    - The owner, 16:1x (relayed): "ok i just checked and confirmed that all were seated. it could be the deflectin of
      the small spehre indenter. also, this is probably going to be the last calibration we do for a while, we're
      mounting the sr2 hand after this. so thsi rod experiment better be good."
    - The owner, 16:2x (relayed): "to be clear we can still do the rod experiments, just tmw the hand is going on. so i
      pray to god you get as good information as we did ont he v3 ids 1-3 fingertips + the better coverage + angles i
      was hoping for with the inverted rig".
    - The lab recommends the rod now, without the 2 mm sphere's sweep test: the rod's first sweeps test the code.
  - **The rod is on** (the owner, 16:5x in the lab's session: "ok it's on there godspeed go"). The arm moved in the swap:
    joints (49.72, -83.46, 116.43, -141.09, -95.65, 0.13) deg, TCP (-240.3, -442.0, 249.3) mm. The lab's replay of
    seat 2's first travel and pass 1 from there with the workstation's checker flagged 0 of 20.
  - 16:58:23: the night's chain started seat 2. Its pass 1 met the pad from -3.29 to +1.33 mm of the expected, and the
    rod's fit refused it (0.229 mm rms, 2 left out). The session stopped before the grid at 17:15:54, as designed.
    The rod tool sat a quarter turn off on its square boss: the workstation's CAD model, the lab's forward model and
    the camera agree (`20261103-165823-rod21-turned/`).
  - 17:2x, the owner: "ok it's rotated please go". At 17:26:47 the chain started again from a fresh seed of s2sph2.
    - Pass 1: 20 touches at +0.13 to +1.33 mm. The fit is 0.054 mm rms, none left out; the pad is 0.65 mm and
      0.87 deg from the seed. The rod's c_M is at (-0.838, 0, 106.687) in M, turned -5.40 deg about z_M (sd 2.5) and
      +0.77 deg across (`20261103-172647-rod21/`).
    - The grid started at 17:46. Grids 1 to 3 ran to 8 N, the first sweeps on the robot. The friction check skipped
      some offsets and slides.
    - With the fitted turn, the workstation's exact margins put a few slides within 0.2 mm of a tip body. The lab
      stopped the chain (17:55:11) and the program after grid 3 (17:55:46), and took them out of all three seats.
      - Seat 2: the az 90 slides of edge-569-r0, the az 0 slides of edge-618-r90, and edge-642-r90 whole.
      - Seat 3: edge-642-r90; the az 0 slides of edge-618-r90; the az 270 slides of edge-573-r0 and edge-651-r90.
      - Seat 1: edge-651-r90 and edge-642-r90; the az 90 slides of edge-569-r0; the az 0 slides of edge-618-r90.
      - The plans are `20261103-plan_probe_table_rod_seat{2,3,1}_v3id{5,4,6}_sweep8_fit.json`.
  - **v3 ID 3 in seat 1 (14:3x on 11-04): tip or rig.** The owner put ID 3 where ID 6 was; the rod stays on the
    arm. Their words: "i placed v3 id 3 where v3 id6 is. let's settle this quickly" (relayed by the workstation) and
    "v3 id3 is now mounted where v3 id6 was" (in the lab's session).
    - The question: IDs 4 to 6 on the inverted rig place near-normal loads to 0.48 to 0.67 mm, IDs 1 to 3 in tip
      mode to 0.39 to 0.40. ID 3 on the inverted rig tells whether the tips or the rig explain it.
    - The workstation's plan, being built: a new session on seat 1 with its own pass 1 (20 touches, as the night's),
      8 reference presses back to back, then one normal rod press at each face and edge point 1.0 mm apart (about
      105, 0.5 to 3 N), a reference every 25 and last. About 50 min.
    - Checked at 14:40, nothing running: ID 3's board streams on channel 3, and its rest field is ID 3's (NOTES).
      The arm is not at the 10:45 pose: the owner moved it by hand, 11.2 mm up. Joints (49.05, -78.46, 106.71,
      -124.79, -91.31, -38.05) deg, flange (-267.8, -474.1, 248.6) mm. Slider 0.30, safety normal.
    - **Done, 16:13:01: id3pts11, every action clean.** Pass 1 landed 20 of 20 (0.060 mm rms; ID 3's pad 0.71 mm
      and 1.55 deg from ID 6's). The grid ran 125 of 125 to 3 N in 61.2 min with no skip, abort, early contact or
      missed level. The HEX-E gave 483,589 samples, none timed out. The plan was the workstation's of 2d9820f3f
      (`../../20261030-inverted_rig_seat_block/data/20261104-plan_probe_table_rod_seat1_v3id3_swap_draft.json`), and
      both dry-checks from the arm's joints flagged none of 145 (`20261104-id3_swap_run_check.json`).
    - The session, whole: `20261104-145215-id3pts11/` (live: `logs/20261104-tip_calibration/145215-id3pts11_seed`).
    - The arm waits at seat 1's reference rest, the rod a few mm over ID 3's pad: joints (49.31, -76.80, 105.76,
      -123.11, -91.97, -36.55) deg. Move it clear before mounting the SR2 hand. Nothing runs on the robot. The chain
      ended, the 17:00 stop was stopped by its PID, and the time-lapse stopped at 16:13:15.
    - **ID 3's result (workstation, fc09a648c): the gap is the rig's, not the tips'.** The same tip places near-normal
      contact to 0.592 mm on the inverted rig and 0.319 mm in tip mode. Its board's uT/N at the reference presses
      varies 7.9 % here against 2.2 to 2.5 % in tip mode, while the force varies alike. The workstation suspects the
      board's clamping in the seat block (NOTES).
    - **16:2x, relayed: the owner is taking the tips off for the SR2 hand.** They will move the arm clear by hand first.
      Nothing more is planned on the robot today.
  - **The calibration is done (the owner, 16:4x on 11-04): "if the inverted rig is marginally worse, then so be it. it
    seems like it is marginally more reliable, hex-e wise. but we are done calibrating now".** The arm was moved
    clear by hand (16:44: joints 43.45, -97.96, 105.78, -123.23, -91.95, -36.55 deg; flange 422.6 mm up). The rod
    stays on the probe base, and the tips go to the SR2 hand. Nothing runs on the robot except the read-only
    `robot_messages.py` (PID 2632536).
    - The likely cause of the rig's gap, from the owner taking the tips off: "actually i do think the self-threading
      holes stripped after repeated screws". The M2 clamp the board in the seat block by threads cut in the print, so
      stripped holes leave it loose. That fits the board's 7.9 % uT/N scatter (NOTES). A later seat block would want
      heat-set inserts or nuts.
  - **Done, 13:16:58 on 11-04: the rod points, every step clean: 232 of 232 actions, no skip, abort or early contact.**
    The arm waited at seat 1's reference rest, the pose of 10:45 (joints 48.99, -77.26, 106.61, -124.34, -91.36, -38.05
    deg), the rod a few mm over seat 1's pad. Nothing runs on the robot: the chain ended, and the 14:15 stop (PID
    2960534) was stopped by its PID at 13:17:14.
    - The run: `logs/20261104-chain_rod_points.sh` on `logs/20261104-rod_points_steps.txt` (copies here), 11:17:06 to
      13:16:58; the robot had to be free at 14:15 (the owner chose it at 10:55, relayed).
    - Steps: face and edges on seats 1, 3 and 2 (75, 70 and 70 grid actions), then flanks and tip on seats 2, 3 and 1
      (6, 5 and 6), each resuming its seat's rod session. Plans: the workstation's
      `../../20261030-inverted_rig_seat_block/data/20261104-plan_probe_table_rod_seat{1,3,2}_v3id{6,4,5}_dense_{fe,ft}_draft.json`
      (b73de9a1d).
    - Checked: no id ended ok in its session before; the lab's chained dry-check from the arm's joints flags 0 of the
      232 (`20261104-rod_points_run_check.json`), as the workstation's; at the joint fit's 2 sd corners every action
      keeps 0.2 mm from the tip body (the workstation's).
    - Done: seat 1's face and edges (pts11), 11:17:06 to 11:54:46, 75 of 75 to 3 N, no skip or abort. From 11:54:51,
      seat 3's (pts32).
    - Done: seat 3's face and edges (pts32), 11:54:51 to 12:29:23, 70 of 70 to 3 N, no skip or abort. From 12:29:27,
      seat 2's (pts23).
    - Done: seat 2's face and edges (pts23), 12:29:27 to 13:04:04, 70 of 70 to 3 N, no skip or abort. All three seats'
      face and edges are done, 215 of 215.
    - Done: seat 2's flanks and tip (pts24), 13:04:12 to 13:08:37, 6 of 6; seat 3's (pts35), 13:08:37 to 13:12:18, 5
      of 5; seat 1's (pts16), 13:12:23 to 13:16:53, 6 of 6. The flanks and tip reached their 2 N top, the reference
      presses 3 N.
    - The HEX-E gave 712,913 samples over the six runs, none timed out; each run's two gaps came 1 to 2 s before its
      preflight. Each session's new bytes are with the workstation by git (`20261104-rod{13,32,21}_tail/`,
      `20261104-rod{13,32}_ft_tail/`).
  - **The go for the rod points (11-04, 10:44).** The owner, relayed at 10:44: "if we
    just lost density of coverage why dont we just add points now. i know i said we were done earlier, but we should
    have a few hours left to get that density". The held-out GP places contact to 0.78 to 0.82 mm on IDs 4 to 6,
    against 0.46 to 0.51 on IDs 1 to 3; the night's points stand 2.0 mm apart, IDs 1 to 3's 1.0 mm.
    - The workstation's plans: one normal press per new point (0.5, 1, 2 and 3 N), reference presses first, every 25
      actions and last. Per seat, `_dense_fe` (face and edges at 1.0 mm, about 75 presses, about 35 min) and then
      `_dense_ft` (flanks and tip at 1.3 mm) if time remains. Order 1, 3, 2, each resuming its rod session.
    - The cell at 10:45, unchanged since 07:11: the arm at seat 1's probe-ref-x1 rest (joints 48.99, -77.26, 106.61,
      -124.34, -91.36, -38.05 deg), slider 0.30, no program, nothing in the controller's log since 07:11:21. The HEX-E
      streams with the night's noise; its zero drifted 0.26 N in Fz, which each action's tare takes out.
    - The run: `logs/20261104-chain_rod_points.sh STEPS LAST_START STOP_AT` (the night's chain with the day's times)
      and `logs/20261104-deadline_stop.sh`. The owner's latest time for the robot is asked.
  - **Done, 07:11:26 on 11-04: the night's rod run, every step clean, no abort.** The arm waits at the rest of seat 1's
    last reference press (probe-ref-x1), the rod a few mm over seat 1's pad. Move it clear before working at the rig.
    Nothing runs on the robot: the chain ended, and the 08:30 stop (PID 2877930) was stopped by its PID at 07:12:29.
    All nine probe sessions' streams were on GitHub by 09:20:49. The push stalled from 07:15 to 09:15 on the lab's
    uncommitted edit (NOTES, "The morning"). The night's sweeps per seat (friction, the offsets' stiffness, the
    reference presses) are in `20261104-rod_night_sweep_summary.json` (39b814e1c).
  - **The night (19:15:01 to 07:11:26): `logs/20261103-chain_rod_night.sh` on `logs/20261103-rod_night_steps3.txt`**
    (PID 2881408; its log `logs/20261103-chain_rod_night.log`): the night in halves (NOTES, "The night in halves").
    - The order: seat 2's first half (resuming `172647-rod21_seed`), seat 3's (a new session from s3v4, with its
      pass 1), seat 1's (from s1sph2b, with its pass 1); then the second halves of seats 1, 3 and 2; then the extras of
      seats 2, 3 and 1. Plans: `20261103-plan_probe_table_rod_seat{2,3,1}_v3id{5,4,6}_half_{a,b}.json` and
      `..._extras_night.json`, written by `20261103-rod_halves.py`.
    - Every sweep location of the _fit plans stays, lighter: offsets at 4 azimuths (32 a sweep), slides at az 0 and
      90 at 2 and 4 N (4 a sweep). Half b ends with the 5 N twists, a reference press and the 4 N rolls. The extras
      hold the 1.5 N rolls and the 20 and 35 deg obliques, those at azimuths 0 and 180 first.
    - Light sweeps take 103 to 165 s on the robot (median 150 s). The first halves are done, all clean: seat 2's at
      20:20:11, seat 3's at 21:47:50 (`202017-rod32_seed`), seat 1's at 23:29:10 (`214753-rod13_seed`; its fit at
      22:07:10 turned the rod -2.90 deg, inside the envelope). Seat 1's half b ended cleanly at 01:19:52 on 11-04, so
      seat 1 has every sweep location, twist and 4 N roll. Seat 3's half b ended cleanly at 03:01:39, so seat 3 has
      them too. Seat 2's half b ended cleanly at 04:39:56: every tip has all its sweep locations, twists and 4 N rolls.
    - The extras ran from 04:39:59: seat 2's ended cleanly at 05:28:29, seat 3's at 06:21:09 and seat 1's at 07:11:24
      (obliques about 25 s each).
    - Seat 2 so far: 7 sweeps in full (64 offsets, 8 slides) and probe-ref-00 before the switch; half a done (26
      light sweeps, probe-ref-a1; 809 of 832 offsets, 102 of 104 slides); 3 of half b's sweeps done.
    - The workstation's checker on the whole chain in run order flagged none of its 589 actions (19:29;
      `20261103-rod_night_halves_run_check.json`, NOTES).
    - Since 19:4x, 1x's probe-edge-654-o20-a90-r90 travels by the hub. With the rod turned as seat 2's fit found it,
      its joint travel reached 0.31 mm into seat 1's pad (the workstation's check of 19:43). Checked again: the
      travel keeps 1.23 mm or more (NOTES).
    - Seat 3's fit (20:36:49) turned the rod +10.75 deg about z_M, against seat 2's -5.40. Since 20:53 and 21:42,
      the plans still to run leave out what comes under 0.2 mm from the tip body at any turn from -11 to +11 deg. That
      is a few sweep azimuths, seat 1's sweep-edge-569-r0, and 8 to 13 obliques a seat in the extras. Four travels
      go by the hub. 3a runs as it was (NOTES, "the turn envelope").
    - The workstation's joint rod fit (23:48) turns the rod -0.22 deg about z_M and -1.90 across. Since 23:55 the
      extras also leave out eight edge-573 obliques that come within 0.20 mm of the tip body at its 2 sd corners.
      Checked again at the fit and its corners, no travel is flagged (NOTES).
    - No step was to start after 08:00 (the owner, relayed at 19:1x: "i mean i wont be in until 9"), with the hard
      stop at 08:30 (`logs/20261103-deadline_stop.sh`). The chain ended at 07:11:26, before either.
  - The earlier chain (18:01:10 to 19:09:41, PID 2867047, `logs/20261103-rod_night_steps2.txt`) ran seat 2 on the _fit
    plan from grid 4 to 8.
    - The HEX-E dropped at 18:01:2x, on the resume's first travel, and streamed again from 18:51:23 after the owner
      reseated the connector. The first resume (18:51:28) lost it again on its first travel; the second (18:52:17)
      ran on.
    - Sweeps took 258 s each, so the _fit plans would have ended seat 2 about 23:40 and left seat 1 its pass 1 and
      about 10 sweeps before the stop, then at 05:50. The chain was stopped by its PID at 19:09:41, then the program
      by SIGINT after grid 8 (exit 0 at 19:13:38).
  - **Tonight's rod run** (the workstation's proposal of 16:22, in priority order): sweeps on all three seats with the rod
    across and along (rod_deg 90 and 0), levels 0.5, 1, 2, 4 and 8 N, slides at 2 and 4 N, about 10 h; twists at about
    8 points a seat; rolls only if written and checked tonight. The lab's answers (16:27; the workstation's session
    could not be reached, so they are here):
    - 8 N: yes. abort_N is a plan field (safety.abort_N, default 10): 12 for the 8 N plans; travel_abort_N 8 and
      abort_Nm 1.5 stay. ft_t_cap_N 3.0 would abort 8 N offsets (0.45 x 8 N = 3.6 N across): about 6 in the 8 N plans.
      The lab is simulating two 8 N sweeps of seat 2 at mu 0.45 and 1.0.
    - Twists: yes, with the program's fix (the twist now backs off to 0.5 mm behind its rest; st + 0.5 from its stop
      left a deep press on the pad) and a simulated face and edge twist. v3 ID 3's twists used turn_v 0.016 at 2.5 N,
      5, 10, 15, 10, 5 deg (its 0.04 was for slides of under 1 deg): use 0.016. Check the turn poses for joint 6
      (seat 1 reaches -61 deg; the cap is -65) and the rod's ends against the body.
    - Rolls: not in the first chain. The lab can write a probe roll for the rod during the night; appended after the
      sweeps if the workstation checks its poses.
    - Presses at 0.15 mm/s (v 0.0005) at the 0.30 slider: agreed.
    - The workstation's session restarted at 16:2x. Its new address works (16:33 on); this session is hand-e3 [27e636].
  - chain_sphere2e was stopped by its PID at 15:42 so that seat 3's 2 mm grid does not start. Seat 3 has no 2 mm sphere
    session today; its rod seed comes from its 5 mm sphere session s3v4 (the two-key frame and that fit's T_AP).
  - The rod tonight (the owner, relayed 15:3x: "sure, yes to all these suggestions. we can put the rod on hopefully
    soon"): the workstation's sweep plans across the tip (rod_deg 90; c867f7d4), about 1.3 h a seat.
    - They are `../../20261030-inverted_rig_seat_block/data/20261103-plan_probe_table_rod_seat{2,3,1}_v3id{5,4,6}_sweep_draft.json`.
    - Each has 20 touches, all 2.68 mm or more from the tip's body, and 31 or 32 sweeps between two reference presses.
      A sweep has levels 0.5 to 3 N, 48 offsets on rings of 0.06 to 0.18 mm, and four 1 mm slides at 2 N.
    - Chained 2, 3, 1 from seat 2's hub, they replay with no flags; seat 3 also with s3v4's fit.
    - The lab is simulating three of seat 2's sweeps, since the sweep with the rod is a new pairing of code.
  - Tomorrow: the slider at 1.0 (needs v_ref_slider in the program, scaling every contact speed, the hard-coded ones
    too), forces to 8 N (abort_N to about 12 N), and twists and rolls inside a sweep.
  - **Since 14:17:57: `logs/20261103-chain_sphere2e.sh`** (copy here; PID 2838758; its log
    `logs/20261103-chain_sphere2e.log`). It waited for s1sph2b's end, then for the marker
    `logs/20261103-ok-fast`.
    - The lab writes the marker once the sims of `..._quick_neck_fast.json` pass (n_sph2_2, n_sph2_3, at about 14:50).
      Those plans have the grid presses at v 0.002 (0.6 mm/s) and pass 1 as before.
    - Each seat runs on _quick_neck_fast if the marker exists at its start, else on _quick_neck.
    - With the fast presses, seat 3 ends at about 18:15.
    - To stop the chain, kill its PID (a running program goes on).
    - The marker was written at 14:24:43, without the fast sims' end. The owner, relayed by the workstation at 14:23:
      "isn't it the exact same trajectories, just faster? what is high fidelity sim going to catch here".
      - The fast plans differ from the simulated and replayed _quick_neck plans only in v.
      - The fast sims had run 96 and 88 grid presses clean when /dev/shm filled at 14:02 and stopped them.
  - /dev/shm (16 GB of RAM) filled at 14:02 with the lab's sim sessions. The sims and the time-lapse died; s1sph2b
    writes to the disk and ran on. The lab deleted the sims of 10-23 to 11-02 (their results are in NOTES), which freed
    10 GB, and restarted the time-lapse at 14:24 (to about 19:25).
  - s1sph2b's pass 1: 0.060 mm rms over 19 touches, none left out. Its touches stand within 0.135 mm of s1sph2's
    (rms 0.068 mm). The pad's fit moved 0.27 mm and 0.6 deg from the morning's 19-touch refit, so the tip may have
    turned slightly in its seat in the incident or the owner's look.
  - The grid's flank obliques capped at max_depth_mm 1.0 stop at 1 N with the 2 mm sphere (reached_max). The
    simulator and the 5 mm sphere reached 2 N there.
- The chain as it was started (from 11:00:37), `logs/20261103-chain_sphere2.sh` (copy here; its log
  `logs/20261103-chain_sphere2.log`):
  - seat 1 (`s1sph2`), then seat 2 (`s2sph2`), then seat 3 (`s3sph2`);
  - each run is a new session with its own pass 1 (`20261103-plan_probe_table_sphere2_seat*`, stages
    preflight,homing,pad,register,grid), with `--frame` the pad-fitted frame
    `../../20261030-inverted_rig_seat_block/data/20261102-strip_frame_pads.json`;
  - simulated chained at the real speeds and replayed clean by the workstation (0 of 860 travels flagged);
  - about 2.3 h a seat: seat 1 to about 13:20, seat 2 to about 15:40, seat 3 to about 18:00;
  - a run stopped by a HEX-E loss is resumed once `hexe_probe.py` finds the sensor streaming, and anything else stops
    the chain;
  - to stop the chain, kill its PID (a running program goes on).
- The rod's grid plans below are replaced by its sweep plans (above, c867f7d4, in the order 2, 3, 1). They stay for the
  record. The rod's grid plans: (`20261103-plan_probe_table_rod_seat*`, converted again at 12:0x on the workstation's
  d8cecd9e, fe4e1914), in the order 3, 2, 1 from over seat 3.
  - Pass 1 now takes only presses whose touch-off keeps 2.0 mm from the tip's body. That gives 20 touches a seat, all
    3.14 mm or more. ce3a3f0b's plans, with five touches at 1.21 to 1.48 mm, are in `superseded/`.
  - The depth is lowered on 11 presses a seat.
  - The workstation's replay (bb79d68e): none of 587 travels flagged.
  - If the owner moves the arm for the tool change, replay the first travel from the robot's joints (`dry-check --real
    --start-q`). Each seat's session is seeded from its 2 mm sphere session
  (`probe_seed_session.py`) and runs `--resume SEED --stages preflight,pad,register,grid` on the same frame. The rod code
  is installed (ce3a3f0b). In the simulator its pass 1 fits to 0.013 to 0.022 mm rms on the three seats, and seat 1's
  grid ran 177 of 177.
  - The workstation replayed the rod plans at 11:0x (4dec06c1), chained 3, 2, 1 from the end of the 2 mm sphere's seat 3:
    0 of 584 travels flagged, the seams' least stem 12.85 mm.
  - Two presses read 4.5 and 4.7 mm into the tip's body by the k-d tree. On the exact meshes they stand 1.65 and 1.56 mm
    off it.
  - The rod's turn about the tool's axis stays unseen (sd about 3 deg), even with rod-along touches; it barely changes a
    contact.
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
