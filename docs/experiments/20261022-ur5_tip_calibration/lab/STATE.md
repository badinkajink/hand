# UR5 cell: current state

Updated 2026-10-29 08:15 by the lab. From the tip swap on, the lab session updates this file at every session
start and end. This file gives the present state of the cell. `../QUEUE.md` and `NOTES.md` are the history. Every
request to the owner must come from this file.

## On the robot

| Item | Now | Registration (folder here) |
|---|---|---|
| Tip | v3 ID 2 since the swap at about 18:40: the owner took it from the holder's right-side station and put v3 ID 1 there | tool: block 1 runs (`20261028-184415-spherev3id2`), seeded from `20261027-225213-spherev3id1` |
| Mount | v7 standoff fork on the HEX-E; stayed on the flange through the swap | homing pose: `20261027-225213-spherev3id1` |
| Sphere | low sphere, not moved since 10-27 | `20261027-225213-spherev3id1`; with ID 2's tool `20261028-184415-spherev3id2` |
| Rod | on, at its third seating | `20261028-102701-rodv3id1`; with ID 2's tool `20261029-031559-rodv3id2`, 0.16 mm from 102701's |
| Tall indenter | on | `20261028-114908-tallv3id1`; with ID 2's tool `20261028-200508-tallv3id2`, 1.85 mm from 114908's along -y, within the pass's own error in y |
| Holder | both held tips pad-up, seating 5: v3 ID 1 went into the right-side station pad-up (owner, 18:5x: "yes it went pad up") | `20261027-holder-apex-both` |

`cell.json` holds the same registrations for the path checks and the renders.

The HEX-E's link fails about once per 6 to 9 h of running: 7 losses since 10-25 (`../QUEUE.md` R31, 17:42 on 10-28).
A loss aborts three actions and stops the session, and the robot then waits for a reseat. The cable is clamped 3 to 5
cm from the connector with slack. The fix is a new cable.

## Rules in force

- Every robot run needs the owner's go in the lab's session.
- Joint 6 goes to +45 deg at most without the owner, and to +60 deg with the owner present.
- No gravity sweeps.
- travel.speed is 1.0. The tall plans lift 20 mm between actions.
- Before it runs, every plan is path-checked on the current registrations and `cell.json` with
  `scripts/ur5_tip_calibration_holder_paths.py` (margins: robot 8 mm, mount 3.0 mm or 2.5 mm for touch-offs, finger
  0.8 mm, HEX-E 6 mm), and the lab simulates it.
- No force control, contact motion or safety threshold is set from a borrowed calibration that was not validated.
- The workstation reads corvus over SSH and never writes there; plans reach corvus by git.

## Last night: the next tip's night suite, done

Built by `scripts/ur5_tip_calibration_next_tip_suite.py` and path-checked on v3 ID 1's registrations. About 11.7 h.
Done at 08:13:17: blocks 1 to 6 and the sphere's top edge, 488 actions, all ok, with no abort, no early contact and no HEX-E time-out. They took 12.6 h of robot time; block 2 ran 0.5 h over its estimate. The blocks ran in the order 3, 2, 5, 4, 6 (`NOTES.md`, from "Block 1"). The robot has been idle since 08:13:17, at the rest pose of block 6's last press above the rod. The owner's go came in the lab's session at 18:3x; the owner left at about 19:25 ("worst case i can come back at ~9:30 to reseat").

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
| `20261028-plan_holder_lab_side_norod30_v3id1.json` | 30 hovers | the rod fixture off and the -y held tip pad-down |

## Waiting on the owner

- What runs next. The robot has been idle since 08:13:17, and the arm stays where it is until the owner says.
- Faster touch-offs ("can you go faster on the approaches"): their slow approach at v 0.0005 (0.15 mm/s at the
  slider), as the presses run, moves the onset by 0.000 mm (SE 0.006) in block 2's presses, and a touch-off takes
  about 50 s instead of 81 to 108 s (`NOTES.md`, "Faster touch-offs"). The plans are unchanged until the owner says.
- When v3 ID 1 goes back on, the four plans above.
- Indenter plate v2d: the front row at 95 or 100 mm, a support under the pad-down block, the plate's place and timing,
  and plans for the 12.5 mm sphere.
- A new HEX-E cable.

## Where things are

- History: `../QUEUE.md` (R31 covers v3 ID 1) and `NOTES.md` (the lab's log).
- Results: `docs/experiments/20261027-v3_tip_calibration_and_crosstalk/` (README, `data/`, `feed.jsonl`) and the
  fingertip calibration page.
- Plan files that are no longer current, and their path checks: `superseded/`, under the same names. A plan that ran
  also has a copy in its session folder.
