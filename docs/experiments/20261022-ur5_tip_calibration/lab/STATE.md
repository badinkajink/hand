# UR5 cell: current state

Updated 2026-10-30 09:43 by the lab. From the tip swap on, the lab session updates this file at every session
start and end. This file gives the present state of the cell. `../QUEUE.md` and `NOTES.md` are the history. Every
request to the owner must come from this file.

## On the robot

| Item | Now | Registration (folder here) |
|---|---|---|
| Tip | v3 ID 3 since the swap at about 13:3x on 10-29: the owner took it from a holder station and put v3 ID 2 there | tool: block 1 (`20261029-134508-spherev3id3`), seeded from `20261028-184415-spherev3id2` |
| Mount | v7 standoff fork on the HEX-E; stayed on the flange through both swaps | homing pose: `20261027-225213-spherev3id1` |
| Sphere | low sphere, not moved since 10-27 | `20261027-225213-spherev3id1`; with ID 2's tool `20261028-184415-spherev3id2`; with ID 3's `20261029-134508-spherev3id3` |
| Rod | on, at its third seating | `20261028-102701-rodv3id1`; with ID 2's tool `20261029-031559-rodv3id2`, 0.16 mm from 102701's; with ID 3's `20261029-230820-rodv3id3`, 0.67 mm from 102701's and 0.54 mm lower (NOTES, block 5) |
| Tall indenter | on | `20261028-114908-tallv3id1`; with ID 2's tool `20261028-200508-tallv3id2`, 1.85 mm from 114908's along -y, within the pass's own error in y |
| Holder | both held tips pad-up, seating 5: v3 ID 1 in the right-side station since 10-28; v3 ID 2 in the station ID 3 left (owner, 13:4x on 10-29) | `20261027-holder-apex-both` |

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

## Today and tonight: v3 ID 3's suite

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
| block 4 | `20261029-plan_tall_lab_stage2_night_v3id3.json` | 101 | **Stopped at 00:08:00 on 10-30** after 46 of 101 ok (grid 46 of 56), on a HEX-E loss: tall-edge-626, -668 and -670 aborted with no HEX-E sample. `--resume` on block 3's folder continues it (55 to do, about 1.9 h). The remainder simulated clean from home: mount 4.09 mm, finger 2.48 mm, joint 5 up to 97.6 deg. Either of the workstation's wrist orders of the same 55 can take its place: `20261030-plan_tall_lab_stage2_night_wrist_v3id3.json` (c5c34f60, simulated clean from home, and passed the gate at 09:09) or `20261030-plan_tall_lab_stage2_night_wrist2_v3id3.json` (21ee5fa8, simulated clean from home, and passed the gate at 09:43). |
| block 6 | `20261029-plan_rod_lab_stage2_night_v3id3.json` | 100 | Waits for block 4. Its 2 chunks simulated clean on block 5's registration (23:38), and their cell checks passed: mount 3.06 mm, finger 1.26 mm. About 1.4 h. |

- **The night stopped at 00:08 on 10-30 on a HEX-E loss.** At the end of the travel into tall-edge-626, wrist 1 swung
  from -36.6 to +19.3 deg with wrist 2 rising from 54 to 90 deg, the signature of both losses on 10-28. The DAQ
  answers, but the sensor sends no samples (00:10). The arm stands still in free air, its pad about 12 mm above
  tall-edge-626's point (joint 5 90.5 deg, joint 6 31.4 deg).
- **The HEX-E came back by itself at 01:37:18 on 10-30, 89 min after the loss.** A 20 s read gave 2001 samples, with
  no time-out and no gap. The lab reads it for 10 s every 10 min (scratchpad `hexe_stay.sh`): all 37 reads from 01:39
  to 07:40 were clean. The arm has not moved.
- The night ran in ID 2's order 3, 2, 5, 4, 6, through scratchpad `id3_chain.sh`, which ended at the stop.
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

- **The go to resume the night.** The HEX-E is back (01:37 on 10-30) without a reseat; if it drops again, reseat its
  connector first. On the owner's go, the lab runs scratchpad `resume_id3.sh`. It checks a 20 s stream and goes home
  on a checked path. It repeats three sphere touch-offs (`20261030-plan_sphere_lab_check_v3id3.json`) in a seed of
  block 1's registration, and stops if any onset is more than 0.1 mm off, which would mean the tool moved. Then it
  resumes block 4 (55 actions) and block 6 (100), about 3.5 h. All four of its simulations pass its gate (01:39).
- **Which order for block 4's 55.** In the night's order, 7 of its travels swing wrist 1 from below 0 to above it while
  wrist 2 rises, the signature of the HEX-E's losses: the first, from home into tall-edge-626, and six slides. It also
  has 12 swings of wrist 1 over 40 deg; R29's loss came after a downward one. The workstation's two reorders keep the
  path-check margins:
  - wrist (c5c34f60): none of the upward travels, but one downward crossing with a 55 deg swing, into
    tall-slide-edge-643-1N. The lab's simulation found the same. It passed the gate at 09:09: mount 4.09 mm, finger
    2.50 mm, joint 5 2.4 deg inside its window, the night order's margins.
  - wrist2 (21ee5fa8): neither. The lab's simulation from home confirmed it at 08:29: its largest swing is +39.9 deg,
    with wrist 2 up 5 deg. It passed the gate at 09:43 with the same margins: mount 4.09 mm, finger 2.48 mm.
    The lab recommends wrist2.
  `B4PLAN=<plan> B4SIM=n3b4w resume_id3.sh`, or `B4SIM=n3b4w2` for wrist2, runs the chosen order in place of the
  night's.
- When v3 ID 1 goes back on, the three plans above.
- A new HEX-E cable.

## Where things are

- History: `../QUEUE.md` (R31 covers v3 ID 1) and `NOTES.md` (the lab's log).
- Results: `docs/experiments/20261027-v3_tip_calibration_and_crosstalk/` (README, `data/`, `feed.jsonl`) and the
  fingertip calibration page.
- Plan files that are no longer current, and their path checks: `superseded/`, under the same names. A plan that ran
  also has a copy in its session folder.
