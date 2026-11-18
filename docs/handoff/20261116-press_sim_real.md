Status: done
Task: 2026-11-16 23:0x, owner: "sure, send off a background autopilot run to churn on the press task calibration". After
the blind twists and the pick-up hang, compare the hand's two press trials of 11-16 with simulation under the calibrated
servo, per fingertip contact variant (1 mm sphere pads or point contact; soft or stiff tangential contact; noslip), as the
twist and pick-up comparisons did. Owner: "we have like 1.5 good trials".
Read first: docs/experiments/20261115-hom_bench_trials/README.md (sections of 20:10 and 13:21; lines on the press, "No
ground-clear press grasp exists"); its runs/20261116-131243_press_d25_trial000.jsonl and 131932 (HOM server records),
runs_corvus/20261116-131239_press_corvus.jsonl and 131927; data/20261116-trial_tracking.json (scripts/hardware/
hom_trial_tracking.py scores the press: err = RMS angle between the tracked cylinder axis and the start axis turned by
-theta_ref about x_P); scripts/hardware/hom_blind_compare.py and hom_pickup_compare.py (the structure to follow);
logs/20261116-sim_real_reversal/: rev_stem.py (the sphere's real object), rev_gear.py, rev_clutch2.py (the calibrated
servo: gears the load cannot backdrive, --stiff, index and middle yaw kp 0.35), rev_predict.py (--scene --mu --trace),
pickup_sim.py (SCENES: the contact variants), scenes/; logs/20261116-hom_setup/press_lat_rerun.py, press_lat_plan.py and
press_lat_score.py (how the lat0 grasp was made and scored on the servo plant); memory
project_hom_sim_real_reversal_2026_11_16.md.
State:
- Twist (hand 6bc0bca5f, ec57a06d9): calibrated sim within 0.06 of the hand's tracking blind, pads or point contact alike;
  the design pipeline's model 0.25. Pick-up hang (hand 31a9b46a3 to d61cccac5): pads with creep-free stiff contact hold
  the hanging sphere, point contact swings it, Newton hydroelastic (joints locked) holds 32 deg.
- Press trials (both at (70, 110), plant spec kp0.55_kv0_fr0.18_fl0_dp0.011, no torque budget): 13:12 the generalist
  press (plans/20261115-generalist_press_plan.json), drift guard at ~5.5 s, the cylinder dropped, track 0.469; 13:19 the
  pinch raised to the axis (logs/20261116-hom_setup/20261116-press_x70_y110_lat0_plan.json; copy in plans/), full
  schedule, held, track 0.452, fingertip boards recorded. Press schedule t_end 5.5 s, scored from 2.0 s. In both the
  arm lifted the hand 40 mm at 1.6 s (the corvus record's header "lift"), and the cylinder carried the sphere's tag block
  (ids 16-21, stem 24.41 mm as assembled; object_tracking_mjcf.BLOCK_ON and assembled_stem_mm; the trials' map
  maps/20261116-cylinder_fitted_map_O.json). Measured masses: cylinder 23.6 g, block 6.5 g (README, changes of 11-16).
- Step 1 (23:2x). Press angle = turn of the tracked axis about x_P from its 2.0 s pose, + in theta_ref's sense
  (ref 0 -> 20 deg 2-3 s, held to 4 s, back to 0 at 5 s, held to 5.5 s; ref RMS 13.8). Generalist: 13.6 deg at 3.0 s,
  15.2 mean over the hold, 15.8 max; back through 0 at about 4.8 s and on to -7.3 at 5.0, -29 at 5.48 as the cylinder
  fell (z first 5 mm low at 5.1 s, -24.9 mm at the end); the middle PIP flexed with its target to 68 deg in the stroke.
  lat0: 6.1 at 3.0 s, 7.7 mean over the hold, 9.4 max, 4.8 at 5.0, 2.0 at 5.48; held. In both the close pushed the
  cylinder about 9.5 mm toward -x_P and turned its axis 17-22 deg about the vertical before 2.0 s.
- Station rejections: HOM's limit target for the middle PIP is -16.110026 servo deg (the model's joint range in rad),
  beyond the station's -16.11, and the station refuses the whole frame. lat0: 153 of 301 periods at that target from
  0.94 s (106 of 175 in the task), 147 of 301 frames reached the hand, HOM ran on its reference joints in 69 of 175
  task periods; the hand's middle PIP stayed at -7..+4 deg (median -0.4) where HOM commanded -16.1; HOM's target cycled
  3 periods at the limit, 2 near 0. Generalist: first at 5.10 s, after the stroke (the fall).
- Step 2 done: press_stem.py, 8 scenes (30.1 g, centre of mass 10.3 mm below the axis, 38.2 mm from the pinch line).
- Steps 3, 7, 4, 5 done (23:2x-23:45): press_predict.py run (144 closed-loop runs: station on/off, mu 1.0 and 0.5),
  replay (48: the hand's accepted targets open loop, table mu 1.0/0.35, mu 0.5), pipeline (6); rows and script copies in
  docs/experiments/20261115-hom_bench_trials/data/20261116-press/sim/, comparison 20261116-press_contact_compare.json,
  figure media/20261116-press_sim_vs_hand.png, README section "State on 2026-11-16 23:45". Closed loop fits neither
  trial; replay favours soft contact (point, mu 0.5: 2.6 / 3.7 deg RMS, generalist falls 5.14 s vs hand 5.05 s).
- The deploy bug found: HomBench.step clips only beyond 1e-3 deg, the model's joint limits lie 2.5e-5 deg past the
  station's, and the station refuses the whole frame (owner's call; hom_bench.py is outside this job's claim).
- 23:34:54 the resguard watchdog killed the four HOM twist servers 47035-47038 and another session's inner_loop_mpc run
  (not this job's); step 8 waits on the owner for their relaunch.
<!-- ap:steps: rendered by ~/.claude/autopilot/ap from its ledger; change it with `ap step`, not by hand -->
Steps, in order (the ledger; `ap step done <job> <id> "result"` marks one, `ap step add <job> "text"` adds one):
- [x] 1. Read both press trials (server and corvus records, the tracking rows): the plan, the object's pose and press … Done 11-16 23:14: Both trials read: press angle generalist 15.2 deg mean over the hold then a fall from 5.1 s; lat0 7.7 deg, held; lat0's frames refused by the station whenever HOM set the middle PIP at its limit (153 of 301)
- [x] 2. Build the press scene with the real cylinder (25 mm, its stem and the sphere's tag block as glued on 11-16; g… Done 11-16 23:25: 8 scenes logs/20261116-sim_real_reversal/scenes/cyl025_stem_block_{pads,pt}_imp{1,100}[_ns30].xml (press_stem.py): 30.1 g, centre of mass 10.29 mm below the axis at mid-length, 38.2 mm from the pinch line toward the middle (gravity 11.3 mN m about it); table under the block off at 1.6 s
- [x] 3. Write logs/20261116-sim_real_reversal/press_predict.py (new; do not change rev_predict.py or rev_gear.py): HO… Done 11-16 23:38: press_predict.py run: 144 closed-loop runs (8 scenes x 2 plans x seeds 0-2, station on/off, mu 1.0 and 0.5) and 6 pipeline runs: no variant fits both trials (stiff holds, generalist 0.73-0.77 vs hand 0.469, lat0 0.09-0.18 vs 0.452; soft drops lat0 at the lift); pipeline 0.91/0.23
- [x] 4. scripts/hardware/hom_press_compare.py (new): hand against each variant (tracking error, press-angle trace RMS… Done 11-16 23:38: scripts/hardware/hom_press_compare.py -> data/20261116-press/20261116-press_contact_compare.json and media/20261116-press_sim_vs_hand.png (checked at 900 px); rows and scripts copied to data/20261116-press/sim/
- [x] 5. A section "State on 2026-11-17 hh:mm: the press against simulation" in the bench README (what fits, what does… Done 11-16 23:43: README section 'State on 2026-11-16 23:45: the press against simulation' (stamped from date; prose_lint within targets) and 9 lines in the memory file, its outdated 'not yet done' line replaced
- [x] 6. Commit by path (never git commit -a), push with GIT_SSH_COMMAND=ssh git push origin main in a retry loop. Done 11-16 23:45: Committed by path (f9f2bbb3d step 1, edd1616cb the comparison) and pushed: 127321710..edd1616cb on the first try
- [x] 7. Replay the hand's applied targets open loop on each scene variant (press_predict.py replay): HOM's frames tha… Done 11-16 23:38: press_predict.py replay, 48 runs: soft tangential contact follows the hand's press (mean |track error| 0.16 vs 0.30-0.36 stiff); point contact, soft, mu 0.5 follows both traces within 2.6 and 3.7 deg RMS, the generalist falling at 5.14 s (hand 5.05 s)
- [x] 8. resguard watchdog, 2026-11-16 23:34:54 (PSI full 26.5, MemAvailable 13.8 GB): killed the four HOM twist serve… Done 11-17 17:08: Closed 11-17 17:1x by hand-4a, nothing relaunched: superseded, the bench sessions of 11-17 started their own servers (47041-47050; 47049/47050 at the 7.5 V supply), so 47035-47038 stay down (owner 17:0x: see through the remaining tasks)
<!-- /ap:steps -->
Claimed: logs/20261116-sim_real_reversal/press_*, scripts/hardware/hom_press_compare.py,
docs/experiments/20261115-hom_bench_trials/data/20261116-press/, docs/experiments/20261115-hom_bench_trials/media/20261116-press_*,
docs/experiments/20261115-hom_bench_trials/README.md (a new section only)
Waiting on owner: step 8 (relaunch the HOM servers 47035-47038, or leave them down); the clip fix in HomBench.step or
lab/hom_trial.py, and a rerun of the lat0 press on the hand (README section 23:45, Future work).
Do not: edit the talk deck (the contact repository's talks/20261117-hand_design/ or build_talk.py; the talk is 11-17
at 12:00 and hand-4a owns it); touch the hardware, corvus or the CB1 (the HOM servers on ports 47035-47038 may stay up);
run GPU jobs; change the outputs of the existing rev_*, pickup_* or hom_*_compare scripts (copy them into new press_*
files); run CPU jobs outside ~/.claude/bin/resguard.sh run (measure one run's RSS first).
