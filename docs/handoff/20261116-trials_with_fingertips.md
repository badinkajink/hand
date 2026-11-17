Status: active
Task: 2026-11-16 23:4x, owner: "lets make sure the next time we run trials we have the fingertip sensing going. ideally
with pose tracking, closed loop control too. that can be churned away on overnight to be ready for evaluation". Every HOM
bench trial records the three fingertip boards and feeds their contact estimates to HOM beside the tracked pose, tested
offline on the 11-16 recordings, so the next bench session runs it and compares HOM with and without the tips.
Read first: docs/experiments/20261115-hom_bench_trials/README.md (the 11-16 sections: "Logging", boards ch0 middle, ch1
index, ch3 thumb; "State on 2026-11-16 13:21"); its lab/hom_trial.py (trial loop, tracker, StateMsg), lab/pinch_pickup.py
(reads the mux log live: each board's field change from a rest taken before the close), lab/trial_cycle.sh;
src/sr2hand/control/link.py (StateMsg.cop: per finger x, y, z in P mm and |F_n|, NaN when absent), hom_server.py
(_contacts, Server.step passes them to HomBench.step), hom_bench.py, hom.py (Measurement.cop, ContactFilter, gcf_from_cop);
src/sr2hand/driver/reskin5x/host/reskin5x_mux.py (the logger and its .bin records), wrench_viewer.py (App: physical model
fitted at rest, --board-gp, --calib, --correction, --replay, --mux, --names); scripts/fingertip/viewer_roll_film.py (the
viewer's App on a set clock); hom_tasks.VARIANTS (tip, cal_tip: HOM with the fingertips' model in simulation); memory
project_sr2_crosstalk_night_2026_11_04.md, project_hom_bench_trials_2026_11_15.md, project_hom_sim_real_reversal_2026_11_16.md.
State:
- 11-16 trials ran HOM closed loop on the tracked object pose (hom_server serve --plant measured; corvus's tracker sends
  p_PO, q_PO at 50 Hz) but sent no contacts: StateMsg.cop stayed NaN. reskin5x_mux.py recorded the boards separately,
  only for the 13:19 press (20261116-131831-mux.bin) and 15:40-17:01 (20261116-154014-mux.bin: the three pick-ups and
  the twists at 16:12 and 16:47). Copies on the workstation in logs/20261116-three_finger_mux/ (gitignored); the trial
  videos of the press, the first pick-up and those twists in logs/20261116-fingertip_film/raw/.
- No calibration maps the hand's three boards to contact points and forces yet (README 13:21).
<!-- ap:steps: rendered by ~/.claude/autopilot/ap from its ledger; change it with `ap step`, not by hand -->
Steps, in order (the ledger; `ap step done <job> <id> "result"` marks one, `ap step add <job> "text"` adds one):
- [ ] 1. Read the files above. Write into State how the mux logger is started on corvus, its record format and clock, which
    tips (case, ID) sit on the three fingers, and what HomBench.step does with contacts.
- [ ] 2. A contact estimator for the three boards (new src/sr2hand/driver/reskin5x/host/tip_contacts.py): per board the
    viewer's model (the rest fit taken at the trial's open pose before the close, or the UR5 GP of that tip where one
    exists), the contact point in the tip frame turned into P mm through the SR2 hand's kinematics at the measured
    joints (sr2hand.robot.tool_chain SR2Hand, assets/cad_v3/frames.json), the normal force in N, no contact below
    0.25 N as in the viewer. It runs under corvus's /usr/bin/python3 3.10 (numpy 1.26, scipy 1.12; no MuJoCo, no torch).
- [ ] 3. hom_trial.py (and pinch_pickup.py) start or attach to the mux logger for every trial, refuse to start unless all three
    boards stream (override --no-tips), write the raw board records into the trial's run folder on the trial's clock, and
    fill StateMsg.cop from the estimator every period. trial_cycle.sh passes the flags.
- [ ] 4. hom_server: a --tips flag with which HomBench uses the measured contacts (as the tip variant does in simulation) and
    records them per period. Default off, so 11-16's behaviour is unchanged.
- [ ] 5. Offline test, no hardware: replay the 11-16 press (runs_corvus 131927 + 131831 mux) and the first pick-up through the
    estimator and a local hom_server dry run with --tips on and off. Check the frames (contacts on the tip surfaces and
    within a few mm of the tracked object's surface in P), the timing (estimator and message inside the 20 ms period on
    one core) and the server's targets. Report the numbers in the runbook section of step 6.
- [ ] 6. A section "Trials with the fingertips (from 2026-11-17)" in the bench README: the commands (mux logger, rest, trial
    with tips, trial_cycle.sh), the checks, and the evaluation for the next session (HOM with and without --tips on the
    same grasps, two trials each, the tracking score and the contact traces).
- [ ] 7. Commit by path (never git commit -a); push with GIT_SSH_COMMAND=ssh git push origin main in a retry loop.
<!-- /ap:steps -->
Claimed: docs/experiments/20261115-hom_bench_trials/lab/ (hom_trial.py, pinch_pickup.py, trial_cycle.sh, new files),
src/sr2hand/driver/reskin5x/host/tip_contacts.py (new), src/sr2hand/control/hom_server.py and hom_bench.py (the --tips
path only), the bench README (one new section)
Waiting on owner: none
Do not: touch the hardware, corvus or the CB1 (copying files from corvus is fine; nothing that changes them); change the
default behaviour of hom_server or hom_trial without a flag; edit the talk deck or the press autopilot's files
(docs/handoff/20261116-press_sim_real.md's Claimed line); run GPU jobs; commit the mux .bin files (over 2 MB: out of git).
