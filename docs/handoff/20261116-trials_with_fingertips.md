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
- (step 1, 11-17 00:2x) Mux logger: started apart, on corvus from /home/will/workspace/hand, as `/usr/bin/python3
  src/sr2hand/driver/reskin5x/host/reskin5x_mux.py [--stat-s 10] > logs/<date>-three_finger_mux/<date>-splitter.log`
  (11-16 13:18:31 and 15:40:14; /dev/ttyACM0, channels [0, 1, 3], 160.6 records/s). Writes
  logs/<date>-three_finger_mux/<stamp>-mux.bin: 8 + 80 n bytes a record, corvus's time.time() f8, then per board in channel
  order (ch0 middle, ch1 index, ch3 thumb) 5 chips x (T degC, x, y, z uT) f4. Stamp = the serial read's time, a read's
  earlier records back-dated at 1/150 s (237 backward steps in 18,786); some records are framing garbage (T 3.7e26) that
  the viewer's filter drops. The .bin reaches disk in 8 KB blocks (~0.2 s); the per-board pseudo-terminals
  /tmp/reskin5x_mux/ch<c> get each read at once (~50 ms); nobody read them on 11-16. Clock = corvus's epoch, hom_trial's.
- Tips: thumb v3 ID 1 (ch3), index v3 ID 3 (ch1), middle v3 ID 2 (ch0) (crosstalk README; viewer inputs in
  docs/experiments/20261022-ur5_tip_calibration/data/20261105-viewer_inputs/, board GP 20261105-board_gp_v3_id1_id3_rig_id3_id6.npz).
- HomBench.step: hom_server already passes every finite StateMsg.cop to it (Server.handle `_contacts`); HomController.gcf
  takes a finger's GCF from its CoP (Mirror.gcf_from_cop: closest points of pad and object to the CoP) when F >=
  prm.contact_force_min (0.02 N; tip variant 0.25 N), else witness points; F also sets `touching`. So --tips must gate it.
- Physical model alone misplaces a 1-2 N apex press by ~10 mm (synthetic, ID 1); the viewer's board-GP blend is needed.
- (steps 2, 8, 11-17 00:3x) tip_contacts.py done; replay of press 13:19 (logs/20261117-tip_contacts/): rest keeps the
  saved models (0.10-0.26 vs new fits 0.22-0.35 mT); thumb 2.8 N from 0.44 s, index 2.4 N from 0.86 s, middle 0.5 N from
  1.68 s (own-pose on). Gap to the tracked cylinder: thumb -4.0, index -8.7, middle +3.3 mm; the index's pad APEX is
  itself -7 to -10 mm inside at the measured joints (hand model/tracker offset, not the estimator). Live path reads the
  pty outlets (one reader at a time). Next: hom_trial integration (tips default on, --no-tips), then hom_server --tips
  {off,on,raw} (raw keeps sil's in-process equality), then the replays and the README.
- (steps 3-4 written, 11-17 00:4x, uncommitted) hom_trial.py: open_tips (attach/start splitter, refuse unless 3 boards
  >= 100/s, records OUT/<stamp>_<task>_tip_<finger>.bin from 2 s back), rest before t0, estimator thread, StateMsg.cop,
  row "tips"/"tips_age_ms"; new `replay` (a record + its mux log -> a server, offline) and test-only --tips-links.
  pinch_pickup.py Pads now on the live pty feed + TipContacts (rest state in REST json for lift/slip), --no-tips alias.
  trial_cycle.sh: --plan $P $TIP_FLAGS, exit 4 on a refusal. hom_server: --tips off|on|raw, header + per-period "gcf".
  tip_contacts.py: play (a .bin into ptys), rest_state/load_rest_state. lab/tips_live_check.py: the live path offline
  (played .bin, stand-in station, local server); first run: 155 rec/s per board, 7.6 ms median estimate on one core.
- (steps 3-6 done, 11-17 00:4x) Offline numbers in the README section "Trials with the fingertips (from 2026-11-17)" and
  data/20261117-tip_contacts/*.json (lab/tips_offline_check.py press|pickup|hom, lab/tips_live_check.py). Live path on one
  core: link period 20.0 ms (max 22.9), estimate 7.5 ms, data age 41 ms. HOM --tips off reproduces 11-16's targets to
  0.017 deg and ignores contacts; --tips on moves them 0.86 deg median, 18 max. sil with --tips raw: UDP = loopback.
  Step 7 (commit, push) next; nothing waits on the owner until the bench session (README's evaluation list).
<!-- ap:steps: rendered by ~/.claude/autopilot/ap from its ledger; change it with `ap step`, not by hand -->
Steps, in order (the ledger; `ap step done <job> <id> "result"` marks one, `ap step add <job> "text"` adds one):
- [x] 1. Read the files above. Write into State how the mux logger is started on corvus, its record format and clock, … Done 11-17 00:03: State lines: logger command, .bin format and clock (corvus epoch, 1/150 s back-dating), tips ID1 thumb ch3 / ID3 index ch1 / ID2 middle ch0, HomBench.step's CoP GCF; hom_server already forwards any finite cop
- [x] 2. A contact estimator for the three boards (new src/sr2hand/driver/reskin5x/host/tip_contacts.py): per board th… Done 11-17 00:13: src/sr2hand/driver/reskin5x/host/tip_contacts.py: viewer's estimator per board (saved v3 calib, rest refit rule, board GP blend), pad->tip->P transform, pty-outlet reader, .bin replay; runs under py3.10/numpy 1.26/scipy 1.12; 6.3 ms median for 3 boards on one core
- [x] 3. hom_trial.py (and pinch_pickup.py) start or attach to the mux logger for every trial, refuse to start unless … Done 11-17 00:32: hom_trial.py (open_tips: attach/start splitter, refuse unless 3 boards >= 100/s, records OUT/<stamp>_<task>_tip_<finger>.bin, rest before t0, estimator thread, StateMsg.cop, row tips; replay command), pinch_pickup.py Pads on the live feed + TipContacts (--no-tips), trial_cycle.sh --plan $P $TIP_FLAGS (exit 4 on refusal)
- [x] 4. hom_server: a --tips flag with which HomBench uses the measured contacts (as the tip variant does in simulati… Done 11-17 00:32: hom_server serve --tips off|on|raw (off default: contacts recorded, not used; on: tip variant floor 0.25 N; raw: sil), header tips + floor, per-period gcf source/dist; sil press with raw: UDP = loopback 0.0 deg, float64 0.0008 deg, J 0.1419
- [x] 5. Offline test, no hardware: replay the 11-16 press (runs_corvus 131927 + 131831 mux) and the first pick-up thr… Done 11-17 00:32: lab/tips_offline_check.py + tips_live_check.py, data/20261117-tip_contacts/: press thumb 2.8 N gap -4.0 mm (apex +1.3), index 2.4 N -8.7 (apex -8.2), middle 0.53 N +3.3; pick-up thumb 1.29 N +0.3 mm, middle 1.26 N +1.6, index none; one core: estimate 7.5 ms, link period 20.0 ms (max 22.9), estimate age 7.8 ms, data age 41 ms; HOM off = 11-16 targets to 0.017 deg, contacts change nothing; on: 0.86 deg median, 18 max (middle MCP/PIP)
- [x] 6. A section "Trials with the fingertips (from 2026-11-17)" in the bench README: the commands (mux logger, rest,… Done 11-17 00:35: README section 'Trials with the fingertips (from 2026-11-17)': terms, estimator, commands (logger, rest, server --tips, trial, trial_cycle.sh), the records, the server's switch, the offline numbers (press, pick-up, timing, HOM dry runs), the evaluation (g13, g12, press lat0; off/on/on/off; tracking score, drift, contact traces) and two Future work items; prose lint 16 words mean, 2.1 % over 30
- [ ] 7. Commit by path (never git commit -a); push with GIT_SSH_COMMAND=ssh git push origin main in a retry loop.
- [x] 8. Own-pose crosstalk: a per-board GP on the finger's own three joint angles (from the 11-04/05 crosstalk rows) … Done 11-17 00:13: OwnPose GP (data/20261117-tip_contacts/20261117-own_pose_gp.npz, 598 KB) default on: press 13:19 middle's approach change 28-31 uT predicted 35-37 uT, chi2 20 -> <2, first contact 1.20 -> 1.68 s at 0.43-0.64 N; thumb/index change < 0.15 N
<!-- /ap:steps -->
Claimed: docs/experiments/20261115-hom_bench_trials/lab/ (hom_trial.py, pinch_pickup.py, trial_cycle.sh, new files),
src/sr2hand/driver/reskin5x/host/tip_contacts.py (new), src/sr2hand/control/hom_server.py and hom_bench.py (the --tips
path only), the bench README (one new section)
Waiting on owner: none
Do not: touch the hardware, corvus or the CB1 (copying files from corvus is fine; nothing that changes them); change the
default behaviour of hom_server or hom_trial without a flag; edit the talk deck or the press autopilot's files
(docs/handoff/20261116-press_sim_real.md's Claimed line); run GPU jobs; commit the mux .bin files (over 2 MB: out of git).
