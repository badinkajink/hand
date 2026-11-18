Status: active
Task: 2026-11-17 18:01, owner: "what the high priority experiments are, and coordinate with the ur5 session to fill out
our understanding of the system as-is". The workstation half of docs/experiments/20261117-sr2_system_identification/
QUEUE.md: fit the lab's overnight data (W2) and score tonight's A/B twist repeats (W3) as they reach origin/main.
Read first: docs/experiments/20261117-sr2_system_identification/QUEUE.md (requests S1-S5, W lines with every result so
far); the lab's page docs/experiments/20261117-sr2_bench_session/20261117-sr2_hand_bench_session_7v5.html;
scripts/hardware/sr2_servo_sysid_fit.py (11-06's fit); scripts/hardware/hom_trial_tracking.py;
docs/experiments/20261117-sr2_system_identification/data/20261117-g0_x70y110_twist_predictions.json; memory
project_sr2_identification_night_2026_11_17.md.
State:
- Done and pushed (QUEUE W1, W1b, W1c, W3 lines): g13 and press open poses inside the MCP stops, blind predictions for
  g13 and g0 (70, 110), finger gaps and the 22-layout screen, press closure and chatter, HOM flags --filter-step-s,
  --contact-off, --eq1-frames (off by default; e0e14dcf1).
- The lab's night ended 21:33 (NOTES.md): S2, S1(b) thumb, S1(a) with redo, the boards' sweep (327 poses), both A/B
  blocks (9 of 12 cycles of the second), all on origin/main at c03a378d7. Lights off; nothing more comes tonight.
- Autopilot 23:1x: step 6 done (QUEUE W2 S1(a) line; sim/20261117-s1a_ranges_vs_limits.py,
  data/s1/20261117-s1a_ranges_vs_limits.json). Next: step 7 (own-pose GP from the sweep and the mux file), then 8.
<!-- ap:steps: rendered by ~/.claude/autopilot/ap from its ledger; change it with `ap step`, not by hand -->
Steps, in order (the ledger; `ap step done <job> <id> "result"` marks one, `ap step add <job> "text"` adds one):
- [x] 1. When S2's tables are under data/s2/ (dwells.tsv, holds_c.tsv, registers.txt, the plan steps): give scripts/ha… Done 11-17 19:38: Done in the window 19:4x: data/s2/20261117-sr2_servo_7v5_fit.json; QUEUE W2 line
- [x] 2. When S1(b) is under data/s1/: per pose the tag centre's and normal's error against the model at the station's… Done 11-17 19:42: Thumb done in the window 19:5x (data/s1/20261117-thumb_zero_fit.json, QUEUE W2 S1(b)); middle not detected, index not attempted by the lab
- [x] 3. When the A/B repeats are pushed (runs/ and runs_corvus/ 20261117-19*, 20*): score each with scripts/hardware/… Done 11-17 20:14: Done in the window 20:2x: the lab scored the A/B (data/20261117-night_repeats_tracking.json); QUEUE W3 A/B line against the blind predictions; --filter-exact added (the sub-steps' 12.5 ms compute explains the fragments)
- [x] 6. When S1(a)'s joint ranges are pushed (lab/s1a_ranges.py via night_rest3.sh, after the board sweep): per joint… Done 11-17 23:20: QUEUE W2 S1(a) 23:3x: index and middle MCP lows stall at -10.3/-8.5 deg (model -15), moving to -11.4/-10.0 at 21-22 % duty; other ends reached their limits or the caps; safe range 4.7 deg inside the model at 4 ends; data/s1/20261117-s1a_ranges_vs_limits.json
- [ ] 7. When the boards' free-air sweep at the trial layout is pushed (the owner's 18:0x choice: 'a free-air sweep of the fingertip boards at random poses ... for the own-pose GP at the trial layout'): build the own-pose rows in the format of scripts/fingertip/sr2_crosstalk_hand.py own_models (q, <finger>_rows, <finger>_Y: each board's change from the layout's zero pose), fit with tip_contacts.fit_own_pose to a NEW file beside the default (never overwrite tip_contacts.OWN_POSE, which the trials load), and report each board's five-fold error against the 11-04/05 fit (5.4-6.1 uT median) and the old GP's error on the new sweep; one QUEUE line (W2, own-pose GP). Commit, push.
- [ ] 8. When the second A/B (the lab's night_repeats2, about 21:07-22:15: g0 at (70, 110), tips off base 47051 / tips off --filter-exact 47059 / tips on base 47052 / tips on --filter-exact --contact-off 0.10 47060, three cycles each) is pushed with the lab's scores: one QUEUE line (W3, A/B 2) per arm against the first A/B and the blind g0 predictions; check from the server records (runs/) that the --filter-exact arms' compute_ms median stays near 2.6-2.9 ms and that no one-period fragment records appear. Commit, push.
Standing directions (for every step):
- S2 arrives without registers.txt (it needs the station stopped and the owner): take the supply voltage for the fit's torque scale from data/s2/supply_diag.txt (each servo's present-voltage register read by hom_setup.py diag before and after every block; idle 7.4-7.7 V at 19:06), and say so in the QUEUE line
- S1(b) arrives as data/s1/rows.jsonl, one pose per line: commanded and measured joints of all three fingers, the station's mounts, the plan's mounts in P, the tag's measured and predicted centre and normal, the planned camera and view angle
- Night's data (lab, 21:3x; all on origin/main at c03a378d7, details in docs/experiments/20261117-sr2_system_identification/NOTES.md): the boards' sweep (327 poses, 20:10-20:53) in data/sweep/; the boards' readings in the splitter's file logs/20261117-three_finger_mux/20261117-180517-mux.bin (495 MB, gitignored, on the workstation; 18:05-21:3x) with the S2, sweep and S1(a) telemetry beside it; S1(a) in data/s1/20261117-s1a_* (with a stall test every end reached except the low MCPs: index -10.3, middle -8.5 deg); the second A/B scored with the first in docs/experiments/20261115-hom_bench_trials/data/20261117-night_repeats_tracking.json (9 of 12 cycles before the lights went off). Take the lab's scores; do not rescore.
<!-- /ap:steps -->
Claimed: docs/experiments/20261117-sr2_system_identification/ (QUEUE.md W lines, data/, sim/),
scripts/hardware/sr2_servo_sysid_fit.py
Waiting on owner: none
Do not: touch the hand, corvus or the CB1; run GPU jobs; start W4; edit the lab's page or the bench README; commit data
files over 2 MB.
