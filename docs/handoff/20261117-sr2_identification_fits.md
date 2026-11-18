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
- The lab (session "Physical model vs GP calibration", hand-18) runs S2 (the 11-06 free-air chain, ends ~19:40), S1(b)
  (thumb and middle DIP tags, nine poses each; index later), the A/B g0 repeats at (70, 110) ~19:50-21:00 (servers
  47051 tips off base, 47057 tips off --filter-step-s 0.002, 47052 tips on base, 47058 tips on + --contact-off 0.10;
  labels g0x70y110_tips{off,on}_{base,fstep,fstep_hyst}), then the board sweep. Asked to push S2 to data/s2/ and S1(b)
  to data/s1/ (19:2x).
<!-- ap:steps: rendered by ~/.claude/autopilot/ap from its ledger; change it with `ap step`, not by hand -->
Steps, in order (the ledger; `ap step done <job> <id> "result"` marks one, `ap step add <job> "text"` adds one):
- [x] 1. When S2's tables are under data/s2/ (dwells.tsv, holds_c.tsv, registers.txt, the plan steps): give scripts/ha… Done 11-17 19:38: Done in the window 19:4x: data/s2/20261117-sr2_servo_7v5_fit.json; QUEUE W2 line
- [x] 2. When S1(b) is under data/s1/: per pose the tag centre's and normal's error against the model at the station's… Done 11-17 19:42: Thumb done in the window 19:5x (data/s1/20261117-thumb_zero_fit.json, QUEUE W2 S1(b)); middle not detected, index not attempted by the lab
- [ ] 3. When the A/B repeats are pushed (runs/ and runs_corvus/ 20261117-19*, 20*): score each with scripts/hardware/hom_trial_tracking.py; one table per arm (tracking, twist at the end, drift, rise, tilt), the base arms against data/20261117-g0_x70y110_twist_predictions.json; QUEUE line (W3). Commit, push.
Standing directions (for every step):
- S2 arrives without registers.txt (it needs the station stopped and the owner): take the supply voltage for the fit's torque scale from data/s2/supply_diag.txt (each servo's present-voltage register read by hom_setup.py diag before and after every block; idle 7.4-7.7 V at 19:06), and say so in the QUEUE line
- S1(b) arrives as data/s1/rows.jsonl, one pose per line: commanded and measured joints of all three fingers, the station's mounts, the plan's mounts in P, the tag's measured and predicted centre and normal, the planned camera and view angle
<!-- /ap:steps -->
Claimed: docs/experiments/20261117-sr2_system_identification/ (QUEUE.md W lines, data/, sim/),
scripts/hardware/sr2_servo_sysid_fit.py
Waiting on owner: none
Do not: touch the hand, corvus or the CB1; run GPU jobs; start W4; edit the lab's page or the bench README; commit data
files over 2 MB.
