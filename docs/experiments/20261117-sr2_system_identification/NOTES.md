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
