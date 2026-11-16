Copies of the simulation scripts of 2026-11-16's sim-to-real study, from logs/20261116-sim_real_reversal/ on the
workstation (gitignored), where they run: `cd logs/20261116-sim_real_reversal && OMP_NUM_THREADS=1
~/Programs/hand/.venv/bin/python <script> ...` (each script's docstring gives its command). They import each other by
bare name and sr2_hom_replay, hom_bench and hom_tasks from the repository. rev_closed.py, rev_stem.py and rev_sweep.py are
the corvus session's helper's; the others are hand-4a's. The rows they wrote are in the folder above.

- rev_rescore.py, rev_rescore_report.py, rev_rank100_report.py: the twist grasps re-scored on the real object
  (rescore_grids_talk2_summary.json, rank100_summary.json)
- rev_lat.py: grasp 10's latitude variants (lat_g10.jsonl, tau_lat_g10.jsonl)
- rev_joint_gaps.py, rev_joint_gaps2.py, rev_inputs.py, rev_trace_one.py: joint gaps, dead band and joint friction, HOM's
  inputs as on the hand, one run's joint trace
- rev_free_air.py, rev_joint_sag.py: the free-air sag steps (free_air_middle_yaw.json, joint_sag_sim.json)
- rev_fric.py, rev_fric2.py, rev_kp.py: constant friction and softer gains (fric_soft.json, fric2_soft.json, kp_soft.json)
- rev_gear.py: friction c x |servo torque| (gear_soft.json)
- rev_clutch.py (a kinematic reset; it broke the grasps), rev_clutch2.py: servos the load cannot backdrive, with
  `--stiff` for a joint friction that does not creep (clutch2_soft.json, clutch2_stiff_soft.json);
  rev_trace_clutch2.py: one run's joint trace on that plant
