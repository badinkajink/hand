Status: staged, not armed (owner 2026-11-09: "you can stage the execution plan for post weekly reset. i agree with your
order of runs"). Arm it after the weekly reset: ~/.claude/autopilot/ap arm docs/handoff/20261109-post_reset_todo.md
Task: owner 2026-11-09: "we should have a good list of to-do tasks that we can run with full comprehensiveness to get
everything up-to-date post-weekly update"; "part of me also wants to just send it on new mujoco ... save the latter task
for post-weekly reset"; "we previously re-did sysid from 0.72 nmrad to 0.55nmrad, but i dont want to rerun everything, so
we can stick to 0.72 for now".
Read first: docs/experiments/20261029-foundational_pose_review/20261109-hom_search_methods.html (six search methods on the
stored grids), 20261030-hom_layout_trends.html (size study, Table 2: slopes), docs/experiments/20261103-hom_score_refine/
20261103-hom_score_refine.html (CEM refinement), docs/experiments/20261103-hom_task_conflict/20261104-hom_task_conflict.html,
the MuJoCo 4.0 port page 20261108-mujoco4_contact_and_stack.html (INDEX.md), docs/experiments/20261106-sr2_servo_sysid/
20261106-sr2_servo_identification.html, CLAUDE.md (LFS bullet: GH008), memory project_hand_design_talk_2026_11_17.md.
State: 2026-11-09: search methods page and scripts/hom/hom_search_compare.py (replays, no new rollouts); talk deck and guide
updated in ~/wxie_workspace/contact/talks/20261117-hand_design/ (practice 11-11, talk 11-17).
Next:
1. The measured servo first: HOS_PLANT=kp0.55_kv0_fr0.18_fl0_dp0.011 (damping 0.02 s x kp, as the servo plant's),
   HOM_ROWS_TAG=_sv55, the 2.5 MPa scene cache of the _sv25 grids; the four default-object grids as the _sv25 ones were
   run (logs/20261102-hom_servo_plant/gpu_queue.sh). Spearman and best layouts against grid2dff_v4b_sv25. Add the 0.59 deg
   dead band only if the stiffness alone moves the rankings. Keep 0.72 unless they move (owner, 11-09).
2. MuJoCo 4.0 on the plant that survives item 1: the four default grids (rows tag e.g. _sv25_mj4), Spearman against the
   MuJoCo 3 grids and the best layouts; then the size study's EA at the same sizes. Quote every landscape with its
   MuJoCo version, plant and modulus.
3. PSO against CEM for the per-layout refinement (hom_grasp_refine.py): 30 best SCS0009 layouts per task, same rollouts,
   validated at seeds 20-24; CEM stays if PSO's median gain is within 0.02 on every task.
4. The six search methods on the six-coordinate layouts (no full grid): score against the best layout any method found,
   rerun at seeds 20-24.
5. Live EA and PSO searches at new object sizes, three random starts each.
6. Design page lede: still quotes 1.0-1.2 mm per mm from the grids with every grasp; with clear grasps 0.83-1.06 (trends
   page, Table 2). Add the clear-grasp values and rebuild fp_review_page.py.
7. Talk: after any hand page that inserts figures or tables, rerun the FIGS audit before rebuilding the deck; push the
   contact repo only on the owner's word.
8. Corvus: after its next pull, run python3 scripts/docs/lfs_lean.py install-hook there (relay through the lab session),
   so its data files over 2 MB stay out of git as on the workstation.
Claimed: none.
Decided 11-09: raw data stays offline (no data rows to GitHub's LFS); data files under 2 MB are ordinary git files, the
pre-commit hook stops larger ones; plans and trajectories converted to plain git; the bare copy hand-lean.git deleted.
Waiting on owner: nothing.
Do not: multi-hour GPU grids without the owner's go and a resguard check; git lfs push --all; contact repo pushes without
the owner's word.
