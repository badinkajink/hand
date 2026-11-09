Status: waiting (not armed; owner 2026-11-09: run after the weekly usage reset)
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
1. MuJoCo 4.0: rerun the four default-object grids on the SCS0009 plant at 0.72 N m/rad and 2.5 MPa (rows tag e.g.
   _sv25_mj4), then Spearman against grid2dff_v4b_sv25 and the best layouts; then the size study's EA at the same sizes.
   Quote every landscape with its MuJoCo version, plant and modulus.
2. The measured servo (0.55 N m/rad at the datasheet's torque, 0.59 deg dead band): a HOS_PLANT variant, the four
   default grids, Spearman against 0.72. Rerun the rest only if the rankings move (owner: stay on 0.72 until then).
3. PSO against CEM for the per-layout refinement (hom_grasp_refine.py): 30 best SCS0009 layouts per task, same rollouts,
   validated at seeds 20-24; CEM stays if PSO's median gain is within 0.02 on every task.
4. The six search methods on the six-coordinate layouts (no full grid): score against the best layout any method found,
   rerun at seeds 20-24.
5. Live EA and PSO searches at new object sizes, three random starts each.
6. Design page lede: still quotes 1.0-1.2 mm per mm from the grids with every grasp; with clear grasps 0.83-1.06 (trends
   page, Table 2). Add the clear-grasp values and rebuild fp_review_page.py.
7. Talk: after any hand page that inserts figures or tables, rerun the FIGS audit before rebuilding the deck; push the
   contact repo only on the owner's word.
Claimed: none.
Waiting on owner: the data-row rule for GitHub's LFS (GH008 blocks pushes of commits with data rows; proposal: upload rows
under 5 MB); plan files to corvus (plain git or the upload set) and the filtered hook there; item 2's go.
Do not: multi-hour GPU grids without the owner's go and a resguard check; git lfs push --all; contact repo pushes without
the owner's word.
