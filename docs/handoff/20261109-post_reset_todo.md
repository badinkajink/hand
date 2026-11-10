Status: armed 2026-11-09 15:4x for Tuesday 2026-11-10 03:05 MST, after the weekly reset (owner: "can you arm it yourself.
reset is at 3am mt tuesday"; "you can stage the execution plan for post weekly reset. i agree with your order of runs").
Owner's go for the GPU grids of items 1 and 2, each launch through resguard after its status check.
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
State 11-10 03:10: step 1 queue logs/20261110-hom_post_reset/gpu_queue_step1.sh running (log gpu_queue_step1.log): twist,
roll, press grids on kp0.55 (rows grid2dff_v4b_sv55), then the 0.72 grids rerun with the tracking fields (grid2dff_v4b_sv25r);
about 2.2 h (tracking metrics add ~45 s of CPU scoring per 2,048-world batch); .done markers per grid, relaunch the
script to resume. Smoke: 2.43 GB RSS, rows carry mujoco 3.6.0. Analysis: hom_plant_transfer.py --set stiffness [--score
track]. Step 6 done 03:3x (74bac6e77).
State 11-10 03:25: step 9 started beside the queue: scripts/hom/hom_torque.py (TorquePlant, inverse dynamics at 1 ms,
wn 25.6 and 63 rad/s), comparison rows docs/experiments/20261110-hom_torque_control/rows/20261110-torque_compare.jsonl
(3 CPU parts, logs torque_compare_*.log) run through a scratch hook; hom_tasks.rollout's set_hom hook and a faster
track_errors wait in logs/20261110-hom_post_reset/apply_hom_tasks_patch.py, to apply when the GPU queue is idle (imports).
State 11-10 03:35: step 9 comparison done (207 rows, 3 diverged press rollouts of (70, 110) at 10-13 mm): tracking score
twist 0.239/0.252/0.234, roll 0.108/0.078/0.047, press 0.381/0.372/0.438 (position / torque wn 25.6 / 63); off-axis
twist 33.8/19.4/11.8 deg/s; torque-limit share twist 0.86/0.83/0.59. Next: page builder scripts/hom/hom_torque_page.py,
films (hom_torque.py film) once the GPU queue is idle. Step 8 waits on the lab session (corvus at eb14ac06d, 11-05).
Step 1 early: 384 twist layouts at 0.55 against 0.72: Spearman 0.72, best tenth kept 0.50 (noise floor from sv25r).
<!-- ap:steps: rendered by ~/.claude/autopilot/ap from its ledger; change it with `ap step`, not by hand -->
Steps, in order (the ledger; `ap step done <job> <id> "result"` marks one, `ap step add <job> "text"` adds one):
- [ ] 1. The measured servo first: HOS_PLANT=kp0.55_kv0_fr0.18_fl0_dp0.011 (damping 0.02 s x kp, as the servo plant's),
    HOM_ROWS_TAG=_sv55, the 2.5 MPa scene cache of the _sv25 grids; the four default-object grids as the _sv25 ones were
    run (logs/20261102-hom_servo_plant/gpu_queue.sh). Spearman and best layouts against grid2dff_v4b_sv25. Add the 0.59 deg
    dead band only if the stiffness alone moves the rankings. Keep 0.72 unless they move (owner, 11-09).
- [ ] 2. MuJoCo 4.0 on the plant that survives item 1: the four default grids (rows tag e.g. _sv25_mj4), Spearman against the
    MuJoCo 3 grids and the best layouts; then the size study's EA at the same sizes. Quote every landscape with its
    MuJoCo version, plant and modulus.
- [ ] 3. PSO against CEM for the per-layout refinement (hom_grasp_refine.py): 30 best SCS0009 layouts per task, same rollouts,
    validated at seeds 20-24; CEM stays if PSO's median gain is within 0.02 on every task.
- [ ] 4. The six search methods on the six-coordinate layouts (no full grid): score against the best layout any method found,
    rerun at seeds 20-24.
- [ ] 5. Live EA and PSO searches at new object sizes, three random starts each.
- [x] 6. Design page lede: still quotes 1.0-1.2 mm per mm from the grids with every grasp; with clear grasps 0.83-1.06… Done 11-10 03:15: Design page lede quotes 1.00-1.21 mm/mm with every grasp and 0.83-1.06 with clear grasps; rebuilt, 74bac6e77
- [ ] 7. Talk: after any hand page that inserts figures or tables, rerun the FIGS audit before rebuilding the deck; push the
    contact repo only on the owner's word.
- [?] 8. Corvus: after its next pull, run python3 scripts/docs/lfs_lean.py install-hook there (relay through the lab s… Waiting on owner: corvus's clone is still at eb14ac06d (11-05 21:03), with no pull since and uncommitted sysid files in the lab's working copy; scripts/docs/lfs_lean.py is not there yet. The install needs the lab session's next pull (no lab session is reachable from here): when a lab session runs, relay 'git pull, then python3 scripts/docs/lfs_lean.py install-hook'.
- [ ] 9. Owner, 11-09: implement, test and analyse inverse-dynamics torque control of the fingers in place of MuJoCo's position actuators ("that might help things?"). Keep Eq. 2's joint velocities u. Drive motor actuators with tau = M(q) qdd_ref + c(q, qd) + g(q) - J^T f_cmd, with qdd_ref from u (or a PD on theta_ref) and f_cmd the commanded pad forces, clipped at the SCS0009's 0.18 N m. Compare it with the position plant (kp 0.72 N m/rad) on the twist, roll and press at the films' sizes (twist and roll 25, 30, 63, 80 mm; press 10, 13, 40, 50 mm), on layout (70, 110) and the per-size best layouts, seeds 0-2: score, contact time per finger, off-axis angular-velocity RMSE. Before any hardware claim, state how a torque reaches the position-only servo (an offset tau/kp through its 0.55-0.72 N m/rad spring and 0.59 deg dead band).
- [ ] 11. The roll at 25, 30 and 80 mm: no candidate of the short lists rolls (tracking score 0.03-0.10, page 20261109-hom_tracking_score.html). Run the live EA for the roll at these sizes on the GPU with hom_landscape.track_score as its fitness (new rows record track_err_rms_deg, track_ref_rms_deg and contact_all_frac). Refilm with scripts/hom/hom_track_films.py if a layout passes 0.2 at a size.
- [ ] 13. Owner, 11-09: the roll is the weakest demonstration because it departs from the HOM paper's video, where the thumb stays extended and rigid and the two other fingers roll the ball up along it. Add that roll to hom_tasks.Schedule as a variant: the thumb holds its extended pose and may lose fingertip contact; the sphere rides on the thumb as a rail (its links may touch the sphere: the CPU plant has every contact, and the GPU filter must admit the thumb's links); index and middle roll it. On the 25 and 30 mm spheres also try one rolling finger. Score with hom_landscape.track_score, its contact share taken over the rolling fingers' pads only. Search 25, 30, 63 and 80 mm on the 2-D lattice (a CPU sample or the live EA on the GPU), film the best from below (HOM_FILM_VIEW=60,55,0.16) and add a section to 20261109-hom_tracking_score.html.
- [ ] 14. Owner, 11-09 (an exception to the pin-push rule, exploratory): push with the unsensorized back of the pusher's fingertip. Estimate where and how hard the back of the pusher touches the cylinder from the pusher's joint kinematics and the sensed thumb and index tips, with the boards' noise (hom_tasks TIP_SIGMA_MM, TIP_NOISE_N), and close the pusher's loop on that estimate. Report the contact-location error (mm) and the force error (N) against the simulator's contacts, and the pin-push tracking score with the estimate against exact sensing.
- [-] 15. Owner, 11-10 ('this is pretty important'): finish and double-check the talk's co-design formulas and visuals,… Dropped: taken back by the window at 01:40 on 11-10 (owner: more effort on the visuals, reorder the deck); the window re-adds a step if it does not finish
Standing directions (for every step):
- Pin-push dropped for now (owner, 11-09: "effective pin-push trajectories are effectively press trajectories"): grids, searches and films run the twist, roll and press.
- Rank layouts by hom_landscape.track_score (owner's go, 11-09; page 20261109-hom_tracking_score.html) and quote the earlier rollout_score beside it. Rows from before 11-09 lack the tracking fields, so a stored landscape is rescored only by rerunning it.
<!-- /ap:steps -->
Claimed (autopilot, 11-10): logs/20261110-hom_post_reset/, scripts/hom/hom_gpu.py (rows name their MuJoCo version),
scripts/hom/hom_plant_transfer.py, rows grid2dff_v4b_sv55 and grid2dff_v4b_sv25r.
Decided 11-09: raw data stays offline (no data rows to GitHub's LFS); data files under 2 MB are ordinary git files, the
pre-commit hook stops larger ones; plans and trajectories converted to plain git; the bare copy hand-lean.git deleted.
Waiting on owner: nothing.
Do not: GPU grids beyond items 1 and 2 without the owner's go, or any launch without a resguard check; git lfs push --all; contact repo pushes without
the owner's word.
