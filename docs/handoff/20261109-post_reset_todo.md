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
State 11-10 03:46: step 9 committed 6175313c1 (page without films); hom_ea.py --score track / --method pso ea23c25ad.
Full 0.55 twist grid vs stored 0.72: rho 0.78, best tenth kept 0.50. The hom_tasks patch now also has the rail roll
(prm rail 1/2, step 13), checked on a copy (logs/.../check_patch_*.log: default roll identical, rail 1 and 2 run); apply
with `python3 logs/20261110-hom_post_reset/apply_hom_tasks_patch.py --apply` when the step-1 queue ends (~05:15),
then rerun 1-2 torque rows with the official hook, commit, then launch step 2.
State 11-10 03:58: 0.55 vs stored 0.72 (earlier score): twist rho 0.78, roll 0.84 (same best layout (85, 170)); press
running. Ready in logs/20261110-hom_post_reset/: gap_patch.sh (both patches + import checks), gpu_queue_step1b.sh (dead
band 0.59 deg on 0.55: GPU check of motors vs position actuators, auto-stop on mismatch, then 3 grids _sv55db; a HOLD file
pauses), gpu_queue_step2.sh PLANT TAG [DB] (MuJoCo 4.0 grids + EAs). hom_grasp_refine.py has --method pso and
validate_pso (step 3, untested on GPU). Builders started: hom_stiffness_page.py (steps 1-2, maps only so far),
hom_rail_roll.py (step 13, CPU sample). Step 14 idea: pusher contact unsensed on the current pin-push grasp (no back-contact
grasp design yet; the demo clip's pusher pushes with the back of its tip, friction drive both ways).
State 11-10 04:12: step 9 extended at 0.515 N m (5ef4c5b3e): 63 rad/s loop still loses contact; page done but for films.
0.55 vs stored 0.72 (earlier score): press rho 0.83 too. Toy grids for step 5's new sizes running (cpu_queue_toy.sh:
spheres 28, 67, cylinders 18, 45; 28 done). Code committed f37eb65cf; queues ready: gpu_queue_step1b.sh, _step2.sh,
_step3.sh, cpu_queue_rail_pusher.sh (needs the patch). At the gap (~05:02): gap_patch.sh, step-9 films, then queues.
State 11-10 04:18: gpu_master.sh chains steps 1b, 11, then waits for logs/.../step2.args ("PLANT TAG [DB]", written after
the step-1 analysis), then steps 2, 3, 4 (hom_search6d.py, c466c1ca9) and 5 (EA and PSO at spheres 28/67, cylinders
18/45, seeds 0-2). Partial rerun check: sv25r vs stored sv25 twist rho 0.96 on the first 170 layouts.
State 11-10 05:05: step-1 grids done 04:54. Earlier score: 0.72 rerun vs stored rho 0.96/0.98/0.98 (twist/roll/press), best
tenth kept 0.94; 0.55 vs 0.72 rho 0.79/0.83/0.84, kept 0.48/0.62/0.69 -> stiffness moves the rankings. Tracking score 0.55
vs 0.72: rho 0.91/0.73/0.81 (its noise floor from the partial rerun sv25r2 in step 1b). Patches applied and committed
4880f6e69 (hook reproduces torque rows exactly). Torque films + page 2b0f8de21: step 9 complete -> mark done.
gpu_master.sh running since 04:58 (dead-band check, partial rerun, sv55db grids, step 11, then waits for step2.args).
cpu_queue_rail_pusher.sh running (rail sample/refine); pusher rows (step 14) + tip0/tip0_pe variants; page builder
scripts/hom/hom_pusher_page.py written (rows to move to docs/experiments/20261110-hom_pusher_estimate/rows/).
State 11-10 05:12: steps 9 and 14 done (e0c123fe0: pusher page). --max in hom_gpu counts layouts, so gpu_queue_step1b.sh's
partial rerun would have been a full grid: HOLD file set; step1b stops after the dead-band check, run_1b2.sh then runs
gpu_queue_step1b2.sh (partial rerun --max 200 layouts, sv55db grids) and removes HOLD, and the master goes on to step 11
and waits for step2.args. Expected: sv55db grids done ~06:30.
State 11-10 05:19: dead-band check passed (db_check_compare.txt: motors with a 1e-9 deg band vs position actuators on
8,192 twist worlds: mean 0.078 vs 0.077, 5.5 % of worlds differ by > 0.02, held 5,173 vs 5,178). gpu_queue_step1b2.sh
running since 05:18 (partial reruns, then sv55db grids); step 11 waits on HOLD; stiffness page sections drafted.
State 11-10 06:22: sv55db grids done 06:20. Tracking-score rho, best of 16: 0.72 rerun on 200 layouts 0.999/0.987/0.997
(twist/roll/press); 0.55 vs 0.72 0.91/0.73/0.81; dead band vs 0.55 alone 0.94/0.79/0.90. So step2.args = measured servo
with the band (kp0.55_kv0_fr0.18_fl0_dp0.011 _sv55db 0.59); step 11 (roll EA, tracking fitness) on the GPU since 06:20,
step 2 follows it. Rail roll refined (rail 1: 0.28/0.32/0.15/0.29 at 25/30/63/80 mm; rail 2: 0.04/0.12). Films of the rail
roll and of the stiffness page render on Mesa's llvmpipe through EGL (__EGL_VENDOR_LIBRARY_FILENAMES=.../50_mesa.json,
EGL_PLATFORM=surfaceless), off the GPU.
State 11-10 06:26: step 11's roll EA at 25 mm done (incumbent (90, 60) 0.29 on the GPU's tracking score); 30 mm running.
cpu_step11_check.sh SIZE (one per size, each waits for its EA) reruns the EA's six best on the CPU at seeds 0-2
(hom_track_films.py rerun roll with TRACK_EA=1). Stiffness page builds within the prose targets but for films and MuJoCo 4.
State 11-10 06:40: step 13 done (dba1c25f5). Step 11's EA and CPU checks done: new roll picks (90, 60) 0.28 at 25 mm
(was 0.23) and (120, 155) 0.22 at 80 mm (was 0.04); 30 mm keeps (125, 55) 0.28 (EA's best 0.26). Roll refilm running
(film_track_roll.log; picks backed up in logs/.../20261109-track_picks_sv25.before_step11.json); tracking page text for it
written. Stiffness clips re-rendering with shorter labels, then commit step 1. Step 2 started 06:33 (MJ4 smoke passed with
the dead band; twist grid). Step 7 audit script: logs/20261110-hom_post_reset/figs_audit.py (read-only; 0 moved at 06:2x).
State 11-10 06:53: step 11 done (651eaa1f3: roll EA picks at 25/80 mm, refilm, tracking page). Stiffness clips
re-rendered with short labels; step 1 commit next (page + indices). Step 2: MJ4 twist grid done 06:50 (63 vs 71 s per
2,048-world batch on 3.6), roll grid running; then press, then the EA at the size study's sizes.
State 11-10 07:24: step 1 done (2eb2bfe31, page 20261110-hom_servo_stiffness_mujoco4.html). Step 2 grids done 07:13:
MJ4 vs MJ3.6 on the measured servo, tracking rho 0.97/0.94/0.95, same best layouts; builder and data committed
3d162ccf1 (page rebuilt but not committed until the EA ends). Step 2's 26 EA runs (ea_live_sv55db_mj4) since 07:13,
~09:15 done; then steps 3, 4, 5 run from gpu_master.sh. MJW 4.0 logs ~100 MB a run (warnings): gzip them after.
Refinement page builder has sec_pso (step 3) waiting for val_pso rows.
State 11-10 09:27: step 2 EA done 09:22 (26 runs, no failures, logs gzipped): best earlier score within 0.03 of the stored
runs at 21 of 26 sizes; near-set x_sep slope 0.98/0.49/1.09 mm/mm (twist/roll/press) vs 1.42/0.56/1.09 on the model
servo; page Section 8 + Figure 3 (20261110-ea_size_slopes.png). Default sizes (40, 25 mm) left to the full grids. Step 3's
smoke failed 09:22 (negative RNG seed in cmd_run_pso; fixed e7be93d39, CPU path checked on 2 layouts per task); step 3's
queue now runs at the head of gpu_queue_step5.sh, after step 4 (search6d twist running since 09:22).
State 11-10 09:31: step 2 marked done. Step 4 twist at round 7 (09:30): random/Sobol spend ~190 rollouts a round, the EA
~50, PSO 16 ~35, so ~1-2 h per task; then step 3 (~1 h), then step 5 (36 runs, ~3 h). hom_search_methods_page.py has
sec_search6d (summary rows/20261110-search6d_sv25_summary.json, figure 20261110-search6d_curves.png, table), inert until
the _gens files land; its prose waits for the data. Claimed line extended (09:28) to the two page builders of steps 3-5.
State 11-10 11:16: step 4's searches ended 11:03 (all three tasks validated). The EA and PSO 16 stalled at the 400-round
cap after 324-630 rollouts (children snap back to visited Sobol layouts, median spacing 0.19); RRT within 0.05 from 8 of
9 starts, random 7, EA 4; roll incumbents track 0.00-0.04 (rho earlier vs tracking 0.40 on the roll). Page section
written (search methods page Section 10), not yet committed. Step 18 added: hom_search6d.py validate --track-top 5
(gpu_queue_step4b.sh, launched detached at 11:10, waits for "gpu master done"). Step 3 (PSO vs CEM) on the GPU since
11:04 (smoke passed). .gitignore covers search6d *_d[0-9][0-9].jsonl and step 5's per-run rows.
State 11-10 11:21: step 4 committed 5d8477f06 and marked done. Step 3's twist swarm at iteration 5 of 8 at 11:19
(~2.5 min an iteration), then its validation, the roll and the press (~12:10). sec_pso of hom_score_refine_page.py
applies the rule on the earlier score and quotes the tracking score; rebuild it when pso_val_press.done appears.
State 11-10 11:58: step 3 done 11:53, committed ad2fa44bb: the swarm's median gain over S1 0.24 vs CEM's 0.20 on the
press (22 of 30 layouts above by >0.02, none below), within 0.02 on twist and roll; the press fails the rule. Refinement
page Section 9, lede, a future item on refining the tracking score; search methods page quotes it in Section 7. Step 5
(36 hom_ea.py runs) started 11:53 with ea twist 28 seed 0; ~5 min a run, ~15:00 done; then step 18's queue.
State 11-10 13:18: step 5's 12 runs at 28 mm done 13:11 (6.5 min a run; the rest ~15:50). Runs that shared a layout
scored it differently (twist 28: 216 of 726 shared pairs differ by >0.02; hom_gpu seeds the per-period noise per
batch). New scripts/hom/hom_ea_validate.py reruns each run's incumbent and the top 5 pooled layouts by each score at
seeds 20-24 (whole grasp sets, 800 rollouts per task and size); gpu_queue_step5b.sh runs it after "step 4b queue
done". The section waits for those rows; s5_summary needs them added.
<!-- ap:steps: rendered by ~/.claude/autopilot/ap from its ledger; change it with `ap step`, not by hand -->
Steps, in order (the ledger; `ap step done <job> <id> "result"` marks one, `ap step add <job> "text"` adds one):
- [x] 1. The measured servo first: HOS_PLANT=kp0.55_kv0_fr0.18_fl0_dp0.011 (damping 0.02 s x kp, as the servo plant's)… Done 11-10 06:53: Measured servo moves the rankings: tracking-score rho vs 0.72 N m/rad 0.91/0.73/0.81 (0.55) and 0.91/0.72/0.79 (0.55 + 0.59 deg band), band vs 0.55 alone 0.94/0.79/0.90, rerun 0.99-1.00 (twist/roll/press); model's best layouts lose 0.14-0.29 on the measured servo; step 2 runs on 0.55 + band; page 20261029-foundational_pose_review/20261110-hom_servo_stiffness_mujoco4.html (2eb2bfe31)
- [x] 2. MuJoCo 4.0 on the plant that survives item 1: the four default grids (rows tag e.g. _sv25_mj4), Spearman agai… Done 11-10 09:27: MuJoCo 4.0 on the measured servo (0.55 N m/rad + 0.59 deg band, 2.5 MPa): grids keep tracking rho 0.97/0.94/0.95 vs 3.6 with the same best layouts; size-study EA at 26 sizes reaches the stored best within 0.03 at 21, near-set slopes 0.98/0.49/1.09 vs 1.42/0.56/1.09 mm/mm; page 20261110-hom_servo_stiffness_mujoco4.html Sections 7-8 (f933beb09)
- [x] 3. PSO against CEM for the per-layout refinement (hom_grasp_refine.py): 30 best SCS0009 layouts per task, same r… Done 11-10 11:58: PSO (32 particles, Clerc-Kennedy, CEM's rollouts) vs CEM at level full on the 30 best SCS0009 layouts, seeds 20-24: median gain over S1 press 0.24 vs 0.20 (22/30 layouts better by >0.02, 0 worse), twist 0.10 vs 0.12, roll 0.04 vs 0.05; the press fails the rule, so CEM does not keep its place there; tracking: -0.03/+0.03/+0.07; refinement page Section 9 (commit ad2fa44bb)
- [x] 4. The six search methods on the six-coordinate layouts (no full grid): score against the best layout any method… Done 11-10 11:20: Six methods live on the 8,192 six-coordinate Sobol layouts (SCS0009 0.72, 2.5 MPa, MJ 3.6): best validated 0.87/0.59/0.83 (twist/roll/press, seeds 20-24), all among random search's incumbents; within 0.05 from RRT 8/9 starts, random 7, ring 6, PSO16 5, EA 4; EA and PSO16 stalled at the 400-round cap after 324-630 rollouts (children snap to visited layouts); roll incumbents track 0.00-0.04; search methods page Section 10 (5d8477f06)
- [ ] 5. Live EA and PSO searches at new object sizes, three random starts each.
- [x] 6. Design page lede: still quotes 1.0-1.2 mm per mm from the grids with every grasp; with clear grasps 0.83-1.06… Done 11-10 03:15: Design page lede quotes 1.00-1.21 mm/mm with every grasp and 0.83-1.06 with clear grasps; rebuilt, 74bac6e77
- [ ] 7. Talk: after any hand page that inserts figures or tables, rerun the FIGS audit before rebuilding the deck; push the
    contact repo only on the owner's word.
- [?] 8. Corvus: after its next pull, run python3 scripts/docs/lfs_lean.py install-hook there (relay through the lab s… Waiting on owner: corvus's clone is still at eb14ac06d (11-05 21:03), with no pull since and uncommitted sysid files in the lab's working copy; scripts/docs/lfs_lean.py is not there yet. The install needs the lab session's next pull (no lab session is reachable from here): when a lab session runs, relay 'git pull, then python3 scripts/docs/lfs_lean.py install-hook'.
- [x] 9. Owner, 11-09: implement, test and analyse inverse-dynamics torque control of the fingers in place of MuJoCo's… Done 11-10 05:03: Inverse-dynamics torque plant (hom_torque.py) vs SCS0009 position servos on the films' 23 configurations: tracking 0.24/0.25/0.23 twist, 0.11/0.08/0.05 roll, 0.38/0.37/0.44 press (position / 25.6 / 63 rad/s); off-axis twist 34->12 deg/s but contact lost; same at 0.515 N m; hardware path stated; page docs/experiments/20261110-hom_torque_control/20261110-hom_inverse_dynamics_control.html (2b0f8de21)
- [x] 11. The roll at 25, 30 and 80 mm: no candidate of the short lists rolls (tracking score 0.03-0.10, page 20261109-… Done 11-10 06:42: Roll EA with track_score fitness at 25/30/80 mm (GPU), six best each rerun on the CPU at seeds 0-2: new picks (90, 60) 0.28 at 25 mm (was 0.23) and (120, 155) 0.22 at 80 mm (was 0.04, three pads 0.46); 30 mm keeps (125, 55) 0.28 (EA best 0.26); roll refilmed; 20261109-hom_tracking_score.html Section 4 (651eaa1f3)
- [x] 13. Owner, 11-09: the roll is the weakest demonstration because it departs from the HOM paper's video, where the … Done 11-10 06:28: Rail roll (thumb held as a rail, CPU, 2,688 + 2,820 rollouts): index+middle 0.28/0.32/0.15/0.29 at 25/30/63/80 mm vs tripod picks 0.23/0.28/0.32/0.04; index alone 0.04 (25) and 0.12 (30); films from below; section 6 of 20261109-hom_tracking_score.html (dba1c25f5)
- [x] 14. Owner, 11-09 (an exception to the pin-push rule, exploratory): push with the unsensorized back of the pusher'… Done 11-10 05:07: Unsensed pusher on 6 pin-push layouts x 5 seeds: point within 0.22 mm, force within 0.25 N of 0.45 N (boards' noise); tracking 0.34 exact, 0.23 three boards, 0.27 pusher estimated; at 50 Hz/40 ms late held 9 vs 7 of 30; pad contact with reading withheld, back-contact grasp is future work; page docs/experiments/20261110-hom_pusher_estimate/20261110-hom_unsensed_pusher.html (e0c123fe0)
- [-] 15. Owner, 11-10 ('this is pretty important'): finish and double-check the talk's co-design formulas and visuals,… Dropped: taken back by the window at 01:40 on 11-10 (owner: more effort on the visuals, reorder the deck); the window re-adds a step if it does not finish
- [ ] 18. After step 4: rerun at seeds 20-24 the five layouts of best tracking score at seeds 0-2 among every layout step 4's searches ran (hom_search6d.py validate --track-top 5, logs/20261110-hom_post_reset/gpu_queue_step4b.sh, starts when gpu_master.sh ends) and add them to the search methods page's six-coordinate section; the searches' fitness was the earlier score, and the roll's incumbents track at 0.00-0.04
Standing directions (for every step):
- Pin-push dropped for now (owner, 11-09: "effective pin-push trajectories are effectively press trajectories"): grids, searches and films run the twist, roll and press.
- Rank layouts by hom_landscape.track_score (owner's go, 11-09; page 20261109-hom_tracking_score.html) and quote the earlier rollout_score beside it. Rows from before 11-09 lack the tracking fields, so a stored landscape is rescored only by rerunning it.
<!-- /ap:steps -->
Claimed (autopilot, 11-10): logs/20261110-hom_post_reset/, scripts/hom/hom_gpu.py (rows name their MuJoCo version),
scripts/hom/hom_plant_transfer.py, rows grid2dff_v4b_sv55 and grid2dff_v4b_sv25r; also (06:25) scripts/hom/hom_tasks.py,
hom_torque*.py, hom_pusher*.py, hom_rail_roll.py, hom_ea.py, hom_grasp_refine.py, hom_search6d.py, hom_track_films.py,
hom_track_score_page.py, hom_stiffness_page.py, src/sr2hand/geometry/rotations.py (quat_integrate), the pages these
builders write, and the post-reset block of .gitignore; also (09:28, the sections of steps 3-5)
scripts/hom/hom_score_refine_page.py, hom_search_methods_page.py and hom_search_compare.py; also (13:18, step 5's
validation) the new scripts/hom/hom_ea_validate.py.
Decided 11-09: raw data stays offline (no data rows to GitHub's LFS); data files under 2 MB are ordinary git files, the
pre-commit hook stops larger ones; plans and trajectories converted to plain git; the bare copy hand-lean.git deleted.
Waiting on owner: nothing.
Do not: GPU grids beyond items 1 and 2 without the owner's go, or any launch without a resguard check; git lfs push --all; contact repo pushes without
the owner's word.
