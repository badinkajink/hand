Status: done
Task: (owner, 2026-11-03 11:0x) "repo clean up + refresh on the README", "the absolutely abysmal scripts folder", and
"rewriting rotations, geometry, math helpers everywhere with custom math instead of using proven library calls / a
single help file(s)". Owner's choices at 11:5x: ten topic folders, switched by this autopilot after the day's lab work;
shared math modules with the live copies migrated; the 48 unreferenced scripts archived. At 12:4x: "change any
mention of morphohand to sr2 hand, or even just hand where appropriate" (the package becomes sr2hand in the same
switch), and no new unit tests ("very hesitant now to run endless unit tests that exist only so that you can pass them").
Read first: docs/notes/20261103-scripts_cleanup/RUNBOOK.md (conditions, commands, expected checks, migration
procedure, bugs), audit_rotations.md and audit_geometry.md beside it, the lab's
docs/experiments/20261022-ur5_tip_calibration/lab/STATE.md, `~/.claude/autopilot/ap status`, ~/.claude/writing_reference.md.
State: 11-03 13:1x. Committed by the window: README and .gitmodules (6eabaf56b), the rotations module (ab59888dd;
the ledger's 529a91a48 is the same commit before its message was corrected), pyserial in the dev extra (1feafb8b1),
"SR² Hand" in the page headers and CLAUDE.md (cdf91ad3f), the tests cut to 10 files (fd2100da0: 114 pass, the floor
datum fails and waits on the owner). move_scripts.py now also renames the package morphohand to sr2hand (RUNBOOK, "The
package rename"). Dry run of the move and the rename at 4f54fc805 in two worktrees: 754 files moved, 252 rewritten for
the package, 2,033 script imports resolve, no live import of morphohand, the same 15 known problems, and pytest
identical in both layouts (108 passed, 7 failed in worktrees). Nothing is switched yet. The rule below on tests
overrides the test parts of steps 5 and 8.
State 11-05 01:1x (autopilot), done. Commits on main: the switch a41ffa27b at 00:50 (787 files in ten folders and archive, package sr2hand, venv
reinstalled; second pass move_extra_live_code.py for 49 lab, study and poster files, .gitignore and the workspace), lab bullet 0e3a94041, the 23
migration commits e7040b5b4..c86d3fe66 at 00:52 with no conflict, Spearman rerun and three pages f15190386, wrap180 73c07869a, CLAUDE.md's line, and
scripts/README.md with the runbook's outcome b35a4093f. The crosstalk job was re-armed with its rule step 53 and runs.
Copies removed per commit (function definitions removed; lines added/removed): src 1 (20/51); contact_surface 0 (6/12); design 1 (8/13); docs 0 (4/7);
hardware 1 (19/41); cad 0 (21/54); mechanisms 7 (37/105); contact 3 (40/72); rl and src/sr2hand/rl 5 (92/109); fingertip 6 (78/205); ur5 11 (81/181);
hom and src/sr2hand/control 18 (110/295); the audits' prose-listed copies 3 (49/65); hom unwraps 0 (9/4); shared additions 0 (8/0). Total 56 definitions,
582 lines added, 1,214 removed, besides the copies written inline. Bugs fixed, one commit each: plant_drop_gate's yaw, ur5_tip_calibration_film's quat,
holder_paths.rot_from_to, probe_real_v1_chain._align (removed), terms_common._seg_seg_dist, policy_cpu_rollout.seg_seg, the four ordinal Spearman
copies, build_cad_v1_finger.unit. Reruns: hom_plant_transfer's five sets (largest rho change 0.001 plants and sts_modulus, 0.008 modulus, 0.010 factorial,
0.062 sts_friction, a grid with many tied scores). plant_drop_gate's yaw changes 20260902-residual-bench and plant_ranking_test on a rerun; neither was
rerun.
Checks after the landing: check_scripts the 15 known problems; pytest 114 passed + the floor datum, as before; 0 undefined names; 0 attribute misses
against the switch's tree; smoke 125 of 130, the other 5 not from the move (DLO_REPO, newton, three contact_surface modules that import as a package).
HOM rollout before and after (logs/20261103-scripts_cleanup/hom_rollout_*.json, sv25, seed 0): twist layout 304 is identical in every row and scores
1.0. Press layout 170 scores 0.891 in both, and its rows differ from row 87 by up to 2 um and 0.7 mm/s. A move of the open pose by 1e-14 rad on the old code
gives the same spread (2 um, 0.67 mm/s), and reverting hom.py alone restores identity: rounding in hom.py's helpers, amplified by contact.
Publish pending (rebuilt locally; this hand-off does not republish): docs/experiments/20260906-screen_vs_chain/20260906-screen_vs_chain.html
(https://claude.ai/code/artifact/28a84ea8-a8b2-405e-aea6-e6a626baa3b0); docs/experiments/20261029-foundational_pose_review/20261102-hom_plant_modulus.html
and 20261029-foundational_pose_hand_design.html, which have no artifact in INDEX.md.
Checks and edit scripts: logs/20261103-scripts_cleanup/equiv/ (oldcode.py loads a copy from 91ab66052; check_*.py, run_all.sh, migrate_*.py,
fix_bugs.py, smoke.sh, undefined.py, attrs.py, survivors.py, show.py, rows.py, edit.py). attrs.py found six calls M.name whose target the
migration had removed (B.heat in the contact beds, G.fib_cap in the contact overview, B.perp in the CAD page); fixed in 92d136d5e and 517cd84f5.
Kept on purpose, for the report: bench/tags.py (hardware file shipped alone), fingertip_crosstalk.py and the crosstalk job's claimed files,
reference_trajectory._slerp (sign seen by policies), real_v1_d1_continuation.vertical_cos (JAX), the warp kernels, _corner_dirs (another
lattice), area samplers with their own draws, Drake field classes and dipoles (physics), CAD-kernel wrappers, swept clearances, tube mesh,
R.T @ (p - o) one-liners, ur5_tip_calibration_lab_page.ref_rows (audit bug 6, unchecked against the presses), contact_surface
Surface.parameters (it reports both area conventions), wilson (z = 1.96, counts) and clopper_pearson (over scipy's beta), probe_thumb_reach
dist_to_segment (returns the closest point), the textbook inertia one-liners of hom_contact_rig, contact_surface.mujoco_model,
apply_measured_plant and newton_pinch_probe, and the three heat colour maps with their own palettes. New finding: trimesh second_moments
adds a clockwise hole's moments (mesh.polygon_section works around it).
<!-- ap:steps: rendered by ~/.claude/autopilot/ap from its ledger; change it with `ap step`, not by hand -->
Steps, in order (the ledger; `ap step done <job> <id> "result"` marks one, `ap step add <job> "text"` adds one):
- [x] 1. Wait for the switch conditions: run `bash docs/notes/20261103-scripts_cleanup/wait_switch_conditions.sh 230` … Done 11-05 00:43: Conditions held at 00:42 on 11-05: fingertip-shape rerun done, no other job running, 45 quiet minutes under scripts/, the lab and the SR2 folder; STATE.md: the hand stopped until the owner re-routes the index cable, no session runs, R33 and the layout sweep wait for the owner
- [x] 2. Switch per RUNBOOK "The switch": stop the crosstalk job and its watcher, `git pull --no-rebase`, run move_scr… Done 11-05 00:50: a41ffa27b pushed 00:50 on 11-05: 787 files in ten folders and archive, package sr2hand, venv reinstalled; check_scripts 790 files, 2,242 imports, the 15 known problems; pytest 114 passed + floor datum before and after; second pass move_extra_live_code.py for 49 lab, study and poster files, .gitignore and the workspace
- [x] 3. After the push, per RUNBOOK: the bullet in the lab's STATE.md (commit, push), the crosstalk job's rule step a… Done 11-05 00:51: Lab STATE bullet 0e3a94041 pushed; the crosstalk job has its rule step 53 and is armed --at now; memory project_scripts_topic_folders_2026_11_03 with its MEMORY.md line
- [x] 4. src/morphohand/geometry/rotations.py with tests/test_geometry_rotations.py, per RUNBOOK "Shared math modules"… Done 11-03 11:55: Done by the arming window 11-03 12:1x: 529a91a48 src/morphohand/geometry/rotations.py (51 functions over scipy) and tests/test_geometry_rotations.py (15 passing with scipy 1.18 and with scipy 1.12/numpy 1.26), CLAUDE.md and AGENTS.md rule
- [x] 5. The other modules (geometry vectors, distances, sdf, mesh, sampling; numerics stats, signal, angles), each wi… Done 11-03 23:52: 7467a2e99 on main: geometry vectors, distances, sdf, mesh, sampling and numerics stats, signal, angles under src/morphohand/ (renamed by the switch), plus quat_twist, rot_slerp, adjoint in rotations; each checked once in logs/20261103-scripts_cleanup/equiv/check_*.py. No script calls them until the landing.
- [x] 6. Migrate the copies folder by folder per RUNBOOK "Migration", with the 1e-12 checks, one commit per folder; no… Done 11-05 00:58: e7040b5b4..af6a143d4, 83aeae6cd, 32c241ff8, c86d3fe66 on main at 00:52: one commit per folder, check_scripts 15 known problems, pytest as before, 0 undefined names, 0 attribute misses, smoke 125 of 130 (5 not from the move), twist rollout identical, press within the 1e-14 sensitivity
- [x] 7. Fix the bugs in RUNBOOK's table, one commit each, naming the pages and runs whose numbers change; rebuild tho… Done 11-05 01:02: ae3d6115d..770886c9c, 6aa06d6b2 on main, one commit per bug; hom_plant_transfer reran, plant-modulus, foundational-pose and screen-chain pages rebuilt locally; largest rho change 0.062 on a tied friction grid; plant_drop_gate's yaw names 20260902-residual-bench and plant_ranking_test, not rerun
- [x] 8. tests/test_no_private_math.py with its allow-list, and the section in CLAUDE.md and AGENTS.md. Done 11-05 01:02: No test file, per the owner's direction; the section in CLAUDE.md and AGENTS.md says the hand-off migrated the live copies
- [x] 9. Report in State: the commits, copies removed per folder, bugs fixed; regenerate scripts/README.md (scripts/do… Done 11-05 01:03: Reported in the note's State; scripts/README.md regenerated b35a4093f; worktree, branch and base_tree removed
- [x] 17. Search the scripts added since 11-03 for private copies of rotations, frames, Spearman, smoothing and angle w… Done 11-05 01:02: 31 scripts added since 11-03 searched; wrap180 in cad_v3_page migrated; no other private copy
Standing directions (for every step):
- The lab's STATE.md (docs/experiments/20261022-ur5_tip_calibration/lab/STATE.md), step 3: the switch's change there is ONE bullet appended at the end of its section 'Where things are' (the commit, the new paths of the scripts the lab runs), and nothing elsewhere in the file. The lab writes STATE.md during runs, and a one-bullet change merges cleanly (lab, 12:04 on 11-03, relayed by the coordinator window).
- Owner 11-03 12:4x: no new test files. The modules of step 5 are checked once against scipy, trimesh or MuJoCo in scratch scripts under logs/20261103-scripts_cleanup/equiv/, not in tests/; step 8 writes only the section in CLAUDE.md and AGENTS.md, and no tests/test_no_private_math.py.
- Owner 11-03 12:4x: the switch (step 2) also renames the package morphohand to sr2hand; move_scripts.py does it after the scripts move (RUNBOOK, 'The package rename'): imports, dotted names, paths, MorphoHandEnvCfg to HandEnvCfg, pyproject.toml and uv.lock, the venv's editable install. Check '.venv/bin/python -c "import sr2hand"' after it. External names stay (Lightning Grasp's and GraspGenX's morphohand assets, the W&B project morphohand-rl), and the CB1 keeps src/morphohand/.
- Window 220623e7 (HOM score levels, 11-03 18:27): the queue logs/20261103-hom_score_refine/queue.sh runs on the GPU until about 04:30 on 11-04; its batch children are spawned processes that import scripts/hom_gpu.py, hom_tasks.py, hom_ea.py, hom_landscape.py, fp_sr2_toy.py and src/morphohand by path at every batch. Do not switch while it runs: pgrep -f '[h]om_score_refine/queue.sh' must print nothing (its python steps show as scripts/hom_grasp_refine.py, which wait_switch_conditions.sh sees, but the queue pauses between steps). Put scripts/hom_grasp_refine.py and its page builder scripts/hom_score_refine_page.py (+ .template.html) under hom in plan.json. The autopilot jobs hand-20261104-hom_* that follow it edit scripts/hom_*: migrate scripts/hom/ when none of them is running (ap status).
- Owner 2026-11-03 19:3x (relayed by window 92ca1bc5): this cleanup is lower priority than the fingertip-shape rerun, hand-20261103-fingertip_shape_rerun, which is armed after the HOM chain. Do not start step 2 (the move of scripts/, src/ and tests/) until ap status shows that job done or stopped: it runs scripts/tip_codesign_*.py, curved_tips_*.py, tpu_tip_magnet_sensing.py and the HOM scripts by path.
- New HOM script of 11-03 20:5x (window 220623e7): scripts/hom_task_conflict.py, and scripts/hom_task_conflict_page.py when it appears, go under scripts/hom/ with the other hom_* scripts (add them to plan.json); the job hand-20261104-hom_task_conflict claims them and starts at 09:15 on 11-04 or later, after the page job.
- Autopilot 00:1x on 11-04, from direction 14: step 1's deadline of 09:00 on 11-04 no longer applies, because the switch waits for hand-20261103-fingertip_shape_rerun. This job is armed --after that job. Once a session has started after it, if the lab is still not idle 24 hours later, ap step wait with what the lab is doing.
<!-- /ap:steps -->
Claimed: nothing; the job is done.
Waiting on owner: the floor datum of src/sr2hand/bench/tags.py (85.0 mm beside the comment "measured 65.0"), the one failing test; the global
~/.claude/CLAUDE.md still names scripts/texsvg.py and scripts/docs_index.py (now under scripts/docs/); migrating src/sr2hand/driver/ and the UR5 programs that move the robot (ur5_inverted_rig.py,
the plan generators, ur5_tip_calibration_holder_paths.py) needs the lab's simulator and the owner.
Do not: switch while the lab runs or another window has uncommitted script changes; push anything other than this
task's commits; edit the crosstalk job's files beyond the switch's renames; republish artifacts from this login.
