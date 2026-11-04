Status: active
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
State 11-04 00:1x (autopilot): nothing is switched, and the switch now waits for the fingertip-shape rerun. Direction 14 (owner, 19:3x on
11-03, relayed) forbids step 2 until `ap status` shows hand-20261103-fingertip_shape_rerun done or stopped; that job runs after the HOM
chain. This job is armed --after it, so the kit starts a fresh session when it ends; the session of 11-03 ended itself to spare the login.
When a session starts here: `ap status`; `bash docs/notes/20261103-scripts_cleanup/wait_switch_conditions.sh 230` in the background with
`ap snooze <job> 240 "..."` (it checks the rerun, other running jobs and every queue under logs/, and holds until 19:00 unless a second
argument says otherwise); the lab's STATE.md; `python3 docs/notes/20261103-scripts_cleanup/move_scripts.py docs/notes/20261103-scripts_cleanup/plan.json --dry-run`
for scripts that are not in the plan (hom_task_conflict_page.py and scripts/fingertip_shape_rerun_*.py will be new); RUNBOOK "The switch";
RUNBOOK "Landing the migration after the switch". Use .venv/bin/python -m pytest with PYTEST_DISABLE_PLUGIN_AUTOLOAD=1, not uv run. Dirty
docs/handoff notes can stop `git pull`: stash those paths, pull, pop.
On main (7467a2e99): the modules of step 5 under src/morphohand/geometry and src/morphohand/numerics, and the section of CLAUDE.md and AGENTS.md;
no script calls them yet, and the switch renames them. Step 8 stays open until CLAUDE.md's "migrates the live copies once ..." becomes
"migrated the live copies." after the landing.
Steps 6-7 are prepared in the worktree /home/humanoid/Programs/hand_wt_migrate, local branch scripts-migrate (never pushed), above a rehearsal
of the switch (91ab66052, never picked). land_migration.sh picks 23 of its 25 commits, one per folder and one per bug, and keeps HEAD's module
files; it skips 9934b3257 and 1e33feb91, whose content is on main. The rehearsal of the landing ended with an empty diff against the branch.
Conflicts are expected only where someone edited the same lines after dc1830409. The HOM commits 72c82f2e3, f719012cd and 517cd84f5 wait if a
hom job runs. Before 72c82f2e3 record one CPU HOM rollout on the moved tree and compare after it. Mark steps 6-7 done only when their commits
are on main. After Spearman lands: rerun hom_plant_transfer --set for the JSON, rebuild the plant-modulus and foundational-pose pages and the
screen chain page locally; list them under Publish pending. plant_drop_gate's yaw (0bb15c5d2) names 20260902-residual-bench and
plant_ranking_test, not rerun. Then regenerate scripts/README.md, remove the worktree, the branch and logs/20261103-scripts_cleanup/base_tree,
and report per step 9.
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
- [ ] 1. Wait for the switch conditions: run `bash docs/notes/20261103-scripts_cleanup/wait_switch_conditions.sh 230` as a background command with `ap snooze <job> 240 "waiting for the lab"`, rerun it when it gives up; when it exits 0, read the lab's STATE.md and wait again if a run is going or queued. If the lab is still not idle at 09:00 on 11-04, `ap step wait` with what it is doing.
- [ ] 2. Switch per RUNBOOK "The switch": stop the crosstalk job and its watcher, `git pull --no-rebase`, run move_scripts.py, check_scripts.py (the 15 known problems only), the tests against a baseline of the same commit, one commit, push.
- [ ] 3. After the push, per RUNBOOK: the bullet in the lab's STATE.md (commit, push), the crosstalk job's rule step and `ap arm docs/handoff/20261027-crosstalk_feed_poller.md --at now`, the memory file and its MEMORY.md line.
- [x] 4. src/morphohand/geometry/rotations.py with tests/test_geometry_rotations.py, per RUNBOOK "Shared math modules"… Done 11-03 11:55: Done by the arming window 11-03 12:1x: 529a91a48 src/morphohand/geometry/rotations.py (51 functions over scipy) and tests/test_geometry_rotations.py (15 passing with scipy 1.18 and with scipy 1.12/numpy 1.26), CLAUDE.md and AGENTS.md rule
- [x] 5. The other modules (geometry vectors, distances, sdf, mesh, sampling; numerics stats, signal, angles), each wi… Done 11-03 23:52: 7467a2e99 on main: geometry vectors, distances, sdf, mesh, sampling and numerics stats, signal, angles under src/morphohand/ (renamed by the switch), plus quat_twist, rot_slerp, adjoint in rotations; each checked once in logs/20261103-scripts_cleanup/equiv/check_*.py. No script calls them until the landing.
- [ ] 6. Migrate the copies folder by folder per RUNBOOK "Migration", with the 1e-12 checks, one commit per folder; not src/morphohand/driver/ and not the UR5 programs that move the robot.
- [ ] 7. Fix the bugs in RUNBOOK's table, one commit each, naming the pages and runs whose numbers change; rebuild those pages locally and list them under "Publish pending".
- [ ] 8. tests/test_no_private_math.py with its allow-list, and the section in CLAUDE.md and AGENTS.md.
- [ ] 9. Report in State: the commits, copies removed per folder, bugs fixed; regenerate scripts/README.md (scripts/docs/scripts_map.py); disarm.
Standing directions (for every step):
- The lab's STATE.md (docs/experiments/20261022-ur5_tip_calibration/lab/STATE.md), step 3: the switch's change there is ONE bullet appended at the end of its section 'Where things are' (the commit, the new paths of the scripts the lab runs), and nothing elsewhere in the file. The lab writes STATE.md during runs, and a one-bullet change merges cleanly (lab, 12:04 on 11-03, relayed by the coordinator window).
- Owner 11-03 12:4x: no new test files. The modules of step 5 are checked once against scipy, trimesh or MuJoCo in scratch scripts under logs/20261103-scripts_cleanup/equiv/, not in tests/; step 8 writes only the section in CLAUDE.md and AGENTS.md, and no tests/test_no_private_math.py.
- Owner 11-03 12:4x: the switch (step 2) also renames the package morphohand to sr2hand; move_scripts.py does it after the scripts move (RUNBOOK, 'The package rename'): imports, dotted names, paths, MorphoHandEnvCfg to HandEnvCfg, pyproject.toml and uv.lock, the venv's editable install. Check '.venv/bin/python -c "import sr2hand"' after it. External names stay (Lightning Grasp's and GraspGenX's morphohand assets, the W&B project morphohand-rl), and the CB1 keeps src/morphohand/.
- Window 220623e7 (HOM score levels, 11-03 18:27): the queue logs/20261103-hom_score_refine/queue.sh runs on the GPU until about 04:30 on 11-04; its batch children are spawned processes that import scripts/hom_gpu.py, hom_tasks.py, hom_ea.py, hom_landscape.py, fp_sr2_toy.py and src/morphohand by path at every batch. Do not switch while it runs: pgrep -f '[h]om_score_refine/queue.sh' must print nothing (its python steps show as scripts/hom_grasp_refine.py, which wait_switch_conditions.sh sees, but the queue pauses between steps). Put scripts/hom_grasp_refine.py and its page builder scripts/hom_score_refine_page.py (+ .template.html) under hom in plan.json. The autopilot jobs hand-20261104-hom_* that follow it edit scripts/hom_*: migrate scripts/hom/ when none of them is running (ap status).
- Owner 2026-11-03 19:3x (relayed by window 92ca1bc5): this cleanup is lower priority than the fingertip-shape rerun, hand-20261103-fingertip_shape_rerun, which is armed after the HOM chain. Do not start step 2 (the move of scripts/, src/ and tests/) until ap status shows that job done or stopped: it runs scripts/tip_codesign_*.py, curved_tips_*.py, tpu_tip_magnet_sensing.py and the HOM scripts by path.
- New HOM script of 11-03 20:5x (window 220623e7): scripts/hom_task_conflict.py, and scripts/hom_task_conflict_page.py when it appears, go under scripts/hom/ with the other hom_* scripts (add them to plan.json); the job hand-20261104-hom_task_conflict claims them and starts at 09:15 on 11-04 or later, after the page job.
- Autopilot 00:1x on 11-04, from direction 14: step 1's deadline of 09:00 on 11-04 no longer applies, because the switch waits for hand-20261103-fingertip_shape_rerun. This job is armed --after that job. Once a session has started after it, if the lab is still not idle 24 hours later, ap step wait with what the lab is doing.
<!-- /ap:steps -->
Claimed: during step 2 every file under scripts/, src/ and tests/, and pyproject.toml and uv.lock; after it, the
folder being migrated; src/sr2hand/geometry/, src/sr2hand/numerics/, docs/notes/20261103-scripts_cleanup/,
scripts/README.md, and the scripts and math lines of README.md, CLAUDE.md, AGENTS.md.
Waiting on owner: migrating src/sr2hand/driver/ and the UR5 programs that move the robot (ur5_inverted_rig.py,
the plan generators, ur5_tip_calibration_holder_paths.py) needs the lab's simulator and the owner.
Do not: switch while the lab runs or another window has uncommitted script changes; push anything other than this
task's commits; edit the crosstalk job's files beyond the switch's renames; republish artifacts from this login.
