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
<!-- ap:steps: rendered by ~/.claude/autopilot/ap from its ledger; change it with `ap step`, not by hand -->
Steps, in order (the ledger; `ap step done <job> <id> "result"` marks one, `ap step add <job> "text"` adds one):
- [ ] 1. Wait for the switch conditions: run `bash docs/notes/20261103-scripts_cleanup/wait_switch_conditions.sh 230` as a background command with `ap snooze <job> 240 "waiting for the lab"`, rerun it when it gives up; when it exits 0, read the lab's STATE.md and wait again if a run is going or queued. If the lab is still not idle at 09:00 on 11-04, `ap step wait` with what it is doing.
- [ ] 2. Switch per RUNBOOK "The switch": stop the crosstalk job and its watcher, `git pull --no-rebase`, run move_scripts.py, check_scripts.py (the 15 known problems only), the tests against a baseline of the same commit, one commit, push.
- [ ] 3. After the push, per RUNBOOK: the bullet in the lab's STATE.md (commit, push), the crosstalk job's rule step and `ap arm docs/handoff/20261027-crosstalk_feed_poller.md --at now`, the memory file and its MEMORY.md line.
- [x] 4. src/morphohand/geometry/rotations.py with tests/test_geometry_rotations.py, per RUNBOOK "Shared math modules"… Done 11-03 11:55: Done by the arming window 11-03 12:1x: 529a91a48 src/morphohand/geometry/rotations.py (51 functions over scipy) and tests/test_geometry_rotations.py (15 passing with scipy 1.18 and with scipy 1.12/numpy 1.26), CLAUDE.md and AGENTS.md rule
- [ ] 5. The other modules (geometry vectors, distances, sdf, mesh, sampling; numerics stats, signal, angles), each with its test against scipy or trimesh.
- [ ] 6. Migrate the copies folder by folder per RUNBOOK "Migration", with the 1e-12 checks, one commit per folder; not src/morphohand/driver/ and not the UR5 programs that move the robot.
- [ ] 7. Fix the bugs in RUNBOOK's table, one commit each, naming the pages and runs whose numbers change; rebuild those pages locally and list them under "Publish pending".
- [ ] 8. tests/test_no_private_math.py with its allow-list, and the section in CLAUDE.md and AGENTS.md.
- [ ] 9. Report in State: the commits, copies removed per folder, bugs fixed; regenerate scripts/README.md (scripts/docs/scripts_map.py); disarm.
Standing directions (for every step):
- The lab's STATE.md (docs/experiments/20261022-ur5_tip_calibration/lab/STATE.md), step 3: the switch's change there is ONE bullet appended at the end of its section 'Where things are' (the commit, the new paths of the scripts the lab runs), and nothing elsewhere in the file. The lab writes STATE.md during runs, and a one-bullet change merges cleanly (lab, 12:04 on 11-03, relayed by the coordinator window).
- Owner 11-03 12:4x: no new test files. The modules of step 5 are checked once against scipy, trimesh or MuJoCo in scratch scripts under logs/20261103-scripts_cleanup/equiv/, not in tests/; step 8 writes only the section in CLAUDE.md and AGENTS.md, and no tests/test_no_private_math.py.
- Owner 11-03 12:4x: the switch (step 2) also renames the package morphohand to sr2hand; move_scripts.py does it after the scripts move (RUNBOOK, 'The package rename'): imports, dotted names, paths, MorphoHandEnvCfg to HandEnvCfg, pyproject.toml and uv.lock, the venv's editable install. Check '.venv/bin/python -c "import sr2hand"' after it. External names stay (Lightning Grasp's and GraspGenX's morphohand assets, the W&B project morphohand-rl), and the CB1 keeps src/morphohand/.
<!-- /ap:steps -->
Claimed: during step 2 every file under scripts/, src/ and tests/, and pyproject.toml and uv.lock; after it, the
folder being migrated; src/sr2hand/geometry/, src/sr2hand/numerics/, docs/notes/20261103-scripts_cleanup/,
scripts/README.md, and the scripts and math lines of README.md, CLAUDE.md, AGENTS.md.
Waiting on owner: migrating src/sr2hand/driver/ and the UR5 programs that move the robot (ur5_inverted_rig.py,
the plan generators, ur5_tip_calibration_holder_paths.py) needs the lab's simulator and the owner.
Do not: switch while the lab runs or another window has uncommitted script changes; push anything other than this
task's commits; edit the crosstalk job's files beyond the switch's renames; republish artifacts from this login.
