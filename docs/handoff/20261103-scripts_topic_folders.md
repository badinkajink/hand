Status: active
Task: (owner, 2026-11-03 11:0x) "repo clean up + refresh on the README", "the absolutely abysmal scripts folder", and
"rewriting rotations, geometry, math helpers everywhere with custom math instead of using proven library calls / a
single help file(s)". Owner's choices at 11:5x: ten topic folders, switched by this autopilot after the day's lab work;
shared math modules with the live copies migrated; the 48 unreferenced scripts archived.
Read first: docs/notes/20261103-scripts_cleanup/RUNBOOK.md (conditions, commands, expected checks, migration
procedure, bugs), audit_rotations.md and audit_geometry.md beside it, the lab's
docs/experiments/20261022-ur5_tip_calibration/lab/STATE.md, `~/.claude/autopilot/ap status`, ~/.claude/writing_reference.md.
State: 11-03 11:5x. README and .gitmodules committed (6eabaf56b). Dry run of the switch at 6eabaf56b in two
worktrees: 754 files moved, 2,072 script imports resolve, 15 dangling references that predate the move (RUNBOOK
lists them), pytest identical in both layouts (302 passed, 43 failed, 43 skipped, 3 errors with LFS pointers), and no
script that imported in the flat layout fails in the moved one. The tools and plan.json are in
docs/notes/20261103-scripts_cleanup/. Nothing is switched yet. Step 4 is done: the rotations module and its tests are
ab59888dd (the ledger's 529a91a48 is the same commit before its message was corrected), with the rule in CLAUDE.md
and AGENTS.md; the worktrees of the dry run are removed.
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
<!-- /ap:steps -->
Claimed: during step 2 every file under scripts/; after it, the folder being migrated; src/morphohand/geometry/,
src/morphohand/numerics/, tests/test_geometry_*.py, tests/test_numerics_*.py, tests/test_no_private_math.py,
docs/notes/20261103-scripts_cleanup/, scripts/README.md, and the scripts and math lines of README.md, CLAUDE.md, AGENTS.md.
Waiting on owner: migrating src/morphohand/driver/ and the UR5 programs that move the robot (ur5_inverted_rig.py,
the plan generators, ur5_tip_calibration_holder_paths.py) needs the lab's simulator and the owner.
Do not: switch while the lab runs or another window has uncommitted script changes; push anything other than this
task's commits; edit the crosstalk job's files beyond the switch's renames; republish artifacts from this login.
