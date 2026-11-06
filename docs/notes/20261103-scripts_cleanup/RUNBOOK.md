# Scripts in topic folders, the package renamed sr2hand, and shared math modules: runbook (2026-11-03)

The owner asked at 11:0x on 2026-11-03 for "a repo clean up + refresh on the README", naming "the absolutely abysmal
scripts folder", and added: "also very concerned about us rewriting rotations, geometry, math helpers everywhere with
custom math instead of using proven library calls / a single help file(s) for these broad categorical functions".
At 11:5x the owner chose ten topic folders, an autopilot for the switch after the day's lab work, a shared math
module with the live copies migrated to it, and the 48 unreferenced scripts archived. At 12:4x the owner asked to
"change any mention of morphohand to sr2 hand, or even just hand where appropriate" and chose `sr2hand` for the
package, renamed in the same switch; and, "very hesitant now to run endless unit tests that exist only so that you can
pass them", cut the tests to the ten files that exercise the hardware path (`fd2100da0`): no new test files. The hand-off
note is `docs/handoff/20261103-scripts_topic_folders.md`.

## Outcome (2026-11-05)

The switch landed at 00:50 on 11-05 as `a41ffa27b` and the migration at 00:52 (`e7040b5b4` to `c86d3fe66`). The sections below describe the
procedure as planned. Two things differed from the dry runs. A script that the coordinator committed 45 minutes earlier was not in the plan
and was moved by hand, so a dry run belongs immediately before the switch. `move_scripts.py` left out the code under `docs/experiments/` and
`paper/`, `.gitignore` and the VS Code workspace, and `move_extra_live_code.py` did those. The Spearman fix changed the plant-transfer rows by up to
0.062, on a friction grid with many tied scores, where the table below says 0.003.

## Files in this folder

| file | what it is |
|---|---|
| `plan.json` | each script's folder; 652 scripts in ten folders and 48 to `archive`; a value `folder/new_name.py` also renames (`build_morphohand_urdf.py`, `graspgenx_make_morphohand.py`) |
| `move_scripts.py` | the switch: `git mv`, the root depth and `import _paths` in each moved script, every `scripts/<name>` in the live text, `_paths.py`, `MOVED.tsv`, then the package rename, the archive README, the map, the checker, and the special edits |
| `verify_scripts.py` | static checks, installed as `scripts/docs/check_scripts.py` |
| `scripts_map.py` | writes `scripts/README.md`, installed as `scripts/docs/scripts_map.py` |
| `smoke_imports.py` | imports every script with a main guard, each in its own process |
| `wait_switch_conditions.sh` | waits until the mechanical conditions of the switch hold |
| `land_migration.sh` | cherry-picks the 23 migration commits of the local branch `scripts-migrate`; the branch was removed on 11-05, and its commits are on main |
| `move_extra_live_code.py` | the second pass of the switch, for the code that `move_scripts.py` leaves out |
| `audit_rotations.md` | 309 hand-written rotation and transform copies in 128 files, the library call for each kind, four bugs |
| `audit_geometry.md` | 165 geometry and numerics copies (143 live), the library call for each kind, the bugs |

The dry runs' logs are in `logs/20261103-scripts_cleanup/` on the workstation.

## The package rename

`move_scripts.py` renames the package after moving the scripts: `git mv src/morphohand src/sr2hand`; in the live code
and text, the imports, the dotted names (`morphohand.rl` and the other subpackages), the paths `src/morphohand/`,
`MorphoHandEnvCfg` to `HandEnvCfg`, the private attributes `env._morphohand_*` to `env._sr2hand_*`, the variable
`MORPHOHAND_NCONMAX` to `SR2HAND_NCONMAX`, the display name to "SR2 Hand" (code) or "SR² Hand" (markdown), and "the
morphohand" in prose to "the hand"; the project name in `pyproject.toml` and `uv.lock`; and the venv's editable install
(`uv pip uninstall morphohand`, `uv pip install --no-deps -e .`; it puts `src/` on `sys.path`, so `import sr2hand`
works at once and a leftover `import morphohand` fails).

Names that stay: Lightning Grasp's and GraspGenX's assets (`morphohand.urdf`, `assets/hand/morphohand`,
`x_grippers/morphohand`, `morphohand_grasp.json`, the gripper name), the W&B project `morphohand-rl`, an SVG hash salt,
`scripts/archive/` (some of its scripts run on the CB1), `docs/RESEARCH_STATE.md` and the result pages (history). The
CB1's checkout is never pulled and keeps `src/morphohand/`, so the runbook `docs/hardware_control_station.md` keeps its
CB1 commands and gains a paragraph that says so; its two workstation commands change. The driver code there imports no
`morphohand`, so a file shipped to the CB1 still runs. Run configs name the package only as the W&B project; no pickled
class path was found in 719 config files. A checkpoint that fails with "No module named 'morphohand'" names a class by
its old module: load it with `sys.modules["morphohand"] = sr2hand` (and its submodules) in that one loader.

## Dry run of 2026-11-03, in a worktree at 4f54fc805

A worktree has LFS pointers in place of data, empty submodules and no `.dotenv`, so its test numbers differ from the
main checkout's; the two layouts were compared in two worktrees at the same commit.

- 754 files moved; script references rewritten in 716 files; the package rename rewrote 252 files. The tool lists 36
  lines that use `__file__` for something other than the root; each was read (FreeCAD fallbacks, scripts that relaunch
  themselves, templates beside their builder). It lists the lines that still say "morphohand": the names that stay.
- `check_scripts.py`: 748 files, 2,033 imports of scripts, all resolving to the moved file, and no live import of
  `morphohand`; 15 problems, each a reference to a script that did not exist before the move either: `d1_ilqr.py`,
  `hand_object_scale_export.py`, `perp_phalanx_variant.py`, `pinch_brake_page.py` and its template,
  `probe_contact_stiffness.py`, `real_v1_hand_plan.py`, `rl_record_reorient_fingertip_traj.py`, `servo_id_fit.py`,
  `servo_id_log.py`, the example `scripts/X.py` in `prose_lint.py`, and `lightning_grasp_runner.py`,
  `rl_demo_handoff.py`, `run6_combined_multitask.py` (in `archive/`).
- pytest: 108 passed and 7 failed in both layouts, the same 7 (LFS plan files, empty submodules, no `.dotenv`, and the
  floor datum below). In the main checkout the suite gives 114 passed and 1 failed: the floor datum of
  `tests/test_bench_tags.py`, which waits on the owner (`src/sr2hand/bench/tags.py` has 85.0 mm beside the comment
  "measured 65.0").
- Imports: see the note's State for the smoke test of the moved and renamed layout.

## The switch

Before it, `python3 move_scripts.py PLAN.json --dry-run` prints "not in the plan" for scripts added since the last dry run: add
them to `plan.json`. Conditions, all at once (check each; if one fails, `ap snooze <job> 60 "<which>"` and check again):

0. `ap status` shows `hand-20261103-fingertip_shape_rerun` done or stopped, and no autopilot job runs but the crosstalk
   poller. The owner ranked this cleanup below that rerun at 19:3x on 11-03, and the rerun runs after the HOM chain.
   `wait_switch_conditions.sh` checks this, the conditions below and every queue under `logs/`.
1. The lab is idle: `docs/experiments/20261022-ur5_tip_calibration/lab/STATE.md` (after `git pull`) says no session
   runs and none is queued, the newest "UR5 lab" commit is over 45 minutes old, and `ssh` is not needed: the
   workstation never writes to corvus. If the lab runs overnight, wait for its morning commit.
2. No process on the workstation runs a file under `scripts/`: `pgrep -af "scripts/[A-Za-z0-9_]+\.(py|sh)"` lists
   none other than the crosstalk job's watcher, `crosstalk_feed_watch.py`, which is stopped with its job below. The
   fingertip viewer under `src/` may keep running: it imported its modules at start. Restart it after the switch only
   if the owner asks; its path is then `src/sr2hand/driver/reskin5x/host/wrench_viewer.py`.
3. No uncommitted change under `scripts/`, `src/`, `tests/`, `CLAUDE.md`, `AGENTS.md`, `README.md` and the lab's
   `STATE.md` (`git status --porcelain -- scripts src tests CLAUDE.md AGENTS.md README.md
   docs/experiments/20261022-ur5_tip_calibration/lab/STATE.md`; untracked files may stay).
4. No commit touching `scripts/` from another window in the last 45 minutes.

Then:

```bash
~/.claude/autopilot/ap status                     # the crosstalk job's session: wait while it is busy
~/.claude/autopilot/ap disarm hand-20261027-crosstalk_feed_poller --stop
pgrep -af "[c]rosstalk_feed_watch.py"             # its detached watcher: kill <pid> (resguard logs the signal)
git pull --no-rebase                              # this repository merges origin/main
python3 docs/notes/20261103-scripts_cleanup/move_scripts.py docs/notes/20261103-scripts_cleanup/plan.json
.venv/bin/python scripts/docs/check_scripts.py    # expect exactly the 15 problems above (system python3 3.10 adds 3 parse errors: f-strings of 3.12)
.venv/bin/python -c "import sr2hand, sr2hand.geometry.rotations"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/ -q   # 114 passed, the floor datum failing
git status --short | grep -v '^R ' | head -50     # read what changed besides the renames
```

Then commit the switch as one commit (`git add -A scripts src tests configs CLAUDE.md AGENTS.md README.md
pyproject.toml uv.lock`, the rewritten READMEs under `docs/experiments/`,
`docs/hardware_control_station.md`, `docs/rl/morph_sweep_STATUS.md`) and push. If the push is refused,
`git pull --no-rebase`, rerun the checker, push.

After the push:

- Append ONE bullet at the end of the section "Where things are" of the lab's `STATE.md`, and change nothing else in
  that file: the lab writes it during runs, and one bullet merges cleanly (lab, 12:04 on 11-03). The bullet: the
  scripts moved into topic folders and the package `src/morphohand/` became `src/sr2hand/` at <time> (<commit>); corvus
  pulls before its next run; the session program is `src/sr2hand/driver/reskin5x/host/ur5_tip_calibration.py`, and
  `scripts/MOVED.tsv` maps the old paths of the scripts the lab runs. `move_scripts.py` does not rewrite `STATE.md`.
  Commit and push it.
- `ap step add hand-20261027-crosstalk_feed_poller --rule "Scripts moved into topic folders and the package renamed
  sr2hand at <time> (<commit>): the claimed scripts are now under scripts/fingertip/ and scripts/docs/
  (scripts/MOVED.tsv). Run them there, restart the watcher from scripts/fingertip/crosstalk_feed_watch.py, and correct
  the note's Claimed line."`, then `ap arm docs/handoff/20261027-crosstalk_feed_poller.md --at now` (it keeps its
  account and parallel flag).
- A memory file `project_scripts_topic_folders_2026_11_03.md` in the project's memory directory with its line in
  `MEMORY.md`: the folders, `_paths`, `MOVED.tsv`, the checker, the map, and the package `sr2hand`.

## Landing the migration after the switch

The migration was prepared before the switch, on the local branch `scripts-migrate` in the worktree
`/home/humanoid/Programs/hand_wt_migrate`, above a rehearsal of the switch (`91ab66052`, never picked). The geometry and
numerics modules and the shared-code section of `CLAUDE.md` and `AGENTS.md` went to main before the switch, so the switch
renames them with everything else. After the switch commit and its follow-ups:

```bash
~/.claude/autopilot/ap status                          # no hand-*-hom_* job runs; no window has uncommitted hom scripts
bash docs/notes/20261103-scripts_cleanup/land_migration.sh     # 23 commits in order, HEAD's module files kept
```

The rehearsal of 11-04 on `91ab66052` ended with an empty diff against the branch. A conflict means that someone edited the
same lines after `dc1830409` (the HOM chain, the lab's rod scripts). Take their change, apply the same replacement to it,
`git add` and `git commit -C <commit>`, then rerun the script from the next commit. Before `72c82f2e3`, record one CPU HOM
rollout on the moved tree, and compare it after that commit. If a HOM job runs, stop before `72c82f2e3` with
`land_migration.sh 83dd092a1 71be5fc56`, land `0bb15c5d2` to `d5198b948` and `92d136d5e` the same way, and leave
`72c82f2e3`, `f719012cd` and `517cd84f5` for when it has ended.

Checks after the picks, from the repository root:

```bash
.venv/bin/python scripts/docs/check_scripts.py         # 11 known problems since 11-05 (scripts still to be written, scripts removed before 10-09, one driver comment)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/ -q     # 114 passed, the floor datum failing
E=logs/20261103-scripts_cleanup/equiv
WT=$PWD bash $E/smoke.sh <switch commit>               # imports every touched script, GPU hidden
.venv/bin/python -W ignore $E/undefined.py <switch commit>      # bare names without a binding
BASE=logs/20261103-scripts_cleanup/base_tree .venv/bin/python -W ignore $E/attrs.py $PWD
    # `M.name` and `from M import name` whose target is gone; it found six on 11-03 that smoke.sh and undefined.py cannot see
git diff scripts-migrate -- src/sr2hand/geometry src/sr2hand/numerics    # only functions that chain sessions added
.venv/bin/python -W ignore $E/survivors.py             # names of the audits still defined; check scripts added since 11-03
```

`smoke.sh` reports `geometry` (it is `contact_surface.geometry`). In a worktree it also reports JSON errors from LFS pointers,
which main does not have. Then replace "migrates the live copies once the scripts have moved into topic folders, after the
fingertip-shape rerun." in `CLAUDE.md` with "migrated the live copies.", and search scripts added since 11-03 for new private
copies. The worktree and the branch are removed at the very end: `oldcode.py` reads `91ab66052`.

## Shared math modules

Conventions, from `audit_rotations.md`: quaternions are wxyz as in MuJoCo, products are Hamilton, rotations are
active, angles are radians, and every function takes batches. The modules import numpy and scipy only: the `femtip`
environment has no MuJoCo, and corvus has scipy 1.12, so reorder quaternions by indexing (`q[..., [1, 2, 3, 0]]`)
and never pass `scalar_first`. `rot_between` keeps the explicit form of `ur5_tip_calibration.rot_from_to`, with the
antiparallel case turned about a perpendicular axis (a rotation, never −I).

- `src/sr2hand/geometry/rotations.py` exists (`ab59888dd`): quaternions, rotation matrices, Euler angles and rotation
  vectors, 4 × 4 transforms and UR poses, Kabsch and Umeyama.
- `src/sr2hand/geometry/vectors.py`, `distances.py`, `sdf.py`, `mesh.py`, `sampling.py` and
  `src/sr2hand/numerics/stats.py`, `signal.py`, `angles.py` (`audit_geometry.md`, "Recommended layout"). Prefer
  the library call itself where one exists (scipy.stats.spearmanr, scipy.stats.bootstrap, trimesh); a module function
  is written only where no library has it.
- No new test files (owner, 2026-11-03). Check each new function once against scipy, trimesh or MuJoCo in a scratch
  script under `logs/20261103-scripts_cleanup/equiv/`.

## Migration, folder by folder

For each folder of `scripts/` (and `src/sr2hand/` outside `driver/`):

1. Replace each copy listed in the audits with a call to the module or the library. Keep a script's own function
   name as a one-line wrapper when other scripts import it.
2. Before deleting a copy, check it against its replacement on 10,000 random inputs, including the edge cases the
   audit names (antiparallel vectors, half turns, zero-length segments, ties), to 1e-12. Keep the check scripts in
   `logs/20261103-scripts_cleanup/equiv/`.
3. A copy that disagrees with its replacement is a bug or a different convention: leave it in place, and fix it in
   its own commit that names the pages and runs whose numbers it changes.
4. Run `scripts/docs/check_scripts.py`, `smoke_imports.py --names` on the touched scripts, and the tests; commit the
   folder.

Not migrated by the autopilot: `src/sr2hand/driver/` and the UR5 programs that move the robot
(`scripts/ur5/ur5_inverted_rig.py`, the plan generators and `ur5_tip_calibration_holder_paths.py`). They waited for a
window with the owner and the lab, whose simulator checks every plan; the branch `ur5-shared-math` of 2026-11-05 moved
them (next section). `scripts/archive/`, the CB1's `src/sr2hand/driver/manta/` and `src/sr2hand/bench/tags.py` are never
migrated.

## The UR5 programs on the shared math (branch `ur5-shared-math`, 2026-11-05)

The owner's go came at 12:0x on 11-05 (QUEUE R35, part B). The lab program
`src/sr2hand/driver/reskin5x/host/ur5_tip_calibration.py` puts `src/` on `sys.path` through `scripts/_paths.py`, as its
simulator already did. It still runs as `/usr/bin/python3 <path>` from corvus's clone; this was checked under Python
3.10, numpy 1.26.4 and scipy 1.12.0. Its `C.*` names keep their signatures and call the modules.

Replaced:
- lab program: `skew`, `exp_so3`, `Tm`, `inv`, `apply`, `ur_to_T`, `rot` (degrees), `rot_from_to`, `perturb`,
  `turn_about`, `closest_on_triangles` (moved to `geometry.distances`), the facet normals, `line_distance`'s projection,
  `fit_gravity`'s centre of mass and turn, and `check_placement`'s yaw, tilt and wrap. `rot_from_to` keeps the identity
  and the exact half turn within 1e-10 of parallel by an explicit branch.
- simulator: the wall contact's inside test (`distances.barycentric_inside`).
- inverted rig: its frames (`fingertip_T_MF`, `T_AF`, `table_T_BA`, `search_rod`, `frames_for` with its old fallback),
  the nearest rotations, `pose_B`, `_inv`, `_flat_R`, `fib_dirs`, `_sphere_pts`, `_tree_sdf`, `hull_sdf`, `_slerp`,
  `_on_face`, the joint wraps and the 4 x 4 builds; `indenter_array._Rz` and its 4 x 4 builds.
- holder paths: `rot_z`, `rot_from_to` (the lab program's) and `slerp_T`.
- plan generators: the ATI plan's `fib_sphere` and `rot_to` (its half turn under a . b = -0.999999 kept), the joint 6
  wraps of the tip plan, R24 pass 3, the tall pass, the trimmed cap and the tall indenter, and the headings of the probe
  plan and the session checker.
- new in the modules: `distances.closest_on_triangles` and `barycentric_inside`, `rotations.tf_interp`.

Kept, with the reason:
- the lab program's `log_so3` trace formula, and `T_to_ur`, `pose_error` and `mean_pose`, which call it (its SVD
  projection and vee call the module). Every target sent to the robot passes through `T_to_ur`. Near a half turn the
  formula stands off scipy's by about 1e-15 / (pi - angle)^2 rad: 3e-4 rad at 2e-6 rad from a half turn, 1.4e-9 rad at
  1e-3. Within 1e-6 rad of a half turn it returns the axis with its largest component positive. The tool-down targets of
  the probe plans lie near a half turn. A switch to `rotations.log_so3` waits for the owner and a run on the lab's
  simulator.
- `fit_onset`'s `np.convolve(mode="same")`: `numerics.signal` has no zero-padded edge, and `fit_onset` leaves out the
  last `win // 2` samples for it. `fit_gravity`'s signed polar factor fits the HEX-E's sign. `fit_rod_registration`'s
  basis defines its parameters and their priors. `lm`: the fits' results depend on it.
- the simulator's DH chain, its tangent bases (cross products with a reference axis; `rotations.tangent_basis` is
  another basis, and the friction solve runs in it) and `seg_pose`, which calls the lab program's `log_so3`.
- the rig's `tangent_basis`: it takes the seat plans' stored normals, rounded to 5 decimals and up to 1e-5 off unit
  length, as they are, and `frame_from_axis` would normalize them and move a sweep's `t1` by up to 1e-5. Its
  `cmd_seat_beam.props` takes the moments about the centroid, where `mesh.polygon_section` takes them about the origin
  and loses up to 8e-9 relative for small sections 100 to 200 mm from it. `_winding` has no library equivalent; the DH
  kinematics (`_dh`, `fk_many`, `ik_all`) are the robot's model.
- the ATI plan's `hanging_points` perpendiculars, which set the phase of the 16 rim samples about g, and
  `indenter_array.T_AF_press`'s columns (n, z x n, z), whose hint sits in the column before the axis.

Checks, in `logs/20261103-scripts_cleanup/equiv/ur5/` on the workstation:
- `check_driver.py` (95 of 95) and `check_scripts.py` (54 of 54), old against new on 10,000 inputs each and the edge
  cases, in the venv (numpy 2.4.4, scipy 1.17.1) and in a Python 3.10 venv with numpy 1.26.4 and scipy 1.12.0. Most
  replaced functions are bitwise identical; the rest agree to 8e-16, and `apply` and `turn_about` to 1.1e-13 and 4.3e-13
  relative on millimetre values up to 3,500 mm. The registration fits move by 2e-8 to 4.6e-7 mm and rad. The old code
  moves as far when its touches move by 1e-15 relative, the resolution of its hand-written Levenberg-Marquardt.
- plans made again with the old and the new code: the 24 probe plans converted from the 11-03 and 11-04 seat plans are
  byte-identical. The rig's replays of the 2 mm sphere and the rod on seat 1 are identical in every number, and so is
  `dry-check --real --chain` on the lab's two seat 1 program plans. The holder path check of the 10-29 side map is
  byte-identical.
- the tip plan builder and the ATI plan builder call the simulator's damped least-squares IK (`SIM.ik`). On some poses
  it returns another joint solution when the pose changes in its last bits; one pose gave 10 different joint 5 values
  over 20 changes of 1e-15 relative. Their plans differ between the old and the new code (two tip end actions kept, one
  twist of -60 deg in place of -90, 151 ATI directions against 149). The old builders differ as much with their
  placement moved by 1e-12 mm: the two actions kept in 4 of 6 runs, 149 to 151 ATI directions.
- the simulator with its clock fixed (`simrun.py`, the clock's start in `World` set to 1762300000 s) repeats itself
  bitwise. On a probe plan (50 actions) and a tip plan (38 actions) old and new give the same samples and the same
  events, every action ok. Forces agree within 2.1e-6 N, joints within 7e-9 rad and registrations within 8.7e-7 mm.
- pytest in the two worktrees: 113 passed and the same 2 failed (the floor datum, and the uv lock test, which needs the
  submodules a worktree lacks).
- cost: the replays and the run check took 6 to 9 % longer, the simulator sessions 13 to 16 %. `apply` on 10,000
  points takes 125 us against 44, through `einsum`.

## Closed-form IK for the plan builders (branch `ur5-closed-form-ik`, 2026-11-05)

The plan builders now take their joints from `sr2hand.robot.ur_kinematics.ik_nearest`, the UR5's IK in closed form with
one fixed choice, in place of the simulator's damped least squares (`SIM.ik`). Two builds of one plan now agree in every
field but the build's wall time (`checks.seconds`). Rebuilt with `--q-seed`, as the lab builds them, the tip plans keep
the old code's actions, twists and joint targets. The owner asked for it at 13:5x on 11-05 (QUEUE R35, part C): "closed
form IK sounds good. make sure that it can handle parameterization, in case dimensions change down the line, or if we
ever add the proximal phalange extender".

`SIM.ik` runs damped least squares from a seed. On the old tip build of the lab's measured-rod cell it returned another
solution than the nearest one on 150 of 5,330 pose checks, and the same joints a whole turn apart on 18. On 118 of those
168 its iterations had wandered by whole turns, up to tens of thousands of degrees, and the builder read the joints past
+-357 deg as no IK. One tip pose gave 10 different joint 5 values over 20 changes of 1e-15 relative (section above).

The module, `src/sr2hand/robot/`:
- `ur_kinematics.py`: `URParams`, the DH table as data. `UR5_CB3` is the simulator's `DH_CB3`, float for float, with the
  joint limits of the cell's `ur5.urdf` (+-360 deg, the elbow +-180 deg). Another UR arm loads with `from_dict` or
  `from_json`; a table without the UR geometry (a calibrated one) is refused. `fk`; `ik`, every closed-form solution in
  a fixed branch order (shoulder, wrist, elbow) with flags for unreachable and singular branches; `ik_near` and
  `ik_nearest`, the solution nearest a reference in Euclidean joint distance, joints within half a turn of the reference
  and inside the limits, ties within 1e-9 rad to the lower branch. `ik_nearest` first takes the target's rotation to the
  nearest rotation, as the least squares measured its error, and keeps its acceptance of 1e-4 mm and 1e-6 rad. The rig's
  `ik_all` moved here with its arithmetic unchanged.
- `tool_chain.py`: `Link` and `ToolChain`, named rigid transforms from the flange, with JSON in and out.
  `calibration_tool` and `ati_tool` are the builders' stacks (the HEX-E 35.0 mm out with the mount turned -92.15 deg,
  then the pad frame or the ATI frame), equal to the builders' `T_EM`, `T_EP` and `T_EA` bit for bit. `SR2Hand` composes
  the flange, the HEX-E, the adaptor at clocking k and the palm P from `assets/cad_v3/frames.json`, then one finger's
  gantry, yaw, MCP, PIP, tip body and contact point. `SR2Hand.ik` solves a contact-point target for given gantry
  positions and finger joints.
- `sr2_hand.json`: the finger's link lengths with their sources: `yaw_to_mcp_mm` 20.75 and `mcp_to_pip_mm` 30.8
  (cad_v1's `assembly.json`, `kinematics_mm`), `tip_body_mm` 36.36 (cad_v2's `assembly.json`), `reach_mm` 10.55 (the v4b
  pad's apex, `tip_codesign_geometry.REACH`), `proximal_extension_mm` 0; per finger `flex` (the MCP and PIP axes' y
  sign) and `face_sign` (the palmar side). `SR2Hand.load(proximal_extension_mm=e)` or `with_geometry(finger, ...)`
  changes them.

Switched from `SIM.ik` to `ik_nearest`: the tip plan builder (`Checker.ik`; its `seed` now lists the closed form's eight
solutions), the ATI plan builder (on the homing pose's branch, as its docstring always said), the holder path checker,
the holder plan, R24 pass 3, the tall pass, the top-edge pass, the trimmed-cap reach, the v3 suite's replay, the wrist
order, the indenter plate layout, the indenter array, the tall indenter and the fingertip holder; and the rig's homing
in its plate cell. Left on `SIM.ik`: the simulator itself, the lab's tools that move the robot (`lab/move_guarded.py`,
`lab/leave_tall_key.py`), the films (`ur5_cell_render.py`, `fingertip_calibration_cell_render.py`), the results page
(`ur5_tip_calibration_page.py`) and the cad_v3 analysis's `build_ur5_mjcf.py`.

How the builders' joints reach the robot:
- tip plans: the builder uses the joints in `check` and `joint6_deg` for its own reach, window and clearance checks and
  its order by joint 6; the program solves each move with the controller's IK nearest the homing's joints. Only
  `sim.q_seed` enters the program's simulator, as the seed of its start pose.
- ATI plans: each station's `q_deg` is the moveJ target of `ati_calibration_session.py`.
- the rig's probe plans: `home_path_q` are moveJ waypoints; `q_rest` and the hub's `q` are the seeds of the controller's
  IK.

Checks, in `logs/20261103-scripts_cleanup/equiv/ik/` on the workstation, under Python 3.12 with numpy 2.4.4 and under
Python 3.10 with numpy 1.26.4 and scipy 1.12.0 (corvus's versions: a uv venv on the system's Python 3.10 with those two,
trimesh 5.2.0 and pyserial, in the session's scratchpad as `corvus_env`):
- `check_ur_ik.py`: fk equals the simulator's fk on 10,000 random joint vectors, bit for bit. fk(ik(T)) lies within
  1.5e-12 m and 1.8e-11 rad of T for all 71,416 solutions of 10,000 random reachable poses, and the generating joints
  are among the solutions. At and near the three singularities (4,200 poses and 16,000 at the shoulder) every solution
  is within 2e-12 m; poses at the wrist singularity with a stretched elbow have no solution when joint 6 is held at
  q6_des. Of 3,000 random poses in a 2.4 m box, 2,348 have none, and the damped least squares from 12 seeds reaches none
  of 400 of them. The rig's old `ik_all`, `fk_many`, `ik_near` and `branch_of` are bitwise equal on the 10,000 poses.
  Twenty changes of up to 1e-12 per pose (200,000 poses, two reference sets) change no branch and move joints by at most
  9.8e-10 rad on the poses 0.01 or more from a singularity. Poses rounded to 4, 5 and 7 decimals are solved through the
  projection. The UR5e's nominal table loads from a dict in metres and passes the same check; a calibrated table is
  refused.
- `cross_env.py`: the same 10,000 poses in both environments give the same branches; the joints differ by 5e-14 rad at
  most away from singularities and by 3.2e-12 rad at |sin q3| 1e-4 (numpy's arccos and arctan2 differ in their last
  bit).
- `check_sr2_fk.py`: against `ur5_full_k1.xml` at 1,000 random arm, gantry and finger configurations, tool0 agrees to
  5.7e-16 m, the tip frames to 2.2e-10 m and 6.4e-10 rad, the 3,054 pads to 2.3e-10 m. The four clockings' scenes agree
  to 1.9e-10 m. The scene with every `<f>_pip_frame` moved e further (-3, 5, 10 and 25 mm) agrees with
  `proximal_extension_mm=e` to 2.2e-10 m, and an extension moves the contact point by e along the MCP link to 2.8e-13
  mm. `SR2Hand.ik` solves 2,000 targets; `calibration_tool` and `ati_tool` equal the builders' matrices bit for bit.
- the rig (`rig/`): the table replays of seat 1 with the rod and with the 2 mm sphere give byte-identical plans, and
  path files equal but for the wall time. The plate cell's replay with the rod (`--cell v1030 --plate block`, where the
  homing's joints now come from `ik_nearest`) gives a plan equal in every number. `dry-check --real --chain` on the
  lab's two seat 1 program plans of 11-03 gives the same result in every number but the wall time. The cad_v3 scenes'
  `over_object` keyframes come out bitwise equal at k = 0 to 3.
- the builders (`plans/`), each built twice with the branch's code: every file equal to its twin but for the build's
  wall time. Against the base code's build: R24 pass 3, the holder plan, the wrist order, the tall pass and the holder
  path checks (the 10-29 side map, the 10-29 sphere stage 2) byte-identical, as are the v3 suite's tall2 and the
  indenter array's homing. The tall indenter's reach gains 12 twists at 40 deg and loses none. The trimmed-cap reach
  keeps every result; 357 failure reasons change from joint 6 to joint 5. The plate layout's replay differs in 3 of
  67,600 swept-map cells by one float32 step, the fingertip holder's homing by 6e-14 mm. The corvus Python builds the
  same tip and ATI plans; the gravity stations' axes differ by 6e-14 between the numpy versions. The holder path
  checker, move by move as the program makes them, gives the same 136 rows for the rebuilt q-seed sphere plan of each
  code, 39 flagged in both (30 for the joint windows the plans got after 10-22, 9 for the mount's clearance in this
  later cell model).
- `ikrec.py` records each `SIM.ik` call of a base-tree build with the closed form's answer beside it (`plans/ikrec/`).
- the simulator (`sim/`, `simrun.py` with the clock at 1762300000 s): the lab-cell sphere plan with `--q-seed`, built by
  each code and run by each tree, gives the same 310 events at the same times, all 154 actions ok; forces agree within
  2.0e-6 N, joints within 1.9e-8 rad, the registrations within 1.2e-6 mm and rad.
- pytest in the worktree: 113 passed and the same 2 failed as on the base.

What changed in the rebuilt plans, and why:
- tip plans with `--q-seed` (the lab's way since 10-22; the lab's sphere cell,
  `plan/20261022-plan_sphere_lab_turns_checked.json`, and rod cell, `plan/20261023-plan_rod_lab.json`): the same
  actions, twists and joint targets as the base code's rebuild, but one joint vector and two `joint6_deg` values at the
  0.1 deg rounding, and `sim.q_seed` by 4e-9 deg (the least squares' tolerance). Dropped gravity stations now say "joint
  5 outside the wrist window" where the least squares found no solution. Against the lab's files, both rebuilds differ
  alike, in 150 and 87 joint vectors, and the sphere cell's rebuilds have 10 more actions: the code since 10-22 (mount
  v7, the trimmed cap), not the IK.
- tip plans without `--q-seed`: in the lab's measured-rod cell, 12 actions now take twist -90 deg where the old build
  took -60, -120 or -150 deg: at -90 the least squares wandered whole turns or reached another branch, and the builder
  read no IK or a joint outside its window. Three travels between them now dip joint 5 to 27 to 30 deg, reported in
  `checks.travel_joint_paths_bad`. On the default sphere cell, two homing branches tie in the seed's cost (60 mm of
  clearance at the cap, joint 6 at -90 deg). The base code took joint 1 at -56 deg (branch 7) by its seeds' order; the
  closed form's branch order takes +70 deg (branch 0, the branch of the lab robot's seatings), and the plan changes with
  it. The seed now keeps the first branch when two costs lie within 1e-9.
- the path checks and pass plans: the same as the base code's, byte for byte. Against the lab's files of 10-25 to 10-30
  they differ alike in both codes: the holder path checks' margins by up to 0.005 deg and mm, the pass plans in their
  targets and twists (the code and the options since those files were built).
- the ATI plan (stations are moveJ targets): the same 41 gravity directions and per-eye counts. Two stations, st-23 and
  st-32, take another of the 24 yaws for the same direction, so their joints move by up to 10.7 deg. The -x block, with
  st-23 among its stations, visits them in the reverse order. A cycle and its reverse cost the same, and their sums'
  last bits had decided between them; the builder now keeps the first cycle in the permutations' order when two lie
  within 1e-9 rad (`ORDER_TIE`), so the workstation's and corvus's builds agree. The lab's plan of 10-23 has 39
  stations; the base code's rebuild already differs from it, through the least squares.

Open for the owner and the lab:
- the controller's own IK (`getInverseKinematics`) chooses by a metric UR does not document; the builders now model it
  as the Euclidean nearest. The lab's simulator still uses the damped least squares for its moves, as asked.
- the 12 twists and the three travels of the measured-rod cell's plan without `--q-seed`; the lab builds with
  `--q-seed`.
- the SR2 hand's gantry positions count from the CAD's travel-box centres (frames.json), 6.2 mm along x_P and 2.4 to 3.4
  mm along y_P from the driver's `FINGER_GEOMETRY` origins; the module has no conversion from the firmware's stepper
  millimetres.
- `lab/move_guarded.py` and `lab/leave_tall_key.py` still check their moves with `SIM.ik`.
- `UR5_CB3` limits the elbow to +-180 deg, as the cell's `ur5.urdf` does; the controller's own limit for every joint is
  +-360 deg. A wider limit is one field of `URParams`.

## Bugs found by the audits

| where (path before the move) | what | effect |
|---|---|---|
| `scripts/plant_drop_gate.py:79-86` | the yaw jitter multiplies q·dq, a turn about the tool's own shaft | `--jitter-yaw` left the heading at −90.000° in every draw |
| `scripts/ur5_tip_calibration_film.py:38` | `quat(R)` loses signs at half turns | films show robot links turned 120° wrong at some poses |
| `scripts/ur5_tip_calibration_holder_paths.py:70-71` | `rot_from_to` returns −I for antiparallel vectors | not reached by the holder plans so far |
| `scripts/probe_real_v1_chain.py:70-71` | `_align`, the same −I | no caller |
| `src/morphohand/rl/terms_common.py:198` | `_seg_seg_dist` gives 1.414 where the distance is 1.0 | not reached on the real_v1 hand's links |
| `scripts/policy_cpu_rollout.py:78` | `seg_seg` returns NaN for a zero-length segment | - |
| `scripts/fp_review_page.py:147`, `fp_review_figures.py:372`, `hom_plant_transfer.py:86`, `real_v1_screen_chain_page.py:46` | Spearman by ordinal ranks: ties depend on row order | at most 0.003 on the plant-transfer rows |
| `scripts/build_cad_v1_finger.py:70` | `unit()` has no zero guard | `common_perp` returns NaN for coaxial lines |

## Shared code in CLAUDE.md and AGENTS.md

Both have a short section (since `ab59888dd`): rotations, quaternions and rigid transforms come from
`sr2hand.geometry.rotations`; a missing function is added there and checked once against MuJoCo or scipy. When the
geometry and statistics modules exist, the section names them too. There is no guard test.
