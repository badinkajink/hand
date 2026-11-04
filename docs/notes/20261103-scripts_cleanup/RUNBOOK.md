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

## Files in this folder

| file | what it is |
|---|---|
| `plan.json` | each script's folder; 652 scripts in ten folders and 48 to `archive`; a value `folder/new_name.py` also renames (`build_morphohand_urdf.py`, `graspgenx_make_morphohand.py`) |
| `move_scripts.py` | the switch: `git mv`, the root depth and `import _paths` in each moved script, every `scripts/<name>` in the live text, `_paths.py`, `MOVED.tsv`, then the package rename, the archive README, the map, the checker, and the special edits |
| `verify_scripts.py` | static checks, installed as `scripts/docs/check_scripts.py` |
| `scripts_map.py` | writes `scripts/README.md`, installed as `scripts/docs/scripts_map.py` |
| `smoke_imports.py` | imports every script with a main guard, each in its own process |
| `wait_switch_conditions.sh` | waits until the mechanical conditions of the switch hold |
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
.venv/bin/python scripts/docs/check_scripts.py         # the 15 known problems
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
(`scripts/ur5/ur5_inverted_rig.py`, the plan generators and `ur5_tip_calibration_holder_paths.py`). They wait for a
window with the owner and the lab, whose simulator checks every plan. `scripts/archive/` is never migrated.

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
