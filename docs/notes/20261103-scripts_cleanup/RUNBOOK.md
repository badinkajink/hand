# Scripts in topic folders and shared math modules: runbook (2026-11-03)

The owner asked at 11:0x on 2026-11-03 for "a repo clean up + refresh on the README", naming "the absolutely abysmal
scripts folder", and added: "also very concerned about us rewriting rotations, geometry, math helpers everywhere with
custom math instead of using proven library calls / a single help file(s) for these broad categorical functions".
At 11:5x the owner chose ten topic folders, an autopilot for the switch after the day's lab work, a shared math
module with the live copies migrated to it, and the 48 unreferenced scripts archived. The README and `.gitmodules`
were committed in `6eabaf56b`. The hand-off note is `docs/handoff/20261103-scripts_topic_folders.md`.

## Files in this folder

| file | what it is |
|---|---|
| `plan.json` | each script's folder; 652 scripts in ten folders and 48 to `archive` (templates and data files follow their script) |
| `move_scripts.py` | the switch: `git mv`, the root depth and `import _paths` in each moved script, every `scripts/<name>` in the live text, `_paths.py`, `MOVED.tsv`, the archive README, the map, the checker, and the special edits (README.md and CLAUDE.md lines, five computed paths) |
| `verify_scripts.py` | static checks, installed as `scripts/docs/check_scripts.py` |
| `scripts_map.py` | writes `scripts/README.md`, installed as `scripts/docs/scripts_map.py` |
| `smoke_imports.py` | imports every script with a main guard, each in its own process |
| `audit_rotations.md` | 309 hand-written rotation and transform copies in 128 files, the library call for each kind, four bugs |
| `audit_geometry.md` | 165 geometry and numerics copies (143 live), the library call for each kind, the bugs |

The dry runs' logs are in `logs/20261103-scripts_cleanup/` on the workstation.

## Dry run of 2026-11-03, in a worktree at 6eabaf56b

A worktree has LFS pointers in place of data and empty submodules, so its numbers differ from the main checkout's;
the two layouts were compared in two worktrees at the same commit.

- 754 files moved; references rewritten in 724 files. The tool lists 36 lines that use `__file__` for something other
  than the root; each was read (FreeCAD fallbacks, scripts that relaunch themselves, templates beside their builder).
- `check_scripts.py`: 780 files, 2,072 imports of scripts, all resolving to the moved file; 15 problems, each a
  reference to a script that did not exist before the move either: `d1_ilqr.py`, `hand_object_scale_export.py`,
  `perp_phalanx_variant.py`, `pinch_brake_page.py` and its template, `probe_contact_stiffness.py`,
  `real_v1_hand_plan.py`, `rl_record_reorient_fingertip_traj.py`, `servo_id_fit.py`, `servo_id_log.py`, the example
  `scripts/X.py` in `prose_lint.py`, and `lightning_grasp_runner.py`, `rl_demo_handoff.py`, `run6_combined_multitask.py`
  (in `archive/`).
- Imports: 573 scripts were imported in both layouts. None that imported in the flat layout failed in the moved one;
  three cluster scripts import only in the moved one, because `_paths` puts `cluster/` on `sys.path`.
- pytest: see the note's State for the comparison at 6eabaf56b. In the main checkout on 11-03 the suite gave 360
  passed, 25 failed, 3 errors; 23 of those are the Manta hardware tests without pyserial.

## The switch

Conditions, all at once (check each; if one fails, `ap snooze <job> 60 "<which>"` and check again):

1. The lab is idle: `docs/experiments/20261022-ur5_tip_calibration/lab/STATE.md` (after `git pull`) says no session
   runs and none is queued, the newest "UR5 lab" commit is over 45 minutes old, and `ssh` is not needed: the
   workstation never writes to corvus. If the lab runs overnight, wait for its morning commit.
2. No process on the workstation runs a file under `scripts/`: `pgrep -af "scripts/[A-Za-z0-9_]+\.(py|sh)"` lists
   none other than the crosstalk job's watcher, `crosstalk_feed_watch.py`, which is stopped with its job below. The
   fingertip viewer under `src/` may keep running: it imported its modules at start. Restart it after the switch only
   if the owner asks.
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
python3 scripts/docs/check_scripts.py             # expect exactly the 15 problems listed above
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --extra dev --with pyserial python -m pytest tests/ -q   # through resguard
git status --short | grep -v '^R ' | head -50     # read what changed besides the renames
```

The tests must fail no differently from the baseline of the same commit; run the baseline first if any commit came
in since 11-03 (`git stash` is shared between sessions, so take the baseline in a worktree:
`GIT_LFS_SKIP_SMUDGE=1 git worktree add --detach <dir> HEAD`, and run both in worktrees as the dry run did). Then
commit the switch as one commit (`git add -A scripts src tests configs .claude CLAUDE.md AGENTS.md README.md`, the
rewritten READMEs under `docs/experiments/`, the lab's `STATE.md`, `docs/hardware_control_station.md`,
`docs/rl/morph_sweep_STATUS.md`) and push. If the push is refused, `git pull --no-rebase`, rerun the checker, push.

After the push:

- Add a bullet under "Rules in force" in the lab's `STATE.md`: the scripts moved into topic folders at <time>
  (<commit>); corvus pulls before its next run; `scripts/MOVED.tsv` maps the old paths. Commit and push it.
- `ap step add hand-20261027-crosstalk_feed_poller --rule "Scripts moved into topic folders at <time> (<commit>):
  the claimed scripts are now under scripts/fingertip/ and scripts/docs/ (scripts/MOVED.tsv). Run them there,
  restart the watcher from scripts/fingertip/crosstalk_feed_watch.py, and correct the note's Claimed line."`, then
  `ap arm docs/handoff/20261027-crosstalk_feed_poller.md --at now` (it keeps its account and parallel flag).
- A memory file `project_scripts_topic_folders_2026_11_03.md` in the project's memory directory with its line in
  `MEMORY.md`: the folders, `_paths`, `MOVED.tsv`, the checker and the map.

## Shared math modules

Conventions, from `audit_rotations.md`: quaternions are wxyz as in MuJoCo, products are Hamilton, rotations are
active, angles are radians, and every function takes batches. The modules import numpy and scipy only: the `femtip`
environment has no MuJoCo, and corvus has scipy 1.12, so reorder quaternions by indexing (`q[..., [1, 2, 3, 0]]`)
and never pass `scalar_first`. `rot_between` keeps the explicit form of `ur5_tip_calibration.rot_from_to`, with the
antiparallel case turned about a perpendicular axis (a rotation, never −I).

- `src/morphohand/geometry/rotations.py`: quaternions, rotation matrices, Euler angles and rotation vectors, 4 × 4
  transforms and poses, Kabsch and Umeyama, mean rotation (`audit_rotations.md`, "Recommended module").
- `src/morphohand/geometry/vectors.py`, `distances.py`, `sdf.py`, `mesh.py`, `sampling.py` and
  `src/morphohand/numerics/stats.py`, `signal.py`, `angles.py` (`audit_geometry.md`, "Recommended layout"). Prefer
  the library call itself where one exists (scipy.stats.spearmanr, scipy.stats.bootstrap, trimesh); a module function
  is written only where no library has it.
- A test per module that compares it with scipy (and with MuJoCo's `mju_*` where MuJoCo is installed) on random
  inputs.

## Migration, folder by folder

For each folder of `scripts/` (and `src/morphohand/` outside `driver/`):

1. Replace each copy listed in the audits with a call to the module or the library. Keep a script's own function
   name as a one-line wrapper when other scripts import it.
2. Before deleting a copy, check it against its replacement on 10,000 random inputs, including the edge cases the
   audit names (antiparallel vectors, half turns, zero-length segments, ties), to 1e-12. Keep the check scripts in
   `logs/20261103-scripts_cleanup/equiv/`.
3. A copy that disagrees with its replacement is a bug or a different convention: leave it in place, and fix it in
   its own commit that names the pages and runs whose numbers it changes.
4. Run `scripts/docs/check_scripts.py`, `smoke_imports.py --names` on the touched scripts, and the tests; commit the
   folder.

Not migrated by the autopilot: `src/morphohand/driver/` and the UR5 programs that move the robot
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

## Guard against new copies

`tests/test_no_private_math.py` lists the functions in `scripts/` and `src/` whose names or bodies match the audits'
patterns (quaternion products, quaternion to matrix, axis rotations, Rodrigues, skew, segment distances, Spearman,
Fibonacci lattices) outside `morphohand.geometry` and `morphohand.numerics`, and fails when the list grows past an
allow-list kept beside it; the allow-list only shrinks. CLAUDE.md and AGENTS.md get a short section: rotations,
geometry and statistics come from these modules or the library under them, and a missing function is added there
with a test.
