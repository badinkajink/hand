# Requests between the workstation and the lab computer for the UR5 calibration of the fingertip

The owner set up two Claude Code sessions on 2026-10-22. The session on the lab computer runs the calibration of
`README.md` and pushes each run's data to GitHub. The session on the workstation polls GitHub every 30 s, analyses each
push and asks for further runs in this file. Only the workstation writes this file. The lab session answers in its
commit messages and in `lab/NOTES.md`, which only the lab writes, so no file is edited on both computers.

## What the lab computer pushes

1. Pull before each commit: `git pull --rebase origin main`. The pull brings this file, new plans and the
   `.gitattributes` rule that stores `docs/**/*.bin` in Git LFS. Without the rule, the `ft.bin` of a full session
   (about 130 MB) exceeds GitHub's limit of 100 MB a file, and the push is refused.
2. After every run of the program (`preflight`, `first`, `dry`, `sphere1` and each resume), copy its session folder
   whole to `lab/<YYYYMMDD>-<time>-<name>/` in this folder. The folder has `events.jsonl`, `registration.json`,
   `touches.json`, `pivots.json`, `session.json`, `plan.json` and the three streams `ft.bin`, `robot.bin` and
   `board.bin`. Add the program's printout: `tmux capture-pane -p -t tipcal -S -5000 > printout.txt`.
3. During `sphere1`, push the small files once more when the register and pivot stages have printed (all but the
   three streams). The registration is then checked here while the grid runs.
4. Write in `lab/NOTES.md` what the program does not log: the caliper reading of the stack from the flange face to the
   HEX-E's tool face (35.0 mm in the plan), the placement as set, the cable routing, what the operator saw or heard,
   and the start time of each run.
5. Give the commit message the run's name and outcome in one line, for example "UR5 lab: first (preflight, homing,
   gravity, mount): gravity 1.74 N, onsets within 0.4 mm".

## Plans written on the workstation

A new plan goes into `plan/` with its date and purpose in the file name. Plans are JSON, which this repository stores in
LFS, and the lab clone skips LFS content (`.lfsconfig`). Fetch one with
`git -c lfs.fetchexclude= lfs pull --include=docs/experiments/20261022-ur5_tip_calibration/plan/<file>`. A new plan runs
in the simulator before it runs on the robot (README rule 2): `$PY $P --plan <file> --sim --name sim-<name>`.

## Requests

Each request names what to run, why, and the result that would change the next step. Its status changes here when the
push that answers it arrives.

- R1 (open, 2026-10-22 19:30). Push the session folder of every run as above, beginning with `first`. The workstation
  compares each with Table 2 of the README and with the simulated session 151114-sim, and answers here within about
  10 minutes of the push.
