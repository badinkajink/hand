# Runbook of the fingertip calibration session on the UR5 (CB3) with the OptoForce HEX-E

This runbook covers one calibration session of the magnetic fingertip, run from a fresh clone on the UR5's computer.
The steps are a setup check, a dry run 15 mm off the surface, the session itself and the transfer of its folder back to the
workstation. The dry run takes about 28 min and the session about 60 min. The runbook is written for the operator and
for a Claude Code session started in the clone. The protocol, the frames and a simulated session are on the page
`docs/experiments/20261022-ur5_tip_calibration/20261022-ur5_tip_calibration_protocol.html`: Section 2 covers the tool
stack, Section 3 the cell and Section 4 the stages. The setup script pulls that page.

The program is `src/morphohand/driver/reskin5x/host/ur5_tip_calibration.py`. It reads the plan
`plan/20261022-plan_sphere.json` (159 actions, built on the workstation by `scripts/ur5_tip_calibration_plan.py`). It
writes every stream and event to a session folder `logs/<date>-tip_calibration/<time>-<name>/`.

## Intent of the session

The owner asked for the TCP to be "super dialed in" (2026-10-22). Slides, rolls and twists turn the pad about points
on the sphere, so an error in the pad's frame on the flange turns into slip and false shear. Homing on the indexer, the
gravity sweep, the 38 touch-offs and the registration fit measure that frame, and every later result depends on them.
When their printouts are off Table 2, stop and find the cause before the grid.

The session then gives the pad's response to a 5 mm sphere at known places, depths and motions, for three uses:

- ground truth for the board's contact estimator with the present, weak magnets; with them it fails below 1 N, so the
  0.25 and 0.5 N levels feed only the uses below. The N52 magnets get the same session later;
- the stiffness of the printed pad against its FE model: the push against depth at 68 grid points, 0.25 to 5 N;
- the sphere-pad contact model of the simulators (1 mm pads, builder v2): push against depth, shear against slide, the
  rolls, and the torsion loop of the twists. mount-0 and mount-2 push the rigid column to 5 N, which measures the HEX-E's
  deflection (0.19 mm of a 5 N face press by the data sheet). The twists' torsion, about 2 N mm at 2.5 N, sits under
  about 230 N mm of the push's moment at the HEX-E, so the analysis takes the difference of the +20 and -20 deg turns.

Four choices come from the simulated sessions and stay as they are:

- the pivot presses are a check: folded into the fit they moved the contact point from 0.065 to 0.35 mm off in
  simulation, because the board's offsets are biased about 0.1 mm;
- the indenter sits on the 38 mm block: on the plywood alone the wrist comes within 5 to 7 mm of it, and 24 of 161
  actions fail the clearance checks;
- the program polls the HEX-E itself: magpie_control's reader of 2024 returned stale samples, which spoiled the force
  data of the owner's earlier paper;
- the neighbouring fingers' fields come later, as a calibration against the hand's kinematics; this session has one
  finger on the robot.

If the session must be cut short, keep homing through the register stage and the grid. The slides, rolls and twists
come next, and the final reference press and gravity sweep last.

## Rules for a Claude Code session on the lab computer

**Changed for the night of 2026-10-22 (owner).** After the TCP check passed, the owner handed the runs to the lab
computer's session ("once that passes, Ill trust you to run things autonomously ... you can stop whatever the hell youre
doing (+ the ur5s built in safety stop)"). That session now:
- starts the program itself, with `--home-from` in place of the freedrive seatings;
- moves the robot with `lab/move_guarded.py` between runs;
- commits and pushes;
- leaves the heavy analysis to the work computer's session, which polls the repository.

`lab/README.md` holds the cell as measured, the plan for it, and the commands. The rules below still hold where they
don't conflict: one reader on the DAQ, simulator first, and Table 2 after each run.

1. The operator starts every run of the program in a tmux session named `tipcal`. The program asks for Enter during
   homing and after a protective stop, and a Claude Code shell has no keyboard. Claude reads the program's output with
   `tmux capture-pane -p -t tipcal -S -300` and reads the session folder's `events.jsonl`. It never sends keys to that
   pane.
2. Only the program moves the robot. Claude writes no RTDE or URScript commands of its own and does not edit the plan's
   actions. A change to the program runs in the simulator (`--sim`, step 5 of the setup script) before it runs on the
   robot.
3. Only one program talks to the HEX-E's DAQ. A second reader sends its own rate and bias commands, and a bias command
   zeroes the sensor in the middle of the session. Examples are a ROS `ft_sensor_node` or a magpie_control script.
   Stop them before the session.
4. After each run, compare the printout with Table 2 and tell the operator what differs before the next command.
5. The workstation's memory guard (`~/.claude/bin/resguard.sh`) and its usage-window hand-offs do not exist on this
   computer. The program is one process of about 250 MB.
6. After each run, the session folder goes to GitHub as `QUEUE.md` describes. The workstation's session analyses each
   push and asks there for further runs (owner, 2026-10-22). A change to the program is pushed as well, with the
   simulator run that tested it named in the commit message.

## Hardware, network and tool stack

Table 1. The cell in the plan.

| Item | Value |
|---|---|
| Robot | UR5 CB3; RTDE at 192.168.0.4, 125 Hz; the program sets the RTDE TCP to the flange and the payload itself |
| F/T sensor | OptoForce HEX-E (HEX-70-XE-200N of 2017); Ethernet DAQ on UDP port 49152 at 192.168.0.5 (magpie_control's address) or 192.168.0.3 (magpie_force_control's); the setup script probes both |
| Tool stack | HEX-E 35.0 mm from the flange face to its tool face (data sheet V1.0), mount v3, DIP link b v4 with the board, pad, cap TW15H3 on the tip end |
| Pad face centre | (10.55, 0, 128.36) mm from the flange, in the flange frame |
| Payload | 0.384 kg at (0, -0.5, 32.5) mm from the flange (HEX-E 260 g, mount and finger 124 g) |
| Board | ReSkin 5X on the QT Py ESP32-S3; USB serial `/dev/serial/by-id/*QT_Py*`; 140 records/s |
| Indenter | the sphere indenter with its homing key, on a 38 mm block (a 2 by 4 laid flat) clamped on 18 mm plywood |
| Placement | the indenter's origin at x 0, y -450, z 56 mm in the robot's base frame; its x axis along the base's -y, away from the robot; within 20 mm, 5 deg of yaw and 2 deg of tilt |

The plan's reach and clearance were checked at this placement only. Homing measures where the indenter is and stops the
program past those limits. The plywood is then moved, or the plan is rebuilt on the workstation with
`scripts/ur5_tip_calibration_plan.py --placement x,y,z,yaw --block 38` (20 s). The program prints the measured values for
`--placement`. The tip end of the pad cannot be pressed, because the cap covers it.

## Setup of the lab computer

```
git clone --depth 1 git@github.com:badinkajink/hand.git ~/hand      # private repository: SSH key or token first
cd ~/hand
bash scripts/ur5_tip_calibration_lab_setup.sh                      # --ft IP, --robot IP, --venv DIR, --python PATH
```

The clone skips every git-lfs file (`.lfsconfig`). The setup script changes nothing on the robot and does six things:

- it pulls the plan, the protocol page and the simulator's four data files through git-lfs (17 MB);
- it makes a venv `~/.venvs/tipcal` with numpy, scipy, pyserial and ur_rtde; `--python PATH` uses an interpreter that
  already has them, such as magpie_control's;
- it lists every copy of magpie_control's OptoForce reader under `$HOME` and marks the old one;
- it checks the robot's ports 30002 and 30004 and asks 192.168.0.5 and 192.168.0.3 for one HEX-E sample;
- it looks for the board under `/dev/serial/by-id` and checks that the user may open it (group `dialout`);
- it runs preflight, homing and gravity in the simulator, which takes about 10 s.

Each check prints `ok` or `FIX`. Fix the FIX lines and run the script again. Its last lines give the `PY` and `PLAN`
values used below.

The setup script marks a reader as old when it lacks `VALID_RATES`. That reader is magpie_control's from 2024-11 to
2026-03. It sleeps 20 ms after each request, so it reads at most 50 Hz, and its socket keeps late answers. Each late
answer then passes for a newer one, so the readings fall further behind over a run. Niraj Pudasaini's commit 806b94a
(2026-03-06) replaced it on main, and the C++ driver in magpie_force_control got the same change that day. The calibration program
reads the DAQ itself and needs neither. An old checkout still goes to main before anyone uses it:

```
git -C <checkout> status --short          # local changes first
git -C <checkout> fetch origin && git -C <checkout> switch main && git -C <checkout> pull --ff-only
```

## Setup of the cell

1. Measure the stack with calipers, from the flange face to the HEX-E's tool face. The plan has 35.0 mm. Another value
   goes into the plan at `geometry.T_EM[2][3]` (mm). Record the measured value in the session's notes either way.
2. Mount the HEX-E on the flange through its built-in UR adapter, then mount v3 on the HEX-E. The washers under the
   M6 x 20 are covered in Section 5 of mount v3's page. Then fit the DIP link with the board and the pad into the mount,
   and the cap on the tip end.
3. Leave slack in the board's USB cable and the HEX-E's cable. The flank presses turn the tool 90 deg about its axis.
4. Clamp the plywood and the block, and put the indenter at the placement of Table 1. To find x and y, hold the flange
   over the indenter's centre in freedrive and read the pendant's Base coordinates, with the pendant's active TCP at
   zero.
5. On the pendant: the robot initialized, brakes released and no program running (ur_rtde uploads its own script). Put
   the speed slider at 30 % for step 2 and the first minutes of step 3. The operator keeps the E-stop in reach.

## Running the session

All commands run in `tmux new -s tipcal`, from the clone's root, with these variables (the setup script prints the
first two):

```
PY=~/.venvs/tipcal/bin/python
PLAN=docs/experiments/20261022-ur5_tip_calibration/plan/20261022-plan_sphere.json
P=src/morphohand/driver/reskin5x/host/ur5_tip_calibration.py
```

1. Check the streams and set the payload, with no motion (5 s):
   `$PY $P --plan $PLAN --stages preflight --name preflight`
2. Run homing, gravity and the mount's touch-offs at 30 % speed (about 5 min). Homing asks three times for the mount's
   socket to be seated on the indenter's key in freedrive: palmar face down, the finger along the indenter's +x. Press
   Enter while it is seated. Watch the first touch-off with a hand on the E-stop.
   `$PY $P --plan $PLAN --stages preflight,homing,gravity,mount --name first`
3. Run the whole session 15 mm off the surface, to check reach, cables and the turns (about 28 min). Homing runs again
   in every new session.
   `$PY $P --plan $PLAN --dry 15 --name dry`
4. Run the session (about 60 min at full speed):
   `$PY $P --plan $PLAN --name sphere1`
5. Resume after an interruption (Ctrl-C, three aborts in a row, a stop of the program). Finished actions and the homing
   are kept; aborted actions run again.
   `$PY $P --plan $PLAN --resume logs/<date>-tip_calibration/<time>-sphere1`

`--step` asks for Enter before every action. `--ft IP` and `--board PORT` override the addresses.

Table 2. What each stage prints and what to expect. The values in brackets come from the simulated session 151114-sim
of 2026-10-22, whose tool had errors injected (the mount turned 0.9 deg, the pad 0.3 mm off its seat, the HEX-E turned
7.3 deg).

| Stage | Printout | Expected | Otherwise |
|---|---|---|---|
| preflight | rates of the HEX-E, the board and the robot | 500 Hz, 140 records/s, 125 Hz | the DAQ's address (`--ft`), the board's cable or port (`--board`) |
| preflight | HEX-E noise, 1 sigma | about 0.035, 0.035, 0.15 N and 0.002, 0.002, 0.001 N m (data sheet) | several times higher: a loose stack, a cable pulling, a pump or fan on the table |
| preflight | HEX-E reads | 0 timed out, 0 late datagrams | another program polls the DAQ, or the link loses packets |
| homing | spread of the three seatings | under 0.2 mm and 0.2 deg (0.08 mm, 0.00 deg) | the socket was not fully seated: run homing again |
| homing | indenter position, yaw, tilt | within 20 mm, 5 deg and 2 deg of (0, -450, 56) mm and -90 deg ((2.4, -451.3, 55.6) mm, -89.7 deg, tilt 0.79 deg) | the program stops: move the plywood, or rebuild the plan |
| gravity | weight and centre of mass below the HEX-E | 1.2 to 2.0 N: 124 g of mount and finger plus the HEX-E's tool side (1.808 N with 184 g simulated); residual under 0.02 N (0.006 N) | a cable pulls on the tool: free it and run gravity again |
| mount, pad | onset of each touch-off from the expected | within 1 mm (-0.28 to +0.63 mm) | over 2 mm: the stack's height, the mount's seat or the indenter moved; stop and check |
| mount | mount-0 and mount-2 push the column on to 5 N | no abort; the travel past the onset is about 0.12 and 0.16 mm by the data sheet (0.16 mm for mount-2 in simulation) | an abort at 5 N: the column or the mount is loose |
| register | rms of the pad and face touches; change of the TCP | under 0.05 mm; about 1 mm and 1 deg (0.007 mm; 0.99 mm, 1.11 deg) | rms over 0.1 mm: a touch-off slipped or the pad moved in its seat |
| pivot | board and registration apart | under 0.5 mm (0.22 mm) | a WARNING line: check the mount's seat and the stack's height before the grid |
| final | weight of the second gravity sweep | within 0.01 N of the first (1.802 against 1.808 N) | the pad or a cable moved during the session |

## Stops and aborts

Each action watches the HEX-E and the robot. An action aborts for any of these reasons:

- a load over 10 N or 1.5 N m during a press;
- a load over 8 N while travelling;
- no HEX-E sample for 100 ms;
- a protective or emergency stop;
- no IK solution, or a target outside the 400 mm box around the indenter;
- a travel over 60 s, or the robot stopping away from its target.

After an abort in a protective stop, the program waits for the operator to clear the stop on the pendant and press
Enter. It then lifts the tool 5 mm away from the indenter and goes on to the next action. Three aborts in a row stop the program; step
5 resumes it. Ctrl-C stops the robot and saves the registration.

## After the session

The session folder is about 270 MB. It reaches the workstation through GitHub as `QUEUE.md` describes; the copy below
is the fallback. Copy it to the workstation's `logs/20261022-tip_calibration/` (or a folder dated
with the session's day), with scp, rsync or a USB drive. On the workstation:

```
.venv/bin/python scripts/ur5_tip_calibration_analysis.py logs/<date>-tip_calibration/<session>
.venv/bin/python scripts/ur5_tip_calibration_page.py collect logs/<date>-tip_calibration/<session>
.venv/bin/python scripts/ur5_tip_calibration_page.py
```

The analysis gives the ground truth of every hold and the estimator's scores. `collect` copies the session's small files
into this folder's `data/` and writes `results.json`, and the last command rebuilds the page.
