# SR2 hand servos in free space, 11-06

The SR2 hand hangs on the UR5 with nothing in its fingers. The owner, leaving at 12:2x on 11-06: "maybe we can
double-check our sysid? the hands are only in free space right now, so we should be able to reproduce the previous
free-space only sysid". The coordinator reviewed the plan at 12:43 and set its blocks, their order and the checks.
The run is unattended. Corvus drives the CB1's control station, as for the crosstalk sweep
(`docs/experiments/20261104-sr2_hand_crosstalk/`, whose `lab/` holds every script named here).

## Blocks

Every joint not named stays at zero. One joint moves at a time, at speed 80, at the gantries' home layout. No stepper,
arm or contact moves.

| block | what | dwell |
|---|---|---|
| a1, a2 | 09-02's staircase (`scripts/archive/servo_hysteresis_sweep.py`) on each mcp and pip: 5 to 61 deg and back in 4 deg steps, the top twice; the index's mcp to 49 | 0.45 s (a1, as on 09-02); 2 s (a2, the steady droop) |
| b1, b2 | the same staircase on each yaw, -36 to +36 deg and back | 0.45 s; 2 s |
| c | yaw stiction: each yaw 0, +A, 0, -A, 0 for A = 20, 30, 40 deg | 10 s each |
| c40 | c with the tested finger's mcp at 40 deg | 10 s each |

The 09-02 sweep differs in three ways. Its other flexion joints stood at 5 deg, here at 0. It read the angle 0.45 s
after each command; run_plan.py starts the dwell once a sample is within 2 deg, so a1's dwell ends 0.75 s or more after
the command. And on the real_v1 hand of 09-02 the yaw carried no gravity load, while on the SR2 hand it does.

## Safety

- Every motion goes through `lab/run_plan.py` with `--ramp-deg 0 --min-clearance 5 --max-blocked 1 --keep-home`.
  `--max-err-yaw` is 10 for a and b, 15 for c and c40. A blocked pose ends the chain until the owner looks, and the
  servos stop at 60 C.
- `lab/make_sysid_plan.py` checks every pose and every move in the CAD hand at four positions of the middle finger's
  gantry. These are home (80, -85), cover (25, -37.5) and the two corners between them, in palm mm, since one axis may
  have skipped 11-05's last move. The least clearance in the tolerance boxes is 20.2 mm, against a 12 mm margin
  (`20261106-sysid_clearance.txt`).
- The registers are read last, with the station stopped. `lab/scs0009_registers.py` sends only READ instructions.
  Then `lab/rearm_home.py --read-only`, and the station stays stopped: servo torque off, fingers hanging. No re-arm
  until the owner has looked at the middle finger's stages.

## Files

GitHub refuses LFS transfers since 21:37 on 11-05 (the repository's LFS budget). So this folder holds only plain text,
and corvus keeps the full logs in `logs/20261106-sr2_sysid/`: each block's events.jsonl and telemetry.jsonl, the plans'
JSON, the D435 frames at the start and the end, and the registers' JSON. The board logger keeps logging into
`logs/20261105-sr2_crosstalk/boards/`.

- `20261106-sysid_plan_steps.tsv`: every step of every block, with its clearance at home and in the boxes at the four
  gantry positions.
- `20261106-sysid_clearance.txt`: the check's least clearances per block and gantry position.
- `data/dwells.tsv`: one row per step. It gives the block, pass (a1 and b1 are pass 1, a2 and b2 pass 2), step, tested
  finger, joint, direction and command (plan degrees and servo degrees). Then the runner's times (t_cmd, settle_s,
  t_settled, t_end, corvus's clock), error, measured clearance and blocked flag. Last, the dwell's sample count and the
  mean and sd of each of the nine servos' angle and load.
- `data/holds_c.tsv`: blocks c and c40, the moving yaw's samples from each command to the step's end. Each sample has t
  from the command, the angle and the load.
- `data/registers.txt`: the nine servos' registers with their raw bytes.

Angles are the station's servo degrees from each calibrated zero; the index's and the middle's yaw are minus the plan's
yaw. Loads are the present load, in per mille of full duty with the sign. Telemetry runs at about 8.6 Hz, and each
sample's time is its receipt less its age.

## Rerun

```
cd docs/experiments/20261104-sr2_hand_crosstalk/lab
uv run --no-project --python 3.10 --with mujoco --with numpy --with scipy python make_sysid_plan.py ../../../../logs/20261106-sr2_sysid/plan
cd ../../../.. && nohup setsid bash docs/experiments/20261104-sr2_hand_crosstalk/lab/sysid_chain.sh >> logs/20261106-sr2_sysid/chain.out 2>&1 &
```
