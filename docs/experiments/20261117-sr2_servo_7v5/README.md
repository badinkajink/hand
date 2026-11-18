# SR2 hand servos at 7.5 V: how to rerun the free-air steps

The owner moved the SR2 hand's servos to a 7.5 V supply on 2026-11-17 and asked to see "if we can improve
stiffness/speed/force". The findings are in the bench runbook, `docs/experiments/20261115-hom_bench_trials/README.md`,
section "State on 2026-11-17 17:22". This folder holds the free-air step test and its data.

- `lab/step_test.py`: one joint at a time, from the hand's hold pose, through the station's policy stream with
  read-back at 50 Hz. The thumb's yaw steps +-20 deg, the index's and middle's +-15 deg, and each MCP and PIP flexes
  +20 deg. Every frame is clipped to the safe range (`sr2hand.control.servo_limits`). It needs the station homed,
  torque on and a plan loaded and applied (`hom_setup.py layout --go`). Without `--go` it only prints the sequence.
- `lab/step_analysis.py`: per step, the times to 10, 50 and 90 % of the step, the peak slew, the overshoot and the end
  error. The yaw holds of 2026-11-06 at 5 V sit beside them (samples 0.1 s apart).
- `data/20261117-164148_steps_speed80.jsonl`, `data/20261117-164227_steps_speed40.jsonl`: the two runs at 7.5 V.
  `data/20261117-step_responses.json`: the analysis of both.

Run from the repository on corvus:

    /usr/bin/python3 docs/experiments/20261117-sr2_servo_7v5/lab/step_test.py --speed 80 --go
    cd docs/experiments/20261117-sr2_servo_7v5
    /usr/bin/python3 lab/step_analysis.py data/*_steps_speed*.jsonl --out data/20261117-step_responses.json

Before the steps, check that no finger can reach another at the step poses. On 2026-11-17 at layout (105, 70) the
closest fingertip centres were 56 mm apart.
