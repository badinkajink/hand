# manta_hand (Python host package)

Runs on the CB1, talks to the firmware over `/dev/ttyACM0` (USB-CDC, see
`../docs/protocol.md`).

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[servo]'
```

```python
from manta_hand import MantaHandDriver

with MantaHandDriver("/dev/ttyACM0") as hand:
    hand.joints[0].set_current(run_ma=300, hold_ma=150)  # use your motor's real rating
    hand.joints[0].enable()
    hand.joints[0].move_to(3200, velocity=1600, accel=4000)
    print(hand.joints[0].status)
```

`examples/bringup_demo.py` is the first thing to run against real hardware
-- see `../docs/bringup.md`.

## Web control and experiment logging

`manta-hand-web` is the CB1 service and browser control station. It loads validated
`HandPlan` JSON, homes once per daemon session behind an exact confirmation, moves to a
morphology, commands the open/grip poses, buffers the reorientation on the CB1, and records
command/telemetry/event/object JSONL with instrument and manual-score summaries. When launched
with `--tracker-url http://10.99.99.50:8770`, every reorientation automatically owns a trace from
the workstation companion `scripts/real_v1_tracker_service.py`.

Exercise it without hardware first:

```sh
manta-hand-web --mock --host 127.0.0.1 --port 8765
```

The measured yaw signs are recorded in `plan.py`; real hardware requires a control token:

```sh
.venv/bin/manta-hand-web --token "$MANTA_TOKEN" --telemetry-hz 0
```

The recorded sim-to-servo yaw signs are thumb `+1`, index `-1`, middle `-1`, measured on
2026-08-29. `--aa-signs` can explicitly override them after a new hardware check. Full topology,
launch, telemetry benchmark, API, logging, and safety notes are in
[`docs/hardware_control_station.md`](../../../../../docs/hardware_control_station.md).

## Feetech SCS0009 servos (via U2D2)

Separate physical link from the Manta M8P -- the U2D2 is its own USB-TTL
adapter (typically `/dev/ttyUSB0`, not `/dev/ttyACM0`), talking directly to
the servo bus, nothing routed through the STM32 firmware. Built on
`rustypot`, the same library Pollen Robotics' own AmazingHand reference code
uses -- not Feetech's own SDK. See `manta_hand/servos.py`'s module
docstring for why.

3 fingers, 3 independently-driven servos each -- one servo per DOF: `aa`
(adduction/abduction), `fe1` (proximal phalanx flexion), `fe2` (distal
phalanx flexion). Finger 0 = IDs 0/1/2, finger 1 = IDs 3/4/5, finger 2 =
IDs 6/7/8 (`FINGER_JOINTS` in `servos.py`). Each joint moves in plain
degrees relative to its own calibrated zero reference, bounds-checked
against that joint's real measured range before anything moves.

The `servo` extra is installed by the virtual-environment command above.

```python
from manta_hand import ServoBus

with ServoBus("/dev/ttyUSB0") as bus:
    finger0 = bus.finger(0)
    finger0.enable()
    finger0.set_joint("fe1", 45.0, speed=200)  # degrees relative to fe1's zero
    print(finger0.servo(0).status)

    # Move joints across multiple fingers in one synchronized bus
    # transaction (~1ms for all servos involved) instead of looping
    # individual moves, which is visibly staggered across servos:
    bus.sync_set_joints({0: {"fe1": 0.0}, 1: {"fe1": 0.0}, 2: {"fe1": 0.0}})
```

See `examples/movement_examples.py` for a runnable example of every
movement option, including this one (`move_single_servo_joint`) and the
real-time multi-joint path (`move_all_joints_realtime`). Each SCS0009
needs a unique bus ID assigned one servo at a time on an otherwise-empty
bus (they ship at ID 1 by default) before wiring them all together --
`examples/set_servo_id.py` / `set_servo_id_scs0009.py` do that.
