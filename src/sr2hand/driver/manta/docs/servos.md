# Finger servo subsystem

Completely separate physical link from the M8P/STM32 firmware: a
Waveshare/Robotis U2D2 USB-TTL adapter on the CB1 (`/dev/ttyUSB0`, distinct
from the M8P's `/dev/ttyACM0`), driving Feetech serial bus servos (SCS0009
now, STS3215 planned for a future hardware version) directly. None of this
goes through the stepper protocol in `docs/protocol.md` at all. See
`host/manta_hand/servos.py` for the implementation and `docs/bringup.md`'s
physical setup section for the cabling.

## Current hardware: 3 servos/finger, independent

What's actually installed and calibrated right now, on all 3 fingers
(thumb/index/middle -- see `host/examples/hand_control.py`'s
`FINGER_NAMES`): one independently-driven servo per degree of freedom, no
shared or differential linkage between servos.

- `aa` -- adduction/abduction, its own dedicated actuator.
- `fe1` -- proximal phalanx flexion.
- `fe2` -- distal phalanx flexion, independently controllable from `fe1`.

`FINGER_JOINTS` in `servos.py` declares each joint as
`(servo_id, zero_deg, (min_rel_deg, max_rel_deg))`. `zero_deg` is that
servo's own measured raw position at its "0 degrees" reference (per-servo,
not interchangeable between units); the range is the intersection of the
originally-declared nominal contract with each servo's real measured
hardstop (from a manual torque-free sweep, 2026-08-29 -- see
`host/examples/servo_manual_range.csv` and `servo_calibration_notes.md` for
the raw readings). `Finger.set_joint(name, relative_deg)` and
`ServoBus.sync_set_joints({finger_id: {name: relative_deg}})` are the two
ways to move these joints -- see `host/examples/movement_examples.py` for a
runnable example of each.

- Servo IDs: finger 0 (thumb) = 0/1/2, finger 1 (index) = 3/4/5, finger 2
  (middle) = 6/7/8 (`aa`/`fe1`/`fe2` in that order).
- Calibration status: all 9 servos' `zero_deg` and effective range are
  measured and set, all 3 fingers.

Servo models and power specs are in `docs/bringup.md`'s physical setup
section, alongside the rest of the hand's power wiring.

## Superseded: 2 servos/finger, differential

An earlier design shared 2 servos per finger through a differential
linkage (one "flexion" DOF driving both servos together, plus a
common-mode "adduction" delta layered on top) -- adapted from Pollen
Robotics' AmazingHand reference code. It's gone from the codebase, not
just deprecated: there's no `FINGERS` dict, `Finger.set_flexion`,
`Finger.set_pose`, or `ServoBus.sync_move` to fall back to. If you're
reading old notes or `servo_calibration_notes.md` that mention any of
those, they describe hardware and API that no longer exist here.
