# manta

Python-driven robotic hand controller for the `real_v1` hardware prototype
(the gantry-base + servo-joint build referenced throughout
`docs/experiments/20260827-real_v1` and `docs/notes/perp_hardware_prep.md`),
built on a BIGTREETECH Manta M8P V2.0 + CB1. Custom STM32H723 firmware (no
Klipper) for 6 independent stepper axes (2 per finger, X/Y gantry rails via
a TMC5160 SPI driver per axis with StallGuard2 sensorless homing), plus a
separate Feetech SCS0009 servo subsystem (3 servos per finger -- yaw/MCP/PIP,
`aa`/`fe1`/`fe2` -- driven directly from the CB1 over a U2D2 USB adapter,
entirely independent of the STM32). Sibling to `../hand/` (the older
Arduino + L298N single-mechanism rig) under `driver/`.

## Status

Running on real hardware. All 6 stepper axes home reliably via StallGuard2
(see `docs/bringup.md` for the per-axis current/speed/SGT tuning process),
and all 9 servos (3 fingers x aa/fe1/fe2) move through calibrated ranges
per `host/manta_hand/servos.py`'s `FINGER_JOINTS`. The `manta_hand` Python
package (`host/manta_hand/`) is the current and only API -- an earlier
2-servo differential scheme has been fully removed, not just deprecated.
See `host/examples/movement_examples.py` for a runnable example of every
movement option (homing, gantry moves in both coordinate frames, per-joint
and combined servo moves, the real-time control-loop path).
`host/examples/hand_control.py` is an interactive text-command REPL for
day-to-day driving.

## Layout

- `firmware/` -- STM32H723 C firmware (HAL-based, CLI Makefile build).
- `host/` -- `manta_hand` Python package + examples, runs on the CB1.
  - `host/manta_hand/plan.py` -- the bridge to the simulator: a hand DESIGN and its open-loop
    grasp/turn, converted into gantry positions and servo angles. Its docstring is the record of
    which parts of that conversion are verified (the mount frame is the identity map; the joint
    names are forced by their ranges) and which are not (the aa sign, per finger). Pure
    arithmetic over this package's own calibration tables -- no serial, no mujoco -- so a plan
    can be checked on a workstation before anyone stands next to the hardware.
  - `host/examples/verify_frame_mapping.py` -- the three questions only the hardware answers:
    which physical block is which finger, which way a positive aa command actually swings a
    fingertip, and (resolved 2026-09-01, no shortfall -- see `kinematics.py`'s
    `FULL_EXTENSION_MM` comment) whether J1/J3/J5's apparent 4-6 mm travel shortfall was a scale
    error or a wall.
  - `host/examples/movement_examples.py` -- one example per movement option, run any one by name.
  - `host/examples/hand_control.py` -- interactive text-command REPL (day-to-day driving).
- `docs/` -- pin mapping, wire protocol spec, bring-up checklist.

## Network

The CB1 has a static IP on the lab network: `10.99.99.2` (SSH user `irlab`).

The current network control path is `host/manta_hand/web.py`: a token-protected HTTP/JSON
service with a browser UI, cached telemetry, buffered trajectory playback, and run logging.
See the repository-level `docs/hardware_control_station.md`. It does not auto-home on startup.
The older `hand_control.py` REPL still homes automatically and should remain a bench bring-up
tool, not the long-running network manager.

## Startup / power-up

Every power-up (and after any DFU-mode blip -- the M8P's USB-CDC device,
`/dev/ttyACM0`, can drop out and needs this to come back), the M8P's STM32
needs a manual DFU-then-normal reset cycle before the CB1 will see it as a
serial device at all:

1. Hold **BOOT0** down.
2. Press and release **RESET** (while still holding BOOT0) -- this boots
   into the ROM DFU bootloader.
3. Release **BOOT0**.
4. Press and release **RESET** again -- this boots the actual firmware
   normally (not DFU), and `/dev/ttyACM0` should enumerate on the CB1.

Once that's done:

```sh
ssh irlab@10.99.99.2
cd manta-hand/host/examples   # CB1's own checkout path -- update once the CB1 is redeployed from src/morphohand/driver/manta/
python3 hand_control.py                # interactive REPL
python3 hand_control.py "thumb_fe1 40" # one-shot command
```

`hand_control.py` homes all 6 stepper axes and zeros all 9 servos
automatically on startup before accepting any position command -- see its
docstring for the full command language. All stepper axis RAM state
(SETSCALE calibration, StallGuard SGT tuning, position zero-reference) is
lost on every M8P power cycle / DFU reset and gets re-established fresh
each run; nothing needs to be reflashed for this.

## Firmware build

```sh
cd firmware && make && make flash   # see docs/bringup.md for DFU steps
```

## Docs

- `docs/bringup.md` -- StallGuard2 tuning method, current/safety notes, known hardware quirks.
- `docs/pinout.md` -- Manta M8P pin mapping used by this firmware.
- `docs/protocol.md` -- USB-CDC text command reference (STM32 side).
- `docs/servos.md` -- finger servo subsystem: the current 3-servo independent-joint hardware (aa/fe1/fe2).
