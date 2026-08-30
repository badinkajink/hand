# MorphoHand hardware control station

This is the operator and experiment-recording path for the `real_v1` hand. The browser and
any policy inference run on the workstation; one small service on the CB1 owns both serial
links. Open-loop trajectories are sent as plans and timed on the CB1, so Ethernet jitter does
not become servo jitter.

```text
workstation                              CB1 (10.99.99.2)              hand
browser UI ───── HTTP/JSON ─────┐
local inference client ─────────┴──> HandRuntime ── USB-CDC ──> 6 gantry steppers
                                      │
                                      └──────── U2D2 TTL ────> 9 SCS0009 servos
```

`HandRuntime` is the sole hardware owner. UI refreshes read a cached status document; they do
not generate servo reads. Only one homing, gantry, pose, trajectory, or policy-stream operation
can own the hand at a time.

## The 2026-08-29 incident, and what it changed

The first bench session with this service went: home the hand, become suspicious of how long
an axis was grinding, press **Stop motion**, watch the home continue anyway, press **Move
gantries to morphology**, lose the service. Afterwards the M8P was queried directly and had
never crashed — `ttyACM0` was alive, all six axes read `enabled`, and four of six reported a
StallGuard2 stall. Every fault was in the host software.

Four defects, all fixed, all now covered by `tests/test_manta_hardware_faults.py`:

1. **Home never zeroed the servos.** `examples/hand_control.py` enables all nine servos before
   it calls `home_all()`; the service did not. A torque-OFF SCS0009 *accepts a goal-position
   write and reads it back correctly without moving*, so `zero_joints()` succeeded silently
   having moved nothing. The backend now enables torque at connect, `Hand.home_all` takes
   `require_torque=True` and refuses to run otherwise, and a pose command is rejected outright
   while torque is off.

2. **Stop did not stop a home.** The host's homing loop had no cancellation check, so `STOPALL`
   only ended the axis that happened to be moving and the sequence walked on to the next five.
   Worse, the firmware's `Stepper_Stop()` sets `target = position` but leaves `homing_result`
   at 1; the supervisor tick later flips it to 3 ("timed out"), and the host read 3 as the
   *timeout guarantee* and called `ZERO`. A cancelled home therefore planted step-0 in the
   middle of a rail, and `MOVEMM` — which has no stall protection — would then drive away from
   that bogus origin. Cancellation is now cooperative down to `kinematics._home_one_axis`; an
   aborted axis is stopped, **disabled** (the only thing that clears `homing_result`) and never
   zeroed, and the session's `homed` flag is dropped with the reason recorded.

3. **The morphology move fired all six axes at once.** `plan.apply_mounts` issued six `MOVEMM`s
   back to back and waited on none of them. Nothing validated on this hand has ever moved more
   than one axis at a time: six simultaneous starts means six simultaneous `TMC5160_StartMotionKick`
   current kicks (IRUN 1 → 7 for 500 ms each) on the 19 V rail that also backfeeds the CB1, and
   six axes of step ISRs at NVIC priority 2 preempting the USB interrupt at priority 3 while the
   host polls `STATALL` — nine CDC packets — over that same link at 10 Hz. It now moves one axis
   at a time and waits for each, and the settle poll uses single-axis `STAT`.

4. **Nothing distinguished "command refused" from "the board is gone".** A serial timeout landed
   in `last_error` while the session still claimed to be homed with its mounts applied, so the
   next command acted on a reference that no longer existed. Link failures now latch, invalidate
   the home, and demand an explicit `POST /api/v1/reconnect`.

The HTTP layer was hardened alongside: a rejected token used to return without reading the
request body, which desyncs a keep-alive connection and makes every subsequent request on it
fail — a browser reports that as "Failed to fetch" with nothing wrong server-side.

### Homing takes minutes, and that is expected

Homing is sequential and each axis gets a timeout sized to guarantee it covered its full
measured travel (`kinematics._home_timeout_ms`). Worst case for all six is **176 seconds**:

| axis | finger | travel | home window |
|---|---|---|---|
| J0 | thumb x | 112.4 mm | 46.4 s |
| J1 | thumb y | 56.2 mm | 22.4 s |
| J2 | index x | 62.5 mm | 28.0 s |
| J3 | index y | 56.0 mm | 25.0 s |
| J4 | middle x | 62.2 mm | 29.4 s |
| J5 | middle y | 54.1 mm | 24.4 s |

An axis whose StallGuard2 fires finishes early. An axis whose StallGuard2 does *not* fire runs
its whole window pressed against the hardstop, and the home is still trustworthy because the
window cannot expire before the full travel has been covered. **On this hand, J3 and J5 do not
trip StallGuard2** — confirmed on hardware 2026-08-29 by reading the board after a
`hand_control.py` home: `homing_result` was 2 on J0/J1/J2/J4 and 3 on J3/J5. That is the
current SGT tuning (see `kinematics.HOME_COOLCONF`), not a fault, and it is why homing sounds
alarming. The UI now shows the per-axis outcome and the expected window so the two cases can be
told apart while you are watching.

## Trying it without hardware

Two different things, and the difference matters:

```bash
python -m manta_hand.web --mock   # replaces the BACKEND: no driver code runs at all
python -m manta_hand.web --fake   # replaces the DEVICE: the whole driver stack runs
```

`--fake` puts a simulated M8P on a pty (`manta_hand.fake_hardware`) and a simulated SCS0009 bus
behind a real `RealHardwareBackend`, so `MantaHandDriver`, `Joint`, `Gantry`, `Hand`, `ServoBus`
and every serial round trip execute for real. It models the behaviours that caused the incident:
`STOP` leaving `homing_result` untouched, `DIS` clearing it, torque-off servos accepting writes
without moving, `--fake-stall-axes` (default `0,1,2,4`, matching the real hand), `drop_lines` for
a short `STATALL`, and `answering = False` for a board that has fallen off the bus. Use this, not
`--mock`, to reproduce anything involving homing, gantry motion, or the link.


## What is deployable now

| Control method | Status | Feedback requirement |
|---|---|---|
| Morphology move | Ready after one home | Stepper status only |
| CEM grasp keyframe | Ready after yaw-sign check | None during motion |
| CEM open-loop reorientation | Ready after yaw-sign check | None; buffered/timed on CB1 |
| Workstation joint-command stream | Experimental, write-only | None; holds last goal if lease expires |
| Learned A/B RL policies | Not deployable closed-loop | Missing most of their observation vector |

Feedback available on this hand, measured: **nine joint positions at 111 Hz** (or 55 Hz
with the load proxy alongside), and nothing else. No object pose, no contact, no current.
The rate is no longer the blocker it was assumed to be — see the telemetry section — but
the observation vector still is.

The learned actor observation is 65 values for A and 66 for B: 15 robot joint positions, 15
joint velocities, relative object position, actual object pose, reference finger/object poses,
and the previous 9 actions; B adds target-axis misalignment. The current prototype can measure
only nine finger joint positions. It has no palm stage, object pose/velocity, or contact sensor.
Feeding zeros or simulation references for those missing values would be an out-of-distribution
policy test, not closed-loop deployment.

## First run: mock mode on the workstation

From the repository root:

```bash
PYTHONPATH=src/morphohand/driver/manta/host \
  python -m manta_hand.web --mock --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765`. The full state machine, UI, event stream, command timing, logging,
and manual scoring run against a deterministic fake hand. Use this mode when editing the UI.

## CB1 install and launch

Copy/pull the repository on the CB1, then install the small host package and servo extra in a
virtual environment. Debian's system Python is externally managed; do not use
`--break-system-packages`:

```bash
cd src/morphohand/driver/manta/host
sudo apt install python3-venv
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[servo]'
```

The yaw signs were measured on hardware 2026-08-29 and are recorded as thumb `+1`, index `-1`,
middle `-1`. Re-run `examples/verify_frame_mapping.py` if a servo/module is rewired or replaced.
After the documented M8P BOOT0/RESET cycle makes `/dev/ttyACM0` appear, launch with:

```bash
export MANTA_TOKEN="$(openssl rand -hex 16)"
.venv/bin/manta-hand-web \
  --host 0.0.0.0 --port 8765 \
  --stepper-port /dev/ttyACM0 --servo-port /dev/ttyUSB0 \
  --plans-dir /path/to/hand/docs/experiments/20260829-real_v1_deploy/deploy \
  --logs-dir /path/to/hand/logs/hardware \
  --telemetry-hz 5 \
  --log-file ~/mantalogs/web.log \
  --token "$MANTA_TOKEN"
```

`--telemetry-hz 5` is safe now that the read path works and the FTDI timer is tuned
(measured ceiling 111 Hz; 5 Hz ran 11,115 transactions with zero timeouts during a live
home). Raise it if you want, but benchmark first and remember that servo polling is
suspended while a writer owns the bus. `--log-file` is not optional in practice — see
"Safety and recovery notes".

`--aa-signs thumb:+1,index:-1,middle:-1` remains available as an explicit runtime override after
a fresh measurement, but is no longer required for the recorded hardware configuration.

Real hardware requires a command token. The service is not an internet-facing application;
keep it on the isolated lab network and firewall port 8765 from other interfaces.

Open the UI from the workstation with:

```text
http://10.99.99.2:8765/?token=<MANTA_TOKEN>
```

The token is retained in that browser's local storage. The normal operator sequence is:

1. Load a candidate. Loading validates every mount and every pose before movement.
2. Press **Home gantries**, type the exact confirmation, and stay with the hand. It takes
   about two minutes and the UI shows which axis it is on and how long that axis's window
   is. Four of six axes stall out in ~15 s; J3 and J5 grind for ~33 s each.
3. Move the gantries to the candidate morphology — one axis at a time, each waited on.
4. Command the `open` pose, position the cylinder, then command the CEM `grip` pose.
5. Run the buffered reorientation at a selected speed ratio.
6. Score the run and download its raw JSONL.

### Where the UI runs

Opening the CB1 URL does **not** run the UI workload on the CB1. The CB1 sends three small static
files once; JavaScript execution, SVG rendering, and operator interaction happen in the
workstation's browser. The CB1 only handles the hardware API, cached status responses, and serial
I/O. Policy inference likewise stays on the workstation and sends joint targets to the CB1 API.

If the static files should also be served by the workstation, run this there from the repository
root:

```bash
python3 -m http.server 8766 --bind 127.0.0.1 \
  --directory src/morphohand/driver/manta/host/manta_hand/static
```

Then open:

```text
http://127.0.0.1:8766/?api=http://10.99.99.2:8765&token=<MANTA_TOKEN>
```

The UI supports this split through its `api` query parameter, and the CB1 API permits
cross-origin requests. This changes where the few kilobytes of static content are served, but
not the control architecture or CB1 load in any material way.

A daemon restart deliberately forgets `homed`. That is correct: the M8P scale and zero state is
RAM-only, and a serial/board reset invalidates the previous reference.

## Servo telemetry: measured, 2026-08-29

All numbers below are from the real nine-servo chain on the CB1, not arithmetic.

### SYNC READ does not work on this hardware

The SCS protocol manual documents READ and SYNC WRITE but not SYNC READ; `rustypot`
exposes `sync_read_present_*` for the SCS0009 anyway. **Every sync read of every field
times out after 500 ms, every time** — position, load, speed, voltage, temperature,
status, torque_enable — while all nine servos answer a plain per-servo READ immediately.
`ServoBus.sync_read_joint_positions()` was built on the sync path, so the only servo
telemetry the service had did not work at all; `--telemetry-hz 0` was hiding it. It now
reads per servo. SYNC **WRITE** is fine and is still what `sync_set_joints` uses.

### The FTDI latency timer is 16× the whole servo read rate

The U2D2 here is an FT232H. `ftdi_sio` defaults its latency timer to 16 ms, and every
SCS response is a few bytes, so it never fills a buffer and every read waits out the
full timer. At 1 Mbaud the actual wire time for one exchange is ~0.15 ms — the timer was
99% of the cost.

| latency timer | 9 joint positions | rate | + `present_load` |
|---|---|---|---|
| 16 ms (default) | 144.0 ms | **6.95 Hz** | 3.47 Hz |
| 1 ms | 9.0 ms | **111.0 Hz** | 55.6 Hz |

Zero errors in either configuration (445 and 556 bundles respectively). `ServoBus` now
lowers the timer on open and reports what happened in `health()["ftdi_latency"]`; the
write needs root, so make it permanent with a udev rule:

```
# /etc/udev/rules.d/99-ftdi-latency.rules
SUBSYSTEM=="usb-serial", DRIVER=="ftdi_sio", ATTR{latency_timer}="1"
```

Without it, a replug silently costs you 16× and nothing errors.

**This is the number that decides whether closed-loop control is possible at all.** The
working assumption was ~10 Hz, which is roughly what the default configuration gives.
The bus is not the constraint it appeared to be; 111 Hz position feedback, or 55 Hz with
the load proxy, is available for one sysfs write. What is still missing for the learned
policies is the rest of the observation vector, not the rate — see the table above.

### The 300 ms inter-command gap is a WRITE requirement

`servos.INTER_CMD_DELAY_S` was applied after every call. Measured:

| pattern | rate | errors |
|---|---|---|
| reads only, no gap | 111 Hz | **0 / 6003 transactions** |
| sync write then a read sweep, no gap | 12 Hz | 14 / 91 (15%) |
| alternating single write / read, no gap | 46 Hz | 2 / 278 |

A transaction issued too soon after a *write* fails; a read issued straight after
another read never did. `ServoBus` now keeps the full gap around every write — before it
as well as after — and lets consecutive reads run at bus speed. Every ordering is at
least as protected as it was; only read-after-read got faster.

### Force, current, and servo resets

- **There is no present-current register on the SCS0009.** Not a read-cost question —
  the data does not exist on the part.
- `present_load` is the only force proxy and it is an uncalibrated duty-cycle-like
  number. Nothing converts it to newtons. It is logged as a raw covariate next to the
  manual score; treat any force claim built on it as unfounded until someone calibrates
  it against a load cell. It costs a second nine-transaction sweep (111 Hz → 55.6 Hz).
- `read_status` works and returns each servo's latched alarm byte (overload,
  over-temperature, over-voltage). This is the closest thing this bus has to the
  packet-storm watchdog a Dynamixel chain gives you. All nine read 0 on this hand.
- **`torque_enable` reads 0 at power-up and 2 after an explicit disable.** Both mean
  "not holding", but a servo reading 0 *after* the service set it to 1 or 2 has reset
  itself — see `servos.TORQUE_UNSET` and `ServoBus.suspected_resets`.
- `ServoBus.health()` counts timeouts and consecutive timeouts. `Servo._call` retries
  on timeout, which is what makes a degrading chain look healthy right up until it
  stops; the counters are what make it visible. Shown in the UI's telemetry card.

Re-run `examples/benchmark_servo_telemetry.py` before trusting any of this on different
servos, a different adapter, or a longer chain.

## Local policy command stream

The API has a lease-based, write-only stream for experiments that produce nine sim-frame joint
targets on the workstation. It is deliberately called experimental because it does not solve
the learned policies' missing observations.

```text
POST /api/v1/stream/start   {"timeout_s": 0.25}       -> token
POST /api/v1/stream/frame   {"token": ..., "joints": {thumb/index/middle...}}
POST /api/v1/stream/end     {"token": ...}
```

Every frame is finite/range checked and transformed through the same yaw/MCP/PIP mapping as a
buffered plan. If frames stop arriving, the lease expires and the servos hold their last valid
position goal. A future real closed-loop client should use a persistent binary transport rather
than HTTP requests, but only after an observation source exists and the measured control-rate
budget justifies it.

## Logs and simulation replay

Each reorientation creates:

- `<run_id>.jsonl`: timestamped commanded sim joint positions, measured telemetry samples when
  polling is enabled, and runtime events.
- `<run_id>_SUMMARY.json`: plan metadata, rates, sample counts, completion/error state, and the
  operator's success / approximate-angle / notes score.

The commanded values remain in sim joint names and degrees so they can be mapped directly back
to `thumb_yaw...middle_pip`. The run summary also freezes the per-finger joint signs used for
that run. Measured servo values are stored separately. Servo polling is suspended while command
frames are being written; one deliberate nine-servo snapshot is taken before and after the
trajectory so telemetry cannot steal half-duplex bus time from control.

Replay the exact host command timing into the plan's exported scene with:

```bash
uv run --extra rl python scripts/replay_real_v1_hardware_log.py \
  --log logs/hardware/<run>.jsonl \
  --plan docs/experiments/20260829-real_v1_deploy/deploy/g12_plan.json \
  --out logs/hardware/<run>_sim.npz
```

The NPZ contains simulation time, object pose/axis alignment, full `qpos`/`qvel`/`ctrl`, and
finger controls. Its sidecar summary compares the before/after servo snapshots to their expected
commands when available. This supports timing and endpoint-tracking comparisons, but it is not a
reconstruction of real object motion; that requires an external camera or other pose sensor. The
replayer requires the state-complete metadata produced by the current exporter and refuses older
plans whose fitted palm/object initial state is missing.

## Safety and recovery notes

- **Stop motion** cancels the running operation and sends `STOPALL`. Two things it does
  not do: SCS position servos hold their last goal (it is not a torque-off button), and
  `STOPALL` decelerates at the axis's configured accel — from 12000 sps at 2000 sps² that
  is about six seconds of continued travel. Cancelling a home discards the whole home
  reference, by design: the interrupted axis is stopped, disabled and **not** zeroed.
- **Disable motors** is the real kill switch: `STOPALL`, then `DIS` on all six steppers
  and torque off on all nine servos. Use it when an axis is grinding. It invalidates the
  home, because a de-energised stepper's step counter no longer describes where the axis
  physically is.
- Put an actual power cut within reach. Software stop is not an emergency-stop substitute.
- `MOVEMM` has no stall protection — only `HOME` does. That is why a bogus zero is
  dangerous and why plan validation uses measured travel and never widens it.
- Yaw is conservatively bounded to ±70° on all three fingers. The sign is a per-finger
  hardware measurement (thumb +1, index −1, middle −1, measured 2026-08-29).
- The servo bus and the M8P link are **exclusive**. `examples/hand_control.py` and this
  service cannot run at the same time; the second one to start gets
  `OSError: Device or resource busy` on `/dev/ttyUSB0`. Check with `pgrep -af hand_control`.
- Servo packet errors are counted, not hidden, and telemetry backs off geometrically on
  consecutive failures instead of hammering a sick bus. There is no automatic servo
  reboot: the observed SCS0009 register/physical-state mismatch makes blind reset unsafe.
- Run the service with `--log-file`. It is normally started in an interactive SSH shell,
  so without one the only record of a failure is that terminal's scrollback — gone at
  exactly the moment you need it. `SIGHUP` is handled, so the service stops the hand
  cleanly when its shell dies rather than leaving it holding a goal.

### If it stops responding

1. `curl -s http://10.99.99.2:8765/api/v1/state` — if this answers, the service is alive
   and the problem is the browser or the network, not the hand.
2. Check the UI's banner. A **red serial-link banner** means the board stopped answering:
   do the BOOT0/RESET cycle, then `POST /api/v1/reconnect`, then re-home.
3. `tail /tmp/mantalogs/web.log` (or whatever `--log-file` you passed) for the traceback.
4. `pgrep -af manta_hand` — a second instance cannot bind the port and cannot open the
   serial ports; the running one owns the hand.
5. Query the board directly to find out what it actually thinks, without the service:

   ```bash
   cd src/morphohand/driver/manta/host
   .venv/bin/python -c "
   from manta_hand.driver import MantaHandDriver
   from manta_hand.kinematics import STEPS_PER_MM
   with MantaHandDriver('/dev/ttyACM0') as d:
       for i, s in enumerate(d.get_all_status()[:6]):
           print(f'J{i} {s.position/STEPS_PER_MM[i]:+8.2f}mm moving={s.moving} '
                 f'enabled={s.enabled} homing_result={s.homing_result}')"
   ```

   `homing_result` is 0 idle / 1 homing / 2 stalled / 3 timed out. This is how the
   2026-08-29 incident was diagnosed: the board had never crashed.

## API summary

Read-only endpoints are `GET /api/v1/state`, `/events`, `/plans`, and `/logs`. Command endpoints
are `POST /plans/load`, `/home`, `/morphology`, `/pose`, `/reorient`, `/stop`, `/stream/*`, and
`/logs/<run_id>/score`. The API schema is version 1 and all command posts require
`X-Manta-Token` on real hardware.
