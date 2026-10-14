# MorphoHand hardware control station

Operator runbook for the `real_v1` hand. The browser and any policy inference run on the
workstation; one service on the CB1 owns both serial links. Open-loop trajectories are timed on
the CB1, so Ethernet jitter does not become servo jitter.

```text
workstation (10.99.99.50)               CB1 (10.99.99.2)              hand
browser UI ───── HTTP/JSON ─────┐
local inference client ─────────┴──> HandRuntime ── USB-CDC ──> 6 gantry steppers
AprilTag service <── start/stop/run id ─┤
       └──────── object samples ───────>│
                                      │
                                      └──────── U2D2 TTL ────> 9 SCS0009 servos
```

`HandRuntime` is the sole hardware owner. UI refreshes read a cached status document; they do
not generate servo reads. Only one homing, gantry, pose, trajectory, or policy-stream operation
can own the hand at a time.

> **Why this service works the way it does** — the 2026-08-29 bench failure, the four faults it
> exposed, and the bus measurements behind every number here — is written up in
> `webpaper/src/control.typ` (build with `webpaper/build.sh`, then open `control.html`). This
> file is the runbook; that page is the reasoning.

## What is deployable now

| Control method | Status | Feedback requirement |
|---|---|---|
| Morphology move | Ready after one home | Stepper status only |
| CEM grasp keyframe | Ready after yaw-sign check | None during motion |
| CEM open-loop reorientation | Ready after yaw-sign check | None; buffered/timed on CB1 |
| Workstation joint-command stream | Experimental, write-only | None; holds last goal if lease expires |
| Learned A/B RL policies | Not deployable closed-loop | Missing most of their observation vector |

Feedback available on this hand: **nine joint positions at up to 111 Hz** (55 Hz with the load
proxy alongside), and nothing else. No object pose, no contact, no current. A full write + read
closed loop measures 75 Hz through the driver API, so ~50 Hz closed-loop control is feasible —
the observation vector is the blocker, not the rate. The learned actors consume 65 (A) / 66 (B)
values including object pose, velocity and reference poses; feeding zeros or simulation
references for the missing ones is an out-of-distribution test, not deployment.

## Trying it without hardware

```bash
PYTHONPATH=src/morphohand/driver/manta/host python -m manta_hand.web --fake --port 8765
```

`--fake` runs the **real** driver stack against a simulated M8P on a pty and a simulated servo
bus (`manta_hand.fake_hardware`), so homing, gantry motion and the serial link all execute for
real. Use it for any UI or driver work. `--mock` also exists but replaces the whole backend and
exercises none of the driver code.

`--fake-stall-axes` defaults to `0,1,2,4`, matching the real hand.

## CB1 install

Debian's system Python is externally managed; do not use `--break-system-packages`.

```bash
cd src/morphohand/driver/manta/host
sudo apt install python3-venv
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[servo]'
```

**Install the FTDI latency udev rule.** Without it every servo read costs 16 ms instead of 1 ms
— a 16× penalty on all feedback, silent, with nothing logged or errored:

```
# /etc/udev/rules.d/99-ftdi-latency.rules
SUBSYSTEM=="usb-serial", DRIVER=="ftdi_sio", ATTR{latency_timer}="1"
```

The service lowers the timer itself on open when it has permission, and reports the result in
`telemetry.servo_bus.ftdi_latency`. The rule is what makes it survive a replug.

## Launch

Both hosts need starting, and the workstation needs its bench address back after every reboot.

### 0. Workstation address (once per boot)

`10.99.99.50/24` is not in any netplan or NetworkManager profile, so a reboot drops it and the
CB1 becomes unreachable at all — which looks like a dead box and is not:

```bash
sudo ip addr add 10.99.99.50/24 dev enp130s0     # or, once and for all:
sudo nmcli connection modify "Wired connection 1" +ipv4.addresses 10.99.99.50/24
```

### 1. Tracker companion, on the workstation (once per boot, not per trial)

```bash
scripts/run_tracker_service.sh start          # also: stop | restart | status | log
```

That is the whole command. The launcher sources `.dotenv`, resolves the repo from its own
location, and launches detached under `~/miniconda3/bin/python` — the camera stack is not in the
uv env. It prints `station push token accepted` before it serves; if it prints anything else,
stop and read it, because the alternative is finding out mid-trial.

**Why there is a launcher.** The service reads `--station-token` from `MANTA_TOKEN`, and
unset it does not complain — it starts, reports itself healthy, and every sample it pushes is
answered **401** by the station. The station then sees no samples, the fail-closed
`automatic-required` interlock refuses the turn, and the run dies 20 s in with the cylinder
already grasped and `"samples": {"object": 0}` in its summary. On 2026-09-09 that consumed a
session. The trap in the hand-typed form was the first line: the workstation repo is
`/home/humanoid/Programs/hand`, so a `cd ~/hand` copied from the CB1 side fails, `.dotenv` is
never sourced, and `MANTA_TOKEN` is silently empty.

Since 2026-09-09 the service itself also refuses to start on a token the station rejects, and
re-checks it each time the CB1 arms a run, so this cannot reach a trial again. To launch by
hand anyway, export the token first — assignment is not export:

```bash
set -a && . /home/humanoid/Programs/hand/.dotenv && set +a
~/miniconda3/bin/python scripts/real_v1_tracker_service.py --host 10.99.99.50 --port 8770 \
  --tracker-arg=--width --tracker-arg=1280 --tracker-arg=--height --tracker-arg=800 \
  --tracker-arg=--fps --tracker-arg=30
```

Do **not** re-type the rig's flags. Since 2026-09-02 `BENCH_DEFAULTS` in the service carries
`--shaft-axis=-x --tag-end top --symmetric-object --heading-deg 90 --exposure 4000 --gain 64
--video-hz 4 --video-scale 1.0 --video-quality 92`, and `--tracker-arg` is appended *after*
them, so overrides win without dropping the rest. A session was lost on 2026-09-01 to launching
this bare.

`--width 1280 --height 800` is the only thing left to pass: the tag tool still defaults to
1280x720. 800 is the D435's native IR sensor size and 720 a vertical crop of it — same `fx =
642.7`, same 80.1 px id6 edge at 327 mm, 80 more rows, which is about 20 mm of extra view below
the tag on the axis where trials are lost.

**`--symmetric-object` is correct for the 100x25 cylinder and WRONG for the screwdriver.** It
folds cos to [0,1], so a tool standing on its handle scores the same as one standing on its tip.
It is `store_true` with no negating flag, and `BENCH_DEFAULTS` is prepended, so it cannot be
turned off from the command line — for screwdriver work edit `BENCH_DEFAULTS` in
`scripts/real_v1_tracker_service.py` (or add a `--no-symmetric-object` to
`scripts/real_v1_tag_tracker.py`, which does not exist yet).

Optional pre-flight, one frame and exit — with `--tag-end` it *checks* the sign of
`--shaft-axis` rather than narrating it:

```bash
~/miniconda3/bin/python scripts/real_v1_tag_tracker.py --probe \
  --exposure 4000 --gain 64 --shaft-axis=-x --tag-end top --symmetric-object --heading-deg 90
```

### 2. Control station, on the CB1

After the documented M8P BOOT0/RESET cycle makes `/dev/ttyACM0` appear, use the script — it
carries the token, the plans directory and the tracker URL:

```bash
ssh irlab@10.99.99.2 '~/run_control_station.sh start'    # also: stop | status | log
```

It runs this, and typing it by hand is how the 2026-09-02 flags get lost:

```bash
cd ~/hand/src/morphohand/driver/manta/host
.venv/bin/python -m manta_hand.web \
  --host 0.0.0.0 --port 8765 \
  --stepper-port /dev/ttyACM0 --servo-port /dev/ttyUSB0 \
  --plans-dir ~/hand/docs/experiments/20260902-residual-bench/deploy \
  --logs-dir ~/hand/logs/hardware \
  --telemetry-hz 20 --servo-fields load,temperature \
  --tracker-url http://10.99.99.50:8770 \
  --log-file /tmp/mantalogs/web.log \
  --token "$MANTA_TOKEN"
```

Then open, from the workstation:

```text
http://10.99.99.2:8765/?token=<MANTA_TOKEN>
```

To stop it: `pkill -f "manta_hand[.]web"`. To check whether one is already running:
`pgrep -af manta_hand` — both serial ports are exclusive, so a second instance cannot start and
`examples/hand_control.py` cannot run at the same time.

### What the three 2026-09-02 flags buy

- `--plans-dir …/20260902-residual-bench/deploy`. The default and the old hand-typed value both
  point at `20260829-real_v1_deploy/deploy`, whose CB1 copy holds `g12`, `g23`, `g24` and
  `rv04_mid` — and the clearance gate **excluded** `g23` (+0.8 mm), `rv04_mid` (−2.6 mm) and
  `g24` (−5.2 mm) for interpenetrating their own fingers. Serving those is one misclick from
  driving index into middle. A UI listing exactly those four means the station came up on its
  default and the other two flags are missing too.
- `--tracker-url http://10.99.99.50:8770`. Without it `/reorient` is not fail-closed for
  tracking and will silently produce untracked trials.
- `--servo-fields load,temperature`. `load` alone is the default; `temperature` is what makes
  the SCS0009's 70 C cutoff and the `protective_torque` work measurable.

With `--tracker-url` configured, `/reorient` creates the run id, asks the workstation to start
that exact trace, waits until static id6 is latched and the first id0 sample has reached the
CB1, then moves. It stops/finalizes the tracker before closing the run summary. A missing
camera, invisible id6, or dead companion refuses motion rather than producing an untracked
trial. The service's 300 s cap is only crash protection; normal traces stop with the trajectory.

Notes on the other flags:

- `--token` is mandatory with real hardware (`openssl rand -hex 16`). The token is kept in that
  browser's local storage. Keep the service on the isolated lab network.
- `--log-file` is effectively mandatory. This is normally started in an SSH shell, so without it
  the only record of a failure is that terminal's scrollback. `SIGHUP` is handled, so the hand
  is stopped and de-energised when the shell dies.
- `--telemetry-hz 20` is what the launcher ships. 5 was the earlier value and is also
  comfortable (measured: 11,115 transactions, zero timeouts, during a live home); the ceiling is
  much higher, and servo polling is suspended automatically while a writer owns the half-duplex
  bus.
- `--aa-signs thumb:+1,index:-1,middle:-1` overrides the recorded yaw signs after a fresh
  hardware measurement. Not needed for the current configuration.

### Which plans are on the CB1

There is no filesystem share and no push helper: the CB1 holds only what was copied. As of
2026-09-08 the served directory holds all eight hands of the transfer study, each re-checked
against `scripts/real_v1_trajectory_clearance.py` (finger<->finger, PASS >= 5.0 mm):

| plan | id | chord | csv |
|---|---|---|---|
| `sv1_w6689_b060` | D1 | +8.5 | +8.5 |
| `sv1_w2360_b075` | D2 | +10.4 | +10.4 |
| `sv1_u1364_b080` | D3 | +5.0 | +5.2 |
| `g12_b095` | D4 | +8.7 | +8.8 |
| `sv1_u0060_b75` | D5 | +9.9 | +9.9 |
| `sv1_u0308_b050` | D6 | +9.2 | +9.2 |
| `rv05_manual_b85` | D7 | +10.3 | +10.3 |
| `sv1_w0099_b100` | D8 | +8.2 | +8.2 |

Run **D3 from its CSV**, not the 3-point chord: its chord is +5.0 mm, exactly at threshold. A
plan's chord and its `_traj.csv` are different paths and pass the gate separately.

What is deliberately NOT served is the rest of `20260829-real_v1_deploy/deploy`, which the CB1
still holds a pruned copy of. `g23` (+0.8 mm), `rv04_mid` (-2.6 mm) and `g24` (-5.2 mm) are
excluded by the clearance gate for interpenetrating their own fingers. A UI listing exactly
`g12`, `g23`, `g24` and `rv04_mid` means the station came up on its default `--plans-dir` and
the tracker and servo-field flags are missing too.

Re-run the gate on whatever you are about to serve before every session.

### Serving the UI from the workstation instead

Opening the CB1 URL does not run the UI workload on the CB1 — it sends three static files and
the browser does the rest. To serve those files locally anyway:

```bash
python3 -m http.server 8766 --bind 127.0.0.1 \
  --directory src/morphohand/driver/manta/host/manta_hand/static
# then: http://127.0.0.1:8766/?api=http://10.99.99.2:8765&token=<MANTA_TOKEN>
```

## Operating sequence

1. **Load a candidate.** Validates every mount and every pose before anything moves.
2. **Home gantries**, type the exact confirmation, and stay with the hand. Takes about two
   minutes; see below.
3. **Move gantries to morphology** — one axis at a time, each waited on (~95 s for a full
   change).
4. Command the `open` pose, position the object, then command the CEM `grip` pose.
5. Run the buffered reorientation at a chosen speed ratio. The button automatically arms and
   finalizes its AprilTag trace when `--tracker-url` is configured.
6. Score the run and download its JSONL.

### Homing takes about two minutes

Each axis gets a timeout sized to guarantee it covered its full measured travel, so a home
cannot be shortened without making it untrustworthy. Worst case for all six is 179 s.

`FULL_EXTENSION_MM` corrected 2026-09-01 -- all six axes reach their full nominal travel (110mm
on J0, 60mm on the rest); the previous J1/J3/J5 entries (56.2/56.0/54.1) implied a shortfall that
was never real, and travel/window below reflect the fix.

| axis | finger | travel | window | typical outcome |
|---|---|---|---|---|
| J0 | thumb x | 110.0 mm | 45.5 s | StallGuard2, ~16 s |
| J1 | thumb y | 60.0 mm | 26.7 s | StallGuard2, ~15 s |
| J2 | index x | 60.0 mm | 26.7 s | StallGuard2, ~16 s |
| J3 | index y | 60.0 mm | 26.7 s | **timeout, ~33 s** |
| J4 | middle x | 60.0 mm | 26.7 s | StallGuard2, ~15 s |
| J5 | middle y | 60.0 mm | 26.7 s | **timeout, ~33 s** |

**J3 and J5 do not trip StallGuard2** under the current SGT tuning — they press against their
hardstop for their whole window, every time. The home is still trustworthy; it just sounds
alarming. The UI shows the per-axis outcome live so the two cases are distinguishable.

### Adopting an existing home

The M8P keeps its step counters, `SETSCALE` calibration and per-axis `homing_result` across a
**host** restart — only a board reset or a `DIS` clears them. After restarting the service
against an already-homed board, use **Adopt the board's existing home** (or
`POST /api/v1/home/adopt`) instead of re-homing. It checks the board's own `homing_result` on
all six axes, and also adopts the morphology position when the gantries are already within
0.5 mm of the loaded plan's targets. Nothing moves.

## Safety

- **Stop motion** cancels the running operation and sends `STOPALL`. It is *not* a torque-off
  button — SCS position servos hold their last goal — and `STOPALL` decelerates at the axis's
  configured accel, about six seconds from full speed. Cancelling a home discards the whole home
  reference by design: the interrupted axis is stopped, disabled, and **not** zeroed.
- **Disable motors** is the real kill switch: `STOPALL`, then `DIS` on all six steppers and
  torque off on all nine servos. Use it when an axis is grinding. It invalidates the home.
- Put an actual power cut within reach. Software stop is not an emergency-stop substitute.
- `MOVEMM` has no stall protection; only `HOME` does. Plan validation uses measured travel and
  never widens it.
- Yaw is bounded to ±70° on all three fingers. Signs measured 2026-08-29: thumb +1, index −1,
  middle −1.
- Torque must be ON to command a pose. A torque-off SCS0009 accepts a position write and reads
  it back correctly without moving, so the service refuses rather than pretending.

### If it stops responding

1. `curl -s http://10.99.99.2:8765/api/v1/state` — if this answers, the service is alive and the
   problem is the browser or the network.
2. Check the UI banners. A red **serial link** banner means the board stopped answering: do the
   BOOT0/RESET cycle, `POST /api/v1/reconnect`, then re-home or adopt.
3. `tail /tmp/mantalogs/web.log` for the traceback.
4. `pgrep -af manta_hand` — a second instance owns nothing; the first one owns the hand.
5. Ask the board directly, without the service:

   ```bash
   cd src/morphohand/driver/manta/host
   .venv/bin/python -c "
   from manta_hand.driver import MantaHandDriver
   from manta_hand.kinematics import STEPS_PER_MM
   with MantaHandDriver('/dev/ttyACM0') as d:
       for i, s in enumerate(d.get_all_status()[:6]):
           print(f'J{i} {s.position/STEPS_PER_MM:+8.2f}mm moving={s.moving} '
                 f'enabled={s.enabled} homing_result={s.homing_result}')"
   ```

   `homing_result`: 0 idle / 1 homing / 2 stalled / 3 timed out. This is the fastest way to tell
   a host bug from a hardware one.

## Servo feedback: what exists

| field | works | cost (9 servos, 1 ms latency timer) |
|---|---|---|
| `present_position` | yes | 9 ms → 111 Hz |
| `present_load` | yes, **uncalibrated** | +9 ms → 55 Hz combined |
| `present_voltage`, `present_temperature` | yes | +9 ms each |
| `status` (latched alarm byte) | yes | +9 ms |
| `present_current` | **does not exist on the SCS0009** | — |
| any `sync_read_*` | **times out, always** | — |

Three things to keep in mind:

- **SYNC READ does not work on this hardware.** The SCS manual documents READ and SYNC WRITE but
  not SYNC READ; every sync read of every field times out after 500 ms. All reads are per-servo.
  SYNC **WRITE** is fine and is what `sync_set_joints` uses.
- **`present_load` is not force.** It is an uncalibrated duty-cycle-like number. Log it as a raw
  covariate; do not build a force claim on it without a load-cell calibration.
- **A servo reporting `torque_enable == 0` after the service set it has rebooted.** 0 is the
  power-on default; an explicit disable writes 2.

Re-run `examples/benchmark_servo_telemetry.py` before trusting any of this on different servos,
a different adapter, or a longer chain.

## Local policy command stream

A lease-based, write-only stream for experiments that produce nine sim-frame joint targets on
the workstation:

```text
POST /api/v1/stream/start   {"timeout_s": 0.25}       -> token
POST /api/v1/stream/frame   {"token": ..., "joints": {thumb/index/middle...}}
POST /api/v1/stream/end     {"token": ...}
```

Every frame is finite/range checked and mapped through the same yaw/MCP/PIP transform as a
buffered plan. If frames stop arriving the lease expires and the servos hold their last valid
goal. A real closed-loop client should use a persistent binary transport rather than HTTP — but
only once an observation source exists.

## Logs and replay

Each reorientation writes `<run_id>.jsonl` (timestamped commanded sim joint positions, telemetry
samples, runtime events, and AprilTag object samples) and `<run_id>_SUMMARY.json` (plan metadata,
rates, sample counts, completion state, instrument summary, the operator's separate manual score,
and the per-finger joint signs used). Commanded
values stay in sim joint names and degrees so they map straight back to `thumb_yaw...middle_pip`.

Servo polling is suspended while command frames are being written; one nine-servo snapshot is
taken before and after the trajectory. **Telemetry read immediately after an operation may be
stale** — check `telemetry.servo_age_s` before believing it.

Replay a run's exact host command timing into the plan's exported scene:

```bash
uv run --extra rl python scripts/replay_real_v1_hardware_log.py \
  --log logs/hardware/<run>.jsonl \
  --plan docs/experiments/20260829-real_v1_deploy/deploy/g12_plan.json \
  --out logs/hardware/<run>_sim.npz
```

This replays commands, not measured object motion — reconstructing the latter needs an external
camera or another pose sensor.

## API summary

Read-only: `GET /api/v1/state`, `/events`, `/plans`, `/logs`, `/logs/<run_id>.jsonl`.

Commands (all require `X-Manta-Token` on real hardware): `POST /plans/load`, `/home`,
`/home/adopt`, `/morphology`, `/pose`, `/reorient`, `/stop`, `/motors/disable`,
`/servos/torque`, `/reconnect`, `/stream/*`, `/logs/<run_id>/score`.

Schema version 1.
