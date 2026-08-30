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

## What is deployable now

| Control method | Status | Feedback requirement |
|---|---|---|
| Morphology move | Ready after one home | Stepper status only |
| CEM grasp keyframe | Ready after yaw-sign check | None during motion |
| CEM open-loop reorientation | Ready after yaw-sign check | None; buffered/timed on CB1 |
| Workstation joint-command stream | Experimental, write-only | None; holds last goal if lease expires |
| Learned A/B RL policies | Not deployable closed-loop | Missing most of their observation vector |

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
  --telemetry-hz 0 \
  --token "$MANTA_TOKEN"
```

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
2. Press **Home once**, type the exact confirmation, and stay with the hand. A second home is
   rejected unless an API client explicitly sends `force: true`.
3. Move the gantries to the candidate morphology.
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

## Servo telemetry: measure before enabling

The current service defaults to `--telemetry-hz 0`. Run the read-only benchmark on the CB1:

```bash
cd src/morphohand/driver/manta/host
python3 examples/benchmark_servo_telemetry.py \
  --port /dev/ttyUSB0 --seconds 5 \
  --json 20260829-servo-telemetry.json
```

It measures a complete nine-servo synchronized position read, each other API exposed by the
installed `rustypot`, and the combined bundle. Use the sustained bundle rate—not 1 Mbaud packet
arithmetic—as the polling ceiling. Start the service at no more than half the error-free bundle
rate and check trajectory timing again.

The SCS0009 feedback contract advertises load, speed, and input voltage. It does not provide a
calibrated force measurement, and the present-load value is not the same as measured motor
current. The service therefore exposes position only today and labels load/current unavailable.
Adding another field must include a bus benchmark and an empirical calibration before it is
called force.

Telemetry and trajectory writes share a bus-wide lock. A synchronized read is allowed between
writes; nine slow verified reads are never used as a fallback. If synchronized read is missing
or unhealthy, the UI reports stale/unavailable telemetry instead of degrading command timing.

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

- The red stop cancels future servo frames and sends `STOPALL` to the steppers. SCS position
  servos hold their last goal; it is not a torque-off button.
- `MOVEMM` has no stall protection. Plan validation uses measured travel and never widens it.
- Yaw is conservatively bounded to ±70 degrees on all three fingers. The sign remains a separate
  per-finger hardware measurement and is mandatory at real-server startup.
- Servo packet errors are reported and polling backs off. There is no automatic servo reboot:
  the observed SCS0009 register/physical-state mismatch makes blind reset/re-enable unsafe.
- Put an actual power cut within reach. Software stop is not an emergency-stop substitute.

## API summary

Read-only endpoints are `GET /api/v1/state`, `/events`, `/plans`, and `/logs`. Command endpoints
are `POST /plans/load`, `/home`, `/morphology`, `/pose`, `/reorient`, `/stop`, `/stream/*`, and
`/logs/<run_id>/score`. The API schema is version 1 and all command posts require
`X-Manta-Token` on real hardware.
