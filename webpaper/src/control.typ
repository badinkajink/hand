#import "template.typ": conf, callout, det, fig, refbox
#show: conf.with(title: "Control Station — MorphoHand", current: "control")

= The Control Station

Between a simulated hand and a real one sits a layer nobody writes a paper about: the
service that owns the serial ports, the state machine that decides a command is safe,
and the browser page an operator watches while the rails move. This page is the record
of building that layer for `real_v1`, of the day it failed at the bench, and of what
measuring the hardware honestly turned out to be worth.

The short version: the failure was four host-software bugs, not a hardware fault, and
chasing them produced a servo bus sixteen times faster than anyone had assumed it could
be.

#fig(
  "assets/20260829-control-station-ui.png",
  label: "Fig 1.",
  caption: [The operator page driving the real hand mid-sequence, after a full home.
  The per-axis home card exists because "is it stuck, or is this normal?" was
  unanswerable from the old UI.],
)

== Architecture

Everything with a serial cable attached lives on the CB1; everything that thinks lives
on the workstation.

#table(
  columns: (auto, 1fr),
  table.header([*Where*], [*What*]),
  [Workstation], [browser UI, policy inference, plan export, log analysis],
  [CB1], [`HandRuntime`: one owner of both links, one operation at a time, cached status],
  [Manta M8P], [6 gantry stepper axes over USB-CDC. Firmware-timed step generation],
  [U2D2 / SCS bus], [9 Feetech SCS0009 servos over a single half-duplex TTL chain],
)

Open-loop trajectories are timed on the CB1, not streamed from the workstation, so
Ethernet jitter never becomes servo jitter. The UI polls a cached status document; a
browser refresh never turns into a servo packet.

== The bench failure, 2026-08-29

The first real session went like this. Home the hand. Become suspicious of how long an
axis is grinding. Press *Stop motion*. Watch the home continue anyway. Press *Move
gantries to morphology*. Lose the service.

The obvious hypotheses were all electrical or firmware: six motors starting at once
browning out the 19 V rail that also backfeeds the CB1; step interrupts at NVIC priority
2 starving the USB interrupt at priority 3; the STM32 hanging. Each is plausible, each
is memorable, and each is expensive to test.

#callout(tag: "The board was fine the whole time", kind: "note")[
  One read-only `STATALL` over SSH answered it. `/dev/ttyACM0` was alive, all six axes
  enabled, four of six reporting a StallGuard2 stall, and the kernel log showed no USB
  disconnect since boot. The M8P had never crashed. Every fault was in the host
  software.
]

That one query eliminated the entire hardware half of the hypothesis space in a few
seconds, and it is the lesson worth keeping: a device's own status registers are free to
read, and reading them first is cheaper than reasoning about anything else.

=== Fault 1 — the home that zeroed nothing

The operator's first report was that homing moved the gantries but not the servos.

`examples/hand_control.py`, the collaborator's working reference, enables all nine
servos before it calls `home_all()`. The service never enabled a single one. And a
torque-OFF SCS0009 *accepts a goal-position write and reads it back correctly without
moving* — so `zero_joints()` performed nine verified writes, confirmed every one, and
moved nothing. No exception, no warning, nothing in the log.

#det("Why the write-verify pattern could not catch it", kind: "detail")[
  `Servo.move_to_deg` uses `_call_verified`, which writes and then reads back to confirm
  — a defence added because this exact part is known to report local success while
  ignoring a write. But it reads back `goal_position`, the register it just wrote, not
  `present_position`. The goal register accepts writes regardless of torque state. The
  verification was checking that the servo heard the command, which it had; nothing was
  checking that the horn moved.

  The fix is at connect, not in the verify: `ServoBus.set_torque_all` enables all nine in
  one sync-write transaction plus a verifying read sweep, and `Hand.home_all` now takes
  `require_torque=True` and refuses to run at all if any servo reports torque off.
  Commanding a pose while torque is off is refused for the same reason.
]

=== Fault 2 — stop did not stop, and then zeroed the wrong place

Pressing stop sent `STOPALL`, which ended the motion of whichever axis was moving. The
host's homing loop had no cancellation check at all, so it proceeded to the next axis
and the next, and the operator watched a "stopped" home run to completion.

The second half is worse, and it lives in the firmware. `Stepper_Stop()` sets
`target = position` — it does *not* clear `homing_result`. The axis stays logically
"homing" at state 1, and the 1 kHz supervisor tick eventually flips it to 3, meaning
"timed out". The host read 3 and applied the timeout guarantee:

#callout(tag: "The guarantee that stopped being true", kind: "warn")[
  A homing timeout is normally trustworthy. `_home_timeout_ms` is computed so the window
  cannot expire before the axis has physically covered its entire measured travel — so
  reaching it means the axis must be at the hardstop, whether or not StallGuard noticed,
  and calling `ZERO` is correct.

  That reasoning is void once the motion was cancelled part-way. The old code applied it
  anyway, planting step-zero somewhere in the middle of the rail. Every later `MOVEMM`
  then measured from a fiction — and `MOVEMM`, unlike `HOME`, performs no stall
  detection whatsoever.
]

Cancellation is now cooperative all the way down to `kinematics._home_one_axis`. An
aborted axis is stopped, then *disabled* — `Stepper_Disable()` is the only firmware call
that clears `homing_result` — and never zeroed. The session's home is discarded with the
reason recorded, because a reference is only as good as its worst axis.

#det("The re-enable that made the session recoverable", kind: "detail")[
  Disabling to cancel a home has a consequence: `Stepper_Home` refuses with `ERR NODIAG`
  on an axis that is not enabled. The first version of this fix therefore turned one
  cancelled home into a dead session — the operator could never home again.

  `Gantry.prepare()` now re-enables every axis and re-sends its `SETSCALE` calibration
  before each home and each gantry move. Both are single cheap commands, and both are
  lost more often than they look: `EN` by any `DIS`, and `SETSCALE` by a board reset.
]

=== Fault 3 — six axes at once

`plan.apply_mounts` issued six `MOVEMM` commands back to back and waited on none of
them. It then polled `STATALL` at 10 Hz to find out when they finished.

Nothing validated on this hand had ever moved more than one axis at a time. The REPL
moves one per typed line; the homing sequence is explicitly sequential. Six simultaneous
starts means six simultaneous `TMC5160_StartMotionKick` current kicks — IRUN 1 to 7 for
500 ms each — on the rail that also backfeeds the CB1, and six axes of step interrupts
preempting the USB interrupt while the host polls that same link nine packets at a time.

The gantry move is now sequential, each axis waited on with a single-axis `STAT` rather
than a whole-board `STATALL`.

#callout(tag: "An honest correction", kind: "note")[
  Sequential was the right response to an unexplained failure, but it is probably not
  the right permanent answer. The M8P is a 3D-printer controller designed to run eight
  steppers simultaneously. What made simultaneous motion dangerous *that day* was fault
  2: several axes were commanded past their hardstops from a bogus origin at the same
  moment. With the origin trustworthy, parallel motion deserves a real test — it is
  worth roughly 95 seconds against 15 for a morphology change, because at 12000 steps/s
  over ~3216 steps/mm an axis only travels 3.7 mm/s. Homing must stay sequential
  regardless: that is a StallGuard requirement, not a power one.
]

=== Fault 4 — no difference between "refused" and "gone"

A serial timeout landed in `last_error` while the session still advertised itself as
homed with its mounts applied. The next command then acted on a reference that no longer
existed. Link failures now latch, invalidate the home, and require an explicit
reconnect; the UI shows a banner that does not disappear on its own.

The transport had a matching defect. A rejected token returned without reading the
request body, and on a keep-alive connection those unread bytes become the next request
line — every subsequent request on that connection fails, which a browser reports as
"Failed to fetch" with nothing at all wrong on the server.

== Reproducing hardware failures without hardware

Three of the four faults are now regression tests, which required something that did not
exist: a way to run the real driver stack with no hand attached.

`manta_hand.fake_hardware` puts a simulated M8P on a pty and a simulated SCS0009 bus
behind a real `RealHardwareBackend`, so `MantaHandDriver`, `Joint`, `Gantry`, `Hand` and
`ServoBus` all execute for real. It is a model of the firmware, not a convenience stub,
and it deliberately reproduces the behaviours that caused the incident.

#det("What the fake models on purpose", kind: "detail")[
  - `STOP` sets `target = position` and leaves `homing_result` untouched; `DIS` clears it.
  - A torque-off servo accepts a goal write, reads it back correctly, and does not move.
  - `HOME` ends in state 2 (stalled) or 3 (timed out) and never zeroes by itself.
  - `--fake-stall-axes` defaults to `0,1,2,4`, matching the real hand as measured.
  - `drop_lines` truncates a `STATALL` reply, modelling `cdc_send_blocking` giving up on
    a busy IN endpoint.
  - `answering = False` models the board falling off the USB bus.

  `python -m manta_hand.web --fake` serves the whole control station against it. This is
  a different thing from `--mock`, which replaces the backend and therefore exercises
  none of the driver code.
]

== What the servo bus can actually do

The working assumption, from the collaborator and consistent with cheap serial servos,
was about 10 Hz for the nine joints. A second opinion claimed over 100 Hz was reachable
and was disbelieved. Both turned out to be describing the same hardware in different
configurations.

=== Sync reads do not work at all

The SCS protocol manual documents READ and SYNC WRITE, but not SYNC READ. `rustypot`
exposes `sync_read_present_position` for the SCS0009 anyway, and the driver's only servo
telemetry path was built on it.

#callout(tag: "Measured", kind: "warn")[
  Every sync read of every field — position, load, speed, voltage, temperature, status,
  torque enable — times out after 500 ms, every time. All nine servos answer a plain
  per-servo READ immediately.
]

So the control station's servo telemetry had never worked. The default of
`--telemetry-hz 0` had been hiding it. The read path is now per-servo. SYNC *WRITE* is
fine, and remains what `sync_set_joints` uses.

=== The latency timer was ninety-nine percent of the cost

Nine per-servo reads cost 144 ms, and every individual read cost almost exactly 16 ms
regardless of which register it asked for. That uniformity is the tell: the U2D2 here is
an FT232H, and `ftdi_sio` defaults its latency timer to 16 ms. Every SCS response is a
few bytes, so it never fills a buffer and always waits out the full timer. At 1 Mbaud
the actual wire time for one exchange is about 0.15 ms.

#table(
  columns: (auto, auto, auto, auto),
  table.header([*latency timer*], [*9 joint positions*], [*rate*], [*with `present_load`*]),
  [16 ms (default)], [144.0 ms], [6.95 Hz], [3.47 Hz],
  [1 ms], [9.0 ms], [111.0 Hz], [55.6 Hz],
)

Zero errors in either configuration, across 445 and 556 bundles respectively. One sysfs
write is worth sixteen times the entire servo feedback rate, and nothing errors or warns
when it is missing — it simply runs slowly forever. `ServoBus` now lowers it on open and
reports the outcome; a udev rule makes it survive a replug.

=== The inter-command gap is a write requirement

The driver enforced a 300 ms quiet period after every single call, an empirical fix for
transactions that reproducibly timed out when issued back to back. Applied uniformly, it
also throttled reads.

#table(
  columns: (1fr, auto, auto),
  table.header([*pattern, no gap*], [*rate*], [*errors*]),
  [reads only], [111 Hz], [0 / 6003],
  [sync write, then a read sweep], [12 Hz], [14 / 91],
  [alternating single write and read], [46 Hz], [2 / 278],
)

What fails is a transaction issued too soon *after a write*; a read following another
read never did. Reads and writes now wait on separate clocks, so consecutive reads run
at bus speed while every write keeps the full quiet period around it — every ordering is
at least as protected as before, and only read-after-read got faster.

=== The rate that decides closed-loop feasibility

A later measurement, writing each servo the pose it was already holding so nothing
moved, separated the two halves properly.

#table(
  columns: (1fr, auto, auto),
  table.header([*pattern*], [*rate*], [*errors*]),
  [sync writes alone, back to back], [2859 Hz], [0 / 14298],
  [write, 0 ms, then 9 reads], [90 Hz], [1 / 451],
  [write, 1 ms, then 9 reads], [100 Hz], [0 / 500],
  [write, 2 ms, then 9 reads], [91 Hz], [0 / 454],
  [write, 10 ms, then 9 reads], [52 Hz], [0 / 263],
  [through the driver API, write plus full read], [75 Hz], [0 / 377],
)

#callout(tag: "The conclusion that matters", kind: "note")[
  A 50 Hz closed loop on this hand has real headroom. The bus was never the constraint;
  a default USB driver setting was. What still blocks the learned policies is the
  observation vector — they consume roughly 65 values including object pose and
  velocity, and this hand measures nine joint positions and nothing else.
]

#det("A regression this measurement caught", kind: "detail")[
  Encoding the write-gap rule initially applied the 300 ms delay to `sync_set_joints` —
  the one real-time path on the bus, whose own docstring says that a control loop
  calling it every frame cannot afford one. Consecutive frames were capped at 3.33 Hz,
  so a 50 Hz trajectory would have replayed fifteen times too slowly.

  That is the dangerous shape of bug: the moves all happen, in the right order, with no
  error anywhere — the experiment is simply wrong. It was caught because an operator
  read the documented numbers and asked whether they implied a 3.33 Hz write rate. There
  is now a test asserting the trajectory path sustains well above its own rate.
]

=== What the servos can and cannot tell us

- There is *no present-current register* on the SCS0009. Not a question of read cost —
  the data does not exist on the part.
- `present_load` is the only force proxy, is an uncalibrated duty-cycle-like number, and
  costs a second nine-transaction sweep. Any force claim built on it is unfounded until
  someone calibrates it against a load cell.
- `read_status` returns each servo's latched alarm byte, the closest thing this bus has
  to the packet-storm watchdog a Dynamixel chain provides.
- `torque_enable` reads 0 at power-up and 2 after an explicit disable. Both mean "not
  holding", but a servo reading 0 *after* the service set it means it rebooted.

== Homing takes two minutes, and that is correct

Each axis gets a timeout sized to guarantee it covered its full measured travel, so the
worst case for all six is 176 seconds. Measured on the real hand:

#table(
  columns: (auto, auto, auto, auto, auto),
  table.header([*axis*], [*finger*], [*travel*], [*window*], [*measured outcome*]),
  [J0], [thumb x], [112.4 mm], [46.4 s], [StallGuard2, 15.7 s],
  [J1], [thumb y], [56.2 mm], [22.4 s], [StallGuard2, 14.5 s],
  [J2], [index x], [62.5 mm], [28.0 s], [StallGuard2, 15.7 s],
  [J3], [index y], [56.0 mm], [25.0 s], [*timeout, 33.3 s*],
  [J4], [middle x], [62.2 mm], [29.4 s], [StallGuard2, 15.3 s],
  [J5], [middle y], [54.1 mm], [24.4 s], [*timeout, 32.7 s*],
)

StallGuard2 fires on four of six. J3 and J5 press against their hardstop for their whole
window, every time, under the current SGT tuning. The home is still trustworthy — the
window cannot expire early — but it sounds exactly like a hang, which is why the
operator was right to be suspicious and why the UI now shows the per-axis outcome and
the expected duration live.

#det("Adopting a home the board still vouches for", kind: "detail")[
  The M8P keeps its step counters, its `SETSCALE` calibration and its per-axis
  `homing_result` across a *host* restart. Only a board reset or an explicit `DIS`
  clears them. The service nonetheless discarded its home on every restart, demanding
  two minutes of re-homing for no new information, on a hand that might be holding
  something.

  `adopt_home` checks the board's own `homing_result` on all six axes and takes the
  reference over, and additionally adopts the morphology position when the gantries are
  already within 0.5 mm of the loaded plan's targets. It is logged as adopted, never as
  performed, so a run summary never claims a home this process did not watch.
]

== The first real trajectory

With the hand homed, positioned at the `g12` design and gripping a propped cylinder, the
exported open-loop reorientation ran end to end.

#table(
  columns: (1fr, auto),
  table.header([*measure*], [*value*]),
  [command frames], [55 over 1.08 s],
  [achieved rate], [50.9 Hz (target 50)],
  [frame interval], [median 20.0 ms, worst 21.8 ms],
  [servo bus errors], [0 in 24 021 transactions],
)

Timing was not the problem. Tracking was.

#table(
  columns: (auto, auto, auto, auto),
  table.header([*joint*], [*thumb*], [*index*], [*middle*]),
  [yaw error], [+1.2°], [*−5.6°*], [*+4.3°*],
  [mcp error], [−0.9°], [+0.4°], [0.0°],
  [pip error], [−0.9°], [−0.9°], [−0.3°],
)

#callout(tag: "The result", kind: "warn")[
  *The object dropped, almost immediately.* The flexion joints tracked their commands to
  under one degree, but the yaw joints on index and middle — precisely the joints
  producing the commanded 70° turn — arrived four to six degrees short. Earlier, at the
  grip set-point, the thumb's MCP was 10.1° short while the middle finger's was 2.7°.
]

The pattern is consistent and informative: the servos give up travel exactly where they
are loaded, the thumb carries the grip load, and the pair carries the turn load. The
simulation assumes a commanded angle is an achieved angle. On this hardware, under
contact, it is short by several degrees on the joints that matter most — and a
reorientation planned open-loop has no way to notice.

== Where this goes next

Three directions came directly out of watching the drop.

*Slower, and verified.* The trajectory ran at the speed it was exported at. Nothing
required that. Running the same set-points substantially slower, and checking arrival at
each one before proceeding, separates "the plan is wrong" from "the servos could not get
there in time".

*Load-gated stepping.* A debug mode that advances through the trajectory only when
`present_load` indicates the expected contact state, rather than on a wall-clock ramp.
This is the same shape as estimating a grasp wrench boundary from noisy normal-force or
motor-current data with semi-known contact positions — an uncalibrated proxy is still a
usable event detector even when it is not a usable force.

*A less optimal but longer trajectory.* The manually designed `r05` keyframe produced a
notably extended reorientation. It scores worse by the RL metrics, but "worse in
simulation and survivable on hardware" is the more useful trade right now.

== The road to the bench: artifact catalogue

The control station is the last link in about a week of work that took `real_v1` from a
CAD drawing to a hand with an exported trajectory. Those studies produced their own
self-contained reports, and they are the primary record — this page assumes their
conclusions rather than repeating them. Everything below is in the repository under
`docs/experiments/`, and mirrored into `artifacts/` beside this page when the site is
built.

=== Reports

#table(
  columns: (auto, 1fr),
  table.header([*Report*], [*What it settles*]),
  [#link("artifacts/20260827-real_v1/report.html")[real_v1 First Night] \ #text(size: 0.82em)[2026-08-27]],
  [The hardware model going from two hand-authored files to a topology the whole
   pipeline can run: base pair, per-design grasp keyframes, six designs through CEM, an
   A→B queue. Also where the screwdriver was lost, and the root cause turned out not to
   be friction or pad geometry.],

  [#link("artifacts/20260827-real_v1/report_mechanism.html")[The Rotational Lock] \ #text(size: 0.82em)[2026-08-28]],
  [Four designs whose mounts differ by 60 mm carried the shaft through the handoff and
   rotated it by under four degrees each. A result that survives that much change of
   hand is not a morphology result — it is the grasp acting as a rotational lock.],

  [#link("artifacts/20260827-real_v1/report_methods.html")[Two Routes to Vertical] \ #text(size: 0.82em)[2026-08-28]],
  [Head-to-head of the two things that stand a screwdriver up on this hand: a residual
   RL policy riding a scheduled second grasp anchor, and an open-loop geometric carry
   derived from the object's rigid-body kinematics. The open-loop carry is what the
   control station now replays.],

  [#link("artifacts/20260828-real_v1_search/report_search.html")[Which Hands Turn the Shaft] \ #text(size: 0.82em)[2026-08-28]],
  [108 hands over the CAD gantry workspace, each with a grasp sweep and a schedule
   sweep, each rollout repeated under spawn jitter. 49 stand the shaft up and keep it,
   and the deciding variable is not mount geometry but where the thumb pad sits.],
)

=== Working notes

#table(
  columns: (auto, 1fr),
  table.header([*Document*], [*Subject*]),
  [#link("artifacts/20260827-real_v1/SETUP.md")[SETUP.md]],
  [The 15-DoF morphology-actuated scene carrying the real finger and the real XY gantry
   travel, plus the first frozen example design.],
  [#link("artifacts/20260827-real_v1/SLIP.md")[SLIP.md]],
  [Why the screwdriver slowly slipped out during Policy A training, and the
   grasp-elevation fix. The origin of the "grasp at or below the equator" rule.],
  [#link("artifacts/20260827-real_v1/MECHANISM.md")[MECHANISM.md]],
  [The reorientation mechanism itself: workspace, carry sweeps, hold anchors, pivot.],
  [#link("artifacts/20260828-real_v1_search/REPORT.md")[REPORT.md]],
  [The 108-hand search in numbers: 49 of 108 hold a reorientation, 80 graspable at all.],
  [#link("artifacts/20260829-real_v1_deploy/DEPLOY.md")[DEPLOY.md]],
  [Written for the bench day. The open-loop schedule, the perturbation envelope, and the
   cell sweep that picks each design's operating point — the document the exported
   `g12` plan came from.],
)

=== Renders

#table(
  columns: (auto, 1fr),
  table.header([*Media*], [*Shows*]),
  [#link("artifacts/20260828-real_v1_search/figs/landscape.png")[landscape.png],
   #link("artifacts/20260828-real_v1_search/figs/scores.png")[scores.png]],
  [The 108-hand design landscape and its score distribution.],
  [#link("artifacts/20260828-real_v1_search/20260828-videos/rv04_mid_thAx20.mp4")[rv04_mid_thAx20.mp4],
   #link("artifacts/20260828-real_v1_search/20260828-videos/rv04_mid_thAx0.mp4")[rv04_mid_thAx0.mp4]],
  [The thumb-axial ablation that took `rv04_mid` from a score of 0.000 to 0.972 — the
   single most decisive knob in the search.],
  [#link("artifacts/20260828-real_v1_search/20260828-videos/rv05_manual_best.mp4")[rv05_manual_best.mp4],
   #link("artifacts/20260828-real_v1_search/20260828-videos/rv05_manual_openloop.mp4")[rv05_manual_openloop.mp4],
   #link("artifacts/20260828-real_v1_search/20260828-videos/rv05_manual_fail.mp4")[rv05_manual_fail.mp4]],
  [The hand-designed `rv05` keyframe, including its open-loop carry. It scores worse
   than the searched designs, and its reorientation is visibly longer and more extended
   — which is exactly why it is a candidate to retry on hardware after the `g12` drop.],
  [#link("artifacts/20260828-real_v1_search/20260828-videos/rv00_wide_best.mp4")[rv00_wide_best.mp4],
   #link("artifacts/20260828-real_v1_search/20260828-videos/rv03_narrowy_best.mp4")[rv03_narrowy_best.mp4]],
  [The wide and narrow-Y ends of the mount range, for contrast.],
  [#link("artifacts/20260827-reorient_primitive/primitive_compare.png")[primitive_compare.png]],
  [The reorientation primitive compared across hands — the study that found the turn is
   mostly floor and gravity rather than wrist.],
  [#link("artifacts/20260827-linklen_renders/76x41.png")[76x41.png]],
  [Link-length renders from the hardware finger geometry work.],
)

#callout(tag: "Dating is load-bearing", kind: "note")[
  Every generated folder carries a `YYYYMMDD-` prefix. This program reruns the same
  names constantly, and an undated folder is indistinguishable from a rerun three weeks
  later against different code. The links above are the same strings as the repository
  paths for exactly that reason.
]

#refbox[
  Operator runbook: `docs/hardware_control_station.md`. Driver and firmware:
  `src/morphohand/driver/manta/`. Regression tests, no hardware required:
  `tests/test_manta_hardware_faults.py`. Run logs: `logs/hardware/`. Study artifacts:
  `docs/experiments/`.
]
