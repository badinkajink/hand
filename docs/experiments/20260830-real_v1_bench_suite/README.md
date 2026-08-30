# real_v1 bench suite — offline data-collection protocol (2026-08-30)

A protocol you can run **alone**, without a turn-by-turn agent, that produces a
directory per session which is interpretable weeks later by someone who was not
there.  Written because the 2026-08-29 bench produced four good runs and three
orphan JSONLs whose meaning had to be reconstructed the next day from timestamps.

Everything below runs from the workstation against the CB1 service.  One command
per session:

```bash
python3 scripts/real_v1_bench_session.py --design g12 --note "first full sweep"
```

Hand back the whole `docs/experiments/20260830-real_v1_bench_suite/<stamp>-<design>/`
directory.  It contains `MANIFEST.json` (preflight state, plan facts, every
command run, every operator observation), the raw per-step JSONL for each run,
and the stdout of each script.  Nothing else is needed to read it.

---

## 1. All four designs are runnable, three of them truncated

The other agent's verdict — "only g12 is bench-ready" — is about the **full**
trajectory.  Every design's turn is safe up to the point where two fingers
would meet; the fix is to stop there rather than skip the design.
`scripts/real_v1_trajectory_clearance.py --all` gives the crossing, and
`--max-u` (new) stops the run before it.

| design | drivers (relieve) | holders (keep firm) | joints clipped at ±0.5 rad | safe `max_u`, chord | csv | min clearance |
|---|---|---|---|---|---|---|
| **g12** | middle | thumb, index | 2/9 | **1.00** | 1.00 | +8.7 mm |
| **g23** | middle | thumb, index | 2/9 | **0.92** | 0.84 | +0.8 mm |
| **g24** | thumb, middle | index | **5/9** | **0.55** | 0.10 | −5.2 mm |
| **rv04_mid** | thumb, middle | index | 3/9 | **0.65** | 0.70 | −2.6 mm |

The session driver picks `max_u` from this table automatically.  **Do not raise it**
without re-running the clearance scan — the sim's links are thinner than the
printed parts, so these numbers are already optimistic.

Two things fall out of the table before any hardware runs:

- **g24 and rv04_mid have only one holder.**  Two of three fingers travel more than
  60 % of what the busiest one does, so a single finger has to anchor the shaft
  while the other two turn it.  These are also exactly the two designs that
  interpenetrate.  If they drop the shaft twice, that is the result, not bad
  staging — record it and move on.
- **Every design saturates joints at the ±0.5 rad (28.648°) residual clip**, g24 on
  five of nine.  That clip is inherited from Policy B's action budget and has no
  business constraining an open-loop plan; hardware yaw range is ±85°.  Nothing in
  this protocol fixes that — it is a re-export, noted here because it caps what any
  of these runs can show.

## 2. Bring-up (web app), once per sitting

1. `ssh irlab@10.99.99.2` → `~/run_control_station.sh start` (password in `.dotenv`,
   `MANTA_IRLAB_PW`).
2. Open `http://10.99.99.2:8765/`.  Confirm **backend real**, telemetry ticking.
3. Enable **servo torque**.  Home the gantries if you need them; the finger work
   does not require homing.
4. Leave the web app open for the whole session — it is the E-stop (`Stop`,
   `Motors disable`) and the only live view of load.
5. The session driver refuses to start if torque is off, the service is busy, or
   telemetry is suspended.  **Telemetry suspends whenever a writer owns the bus**,
   so a "stale" reading during a run is expected; a stale reading *between* runs
   means something else is holding the bus.

## 3. The three arms, in order

The driver runs them in this order by default (`--arms freeair,grip,loaded`).
Order matters: the free-air control is what lets a loaded shortfall be attributed
to grip rather than to the design.

### freeair — nothing in the hand
The whole turn, no object, no clamp, no risk.  Measures the design's own tracking
floor.  On g12 this was 25.8° of a commanded 28.6°, i.e. ~3° of static servo droop
(`minimum_startup_force = 45`, a 4.5 % deadband).  **Run this for every design,
even ones you do not intend to load** — it is free and it is the only way to tell
a stalled finger from a droopy one.

### grip — seat, then relieve
`real_v1_bench_grip.py` ramps open→grip over ~1 s (never a single set-joints: the
grip is 30°+ from open on all three MCPs).  Then `real_v1_bench_regrip.py` walks
each finger's MCP down until its load enters a band.

The bands are **per finger, by role**, which the driver sets from the plan:
holders to |load| ≈ 440, drivers to ≈ 0.  Uniform preload stalls the turn; full
relief drops the shaft; relieving the drivers alone did both on g12.

### loaded — the instrumented turn, ×3
Arrival-gated, stall-gated, one measured servo sample per step, then a 4 s hold so
the log records whether the shaft was still there a second later.  **Three repeats
minimum.**  A single run cannot distinguish a design from a staging accident, and
the 2026-08-29 stall was only believable because two runs hit it within 0.0°.

Between repeats the driver waits for you to re-stage the shaft.  Stage it the same
way every time and say so in the notes if you cannot.

## 4. What to write down

After every loaded run the driver asks six questions.  Answer them even when the
answer is boring — a blank is indistinguishable from "did not happen".

| field | why it exists |
|---|---|
| `held` | the log ends at the trajectory; whether the shaft is still in the hand is not in it |
| `rotation_deg` | **the actual dependent variable.** There is no object-pose sensor |
| `rotation_source` | eye / protractor / apriltag / video — an eyeball estimate and a tag read are not the same measurement |
| `fingers_touched` | ground truth for the clearance table above |
| `slipped` | distinguishes "the fingers turned the shaft" from "the shaft rolled against them" — the same yaw excursion, opposite meaning |
| `media` | filename of any photo/video, so it can be found later |

**On rotation measurement.** Until the AprilTag mount exists, the honest fallback
is a fixed phone on a tripod, perpendicular to the pinch axis, with a paper
protractor behind the shaft, and a still at the grip pose and at the end pose.
That is ±3–5°, which is enough to separate 4° from 15° but not enough to rank two
designs that land within 5° of each other.  Record `rotation_source` so we know
which claims the data can support.

Useful proxy in the meantime: **object rotation tracked driver-finger yaw close to
1:1** on g12 (15° of cylinder for 13.2° of middle yaw), and driver yaw *is* in the
log.  Do not treat it as a substitute — it is a hypothesis from one run.

## 5. Suggested sweep, in priority order

Roughly 20 minutes per session.  Stop whenever the hand starts behaving
differently from earlier in the day (servos warm up and `minimum_startup_force`
is temperature sensitive).

| # | command | what it settles |
|---|---|---|
| 1 | `--design g12 --arms freeair,grip,loaded --repeats 3` | reproduces the only known-good run, three times.  If this does not reproduce, stop — something changed since 2026-08-29 and nothing after it is interpretable |
| 2 | `--design g23 --arms freeair,grip,loaded --repeats 3` | the nearest neighbour that clears to u=0.92.  Same two-holder structure as g12, so it should behave like it |
| 3 | `--design rv04_mid --arms freeair,grip,loaded --repeats 2` | the search's **best** design (nominal cos 0.997) and the worst deployable one.  One holder, truncated at u=0.65.  Expect drops; that is the point |
| 4 | `--design g24 --arms freeair,grip,loaded --repeats 2` | one holder, 5/9 joints clipped, truncated at u=0.55.  Least likely to work, cheapest to falsify |
| 5 | `--design g12 --arms loaded --repeats 3 --path csv` | chord vs dense path on the one design where both clear.  They are **different trajectories** — the CSV keeps per-joint timing the 3-set-point export dropped |
| 6 | `--design g12 --arms protection` | the overload write test, below |

If usage or time runs out, rows 1–3 are the ones worth having.

## 6. The overload-protection write test

The 2026-08-29 stall was **not mechanical**.  The middle-yaw load rose 615 → 630 →
705 → 690 and then dropped in one control step to exactly **200** — 20 % of the
0–1000 scale, the servos' `protective_torque` — and sat there for 32 consecutive
steps without varying by one unit.  200 is the only plateau in the whole bench
dataset that is not a multiple of 15, so it is not a load measurement at all.
Sustained load above `overload_torque` (80 %) trips the servo into unloading itself.

All four protection registers are per-servo writable, so this may be a
**configuration** ceiling.  The test needs the raw bus, so it runs on the CB1 with
the control station stopped — `--arms protection` prints the exact procedure and
records your readings.  The number to read is the **plateau value**: 200 before,
400 after, if the mechanism is what we think.

**Put `protective_torque` back to 20 afterwards.**  That protection exists for a
thermal reason and we have no temperature margin data.

## 7. Things not to do

- Do not command the plan's grip pose directly.  `<design>_build.txt` asks for
  10 mm of "squeeze" (pads driven inside the object surface) — that is soft-contact
  compliance in MuJoCo and pure clamping force on a printed shaft.  Always go
  through `grip` → `regrip`.
- Do not use `--load-delta` without `--stall-deg`.  Load rising *is* the turn
  working; only load-plus-no-motion is a fault.  A bare load abort killed run 1 at
  step 10 on a healthy trajectory.
- Do not `--preload-start` below ~9.0.  Starting the walk-down at 5.0 capped the
  thumb at load 270 and it never reached its 450 target.
- Do not run a design at a `max_u` above the table without re-running the
  clearance scan.
- Do not read servo telemetry while a writer owns the bus; you will get one stale
  sample repeated.  Every helper here already asserts
  `servo_polling_suspended == False` and a fresh `servo_age_s`.

## 8. Handing the data back

```
docs/experiments/20260830-real_v1_bench_suite/20260830-1530-g12/
  MANIFEST.json          preflight, plan facts, commands, observations
  freeair.jsonl          per-step commanded/achieved/load
  grip_seat.stdout.txt
  regrip.stdout.txt
  regrip_pose.json       the relieved grip actually used
  loaded_1.jsonl  loaded_2.jsonl  loaded_3.jsonl
  loaded_*.stdout.txt
```

Commit the directory (these are small) and say which sessions are new.  The first
thing worth extracting from a loaded log is the driver finger's **yaw at the last
step** against the free-air value from the same day — that difference is grip load
and nothing else.
