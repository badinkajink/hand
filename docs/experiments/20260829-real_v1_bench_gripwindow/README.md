# real_v1 bench, 2026-08-29 evening — the grip window

Four instrumented open-loop turns on **g12** with the shaft in hand, every step
commanded-and-verified with a fresh servo sample.  The 2026-08-29 morning run
had no per-step feedback at all: the writer owned the servo bus for the whole
2.78 s, so its log holds 14 telemetry records and not one contains a servo
reading.  These four have one per step.

Three earlier logs (`205127`, `205151`, `205525`) are the same trajectory run
**before the shaft was placed** — ungated, arrival-gated, and chord-vs-CSV.
They were shakedowns of the stepping code, but they turn out to be the control
the loaded runs need, so they are archived here too.

## What the runs establish

**Grip force and reorientation are in direct conflict on this hand, and both
failure modes sit within ~10 degrees of MCP command.**

| run | grip load th/ix/md | middle yaw reached | outcome |
|---|---|---|---|
| 211406 — plan grip | 525 / 350 / 210 | +4.1 deg of +12.5 | stall abort at step 10; held; operator saw a small reorientation |
| 211644 — full relief | 105 / 60 / 0 | +25.5 deg of +28.6 | ran the whole path; **dropped the shaft** |
| 212031 — uniform preload | 285 / 300 / 195 | +12.6 deg | stall abort at step 24, middle yaw load 855 (saturated) |
| **212231 — asymmetric** | **435 / 210 / 0** | **+13.2 deg** | **ran the whole path and held 4 s; operator measured ~15 deg of real cylinder rotation** |

**Object rotation tracks middle yaw close to 1:1.**  Run 4 turned the cylinder
about 15 degrees (operator estimate) while middle yaw travelled 13.2 degrees.
There is no object-pose sensor on this hand, so middle's yaw excursion is the
best available proxy for the reorientation, and the +28.6 deg the plan commands
would be worth roughly a 28 deg turn if middle could complete it.

**The shortfall is the clamp, and nothing else.**  The same commanded middle-yaw
sweep, measured at the last step:

| condition | middle yaw reached of +28.6 | error |
|---|---|---|
| 205525 — no shaft in the hand | +25.8 | -2.9 |
| 211644 — shaft, clamp fully relieved (dropped) | +25.5 | -3.2 |
| 212031 — shaft, uniform preload | +12.6 | -15.9 |
| 212231 — shaft, asymmetric grip (held) | +12.6 | -16.1 |

Free air and a shaft the hand is about to drop give the *same* tracking, so the
3 deg residual is the static servo droop already documented and the other 13 deg
is load.  The two clamped runs stall within 0.0 deg of each other, which makes
12.6 a repeatable ceiling rather than noise — and run 4 reached it by step ~24,
so the last 30 steps of commanded yaw moved the finger not at all.

## Why the plan over-clamps

`g12_build.txt` states the grasp it assumes:

    squeeze (pads driven inside the surface)   10.0 mm

Ten millimetres of pad penetration is soft-contact compliance in MuJoCo and pure
force on a printed shaft.  The exported grip is a POSITION the sim reaches with
the object already contacted; on hardware the fingers meet the shaft 7-10 degrees
earlier and position control converts the remaining travel into clamping force.
`real_v1_bench_regrip.py` anchors at the measured stall angle instead and walks
to a load band, which is what made runs 2 and 4 possible at all.

## Why the fingers want different things

The same build report gives the thumb a 20 mm axial offset ("thumb slid along
the tool"), the `--thumb-axial` term that took rv04_mid from 0.000 to 0.972 in
the design search.  That is a GRASP parameter, not a motion: the thumb's
contribution is torque from force at a moment arm, so it needs grip and has
almost no joint excursion (11 deg of yaw and 4 deg of mcp across the whole turn,
load never outside 90-195).  Middle has the opposite job — 28.6 deg of yaw — and
its own grip is what stalls it.  Relieving both together drops the shaft
(run 2); relieving neither stalls the turn (runs 1, 3); relieving middle alone
while keeping the thumb firm does both (run 4).

## The stall is the servo's overload protection, not a mechanical stall

Read from all nine servos on 2026-08-30 with the bus free (factory defaults, identical
across every servo):

| register | value |
|---|---|
| `max_torque_limit` | 100 % — **not** a limiter |
| `overload_torque` | 80 % |
| `protective_torque` | **20 %** |
| `protection_time` | 100 |
| `unloading_condition` | 32 |
| `minimum_startup_force` | 45 |

The SCS0009 protection scheme is: sustained load above `overload_torque` for
`protection_time` drops the servo's output to `protective_torque` and holds it there.
Run 4's middle-yaw load trace is exactly that signature:

```
 i   cmd_yaw  got_yaw  load
20   +23.92   +13.18    615
21   +25.06   +12.89    630
22   +26.20   +12.89    705
23   +27.34   +13.77    690
24   +28.48   +12.89    200   <- instantaneous 505-unit drop
25   +28.65   +12.89    200
...                     200   for 32 consecutive steps, never varying by one unit
```

Two things make this more than a coincidence. **200 is 20 % of a 0-1000 load scale**,
i.e. `protective_torque` exactly. And **200 is the only plateau value in the entire
bench dataset that is not a multiple of 15** — every genuine load reading in all seven
runs is quantized to 15, so 200 is not a load measurement at all. The other flat
stretches in the data (135, 195, 225) are ordinary: the load was already at that level
before the plateau. This one drops 505 units in a single step and then never moves.

Run 3 corroborates from the other side: its middle-yaw load reached 855 — above the
80 % threshold — at step 24, the same step at which run 4 tripped, and the stall abort
fired there. Free air never gets close: peak middle-yaw load 255-285, which is why the
same command reaches 25.8 deg with no shaft in the hand.

So the mechanism is: **grip load drives the yaw servo past 80 %, protection fires, the
joint is left with 20 % torque and cannot move for the rest of the trajectory.** The
finger is not mechanically stalled and the trajectory is not asking for something
impossible — the actuator has unloaded itself. `overload_torque`, `protective_torque`,
`protection_time` and `unloading_condition` are all per-servo writable
(`rustypot` `write_*`), so this is a configuration ceiling, not a hardware one.

**Not yet confirmed by a write test.** Setting `protective_torque` to 40 on
middle_yaw and re-running should pin the plateau at 400 instead of 200; that is the
decisive experiment and it has not been run. Raising it also removes a thermal
protection that exists for a reason.


## Reproducing run 4

With the shaft staged in the open hand and the CB1 service up:

```bash
python3 scripts/real_v1_bench_grip.py g12
python3 scripts/real_v1_bench_regrip.py --plan g12 \
    --target-thumb 450 --target-index 220 --target-middle 0 --preload-start 9.0
python3 scripts/real_v1_bench_stepped_run.py --plan g12 --csv --steps 55 \
    --gate 5.0 --gate-timeout 0.8 --dwell 0.1 --speed 80 \
    --regrip logs/regrip_pose.json --load-delta 400 --stall-deg 0.5 --stall-window 5 --hold 4
```

`--load-delta` alone aborted run 1 at step 10 on load that was the turn *working*;
the abort has to require a stalled yaw as well, which is what `--stall-deg` adds.
`--preload-start 9.0` matters: starting the walk-down at 5.0 capped the thumb at
load 270 and it could not reach the 450 target.

## Superseded

- The yaw servos are not weak and the fingers were not crashing into each other
  on g12.  With the clamp relieved, middle yaw tracked to within 3 degrees of a
  28 degree command (run 2).
- An earlier "g12 chord closes to 0.0 mm" reading was an artifact of
  `mj_geomDistance` on box-box pairs, not a collision — see
  `real_v1_trajectory_clearance.py`.  g12 clears on both paths.

## Open

- The overload-protection write test above: does `protective_torque = 40` move the
  plateau to 400?  That decides whether the missing 13 deg of middle yaw -- worth
  another ~13 deg of cylinder at 1:1 -- is recoverable by configuration.
- Index yaw does not return to its commanded zero (sits at -4.7 deg with a
  standing load ~240-270).  That axis is preloaded against something.
- No object-pose measurement exists.  Every reorientation verdict here is the
  operator's eyes.
