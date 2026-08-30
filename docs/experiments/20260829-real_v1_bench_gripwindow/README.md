# real_v1 bench, 2026-08-29 evening — the grip window

Four instrumented open-loop turns on **g12** with the shaft in hand, every step
commanded-and-verified with a fresh servo sample.  The 2026-08-29 morning run
had no per-step feedback at all: the writer owned the servo bus for the whole
2.78 s, so its log holds 14 telemetry records and not one contains a servo
reading.  These four have one per step.

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

## Superseded

- The yaw servos are not weak and the fingers were not crashing into each other
  on g12.  With the clamp relieved, middle yaw tracked to within 3 degrees of a
  28 degree command (run 2).
- An earlier "g12 chord closes to 0.0 mm" reading was an artifact of
  `mj_geomDistance` on box-box pairs, not a collision — see
  `real_v1_trajectory_clearance.py`.  g12 clears on both paths.

## Open

- Middle yaw plateaus near 13 deg at load exactly 200 for 30+ steps whenever the
  thumb is firm.  Partial turn, stable hold; the remaining 15 deg is unexplained
  and, at ~1:1, is worth roughly another 15 deg of cylinder rotation.  This is
  the single highest-value thing to chase next.
- Index yaw does not return to its commanded zero (sits at -4.7 deg with a
  standing load ~240-270).  That axis is preloaded against something.
- No object-pose measurement exists.  Every reorientation verdict here is the
  operator's eyes.
