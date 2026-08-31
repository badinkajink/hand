g12w08 / g12w11 -- g12 re-exported with a wider per-joint action budget (2026-08-30).

The shipped g12 was clipped at +-0.5 rad (28.648 deg), which is Policy B's RESIDUAL
action budget and has nothing to do with an open-loop plan.  It bound: middle_yaw
and middle_pip both terminate at exactly 28.65 deg.  The trajectory actually wants
62.65 deg of middle yaw, so the shipped plan commanded 46% of its own turn.

Reproduce (budget 0.5 reproduces the shipped plan EXACTLY, so this is one variable):

  uv run python scripts/real_v1_export_plan.py --design g12 --object medium \
    --scene assets/mjcf/experimental/20260829-deploy_envelope/scene_tp0d0075p0d0000p0d0000_in0d0075n0d0150p0d0000_mn0d0075p0d0150p0d0000__medium__bench100py-35__flat15w21T.xml \
    --straddle-mm 40 --thumb-axial-mm 20 --squeeze-mm 10 --axis-k 0.100 --angle-deg -70 \
    --bench-height 100 --post-y -35 --budget <B> --out <dir>

  budget   thumb yaw   index yaw   middle yaw   middle pip   min clearance
  0.5      -11.08      -16.74      +28.65 clip  +28.65 clip  +8.7 / +8.8 mm   (== shipped)
  0.8      -11.08      -16.74      +45.84 clip  +45.84 clip  +8.7 / +8.8 mm
  1.1      -11.08      -16.74      +62.65 FREE  +63.02 clip  +8.7 / +8.8 mm

Finger-finger clearance is UNCHANGED -- the minimum is at u=0.00, the grip pose,
which the budget does not touch.  Widening the turn costs nothing geometrically.
Thumb and index never reached the clip at any budget, so they are unaffected.

UNTESTED ON HARDWARE.  A wider commanded turn is a larger load excursion, and the
middle yaw servo already trips its overload protection at the 28.65 deg command.
Expect the trip to come EARLIER, not later, until protective_torque is raised.

----------------------------------------------------------------------------------
sv1_u0060_b75 / rv05_manual_b85 / sv1_u0100_b70 -- Sobol-128 finalists (2026-08-30)

Promoted from the 128-hand pilot.  `_bNN` is the per-joint clip in centiradians,
the same thing `w08`/`w11` mean above.  Each design's clip was tuned to its own
hold band, not shared: see
  docs/experiments/20260830-real_v1-sobol128/deploy/README.md
for the band scan, the two designs that were rejected, and the gates that were run.
UNTESTED ON HARDWARE.

A later uniform budget-1.0 audit exported u0060/u0100/w0099/w0116 but added none here:
all four cleared simulated finger geometry, while none passed both driver validation and ten
repeated aligned 9.6-second holds. Evidence:
  docs/experiments/20260830-real_v1-sobol128/deploy_b100/README.md

----------------------------------------------------------------------------------
2026-08-30 late: the four budget-1.0 exports ARE here now, and g12 gets its own clip

ADDED AT THE USER'S REQUEST, after the audit above declined to add them.  The audit's
verdicts were re-checked independently and stand -- an independent 4-rep clip scan
agrees with its 10-rep holds design for design (u0060 4/4 vs 10/10, u0100 0/4 vs 0/10,
w0099 3/4 vs 7/10, w0116 2/4 vs 5/10), and HandPlan.validate reproduces both servo
violations to the hundredth of a degree.  They are here because four more points on
the clip axis is what a hardware/simulation disagreement needs, not because the audit
was wrong.

  sv1_u0100_b100   loads.  Drops in every sim rollout: 1.00 is above its 0.60-0.90 band.
  sv1_w0099_b100   loads.  Holds 3/4.  This design has no contiguous band at any clip.
  sv1_u0060_b100   WILL NOT LOAD -- middle yaw 3.88 deg outside the +-70 deg cap.
  sv1_w0116_b100   WILL NOT LOAD -- middle yaw 6.27 deg outside it.

HandRuntime.load_plan RAISES on any validate violation, so the last two will appear in
the dropdown and refuse to run.  Nothing is silently clamped.  Both are inside the
servos' declared +-85 deg contract and inside the measured hardstop; the +-70 is the
conservative cap set by hand in servos.py, and relaxing it is a decision for whoever
owns the hardware, not something a plan should route around.

  Largest clip each design can be COMMANDED (anchor and delta do not depend on the
  clip, so this is exact, not sampled):

    most of the family   ~1.345    first over: turn_end middle_fe2 (+77.05 deg cap)
    sv1_w0099             1.212    turn_end middle_aa
    sv1_u0060             0.932    turn_end middle_aa
    sv1_w0116             0.891    turn_end middle_aa
    g24                   0.650    turn_end thumb_fe1

  sv1_w0116's hold band is 1.45-2.00.  It lies entirely above 0.891.  There is no clip
  at which that hand both holds the tool and can be told to do so.

g12_b095 -- NEW, and it supersedes g12w08 and g12w11 for bench use.
  A clip scan on g12's own deployed cell (not the pilot's) puts its band at 0.70-1.05
  and its best alignment at 0.95, not at the lower edge:

    clip   0.70  0.75  0.80  0.85  0.90  0.95  1.00  1.05  1.10
    cos    0.61  0.52  0.48  0.49  0.60  0.63  0.59  0.52  0.27
    kept    4/4   4/4   4/4   4/4   4/4   4/4   4/4   4/4   1/4

  So the shipped g12 (0.50) is below the band and drops 4/4, g12w11 (1.10) is above it
  and holds 1/4, and g12w08 (0.80) sits in the band's dip.  g12_b095 holds 4/4 at cos
  0.627 -- the best simulated alignment of any plan in this directory.

  Reproduce: as the g12 line above, with --budget 0.95 --tag g12_b095.

Per-plan bands, clearances, expected ranking and a sentence of what to expect are in
catalog.json (the station shows them), generated by scripts/real_v1_plan_catalog.py from
docs/experiments/20260830-real_v1-budget-rescreen/.

READ THIS BEFORE TRUSTING ANY OF IT.  The 2026-08-30 bench session measured that the
hand performs 0.44-0.90 of the yaw travel it is commanded, stalling on middle yaw at a
servo load of 900+ against a trip near 800.  Everything above is commanded-clip
simulation, and the achieved clip is smaller by a design-dependent factor.  See
docs/experiments/20260830-real_v1_bench_sobol/README.md.

----------------------------------------------------------------------------------
2026-08-30 overnight: three hands from the clip re-screen, and they top the ranking

The 4,096-hand population was re-screened across five clips (28,480 rollouts) instead
of at the single inherited 0.5 rad, with a servo-command gate running inside the screen
for the first time.  119 of 269 morphologies survive a 5-repeat confirmation, and 50 of
those confirm at no clip the old screen ever tried.

Sixteen went through export / servo range / gantry travel / clearance.  Thirteen passed,
and NO design failed on servo range -- the new upstream gate did its job.  But those
sixteen were selected under the RETENTION maneuver (60 mm proof lift, five free
seconds), and the bench runs the FIXED-PALM schedule with the tool on a post.  Band-
scanned on the bench schedule, eleven of the sixteen keep the tool at NO clip at all,
including both new leaders (sv1_u1939, sv1_u2745).  A hand chosen for proof-lift
retention is not a hand that rolls a tool resting on a post.

Three survive both, re-exported at their BENCH clip:

  sv1_w2360_b075   band 0.50-0.80   cos 0.726   clearance +10.4 mm
  sv1_u1364_b080   band 0.40-0.95   cos 0.711   clearance  +5.0 mm  <- thin
  sv1_u0308_b050   band 0.40-0.85   cos 0.585   clearance  +9.2 mm

sv1_w3408 holds well (0.708) and is NOT here: its export clearance is +4.6 mm at every
clip, because the minimum sits at the GRIP pose, which the clip does not touch.
sv1_u1676 repeats sv1_w0116 exactly -- a band (1.05-2.00) entirely above the largest
clip its servos can be commanded to (0.80).

RANKING NOW ON THE STATION (simulation, bench schedule, 9.6 s hold, 4 repeats):

   1  sv1_w2360_b075   0.726  4/4   +10.4 mm     NEW
   2  sv1_u1364_b080   0.711  4/4    +5.0 mm     NEW
   3  g12_b095         0.627  4/4    +8.7 mm
   4  sv1_u0060_b75    0.597  4/4    +9.9 mm
   5  sv1_u0308_b050   0.585  4/4    +9.2 mm     NEW
   6  rv05_manual_b85  0.568  4/4   +10.3 mm     ran on the bench
   7  g12w08           0.484  4/4    +8.7 mm
   8  sv1_u0100_b70    0.231  4/4    +7.5 mm     ran on the bench

  not ranked: sv1_u0060_b100 and sv1_w0116_b100 will not load (yaw cap);
  g12, g23, rv04_mid, sv1_u0100_b100 drop 4/4 at their own clip; g12w11 holds 1/4;
  sv1_w0099_b100 has no band.  g23 (+0.8), g24 (-5.3) and rv04_mid (-2.7) also fail
  clearance and should not be run at all.

Full write-up: docs/experiments/20260830-real_v1-budget-rescreen/README.md

================================================================================
2026-08-31 -- THE YAW CAP CAME OFF, AND THE BENCH GOT AN OBJECT SENSOR
================================================================================

servos.FINGER_JOINTS' aa/yaw range went +-70 -> +-85 deg, the declared contract.
The +-70 was a conservative cap set before any plan had been driven, and it was
not protecting the servo -- it was deciding how far the hand may turn.  Both of
the plans it blocked overran on MIDDLE YAW at turn_end, and servo 6 was swept
-162.6..+136.8 deg by hand, so the restored bound is inside demonstrated travel.

All 17 plans now validate, verified on the CB1 itself.  Ranking, unchanged for
everything that already loaded:

   1  sv1_w2360_b075   0.726  4/4   +10.4 mm
   2  sv1_u1364_b080   0.711  4/4    +5.0 mm
   3  g12_b095         0.627  4/4    +8.7 mm
   4  sv1_u0060_b75    0.597  4/4    +9.9 mm
   5  sv1_u0308_b050   0.585  4/4    +9.2 mm
   6  rv05_manual_b85  0.568  4/4   +10.3 mm   ran on the bench
   7  g12w08           0.484  4/4    +8.7 mm
   8  sv1_u0060_b100   0.441  4/4    +9.9 mm   NEWLY LOADABLE
   9  sv1_u0100_b70    0.231  4/4    +7.5 mm   ran on the bench

  sv1_w0116_b100 now loads but stays unranked: it holds 2/4 at its own clip of
  1.00 and its band is 1.45-2.00, still above the 1.30 its servos can be told.
  It is a plan running 45 centirad below the band it wants.

The clip ceilings all moved.  sv1_u0060 is now commandable across its entire
0.65-2.00 band (it was capped at 0.90); sv1_u0308 and sv1_u1364 went 0.80 ->
2.00; sv1_w2360 0.85 -> 1.10; sv1_w0099 and sv1_w0116 to 1.30.

OBJECT TRACKING.  Two AprilTags now give the shaft's pose: turn angle, height
above the bench floor (and in the simulator's own z, one subtraction away), slip
and drop time, at 0.017 deg / 0.03 mm rms on a static target.  The station shows
it live and files it into each run's log next to -- never instead of -- the
operator's own reading.  Which means every number in the ranking above is now
falsifiable on the bench rather than merely written down.

  docs/experiments/20260831-real_v1-object-tracking/README.md
