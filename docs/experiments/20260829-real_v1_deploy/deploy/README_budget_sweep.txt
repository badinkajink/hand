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
