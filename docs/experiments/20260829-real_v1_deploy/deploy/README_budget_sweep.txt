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
