# The shaft was creeping out of the real_v1 grip. It was the grasp pose, not the contact model.

Reported 2026-08-27 21:50 while Policy A was training on `rv00_wide`: the screwdriver slowly
slips out. The trainer agreed — `Episode_Termination/tip_lost` ran 1.4–2.3 per batch while
`Metrics/lift_height/object_height` sat at 0.106 against a 0.10 target. The lift worked and then
the grip degraded, which is a contact problem rather than a lift problem, so it should reproduce
with no policy at all. It does.

## What it actually is

`scripts/probe_real_v1_slip.py` loads a design's frozen scene, applies that design's own CEM grip,
ramps the palm up and then just holds. On `rv00_wide` the shaft rides the entire 10 cm lift with
three fingertip contacts, and then, holding still:

    hold step     0    40    80   120   160   200   240   280   320
    thumb   N  1.75  0.91  0.71  0.54  0.40  0.29  0.20  0.16  0.00
    index   N  0.87  0.46  0.36  0.27  0.20  0.14  0.10  0.08  0.00
    middle  N  0.88  0.46  0.36  0.27  0.20  0.14  0.10  0.08  0.00
    object z  .1069 .1108 .1107 .1105 .1103 .1101 .1098 .1095 .0929   -> on the floor by 360

The normal force decays monotonically to zero and the shaft falls straight down: its x and y move
by less than 1 mm over the whole failure. It is not sliding sideways and it is not rolling out —
the fingers open.

The cause is where the pads sit on the shaft. Measured on the fitted `open_ik` pose:

    thumb   pad centre  +4.25 mm above the shaft's axis  ->  elevation +10.1 deg
    index   pad centre  +4.25 mm                         ->  elevation +10.1 deg
    middle  pad centre  +4.25 mm                         ->  elevation +10.1 deg

All three contacts are on the shaft's UPPER hemisphere. The contact normals therefore tilt up and
outward, so the wedge drives the shaft down out of the pinch and only friction opposes it. Worse,
that friction acts against a normal force that shrinks as the shaft creeps down and the pads ride
toward the top of the cylinder, where the surface curves away from them. Positive feedback: a
grasp that is stable for the whole lift unwinds in 0.6 s of standing still.

`--elevation 10.0` was my own default in `fit_real_v1_pose.py`, chosen because a 21 mm-wide pad
cannot get under a 25 mm shaft lying on a table without going through the floor. At elevation 0
the pad's underside clears the floor by 2.0 mm, which is enough.

## The four candidate fixes, measured

`probe_real_v1_slip.py` on `rv00_wide_sp40`, 4 s of hold after a 10 cm lift. `held` is the lift
still present at the end; the base case is the failure above.

    variant                held   depth   axial  radial   tilt   contacts   force
    base                 -0.1mm  60.0mm   +0.1mm  94.7mm   0.4d  0 / 0.44   0.16 N
    pads_soft            -0.1mm       -   -0.0mm  31.2mm   0.0d  0 / 0.00   0.00 N
    pads_softer          -0.1mm       -   -0.0mm  74.5mm   0.0d  0 / 0.00   0.00 N
    fric_torsion         -0.1mm  60.0mm   +0.1mm  94.7mm   0.4d  0 / 0.44   0.16 N
    fric_mu4             -0.1mm  59.8mm   +1.2mm 100.4mm   1.0d  0 / 0.32   0.05 N
    palm_down5          101.2mm  53.7mm   +0.8mm   5.8mm   0.3d  3 / 3.00   2.14 N
    palm_up5             -0.1mm       -   +0.0mm   0.0mm   0.0d  0 / 0.00   0.00 N
    pads_soft+torsion    -0.1mm       -   -0.0mm  31.2mm   0.0d  0 / 0.00   0.00 N

Making the pads compliant (TPU-like `solref` 0.006 → 0.02–0.03 s, lower `solimp` dmax) does not
help and makes it slightly worse: a softer pad on the upper hemisphere sinks in and slides off
more easily. Torsional and rolling friction change nothing, because nothing is rotating.
Quadrupling sliding friction changes nothing, because the shaft is not sliding against the pads —
they have let go. **Wedge sign beats friction**, exactly as the `mujoco-eyes` skill says.

Lowering the palm 5 mm fixes it completely (3.00 mean contacts, 2.14 N). That is the same fix as
the elevation, arrived at from the other end: dropping the palm 5 mm moves the pads from +4.25 mm
above the axis to roughly 1 mm below it.

## Sweeping the elevation directly

Refitting `rv00_wide` at three elevations and scoring each candidate palm height with the pose
fitter's own close → lift → hold probe:

    elevation +10 deg   held  -0.10 /  -0.10 /  -0.10 mm    (fails at every palm height)
    elevation   0 deg   held  -0.10 / +60.53 / +60.55 mm
    elevation  -8 deg   held +56.33 / +58.14 / +58.12 mm

**Elevation 0 is now the default.** Going below the equator also works but costs grip depth
(52.0 mm at −8 deg against a 50 mm shaft half-length, i.e. 2 mm of clearance headroom, versus
55.5 mm at 0 deg) and grip depth is what clears the shaft's upper half when it stands up.

## The independent confirmation

The user hand-authored a morphology and a grasp in the actuated explorer the same evening,
without seeing any of the above. Its pads land at **+1.9 / −3.2 / +7.5 deg** — on the equator. A
person eyeballing the grasp put the contacts where the wedge works; the reachability-driven fitter
did not. That design is now `rv05_manual` in the design set, with the user's pose stored verbatim
rather than re-fitted.

## What changed

* `fit_real_v1_pose.py`: `--elevation` default 10.0 → 0.0, with the reasoning in `tip_targets`.
* `fit_real_v1_pose.py`: `hold_probe`'s hold 150 → 600 steps. At 150 steps (0.3 s) the probe
  reports this exact failure as a healthy grasp — the decay takes ~0.6 s to complete.
* `probe_real_v1_slip.py`: new, and the tool to reach for when a grip "should work".
* `real_v1_pipeline.py`: `rv05_manual` added, with `pose: "stored"`.

The pre-fix grasp table is kept as `PRE-SLIPFIX_real_v1.{json,txt}` so the comparison survives.
Policy A runs `20260827-2112-policyA_rv00_wide_t1` and its siblings were trained against the
broken grasp and are superseded.
