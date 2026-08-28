# real_v1 — the CAD-matched hand, set up for grasp + reorient (2026-08-27)

The hardware model landed as `assets/mjcf/real_v1/`: a 15-DoF morphology-actuated scene carrying
the real finger and the real XY gantry travel, plus one frozen example design. This is the setup
record for taking it through the existing pipeline — what the geometry is, what had to be fixed
before anything could be measured on it, and what the design set is.

Read `assets/mjcf/real_v1/8-27_LINK_LENGTH_GATE.txt` first for the geometry study that preceded
this, and `assets/mjcf/real_v1/real_hand_kinematics_tikz.tex` for the DH table.


## 1. The hand

Three identical fingers, three joints each (yaw / MCP / PIP), on independent XY gantries.

    phalange half-width                 r = 10.55 mm
    yaw link,    axis to surface            33.45 mm
    middle link, axis to surface            33.45 mm
    distal link, axis to pad surface        37.16 mm
    overhang past the next joint axis       12.70 mm  =>  joint spacing 20.75 mm
    flexing reach, mount to pad surface      78.66 mm

    yaw   +/- 85 deg   (0 = straight, + = abduction; axis is the finger's own long axis, a ROLL)
    MCP    -15 .. +92  (0 = straight, + = flexion)
    PIP    -18 .. +92

Consecutive links OVERLAP by 12.70 mm, which is why the joint spacing (20.75 mm) is smaller than
the link length (33.45 mm) — the distinction that the whole 2026-08-27 length study was about.
The thumb's flexion axes are mirrored so that positive flexion closes all three fingers.

Mounts and gantry travel, from `XY_space.png`, registered into the palm frame:

    thumb  (-50,   0) mm   travel  +/-30 x   +/-55 y     (60 x 110 box)
    index  (+50, +55) mm   travel  +/-30 x   +/-30 y     (60 x  60)
    middle (+50, -55) mm   travel  +/-30 x   +/-30 y     (60 x  60)

40 mm of clearance between the thumb band and the pair band, 50 mm between index and middle.
There is no proximal-length DoF: the links are CAD parts. The design space is the 6 XY dims.


## 2. Three defects fixed before anything was measured

All three are in `scripts/build_real_v1_scenes.py`, which now emits the base hand, the base
scene and the actuated explorer from one spec (`--check` fails if they drift).

**`<f>_tip` had no geom.** The RL fingertip contact sensor matches by BODY on
`("thumb_tip","index_tip","middle_tip")` (`rl/env_build.py::_build_sensors`). A tip body with no
geom can never appear in a contact, so the fingertip-contact reward, the tip-loss termination and
the grip-force scorecard would all have read exactly zero for a whole training run — and would
have looked identical to a policy that never touched the object. The distal capsule's terminal
cap now lives on the tip body as a sphere of the same radius at the same place: same collision
envelope, pad on its own body. (This is the baseline hand's convention.)

**223 N of self-interpenetration at rest.** The yaw capsule's cap is centred ON the mounting
plane, so it sits inside the palm plate; and the 12.70 mm overhang means the yaw and distal links
genuinely overlap. MuJoCo only auto-filters direct parent/child pairs, so both showed up as
permanent contact forces (223 N palm, 38 N per finger) before a policy did anything. 21
`<contact><exclude>` entries, mirroring what `assets/mjcf/perp/perp_hand.xml` carries for the
same reason. The hand now rests at 0.000 N.

**The palm was at the old hand's height.** 0.134 m was fitted to a 117 mm finger; this one reaches
78.66 mm, so at 0.134 the fingertips stop 30 mm above a shaft lying on the table and no grasp
exists at all. Palm pose is the arm's pose, not a hardware parameter, so it is SOLVED per design
by `scripts/fit_real_v1_pose.py`. Everything the CAD does fix — link lengths, ROM, mounts,
workspace — is taken from the user's model verbatim.


## 3. The grasp keyframe is authored, not retargeted

`retarget_keyframe_ik.py` transfers a known-good grasp across morphologies by IK-ing the
fingertips onto the same world positions with the palm held where the source keyframe put it.
Neither half of that applies here: there is no known-good pose on this topology, and palm height
is not shared across designs (a hand whose mounts sit 100 mm apart has to hang lower than one
whose mounts sit 40 mm apart). Inheriting one palm height for all of them is trap #1 in
LINK_LENGTH_GATE — it produced a spurious "2-FINGER, ungraspable" CEM verdict on a design that
grasps fine.

`fit_real_v1_pose.py` therefore solves palm x/y/z and the 9 finger angles together:

  * fingertip targets ring the shaft axis at pad-contact distance (shaft radius + pad radius +
    approach gap), thumb from -x and index/middle from +x, lifted 10 deg above the shaft's
    mid-height — a pad 21 mm across cannot get under a 25 mm shaft lying on a table without
    going through the floor;
  * palm x/y re-centre the thumb-pair midpoint over the shaft;
  * palm z is the DEEPEST height at which all three fingers still reach with every joint off its
    stop, because grip depth below the mounting plane is what buys clearance for the shaft's
    upper half when it stands up;
  * within a straddle, the palm height is chosen by an actual scripted close -> lift -> hold
    rollout (`hold_probe`), not by reachability: the pads are IK'd 4 mm INSIDE the shaft's
    surface to make a per-finger squeeze, and the candidate that still has the shaft up at the
    end wins. A uniform "+0.15 rad at MCP and PIP" squeeze was tried first and dropped the
    shaft on every candidate of every design, including ones CEM grasps perfectly — flexion is
    not "close", it curls the tip past the object.

It writes `open_ik` (the CEM seed, and what the RL env reads via `--open-finger-from-keyframe`)
and an `open` approach pose backed off by 8 mm at the same palm height, so closing is a pure
finger motion.

### The straddle is chosen by CEM, because nothing cheaper works

How far apart index and middle sit along the shaft is a choice about where to grasp, not a
hardware parameter — and on this hand it decides verdicts. Three ways of picking it from geometry
alone were tried, and all three produced a false verdict on some design:

    nominal 30 mm, first reachable palm height
        rv01_compact's CEM grasp lifted the shaft and then DROPPED it — held lift -1.4 mm,
        contact persistence 0.86, score 6.87. The identical hand fitted at 40 mm: held +47.0 mm,
        persistence 1.00, score 8.26. The design was never the problem.
    widest reachable straddle
        fixed rv01 and broke rv00_wide the same way — it had been holding +48.2 mm at 30 mm and
        dropped at ~42 mm.
    scripted hold probe over (straddle, palm height)
        ranks palm heights usefully, but reports that rv01_compact has no holding pose at all,
        when CEM finds one. A geometric grip is not a CEM grip.

Reachability cannot see the failure mode. The thumb sits at y=0 and the pair straddles it; those
three contacts are the tripod's base against the shaft PITCHING about the pinch axis, and whether
it resists depends on the straddle AND on how extended the fingers are to hold it, which trade
against each other. So the pipeline fits and CEMs each design at BOTH candidate straddles and
keeps whichever actually holds. One extra CEM per design, no guessing.

Worth knowing when reading the per-design numbers: because the fitter places fingertips at WORLD
targets, every design that can reach them grasps the shaft at the same three points with the same
pad geometry. The designs differ in the joint configuration that gets there — hence in effective
stiffness at the contact and in what the hand can do NEXT — not in where they touch.


## 4. The design set is 2-dimensional, not 3

The workspace has three natural knobs (thumb forward / pair back / pair inward), but once the
fitter re-centres palm X over the shaft, **"thumb forward" and "pair back" are the same hand**:
(1,0,0) and (0,1,0) both put the thumb 70 mm from the pair and both fit at 68.0 mm of grip depth
with identical joint angles and identical residuals. Only two things vary:

    thumb <-> pair separation along X    100 mm  ..  40 mm
    index <-> middle separation along Y  110 mm  ..  50 mm

So the set is a 2x2 factorial in those two plus a centre point, and the redundant single-knob
designs are dropped:

    rv00_wide      X 100   Y 110    CAD-nominal gantry centres
    rv01_compact   X  40   Y  50    every gantry inboard
    rv02_narrowx   X  40   Y 110    tight pinch, wide straddle
    rv03_narrowy   X 100   Y  50    wide pinch, narrow straddle
    rv04_mid       X  70   Y  80

X separation sets how hard the pinch has to squeeze and how deep the palm can sit.

Y separation is the interesting one, and the prediction is sharp. The pinch axis is X and the
shaft lies along Y, so bringing the shaft vertical means rotating it about the pinch axis. The
thumb sits at y=0 and acts as the pivot; index and middle straddle it at +/-Y and have to drive
the rotation differentially — index tip up, middle tip down. A 90 deg rotation moves each pair
contact by its own Y offset in Z. So:

    rv00_wide / rv02_narrowx   pair at +/-55 mm   =>  each tip must travel ~55 mm in Z
    rv01_compact / rv03_narrowy pair at +/-25 mm  =>  ~25 mm

against 78.66 mm of total flexing reach, while keeping the grip. That predicts the wide-straddle
designs are reorient-HOSTILE for kinematic reasons even though they grasp identically well, and
it is falsifiable: if the wide designs reorient anyway, the mechanism is not this one. Nothing
about the grasp stage can see this — pick-up and hold have been solved everywhere in this
program, on every hand it has built.

Results land in `real_v1.txt` / `real_v1.json` next to this file. Read the grasp column as the
HELD lift (`cube_z_after_hold - cube_z_before_lift`), not CEM's `cube_lift`, which is a PEAK and
scores a grasp that lifts and then drops identically to one that holds — on the first pass of
this set, two designs of five reported `cube_lift` 0.047 with held lifts of -0.001 and +0.000.

Lift height is 0.10 m, so the shaft's lower end sits at z = 0.0625 when vertical: this is a TRUE
in-hand reorient with no floor to brace against. The 2026-08-26 primitive study found 69% / 46%
of m05's and perp's alignment gain happened on the floor; that route is not available here.


## 5. Running it

    # cheap prefix: generate + pose + CEM for every design, ~6 min each
    MUJOCO_GL=egl uv run --extra rl --extra gpu python scripts/real_v1_pipeline.py --stages grasp

    # expensive: A (from scratch) -> B (live-A reset, warmstart THIS design's A) -> handoff eval
    nohup setsid env MUJOCO_GL=egl uv run --extra rl --extra gpu \
      python scripts/real_v1_pipeline.py --stages all --only rv00_wide,rv01_compact \
      > logs/real_v1_pipeline.run.log 2>&1 & disown

Nothing transfers onto this hand. Policy A trains from scratch per design (`WARMSTART=none`) and
Policy B warmstarts only from that design's own A — b33 did not survive even a proximal-length
change on the SAME topology, let alone a new one.

Geometry is pinned by `tests/test_real_v1_scene.py`: CAD lengths, the yaw link's perpendicular
offset (zero there is the coincident-joint bug), ROM, workspace registration, tip geoms, zero
self-collision at rest, hand/scene parity, and the generator round-trip.
