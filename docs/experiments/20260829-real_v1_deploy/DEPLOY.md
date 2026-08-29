# What survives a hand that is not the simulated hand

2026-08-29. Written for a bench day. Everything here is the **open-loop schedule** — close to a
fitted grasp, then run a straight line in joint space to one end pose — because that is a list of
joint set-points streamed to position servos, which needs no object-pose estimate, no camera and
no policy. The trained anchored B policies need an observation vector the prototype does not have.

Harness: `scripts/real_v1_deploy_envelope.py`. It plans the trajectory on the nominal hand and
replays **that same trajectory, unchanged**, on a hand / tool / servo that differs from it.
Re-fitting on the perturbed scene would measure whether the *pipeline* can adapt, which is not
what happens when the plan is already loaded on the robot.

`win` throughout = the tool is still held (contact, and no more than 20 mm of height lost) AND
final cos >= 0.7. Cosine alone scores a shaft lying on the table at 1.0.

## 1. The failure is in the turn, not the grasp

Pooled over 18 designs, by phase:

    perturbation            grasped   lifted   kept through the turn   right pole   win
    none (0.5 mm jitter)      100%     100%            77%                73%       72%
    servo zeros +-1 deg       100%     100%            60%                56%       42%
    servo zeros +-2 deg       100%     100%            39%                36%       25%
    servo zeros +-4 deg        73%      76%            20%                17%       10%
    mount error 2 mm          100%     100%            80%                75%       55%
    shaft 6% thinner          100%     100%             0%                 0%        0%
    everything at once         87%      89%            18%                14%        7%

Of the runs that lift and then lose it, 61% (at 2 deg) to 79% (full ensemble) **drop the tool
during the turn**; only 4% go to the wrong pole and 9% under-turn.

So gating the close on contact force buys nothing at the errors that matter: at 1-2 degrees the
close is already perfect. The grasp and hold are the solved part of this program and they stay
solved. What does not survive is the rotation.

## 2. Why the turn is fragile: the grip extinguishes itself

Per-finger normal force through g14's open-loop turn, nothing perturbed:

    step   cos     thumb  index  middle  total
      55  -0.03    13.9    8.0     5.9   27.8 N
     165  +0.28    14.7    4.3    10.5   29.5
     275  +0.63     5.5    0.0     6.4   15.9
     385  +0.63     0.1    0.0     0.0    0.3
     550  +0.77     0.1    0.0     0.0    0.1
    hold  +0.93     0.1    0.0     0.1    0.2

The index pad leaves at ~50% of the turn and the grip is gone by 70%. **The last third of the
rotation is the shaft settling into vertical under gravity in a hand that is barely holding it.**
That is why millimetres and degrees decide the outcome.

## 3. Three proposed fixes, all measured, all negative

**A precomputed re-squeeze and a slower turn are worse.** Pooled over 1,980 cells:

    turn steps   re-squeeze   nominal kept   ensemble win
       550          0.0 mm        61%            5%
       550          3.0 mm        74%            2%
      1100          0.0 mm        51%            3%
      1100          3.0 mm        61%            2%

**Closing a current loop on the fingers trades turning for holding.** 150 draws per arm, the
identical draws across arms, pooled over six designs at half the plausible bench error:

    per-finger force set-point   0 N    1 N    2 N    4 N    8 N
    win, loop through the turn   31%    20%    19%    19%    15%
    kept, loop through the turn  57%    59%    65%    68%    69%
    win, loop only after it      31%    28%    28%    28%    28%

It works exactly as the mechanism predicts: squeeze harder and the shaft cannot settle. The loop
is per-finger normal force via a Jacobian squeeze toward the pinch axis — encoders and servo
current only, no object sensing — so the negative result is about the physics, not the sensing.

## 4. What to control, and to what

A 1 mm / 2 degree placement map, 20 draws per point, 12 designs. Width of the band that stays
above 50%:

    design        along the tool     across it        yaw
    g14           -4 .. 0 mm         -2 .. +1 mm      -4 .. +12 deg
    g03           -1 .. 0            -8 .. +6         -20 .. +20
    g12           +0 .. +1           -6 .. +3         -18 .. +12
    rv00_wide     -3 .. 0            -8 .. +8         -20 .. +6
    ax_asym-10    +0 .. +4           -3 .. +4         -6 .. +10
    rv04_mid      -1 .. +2           -4 .. +3         -18 .. +8

**Axial placement is the tight one and its basin is not centred on the plan.** The fitter always
rings the pads on the tool's mid-length; for half these designs the working band sits 1-4 mm to
one side of that, so the shipped plan starts at the edge of its own tolerance. Transverse
placement and yaw are forgiving on the better designs.

Tool build: the scene's cylinder is 24.5 g, which is PLA at about 40% effective density. Solid
PLA is 2.5x that and 2x already drops it, so **print at or below ~40% infill**. Diameter is
one-sided: 6% under nominal drops it on every design, 6% over is fine or better.

Servo torque is not a constraint — a quarter of the modelled +-10 N.m still works. Servo
stiffness matters more than torque and softer is safer than stiffer.

## 5. Robustness re-ranks the designs, and the operating point is worth as much as the hand

Nominal repeatability at n=20 is 45-95%, not the search's 3/3: rv04_mid, published at
0.997 +- 0.002, wins 80%; g10, published at 0.967, wins 45%. **n=3 was too small.**

Choosing each design's operating point for the behaviour of its *neighbours* rather than for its
own peak roughly doubles the ensemble win rate — g03 12 -> 25%, rv04_mid 12 -> 21%, r08, g12 and
rv00_wide all 0 -> 12%. The winners of the 108-hand search were peaks of narrow resonances.

Under perturbation the ranking is g14 (29%), g03 (25%), then rv04_mid, g24, ax_py-20 (21%).
g03 and g14 placed 26th and 10th in the original search.

## 6. Without the z stage, the schedule does not transfer — the operating point does not, anyway

The prototype has no palm stage: the hand is fixed and the tool sits on a small platform with
~100 mm of clear space beneath it. That is a different problem from "lift it to the same height",
and the difference is not small. Replaying each design's lift-tuned plan on the bench, 40 draws
at half the plausible bench error:

    design        plan tuned on the LIFT     plan tuned on the BENCH
    g24             5% win /  5% kept         30% win / 45% kept
    g12            15% win / 70% kept         32% win / 70% kept
    g14            12% win / 65% kept         28% win / 50% kept
    rv00_wide       0% win / 18% kept         20% win / 68% kept
    ax_asym-10      0% win / 65% kept         10% win / 55% kept
    rv04_mid        5% win /  5% kept          2% win / 60% kept
    POOLED          6%                        20%

At its own lift-tuned cell every design simply puts the tool on the floor. The bench-optimal
cells are systematically elsewhere — larger turn angles, different straddles, different pivot
heights — so **the search has to be redone for a stationary hand; it cannot be inherited.**

Two consequences worth stating plainly. **Policy A is the lift**, so on a fixed palm the A->B
architecture has no A and the deployable is scripted-close -> scripted-turn. And the support is
an obstacle: under a -90 deg turn the +Y end of the tool descends through where a CENTRED post
is, and 55% of runs end resting on it. Moving the support to y = -35 mm (under the half that
RISES) drops that to 5-15%.

## 7. The finger is not a 21.1 mm capsule, and that changes the contact

Correction from the user, 2026-08-29: the built finger is **14.8 mm across with a flat
object-facing face**, and only the 10.55 mm from joint axis to that face is what the shipped
scene got right. Every phalanx in the MJCF is a capsule of radius 10.55 mm and every fingertip a
sphere of the same radius, so the simulated hand is round and makes point contact.

Measured, at a settled grasp, contact in the tip's own frame:

    pads          contact point (local x, y, z, mm)        normal force
    round, g14    (+9.0, 0.0, -3.2) (-8.6, +0.4, -4.6)     12.5 / 9.8 / 2.7 N
    flat,  g14    (+10.4, +7.4, -4.9) (-10.4, +7.4, -0.8)   0.7 / 0.5 / 0.2 N

Two things. The round pad touches 20-35 degrees off the finger's own facing direction, which a
flat face cannot do — the flat pad meets the shaft as a LINE along it, and MuJoCo puts the
contact at the line's end, hence y = +-7.4 mm. And the grip force collapses at identical joint
commands, because `fit_real_v1_pose` places pads at `object_radius + PAD_RADIUS + gap`, exact for
a sphere and short by `(R + 10.55)(1/cos t - 1)` ~ 2-5 mm for a flat face meeting the shaft at
angle t. **Every grasp in the 108-hand search was fitted for a hand that does not exist.**

The correction is `Scene.set_finger_flat_pads`, whose one assumption is the part's thickness
behind the face (the two stated numbers do not fix it); the default fills from the joint axis to
the face, which cannot invent collision geometry.

## 8. The real hand on the real bench, re-searched

8,190 cells: 14 designs x straddle {25,32,40} x thumb offset {0,10,20} x 13 pivot heights x 5
turn angles, flat pads, fixed palm, support at y = -35 mm. **212 cells reorient and keep the
tool.** Ranked by the robustness of the cell AND its neighbours in pivot height:

    design      straddle  thumb  axis_k  angle |  nom cos  kept   ensemble win  plateau
    rv04_mid      40 mm    10mm   0.05    -70  |   0.794   7/8        50%         41%
    g23           40       10     0.05    -70  |   0.852   8/8        38%         38%
    g24           40       10     0.05    -70  |   0.919   8/8        38%         31%
    g14           40       20     0.15    -80  |   0.552   8/8        19%         13%
    g01           25       10     0.05    -80  |   0.667   6/8        12%         12%

The corrected hand is **more** robust than the round-pad model, not less: 38-50% under the full
joint ensemble against 20% for the round pads on the same bench. A line contact along the shaft
resists axial and yaw disturbance in a way a point contact does not.

The thumb-offset result from the 108-hand search survives the geometry correction: 95 working
cells at 20 mm against 48 at 0 mm.

**The three best hands differ only in pair-Y separation** — thumb at (-35, 0) and the pair at
x = +35 with y = +-40 (rv04_mid), +-32.5 (g23), +-25 (g24). One gantry axis slides between all
three, so all three can be tried in a single bench session without rebuilding.

Build sheets and 50 Hz servo trajectories: `deploy/<design>_{build.txt,poses.txt,traj.csv}`,
from `scripts/real_v1_export_plan.py`.

## 9. What the z stage is worth, on the corrected hand

The same 8,190-cell search run twice on the flat-pad hand — once with the palm lift, once fixed
on the bench. Best cell per design, `nominal cos / ensemble win`:

    design        with the palm lift              fixed palm on a platform
    ax_tx+20      sp40 t20 k0.65 a-70  0.93/69%   sp40 t0  k0.05 a-100  0.70/12%
    g10           sp40 t20 k0.65 a-70  0.80/62%   sp40 t20 k0.65 a-70   0.13/12%
    g12           sp32 t10 k0.65 a-80  0.91/44%   sp32 t20 k0.65 a-80   0.82/19%
    g23           sp32 t10 k0.60 a-90  0.99/31%   sp40 t10 k0.05 a-70   0.85/38%
    g24           sp32 t0  k0.65 a-110 0.98/31%   sp40 t10 k0.05 a-70   0.92/38%
    rv04_mid      sp32 t20 k0.65 a-80  0.83/31%   sp40 t10 k0.05 a-70   0.79/50%
    ax_asym-10    nothing holds                   sp25 t0  k0.35 a-110  0.43/12%

312 working cells with the lift against 212 without it, and the best cell 69% against 50%. **The
z stage is worth about 19 points of robustness and is not a prerequisite.** No design's operating
point is the same in both columns, and three designs are better without the lift than with it —
which is the same lesson as section 6, now on the hand that exists.

The whole envelope is better than the round-capsule model said: 69% / 50% under the full joint
ensemble against 31% / 20% before. Correcting the geometry did not cost robustness, it revealed
some. Best single recommendation with a z stage is `ax_tx+20`; without one, `rv04_mid`.

## 10. Paying back the flat pad's under-grip

2,268 cells on the bench at squeeze 4 / 7 / 10 mm:

    squeeze   cells that work   nominal kept   mean nom cos   best ensemble win
     4.0 mm         19              18%           0.050            31%
     7.0 mm         63              26%           0.069            44%
    10.0 mm        113              40%           0.132            31%

A deeper squeeze **widens where the schedule works** — six times as many viable cells at 10 mm —
but re-measured at 80 draws on the cells already selected, 7 mm is a wash (49% against 48% at
half error, 15% against 19% at full). So: keep 4 mm at a tuned operating point; reach for more
squeeze only when searching for one, where it is the difference between 19 cells and 113.

## 11. Bench tolerances on the hand that exists

Corrected geometry, fixed palm, each design at its own bench-tuned cell, 20 draws per point.
Width of the band that stays above 50%:

    design      platform height   along the tool   across it      yaw
    g23          -2 .. +4 mm       -8 .. +3 mm     -7 .. +8 mm    -12 .. +18 deg
    g24          -3 .. +4          -8 .. +2        -7 .. +8       -12 .. +18
    g12          -1.5 .. +4        -2 .. +4        -8 .. +8        -8 .. +2
    g14          -0.5 .. +4        -2 .. +2        -5 .. +2        -2 .. +0
    g01          -4 .. +4          -3 .. +1        -4 .. +7         0 .. +10
    rv04_mid     +0.5 .. +4        -8 .. -1        +3 .. +8        +2 .. +12

**g23 and g24 are far more tolerant than anything measured on the round-capsule model** — an
8 mm axial band and +-15 degrees of yaw, against +-1-2 mm axially before. The flat pad's line
contact along the shaft is what buys it.

**Platform height is one-sided.** Every design tolerates +4 mm better than -4 mm, because too low
is too little squeeze and too little grip, while too high is only more squeeze. Build the
platform 1-2 mm proud rather than shy.

**rv04_mid's plan is mis-registered.** It reads 40% at nominal and 100% from -8 to -1 mm along
the tool, so its basin centre sits about 4 mm off where the fitter put it. Shifting the grasp's
axial offset would recover it — the same effect section 4 found on the lift schedule, and a free
fix wherever the map shows an off-centre band.

## Verdict, 80 draws per point

    design      careful bench   everything wrong at once
    g23              64%                 16%
    g12              62%                 40%
    g24              49%                 21%
    rv04_mid         48%                 18%
    g01              45%                 12%
    g14              22%                  6%

**Build g23 or g24.** They are the same hand but for pair-Y separation (+-32.5 vs +-25 mm), both
run at straddle 40 mm, thumb offset 10 mm, pivot 0.05, turn -70 degrees, squeeze 4 mm, and both
carry an 8 mm axial and 30 degree yaw tolerance. g12 is the pick if the bench turns out worse
than expected — it degrades most gracefully.

Calibrate the servo zeros against a measured pose before anything else: 1 degree of per-joint
error costs about 30 points of success and no other knob on this list costs that much.

## 9. The fingertip: flat is not the question, WIDE is

The user's follow-up: only the fingertip dimensions really matter, and swapping a 14.8 mm tip for
a 21.1 mm one is a free reprint. Three tips, links left round, 504 matched cells each, on the
fixed-palm bench:

    tip                              cells that reorient+keep   best cell   paired vs 14.8 flat
    round sphere, r 10.55 (shipped)        21 / 504 ( 4%)         0.787     -0.021 +- 0.008
    flat pad, 14.8 mm across (built)       31 / 504 ( 6%)         0.913     --
    flat pad, 21.1 mm across (reprint)     37 / 504 ( 7%)         0.929     +0.021 +- 0.008

Paired on the identical 504 cells, **21.1 beats 14.8 by +0.021 +- 0.008 held cos (t = 2.75)** —
small, but resolved, and free. Neither flat pad separates from the round sphere in the pooled mean
(t = -1.22 and +0.86); what the flat pads do is reach higher best cells and work in more of them.
So: **print the wider tip**, and do not expect the round-vs-flat distinction to be the thing that
decides the day.

## 10. The flat pad needs a deeper squeeze, and that is worth more than the width

Because `fit_real_v1_pose` models the pad as a sphere, its commanded squeeze lands the flat face
2-5 mm short. Paying it back directly, 2,268 cells:

    squeeze   mean nom cos   kept   cells that reorient+keep   best cell
     4 mm        0.050        18%          19 / 756 ( 3%)        0.905
     7 mm        0.069        26%          63 / 756 ( 8%)        0.917
    10 mm        0.132        40%         113 / 756 (15%)        0.847

**Five times as many working cells.** This is a much larger effect than the pad width, and it
means every flat-pad result computed at the fitter's default 4 mm — including the section 8
shortlist — was measured in an under-gripped regime.

## 11. Confirmation, on draws that did not select the cell

Ranking 8,190 cells by their own 16-draw ensemble score selects partly on noise — the same error
as picking the 108-hand search's winners at n=3. Each design's best cell was therefore re-measured
against **200 fresh draws under a different RNG seed**:

    design      careful bench (level 0.5)   full error (level 1.0)   selected at
    g12            65% win / 84% kept          32% / 66%                 19%
    g23            64% / 76%                   28% / 41%                 38%
    g24            60% / 62%                   28% / 36%                 38%
    rv04_mid       46% / 47%                   26% / 31%                 50%
    g01            42% / 42%                   20% / 22%                 12%
    g14            26% / 50%                   10% / 24%                 19%

Regression to the mean is visible and in the expected direction: the three cells selected highest
(rv04_mid 50%, g23 and g24 38%) all came back lower, and g12 — selected at 19% — came back top.
**g12 is the best hand on the corrected geometry**, and it fails safe: it keeps the tool in 66-84%
of runs even when the turn does not complete.

## 12. Bench tolerances on the corrected hand

20 draws per point, at each design's own confirmed operating point. Band = the widest run staying
above 50%:

    design      platform height   along the tool   across it    yaw
    g23         -2 .. +4 mm       -8 .. +3 mm      -7 .. +8 mm  -12 .. +18 deg
    g24         -3 .. +4          -8 .. +2         -7 .. +8     -12 .. +18
    g12         -1.5 .. +4        -2 .. +4         -8 .. +8      -8 .. +2
    rv04_mid    +0.5 .. +4        -8 .. -1         +3 .. +8      +2 .. +12

**Platform height is the tolerance the bench adds** — with a fixed palm the platform height *is*
the grip depth, and nothing downstream corrects it. It is one-sided: up to 4 mm too HIGH is fine
on every design, 3 mm too LOW is not. Build the platform high and shim down.

Axial placement, the tight tolerance on the round-pad lift schedule (+-1-2 mm), is **10-11 mm wide**
on g23 and g24 here. Transverse is the full +-8 mm tested and yaw is 30 degrees. The tool can be
placed by hand.

There is a real trade in the shortlist: g12 wins most often at its centre but has the tightest
placement bands (6 mm axial, 10 deg yaw); g23 and g24 win slightly less often and tolerate far
more error.

## What to do on the bench

1. **Squeeze 10 mm, not 4.** Biggest single lever found today: 2.6% of cells work at 4 mm,
   17% at 10 mm.
2. **Print the 21.1 mm flat tip**, not the 14.8 mm one. Small (+0.021 +- 0.008 held cos), free.
3. **Build g12 first** -- thumb (-42.5, 0), pair (+42.5, +-40), straddle 40, thumb offset 20,
   pivot k 0.10, turn -70 deg. Then the g23 / g24 / rv04_mid setup — thumb at (-35, 0), pair at x = +35, y = +-32.5 (g23)
   or +-25 (g24); rv04_mid is the same setup at +-40, so one build covers three hands by sliding
   the pair mounts in Y. g12 needs both gantries 7.5 mm further out (thumb -42.5, pair +42.5).
4. **Support the tool under the half that RISES** (y = -35 mm), never under its middle: a centred
   post is struck by the descending end in 55% of runs.
5. **Platform 2 mm high rather than low**, tool placed by hand is fine.
6. Do NOT add a grip-force loop through the turn, and do NOT slow the turn down. Both measured
   negative.
7. Expect **roughly 3 in 5 at a careful bench and 1 in 3 with everything wrong**, on the turn.
   The grasp and lift are not the risk.

## 13. Final: the corrected hand, the corrected squeeze, re-searched and re-confirmed

3,120 cells at squeeze 10 mm with the 21.1 mm flat tip, fixed palm, support at y = -35 mm:
**522 cells reorient and keep the tool — 17%, against 2.6% at the fitter's 4 mm squeeze with the
built 14.8 mm tip.** The working region is six and a half times larger, which is the largest single
effect measured today and larger than any choice of hand.

Confirmed on 200 draws under a seed that did not select the cell:

    design      careful bench   kept   full error   operating point
    g12             70%          89%      27%       sp40 t20 k0.10 a-70
    g23             49%          63%      28%       sp40 t10 k0.10 a-80
    g24             44%          52%      22%       sp32 t10 k0.55 a-110
    rv04_mid        40%          56%      22%       sp32 t10 k0.45 a-80
    ax_tx+20        30%          68%      15%       sp40 t10 k0.65 a-80
    g14             26%          47%      20%       sp40 t20 k0.10 a-90
    g10             25%          73%      10%       sp40 t20 k0.45 a-110

**g12 is the hand**: 70% of runs stand the tool up at a careful bench and 89% keep hold of it, so
its common failure is a tool still in the hand rather than a tool on the floor.

A caveat on reading the per-design change against section 11: the search re-picked each design's
cell at the new squeeze, so a design whose number fell (g23 64 -> 49) has moved operating point as
well as squeeze, and the two cannot be separated from these runs. What is comparable, and what
matters, is the size of the working region and the best confirmed hand.

Build sheets at these points: `deploy/{g12,g23,g24,rv04_mid}_build.txt` with 50 Hz trajectories
beside them. g12 needs both gantries 7.5 mm further out than the other three (thumb x -42.5,
pair x +42.5); g23, g24 and rv04_mid share thumb (-35, 0) and pair x = +35 and differ only in
pair-Y, so one build covers three hands.

## 14. The hand the firmware can actually build

The simulator's design space and the driver's rails were written months apart from the same
drawing (`XY_space.png`) and had never been compared. They are now, by
`scripts/real_v1_hand_commands.py --audit`, and the comparison holds in three places and fails
in one.

**The frame is the identity map, and that is verified.** `REAL_V1_MOUNTS` gives thumb (-50, 0),
index (50, 55), middle (50, -55) mm; `kinematics.FINGER_GEOMETRY` gives the same three pairs as
the origins of {P}. Nothing rotates and nothing flips: a mount position in the sim, in mm, *is* a
{P} coordinate. From there `manta_hand.plan` translates into each finger's local frame and applies
the firmware's x/y swap and per-finger homing sign, which `kinematics.py` had already derived.

**The joint names are forced.** The scene's limits are yaw ±85, mcp [-15, +92], pip [-18, +92];
the servo contract is aa ±85, fe1 [-15, +92], fe2 [-18, +92]. Three distinct ranges, three exact
matches — there is no other assignment that fits. All four shortlisted trajectories also stay
inside the servos' *measured* ranges, which are tighter than the scene's in places (finger 1's
fe1 stops at +64.75° where the scene allows +92°).

**The travel envelope does not hold.** The sim declares a ±30 mm box in x for every finger. The
rails deliver thumb x_max +26.2, index x_min −26.0, middle x_min −24.1 — and the shortfall is on
exactly the axis the compactness knobs push against, since `real_v1_compact_design` drives the
thumb toward +x and the pair toward −x.

    finger  axis         declared        reachable          lost
    thumb   x    [ -30.0,  30.0] [-30.00, 26.20]    3.8 at max
    index   x    [ -30.0,  30.0] [-26.00, 30.00]    4.0 at min
    middle  x    [ -30.0,  30.0] [-24.10, 30.00]    5.9 at min

**97 of the 108 sampled designs still fit, and all four shortlisted hands are among them.** g12,
g23, g24 and rv04_mid sit at most 15 mm off centre on any axis, less than half the available
travel. The 11 that fail are the compact corner — `rv01_compact`, the whole `g4*` column,
`ax_px-30`, `ax_tx+30`, and three random draws that happened to land near it.

**And that shortfall may not be real.** It lands on firmware joints J1/J3/J5, whose
`STEPS_PER_MM` is, by its own comment, "back-calculated from a known-good 10mm move and hasn't
been individually ruler-checked" — while J0, the one axis that *is* ruler-verified, measures
112.4 mm against a 110 mm nominal, i.e. over rather than under. Re-run the audit assuming the
~59.8 mm these rails were previously reported to travel and the entire design box comes back
within 0.2 mm:

    uv run python scripts/real_v1_hand_commands.py --audit --travel 1:59.8,3:59.8,5:59.8

This is worth a caliper before it is worth anything else, because the two cases need opposite
responses and the scale case reaches further than the design set. If commanding 45 mm moves
45 mm, the rails really are short and those 11 designs are unbuildable. If commanding 45 mm moves
48–50 mm, the scale is 6–10% high, the missing travel is recovered — and **every mm this driver
has ever commanded on those three axes was short by that much, mount positions included**. Do not
widen `FULL_EXTENSION_MM` on the hunch: `MOVEMM` does not stall-check, so an optimistic number
drives an axis into its own hardstop unprotected. `examples/verify_frame_mapping.py --travel`
runs the probe.

**Two things the audit cannot settle, and one of them will cost a run.** Which physical gantry is
finger 1 versus finger 2 comes from the drawing, not from watching a block move. And the *sign* of
each aa joint is unmeasured: the flexion joints give themselves away (a [-15, +92]-shaped range is
only explicable as a little hyperextension and a lot of flexion, and the scene mirrors mcp/pip per
finger so positive is inward on all three), but aa's range is symmetric and carries no such
fingerprint — and the scene gives all three fingers the *same* yaw axis while mirroring mcp/pip,
which is not what a rigid 180°-rotated thumb module would do. g12 grips with the thumb at +17.7°
of yaw, so a flipped sign rolls that pad off the tool. `plan.HandPlan.run_trajectory` refuses to
run until someone passes `signs_checked=True`.

### Commanding a hand

    # the audit, and what a re-calibrated rail would buy
    uv run python scripts/real_v1_hand_commands.py --audit

    # a design as literal wire-protocol lines (docs/protocol.md) plus the servo set-points
    uv run python scripts/real_v1_hand_commands.py --plan deploy/g12_plan.json

    # on the CB1
    python3 -c "from manta_hand.plan import HandPlan; HandPlan.from_json('g12_plan.json').apply_mounts(hand)"

g12's six gantry targets, for the record: thumb J0=55.00 J1=37.50, index J2=45.00 J3=37.50,
middle J4=45.00 J5=37.50 mm from home. All three fingers happen to sit at local (±7.5, 0/±15).
