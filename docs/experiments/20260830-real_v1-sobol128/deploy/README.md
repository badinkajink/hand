# Clearing the Sobol-128 finalists onto hardware (2026-08-30)

> **Uniform budget-1.0 follow-up:** all four Sobol finalists were re-exported and passed both
> trajectory-clearance paths, but none passed the combined driver + repeated aligned-hold gates.
> No `_b100` plan was added to the runnable station set. See
> [`../deploy_b100/README.md`](../deploy_b100/README.md). The per-design tuned plans below are a
> separate result and remain untested on hardware.

The pilot ([`../REPORT.md`](../REPORT.md)) ranked 134 hands but stopped short of exporting any of
them, and its own ranking was measured at a 1.6 s hold and a 0.5 rad residual clip -- both of
which [`../HOLD_REVIEW.md`](../HOLD_REVIEW.md) showed to be wrong. This is the promotion pass:
re-tune, re-score, run the two hardware gates, ship what survives.

## What ships

| plan | design | budget | held cos @9.6 s | clearance (chord/csv) | where the minimum is |
|---|---|---:|---:|---:|---|
| `sv1_u0060_b75` | sv1_u0060 | 0.75 rad | 0.723 at the band edge | +9.9 / +9.9 mm | thumb_tip <-> index_tip |
| `rv05_manual_b85` | rv05_manual | 0.85 rad | 0.638 | +10.3 / +10.3 mm | index_pip <-> middle_pip |
| `sv1_u0100_b70` | sv1_u0100 | 0.70 rad | 0.541 | +7.5 / +7.5 mm | index_mcp <-> middle_pip |

Two designs did not make it, for two different reasons -- see "What does not ship".

The `_bNN` suffix is the per-joint residual clip in centiradians. `g12w11` in the shipped deploy
folder is the same idea under the older spelling.

## The budget band, and why the shipped budget is not the best one

Sweeping the clip from 0.40 to 2.00 rad on each finalist's own saved plan (4 reps, 9.6 s hold,
`scripts/probe_hold_convergence.py`) turns up a structure the pilot's single 0.5 rad point could
not see. **Every design has a contiguous band of clip values inside which it keeps the tool, and
drops it on both sides.**

```
             .40 .45 .50 .55 .60 .65 .70 .75 .80 .85 .90 .95 1.0 1.05 1.1 ... 2.00      (reps kept /4)
g12          .   .   .   .   .   .   .   .   .   .   .   2   4   4   4   4   4   4   4   4   4   4   4   .   .   .
rv05_manual  4   .   .   .   .   .   .   4   4   4   4   4   4   4   4   4   4   4   3   3   .   .   .   .   .   .
sv1_u0060    .   .   .   .   3   4   4   4   4   4   4   4   4   4   4   4   4   4   4   4   4   4   4   4   4   4
sv1_u0100    .   .   .   .   4   4   4   4   4   4   4   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
sv1_w0099    4   1   .   .   .   .   .   .   .   2   2   4   3   2   .   2   .   1   1   2   .   .   .   .   .   .
sv1_w0116    .   .   .   .   .   .   .   .   .   .   .   .   2   .   .   .   .   .   .   .   2   4   4   4   4   4
```

Both edges are mechanical, not statistical. Below the band the clipped trajectory stops while the
shaft is still rotating under gravity, and the fingers -- which pay for the turn in extension --
run out of commanded travel and let go. Above it the turn overdrives and ejects. Inside the band
held cosine falls monotonically with the clip, so the best alignment sits at the LOWER edge:

| design | band (rad) | band (deg) | best cos, at |
|---|---|---|---|
| sv1_u0060 | 0.65 - 2.00 | 37 - 115 | **0.723** at 0.65 |
| g12 | 1.00 - 1.50 | 57 - 86 | 0.695 at 1.00 |
| rv05_manual | 0.75 - 1.25 | 43 - 72 | 0.638 at 0.75 |
| sv1_w0116 | 1.45 - 2.00 | 83 - 115 | 0.633 at 1.45 |
| sv1_u0100 | 0.60 - 0.90 | 34 - 52 | 0.541 at 0.60 |
| sv1_w0099 | none | | 0.148 |

**The shipped budget is one step ABOVE each band's lower edge, on purpose.** The first bench run
(2026-08-29) measured the yaw joints arriving 4-6 deg short under load, and the plan assumes
commanded == achieved. A commanded clip of `edge + 0.10 rad` is expected to be ACHIEVED at the
edge, where the alignment is best. That is an approximation in two ways worth stating: the droop
was measured on one design's yaw joints under one load, and the clip binds on whichever joint
wants the most travel (middle yaw and middle pip on every design here), so it is only the droop of
that joint that matters. If a bench run under-turns, the fix is to raise the clip, not lower it.

This also retro-explains the bench: shipped `g12` at 0.5 rad is far below g12's band and drops;
`g12w11` at 1.1 rad lands just inside it, which is the "works pretty well" the operator saw.

## What does not ship

**`sv1_w0116` -- fails the servo gate at the GRIP, not the turn.** Its grasp anchor puts middle
yaw at -84.16 deg against a +-70 deg command cap, and that number is identical at every budget
from 1.20 to 1.55, so no clip setting rescues it. Its hold band (1.45 - 2.00 rad) additionally
starts above the middle-pip ceiling: the largest clip that validates is about 1.30 rad, which is
below the band, so even with the yaw cap lifted to the +-85 deg mechanical limit the design has no
setting that both holds and is commandable. The pilot's nominal winner (0.918 held cosine, 11.66 mm
clearance) is not buildable on this machine.

**`sv1_w0099` -- no band.** Its kept counts never form a contiguous run; 0.40 and 0.95 are isolated
4/4 spikes with 0/4 and 2/4 immediately either side. The pilot already called it "noisy and
brittle" from the ensemble; the band scan says the same thing in a way that cannot be tuned around.

## The pilot's robustness ranking inverts

`sv1_u0100` was the pilot's "strongest new robust hand" (45% full-error wins vs g12's 25%). Under
a convergent hold it has the **narrowest** band of any design that holds at all (0.60 - 0.90 rad,
0.30 wide, against sv1_u0060's 1.35) and the lowest held cosine of the three that ship. Its
robustness was measured against pose error at a fixed clip; the clip itself is the more sensitive
axis, and it was held constant across the whole pilot.

`sv1_u0060` -- the pilot's third pick, "balanced nominal/clearance candidate" -- wins on both
counts here: the highest held cosine and a band so wide it never drops above 0.65 rad.

## Gates run

1. **Export parity.** Re-exported at budget 0.5 and diffed against the pilot's own saved plans:
   `grip_ctrl` and `open_qpos` match to 0.00e+00 on all six, same scene file. The thing being
   shipped is the thing that was scored.
2. **Gantry travel.** All six mount triples are inside the measured `palm_envelope`; firmware
   targets range J0 52.7 - 100.8 mm. Nothing needs a re-calibration to reach.
3. **Servo range** (`HandPlan.validate`). 0 schema errors and 0 violations on the three that ship;
   2 violations on `sv1_w0116` as above.
4. **Finger-finger clearance along the trajectory** (`real_v1_trajectory_clearance.py`, 8 substeps,
   both the plan's 3-set-point chord and the exported CSV). All three pass >= 5 mm.

`real_v1_trajectory_clearance.py` now names the two BODIES at the minimum rather than the two
fingers. That matters here: the sim's proximal and mid links are 21.2 mm capsules on the joint
axes, and the printed parts put a servo body and bracket around exactly those. A margin held
between two tips is worth more than the same margin between two mid-links, and `sv1_u0100`'s
+7.5 mm is both the tightest of the three and the one at `index_mcp <-> middle_pip` -- the
worst-modelled place for it to be. Watch that pair on its first run.

## Still not closed

The pilot's remaining caveat -- servo body and mount side profiles are absent from the MJCF -- is
narrowed but not closed. Gate 4 now tells you WHERE the margin is, so a fatter real link can be
reasoned about, but nothing measures the housings. The cheap way to close it is a dry mount move:
with the fingers at the open pose, drive the gantries to a design's mount coordinates from the
station's Manual control tab and look. That costs nothing and needs no plan.

The operating cell (`axis_k`, `angle_deg`, straddle, thumb axial) is still the one the pilot chose
at 0.5 rad. Only the clip was re-tuned. A joint cell x budget sweep is the obvious next thing and
was not run.

## Reproduce

```bash
# the band scan
for B in $(python3 -c "print(' '.join(f'{0.40+0.05*i:.2f}' for i in range(33)))"); do
  python3 scripts/probe_hold_convergence.py \
    --plans docs/experiments/20260830-real_v1-sobol128/plans \
    --holds 4800 --reps 4 --budget $B --out band/b_$B.json
done

# one export (cell values from ../selected_safe_cells.json, scene from ../plans/<design>.json)
python3 scripts/real_v1_export_plan.py --design sv1_u0060 --object medium \
  --design-table docs/experiments/20260830-real_v1-sobol128/pilot_table.json \
  --scene <the pilot's scene> --straddle-mm 40 --thumb-axial-mm 10 --squeeze-mm 10 \
  --axis-k 0.15 --angle-deg -90 --turn-steps 550 --hold-squeeze-mm 0 \
  --bench-height 100 --post-y -35 --flat-pads --pad-width-mm 21.1 \
  --budget 0.75 --out <dir>

# the gates
python3 scripts/real_v1_trajectory_clearance.py --deploy-dir <dir> --plan <tag> --substeps 8
```

## Two metadata bugs found while diffing, both fixed, neither behavioural

Re-exporting `g12` from the default table and diffing against the shipped plan turned up
`bench_height_mm: 100000.0` on `g12w08` and `g12w11`. `--bench-height` is documented in METRES
and every recorded invocation in this repo passes `100`; nothing downstream reads the number (the
flag is used as a boolean for the no-palm-ramp branch, and for meta), so it stayed invisible.
`g12w08`/`g12w11` also recorded `flat_pads: false` while running against the flat-pad scene,
because the scene was passed with `--scene` and the pad flags only feed scene construction.

Both plans were re-exported; `git diff` on the trajectory CSVs is empty and only the three meta
keys moved, so no commanded angle changed. `--bench-height` now reads a value above 1 as
millimetres and prints that it did, rather than multiplying it by another 1000.

## The other six plans on the station

Only six of the nine plans the station now serves are clearance-safe. `g23` (+0.8 mm), `g24`
(-5.3 mm) and `rv04_mid` (-2.7 mm) still fail `real_v1_trajectory_clearance.py` -- the 2026-08-29
finding that three of four deployed designs interpenetrate in the sim's own geometry. They are
left on the station because they are the historical baselines, not because they are runnable.
