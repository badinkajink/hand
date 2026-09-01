# Stage 1, first sitting — sv1_w6689, five trials

**2026-08-31, 18:20–18:28** (workstation clock; the CB1 stamps run ids ~8 h earlier).
One plan, `sv1_w6689`, loaded once and re-run — `design_table`
`docs/experiments/20260831-real_v1-sobol8192/selected/confirmed/selected_table.json`,
straddle 32.0 mm, object `medium`. Plus one earlier trial at 17:46 recorded before the
`--shaft-axis` sign was fixed.

This is the protocol's Stage 1: the sitting exists to measure the **instrument**, not the hand.
The question it has to answer before Stage 2 is worth running is whether two runs of the *same*
plan land close enough together for a difference between *different* plans to mean anything.

## What the five trials did

Re-scored under the corrected hold window (see below), so these numbers differ from what the
station displayed at the bench.

| run | vis | dur (s) | cos start | peak | **hold** | turn | z start→final (mm) | slip (mm) |
|---|---|---|---|---|---|---|---|---|
| `…-092657-…ab06a0` | 100% | 3.23 | +0.049 | +0.856 | +0.855 | +55.9° | 55.8 → −45.9 | 55.1 |
| `…-100100-…701905` | 100% | 3.29 | +0.054 | +0.826 | **+0.826** | +52.6° | 66.1 → 82.9 | 4.9 |
| `…-100313-…9c2fc4` | 100% | 3.23 | +0.069 | +0.849 | **+0.849** | +54.2° | 65.7 → 81.7 | 4.5 |
| `…-100420-…efa110` | 44% | 1.44 | +0.056 | +0.704 | *unobserved* | +39.5° | 66.6 → 45.8 | 14.9 |
| `…-100654-…ba9fe7` | 40% | 1.26 | +0.112 | +0.846 | *unobserved* | +51.4° | 64.7 → 78.8 | 3.9 |
| `…-100849-…3a55da` | 100% | 3.29 | +0.070 | +0.874 | **+0.874** | +56.9° | 65.5 → 76.4 | 6.9 |

The 17:46 trial is excluded on its own evidence: the centre reads −45.9 mm, below the bench
floor, which is not a pose. That is the inverted `--shaft-axis` the probe was later taught to
catch, and the summariser flags it in the trace itself.

## The headline: three replicates of one plan agree to sd 0.024

| | |
|---|---|
| bench `cos_hold`, mean of 3 | **+0.850** |
| bench replicate **sd** | **0.024**, 95% CI [0.012, 0.148] |
| simulated between-plan sd (16 plans) | 0.240 |
| ratio | **10×** |

Taken at face value this is the result Stage 1 was for: the spread between two runs of one hand
is a tenth of the spread the simulator predicts between hands, so the bench can in principle
resolve the ranking. `deg_turned` agrees to 2.2°, `slip_mm` to 1.3 mm, and `z_final` to 3.4 mm.

**Do not spend it yet.** An sd from n = 3 is barely an estimate — the upper end of its own
interval is 0.148, and at that value four trials per plan resolve a difference of 0.29 rather
than 0.05, which would make Stage 2 as designed underpowered against everything except the
extremes. Stage 1's full n = 10 is what settles which end of that interval we are on, and it
should be finished before the 16-plan sweep starts.

## The finding that mattered more: two of five endings were never observed

Both `efa110` and `ba9fe7` lost the cylinder tag partway through — at 1.5 s and 1.3 s of a
3.3 s trajectory — and never got it back. Three things follow.

**The hold window was measured from the wrong end.** It was anchored to the last *detection*,
so on these two runs it slid backwards into the middle of the turn and reported +0.359 and
+0.386 as "the pose the hand ended holding", while the shaft was in fact still rotating and its
last seen alignment was +0.678 and +0.846. Fixed: the window is anchored to when *recording*
stopped, and a window containing no detections yields `cos_hold = None` — a void, explicitly,
rather than a plausible-looking low number. This is the failure mode the metric was introduced
to prevent, reappearing one level down.

**There was no hold to measure in the first place.** Every trace ends at 3.2–3.3 s, which is
when the last ramp finishes: the station stopped the camera the instant the motion did. So
"the last 1.0 s of the hold" was the last second of the *turn* on all five trials, and a hand
that reached vertical and let go a moment later would have scored exactly like one that reached
vertical and kept it. The service now keeps recording `--post-roll 2.5` s past the end of the
trajectory. The servos hold their last commanded position, so the dwell costs only the seconds
— and it is the difference between measuring a turn and measuring a grasp.

**A void rule with no bookkeeping deletes failures.** The tag vanishes abruptly — decision
margin 63–78 right up to the last frame, no decay — while `range_mm` climbs 30–40 mm in the
last 270 ms, i.e. the tool is moving away from the camera fast. On `efa110` the centre also
falls 78.6 → 45.8 mm in three frames, which is close to free fall. That is what a tool being
flung out of view looks like, and a study that silently drops those trials removes failures
from exactly the plans that fail. The analysis now attributes every exclusion to its plan and
flags any plan whose exclusions outnumber its scored trials.

Both lost trials used the same plan as the three clean ones, so the 2-in-5 rate is a property
of this staging, not a difference between hands.

## What is now recorded that was not

- `u_px, v_px` — where the tag sat in the image. Nothing in the trace could distinguish an
  occluded tool from one that swung out of frame; that was the open question here.
- **the tape** — one IR frame every 0.25 s, ~1 MB a run, rendered by
  `scripts/real_v1_tracker_replay.py` into an annotated mp4 and a phase-aligned filmstrip.
  Verified end to end on a live arm/stop; 31 frames over 8 s, none dropped.

Re-run the three clean trials with the tape on before Stage 1 continues, and the question of
what happened at 1.3 s stops being an inference from `range_mm`.
