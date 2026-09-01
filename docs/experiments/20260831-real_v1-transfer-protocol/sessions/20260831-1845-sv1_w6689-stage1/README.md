# Stage 1 — repeatability on sv1_w6689, 12 runs, 2026-08-31 18:26–18:57

Twelve consecutive open-loop reorientations on one plan (`sv1_w6689_b060`), full re-stage
between each, tape on at 4 Hz, tracker post-roll 2.5 s. Two runs were staged outside the
protocol by the operator and are excluded; one of those (`374e49`) is visibly a drop — the
last tape frame shows an empty gripper.

## The headline

Seven runs carry `cos_hold`:

| run | cos_hold | cos_peak | deg turned | visibility |
|---|---|---|---|---|
| c8b6aa | +0.872 | +0.872 | 55.4 | 1.00 |
| 08f327 | +0.893 | +0.894 | 57.1 | 1.00 |
| b968dd | +0.880 | +0.881 | 55.7 | 1.00 |
| 510c4a | +0.896 | +0.898 | 58.5 | 1.00 |
| b64af8 | +0.868 | +0.869 | 54.4 | 1.00 |
| b80ead | +0.885 | +0.885 | 57.8 | 1.00 |
| f02a18 | +0.871 | +0.871 | 56.0 | 1.00 |

**mean +0.8806, sd 0.0110**, 95% CI on the sd `[0.0071, 0.0243]`.

Against the simulated between-plan sd of 0.240 that is a 22:1 ratio. At n = 4 trials per plan
Stage 2 resolves a design difference of 0.022; even at the pessimistic end of the CI it
resolves 0.048. **Stage 2's shape is confirmed as designed** — 16 plans x 4 trials.

`|cos_hold − cos_final| <= 0.0013` across all seven, so the 2.5 s post-roll dwell changes
nothing: the hand is genuinely static once the pose loop ends, and the value measured during
the turn's last second and the value measured 2.5 s later are the same number. The post-roll
is cheap insurance, not a correction.

## The defect Stage 1 actually found: the vane tag goes dark on ~40% of runs

Five of twelve traces (`069f33 37c996 c801b6 374e49 d6dd81`) lose the cylinder tag at
1.4–2.4 s into a 5.8 s recording and never regain it, so they have no `cos_hold` at all.
This is not the tool leaving: `z_bench` at the last detection is 71–79 mm, the same as the
seven complete runs (76–83 mm), and the tape shows four of the five still holding the tool
in the final frame.

What it is:

- The **reference tag (id6, 40 mm, static, frontal) decodes in every frame of every run**,
  margin 60–61, including all the frames where the cylinder tag is missing. Camera, exposure,
  IR emitter and detector are all healthy throughout.
- The **cylinder tag (id0, 30 mm, on a vane that swings as the tool turns)** is the only thing
  that fails, and it fails as a cliff, not a decay — decision margin is 71–86 on the last
  frame it is seen, then nothing.

The obvious lever is tag size: the 40 mm tag survives what the 30 mm tag does not, in the same
frames. Reprint the vane at 40 mm before Stage 2.

Do not diagnose this from the tape's own pixels. The tape is written at half scale, where the
cylinder tag falls below the decode threshold in the *complete* runs too — offline
re-detection on tape frames reproduces the failure on good and bad runs alike and is
therefore not evidence about the live stream.

## Two consequences for the protocol

1. **The tape is load-bearing, not a convenience.** `374e49` dropped the tool *after* its
   trace went dark, and the trace cannot show that — `z_drop` reads 0.0 mm. Only the last
   tape frame distinguishes "held" from "dropped" in a voided run. Never score a void run's
   outcome from the CSV alone.
2. **The void rule is not outcome-neutral.** `c801b6` reached +0.978, the highest alignment of
   the twelve, and is a void. Silently dropping voids removes the tails of the distribution.
   Voids must be reported per plan with a count, which `real_v1_transfer_study.py` now does.
