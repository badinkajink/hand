# The screen's "held" is a snapshot part-way through a fall

**2026-08-30, review of the Sobol-128 pilot.  Written after the user watched the six
renders and said the cylinder "gets reoriented but looks like it's about to fall
out."  That read is correct, and it is measurable.**

Reproduce: `python3 scripts/probe_hold_convergence.py --plans <this dir>/plans`.
Raw output: [`hold_convergence.json`](hold_convergence.json).
Renders tiled at 30/55/75/90/100 % of each clip:
[`videos/FILMSTRIP_six_designs.png`](videos/FILMSTRIP_six_designs.png).

## Every finalist drops the shaft. They differ only in how long they take.

The screen scores a rollout **1.6 s** after the turn command ends (`--hold-steps 800`
at dt = 0.002).  Re-running each finalist's own saved plan at longer holds, nothing
else changed:

| design | nominal cos @1.6 s | 1.6 s | 3.2 s | 4.8 s | 6.4 s | 8.0 s | 9.6 s |
|---|---:|:--:|:--:|:--:|:--:|:--:|:--:|
| sv1_w0099 | **0.955** | held | **drop** | drop | drop | drop | drop |
| sv1_w0116 | **0.918** | held | **drop** | drop | drop | drop | drop |
| g12 | 0.911 | held | **drop** | drop | drop | drop | drop |
| sv1_u0060 | 0.901 | held | **drop** | drop | drop | drop | drop |
| rv05_manual | 0.759 | held | held | **drop** | drop | drop | drop |
| **sv1_u0100** | **0.792** | held | held | held | **drop** | drop | drop |

"drop" means the tool is on the table: descent goes from 8–13 mm to **109–113 mm**,
and the bench is 100 mm high.  There is no intermediate state — it is on the fingers
or it is on the table.

**The four designs with the highest nominal cosine are the four that fall fastest,
and the design with the lowest nominal cosine survives longest.**

## Why the gate let it through

```python
"ok": bool(nh >= 1 and min_z > lifted["z"] - 0.02 and on_post == 0)
```

- `nh >= 1` — **one** fingertip contact counts as held.  12 of 179 scored wins end on
  a single contact.
- `min_z > lifted_z - 0.02` — a 20 mm descent allowance.  At the 1.6 s measuring
  instant the finalists have fallen **8–13 mm**: inside the allowance, and still
  going.  `sv1_u0100` at 6.4 s reads cos +0.804 and fails only because it has by then
  fallen 20.4 mm.
- The measurement instant is fixed rather than convergent, so nothing distinguishes a
  settled grasp from a slow fall.

Terminal grip force tells the same story from the other side.  Across the 179 scored
wins in [`robustness_ensemble.json`](robustness_ensemble.json) the **median total hand
force on the object is 0.25 N**.  The tool weighs 24.5 g = **0.24 N**.  The hand is
supplying about one object-weight in total, across every contact — that is a shaft
resting on fingers, not a grip.

## The turn and the fall are partly the same event

Over the 268 retained rollouts in the ensemble:

```
spearman(final_cos, descent) = +0.540
  cos >= 0.7  ->  median descent 9.0 mm
  cos <  0.7  ->  median descent 2.4 mm
```

The rollouts that align are the rollouts that fall.  This is the floor-assisted
reorient of `REORIENT_PRIMITIVE.txt` in a new costume: there the shaft got its
alignment from the table, here it gets it from swinging down out of the pinch.  The
`on_post` check catches the first version and not the second.

## Re-scoring the ensemble under a gate that means "held"

No re-simulation — the same 480 rollouts, stricter arithmetic.  Win % / kept %:

| design | as scored | +≥2 contacts | +force ≥ 1 weight | +descent ≤ 5 mm |
|---|---:|---:|---:|---:|
| rv05_manual | 62.5 / 86.2 | 62.5 / 86.2 | 48.8 / 71.2 | 3.8 / 15.0 |
| sv1_u0100 | 47.5 / 81.2 | 47.5 / 81.2 | 35.0 / 68.8 | 0.0 / 28.8 |
| sv1_u0060 | 32.5 / 70.0 | 31.2 / 68.8 | 17.5 / 48.8 | 1.2 / 28.8 |
| g12 | 30.0 / 31.2 | 22.5 / 23.8 | 7.5 / 8.8 | 2.5 / 3.8 |
| sv1_w0099 | 26.2 / 35.0 | 22.5 / 31.2 | 5.0 / 13.8 | 0.0 / 0.0 |
| sv1_w0116 | 25.0 / 31.2 | 22.5 / 28.8 | 3.8 / 8.8 | 1.2 / 6.2 |

The contacts requirement costs almost nothing; the force requirement costs a third;
requiring the tool to have stayed where it was put collapses everything to ~0–4 %.

## The built-in remedy does not fix it

`execute()` already carries a hold-phase force regulator, with a comment naming this
exact phase as "where the open-loop schedule is left holding 0.2 N and loses the tool
to any disturbance."  The pilot ran it at `force_target = 0.0`.  Turning it on:

| sv1_w0116, hold-phase regulator | 9.6 s | 19.2 s |
|---|:--:|:--:|
| 0 N (as run) | drop | drop |
| 1 N | **held, cos 0.874** | drop |
| 2 N | **held, cos 0.814** | drop |
| 3 N | **held, cos 0.816** | drop |
| 6 N | drop (ejected) | drop |

`sv1_w0116` is the **only** finalist the regulator can save, and only for ~10 s, and
only inside a 1–3 N window — 6 N squeezes the tool out, which is the same grip window
the hardware bench found.  Every other design drops at 9.6 s at every force setting.

## What this does and does not invalidate

**Still good.** The sampler, the 134 scenes, the grasp screen, the clearance tracing,
the gantry-reachability audit.  Graspability and clearance are measured correctly and
do not depend on the hold.

**Not supported.** Any statement of the form "design X reorients and holds."  No hand
in the finalist set holds the tool after the turn.

**Weaker than stated, but not wrong.** `sv1_u0100` really is the best of the six — it
survives 4× longer than the median finalist and it is the only one whose alignment is
not bought by falling.  What the pilot could not know is that "robust" here means
*slowest to fall*, not *holds*.  `sv1_w0116`'s 0.918 ± 0.001 is a measurement taken
1.6 s into a fall that completes by 3.2 s; its low variance is the repeatability of a
falling object, and it is the one design a hold controller can rescue.

## The fix, and it is nearly free

84 rollouts at holds up to 12.8 s took **2.6 s wall clock** on six workers.  A 10 s
hold would have cost the 960-cell sweep a few minutes.  There was never a compute
reason to measure at 1.6 s.

1. **Score at a convergent hold, not a fixed one.**  Run until |dz| over the last
   0.5 s is below a threshold, capped at ~10 s.  A settled grasp is invariant to the
   hold length; a fall is not.
2. **Tighten `ok`** to something that means held:
   ```python
   "ok": bool(nh >= 2 and foh >= 1.0 * OBJ_WEIGHT_N
              and min_z > lifted["z"] - 0.005 and on_post == 0)
   ```
3. **Report time-to-drop as the primary robustness number.**  It is continuous, so it
   is a far better search signal than a 0/1 gate — the whole finalist set is "0" under
   a correct gate but they span 3.2 s to 8.0 s, which is exactly the ordering a
   morphology search needs.
4. **Re-score, do not re-sample.**  Only the deploy-cell and robustness stages need
   re-running; the grasp screen stands.

Then the platform question the pilot was built to answer — *does open-loop
reorientation on this hand hold the tool at the end?* — gets a real answer instead of
one taken 1.6 s in.  On today's evidence the answer is **no for all six finalists**,
which is a result worth having: it says the missing piece is a hold controller, not a
better hand.
