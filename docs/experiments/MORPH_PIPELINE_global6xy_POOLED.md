# global6xy — pooled replica table (24/24 complete 2026-07-22 14:50)

The 6-dim **XY-only** sweep: LHS over the 6 finger-placement dims (x,y per finger) with the three
**proximal-phalange lengths FROZEN at m05** (0.0108 / 0.0123 / 0.0159). Two independent full-pipeline
draws/design (CEM → A best-of-**3** → imit-B → handoff eval). Per H2, n=2 verdicts are binned, not
ranked. Reference band, m05 clean draws under the same evaluator family: {0.82 (a10→b33), 0.49,
−0.16} (mean ≈ 0.38). Quick-fix analysis + why no cheap A-selector: `docs/notes/policy_bottleneck_quickfixes.md`.

| design | r0 cos | r1 cos | **mean** | max | A-aborts/legs | note |
|---|---|---|---|---|---|---|
| **H06_04** | +0.852 | +0.741 | **+0.796** | +0.852 | 2/5 | **program-strongest — expresses HIGH on BOTH draws**; candidate for n=4 confirm |
| H06_08 | +0.204 | +0.780 | **+0.492** | +0.780 | 3/4 | wide (static + strong) — second candidate |
| H06_11 | +0.054 | +0.671 | +0.363 | +0.671 | 5/6 | expresses in r1 but **A-fragile** (5/6 legs abort) |
| H06_00 | −0.044 | +0.611 | +0.283 | +0.611 | 0/3 | static→express; clean trainability |
| H06_03 | +0.006 | +0.556 | +0.281 | +0.556 | 3/6 | static→express |
| H06_01 | −0.048 | +0.510 | +0.231 | +0.510 | 2/4 | static→express |
| H06_05 | +0.340 | +0.092 | +0.216 | +0.340 | 0/2 | partial + static; clean trainability |
| H06_07 | +0.547 | −0.268 | +0.140 | +0.547 | 1/6 | express + wrong-way (widest swing) |
| H06_09 | −0.282 | +0.356 | +0.037 | +0.356 | 3/5 | wrong-way → express |
| H06_10 | −0.274 | +0.223 | −0.026 | +0.223 | 0/2 | wrong-way + partial |
| H06_02 | −0.119 | −0.091 | −0.105 | −0.091 | 0/2 | **consistently static** (both draws agree) — genuine low design |
| H06_06 | MISS | +0.055 | +0.055 | +0.055 | 5/6 | **lift-hostile candidate** (r0: all 3 A collapse to objheight 0.0; r1 barely lifts; 5/6 legs abort) |

Pooled: design-mean spread **−0.105 … +0.796**, per-draw expression rate 8/23 ≈ 35%, held on every
leg that produced a policy (min-z ≥ 0.103; the only non-hold is H06_06's r0 lift-miss). A-leg abort
rate 24/49 ≈ **49%**.

## 6-dim (XY-only) vs 9-dim (full box, global12x2)

| statistic | 6-dim XY-only | 9-dim full |
|---|---|---|
| design-mean **floor** (min) | **−0.105** | −0.388 |
| design-mean **median** | +0.224 | +0.274 |
| design-mean **peak** (max) | **+0.796** | +0.482 |
| designs with mean ≥ 0.5 | **1** (H06_04) | 0 |
| pick-up/hold | solved (11/12; 1 lift-miss) | solved |
| per-draw reorient sd | ~0.33 (unchanged wall) | ~0.3–0.5 |
| A-leg abort rate | ~49% | ~47% |

**Reading — freezing the proximal length narrows the landscape UPWARD, but does not touch the wall.**
1. **Higher floor + higher peak, same median.** Removing the length axis eliminated the worst
   designs (no −0.39-class point; 9-dim's floor came from bad length combos) *and* surfaced a peak
   (H06_04 0.796) above any 9-dim design mean — while the typical design is unchanged (median ≈ 0.22
   vs 0.27). Consistent with the length dimension contributing mostly *downside* variance: the good
   reorienters live in the XY placement, and constraining to it concentrates the box on the better
   region. (Caveat: n=2 design means have SEM ≈ 0.2; the floor/peak shift is suggestive, the median
   equality is the robust part.)
2. **The per-draw expression wall is UNCHANGED.** Reorient is still draw-gated (sd 0.33, 8/23 draws
   express, several designs static-in-one-draw/express-in-the-other), and A-training is still
   ~49% fragile with collapse propensity clustering by design (H06_06/H06_11 = 5/6 aborts vs
   H06_02/05/10 = 0/2). So the *bottleneck* is identical; freezing len changed the *terrain*, not
   the measurement noise. This is exactly what the quick-fix analysis predicted (no cheap variance
   fix; the length axis was not a special noise source — confirmed here: 49% ≈ 47%).

## Verdict + next

- **No promotion yet.** H06_04 (mean 0.796, both draws high) is the **program's strongest candidate
  by a clear margin** — but n=2, and 9-dim's G02_00 looked comparably good at n=2 (0.570) before a
  static draw pulled it to 0.482 at n=4. Same discipline ⇒ **H06_04 (+ H06_08) get an n=4 confirm**
  before any claim. m05 (a10→b33) remains the reference until a confirm clears the bar.
- **H06_06** is the program's clearest **lift-hostile geometry** candidate (worse than 9-dim's
  G02_11 — both replicas struggled, 5/6 A-legs abort). A genuine "trainability-hostile design" data
  point on the trainability axis.
- The structural conclusion is unchanged and reinforced: the landscape is **draw-gated, not
  geometry-flat**; the morphology-conditioned policy remains the real fix. Freezing len is a mild
  terrain improvement, not a bottleneck fix.
