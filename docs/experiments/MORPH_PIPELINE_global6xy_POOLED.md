# global6xy — pooled replica table (6-DIM XY-ONLY sweep, DONE incl. n=4 confirm 2026-07-22 20:54, 28 records)

**Design:** like `global12x2` but the three proximal-phalange **lengths are FROZEN at m05**
(0.0108 / 0.0123 / 0.0159 m) — only the **6 XY placement dims** are LHS-sampled. Per design: two
independent full-pipeline draws (CEM → native A best-of-3 → imit-B live-A reset → continuous
handoff + trajectory-health scorecard), plus an **n=4 confirm** on the two n=2 standouts
(H06_04, H06_08). `--b-recipe imit`, `--a-attempts 3`.

Reference band, m05 clean draws under the same evaluator family: **{0.82 (a10→b33), 0.49, −0.16}
(mean ≈ 0.38)**. Judge on the deterministic **held-cos_tail** + the health verdict, never reward
sums. "—" = leg produced no policy (A never lifted). Verdict per leg: WARN = passes the gate
(reorient counts), FAIL = a health check tripped (jitter / drop / phantom-1-finger), PASS = clean
gate (none this sweep).

| design | draws (cos, verdict) | mean | max | express ≥0.5 | pass-gate | A-abort/legs | note |
|---|---|---|---|---|---|---|---|
| **H06_04** | 0.852 W / 0.741 W / 0.808 **F** / 0.593 W | **0.748** (n=4) | 0.852 | **4/4** | **3/4** | 4/9 | **PROGRAM-STRONGEST — CONFIRMED. Every draw drives the screwdriver near-vertical (all ≥0.59, peak ≥0.94). 3/4 pass the gate; the one FAIL (r2) is jitter-only (ang-jerk 159). Clears the ≥0.5 bar AND replicates.** |
| H06_11 | 0.054 F / 0.671 W | 0.363 | 0.671 | 1/2 | 1/2 | 5/6 | draw-gated expresser (clean 3-finger on r1, phantom-static on r0) |
| H06_08 | 0.204 W / 0.78 **F** / −0.012 W / 0.414 **F** | 0.346 (n=4) | 0.780 | 1/4 | 2/4 | 7/9 | **confirm FAILED the bar** (n=2 looked 0.492) — the 0.78 (r1) is a phantom **1-finger** (force 0.3 N, gate-rejected); r2 static, r3 a drop (minZ 0.007, drift 10.5 cm). Wide-band, not promotable |
| H06_00 | −0.044 W / 0.611 W | 0.283 | 0.611 | 1/2 | 2/2 | 0/3 | static→reorienter flip; zero A-abort |
| H06_03 | 0.006 W / 0.556 W | 0.281 | 0.556 | 1/2 | 2/2 | 3/6 | static→reorienter flip |
| H06_01 | −0.048 W / 0.51 W | 0.231 | 0.510 | 1/2 | 2/2 | 2/4 | static→partial flip |
| H06_05 | 0.34 **F** / 0.092 W | 0.216 | 0.340 | 0/2 | 1/2 | 0/2 | jitter-FAIL partial then static |
| H06_07 | 0.547 W / −0.268 **F** | 0.140 | 0.547 | 1/2 | 1/2 | 1/6 | expresser draw + a drop draw (r1 jerk 150, minZ 0.045) |
| H06_09 | −0.282 W / 0.356 **F** | 0.037 | 0.356 | 0/2 | 1/2 | 3/5 | static + genuine-3-finger-but-jitters (r1 cos 0.356, jerk 44) |
| H06_06 | — / 0.055 W | 0.055 | 0.055 | 0/1 | 1/1 | 5/6 | **clearest LIFT-HOSTILE design** — A never lifted on r0 (best objheight 0.0, 3 attempts); r1 barely lifted then static |
| H06_10 | −0.274 **F** / 0.223 W | −0.026 | 0.223 | 0/2 | 1/2 | 0/2 | jitter-FAIL then static |
| H06_02 | −0.119 W / −0.091 W | −0.105 | −0.091 | 0/2 | 2/2 | 0/2 | replica-consistent STATIC (the floor design) |

## Confirm close-out (n=4 on both n=2 candidates)

- **H06_04 REPLICATES and clears the bar.** n=2 was {0.852, 0.741} mean 0.796; the two confirm draws
  landed {0.808 (FAIL-jitter), 0.593 (WARN)} → **n=4 mean 0.748**, and **all four draws drive the
  object substantially toward vertical** (min 0.593, three ≥ 0.74, peak ≥ 0.94). 3/4 clear the full
  health gate; the r2 FAIL reoriented and then shook (ang-jerk 159). This is the **opposite** of the
  9-dim standout G02_00, which *regressed* 0.57→0.48 by n=4. H06_04 is the program's first design
  whose reorient mean is both **high (0.748)** and **stable across independent draws**.
- **H06_08 FAILS the confirm.** n=2 looked promising (0.492) but rested on the 0.78 (r1) — the confirm
  exposed that as a phantom **1-finger** pinch (force 0.3 N, gate-rejected), and the two new draws were
  a static (−0.012) and a **drop** (0.414 but minZ 0.007, drift 10.5 cm). n=4 mean **0.346**, only
  2/4 pass the gate. Wide-band, **not promotable** — the same "0.78 was never real" trap the health
  scorecard exists to catch.

## 6-DIM (XY-only) vs 9-DIM (global12x2) — apples-to-apples

Both sweeps: 12 designs, 28 records, same evaluator family, same `--a-attempts 3` + imit-B.

| statistic | 9-dim (global12x2) | 6-dim XY-only (global6xy) | reading |
|---|---|---|---|
| leg-level cos **floor** | −0.499 | **−0.282** | freezing len removes the worst legs |
| leg-level cos **peak** | 0.887 (G02_05_r1) | 0.852 (H06_04_r0) | ~tied best single policy |
| leg-level **median** | 0.149 | **0.223** | 6-dim landscape sits higher |
| leg-level **mean** | 0.232 | 0.269 | slightly higher |
| **leg-level sd** | **0.339** | **0.354** | **UNMOVED — the per-draw variance wall is exactly as high** |
| per-design MEAN floor | −0.388 | **−0.105** | narrows upward at the bottom |
| per-design MEAN peak | 0.482 (G02_00 @n=4) | **0.748 (H06_04 @n=4)** | **6-dim produces a genuine standout that HOLDS at n=4** |
| per-design MEAN median | 0.274 | 0.224 | ~same |
| **designs with mean ≥ 0.5** | **0** | **1 (H06_04)** | **first design to clear the promotion bar AND replicate** |
| A-abort rate | 40% (18/45) | **50% (30/60)** | **NOT lower with len frozen — length is not a special A-collapse source** |
| never-lifted legs | 3 | 1 | pick-up marginally more reliable |

## Headline conclusions

1. **Freezing the phalange lengths NARROWS THE LANDSCAPE UPWARD** — the per-design mean floor rises
   (−0.39 → −0.11), the leg-level median rises (0.15 → 0.22), never-lifted legs drop (3 → 1). The
   6-dim XY box is a "nicer" place to search: fewer catastrophic designs, a higher typical outcome.
2. **But the per-draw variance WALL is untouched** (sd 0.354 vs 0.339). The A-draw noise that made
   the 9-dim search unable to rank designs is *exactly as high* in the XY-only box. And **A-collapse
   is not lower** (50% vs 40%) — so **length is NOT a special noise source**; freezing it buys a
   better-behaved *mean surface*, not a quieter *evaluator*.
3. **H06_04 is the program's first genuine standout.** At n=4 its mean is **0.748** — roughly double
   the m05 band mean (≈0.38) and above m05's best single draw (0.82 ≈ H06_04's 0.852 peak). It wins
   by having a higher **true mean**, not by reducing draw noise — which is exactly the regime where
   replication *can* rank a design.
4. **H06_04 geometry** (deltas from m05): thumb +8.4 mm x / **−22.2 mm y** (large reposition toward
   opposition), index +11.1 mm x / +12.0 mm y (spread out), middle −5.7/−6.7 mm (small inward). The
   thumb-toward-opposition + index-spread pattern is the mechanistic story the morphology program
   set out to find (thumb opposition for a genuine 3-finger reorient). Lengths held at m05.
5. **H06_06 is the clearest lift-hostile design** (thumb −27 mm x, middle −30.6 mm y — a splayed,
   under-palm config): A never lifted on r0 and only barely on r1. A useful negative-space datum.

## Verdict (analysis only — promotion is the user's decision)

The XY-only sweep delivered what the 9-dim sweep could not: **one design (H06_04) whose reorient
mean clears the ≥0.5 bar and replicates across 4 independent draws.** It is a real candidate to
promote as a co-designed reference alongside/over m05 (a10→b33). It does **not** resolve the core
program finding — the per-draw A-variance wall stands, so a *broad* design search is still
gate-limited — but it is the first concrete design win, and it strengthens the case that the
**morphology-conditioned policy** (learn P(express | design) directly, amortizing the A-draw) is
the right next lever. **Do NOT auto-launch promotion or the conditioned-policy build — user call.**
