# Policy-learning bottleneck: decomposition + quick fixes (2026-07-20)

User ask (2026-07-20, heading out overnight): run a sweep like global12x2 but **freeze the
proximal-phalange lengths** → morphology exploration over the **6 XY placement dims only**; AND
"analyse our past results and policy performance and see if there's any quick fixes we can apply
now to make policy learning — which we established as the real bottleneck limiting understanding
of morphological limitations — more robust and reaching its potential."

This note is the analysis + the fix I validated + what I launched. Companion runbook section:
`docs/rl/morph_sweep_STATUS.md`; full arc: `docs/rl/reorientation.md`.

## 1. What the bottleneck actually is (decomposed from the probe suite)

The co-design landscape is unresolvable **not** because morphology doesn't matter, but because the
per-design evaluator (from-scratch A→B, judged on deterministic held-cos) has a per-draw noise
floor that sits right where the design effects live. Decomposed:

| term | magnitude | status |
|---|---|---|
| **B-side seed noise** (fixed A, vary B seed) | sd **0.09** self / **0.02** imit | **SOLVED** — imit prior (object-frame fingertip demo) is design-neutral, anneals out, sd ±0.02 |
| **A-draw noise** (which delivered grip you get) | sd **0.3–0.5** | **DOMINANT, open** — even m05's clean draws span {0.82, 0.49, −0.16} |
| **A collapse** (mid-training PPO collapse) | ~**47%** of legs abort | reactively patched by best-of-N + watchdog salvage; lossy/expensive |

Two findings sharpen *why* the A-draw term is hard (reorientation.md §P2 avar):

1. **The A trajectory-health GATE cannot see the variance that matters.** m05 draws k0 and k1 are
   gate-identical (same kept ckpt index, objheight 0.1176 vs 0.1175, both WARN) yet deliver states
   that send the SAME imit-B recipe to cos **0.49 vs −0.16**. "Best-of-N by the gate buys collapse
   insurance only; it cannot select for downstream reorientability."
2. **The gate is worse than uninformative — it mildly ANTI-orders outcomes.** Within G02_00, the two
   health-**FAIL** As produced its two **best** B draws (0.635, 0.681) while the WARN As produced
   0.504/0.107. Mechanistically plausible: a firm textbook 3-finger clamp (health-PASS) is *harder*
   to roll (→ static), whereas a slightly looser/mobile grip (health-FAIL on over-clamp/idle-finger)
   is *more* reorientable. The health gate optimizes "stable hold"; B needs "reorientable hold."

**So the single highest-leverage lever is a cheap A-SELECTOR that scores an A draw by DOWNSTREAM
reorientability** — the exact thing the health gate can't do. If we can pick the reorientable A
draw cheaply, we convert "average many full A→B draws" (days, and the confirm sweep showed it still
can't separate designs whose true means differ ≲0.25) into "draw a few A's, probe each, train ONE
imit-B on the best" — the evaluator finally reaching its potential.

## 2. Quick-fix menu (ranked; all cheap, none is the 2–4 day conditioned-policy build)

- **QF1 — cheap downstream-reorientability A-selector (the one that attacks the dominant term).**
  After each from-scratch A draw, run the proven reorienter **b33 ZERO-SHOT** (no training) as
  Policy B on that A's delivered grip through the exact continuous-handoff eval (~10 s). Keep the A
  with the best probe held-cos, then train the real imit-B on it. b33 is m05-biased, so the probe
  is only trusted for **within-design ranking** (bias ≈ constant per geometry, cancels). Validated
  tonight on 31 on-disk (A ckpt → known trained-imit-B cos) pairs — **see §3**.
- **QF2 — `--a-attempts 3` (blessed protocol bump).** The confirm close-out explicitly asked for
  this: best-of-2 can't rank two-FAIL draws and 3/16 legs were lost entirely at n=2. With a
  *reorientability* selector (QF1) the extra attempt also gives the selector more to choose from.
- **QF3 — keep imit-B as the evaluator** (design-neutral, sd ±0.02; already the P4 choice). Not new,
  but it is the load-bearing assumption that makes QF1 worth doing (a low-variance B means the only
  thing left to fix is the A it rides).
- **QF4 — (considered, rejected as the primary) shared b33 *warmstart* for B.** Variance study:
  fix-A + shared-b33-warmstart = 0.86 ± 0.04, tightest band and highest mean. BUT it is a
  *persistent* basin (doesn't anneal like the imit reward), so it biases the map toward m05-similar
  grips — the exact "m05-similarity map" failure mode (H3). Using b33 only as a zero-shot *probe*
  (QF1) gets its selectivity without baking its bias into the trained policy. Kept as a fallback
  lever, not the default.

## 3. Tonight's validation — does the QF1 probe predict the trained outcome? (zero GPU training)

`scripts/probe_a_reorientability.py`: for all 31 on-disk (kept-A checkpoint, its CEM/scene, its
KNOWN trained-imit-B held_cos_tail) pairs from global12x2 + confirm, roll **b33 zero-shot** on the
A delivery and correlate probe held-cos vs the trained-imit-B held-cos. Within-design replicates
(same geometry, vary the A draw) are the clean test: G02_00 n=4, G02_05 n=4, cf_m05 n=3, cf_l13 n=3,
+ 7 designs at n=2.

**VERDICT: QF1 does NOT work as a fine-grained A-selector — an honest, useful negative.**
Data `docs/experiments/PROBE_A_REORIENTABILITY.{json,txt}` (31/31 evaluable, ~10 s each, zero training):

- **Mean within-design Spearman = +0.345; probe picked the best-A in only 6/11 designs.** Weak.
- **It fails on exactly the standout draws** — the ones we most want to keep:
  - G02_00 (n=4): rho **−0.20**. Best trained draw r3 (cos **0.681**) got the *lowest* probe (0.14);
    the probe's top pick r1 (0.98) was only the 2nd-best trained (0.635).
  - G02_05 (n=4): rho 0.00. The program-best draw r1 (cos **0.887**) got probe **0.00** — b33 dropped it.
  - cf_m05 / cf_l13 (n=3 each): rho **−0.50** — mildly anti-correlated (b33 overrated cf_l13_s1:
    probe 0.78 but trained −0.36).
- It retains only **coarse** signal: designs with trained ≥ 0.5 average probe **+0.39** vs **+0.07**
  for trained < 0.2 (gap +0.31). Enough to flag "obviously undeliverable" A's (which the drop check
  already catches), not enough to rank the good ones.

**Why it fails, stated as the finding:** *zero-shot reorientability ≠ trainable reorientability.*
b33 rolls m05-like grips; the best trainable deliveries (G02_00_r3, G02_05_r1) are grips a
*from-scratch B learns* to roll but a fixed reorienter cannot roll cold — so the probe systematically
discards the highest performers. This is consistent with, and extends, `a_quality_predictor.md`
(no cheap single-draw A-quality predictor from scorecard metrics): a *downstream* zero-shot probe is
better than scorecard metrics but still insufficient. **Conclusion: there is no cheap A-selector
shortcut; the A-draw variance is intrinsic, which is precisely why the morphology-CONDITIONED policy
(amortize A/B optimization across designs) remains the real fix.** Not baked into the sweep.

## 4. What I launched — the 6-dim (XY-only) sweep, blessed evaluator + the one endorsed bump

Since no quick variance *fix* survived validation, the 6-dim sweep runs the **blessed, comparable**
evaluator, with the single robustness bump the confirm close-out explicitly endorsed:

```
scripts/morph_pipeline_sweep.py --morph-set global --freeze-len --n 12 --seed 6 --replicas 2 \
  --tag global6xy --b-recipe imit --a-attempts 3
```

- **`--freeze-len`**: LHS over the **6 XY placement dims only**; the three proximal-phalange lengths
  are held frozen at m05 (0.0108 / 0.0123 / 0.0159). Design ids prefixed **H** (H06_00…H06_11).
- **`--b-recipe imit`** (QF3): the design-neutral object-frame imitation evaluator, sd ±0.02.
- **`--a-attempts 3`** (QF2): reduces the ~19% "leg lost entirely" rate seen at best-of-2 in P4.
- **`--replicas 2`**: two independent full draws/design, pooled by mean (H2: n=1 draws are not
  measurements) — directly comparable to global12x2.

**Honest expectation:** with the draw-noise term unfixed (QF1 failed), the 6-dim landscape will most
likely hit the same wall as 9-dim. Its real scientific yield is: (a) does removing the len axis lower
the A-collapse / hostile-geometry incidence (were len extremes causing some collapses?), and (b) is
an XY-only landscape any more resolvable at fixed n? Monitoring + decision tree: `morph_sweep_STATUS.md`
§"6-DIM XY-ONLY SWEEP (2026-07-20)".
