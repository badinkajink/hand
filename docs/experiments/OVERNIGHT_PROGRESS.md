# Overnight morphology progress (2026-06-22 → 23)

Autonomous run while you sleep. Branch: `graspgenx-morphology-eval` (also pushed to `origin/main`).

## The headline result so far: morphology FIXES the lopsided grip (at the grasp level)
The "excessive force" is a structural defect — the thumb is too **short** to reach the object, so
it sits idle and the other two fingers over-clamp. The thumb-opposition grasp sweep
(`MORPH_GRASP_SWEEP_RESULTS.txt`, 40 thumb positions × CEM grasp) proves it:

| morphology (thumb x,y,len) | fingers in contact | persistence t/i/m | imbalance |
|---|---|---|---|
| baseline (0, .02, 0) | **2** | 0.20 / 0.21 / **0.00** | 0.21 |
| **winner (+.02, +.02, +.02)** | **3** | **1.00 / 1.00 / 1.00** | **0.000** |

Moving the thumb out and **lengthening it** turns the degenerate 2-finger pinch into a balanced
tripod that still lifts. `thumb_len` is the key knob. Grasp video:
`results/phase1/morph_winner/thumb_opposed_balanced/best_rollout.mp4`.

## What's running now
- **Skip-lift reorienter on the winner morphology** (`train_reorient_on_morph.sh`, warmstart B4,
  B4's exact recipe) — the direct test of "balanced grasp → balanced, low-force grip", no Policy A
  retrain needed. ~40 min. A background waiter will then run `probe_grip_balance.py` and write
  **`MORPH_REORIENT_EVAL.txt`** (per-finger force: new morphology vs old-morphology B4/gentleB).
  - **Success looks like:** all three fingers share load (thumb no longer ~1.6 N idle), lower
    per-finger force than old gentleB (thumb 1.6 / index 8.0 / mid 6.4 N).

## Morning update (2026-06-23 ~10:30) — what the overnight runs showed

**The morphology fixes the static GRASP, but the REORIENT grip is a warmstart-transfer problem.**

1. **Grasp (CEM): SOLVED.** Thumb (+.02,+.02,+.02) → balanced tripod (persist 1/1/1, imbalance 0).
2. **Reorient on the new morphology still idles a finger** — but a *different* one (index, not
   middle), for *every* recipe tried (B4-recipe, and gentle+spread):
   - newmorph_B4recipe: cos 0.81, thumb 8.9 / **index 0.0** / mid 17.1 (over-clamp)
   - newmorph_gentle+spread: cos 0.51, thumb 0.2 / **index 0.0** / mid 1.9 (barely grips)
   - oldmorph_B4 (ref): cos 0.99, 7/10/10 — **all 3 engaged**
3. **Diagnosis:** B4 keeps all three fingers engaged *on its own morphology* but idles one when
   warmstarted onto a different geometry → the reorient grip balance needs a warmstart that **matches
   the morphology**, not just a balanced grasp. The spread penalty couldn't recruit the idle finger
   (same as on the old morphology — it's still structural at the policy/warmstart level).
4. **Policy A retrained fine** on the new morphology (lifts to 0.060).

**Running now:** a reorienter **warmstarted from the NEW A** (native to the balanced morphology)
instead of B4 — the clean test of the warmstart-match hypothesis. Waiter → `MORPH_FROMA_EVAL.txt`.

## Throughput (measured) — `docs/rl/morphology_throughput.md`
~8.9k steps/s per 2048-env run, ~17.8k for 2 parallel (near-linear), GPU only 18–31% util →
headroom for 4–6 parallel. Funnel: CEM grasp screen ~36–130 designs/hr → A+B retrain ~16–24/night.

## Open decision for you
If the from-A reorienter *also* idles a finger, the reorient task itself wants an asymmetric
2-finger roll and the lever shifts to either (a) a stronger all-finger-contact constraint during
reorient, or (b) the **seating** morphology direction (longer fingers/palm → palm bears load), which
is the orthogonal low-force unlock and the next thing I'd sweep.

## Key files
- `docs/rl/morphology_optimization_plan.md` — the full plan + the honest VGDS read.
- `MORPH_GRASP_SWEEP_RESULTS.txt` — the thumb sweep + ranked winners.
- `scripts/sweep_thumb_grasp.py`, `scripts/train_reorient_on_morph.sh` — the harnesses.
- Commit trail on `origin/main` documents each step.
