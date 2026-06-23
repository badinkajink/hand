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

## What I'll do when that result lands (autonomously)
- If balanced + lower force → **morphology confirmed**: commit, then (a) widen the sweep toward
  *seating* (longer fingers / palm → palm bears load → ~3 N fingers — the true low-force unlock),
  and/or (b) retrain Policy A on the winner for the full A→B handoff.
- If not → document the gap and iterate the design (index/middle reposition, fingertip size).

## Key files
- `docs/rl/morphology_optimization_plan.md` — the full plan + the honest VGDS read.
- `MORPH_GRASP_SWEEP_RESULTS.txt` — the thumb sweep + ranked winners.
- `scripts/sweep_thumb_grasp.py`, `scripts/train_reorient_on_morph.sh` — the harnesses.
- Commit trail on `origin/main` documents each step.
