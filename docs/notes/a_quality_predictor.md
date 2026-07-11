# A-side predictor of B fate — can any A-scorecard metric restore single-draw eval?

*2026-07-11 (autonomous tick, GPU busy on P4 global12x2). Standing CPU task from
`docs/rl/morph_sweep_STATUS.md`. Analysis script: `scripts/a_quality_predictor.py` (rerun with
`--tags confirm large16 rescue avar global12x2` once P4 lands; needs the runs still in
`results/rl/` — results is gitignored, scorecards vanish if runs are pruned).*

## Question

P2 `avar` showed per-draw B held-cos sd ≈ 0.3–0.5 with the A health gate blind to the difference
(m05 k0 vs k1: gate-identical As → B cos 0.49 vs −0.16). P4 pays for this with 2 full replicas
per design. If some *fine-grained* metric of the kept A checkpoint's trajectory-health scorecard
(`results/rl/<run>/tensorboard/<ckpt>.health.json`) predicted the downstream B fate, a single-draw
evaluator would come back and landscape cost would halve.

## Data

32 (A, B) records across `confirm` / `large16` / `rescue` / `avar`; **26 still have the A
scorecard on disk** (25 completed-B pairs + 1 B-collapse pair; the other B-collapses had aborted
As with no health JSON). Candidate predictors: hold min-z, ang-jerk, net drift, slide ratio, mean
tip force, contact spread, per-finger force min/max, min touch fraction, contact count.

## Verdict: NO usable single-draw predictor. Keep P4's 2-replica design.

Spearman(A metric → B held-cos), n=25 completed pairs:

| A metric | rho | | A metric | rho |
|---|---|---|---|---|
| mean tip force | **+0.44** | | min touch frac | +0.32 |
| max finger force | +0.41 | | contact count | +0.29 |
| min finger force | +0.36 | | ang-jerk / drift / slide | ≈ 0 |

- The best value (+0.44) sits at the p≈0.05 boundary for n=25 and the median-split gap is thin
  (hi-tipF median cos +0.55 vs lo +0.48; means +0.50/+0.29 — tail-driven).
- **The "grip richness" trend is mostly a geometry confound, not a draw-quality signal.** Within
  m05 (5 same-geometry draws) tip force vs B cos is non-monotone: 7.0 N → 0.49, 8.1 N → −0.16,
  11.3 N → 0.23, 2.8 N → 0.21, 3.5 N → −0.29. Designs whose geometry affords a firm 3-finger
  delivery also tend to reorient — that is the landscape itself, not an evaluator shortcut.
- Coarse A verdict is already known-uninformative (A-FAIL → B 0.899 on L01_06; A-WARN → −0.16 on
  m05 k1). Confirmed here in the fine metrics too.

## Two actionable side-findings (cheap, adopt)

1. **Idle-finger veto at A-accept.** The single scored B-collapse pair (rs_L01_05 kept A t1) is
   the only A in the whole set with `min force = 0.0 N` / `min touch frac = 0.0` — a delivery
   where one finger never touches. Both its B recipes watchdog-collapsed. n=1, but the veto is
   free (the metric is already in the gate's scorecard) and it can only reject a delivery that
   is degenerate by construction. → candidate `--a-attempts` retry trigger in
   `morph_pipeline_sweep.py::train_A`: retry when `min(force_mean) < 0.5 N`.
2. **The A scorecard's drop check has a pre-lift-window artifact.** 4/26 kept As show
   `min_z_hold` ≈ 0.006–0.008 (floor) and hence a spurious drop-FAIL verdict — yet 3 of the 4
   produced good Bs (0.899, 0.565, 0.21). This is the known "whole-rollout min-z is dominated by
   pre-lift floor z≈0.012" trap leaking into the A-side hold-phase window. It explains part of
   why gate verdicts don't rank draws. Fix = clamp the hold-phase window start to the scripted
   lift completion in `trajectory_health.py` A-eval, or ignore the drop check when ranking
   attempts (`VERDICT_RANK` already only breaks ties, so impact is limited to attempt selection).

## Revisit trigger

Re-run after P4 `global12x2` (+24 pairs incl. per-attempt collapse records, 2 draws/design →
within-design pairs at scale). If a predictor emerges there at n≈50, wire it as a gate; until
then the replication cost stands and the step-8a morphology-conditioned-policy spike remains the
structural fix.
