# global12x2 — pooled replica table (24/24 complete 2026-07-12 22:13)

Per design: two independent full-pipeline draws (CEM → A best-of-2 → imit-B → handoff eval).
Reference band, m05 clean draws under the same evaluator family: {0.82 (a10→b33), 0.49, −0.16}
(mean ≈ 0.38). Per H2, n=2 verdicts are binned coarsely, not ranked; "—" = leg produced no
policy (both A attempts collapsed).

| design | r0 cos | r1 cos | mean | max | A-aborts/legs | n=2 bin |
|---|---|---|---|---|---|---|
| G02_00 | 0.504 | 0.635 | **0.570** | 0.635 | 1/3 | **replica-consistent REORIENTER** (confirm r2/r3 running) |
| G02_05 | −0.499 | **0.887** | 0.194 | **0.887** | 1/3 | irresolvable: program-best draw + A-defect draw (confirm r2/r3 running) |
| G02_03 | 0.568 | 0.333 | 0.451 | 0.568 | 3/4 | both replicas attempt (r1 peaked 0.999 then dropped) |
| G02_07 | 0.333 | 0.366 | 0.350 | 0.366 | 1/3 | replica-consistent sustained-PARTIAL (Δ0.033, tightest) |
| G02_10 | 0.117 | 0.576 | 0.346 | 0.576 | 1/3 | irresolvable: same-grade-A flip (static → reorients-but-jitters) |
| G02_04 | 0.148 | 0.528 | 0.338 | 0.528 | 0/3 | irresolvable: PASS-static → reorienter flip |
| G02_11 | — | 0.445 | 0.445 | 0.445 | 3/4 | r0 total-miss = draw luck (r1 lifted + partial reorient) |
| G02_06 | 0.127 | −0.018 | 0.054 | 0.127 | 1/3 | replica-consistent STATIC |
| G02_09 | −0.102 | 0.149 | 0.024 | 0.149 | 1/4 | replica-consistent static; only all-health-FAIL-A design (3/3) |
| G02_08 | −0.099 | 0.074 | −0.013 | 0.074 | 0/2 | replica-consistent static; only zero-A-abort design |
| G02_01 | −0.134 | — | −0.134 | −0.134 | 2/3 | static draw + never-lifted draw |
| G02_02 | −0.388 | — | −0.388 | −0.388 | 3/4 | A-defect draw + never-lifted draw |

Headlines: pick-up/hold solved across the full box (min-z ≥ 0.103 on every leg that produced a
policy; every "never lifted" leg is a design whose other replica lifted); 6/10 both-evaluable
designs attempt a reorient in ≥1 replica ⇒ capability is common, EXPRESSION is draw-gated;
n=2 resolves ~half the designs (5 consistent bins), while G02_04/G02_05/G02_10 prove a third
are unbinnable at n=2; A-leg abort rate ~47% program-wide, uncorrelated with capability
(G02_08 zero-abort static vs G02_05 aborting program-best).
