# Morphology sweep — autonomous run STATUS / runbook (2026-07-03)

Live status + resume commands for the co-design morphology sweep launched 2026-07-03 (user away
for the day; "build on top of whatever we get"). This file is the single place to see what is
running, what was decided, and how to continue/intervene. Updated as stages complete.

## What this is

Explore hand morphologies using the **clean, health-gated m05 pipeline** (the policy in
`docs/rl/videos/reorient/handoff_m05_FIXED.mp4` = **a10** native lift → **b33** live-A-reset
reorient). Per design, the FULL honest pipeline runs — no cheap skip-lift proxy:

```
9-param design → generate scene → IK-retarget open_ik keyframe → CEM grasp (graspability gate)
  → native Policy A (from scratch, open-finger-from-keyframe, deliver@0.10, ~55 min)
  → Policy B reorient (live-A reset, warmstart the hold-first A, ~36 min)
  → continuous A→B handoff eval → trajectory-health scorecard (.health.json)
```

Orchestrator: `scripts/morph_pipeline_sweep.py` (resumable, per-design JSON checkpoint, per-design
try/except, DONE sentinel). Analysis: `scripts/morph_pipeline_plots.py`.

## Decisions made (change if you disagree)

- **Search = LOCAL refinement around m05**, not a fresh global LHS. Rationale: the 2026-06-25
  landscape already mapped the global picture (held-cos −0.68…+0.93, m05 best); the documented
  next step (`morphology_optimization_plan.md` Stage 2) is a local refine around m05 scored on the
  full A→B rollout. This builds directly on the winner and targets the OPEN goals (lower force,
  smoother, more balanced / recruit the thumb / seat toward the palm).
- **Initial sweep = 8 interpretable coordinate moves** around m05 (Stage-1(a/b/c) hypotheses):
  `s00_m05anchor` (reproduce), `s01_baseline` (m00 reference), `s02_thumbreach`, `s03_thumblong`,
  `s04_seat_allen`, `s05_shortgrasp`, `s06_middlein`, `s07_thumb_opp`. See `morph_set()` in the
  orchestrator for exact Δ-vectors.
- **Larger sweep = 16 seeded local Gaussian samples** around the best non-FAIL design from the
  initial 8 (`--morph-set local --n 16 --center best`), same full pipeline.
- **Timesteps kept at the validated m05 recipe** (A 30M, B 20M) for fidelity over speed.
  ~100 min per graspable design; ungraspable/aborted designs are cheap (gated).

## Staged plan + triggers

1. **initial8 sweep** (~11–13 h) — running/queued. Detached process + a `run_in_background` waiter
   on the `docs/experiments/MORPH_PIPELINE_initial8.DONE` sentinel re-invokes the session on completion (or crash).
2. On completion → **analysis**: `morph_pipeline_plots.py --tag initial8` (summary + training
   figures + markdown table), render a comparison of the best handoffs, write up in
   `reorientation.md`, update memory.
3. → **launch larger sweep** (`--morph-set local --n 16 --center best`), same waiter pattern.
4. On completion → analysis + docs again.

## Monitor / resume / intervene

```bash
cd /home/humanoid/Programs/hand
# progress (one line per finished design):
cat docs/experiments/MORPH_PIPELINE_initial8.txt
tail -f sweep_initial8.run.log            # live stage markers ([HH:MM:SS] <id>: ...)
python3 -c "import json;print(len(json.load(open('docs/experiments/MORPH_PIPELINE_initial8.json'))),'designs done')"
nvidia-smi                                 # is a trainer running?

# RESUME after any crash/kill (skips finished designs — safe to re-run):
MUJOCO_GL=egl uv run --extra rl --extra gpu \
  python scripts/morph_pipeline_sweep.py --morph-set initial8 > sweep_initial8.run.log 2>&1 &

# STOP everything:
pkill -f morph_pipeline_sweep.py ; pkill -f rl_train_cube.py   # (never pkill from inside its own cmd)
```

Outputs: `docs/experiments/MORPH_PIPELINE_<tag>.{json,txt}`, `sweep_{A,B}_<id>.trainer.log`, run dirs
`results/rl/<ts>-policy{A,B}_<id>*` (auto-`bx_`'d by `rename_results_bids.sh`), handoff videos +
`.health.json` in `docs/rl/videos/reorient/sweep/`.

## STATUS LOG

- **2026-07-03 12:04** — cleanup done (a10/b33 promoted, 34 explorations `bx_`'d, REGISTRY
  idempotent). Orchestrator built + **smoke-validated end-to-end** (one min-z parse bug found +
  fixed). **initial8 sweep LAUNCHED** (detached, from-scratch A recipe).
- **2026-07-03 21:19 — initial8 DONE (8/8), headline = PIPELINE FRAGILITY.** All designs
  grasp-equivalent (CEM 0.05 / 1·1·1); but 6/8 aborted in A or B training, incl. the m05 anchor
  `s00` (A collapsed 0.127→0.026 at iter 55 → confirms a10/m05 was a lucky single seed). Only
  `s05_shortgrasp` held+reoriented (FAIL on jitter/clamp only — policy quality). Analysis:
  `morph_pipeline_initial8_summary.png` / `..._training.png` / `docs/experiments/MORPH_PIPELINE_initial8_TABLE.md`;
  writeup in `reorientation.md` → "co-design morphology sweep". **Diagnosis: bottleneck is RL
  robustness, not morphology.**
- **2026-07-03 21:25 — fix attempt 1 (warmstart a10/b33) → canary `valfix` FAILED.** a10-warmstart
  A never lifted (residual ejects the re-CEM'd object — the documented reason A must be
  from-scratch). Misdiagnosis; reverted.
- **2026-07-03 21:45 — ROOT-CAUSED.** Real causes: (1) B step **omitted
  `--open-finger-from-keyframe`** → wrong open pose → drop (sank s03/s04/s06/s07; verified in
  configs); (2) from-scratch PPO mid-training collapse (s00 A@55, s01 B@205).
- **2026-07-03 22:xx — `valfix2` = WARN-but-idle-finger FAIL; found the LAST bug.** "best A ckpt by
  object-height" wrongly preferred the undertrained **model_50** (higher raw lift) over the
  fully-trained **model_609** → under-refined grip → idle finger. Fix: **final ckpt on clean
  completion; salvage earlier only on abort**.
- **2026-07-04 00:00 — `valfix3` VALIDATED the corrected pipeline** (reused A model_609 + retrained
  B): **WARN**, all 3 fingers loaded (idle PASS), held (min-z 0.111), smooth (jitter 15) — same
  class as b33 (cos 0.66/firm 12N here = from-scratch seed variance vs a10→b33's 0.90/7N). Video
  `docs/rl/videos/reorient/sweep/valfix3_m609_handoff.mp4`.
- **2026-07-04 00:04 — LARGE16 LAUNCHED** (`--morph-set local --n 16 --center m05 --seed 1 --tag
  large16`, corrected pipeline; design 0 = m05 anchor). Waiter fires at 4-design milestone (~6h) /
  completion / crash. ETA ~24h. **Next on completion:** `morph_pipeline_plots.py --tag large16` →
  rank designs vs m05 (best held-cos / lowest force / balance) → render top handoffs → docs +
  memory → promote a winner if one beats m05.
- **2026-07-04 05:0x — large16 MILESTONE (4/16): HEALTHY, real signal.** Force/verticality tradeoff
  emerging; no systematic failure. Completion waiter re-armed.
- **2026-07-04 23:29 — large16 DONE (16/16). SEED VARIANCE dominates the ranking.** 14/16
  held+reoriented (no drops); 1 A-collapse (gated), 1 wrong-way. Ranked by held-cos: L01_06 0.90
  (≈m05 geometry → seed luck), m05-anchor 0.78, **L01_13 0.76 @ force 7.4 / jerk 6.0 = best design
  lead** (thumb_x +9mm; lower force + half the jerk of m05). Health monitor caught L01_02's 2.0N as
  degenerate (idle-finger FAIL). **m05 across seeds = 0.66/0.78/0.90 → single-seed can't separate
  design from luck.** Analysis: `morph_pipeline_large16_*.png`, `docs/experiments/MORPH_PIPELINE_large16_TABLE.md`;
  writeup in `reorientation.md`. Videos `docs/rl/videos/reorient/sweep/L01_{06,13,00_center}_handoff.mp4`.
- **2026-07-04 23:34 — CONFIRM sweep launched** (m05 ×3 + L01_13 ×3 fresh seeds).
- **2026-07-05 ~06:00 — CONFIRM DONE → NEGATIVE (definitive).** Pooled seed bands: **m05 cos
  0.32±0.38 [−0.29,0.78] n=5; L01_13 cos 0.38±0.44 [−0.36,0.76] n=4.** Gap 0.07 << pooled sd 0.41 →
  **L01_13 INDISTINGUISHABLE from m05** (also force/jerk). The large16 "lead" was one lucky draw.
  **Conclusion: per-design from-scratch A→B reorient quality has huge seed variance (cos sd ≈0.4)
  that SWAMPS any local design effect — bottleneck is RL seed-sensitivity, not geometry. No design
  promoted.** Figure `img/morph_confirm_seedbands.png`; full writeup in `reorientation.md`.
  **STOPPED here (correct):** more single-seed sweeps won't help. Next requires a METHODOLOGY change
  (user's call): (a) ≥5–10 seeds/design averaging; (b) variance-reduced/stable reorient trainer
  (shared warmstart prior, not each design's noisy A); (c) cheaper low-variance proxy score; OR
  accept m05 + do the deferred hard-contact sim2real pass. GPU idle; nothing running.

## Variance-reduction experiments (2026-07-06) — toward statistically separating morphologies

The large16/confirm finding was that per-design reorient held-cos has sd ≈ 0.4 (spans negative to
0.8), from training-convergence luck (peak cos varies 0.02→0.81 by seed), so designs can't be
separated. Three levers, per the user:

- **#1 shared-warmstart-B + #2 N-seed — DONE (2026-07-06): VARIANCE SOLVED.** `reorient_variance_study.py`.
  **Fix-A cut cos sd 4× (0.38→0.09); shared-warmstart-B halved again (→0.04), ~8× total.** Delineates
  m05 vs L01_13: cos equivalent (fair self mode) but **m05 lower-force (7–8 vs 10–12 N, separable)** →
  L01_13's single-seed "lead" was a seed artifact, **m05 validated as reference.** Shared mode's cos
  separation is confounded (b33 = m05's own reorienter). Recommended evaluator: **fixed-A +
  self-warmstart-B + ~3 seeds**. Fig `img/variance_reduction_bands.png`; writeup in `reorientation.md`.
- **#3 object-relative fingertip imitation (CORE BUILT, pending GPU-free smoke-test).** Record the
  blessed a10→b33 reorient's **object-frame** fingertip trajectory (transferable across morphologies,
  unlike joint angles) and imitate it with a curriculum. Built: `src/morphohand/rl/imitation.py`
  (reference loader + `track_fingertip_obj` reward, math-validated) + `--record-fingertip-traj` in
  `rl_demo_handoff_continuous.py`. TODO (needs GPU to smoke-test, so queued behind #1+#2): wire the
  reward into `env_cfg`/`rl_train_cube` CLI + a weight curriculum + a training script; record the
  reference; train B on m05 (+ L01_13 to test transfer); measure the band. Design/priority may
  update from #1+#2's outcome.

## FINAL STATE (2026-07-05)

Nothing running, GPU idle. Deliverables from this autonomous run:
- **Cleanup + naming:** a10/b33 canonical, REGISTRY idempotent, `*.COLLAPSED` gitignored.
- **Pipeline:** `scripts/morph_pipeline_sweep.py` (health-gated per-design A→B, resumable) +
  `morph_pipeline_plots.py`; 3 bugs found+fixed (B open-finger, final-vs-early ckpt, no warmstart).
- **Sweeps:** initial8 (fragility→bugs), large16 (16 designs ranked), confirm (6-run seed test).
  Data: `docs/experiments/MORPH_PIPELINE_{initial8,large16,confirm}.{json,txt}`, `*_TABLE.md`, figs in `docs/rl/img/`,
  handoff videos in `docs/rl/videos/reorient/sweep/`.
- **Result:** honest NEGATIVE — no local design beats m05 within seed noise; the win is the
  pipeline + variance characterization. m05 (a10→b33) remains the reference.

## FINAL STATE (2026-07-08) — variance solved, sim2real characterized

Nothing running, GPU idle. Since 07-05:
- **Variance SOLVED** (`scripts/reorient_variance_study.py`): fix-A cuts held-cos sd 4×
  (0.38→0.09); a design-neutral **object-relative fingertip imitation** prior
  (`src/morphohand/rl/imitation.py`, recorded via `--record-fingertip-traj`) cuts it to ±0.02 and
  gives the smoothest/lowest-force policies. Designs now separate: **m05 0.82 > L01_13 0.72**,
  m05 lower-force → **m05 validated as reference**. Fig `img/variance_reduction_bands.png`.
- **Sim2real contact hardening** (`solimp 0.97/0.995 → 0.985/0.999`): **grasp transfers** (retrain
  holds), **reorient does NOT** (rolling needs compliance; align 13 vs 48). Eval-only
  **compliance-robustness sweep** (`scripts/compliance_robustness_sweep.py`): trained policies are
  **fragile/non-monotonic** to stiffness; single-stiffness training overfits; imitation degrades
  most gracefully. Figs `img/compliance_robustness.png`; scenes `assets/mjcf/experimental/sim2real/`.
- **Docs updated:** `reorientation.md` (full arc), webpaper `rl.typ` (co-design + variance + sim2real
  sections, builds clean), `paper/main.tex` (Method/Experiments/appendix incl. app:variance,
  app:sim2real; also fixed a pre-existing `\labelindent` preamble breakage). `CLAUDE.md` created.
- **NEXT (spec written, not run): compliance domain randomization** — `docs/rl/compliance_dr_plan.md`.
  Randomize `solimp` per episode over [soft, hard] → a stiffness-robust policy; retrain A +
  imitation-B, re-run the compliance sweep, expect a flat curve.

## POLICY-BOTTLENECK PROBES (2026-07-10) — is the landscape gated by the optimizer, not the designs?

**User directive (2026-07-10, signing off):** validate the intuition that the bottleneck on
morphology exploration is *the policy itself* — "many of the morphologies sampled never learned at
all to pick up and reorient the screwdriver, even after fixing the initial keyframe retargeting."
Compliance/DR tangent is CLOSED (rate sweep: DR mirrors, doesn't dominate, baseline). Core problem
resumed: characterize the joint performance×morphology landscape (>16 designs), which first needs
an evaluator whose verdicts aren't optimizer noise.

**Evidence already in hand (why the intuition is probably right):**
- confirm sweep: m05 — the *best known* design — spans held-cos **−0.29..0.78 over 5 joint A+B
  retrains** (sd 0.41). Under n=1 evaluation, even m05 "never learns" ~40% of the time.
- variance study: with A fixed, B-only sd = 0.09 (self) / **0.02 (imit)** ⇒ **Policy A's
  from-scratch draw is the dominant noise term**, not B.
- ALL 5 large16 failures had an A-side event: L01_03 collapsed at iter 0; L01_05 late-collapse
  (salvaged an undertrained model_50 — the known-bad valfix2 mode); L01_02/07/09 completed but
  delivery health-FAIL. **No large16 failure is yet evidence about the morphology itself.**
- Caveat that keeps this falsifiable: collapse/health-FAIL rates could themselves be
  design-dependent (geometry → bad A basin). That is exactly what P2 measures.

**Hypotheses:**
- **H1 (headline):** most "never learned" verdicts flip under a stronger optimizer draw
  (A best-of-2 + imitation-B) ⇒ bottleneck = policy optimization, verdicts were Type-II noise.
- **H2:** with collapse-retry, the residual A-draw variance is small enough that
  **CEM → A(best-of-2) → imit-B ×1** is a sound per-design landscape evaluator (~2 h/design).
- **H3:** the m05-recorded object-frame imitation prior is *fair* off-m05 (it anneals out over
  150 iters): per-design Δcos(imit − self) on the SAME A is ≥ −0.05. If it handicaps distant
  designs, the imit landscape would be an "m05-similarity map", not a capability map.

**Queue (launched 2026-07-10 eve, detached):** `scripts/probe_queue.sh` → `logs/PROBE_QUEUE.log`
- **P1 `rescue`** (~12–16 h): 5 large16 failures (`rs_L01_{02,03,05,07,09}`), A best-of-2 (every
  attempt recorded in the JSON = raw draw data), then **both** B recipes on the same kept A
  (`b_liveA_imit` → `handoff`, `b_liveA` → `handoff_self`). Flip = held (min-z>0.05) + cos ≥ 0.5
  + verdict ≠ FAIL.
- **P2 `avar`** (~8 h): raw A draws, NO retry: `av_m05_k{0..2}` + `av_L01_05_k{0,1}` (+ pool
  rs_L01_05's P1 attempts), imit-B per viable A. Yields per-design P(collapse), P(health-FAIL),
  and cos spread across A draws.
- Outputs: `docs/experiments/MORPH_PIPELINE_{rescue,avar}.{json,txt}` (JSON has per-attempt A
  records), videos `docs/rl/videos/reorient/sweep/rs_*_handoff{,_self}.mp4`, sentinels
  `logs/MORPH_PIPELINE_{rescue,avar}.DONE`, `logs/PROBE_QUEUE.DONE`.
- **P4 ready to fire (do NOT start until the decision tree says so):** `--morph-set global`
  = Latin hypercube over the FULL 9-param box (the honest-pipeline replacement for the 06-25
  teleport-proxy global map).

**DECISION TREE for the next session (pulse- or waiter-triggered) — execute, don't re-derive:**
1. **Safety first:** if `pgrep -f "[m]orph_pipeline_sweep|[r]l_train_cube|[p]robe_queue"` shows a
   live worker → analysis/docs/commit work ONLY, never launch GPU jobs (single 16 GB GPU).
2. **Crashed/stuck queue** (no worker, no `PROBE_QUEUE.DONE`, stale run logs >30 min):
   relaunch `nohup setsid bash scripts/probe_queue.sh > logs/PROBE_QUEUE.log 2>&1 &` (resumable);
   log the incident here.
3. **P1 done → score H1:** flip fraction of 5. ≥3/5 ⇒ H1 CONFIRMED (write up + memory).
   ≤1/5 ⇒ failures are real geometry effects ⇒ landscape resolvable already ⇒ jump to (5).
4. **P1 done → score H3:** per-design Δcos(imit−self, same A). If Δ ≥ −0.05 everywhere ⇒ imit-B
   stays THE evaluator. If imit systematically loses off-m05 ⇒ evaluator = self-B, and P4 needs
   ≥2 B seeds/design (halve `--n`).
5. **P2 done → score H2:** if non-collapsed draws are tight (cos sd ≤ 0.10) and collapse is
   design-independent luck ⇒ evaluator = CEM → A(best-of-2) → imit-B ×1. If A quality varies
   continuously per design ⇒ P4 uses `--a-attempts 2` AND averages 2 full draws/design (halve n).
   If L01_05's draws are *systematically* bad (e.g. ≥4/5 collapse or all cos<0.3 while m05's are
   fine) ⇒ **the design effect on TRAINABILITY is real** — that is itself a landscape axis; keep
   per-design collapse rate as a first-class output in P4.
6. **Launch P4** (once 3–5 answered, GPU free):
   `nohup setsid env MUJOCO_GL=egl uv run --extra rl --extra gpu python
   scripts/morph_pipeline_sweep.py --morph-set global --n 24 --seed 2 --tag global24
   --b-recipe imit --a-attempts 2 > logs/sweep_global24.run.log 2>&1 &`
   (~2 days; resumable; analyze with `morph_pipeline_plots.py --tag global24`; extend with
   `--seed 3` batches in later windows).
7. **Every touchpoint:** update this STATUS log (dated bullet), append results to
   `docs/rl/reorientation.md`, commit code+docs (never `results/`), refresh
   `~/.claude/.../memory/` if a durable conclusion landed.
8. **Idle-GPU fallback** (queue done, P4 not yet justified): (a) timeboxed feasibility spike —
   mjwarp per-env geometry batching (body_pos/geom_size per world) for a
   **morphology-CONDITIONED policy** (one policy conditioned on the 9-vector across randomized
   morphologies = the fundamental fix to the evaluate-requires-optimize chicken-and-egg; write
   findings to `docs/notes/morph_conditioned_policy_spike.md`); (b) render/inspect `rs_*` videos;
   (c) L01_03 forensics (its CEM said graspable-1.0 — watch its A attempts' first 50 iters).

**Monitoring:**
```bash
tail -f logs/PROBE_QUEUE.log                     # stage banners
tail -f logs/sweep_rescue.run.log                # live per-design stage markers
cat docs/experiments/MORPH_PIPELINE_rescue.txt   # finished-design rows
pgrep -af "morph_pipeline_sweep|probe_queue"; nvidia-smi
ls logs/*.DONE
```
claude-pulse (deployed 2026-07-10): config `~/.config/claude-pulse/config.toml`, cron `*/15`
tick; pokes an autonomous session pointed at THIS section when a usage window idles ≥75 min.

**Probe progress log:**
- **2026-07-10 17:39–17:47 — smoke PASSED (mechanics).** `MORPH_PIPELINE_smokeprobe.*`: full
  pipeline on rs_L01_05 with truncated training (A stopped at model_19; 451 s total). Verdict FAIL
  cos −0.36 is expected at that training length and is NOT evidence about the design. What it
  proved: `--b-recipe both` runs imit+self B on the same kept A, both handoff evals produce
  scorecards + videos (`rs_L01_05_handoff{,_self}.mp4`), and the JSON records per-attempt A draws.
  (These smoke videos will be overwritten when the real rescue stage reaches rs_L01_05.)
- **2026-07-10 ~19:15 tick — queue healthy, P1 in flight on design 1/5 (rs_L01_02).** CEM lift
  0.055 persist 1/1/1 (4 min). Policy A accepted on its **first** draw (kept model_609, objheight
  0.117, no abort, no health-FAIL → best-of-2 short-circuited). Early H1-relevant note: rs_L01_02
  failed large16 via delivery health-FAIL, yet a fresh from-scratch A draw passed the gate
  immediately — consistent with A-draw noise, pending the B verdict. imit-B near done (ETA ~7 min
  at check), self-B next. No collapse sentinel; GPU normal.
- **2026-07-10 ~20:20 tick — rs_L01_02 COMPLETE (7856 s): partial flip; H1 strict 0/1, capability
  flip 1/1; H3 1/1 imit-fair.** imit-B **holds and reorients** (post-handoff min-z 0.1115,
  held-cos 0.561, peak 0.678, 3-finger 9–10 N contact) — verdict FAIL on **jitter only**
  (ang-jerk 44.1 vs bar 40; b33 ref 9.6), WARN sliding (path 22 cm/net 1.3) + over-clamp 9.7 N.
  Paired self-B on the SAME A **drops** (min-z 0.0062; its peak-cos 0.999 is floor-bracing).
  H3: Δcos −0.02 ≥ −0.05 AND imit held where self dropped ⇒ prior not a handicap here. Emerging
  split to score across the remaining 4: **trainability flips** (stronger draw rescues lift+hold+
  reorient) vs **quality residual** (jitter) — if jitter FAILs persist, consider whether the
  strict flip bar conflates the two axes. Queue healthy, now on rs_L01_03 (the iter-0 collapse
  design) — its A t0 draw training since 20:03; watchdog armed. Full analysis appended to
  reorientation.md §P1 interim.
- **2026-07-10 ~22:20 tick — rs_L01_03 COMPLETE (7775 s): A rescued (2/2), but NO reorient under
  EITHER B recipe; H1 strict 0/2, capability 1/2; H3 1/2.** The iter-0-collapse design trained A
  clean on its **first** fresh draw (model_609, objheight 0.1115, WARN) — both large16 A-side
  failures so far were draw luck. imit-B holds (min-z 0.1092, all 3 fingers 8.5–17.6 N, thumb
  engaged) but held-cos **−0.047** (peak 0.109): a static hard clamp, verdict WARN (jitter 31.6 /
  over-clamp 13.6 N / micro-slip 8 cm path, 0.2 net). Paired self-B also holds (min-z 0.1126) and
  also fails to reorient (cos 0.174, peak 0.309, jitter-FAIL 114). Both recipes failing on the
  same A leans **real reorient-hostile geometry** — first data point against pure H1 on the
  reorient axis; the trainability/capability split sharpens (L01_02 reorients-but-jitters,
  L01_03 holds-but-static). H3: Δcos(imit−self) = −0.22 < −0.05 — first fairness strike, but
  marginal (both cos in no-reorient territory, self edge rides jerk 114, single-draw sd ~0.09);
  if it repeats, P4 evaluator → self-B ×2 seeds. Queue healthy, now on rs_L01_05 (the
  late-collapse design): CEM lift 0.056 persist 1/1/1, A t0 training since 22:13. Full analysis
  in reorientation.md §P1 interim design 2/5.
- **2026-07-10 ~23:45 tick — rs_L01_05 IN FLIGHT: first trainability-hostile design. A t0
  COLLAPSED (iter 98), t1 clean (kept model_609, objheight 0.1236 — first best-of-2 retry
  actually used); then imit-B ALSO watchdog-collapsed (iter 194, first B-side collapse in the
  program).** Salvaged imit model_150: FAIL idle_finger — one-finger pin (middle 7.4 N, thumb+
  index 0 N), min-z 0.1217, cos −0.469. Self-B (same A) training since 23:41 and struggling
  (22 live-A "trainable frac=0" guard hits in ~34 iters). Cumulative A-draw record for this
  design: 2 collapses/3 draws vs 0/2 first-draws for L01_02/03 ⇒ collapse propensity looks
  design-dependent — the step-5 clause; P2 `avar` (av_L01_05_k{0,1} + pooled attempts) decides.
  Interim tallies: H1 strict 0/3, capability 1/3; H3 design-3 pending self-B. Analysis in
  reorientation.md §P1 interim design 3/5. Queue healthy; GPU busy (self-B) — no launches.
- **2026-07-11 ~03:0x tick — designs 3+4 COMPLETE, design 5/5 (rs_L01_09) on its last leg.
  H1 strict is DECIDED: 0/4 complete ⇒ ≤1/5 ⇒ step-3 branch "failures are real geometry
  effects" — but axis-split (see below).** rs_L01_05 finished trainability-hostile: self-B
  ALSO watchdog-collapsed; salvage = no-grip drop (all fingers 0 N, min-z 0.0441, cos −0.969).
  Design ledger: 4 training legs, 3 collapses, 0 viable policies. H3 there: Δ=+0.50 pro-imit
  but salvage-garbage ⇒ uninformative. rs_L01_07 (7769 s): A first-draw clean; BOTH B recipes
  hold beautifully (min-z ~0.112, 3 fingers 8–12 N) and neither reorients (imit cos 0.032 /
  self 0.059, peaks ≤0.10) — second **holds-but-static** design after L01_03; H3 Δ=−0.027
  fair (strike did not repeat; H3 tally fair 2 / strike 1 / uninformative 1 ⇒ imit-B stays
  the evaluator, pending design 5). rs_L01_09 in flight: BOTH A attempts aborted (kept =
  salvaged t1 model_150, objheight 0.128 — best-of-2's first total miss); imit-B collapsed
  iter ~100 (salvage: thumb-idle 2-finger pinch, cos 0.249, FAIL) ⇒ **second
  trainability-hostile design**; self-B running (last P1 leg). Emerging class map within
  ±8 mm of m05: reorients (L01_02) / holds-but-static (L01_03, L01_07) /
  trainability-hostile (L01_05, L01_09?). Next per tree: P1 DONE → final H1/H3 writeup;
  P2 avar auto-starts (queue); step-5 (H2 + design-dependent collapse) is now the pivotal
  question — P4 shape hinges on it (collapse rate as first-class output; likely
  --a-attempts 2 + salvage-A quality gate). Full analysis in reorientation.md §P1 designs
  3–5. Queue healthy; GPU busy (self-B rs_L01_09) — no launches this tick.
- **2026-07-11 03:32 — P1 rescue CLOSED (5/5, rc=0, 9h44m); avar auto-started (waiter fired).**
  rs_L01_09's self-B trained CLEAN (softens it to *A-fragile*, not fully hostile) but static:
  min-z 0.1249, cos −0.014, FAIL. **FINAL: H1 strict 0/5** (tree step-3 low branch) with the
  axis split now definitive — **pick-up/hold 5/5 rescued** (min-z 0.109–0.125 both evals, every
  design) = the "never learned to pick up" claims were pure A-draw noise; **reorient 0/5**
  rescued, paired recipes agree per design ⇒ real sub-cm geometry signal (m05 0.82 → 0.56 /
  ~0.0×3 / −0.5 within ±8.2 mm); **trainability clusters by design** (0/6 collapse legs on
  L01_02/03/07 vs 3/4 on L01_05 and 3/4 on L01_09). **H3 CLOSED: keep imit-B** (fair 3 /
  strike 1 / uninformative 1, informative-mean Δ≈0, imit gave the only reorient + held where
  self dropped). Full close-out in reorientation.md §P1 CLOSED. **Next: score H2 on avar**
  (running; ETA ~12:00–14:00): m05 raw-draw collapse rate vs L01_05's pooled 3-of-4 → decides
  step-5/6 and P4's `--a-attempts`/collapse-rate-output shape. NOTE for P4 scoring: consider a
  softened flip bar (cos ≥ 0.5 + held, jitter graded separately) so the quality axis doesn't
  mask capability — L01_02 was a capability flip hidden by a jitter-only FAIL.
- **2026-07-11 ~05:00 tick — P2 avar healthy, design 1/5 (av_m05_k0) on its B leg. First H2
  raw-draw data point: m05 draw 1 trained A CLEAN** (CEM lift 0.054 persist 1/1/1; A model_609,
  objheight 0.1176, no abort/collapse — 54 min). imit-B in flight since 04:30: iter 232/271,
  object_height 0.128 (≫ 0.030 watchdog bar), only 2 trainable-frac guard hits (L01_05's
  struggling leg had 22) — no collapse sentinel. Running m05 collapse tally: 0/1 raw draws vs
  L01_05's pooled 2-of-3 A-side (H2/step-5 scoring waits for all 3 m05 + 2 L01_05 draws).
  Committed the P1 stragglers that landed after the 03:32 close-out commit (rs_L01_09 self-B
  video + health JSON, avar table header). GPU busy — no launches this tick.
- **2026-07-11 ~08:10 tick — m05 control COMPLETE (3/3 raw draws): H2's tightness clause is
  DEAD early; step-5 middle branch effectively selected.** k0 A clean → imit-B WARN cos 0.488
  jerk 9.8 (real reorienter); k1 A clean and *gate-identical to k0* (objheight 0.1175 vs
  0.1176, both WARN, same ckpt index) → holds-but-static cos −0.158 jerk-FAIL 64.6; k2 A
  COLLAPSED (1/3 raw-draw collapse on the best design; salvage-B drops, uninformative). Clean
  m05 draws + the a10→b33 reference give a per-draw cos distribution {0.82, 0.49, −0.16},
  sd ≈ 0.49 ≫ the 0.10 tightness bar — and the A health gate can't see the difference, so
  best-of-N-by-gate is collapse insurance only. **Consequence: P1's reorient-axis "real
  geometry signal" is WEAKENED (not overturned)** — paired-recipe agreement controlled B-seed
  noise, not A-draw noise; m05's own draw spread covers every P1 verdict except L01_05's
  salvage (counterpoint: 3/4 completed designs all drawing static ≈ 4% likely if
  m05-equivalent). Trending P4 shape: `--n 12` × 2 full-draw replicas (`_r0/_r1` in
  `morph_set()` "global" branch, ~3 lines) + `--a-attempts 2`, per-design mean cos, collapse
  count descriptive; step-8a morph-conditioned-policy spike gains priority (2-draw mean sd
  still ≈ 0.32). Formal H2 close-out + P4 launch when av_L01_05_k{0,1} land (queue on
  av_L01_05_k0's A leg, ETA ~12:00). Full analysis in reorientation.md §P2 avar interim.
  GPU busy — no launches this tick.
- **2026-07-11 09:15 — PROBE QUEUE COMPLETE (rescue+avar, all rc=0); 09:22 — P4 LAUNCHED.**
  Final avar rows: **av_L01_05_k0 = clean A → imit-B cos 0.480/jerk 12.5 WARN** — matches m05's
  best raw draw (0.488/9.8); the P1 "trainability-hostile" class for L01_05 is contradicted by
  one uncensored draw. av_L01_05_k1 = iter-0 A collapse (raw tallies: m05 1/3, L01_05 2/4 —
  design-dependence unresolved, keep descriptive). **Formal H2: step-5 middle branch** —
  per-draw cos sd 0.3–0.5, gate-invisible (k0/k1), design-overlapping (L01_05≈m05) ⇒ single
  draws are not measurements; score designs on mean cos over replicated full draws. **P1
  REVISION recorded** (reorientation.md §P2 avar CLOSED): pick-up rescue stands 5/5; the
  reorient class map is draw-dominated (only the ~4% joint-static counterpoint survives);
  probe-suite conclusion = the policy DRAW is THE landscape bottleneck (user's intuition,
  full depth). **P4 global12x2 RUNNING** (`--morph-set global --n 12 --seed 2 --replicas 2
  --tag global12x2 --b-recipe imit --a-attempts 2`, launched 09:22, ETA ~44 h): replica-major
  → complete r0/n=1 map ~20 h, then r1; waiter armed (r0-complete ≥12 records / crash / DONE;
  NOTE: after the r0 event, re-arm with DONE-only — the ≥12 condition stays true). Analysis on
  events: pool `_r0/_r1` (mean/max cos, collapse count), `morph_pipeline_plots.py --tag
  global12x2` (group by stripping the `_r\d` suffix). **Standing idle-tick CPU task:** A-side
  predictor of B fate from the ~20 accumulated (A scorecard, B outcome) pairs →
  `docs/notes/a_quality_predictor.md` (a predictive gate would restore single-draw eval).
- **2026-07-11 ~10:50 tick — P4 healthy (design 1/24 G02_00_r0: CEM lift 0.052, A first-draw
  clean model_609 objheight 0.1156, imit-B training since 10:20). Standing CPU task DONE:
  A-side predictor analyzed over 26 (A scorecard, B outcome) pairs → NO usable single-draw
  predictor; P4's 2-replica design stands.** Best A metric (mean tip force) rho +0.44 vs B
  held-cos, but within-m05 (5 same-geometry draws) the relation is non-monotone ⇒ the "grip
  richness" trend is the geometry landscape itself, not an evaluator shortcut. Two cheap
  adoptables: (a) idle-finger veto at A-accept (`min(force_mean) < 0.5 N` — the only scored
  B-collapse pair, rs_L01_05, had fmin=0.0/tfmin=0.0); (b) the A scorecard's drop check FAILs
  spuriously on a pre-lift-window min-z artifact (4/26 kept As at minz≈0.007; 3/4 produced
  good Bs) — part of why gate verdicts can't rank draws. Full note
  `docs/notes/a_quality_predictor.md`; rerunnable `scripts/a_quality_predictor.py` (re-run
  with `--tags … global12x2` after P4 → n≈50). Committed avar stragglers (av_L01_05_k0
  video+health, global12x2 table header). GPU busy — no launches this tick.
- **2026-07-11 ~12:45 tick — P4 designs 1–2/24 complete; FIRST FULL-BOX POINT REORIENTS.**
  G02_00_r0: A first-draw clean → imit-B holds AND reorients (cos 0.504 / peak 0.524 /
  jerk 9.3, WARN sliding+over-clamp only) at m05-clean-draw level — despite sitting up to
  3.9 cm from m05 (far outside the ±8 mm local box) with the worst accepted thumb IK
  residual (4.4 mm). G02_01_r0: same clean A/hold profile but static (cos −0.134, peak
  0.147, max delta 4.5 cm). Trainability: 2/2 first-draw-clean As, 0 collapses. Per H2, no
  single-draw conclusions — `_r1` replicas decide; but capability clearly extends beyond
  m05's neighborhood. Pace ~94 min/design ⇒ r0 map ETA ~07-12 early AM, full 24 ~07-12
  late night (ahead of the 44 h estimate). Design 3 (G02_02_r0) on its A leg since 12:36.
  Committed rows + videos + health JSONs + this analysis (reorientation.md §P4 interim).
  Also ran the step-8a CPU spike: **morphology-conditioned policy is FEASIBLE with zero
  mjwarp changes** — all mjwarp kernels read model fields `[worldid % shape[0]]` (batched
  geometry native) and mjlab already ships `Simulation.expand_model_fields()` +
  `recompute_constants()` with CUDA-graph recapture handled. Remaining work is project-side
  plumbing (per-design field scatter from CPU-compiled MjModels, per-world open_ik reset
  table, 9-vector in obs; est. 2–4 days; conditioned-A-before-conditioned-B ordering).
  Full note: `docs/notes/morph_conditioned_policy_spike.md`. GPU busy — no launches this
  tick.
- **2026-07-11 ~17:30 tick — P4 designs 3–5/24 complete (rows 3–5 analyzed; the ~15:00 tick's
  design-3 writeup was in the tree but uncommitted — committed now).** G02_02_r0 = **A-defect
  row** (A t0 idle-index FAIL kept after t1 collapsed; worst index IK residual 6.76 mm; B
  static, cos −0.388) — score it as A-defect, not geometry; its `_r1` is the arbiter.
  G02_03_r0 = **best P4 cos so far (0.568, peak 0.765) from the WORST inputs** — thumb-dead
  CEM grasp (persist 0.00), thumb IK residual 11.06 mm, A t0 collapse; B reorients on a
  2-finger grip, verdict FAIL on idle-thumb only ⇒ second capability-flip-behind-a-FAIL
  (L01_02 pattern; softened flip bar keeps earning its place). This kills the ~15:00
  hypothesis "degraded grasp input ⇒ hostile design" (it survives only as collapse
  propensity). G02_04_r0 = **first clean-sheet PASS of the honest sweeps — and static**
  (cos 0.148, all 6 health checks PASS, jerk 6.5): health⊥capability now demonstrated in
  both directions same-day. r0 tally 5/12: cos {0.504, −0.134, −0.388ᴬᵈᵉᶠ, 0.568, 0.148},
  2/5 ≥ 0.5 ≈ m05's clean-draw hit rate, at 3–4 cm from m05 ⇒ reorienters are NOT rare in
  the full box (pending r1 pooling). A-leg collapses 2/7. Design 6 (G02_05_r0) A training
  since ~17:25; pace ~95 min/design; r0 ETA ~07-12 04:30, full 24 ~07-12 ~23:30. GPU busy —
  analysis/commit only; waiter armed. Full analysis reorientation.md §P4 interim designs 4–5.
- **2026-07-11 ~19:30 tick — P4 design 6/24 complete: G02_05_r0 = second A-defect row; the
  index-IK-residual→idle-index pairing repeats.** Clean CEM grasp (persist 1/1/1, imbal 0.0)
  but the **worst accepted index IK residual yet (12.84 mm)**; A t0 watchdog-collapsed iter 40,
  kept t1 FAILs idle-index (0.9 N) with a thumb+middle clamp; imit-B trains clean, holds
  (min-z 0.105) but static — tail cos −0.499 ≈ A's delivery −0.529 (held unmoved, not
  anti-reoriented), idle thumb+index FAIL. Pairing ledger: index-resid→idle-index now 2/2
  (G02_02 6.76 mm, G02_05 12.84 mm), thumb-resid 11 mm did NOT block (G02_03) — candidate
  asymmetry (index-dead leaves thumb+middle pinch = no rolling pair; thumb-dead leaves
  index+middle gait); n=2 descriptive; possible P5 fix = IK-residual acceptance bar or
  re-CEM on the retargeted keyframe. r0 tally 6/12: cos {0.504, −0.134, −0.388ᴬᵈᵉᶠ, 0.568,
  0.148, −0.499ᴬᵈᵉᶠ}, softened-bar 2/6, A-defect 2/6 (= the two worst index residuals);
  A-leg collapses 3/9. Design 7 (G02_06_r0) A training since 19:03; pace ~96 min/design; r0
  ETA ~07-12 ~04:40, full 24 ~07-13 ~00:00. GPU busy — analysis/commit only; waiter armed.
  Full analysis reorientation.md §P4 interim design 6.
- **2026-07-11 ~21:00 tick — P4 design 7/24 complete: G02_06_r0 = cleanest input row yet,
  still static (third holds-but-static).** CEM clean (persist 1/1/1, imbal 0.0), index IK
  residual 0.09 mm (near-perfect), A first-draw clean — and B holds perfectly (min-z 0.108,
  3 fingers touch-frac 1.00 at 6.7–10.3 N, jerk 10.0 PASS, net drift 0.0 cm) but tail cos
  0.127 (peak 0.397 = attempts a partial reorient, settles back; not a frozen clamp).
  Residual asymmetry repeats in the same direction: thumb resid 9.66 mm harmless here
  (full engagement) vs index resids 6.76/12.84 mm → idle-index A-defects 2/2; ledger now
  thumb-tolerable n=2 / index-harmful n=2, still descriptive. r0 tally 7/12: cos {0.504,
  −0.134, −0.388ᴬᵈᵉᶠ, 0.568, 0.148, −0.499ᴬᵈᵉᶠ, 0.127}, softened-bar 2/7, A-defect 2/7,
  clean-static 3/7; A-leg collapses 3/10; **hold min-z ≥0.105 on 7/7 — pick-up/hold is a
  solved constant of the full box, all variance is the reorient axis.** Design 8
  (G02_07_r0) A training since 20:37; pace ~96 min/design; r0 ETA ~07-12 ~04:40, full 24
  ~07-12 ~23:45. GPU busy — analysis/commit only; waiter armed. Full analysis
  reorientation.md §P4 interim design 7.
- **2026-07-11 ~23:05 tick — P4 design 8/24 complete: G02_07_r0 = first PARTIAL reorienter
  (cos 0.333 / peak 0.493, WARN), from the cleanest inputs of the sweep.** All three IK
  residuals ≤ 0.1 mm (only such row), CEM clean, A first-draw clean — and it's the closest
  LHS point to m05 (max |Δ| 2.18 cm). Sustained partial reorient (tail = 68% of peak, unlike
  G02_06's settle-back), full 3-finger grip (8.1/13.1/11.5 N, touch-frac 1.00), min-z 0.1104,
  jerk PASS; WARNs are sliding (rolling artifact) + over-clamp 10.9 N (highest tip force of
  r0 — grip-richness trend direction). Reading: near-m05 geometry + perfect seeds + clean A
  lands mid-distribution, exactly where m05's own draw spread {0.82, 0.49, −0.16} predicts —
  reinforces H2 (single draws aren't measurements) rather than any input→outcome story.
  r0 tally 8/12: cos {0.504, −0.134, −0.388ᴬᵈᵉᶠ, 0.568, 0.148, −0.499ᴬᵈᵉᶠ, 0.127, 0.333},
  softened-bar 2/8, partial 1/8, A-defect 2/8, clean-static 3/8; A-leg collapses 3/11;
  hold min-z ≥0.105 on 8/8. Design 9 (G02_08_r0) CEM clean (lift 0.050, persist 1/1/1), A
  training since 22:11; pace ~94 min/design; r0 ETA ~07-12 ~04:30, full 24 ~07-12 ~23:30.
  GPU busy — analysis/commit only; waiter armed. Full analysis reorientation.md §P4 interim
  design 8.
- **2026-07-12 ~02:00 tick — P4 designs 9–10/24 complete: G02_08_r0 = flattest clamp of r0
  (peak cos 0.041); G02_09_r0 = third A-defect row, the FIRST idle-THUMB one — the
  residual-asymmetry ledger gets its first split.** G02_08: clean inputs (resid ≤ 3 mm, CEM
  clean, A first-draw), textbook hold (3× touch-frac 1.00, min-z 0.1151) but peak 0.041 —
  never even attempts a reorient; fourth clean-static row. G02_09: thumb resid 6.77 mm →
  A t0 kept-FAIL (t1 collapsed; same kept-FAIL signature as G02_02/G02_05); B holds on a
  thumb-idle 2-finger clamp (1.3 vs 10.8/16.3 N), jerk-FAIL 112.4 (worst of r0), peak 0.342
  settle-back. Ledger: index-harmful stays 2/2; thumb splits (9.66/11.06 mm tolerated vs
  6.77 mm → idle+static) ⇒ residual MAGNITUDE is not the axis, finger identity + draw luck
  are; G02_03 vs G02_09 = same thumb-idle grip topology, opposite outcomes (0.568 vs −0.102)
  — within H2 single-draw spread. r0 tally 10/12: cos {0.504, −0.134, −0.388ᴬᵈᵉᶠ, 0.568,
  0.148, −0.499ᴬᵈᵉᶠ, 0.127, 0.333, −0.099, −0.102ᴬᵈᵉᶠ}, softened-bar 2/10, partial 1/10,
  A-defect 3/10, clean-static 4/10; A-leg collapses 4/14 (+ in-flight G02_10 t0 → 5/15);
  hold min-z ≥0.105 on 10/10 — pick-up solved across the box, all variance reorient-axis.
  Design 11 (G02_10_r0) on A t1 since ~01:30 (t0 watchdog-collapsed); r0 ETA ~07-12 ~06:00,
  full 24 ~07-13 early AM; waiter armed (fires at ≥12 records). GPU busy — analysis/commit
  only. Full analysis reorientation.md §P4 interim designs 9–10.
- **2026-07-12 03:12 — r0 PASS COMPLETE (12/12); r1 pass auto-continuing (G02_00_r1 in CEM).**
  Final two rows: **G02_10_r0** cos 0.117 jerk 36.3 WARN (A t0 collapsed → t1 clean; fifth
  clean-static); **G02_11_r0 = the program's FIRST lift-level best-of-2 total miss** — BOTH A
  draws collapsed at objheight 0.0 (753 s) despite clean CEM (lift 0.055, persist 1/1/1). Watch
  its r1: 4/4 all-collapse ⇒ first genuinely lift-hostile geometry candidate (vs L01_05's
  2-clean-of-5); 1+ clean ⇒ draw luck again. **r0 tally (n=1 pass, NOT rankings — H2):**
  evaluable cos {0.504, −0.134, −0.388ᴬᵈᵉᶠ, 0.568, 0.148, −0.499ᴬᵈᵉᶠ, 0.127, 0.333, −0.099,
  −0.102ᴬᵈᵉᶠ, 0.117}; softened-bar 2/11 (G02_00, G02_03 — both 3–4 cm from m05: reorient
  capability EXISTS far from m05); A-defect 3, clean-static 5, partial 1; A-leg collapses 7/17;
  **every policy that trained held (11/11 min-z ≥ 0.105)** — pick-up solved across the full box
  *when A trains*, G02_11 the lone train-failure. r1 ETA ~07-12 ~22:30 → full-24 analysis:
  pool `_r0/_r1` per design (mean/max cos, collapse count), `morph_pipeline_plots.py --tag
  global12x2`, rank vs m05 draw-band, promote nothing on n=2 without a confirm. Waiter re-armed
  DONE-only. GPU busy (r1) — no launches.
- **2026-07-12 15:53 — machine CRASHED + rebooted (user); sweep RESUMED 15:54 clean.** 21/24
  records survived (per-design checkpointing worked as designed); only G02_09_r1 (killed
  mid-A), G02_10_r1, G02_11_r1 remain (~4.5 h, ETA ~20:30). CUDA healthy post-reboot
  (torch.cuda True), resume correctly skipped all 21; pulse crontab intact (window-capped 2/2,
  resumes next window). Waiter re-armed DONE-only.
- **2026-07-12 ~04:45 tick — r1 pass healthy, design 13/24 (G02_00_r1) on its last B leg.**
  CEM clean (lift 0.052, persist 1/1/1); A ran both attempts — t1 watchdog-collapsed iter 45
  (objheight 0.021), kept t0 model_609 objheight 0.1139 (r0's G02_00 was first-draw clean:
  replica-level draw variance on the same geometry, as H2 predicts). imit-B iter 235/271,
  object_height 0.123, no collapse sentinel — row lands ~04:55. Cumulative A-leg collapses
  8/19. No new rows since the 03:12 r0 close-out; committed the G02_10_r0 video+health
  stragglers. GPU busy — analysis/commit only; waiter armed (DONE-only).
- **2026-07-12 ~07:00 tick — r1 designs 13–16 landed: G02_00 = FIRST replica-consistent
  reorienter (0.504/0.635, both held, jerk 9.3/9.9); G02_03_r1 peaked cos 0.999 then DROPPED;
  G02_01_r1 + G02_02_r1 double-A-collapse where their r0s lifted.** G02_00_r1 WARN cos 0.635
  (min-z 0.1119) on a kept-FAIL A — verdict survives the A draw, top confirm-candidate.
  G02_03_r1 FAIL: thumb-dead seed reached peak cos 0.999 (sweep-program record) on a salvaged
  undertrained model_50 A, then dropped (hold min-z 0.0463, drift 4.2 cm, jerk 229.6) — replica
  pair 0.568-held/0.333-dropped, both replicas ATTEMPT big reorients (peaks 0.678/0.999);
  G02_03 + G02_00 = the only 2 designs with reorient signal in both replicas. G02_01/G02_02 r1
  double-collapses (~12 min each) prove best-of-2 total miss is draw luck, not design fate ⇒
  G02_11 lift-hostility verdict must wait for its r1. Evaluator cost: A-leg aborts 14/26 (54%),
  3/16 legs lost entirely ⇒ argue --a-attempts 3 or collapse-rate-as-output for P5. r1 opened
  7/8 attempts aborted (~1% under r0's 41% if i.i.d.) — flagged, then cleared: G02_04_r1 t0
  (design 17) completed a clean full run this tick; watch whether the rate reverts. ETA full-24
  ~07-13 early AM. GPU busy — analysis/commit only. Full analysis reorientation.md §P4 r0
  close-out + r1 designs 13–16 (also backfills the r0 12/12 close-out there).
- **2026-07-12 ~09:00 tick — r1 design 17 landed: G02_04 flips PASS-static → reorienter, the
  sharpest replica inconsistency yet (r0 cos 0.148 all-PASS / r1 cos 0.528 peak 0.710 WARN).**
  Identical clean inputs (resid 3.7–4.9 mm, CEM 1/1/1), opposite capability verdicts — H2's
  cleanest head-to-head; the r0 "clean-static" class was a draw artifact. Health⊥capability
  inside ONE design: the PASS draw is static, the WARN draw on a kept-FAIL A reorients (second
  reorient on a kept-FAIL A after G02_00_r1). Grip-richness direction repeats (reorienting
  replica: thumb 6.6 N recruited, tip mean 9.2 N vs r0's 1.3 N idle-ish thumb / 4.5 N). New
  evaluator-cost mode: BOTH A attempts completed but BOTH health-FAILed ⇒ best-of-2 spent its
  full budget (8819 s, longest leg) then picked between two FAILs by objheight — the gate can't
  rank draws it can't pass. r1 abort-cluster further cleared (G02_04_r1 0 aborts; r1 attempts
  7/10, cumulative 14/28 = 50%). Reorient-signal designs now G02_00 (both), G02_03 (both
  attempt), G02_04 (r1 only; pooled mean 0.338); hold min-z ≥ 0.105 on 14/14 evaluable. Design
  18 (G02_05_r1 — r0 was A-defect idle-index) A training since 08:33; full-24 ETA ~07-12 late
  night–07-13 early AM. GPU busy — analysis/commit only; waiter armed (DONE-only). Full
  analysis reorientation.md §P4 r1 design 17.
- **2026-07-12 ~12:00 tick — r1 designs 18–19 landed: G02_05_r1 = PROGRAM-BEST reorient
  (cos 0.887 / peak 0.927, jerk 7.8 PASS — first sweep reorienter smoother than the b33
  reference 9.6), on the design r0 scored as A-defect; G02_06 = first replica-consistent
  STATIC design (both legs evaluable, tails 0.127/−0.018).** G02_05_r1: A first-draw clean,
  full 3-finger grip (9.2/10.8/5.7 N), min-z 0.1063, WARN sliding+over-clamp only; beats
  m05's best clean draw 0.82 at 3.2 cm from m05 — existence proof (n=1, H2 forbids ranking),
  joins G02_00 atop the confirm-candidate list. **Index-residual→idle-index pairing BREAKS:**
  same deterministic 12.84 mm residual both replicas, r0 index idle 0.9 N vs r1 index 10.8 N
  ⇒ pairing 2/3, residual exonerated as cause (finger identity + draw luck stand). Replica
  span −0.499→0.887 (Δ1.386, widest) = supersession not contradiction (r0 measured a broken
  A). G02_06_r1: A t0 kept-FAIL after t1 collapse (third kept-FAIL leg, gate spent full
  budget rankless); holds (min-z 0.1036, new sweep floor, ≫0.05) but static + jitter-FAIL
  48.7; both-static pair ~11% likely under m05-equivalence ⇒ weakest reorient candidate,
  not a proven null. Tallies 19/24: reorient-signal G02_00 (both) / G02_03 (both attempt) /
  G02_04 (r1) / G02_05 (r1 best); A-leg aborts 15/31 = 48% (in-flight G02_07_r1 t0 also
  collapsed → 16/32 trending 50%); hold 15/16 evaluable (G02_03_r1 the lone drop). Design 20
  (G02_07_r1, r0 = sustained-partial 0.333) on A t1 since ~11:50; full-24 ETA ~07-12
  ~20:30–22:00. GPU busy — analysis/commit only; waiter armed (DONE-only). Full analysis
  reorientation.md §P4 r1 designs 18–19.
- **2026-07-12 ~14:30 tick — r1 design 20 landed: G02_07 = THIRD replica-consistent design,
  a sustained-partial reorienter (tails 0.333/0.366, Δ 0.033 — tightest replica pair of the
  sweep) — and the smoothest leg yet (ang-jerk 6.6 PASS, below G02_05_r1's 7.8 and b33's
  9.6).** A t0 collapsed → t1 clean (textbook best-of-2 rescue, not a kept-FAIL). The
  replica-consistent set now spans the outcome axis: G02_00 reorienter (0.504/0.635) /
  G02_07 partial (0.333/0.366) / G02_06 static (0.127/−0.018) ⇒ at n=2 the evaluator CAN
  resolve some designs — while G02_04 (Δ0.380) and G02_05 (Δ1.386) prove it can't resolve
  all, and the class isn't knowable in advance. Agreement came from DIFFERENT grips (r0
  hard-clamp 8–13 N jerk 11.7 vs r1 light 3.8–9 N jerk 6.6): convergent capability, not a
  repeated policy — the number belongs to the geometry. Cleanest-inputs design (resid
  0.09–0.10 mm) stays sub-bar both replicas: input cleanliness neither predicts (design 7)
  nor caps (G02_05, 12.84 mm → 0.887) reorient capability. Tallies 20/24: A-leg aborts
  16/33 (48%); hold 16/17 evaluable (min-z ≥ 0.1036). Design 21 (G02_08_r1, r0
  flattest-clamp −0.099) A training since ~13:25; then G02_09/G02_10/G02_11_r1 (the
  lift-hostility arbiter). Full-24 ETA ~07-12 ~20:30–22:00. GPU busy — analysis/commit
  only; waiter armed (DONE-only). Full analysis reorientation.md §P4 r1 design 20.
- **2026-07-12 ~17:20 tick — r1 design 21 landed: G02_08 = FOURTH replica-consistent design,
  second consistent STATIC (tails −0.099/0.074, peaks ≤ 0.11) — and the only design so far
  whose A passed FIRST-draw in BOTH replicas (0 aborts/2 draws vs ~48% program abort rate).**
  r1 holds clean (min-z 0.1098, 3 fingers 6.4–8.7 N, jerk 13.8 PASS) but never reorients.
  Convergent-verdict-from-different-grips repeats (r0 thumb-heavy clamp 12.2 N jerk-WARN vs
  r1 balanced jerk-PASS): grip style is a draw property, the capability tail is geometry.
  New decoupling: easiest-trainability design is capability-static — trainability, health,
  input cleanliness, and reorient capability are now ALL pairwise-decoupled axes. Consistent
  set: G02_00 reorienter / G02_07 partial / G02_06+G02_08 static = 4 of 7 both-evaluable
  designs resolve at n=2 (G02_04/G02_05 the proven exceptions). Tallies 21/24: A-leg aborts
  16/34 (47%); hold 17/18 evaluable. Reboot 15:53 already logged; sweep resumed clean. In
  flight: G02_09_r1 — t0 completed but gate-rejected, t1 training since ~16:56; then
  G02_10_r1, G02_11_r1 (lift-hostility arbiter). Full-24 ETA ~07-12 ~21:30–23:00. GPU busy —
  analysis/commit only; waiter armed (DONE-only). Full analysis reorientation.md §P4 r1
  design 21.
- **2026-07-12 ~19:00 tick — r1 design 22 landed: G02_09 = FIFTH replica-consistent design,
  third consistent STATIC (tails −0.102/0.149, peaks 0.342/0.204) — and the only design
  whose every completed A draw is health-FAIL (3/3 + 1 abort; the anti-G02_08).** r1 spent
  the full best-of-2 budget on two FAIL-grade As (second both-FAIL leg after G02_04_r0;
  9076 s = longest leg) and still held (min-z 0.1141) on a NEW failure flavor: loose
  intermittent low-force grip (touch-frac 0.42–0.56, 4.1 N mean = lowest-force evaluable
  leg, jerk 10.2 PASS). Fourth convergent-verdict-from-different-grips instance, the
  starkest: r0 hard 2-finger clamp jerk-112 vs r1 light 3-finger juggle — same static
  verdict. Residual→idle-finger pairing degrades again: same 6.77 mm thumb residual, r0
  idle-thumb vs r1 weakest-is-index. Trainability axis SPLITS: collapse propensity ⊥
  delivery health (G02_09 abort-normal/health-hostile; G02_08 the clean pole) ⇒ P5 should
  emit per-design health-FAIL rate alongside collapse rate. Consistent set = G02_00
  reorienter / G02_07 partial / G02_06+G02_08+G02_09 static ⇒ 5 of 8 both-evaluable resolve
  at n=2. Tallies 22/24: A-leg aborts 16/36 (44%); hold 18/19 evaluable (min-z ≥ 0.1036).
  In flight: G02_10_r1 A t0 since 18:30 (r0 WARN 0.117 via abort→WARN); then G02_11_r1 =
  lift-hostility arbiter. Full-24 ETA ~20:15–23:00. GPU busy — analysis/commit only; waiter
  armed (DONE-only). Full analysis reorientation.md §P4 r1 design 22.
- **2026-07-12 ~20:45 tick — r1 design 23 landed: G02_10 = THIRD irresolvable-at-n=2 design
  (tails 0.117→0.576, Δ0.459) — and the first replica flip between SAME-grade A draws (both
  kept As health-WARN, objheight 0.1131/0.1166).** r1 is a genuine sustained reorienter
  (peak 0.657, min-z 0.1167, 3 fingers 5.9–13.0 N) failing on jitter alone (ang-jerk 44.5 vs
  bar 40) — near-twin of P1's rs_L01_02 (0.561/44.1), the reorients-but-jitters signature
  again. G02_04's flip rode WARN→FAIL-A (inverted — the FAIL A fed the better B), G02_05's
  rode broken→clean; G02_10 removes the excuse entirely: with imit-B sd 0.02 on a FIXED A,
  the Δ0.459 must live in delivery-state differences between same-grade draws that the
  scorecard grade doesn't see ⇒ A health grade = gate, NOT a sufficient statistic of
  delivery; P5 ranking needs capability probes on the delivered state (or A-draw pooling),
  not better A grading. Ledger at 9 both-evaluable: consistent 5 (G02_00/G02_07/
  G02_06+G02_08+G02_09) / irresolvable 3 (G02_04 Δ0.380, G02_05 Δ1.386, G02_10 Δ0.459) /
  ambiguous 1 (G02_03) — a third of designs can't be binned at n=2, and the irresolvable
  class keeps producing the sweep's best reorients. Reorient-signal census: 6 of 10
  any-evaluable designs attempt in ≥1 replica ⇒ capability is common, EXPRESSION is
  draw-gated — the landscape's real observable is fraction-of-draws-that-express. Tallies
  23/24: A-leg aborts 16/37 (43%); hold 19/20 evaluable. In flight: G02_11_r1 A t0 since
  20:05 = lift-hostility arbiter (r0 0/2 lifts; abort×2 ⇒ first consistent never-lift,
  lift ⇒ total-miss=draw-luck confirmed). GPU busy — analysis/commit only; waiter armed
  (DONE-only). Full analysis reorientation.md §P4 r1 design 23.
- **2026-07-12 22:13 — P4 global12x2 COMPLETE (24/24); 22:19 — CONFIRM r2/r3 LAUNCHED for
  G02_00 + G02_05 (same tag/store via `--replicas 4 --only …`, 4 runs, ETA ~04:40).**
  G02_11_r1 arbitrated: lifted + held 0.1197 + cos 0.445 ⇒ **no lift-hostile geometry in the
  box; pick-up/hold solved everywhere** (min-z ≥ 0.103 on all 20 policy legs). Census: 5/12
  replica-consistent (G02_00 reorienter, G02_07 partial, G02_06/G02_08/G02_09 static), 3/12
  irresolvable at n=2 (Δ 0.38–1.39; contains the best draws incl. G02_05's 0.887), rest
  luck-censored. Pooled table `docs/experiments/MORPH_PIPELINE_global12x2_POOLED.md`; figs
  `img/morph_pipeline_global12x2_*.png`; synthesis reorientation.md §P4 COMPLETE. NOTE:
  `MORPH_PIPELINE_best_center.json` now points at G02_05 (plots side-effect) — do NOT chain
  `--center best` off a single draw. **On confirm completion:** n=4 bands for both candidates
  → if means ≥0.5 hold, head-to-head vs m05 with matched draws before promotion; else the
  irresolvable verdict extends and the **morphology-conditioned policy build is the default
  next move (user decision pending)**. Waiter re-armed (DONE reappears when confirm ends).
- **2026-07-13 ~00:00 tick — confirm leg 1/4 landed: G02_00_r2 = cos 0.107 (peak 0.159) —
  the sweep's only "replica-consistent reorienter" BREAKS at n=3 (draws 0.504/0.635/0.107);
  the consistent bin was itself draw luck.** A passed first-draw (WARN, objh 0.110); B holds
  clean (min-z 0.1108, 3×100% touch) but never attempts — the static draw is also the
  design's hardest clamp (thumb 18.0 N, mean tip 12.1 N over-clamp WARN, vs lighter grips on
  both expressing draws). THIRD same-grade-A flip datum (r0 WARN 0.504 vs r2 WARN 0.107,
  Δ0.397) after G02_10 — A-grade-not-a-sufficient-statistic hardens. Fallout: the 5/12
  consistent census is optimistic by construction (consistency-at-n=2 is a draw-luck
  observable); P(express|design) framing strengthens (G02_00 expresses 2/3). Confirm-bar
  math: G02_00 mean 0.415 over 3 ⇒ needs r3 ≥ 0.754 for the ≥0.5 promotion bar — trending
  toward irresolvable-verdict-extends (⇒ conditioned-policy default). Hold streak 21/21
  legs ≥ 0.103. In flight: G02_05_r2 A t0 since 23:57 (leg ETA ~01:30, batch ETA ~04:40).
  GPU busy — analysis/commit only; waiter armed (DONE-only). Full analysis
  reorientation.md §P4 confirm leg 1/4.
- **2026-07-13 ~02:00 tick — confirm leg 2/4 landed: G02_05_r2 = cos −0.079 (peak 0.046,
  fully static) — the program-best design's ≥0.5 promotion bar is now MATHEMATICALLY
  unreachable (draws −0.499/0.887/−0.079, mean 0.103 ⇒ r3 would need 1.69 > 1).** A passed
  first-draw (WARN, objh 0.1064); B holds clean (min-z 0.107, 3×100% touch, jerk 13.9
  PASS) but never attempts. FOURTH same-grade-A flip and the LARGEST (r1 vs r2 both
  WARN first-draw As, objh 0.106 both, Δcos 0.966) ⇒ A-grade-not-a-sufficient-statistic
  is settled. With leg 1's G02_00 needing r3 ≥ 0.754 (above all 3 observed draws), BOTH
  candidates now need a better-than-any-observed draw ⇒ head-to-head-vs-m05 branch
  effectively dead; **irresolvable-verdict-extends ⇒ conditioned-policy default is all but
  confirmed** (r3 legs now measure bands/expression, not promotion). Within-design note:
  expression tracks index recruitment 3/3 in G02_05 (index 0.9 N static / 10.8 N cos 0.887 /
  3.9 N static; same 12.84 mm residual) — the design's load-bearing finger is the index and
  draws flip on whether they recruit it. Expression census: G02_00 2/3, G02_05 1/3. Hold
  streak 22/22 legs ≥ 0.103. In flight: G02_00_r3 A t0 since 01:32; G02_05_r3 last; batch
  ETA ~04:40. GPU busy — analysis/commit only; waiter armed (DONE-only). Full analysis
  reorientation.md §P4 confirm leg 2/4.
- **2026-07-13 ~05:00 tick — confirm leg 3/4 landed: G02_00_r3 = cos 0.681 (peak 0.734,
  jerk 9.8 PASS) — the design's BEST draw, and still 0.018 short: n=4 mean 0.482 < 0.5 ⇒
  the confirm question is formally CLOSED, both candidates miss, irresolvable-verdict-extends
  ⇒ conditioned-policy build = default next (user decision pending).** Leg detail: both A
  attempts completed and both health-FAILed (third both-FAIL leg; 8904 s); B holds
  (min-z 0.1156) and reorients on an index-led light grip (6.7 N/1.00 touch vs thumb 3.8/
  middle 2.4 intermittent); verdict FAIL on idle_finger alone = fourth
  capability-behind-a-FAIL row. **A-grade inversion inside one design:** G02_00's WARN As
  → cos 0.504/0.107, FAIL As → 0.635/0.681 — the grade anti-orders outcomes at n=4.
  **Grip coin is design-specific:** G02_00's static draw = its hardest clamp (thumb 18 N,
  over-clamp WARN) and its best = its lightest grip (tip mean 4.3 N); G02_05's coin was
  index recruitment ⇒ draw picks a grip style, grip style decides expression, the deciding
  axis varies per design (no program-wide scalar predictor). Band read: G02_00 = m05-class
  at 3.9 cm (mean 0.482 vs m05 0.383), expresses 3/4 = best expression fraction in the
  program. Hold streak 23/23 legs ≥ 0.103. In flight: G02_05_r3 (last leg) — A t0
  completed-FAIL → t1 training since ~04:56 (objh 0.1106 healthy); batch ETA ~07:30; its
  row = band estimation only. On DONE: n=4 close-out + program synthesis, GPU free. GPU
  busy — analysis/commit only; waiter armed (DONE-only). Full analysis reorientation.md
  §P4 confirm leg 3/4.
- **2026-07-13 06:09 — CONFIRM COMPLETE (4/4) → PROBE+P4 PROGRAM CLOSED; GPU FREE.**
  G02_05_r3 = cos 0.532 / jerk 44.9 FAIL (A t0 completed-FAIL → t1; expressing draw with a
  thrashy grip; min-z 0.0903 — first sub-0.103 hold of the program, still ≫ 0.05 held-bar).
  **Final n=4 bands: G02_00 {0.504, 0.635, 0.107, 0.681} mean 0.482, expresses 3/4;
  G02_05 {−0.499, 0.887, −0.079, 0.532} mean 0.210, expresses 2/4; m05 reference {0.82,
  0.49, −0.16} mean 0.383.** Verdict per the pre-registered bar: NO promotion — G02_00 is
  m05-class (its 3/4 expression fraction is the program's best; a real design 3.9 cm from
  m05 that matches the reference), G02_05 stays the program-best-single-draw wide-band
  design. Program synthesis in reorientation.md §PROGRAM CLOSE-OUT; pooled table updated.
  **NEXT (user decision):** morphology-conditioned policy build (~2–4 days; spike-verified
  zero-mjwarp-changes) = the default recommendation; alternatives: P5 replication redesign
  (--a-attempts 3, expression-fraction observable, capability probes on delivered state)
  or accept m05 + return to sim2real. **Idle-tick CPU tasks while undecided:** sync the
  probe+P4 arc into webpaper/rl.typ + paper/main.tex appendix (CLAUDE.md three-doc rule);
  render/inspect the G02_00_r3 + G02_05_r1 videos vs b33.
- **2026-07-13 ~08:00 tick — DOC SYNC DONE (idle-tick task 1/2; program stays CLOSED, GPU
  free, no launches).** webpaper/src/rl.typ: appended the compliance-DR closure verdict +
  a new top-level section "The policy-bottleneck program" (probes H1–H3 with the P2 draw
  distribution, A-predictor negative, global12x2 pooled table as a dropdown, n=4 confirm
  bands, close-out + conditioned-policy recommendation); builds clean, HTML verified.
  paper/main.tex: durable-findings paragraph appended to §codesign-results (body) + new
  App. "Policy-bottleneck probes and the global landscape sweep" (label app:bottleneck)
  with probe/sweep detail per the body/appendix split; also fixed FIVE pre-existing compile
  errors while verifying (missing \R \bX \bt \bw macro defs, a \xml typo, and the
  li2024lightninggrasp cite key → the real yin2025lightninggrasp entry appended to
  references.bib from the upstream README) — paper now compiles rc=0 (was rc=1 for a
  while; paper/ is gitignored so these fixes are local-only by design). **Idle-tick task
  2/2 also done — video frame inspection (ffmpeg strips + final-frame crops) of
  G02_00_r3 / G02_05_r1 vs the b33 reference: all three visually consistent with their
  deterministic held-cos.** G02_05_r1's cylinder ends near-vertical in a full 3-finger
  wrap well off the floor (matches 0.887); G02_00_r3 sustains a ~45° tilt on its
  index-led light grip (matches 0.681, cos⁻¹≈47°); b33 shows the familiar slightly-tilted
  vertical (0.82). No floor contact in any final frame; existing sweep videos were
  sufficient (no re-render needed). Both idle-tick close-out tasks complete;
  conditioned-policy build decision still with the user — next tick has no queued work
  beyond that decision.
- **2026-07-13 ~10:00 tick — no-op confirmation; program stays CLOSED, no launches.** Decision
  tree re-run top to bottom: no morph worker (`pgrep` clean), both `PROBE_QUEUE.DONE` +
  `MORPH_PIPELINE_global12x2.DONE` present, working tree clean (only the pre-existing
  `external/mujoco_warp` submodule-pointer diff, untouched). All step-8 idle-GPU fallback CPU
  tasks are already done+committed (doc sync 6f94dbe, A-predictor note, video inspection d5a8f0e,
  sweep comparison videos c3ce813). **GPU note for future ticks:** ~9.2 GB is in use, but by
  UNRELATED external ROS jobs (`sam_server` + `graspgen_server` under `/home/code/core_ws`, PIDs
  423341/423345) — NOT our workers; do not misread that as an in-flight sweep, and note the GPU is
  effectively NOT free (~7 GB headroom, another team's processes) should the conditioned-policy
  build be greenlit. Only remaining move = the morphology-conditioned policy build, which needs the
  user's go-ahead (2–4 day GPU job) — not launched autonomously. No STATUS/reorientation.md content
  change beyond this bullet.
- **2026-07-14 ~00:00 tick — no-op confirmation across the day boundary; program stays CLOSED, no
  launches.** Decision tree re-run: no morph worker (`pgrep` clean — only the pulse process), both
  `PROBE_QUEUE.DONE` + `MORPH_PIPELINE_global12x2.DONE` present, working tree clean (only the
  pre-existing `external/mujoco_warp` submodule-pointer diff). All step-8 idle-GPU CPU tasks remain
  done+committed (doc sync 6f94dbe, A-predictor note, video inspection d5a8f0e, comparison grids
  c3ce813, prior no-op ec9caf3). GPU still NOT free: ~7.4 GB held by the unrelated ROS jobs
  (`graspgen_server`/`sam_server`, PIDs 423341/423345 under another team) — not our workers. Only
  remaining move = the morphology-conditioned policy build (2–4 day GPU job), awaiting the user's
  go-ahead — not launched autonomously. No reorientation.md content change beyond this bullet.
- **2026-07-14 ~14:00 tick — no-op confirmation; program stays CLOSED, no launches.** Decision tree
  re-run top to bottom: no morph worker (`pgrep -f "[m]orph_pipeline_sweep|[r]l_train_cube|[p]robe_queue"`
  clean — only the pulse process), both `PROBE_QUEUE.DONE` + `MORPH_PIPELINE_global12x2.DONE` present,
  working tree clean (only the pre-existing `external/mujoco_warp` submodule-pointer diff). All step-8
  idle-GPU CPU tasks remain done+committed (doc sync 6f94dbe, A-predictor note, video inspection
  d5a8f0e, comparison grids c3ce813, prior no-ops ec9caf3/18c6255). GPU still NOT free: ~7.4 GB held
  by the unrelated ROS jobs (`graspgen_server`/`sam_server`, PIDs 423341/423345, ~4.5+2.9 GB) — not
  our workers. Only remaining move = the morphology-conditioned policy build (2–4 day GPU job),
  awaiting the user's go-ahead — not launched autonomously. No reorientation.md content change beyond
  this bullet.
- **2026-07-14 ~19:00 tick — no-op confirmation; program stays CLOSED, no launches. GPU now FREE.**
  Decision tree re-run top to bottom: no morph worker (`pgrep -f "[m]orph_pipeline_sweep|[r]l_train_cube|[p]robe_queue"`
  matches only the pulse process), both `PROBE_QUEUE.DONE` + `MORPH_PIPELINE_global12x2.DONE` present,
  working tree clean (only the pre-existing `external/mujoco_warp` submodule-pointer diff). All step-8
  idle-GPU CPU tasks remain done+committed (doc sync 6f94dbe, A-predictor note, video inspection
  d5a8f0e, comparison grids c3ce813, prior no-ops ec9caf3/18c6255/b62f6b5). **CHANGE from prior ticks:
  the unrelated ROS jobs have exited — GPU is now effectively free (1.0 GB / 16 GB used, only
  gnome-remote-desktop-daemon).** This removes the earlier headroom constraint, but does NOT change the
  decision: the only remaining move = the morphology-conditioned policy build (2–4 day GPU job), which
  per the close-out is a USER decision — not launched autonomously. No reorientation.md content change
  beyond this bullet.
- **2026-07-14 ~22:00 tick — no-op confirmation; program stays CLOSED, no launches. GPU still FREE.**
  Decision tree re-run top to bottom: no morph worker (`pgrep -f "[m]orph_pipeline_sweep|[r]l_train_cube|[p]robe_queue"`
  matches only the pulse process), both `PROBE_QUEUE.DONE` + `MORPH_PIPELINE_global12x2.DONE` present,
  working tree clean (only the pre-existing `external/mujoco_warp` submodule-pointer diff). All step-8
  idle-GPU CPU tasks remain done+committed (doc sync 6f94dbe, A-predictor note, video inspection
  d5a8f0e, comparison grids c3ce813, prior no-ops ec9caf3/18c6255/b62f6b5/837e63f). GPU stays free
  (926 MiB / 16 GB used, ROS jobs still exited). Decision unchanged: the only remaining move = the
  morphology-conditioned policy build (2–4 day GPU job), a USER decision — not launched autonomously.
  No reorientation.md content change beyond this bullet.
- **2026-07-15 ~01:00 tick — no-op confirmation; program stays CLOSED, no launches. GPU FREE.**
  Decision tree re-run top to bottom: no morph worker (`pgrep -f "[m]orph_pipeline_sweep|[r]l_train_cube|[p]robe_queue"`
  matches only the pulse process), both `PROBE_QUEUE.DONE` (07-11 09:15) + `MORPH_PIPELINE_global12x2.DONE`
  (07-13 06:09) present, working tree clean (only the pre-existing `external/mujoco_warp` submodule-pointer
  diff). All step-8 idle-GPU CPU tasks remain done+committed (doc sync 6f94dbe, A-predictor note, video
  inspection d5a8f0e, comparison grids c3ce813, prior no-ops ec9caf3/18c6255/b62f6b5/837e63f/d1af772).
  GPU free (1.3 GB / 16 GB used, only gnome-remote-desktop-daemon; ROS jobs still exited). Decision
  unchanged: the only remaining move = the morphology-conditioned policy build (2–4 day GPU job), a USER
  decision — not launched autonomously. No reorientation.md content change beyond this bullet.
- **2026-07-14 ~22:30 MDT tick — no-op confirmation; program stays CLOSED, no launches. GPU FREE.**
  Decision tree re-run top to bottom: no morph worker (`pgrep -f "[m]orph_pipeline_sweep|[r]l_train_cube|[p]robe_queue"`
  matches only the pulse process), both `PROBE_QUEUE.DONE` (07-11 09:15) + `MORPH_PIPELINE_global12x2.DONE`
  (07-13 06:09, 28 records) present, working tree clean (only the pre-existing `external/mujoco_warp`
  submodule-pointer diff). All step-8 idle-GPU CPU tasks remain done+committed (doc sync 6f94dbe,
  A-predictor note, video inspection d5a8f0e, comparison grids c3ce813, prior no-ops
  ec9caf3/18c6255/b62f6b5/837e63f/d1af772/9270a86). GPU free (1.3 GB / 16 GB used, only
  gnome-remote-desktop-daemon; ROS jobs still exited). Decision unchanged: the only remaining move =
  the morphology-conditioned policy build (2–4 day GPU job), a USER decision — not launched
  autonomously. No reorientation.md content change beyond this bullet.
- **2026-07-15 00:00 MDT tick — no-op confirmation; program stays CLOSED, no launches. GPU FREE.**
  Decision tree re-run top to bottom: no morph worker (`pgrep -f "[m]orph_pipeline_sweep|[r]l_train_cube|[p]robe_queue"`
  matches only the pulse process + own shell), both `PROBE_QUEUE.DONE` (07-11 09:15) +
  `MORPH_PIPELINE_global12x2.DONE` (07-13 06:09) present, working tree clean (only the pre-existing
  `external/mujoco_warp` submodule-pointer diff). All step-8 idle-GPU CPU tasks remain done+committed
  (doc sync 6f94dbe, A-predictor note, video inspection d5a8f0e, comparison grids c3ce813, prior no-ops
  ec9caf3/18c6255/b62f6b5/837e63f/d1af772/9270a86/9c4fa9c). GPU free (1264 MiB / 16 GB used, only
  gnome-remote-desktop-daemon; ROS jobs still exited). Decision unchanged: the only remaining move =
  the morphology-conditioned policy build (2–4 day GPU job), a USER decision — not launched
  autonomously. No reorientation.md content change beyond this bullet.
- **2026-07-15 ~01:30 MDT tick — no-op confirmation; program stays CLOSED, no launches. GPU FREE.**
  Decision tree re-run top to bottom: no morph worker (`pgrep -f "[m]orph_pipeline_sweep|[r]l_train_cube|[p]robe_queue"`
  matches only the pulse process + own shell), both `PROBE_QUEUE.DONE` (07-11 09:15) +
  `MORPH_PIPELINE_global12x2.DONE` (07-13 06:09, 28 records) present, working tree clean (only the
  pre-existing `external/mujoco_warp` submodule-pointer diff). All step-8 idle-GPU CPU tasks remain
  done+committed (doc sync 6f94dbe, A-predictor note, video inspection d5a8f0e, comparison grids
  c3ce813, prior no-ops ec9caf3/18c6255/b62f6b5/837e63f/d1af772/9270a86/9c4fa9c/264d28e). GPU free
  (1172 MiB / 16 GB used, only gnome-remote-desktop-daemon; ROS jobs still exited). Decision
  unchanged: the only remaining move = the morphology-conditioned policy build (2–4 day GPU job), a
  USER decision — not launched autonomously. No reorientation.md content change beyond this bullet.
- **2026-07-15 03:00 MDT tick — no-op confirmation; program stays CLOSED, no launches. GPU FREE.**
  Decision tree re-run top to bottom: no morph worker (`pgrep -f "[m]orph_pipeline_sweep|[r]l_train_cube|[p]robe_queue"`
  matches only the pulse invocation itself), both `PROBE_QUEUE.DONE` (07-11 09:15) +
  `MORPH_PIPELINE_global12x2.DONE` (07-13 06:09, 28 records) present, working tree clean (only the
  pre-existing `external/mujoco_warp` submodule-pointer diff). All step-8 idle-GPU CPU tasks remain
  done+committed (doc sync 6f94dbe, A-predictor note, video inspection d5a8f0e, comparison grids
  c3ce813, prior no-ops ec9caf3/18c6255/b62f6b5/837e63f/d1af772/9270a86/9c4fa9c/264d28e/641ad93). GPU
  free (1172 MiB / 16 GB used, only gnome-remote-desktop-daemon; ROS jobs still exited). Decision
  unchanged: the only remaining move = the morphology-conditioned policy build (2–4 day GPU job), a
  USER decision — not launched autonomously. No reorientation.md content change beyond this bullet.
- **2026-07-15 04:31 MDT tick — no-op confirmation; program stays CLOSED, no launches. GPU FREE.**
  Decision tree re-run top to bottom: no morph worker (`pgrep -f "[m]orph_pipeline_sweep|[r]l_train_cube|[p]robe_queue"`
  matches only the pulse invocation itself), both `PROBE_QUEUE.DONE` (07-11 09:15) +
  `MORPH_PIPELINE_global12x2.DONE` (07-13 06:09, 28 records) present, working tree clean (only the
  pre-existing `external/mujoco_warp` submodule-pointer diff). All step-8 idle-GPU CPU tasks remain
  done+committed (doc sync 6f94dbe, A-predictor note, video inspection d5a8f0e, comparison grids
  c3ce813, prior no-ops ec9caf3/18c6255/b62f6b5/837e63f/d1af772/9270a86/9c4fa9c/264d28e/641ad93/de84c6f).
  GPU free (1170 MiB / 16 GB used, only gnome-remote-desktop-daemon; ROS jobs still exited). Decision
  unchanged: the only remaining move = the morphology-conditioned policy build (2–4 day GPU job), a
  USER decision — not launched autonomously. No reorientation.md content change beyond this bullet.
- **2026-07-15 06:00 MDT tick — no-op confirmation; program stays CLOSED, no launches. GPU FREE.**
  Decision tree re-run top to bottom: no morph worker (`pgrep -f "[m]orph_pipeline_sweep|[r]l_train_cube|[p]robe_queue"`
  matches only the pulse invocation itself), both `PROBE_QUEUE.DONE` (07-11 09:15) +
  `MORPH_PIPELINE_global12x2.DONE` (07-13 06:09, 28 records) present, working tree clean (only the
  pre-existing `external/mujoco_warp` submodule-pointer diff). All step-8 idle-GPU CPU tasks remain
  done+committed (doc sync 6f94dbe, A-predictor note, video inspection d5a8f0e, comparison grids
  c3ce813, prior no-ops ec9caf3/18c6255/b62f6b5/837e63f/d1af772/9270a86/9c4fa9c/264d28e/641ad93/de84c6f/9c8155f).
  GPU free (1173 MiB / 16 GB used, only gnome-remote-desktop-daemon; ROS jobs still exited). Decision
  unchanged: the only remaining move = the morphology-conditioned policy build (2–4 day GPU job), a
  USER decision — not launched autonomously. No reorientation.md content change beyond this bullet.
- **2026-07-15 08:00 MDT tick — no-op confirmation; program stays CLOSED, no launches. GPU FREE.**
  Decision tree re-run top to bottom: no morph worker (`pgrep -f "[m]orph_pipeline_sweep|[r]l_train_cube|[p]robe_queue"`
  matches only the pulse invocation itself), both `PROBE_QUEUE.DONE` (07-11 09:15) +
  `MORPH_PIPELINE_global12x2.DONE` (07-13 06:09, 28 records) present, working tree clean (only the
  pre-existing `external/mujoco_warp` submodule-pointer diff). All step-8 idle-GPU CPU tasks remain
  done+committed (doc sync 6f94dbe, A-predictor note, video inspection d5a8f0e, comparison grids
  c3ce813, prior no-ops ec9caf3/18c6255/b62f6b5/837e63f/d1af772/9270a86/9c4fa9c/264d28e/641ad93/de84c6f/9c8155f/ecab5ea).
  GPU free (1168 MiB / 16 GB used, only gnome-remote-desktop-daemon; ROS jobs still exited). Decision
  unchanged: the only remaining move = the morphology-conditioned policy build (2–4 day GPU job), a
  USER decision — not launched autonomously. No reorientation.md content change beyond this bullet.
- **2026-07-15 09:30 MDT tick — no-op confirmation; program stays CLOSED, no launches. GPU FREE.**
  Decision tree re-run top to bottom: no morph worker (`pgrep -f "[m]orph_pipeline_sweep|[r]l_train_cube|[p]robe_queue"`
  matches only the pulse invocation itself), both `PROBE_QUEUE.DONE` (07-11 09:15) +
  `MORPH_PIPELINE_global12x2.DONE` (07-13 06:09, 28 records) present, working tree clean (only the
  pre-existing `external/mujoco_warp` submodule-pointer diff). All step-8 idle-GPU CPU tasks remain
  done+committed (doc sync 6f94dbe, A-predictor note, video inspection d5a8f0e, comparison grids
  c3ce813, prior no-ops ec9caf3/18c6255/b62f6b5/837e63f/d1af772/9270a86/9c4fa9c/264d28e/641ad93/de84c6f/9c8155f/ecab5ea/7a9a498).
  GPU free (1170 MiB / 16 GB used, only gnome-remote-desktop-daemon; ROS jobs still exited). Decision
  unchanged: the only remaining move = the morphology-conditioned policy build (2–4 day GPU job), a
  USER decision — not launched autonomously. No reorientation.md content change beyond this bullet.
- **2026-07-15 11:00 MDT tick — no-op confirmation; program stays CLOSED, no launches. GPU FREE.**
  Decision tree re-run top to bottom: no morph worker (`pgrep -f "[m]orph_pipeline_sweep|[r]l_train_cube|[p]robe_queue"`
  matches only the pulse invocation itself), both `PROBE_QUEUE.DONE` (07-11 09:15) +
  `MORPH_PIPELINE_global12x2.DONE` (07-13 06:09, 28 records) present, working tree clean (only the
  pre-existing `external/mujoco_warp` submodule-pointer diff). All step-8 idle-GPU CPU tasks remain
  done+committed (doc sync 6f94dbe, A-predictor note, video inspection d5a8f0e, comparison grids
  c3ce813). GPU free (1220 MiB / 16 GB used, only gnome-remote-desktop-daemon; ROS jobs still exited).
  Decision unchanged: the only remaining move = the morphology-conditioned policy build (2–4 day GPU
  job), a USER decision — not launched autonomously. No reorientation.md content change beyond this
  bullet.
- **2026-07-15 14:46 MDT tick — no-op confirmation; program stays CLOSED, no launches. GPU FREE.**
  Decision tree re-run top to bottom: no morph worker (`pgrep -f "[m]orph_pipeline_sweep|[r]l_train_cube|[p]robe_queue"`
  matches only the pulse invocation itself), both `PROBE_QUEUE.DONE` (07-11 09:15) +
  `MORPH_PIPELINE_global12x2.DONE` (07-13 06:09, 28 records) present, working tree clean (only the
  pre-existing `external/mujoco_warp` submodule-pointer diff). All step-8 idle-GPU CPU tasks remain
  done+committed (doc sync 6f94dbe, A-predictor note, video inspection d5a8f0e, comparison grids
  c3ce813). GPU free (1265 MiB / 16 GB used, only gnome-remote-desktop-daemon; ROS jobs still exited).
  Decision unchanged: the only remaining move = the morphology-conditioned policy build (2–4 day GPU
  job), a USER decision — not launched autonomously. No reorientation.md content change beyond this
  bullet.
- **2026-07-15 16:15 MDT tick — no-op confirmation; program stays CLOSED, no launches. GPU FREE.**
  Decision tree re-run top to bottom: no morph worker (`pgrep -f "[m]orph_pipeline_sweep|[r]l_train_cube|[p]robe_queue"`
  matches only the pulse invocation itself), both `PROBE_QUEUE.DONE` (07-11 09:15) +
  `MORPH_PIPELINE_global12x2.DONE` (07-13 06:09, 28 records) present, working tree clean (only the
  pre-existing `external/mujoco_warp` submodule-pointer diff). All step-8 idle-GPU CPU tasks remain
  done+committed (doc sync 6f94dbe, A-predictor note, video inspection d5a8f0e, comparison grids
  c3ce813). GPU free (1944 MiB / 16 GB used, only gnome-remote-desktop-daemon; ROS jobs still exited).
  Decision unchanged: the only remaining move = the morphology-conditioned policy build (2–4 day GPU
  job), a USER decision — not launched autonomously. No reorientation.md content change beyond this
  bullet.
- **2026-07-15 18:01 MDT tick — no-op confirmation; program stays CLOSED, no launches. GPU FREE.**
  (Terse by design — substantive state unchanged from the 16:15 bullet; see it for the full
  decision-tree walk.) No morph worker, `PROBE_QUEUE.DONE`+`MORPH_PIPELINE_global12x2.DONE` present,
  all step-8 idle-GPU tasks done+committed, spike doc `docs/notes/morph_conditioned_policy_spike.md`
  intact. GPU free (1.8 GB used). Working tree clean bar the pre-existing `external/mujoco_warp`
  submodule-pointer diff. Only remaining move = the morphology-conditioned policy build, a USER
  decision — not launched autonomously.
- **2026-07-15 19:30 MDT tick — no-op confirmation; program stays CLOSED, no launches. GPU FREE.**
  (Terse by design — substantive state unchanged from the 16:15 bullet; see it for the full
  decision-tree walk.) Bracketed `pgrep -f "[m]orph_pipeline_sweep|[r]l_train_cube|[p]robe_queue"`
  matches only the pulse invocation itself (its prompt text contains the pattern strings) → no morph
  worker; both `PROBE_QUEUE.DONE` (07-11 09:15) + `MORPH_PIPELINE_global12x2.DONE` (07-13 06:09, 28
  records) present; all step-8 idle-GPU CPU tasks done+committed; spike doc
  `docs/notes/morph_conditioned_policy_spike.md` intact. GPU free (1871 MiB used, only
  gnome-remote-desktop-daemon). Working tree clean bar the pre-existing `external/mujoco_warp`
  submodule-pointer diff. Only remaining move = the morphology-conditioned policy build, a USER
  decision — not launched autonomously.
- **2026-07-15 21:00 MDT tick — no-op confirmation; program stays CLOSED, no launches. GPU FREE.**
  (Terse by design — substantive state unchanged from the 16:15 bullet; see it for the full
  decision-tree walk.) Bracketed `pgrep -f "[m]orph_pipeline_sweep|[r]l_train_cube|[p]robe_queue"`
  matches only the pulse invocation itself (its prompt text contains the pattern strings) → no morph
  worker; both `PROBE_QUEUE.DONE` (07-11 09:15) + `MORPH_PIPELINE_global12x2.DONE` (07-13 06:09, 28
  records) present; all step-8 idle-GPU CPU tasks done+committed; spike doc
  `docs/notes/morph_conditioned_policy_spike.md` intact. GPU free (1532 MiB used, only
  gnome-remote-desktop-daemon). Working tree clean bar the pre-existing `external/mujoco_warp`
  submodule-pointer diff. Only remaining move = the morphology-conditioned policy build, a USER
  decision — not launched autonomously.
- **2026-07-15 23:01 MDT tick — no-op confirmation; program stays CLOSED, no launches. GPU FREE.**
  (Terse by design — substantive state unchanged from the 16:15 bullet; see it for the full
  decision-tree walk.) Bracketed `pgrep -f "[m]orph_pipeline_sweep|[r]l_train_cube|[p]robe_queue"`
  matches only the pulse invocation itself → no morph worker; both `PROBE_QUEUE.DONE` (07-11 09:15)
  + `MORPH_PIPELINE_global12x2.DONE` (07-13 06:09, 28 records) present; POOLED table + n=4 confirm
  close-out intact (G02_00 mean 0.482 = m05-class, misses ≥0.5 bar by 0.018; G02_05 program-best
  single draw 0.887; nothing promoted); spike doc `docs/notes/morph_conditioned_policy_spike.md`
  intact; pulse cron alive (`*/15`). GPU free (1529 MiB used, only gnome-remote-desktop-daemon).
  Working tree clean bar the pre-existing `external/mujoco_warp` submodule-pointer diff. Only
  remaining move = the morphology-conditioned policy build, a USER decision — not launched
  autonomously.
- **2026-07-16 00:30 MDT tick — no-op confirmation across the day boundary; program stays CLOSED,
  no launches. GPU FREE.** (Terse by design — substantive state unchanged from the 16:15 bullet;
  see it for the full decision-tree walk.) Bracketed
  `pgrep -f "[m]orph_pipeline_sweep|[r]l_train_cube|[p]robe_queue"` matches only the pulse invocation
  itself (its prompt text contains the pattern strings) → no morph worker; both `PROBE_QUEUE.DONE`
  (07-11 09:15) + `MORPH_PIPELINE_global12x2.DONE` (07-13 06:09, 28 records) present; POOLED table +
  n=4 confirm close-out intact (G02_00 mean 0.482 = m05-class, misses ≥0.5 bar by 0.018; G02_05
  program-best single draw 0.887; nothing promoted); spike doc
  `docs/notes/morph_conditioned_policy_spike.md` intact; pulse cron alive (`*/15`). GPU free
  (1498 MiB used, only gnome-remote-desktop-daemon; unrelated ROS jobs still exited). Working tree
  clean bar the pre-existing `external/mujoco_warp` submodule-pointer diff. Only remaining move =
  the morphology-conditioned policy build (2–4 day GPU job), a USER decision — not launched
  autonomously.
- **2026-07-16 02:00 MDT tick — no-op confirmation; program stays CLOSED, no launches. GPU FREE.**
  (Terse by design — substantive state unchanged from the 16:15 bullet; see it for the full
  decision-tree walk.) Bracketed `pgrep -f "[m]orph_pipeline_sweep|[r]l_train_cube|[p]robe_queue"`
  matches only the pulse invocation itself → no morph worker; both `PROBE_QUEUE.DONE` (07-11 09:15)
  + `MORPH_PIPELINE_global12x2.DONE` (07-13 06:09, 28 records) present; POOLED table + n=4 confirm
  close-out intact (G02_00 mean 0.482 = m05-class, misses ≥0.5 bar by 0.018; G02_05 program-best
  single draw 0.887; nothing promoted); spike doc `docs/notes/morph_conditioned_policy_spike.md`
  intact; pulse cron alive (`*/15`). GPU free (1588 MiB used, only gnome-remote-desktop-daemon;
  unrelated ROS jobs still exited). Working tree clean bar the pre-existing `external/mujoco_warp`
  submodule-pointer diff. Only remaining move = the morphology-conditioned policy build (2–4 day GPU
  job), a USER decision — not launched autonomously.
- **2026-07-16 04:00 MDT tick — no-op confirmation; program stays CLOSED, no launches. GPU FREE.**
  (Terse by design — substantive state unchanged from the 16:15 bullet; see it for the full
  decision-tree walk.) Bracketed `pgrep -f "[m]orph_pipeline_sweep|[r]l_train_cube|[p]robe_queue"`
  matches only the pulse invocation itself → no morph worker; both `PROBE_QUEUE.DONE` (07-11 09:15)
  + `MORPH_PIPELINE_global12x2.DONE` (07-13 06:09, 28 records) present; POOLED table + n=4 confirm
  close-out intact (G02_00 mean 0.482 = m05-class, misses ≥0.5 bar by 0.018; G02_05 program-best
  single draw 0.887; nothing promoted); spike doc `docs/notes/morph_conditioned_policy_spike.md`
  intact; pulse cron alive (`*/15`). GPU free (1590 MiB used, only gnome-remote-desktop-daemon;
  unrelated ROS jobs still exited). Working tree clean bar the pre-existing `external/mujoco_warp`
  submodule-pointer diff. Only remaining move = the morphology-conditioned policy build (2–4 day GPU
  job), a USER decision — not launched autonomously.
- **2026-07-16 05:30 MDT tick — no-op confirmation; program stays CLOSED, no launches. GPU FREE.**
  (Terse by design — substantive state unchanged from the 16:15 bullet; see it for the full
  decision-tree walk.) Bracketed `pgrep -f "[m]orph_pipeline_sweep|[r]l_train_cube|[p]robe_queue"`
  matches only the pulse invocation itself → no morph worker; both `PROBE_QUEUE.DONE` (07-11 09:15)
  + `MORPH_PIPELINE_global12x2.DONE` (07-13 06:09, 28 records) present; POOLED table + n=4 confirm
  close-out intact (G02_00 mean 0.482 = m05-class, misses ≥0.5 bar by 0.018; G02_05 program-best
  single draw 0.887; nothing promoted); spike doc `docs/notes/morph_conditioned_policy_spike.md`
  intact; pulse cron alive (`*/15`). GPU free (1525 MiB used, only gnome-remote-desktop-daemon;
  unrelated ROS jobs still exited). Working tree clean bar the pre-existing `external/mujoco_warp`
  submodule-pointer diff. Only remaining move = the morphology-conditioned policy build (2–4 day GPU
  job), a USER decision — not launched autonomously.
- **2026-07-16 07:00 MDT tick — no-op confirmation; program stays CLOSED, no launches. GPU FREE.**
  (Terse by design — substantive state unchanged from the 16:15 bullet; see it for the full
  decision-tree walk.) Bracketed `pgrep -f "[m]orph_pipeline_sweep|[r]l_train_cube|[p]robe_queue"`
  matches only the pulse invocation itself → no morph worker; both `PROBE_QUEUE.DONE` (07-11 09:15)
  + `MORPH_PIPELINE_global12x2.DONE` (07-13 06:09, 28 records) present; POOLED table + n=4 confirm
  close-out intact (G02_00 mean 0.482 = m05-class, misses ≥0.5 bar by 0.018; G02_05 program-best
  single draw 0.887; nothing promoted); spike doc `docs/notes/morph_conditioned_policy_spike.md`
  intact; pulse cron alive (`*/15`). GPU free (1526 MiB used, only gnome-remote-desktop-daemon;
  unrelated ROS jobs still exited). Working tree clean bar the pre-existing `external/mujoco_warp`
  submodule-pointer diff. Only remaining move = the morphology-conditioned policy build (2–4 day GPU
  job), a USER decision — not launched autonomously.
- **2026-07-16 09:00 MDT tick — no-op confirmation; program stays CLOSED, no launches. GPU FREE.**
  (Terse by design — substantive state unchanged from the 16:15 bullet; see it for the full
  decision-tree walk.) Bracketed `pgrep -f "[m]orph_pipeline_sweep|[r]l_train_cube|[p]robe_queue"`
  matches only the pulse invocation itself → no morph worker; both `PROBE_QUEUE.DONE` (07-11 09:15)
  + `MORPH_PIPELINE_global12x2.DONE` (07-13 06:09, 28 records) present; POOLED table + n=4 confirm
  close-out intact (G02_00 mean 0.482 = m05-class, misses ≥0.5 bar by 0.018; G02_05 program-best
  single draw 0.887; nothing promoted); spike doc `docs/notes/morph_conditioned_policy_spike.md`
  intact; pulse cron alive (`*/15`). GPU free (1518 MiB used, only gnome-remote-desktop-daemon;
  unrelated ROS jobs still exited). Working tree clean bar the pre-existing `external/mujoco_warp`
  submodule-pointer diff. Only remaining move = the morphology-conditioned policy build (2–4 day GPU
  job), a USER decision — not launched autonomously.
