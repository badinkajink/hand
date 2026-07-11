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
