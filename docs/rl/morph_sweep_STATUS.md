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
   on the `MORPH_PIPELINE_initial8.DONE` sentinel re-invokes the session on completion (or crash).
2. On completion → **analysis**: `morph_pipeline_plots.py --tag initial8` (summary + training
   figures + markdown table), render a comparison of the best handoffs, write up in
   `reorientation.md`, update memory.
3. → **launch larger sweep** (`--morph-set local --n 16 --center best`), same waiter pattern.
4. On completion → analysis + docs again.

## Monitor / resume / intervene

```bash
cd /home/humanoid/Programs/hand
# progress (one line per finished design):
cat MORPH_PIPELINE_initial8.txt
tail -f sweep_initial8.run.log            # live stage markers ([HH:MM:SS] <id>: ...)
python3 -c "import json;print(len(json.load(open('MORPH_PIPELINE_initial8.json'))),'designs done')"
nvidia-smi                                 # is a trainer running?

# RESUME after any crash/kill (skips finished designs — safe to re-run):
MUJOCO_GL=egl uv run --extra rl --extra gpu \
  python scripts/morph_pipeline_sweep.py --morph-set initial8 > sweep_initial8.run.log 2>&1 &

# STOP everything:
pkill -f morph_pipeline_sweep.py ; pkill -f rl_train_cube.py   # (never pkill from inside its own cmd)
```

Outputs: `MORPH_PIPELINE_<tag>.{json,txt}`, `sweep_{A,B}_<id>.trainer.log`, run dirs
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
  `morph_pipeline_initial8_summary.png` / `..._training.png` / `MORPH_PIPELINE_initial8_TABLE.md`;
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
  design from luck.** Analysis: `morph_pipeline_large16_*.png`, `MORPH_PIPELINE_large16_TABLE.md`;
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

## FINAL STATE (2026-07-05)

Nothing running, GPU idle. Deliverables from this autonomous run:
- **Cleanup + naming:** a10/b33 canonical, REGISTRY idempotent, `*.COLLAPSED` gitignored.
- **Pipeline:** `scripts/morph_pipeline_sweep.py` (health-gated per-design A→B, resumable) +
  `morph_pipeline_plots.py`; 3 bugs found+fixed (B open-finger, final-vs-early ckpt, no warmstart).
- **Sweeps:** initial8 (fragility→bugs), large16 (16 designs ranked), confirm (6-run seed test).
  Data: `MORPH_PIPELINE_{initial8,large16,confirm}.{json,txt}`, `*_TABLE.md`, figs in `docs/rl/img/`,
  handoff videos in `docs/rl/videos/reorient/sweep/`.
- **Result:** honest NEGATIVE — no local design beats m05 within seed noise; the win is the
  pipeline + variance characterization. m05 (a10→b33) remains the reference.
