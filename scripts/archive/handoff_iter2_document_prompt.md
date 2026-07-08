You are resuming the in-hand reorientation **A→B handoff** work autonomously, in a fresh
context. The training you were waiting on has finished and been evaluated; the numbers are
in `STATE_HANDOFF_RESULTS.txt`. Your job is to **assess + document**, NOT to launch new
training (compute budget is spent for this session).

## Read first (in order)
1. `STATE_HANDOFF_RESULTS.txt` — the latest eval of B12 / B13 (held-cos / obj_jerk / standalone, plus continuous-handoff min-z, plus rendered video paths).
2. `RESEARCH_STATE.md` — the living handoff doc.
3. `webpaper/src/rl.typ` — the polished write-up; the "policy-switching" section is where this work lives.
4. `webpaper/README.md` — the Typst-HTML authoring gotchas (READ before editing any `.typ`).

## Context (what is true going in)
- The two-policy split is A (lift) → B (reorient). The seam between them was the open problem.
- **The hold-only warmstart fixed the seam collapse.** `B10` (grace window, warmstarted from the
  v3 hold-only control) is the FIRST policy to both survive the A→B handoff AND reorient
  (standalone held-cos **0.977**) — but it is **VIOLENT at the transition** (obj_jerk **108** vs
  the skip-lift reference B4's **27**). `B11` (soft, residual 0.4, alpha curriculum over 150
  iters) holds but **never tilts** (held-cos −0.137): too dilute.
- This iteration's two runs target that:
  - **B12** = smoothness finetune *of B10* (ramp action-rate + ang-accel penalties in late,
    curriculum start iter 40) — "learn it then smooth it", the recipe that worked for the v2
    smoothness finetunes (B2). Goal: keep the tilt, cut the jerk.
  - **B13** = soft-but-**committing** onset from hold-only (residual 0.5, alpha 0.5→4 over just
    40 iters) — between B10 (too hard) and B11 (too soft).
- **Judge on deterministic metrics, never reward sums.** The decisive ones: continuous-handoff
  **min-z > 0.05** (B survives A's delivery), **held-cos** near B4's 0.988 (it actually reaches
  vertical), and **obj_jerk DOWN from 108** (the seam is less violent). The standalone
  `rl_eval_reorient_metrics` env is SKIP-LIFT, which is OOD for normal-lift-trained Bs, so its
  `drop=1.0` is an artifact — trust the continuous-handoff render + min-z for survival, and
  held-cos/jerk for quality/violence.

## Your task
1. **Assess** B12 and B13 from `STATE_HANDOFF_RESULTS.txt`. Which (if any) gives
   survive-the-seam + reorient + *lower jerk*? Watch the rendered videos' paths are recorded.
2. **Document** into existing files (the user wants sections appended, NOT new doc files):
   - In `webpaper/src/rl.typ`, under the handoff material, add/extend a subsection with the
     B10/B11/B12/B13 results table and the verdict. Match the house style: inline math as
     Unicode (never bare `$...$` in prose), display math as `$ ... $`, `#det(...)` collapsibles,
     `#media("assets/<file>.mp4", ...)` for any rendered video you copy into `webpaper/src/assets/`,
     `#callout(...)` for the takeaway. **Escape prose underscores** and avoid the markup hazards in
     `webpaper/README.md`. Then `bash webpaper/build.sh` and fix any Typst compile errors until all
     four pages build clean.
   - Update `RESEARCH_STATE.md` (handoff section + the B-registry: B12/B13).
3. **Commit** (do not push): `git add` the changed `.typ`/`.md`/new assets and commit with
   `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Do NOT stage `external/mujoco_warp`.
4. If neither B12 nor B13 crosses min-z > 0.05 at acceptable jerk, write the **single best next
   experiment** (one paragraph) into RESEARCH_STATE.md and STOP. Candidates worth considering:
   deploy-time action low-pass / blend-window (branch E, no retrain — check whether the rendered
   blend variants already tamed B10's jerk), or branch B (un-freeze Policy A and fine-tune it
   toward B's basin via terminal-state regularization / B's value as reward).

Be concise, honest, and leave the repo building clean.
