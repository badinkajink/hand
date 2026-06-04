You are finishing an in-hand reorientation RL research thread autonomously, bypass perms, on
the user's machine (`/home/humanoid/Programs/hand`, branch `main`). Commit LOCAL only — NEVER
push. This run is a DOCUMENT-AND-SYNTHESIZE pass — do NOT launch another training sweep
(bounded autonomy budget is spent). Read `RESEARCH_STATE.md` and the latest dated section of
`docs/rl/reorientation.md` first for full context.

# What just happened
`scripts/v3b_eval_trigger.sh` waited for the two v3b grace-window normal-lift Policy B runs
(`policyB_normallift_v3b_repro`, `policyB_normallift_v3b_soft`) to finish and wrote fresh
metrics to `STATE_HANDOFF_RESULTS.txt`: deterministic held-cos/jerk/min_z/drop, and the
continuous A→B handoff hold (post-handoff min-z; **hold ⇔ min-z > 0.05**) for each variant,
with handoff videos at `docs/rl/videos/reorient/handoff_v3b_{repro,soft}.mp4`.

The goal of v3b: B trained in the **normal-lift env with a grace window** (hold from step 35,
reorient from step 50) should HOLD A's delivery through the seam (min-z > 0.05) while keeping
held-cos near P2's 0.988. (v2-no-grace collapsed; v3 grace held reward flat ~10 but NaN'd at
iter 60/750; v3b reran it to completion, NaN-resilient via 2 parallel variants.)

# Tasks
1. **Read `STATE_HANDOFF_RESULTS.txt`.** If a checkpoint is missing (a variant NaN'd again),
   note it honestly. Re-run any eval yourself if a number looks off
   (`scripts/rl_eval_reorient_metrics.py`, `scripts/rl_demo_handoff_continuous.py
   --handoff-step 40 --blend-steps 8`).
2. **Decide the verdict, HONESTLY:**
   - Did either v3b variant achieve the **seamless handoff** (min-z > 0.05 post-handoff)? At
     what held-cos cost vs P2's 0.988? Repro vs soft — which is better, and why?
   - If BOTH still drop: say so plainly. The likely remaining cause is that even the grace
     window can't make the skip-lift-prior warmstart robust to the seam in 40M ts; candidate
     next steps to document (do NOT run): longer training, warmstart the hold-only control
     instead of P2, or add the handoff state-bank (P3) to the normal-lift env.
3. **Pick + name the seamless-handoff policy** if one holds; render/keep its
   `handoff_v3b_*.mp4` as the headline seamless A→B video (confirm min-z > 0.05).
4. **DOCUMENT:** append a dated subsection to the "Phase: de-centering + seamless A→B handoff"
   section of `docs/rl/reorientation.md` with the v3b outcome; refresh `RESEARCH_STATE.md`
   TL;DR + the "Normal-lift B history" block; update the memory file
   `/home/humanoid/.claude/projects/-home-humanoid-Programs-hand/memory/project_policyB_v2_overnight.md`
   if the recommended/seamless-handoff policy changed. Regenerate the comparison plot if useful
   (`scripts/rl_plot_reorient.py`, add v3b to V2_RUNS). **Commit after each change** (footer
   `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`, never push).
5. Write a clear `STATE_HANDOFF_RESULTS.txt`-style summary of where the handoff thread stands
   (solved / partially / open + the single best next experiment) and STOP. Do not launch
   training or set up another trigger.

GOTCHAS (don't relearn): judge on deterministic held-cos + handoff min-z, NOT reward sums;
commit after every change (SSH revert gremlin — if a tracked file looks stale,
`git checkout HEAD -- <file>`); the warmstart critic must be ON. When done print
"V3B DOCUMENT RUN COMPLETE".
