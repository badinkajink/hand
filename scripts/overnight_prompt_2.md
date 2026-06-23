You are finalizing an overnight in-hand reorientation RL experiment autonomously, with
bypass permissions. Authorized to edit code and `git commit` (local only) on `main` in
`/home/humanoid/Programs/hand`. Do NOT `git push`, do NOT delete results/docs/external,
do NOT touch files outside the repo. Commit footer:
`Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

# Context
"Policy B v2 Stage 2" has now FINISHED (a bash orchestrator waited for it). Stage 2
warmstarted the best Stage-1 checkpoint and added the "quick / shorter trajectory"
mechanisms (alignment-success termination + bonus, per-step time cost, early-crossing
speed bonus) on top of the smoothness ramp + signed progress. Run dirs match
results/rl/*-policyB_v2_smooth5x_quick and *-policyB_v2_smooth10x_quick. Log-derived
metrics: `STAGE2_RESULTS.txt`. Logs: policyB_v2_5x.log, policyB_v2_10x.log (these were
reused for Stage 2; check timestamps / tags inside). Stage-1 assessment + the
"Phase Policy B v2 — Stage 1" section are already in docs/rl/reorientation.md.

Reference — Policy B v1 (results/rl/b01_20260601-1033-policyB_v1): mean_reward 402,
target_axis_alignment 87, target_axis_progress +0.25, tip_lost 1.17, object_ang_acc_l2
~-2.5, action_rate_l2 ~-0.13.

# Tasks
1. Assess Stage 2 honestly: did the quick mechanisms make reorientation FASTER
   (time-to-first-alignment, alignment_success rate, shorter episodes / earlier success
   termination) and reduce slip-back, WHILE keeping grip (tip_lost low) and the Stage-1
   smoothness? Report regressions plainly. Pick the recommended final Policy B v2 config.
2. Render verification videos for the Stage-2 runs via `scripts/rl_demo_handoff.py`
   (adapt as in Stage 1); save under docs/rl/videos/reorient/. Confirm real in-hand
   reorientation, smoother + holding vertical vs v1.
3. Append a "## Phase Policy B v2 — Stage 2" section to docs/rl/reorientation.md (metrics
   table v1 vs v2-stage1 vs v2-stage2, video links, takeaway, recommended config).
4. Regenerate the comparison figure: `uv run --extra rl python scripts/rl_plot_reorient.py`
   (extend it to include the v2 runs if needed). Commit.
5. Write `OVERNIGHT_SUMMARY.md` at repo root: what ran, headline metrics table
   (v1 / v2-stage1-best / v2-stage2-best), the recommended Policy B v2 config + checkpoint
   path, what you changed and why, and concrete next steps for the user. Commit.

When done, print "OVERNIGHT RUN COMPLETE" and stop.
