You are continuing an in-hand reorientation RL project autonomously, overnight, with
bypass permissions, on the user's research machine. You ARE authorized to run GPU
training, edit code, and `git commit` (local only) to this repo. Work on `main`, in
`/home/humanoid/Programs/hand`.

# Hard guardrails
- Launch EVERY training run fully detached so it survives SSH/laptop disconnect:
  `nohup setsid bash <script> > <log> 2>&1 < /dev/null & disown`.
- Do NOT `git push` (local commits only). Do NOT delete anything under results/, docs/,
  external/. Do NOT touch files outside the repo. Do NOT force anything.
- 16 GB GPU fits exactly two 1024-env runs. Confirm GPU is free with `nvidia-smi` before
  launching, and only one sweep (two runs) at a time.
- Commit after each logical step; message footer:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
- This is a BOUNDED job. Do Task A then Task B, then STOP (do NOT wait for Stage 2 to
  finish — a bash orchestrator handles that and will start a second Claude run).

# Context
"Policy B v2 Stage 1" just finished: it finetuned Policy B v1 (the breakthrough in-hand
reorient policy) with ONLY a smoothness-weight ramp (action_rate_l2 + object_ang_acc_l2,
ramp iters 200->600) + signed target_axis_progress (slip-back penalty). NO "quick"
mechanisms. Two runs swept the smoothness target:
- 5x:  run dir matching results/rl/*-policyB_v2_smooth5x   (NOT the _quick suffix)
- 10x: run dir matching results/rl/*-policyB_v2_smooth10x
Log-derived metrics: `STAGE1_RESULTS.txt`. Logs: policyB_v2_5x.log, policyB_v2_10x.log.
Read `docs/rl/reorientation.md` for full history. Reference — Policy B v1 converged
(results/rl/20260601-1033-policyB_v1): mean_reward 402, target_axis_alignment 87,
target_axis_progress +0.25, tip_lost 1.17/iter, object_ang_acc_l2 sum ~-2.5,
action_rate_l2 ~-0.13. v2 goal: SMOOTHER + holds-vertical, without wrecking rotation/grip.

Key files: trainer scripts/rl_train_cube.py; env src/morphohand/rl/env_cfg.py; reward/term/
curriculum src/morphohand/rl/mjlab_terms.py; sweep launcher
scripts/queue_reorient_smooth_sweep.sh (env knobs QUICK, TAG_SUFFIX, WARMSTART, TOTAL_TS,
SMOKE). Tests: `env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MUJOCO_GL=egl uv run
--extra rl --extra gpu --with pytest python -m pytest tests/test_rl_*.py -q`.

# Task A — Assess Stage 1
1. Read STAGE1_RESULTS.txt + both logs. Compare at convergence (iter >600, after ramp):
   object_ang_acc_l2 & action_rate_l2 (smoothness; want lower-magnitude jitter than v1),
   target_axis_alignment & target_axis_progress (rotation; stay high), tip_lost (grip;
   ~v1's 1.17). 
2. Render a verification video for BOTH 5x and 10x. The correct tool for skip-lift
   reorient policies is the handoff demo `scripts/rl_demo_handoff.py` (how v1 videos were
   made) — read it and adapt. `rl_eval_object.py` does NOT take skip-lift flags, so it is
   wrong unless you make it load the run's config.yaml. Save videos under
   docs/rl/videos/reorient/ and confirm they show real in-hand reorientation (no floor
   contact).
3. Decide which smoothness level (5x vs 10x) is the better smoothness-vs-rotation
   tradeoff. Add a "## Phase Policy B v2 — Stage 1" section to docs/rl/reorientation.md
   (metrics table vs v1 + video links + takeaway). Commit.

# Task B — Launch Stage 2 (quick mechanisms), then STOP
The earlier run that enabled all quick mechanisms from iter 0 destabilized grip
(tip_lost spiked to ~22/iter). So:
1. Choose the better Stage-1 checkpoint (late/converged: final, or best-alignment with
   low tip_lost).
2. Inspect the QUICK_ARGS block in scripts/queue_reorient_smooth_sweep.sh (success_bonus
   50, time_cost -0.05, speed_bonus 20). If your Stage-1 grip read suggests these are too
   aggressive, SOFTEN them and document why. Consider setting the smoothness base weights
   (--action-rate-weight / --object-ang-acc-weight in COMMON_ARGS) to the Stage-1 FINAL
   values so the already-smooth warmstart does not regress; document any edit.
3. Launch Stage 2 DETACHED:
   `WARMSTART=<best_stage1_ckpt> QUICK=1 TAG_SUFFIX=_quick nohup setsid bash scripts/queue_reorient_smooth_sweep.sh > stage2_sweep.log 2>&1 < /dev/null & disown`
4. Wait ~3 min, then verify BOTH Stage-2 runs are compiling/training (pgrep -af
   rl_train_cube; check stage2_sweep.log shows run A then run B). If only one launched or
   it errored, fix and relaunch. Commit any code edits.
5. Print "STAGE2 LAUNCHED" and STOP. Do not poll Stage 2 to completion.
