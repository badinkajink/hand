You are continuing an in-hand reorientation RL research project autonomously, with bypass
permissions, on the user's machine. You ARE authorized to run GPU training, edit code, and
`git commit` (LOCAL only — never push). Work in `/home/humanoid/Programs/hand` on `main`.

# FIRST: load context (you have none — read these)
1. `RESEARCH_STATE.md` (repo root) — current state, in-progress runs, eval commands, GOTCHAS.
2. `docs/rl/reorientation.md` — full chronological log (read at least the "CRITICAL CORRECTION"
   and the v2.1/handoff/bracing sections).
3. `STATE_HANDOFF_RESULTS.txt` (repo root) — the just-finished P1/P2/P3 metrics this trigger wrote.
4. Skim `scripts/rl_eval_reorient_metrics.py`, `scripts/rl_demo_handoff_continuous.py`,
   `scripts/rl_train_cube.py` (the knobs).

# HARD RULES (the project has been bitten by all of these)
- **Commit after EVERY change** (`git add -A && git commit`, footer
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`). NEVER `git push`.
  If a tracked file looks truncated/stale (the SSH "revert gremlin"), `git checkout HEAD -- <file>`.
- **Launch ALL training DETACHED:** `nohup setsid bash <script> >log 2>&1 </dev/null & disown`.
- **Parallel:** per-process `WARP_CACHE_PATH=$(mktemp -d)`; stagger ~60-90s. VRAM is cheap
  (3×3072 ≈ 11 GB/50%); PUSH THROUGHPUT — use 4096 envs and/or 3-4 parallel runs.
- **`--warmstart-critic` is default ON; keep it.** Warmstart from a stable base.
- **Judge on `scripts/rl_eval_reorient_metrics.py` (deterministic held-cos / jerk / min_z / drop),
  NOT training reward sums.** Render videos to sanity-check (videos -> docs/rl/videos/reorient/).
- **Add ONE new constraint at a time** — stacking DR+lateral+brace on the fragile finger-only
  reorient diverges (proven repeatedly). 
- Bounded autonomy: at most ~2 sweep generations (this assessment + one iteration), then
  synthesize and stop. Don't spin indefinitely or chase diverged runs (kill them).

# TASKS
1. **Assess P1/P2/P3** (from STATE_HANDOFF_RESULTS.txt + your own `rl_eval_reorient_metrics.py`
   runs + handoff tests via `rl_demo_handoff_continuous.py`). Decide, per the honest metrics:
   - P1 (handoff curriculum-DR only): does B hold A's handoff (min-z>0.05)? at what verticality cost?
   - P2 (gentle lateral only): did xy-drift drop vs signed+critic (5.1 cm) WITHOUT collapsing grip/verticality?
   - P3 (state bank): does B hold the handoff best? (early drops were lowest.)
   Pick the best handoff policy and the best de-centering approach.
2. **Iterate ONE generation** based on the assessment, e.g.: combine the winning handoff approach
   with the gentle lateral (carefully, one-at-a-time staged or curriculum'd if it destabilizes);
   or push the better of P1/P3 longer; or, if P3 (state bank) wins, refine it (more states, add
   the lateral gently). Launch the new run(s) DETACHED, high-throughput, warmstarting a stable base.
   Run `rl_record_handoff_states.py` again if you change Policy A or the lift height.
3. **Produce the seamless A→B video** with the best handoff policy
   (`rl_demo_handoff_continuous.py`), confirm min-z>0.05 (holds), save to docs/rl/videos/reorient/.
4. **DOCUMENT + SYNTHESIZE:** update `docs/rl/reorientation.md` with a new dated section covering
   P1/P2/P3 results, your iteration, and the seamless-handoff outcome; refresh `RESEARCH_STATE.md`
   TL;DR + in-progress list; regenerate the comparison plot if useful
   (`scripts/rl_plot_reorient.py`, add new runs to V2_RUNS). Update the project memory file
   `/home/humanoid/.claude/projects/-home-humanoid-Programs-hand/memory/project_policyB_v2_overnight.md`
   if the recommended policy changes. Commit each.
5. If you launched a new sweep, you may set up a follow-on trigger for it (mirror
   `scripts/overnight_iterate_trigger.sh`) so the loop continues; otherwise write a clear
   `STATE_HANDOFF_RESULTS.txt`-style summary of where things stand and STOP.

Be rigorous and HONEST (report regressions/divergences plainly; the user values that over
optimistic spin). When done, print "ITERATE RUN COMPLETE".
