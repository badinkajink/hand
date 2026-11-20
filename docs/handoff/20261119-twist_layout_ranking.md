Status: active
Task: (owner, 2026-11-19 17:2x) "re-running the full task space search, let's say for just the sphere twist, with HOM
... deploying as many morphologies (and top-N controls) on M trials, and seeing how well our real results match with
sim"; the search "definitely get this done overnight, no matter what", and be "ready to fully evaluate starting tomorrow,
in real, how well simulation rankings hold up". Also: "one final valiant pass at improving simulation, improving (closed
loop) controls & stability/smoothness/accuracy of contact sensing" (per-control-step gap over contact model, pose filter
and servo model).
Read first: docs/experiments/20261119-twist_layout_ranking/sim/20261119-twist_ranking_search.py (docstring: model,
stages, rows, heartbeat); logs/20261119-twist_layout_ranking/lane.sh and lane.log; the 11-17 model it reuses,
docs/experiments/20261117-sr2_system_identification/QUEUE.md "W3, A/B" (blind predictions 0.437 vs the hand's 0.447);
docs/experiments/20261115-hom_bench_trials/README.md (how trials run); memory project_sr2_identification_night_2026_11_17.md.
State: 18:07. Three detached lanes in logs/20261119-twist_layout_ranking/: lane.sh (stages A-D, then the report;
stage A relaunched once at 17:55 after a heartbeat race, since fixed; A ends ~19:10, D ~20:20), lane2.sh (waits for
LANE_DONE: stage E HOM forces, then sim/20261119-trial_plans.py writes plans 20261119-rank_* into
docs/experiments/20261115-hom_bench_trials/plans/, then the page), lane3.sh (waits for LANE2_DONE: the film
media/20261119-simulated_twist_three_layouts.mp4, then the page again). The window gap is done
(logs/.../gap/, GAP_DONE; data/20261119-segment_gap.json is written by the page builder). Scripts committed e17f3492b;
the film script and the page's film block are not committed yet. The lab (corvus) runs nothing on the hand tonight; the
200 g dead-weight mass still hangs from the index fingertip (lab, 18:0x).
<!-- ap:steps: rendered by ~/.claude/autopilot/ap from its ledger; change it with `ap step`, not by hand -->
Steps, in order (the ledger; `ap step done <job> <id> "result"` marks one, `ap step add <job> "text"` adds one):
- [ ] 1. Check the CPU lane: lane.log, heartbeat.json (rewritten after every run), LANE_DONE / LANE_STOPPED / FAILED_<stage>,
    and ~/.claude/resguard_events.jsonl. If no lane.sh process runs (pgrep -f "[l]ane.sh") and neither LANE_DONE nor
    LANE_STOPPED exists, relaunch: cd ~/Programs/hand && nohup setsid logs/20261119-twist_layout_ranking/lane.sh >
    /dev/null 2>&1 < /dev/null & (stages resume from their rows). A FAILED stage: read the last rows' "error", fix the
    cause in the driver, rerun that stage alone with the lane's command line, then the later stages.
- [ ] 2. When LANE_DONE exists: run the report (sim/20261119-twist_ranking_report.py) if lane.log says it was skipped, build
    the page, commit the study folder's sim/ and data/ files by path (never git commit -a) and push (fetch, merge
    origin/main with GIT_LFS_SKIP_SMUDGE=1, push).
- [ ] 3. When logs/20261119-twist_layout_ranking/LANE3_DONE exists (or LANE2_DONE with lane3 dead: run lane3.sh's two commands by hand): look at the page docs/experiments/20261119-twist_layout_ranking/20261119-sphere_twist_layout_ranking.html (extract its figures and read them; read the film's still media/20261119-simulated_twist_three_layouts.png), run python3 scripts/docs/prose_lint.py --brief on it and fix the builder's worst sentences if above target, rebuild. Commit by path: the study's sim/ data/ media/ and the page, scripts/hom/twist_ranking_page.py, the plans docs/experiments/20261115-hom_bench_trials/plans/20261119-rank_*, the docs index files the build rewrote; push (fetch, GIT_LFS_SKIP_SMUDGE=1 merge, push).
- [ ] 4. Publish the page as an artifact (Artifact tool, icon chart; films as files via python3 scripts/docs/page_films.py files PAGE), then rebuild so INDEX.md takes the URL, commit, push. Then SendMessage to 'Physical model vs GP calibration' (the corvus lab): the plans list data/20261119-trial_plans.json with predicted tracking per plan, the proposal (three trials per plan, random order of layouts, tips recorded, both cameras), and that the predictions sit in each plan's meta.ranking, written before any trial. Write the outcome into this note's State for the owner.
<!-- /ap:steps -->
Claimed: docs/experiments/20261119-twist_layout_ranking/, logs/20261119-twist_layout_ranking/, scripts/hom/twist_ranking*
Waiting on owner: the trial count for 11-20 (proposed: about 10 layouts spread over the predicted range, 2 grasps each,
3 trials per pair); switching tip_contacts.OWN_POSE to the (70, 110) GP of 11-19.
Do not: run anything on the hand or the UR5; start a GPU job while a GPU lane runs; change the trials' defaults.
