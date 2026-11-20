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
State: 17:41 the CPU lane (lane.sh, detached, resguard 7G, cores 10-18, 8 workers) started stage A: 15,424 runs, about
3 runs/s. Stages B-D follow in the same lane, then the report script if it exists. Rows logs/20261119-twist_layout_ranking/
rows_<stage>.jsonl (fsynced, resumable). The lab (corvus) runs nothing on the hand tonight (17:3x).
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
<!-- /ap:steps -->
Claimed: docs/experiments/20261119-twist_layout_ranking/, logs/20261119-twist_layout_ranking/, scripts/hom/twist_ranking*
Waiting on owner: the trial count for 11-20 (proposed: about 10 layouts spread over the predicted range, 2 grasps each,
3 trials per pair); switching tip_contacts.OWN_POSE to the (70, 110) GP of 11-19.
Do not: run anything on the hand or the UR5; start a GPU job while a GPU lane runs; change the trials' defaults.
