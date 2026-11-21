Status: active
Task (owner, 2026-11-20 11:5x): "deliver me a wonderful chart showing whatever sim2real landscape/ranking correspondence
we've determined". After the light goes (about 21:00), refresh the sim-to-real page from the day's last trials, publish it
and the controller rematch page from the personal login, and send the owner one notification with both local paths and links.
Read first: docs/notes/20261120-twist_ranking_decisions.md (the day's decisions, newest last);
docs/experiments/20261120-twist_sim_to_real/20261120-twist_ranking_sim_to_real.html (builder scripts/hom/twist_sim_to_real_page.py);
docs/experiments/20261120-controller_rematch/20261120-controllers_tuned_forces_calibrated_plant.html (builder scripts/hom/controller_rematch_page.py);
CLAUDE.md "Result pages"; ~/.claude/CLAUDE.md "Artifacts are never the only copy".
State (2026-11-20 20:2x, hand-4a on colorado): both pages final for the day and pushed (d4ae823be). The sim-to-real
page has 134 trials on 56 plans (rho 0.58), the friction fit (0.4: r 0.72, mean absolute error 0.105), two films in its
media/ (films.json) and the bench events; the bench ended at 19:49. Neither page has an artifact yet. The refresh script
logs/20261120-controller_rematch/refresh_s2r.sh pulls corvus's records and rows, scores, counts the bench events and
rebuilds the sim-to-real page (log refresh_s2r.log next to it). Overnight lanes (lane_mu04 on the GPU, lane_f1mu04 on the
CPU) are not this job's.
<!-- ap:steps: rendered by ~/.claude/autopilot/ap from its ledger; change it with `ap step`, not by hand -->
Steps, in order (the ledger; `ap step done <job> <id> "result"` marks one, `ap step add <job> "text"` adds one):
- [ ] 1. Run `bash logs/20261120-controller_rematch/refresh_s2r.sh` from the repository, read the last lines of logs/20261120-controller_rematch/refresh_s2r.log (no Traceback; a page size and a prose line), then commit by path (never `git commit -a`): the page folder, docs/experiments/20261119-twist_layout_ranking/data/20261120-trials_vs_prediction.json, docs/experiments/20261115-hom_bench_trials/analysis/20261120-recovery_metrics.json, the new runs_corvus/*_corvus.jsonl and series rows, docs/experiments/INDEX.md, docs/changelog.html, docs/index.html, docs/pages.html, docs/pages.json and docs/overviews/morphology_design.html; push (pull --no-rebase and push again when refused).
- [ ] 2. Publish docs/experiments/20261120-twist_sim_to_real/20261120-twist_ranking_sim_to_real.html as a new claude.ai artifact (icon chart), read the whole file first; write its URL to docs/experiments/20261120-twist_sim_to_real/artifact_url.txt; run `python3 scripts/docs/docs_index.py`; commit and push as in step 1.
- [ ] 3. Publish docs/experiments/20261120-controller_rematch/20261120-controllers_tuned_forces_calibrated_plant.html the same way (artifact_url.txt in its folder); docs_index.py; commit; push.
- [ ] 4. Send the owner one PushNotification naming both local paths and both links (under 200 characters each, or two notifications at most), and add a line with both URLs to docs/notes/20261120-twist_ranking_decisions.md under its last entry; commit; push.
Standing directions (for every step):
- Publish the sim-to-real page with its two films: pass the output of 'python3 scripts/docs/page_films.py files docs/experiments/20261120-twist_sim_to_real/20261120-twist_ranking_sim_to_real.html' as the publish's files, and check both films play in the artifact.
<!-- /ap:steps -->
Claimed: docs/experiments/20261120-twist_sim_to_real/artifact_url.txt, docs/experiments/20261120-controller_rematch/artifact_url.txt, both pages' artifacts, the artifact column of their INDEX.md rows.
Waiting on owner: none.
Do not: edit either builder or page by hand; start, stop or change anything on corvus, the hand or the HOM servers; publish from the Max seat; create a second artifact for either page; push data files over 2 MB; force-push.
