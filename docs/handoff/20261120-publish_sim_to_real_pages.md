Status: done
Task (owner, 2026-11-20 11:5x): "deliver me a wonderful chart showing whatever sim2real landscape/ranking correspondence
we've determined". After the light goes (about 21:00), refresh the sim-to-real page from the day's last trials, publish it
and the controller rematch page from the personal login, and send the owner one notification with both local paths and links.
Read first: docs/notes/20261120-twist_ranking_decisions.md (the day's decisions, newest last);
docs/experiments/20261120-twist_sim_to_real/20261120-twist_ranking_sim_to_real.html (builder scripts/hom/twist_sim_to_real_page.py);
docs/experiments/20261120-controller_rematch/20261120-controllers_tuned_forces_calibrated_plant.html (builder scripts/hom/controller_rematch_page.py);
CLAUDE.md "Result pages"; ~/.claude/CLAUDE.md "Artifacts are never the only copy".
State (2026-11-20 21:0x, autopilot on personal): steps 1-4 done. Sim-to-real page
https://claude.ai/artifact/9kwvgF1pN6cvPGqBrcQQMf with both films (served as video/mp4, byte-identical to media/, 420
frames decode); controller rematch page https://claude.ai/artifact/M15KPLJ9VkqdheGMVye486 (the 13:58 build). Two
terminal notifications; no phone push (Remote Control off). For the owner, in the decision log's "Waiting on owner": the
sim-to-real lede says all 12 wide layouts lost the sphere (8 did), and Figure 4's labels overlap.
State (2026-11-20 20:4x, hand-4a on colorado): both pages final for the day and pushed (9edb54880). The sim-to-real
page was corrected at 20:4x after the owner's note that the sphere is covered in grip tape (the page had said it was
bare): friction 0.4 is now an effective fit, the plants section gives the fit per contact model and for narrow and wide
layouts, and the jitter section has Table 3 (commands by force). It has 134 trials on 56 plans (rho 0.58), two films in
its media/ (films.json) and the bench events; the bench ended at 19:49. Neither page has an artifact yet. The refresh
script logs/20261120-controller_rematch/refresh_s2r.sh pulls corvus's records and rows, scores, counts the bench events
and rebuilds the sim-to-real page (log refresh_s2r.log next to it). Overnight lanes (lane_mu04 on the GPU, lane_f1mu04
on the CPU) are not this job's.
<!-- ap:steps: rendered by ~/.claude/autopilot/ap from its ledger; change it with `ap step`, not by hand -->
Steps, in order (the ledger; `ap step done <job> <id> "result"` marks one, `ap step add <job> "text"` adds one):
- [x] 1. Run `bash logs/20261120-controller_rematch/refresh_s2r.sh` from the repository, read the last lines of logs/2… Done 11-20 20:58: refresh at 20:56 clean (0.70 MB, prose within targets); no trials after 20:03 (134 on 56 plans); committed 2dc618d8f and pushed
- [x] 2. Publish docs/experiments/20261120-twist_sim_to_real/20261120-twist_ranking_sim_to_real.html as a new claude.a… Done 11-20 21:00: https://claude.ai/artifact/9kwvgF1pN6cvPGqBrcQQMf with both films (served as video/mp4, byte-identical, 420 frames decode; no browser here to watch playback); artifact_url.txt, INDEX row; 3b9dc9363 pushed
- [x] 3. Publish docs/experiments/20261120-controller_rematch/20261120-controllers_tuned_forces_calibrated_plant.html … Done 11-20 21:01: https://claude.ai/artifact/M15KPLJ9VkqdheGMVye486 (the 13:58 build, published as is); artifact_url.txt, INDEX row, two overviews; eb7823176 pushed
- [x] 4. Send the owner one PushNotification naming both local paths and both links (under 200 characters each, or two… Done 11-20 21:02: two terminal notifications (no phone push: Remote Control off); decision-log line with both URLs and a Waiting-on-owner item on the sim-to-real lede (8 of 12 wide layouts lost the sphere, not 12) and Figure 4's labels; 5cf913651 pushed
Standing directions (for every step):
- Publish the sim-to-real page with its two films: pass the output of 'python3 scripts/docs/page_films.py files docs/experiments/20261120-twist_sim_to_real/20261120-twist_ranking_sim_to_real.html' as the publish's files, and check both films play in the artifact.
<!-- /ap:steps -->
Claimed: docs/experiments/20261120-twist_sim_to_real/artifact_url.txt, docs/experiments/20261120-controller_rematch/artifact_url.txt, both pages' artifacts, the artifact column of their INDEX.md rows.
Waiting on owner: none.
Do not: edit either builder or page by hand; start, stop or change anything on corvus, the hand or the HOM servers; publish from the Max seat; create a second artifact for either page; push data files over 2 MB; force-push.
