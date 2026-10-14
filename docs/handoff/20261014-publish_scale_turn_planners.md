Status: done
Task (owner, 2026-10-13): "can we update the docpage with better, more stable videos (as well as a planner comparison?)". The window that built it republishes nothing (max seat); this job republishes the updated page to its existing artifact. It edits no page or builder.
Read first: docs/experiments/20261013-scale_turn_controllers/20261013-scale_turn_hom_mpc.html (built 2026-10-14 by scripts/scale_turn_controllers_page.py, commit named in State); docs/experiments/INDEX.md (its 2026-10-13 scale-turn row, URL https://claude.ai/artifact/JdCgjBHdcqu99dPgRwv1Vo); the Artifact tool's rules: read the artifact before publishing to it, and read every page's content before publishing it.
State (2026-10-14 01:00): page rebuilt with the section "Sampling planners compared on four cells" (Tables 3-5, Figures 5-7, Algorithm 2, Eqs. 4-5), the pad and GPU cost tables, four new 2x2 films (HOM, predictive sampling, pinned predictive sampling, pinned iCEM; 1280x960, 157 frames) and one earlier three-panel film; 5.97 MB, media inline as data URIs.
<!-- ap:steps: rendered by ~/.claude/autopilot/ap from its ledger; change it with `ap step`, not by hand -->
Steps, in order (the ledger; `ap step done <job> <id> "result"` marks one, `ap step add <job> "text"` adds one):
- [x] 1. Review the page: no "pending" or "running" text; every figure renders; the five films decode and play; chart … Done 10-14 14:56: Page reviewed: no pending text, 73 inline SVG figures render, 5 h264 films decode (157 frames), labels readable at 900 px
- [x] 2. Republish docs/experiments/20261013-scale_turn_controllers/20261013-scale_turn_hom_mpc.html to https://claude… Done 10-14 14:56: Republished to JdCgjBHdcqu99dPgRwv1Vo as version 2
- [x] 3. Commit any change to the INDEX row or the URL file and push to origin main. Done 10-14 14:56: INDEX row and artifact_url.txt already carry the URL; nothing to commit, nothing unpushed
Standing directions (for every step):
- Owner-side note 2026-10-14 14:55: the six launches of 04:03-06:48 died before Claude started (the tick's tmux server was killed with the oneshot service; fixed in aplib.launch by a systemd-run scope). Nothing in the note changed; publish as written.
<!-- /ap:steps -->
Claimed: docs/experiments/INDEX.md (artifact column of the 2026-10-13 scale-turn row), docs/experiments/20261013-scale_turn_controllers/artifact_url.txt.
Waiting on owner: none for this job. Owner decisions named on the page: pinned iCEM at every cell and placement (about 4 h), the iLQR test, the plant-mismatch rerun, the SGS policy over the cells.
Do not: edit pages, builders or templates; publish other pages; delete any artifact; force-push.
