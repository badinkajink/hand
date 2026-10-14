Status: done
Task (owner, 2026-10-13): publish the new page on the scale-study turn under HOM and sampling MPC ("let's spin out these experiments ... please make sure to include video renders and beautiful figures to understand these methods and their relative performance in the html docpage(s)"). This job edits no page or builder.
Read first: docs/experiments/20261013-scale_turn_controllers/20261013-scale_turn_hom_mpc.html (new, no artifact yet; builder scripts/scale_turn_controllers_page.py); docs/experiments/INDEX.md (its row says "publish job pending"); the Artifact tool's rules: read every page's content before publishing it.
State (2026-10-13): page built and committed by the window; four three-panel films (HOM, MPC, three-pad MPC) inline as data URIs; 4 figures with charts, 3 tables, one algorithm block, LaTeX as inline SVG. Published 2026-10-13: https://claude.ai/artifact/JdCgjBHdcqu99dPgRwv1Vo.
<!-- ap:steps: rendered by ~/.claude/autopilot/ap from its ledger; change it with `ap step`, not by hand -->
Steps, in order (the ledger; `ap step done <job> <id> "result"` marks one, `ap step add <job> "text"` adds one):
- [x] 1. Review the page: no "pending" or "running" text; every figure renders; the four films decode and play; chart … Done 10-13 18:06: no pending text; 47 svg figures, 4 films decode (h264 1920x480, 157 frames), labels readable at 828 px
- [x] 2. Publish docs/experiments/20261013-scale_turn_controllers/20261013-scale_turn_hom_mpc.html as a new artifact (… Done 10-13 18:06: published https://claude.ai/artifact/JdCgjBHdcqu99dPgRwv1Vo; URL file and INDEX row updated
- [x] 3. Commit the URL changes and push to origin main. Done 10-13 18:06: committed 879561fe, pushed to origin main
<!-- /ap:steps -->
Claimed: docs/experiments/INDEX.md (artifact column of the 2026-10-13 scale-turn row), docs/experiments/20261013-scale_turn_controllers/artifact_url.txt.
Waiting on owner: none for this job. Owner decisions named on the page: three-pad MPC at placements 1 and 2 (about 3.6 h on 12 threads); the SGS policy over the cells (GPU, multi-hour).
Do not: edit pages, builders or templates; publish other pages; delete any artifact; force-push.
