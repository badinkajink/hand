Status: armed
Task (owner, 2026-10-13): publish the new page on the scale-study turn under HOM and sampling MPC ("let's spin out these experiments ... please make sure to include video renders and beautiful figures to understand these methods and their relative performance in the html docpage(s)"). This job edits no page or builder.
Read first: docs/experiments/20261013-scale_turn_controllers/20261013-scale_turn_hom_mpc.html (new, no artifact yet; builder scripts/scale_turn_controllers_page.py); docs/experiments/INDEX.md (its row says "publish job pending"); the Artifact tool's rules: read every page's content before publishing it.
State (2026-10-13): page built and committed by the window; four three-panel films (HOM, MPC, three-pad MPC) inline as data URIs; 4 figures with charts, 3 tables, one algorithm block, LaTeX as inline SVG.
<!-- ap:steps: rendered by ~/.claude/autopilot/ap from its ledger; change it with `ap step`, not by hand -->
Steps, in order (the ledger; `ap step done <job> <id> "result"` marks one, `ap step add <job> "text"` adds one):
- [ ] 1. Review the page: no "pending" or "running" text; every figure renders; the four films decode and play; chart text and film labels readable at the 828 px column.
- [ ] 2. Publish docs/experiments/20261013-scale_turn_controllers/20261013-scale_turn_hom_mpc.html as a new artifact (icon "chart", one-sentence description); write its URL to docs/experiments/20261013-scale_turn_controllers/artifact_url.txt and replace "*(publish job pending)*" in its INDEX.md row with the URL.
- [ ] 3. Commit the URL changes and push to origin main.
<!-- /ap:steps -->
Claimed: docs/experiments/INDEX.md (artifact column of the 2026-10-13 scale-turn row), docs/experiments/20261013-scale_turn_controllers/artifact_url.txt.
Waiting on owner: none for this job. Owner decisions named on the page: three-pad MPC at placements 1 and 2 (about 3.6 h on 12 threads); the SGS policy over the cells (GPU, multi-hour).
Do not: edit pages, builders or templates; publish other pages; delete any artifact; force-push.
