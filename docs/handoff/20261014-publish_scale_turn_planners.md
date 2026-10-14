Status: new
Task (owner, 2026-10-13): "can we update the docpage with better, more stable videos (as well as a planner comparison?)". The window that built it republishes nothing (max seat); this job republishes the updated page to its existing artifact. It edits no page or builder.
Read first: docs/experiments/20261013-scale_turn_controllers/20261013-scale_turn_hom_mpc.html (built 2026-10-14 by scripts/scale_turn_controllers_page.py, commit named in State); docs/experiments/INDEX.md (its 2026-10-13 scale-turn row, URL https://claude.ai/artifact/JdCgjBHdcqu99dPgRwv1Vo); the Artifact tool's rules: read the artifact before publishing to it, and read every page's content before publishing it.
State (2026-10-14 01:00): page rebuilt with the section "Sampling planners compared on four cells" (Tables 3-5, Figures 5-7, Algorithm 2, Eqs. 4-5), the pad and GPU cost tables, four new 2x2 films (HOM, predictive sampling, pinned predictive sampling, pinned iCEM; 1280x960, 157 frames) and one earlier three-panel film; 5.97 MB, media inline as data URIs.
Next:
1. Review the page: no "pending" or "running" text; every figure renders; the five films decode and play; chart and film label text readable at the 828 px column.
2. Republish docs/experiments/20261013-scale_turn_controllers/20261013-scale_turn_hom_mpc.html to https://claude.ai/artifact/JdCgjBHdcqu99dPgRwv1Vo (read it first, then publish with that url; same URL, no new artifact).
3. Commit any change to the INDEX row or the URL file and push to origin main.
Claimed: docs/experiments/INDEX.md (artifact column of the 2026-10-13 scale-turn row), docs/experiments/20261013-scale_turn_controllers/artifact_url.txt.
Waiting on owner: none for this job. Owner decisions named on the page: pinned iCEM at every cell and placement (about 4 h), the iLQR test, the plant-mismatch rerun, the SGS policy over the cells.
Do not: edit pages, builders or templates; publish other pages; delete any artifact; force-push.
