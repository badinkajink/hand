Status: active
Task (owner, 2026-10-12): publish the revised printed TPU fingertip page ("i am going to print fingertips out pretty soon ... let's just get our analysis and infrastucture shored up so that i can read the page fully and better understand what fingers to send off"). This job edits no page or builder.
Read first: docs/experiments/20261009-tpu_tip_fem/20261009-printed_tpu_fingertip_finite_elements.html and its artifact_url.txt (https://claude.ai/artifact/MSLurN7jPTaFfXUdsxJMZv); docs/experiments/20261009-tpu_tip_fem/print/README.md; the Artifact tool's rules: read an artifact with action "read" before republishing to it, and read every page's content before publishing it.
State (2026-10-12): the page was rebuilt by the window with the print plan, the embedded magnet, the expected sim-to-real gap and the calibration section (scripts/tpu_tip_fem_page.py, scripts/tpu_tip_print_plan.py; rows magnet_runs.jsonl).
State (autopilot, 10-12 19:32): step 1 passed; screenshots in logs/20261012-figure_check/ (0 flagged). Tables 5 and 6 are 1651 and 1470 px wide and scroll sideways in the 828 px column (Table 4 by 32 px), because the house CSS th{white-space:nowrap} survives retro_style; the fix belongs in the builder, for the owner or a docs job. Next: publish the page with its 7 linked print/ files (2 STL, 5 sheets) so the links work in the artifact.
<!-- ap:steps: rendered by ~/.claude/autopilot/ap from its ledger; change it with `ap step`, not by hand -->
Steps, in order (the ledger; `ap step done <job> <id> "result"` marks one, `ap step add <job> "text"` adds one):
- [x] 1. Review the page: no "Runs in progress" or "running" cell, every figure renders, Tables 1-3 readable, links to… Done 10-12 19:32: Reviewed 10-12 19:35: no run markers; 7 figures, 98 SVGs, 2 PNGs, 3 films load (figure check 0 flagged); Tables 1-3 fit the 828 px column; 7 print/ links exist
- [ ] 2. Republish it to https://claude.ai/artifact/MSLurN7jPTaFfXUdsxJMZv (action "read" first).
- [ ] 3. Record any URL change in docs/experiments/INDEX.md and the folder's artifact_url.txt; commit; git push origin main (fast-forward only).
<!-- /ap:steps -->
Claimed: docs/experiments/INDEX.md (artifact column), docs/experiments/20261009-tpu_tip_fem/artifact_url.txt.
Waiting on owner: none.
Do not: edit pages, builders or templates; publish other pages; delete any artifact; force-push.
