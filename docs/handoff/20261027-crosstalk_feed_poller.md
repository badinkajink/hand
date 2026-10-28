Status: active
Task: (owner, 2026-10-27 20:5x) "set up a periodic poller ... to fetch results from the ur5 calibration coordinator and
update results ... namely for the crosstalk docpage". Keep the crosstalk page, and the crosstalk sentences of the fingertip
overview, current with what the coordinator chat appends to the v3 folder's feed.jsonl through the night and after.
Read first: the crosstalk page docs/experiments/20261027-fingertip_crosstalk/20261027-magnetic_crosstalk_between_neighbouring_fingertips.html
and its builder scripts/fingertip_crosstalk_page.py; docs/experiments/20261027-v3_tip_calibration_and_crosstalk/README.md
(the plan, and its section "Results feed for the doc-writing session") and feed.jsonl beside it;
docs/experiments/20261027-fingertip_holder/README.md; the docstring of scripts/crosstalk_feed_watch.py;
~/.claude/writing_reference.md; memory project_fingertip_crosstalk_budget_2026_10_27.md and project_v3_tips_plan_2026_10_27.md.
State: armed 2026-10-27 21:00 from the colorado window. 21:06: the feed's 4 lines (20:36 apex search, 20:45 +y apex
registration, 20:45 map plan, 20:47 cell renders) and the "Crosstalk:" commit e19945e7 are on the page (35a06b13) and
acked in logs/crosstalk_feed_watch/state.json. The page's holder item has the 20:45 plan with twists -20 to +10 deg, as
the plan file has them (the feed line says +20); Figure 3 is the hover render. Watch loop (step 2) running from 21:07.
Commits of this job start "Fingertip crosstalk page:", never "Crosstalk:" (the watcher keys on that prefix).
<!-- ap:steps: rendered by ~/.claude/autopilot/ap from its ledger; change it with `ap step`, not by hand -->
Steps, in order (the ledger; `ap step done <job> <id> "result"` marks one, `ap step add <job> "text"` adds one):
- [x] 1. Bring the page up to the feed's four lines. Rewrite the future-work item "The neighbour's field on the holder… Done 10-27 21:06: Page's holder-map item on the 20:45 plan (twists -20 to +10 as the plan file has them; the feed line says +20), pinch and five-magnet items on the README, Figure 3 the hover render; 35a06b13; feed acked (4 lines, 1 commit)
- [ ] 2. Watch loop, open until the owner says stop or the pinch's analysis is on the page. Launch .venv/bin/python scripts/crosstalk_feed_watch.py with run_in_background, run ~/.claude/autopilot/ap snooze hand-20261027-crosstalk_feed_poller 235 "waiting for the coordinator's crosstalk feed", and end the turn. On exit 0, handle each new line by the standing directions (ap step add a step for any change larger than an edit, do it, mark it done), rebuild, commit, --ack, update State, relaunch. On exit 3 (no news in 230 min), set State's last check from date and relaunch.
Standing directions (for every step):
- What goes on the crosstalk page, by the feed line's kind. plan and registration: the future-work items on the holder map and the pinch follow the current plan, a sentence or two each. session: the item says what ran (date, counts) until its analysis arrives; a partial or aborted run gets one sentence. analysis: a results section ('Neighbour field measured on the fingertip holder', later 'Pinch between two v3 tips') with the coordinator's numbers as given and its files cited by path, a table for more than four numbers, its figure or one drawn in the builder from its data files (sentence case, text 13 px or more as displayed, looked at before committing); the lede, the docs-summary and the figure and table numbers follow, and the future-work item becomes what remains, with its falsification criterion. figure: use it only when it shows a result or the measurement's geometry. When an analysis changes a number that the fingertip overview quotes on crosstalk, update that sentence in scripts/overview_fingertip_text.py and rebuild the overview.
- Writing and compute: the Writing rules of ~/.claude/CLAUDE.md, with ~/.claude/writing_reference.md read once per session; every build line within the prose targets (scripts/prose_lint.py --top 20 PAGE lists the sentences to fix); anything over a minute or more than one core goes through ~/.claude/bin/resguard.sh run with a measured --mem; commit after each page change, with the refreshed indices, and only the files you changed.
<!-- /ap:steps -->
Claimed: scripts/fingertip_crosstalk_page.py, docs/experiments/20261027-fingertip_crosstalk/, scripts/crosstalk_feed_watch.py,
logs/crosstalk_feed_watch/, scripts/overview_fingertip_text.py (its crosstalk sentences only), docs/overviews/printed_fingertips.html
Waiting on owner: none
Do not: git pull, fetch or push (the coordinator syncs main and pushes; this job's commits go out with its pushes); ssh
to corvus or touch the robot; edit lab/, QUEUE.md, the v3 folder (README, feed.jsonl, data, media),
scripts/fingertip_crosstalk.py (the coordinator's map fit goes there) or other coordinator scripts; refit or reanalyse
what the coordinator analyses; publish artifacts (the owner reads the local pages).
