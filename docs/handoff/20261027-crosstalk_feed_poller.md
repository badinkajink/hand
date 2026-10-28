Status: active
Task: (owner, 2026-10-27 20:5x) "set up a periodic poller ... to fetch results from the ur5 calibration coordinator and
update results ... namely for the crosstalk docpage". Keep the crosstalk page, and the crosstalk sentences of the fingertip
overview, current with what the coordinator chat appends to the v3 folder's feed.jsonl through the night and after.
Read first: the crosstalk page docs/experiments/20261027-fingertip_crosstalk/20261027-magnetic_crosstalk_between_neighbouring_fingertips.html
and its builder scripts/fingertip_crosstalk_page.py; docs/experiments/20261027-v3_tip_calibration_and_crosstalk/README.md
(the plan, and its section "Results feed for the doc-writing session") and feed.jsonl beside it;
docs/experiments/20261027-fingertip_holder/README.md; the docstring of scripts/crosstalk_feed_watch.py;
~/.claude/writing_reference.md; memory project_fingertip_crosstalk_budget_2026_10_27.md and project_v3_tips_plan_2026_10_27.md.
State: armed 2026-10-27 21:00 from the colorado window. The feed has 4 lines (20:36 apex search, 20:45 +y apex
registration, 20:45 map plan, 20:47 cell renders) and 1 "Crosstalk:" commit (e19945e7); none handled, and the watcher's
state logs/crosstalk_feed_watch/state.json does not exist yet. The page's future-work item on the holder still describes
the 18:01 plan (45 facing and 31 side-by-side poses, a run with the stations empty first).
Next:
1. Bring the page up to the feed's four lines. Rewrite the future-work item "The neighbour's field on the holder" to the 20:45 plan: both held tips up, the +y one turned 180 deg; five -y touch-offs, 49 facing hovers over +y (W 0.5 to 45 mm, twists -20 to +20 deg, 3 mm offsets), 21 over -y at W 10 mm and over, 20 repeats, about 45 min; no empty run and no side-by-side poses; the +y apex measured 2.86 mm lower and 2.50 mm toward -y_I than the seating put it, the holder level. Keep its falsification criterion. Rebuild with .venv/bin/python scripts/fingertip_crosstalk_page.py (the build line within the prose targets), commit the page with the refreshed indices, then run .venv/bin/python scripts/crosstalk_feed_watch.py --ack.
2. Watch loop, open until the owner says stop or the pinch's analysis is on the page. Launch .venv/bin/python scripts/crosstalk_feed_watch.py with run_in_background, run ~/.claude/autopilot/ap snooze hand-20261027-crosstalk_feed_poller 235 "waiting for the coordinator's crosstalk feed", and end the turn. On exit 0, handle each new line by the standing directions (ap step add a step for any change larger than an edit, do it, mark it done), rebuild, commit, --ack, update State, relaunch. On exit 3 (no news in 230 min), set State's last check from date and relaunch.
Claimed: scripts/fingertip_crosstalk_page.py, docs/experiments/20261027-fingertip_crosstalk/, scripts/crosstalk_feed_watch.py,
logs/crosstalk_feed_watch/, scripts/overview_fingertip_text.py (its crosstalk sentences only), docs/overviews/printed_fingertips.html
Waiting on owner: none
Do not: git pull, fetch or push (the coordinator syncs main and pushes; this job's commits go out with its pushes); ssh
to corvus or touch the robot; edit lab/, QUEUE.md, the v3 folder (README, feed.jsonl, data, media),
scripts/fingertip_crosstalk.py (the coordinator's map fit goes there) or other coordinator scripts; refit or reanalyse
what the coordinator analyses; publish artifacts (the owner reads the local pages).
