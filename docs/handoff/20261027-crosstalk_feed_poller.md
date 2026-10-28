Status: active
Task: (owner, 2026-10-27 20:5x) "set up a periodic poller ... to fetch results from the ur5 calibration coordinator and
update results ... namely for the crosstalk docpage". Keep the crosstalk page, and the crosstalk sentences of the fingertip
overview, current with what the coordinator chat appends to the v3 folder's feed.jsonl through the night and after.
Read first: the crosstalk page docs/experiments/20261027-fingertip_crosstalk/20261027-magnetic_crosstalk_between_neighbouring_fingertips.html
and its builder scripts/fingertip_crosstalk_page.py; docs/experiments/20261027-v3_tip_calibration_and_crosstalk/README.md
(the plan, and its section "Results feed for the doc-writing session") and feed.jsonl beside it;
docs/experiments/20261027-fingertip_holder/README.md; the docstring of scripts/crosstalk_feed_watch.py;
~/.claude/writing_reference.md; memory project_fingertip_crosstalk_budget_2026_10_27.md and project_v3_tips_plan_2026_10_27.md.
State: armed 2026-10-27 21:00 from the colorado window. 10-28 00:36: all 12 feed lines (to 00:18) and the 6
"Crosstalk:" commits are handled and acked (logs/crosstalk_feed_watch/state.json). Page a86135e0: Section 3 "Neighbour
field measured on the fingertip holder" has the touch-off approaches (Figure 2, Table 2), the map's 89 hovers and the 48
close -y hovers against the net moments fitted on the touch-offs (Figure 3, Table 3, -y pooled: held boards 97.4 and
97.0 %, robot's 91 and 94 %); lede and summary lead with 97 % at either held board in 137 hovers; Figure 5 the hover
render (re-made 23:13 with the tool by v3 ID 1's sphere registration, looked at); numbering by SEC/FIG/TAB in the
builder. Line 12 (00:18, four magnets against five, partial): v3 ID 1's face press signal, 22.4 uT/N against v2 ID 1's
51.4, is the lede's and summary's yardstick for the 39 to 60 uT at contact and is in Section 3 and the Press signal term,
read at HEAD from FVF (data/20261028-four_vs_five_partial.json); the full comparison is due when stage 2 ends, about
06:00: point FVF at its file (press_gain drops "the first" when the note no longer says partial). Overview: crosstalk
paragraph and two bullets (fd38e26c to dc8b6478), unchanged by line 12 (its quoted crosstalk numbers stand). The builder
reads the coordinator's v3 files AT HEAD (at_head) and warns when the holder file's "made" differs from HOLD_MADE
(22:50): its hand-written sentences then need the newest line. Remaining on the holder: the five-fold pose fit
(fingertip_crosstalk.py, coordinator), then the pinch plan; neither is in R31's night queue (the v3 ID 1 suite).
For the owner: the 21:44 line's "matches the robot's board within 10 % to W 5 mm" is not in its bins (model +4 % at
contact, -10 to -22 % from W 0.3 to 5 mm), so the page leaves it out; the 20:45 feed line says twists to +20 deg, the
plan file stops at +10; the 22:52 line's "at most 0.4 N on the HEX-E" is 0.455 N in the file, and the page says the
hovers stayed below the 0.5 N guard. The overview's magnets bullet ("Four magnets (0.051 N) or five (0.029 N) waits on
the owner") now has a measured partial counterpart (one record's noise equals 61 and 32 mN at 1 to 2 N); it is not a
crosstalk sentence, so this job leaves it. Line 11 (23:13, the sphere registration) was acked without a page change.
Watch loop relaunched 00:36.
Before each --ack, run crosstalk_feed_watch.py --once and check HEAD shows only the lines handled (ack takes all of HEAD).
Commits of this job start "Fingertip crosstalk page:", never "Crosstalk:" (the watcher keys on that prefix).
<!-- ap:steps: rendered by ~/.claude/autopilot/ap from its ledger; change it with `ap step`, not by hand -->
Steps, in order (the ledger; `ap step done <job> <id> "result"` marks one, `ap step add <job> "text"` adds one):
- [x] 1. Bring the page up to the feed's four lines. Rewrite the future-work item "The neighbour's field on the holder… Done 10-27 21:06: Page's holder-map item on the 20:45 plan (twists -20 to +10 as the plan file has them; the feed line says +20), pinch and five-magnet items on the README, Figure 3 the hover render; 35a06b13; feed acked (4 lines, 1 commit)
- [ ] 2. Watch loop, open until the owner says stop or the pinch's analysis is on the page. Launch .venv/bin/python scripts/crosstalk_feed_watch.py with run_in_background, run ~/.claude/autopilot/ap snooze hand-20261027-crosstalk_feed_poller 235 "waiting for the coordinator's crosstalk feed", and end the turn. On exit 0, handle each new line by the standing directions (ap step add a step for any change larger than an edit, do it, mark it done), rebuild, commit, --ack, update State, relaunch. On exit 3 (no news in 230 min), set State's last check from date and relaunch.
- [x] 5. Feed lines of 21:34-21:47 (commits 53993cbc, 1114cb94): results section 'Neighbour field measured on the fing… Done 10-27 22:17: Section 3 'Neighbour field measured on the fingertip holder' (Figure 2 from the file's bins, Table 2), lede, summary, numbering, limitations, holder item to the both-apex frame + 86 hovers + close -y plan (e96bf3f3); overview's 3 crosstalk sentences (fd38e26c); acked 8 lines, 3 commits. Builder now reads the coordinator's files at HEAD. The 21:44 'within 10 % to W 5 mm' left out (bins disagree)
- [x] 6. Feed line of 22:25 (commits 88fc93b5, 70dcfb11): the holder map's 89 hovers in Section 3 (net moments from th… Done 10-27 22:47: Hovers in Section 3: Figure 3 drawn from the file (2 x 2), Table 3 explained shares; lede/summary lead with the held-out 97 and 94 %; holder item = close -y plan + five-fold fit; overview +1 sentence; 7a66bfc9; acked 9 lines, 5 commits
- [x] 7. Feed line of 22:52 (commit d4fa2c07): the close -y hovers (222640-holdclosev3id1, 28 hovers + 20 repeats at W… Done 10-27 23:07: Close -y hovers in Section 3 (map and close session apart, -y pooled in Figure 3/Table 3: 97.0 % held, 94 % robot); lede 97 % at either held board in 137 hovers; holder item = five-fold fit; overview sentence; dc8b6478; acked 10 lines, 6 commits
- [x] 8. Feed line of 00:18 (four magnets against five, partial; c5fe7775): v3 ID 1's change per newton (face 22.4 aga… Done 10-28 00:36: Done 10-28 00:37: v3 ID 1's face press signal 22.4 uT/N (v2 ID 1 51.4) from the partial four-vs-five file at HEAD: lede and summary yardstick for the 39-60 uT at contact, the Press signal term, two sentences in Section 3; byline + close session; Figure 5's re-made render looked at; a86135e0; overview unchanged (its quoted crosstalk numbers stand); acked 12 lines, 6 commits
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
