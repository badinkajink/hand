Status: active
Task: (owner, 2026-10-27 20:5x) "set up a periodic poller ... to fetch results from the ur5 calibration coordinator and
update results ... namely for the crosstalk docpage". Keep the crosstalk page, and the crosstalk sentences of the fingertip
overview, current with what the coordinator chat appends to the v3 folder's feed.jsonl through the night and after.
Read first: the crosstalk page docs/experiments/20261027-fingertip_crosstalk/20261027-magnetic_crosstalk_between_neighbouring_fingertips.html
and its builder scripts/fingertip_crosstalk_page.py; docs/experiments/20261027-v3_tip_calibration_and_crosstalk/README.md
(the plan, and its section "Results feed for the doc-writing session") and feed.jsonl beside it;
docs/experiments/20261027-fingertip_holder/README.md; the docstring of scripts/crosstalk_feed_watch.py;
~/.claude/writing_reference.md; memory project_fingertip_crosstalk_budget_2026_10_27.md and project_v3_tips_plan_2026_10_27.md.
State: armed 2026-10-27 21:00 from the colorado window. 10-28 02:58: all 17 feed lines (to 02:44) and the 6
"Crosstalk:" commits are handled and acked (logs/crosstalk_feed_watch/state.json); watch loop relaunched 02:58.
Crosstalk page b50e44b4: Section 3 "Neighbour field measured on the fingertip holder" has the touch-offs (Figure 2,
Table 2), the 137 hovers against the net moments (Figure 3, Table 3: 97 % at either held board), and since line 15
(01:28, owner's sharpie question) the change as a face force (Figure 4, Table 4: W 7-11 mm 0.31 and 0.64 N, after the
model 0.23 and 0.12 N, five-N52 model 3.8 N; the W 7-11 medians are computed from the hover file and match the line).
Lede and summary lead with the sharpie width; press signal per newton from data/20261028-four_vs_five_grid.json (21.9
against 48.1 uT/N); Figure 6 the hover render; numbering by SEC/FIG/TAB; future work has the side-by-side plan of the
owner's morning (c810bcdd). The builder reads the coordinator's v3 files AT HEAD and warns when the holder file's
"made" differs from HOLD_MADE (22:50). Overview: crosstalk paragraph and two bullets (fd38e26c to dc8b6478), unchanged
since (no quoted crosstalk number changed). Remaining on the holder: the five-fold pose fit (fingertip_crosstalk.py,
coordinator), the side-by-side hovers (morning), the pinch plan.
Calibration page 41747b29 (step 9, done 02:05; the ledger's "02:2x" is wrong): Sections 10 (v2 ID 1) and 11 (v3 ID 1,
with line 16's FE calibration as Table 8, and line 17's "Where the magnets sit" in 1d775a10); direction 11 sends later
calibration lines there (stage 2's slides, rolls and twists about 06:00, the tall about 09:00, the rod about 11:00).
For the owner: the 21:44 line's "matches the robot's board within 10 % to W 5 mm" is not in its bins (model +4 % at
contact, -10 to -22 % from W 0.3 to 5 mm), so the page leaves it out; the 20:45 feed line says twists to +20 deg, the
plan file stops at +10; the 22:52 line's "at most 0.4 N on the HEX-E" is 0.455 N in the file (the page says below the
0.5 N guard). The overview's magnets bullet ("Four magnets (0.051 N) or five (0.029 N) waits on the owner") now has a
measured counterpart (63 and 35 mN at 1 to 2 N, ratio 1.8; calibrated errors alike); it is not a crosstalk sentence,
so this job leaves it. Lines 11 and 14 were acked without a page change.
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
- [x] 9. Calibration page up to 10-27 and the v3 tips (owner, 10-28 01:5x, through the coordinator: "are the calibrati… Done 10-28 02:05: Done 10-28 02:2x: calibration page 41747b29: session table from holds.npz (11 sessions, 11,513 holds; cb3t done, v3 grid stage), Section 10 v2 ID 1 (Table 5 tips, Figure 9 hard presses, rod, Table 6 tall vs sphere, twists, Table 7 suite 15.0 -> 9.1 h), Section 11 v3 ID 1 (Figure 10 four vs five, Table 8 physical model on pad 05R4's FE basis from feed line 16), lede/summary/terms/limits/future; prose 309 sentences, 17.0 mean, 2.3 % over 30, 275/1000
- [x] 10. Feed line of 01:28 (owner, 01:2x: 'at what distances do the effects resolve, compared to the 5 n52 magnets? a… Done 10-28 02:10: Done 10-28 02:10: crosstalk page b50e44b4: Figure 4 (centred hovers as face force on 21.9 uT/N, five-N52 model on 48.1), Table 4 (W 7-11 mm: 0.31 and 0.64 N, after the model 0.23 and 0.12 N; five-N52 3.8 N), lede and summary on the sharpie width, side-by-side future item (c810bcdd); overview unchanged; acked 16 lines with line 16 (calibration page 41747b29)
Standing directions (for every step):
- What goes on the crosstalk page, by the feed line's kind. plan and registration: the future-work items on the holder map and the pinch follow the current plan, a sentence or two each. session: the item says what ran (date, counts) until its analysis arrives; a partial or aborted run gets one sentence. analysis: a results section ('Neighbour field measured on the fingertip holder', later 'Pinch between two v3 tips') with the coordinator's numbers as given and its files cited by path, a table for more than four numbers, its figure or one drawn in the builder from its data files (sentence case, text 13 px or more as displayed, looked at before committing); the lede, the docs-summary and the figure and table numbers follow, and the future-work item becomes what remains, with its falsification criterion. figure: use it only when it shows a result or the measurement's geometry. When an analysis changes a number that the fingertip overview quotes on crosstalk, update that sentence in scripts/overview_fingertip_text.py and rebuild the overview.
- Writing and compute: the Writing rules of ~/.claude/CLAUDE.md, with ~/.claude/writing_reference.md read once per session; every build line within the prose targets (scripts/prose_lint.py --top 20 PAGE lists the sentences to fix); anything over a minute or more than one core goes through ~/.claude/bin/resguard.sh run with a measured --mem; commit after each page change, with the refreshed indices, and only the files you changed.
- Calibration feed lines (kind calibration, and analyses of v3 ID 1's suite: stage 2's slides, rolls and twists about 06:00, the tall about 09:00, the rod about 11:00, the physical model) extend the calibration page (scripts/fingertip_calibration_text.py Sections 10 and 11, Table 4's v3 row, figures in scripts/fingertip_calibration_figures.py drawn from the coordinator's files at HEAD, looked at before committing); lede and summary follow; commits start 'Calibration page:'
<!-- /ap:steps -->
Claimed: scripts/fingertip_crosstalk_page.py, docs/experiments/20261027-fingertip_crosstalk/, scripts/crosstalk_feed_watch.py,
logs/crosstalk_feed_watch/, scripts/overview_fingertip_text.py (its crosstalk sentences only), docs/overviews/printed_fingertips.html,
scripts/fingertip_calibration_page.py, scripts/fingertip_calibration_figures.py, scripts/fingertip_calibration_text.py (its
prose), docs/experiments/20261026-fingertip_calibration/ (step 9)
Waiting on owner: none
Do not: git pull, fetch or push (the coordinator syncs main and pushes; this job's commits go out with its pushes); ssh
to corvus or touch the robot; edit lab/, QUEUE.md, the v3 folder (README, feed.jsonl, data, media),
scripts/fingertip_crosstalk.py (the coordinator's map fit goes there) or other coordinator scripts; refit or reanalyse
what the coordinator analyses; publish artifacts (the owner reads the local pages).
