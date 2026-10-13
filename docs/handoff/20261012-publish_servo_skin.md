Status: done
Task (owner, 2026-10-12): publish the pages the servo-plant skin job changes (docs/handoff/20261012-servo_skin_policies.md): the servo-plant policy page, the contact overview and the RL policies overview. This job edits no page or builder.
Read first: docs/handoff/20261012-servo_skin_policies.md (its State names the pages); docs/experiments/INDEX.md and docs/overviews/artifact_urls.txt; the Artifact tool's rules: read an artifact with action "read" before republishing to it, and read every page's content before publishing it.
State (2026-10-12): all pages published; sprung skin page at https://claude.ai/artifact/FPiCbZKt1NF6J3J7HDgL46; committed and pushed.
<!-- ap:steps: rendered by ~/.claude/autopilot/ap from its ledger; change it with `ap step`, not by hand -->
Steps, in order (the ledger; `ap step done <job> <id> "result"` marks one, `ap step add <job> "text"` adds one):
- [x] 1. List the pages that job changed (git log of its commits); review each: no placeholders or pending notes left,… Done 10-12 18:08: 5 pages scanned: no placeholders/pending text, every video has an inline poster and data source
- [x] 2. Republish to the URLs of record: the servo-plant page https://claude.ai/artifact/KUbENqD5kWU7yhZnbymXTH, the … Done 10-12 18:09: republished servo-plant page, native compliance page, contact and RL overviews; new sprung-skin page published
- [ ] 3. Record any new URL in INDEX.md, docs/overviews/artifact_urls.txt and the page's artifact_url.txt; commit; git push origin main (fast-forward only).
Standing directions (for every step):
- The contact overview also carries commit e9ef7730 (skin section corrected against the deck 23 math pass: converged loop 2.58 uJ, armature, impratio, 3 N twist); republish it even if the skin job leaves that page unchanged, and check its skin section and Table 3 when reviewing.
- Also republish docs/experiments/20261007-native_compliance/20261007-native_presliding_compliance.html to https://claude.ai/artifact/D38s9bTzL9szVLNnuYk698 (correction note of 2026-10-12, commit a0f2c738), and the contact overview for 81e20730 (skin at the trainer's impratio of 10, Figure 14b), even if the skin job leaves them unchanged; read each artifact first.
- Publish the new page docs/experiments/20261012-sprung_skin/20261012-sprung_skin_presliding.html as a new artifact (icon 'chart', one-sentence description), record its URL in docs/experiments/INDEX.md and docs/experiments/20261012-sprung_skin/artifact_url.txt, and republish both overviews.
<!-- /ap:steps -->
Claimed: docs/experiments/INDEX.md (artifact column), docs/overviews/artifact_urls.txt, artifact_url.txt files.
Waiting on owner: none.
Do not: edit pages, builders or templates; publish pages that job did not change; delete any artifact; force-push.
