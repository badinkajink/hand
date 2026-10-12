Status: active
Task (owner, 2026-10-12): publish the pages the servo-plant skin job changes (docs/handoff/20261012-servo_skin_policies.md): the servo-plant policy page, the contact overview and the RL policies overview. This job edits no page or builder.
Read first: docs/handoff/20261012-servo_skin_policies.md (its State names the pages); docs/experiments/INDEX.md and docs/overviews/artifact_urls.txt; the Artifact tool's rules: read an artifact with action "read" before republishing to it, and read every page's content before publishing it.
State (2026-10-12 11:15): queued after hand-20261012-servo_skin_policies.
<!-- ap:steps: rendered by ~/.claude/autopilot/ap from its ledger; change it with `ap step`, not by hand -->
Steps, in order (the ledger; `ap step done <job> <id> "result"` marks one, `ap step add <job> "text"` adds one):
- [ ] 1. List the pages that job changed (git log of its commits); review each: no placeholders or pending notes left, every figure renders, films play and show their poster.
- [ ] 2. Republish to the URLs of record: the servo-plant page https://claude.ai/artifact/KUbENqD5kWU7yhZnbymXTH, the overviews at the URLs in docs/overviews/artifact_urls.txt; action "read" before each.
- [ ] 3. Record any new URL in INDEX.md, docs/overviews/artifact_urls.txt and the page's artifact_url.txt; commit; git push origin main (fast-forward only).
<!-- /ap:steps -->
Claimed: docs/experiments/INDEX.md (artifact column), docs/overviews/artifact_urls.txt, artifact_url.txt files.
Waiting on owner: none.
Do not: edit pages, builders or templates; publish pages that job did not change; delete any artifact; force-push.
