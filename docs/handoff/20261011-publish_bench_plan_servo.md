Status: active
Task (owner, 2026-10-11): publish the bench plan revised on the servo-limited plant (docs/handoff/20261011-bench_plan_servo_plant.md) before the owner's review on 10-12. This job edits no page or builder.
Read first: docs/handoff/20261011-bench_plan_servo_plant.md (its State names what changed); docs/experiments/INDEX.md and docs/overviews/artifact_urls.txt; the Artifact tool's rules: read an artifact with action "read" before republishing to it, and read every page's content before publishing it.
State (2026-10-11 22:15): queued after hand-20261011-bench_plan_servo_plant.
<!-- ap:steps: rendered by ~/.claude/autopilot/ap from its ledger; change it with `ap step`, not by hand -->
Steps, in order (the ledger; `ap step done <job> <id> "result"` marks one, `ap step add <job> "text"` adds one):
- [ ] 1. Review docs/experiments/20261011-bench_plan/20261011-bench_plan_finger_spacing.html: no placeholders, every figure renders, the lede states the revised numbers, the INDEX row exists.
- [ ] 2. Republish it to its URL, https://claude.ai/artifact/B4HkFKe9xXwCqXa3pJi9DD (action "read" first); also republish any overview whose file the bench-plan job changed, to the URL in docs/overviews/artifact_urls.txt.
- [ ] 3. Record any new URL in INDEX.md, docs/overviews/artifact_urls.txt and the page's artifact_url.txt; commit; git push origin main (fast-forward only).
<!-- /ap:steps -->
Claimed: docs/experiments/INDEX.md (artifact column), docs/overviews/artifact_urls.txt, artifact_url.txt files.
Waiting on owner: none.
Do not: edit pages, builders or templates; publish pages the bench-plan job did not change; delete any artifact; force-push.
