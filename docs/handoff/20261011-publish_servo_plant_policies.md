Status: active
Task (owner, 2026-10-11): publish the pages of the servo-limited-plant policy study (docs/handoff/20261011-servo_plant_policies.md): its new result page, the corrected 2026-10-08 policy page and the overviews it changed. This job edits no page or builder.
Read first: docs/handoff/20261011-servo_plant_policies.md (its State names the pages); docs/experiments/INDEX.md and docs/overviews/artifact_urls.txt; the Artifact tool's rules: read an artifact with action "read" before republishing to it, and read every page's content before publishing it.
State (2026-10-11 22:15): queued after hand-20261011-servo_plant_policies.
Next:
1. List the pages that job changed or added (git log of its commits); review each: no placeholders, figures render, interactive figures show their static default view.
2. Publish: the new page as a new artifact (one-word icon, one-sentence description); the 10-08 page to https://claude.ai/artifact/XvNi2tMdCgYBBGp3BZQWbh; overviews to the URLs in docs/overviews/artifact_urls.txt; action "read" before each republish.
3. Record new URLs in INDEX.md, docs/overviews/artifact_urls.txt and each page's artifact_url.txt; commit; git push origin main (fast-forward only).
Claimed: docs/experiments/INDEX.md (artifact column), docs/overviews/artifact_urls.txt, artifact_url.txt files.
Waiting on owner: none.
Do not: edit pages, builders or templates; publish pages that job did not change; delete any artifact; force-push.
