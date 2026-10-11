Status: queued 2026-10-11 after hand-20261011-docs_followup, on personal (pinned, weekly stop 100 %, Sonnet at medium effort: a rote task)
Task (owner, 2026-10-11): "can we publish/update the artifacts via a personal autopilot". This pass runs after the docs follow-up job (docs/handoff/20261011-docs_followup.md: durable evaluation traces, interactive figures, mkdocs folded into the HTML docs) and publishes what that job changed or added. It edits no page or builder.
Read first: docs/handoff/20261011-docs_followup.md (its State names the pages it changed and added); docs/handoff/20261011-publish_personal.md (the review step); docs/experiments/INDEX.md and docs/overviews/artifact_urls.txt (local file and artifact of every page); the Artifact tool's rules: read an artifact with action "read" before republishing to it, and read every page's content before publishing it.
State (2026-10-11 15:50): queued. The URLs of record are those in INDEX.md and docs/overviews/artifact_urls.txt at commit a4db72c2 or later.
<!-- ap:steps: rendered by ~/.claude/autopilot/ap from its ledger; change it with `ap step`, not by hand -->
Steps, in order (the ledger; `ap step done <job> <id> "result"` marks one, `ap step add <job> "text"` adds one):
- [ ] 1. List the pages to publish: every page under docs/ with an artifact URL whose file changed after commit a4db72c2 (git log a4db72c2..HEAD -- FILE), the home page docs/index.html, and every page the docs follow-up job added (its State; for example docs/overviews/code_and_methods.html). Write the list into State.
- [ ] 2. Review each page as step 1 of docs/handoff/20261011-publish_personal.md says; an interactive figure must show its default view with JavaScript off, and its controls must work in the published artifact (open it and try one). Defects go into State; skip a defective page.
- [ ] 3. Publish: action "read" before each republish to an existing URL; new pages get a one-word icon and a one-sentence description.
- [ ] 4. Record new URLs in INDEX.md (artifact column), docs/overviews/artifact_urls.txt and each page's artifact_url.txt; commit only those files; git push origin main (owner 2026-10-11; fast-forward only, never force).
<!-- /ap:steps -->
Claimed: docs/experiments/INDEX.md (artifact column), docs/overviews/artifact_urls.txt, artifact_url.txt files.
Waiting on owner: none.
Do not: edit pages, builders or templates; publish pages not in step 1's list (the 10-07 contact-gait and DiffMJX pages are local studies by choice); delete any artifact; force-push.
