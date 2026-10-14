Status: done
Task (owner, 2026-10-11): "can we publish/update the artifacts via a personal autopilot" and, on the two pages the last pass could not publish, "seems like you should fix that url issue?". The artifacts belong to the personal login; windows and jobs on the Max seat build local pages only. This pass fixes the URLs and publishes the new bench plan page, which the owner reviews on 2026-10-12.
Read first: docs/handoff/20261009-publish_pages_personal.md (the earlier pass: its review step, and its State on the four pages it skipped); docs/experiments/INDEX.md (each row's local file and artifact column); the Artifact tool's rules: read an artifact with action "read" before republishing to it, and read every page's content before publishing it.
State (2026-10-11): all pages reviewed, no defects. Old HOM URLs Rvfw1yQfgfWV8PdTvv7jsc, DJLxG63vLZXs2Ey6hiCBuc, Lya2Bf6HnW81bEPRPjsjYf returned not found; new: contact B2DR23gn2HeBNg9RJGnXcS, brake 335iFyMojxjveNyC66MEAg, chain RN5kgsnDiKyuVSZYWrQexs. servo_sysid VLJvFQcYJw3HTgFEpXZ4t2. bench plan B4HkFKe9xXwCqXa3pJi9DD. Pushed to origin main (79e09621).
<!-- ap:steps: rendered by ~/.claude/autopilot/ap from its ledger; change it with `ap step`, not by hand -->
Steps, in order (the ledger; `ap step done <job> <id> "result"` marks one, `ap step add <job> "text"` adds one):
- [x] 1. Review each page of steps 2 and 3 before publishing it: extract its visible text with a short script in the s… Done 10-11 15:11: done, see State
- [x] 2. URL fixes. (a) docs/experiments/20260902-servo-sysid/20260902-servo_sysid.html has no artifact of its own: it… Done 10-11 15:11: done, see State
- [x] 3. Publish docs/experiments/20261011-bench_plan/20261011-bench_plan_finger_spacing.html as a new artifact (one-w… Done 10-11 15:11: done, see State
- [x] 4. Commit only INDEX.md and the artifact_url.txt files, then git push origin main (owner 2026-10-11 asked for co… Done 10-11 15:11: done, see State
<!-- /ap:steps -->
Claimed: docs/experiments/INDEX.md (the artifact column of these five rows), the artifact_url.txt files of these five pages.
Waiting on owner: none.
Do not: edit pages, builders or templates (hand-20261011-docs_followup runs in the same working tree and claims many of them); publish pages not listed here; delete any artifact; git add anything but the files under Claimed; force-push.
