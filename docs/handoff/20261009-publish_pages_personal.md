Status: armed 2026-10-09 on personal (pinned, weekly stop 100 %, Sonnet at medium effort: a rote task)
Task (owner, 2026-10-09 12:35): "is it possible to republish the artifacts via the autopilot on my personal?" The artifacts belong to the personal (Pro) login. The owner's windows and the other jobs now run on the Team Max seat, which cannot write them. This job publishes finished local pages from the personal login and edits no page or builder.
Read first: docs/experiments/INDEX.md (each row's local file and artifact column); the Artifact tool's rules: read an existing artifact with action "read" before republishing to it, and read every page's content before publishing it.
State (2026-10-09): nothing published yet.
<!-- ap:steps: rendered by ~/.claude/autopilot/ap from its ledger; change it with `ap step`, not by hand -->
Steps, in order (the ledger; `ap step done <job> <id> "result"` marks one, `ap step add <job> "text"` adds one):
- [ ] 1. Review each page of steps 2 and 3 before publishing it: extract its visible text with a short script in your scratchpad (html.parser; skip script and style elements and data: URIs) and read all of it. Check the title, that the lede states numbers, that no {{PLACEHOLDER}} or "Publish pending" text remains, and that every img and video element has a source. Write any defect into State and publish anyway, unless the page is broken (then `ap step wait` with the defect).
- [ ] 2. Republish to existing artifacts (action "read" first): docs/experiments/20261005-contact_bed/20261005-contact_model_bed.html to https://claude.ai/artifact/5MUdW6D2Xje2yKHpp1YUVt; docs/experiments/20261009-contact_overview/20261009-sphere_pad_contact_model.html to https://claude.ai/artifact/4w96Kt4J3tunu1xuz4XbdN.
- [ ] 3. Publish as new artifacts, each with a one-word generic icon and a one-sentence description: docs/experiments/20261007-newton_hydro_tests/20261007-newton_hydroelastic_friction_reduction_edge.html; docs/experiments/20261007-native_compliance/20261007-native_presliding_compliance.html; docs/experiments/20261008-hand_object_scale/20261008-finger_spacing_object_size.html.
- [ ] 4. Record every URL in the page's INDEX.md row (artifact column) and in artifact_url.txt beside each page; commit only those files.
<!-- /ap:steps -->
Claimed: docs/experiments/INDEX.md (the artifact column of these rows), docs/experiments/*/artifact_url.txt of these pages.
Waiting on owner:
Do not: edit pages, builders or templates (other jobs claim them); publish pages not listed here (the 10-07 contact-gait and DiffMJX pages are local studies by choice); delete any artifact; push.
