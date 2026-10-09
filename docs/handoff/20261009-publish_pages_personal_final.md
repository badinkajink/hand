Status: queued 2026-10-09 after hand-20261007-tpu_tip_finite_elements, on personal (pinned, weekly stop 100 %, Sonnet at medium effort: a rote task)
Task (owner, 2026-10-09 12:35): "is it possible to republish the artifacts via the autopilot on my personal?" The artifacts belong to the personal (Pro) login. The owner's windows and the other jobs now run on the Team Max seat, which cannot write them. This second pass runs after the TPU tip finite-element job, the last of the queue, and publishes the pages that queue finished; it edits no page or builder.
Read first: docs/experiments/INDEX.md (each row's local file and artifact column); the Artifact tool's rules: read an existing artifact with action "read" before republishing to it, and read every page's content before publishing it.
State (2026-10-09): queued; the first pass is docs/handoff/20261009-publish_pages_personal.md.
<!-- ap:steps: rendered by ~/.claude/autopilot/ap from its ledger; change it with `ap step`, not by hand -->
Steps, in order (the ledger; `ap step done <job> <id> "result"` marks one, `ap step add <job> "text"` adds one):
- [ ] 1. Pages: the newest dated docs/experiments/*-contact_overview/*-sphere_pad_contact_model.html (republish to https://claude.ai/artifact/4w96Kt4J3tunu1xuz4XbdN); docs/experiments/20261008-contact_model_policies/20261008-fingertip_contact_model_policies.html (new artifact); the TPU tip finite-element page docs/experiments/*-tpu_tip_fem/*.html (new artifact); and every page with an artifact in INDEX.md whose file changed after the commit that recorded its URL (republish it).
- [ ] 2. Review each page before publishing it, as step 1 of the first pass says (visible text read in full; title, numbers in the lede, no placeholder or Publish pending text, every img and video with a source); write defects into State.
- [ ] 3. Publish: action read before each republish; new artifacts get a one-word generic icon and a one-sentence description.
- [ ] 4. Record every URL in the page's INDEX.md row (artifact column) and in artifact_url.txt beside each page; commit only those files.
<!-- /ap:steps -->
Claimed: docs/experiments/INDEX.md (the artifact column of the rows it publishes), docs/experiments/*/artifact_url.txt of these pages.
Waiting on owner:
Do not: edit pages, builders or templates (other jobs claim them); publish pages not listed here (the 10-07 contact-gait and DiffMJX pages are local studies by choice); delete any artifact; push.
