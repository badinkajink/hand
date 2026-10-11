Status: done
Task (owner, 2026-10-09 12:35): "is it possible to republish the artifacts via the autopilot on my personal?" The artifacts belong to the personal (Pro) login. The owner's windows and the other jobs now run on the Team Max seat, which cannot write them. This second pass runs after the TPU tip finite-element job, the last of the queue, and publishes the pages that queue finished; it edits no page or builder.
Read first: docs/experiments/INDEX.md (each row's local file and artifact column); the Artifact tool's rules: read an existing artifact with action "read" before republishing to it, and read every page's content before publishing it.
State (2026-10-10): done. Policies page https://claude.ai/artifact/XvNi2tMdCgYBBGp3BZQWbh, TPU tip page https://claude.ai/artifact/MSLurN7jPTaFfXUdsxJMZv; the contact overview artifact already held the 10-10 03:27 build.
<!-- ap:steps: rendered by ~/.claude/autopilot/ap from its ledger; change it with `ap step`, not by hand -->
Steps, in order (the ledger; `ap step done <job> <id> "result"` marks one, `ap step add <job> "text"` adds one):
- [x] 1. Pages: the newest dated docs/experiments/*-contact_overview/*-sphere_pad_contact_model.html (republish to htt… Done 10-10 05:45: Contact overview artifact already current (live version built 2026-10-10 03:27 = docs/overviews/contact_model.html); no other INDEX page changed since 983c6923; two new pages identified
- [x] 2. Review each page before publishing it, as step 1 of the first pass says (visible text read in full; title, nu… Done 10-10 05:45: Both new pages read: titles, ledes with numbers, no placeholder text, all 15 and 5 media have sources; no defects
- [x] 3. Publish: action read before each republish; new artifacts get a one-word generic icon and a one-sentence desc… Done 10-10 05:45: Published policies page (XvNi2tMdCgYBBGp3BZQWbh) and TPU tip page (MSLurN7jPTaFfXUdsxJMZv); contact overview read, not republished (identical build)
- [x] 4. Record every URL in the page's INDEX.md row (artifact column) and in artifact_url.txt beside each page; commi… Done 10-10 05:45: INDEX.md rows and artifact_url.txt updated, commit 81b28543
Standing directions (for every step):
- Repository history was rewritten on 2026-10-09 at 14:00 (owner-approved slimming): commit hashes from before then resolve only in git@github.com:badinkajink/hand-archive.git. logs/, results/, wandb/ and relevant_papers/ are no longer tracked (their files stay on disk; do not re-add them). Media, data rows (json, jsonl, csv, npz) and detailed pages under docs/ are Git LFS files: commit them as usual.
<!-- /ap:steps -->
Claimed: docs/experiments/INDEX.md (the artifact column of the rows it publishes), docs/experiments/*/artifact_url.txt of these pages.
Waiting on owner:
Do not: edit pages, builders or templates (other jobs claim them); publish pages not listed here (the 10-07 contact-gait and DiffMJX pages are local studies by choice); delete any artifact; push.
