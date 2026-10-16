Status: active
Task (owner, 2026-10-16): Publish the finished dated HTML result page for the CAD-hand fingertip and control deployment-readiness study, and connect its local page and artifact from the readable documentation.
Read first: docs/handoff/20261016-sim2real_fingertip_readiness.md; docs/experiments/20261016-sim2real_fingertip_readiness/20261016-cad_hand_fingertip_deployment_readiness.html when it exists; docs/experiments/INDEX.md; CLAUDE.md Result pages and ~/.claude/CLAUDE.md Writing.
State (2026-10-16): This personal-login job waits for hand-20261016-sim2real_fingertip_readiness. The main job builds and checks the page locally; only this login can publish its artifact.
<!-- ap:steps: rendered by ~/.claude/autopilot/ap from its ledger; change it with `ap step`, not by hand -->
Steps, in order (the ledger; `ap step done <job> <id> "result"` marks one, `ap step add <job> "text"` adds one):
- [ ] 1. Confirm the main job is done and its page, figures, films and INDEX row exist. Read the full local page and check that its measured/model-derived labels, links, math and media render. If the main job stopped incomplete, mark this step waiting and publish nothing.
- [ ] 2. Publish the dated result page as a new claude.ai artifact from the personal login. Save its URL in docs/experiments/20261016-sim2real_fingertip_readiness/artifact_url.txt and in the INDEX row beside the local page; commit and push the link changes.
- [ ] 3. If the main job changed an already-published overview, republish that overview to its existing artifact URL from personal and verify the new result link. Record the local overview path and URL together in this note's State.
<!-- /ap:steps -->
Claimed: docs/experiments/20261016-sim2real_fingertip_readiness/artifact_url.txt, the artifact column of this study's row in docs/experiments/INDEX.md, the result page's claude.ai artifact, and any existing overview artifacts whose local pages the main job changed.
Waiting on owner: none. A stopped or incomplete main job keeps step 1 waiting.
Do not: edit the result page or its builder, publish an incomplete page, alter another study's INDEX row, create a second artifact for an existing overview, or force-push.
