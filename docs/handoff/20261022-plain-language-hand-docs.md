Status: active
Task (owner, 2026-10-22): "make a pass over the hand docs ... it's really important when explaining math to people not as familiar with the math to follow these communicability guidelines"; then "we can lower the model quality if we can automate the checks well. but we can also offload work onto this login, and then find a good stopping point and start back up when weekly reset hits". Queued behind the contact decks' pass (contact-20261022-plain-language-decks).
Read first: docs/handoff/20261022-plain-language-hand-procedure.md (the per-page procedure and the judge's prompt); ~/.claude/CLAUDE.md section Writing and ~/.claude/writing_reference.md; CLAUDE.md sections Writing and Result pages; the docstrings of scripts/prose_lint.py and scripts/prose_gate.py; commit 2c763cf6 (the contact overview's pass, the worked example).
State (2026-10-22 15:50, window): nothing started. The contact overview is done (2c763cf6: mean 26.9 to 16.7 words a sentence, over 30 words 34.9 to 3.9 %). Baseline of the pages below in docs/handoff/plain-language-hand/20261022-run_table.md.
Next:
1. docs/overviews/rl_policies.html (scripts/overview_policies_text.py; topic_overview_page.py policies), per the procedure.
2. docs/overviews/printed_fingertips.html (scripts/overview_fingertip_text.py; key fingertip), per the procedure.
3. docs/overviews/hand_plant_bench.html (scripts/overview_hardware_text.py; key hardware), per the procedure.
4. docs/overviews/grasp_to_gait_chain.html (scripts/overview_chain_text.py; key chain), per the procedure.
5. docs/overviews/reorientation.html (scripts/overview_mechanisms_text.py; key mechanisms), per the procedure.
6. docs/overviews/morphology_design.html (scripts/overview_design_text.py; key design), per the procedure.
7. docs/overviews/code_and_methods.html (scripts/overview_code_text.py; key code), per the procedure.
8. docs/index.html (scripts/docs_home_page.py), per the procedure.
9. Result pages: run `python3 scripts/prose_lint.py --stats` on every page under docs/experiments/*/ into the run table, then add one step per page that is above target with `ap step add`, the pages the topic overviews link first and the densest first within them.
10. Close: the run table's totals, one sentence in CLAUDE.md section Writing, Status done.
Claimed: scripts/overview_*_text.py, docs/overviews/ except contact_model.html, scripts/docs_home_page.py, docs/index.html, docs/handoff/plain-language-hand/, and the builder, template, text module and page of the result-page step in hand.
Waiting on owner: republishing the edited pages' artifacts (they belong to the personal login; this job builds local pages only); the owner's reading of the contact overview (2c763cf6): if its style is wrong, the owner disarms this job before it starts.
Do not: change a number, computation, claim, link, formula's meaning, figure or table value; rebuild pages other than the step's; run GPU jobs; force-push; edit the contact repository.
