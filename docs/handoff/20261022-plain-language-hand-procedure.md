Procedure of the plain-language pass over the hand's documentation pages (owner, 2026-10-22)

Every page is edited to read as plain researcher's prose under the thirteen "Elements of style" of `~/.claude/CLAUDE.md`
(section Writing); their examples are in `~/.claude/writing_reference.md`. The worked example is the contact overview,
commit 2c763cf6: each derivation step opens with its result in words, defines every symbol before its equation and says
what the equation means after it, and each sentence carries one comparison or step. An edit changes wording only: never a
number, a computation, a claim, a condition, a link, a formula's meaning, a figure or a table's values. Numbers stay
computed from the data; a number a variable already gives is never typed into the source.

Model (owner, 2026-10-22: a cheaper model is fine "if we can automate the checks well"): the editing runs on Sonnet at
medium effort, because the checks are mechanical (`scripts/prose_gate.py`, the build, `scripts/prose_lint.py`) and the
judge runs on Opus.

Names shared with the contact overview and the contact tutorial's deck 23: the two-pad comparison rig is "the test bed"
(its tasks "task N on the test bed"); the spring model is "the Winkler foundation"; the finger-servo models are "the
servo plant" (the SCS0009's 0.72 N m/rad and 0.18 N m) and "the 1 N m plant". One word never names two things.

Per page:
1. Baseline: `python3 scripts/prose_lint.py --stats PAGE` (its row goes into the run table) and
   `python3 scripts/prose_lint.py --top 40 PAGE`, the sentences to fix with their file and line in the page's sources.
   A topic overview's prose is the module `scripts/overview_KEY_text.py`; a result page's sources are the builder its
   byline names, its `.template.html` and its `_text.py` module if there is one. Read those lines and their paragraphs,
   not the whole builder.
2. Edit the flagged sentences in place, and any paragraph around them that needs it: split a sentence over 30 words or
   with more than four numbers; put the thing acted on first in the passive where a thing sees, decides, lets or finds;
   write out stacked nouns and telegraphic fragments; replace or define coined terms; remove clefts, "X, not Y", "rather
   than" and the banned words. Keep the computed values and their formatting code untouched.
3. Rebuild: `~/.claude/bin/resguard.sh run --mem 4G --cpu 200 -- .venv/bin/python scripts/BUILDER.py [KEY]`
   (`topic_overview_page.py KEY` for an overview, `docs_home_page.py` for the home page). The build prints a
   `prose:` line per page; the target is "within the targets" (mean at most 18 words, at most 5 % over 30, at most 300
   flagged per 1000), or a better line than the baseline when the page is mostly tables and captions.
4. Gate: `python3 scripts/prose_gate.py HEAD:PAGE PAGE --pairs docs/handoff/plain-language-hand/NAME_pairs.jsonl` prints
   PASS. On FAIL restore what it lists. A number or formula reworded into words on purpose passes with `--allow`, and
   the reason goes into the commit message.
5. Judge: one fresh subagent on Opus (the Agent tool with `model: "opus"`), given the judge prompt below, the pairs file
   and the gate's notes. It writes `docs/handoff/plain-language-hand/NAME_judge.jsonl` and edits nothing. Restore or
   repair every pair judged worse, meaning or broken, rebuild and gate again. If fewer than half of the pairs are judged
   better, restore the page's sources and the page (`git checkout`) and write why in the run table.
6. Record and commit: one row in `docs/handoff/plain-language-hand/20261022-run_table.md` (page; words, mean, % over 30,
   grade and flagged per 1000 before and after; the judge's counts). Commit the sources, the page, the index files the
   build refreshed and the `docs/handoff/plain-language-hand/` files with explicit paths, then `git push origin main`.
   Artifacts are not republished here (Waiting on owner).

Judge prompt (fill in the paths): "You judge wording edits to a documentation page of a robotics research project. Its
readers are engineers who know robotics but not this page's mathematics or jargon. Each line of PAIRS is one paragraph
before and after an edit; X stands for a rendered formula. The editing rules are the 'Elements of style' in
~/.claude/CLAUDE.md (section Writing). For each pair answer: better (easier to read, same content), same, worse (harder
to read, longer for no gain, an awkward passive, a lost connection between ideas), meaning (a number, claim, condition,
quantity, term or link changed or lost), or broken (garbled text or markup). Length and passives earn nothing by
themselves. The gate's notes list formulas, numbers and code that the edit added; check each against its sentence.
Write one JSON line per pair to OUT: {"id": ..., "verdict": ..., "why": "at most 15 words"}. Then report the counts and
the ids of every pair judged worse, meaning or broken. Edit no other file."
