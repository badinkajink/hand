# AGENTS.md

Standing guidance for this repo is `CLAUDE.md`; read it first (hardware, compute safety, result pages, naming).

Readable docs: open `docs/index.html` in a browser; it links the topic overviews `docs/overviews/<topic>.html` (stable
undated names) and they link the dated result pages. `webpaper/` is retired.

Contact model docs: deck 23 of the contact tutorial (`/home/humanoid/wxie_workspace/contact/tutorial/slides/
23_soft_fingertip_contact.html`) explains `docs/overviews/contact_model.html` step by step. After a change to a derivation,
a quoted number or a section of that overview, run `python3 scripts/contact_overview_page.py --deck-check` and update the
deck in its own repository, or set `DECK_CHECKED` in the builder when nothing in the deck changes.

Result pages carry `docs-topic`, `docs-date` and `docs-summary` meta tags (`scripts/docs_index.py tag`), and the indices
follow them: a builder that writes a page under `docs/experiments/` reruns `scripts/docs_index.py` when it exits, which
rewrites `docs/experiments/INDEX.md`, `docs/index.html`, `docs/pages.html`, the overviews' index tables and
`docs/pages.json`. Commit those with the page; never edit an INDEX.md row by hand.

Large files: media, data rows and detailed result pages under `docs/` are Git LFS files (`.gitattributes`); the home
page, the overviews and the pages marked `!filter` there stay ordinary files. History before 2026-10-09 is in
`badinkajink/hand-archive`.

## Writing

The canonical rules are the Writing section of `~/.claude/CLAUDE.md`; this is a copy for Codex, and the global
file wins where they differ. The examples and history behind each rule are in `~/.claude/writing_reference.md`.
They apply to every page, report, commit message and runbook.

### Chat replies

Plain paragraphs to a colleague (owner, 2026-10-11: "stop using these horrible ass header styles"). No markdown
headers, numbered section titles, bold lead-in labels, horizontal rules or emoji; bold at most one number. Answer in
the first sentence or two, then the supporting facts with their numbers, then what happens next; no closing summary,
no praise of the user's idea. A list only for paths, links or commands, never nested; a table only for a small numeric
comparison. One topic per paragraph, one screen. The elements below apply in chat too.

### Elements of style

Language models write dense prose that people read slowly: things act like people, nouns pile up, words are coined
and every claim carries a caveat. Clark, Martin and Mahowald (arXiv 2610.29162, 2026) measured these habits in
frontier models, Claude among the heaviest users. Each element removes one.

1. **Put the outcome first.** Open with an executive summary: the result and its numbers, then the mechanism, then
   the caveat.
2. **One comparison or step per sentence** (owner, 2026-10-22). Split a sentence over about 30 words, even when the
   total grows; put a list of more than four numbers in a table. "The bed gives 30.76, 3.08 and 1.03 µm/s at
   impratio 100, 1000 and 3000, and 12.31 µm/s at τ = 25 ms" → "The creep is 30.76 µm/s at impratio 100. It falls
   to 3.08 and 1.03 µm/s at 1000 and 3000. A longer τ lowers it as well, to 12.31 µm/s at 25 ms."
3. **Things are acted on; people act** (owner, 2026-10-22). A model, method, quantity or document does not see,
   know, want, decide, let, find, eat or bite. Put the thing acted on first in the passive, about 15 % of sentences,
   or state the physical action: "the coefficient encodes the patch" → "the patch is encoded in the coefficient";
   "the point contacts let it swing to 175°" → "between the point contacts it swings to 175°". Physical verbs stay
   active: the springs push back, the thumb presses the cylinder.
4. **Do not compress.** No stacks of three or more nouns ("reader-patience budget"), no nouns used as verbs
   ("calendar it"), keep "that" in relative clauses, keep the subject near its verb, write no telegraphic fragments:
   "arm 0.91 and 0.93" → "its arm is 0.91 and 0.93 times the elements' at 1 and 3 N".
5. **Coin no terms; define every term.** Use the field's word. Describe a new concept in plain words; if it needs a
   name, define it at first use and keep it. Define every metric, symbol and abbreviation at first use: what is
   measured, how, units, reference value. Result pages open with "Terms and metrics", and every table column and axis
   maps to a defined term.
6. **Emphasize with facts.** No clefts ("It is X that…", "What makes X work is…"), no inversions ("It is not X, it is
   Y"), no "X, not Y" or "rather than" doublets (give both numbers), no triads, no portentous closing sentence or
   aphorism, no moral after a table, no closing recap, at most one em-dash per paragraph, bold only a term or number.
7. **Use plain verbs.** Banned: "holds" as a container verb (the hand holds the tool is fine), "carries" in any
   sense, "serves", "settle", "lands", "lives in", "buys", "pays for", "leverage", "that is not there" (write what
   the model has: "point contact has no torque about the normal"). Banned praise and inflation: notably, importantly,
   crucially, worth noting, it turns out, the key insight, fundamentally, elegant, powerful, robust as praise; and,
   except once where precise, matched, isolate, diagnostic, bounded, conservative, scope, explicitly, directly,
   specifically, interpret, evidence.
8. **State what was done and measured.** Never what is not claimed, how a result should not be read, or what an
   alternative would have done. No hedge stacks, no hedge label in place of a result ("a matched diagnostic" →
   "ACT succeeded in 74/90 rollouts"), no narrating the document, no effort reports. A limitation is stated once, in
   its section; numbers and sample sizes show the rest.
9. **Leave things out.** Decide the two or three takeaways, make each point once, and name the section where its
   detail is (no change IDs or ledgers). A word budget, about 100 words for a lede, counts ideas; the words a split
   adds do not count against it.
10. **Name the subject in titles.** Titles and headings are descriptive noun phrases that would work as IROS section
    headings: "Wrist geometry and grasp preload on the UR5e bench", never "A Hundred Millimetres of Wrist", "How well
    it works" or "The X and the Y"; no cute half after a colon. Date-prefix filenames: `20260904-real_v1_bench.html`.
11. **Report failures flatly.** Negative results in the same voice as positive ones. A "Future work" section lists
    actionable items naming the measurement, script or flag and what would falsify the claim.
12. **Set mathematics in LaTeX.** Equations and algorithms on a page are LaTeX rendered at build time
    (`scripts/texsvg.py`), never code font or `<sub>`; an algorithm is an algorithmic block. Code font is for file
    names, identifiers and commands.
13. **Use sentence case** for labels, table text, legends and axis titles.

Before publishing, search the text for the banned words, read the lede alone for the outcome and its numbers, and ask
of each paragraph what it adds. In the contact repository `docs/handoff/20261022-passive_voice_deck.py measure` and
`length` count the passive share and the sentence lengths of a deck.
