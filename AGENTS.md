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

The canonical rules are the **Writing** section of `~/.claude/CLAUDE.md`; this is a condensed copy, and the global
file wins where they differ. They apply to every page, report, commit message and runbook.

- Write as a researcher: plain declarative sentences, the technical noun as subject, numbers with units, the finding
  before the story. The style reference is `paper/hand_iros26-4.pdf`.
- Chat replies are plain paragraphs to a colleague (owner, 2026-10-11): no markdown headers, numbered section titles,
  bold lead-in labels or nested bullets; the answer first, then its numbers, then what happens next; a list only for
  paths, links or commands.
- Open with an executive summary: the outcome and the two or three findings, before any outline. Then answer each point
  in one or two sentences and say where the detail is.
- Use editorial judgement. Leave things out; do not re-summarize, answer every possible objection, or qualify every
  sentence. A document of individually defensible sentences can still be unreadable.
- Define every metric, abbreviation and piece of jargon at first use: what is measured, how, units, reference value.
  Result pages open with a "Terms and metrics" block.
- Titles and headings are descriptive noun phrases ("Components of a fingertip contact model"), never phrases such as
  "What X has to carry" or "How well it works". Date-prefix filenames.
- Delete on sight:
  - "X, not Y" and "we do not claim X; we Y";
  - scope bookkeeping ("should not be read as", "this does not establish", "a reference, not the truth");
  - inflated qualifiers ("matched", "isolate", "diagnostic", "bounded", "conservative", "explicitly", "directly");
  - vague metaphorical verbs ("lives in", "buys", "holds" for "contains"; the grasp sense stays); "carries" in any sense;
  - "that is not there" and its variants ("a torque that is not there"): state what the model has;
  - personified mechanisms;
  - triads;
  - portentous closing sentences;
  - "notably/importantly/crucially";
  - more than one em-dash per paragraph;
  - closing recap sections;
  - hedging stacks.
- State what was done and what was measured. Never state what is not claimed, how a result should not be read, or
  what an alternative would have done.
- Make each point once and name where its detail is; do not restate earlier answers or route through change IDs.
- Delete every qualifier, caveat and method adjective no reader asked for. The numbers and sample sizes already show
  the limits.
- State negative results flatly. A "Future work" section lists actionable items, each naming the measurement, script or
  flag. Never use "serves" or "settle".
- Sentence case for labels, table text, legends and axis titles. Fix a word budget before drafting (a lede of about
  100 words) and cut to it.
