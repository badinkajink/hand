# AGENTS.md

Standing guidance for this repo is `CLAUDE.md`; read it first (hardware, compute safety, result pages, naming).

## Writing

The canonical rules are the **Writing** section of `~/.claude/CLAUDE.md`; this is a condensed copy, and the global
file wins where they differ. They apply to every page, report, commit message and runbook.

- Write as a researcher: plain declarative sentences, the technical noun as subject, numbers with units, the finding
  before the story. The style reference is `paper/hand_iros26-4.pdf`.
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
  - vague metaphorical verbs ("carries", "lives in", "buys");
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
- State negative results flatly. A "what this does not settle" section lists actionable items, each naming the
  measurement, script or flag.
