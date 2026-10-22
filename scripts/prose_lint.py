#!/usr/bin/env python3
"""Flag the phrasings the owner's Writing rules (~/.claude/CLAUDE.md) ban, and count words, in a page or a text file;
measure a page's sentences and flag the ones to rewrite under the elements of style.

    python3 scripts/prose_lint.py PAGE.html [PAGE2.html ...]     # banned phrasings per page (no spaCy)
    python3 scripts/prose_lint.py --lede PAGE.html                # word count of the lede alone
    python3 scripts/prose_lint.py --stats PAGE.html ...           # one row per page: sentence lengths, grade, passive
                                                                  # share, numbers per sentence, flags per 1000
    python3 scripts/prose_lint.py --top 40 PAGE.html              # the 40 worst sentences, worst first, with their
                                                                  # section and their line in the builder's sources
    python3 scripts/prose_lint.py --flags PAGE.html               # one JSON line per sentence with its flags
    python3 scripts/prose_lint.py --terms PAGE.html               # abbreviations, symbols and repeated terms with
                                                                  # their first use, for a definition check

The last four modes parse sentences with spaCy (en_core_web_sm 3.8.0) and re-run themselves under `uv run` with it
when it is missing; the first run downloads it into uv's cache. The flags follow the contact repository's
docs/handoff/20261022-passive_voice_deck.py, with two more for the density of the hand's pages:

    long    more than 30 words (element 2)
    nums    more than four numbers (element 2: a table)
    math    more than three inline formulas (one step per sentence)
    anim    a thing as the subject of a verb of perception, intention, speech or the body (element 3)
    stack   three or more nouns in a row (element 4)
    relcl   a relative clause without that, which or who (element 4)
    cleft   a cleft or an inversion (element 6)
    notx    ", not", "rather than" or "not ..., but" (element 6)
    dash    an em-dash in a paragraph with more than one (element 6)
    banned  a word of element 7

The flags are deliberately broad: a flag marks a sentence to read, and most long or number-heavy sentences of a
derivation are worth splitting. A clean report does not make a page clear; the reader, the two or three takeaways and
the word budget are decided before drafting.

Accuracy (2026-10-22): 100 sentences drawn at random from six topic overviews were labelled blind by Claude, 51 of them
as needing a rewrite (data and scripts in docs/experiments/20261022-prose_flagger_eval/). A sentence with any flag needs
a rewrite with precision 0.76 and recall 0.82. Ranked by the score of --top, the sentences to rewrite come first with a
ROC area of 0.89, and those to split with 0.98. Local models ranked no better. Qwen3.5-4B answered "rewrite" for 94 of
the 100. Qwen3.5-9B in 4 bits ranked coined or undefined terms at 0.84 and animism at 0.86, where these flags give 0.50
and 0.65, but the whole question at 0.63; added to the score it gives 0.90. No method found compressed prose (noun
stacks, telegraphic fragments) better than 0.61. The flags find the sentences; the rewriting and the judgement of it
still need a reader.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from html.parser import HTMLParser

WORDS = ["serves", "serve as", "settle", "worth noting", "importantly", "crucially", "notably", "it turns out",
         "key insight", "fundamentally", "elegant", "powerful", "leverage", "matched", "isolate", "diagnostic", "bounded",
         "conservative", "scope", "explicitly", "directly", "specifically", "interpret", "evidence", "carries", "lives in",
         "buys", "pays for", "lands"]
PATTERNS = [
    (r"\brather than\b", "'rather than' doublet"),
    (r"\b(is|are|was|were) not \w+(,| but| it is| they are)", "'X, not Y' inversion"),
    (r"\w, not (a |an |the )?\w+", "'X, not Y' contrast"),
    (r"\bnot only\b", "'not only' construction"),
    (r"\b(should|must) not be read\b|\bdoes not establish\b|\bwe do not claim\b|\bis not a (detail|tuning)\b", "scope bookkeeping"),
    (r"\bIt is (\w+ ){0,6}that\b", "cleft sentence"),
    (r"\bwhat this does not\b", "banned heading"),
    (r"\b(may|might|could) (potentially|possibly)\b", "hedging stack"),
]
SKIP = re.compile(r"<(script|style|svg|code|pre|math)\b.*?</\1>", re.S | re.I)


def text_of(path):
    s = open(path, encoding="utf-8", errors="replace").read()
    if not path.endswith((".html", ".htm")):
        return s, []
    cells = [html.unescape(re.sub(r"<[^>]+>", "", c)).strip() for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", s, re.S)]
    s = SKIP.sub(" ", s)
    s = re.sub(r"<(p|li|h[1-6]|figcaption|dd|dt|td|th|div|br)[^>]*>", "\n", s)
    return html.unescape(re.sub(r"<[^>]+>", "", s)), cells


def lede_of(path):
    s = open(path, encoding="utf-8", errors="replace").read()
    m = re.search(r'<p class="lede"[^>]*>(.*?)</p>', s, re.S)
    return html.unescape(re.sub(r"<[^>]+>", "", SKIP.sub(" ", m.group(1)))) if m else ""


def report(path):
    t, cells = text_of(path)
    paras = [p.strip() for p in t.split("\n") if len(p.split()) > 3]
    hits = []
    for i, p in enumerate(paras):
        low = p.lower()
        for w in WORDS:
            if re.search(r"\b" + re.escape(w) + r"\b", low):
                hits.append((w, p))
        for rx, name in PATTERNS:
            if re.search(rx, p, re.I):
                hits.append((name, p))
        if p.count("—") > 1:
            hits.append(("more than one em dash", p))
        if re.search(r"\?\s*$", p) or re.search(r"\?\s+[A-Z]", p):
            hits.append(("question in prose", p))
    lower_cells = [c for c in cells if c and c[0].isalpha() and c[0].islower() and not re.match(r"^[a-z_]+\(|^\w+\.\w+", c)]
    words = sum(len(p.split()) for p in paras)
    print(f"{path}: {words} words of prose, lede {len(lede_of(path).split())} words, {len(hits)} flags, "
          f"{len(lower_cells)} table cells in lower case")
    for name, p in hits:
        print(f"  [{name}] {p[:160]}")
    for c in lower_cells[:12]:
        print(f"  [lower-case cell] {c[:80]}")
    return len(hits)


# ---- sentences and flags (spaCy) ---------------------------------------------------------------------------------

SPACY = ["uv", "run", "--no-project", "--python", "3.12", "--with", "spacy==3.8.7", "--with", "textstat", "--with",
         "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/"
         "en_core_web_sm-3.8.0-py3-none-any.whl", "python", "-I"]

AGENT_VERBS = {
    "see", "feel", "know", "want", "need", "decide", "choose", "prefer", "try", "refuse", "ignore", "miss", "notice",
    "find", "learn", "remember", "forget", "expect", "hope", "like", "believe", "think", "understand", "care", "ask",
    "tell", "say", "report", "answer", "give", "take", "pay", "buy", "earn", "win", "lose", "allow", "let", "accept",
    "reject", "recognize", "agree", "disagree", "insist", "demand", "promise", "claim", "admit", "assume", "predict",
    "guess", "treat", "read", "count", "keep", "hold", "carry", "get", "make", "handle", "obey", "follow", "honour",
    "honor", "respect", "forgive", "pick", "catch", "fight", "resist", "protect", "help", "serve", "own", "use",
    "spend", "save", "afford", "trust", "settle", "land", "live", "buy", "eat", "swallow", "bite", "chew", "kill", "die",
    "breathe", "hide", "starve", "feed", "suffer", "sleep", "wake",
}
PEOPLE = {"we", "you", "i", "reader", "readers", "owner", "author", "authors", "they", "he", "she", "one", "user",
          "users", "codex", "operator", "student", "students"}
BANNED_LEMMAS = {"carry", "serve", "settle", "buy", "leverage", "isolate", "interpret"}
# "hold" only as a container verb: a page, file, folder or model that holds things (the hand holds the tool)
CONTAINERS = {"page", "table", "file", "folder", "section", "list", "package", "note", "record", "row", "module",
              "script", "directory", "repository", "log", "array", "dict", "entry", "column", "figure", "document"}
BANNED_VERB_LEMMAS = {"land"}
BANNED_WORDS = {"notably", "importantly", "crucially", "fundamentally", "elegant", "elegantly", "powerful", "robust",
                "robustly", "matched", "diagnostic", "diagnostics", "bounded", "conservative", "scope", "explicitly",
                "directly", "specifically", "evidence"}
BANNED_PHRASES = [r"\bworth noting\b", r"\bit turns out\b", r"\bkey insight\b", r"\blives? in\b", r"\blived in\b",
                  r"\bpa(?:y|ys|id) for\b", r"\bnot there\b"]
RELATIVIZERS = {"that", "which", "who", "whom", "whose", "where", "when", "why"}
FLAGS = ["long", "nums", "math", "anim", "stack", "relcl", "cleft", "notx", "dash", "banned"]
WEIGHT = {"long": 2, "nums": 2, "math": 1, "anim": 2, "stack": 1, "relcl": 1, "cleft": 2, "notx": 2, "dash": 1,
          "banned": 2}
SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
WORD = re.compile(r"[A-Za-z0-9]")
NUMBER = re.compile(r"(?<![A-Za-z_\d.])[-−+]?\d+(?:[.,]\d+)*")

PROSE = {"p", "li", "figcaption", "dd", "blockquote"}
SKIP_TAGS = {"script", "style", "table", "pre", "nav", "button", "select", "label", "math"}
HEADS = {"h1", "h2", "h3", "h4"}


class Prose(HTMLParser):
    """Prose blocks of a result page: paragraphs, list items, captions and glossary definitions, each with the heading
    of its section. A formula (an inline SVG from scripts/texsvg.py) becomes the word "X" and its LaTeX is kept."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks, self.buf, self.btag, self.skip, self.svg = [], None, None, 0, 0
        self.head, self.hbuf, self.tex = "", None, []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "svg":
            if self.svg == 0 and not self.skip:
                label = a.get("aria-label") or ""
                if "tex" in (a.get("class") or "").split() and self.buf is not None:
                    self.buf.append(" X ")
                    self.tex.append(label)
            self.svg += 1
            return
        if self.svg:
            return
        if tag in SKIP_TAGS:
            self.skip += 1
        elif tag in HEADS and not self.skip:
            self.hbuf = []
        elif tag in PROSE and self.buf is None and not self.skip:
            self.buf, self.btag, self.tex = [], tag, []
            self.cls = a.get("class") or ""

    def handle_endtag(self, tag):
        if tag == "svg":
            self.svg = max(0, self.svg - 1)
            return
        if self.svg:
            return
        if tag in SKIP_TAGS and self.skip:
            self.skip -= 1
        elif tag in HEADS and self.hbuf is not None:
            self.head = re.sub(r"\s+", " ", "".join(self.hbuf)).strip()
            self.hbuf = None
        elif self.buf is not None and tag == self.btag:
            text = re.sub(r"\s+", " ", "".join(self.buf)).strip()
            if text:
                self.blocks.append({"section": self.head, "tag": tag, "cls": self.cls, "text": text, "tex": self.tex})
            self.buf = None

    def handle_data(self, data):
        if self.svg or self.skip:
            return
        if self.hbuf is not None:
            self.hbuf.append(data)
        if self.buf is not None:
            self.buf.append(data)


def blocks_of(path):
    p = Prose()
    p.feed(open(path, encoding="utf-8", errors="replace").read())
    p.close()
    return p.blocks


def nwords(s):
    return len([w for w in s.split() if WORD.search(w)])


def flags_of(sent, passive, nw, block_dashes):
    f, s = {}, sent.text.strip()
    if nw > 30:
        f["long"] = nw
    nums = NUMBER.findall(s)
    if len(nums) > 4:
        f["nums"] = len(nums)
    nx = len(re.findall(r"\bX\b", s))
    if nx > 3:
        f["math"] = nx
    if not passive:
        for t in sent:
            if t.dep_ == "nsubj" and t.head.pos_ in ("VERB", "AUX") and t.head.lemma_.lower() in AGENT_VERBS \
                    and t.lower_ not in PEOPLE and t.pos_ in ("NOUN", "PROPN") and t.text != "X":
                f["anim"] = f"{t.text} {t.head.text}"
                break
    run, stacks = [], []
    for t in sent:
        if t.text == "-" and run:
            continue
        if t.pos_ in ("NOUN", "PROPN") and t.text != "X":
            run.append(t)
            if t.dep_ != "compound":
                if len(run) >= 3:
                    stacks.append(" ".join(x.text for x in run))
                run = []
        else:
            run = []
    if stacks:
        f["stack"] = stacks
    rel = [f"{t.head.text} | {' '.join(x.text for x in t.subtree if x.i <= t.i)}" for t in sent
           if t.dep_ == "relcl" and not any(x.lower_ in RELATIVIZERS for x in t.subtree if x.i <= t.i)]
    if rel:
        f["relcl"] = rel
    if re.match(r"(It|This|That)\s+(is|was)\b.*\b(that|who|which)\b", s) or re.match(r"(It|This|That)\s+(is|was)\s+not\b", s) \
            or (re.match(r"What\b.*\b(is|was)\b", s) and not s.endswith("?")):
        f["cleft"] = " ".join(s.split()[:6])
    m = re.search(r",\s+not\s+\w|\brather than\b|\bnot\b[^.;:]*,\s*but\b", s)
    if m:
        f["notx"] = m.group(0)[:40]
    if block_dashes > 1 and "—" in s:
        f["dash"] = block_dashes
    ban = [t.text for t in sent if t.lemma_.lower() in BANNED_LEMMAS
           or (t.lemma_.lower() in BANNED_VERB_LEMMAS and t.pos_ == "VERB") or t.lower_ in BANNED_WORDS
           or (t.lemma_.lower() == "hold" and t.pos_ in ("VERB", "AUX")
               and any(c.dep_ == "nsubj" and c.lemma_.lower() in CONTAINERS for c in t.children))]
    ban += [m.group(0) for p in BANNED_PHRASES for m in re.finditer(p, s, flags=re.I)]
    if ban:
        f["banned"] = ban
    return f


def sentences(nlp, path):
    rows = []
    for bi, b in enumerate(blocks_of(path)):
        doc = nlp(b["text"])
        dashes = b["text"].count("—")
        for whole in doc.sents:
            parts = SPLIT.split(whole.text)
            for sent in [whole] if len(parts) == 1 else [nlp(p) for p in parts]:
                if sum(t.is_alpha for t in sent) < 3:
                    continue
                passive = any(t.dep_ in ("nsubjpass", "csubjpass", "auxpass") for t in sent)
                nw = nwords(sent.text)
                fl = flags_of(sent, passive, nw, dashes)
                rows.append({"section": b["section"], "block": bi, "tag": b["tag"], "lede": b["cls"] == "lede",
                             "words": nw, "passive": passive, "flags": fl,
                             "score": sum(WEIGHT[k] for k in fl) + max(0, nw - 30) / 10, "text": sent.text.strip()})
    return rows


def stats(path, rows):
    import statistics
    import textstat
    n = max(len(rows), 1)
    w = [r["words"] for r in rows]
    toks = [x for r in rows for x in r["text"].split() if WORD.search(x)]
    syl = sum(textstat.syllable_count(re.sub(r"[^A-Za-z]", "", x)) or 1 for x in toks)
    blocks = {}
    for r in rows:
        blocks[r["block"]] = blocks.get(r["block"], 0) + r["words"]
    nums = [len(NUMBER.findall(r["text"])) for r in rows]
    out = {"page": os.path.basename(path), "words": sum(w), "sentences": len(rows),
           "mean": round(sum(w) / n, 1), "median": statistics.median(w) if w else 0,
           "over30": round(100 * sum(x > 30 for x in w) / n, 1),
           "grade": round(0.39 * len(toks) / n + 11.8 * syl / max(len(toks), 1) - 15.59, 1),
           "passive": round(100 * sum(r["passive"] for r in rows) / n, 1),
           "nums_per_sent": round(sum(nums) / n, 2),
           "para_mean": round(sum(blocks.values()) / max(len(blocks), 1)),
           "para_over120": round(100 * sum(v > 120 for v in blocks.values()) / max(len(blocks), 1), 1)}
    for k in FLAGS:
        out[k] = round(1000 * sum(k in r["flags"] for r in rows) / n)
    out["any"] = round(1000 * sum(bool(r["flags"]) for r in rows) / n)
    return out


STAT_COLS = [("page", "page"), ("words", "words"), ("sentences", "sent."), ("mean", "mean"), ("median", "median"),
             ("over30", "% >30"), ("grade", "grade"), ("passive", "% passive"), ("nums_per_sent", "numbers/sent."),
             ("para_mean", "words/para."), ("para_over120", "% para. >120")] + [(k, k) for k in FLAGS] + [("any", "any")]


def sources_of(path, extra):
    """The builder, template and text module named in the page ("builder scripts/X.py"), plus any given."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = list(extra or [])
    if not src:
        page = open(path, encoding="utf-8", errors="replace").read()
        for m in dict.fromkeys(re.findall(r"scripts/(\w+_page)\.py", page)):
            for cand in (f"scripts/{m}.py", f"scripts/{m}.template.html", f"scripts/{m.replace('_page', '_text')}.py"):
                if os.path.exists(os.path.join(root, cand)):
                    src.append(cand)
            break
    return [(s, open(os.path.join(root, s) if not os.path.isabs(s) else s, encoding="utf-8").read()) for s in src]


GAP = r"(?:[^A-Za-z]|&[A-Za-z]+;|<[^>]*>|\\[a-z]|(?<![A-Za-z])f(?=[\"']))+"


def locate(text, srcs):
    """file:line of a sentence in the sources, from runs of four words that sit side by side in the page."""
    toks = re.findall(r"[A-Za-z]+|\S", text)
    runs, cur = [], []
    for t in toks:
        if re.fullmatch(r"[A-Za-z]+", t) and t != "X":
            cur.append(t)
        elif re.fullmatch(r"[,.;:()'’-]", t) and cur:
            cur.append(None)
        else:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    cands = []
    for run in runs:
        ws = [w for w in run if w]
        for i in range(0, max(1, len(ws) - 3), 3):
            if len(ws[i:i + 4]) == 4:
                cands.append(ws[i:i + 4])
    for c in cands[:6] + cands[-3:]:
        rx = re.compile(r"\b" + GAP.join(re.escape(w) for w in c) + r"\b")
        for name, s in srcs:
            m = rx.search(s)
            if m:
                return f"{name}:{s.count(chr(10), 0, m.start()) + 1}"
    return "?"


def terms(nlp, path, rows):
    """Abbreviations, symbols and repeated multi-word terms, each with its count and the sentence of its first use."""
    text = " ".join(r["text"] for r in rows)
    common = {"CPU", "GPU", "RAM", "PLA", "TPU", "CAD", "STL", "PDF", "HTML", "JSON", "API", "USB", "ID", "OK", "MPa",
              "GPa", "kPa", "mN", "Hz", "kHz", "PC", "SR2", "SR²"}
    print("Abbreviations (first use; 'def' when the sentence spells it out in parentheses):")
    seen = {}
    for r in rows:
        for a in re.findall(r"\b(?:[A-Z][A-Za-z]*[A-Z][A-Za-z0-9]*|[A-Z]{2,}[0-9]*)\b", r["text"]):
            if a in common or a in seen or len(a) > 8:
                continue
            seen[a] = r
    for a, r in seen.items():
        n = len(re.findall(r"\b" + re.escape(a) + r"\b", text))
        d = "def" if re.search(r"\(" + re.escape(a) + r"\)|\b" + re.escape(a) + r"\s*\(", r["text"]) else "   "
        print(f"  {d} {a:<10} x{n:<3} [{r['section'][:30]}] {r['text'][:150]}")
    print("\nSymbols (first formula that uses each, from the formulas' LaTeX):")
    sym_first, sym_n = {}, {}
    for b in blocks_of(path):
        for t in b["tex"]:
            for s in set(re.findall(r"\\[A-Za-z]+(?:_\{?[A-Za-z0-9,]+\}?)?|(?<![\\A-Za-z])[A-Za-z](?:_\{?[A-Za-z0-9,]+\}?)?(?![A-Za-z])", t)):
                if s.startswith(("\\frac", "\\left", "\\right", "\\mathrm", "\\text", "\\quad", "\\cdot", "\\approx",
                                 "\\le", "\\ge", "\\sum", "\\int", "\\sqrt", "\\times", "\\dots", "\\,", "\\operatorname",
                                 "\\big", "\\Big", "\\mathbf", "\\boldsymbol", "\\hat", "\\bar", "\\tilde", "\\max",
                                 "\\min", "\\exp", "\\log", "\\ln", "\\to", "\\in", "\\propto", "\\pm", "\\ll", "\\gg",
                                 "\\infty", "\\partial", "\\mathcal", "\\begin", "\\end", "\\qquad", "\\displaystyle")):
                    continue
                sym_n[s] = sym_n.get(s, 0) + 1
                sym_first.setdefault(s, (b["section"], b["text"]))
    for s, (sec, txt) in sorted(sym_first.items(), key=lambda kv: -sym_n[kv[0]]):
        print(f"  {s:<14} x{sym_n[s]:<3} [{sec[:30]}] {txt[:130]}")
    print("\nRepeated terms (noun phrases of two or more words used three times or more; first use):")
    counts, first = {}, {}
    for r in rows:
        for ch in nlp(r["text"]).noun_chunks:
            toks = [t for t in ch if t.pos_ not in ("DET", "PRON", "NUM") and t.text != "X" and not t.is_punct]
            if len(toks) < 2 or any(NUMBER.fullmatch(t.text) for t in toks):
                continue
            k = " ".join(t.lemma_.lower() for t in toks)
            counts[k] = counts.get(k, 0) + 1
            first.setdefault(k, r)
    for k, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        if n >= 3:
            r = first[k]
            print(f"  x{n:<3} {k:<34} [{r['section'][:30]}] {r['text'][:120]}")


def spacy_main(args):
    try:
        import spacy
    except ImportError:
        os.execvp(SPACY[0], SPACY + [os.path.abspath(__file__)] + sys.argv[1:])
    nlp = spacy.load("en_core_web_sm")
    if args.stats:
        out = [stats(p, sentences(nlp, p)) for p in args.paths]
        print("| " + " | ".join(h for _, h in STAT_COLS) + " |")
        print("|" + "---|" * len(STAT_COLS))
        for o in out:
            print("| " + " | ".join(str(o[k]) for k, _ in STAT_COLS) + " |")
        return
    for p in args.paths:
        rows = sentences(nlp, p)
        if args.flags:
            for r in rows:
                print(json.dumps(r, ensure_ascii=False))
        elif args.terms:
            terms(nlp, p, rows)
        elif args.top:
            srcs = sources_of(p, args.src)
            worst = sorted((r for r in rows if r["flags"]), key=lambda r: -r["score"])[:args.top]
            print(f"{p}: {len(worst)} of {sum(bool(r['flags']) for r in rows)} flagged sentences "
                  f"({len(rows)} in all); sources {', '.join(n for n, _ in srcs) or 'none'}")
            for r in worst:
                fl = " ".join(f"{k}={v if not isinstance(v, list) else '|'.join(v)}" for k, v in r["flags"].items())
                print(f"\n[{r['section'][:40]}] {locate(r['text'], srcs)}  {fl}\n  {r['text']}")


def main():
    a = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    a.add_argument("paths", nargs="+")
    a.add_argument("--lede", action="store_true")
    a.add_argument("--stats", action="store_true", help="one table row per page")
    a.add_argument("--flags", action="store_true", help="one JSON line per sentence")
    a.add_argument("--top", type=int, default=0, help="the N worst flagged sentences with their source lines")
    a.add_argument("--terms", action="store_true", help="abbreviations, symbols and repeated terms with first use")
    a.add_argument("--src", nargs="*", help="source files for --top (default: the builder the page names)")
    args = a.parse_args()
    if args.lede:
        for p in args.paths:
            print(len(lede_of(p).split()), p)
        return
    if args.stats or args.flags or args.top or args.terms:
        return spacy_main(args)
    sys.exit(1 if sum(report(p) for p in args.paths) else 0)


if __name__ == "__main__":
    main()
