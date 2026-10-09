#!/usr/bin/env python3
"""Flag the phrasings the owner's Writing rules (~/.claude/CLAUDE.md) ban, and count words, in a page or a text file.

    python3 scripts/prose_lint.py PAGE.html [PAGE2.html ...]     # one report per page
    python3 scripts/prose_lint.py --lede PAGE.html                # word count of the lede alone

A hit marks a sentence to rewrite; a clean report does not make a page clear. Global clarity comes from deciding the
reader, the two or three takeaways and the word budget before drafting, and from a cutting pass after it.
"""
from __future__ import annotations

import argparse
import html
import re
import sys

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


def main():
    a = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    a.add_argument("paths", nargs="+")
    a.add_argument("--lede", action="store_true")
    args = a.parse_args()
    if args.lede:
        for p in args.paths:
            print(len(lede_of(p).split()), p)
        return
    sys.exit(1 if sum(report(p) for p in args.paths) else 0)


if __name__ == "__main__":
    main()
