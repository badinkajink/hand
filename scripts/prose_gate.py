#!/usr/bin/env python3
"""Mechanical gate of a wording pass over a built page (owner, 2026-10-22: a cheaper model edits prose only "if we can
automate the checks well"). Compares the page after the pass with the page before it and fails when anything but
wording changed:

    numbers   every number of the visible text outside formulas, code and figures (the build time excluded); a
              number may not appear fewer times (--allow lists those whose loss the commit message explains)
    formulas  the LaTeX of every rendered formula (the aria-label of scripts/texsvg.py's SVGs); none may be lost
    links     every href, as a multiset
    media     the counts of figures, tables, films, images and non-formula SVGs
    code      every <code> text; none may be lost

Added numbers, formulas and code, and changed headings, are listed as notes for the judge. With --pairs OUT the
paragraphs whose text changed are written as JSON lines {id, section, before, after}, for a judge that reads only them.

    python3 scripts/prose_gate.py HEAD:docs/overviews/rl_policies.html docs/overviews/rl_policies.html \\
        [--allow 0.380 --allow '\\mu=1'] [--pairs docs/handoff/plain-language-hand/rl_policies_pairs.jsonl]

OLD may be a file or REV:PATH (git show, through git lfs smudge for LFS pages). Exits 1 on FAIL.
"""
from __future__ import annotations

import argparse
import collections
import html
import json
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prose_lint as PL  # noqa: E402

NUMBER = re.compile(r"(?<![\w.])[-−+]?\d+(?:[.,]\d+)*")


def read(spec):
    if re.match(r"^(HEAD|[0-9a-fA-F]{6,40})[~^0-9]*:", spec):
        s = subprocess.run(["git", "show", spec], capture_output=True, check=True).stdout
        if s.startswith(b"version https://git-lfs"):
            s = subprocess.run(["git", "lfs", "smudge"], input=s, capture_output=True, check=True).stdout
        return s.decode("utf-8")
    return open(spec, encoding="utf-8").read()


def parts(src):
    src = re.sub(r"<!--.*?-->", " ", src, flags=re.S)
    tex = re.findall(r"<svg\b[^>]*class=\"tex[^\"]*\"[^>]*aria-label=\"([^\"]*)\"", src)
    out = {"formulas": collections.Counter(re.sub(r"(\\[,;!]|[\s,.;:])+$", "", html.unescape(t)) for t in tex),
           "links": collections.Counter(re.findall(r"<a\b[^>]*\bhref\s*=\s*\"([^\"]*)\"", src)),
           "code": collections.Counter(re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", c))).strip()
                                       for c in re.findall(r"<code\b[^>]*>(.*?)</code>", src, re.S)),
           "headings": [re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", h))).strip()
                        for h in re.findall(r"<h[23]\b[^>]*>(.*?)</h[23]>", src, re.S)],
           "media": {"figure": len(re.findall(r"<figure\b", src)), "table": len(re.findall(r"<table\b", src)),
                     "video": len(re.findall(r"<video\b", src)), "img": len(re.findall(r"<img\b", src)),
                     "svg": len(re.findall(r"<svg\b", src)) - len(tex)}}
    vis = re.sub(r"<(script|style|svg|code|pre)\b.*?</\1>", " ", src, flags=re.S | re.I)
    text = html.unescape(re.sub(r"<[^>]+>", " ", vis))
    text = re.sub(r"built \d{4}-\d{2}-\d{2} \d{2}:\d{2}", " ", text)
    out["numbers"] = collections.Counter(NUMBER.findall(text))
    return out


def diff(a, b):
    return ({k: v - b.get(k, 0) for k, v in a.items() if v > b.get(k, 0)},
            {k: v - a.get(k, 0) for k, v in b.items() if v > a.get(k, 0)})


def show(d, n=12):
    items = sorted(d.items(), key=lambda kv: str(kv[0]))
    s = "; ".join(f"{str(k)[:60]!r} x{v}" for k, v in items[:n])
    return s + (f"; and {len(items) - n} more" if len(items) > n else "")


def pairs(old_src, new_src, out):
    with tempfile.TemporaryDirectory() as d:
        po, pn = os.path.join(d, "old.html"), os.path.join(d, "new.html")
        open(po, "w", encoding="utf-8").write(old_src)
        open(pn, "w", encoding="utf-8").write(new_src)
        a, b = PL.blocks_of(po), PL.blocks_of(pn)
    def sections(blocks):
        out = []
        for x in blocks:
            if not out or out[-1][0] != x["section"]:
                out.append((x["section"], []))
            out[-1][1].append(x["text"])
        return out
    sa, sb = sections(a), sections(b)
    rows = []
    # sections are matched in order (a reworded heading keeps its place); paragraphs by position within a section,
    # or the whole section as one pair when the pass split or merged paragraphs
    for k in range(max(len(sa), len(sb))):
        (ha, ta), (hb, tb) = (sa[k] if k < len(sa) else ("", [])), (sb[k] if k < len(sb) else ("", []))
        if len(ta) == len(tb):
            for x, y in zip(ta, tb):
                if x != y:
                    rows.append({"id": len(rows), "section": (hb or ha)[:60], "before": x, "after": y})
        elif ta != tb:
            rows.append({"id": len(rows), "section": (hb or ha)[:60], "before": " ".join(ta), "after": " ".join(tb)})
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--allow", action="append", default=[], help="a number or formula whose loss the commit explains")
    ap.add_argument("--pairs", help="write the changed paragraphs to this JSONL file")
    a = ap.parse_args()
    so, sn = read(a.old), read(a.new)
    old, new = parts(so), parts(sn)
    fail, note = [], []
    for k in ("numbers", "formulas", "code"):
        lost, added = diff(old[k], new[k])
        lost = {x: v for x, v in lost.items() if x not in a.allow}
        if lost:
            fail.append(f"{k} lost (restore them, or --allow with the reason in the commit message): {show(lost)}")
        if added:
            note.append(f"{k} added (check each against its sentence): {show(added)}")
    lost, added = diff(old["links"], new["links"])
    if lost or added:
        fail.append(f"links changed: lost {show(lost) or 'none'}; added {show(added) or 'none'}")
    bad = {t: (old["media"][t], new["media"][t]) for t in old["media"] if old["media"][t] != new["media"][t]}
    if bad:
        fail.append(f"media counts differ (old, new): {bad}")
    if old["headings"] != new["headings"]:
        ch = [(x, y) for x, y in zip(old["headings"], new["headings"]) if x != y]
        note.append(f"headings changed: {ch[:6]}" + ("" if len(old["headings"]) == len(new["headings"]) else
                                                     f" (count {len(old['headings'])} -> {len(new['headings'])})"))
    print("FAIL" if fail else "PASS", a.new)
    for line in fail:
        print("  " + line)
    for line in note:
        print("  note: " + line)
    if a.pairs:
        print(f"  pairs: {pairs(so, sn, a.pairs)} changed paragraphs -> {a.pairs}")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
