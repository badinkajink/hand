#!/usr/bin/env python3
"""Indices of the documentation, rebuilt from the result pages themselves (owner, 2026-10-20: "we dont have a good way
to automate index updates ... pages that have a particular category tag (and ... the tldr contents summary and date)").

A result page describes itself in its <head>:

    <meta name="docs-topic" content="fingertip">      a key of docs_inventory.TOPICS
    <meta name="docs-date" content="2026-10-20">      optional; else the date of the page's folder
    <meta name="docs-summary" content="...">          the tl;dr: the outcome with its numbers, in a sentence or two
    <meta name="docs-status" content="retracted">     only for a page that is retracted or superseded

tag(html, topic, summary) sets the tags (a builder calls it before writing; `docs_index.py tag PAGE --topic KEY` tags a
built page). A page written without them gets them when its builder's process ends (see below): the topic of its folder
(docs_inventory.FOLDER_TOPIC, else its INDEX.md row), the folder's date and the first two sentences of its lede
(<p class="lede">) as the summary. The pages built before 2026-10-20 were tagged the same way (`docs_index.py tag-all`).

refresh() rewrites every index from docs_inventory.records():
  - docs/experiments/INDEX.md: a row for every page that has none, in date order; the row of a page that carries
    docs-summary follows the page (title and summary); other rows stay as written;
  - docs/index.html (with the newest pages) and docs/pages.html (with a search over titles and summaries), through
    scripts/docs_home_page.py;
  - the index table of each topic overview built by scripts/topic_overview_page.py, between its <!--docs-index-->
    markers, without rebuilding the overview's figures: the pages its text module lists, then every other page of the
    topic with its summary;
  - docs/pages.json, the same records for scripts and agents.
A page in docs/ cannot read other files when opened from the file system, so the pages are found here, in Python, and
written into the indices; the search on pages.html runs over a list embedded in it.

retro_style.apply, which every builder calls, runs refresh() when the builder's process ends if the process wrote a page
under docs/experiments/; DOCS_REFRESH=0 turns that off. Commit the changed indices with the page.

    python3 scripts/docs_index.py                 # refresh every index
    python3 scripts/docs_index.py --check         # list what a refresh would change, write nothing
    python3 scripts/docs_index.py tag PAGE.html --topic fingertip [--summary "..."] [--date 2026-10-20]
    python3 scripts/docs_index.py tag-all         # tag every page that lacks a tag, from the inventory
"""
from __future__ import annotations

import argparse
import html as htmllib
import importlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import docs_inventory as D  # noqa: E402

ROOT = D.ROOT
INDEX_MD = os.path.join(D.EXP, "INDEX.md")
PAGES_JSON = os.path.join(ROOT, "docs", "pages.json")
BLOCK_RE = re.compile(r"<!--docs-index-->.*?<!--/docs-index-->", re.S)
UNPUBLISHED = "*(local study; not published)*"


# ------------------------------------------------------------------------------------------------ page tags

def tag(html: str, topic: str | None = None, summary: str | None = None, date: str | None = None,
        status: str | None = None) -> str:
    """The page with its docs-topic, docs-summary, docs-date (and docs-status) meta tags set, replacing any it had, after
    its charset."""
    if topic is not None and topic not in D.TOPIC_TITLE:
        raise ValueError(f"unknown topic {topic!r}; one of {sorted(D.TOPIC_TITLE)}")
    for name, value in (("topic", topic), ("summary", summary), ("date", date), ("status", status)):
        if value is None:
            continue
        el = f'<meta name="docs-{name}" content="{htmllib.escape(" ".join(str(value).split()), quote=True)}">'
        pat = re.compile(rf'<meta\s+name="docs-{name}"\s+content="[^"]*"\s*/?>', re.I)
        if pat.search(html):
            html = pat.sub(lambda _: el, html, count=1)
            continue
        # after the charset, else after <head>, else (a page written as an artifact fragment, which opens with its
        # <title>) after the title, so the tag stays in the head the parser implies
        m = (re.search(r"<meta\s+charset[^>]*>", html, re.I) or re.search(r"<head(?=[\s>])[^>]*>", html, re.I)
             or re.search(r"</title>", html, re.I))
        html = html[:m.end()] + "\n" + el + html[m.end():] if m else el + "\n" + html
    return html


def tag_pages(paths=None) -> list[str]:
    """Add the tags a page lacks, from the inventory's record of it (topic, date, lede summary, status); paths limits
    this to those pages. Returns the pages changed. A Git LFS pointer is left alone."""
    want = None if paths is None else {os.path.relpath(os.path.abspath(p), ROOT) for p in paths}
    done = []
    for r in D.records():
        if r["tagged"] or (want is not None and r["file"] not in want):
            continue
        path = os.path.join(ROOT, r["file"])
        html = open(path, encoding="utf-8", errors="replace").read()
        if html.startswith("version https://git-lfs"):
            continue
        have = {k.lower() for k, _ in D.META_RE.findall(html[:html.find("</head>")] if "</head>" in html else "")}
        new = tag(html, topic=r["topic"] if "topic" not in have and r["topic"] else None,
                  summary=r["summary"] if "summary" not in have and r["summary"] else None,
                  date=r["date"] if "date" not in have else None,
                  status=r["status"] if "status" not in have and r["status"] != "current" else None)
        if new != html:
            st = os.stat(path)
            _write(path, new)
            os.utime(path, (st.st_atime, st.st_mtime))   # a tag is not an edit: caches keyed on the page's time stay valid
            done.append(r["file"])
    return done


# ------------------------------------------------------------------------------------------------ INDEX.md

def _md(text: str) -> str:
    return " ".join(text.replace("|", "/").split())


def _row(r: dict, artifact_cell: str | None = None) -> str:
    rel = os.path.relpath(os.path.join(ROOT, r["file"]), D.EXP)
    subject = r["page_title"] or r["title"]
    if r["status"] == "retracted":
        subject += " **(RETRACTED)**"
    elif r["status"] == "superseded":
        subject += " (superseded)"
    if r["summary"]:
        subject = subject.rstrip(".") + ". " + r["summary"]
    art = artifact_cell or (f"[artifact]({r['artifact']})" if r["artifact"] else UNPUBLISHED)
    return (f"| {r['date']} | {D.TOPIC_TITLE.get(r['topic'], 'Unsorted')} | {_md(subject)} | [{rel}]({rel}) | "
            f"{art.strip()} |")


def index_md(recs: list[dict]) -> tuple[str, list[str]]:
    """The new text of INDEX.md and a line per change."""
    text = open(INDEX_MD, encoding="utf-8").read()
    lines = text.split("\n")
    hdr = next(i for i, l in enumerate(lines) if re.match(r"\|\s*Date\s*\|", l))
    i = hdr + 2
    while i < len(lines) and lines[i].startswith("|"):
        i += 1
    rows, end = lines[hdr + 2:i], i
    name_of = []
    for l in rows:
        m = re.search(r"\(([^)\s]+\.html)\)", l)
        name_of.append(os.path.basename(m.group(1)) if m else None)
    at = {n: k for k, n in enumerate(name_of) if n}
    changes = []
    for r in recs:
        name = os.path.basename(r["file"])
        if name in at:
            if r["summary_tagged"]:
                cells = rows[at[name]].strip().strip("|").split("|")
                new = _row(r, cells[-1] if cells and "artifact" in cells[-1] else None)
                if new != rows[at[name]]:
                    rows[at[name]] = new
                    changes.append(f"INDEX.md: row of {name} follows the page")
            continue
        new = _row(r)
        k = next((k for k, l in enumerate(rows) if (l[2:12] if l.startswith("| 20") else "9999") < r["date"]), len(rows))
        rows.insert(k, new)
        name_of.insert(k, name)
        at = {n: j for j, n in enumerate(name_of) if n}
        changes.append(f"INDEX.md: row added for {name}")
    return "\n".join(lines[:hdr + 2] + rows + lines[end:]), changes


# ------------------------------------------------------------------------------------------------ overviews

def overview_blocks(recs: list[dict]) -> list[tuple[str, str, str]]:
    """(path, old text, new text) of each topic overview whose index block would change."""
    import retro_style
    import topic_overview_page as TOP
    out = []
    for key, fname in D.OVERVIEW.items():
        path = os.path.join(ROOT, "docs", "overviews", fname)
        try:
            M = importlib.import_module(f"overview_{key}_text")
        except ModuleNotFoundError:
            continue                                 # an overview with its own builder (the contact model)
        if not os.path.exists(path):
            continue
        old = open(path, encoding="utf-8").read()
        if not BLOCK_RE.search(old):
            continue                                 # built before the markers: the next full build adds them
        block = "<!--docs-index-->" + retro_style.sentence_case_headers(TOP.index_table(M, recs, quiet=True)) + "<!--/docs-index-->"
        new = BLOCK_RE.sub(lambda _: block, old, count=1)
        if new != old:
            out.append((path, old, new))
    return out


# ------------------------------------------------------------------------------------------------ refresh

def manifest(recs: list[dict]) -> str:
    keep = ("date", "file", "title", "page_title", "topic", "summary", "builder", "artifact", "status", "mb")
    return json.dumps([{k: r.get(k) for k in keep} for r in sorted(recs, key=lambda r: (r["date"], r["file"]), reverse=True)],
                      indent=1, ensure_ascii=False) + "\n"


def _write(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def refresh(check: bool = False, quiet: bool = False, written=None) -> list[str]:
    """Rebuild every index (see the module docstring); with check, only report. written: pages a builder just wrote,
    which get the tags they lack first. Returns the changes."""
    os.environ["DOCS_INDEX_RUNNING"] = "1"           # the builders called here do not schedule another refresh
    changes = []
    if written and not check:
        changes += [f"{p}: docs-* tags" for p in tag_pages(written)]
    recs = D.records()
    text, ch = index_md(recs)
    if ch:
        changes += ch
        if not check:
            _write(INDEX_MD, text)
            recs = D.records()                      # the inventory takes the titles from INDEX.md's rows
    for path, _, new in overview_blocks(recs):
        changes.append(f"{os.path.relpath(path, ROOT)}: index of linked pages")
        if not check:
            _write(path, new)
    js = manifest(recs)
    if not os.path.exists(PAGES_JSON) or open(PAGES_JSON, encoding="utf-8").read() != js:
        changes.append("docs/pages.json")
        if not check:
            _write(PAGES_JSON, js)
    if not check:
        import docs_home_page
        for path in docs_home_page.main(quiet=True):
            changes.append(os.path.relpath(path, ROOT))
    if not quiet:
        print("\n".join(changes) if changes else "every index is up to date")
    return changes


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    t = sub.add_parser("tag", help="set a built page's docs-* meta tags in place")
    t.add_argument("page")
    t.add_argument("--topic")
    t.add_argument("--summary")
    t.add_argument("--date")
    ap.add_argument("--check", action="store_true", help="report what a refresh would change")
    sub.add_parser("tag-all", help="add the tags every page lacks, from the inventory")
    a = ap.parse_args()
    if a.cmd == "tag":
        html = open(a.page, encoding="utf-8").read()
        _write(a.page, tag(html, a.topic, a.summary, a.date))
        print(f"tagged {a.page}")
        return
    if a.cmd == "tag-all":
        done = tag_pages()
        print(f"tagged {len(done)} pages")
        return
    refresh(check=a.check)


if __name__ == "__main__":
    main()
