#!/usr/bin/env python3
"""Revisions of the documentation pages (owner, 2026-11-02: "we should also have a changelog for these pages ... we should
still be able to access former versions of the same page via a linked git commit (to github?). it's helpful to know, in
short, how documentation has changed and where we can access older versions").

docs/changelog.html lists every page under docs/ (the home page, the indices, the topic overviews and the dated result
pages) with the commits that changed it, newest first: the date, a one-line summary, and links to the commit and to the
page as that commit left it on GitHub. The summary is the commit's subject unless scripts/docs_changelog_notes.json gives one for
that page and commit; the notes also list the revisions made before a page had a stable name (the contact overview's
dated files of 2026-10-05 to 10-09). A page moved to a new path starts a new history.

docs_index.refresh(), which runs after every page build, rewrites the changelog; it changes only when a commit has
touched a page, and it lists the commits up to the build, so a page's newest commit appears at the next build. Result
pages are Git LFS files: GitHub shows each version's blob page with a "View raw" link, and a clone opens one with
    git show COMMIT:PATH | git lfs smudge > /tmp/page.html

    python3 scripts/docs_changelog.py          # write docs/changelog.html
"""
from __future__ import annotations

import html as htmllib
import json
import os
import re
import subprocess
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import docs_inventory as D  # noqa: E402

ROOT = D.ROOT
OUT = os.path.join(ROOT, "docs", "changelog.html")
NOTES = os.path.join(HERE, "docs_changelog_notes.json")    # docs/**/*.json are LFS files
SKIP = ("docs/changelog.html", "docs/archive/")
RECENT = 30          # commits in the list of recent changes
SUBJECT_MAX = 160    # characters of a commit subject shown as its summary


def esc(s: str) -> str:
    return htmllib.escape(s, quote=True)


def repo_url() -> str:
    url = subprocess.run(["git", "remote", "get-url", "origin"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    m = re.match(r"(?:git@github\.com:|https://github\.com/)([^/]+)/(.+?)(?:\.git)?$", url)
    return f"https://github.com/{m.group(1)}/{m.group(2)}" if m else ""


def history() -> list[dict]:
    """Every commit that changed an HTML file under docs/, newest first: hash, date, subject, files."""
    out = subprocess.run(["git", "-c", "core.quotepath=off", "log", "--format=%x1e%H%x1f%ad%x1f%s", "--date=short",
                          "--name-only", "--", "docs"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    commits = []
    for block in out.split("\x1e")[1:]:
        head, _, names = block.partition("\n")
        h, d, s = head.split("\x1f", 2)
        files = [n for n in names.split("\n") if n.endswith(".html") and not n.startswith(SKIP)]
        if files:
            commits.append({"hash": h, "date": d, "subject": s.strip(), "files": files})
    return commits


def title_of(path: str, recs: dict) -> str:
    if path in recs:
        return recs[path]
    try:
        with open(os.path.join(ROOT, path), encoding="utf-8", errors="replace") as f:
            m = re.search(r"<title>(.*?)</title>", f.read(200_000), re.S)
        if m:
            return htmllib.unescape(re.sub(r"\s+", " ", m.group(1)).strip())
    except OSError:
        pass
    return os.path.basename(path)


def summary(c: dict, path: str, notes: dict) -> str:
    over = notes.get("pages", {}).get(path, {}).get("commits", {})
    for k, v in over.items():
        if c["hash"].startswith(k):
            return v
    s = c["subject"]
    return s if len(s) <= SUBJECT_MAX else s[:SUBJECT_MAX].rsplit(" ", 1)[0] + " &#8230;"


def anchor(path: str) -> str:
    return "p-" + re.sub(r"[^A-Za-z0-9]+", "-", path.removeprefix("docs/").removesuffix(".html")).strip("-")


def rel(path: str) -> str:
    return os.path.relpath(os.path.join(ROOT, path), os.path.dirname(OUT))


def build() -> str:
    gh = repo_url()
    notes = json.load(open(NOTES, encoding="utf-8")) if os.path.exists(NOTES) else {}
    recs = {r["file"]: r["title"] for r in D.records()}
    dates = {r["file"]: r["date"] for r in D.records()}
    commits = history()
    by_page = defaultdict(list)
    for c in commits:
        for f in c["files"]:
            if os.path.exists(os.path.join(ROOT, f)):
                by_page[f].append(c)
    titles = {p: title_of(p, recs) for p in by_page}
    n_rev = sum(len(v) for v in by_page.values())

    def link_commit(c):
        return f'<a href="{gh}/commit/{c["hash"]}">{c["hash"][:9]}</a>' if gh else c["hash"][:9]

    def link_version(c, path):
        return f'<a href="{gh}/blob/{c["hash"]}/{path}">page</a>' if gh else ""

    def page_block(path):
        cs = by_page[path]
        earlier = notes.get("pages", {}).get(path, {}).get("earlier", [])
        head = (f'<h3 id="{anchor(path)}"><a href="{esc(rel(path))}">{esc(titles[path])}</a></h3>'
                f'<p class="meta"><code>{esc(path)}</code> &#183; {len(cs)} revision{"s" if len(cs) != 1 else ""}, '
                f'{cs[-1]["date"]} to {cs[0]["date"]}'
                + (f' &#183; {len(earlier)} earlier dated files' if earlier else "")
                + (f'<br>Latest: {cs[0]["date"]} &#183; {summary(cs[0], path, notes)}' if len(cs) > 1 else "") + "</p>")
        items = "".join(f'<li>{c["date"]} &#183; {summary(c, path, notes)} &#183; {link_commit(c)} '
                        f'{link_version(c, path)}</li>' for c in cs)
        before = "".join(f'<li>{e["date"]} &#183; {esc(e["summary"])} &#183; <a href="{esc(rel(e["file"]))}">dated file</a></li>'
                         for e in earlier)
        if len(cs) == 1 and not earlier:
            return head.replace("</p>", "") + f' &#183; {link_commit(cs[0])} {link_version(cs[0], path)}</p>'
        body = f"<ol class='revs'>{items}</ol>"
        if before:
            body += f"<p>Before the page had a stable name, each revision was a dated file:</p><ol class='revs'>{before}</ol>"
        return head + f"<details><summary>Revisions, newest first</summary>{body}</details>"

    homes = [p for p in ("docs/index.html", "docs/pages.html") if p in by_page]
    overviews = sorted((p for p in by_page if p.startswith("docs/overviews/")), key=lambda p: titles[p])
    results = sorted((p for p in by_page if p not in homes and p not in overviews),
                     key=lambda p: (dates.get(p, by_page[p][-1]["date"]), p), reverse=True)

    recent = []
    for c in commits[:RECENT]:
        pages = [f for f in c["files"] if f in by_page]
        if not pages:
            continue
        shown = ", ".join(f'<a href="#{anchor(f)}">{esc(titles[f])}</a>' for f in pages[:4])
        more = f" and {len(pages) - 4} more" if len(pages) > 4 else ""
        recent.append(f'<li>{c["date"]} &#183; {summary(c, pages[0], notes)} &#183; {link_commit(c)}<br>'
                      f'<span class="pages">{shown}{more}</span></li>')

    most = sorted(by_page, key=lambda p: (-len(by_page[p]), titles[p]))[:15]
    most_rows = "".join(f'<tr><td><a href="#{anchor(p)}">{esc(titles[p])}</a></td><td class="n">{len(by_page[p])}</td>'
                        f'<td>{by_page[p][-1]["date"]}</td><td>{by_page[p][0]["date"]}</td></tr>' for p in most)
    one = sum(1 for p in by_page if len(by_page[p]) == 1)
    last = commits[0]["date"] if commits else ""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Revisions of the documentation pages</title>
<style>
.meta {{ margin-top: 0; font-size: 0.92em; }}
ol.revs li, ol.recent li {{ margin-bottom: 0.25em; }}
.pages {{ font-size: 0.92em; }}
td.n {{ text-align: right; }}
h3 {{ margin-bottom: 0.2em; }}
</style>
</head>
<body>
<p><a href="index.html">SR&#178; Hand documentation</a> &#183; last page change {last}</p>
<h1>Revisions of the SR&#178; Hand documentation pages</h1>
<p class="lede">The {len(by_page)} pages under <code>docs/</code> have {n_rev} revisions in git. {one} pages have one, and
the {len(most)} most revised are listed below. Each revision links to its commit and to the page as that commit left it
on GitHub. A result page is a Git LFS file, so GitHub shows it as a blob page with a &#8220;View raw&#8221; link. In a
clone, <code>git show COMMIT:PATH | git lfs smudge &gt; /tmp/page.html</code> writes an older version to open in a
browser. History before the repository&#8217;s rewrite of 2026-10-09 keeps its old commit hashes in
<a href="https://github.com/badinkajink/hand-archive">hand-archive</a>.</p>

<h2>Recent changes</h2>
<p>The {len(recent)} newest commits that changed a page, with the pages they changed.</p>
<ol class="recent">{"".join(recent)}</ol>

<h2>Most revised pages</h2>
<table><thead><tr><th>Page</th><th>Revisions</th><th>First</th><th>Last</th></tr></thead><tbody>{most_rows}</tbody></table>

<h2>Home page and indices</h2>
{"".join(page_block(p) for p in homes)}

<h2>Topic overviews</h2>
{"".join(page_block(p) for p in overviews)}

<h2>Result pages, newest first</h2>
{"".join(page_block(p) for p in results)}

<footer><p>Built by <code>python3 scripts/docs_changelog.py</code>, which <code>scripts/docs_index.py</code> runs after
every page build. One-line summaries other than the commit subjects are in <code>scripts/docs_changelog_notes.json</code>.</p>
</footer>
</body>
</html>
"""


def main(quiet: bool = False) -> bool:
    """Write docs/changelog.html; returns whether it changed."""
    import retro_style
    html = retro_style.apply(build())
    old = open(OUT, encoding="utf-8").read() if os.path.exists(OUT) else None
    if old == html:
        if not quiet:
            print(f"unchanged {os.path.relpath(OUT, ROOT)}")
        return False
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(html)
    os.replace(tmp, OUT)
    if not quiet:
        print(f"wrote {os.path.relpath(OUT, ROOT)} ({os.path.getsize(OUT) / 1e3:.0f} kB)")
    return True


if __name__ == "__main__":
    main()
