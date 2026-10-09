#!/usr/bin/env python3
r"""Build the topic overviews, docs/overviews/<file> (names in docs_inventory.OVERVIEW): what a collaborator new to the project needs from one topic
of the result pages, with every finding linked to the page that measured it.

    python3 scripts/topic_overview_page.py mechanisms        # one topic
    python3 scripts/topic_overview_page.py --all             # every topic with a text module
    python3 scripts/topic_overview_page.py policies --refresh-figures

The prose of each topic is in scripts/overview_<topic>_text.py, a module with TITLE, LEDE, TERMS, SECTIONS, FIGURES,
FUTURE and PAGES (see overview_mechanisms_text.py). In its HTML, [text](exp:FOLDER/FILE.html#anchor) links a result
page under docs/experiments/, {{FIG name}} places a figure, and \( \) is inline LaTeX, rendered to SVG by
scripts/texsvg.py and cached in docs/overviews/media/texsvg_cache.json.

Figures are taken from the detailed pages: scripts/page_figure_check.render_figure lays the figure out alone with its
page's styles in headless Chrome (caption hidden) at twice the display resolution; the PNG is cached in
docs/overviews/media/ and inlined, quantized, as a data URI, so the overview opens from the file system (owner,
2026-10-09). A cached figure is re-rendered when its source page is newer or with --refresh-figures. The pages are
ordinary git files (docs/overviews/ is outside the LFS pattern for docs/experiments); keep each under 2 MB.
"""
from __future__ import annotations

import argparse
import base64
import glob
import importlib
import io
import os
import re
import sys
import time

import docs_inventory as D
import page_figure_check as F
import retro_style
import texsvg

ROOT = D.ROOT
OUTDIR = os.path.join(ROOT, "docs", "overviews")
MEDIA = os.path.join(OUTDIR, "media")
TEX_CACHE = os.path.join(MEDIA, "texsvg_cache.json")
TEX_RE = re.compile(r"\\\[(.+?)\\\]|\\\((.+?)\\\)", re.S)
LINK_RE = re.compile(r"\[([^\]]+)\]\(exp:([^)#\s]+)(#[^)\s]*)?\)")
FIG_RE = re.compile(r"\{\{FIG ([\w-]+)\}\}")
DISPLAY_W = 828                       # px, the plain style's text column

CSS = """
nav.toc ol{columns:2;column-gap:40px;margin:4px 0 0;padding-left:22px}
@media (max-width:640px){nav.toc ol{columns:1}}
dl.glossary{display:grid;grid-template-columns:minmax(8em,12em) 1fr;gap:6px 18px;margin:8px 0 0}
dl.glossary dt{font-weight:bold}
dl.glossary dd{margin:0}
@media (max-width:640px){dl.glossary{grid-template-columns:1fr}dl.glossary dd{margin-bottom:6px}}
figure img{max-width:100%;height:auto}
.tw{overflow-x:auto}
table.index{width:100%}
table.index td:first-child{white-space:nowrap}
.tex-d{display:block;margin:.6em auto;max-width:100%;height:auto}
"""


def rel(path_from_root: str) -> str:
    return os.path.relpath(os.path.join(ROOT, path_from_root), OUTDIR)


def expand_links(html: str) -> str:
    def sub(m):
        text, page, anchor = m.group(1), m.group(2), m.group(3) or ""
        path = os.path.join("docs", "experiments", page)
        if not os.path.exists(os.path.join(ROOT, path)):
            raise SystemExit(f"link to a missing page: {page}")
        return f'<a href="{rel(path)}{anchor}">{text}</a>'
    return LINK_RE.sub(sub, html)


def figure_png(key: str, name: str, spec: dict, refresh: bool) -> str:
    if spec.get("image"):                     # an image file of its own (a photo), path from the repository root
        return os.path.join(ROOT, spec["image"])
    src = os.path.join(ROOT, "docs", "experiments", spec["page"])
    cache = os.path.join(MEDIA, f"{key}_{name}.png")
    stale = (not os.path.exists(cache)) or os.path.getmtime(src) > os.path.getmtime(cache)
    if refresh or stale:
        os.makedirs(MEDIA, exist_ok=True)
        F.render_figure(src, spec["n"], cache, scale=2.0)
    return cache


def figure_html(key: str, name: str, spec: dict, refresh: bool) -> tuple[str, int]:
    from PIL import Image
    im = Image.open(figure_png(key, name, spec, refresh)).convert("RGB")
    if spec.get("crop"):                                # (left, top, right, bottom) as fractions of the image
        l, t, r, b = spec["crop"]
        im = im.crop((int(l * im.width), int(t * im.height), int(r * im.width), int(b * im.height)))
    disp = min(DISPLAY_W, spec.get("width", DISPLAY_W), im.width if spec.get("image") else im.width // 2)
    w = min(im.width, 2 * disp)
    if im.width > w:
        im = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    if spec.get("photo"):
        im.save(buf, "JPEG", quality=80, optimize=True, progressive=True)
        mime = "image/jpeg"
    else:
        im.quantize(colors=spec.get("colors", 96), method=Image.Quantize.FASTOCTREE).save(buf, "PNG", optimize=True)
        mime = "image/png"
    data = buf.getvalue()
    h = round(im.height * disp / im.width)
    uri = f"data:{mime};base64," + base64.b64encode(data).decode()
    cap = spec["caption"].rstrip()
    if spec.get("page"):
        src_path = os.path.join("docs", "experiments", spec["page"])
        cap += (f' From <a href="{rel(src_path)}">{spec.get("source", "the source page")}</a>, '
                f'Figure&#160;{spec.get("source_n", spec["n"])}.')
    return (f'<figure><img src="{uri}" width="{disp}" height="{h}" alt="{spec.get("alt", "")}">'
            f'<figcaption>Figure&#160;{{N}}. {cap}</figcaption></figure>'), len(data)


def render_tex(t: str) -> tuple[str, int]:
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in TEX_RE.finditer(t)]
    if not items:
        return t, 0
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE, scale=1.1))
    return TEX_RE.sub(lambda m: next(svgs), t), len(items)


def index_table(M) -> str:
    """Index of linked pages: the topic's pages from the inventory (newest first) and the module's extra pages, each with
    date, title linked to the local file, contents and the artifact link."""
    recs = {os.path.relpath(r["file"], "docs/experiments"): r for r in D.records()}
    rows, group = [], None
    out = ["<table class='index'><thead><tr><th>Date</th><th>Page</th><th>Contents</th><th>Artifact</th></tr></thead><tbody>"]
    for g, page, contents in M.PAGES:
        r = recs.get(page)
        if r is None:
            raise SystemExit(f"PAGES names a page the inventory does not know: {page}")
        if g != group:
            out.append(f"<tr class='grp'><td colspan='4'>{g}</td></tr>")
            group = g
        art = f"<a href='{r['artifact']}'>Link</a>" if r["artifact"] else ""
        title = r["title"]
        if r["status"] != "current":
            title += f" ({r['status']})"
        out.append(f"<tr><td class='num'>{r['date']}</td><td><a href='{rel(r['file'])}'>{title}</a></td>"
                   f"<td>{contents}</td><td>{art}</td></tr>")
    out.append("</tbody></table>")
    covered = {p for _, p, _ in M.PAGES}
    missing = [os.path.relpath(r["file"], "docs/experiments") for r in recs.values()
               if r["topic"] == M.KEY and os.path.relpath(r["file"], "docs/experiments") not in covered]
    if missing:
        print(f"  note: {len(missing)} pages of the topic are not in PAGES: {missing}")
    return "".join(out)


def build(key: str, refresh: bool = False) -> str:
    M = importlib.import_module(f"overview_{key}_text")
    titles = {k: t for k, t, _ in D.TOPICS}
    fig_n, size = {}, 0

    def place(m):
        nonlocal size
        name = m.group(1)
        fig_n[name] = len(fig_n) + 1
        h, nbytes = figure_html(key, name, M.FIGURES[name], refresh)
        size += nbytes
        return h.replace("{N}", str(fig_n[name]))

    toc = ["<li><a href='#terms'>Terms</a></li>"]
    body = []
    for sid, head, html in M.SECTIONS:
        toc.append(f"<li><a href='#{sid}'>{head}</a></li>")
        body.append(f"<section id='{sid}'><h2>{head}</h2>\n{html}\n</section>")
    toc += ["<li><a href='#future'>Future work</a></li>", "<li><a href='#index'>Index of linked pages</a></li>"]
    body_html = FIG_RE.sub(place, "\n".join(body))
    # "Figure name" references in the prose become numbers
    body_html = re.sub(r"\{\{FIGREF ([\w-]+)\}\}", lambda m: f"Figure&#160;{fig_n[m.group(1)]}", body_html)
    terms = "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in M.TERMS)
    future = "".join(f"<li>{x}</li>" for x in M.FUTURE)
    home = os.path.relpath(os.path.join(ROOT, "docs", "index.html"), OUTDIR)
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{M.SHORT_TITLE}</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
<header>
<div class="eyebrow"><a href="{home}">SR&#178; Hand documentation</a> &#183; topic overview: {titles[key].lower()} &#183; updated {time.strftime("%Y-%m-%d")}</div>
<h1>{M.TITLE}</h1>
<p class="lede">{M.LEDE}</p>
</header>
<nav class="toc"><ol>{"".join(toc)}</ol></nav>
<section id="terms"><h2>Terms</h2><dl class="glossary">{terms}</dl></section>
{body_html}
<section id="future"><h2>Future work</h2><ol>{future}</ol></section>
<footer>
<h2 id="index">Index of linked pages</h2>
<div class="tw">{index_table(M)}</div>
<p>Built by <code>python3 scripts/topic_overview_page.py {key}</code> from <code>scripts/overview_{key}_text.py</code>;
figures are rendered from the linked pages and cached in <code>docs/overviews/media/</code>.</p>
</footer>
</div>
</body>
</html>
"""
    html = expand_links(html)
    html, n_tex = render_tex(html)
    html = retro_style.apply(html)
    out = os.path.join(OUTDIR, D.OVERVIEW[key])
    open(out, "w", encoding="utf-8").write(html)
    mb = os.path.getsize(out) / 1e6
    print(f"wrote {os.path.relpath(out, ROOT)}: {mb:.2f} MB, {len(fig_n)} figures ({size / 1e3:.0f} kB), {n_tex} formulas")
    if mb > 2.0:
        print("  over the 2 MB budget: crop or downsample a figure")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("topics", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--refresh-figures", action="store_true")
    a = ap.parse_args()
    keys = list(a.topics)
    if a.all:
        keys = sorted(os.path.basename(p)[len("overview_"):-len("_text.py")]
                      for p in glob.glob(os.path.join(ROOT, "scripts", "overview_*_text.py")))
    if not keys:
        ap.error("name a topic or pass --all")
    for k in keys:
        build(k, a.refresh_figures)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
