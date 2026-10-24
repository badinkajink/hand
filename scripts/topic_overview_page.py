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

Films (owner, 2026-10-12: "missing videos"): a FIGURES entry with film=[[(path, label[, crop]), ...], ...] sets the
clips of each row side by side and the rows one above the other, each clip under a white label bar, at 25 fps; shorter
clips hold their last frame; crop "w:h:x:y" (source pixels) cuts one panel out of a multi-panel film. The composite is encoded with x264 (crf 28 unless the entry sets crf), cached as
docs/overviews/media/<topic>_<name>.mp4 with a poster frame, and inlined like the images.

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
import json
import os
import re
import subprocess
import sys
import tempfile
import time

import docs_inventory as D
import interactive_figure as IF
import overview_figures as OF
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
FILM_FONT = next((p for p in ("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
                              "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf") if os.path.exists(p)), None)
FILM_LABEL_H, FILM_GAP = 30, 4        # px: label bar above each clip, white gap between clips and rows

CSS = """
nav.toc ol{columns:2;column-gap:40px;margin:4px 0 0;padding-left:22px}
@media (max-width:640px){nav.toc ol{columns:1}}
dl.glossary{display:grid;grid-template-columns:minmax(8em,12em) 1fr;gap:6px 18px;margin:8px 0 0}
dl.glossary dt{font-weight:bold}
dl.glossary dd{margin:0}
@media (max-width:640px){dl.glossary{grid-template-columns:1fr}dl.glossary dd{margin-bottom:6px}}
figure img{max-width:100%;height:auto}
figure video{display:block;margin:0 auto;max-width:100%;height:auto;background:#fff}
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


def _probe(path: str) -> tuple[int, int, float]:
    j = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "stream=width,height:format=duration", "-of", "json", path],
                                  capture_output=True, text=True, check=True).stdout)
    return int(j["streams"][0]["width"]), int(j["streams"][0]["height"]), float(j["format"]["duration"])


def film_files(key: str, name: str, spec: dict, refresh: bool) -> tuple[str, str]:
    """The composite film of a film= entry and its poster, cached in docs/overviews/media/ (see the module docstring)."""
    rows = spec["film"]
    srcs = [os.path.join(ROOT, c[0]) for row in rows for c in row]
    base = os.path.join(MEDIA, f"{key}_{name}")
    mp4, jpg, sig_path = base + ".mp4", base + ".jpg", base + ".json"
    sig = json.dumps({k: spec.get(k) for k in ("film", "width", "crf", "poster_t")}, sort_keys=True)
    fresh = (all(os.path.exists(x) for x in (mp4, jpg, sig_path)) and open(sig_path).read() == sig
             and all(os.path.getmtime(x) <= os.path.getmtime(mp4) for x in srcs))
    if fresh and not refresh:
        return mp4, jpg
    os.makedirs(MEDIA, exist_ok=True)
    W = spec.get("width", DISPLAY_W) // 2 * 2
    labelled = any(c[1] for row in rows for c in row)
    probes = {x: _probe(x) for x in srcs}
    D = max(d for _, _, d in probes.values())
    tmp = tempfile.mkdtemp(prefix="film_")
    chains, row_tags, k = [], [], 0
    for ri, row in enumerate(rows):
        n = len(row)
        tw = (W - FILM_GAP * (n - 1)) // n // 2 * 2
        tags = []
        for ci, (p, lab, *crop) in enumerate(row):
            w0, h0, d0 = probes[os.path.join(ROOT, p)]
            if crop:
                w0, h0 = (int(v) for v in crop[0].split(":")[:2])
            th = round(h0 * tw / w0) // 2 * 2
            f = f"[{k}:v]" + (f"crop={crop[0]}," if crop else "") + f"fps=25,scale={tw}:{th}:flags=lanczos,setsar=1"
            if D - d0 > 0.04:
                f += f",tpad=stop_mode=clone:stop_duration={D - d0:.3f}"
            if labelled:
                f += f",pad={tw}:{th + FILM_LABEL_H}:0:{FILM_LABEL_H}:white"
                if lab:
                    tf = os.path.join(tmp, f"label{k}.txt")
                    open(tf, "w", encoding="utf-8").write(lab)
                    f += (f",drawtext=fontfile={FILM_FONT}:textfile={tf}:x=4:y=({FILM_LABEL_H}-th)/2:fontsize=19"
                          f":fontcolor=black")
            if ci < n - 1:
                f += f",pad=iw+{FILM_GAP}:ih:0:0:white"
            chains.append(f + f"[t{k}]")
            tags.append(f"[t{k}]")
            k += 1
        row_out = f"[r{ri}]"
        if n > 1:
            chains.append("".join(tags) + f"hstack=inputs={n}[h{ri}]")
            src_tag = f"[h{ri}]"
        else:
            src_tag = tags[0]
        pad_b = FILM_GAP if ri < len(rows) - 1 else 0
        chains.append(f"{src_tag}pad={W}:ih+{pad_b}:(ow-iw)/2:0:white{row_out}")
        row_tags.append(row_out)
    if len(rows) > 1:
        chains.append("".join(row_tags) + f"vstack=inputs={len(rows)}[out]")
    else:
        chains.append(f"{row_tags[0]}null[out]")
    cmd = ["ffmpeg", "-y", "-v", "error"]
    for x in srcs:
        cmd += ["-i", x]
    cmd += ["-filter_complex", ";".join(chains), "-map", "[out]", "-an", "-c:v", "libx264", "-preset", "slow",
            "-crf", str(spec.get("crf", 28)), "-pix_fmt", "yuv420p", "-movflags", "+faststart", mp4]
    subprocess.run(cmd, check=True)
    t = spec.get("poster_t", 0.75) * D
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", mp4, "-frames:v", "1", "-q:v", "3", jpg],
                   check=True)
    open(sig_path, "w").write(sig)
    return mp4, jpg


def film_html(key: str, name: str, spec: dict, refresh: bool) -> tuple[str, int]:
    mp4, jpg = film_files(key, name, spec, refresh)
    w, h, _ = _probe(mp4)
    disp = min(DISPLAY_W, w)
    data, poster = open(mp4, "rb").read(), open(jpg, "rb").read()
    cap = spec["caption"].rstrip()
    pages = spec.get("pages") or ([(spec["page"], spec.get("source", "the source page"))] if spec.get("page") else [])
    if pages:
        links = [f'<a href="{rel(os.path.join("docs", "experiments", pg))}">{lab}</a>' for pg, lab in pages]
        cap += f' {"Film" if len(links) == 1 and len(spec["film"]) == 1 and len(spec["film"][0]) == 1 else "Films"} from ' \
               + (links[0] if len(links) == 1 else ", ".join(links[:-1]) + " and " + links[-1]) + "."
    return (f'<figure><video src="data:video/mp4;base64,{base64.b64encode(data).decode()}" '
            f'poster="data:image/jpeg;base64,{base64.b64encode(poster).decode()}" width="{disp}" '
            f'height="{round(h * disp / w)}" controls muted loop playsinline preload="metadata"></video>'
            f'<figcaption>Figure&#160;{{N}}. {cap}</figcaption></figure>'), len(data) + len(poster)


def figure_html(key: str, name: str, spec: dict, refresh: bool) -> tuple[str, int]:
    if spec.get("film"):                      # a film or a side-by-side of films from the result pages (2026-10-12)
        return film_html(key, name, spec, refresh)
    if spec.get("fn"):                        # an interactive data figure drawn from the source page's rows (2026-10-11)
        fig = getattr(OF, spec["fn"])(f"{key}-{name}")
        cap = spec["caption"].rstrip()
        if spec.get("page"):
            src_path = os.path.join("docs", "experiments", spec["page"])
            cap += (f' From <a href="{rel(src_path)}">{spec.get("source", "the source page")}</a>, '
                    f'Figure&#160;{spec.get("source_n", spec["n"])}.')
        fig.caption = f"Figure&#160;{{N}}. {cap}"
        return fig.html(), len(fig.data_json())
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


def index_table(M, recs=None, quiet=False) -> str:
    """Index of linked pages: the pages the module's PAGES names, in its groups and order, with its contents line; then
    every other page of the topic in the inventory, newest first, with its summary (docs_inventory.records(): the page's
    docs-summary tag or the first sentences of its lede), so a new page appears without an edit here. scripts/docs_index.py
    rewrites this table in a built overview between its <!--docs-index--> markers."""
    recs = {os.path.relpath(r["file"], "docs/experiments"): r for r in (recs or D.records())}
    group = None
    out = ["<table class='index'><thead><tr><th>Date</th><th>Page</th><th>Contents</th><th>Artifact</th></tr></thead><tbody>"]

    def row(r, contents):
        art = f"<a href='{r['artifact']}'>Link</a>" if r["artifact"] else ""
        title = htmlmod_escape(r["title"])
        if r["status"] != "current":
            title += f" ({r['status']})"
        return (f"<tr><td class='num'>{r['date']}</td><td><a href='{rel(r['file'])}'>{title}</a></td>"
                f"<td>{contents}</td><td>{art}</td></tr>")
    for g, page, contents in M.PAGES:
        r = recs.get(page)
        if r is None:
            raise SystemExit(f"PAGES names a page the inventory does not know: {page}")
        if g != group:
            out.append(f"<tr class='grp'><td colspan='4'>{g}</td></tr>")
            group = g
        out.append(row(r, contents))
    covered = {p for _, p, _ in M.PAGES}
    rest = sorted((r for p, r in recs.items() if r["topic"] == M.KEY and p not in covered),
                  key=lambda r: (r["date"], r["file"]), reverse=True)
    if rest:
        out.append(f"<tr class='grp'><td colspan='4'>{'Other pages of the topic' if M.PAGES else 'Pages of the topic'}</td></tr>")
        out += [row(r, htmlmod_escape(r["summary"])) for r in rest]
        if not quiet:
            print(f"  {len(rest)} pages of the topic not in PAGES, listed with their summaries")
    out.append("</tbody></table>")
    return "".join(out)


def htmlmod_escape(s: str) -> str:
    import html
    return html.escape(s, quote=False)


def build(key: str, refresh: bool = False) -> str:
    M = importlib.import_module(f"overview_{key}_text")
    titles = {k: t for k, t, _ in D.TOPICS}
    # an overview outside the six result topics (the code and methods page) names its own file and label
    titles.setdefault(key, getattr(M, "TOPIC_TITLE", M.SHORT_TITLE))
    out_name = getattr(M, "FILE", None) or D.OVERVIEW[key]
    fig_n, size = {}, 0

    def place(m):
        nonlocal size
        name = m.group(1)
        fig_n[name] = len(fig_n) + 1
        h, nbytes = figure_html(key, name, M.FIGURES[name], refresh)
        size += nbytes
        return h.replace("{N}", str(fig_n[name]))

    intro = getattr(M, "INTRO", "")
    intro_title = getattr(M, "INTRO_TITLE", "Physical example")
    toc = ([f"<li><a href='#example'>{intro_title}</a></li>"] if intro else [])
    toc.append("<li><a href='#terms'>Terms</a></li>")
    body = []
    for sid, head, html in M.SECTIONS:
        toc.append(f"<li><a href='#{sid}'>{head}</a></li>")
        body.append(f"<section id='{sid}'><h2>{head}</h2>\n{html}\n</section>")
    toc += ["<li><a href='#future'>Future work</a></li>", "<li><a href='#index'>Index of linked pages</a></li>"]
    body_html = FIG_RE.sub(place, "\n".join(body))
    # "Figure name" references in the prose become numbers
    body_html = re.sub(r"\{\{FIGREF ([\w-]+)\}\}", lambda m: f"Figure&#160;{fig_n[m.group(1)]}", body_html)
    body_html = re.sub(r"\{\{FIGN ([\w-]+)\}\}", lambda m: str(fig_n[m.group(1)]), body_html)   # the number alone
    terms = "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in M.TERMS)
    future = "".join(f"<li>{x}</li>" for x in M.FUTURE)
    home = os.path.relpath(os.path.join(ROOT, "docs", "index.html"), OUTDIR)
    pages = os.path.relpath(os.path.join(ROOT, "docs", "pages.html"), OUTDIR)
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
<div class="eyebrow"><a href="{home}">SR&#178; Hand documentation</a> &#183; <a href="{pages}">index of result pages</a> &#183; topic overview: {titles[key].lower()} &#183; updated {time.strftime("%Y-%m-%d")}</div>
<h1>{M.TITLE}</h1>
<p class="lede">{M.LEDE}</p>
</header>
<nav class="toc"><ol>{"".join(toc)}</ol></nav>
{f"<section id='example'><h2>{intro_title}</h2>{intro}</section>" if intro else ""}
<section id="terms"><h2>Terms</h2><dl class="glossary">{terms}</dl></section>
{body_html}
<section id="future"><h2>Future work</h2><ol>{future}</ol></section>
<footer>
<h2 id="index">Index of linked pages</h2>
<div class="tw"><!--docs-index-->{index_table(M)}<!--/docs-index--></div>
<p>Built by <code>python3 scripts/topic_overview_page.py {key}</code> from <code>scripts/overview_{key}_text.py</code>;
figures are rendered from the linked pages and cached in <code>docs/overviews/media/</code>.</p>
</footer>
</div>
</body>
</html>
"""
    html = expand_links(html)
    html, n_tex = render_tex(html)
    html = IF.add_assets(retro_style.apply(html))
    out = os.path.join(OUTDIR, out_name)
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
