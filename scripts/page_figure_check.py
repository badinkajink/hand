#!/usr/bin/env python3
"""Figure check for the result pages: one screenshot per figure with the page's own styles, and the chart text as
displayed, in headless Chrome.

    python3 scripts/page_figure_check.py PAGE.html [PAGE2.html ...]        # report + screenshots
    python3 scripts/page_figure_check.py --no-shots PAGE.html              # report only
    python3 scripts/page_figure_check.py --all                             # every page in docs_inventory

For every <figure> on the page (and every chart <svg>, .chart or .diagram outside one) the script lays the figure
out alone at the page's width (900 px window, the plain style's 860 px column) and measures, for each <text> of each
inline SVG chart, the font size as displayed: the computed size times the SVG's scale on screen. It flags text under
13 px (owner 2026-10-09: chart text at least 13 px as displayed), pairs of labels whose boxes overlap, and labels
that run outside the SVG's box, where the browser clips them. Raster images report their downscale (displayed over
natural width); their text has to be read from the screenshot.

Screenshots and report.json go to logs/<date>-figure_check/<page>/ (gitignored), fig01.png, fig02.png, ...
LaTeX formula SVGs (class "tex") and videos are not measured; a video's poster is in the screenshot.
"""
from __future__ import annotations

import argparse
import html as htmlmod
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROME = shutil.which("google-chrome") or shutil.which("google-chrome-stable") or shutil.which("chromium")
WIDTH = 900
MIN_PX = 13.0
CHUNK_PX = 7000                     # screenshot height per Chrome launch

TAG_RE = re.compile(r"<(/?)(figure|svg|div)\b([^>]*)>", re.I)
STYLE_RE = re.compile(r"<style\b[^>]*>.*?</style>", re.S | re.I)

MEASURE_JS = r"""
window.addEventListener('load', function(){ setTimeout(function(){
  function rect(e){var r=e.getBoundingClientRect();return {x:r.left,y:r.top+window.scrollY,w:r.width,h:r.height};}
  function inter(a,b){var x=Math.max(0,Math.min(a.x+a.w,b.x+b.w)-Math.max(a.x,b.x));
                      var y=Math.max(0,Math.min(a.y+a.h,b.y+b.h)-Math.max(a.y,b.y));return x*y;}
  var out=[];
  document.querySelectorAll('.figcheck').forEach(function(box){
    var R={i:+box.dataset.i, top:rect(box).y, height:box.offsetHeight, svgs:[], imgs:[], videos:0};
    box.querySelectorAll('svg').forEach(function(svg){
      if(svg.classList.contains('tex')||svg.closest('svg.tex')) return;
      if(svg.parentElement && svg.parentElement.closest('svg')) return;   // nested svg: measured with its root
      var texts=[].slice.call(svg.querySelectorAll('text')).filter(function(t){return (t.textContent||'').trim().length>0;});
      var sr=rect(svg), S={w:Math.round(sr.w), h:Math.round(sr.h), n_text:texts.length, min_px:null, small:[], overlaps:[], clipped:[]};
      if(svg.viewBox && svg.viewBox.baseVal && svg.viewBox.baseVal.width) S.vb_w=svg.viewBox.baseVal.width;
      var boxes=[];
      texts.forEach(function(t){
        var fs=parseFloat(getComputedStyle(t).fontSize)||0, m=t.getScreenCTM(), k=m?Math.sqrt(m.a*m.a+m.b*m.b):1;
        var px=fs*k, r=rect(t), s=t.textContent.trim().slice(0,40);
        if(S.min_px===null||px<S.min_px) S.min_px=px;
        if(px<13-0.05) S.small.push([s, Math.round(px*10)/10]);
        if(r.w>0&&r.h>0){
          if(r.x<sr.x-1||r.x+r.w>sr.x+sr.w+1||r.y<sr.y-1||r.y+r.h>sr.y+sr.h+1) S.clipped.push(s);
          boxes.push([r,s]);
        }
      });
      for(var a=0;a<boxes.length;a++) for(var b=a+1;b<boxes.length;b++){
        var A=boxes[a][0],B=boxes[b][0],o=inter(A,B),m=Math.min(A.w*A.h,B.w*B.h);
        if(m>0&&o>0.15*m) S.overlaps.push([boxes[a][1],boxes[b][1]]);
      }
      if(S.min_px!==null) S.min_px=Math.round(S.min_px*10)/10;
      S.small=S.small.slice(0,8); S.overlaps=S.overlaps.slice(0,8); S.clipped=S.clipped.slice(0,8);
      R.svgs.push(S);
    });
    box.querySelectorAll('img').forEach(function(im){
      R.imgs.push({nat_w:im.naturalWidth, disp_w:Math.round(im.getBoundingClientRect().width),
                   ratio:im.naturalWidth?Math.round(im.getBoundingClientRect().width/im.naturalWidth*100)/100:null});
    });
    R.videos=box.querySelectorAll('video').length;
    out.push(R);
  });
  var pre=document.createElement('pre'); pre.id='figcheck-report'; pre.textContent=JSON.stringify(out);
  document.body.appendChild(pre);
}, 300); });
"""

HARNESS_CSS = (".figcheck{display:flow-root;padding:6px 0;margin:0;border:0}"
               "#figcheck-report{display:none}")


def figures(page: str):
    """Top-level figure blocks: <figure> elements, and chart SVGs, .chart and .diagram divs outside any figure."""
    out, stack, start = [], [], None
    for m in TAG_RE.finditer(page):
        close, tag, attrs = m.group(1) == "/", m.group(2).lower(), m.group(3)
        if not close and attrs.rstrip().endswith("/"):
            continue
        if not close:
            kind = None
            if tag == "figure":
                kind = "figure"
            elif tag == "svg" and 'class="tex' not in attrs and "class='tex" not in attrs:
                kind = "svg"
            elif tag == "div" and re.search(r"""class=["'][^"']*\b(chart|diagram)\b""", attrs):
                kind = "chart"
            inside = any(k for _, k in stack)
            stack.append((tag, kind if not inside else None))
            if kind and not inside:
                start = m.start()
        else:
            # pop to the matching open tag of the same name
            for j in range(len(stack) - 1, -1, -1):
                if stack[j][0] == tag:
                    _, kind = stack[j]
                    del stack[j:]
                    if kind and start is not None and not any(k for _, k in stack):
                        out.append(page[start:m.end()])
                        start = None
                    break
    return out


def harness(page: str, figs, with_script: bool, base: str = "") -> str:
    styles = "\n".join(STYLE_RE.findall(page))
    body = "".join(f'<div class="figcheck" data-i="{i}">{f}</div>' for i, f in figs)
    script = f"<script>{MEASURE_JS}</script>" if with_script else ""
    base = f'<base href="file://{base.rstrip("/")}/">' if base else ""
    return (f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
            f'{base}{styles}<style>{HARNESS_CSS}</style></head><body><div class="wrap"><section><div class="col">'
            f'{body}</div></section></div>{script}</body></html>')


def chrome(args, timeout=180):
    cmd = [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--mute-audio", "--no-first-run",
           "--no-default-browser-check", "--disable-extensions", f"--user-data-dir={tempfile.mkdtemp(prefix='figchk-')}"]
    r = subprocess.run(cmd + args, capture_output=True, text=True, timeout=timeout)
    return r.stdout


def measure(page: str, figs, tmpdir: str, base: str):
    path = os.path.join(tmpdir, "measure.html")
    open(path, "w", encoding="utf-8").write(harness(page, figs, True, base))
    dom = chrome([f"--window-size={WIDTH},2000", "--virtual-time-budget=15000", "--dump-dom", "file://" + path])
    m = re.search(r'<pre id="figcheck-report">(.*?)</pre>', dom, re.S)
    if not m:
        raise RuntimeError("no report from Chrome")
    return json.loads(htmlmod.unescape(m.group(1)))


def shots(page: str, figs, rep, tmpdir: str, outdir: str, base: str):
    from PIL import Image
    h = {r["i"]: r["height"] for r in rep}
    chunk, acc = [], 0
    chunks = []
    for i, f in figs:
        if chunk and acc + h.get(i, 0) > CHUNK_PX:
            chunks.append(chunk)
            chunk, acc = [], 0
        chunk.append((i, f))
        acc += h.get(i, 0)
    if chunk:
        chunks.append(chunk)
    for c, chunk in enumerate(chunks):
        path = os.path.join(tmpdir, f"chunk{c}.html")
        # the measure script runs here too, after the load event (before it the headless viewport has no width)
        open(path, "w", encoding="utf-8").write(harness(page, chunk, True, base))
        dom = chrome([f"--window-size={WIDTH},2000", "--virtual-time-budget=15000", "--dump-dom", "file://" + path])
        m = re.search(r'<pre id="figcheck-report">(.*?)</pre>', dom, re.S)
        pos = {r["i"]: (r["top"], r["height"]) for r in json.loads(htmlmod.unescape(m.group(1)))} if m else {}
        total = int(max((t + hh for t, hh in pos.values()), default=1000)) + 20
        png = os.path.join(tmpdir, f"chunk{c}.png")
        chrome([f"--window-size={WIDTH},{total}", "--virtual-time-budget=15000", f"--screenshot={png}", "file://" + path])
        if not os.path.exists(png):
            continue
        im = Image.open(png)
        for i, _ in chunk:
            if i not in pos:
                continue
            t, hh = pos[i]
            box = (0, max(0, int(t)), im.width, min(im.height, int(t + hh)))
            if box[3] > box[1]:
                im.crop(box).save(os.path.join(outdir, f"fig{i + 1:02d}.png"))


def flags(r):
    out = []
    for k, s in enumerate(r["svgs"]):
        tag = f"svg{k + 1}" if len(r["svgs"]) > 1 else "svg"
        if s["small"]:
            out.append(f"{tag}: {len(s['small'])}+ labels under {MIN_PX:g} px (min {s['min_px']} px), e.g. "
                       + "; ".join(f"'{a}' {b}" for a, b in s["small"][:3]))
        if s["overlaps"]:
            out.append(f"{tag}: {len(s['overlaps'])} overlapping label pairs, e.g. "
                       + "; ".join(f"'{a}'/'{b}'" for a, b in s["overlaps"][:2]))
        if s["clipped"]:
            out.append(f"{tag}: {len(s['clipped'])} labels outside the chart box, e.g. "
                       + "; ".join(f"'{a}'" for a in s["clipped"][:3]))
    for im in r["imgs"]:
        if im["ratio"] and im["ratio"] < 0.75:
            out.append(f"image shown at {im['ratio']:.2f} of its {im['nat_w']} px width: read its text in the screenshot")
    return out


def check(path: str, out_root: str, do_shots: bool):
    page = open(path, encoding="utf-8", errors="replace").read()
    figs = list(enumerate(figures(page)))
    stem = os.path.splitext(os.path.basename(path))[0]
    outdir = os.path.join(out_root, stem)
    os.makedirs(outdir, exist_ok=True)
    if not figs:
        print(f"{os.path.relpath(path, ROOT)}: no figures")
        json.dump([], open(os.path.join(outdir, "report.json"), "w"))
        return []
    tmpdir = tempfile.mkdtemp(prefix="figchk-")
    try:
        base = os.path.dirname(os.path.abspath(path))   # relative media resolve against the page's folder
        rep = measure(page, figs, tmpdir, base)
        if do_shots:
            shots(page, figs, rep, tmpdir, outdir, base)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    json.dump(rep, open(os.path.join(outdir, "report.json"), "w"), indent=1)
    flagged = [(r["i"], flags(r)) for r in rep]
    flagged = [(i, f) for i, f in flagged if f]
    n_svg = sum(len(r["svgs"]) for r in rep)
    print(f"{os.path.relpath(path, ROOT)}: {len(figs)} figures, {n_svg} charts, {len(flagged)} flagged -> {outdir}")
    for i, fl in flagged:
        for f in fl:
            print(f"  fig{i + 1:02d} {f}")
    return flagged


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pages", nargs="*")
    ap.add_argument("--all", action="store_true", help="every page of scripts/docs_inventory.py")
    ap.add_argument("--no-shots", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", time.strftime("%Y%m%d") + "-figure_check"))
    a = ap.parse_args()
    if not CHROME:
        raise SystemExit("needs google-chrome or chromium on PATH")
    pages = list(a.pages)
    if a.all:
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import docs_inventory
        pages += [os.path.join(ROOT, r["file"]) for r in docs_inventory.records()]
    for p in pages:
        check(p, a.out, not a.no_shots)


if __name__ == "__main__":
    main()
