#!/usr/bin/env python3
r"""Pre-render LaTeX math to inline SVG for static HTML result pages: no JavaScript, CDN or font files.

    import texsvg
    svgs = texsvg.render([(r"\bar r = c\,N^{1/4}", False),          # inline, sits on the baseline
                          (r"\frac{W}{2\mu N}", True)],               # display, class "tex tex-d"
                         cache_path="docs/experiments/<dir>/texsvg_cache.json")
    page = template.replace("{{EQ_ARM}}", svgs[0])                    # and add texsvg.CSS to <style>

One `latex` run (DVI mode) typesets the whole batch, one `preview` page per item; inline items are set
as $...$ and display items as $\displaystyle ...$ (so use aligned/gathered/split, not align).
`dvisvgm --no-fonts` converts every page to glyph outlines. Each returned <svg>
  - is sized in em, 10 pt TeX = `scale` em of the surrounding text; SCALE = 1.2 sets the Computer
    Modern x-height (4.31 pt) to 0.517 em, which matches Arial/Helvetica/Roboto/Segoe UI body text;
  - sits on the text baseline (vertical-align = -depth) with a viewBox that is the union of the TeX
    box and the glyph ink, so nothing is clipped; inline items keep TeX's side bearings via margins;
  - draws in currentColor, so it follows the page's light/dark theme (\textcolor colours are kept);
  - prefixes its ids with a hash of its source, so any number of formulas can share one document.
With `cache_path` each formula is rendered once, and a page rebuilds on a machine without TeX as long
as every formula is in the cache. A LaTeX error raises RuntimeError naming the item that broke.

Box metrics come from TeX itself: each item is measured in an \sbox and \typeout'd to the log, and a
`dvisvgm:raw` special at the box's reference point gives the baseline in SVG coordinates.

Check:  python3 scripts/texsvg.py --demo /tmp/texsvg_demo.html     (light + dark + 12-formula id test)
"""
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCALE = 1.2                                   # em per 10 pt; see the docstring
PREAMBLE = "\\usepackage{amsmath,amssymb,bm,xcolor}\n"
CSS = ".tex-d{display:block;margin:.6em auto;max-width:100%;height:auto}"
_VERSION = "texsvg 1"                         # part of the cache key; bump when the output changes
_BP = 72 / 72.27                              # TeX pt -> bp, the SVG user unit dvisvgm writes
_ITEM = r"""\typeout{texsvg-begin:@I}%
\sbox\texsvgbox{\special{dvisvgm:raw <g data-texsvg='{?x} {?y}'/>}$@S%
@TEX%
$}%
\typeout{texsvg-dims:@I:\the\ht\texsvgbox:\the\dp\texsvgbox:\the\wd\texsvgbox}%
\begin{@ENV}\usebox\texsvgbox\end{@ENV}

"""


def render(items, cache_path=None, scale=SCALE, preamble_extra=""):
    """items: list of (tex, display) with tex the math body WITHOUT delimiters and display a bool.
    Returns one inline <svg> string per item, in order."""
    items, scale = [(str(t), bool(d)) for t, d in items], float(scale)
    pre = PREAMBLE + preamble_extra
    keys = [hashlib.sha1(f"{_VERSION}\n{pre}\n{t}\n{d}\n{scale!r}".encode()).hexdigest() for t, d in items]
    path = Path(cache_path) if cache_path else None
    cache = json.loads(path.read_text()) if path and path.exists() else {}
    todo = {k: it for k, it in zip(keys, items) if k not in cache}
    if todo:
        cache.update(zip(todo, _render_batch(list(todo.values()), list(todo), pre, scale)))
        if path:
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(cache, indent=0, sort_keys=True))
            os.replace(tmp, path)
    out, seen = [], {}
    for k in keys:                            # a repeated formula gets its own ids too
        n = seen[k] = seen.get(k, -1) + 1
        out.append(cache[k] if n == 0 else cache[k].replace(_prefix(k), _prefix(k, n)))
    return out


def _prefix(key, n=0):
    return f"t{key[:10]}{'r%d' % n if n else ''}-"


def _render_batch(items, keys, pre, scale):
    missing = [tool for tool in ("latex", "dvisvgm", "kpsewhich") if not shutil.which(tool)]
    if missing:
        raise RuntimeError(f"{', '.join(missing)} not found and {len(items)} formula(s) are not cached, "
                           f"e.g. {items[0][0]!r}")
    if os.environ.get("TEXSVG_NO_PREVIEW") != "1" and subprocess.run(
            ["kpsewhich", "preview.sty"], capture_output=True, text=True).stdout.strip():
        env = "preview"
        head = ("\\documentclass[10pt]{article}\n" + pre +
                "\\usepackage[active,tightpage]{preview}\\setlength\\PreviewBorder{0pt}\n")
    else:                                     # no preview.sty: standalone, one page per environment
        env = "texsvgpage"
        head = f"\\documentclass[10pt,crop,preview=false,border=0pt,multi={env}]{{standalone}}\n" + pre
    body = "".join(_ITEM.replace("@I", str(i)).replace("@S", "\\displaystyle" if d else "")
                   .replace("@ENV", env).replace("@TEX", t) for i, (t, d) in enumerate(items))
    with tempfile.TemporaryDirectory(prefix="texsvg-") as tmp:
        tmp = Path(tmp)
        (tmp / "b.tex").write_text(f"{head}\\newsavebox\\texsvgbox\n\\begin{{document}}\n{body}\\end{{document}}\n")
        r = subprocess.run(["latex", "-interaction=nonstopmode", "-halt-on-error", "b.tex"], cwd=tmp,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600,
                           env=dict(os.environ, max_print_line="10000"))
        log = (tmp / "b.log").read_text(errors="replace") if (tmp / "b.log").exists() else ""
        if r.returncode or not (tmp / "b.dvi").exists():
            raise RuntimeError(_latex_error(log, items))
        dims = {int(i): tuple(map(float, d)) for i, *d in
                re.findall(r"^texsvg-dims:(\d+):(-?[\d.]+)pt:(-?[\d.]+)pt:(-?[\d.]+)pt$", log, re.M)}
        r = subprocess.run(["dvisvgm", "--no-fonts", "--exact-bbox", "--precision=3", "--relative",
                            "--page=1-", "--output=p%p.svg", "b.dvi"], cwd=tmp, capture_output=True, timeout=600)
        err = r.stderr.decode(errors="replace")
        pages = sorted(tmp.glob("p*.svg"), key=lambda p: int(p.stem[1:]))
        if r.returncode or len(pages) != len(items) or len(dims) != len(items):
            raise RuntimeError(f"{len(items)} items, but latex logged {len(dims)} boxes and dvisvgm wrote "
                               f"{len(pages)} pages:\n{err[-3000:]}")
        for line in err.splitlines():
            if "WARNING" in line or "ERROR" in line:
                print(f"texsvg: dvisvgm {line.strip()}", file=sys.stderr)
        return [_finish(p.read_text(), dims[i], items[i], scale, _prefix(keys[i])) for i, p in enumerate(pages)]


def _latex_error(log, items):
    begun = re.findall(r"^texsvg-begin:(\d+)$", log, re.M)
    if begun:
        tex, d = items[int(begun[-1])]
        where = f"item {begun[-1]} ({'display' if d else 'inline'}) {tex.strip()!r}"
    else:
        where = "the preamble (check preamble_extra)"
    lines = [s.rstrip() for s in log.splitlines()]
    start = next((n for n, s in enumerate(lines) if s.startswith("!")), max(0, len(lines) - 8))
    excerpt = []                                          # the "!" message down to TeX's "l.<n>" context
    for s in lines[start:start + 40]:
        if s.startswith("Here is how much"):
            break
        if s.strip() and not s.startswith(("See the ", "Type  H ")):
            excerpt.append(s)
        if re.match(r"l\.\d+ ", s) or len(excerpt) == 12:
            break
    return f"LaTeX failed on {where}:\n" + "\n".join(excerpt)


def _n(v):
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def _finish(svg, dims, item, scale, prefix):
    """dvisvgm page -> inline <svg>: em size, baseline, currentColor, prefixed ids, no XML noise."""
    tex, display = item
    ht, dp, wd = (v * _BP for v in dims)
    m = re.search(r"""<g data-texsvg=['"](\S+) (\S+)['"]\s*/>""", svg)
    if not m:
        raise RuntimeError(f"dvisvgm dropped the position marker of {tex!r}; it needs raw-special support")
    x0, y0 = float(m[1]), float(m[2])                     # TeX reference point = left end of baseline
    svg = svg[:m.start()] + svg[m.end():]
    vb = re.search(r"""viewBox=['"]([^'"]*)['"]""", svg)
    ix, iy, iw, ih = map(float, vb[1].split()) if vb else (0, 0, 0, 0)
    if iw <= 0 and ih <= 0:                               # no ink (empty formula)
        ix, iy, iw, ih = x0, y0 - ht, wd, ht + dp
    left, right = min(x0, ix), max(x0 + wd, ix + iw)      # union of TeX box and ink box, in bp
    top, bottom = min(y0 - ht, iy), max(y0 + dp, iy + ih)
    k = scale / 10 / _BP                                  # em per bp
    style = f"vertical-align:{_n(-(bottom - y0) * k)}em"
    if not display:                                       # ink overhang must not widen the TeX box
        for side, over in (("left", x0 - left), ("right", right - x0 - wd)):
            if over * k >= 5e-4:
                style += f";margin-{side}:{_n(-over * k)}em"
    body = re.sub(r"<!--.*?-->", "", svg[svg.index("<svg"):], flags=re.S)
    body = body[body.index(">") + 1:]                      # drop dvisvgm's root tag
    body = re.sub(r"='([^']*)'", r'="\1"', body.replace("xlink:href=", "href="))
    body = re.sub(r'\s+id="page\d+"', "", body)
    body = re.sub(r'\b(fill|stroke)="(#000|#000000|black)"', r'\1="currentColor"', body, flags=re.I)
    body = re.sub(r'(\bid="|href="#|url\(#)', lambda g: g[1] + prefix, body)
    body = re.sub(r">\s+<", "><", body).strip()
    label = html.escape(" ".join(tex.split()), quote=True)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" class="{"tex tex-d" if display else "tex"}" role="img" '
            f'aria-label="{label}" focusable="false" width="{_n((right - left) * k)}em" '
            f'height="{_n((bottom - top) * k)}em" viewBox="{_n(left)} {_n(top)} {_n(right - left)} '
            f'{_n(bottom - top)}" overflow="visible" style="{style}" fill="currentColor">{body}')


# ---------------------------------------------------------------- demo page (python3 texsvg.py --demo out.html)
_DEMO_INLINE = [r"\bar r = c\,N^{1/4}", r"\mu_t = \mu\,\bar r(N)", r"q = q_\text{touch} + J^\top(-F\,n)/k_p",
                r"x^2_{i}", r"\sqrt{2R\delta_0}", r"\tfrac{W}{2\mu N}", r"\bm{F}^\top\mathbf{n} \ge 0",
                r"\left(\frac{a}{b}\right)", r"\textcolor{red!70!black}{\theta_\text{max}}"]
_DEMO_DISPLAY = [
    r"""\begin{aligned}
\tau_\text{brake} &= \underbrace{\mu\,N\,\bar r(N)}_{\text{spin friction}} + \underbrace{\bm{F}^\top \mathbf{n}\, d}_{\text{pad moment}} \\
\bar r(N) &= 1.00\,\text{mm}\,\Bigl(\frac{N}{1\,\text{N}}\Bigr)^{1/4}
\end{aligned}""",
    r"\cos\theta_\text{max} = \frac{\mu_t}{\mu\, d} = \frac{\displaystyle\sum_{i=1}^{n} w_i\,\bar r_i}{\displaystyle \mu \int_0^{L} \rho(s)\, s \,\mathrm{d}s}",
    r"F_\text{pinch} = \frac{W}{2\mu}\,\sqrt{1-\left(\frac{W}{2\mu N}\right)^2},\qquad \operatorname{arg\,min}_{q}\ \lVert J q - v\rVert^2",
]
_DEMO_IDTEST = [r"a^2+b^2=c^2", r"\alpha\beta\gamma", r"\int_0^1 x\,dx", r"\sum_{k=1}^n k", r"e^{i\pi}+1=0",
                r"\nabla\cdot\mathbf{E}", r"\hat{x}_k", r"\lim_{t\to 0} f(t)", r"\mathcal{O}(n\log n)",
                r"\binom{n}{k}", r"\oint \vec{B}\cdot d\vec\ell", r"\det(\mathbf{A}-\lambda I)"]


def _demo_block(s):
    i = s[:len(_DEMO_INLINE)]
    return (f"<p>The pad resists spin with a friction arm {i[0]}, so the torsional coefficient {i[1]} falls "
            f"with the pinch force. Each finger is commanded {i[2]}, which holds the force while the tool "
            f"swings. Sample moments {i[3]} and the contact radius {i[4]} set the patch; the load ratio {i[5]} "
            f"stays below one, the normal term {i[6]} keeps the grip, the ratio {i[7]} is a fraction in "
            f"parentheses and {i[8]} is coloured. Descenders: gjpqy, figures 0123456789.</p>"
            + "".join(s[len(_DEMO_INLINE):]))


def _demo(out, scale):
    items = [(t, False) for t in _DEMO_INLINE] + [(t, True) for t in _DEMO_DISPLAY]
    s = render(items * 2, scale=scale)                   # light and dark copy: repeats get their own ids
    light, dark = _demo_block(s[:len(items)]), _demo_block(s[len(items):])
    ids = "".join(f"<div class=cell>{render([(t, j % 2 == 1)], scale=scale)[0]}<code>{html.escape(t)}</code></div>"
                  for j, t in enumerate(_DEMO_IDTEST))   # one LaTeX batch each: dvisvgm numbers fonts per batch
    Path(out).write_text(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Inline SVG math check</title>
<style>
body{{margin:0;padding:24px 16px;background:#EEF1F4;color:#1A2127;
  font:400 17px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}}
.blk{{max-width:44rem;margin:0 auto 18px;padding:14px 22px;border-radius:10px;background:#fff}}
.dark{{background:#12171B;color:#E5EBF0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(13rem,1fr));gap:8px}}
.cell{{background:#F6F8FA;border-radius:6px;padding:6px 10px}} .cell code{{display:block;font-size:12px;color:#77848D}}
{CSS}
</style></head><body>
<div class="blk">{light}</div>
<div class="blk dark">{dark}</div>
<div class="blk"><p>Twelve formulas, each from its own LaTeX batch, on one page (odd ones as display):</p>
<div class="grid">{ids}</div></div>
</body></html>
""")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--demo", metavar="OUT.html", required=True, help="write a demo page")
    ap.add_argument("--scale", type=float, default=SCALE)
    a = ap.parse_args()
    _demo(a.demo, a.scale)
    print(a.demo)
