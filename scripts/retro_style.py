#!/usr/bin/env python3
"""Plain 1990s-homepage style for the result pages: Times on white, default link colours, 1px black rules.

The look follows the owner's contact tutorial (~/wxie_workspace/contact/tutorial/slides/lib/retro.css) and
sr2-hand.github.io. The house template that every builder copies defines its colours and fonts as CSS variables
(--ink, --paper, --rule, --f-body, ...) and its inline SVG charts read the same variables, so this sheet redefines
the variables and flattens the components; LaTeX SVGs, figures, videos and charts are untouched.

    python3 scripts/retro_style.py PAGE.html [--out OUT.html]   # restyle a built page (in place without --out)

Builders call apply(html) on the finished page before writing it.
"""
from __future__ import annotations

import argparse
import re

RETRO_CSS = r"""
/* retro_style.py: plain page style (owner, 2026-10-09) */
:root, :root[data-theme], html {
  color-scheme: light !important;
  --paper:#fff !important; --card:#fff !important; --sunk:#f2f2f2 !important; --shadow:none !important;
  --ink:#000 !important; --ink2:#000 !important; --ink3:#444 !important;
  --rule:#000 !important; --rule2:#999 !important;
  --s1:#BE7514 !important; --s2:#4A7FC4 !important; --s3:#5E8F5A !important;  /* chart series: amber, blue, green */
  --good:#008800 !important; --bad:#cc0000 !important; --ref:#666 !important;
  --f-body:"Times New Roman",Times,serif !important; --f-display:"Times New Roman",Times,serif !important;
  --f-mono:"Courier New",Courier,monospace !important;
}
html, body { background:#fff !important; color:#000 !important; }
body { font:18px/1.4 "Times New Roman",Times,serif !important; margin:0 !important; }
.wrap, main, body > .col { max-width:860px !important; margin:0 auto !important; padding:24px 16px 48px !important; }
.col { max-width:none !important; margin:0 !important; padding:0 !important; }
header, footer, section, nav, figure, .numbers, .keep, .handoff, .step, .derive, .card, .panel, .box {
  background:none !important; box-shadow:none !important; border-radius:0 !important;
}
header { border:0 !important; padding:0 0 6px !important; margin:0 !important; }
section { border:0 !important; padding:0 !important; margin:0 0 8px !important; }
h1, h2, h3, h4, .tag, th, .eyebrow, .lede, .byline, nav, figcaption, .note, .sub, .chart h4 {
  font-family:"Times New Roman",Times,serif !important; letter-spacing:normal !important; text-transform:none !important;
}
h1 { font-size:27px !important; line-height:1.2 !important; font-weight:bold !important; margin:12px 0 8px !important; max-width:none !important; }
h2 { font-size:21px !important; font-weight:bold !important; border-bottom:1px solid #000 !important;
     margin:30px 0 10px !important; padding:0 0 2px !important; }
h3 { font-size:18px !important; font-weight:bold !important; margin:18px 0 6px !important; }
h2 + .sub, .sub { font-size:16px !important; color:#444 !important; }
.eyebrow { font-size:15px !important; font-weight:normal !important; color:#444 !important; margin:0 !important; }
.lede { font-size:18px !important; line-height:1.4 !important; font-weight:normal !important; color:#000 !important; }
.byline { font-size:15px !important; color:#444 !important; border:0 !important; }
p, li { font-size:18px !important; line-height:1.4 !important; color:#000 !important; max-width:none !important; }
strong, b { font-weight:bold !important; }
a, .col a, footer a, nav a, nav.toc a { color:#0000ee !important; text-decoration:underline !important; }
a:visited, .col a:visited { color:#551a8b !important; }
nav.toc { border-bottom:1px solid #000 !important; padding:6px 0 10px !important; }
nav.toc ol { font:16px/1.5 "Times New Roman",Times,serif !important; }
hr { border:0 !important; border-top:1px solid #000 !important; }
code, pre, kbd, .mono, td.num, th.num {
  font-family:"Courier New",Courier,monospace !important;
}
code, .m, kbd, samp { font-size:0.9em !important; background:none !important; padding:0 !important;
  color:#000 !important; font-family:"Courier New",Courier,monospace !important; }
pre { background:#f2f2f2 !important; border:0 !important; border-radius:0 !important; padding:10px 14px !important;
      font-size:15px !important; }
.note, .keep, .callout, .warn {
  background:#ffffcc !important; border:1px solid #000 !important; padding:8px 12px !important; color:#000 !important;
}
.numbers, .handoff { border:1px solid #000 !important; padding:8px 12px !important; }
.derive, .step { border:0 !important; padding:0 !important; overflow:visible !important; }
table { border-collapse:collapse !important; max-width:100% !important; font:16px/1.3 "Times New Roman",Times,serif !important; }
th, td { border:1px solid #000 !important; padding:3px 7px !important; background:#fff !important; color:#000 !important; }
th { background:#f2f2f2 !important; font-weight:bold !important; font-size:16px !important; }
td.num, th.num { font-size:16px !important; }
table.index td, table.index th { font-size:16px !important; vertical-align:top !important; }
tr.grp td, table.index tr.grp td { background:#f2f2f2 !important; font:bold 16px/1.3 "Times New Roman",Times,serif !important; }
td.cell { background:#fff !important; }
td.c4 { background:#b8eeb8 !important; } td.c3 { background:#e2f7e2 !important; }
td.c2 { background:#ffe2a8 !important; } td.c1, td.c0 { background:#ffc4c4 !important; }
.tw { overflow-x:auto !important; border:0 !important; max-width:100% !important; }
.tw table { max-width:none !important; }
.tnote { font:15px/1.35 "Times New Roman",Times,serif !important; color:#000 !important; }
figure { margin:16px 0 !important; padding:0 !important; border:0 !important; text-align:center !important; }
figure > img, figure > video, figure > svg, figure > .chart, figure > .duo, .chart > svg, .diagram > svg {
  display:block !important; margin-left:auto !important; margin-right:auto !important; }
.chart .legendrow, .chart .legend { justify-content:center !important; }
/* two-up chart grids stack: a chart drawn for the full column shows its text at half size side by side */
.duo, .pair, .two, .grid2, .fig-row { grid-template-columns:1fr !important; }
figure img, figure video, figure svg { border:1px solid #000 !important; border-radius:0 !important; max-width:100% !important; }
.eq svg, .tex-d svg, .tex-i svg, svg.tex, .chart svg, .diagram svg { border:0 !important; }
figcaption { text-align:left !important; font-size:16px !important; line-height:1.35 !important; color:#000 !important; margin-top:4px !important; }
.chart, .duo, .diagram { background:#fff !important; border:0 !important; padding:0 !important; max-width:100% !important; overflow-x:auto !important; }
.chart svg, .diagram > svg { max-width:100% !important; height:auto !important; }
.chart .tip { background:#ffffcc !important; color:#000 !important; border:1px solid #000 !important; border-radius:0 !important;
              box-shadow:none !important; font-family:"Times New Roman",Times,serif !important; }
.key { border-radius:0 !important; }
footer, footer p { border-top:0 !important; font:16px/1.4 "Times New Roman",Times,serif !important; color:#000 !important; }
footer { border-top:1px solid #000 !important; margin-top:32px !important; padding-top:6px !important; }
footer code { font-family:"Courier New",Courier,monospace !important; font-size:14px !important; }
.chip, .pill, .badge { background:none !important; border:0 !important; padding:0 !important; border-radius:0 !important;
                       font:inherit !important; color:#000 !important; }
button.theme, .theme-toggle, [data-theme-toggle] { display:none !important; }
"""

FONT_LINK_RE = re.compile(r'<link[^>]+(fonts\.googleapis\.com|fonts\.gstatic\.com)[^>]*>\s*', re.I)
OLD_RE = re.compile(r'<style id="retro">.*?</style>\s*', re.S)
TH_RE = re.compile(r'(<th\b[^>]*>)((?:\s|<[^>]+>)*)([a-z][^<]*)')
# lower-case words that name a parameter, a unit or an abbreviation and stay as written at the start of a header
KEEP = {"condim", "impratio", "solref", "solimp", "mjlab", "rms", "sd", "id", "ok", "kp", "kv", "dt", "nv", "nc", "rel"}


def sentence_case_headers(html: str) -> str:
    """Table headers in sentence case (owner, 2026-10-09): capitalise the first letter of a header that starts with a
    plain lower-case word of three or more letters; symbols (q, kp, z_rel), identifiers (rv05_manual) and KEEP stay."""
    def sub(m):
        text = m.group(3)
        word = re.match(r"[A-Za-z]+", text).group(0)
        rest = text[len(word):]
        if len(word) < 3 or word in KEEP or (rest[:1] in ("_",) or rest[:1].isdigit()):
            return m.group(0)
        return m.group(1) + m.group(2) + text[0].upper() + text[1:]
    return TH_RE.sub(sub, html)


def apply(html: str) -> str:
    """Restyle a built page: drop web-font links, add the plain sheet after every other style block, and set the table
    headers in sentence case."""
    html = sentence_case_headers(FONT_LINK_RE.sub('', OLD_RE.sub('', html)))
    tag = '<style id="retro">' + RETRO_CSS.strip() + '</style>\n'
    i = html.lower().rfind('</head>')
    if i < 0:  # artifact-style fragment without <head>: the sheet goes after the last style block
        j = html.lower().rfind('</style>')
        i = j + len('</style>') if j >= 0 else 0
    return html[:i] + tag + html[i:]


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('page')
    p.add_argument('--out')
    a = p.parse_args()
    html = open(a.page, encoding='utf-8').read()
    open(a.out or a.page, 'w', encoding='utf-8').write(apply(html))


if __name__ == '__main__':
    main()
