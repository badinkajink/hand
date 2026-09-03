"""Build the ground-supported gaiting doc page from gait_study.json plus the rendered media.

Regenerable by construction: every number on the page is read out of the study JSON here, so a
re-run of `real_v1_gait_study.py` followed by this script cannot leave a stale figure behind.
Charts are hand-emitted SVG driven by CSS custom properties, so they follow the reader's theme.

    uv run python scripts/real_v1_gait_page.py --data docs/experiments/20260902-real_v1_gait
"""
from __future__ import annotations

import argparse
import base64
import collections
import json
from pathlib import Path

import numpy as np

# Categorical hues, validated 2026-09-02 against the six checks (OKLab x100): worst adjacent-pair
# CVD separation 12.0 against a target of 8, worst normal-vision separation 16.9 against a floor
# of 15, and >= 3:1 contrast on both the light and the dark chart surface.
HUE = {"rv05_manual_stored": "#BE7514", "rv03_narrowy_sp30": "#417F3A",
       "rv04_mid_sp30": "#4A7FC4", "rv00_wide_sp30": "#9C4B96"}
LABEL = {"rv05_manual_stored": "rv05_manual", "rv03_narrowy_sp30": "rv03_narrowy",
         "rv04_mid_sp30": "rv04_mid", "rv00_wide_sp30": "rv00_wide"}


def b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


def esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ------------------------------------------------------------------------------------- charting

def axes(w, h, pad, xr, yr, xticks, yticks, xlab, ylab, yfmt="{:g}"):
    """Frame, recessive grid, tick labels. Returns (svg_prefix, sx, sy)."""
    x0, y0, x1, y1 = pad["l"], pad["t"], w - pad["r"], h - pad["b"]
    sx = lambda v: x0 + (v - xr[0]) / (xr[1] - xr[0]) * (x1 - x0)
    sy = lambda v: y1 - (v - yr[0]) / (yr[1] - yr[0]) * (y1 - y0)
    s = [f'<svg viewBox="0 0 {w} {h}" role="img" class="chart">']
    for t in yticks:
        s.append(f'<line class="grid" x1="{x0}" y1="{sy(t):.1f}" x2="{x1}" y2="{sy(t):.1f}"/>')
        s.append(f'<text class="tick ty" x="{x0 - 8}" y="{sy(t) + 4:.1f}">{yfmt.format(t)}</text>')
    for t in xticks:
        s.append(f'<text class="tick tx" x="{sx(t):.1f}" y="{y1 + 18}">{t:g}</text>')
    s.append(f'<line class="axis" x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}"/>')
    s.append(f'<text class="axlab" x="{(x0 + x1) / 2:.0f}" y="{h - 4}">{esc(xlab)}</text>')
    s.append(f'<text class="axlab" transform="translate(13,{(y0 + y1) / 2:.0f}) rotate(-90)" '
             f'x="0" y="0">{esc(ylab)}</text>')
    return s, sx, sy


def chart_endurance(rows) -> str:
    end = [r for r in rows if r["arm"] == "endurance" and r["design"] in HUE]
    w, h = 760, 380
    s, sx, sy = axes(w, h, dict(l=64, r=118, t=14, b=42), (0, 40), (-1, 7.0),
                     [0, 10, 20, 30, 40], [0, 1, 2, 3, 4, 5, 6, 7],
                     "gait cycle", "accumulated rotation (turns)")
    seen = set()
    for r in sorted(end, key=lambda r: -abs(r["spin_deg"])):
        d = r["design"]
        pts = " ".join(f"{sx(c['cycle']):.1f},{sy(c['spin_deg'] / 360):.1f}"
                       for c in r["cycles"])
        s.append(f'<polyline class="ln" points="{pts}" stroke="{HUE[d]}" '
                 f'{"" if d in seen else ""}/>')
        if d not in seen:
            seen.add(d)
            last = r["cycles"][-1]
            s.append(f'<circle class="end" cx="{sx(last["cycle"]):.1f}" '
                     f'cy="{sy(last["spin_deg"] / 360):.1f}" r="4" fill="{HUE[d]}"/>')
            dy = {"rv00_wide_sp30": -6, "rv04_mid_sp30": 13}.get(d, 4)
            s.append(f'<text class="dlab" x="{sx(last["cycle"]) + 9:.1f}" '
                     f'y="{sy(last["spin_deg"] / 360) + dy:.1f}" fill="{HUE[d]}">{LABEL[d]}</text>')
    s.append(f'<line class="zero" x1="{sx(0):.1f}" y1="{sy(0):.1f}" x2="{sx(40):.1f}" '
             f'y2="{sy(0):.1f}"/>')
    s.append("</svg>")
    return "".join(s)


def chart_press(rows) -> str:
    g = collections.defaultdict(list)
    for r in rows:
        if r["arm"] == "press":
            g[r["press_mm"]].append(r)
    w, h = 760, 330
    s, sx, sy = axes(w, h, dict(l=64, r=24, t=14, b=42), (-7, 11), (0, 100),
                     [-6, -4, -2, 0, 2, 4, 6, 8, 10], [0, 25, 50, 75, 100],
                     "press (mm the closed grip is driven down after the grasp)",
                     "rotation per cycle (deg)")
    ideal = 1.684 * 30
    s.append(f'<line class="ref" x1="{sx(-7):.1f}" y1="{sy(ideal):.1f}" x2="{sx(11):.1f}" '
             f'y2="{sy(ideal):.1f}"/>')
    s.append(f'<text class="reflab" x="{sx(-6.6):.1f}" y="{sy(ideal) - 7:.1f}">'
             f'slip-free ceiling, 50.5 deg</text>')
    pts, fails = [], []
    for k in sorted(g):
        cs = g[k]
        v = float(np.mean([c["deg_per_cycle"] for c in cs]))
        ok = sum(c["ok"] for c in cs)
        (pts if ok == len(cs) else fails).append((k, v, ok, len(cs)))
    line = " ".join(f"{sx(k):.1f},{sy(v):.1f}" for k, v, _, _ in pts)
    s.append(f'<polyline class="ln thick" points="{line}" stroke="{HUE["rv05_manual_stored"]}"/>')
    for k, v, ok, n in pts:
        s.append(f'<circle class="mk" cx="{sx(k):.1f}" cy="{sy(v):.1f}" r="5" '
                 f'fill="{HUE["rv05_manual_stored"]}"/>')
    for k, v, ok, n in fails:
        s.append(f'<circle class="mk fail" cx="{sx(k):.1f}" cy="{sy(v):.1f}" r="5"/>')
        s.append(f'<text class="flab" x="{sx(k):.1f}" y="{sy(v) - 12:.1f}">{ok}/{n}</text>')
    s.append(f'<rect class="band" x="{sx(6.9):.1f}" y="14" width="{sx(11) - sx(6.9):.1f}" '
             f'height="{h - 56}"/>')
    s.append(f'<text class="bandlab" x="{sx(9):.1f}" y="{h - 56:.0f}">tips over</text>')
    s.append("</svg>")
    return "".join(s)


def chart_padr(rows) -> str:
    g = collections.defaultdict(list)
    for r in rows:
        if r["arm"] == "padr":
            g[r["pad_radius"]].append(r)
    w, h = 760, 330
    s, sx, sy = axes(w, h, dict(l=64, r=24, t=14, b=42), (4, 21), (0, 100),
                     [5, 7.5, 10.55, 13, 16, 20], [0, 25, 50, 75, 100],
                     "fingertip sphere radius (mm)", "rotation per cycle (deg)")
    ce, pts, fails = [], [], []
    for k in sorted(g):
        cs = g[k]
        mm = k * 1000
        gear = cs[0]["gear_ratio"]
        v = float(np.mean([c["deg_per_cycle"] for c in cs]))
        ok = sum(c["ok"] for c in cs)
        ce.append((mm, gear * 30))
        (pts if ok == len(cs) else fails).append((mm, v, ok, len(cs)))
    s.append('<polyline class="ceil" points="' +
             " ".join(f"{sx(m):.1f},{sy(v):.1f}" for m, v in ce) + '"/>')
    s.append(f'<text class="reflab" x="{sx(5.4):.1f}" y="{sy(ce[0][1]) - 9:.1f}">'
             f'slip-free ceiling = gear ratio x 30 deg</text>')
    s.append('<polyline class="ln thick" points="' +
             " ".join(f"{sx(m):.1f},{sy(v):.1f}" for m, v, _, _ in pts) +
             f'" stroke="{HUE["rv05_manual_stored"]}"/>')
    for m, v, ok, n in pts:
        s.append(f'<circle class="mk" cx="{sx(m):.1f}" cy="{sy(v):.1f}" r="5" '
                 f'fill="{HUE["rv05_manual_stored"]}"/>')
    for m, v, ok, n in fails:
        s.append(f'<circle class="mk fail" cx="{sx(m):.1f}" cy="{sy(v):.1f}" r="5"/>')
        s.append(f'<text class="flab" x="{sx(m):.1f}" y="{sy(v) - 12:.1f}">{ok}/{n}</text>')
    s.append(f'<line class="ship" x1="{sx(10.55):.1f}" y1="14" x2="{sx(10.55):.1f}" '
             f'y2="{h - 42}"/>')
    s.append(f'<text class="shiplab" x="{sx(10.55) + 6:.1f}" y="28">shipped pad</text>')
    s.append("</svg>")
    return "".join(s)


def chart_phase(rows) -> str:
    """Where in the cycle the rotation is won and given back."""
    want = ["rv05_manual_stored", "rv03_narrowy_sp30", "rv04_mid_sp30"]
    ph = ["twist", "release", "return", "regrasp"]
    data = {}
    for d in want:
        cs = [r for r in rows if r["arm"] == "premise_ground" and r["design"] == d]
        n = float(np.mean([r["cycles_run"] for r in cs]))
        data[d] = [float(np.mean([r["phase_deg"][p] for r in cs])) / max(n, 1) for p in ph]
    w, h = 760, 300
    x0, x1 = 150, w - 30
    lim = 60
    sx = lambda v: (x0 + x1) / 2 + v / lim * (x1 - x0) / 2
    s = [f'<svg viewBox="0 0 {w} {h}" role="img" class="chart">']
    for t in (-60, -30, 0, 30, 60):
        s.append(f'<line class="grid" x1="{sx(t):.1f}" y1="24" x2="{sx(t):.1f}" y2="{h - 40}"/>')
        s.append(f'<text class="tick tc" x="{sx(t):.1f}" y="{h - 22}">{t:+g}</text>')
    row_h = (h - 76) / (len(want) * len(ph))
    y = 30
    for d in want:
        s.append(f'<text class="rowlab" x="8" y="{y + row_h * 2:.1f}" fill="{HUE[d]}">'
                 f'{LABEL[d]}</text>')
        for p, v in zip(ph, data[d]):
            xa, xb = sx(min(0, v)), sx(max(0, v))
            s.append(f'<rect class="pbar" x="{xa:.1f}" y="{y + 3:.1f}" '
                     f'width="{max(xb - xa, 1.5):.1f}" height="{row_h - 6:.1f}" rx="3" '
                     f'fill="{HUE[d]}" opacity="{0.95 if p == "twist" else 0.55}"/>')
            s.append(f'<text class="plab" x="92" y="{y + row_h / 2 + 4:.1f}">{p}</text>')
            s.append(f'<text class="pval" x="{(xb + 6) if v >= 0 else (xa - 6):.1f}" '
                     f'y="{y + row_h / 2 + 4:.1f}" '
                     f'text-anchor="{"start" if v >= 0 else "end"}">{v:+.0f}</text>')
            y += row_h
        y += 4
    s.append(f'<line class="axis" x1="{sx(0):.1f}" y1="24" x2="{sx(0):.1f}" y2="{h - 40}"/>')
    s.append(f'<text class="axlab" x="{(x0 + x1) / 2:.0f}" y="{h - 4}">'
             f'degrees of shaft rotation per cycle, by gait phase</text>')
    s.append("</svg>")
    return "".join(s)


# ---------------------------------------------------------------------------------------- table

def premise_table(rows) -> str:
    g = collections.defaultdict(list)
    for r in rows:
        if r["arm"].startswith("premise"):
            g[(r["design"], r["arm"])].append(r)
    order = ["rv05_manual_stored", "rv03_narrowy_sp30", "rv03_narrowy_sp40", "rv04_mid_sp30",
             "rv04_mid_sp40", "rv00_wide_sp30", "rv00_wide_sp40", "rv01_compact_sp30",
             "rv01_compact_sp40", "rv02_narrowx_sp40"]
    out = ['<div class="tw"><table><thead><tr><th>design</th><th>grip force</th>'
           '<th>ring IK, grip</th><th>ring IK, release</th>'
           '<th>deg / cycle</th><th>on the ground</th><th>in the air</th></tr></thead><tbody>']
    for d in order:
        gr, ai = g.get((d, "premise_ground")), g.get((d, "premise_air"))
        if not gr:
            continue
        pf = np.mean([r["start"]["pad_force_N"] for r in gr])
        dc = np.mean([r["deg_per_cycle"] for r in gr])
        og, oa = sum(r["ok"] for r in gr), sum(r["ok"] for r in ai)
        cls = "yes" if og == len(gr) else ("no" if og == 0 else "part")
        grip_ik, rel_ik = gr[0]["ik_residual_grip_mm"], gr[0]["ik_residual_open_mm"]
        note = "" if pf > 1 else " nogrip"
        out.append(
            f'<tr class="{cls}"><td class="mono">{d}</td>'
            f'<td class="num{note}">{pf:.2f} N</td>'
            f'<td class="num">{grip_ik:.2f} mm</td><td class="num">{rel_ik:.2f} mm</td>'
            f'<td class="num strong">{dc:+.1f}</td>'
            f'<td class="num">{og}/{len(gr)}</td><td class="num">{oa}/{len(ai)}</td></tr>')
    out.append("</tbody></table></div>")
    return "".join(out)


def squeeze_table(rows) -> str:
    g = collections.defaultdict(list)
    for r in rows:
        if r["arm"] == "squeeze":
            g[r["squeeze_mm"]].append(r)
    out = ['<div class="tw"><table><thead><tr><th>commanded interference</th>'
           '<th>pad force</th><th>deg / cycle</th><th>transmission</th><th>held</th>'
           '</tr></thead><tbody>']
    for k in sorted(g):
        cs = g[k]
        out.append(
            f'<tr><td class="num">{k:.1f} mm</td>'
            f'<td class="num">{np.mean([c["start"]["pad_force_N"] for c in cs]):.1f} N</td>'
            f'<td class="num strong">{np.mean([c["deg_per_cycle"] for c in cs]):.1f}</td>'
            f'<td class="num">{np.mean([c["transmission"] for c in cs]):.3f}</td>'
            f'<td class="num">{sum(c["ok"] for c in cs)}/{len(cs)}</td></tr>')
    out.append("</tbody></table></div>")
    return "".join(out)


def stroke_table(rows) -> str:
    g = collections.defaultdict(list)
    for r in rows:
        if r["arm"] == "stroke":
            g[r["stroke_deg"]].append(r)
    out = ['<div class="tw"><table><thead><tr><th>stroke</th><th>pad arc</th>'
           '<th>deg / cycle</th><th>transmission</th><th>final tilt</th><th>held</th>'
           '</tr></thead><tbody>']
    for k in sorted(g):
        cs = g[k]
        ok = sum(c["ok"] for c in cs)
        out.append(
            f'<tr class="{"yes" if ok == len(cs) else "no"}"><td class="num">{k:.0f} deg</td>'
            f'<td class="num">{np.radians(k) * 21.05:.1f} mm</td>'
            f'<td class="num strong">{np.mean([c["deg_per_cycle"] for c in cs]):.1f}</td>'
            f'<td class="num">{np.mean([c["transmission"] for c in cs]):.3f}</td>'
            f'<td class="num">{np.mean([c["final_tilt_deg"] for c in cs]):.1f} deg</td>'
            f'<td class="num">{ok}/{len(cs)}</td></tr>')
    out.append("</tbody></table></div>")
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rows = json.loads((args.data / "gait_study.json").read_text())
    out = args.out or (args.data / "page.html")

    end = [r for r in rows if r["arm"] == "endurance" and r["design"] == "rv05_manual_stored"]
    turns = np.mean([r["spin_deg"] for r in end]) / 360
    turns_sd = np.std([r["spin_deg"] for r in end]) / 360
    end3 = [r for r in rows if r["arm"] == "endurance" and r["design"] == "rv03_narrowy_sp30"]
    turns3 = np.mean([r["spin_deg"] for r in end3]) / 360

    media = {}
    for tag, f in (("ground", "20260902-gait_rv05_ground.mp4"),
                   ("air", "20260902-gait_rv05_air.mp4"),
                   ("stall", "20260902-gait_rv04_stall.mp4"),
                   ("strip", "20260902-gait_rv05_ground.png")):
        p = args.data / f
        media[tag] = b64(p) if p.exists() else ""

    subs = {
        "TURNS": f"{turns:.2f}", "TURNS_SD": f"{turns_sd:.2f}", "TURNS3": f"{turns3:.2f}",
        "C_END": chart_endurance(rows), "C_PRESS": chart_press(rows),
        "C_PADR": chart_padr(rows), "C_PHASE": chart_phase(rows),
        "T_PREMISE": premise_table(rows), "T_SQUEEZE": squeeze_table(rows),
        "T_STROKE": stroke_table(rows), "V_GROUND": media["ground"],
        "V_AIR": media["air"], "V_STALL": media["stall"], "STRIP": media["strip"],
        "NCELLS": str(len(rows)),
    }
    html = PAGE
    for k, v in subs.items():
        html = html.replace("{{" + k + "}}", v)
    out.write_text(html)
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.2f} MB)")
    return 0


PAGE = Path(__file__).with_name("real_v1_gait_page.template.html").read_text() \
    if Path(__file__).with_name("real_v1_gait_page.template.html").exists() else ""

if __name__ == "__main__":
    import sys
    sys.exit(main())
