#!/usr/bin/env python3
"""Build docs/experiments/20261001-hom_contact_patch/20261001-hom_pinch_contact_models.html from the study's JSON.

    python3 scripts/hom_contact_patch_page.py

Reads torsion.jsonl, brake.jsonl, cop.jsonl, calibration.json and media/ written by
scripts/hom_contact_study.py; every number quoted in the prose is computed here from those rows.
"""
from __future__ import annotations

import base64
import html
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import hom_contact_rig as H  # noqa: E402  (numpy-only at import; no simulator needed)

D = os.path.join(ROOT, "docs/experiments/20261001-hom_contact_patch")
OUT = os.path.join(D, "20261001-hom_pinch_contact_models.html")
TPL = os.path.join(ROOT, "scripts/hom_contact_patch_page.template.html")
DRAKE_V, MUJOCO_V = "1.57.0", "3.6.0"           # the study venv, logs/20261001-hom_contact/venv

CAL = json.load(open(os.path.join(D, "calibration.json")))
C1, CF, PF = CAL["mu_t_at_1N_m"], CAL["fit_c_m"], CAL["fit_p"]
S = {
    "dref": "drake:hydro:E1e7:r1:rt0.01",
    "d02": "drake:hydro:E1e7:r1:rt0.02",
    "ddef": "drake:hydro:E1e7:r1",
    "dpt": "drake:point",
    "mp3": "mj:point3",
    "p4": f"mj:point4:mt{C1:.6g}:ir100",
    "p4ir10": f"mj:point4:mt{C1:.6g}",
    "p4s": f"mj:point4s:fit{CF:.6g}x{PF:.4f}:ir100",
    "s2": "mj:spheres:s2:rs0.75:ir100:tr0.02",
    "s1": "mj:spheres:s1:rs0.75:ir100:tr0.02",
    "s05": "mj:spheres:s0.5:rs0.75:ir100:tr0.02",
    "s05tr01": "mj:spheres:s0.5:rs0.75:ir100:tr0.1",
    "s05ir10": "mj:spheres:s0.5:rs0.75:tr0.02",
}
LABEL = {
    "dref": "Drake hydroelastic, relaxation 0.01&#8202;s",
    "d02": "Drake hydroelastic, relaxation 0.02&#8202;s",
    "ddef": "Drake hydroelastic, relaxation 0.1&#8202;s (default)",
    "dpt": "Drake point contact",
    "mp3": "MuJoCo point contact, condim&#160;3",
    "p4": "MuJoCo condim&#160;4, constant &#956;<sub>t</sub>",
    "p4ir10": "MuJoCo condim&#160;4, constant &#956;<sub>t</sub>, impratio&#160;10",
    "p4s": "MuJoCo condim&#160;4, &#956;<sub>t</sub> rescheduled",
    "s2": "MuJoCo sphere pads, 2&#8202;mm",
    "s1": "MuJoCo sphere pads, 1&#8202;mm",
    "s05": "MuJoCo sphere pads, 0.5&#8202;mm",
    "s05tr01": "MuJoCo sphere pads, 0.5&#8202;mm, relaxation 0.1&#8202;s",
    "s05ir10": "MuJoCo sphere pads, 0.5&#8202;mm, impratio&#160;10",
}
COL = {"drake": "var(--c-drake)", "sphere": "var(--c-sphere)", "c4": "var(--c-c4)", "ref": "var(--c-ref)"}


def rows(name):
    p = os.path.join(D, name)
    return [json.loads(l) for l in open(p) if l.strip()]


TOR, BRK, COP, PLATE = rows("torsion.jsonl"), rows("brake.jsonl"), rows("cop.jsonl"), rows("plate.jsonl")


def tor(key, N, w=2.0):
    for r in TOR:
        if r["spec"] == S[key] and r["N"] == N and r["omega"] == w:
            return r
    return None


def rb(key, N, w=2.0):
    r = tor(key, N, w)
    return None if r is None else r["rbar_per_pad_mm"]


def brk(key, d=15.0, T=4.0):
    for r in BRK:
        if r["spec"] == S[key] and abs(r["d_mm"] - d) < 1e-6 and abs(r["T_ramp"] - T) < 1e-6:
            return r
    return None


def f2(v, nd=2, none="&#8211;"):
    return none if v is None else f"{v:.{nd}f}".replace("-", "&#8722;")


def sgn(v, nd=0):
    """Signed percentage with a typographic minus."""
    return f"{v:+.{nd}f}".replace("-", "&#8722;") + "&#8202;%"


def pct(a, b):
    return 100.0 * (a - b) / b


# ------------------------------------------------------------------- quasi-static brake law

W = H.M_TOOL * H.G


def rbar_fit(N):
    return CF * N ** PF


def phi_qs(N, d):
    cap = 2 * H.MU * rbar_fit(N) * math.sqrt(max(N * N - (W / (2 * H.MU)) ** 2, 0.0))
    c = min(1.0, cap / (W * d))
    return math.degrees(math.acos(c))


def n_qs(phi_deg, d):
    lo, hi = 0.13, 20.0
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        if phi_qs(mid, d) > phi_deg:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


# -------------------------------------------------------------------------------- tables

def table(head, body, cls_num=None):
    cls_num = cls_num or set()
    out = ['<div class="tw"><table><thead><tr>']
    for i, h in enumerate(head):
        out.append(f'<th{" class=num" if i in cls_num else ""}>{h}</th>')
    out.append("</tr></thead><tbody>")
    for r in body:
        if isinstance(r, str):
            out.append(f'<tr class="grp"><td colspan="{len(head)}">{r}</td></tr>')
            continue
        out.append("<tr>")
        for i, c in enumerate(r):
            if isinstance(c, tuple):
                out.append(f'<td class="{c[1]}">{c[0]}</td>')
            else:
                out.append(f'<td{" class=num" if i in cls_num else ""}>{c}</td>')
        out.append("</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def table_models():
    head = ["model", "contact", "parameters"]
    body = [
        ["Drake hydroelastic", "compliant pads, rigid tool, SAP",
         "E 10&#8202;MPa (Drake&#8217;s default), 1&#8202;mm tetrahedra, tool mesh 0.5&#8202;mm; relaxation 0.1&#8202;s (default), 0.02 or 0.01&#8202;s"],
        ["Drake point", "one point pair per pad, SAP", "point stiffness 10<sup>4</sup>&#8202;N/m"],
        ["MuJoCo point", "one contact per pad, condim&#160;3", "solref 0.006&#8202;1, solimp 0.97&#8202;0.995 (the real_v1 fingertip default)"],
        ["MuJoCo condim&#160;4", "one contact per pad, condim&#160;4",
         f"&#956;<sub>t</sub> constant at Drake&#8217;s arm at 1&#8202;N ({C1 * 1e3:.3f}&#8202;mm), or rescheduled every step as "
         f"&#956;&#183;{CF * 1e3:.3f}&#8202;mm&#183;(N&#8202;/&#8202;1&#8202;N)<sup>{PF:.3f}</sup> from the previous step&#8217;s normal force"],
        ["MuJoCo sphere pads", "35&#176; cap of 0.75&#8202;mm spheres, condim&#160;3 each",
         "envelope = the fingertip sphere; spacing 2, 1, 0.5&#8202;mm (31, 126, 506 spheres per pad); per-sphere "
         "stiffness E/R &#215; area per sphere &#215; ((R&#8722;r<sub>s</sub>)/R)&#178;; solref (r/2, 1), relaxation 0.02 or 0.1&#8202;s"],
    ]
    t = table(head, body)
    return t + ('<p class="note" style="font-size:13.5px;color:var(--ink3)">All MuJoCo variants: elliptic cone, Newton, '
                'implicitfast, impratio&#160;100 unless marked impratio&#160;10. The two pads and the tool share &#956;&#8202;=&#8202;1.0 in both '
                'simulators.</p>')


def closeness(v, ref, nd=3):
    if v is None or ref is None:
        return ("&#8211;", "cell")
    e = abs(pct(v, ref))
    cls = "c4" if e <= 5 else "c3" if e <= 10 else "c2" if e <= 25 else "c0"
    return (f"{v:.{nd}f}", f"cell {cls}")


def table_torsion():
    Ns = (0.25, 0.5, 1.0, 2.0, 4.0)
    head = ["model"] + [f"{n:g}&#8202;N" for n in Ns] + ["worst vs Drake", "contacts at 1&#8202;N"]
    ref = {N: rb("dref", N) for N in Ns}
    body = [["elastic foundation, analytic"] + [f2(H.winkler_law(N, 1e7)[1] * 1e3, 3) for N in Ns] + ["", ""]]
    for k in ("dref", "d02", "ddef", "dpt", "mp3", "p4", "p4s", "s2", "s1", "s05", "s05tr01"):
        vals = [rb(k, N) for N in Ns]
        if k == "dref":
            cells = [f2(v, 3) for v in vals]
            worst = "reference"
        elif k in ("dpt", "mp3"):
            cells = [("0", "cell c0") for _ in vals]
            worst = "&#8722;100&#8202;%"
        else:
            cells = [closeness(v, ref[N]) for v, N in zip(vals, Ns)]
            ws = [pct(v, ref[N]) for v, N in zip(vals, Ns) if v is not None]
            worst = sgn(max(ws, key=abs))
        r1 = tor(k, 1.0)
        nc = r1["n_L"] if r1 else "&#8211;"
        body.append([LABEL[k]] + cells + [worst, nc])
    return table(head, body, cls_num={1, 2, 3, 4, 5, 6, 7}) + (
        '<p class="note" style="font-size:13.5px;color:var(--ink3)">r&#772; in mm per pad at 2&#8202;rad/s; shading is the distance '
        'from Drake at 0.01&#8202;s: within 5&#8202;%, 10&#8202;%, 25&#8202;%, beyond. Contacts are the pad&#8217;s faces in Drake and its '
        'contacts in MuJoCo.</p>')


def table_rate():
    head = ["model", "force", "0.5&#8202;rad/s", "2&#8202;rad/s", "change"]
    spec = [("ddef", 1.0), ("d02", 1.0), ("dref", 1.0), ("s05tr01", 1.0), ("s05", 1.0),
            ("p4ir10", 4.0), ("p4", 4.0), ("s05ir10", 4.0), ("s05", 4.0)]
    body = []
    for k, N in spec:
        a, b = rb(k, N, 0.5), rb(k, N, 2.0)
        ch = pct(b, a) if (a and b) else None
        cls = "cell c0" if ch is not None and abs(ch) > 10 else "cell c4"
        body.append([LABEL[k], f"{N:g}&#8202;N", f2(a, 3), f2(b, 3), (sgn(ch) if ch is not None else "&#8211;", cls)])
    return table(head, body, cls_num={2, 3})


def brake_cells(r, d):
    qs = {a: n_qs(a, d * 1e-3) for a in (45, 80)}
    if r is None:
        return ["&#8211;"] * 8
    point = r["N_at_10"] == r["N0"] and r["phi_max"] > 150
    last = r["trace"][-1]
    # dropped = a pad lost contact or the tool slid more than 20 mm through the pinch (the centre of
    # mass falls by d whenever the tool reaches vertical, so its height is no test)
    dropped = last[7] == 0 or last[8] == 0 or abs(r["slip_end_mm"]) > 20
    c10 = "at release" if point else f2(r["N_at_10"])
    cells = [c10]
    for a in (45, 80):
        v = r[f"N_at_{a}"]
        cells.append(("at release", "cell c0") if point else closeness(v, qs[a], nd=2))
    mx = r["phi_max"]
    cells.append((f"{mx:.0f}&#176;", "cell c0" if mx > 100 else "cell c4"))
    cells.append("dropped" if dropped else ("swinging" if point else f"{r['phi_end']:.1f}&#176;"))
    cells.append("&#8211;" if dropped else f"{abs(r['slip_end_mm']):.2f}")
    cells.append(f"{r['peak_rate_dps']:.0f}")
    cells.append(f"{r['wall_s']:.1f}")
    return cells


def table_brake():
    head = ["model", "N at 10&#176;", "N at 45&#176;", "N at 80&#176;", "max &#966;", "final &#966;", "slip mm",
            "peak deg/s", "CPU s"]
    keys = ("dref", "d02", "ddef", "dpt", "mp3", "p4", "p4ir10", "p4s", "s2", "s1", "s05", "s05tr01", "s05ir10")
    body = []
    for d, T in ((15.0, 4.0), (30.0, 4.0), (15.0, 12.0)):
        body.append(f"d&#8202;=&#8202;{d:g}&#8202;mm, {T:g}&#8202;s ramp")
        q = [f2(n_qs(a, d * 1e-3)) for a in (10, 45, 80)]
        body.append(["quasi-static law, Drake reference r&#772;(N)"] + q + ["", "90&#176;", "", "", ""])
        for k in keys:
            r = brk(k, d, T)
            if r is None:
                continue
            body.append([LABEL[k]] + brake_cells(r, d))
    return table(head, body, cls_num={1, 2, 3, 4, 5, 6, 7, 8}) + (
        '<p class="note" style="font-size:13.5px;color:var(--ink3)">N at an angle = pinch force per pad when the tool first passes '
        'it; shaded by distance from the quasi-static law (within 5, 10, 25&#8202;%, beyond). Slip = travel of the pinch point '
        'along the tool axis. CPU s = wall time of the 5.5&#8202;s run on one core; Drake&#8217;s includes 5500 Python '
        '<code>AdvanceTo</code> calls.</p>')


def table_cop():
    head = ["model", "contacts", "max error mm", "rms mm", "largest jump per 0.25&#176; mm"]
    order = [("dref", "Drake hydroelastic, 1&#8202;mm tetrahedra"), (None, "Drake hydroelastic, 0.5&#8202;mm tetrahedra"),
             ("mp3", LABEL["mp3"]), ("s2", LABEL["s2"]), ("s1", LABEL["s1"]), ("s05", LABEL["s05"])]
    body = []
    for k, lab in order:
        spec = S[k] if k else "drake:hydro:E1e7:r0.5"
        r = next((x for x in COP if x["spec"] == spec), None)
        if r is None:
            continue
        nc = f"{min(r['n'])}" if min(r["n"]) == max(r["n"]) else f"{min(r['n'])}&#8211;{max(r['n'])}"
        body.append([lab, nc, f"{r['err_max_mm']:.4f}", f"{r['err_rms_mm']:.4f}",
                     f"{r['max_step_mm']:.4f}"])
    return table(head, body, cls_num={1, 2, 3, 4})


# -------------------------------------------------------------------------------- charts

def svg_chart(title, unit, series, legend, *, xlog, xmin, xmax, xticks, ymin, ymax, yticks, rev=False,
              xname="N", xd=2, yd=3, xlabel="", w=640, h=360, caption="", narrow=False):
    m = dict(l=50, r=18, t=12, b=42)
    x0, x1, y0, y1 = m["l"], w - m["r"], h - m["b"], m["t"]

    def X(x):
        f = (math.log(x) - math.log(xmin)) / (math.log(xmax) - math.log(xmin)) if xlog else (x - xmin) / (xmax - xmin)
        return x0 + ((1 - f) if rev else f) * (x1 - x0)

    def Y(y):
        return y0 - (y - ymin) / (ymax - ymin) * (y0 - y1)

    p = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(title)}">']
    for yt in yticks:
        p.append(f'<line class="gl" x1="{x0}" x2="{x1}" y1="{Y(yt):.1f}" y2="{Y(yt):.1f}"/>')
        p.append(f'<text class="tk" x="{x0 - 8}" y="{Y(yt) + 4:.1f}" text-anchor="end">{yt:g}</text>')
    for xt in xticks:
        p.append(f'<line class="gl" x1="{X(xt):.1f}" x2="{X(xt):.1f}" y1="{y0}" y2="{y1}"/>')
        p.append(f'<text class="tk" x="{X(xt):.1f}" y="{y0 + 17}" text-anchor="middle">{xt:g}</text>')
    p.append(f'<line class="ax" x1="{x0}" x2="{x1}" y1="{y0}" y2="{y0}"/>')
    p.append(f'<text class="at" x="{(x0 + x1) / 2:.1f}" y="{h - 6}" text-anchor="middle">{xlabel}</text>')
    js = []
    for s in series:
        pts = [(x, y) for x, y in s["pts"] if y is not None and xmin <= x <= xmax]
        if not pts:
            continue
        dpath = "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in pts)
        dash = ' stroke-dasharray="6 4"' if s.get("dash") else ""
        wid = f' stroke-width="{s["w"]}"' if s.get("w") else ""
        p.append(f'<path class="ln" d="{dpath}" style="stroke:{s["color"]}"{dash}{wid}/>')
        if s.get("markers"):
            for x, y in pts:
                p.append(f'<circle class="mk" cx="{X(x):.1f}" cy="{Y(y):.1f}" r="4" style="fill:{s["color"]}"/>')
        if s.get("label"):
            lx, ly = pts[-1] if not s.get("label_at") else s["label_at"]
            dx, dy = s.get("label_d", (-6, -8))
            p.append(f'<text class="lb" x="{X(lx) + dx:.1f}" y="{Y(ly) + dy:.1f}" text-anchor="end">{s["label"]}</text>')
        if not s.get("nohover"):
            js.append({"name": s["name"], "color": s["color"], "pts": [[round(x, 4), round(y, 4)] for x, y in pts]})
    p.append(f'<line class="xh" x1="0" x2="0" y1="{y1}" y2="{y0}"/>')
    p.append("</svg>")
    leg = "".join(f'<span><span class="key{" dash" if l.get("dash") else ""}" style="border-color:{l["color"]}"></span>'
                  f'{l["name"]}</span>' for l in legend)
    data = {"plot": {"x0": x0, "x1": x1, "xmin": xmin, "xmax": xmax, "log": xlog, "rev": rev},
            "series": js, "xname": xname, "xd": xd, "yd": yd}
    fig = '<figure class="narrow">' if narrow else '<figure style="margin:0">'
    return (f'{fig}<div class="chart"><h4>{title}</h4><div class="unit">{unit}</div>'
            f'<div class="legend">{leg}</div>{"".join(p)}<div class="tip"></div>'
            f'<script type="application/json">{json.dumps(data, separators=(",", ":"))}</script></div>'
            + (f"<figcaption>{caption}</figcaption>" if caption else "") + "</figure>")


def chart_torsion():
    Ns = (0.25, 0.5, 1.0, 2.0, 4.0)
    an = [(N, H.winkler_law(N, 1e7)[1] * 1e3) for N in np.geomspace(0.25, 4.0, 25)]
    series = [
        {"name": "elastic foundation", "color": COL["ref"], "dash": True, "pts": an},
        {"name": "point contact", "color": COL["ref"], "pts": [(N, 0.0) for N in Ns], "w": 1.5,
         "label": "point contact, Drake and MuJoCo: 0", "label_d": (-4, -8)},
        {"name": "MuJoCo condim 4, constant", "color": COL["c4"], "dash": True, "markers": True,
         "pts": [(N, rb("p4", N)) for N in Ns], "label": "condim 4, constant &#956;t", "label_d": (-4, 18)},
        {"name": "Drake hydroelastic 0.01 s", "color": COL["drake"], "markers": True, "pts": [(N, rb("dref", N)) for N in Ns]},
        {"name": "MuJoCo 0.5 mm sphere pads", "color": COL["sphere"], "markers": True, "pts": [(N, rb("s05", N)) for N in Ns]},
    ]
    legend = [{"name": "Drake hydroelastic, 0.01&#8202;s", "color": COL["drake"]},
              {"name": "MuJoCo 0.5&#8202;mm sphere pads", "color": COL["sphere"]},
              {"name": "MuJoCo condim&#160;4, constant &#956;<sub>t</sub>", "color": COL["c4"], "dash": True},
              {"name": "elastic foundation, analytic", "color": COL["ref"], "dash": True}]
    return svg_chart("Friction torque arm per pad", "r&#772; in mm against pinch force per pad, log scale", series, legend,
                     xlog=True, xmin=0.22, xmax=4.4, xticks=(0.25, 0.5, 1, 2, 4), ymin=-0.05, ymax=1.6,
                     yticks=(0, 0.4, 0.8, 1.2, 1.6), xname="N", xd=2, yd=3, xlabel="pinch force per pad, N", narrow=True,
                     caption="Hover for values. The rescheduled condim&#160;4 lies on the Drake curve to within 1&#8202;% and is "
                             "left out of the plot; every model is in the table below.")


def brake_series(key, d=15.0, T=4.0):
    r = brk(key, d, T)
    tr = np.array(r["trace"])
    m = tr[:, 0] >= 0.5
    t = tr[m]
    t = t[::3]
    return [(float(n), float(ph)) for n, ph in zip(t[:, 1], t[:, 2])]


def chart_brake(which):
    qs = [(N, phi_qs(N, 0.015)) for N in np.geomspace(6.0, 0.2, 160)]
    ref = {"name": "quasi-static law", "color": COL["ref"], "dash": True, "pts": qs}
    if which == "A":
        series = [ref,
                  {"name": "Drake hydroelastic 0.01 s", "color": COL["drake"], "pts": brake_series("dref")},
                  {"name": "MuJoCo 0.5 mm sphere pads 0.02 s", "color": COL["sphere"], "pts": brake_series("s05")},
                  {"name": "MuJoCo condim 4 rescheduled", "color": COL["c4"], "dash": True, "pts": brake_series("p4s")}]
        legend = [{"name": "Drake hydroelastic, 0.01&#8202;s", "color": COL["drake"]},
                  {"name": "MuJoCo 0.5&#8202;mm sphere pads, 0.02&#8202;s", "color": COL["sphere"]},
                  {"name": "MuJoCo condim&#160;4, rescheduled", "color": COL["c4"], "dash": True},
                  {"name": "quasi-static law", "color": COL["ref"], "dash": True}]
        title = "Short relaxation: the brake follows the law"
    else:
        series = [ref,
                  {"name": "Drake hydroelastic 0.1 s", "color": COL["drake"], "pts": brake_series("ddef")},
                  {"name": "MuJoCo 0.5 mm sphere pads 0.1 s", "color": COL["sphere"], "pts": brake_series("s05tr01")}]
        legend = [{"name": "Drake hydroelastic, 0.1&#8202;s (default)", "color": COL["drake"]},
                  {"name": "MuJoCo 0.5&#8202;mm sphere pads, 0.1&#8202;s", "color": COL["sphere"]},
                  {"name": "quasi-static law", "color": COL["ref"], "dash": True}]
        title = "0.1&#8202;s relaxation: the brake gives way"
    return svg_chart(title, "&#966; in degrees against pinch force per pad, d&#8202;=&#8202;15&#8202;mm, 4&#8202;s ramp", series, legend,
                     xlog=True, xmin=0.18, xmax=6.6, xticks=(6, 4, 2, 1, 0.5, 0.2), ymin=-5, ymax=130,
                     yticks=(0, 30, 60, 90, 120), rev=True, xname="N", xd=2, yd=1,
                     xlabel="pinch force per pad, N (falls left to right)", w=560, h=380)


# -------------------------------------------------------------------------------- build

def data_uri(path, mime):
    with open(path, "rb") as fh:
        return f"data:{mime};base64," + base64.b64encode(fh.read()).decode()


def main():
    t = open(TPL).read()
    v = {}
    v["BUILT"] = time.strftime("%Y-%m-%d %H:%M")
    v["DRAKE_V"], v["MUJOCO_V"] = DRAKE_V, MUJOCO_V
    v["C_FIT"], v["P_FIT"] = f"{CF * 1e3:.2f}", f"{PF:.2f}"
    Ns = (0.25, 0.5, 1.0, 2.0, 4.0)
    ref = {N: rb("dref", N) for N in Ns}
    errs = lambda k: max(abs(pct(rb(k, N), ref[N])) for N in Ns)  # noqa: E731
    v["ERR_P4S"] = f"{errs('p4s'):.0f}" if errs("p4s") >= 1 else "1"
    v["ERR_SP"] = f"{min(errs('s05'), errs('s1')):.0f}&#8211;{max(errs('s05'), errs('s1')):.0f}"
    v["ERR_S1"], v["ERR_S05"] = f"{errs('s1'):.0f}", f"{errs('s05'):.0f}"
    v["NC_S1"], v["NC_S05"] = str(tor("s1", 1.0)["n_L"]), str(tor("s05", 1.0)["n_L"])
    v["R025_S2"] = f"{rb('s2', 0.25):.2f}"
    v["R1_DRAKE"] = f"{ref[1.0]:.2f}"
    v["R1_WINKLER"] = f"{H.winkler_law(1.0, 1e7)[1] * 1e3:.2f}"
    v["GROW4"] = f"{(4 ** PF - 1) * 100:.0f}"
    v["ERR_P4_025"] = f"{abs(pct(rb('p4', 0.25), ref[0.25])):.0f}"
    v["ERR_P4_4"] = f"{abs(pct(rb('p4', 4.0), ref[4.0])):.0f}"
    v["RT01_W05"], v["RT01_W2"] = f"{rb('ddef', 1.0, 0.5):.2f}", f"{rb('ddef', 1.0, 2.0):.2f}"
    loss = lambda k, N=1.0: abs(pct(rb(k, N, 2.0), rb(k, N, 0.5)))  # noqa: E731
    v["SP01_LOSS"] = f"{loss('s05tr01'):.0f}"
    v["SHORT_CHANGE"] = f"{max(loss('s05'), loss('dref'), loss('d02')):.0f}"
    v["IR10_C4"] = f"{100 * rb('p4ir10', 4.0, 0.5) / rb('p4ir10', 4.0, 2.0):.0f}"
    v["IR10_SP"] = f"{100 * rb('s05ir10', 4.0, 0.5) / rb('s05ir10', 4.0, 2.0):.0f}"
    d15 = 0.015
    v["N45_QS"], v["N80_QS"] = f"{n_qs(45, d15):.2f}", f"{n_qs(80, d15):.2f}"
    for key, tag in (("dref", "REF"), ("s05", "SP"), ("p4s", "P4S"), ("ddef", "DEF"), ("d02", "RT02")):
        r = brk(key)
        v[f"N45_{tag}"], v[f"N80_{tag}"] = f"{r['N_at_45']:.2f}", f"{r['N_at_80']:.2f}"
    v["MAX_DEF"] = f"{brk('ddef')['phi_max']:.0f}"
    v["MAX_SP01"] = f"{brk('s05tr01')['phi_max']:.0f}"
    v["MAX_DPT"], v["MAX_MPT"] = f"{brk('dpt')['phi_max']:.0f}", f"{brk('mp3')['phi_max']:.0f}"
    v["PEAK4_REF"], v["PEAK12_REF"] = f"{brk('dref')['peak_rate_dps']:.0f}", f"{brk('dref', 15.0, 12.0)['peak_rate_dps']:.0f}"
    v["N80_RT02_12"] = f"{brk('d02', 15.0, 12.0)['N_at_80']:.2f}"
    pl = [r for r in PLATE if r["E"] == 1e7]
    v["PLATE_ERR"] = f"{max(abs(r['F_winkler_at_delta'] / r['N'] - 1) for r in pl) * 100:.0f}"
    v["PLATE_RATIO"] = f"{min(r['rbar_over_a'] for r in pl):.3f}&#8211;{max(r['rbar_over_a'] for r in pl):.3f}"
    v["PEAK4_SP"], v["PEAK12_SP"] = f"{brk('s05')['peak_rate_dps']:.0f}", f"{brk('s05', 15.0, 12.0)['peak_rate_dps']:.0f}"
    v["WALL_S1"], v["WALL_P4S"], v["WALL_DRAKE"] = (f"{brk('s1')['wall_s']:.1f}", f"{brk('p4s')['wall_s']:.1f}",
                                                    f"{brk('dref')['wall_s']:.1f}")
    cop = {r["spec"]: r for r in COP}
    v["COP_DRAKE"] = f"{cop[S['dref']]['err_max_mm'] * 1e3:.1f}"
    v["COP_S05"], v["COP_S1"] = f"{cop[S['s05']]['err_max_mm']:.3f}", f"{cop[S['s1']]['err_max_mm']:.2f}"
    v["COP_S2"], v["STEP_S2"] = f"{cop[S['s2']]['err_max_mm']:.2f}", f"{cop[S['s2']]['max_step_mm']:.2f}"
    v["N_TORSION"], v["N_BRAKE"] = str(len(TOR)), str(len(BRK))
    v["TABLE_MODELS"] = table_models()
    v["TABLE_TORSION"] = table_torsion()
    v["TABLE_RATE"] = table_rate()
    v["TABLE_BRAKE"] = table_brake()
    v["TABLE_COP"] = table_cop()
    v["CHART_TORSION"] = chart_torsion()
    v["CHART_BRAKE_A"] = chart_brake("A")
    v["CHART_BRAKE_B"] = chart_brake("B")
    films = json.load(open(os.path.join(D, "films.json")))
    v["FILM"] = data_uri(os.path.join(D, films["file"]), "video/mp4")
    v["POSTER"] = data_uri(os.path.join(D, "media/20261001-brake_eight_models_poster.png"), "image/png")
    v["FILM_PATH"] = "docs/experiments/20261001-hom_contact_patch/" + films["file"]
    for k, val in v.items():
        t = t.replace("{{" + k + "}}", val)
    left = sorted(set(x.split("}}")[0] for x in t.split("{{")[1:]))
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    with open(OUT, "w") as fh:
        fh.write(t)
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
