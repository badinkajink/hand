#!/usr/bin/env python3
"""Build the pinch-and-swing page from swing_reorient.json.

    python3 scripts/swing_reorient_data.py
    python3 scripts/swing_reorient_page.py
"""
from __future__ import annotations

import base64
import collections
import json
import mimetypes
import os
import re

import numpy as np

ROOT = "docs/experiments/20260916-swing_reorient"
DATA = f"{ROOT}/swing_reorient.json"
MEDIA = f"{ROOT}/media"
TPL = "scripts/swing_reorient_page.template.html"
OUT = f"{ROOT}/20260916-swing_reorient.html"
W, PAD_L, PAD_R = 1020, 70, 24
HANDS = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"]


def uri(name):
    p = os.path.join(MEDIA, name)
    return f"data:{mimetypes.guess_type(p)[0]};base64," + base64.b64encode(open(p, "rb").read()).decode()


def table(head, rows):
    out = ['<div class="tw"><table><thead><tr>']
    for h in head:
        out.append(f'<th{" class=num" if h.startswith("#") else ""}>{h.lstrip("#")}</th>')
    out.append("</tr></thead><tbody>")
    for r in rows:
        klass = ""
        if isinstance(r, tuple) and len(r) == 2 and isinstance(r[1], str) and isinstance(r[0], list):
            r, klass = r
        out.append(f'<tr{" class=" + klass if klass else ""}>')
        for h, c in zip(head, r):
            if isinstance(c, tuple):
                out.append(f'<td class="{c[1]}">{c[0]}</td>')
            else:
                out.append(f'<td{" class=num" if h.startswith("#") else ""}>{c}</td>')
        out.append("</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def cellc(h, n, extra=""):
    frac = h / n if n else 0
    cls = "c4" if frac >= 0.99 else "c3" if frac >= 0.6 else "c2" if frac >= 0.4 else "c1" if frac > 0 else "c0"
    return (f"{h}/{n}{extra}", f"cell {cls}")


def chart_trace(d):
    """cos and yaw command/achieved over the turn, D6 default plant vs elliptic."""
    tr = d["trace_default"]
    te = d["trace_elliptic"]
    h = 480
    x0, x1 = PAD_L, W - PAD_R
    umin, umax = -0.05, 1.55
    out = [f'<svg class="chart" viewBox="0 0 {W} {h}" role="img" aria-label="D6 turn record">']
    # panel 1: yaw cmd vs achieved (default plant)
    t0, b0 = 34, 214
    out.append(f'<text class="axlab" x="{x0}" y="18">YAW JOINTS, DEG: COMMANDED (DASHED) AND ACHIEVED (SOLID) &#183; D6, kp 0.5 / kv 0.02, TEMPLATE CONTACT &#183; u = FRACTION OF THE TURN</text>')
    lo, hi = -35, 55
    for v in range(-30, 60, 15):
        y = b0 - (v - lo) / (hi - lo) * (b0 - t0)
        out.append(f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end">{v}</text>')
    cols = {"thumb": "var(--s1)", "index": "var(--s2)", "middle": "var(--s3)"}
    X = lambda u: x0 + (u - umin) / (umax - umin) * (x1 - x0)
    for f, col in cols.items():
        for key, dash in (("yaw_cmd", "4 3"), ("yaw", None)):
            pts = " ".join(f"{X(s['u']):.1f},{b0 - (s[key][f] - lo) / (hi - lo) * (b0 - t0):.1f}" for s in tr)
            out.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"' + (f' stroke-dasharray="{dash}"' if dash else "") + '/>')
    # panel 2: cos, both contact models
    t1, b1 = 262, 420
    out.append(f'<text class="axlab" x="{x0}" y="{t1 - 10}">SIGNED COSINE OF THE TOOL AXIS &#183; TEMPLATE CONTACT (GREY) AND ELLIPTIC / IMPRATIO 10 (BLACK)</text>')
    clo, chi = -0.3, 1.05
    for v in (-0.25, 0, 0.25, 0.5, 0.75, 1.0):
        y = b1 - (v - clo) / (chi - clo) * (b1 - t1)
        out.append(f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end">{v:+.2f}</text>')
    for ss, col in ((tr, "var(--ref)"), (te, "var(--ink)")):
        pts = " ".join(f"{X(s['u']):.1f},{b1 - (max(clo, min(chi, s['cos'])) - clo) / (chi - clo) * (b1 - t1):.1f}" for s in ss)
        out.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.2"/>')
    for u in (0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5):
        out.append(f'<text class="tick" x="{X(u):.1f}" y="{b1 + 16}" text-anchor="middle">{u:g}</text>')
    xu = X(1.0)
    out.append(f'<line x1="{xu:.1f}" y1="{t0}" x2="{xu:.1f}" y2="{b1}" stroke="var(--rule)" stroke-dasharray="2 4"/>')
    out.append(f'<text class="note" x="{xu + 4:.1f}" y="{b0 - 6}">last turn command</text>')
    lx = x0
    for lab, col in (("thumb", "var(--s1)"), ("index", "var(--s2)"), ("middle", "var(--s3)")):
        out.append(f'<rect x="{lx}" y="{h - 14}" width="14" height="3" fill="{col}"/>')
        out.append(f'<text class="note" x="{lx + 20}" y="{h - 9}">{lab} yaw</text>')
        lx += 100
    out.append(f'<text class="note" x="{lx + 10}" y="{h - 9}">elliptic run: the index pad leaves the shaft at u = 0.91 and the tool falls (cos rises as it drops)</text>')
    return "\n".join(out) + "</svg>"


def table_probes(d):
    P = d["probes"]
    rows = []
    for name, lab in (("sv1_u0308_b050__fast__sq10_k0.15_b0.5_a-90", "chain turn, budget 0.5, template contact"),
                      ("sv1_u0308_b050__fast__sq10_k0.15_b1.3_a-90", "budget 1.3, template contact"),
                      ("sv1_u0308_b050__fast__sq10_k0.15_b0.5_a-90_elliptic_ir10", "budget 0.5, elliptic / impratio 10"),
                      ("sv1_u0308_b050__fast__sq10_k0.15_b1.3_a-90_elliptic_ir10", "budget 1.3, elliptic / impratio 10")):
        p = P[name]
        a = p["analysis"]
        fs = a["fingers"]
        yaw = " / ".join(f"{fs[f]['dq_deg'][f + '_yaw']:+.1f} of {fs[f]['dq_cmd_deg'][f + '_yaw']:+.1f}" for f in ("thumb", "index", "middle"))
        slip = " / ".join(f"{fs[f]['slip_mean_mm_s']:.0f}" for f in ("thumb", "index", "middle"))
        N = " / ".join(f"{fs[f]['pad_n1']:.2f}" for f in ("thumb", "index", "middle"))
        rows.append([lab, f"{a['turn_deg']:+.1f}", f"{p['carry_ik_mm']:.0f}", yaw, slip, N, str(p["drop"])])
    return table(["cell (D6, kp 0.5 / kv 0.02)", "#tool turn, deg", "#IK residual, mm", "#yaw achieved of commanded, deg (th / ix / md)",
                  "#pad creep, mm/s", "#pad N at the end", "dropped at"], rows)


def table_feas(d):
    F = d["feasibility_d6"]
    angles = ["-10", "-20", "-30", "-40", "-50", "-60", "-75", "-90"]
    angles = [a for a in angles if a in next(iter(next(iter(F["cells"].values())).values()))]
    rows = []
    for pv, lab in (("centroid", "pivot between the pads"), ("thumb", "pivot at the thumb pad")):
        for k in ("0.0", "0.15", "0.3", "0.5", "1.0"):
            if k not in F["cells"][pv]:
                continue
            cells = [lab if k == "0.0" else "", k]
            for a in angles:
                r = F["cells"][pv][k][a]
                mx = max(r.values())
                worst = {"thumb": "th", "index": "ix", "middle": "md"}[max(r, key=r.get)]
                cells.append((f"{mx:.0f} <span class=dim>{worst}</span>", "cell " + ("c4" if mx <= 3 else "c2" if mx <= 8 else "c0")))
            rows.append(cells)
    return table(["pivot", "height, half-straddles"] + [f"{abs(int(a))}&#176;" for a in angles], rows)


def table_workspace(d):
    ws = d["workspace_d6"]
    rows = []
    for r in ws:
        if not r["fit"]:
            continue
        rows.append([f"{r['depth_mm']:g}", f"{r['elevation_deg']:g}", f"{r['straddle_mm']:g}", f"{r['depth_fit_mm']:.1f}",
                     f"{r['q_grip_deg']['index_mcp']:.0f} / {r['q_grip_deg']['middle_mcp']:.0f}",
                     (f"{r['reach_deg']}&#176;", "cell " + ("c3" if r["reach_deg"] >= 20 else "c2" if r["reach_deg"] else "c0"))])
    nofit = sum(1 for r in ws if not r["fit"])
    t = table(["#asked depth, mm", "#elevation, deg", "#straddle, mm", "#fitted depth", "#index / middle mcp at grip, deg", "reachable rotation (residual &#8804; 3 mm)"], rows)
    return t + f'<p class="dim">{nofit} of {len(ws)} grasp cells have no fit: every straddle but 40 mm, every depth under 62.5 mm at elevation 0, and every elevation past &#8722;15&#176; at the plans&#8217; depth.</p>'


def table_d6pinch(d):
    P = d["probes"]
    rows = []
    for name, lab in (("d6_pinch_middle", "release middle 20 mm, template contact"),
                      ("d6_pinch_index", "release index 20 mm, template contact"),
                      ("d6_pinch_middle_mu1.0", "release middle, elliptic / 10, &#956; 1.0"),
                      ("d6_pinch_middle_mu0.6", "release middle, elliptic / 10, &#956; 0.6"),
                      ("d6_turn_mu1.0", "chain turn (&#8722;90&#176;), elliptic / 10, &#956; 1.0"),
                      ("d6_turn_mu0.6", "chain turn, elliptic / 10, &#956; 0.6")):
        p = P[name]
        s = p["seams"]
        cells = [lab]
        for ph in ("lifted", "turned", "reoriented", "staged"):
            v = s.get(ph, {})
            cells.append(f"{v.get('cos', 0):+.2f} &#183; {v.get('pad_contacts', 0)}p &#183; {v.get('pad_force_N', 0):.2f} N")
        cells.append(str(p["drop"]))
        rows.append(cells)
    return table(["D6, kp 0.5 / kv 0.02, heading 0 (the swing goes handle-down)", "#lifted", "#turned", "#reoriented", "#staged", "dropped at"], rows)


def agg_swing(d, flt):
    agg = collections.OrderedDict()
    for r in d["swing"]:
        if not flt(r):
            continue
        k = (r["hand"], r["plant"], r["squeeze"], r["thumb"], r["pinch"], r["yaw"])
        a = agg.setdefault(k, {"n": 0, "lift": 0, "swung": 0, "swung_held": 0, "regrip": 0, "staged": 0, "carried": 0, "N": [], "cos": []})
        a["n"] += 1
        a["lift"] += r["lift_held"]
        sw = (r["reor_cos"] or 0) > 0.85
        a["swung"] += sw
        a["swung_held"] += sw and r["reor_held"]
        a["regrip"] += (r["regrip_cos"] or 0) > 0.85 and r["regrip_held"]
        a["staged"] += (r["staged_cos"] or 0) > 0.85 and r["staged_held"]
        a["carried"] += r["carried"]
        if r["reor_held"]:
            a["N"].append(r["reor_N"]); a["cos"].append(r["reor_cos"])
    return agg


def table_hands(d):
    agg = agg_swing(d, lambda r: r["yaw"] == 180 and r["squeeze"] == 10 and r["plant"] == "cal"
                    and r["pinch"] in ("m20", "m20t-4i-4", "m20t-8i-8") and not r["sweep"].endswith("swing_seeds"))
    # columns: (thumb, pinch)
    cols = [(10, "m20"), (10, "m20t-4i-4"), (10, "m20t-8i-8"), (20, "m20"), (20, "m20t-4i-4"), (30, "m20"), (30, "m20t-4i-4")]
    rows = []
    for hd in HANDS:
        cells = [hd, d["hw"][hd]]
        for ta, pn in cols:
            a = agg.get((hd, "cal", 10, ta, pn, 180.0))
            if a is None:
                cells.append(("&#8212;", "cell"))
                continue
            extra = ""
            if a["cos"]:
                extra = f'<br><span class="dim">{np.median(a["cos"]):+.2f} / {np.median(a["N"]):.1f} N</span>'
            if a["carried"]:
                extra += f'<br><span class="dim">carried {a["carried"]}</span>'
            cells.append(cellc(a["swung_held"] + (0 if a["swung_held"] else 0), a["n"], extra) if True else None)
            # show "regrip" success when the swing was partial
            if a["swung_held"] == 0 and a["regrip"]:
                cells[-1] = (f"0/{a['n']} <span class=dim>swing</span><br>{a['regrip']}/{a['n']} <span class=dim>after regrip</span>{extra}", "cell c2")
        rows.append(cells)
    head = ["hand", "bench hold", "thumb 10, release", "thumb 10, +4 squeeze", "thumb 10, +8", "thumb 20, release", "thumb 20, +4", "thumb 30, release", "thumb 30, +4"]
    return table(head, rows) + '<p class="dim">Cells: swings held at <code>reoriented</code> (cos &gt; 0.85 on two pads at &#8805; 0.240 N) over rollouts, with the median cosine and pad force; &#8220;carried&#8221; counts rollouts that never dropped the tool through the whole chain. The plans&#8217; grasp is thumb 10.</p>'


def table_seeds(d):
    agg = agg_swing(d, lambda r: r["sweep"].split("-")[-1] in ("swing_seeds", "swing_cal25", "swing_pinch") and r["yaw"] == 180 and r["thumb"] == 10)
    rows = []
    for (hd, pl, sq, ta, pn, yw), a in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], str(kv[0][4]))):
        if hd not in ("D3", "D6", "D8", "D2", "D7") or pn not in ("m20", "m20t-4i-4"):
            continue
        cr = f"{min(a['cos']):+.2f}&#8211;{max(a['cos']):+.2f} / {min(a['N']):.1f}&#8211;{max(a['N']):.1f}" if a["cos"] else "&#8212;"
        rows.append([hd, "kp 0.5" if pl == "cal" else "kp 0.25", f"{sq}", pn.replace("m20", "release").replace("t-4i-4", " +4 mm"),
                     f"{a['n']}", f"{a['lift']}", cellc(a["swung_held"], a["n"]), f"{a['regrip']}", f"{a['staged']}", f"{a['carried']}", cr])
    return table(["hand", "plant", "#squeeze, mm", "pinch arm", "#n", "#lifted", "swung, held", "#upright after regrip", "#staged held", "#carried", "#reoriented cos / N (held)"], rows)


def table_sens(d):
    S = d["sens"]
    rows = []
    for k in ("mu0.6", "mu0.8", "mu1.0", "mu1.2", "mu1.5", "tor0.001", "tor0.003", "tor0.01"):
        if k not in S:
            continue
        s = S[k]
        lab = ("pad &#956; " + k[2:]) if k.startswith("mu") else ("torsional " + k[3:] + " (&#956; 1.0)")
        cells = [lab]
        for ph in ("lifted", "turned", "reoriented", "staged"):
            v = s.get(ph, {})
            cells.append(f"{v.get('cos', 0):+.2f} &#183; {v.get('pad_contacts', 0)}p &#183; {v.get('pad_force_N', 0):.2f} N")
        cells.append(str(s["drop"]))
        rows.append(cells)
    return table(["D8, 12 mm squeeze, release middle +4 mm pinch", "#lifted", "#turned", "#reoriented", "#staged", "dropped at"], rows)


def main():
    d = json.load(open(DATA))
    html = open(TPL).read()
    subs = {
        "N_SWING": str(d["n_rollouts"]["swing"]), "N_PROBE": str(d["n_rollouts"]["probes"]),
        "CHART_TRACE": chart_trace(d), "TABLE_PROBES": table_probes(d), "TABLE_FEAS": table_feas(d),
        "TABLE_WORKSPACE": table_workspace(d), "TABLE_D6PINCH": table_d6pinch(d), "TABLE_HANDS": table_hands(d),
        "TABLE_SEEDS": table_seeds(d), "TABLE_SENS": table_sens(d),
        "I_D6_STRIP": uri("d6_swing_strip.png"), "V_D6": uri("d6_swing.mp4"), "V_D8": uri("d8_swing.mp4"),
        "I_D2_SEAMS": uri("d2_swing_seams.png"),
    }
    for k, v in subs.items():
        html = html.replace("{{" + k + "}}", v)
    missing = re.findall(r"\{\{[A-Z_0-9]+\}\}", html)
    if missing:
        raise SystemExit(f"unfilled placeholders: {sorted(set(missing))}")
    open(OUT, "w").write(html)
    print(f"wrote {OUT}  {os.path.getsize(OUT) / 1048576:.1f} MB")


if __name__ == "__main__":
    main()
