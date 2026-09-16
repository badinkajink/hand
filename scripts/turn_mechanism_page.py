#!/usr/bin/env python3
"""Build the turn-mechanism page from turn_mechanism.json.

    python3 scripts/turn_mechanism_data.py
    python3 scripts/turn_mechanism_page.py
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os

import numpy as np

ROOT = "docs/experiments/20260916-turn_mechanism"
DATA = f"{ROOT}/turn_mechanism.json"
MEDIA = f"{ROOT}/media"
TPL = "scripts/turn_mechanism_page.template.html"
OUT = f"{ROOT}/20260916-turn_mechanism.html"
W, PAD_L, PAD_R = 1020, 70, 24
HANDS = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"]
PLANT_LABEL = {"shipped": "shipped, kp 30 / kv 0.5", "corrected": "calibrated, kp 0.5 / kv 0.6",
               "fast": "calibrated, kp 0.5 / kv 0.02", "soft": "settled re-fit, kp 0.25 / kv 0.02"}


def uri(name):
    p = os.path.join(MEDIA, name)
    return f"data:{mimetypes.guess_type(p)[0]};base64," + base64.b64encode(open(p, "rb").read()).decode()


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def f(v, fmt="{:.2f}", none="&#8212;"):
    return none if v is None else fmt.format(v)


def table(head, rows, cls=None):
    out = ['<div class="tw"><table><thead><tr>']
    for h in head:
        num = h.startswith("#")
        out.append(f'<th{" class=num" if num else ""}>{h.lstrip("#")}</th>')
    out.append("</tr></thead><tbody>")
    for r in rows:
        klass = ""
        if isinstance(r, tuple) and len(r) == 2 and isinstance(r[1], str) and isinstance(r[0], (list, tuple)):
            r, klass = r
        out.append(f'<tr{" class=" + klass if klass else ""}>')
        for h, c in zip(head, r):
            num = h.startswith("#")
            if isinstance(c, tuple):
                c, cc = c
                out.append(f'<td class="{cc}">{c}</td>')
            else:
                out.append(f'<td{" class=num" if num else ""}>{c}</td>')
        out.append("</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def by_plant(rows, plant):
    return [r for r in rows if r["key"][1] == plant]


def agg(rows, field):
    v = [r[field] for r in rows if r[field] is not None]
    return (float(np.min(v)), float(np.max(v))) if v else (None, None)


def rng(rows, field, fmt="{:.1f}"):
    lo, hi = agg(rows, field)
    if lo is None:
        return "&#8212;"
    return fmt.format(lo) if abs(hi - lo) < 1e-9 else f"{fmt.format(lo)}&#8211;{fmt.format(hi)}"


def table_three(d):
    bp = d["bench_plants"]
    ck = d["chain_kv"]
    hw = d["hw"]
    rows = []
    # bench maneuver
    hwt = [v[1] for v in hw.values()]
    rows.append(("bench maneuver", "hardware, 8 hands, 251 runs", "0.8&#8211;3.4 (regrip band)",
                 f"{min(hwt):.0f}&#8211;{max(hwt):.0f}", "47 of 73 held; 2&#8211;10 of 10 per hand"))
    for p in ("shipped", "corrected", "fast"):
        rs = [r for r in bp if r["plant"] == p]
        t = [r["sim_turn_deg"] for r in rs]
        held = sum(r["sim_held"] for r in rs)
        n = sum(r["n"] for r in rs)
        fe = [r["sim_pad_N_end"] for r in rs]
        rows.append(("bench maneuver", PLANT_LABEL[p],
                     f"{min(fe):.1f}&#8211;{max(fe):.1f} at the end",
                     f"{min(t):+.0f}&#8211;{max(t):+.0f} (sd 4&#8211;58 across seeds)",
                     f"{held} of {n} held"))
    # chain finger turn
    rows.append(("chain finger turn, 2 mm squeeze", "shipped (2026-09-06, D6)", "6.9 on 3 pads",
                 "90", "3 of 4 seeds carried to the gait"))
    for p in ("shipped", "corrected", "fast"):
        rs = by_plant(ck, p)
        rows.append(("chain finger turn, 10 mm squeeze", PLANT_LABEL[p],
                     f"{rng(rs, 'lift_N')} on {rng(rs, 'lift_pads', '{:.0f}')} pads",
                     f"{rng(rs, 'turn_deg', '{:.0f}')}, slide {rng(rs, 'turn_slide')} mm",
                     f"{rng(rs, 'turn_pads', '{:.0f}')} pads at {rng(rs, 'turn_N')} N after the turn"))
    # chain arm turn
    for p in ("shipped", "corrected", "fast"):
        rs = by_plant(ck, p)
        held = sum(r["held_staged"] for r in rs)
        n = sum(r["n"] for r in rs)
        rows.append(("chain arm turn", PLANT_LABEL[p],
                     f"{rng(rs, 'lift_N')} at the lift",
                     "90 (the arm)",
                     f"{held} of {n} held at <code>staged</code>; pads {rng(rs, 'staged_pads', '{:.0f}')} at {rng(rs, 'staged_N')} N"))
    return table(["maneuver", "plant", "#grip force, N", "#turn, deg", "carried"],
                 [(list(r), "hi") if r[1].startswith("hardware") else list(r) for r in rows])


def chart_trace(d):
    tr = d["bench_trace"]
    trk = d["bench_trace_kv"]
    series = [("shipped", tr["shipped"], "var(--ref)"), ("corrected", tr["corrected"], "var(--s1)"),
              ("kv0.01", trk["kv0.01"], "var(--s2)")]
    h = 470
    x0, x1 = PAD_L, W - PAD_R
    tmax = max(s["t"] for _, ss, _ in series for s in ss)
    out = [f'<svg class="chart" viewBox="0 0 {W} {h}" role="img" aria-label="Bench replay trace">']
    # panel 1: cos
    t0, b0 = 34, 214
    out.append(f'<text class="axlab" x="{x0}" y="18">SIGNED COSINE OF THE TOOL AXIS (+1 = TIP DOWN) &#183; rv05_manual_b85 PLAN REPLAY</text>')
    for v in (-0.25, 0, 0.25, 0.5, 0.75, 1.0):
        y = b0 - (v + 0.3) / 1.35 * (b0 - t0)
        out.append(f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end">{v:+.2f}</text>')
    # panel 2: pad force
    t1, b1 = 262, 420
    out.append(f'<text class="axlab" x="{x0}" y="{t1 - 10}">TOTAL PAD FORCE, N &#183; TOOL WEIGHT 0.240 N DASHED</text>')
    fmax = 4.0
    for v in (0, 1, 2, 3, 4):
        y = b1 - v / fmax * (b1 - t1)
        out.append(f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end">{v}</text>')
    yw = b1 - 0.24 / fmax * (b1 - t1)
    out.append(f'<line x1="{x0}" y1="{yw:.1f}" x2="{x1}" y2="{yw:.1f}" stroke="var(--bad)" stroke-dasharray="3 3"/>')
    # phase markers from the corrected trace
    phases = tr["corrected"]
    t_turn0 = next(s["t"] for s in phases if s["phase"] == "turn")
    t_hold0 = next(s["t"] for s in phases if s["phase"] == "hold")
    for tt, lab in ((t_turn0, "turn starts"), (t_hold0, "hold")):
        x = x0 + tt / tmax * (x1 - x0)
        out.append(f'<line x1="{x:.1f}" y1="{t0}" x2="{x:.1f}" y2="{b1}" stroke="var(--rule)" stroke-dasharray="2 4"/>')
        out.append(f'<text class="note" x="{x + 4:.1f}" y="{t0 + 12}">{lab}</text>')
    for v in range(0, int(tmax) + 1):
        x = x0 + v / tmax * (x1 - x0)
        out.append(f'<text class="tick" x="{x:.1f}" y="{b1 + 16}" text-anchor="middle">{v} s</text>')
    for name, ss, col in series:
        pts = " ".join(f"{x0 + s['t'] / tmax * (x1 - x0):.1f},{b0 - (s['cos'] + 0.3) / 1.35 * (b0 - t0):.1f}" for s in ss)
        out.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.2" stroke-linejoin="round"/>')
        pts = " ".join(f"{x0 + s['t'] / tmax * (x1 - x0):.1f},{b1 - min(s['f_pad'], fmax) / fmax * (b1 - t1):.1f}" for s in ss)
        out.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.2" stroke-linejoin="round"/>')
    lx = x0
    for lab, col in (("shipped, kp 30", "var(--ref)"), ("calibrated, kp 0.5 / kv 0.6", "var(--s1)"),
                     ("calibrated, kp 0.5 / kv 0.01", "var(--s2)")):
        out.append(f'<rect x="{lx}" y="{h - 14}" width="14" height="3" fill="{col}"/>')
        out.append(f'<text class="note" x="{lx + 20}" y="{h - 9}">{lab}</text>')
        lx += 34 + len(lab) * 6.9
    return "\n".join(out) + "</svg>"


def table_mech(d):
    rows = []
    for r in d["bench_mechanism"]:
        if r["plant"] not in ("shipped", "corrected"):
            continue
        rows.append((PLANT_LABEL[r["plant"]], "kept" if r["post"] else "removed after the grip",
                     "on" if r["gravity"] else "off",
                     f"{r['f_post_max_turn']:.2f}", f"{r['cos_grip']:+.3f}", f"{r['cos_end']:+.3f}",
                     f"{r['turn_deg']:+.1f}", f"{r['f_pad_grip']:.2f} &#8594; {r['f_pad_end']:.2f} / {r['n_pad_end']}",
                     '<span class="ok">held</span>' if not r["dropped"] else '<span class="no">dropped</span>'))
    return table(["plant", "post", "gravity", "#post force, N (max)", "#cos at grip", "#cos at end",
                  "#turn, deg", "#pads N grip &#8594; end / n", "outcome"], rows)


def table_chainkv(d):
    ck = d["chain_kv"]
    rows = []
    for hand in ("D3", "D5", "D6", "D7"):
        for p in ("shipped", "corrected", "fast"):
            rs = [r for r in ck if r["key"][0] == hand and r["key"][1] == p]
            held = sum(r["held_staged"] for r in rs)
            n = sum(r["n"] for r in rs)
            drops = sorted({x for r in rs for x in r["drop"]})
            rows.append(([hand, PLANT_LABEL[p],
                          f"{rng(rs, 'lift_N')} / {rng(rs, 'lift_pads', '{:.0f}')}",
                          rng(rs, "turn_deg", "{:+.0f}"),
                          f"{rng(rs, 'turn_N')} / {rng(rs, 'turn_pads', '{:.0f}')}",
                          rng(rs, "turn_slide"),
                          rng(rs, "q_err_max", "{:.0f}"),
                          rng(rs, "staged_cos", "{:+.2f}"),
                          f"{rng(rs, 'staged_N')} / {rng(rs, 'staged_pads', '{:.0f}')}",
                          f"{held} / {n}", ", ".join(drops)],
                         "hi" if p == "fast" else ""))
    return table(["hand", "plant", "#lift N / pads", "#finger turn, deg", "#turn N / pads", "#slide, mm",
                  "#max q_err, deg", "#staged cos", "#staged N / pads", "#held at staged", "drop stage"], rows)


def table_matrix(d):
    c16 = {(r["key"][0], r["key"][1]): r for r in d["chain16"]}
    a0 = {r["key"][0]: r for r in d["chain_angle0"]}
    rel = {(r["key"][0], r["key"][1]): r for r in d["chain_relief"]}
    pt = {r["key"][0]: r for r in d["chain_planturn"]}
    soft = {r["key"][0]: r for r in d["chain_soft"]}
    hw = d["hw"]

    def cell(r):
        if r is None:
            return ("&#8212;", "cell")
        h, n = r["held_staged"], r["n"]
        frac = h / n if n else 0
        cls = "c4" if frac >= 0.99 else "c3" if frac >= 0.6 else "c2" if frac >= 0.4 else "c1" if frac > 0 else "c0"
        cos = "" if r["staged_cos"] is None or h == 0 else f'<br><span class="dim">{r["staged_cos"]:+.2f} / {r["staged_N"]:.1f} N</span>'
        return (f"{h}/{n}{cos}", f"cell {cls}")

    rows = []
    for hd in HANDS:
        rows.append((hd, hw[hd][0], cell(c16.get((hd, 0.15))), cell(c16.get((hd, 0.25))), cell(a0.get(hd)),
                     cell(rel.get((hd, 0.004))), cell(rel.get((hd, 0.008))), cell(pt.get(hd)), cell(soft.get(hd))))
    head = ["hand", "bench hold", "finger turn, pivot 0.15", "pivot 0.25", "arm only (angle 0)",
            "relief 4 mm", "relief 8 mm", "plan&#8217;s own turn", "kp 0.25 plant"]
    out = table(head, rows)
    setb = [r for r in d["chain16"] if r["key"][0] not in HANDS]
    b = {}
    for r in setb:
        b.setdefault(r["key"][0], []).append(r)
    brow = []
    for tag, rs in sorted(b.items()):
        held = sum(r["held_staged"] for r in rs)
        n = sum(r["n"] for r in rs)
        brow.append((f"<code>{tag}</code>", "not built", f"{held} / {n}", rng(rs, "staged_cos", "{:+.2f}"),
                     rng(rs, "staged_N"), rng(rs, "lift_N")))
    out += "\n" + table(["set-B hand", "bench", "#held at staged (both pivots)", "#staged cos", "#staged N", "#lift N"], brow)
    return out


def table_hwlag(d):
    rows = []
    for run in d["hw_lag"]:
        first = True
        for j in run["joints"]:
            rows.append((run["label"] if first else "", f"{run['step_s']:.2f} s / {run['total_s']:.0f} s" if first else "",
                         f"<code>{j['joint']}</code>", f"{j['travel']:+.1f}", f"{j['rate']:+.1f}",
                         f"{j['err_med']:+.1f}", f"{j['err_last']:+.1f}", f"{j['load_med']:.0f}"))
            first = False
    return table(["run (g12, 2026-08-29)", "step / total", "joint", "#travel, deg", "#rate, deg/s",
                  "#achieved &#8722; commanded, median", "#at the last step", "#yaw load, median"], rows)


def table_kv(d):
    rows = []
    base = [r for r in d["bench_mechanism"] if r["plant"] == "corrected" and r["post"] and r["gravity"]][0]
    rows.append(("0.6 (default)", "1.2 s", f"{base['turn_deg']:+.1f}", f"{base['cos_end']:+.3f}", f"{base['f_pad_end']:.2f}"))
    for r in d["bench_kv"]:
        kv = float(r["plant"][2:])
        rows.append((f"{kv:g}", f"{kv / 0.5:.2f} s", f"{r['turn_deg']:+.1f}", f"{r['cos_end']:+.3f}", f"{r['f_pad_end']:.2f}"))
    ck = d["chain_kv"]
    for p, lab in (("corrected", "0.6"), ("fast", "0.02")):
        rs = by_plant(ck, p)
        rows.append((f"{lab} (chain, 4 hands)", f"{float(lab) / 0.5:.2f} s", rng(rs, "turn_deg", "{:+.0f}"),
                     rng(rs, "staged_cos", "{:+.2f}"), rng(rs, "turn_N")))
    return table(["kv at kp 0.5", "#time constant", "#turn, deg", "#cos at end / staged", "#pad N"], rows)


def table_kpfit(d):
    old = {r["kp"]: r for r in d["kp_fit_kv0.6"]["rows"]}
    new = {r["kp"]: r for r in d["kp_refit"]["rows"]}
    meas = d["kp_refit"]["measured"]
    rows = [("bench (n = 3)", "", f"{meas['middle_yaw']:+.2f}", f"{meas['thumb_mcp']:+.2f}", f"{meas['thumb_yaw']:+.2f}", f"{meas['index_yaw']:+.2f}", "")]
    for kp in sorted(set(old) | set(new), reverse=True):
        for lab, src in (("kv 0.6", old), ("kv 0.02", new)):
            r = src.get(kp)
            if r is None:
                continue
            s = r["sim"]
            rows.append(([f"{kp:g}", lab, f"{s['middle_yaw']:+.2f}", f"{s['thumb_mcp']:+.2f}", f"{s['thumb_yaw']:+.2f}",
                          f"{s['index_yaw']:+.2f}", f"{r['mae']:.2f}"],
                         "hi" if (lab == "kv 0.6" and kp == 0.5) or (lab == "kv 0.02" and kp == 0.25) else ""))
    return table(["kp", "kv", "#middle_yaw", "#thumb_mcp", "#thumb_yaw", "#index_yaw", "#MAE, deg"], rows)


def chart_bench(d):
    bp = d["bench_plants"]
    hw = d["hw"]
    h = 360
    x0, x1, t, b = PAD_L + 20, W - PAD_R - 20, 40, h - 76
    lo, hi = -35, 75
    out = [f'<svg class="chart" viewBox="0 0 {W} {h}" role="img" aria-label="Bench turn per hand per plant">']
    out.append(f'<text class="axlab" x="{x0}" y="18">NET TURN ON THE BENCH MANEUVER, DEG &#183; BENCH (HELD TRIALS) AGAINST EACH PLANT&#8217;S 4-SEED MEAN &#177; SD</text>')
    for v in range(-20, 80, 20):
        y = b - (v - lo) / (hi - lo) * (b - t)
        out.append(f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end">{v}</text>')
    y0 = b - (0 - lo) / (hi - lo) * (b - t)
    out.append(f'<line x1="{x0}" y1="{y0:.1f}" x2="{x1}" y2="{y0:.1f}" stroke="var(--rule)"/>')
    cols = {"shipped": "var(--ref)", "corrected": "var(--s1)", "fast": "var(--s2)"}
    n = len(HANDS)
    for i, hd in enumerate(HANDS):
        xc = x0 + (i + 0.5) / n * (x1 - x0)
        out.append(f'<text class="ser" x="{xc:.1f}" y="{b + 18}" text-anchor="middle" fill="var(--ink)">{hd}</text>')
        out.append(f'<text class="note" x="{xc:.1f}" y="{b + 32}" text-anchor="middle">{hw[hd][0]}</text>')
        yh = b - (hw[hd][1] - lo) / (hi - lo) * (b - t)
        out.append(f'<line x1="{xc - 26:.1f}" y1="{yh:.1f}" x2="{xc + 26:.1f}" y2="{yh:.1f}" stroke="var(--ink)" stroke-width="2.5"/>')
        for j, p in enumerate(("shipped", "corrected", "fast")):
            r = next(r for r in bp if r["hand"] == hd and r["plant"] == p)
            x = xc + (j - 1) * 14
            ym = b - (r["sim_turn_deg"] - lo) / (hi - lo) * (b - t)
            ya = b - (min(hi, r["sim_turn_deg"] + r["sim_turn_sd"]) - lo) / (hi - lo) * (b - t)
            yb = b - (max(lo, r["sim_turn_deg"] - r["sim_turn_sd"]) - lo) / (hi - lo) * (b - t)
            out.append(f'<line x1="{x:.1f}" y1="{ya:.1f}" x2="{x:.1f}" y2="{yb:.1f}" stroke="{cols[p]}" stroke-width="1.5" opacity=".7"/>')
            out.append(f'<circle cx="{x:.1f}" cy="{ym:.1f}" r="4.5" fill="{cols[p]}" stroke="var(--card)" stroke-width="1.5"/>')
    lx = x0
    out.append(f'<line x1="{lx}" y1="{h - 10}" x2="{lx + 14}" y2="{h - 10}" stroke="var(--ink)" stroke-width="2.5"/>')
    out.append(f'<text class="note" x="{lx + 20}" y="{h - 6}">bench, net turn on held trials</text>')
    lx += 240
    for lab, p in (("shipped", "shipped"), ("calibrated kv 0.6", "corrected"), ("calibrated kv 0.02", "fast")):
        out.append(f'<circle cx="{lx + 5}" cy="{h - 10}" r="4.5" fill="{cols[p]}"/>')
        out.append(f'<text class="note" x="{lx + 16}" y="{h - 6}">{lab}</text>')
        lx += 30 + len(lab) * 6.9
    return "\n".join(out) + "</svg>"


def table_benchplants(d):
    bp = d["bench_plants"]
    hw = d["hw"]
    rows = []
    for hd in HANDS:
        rs = {r["plant"]: r for r in bp if r["hand"] == hd}
        tag = rs["shipped"]["tag"]
        cells = [hd, f"<code>{tag}</code>", hw[hd][0], f"{hw[hd][1]:.1f}"]
        for p in ("shipped", "corrected", "fast"):
            r = rs[p]
            cells.append(f"{r['sim_turn_deg']:+.1f} &#177; {r['sim_turn_sd']:.0f} &#183; {r['sim_held']}/{r['n']}")
        rows.append(cells)
    return table(["hand", "plan", "bench hold", "#bench turn, deg", "#shipped: turn &#183; held", "#kv 0.6", "#kv 0.02"], rows)


def rho(d, plant):
    from scipy.stats import spearmanr
    hw = [d["hw"][h][1] for h in HANDS]
    st = [next(r["sim_turn_deg"] for r in d["bench_plants"] if r["hand"] == h and r["plant"] == plant) for h in HANDS]
    return f"{spearmanr(hw, st).correlation:+.2f}"


def main():
    d = json.load(open(DATA))
    html = open(TPL).read()
    subs = {
        "N_CHAIN": str(d["n_rollouts"]["chain"]), "N_BENCH": str(d["n_rollouts"]["bench"]),
        "TABLE_THREE": table_three(d), "CHART_TRACE": chart_trace(d), "TABLE_MECH": table_mech(d),
        "TABLE_CHAINKV": table_chainkv(d), "TABLE_MATRIX": table_matrix(d), "TABLE_HWLAG": table_hwlag(d),
        "TABLE_KV": table_kv(d), "TABLE_KPFIT": table_kpfit(d), "CHART_BENCH": chart_bench(d),
        "TABLE_BENCHPLANTS": table_benchplants(d),
        "RHO_SHIPPED": rho(d, "shipped"), "RHO_CORRECTED": rho(d, "corrected"), "RHO_FAST": rho(d, "fast"),
        "V_BENCH_SHIPPED": uri("bench_d7_shipped.mp4"), "V_BENCH_CORRECTED": uri("bench_d7_corrected.mp4"),
        "I_BENCH_SHIPPED": uri("bench_d7_shipped_strip.png"), "I_BENCH_CORRECTED": uri("bench_d7_corrected_strip.png"),
        "V_D7_SHIPPED": uri("chain_d7_shipped.mp4"), "V_D6_SHIPPED": uri("chain_d6_shipped.mp4"),
        "I_D6_SHIPPED": uri("chain_d6_shipped_seams.png"),
        "V_D6_FAST": uri("chain_d6_fast.mp4"), "I_D6_FAST": uri("chain_d6_fast_seams.png"),
    }
    for k, v in subs.items():
        html = html.replace("{{" + k + "}}", v)
    left = [w for w in ("{{" + k + "}}" for k in subs) if w in html]
    import re
    missing = re.findall(r"\{\{[A-Z_]+\}\}", html)
    if missing:
        raise SystemExit(f"unfilled placeholders: {sorted(set(missing))}")
    open(OUT, "w").write(html)
    print(f"wrote {OUT}  {os.path.getsize(OUT) / 1048576:.1f} MB")


if __name__ == "__main__":
    main()
