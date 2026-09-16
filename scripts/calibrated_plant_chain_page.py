#!/usr/bin/env python3
"""Build the calibrated-plant chain page from calibrated_plant.json.

    python3 scripts/calibrated_plant_chain_data.py
    python3 scripts/calibrated_plant_chain_page.py
"""
from __future__ import annotations
import base64, json, mimetypes, os

ROOT = "docs/experiments/20260916-calibrated_plant_chain"
DATA = f"{ROOT}/calibrated_plant.json"
MEDIA = f"{ROOT}/media"
TPL = "scripts/calibrated_plant_chain_page.template.html"
OUT = f"{ROOT}/20260916-calibrated_plant_chain.html"
W, PAD_L, PAD_R = 1020, 132, 24


def uri(name):
    p = os.path.join(MEDIA, name)
    return f"data:{mimetypes.guess_type(p)[0]};base64," + base64.b64encode(open(p, "rb").read()).decode()


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def chart_lift(d):
    """Pad force at the lift, shipped against corrected, at the chain's 2 mm squeeze."""
    rows = d["compare"]
    bh, sub, gap = 12, 3, 14
    grp = bh * 2 + sub
    h = 60 + len(rows) * (grp + gap) + 40
    x0, x1 = PAD_L, W - PAD_R - 90
    mx = max(max(r["shipped"]["lift_N"] or 0, r["corrected"]["lift_N"] or 0) for r in rows)
    out = [f'<svg class="chart" viewBox="0 0 {W} {h}" role="img" aria-label="Pad force at the lift on both plants">']
    out.append(f'<text class="axlab" x="{x0}" y="18">PAD FORCE AT THE LIFT, N &#183; THE CHAIN&#8217;S 2 MM SQUEEZE &#183; TOOL WEIGHT 0.240 N</text>')
    for v in (0, 5, 10, 15, 20, 25):
        if v > mx:
            break
        x = x0 + v / mx * (x1 - x0)
        out.append(f'<line class="grid" x1="{x:.1f}" y1="30" x2="{x:.1f}" y2="{h - 40}"/>')
        out.append(f'<text class="tick" x="{x:.1f}" y="{h - 26}" text-anchor="middle">{v}</text>')
    xw = x0 + 0.240 / mx * (x1 - x0)
    out.append(f'<line x1="{xw:.1f}" y1="30" x2="{xw:.1f}" y2="{h - 40}" stroke="var(--bad)" stroke-dasharray="3 3"/>')
    for i, r in enumerate(rows):
        y = 38 + i * (grp + gap)
        out.append(f'<text class="ser" x="{x0 - 10}" y="{y + grp / 2 + 4}" text-anchor="end" fill="var(--ink)">{r["id"]}</text>')
        for j, (p, col) in enumerate((("shipped", "var(--ref)"), ("corrected", "var(--s1)"))):
            v = r[p]["lift_N"] or 0
            yy = y + j * (bh + sub)
            out.append(f'<rect x="{x0}" y="{yy}" width="{max(v / mx * (x1 - x0), 1.5):.1f}" height="{bh}" rx="3" fill="{col}"/>')
            out.append(f'<text class="val" x="{x0 + v / mx * (x1 - x0) + 8:.1f}" y="{yy + bh - 1}">{v:.2f}</text>')
    lx = x0
    for lab, col in (("shipped plant, kp 30", "var(--ref)"), ("calibrated plant, kp 0.5", "var(--s1)")):
        out.append(f'<rect x="{lx}" y="{h - 12}" width="9" height="9" rx="2" fill="{col}"/>')
        out.append(f'<text class="note" x="{lx + 14}" y="{h - 4}">{lab}</text>')
        lx += 28 + len(lab) * 6.9
    out.append(f'<text class="note" x="{lx + 10}" y="{h - 4}" fill="var(--bad)">dashed: the tool&#8217;s own weight</text>')
    return "\n".join(out) + "</svg>"


def chart_squeeze(d):
    """Lift force against squeeze on the calibrated plant, one line per hand."""
    sq = d["squeeze"]
    vals, rows = sq["values"], sq["rows"]
    h = 300
    x0, x1, t, b = PAD_L, W - PAD_R - 20, 40, h - 60
    mx = max(v for r in rows for v in r["lift_N"] if v is not None)
    out = [f'<svg class="chart" viewBox="0 0 {W} {h}" role="img" aria-label="Lift force against squeeze">']
    out.append(f'<text class="axlab" x="{x0}" y="18">PAD FORCE AT THE LIFT ON THE CALIBRATED PLANT, BY SQUEEZE &#183; MEDIAN OF 4 ROLLOUTS</text>')
    xs = [x0 + i / (len(vals) - 1) * (x1 - x0) for i in range(len(vals))]
    for v in (0, 1, 2, 3, 4, 5):
        y = b - v / mx * (b - t)
        if y < t:
            break
        out.append(f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{x0 - 10}" y="{y + 4:.1f}" text-anchor="end">{v} N</text>')
    for x, v in zip(xs, vals):
        out.append(f'<text class="tick" x="{x:.1f}" y="{b + 18}" text-anchor="middle">{v:g} mm</text>')
    for r in rows:
        pts = [(x, r["lift_N"][i]) for i, x in enumerate(xs) if r["lift_N"][i] is not None]
        col = "var(--s2)" if r["id"] in ("D3", "D4", "D5", "D6", "D7") else "var(--ref)"
        out.append('<polyline fill="none" stroke="' + col + '" stroke-width="2" points="'
                   + " ".join(f"{x:.1f},{b - v / mx * (b - t):.1f}" for x, v in pts) + '"/>')
        x, v = pts[-1]
        out.append(f'<text class="ser" x="{x + 8:.1f}" y="{b - v / mx * (b - t) + 4:.1f}" fill="{col}">{r["id"]}</text>')
    out.append(f'<text class="note" x="{x0}" y="{h - 8}">Blue: the five hands that go on to carry the tool to the countersink. Grey: D1, D2, D8. D8 does not close on the tool below 6 mm.</text>')
    return "\n".join(out) + "</svg>"


def chart_funnel(d):
    """Rollouts still carried at each seam, calibrated plant, per hand."""
    f = d["funnel"]
    ph, rows = f["phases"], f["rows"]
    h = 330
    x0, x1, t, b = PAD_L, W - PAD_R - 30, 40, h - 90
    xs = [x0 + i / (len(ph) - 1) * (x1 - x0) for i in range(len(ph))]
    out = [f'<svg class="chart" viewBox="0 0 {W} {h}" role="img" aria-label="Rollouts carried at each seam">']
    out.append(f'<text class="axlab" x="{x0}" y="18">ROLLOUTS STILL CARRYING THE TOOL (&#8805;2 PADS &#8805; 0.240 N) AT EACH SEAM &#183; CALIBRATED PLANT, 10 MM SQUEEZE &#183; 20 PER HAND</text>')
    for v in (0, 5, 10, 15, 20):
        y = b - v / 20 * (b - t)
        out.append(f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{x0 - 10}" y="{y + 4:.1f}" text-anchor="end">{v}</text>')
    for x, p in zip(xs, ph):
        out.append(f'<text class="tick" x="{x:.1f}" y="{b + 14}" text-anchor="end" transform="rotate(-40 {x:.1f} {b + 14})">{esc(p)}</text>')
    for r in rows:
        col = "var(--s2)" if r["id"] in ("D3", "D4", "D5", "D6", "D7") else "var(--ref)"
        out.append('<polyline fill="none" stroke="' + col + '" stroke-width="2" points="'
                   + " ".join(f"{x:.1f},{b - v / 20 * (b - t):.1f}" for x, v in zip(xs, r["carried"])) + '"/>')
        out.append(f'<text class="ser" x="{xs[-1] + 8:.1f}" y="{b - r["carried"][-1] / 20 * (b - t) + 4:.1f}" fill="{col}">{r["id"]}</text>')
    return "\n".join(out) + "</svg>"


def table_compare(d):
    r = []
    for x in d["compare"]:
        s, c = x["shipped"], x["corrected"]
        r.append(f'<tr><td>{x["id"]}</td><td class="num">{x["bench_hold"]}</td>'
                 f'<td class="num">{s["lift_pads"]}p {s["lift_N"]:.2f}</td><td class="num">{c["lift_pads"]}p {c["lift_N"]:.2f}</td>'
                 f'<td class="num">{s["held_turn"]}/{s["n"]}</td><td class="num">{c["held_turn"]}/{c["n"]}</td>'
                 f'<td class="num">{s["best_cos"]:+.3f}</td><td class="num">{c["best_cos"]:+.3f}</td>'
                 f'<td class="num">{s["chains"]}</td><td class="num">{c["chains"]}</td></tr>')
    return "\n".join(r)


def table_turn(block, unit):
    vals = block["values"]
    head = "".join(f'<th class="num">{v:g} {unit}</th>' for v in vals)
    r = []
    for x in block["rows"]:
        cells = "".join(f'<td class="num">{t:+.1f}&#176; <span class="dim">{f:.1f} N</span></td>'
                        for t, f in zip(x["turn_deg"], x["force_N"]))
        r.append(f'<tr><td>{x["id"]}</td>{cells}<td class="num">{sum(x["held"])}/{sum(x["n"])}</td></tr>')
    return head, "\n".join(r)


def table_funnel(d):
    r = []
    i = {p: j for j, p in enumerate(d["funnel"]["phases"])}
    for x in d["funnel"]["rows"]:
        c = x["carried"]
        hi = ' class="hi"' if c[i["staged"]] >= 17 else ""
        tilt = "&#8212;" if x["final_tilt_med"] is None else f'{x["final_tilt_med"]:.1f}'
        turns = "&#8212;" if x["turns_med"] is None else f'{x["turns_med"]:.3f}'
        r.append(f'<tr{hi}><td>{x["id"]}</td><td class="num">{x["bench_hold"]}</td>'
                 f'<td class="num">{c[i["lifted"]]}</td><td class="num">{c[i["turned"]]}</td>'
                 f'<td class="num">{c[i["regripped"]]}</td><td class="num">{c[i["staged"]]}</td>'
                 f'<td class="num">{c[i["set_down"]]}</td><td class="num">{c[i["handover_grip"]]}</td>'
                 f'<td class="num">{c[i["gaited"]]}</td>'
                 f'<td class="num">{x["staged_cos"]:+.3f} / {x["staged_N"]:.2f}</td>'
                 f'<td class="num">{tilt}</td><td class="num">{turns}</td>'
                 f'<td class="num">{x["stood"]}</td></tr>')
    return "\n".join(r)


def table_probe(d, plant):
    r = []
    keep = ("lifted", "turned", "reoriented", "staged", "set_down", "pressed", "handover_thumb",
            "handover_index", "handover_middle", "handover_grip", "gaited")
    for s in d["probe"][plant]["seams"]:
        if s["phase"] not in keep:
            continue
        q, sat = s["q_err"], s["sat"]
        r.append(f'<tr><td class="mono">{esc(s["phase"])}</td><td class="num">{s["cos"]:+.3f}</td>'
                 f'<td class="num">{s["pads"]}</td><td class="num">{s["N"]:.2f}</td><td class="num">{s["slide"]:.1f}</td>'
                 f'<td class="num">{q.get("thumb", 0):.0f} / {q.get("index", 0):.0f} / {q.get("middle", 0):.0f}</td>'
                 f'<td class="num">{sat.get("thumb", 0)} / {sat.get("index", 0)} / {sat.get("middle", 0)}</td></tr>')
    return "\n".join(r)


def main():
    d = json.load(open(DATA))
    tpl = open(TPL).read()
    bh, bt = table_turn(d["budget"], "rad")
    rh, rt = table_turn(d["relief"], "mm")
    sub = {
        "N_TOTAL": f'{sum(d["n"].values()):,}',
        "CHART_LIFT": chart_lift(d), "CHART_SQUEEZE": chart_squeeze(d), "CHART_FUNNEL": chart_funnel(d),
        "TABLE_COMPARE": table_compare(d),
        "BUDGET_HEAD": bh, "TABLE_BUDGET": bt, "RELIEF_HEAD": rh, "TABLE_RELIEF": rt,
        "TABLE_FUNNEL": table_funnel(d),
        "TABLE_PROBE_CORRECTED": table_probe(d, "corrected"),
        "TABLE_PROBE_SHIPPED": table_probe(d, "shipped"),
        "PROBE_TILT": f'{d["probe"]["corrected"]["final_tilt"]:.1f}',
        "PROBE_TURNS": f'{d["probe"]["corrected"]["turns"]:.3f}',
    }
    for name in sorted(os.listdir(MEDIA)):
        sub["MEDIA_" + name.replace(".", "_").upper()] = uri(name)
    for k, v in sub.items():
        tpl = tpl.replace("{{" + k + "}}", v)
    open(OUT, "w").write(tpl)
    left = sorted({t.split("}}")[0] for t in tpl.split("{{")[1:] if "}}" in t})
    print(f"-> {OUT}  {os.path.getsize(OUT) / 1e6:.2f} MB" + (f"  UNSUBSTITUTED {left}" if left else ""))


if __name__ == "__main__":
    main()
