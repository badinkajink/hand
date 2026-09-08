#!/usr/bin/env python3
"""Build the reorientation journey page from journey.json.

    python3 scripts/reorientation_journey_data.py     # collect the sweeps
    python3 scripts/reorientation_journey_page.py     # build the page

Media under docs/experiments/20260908-reorientation_journey/media/ are inlined as data
URIs, so the output file is the whole artifact.
"""
from __future__ import annotations
import base64, json, mimetypes, os

ROOT = "docs/experiments/20260908-reorientation_journey"
DATA = f"{ROOT}/journey.json"
MEDIA = f"{ROOT}/media"
TPL = "scripts/reorientation_journey_page.template.html"
OUT = f"{ROOT}/20260908-reorientation_journey.html"

W, H = 1020, 0          # charts are full-column width
PAD_L, PAD_R = 132, 24


def uri(name: str) -> str:
    p = os.path.join(MEDIA, name)
    mt = mimetypes.guess_type(p)[0] or "application/octet-stream"
    return f"data:{mt};base64," + base64.b64encode(open(p, "rb").read()).decode()


def esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------- charts

def chart_attrition(d) -> str:
    """Where the tool is lost, over every air-mode rollout."""
    rows = d["attrition"]
    tot = sum(r["n"] for r in rows)
    bh, gap = 22, 9
    h = 44 + len(rows) * (bh + gap) + 30
    x0, x1 = PAD_L, W - PAD_R - 58
    mx = max(r["n"] for r in rows)
    out = [f'<svg class="chart" viewBox="0 0 {W} {h}" role="img" '
           f'aria-label="Where the tool is lost across {tot} air-mode rollouts">']
    out.append(f'<text class="axlab" x="{PAD_L}" y="18">SEAM AT WHICH THE TOOL IS ON THE '
               f'FLOOR WITH NO PAD ON IT &#183; n = {tot}</text>')
    for i, r in enumerate(rows):
        y = 36 + i * (bh + gap)
        wpx = (r["n"] / mx) * (x1 - x0)
        fill = "var(--good)" if r["phase"] == "carried" else (
            "var(--s1)" if r["phase"] in ("turned", "reoriented") else "var(--s2)")
        out.append(f'<text class="tick" x="{PAD_L - 10}" y="{y + bh * 0.72}" '
                   f'text-anchor="end">{esc(r["phase"])}</text>')
        out.append(f'<rect x="{x0}" y="{y}" width="{max(wpx, 2):.1f}" height="{bh}" '
                   f'rx="4" fill="{fill}"/>')
        out.append(f'<text class="val" x="{x0 + wpx + 9:.1f}" y="{y + bh * 0.74}">'
                   f'{r["n"]}  <tspan fill="var(--ink3)">{100 * r["n"] / tot:.1f}%</tspan></text>')
    out.append(f'<text class="note" x="{PAD_L}" y="{h - 8}">The turn and the settle that '
               f'follows it account for {sum(r["n"] for r in rows if r["phase"] in ("turned", "reoriented")) / tot:.0%} '
               f'of all losses. Amber = during the reorientation.</text>')
    return "\n".join(out) + "</svg>"


def chart_hands(d) -> str:
    """Per hand: how far the tool gets, as a fraction of that hand's rollouts."""
    rows = d["hands"]
    bh, sub, gap = 10, 3, 16
    grp = bh * 3 + sub * 2
    h = 56 + len(rows) * (grp + gap) + 52
    x0, x1 = PAD_L, W - PAD_R - 96
    out = [f'<svg class="chart" viewBox="0 0 {W} {h}" role="img" '
           f'aria-label="Per-hand outcome fractions">']
    out.append(f'<text class="axlab" x="{PAD_L}" y="18">FRACTION OF THAT HAND\'S AIR-MODE '
               f'ROLLOUTS</text>')
    for f in (0, .25, .5, .75, 1.0):
        x = x0 + f * (x1 - x0)
        out.append(f'<line class="grid" x1="{x:.1f}" y1="30" x2="{x:.1f}" y2="{h - 56}"/>')
        out.append(f'<text class="tick" x="{x:.1f}" y="{h - 42}" text-anchor="middle">'
                   f'{f:.0%}</text>')
    keys = [("survive", "still on a pad after the turn", "var(--s2)"),
            ("held_turn", "held tip-down turn", "var(--s1)"),
            ("chain", "chain completed", "var(--good)")]
    for i, r in enumerate(rows):
        y = 38 + i * (grp + gap)
        out.append(f'<text class="ser" x="{PAD_L - 10}" y="{y + grp / 2 + 4}" '
                   f'text-anchor="end" fill="var(--ink)">{r["id"]}</text>')
        out.append(f'<text class="tick" x="{PAD_L - 34}" y="{y + grp / 2 + 4}" '
                   f'text-anchor="end">{esc(r["tag"])}</text>')
        for j, (k, _lab, col) in enumerate(keys):
            frac = r[k] / r["n"]
            yy = y + j * (bh + sub)
            out.append(f'<rect x="{x0}" y="{yy}" width="{x1 - x0}" height="{bh}" rx="3" '
                       f'fill="var(--sunk)"/>')
            out.append(f'<rect x="{x0}" y="{yy}" width="{max(frac * (x1 - x0), 1.5):.1f}" '
                       f'height="{bh}" rx="3" fill="{col}"/>')
            out.append(f'<text class="val" x="{x1 + 8}" y="{yy + bh - 1}">'
                       f'{r[k]}/{r["n"]}</text>')
    lx = PAD_L
    for k, lab, col in keys:
        out.append(f'<rect x="{lx}" y="{h - 14}" width="9" height="9" rx="2" fill="{col}"/>')
        out.append(f'<text class="note" x="{lx + 14}" y="{h - 6}">{lab}</text>')
        lx += 26 + len(lab) * 6.9
    return "\n".join(out) + "</svg>"


def chart_pivot(d) -> str:
    """Best signed cosine at the turn, per hand per pivot height. Diverging about 0."""
    ks, rows = d["pivot"]["axis_k"], d["pivot"]["rows"]
    cw, ch, gap = 96, 34, 6
    h = 84 + len(rows) * (ch + gap) + 44
    x0 = PAD_L
    out = [f'<svg class="chart" viewBox="0 0 {W} {h}" role="img" '
           f'aria-label="Best signed cosine at the turn, by hand and pivot height">']
    out.append(f'<text class="axlab" x="{x0}" y="18">BEST SIGNED COSINE AT THE TURN &#183; '
               f'OPEN LOOP &#183; +1 IS TIP DOWN, &#8722;1 IS HANDLE DOWN</text>')
    for j, k in enumerate(ks):
        out.append(f'<text class="tick" x="{x0 + j * (cw + gap) + cw / 2:.0f}" y="46" '
                   f'text-anchor="middle">axis_k {k:.2f}</text>')
    for i, r in enumerate(rows):
        y = 56 + i * (ch + gap)
        out.append(f'<text class="ser" x="{x0 - 10}" y="{y + ch * 0.68:.0f}" '
                   f'text-anchor="end" fill="var(--ink)">{r["id"]}</text>')
        for j, v in enumerate(r["cos"]):
            x = x0 + j * (cw + gap)
            if v is None:
                out.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="5" '
                           f'fill="var(--sunk)"/>')
                out.append(f'<text class="tick" x="{x + cw / 2:.0f}" y="{y + ch * 0.68:.0f}" '
                           f'text-anchor="middle">&#183;</text>')
                continue
            a = min(abs(v), 1.0)
            hue = "var(--s2)" if v >= 0 else "var(--s1)"
            out.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="5" '
                       f'fill="color-mix(in oklab,{hue} {12 + a * 76:.0f}%,var(--card))"/>')
            ink = "#FFFFFF" if a > 0.62 else "var(--ink)"
            out.append(f'<text class="val" x="{x + cw / 2:.0f}" y="{y + ch * 0.68:.0f}" '
                       f'text-anchor="middle" fill="{ink}">{v:+.3f}</text>')
    out.append(f'<text class="note" x="{x0}" y="{h - 24}">Best of every open-loop rollout in '
               f'that cell. A hand&#8217;s good pivot is not the neighbouring hand&#8217;s: D7 '
               f'peaks at 0.05, D6 and D5 at 0.15, D1 at 0.35.</text>')
    out.append(f'<text class="note" x="{x0}" y="{h - 8}">D8 never turns the tool at any pivot '
               f'height tried. Dots are cells that were never run.</text>')
    return "\n".join(out) + "</svg>"


def chart_ledger(d) -> str:
    """Every sweep, in the order it was run, sized by rollout count."""
    rows = d["ledger"]
    bw, gap = 40, 16
    h = 250
    x0 = PAD_L - 60
    mx = max(r["n"] for r in rows)
    top, bot = 44, h - 62
    out = [f'<svg class="chart" viewBox="0 0 {W} {h}" role="img" '
           f'aria-label="Rollouts per sweep and what survived">']
    out.append(f'<text class="axlab" x="{x0}" y="18">ROLLOUTS PER SWEEP, AND WHAT SURVIVED '
               f'THE TURN</text>')
    for i, r in enumerate(rows):
        x = x0 + i * (bw + gap)
        hh = (r["n"] / mx) * (bot - top)
        y = bot - hh
        hs = (r["survive_turn"] / mx) * (bot - top)
        ht = (r["held_turn"] / mx) * (bot - top)
        out.append(f'<rect x="{x}" y="{y:.1f}" width="{bw}" height="{hh:.1f}" rx="3" '
                   f'fill="var(--sunk)"/>')
        out.append(f'<rect x="{x}" y="{bot - hs:.1f}" width="{bw}" height="{hs:.1f}" rx="3" '
                   f'fill="var(--s2)" opacity=".55"/>')
        out.append(f'<rect x="{x}" y="{bot - ht:.1f}" width="{bw}" height="{max(ht, 1.5):.1f}" '
                   f'rx="3" fill="var(--s1)"/>')
        out.append(f'<text class="val" x="{x + bw / 2:.0f}" y="{y - 6:.1f}" '
                   f'text-anchor="middle">{r["n"]}</text>')
        out.append(f'<text class="tick" x="{x + bw / 2:.0f}" y="{bot + 16}" '
                   f'text-anchor="middle">{i + 1}</text>')
        if r["chain"]:
            col = "var(--good)" if r["mode"] == "air" else "var(--ref)"
            out.append(f'<circle cx="{x + bw / 2:.0f}" cy="{y - 20:.1f}" r="4.5" '
                       f'fill="{col}"/>')
    lx = x0
    for lab, col, op in (("rollouts", "var(--sunk)", "1"), ("survive the turn", "var(--s2)", ".55"),
                         ("held tip-down turn", "var(--s1)", "1"),
                         ("a chain completed", "var(--good)", "1"),
                         ("in table mode: a stand", "var(--ref)", "1")):
        out.append(f'<rect x="{lx}" y="{h - 14}" width="9" height="9" rx="2" fill="{col}" '
                   f'opacity="{op}"/>')
        out.append(f'<text class="note" x="{lx + 14}" y="{h - 6}">{lab}</text>')
        lx += 28 + len(lab) * 6.9
    return "\n".join(out) + "</svg>"


def chart_d6(d) -> str:
    """The completed chain, seam by seam: orientation above, pad force below."""
    sm = d["d6"]["seams"]
    n = len(sm)
    h = 348
    x0, x1 = PAD_L, W - PAD_R - 20
    t, b = 40, 150          # cos panel
    t2, b2 = 190, 258       # force panel
    xs = [x0 + i * (x1 - x0) / (n - 1) for i in range(n)]

    def ycos(v):
        return b - (v + 1) / 2 * (b - t)
    fmax = max(s["pad_force_N"] for s in sm) or 1

    def yf(v):
        return b2 - v / fmax * (b2 - t2)
    out = [f'<svg class="chart" viewBox="0 0 {W} {h}" role="img" '
           f'aria-label="The completed chain on D6, seam by seam">']
    out.append(f'<text class="axlab" x="{x0}" y="18">D6 &#183; SIGNED COSINE (ABOVE) AND PAD '
               f'FORCE (BELOW), AT EVERY SEAM</text>')
    for v, lab in ((1, "+1 tip down"), (0, "0 horizontal"), (-1, "&#8722;1 handle down")):
        y = ycos(v)
        out.append(f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{x0 - 10}" y="{y + 4:.1f}" text-anchor="end">'
                   f'{lab}</text>')
    out.append(f'<polyline fill="none" stroke="var(--s2)" stroke-width="2" points="'
               + " ".join(f"{x:.1f},{ycos(s['cos']):.1f}" for x, s in zip(xs, sm)) + '"/>')
    for x, s in zip(xs, sm):
        out.append(f'<circle cx="{x:.1f}" cy="{ycos(s["cos"]):.1f}" r="4.5" fill="var(--s2)" '
                   f'stroke="var(--card)" stroke-width="2"/>')
    for i in (0, 1, 4, 12):
        out.append(f'<text class="val" x="{xs[i]:.1f}" y="{ycos(sm[i]["cos"]) - 12:.1f}" '
                   f'text-anchor="middle">{sm[i]["cos"]:+.3f}</text>')
    out.append(f'<text class="tick" x="{x0 - 10}" y="{b2 + 4}" text-anchor="end">0 N</text>')
    out.append(f'<line class="grid" x1="{x0}" y1="{b2}" x2="{x1}" y2="{b2}"/>')
    bw = (x1 - x0) / n * 0.5
    for x, s in zip(xs, sm):
        yy = yf(s["pad_force_N"])
        out.append(f'<rect x="{x - bw / 2:.1f}" y="{yy:.1f}" width="{bw:.1f}" '
                   f'height="{max(b2 - yy, 1.5):.1f}" rx="3" fill="var(--s1)"/>')
        out.append(f'<text class="mark" x="{x:.1f}" y="{b2 + 14}" text-anchor="middle">'
                   f'{s["pad_contacts"]}</text>')
    out.append(f'<text class="note" x="{x0 - 10}" y="{b2 + 14}" text-anchor="end">pads</text>')
    for x, s in zip(xs, sm):
        out.append(f'<text class="tick" x="{x:.1f}" y="{b2 + 30}" text-anchor="end" '
                   f'transform="rotate(-38 {x:.1f} {b2 + 30})">{esc(s["phase"])}</text>')
    out.append(f'<text class="note" x="{x0}" y="{h - 8}">Three pads carry the tool at every one '
               f'of the thirteen seams; the load-test floor is the tool&#8217;s own 0.240 N. The '
               f'rise between regripped and staged is the arm, not the fingers.</text>')
    return "\n".join(out) + "</svg>"


# ---------------------------------------------------------------- tables

def _chain_cell(x) -> str:
    if not x["chain"]:
        return "0"
    if x["mode"] == "table":
        return f'<span style="color:var(--ink3)">{x["chain"]}*</span>'
    return f'<span class="ok">{x["chain"]}</span>'


def table_ledger(d) -> str:
    r = []
    for i, x in enumerate(d["ledger"], 1):
        r.append(f'<tr><td class="num">{i}</td>'
                 f'<td class="mono">{esc(x["sweep"])}</td><td>{esc(x["mode"])}</td>'
                 f'<td class="num">{x["n"]}</td><td class="num">{x["survive_turn"]}</td>'
                 f'<td class="num">{x["held_turn"]}</td>'
                 f'<td class="num">{_chain_cell(x)}</td></tr>')
    return "\n".join(r)


def table_d6(d) -> str:
    r = []
    for s in d["d6"]["seams"]:
        cls = ' class="hi"' if s["phase"] == "staged" else ""
        r.append(f'<tr{cls}><td class="mono">{esc(s["phase"])}</td>'
                 f'<td class="num">{s["cos"]:+.3f}</td><td class="num">{s["tilt_deg"]:.1f}</td>'
                 f'<td class="num">{s["slide_mm"]:.1f}</td>'
                 f'<td class="num">{s["pad_contacts"]}</td>'
                 f'<td class="num">{s["pad_force_N"]:.2f}</td>'
                 f'<td class="num">{s["z"] * 1000:.0f}</td></tr>')
    return "\n".join(r)


def table_completed(d) -> str:
    r = []
    for x in d["completed"]:
        cls = ' class="hi"' if x["id"] == "D6" else ""
        r.append(f'<tr{cls}><td>{esc(x["id"])}</td><td class="mono">{esc(x["tag"])}</td>'
                 f'<td class="num">{x["axis_k"]:.2f}</td><td class="num">{x["seed"]}</td>'
                 f'<td class="mono">{esc(", ".join(x["sweeps"]))}</td>'
                 f'<td class="num">{x["staged_cos"]:+.3f}</td>'
                 f'<td class="num">{x["gaited_cos"]:+.3f}</td></tr>')
    return "\n".join(r)


ABL_WHAT = {
    "baseline": "the reference, unchanged",
    "tips=box": "spherical pads &#8594; the chain&#8217;s flat pads",
    "grasp=fit": "stored CEM grasp &#8594; the deployment fitter",
    "axis_k=0.05": "pivot 0.25 &#8594; the deployed plans&#8217; 0.05",
    "angle=-60": "turn angle &#8722;90&#176; &#8594; &#8722;60&#176;",
    "clip=0.85": "finger travel budget 0.50 &#8594; 0.85 rad",
    "turn=plan": "the plan&#8217;s own pivot and angle together",
    "all=deployed": "every deployed setting at once",
}


def table_ablation(d) -> str:
    r = []
    for x in d["ablation"]:
        drop = ", ".join(x["drops"]) if x["drops"] else "&#8212;"
        cls = ' class="hi"' if x["arm"] in ("axis_k=0.05", "grasp=fit") else ""
        v = (f'<span class="ok">{x["ok"]}/{x["n"]}</span>' if x["ok"] == x["n"]
             else (f'<span class="no">{x["ok"]}/{x["n"]}</span>' if not x["ok"]
                   else f'{x["ok"]}/{x["n"]}'))
        r.append(f'<tr{cls}><td class="mono">{esc(x["arm"])}</td>'
                 f'<td class="wrap">{ABL_WHAT.get(x["arm"], "")}</td>'
                 f'<td class="num">{v}</td><td class="mono">{drop}</td></tr>')
    return "\n".join(r)


def table_budget(d) -> str:
    r = []
    for x in d["d6"]["budget"]:
        cls = ' class="hi"' if x["chains"] else ""
        v = (f'<span class="ok">{x["chains"]}/{x["seeds"]}</span>' if x["chains"]
             else f'<span class="no">0/{x["seeds"]}</span>')
        r.append(f'<tr{cls}><td class="num">{x["budget_rad"]:.2f}</td>'
                 f'<td class="num">{x["seeds"]}</td><td class="num">{v}</td></tr>')
    return "\n".join(r)


def table_force(d) -> str:
    r = []
    for x in d["force"]:
        lab = "open loop" if x["target_N"] == 0 else f'{x["target_N"]:.0f} N per finger'
        r.append(f'<tr><td>{lab}</td><td class="num">{x["n"]}</td>'
                 f'<td class="num">{x["held_turn"]}</td>'
                 f'<td class="num">{x["median_slide_mm"]:.1f}</td></tr>')
    return "\n".join(r)


def table_d5(d) -> str:
    r = []
    for x in d["d5_control"]:
        lab = "open loop" if not x["force_target"] else f'{x["force_target"]:.0f} N'
        cls = ' class="hi"' if not x["force_target"] else ""
        r.append(f'<tr{cls}><td>{lab}</td><td class="num">{x["seed"]}</td>'
                 f'<td class="num">{x["cos"]:+.3f}</td><td class="num">{x["slide_mm"]:.1f}</td>'
                 f'<td class="num">{x["pads"]}</td><td class="num">{x["force_N"]:.2f}</td></tr>')
    return "\n".join(r)


def table_track(d) -> str:
    ids = sorted({x["id"] for x in d["track"]})
    gains = sorted({x["gain"] for x in d["track"]})
    m = {(x["id"], x["gain"]): x["median_slide_mm"] for x in d["track"]}
    r = []
    for i in ids:
        cells = "".join(f'<td class="num">{m.get((i, g), float("nan")):.1f}</td>' for g in gains)
        r.append(f"<tr><td>{i}</td>{cells}</tr>")
    return "\n".join(r), "".join(f'<th class="num">gain {g:g}</th>' for g in gains)


def main():
    d = json.load(open(DATA))
    tpl = open(TPL).read()
    trk_rows, trk_head = table_track(d)
    sub = {
        "N_ROLLOUTS": f'{d["n_rollouts"]:,}',
        "N_AIR": f'{d["n_air"]:,}',
        "N_SWEEPS": str(d["n_sweeps"]),
        "N_COMPLETED": str(len(d["completed"])),
        "CHART_ATTRITION": chart_attrition(d),
        "CHART_HANDS": chart_hands(d),
        "CHART_PIVOT": chart_pivot(d),
        "CHART_LEDGER": chart_ledger(d),
        "CHART_D6": chart_d6(d),
        "TABLE_LEDGER": table_ledger(d),
        "TABLE_D6": table_d6(d),
        "TABLE_COMPLETED": table_completed(d),
        "TABLE_ABLATION": table_ablation(d),
        "TABLE_BUDGET": table_budget(d),
        "TABLE_FORCE": table_force(d),
        "TABLE_D5": table_d5(d),
        "TABLE_TRACK": trk_rows,
        "TRACK_HEAD": trk_head,
        "D6_CELL_N": str(d["d6"]["cell_seeds"]),
        "D6_CELL_OK": str(d["d6"]["cell_ok"]),
        "D6_PIVOT15_N": str(d["d6"]["pivot15_n"]),
        "D6_PIVOT15_OK": str(d["d6"]["pivot15_ok"]),
        "D6_SLIDE_MAX": f'{d["d6"]["slide_max_mm"]:.1f}',
        "D6_ARM": esc(d["d6"]["arm"]),
    }
    for name in sorted(os.listdir(MEDIA)):
        sub["MEDIA_" + name.replace(".", "_").upper()] = uri(name)
    for k, v in sub.items():
        tpl = tpl.replace("{{" + k + "}}", v)
    open(OUT, "w").write(tpl)
    print(f"-> {OUT}  {os.path.getsize(OUT) / 1e6:.2f} MB")
    left = [t for t in tpl.split("{{")[1:] if "}}" in t]
    if left:
        print("UNSUBSTITUTED:", sorted({t.split('}}')[0] for t in left}))


if __name__ == "__main__":
    main()
