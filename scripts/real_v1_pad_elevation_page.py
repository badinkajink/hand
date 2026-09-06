#!/usr/bin/env python3
"""Build the pad-elevation result page from the studies' own JSON.

    uv run python scripts/real_v1_pad_elevation_page.py

Reads the closure-geometry probe, the straddle/depth grid, the grip-loop sweep, the air-mode
screened turn and the elevation sweep, and writes
docs/experiments/20260906-pad_elevation/20260906-pad_elevation.html. Media are inlined as data
URIs because the published artifact is a single file.
"""
from __future__ import annotations

import base64, json, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "docs/experiments"
OUTDIR = E / "20260906-pad_elevation"
OUT = OUTDIR / "20260906-pad_elevation.html"
TPL = Path(__file__).with_name("real_v1_pad_elevation_page.template.html")
MEDIA = OUTDIR / "media"

DID = {"sv1_w6689_b060": "D1", "sv1_w2360_b075": "D2", "sv1_u1364_b080": "D3",
       "g12_b095": "D4", "sv1_u0060_b75": "D5", "sv1_u0308_b050": "D6",
       "rv05_manual_b85": "D7", "sv1_w0099_b100": "D8"}
ORDER = list(DID)
# Hardware retention from the CB1 archive, `manual_score.success` over completed reorientation
# trials (docs/experiments/20260902-cb1-log-archive/logs; scripts/real_v1_bench_trials.py).
BENCH = {"D1": "6/7", "D2": "10/10", "D3": "4/10", "D4": "9/10",
         "D5": "3/10", "D6": "2/10", "D7": "10/10", "D8": "4/10"}
ELEVS = [0, -6, -12, -18, -24]


def b64(p: Path, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()


def table(head, rows, nums, hi=frozenset(), cls=""):
    th = "".join(f'<th{" class=num" if i in nums else ""}>{h}</th>' for i, h in enumerate(head))
    body = []
    for k, r in enumerate(rows):
        td = "".join(f'<td{" class=num" if i in nums else ""}>{c}</td>' for i, c in enumerate(r))
        body.append(f'<tr{" class=hi" if k in hi else ""}>{td}</tr>')
    return (f'<div class="tw {cls}"><table><thead><tr>' + th + "</tr></thead><tbody>"
            + "".join(body) + "</tbody></table></div>")


def rows_of(path: Path):
    return json.loads(path.read_text())["rows"]


def cnt(g, k):
    return sum(1 for r in g if r.get(k))


def funnel(g):
    return {k: cnt(g, k) for k in ("held_lift", "carry_ok", "held_turn", "stood_ok",
                                   "grip_ok", "ok")}


def seam_mean(g, phase, field):
    v = []
    for r in g:
        s = {x["phase"]: x for x in r.get("seams", [])}.get(phase)
        if s and s.get(field) is not None:
            v.append(s[field])
    return st.mean(v) if v else None


# --------------------------------------------------------------------------------- figures

def fig_closure(clo: dict) -> str:
    """z_rel per hand on a shaft cross-section axis, with the equator marked."""
    W, H, L, R = 1020, 300, 168, 46
    x0, x1 = L, W - R
    lo, hi = -0.60, 0.30
    sx = lambda z: x0 + (z - lo) / (hi - lo) * (x1 - x0)
    p = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" '
         f'aria-label="pad contact height per hand">']
    for z in (-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3):
        p.append(f'<line class="grid" x1="{sx(z):.1f}" y1="34" x2="{sx(z):.1f}" y2="{H-38}"/>')
        p.append(f'<text class="tick" x="{sx(z):.1f}" y="{H-22}" text-anchor="middle">'
                 f'{z:+.1f}</text>')
    p.append(f'<line x1="{sx(0):.1f}" y1="28" x2="{sx(0):.1f}" y2="{H-38}" '
             f'stroke="var(--ink3)" stroke-width="1.5"/>')
    p.append(f'<text class="mark" x="{sx(0):.1f}" y="20" text-anchor="middle">EQUATOR</text>')
    y = 44
    for tag in ORDER:
        c = clo[tag]
        held = c["Fz_pad"] > 0
        col = "var(--s2)" if held else "var(--s1)"
        p.append(f'<text class="ser" x="{L-12}" y="{y+4}" text-anchor="end" fill="var(--ink2)">'
                 f'{DID[tag]} {tag}</text>')
        for pad in c["pads"]:
            p.append(f'<circle cx="{sx(pad["z_rel"]):.1f}" cy="{y}" r="4.5" fill="{col}" '
                     f'fill-opacity="0.5"/>')
        p.append(f'<circle cx="{sx(c["z_rel_mean"]):.1f}" cy="{y}" r="6.5" fill="{col}"/>')
        p.append(f'<text class="val" x="{x1+8}" y="{y+4}" fill="{col}">'
                 f'{c["Fz_pad"]:+.3f} N</text>')
        y += 27
    p.append(f'<text class="axlab" x="{(x0+x1)/2:.0f}" y="{H-6}" text-anchor="middle">'
             f'PAD CONTACT HEIGHT ABOVE THE SHAFT AXIS / SHAFT RADIUS</text>')
    p.append("</svg>")
    return "".join(p)


def fig_elev(grid: dict) -> str:
    """Full chain completions out of six, per hand per pad-ring elevation."""
    W, H, L, T = 1020, 300, 168, 54
    cw, ch = 132, 34
    p = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" '
         f'aria-label="chain completions by pad elevation">']
    for j, el in enumerate(ELEVS):
        p.append(f'<text class="tick" x="{L+j*cw+cw/2:.0f}" y="{T-14}" text-anchor="middle">'
                 f'{el:+d}&#176;</text>')
    for i, tag in enumerate(ORDER):
        yy = T + i * ch
        p.append(f'<text class="ser" x="{L-12}" y="{yy+21}" text-anchor="end" '
                 f'fill="var(--ink2)">{DID[tag]} {tag}</text>')
        best = max((grid.get((tag, e), {}).get("ok", -1) for e in ELEVS))
        for j, el in enumerate(ELEVS):
            g = grid.get((tag, el))
            x = L + j * cw
            if g is None:
                p.append(f'<rect x="{x}" y="{yy+4}" width="{cw-6}" height="{ch-10}" rx="4" '
                         f'fill="none" stroke="var(--rule)" stroke-dasharray="3 3"/>')
                p.append(f'<text class="note" x="{x+(cw-6)/2:.0f}" y="{yy+21}" '
                         f'text-anchor="middle">no fit</text>')
                continue
            f = g["ok"] / 6.0
            p.append(f'<rect x="{x}" y="{yy+4}" width="{cw-6}" height="{ch-10}" rx="4" '
                     f'fill="var(--s2)" fill-opacity="{0.10 + 0.72*f:.2f}" '
                     f'stroke="{"var(--s1)" if g["ok"] == best and best > 0 else "none"}" '
                     f'stroke-width="1.6"/>')
            p.append(f'<text class="val" x="{x+(cw-6)/2:.0f}" y="{yy+21}" text-anchor="middle" '
                     f'fill="{"var(--ink)" if f > 0.45 else "var(--ink2)"}">'
                     f'{g["ok"]}/6 &#183; {g["stood_ok"]} stood</text>')
    p.append(f'<text class="axlab" x="{L+2.5*cw:.0f}" y="{H-8}" text-anchor="middle">'
             f'PAD-RING ELEVATION ABOUT THE SHAFT</text>')
    p.append("</svg>")
    return "".join(p)


def main() -> int:
    clo = {r["tag"]: r for r in json.loads(
        (E / "20260906-griploop/grasp_closure.json").read_text())}
    elev_clo = json.loads((E / "20260906-griploop/grasp_closure_elev.json").read_text())
    depth_clo = json.loads((E / "20260906-griploop/grasp_closure_d1d8.json").read_text())
    grid_rows = rows_of(E / "20260906-d1_graspgrid/chain_hands.json")
    loop_rows = rows_of(E / "20260906-griploop/chain_hands.json")
    air_rows = rows_of(E / "20260906-screened_turn/chain_hands.json")
    ele_rows = rows_of(E / "20260906-elevation/chain_hands.json")
    band_f = E / "20260906-turn_band/chain_hands.json"
    band_rows = rows_of(band_f) if band_f.exists() else []
    te_f = E / "20260906-turn_elev/chain_hands.json"
    te_rows = rows_of(te_f) if te_f.exists() else []

    V = {}

    # ---- 1. closure geometry, the shipped grasp
    ref = {t: [r for r in loop_rows if r["tag"] == t and r["arm"] == "load0"] for t in ORDER}
    V["CLOSURE_FIG"] = fig_closure(clo)
    V["CLOSURE_TBL"] = table(
        ["hand", "id", "pads below<br>the equator", "z_rel", "span", "F<sub>z,pad</sub>",
         "&Delta;z at close", "carry (chain)", "bench<br>retention"],
        [[f"<code>{t}</code>", DID[t], f'{clo[t]["n_below"]}/{clo[t]["n_pads"]}',
          f'{clo[t]["z_rel_mean"]:+.3f}', f'{clo[t]["span_deg"]:.0f}&#176;',
          f'<span class="{"ok" if clo[t]["Fz_pad"] > 0 else "no"}">{clo[t]["Fz_pad"]:+.3f} N</span>',
          f'{clo[t]["dz_mm"]:+.2f} mm',
          f'{cnt(ref[t], "carry_ok")}/{len(ref[t])}', BENCH[DID[t]]] for t in ORDER],
        nums={2, 3, 4, 5, 6, 7, 8},
        hi={i for i, t in enumerate(ORDER) if clo[t]["Fz_pad"] < 0})

    # ---- 2. what it is not
    gd = {}
    for r in grid_rows:
        if r.get("parent") != "sv1_w6689_b060":
            continue
        gd.setdefault((r["straddle_mm"], r["depth_ask_mm"]), []).append(r)
    depths = sorted({k[1] for k in gd})
    V["GRID_TBL"] = table(
        ["straddle"] + [f"depth {d:g} mm" for d in depths],
        [[f"{s:g} mm"] + [(lambda g: f'{cnt(g, "held_lift")}/{len(g)} lift &#183; '
                                     f'{cnt(g, "carry_ok")}/{len(g)} carry')(gd[(s, d)])
                          if (s, d) in gd else "&#8212;" for d in depths]
         for s in sorted({k[0] for k in gd})],
        nums=set(range(1, len(depths) + 1)))
    V["GRID_N"] = str(sum(len(v) for v in gd.values()))
    V["GRID_CARRY"] = str(sum(cnt(v, "carry_ok") for v in gd.values()))

    d1 = [r for r in depth_clo if r["tag"].startswith("sv1_w6689")]
    V["D1_DEPTH_MIN"] = f'{min(r["z_rel_mean"] for r in d1):+.3f}'
    V["D1_DEPTH_N"] = str(len(d1))

    loop = {}
    for r in loop_rows:
        loop.setdefault((r["tag"], r["arm"]), []).append(r)
    arms = ["load0", "load150", "load250"]
    V["LOOP_TBL"] = table(
        ["hand", "id"] + [a.replace("load", "load target ") for a in arms],
        [[f"<code>{t}</code>", DID[t]] +
         [(lambda g: f'{cnt(g, "stood_ok")}/{len(g)} stood &#183; '
                     f'{seam_mean(g, "lifted", "pad_force_N"):.1f} N')(loop[(t, a)])
          for a in arms] for t in ORDER],
        nums={2, 3, 4})

    # ---- 3. the screened turn at the plan's own setting
    air0 = {t: [r for r in air_rows if r["tag"] == t and r["arm"] == "load0"] for t in ORDER}
    V["AIR_N"] = str(len(air_rows))
    V["AIR_STOOD"] = str(sum(cnt(v, "stood_ok") for v in air_rows and
                             [[r] for r in air_rows] or []))
    rv = air0["rv05_manual_b85"]
    V["AIR_TBL"] = table(
        ["hand", "id", "plan angle", "tilt at <code>lifted</code>", "tilt at <code>turned</code>",
         "pad force<br>at <code>lifted</code>", "pad force<br>at <code>turned</code>",
         "stood"],
        [[f"<code>{t}</code>", DID[t], f'{air0[t][0].get("angle_deg"):.0f}&#176;',
          f'{seam_mean(air0[t], "lifted", "tilt_deg"):.1f}&#176;',
          f'{seam_mean(air0[t], "turned", "tilt_deg"):.1f}&#176;',
          f'{seam_mean(air0[t], "lifted", "pad_force_N"):.2f} N',
          f'<span class="no">{seam_mean(air0[t], "turned", "pad_force_N"):.2f} N</span>',
          f'{cnt(air0[t], "stood_ok")}/{len(air0[t])}'] for t in ORDER],
        nums={2, 3, 4, 5, 6, 7})

    # ---- 4. the elevation sweep
    grid = {}
    for r in ele_rows:
        grid.setdefault((r["parent"], int(r["elevation_deg"])), []).append(r)
    G = {k: funnel(v) | {"n": len(v)} for k, v in grid.items()}
    V["ELEV_FIG"] = fig_elev(G)
    best = {t: max((e for e in ELEVS if (t, e) in G), key=lambda e: (G[(t, e)]["ok"],
                                                                    G[(t, e)]["stood_ok"]))
            for t in ORDER}
    V["ELEV_TBL"] = table(
        ["hand", "id", "shipped 0&#176;", "best elevation", "chain there",
         "carry", "stood", "gait re-grip"],
        [[f"<code>{t}</code>", DID[t],
          f'{G[(t, 0)]["ok"]}/6',
          f'{best[t]:+d}&#176;',
          f'<span class="{"ok" if G[(t, best[t])]["ok"] >= 4 else ""}">'
          f'{G[(t, best[t])]["ok"]}/6</span>',
          f'{G[(t, best[t])]["carry_ok"]}/6', f'{G[(t, best[t])]["stood_ok"]}/6',
          f'{G[(t, best[t])]["grip_ok"]}/6'] for t in ORDER],
        nums={2, 3, 4, 5, 6, 7},
        hi={i for i, t in enumerate(ORDER) if G[(t, best[t])]["ok"] > G[(t, 0)]["ok"]})
    V["ELEV_BEST_SUM"] = str(sum(G[(t, best[t])]["ok"] for t in ORDER))
    V["ELEV_SHIP_SUM"] = str(sum(G[(t, 0)]["ok"] for t in ORDER))
    V["ELEV_CARRY_SHIP"] = str(sum(1 for t in ORDER if G[(t, 0)]["carry_ok"] >= 4))
    V["ELEV_CARRY_BEST"] = str(sum(1 for t in ORDER if G[(t, best[t])]["carry_ok"] >= 4))

    ec = {(r["tag"], r["elevation_deg"]): r for r in elev_clo}
    V["ELEV_CLO_TBL"] = table(
        ["hand", "id"] + [f"{e:+d}&#176;" for e in ELEVS],
        [[f"<code>{t}</code>", DID[t]] +
         [(lambda c: f'{c["n_below"]}/3 &#183; {c["Fz_pad"]:+.3f} N')(ec[(t, float(e))])
          if (t, float(e)) in ec else "&#8212;" for e in ELEVS] for t in ORDER],
        nums=set(range(2, 2 + len(ELEVS))))

    # ---- 5. the turn band
    if band_rows:
        bb = {}
        for r in band_rows:
            bb.setdefault((r["tag"], r["arm"]), []).append(r)
        arms_b = sorted({r["arm"] for r in band_rows})
        V["BAND_TBL"] = table(
            ["hand", "id"] + [a.replace("load0_", "").replace("t", "steps ").replace("_b", " / clip ")
                              for a in arms_b],
            [[f"<code>{t}</code>", DID[t]] +
             [(lambda g: f'{cnt(g, "held_turn")}/{len(g)} held &#183; '
                         f'{seam_mean(g, "turned", "tilt_deg"):.0f}&#176;')(bb[(t, a)])
               if (t, a) in bb else "&#8212;" for a in arms_b] for t in ORDER],
            nums=set(range(2, 2 + len(arms_b))))
        held = [(t, a, cnt(bb[(t, a)], "held_turn"), seam_mean(bb[(t, a)], "turned", "tilt_deg"))
                for (t, a) in bb]
        good = [h for h in held if h[2] == 4 and h[3] is not None and h[3] < 60]
        V["BAND_GOOD"] = str(len(good))
        V["BAND_N"] = str(len(band_rows))
        V["BAND_NOTE"] = ""
    else:
        V["BAND_TBL"] = '<p class="pending">The sweep is still running.</p>'
        V["BAND_GOOD"] = V["BAND_N"] = "&#8212;"
        V["BAND_NOTE"] = ""

    # ---- 6. the turn and the elevation together
    if te_rows:
        tb = {}
        for r in te_rows:
            tb.setdefault((r["parent"], int(r["elevation_deg"]), r["arm"]), []).append(r)
        arms_t = sorted({r["arm"] for r in te_rows})
        V["TE_N"] = str(len(te_rows))
        for el in (0, -12):
            sub = [r for r in te_rows if int(r["elevation_deg"]) == el]
            V[f"TE_CARRY_{abs(el)}"] = f'{cnt(sub, "carry_ok")} of {len(sub)}'
        V["TE_OK"] = str(cnt(te_rows, "ok"))

        def cell(g):
            if g is None:
                return "&#8212;"
            t = seam_mean(g, "turned", "tilt_deg")
            f = seam_mean(g, "turned", "pad_force_N")
            k = cnt(g, "carry_ok")
            c = "ok" if k == len(g) else ("no" if k == 0 else "")
            turned = 90.0 - t
            return (f'<span class="{c}">{k}/{len(g)}</span> &#183; '
                    f'{turned if abs(turned) >= 0.5 else 0.0:.0f}&#176; &#183; {f:.2f} N')
        V["TE_TBL"] = table(
            ["hand", "id", "elev"] + [a.replace("load0_t", "").replace("_b", " steps, clip ")
                                      for a in arms_t],
            [[f"<code>{t}</code>" if el == 0 else "", DID[t] if el == 0 else "",
              f"{el:+d}&#176;"] + [cell(tb.get((t, el, a))) for a in arms_t]
             for t in ORDER for el in (0, -12)],
            nums={3, 4, 5, 6},
            hi={2 * i + 1 for i, t in enumerate(ORDER)
                if cnt([r for r in te_rows if r["parent"] == t
                        and int(r["elevation_deg"]) == -12], "carry_ok")
                > cnt([r for r in te_rows if r["parent"] == t
                       and int(r["elevation_deg"]) == 0], "carry_ok")})
        b = tb[("rv05_manual_b85", -12, "load0_t250_b0.8")]
        V["TE_BEST"] = (f'{90.0 - seam_mean(b, "turned", "tilt_deg"):.1f}&#176; at '
                        f'{seam_mean(b, "turned", "pad_force_N"):.2f} N, carried '
                        f'{cnt(b, "carry_ok")}/{len(b)} and stood {cnt(b, "stood_ok")}/{len(b)}')
    else:
        V["TE_TBL"] = '<p class="pending">The sweep is still running.</p>'
        V["TE_N"] = V["TE_OK"] = V["TE_BEST"] = "&#8212;"
        V["TE_CARRY_0"] = V["TE_CARRY_12"] = "&#8212;"

    V["D1_LOOP_N"] = str(sum(len(v) for (t, _), v in loop.items() if t == "sv1_w6689_b060"))
    V["ELEV_STOOD6"] = str(sum(1 for t in ORDER
                               if any(G[(t, e)]["stood_ok"] == 6 for e in ELEVS if (t, e) in G)))
    V["N_ROLLOUTS"] = f"{len(grid_rows) + len(loop_rows) + len(air_rows) + len(ele_rows) + len(band_rows) + len(te_rows):,}"
    V["N_FITS"] = str(len(clo) + len(elev_clo) + len(depth_clo))

    # ---- media
    V["FIG_D1_E0"] = b64(MEDIA / "20260906-d1_e0_drop.png", "image/png")
    V["FIG_D1_E24"] = b64(MEDIA / "20260906-d1_e24_chain.png", "image/png")
    V["FIG_D8_E12"] = b64(MEDIA / "20260906-d8_e12_chain.png", "image/png")
    V["VID_D1"] = b64(MEDIA / "20260906-d1_e24_chain.mp4", "video/mp4")

    html = TPL.read_text()
    for k, v in V.items():
        html = html.replace("{{" + k + "}}", v)
    assert "{{" not in html, [s[:40] for s in html.split("{{")[1:]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print(f"-> {OUT}  ({OUT.stat().st_size/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
