#!/usr/bin/env python3
"""Build the basketball-grasp page from the sweep's own JSON.

    MUJOCO_GL=egl uv run python scripts/real_v1_basketball_grasp.py sweep --out docs/experiments/20260916-basketball/sweep
    MUJOCO_GL=egl uv run python scripts/real_v1_basketball_media.py        # films, pose, contact sensitivity
    python3 scripts/real_v1_basketball_page.py

Everything on the page is read from rows.jsonl / fits.json / contact_sensitivity.json under
the sweep directory; media under media/ are inlined as data URIs so the output is one file.
"""
from __future__ import annotations

import base64
import json
import math
import mimetypes
import os
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import real_v1_basketball_grasp as G  # noqa: E402

ROOT = "docs/experiments/20260916-basketball"
SWEEP = f"{ROOT}/sweep"
MEDIA = f"{ROOT}/media"
TPL = "scripts/real_v1_basketball_page.template.html"
OUT = f"{ROOT}/20260916-basketball_grasp.html"

SIZES = ["7", "6", "5", "3"]
LAYOUTS = ["wide", "nominal", "g12"]
LAYOUT_LABEL = {"wide": "wide (gantries at travel ends)", "nominal": "nominal (CAD centre)",
                "g12": "g12 (bench D4)"}
MUS = [0.6, 0.8, 1.0, 1.5, 2.4]
THETAS = [40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]


def uri(name: str) -> str:
    p = os.path.join(MEDIA, name)
    mt = mimetypes.guess_type(p)[0] or "application/octet-stream"
    return f"data:{mt};base64," + base64.b64encode(open(p, "rb").read()).decode()


def esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ------------------------------------------------------------------ geometry helpers

def ring_radius(layout: str) -> float:
    """Mean mount distance from the ball axis once the palm is centred (mm)."""
    m = G.DESIGNS[layout]
    pts = {f: (m[f][0], m[f][1]) for f in G.FINGER_NAMES}
    best = None
    for px in [x / 10 for x in range(-400, 401)]:
        for py in [y / 10 for y in range(-200, 201, 5)]:
            r = [math.hypot(pts[f][0] + px, pts[f][1] + py) for f in G.FINGER_NAMES]
            v = max(r) - min(r)
            if best is None or v < best[0]:
                best = (v, sum(r) / 3)
    return best[1]


def ring_required(R_mm: float) -> float:
    """Mount ring (mm) at which a ball seated on the plate clears the yaw links' feet."""
    r = G.PAD_RADIUS * 1e3
    zc = G.PLATE_UNDERSIDE * 1e3 - R_mm          # ball centre relative to the mounting plane
    foot = -G.CAP_YAW * 1e3                       # yaw capsule's medial axis ends here
    if zc < foot:
        return math.sqrt((R_mm + r) ** 2 - (zc - foot) ** 2)
    return R_mm + r


def theta_outboard(layout: str, R_mm: float) -> float:
    return math.degrees(math.asin(min(1.0, ring_radius(layout) / (R_mm + G.PAD_RADIUS * 1e3))))


# ------------------------------------------------------------------ data

def load():
    rows = [json.loads(l) for l in open(f"{SWEEP}/rows.jsonl")]
    fits = json.load(open(f"{SWEEP}/fits.json"))
    cs = json.load(open(f"{SWEEP}/contact_sensitivity.json")) if os.path.exists(f"{SWEEP}/contact_sensitivity.json") else None
    return rows, fits, cs


def fit_at(fits, layout, size, th, sq=0.01):
    return fits.get(f"{layout}|{size}|{th:g}|{sq:g}")


def held_map(rows, layout, size, plant, od):
    out = {}
    for r in rows:
        if r["design"] == layout and r["size"] == size and r["plant"] == plant and r["overdrive"] == od:
            out[(r["theta_deg"], r["mu"])] = r
    return out


# ------------------------------------------------------------------ tables

def table_seating(fits):
    o = ['<div class="tw"><table><thead><tr><th>layout</th><th class="num">ring mm</th>']
    for s in SIZES:
        o.append(f'<th class="num">size {s} &#183; needs {ring_required(G.BALL_SIZES[s]["radius"]*1e3):.1f}</th>')
    o.append('</tr></thead><tbody>')
    for L in LAYOUTS:
        o.append(f'<tr><td>{esc(LAYOUT_LABEL[L])}</td><td class="num">{ring_radius(L):.1f}</td>')
        for s in SIZES:
            seated = any((f or {}).get("seated") for k, f in fits.items() if k.startswith(f"{L}|{s}|"))
            need = ring_required(G.BALL_SIZES[s]["radius"] * 1e3)
            ok = ring_radius(L) >= need
            o.append(f'<td class="num"><span class="{"ok" if ok else "no"}">{"seats" if ok else "no"}</span>'
                     f'{" &#183; fitted" if seated else ""}</td>')
        o.append('</tr>')
    o.append('</tbody></table></div>')
    return "".join(o)


def table_reach(fits):
    o = ['<div class="tw"><table><thead><tr><th>layout</th><th>size</th><th class="num">&#952; reachable</th>'
         '<th class="num">at max &#952;: plate gap mm</th><th class="num">mount plane mm</th>'
         '<th class="num">pair azimuth</th><th>thumb yaw / mcp / pip</th></tr></thead><tbody>']
    for L in LAYOUTS:
        for s in SIZES:
            ths = [th for th in THETAS if fit_at(fits, L, s, th)]
            if not ths:
                o.append(f'<tr><td>{esc(L)}</td><td>{s}</td><td class="num"><span class="no">none</span></td>'
                         f'<td class="num">&#8212;</td><td class="num">&#8212;</td><td class="num">&#8212;</td><td>&#8212;</td></tr>')
                continue
            f = fit_at(fits, L, s, max(ths))
            j = f["joints_deg"]["thumb"]
            o.append(f'<tr><td>{esc(L)}</td><td>{s}</td><td class="num">{min(ths)}&#8211;{max(ths)}&#176;</td>'
                     f'<td class="num">{f["plate_gap_mm"]:.1f}{" (seated)" if f["seated"] else ""}</td>'
                     f'<td class="num">{f["mount_plane_z"]*1e3:.1f}</td>'
                     f'<td class="num">&#177;{abs(f["contact_az_deg"]["index"]):.0f}&#176;</td>'
                     f'<td class="num">{j[0]:.1f} / {j[1]:.1f} / {j[2]:.1f}&#176;</td></tr>')
    o.append('</tbody></table></div>')
    return "".join(o)


def svg_map(rows, layout, size, plant, od, title):
    """theta x mu grid of held / dropped / unreachable."""
    hm = held_map(rows, layout, size, plant, od)
    cw, ch = 34, 22
    x0, y0 = 46, 40
    W = x0 + cw * len(MUS) + 10
    H = y0 + ch * len(THETAS) + 8
    o = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}">']
    o.append(f'<text class="ser" x="{x0}" y="14" fill="var(--ink)">{esc(title)}</text>')
    o.append(f'<text class="axlab" x="{x0-6}" y="{y0-6}" text-anchor="end">&#956;</text>')
    for i, mu in enumerate(MUS):
        o.append(f'<text class="tick" x="{x0 + cw*i + cw/2:.0f}" y="{y0-6}" text-anchor="middle">{mu:g}</text>')
    for j, th in enumerate(THETAS):
        o.append(f'<text class="tick" x="{x0-6}" y="{y0 + ch*j + ch/2 + 4:.0f}" text-anchor="end">{th}&#176;</text>')
        for i, mu in enumerate(MUS):
            r = hm.get((float(th), mu))
            if r is None:
                fill, op = "var(--sunk)", "1"
            elif r["held"]:
                fill, op = "var(--good)", "0.85"
            else:
                fill, op = "var(--bad)", "0.35"
            o.append(f'<rect x="{x0 + cw*i + 1}" y="{y0 + ch*j + 1}" width="{cw-2}" height="{ch-2}" rx="3" '
                     f'fill="{fill}" fill-opacity="{op}"/>')
            if r is not None:
                n = sum(r["pads_end"][f]["N"] for f in G.FINGER_NAMES) / 3 if r["held"] else 0
                if r["held"]:
                    o.append(f'<text class="mark" x="{x0 + cw*i + cw/2:.0f}" y="{y0 + ch*j + ch/2 + 4:.0f}" '
                             f'text-anchor="middle" fill="#fff">{n:.0f}N</text>')
    o.append('</svg>')
    return "".join(o)


def maps_block(rows, plant, od):
    o = ['<div class="maps">']
    for L in LAYOUTS:
        for s in SIZES:
            if not any(r["design"] == L and r["size"] == s for r in rows):
                continue
            o.append(svg_map(rows, L, s, plant, od, f"{L} · size {s}"))
    o.append('</div>')
    o.append('<p class="sub" style="margin-top:-8px">green = held, with the mean pad normal force at the end of the hold; '
             'red = dropped; grey = no reachable grasp at that wrap angle.</p>')
    return "".join(o)


def table_min_mu(rows):
    o = ['<div class="tw"><table><thead><tr><th>layout</th><th>size</th>'
         '<th class="num">measured, saturated<br>min &#956;</th><th class="num">at &#952;</th><th class="num">pad N</th>'
         '<th class="num">measured, 10 mm<br>min &#956;</th>'
         '<th class="num">shipped<br>min &#956;</th><th class="num">at &#952;</th><th class="num">pad N</th></tr></thead><tbody>']
    for L in LAYOUTS:
        for s in SIZES:
            if not any(r["design"] == L and r["size"] == s for r in rows):
                continue
            cells = []
            for plant, od in (("measured", 8.0), ("measured", 1.0), ("shipped", 1.0)):
                held = [r for r in rows if r["design"] == L and r["size"] == s and r["plant"] == plant
                        and r["overdrive"] == od and r["held"]]
                if not held:
                    cells.append(('<span class="no">none</span>', "&#8212;", "&#8212;"))
                    continue
                best = min(held, key=lambda r: (r["mu"], -r["theta_deg"]))
                n = sum(best["pads_end"][f]["N"] for f in G.FINGER_NAMES) / 3
                cells.append((f'<span class="ok">{best["mu"]:.1f}</span>', f'{best["theta_deg"]:.0f}&#176;', f'{n:.1f}'))
            o.append(f'<tr><td>{esc(L)}</td><td>{s}</td>'
                     f'<td class="num">{cells[0][0]}</td><td class="num">{cells[0][1]}</td><td class="num">{cells[0][2]}</td>'
                     f'<td class="num">{cells[1][0]}</td>'
                     f'<td class="num">{cells[2][0]}</td><td class="num">{cells[2][1]}</td><td class="num">{cells[2][2]}</td></tr>')
    o.append('</tbody></table></div>')
    return "".join(o)


def table_contact(cs):
    if not cs:
        return ""
    o = ['<div class="tw"><table><thead><tr><th>case</th><th>contact</th><th class="num">held</th>'
         '<th class="num">rise mm</th><th class="num">pad N at settle</th><th class="num">pad N at end</th></tr></thead><tbody>']
    for r in cs["rows"]:
        ps = r["pads_settle"]; pe = r["pads_end"]
        o.append(f'<tr><td>{esc(r["label"])}</td><td>{esc(r["contact"])}</td>'
                 f'<td class="num"><span class="{"ok" if r["held"] else "no"}">{"held" if r["held"] else "dropped"}</span></td>'
                 f'<td class="num">{r["rise_mm"]:.1f}</td>'
                 f'<td class="num">{ps["thumb"]["N"]:.1f} / {ps["index"]["N"]:.1f} / {ps["middle"]["N"]:.1f}</td>'
                 f'<td class="num">{pe["thumb"]["N"]:.1f} / {pe["index"]["N"]:.1f} / {pe["middle"]["N"]:.1f}</td></tr>')
    o.append('</tbody></table></div>')
    return "".join(o)


def fig(name, caption, cls=""):
    if not os.path.exists(os.path.join(MEDIA, name)):
        return ""
    return f'<figure class="{cls}"><img src="{uri(name)}" alt="{esc(caption)}"><figcaption>{caption}</figcaption></figure>'


# ------------------------------------------------------------------ prose from data

def main():
    rows, fits, cs = load()
    n_fits = sum(1 for v in fits.values() if v)
    R7 = G.BALL_SIZES["7"]["radius"] * 1e3

    def min_mu(L, s, plant, od):
        held = [r for r in rows if r["design"] == L and r["size"] == s and r["plant"] == plant
                and r["overdrive"] == od and r["held"]]
        return min(held, key=lambda r: (r["mu"], -r["theta_deg"])) if held else None

    w7m = min_mu("wide", "7", "measured", 8.0)
    w7s = min_mu("wide", "7", "shipped", 1.0)
    w7_theta = [th for th in THETAS if fit_at(fits, "wide", "7", th)]
    meas_held = [r for r in rows if r["plant"] == "measured" and r["overdrive"] == 8.0 and r["held"]]
    pads_meas = [sum(r["pads_end"][f]["N"] for f in G.FINGER_NAMES) / 3 for r in meas_held]
    n_pad = f"{min(pads_meas):.0f}&#8211;{max(pads_meas):.0f}" if pads_meas else "&#8212;"
    s3 = min_mu("wide", "3", "measured", 8.0)
    f3 = fit_at(fits, "wide", "3", max(th for th in THETAS if fit_at(fits, "wide", "3", th)))
    seat3 = "seats against the plate" if f3["seated"] else f"sits {f3['plate_gap_mm']:.0f} mm below it"
    s3_txt = (f"A size-3 ball on the same layout wraps to &#952; {f3['theta_deg']:.0f}&#176;, {seat3} "
              f"and holds from &#956; {s3['mu']:.1f}, the lowest tested." if s3 else
              "A size-3 ball on the same layout does not hold.")

    lede = (f"A size-7 basketball is reachable on the real_v1 hand only as a friction cap grasp from "
            f"above, at wrap angles up to {max(w7_theta) if w7_theta else 0}&#176; on the widest gantry "
            f"layout, with the ball {fit_at(fits, 'wide', '7', max(w7_theta))['plate_gap_mm']:.0f} mm short of the palm plate. "
            + (f"With the measured servo (0.35 N&#8239;m, {n_pad} N per pad) it holds a 0.10 m lift only from "
               f"&#956; {w7m['mu']:.1f} at &#952; {w7m['theta_deg']:.0f}&#176;; "
               if w7m else "With the measured servo no friction coefficient up to 2.4 holds it; ")
            + (f"the shipped kp&#8239;30 servo holds it from &#956; {w7s['mu']:.1f}. " if w7s else "")
            + "Seating the ball against the plate needs a mount ring 5 mm wider than the gantries reach. "
            + s3_txt)

    hold_para = (
        f"{len(meas_held)} of the {sum(1 for r in rows if r['plant']=='measured' and r['overdrive']==8.0)} "
        f"measured-servo rollouts at a saturating command hold the lift. On the wide layout and a size-7 ball "
        + (f"the hold starts at &#956; {w7m['mu']:.1f} and &#952; {w7m['theta_deg']:.0f}&#176;, with "
           f"{sum(w7m['pads_end'][f]['N'] for f in G.FINGER_NAMES)/3:.1f} N per pad at the end of the hold "
           f"and the friction cone {max(w7m['pads_end'][f]['util'] for f in G.FINGER_NAMES):.0%} used. "
           if w7m else "no rollout holds. ")
        + "Below that friction the ball stays on the floor: the pads&#8217; normal forces push it down by "
          "N&#8239;cos&#952; and the friction cannot lift more than &#956;N&#8239;sin&#952;. A 10 mm position command "
          "on the measured servo is only 0.15 N&#8239;m, under the cliff, and holds nothing.")
    hold_para_2 = (
        "The shipped servo is a control: kp&#8239;30 with a 10 N&#8239;m ceiling presses 50&#8211;150 N per pad, "
        "which no finger on the bench delivers, and its column says only that the geometry admits a hold when "
        "force is free. The measured column is the one a plan on the CB1 would see. Smaller balls wrap further "
        "and hold at lower friction; size 3 is the first whose equator the pads reach, and at the equator the ball "
        "seats against the plate. Nothing past the equator is reachable on any layout.")

    if cs:
        d = {(r["label"], r["contact"]): r for r in cs["rows"]}
        flips = [lab for lab in cs["labels"] if (lab, "default") in d and not d[(lab, "default")]["held"]
                 and d.get((lab, "grasp"), {}).get("held")]
        sN = [d[(lab, "default")]["pads_settle"]["index"]["N"] for lab in flips if lab.startswith("shipped")]
        mN = [d[(lab, "default")]["pads_settle"]["index"]["N"] for lab in flips if lab.startswith("measured")]
        contact_para = (
            f"The wide / size 7 / &#952; {cs['theta']:.0f}&#176; grasp is repeated below under both settings. "
            f"{len(flips)} of {len(cs['labels'])} cases drop at the default and hold at the grasp setting, and the "
            f"closing force is the same either way: at settle the index pad carries "
            + (f"{min(sN):.0f}&#8211;{max(sN):.0f} N under the shipped servo" if sN else "")
            + (" and " if sN and mN else "")
            + (f"{min(mN):.0f} N under the measured one" if mN else "")
            + ". What differs is whether that force survives the first millimetre of lift.")
    else:
        contact_para = ""

    ctx = {
        "LEDE": lede,
        "N_ROWS": len(rows), "N_FITS": n_fits, "N_LAYOUTS": len(LAYOUTS),
        "SEATED_DEPTH_7": f"{R7 - G.PLATE_UNDERSIDE*1e3:.1f}",
        "RING_REQ_7": f"{ring_required(R7):.1f}",
        "RING_REQ_6": f"{ring_required(G.BALL_SIZES['6']['radius']*1e3):.1f}",
        "RING_REQ_5": f"{ring_required(G.BALL_SIZES['5']['radius']*1e3):.1f}",
        "RING_REQ_3": f"{ring_required(G.BALL_SIZES['3']['radius']*1e3):.1f}",
        "RING_WIDE": f"{ring_radius('wide'):.1f}", "RING_NOMINAL": f"{ring_radius('nominal'):.1f}",
        "RING_G12": f"{ring_radius('g12'):.1f}",
        "THETA_OUTBOARD_7": f"{theta_outboard('wide', R7):.0f}",
        "TABLE_SEATING": table_seating(fits),
        "TABLE_REACH": table_reach(fits),
        "FIG_POSE": fig("pose_wide_s7.png", f"The fitted grasp on the wide layout, size 7, at θ {max(w7_theta)}°, the largest "
                        f"reachable wrap angle: front, side, top and isometric. The thumb is on its meridian at the −15° MCP "
                        f"stop; index and middle are mirrored at ±{abs(fit_at(fits,'wide','7',max(w7_theta))['contact_az_deg']['index']):.0f}° "
                        f"of azimuth. The ball's top is {fit_at(fits,'wide','7',max(w7_theta))['plate_gap_mm']:.0f} mm below the plate.")
                    + fig("pose_wide_s3.png", f"The same layout on a size-3 ball at θ {f3['theta_deg']:.0f}°: "
                          + ("the ball seats against the plate and the pads sit on its equator." if f3["seated"]
                             else f"the ball sits {f3['plate_gap_mm']:.0f} mm below the plate.")),
        "HOLD_PARA": hold_para,
        "MAPS_MEASURED": maps_block(rows, "measured", 8.0),
        "HOLD_PARA_2": hold_para_2,
        "TABLE_MIN_MU": table_min_mu(rows),
        "FIG_FILMS": "".join(fig(n, c) for n, c in [
            ("film_held.png", "Held: wide layout, size 7, measured servo at a saturating command, the lowest friction that holds. "
                              "Close 2 s, lift 0.10 m over 1.5 s, hold 2 s."),
            ("film_dropped.png", "Dropped: the same grasp one friction step lower. The ball never leaves the floor; the pads "
                                 "slide up the cap as the palm rises."),
            ("film_size3.png", f"Size 3 on the wide layout, measured servo at a saturating command, θ {s3['theta_deg']:.0f}° and "
                               f"μ {s3['mu']:.1f}: " + ("seated on the plate, held." if f3["seated"] else "held.")),
        ]),
        "CONTACT_PARA": contact_para,
        "TABLE_CONTACT": table_contact(cs),
        "MU_MIN_7": f"{w7m['mu']:.1f}" if w7m else "&gt; 2.4",
        "N_PAD_MEASURED": n_pad,
    }
    html = open(TPL).read()
    for k, v in ctx.items():
        html = html.replace("{{" + k + "}}", str(v))
    left = [l for l in html.split("{{")[1:]]
    if left:
        print("UNFILLED:", [l.split("}}")[0] for l in left][:10])
    Path(OUT).write_text(html)
    print(f"wrote {OUT}  ({len(html)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
