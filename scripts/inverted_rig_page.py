#!/usr/bin/env python3
"""Page on the inverted calibration rig for the UR5 cell: a probe on the HEX-E presses fingertips held on indenter plate
v2e (owner, 2026-10-29 13:0x: "designing the adapter, modifying the v2d plate one more time, running the plan would be
helpful"). Reads what scripts/ur5_inverted_rig.py wrote (the reach by seat orientation, the plan replay, the CAD parts)
and the finger-on-robot reach of 2026-10-27 (docs/experiments/20261026-tall_indenter/data/20261027-trimmed_cap_reach.json),
draws the figures into media/ and writes docs/experiments/20261029-inverted_calibration_rig/20261029-inverted_rig.html.

    .venv/bin/python scripts/inverted_rig_page.py [--reach FILE] [--paths FILE]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ur5_inverted_rig as R  # noqa: E402
import ur5_tip_calibration_page as P  # noqa: E402   (esc, fig, film, table, plt_setup, TEX_RE)

D = R.OUT
DATA, MEDIA, CAD = D / "data", D / "media", D / "cad"
STEM = R.STEM
PAGE = D / f"{STEM}.html"
TEMPLATE = ROOT / "scripts/inverted_rig_page.template.html"
TEX_CACHE = D / "texsvg_cache.json"
EXP = ROOT / "docs/experiments"
L = dict(plate=EXP / "20261028-indenter_plate_layout/20261028-indenter_plate_layout.html",
         motions=EXP / "20261029-calibration_motions/20261029-calibration_motions_v3.html",
         mount=EXP / "20261025-ur5_calibration_mount_v7", holder=EXP / "20261027-fingertip_holder/README.md",
         tall=EXP / "20261026-tall_indenter/README.md", v3=EXP / "20261027-v3_tip_calibration_and_crosstalk/README.md",
         state=EXP / "20261022-ur5_tip_calibration/lab/STATE.md", queue=EXP / "20261022-ur5_tip_calibration/QUEUE.md",
         status=EXP / "20261018-printed_fingertip_status/20261018-printed_fingertip_and_dip_link_status.html")
REGIONS = ("face", "edge", "flank", "tip")
REG_NAME = dict(face="Face", edge="Edges", flank="Flanks", tip="Tip's end")
ORIENT_NAME = dict(up_y="Pad up, end sideways", up_xm="Pad up, end toward the robot", front_up="Pad toward the robot, end up",
                   front_y="Pad toward the robot, end sideways", tilt_up="Pad 45 deg up from the robot, end up",
                   side_up="Pad sideways, end up")
FINGER_KEY = "trimmed_finger0.8_j6max45"
PLA = "#b7a4d6"          # the fixtures' purple in the cell's renders
DARK = "#4a3a6a"


def link(p, text):
    return f'<a href="{P.esc(os.path.relpath(p, D))}">{text}</a>'


def code(s):
    return f"<code>{P.esc(s)}</code>"


# ------------------------------------------------------------------------------------------------ figures
def _poly(ax, pts, fc, ec="0.2", lw=1.0, z=2, **kw):
    from matplotlib.patches import Polygon
    ax.add_patch(Polygon(np.asarray(pts, float), closed=True, fc=fc, ec=ec, lw=lw, zorder=z, **kw))


def _rect(ax, x0, x1, z0, z1, fc, **kw):
    _poly(ax, [(x0, z0), (x1, z0), (x1, z1), (x0, z1)], fc, **kw)


def probe_section(ax, tool):
    """The tool stack in M cut at y_M = 0: x_M across, z_M up the tool axis."""
    from matplotlib.patches import Circle, Wedge
    p = R.PROBE
    f, c, b, fl = p["foot"], p["col"], p["boss"], p["flange"]
    _rect(ax, -35, 35, -35, 0, "0.55")
    _rect(ax, -31.5, 31.5, 0, 15, "0.85")
    _rect(ax, -f["r"], f["r"], f["z0"], f["z1"], PLA)
    t = math.tan(math.radians(20.0))
    zm, half, dep = 49.0, 9.0, 5.3
    _poly(ax, [(c["x0"], f["z1"]), (c["x1"], f["z1"]), (c["x1"], zm - half), (c["x1"] - dep, zm - half + dep * t),
               (c["x1"] - dep, zm + half - dep * t), (c["x1"], zm + half), (c["x1"], c["z1"]), (c["x0"], c["z1"])], PLA)
    tb = math.tan(math.radians(b["draft"]))
    _poly(ax, [(-5, c["z1"]), (5, c["z1"]), (5 - b["h"] * tb, c["z1"] + b["h"]), (-5 + b["h"] * tb, c["z1"] + b["h"])], DARK)
    _rect(ax, -fl["r"], fl["r"], c["z1"], fl["z1"], "#d9cfe9")
    sr, tz = p["stem_r"], p["tip_z"]
    if tool in R.SPHERES:
        s = p[R.SPHERES[tool]]
        z_neck = tz - s["neck_dz"]
        z_c0 = z_neck - (sr - s["neck_r"]) / math.tan(math.radians(s["taper_deg"]))
        _rect(ax, -sr, sr, fl["z1"], z_c0, "#d9cfe9")
        _poly(ax, [(-sr, z_c0), (sr, z_c0), (s["neck_r"], z_neck), (-s["neck_r"], z_neck)], "#d9cfe9")
        ax.add_patch(Circle((0, tz - s["R"]), s["R"], fc="#d9cfe9", ec="0.2", lw=1.0, zorder=3))
    elif tool == "rod":
        r = p["rod"]
        za = tz - r["R"]
        zb0 = za - r["box_dz"]
        zh0 = zb0 - r["hull_dz"]
        _rect(ax, -sr, sr, fl["z1"], zh0, "#d9cfe9")
        _poly(ax, [(-r["hull_half"], zh0), (r["hull_half"], zh0), (r["R"], zb0), (-r["R"], zb0)], "#d9cfe9")
        _rect(ax, -r["R"], r["R"], zb0, za, "#d9cfe9")
        ax.add_patch(Wedge((0, za), r["R"], 0, 180, fc="#d9cfe9", ec="0.2", lw=1.0, zorder=3))
    else:
        fp = p["flat"]
        _rect(ax, -sr, sr, fl["z1"], tz - fp["ch"], "#d9cfe9")
        _poly(ax, [(-sr, tz - fp["ch"]), (sr, tz - fp["ch"]), (fp["r"], tz), (-fp["r"], tz)], "#d9cfe9")
    ax.axhline(93.36, color="#c0392b", ls=(0, (4, 3)), lw=1.0, zorder=1)
    ax.set_aspect("equal")
    ax.set_xlim(-48, 48)
    ax.set_ylim(-40, 128)


def fig_probe(out):
    plt = P.plt_setup()
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 6.2), sharey=True, constrained_layout=True)
    names = dict(sphere5="5 mm sphere", sphere2="2 mm sphere", rod="Half rod R 12.5,\nend on", flat="Flat face,\n14 mm across")
    for ax, tool in zip(axes, R.TOOLS):
        title = names[tool]
        probe_section(ax, tool)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("x$_M$ (mm)", fontsize=13)
    axes[0].set_ylabel("z$_M$ along the tool axis (mm)", fontsize=13)
    ax = axes[0]
    for z, txt in ((-17.5, "HEX-E"), (7.5, "adapter disc"), (21.5, "foot"), (36, "column"), (67, "flange"), (84, "stem")):
        ax.annotate(txt, (-36 if z < 30 else -14, z), (-47, z), fontsize=12, va="center", ha="left", zorder=6,
                    bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.5))
    ax.annotate("homing socket", (14.5, 49), (20, 30), fontsize=12, ha="left", arrowprops=dict(arrowstyle="-", lw=0.8))
    axes[1].text(46, 95.5, "finger's pad apex\non mount v7", fontsize=12, ha="right", va="bottom", color="#c0392b")
    fig.savefig(out, dpi=110)
    plt.close(fig)


def tip_outline_A(T):
    """The held tip's box (cassette and pad, F) as corners in A."""
    lo, hi = (-10.8, -10.5, -98.46), (3.8, 10.5, -77.05)
    return R.C.apply(T, np.array([[u, v, w] for u in (lo[0], hi[0]) for v in (lo[1], hi[1]) for w in (lo[2], hi[2])]))


def _hull2(ax, pts, fc, z=3, ec="0.2", lw=1.0):
    from scipy.spatial import ConvexHull
    pts = np.asarray(pts, float)
    h = ConvexHull(pts)
    _poly(ax, pts[h.vertices], fc, z=z, ec=ec, lw=lw)


def fig_plate(out, g):
    """Plate v2e in plan (x away from the robot, y) and from -y (x, z)."""
    from matplotlib.patches import Circle
    plt = P.plt_setup()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 6.0), constrained_layout=True,
                                 gridspec_kw=dict(width_ratios=(1.0, 1.0)))
    x0, x1, y0, y1, tb = g["params"]["base"]
    _rect(a1, x0, x1, y0, y1, "0.93", z=0)
    _rect(a2, x0, x1, 0, tb, "0.85", z=0)
    kd = g["key"]
    bx, by = kd["k0"][0] + kd["margin"], kd["k0"][1] + kd["margin"]
    for i, k in enumerate(g["keys"]):
        Rz = R.IA._Rz(k["yaw"])[:2, :2]
        pts = np.array([[sx * bx, sy * by] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]) @ Rz.T + k["c"]
        _poly(a1, pts, "0.75")
        a1.text(k["c"][0] - 13, k["c"][1], f"key {i + 1}", ha="right", va="center", fontsize=12)
        _rect(a2, k["c"][0] - bx, k["c"][0] + bx, tb, k["pin_top"], "0.75", z=1)
    s = g["sphere5"]
    a1.add_patch(Circle(s["c"], s["col_r"], fc=PLA, ec="0.2", lw=1.0, zorder=3))
    a1.text(s["c"][0], s["c"][1] + s["col_r"] + 4, "5 mm sphere", ha="center", fontsize=12)
    _poly(a2, [(s["c"][0] - s["col_r"], tb), (s["c"][0] + s["col_r"], tb), (s["c"][0] + s["col_r"], s["z_col"]),
               (s["c"][0] + s["neck_r"], s["z_neck"]), (s["c"][0] - s["neck_r"], s["z_neck"]),
               (s["c"][0] - s["col_r"], s["z_col"])], PLA, z=2)
    a2.add_patch(Circle((s["c"][0], s["top"] - s["R"]), s["R"], fc=PLA, ec="0.2", lw=1.0, zorder=3))
    r = g["rod"]
    _rect(a1, r["c"][0] - r["R"], r["c"][0] + r["R"], r["c"][1] - r["half_len"], r["c"][1] + r["half_len"], PLA)
    a1.text(r["c"][0], r["c"][1] - r["half_len"] - 9, "rod", ha="center", fontsize=12)
    _poly(a2, [(r["c"][0] - r["col_r"], tb), (r["c"][0] + r["col_r"], tb), (r["c"][0] + r["R"], r["box_z0"]),
               (r["c"][0] + r["R"], r["z_axis"]), (r["c"][0] - r["R"], r["z_axis"]), (r["c"][0] - r["R"], r["box_z0"])], PLA,
          z=1)
    from matplotlib.patches import Wedge
    a2.add_patch(Wedge((r["c"][0], r["z_axis"]), r["R"], 0, 180, fc=PLA, ec="0.2", lw=1.0, zorder=1))
    for i, st in enumerate(g["seats"]):
        T = np.asarray(st["T_AF"], float)
        sup = np.asarray(st["support"], float)
        _hull2(a1, sup[:, :2], PLA, z=4)
        _hull2(a2, sup[:, [0, 2]], PLA, z=4 if i == 1 else 2)
        tip = tip_outline_A(T)
        _hull2(a1, tip[:, :2], "0.25", z=5)
        if i == 1:
            _hull2(a2, tip[:, [0, 2]], "0.25", z=5)
        ap = st["apex"]
        a1.plot(ap[0], ap[1], "o", ms=5, color="#c0392b", zorder=6)
        a1.text(ap[0] + 32, ap[1], f"seat {i + 1}", va="center", fontsize=12)
    ap = g["seats"][1]["apex"]
    a2.plot(ap[0], ap[2], "o", ms=5, color="#c0392b", zorder=6)
    a2.annotate("pad's apex", (ap[0], ap[2]), (ap[0] - 70, ap[2] + 22), fontsize=12,
                arrowprops=dict(arrowstyle="-", lw=0.8))
    a1.annotate("", (-150, -100), (-112, -100), arrowprops=dict(arrowstyle="<-", lw=1.0))
    a1.text(-131, -92, "robot", ha="center", fontsize=12)
    for ax, (xl, yl), lab in ((a1, ((-160, 150), (-135, 135)), ("x (mm), away from the robot", "y (mm)")),
                              (a2, ((-160, 150), (-5, 210)), ("x (mm), away from the robot", "z (mm) over the base's underside"))):
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
        ax.set_aspect("equal")
        ax.set_xlabel(lab[0], fontsize=13)
        ax.set_ylabel(lab[1], fontsize=13)
    a1.set_title("Plan view", fontsize=14)
    a2.set_title("From -y, the middle seat", fontsize=14)
    fig.savefig(out, dpi=110)
    plt.close(fig)


def summarize(rows, thetas=("20", "35")):
    """Per region: points, normal presses reached, twists both ways, and the share of the push directions reached."""
    out = {}
    for rg in REGIONS:
        sel = [r for r in rows if r["region"] == rg]
        if not sel:
            continue
        s = dict(n=len(sel), normal=sum(bool(r["normal"]) for r in sel),
                 twist=sum(len(r.get("twist", [])) == 2 for r in sel))
        for th in thetas:
            s[f"o{th}"] = float(np.mean([np.mean(r["oblique"][th]) for r in sel]))
            s[f"o{th}_all"] = sum(all(r["oblique"][th]) for r in sel)
        out[rg] = s
    return out


def load_orients(tool):
    """{orientation: (summary, rows)} of the reach runs at 1.5 mm spacing for one tool."""
    out = {}
    suf = "" if tool == "sphere5" else f"_{tool}"
    for o in R.ORIENT:
        f = DATA / f"{STEM}_reach_h160_{o}{suf}.json"
        if f.exists():
            d = json.loads(f.read_text())
            rows = d["orients"][o]["rows"]
            out[o] = (summarize(rows), rows)
    return out


def load_full(tool, orient):
    """The rows of the run over all of the pad's points (its parts merged), or None."""
    rows = []
    for f in sorted(DATA.glob(f"{STEM}_reach_full_{orient}_{tool}_part*.json")):
        rows += json.loads(f.read_text())["orients"][orient]["rows"]
    return sorted(rows, key=lambda r: r["i"]) or None


def finger_rows(fr):
    """The finger on the robot's reach per pad point (index into the surface): reached with any turn."""
    return {p["i"]: dict(region=p["region"], p_P=p["p_P"], normal=bool(p["turns"][FINGER_KEY])) for p in fr["points"]}


def fig_reach(out, fin, full):
    """Pad points reached by a normal press, by region: the finger on the robot and the probe's tools."""
    plt = P.plt_setup()
    x = np.arange(len(REGIONS))
    series = [("Finger on the robot, tall indenter", fin, "0.6")]
    for lab, rows, col in (("Probe, 5 mm sphere", full.get("sphere5"), "#b7a4d6"), ("Probe, 2 mm sphere", full.get("sphere2"), "#8e7cc3"),
                           ("Probe, either sphere", full.get("either"), DARK)):
        if rows:
            series.append((lab, rows, col))
    w = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(10.5, 4.9), constrained_layout=True)
    for j, (lab, rows, col) in enumerate(series):
        v = []
        for rg in REGIONS:
            sel = [r for r in rows if r["region"] == rg]
            v.append(100.0 * sum(bool(r["normal"]) for r in sel) / max(1, len(sel)))
        ax.bar(x + (j - (len(series) - 1) / 2) * w, v, w, color=col, ec="0.2", lw=0.6, label=lab)
        for xi, vi in zip(x, v):
            ax.text(xi + (j - (len(series) - 1) / 2) * w, vi + 1.0, f"{vi:.0f}", ha="center", va="bottom", fontsize=11)
    ax.set_xticks(x, [REG_NAME[r] for r in REGIONS], fontsize=13)
    ax.set_ylabel("Pad points reached by a press (%)", fontsize=13)
    ax.set_ylim(0, 112)
    ax.legend(fontsize=12, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    fig.savefig(out, dpi=110)
    plt.close(fig)


def fig_pads(out, fin, probe):
    """The pad's points in P from the palm (y, z), from +y (x, z) and from the tip's end (y, x), filled where a normal
    press reaches them: the finger on the robot and the probe with either sphere."""
    plt = P.plt_setup()
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 8.2), constrained_layout=True)
    views = (("from the palm", (1, 2), lambda p: p[0] > 3.0, ("y$_P$ (mm)", "z$_P$ (mm)")),
             ("from +y", (0, 2), lambda p: p[1] > 1.5, ("x$_P$ (mm)", "z$_P$ (mm)")),
             ("from the tip's end", (1, 0), lambda p: p[2] > 15.0, ("y$_P$ (mm)", "x$_P$ (mm)")))
    for i, (name, rows) in enumerate((("Finger on the robot", fin), ("Probe, either sphere", probe))):
        P_ = np.array([r["p_P"] for r in rows])
        ok = np.array([bool(r["normal"]) for r in rows])
        for j, (vt, (a, b), sel, (xl, yl)) in enumerate(views):
            ax = axes[i][j]
            m = np.array([sel(p) for p in P_])
            ax.scatter(P_[m & ok, a], P_[m & ok, b], s=14, c=DARK, label="reached")
            ax.scatter(P_[m & ~ok, a], P_[m & ~ok, b], s=14, facecolors="none", edgecolors="#c0392b", lw=0.8,
                       label="not reached")
            ax.set_aspect("equal")
            ax.set_title(f"{name}, {vt}", fontsize=13)
            ax.set_xlabel(xl, fontsize=13)
            ax.set_ylabel(yl, fontsize=13)
    axes[0][0].legend(fontsize=12, loc="lower left", framealpha=0.9)
    fig.savefig(out, dpi=110)
    plt.close(fig)


def either(a_rows, b_rows):
    """Per point: reached by either tool (normal, twists and each push direction)."""
    b = {r["i"]: r for r in b_rows}
    out = []
    for r in a_rows:
        s = b.get(r["i"])
        if s is None:
            continue
        o = {th: [x or y for x, y in zip(r["oblique"][th], s["oblique"][th])] for th in r["oblique"]}
        out.append(dict(i=r["i"], region=r["region"], p_P=r["p_P"], normal=bool(r["normal"] or s["normal"]),
                        twist=r.get("twist", []) if len(r.get("twist", [])) >= len(s.get("twist", [])) else s.get("twist", []),
                        oblique=o))
    return out


# ------------------------------------------------------------------------------------------------ text
def pct(a, b):
    return f"{100.0 * a / max(1, b):.0f}"


def cnt(rows, rg, key="normal"):
    sel = [r for r in rows if r["region"] == rg]
    return sum(bool(r[key]) for r in sel), len(sel)


def terms():
    items = [
        ("Probe", "the tool on the robot in place of the finger: a PLA base on mount v7's adapter disc, which stays "
                  "on the HEX-E, and an interchangeable tool screwed to the base's top."),
        ("Seat", "a station block of the fingertip holder (scripts/fingertip_holder.py) on a support on the plate; a tip "
                 "is screwed to it by its four M2 &#215; 8 with its board between them."),
        ("Pad point, normal", "one of the 808 points of the pad's FE surface that the plan builder sorts into face, "
                              "edges, flanks and the tip's end, with its outward normal $\\mathbf{n}$."),
        ("Press", "the sphere's centre moves along a push direction $\\mathbf{d}$ from 1.5 mm short of contact to 1.0 mm "
                  "past it. For a normal press $\\mathbf{d} = -\\mathbf{n}$."),
        ("Push angle", "the angle $\\theta$ between $\\mathbf{d}$ and $-\\mathbf{n}$. Six pushes at $\\theta$ = 20 and 35 deg "
                       "are tried at each point, their tangential parts 60 deg apart; with the friction of TPU on PLA the "
                       "sphere stays put while $\\tan\\theta$ is under the friction coefficient."),
        ("Stem direction", "the unit vector $\\mathbf{u}$ from the sphere's centre toward the HEX-E. A sphere's contact "
                           "does not depend on $\\mathbf{u}$, so the search tries about 250 stem directions per press "
                           "and takes the first that passes."),
        ("Twist", "the pressed sphere turned 15 deg each way about $\\mathbf{n}$ through its centre at its depth."),
        ("Reached", "a press (or a push, a twist) whose rest and deep poses pass every check: the UR5 on the homing "
                    "pose's branch with joint 5 in 35 to 97 deg, joint 6 in -62 to +45 deg, the elbow 30 deg from "
                    "straight, the wrist 150 mm from joint 1's axis; the HEX-E's cable port 15 mm from the arm; the "
                    "clearances robot links 8, HEX-E 6, adapter and probe base 3, the tool's stem 1.0 and its sphere "
                    "0.3 mm, from the plate, the seats, the three held tips, the 2x4s and the table."),
        ("Finger on the robot", "the reach of 2026-10-27 (scripts/ur5_tip_calibration_trimmed_cap_reach.py): presses of "
                                "the tall indenter's 5 mm sphere on the robot's tip, trimmed cap, finger margin 0.8 mm, "
                                "joint 6 to +45 deg, a point reached when one of 13 turns passes."),
    ]
    return '<dl class="terms">' + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in items) + "</dl>"


def probe_text(parts):
    p = R.PROBE
    pr = parts["parts"]
    m_base = pr["probe_base"]["mass_g_solid"]
    tools = {t: pr[f"tool_{t}"]["mass_g_solid"] for t in R.TOOLS}
    rows = [["Probe base", f"{pr['probe_base']['volume_cm3']:.1f}", f"{m_base:.0f}",
             f"{pr['probe_base']['com_mm'][2]:.1f}", "v7's foot and homing socket, the tool interface"]]
    names = dict(sphere5="5 mm sphere", sphere2="2 mm sphere", rod="Half rod R 12.5", flat="Flat face, 14 mm")
    for t in R.TOOLS:
        q = pr[f"tool_{t}"]
        rows.append([f"Tool: {names[t]}", f"{q['volume_cm3']:.1f}", f"{q['mass_g_solid']:.0f}", f"{q['com_mm'][2]:.1f}",
                     "flange, stem and end; two M3 &#215; 10"])
    body = (
        f"<p>The probe base takes the standoff fork's place on mount v7's adapter disc, which stays on the HEX-E with its "
        f"four M6 &#215; 20. Its foot is v7's: the four M3 &#215; 12 go into the disc's brass inserts, and the M6 heads "
        f"sit in pockets with a hex key's way over each. Changing the fork for the probe therefore takes four M3 screws. "
        f"The column has v7's homing socket at z<sub>M</sub> 49 on its +x<sub>M</sub> face, so the plate's keys home the "
        f"probe as they home the fork, with the same flange-to-mount frame T<sub>EM</sub>.</p>"
        f"<p>A tool sits on the column's top on a drafted square boss, 10 mm across and 3 mm tall at 20 deg, and is held by "
        f"two M3 &#215; 10 in self-forming holes. The boss locates it in x, y and its turn, and the faces around it set its "
        f"height. The column has four pilot holes, so a tool can go on in four turns 90 deg apart. Every tool's end stands "
        f"at z<sub>M</sub> {p['tip_z']:.0f} mm, 27 mm past the finger's pad apex on mount v7 (Figure 2). The 5 mm sphere "
        f"and its neck are the tall indenter's. The half rod is the plate's rod, its axis across the tool axis. The flat face "
        f"is 14 mm across. A 2 mm sphere was added for the flanks by the cassette's walls (section 5).</p>"
        f"<p>With the 5 mm sphere, the tool beyond the HEX-E weighs about {m_base + tools['sphere5'] + 55:.0f} g with the "
        f"adapter disc, against 403 g for the fork, the DIP link and the tip. The program's payload must be set to it. "
        f"A beam estimate of the PLA stem and column, at 2.5 GPa, moves the 5 mm sphere's centre 5.8 &#181;m/N across the "
        f"tool axis and the 2 mm sphere's 9.3 &#181;m/N. The HEX-E's own data-sheet compliance, 2.5 deg at 10 N m, moves "
        f"the sphere's centre 77 &#181;m/N at its 132 mm from the sensing element, against 54 &#181;m/N for the finger's "
        f"pad apex at 111 mm.</p>"
        + P.table(["Part", "Volume (cm&#179;)", "Mass, solid PLA (g)", "Centre of mass, z<sub>M</sub> (mm)", "What it is"],
                  rows, (1, 2, 3), "The probe's printed parts (the build's parts file), in the mount frame M.", 1))
    return body


def plate_text(g):
    lay = g["layout"]
    st = g["seats"]
    return (
        f"<p>Plate v2e keeps v2d's base (220 &#215; 236 mm, 6 mm thick, six wood screws), its three homing keys at the "
        f"front and its rod at ({lay['rod']['c'][0]:.0f}, {lay['rod']['c'][1]:.0f}) mm with its top at "
        f"{lay['rod']['top']:.0f} mm. The 5 mm sphere moves to where v2d had the 12.5 mm sphere, ({lay['sphere5']['c'][0]:.0f}, "
        f"{lay['sphere5']['c'][1]:.0f}) mm, its top at {lay['sphere5']['top']:.0f} mm; the 12.5 mm sphere is left out "
        f"(owner, 10-29: the rod stands in for it). These keep the finger-on-robot suite of v3 ID 1 to ID 3 possible on the "
        f"same plate.</p>"
        f"<p>The back row has three seats, their pads' apexes at x {st[0]['apex'][0]:.0f} mm, y "
        f"{', '.join(f'{s['apex'][1]:.0f}' for s in st)} mm and {lay['seat_h']:.0f} mm over the base's underside (Figure 3). "
        f"Each is a station block of the fingertip holder, 12 mm thick behind the tip's board, on a solid support down to "
        f"the base. Its four 1.9 mm holes take the tip's M2 &#215; 8, as on the holder. A tip faces the robot with its end "
        f"up, a finger pointing up (section 4 compares six ways of holding it). The plate is 220 &#215; 236 &#215; "
        f"{g['zmax'] - 25.0:.0f} mm and fits the P1S's bed.</p>")


def orient_table(o5, o2):
    rows = []
    for o in R.ORIENT:
        if o not in o5:
            continue
        S = o5[o][0]
        n = sum(S[r]["n"] for r in REGIONS)
        tot = sum(S[r]["normal"] for r in REGIONS)
        rows.append([ORIENT_NAME[o]] + [f"{S[r]['normal']}" for r in REGIONS] + [
            f"{tot} of {n}", f"{100 * np.mean([S[r]['o35'] for r in ('face', 'edge')]):.0f}",
            f"{sum(S[r]['twist'] for r in REGIONS)}"])
    n_r = {r: o5[next(iter(o5))][0][r]["n"] for r in REGIONS}
    head = (["Seat orientation"] + [f"{REG_NAME[r]} ({n_r[r]})" for r in REGIONS]
            + ["All", "Face and edge pushes at 35 deg (%)", "Twists both ways"])
    return P.table(head, rows, tuple(range(1, 8)), "Pad points a normal press of the 5 mm sphere reaches, by region, on "
                   "the middle seat's tip, for six ways of holding the tip; the pad's points 1.5 mm apart (201 points). "
                   "The pushes column gives the share of the six pushes at 35 deg reached on the face and edges; the "
                   "twists column the points where both twists pass.", 2)


def compare_text(fin, full, figs):
    e = full["either"]
    s5, s2 = full["sphere5"], full["sphere2"]
    rows = []
    for rg in REGIONS:
        nf, N = cnt(fin, rg)
        n5, _ = cnt(s5, rg)
        n2, _ = cnt(s2, rg)
        ne, _ = cnt(e, rg)
        S = summarize([r for r in e if r["region"] == rg])[rg]
        rows.append([REG_NAME[rg], str(N), f"{nf} ({pct(nf, N)} %)", f"{n5} ({pct(n5, N)} %)", f"{n2} ({pct(n2, N)} %)",
                     f"{ne} ({pct(ne, N)} %)", f"{100 * S['o20']:.0f}", f"{100 * S['o35']:.0f}"])
    NF = sum(cnt(fin, r)[0] for r in REGIONS)
    NE = sum(cnt(e, r)[0] for r in REGIONS)
    N = sum(cnt(e, r)[1] for r in REGIONS)
    tip_f, tip_n = cnt(fin, "tip")
    tip_e = cnt(e, "tip")[0]
    fl_f, fl_n = cnt(fin, "flank")
    fl_e = cnt(e, "flank")[0]
    body = (
        f"<p>On the chosen seat, a press of either sphere reaches {NE} of the pad's {N} points, against {NF} for the finger "
        f"on the robot (Table 3). The gain is on the tip's end, {tip_e} of {tip_n} points against {tip_f}, and on the flanks, "
        f"{fl_e} of {fl_n} against {fl_f}. On the face and the edges the two reach {cnt(e, 'face')[0]} and {cnt(fin, 'face')[0]} of "
        f"{cnt(e, 'face')[1]}, and {cnt(e, 'edge')[0]} and {cnt(fin, 'edge')[0]} of {cnt(e, 'edge')[1]} (Figure 4). The finger on "
        f"the robot pushes only along the normal, with slides of 2.3 deg about the sphere's centre for shear. The probe "
        f"also pushes 20 and 35 deg off the normal, and the last two columns of Table 3 give the share of those pushes "
        f"reached.</p>"
        f"<p>The points no press reaches lie by the cassette's walls and end plate (Figure 5). There the walls stand 3 mm "
        f"proud of the pad's flanks, and a sphere meets them before it meets the pad. The 2 mm sphere reaches closer to "
        f"them than the 5 mm one. Its neck, 1.3 mm in radius, is the thinnest printed part of the probe.</p>"
        + P.table(["Region", "Points", "Finger on the robot", "Probe, 5 mm sphere", "Probe, 2 mm sphere", "Probe, either",
                   "Pushes at 20 deg (%)", "Pushes at 35 deg (%)"], rows, tuple(range(1, 8)),
                  "Pad points a normal press reaches, all 808 points of the pad's FE surface, the tip held facing the "
                  "robot with its end up on the middle seat. The push columns give the share of the six pushes reached "
                  "with either sphere, over all points of the region.", 3)
        + P.fig(4, figs["reach"], "Share of each region's pad points that a normal press reaches: the finger on the "
                "robot pressing the tall indenter (2026-10-27) and the probe on the held tip, by tool.")
        + P.fig(5, figs["pads"], "The pad's points, filled where a normal press reaches them and open where none does: "
                "the finger on the robot (top) and the probe with either sphere (bottom). Seen from the palm, from +y, "
                "and from the tip's end; each view shows the points on its side of the pad."))
    return body


def plan_text(paths):
    if not paths:
        return "<p>The plan's replay has not run.</p>"
    k = paths["kinds"]
    tr = paths.get("travel", {})
    least = paths["least"]
    rows = [[g, f"{least[g][0]:.2f}", P.esc(least[g][1])] for g in ("robot", "hex_e", "mount", "stem", "tip", "port")
            if g in least]
    return (
        f"<p>A plan of {paths['n']} actions on the middle seat's tip was replayed move by move, as the path checker "
        f"replays the lab's plans: {k.get('press', 0)} normal presses at pad points 2 mm apart, {k.get('oblique', 0)} pushes "
        f"at 20 and 35 deg off the normal at every third point, {k.get('twist', 0)} twists and {k.get('reference', 0)} "
        f"reference presses. Between presses the probe backs 12 mm along its stem, moves across and goes in "
        f"({tr.get('straight', 0)} times). When the stems differ by more than 20 deg it backs 45 mm and makes a joint move "
        f"({tr.get('joint', 0)} times), by way of a pose over the seat if the joint move would leave a window "
        f"({tr.get('hub', 0)} times). {paths['flagged']} of the {paths['n']} actions come under a margin or out of a "
        f"window at any sample. Joint 5 stays in {paths['j5_deg'][0]:.0f} to {paths['j5_deg'][1]:.0f} deg and joint 6 in "
        f"{paths['j6_deg'][0]:.0f} to {paths['j6_deg'][1]:.0f} deg. The least clearances over the plan are in Table 4.</p>"
        + P.table(["Group", "Least clearance (mm)", "Part"], rows, (1,), "Least clearance of each group over the plan's "
                  "moves, every 2 mm sample of a straight move and every fifth of a joint move.", 4))


def program_text():
    items = [
        "A probe action: the target pad point and its normal in the held tip's frame, the push direction, the stem's "
        "direction and turn, and the force levels. The robot moves the probe's sphere against a pad that stands still, and a twist turns it "
        "about the pad's normal through the sphere's centre.",
        "The held board is the measured board: the program already logs the held boards (--held-board). No board is "
        "on the robot.",
        "Registration with the roles swapped: the sphere's centre in M (3 unknowns) and the held pad's pose (6) from "
        "touch-offs of the probe on the held pad, as pass 1 now finds the pad in M and the sphere on the plate. Homing on "
        "the plate's keys gives the plate's frame as before.",
        "The payload: about 0.16 kg past the sensing element with the 5 mm sphere, against 0.403 kg now. Each press is "
        "tared at its rest, as now; the twists turn the payload and need its gravity model, fitted from the rests of the "
        "plan's own presses (no sweep).",
        "The force on the pad is the opposite of the HEX-E's reading, turned into the held pad's frame by the "
        "registration.",
    ]
    return ("<p>The lab program drives the finger against fixed indenters. The inverted rig needs these changes:</p>"
            "<ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>")


def future_text():
    items = [
        "Print the probe base and the 5 mm sphere tool on the P1S in PLA, fit them on the adapter disc, and touch the "
        "plate's keys: the homing pose must match the fork's within 0.1 mm (the registration's T<sub>BE</sub> home).",
        "Measure the probe's stiffness at the sphere with the HEX-E and a dial indicator at 1 to 5 N across the stem. "
        "Over 20 &#181;m/N beyond the HEX-E's own means a thicker stem.",
        "Hover the probe at rest poses around a held tip and log the held board: the field that the moving wrist and "
        "HEX-E bring to the held chips. Over 1 &#181;T between a press's rest and its depth means the tare must move "
        "to the depth.",
        "Run one seat's plan with v3 ID 3 held on it, after the program's probe mode exists, and score the direction GP "
        "on presses held out: location within 0.3 mm and force within 6 % on the face would match the finger on the "
        "robot's sphere suite.",
        "Score what the oblique pushes add to the contact direction: the force's angle error on presses more than 20 deg "
        "off the normal, with and without the pushes in training (scripts/ur5_tip_calibration_motion_value.py's folds).",
    ]
    return "<ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>"


def lede_text(fin, full):
    e = full["either"]
    N = sum(cnt(e, r)[1] for r in REGIONS)
    NE, NF = sum(cnt(e, r)[0] for r in REGIONS), sum(cnt(fin, r)[0] for r in REGIONS)
    (te, tn), tf = cnt(e, "tip"), cnt(fin, "tip")[0]
    (fe, fn), ff = cnt(e, "flank"), cnt(fin, "flank")[0]
    S = summarize(e)
    o35 = 100 * np.mean([S[r]["o35"] for r in ("face", "edge")])
    lede = (f"A probe on the HEX-E pressing a fingertip held on the plate reaches {NE} of the pad's {N} points, against "
            f"{NF} for the finger on the robot pressing the tall indenter. On the tip's end it reaches {te} of {tn} points "
            f"(finger: {tf}) and on the flanks {fe} of {fn} (finger: {ff}). It also pushes 35 deg off the normal, which the "
            f"finger on the robot cannot: {o35:.0f} % of those pushes are reached on the face and edges. The probe base "
            f"replaces the fork on mount v7's adapter disc with four M3 screws and homes on the plate's keys; plate v2e "
            f"has three seats, each with a tip facing the robot. The rest of the pad lies by the cassette's walls, where no sphere fits; "
            f"the lab program needs a probe mode before the rig can run.")
    summary = (f"Probe on the HEX-E (5 and 2 mm spheres, half rod, flat) pressing tips held on plate v2e's three seats: "
               f"{NE} of {N} pad points reached against {NF} for the finger on the robot; tip's end {te}/{tn} (finger {tf}), "
               f"flanks {fe}/{fn} (finger {ff}); pushes 35 deg off the normal on {o35:.0f} % of face and edge points.")
    return lede, summary


def orient_text(o5, o2):
    if not o5:
        return ""
    tot = {o: sum(v[0][r]["normal"] for r in REGIONS) for o, v in o5.items()}
    n = sum(next(iter(o5.values()))[0][r]["n"] for r in REGIONS)
    best = sorted(tot, key=lambda o: -tot[o])
    return (f"<p>The reach was searched for six ways of holding the tip, on the middle seat with its neighbours in place "
            f"(Table 2). With the 5 mm sphere, {', '.join(ORIENT_NAME[o].lower() for o in best[:3])} reach "
            f"{tot[best[2]]} to {tot[best[0]]} of {n} points. The other three reach {min(tot.values())} to "
            f"{tot[best[3]]}. Facing the robot with the end up is a finger pointing up, and its block stands on a plain "
            f"upright support, so the tips on plate v2e face the robot with their ends up.</p>")


def build(a):
    import docs_index
    import retro_style
    import texsvg
    MEDIA.mkdir(parents=True, exist_ok=True)
    figs = dict(probe=MEDIA / f"{STEM}_probe.png", plate=MEDIA / f"{STEM}_plate.png", reach=MEDIA / f"{STEM}_reach.png",
                pads=MEDIA / f"{STEM}_pads.png")
    g = R.plate()
    fig_probe(figs["probe"])
    fig_plate(figs["plate"], g)
    if a.figures_only:
        return
    fr = json.loads(R.FINGER_REACH.read_text())
    fmap = finger_rows(fr)
    o5, o2 = load_orients("sphere5"), load_orients("sphere2")
    rows5, rows2 = load_full("sphere5", a.orient), load_full("sphere2", a.orient)
    full = dict(sphere5=rows5, sphere2=rows2, either=either(rows5, rows2))
    fin = [dict(fmap[r["i"]], i=r["i"]) for r in full["either"] if r["i"] in fmap]
    fig_reach(figs["reach"], fin, full)
    fig_pads(figs["pads"], fin, full["either"])
    pf = DATA / f"{STEM}_paths_{a.orient}_seat2.json"
    paths = json.loads(pf.read_text()) if pf.exists() else None
    parts = json.loads((CAD / f"{STEM}_parts.json").read_text())
    S = [("terms", "Terms and metrics", terms()),
         ("probe", "Probe on the HEX-E", probe_text(parts) + P.fig(2, figs["probe"], "The tool stack cut through the tool "
          "axis: the HEX-E, mount v7's adapter disc, the probe base and the four tools. The dashed line marks the "
          "finger's pad apex on mount v7.")),
         ("plate", "Plate v2e with three tip seats", plate_text(g) + P.fig(3, figs["plate"], "Plate v2e in plan and from "
          "-y: the keys at the front, the rod and the 5 mm sphere, and the three seats with their tips (dark) facing "
          "the robot.")),
         ("orient", "Reach on six ways of holding the tip", orient_text(o5, o2) + orient_table(o5, o2)),
         ("compare", "Probe presses against the finger on the robot", compare_text(fin, full, figs)),
         ("plan", "A plan of probe presses on one seat", plan_text(paths)),
         ("program", "What the lab program needs for the probe", program_text()),
         ("future", "Future work", future_text())]
    film_mp4 = MEDIA / f"{STEM}_film.mp4"
    film_html = P.film(1, film_mp4, MEDIA / f"{STEM}_film_poster.png",
                       "The probe on the HEX-E pressing the middle seat's tip on plate v2e: a press on the face, an edge, a "
                       "flank and the tip's end, a push 35 deg off the normal and a twist, each from the previous one as "
                       "the replay moves. Left, the cell from -y; right, close to the seat.") if film_mp4.exists() else ""
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {hd}</a></li>' for k, (sid, hd, _) in enumerate(S, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {hd}</h2>{body}</section>' for k, (sid, hd, body) in enumerate(S, 1))
    lede, summary = lede_text(fin, full)
    byline = ("Reach and replay by " + code("scripts/ur5_inverted_rig.py") + " on the cell's model (the UR5's DH model, "
              "mount v7's disc, the robot's and the tip's meshes), rows in " + link(DATA, "data/") + "; CAD in "
              + link(CAD, "cad/") + "; page by " + code("scripts/inverted_rig_page.py") + ". The finger-on-robot reach: "
              + link(R.FINGER_REACH, "20261027-trimmed_cap_reach.json") + ". Plate v2d: " + link(L["plate"], "plate layout")
              + "; motions: " + link(L["motions"], "calibration motions") + "; the cell: " + link(L["state"], "lab STATE.md") + ".")
    page = TEMPLATE.read_text()
    title = "Probe on the HEX-E pressing fingertips held on indenter plate v2e"
    for k, v in {"TITLE": title, "H1": title, "EYEBROW": "Printed fingertips &#183; 2026-10-29", "LEDE": lede,
                 "BYLINE": byline, "FILM": film_html, "TOC": toc, "SECTIONS": secs, "FOOTER": str(PAGE.relative_to(ROOT)),
                 "TEXCSS": texsvg.CSS}.items():
        page = page.replace("{{" + k + "}}", v)
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in P.TEX_RE.finditer(page)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE))
    page = P.TEX_RE.sub(lambda m: next(svgs), page)
    page = retro_style.apply(page)
    page = docs_index.tag(page, topic="fingertip", summary=summary, date="2026-10-29")
    PAGE.write_text(page)
    print(f"{PAGE}: {PAGE.stat().st_size / 1e6:.2f} MB, {len(items)} formulas", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--orient", default="front_up")
    ap.add_argument("--figures-only", action="store_true")
    build(ap.parse_args())
