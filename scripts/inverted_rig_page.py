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
DATA, MEDIA, CAD, PRINT = D / "data", D / "media", D / "cad", D / "print"
BEAM = DATA / f"{R.STEM}_seat_beam.json"
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
        zd = tz - fp["t"]
        zn = zd - fp["neck_l"]
        zc0 = zn - (sr - fp["neck_r"]) / math.tan(math.radians(30.0))
        _rect(ax, -sr, sr, fl["z1"], zc0, "#d9cfe9")
        _poly(ax, [(-sr, zc0), (sr, zc0), (fp["neck_r"], zn), (-fp["neck_r"], zn)], "#d9cfe9")
        Rt = R._flat_R(tool)
        turn = lambda pts: R._turn_flat(np.array([[x, 0.0, z] for x, z in pts]), Rt, tz)[:, [0, 2]]   # noqa: E731
        neck = turn([(-fp["neck_r"], zd), (fp["neck_r"], zd)])
        _poly(ax, [(-fp["neck_r"], zn), (fp["neck_r"], zn), tuple(neck[1]), tuple(neck[0])], "#d9cfe9")
        _poly(ax, turn([(-fp["r"], zd), (fp["r"], zd), (fp["r"], tz - fp["ch"]), (fp["r"] - fp["ch"], tz),
                        (-fp["r"] + fp["ch"], tz), (-fp["r"], tz - fp["ch"])]), "#d9cfe9")
    ax.axhline(93.36, color="#c0392b", ls=(0, (4, 3)), lw=1.0, zorder=1)
    ax.set_aspect("equal")
    ax.set_xlim(-48, 48)
    ax.set_ylim(-40, 128)


def fig_probe(out):
    """The whole stack with the 5 mm sphere, and the other tools' ends at the same scale."""
    plt = P.plt_setup()
    fig = plt.figure(figsize=(10.5, 7.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=(1.3, 1.0, 1.0))
    ax0 = fig.add_subplot(gs[:, 0])
    axes = [fig.add_subplot(gs[r, c]) for r in (0, 1) for c in (1, 2)]
    names = dict(sphere5="5 mm sphere", sphere2="2 mm sphere", rod="Half rod R 12.5, end on", flat="Flat, 14 mm",
                 flat30="Flat at 30 deg, 14 mm")
    probe_section(ax0, "sphere5")
    ax0.set_title(names["sphere5"], fontsize=14)
    ax0.set_xlabel("x$_M$ (mm)", fontsize=13)
    ax0.set_ylabel("z$_M$ along the tool axis (mm)", fontsize=13)
    for z, txt in ((-17.5, "HEX-E"), (7.5, "adapter disc"), (21.5, "foot"), (36, "column"), (67, "flange"), (84, "stem")):
        ax0.annotate(txt, (-36 if z < 30 else -14, z), (-47, z), fontsize=13, va="center", ha="left", zorder=6,
                     bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.5))
    ax0.annotate("homing\nsocket", (14.5, 49), (21, 28), fontsize=13, ha="left", arrowprops=dict(arrowstyle="-", lw=0.8))
    ax0.text(46, 95.5, "finger's pad apex\non mount v7", fontsize=13, ha="right", va="bottom", color="#c0392b")
    for ax, tool in zip(axes, [t for t in R.TOOLS if t != "sphere5"]):
        probe_section(ax, tool)
        ax.set_xlim(-24, 24)
        ax.set_ylim(62, 126)
        ax.set_title(names[tool], fontsize=14)
        ax.tick_params(labelsize=12)
    for ax in axes[2:]:
        ax.set_xlabel("x$_M$ (mm)", fontsize=13)
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
    rx0, rx1, _, _, rt = g["rail"]
    _rect(a1, x0, x1, y0, y1, "0.93", z=0)
    _rect(a1, rx0, rx1, y0, y1, "0.87", z=0)
    _rect(a2, x0, x1, 0, tb, "0.85", z=0)
    _rect(a2, rx0, rx1, 0, rt, "0.85", z=0)
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
        a1.text(x1 + 3, ap[1], f"seat {i + 1}", va="center", fontsize=12)
    ap = g["seats"][1]["apex"]
    a2.plot(ap[0], ap[2], "o", ms=5, color="#c0392b", zorder=6)
    a2.annotate("pad's apex", (ap[0], ap[2]), (ap[0] - 70, ap[2] + 22), fontsize=12,
                arrowprops=dict(arrowstyle="-", lw=0.8))
    a1.annotate("", (-150, -122), (-112, -122), arrowprops=dict(arrowstyle="<-", lw=1.0))
    a1.text(-131, -114, "robot", ha="center", fontsize=12)
    for ax, (xl, yl), lab in ((a1, ((-160, 175), (-135, 135)), ("x (mm), away from the robot", "y (mm)")),
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


FULL = [True]                                          # False when a run at 1.5 mm spacing stands in for the full pad


def load_full(tool, orient):
    """The rows of the run over all of the pad's points (its parts merged); without it, the run at 1.5 mm spacing."""
    rows = []
    for f in sorted(DATA.glob(f"{STEM}_reach_full_{orient}_{tool}_part*.json")):
        rows += json.loads(f.read_text())["orients"][orient]["rows"]
    if not rows:
        FULL[0] = False
        suf = "" if tool == "sphere5" else f"_{tool}"
        f = DATA / f"{STEM}_reach_h160_{orient}{suf}.json"
        rows = json.loads(f.read_text())["orients"][orient]["rows"] if f.exists() else []
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
        ("Probe", "the tool on the robot in place of the finger. It is a PLA base on mount v7's adapter disc, which stays "
                  "on the HEX-E, with an interchangeable tool screwed to its top."),
        ("Seat", "a station block of the fingertip holder (scripts/fingertip_holder.py) on a support on the plate. A tip is "
                 "screwed to it by its four M2 &#215; 8, with its board between them."),
        ("Pad point, normal", "one of the 808 points of the pad's FE surface that the plan builder sorts into face, "
                              "edges, flanks and the tip's end, with its outward normal $\\mathbf{n}$."),
        ("Press", "the sphere's centre moves along a push direction $\\mathbf{d}$, from 1.5 mm short of contact to 1.0 mm "
                  "past it. For a normal press $\\mathbf{d} = -\\mathbf{n}$."),
        ("Push angle", "the angle $\\theta$ between $\\mathbf{d}$ and $-\\mathbf{n}$. Six pushes at $\\theta$ = 20 and 35 deg "
                       "are tried at each point, their tangential parts 60 deg apart. The sphere does not slip while "
                       "$\\tan\\theta$ stays under the friction coefficient of TPU on PLA."),
        ("Stem direction", "the unit vector $\\mathbf{u}$ from the sphere's centre toward the HEX-E. A sphere's contact "
                           "does not depend on $\\mathbf{u}$. The search tries about 250 stem directions per press and "
                           "takes the first that passes."),
        ("Twist", "the pressed sphere turned 15 deg each way about $\\mathbf{n}$ through its centre, at its depth."),
        ("Reached", "a press, push or twist whose rest and deep poses pass every check. The UR5 stays on the homing "
                    "pose's branch, with joint 5 in 35 to 97 deg and joint 6 in -62 to +45 deg. The elbow stays 30 deg "
                    "from straight and the wrist 150 mm from joint 1's axis. The HEX-E's cable port stays 15 mm from the "
                    "arm. The clearances are 8 mm for the robot's links, 6 for the HEX-E, 3 for the adapter and the probe "
                    "base, 1.0 for the tool's stem and 0.3 for its sphere. They are taken from the plate, the seats, the "
                    "three held tips, the 2x4s and the table."),
        ("Finger on the robot", "the reach of 2026-10-27 (scripts/ur5_tip_calibration_trimmed_cap_reach.py): presses of "
                                "the tall indenter's 5 mm sphere on the robot's tip with its trimmed cap. The finger's "
                                "margin is 0.8 mm and joint 6 goes to +45 deg. A point counts as reached when one of 13 "
                                "turns about the vertical passes."),
    ]
    return '<dl class="terms">' + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in items) + "</dl>"


def probe_text(parts):
    p = R.PROBE
    pr = parts["parts"]
    m_base = pr["probe_base"]["mass_g_solid"]
    tools = {t: pr[f"tool_{t}"]["mass_g_solid"] for t in R.TOOLS}
    rows = [["Probe base", f"{pr['probe_base']['volume_cm3']:.1f}", f"{m_base:.0f}",
             f"{pr['probe_base']['com_mm'][2]:.1f}", "v7's foot and homing socket, the tool interface"]]
    names = dict(sphere5="5 mm sphere", sphere2="2 mm sphere", rod="Half rod R 12.5", flat="Flat face, 14 mm",
                 flat30="Flat face at 30 deg, 14 mm")
    for t in R.TOOLS:
        q = pr[f"tool_{t}"]
        rows.append([f"Tool: {names[t]}", f"{q['volume_cm3']:.1f}", f"{q['mass_g_solid']:.0f}", f"{q['com_mm'][2]:.1f}",
                     "flange, stem and end; two M3 &#215; 10"])
    return (
        f"<p>The probe base takes the standoff fork's place on mount v7's adapter disc. The disc stays on the HEX-E with "
        f"its four M6 &#215; 20. The base's foot is v7's: its four M3 &#215; 12 go into the disc's brass inserts, and the "
        f"M6 heads sit in pockets with a hex key's way over each. Changing the fork for the probe takes four M3 screws. "
        f"The column has v7's homing socket at z<sub>M</sub> 49 on its +x<sub>M</sub> face. The plate's keys therefore "
        f"home the probe as they home the fork, with the same flange-to-mount frame T<sub>EM</sub>.</p>"
        f"<p>A tool sits on a drafted square boss on the column's top, 10 mm across and 3 mm tall at 20 deg. Two M3 "
        f"&#215; 10 in self-forming holes fasten it. The boss sets the tool's place and turn, and the faces around it set "
        f"its height. The column has four pilot holes, so a tool can go on at four turns 90 deg apart. Every tool's end "
        f"stands at z<sub>M</sub> {p['tip_z']:.0f} mm, 27 mm past the finger's pad apex on mount v7 (Figure 2). The 5 mm "
        f"sphere and its neck are the tall indenter's. The half rod is the plate's rod, its axis across the tool axis. "
        f"The flat face is a disc 14 mm across and 3 mm thick on a 9 mm neck. A second flat tool has the same disc "
        f"turned 30 deg about its face's centre. A 2 mm sphere reaches closer to the cassette's walls (section 5).</p>"
        f"<p>With the 5 mm sphere and the adapter disc, the tool past the HEX-E weighs about "
        f"{m_base + tools['sphere5'] + 55:.0f} g. The fork with the DIP link and the tip weighs 403 g, so the program's "
        f"payload must change. A beam estimate of the PLA stem and column, at 2.5 GPa, moves the 5 mm sphere's centre "
        f"5.8 &#181;m/N across the tool axis. The 2 mm sphere's centre moves 9.3 &#181;m/N. The HEX-E is the softer part: "
        f"its data sheet gives 2.5 deg at 10 N m. Taken at the sensor's mid-height, that moves the sphere's centre "
        f"77 &#181;m/N, 132 mm away. The finger's pad apex, 111 mm away, moves 54 &#181;m/N.</p>"
        + P.table(["Part", "Volume (cm&#179;)", "Mass, solid PLA (g)", "Centre of mass, z<sub>M</sub> (mm)", "What it is"],
                  rows, (1, 2, 3), "The probe's printed parts in the mount frame M, from the build's parts file.", 1))


def plate_text(g):
    lay = g["layout"]
    st = g["seats"]
    rx0, rx1, _, _, rt = g["rail"]
    ys = ", ".join("%.0f" % s_["apex"][1] for s_ in st)
    bm = json.loads(BEAM.read_text())["holds"][lay["orient"]] if BEAM.exists() else None
    stiff = (f" A beam estimate in PLA, printed with five walls and 25 % infill, puts the pad's apex at "
             f"{bm['printed']['z']:.1f} &#181;m/N along a face press, which loads the column along its axis. Across it, "
             f"the apex moves {bm['printed']['y']:.1f} &#181;m/N toward the flanks and {bm['printed']['x']:.1f} &#181;m/N "
             f"toward the tip's end." if bm else "")
    return (
        f"<p>Plate v2e keeps v2d's base, 220 &#215; 236 mm and 6 mm thick on wood screws. It keeps v2d's three homing "
        f"keys at the front and its rod at ({lay['rod']['c'][0]:.0f}, {lay['rod']['c'][1]:.0f}) mm, its top at "
        f"{lay['rod']['top']:.0f} mm. The 5 mm sphere moves to the 12.5 mm sphere's place in v2d, "
        f"({lay['sphere5']['c'][0]:.0f}, {lay['sphere5']['c'][1]:.0f}) mm, with its top at {lay['sphere5']['top']:.0f} mm. "
        f"The 12.5 mm sphere is left out, since the rod stands in for it (owner, 10-29). With the sphere, the rod and "
        f"the keys, the finger-on-robot suite of v3 ID 1 to ID 3 still runs on this plate.</p>"
        f"<p>The back row has three seats. Their pads' apexes stand at x {st[0]['apex'][0]:.0f} mm and y "
        f"{ys} mm, {lay['seat_h']:.0f} mm over the base's underside "
        f"(Figure 3). Each tip lies pad up with its end toward the robot (section 4 compares six holds). It is screwed "
        f"down onto a station block of the fingertip holder by its four M2 &#215; 8 into 1.9 mm holes, as on the "
        f"holder's up station. A solid column flares from the block to a footprint 35 mm deep and 44 mm wide.{stiff} "
        f"Under the seats the base thickens to a rail {rt:.0f} mm thick, from x {rx0:.0f} mm to its back edge. "
        f"There, where the columns pull up on it, the rail is eight times as stiff as the 6 mm base. It takes the four "
        f"back screws between the columns. The board's cable leaves each tip away from the robot and runs down behind its column. The plate "
        f"measures 220 &#215; 236 &#215; {g['zmax'] - 25.0:.0f} mm and fits the P1S's bed.</p>")


def face_c(bm, o):
    """The printed seat's compliance along a face press (um/N) for hold o: the apex's along the face's normal."""
    n = np.abs(np.asarray(R.ORIENT[o][0], float))
    c = bm["holds"][o]["printed"]
    return float(n @ [c["x"], c["y"], c["z"]])


def orient_table(o5, o2):
    bm = json.loads(BEAM.read_text()) if BEAM.exists() else None
    rows = []
    for o in R.ORIENT:
        if o not in o5:
            continue
        S = o5[o][0]
        n = sum(S[r]["n"] for r in REGIONS)
        tot = sum(S[r]["normal"] for r in REGIONS)
        rows.append([ORIENT_NAME[o]] + [f"{S[r]['normal']}" for r in REGIONS] + [
            f"{tot} of {n}", f"{100 * np.mean([S[r]['o35'] for r in ('face', 'edge')]):.0f}",
            f"{sum(S[r]['twist'] for r in REGIONS)}"] + ([f"{face_c(bm, o):.1f}"] if bm else []))
    n_r = {r: o5[next(iter(o5))][0][r]["n"] for r in REGIONS}
    head = (["Hold"] + [f"{REG_NAME[r]} ({n_r[r]})" for r in REGIONS]
            + ["All", "Face and edge pushes at 35 deg (%)", "Twists both ways"]
            + (["Seat along a face press (&#181;m/N)"] if bm else []))
    return P.table(head, rows, tuple(range(1, len(head))), "Pad points a normal press of the 5 mm sphere reaches, by "
                   "region, on the middle seat's tip, for six holds; the pad's points 1.5 mm apart (201 points). The "
                   "pushes column gives the share of the six pushes at 35 deg reached on the face and edges; the twists "
                   "column the points where both twists pass, joint 5 inside 35 to 97 deg. The last column is the "
                   "seat's beam estimate at the pad's apex along the face's normal, printed with five walls and 25 % "
                   "infill.", 2)


def orient_text(o5, o2):
    if not o5:
        return ""
    bm = json.loads(BEAM.read_text()) if BEAM.exists() else None
    tot = {o: sum(v[0][r]["normal"] for r in REGIONS) for o, v in o5.items()}
    tw = {o: sum(v[0][r]["twist"] for r in REGIONS) for o, v in o5.items()}
    n = sum(next(iter(o5.values()))[0][r]["n"] for r in REGIONS)
    best = sorted(tot, key=lambda o: -tot[o])
    top = [o for o in best if tot[o] == tot[best[0]]]
    rest = [o for o in best if tot[o] < tot[best[0]]]
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
    low = "; ".join(f"{ORIENT_NAME[o].lower()}, {tot[o]}" for o in rest)
    ch = R.LAYOUT["orient"]
    out = (f"<p>The reach was searched for six holds of the tip on the middle seat, with its neighbours in place "
           f"(Table 2). With the 5 mm sphere, {words[len(top)]} holds reach {tot[best[0]]} of {n} points and the other "
           f"{words[len(rest)]} reach fewer ({low}). The joints never limit a press: in a search of the joints alone, "
           f"every hold reaches every point, normal and off the normal. The clearances set the counts, and they are "
           f"set by the cassette's walls and end plate, which every hold brings along.</p>")
    if bm:
        fu, ch_c = face_c(bm, "front_up"), face_c(bm, ch)
        out += (f"<p>The holds differ in what the reach does not count. A face press bends the seat of a tip that faces "
                f"the robot with its end up. A beam estimate over the column's height puts its apex at {fu:.1f} "
                f"&#181;m/N as printed. On a tip that lies pad up, the same press loads the column along its axis at "
                f"{ch_c:.2f} &#181;m/N. The flanks bend either seat by about 8 &#181;m/N. Face presses are the most "
                f"numerous and the hardest, so the plate holds its tips pad up with the end toward the robot. "
                f"That hold also reaches {tw[ch]} points with both twists against {tw['front_up']}. Its face presses "
                f"come from above with the stem near vertical, so a twist turns the probe about the line of its own "
                f"weight. A tip goes on as on the holder's up station, its four screws driven from above.</p>")
    return out


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
    (tip_e, tip_n), tip_f = cnt(e, "tip"), cnt(fin, "tip")[0]
    (fl_e, fl_n), fl_f = cnt(e, "flank"), cnt(fin, "flank")[0]
    (fa_e, fa_n), fa_f = cnt(e, "face"), cnt(fin, "face")[0]
    (ed_e, ed_n), ed_f = cnt(e, "edge"), cnt(fin, "edge")[0]
    xs = np.array([-r["p_P"][0] - 2.05 for r in e if r["region"] == "flank" and not r["normal"]])
    scope = "all 808 points" if FULL[0] else "the 201 points 1.5 mm apart"
    return (
        f"<p>On the chosen seat, a press of either sphere reaches {NE} of the pad's {N} points (Table 3). The finger on "
        f"the robot reaches {NF}. On the tip's end the probe reaches {tip_e} of {tip_n} points and the finger "
        f"{tip_f}. On the flanks the probe reaches {fl_e} of {fl_n} and the finger {fl_f}. On the face the two reach "
        f"{fa_e} and {fa_f} of {fa_n}, and on the edges {ed_e} and {ed_f} of {ed_n} (Figure 4).</p>"
        f"<p>The finger on the robot pushes along the normal only. For shear it slides, turning 2.3 deg about the "
        f"sphere's centre. The probe also pushes 20 and 35 deg off the normal, and the last two columns of Table 3 "
        f"give the share of those pushes reached.</p>"
        f"<p>Of the flank points that no press reaches, {100 * np.mean(xs > -3.5):.0f} % lie within 3.4 mm of the cassette's "
        f"walls, at x<sub>F</sub> above -3.5 mm (Figure 5). The walls' palmar face stands at x<sub>F</sub> -0.10 mm, 3 mm "
        f"proud of the pad's flanks, and a sphere meets them before it meets the pad. The tip's end has the same limit at the cassette's end plate. The 2 mm "
        f"sphere reaches nearer to both than the 5 mm one. Its neck, 1.3 mm in radius, is the thinnest printed part of "
        f"the probe.</p>"
        + P.table(["Region", "Points", "Finger on the robot", "Probe, 5 mm sphere", "Probe, 2 mm sphere", "Probe, either",
                   "Pushes at 20 deg (%)", "Pushes at 35 deg (%)"], rows, tuple(range(1, 8)),
                  f"Pad points a normal press reaches, over {scope} of the pad's FE surface, on the middle seat's tip "
                  f"({ORIENT_NAME[R.LAYOUT['orient']].lower()}). The push columns give the share of the six pushes reached with either "
                  f"sphere, over all points of the region.", 3)
        + P.fig(4, figs["reach"], "Share of each region's pad points that a normal press reaches: the finger on the "
                "robot pressing the tall indenter (2026-10-27), and the probe on the held tip by tool.")
        + P.fig(5, figs["pads"], "The pad's points, filled where a normal press reaches them and open where none does: "
                "the finger on the robot (top) and the probe with either sphere (bottom). Each view, from the palm, from "
                "+y and from the tip's end, shows the points on its side of the pad."))


def flat_text(orient):
    def load(o, tool):
        f = DATA / f"{STEM}_reach_h160_{o}_{tool}.json"
        return summarize(json.loads(f.read_text())["orients"][o]["rows"]) if f.exists() else None

    def says(S):
        a, b = S["face"], S["edge"]
        c, d = S["flank"], S["tip"]
        return (f"{a['normal']} of {a['n']} face points and {b['normal']} of {b['n']} edge points. On the flanks it "
                f"reaches {c['normal']} of {c['n']}, and on the tip's end {d['normal']} of {d['n']}")

    fl, f30, fu = load(orient, "flat"), load(orient, "flat30"), load("front_up", "flat")
    if not fl:
        return ""
    out = (f"<p>The flat face meets the pad tangent to it, so the straight flat's stem lies along the pad's normal; its "
           f"centre may sit up to 4 mm from the contact. On the middle seat (the pad's points 1.5 mm apart) it "
           f"reaches {says(fl)}.")
    if fu and orient != "front_up":
        out += (f" On a tip that faces the robot with its end up it reached {fu['face']['normal']} of "
                f"{fu['face']['n']} face points. From above, its stem stands near vertical, and joint 5 leaves its "
                f"margin off the face's middle.")
    out += "</p>"
    if f30:
        out += (f"<p>The second flat tool has the same disc turned 30 deg about the face's centre, so its stem leans "
                f"30 deg off the normal, as a sphere's may. Its azimuth about the normal is searched in 15 deg steps, "
                f"those toward the robot first. It reaches {says(f30)}. Its face slopes 30 deg on the bed, so it "
                f"prints at the P1S's finest layer, 0.08 mm.</p>")
    return out


def plan_text(seats):
    """seats: the replays' records of seats 1 to 3 (None where missing)."""
    if not any(seats):
        return "<p>The plans' replays have not run.</p>"
    mid = seats[1] or next(x for x in seats if x)
    k, tr = mid["kinds"], mid.get("travel", {})
    rows = []
    for i, x in enumerate(seats):
        if not x:
            continue
        kk, le = x["kinds"], x["least"]
        rows.append([f"{i + 1}", str(x["n"]), str(kk.get("press", 0)), str(kk.get("oblique", 0)), str(kk.get("twist", 0)),
                     str(x["flagged"]), f"{x['j5_deg'][0]:.0f} to {x['j5_deg'][1]:.0f}",
                     f"{x['j6_deg'][0]:.0f} to {x['j6_deg'][1]:.0f}", f"{le['mount'][0]:.1f}", f"{le['stem'][0]:.2f}",
                     f"{le['tip'][0]:.2f}"])
    flagged = sum(x["flagged"] for x in seats if x)
    return (
        f"<p>A plan of probe presses was built on each seat and replayed move by move, as the path checker replays the "
        f"lab's plans (Table 4). On the middle seat it has {k.get('press', 0)} normal presses at pad points 2 mm apart "
        f"and {k.get('oblique', 0)} pushes at 20 and 35 deg off the normal at every third point. It also has "
        f"{k.get('twist', 0)} twists and {k.get('reference', 0)} reference presses. Between presses the probe backs "
        f"12 mm along its stem, moves across and goes in ({tr.get('straight', 0)} times). When the stems differ by more "
        f"than 20 deg, it backs 45 mm and makes a joint move ({tr.get('joint', 0)} times). A move is taken only when "
        f"its joints and clearances pass at every sample. Otherwise the next way is tried, the last a joint move by "
        f"way of a pose over the seat. The first move from the homing lifts the probe 30 mm out of the key, or 150 mm "
        f"when the lower lift would sweep it through a seat.</p>"
        f"<p>{'None' if flagged == 0 else flagged} of the actions on the three seats "
        f"{'comes' if flagged == 0 else 'come'} under a margin or leaves a window at any sample. The twists are held to "
        f"joint 5's margin of 35 to 97 deg, as the presses are.</p>"
        + P.table(["Seat", "Actions", "Presses", "Pushes", "Twists", "Flagged", "Joint 5 (deg)", "Joint 6 (deg)",
                   "Adapter and base (mm)", "Stem (mm)", "Sphere (mm)"], rows, tuple(range(0, 11)),
                  "The plan on each seat, replayed from 12 mm out of key 2: the actions by kind, those flagged, the joint "
                  "ranges over every sample, and the least clearances of the adapter and probe base, the tool's stem and "
                  "its sphere (the sphere's to the pad away from its contact).", 4))


def print_text():
    recs = [json.loads(f.read_text()) for f in sorted(PRINT.glob(f"{STEM}_p1s_*.json"))]
    if not recs:
        return ""
    rows = []
    for r in recs:
        for o in r["objects"]:
            st = o["settings"]
            rows.append([f"{r['plate']}", P.esc(o["label"]), o["id"], st["layer_height"],
                         f"{st['wall_loops']} walls, {st['sparse_infill_density']}", f"{o['filament_g']:.0f}"])
    plates = "; ".join(f"plate {r['plate']}, {r['time']} and {r['filament_g']:.0f} g" for r in recs)
    return (
        f"<p>The parts print in PLA on the Bambu Lab P1S, in their CAD orientation and without supports (Table 5). "
        f"Bambu Studio's command line slices the two projects to {plates}. Each part's settings are written into its "
        f"project, so a slice again in the GUI keeps them. The tools and the probe base print solid, since the force "
        f"passes through them. The plate starts 19 mm from the bed's left edge, clear of the wiper's corner, and has no "
        f"brim. The hardware, the assembly and the first session are listed in the folder's "
        f"{link(D / 'README.md', 'README.md')}.</p>"
        + P.table(["Plate", "Part", "ID", "Layer (mm)", "Walls, infill", "Filament (g)"], rows, (3, 5),
                  "The print: two plates on the P1S, the projects in print/ beside each part's STL in print orientation.",
                  5))


def program_text():
    items = [
        "A probe action: the target pad point and its normal in the held tip's frame, the push direction, the stem's "
        "direction and turn, and the force levels. The robot moves the probe's sphere against a pad that stands still. "
        "A twist turns the sphere about the pad's normal through its centre.",
        "The held board is the measured board. The program already logs the held boards (--held-board), and no board is "
        "on the robot.",
        "A registration with the roles swapped. Its unknowns are the sphere's centre in M (3) and the held pad's pose (6). "
        "They come from touch-offs of the probe on the held pad, as pass 1 now finds the pad in M and the sphere on the "
        "plate. Homing on the plate's keys gives the plate's frame, as before.",
        "The payload: about 0.16 kg past the sensing element with the 5 mm sphere, against 0.403 kg now. Each press is "
        "tared at its rest, as now. The twists turn the payload, so they need its gravity model, fitted from the rests "
        "of the plan's own presses with no sweep.",
        "The force on the pad: the opposite of the HEX-E's reading, turned into the held pad's frame by the registration.",
    ]
    return ("<p>The lab program drives the finger against fixed indenters. The inverted rig needs these changes:</p>"
            "<ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>")


def future_text():
    items = [
        "Print plates a and b, fit the probe base on the adapter disc and home on the plate's keys: the homing pose must "
        "match the fork's within 0.1 mm.",
        "Measure each seat with a dial indicator on its block while the probe presses the face at 1 to 5 N. More than "
        "1 &#181;m/N along the press, against the beam's 0.2, means the column or the rail flexes.",
        "Measure the probe's compliance at the sphere with a dial indicator at 1 to 5 N across the stem. More than "
        "20 &#181;m/N beyond the HEX-E's own calls for a thicker stem.",
        "Hover the probe at rest poses around a held tip and log the held board. This measures the field that the "
        "moving wrist and HEX-E bring to the held chips. More than 1 &#181;T between a press's rest and its depth means "
        "the tare must move to the depth.",
        "Once the program has a probe mode, run one seat's plan with v3 ID 3 held on it. Score the direction GP on "
        "presses held out of training. Within 0.3 mm and 6 % on the face, it matches the finger on the robot's sphere "
        "suite.",
        "Score what the pushes off the normal add to the contact direction. Take the force's angle error on presses "
        "more than 20 deg off the normal, with and without the pushes in training "
        "(scripts/ur5_tip_calibration_motion_value.py's folds).",
        "Search the half rod's reach as the spheres' and the flat's: its axis must lie across the pad's normal, so the "
        "search runs over the rod's turn about the normal and its tilt about its own axis.",
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
    lede = (f"A probe on the HEX-E, pressing a fingertip held on the plate, reaches {NE} of the pad's {N} points. The "
            f"finger on the robot, pressing the tall indenter, reaches {NF}. On the tip's end the probe reaches {te} of "
            f"{tn} points against {tf}, and on the flanks {fe} of {fn} against {ff}. It also pushes 35 deg off the "
            f"normal, which the finger on the robot cannot: {o35:.0f} % of those pushes are reached on the face and "
            f"edges. The rest of the pad lies by the cassette's walls, where no sphere fits. The probe base replaces the "
            f"fork on mount v7's adapter disc with four M3 screws and homes on the plate's keys. Plate v2e holds three "
            f"tips pad up, where a face press loads each seat along its column. The lab program needs a probe mode "
            f"before the rig can run.")
    summary = (f"Probe on the HEX-E (5 and 2 mm spheres, half rod, two flats) pressing tips held pad up on plate v2e's "
               f"three seats: "
               f"{NE} of {N} pad points reached against {NF} for the finger on the robot; tip's end {te}/{tn} (finger {tf}), "
               f"flanks {fe}/{fn} (finger {ff}); pushes 35 deg off the normal on {o35:.0f} % of face and edge points.")
    return lede, summary


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
    seats = [json.loads(f.read_text()) if f.exists() else None
             for f in (DATA / f"{STEM}_paths_{a.orient}_seat{i}.json" for i in (1, 2, 3))]
    parts = json.loads((CAD / f"{STEM}_parts.json").read_text())
    S = [("terms", "Terms and metrics", terms()),
         ("probe", "Probe on the HEX-E", probe_text(parts) + P.fig(2, figs["probe"], "The tool stack cut through the tool "
          "axis: the HEX-E, mount v7's adapter disc, the probe base and the five tools. The dashed line marks the "
          "finger's pad apex on mount v7.")),
         ("plate", "Plate v2e with three tip seats", plate_text(g) + P.fig(3, figs["plate"], "Plate v2e in plan and from "
          "-y: the keys at the front, the rod, the 5 mm sphere and the three seats. The tips (dark) lie pad up, each "
          "end toward the robot. The rail under the seats is shaded darker.")),
         ("orient", "Reach and seat compliance for six holds of the tip", orient_text(o5, o2) + orient_table(o5, o2)),
         ("compare", "Probe presses against the finger on the robot", compare_text(fin, full, figs) + flat_text(a.orient)),
         ("plan", "Plans of probe presses on the three seats", plan_text(seats)),
         ("print", "Print files", print_text()),
         ("program", "What the lab program needs for the probe", program_text()),
         ("future", "Future work", future_text())]
    film_mp4 = MEDIA / f"{STEM}_film.mp4"
    film_html = P.film(1, film_mp4, MEDIA / f"{STEM}_film_poster.png",
                       "The probe on the HEX-E pressing the middle seat's tip on plate v2e. It presses the face, an edge, a "
                       "flank and the tip's end, then pushes 35 deg off the normal and twists. Each move starts where the "
                       "previous one ended, as in the replay. Left, the cell from -y; right, close to the seat.") if film_mp4.exists() else ""
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
    ap.add_argument("--orient", default=R.LAYOUT["orient"])
    ap.add_argument("--figures-only", action="store_true")
    build(ap.parse_args())
