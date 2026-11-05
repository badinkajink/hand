#!/usr/bin/env python3
"""Labelled drawing of the UR5 adaptor in the robot mount (hand frame A) and of the hand on the UR5 flange (tool0, E) for
the four clockings k, drawn from parts.json alone (no meshes): the adaptor and the pocket as rounded squares from their
fitted sizes, the holes from their cylinder axes, the plates, gantry frames, fingers and camera parts as bounding boxes.
Reads flange.json; writes flange.png.

    .venv/bin/python logs/20261104-cad_v3/flange/render_flange.py
"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

REPO = Path("/home/humanoid/Programs/hand")
WORK = REPO / "logs/20261104-cad_v3"
OUT = WORK / "flange"
PARTS = json.load(open(WORK / "extract/parts.json"))["parts"]
F = json.load(open(OUT / "flange.json"))
plt.rcParams.update({"font.size": 12, "font.family": "serif", "axes.titlesize": 13})
COL = dict(base="#9a9a9a", mount="#444444", adaptor="#d9822b", gantry="#2f62ad", finger="#9bb7d9", cam="#2e8b57",
           flange="#111111", conn="#c0392b", dowel="#7d3c98", plate="#1f77b4", rmhole="#c0392b")


def part(f, n):
    return next(p for p in PARTS if p["file"] == f and p["name"] == n)


def circle(c, r, z, n=72):
    a = np.linspace(0, 2 * np.pi, n + 1)
    return np.c_[c[0] + r * np.cos(a), c[1] + r * np.sin(a), np.full(n + 1, z)]


def rounded_square(c, half, rc, ang_deg, z, n=24):
    """Outline of a square of half width `half` with corner radius rc, turned by ang about its centre c."""
    pts = []
    k = half - rc
    for i, (sx, sy) in enumerate(((1, 1), (-1, 1), (-1, -1), (1, -1))):
        a0 = math.radians(90 * i)
        for a in np.linspace(a0, a0 + math.pi / 2, n):
            pts.append((sx * k + rc * math.cos(a), sy * k + rc * math.sin(a)))
    P = np.array(pts + pts[:1])
    t = math.radians(ang_deg)
    R = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    Q = P @ R.T + c
    return np.c_[Q, np.full(len(Q), z)]


def box(p, z=None):
    b = p["bbox"]
    zz = (b[2] + b[5]) / 2 if z is None else z
    return np.array([[b[0], b[1], zz], [b[3], b[1], zz], [b[3], b[4], zz], [b[0], b[4], zz], [b[0], b[1], zz]])


def to_E(T, P):
    T = np.asarray(T)
    return P @ T[:3, :3].T + T[:3, 3]


def plot(ax, P, T=None, **kw):
    Q = P if T is None else to_E(T, P)
    ax.plot(Q[:, 0], Q[:, 1], **kw)


def fill(ax, P, T=None, **kw):
    Q = P if T is None else to_E(T, P)
    ax.fill(Q[:, 0], Q[:, 1], **kw)


def main():
    ad_p = part("hand", "004")
    side = F["adaptor_details"]["side_normals_A_deg"]
    a_sq = min(side, key=lambda a: abs(a))                      # 24 deg
    bc = np.array(F["m6_bolt_circle"]["centre_A_mm"])
    half_ad = F["adaptor_details"]["half_width_mm"][0]
    half_po = F["adaptor_details"]["pocket_half_width_mm"][0]
    rc = [c["r"] for c in ad_p["cylinders"] if c["r"] > 10][0]
    geo = dict(
        adaptor=rounded_square(bc, half_ad, rc, a_sq, 27.0),
        pocket=rounded_square(bc, half_po, rc, a_sq, 27.0),
        base=box(part("hand", "002"), 1.5), mount=box(part("hand", "001"), 15.0),
        g_thumb=box(part("hand", "045")), g_index=box(part("hand", "074")), g_middle=box(part("hand", "102")),
        cam=[box(part("hand", "000")), box(part("hand", "005"))])
    fingers = {}
    for p in PARTS:
        if p["file"] == "hand" and p["chain"][0].startswith("Finger v3"):
            b = fingers.setdefault(p["chain"][0], [1e9, 1e9, 1e9, -1e9, -1e9, -1e9])
            for i in range(3):
                b[i] = min(b[i], p["bbox"][i])
                b[i + 3] = max(b[i + 3], p["bbox"][i + 3])
    geo["fingers"] = [box(dict(bbox=b)) for b in fingers.values()]
    raw = json.load(open(OUT / "flange_raw.json"))["adaptor_in_A"]
    m6 = [np.array(h["centre_A_mm"]) for h in raw["m6_holes"]]
    m6_ang = [h["angle_A_deg"] for h in raw["m6_holes"]]
    plate = raw["plate_side_holes"]
    c_A = np.array(F["flange_centre_A_mm"])
    off = F["offsets"]
    gc = off["gantry_frame_centres_A_mm"]

    fig = plt.figure(figsize=(19.5, 15.0), dpi=110)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1.0], hspace=0.55, wspace=0.16)
    axA = fig.add_subplot(gs[0, 0:2])
    axZ = fig.add_subplot(gs[0, 2:4])
    axK = [fig.add_subplot(gs[1, i]) for i in range(4)]

    # (a) footprint in A, seen from the robot side (+z_A toward the viewer)
    ax = axA
    fill(ax, geo["base"], color="#eeeeee", lw=0)
    plot(ax, geo["base"], color=COL["base"], lw=1.2)
    plot(ax, geo["mount"], color=COL["mount"], lw=1.0, ls="--")
    for P in geo["fingers"]:
        fill(ax, P, color=COL["finger"], alpha=0.45, lw=0)
    for g in ("thumb", "index", "middle"):
        plot(ax, geo["g_" + g], color=COL["gantry"], lw=1.6)
        c = gc[g]
        ax.text(c[0], c[1] + (44 if g == "thumb" else -44), f"{g} gantry", ha="center", va="center",
                color=COL["gantry"], fontsize=13, fontweight="bold")
    for P in geo["cam"]:
        plot(ax, P, color=COL["cam"], lw=1.5)
    ax.text(30, 282, "camera and\nAprilTag parts", color=COL["cam"], ha="left", va="center", fontsize=12)
    plot(ax, geo["adaptor"], color=COL["adaptor"], lw=1.6)
    plot(ax, circle(c_A, 31.5, 28), color=COL["flange"], lw=1.2, ls="--")
    ax.plot(*c_A[:2], "+", color=COL["flange"], ms=14, mew=2)
    bpc, gfc = np.array(off["base_plate_bbox_centre_A_mm"]), np.array(off["gantry_frames_P_like_centre_A_mm"])
    ax.plot(*bpc, "s", color="white", mec="k", ms=7)
    ax.plot(*gfc, "D", color=COL["gantry"], mec="k", ms=6)
    kw = dict(fontsize=12, bbox=dict(fc="white", ec="none", alpha=0.9, pad=1.5))
    ax.annotate(f"flange axis ({c_A[0]:.1f}, {c_A[1]:.1f})", c_A[:2], (-25, 160), arrowprops=dict(arrowstyle="-", lw=0.8),
                **kw)
    ax.annotate(f"base plate centre ({bpc[0]:.1f}, {bpc[1]:.2f})", bpc, (-298, 268), arrowprops=dict(arrowstyle="-", lw=0.8),
                **kw)
    ax.annotate(f"gantry frames' centre ({gfc[0]:.1f}, {gfc[1]:.2f})", gfc, (-298, -30),
                arrowprops=dict(arrowstyle="-", lw=0.8), **kw)
    for v, lab, dxy in (((40, 0), r"$x_A$", (47, 0)), ((0, 40), r"$y_A$", (0, 48))):
        ax.annotate("", v, (0, 0), arrowprops=dict(arrowstyle="->", lw=2, color="k"))
        ax.text(*dxy, lab, ha="center", va="center", fontsize=14)
    ax.plot(0, 0, "ko", ms=4)
    ax.text(5, -14, "origin of A", fontsize=11)
    ax.set_aspect("equal")
    ax.set_xlim(-305, 85)
    ax.set_ylim(-48, 305)
    ax.set_xlabel(r"$x_A$ (mm)")
    ax.set_ylabel(r"$y_A$ (mm)")
    ax.set_title("(a) Hand frame A seen from the robot side. Grey: base plate; dashed box: robot mount;\n"
                 "blue: gantry frames and fingers; orange: adaptor; dashed circle: 63 mm flange face", fontsize=13)

    # (b) the adaptor in the pocket
    ax = axZ
    plot(ax, geo["pocket"], color=COL["mount"], lw=1.2)
    plot(ax, geo["adaptor"], color=COL["adaptor"], lw=1.8)
    plot(ax, circle(bc, 25.0, 28), color="k", lw=0.8, ls="-.")
    plot(ax, circle(bc, 31.5, 28), color=COL["flange"], lw=1.0, ls="--")
    for c, a in zip(m6, m6_ang):
        plot(ax, circle(c, 3.2, 28), color="k", lw=1.4)
        plot(ax, circle(c, 5.5, 6), color=COL["adaptor"], lw=0.9, ls="--")
        ax.plot([bc[0], c[0]], [bc[1], c[1]], color="k", lw=0.6)
        t = math.radians(a)
        lab = f"M6 at {a:.0f}°" if a > -45 else f"M6 at {a % 360:.0f}°"
        ax.text(bc[0] + 40 * math.cos(t), bc[1] + 40 * math.sin(t), lab, ha="center", va="center", fontsize=12,
                bbox=dict(fc="white", ec="none", alpha=0.9, pad=1))
    spots = {(-116.5, 100.0): (-152, 76), (-105.5, 100.0): (-70, 76), (-133.0, 117.0): (-176, 136),
             (-89.0, 117.0): (-46, 136)}
    for h in plate:
        a, r = np.array(h["adaptor_hole_A_mm"]), np.array(h["robot_mount_hole_A_mm"])
        plot(ax, circle(a, 2.2, 28), color=COL["plate"], lw=1.6)
        plot(ax, circle(r, 4.25, 4), color=COL["rmhole"], lw=0.9, ls=":")
        ax.plot(*r, "x", color=COL["rmhole"], ms=8, mew=1.6)
        key = min(spots, key=lambda s: np.hypot(s[0] - a[0], s[1] - a[1]))
        ax.annotate(f"({a[0]:.1f}, {a[1]:.1f})\noffset {h['residual_mm']:.0e} mm", a, spots[key], fontsize=11,
                    color=COL["plate"], ha="center", va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.7, color=COL["plate"]))
    ax.plot(*bc, "+", color="k", ms=14, mew=2)
    handles = [plt.Line2D([], [], color=COL["adaptor"], lw=1.8, label="adaptor, 60.4 mm, turned 24° in A"),
               plt.Line2D([], [], color=COL["mount"], lw=1.2, label="robot mount's pocket, 61.0 mm"),
               plt.Line2D([], [], color="k", lw=1.4, label="6.4 mm M6 clearance holes, 50 mm circle"),
               plt.Line2D([], [], color=COL["adaptor"], lw=0.9, ls="--", label="11 mm counterbores, hand side"),
               plt.Line2D([], [], color=COL["plate"], lw=1.6, label="adaptor's 4.4 mm plate-side holes"),
               plt.Line2D([], [], color=COL["rmhole"], marker="x", lw=0.9, ls=":",
                          label="robot mount's 4.4 mm holes and 8.5 mm counterbores"),
               plt.Line2D([], [], color=COL["flange"], lw=1.0, ls="--", label="UR flange face, 63 mm"),
               plt.Line2D([], [], color="k", marker="+", ms=12, mew=2, lw=0,
                          label=f"flange axis at ({bc[0]:.1f}, {bc[1]:.1f}) in A")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=11, framealpha=0.95)
    ax.set_aspect("equal")
    ax.set_xlim(bc[0] - 80, bc[0] + 80)
    ax.set_ylim(bc[1] - 50, bc[1] + 46)
    ax.set_xlabel(r"$x_A$ (mm)")
    ax.set_ylabel(r"$y_A$ (mm)")
    ax.set_title("(b) Adaptor in the robot mount's pocket, seen from the robot side. The plate-side holes\n"
                 "lie on the robot mount's trapezoid; the M6 holes on a 50 mm circle about the flange axis", fontsize=13)

    # (c-f) the four clockings in E, seen from the tool side (z_E toward the viewer)
    for k, ax in enumerate(axK):
        ck = F["clockings"][k]
        T = ck["T_E_from_A"]
        fill(ax, geo["base"], T, color="#eeeeee", lw=0)
        plot(ax, geo["base"], T, color=COL["base"], lw=1.0)
        for P in geo["fingers"]:
            fill(ax, P, T, color=COL["finger"], alpha=0.45, lw=0)
        for g in ("thumb", "index", "middle"):
            plot(ax, geo["g_" + g], T, color=COL["gantry"], lw=1.3)
            c = to_E(T, np.array([[*gc[g], -11.0]]))[0][:2] * 1.75
            ax.text(c[0], c[1], g, ha="center", va="center", color=COL["gantry"], fontsize=13, fontweight="bold",
                    bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
        for P in geo["cam"]:
            plot(ax, P, T, color=COL["cam"], lw=1.3)
        plot(ax, geo["adaptor"], T, color=COL["adaptor"], lw=1.3)
        plot(ax, circle((0, 0), 31.5, 0), color=COL["flange"], lw=1.5)
        plot(ax, circle((0, 0), 37.5, 0), color=COL["flange"], lw=0.8, ls=":")
        for a in (45, 135, 225, 315):
            ax.add_patch(plt.Circle((25 * math.cos(math.radians(a)), 25 * math.sin(math.radians(a))), 3.2,
                                    color="k", lw=0))
        ax.add_patch(plt.Circle((0, -25), 3.0, fill=False, color=COL["dowel"], lw=1.5))
        ax.add_patch(Rectangle((-3.5, -46.6), 7.0, 9.1, color=COL["conn"], lw=0))
        ax.annotate("tool connector and\ndowel hole, on −y_E", (0, -47), (0, -178), ha="center", va="center",
                    fontsize=11.5, color=COL["conn"], arrowprops=dict(arrowstyle="-", color=COL["conn"], lw=1.0),
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
        for v, lab in (((1, 0), r"$x_E$"), ((0, 1), r"$y_E$")):
            ax.annotate("", (70 * v[0], 70 * v[1]), (0, 0), arrowprops=dict(arrowstyle="->", lw=2, color="k"))
            ax.text(82 * v[0], 82 * v[1], lab, ha="center", va="center", fontsize=14)
        for v, lab in ((ck["x_A_in_E"], r"$x_A$"), (ck["y_A_in_E"], r"$y_A$")):
            ax.annotate("", (48 * v[0], 48 * v[1]), (0, 0),
                        arrowprops=dict(arrowstyle="->", lw=1.6, color=COL["adaptor"]))
            ax.text(57 * v[0], 57 * v[1], lab, color=COL["adaptor"], ha="center", va="center", fontsize=13)
        d = ck["gantry_direction_from_flange_axis_E_deg"]
        rel = ck["connector_minus_gantry_direction_deg"]
        near = min(rel, key=lambda g: abs(rel[g]))
        ax.set_title(f"({'cdef'[k]}) k = {k}: x_A at {ck['theta_deg']:.0f}° from x_E\n"
                     f"thumb at {d['thumb']:.0f}°; connector {abs(rel[near]):.0f}° from {near}", fontsize=12.5)
        ax.set_aspect("equal")
        ax.set_xlim(-215, 215)
        ax.set_ylim(-215, 215)
        ax.set_xlabel(r"$x_E$ (mm)")
        if k == 0:
            ax.set_ylabel(r"$y_E$ (mm)")
        ax.grid(True, lw=0.3, alpha=0.5)
    fig.text(0.5, 0.415, "(c–f) The hand on the flange for the four clockings k in tool0 (E), seen from the tool side "
             "(from below the fingers, looking at the flange face).\nBlack dots: the CB3's M6 holes at 45°, 135°, 225° "
             "and 315° from x_E; purple circle: its dowel hole; red: the tool connector.", ha="center", va="center",
             fontsize=13)
    fig.savefig(OUT / "flange.png", bbox_inches="tight")
    print("wrote", OUT / "flange.png")


if __name__ == "__main__":
    main()
