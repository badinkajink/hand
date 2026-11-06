#!/usr/bin/env python3
"""Explanatory figures of Eq. (1), the relative contact twist, and of the contact-point filter, Eqs. (4) to (6), of the
HOM paper (S. Wang, J. Oh and N. S. Pollard, arXiv 2609.25619v2) for the HOM review page
(scripts/hom/hom_review_page.py), 2026-11-06.

Owner, 2026-11-06: "a really high quality visual, diagrammed explanation of the physical phenomena"; "a high quality,
very clear visual and explainer of eqn (1) ... explicitly understand each transformation (and its correspondence to the
notation), what the goddamn adjoint actually does (as a transformation of the 6x6 twist from one frame to another)".

  frames     the frames and the poses of Eq. (1) on a planar SR2 finger with the v4b tip beside a cylinder
  adjoint    what Ad_T does to a twist: the same rigid motion measured at another point, in other axes
  terms      Eq. (1) term by term: the finger's and the object's velocity at {L1}, and their difference
  geodesic   one step of the geodesic tracer and of the filter on the v4b pad's section

Every planar figure has all joint axes pointing out of the page. Each frame's y axis therefore points at the reader, its
z axis is its x axis turned 90 deg clockwise (z = x cross y), and a positive rotation about y turns counter-clockwise on
the page. These are the conventions of src/sr2hand/control/hom.py: x1 is the finger's contact normal toward the object,
y1 the joint axis laid into the tangent plane, and a relative twist is (spin, roll y, roll z, sep, slide y, slide z).

    .venv/bin/python scripts/hom/hom_eq1_figures.py [frames adjoint terms geodesic]
Writes docs/experiments/20261029-hom_control_review/media/20261106-eq1_*.png and 20261106-geodesic_filter.png; prints
the numbers that the page quotes and writes them to rows/20261106-eq1_figures.json.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path

D = ROOT / "docs/experiments/20261029-hom_control_review"
MEDIA, ROWS = D / "media", D / "rows"
NUMBERS = ROWS / "20261106-eq1_figures.json"

# the SR2 finger with the v4b tip (scripts/hom/hand_object_scale_kin.py, SCALE_HAND=cad_v2), mm
L_PROX, L_DIST, RHO = 30.80, 36.36, 10.55        # MCP axis to PIP axis, PIP axis to the pad's centre, apex radius
R_OBJ = 12.5                                      # the 25 mm cylinder of the pin-push
TH = np.radians([20.0, 40.0])                     # MCP and PIP angles from straight down, counter-clockwise
GAP = 14.0                                        # mm between the pad and the cylinder
PSI_OBJ = math.radians(28.0)                      # the object frame's turn on the page
# the motion of the worked example: joint rates (deg/s) and the object's body twist (deg/s; mm/s in its own axes)
U_DEG = (-20.0, 70.0)
W_OBJ_DEG, V_OBJ_BODY = -40.0, (-14.2, -4.6)

C_FIN, C_FIN_L = "#00838f", "#d4ecee"            # the finger: teal
C_OBJ, C_OBJ_L = "#d9730d", "#fbe6cf"            # the object: orange
C_LEV = "#7b3fa0"                                 # the lever-arm term w x r: purple
C_REL = "#111111"                                 # the relative twist: black
C_POSE = "#7a7a7a"                                # poses T_XY: grey
AX_X, AX_Y, AX_Z = "#d62728", "#2ca02c", "#1f63b4"


def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 12.5,
                         "mathtext.fontset": "stix", "axes.titlesize": 13, "axes.labelsize": 12.5,
                         "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 12})
    return plt


def save(fig, name):
    MEDIA.mkdir(parents=True, exist_ok=True)
    fig.savefig(MEDIA / name, dpi=200, facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("wrote", (MEDIA / name).relative_to(ROOT), flush=True)


# ------------------------------------------------------------------------------------------------ planar kinematics
def rot(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def cw(v):
    """v turned 90 deg clockwise: z = x cross y for a frame whose y axis points out of the page."""
    return np.array([v[1], -v[0]])


def wx(w, r):
    """w cross r for an angular velocity w (rad/s) about the axis out of the page: r turned 90 deg counter-clockwise."""
    return w * np.array([-r[1], r[0]])


def finger(th=TH):
    """Joint positions on the page (mm): the MCP at the origin of the palm frame {P}, the PIP (origin of the distal link's
    frame {B1}) and the pad's centre; the links hang down at zero angle and turn counter-clockwise."""
    pip = L_PROX * np.array([math.sin(th[0]), -math.cos(th[0])])
    pad = pip + L_DIST * np.array([math.sin(th[0] + th[1]), -math.cos(th[0] + th[1])])
    return np.zeros(2), pip, pad


def scene(th=TH, gap=GAP):
    mcp, pip, pad = finger(th)
    al = math.radians(-20.0)
    C = pad + (RHO + R_OBJ + gap) * np.array([math.cos(al), math.sin(al)])
    x1 = (C - pad) / np.linalg.norm(C - pad)
    p1 = pad + RHO * x1                           # {L1}: the pad's witness point
    p0 = C - R_OBJ * x1                           # {L0}: the cylinder's witness point
    return dict(mcp=mcp, pip=pip, pad=pad, C=C, x1=x1, z1=cw(x1), p1=p1, p0=p0, th=np.asarray(th, float))


def twists(S, u_deg=U_DEG, w_obj_deg=W_OBJ_DEG, v_obj_body=V_OBJ_BODY):
    """The planar twists of the worked example. A planar twist is (w, v): w about the axis out of the page (rad/s), v on
    the page (mm/s). Returned in page axes; the relative twist also in {L1}'s axes (roll y, sep x, slide z)."""
    u = np.radians(u_deg)
    w1 = float(u.sum())                            # the distal link turns with both joints
    v_b1 = wx(u[0], S["pip"] - S["mcp"])           # its origin, the PIP, moves with the MCP joint only
    v1 = v_b1 + wx(w1, S["p1"] - S["pip"])         # the finger's material point at p1
    w0 = math.radians(w_obj_deg)
    x_b0 = rot(PSI_OBJ) @ np.array([1.0, 0.0])
    v_b0 = v_obj_body[0] * x_b0 + v_obj_body[1] * cw(x_b0)   # the object's origin, from its (x, z) axes to the page
    v0 = v_b0 + wx(w0, S["p1"] - S["C"])           # the object's motion extended to p1
    w_rel, v_rel = w1 - w0, v1 - v0
    rel = dict(roll_y=w_rel, sep=float(v_rel @ S["x1"]), slide_z=float(v_rel @ S["z1"]))
    return dict(u=u, w1=w1, v_b1=v_b1, v1=v1, w0=w0, v_b0=v_b0, v0=v0, w_rel=w_rel, v_rel=v_rel, rel=rel,
                lever1=wx(w1, S["p1"] - S["pip"]), lever0=wx(w0, S["p1"] - S["C"]))


# ------------------------------------------------------------------------------------------------ drawing helpers
def arrow(ax, p, q, color, lw=2.2, ls="-", ms=16, z=5, style="-|>", shrink=0.0):
    from matplotlib.patches import FancyArrowPatch
    a = FancyArrowPatch(tuple(p), tuple(q), arrowstyle=style, mutation_scale=ms, color=color, lw=lw, ls=ls, zorder=z,
                        shrinkA=shrink, shrinkB=shrink, joinstyle="miter", capstyle="butt")
    ax.add_patch(a)
    return a


def curved(ax, c, r, a0, a1, color, lw=2.0, ms=14, z=6):
    """A counter-clockwise (a1 > a0) or clockwise arc arrow of radius r about c, angles in deg."""
    from matplotlib.patches import FancyArrowPatch
    t = np.radians(np.linspace(a0, a1, 40))
    pts = c + r * np.c_[np.cos(t), np.sin(t)]
    ax.plot(pts[:-3, 0], pts[:-3, 1], color=color, lw=lw, zorder=z, solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch(tuple(pts[-4]), tuple(pts[-1]), arrowstyle="-|>", mutation_scale=ms, color=color,
                                 lw=lw, zorder=z, shrinkA=0, shrinkB=0))


def frame(ax, o, x, name, L=9.0, lw=2.0, label_at=None, fs=14, color="k", ms=13, yaxis=True):
    """A planar frame: x and z = x turned clockwise as arrows, y out of the page as a dot in a circle."""
    from matplotlib.patches import Circle
    x = np.asarray(x, float) / np.linalg.norm(x)
    arrow(ax, o, o + L * x, AX_X, lw=lw, ms=ms, z=7)
    arrow(ax, o, o + L * cw(x), AX_Z, lw=lw, ms=ms, z=7)
    if yaxis:
        ax.add_patch(Circle(tuple(o), 1.25, fc="white", ec=AX_Y, lw=1.6, zorder=8))
        ax.add_patch(Circle(tuple(o), 0.42, fc=AX_Y, ec=AX_Y, lw=0, zorder=9))
    if name:
        lp = o + (label_at if label_at is not None else np.array([-2.0, 2.5]))
        ax.text(*lp, name, fontsize=fs, color=color, ha="center", va="center", zorder=10,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))


def axis_letters(ax, o, x, L=9.0, fs=11.5, sub=""):
    x = np.asarray(x, float) / np.linalg.norm(x)
    for v, nm, c in ((x, "x", AX_X), (cw(x), "z", AX_Z)):
        q = o + (L + 2.4) * v
        ax.text(*q, f"${nm}_{{{sub}}}$" if sub else f"${nm}$", fontsize=fs, color=c, ha="center", va="center", zorder=10,
                bbox=dict(boxstyle="round,pad=0.05", fc="white", ec="none", alpha=0.85), clip_on=True)


def draw_hand(ax, S, fin_alpha=1.0, obj_alpha=1.0, finger_fc="#e9e9e9", pad_fc=C_FIN_L, obj_fc=C_OBJ_L):
    from matplotlib.patches import Circle, Polygon, Rectangle
    mcp, pip, pad = S["mcp"], S["pip"], S["pad"]
    alpha, obj = fin_alpha, obj_alpha > 0
    ax.add_patch(Rectangle((-16, 0), 32, 7, fc="#cfcfcf", ec="#555555", lw=1.2, zorder=1, alpha=alpha))
    ax.text(-15, 8.5, "palm", fontsize=12, color="#555555", va="bottom", clip_on=True, alpha=alpha)
    for a, b, w in ((mcp, pip, 4.2), (pip, pad, 4.2)):
        d = (b - a) / np.linalg.norm(b - a)
        n = np.array([-d[1], d[0]])
        poly = [a + w * n, b + w * n, b - w * n, a - w * n]
        ax.add_patch(Polygon(poly, closed=True, fc=finger_fc, ec="#555555", lw=1.2, zorder=2, alpha=alpha))
    ax.add_patch(Circle(tuple(pad), RHO, fc=pad_fc, ec=C_FIN, lw=1.6, zorder=3, alpha=alpha))
    for j in (mcp, pip):
        ax.add_patch(Circle(tuple(j), 2.4, fc="white", ec="#555555", lw=1.3, zorder=4, alpha=alpha))
    if obj:
        ax.add_patch(Circle(tuple(S["C"]), R_OBJ, fc=obj_fc, ec=C_OBJ, lw=1.6, zorder=3, alpha=obj_alpha))


def label(ax, p, s, color="k", fs=13, ha="center", va="center", z=11, box=True):
    kw = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9) if box else None
    ax.text(*p, s, fontsize=fs, color=color, ha=ha, va=va, zorder=z, bbox=kw, clip_on=True)


# ------------------------------------------------------------------------------------------------ Eq. (1)
LIM = ((-25.0, 93.0), (-80.0, 12.5))             # mm: the window of panel (a), the whole finger
LIMZ = ((-2.0, 92.0), (-82.0, -8.3))              # mm: the window of panels (b) to (d), PIP to cylinder
SC = 1.0                                          # mm of arrow per mm/s


def curved_pose(ax, p, q, color, rad, lw=1.6, ls=(0, (4, 2.5)), ms=13, shrink=3.0, z=3):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(tuple(p), tuple(q), arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=shrink, shrinkB=shrink, zorder=z))


def panel(ax, S, title, color="k", fin=1.0, obj=1.0, lim=LIM):
    draw_hand(ax, S, fin_alpha=fin, obj_alpha=obj)
    ax.set_aspect("equal")
    ax.set_xlim(*lim[0])
    ax.set_ylim(*lim[1])
    ax.axis("off")
    ax.set_title(title, color=color, fontsize=13.5, pad=3)


def fig_eq1():
    """Eq. (1) on the planar finger in four panels: the frames and poses; the finger's twist moved to {L1}; the object's
    twist moved to {L1}; their difference in {L1}'s axes."""
    plt = plt_setup()
    S = scene()
    T = twists(S)
    p1, pip, C = S["p1"], S["pip"], S["C"]
    fig, axs = plt.subplots(2, 2, figsize=(8.28, 7.0))
    a, b, c, d = axs.flat
    # (a) frames and poses, the whole finger
    panel(a, S, "(a) Frames and the poses of Eq. (1)", lim=LIM)
    x_b1 = (S["pad"] - pip) / np.linalg.norm(S["pad"] - pip)
    curved_pose(a, S["mcp"], pip, C_POSE, rad=0.4)
    curved_pose(a, S["mcp"], C, C_POSE, rad=-0.3)
    curved_pose(a, p1, pip, C_FIN, rad=-0.35, lw=2.0, ls="-", ms=15)
    curved_pose(a, p1, C, C_OBJ, rad=0.75, lw=2.0, ls="-", ms=15)
    label(a, (-15.5, -18.0), r"$T_{PB_1}$", C_POSE, fs=14)
    label(a, (52.0, -12.0), r"$T_{PB_0}$", C_POSE, fs=14)
    label(a, (25.5, -57.5), r"$T_{L_1B_1}$", C_FIN, fs=14)
    label(a, (64.0, -77.0), r"$T_{L_1B_0}$", C_OBJ, fs=14, va="bottom")
    frame(a, S["mcp"], (1.0, 0.0), r"$\{P\}$", L=8, label_at=np.array([10.5, 6.0]), ms=11, lw=1.7)
    frame(a, pip, x_b1, r"$\{B_1\}$", L=8, label_at=np.array([-10.0, -1.0]), ms=11, lw=1.7)
    frame(a, C, rot(PSI_OBJ) @ np.array([1.0, 0.0]), r"$\{B_0\}$", L=7, label_at=np.array([3.5, -9.0]), ms=11,
          lw=1.7)
    frame(a, p1, S["x1"], r"$\{L_1\}$", L=5.5, label_at=np.array([-8.5, 5.5]), ms=10, lw=1.6)
    frame(a, S["p0"], -S["x1"], r"$\{L_0\}$", L=5.0, label_at=np.array([2.5, 8.0]), ms=9, lw=1.4)
    axis_letters(a, p1, S["x1"], L=5.5, sub="1", fs=12.5)
    a.plot(*np.c_[p1, S["p0"]], color="k", lw=1.0, ls=":", zorder=5)
    # (b) the finger's twist at {L1}
    panel(b, S, r"(b) Finger: $\mathrm{Ad}_{T_{L_1B_1}}\,\mathcal{V}^{b}_{PB_1}$", C_FIN, obj=0.35, lim=LIMZ)
    frame(b, pip, x_b1, None, L=6, ms=10, lw=1.4)
    b.plot(*np.c_[pip, p1], color=C_FIN, lw=1.2, ls=(0, (3, 2)), zorder=6)
    label(b, 0.5 * (pip + p1) + np.array([-1.5, 4.5]), r"$r_1$", C_FIN, fs=14)
    curved(b, pip, 4.6, 100, 330, C_FIN, lw=1.8, ms=12)
    label(b, pip + np.array([1.0, 8.5]), r"$\omega_1$", C_FIN, fs=14.5)
    arrow(b, pip, pip + SC * T["v_b1"], C_FIN, lw=2.6, ms=16, z=8)
    label(b, pip + SC * T["v_b1"] + np.array([3.5, -4.5]), r"$v_{B_1}$", C_FIN, fs=14)
    arrow(b, p1, p1 + SC * T["v_b1"], C_FIN, lw=1.8, ms=13, ls=(0, (3, 2)), z=8)
    arrow(b, p1 + SC * T["v_b1"], p1 + SC * T["v1"], C_LEV, lw=2.4, ms=15, z=8)
    arrow(b, p1, p1 + SC * T["v1"], C_FIN, lw=3.0, ms=18, z=9)
    label(b, p1 + SC * (T["v_b1"] + 0.45 * T["lever1"]) + np.array([-2.5, 1.5]), r"$\omega_1\times r_1$", C_LEV,
          fs=14, ha="right")
    label(b, p1 + SC * T["v1"] + np.array([3.0, 1.0]), r"$v_1(p_1)$", C_FIN, fs=14.5, ha="left")
    # (c) the object's twist at {L1}
    panel(c, S, r"(c) Object: $\mathrm{Ad}_{T_{L_1B_0}}\,\mathcal{V}^{b}_{PB_0}$", C_OBJ, fin=0.35, lim=LIMZ)
    frame(c, C, rot(PSI_OBJ) @ np.array([1.0, 0.0]), None, L=6, ms=10, lw=1.4)
    c.plot(*np.c_[C, p1], color=C_OBJ, lw=1.2, ls=(0, (3, 2)), zorder=6)
    label(c, 0.5 * (C + p1) + np.array([-1.0, 4.5]), r"$r_0$", C_OBJ, fs=14)
    curved(c, C, 6.0, 330, 80, C_OBJ, lw=1.8, ms=12)
    label(c, C + np.array([9.5, 2.0]), r"$\omega_0$", C_OBJ, fs=14.5)
    arrow(c, C, C + SC * T["v_b0"], C_OBJ, lw=2.4, ms=15, z=8)
    label(c, C + SC * T["v_b0"] + np.array([1.0, -5.0]), r"$v_{B_0}$", C_OBJ, fs=14)
    arrow(c, p1, p1 + SC * T["v_b0"], C_OBJ, lw=1.8, ms=13, ls=(0, (3, 2)), z=8)
    arrow(c, p1 + SC * T["v_b0"], p1 + SC * T["v0"], C_LEV, lw=2.4, ms=15, z=8)
    arrow(c, p1, p1 + SC * T["v0"], C_OBJ, lw=3.0, ms=18, z=9)
    label(c, p1 + SC * (T["v_b0"] + 0.5 * T["lever0"]) + np.array([-2.0, 1.0]), r"$\omega_0\times r_0$", C_LEV,
          fs=14, ha="right")
    label(c, p1 + SC * T["v0"] + np.array([2.5, 2.5]), r"$v_0(p_1)$", C_OBJ, fs=14.5, ha="left")
    # (d) the difference, in {L1}'s axes
    rel = T["rel"]
    panel(d, S, r"(d) Difference: $\mathcal{V}^{b}_{L_0L_1}$ in $\{L_1\}$'s axes", fin=0.35, obj=0.35, lim=LIMZ)
    arrow(d, p1, p1 + SC * T["v1"], C_FIN, lw=1.8, ms=13, z=8)
    arrow(d, p1, p1 + SC * T["v0"], C_OBJ, lw=1.8, ms=13, z=8)
    arrow(d, p1 + SC * T["v0"], p1 + SC * T["v1"], C_REL, lw=2.8, ms=16, z=9)
    label(d, p1 + SC * T["v1"] + np.array([2.0, -1.0]), r"$v_1$", C_FIN, fs=14, ha="left")
    label(d, p1 + SC * T["v0"] + np.array([-2.0, 1.5]), r"$v_0$", C_OBJ, fs=14, ha="right")
    label(d, p1 + SC * (T["v0"] + 0.5 * T["v_rel"]) + np.array([-2.0, 2.5]), r"$v_1 - v_0$", C_REL, fs=14,
          ha="right")
    q = np.array([8.0, -74.0])                    # v_1 - v_0 again, split along x1 (sep) and z1 (slide)
    frame(d, q, S["x1"], None, L=5.0, ms=10, lw=1.2)
    for v, nm, col, off in ((S["x1"], "x", AX_X, (0.5, 3.0)), (S["z1"], "z", AX_Z, (-3.0, 0.0))):
        label(d, q + 6.2 * v + np.array(off), f"${nm}_1$", col, fs=12.5)
    end_sep = q + SC * rel["sep"] * S["x1"]
    arrow(d, q, end_sep, AX_X, lw=2.6, ms=14, z=9)
    arrow(d, end_sep, q + SC * T["v_rel"], AX_Z, lw=2.6, ms=14, z=9)
    arrow(d, q, q + SC * T["v_rel"], C_REL, lw=2.6, ms=15, z=9)
    label(d, end_sep + np.array([2.5, -2.5]), "sep", AX_X, fs=13, ha="left")
    label(d, 0.5 * (end_sep + q + SC * T["v_rel"]) + np.array([2.0, -1.0]), "slide", AX_Z, fs=13, ha="left")
    curved(d, p1, 5.0, 150, 420, C_REL, lw=1.7, ms=11)
    label(d, p1 + np.array([-9.0, -6.0]), "roll", C_REL, fs=13)
    d.text(LIMZ[0][0] + 1.5, LIMZ[1][1] - 2.5,
           f"roll $y$   {math.degrees(rel['roll_y']):+.0f} deg/s\n"
           f"sep       {rel['sep']:+.1f} mm/s\nslide $z$  {rel['slide_z']:+.1f} mm/s",
           fontsize=13, va="top", zorder=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#aaaaaa", lw=0.8))
    fig.subplots_adjust(0.005, 0.005, 0.995, 0.955, wspace=0.04, hspace=0.11)
    save(fig, "20261106-eq1_terms.png")
    return dict(u_deg=list(U_DEG), w_obj_deg=W_OBJ_DEG, v_obj_body=list(V_OBJ_BODY), w1_deg=math.degrees(T["w1"]),
                v_b1=T["v_b1"].round(3).tolist(), v1=T["v1"].round(3).tolist(), v_b0=T["v_b0"].round(3).tolist(),
                v0=T["v0"].round(3).tolist(), lever1=T["lever1"].round(3).tolist(), lever0=T["lever0"].round(3).tolist(),
                rel={k: (math.degrees(v) if k == "roll_y" else v) for k, v in rel.items()},
                r1=float(np.linalg.norm(pip - p1)), r0=float(np.linalg.norm(C - p1)),
                gap=float(np.linalg.norm(S["p0"] - p1)), sc_mm_per_mm_s=SC, table=twist_table(S, T))


def twist_table(S, T):
    """The worked example's planar twists, each in the axes of the frame it is written in: (w_y deg/s, v_x, v_z mm/s)."""
    def comp(v, x):
        x = np.asarray(x, float) / np.linalg.norm(x)
        return [float(v @ x), float(v @ cw(x))]
    x_b1 = (S["pad"] - S["pip"]) / np.linalg.norm(S["pad"] - S["pip"])
    x_b0 = rot(PSI_OBJ) @ np.array([1.0, 0.0])
    return {"V_PB1": [math.degrees(T["w1"])] + comp(T["v_b1"], x_b1),
            "Ad_L1B1_V_PB1": [math.degrees(T["w1"])] + comp(T["v1"], S["x1"]),
            "V_PB0": [math.degrees(T["w0"])] + comp(T["v_b0"], x_b0),
            "Ad_L1B0_V_PB0": [math.degrees(T["w0"])] + comp(T["v0"], S["x1"]),
            "V_L0L1": [math.degrees(T["w_rel"])] + comp(T["v_rel"], S["x1"])}


# ------------------------------------------------------------------------------------------------ adjoint
def fig_adjoint():
    """What Ad_{T_AB} does: the twist (w, v_B) of a rigid body, measured at {B}, re-measured at {A}."""
    plt = plt_setup()
    from matplotlib.patches import FancyBboxPatch, Rectangle
    from matplotlib.transforms import Affine2D
    fig = plt.figure(figsize=(8.28, 5.6))
    ax = fig.add_axes((0.0, 0.37, 1.0, 0.63))
    bx = fig.add_axes((0.0, 0.0, 1.0, 0.37))
    psi = math.radians(15.0)
    Rb = rot(psi)
    oB = np.zeros(2)
    w = math.radians(-25.0)                       # rad/s: clockwise on the page
    vB = np.array([4.0, 13.0])                    # mm/s, page axes
    body = FancyBboxPatch((-12, -13), 70, 26, boxstyle="round,pad=0,rounding_size=6", fc="#efefef", ec="#555555",
                          lw=1.4, zorder=1)
    body.set_transform(Affine2D().rotate(psi) + ax.transData)
    ax.add_patch(body)
    sc = 0.42
    for gx in np.arange(-4, 56, 10.0):
        for gz in (-7.0, 7.0):
            p = Rb @ np.array([gx, gz])
            arrow(ax, p, p + sc * (vB + wx(w, p - oB)), "#a9a9a9", lw=1.1, ms=9, z=2)
    icr = oB + np.array([-vB[1], vB[0]]) / w      # the point that does not move
    ax.plot(*icr, marker="x", ms=8, mew=2, color="#7a7a7a", zorder=3)
    ax.text(icr[0] + 2.2, icr[1] - 1.5, "a point at rest:\nthe body turns about it", fontsize=12, color="#666666",
            va="top")
    oA = Rb @ np.array([47.0, 6.0])
    psiA = psi + math.radians(235.0)
    xA, xB = rot(psiA) @ np.array([1.0, 0.0]), Rb @ np.array([1.0, 0.0])
    r = oA - oB
    lev = wx(w, r)
    vA = vB + lev
    frame(ax, oB, xB, r"$\{B\}$", L=7, label_at=np.array([-6.0, -6.5]), lw=1.6, ms=11)
    frame(ax, oA, xA, r"$\{A\}$", L=7, label_at=np.array([-9.0, 3.0]), lw=1.6, ms=11)
    curved(ax, oB, 4.6, 160, -110, C_REL, lw=1.8, ms=12)
    label(ax, oB + np.array([-8.0, 3.5]), r"$\omega$", C_REL, fs=15)
    arrow(ax, oB, oB + sc * vB, C_FIN, lw=2.8, ms=17, z=8)
    label(ax, oB + sc * vB + np.array([-3.5, 1.5]), r"$v_B$", C_FIN, fs=15)
    ax.plot(*np.c_[oB, oA], color="#555555", lw=1.2, ls=(0, (3, 2)), zorder=3)
    label(ax, 0.5 * (oB + oA) + np.array([-1.0, 3.6]), r"$r$", "#444444", fs=14)
    arrow(ax, oA, oA + sc * vB, C_FIN, lw=2.0, ms=14, ls=(0, (4, 2)), z=8)
    arrow(ax, oA + sc * vB, oA + sc * vA, C_LEV, lw=2.6, ms=16, z=8)
    arrow(ax, oA, oA + sc * vA, C_REL, lw=3.0, ms=18, z=9)
    label(ax, oA + sc * (vB + 0.5 * lev) + np.array([2.5, 1.5]), r"$\omega\times r$", C_LEV, fs=14.5, ha="left")
    label(ax, oA + sc * vA + np.array([3.5, -1.5]), r"$v_A$", C_REL, fs=15, ha="left")
    ax.set_aspect("equal")
    ax.set_xlim(-18, 80)
    ax.set_ylim(-17.5, 29.0)
    ax.axis("off")
    # the block matrix
    bx.set_xlim(0, 100)
    bx.set_ylim(0, 36)
    bx.axis("off")

    def bracket(x0, x1, y0, y1, lw=1.6):
        for x, s in ((x0, 1), (x1, -1)):
            bx.plot([x + s * 1.1, x, x, x + s * 1.1], [y1, y1, y0, y0], color="k", lw=lw)

    yc, h = 20.0, 6.6
    bracket(1.5, 11.5, yc - h - 0.8, yc + h + 0.8)
    bx.text(6.5, yc + h / 2, r"$\omega_A$", ha="center", va="center", fontsize=16)
    bx.text(6.5, yc - h / 2, r"$v_A$", ha="center", va="center", fontsize=16, color=C_REL)
    bx.text(14.0, yc, "=", ha="center", va="center", fontsize=17)
    cells = [((17.5, yc), r"$R_{AB}$", "#ececec", "k"), ((35.5, yc), r"$0$", "white", "k"),
             ((17.5, yc - h - 0.5), r"$[p_{AB}]\,R_{AB}$", "#eadcf2", C_LEV),
             ((35.5, yc - h - 0.5), r"$R_{AB}$", "#ececec", "k")]
    for (x, y), s, fc, col in cells:
        bx.add_patch(Rectangle((x, y + 0.3), 17, h - 0.6, fc=fc, ec="none"))
        bx.text(x + 8.5, y + h / 2, s, ha="center", va="center", fontsize=15.5, color=col)
    bracket(16.5, 53.5, yc - h - 0.8, yc + h + 0.8)
    bracket(56.0, 65.0, yc - h - 0.8, yc + h + 0.8)
    bx.text(60.5, yc + h / 2, r"$\omega$", ha="center", va="center", fontsize=16)
    bx.text(60.5, yc - h / 2, r"$v_B$", ha="center", va="center", fontsize=16, color=C_FIN)
    RA = rot(psiA)
    nB = np.array([vB @ xB, vB @ cw(xB)])
    nA = np.array([vA @ xA, vA @ cw(xA)])
    bx.text(67.5, yc + h / 2, r"$\mathcal{V}_B$: " + f"({math.degrees(w):.0f} deg/s;  {nB[0]:.1f}, {nB[1]:.1f} mm/s)"
            + "\nin $\\{B\\}$'s axes", fontsize=12.5, va="center")
    bx.text(67.5, yc - h / 2 - 1.0, r"$\mathcal{V}_A$: " + f"({math.degrees(w):.0f} deg/s;  {nA[0]:.1f}, {nA[1]:.1f} mm/s)"
            + "\nin $\\{A\\}$'s axes", fontsize=12.5, va="center")
    kw = dict(fontsize=12.5, va="center", ha="center")
    bx.text(26.0, yc + h + 3.4, "$\\omega$ in $\\{A\\}$'s axes", color="#444444", **kw)
    bx.text(26.0, yc - h - 4.2, "$p_{AB}\\times\\omega = \\omega\\times r$", color=C_LEV, **kw)
    bx.text(44.0, yc - h - 4.2, "$v_B$ in $\\{A\\}$'s axes", color=C_FIN, **kw)
    save(fig, "20261106-eq1_adjoint.png")
    return dict(w_deg=math.degrees(w), vB_page=vB.tolist(), r=r.round(3).tolist(), lever=lev.round(3).tolist(),
                vA_page=vA.round(3).tolist(), V_B_in_B=[math.degrees(w)] + nB.round(3).tolist(),
                V_A_in_A=[math.degrees(w)] + nA.round(3).tolist(), icr=icr.round(2).tolist(),
                psi_B_deg=15.0, psi_A_deg=55.0)


# ------------------------------------------------------------------------------------------------ geodesic and filter
PAD_SCENE = ROOT / "logs/20261030-hand_object_scale_cad_v2/scenes/sph040.0/x092.5_y102.5_pads_bv2.xml"
ELL = (8.5, 7.4, 11.0, 22.0)                      # mm: the v4b pad's semi-axes, depth, across, distal, proximal
PAD_Z = (-10.55, 10.86)                           # mm: the printed pad's extent along the finger
C_PAD = RHO - ELL[0]                              # mm: the ellipsoid's centre from the finger's axis (palmar side)


def pad_spheres(finger="thumb"):
    """The plant's sphere pads of one v4b fingertip, tip frame (mm), from a cached scene of the HOM study (pad builder v2,
    radius 0.75 mm); the thumb's pad faces +x."""
    import re
    s = PAD_SCENE.read_text()
    pads = re.findall(rf'<geom name="{finger}_pad\d+" type="sphere" size="([0-9.e-]+)" pos="([^"]+)"', s)
    P = 1e3 * np.array([[float(v) for v in p_[1].split()] for p_ in pads])
    return P, 1e3 * float(pads[0][0])


def ellipsoid():
    from sr2hand.control.surfaces import HalfEllipsoid
    return HalfEllipsoid(*ELL)


def section(z):
    """The pad's palmar outline in the plane y = 0 of the tip frame: x (mm) from the finger's axis at z (mm)."""
    az = np.where(np.asarray(z) < 0, ELL[2], ELL[3])
    return C_PAD + ELL[0] * np.sqrt(np.clip(1.0 - (np.asarray(z) / az) ** 2, 0.0, 1.0))


def fig_geodesic():
    """(a) The pad in the plant (sphere pads, their contacts with a 40 mm sphere and the centre of pressure) and in the
    controller (the analytic half-ellipsoid onto which the centre of pressure is projected); (b) one period of the
    filter: the process and correction steps and the osculating-circle arc of the geodesic tracer; (c) the gain of
    Eq. (6) against the reference geodesic speed."""
    plt = plt_setup()
    from matplotlib.patches import Circle
    E = ellipsoid()
    cen = np.array([C_PAD, 0.0, 0.0])
    P, rp = pad_spheres()
    dist = np.array([E.closest(p_ - cen)[0] for p_ in P])
    palm = P[:, 0] > C_PAD                         # the palmar half, where objects touch
    env = dist[palm] + rp
    # a 40 mm sphere pressed 0.3 mm into the analytic surface at z = -3 mm on the midline
    R, pen, z0 = 20.0, 0.3, -3.0
    _, q0, n0 = E.closest(np.array([section(z0) - C_PAD + 1.0, 0.0, z0]))
    q0 = q0 + cen
    o = q0 + (R - pen) * n0
    d_o = np.linalg.norm(P - o, axis=1)
    over = R + rp - d_o
    hit = over > 0
    u = (o - P[hit]) / d_o[hit, None]
    cp = P[hit] + u * (rp - 0.5 * over[hit, None])  # the middle of each sphere pair's overlap
    f = over[hit]                                    # a force per contact that grows with the overlap
    cop = (f[:, None] * cp).sum(0) / f.sum()
    _, pl, nl = E.closest(cop - cen)
    pl = pl + cen
    fig = plt.figure(figsize=(8.28, 8.4))
    ax = fig.add_axes((0.075, 0.535, 0.915, 0.43))
    # (a) section through the pad, page: z horizontal (tip to the left), x vertical (palmar up)
    zz = np.linspace(PAD_Z[0], PAD_Z[1], 400)
    for k in np.flatnonzero((np.abs(P[:, 1]) < 0.45)):
        c_ = P[k]
        hk = hit[k]
        ax.add_patch(Circle((c_[2], c_[0]), rp, fc="#f3c39b" if hk else "#f6efe6", ec=C_OBJ if hk else "#c9b8a5",
                            lw=1.0 if hk else 0.6, zorder=2))
    ax.plot(zz, section(zz), color=C_FIN, lw=2.2, zorder=4)
    ax.text(5.6, 11.0, "the controller's surface:\nthe analytic half-ellipsoid", fontsize=12.5, color=C_FIN, va="bottom")
    ax.add_patch(Circle((o[2], o[0]), R, fc=C_OBJ_L, ec="#555555", lw=1.6, zorder=1, alpha=0.8))
    ax.text(-11.8, 10.6, "40 mm sphere", fontsize=12.5, color="#444444", va="bottom")
    on = np.abs(cp[:, 1]) < 0.45
    ax.plot(cp[on, 2], cp[on, 0], "o", ms=3.2, color=C_OBJ, zorder=6)
    ax.plot(cop[2], cop[0], "o", ms=8, mfc="k", mec="white", mew=1.0, zorder=8)
    ax.plot(pl[2], pl[0], "o", ms=9, mfc="none", mec=C_FIN, mew=2.2, zorder=9)
    arrow(ax, (pl[2], pl[0]), (pl[2] + 2.2 * nl[2], pl[0] + 2.2 * nl[0]), AX_X, lw=2.0, ms=13, z=9)
    ax.annotate(f"{int(hit.sum())} contacts of the sphere pads", (cp[on, 2].min(), cp[on, 0][np.argmin(cp[on, 2])]),
                xytext=(-11.8, 12.6), fontsize=12.5, color=C_OBJ, arrowprops=dict(arrowstyle="-", color=C_OBJ, lw=1.0))
    ax.annotate("their centre of pressure: the plant's\nmeasurement (on the bench, the tip's)", (cop[2], cop[0]),
                xytext=(0.3, 12.3), fontsize=12.5, arrowprops=dict(arrowstyle="-", color="k", lw=1.0),
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
    ax.annotate(r"its closest point $p_{L_1}$ on the surface" + "\n" + r"and the normal $x_1$ there", (pl[2], pl[0]),
                xytext=(1.0, 7.0), fontsize=12.5, color=C_FIN, arrowprops=dict(arrowstyle="-", color=C_FIN, lw=1.0),
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))
    ax.text(2.0, 4.6, f"sphere pads of radius {rp:.2f} mm, the plant's contact model: the slice at\ny = 0 of "
            f"{len(P):,}; on the palmar face their outer surface lies within\n0.05 mm of the half-ellipsoid",
            fontsize=12, color="#8a7560", va="center", ha="center")
    ax.set_aspect("equal")
    ax.set_xlim(-12.3, 12.0)
    ax.set_ylim(3.0, 14.2)
    ax.set_xlabel("z along the finger (mm): toward the tip < 0 < toward the PIP")
    ax.set_ylabel("x, palmar (mm)")
    ax.set_title("(a) Section through the v4b pad at y = 0", fontsize=13.5)
    # (b) one filter period on the section (step exaggerated)
    bx = fig.add_axes((0.06, 0.06, 0.52, 0.38))
    zz2 = np.linspace(-10.5, 4.0, 300)
    bx.plot(zz2, section(zz2), color=C_FIN, lw=2.0, zorder=3)
    zc, zm = -1.5, -8.0                            # the filter's point and the measured point (z, mm)
    pc = np.array([section(zc), zc])               # (x, z)
    pm = np.array([section(zm), zm])
    _, qc, ncv = E.closest(np.array([pc[0] - C_PAD, 0.0, pc[1]]))
    nc = np.array([ncv[0], ncv[2]])
    t = np.array([-nc[1], nc[0]])                  # tangent toward the tip (-z), in (x, z)
    if t[1] > 0:
        t = -t
    S3 = E.shape_operator(qc, ncv)
    t3 = np.array([t[0], 0.0, t[1]])
    kap = float(t3 @ S3 @ t3)
    k_gain, dt_x = 5.0, 0.05                      # a gain and a period chosen so that the step shows (not to scale)
    g_ref = -30.0                                  # mm/s along t (toward the tip): a commanded roll
    corr = k_gain * float((pm - pc) @ t) * dt_x   # E projects the error onto the tangent plane
    proc = -g_ref * dt_x                           # mm along t
    step = proc + corr
    oc = pc - nc / kap                             # the osculating circle's centre, (x, z)
    ang = np.linspace(0, 2 * math.pi, 400)
    bx.plot(oc[1] + np.sin(ang) / kap, oc[0] + np.cos(ang) / kap, color="#9a9a9a", lw=1.0, ls=(0, (4, 3)), zorder=2)
    bx.text(1.2, 6.2, f"osculating\ncircle, radius\n{1 / kap:.1f} mm", fontsize=12, color="#777777", ha="center")

    def turn(v, a_):
        return np.array([math.cos(a_) * v[0] - math.sin(a_) * v[1], math.sin(a_) * v[0] + math.cos(a_) * v[1]])
    rv = pc - oc
    sense = 1.0 if np.array([-rv[1], rv[0]]) @ t > 0 else -1.0
    arc = np.array([oc + turn(rv, sense * a_ * kap) for a_ in np.linspace(0.0, step, 60)])
    end = arc[-1]
    _, qe, _ = E.closest(np.array([end[0] - C_PAD, 0.0, end[1]]))
    bx.plot(arc[:, 1], arc[:, 0], color="k", lw=3.0, zorder=6)
    bx.plot(pc[1], pc[0], "o", ms=10, color=C_FIN, zorder=8)
    bx.plot(pm[1], pm[0], "o", ms=9, mfc="white", mec="k", mew=1.8, zorder=8)
    bx.plot(qe[2], qe[0] + C_PAD, "o", ms=13, mfc="none", mec=C_FIN, mew=2.2, zorder=8)
    tt = np.array([t[0], t[1]])
    arrow(bx, (pc[1], pc[0] + 0.9), (pc[1] + proc * tt[1], pc[0] + 0.9 + proc * tt[0]), C_OBJ, lw=2.2, ms=13, z=9)
    arrow(bx, (pc[1] + proc * tt[1], pc[0] + 0.9 + proc * tt[0]),
          (pc[1] + step * tt[1], pc[0] + 0.9 + step * tt[0]), C_LEV, lw=2.2, ms=13, z=9)
    bx.text(pc[1] + 0.5 * proc * tt[1] + 0.3, pc[0] + 1.4 + 0.5 * proc * tt[0], r"$\dot g_{\mathrm{ref}}\Delta t$",
            color=C_OBJ, fontsize=13.5, ha="left", va="bottom")
    bx.text(pc[1] + (proc + 0.5 * corr) * tt[1] - 0.3, pc[0] + 1.4 + (proc + 0.5 * corr) * tt[0],
            r"$k\,E\,(p_L - p_C)\,\Delta t$", color=C_LEV, fontsize=13.5, ha="right", va="bottom")
    bx.annotate(r"$p_C[t]$", (pc[1], pc[0]), xytext=(pc[1] + 0.4, pc[0] - 1.5), fontsize=13.5, color=C_FIN)
    bx.annotate(r"$p_L[t]$, measured", (pm[1], pm[0]), xytext=(pm[1] + 1.2, pm[0] - 2.4), fontsize=13.5,
                arrowprops=dict(arrowstyle="-", lw=0.8))
    bx.annotate(r"$p_C[t+1]$", (qe[2], qe[0] + C_PAD), xytext=(qe[2] - 0.6, qe[0] + C_PAD - 2.6), fontsize=13.5,
                color=C_FIN, arrowprops=dict(arrowstyle="-", lw=0.8, color=C_FIN))
    bx.set_aspect("equal")
    bx.set_xlim(-10.5, 3.5)
    bx.set_ylim(4.5, 13.5)
    bx.set_xlabel("z (mm)")
    bx.set_ylabel("x (mm)")
    bx.set_title("(b) One filter period (lengths not to scale)", fontsize=13.5)
    # (c) the gain of Eq. (6)
    cx = fig.add_axes((0.69, 0.06, 0.30, 0.38))
    gs = np.linspace(0, 60, 400)
    kk = 5.0 + 95.0 * (1 - np.exp(-3 * gs / 20.0))
    cx.plot(gs, kk, color="k", lw=2.0)
    cx.axvline(20.0, color="#999999", lw=1.0, ls=":")
    cx.text(21.0, 60.0, r"$\alpha$ = 20 mm/s", fontsize=12, color="#666666")
    for kdt, hz, col in ((100, "100 Hz", C_FIN), (50, "50 Hz", C_OBJ)):
        cx.axhline(kdt, color=col, lw=1.2, ls="--")
        cx.text(59, kdt - 2.5, f"$k\\,\\Delta t = 1$ at {hz}", color=col, fontsize=11.5, ha="right", va="top")
    cx.plot([0], [5.0], "o", ms=8, color=C_LEV, zorder=5)
    cx.annotate("the four tasks:\nno commanded\nroll or slide", (0, 5.0), xytext=(24, 14), fontsize=12, color=C_LEV,
                arrowprops=dict(arrowstyle="-", color=C_LEV, lw=1.0))
    cx.set_xlim(-1, 60)
    cx.set_ylim(0, 110)
    cx.set_xlabel(r"reference speed $|\dot g_{\mathrm{ref}}|$ (mm/s)")
    cx.set_ylabel(r"gain $k$ (1/s)")
    cx.set_title("(c) Gain, Eq. (6)", fontsize=13.5)
    cx.grid(color="#eeeeee")
    save(fig, "20261106-geodesic_filter.png")
    # the tracer's error on the real scale: a 0.2 mm step (20 mm/s for 10 ms) along the pad's tip end
    from sr2hand.control.surfaces import HalfEllipsoid
    errs = []
    for zs in np.linspace(-9.0, 9.0, 37):
        _, q, n = E.closest(np.array([section(zs) - C_PAD + 1.0, 0.0, zs]))
        for tv in (np.array([0.0, 1.0, 0.0]), np.cross(n, [0.0, 1.0, 0.0])):
            tv = tv - (tv @ n) * n
            tv /= np.linalg.norm(tv)
            for st in (0.2e-3 * 1e3, 1.0):
                kq = float(tv @ E.shape_operator(q, n) @ tv)
                e_ = q + (math.sin(kq * st) / kq) * tv - ((1 - math.cos(kq * st)) / kq) * n
                errs.append((st, abs(E.closest(e_)[0])))
    er = {st: max(e for s_, e in errs if s_ == st) for st in (0.2, 1.0)}
    return dict(n_pads=int(len(P)), r_pad=rp, n_palmar=int(palm.sum()),
                palmar_envelope_within_0p05mm=float(np.mean(np.abs(env) < 0.05)),
                palmar_envelope_p95_mm=float(np.percentile(np.abs(env), 95)),
                spacing_mm=float(np.median(np.sort(np.linalg.norm(P[:, None] - P[None], axis=2), 1)[:, 1])),
                contacts=int(hit.sum()), cop_to_surface_mm=float(np.linalg.norm(cop - pl)),
                arc_end_off_surface_um={str(k): 1e3 * v for k, v in er.items()}, kappa_at_pc=kap)


def main():
    which = sys.argv[1:] or ["eq1", "adjoint"]
    out = json.loads(NUMBERS.read_text()) if NUMBERS.exists() else {}
    for w in which:
        r = {"eq1": fig_eq1, "adjoint": fig_adjoint, "geodesic": fig_geodesic}[w]()
        if isinstance(r, dict):
            out[w] = r
            print(w, json.dumps(r))
    NUMBERS.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
