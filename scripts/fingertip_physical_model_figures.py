#!/usr/bin/env python3
"""Figures and numbers of the page on the magnetic fingertip's physical model (docs/experiments/20261026-fingertip_physical_model/):
the pad in section from the slicer's material classes, the finite elements under a press and under shear, the magnets'
field for two pole patterns, the reading change a press makes at the five chips, the estimator's likelihood over the pad,
and the untrained model against the UR5 sessions.

Two magnet layouts (owner, 2026-10-28: "i think we will stick with the 4 magnet config"; the five-magnet figures stay on
the page in a collapsed view):
  four  pad 05R4 of the v3 tips: four N52 magnets in a 5.0 mm square centred on chip U1, poles (1, -1, -1, 1) in the
        order mount -y, mount +y, tip -y, tip +y (FE case b_05R4, field-model case four_05R4); media/20261028-*_four.png
  five  pad B05R: a centre magnet and four corners (FE case b_v2); media/20261026-*.png

Inputs: the FE outputs of scripts/fingertip_physical_model_fe.py (data/20261026-fe_sections.npz and
20261026-aux_class_sections.npz; with --case b_05R4 data/20261028-fe_sections_05R4.npz and
20261028-aux_class_sections_05R4.npz), the estimator's surface bases (scripts/fingertip_wrench_fe.py --surface), the field
model (scripts/fingertip_wrench_model.py), the UR5 sessions' summary.json and calib_rest.json, and for the four magnets:
  data/20261028-v3id1_calib.json  v3 ID 1's rest fit: the session's calib_rest.json (logs/20261028-spherev3id1_grid_phys/,
                                  the UR5 coordinator's analysis of 2026-10-28) with the strength and offsets of the rest
                                  fit bounded to v2 ID 1's board offset, from the coordinator's fe_calibration.json
  20261028-fe_calibration.json    (docs/experiments/20261027-v3_tip_calibration_and_crosstalk/data/) the untrained model's
                                  scores of v3 ID 1 and v2 ID 1 on the same 818 holds of the low sphere's grid stage

    .venv/bin/python scripts/fingertip_physical_model_figures.py [--layout four|five|both] [--only NAME ...]
writes media/*.png and data/20261026-numbers.json in the page's folder.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src/morphohand/driver/reskin5x/host"))
import fingertip_wrench_model as W  # noqa: E402

D = ROOT / "docs/experiments/20261026-fingertip_physical_model"
DATA, MEDIA = D / "data", D / "media"
FE_NPZ = DATA / "20261026-fe_sections.npz"
AUX_NPZ = DATA / "20261026-aux_class_sections.npz"
NUMBERS = DATA / "20261026-numbers.json"
EST_DATA = ROOT / "docs/experiments/20261020-fingertip_contact_estimation/data"
BASIS = EST_DATA / "20261020-b_v2_surface_basis.npz"
UR5 = ROOT / "docs/experiments/20261022-ur5_tip_calibration/data"
SESSIONS = [("20261022-210852-sphere2", "Tip 0 (sphere2), magnets one way"),
            ("20261023-203938-cb1t", "Tip 1 (cb1t), checkerboard"),
            ("20261025-114653-cb1r", "Tip 1 again (cb1r)"),
            ("20261025-152920-cb2t", "Tip 2 (cb2t), checkerboard")]
CB2T = UR5 / "20261025-152920-cb2t/calib_rest.json"
V3_CAL = DATA / "20261028-v3id1_calib.json"
FE_CAL = ROOT / "docs/experiments/20261027-v3_tip_calibration_and_crosstalk/data/20261028-fe_calibration.json"
CHECKER = (1, -1, -1, -1, 1)          # poles toward the palm in the FE row's order (mount -y, mount +y, centre, tip -y, tip +y)
ALL_N = (1, 1, 1, 1, 1)
CHECKER4 = (1, -1, -1, 1)             # pad 05R4 (mount -y, mount +y, tip -y, tip +y): each diagonal one way
ALL_N4 = (1, 1, 1, 1)
RED, BLUE = "#c0392b", "#2e5fa8"
FOUR_C, FIVE_C = "#b06010", "#2e5fa8"  # the calibration page's colours of the two tips
TPU_SOLID, TPU_CORE = "#b89b72", "#efe6d6"
U_NAMES = ("U1", "U2", "U3", "U4", "U5")
NUM = {}
# Each layout: field-model case, pole patterns, FE outputs, surface basis, the FE section drawn (0: y = 0, through the
# five-magnet pad's centre magnet; 1: y = 2.5 mm, through two of the four magnets), file names and the numbers' key suffix
LAYOUTS = dict(
    five=dict(case="b_v2", checker=CHECKER, all_n=ALL_N, fe=FE_NPZ, aux=AUX_NPZ, basis=BASIS, sec=0,
              name="20261026-{}.png", key=""),
    four=dict(case="four_05R4", checker=CHECKER4, all_n=ALL_N4, fe=DATA / "20261028-fe_sections_05R4.npz",
              aux=DATA / "20261028-aux_class_sections_05R4.npz", basis=EST_DATA / "20261028-b_05R4_surface_basis.npz",
              sec=1, name="20261028-{}_four.png", key="_four"))
WORD = {4: "four", 5: "five"}


def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 13, "axes.titlesize": 13, "axes.labelsize": 13,
                         "legend.fontsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12, "mathtext.fontset": "dejavuserif"})
    return plt


def save(fig, name):
    MEDIA.mkdir(parents=True, exist_ok=True)
    fig.savefig(MEDIA / name, dpi=150, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("wrote", (MEDIA / name).relative_to(ROOT), flush=True)


def design_model(poles, k=1.0, case="b_v2"):
    """The field model with the design's magnets (FE row b_v2, or pad 05R4's four on it), the chips in their design
    places, the chip frames of the board (theta0 95.9 deg, the UR5 sessions' fit) and the stream order of the board."""
    return W.Model(W.Calib(case=case, poles=tuple(poles), k=k, theta0=95.89448285681775, perm=(0, 2, 4, 1, 3)))


def chips_U():
    """Chip sensing points in U order (U1-U5), tip frame (mm)."""
    return W.chip_points()


def stream_to_U(model):
    """Index of each U position in the model's stream order."""
    return [list(model.cal.perm).index(u) for u in range(5)]


def field_at(model, pts):
    """Magnets' field (T, tip frame) at points (n, 3) mm."""
    off = model.q_off[None].repeat(len(model.mc), 0)
    P = (model.mc[:, None, :] + off).reshape(-1, 3)
    mom = (model.mu[:, None, :] * model.q_w[None, :, None]).reshape(-1, 3)
    out = np.zeros((len(pts), 3))
    for s in range(0, len(pts), 4096):
        out[s:s + 4096] = W.dipole_field(pts[s:s + 4096], P, mom)
    return out


def pad_outline_yz():
    """The pad's outline seen from the palm (y, z), from the surface basis's points."""
    B = W.load_basis(BASIS)
    from scipy.spatial import ConvexHull
    P = B["pts"][:, 1:]
    h = ConvexHull(P)
    v = np.r_[h.vertices, h.vertices[:1]]
    return P[v]


def blobs(mask):
    """Index bounding boxes (i0, i1, k0, k1), end exclusive, of the connected parts of a 2-D mask."""
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    out = []
    for b in range(1, n + 1):
        idx = np.argwhere(lab == b)
        out.append((idx[:, 0].min(), idx[:, 0].max() + 1, idx[:, 1].min(), idx[:, 1].max() + 1))
    return sorted(out, key=lambda t: t[2])


# ------------------------------------------------------------------------------------------------ figure 1: anatomy
def fig_anatomy(lay="five"):
    Lg = LAYOUTS[lay]
    plt = plt_setup()
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Circle, Rectangle
    A = np.load(Lg["aux"])
    s = Lg["sec"]
    cls, o, h = A["cls"], A["origin"], float(A["h"])
    yv = float(A["y"][s])
    mc, _ = W.magnets(W.fe_row(Lg["case"]))
    fig = plt.figure(figsize=(10.0, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.0, 2.4), wspace=0.12)
    ax = fig.add_subplot(gs[1])
    # (b) a section along the finger: x up (toward the palm), z to the right (toward the tip end)
    c0 = cls[s].astype(float)                         # (nx, nz)
    nx, nz = c0.shape
    xe = o[0] + h * np.arange(nx + 1)
    ze = o[2] + h * np.arange(nz + 1)
    img = np.full(c0.shape, np.nan)
    for v in (1, 2, 3, 4):
        img[c0 == v] = v
    cmap = ListedColormap([TPU_SOLID, "#9c7f58", TPU_CORE, "#888888"])
    ax.pcolormesh(ze, xe, np.ma.masked_invalid(img), cmap=cmap, vmin=0.5, vmax=4.5, shading="flat", rasterized=True)
    # the gyroid's beads in the core: zero set of g at the 30 % setting's period (scripts/tpu_tip_fem_cells.py)
    L = 2 * math.pi * 0.407 / (2.44 * 0.3)
    xc = 0.5 * (xe[1:] + xe[:-1])
    zc = 0.5 * (ze[1:] + ze[:-1])
    Zg, Xg = np.meshgrid(zc, xc)
    k = 2 * math.pi / L
    g = np.sin(k * Xg) * np.cos(k * yv) + np.sin(k * yv) * np.cos(k * Zg) + np.sin(k * Zg) * np.cos(k * Xg)
    g = np.where(c0 == 3, g, np.nan)
    ax.contour(Zg, Xg, g, levels=[0.0], colors=["#a88a5e"], linewidths=1.6)
    # the magnets the section cuts, each with the pole that faces the palm on top (red north, blue south)
    tip_edge = None
    for i0, i1, k0, k1 in blobs(c0 == 4):
        x0, x1, z0, z1 = xe[i0], xe[i1], ze[k0], ze[k1]
        j = int(np.argmin(np.abs(mc[:, 2] - 0.5 * (z0 + z1)) + 10 * np.abs(mc[:, 1] - yv)))
        pole = Lg["checker"][j]
        top, bot = (RED, BLUE) if pole > 0 else (BLUE, RED)
        xm = 0.5 * (x0 + x1)
        ax.add_patch(Rectangle((z0, xm), z1 - z0, x1 - xm, fc=top, ec="black", lw=0.8, zorder=3))
        ax.add_patch(Rectangle((z0, x0), z1 - z0, xm - x0, fc=bot, ec="black", lw=0.8, zorder=3))
        ax.text(0.5 * (z0 + z1), x1 + 0.35, "N" if pole > 0 else "S", ha="center", va="bottom", fontsize=12, color=top,
                weight="bold")
        ax.text(0.5 * (z0 + z1), x0 - 0.35, "S" if pole > 0 else "N", ha="center", va="top", fontsize=12, color=bot,
                weight="bold")
        tip_edge = (z1, x0 + 0.5)
    # board and chips under the cut face; chips off the section (y = 0 behind y = 2.5 mm) in grey
    xs_plane = float(chips_U()[0, 0])
    behind = abs(yv) > 0.5
    ax.add_patch(Rectangle((-1.2, xs_plane - 1.9), 23.8, 1.0, fc="#7fae7f", ec="#3d6b3d", lw=0.8))
    for u, (x, y, z) in zip(U_NAMES, chips_U()):
        if abs(y) < 0.1:
            ax.add_patch(Rectangle((z - 1.5, xs_plane - 0.9), 3.0, 0.9, fc="#9a9a9a" if behind else "#2b2b2b",
                                   ec="black", lw=0.6))
            ax.plot(z, xs_plane, "o", ms=4.5, color="#f2c200", mec="black", mew=0.5, zorder=5)
            ax.text(z, xs_plane - 2.1, u, ha="center", va="top", fontsize=12)
    # the 5 mm sphere at first touch over the apex (y 0.5 mm), cut by the section
    R = 5.0
    r = math.sqrt(max(R ** 2 - (yv - 0.5) ** 2, 0.0))
    th = np.linspace(math.pi * 1.08, math.pi * 1.92, 200)
    ax.fill_between(10.36 + r * np.cos(th), 8.5 + R + r * np.sin(th), 8.5 + R + 0.2, color="#d8dde3", ec="#555555", lw=1.0)
    ax.text(10.36 - 5.4, 12.4, "Sphere indenter,\nradius 5 mm", fontsize=12, ha="right", va="center")
    # labels
    lab = dict(fontsize=12, ha="left", va="center")
    ax.annotate("Walls: two 0.45 mm\nperimeters", xy=(19.6, 6.4), xytext=(23.4, 9.4), arrowprops=dict(arrowstyle="-", lw=0.8), **lab)
    ax.annotate("Skins: four 0.2 mm\nlayers", xy=(21.2, 2.0), xytext=(23.4, 5.2), arrowprops=dict(arrowstyle="-", lw=0.8), **lab)
    many = lay == "four"
    ax.annotate("Gyroid core,\n30 % setting", xy=(18.0, 1.5) if many else (16.6, -3.0), xytext=(23.4, 1.0),
                arrowprops=dict(arrowstyle="-", lw=0.8), **lab)
    ax.annotate(("Magnets" if many else "Magnet") + ", 3 × 2 mm,\n7.5 mm under the apex",
                xy=tip_edge if many else (11.9, -0.5), xytext=(23.4, -3.0), arrowprops=dict(arrowstyle="-", lw=0.8), **lab)
    ax.annotate("Board: five MLX90393\nchips, 7 mm apart", xy=(19.5, xs_plane - 1.4), xytext=(23.4, -7.2),
                arrowprops=dict(arrowstyle="-", lw=0.8), **lab)
    ax.annotate("", xy=(-1.2, 3.0), xytext=(-1.2, -2.0), arrowprops=dict(arrowstyle="-|>", lw=1.2))
    ax.text(-1.6, 0.5, "x, toward the palm", rotation=90, ha="right", va="center", fontsize=12)
    ax.annotate("", xy=(4.0, -10.4), xytext=(-1.0, -10.4), arrowprops=dict(arrowstyle="-|>", lw=1.2))
    ax.text(4.4, -10.4, "z, toward the tip end", ha="left", va="center", fontsize=12)
    ax.set_xlim(-3.2, 34.5)
    ax.set_ylim(-11.2, 14.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(b) Section along the finger at y = 2.5 mm, through two magnets" if many else
                 "(b) Section along the finger through the centre magnet", fontsize=13)
    # (a) seen from the palm: magnets with their poles and the chips beneath
    ax2 = fig.add_subplot(gs[0])
    out = pad_outline_yz()
    ax2.fill(out[:, 0], out[:, 1], color=TPU_CORE, ec=TPU_SOLID, lw=2.0)
    for u, (x, y, z) in zip(U_NAMES, chips_U()):
        ax2.add_patch(Rectangle((y - 1.5, z - 1.5), 3.0, 3.0, fc="white", ec="0.35", lw=1.0, ls="--", zorder=1))
        dy, dz, ha = {"U1": (2.0, -2.6, "left"), "U2": (0, 2.0, "center"), "U3": (0, -2.4, "center"),
                      "U4": (0, -2.4, "center"), "U5": (0, -2.4, "center")}[u]
        if many and u == "U1":
            dy, dz, ha = 0.0, 0.0, "center"          # U1 sits in the middle of the four magnets' square
        ax2.text(y + dy, z + dz, u, fontsize=12, color="0.3", ha=ha, va="center")
    for (y, z), sg in zip(mc[:, 1:], Lg["checker"]):
        ax2.add_patch(Circle((y, z), 1.5, fc=RED if sg > 0 else BLUE, ec="black", lw=0.8, zorder=2))
        ax2.text(y, z, "N" if sg > 0 else "S", ha="center", va="center", color="white", fontsize=11, weight="bold", zorder=3)
    if many:
        ax2.axvline(yv, color="0.25", lw=1.0, ls=":", zorder=4)
        ax2.text(yv + 0.4, 22.6, "section (b)", fontsize=11, color="0.25", ha="left", va="top")
    ax2.set_xlim(-10.6, 10.6)
    ax2.set_ylim(-1.5, 23.5)
    ax2.set_aspect("equal")
    ax2.set_xlabel("y, across the finger (mm)")
    ax2.set_ylabel("z, from the mount end (mm)")
    ax2.set_title("(a) Seen from the palm", fontsize=13)
    save(fig, Lg["name"].format("fingertip_anatomy"))


# ------------------------------------------------------------------------------------------------ figure 2: the chain
def fig_chain(lay="five"):
    Lg = LAYOUTS[lay]
    plt = plt_setup()
    from matplotlib.patches import FancyBboxPatch
    n = len(Lg["checker"])
    dof = int(np.load(Lg["fe"])["n_dof"])
    B = W.load_basis(Lg["basis"])
    n_pts = len(B["pts"])
    n_cand = len(W.subdivide(B["pts"], B["normals"], B["tris"], 2)[0])
    NUM.setdefault("chain", {})[lay] = dict(magnets=n, dof=dof, fe_points=n_pts, candidates=n_cand)
    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(-2, 57)
    ax.axis("off")

    def box(x, y, w, hgt, title, body, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, hgt, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec="black", lw=1.0))
        ax.text(x + w / 2, y + hgt - 1.6, title, ha="center", va="top", fontsize=12.5, weight="bold")
        ax.text(x + w / 2, y + hgt - 5.8, body, ha="center", va="top", fontsize=11, linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=1.4, color="black"))

    fwd = "#eef3fa"
    inv = "#fbf1e4"
    box(0.6, 31, 22.6, 19, "Printed pad", f"Siraya 85A TPU\n2 walls, 4 skins\n30 % gyroid core\n{n} magnets, 3 × 2 mm", fwd)
    box(25.9, 31, 22.6, 19, "Finite elements", f"Voxels of 0.05 mm\nhomogenized core\n{dof:,} unknowns\n"
        f"{n_pts} points × 3 loads", fwd)
    box(51.2, 31, 22.6, 19, "Magnet motion", f"M(p): translation and\nturn of each magnet\nper newton at point p\n"
        f"{6 * n} numbers a load", fwd)
    box(76.5, 31, 22.6, 19, "Field model", "72 dipoles a magnet\nJ: change of the 15\nreadings per µm\nand per mrad", fwd)
    arrow(23.6, 40.5, 25.5, 40.5)
    arrow(48.9, 40.5, 50.8, 40.5)
    arrow(74.2, 40.5, 76.1, 40.5)
    box(62.5, 3, 36.6, 19, "Estimator", "S(p) = J M(p): change per newton\nΔb = S(p) f + Earth's field + noise\nbest p "
        f"and f by the marginal\nlikelihood over {n_cand:,} points", inv)
    ax.annotate("", xy=(84, 22.6), xytext=(88, 30.6), arrowprops=dict(arrowstyle="-|>", lw=1.4))
    box(0.6, 3, 27, 19, "Rest calibration", "2 s untouched: the 15\nreadings at rest fix the\npoles, the strength k,\nthe chip map and offsets", inv)
    ax.annotate("", xy=(79.0, 30.6), xytext=(14.0, 22.6), arrowprops=dict(arrowstyle="-|>", lw=1.2, ls="--",
                                                                         connectionstyle="arc3,rad=-0.12"))
    box(31.0, 3, 27.5, 19, "Board reading", "15 numbers, µT\n141 Hz per chip\nnoise 1.5–2.6 µT\nless the rest: Δb", inv)
    arrow(58.9, 12.5, 62.1, 12.5)
    ax.text(50, 54.5, "Forward model, computed once per pad design", ha="center", fontsize=11.5, color="0.25")
    ax.text(50, -0.6, "Inverse model, run at every frame of the board", ha="center", fontsize=11.5, color="0.25")
    save(fig, Lg["name"].format("model_chain"))


# ------------------------------------------------------------------------------------------------ figure 3: FE sections
def fig_fe(lay="five"):
    Lg = LAYOUTS[lay]
    plt = plt_setup()
    from matplotlib.patches import Rectangle, Circle
    Z = np.load(Lg["fe"])
    s = Lg["sec"]
    xs, zs = Z["xs"], Z["zs"]
    cls = Z["cls"][s]
    X = Z[f"sec{s}_X"]
    ysec = float(np.median(X[:, 1]))
    ix = np.searchsorted(xs, X[:, 0] - 1e-9)
    iz = np.searchsorted(zs, X[:, 2] - 1e-9)
    loads = [("press1", "(a) Press, 1 N along the normal"), ("shear_z", "(b) Shear, 1 N along the finger"),
             ("shear_y", "(c) Shear, 1 N across the finger")]
    fig = plt.figure(figsize=(10.0, 12.6))
    gs = fig.add_gridspec(3, 4, width_ratios=(2.25, 0.08, 0.62, 1.0), wspace=0.06, hspace=0.32)   # room for the colorbar label
    mc = Z["magnet_nodes_centre"]
    in_sec = [j for j in range(len(mc)) if abs(mc[j, 1] - ysec) < 1.0]
    out = pad_outline_yz()
    mags = blobs(cls == 2)
    for row, (name, title) in enumerate(loads):
        ax = fig.add_subplot(gs[row, 0])
        cax = fig.add_subplot(gs[row, 1])
        u = Z[f"{name}_sec{s}_u"]
        U = np.full((len(xs), len(zs)), np.nan)
        U[ix, iz] = np.linalg.norm(u, axis=1)
        vmax = float(np.nanpercentile(U, 99.5))
        Zg, Xg = np.meshgrid(zs, xs)
        pc = ax.pcolormesh(Zg, Xg, U, cmap="magma_r", vmin=0, vmax=vmax, shading="gouraud", rasterized=True)
        for i in range(len(xs) - 1):
            for k in range(len(zs) - 1):
                if cls[i, k] >= 0:
                    ax.add_patch(Rectangle((zs[k], xs[i]), zs[k + 1] - zs[k], xs[i + 1] - xs[i], fc="none", ec="white",
                                           lw=0.25, alpha=0.5))
        for i0, i1, k0, k1 in mags:
            ax.add_patch(Rectangle((zs[k0], xs[i0]), zs[k1] - zs[k0], xs[i1] - xs[i0], fc="none", ec="#1f4f99", lw=1.8,
                                   zorder=4))
        top = {}
        for (x, y, z), du in zip(X, u):
            if z not in top or x > top[z][0]:
                top[z] = (x, du)
        zz = np.array(sorted(top))
        xt = np.array([top[z][0] for z in zz])
        dut = np.array([top[z][1] for z in zz])
        mag = 1.2e-3 * 1000 / max(vmax, 1e-9)                       # the largest displacement drawn as 1.2 mm
        ax.plot(zz, xt, "-", color="0.55", lw=1.0)
        ax.plot(zz + mag * dut[:, 2], xt + mag * dut[:, 0], "-", color="black", lw=1.6)
        mot = Z[f"{name}_magnets"]
        for j in in_sec:
            c, m = mc[j], mot[j]
            ax.annotate("", xy=(c[2] + 0.3 * m[2], c[0] + 0.3 * m[0]), xytext=(c[2], c[0]),
                        arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#1f4f99"), zorder=5)
        ttl = title + (f": the face sinks {float(Z[name + '_approach_um']):.0f} µm" if name.startswith("press") else "")
        ax.set_title(ttl, fontsize=13)
        cb = fig.colorbar(pc, cax=cax)
        cb.set_label("Displacement (µm)")
        ax.set_aspect("equal")
        ax.set_xlim(-0.3, 21.8)
        ax.set_ylim(-5.4, 9.8)
        ax.set_ylabel("x (mm)")
        if row == 2:
            ax.set_xlabel("z, from the mount end toward the tip end (mm)")
        # palm view: each magnet's translation in the face's plane (arrow, 0.25 mm per µm) and along x (label)
        a2 = fig.add_subplot(gs[row, 3])
        a2.fill(out[:, 0], out[:, 1], color="#f6f1e8", ec="0.6", lw=0.8)
        P = Z[f"{name}_contact_X"]
        a2.plot(np.mean(P[:, 1]), np.mean(P[:, 2]), "*", ms=14, mfc="#f2c200", mec="black", zorder=5)
        for cc, mm, s_ in zip(mc, mot, Lg["checker"]):
            a2.add_patch(Circle((cc[1], cc[2]), 1.5, fc=RED if s_ > 0 else BLUE, ec="black", lw=0.6, alpha=0.35, zorder=2))
            a2.annotate("", xy=(cc[1] + 0.25 * mm[1], cc[2] + 0.25 * mm[2]), xytext=(cc[1], cc[2]),
                        arrowprops=dict(arrowstyle="-|>", lw=1.8, color="black", shrinkA=0, shrinkB=0), zorder=4)
            a2.text(cc[1] + (1.7 if cc[1] > 0 else -1.7), cc[2] - 1.6, f"{mm[0]:+.1f}", fontsize=11,
                    ha="left" if cc[1] > 0 else "right", va="center", color="#1f4f99")
        a2.set_xlim(-10.5, 10.5)
        a2.set_ylim(-1.0, 23.0)
        a2.set_aspect("equal")
        a2.set_title("Magnets, seen from the palm", fontsize=12.5)
        a2.set_xticks([-10, 0, 10])
        if row == 2:
            a2.set_xlabel("y (mm)")
        a2.set_ylabel("z (mm)")
    save(fig, Lg["name"].format("fe_sections"))
    NUM["fe" + Lg["key"]] = {n: dict(approach_um=float(Z[n + "_approach_um"]) if n.startswith("press") else None,
                                     magnet_travel_um=np.round(np.linalg.norm(Z[n + "_magnets"][:, :3], axis=1), 2).tolist(),
                                     magnet_x_um=np.round(Z[n + "_magnets"][:, 0], 2).tolist(),
                                     magnet_turn_mrad=np.round(np.linalg.norm(Z[n + "_magnets"][:, 3:], axis=1), 3).tolist())
                             for n in ("press1", "press3", "shear_z", "shear_y")}


# ------------------------------------------------------------------------------------------------ figure 4: the field
def fig_field(lay="five"):
    Lg = LAYOUTS[lay]
    plt = plt_setup()
    from matplotlib.patches import Rectangle
    from matplotlib.colors import LogNorm, TwoSlopeNorm
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 10.2), gridspec_kw=dict(height_ratios=(0.9, 1.1)))
    chips = chips_U()
    out = pad_outline_yz()
    import tip_codesign_geometry as TG
    sh = TG.shape("tcd_ell7.4x8.5x11x22")
    many = lay == "four"
    ysec = 2.5 if many else 0.0                      # the section through magnets: the centre one, or two of the four
    zprof = np.linspace(0.0, 21.41, 300)
    xprof = sh.fe_surface_x(np.full_like(zprof, ysec), zprof)
    nums = {}
    w = WORD[len(Lg["checker"])]
    pats = ((Lg["all_n"], f"All {w} north toward the palm"),
            (Lg["checker"], "Checkerboard (pad 05R4)" if many else "Checkerboard (tips 1 and 2)"))
    for col, (poles, title) in enumerate(pats):
        M = design_model(poles, case=Lg["case"])
        # (top) a section along the finger: field magnitude and field lines
        ax = axes[0, col]
        z = np.linspace(-2.0, 23.0, 260)
        x = np.linspace(-8.0, 10.0, 190)
        Zg, Xg = np.meshgrid(z, x)
        P = np.c_[Xg.ravel(), np.full(Xg.size, ysec), Zg.ravel()]
        Bv = field_at(M, P).reshape(len(x), len(z), 3) * 1e3        # mT
        inmag = np.zeros(Xg.shape, bool)
        for c in M.mc:
            if abs(c[1] - ysec) < 1e-6:
                inmag |= (np.abs(Xg - c[0]) <= 1.0) & (np.abs(Zg - c[2]) <= 1.5)
        Bn = np.linalg.norm(Bv, axis=2)
        Bn[inmag] = np.nan
        ax.pcolormesh(Zg, Xg, Bn, norm=LogNorm(1, 300), cmap="Greys", shading="auto", rasterized=True, alpha=0.75)
        bx, bz = Bv[..., 0].copy(), Bv[..., 2].copy()
        bx[inmag], bz[inmag] = np.nan, np.nan
        ax.streamplot(z, x, bz, bx, color="#3b3b3b", density=1.25, linewidth=0.7, arrowsize=0.8)
        ax.plot(zprof, xprof, "-", color=TPU_SOLID, lw=2.2)
        ax.plot([0, 21.41], [-4.95, -4.95], "-", color=TPU_SOLID, lw=2.2)
        for c, sg in zip(M.mc, poles):
            if abs(c[1] - ysec) < 1e-6:
                ax.add_patch(Rectangle((c[2] - 1.5, c[0]), 3.0, 1.0, fc=RED if sg > 0 else BLUE, ec="black", lw=0.8, zorder=4))
                ax.add_patch(Rectangle((c[2] - 1.5, c[0] - 1.0), 3.0, 1.0, fc=BLUE if sg > 0 else RED, ec="black", lw=0.8,
                                       zorder=4))
        for (cx, cy, cz) in chips:
            if abs(cy) < 0.1:                         # U4, U1 and U2; grey when they lie behind the section
                ax.add_patch(Rectangle((cz - 1.5, cx - 0.9), 3.0, 0.9, fc="#9a9a9a" if many else "#2b2b2b", ec="black",
                                       lw=0.6, zorder=4))
        ax.set_xlim(-1.0, 22.4)
        ax.set_ylim(-7.5, 9.5)
        ax.set_aspect("equal")
        ax.set_title(f"({'ab'[col]}) {title}", fontsize=13)
        ax.set_xlabel("z (mm)")
        if col == 0:
            ax.set_ylabel(f"x (mm), section y = {ysec:g}" + (" mm" if many else ""))
        # (bottom) sensing plane: normal component (colour) and in-plane field (arrows)
        ax = axes[1, col]
        y = np.linspace(-11.0, 11.0, 221)
        z = np.linspace(-2.0, 23.5, 256)
        Yg, Zg = np.meshgrid(y, z)
        P = np.c_[np.full(Yg.size, chips[0, 0]), Yg.ravel(), Zg.ravel()]
        Bv = field_at(M, P).reshape(len(z), len(y), 3) * 1e3
        pc2 = ax.pcolormesh(Yg, Zg, Bv[..., 0], cmap="RdBu_r", norm=TwoSlopeNorm(0, -30, 30), shading="auto", rasterized=True)
        sk = (slice(6, None, 13), slice(6, None, 13))
        ax.quiver(Yg[sk], Zg[sk], Bv[..., 1][sk], Bv[..., 2][sk], color="#333333", scale=180, width=0.004)
        ax.plot(out[:, 0], out[:, 1], "-", color="0.35", lw=1.0)
        Bc = field_at(M, chips) * 1e3
        for u, (cx, cy, cz), b in zip(U_NAMES, chips, Bc):
            ax.add_patch(Rectangle((cy - 1.5, cz - 1.5), 3.0, 3.0, fc="none", ec="black", lw=1.2, zorder=4))
            ax.text(cy, cz - 2.0, f"{u} {np.linalg.norm(b):.1f}", ha="center", va="top", fontsize=11,
                    bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.6), zorder=5)
        nums["all_N" if col == 0 else "checkerboard"] = {u: round(float(np.linalg.norm(b)), 3) for u, b in zip(U_NAMES, Bc)}
        ax.set_xlim(-11.0, 11.0)
        ax.set_ylim(-2.0, 23.5)
        ax.set_aspect("equal")
        ax.set_xlabel("y (mm)")
        if col == 0:
            ax.set_ylabel(f"z (mm), sensing plane x = {chips[0, 0]:.2f}")
        ax.set_title(f"({'cd'[col]}) {'All north' if col == 0 else 'Checkerboard'}: field at the chips' plane", fontsize=13)
    cb = fig.colorbar(pc2, ax=axes[1, :], orientation="horizontal", fraction=0.04, pad=0.09)
    cb.set_label("Field normal to the board, toward the palm (mT)")
    save(fig, Lg["name"].format("magnet_field"))
    NUM["field_at_chips_mT_N52" + Lg["key"]] = nums


# ------------------------------------------------------------------------------------------------ figure 5: press patterns
def signature(model, basis, i, f):
    """Reading change (uT, tip frame, U order (5, 3)) for force f (N, tip frame) at basis point i."""
    J = model.motion_jacobian()
    S = np.einsum("ij,kj->ik", J, basis["motions"][i].reshape(3, -1))        # (15, 3) per N along x, y, z
    d = (S @ f).reshape(5, 3)                                                 # stream order, chip frames
    tip = np.einsum("cji,cj->ci", model.R, d)                                 # back to the tip frame
    return tip[stream_to_U(model)], S


def pick(basis, x=None, y=0.0, z=None, n=None):
    P, N = basis["pts"], basis["normals"]
    score = np.zeros(len(P))
    if n is not None:
        score += 6.0 * (1 - N @ np.asarray(n) / np.linalg.norm(n))
    if y is not None:
        score += np.abs(P[:, 1] - y)
    if z is not None:
        score += np.abs(P[:, 2] - z)
    if x is not None:
        score += np.abs(P[:, 0] - x)
    return int(np.argmin(score))


def fig_patterns(lay="five"):
    Lg = LAYOUTS[lay]
    plt = plt_setup()
    from matplotlib.patches import Circle, Rectangle
    basis = W.load_basis(Lg["basis"])
    M = design_model(Lg["checker"], case=Lg["case"])
    P, N = basis["pts"], basis["normals"]
    apex = int(np.argmax(P[:, 0]))
    cases = [("(a) Press at the face centre", apex, "n"),
             ("(b) Press 5 mm toward the tip end", pick(basis, y=0.0, z=15.4, n=(1, 0, 0.1)), "n"),
             ("(c) Press on the edge, −y side", pick(basis, y=-5.4, z=10.4, n=(0.6, -0.8, 0)), "n"),
             ("(d) Press on the +y flank", pick(basis, y=7.0, z=10.4, x=1.0, n=(0, 1, 0)), "n"),
             ("(e) Shear along the finger", apex, (0, 0, 1)),
             ("(f) Shear across the finger", apex, (0, 1, 0))]
    out = pad_outline_yz()
    chips = chips_U()
    mc = M.mc
    fig, axes = plt.subplots(2, 3, figsize=(10.0, 8.6))
    scale = 0.075                       # mm of arrow per uT
    nums = {}
    for ax, (title, i, how) in zip(axes.ravel(), cases):
        f = -N[i] if how == "n" else np.array(how, float)
        d, _ = signature(M, basis, i, f)
        ax.fill(out[:, 0], out[:, 1], color="#f6f1e8", ec="0.6", lw=0.8)
        if lay == "four":                # the magnets' places, faint, so that the chips' changes read against them
            for (cy, cz), sg in zip(mc[:, 1:], Lg["checker"]):
                ax.add_patch(Circle((cy, cz), 1.5, fc="none", ec=RED if sg > 0 else BLUE, lw=1.0, ls=":", zorder=1))
        for (cx, cy, cz), b in zip(chips, d):
            ax.add_patch(Rectangle((cy - 1.5, cz - 1.5), 3.0, 3.0, fc="white", ec="0.45", lw=0.8, zorder=1))
            r = math.sqrt(abs(b[0])) * 0.36
            ax.add_patch(Circle((cy, cz), r, fc=RED if b[0] > 0 else BLUE, ec="none", alpha=0.85, zorder=2))
            ax.annotate("", xy=(cy + scale * b[1], cz + scale * b[2]), xytext=(cy, cz),
                        arrowprops=dict(arrowstyle="-|>", lw=1.6, color="black", shrinkA=0, shrinkB=0), zorder=3)
        p = P[i]
        ax.plot(p[1], p[2], "*", ms=15, mfc="#f2c200", mec="black", zorder=4)
        if how != "n":
            ax.annotate("", xy=(p[1] + 3.2 * f[1], p[2] + 3.2 * f[2]), xytext=(p[1], p[2]),
                        arrowprops=dict(arrowstyle="-|>", lw=2.2, color="#f2a900"), zorder=4)
        tot = float(np.linalg.norm(d))
        ax.text(0.5, 0.015, f"|Δb| = {tot:.0f} µT per newton", transform=ax.transAxes, ha="center", va="bottom", fontsize=12)
        nums[title] = dict(point=np.round(p, 2).tolist(), norm_uT=round(tot, 1),
                           per_chip_uT=np.round(np.linalg.norm(d, axis=1), 1).tolist(),
                           per_chip_x_uT=np.round(d[:, 0], 1).tolist())
        ax.set_title(title, fontsize=12.5)
        ax.set_xlim(-11, 11)
        ax.set_ylim(-3.5, 23.5)
        ax.set_aspect("equal")
        ax.set_xticks([-10, 0, 10])
        ax.set_yticks([0, 10, 20])
    for ax in axes[1]:
        ax.set_xlabel("y (mm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("z (mm)")
    k = axes[0, 0]                                             # scale key
    k.annotate("", xy=(-10.2 + 50 * scale, 21.6), xytext=(-10.2, 21.6),
               arrowprops=dict(arrowstyle="-|>", lw=1.6, color="black", shrinkA=0, shrinkB=0))
    k.text(-10.2, 20.2, "50 µT", fontsize=11, va="top")
    k.add_patch(Circle((8.0, 20.6), math.sqrt(50) * 0.36, fc="0.6", ec="none", alpha=0.85))
    k.text(8.0, 17.6, "50 µT", fontsize=11, ha="center", va="top")
    fig.tight_layout()
    save(fig, Lg["name"].format("press_patterns"))
    NUM["patterns_N52_checkerboard" if lay == "five" else "patterns_N52_four"] = nums
    # the same at the face centre for the as-built magnets (tip 2; v3 ID 1) and for the design's N52 in both patterns
    if lay == "five":
        models = (("as_built_tip2", W.Model(W.Calib.load(CB2T))), ("all_N_N52", design_model(ALL_N)), ("checker_N52", M))
    else:
        models = (("as_built_v3id1", W.Model(W.Calib.load(V3_CAL))),
                  ("all_N_N52_four", design_model(ALL_N4, case=Lg["case"])), ("checker_N52_four", M))
    for name, model in models:
        d, S = signature(model, basis, apex, -N[apex])
        NUM.setdefault("face_centre_press_uT_per_N", {})[name] = round(float(np.linalg.norm(d)), 1)
        NUM.setdefault("face_centre_shear_uT_per_N", {})[name] = [round(float(np.linalg.norm(S[:, 2])), 1),
                                                                  round(float(np.linalg.norm(S[:, 1])), 1)]


# ------------------------------------------------------------------------------------------------ figure 6: likelihood
def evidence(est, d):
    """Log marginal likelihood (score) of a contact at every candidate point for reading change d, and the force there;
    points whose force does not press inside the friction cone or exceeds f_max are nan."""
    dw = est.Linv @ d
    T = np.einsum("nik,i->nk", est.Sw, dw)
    f = np.einsum("nkl,nl->nk", est.Ginv, T)
    e = np.einsum("nk,nk->n", T, f)
    fn = np.einsum("nk,nk->n", f, est.Cn)
    ft = np.linalg.norm(f - fn[:, None] * est.Cn, axis=1)
    score = e - est.logdet
    bad = (fn >= 0) | (ft > est.mu_max * -fn) | (np.einsum("nk,nk->n", f, f) > est.f_max ** 2)
    score[bad] = np.nan
    return score, f


def fig_likelihood(lay="five"):
    Lg = LAYOUTS[lay]
    plt = plt_setup()
    basis = W.load_basis(Lg["basis"])
    cal = W.Calib.load(CB2T if lay == "five" else V3_CAL)
    model = W.Model(cal)
    est = W.SurfaceEstimator(model, basis, floor_uT=2.0)
    est.set_noise_scale(1 / math.sqrt(7))
    P, N = basis["pts"], basis["normals"]
    apex = int(np.argmax(P[:, 0]))
    rng = np.random.default_rng(3)
    k0 = int(np.argmin(np.linalg.norm(est.C - P[apex], axis=1)))
    S0 = est.S[k0]
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 5.2), sharey=True)
    face = est.Cn[:, 0] > 0.25
    nums = {}
    for ax, F in zip(axes, (5.0, 2.0, 0.5)):
        d = S0 @ (-est.Cn[k0] * F) + rng.normal(0, 1, 15) * est.sig_now
        score, f = evidence(est, d)
        best = int(np.nanargmax(score))
        rel = 0.5 * (score - score[best])                     # log marginal likelihood relative to the best point (natural log)
        sc = ax.scatter(est.C[face, 1], est.C[face, 2], c=np.clip(rel[face], -12, 0), cmap="viridis", s=4, vmin=-12, vmax=0,
                        rasterized=True)
        ax.scatter(est.C[face & np.isnan(rel), 1], est.C[face & np.isnan(rel), 2], c="0.85", s=4, rasterized=True)
        ax.plot(P[apex, 1], P[apex, 2], "*", ms=16, mfc="#f2c200", mec="black", zorder=4)
        ax.plot(est.C[best, 1], est.C[best, 2], "x", ms=13, mew=2.6, color="#e0301e", zorder=5)
        err = float(np.linalg.norm(est.C[best] - P[apex]))
        within = int(np.sum(rel[face] > -2.0))
        ax.set_title(f"({'abc'[[5.0, 2.0, 0.5].index(F)]}) {F:g} N: estimate {err:.1f} mm off\n{within} points within e² of the best",
                     fontsize=12.5)
        nums[f"{F:g}N"] = dict(error_mm=round(err, 2), points_within_2=within, f_n_est=round(float(-f[best] @ est.Cn[best]), 3))
        ax.set_aspect("equal")
        ax.set_xlim(-8.5, 8.5)
        ax.set_ylim(-0.5, 22.5)
        ax.set_xlabel("y (mm)")
    axes[0].set_ylabel("z (mm)")
    cb = fig.colorbar(sc, ax=axes, orientation="horizontal", fraction=0.05, pad=0.13)
    cb.set_label("Log marginal likelihood of a contact at each surface point, relative to the best")
    save(fig, Lg["name"].format("likelihood_maps"))
    # the same press drawn 100 times with other noise (its own generator, so that the figure's draws stay as they were)
    rng2 = np.random.default_rng(4)
    for F in (5.0, 3.0, 2.0, 1.0, 0.5):
        errs = []
        for _ in range(100):
            d = S0 @ (-est.Cn[k0] * F) + rng2.normal(0, 1, 15) * est.sig_now
            score, _ = evidence(est, d)
            errs.append(float(np.linalg.norm(est.C[int(np.nanargmax(score))] - P[apex])))
        errs = np.array(errs)
        nums.setdefault("draws100", {})[f"{F:g}N"] = dict(median_mm=round(float(np.median(errs)), 2),
                                                          p90_mm=round(float(np.percentile(errs, 90)), 2),
                                                          within_1mm=round(float(np.mean(errs <= 1.0)), 2))
    NUM["likelihood_tip2" if lay == "five" else "likelihood_v3id1"] = nums


# ------------------------------------------------------------------------------------------------ figure 7: UR5 check
def fig_ur5(lay="five"):
    if lay == "four":
        return fig_ur5_four()
    plt = plt_setup()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 4.4), gridspec_kw=dict(width_ratios=(1.5, 1.0)))
    cols = ["#555555", "#2e5fa8", "#7fa6d9", "#b06010"]
    mk = ["o", "s", "D", "^"]
    nums = {}
    for (sess, name), c, m in zip(SESSIONS, cols, mk):
        s = json.loads((UR5 / sess / "summary.json").read_text())
        lv = s["estimator_by_level"]
        L = sorted(float(k) for k in lv)
        a1.plot(L, [lv[f"{l:g}"]["loc_median_mm"] for l in L], "-" + m, color=c, label=name, ms=6)
        nums[sess] = dict(loc_by_level={f"{l:g}": round(lv[f"{l:g}"]["loc_median_mm"], 2) for l in L},
                          force_scale={r: round(v["force_scale"], 2) for r, v in s["by_region"].items()})
    a1.axhline(2.0, color="0.6", lw=1, ls="--")
    a1.set_xscale("log")
    a1.set_xticks([0.25, 0.5, 1, 2, 3, 5])
    a1.set_xticklabels(["0.25", "0.5", "1", "2", "3", "5"])
    a1.set_xlabel("Force level of the hold (N)")
    a1.set_ylabel("Location error, median (mm)")
    a1.set_title("(a) Contact point of the untrained model", fontsize=13)
    a1.legend(fontsize=11, frameon=False)
    regs = ["face", "edge", "flank"]
    w = 0.2
    for j, ((sess, name), c) in enumerate(zip(SESSIONS, cols)):
        v = [nums[sess]["force_scale"][r] for r in regs]
        a2.bar(np.arange(3) + (j - 1.5) * w, v, w, color=c)
    a2.axhline(1.0, color="black", lw=1)
    a2.set_xticks(range(3))
    a2.set_xticklabels(["Face", "Edges", "Flanks"])
    a2.set_ylim(0, 1.45)
    a2.set_ylabel("Model force / HEX-E force")
    a2.set_title("(b) Normal force of the untrained model", fontsize=13)
    fig.tight_layout()
    save(fig, "20261026-ur5_untrained_model.png")
    NUM["ur5"] = nums


def fig_ur5_four():
    """The untrained model of v3 ID 1 (four N52) and v2 ID 1 (five N52) on the same 818 press holds of the low sphere's
    grid stage, as the coordinator's fe_calibration.json scores it (the FE and the rest fit alone, compliance scale 1)."""
    plt = plt_setup()
    G = json.loads(FE_CAL.read_text())
    bins = ["0-1", "1-2", "2-3.5", "3.5-8"]
    regs = [("face", "Face"), ("edge", "Edges"), ("flank", "Flanks")]
    tips = [("four_v3id1", "Four N52 magnets (v3 ID 1)", FOUR_C), ("five_v2id1", "Five N52 magnets (v2 ID 1)", FIVE_C)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 4.6), gridspec_kw=dict(width_ratios=(1.35, 1.0)))
    w = 0.38
    nums = {}
    for j, (key, lab, col) in enumerate(tips):
        u = G[key]["untrained"]
        x = np.arange(len(bins)) + (j - 0.5) * w
        v = [u[b]["loc_mm"] for b in bins]
        a1.bar(x, v, w, color=col, label=lab)
        for xi, vi in zip(x, v):
            a1.text(xi, vi + 0.12, f"{vi:.1f}", ha="center", va="bottom", fontsize=11)
        x2 = np.arange(len(regs)) + (j - 0.5) * w
        v2 = [100 * u[r]["fn_rel"] for r, _ in regs]
        a2.bar(x2, v2, w, color=col)
        for xi, vi in zip(x2, v2):
            a2.text(xi, vi + 0.6, f"{vi:.0f}", ha="center", va="bottom", fontsize=11)
        nums[key] = {k: dict(n=v_["n"], loc_mm=round(v_["loc_mm"], 3), fn_rel=round(v_["fn_rel"], 4),
                             fn_err_N=round(v_["fn_err_N"], 3)) for k, v_ in u.items()}
    a1.axhline(2.0, color="0.6", lw=1, ls="--")
    a1.set_xticks(range(len(bins)))
    a1.set_xticklabels(["0–1", "1–2", "2–3.5", "3.5–8"])
    a1.set_xlabel("Normal force of the hold (N)")
    a1.set_ylabel("Location error, median (mm)")
    a1.set_ylim(0, 9.4)
    a1.set_title("(a) Contact point of the untrained model", fontsize=13)
    a1.legend(fontsize=11, frameon=False, loc="upper right")
    a2.set_xticks(range(len(regs)))
    a2.set_xticklabels([t for _, t in regs])
    a2.set_ylabel("Normal force error, median (%)")
    a2.set_ylim(0, 38)
    a2.set_title("(b) Normal force of the untrained model", fontsize=13)
    fig.tight_layout()
    save(fig, LAYOUTS["four"]["name"].format("ur5_untrained_model"))
    NUM["ur5_four"] = nums


V3_CHECK = DATA / "20261029-v3_tips_model_check.json"
GAIN_GROUPS = (("face centre, other plans", "Face\ncentre"), ("grid, face", "Face"), ("top-edge pass, tip", "Tip\nend"),
               ("grid, edge", "Edges"), ("grid, flank", "Flanks"))


def fig_gain_regions():
    """The board's signal per newton by region of the pad on the low sphere: measured on v3 ID 1 and v3 ID 2 (four N52)
    and v2 ID 1 (five N52), and predicted for each hold from its registered point and the HEX-E's force by the model,
    untrained (rest fit, compliance scale 1) and fitted to the presses (scripts/fingertip_physical_model_v3_check.py)."""
    plt = plt_setup()
    G = json.loads(V3_CHECK.read_text())
    panels = [("(a) Four N52 magnets, pad 05R4", [("v3id1", "v3 ID 1", FOUR_C), ("v3id2", "v3 ID 2", "#e0a35e")]),
              ("(b) Five N52 magnets, pad B05R", [("v2id1", "v2 ID 1", FIVE_C)])]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.8), gridspec_kw=dict(width_ratios=(1.5, 1.0)), sharey=True)
    top = max(max(r[k] for k in ("measured", "untrained", "fitted") if r.get(k) is not None)
              for t in G["gain"].values() for r in t.values() if r.get("measured") is not None)
    for ax, (title, tips) in zip(axes, panels):
        groups = [(g, lab) for g, lab in GAIN_GROUPS
                  if any((G["gain"][k].get(g) or {}).get("measured") is not None for k, _, _ in tips)]
        w = 0.8 / len(tips)
        for j, (key, lab, col) in enumerate(tips):
            for i, (g, _) in enumerate(groups):
                r = G["gain"][key].get(g)
                if not r or r["measured"] is None:
                    continue
                x = i + (j - (len(tips) - 1) / 2) * w
                ax.bar(x, r["measured"], w * 0.9, color=col, label=f"Measured, {lab}" if i == 0 else None)
                ax.plot([x - w * 0.45, x + w * 0.45], [r["fitted"]] * 2, color="black", lw=2.4,
                        label="Model fitted to the presses" if (i == 0 and j == len(tips) - 1) else None)
                ax.plot([x - w * 0.45, x + w * 0.45], [r["untrained"]] * 2, color="0.25", lw=1.6, ls=(0, (2.5, 1.5)),
                        label="Untrained model" if (i == 0 and j == len(tips) - 1) else None)
                ax.text(x, 1.5, f"{r['measured']:.0f}", ha="center", va="bottom", fontsize=11, color="white")
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([lab for _, lab in groups])
        ax.set_xlim(-0.6, len(groups) - 0.4)
        ax.set_title(title, fontsize=13)
        h, lab = ax.get_legend_handles_labels()
        order = sorted(range(len(lab)), key=lambda i: (not lab[i].startswith("Measured"), i))
        ax.legend([h[i] for i in order], [lab[i] for i in order], fontsize=11, frameon=False, loc="upper left")
    axes[0].set_ylabel("Signal per newton, median (µT/N)")
    axes[0].set_ylim(0, 1.32 * top)
    fig.tight_layout()
    save(fig, "20261029-signal_by_region.png")
    TT = json.loads((FE_CAL.parent / "20261029-tip_vs_tip_sphere_v3id2_v3id1.json").read_text())
    NUM["v3_check"] = dict(
        shared_actions=G["shared_actions"], shared_grid_actions=G["shared_grid_actions"],
        gain_1_2_tip_vs_tip={k: round(TT["tips"][t]["gain_uT_per_N"]["1-2"], 2) for k, t in (("v3id1", "v3 ID 1"),
                                                                                           ("v3id2", "v3 ID 2"))},
        depth={k: {g: dict(slope_um_per_N=round(v["slope_um_per_N"], 1), fe_give_um_per_N=round(v["fe_give_um_per_N"], 1))
                   for g, v in r["depth"].items()} for k, r in G["fit"].items()},
        rest={k: {n: dict(k=round(v["k"], 3), dx=round(v["dx"], 3), rms_mT=round(v["rms_mT"], 3),
                          rms_next_mT=round(v["rms_next_mT"], 3)) for n, v in r.items() if n != "dx_bounds"}
              for k, r in G["rest"].items()},
        fit={k: dict(k=round(r["fit"]["k"], 3), c=round(r["fit"]["c"], 3), E_fit_MPa=round(r["fit"]["E_fit_MPa"], 1),
                     board_dx=round(r["fit"]["dx_mm"] + r["rest_used"]["dx"], 3), rms_uT=round(r["fit"]["rms_uT"], 2),
                     holds=r["holds"],
                     untrained={b: dict(n=v["n"], loc_mm=round(v["loc_mm"], 3), fn_rel=round(v["fn_rel"], 4))
                                for b, v in r["untrained"].items() if v["loc_mm"] is not None},
                     cv={b: dict(n=v["n"], loc_mm=round(v["loc_mm"], 3), fn_rel=round(v["fn_rel"], 4))
                         for b, v in r["cv"]["table"].items() if v["loc_mm"] is not None})
             for k, r in G["fit"].items()},
        gain={k: {g: {n: (round(v, 4) if isinstance(v, float) else v) for n, v in r.items()} for g, r in t.items()}
              for k, t in G["gain"].items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--layout", default="both", choices=("four", "five", "both"))
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    figs = dict(anatomy=fig_anatomy, chain=fig_chain, fe=fig_fe, field=fig_field, patterns=fig_patterns,
                likelihood=fig_likelihood, ur5=fig_ur5)
    if NUMBERS.exists():
        NUM.update(json.loads(NUMBERS.read_text()))
    if a.only is not None and "gain" in a.only:
        fig_gain_regions()
    for lay in (("four", "five") if a.layout == "both" else (a.layout,)):
        for k, f in figs.items():
            if a.only is None or k in a.only:
                f(lay)
    if a.only is None:
        fig_gain_regions()
    NUMBERS.write_text(json.dumps(NUM, indent=1) + "\n")
    print("wrote", NUMBERS.relative_to(ROOT))


if __name__ == "__main__":
    main()
