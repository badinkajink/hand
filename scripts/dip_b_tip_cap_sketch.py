#!/usr/bin/env python3
"""Sketch of a T-shaped cap over the tip end of concept b v3 (2026-10-20 evening), drawn from dip_link_concept_b's
numbers: the palmar view and the view from the tip end, with the pad, the cover with its walls and rails, the board's
screws, chips and magnets, and the cap (end plate, side walls along the pad's distal flanks, ears under the two distal
board screws, a pin into the pad's tip end).

Owner, 2026-10-20 19:48: "the finger is getting popped out a lot ... a separate fingertip that goes ontop of and fastens to
both the dip link (top of the magnetometer plate) and the top of the fingertip. im imagining a kind of T-shaped plate ...
you could trick out this T-plate and extrude it down to mold around the top of the fingertip (and around the screws too)
for additional side-to-side retaining". Output:
docs/experiments/20261018-dip_link_board_mount/v3/b/media/20261020-dip_link_v3b_tip_cap_sketch.png

    .venv/bin/python scripts/dip_b_tip_cap_sketch.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("DIPB_PARAMS", str(ROOT / "docs/experiments/20261018-dip_link_board_mount/params/20261020-v3_concept_b.json"))
sys.path.insert(0, str(ROOT / "scripts"))
import dip_link_concept_b as B  # noqa: E402

OUT = ROOT / "docs/experiments/20261018-dip_link_board_mount/v3/b/media/20261020-dip_link_v3b_tip_cap_sketch.png"
ZC, ZE, ZM = B.Z_CONTACT, B.TIP["z_end"], B.TIP["z_mount"]          # -87.91, -98.46, -77.05
X_FLANK = -2.05                     # palmar end of the pad's straight flank (x_t 0 of the FE frame)
X_TONGUE = (B.X_SEAT, B.X_TONGUE_BACK)                               # 4.40, 8.00
CAP = dict(t_end=1.2, t_wall=1.2, z0=-89.80, z_ear=-93.40, ear_t=3.0, pin=(1.40, 0.0, 1.0, 2.7))
COL = dict(pad="#dcc28f", cover="#3d6fb8", link="#c8c8c2", cap="#e07b24", board="#2f7d4f", mag="#8a8f99", screw="#c9a54a")
ARROW = dict(arrowstyle="-", lw=0.8)


def half_w(z):
    """The pad's palmar-view half-width at z_F: half-ellipses of 22 mm toward the mount face and 11 mm toward the tip."""
    dz = np.asarray(z, float) - ZC
    a = np.where(dz > 0, 22.0, 11.0)
    return 7.4 * np.sqrt(np.clip(1 - (dz / a) ** 2, 0, None))


def band(z, inner, outer):
    """Polygon between two y(z) curves (arrays over z)."""
    return np.c_[np.r_[inner, outer[::-1]], np.r_[z, z[::-1]]]


def palmar(ax):
    """Palmar view: y across, z along the finger with the tip end at the bottom."""
    ax.add_patch(Rectangle((-B.Y_RIM, ZE), 2 * B.Y_RIM, ZM - ZE, fc=COL["cover"], ec="k", lw=0.8, alpha=0.30))
    zw = np.linspace(ZC, ZM, 120)
    for s in (-1, 1):                    # the cover's walls as printed: the pad's flank to the rim, mount face to the apex
        ax.add_patch(Polygon(band(zw, s * half_w(zw), s * B.Y_RIM * np.ones_like(zw)), closed=True, fc=COL["cover"],
                             ec="k", lw=0.6, alpha=0.85))
        ax.add_patch(Rectangle((s * 7.4 if s > 0 else -8.9, ZC - 1.9), 1.5, 3.8, fc="white", ec="none"))
    z = np.linspace(ZE, ZM, 300)
    w = half_w(z)
    ax.fill_betweenx(z, -w, w, color=COL["pad"], alpha=0.92, lw=0)
    ax.plot(np.r_[w, -w[::-1], w[:1]], np.r_[z, z[::-1], z[:1]], color="0.25", lw=1.0)
    for n, (y, zz) in {"U1": (0.0, ZC), "U2": (0.0, ZC - 7.0), "U4": (0.0, ZC + 7.0), "U3": (7.0, ZC),
                       "U5": (-7.0, ZC)}.items():
        ax.add_patch(Rectangle((y - 1.5, zz - 1.5), 3.0, 3.0, fill=False, ec="0.3", lw=0.9, ls=":"))
        ax.text(y + (1.7 if y >= 0 else -1.7), zz + 1.7, n, ha="left" if y >= 0 else "right", va="bottom", fontsize=12,
                color="0.25")
    for sy in (-1, 1):                                                               # rails under the pad
        ax.add_patch(Rectangle((sy * B.RAIL_Y - B.RAIL_TOP / 2, B.RAIL_Z[0]), B.RAIL_TOP, B.RAIL_Z[1] - B.RAIL_Z[0],
                               fill=False, ec=COL["cover"], lw=1.3, ls="--"))
    for y, zz in [(sy * 2.5, ZC + 0.5 + sz * 4.9) for sy in (-1, 1) for sz in (-1, 1)] + [(0.0, ZC + 0.5)]:
        ax.add_patch(Circle((y, zz), 1.5, fc=COL["mag"], ec="k", lw=0.5, alpha=0.65))
    zc_ = np.linspace(ZE, CAP["z0"], 120)                    # the cap: side walls, ears to the rim, end plate
    ze = np.linspace(ZE, CAP["z_ear"], 60)
    for s in (-1, 1):
        ax.add_patch(Polygon(band(zc_, s * half_w(zc_), s * (half_w(zc_) + CAP["t_wall"])), closed=True, fc=COL["cap"],
                             ec="k", lw=0.7))
        ax.add_patch(Polygon(band(ze, s * half_w(ze), s * B.Y_RIM * np.ones_like(ze)), closed=True, fc=COL["cap"],
                             ec="k", lw=0.7))
    ax.add_patch(Rectangle((-B.Y_RIM, ZE - CAP["t_end"]), 2 * B.Y_RIM, CAP["t_end"], fc=COL["cap"], ec="k", lw=0.7))
    for y, zz in B.screw_yz():                                                       # the board screws' heads
        ax.add_patch(Circle((y, zz), B.HEAD_D / 2, fc=COL["screw"], ec="k", lw=0.7, zorder=5))
    xp, yp, rp, lp = CAP["pin"]
    ax.add_patch(Rectangle((yp - rp, ZE), 2 * rp, lp, fill=False, ec="k", lw=1.3, ls="--", zorder=6))
    for txt, xy, at in (("Pin, 2 mm, 2.7 mm into\nthe pad's tip end", (yp + rp, ZE + 2.0), (11.6, ZE - 3.4)),
                        ("Ear under the distal\nboard screw (M2 x 8)", (8.0 + 1.9, -95.81), (11.6, -93.4)),
                        ("Rails end at\nz_F -93.55", (B.RAIL_Y + 1.0, B.RAIL_Z[0]), (11.6, -89.0)),
                        ("Cap side wall 1.2 mm, from\n0.2 mm past U5's window", (-half_w(-92.0) - 0.6, -92.0),
                         (-29.5, -94.5)),
                        ("Cover walls as printed\n(mount face to the apex)", (-9.2, -81.5), (-29.5, -80.5)),
                        ("End plate 1.2 mm", (-6.0, ZE - CAP["t_end"] / 2), (-29.5, ZE - 2.4))):
        ax.annotate(txt, xy, at, fontsize=13, arrowprops=ARROW, ha="left", va="center")
    ax.set_aspect("equal")
    ax.set_xlim(-30, 31)
    ax.set_ylim(ZE - 5.0, ZM + 1.0)
    ax.set_xlabel("Across the finger, y_F (mm)")
    ax.set_ylabel("Along the finger, z_F (mm)")
    ax.set_title("Palmar view, tip end down", fontsize=15)


def end_face():
    """The pad's flat tip end (z_F -98.46): the straight flanks |y| <= 2.09 from the back face to x_F -2.05, and the
    11/22 half-ellipsoid's section below (8.5 mm semi-axis along x)."""
    k = 1 - (10.55 / 11.0) ** 2
    yy = np.linspace(-7.4 * np.sqrt(k), 7.4 * np.sqrt(k), 60)
    xd = X_FLANK - 8.5 * np.sqrt(np.clip(k - (yy / 7.4) ** 2, 0, None))
    return np.c_[np.r_[yy[0], yy, yy[-1]], np.r_[B.X_SLOT, xd, B.X_SLOT]]


def end_view(ax):
    """The tip end seen from beyond the tip: y across, x dorsal up; the T of the tongue and cover over the pad's end."""
    ax.add_patch(Rectangle((-B.Y_RIM, X_TONGUE[0]), 2 * B.Y_RIM, X_TONGUE[1] - X_TONGUE[0], fc=COL["link"], ec="k", lw=0.8))
    ax.add_patch(Rectangle((-10.0, B.X_COMP), 20.0, B.FR4, fc=COL["board"], ec="k", lw=0.6))
    ax.add_patch(Rectangle((-B.Y_RIM, B.X_SLOT), 2 * B.Y_RIM, B.X_COMP - B.X_SLOT, fc=COL["cover"], ec="k", lw=0.8))
    f = end_face()
    ax.add_patch(Polygon(f, closed=True, fc=COL["pad"], ec="k", lw=0.8))
    xe = B.X_SLOT - CAP["ear_t"]                                 # the ears' palmar face
    yw = 7.4 * np.sqrt(1 - (10.55 / 11.0) ** 2) + CAP["t_wall"]   # the side walls' outer face at the tip end
    out = [(-B.Y_RIM, X_TONGUE[1]), (B.Y_RIM, X_TONGUE[1]), (B.Y_RIM, xe), (yw, xe), (yw, X_FLANK)]
    yy = np.linspace(yw, -yw, 40)
    k = 1 - (10.55 / 11.0) ** 2
    out += list(zip(yy, X_FLANK - 0.6 - 8.5 * np.sqrt(np.clip(k - (np.clip(np.abs(yy) - 1.2, 0, None) / 7.4) ** 2, 0, None))))
    out += [(-yw, X_FLANK), (-yw, xe), (-B.Y_RIM, xe)]
    ax.add_patch(Polygon(np.array(out), closed=True, fill=False, ec=COL["cap"], lw=2.4, zorder=7))
    for s in (-1, 1):                                             # board screws along x: head on the ear, shank to x 7.9
        ax.add_patch(Rectangle((s * 8.0 - 1.9, xe - B.HEAD_H), 3.8, B.HEAD_H, fc=COL["screw"], ec="k", lw=0.6, zorder=8))
        ax.add_patch(Rectangle((s * 8.0 - 1.0, xe), 2.0, 7.9 - xe, fc=COL["screw"], ec="k", lw=0.6, alpha=0.7, zorder=8))
    xp, yp, rp, _ = CAP["pin"]
    ax.add_patch(Circle((yp, xp), rp, fc="white", ec="k", lw=1.2, zorder=9))
    ax.text(0.0, 9.0, "Tongue, 3.6 mm behind the board", ha="center", fontsize=13)
    ax.annotate("Pad's flat end\n4.4 x 7.5 mm", (0.0, -3.4), (-12.0, -9.6), fontsize=13, arrowprops=ARROW)
    ax.annotate("Pin at x_F 1.4", (rp * 0.7, xp - rp * 0.7), (2.5, -9.6), fontsize=13, arrowprops=ARROW)
    ax.annotate("Cap outline", (B.Y_RIM, 1.5), (7.0, -5.6), fontsize=13, arrowprops=ARROW, color=COL["cap"])
    ax.set_aspect("equal")
    ax.set_xlim(-13.0, 13.0)
    ax.set_ylim(-11.0, 10.5)
    ax.set_xlabel("Across the finger, y_F (mm)")
    ax.set_ylabel("Toward the back, x_F (mm)")
    ax.set_title("Tip end seen from beyond the tip", fontsize=15)


def main():
    plt.rcParams.update({"font.family": "serif", "font.size": 14})
    fig = plt.figure(figsize=(18.0, 7.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.85, 1.0], wspace=0.16)
    palmar(fig.add_subplot(gs[0, 0]))
    end_view(fig.add_subplot(gs[0, 1]))
    handles = [Rectangle((0, 0), 1, 1, fc=COL[k], ec="k", alpha=a) for k, a in
               (("pad", 0.92), ("cover", 0.85), ("link", 1.0), ("board", 1.0), ("cap", 1.0), ("mag", 0.65), ("screw", 1.0))]
    fig.legend(handles, ["85A pad B00", "Cover (PLA) and its walls", "DIP link tongue", "ReSkin 5X board",
                         "Cap (PLA), proposed", "Magnets 3 x 2 mm", "Board screws M2"], loc="lower center", ncol=4,
               fontsize=13, frameon=False, bbox_to_anchor=(0.5, 0.0), bbox_transform=fig.transFigure)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(bottom=0.24)
    fig.savefig(OUT, dpi=100, bbox_inches="tight", facecolor="white")
    print("->", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
