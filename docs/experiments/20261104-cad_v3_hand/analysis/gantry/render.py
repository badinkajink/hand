#!/usr/bin/env python3
"""Figures of the gantry analysis in the palm frame P: a top view with the travel boxes and a side view with the
heights. Mode "brep" draws each part's bounding box from parts.json (no meshes); mode "mesh" projects the part meshes
(run it through resguard). The thumb gantry is drawn moved +2.5 mm in y_A onto the base plate's holes.

    .venv/bin/python logs/20261104-cad_v3/gantry/render.py brep
    ~/.claude/bin/resguard.sh run --mem 3G --cpu 100 -- .venv/bin/python logs/20261104-cad_v3/gantry/render.py mesh
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import classify  # noqa: E402

REPO = HERE.parents[2]
G = json.loads((HERE / "gantry.json").read_text())
R = np.asarray(G["R_P_from_A"], float)
t = np.asarray(G["t_P_from_A_mm"], float)
MODE = sys.argv[1] if len(sys.argv) > 1 else "brep"
COL = {"thumb": "#c0392b", "index": "#2471a3", "middle": "#1e8449"}
NOM = {"thumb": (-50.0, 0.0, 60.0, 110.0), "index": (50.0, 55.0, 60.0, 60.0), "middle": (50.0, -55.0, 60.0, 60.0)}
plt.rcParams.update({"font.size": 13, "font.family": "serif"})

roles, finger_of = classify.classify()
owner = {}
for g, nm in (("g1", "index"), ("g2", "middle"), ("gt", "thumb")):
    for k, v in roles[g].items():
        for n in v:
            owner[n] = (nm, "fixed" if k in ("frame", "x_rail", "x_screw", "x_motor", "x_coupler", "x_shaft",
                                             "x_bearing", "x_mount") else "beam" if k != "y_block" else "finger")
    for p in classify.PARTS.values():
        if classify.sub(p) == finger_of[g]:
            owner[p["name"]] = (nm, "finger")
SHIFT = np.asarray(G["thumb_gantry_correction"]["shift_A_mm"], float)
STAY = set(G["thumb_gantry_correction"]["parts_that_stay"])


def shift_of(n):
    o = owner.get(n)
    return SHIFT if (o and o[0] == "thumb" and n not in STAY) else np.zeros(3)


def to_P(pts_A):
    return pts_A @ R.T + t


def part_style(n):
    o = owner.get(n)
    if o is None:
        return dict(fc="#d9d9d9", ec="#7f7f7f", alpha=0.25, z=0)
    nm, kind = o
    if kind == "fixed":
        return dict(fc="#bfbfbf", ec="#595959", alpha=0.35, z=1)
    if kind == "beam":
        return dict(fc=COL[nm], ec=COL[nm], alpha=0.12, z=2)
    return dict(fc=COL[nm], ec=COL[nm], alpha=0.30, z=3)


def draw_parts(ax, i, j, names, sign_j=1.0):
    for n in names:
        p = classify.PARTS[n]
        st = part_style(n)
        sh = shift_of(n)
        if MODE == "mesh":
            import trimesh
            m = trimesh.load(REPO / p["stl"], force="mesh")
            V = to_P(np.asarray(m.vertices) + sh)
            tri = V[m.faces][:, :, [i, j]]
            ax.add_collection(PolyCollection(tri, facecolors=[st["fc"]], edgecolors="none", alpha=st["alpha"] * 0.6,
                                             zorder=st["z"]))
        else:
            lo, hi = np.asarray(p["bbox"][:3]) + sh, np.asarray(p["bbox"][3:]) + sh
            c = to_P(np.array([lo, hi]))
            a0, a1 = sorted(c[:, i])
            b0, b1 = sorted(c[:, j])
            ax.add_patch(Rectangle((a0, b0), a1 - a0, b1 - b0, fc=st["fc"], ec=st["ec"], alpha=st["alpha"], lw=0.6,
                                   zorder=st["z"]))


def top():
    fig, ax = plt.subplots(figsize=(15, 13.5))
    names = [n for n in classify.PARTS if n not in ("001", "004")]     # robot mount and adaptor sit above, drawn as outlines
    draw_parts(ax, 0, 1, names)
    for n, lab in (("001", "robot mount"), ("004", "UR5 adaptor"), ("002", "base plate")):
        p = classify.PARTS[n]
        c = to_P(np.array([p["bbox"][:3], p["bbox"][3:]]))
        a0, a1 = sorted(c[:, 0])
        b0, b1 = sorted(c[:, 1])
        ax.add_patch(Rectangle((a0, b0), a1 - a0, b1 - b0, fc="none", ec="k", lw=1.0, ls=(0, (6, 3)), zorder=6))
        if n == "004":
            ax.text(a0 + 2, b0 + 2, lab, ha="left", va="bottom", fontsize=13, zorder=7)
        else:
            ax.text(a1 - 2, b1 - 2, lab, ha="right", va="top", fontsize=13, zorder=7)
    for nm in ("thumb", "index", "middle"):
        f = G["fingers"][nm]
        P = f["P_with_thumb_on_base_plate_holes"]
        bc = np.asarray(P["box_centre_P_mm"])
        sx, sy = P["box_size_mm"]["commanded_xP_yP"]
        gx, gy = P["box_size_mm"]["stop_to_stop_xP_yP"]
        gc = np.asarray(P["stroke_centre_P_mm"])
        ax.add_patch(Rectangle((bc[0] - sx / 2, bc[1] - sy / 2), sx, sy, fc="none", ec=COL[nm], lw=2.6, zorder=8))
        ax.add_patch(Rectangle((gc[0] - gx / 2, gc[1] - gy / 2), gx, gy, fc="none", ec=COL[nm], lw=1.4,
                               ls=(0, (5, 3)), zorder=8))
        nx, ny, nw, nh = NOM[nm]
        ax.add_patch(Rectangle((nx - nw / 2, ny - nh / 2), nw, nh, fc="none", ec="k", lw=1.6, ls=(0, (1, 2)), zorder=9))
        ax.plot(*bc[:2], marker="+", ms=16, mew=2.5, color=COL[nm], zorder=10)
        ax.plot(nx, ny, marker="x", ms=10, mew=2, color="k", zorder=10)
        yc = np.asarray(P["cad_pose_yaw_centre_P_mm"])
        ax.plot(*yc[:2], marker="o", ms=9, color=COL[nm], mec="k", zorder=10)
        # home corner: the motor ends of both stages
        tr = f["travel_A_mm"]
        hxA = tr["x_stage"]["motor_end"]
        hyA = tr["y_stage"]["motor_end"]
        hx_P = bc[0] + (sx / 2) * (-1 if hyA == "+y_A" else 1)      # x_P = -y_A
        hy_P = bc[1] + (sy / 2) * (1 if hxA == "+x_A" else -1)      # y_P = x_A
        ax.plot(hx_P, hy_P, marker="s", ms=12, color=COL[nm], mec="k", zorder=11)
        ax.annotate("home", (hx_P, hy_P), xytext=(8 if hx_P > bc[0] else -8, 8 if hy_P > bc[1] else -16),
                    textcoords="offset points", ha="left" if hx_P > bc[0] else "right", fontsize=13, zorder=12)
        jx, jy = tr["x_stage"]["firmware_joint"], tr["y_stage"]["firmware_joint"]
        ax.text(bc[0], bc[1] + 4, f"{nm}\ncentre ({bc[0]:.1f}, {bc[1]:.1f})\n{jy} along x_P, {jx} along y_P",
                ha="center", va="bottom", fontsize=13, color=COL[nm], zorder=12,
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))
    ax.annotate("", xy=(20, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=2), zorder=12)
    ax.annotate("", xy=(0, 20), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=2), zorder=12)
    ax.text(21, -1, "x_P", fontsize=13, va="top")
    ax.text(1, 21, "y_P", fontsize=13)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color="0.3", lw=2.6, label="commanded box from home (60 or 110 mm), yaw centre"),
         Line2D([], [], color="0.3", lw=1.4, ls=(0, (5, 3)), label="stop-to-stop travel of the yaw centre"),
         Line2D([], [], color="k", lw=1.6, ls=(0, (1, 2)), label="simulator / driver nominal box"),
         Line2D([], [], color="0.3", marker="o", ls="none", ms=9, label="yaw centre in the CAD pose"),
         Line2D([], [], color="0.3", marker="+", ls="none", ms=14, mew=2.5, label="box centre from the CAD"),
         Line2D([], [], color="k", marker="x", ls="none", ms=10, mew=2, label="nominal box centre"),
         Line2D([], [], color="0.3", marker="s", ls="none", ms=11, label="home corner (stepper 0 mm)")]
    ax.legend(handles=h, loc="lower left", fontsize=12, framealpha=0.9)
    ax.set_aspect("equal")
    ax.set_xlim(-200, 175)
    ax.set_ylim(-175, 175)
    ax.set_xlabel("x_P (mm)")
    ax.set_ylabel("y_P (mm)")
    ax.grid(True, lw=0.3)
    ttl = "Gantry travel boxes in the palm frame P, top view (z_P toward the viewer)"
    ax.set_title(ttl + ("\nparts drawn as bounding boxes from the STEP B-rep" if MODE == "brep" else "\nparts projected from the CAD meshes")
                 + "; thumb gantry moved +2.5 mm in y_A onto the base plate's holes", fontsize=14)
    fig.tight_layout()
    out = HERE / f"20261104-gantry_top_P_{MODE}.png"
    fig.savefig(out, dpi=90)
    print("wrote", out)


def side():
    fig, ax = plt.subplots(figsize=(17, 10))
    names = [n for n in classify.PARTS]
    draw_parts(ax, 0, 2, names)
    zt = -G["yaw_plane_z_A_mm"]
    H = G["heights"]
    xs = ax.get_xlim()
    lv = sorted(((zA + zt, zA, lab) for zA, lab in H["levels_A_mm"].values()), reverse=True)
    ys, last = [], None
    for zP, _, _ in lv:                       # label rows at least 9.5 mm apart, top down
        y = zP if last is None else min(zP, last - 9.5)
        ys.append(y)
        last = y
    for (zP, zA, lab), y in zip(lv, ys):
        ax.axhline(zP, color="k", lw=0.8, ls=(0, (4, 3)), zorder=5)
        ax.annotate(f"{lab}: z_P {zP:+.1f} (z_A {zA:+.1f})", xy=(165, zP), xytext=(188, y), textcoords="data",
                    va="center", ha="left", fontsize=13, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", lw=0.7, color="0.3", shrinkA=0, shrinkB=0), zorder=6)
    ax.axhspan(23.5, 26.5, color="#f39c12", alpha=0.25, zorder=0)
    ax.text(-198, 15, "palm_plate of the cad_v2 scene (z_P 23.5 to 26.5)", va="center", fontsize=12.5, color="#a04000")
    ax.set_xlim(-200, 165)
    ax.set_ylim(-95, 112)
    ax.set_aspect("equal")
    ax.set_xlabel("x_P (mm)")
    ax.set_ylabel("z_P (mm), 0 = yaw-centre plane")
    ax.grid(True, lw=0.3)
    ax.set_title("Heights of the hand stack in P, side view along y_P (thumb left, index and middle right)"
                 + ("\nparts drawn as bounding boxes from the STEP B-rep" if MODE == "brep" else "\nparts projected from the CAD meshes"),
                 fontsize=14)
    fig.subplots_adjust(left=0.055, right=0.60, top=0.91, bottom=0.07)
    out = HERE / f"20261104-gantry_side_P_{MODE}.png"
    fig.savefig(out, dpi=90)
    print("wrote", out)


if __name__ == "__main__":
    top()
    side()
