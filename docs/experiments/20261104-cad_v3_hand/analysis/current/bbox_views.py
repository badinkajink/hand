"""Top and side views of the hand file's part bounding boxes (frame A, mm) with the inferred P axes, the yaw centres of
the three CAD fingers and the cad_v2 palm_plate drawn where it would sit if P's origin were at (-111, 120.2, -75.9).
JSON only (no meshes). Writes bbox_views.png beside this file."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = "/home/humanoid/Programs/hand/logs/20261104-cad_v3"
P = json.load(open(f"{ROOT}/extract/parts.json"))["parts"]
Y = json.load(open(f"{ROOT}/current/yaw_centres.json"))
COL = {"hand v3 (anthro config)": "0.45", "gantry_v3 <1>": "tab:blue", "gantry_v3 <2>": "tab:green",
       "gantry_v3_thumb <1>": "tab:red", "Finger v3 <1>": "tab:orange", "Finger v3 <2>": "tab:orange",
       "Finger v3 <3>": "tab:brown"}
fig, ax = plt.subplots(1, 2, figsize=(20, 9))
for p in P:
    if p["file"] != "hand":
        continue
    b = p["bbox"]
    c = COL[p["chain"][0]]
    big = p["name"] in ("001", "002")
    for a, (i, j) in zip(ax, ((0, 1), (0, 2))):
        a.add_patch(Rectangle((b[i], b[j]), b[i + 3] - b[i], b[j + 3] - b[j], fill=False, ec=c,
                              lw=0.6 if big else 1.0, ls="--" if big else "-"))
        if p["volume"] > 9000 and not big and not p["chain"][0].startswith("Finger"):
            a.text((b[i] + b[i + 3]) / 2, (b[j] + b[j + 3]) / 2, p["name"], fontsize=8, ha="center", va="center", color=c)
o = (-111.0, 120.2, -75.9)
for a, (i, j) in zip(ax, ((0, 1), (0, 2))):
    for f, v in Y.items():
        a.plot(v["yaw_centre_A"][i], v["yaw_centre_A"][j], "k+", ms=14, mew=2)
        a.text(v["yaw_centre_A"][i] + 3, v["yaw_centre_A"][j] + 3, f.replace("Finger v3 ", "F"), fontsize=10)
    a.plot(o[i], o[j], "mo", ms=8)
    a.set_aspect("equal"); a.grid(alpha=0.3)
ax[0].add_patch(Rectangle((o[0] - 90, o[1] - 85), 180, 170, fill=False, ec="m", lw=2))
ax[0].annotate("", xy=(o[0], o[1] - 40), xytext=(o[0], o[1]), arrowprops=dict(arrowstyle="->", color="m", lw=2))
ax[0].text(o[0] + 3, o[1] - 40, "x_P (inferred)", color="m", fontsize=11)
ax[0].annotate("", xy=(o[0] + 40, o[1]), xytext=(o[0], o[1]), arrowprops=dict(arrowstyle="->", color="m", lw=2))
ax[0].text(o[0] + 40, o[1] + 3, "y_P (inferred)", color="m", fontsize=11)
ax[0].set_xlim(-290, 70); ax[0].set_ylim(-45, 305)
ax[0].set_title("Top view, x_A right, y_A up (z_A toward the viewer); magenta: cad_v2 palm_plate footprint", fontsize=13)
ax[0].set_xlabel("x_A (mm)", fontsize=13); ax[0].set_ylabel("y_A (mm)", fontsize=13)
ax[1].add_patch(Rectangle((o[0] - 90, o[2] + 23.5), 180, 3, fill=True, fc="m", alpha=0.4))
ax[1].axhline(o[2], color="m", lw=1, ls=":")
ax[1].text(-285, o[2] + 2, "yaw-centre plane z_A -75.9", color="m", fontsize=11)
ax[1].set_xlim(-290, 70); ax[1].set_ylim(-165, 40)
ax[1].set_title("Side view, x_A right, z_A up; magenta: cad_v2 palm_plate (z_P 23.5..26.5)", fontsize=13)
ax[1].set_xlabel("x_A (mm)", fontsize=13); ax[1].set_ylabel("z_A (mm)", fontsize=13)
fig.tight_layout()
fig.savefig(f"{ROOT}/current/bbox_views.png", dpi=70)
print("wrote", f"{ROOT}/current/bbox_views.png")
