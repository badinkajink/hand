"""Holes and the connector of the UR5 wrist 3 visual mesh, in tool0 (E), from probe_wrist3.py's vertex dump.

    ~/.claude/bin/resguard.sh run --mem 3G -- .venv/bin/python logs/20261104-cad_v3/flange/probe_wrist3_holes.py
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
VE = np.load(OUT / "wrist3_visual_E.npy")
F = json.load(open(OUT / "wrist3_visual_faces.json"))
faces = np.array(F["faces"])
mats = np.array(F["materials"])


def circle_fit(P):
    """Least-squares circle (Kasa) through 2-D points: centre, radius, rms residual."""
    A = np.c_[2 * P[:, 0], 2 * P[:, 1], np.ones(len(P))]
    b = (P ** 2).sum(1)
    cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
    r = np.sqrt(c + cx ** 2 + cy ** 2)
    res = np.hypot(P[:, 0] - cx, P[:, 1] - cy) - r
    return np.array([cx, cy]), float(r), float(np.sqrt((res ** 2).mean()))


out = {}
# 1. the flange face: vertices at z ~ 0, radius 21..29 (the holes on the 50 mm bolt circle)
sel = (np.abs(VE[:, 2] - 0.02) < 0.05)
P = VE[sel]
r = np.hypot(P[:, 0], P[:, 1])
band = P[(r > 21.5) & (r < 29.0)]
ang = np.degrees(np.arctan2(band[:, 1], band[:, 0]))
# cluster by angle (gaps > 10 deg)
order = np.argsort(ang)
a_sorted = ang[order]
splits = np.where(np.diff(a_sorted) > 10)[0]
clusters = np.split(order, splits + 1)
# wrap: merge first and last if they straddle +-180
if len(clusters) > 1 and (a_sorted[0] + 360 - a_sorted[-1]) < 10:
    clusters[0] = np.r_[clusters[-1], clusters[0]]
    clusters = clusters[:-1]
holes = []
for cl in clusters:
    c, rad, rms = circle_fit(band[cl, :2])
    holes.append(dict(n=int(len(cl)), centre_E_mm=np.round(c, 3).tolist(), r_mm=round(rad, 3), rms_mm=round(rms, 4),
                      bolt_circle_r_mm=round(float(np.hypot(*c)), 3),
                      angle_E_deg=round(float(np.degrees(np.arctan2(c[1], c[0]))), 3)))
print("holes on the flange face (z_E = 0):")
for h in holes:
    print("  ", h)
out["face_holes"] = holes
# depth of each hole: vertices within 0.5 mm of the hole's circle, below the face
for h in holes:
    c = np.array(h["centre_E_mm"])
    d = np.hypot(VE[:, 0] - c[0], VE[:, 1] - c[1])
    s = np.abs(d - h["r_mm"]) < 0.3
    h["z_range_E_mm"] = [round(float(VE[s, 2].min()), 3), round(float(VE[s, 2].max()), 3)]
    print("   hole at", h["angle_E_deg"], "deg: wall vertices z", h["z_range_E_mm"])
# 2. the recess and the outer edge on the face
for lo, hi, name in ((14, 17, "recess"), (30.5, 32.5, "outer")):
    s = sel & (np.hypot(VE[:, 0], VE[:, 1]) > lo) & (np.hypot(VE[:, 0], VE[:, 1]) < hi)
    c, rad, rms = circle_fit(VE[s, :2])
    print(name, "circle on face: centre", np.round(c, 3), "r", round(rad, 3), "rms", round(rms, 4), "n", int(s.sum()))
    out[name] = dict(centre_E_mm=np.round(c, 3).tolist(), r_mm=round(rad, 3))
    sz = (np.abs(np.hypot(VE[:, 0] - c[0], VE[:, 1] - c[1]) - rad) < 0.2)
    print("   its wall z range", round(float(VE[sz, 2].min()), 2), round(float(VE[sz, 2].max()), 2))
# 3. the connector: vertices beyond the 37.5 mm body radius
rr = np.hypot(VE[:, 0], VE[:, 1])
far = VE[rr > 38.0]
print("vertices beyond r 38 mm:", len(far), "x", np.round(far[:, 0].min(), 2), np.round(far[:, 0].max(), 2), "y",
      np.round(far[:, 1].min(), 2), np.round(far[:, 1].max(), 2), "z", np.round(far[:, 2].min(), 2), np.round(far[:, 2].max(), 2))
cen = far.mean(0)
print("  centroid", np.round(cen, 2), "angle", round(float(np.degrees(np.arctan2(cen[1], cen[0]))), 2))
out["connector"] = dict(n_verts=int(len(far)), centroid_E_mm=np.round(cen, 2).tolist(),
                        angle_E_deg=round(float(np.degrees(np.arctan2(cen[1], cen[0]))), 2),
                        bbox_E_mm=[np.round(far.min(0), 2).tolist(), np.round(far.max(0), 2).tolist()])
# 4. small-radius features: everything within r < 6 mm of the axis
s = rr < 6
print("near-axis vertices:", int(s.sum()), "z range", np.round(VE[s, 2].min(), 2), np.round(VE[s, 2].max(), 2))
# 5. the second material
m2 = np.unique(faces[mats != mats[0]])
V2 = VE[m2]
print("material 2 verts", len(V2), "bbox", np.round(V2.min(0), 2), np.round(V2.max(0), 2))
json.dump(out, open(OUT / "wrist3_features.json", "w"), indent=1)
