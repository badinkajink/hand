"""The tool connector boss of the wrist-3 visual mesh in tool0: vertices beyond the 75 mm body (rho > 38 mm).
    ~/.claude/bin/resguard.sh run --mem 1G -- .venv/bin/python -I logs/20261104-cad_v3/verify_flange_convention/mesh_wrist3_connector.py
"""
import json, math, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mesh_wrist3 import read_dae, UR
P, F, M, up = read_dae(UR / "meshes/ur5/visual/wrist3.dae")
P = P * 1e3
E = np.c_[P[:, 0], -P[:, 2], P[:, 1] - 81.8]
rho = np.hypot(E[:, 0], E[:, 1])
out = {}
for name, sel in (("boss_rho_gt_38", rho > 38.0), ("body_rho_37p3_37p7", (rho > 37.3) & (rho < 37.7))):
    Q = E[sel]
    a = np.degrees(np.arctan2(Q[:, 1], Q[:, 0]))
    out[name] = dict(n=int(sel.sum()), centroid_E_mm=np.round(Q.mean(0), 3).tolist(),
                     angle_E_deg=round(float(np.degrees(math.atan2(Q[:, 1].mean(), Q[:, 0].mean()))), 3),
                     angle_range_deg=[round(float(a.min()), 2), round(float(a.max()), 2)],
                     x_range_mm=[round(float(Q[:, 0].min()), 3), round(float(Q[:, 0].max()), 3)],
                     z_range_E_mm=[round(float(Q[:, 2].min()), 3), round(float(Q[:, 2].max()), 3)],
                     z_mid_E_mm=round(float((Q[:, 2].min() + Q[:, 2].max()) / 2), 3),
                     rho_range_mm=[round(float(rho[sel].min()), 3), round(float(rho[sel].max()), 3)])
# any 6 mm hole anywhere on the face side: circles of r ~3 mm among vertices within 10 mm below the face
face = E[:, 2].max()
near = E[(E[:, 2] > face - 10) & (rho > 10) & (rho < 32)]
out["face_z_E_mm"] = round(float(face), 3)
r_from_axis = np.unique(np.round(np.hypot(near[:, 0], near[:, 1]), 1))
out["note"] = "6 mm pin hole search: see holes_near_face in mesh_wrist3.json (only 5.0 mm and 2.5 mm circles found)"
(HERE / "mesh_wrist3_connector.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
