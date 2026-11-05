import json, time, sys
from pathlib import Path
import numpy as np, trimesh
sys.path.insert(0, str(Path(__file__).parent))
import sweep
REPO = Path(__file__).resolve().parents[3]
P = {p["name"]: p for p in json.loads((REPO / "logs/20261104-cad_v3/extract/parts.json").read_text())["parts"] if p["file"] == "hand"}
h = 0.25
for nm in ["074", "081", "079", "078", "094", "093", "064", "080", "045", "001"]:
    m = trimesh.load(REPO / P[nm]["stl"], force="mesh")
    t = time.time()
    for k in (0, 1):
        R = sweep.rasterize(m, k, h)
        v = (R["hi"] - R["lo"]).sum() * h * h
        print(nm, P[nm]["label"], "k", k, "watertight", m.is_watertight, "wind", m.is_winding_consistent, "mesh vol %.0f" % m.volume,
              "cad vol %.0f" % P[nm]["volume"], "raster vol %.0f" % v, "ratio %.4f" % (v / P[nm]["volume"]), "bad lines", R["bad"],
              "intervals", len(R["lo"]), "%.1fs" % (time.time() - t))
