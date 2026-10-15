#!/usr/bin/env python3
"""Hull resolution of the Drake reference on the as-built design surface (fingertip co-design study, the open check of
the hand-off note): Drake's 1 rad/s spin arm on tcd_rs3_re3 is 1.15 times the Drake-field law's, against 0.97-1.12
on the flat family and 1.02 on the printed hull, and its contact surface has 39 quadrature points (64 on the printed
hull). This runs the bed's Drake spin at 1 N and 1 rad/s and the static case at 1 N with the tip mesh built from
surface points thinned to 0.5 mm (the study's meshes) and 0.25 mm, and the law's spin arm (psi 23/46/69 deg) at 1 N on
the same two hulls. Rows: deterioration/hullres.jsonl (bed) and hullres_law.jsonl (law).

    PY=logs/20261001-hom_contact/venv/bin/python
    $PY scripts/tip_codesign_hullres.py --key tcd_rs3_re3 --res 0.0005 0.00025
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tip_codesign_bed as B  # noqa: E402

OUT = ROOT / "docs/experiments/20261014-tip_codesign/deterioration/hullres.jsonl"
OUT_LAW = OUT.with_name("hullres_law.jsonl")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", default="tcd_rs3_re3")
    ap.add_argument("--res", nargs="+", type=float, default=[0.0005, 0.00025])
    a = ap.parse_args()
    import tip_codesign_law as L
    import tip_codesign_geometry as TG
    for res in a.res:
        _, (spec, chain) = B.tcd_model(a.key, "" if res == 0.0005 else f"mres{res:g}", drake=True)
        name = f"drake_{a.key}_hull{res * 1e3:g}mm"
        B.register({name: (spec, chain)})
        B.run([name], ["static"], [1.0], [None], [1.0], OUT)
        B.run([name], ["spin"], [1.0], [1.0], [1.0], OUT)
        t0 = time.time()
        f = L.DrakeField(L.shape_of(a.key), res=res)
        arm = float(np.mean([L.solve(f, math.radians(ps), 1.0)[1] for ps in (23, 46, 69)]) * 1e3)
        d0 = L.solve(f, 0.0, 1.0)
        V, F = TG.shape(a.key).hull(res)
        row = {"case": "law", "model": name, "key": a.key, "hull_res_m": res, "n_hull_vertices": int(len(V)),
               "n_hull_facets": int(len(F)), "law_spin_arm_1N_mm": arm, "law_delta_1N_um": d0[0] * 1e6,
               "law_onset_arm_1N_mm": d0[1] * 1e3, "wall_s": time.time() - t0, "script": "scripts/tip_codesign_hullres.py",
               "when": time.strftime("%Y-%m-%d %H:%M")}
        with open(OUT_LAW, "a") as fh:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
