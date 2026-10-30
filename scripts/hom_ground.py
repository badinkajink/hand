#!/usr/bin/env python3
"""Ground clearance of the grids' grasps: how far the plant's finger spheres reach below the plane through the object's
lowest point, the table an object would rest on (owner, 2026-10-30: "only allow grasps that would not penetrate the
ground (with a tiny bit of tolerance)", in place of grasping from the ground, raising and then manipulating).

The HOM tasks rest the object on a 3 mm post whose collisions end when the squeeze does, so a grasp may put fingertips
below the object. For every grasp of the pose toy's grids (rows grid2dff_*, scripts/fp_sr2_toy.py grid2d) this script
places the plant's collision spheres (fp_sr2_toy.finger_frames, the spheres of ff_clearance_v4b) at the touch pose and at
the open pose 8 mm further out along each contact normal (hom_tasks.open_pose), and takes the smallest height of a sphere's
bottom above that plane. A grasp is ground-feasible when the touch pose's value is at least -TOL mm.

    .venv/bin/python scripts/hom_ground.py --shape sphere --d 40       # one object
    .venv/bin/python scripts/hom_ground.py --all                       # every size of hom_ea.SIZES
Rows: docs/experiments/20261029-foundational_pose_review/rows/grid2dff_v4b/ground/<pose><d>.jsonl, one per grasp:
pose family, grid index, grasp index, touch_mm, open_mm.
"""
from __future__ import annotations

import os

os.environ.setdefault("SCALE_HAND", "cad_v2")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fp_sr2_toy as FP  # noqa: E402

SRC = "grid2dff"
OUT = FP.ROWS / "grid2dff_v4b" / "ground"
TOL = 1.0                                     # mm a sphere may reach below the ground plane
OPEN = 0.008                                  # hom_tasks.open_pose's 8 mm
SIZES = {"sphere": (25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 56.0, 63.0, 71.0, 80.0),
         "cylinder": (8.0, 10.0, 13.0, 16.0, 20.0, 25.0, 32.0, 40.0, 50.0)}


def lowest(qs, M, z):
    """Smallest height (m) of a collision sphere's bottom above the plane at height z, palm frame (z up)."""
    best = np.inf
    for f in FP.FINGERS:
        S = FP.spheres(f)
        for b, (R, p) in FP.finger_frames(f, qs[f], M[f]).items():
            c, r = S[b][0], S[b][1]
            best = min(best, float(((c @ R[2]) + p[2] - r).min()))
    return best - z


def clearance(fp, v, d6, branch):
    """(touch, open) ground clearance (mm) of grasp v of pose family fp on layout d6 (mm)."""
    M = FP.mounts6(d6)
    obj, tg, n, pc, _ax = FP.geometry(fp, np.asarray(v, float), M)
    z = float(obj["centre"][2]) - obj["r"]                       # the object's lowest point (cylinder lying flat)
    qt, qo = {}, {}
    for f in FP.FINGERS:
        q0 = FP.finger_state(f, tg[f], M[f], branch[f])[0]
        qt[f] = FP.touch_q(f, pc[f], n[f], M[f], branch[f], q0)
        h = float(FP.support(f, qt[f], n[f]))
        qo[f] = FP.finger_state(f, pc[f] + (h + OPEN) * n[f], M[f], branch[f])[0]
    return 1e3 * lowest(qt, M, z), 1e3 * lowest(qo, M, z)


def run(shape, d):
    FP.set_sizes(**({"sph_d_mm": d} if shape == "sphere" else {"cyl_d_mm": d}))
    name = FP.size_name(SRC, shape, d)
    rows = FP.load(name)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{'tripod' if shape == 'sphere' else 'pinch'}{d:g}.jsonl"
    t0, n = time.time(), 0
    with open(path, "w") as fh:
        for r in rows:
            for fp, rec in r["fp"].items():
                for j, g in enumerate((rec or {}).get("grasps") or []):
                    tm, om = clearance(fp, g["v"], r["d"], g["branch"])
                    fh.write(json.dumps({"pose": fp, "call": int(r["call"]), "grasp": j, "touch_mm": round(tm, 3),
                                         "open_mm": round(om, 3)}) + "\n")
                    n += 1
        fh.flush()
        os.fsync(fh.fileno())
    print(f"{name}: {n} grasps in {time.time() - t0:.0f} s -> {path.relative_to(ROOT)}", flush=True)


def load(shape, d, pose):
    """{(grid index, grasp index): (touch_mm, open_mm)} of one pose family, or {} before the script has run."""
    p = OUT / f"{'tripod' if shape == 'sphere' else 'pinch'}{d:g}.jsonl"
    out = {}
    for x in (p.read_text().splitlines() if p.exists() else []):
        if x.strip():
            r = json.loads(x)
            if r["pose"] == pose:
                out[(r["call"], r["grasp"])] = (r["touch_mm"], r["open_mm"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shape", choices=("sphere", "cylinder"))
    ap.add_argument("--d", type=float)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    specs = [(s, d) for s in ("sphere", "cylinder") for d in SIZES[s]] if a.all else [(a.shape, a.d)]
    for shape, d in specs:
        if not (FP.ROWS / f"{FP.size_name(SRC, shape, d)}.jsonl").exists():
            print(f"{shape} {d:g} mm: no toy rows yet", flush=True)
            continue
        run(shape, d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
