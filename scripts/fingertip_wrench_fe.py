"""FE basis of the contact estimator: magnet motions and palmar-face displacements per newton of patch load on a dense grid
over a pad, for scripts/fingertip_wrench_model.py and the live viewer.

The model and loads are those of v2_tip_fe_checks.run_sense (imported, not changed): the pad's hex mesh with the gyroid
infill's homogenized stiffness, its back fixed, and Gaussian patch loads (sigma 0.6 mm) of 1 N along -x (pressing), +y and
+z on the palmar face. Here the patch centres lie on a 1 mm grid over the face where the face extends at least 1 mm beyond
the centre, and every load's full displacement is kept on a display subset of the face (one node per 0.5 mm cell). All
load cases share one factorization and are solved in blocks.

    ~/miniconda3/envs/femtip/bin/python scripts/fingertip_wrench_fe.py [--case b_v2]
writes docs/experiments/20261020-fingertip_contact_estimation/data/20261020-<case>_patch_basis.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "docs/experiments/20261020-fingertip_contact_estimation/data"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", default="b_v2")
    ap.add_argument("--step", type=float, default=1.0, help="patch grid step (mm)")
    ap.add_argument("--block", type=int, default=48, help="load cases per solve")
    a = ap.parse_args()
    import fe_contact_calibration_fe as FE
    import tpu_tip_design as TD
    import tpu_tip_magnet as TM
    import tpu_tip_magnet_layouts as TL
    import v2_tip_fe_checks as V

    t0 = time.time()
    c = V.CASES[a.case]
    sh, G, mat_id, mats, lay, counts, recipe = V.mat_ids(c)
    model = V.make_model(c, G, mat_id, mats)
    del mat_id
    mn = FE.magnet_nodes(model, lay)
    face = model.face_nodes(zlim=(0.5, TD.Z_END - 0.5))
    Xf = model.X[face]
    print(f"model {time.time() - t0:.0f} s: {len(model.free)} dof, {len(face)} face nodes, magnets {[len(i) for i in mn]}",
          flush=True)

    # patch centres where the face extends 1 mm around them
    ys = np.arange(-7.0, 7.0 + 1e-9, a.step)
    zs = np.arange(1.0, TD.Z_END - 1.0 + 1e-9, a.step)
    pts = []
    for y in ys:
        for z in zs:
            d2 = (Xf[:, 1] - y) ** 2 + (Xf[:, 2] - z) ** 2
            ring = [(y + dy, z + dz) for dy, dz in ((1, 0), (-1, 0), (0, 1), (0, -1))]
            if d2.min() < 0.3 ** 2 and all(((Xf[:, 1] - yy) ** 2 + (Xf[:, 2] - zz) ** 2).min() < 0.3 ** 2 for yy, zz in ring):
                pts.append((y, z))
    pts = np.array(pts)

    # display subset: one face node per 0.5 mm cell, the one nearest the cell centre
    cell = np.floor(Xf[:, 1:] / 0.5).astype(int)
    cen = (cell + 0.5) * 0.5
    dist = np.linalg.norm(Xf[:, 1:] - cen, axis=1)
    order = np.lexsort((dist, cell[:, 1], cell[:, 0]))
    first = np.r_[True, np.any(np.diff(cell[order], axis=0) != 0, axis=1)]
    disp = face[order[first]]
    print(f"{len(pts)} patch centres ({a.step} mm grid), {len(disp)} display nodes", flush=True)

    loads = [(i, k) for i in range(len(pts)) for k in range(3)]
    mot = np.zeros((len(pts), 3, len(mn), 6))          # per patch, load direction (-x, +y, +z): magnet t (um), w (mrad)
    ufc = np.zeros((len(pts), 3, len(disp), 3), np.float32)   # display-node displacement (um) per patch, direction
    face_u = np.zeros((len(pts), 3))                    # load-weighted face displacement along the load (um)
    nfree = len(model.free)
    for s in range(0, len(loads), a.block):
        blk = loads[s:s + a.block]
        B = np.zeros((nfree, len(blk)))
        wsel = []
        for col, (i, k) in enumerate(blk):
            y, z = pts[i]
            wgt = np.exp(-((Xf[:, 1] - y) ** 2 + (Xf[:, 2] - z) ** 2) / (2 * TL.SIGMA ** 2))
            wgt[wgt < 1e-4 * wgt.max()] = 0
            wgt /= wgt.sum()
            sel = wgt > 0
            sgn = -1.0 if k == 0 else 1.0
            d = model.dof_map[3 * face[sel] + k]
            ok = d >= 0
            np.add.at(B[:, col], d[ok], sgn * wgt[sel][ok])
            wsel.append((sel, wgt, sgn))
        Xs = model.fac(B)
        for col, (i, k) in enumerate(blk):
            u = np.zeros(3 * model.nn)
            u[model.free] = Xs[:, col]
            u = u.reshape(-1, 3)
            for j, nodes in enumerate(mn):
                t, w = TM.rigid_fit(model.X[nodes], u[nodes])
                mot[i, k, j] = np.r_[1e3 * t, 1e3 * w]
            ufc[i, k] = 1e3 * u[disp]
            sel, wgt, sgn = wsel[col]
            face_u[i, k] = 1e3 * sgn * (u[face[sel], k] * wgt[sel]).sum()
        print(f"  loads {s + len(blk)}/{len(loads)}, {time.time() - t0:.0f} s", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"20261020-{a.case}_patch_basis.npz"
    np.savez_compressed(out, case=a.case, pts=pts, motions=mot, disp_nodes=disp, disp_X=model.X[disp], disp_u=ufc,
                        face_u=face_u, magnet_centres=np.array([model.X[n].mean(0) for n in mn]),
                        node_centroid=model.X.mean(0), sigma_mm=TL.SIGMA, n_dof=len(model.free),
                        when=time.strftime("%Y-%m-%d %H:%M"), script="scripts/fingertip_wrench_fe.py")
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.1f} MB), {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
