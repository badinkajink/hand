"""FE basis of the contact estimator: magnet motions and palmar-face displacements per newton of patch load on a dense grid
over a pad, for scripts/fingertip_wrench_model.py and the live viewer.

The model and loads are those of v2_tip_fe_checks.run_sense (imported, not changed): the pad's hex mesh with the gyroid
infill's homogenized stiffness, its back fixed, and Gaussian patch loads (sigma 0.6 mm) of 1 N along -x (pressing), +y and
+z on the palmar face. Here the patch centres lie on a 1 mm grid over the face where the face extends at least 1 mm beyond
the centre, and every load's full displacement is kept on a display subset of the face (one node per 0.5 mm cell). All
load cases share one factorization and are solved in blocks.

    ~/miniconda3/envs/femtip/bin/python scripts/fingertip_wrench_fe.py [--case b_v2]
writes docs/experiments/20261020-fingertip_contact_estimation/data/20261020-<case>_patch_basis.npz

With --surface the patch centres are the surface points of the tip's 1 mm sphere pads (tip_codesign_geometry pads, the
contact model of the simulations) on the whole free surface of the pad, sides included, and each centre takes 1 N along
+x, +y and +z of the tip frame, spread over the FE's free surface nodes by the same Gaussian (nodal areas projected on
the smoothed normal, nodes facing the same way); the display points are the pads' surface points, triangulated:
    ~/miniconda3/envs/femtip/bin/python scripts/fingertip_wrench_fe.py --surface
writes .../data/20261020-<case>_surface_basis.npz
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


def free_surface(model, solid, G, smooth_mm=1.0):
    """Free-surface nodes of the voxel model (compact ids), their outward unit normals (the cell faces' normals summed
    over smooth_mm) and areas (mm^2): every face between a solid cell and a void one, or the grid's edge, gives a quarter
    of its area, projected on the smoothed normal, to each corner. Nodes held fixed are left out."""
    from scipy.spatial import cKDTree
    nc = solid.shape
    hs = (G.hx, G.hy, G.hz)
    acc = np.zeros((model.nn, 3))
    on = np.zeros(model.nn, bool)
    groups = []
    for ax in range(3):
        o1, o2 = [b for b in range(3) if b != ax]
        for sgn in (-1, 1):
            nb = np.zeros(nc, bool)
            src, dst = [slice(None)] * 3, [slice(None)] * 3
            if sgn > 0:
                dst[ax], src[ax] = slice(0, nc[ax] - 1), slice(1, nc[ax])
            else:
                dst[ax], src[ax] = slice(1, nc[ax]), slice(0, nc[ax] - 1)
            nb[tuple(dst)] = solid[tuple(src)]
            idx = np.argwhere(solid & ~nb)
            quarter = 0.25 * hs[o1][idx[:, o1]] * hs[o2][idx[:, o2]]
            nvec = np.zeros(3)
            nvec[ax] = sgn
            for d1 in (0, 1):
                for d2 in (0, 1):
                    off = np.zeros(3, int)
                    off[ax], off[o1], off[o2] = (1 if sgn > 0 else 0), d1, d2
                    I = idx + off
                    cid = model.remap[(I[:, 0] * G.nn[1] + I[:, 1]) * G.nn[2] + I[:, 2]]
                    np.add.at(acc, cid, quarter[:, None] * nvec[None])
                    on[cid] = True
                    groups.append((cid, nvec, quarter))
    surf = np.where(on)[0]
    tree = cKDTree(model.X[surf])
    nrm = np.array([acc[surf[nb]].sum(0) for nb in tree.query_ball_point(model.X[surf], smooth_mm)])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    full_n = np.zeros((model.nn, 3))
    full_n[surf] = nrm
    area = np.zeros(model.nn)
    for cid, nvec, quarter in groups:
        np.add.at(area, cid, quarter * np.maximum(full_n[cid] @ nvec, 0.0))
    keep = surf[~model.fixed_nodes[surf]]
    return keep, full_n[keep], area[keep]


def pad_points(key, x_cut, margin=0.3, rs=0.00075):
    """Surface points (mm, tip frame), outward normals and areas (mm^2) of the tip's 1 mm sphere pads in front of the
    pad's cut face (tip_codesign_geometry.TipShape.pads: centres rs inside the surface along the normal)."""
    import tip_codesign_geometry as TG
    C, N, A, _ = TG.shape(key).pads(full=True, rs=rs)
    P = C + rs * N
    X = np.column_stack([(P[:, 0] - (TG.REACH - 0.0085)) * 1e3, P[:, 1] * 1e3, (TG.Z_MOUNT - P[:, 2]) * 1e3])
    n = np.column_stack([N[:, 0], N[:, 1], -N[:, 2]])
    keep = X[:, 0] > x_cut + margin
    return X[keep], n[keep], A[keep] * 1e6


def triangulate_star(P, centre, max_edge):
    """Triangles over points of a surface that is star-shaped about `centre` (the convex hull of their directions),
    without those that close the open boundary (an edge longer than max_edge), oriented outward."""
    from scipy.spatial import ConvexHull
    D = P - centre
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    tri = ConvexHull(D).simplices
    Q = P[tri]
    edge = np.max([np.linalg.norm(Q[:, i] - Q[:, (i + 1) % 3], axis=1) for i in range(3)], axis=0)
    tri = tri[edge < max_edge]
    Q = P[tri]
    nrm = np.cross(Q[:, 1] - Q[:, 0], Q[:, 2] - Q[:, 0])
    flip = np.einsum("ij,ij->i", nrm, Q.mean(1) - centre) < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    return tri


def surface_basis(a):
    import fe_contact_calibration_fe as FE
    import tpu_tip_magnet as TM
    import tpu_tip_magnet_layouts as TL
    import v2_tip_fe_checks as V
    from scipy.spatial import cKDTree

    t0 = time.time()
    c = V.CASES[a.case]
    sh, G, mat_id, mats, lay, counts, recipe = V.mat_ids(c)
    solid = mat_id >= 0
    model = V.make_model(c, G, mat_id, mats)
    del mat_id
    mn = FE.magnet_nodes(model, lay)
    x_cut = V.X_CUT_B if c.get("body") == "b" else -8.5
    snodes, sn, sa = free_surface(model, solid, G)
    Xs = model.X[snodes]
    pts, nrm, pad_area = pad_points(V.KEY, x_cut)
    print(f"model {time.time() - t0:.0f} s: {len(model.free)} dof, {len(snodes)} free surface nodes "
          f"({sa.sum():.0f} mm2), {len(pts)} pad points ({pad_area.sum():.0f} mm2), magnets {[len(i) for i in mn]}", flush=True)

    # display: each pad point takes the displacement of its nearest free surface nodes facing its way
    tree = cKDTree(Xs)
    dd, ii = tree.query(pts, k=8)
    ok = np.einsum("pkj,pj->pk", sn[ii], nrm) > 0.3
    wd = np.where(ok, 1 / np.maximum(dd, 0.05) ** 2, 0.0)
    wd /= wd.sum(1, keepdims=True)
    tris = triangulate_star(pts, model.X.mean(0), max_edge=2.6)

    sig = TL.SIGMA
    loads = [(i, k) for i in range(len(pts)) for k in range(3)]
    mot = np.zeros((len(pts), 3, len(mn), 6))            # per patch and load direction (+x, +y, +z): magnet t (um), w (mrad)
    ud = np.zeros((len(pts), 3, len(pts), 3), np.float16)  # display-point displacement (um) per patch and direction
    load_u = np.zeros((len(pts), 3, 3))                  # load-weighted displacement at the patch (um) per direction
    n_nodes = np.zeros(len(pts), int)
    wts = []
    for i in range(len(pts)):
        d2 = ((Xs - pts[i]) ** 2).sum(1)
        sel = np.where((d2 < (3 * sig) ** 2) & (sn @ nrm[i] > 0.3))[0]
        w = sa[sel] * np.exp(-d2[sel] / (2 * sig ** 2))
        wts.append((sel, w / w.sum()))
        n_nodes[i] = len(sel)
    nfree = len(model.free)
    for s in range(0, len(loads), a.block):
        blk = loads[s:s + a.block]
        B = np.zeros((nfree, len(blk)))
        for col, (i, k) in enumerate(blk):
            sel, w = wts[i]
            dof = model.dof_map[3 * snodes[sel] + k]
            good = dof >= 0
            np.add.at(B[:, col], dof[good], w[good])
        Xsol = model.fac(B)
        for col, (i, k) in enumerate(blk):
            u = np.zeros(3 * model.nn)
            u[model.free] = Xsol[:, col]
            u = u.reshape(-1, 3)
            for j, nodes in enumerate(mn):
                t, w = TM.rigid_fit(model.X[nodes], u[nodes])
                mot[i, k, j] = np.r_[1e3 * t, 1e3 * w]
            us = u[snodes]
            ud[i, k] = (1e3 * np.einsum("pk,pkc->pc", wd, us[ii])).astype(np.float16)
            sel, w = wts[i]
            load_u[i, k] = 1e3 * (w[:, None] * us[sel]).sum(0)
        if (s // a.block) % 10 == 0 or s + a.block >= len(loads):
            print(f"  loads {s + len(blk)}/{len(loads)}, {time.time() - t0:.0f} s", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    out = Path(a.out).resolve() if a.out else OUT / f"20261020-{a.case}_surface_basis.npz"
    np.savez_compressed(out, case=a.case, key=V.KEY, pts=pts, normals=nrm, pad_area=pad_area, dirs=np.eye(3), motions=mot,
                        load_u=load_u, n_load_nodes=n_nodes, disp_X=pts, disp_u=ud, tris=tris,
                        magnet_centres=np.array([model.X[n].mean(0) for n in mn]), node_centroid=model.X.mean(0),
                        x_cut=x_cut, sigma_mm=sig, n_dof=len(model.free), n_surface_nodes=len(snodes),
                        when=time.strftime("%Y-%m-%d %H:%M"), script="scripts/fingertip_wrench_fe.py --surface")
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.1f} MB), {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", default="b_v2")
    ap.add_argument("--step", type=float, default=1.0, help="patch grid step (mm)")
    ap.add_argument("--block", type=int, default=48, help="load cases per solve")
    ap.add_argument("--surface", action="store_true", help="patches at the sphere pads over the whole free surface")
    ap.add_argument("--out", default=None, help="the surface basis file (default data/20261020-<case>_surface_basis.npz)")
    a = ap.parse_args()
    if a.surface:
        return surface_basis(a)
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
