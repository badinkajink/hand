#!/usr/bin/env python3
"""Sections of the magnetic fingertip's finite-element model under a press, for the page on the fingertip's physical model
(owner, 2026-10-26: "clearly understand the role and implementation of FEA, its relation to the printed tip, and it + the
magnetic field sensing composing the physical model").

The model is the one of the estimator's surface basis (scripts/fingertip_wrench_fe.py --surface, which imports
v2_tip_fe_checks.run_sense's model): concept b's pad of the 11/22 half-ellipsoid tip, walls and skins of solid 85A, the
homogenized 30 % gyroid core, five bonded 3 x 2 mm magnets 7.5 mm under the apex, the cut face and the mount face fixed.
A rigid frictionless sphere of radius 5 mm, the UR5 calibration's indenter, presses the face at the apex along -x to 1 N
and 3 N (the contact solved as a linear complementarity problem on the face nodes, as every FE probe of these studies);
a tangential load of 1 N along +z (toward the tip end) and one along +y are then applied over the 1 N patch. For each load
the script keeps the displacement of every node in two sections, y = 0 (through the centre magnet and the chips U4, U1
and U2) and y = 2.5 mm (through two corner magnets), the material class of the cells there, and every magnet's rigid
motion.

    ~/.claude/bin/resguard.sh run --mem 4G --cpu 300 -- ~/miniconda3/envs/femtip/bin/python \
        scripts/fingertip_physical_model_fe.py
writes docs/experiments/20261026-fingertip_physical_model/data/20261026-fe_sections.npz
"""
from __future__ import annotations

import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "docs/experiments/20261026-fingertip_physical_model/data/20261026-fe_sections.npz"
R_SPHERE = 5.0
APEX_YZ = (0.5, 10.36)                    # the pad's apex (y, z) in the tip frame, mm (the surface basis's highest point)
SECTIONS = (0.0, 2.5)                     # y of the two sections, mm


def main():
    import fe_contact_calibration_fe as FE
    import tpu_tip_fem_tip as T
    import tpu_tip_magnet as TM
    import v2_tip_fe_checks as V

    t0 = time.time()
    c = V.CASES["b_v2"]
    sh, G, mat_id, mats, lay, counts, recipe = V.mat_ids(c)
    # material class of every cell in each section: -1 void, 0 wall and skin (solid TPU), 1 core, 2 magnet
    jsec = [int(np.argmin(np.abs(G.cy - y))) for y in SECTIONS]
    cls_sec = np.stack([mat_id[:, j, :] for j in jsec])
    model = V.make_model(c, G, mat_id, mats)
    del mat_id
    mn = FE.magnet_nodes(model, lay)
    print(f"model {time.time() - t0:.0f} s: {len(model.free)} dof, magnet nodes {[len(i) for i in mn]}", flush=True)

    # sphere press at the apex: candidate face nodes and their gaps to the sphere at first touch
    nodes = model.face_nodes(zlim=(0.4, 21.41))
    X = model.X[nodes].copy()
    xt = sh.fe_surface_x(X[:, 1], X[:, 2])
    X[:, 0] = np.where(np.isfinite(xt), xt, X[:, 0])
    y0, z0 = APEX_YZ
    r2 = (X[:, 1] - y0) ** 2 + (X[:, 2] - z0) ** 2
    g_grid = np.arange(-R_SPHERE, R_SPHERE + 1e-9, 0.02)
    Yg, Zg = np.meshgrid(y0 + g_grid, z0 + g_grid, indexing="ij")
    rr = (Yg - y0) ** 2 + (Zg - z0) ** 2
    ok = rr < R_SPHERE ** 2
    xc = float(np.nanmax(sh.fe_surface_x(Yg[ok], Zg[ok]) + np.sqrt(R_SPHERE ** 2 - rr[ok])))   # centre at first touch
    surf = np.full(len(X), np.inf)
    inside = r2 < R_SPHERE ** 2
    surf[inside] = xc - np.sqrt(R_SPHERE ** 2 - r2[inside])
    g0 = np.maximum(surf - X[:, 0], 0.0)
    cand = np.where(g0 < 1.5)[0]
    W = model.compliance(nodes[cand], block=100)
    out = dict(sections_y=np.array(SECTIONS), xs=G.xs, zs=G.zs, cy=G.cy[jsec], cls=cls_sec,
               magnet_nodes_centre=np.array([model.X[i].mean(0) for i in mn]), sphere_R=R_SPHERE, sphere_xc_first_touch=xc,
               apex_yz=np.array(APEX_YZ), n_dof=len(model.free), recipe=str(recipe))
    sec_nodes = [np.where(np.abs(model.X[:, 1] - G.ys[np.argmin(np.abs(G.ys - y))]) < 1e-9)[0] for y in SECTIONS]
    for k, ids in enumerate(sec_nodes):
        out[f"sec{k}_X"] = model.X[ids]
    loads = {}
    for F in (1.0, 3.0):
        d, p, _ = T.force_controlled(W, g0[cand], F)
        act = np.where(p > 1e-12)[0]
        ids = nodes[cand[act]]
        Fn = np.zeros((len(act), 3))
        Fn[:, 0] = -p[act]
        loads[f"press{F:g}"] = (ids, Fn, d, p[act])
        if F == 1.0:
            for axis, kk in (("shear_z", 2), ("shear_y", 1)):
                Ft = np.zeros((len(act), 3))
                Ft[:, kk] = p[act] / p[act].sum()
                loads[axis] = (ids, Ft, np.nan, p[act])
    for name, (ids, Fv, d, pa) in loads.items():
        u = model.solve_full((ids, Fv))
        mot = np.array([np.r_[1e3 * t, 1e3 * w] for t, w in (TM.rigid_fit(model.X[i], u[i]) for i in mn)])
        out[f"{name}_approach_um"] = 1e3 * d
        out[f"{name}_contact_X"] = model.X[ids]
        out[f"{name}_contact_p"] = pa
        out[f"{name}_magnets"] = mot                    # (5, 6): translation (um) and rotation (mrad)
        for k, sids in enumerate(sec_nodes):
            out[f"{name}_sec{k}_u"] = (1e3 * u[sids]).astype(np.float32)   # um
        surf_u = 1e3 * u[nodes]
        out[f"{name}_face_X"] = model.X[nodes]
        out[f"{name}_face_u"] = surf_u.astype(np.float32)
        print(f"{name}: approach {1e3 * d:.1f} um, contact nodes {len(ids)}, magnet travel "
              f"{np.round(np.linalg.norm(mot[:, :3], axis=1), 2)} um", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **out)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.1f} MB), {time.time() - t0:.0f} s, peak RSS "
          f"{peak:.2f} GB", flush=True)


def aux_classes():
    """The slicer's material classes on the 0.05 mm auxiliary grid (0 void, 1 wall, 2 skin, 3 core, 4 magnet) in the two
    sections, as v2_tip_fe_checks.mat_ids computes them before sampling them onto the FE cells."""
    import curved_tips_fe as CF
    import fe_contact_calibration_fe as FE
    import tpu_tip_design as TD
    import tpu_tip_fem_tip as T
    import tpu_tip_magnet as TM
    import v2_tip_fe_checks as V
    c = V.CASES["b_v2"]
    sh = V.TG.shape(V.KEY)
    CF.install(sh)
    lay = V.layout_of(c)
    inside = TD.inside_body
    TD.inside_body = lambda X, Y, Z: inside(X, Y, Z) & (X >= V.X_CUT_B)
    try:
        cls, o = FE.classes(lay, V.BUILD)
    finally:
        TD.inside_body = inside
    h = TM.H_AUX
    js = [int(round((y - o[1]) / h - 0.5)) for y in SECTIONS]
    out = OUT.with_name("20261026-aux_class_sections.npz")
    np.savez_compressed(out, cls=np.stack([cls[:, j, :] for j in js]), origin=np.array(o), h=h,
                        y=np.array([o[1] + (j + 0.5) * h for j in js]), shape=np.array(cls.shape))
    print(f"wrote {out.relative_to(ROOT)}: classes {cls.shape}, sections at y {[o[1] + (j + 0.5) * h for j in js]}",
          flush=True)


if __name__ == "__main__":
    if "--classes" in sys.argv:
        aux_classes()
    else:
        main()
