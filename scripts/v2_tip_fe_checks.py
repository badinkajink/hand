#!/usr/bin/env python3
"""Finite-element checks of the 11/22 half-ellipsoid fingertip that the DIP-link v2 carries (2026-10-19 night).

The calibration of 2026-10-18 (scripts/fe_contact_calibration_fe.py, docs/experiments/20261018-fe_contact_calibration/)
fitted Drake's field and the pads to finite elements of the 11/22 print with four 3 x 2 mm magnets 7.0 mm deep on its base
grid; its grid check ran on the capsule alone. The v2 tip (dip_link_v2.FIVE_TRIES, v2/20261018-v2_tips.json) has five
magnets 7.5 mm deep (five_y2p5_z9p8_s0p5: corners at y +-2.5 mm and 4.9 mm along the finger either side of a centre magnet,
the group 0.5 mm toward the mount face), and the owner's prints string and may be wet (2026-10-19 20:20). Cases, each under
the calibration's 18 probes at 0.5, 1, 3 and 6 N (same model: two 0.45 mm perimeters and four 0.2 mm skins of solid TPU over
the homogenized 30 % gyroid, solid 12 MPa, bonded magnets, back and mount faces fixed, printed upright):

  cal4_fine      the calibration's tip on the fine grid (0.2 mm over |y| <= 2 mm and 2.5 mm either side of the contact
                 point): grid convergence of the tip the calibration fitted
  v2_5           the v2 magnets, base grid: does the calibration's tip stand for the v2 tip?
  v2_5_fine      the v2 magnets, fine grid
  v2_none        no magnets: the magnets' own share of the stiffness
  v2_5_uex05/10  5 and 10 % under-extrusion (wet filament, ooze, a partial clog): every bead narrower by that share, so
                 the perimeters 0.855 / 0.81 mm and the gyroid's beads 0.4275 / 0.405 mm wide on the same toolpath (the
                 slicer cell of scripts/tpu_tip_fem_cells.py homogenized again, rows in cells.jsonl); skins unchanged
  v2_5_gyr10     the gyroid's beads alone 10 % narrower
  v2_5_wall081   the perimeters alone 10 % thinner (0.81 mm)
  v2_5_skin07    skins 0.7 mm (seven 0.1 mm layers; the curved tips print at 0.1 mm with top_solid_min_thickness 0.7)
  a_v2           concept a's tip as printed: the v2 magnets and the two dovetail key channels cut from the back face
                 (dip_link_concept_a.tip_cutter at the design's fit_flank 0.05: 1.9 mm deep, 2.3-3.5 mm wide at y +-3.6 mm,
                 5.1 mm from the mount face, with the detent groove); the channel walls free
  b_v2           concept b's pad as printed: the v2 magnets and the tip cut 3.55 mm short at the back (dip_link_concept_b
                 X_SLOT: 13.45 mm deep), the cut face fixed on the cover; the rail grooves and screw-head notches beside the
                 cut face left out

Rows: docs/experiments/20261019-v2_tip_fe_checks/fe.jsonl (one fsynced row per case and probe, the calibration's columns)
and meta.jsonl (grid, dof, timings, peak memory). Resumable per probe.

    ~/.claude/bin/resguard.sh run --mem 6G --cpu 900 -- taskset -c 4-11 env OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \\
        NUMPY_MADVISE_HUGEPAGE=0 ~/miniconda3/envs/femtip/bin/python scripts/v2_tip_fe_checks.py --cases v2_5 cal4_fine
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fe_contact_calibration_fe as FE  # noqa: E402
import tip_codesign_geometry as TG  # noqa: E402

OUT = ROOT / "docs/experiments/20261019-v2_tip_fe_checks"
SCRIPT = "scripts/v2_tip_fe_checks.py"
KEY, BUILD = "tcd_ell7.4x8.5x11x22", "z"


def five(a=2.5, p=9.8, shift=0.5):
    """dip_link_v2.five_layout('five_y<a>_z<p>_s<shift>') as (diameter, thickness, y, z) in tpu_tip_design's frame."""
    import tpu_tip_design as TD
    zc = TD.Z_C - shift
    cs = [(sy * a, zc + sz * p / 2) for sy in (-1, 1) for sz in (-1, 1)] + [(0.0, zc)]
    return [(FE.MAGNET[0], FE.MAGNET[1], y, z) for y, z in sorted(cs, key=lambda c: (c[1], c[0]))]


CASES = {   # name: grid, magnet depth (mm), magnets (None: the calibration's four), recipe changes, wall/skin changes
    "cal4_fine": dict(grid="fine", depth=7.0, mags=None),
    "v2_5": dict(grid="base", depth=7.5, mags="five"),
    "v2_5_fine": dict(grid="fine", depth=7.5, mags="five"),
    "v2_none": dict(grid="base", depth=7.5, mags="none"),
    "v2_5_uex05": dict(grid="base", depth=7.5, mags="five", wall=0.855, bead=0.4275),
    "v2_5_uex10": dict(grid="base", depth=7.5, mags="five", wall=0.81, bead=0.405),
    "v2_5_gyr10": dict(grid="base", depth=7.5, mags="five", bead=0.405),
    "v2_5_wall081": dict(grid="base", depth=7.5, mags="five", wall=0.81),
    "v2_5_skin07": dict(grid="base", depth=7.5, mags="five", n_skin=7, layer=0.1),
    "a_v2": dict(grid="base", depth=7.5, mags="five", body="a"),
    "b_v2": dict(grid="base", depth=7.5, mags="five", body="b"),
    # 2026-10-20: b's pad with the v3 cover's side walls (owner: "if we add side volume to mold/constrain around the finger,
    # that should help with the shear force detachment"): the flank's nodes held in y from the cut face to wall_h toward the
    # palmar side, from the mount face to the apex's z (the pad's widest section); a bilateral hold, so the bound on how much
    # the walls stiffen the pad and change the magnets' motion (the printed walls stand 0.10 mm off and hold only outward)
    "b_v2_wall3": dict(grid="base", depth=7.5, mags="five", body="b", wall_h=3.0),
    "b_v2_wall495": dict(grid="base", depth=7.5, mags="five", body="b", wall_h=4.95),
    # 2026-10-20 evening (owner, after printing pad B00 with five magnets: "with the 5 magnets, we have to lower the magnets
    # farther away from the tip / closer to the magnetometer. it looks like it wont deform very much ... did you measure this
    # difference much?"): b's pad with the four magnets of the 10-18 print (the calibration's grid at 7.5 mm, the 2.9 mm
    # seat, and at 7.0 mm, the 2.7 mm seat), and the five magnets 1.0 and 0.5 mm shallower and 1.0 mm deeper
    "b_v2_four": dict(grid="base", depth=7.5, mags=None, body="b"),
    "b_v2_four_d70": dict(grid="base", depth=7.0, mags=None, body="b"),
    "b_v2_d65": dict(grid="base", depth=6.5, mags="five", body="b"),
    "b_v2_d70": dict(grid="base", depth=7.0, mags="five", body="b"),
    "b_v2_d85": dict(grid="base", depth=8.5, mags="five", body="b"),
    # the same on grid "bmag": on the base grid b's magnets lie in 1.1 mm elements along x at 7.5 mm and cross into 0.55 mm
    # ones at 6.5-7.0 mm, and the five's proximal pair reaches the 0.8 mm zone along z, so the depth sweep mixes grids
    "bm_five_d65": dict(grid="bmag", depth=6.5, mags="five", body="b"),
    "bm_five_d70": dict(grid="bmag", depth=7.0, mags="five", body="b"),
    "bm_five_d75": dict(grid="bmag", depth=7.5, mags="five", body="b"),
    "bm_five_d85": dict(grid="bmag", depth=8.5, mags="five", body="b"),
    "bm_four_d70": dict(grid="bmag", depth=7.0, mags=None, body="b"),
    "bm_four_d75": dict(grid="bmag", depth=7.5, mags=None, body="b"),
    # owner, 2026-10-20 18:35: "is it possible to run just one wall around the magnets instead of 2?": one 0.45 mm perimeter
    # along each pocket, two along the outer outline (metric_magnet_prints.onewall_modifier prints it)
    "bm_five_d75_w1": dict(grid="bmag", depth=7.5, mags="five", body="b", pocket_walls=1),
    "bm_four_d75_w1": dict(grid="bmag", depth=7.5, mags=None, body="b", pocket_walls=1),
    # owner, 2026-10-20 19:48 (the pad pops off its rails): a separate T-shaped cap over the tip end, fastened to the DIP link
    # and to the pad's tip end and "extruded down to mold around the top of the fingertip ... for additional side-to-side
    # retaining". On grid bmag: v3's walls as printed (the 6 mm walls hold the whole 4.95 mm straight flank); the walls and
    # the cap's end plate (the tip end face's nodes held in all three directions) with a 2 mm pin 3 mm into the tip end,
    # 1.5 mm in front of the back face; and those with the cap's side walls along the straight flank from z_t 12.75
    # (z_F -89.80, 0.2 mm past the U3/U5 windows) to the tip end. Bilateral holds, so bounds on the stiffening.
    "bm_five_d75_walls": dict(grid="bmag", depth=7.5, mags="five", body="b", wall_h=4.95),
    "bm_five_d75_capend": dict(grid="bmag", depth=7.5, mags="five", body="b", wall_h=4.95,
                               cap=dict(end=True, pin=(-3.45, 0.0, 1.0, 3.0))),
    "bm_five_d75_cap": dict(grid="bmag", depth=7.5, mags="five", body="b", wall_h=4.95,
                            cap=dict(end=True, pin=(-3.45, 0.0, 1.0, 3.0), shroud=(12.75, 4.95))),
    # 2026-10-21 (owner, 2026-10-20 23:40: "yes please run these next jobs"): the homogenized references of the resolved-bead
    # check (scripts/v2_tip_resolved_check.py): b's pad without magnets, and both bodies with the 30 % gyroid's tensor from
    # the 0.1 mm cell (the resolved model's voxel size) in place of the 0.05 mm cell that every other case takes
    "b_none": dict(grid="base", depth=7.5, mags="none", body="b"),
    "v2_none_c01": dict(grid="base", depth=7.5, mags="none", cell_h=0.1),
    "b_none_c01": dict(grid="base", depth=7.5, mags="none", body="b", cell_h=0.1),
}
X_CUT_B = -8.5 + (17.0 - 13.45)        # b's pad: 13.45 mm deep (v2 fit-size parts B00-B15) of the tip's 17.0 mm


def a_channels(X, Y, Z, fl=0.05, depth=1.8, floor=0.1, root=2.4, head=3.6, yc=3.6, length=5.1, det_z=4.3, groove_r=0.35):
    """Concept a's two channels in the insert frame (x_t = -x_F - 2.05, y_t = -y_F, z_t = Z_MOUNT - z_F): dovetails from the
    back face (x_t = -8.5), half-width root/2 - fl at the face widening to head/2 - fl at `depth`, `floor` deeper, from the
    mount face to `length`; the detent groove a cylinder along y at the floor, z_t det_z (detent_z -71.6 in F)."""
    d = X + 8.5                                            # depth into the tip from the back face
    half = (root / 2 - fl) + ((head - root) / 2 / depth) * d
    out = np.zeros(np.shape(X), bool)
    for s in (-1, 1):
        ch = (d <= depth + floor) & (np.abs(Y - s * yc) <= half) & (Z <= length)
        gr = ((X - (-8.5 + depth + floor)) ** 2 + (Z - det_z) ** 2 <= groove_r ** 2) & (np.abs(Y - s * yc) <= head / 2)
        out |= ch | gr
    return out


def wall_dofs(h, z1=10.86, z0=-1.0):
    """The y DOF of b's flank nodes beside the v3 cover's walls: in each (x, z) row of nodes with x within h of the cut
    face and z from z0 (default: the mount face) to z1 (the apex's z in the insert frame), the nodes with the least and the
    largest y."""
    def f(X):
        m = np.zeros(X.shape, bool)
        idx = np.where((X[:, 0] <= X_CUT_B + h + 1e-6) & (X[:, 2] >= z0 - 1e-6) & (X[:, 2] <= z1 + 1e-6))[0]
        key = np.round(X[idx, 0], 5) * 1e5 + np.round(X[idx, 2], 5)
        order = np.lexsort((X[idx, 1], key))
        ks = key[order]
        m[idx[order[np.r_[True, ks[1:] != ks[:-1]]]], 1] = True
        m[idx[order[np.r_[ks[1:] != ks[:-1], True]]], 1] = True
        return m
    return f


def cap_dofs(end=True, pin=None, shroud=None, z_end=21.41):
    """The DOFs a T-cap over b's tip end holds: every DOF of the tip end face's nodes (end), every DOF of the nodes inside a
    pin (x_t, y_t, radius, depth from the tip end), and the y DOF of the flank nodes beside the cap's side walls
    (shroud = (z_t from which they run to the tip end, height from the cut face))."""
    def f(X):
        m = np.zeros(X.shape, bool)
        if end:
            m[X[:, 2] >= z_end - 1e-6] = True
        if pin is not None:
            xc, yc, r, depth = pin
            m[((X[:, 0] - xc) ** 2 + (X[:, 1] - yc) ** 2 <= r * r + 1e-9) & (X[:, 2] >= z_end - depth - 1e-6)] = True
        if shroud is not None:
            m |= wall_dofs(shroud[1], z1=z_end, z0=shroud[0])(X)
        return m
    return f


def hold_dofs(c):
    """The DOFs a case holds besides the cut face and the mount face: the v3 cover's walls and the T-cap."""
    fs = ([wall_dofs(c["wall_h"])] if "wall_h" in c else []) + ([cap_dofs(**c["cap"])] if "cap" in c else [])
    return (lambda X: np.logical_or.reduce([f(X) for f in fs])) if fs else None


def make_model(c, G, mat_id, mats):
    """TipModel of a case: b's back at its cut face, and the walls' and the cap's holds when the case has them."""
    import tpu_tip_fem_tip as T
    x_back0 = T.X_BACK
    T.X_BACK = X_CUT_B if c.get("body") == "b" else x_back0      # TipModel fixes the nodes at x <= X_BACK
    try:
        return T.TipModel(G, mat_id, mats, verbose=True, fix_dofs=hold_dofs(c))
    finally:
        T.X_BACK = x_back0


def gyroid_cell(w, h=0.05, rho=0.3):
    """Homogenized 6x6 tensor (per unit solid modulus) of the slicer gyroid at density setting rho with beads w mm wide
    (scripts/tpu_tip_fem_cells.slicer_cell), cached in OUT/cells.jsonl."""
    import tpu_tip_fem_cells as TC
    import tpu_tip_fem_core as C
    for r in FE.rows(OUT / "cells.jsonl"):
        if abs(r["w_mm"] - w) < 1e-9 and abs(r["h_mm"] - h) < 1e-9 and abs(r["rho_req"] - rho) < 1e-9:
            return np.array(r["C"]), r
    t0 = time.time()
    occ, L = TC.slicer_cell(rho, h, w=w)
    r = C.homogenize(occ, h, 1.0, 0.45, solver="amg")
    row = dict(kind="slicer", rho_req=rho, w_mm=w, h_mm=h, period_mm=L, rho=float(occ.mean()), n_dof=r["n_dof"],
               C=np.asarray(r["C"]).tolist(), **TC.summarize(np.asarray(r["C"])), t_s=round(time.time() - t0, 1),
               script=SCRIPT, when=time.strftime("%Y-%m-%d %H:%M"))
    FE.append(OUT / "cells.jsonl", row)
    print(f"gyroid cell w {w} mm: solid fraction {row['rho']:.4f}, Ex {row['Ex']:.5f} Ez {row['Ez']:.5f} ({row['t_s']} s)",
          flush=True)
    return np.array(row["C"]), row


def layout_of(c):
    if c["mags"] is None:
        return FE.layout(c["depth"])
    if c["mags"] == "none":
        return dict(depth=c["depth"], mags=[])
    return dict(depth=c["depth"], mags=five())


def classes_pocket_walls(lay, build, pocket_walls):
    """FE.classes with `pocket_walls` perimeters of 0.45 mm along each magnet's outline and two (tpu_tip_fem_tip.WALL)
    along the outer outline, for any build direction (metric_magnet_prints.classes_walls builds along x only); with
    pocket_walls 2 it is tpu_tip_magnet.classes."""
    from scipy import ndimage
    import tpu_tip_fem_tip as T
    import tpu_tip_magnet as TM
    import tpu_tip_magnet_layouts as TL
    H = TM.H_AUX
    occ, o = TL.occupancy(H, lay, 2)
    tpu, mag = occ == 1, occ == 2
    body = tpu | mag
    ax = {"x": 0, "y": 1, "z": 2}[build]
    d_out = np.zeros(tpu.shape, np.float32)
    d_mag = np.full(tpu.shape, np.inf, np.float32)
    for k in range(tpu.shape[ax]):
        sl = [slice(None)] * 3
        sl[ax] = k
        sl = tuple(sl)
        if body[sl].any():
            d_out[sl] = ndimage.distance_transform_edt(np.pad(body[sl], 1))[1:-1, 1:-1] * H
        if mag[sl].any():
            d_mag[sl] = ndimage.distance_transform_edt(~mag[sl]) * H
    wall = tpu & ((d_out <= T.WALL + 1e-9) | (d_mag <= pocket_walls * T.WALL / 2 + 1e-9))
    n = tpu.shape[ax]
    m = np.moveaxis(tpu, ax, -1)
    up = np.zeros(m.shape, np.float32)
    down = np.zeros(m.shape, np.float32)
    last = np.full(m.shape[:-1], -1, int)
    for k in range(n):
        last = np.where(~m[..., k], k, last)
        down[..., k] = np.where(m[..., k], (k - last) * H, 0)
    nxt = np.full(m.shape[:-1], n, int)
    for k in range(n - 1, -1, -1):
        nxt = np.where(~m[..., k], k, nxt)
        up[..., k] = np.where(m[..., k], (nxt - k) * H, 0)
    skin = np.moveaxis(m & ((up <= T.N_SKIN * T.LAYER + 1e-9) | (down <= T.N_SKIN * T.LAYER + 1e-9)), -1, ax) & ~wall
    cls = np.zeros(tpu.shape, np.uint8)
    cls[tpu] = 3
    cls[skin] = 2
    cls[wall] = 1
    cls[mag] = 4
    return cls, o


def mat_ids(c):
    """FE.mat_ids for a case: the 11/22 shape, the case's magnets, recipe and shell."""
    import curved_tips_fe as CF
    import tpu_tip_fem_core as C
    import tpu_tip_fem_tip as T
    import tpu_tip_magnet as TM
    import tpu_tip_design as TD
    sh = TG.shape(KEY)
    CF.install(sh)
    g = FE.grid("base" if c["grid"] == "bmag" else c["grid"])
    if c.get("body") == "b":                               # the grid starts at the cut face, which is fixed
        g["x"] = [(X_CUT_B, X_CUT_B + 0.9, 0.45), (X_CUT_B + 0.9, 1.0, 1.1)] + g["x"][2:]
    if c["grid"] == "bmag":                                # every depth and layout's magnets in 0.5 mm along x, 0.3 mm along z
        g["x"] = [(X_CUT_B, 5.4, 0.5), (5.4, 8.5, 0.25)]
        g["z"] = [(0.0, 3.5, 0.8), (3.5, 18.5, 0.3), (18.5, 21.41, 0.4)]
    G = T.Grid(T.zoned_axis(g["x"]), T.zoned_axis(g["y"]), T.zoned_axis(g["z"]))
    lay = layout_of(c)
    saved = (T.WALL, T.N_SKIN, T.LAYER, TD.inside_body)
    T.WALL, T.N_SKIN, T.LAYER = c.get("wall", T.WALL), c.get("n_skin", T.N_SKIN), c.get("layer", T.LAYER)
    inside = TD.inside_body
    if c.get("body") == "a":
        TD.inside_body = lambda X, Y, Z: inside(X, Y, Z) & ~a_channels(X, Y, Z)
    elif c.get("body") == "b":
        TD.inside_body = lambda X, Y, Z: inside(X, Y, Z) & (X >= X_CUT_B)
    try:
        cls, o = (FE.classes(lay, BUILD) if c.get("pocket_walls", 2) == 2
                  else classes_pocket_walls(lay, BUILD, c["pocket_walls"]))
    finally:
        T.WALL, T.N_SKIN, T.LAYER, TD.inside_body = saved
    counts = {k: int((cls == v).sum()) for k, v in (("wall", 1), ("skin", 2), ("core", 3), ("magnet", 4))}
    cc = T.sample_cells(G, cls, o, TM.H_AUX)
    del cls
    recipe = dict(TM.RECIPES["w85_30"], build=BUILD)
    mats, cmap = T.materials(recipe)
    if "bead" in c:                                        # the under-extruded gyroid in place of the table's cell
        C6, cell = gyroid_cell(c["bead"])
        mats = [mats[0], T.rotate_C(C6, BUILD) * recipe["E_s"]]
        recipe.update(bead_mm=c["bead"], core_solid_fraction=cell["rho"])
    if "cell_h" in c:                                      # the gyroid's tensor from the cell at that voxel size
        cell = [r for r in FE.rows(ROOT / "docs/experiments/20261009-tpu_tip_fem/cells.jsonl") if r["kind"] == "slicer"
                and r.get("rho_req") is not None and abs(r["rho_req"] - recipe["rho"]) < 1e-9 and abs(r["h_mm"] - c["cell_h"]) < 1e-9][0]
        mats = [mats[0], T.rotate_C(np.array(cell["C"]), BUILD) * recipe["E_s"]]
        recipe.update(cell_h_mm=c["cell_h"], core_solid_fraction=cell["rho"])
    mats = list(mats) + [C.iso_D(TM.E_MAG, TM.NU_MAG)]
    cmap = dict(cmap)
    cmap[4] = 2
    mat_id = -np.ones(cc.shape, int)
    for k, v in cmap.items():
        mat_id[cc == k] = v
    return sh, G, mat_id, mats, lay, counts, recipe


def run_case(name, max_dof):
    import tpu_tip_fem_tip as T
    import tpu_tip_magnet as TM
    c = CASES[name]
    have = {r["probe"] for r in FE.rows(OUT / "fe.jsonl") if r["case"] == name}
    todo = [p for p in FE.probes() if p[0] not in have]
    if not todo:
        print(name, "done", flush=True)
        return
    t0 = time.time()
    sh, G, mat_id, mats, lay, counts, recipe = mat_ids(c)
    n_dof, n_el = FE.count_dof(G, mat_id)
    print(f"{name}: {n_el} elements, {n_dof} dof, aux voxels {counts}", flush=True)
    if n_dof > max_dof:
        raise SystemExit(f"{name}: {n_dof} dof above --max-dof {max_dof}")
    model = make_model(c, G, mat_id, mats)
    del mat_id
    nodes = model.face_nodes(zlim=(0.4, 21.41))
    X = model.X[nodes].copy()
    xt = sh.fe_surface_x(X[:, 1], X[:, 2])
    stair = np.where(np.isfinite(xt), X[:, 0] - xt, np.nan)
    X[:, 0] = np.where(np.isfinite(xt), xt, X[:, 0])
    tc = time.time()
    W = model.compliance(nodes, block=100)
    tc = time.time() - tc
    mn = FE.magnet_nodes(model, lay) if lay["mags"] else []
    base = dict(case=name, tip="ell11", body=c.get("body", "whole"), wall_hold_mm=c.get("wall_h"), x_back_mm=X_CUT_B if c.get("body") == "b" else -8.5,
                fe_key=KEY, grid=c["grid"], build=BUILD, rho=recipe["rho"], E_s_MPa=recipe["E_s"],
                bead_mm=recipe.get("bead_mm", 0.45), core_solid_fraction=recipe.get("core_solid_fraction"),
                cell_h_mm=recipe.get("cell_h_mm", 0.05), wall_mm=c.get("wall", T.WALL), skin_mm=round(c.get("n_skin", T.N_SKIN) * c.get("layer", T.LAYER), 3),
                magnets=[list(m) for m in lay["mags"]], magnet_top_depth_mm=c["depth"], n_dof=len(model.free),
                n_el=model.n_el, n_face_nodes=len(nodes), n_magnet_nodes=[int(len(i)) for i in mn],
                staircase_mm=[float(np.nanpercentile(stair, q)) for q in (5, 50, 95)], script=SCRIPT)
    print(f"{name}: {len(nodes)} face nodes, compliance {tc:.0f} s", flush=True)
    for pname, kind, R, y_t, z_t in todo:
        z_ins = TG.Z_MOUNT * 1e3 - z_t
        xc = FE.first_touch(sh, kind, R, y_t, z_ins)
        surf = FE.probe_surface(kind, R, xc, y_t, z_ins, X[:, 1], X[:, 2])
        g0 = np.maximum(surf - X[:, 0], 0.0)
        cand = np.where(g0 < 1.5)[0]
        Wc, gc, Xc = W[np.ix_(cand, cand)], g0[cand], X[cand]
        res = {}
        t1 = time.time()
        for F in FE.FORCES:
            d, p, k = FE.solve_probe(Wc, Xc, gc, F)
            res[f"{F:g}"] = dict(approach_um=1e3 * d, k_N_per_mm=k, **FE.patch(Xc, p))
            if abs(F - 1.0) < 1e-9 and kind == "cx" and z_t == 0.0 and R == 12.5 and mn:
                act = np.where(p > 1e-12)[0]
                ids = nodes[cand[act]]
                Fn = np.zeros((len(act), 3))
                Fn[:, 0] = -p[act]
                u = model.solve_full((ids, Fn))
                res["1"]["normal_magnets_um_mrad"] = [[(1e3 * t).tolist(), (1e3 * w).tolist()]
                                                      for t, w in (TM.rigid_fit(model.X[i], u[i]) for i in mn)]
        FE.append(OUT / "fe.jsonl", dict(base, probe=pname, kind=kind, R_mm=R, y_mm=y_t, z_mm=z_t, x_first_touch_mm=xc,
                                         n_candidates=int(len(cand)), per_force=res, t_probe_s=round(time.time() - t1, 1),
                                         when=time.strftime("%Y-%m-%d %H:%M")))
        print(f"  {pname:10s} " + "  ".join(f"{F}: {v['approach_um']:6.1f} um rbar {v['r_bar_mm']:.2f}"
                                           for F, v in res.items()), flush=True)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    FE.append(OUT / "meta.jsonl", dict(base, grid_zones=FE.grid(c["grid"]), aux_voxels=counts, t_assemble_s=model.t_assemble,
                                       t_factor_s=model.t_factor, t_compliance_s=tc, t_total_s=time.time() - t0,
                                       peak_rss_GB=peak, when=time.strftime("%Y-%m-%d %H:%M")))
    print(f"{name}: {time.time() - t0:.0f} s, peak RSS {peak:.2f} GB", flush=True)


SENSE_GAP = {"a_v2": 0.37, **{k: 3.32 - 6.45 for k, c in CASES.items() if c.get("body") == "b"}}   # sensing plane behind the whole tip's back face, mm (DIP study levels:
                                               # a's sense_x 6.82 against the back face's x_F 6.45; b's sensing_plane x_F 3.32)


def run_sense(name):
    """Concept a's tip or b's pad in tpu_tip_magnet_layouts.run's row format, for tpu_tip_magnet_sensing: the rod at the
    contact point at 0.5, 1 and 3 N (approach, the magnets' rigid motions), the pressure-law modulus, and unit normal and
    shear patch loads (Gaussian, sigma 0.6 mm) at y -3, 0, 3 and z_c -4 ... +4 mm with the magnets' motions."""
    import tpu_tip_design as TD
    import tpu_tip_fem_tip as T
    import tpu_tip_magnet as TM
    import tpu_tip_magnet_layouts as TL
    if any(r.get("case") == name for r in FE.rows(OUT / "sense_fe.jsonl")):
        print(name, "sense row done", flush=True)
        return
    c = CASES[name]
    t0 = time.time()
    sh, G, mat_id, mats, lay, counts, recipe = mat_ids(c)
    model = make_model(c, G, mat_id, mats)
    del mat_id
    mn = FE.magnet_nodes(model, lay)
    mot = lambda u: [[(1e3 * t).tolist(), (1e3 * w).tolist()] for t, w in (TM.rigid_fit(model.X[i], u[i]) for i in mn)]  # noqa
    nodes = model.face_nodes(zlim=(TD.Z_C - 3.5, TD.Z_C + 3.5))
    W = model.compliance(nodes)
    g0 = TM.gaps(model, nodes)
    A = T.tributary_areas(model, nodes)
    per, d_fe = {}, []
    for F in (0.5, 1.0, 3.0):
        d, p, _ = T.force_controlled(W, g0, F)
        act = np.where(p > 0)[0]
        Fn = np.zeros((len(act), 3))
        Fn[:, 0] = -p[act]
        per[f"{F:g}"] = dict(delta_um=1e3 * d, magnets=mot(model.solve_full((nodes[act], Fn))))
        d_fe.append(d)
    E_best, worst = TM.best_modulus(lambda dd: T.law_force(A, g0, dd, 1.0), (0.5, 1.0, 3.0), d_fe)
    face = model.face_nodes(zlim=(0.5, TD.Z_END - 0.5))
    Xf = model.X[face]
    patches = []
    for y in TL.PATCH_Y:
        for dz in TL.PATCH_DZ:
            z = TD.Z_C + dz
            wgt = np.exp(-((Xf[:, 1] - y) ** 2 + (Xf[:, 2] - z) ** 2) / (2 * TL.SIGMA ** 2))
            wgt[wgt < 1e-4 * wgt.max()] = 0
            wgt /= wgt.sum()
            sel = wgt > 0
            rec = dict(y=y, z=z)
            for key, (k, sgn) in (("normal", (0, -1.0)), ("shear_y", (1, 1.0)), ("shear_z", (2, 1.0))):
                Fv = np.zeros((sel.sum(), 3))
                Fv[:, k] = sgn * wgt[sel]
                u = model.solve_full((face[sel], Fv))
                rec[key] = mot(u)
                rec[key + "_face_um_per_N"] = float(1e3 * sgn * (u[face[sel], k] * wgt[sel]).sum())
            patches.append(rec)
    row = dict(layout=f"v2_{name}", case=name, body=c.get("body"), wall_hold_mm=c.get("wall_h"), cap=c.get("cap"),
               depth_mm=lay["depth"], magnets=[list(m) for m in lay["mags"]],
               recipe="w85_30", n_dof=len(model.free), n_magnet_nodes=[int(len(i)) for i in mn], per_force=per, E_best_MPa=E_best,
               approach_err_max=worst, patches=patches, sigma_mm=TL.SIGMA, t_total_s=round(time.time() - t0, 1), script=SCRIPT,
               when=time.strftime("%Y-%m-%d %H:%M"))
    FE.append(OUT / "sense_fe.jsonl", row)
    print(f"{name}: E* {E_best:.2f} MPa, approach at 1 N {per['1']['delta_um']:.1f} um, magnet travel "
          f"{np.mean([np.linalg.norm(t) for t, _ in per['1']['magnets']]):.2f} um/N, {len(patches)} patches, "
          f"{time.time() - t0:.0f} s", flush=True)


def sense():
    """Force resolution of the v2 tips from sense_fe.jsonl (tpu_tip_magnet_sensing.analyse, five chips, contact position and a
    uniform background field unknown), at the datasheet noise of GAIN_SEL 7 and at the noise of the gain the field needs
    (dip_link_concept_b.sensing_point's rule), -> OUT/sensing.json."""
    import mlx90393_model as MLX
    import tpu_tip_design as TD
    import tpu_tip_magnet_sensing as MS
    out = []
    for r in FE.rows(OUT / "sense_fe.jsonl"):
        gap = SENSE_GAP[r["case"]]
        ch = MS.chips(r, "board", gap)
        B = MS.B0(r, ch) * 1e3
        bmax = float(np.abs(B).max())
        g = res = None
        for rr in (0, 1):
            for gg in range(7, -1, -1):
                rxy, rz = MLX.range_mT(gg, rr)
                rng = min(rxy, rz, 4.8 * MLX.SENS[gg][rr][0] / MLX.SENS[7][0][0])
                if rng >= 1.25 * bmax and g is None:
                    g, res = gg, rr
        rec = dict(case=r["case"], body=r["body"], gap_mm=gap, B_max_component_mT=round(bmax, 2), gain_sel=g, res=res,
                   magnet_travel_um_per_N=round(float(np.mean([np.linalg.norm(t) for t, _ in r["per_force"]["1"]["magnets"]])), 2),
                   approach_1N_um=round(r["per_force"]["1"]["delta_um"], 1), E_star_MPa=round(r["E_best_MPa"], 2))
        saved = MS.NOISE
        for tag, noise in (("gain7", None), ("gain", g)):
            if noise is not None:
                nxy, nz = MLX.noise_uT(g, res, 0, 5)
                MS.NOISE = np.array([nz, nxy, nxy])
            try:
                a = MS.analyse(r, "board", gap)
            finally:
                MS.NOISE = saved
            both = [p["sigF_n_unk_bg"] for p in a["points"]]
            rec[tag] = dict(centre_force_N=round(a["centre"]["sigF_n_unk_bg"], 4), median_force_N=round(float(np.median(both)), 4),
                            range_force_N=[round(min(both), 4), round(max(both), 4)],
                            points_y_dz_force_N=[[p["y"], round(p["z"] - TD.Z_C, 2), round(p["sigF_n_unk_bg"], 4)]
                                                 for p in a["points"]],
                            centre_sig_y_mm=round(a["centre"]["sig_y_mm_bg"], 3), centre_sig_z_mm=round(a["centre"]["sig_z_mm_bg"], 3),
                            saturated=a["saturated"])
        out.append(rec)
        print(json.dumps(rec), flush=True)
    (OUT / "sensing.json").write_text(json.dumps(dict(rows=out, script=SCRIPT, when=time.strftime("%Y-%m-%d %H:%M")), indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    ap.add_argument("--max-dof", type=int, default=320000)
    ap.add_argument("--sense", nargs="*", default=None, choices=list(SENSE_GAP),
                    help="patch-load rows of these cases for the readout model, then the force resolution of every sense row")
    ap.add_argument("--rows-only", action="store_true", help="with --sense: the patch-load rows alone, no force resolution")
    a = ap.parse_args()
    if a.sense is not None:
        for name in a.sense:
            run_sense(name)
        if not a.rows_only:
            sense()
        return
    for name in a.cases:
        run_case(name, a.max_dof)


if __name__ == "__main__":
    main()
