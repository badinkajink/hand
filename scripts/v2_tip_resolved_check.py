#!/usr/bin/env python3
"""Resolved-bead check of the homogenized 11/22 fingertip model (2026-10-21; owner, 2026-10-20 23:40: "yes please run
these next jobs", the item "FE of the 11/22 pad with its beads resolved" of the printed-fingertips overview).

The homogenized model of scripts/v2_tip_fe_checks.py (two 0.45 mm perimeters and four 0.2 mm skins of solid TPU over the
30 % gyroid's effective tensor, graded hexahedra of 0.2-1.1 mm, CHOLMOD) is compared with the same tip whose gyroid core
is drawn bead by bead on 0.1 mm voxels (scripts/tpu_tip_fem_cells.slicer_layer_mask, the bead pattern the homogenized
cell was computed from) and solved on the GPU (scripts/tpu_tip_fem_gpu.py: hex8 with Wilson-Taylor incompatible modes,
one solid TPU, matrix-free conjugate gradients). Both models take their wall, skin and core classes from the same
0.05 mm auxiliary voxels (fe_contact_calibration_fe.classes on the installed 11/22 shape, printed upright, layers across
z), fix the same faces (the back face x = -8.5 mm, or b's cut face, and the mount face z = 0), measure each face node's
gap from the analytic surface, and press the same rigid frictionless probes (fe_contact_calibration_fe.probes) from first
touch. Solid 85A: 12 MPa, Poisson 0.45. The GPU solver is displacement-controlled; the approach at 0.5, 1 and 3 N comes
from secant steps on log F against log approach (each force met within 1 %, then interpolated).

Bodies (homogenized references in docs/experiments/20261019-v2_tip_fe_checks/fe.jsonl):
  whole   the 11/22 tip without magnets (cases v2_none, 0.05 mm cell, and v2_none_c01, 0.1 mm cell)
  b       concept b's pad, cut 3.55 mm short at the back with the cut face fixed, without magnets (b_none, b_none_c01)
  b00     pad B00 as PrusaSlicer printed it on the XL (`gvoxels`: its G-code's beads, 0.1 mm layers, the five magnets as
          bonded TPU; references b_v2, the design pad with five bonded NdFeB magnets, and b_none)
  b00m    the same with every bead narrowed by one factor until the voxels hold the weighed B parts' 1732.8 mm3 of TPU
          (`gvoxels --match`; references b_v2 and b_v2_cl01, the design pad homogenized with a 0.1 mm-layer cell)

    ~/.claude/bin/resguard.sh run --mem 8G --cpu 300 -- env NUMPY_MADVISE_HUGEPAGE=0 \\
        ~/miniconda3/envs/femtip/bin/python scripts/v2_tip_resolved_check.py voxels --body whole
    ~/.claude/bin/resguard.sh run --mem 6G --cpu 200 -- env WARP_CACHE_PATH=$(mktemp -d) .venv/bin/python \\
        scripts/v2_tip_resolved_check.py solve --body whole --probes cx12.5_z0 s_apex s_y4.1 cz12.5_y0
Rows: docs/experiments/20261019-v2_tip_fe_checks/resolved.jsonl, one fsynced row per solve (kind "solve") and one per
body, probe and force (kind "force"); resumable per probe. Voxels: assets/fem/20261021-v2_tip_resolved/ (gitignored).
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fe_contact_calibration_fe as FE  # noqa: E402
import tip_codesign_geometry as TG  # noqa: E402
import v2_tip_fe_checks as V  # noqa: E402

OUT = V.OUT / "resolved.jsonl"
VOX = ROOT / "assets/fem/20261021-v2_tip_resolved"
SCRIPT = "scripts/v2_tip_resolved_check.py"
E_S, NU, RHO, H = 12.0, 0.45, 0.3, 0.1
FORCES = (0.5, 1.0, 3.0)
REF = {"whole": ("v2_none", "v2_none_c01"), "b": ("b_none", "b_none_c01"), "b00": ("b_v2", "b_none"),
       "b00m": ("b_v2", "b_v2_cl01")}
X0 = {"whole": -8.5, "b": V.X_CUT_B, "b00": V.X_CUT_B, "b00m": V.X_CUT_B}
GCODE_BODIES = ("b00", "b00m")
FACE_DEPTH = 0.2                           # mm: depth from a fixed face within which a G-code model's first solid node is fixed
B_WEIGHED_MM3 = 1732.8                     # the B fit parts on the owner's scale, 2.01 g at 1.16 g/cm3 (print_audit.json)


def voxels(body, h=H, rho=RHO):
    """Occupancy on h-voxels: walls and skins solid, the core filled with the slicer's gyroid beads layer by layer along
    z (tpu_tip_fem_tip.resolved_voxels on the classes of the homogenized model); node planes on the fixed faces."""
    import curved_tips_fe as CF
    import tpu_tip_design as TD
    import tpu_tip_fem_cells as CE
    import tpu_tip_fem_core as C
    import tpu_tip_magnet as TM
    t0 = time.time()
    sh = TG.shape(V.KEY)
    CF.install(sh)
    inside = TD.inside_body
    if body == "b":
        TD.inside_body = lambda X, Y, Z: inside(X, Y, Z) & (X >= V.X_CUT_B)
    try:
        cls, o = FE.classes(dict(depth=7.5, mags=[]), V.BUILD)
    finally:
        TD.inside_body = inside
    ha = TM.H_AUX
    i0, k0 = int(round((X0[body] - o[0]) / ha)), int(round((0.0 - o[2]) / ha))
    assert abs(o[0] + i0 * ha - X0[body]) < 1e-6 and abs(o[2] + k0 * ha) < 1e-6, (o, i0, k0)
    assert not cls[:i0].any() and not cls[:, :, :k0].any()
    cls = cls[i0:, :, k0:]
    origin = np.array([o[0] + i0 * ha, o[1], o[2] + k0 * ha])
    counts = {k: int((cls == v).sum()) for k, v in (("wall", 1), ("skin", 2), ("core", 3))}
    step = int(round(h / ha))
    c = cls[step // 2::step, step // 2::step, step // 2::step]
    del cls
    solid, core = (c == 1) | (c == 2), c == 3
    L, _ = CE.slicer_period(rho)
    n, per = int(round(L / h)), int(round(CE.LAYER / h))
    beads = np.zeros(c.shape, bool)
    masks = {}
    for kb in range(c.shape[2]):
        zc = (kb // per + 0.5) * CE.LAYER
        key = round(zc % L, 6)
        if key not in masks:
            masks[key] = CE.slicer_layer_mask(L, zc, h)
        reps = (c.shape[0] // n + 2, c.shape[1] // n + 2)
        beads[:, :, kb] = np.tile(masks[key], reps)[:c.shape[0], :c.shape[1]]
    occ = C.largest_component(solid | (core & beads))
    info = dict(body=body, h_mm=h, rho_req=rho, period_mm=L, aux_voxels=counts, shape=list(occ.shape),
                n_solid=int(occ.sum()), n_shell=int(solid.sum()), core_solid_fraction=float((occ & core).sum() / max(core.sum(), 1)),
                dropped_islands=int((solid | (core & beads)).sum() - occ.sum()), t_s=round(time.time() - t0, 1),
                peak_rss_GB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
    VOX.mkdir(parents=True, exist_ok=True)
    path = VOX / f"{body}_g30_h{h:g}.npz"
    np.savez_compressed(path, occ=occ, origin=origin, h=h, info=json.dumps(info))
    print(f"{path}: {occ.shape}, {info['n_solid']} solid voxels, core solid fraction {info['core_solid_fraction']:.4f}, "
          f"islands dropped {info['dropped_islands']}, {info['t_s']} s, peak RSS {info['peak_rss_GB']:.2f} GB", flush=True)
    return path


GCODE_B00 = ROOT / "logs/20261020-dip_link/v3/b/print_b00_five/20261020-dip_link_v3b_print_pad_ell11_5_B00_85a_t2_g30.gcode"
SUB = 4                                                            # raster pixels per voxel side (0.025 mm)


def _raster(layers, sx, sy, cb, ct, nx, ny, h, ws=1.0):
    """Per layer, the voxels (nx, ny) at least half covered by the layer's beads drawn at their widths, the bed frame mapped
    to the tip frame by x_t = sx (x_b - cb_x) + ct_x, y_t = sy (y_b - cb_y) + ct_y (origin at (ct - extent / 2))."""
    from PIL import Image, ImageDraw
    r = h / SUB
    x_lo, y_lo = X0["b"], -7.4
    out = []
    for l in layers:
        img = Image.new("L", (nx * SUB, ny * SUB), 0)
        dr = ImageDraw.Draw(img)
        s = l["seg"]
        xa = (sx * (s[:, 0] - cb[0]) + ct[0] - x_lo) / r
        ya = (sy * (s[:, 1] - cb[1]) + ct[1] - y_lo) / r
        xb = (sx * (s[:, 2] - cb[0]) + ct[0] - x_lo) / r
        yb = (sy * (s[:, 3] - cb[1]) + ct[1] - y_lo) / r
        for k in range(len(s)):
            wk = ws * s[k, 4]
            q = 0.5 * wk / r
            dr.line([(xa[k], ya[k]), (xb[k], yb[k])], fill=255, width=max(1, int(round(wk / r))))
            dr.ellipse((xa[k] - q, ya[k] - q, xa[k] + q, ya[k] + q), fill=255)
            dr.ellipse((xb[k] - q, yb[k] - q, xb[k] + q, yb[k] + q), fill=255)
        A = np.asarray(img, np.float32) / 255.0
        out.append(A.reshape(ny, SUB, nx, SUB).mean((1, 3)).T >= 0.5)
    return out


def gcode_voxels(gcode=GCODE_B00, h=H, name="b00", match_mm3=None):
    """Occupancy of b's pad as PrusaSlicer printed it (scripts/eflesh_tip_fem.py's method for the upright pad): every
    layer's beads (scripts/gcode_audit.py, arcs split into 0.1 mm chords) drawn at their widths on a 0.025 mm raster and
    averaged onto h voxels, each layer filling the voxel layers between its own z and the previous layer's. Bed z is the
    tip's z (the mount face on the bed); bed x and y map to the tip's x and y with the signs that best overlap the design
    pad, the bead extents centred on the design's. The five magnets fill their design cylinders as bonded TPU. match_mm3:
    every bead's width scaled by one factor, found by bisection, so that the printed voxels hold that volume of TPU (the
    weighed parts' volume, print_audit.json) in place of the beads' nominal widths."""
    import curved_tips_fe as CF
    import gcode_audit as GA
    import tpu_tip_design as TD
    import tpu_tip_fem_core as C
    import tpu_tip_magnet_layouts as TL
    t0 = time.time()
    L, types = GA.parse(gcode, 0)
    s = np.vstack([l["seg"] for l in L])
    w2 = 0.5 * float(np.median(s[:, 4]))
    lo = np.array([min(s[:, 0].min(), s[:, 2].min()), min(s[:, 1].min(), s[:, 3].min())]) - w2
    hi = np.array([max(s[:, 0].max(), s[:, 2].max()), max(s[:, 1].max(), s[:, 3].max())]) + w2
    cb = 0.5 * (lo + hi)
    ext_t = np.array([8.5 - X0["b"], 14.8])
    ct = np.array([X0["b"], -7.4]) + ext_t / 2
    nx, ny = int(math.ceil(ext_t[0] / h - 1e-6)), int(math.ceil(ext_t[1] / h - 1e-6))
    zs = [l["z"] for l in L]
    nz = int(round(max(zs) / h))
    sh = TG.shape(V.KEY)
    CF.install(sh)
    xs, ys, zc = (X0["b"] + h * (np.arange(nx) + 0.5), -7.4 + h * (np.arange(ny) + 0.5), h * (np.arange(nz) + 0.5))
    Xg, Yg, Zg = np.meshgrid(xs, ys, zc, indexing="ij")
    design = TD.inside_body(Xg, Yg, Zg) & (Xg >= X0["b"])
    sample = list(range(5, len(L), 12))                              # orientation from every twelfth layer
    iou = {}
    for sx in (1, -1):
        for sy in (1, -1):
            ms = _raster([L[i] for i in sample], sx, sy, cb, ct, nx, ny, h)
            inter = union = 0
            for i, m in zip(sample, ms):
                k = min(nz - 1, int(round(L[i]["z"] / h)) - 1)
                inter += int((m & design[:, :, k]).sum())
                union += int((m | design[:, :, k]).sum())
            iou[(sx, sy)] = inter / union
    sx, sy = max(iou, key=iou.get)

    def stack(ws):
        occ = np.zeros((nx, ny, nz), bool)
        z_prev = 0.0
        for l, m in zip(L, _raster(L, sx, sy, cb, ct, nx, ny, h, ws)):
            k0, k1 = int(round(z_prev / h)), int(round(l["z"] / h))
            occ[:, :, k0:max(k1, k0 + 1)] |= m[:, :, None]
            z_prev = l["z"]
        return occ
    ws, steps = 1.0, []
    occ = stack(ws)
    if match_mm3:
        lo_, hi_ = 0.6, 1.0
        for _ in range(12):
            ws = 0.5 * (lo_ + hi_)
            occ = stack(ws)
            v = float(occ.sum() * h ** 3)
            steps.append((round(ws, 5), round(v, 1)))
            if abs(v / match_mm3 - 1) < 0.002:
                break
            lo_, hi_ = (ws, hi_) if v < match_mm3 else (lo_, ws)
    printed = occ.copy()
    mag = np.zeros_like(occ)
    for m in V.five():
        mag |= TL.in_mag(Xg, Yg, Zg, TL.magnet_cyl(m, 7.5))
    occ |= mag
    occ = C.largest_component(occ)
    vol = float(sum(l["vol"] for l in L))
    info = dict(body=name, gcode=str(Path(gcode).relative_to(ROOT)), h_mm=h, sign_x=sx, sign_y=sy, width_scale=ws,
                match_mm3=match_mm3, width_steps=steps,
                iou={f"{a:+d}{b:+d}": round(v, 4) for (a, b), v in iou.items()}, bead_extent_mm=(hi - lo).tolist(),
                layers=len(L), shape=list(occ.shape), n_solid=int(occ.sum()), printed_voxel_mm3=float(printed.sum() * h ** 3),
                extruded_mm3=vol, design_mm3=float(design.sum() * h ** 3), magnet_voxels=int(mag.sum()),
                magnet_voxels_printed=int((mag & printed).sum()),
                design_iou=float(((printed | mag) & design).sum() / ((printed | mag) | design).sum()),
                dropped_islands=int((printed | mag).sum() - occ.sum()), t_s=round(time.time() - t0, 1),
                peak_rss_GB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
    VOX.mkdir(parents=True, exist_ok=True)
    path = VOX / f"{name}_g30_h{h:g}.npz"
    np.savez_compressed(path, occ=occ, origin=np.array([X0["b"], -7.4, 0.0]), h=h, info=json.dumps(info))
    print(json.dumps(info), flush=True)
    return path


def ref_rows(case):
    return {r["probe"]: r for r in FE.rows(V.OUT / "fe.jsonl") if r["case"] == case}


def loglog(pts, F):
    """Approach (mm) and r_bar at load F from (delta, F, r_bar) points: the two nearest in log F, interpolated linearly in
    log F against log approach."""
    pts = sorted(pts, key=lambda q: abs(math.log(q[1] / F)))[:2]
    (d0, f0, r0), (d1, f1, r1) = sorted(pts, key=lambda q: q[1])
    if abs(math.log(f1 / f0)) < 1e-12:
        return d0, r0
    t = math.log(F / f0) / math.log(f1 / f0)
    return math.exp(math.log(d0) + t * (math.log(d1) - math.log(d0))), r0 + t * (r1 - r0)


def solve(body, probes, tol=1e-6, ftol=0.01, max_it=5):
    import torch
    import tpu_tip_fem_gpu as GT
    D = np.load(VOX / f"{body}_g30_h{H:g}.npz")
    occ, origin, h = D["occ"], D["origin"], float(D["h"])
    vinfo = json.loads(str(D["info"]))
    t0 = time.time()
    tip = GT.GpuTip(occ, origin, h, E_S, NU)
    if body in GCODE_BODIES:
        # a printed face is not a full voxel plane (narrowed beads leave the outer voxel layer of the cut face nearly empty):
        # fix the first solid node of every row normal to the cut face and to the mount face within FACE_DEPTH of it
        I = np.round((tip.X - origin) / h).astype(int)
        fx = np.zeros(len(I), bool)
        for a, (b1, b2), x0 in ((0, (1, 2), X0[body]), (2, (0, 1), 0.0)):
            m = np.full((I[:, b1].max() + 1, I[:, b2].max() + 1), 1 << 30)
            np.minimum.at(m, (I[:, b1], I[:, b2]), I[:, a])
            first = I[:, a] == m[I[:, b1], I[:, b2]]
            fx |= first & (tip.X[:, a] <= x0 + FACE_DEPTH + 1e-9)
        tip.fixed_dof, tip.n_fixed = np.repeat(fx, 3), int(fx.sum())
    fixed = tip.fixed_dof[0::3]
    assert abs(tip.X[:, 0].min() - X0[body]) < 1e-6 and abs(tip.X[:, 2].min()) < 1e-6
    nodes = GT.face_nodes(tip.X, fixed)
    nodes = nodes[tip.X[nodes, 2] >= 0.4]                          # zlim of the homogenized model's face nodes
    sh = TG.shape(V.KEY)
    X = tip.X[nodes].copy()
    xt = sh.fe_surface_x(X[:, 1], X[:, 2])
    stair = np.where(np.isfinite(xt), X[:, 0] - xt, np.nan)
    X[:, 0] = np.where(np.isfinite(xt), xt, X[:, 0])
    print(f"{body}: {tip.n_el} elements, {tip.ndof} dof, {len(nodes)} face nodes, {tip.n_fixed} fixed nodes, staircase "
          f"5/50/95 % {np.nanpercentile(stair, 5):.3f}/{np.nanpercentile(stair, 50):.3f}/{np.nanpercentile(stair, 95):.3f} mm, "
          f"setup {tip.t_setup:.1f} s", flush=True)
    refs = {c: ref_rows(c) for c in REF[body]}
    done = {r["probe"] for r in FE.rows(OUT) if r.get("kind") == "force" and r["body"] == body}
    base = dict(body=body, fe_key=V.KEY, build=V.BUILD, rho=RHO, E_s_MPa=E_S, nu=NU, h_mm=h, n_el=tip.n_el, n_dof=tip.ndof,
                n_face_nodes=len(nodes), x_fixed_mm=X0[body], voxels=vinfo, n_fixed_nodes=tip.n_fixed,
                fixed_rule=(f"first solid node of each row within {FACE_DEPTH} mm of the cut face and the mount face"
                            if body in GCODE_BODIES else "nodes on the cut or back face plane and the mount face plane"),
                staircase_mm=[float(np.nanpercentile(stair, q)) for q in (5, 50, 95)], script=SCRIPT)
    allp = {p[0]: p for p in FE.probes()}
    for pname in probes:
        if pname in done:
            print(f"  {pname} done", flush=True)
            continue
        _, kind, R, y_t, z_t = allp[pname]
        z_ins = TG.Z_MOUNT * 1e3 - z_t
        xc = FE.first_touch(sh, kind, R, y_t, z_ins)
        surf = FE.probe_surface(kind, R, xc, y_t, z_ins, X[:, 1], X[:, 2])
        g0 = np.maximum(surf - X[:, 0], 0.0)
        cand = np.where(g0 < 1.5)[0]
        nc, gc = nodes[cand], g0[cand]
        ref = refs[REF[body][0]][pname]["per_force"]
        dref = {F: 1e-3 * ref[f"{F:g}"]["approach_um"] for F in FORCES}
        x, active, pts = None, None, []
        tp = time.time()
        for i, Ft in enumerate(FORCES):
            if pts:                                                 # from the closest solve along the reference's exponent
                Fp = FORCES[i - 1]
                n_ref = math.log(Ft / Fp) / math.log(dref[Ft] / dref[Fp])
                d_last, f_last, _ = min(pts, key=lambda q: abs(math.log(q[1] / Ft)))
                d = d_last * (Ft / f_last) ** (1 / min(2.0, max(0.8, n_ref)))
            else:
                d = dref[Ft]
            for it in range(max_it):
                ts = time.time()
                p, x, active, info = GT.contact(tip, nc, gc, d, x0=x, active0=active, tol=tol)
                torch.cuda.synchronize()
                m = GT.metrics(tip.X, nc, p, gc, d)
                pts.append((d, m["F"], m["arm"]))
                FE.append(OUT, dict(base, kind="solve", probe=pname, target_N=Ft, delta_mm=d, n_candidates=len(cand),
                                    x_first_touch_mm=xc, t_solve_s=round(time.time() - ts, 1),
                                    gpu_mem_GB=torch.cuda.max_memory_allocated() / 1e9, **info, **m,
                                    when=time.strftime("%Y-%m-%d %H:%M")))
                print(f"  {pname} target {Ft:g} N: approach {d * 1e3:.1f} um -> F {m['F']:.4f} N, r_bar {m['arm']:.3f} mm, "
                      f"CG {info['cg_iterations']} in {time.time() - ts:.0f} s", flush=True)
                if abs(m["F"] / Ft - 1) < ftol:
                    break
                near = sorted(pts, key=lambda q: abs(math.log(q[1] / Ft)))[:2]
                if len(near) == 2 and abs(math.log(near[0][1] / near[1][1])) > 1e-6:
                    (d0, f0, _), (d1, f1, _) = near
                    n_loc = min(2.0, max(0.8, math.log(f1 / f0) / math.log(d1 / d0)))
                else:
                    n_loc = 1.2
                d = near[0][0] * (Ft / near[0][1]) ** (1 / n_loc)
        for Ft in FORCES:
            d, rbar = loglog(pts, Ft)
            row = dict(base, kind="force", probe=pname, F_N=Ft, approach_um=1e3 * d, r_bar_mm=rbar,
                       n_solves=len(pts), t_probe_s=round(time.time() - tp, 1), when=time.strftime("%Y-%m-%d %H:%M"))
            for c in REF[body]:
                rr = refs[c].get(pname)
                if rr:
                    v = rr["per_force"][f"{Ft:g}"]
                    row[f"ref_{c}"] = dict(approach_um=v["approach_um"], r_bar_mm=v["r_bar_mm"],
                                           ratio=1e3 * d / v["approach_um"] - 1)
            FE.append(OUT, row)
            r0 = row.get(f"ref_{REF[body][0]}", {})
            print(f"  {pname} {Ft:g} N: resolved {1e3 * d:.1f} um, homogenized {r0.get('approach_um', float('nan')):.1f} um "
                  f"({100 * r0.get('ratio', float('nan')):+.1f} %), r_bar {rbar:.3f} vs {r0.get('r_bar_mm', float('nan')):.3f} mm",
                  flush=True)
    print(f"{body}: {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("voxels")
    v.add_argument("--body", choices=["whole", "b"], required=True)
    g = sub.add_parser("gvoxels", help="b's pad B00 from its G-code (body b00; --match: b00m, widths matched to the weighed TPU)")
    g.add_argument("--match", action="store_true")
    s = sub.add_parser("solve")
    s.add_argument("--body", choices=list(X0), required=True)
    s.add_argument("--probes", nargs="+", default=["cx12.5_z0", "s_apex", "s_y4.1", "cz12.5_y0"])
    s.add_argument("--tol", type=float, default=1e-6)
    a = ap.parse_args()
    if a.cmd == "voxels":
        voxels(a.body)
    elif a.cmd == "gvoxels":
        gcode_voxels(name="b00m", match_mm3=B_WEIGHED_MM3) if a.match else gcode_voxels()
    else:
        solve(a.body, a.probes, tol=a.tol)


if __name__ == "__main__":
    main()
