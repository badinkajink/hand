#!/usr/bin/env python3
"""Does each tip of the fingertip-shape rerun mount on the v4b finger as the 11/22 pad does (ledger step 2)?

The v4b pad (assets/cad_v2, pad 05R/05R4) is the 11/22 half-ellipsoid cut 13.45 mm under its apex, at x_F 2.90: its back
face rests on the cassette 3T and its dovetail grooves take the cassette's rails, the cassette's 3 mm side walls run along
its flanks, the top cover TW15H3 (trimmed at x_F -0.10, four M2 button heads) closes its distal end with an end plate,
3 mm side blocks and a 2.0 mm pin in the pad's 1.6 mm hole, and its mount face rests on the DIP link's column. A tip
mounts as the 11/22 does when its pad, the tip's shape cut at the same plane, keeps the TPU that bears on those parts and
puts no TPU inside them where the 11/22 has none; the grooves, pin hole and notches are then cut as in 05R4.

On one 0.2 mm voxel grid in frame F (assets/cad_v2/README.md), each tip's design surface is placed as the scenes place it
(reorient_backends.replace_tips on the index finger: x_F = -x_t, y_F = y_t, z_F = z_t - 87.91 mm) and compared with the
rigid parts and the 05R pad's CAD mesh:
  extra_mm3[part]     TPU of the tip's pad inside a rigid part where the 11/22's analytic shape has none
  kept[part]          of the 05R pad's surface that bears on a part (within 0.3 mm of it), the share inside the tip's pad
                      (0.15 mm tolerance): the back face and grooves on the cassette, the end and pin hole on the cap, the
                      mount face on the DIP link
  magnets             pad 05R4's four 3 x 2 mm magnets (a 5.0 mm square about the contact point, tops 7.5 mm under the
                      apex): inside the pad, and the least TPU between a magnet and the pad's surface (the print rule is
                      1.8 mm, scripts/cad/metric_magnet_prints.py)
  verdict             'mounts' when no part gains more than 2 mm^3 of TPU and every part keeps 95 % of its bearing surface;
                      otherwise 'envelope alone': the tip is simulated as its shape on the v4b finger's rigid parts, which
                      would need a cassette and cap of its own

Rows: docs/experiments/20261104-fingertip_shape_rerun/mount/mount.json, and mount/sections.npz (masks on the y = 0,
z_t = 0 and z_t = -6 mm planes and on the pad's back plane, for the page's figure).

    ~/.claude/bin/resguard.sh run --mem 3G --cpu 100 -- taskset -c 19 env NUMPY_MADVISE_HUGEPAGE=0 \\
        .venv/bin/python scripts/fingertip/fingertip_shape_rerun_mount.py
"""
from __future__ import annotations

import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("UR5_CAP_TRIM", "-0.10")
os.environ.setdefault("UR5_SCREW_HEADS", "button")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import fingertip_shape_rerun_common as C  # noqa: E402
import tip_codesign_geometry as TG  # noqa: E402

OUT = C.D / "mount"
SCRIPT = "scripts/fingertip/fingertip_shape_rerun_mount.py"
PITCH = 0.2                                  # mm
Z_TIP_F = -51.55 - 36.36                     # the tip body's origin, z_F (build_cad_v2_finger: PIP axis, TIP_BODY_MM)
BEAR_MM, KEEP_TOL_MM = 0.3, 0.15
EXTRA_MAX_MM3, KEEP_MIN = 2.0, 0.95
WALL_RULE_MM = 1.8
MAG_D, MAG_T, MAG_TOP_DEPTH = 3.0, 2.0, 7.5
MAG_YZ = [(sy * 2.5, sz * 2.5) for sz in (1, -1) for sy in (-1, 1)]    # (y_t, z_t), mm: 05R4, proximal row first
BOX_F = ((-11.6, 7.6), (-12.0, 12.0), (-100.0, -74.4))                # mm, the region the tips and their parts occupy


def to_tip(P_F):
    """Points in F (mm) to the tcd tip frame (m)."""
    return np.column_stack([-P_F[:, 0], P_F[:, 1], P_F[:, 2] - Z_TIP_F]) * 1e-3


def grid():
    axes = [np.arange(lo + PITCH / 2, hi, PITCH) for lo, hi in BOX_F]
    shape = tuple(len(a) for a in axes)
    P = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    return axes, shape, P


def filled(mesh, axes):
    """Occupancy of the grid (axes in F, mm) by a watertight mesh, by ray parity: one ray along +x per (y, z) column of the
    grid, its crossings with every facet whose projection on the y-z plane holds the column (barycentric test), and a
    point inside when an odd number of crossings lie before it. Returns a flat mask in grid()'s order. (A filled
    voxelization of the DIP link at 0.1 mm took 4 GB on 11-04; .venv has no rtree for trimesh's ray queries.)"""
    ys, zs = axes[1] + 1.3e-6, axes[2] + 0.7e-6                    # off the facets' edges
    V = np.asarray(mesh.vertices, float)
    T = V[np.asarray(mesh.faces)]                                  # (n, 3 vertices, xyz)
    y0, z0, h = ys[0], zs[0], PITCH
    jy0 = np.clip(np.ceil((T[:, :, 1].min(1) - y0) / h), 0, len(ys)).astype(int)
    jy1 = np.clip(np.floor((T[:, :, 1].max(1) - y0) / h), -1, len(ys) - 1).astype(int)
    jz0 = np.clip(np.ceil((T[:, :, 2].min(1) - z0) / h), 0, len(zs)).astype(int)
    jz1 = np.clip(np.floor((T[:, :, 2].max(1) - z0) / h), -1, len(zs) - 1).astype(int)
    ny, nz = np.maximum(jy1 - jy0 + 1, 0), np.maximum(jz1 - jz0 + 1, 0)
    cnt = ny * nz
    t = np.repeat(np.arange(len(T)), cnt)
    k = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    jy = jy0[t] + k // np.maximum(nz[t], 1)
    jz = jz0[t] + k % np.maximum(nz[t], 1)
    A, B, Cc = T[t, 0], T[t, 1], T[t, 2]
    qy, qz = ys[jy], zs[jz]
    d = (B[:, 1] - A[:, 1]) * (Cc[:, 2] - A[:, 2]) - (Cc[:, 1] - A[:, 1]) * (B[:, 2] - A[:, 2])
    ok = np.abs(d) > 1e-12
    u = ((qy - A[:, 1]) * (Cc[:, 2] - A[:, 2]) - (Cc[:, 1] - A[:, 1]) * (qz - A[:, 2])) / np.where(ok, d, 1.0)
    w = ((B[:, 1] - A[:, 1]) * (qz - A[:, 2]) - (qy - A[:, 1]) * (B[:, 2] - A[:, 2])) / np.where(ok, d, 1.0)
    hit = ok & (u >= 0) & (w >= 0) & (u + w <= 1)
    x = A[hit, 0] + u[hit] * (B[hit, 0] - A[hit, 0]) + w[hit] * (Cc[hit, 0] - A[hit, 0])
    col = jy[hit] * len(zs) + jz[hit]
    order = np.lexsort((x, col))
    col, x = col[order], x[order]
    xs = axes[0]
    occ = np.zeros((len(xs), len(ys) * len(zs)), bool)
    starts = np.searchsorted(col, np.arange(len(ys) * len(zs) + 1))
    for c in np.flatnonzero(np.diff(starts)):
        hc = x[starts[c]:starts[c + 1]]
        occ[:, c] = np.searchsorted(hc, xs) % 2 == 1
    return occ.reshape(len(xs), len(ys), len(zs)).ravel()


def parts(P, axes):
    """{part: occupancy of P}: the DIP link, the cassette 3T, the cap as trimmed with its four heads; and the 05R pad's
    CAD mesh (largest body: the pad without its pockets) with its back plane x_F."""
    import ur5_cb3_cell as CELL
    occ = {}
    occ["dip_link"] = filled(CELL.part_F("dip_link"), axes)
    occ["cassette"] = filled(CELL.part_F("cover"), axes)
    cap = CELL._load(CELL.TOOL_PARTS["cap"][0])                       # as printed (watertight), then the trim
    occ["cap"] = filled(cap, axes) & (P[:, 0] >= CELL.CAP_TRIM_X)
    heads = CELL.heads_mesh_F()
    occ["cap"] |= np.logical_or.reduce([filled(h, axes) for h in heads.split(only_watertight=False)])
    pad = max(CELL.part_F("pad").split(only_watertight=False), key=lambda m: m.volume)
    return occ, pad, float(pad.bounds[1, 0])


def design(key):
    """The tip's design surface (analytic inside test): a printed key's terraces lie within 4.5 um (upright, 0.1 mm) or
    one 0.2 mm layer (the blocks face up) of it, below the grid's pitch."""
    sh = TG.shape(key)
    return sh.design if getattr(sh, "design", None) is not None else sh


def bearing(pad, occ, axes, shape):
    """Surface samples of the 05R pad that bear on each rigid part: within BEAR_MM of the part's voxels (a distance
    transform of the part's occupancy), as {part: (points F mm, area per point mm^2)}."""
    import trimesh
    from scipy import ndimage
    n = int(pad.area / 0.01)                                          # about one sample per 0.01 mm^2
    S, _ = trimesh.sample.sample_surface(pad, n, seed=0)
    a = pad.area / len(S)
    idx = np.clip(np.floor((S - np.array([lo for lo, _ in BOX_F])) / PITCH).astype(int), 0, np.array(shape) - 1)
    out = {}
    for k, o in occ.items():
        dist = ndimage.distance_transform_edt(~o.reshape(shape)) * PITCH
        near = dist[idx[:, 0], idx[:, 1], idx[:, 2]] <= BEAR_MM + PITCH / 2
        out[k] = (S[near], a)
    return out


def magnets_mask(P_t_mm):
    """Pad 05R4's four magnets: cylinders along x_t, tops MAG_TOP_DEPTH under the apex (x_t 10.55 mm)."""
    x0 = TG.REACH * 1e3 - MAG_TOP_DEPTH
    m = np.zeros(len(P_t_mm), bool)
    for y, z in MAG_YZ:
        m |= ((P_t_mm[:, 0] <= x0) & (P_t_mm[:, 0] >= x0 - MAG_T)
              & ((P_t_mm[:, 1] - y) ** 2 + (P_t_mm[:, 2] - z) ** 2 <= (MAG_D / 2) ** 2))
    return m


def main():
    from scipy import ndimage
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    axes, shape, P = grid()
    occ, pad, x_back_F = parts(P, axes)
    print(f"grid {shape} ({len(P) / 1e6:.2f} M points), parts {time.time() - t0:.0f} s; 05R pad back plane x_F "
          f"{x_back_F:.2f} mm, pad volume {pad.volume:.0f} mm^3", flush=True)
    Pt = to_tip(P)
    Pt_mm = Pt * 1e3
    in_pad_region = P[:, 0] <= x_back_F + 1e-9
    v = PITCH ** 3
    ref = design(C.REF).inside(Pt) & in_pad_region
    bear = bearing(pad, occ, axes, shape)
    mag = magnets_mask(Pt_mm)
    rows, sections = {}, {}
    iy0 = int(np.argmin(np.abs(axes[1])))
    iz0 = int(np.argmin(np.abs(axes[2] - Z_TIP_F)))
    izd = int(np.argmin(np.abs(axes[2] - (Z_TIP_F - 6.0))))          # 6 mm distal: through the cap's side blocks
    ix_back = int(np.argmin(np.abs(axes[0] - (x_back_F - PITCH / 2))))
    for k in C.TIPS:
        t1 = time.time()
        sh = design(k)
        ins = sh.inside(Pt) & in_pad_region
        extra = {p: round(float((ins & o & ~ref).sum() * v), 2) for p, o in occ.items()}
        overlap = {p: round(float((ins & o).sum() * v), 2) for p, o in occ.items()}
        kept = {}
        for p, (S, a) in bear.items():
            if not len(S):
                continue
            St = to_tip(S)
            ok = sh.inside(St, tol=KEEP_TOL_MM * 1e-3) & (S[:, 0] <= x_back_F + KEEP_TOL_MM)
            kept[p] = dict(share=round(float(ok.mean()), 4), area_mm2=round(float(len(S) * a), 1))
        # TPU around the magnets: the distance from each magnet voxel to the nearest voxel outside the pad
        body = ins.reshape(shape)
        dist = ndimage.distance_transform_edt(body) * PITCH
        mm = mag.reshape(shape)
        inside_all = bool((mm & ~body).sum() == 0)
        walls = []
        for y, z in MAG_YZ:
            one = magnets_mask(Pt_mm).reshape(shape) & (((Pt_mm[:, 1] - y) ** 2 + (Pt_mm[:, 2] - z) ** 2)
                                                         <= (MAG_D / 2 + 1e-6) ** 2).reshape(shape)
            walls.append(round(float(dist[one].min()) - PITCH / 2, 2) if (one & body).any() else None)
        mounts = max(extra.values()) <= EXTRA_MAX_MM3 and all(q["share"] >= KEEP_MIN for q in kept.values())
        why = [f"{p}: {e:.1f} mm^3 of TPU inside it" for p, e in extra.items() if e > EXTRA_MAX_MM3]
        why += [f"{p}: {100 * (1 - q['share']):.0f} % of its bearing surface lost" for p, q in kept.items()
                if q["share"] < KEEP_MIN]
        mag_fit = inside_all and all(w is not None and w >= WALL_RULE_MM for w in walls)
        rows[k] = dict(label=C.label(k), family=C.family(k), pad_volume_mm3=round(float(ins.sum() * v), 1),
                       tip_volume_mm3=round(float(sh.inside(Pt).sum() * v), 1),
                       overlap_mm3=overlap, extra_mm3=extra, kept=kept, verdict="mounts" if mounts else "envelope alone",
                       reasons=why, magnets_inside=inside_all, magnet_walls_mm=walls,
                       magnets_fit=bool(mag_fit))
        sections[f"{k}|y0"] = ins.reshape(shape)[:, iy0, :]
        sections[f"{k}|z0"] = ins.reshape(shape)[:, :, iz0]
        sections[f"{k}|zd6"] = ins.reshape(shape)[:, :, izd]
        sections[f"{k}|back"] = ins.reshape(shape)[ix_back, :, :]
        print(f"{k:30s} {rows[k]['verdict']:15s} pad {rows[k]['pad_volume_mm3']:7.1f} mm3  extra "
              + " ".join(f"{p} {e:.1f}" for p, e in extra.items()) + "  kept "
              + " ".join(f"{p} {q['share']:.3f}" for p, q in kept.items())
              + f"  magnets inside {inside_all} walls {walls}  ({time.time() - t1:.0f} s)", flush=True)
    for p, o in occ.items():
        sections[f"part:{p}|y0"] = o.reshape(shape)[:, iy0, :]
        sections[f"part:{p}|z0"] = o.reshape(shape)[:, :, iz0]
        sections[f"part:{p}|zd6"] = o.reshape(shape)[:, :, izd]
        sections[f"part:{p}|back"] = o.reshape(shape)[ix_back, :, :]
    sections["magnets|y0"] = mag.reshape(shape)[:, int(np.argmin(np.abs(axes[1] - 2.5))), :]
    sections["magnets|z0"] = mag.reshape(shape)[:, :, int(np.argmin(np.abs(axes[2] - (Z_TIP_F + 2.5))))]
    np.savez_compressed(OUT / "sections.npz", x_F=axes[0], y_F=axes[1], z_F=axes[2], **sections)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    meta = dict(pitch_mm=PITCH, box_F_mm=BOX_F, z_tip_F_mm=Z_TIP_F, pad_back_x_F_mm=round(x_back_F, 3),
                pad_05R_volume_mm3=round(float(pad.volume), 1), rigid_volume_mm3={p: round(float(o.sum() * v), 1)
                                                                                 for p, o in occ.items()},
                bearing_area_mm2={p: round(float(len(S) * a), 1) for p, (S, a) in bear.items()},
                rules=dict(bear_mm=BEAR_MM, keep_tol_mm=KEEP_TOL_MM, extra_max_mm3=EXTRA_MAX_MM3, keep_min=KEEP_MIN,
                           wall_rule_mm=WALL_RULE_MM, magnets="05R4: 3 x 2 mm, (y, z) = (+-2.5, +-2.5) mm, tops 7.5 mm deep"),
                cap_trim_x_F=float(os.environ["UR5_CAP_TRIM"]), wall_s=round(time.time() - t0, 1), peak_rss_GB=round(peak, 2),
                script=SCRIPT, when=time.strftime("%Y-%m-%d %H:%M"))
    (OUT / "mount.json").write_text(json.dumps(dict(meta=meta, tips=rows), indent=1) + "\n")
    print(f"-> {(OUT / 'mount.json').relative_to(ROOT)}; {time.time() - t0:.0f} s, peak RSS {peak:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
