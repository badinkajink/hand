#!/usr/bin/env python3
"""CAD-hand scenes with the DIP-link v2 finger (2026-10-19 night): the readiness study's variants (scripts/
sim2real_fingertip_scenes.py, assets/mjcf/experimental/20261016-sim2real_fingertip_readiness/<variant>/) with concept a's
v2 DIP link, 9.75 mm longer, so that the scale study's cells, pinned iCEM and the first-bench checks run on the finger the
owner prints (docs/experiments/20261018-dip_link_board_mount/v2/).

Per finger, in the pip_frame body (the DIP link; frame F of the DIP study is the body frame with z + 51.55 mm, checked
against the original link STL to 2e-6 mm):
  link      the v2 a DIP link and its clamp plate (v2/a/cad/*_dip_link.stl, *_clamp_plate.stl) as the visual mesh; their
            masses from the P1S G-code of the evening's PLA plate (7.44 g link without supports, 1.96 g clamp plate) spread
            over each part's volume; the board (0.64 g) as before
  spheres   MorphIt-B on a closed part (scripts/cad_v1_spheres.py's settings), else greedy inscribed spheres on a 0.25 mm
            occupancy (the link's STL is 13 touching shells), clipped 0.5 mm behind the tip block's faces
            (cad_v1_spheres.clip_to_pads) at the tip's new place
  tip       the <f>_tip body 9.75 mm further from the PIP axis (pad centre 36.36 mm past the PIP), with its pads, mass and
            magnets unchanged (the readiness variant's)
  board     the board boxes in v2 a's window: FR-4 x 7.3-7.9 mm with the chips to 6.4 mm, centred over the contact point
The thumb's geoms are the index's turned 180 degrees about the vertical, as in the readiness scenes.

    ~/.claude/bin/resguard.sh run --mem 3G --cpu 500 -- .venv/bin/python scripts/v2_finger_scenes.py build \\
        --variants ell11_g30 block5_g30
    .venv/bin/python scripts/v2_finger_scenes.py check --variants ell11_g30
Out: assets/mjcf/experimental/20261019-v2_finger/<variant>_v2a/ (README.json, cad_v1/ with the three XML files, meshes,
assembly.json) and logs/20261019-v2_finger/ (MorphIt work).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

SRC = ROOT / "assets/mjcf/experimental/20261016-sim2real_fingertip_readiness"
OUT = ROOT / "assets/mjcf/experimental/20261019-v2_finger"
WORK = ROOT / "logs/20261019-v2_finger"
V2A = ROOT / "docs/experiments/20261018-dip_link_board_mount/v2/a/cad"
NECK = 0.00975                 # m, v2's lengthening (params neck_len 9.75)
Z_PIP_F = -51.55               # mm, the PIP axis in frame F
MASS_G = {"dip_link": 7.44, "clamp_plate": 1.96}     # P1S G-code of 20261019-v2_ur5_pla_p1s (objects A1, A2), no supports
BOARD_G = 0.64
FINGERS = ("thumb", "index", "middle")
SCRIPT = "scripts/v2_finger_scenes.py"


def part_mesh(name):
    """A v2 a part in the pip_frame body frame, metres."""
    import trimesh
    m = trimesh.load(V2A / f"20261018-dip_link_v2a_{name}.stl", force="mesh")
    V = np.asarray(m.vertices, float).copy()
    V[:, 2] -= Z_PIP_F
    return trimesh.Trimesh(V / 1000.0, np.asarray(m.faces), process=True)


def board_boxes():
    """(name, centre m, half extents m, mass kg) of the board in v2 a's window, index frame."""
    x0, x1 = 6.4e-3, 7.9e-3                               # chip tops to the FR-4's back face (levels chip_top, x_bb)
    zc = -(0.02661 + NECK)
    main = ("board", np.array([(x0 + x1) / 2, 0.0, zc]), np.array([(x1 - x0) / 2, 0.010, 0.010]))
    tab = ("board_tab", np.array([(x0 + x1) / 2, 0.0, zc + 0.0145]), np.array([(x1 - x0) / 2, 0.005, 0.0045]))
    vol = [np.prod(2 * h) for _, _, h in (main, tab)]
    return [(n, c, h, BOARD_G * 1e-3 * v / sum(vol)) for (n, c, h), v in zip((main, tab), vol)]


def inertial(meshes):
    """Mass, centre and 3x3 inertia about the centre (kg, m, kg m^2) of the parts and the board boxes."""
    items = []
    for name, m in meshes.items():
        mass = MASS_G[name] * 1e-3
        mm = m.copy()
        mm.density = mass / mm.volume
        items.append((mass, np.asarray(mm.center_mass), np.asarray(mm.moment_inertia)))
    for _, c, h, mass in board_boxes():
        a, b, d = 2 * h
        I = mass / 12.0 * np.diag([b * b + d * d, a * a + d * d, a * a + b * b])
        items.append((mass, c, I))
    M = sum(i[0] for i in items)
    com = sum(i[0] * i[1] for i in items) / M
    J = np.zeros((3, 3))
    for mass, c, I in items:
        r = c - com
        J += I + mass * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    return M, com, J


def tip_block_bounds():
    """The readiness tip block's bounds in the pip_frame (cad_v1_spheres.tip_block_in_pip_frame), moved by the neck."""
    import cad_v1_spheres as CS
    lo, hi = CS.tip_block_in_pip_frame()
    return lo - np.array([0, 0, NECK]), hi - np.array([0, 0, NECK])


def column_parity(tri, axis, lo, pitch, shape):
    """Occupancy of one closed shell on the grid by ray parity along `axis` through the voxel-column centres."""
    u, v = [k for k in range(3) if k != axis]
    hits = []
    for T in tri:
        P = T[:, [u, v]]
        i0, j0 = np.floor((P.min(0) - lo[[u, v]]) / pitch - 0.5).astype(int)
        i1, j1 = np.ceil((P.max(0) - lo[[u, v]]) / pitch - 0.5).astype(int)
        i0, j0 = max(i0, 0), max(j0, 0)
        i1, j1 = min(i1, shape[u] - 1), min(j1, shape[v] - 1)
        if i1 < i0 or j1 < j0:
            continue
        I, Jj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing="ij")
        cu = lo[u] + (I.ravel() + 0.5) * pitch
        cv = lo[v] + (Jj.ravel() + 0.5) * pitch
        (a, b), (c, d), (e, f) = P
        den = (d - f) * (a - e) + (e - c) * (b - f)
        if abs(den) < 1e-12:
            continue
        l1 = ((d - f) * (cu - e) + (e - c) * (cv - f)) / den
        l2 = ((f - b) * (cu - e) + (a - e) * (cv - f)) / den
        l3 = 1 - l1 - l2
        ok = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
        if ok.any():
            w = l1[ok] * T[0, axis] + l2[ok] * T[1, axis] + l3[ok] * T[2, axis]
            hits.append(np.column_stack([I.ravel()[ok], Jj.ravel()[ok], w]))
    occ = np.zeros(shape, bool)
    if not hits:
        return occ
    H = np.concatenate(hits)
    H = H[np.lexsort((H[:, 2], H[:, 1], H[:, 0]))]
    key = H[:, 0] * shape[v] + H[:, 1]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    ends = np.r_[starts[1:], len(H)]
    wc = lo[axis] + (np.arange(shape[axis]) + 0.5) * pitch
    for s0, s1 in zip(starts, ends):
        if (s1 - s0) % 2:
            continue                                     # a grazing ray: leave the column to the other two axes
        iu, iv = int(H[s0, 0]), int(H[s0, 1])
        for k in range(s0, s1, 2):
            m = (wc >= H[k, 2]) & (wc <= H[k + 1, 2])
            idx = [slice(None)] * 3
            idx[u], idx[v] = iu, iv
            col = occ[tuple(idx)]
            col |= m
            occ[tuple(idx)] = col
    return occ


def occupancy(m, pitch=0.25):
    """Voxel occupancy (bool grid, its corner, pitch; mm) of a mesh of closed shells that may touch or overlap: each shell
    filled by ray parity along x, y and z, a voxel inside when two of the three directions say so, the shells united. The
    grid is bounded by the mesh's box, so memory stays at a few MB (trimesh `voxelized` subdivides long CAD triangles down
    to the pitch and ran a 4 GB job into swap on 2026-10-19)."""
    lo = m.bounds[0] - 2 * pitch
    shape = tuple(int(x) for x in np.ceil((m.bounds[1] + 2 * pitch - lo) / pitch))
    occ = np.zeros(shape, bool)
    for sh in m.split(only_watertight=False):
        tri = np.asarray(sh.triangles, float)
        votes = sum(column_parity(tri, ax, lo, pitch, shape).astype(np.uint8) for ax in range(3))
        occ |= votes >= 2
    return occ, lo, pitch


def greedy_spheres(occ, lo, pitch, budget=40, r_min=0.5, cover=0.92):
    """Inscribed spheres (mm) covering the occupancy: repeatedly the uncovered voxel deepest inside the part, at the depth's
    radius, until `budget` spheres, the radius falls under r_min or `cover` of the voxels are covered."""
    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(occ) * pitch
    covered = np.zeros_like(occ)
    n = int(occ.sum())
    C, R = [], []
    grid = np.indices(occ.shape)
    while len(R) < budget and covered.sum() < cover * n:
        cand = np.where(occ & ~covered, dist, 0.0)
        k = np.unravel_index(int(np.argmax(cand)), occ.shape)
        r = float(dist[k])
        if r < r_min:
            break
        c = lo + (np.array(k) + 0.5) * pitch
        rr = int(np.ceil(r / pitch)) + 1
        sl = tuple(slice(max(k[a] - rr, 0), min(k[a] + rr + 1, occ.shape[a])) for a in range(3))
        P = np.stack([lo[a] + (grid[a][sl] + 0.5) * pitch for a in range(3)], -1)
        covered[sl] |= np.linalg.norm(P - c, axis=-1) <= r
        C.append(c)
        R.append(r)
    return np.array(C), np.array(R), float(covered.sum() / max(n, 1))


def pack(mesh, name, budget):
    """Collision spheres of a part (metres, body frame): MorphIt-B as scripts/cad_v1_spheres.py packs the CAD parts when
    the STL is one closed shell, else greedy inscribed spheres on its voxel occupancy (the v2 link's STL is 13 touching
    shells, which MorphIt's watertight check refuses)."""
    m = mesh.copy()
    m.apply_scale(1000.0)
    if m.is_watertight:
        C, R = morphit(mesh, name, budget)
        return C, R, dict(method="MorphIt-B", budget=budget)
    occ, lo, pitch = occupancy(m)
    C, R, cov = greedy_spheres(occ, lo, pitch, budget=budget)
    vol = float(occ.sum()) * pitch ** 3
    print(f"{name}: occupancy {vol:.0f} mm3 against the mesh's {m.volume:.0f}; {len(R)} inscribed spheres cover {cov:.2f}", flush=True)
    return C / 1000.0, R / 1000.0, dict(method="greedy inscribed spheres on a 0.25 mm occupancy", budget=budget,
                                        occupancy_mm3=round(vol, 1), mesh_mm3=round(float(m.volume), 1), coverage=round(cov, 3))


def morphit(mesh, name, budget=20):
    """MorphIt-B spheres of a part (metres, body frame), as scripts/cad_v1_spheres.py packs the CAD parts."""
    import cad_v1_spheres as CS
    WORK.mkdir(parents=True, exist_ok=True)
    m = mesh.copy()
    m.apply_scale(1000.0)
    stl = WORK / f"{name}.stl"
    m.export(stl)
    js = WORK / "spheres" / f"{name}_MorphIt-B_n{budget}.json"
    if not js.exists():
        CS.run([CS.PY, CS.MORPHIT / "hand_trials/pack_link.py", "--mesh", stl, "--budgets", budget, "--variant", "MorphIt-B",
                "--iterations", 1000, "--seed", 0, "--out", WORK, "--threads", 4], WORK / f"morphit_{name}.log")
    S = json.loads(js.read_text())
    C = np.asarray(S["centers"] if "centers" in S else S["centres"], float) / 1000.0
    R = np.asarray(S["radii"], float).ravel() / 1000.0
    return C, R


def fmt(v, nd=6):
    return " ".join(f"{x:.{nd}f}" for x in np.asarray(v, float).ravel())


def patch(path, spheres, M, com, J):
    """Rewrite one XML file of a variant for the v2 finger. Returns the per-finger record."""
    root = ET.parse(path).getroot()
    rec = {}
    for f in FINGERS:
        b = root.find(f".//body[@name='{f}_pip_frame']")
        if b is None:
            continue
        s = np.array([-1.0, -1.0, 1.0]) if f == "thumb" else np.ones(3)      # 180 degrees about the vertical
        Rm = np.diag(s)
        cls = None
        for g in list(b.findall("geom")):
            n = g.get("name") or ""
            if n.startswith(f"{f}_pip_frame_s"):
                cls = cls or dict(g.attrib)
                b.remove(g)
            elif n in (f"{f}_board", f"{f}_board_tab"):
                b.remove(g)
        vis = b.find(f"geom[@name='{f}_dip_link_visual']")
        k0 = list(b).index(vis) + 1 if vis is not None else len(b)
        new = []
        for i, (c, r) in enumerate(zip(*spheres)):
            a = {k: v for k, v in (cls or {}).items() if k not in ("name", "pos", "size")}
            new.append(ET.Element("geom", dict(name=f"{f}_pip_frame_s{i}", pos=fmt(s * c), size=f"{r:.6f}", **a)))
        for n, c, h, _ in board_boxes():
            new.append(ET.Element("geom", {"name": f"{f}_{n}", "type": "box", "size": fmt(h), "pos": fmt(s * c), "mass": "0"}))
        for k, e in enumerate(new):
            b.insert(k0 + k, e)
        Jf = Rm @ J @ Rm.T
        ine = b.find("inertial")
        ine.set("pos", fmt(s * com))
        ine.set("mass", f"{M:.6f}")
        ine.set("fullinertia", " ".join(f"{x:.6g}" for x in (Jf[0, 0], Jf[1, 1], Jf[2, 2], Jf[0, 1], Jf[0, 2], Jf[1, 2])))
        t = b.find(f"body[@name='{f}_tip']")
        p = np.array([float(x) for x in t.get("pos").split()])
        t.set("pos", fmt(p - np.array([0, 0, NECK])))
        rec[f] = dict(tip_pos_m=(p - np.array([0, 0, NECK])).round(6).tolist(), spheres=len(spheres[1]))
    tmp = path.with_suffix(".tmp")
    ET.ElementTree(root).write(tmp)
    tmp.replace(path)
    return rec


def cmd_build(a):
    import trimesh
    import cad_v1_spheres as CS
    t0 = time.time()
    meshes = {n: part_mesh(n) for n in ("dip_link", "clamp_plate")}
    vis = trimesh.util.concatenate(list(meshes.values()))
    lo, hi = tip_block_bounds()
    C, R, how = [], [], {}
    for n, m in meshes.items():
        c, r, info = pack(m, n, {"dip_link": 40, "clamp_plate": 12}[n])
        C.append(c)
        R.append(r)
        how[n] = info
    C, R = np.concatenate(C), np.concatenate(R)
    past0 = CS.past_pad_faces(C, R, lo, hi)
    C, R, n_clip = CS.clip_to_pads(C, R, lo, hi)
    past1 = CS.past_pad_faces(C, R, lo, hi)
    M, com, J = inertial(meshes)
    print(f"v2 a DIP body: {M * 1e3:.2f} g at {np.round(com * 1e3, 2)} mm; {len(R)} spheres ({n_clip} clipped); "
          f"past pad faces before {past0}, after {past1}", flush=True)
    for v in a.variants:
        dst = OUT / f"{v}_v2a"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(SRC / v, dst)
        vis.export(dst / "cad_v1" / "meshes" / "dip_link.stl")
        rec = {}
        for fn in ("scene_screwdriver_medium.xml", "hand.xml", "finger.xml"):
            p = dst / "cad_v1" / fn
            if p.exists():
                rec[fn] = patch(p, (C, R), M, com, J)
        asm = json.loads((dst / "cad_v1" / "assembly.json").read_text())
        if "body_origins_mm" in asm and "tip" in asm["body_origins_mm"]:
            o = asm["body_origins_mm"]["tip"]
            asm["body_origins_mm"]["tip"] = [o[0], o[1], o[2] - NECK * 1e3]
        asm["v2_finger"] = dict(neck_mm=NECK * 1e3, script=SCRIPT)
        (dst / "cad_v1" / "assembly.json").write_text(json.dumps(asm, indent=1) + "\n")
        readme = json.loads((dst / "README.json").read_text())
        readme.update(variant=f"{v}_v2a", base_variant=v, neck_mm=NECK * 1e3, pip_to_pad_centre_mm=26.61 + NECK * 1e3,
                      dip_body=dict(mass_g=round(M * 1e3, 3), com_mm=np.round(com * 1e3, 3).tolist(),
                                    parts_g=dict(MASS_G, board=BOARD_G), source="v2/a/cad/20261018-dip_link_v2a_{dip_link,"
                                    "clamp_plate}.stl; masses from the P1S G-code of 20261019-v2_ur5_pla_p1s"),
                      spheres=dict(n=int(len(R)), clipped=n_clip, past_pad_faces_mm=dict(fitted=past0, clipped=past1), packing=how),
                      patched=rec, script=f"{SCRIPT} build", when=time.strftime("%Y-%m-%d %H:%M"))
        (dst / "README.json").write_text(json.dumps(readme, indent=1, default=float) + "\n")
        print(f"-> {dst.relative_to(ROOT)}", flush=True)
    print(f"[{time.time() - t0:.0f} s]", flush=True)


def cmd_check(a):
    """Load each scene, compare the tip bodies' world positions at the keyframe-free pose with the readiness variant's
    (they must sit 9.75 mm further along each DIP link) and print the DIP body masses."""
    import mujoco
    for v in a.variants:
        out = {}
        for tag, d in (("readiness", SRC / v), ("v2", OUT / f"{v}_v2a")):
            m = mujoco.MjModel.from_xml_path(str(d / "cad_v1" / "scene_screwdriver_medium.xml"))
            dd = mujoco.MjData(m)
            mujoco.mj_forward(m, dd)
            out[tag] = {f: (dd.xpos[m.body(f"{f}_tip").id].copy(), dd.xpos[m.body(f"{f}_pip_frame").id].copy(),
                            float(m.body_mass[m.body(f"{f}_pip_frame").id]), int(m.nbody), int(m.ngeom)) for f in FINGERS}
        for f in FINGERS:
            (t0, p0, m0, *_), (t1, p1, m1, nb, ng) = out["readiness"][f], out["v2"][f]
            print(f"{v} {f}: tip {np.round(t0 * 1e3, 2)} -> {np.round(t1 * 1e3, 2)} mm, |PIP->tip| "
                  f"{np.linalg.norm(t0 - p0) * 1e3:.2f} -> {np.linalg.norm(t1 - p1) * 1e3:.2f} mm, DIP body "
                  f"{m0 * 1e3:.2f} -> {m1 * 1e3:.2f} g; model {nb} bodies {ng} geoms", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "check"):
        p = sub.add_parser(name)
        p.add_argument("--variants", nargs="+", default=["ell11_g30", "block5_g30"])
    a = ap.parse_args()
    {"build": cmd_build, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
