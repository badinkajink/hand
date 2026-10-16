#!/usr/bin/env python3
"""CAD-hand scenes of the printable 85A fingertips (docs/handoff/20261016-sim2real_fingertip_readiness.md, step 3).

Each variant is the CAD hand (scripts/build_cad_v1_finger.py: the SR2 finger from its part STLs on the real_v1 mounts,
MorphIt link spheres, the servo plant applied by the studies that load it) with one printed fingertip:

  shape       the tip's outer surface as printed (tip_codesign_geometry key): the block tcd_rs3_re3_p0.2 for the four,
              five and one-wall magnet blocks; tcd_cap8.5_u0.1, tcd_ell7.4x8.5x16x32_u0.1, tcd_ell7.4x8.5x11x22_u0.1
  pads        1 mm sphere pads, builder v2, at the pad modulus E of the variant: the 85A finite elements of the magnet
              block (scripts/sim2real_fingertip_fe.py, solid 12 MPa) for the blocks; for the curved tips the same value
              times the curved tip's tool-fitted pad modulus over the block's (2026-10-15 stiffness map, builder v2)
  mass        the printed TPU from the 85A G-code (extrusion on the part, Siraya's 1.26 g/cm^3) spread over the tip's
              volume, plus the 3 x 2 mm N52 magnets (0.106 g each, nominal) at their pocket centres
  board       the five-chip magnetometer board (ReSkin 5X: FR-4 20 x 20 x 0.6 mm, five 3 x 3 x 1 mm MLX90393, a 10 x 9 mm
              connector tab) on the dorsal face of the DIP link's 3.0 mm plate, chips toward the plate, centred over the
              contact point, tab toward the PIP joint; a collision box on the DIP link, its mass added to that link

The DIP link's dorsal plate (x 6.45-9.45 mm in the finger frame, 14.8 mm wide, the whole tip's length) fills the space
behind the tip's back face; the readout model of 2026-10-13/16 put the chips against that back face.

  build   assets/mjcf/experimental/20261016-sim2real_fingertip_readiness/<variant>/cad_v1/ (hand.xml, finger.xml,
          scene_screwdriver_medium.xml) and <variant>/README.json
  sweep   interference through travel of the board, the tip and the DIP link against every CAD part of the same finger
          (yaw x MCP x PIP grid) -> docs/experiments/20261016-sim2real_fingertip_readiness/scenes/interference.json
  table   the variants' modulus and masses

    ~/.claude/bin/resguard.sh run --mem 3G --cpu 200 -- .venv/bin/python scripts/sim2real_fingertip_scenes.py build
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

DATE = "20261016"
D = ROOT / "docs/experiments/20261016-sim2real_fingertip_readiness"
ASSETS = ROOT / "assets/mjcf/experimental/20261016-sim2real_fingertip_readiness"
PRINT = D / "print"
FE_ROWS = D / "fe/fem.jsonl"
CURVED_STIFF = ROOT / "docs/experiments/20261015-curved_tips/stiffness/summary.json"
FINGERS = ("thumb", "index", "middle")
TPU_DENSITY = 1.26e-3            # g/mm^3, Siraya's profile
MAGNET_G = math.pi * 1.5 ** 2 * 2.0 * 7.5e-3       # 3 x 2 mm sintered NdFeB, nominal
E_S_PROXY = 12.0                 # MPa, NinjaFlex 85A
REACH, END, LEN = 10.55, 10.55, 21.41              # mm: finger axis to palmar face, tip origin to end face, tip length
# the board, in the finger frame of the index (palmar -x): the DIP plate's dorsal face at x = 9.45 mm; U1 over the
# contact point (z_F = -78.16 mm, the tip body origin); pip frame origin at z_F = -51.55 mm
PLATE_X = 9.45
BOARD = dict(t_mm=1.6, half_mm=10.0, tab_w_mm=10.0, tab_l_mm=9.0, fr4_t_mm=0.6, chip_mm=(3.0, 3.0, 1.0), pitch_mm=7.0,
             mass_g=0.44 + 5 * 0.02 + 0.10)
TIP_Z_IN_PIP = -26.61            # mm, tip body origin in the pip frame


def shapes():
    return {"block": "tcd_rs3_re3_p0.2", "cap": "tcd_cap8.5_u0.1", "ell16": "tcd_ell7.4x8.5x16x32_u0.1",
            "ell11": "tcd_ell7.4x8.5x11x22_u0.1"}


def fe_rows():
    out = {}
    if FE_ROWS.exists():
        for l in open(FE_ROWS):
            r = json.loads(l)
            out[(r["layout"], r["recipe"])] = r
    return out


def curved_ratio():
    """Tool-fitted pad modulus of each curved tip over the block's (builder v2, 2026-10-15). The 11/22 half-ellipsoid had
    no finite elements; it takes the mean of the capsule's and the 16/32's."""
    s = json.loads(CURVED_STIFF.read_text())["tips"]
    blk = s["tcd_rs3_re3"]["models"]["v2"]["modulus_scale_cyl0"]
    cap = s["tcd_cap8.5"]["models"]["v2"]["modulus_scale_cyl0"] / blk
    e16 = s["tcd_ell7.4x8.5x16x32"]["models"]["v2"]["modulus_scale_cyl0"] / blk
    return {"cap": (cap, "finite elements, 2026-10-15"), "ell16": (e16, "finite elements, 2026-10-15"),
            "ell11": (0.5 * (cap + e16), "mean of the capsule and the 16/32 half-ellipsoid (no finite elements)")}


def variants():
    """name -> spec. The baseline is the modelling default: the as-built block at 10 MPa, solid TPU, no magnets or board."""
    fe = fe_rows()
    ratio = curved_ratio()
    sh = shapes()
    out = {"block_e10": dict(shape="block", key=sh["block"], E_MPa=10.0, E_src="the modelling default (10 MPa)",
                             sheet=None, layout=None, walls=2, infill=None, board=False, magnets=0)}
    for inf in (30, 50):
        rec = f"w85_{inf}"
        for lay, walls, tag in (("four", 2, "block4"), ("five", 2, "block5"), ("four", 1, "block4w1")):
            name = f"four_d3x2_d5" if lay == "four" else "five_d3x2_d5"
            name += "_w1" if walls == 1 else ""
            r = fe.get((name, rec))
            sheet = f"{DATE}-siraya85a_block_{lay}_d3x2_depth5_seat2p7" + ("_pocketwalls1" if walls == 1 else "") + f"_g{inf}"
            out[f"{tag}_g{inf}"] = dict(shape="block", key=sh["block"], E_MPa=(r["E_best_MPa"] if r else None),
                                        E_src=f"finite elements {name} {rec} (solid {E_S_PROXY:g} MPa)", sheet=sheet,
                                        layout=lay, walls=walls, infill=inf, board=True,
                                        magnets=5 if lay == "five" else 4)
        b4 = fe.get(("four_d3x2_d5", rec))
        for s_, tipname, depth in (("cap", "tcd_cap8p5", "5"), ("ell16", "tcd_ell7p4x8p5x16x32", "5p5"),
                                   ("ell11", "tcd_ell7p4x8p5x11x22", "7")):
            k, src = ratio[s_]
            out[f"{s_}_g{inf}"] = dict(shape=s_, key=sh[s_], E_MPa=(b4["E_best_MPa"] * k if b4 else None),
                                       E_src=f"block four {rec} x {k:.3f} ({src})",
                                       sheet=f"{DATE}-siraya85a_{tipname}_four_d3x2_depth{depth}_seat2p7_g{inf}",
                                       layout="four", walls=2, infill=inf, board=True, magnets=4)
    return out


def part_filament_g(gc):
    """Filament mass extruded on the part (not the skirt, start or end code), g."""
    area = math.pi * 0.875 ** 2
    typ, e_sum = "", 0.0
    for line in open(gc):
        if line.startswith(";TYPE:"):
            typ = line[6:].strip()
            continue
        if line[:3] in ("G1 ", "G2 ", "G3 ") and " E" in line and typ not in ("Custom", "Skirt/Brim", "Skirt", "Wipe tower"):
            s = line.split(";", 1)[0]
            if "X" not in s and "Y" not in s:
                continue
            for tok in s.split():
                if tok[0] == "E":
                    e = float(tok[1:])
                    if e > 0:
                        e_sum += e
    return e_sum * area * TPU_DENSITY


def magnet_centres(spec, sheet):
    """Magnet centres in the index tip body frame (m; palmar face toward -x)."""
    if not sheet:
        return []
    depth = sheet["magnet"]["top_depth_mm"]
    out = []
    for p in sheet["pockets"]:
        x = -(REACH - (depth + 1.0))
        z = -END + (LEN - p["z_mm"])
        out.append([x * 1e-3, p["y_mm"] * 1e-3, z * 1e-3])
    return out


def tip_mass_props(spec, sheet):
    """Mass (kg), centre of mass and full inertia (about the centre, index tip body frame) of the printed tip."""
    import tip_codesign_geometry as TG
    import trimesh
    V, F = TG.shape(spec["key"]).hull(0.0005, sign=-1.0)
    m = trimesh.Trimesh(V, F, process=True)
    vol_mm3 = float(m.volume) * 1e9
    if sheet:
        gc = ROOT / sheet["gcode"]
        tpu_g = part_filament_g(gc)
    else:
        tpu_g = vol_mm3 * 1.21e-3                         # solid TPU at 1210 kg/m^3, as the CAD model
    m.density = tpu_g * 1e-3 / float(m.volume)
    I = np.array(m.moment_inertia)                        # about the mesh's centre of mass
    com = np.array(m.center_mass)
    M = tpu_g * 1e-3
    pts = magnet_centres(spec, sheet)
    mm = MAGNET_G * 1e-3
    tot = M + mm * len(pts)
    c = (M * com + mm * np.sum(pts, axis=0)) / tot if pts else com
    J = I + M * (np.dot(com - c, com - c) * np.eye(3) - np.outer(com - c, com - c))
    for p in pts:
        d = np.asarray(p) - c
        r, h = 1.5e-3, 2.0e-3                            # the magnet as a solid cylinder, axis along x
        Ic = np.diag([0.5 * mm * r * r, mm * (3 * r * r + h * h) / 12, mm * (3 * r * r + h * h) / 12])
        J += Ic + mm * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return dict(tpu_g=round(tpu_g, 3), magnets_g=round(MAGNET_G * len(pts), 3), mass_g=round(tot * 1e3, 3),
                volume_mm3=round(vol_mm3, 1), com_m=c.tolist(), inertia=J, magnet_centres_m=pts)


def fullinertia(J):
    return [J[0, 0], J[1, 1], J[2, 2], J[0, 1], J[0, 2], J[1, 2]]


def board_boxes(sign):
    """(name, pos, half-size, mass share) of the board and its tab in the pip frame (m); sign = the finger's palmar sign."""
    t = BOARD["t_mm"]
    x = -sign * (PLATE_X + t / 2)                         # dorsal side is -sign
    z0 = TIP_Z_IN_PIP                                     # U1 over the contact point
    hb = BOARD["half_mm"]
    tab_z = z0 + hb + BOARD["tab_l_mm"] / 2
    return [("board", (x * 1e-3, 0.0, z0 * 1e-3), (t / 2e3, hb / 1e3, hb / 1e3), 0.85),
            ("board_tab", (x * 1e-3, 0.0, tab_z * 1e-3), (t / 2e3, BOARD["tab_w_mm"] / 2e3, BOARD["tab_l_mm"] / 2e3), 0.15)]


def add_mass(inertial, items):
    """Add point-like box masses [(pos m, half-size m, mass kg)] to an <inertial> element (pos, mass, fullinertia)."""
    m0 = float(inertial.get("mass"))
    c0 = np.array([float(v) for v in inertial.get("pos").split()])
    if inertial.get("fullinertia"):
        f = [float(v) for v in inertial.get("fullinertia").split()]
        I0 = np.array([[f[0], f[3], f[4]], [f[3], f[1], f[5]], [f[4], f[5], f[2]]])
    else:
        I0 = np.diag([float(v) for v in inertial.get("diaginertia").split()])
    tot = m0 + sum(m for _, _, m in items)
    c = (m0 * c0 + sum(m * np.asarray(p) for p, _, m in items)) / tot
    d = c0 - c
    J = I0 + m0 * (d @ d * np.eye(3) - np.outer(d, d))
    for p, h, m in items:
        a, b, cc = (2 * np.asarray(h)) ** 2
        Ib = np.diag([m * (b + cc) / 12, m * (a + cc) / 12, m * (a + b) / 12])
        dd = np.asarray(p) - c
        J += Ib + m * (dd @ dd * np.eye(3) - np.outer(dd, dd))
    inertial.set("pos", " ".join(f"{v:.6f}" for v in c))
    inertial.set("mass", f"{tot:.6f}")
    inertial.attrib.pop("diaginertia", None)
    inertial.set("fullinertia", " ".join(f"{v:.6g}" for v in fullinertia(J)))
    return tot


def patch_scene(path, spec, mp):
    """Tip inertial, magnet visuals and the board on each finger of one written scene file."""
    import fingertip_geometry as G
    root = ET.parse(path).getroot()
    for f in FINGERS:
        s = G.FACE_SIGN[f]
        tipb = root.find(f".//body[@name='{f}_tip']")
        if tipb is None:
            continue
        flip = np.array([-s, 1.0, 1.0])                   # mass props were computed for the index (s = -1)
        Rf = np.diag(flip)
        J = Rf @ mp["inertia"] @ Rf
        inert = tipb.find("inertial")
        if inert is None:
            inert = ET.Element("inertial")
            tipb.insert(0, inert)
        c = np.asarray(mp["com_m"]) * flip
        inert.attrib.clear()
        inert.set("pos", " ".join(f"{v:.6f}" for v in c))
        inert.set("mass", f"{mp['mass_g'] * 1e-3:.6f}")
        inert.set("fullinertia", " ".join(f"{v:.6g}" for v in fullinertia(J)))
        for k, p in enumerate(mp["magnet_centres_m"]):
            q = np.asarray(p) * flip
            ET.SubElement(tipb, "geom", name=f"{f}_magnet{k}", type="cylinder", size="0.0015 0.001",
                          pos=" ".join(f"{v:.6f}" for v in q), zaxis="1 0 0", contype="0", conaffinity="0", group="3",
                          rgba="0.55 0.55 0.6 1", mass="0")
        if spec["board"]:
            pipb = root.find(f".//body[@name='{f}_pip_frame']")
            items = []
            for name, pos, half, share in board_boxes(s):
                ET.SubElement(pipb, "geom", name=f"{f}_{name}", type="box", size=" ".join(f"{v:.6f}" for v in half),
                              pos=" ".join(f"{v:.6f}" for v in pos), rgba="0.1 0.45 0.2 1", group="1", mass="0",
                              friction="1 0.005 0.0001")
                items.append((pos, half, BOARD["mass_g"] * share * 1e-3))
            add_mass(pipb.find("inertial"), items)
    ET.indent(root, "  ")
    Path(path).write_text(ET.tostring(root, encoding="unicode") + "\n")


def cmd_build(a):
    import argparse as AP
    import build_cad_v1_finger as BC
    import pad_usability_backends as PUB
    import reorient_backends as RB
    import tip_codesign_rig as R
    V = variants()
    todo = a.variants or list(V)
    for name in todo:
        spec = V[name]
        if spec["E_MPa"] is None:
            print(f"{name}: no finite-element row yet", flush=True)
            continue
        sheet = json.loads((PRINT / f"{spec['sheet']}.json").read_text()) if spec["sheet"] else None
        mp = tip_mass_props(spec, sheet)
        out = ASSETS / name
        ns = AP.Namespace(E=spec["E_MPa"] * 1e6, pf=None, tr=RB.PAD_TR, tag=None, h="local", K="each")
        R.install(ns, force_tip=spec["key"])
        PUB.SCN = ROOT / "logs/20261016-sim2real_fingertip_readiness/scenes_build" / name
        argv = sys.argv
        sys.argv = ["build_cad_v1_finger.py", "--out", str(out / "cad_v1"), "--spheres",
                    str(ROOT / "assets/mjcf/experimental/20261013-cad_v1/spheres.json")]
        try:
            BC.main()
        finally:
            sys.argv = argv
        for fn in ("hand.xml", "finger.xml", "scene_screwdriver_medium.xml"):
            patch_scene(out / "cad_v1" / fn, spec, mp)
        rec = dict(variant=name, **{k: v for k, v in spec.items()}, pad_builder="v2", pad_E_Pa=spec["E_MPa"] * 1e6,
                   tip_mass=dict(tpu_g=mp["tpu_g"], magnets_g=mp["magnets_g"], total_g=mp["mass_g"],
                                 volume_mm3=mp["volume_mm3"], com_m=np.round(mp["com_m"], 6).tolist(),
                                 fullinertia=[float(f"{v:.4e}") for v in fullinertia(mp["inertia"])]),
                   magnet_centres_index_tip_m=np.round(mp["magnet_centres_m"], 6).tolist(),
                   board_geometry=dict(BOARD, placement="dorsal face of the DIP plate, chips toward the plate, U1 over the contact "
                                               "point, tab toward the PIP joint") if spec["board"] else None,
                   files=["cad_v1/hand.xml", "cad_v1/finger.xml", "cad_v1/scene_screwdriver_medium.xml"],
                   script="scripts/sim2real_fingertip_scenes.py build", when=time.strftime("%Y-%m-%d %H:%M"))
        (out / "README.json").write_text(json.dumps(rec, indent=1, default=float))
        print(f"{name}: E {spec['E_MPa']:.2f} MPa, tip {mp['tpu_g']:.2f} g TPU + {mp['magnets_g']:.2f} g magnets = "
              f"{mp['mass_g']:.2f} g -> {out.relative_to(ROOT)}", flush=True)


def cmd_table(a):
    V = variants()
    for name, s in V.items():
        print(f"{name:12s} {s['key']:28s} E {s['E_MPa'] if s['E_MPa'] is None else round(s['E_MPa'], 2)} MPa  {s['E_src']}")


# ------------------------------------------------------------------------------------------------- interference sweep

def sample_mesh_points(m, gid, n_per_mm2=4.0, cap=60000, seed=0):
    """Area-weighted surface points of a mesh geom in the geom frame."""
    mid = m.geom_dataid[gid]
    va, nv = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    fa, nf = m.mesh_faceadr[mid], m.mesh_facenum[mid]
    V = m.mesh_vert[va:va + nv].astype(float)
    F = m.mesh_face[fa:fa + nf]
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    A = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    n = int(min(cap, max(2000, A.sum() * 1e6 * n_per_mm2)))
    rng = np.random.default_rng(seed)
    k = rng.choice(len(F), n, p=A / A.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    return a[k] + u[:, None] * (b[k] - a[k]) + v[:, None] * (c[k] - a[k])


def box_sd(P, pos, R, half):
    """Signed distance of world points P to a box (negative inside)."""
    q = np.abs((P - pos) @ R) - half
    out = np.linalg.norm(np.maximum(q, 0), axis=1)
    ins = np.minimum(q.max(axis=1), 0)
    return out + ins


def cmd_sweep(a):
    """Board and tab against every CAD part of the same finger through the joint ranges, and the tip's own mesh against
    the MCP link and PIP servo; one variant per tip shape (the board is the same on all)."""
    import mujoco
    res = {}
    for name in a.variants:
      for tab_l in a.tab_lengths:
        xml = ASSETS / name / "cad_v1" / "hand.xml"
        m = mujoco.MjModel.from_xml_path(str(xml))
        d = mujoco.MjData(m)
        rows = {}
        for f in a.fingers:
            if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{f}_board_tab") >= 0:
                tg, bg0 = m.geom(f"{f}_board_tab").id, m.geom(f"{f}_board").id
                z0 = m.geom_pos[bg0][2] + m.geom_size[bg0][2]          # the board's proximal edge, pip frame
                m.geom_size[tg][2] = max(tab_l, 1e-4) / 2e3
                m.geom_pos[tg][2] = z0 + max(tab_l, 1e-4) / 2e3
            parts = {}
            for g in range(m.ngeom):
                nm = m.geom(g).name
                if nm.startswith(f + "_") and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and nm.endswith("_visual"):
                    parts[nm[len(f) + 1:-7]] = (g, sample_mesh_points(m, g))
            boxes = [m.geom(f"{f}_board").id] + ([m.geom(f"{f}_board_tab").id] if tab_l > 0 else []) if mujoco.mj_name2id(
                m, mujoco.mjtObj.mjOBJ_GEOM, f"{f}_board") >= 0 else []
            jid = {j: m.joint(f"{f}_{j}").id for j in ("yaw", "mcp", "pip")}
            rng = {j: m.jnt_range[jid[j]] for j in jid}
            grid = dict(yaw=np.radians(a.yaw), mcp=np.arange(rng["mcp"][0], rng["mcp"][1] + 1e-9, np.radians(a.step)),
                        pip=np.arange(rng["pip"][0], rng["pip"][1] + 1e-9, np.radians(a.step)))
            worst = {}
            for qy in grid["yaw"]:
                for qm in grid["mcp"]:
                    for qp in grid["pip"]:
                        d.qpos[:] = m.qpos0
                        d.qpos[m.jnt_qposadr[jid["yaw"]]] = qy
                        d.qpos[m.jnt_qposadr[jid["mcp"]]] = qm
                        d.qpos[m.jnt_qposadr[jid["pip"]]] = qp
                        mujoco.mj_kinematics(m, d)
                        for part, (g, P) in parts.items():
                            if part in ("dip_link",):
                                continue
                            W = P @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
                            for bg in boxes:
                                sd = box_sd(W, d.geom_xpos[bg], d.geom_xmat[bg].reshape(3, 3), m.geom_size[bg])
                                v = float(sd.min())
                                key = f"{m.geom(bg).name[len(f) + 1:]}|{part}"
                                if key not in worst or v < worst[key][0]:
                                    worst[key] = (v, [round(math.degrees(x), 1) for x in (qy, qm, qp)])
            # the board against the DIP link it sits on, at one pose (fixed to the same body)
            g_dip, P_dip = parts["dip_link"]
            mujoco.mj_kinematics(m, d)
            W = P_dip @ d.geom_xmat[g_dip].reshape(3, 3).T + d.geom_xpos[g_dip]
            on_dip = {m.geom(bg).name[len(f) + 1:]: round(1e3 * float(box_sd(W, d.geom_xpos[bg], d.geom_xmat[bg].reshape(3, 3),
                                                                          m.geom_size[bg]).min()), 3) for bg in boxes}
            rows[f] = dict(min_clearance_mm={k: round(1e3 * v[0], 3) for k, v in sorted(worst.items())},
                           at_deg_yaw_mcp_pip={k: v[1] for k, v in sorted(worst.items())}, board_to_dip_link_mm=on_dip,
                           configs=len(grid["yaw"]) * len(grid["mcp"]) * len(grid["pip"]))
            print(name, f"tab {tab_l:g} mm", f, {k: round(1e3 * v[0], 2) for k, v in sorted(worst.items())}, on_dip, flush=True)
        res[f"{name}|tab{tab_l:g}"] = rows
    out = D / "scenes" / "interference.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(out.read_text()) if out.exists() else {}
    old.update(res)
    old["_terms"] = ("min_clearance_mm: least signed distance from the board (or its tab) to the sampled surface of a CAD "
                     "part of the same finger over the yaw x MCP x PIP grid (negative: the part's surface enters the box); "
                     f"grid yaw {a.yaw} deg, MCP and PIP every {a.step} deg over their ranges")
    out.write_text(json.dumps(old, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--variants", nargs="*", default=[])
    sub.add_parser("table")
    s = sub.add_parser("sweep")
    s.add_argument("--variants", nargs="+", default=["block4_g30", "cap_g30", "ell16_g30", "ell11_g30"])
    s.add_argument("--fingers", nargs="+", default=["index", "thumb"])
    s.add_argument("--yaw", nargs="+", type=float, default=[-85, -60, -30, 0, 30, 60, 85])
    s.add_argument("--step", type=float, default=4.0)
    s.add_argument("--tab-lengths", nargs="+", type=float, default=[9.0, 6.0, 3.0, 0.0],
                   help="connector tab length toward the PIP joint, mm (0: no tab, a cable soldered to the board)")
    a = ap.parse_args()
    {"build": cmd_build, "table": cmd_table, "sweep": cmd_sweep}[a.cmd](a)


if __name__ == "__main__":
    main()
