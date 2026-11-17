"""The press scenes with the real cylinder (autopilot hand-20261116-press_sim_real, 2026-11-17, step 2; owner, 2026-11-16
23:0x: "sure, send off a background autopilot run to churn on the press task calibration"). The 25 mm x 100 mm cylinder
of the HOM scenes with the parts glued on it on 2026-11-16 (object_tracking_mjcf.BLOCK_ON): the sphere's 4 mm stem and
hex tag block, from the middle of the cylinder's underside, the block's top 24.41 mm below the cylinder's taped surface
(object_tracking_mjcf.assembled_stem_mm). The block is a cylinder 30 mm across (the hex block's flats) and 17 mm tall, as
rev_stem.py makes the sphere's. Masses as weighed, 30.1 g in all (cylinder 23.6 g, block 6.5 g, README of the bench
trials, changes of 2026-11-16): the cylinder 23.0 g, the stem 0.6 g, the block 6.5 g, the split rev_stem.py made for the
sphere (ball 21.0 g, stem 0.6 g, block 6.5 g of 28.1 g). The support post is replaced by a table under the block, its top
at the block's bottom when the object sits at the scene's start (hom_tasks.Plant.reset), removed at the lift as the post
is (the arm lifted the hand 40 mm from 1.6 s in both trials).

Variants, from the reference cell's scenes at (92.5, 102.5) mm (sr2_hom_replay.ref_scene; the plant moves the mounts):
  pads   the 1 mm sphere pads of the v4b tip (builder v2, 2.5 MPa)      pt   each tip one mesh, point contact
  imp1   impratio 1, the soft tangential contact of the twist's calibration      imp100   the scenes' impratio
  ns30   MuJoCo's noslip solver after the main one, 30 iterations (pickup_sim.py's *_ns scenes)
Scene files: scenes/cyl025_stem_block_{pads,pt}_imp{1,100}[_ns30].xml. --check compiles each and prints the object's
mass, centre of mass and inertia against the parts' sums, and the centre of mass's distance from each plan's pinch line.

    HOS_PLANT=kp0.55_kv0_fr0.18_fl0_dp0.011 .venv/bin/python logs/20261116-sim_real_reversal/press_stem.py [--check]
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("SCALE_HAND", "cad_v2")
os.environ.setdefault("HOS_SCENES", "logs/20261102-hom_servo_plant/scenes")
ROOT = Path.home() / "Programs/hand"
D = ROOT / "logs/20261116-sim_real_reversal"
sys.path.insert(0, str(D)); sys.path.insert(0, str(ROOT / "scripts/hardware")); sys.path.insert(0, str(ROOT / "scripts"))

R_CYL, HL_CYL = 0.0125, 0.050          # m: the taped radius and half length of the HOM scenes' cylinder
STEM, STEM_R, BLOCK_H, BLOCK_R = 0.02441, 0.002, 0.017, 0.015    # m: assembled_stem_mm('cylinder'), the hex block
MASS_G = {"cylinder": 23.0, "stem": 0.6, "block": 6.5}            # g, 30.1 in all
FRICTION = "1 0.005 0.0001"            # the scenes' geom friction (SR2Plant sets the sliding part to its mu)
VARIANTS = [(c, imp, ns) for c in ("pads", "pt") for imp in (1, 100) for ns in (0, 30)]
PLANS = {"generalist": ROOT / "docs/experiments/20261115-hom_bench_trials/plans/20261115-generalist_press_plan.json",
         "lat0": ROOT / "docs/experiments/20261115-hom_bench_trials/plans/20261116-press_x70_y110_lat0_plan.json"}
OLD_GEOM = '<geom type="cylinder" size="0.012500 0.05" mass="0.025" material="object_mat" friction="1 0.005 0.0001" />'


def name(contact, impratio, noslip):
    return f"cyl025_stem_block_{contact}_imp{impratio:g}" + (f"_ns{noslip}" if noslip else "") + ".xml"


def build(contact="pads", impratio=1, noslip=0):
    """One variant's scene (module docstring); returns its path."""
    import sr2_hom_replay as RP
    src = RP.ref_scene("cylinder", 25.0)[0 if contact == "pads" else 1]
    x = Path(src).read_text()
    assert x.count(OLD_GEOM) == 1, "object geom"
    # the object's body frame O: +y up at the start, +z along the cylinder's axis; the stem and block below its middle
    y0, y1 = -R_CYL, -(R_CYL + STEM)
    new = (f'<geom type="cylinder" size="{R_CYL:.6f} {HL_CYL:.2f}" mass="{MASS_G["cylinder"] / 1e3:.5f}" '
           f'material="object_mat" friction="{FRICTION}" />'
           f'<geom name="obj_stem" type="cylinder" fromto="0 {y0 + 0.001:.5f} 0 0 {y1:.5f} 0" size="{STEM_R:.4f}" '
           f'mass="{MASS_G["stem"] / 1e3:.5f}" material="object_mat" friction="{FRICTION}" />'
           f'<geom name="obj_block" type="cylinder" fromto="0 {y1:.5f} 0 0 {y1 - BLOCK_H:.5f} 0" size="{BLOCK_R:.4f}" '
           f'mass="{MASS_G["block"] / 1e3:.5f}" material="object_mat" friction="{FRICTION}" />')
    x = x.replace(OLD_GEOM, new)
    post = re.search(r'<body name="post" pos="[^"]*"><geom name="post"[^/]*/></body>', x)
    assert post, "post"
    import hand_object_scale_sim as S
    z_top = S.Z_OBJ - (R_CYL + STEM + BLOCK_H)          # the block's bottom at the scene's start pose
    table = (f'<body name="post" pos="0 0 {z_top - 0.004:.5f}"><geom name="post" type="box" size="0.06 0.06 0.004" '
             f'rgba="0.4 0.4 0.45 1" friction="{FRICTION}" /></body>')
    x = x[:post.start()] + table + x[post.end():]
    x, n = re.subn(r'impratio="[^"]*"', f'impratio="{impratio:g}"', x)
    assert n == 1, "impratio"
    if noslip:
        x, n = re.subn(r"<option ", f'<option noslip_iterations="{int(noslip)}" ', x)
        assert n == 1, "option"
    out = D / "scenes" / name(contact, impratio, noslip)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(x)
    return out


def parts_com():
    """The object's mass (g), centre of mass in O (mm) and its inertia about that point (g mm^2, diagonal in O) from the
    parts: solid cylinders, the stem 1 mm into the cylinder as built."""
    import numpy as np
    parts = []                                          # (mass g, centre mm, (Ixx, Iyy, Izz) g mm^2 about the centre)
    m, r, L = MASS_G["cylinder"], 1e3 * R_CYL, 2e3 * HL_CYL
    parts.append((m, np.zeros(3), np.array([m * (3 * r ** 2 + L ** 2) / 12, m * (3 * r ** 2 + L ** 2) / 12,
                                            m * r ** 2 / 2])))
    for key, a, b, rr in (("stem", -1e3 * R_CYL + 1.0, -1e3 * (R_CYL + STEM), 1e3 * STEM_R),
                          ("block", -1e3 * (R_CYL + STEM), -1e3 * (R_CYL + STEM + BLOCK_H), 1e3 * BLOCK_R)):
        m, L = MASS_G[key], abs(a - b)
        ax = m * (3 * rr ** 2 + L ** 2) / 12              # about x and z (the part's axis is O's y)
        parts.append((m, np.array([0.0, 0.5 * (a + b), 0.0]), np.array([ax, m * rr ** 2 / 2, ax])))
    M = sum(p[0] for p in parts)
    c = sum(p[0] * p[1] for p in parts) / M
    I = sum(p[2] + p[0] * (np.sum((p[1] - c) ** 2) - (p[1] - c) ** 2) for p in parts)
    return M, c, I


def check(paths):
    """Each scene's object against the parts, and the centre of mass's place relative to each plan's pinch line."""
    import mujoco
    import numpy as np
    M, c, I = parts_com()
    print(f"parts: {M:.1f} g, centre of mass in O {c.round(2).tolist()} mm, inertia about it {I.round(0).tolist()} g mm^2")
    out = {"parts": {"mass_g": round(M, 2), "com_O_mm": c.round(3).tolist(), "inertia_g_mm2": I.round(1).tolist()},
           "scenes": {}, "plans": {}}
    for p in paths:
        m = mujoco.MjModel.from_xml_path(str(p))
        b = m.body("screwdriver_medium").id
        row = {"mass_g": round(1e3 * float(m.body_mass[b]), 3), "com_O_mm": (1e3 * m.body_ipos[b]).round(3).tolist(),
               "inertia_g_mm2": sorted((1e9 * m.body_inertia[b]).round(1).tolist()),
               "impratio": float(m.opt.impratio), "noslip_iterations": int(m.opt.noslip_iterations),
               "pads": int(sum(1 for g in range(m.ngeom) if re.fullmatch(r"(thumb|index|middle)_pad\d+",
                                                                        m.geom(g).name or "")))}
        out["scenes"][p.name] = row
        print(f"{p.name}: {row}")
    # the plans' pinch lines (the thumb's and the index's pad-centre targets in P) and the centre of mass at the start
    from sr2hand.geometry.rotations import quat_to_mat
    for key, f in PLANS.items():
        h = json.loads(f.read_text())["meta"]["hom"]
        st = h["setup"]
        ctr = (np.array([0.0, 0.0, 0.12]) - np.asarray(st["palm_m"])) * 1e3       # the object's centre in P (mm)
        R = quat_to_mat(np.asarray(st["quat"], float))
        com = ctr + R @ c
        a = R[:, 2]
        t_th, t_in = (1e3 * np.asarray(st["targets_m"][k]) for k in ("thumb", "index"))
        u = (t_in - t_th) / np.linalg.norm(t_in - t_th)                        # the pinch line, thumb to index
        dvec = (com - t_th) - u * ((com - t_th) @ u)                           # from the line to the centre of mass
        along = float(dvec @ a)                                                # along the cylinder's axis (+ to the
        below = float(-dvec[2])                                                # middle's end, palm -y)
        ctr_line = (ctr - t_th) - u * ((ctr - t_th) @ u)
        torque = M * 1e-3 * 9.81 * float(np.hypot(dvec[0], dvec[1])) * 1e-3   # gravity's torque about the line
        out["plans"][key] = {"centre_P_mm": ctr.round(2).tolist(), "com_P_mm": com.round(2).tolist(),
                             "pinch_line_to_centre_mm": round(float(np.linalg.norm(ctr_line)), 2),
                             "com_from_pinch_line_along_axis_mm": round(along, 2),
                             "com_below_pinch_line_mm": round(below, 2),
                             "gravity_torque_about_pinch_line_mNm": round(1e3 * torque, 2),
                             "pinch_lat_deg": round(float(np.degrees(h["v"][6])), 1),
                             "pusher_lat_deg": round(float(np.degrees(h["v"][7])), 1)}
        print(f"{key}: {out['plans'][key]}")
    (D / "press_stem_check.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    paths = [build(*v) for v in VARIANTS]
    for p in paths:
        print("scene", p.relative_to(ROOT), flush=True)
    if a.check:
        check(paths)


if __name__ == "__main__":
    main()
