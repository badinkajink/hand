#!/usr/bin/env python3
"""Can the middle finger act as a pusher about the thumb-index pinch axis?

Kinematic check for the pinch + pusher loading of Wang, Oh and Pollard, "Relative Contact
Velocity-Controlled Hand-Object Mechanism for Dexterous Tool Manipulation" (arXiv 2609.25619):
two fingers pinch the tool as a revolute joint and a third regulates its rotation about the
pinch axis. On real_v1 the only free finger in the 2026-09-16 swing grasps is the middle.

    uv run python scripts/real_v1_pusher_reach.py --out docs/experiments/20260926-hom_pusher_reach

Rigid grip pose (anchor) on each swing variant's flat scene. Reports the pinch axis, the gravity
moment about it, the moment arm the middle pad's normal force has about it, and the rigid-IK
residual for the middle tip to follow the tool through theta = 0..90 deg about the pinch axis,
(a) at a fixed material point, (b) free to slide along the shaft (best of s in [-30, 30] mm).
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path("/home/humanoid/Programs/hand")
sys.path.insert(0, str(ROOT / "src"))
from morphohand.tools.keyframe_ik import FINGERS, TIPS, ik_finger  # noqa: E402

SC = ROOT / "assets/mjcf/experimental/20260904-chain_bothsets"
OBJ = "screwdriver_medium"
VARIANTS = {  # D-label: cached swing-study fit tag
    "D8": "sv1_w0099_b100_y180_sq10",
    "D8 sq12": "sv1_w0099_b100_sq12",
    "D6 ta20": "sv1_u0308_b050_ta20_y180_sq10",
    "D4 ta30": "g12_b095_ta30_y180_sq10",
    "D7 ta20": "rv05_manual_b85_ta20_y180_sq10",
    "D2 ta30": "sv1_w2360_b075_ta30_y180_sq10",
}


def rot(axis, a):
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def pad_contact(m, d, f, ogid):
    """Witness points pad->object and the unit normal pointing INTO the object."""
    best = None
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != m.body(TIPS[f]).id or (m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0):
            continue
        ft = np.zeros(6)
        dist = mujoco.mj_geomDistance(m, d, g, ogid, 0.05, ft)
        if best is None or dist < best[0]:
            best = (dist, ft.copy(), g)
    dist, ft, g = best
    return dist, ft[:3], ft[3:], g


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = {}
    for lab, tag in VARIANTS.items():
        fit = SC / f"{tag}_fit.json"
        flat = SC / f"{tag}__flat.xml"
        if not fit.exists() or not flat.exists():
            print(lab, "missing", tag); continue
        rec = json.loads(fit.read_text())
        m = mujoco.MjModel.from_xml_path(str(flat))
        d = mujoco.MjData(m)
        key = m.key("open_ik").id
        mujoco.mj_resetDataKeyframe(m, d, key)
        for j, v in rec["anchor"].items():
            d.qpos[m.jnt_qposadr[m.joint(j).id]] = v
        mujoco.mj_forward(m, d)
        ogid = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(OBJ).id][0]
        r_obj = m.geom_size[ogid, 0]
        c_obj = d.geom_xpos[ogid].copy()
        ax = d.geom_xmat[ogid].reshape(3, 3)[:, 2].copy()
        cg = d.body(OBJ).xipos.copy()
        mass = m.body_subtreemass[m.body(OBJ).id]
        pts, nrm, dist = {}, {}, {}
        for f in FINGERS:
            dd, a, b, g = pad_contact(m, d, f, ogid)
            # contact point on the shaft surface nearest the pad witness; normal into the shaft
            rel = a - c_obj
            radial = rel - ax * (rel @ ax)
            n_in = -radial / np.linalg.norm(radial)
            pts[f] = c_obj + ax * (rel @ ax) - n_in * r_obj
            nrm[f] = n_in
            dist[f] = dd * 1e3
        p_ax = pts["thumb"]
        a_hat = pts["index"] - pts["thumb"]
        a_hat /= np.linalg.norm(a_hat)
        g_vec = mass * m.opt.gravity
        tau_g = np.cross(cg - p_ax, g_vec) @ a_hat
        sgn = np.sign(tau_g) if tau_g != 0 else 1.0
        # middle normal force (1 N into the tool) moment about the pinch axis, same sign as
        # gravity's = pushes the swing along, opposite = can hold it back / lower it
        lever = np.cross(pts["middle"] - p_ax, nrm["middle"]) @ a_hat * 1e3  # mm (N m per N)
        # perpendicular distance of middle contact / CG from the pinch line
        def perp(p):
            v = p - p_ax
            return np.linalg.norm(v - a_hat * (v @ a_hat)) * 1e3
        row = {"tag": tag, "scene": str(flat.relative_to(ROOT)), "mass_g": round(mass * 1e3, 1),
               "pinch_axis_vs_shaft_deg": round(float(np.degrees(np.arccos(abs(a_hat @ ax)))), 1),
               "pinch_axis_vs_horizontal_deg": round(float(np.degrees(np.arcsin(abs(a_hat[2])))), 1),
               "cg_from_pinch_mm": round(perp(cg), 1),
               "gravity_moment_mNm": round(float(tau_g) * 1e3, 2),
               "middle_from_pinch_mm": round(perp(pts["middle"]), 1),
               "middle_lever_mm_signed": round(float(lever * np.sign(-sgn)), 2),
               "pad_dist_mm": {f: round(v, 2) for f, v in dist.items()},
               "middle_normal": np.round(nrm["middle"], 3).tolist()}
        q0 = d.qpos.copy()
        tip0 = d.body(TIPS["middle"]).xpos.copy()
        fixed, slide = {}, {}
        for deg in (10, 20, 30, 45, 60, 75, 90):
            R = rot(a_hat, sgn * np.radians(deg))
            d.qpos[:] = q0
            fixed[deg] = round(ik_finger(m, d, "middle", p_ax + R @ (tip0 - p_ax), iters=400) * 1e3, 1)
            best = 1e9
            for s in np.linspace(-0.03, 0.03, 13):
                d.qpos[:] = q0
                best = min(best, ik_finger(m, d, "middle", p_ax + R @ (tip0 + s * ax - p_ax), iters=300))
            slide[deg] = round(best * 1e3, 1)
        d.qpos[:] = q0
        # how far the middle tip moves of a 10 mm request along each world axis from the grip
        reach = {}
        for name, v in {"-z": (0, 0, -1), "+z": (0, 0, 1), "-x": (-1, 0, 0), "+x": (1, 0, 0),
                        "-y": (0, -1, 0), "+y": (0, 1, 0)}.items():
            d.qpos[:] = q0
            reach[name] = round(10 - ik_finger(m, d, "middle", tip0 + 0.010 * np.array(v, float),
                                               iters=400) * 1e3, 1)
        row["middle_reach_of_10mm"] = reach
        row["middle_q_deg"] = {j: round(float(np.degrees(rec["anchor"][j])), 1) for j in FINGERS["middle"]}
        row["middle_ik_res_fixed_mm"] = fixed
        row["middle_ik_res_slide_mm"] = slide
        out[lab] = row
        print(lab, json.dumps(row))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pusher_reach.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
