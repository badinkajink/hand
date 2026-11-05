#!/usr/bin/env python3
"""Register the cad_v1 finger parts (assets/cad_v1, the parts of assets/mjcf/experimental/20261013-cad_v1) onto the
three Finger v3 instances of file "hand", and express each finger's frame F (cad_v1 convention: origin at the yaw
centre, x_F along the yaw axis toward the dorsal side, z_F up) and its CAD joint angles in A.

Writes register.json beside this file.

    ~/.claude/bin/resguard.sh run --mem 3G --cpu 100 -- .venv/bin/python logs/20261104-cad_v3/gantry/register.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import classify  # noqa: E402

REPO = HERE.parents[2]
ASM = json.loads((REPO / "assets/mjcf/experimental/20261013-cad_v1/assembly.json").read_text())
CAD1 = REPO / "assets/cad_v1"
FILES = {"carriage": "carriage.stl", "yaw_link": "mcp_mount_YAW-link.stl", "mcp_link": "pip_mount_MCP_link.stl",
         "dip_link": "fingertip_mount_DIP_link.stl", "tip": "fingertip.stl", "servo": "feetech scs0009.stl"}
VOL = {"carriage": 23913, "yaw_link": 10683, "mcp_link": 21293, "dip_link": 5189, "tip": 4822, "servo": 6673}
RNG = np.random.default_rng(0)


def T_F_q0(part):
    """cad_v1 STL coordinates -> F at q = 0."""
    Td = np.asarray(ASM["T_F_from_part_design_mm"][part], float)
    src = "servo" if part.startswith("servo") else part
    Ts = np.asarray(ASM["T_part_design_from_stl"][src], float)
    return Td @ Ts


def samples(mesh, n):
    pts, fi = trimesh.sample.sample_surface(mesh, n, seed=int(RNG.integers(1 << 30)))
    return np.asarray(pts), np.asarray(mesh.face_normals[fi])


def icp_p2plane(src_pts, tgt_tree, tgt_pts, tgt_nrm, T0, iters=60):
    T = T0.copy()
    for _ in range(iters):
        p = src_pts @ T[:3, :3].T + T[:3, 3]
        d, i = tgt_tree.query(p)
        q, n = tgt_pts[i], tgt_nrm[i]
        keep = d < max(2.0, 3 * np.median(d))
        p, q, n = p[keep], q[keep], n[keep]
        A = np.hstack([np.cross(p, n), n])
        b = np.einsum("ij,ij->i", q - p, n)
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        dR = Rotation.from_rotvec(x[:3]).as_matrix()
        dT = np.eye(4)
        dT[:3, :3], dT[:3, 3] = dR, x[3:]
        T = dT @ T
        if np.linalg.norm(x) < 1e-9:
            break
    p = src_pts @ T[:3, :3].T + T[:3, 3]
    d, i = tgt_tree.query(p)
    r = np.abs(np.einsum("ij,ij->i", p - tgt_pts[i], tgt_nrm[i]))
    return T, float(np.sqrt(np.mean(r ** 2))), float(np.percentile(d, 99))


def register(src_mesh, tgt_mesh):
    """Rigid transform taking src onto tgt (same part, different tessellation and placement)."""
    sp, _ = samples(src_mesh, 4000)
    tp, tn = samples(tgt_mesh, 60000)
    tree = cKDTree(tp)
    cs, ct = src_mesh.center_mass, tgt_mesh.center_mass
    Es = src_mesh.principal_inertia_vectors.T      # columns = principal axes
    Et = tgt_mesh.principal_inertia_vectors.T
    best = None
    for perm in itertools.permutations(range(3)):
        for s in itertools.product([1, -1], repeat=3):
            R = Et @ (np.diag(s) @ np.eye(3)[list(perm)]) @ Es.T     # signed permutation of principal axes
            if np.linalg.det(R) < 0:
                continue
            T0 = np.eye(4)
            T0[:3, :3] = R
            T0[:3, 3] = ct - R @ cs
            T, rms, p99 = icp_p2plane(sp, tree, tp, tn, T0, iters=40)
            if best is None or rms < best[1]:
                best = (T, rms, p99)
    T, rms, p99 = icp_p2plane(sp, tree, tp, tn, best[0], iters=100)
    # symmetric check: target samples to the moved source surface
    mv = src_mesh.copy()
    mv.apply_transform(T)
    mp, mn = samples(mv, 60000)
    d2, _ = cKDTree(mp).query(tp[:4000])
    return T, rms, p99, float(np.percentile(d2, 99))


def rot_about(R, axis):
    """Angle of rotation R about the unit axis (R assumed a rotation about that axis), rad, right-hand."""
    rv = Rotation.from_matrix(R).as_rotvec()
    ang = np.linalg.norm(rv)
    if ang < 1e-12:
        return 0.0, 0.0
    off = float(np.degrees(np.arccos(np.clip(abs(rv @ axis) / ang, -1, 1))))
    return float(np.sign(rv @ axis) * ang), off


def main():
    src = {k: trimesh.load(CAD1 / f, force="mesh") for k, f in FILES.items()}
    out = {"F_convention": "cad_v1 finger frame: origin = yaw centre, x_F = yaw axis toward the dorsal side "
                           "(bearing wall -> servo), z_F = up (carriage arm up), y_F = z_F x x_F",
           "fingers": {}}
    for inst in classify.FINGERS:
        parts = [p for p in classify.PARTS.values() if classify.sub(p) == inst]
        found = {}
        for k, v in VOL.items():
            c = [p for p in parts if abs(p["volume"] - v) < 3]
            found[k] = c
        res = {"parts": {}}
        T_A = {}
        for k in ("carriage", "yaw_link", "mcp_link", "dip_link", "tip"):
            p = found[k][0]
            tgt = trimesh.load(REPO / p["stl"], force="mesh")
            T, rms, p99, p99b = register(src[k], tgt)
            T_A[k] = T                                   # cad_v1 STL -> A
            res["parts"][k] = dict(name=p["name"], label=p["label"], rms_mm=round(rms, 4), p99_mm=round(p99, 4),
                                   p99_back_mm=round(p99b, 4), det=round(float(np.linalg.det(T[:3, :3])), 6))
        # F in A from the carriage
        T_AF = T_A["carriage"] @ np.linalg.inv(T_F_q0("carriage"))
        R_AF = T_AF[:3, :3]
        # servos: assign each instance to the body whose q=0 relative pose it matches
        servo_rows = []
        for p in found["servo"]:
            tgt = trimesh.load(REPO / p["stl"], force="mesh")
            T, rms, p99, p99b = register(src["servo"], tgt)
            servo_rows.append((p, T, rms, p99, p99b))
        # joint angles: yaw = rotation of the yaw link about x_F through the yaw centre, etc.
        T_F = {k: np.linalg.inv(T_AF) @ T_A[k] for k in T_A}         # cad_v1 STL -> F, CAD pose
        D_yaw = T_F["yaw_link"] @ np.linalg.inv(T_F_q0("yaw_link"))
        yaw, yaw_off = rot_about(D_yaw[:3, :3], np.array([1.0, 0, 0]))
        Ryaw = np.eye(4)
        Ryaw[:3, :3] = Rotation.from_rotvec([yaw, 0, 0]).as_matrix()
        # MCP: in the yaw frame
        T_mcp_y = np.linalg.inv(Ryaw) @ T_F["mcp_link"]
        D_mcp = T_mcp_y @ np.linalg.inv(T_F_q0("mcp_link"))
        mcp, mcp_off = rot_about(D_mcp[:3, :3], np.array([0, 1.0, 0]))
        c_mcp = np.array([0, 0, -20.75])
        Rm = np.eye(4)
        Rm[:3, :3] = Rotation.from_rotvec([0, mcp, 0]).as_matrix()
        Rm[:3, 3] = c_mcp - Rm[:3, :3] @ c_mcp
        T_dip_m = np.linalg.inv(Rm) @ np.linalg.inv(Ryaw) @ T_F["dip_link"]
        D_pip = T_dip_m @ np.linalg.inv(T_F_q0("dip_link"))
        pip, pip_off = rot_about(D_pip[:3, :3], np.array([0, 1.0, 0]))
        c_pip = np.array([0, 0, -51.55])
        Rp = np.eye(4)
        Rp[:3, :3] = Rotation.from_rotvec([0, pip, 0]).as_matrix()
        Rp[:3, 3] = c_pip - Rp[:3, :3] @ c_pip
        # residual translations of the joint fits (how well each relative motion is a rotation about its axis)
        def resid(D, centre, R):
            Tm = np.eye(4)
            Tm[:3, :3] = R[:3, :3] if R.shape == (4, 4) else R
            Tm[:3, 3] = centre - Tm[:3, :3] @ centre
            E = np.linalg.inv(Tm) @ D
            return float(np.linalg.norm(E[:3, 3])), float(np.degrees(np.linalg.norm(Rotation.from_matrix(E[:3, :3]).as_rotvec())))
        r_yaw = resid(D_yaw, np.zeros(3), Ryaw)
        r_mcp = resid(D_mcp, c_mcp, Rm)
        r_pip = resid(D_pip, c_pip, Rp)
        # tip: relative to the DIP link at q = 0
        T_tip_check = np.linalg.inv(Rp) @ T_dip_m
        # body poses at the CAD angles, for the servo assignment
        body = {"mount": np.eye(4), "yaw_frame": Ryaw, "mcp_frame": Ryaw @ Rm, "pip_frame": Ryaw @ Rm @ Rp}
        servo_body = {"servo_yaw": "mount", "servo_mcp": "yaw_frame", "servo_pip": "mcp_frame"}
        servos = {}
        for sname, b in servo_body.items():
            pred = T_AF @ body[b] @ T_F_q0(sname)        # cad_v1 servo STL -> A predicted
            best = min(servo_rows, key=lambda r: np.linalg.norm(r[1][:3, 3] - pred[:3, 3]))
            dpos = float(np.linalg.norm(best[1][:3, 3] - pred[:3, 3]))
            dang = float(np.degrees(np.linalg.norm(Rotation.from_matrix(best[1][:3, :3] @ pred[:3, :3].T).as_rotvec())))
            servos[sname] = dict(name=best[0]["name"], rms_mm=round(best[2], 4), pose_error_mm=round(dpos, 3),
                                 pose_error_deg=round(dang, 3))
        res["servos"] = servos
        # predicted tip vs registered tip
        pred_tip = T_AF @ body["pip_frame"] @ T_F_q0("tip")
        res["tip_pose_error_mm"] = round(float(np.linalg.norm(pred_tip[:3, 3] - T_A["tip"][:3, 3])), 3)
        res["tip_pose_error_deg"] = round(float(np.degrees(np.linalg.norm(Rotation.from_matrix(
            T_A["tip"][:3, :3] @ pred_tip[:3, :3].T).as_rotvec()))), 3)
        res["T_A_from_F"] = np.round(T_AF, 6).tolist()
        res["yaw_centre_A_mm"] = np.round(T_AF[:3, 3], 4).tolist()
        res["axes_A"] = {"x_F (yaw axis, dorsal)": np.round(R_AF[:, 0], 6).tolist(),
                         "y_F": np.round(R_AF[:, 1], 6).tolist(), "z_F (up)": np.round(R_AF[:, 2], 6).tolist()}
        res["mcp_centre_A_mm"] = np.round((T_AF @ body["yaw_frame"] @ np.r_[c_mcp, 1])[:3], 4).tolist()
        res["pip_centre_A_mm"] = np.round((T_AF @ body["mcp_frame"] @ np.r_[c_pip, 1])[:3], 4).tolist()
        res["angles_about_F_axes_deg"] = dict(
            yaw_about_xF=round(float(np.degrees(yaw)), 3), mcp_about_yF=round(float(np.degrees(mcp)), 3),
            pip_about_yF=round(float(np.degrees(pip)), 3))
        res["angle_fit"] = dict(yaw_axis_offset_deg=round(yaw_off, 3), mcp_axis_offset_deg=round(mcp_off, 3),
                                pip_axis_offset_deg=round(pip_off, 3),
                                yaw_resid_mm_deg=[round(v, 4) for v in r_yaw], mcp_resid_mm_deg=[round(v, 4) for v in r_mcp],
                                pip_resid_mm_deg=[round(v, 4) for v in r_pip])
        out["fingers"][inst] = res
        print(inst, json.dumps({k: res[k] for k in ("yaw_centre_A_mm", "axes_A", "angles_about_F_axes_deg", "angle_fit",
                                                     "servos", "tip_pose_error_mm", "tip_pose_error_deg")}), flush=True)
        print("   parts", json.dumps(res["parts"]), flush=True)
    (HERE / "register.json").write_text(json.dumps(out, indent=1))
    print("wrote", HERE / "register.json")


if __name__ == "__main__":
    main()
