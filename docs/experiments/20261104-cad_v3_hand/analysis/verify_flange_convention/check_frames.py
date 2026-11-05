"""Adversarial check of the tool0 convention (no mesh loading): the URDF chain against UR's DH table, the lab's RTDE
poses against the DH flange, the wrist-3 visual origin, and the hand clockings for both hole conventions.

    /home/humanoid/Programs/hand/.venv/bin/python logs/20261104-cad_v3/verify_flange_convention/check_frames.py
"""
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
URDF = Path.home() / "Programs/rh56_controller/magpie_force_control/src/ur_description/ur5.urdf"
POSES = REPO / "docs/experiments/20261022-ur5_tip_calibration/lab/poses.jsonl"


def T_of(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()     # URDF rpy: fixed-axis x, y, z = Rz Ry Rx
    T[:3, 3] = xyz
    return T


def urdf_joints():
    root = ET.parse(URDF).getroot()
    J = {}
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = [float(v) for v in o.get("xyz", "0 0 0").split()] if o is not None else [0, 0, 0]
        rpy = [float(v) for v in o.get("rpy", "0 0 0").split()] if o is not None else [0, 0, 0]
        ax = j.find("axis")
        J[j.get("name")] = dict(type=j.get("type"), parent=j.find("parent").get("link"), child=j.find("child").get("link"),
                                T=T_of(xyz, rpy), axis=np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else None)
    vis = {}
    for l in root.findall("link"):
        v = l.find("visual")
        if v is not None and v.find("origin") is not None:
            o = v.find("origin")
            vis[l.get("name")] = T_of([float(x) for x in o.get("xyz").split()], [float(x) for x in o.get("rpy").split()])
    return J, vis


def chain(J, src, dst):
    by_child = {j["child"]: (n, j) for n, j in J.items()}
    path = []
    link = dst
    while link != src:
        n, j = by_child[link]
        path.append((n, j))
        link = j["parent"]
    return path[::-1]


def fk_urdf(J, q, src="base_link", dst="tool0"):
    names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    qd = dict(zip(names, q))
    T = np.eye(4)
    for n, j in chain(J, src, dst):
        T = T @ j["T"]
        if j["type"] == "revolute":
            R = np.eye(4)
            R[:3, :3] = Rotation.from_rotvec(j["axis"] * qd[n]).as_matrix()
            T = T @ R
    return T


DH = [(0.089159, 0.0, math.pi / 2), (0.0, -0.425, 0.0), (0.0, -0.39225, 0.0), (0.10915, 0.0, math.pi / 2),
      (0.09465, 0.0, -math.pi / 2), (0.0823, 0.0, 0.0)]          # UR's table (d, a, alpha), CB3 UR5


def fk_dh(q):
    T = np.eye(4)
    for (d, a, al), th in zip(DH, q):
        ct, st, ca, sa = math.cos(th), math.sin(th), math.cos(al), math.sin(al)
        T = T @ np.array([[ct, -st * ca, st * sa, a * ct], [st, ct * ca, -ct * sa, a * st], [0, sa, ca, d], [0, 0, 0, 1]])
    return T


def rot_angle_deg(Ra, Rb):
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(Ra.T @ Rb).as_rotvec())))


def about_z_deg(Ra, Rb):
    """Turn about the common z of frame b relative to frame a (deg), from the x axes projected on a's xy plane."""
    xb = Ra.T @ Rb[:, 0]
    return float(np.degrees(math.atan2(xb[1], xb[0])))


def main():
    out = {}
    J, vis = urdf_joints()
    # 1. tool0 in wrist_3_link
    T_w3_t0 = J["wrist_3-flange"]["T"] @ J["flange-tool0"]["T"]
    out["tool0_in_wrist_3_link"] = np.round(T_w3_t0, 12).tolist()
    # 2. the controller base: 'base' in base_link
    T_bl_base = J["base_link-base_fixed_joint"]["T"]
    rng = np.random.default_rng(7)
    errs = []
    for _ in range(200):
        q = rng.uniform(-math.pi, math.pi, 6)
        T_u = np.linalg.inv(T_bl_base) @ fk_urdf(J, q)
        T_d = fk_dh(q)
        errs.append((float(np.linalg.norm(T_u[:3, 3] - T_d[:3, 3]) * 1e3), rot_angle_deg(T_u[:3, :3], T_d[:3, :3])))
    errs = np.array(errs)
    out["urdf_tool0_vs_dh_200_random"] = dict(max_pos_mm=float(errs[:, 0].max()), max_rot_deg=float(errs[:, 1].max()))
    # 3. the lab's RTDE TCP poses against the nominal DH flange (all 27 rows of poses.jsonl)
    rows = [json.loads(l) for l in POSES.read_text().splitlines() if l.strip()]
    lab = []
    for r in rows:
        T_tcp = np.array(r["T_tcp_mm"])
        R_tcp = Rotation.from_rotvec(r["tcp_pose_m_rad"][3:]).as_matrix()
        T_d = fk_dh(r["q_rad"])
        T_u = np.linalg.inv(T_bl_base) @ fk_urdf(J, r["q_rad"])
        lab.append(dict(label=r["label"], rot_deg=round(rot_angle_deg(T_d[:3, :3], R_tcp), 3),
                        about_tool_axis_deg=round(about_z_deg(T_d[:3, :3], R_tcp), 3),
                        urdf_rot_deg=round(rot_angle_deg(T_u[:3, :3], R_tcp), 3),
                        pos_mm=round(float(np.linalg.norm(T_d[:3, 3] * 1e3 - T_tcp[:3, 3])), 2)))
    out["lab_rtde_tcp_vs_dh_flange"] = lab
    a = np.array([[x["rot_deg"], x["about_tool_axis_deg"], x["pos_mm"]] for x in lab])
    out["lab_summary"] = dict(n=len(lab), max_rot_deg=float(a[:, 0].max()), about_tool_axis_deg_range=[float(a[:, 1].min()),
                              float(a[:, 1].max())], pos_mm_range=[float(a[:, 2].min()), float(a[:, 2].max())])
    # 4. the wrist-3 visual mesh in tool0
    T_t0_mesh = np.linalg.inv(T_w3_t0) @ vis["wrist_3_link"]
    out["wrist3_visual_in_tool0"] = np.round(T_t0_mesh, 6).tolist()
    out["mesh_axes_in_tool0"] = {k: np.round(T_t0_mesh[:3, :3] @ v, 6).tolist()
                                 for k, v in (("+x_mesh", [1, 0, 0]), ("+y_mesh", [0, 1, 0]), ("+z_mesh", [0, 0, 1]))}
    # 5. clockings: R_k = Rz(theta) Rx(180), t = -R c_A
    c_A = np.array([-111.0, 116.0, 28.0])
    m6_A = np.array([-21.0, 69.0, 159.0, 249.0])
    def clock(theta):
        R = Rotation.from_euler("z", theta, degrees=True).as_matrix() @ np.diag([1.0, -1.0, -1.0])
        t = -R @ c_A
        q = Rotation.from_matrix(R).as_quat()                 # xyzw
        q = np.array([q[3], q[0], q[1], q[2]])
        m6_E = [(theta - a_) % 360 for a_ in m6_A]
        return dict(theta_deg=theta, pos_A_origin_in_E_mm=np.round(t, 3).tolist(), quat_wxyz=np.round(q, 8).tolist(),
                    m6_E_deg=sorted(round(float(v), 6) for v in m6_E))
    out["claim_clockings_diagonal_holes"] = [clock(24.0 + 90.0 * k) for k in range(4)]
    out["alternative_clockings_holes_on_axes"] = [clock(69.0 + 90.0 * k) for k in range(4)]
    # 6. gantry directions (from the flange axis in A) and where the connector (-y_E) points, both conventions
    gantries = dict(thumb=np.array([-111.0, 180.375]), index=np.array([-55.0, 60.0]), middle=np.array([-167.0, 60.0]))
    ang_A = {k: float(np.degrees(math.atan2(*(v - c_A[:2])[::-1]))) for k, v in gantries.items()}
    def wrap(x):
        return (x + 180.0) % 360.0 - 180.0
    for key, base in (("claim", 24.0), ("holes_on_axes", 69.0)):
        rows_ = []
        for k in range(4):
            th = base + 90.0 * k
            dirs = {g: round(wrap(th - a_), 2) for g, a_ in ang_A.items()}
            rows_.append(dict(k=k, theta=th, gantry_dir_E_deg=dirs,
                              connector_minus_gantry_deg={g: round(wrap(-90.0 - d), 2) for g, d in dirs.items()}))
        out[f"gantry_directions_{key}"] = rows_
    out["gantry_angles_in_A_deg"] = ang_A
    # 7. handedness of the k-rule sentence: angle of x_E from y_E, seen from +z_E (the tool side)
    out["x_from_y_seen_from_tool_side_deg"] = float(np.degrees(math.atan2(np.cross([0, 1, 0], [1, 0, 0])[2], 0.0)))
    (HERE / "check_frames.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "lab_rtde_tcp_vs_dh_flange"}, indent=1))


if __name__ == "__main__":
    main()
