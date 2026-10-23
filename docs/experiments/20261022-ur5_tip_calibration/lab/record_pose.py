"""Record the UR5's pose for the hand homing of 2026-10-22, read-only (RTDE receive, nothing sent to the robot).

    /usr/bin/python3 docs/experiments/20261022-ur5_tip_calibration/lab/record_pose.py LABEL [--seconds 1.0]

Averages the joints and the pendant's TCP pose over the window, adds the flange pose from the joints by the UR5's nominal
DH parameters, and appends one JSON line to poses.jsonl beside this file. For a seating on an indenter's key it also prints
where the indenter would be by the plan's homing geometry (T_BI = T_BE T_EM T_IM_home^-1, with the plan's nominal T_EM:
x_M on the flange's x)."""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import rtde_receive

HERE = Path(__file__).resolve().parent
PLAN = HERE / "20261022-plan_sphere_measured_rod.json"     # T_EM with the measured -92.15 deg turn
DH = [(0.089159, 0.0, math.pi / 2), (0.0, -0.425, 0.0), (0.0, -0.39225, 0.0), (0.10915, 0.0, math.pi / 2),
      (0.09465, 0.0, -math.pi / 2), (0.0823, 0.0, 0.0)]
TOP = {"sphere": 46.0, "rod": 42.5}         # the indenters' tops above their base (scripts/ur5_calibration_mount.py)


def fk(q):
    T = np.eye(4)
    for (d, a, al), th in zip(DH, q):
        ct, st, ca, sa = math.cos(th), math.sin(th), math.cos(al), math.sin(al)
        T = T @ np.array([[ct, -st * ca, st * sa, a * ct], [st, ct * ca, -ct * sa, a * st], [0, sa, ca, d], [0, 0, 0, 1]])
    T[:3, 3] *= 1e3
    return T


def rotvec_to_R(v):
    v = np.asarray(v, float)
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--robot", default="192.168.0.4")
    a = ap.parse_args()
    r = rtde_receive.RTDEReceiveInterface(a.robot, 125.0)
    qs, ps, t_end = [], [], time.time() + a.seconds
    while time.time() < t_end:
        qs.append(r.getActualQ())
        ps.append(r.getActualTCPPose())
        time.sleep(0.008)
    mode, safety = r.getRobotMode(), r.getSafetyMode()
    r.disconnect()
    qs, ps = np.array(qs), np.array(ps)
    q = qs.mean(axis=0)
    T_fk = fk(q)
    T_tcp = np.eye(4)
    T_tcp[:3, :3], T_tcp[:3, 3] = rotvec_to_R(ps[-1, 3:]), ps[:, :3].mean(axis=0) * 1e3
    rec = dict(label=a.label, time=time.strftime("%Y-%m-%d %H:%M:%S"), robot_mode=int(mode), safety_mode=int(safety),
               q_rad=q.tolist(), q_deg=np.degrees(q).round(3).tolist(), q_sd_deg=np.degrees(qs.std(axis=0)).max().round(4),
               tcp_pose_m_rad=ps.mean(axis=0).tolist(), T_tcp_mm=T_tcp.round(4).tolist(), T_flange_fk_mm=T_fk.round(4).tolist())
    with open(HERE / "poses.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    np.set_printoptions(suppress=True, precision=2)
    print(f"{a.label}: recorded at {rec['time']} (robot mode {mode}, safety {safety}, joints still to "
          f"{rec['q_sd_deg']:.4f} deg over {len(qs)} reads)")
    print("  joints deg", np.round(rec["q_deg"], 2))
    print("  flange (pendant TCP) mm", np.round(T_tcp[:3, 3], 1), " FK", np.round(T_fk[:3, 3], 1))
    print("  tool axis", np.round(T_tcp[:3, 2], 3), " flange x", np.round(T_tcp[:3, 0], 3), " flange y", np.round(T_tcp[:3, 1], 3))
    plan = json.loads(PLAN.read_text())
    g = plan["geometry"]
    T_EM, T_IMh = np.array(g["T_EM"]), np.array(g["T_IM_home"])
    for kind in ("sphere", "rod"):
        Th = T_IMh.copy()
        Th[2, 3] += TOP[kind] - TOP["sphere"]                  # the mount sits as far above each indenter's top
        T_BI = T_tcp @ T_EM @ np.linalg.inv(Th)
        R = T_BI[:3, :3]
        yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        tilt = math.degrees(math.acos(float(np.clip(R[2, 2], -1, 1))))
        print(f"  if seated on the {kind}'s key: its base at {np.round(T_BI[:3, 3], 1)} mm, yaw {yaw:.1f} deg, tilt {tilt:.1f} deg"
              f" (plan: (0, -450, 56) mm, yaw -90, tilt 0)")


if __name__ == "__main__":
    main()
