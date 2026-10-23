"""The indenters' poses from the hand seatings in poses.jsonl, with the plan's homing geometry and the mount clocked as
measured on 2026-10-22: palmar (x_M) along the flange's -y, i.e. T_EM turned -90 deg about the tool axis from the plan's
nominal (x_M on the flange's x). T_BI = T_BE T_EM T_IM_home^-1, the mount sitting as far above each indenter's top as the
CAD's homing() puts it (sphere top 46 mm, rod top 42.5 mm)."""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PLAN = HERE / "20261022-plan_sphere_measured_rod.json"     # T_EM with the measured -92.15 deg turn
TOP = {"sphere": 46.0, "rod": 42.5}
Rz_m90 = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # columns: x_M = -y_E, y_M = x_E, z_M = z_E


def T_EM_measured(plan):
    return np.array(plan["geometry"]["T_EM"], float)        # the measured plan's: turned -92.15 deg


def indenter_pose(T_BE, kind, plan, T_EM):
    Th = np.array(plan["geometry"]["T_IM_home"], float)
    Th[2, 3] += TOP[kind] - TOP["sphere"]
    T = np.array(T_BE) @ T_EM @ np.linalg.inv(Th)
    R = T[:3, :3]
    return T, math.degrees(math.atan2(R[1, 0], R[0, 0])), math.degrees(math.acos(float(np.clip(R[2, 2], -1, 1))))


KEY = {"sphere": (-44.36, 30.0, 44.55), "rod": (-44.36, 30.0, 41.05)}   # the key at the socket's mouth (homing())


def tilt_parts(R):
    """The indenter's z axis in its own levelled frame (same yaw, z up): roll about x_I (the tool axis at homing) and pitch
    about y_I (the side axis), deg."""
    xl = np.array([R[0, 0], R[1, 0], 0.0])
    xl /= np.linalg.norm(xl)
    L = np.c_[xl, np.cross([0.0, 0.0, 1.0], xl), [0.0, 0.0, 1.0]]
    a, b, c = L.T @ R[:, 2]
    return L, math.degrees(math.atan2(-b, c)), math.degrees(math.atan2(a, c))


def levelled(T, kind):
    """The pose with its tilt set to 0, turned about the key (the seating fixes the key's place better than the angle)."""
    L, _, _ = tilt_parts(T[:3, :3])
    key_B = T[:3, :3] @ np.array(KEY[kind]) + T[:3, 3]
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = L, key_B - L @ np.array(KEY[kind])
    return out


if __name__ == "__main__":
    plan = json.loads(PLAN.read_text())
    T_EM = T_EM_measured(plan)
    recs = {r["label"]: r for r in map(json.loads, (HERE / "poses.jsonl").read_text().splitlines())}
    for label in sys.argv[1:]:
        kind = "rod" if "rod" in label else "sphere" if "sphere" in label else None
        for k in ([kind] if kind else ("sphere", "rod")):
            T, yaw, tilt = indenter_pose(recs[label]["T_tcp_mm"], k, plan, T_EM)
            _, roll, pitch = tilt_parts(T[:3, :3])
            Tl = levelled(T, k)
            print(f"{label} as the {k}: base origin at {np.round(T[:3, 3], 1)} mm, yaw {yaw:.1f} deg, tilt {tilt:.2f} deg "
                  f"= roll {roll:+.2f} about x_I (tool axis) and pitch {pitch:+.2f} about y_I (side)")
            print(f"    levelled about the key: origin at {np.round(Tl[:3, 3], 1)} mm, yaw {yaw:.1f} deg, tilt 0; "
                  f"{k} top at z {Tl[2, 3] + TOP[k]:.1f} mm, key at {np.round(Tl[:3, :3] @ np.array(KEY[k]) + Tl[:3, 3], 1)}")
    print("plan: sphere indenter at (0, -450, 56) mm, yaw -90 deg (x along the base's -y), tilt 0")
