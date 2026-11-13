#!/usr/bin/env python3
"""Hand-eye calibration of a table-fixed camera with the board screwed to the hand at an unmeasured place: how the
error of the camera's pose depends on how far the hand turns between poses (owner, 2026-11-13: "it will be ok to
arbitrarily fix the checkerboard ... so long as cameras and affixed checkerboard do not move"; "we can really only move
up/down, left/right, with very, very minor rotations. will that be ok?").

Model: in pose i the board's pose in the camera is T_CB(i) = T_CW T_WE(i) T_EB, with W the UR5's base, E its flange
(tool0) and B the board. The unknowns T_CW (the base in the camera) and T_EB (the board on the flange) are fitted by
least squares to the reprojection of the board's corners in every pose, the bundle form of OpenCV's
calibrateRobotWorldHandEye. With the flange's orientation fixed, the board's offset on the flange and the camera's
position enter only as a sum, so translations alone leave three directions of the fit undetermined; turns of the hand
about two axes separate them.

Setup (mm): the paper ChArUco board of scripts/hardware/paper_board_print.py (6 x 6 inner corners, 15 mm), 130 mm to
the side of the flange's axis and 110 mm below the flange, facing a portrait D435 colour camera (fx = fy = 1397 px,
1080 x 1920) 400 mm away. Each calibration has N_POSES poses: the flange turned by a rotation vector whose three
components (about tool0's axes) are drawn from [-a, a], and shifted so that the board's centre lies within +-40 mm of
its start along each axis. In the case "translations, board measured" the board's position on tool0 is taken as
measured with calipers, MEASURED_MM per axis, and only its orientation is fitted. Noise: the corners
SIGMA_PX px; the arm SIGMA_ARM_MM and SIGMA_ARM_DEG per pose. Reported over TRIALS calibrations: the RMS error of the
camera's position and orientation in W, and of the palm's position in the camera (T_CP, with P 138.9 mm along tool0's
z), the quantity the tracker uses.

    ~/.claude/bin/resguard.sh run --mem 2G -- ~/miniconda3/bin/python scripts/hardware/handeye_rotation_sim.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/, for _paths
import _paths  # noqa: E402,F401
from sr2hand.geometry.rotations import exp_so3, frame_from_axis, rot_angle, rot_x, tf, tf_apply, tf_compose, tf_inv  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/experiments/20261113-tracking_prints/rows/handeye_rotation.json"
FX, W, H = 1397.0, 1080, 1920
SQUARE, NC = 15.0, 6
N_POSES, TRIALS, MOVE_MM = 20, 40, 40.0
SIGMA_PX, SIGMA_ARM_MM, SIGMA_ARM_DEG = 0.10, 0.05, 0.005
T_EB_MM = np.array([-130.0, 60.0, 110.0])      # the board on tool0 (z_E out of the flange, down)
P_IN_E_MM = np.array([0.0, 0.0, 138.9])        # the palm frame's origin on tool0's axis
CAM_DIST, CAM_DIR = 400.0, np.array([0.9, -0.35, -0.25])
CASES = ([("three axes", a) for a in (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0)]
         + [("flange axis only", a) for a in (10.0, 30.0)] + [("translations, board measured", 0.0)])
MEASURED_MM, MEASURED_DEG = 0.5, 0.5           # the board's place on the hand measured with calipers, per axis


def setup():
    """True T_WE0 (nominal flange), T_EB, T_CW and the board's corners (mm)."""
    T_WE0 = tf(rot_x(math.pi), np.array([0.0, 0.0, 400.0]))
    b0 = tf_apply(T_WE0, T_EB_MM)
    c = b0 + CAM_DIST * CAM_DIR / np.linalg.norm(CAM_DIR)
    z = (b0 - c) / np.linalg.norm(b0 - c)                      # the camera looks at the board (OpenCV: z forward)
    R_WC = frame_from_axis(z, hint=np.cross([0.0, 0.0, -1.0], z), which=2)  # x right, y down
    T_CW = tf_inv(tf(R_WC, c))
    R_WB = frame_from_axis(z, hint=R_WC[:, 0], which=2)       # the board faces the camera head-on in the nominal pose
    T_EB = tf(T_WE0[:3, :3].T @ R_WB, T_EB_MM)
    g = (np.arange(NC) - (NC - 1) / 2) * SQUARE
    pts = np.array([[x, y, 0.0] for y in g for x in g])
    return T_WE0, T_EB, T_CW, pts


def project(T_CW, T_WE, T_EB, pts):
    p = tf_apply(tf_compose(T_CW, T_WE, T_EB), pts)
    return np.stack([FX * p[:, 0] / p[:, 2] + (W - 1) / 2, FX * p[:, 1] / p[:, 2] + (H - 1) / 2], 1), p[:, 2]


def poses(rng, T_WE0, kind, a_deg):
    """Flange poses: the board's centre moved within +-MOVE_MM of its start along each axis (the flange shifted to
    follow the turn, so the board stays in the camera's view) and the flange turned about tool0's axes."""
    b0 = tf_apply(T_WE0, T_EB_MM)
    out = []
    for _ in range(N_POSES):
        d = rng.uniform(-MOVE_MM, MOVE_MM, 3)
        r = np.deg2rad(rng.uniform(-a_deg, a_deg, 3))
        if kind == "flange axis only":
            r[:2] = 0.0
        R = T_WE0[:3, :3] @ exp_so3(r)
        out.append(tf(R, b0 + d - R @ T_EB_MM))
    return out


def perturb(T, rng, deg, mm):
    return tf(exp_so3(np.deg2rad(rng.normal(0.0, deg, 3))) @ T[:3, :3], T[:3, 3] + rng.normal(0.0, mm, 3))


def fit(obs, T_WEm, pts, T_CW0, T_EB0, fix_board):
    """Least squares over the corners' reprojection, from (T_CW0, T_EB0); the board's position on tool0 held at
    T_EB0's when fix_board (its orientation is fitted)."""
    def unpack(x):
        T_CW = tf(exp_so3(x[:3]) @ T_CW0[:3, :3], T_CW0[:3, 3] + x[3:6])
        t_EB = T_EB0[:3, 3] if fix_board else T_EB0[:3, 3] + x[9:12]
        return T_CW, tf(exp_so3(x[6:9]) @ T_EB0[:3, :3], t_EB)

    def res(x):
        T_CW, T_EB = unpack(x)
        return np.concatenate([(project(T_CW, Te, T_EB, pts)[0] - u).ravel() for Te, u in zip(T_WEm, obs)])

    n = 9 if fix_board else 12
    sol = least_squares(res, np.zeros(n), method="lm", x_scale=np.r_[[1e-3] * 3, [1.0] * 3, [1e-3] * 3, [1.0] * 3][:n])
    J = sol.jac
    sv = np.linalg.svd(J, compute_uv=False)
    return unpack(sol.x), sv


def main():
    T_WE0, T_EB, T_CW, pts = setup()
    T_EP = tf(np.eye(3), P_IN_E_MM)
    c_true = tf_inv(T_CW)[:3, 3]
    CP_true = tf_compose(T_CW, T_WE0, T_EP)
    rows = []
    for kind, a in CASES:
        rng = np.random.default_rng(20261113)
        e_pos, e_rot, e_cp, conds, seen = [], [], [], [], []
        for _t in range(TRIALS):
            Ts = poses(rng, T_WE0, kind, a)
            obs, ok = [], True
            for Te in Ts:
                u, zc = project(T_CW, Te, T_EB, pts)
                ok &= bool((zc > 0).all() and (u[:, 0] > 0).all() and (u[:, 0] < W).all() and (u[:, 1] > 0).all()
                           and (u[:, 1] < H).all())
                obs.append(u + rng.normal(0.0, SIGMA_PX, u.shape))
            seen.append(ok)
            T_WEm = [perturb(Te, rng, SIGMA_ARM_DEG, SIGMA_ARM_MM) for Te in Ts]
            fix = kind == "translations, board measured"
            T_EB0 = perturb(T_EB, rng, MEASURED_DEG, MEASURED_MM) if fix else perturb(T_EB, rng, 1.0, 10.0)
            (T_CWe, _T_EBe), sv = fit(obs, T_WEm, pts, perturb(T_CW, rng, 1.0, 10.0), T_EB0, fix)
            e_pos.append(np.linalg.norm(tf_inv(T_CWe)[:3, 3] - c_true))
            e_rot.append(math.degrees(rot_angle(T_CWe[:3, :3], T_CW[:3, :3])))
            e_cp.append(np.linalg.norm(tf_compose(T_CWe, T_WE0, T_EP)[:3, 3] - CP_true[:3, 3]))
            conds.append(float(sv[-1] / sv[0]))
        rms = (lambda v: float(np.sqrt(np.mean(np.square(v)))))
        determined = min(conds) > 1e-6
        row = {"rotation": kind, "range_deg": a, "trials": TRIALS, "poses": N_POSES, "all_corners_in_view": bool(all(seen)),
               "determined": determined, "min_rel_singular_value": float(np.median(conds)),
               "camera_pos_rms_mm": round(rms(e_pos), 3) if determined else None,
               "camera_rot_rms_deg": round(rms(e_rot), 4),
               "palm_in_camera_pos_rms_mm": round(rms(e_cp), 3) if determined else None}
        rows.append(row)
        print(row, flush=True)
    rep = {"what": "hand-eye calibration error against the hand's rotation range, board at an unmeasured place",
           "script": "scripts/hardware/handeye_rotation_sim.py",
           "setup": {"fx_px": FX, "image_px": [W, H], "corners": NC * NC, "square_mm": SQUARE, "poses": N_POSES,
                     "move_mm": MOVE_MM, "sigma_px": SIGMA_PX, "sigma_arm_mm": SIGMA_ARM_MM,
                     "sigma_arm_deg": SIGMA_ARM_DEG, "board_on_tool0_mm": T_EB_MM.tolist(), "camera_dist_mm": CAM_DIST,
                     "board_measured_mm_deg": [MEASURED_MM, MEASURED_DEG]},
           "rows": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1))
    print("wrote", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
