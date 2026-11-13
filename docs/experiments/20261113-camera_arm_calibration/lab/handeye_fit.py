"""Each D435 colour camera's pose in the UR5's base and the paper board's pose on the flange (tool0), from the arm
poses (record_pose.py), with each camera's lens and the board's shape fixed from lens_board_fit.py --json.

Per pose the flange comes from the joints through sr2hand.robot.ur_kinematics.fk (the nominal UR5 table), or with
--tcp from the controller's TCP pose (its calibrated kinematics plus the pendant's TCP offset, which the board's pose
on the flange absorbs). Model: x = project(lens, T_CB T_BE(q) T_EP P_j). Initial values: OpenCV calibrateHandEye in
the eye-to-hand arrangement for each camera, then all unknowns refined together by least squares over the corners'
pixels (soft L1, 1 px scale): 6 per camera, 6 for the board on the flange (--fit-scale: and one scale factor of the board, which the arm's metric then sets). Prints the RMS per camera and per pose, each
camera's position in the base and its viewing axis, and the per-pose board-pose disagreement between the arm's route
and each camera's own PnP solve.

    /usr/bin/python3 handeye_fit.py OUTDIR LENS.json [--tcp] [--json F]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))
from sr2hand.geometry.rotations import exp_so3, log_so3  # noqa: E402
from sr2hand.robot import ur_kinematics as urk  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("outdir")
ap.add_argument("lens")
ap.add_argument("--tcp", action="store_true")
ap.add_argument("--json", default=None)
ap.add_argument("--fit-scale", action="store_true", help="also fit one scale factor of the board (the arm's metric checks its print)")
a = ap.parse_args()
L = json.load(open(a.lens))
A = cv2.aruco
BOARD = A.CharucoBoard_create(7, 7, 15.0, 10.5, A.getPredefinedDictionary(A.DICT_5X5_50))
P = np.array(BOARD.chessboardCorners, float) + np.array(L["board_offsets_mm"])
recs = [json.loads(l) for l in open(Path(a.outdir) / "poses.jsonl")]
cams = sorted(L["cams"])


def T(R, t):
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, np.ravel(t)
    return M


def inv(M):
    return T(M[:3, :3].T, -M[:3, :3].T @ M[:3, 3])


def flange(r):
    if a.tcp:
        p = np.array(r["arm_before"]["tcp"])
        return T(exp_so3(p[3:]), 1000 * p[:3])
    return urk.fk(np.array(r["arm_before"]["q"]))


lens = {}
for sn in cams:
    l = L["cams"][sn]["lens"]
    lens[sn] = (np.array([[l["fx"], 0, l["cx"]], [0, l["fy"], l["cy"]], [0, 0, 1.0]]),
                np.array([l["k1"], l["k2"], l["p1"], l["p2"], l["k3"]], float))
T_BE = [flange(r) for r in recs]
obs = []                                             # (pose index, camera, ids, pixels)
pnp = {}
for i, r in enumerate(recs):
    for sn in cams:
        c = r["cams"][sn]["color"]
        if len(c["corner_ids"]) < 8:
            continue
        ids, xy = np.array(c["corner_ids"]), np.array(c["corners_mean"], float)
        K, D = lens[sn]
        ok, rv, tv = cv2.solvePnP(P[ids], xy, K, D, flags=cv2.SOLVEPNP_SQPNP)
        ok, rv, tv = cv2.solvePnP(P[ids], xy, K, D, rv, tv, True, cv2.SOLVEPNP_ITERATIVE)
        pnp[(i, sn)] = T(cv2.Rodrigues(rv)[0], tv)
        obs.append((i, sn, ids, xy))

# ---- initial values: eye-to-hand through calibrateHandEye (pass base<-gripper inverted)
init = {}
for sn in cams:
    ks = [i for i in range(len(recs)) if (i, sn) in pnp]
    Rg = [inv(T_BE[i])[:3, :3] for i in ks]
    tg = [inv(T_BE[i])[:3, 3] for i in ks]
    Rt = [pnp[(i, sn)][:3, :3] for i in ks]
    tt = [pnp[(i, sn)][:3, 3] for i in ks]
    R_cb, t_cb = cv2.calibrateHandEye(Rg, tg, Rt, tt, method=cv2.CALIB_HAND_EYE_PARK)
    T_BC = T(R_cb, t_cb)                             # camera in base
    T_CB = inv(T_BC)
    T_EP = [inv(T_BE[i]) @ T_BC @ pnp[(i, sn)] for i in ks]
    init[sn] = (T_CB, T_EP[0])
T_EP0 = init[cams[0]][1]


def pack(T_CBs, T_EP):
    return np.concatenate([np.r_[log_so3(M[:3, :3]), M[:3, 3]] for M in T_CBs] + [np.r_[log_so3(T_EP[:3, :3]), T_EP[:3, 3]]])


NX = 6 * (len(cams) + 1)


def unpack(x):
    Ms = [T(exp_so3(x[6 * k:6 * k + 3]), x[6 * k + 3:6 * k + 6]) for k in range(len(cams) + 1)]
    return dict(zip(cams, Ms[:-1])), Ms[-1]


def scale(x):
    return 1.0 + 1e-3 * x[NX] if a.fit_scale else 1.0


def residuals(x):
    T_CB, T_EP = unpack(x)
    Ps = scale(x) * P
    r = []
    for i, sn, ids, xy in obs:
        M = T_CB[sn] @ T_BE[i] @ T_EP
        K, D = lens[sn]
        pr, _ = cv2.projectPoints(Ps[ids], cv2.Rodrigues(M[:3, :3])[0], M[:3, 3], K, D)
        r.append((pr.reshape(-1, 2) - xy).ravel())
    return np.concatenate(r)


x0 = pack([init[sn][0] for sn in cams], T_EP0)
if a.fit_scale:
    x0 = np.r_[x0, 0.0]
sol = least_squares(residuals, x0, loss="soft_l1", f_scale=1.0, x_scale="jac")
T_CB, T_EP = unpack(sol.x)
r = residuals(sol.x)
s_fit, s_sd = scale(sol.x), None
if a.fit_scale:
    J = sol.jac
    cov = np.linalg.pinv(J.T @ J) * np.sum(r ** 2) / max(len(r) - len(sol.x), 1)
    s_sd = 1e-3 * float(np.sqrt(cov[NX, NX]))
    print(f"board scale from the arm: {s_fit:.5f} +- {s_sd:.5f} (square {15.0 * s_fit:.3f} +- {15.0 * s_sd:.3f} mm)")
print(f"kinematics: {'controller TCP pose' if a.tcp else 'ur_kinematics.fk(q), nominal UR5 table'}; "
      f"{len(recs)} poses, {len(obs)} camera views")
row, per = 0, {}
for i, sn, ids, xy in obs:
    n = 2 * len(ids)
    per.setdefault(sn, []).append((i, np.sqrt(np.mean(r[row:row + n].reshape(-1, 2) ** 2) * 2)))
    row += n
out = dict(kinematics="tcp" if a.tcp else "fk", cams={}, T_EP=T_EP.tolist(), board_scale=s_fit, board_scale_sd=s_sd)
for sn in cams:
    e = np.array([v for _, v in per[sn]])
    T_BC = inv(T_CB[sn])
    ax = T_BC[:3, 2]
    # board pose from the arm vs the camera's own PnP solve
    dt, dr = [], []
    for i, _ in per[sn]:
        Ma, Mp = T_CB[sn] @ T_BE[i] @ T_EP, pnp[(i, sn)]
        c = P.mean(0)
        dt.append(np.linalg.norm((Ma[:3, :3] @ c + Ma[:3, 3]) - (Mp[:3, :3] @ c + Mp[:3, 3])))
        dr.append(np.degrees(np.linalg.norm(log_so3(Ma[:3, :3].T @ Mp[:3, :3]))))
    print(f"{sn}: RMS {np.sqrt(np.mean(e ** 2)):.3f} px over {len(e)} poses (worst {e.max():.3f} at pose "
          f"{per[sn][int(e.argmax())][0]}); camera at {np.round(T_BC[:3, 3], 1).tolist()} mm in base, axis "
          f"{np.round(ax, 3).tolist()}; board centre arm vs PnP {np.median(dt):.2f} mm median ({max(dt):.2f} max), "
          f"{np.median(dr):.2f} deg median ({max(dr):.2f} max)")
    print("   per pose RMS px: " + " ".join(f"{i}:{v:.2f}" for i, v in per[sn]))
    out["cams"][sn] = dict(T_CB=T_CB[sn].tolist(), rms_px=float(np.sqrt(np.mean(e ** 2))),
                           per_pose=[(int(i), float(v)) for i, v in per[sn]], board_arm_vs_pnp_mm=dt, board_arm_vs_pnp_deg=dr)
d = np.linalg.norm(inv(T_CB[cams[0]])[:3, 3] - inv(T_CB[cams[1]])[:3, 3])
ang = np.degrees(np.arccos(np.clip(inv(T_CB[cams[0]])[:3, 2] @ inv(T_CB[cams[1]])[:3, 2], -1, 1)))
print(f"cameras {d:.1f} mm apart, axes {ang:.1f} deg apart; board centre in tool0 "
      f"{np.round(T_EP[:3, :3] @ P.mean(0) + T_EP[:3, 3], 1).tolist()} mm, board normal in tool0 {np.round(T_EP[:3, 2], 3).tolist()}")
if a.json:
    Path(a.json).write_text(json.dumps(out, indent=1, default=float))
