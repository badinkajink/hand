"""Each D435 colour camera's lens together with the paper ChArUco board's 3D shape, from the hand-held views
(lens_views.py) and optionally the arm poses' views (record_pose.py), by bundle adjustment over the corners' pixels.

The board (7 x 7 squares of 15.0 mm printed, owner 11-13: "exactly as it was printed") is not flat. Its 36 inner
corners are free in 3D, shared by every view of both cameras. The gauge (the board frame's 6 DoF and its scale) is held
by stiff priors: zero mean offset, zero mean rotation, zero mean in-plane stretch, so the board keeps its printed size
on average. Each view has its own pose. Each camera has fx, fy, cx, cy, k1, k2, p1, p2, k3 (--no-k3 fixes k3 = 0,
--no-tangential fixes p1 = p2 = 0). Views whose board normal lies more than MAX_TILT from the view axis are dropped.
Soft L1 loss with a 1 px scale. Reports the RMS per camera, the board's out-of-plane shape, the parameters' standard
deviations (from the Jacobian at the solution, scaled by the residual RMS) and the flat-board fit for comparison.

    /usr/bin/python3 lens_board_fit.py OUTDIR [--views lens_paper,lens_paper_b] [--arm] [--json F]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))
from sr2hand.geometry.rotations import exp_so3  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("outdir")
ap.add_argument("--views", default="lens_paper,lens_paper_b")
ap.add_argument("--arm", action="store_true", help="also use the arm poses' colour corners as free-pose views")
ap.add_argument("--no-k3", action="store_true")
ap.add_argument("--no-tangential", action="store_true")
ap.add_argument("--max-tilt", type=float, default=70.0)
ap.add_argument("--flat", action="store_true", help="hold the board flat (the printed model)")
ap.add_argument("--json", default=None)
a = ap.parse_args()
out = Path(a.outdir)
A = cv2.aruco
BOARD = A.CharucoBoard_create(7, 7, 15.0, 10.5, A.getPredefinedDictionary(A.DICT_5X5_50))
P0 = np.array(BOARD.chessboardCorners, float)                     # 36 x 3, mm, z = 0
NP = len(P0)

# ---- views: (camera, ids, pixels)
views = []
for d in a.views.split(","):
    for l in open(out / d / "views.jsonl"):
        v = json.loads(l)
        views.append((v["cam"], np.array(v["ids"]), np.array(v["corners"], float), f"{d}:{v['view']}"))
if a.arm:
    for l in open(out / "poses.jsonl"):
        r = json.loads(l)
        for sn, c in r["cams"].items():
            if len(c["color"]["corner_ids"]) >= 8:
                views.append((sn, np.array(c["color"]["corner_ids"]), np.array(c["color"]["corners_mean"], float),
                              f"arm:{r['pose']}"))
cams = sorted({v[0] for v in views})
intr0 = {}
for d in a.views.split(","):
    intr0.update(json.load(open(out / d / "session.json"))["intrinsics"])

# ---- initial lens: OpenCV's flat-board fit per camera; initial view poses from it
lens0, poses0, keep = {}, [], []
for sn in cams:
    I = intr0[sn]
    K0 = np.array([[I["fx"], 0, I["cx"]], [0, I["fy"], I["cy"]], [0, 0, 1.0]])
    vs = [i for i, v in enumerate(views) if v[0] == sn and len(v[1]) >= 8]
    rms, K, D, rv, tv = cv2.aruco.calibrateCameraCharuco(
        [views[i][2].astype(np.float32).reshape(-1, 1, 2) for i in vs],
        [views[i][1].astype(np.int32).reshape(-1, 1) for i in vs], BOARD, (1920, 1080), K0, np.zeros(5),
        flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K3)
    lens0[sn] = dict(K=K, D=D.ravel(), rms_flat=rms)
    for i, r, t in zip(vs, rv, tv):
        tilt = np.degrees(np.arccos(abs(cv2.Rodrigues(r)[0][2, 2])))
        if tilt <= a.max_tilt:
            keep.append(i)
            poses0.append(np.r_[r.ravel(), t.ravel()])
views = [views[i] for i in keep]
NV = len(views)
print(f"{NV} views kept (tilt <= {a.max_tilt} deg): " + ", ".join(f"{sn} {sum(v[0] == sn for v in views)}" for sn in cams))
for sn in cams:
    print(f"  flat-board OpenCV fit {sn}: RMS {lens0[sn]['rms_flat']:.3f} px")

# ---- parameter vector: lens (9 per camera) | view poses (6 each) | board offsets (3 per corner)
NL = 9
free_lens = np.ones(NL, bool)
if a.no_tangential:
    free_lens[6:8] = False
if a.no_k3:
    free_lens[8] = False


def lens_vec(sn):
    K, D = lens0[sn]["K"], lens0[sn]["D"]
    return np.r_[K[0, 0], K[1, 1], K[0, 2], K[1, 2], D[0], D[1], D[2], D[3], 0.0]


x0 = np.r_[np.concatenate([lens_vec(sn) for sn in cams]), np.concatenate(poses0), np.zeros(3 * NP)]
iL, iV, iB = 0, NL * len(cams), NL * len(cams) + 6 * NV
cam_ix = {sn: k for k, sn in enumerate(cams)}
nres = sum(2 * len(v[1]) for v in views)
W_GAUGE, SIG_OFF = 1e3, 3.0                                       # prior rows: gauge (stiff), offsets (3 mm, weak)


def unpack(x):
    L = x[iL:iV].reshape(len(cams), NL).copy()           # fixed terms are held at 0 by the bounds
    return L, x[iV:iB].reshape(NV, 6), (np.zeros((NP, 3)) if a.flat else x[iB:].reshape(NP, 3))


def residuals(x):
    L, V, dB = unpack(x)
    P = P0 + dB
    r = []
    for (sn, ids, px, _), pose in zip(views, V):
        l = L[cam_ix[sn]]
        K = np.array([[l[0], 0, l[2]], [0, l[1], l[3]], [0, 0, 1.0]])
        proj, _ = cv2.projectPoints(P[ids], pose[:3], pose[3:], K, np.r_[l[4:8], l[8]])
        r.append((proj.reshape(-1, 2) - px).ravel())
    # gauge: mean offset 0, mean rotation 0 (sum P0 x d), mean in-plane stretch 0 (sum P0_xy . d_xy)
    c = P0 - P0.mean(0)
    g = np.r_[dB.mean(0), np.cross(c, dB).mean(0) / 50.0, (c[:, :2] * dB[:, :2]).sum(1).mean() / 50.0]
    return np.concatenate(r + [W_GAUGE * g, dB.ravel() / SIG_OFF])


def sparsity():
    S = lil_matrix((nres + 7 + 3 * NP, len(x0)), dtype=int)
    row = 0
    for k, (sn, ids, px, _) in enumerate(views):
        n = 2 * len(ids)
        ci = cam_ix[sn]
        S[row:row + n, iL + NL * ci:iL + NL * (ci + 1)] = 1
        S[row:row + n, iV + 6 * k:iV + 6 * k + 6] = 1
        for m, j in enumerate(ids):
            S[row + 2 * m:row + 2 * m + 2, iB + 3 * j:iB + 3 * j + 3] = 1
        row += n
    S[row:row + 7, iB:] = 1
    for j in range(3 * NP):
        S[row + 7 + j, iB + j] = 1
    return S


lo = np.full(len(x0), -np.inf)
hi = np.full(len(x0), np.inf)
for k in range(len(cams)):
    for p in np.where(~free_lens)[0]:
        x0[iL + NL * k + p] = 0.0
        lo[iL + NL * k + p], hi[iL + NL * k + p] = -1e-12, 1e-12
if a.flat:
    lo[iB:], hi[iB:] = -1e-12, 1e-12
sol = least_squares(residuals, x0, jac_sparsity=sparsity(), loss="soft_l1", f_scale=1.0, x_scale="jac",
                    bounds=(lo, hi), max_nfev=200, verbose=0)
L, V, dB = unpack(sol.x)
r = residuals(sol.x)[:nres]
J = sol.jac.toarray() if hasattr(sol.jac, "toarray") else sol.jac
res = dict(views=NV, cams={}, board_offsets_mm=dB.tolist(), flat=a.flat)
row = 0
per_cam = {sn: [] for sn in cams}
for sn, ids, px, name in views:
    n = 2 * len(ids)
    per_cam[sn].append(r[row:row + n])
    row += n
dof = max(nres - len(sol.x), 1)
s2 = float(np.sum(r ** 2) / dof)
try:
    cov = np.linalg.pinv(J.T @ J) * s2
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))
except np.linalg.LinAlgError:
    sd = np.full(len(sol.x), np.nan)
names = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"]
for k, sn in enumerate(cams):
    e = np.concatenate(per_cam[sn]).reshape(-1, 2)
    l, s = L[k], sd[iL + NL * k:iL + NL * (k + 1)]
    res["cams"][sn] = dict(rms_px=float(np.sqrt((e ** 2).sum(1).mean())), lens={n: float(v) for n, v in zip(names, l)},
                           sd={n: float(v) for n, v in zip(names, s)}, n_views=len(per_cam[sn]))
    print(f"{sn}: {len(per_cam[sn])} views, RMS {res['cams'][sn]['rms_px']:.3f} px")
    print("   " + "  ".join(f"{n} {v:.4f}+-{u:.4f}" if abs(v) < 10 else f"{n} {v:.1f}+-{u:.1f}" for n, v, u in zip(names, l, s)))
c = P0 - P0.mean(0)
z = (P0 + dB)[:, 2]
A_ = np.c_[c[:, :2], np.ones(NP)]
zr = z - A_ @ np.linalg.lstsq(A_, z, rcond=None)[0]
print(f"board: out-of-plane RMS {np.sqrt(np.mean(zr ** 2)):.3f} mm about its best plane, range {np.ptp(zr):.3f} mm; "
      f"in-plane offsets RMS {np.sqrt(np.mean((dB[:, :2] ** 2).sum(1))):.3f} mm")
print("   out-of-plane by corner (mm, rows as detected):")
print(np.round(zr.reshape(6, 6), 2))
res["board_out_of_plane_rms_mm"] = float(np.sqrt(np.mean(zr ** 2)))
if a.json:
    Path(a.json).write_text(json.dumps(res, indent=1))
