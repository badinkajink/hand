"""Checks of the calibration on data it was not fitted to: the hex block on the table and the D435 depth.

1. Hex block. In every arm pose's first colour frame of each camera the block's tags are detected
   (sr2hand.tracking.pose.detect) and the block solved (pose.solve_body, tags beyond 65 deg dropped) with the printed
   map HexBlock(across_flats=29.985, tag_size=13.33), once with the factory lens and once with the fitted one. With
   each camera's pose in the base (handeye_fit.py --json, fitted with the same lens), the block's centre is put in the
   base. The block did not move, so the two cameras must agree; their difference and each camera's spread over the
   poses are printed.
2. Depth at the block: each solved tag face centre is put into the depth frame (the factory depth-to-colour
   extrinsic) and compared with the burst median of the raw depth there (5 x 5 px median), depth minus solve.
3. Depth at the board, without the colour lens: the board's corners in IR1 with the projector off (the depth stream's
   own frame and factory intrinsics) solve the board's pose with its fitted shape (lens_board_fit.py); the raw depth
   at those pixels minus the solved z, per arm pose and its distance.

    /usr/bin/python3 block_depth_check.py OUTDIR LENS_NEW.json HE_NEW.json LENS_FACTORY.json HE_FACTORY.json
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))
from sr2hand.tracking import pose as tp  # noqa: E402
from sr2hand.tracking.tags import HexBlock  # noqa: E402

out, lens_new, he_new, lens_fac, he_fac = sys.argv[1:6]
out = Path(out)
recs = [json.loads(l) for l in open(out / "poses.jsonl")]
TM = HexBlock(across_flats=29.985, tag_size=13.33).tag_map()
A = cv2.aruco
BOARD = A.CharucoBoard_create(7, 7, 15.0, 10.5, A.getPredefinedDictionary(A.DICT_5X5_50))
det = tp.detector()


def cam_of(L, sn):
    l = json.load(open(L))["cams"][sn]["lens"]
    K = np.array([[l["fx"], 0, l["cx"]], [0, l["fy"], l["cy"]], [0, 0, 1.0]])
    return tp.Camera(K, np.array([l["k1"], l["k2"], l["p1"], l["p2"], l["k3"]], float), 1920, 1080)


def inv(M):
    R, t = M[:3, :3], M[:3, 3]
    N = np.eye(4)
    N[:3, :3], N[:3, 3] = R.T, -R.T @ t
    return N


def depth_at(Z, uv, r=2):
    u, v = int(round(uv[0])), int(round(uv[1]))
    w = Z[max(v - r, 0):v + r + 1, max(u - r, 0):u + r + 1].astype(float)
    w = w[w > 0]
    return np.median(w) if w.size else np.nan


sns = sorted(recs[0]["cams"])
raw = {}
for r in recs:
    for sn in sns:
        raw[(r["pose"], sn)] = np.median(np.load(out / f"p{r['pose']:02d}" / f"{sn}_depth_raw.npz")["z16"].astype(float), 0)
print("1-2. hex block (HexBlock 29.985 / 13.33), centre in the base frame and depth minus solve at the tag faces")
for tag, L, H in (("factory lens", lens_fac, he_fac), ("fitted lens ", lens_new, he_new)):
    he = json.load(open(H))
    cen = {sn: [] for sn in sns}
    dz = {sn: [] for sn in sns}
    for r in recs:
        for sn in sns:
            cam = cam_of(L, sn)
            gray = cv2.imread(str(out / f"p{r['pose']:02d}" / f"{sn}_color.png"), 0)
            obs = tp.detect(det, gray, ids=TM.ids())
            T_CBk, info = tp.solve_body(obs, TM, cam)
            if T_CBk is None:
                continue
            T_CB = np.array(he["cams"][sn]["T_CB"])
            cen[sn].append((inv(T_CB) @ T_CBk)[:3, 3])
            st = r["cams"][sn]["streams"]
            Rdc, tdc = np.array(st["depth_to_color"]["R"]), 1000 * np.array(st["depth_to_color"]["t"])
            di = st["depth"]
            Kd = np.array([[di["fx"], 0, di["cx"]], [0, di["fy"], di["cy"]], [0, 0, 1.0]])
            for i in info["tags"]:
                c, rr, dd, n = TM.frames[i]
                pc = T_CBk[:3, :3] @ c + T_CBk[:3, 3]                       # face centre, colour frame (mm)
                pd = Rdc.T @ (pc - tdc)                                       # depth frame
                uv = (Kd @ pd)[:2] / pd[2]
                z = depth_at(raw[(r["pose"], sn)], uv) * st["depth_scale"] * 1000
                if np.isfinite(z):
                    dz[sn].append((z - pd[2], pd[2]))
    c0, c1 = np.array(cen[sns[0]]), np.array(cen[sns[1]])
    print(f"  {tag}: block centre {sns[0]} {np.round(c0.mean(0), 2).tolist()} (sd {np.round(c0.std(0), 2).tolist()}), "
          f"{sns[1]} {np.round(c1.mean(0), 2).tolist()} (sd {np.round(c1.std(0), 2).tolist()}) mm; "
          f"cameras differ by {np.linalg.norm(c0.mean(0) - c1.mean(0)):.2f} mm")
    for sn in sns:
        d = np.array(dz[sn]).reshape(-1, 2)
        if not len(d):
            print(f"     {sn}: no valid depth at its tag faces (the block nearer than the D435's minimum range)")
            continue
        print(f"     depth - solve at {sn}'s tag faces: {np.median(d[:, 0]):+.2f} mm median "
              f"(IQR {np.percentile(d[:, 0], 25):+.2f} to {np.percentile(d[:, 0], 75):+.2f}, n {len(d)}, "
              f"faces at {d[:, 1].min():.0f}-{d[:, 1].max():.0f} mm)")
print("3. depth minus IR1 board solve (projector off for IR1; board shape fitted; no colour lens)")
Lb = json.load(open(lens_new))
P = np.array(BOARD.chessboardCorners, float) + np.array(Lb["board_offsets_mm"])
for sn in sns:
    rows = []
    for r in recs:
        c = r["cams"][sn]["ir1_off"]
        if len(c["corner_ids"]) < 12:
            continue
        ids, uv = np.array(c["corner_ids"]), np.array(c["corners_mean"], float)
        ii = r["cams"][sn]["streams"]["ir1"]
        K = np.array([[ii["fx"], 0, ii["cx"]], [0, ii["fy"], ii["cy"]], [0, 0, 1.0]])
        ok, rv, tv = cv2.solvePnP(P[ids], uv, K, None, flags=cv2.SOLVEPNP_SQPNP)
        ok, rv, tv = cv2.solvePnP(P[ids], uv, K, None, rv, tv, True, cv2.SOLVEPNP_ITERATIVE)
        Xc = (cv2.Rodrigues(rv)[0] @ P[ids].T).T + tv.ravel()
        sc = r["cams"][sn]["streams"]["depth_scale"] * 1000
        z = np.array([depth_at(raw[(r["pose"], sn)], q, r=1) * sc for q in uv])
        m = np.isfinite(z)
        if m.sum() >= 6:
            rows.append((r["pose"], np.median(Xc[m, 2]), np.median(z[m] - Xc[m, 2]), m.sum()))
    rows = np.array(rows)
    print(f"  {sn}: depth - solve {np.median(rows[:, 2]):+.2f} mm median over {len(rows)} poses "
          f"(range {rows[:, 2].min():+.2f} to {rows[:, 2].max():+.2f}); board at {rows[:, 1].min():.0f}-{rows[:, 1].max():.0f} mm; "
          f"slope {np.polyfit(rows[:, 1], rows[:, 2], 1)[0] * 100:+.2f} mm per 100 mm")
    print("     per pose (pose: z mm, depth-solve mm): " + " ".join(f"{int(p)}:{z:.0f},{d:+.2f}" for p, z, d, n in rows))
