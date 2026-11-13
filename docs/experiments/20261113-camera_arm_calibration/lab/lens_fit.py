"""Each D435 colour camera's lens from the recorded arm poses' ChArUco corners (burst means): OpenCV's
calibrateCameraCharucoExtended with k1 k2 p1 p2 k3, from the factory model as the first guess. Prints the focal
lengths, principal point and distortion with their standard deviations, the RMS reprojection, each view's RMS, and
how far from the principal point the corners reach (the hex block sat 420-500 px out on 11-09). The square is the
printed 15.0 mm (owner, 11-13: "the square size is exactly as it was printed").

    /usr/bin/python3 lens_fit.py OUTDIR [--fix-k3] [--json F]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("outdir")
ap.add_argument("--fix-k3", action="store_true")
ap.add_argument("--json", default=None)
ap.add_argument("--square", type=float, default=15.0)
a = ap.parse_args()
A = cv2.aruco
BOARD = A.CharucoBoard_create(7, 7, a.square, 0.7 * a.square, A.getPredefinedDictionary(A.DICT_5X5_50))
recs = [json.loads(l) for l in open(Path(a.outdir) / "poses.jsonl")]
res = {}
for sn in sorted(recs[0]["cams"]):
    I = recs[0]["cams"][sn]["streams"]["color"]
    K0 = np.array([[I["fx"], 0, I["cx"]], [0, I["fy"], I["cy"]], [0, 0, 1]], float)
    cc, ci, used = [], [], []
    for r in recs:
        c = r["cams"][sn]["color"]
        if len(c["corner_ids"]) >= 8:
            cc.append(np.array(c["corners_mean"], np.float32).reshape(-1, 1, 2))
            ci.append(np.array(c["corner_ids"], np.int32).reshape(-1, 1))
            used.append(r["pose"])
    flags = cv2.CALIB_USE_INTRINSIC_GUESS | (cv2.CALIB_FIX_K3 if a.fix_k3 else 0)
    rms, K, D, rv, tv, sdi, sde, per = cv2.aruco.calibrateCameraCharucoExtended(
        cc, ci, BOARD, (I["w"], I["h"]), K0.copy(), np.zeros(5), flags=flags)
    sdi = sdi.ravel()
    xy = np.concatenate([c.reshape(-1, 2) for c in cc])
    rad = np.linalg.norm(xy - K[:2, 2], axis=1)
    dist = [float(np.linalg.norm(t)) for t in tv]
    res[sn] = dict(poses=used, rms_px=float(rms), K=K.tolist(), dist=D.ravel().tolist(),
                   sd=dict(fx=sdi[0], fy=sdi[1], cx=sdi[2], cy=sdi[3], k1=sdi[4], k2=sdi[5], p1=sdi[6], p2=sdi[7], k3=sdi[8]),
                   per_view_rms=per.ravel().tolist(), radius_max_px=float(rad.max()), radius_p90_px=float(np.percentile(rad, 90)),
                   board_dist_mm=dist)
    print(f"{sn}: {len(used)} views, RMS {rms:.3f} px; fx {K[0, 0]:.1f}+-{sdi[0]:.1f} fy {K[1, 1]:.1f}+-{sdi[1]:.1f} "
          f"(factory {I['fx']:.1f}, {I['fy']:.1f}); cx {K[0, 2]:.1f}+-{sdi[2]:.1f} cy {K[1, 2]:.1f}+-{sdi[3]:.1f} "
          f"(factory {I['cx']:.1f}, {I['cy']:.1f})")
    print(f"   k1 {D.ravel()[0]:+.4f}+-{sdi[4]:.4f} k2 {D.ravel()[1]:+.4f}+-{sdi[5]:.4f} p1 {D.ravel()[2]:+.5f}+-{sdi[6]:.5f} "
          f"p2 {D.ravel()[3]:+.5f}+-{sdi[7]:.5f} k3 {D.ravel()[4]:+.4f}+-{sdi[8]:.4f}; corners reach {rad.max():.0f} px "
          f"(90 % within {np.percentile(rad, 90):.0f}); board {min(dist):.0f}-{max(dist):.0f} mm; "
          f"per-view RMS {np.round(per.ravel(), 3).tolist()}")
if a.json:
    Path(a.json).write_text(json.dumps(res, indent=1, default=float))
