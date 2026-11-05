#!/usr/bin/env python3
"""Scratch (run with ~/miniconda3/bin/python, which has pupil_apriltags and cv2): detect tag36h11 id 6 in the rendered
D435 views and compare the detected pose with the pose computed from the CAD (analysis.json), and read the reference
tag's axes in the 2026-08-30 bench probe (docs/experiments/20260830-apriltag-tracking/probe_ir.png). Writes
validation.json."""
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pupil_apriltags as pa

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
an = json.load(open(HERE / "analysis.json"))
out = {}
det = pa.Detector(families="tag36h11", quad_decimate=1.0)
Td = np.array(an["variants"]["highest"]["T_A_from_depth_optical"]).reshape(4, 4)
Tc = np.array(an["variants"]["highest"]["T_A_from_color_optical"]).reshape(4, 4)
Ttag = np.array(an["T_A_from_tag_pos_face"]).reshape(4, 4)
for name, fn, w, h, fovy, Tcam in (("depth", "boxes_highest_depth_raw.png", 1280, 748, 58.0, Td),
                                    ("color", "boxes_highest_color_raw.png", 1280, 716, 42.0, Tc)):
    im = cv2.imread(str(HERE / fn), cv2.IMREAD_GRAYSCALE)
    f = (h / 2) / math.tan(math.radians(fovy / 2))
    pred = np.linalg.inv(Tcam) @ Ttag
    res = []
    for r in det.detect(im, estimate_tag_pose=True, camera_params=(f, f, w / 2, h / 2), tag_size=0.040):
        R, t = r.pose_R, r.pose_t.ravel() * 1000.0
        dR = R.T @ pred[:3, :3]
        ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1) / 2))))
        res.append(dict(id=int(r.tag_id), margin=round(float(r.decision_margin), 1), t_mm=np.round(t, 2).tolist(),
                        pred_t_mm=np.round(pred[:3, 3], 2).tolist(), rot_err_deg=round(ang, 2),
                        trans_err_mm=round(float(np.linalg.norm(t - pred[:3, 3])), 2)))
    out[f"render_{name}"] = dict(file=fn, f_px=round(f, 1), detections=res)
im = cv2.imread(str(REPO / "docs/experiments/20260830-apriltag-tracking/probe_ir.png"), cv2.IMREAD_GRAYSCALE)
f = 640 / math.tan(math.radians(43.5))
for r in det.detect(im, estimate_tag_pose=True, camera_params=(f, f, 640, 360), tag_size=0.040):
    out["probe_20260830"] = dict(id=int(r.tag_id), margin=round(float(r.decision_margin), 1),
                                 tag_x_in_cam=np.round(r.pose_R[:, 0], 3).tolist(),
                                 tag_y_in_cam=np.round(r.pose_R[:, 1], 3).tolist(),
                                 tag_z_in_cam=np.round(r.pose_R[:, 2], 3).tolist(),
                                 range_mm=round(float(np.linalg.norm(r.pose_t)) * 1000, 1),
                                 note="nominal intrinsics (87 deg HFOV at 1280 px); camera y is down, so tag x = (0,-1,0) "
                                      "means the tag's x axis points up in the image")
(HERE / "validation.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
