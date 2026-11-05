#!/usr/bin/env python3
"""tag36h11 id 6 in the owner's side photo of 2026-11-04 and in a frontal render of the full model (run with
~/miniconda3/bin/python, which has pupil_apriltags and cv2). Writes tag_in_photo.json beside this file. The camera's focal
length is the fit of scripts/cad_v3_film.py clocking on the other photo, taken with the same phone at the same size.

    ~/miniconda3/bin/python docs/experiments/20261104-cad_v3_hand/analysis/tag_in_photo.py
"""
import json
from pathlib import Path

import cv2
import numpy as np
from pupil_apriltags import Detector

HERE = Path(__file__).resolve().parent
MEDIA = HERE.parent / "media"
SIZE = 0.040


def detect(path, f):
    im = cv2.imread(str(path))
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    cam = (f, f, im.shape[1] / 2, im.shape[0] / 2)
    res = Detector(families="tag36h11", quad_decimate=2.0).detect(g, estimate_tag_pose=True, camera_params=cam, tag_size=SIZE)
    out = []
    for r in res:
        x = r.corners[1] - r.corners[0]                     # the corners run (-1,1), (1,1), (1,-1), (-1,-1) in the tag's frame
        y = r.corners[0] - r.corners[3]
        out.append(dict(id=int(r.tag_id), margin=round(float(r.decision_margin), 1), centre_px=r.center.round(1).tolist(),
                        corners_px=r.corners.round(1).tolist(), R_cam_tag=np.asarray(r.pose_R).round(6).tolist(),
                        t_cam_tag_m=np.asarray(r.pose_t).ravel().round(5).tolist(),
                        x_from_image_up_deg=round(float(np.degrees(np.arctan2(x[0], -x[1]))), 1),
                        y_from_image_right_deg=round(float(np.degrees(np.arctan2(y[1], y[0]))), 1),
                        side_px=round(float(np.mean([np.linalg.norm(x), np.linalg.norm(y)])), 1)))
    return out


clock = json.loads((HERE.parent / "20261104-clocking_from_photo.json").read_text())
J = dict(photo=detect(MEDIA / "20261104-owner_photo_side.jpg", clock["fit"]["f_px"]), f_px=clock["fit"]["f_px"], tag_size_m=SIZE)
render = MEDIA / "20261104-tag_render.png"
if render.exists():
    meta = json.loads((MEDIA / "20261104-tag_render.json").read_text())
    J["render"] = detect(render, meta["f_px"])
    J["render_truth"] = meta
(HERE / "tag_in_photo.json").write_text(json.dumps(J, indent=1) + "\n")
print(json.dumps({k: (v if k != "photo" else [dict(d, corners_px=None) for d in v]) for k, v in J.items()}, indent=0)[:1800])
