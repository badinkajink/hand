"""Where the recorded poses put the board's ChArUco corners in each camera's image: one panel per camera, the latest
pose's frame dimmed, each pose's corners in its own colour with its number, and a 3 x 3 grid of image regions with the
count of poses that put a corner in each.

    /usr/bin/python3 coverage.py OUTDIR      (writes OUTDIR/coverage.jpg)
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

out = Path(sys.argv[1])
recs = [json.loads(l) for l in open(out / "poses.jsonl")]
sns = sorted(recs[-1]["cams"])
panels = []
for sn in sns:
    img = cv2.imread(str(out / f"p{recs[-1]['pose']:02d}" / f"{sn}.png"))
    img = (img * 0.45).astype(np.uint8)
    hits = np.zeros((3, 3), int)
    for r in recs:
        c = r["cams"].get(sn)
        if not c or not c["corners_mean"]:
            continue
        xy = np.array(c["corners_mean"])
        col = tuple(int(v) for v in cv2.applyColorMap(np.uint8([[(37 * r["pose"]) % 256]]), cv2.COLORMAP_HSV)[0, 0])
        for x, y in xy:
            cv2.circle(img, (int(x), int(y)), 6, col, -1)
        cv2.putText(img, str(r["pose"]), tuple(int(v) for v in xy.mean(0)), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
        cell = set(zip(np.clip(xy[:, 1] // 360, 0, 2).astype(int), np.clip(xy[:, 0] // 640, 0, 2).astype(int)))
        for i, j in cell:
            hits[i, j] += 1
    for i in range(3):
        for j in range(3):
            cv2.rectangle(img, (640 * j, 360 * i), (640 * j + 639, 360 * i + 359), (200, 200, 200), 2)
            cv2.putText(img, str(hits[i, j]), (640 * j + 20, 360 * i + 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 4)
    cv2.putText(img, sn, (20, 1060), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
    panels.append(cv2.resize(img, (960, 540)))
cv2.imwrite(str(out / "coverage.jpg"), np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 85])
print(out / "coverage.jpg")
