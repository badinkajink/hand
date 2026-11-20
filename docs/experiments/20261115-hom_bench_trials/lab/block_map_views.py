"""Views for the two-camera block map fit (docs/experiments/20261113-camera_arm_calibration/lab/block_map_fit.py) from a
continuous capture of block_map_capture.py (2026-11-19 evening, the owner setting C45 and S67 down in new poses).

Per block: the detection batches give each camera's corners of the block's tags about 4 times a second. A still stretch is
--still-s or more in which every tag both cameras keep seeing moves under --still-px between batches, with at least
--min-tags tags in each camera. Each still stretch is one view: per camera and tag the median corners, kept when their
sd stays under --max-sd px (as block_views.py). Consecutive stretches whose block centres moved under --same-px in both
cameras are one pose (the first kept). Writes OUT/<block>/views.jsonl and session.json in block_views.py's format, and
prints per block the views, the tags seen per camera and the best (lowest-angle) decodes of each face: bits corrected
and margin.

    /usr/bin/python3 block_map_views.py CAPTURE.jsonl [--out DIR]
"""
import argparse
import json
from pathlib import Path

import numpy as np

BLOCKS = {"C45": tuple(range(32, 38)), "S67": tuple(range(38, 44))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("capture")
    ap.add_argument("--out", default=None)
    ap.add_argument("--still-s", type=float, default=1.5)
    ap.add_argument("--still-px", type=float, default=1.0)
    ap.add_argument("--min-tags", type=int, default=2)
    ap.add_argument("--max-sd", type=float, default=0.35)
    ap.add_argument("--same-px", type=float, default=8.0)
    a = ap.parse_args()
    cap = Path(a.capture)
    out = Path(a.out) if a.out else cap.parent
    dets = [d for line in open(cap) for d in json.loads(line)]
    sns = sorted({d["sn"] for d in dets})
    for b, ids in BLOCKS.items():
        # group by (camera, frame time): each camera's frames in time order
        frames = {}
        for d in dets:
            if d["id"] in ids:
                frames.setdefault((d["sn"], d["t"]), {})[d["id"]] = d
        per = {sn: sorted(((t, v) for (s, t), v in frames.items() if s == sn), key=lambda x: x[0]) for sn in sns}
        if any(not per[sn] for sn in sns):
            print(f"{b}: not seen by both cameras")
            continue
        # a common clock: each camera's frames; a sample = the latest frame of each camera at camera 0's frame times
        t0s = [t for t, _ in per[sns[0]]]
        tl = {sn: np.array([t for t, _ in per[sn]]) for sn in sns}
        samples = []
        for t in t0s:
            row = {}
            for sn in sns:
                k = int(np.searchsorted(tl[sn], t, side="right")) - 1
                if k >= 0 and t - tl[sn][k] < 0.6:
                    row[sn] = per[sn][k][1]
            if len(row) == len(sns) and all(len(row[sn]) >= a.min_tags for sn in sns):
                samples.append((t, row))
            else:
                samples.append((t, None))
        views, cur, last_pose = [], [], None

        def close(cur):
            if len(cur) < 2 or cur[-1][0] - cur[0][0] < a.still_s:
                return None
            cams = {}
            for sn in sns:
                common = set.intersection(*[set(r[sn]) for _, r in cur])
                C = {i: np.array([r[sn][i]["c"] for _, r in cur]) for i in common}
                sd = {i: float(np.sqrt(((v - v.mean(0)) ** 2).sum(-1).mean())) for i, v in C.items()}
                keep = {i: np.median(v, 0) for i, v in C.items() if sd[i] < a.max_sd}
                if len(keep) < a.min_tags:
                    return None
                cams[sn] = dict(corners={str(i): np.round(v, 3).tolist() for i, v in keep.items()},
                                sd_px={str(i): round(sd[i], 3) for i in keep},
                                best={str(i): [min(x[sn][i]["h"] for _, x in cur), max(x[sn][i]["m"] for _, x in cur)]
                                      for i in keep})
            return dict(t=cur[0][0], cams=cams)

        for t, row in samples:
            if row is None:
                v = close(cur)
                if v:
                    views.append(v)
                cur = []
                continue
            if cur:
                prev = cur[-1][1]
                moved = any(np.abs(np.array(row[sn][i]["c"]) - np.array(prev[sn][i]["c"])).max() > a.still_px
                            for sn in sns for i in set(row[sn]) & set(prev[sn]))
                if moved:
                    v = close(cur)
                    if v:
                        views.append(v)
                    cur = []
            cur.append((t, row))
        v = close(cur)
        if v:
            views.append(v)
        # one view per pose: drop a view whose block centres stayed within same_px of the previous kept one
        kept = []
        for v in views:
            cen = {sn: np.mean([np.mean(c, 0) for c in v["cams"][sn]["corners"].values()], 0) for sn in sns}
            if last_pose is not None and all(np.linalg.norm(cen[sn] - last_pose[sn]) < a.same_px for sn in sns):
                continue
            kept.append(v)
            last_pose = cen
        d = out / b
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "views.jsonl", "w") as fh:
            for k, v in enumerate(kept):
                fh.write(json.dumps(dict(view=k, t=v["t"], cams={sn: dict(corners=c["corners"], sd_px=c["sd_px"])
                                                                 for sn, c in v["cams"].items()})) + "\n")
        (d / "session.json").write_text(json.dumps(dict(map="HexBlock(29.985, 13.0)", ids=list(ids), extra_ids=[],
                                                        source=str(cap))))
        seen = {i: [sum(str(i) in v["cams"][sn]["corners"] for v in kept) for sn in sns] for i in ids}
        best = {}
        for v in kept:
            for sn in sns:
                for i, (h, m) in v["cams"][sn]["best"].items():
                    if i not in best or (h, -m) < (best[i][0], -best[i][1]):
                        best[i] = (h, m)
        print(f"{b}: {len(kept)} poses seen by both cameras ({len(views)} still stretches); views per tag "
              f"({', '.join(s[-4:] for s in sns)}): " + ", ".join(f"{i}: {s}" for i, s in seen.items())
              + "; best decode per face (bits, margin): " + ", ".join(f"{i}: {best.get(str(i))}" for i in ids)
              + f" -> {d}/views.jsonl")


if __name__ == "__main__":
    main()
