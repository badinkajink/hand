"""Camera check of printed tag blocks (owner, 2026-11-19 17:2x, relayed by hand-9b: "the larger object tag blcoks had
multiple pauses and restarts. as a result, there is some edge bleed and fuzzy corners. are these blocks cooked?" and
"there is one very moderate layer shift. i would guess less than 0.2mm. im gonna place both blocks in view of the
cameras"). No motion: the blocks stay where the owner left them and the arm does not move.

Both D435s at full resolution (the tracker's workers, 1920 x 1080, the 11-13 lens fits) for --seconds; every --every-th
frame of each camera goes through pupil_apriltags at quad_decimate 1 with refined edges. Per camera and tag of the
blocks (default the large set's C45, ids 32-37, and S67, ids 38-43): the share of frames it decodes in, the bits
corrected (hamming), the decision margin, and its median corners. Per camera and block, the CAD map's pose
(tracking.tags.HexBlock(--across-flats, 17, --tag-mm)) from the median corners (tracking.pose.solve_body, no tag
dropped) and each tag's corner RMS against it. For tags both cameras see, the four corners triangulated
(gantry_follow_fit.triangulate): the mean edge against --tag-mm (the 11-07 block's tags measured 12.976-13.017 mm), the
edges' spread, and the corners' RMS off their best plane. One annotated frame per camera.

A face that decodes with bits corrected, or with a margin far under the block's other faces, or with corner residuals
several times theirs, is where a print fault shows.

    /usr/bin/python3 block_print_check.py [--seconds 4] [--out DIR]
"""
import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT / "src"), str(HERE), str(ROOT / "scripts"), str(ROOT / "scripts/hardware")]
import hom_trial as HT  # noqa: E402

BLOCKS = {"C45": tuple(range(32, 38)), "S67": tuple(range(38, 44))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--across-flats", type=float, default=29.985)
    ap.add_argument("--tag-mm", type=float, default=13.0)
    ap.add_argument("--out", default=str(ROOT / "logs" / time.strftime("%Y%m%d-block_print_check")))
    a = ap.parse_args()
    import cv2
    import gantry_follow_fit as GF
    from sr2hand.tracking import live
    from sr2hand.tracking import pose as tp
    from sr2hand.tracking import tags as TG
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    maps = {b: TG.HexBlock(a.across_flats, 17.0, a.tag_mm, ids=ids).tag_map() for b, ids in BLOCKS.items()}
    all_ids = {i for ids in BLOCKS.values() for i in ids}
    mf = out / f"{stamp}_blocks_map.json"
    mf.write_text(json.dumps(maps["C45"].to_json()))
    frames, lock, k = {}, threading.Lock(), {}

    def on_frame(sn, y, det):
        with lock:
            k[sn] = k.get(sn, 0) + 1
            if k[sn] % a.every == 0:
                frames.setdefault(sn, []).append(np.array(y, copy=True))

    class _A:
        map = str(mf)
    arm, tracker = HT.start_tracker(_A, lambda s: None, on_frame)
    try:
        time.sleep(1.5 + a.seconds)
    finally:
        errors = {w.sn: w.error for w in tracker.workers if w.error}
        tracker.stop()
        arm.stop()
    cams = live.load_cameras(HT.CAL / "20261113-lens_board_k3_arm.json",
                             HT.CAL / "20261113-handeye_tcp_fitted_lens_moved836.json")
    det = tp.detector(nthreads=4, quad_decimate=1.0, refine_edges=True, decode_sharpening=0.25)
    res = {"what": __doc__.split("\n\n")[0].replace("\n", " "), "t": time.time(), "camera_errors": errors,
           "frames": {sn: len(v) for sn, v in frames.items()}, "across_flats_mm": a.across_flats, "tag_mm": a.tag_mm,
           "cameras": {}, "blocks": {}}
    med = {}
    for sn, ys in frames.items():
        seen = {}
        for y in ys:
            for d in det.detect(y):
                if d.tag_id in all_ids:
                    seen.setdefault(int(d.tag_id), []).append((int(d.hamming), float(d.decision_margin),
                                                               np.asarray(d.corners, float) - tp.CORNER_OFFSET_PX))
        rec = {}
        for tid, v in sorted(seen.items()):
            C = np.median(np.array([x[2] for x in v]), axis=0)
            med[(sn, tid)] = C
            jit = float(np.median(np.linalg.norm(np.array([x[2] for x in v]) - C, axis=2)))
            rec[tid] = dict(rate=round(len(v) / len(ys), 3), hamming_max=max(x[0] for x in v),
                            hamming_mean=round(float(np.mean([x[0] for x in v])), 3),
                            margin_med=round(float(np.median([x[1] for x in v])), 1),
                            corner_jitter_px=round(jit, 3), edge_px=round(float(np.mean(np.linalg.norm(
                                np.roll(C, -1, 0) - C, axis=1))), 1))
        res["cameras"][sn] = rec
        img = cv2.cvtColor(ys[len(ys) // 2], cv2.COLOR_GRAY2BGR)
        for tid, r in rec.items():
            C = med[(sn, tid)]
            col = (0, 200, 0) if r["hamming_max"] == 0 else (0, 0, 255)
            cv2.polylines(img, [C.astype(np.int32).reshape(-1, 1, 2)], True, col, 2)
            cv2.putText(img, f"{tid} h{r['hamming_max']} m{r['margin_med']:.0f}", tuple(int(v) for v in C[0] - [0, 8]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        cv2.imwrite(str(out / f"{stamp}_{sn}.jpg"), img)
    for b, ids in BLOCKS.items():
        tm = maps[b]
        brec = {"per_camera": {}, "triangulated": {}}
        for sn in frames:
            obs = {tid: med[(sn, tid)] for tid in ids if (sn, tid) in med}
            if len(obs) < 1:
                continue
            cam = cams[sn][0]
            T, info = tp.solve_body(obs, tm, cam, max_rpe=1e9, max_view_deg=None)
            if T is None:
                brec["per_camera"][sn] = {"tags": sorted(obs), "fit": None}
                continue
            per = {}
            for tid in info["tags"]:
                e = np.linalg.norm(tp.project(cam, T, tm.stacked([tid])) - obs[tid], axis=1)
                per[tid] = round(float(np.sqrt((e ** 2).mean())), 3)
            brec["per_camera"][sn] = {"tags": info["tags"], "rms_px": round(info["rms_px"], 3), "tag_rms_px": per,
                                      "view_deg": info.get("view_deg")}
        sns = sorted(frames)
        if len(sns) == 2:
            for tid in ids:
                if (sns[0], tid) in med and (sns[1], tid) in med:
                    X, e = GF.triangulate(cams, [(sns[0], med[(sns[0], tid)]), (sns[1], med[(sns[1], tid)])])
                    edges = np.linalg.norm(np.roll(X, -1, 0) - X, axis=1)
                    c = X.mean(0)
                    _, _, vt = np.linalg.svd(X - c)
                    off = (X - c) @ vt[2]
                    diag = np.linalg.norm(X[0] - X[2]), np.linalg.norm(X[1] - X[3])
                    brec["triangulated"][tid] = dict(edge_mm=round(float(edges.mean()), 3),
                                                     edges_mm=np.round(edges, 3).tolist(),
                                                     diag_ratio=round(float(diag[0] / diag[1]), 4),
                                                     out_of_plane_rms_mm=round(float(np.sqrt((off ** 2).mean())), 4))
        res["blocks"][b] = brec
    f = out / f"{stamp}_block_print_check.json"
    f.write_text(json.dumps(res, indent=1, default=str))
    for sn, rec in res["cameras"].items():
        print(f"camera {sn} ({res['frames'][sn]} frames): " + "; ".join(
            f"{tid} rate {r['rate']:.2f} hamming {r['hamming_max']} margin {r['margin_med']:.0f} edge {r['edge_px']:.0f} px "
            f"jitter {r['corner_jitter_px']:.2f}" for tid, r in rec.items()))
    for b, brec in res["blocks"].items():
        for sn, pc in brec["per_camera"].items():
            print(f"{b} camera {sn}: tags {pc['tags']}, rms {pc.get('rms_px')} px, per tag {pc.get('tag_rms_px')}")
        for tid, tr in brec["triangulated"].items():
            print(f"{b} tag {tid} triangulated: edge {tr['edge_mm']} mm {tr['edges_mm']}, diagonals {tr['diag_ratio']}, "
                  f"off plane {tr['out_of_plane_rms_mm']} mm")
    print("->", f, "and frames", [str(out / f"{stamp}_{sn}.jpg") for sn in frames], "errors", errors)


if __name__ == "__main__":
    main()
