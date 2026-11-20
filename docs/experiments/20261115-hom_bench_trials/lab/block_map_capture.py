"""Continuous camera capture of tag blocks for a map fit (owner, 2026-11-19 17:57, relayed by hand-9b: "ok im at the
bench lets do thsi very quickly"; the owner sets the large blocks C45 and S67 down in a new resting pose every ~6 s where
both cameras see them). No motion. Both D435s at full resolution through the tracker's workers; a detector thread takes
each camera's newest frame about --hz times a second and keeps every detection of the blocks' ids (time, camera, id,
bits corrected, decision margin, corners in OpenCV's pixel convention), with one JPEG per camera every --jpeg-s.
Progress every 10 s: detections per block and camera, and the still poses so far (stretches of --still-s where a
block's tags seen by both cameras moved under --still-px). Writes OUT/<stamp>_block_map_capture.jsonl (one line per
detection batch) at the end and as it goes.

    /usr/bin/python3 block_map_capture.py [--seconds 300] [--hz 4]
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
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--hz", type=float, default=4.0)
    ap.add_argument("--jpeg-s", type=float, default=3.0)
    ap.add_argument("--still-s", type=float, default=1.5)
    ap.add_argument("--still-px", type=float, default=1.0)
    ap.add_argument("--out", default=str(ROOT / "logs" / time.strftime("%Y%m%d-block_map_capture")))
    a = ap.parse_args()
    import cv2
    from sr2hand.tracking import pose as tp
    from sr2hand.tracking import tags as TG
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    all_ids = {i for ids in BLOCKS.values() for i in ids}
    mf = out / f"{stamp}_map.json"
    mf.write_text(json.dumps(TG.HexBlock(29.985, 17.0, 13.0, ids=BLOCKS["C45"]).tag_map().to_json()))
    latest, lock = {}, threading.Lock()

    def on_frame(sn, y, det):
        with lock:
            latest[sn] = (time.time(), y)

    class _A:
        map = str(mf)
    arm, tracker = HT.start_tracker(_A, lambda s: None, on_frame)
    fh = open(out / f"{stamp}_block_map_capture.jsonl", "w")
    det = tp.detector(nthreads=4, quad_decimate=1.0, refine_edges=True, decode_sharpening=0.25)
    t_end = time.time() + 1.5 + a.seconds
    last_jpeg, last_print, done_t = {}, time.time(), {}
    hist = {b: [] for b in BLOCKS}            # (t, {sn: centre px}) for still-pose counting
    counts = {(b, sn): 0 for b in BLOCKS for sn in ("832412070344", "836612071918")}
    print(f"{time.strftime('%H:%M:%S')} recording for {a.seconds:.0f} s -> {fh.name}", flush=True)
    try:
        while time.time() < t_end:
            t_loop = time.time()
            with lock:
                frames = {sn: v for sn, v in latest.items() if done_t.get(sn) != v[0]}
            batch = []
            cent = {b: {} for b in BLOCKS}
            for sn, (tf, y) in frames.items():
                done_t[sn] = tf
                y = np.ascontiguousarray(y)
                for d in det.detect(y):
                    if d.tag_id in all_ids:
                        C = np.asarray(d.corners, float) - tp.CORNER_OFFSET_PX
                        batch.append(dict(t=round(tf, 4), sn=sn, id=int(d.tag_id), h=int(d.hamming),
                                          m=round(float(d.decision_margin), 2), c=np.round(C, 3).tolist()))
                        b = "C45" if d.tag_id in BLOCKS["C45"] else "S67"
                        counts[(b, sn)] = counts.get((b, sn), 0) + 1
                        cent[b].setdefault(sn, []).append(C.mean(0))
                if time.time() - last_jpeg.get(sn, 0.0) > a.jpeg_s:
                    last_jpeg[sn] = time.time()
                    cv2.imwrite(str(out / f"{stamp}_{sn}_{int(tf)}.jpg"), y, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if batch:
                fh.write(json.dumps(batch) + "\n")
                fh.flush()
            for b in BLOCKS:
                if len(cent[b]) == 2:
                    hist[b].append((t_loop, {sn: np.mean(v, 0) for sn, v in cent[b].items()}))
            if time.time() - last_print > 10.0:
                last_print = time.time()
                poses = {}
                for b, h in hist.items():
                    n, start, ref = 0, None, None
                    for t, c in h:
                        if ref is None or any(np.linalg.norm(c[sn] - ref[sn]) > a.still_px for sn in c if sn in ref):
                            ref, start, counted = c, t, False
                        elif t - start >= a.still_s and not counted:
                            n += 1
                            counted = True
                    poses[b] = n
                print(f"{time.strftime('%H:%M:%S')} detections " + ", ".join(f"{b} {sn[-4:]} {v}" for (b, sn), v in
                                                                         counts.items()) + f"; still poses seen by both "
                      f"cameras: {poses}", flush=True)
            time.sleep(max(0.0, 1.0 / a.hz - (time.time() - t_loop)))
    finally:
        fh.close()
        tracker.stop()
        arm.stop()
    print(f"{time.strftime('%H:%M:%S')} done -> {fh.name}", flush=True)


if __name__ == "__main__":
    main()
