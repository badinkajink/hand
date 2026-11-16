"""Blind predictions of twist trials on the hand with the calibrated servo model (hand-4a, 2026-11-16 16:1x, for the
corvus session; owner, relayed: "we should keep running twist trials to verify our many, many real2sim calibration
problems"). The plant is rev_clutch2.py's: forward gear friction 0.6 |servo torque|, joints held against any load that
would backdrive them, joint friction that does not creep (--stiff), optionally the index's and the middle's yaw gain
(--kp-yaw-im) and a breakaway floor (--breakaway); the stem scene with the soft contact (impratio 1, friction 1.0).

Run it with HOS_PLANT set to the HOM server's plant, so that HOM converts its forces to servo offsets as on the bench:
    HOS_PLANT=kp0.55_kv0_fr0.18_fl0_dp0.011 OMP_NUM_THREADS=1 .venv/bin/python rev_predict.py OUT.json \\
        --grasps grid2dff:329:7 grid2dgr:329:13 ... [--kp-yaw-im 0.35] [--breakaway 0.02] [--workers 8]
A grasp is SRC:CALL:K, SRC grid2dff, grid2dgr or lat (rev_lat's rows: k 0-4 = grasp 10 at 18.2/15/12/8/4 deg). Per run,
rev_gear.run's row (track, twist at the end, rise, the yaw and thumb-MCP gaps, the thumb PIP, held); the summary prints the
mean over seeds 0-2 and each grasp's contact latitude below the equator.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path.home() / "Programs/hand"
D = ROOT / "logs/20261116-sim_real_reversal"
sys.path.insert(0, str(D))
import rev_clutch2 as RC2  # noqa: E402
import rev_gear as RG  # noqa: E402

FPR = ROOT / "docs/experiments/20261029-foundational_pose_review/rows"
LAT = {0: 18.2, 1: 15.0, 2: 12.0, 3: 8.0, 4: 4.0}


def latitude(src, call, k):
    if src == "lat":
        return LAT[k]
    for x in (FPR / f"{src}_tripod40_v4b.jsonl").read_text().splitlines():
        if x.strip():
            r = json.loads(x)
            if int(r["call"]) == call:
                return round(math.degrees(r["fp"]["tripod"]["grasps"][k]["v"][6]), 1)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--grasps", nargs="+", required=True)
    ap.add_argument("--kp-yaw-im", type=float, default=None)
    ap.add_argument("--breakaway", type=float, default=0.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--scene", default=None, help="the plant's scene in scenes/ (default rev_fric2's soft-contact scene)")
    a = ap.parse_args()
    RC2.STIFF, RC2.KP_YAW_IM, RC2.BREAKAWAY = True, a.kp_yaw_im, a.breakaway
    if a.scene:
        RG.RF.SCENE = D / "scenes" / a.scene
    RG.gear_class = RC2.clutch_class
    gs = []
    for g in a.grasps:
        src, call, k = g.split(":")
        gs.append((f"{src}_c{call}_g{k}" if src != "lat" else f"g10_lat{LAT[int(k)]:g}", src, int(call), int(k)))
    RG.RF.RL.rows_file()
    import multiprocessing as mp
    jobs = [(n, s, c, k, seed, 0.6) for (n, s, c, k) in gs for seed in a.seeds]
    with mp.get_context("fork").Pool(a.workers, initializer=RG.init, initargs=(str(RG.RF.SCENE),)) as pool:
        rows = pool.map(RG.run, jobs)
    meta = {"hos_plant": os.environ.get("HOS_PLANT"), "kp_yaw_im": a.kp_yaw_im, "breakaway": a.breakaway, "c_f": 0.6,
            "stiff": True, "scene": RG.RF.SCENE.name}
    summ = {}
    for n, s, c, k in gs:
        rs = [r for r in rows if r["grasp"] == n and "error" not in r]
        bad = sum(1 for r in rows if r["grasp"] == n and "error" in r)
        m = (lambda key: round(float(np.mean([r[key] for r in rs])), 2) if rs else None)
        summ[n] = {"src": s, "call": c, "grasp": k, "lat_deg": latitude(s, c, k), "runs": len(rs) + bad, "failed": bad,
                   "held": sum(r["held"] for r in rs), "track": m("track") if rs and all(r["track"] is not None for r in rs)
                   else None, "track_min": min((r["track"] or 0) for r in rs) if rs else None,
                   "track_max": max((r["track"] or 0) for r in rs) if rs else None, "twist_end": m("twist_end"),
                   "rise_max": m("rise_max"), "gap_middle_aa": m("gap_middle_aa"), "gap_index_aa": m("gap_index_aa"),
                   "gap_thumb_fe1": m("gap_thumb_fe1"), "thumb_pip_median": m("thumb_pip_median")}
        v = summ[n]
        print(f"{n:16s} lat {v['lat_deg']!s:>5} held {v['held']}/{v['runs']} track {v['track']} [{v['track_min']}, "
              f"{v['track_max']}] twist {v['twist_end']} rise {v['rise_max']} | gaps middle.aa {v['gap_middle_aa']} "
              f"index.aa {v['gap_index_aa']} thumb.fe1 {v['gap_thumb_fe1']} | thumb PIP {v['thumb_pip_median']}", flush=True)
    Path(a.out).write_text(json.dumps({"meta": meta, "summary": summ, "rows": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
