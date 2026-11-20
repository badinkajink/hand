#!/usr/bin/env python3
"""The CAD-sphere finger gaps of the best deployable layouts along their simulated twists (hand-4a, 2026-11-19 22:0x),
to find layouts whose fingers stay apart in the twist itself. sim/20261119-cad_finger_gaps.py found the index's MCP link
and the middle's PIP link touching (0 mm on the CAD spheres) during the twist ramp at 4 of the 11 proposed layouts,
(135, 65), (115, 70), (105, 60) and (85, 60), all with y_sep at most 70 mm, which the ranking's three-ball test had
passed at 15 mm or more.

For each of the --top best layouts that passed the ranking's deployable test (data/20261119-twist_ranking_summary.json,
"candidates": each layout's best grasp over both models), the ranking's run at seed 0 under point contact and the pads,
with every second period's simulated joints put on the CAD spheres (s3_approach.hand_points, as in
sim/20261119-cad_finger_gaps.py): per finger pair the least gap, its parts and time.

    HOS_PLANT=kp0.55_kv0_fr0.18_fl0_dp0.011 CUDA_VISIBLE_DEVICES= .venv/bin/python \\
        docs/experiments/20261119-twist_layout_ranking/sim/20261119-cad_gap_screen.py --top 60 --workers 6
Writes data/20261119-cad_gap_screen.json.
"""
import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[2]
spec = importlib.util.spec_from_file_location("cad_finger_gaps", HERE / "20261119-cad_finger_gaps.py")
CG = importlib.util.module_from_spec(spec)
sys.modules["cad_finger_gaps"] = CG
spec.loader.exec_module(CG)
SR = CG.SR
import numpy as np  # noqa: E402

_LAB = {}


def lab():
    if not _LAB:
        sys.path[:0] = [str(CG.LAB), str(ROOT / "src"), str(CG.SYSID_LAB)]
        try:
            import cv2  # noqa: F401
        except ImportError:
            import types

            class _Stub(types.ModuleType):
                def __getattr__(self, k):
                    return _Stub(k)

                def __call__(self, *a, **k):
                    return _Stub("call")
            sys.modules["cv2"] = _Stub("cv2")
        import fingertip_tag_check as FT
        import hom_setup as HS
        import s3_approach as SA
        _LAB.update(FT=FT, HS=HS, SA=SA)
    return _LAB["SA"], _LAB["FT"], _LAB["HS"]


def job(c):
    from sr2hand.robot import sr2_layouts as SL
    e = {"call": c["call"], "xy": c["xy"], "k": c["k"], "prm": None}
    tr, r = CG.simulated(e, c["model"])
    SA, FT, HS = lab()
    plan = {"meta": {"hom": {"d6_P_mm": [float(v) for v in SL.sym6(*c["xy"])]}}}
    g = CG.gaps_along(plan, tr, SA, FT, HS, "joints")
    return {**c, "track": r.get("track"), "held": r.get("held"), "gaps": g,
            "least_mm": min(v["gap_mm"] for v in g.values())}


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--ranks", type=int, nargs=2, default=None, help="first and last rank among the deployable layouts "
                    "(in place of --top); the results are added to the file's")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(STUDY / "data/20261119-cad_gap_screen.json"))
    a = ap.parse_args()
    S = json.loads((STUDY / "data/20261119-twist_ranking_summary.json").read_text())
    dep = [b for b in S["candidates"] if b["fits"] and b["held_all"] and (b["im_gap_min_mm"] or 0) >= 15]
    dep.sort(key=lambda b: -b["score"])
    lo, hi = a.ranks if a.ranks else (1, a.top)
    jobs = [{"call": b["call"], "xy": b["xy"], "k": b["k"], "score": b["score"], "rank_among_deployable": i + 1,
             "model": m} for i, b in enumerate(dep) if lo <= i + 1 <= hi for m in SR.MODELS]
    R = SR.load_replan()
    R.RG.gear_class = R.RC2.clutch_class
    R.RC2.STIFF, R.RC2.BREAKAWAY = True, 0.0
    R.RG.RF.RL.rows_file()
    SR._R["mod"] = R
    old = json.loads(Path(a.out).read_text())["runs"] if a.ranks and Path(a.out).exists() else []
    new_keys = {(j["call"], j["k"], j["model"]) for j in jobs}
    out = [r for r in old if (r["call"], r["k"], r["model"]) not in new_keys]
    with mp.get_context("fork").Pool(a.workers, initializer=SR._init) as pool:
        for r in pool.imap_unordered(job, jobs):
            out.append(r)
            print(f"#{r['rank_among_deployable']:3d} {tuple(r['xy'])} g{r['k']:2d} {r['model']:5s} track {r['track']} least "
                  f"{r['least_mm']:6.1f} mm " + " ".join(f"{p} {v['gap_mm']:.1f}" for p, v in r["gaps"].items()),
                  flush=True)
    per = {}
    for r in out:
        per.setdefault((r["call"], r["k"]), []).append(r)
    lay = []
    for (c, k), rr in per.items():
        lay.append({"call": c, "k": k, "xy": rr[0]["xy"], "score": rr[0]["score"],
                    "rank_among_deployable": rr[0]["rank_among_deployable"],
                    "least_mm_both_models": round(min(r["least_mm"] for r in rr), 1),
                    "pair_least": {p: round(min(r["gaps"][p]["gap_mm"] for r in rr), 1) for p in rr[0]["gaps"]}})
    lay.sort(key=lambda x: x["rank_among_deployable"])
    Path(a.out).write_text(json.dumps({"what": __doc__.split("\n\n")[0].replace("\n", " "),
                                       "written": time.strftime("%F %H:%M"), "layouts": lay, "runs": out}, indent=1) + "\n")
    for x in lay:
        print(f"#{x['rank_among_deployable']:3d} {tuple(x['xy'])} g{x['k']:2d} score {x['score']:.3f} least {x['least_mm_both_models']:6.1f} mm")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
