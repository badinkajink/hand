"""The latitude variants of grid2dff grasp 10 at call 329 (105, 70) that the hand ran on 2026-11-16 from 13:36, on the
stem scene (hand-4a, for the corvus session, 13:58): the grasp vectors are the plans' own (meta.hom.v of
logs/20261116-hom_setup/20261116-film_x105_y70_twist[_g10_lat{15,12,8,4}]_plan.json; film_plan.py --lat-deg sets v[6]
and raises the palm so the pad targets keep their height), written as call 329's grasps 0-4 of a rows file that
toy_setup reads in place of the grid's. Closed-loop HOM through rev_rescore (seeds 0-2, the server's plant), rows
appended and fsynced.

    OMP_NUM_THREADS=1 .venv/bin/python rev_lat.py lat_g10.jsonl --mu 0.5 1.0 [--workers 8]
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path.home() / "Programs/hand"
D = ROOT / "logs/20261116-sim_real_reversal"
sys.path.insert(0, str(D))
import rev_rescore as RR  # noqa: E402

PLANS = ROOT / "logs/20261116-hom_setup"
NAMES = [("lat18.2", ""), ("lat15", "_g10_lat15"), ("lat12", "_g10_lat12"), ("lat8", "_g10_lat8"), ("lat4", "_g10_lat4")]
LAT_ROWS = D / "lat_rows"


def rows_file():
    import sr2_hom_replay as RP
    src = Path(RP.TOY_ROWS) / "grid2dff_tripod40_v4b.jsonl"
    row = next(json.loads(x) for x in src.read_text().splitlines() if x.strip() and json.loads(x)["call"] == 329)
    rec = row["fp"]["tripod"]
    gs = []
    for name, tag in NAMES:
        v = json.loads((PLANS / f"20261116-film_x105_y70_twist{tag}_plan.json").read_text())["meta"]["hom"]["v"]
        g = dict(rec["grasps"][10], v=v, name=name)
        gs.append(g)
        print(f"  {name}: v[6] {math.degrees(v[6]):.1f} deg, v[2] {1e3 * v[2]:.2f} mm", flush=True)
    row["fp"]["tripod"] = dict(rec, grasps=gs, v=gs[0]["v"])
    LAT_ROWS.mkdir(exist_ok=True)
    (LAT_ROWS / "grid2dff_tripod40_v4b.jsonl").write_text(json.dumps(row) + "\n")


def init(path):
    RR.init(path)
    import sr2_hom_replay as RP
    RP.TOY_ROWS = LAT_ROWS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--mu", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tape-mm", type=float, default=0.0)
    ap.add_argument("--impratio", type=float, default=None)
    ap.add_argument("--tau-max", type=float, default=None, help="the servos' torque limit (N m; the plant's 0.18)")
    a = ap.parse_args()
    rows_file()
    path = RR.scene(a.tape_mm, a.impratio)
    extra_p = {"tau_max": a.tau_max} if a.tau_max else {}
    jobs = [(f"g10_{name}", "grid2dff", 329, k, seed, {**RR.SERVER, "mu": mu, **extra_p}) for mu in a.mu
            for k, (name, _t) in enumerate(NAMES) for seed in a.seeds]
    import multiprocessing as mp
    with mp.get_context("fork").Pool(a.workers, initializer=init, initargs=(str(path),)) as pool, \
            Path(a.out).open("a") as fh:
        for r in pool.imap_unordered(RR.run, jobs):
            fh.write(json.dumps({**r, "scene": path.name}) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    print("done", len(jobs), flush=True)


if __name__ == "__main__":
    main()
