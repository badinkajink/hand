"""pickup_sim.py --hang (the held phase only, no slip steps) with SR2Plant's plain position servo in place of rev_clutch2's
servo that the load cannot backdrive (kp 0.55 on all nine joints, damping 0.011, 0.18 N m): the MuJoCo control for the
Newton runs of pickup_newton.py, which use that plain servo (2026-11-16).
    HOS_PLANT=kp0.55_kv0_fr0.18_fl0_dp0.011 OMP_NUM_THREADS=1 .venv/bin/python \\
        logs/20261116-sim_real_reversal/pickup_plain_servo.py OUT.json [--variants pads_stiff_ns point_stiff_ns ...]
"""
import argparse
import json
import sys
from pathlib import Path

D = Path.home() / "Programs/hand/logs/20261116-sim_real_reversal"
sys.path.insert(0, str(D))
import pickup_sim as PS  # noqa: E402


def job(variant):
    import rev_clutch2 as RC2
    import sr2_hom_replay as RP

    def plain():
        def make(*a, c=0.6, **k):
            return RP.SR2Plant(*a, **k)
        return make
    RC2.clutch_class = plain
    PS.STEP = 100.0                    # no slip steps after the 10 s held phase
    r = PS.run_hang(variant)
    r["variant"] = variant + "_plain_servo"
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--variants", nargs="+", default=["pads_stiff_ns", "point_stiff_ns", "pads_soft_ns"])
    a = ap.parse_args()
    import multiprocessing as mp
    with mp.get_context("fork").Pool(len(a.variants)) as pool:
        res = pool.map(job, a.variants)
    Path(a.out).write_text(json.dumps(res) + "\n")
    for r in res:
        rows = r["rows"]
        rel = next((x for x in reversed(rows) if x["phase"] == "squeeze (held)"), None)
        held = [x for x in rows if x["phase"] == "held"]
        print(f"{r['variant']:28s} held_at {r['summary']['held_at']} F@1.5 {rel['F'] if rel else None} z end "
              f"{held[-1]['z_mm'] if held else None} | pip@1.5 {rel['pip'] if rel else None}" +
              (f" ERROR {r['error']}" if r.get("error") else ""), flush=True)


if __name__ == "__main__":
    main()
