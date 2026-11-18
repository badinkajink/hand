"""hom_bench.simulate (the sil's in-process path) on the g0 plan over seeds, with and without HOM's 2026-11-17 options."""
import json, os, sys
from dataclasses import replace
from pathlib import Path
ROOT = Path.home() / "Programs/hand"
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts")); import _paths  # noqa: E401,F401
from sr2hand.control import hom_server as HS
import multiprocessing as mp

PLAN = "docs/experiments/20261115-hom_bench_trials/plans/20261116-film_x70_y110_twist_gr_plan.json"

def run(job):
    class A: pass
    a = A(); a.plan, a.task, a.d, a.dt, a.plant, a.hand = PLAN, "twist", 40.0, 0.02, "measured", "cad_v2"
    a.torque_budget = a.pose_tau = None; a.tips = "off"
    a.filter_step_s, a.contact_off, a.eq1_frames = job["step"], None, "measured"
    HB, b = HS._bench(a)
    import hom_landscape as L_
    try:
        stream, res = HB.simulate(b, job["seed"])
        return {**job, "held": res.get("held"), "J": L_.track_score("twist", res)}
    except Exception as e:  # noqa: BLE001
        return {**job, "error": repr(e)[:120]}

if __name__ == "__main__":
    jobs = [{"step": st, "seed": s} for st in (None, 0.002) for s in range(6)]
    with mp.get_context("fork").Pool(6, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(run, jobs):
            print(json.dumps(r), flush=True)
