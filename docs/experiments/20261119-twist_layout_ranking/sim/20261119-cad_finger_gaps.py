#!/usr/bin/env python3
"""Finger-to-finger gaps of the bench plans along their simulated twists, on the CAD hand's collision spheres
(hand-4a, 2026-11-19 21:5x). The lab found at 21:3x that at (70, 90) in the sweep pose "curl 40/40" the thumb's and
index's DIP links met: the bench check's three balls per finger (tip pad, tip end, PIP joint) were 27 mm apart there,
the cad_v3 scene's collision spheres -0.6 mm, and 4.5 mm at the joints where the MCPs stopped. The ranking's
deployable test (index and middle at least 15 mm apart in every simulated run) measured the same three balls.

For each plan of data/20261119-trial_plans.json, the ranking's run (sim/20261119-twist_ranking_search.py's _run, the
calibrated 7.5 V plant, HOM at the plan's forces) at seed 0 under point contact and the pads. Every second period's
simulated joints, and HOM's targets, are put on the CAD spheres with the tip pad, tip end and PIP balls
(s3_approach.hand_points, as the lab's pose_sweep.finger_gap does), and per finger pair the least gap is kept with its
parts and time. The same for the hand's measured joints in the HOM server's records of 2026-11-17 to 11-19 at (70, 110)
grasp 0 and (105, 70) grasp 13, whose index and middle met on 11-17.

    HOS_PLANT=kp0.55_kv0_fr0.18_fl0_dp0.011 CUDA_VISIBLE_DEVICES= .venv/bin/python \\
        docs/experiments/20261119-twist_layout_ranking/sim/20261119-cad_finger_gaps.py [--plans NAME ...]
Writes data/20261119-cad_finger_gaps.json.
"""
import argparse
import glob
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[2]
spec = importlib.util.spec_from_file_location("twist_search", HERE / "20261119-twist_ranking_search.py")
SR = importlib.util.module_from_spec(spec)
sys.modules["twist_search"] = SR
spec.loader.exec_module(SR)
import numpy as np  # noqa: E402

LAB = ROOT / "docs/experiments/20261115-hom_bench_trials/lab"
SYSID_LAB = ROOT / "docs/experiments/20261117-sr2_system_identification/lab"
PLANS_DIR = ROOT / "docs/experiments/20261115-hom_bench_trials/plans"
RUNS = ROOT / "docs/experiments/20261115-hom_bench_trials/runs"
FINGERS = ("thumb", "index", "middle")
PAIRS = (("thumb", "index"), ("thumb", "middle"), ("index", "middle"))
EVERY = 2
REAL = {"20261116-film_x70_y110_twist_gr_plan.json": "(70, 110) g0",
        "20261116-film_x105_y70_twist_gr_plan.json": "(105, 70) g13",
        "20261117-film_x105_y70_twist_gr_g13_mcpstop_plan.json": "(105, 70) g13 mcpstop"}


def deg_dict(sim_rad):
    """hom_bench.sim_from_servo's {finger: rad array (yaw, mcp, pip)} -> {finger: {yaw, mcp, pip} in deg}."""
    return {f: {k: math.degrees(float(v)) for k, v in zip(("yaw", "mcp", "pip"), sim_rad[f])} for f in FINGERS}


def simulated(entry, model):
    """[(t, joints deg, targets deg)] of the ranking's run of a plan list entry at seed 0."""
    from sr2hand.control import hom_bench as HB
    sim0, orig = HB.simulate, HB.HomBench
    got = {}

    def keep(*a, **k):
        stream, res = sim0(*a, **k)
        got["stream"] = stream
        return stream, res
    prm = entry.get("prm") or {}

    class Bench(orig):
        def __init__(self, *a, **k):
            k["task_prm"] = {**(k.get("task_prm") or {}), **prm}
            super().__init__(*a, **k)
    HB.simulate = keep
    if prm:
        HB.HomBench = Bench
    try:
        job = {**SR.base_job(int(entry["call"]), tuple(entry["xy"]), model, contact=model), "k": int(entry["k"]),
               "seed": 0}
        r = SR._run(job)
    finally:
        HB.simulate, HB.HomBench = sim0, orig
    out = []
    for s in (got.get("stream") or [])[::EVERY]:
        out.append((float(s["t"]), deg_dict(HB.sim_from_servo(s["joints"])), deg_dict(HB.sim_from_servo(s["targets"]))
                    if s.get("targets") else None))
    return out, r


def gaps_along(plan, traj, SA, FT, HS, which):
    """Per pair the least gap (mm) over the trajectory's samples, with the parts and the time."""
    M = FT.mounts_P(plan)
    best = {f"{a}-{b}": (math.inf, "", None) for a, b in PAIRS}
    for t, j, tg in traj:
        q = j if which == "joints" else tg
        if q is None:
            continue
        X, R, lab = SA.hand_points(FT, HS, M, q)
        fing = np.array([x[0] for x in lab])
        for a, b in PAIRS:
            ia, ib = np.where(fing == a)[0], np.where(fing == b)[0]
            D = np.linalg.norm(X[ia][:, None] - X[ib][None], axis=2) - R[ia][:, None] - R[ib][None]
            k = np.unravel_index(int(np.argmin(D)), D.shape)
            if D[k] < best[f"{a}-{b}"][0]:
                best[f"{a}-{b}"] = (float(D[k]), f"{a} {lab[ia[k[0]]][1]} and {b} {lab[ib[k[1]]][1]}", round(t, 2))
    return {p: {"gap_mm": round(v[0], 1), "parts": v[1], "t_s": v[2]} for p, v in best.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", nargs="*", default=None, help="plan file names (default: every plan of the list)")
    ap.add_argument("--models", nargs="*", default=list(SR.MODELS))
    ap.add_argument("--out", default=str(STUDY / "data/20261119-cad_finger_gaps.json"))
    a = ap.parse_args()
    # the ranking's plant, as the search driver's run() sets it
    R = SR.load_replan()
    R.RG.gear_class = R.RC2.clutch_class
    R.RC2.STIFF, R.RC2.BREAKAWAY = True, 0.0
    R.RG.RF.RL.rows_file()
    SR._R["mod"] = R
    SR._init()
    lst = json.loads((STUDY / "data/20261119-trial_plans.json").read_text())["plans"]
    if a.plans:
        lst = [e for e in lst if Path(e["plan"]).name in a.plans]
    lst = [e for e in lst if "_cem_" not in e["plan"]]          # the CEM picks' grasps need their own setup
    trajs = []
    for e in lst:
        for m in a.models:
            tr, r = simulated(e, m)
            trajs.append((e, m, tr, r))
            print(f"{Path(e['plan']).name} {m}: {len(tr)} samples, track {r.get('track')}", flush=True)
    sys.path[:0] = [str(LAB), str(ROOT / "src"), str(SYSID_LAB)]
    try:                                    # the lab's modules import the camera stack; only the hand model is used here
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
    rows = []
    for e, m, tr, r in trajs:
        plan = json.loads((ROOT / e["plan"]).read_text())
        g = {w: gaps_along(plan, tr, SA, FT, HS, w) for w in ("joints", "targets")}
        rows.append({"plan": Path(e["plan"]).name, "role": e["role"], "xy": e["xy"], "k": e["k"], "prm": e.get("prm"),
                     "model": m, "track": r.get("track"), "held": r.get("held"), "gaps": g})
        least = min(g["joints"].items(), key=lambda kv: kv[1]["gap_mm"])
        print(f"{rows[-1]['plan']:58s} {m:5s} least {least[0]} {least[1]['gap_mm']:6.1f} mm ({least[1]['parts']}, "
              f"t {least[1]['t_s']} s); targets {min(v['gap_mm'] for v in g['targets'].values()):.1f} mm", flush=True)
    real = []
    for p in sorted(glob.glob(str(RUNS / "2026111[789]-*_twist_d40_trial*.jsonl"))):
        L = [json.loads(x) for x in open(p) if x.strip()]
        if not L or L[0].get("kind") != "header":
            continue
        name = Path(L[0].get("plan") or "").name
        if name not in REAL:
            continue
        smp = [s for s in L[1:] if "t" in s and s.get("joints")]
        if len(smp) < 200:
            continue
        from sr2hand.control import hom_bench as HB
        tr = [(float(s["t"]), deg_dict(HB.sim_from_servo(s["joints"])), None) for s in smp[::EVERY]]
        plan = json.loads((PLANS_DIR / name).read_text())
        g = gaps_along(plan, tr, SA, FT, HS, "joints")
        real.append({"file": Path(p).name, "plan": name, "label": REAL[name], "tips": L[0].get("tips"), "gaps": g})
        least = min(g.items(), key=lambda kv: kv[1]["gap_mm"])
        print(f"hand {Path(p).name} {REAL[name]}: least {least[0]} {least[1]['gap_mm']:.1f} mm ({least[1]['parts']}, "
              f"t {least[1]['t_s']} s)", flush=True)
    Path(a.out).write_text(json.dumps({"what": __doc__.split("\n\n")[0].replace("\n", " "),
                                       "written": time.strftime("%F %H:%M"), "every_nth_period": EVERY,
                                       "plans": rows, "hand_11_17": real}, indent=1) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
