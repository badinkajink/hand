#!/usr/bin/env python3
"""Vetting, planning, briefing and scoring of SR2 hand layouts for the bench: one command per step of the protocol in
.claude/skills/hand-deploy/SKILL.md (hand-4a, 2026-11-19; owner, 23:3x: "develop protocols / automated skills to more
quickly vet a hand and prepare a morphology and its associated controls for real hand deployment, good or bad, as well
as the ensuing evaluation, analysis, and real2sim calibration").

The commands reuse the twist ranking's tested pieces (docs/experiments/20261119-twist_layout_ranking/sim/): the
calibrated plant and its run (20261119-twist_ranking_search.py, the model of the 11-17 blind predictions), the finger
gaps on the CAD collision spheres along a simulated twist (20261119-cad_finger_gaps.py), the plan builder with the
hand's MCP stops and meta.ranking (20261119-trial_plans.py) and the trial scorer (20261120-trials_vs_prediction.py).

    vet LAYOUT ...     LAYOUT is X,Y (the layout's best grasp of the ranking) or X,Y:K (grasp K) on the 5 mm lattice.
                       Per layout: the mounts' fit in the gantries' travel boxes; the twist at seeds 0-2 under point
                       contact and the pads and under stage D's four plants (rows already run are reused); drops and
                       HOM's errors; the least finger gap on the CAD collision spheres along the simulated twist (seed 0,
                       both models). Verdict: clear (gap >= 15 mm), marginal (10-15), close (2-10), touching (< 2),
                       plus "drops" or "unfit". Writes --out (default <study>/data/<date>-vet.json).
    plan VET           Bench plans for the vet file's layouts (all, or --verdicts), their predictions in meta.ranking,
                       the HOM server's load check (HomBench.from_plan at the torque budget); adds them to the study's
                       plan list with their vet fields.
    brief LIST         The message for the lab: per plan its name, predicted tracking, CAD gap and flags, in the
                       proposed order (clear first, then the bad end, then droppers), with the number of trials.
    score              The trials in docs/experiments/20261115-hom_bench_trials/runs/ since --since against the plan
                       list's predictions: Spearman's rho per contact model and per plant.

    HOS_PLANT=kp0.55_kv0_fr0.18_fl0_dp0.011 CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/hom/hand_deploy.py vet 90,135 75,140:5
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: E402,F401

ROOT = _paths.ROOT
STUDY0 = ROOT / "docs/experiments/20261119-twist_layout_ranking"
SIM = STUDY0 / "sim"
PY = str(ROOT / ".venv/bin/python")
CLEAR, MARGINAL, CLOSE = 15.0, 10.0, 2.0
os.environ.setdefault("HOS_PLANT", "kp0.55_kv0_fr0.18_fl0_dp0.011")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def _load(name, file):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SIM / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def verdict(gap, fits, held, errors):
    if not fits:
        return "unfit"
    v = "clear" if gap >= CLEAR else "marginal" if gap >= MARGINAL else "close" if gap >= CLOSE else "touching"
    if errors:
        v += ", HOM errors"
    if not held:
        v += ", drops"
    return v


def parse_layouts(specs, summary):
    """[(call, (x, y), k)] from "X,Y" or "X,Y:K"; the grasp is the ranking's best over both models when not given."""
    SR = _load("twist_search", "20261119-twist_ranking_search.py")
    lays = {tuple(xy): c for c, xy, n in SR.layouts()}
    best = {tuple(b["xy"]): b["k"] for b in summary["candidates"]}
    out = []
    for s in specs:
        xy, _, k = s.partition(":")
        x, y = (float(v) for v in xy.split(","))
        if (x, y) not in lays:
            sys.exit(f"{s}: not a lattice layout with ground-clear grasps (x 40-160, y 50-170 in 5 mm steps)")
        kk = int(k) if k else best.get((x, y))
        if kk is None:
            sys.exit(f"{s}: no ranked grasp; give one as X,Y:K")
        out.append((lays[(x, y)], (x, y), kk))
    return out


# ------------------------------------------------------------------------------------------------ vet

_W = {}


def _winit():
    SR = _load("twist_search", "20261119-twist_ranking_search.py")
    SR._init()


def _wrun(job):
    SR = _load("twist_search", "20261119-twist_ranking_search.py")
    return SR._run(job)


def _wcad(item):
    CS = _load("cad_gap_screen", "20261119-cad_gap_screen.py")
    return CS.job(item)


def cmd_vet(a):
    import multiprocessing as mp
    import numpy as np
    from sr2hand.robot import sr2_layouts as SL
    S = json.loads((STUDY0 / "data/20261119-twist_ranking_summary.json").read_text())
    SR = _load("twist_search", "20261119-twist_ranking_search.py")
    lays = parse_layouts(a.layouts, S)
    have = {}
    for st in ("A", "B", "C", "D", "P", "V"):
        for r in SR.rows(st):
            have[(r["model"], int(r["call"]), int(r["k"]), int(r["seed"]))] = r
    scr = {}
    sp = STUDY0 / "data/20261119-cad_gap_screen.json"
    if sp.exists():
        for x in json.loads(sp.read_text())["layouts"]:
            scr[(x["call"], x["k"])] = x
    plants = [("point", "point", 7.5, 1.0), ("pads", "pads", 7.5, 1.0)] + list(SR.PLANTS_D)
    jobs = []
    for c, xy, k in lays:
        for name, contact, volts, mu in plants:
            for s in (0, 1, 2):
                if (name, c, k, s) not in have:
                    jobs.append({**SR.base_job(c, xy, name, contact=contact, volts=volts, mu=mu), "k": k, "seed": s})
    cad_items = [{"call": c, "xy": list(xy), "k": k, "score": None, "rank_among_deployable": None, "model": m}
                 for c, xy, k in lays if (c, k) not in scr for m in SR.MODELS]
    R = SR.load_replan()
    R.RG.gear_class = R.RC2.clutch_class
    R.RC2.STIFF, R.RC2.BREAKAWAY = True, 0.0
    R.RG.RF.RL.rows_file()
    SR._R["mod"] = R
    print(f"vet: {len(lays)} layouts, {len(jobs)} runs to do, {len(cad_items)} CAD checks", flush=True)
    out_rows = SR.LOGS / "rows_V.jsonl"
    if jobs or cad_items:
        with mp.get_context("fork").Pool(a.workers, initializer=_winit) as pool, out_rows.open("a") as fh:
            for r in pool.imap_unordered(_wrun, jobs):
                r["stage"] = "V"
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
                have[(r["model"], int(r["call"]), int(r["k"]), int(r["seed"]))] = r
            for r in pool.imap_unordered(_wcad, cad_items):
                x = scr.setdefault((r["call"], r["k"]), {"call": r["call"], "k": r["k"], "xy": r["xy"],
                                                        "least_mm_both_models": 99.0, "pair_least": {}})
                x["least_mm_both_models"] = round(min(x["least_mm_both_models"], r["least_mm"]), 1)
                for p_, v in r["gaps"].items():
                    x["pair_least"][p_] = round(min(x["pair_least"].get(p_, 99.0), v["gap_mm"]), 1)
                x.setdefault("runs", []).append({"model": r["model"], "gaps": r["gaps"]})
    res = []
    for c, xy, k in lays:
        rep = SL.layout_report(SL.sym6(*xy))
        pred, held, errs = {}, True, 0
        for name, *_ in plants:
            rr = [have.get((name, c, k, s)) for s in (0, 1, 2)]
            rr = [r for r in rr if r is not None]
            sc = [SR.score(r) for r in rr]
            pred[name] = {"seeds012": round(float(np.mean(sc)), 4) if len(sc) == 3 else None,
                          "per_seed": [round(v, 3) for v in sc],
                          "held": sum(bool(r.get("held")) and "error" not in r for r in rr), "runs": len(rr),
                          "twist_end_deg": round(float(np.mean([r.get("twist_end") or 0 for r in rr])), 1) if rr else None}
            if name in SR.MODELS:
                held &= pred[name]["held"] == pred[name]["runs"] == 3
                errs += sum("error" in r for r in rr)
        g = scr.get((c, k), {})
        gap = g.get("least_mm_both_models", float("nan"))
        both = [pred[m]["seeds012"] for m in SR.MODELS if pred[m]["seeds012"] is not None]
        v = verdict(gap, bool(rep["fits"]), held, errs)
        res.append({"call": c, "xy": list(xy), "k": k, "fits": bool(rep["fits"]), "min_margin_mm": rep["min_margin_mm"],
                    "predicted_both": round(float(np.mean(both)), 4) if len(both) == 2 else None, "predicted": pred,
                    "held_all": held, "hom_errors": errs, "cad_least_mm": gap, "cad_pairs": g.get("pair_least"),
                    "verdict": v})
        print(f"({xy[0]:g}, {xy[1]:g}) g{k:2d}: predicted {res[-1]['predicted_both']}, point "
              f"{pred['point']['seeds012']}, pads {pred['pads']['seeds012']}, CAD {gap:.1f} mm -> {v}", flush=True)
    out = Path(a.out or STUDY0 / f"data/{time.strftime('%Y%m%d')}-vet.json")
    out.write_text(json.dumps({"what": "hand_deploy.py vet", "written": time.strftime("%F %H:%M"),
                               "plant": os.environ["HOS_PLANT"], "thresholds_mm": {"clear": CLEAR, "marginal": MARGINAL,
                                                                                  "close": CLOSE},
                               "layouts": res}, indent=1) + "\n")
    print("wrote", out)
    return 0


# ------------------------------------------------------------------------------------------------ plan

def load_check(paths):
    """HomBench.from_plan for each plan in a fresh process, as hom_server serve builds it: {name: ok or the error}."""
    code = ("import json,sys\nsys.path.insert(0,'src')\nfrom sr2hand.control import hom_server as HS\n"
            "HS.configure('measured','cad_v2')\nfrom sr2hand.control import hom_bench as HB\nout={}\n"
            "for p in sys.argv[1:]:\n    try:\n        b=HB.HomBench.from_plan(p,'twist',40.0,0.02,torque_budget=0.56)\n"
            "        out[p]='ok'\n    except Exception as e:\n        out[p]=repr(e)[:200]\nprint(json.dumps(out))\n")
    r = subprocess.run([PY, "-c", code, *map(str, paths)], cwd=ROOT, capture_output=True, text=True, timeout=900,
                       env={**os.environ, "OMP_NUM_THREADS": "1"})
    line = [x for x in r.stdout.splitlines() if x.startswith("{")]
    return json.loads(line[-1]) if line else {str(p): f"load check failed: {r.stderr[-300:]}" for p in paths}


def cmd_plan(a):
    V = json.loads(Path(a.vet).read_text())
    keep = [L for L in V["layouts"] if L["fits"] and (not a.verdicts or L["verdict"].split(",")[0] in a.verdicts)]
    prop = {"what": f"hand_deploy.py plan from {a.vet}", "pick": [
        {"call": L["call"], "xy": L["xy"], "k": L["k"], "score": L["predicted_both"], "rank_among_deployable": None}
        for L in keep], "anchors": []}
    tmp = Path(a.vet).with_name(Path(a.vet).stem + "_proposal.json")
    tmp.write_text(json.dumps(prop, indent=1) + "\n")
    tl = Path(a.vet).with_name(Path(a.vet).stem + "_plans.json")
    env = {**os.environ, "OMP_NUM_THREADS": "1", "NUMPY_MADVISE_HUGEPAGE": "0"}
    r = subprocess.run([PY, "-u", str(SIM / "20261119-trial_plans.py"), "--proposal", str(tmp), "--out", str(tl)],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=3600)
    print(r.stdout[-3000:])
    if r.returncode:
        sys.exit(f"trial_plans failed: {r.stderr[-800:]}")
    new = json.loads(tl.read_text())["plans"]
    chk = load_check([ROOT / e["plan"] for e in new])
    lp = Path(a.list)
    lst = json.loads(lp.read_text()) if lp.exists() else {"plans": []}
    names = {Path(e["plan"]).name: i for i, e in enumerate(lst["plans"])}
    by = {(L["call"], L["k"]): L for L in keep}
    for e in new:
        L = by[(e["call"], e["k"])]
        e.update({"role": a.role, "score": L["predicted_both"], "cad_least_mm": L["cad_least_mm"],
                  "cad_touch": L["cad_least_mm"] < CLOSE, "verdict": L["verdict"],
                  "plant_predictions": {m: v["seeds012"] for m, v in L["predicted"].items()},
                  "load_check": chk.get(str(ROOT / e["plan"]), "not checked")})
        if Path(e["plan"]).name in names:
            lst["plans"][names[Path(e["plan"]).name]] = e
        else:
            lst["plans"].append(e)
        print(f"{Path(e['plan']).name}: {e['verdict']}, predicted {e['score']}, load {e['load_check']}")
    lp.write_text(json.dumps(lst, indent=1) + "\n")
    print("wrote", lp)
    return 0


# ------------------------------------------------------------------------------------------------ brief, score

def cmd_brief(a):
    lst = json.loads(Path(a.list).read_text())["plans"]
    sel = [e for e in lst if not a.plans or Path(e["plan"]).name in a.plans]
    order = {"clear": 0, "marginal": 1, "close": 2, "touching": 3}

    def vrd(e):
        """The plan's vet verdict, or one made from the list's CAD gap and group for plans planned before vet."""
        v = e.get("verdict")
        if not v:
            g = e.get("cad_least_mm")
            v = ("clear" if g is None or g >= CLEAR else "marginal" if g >= MARGINAL else "close" if g >= CLOSE
                 else "touching") + (", drops" if e.get("group") == "dropper" else "")
        return v

    def key(e):
        v = vrd(e)
        return (1 if "drops" in v else 0, order.get(v.split(",")[0], 4), -(e.get("score") or 0))
    print(f"{len(sel)} plans, {a.trials} trials each, about {4.5 * a.trials * len(sel) / 60:.1f} h at 4.5 min a trial:")
    for e in sorted(sel, key=key):
        g = e.get("cad_least_mm")
        sc = e.get("score")
        print(f"{Path(e['plan']).name} predicted {'?' if sc is None else f'{sc:.3f}'}, CAD "
              f"{'?' if g is None else f'{g:.1f}'} mm, {vrd(e)}")
    return 0


def cmd_score(a):
    cmd = [PY, str(SIM / "20261120-trials_vs_prediction.py"), "--since", a.since, "--plans", a.list]
    if a.out:
        cmd += ["--out", a.out]
    return subprocess.run(cmd, cwd=ROOT).returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("vet")
    v.add_argument("layouts", nargs="+")
    v.add_argument("--workers", type=int, default=6)
    v.add_argument("--out", default=None)
    p = sub.add_parser("plan")
    p.add_argument("vet")
    p.add_argument("--verdicts", nargs="*", default=None, help="keep these verdicts (clear marginal close touching)")
    p.add_argument("--role", default="ranked")
    p.add_argument("--list", default=str(STUDY0 / "data/20261119-trial_plans.json"))
    b = sub.add_parser("brief")
    b.add_argument("--list", default=str(STUDY0 / "data/20261119-trial_plans.json"))
    b.add_argument("--plans", nargs="*", default=None)
    b.add_argument("--trials", type=int, default=3)
    s = sub.add_parser("score")
    s.add_argument("--list", default=str(STUDY0 / "data/20261119-trial_plans.json"))
    s.add_argument("--since", default=time.strftime("%Y%m%d-000000"))
    s.add_argument("--out", default=None)
    a = ap.parse_args()
    return {"vet": cmd_vet, "plan": cmd_plan, "brief": cmd_brief, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
