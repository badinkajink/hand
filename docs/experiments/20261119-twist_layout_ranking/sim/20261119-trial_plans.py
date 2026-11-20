#!/usr/bin/env python3
"""Bench plans for the layouts proposed for the hand (hand-4a, 2026-11-19), with the simulation's predictions written
into each plan before any trial runs.

For each pick of data/20261119-trial_proposal.json (a layout of the symmetric lattice and its ground-clear grasp k),
logs/20261116-hom_setup/film_plan.py builds the plan as it built the g0 and g13 plans of 2026-11-16 (checked here on
2026-11-19: it reproduces 20261116-film_x70_y110_twist_gr_plan.json exactly): the layout's mounts in the driver's palm
coordinates, the open pose (pads 8 mm off the sphere) and the touch pose, and HOM's world with the grasp's toy vector and
HOM's default forces. The open pose's index and middle MCPs are then raised to -11.0 and -9.5 deg where the plan puts
them lower, as the lab's 'mcpstop' plans of 2026-11-17 did (the hand's MCP stops are -11.4 and -10.0 deg in free air).

meta.ranking holds the prediction: per contact model (point and pads, soft tangential contact, the calibrated 7.5 V
plant of sim/20261119-twist_ranking_search.py) the mean tracking over seeds 0-2 and, where run, 10-12, the twist at the
end, and the layout's rank among the deployable layouts.

Where stage E (sim/20261119-twist_force_refine.py, data/20261119-force_refine.json) found HOM forces for the layout that
score at least GAIN above the default at the fresh seeds 10-12 and no lower than the default at friction 0.6, a second
plan carries them as meta.hom.prm (hom_bench_world.bench_world merges the plan's prm into hom_tasks.PRM, so the server
runs them), named ..._f<f_tripod>_w<w_spin_tripod>_plan.json.

Plans go to docs/experiments/20261115-hom_bench_trials/plans/20261119-rank_x<X>_y<Y>_twist_gr_g<K>[_f.._w..]_plan.json,
and their list with the predictions to data/20261119-trial_plans.json.

    .venv/bin/python docs/experiments/20261119-twist_layout_ranking/sim/20261119-trial_plans.py [--dry]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[2]
PLANS = ROOT / "docs/experiments/20261115-hom_bench_trials/plans"
FILM_PLAN = ROOT / "logs/20261116-hom_setup/film_plan.py"
GR_ROWS = ROOT / "logs/20261116-hom_setup/gr_rows"
MCP_OPEN = {"index": -11.0, "middle": -9.5}
GAIN = 0.05
REFINE = STUDY / "data/20261119-force_refine.json"
ENV = {"SCALE_HAND": "cad_v2", "HOS_PLANT": "servo", "HOS_SCENES": "logs/20261102-hom_servo_plant/scenes",
       "HOM_ROWS_TAG": "_sv25r", "CUDA_VISIBLE_DEVICES": "", "NUMPY_MADVISE_HUGEPAGE": "0", "OMP_NUM_THREADS": "1"}


def rows():
    out = []
    for st in ("A", "B", "C"):
        p = ROOT / f"logs/20261119-twist_layout_ranking/rows_{st}.jsonl"
        if p.exists():
            out += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    return out


def prediction(rs, call, k):
    import numpy as np
    out = {}
    for m in ("point", "pads"):
        rr = [r for r in rs if r["model"] == m and int(r["call"]) == call and int(r["k"]) == k]
        sc = lambda r: 0.0 if "error" in r or not r.get("held") else float(r.get("track") or 0.0)   # noqa: E731
        s012 = [sc(r) for r in rr if int(r["seed"]) in (0, 1, 2)]
        s10 = [sc(r) for r in rr if int(r["seed"]) in (10, 11, 12)]
        out[m] = {"track_seeds012": round(float(np.mean(s012)), 3) if s012 else None,
                  "track_seeds101112": round(float(np.mean(s10)), 3) if s10 else None,
                  "twist_end_deg": round(float(np.mean([r.get("twist_end") or 0 for r in rr])), 1) if rr else None,
                  "rise_max_mm": round(float(np.mean([r.get("rise_max") or 0 for r in rr])), 2) if rr else None,
                  "runs": len(rr), "held": sum(bool(r.get("held")) for r in rr)}
    return out


def refined_prediction(ref, pred):
    """The prediction of a plan with stage E's forces: their mean tracking over both models at seeds 0-2 and 10-12 and
    under point contact at friction 0.6 (data/20261119-force_refine.json), with the default forces' prediction kept."""
    return {"both_models_seeds012": ref.get("best_seeds012"), "both_models_seeds101112": ref.get("best_seeds101112"),
            "point_mu06_seeds012": ref.get("best_mu06"), "default_forces": pred}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposal", default=str(STUDY / "data/20261119-trial_proposal.json"))
    ap.add_argument("--out", default=str(STUDY / "data/20261119-trial_plans.json"))
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    prop = json.loads(Path(a.proposal).read_text())
    picks = [{**p, "role": "ranked"} for p in prop["pick"]] + [{**p, "role": "anchor"} for p in prop["anchors"]]
    rs = rows()
    refined = {int(r["call"]): r for r in json.loads(REFINE.read_text())["layouts"]} if REFINE.exists() else {}
    env = {**os.environ, **ENV}
    listing = []
    for p in picks:
        x, y = p["xy"]
        k = int(p["k"])
        name = f"20261119-rank_x{x:g}_y{y:g}_twist_gr_g{k}_plan.json"
        path = PLANS / name
        pred = prediction(rs, int(p["call"]), k)
        listing.append({"plan": str(path.relative_to(ROOT)), "role": p["role"], "call": p["call"], "xy": [x, y], "k": k,
                        "rank_among_deployable": p.get("rank_among_deployable"), "score": p.get("score"),
                        "prediction": pred})
        if a.dry:
            print(name, json.dumps(pred))
            continue
        cmd = [str(ROOT / ".venv/bin/python"), str(FILM_PLAN), str(path), "--xy", f"{x:g}", f"{y:g}", "--grasp",
               str(k), "--grid", "grid2dgr", "--rows", str(GR_ROWS)]
        res = subprocess.run(cmd, cwd=FILM_PLAN.parent, env=env, capture_output=True, text=True, timeout=900)
        if res.returncode != 0:
            listing[-1]["error"] = res.stderr[-600:]
            print(name, "FAILED", res.stderr[-400:], flush=True)
            continue
        plan = json.loads(path.read_text())
        op = plan["poses"][0]
        assert op["name"] == "open", op["name"]
        moved = {}
        for f, lo in MCP_OPEN.items():
            if op["joints"][f]["mcp"] < lo:
                moved[f] = [round(op["joints"][f]["mcp"], 2), lo]
                op["joints"][f]["mcp"] = lo
        plan["design"] += "_mcpstop" if moved else ""
        plan["meta"]["open_pose_stops"] = (f"{time.strftime('%F %H:%M')}: the index and middle MCPs of the open pose "
                                           f"raised to -11.0 and -9.5 deg where lower (the lab's mcpstop of 2026-11-17): "
                                           f"{moved or 'none moved'}")
        plan["meta"]["ranking"] = {"study": "docs/experiments/20261119-twist_layout_ranking", "role": p["role"],
                                   "rank_among_deployable": p.get("rank_among_deployable"),
                                   "score_mean_both_models": p.get("score"), "prediction": pred,
                                   "written_before_trials": time.strftime("%F %H:%M")}
        path.write_text(json.dumps(plan, indent=1) + "\n")
        entry = listing[-1]
        entry.update({"mcp_moved": moved, "fits": plan["meta"]["mounts"]["fits"],
                      "violations": plan["meta"]["driver"]["violations"], "mounts_palm_mm": plan["mounts_palm_mm"]})
        print(name, "ok", "moved", moved, "violations", plan["meta"]["driver"]["violations"], flush=True)
        ref = refined.get(int(p["call"]))
        if ref and ref.get("best_seeds101112") is not None and ref.get("default_seeds012") is not None \
                and ref["best_seeds101112"] >= ref["default_seeds012"] + GAIN \
                and (ref.get("best_mu06") or 0) >= (ref.get("default_mu06") or 0):
            prm = ref["best_prm"]
            plan2 = json.loads(json.dumps(plan))
            h = plan2["meta"]["hom"]
            h["prm"] = {**(h.get("prm") or {}), **prm}
            h["task_prm"] = {**(h.get("task_prm") or {}), **prm}
            plan2["design"] += f"_f{prm['f_tripod']:g}_w{prm['w_spin_tripod']:g}"
            plan2["meta"]["ranking"]["force_refine"] = ref
            path2 = PLANS / name.replace("_plan.json", f"_f{prm['f_tripod']:g}_w{prm['w_spin_tripod']:g}_plan.json")
            path2.write_text(json.dumps(plan2, indent=1) + "\n")
            listing.append({**entry, "plan": str(path2.relative_to(ROOT)), "prm": prm, "force_refine": ref,
                            "prediction": refined_prediction(ref, pred)})
            print(path2.name, "ok (refined forces)", prm, flush=True)
    if not a.dry:
        Path(a.out).write_text(json.dumps({"what": __doc__.split("\n\n")[0].replace("\n", " "),
                                           "written": time.strftime("%F %H:%M"), "plans": listing}, indent=1) + "\n")
        print("wrote", a.out)


if __name__ == "__main__":
    main()
