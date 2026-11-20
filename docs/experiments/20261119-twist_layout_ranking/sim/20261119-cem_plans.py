#!/usr/bin/env python3
"""Bench plans of stage F's picks (sim/20261119-twist_cem_refine.py): each proposed layout with its CEM-refined grasp
and HOM's refined forces and contact modes, where the pick scored at least GAIN above the start at the fresh seeds 10-12
and kept its score within 0.05 at friction 0.6 (hand-4a, 2026-11-19).

The plan is built as logs/20261116-hom_setup/film_plan.py builds a grid grasp's plan: hom_bench_plans.build_plan with
hom_bench_world.study_pick returning this pick (its grasp v, HOM's prm with the forces, the force knots of level knots and
the mode weights), so meta.hom.v and meta.hom.prm carry them and the bench server builds the same world
(hom_bench_world.world_from_plan merges prm into hom_tasks.PRM; hom_tasks.force_at reads the knots). Then the open
pose's index and middle MCPs are raised to -11.0 and -9.5 deg where lower, and meta.ranking holds the prediction.

    SCALE_HAND=cad_v2 HOS_PLANT=servo HOS_SCENES=logs/20261102-hom_servo_plant/scenes HOM_ROWS_TAG=_sv25r \\
        CUDA_VISIBLE_DEVICES= .venv/bin/python docs/experiments/20261119-twist_layout_ranking/sim/20261119-cem_plans.py
Writes docs/experiments/20261115-hom_bench_trials/plans/20261119-rank_x<X>_y<Y>_twist_cem_<level>_plan.json and adds
them to data/20261119-trial_plans.json.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SCALE_HAND", "cad_v2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/hardware"))
import _paths  # noqa: E402,F401
import numpy as np  # noqa: E402

PLANS = Path(os.environ.get("CEM_PLANS_DIR", str(ROOT / "docs/experiments/20261115-hom_bench_trials/plans")))
GAIN = 0.05
MCP_OPEN = {"index": -11.0, "middle": -9.5}


def main():
    import hom_bench_plans as BP
    import hom_bench_world as W
    import hom_films as HF
    from sr2hand.robot import sr2_layouts as SL
    cem = json.loads(Path(os.environ.get("CEM_JSON", str(STUDY / "data/20261119-cem_refine.json"))).read_text())
    lst_path = Path(os.environ.get("CEM_LIST", str(STUDY / "data/20261119-trial_plans.json")))
    lst = json.loads(lst_path.read_text()) if lst_path.exists() else {"plans": []}
    lst["plans"] = [p for p in lst["plans"] if "_cem_" not in p["plan"]]
    for c in cem["picks"]:
        ok = (c.get("best_seeds101112") is not None and c.get("start_seeds012") is not None
              and c["best_seeds101112"] >= c["start_seeds012"] + GAIN
              and (c.get("best_mu06") or 0) >= c["best_seeds101112"] - 0.05)
        if not ok:
            print(f"{c['level']} call {c['call']}: not written (start {c['start_seeds012']}, fresh "
                  f"{c.get('best_seeds101112')}, mu 0.6 {c.get('best_mu06')})")
            continue
        xy = tuple(float(v) for v in c["xy"])
        call = HF.call_at("twist", 40.0, *xy)
        assert call == int(c["call"]), (call, c["call"])
        pick = {"task": "twist", "d_mm": 40.0, "call": call, "xy": list(xy), "d6": [float(v) for v in SL.sym6(*xy)],
                "v": [float(v) for v in c["v"]], "prm": c["prm"], "pick": f"CEM {c['level']} from grid2dgr grasp {c['k']}",
                "variant": "base", "plant_spec": None, "val_seeds": [10, 11, 12],
                "val_scores": [c["best_seeds101112"]], "val_mean": c["best_seeds101112"], "search_mean": c["best_seeds012"],
                "rows": "logs/20261119-twist_layout_ranking/rows_F.jsonl"}
        W.study_pick = lambda task, xy_, pick_="S3", plant="sv25", _p=pick: _p
        plan = BP.build_plan("film", xy, call, "twist", fit=True)
        op = plan["poses"][0]
        moved = {}
        for f, lo in MCP_OPEN.items():
            if op["joints"][f]["mcp"] < lo:
                moved[f] = [round(op["joints"][f]["mcp"], 2), lo]
                op["joints"][f]["mcp"] = lo
        name = f"20261119-rank_x{xy[0]:g}_y{xy[1]:g}_twist_cem_{c['level']}_plan.json"
        plan["design"] = f"hom_rank_x{xy[0]:g}_y{xy[1]:g}_twist_cem_{c['level']}" + ("_mcpstop" if moved else "")
        plan["meta"]["source"] = (f"20261119-cem_plans.py: stage F's {c['level']} pick at ({xy[0]:g}, {xy[1]:g}), from "
                                  f"grid2dgr grasp {c['k']}, built by hom_bench_plans.build_plan")
        plan["meta"]["open_pose_stops"] = f"{time.strftime('%F %H:%M')}: MCPs raised where lower: {moved or 'none moved'}"
        plan["meta"]["ranking"] = {"study": "docs/experiments/20261119-twist_layout_ranking", "role": "cem",
                                   "cem": c, "written_before_trials": time.strftime("%F %H:%M")}
        (PLANS / name).write_text(json.dumps(plan, indent=1) + "\n")
        lst["plans"].append({"plan": str(PLANS / name).replace(str(ROOT) + "/", ""), "role": f"cem {c['level']}", "call": call,
                             "xy": list(xy), "k": c["k"], "prm": c["prm"], "mcp_moved": moved,
                             "fits": plan["meta"]["mounts"]["fits"], "violations": plan["meta"]["driver"]["violations"],
                             "mounts_palm_mm": plan["mounts_palm_mm"],
                             "prediction": {"both_models_seeds012": c["best_seeds012"],
                                            "both_models_seeds101112": c["best_seeds101112"],
                                            "point_mu06_seeds012": c["best_mu06"], "start_seeds012": c["start_seeds012"]}})
        print(name, "ok", "moved", moved, "violations", plan["meta"]["driver"]["violations"], flush=True)
    lst["cem_written"] = time.strftime("%F %H:%M")
    lst_path.write_text(json.dumps(lst, indent=1) + "\n")
    print("wrote", lst_path)


if __name__ == "__main__":
    main()
