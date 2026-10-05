#!/usr/bin/env python3
"""Run the real screwdriver pickup/brake/insertion chain at matched controller settings.

The physical MuJoCo cases use a fixed stiffness-to-solref translation per compiled
sample/tool pair. All cases use the same 100 Hz chain controller, geometry, 24.5 g
tool, contact friction, and nominal seed unless the case says otherwise.
"""

import argparse
import json
import time
import traceback

import hom_chain as chain
import hom_paper_tasks as paper
from contact_surface.records import ROOT, versions, write_json

OUT = ROOT / "results/20261004-distributed-contact/task_chain"
DOC = ROOT / "docs/experiments/20261004-codex"
MEDIA = DOC / "media"

CASES = [
    dict(id="mj_physical_s1_50us", spec="mj:spheres:s1:rs0.75:tr0.03:ir100",
         contact_policy="physical", physics_dt=0.00005, seed=0, film=True),
    dict(id="mj_physical_s1_500us", spec="mj:spheres:s1:rs0.75:tr0.03:ir100",
         contact_policy="physical", physics_dt=0.0005, seed=0, film=True),
    dict(id="mj_physical_s1_2ms", spec="mj:spheres:s1:rs0.75:tr0.03:ir100",
         contact_policy="physical", physics_dt=0.002, seed=0, film=True),
    dict(id="mj_physical_s05_50us", spec="mj:spheres:s0.5:rs0.75:tr0.03:ir100",
         contact_policy="physical", physics_dt=0.00005, seed=0, film=True),
    dict(id="drake_hydro_1ms", spec="drake:hydro:rt0.03:hr1",
         contact_policy="legacy", physics_dt=0.001, seed=0, film=True),
    dict(id="drake_hydro_500us", spec="drake:hydro:rt0.03:hr1",
         contact_policy="legacy", physics_dt=0.0005, seed=0, film=False),
    dict(id="mj_legacy_s1_1ms", spec="mj:spheres:s1:rs0.75:tr0.03:ir100",
         contact_policy="legacy", physics_dt=0.001, seed=0, film=True),
    dict(id="mj_point4s_1ms", spec="mj:point4s", contact_policy="legacy",
         physics_dt=0.001, seed=0, film=False),
    dict(id="mj_physical_s1_50us_seed1", spec="mj:spheres:s1:rs0.75:tr0.03:ir100",
         contact_policy="physical", physics_dt=0.00005, seed=1, film=False),
    dict(id="drake_hydro_1ms_seed1", spec="drake:hydro:rt0.03:hr1",
         contact_policy="legacy", physics_dt=0.001, seed=1, film=False),
]

ROLLING_CASES = [
    dict(id="roll_mj_physical_s1_50us_ctrl500", spec="mj:spheres:s1:rs0.75:tr0.03:ir100",
         contact_policy="physical", physics_dt=0.00005, controller_hz=500, film=True),
    dict(id="roll_mj_physical_s05_50us_ctrl500", spec="mj:spheres:s0.5:rs0.75:tr0.03:ir100",
         contact_policy="physical", physics_dt=0.00005, controller_hz=500, film=True),
    dict(id="roll_drake_hydro_1ms_ctrl500", spec="drake:hydro:rt0.03:hr1",
         contact_policy="legacy", physics_dt=0.001, controller_hz=500, film=True),
    dict(id="roll_mj_legacy_s1_1ms_ctrl500", spec="mj:spheres:s1:rs0.75:tr0.03:ir100",
         contact_policy="legacy", physics_dt=0.001, controller_hz=500, film=True),
    dict(id="roll_mj_physical_s1_50us", spec="mj:spheres:s1:rs0.75:tr0.03:ir100",
         contact_policy="physical", physics_dt=0.00005, film=True),
    dict(id="roll_mj_physical_s1_500us", spec="mj:spheres:s1:rs0.75:tr0.03:ir100",
         contact_policy="physical", physics_dt=0.0005, film=True),
    dict(id="roll_mj_physical_s1_2ms", spec="mj:spheres:s1:rs0.75:tr0.03:ir100",
         contact_policy="physical", physics_dt=0.002, film=False),
    dict(id="roll_mj_physical_s05_50us", spec="mj:spheres:s0.5:rs0.75:tr0.03:ir100",
         contact_policy="physical", physics_dt=0.00005, film=False),
    dict(id="roll_drake_hydro_1ms", spec="drake:hydro:rt0.03:hr1",
         contact_policy="legacy", physics_dt=0.001, film=True),
    dict(id="roll_drake_hydro_500us", spec="drake:hydro:rt0.03:hr1",
         contact_policy="legacy", physics_dt=0.0005, film=False),
    dict(id="roll_mj_legacy_s1_1ms", spec="mj:spheres:s1:rs0.75:tr0.03:ir100",
         contact_policy="legacy", physics_dt=0.001, film=False),
    dict(id="roll_mj_point4s_1ms", spec="mj:point4s",
         contact_policy="legacy", physics_dt=0.001, film=False),
]
ALL_CASES = CASES + ROLLING_CASES


def run_case(case):
    directory = OUT / case["id"]
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "case.json", case)
    film = MEDIA / f"task-{case['id']}.mp4" if case["film"] else None
    start = time.perf_counter()
    try:
        if case["id"].startswith("roll_") or "_roll_" in case["id"]:
            old_rate = chain.RATE
            try:
                chain.RATE = case.get("controller_hz", 100)
                result = paper.exp2(case["spec"], contact_policy=case["contact_policy"],
                                    physics_dt=case["physics_dt"], film=film)
            finally:
                chain.RATE = old_rate
        else:
            result = chain.run_chain(case["spec"], seed=case["seed"], brake="closed",
                                     contact_policy=case["contact_policy"],
                                     physics_dt=case["physics_dt"], film=film)
        trace = result.pop("trace")
        write_json(directory / "trace.json", dict(columns=result["trace_cols"], rows=trace))
        if "rows" in result:
            rows = result.pop("rows")
            write_json(directory / "controller_rows.json", rows)
        result["status"] = "stopped_at_guard" if result.get("guard_failure") else "complete"
        result["film"] = str(film.relative_to(ROOT)) if film else None
    except paper.TaskEscape as error:
        ro = error.rollout
        if film:
            ro.film_out(film)
        write_json(directory / "controller_rows.json", ro.rows)
        result = dict(status="stopped_at_guard", guard_failure=str(error),
                      stopped_time_s=ro.t, stopped_phase=ro.rows[-1]["phase"] if ro.rows else None,
                      sim_s=ro.t, perf=ro.perf, film=str(film.relative_to(ROOT)) if film else None)
    except Exception as error:
        result = dict(status="exception", exception=repr(error), traceback=traceback.format_exc(),
                      film=None)
    result["case"] = case
    result["elapsed_wall_s"] = time.perf_counter() - start
    write_json(directory / "summary.json", result)
    print(case["id"], result["status"], result.get("chain_ok"),
          result.get("sim_s"), round(result["elapsed_wall_s"], 2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=[c["id"] for c in ALL_CASES])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    MEDIA.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "versions.json", versions())
    selected = [c for c in ALL_CASES if args.case is None or c["id"] in args.case]
    summaries = []
    for c in selected:
        path = OUT / c["id"] / "summary.json"
        if path.exists() and not args.force:
            r = json.loads(path.read_text())
            print(c["id"], "cached", r["status"], flush=True)
        else:
            r = run_case(c)
        summaries.append(r)
    write_json(DOC / "data/task_chain_results.json", summaries)


if __name__ == "__main__":
    main()
