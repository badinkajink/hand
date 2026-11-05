#!/usr/bin/env python3
"""The CPU runner's time per rollout with the shared forces and contact modes (2026-11-04, autopilot job
hand-20261104-hom_shared_controller; owner, 11-03: "let's make sure we're timing this all for compute-tracking purposes
and throughput measuring, including cpu vs gpu speedups"). It runs scripts/hom/hom_compute.py's cpu-worker unchanged on the
SCS0009's best S1 layouts and grasps at seed 0, with hom_tasks.PRM updated by the task's level-shared pick
(rows/sv25/shared/<task>_d<d>_pick.json), and reads the GPU's seed-0 score from the rescored grid
(grid2dff_v4b_sv25_shared). Rows: docs/experiments/20261103-hom_score_refine/rows/sv25/shared/cpu/cpu_bench.jsonl (one per
rollout, hom_compute's format), the forces and modes per task in prm.json beside them. One process per task and variant,
one BLAS thread, no GPU:

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \\
      ~/.claude/bin/resguard.sh run --mem 3G --cpu 100 -- .venv/bin/python <this> --task press --variant filter --n 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "scripts/hom/hom_grasp_refine.py").exists())
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
os.environ.setdefault("SCALE_HAND", "cad_v2")
import hom_grasp_refine as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--task", required=True, choices=list(R.TASKS))
    ap.add_argument("--variant", required=True, choices=("plant", "filter"))
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    pick = R.rows_dir("sv25") / "shared" / f"{a.task}_d{R.TASKS[a.task]:g}_pick.json"
    prm = json.loads(pick.read_text())["pick"]["prm"]
    hp, scenes, tag = R.PLANTS["sv25"]
    R.PLANTS["sv25"] = (hp, scenes, tag + "_shared")    # the worker reads the GPU's seed-0 score from this grid
    R.setenv("sv25")                                    # before hom_tasks imports hand_object_scale_sim
    import hom_tasks as T
    T.PRM.update(prm)                                   # the worker passes T.PRM to the schedule and the rollout
    import hom_compute as HC
    HC.COMPUTE = R.rows_dir("sv25") / "shared" / "cpu"
    HC.COMPUTE.mkdir(parents=True, exist_ok=True)
    pp = HC.COMPUTE / "prm.json"
    have = json.loads(pp.read_text()) if pp.exists() else {}
    pp.write_text(json.dumps({**have, a.task: prm}, indent=1))
    HC.cmd_cpu_worker(argparse.Namespace(plant="sv25", tasks=[a.task], n=a.n, seed=a.seed, variant=a.variant, procs=1,
                                         proc=0))


if __name__ == "__main__":
    main()
