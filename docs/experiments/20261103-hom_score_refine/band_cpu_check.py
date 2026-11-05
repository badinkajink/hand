#!/usr/bin/env python3
"""CPU replication of the worst-case validation at one seed (step 4 of hand-20261104-hom_worst_case_score, 11-04).

The film of twist layout 362 (110 x 110 mm) dropped the sphere under the worst-case pick S3w at friction 0.4 on the CPU
plant at seed 20, where the GPU scores that rollout 0.67 and held. This runs, for every layout of the band's sample, the
picks the page compares (S3w, and the S3e picks tuned at friction 1.0 and at 0.4) on the CPU plant at seed 20, with the
v and prm of the GPU's validation row of the same rollout seed, and writes both scores side by side. Run 09:02-09:13
on 11-04 through logs/20261104-hom_worst_case/cpu_check.sh (six single-core processes, 900 rollouts).

    python3 docs/experiments/20261103-hom_score_refine/band_cpu_check.py run --plant k21mu0p4 --tasks twist roll [--variant filter]
    python3 docs/experiments/20261103-hom_score_refine/band_cpu_check.py summary

One process per plant (hom_gpu reads the plant at import). --variant plant: every contact meets the object, as the
films; filter: the GPU's contact filter, as scripts/hom/hom_compute.py cpu-worker applies it. Rows (fsynced) to
docs/experiments/20261103-hom_score_refine/rows/band_k21mu/20261104-cpu_check.jsonl; summary beside this script.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("SCALE_HAND", "cad_v2")
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "scripts/hom/hom_grasp_refine.py").exists())
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import hom_grasp_refine as R  # noqa: E402

NAME = "band_k21mu"
SEED = R.VAL_SEEDS[0]
OUT = R.rows_dir(NAME) / "20261104-cpu_check.jsonl"                # read by scripts/hom/hom_score_refine_page.py
SETS = (("S3w", NAME), ("S3e10", "k21"), ("S3e04", "k21mu0p4"))   # the page's pick sets and the rows they came from


def report():
    return json.loads(sorted((R.rows_dir(NAME)).glob("*-band_levels.json"))[-1].read_text())


def gpu_rows(task, plant, src):
    return {(int(r["call"]), r["pick"]): r for r in R.read_rows(R.val_path(task, NAME, plant, src))
            if int(r["seed"]) == SEED}


def append(row):
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def cmd_run(a):
    from dataclasses import replace
    R.setenv(a.plant)
    import mujoco
    import fp_sr2_toy as FP
    import hom_gpu as HG
    import hom_landscape as L
    T, H, S = HG.T, HG.H, HG.S
    W = report()
    fine = re.compile(r"_(dip_link|cassette|cap)_s\d+$")
    done = {(r["task"], r["plant"], r["variant"], r["call"], r["set"]) for r in R.read_rows(OUT)} if OUT.exists() else set()
    for task in a.tasks:
        FP.set_sizes(**({"sph_d_mm": R.TASKS[task]} if R.POSE[task] == "tripod" else {"cyl_d_mm": R.TASKS[task]}))
        scene = HG.Scene("sphere" if R.POSE[task] == "tripod" else "cylinder", R.TASKS[task])
        var = "hold" if task in ("pinpush", "press") else "base"
        over, noise = T.VARIANTS[var]
        G = {src: gpu_rows(task, a.plant, src) for _s, src in SETS}
        for e in W["tasks"][task]["layouts"][: a.n]:
            c = e["call"]
            for sname, src in SETS:
                if (task, a.plant, a.variant, c, sname) in done:
                    continue
                pick = "S3" if src == NAME else e[f"s3e_pick_{src}"]
                g = G[src][(c, pick)]
                setup = R.setup_of(HG, task, g["d"], np.asarray(g["v"], float))
                bm, q_open = scene.prepare(HG.mount_xy(g["d"]), setup)
                plant = T.Plant(scene.pads_path, setup)
                m_pt = mujoco.MjModel.from_xml_path(str(scene.pt_path))
                for m in (plant.m, m_pt):
                    for f in HG.FINGERS:
                        m.body_pos[m.body(f"{f}_mount").id] = bm[f]
                S.set_palm(m_pt, setup["palm"])
                if a.variant == "filter":      # as scripts/hom/hom_compute.py cmd_cpu_worker
                    m = plant.m
                    filt = np.array([("_pad" in (m.geom(k).name or "")) or bool(fine.search(m.geom(k).name or "")
                                                                             and m.geom_size[k][0] < 0.002)
                                     for k in range(m.ngeom)])
                    tool = np.flatnonzero(m.geom_bodyid == plant.tb)
                    m.geom_contype[filt], m.geom_conaffinity[filt] = 2, 0
                    m.geom_conaffinity[tool] = m.geom_conaffinity[tool] | 2
                fs, os_ = T.sr2_specs(setup)
                mirror = H.Mirror(m_pt, fs, os_)
                t0 = time.perf_counter()
                res, _tr, _ = T.rollout(task, setup, plant, mirror, q_open, SEED, {**T.PRM, **(g.get("prm") or {})},
                                        replace(H.HomParams(dt=T.DT_C), **over), None, noise,
                                        T.PLANT_VARIANTS.get(var, {}).get("hold_s", 0.0))
                cpu = L.rollout_score(task, res)
                row = {"task": task, "plant": a.plant, "variant": a.variant, "call": c, "xy": e["xy"], "set": sname,
                       "pick": pick, "seed": SEED, "cpu": round(cpu, 4), "cpu_held": bool(res.get("held")),
                       "gpu": round(L.rollout_score(task, g), 4), "gpu_held": bool(g.get("held")),
                       "wall_s": round(time.perf_counter() - t0, 2), "when": time.strftime("%Y-%m-%d %H:%M:%S")}
                append(row)
                print(f"{task} {a.plant} {a.variant} {c} {sname} ({pick}): CPU {cpu:.3f} held {row['cpu_held']}, "
                      f"GPU {row['gpu']:.3f} held {row['gpu_held']}, {row['wall_s']} s", flush=True)


def cmd_summary(_a):
    rows = R.read_rows(OUT)
    keys = sorted({(r["task"], r["plant"], r["variant"], r["set"]) for r in rows},
                  key=lambda k: (list(R.TASKS).index(k[0]), k[1], k[2], k[3]))
    out = {}
    for k in keys:
        rs = [r for r in rows if (r["task"], r["plant"], r["variant"], r["set"]) == k]
        d = np.array([r["cpu"] - r["gpu"] for r in rs])
        s = {"n": len(rs), "cpu_held": sum(r["cpu_held"] for r in rs), "gpu_held": sum(r["gpu_held"] for r in rs),
             "held_agree": sum(r["cpu_held"] == r["gpu_held"] for r in rs),
             "within_0.05": int((np.abs(d) <= 0.05).sum()), "median_cpu": round(float(np.median([r["cpu"] for r in rs])), 3),
             "median_gpu": round(float(np.median([r["gpu"] for r in rs])), 3),
             "median_diff": round(float(np.median(d)), 3)}
        out["|".join(k)] = s
        print(" ".join(k), s)
    p = Path(__file__).resolve().parent / "cpu_check_summary.json"
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p.relative_to(ROOT))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run")
    pr.add_argument("--plant", required=True, choices=list(R.BANDS[NAME]))
    pr.add_argument("--tasks", nargs="+", default=list(R.TASKS), choices=list(R.TASKS))
    pr.add_argument("--variant", default="plant", choices=("plant", "filter"))
    pr.add_argument("--n", type=int, default=30, help="the first n layouts of the sample")
    sub.add_parser("summary")
    a = ap.parse_args()
    {"run": cmd_run, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
