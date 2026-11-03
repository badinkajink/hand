#!/usr/bin/env python3
"""Fresh-seed check of the one-design picks of hom_plant_transfer.py (2026-11-03; owner: "let's do those three remaining
tasks", the first being this check).

The picks are each grid's own best layout and grasp, the layout chosen by the worst grid of a subset with its grasp
re-picked on each grid, and the layout and grasp chosen by the worst grid. They were chosen from means over seeds 0-2
among up to 527 layouts x 16 grasps, so their three-seed scores are biased upward. `run` reruns every pick of one grid at
seeds 3-7 on that grid's plant and scene cache with the same GPU evaluator (hom_gpu.batch_child) and a per-period noise
seed no grid used; `report` sets the fresh-seed means beside the three-seed ones for every subset of the transfer files.

    python3 scripts/hom_fresh_seeds.py run --src grid2dff_v4b_sv25     # one grid; rows/fresh_seeds/<src>/<task>_d<d>.jsonl
    python3 scripts/hom_fresh_seeds.py report                           # table; rows/20261103-fresh_seeds.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("SCALE_HAND", "cad_v2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

ROWS = ROOT / "docs/experiments/20261029-foundational_pose_review/rows"
OUT = ROWS / "fresh_seeds"
K21, LG = "kp2.1_kv0_fr0.515_fl0_dp0.042", "logs/20261102-hom_servo_plant/"
ENV = {"grid2dff_v4b": ("work", None), "grid2dff_v4b_w25": ("work", LG + "scenes"),
       "grid2dff_v4b_sv25": ("servo", LG + "scenes"), "grid2dff_v4b_svE1p25": ("servo", LG + "scenes_E1.25"),
       "grid2dff_v4b_svE5": ("servo", LG + "scenes_E5"), "grid2dff_v4b_svE10": ("servo", LG + "scenes_E10"),
       "grid2dff_v4b_k21": (K21, LG + "scenes"), "grid2dff_v4b_k21E1p25": (K21, LG + "scenes_E1.25"),
       "grid2dff_v4b_k21E5": (K21, LG + "scenes_E5"), "grid2dff_v4b_k21E10": (K21, LG + "scenes_E10"),
       "grid2dff_v4b_k21mu0p4": (K21, LG + "scenes_mu0.4"), "grid2dff_v4b_k21mu0p7": (K21, LG + "scenes_mu0.7"),
       "grid2dff_v4b_k21mu2p4": (K21, LG + "scenes_mu2.4")}
FILES = ("20261102-plant_transfer.json", "20261102-modulus_transfer.json", "20261102-plant_modulus_factorial.json",
         "20261102-sts_friction_transfer.json", "20261102-sts_modulus_transfer.json")
DEFAULT = {"twist": 40, "roll": 40, "pinpush": 25, "press": 25}
SEEDS = (3, 4, 5, 6, 7)
GEN_SEED = 100003                    # per-period noise; the grids used 0 plus the batch offset
KINDS = ("own", "one_layout", "one_layout_and_grasp")


def call_of(xy):
    return int((xy[0] - 40) // 5 * 25 + (xy[1] - 50) // 5)      # inverse of hom_plant_transfer.xy


def entries():
    """The robust entries of the transfer files, each with its picks {grid: {kind: (call, grasp)}}."""
    import hom_plant_transfer as PT
    out = []
    for f in FILES:
        if not (ROWS / f).exists():
            continue
        J = json.loads((ROWS / f).read_text())
        for key, r in J["robust"].items():
            task = key.split("_d")[0]
            e = {"file": f, "key": key, "task": task, "src": {n: J["grids"][n] for n in r["grids"]}, "picks": {},
                 "three_seed": {}}
            for n in r["grids"]:
                g = PT.grasp_means(task, ROWS / J["grids"][n] / f"{task}_d{DEFAULT[task]}.jsonl")
                oc = call_of(r["own"][n]["xy"])
                p = {"own": (oc, max(g[oc], key=g[oc].get)),
                     "one_layout": (call_of(r["one_layout"]["xy"]), int(r["one_layout"]["grasp"][n]))}
                if r["one_layout_and_grasp"]["grasp"] is not None:
                    p["one_layout_and_grasp"] = (call_of(r["one_layout_and_grasp"]["xy"]),
                                                 int(r["one_layout_and_grasp"]["grasp"]))
                e["picks"][n] = p
                e["three_seed"][n] = {k: round(g[c][gr], 3) for k, (c, gr) in p.items()}
            out.append(e)
    return out


def run(src):
    plant, scenes = ENV[src]
    os.environ["HOS_PLANT"] = plant                              # before hom_gpu imports hand_object_scale_sim
    if scenes:
        os.environ["HOS_SCENES"] = scenes
    if src != "grid2dff_v4b":
        os.environ["HOM_ROWS_TAG"] = src.removeprefix("grid2dff_v4b")
    import multiprocessing as mp
    import hom_ea as E
    import hom_gpu as HG
    import hom_tasks as T
    want = {}
    for e in entries():
        for n, p in e["picks"].items():
            if e["src"][n] == src:
                want.setdefault(e["task"], set()).update(p.values())
    ctx = mp.get_context("spawn")
    for task, picks in sorted(want.items()):
        d = float(DEFAULT[task])
        path, tpath = OUT / src / f"{task}_d{d:g}.jsonl", OUT / src / f"{task}_d{d:g}_timing.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        done = {(int(r["call"]), int(r.get("grasp", 0)), int(r["seed"]))
                for r in (json.loads(x) for x in (path.read_text().splitlines() if path.exists() else []) if x.strip())}
        scene = HG.Scene(T.SHAPE[task], d)
        worlds = []
        for key, mxy, setup in HG.layout_list(task, f"{E.SRC}_size", 0, d):
            cg = (int(key["call"]), int(key.get("grasp", 0)))
            if cg not in picks:
                continue
            bm, q_open = scene.prepare(mxy, setup)
            worlds += [{"key": key, "seed": sd, "setup": setup, "body_mounts": bm, "q_open": q_open,
                        "var": HG.variant_world(E.VARIANT[task], sd)} for sd in SEEDS if (*cg, sd) not in done]
        print(f"{src} {task}: {len(picks)} picks, {len(worlds)} rollouts to run", flush=True)
        if not worlds:
            continue
        spec = {"shape": T.SHAPE[task], "d_mm": d, "task": task, "worlds": worlds, "dt": worlds[0]["var"]["dt"],
                "gen_seed": GEN_SEED, "batch": 0, "path": str(path), "tpath": str(tpath), "layouts": f"{E.SRC}_size"}
        pr = ctx.Process(target=HG.batch_child, args=(spec,))
        pr.start()
        pr.join()
        if pr.exitcode != 0:
            raise SystemExit(f"{src} {task}: batch failed (exit {pr.exitcode})")


def report():
    import hom_landscape as L
    res = []
    print(f"{'subset':58s} {'grid':24s} {'kind':20s} 3-seed  fresh (seeds 3-7)")
    for e in entries():
        task = e["task"]
        fresh = {}
        for n, p in e["picks"].items():
            path = OUT / e["src"][n] / f"{task}_d{DEFAULT[task]:g}.jsonl"
            sc = {}
            for r in (json.loads(x) for x in (path.read_text().splitlines() if path.exists() else []) if x.strip()):
                sc.setdefault((int(r["call"]), int(r.get("grasp", 0))), []).append(L.rollout_score(task, r))
            fresh[n] = {k: (round(float(np.mean(sc[cg])), 3) if cg in sc else None) for k, cg in p.items()}
            for k in p:
                print(f"{e['key'][:58]:58s} {n[:24]:24s} {k:20s} {e['three_seed'][n][k]:.2f}    "
                      f"{fresh[n][k] if fresh[n][k] is not None else float('nan'):.2f}")
        worst = {k: min((fresh[n].get(k) for n in e["picks"] if fresh[n].get(k) is not None), default=None)
                 for k in KINDS}
        loss = {k: (max(fresh[n]["own"] - fresh[n][k] for n in e["picks"]) if all(
            fresh[n].get(k) is not None and fresh[n]["own"] is not None for n in e["picks"]) else None)
            for k in KINDS[1:]}
        res.append({**{k: e[k] for k in ("file", "key", "task", "src")},
                    "picks": {n: {k: list(v) for k, v in p.items()} for n, p in e["picks"].items()},
                    "three_seed": e["three_seed"], "fresh": fresh, "worst_fresh": worst,
                    "max_loss_vs_own_fresh": {k: (round(v, 3) if v is not None else None) for k, v in loss.items()}})
    p = ROWS / "20261103-fresh_seeds.json"
    p.write_text(json.dumps({"seeds": SEEDS, "gen_seed": GEN_SEED, "entries": res}, indent=1))
    print("wrote", p.relative_to(ROOT))


def main():
    ap = argparse.ArgumentParser(description="fresh-seed check of the one-design picks")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run").add_argument("--src", required=True, choices=sorted(ENV))
    sub.add_parser("report")
    sub.add_parser("srcs")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.src)
    elif a.cmd == "srcs":
        print(" ".join(sorted({s for e in entries() for s in e["src"].values()})))
    else:
        report()


if __name__ == "__main__":
    main()
