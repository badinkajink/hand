#!/usr/bin/env python3
"""Validation of the live 2-D searches of hom_ea.py at seeds 20-24, which no search used (post-reset step 5: the EA and
the 16-particle swarm, three starts each, at object sizes the size study did not run; search methods page).

hom_gpu draws the per-period measurement noise from a torch generator seeded per batch, so a layout's score depends on
where it fell in its batch, and two searches that ran one layout scored it differently. Per task and object size the
layouts rerun here are each search's last incumbent and the N layouts of best earlier score
(hom_landscape.rollout_score) and of best tracking score (hom_landscape.track_score) among every layout the searches
ran, a layout's score there being its best grasp's mean over all rollouts the searches ran of that grasp at seeds 0-2.
Each rerun layout runs its whole grasp set (up to 16 grasps) at seeds 20-24, so its validated score is the searches'
fitness, the best grasp's mean, at five new seeds.
Rows: <rows>/ea_live<tag>/<task>_d<d>_val.jsonl (hom_gpu.py's format) and <task>_d<d>_val_timing.jsonl.

    HOS_PLANT=servo HOS_SCENES=logs/20261102-hom_servo_plant/scenes HOM_ROWS_TAG=_sv25 \\
        .venv/bin/python scripts/hom/hom_ea_validate.py --task twist --d 28 [--top 5] [--dry]
"""
from __future__ import annotations

import os

os.environ.setdefault("SCALE_HAND", "cad_v2")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import hom_ea as E  # noqa: E402
import hom_landscape as L  # noqa: E402

HG, T = E.HG, E.T
VAL_SEEDS = (20, 21, 22, 23, 24)
METHODS = {"ea": E.out_dir(E.SRC, "ea"), "pso": E.out_dir(E.SRC, "pso", 16)}


def runs(task, d):
    """[(method, seed, rows path, gens path)] of the searches that finished on this task and size."""
    out = []
    for m, dr in METHODS.items():
        for seed in range(10):
            gp = dr / f"{task}_d{d:g}_s{seed}_gens.jsonl"
            if gp.exists() and gp.stat().st_size:
                out.append((m, seed, dr / f"{task}_d{d:g}_s{seed}.jsonl", gp))
    return out


def pooled(task, paths, score):
    """{layout: best grasp's mean score over every rollout of the rows in paths}."""
    fn = L.track_score if score == "track" else L.rollout_score
    sc = {}
    for p in paths:
        for x in (p.read_text().splitlines() if p.exists() else []):
            if x.strip():
                r = json.loads(x)
                sc.setdefault(int(r["call"]), {}).setdefault(int(r.get("grasp", 0)), []).append(fn(task, r))
    return {c: max(float(np.mean(v)) for v in g.values()) for c, g in sc.items()}


def chosen(task, d, top):
    """The layouts to rerun: every search's last incumbent, then the top layouts by each pooled score."""
    rs = runs(task, d)
    cells = E.lattice()
    inc = []
    for _m, _s, _rp, gp in rs:
        last = [json.loads(x) for x in gp.read_text().splitlines() if x.strip()][-1]
        inc.append(cells[tuple(float(v) for v in last["incumbent"])])
    out = list(dict.fromkeys(inc))
    for score in ("rollout", "track"):
        f = pooled(task, [rp for _m, _s, rp, _gp in rs], score)
        out += [c for c in sorted(f, key=lambda c: -f[c])[:top] if c not in out]
    return out, len(rs)


def evaluate(task, d, calls, path, seeds=VAL_SEEDS, dry=False):
    """Run each layout's grasp set at the seeds it lacks in the rows, as GPU batches in spawned processes."""
    import multiprocessing as mp
    have = {}
    for x in (path.read_text().splitlines() if path.exists() else []):
        if x.strip():
            r = json.loads(x)
            have.setdefault((int(r["call"]), int(r.get("grasp", 0))), set()).add(int(r["seed"]))
    grasps = {}
    for key, mxy, setup in HG.layout_list(task, f"{E.SRC}_size", 0, d):
        grasps.setdefault(int(key["call"]), []).append((key, mxy, setup))
    todo = [(key, mxy, setup, [s for s in seeds if s not in have.get((c, int(key.get("grasp", 0))), set())])
            for c in calls for key, mxy, setup in grasps.get(c, [])]
    todo = [t for t in todo if t[3]]
    n = sum(len(t[3]) for t in todo)
    if dry or not n:
        return n
    scene = HG.Scene(T.SHAPE[task], d)
    worlds = []
    for key, mxy, setup, sds in todo:
        bm, q_open = scene.prepare(mxy, setup)
        worlds += [{"key": key, "seed": s, "setup": setup, "body_mounts": bm, "q_open": q_open,
                    "var": HG.variant_world(E.VARIANT[task], s)} for s in sds]
    ctx = mp.get_context("spawn")
    tpath = path.with_name(path.stem + "_timing.jsonl")
    n0 = sum(1 for x in (tpath.read_text().splitlines() if tpath.exists() else []) if x.strip())
    b = E.BATCH[task]
    for k, b0 in enumerate(range(0, len(worlds), b)):
        spec = {"shape": T.SHAPE[task], "d_mm": float(d), "task": task, "worlds": worlds[b0:b0 + b],
                "dt": worlds[0]["var"]["dt"], "gen_seed": b0, "batch": n0 + k, "path": str(path),
                "tpath": str(tpath), "layouts": f"{E.SRC}_size"}
        pr = ctx.Process(target=HG.batch_child, args=(spec,))
        pr.start()
        pr.join()
        if pr.exitcode != 0:
            raise RuntimeError(f"{task} {d:g} mm: batch {n0 + k} failed (exit {pr.exitcode})")
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", required=True, choices=sorted(E.VARIANT))
    ap.add_argument("--d", type=float, required=True, help="object diameter (mm)")
    ap.add_argument("--top", type=int, default=5, help="the N layouts of best earlier and of best tracking score")
    ap.add_argument("--dry", action="store_true", help="print the layouts and the rollouts to run, run nothing")
    a = ap.parse_args()
    calls, nr = chosen(a.task, a.d, a.top)
    path = METHODS["ea"] / f"{a.task}_d{a.d:g}_val.jsonl"
    n = evaluate(a.task, a.d, calls, path, dry=a.dry)
    print(f"{a.task} {a.d:g} mm: {nr} searches, {len(calls)} layouts {[xy(c) for c in calls]}, {n} rollouts "
          f"{'to run' if a.dry else 'run'}", flush=True)
    return 0


def xy(c):
    """(x_sep, y_sep) in mm of a lattice index (hom_ea.lattice inverted)."""
    return 40 + 5 * (c // 25), 50 + 5 * (c % 25)


if __name__ == "__main__":
    sys.exit(main())
