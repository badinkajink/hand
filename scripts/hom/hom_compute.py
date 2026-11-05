#!/usr/bin/env python3
"""Compute ledger and CPU-against-GPU throughput of the HOM studies (owner, 2026-11-03: "let's make sure we're timing
this all for compute-tracking purposes and throughput measuring, including cpu vs gpu speedups").

Every GPU batch of scripts/hom/hom_gpu.py writes a timing line beside its rows (<rows>_timing.jsonl: worlds, build_s,
run_s, score_s, max_mem_gb, when), the queues log each step's wall time and peak RSS (/usr/bin/time), and
hom_grasp_refine.py logs each iteration's wall and draw times. This script:
  ledger       sums the timing lines per rows directory (every HOM GPU run since 2026-10-29 under
               docs/experiments/20261029-foundational_pose_review/rows and docs/experiments/20261103-hom_score_refine/rows,
               and from 2026-11-04 the task-conflict study's rows and rows_refine under
               docs/experiments/20261103-hom_task_conflict): rollouts, batches, batch seconds (build + run + score), run
               seconds, ms per rollout, first and last batch, per study; and the queue steps of
               logs/20261103-hom_score_refine (wall, peak RSS, iterations); writes compute/<date>-ledger.json.
  cpu-bench    times the CPU runner (hom_tasks.rollout, one process per core, BLAS threads 1) on the GPU runs' own
               layouts, grasps, plant and seed: the best S1 layouts of a task with their S1 grasps, seed 0. Variants:
               'plant', the CPU plant as the CPU checks run it (every contact, the scene's solver: tolerance 1e-10, 100
               iterations, 50 line-search iterations), and 'filter', with the GPU evaluator's contact filter (pads and
               rigid spheres under 2 mm meet the object only), the CPU model closest to the GPU's. Diagnostics: 'solver'
               sets the GPU's solver settings (tolerance 1e-6, 20 iterations, 10 line-search iterations) and 'matched'
               both. On 2026-11-03 'filter' scored as the GPU on 4 of 4 rollouts and as fast as 'plant'; 'solver' dropped
               every twist and roll sphere that the GPU holds (CPU MuJoCo's Newton solver does not converge the pads in
               20 iterations), and a 'matched' pin-push rollout ran for minutes; so the GPU's solver settings are no CPU
               baseline. --procs P runs P processes at once for the throughput of P cores. Rows compute/cpu_bench.jsonl,
               one per rollout.
  gpu-bench    times GPU batches of 64 to the queue's batch size of the same rollouts (the GPU must be free: it
               refuses while logs/20261103-hom_score_refine/queue.sh runs); rows compute/gpu_bench_<task>.jsonl and
               _timing.jsonl.
  report       compute/<date>-throughput.json: per task, the GPU's ms per rollout at each batch size (run only, and
               with build and scoring), the queue's ms per rollout over whole steps (with the CPU-side draws, open
               poses and process starts), the CPU runner's s per rollout, and the speedups: one GPU against one CPU core.

    python3 scripts/hom/hom_compute.py ledger
    python3 scripts/hom/hom_compute.py cpu-bench --tasks twist roll pinpush press --n 6 --variants plant filter [--procs 4]
    python3 scripts/hom/hom_compute.py gpu-bench --tasks twist roll pinpush press
    python3 scripts/hom/hom_compute.py report
Run cpu-bench and gpu-bench through ~/.claude/bin/resguard.sh run (cpu-bench: about 1 GB per process).
"""
from __future__ import annotations

import os

os.environ.setdefault("SCALE_HAND", "cad_v2")

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import hom_grasp_refine as R  # noqa: E402

STUDY = R.OUT
COMPUTE = STUDY / "compute"
HOM_ROWS = R.GRID_ROWS
CONFLICT = ROOT / "docs/experiments/20261103-hom_task_conflict"    # hom_task_conflict.py: validation and refined layouts
LOGS = ROOT / "logs/20261103-hom_score_refine"
SIZES = (64, 256, 1024)                        # gpu-bench sizes below the queue's batch (hom_ea.BATCH)
GPU_SETTINGS = {"tolerance": 1e-6, "iterations": 20, "ls_iterations": 10}     # hom_gpu.Batch
THREADS_1 = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}


def append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def load1():
    return [round(x, 2) for x in os.getloadavg()]


# ------------------------------------------------------------------------------------------------ ledger

def batch_lines(p):
    return [r for r in R.read_rows(p) if "worlds" in r and "run_s" in r]


def summarize(lines):
    import numpy as np
    w = sum(int(r["worlds"]) for r in lines)
    b = sum(float(r.get("build_s", 0)) for r in lines)
    ru = sum(float(r["run_s"]) for r in lines)
    sc = sum(float(r.get("score_s", 0)) for r in lines)
    full = [r for r in lines if int(r["worlds"]) >= 1024]
    when = sorted(r["when"] for r in lines if r.get("when"))
    return {"batches": len(lines), "rollouts": w, "build_s": round(b, 1), "run_s": round(ru, 1), "score_s": round(sc, 1),
            "batch_s": round(b + ru + sc, 1), "ms_per_rollout_run": round(1000 * ru / w, 2) if w else None,
            "ms_per_rollout_batch": round(1000 * (b + ru + sc) / w, 2) if w else None,
            "ms_per_rollout_run_full_batches": round(float(np.median([1000 * r["run_s"] / r["worlds"] for r in full])), 2)
            if full else None,
            "worlds_max": max(int(r["worlds"]) for r in lines) if lines else None,
            "max_mem_gb": max(float(r.get("max_mem_gb") or 0) for r in lines) if lines else None,
            "tasks": sorted({str(r.get("task")) for r in lines}), "first": when[0] if when else None,
            "last": when[-1] if when else None}


STEP_RE = re.compile(r"^(\d\d:\d\d:\d\d) (\w+) (grasp|full) it (\d+): (\d+) candidates of (\d+) layouts, (\d+) rollouts "
                     r"run, draw (\d+) s, all (\d+) s", re.M)


def steps():
    """The queue's steps: wall time and peak RSS (/usr/bin/time), and each search iteration's wall and draw times."""
    out = {}
    for q in sorted(LOGS.glob("Q_*.log")):
        txt = q.read_text(errors="replace")
        m = re.search(r"peak (\d+) kB, ([\d.]+) s", txt)
        its = [{"time": t, "it": int(k), "candidates": int(c), "layouts": int(nl), "rollouts": int(nr), "draw_s": int(dr),
                "all_s": int(al)} for t, _task, _lvl, k, c, nl, nr, dr, al in STEP_RE.findall(txt)]
        out[q.stem[2:]] = {"wall_s": float(m.group(2)) if m else None,
                           "peak_rss_gb": round(int(m.group(1)) / 1e6, 2) if m else None,
                           "done": (LOGS / f"{q.stem}.done").exists(), "iterations": its,
                           "rollouts": sum(i["rollouts"] for i in its) or None}
    return out


def cmd_ledger(a):
    dirs = {}
    for root in (STUDY / "rows", HOM_ROWS, CONFLICT / "rows", CONFLICT / "rows_refine"):
        for p in sorted(root.rglob("*_timing.jsonl")):
            lines = batch_lines(p)
            if lines:
                dirs.setdefault(str(p.parent.relative_to(ROOT)), []).append((p, lines))
    per_dir, per_file = {}, {}
    for d, files in dirs.items():
        per_dir[d] = summarize([x for _p, ls in files for x in ls])
        for p, ls in files:
            per_file[str(p.relative_to(ROOT))] = summarize(ls)
    study = {k: v for k, v in per_file.items() if k.startswith(str(STUDY.relative_to(ROOT)))}
    conflict = {k: v for k, v in per_file.items() if k.startswith(str(CONFLICT.relative_to(ROOT)))}
    allb = [x for files in dirs.values() for _p, ls in files for x in ls]
    tot = summarize(allb)
    out = {"made": time.strftime("%Y-%m-%d %H:%M"), "host": socket.gethostname(),
           "total_hom_gpu": dict(tot, batch_hours=round(tot["batch_s"] / 3600, 2)),
           "score_refine_study": dict(summarize([x for k in study for x in batch_lines(ROOT / k)]),
                                      files=sorted(study)),
           "per_dir": per_dir, "per_file": per_file, "queue_steps": steps()}
    out["score_refine_study"]["batch_hours"] = round(out["score_refine_study"]["batch_s"] / 3600, 2)
    out["task_conflict_study"] = dict(summarize([x for k in conflict for x in batch_lines(ROOT / k)]),
                                      files=sorted(conflict))
    out["task_conflict_study"]["batch_hours"] = round(out["task_conflict_study"]["batch_s"] / 3600, 2)
    COMPUTE.mkdir(parents=True, exist_ok=True)
    p = COMPUTE / f"{time.strftime('%Y%m%d')}-ledger.json"
    p.write_text(json.dumps(out, indent=1))
    t, s = out["total_hom_gpu"], out["score_refine_study"]
    print(f"HOM GPU since 10-29: {t['rollouts']:,} rollouts in {t['batches']} batches, {t['batch_hours']} h of batches "
          f"({t['first']} to {t['last']}); this study: {s['rollouts']:,} rollouts, {s['batch_hours']} h", flush=True)
    print("wrote", p.relative_to(ROOT))


# ------------------------------------------------------------------------------------------------ CPU bench

def top_layouts(task, plant, n):
    """The n best layouts by S1 on the plant's grid, with S1's grasp (hom_grasp_refine's sample if written)."""
    p = R.rows_dir(plant) / f"{task}_layouts.json"
    if p.exists():
        lays = [L_ for L_ in json.loads(p.read_text())["layouts"] if L_["group"] == "top"]
        return lays[:n]
    s1, toy = R.s1_table(task, plant), R.toy_rows(task)
    ranked = sorted((c for c in s1 if c in toy), key=lambda c: (-s1[c][1], c))[:n]
    return [{"call": c, "xy": toy[c]["xy"], "d": toy[c]["d"], "s1_grasp": s1[c][0],
             "v1": toy[c]["fp"][R.POSE[task]]["grasps"][s1[c][0]]["v"]} for c in ranked]


def cmd_cpu_worker(a):
    """One process: a.n rollouts of each task in turn, timed; the parent set one BLAS thread."""
    from dataclasses import replace
    R.setenv(a.plant)
    import mujoco
    import numpy as np
    import fp_sr2_toy as FP
    import hom_gpu as HG
    import hom_landscape as L
    T, H, S = HG.T, HG.H, HG.S
    fine = re.compile(r"_(dip_link|cassette|cap)_s\d+$")
    out = COMPUTE / "cpu_bench.jsonl"
    for task in a.tasks:
        FP.set_sizes(**({"sph_d_mm": R.TASKS[task]} if R.POSE[task] == "tripod" else {"cyl_d_mm": R.TASKS[task]}))
        scene = HG.Scene("sphere" if R.POSE[task] == "tripod" else "cylinder", R.TASKS[task])
        var = "hold" if task in ("pinpush", "press") else "base"
        over, noise = T.VARIANTS[var]
        grid = {}
        gp = R.GRID_ROWS / f"grid2dff_v4b{R.PLANTS[a.plant][2]}" / f"{task}_d{R.TASKS[task]:g}.jsonl"
        lays = top_layouts(task, a.plant, a.n)
        want = {(L_["call"], L_["s1_grasp"]) for L_ in lays}
        for r in R.read_rows(gp):
            if int(r["seed"]) == a.seed and (int(r["call"]), int(r.get("grasp", 0))) in want:
                grid[int(r["call"])] = L.rollout_score(task, r)
        for L_ in lays:
            t0 = time.perf_counter()
            setup = R.setup_of(HG, task, L_["d"], np.asarray(L_["v1"], float))
            mxy = HG.mount_xy(L_["d"])
            bm, q_open = scene.prepare(mxy, setup)
            plant = T.Plant(scene.pads_path, setup)
            m_pt = mujoco.MjModel.from_xml_path(str(scene.pt_path))
            for m in (plant.m, m_pt):
                for f in HG.FINGERS:
                    m.body_pos[m.body(f"{f}_mount").id] = bm[f]
            S.set_palm(m_pt, setup["palm"])
            m = plant.m
            opt0 = {k: float(getattr(m.opt, k)) for k in GPU_SETTINGS}
            if a.variant in ("matched", "filter"):
                filt = np.array([("_pad" in (m.geom(g).name or "")) or bool(fine.search(m.geom(g).name or "")
                                                                         and m.geom_size[g][0] < 0.002)
                                 for g in range(m.ngeom)])
                tool = np.flatnonzero(m.geom_bodyid == plant.tb)
                m.geom_contype[filt], m.geom_conaffinity[filt] = 2, 0
                m.geom_conaffinity[tool] = m.geom_conaffinity[tool] | 2
            if a.variant in ("matched", "solver"):
                for k, v in GPU_SETTINGS.items():
                    setattr(m.opt, k, v)
            fs, os_ = T.sr2_specs(setup)
            mirror = H.Mirror(m_pt, fs, os_)
            sch = T.Schedule(task, setup, T.PRM)
            t_setup = time.perf_counter() - t0
            ld = load1()
            c0, t1 = time.process_time(), time.perf_counter()
            res, _tr, _ = T.rollout(task, setup, plant, mirror, q_open, a.seed, T.PRM,
                                    replace(H.HomParams(dt=T.DT_C), **over), None, noise,
                                    T.PLANT_VARIANTS.get(var, {}).get("hold_s", 0.0))
            wall, cpu = time.perf_counter() - t1, time.process_time() - c0
            row = {"task": task, "plant": a.plant, "hos_plant": R.PLANTS[a.plant][0], "variant": a.variant,
                   "solver": int(m.opt.solver), "cone": int(m.opt.cone), "impratio": float(m.opt.impratio),
                   "call": L_["call"], "xy": L_["xy"], "grasp": L_["s1_grasp"], "seed": a.seed, "wall_s": round(wall, 3),
                   "cpu_s": round(cpu, 3), "setup_s": round(t_setup, 3), "sim_s": round(float(sch.t_end), 3),
                   "phys_dt": float(m.opt.timestep), "opt_scene": opt0, "score": round(L.rollout_score(task, res), 4),
                   "held": bool(res.get("held")), "gpu_score_seed": grid.get(L_["call"]), "procs": a.procs,
                   "proc": a.proc, "load_before": ld, "load_after": load1(), "host": socket.gethostname(),
                   "when": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "threads": {k: os.environ.get(k) for k in THREADS_1}}
            append(out, row)
            print(f"{task} {a.variant} call {L_['call']}: {wall:.2f} s wall, {cpu:.2f} s CPU for {sch.t_end:.1f} s simulated, "
                  f"score {row['score']:.2f} (GPU {grid.get(L_['call'])})", flush=True)


def cmd_cpu_bench(a):
    """Launch a.procs worker processes per variant at once, each running every task; the parent only waits."""
    env = dict(os.environ, **THREADS_1, CUDA_VISIBLE_DEVICES="", NUMPY_MADVISE_HUGEPAGE="0")
    for variant in a.variants:
        t0 = time.time()
        ps = [subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "cpu-worker", "--plant", a.plant,
                                "--tasks", *a.tasks, "--n", str(a.n), "--variant", variant, "--seed", str(a.seed),
                                "--procs", str(a.procs), "--proc", str(k)], env=env, cwd=ROOT)
              for k in range(a.procs)]
        rc = [p.wait() for p in ps]
        print(f"{variant}: {a.procs} processes, {time.time() - t0:.0f} s, exit codes {rc}", flush=True)
        if any(rc):
            raise SystemExit(1)


# ------------------------------------------------------------------------------------------------ GPU bench

def cmd_gpu_bench(a):
    """Batches of SIZES and the queue's batch size of the same rollouts (the best S1 layouts' S1 grasps, seeds
    cycled), each in its own spawned process as in the studies; the timing lines are the measurement."""
    if subprocess.run(["pgrep", "-f", "[h]om_score_refine/queue.sh"], capture_output=True).stdout and not a.force:
        raise SystemExit("the HOM score queue holds the GPU; gpu-bench waits until it ends (--force overrides)")
    R.setenv(a.plant)
    import multiprocessing as mp
    import numpy as np
    import fp_sr2_toy as FP
    import hom_ea as E
    import hom_gpu as HG
    ctx = mp.get_context("spawn")
    for task in a.tasks:
        FP.set_sizes(**({"sph_d_mm": R.TASKS[task]} if R.POSE[task] == "tripod" else {"cyl_d_mm": R.TASKS[task]}))
        scene = HG.Scene("sphere" if R.POSE[task] == "tripod" else "cylinder", R.TASKS[task])
        lays = top_layouts(task, a.plant, a.n)
        base = []
        for L_ in lays:
            setup = R.setup_of(HG, task, L_["d"], np.asarray(L_["v1"], float))
            bm, q_open = scene.prepare(HG.mount_xy(L_["d"]), setup)
            base.append((L_, setup, bm, q_open))
        path, tpath = COMPUTE / f"gpu_bench_{task}.jsonl", COMPUTE / f"gpu_bench_{task}_timing.jsonl"
        COMPUTE.mkdir(parents=True, exist_ok=True)
        for k, W in enumerate(list(SIZES) + [E.BATCH[task]]):
            worlds = []
            for i in range(W):
                L_, setup, bm, q_open = base[i % len(base)]
                sd = i // len(base) % 3
                worlds.append({"key": {"call": L_["call"], "grasp": L_["s1_grasp"], "bench_worlds": W, "i": i},
                               "seed": sd, "setup": setup, "body_mounts": bm, "q_open": q_open,
                               "var": HG.variant_world(E.VARIANT[task], sd)})
            spec = {"shape": "sphere" if R.POSE[task] == "tripod" else "cylinder", "d_mm": R.TASKS[task], "task": task,
                    "worlds": worlds, "dt": worlds[0]["var"]["dt"], "gen_seed": 7, "batch": k, "path": str(path),
                    "tpath": str(tpath), "layouts": f"gpu_bench_{a.plant}"}
            t0 = time.time()
            pr = ctx.Process(target=HG.batch_child, args=(spec,))
            pr.start()
            pr.join()
            if pr.exitcode != 0:
                raise SystemExit(f"{task} {W} worlds failed (exit {pr.exitcode})")
            append(COMPUTE / "gpu_bench_process.jsonl", {"task": task, "worlds": W, "process_s": round(time.time() - t0, 2),
                                                         "plant": a.plant, "when": time.strftime("%Y-%m-%d %H:%M:%S")})


# ------------------------------------------------------------------------------------------------ report

SESSION_H = 3.0                                # CPU rows within this many hours of a key's newest row form its session


def cpu_sessions(rows):
    """{(plant, task, variant, procs): (newest session's rows, [earlier sessions' summaries])}. The CPU numbers of
    11-03 19:1x ran beside the GPU queue (load about 5); a rerun on the idle box is the session to quote."""
    import datetime as dt
    import numpy as np
    when = lambda r: dt.datetime.strptime(r["when"], "%Y-%m-%d %H:%M:%S")  # noqa: E731
    keyed = {}
    for r in rows:
        keyed.setdefault((r.get("plant", "sv25"), r["task"], r["variant"], int(r["procs"])), []).append(r)
    out = {}
    for k, rs in keyed.items():
        rs.sort(key=when)
        new = [r for r in rs if (when(rs[-1]) - when(r)).total_seconds() <= 3600 * SESSION_H]
        old = [r for r in rs if r not in new]
        prev = [{"n": len(old), "wall_s": round(float(np.median([r["wall_s"] for r in old])), 3),
                 "load_before": float(np.median([r["load_before"][0] for r in old])), "first": old[0]["when"]}] if old else []
        out[k] = (new, prev)
    return out


def cpu_summary(rs, prev):
    import numpy as np
    wall = float(np.median([r["wall_s"] for r in rs]))
    return {"n": len(rs), "wall_s": round(wall, 3), "cpu_s": round(float(np.median([r["cpu_s"] for r in rs])), 3),
            "setup_s": round(float(np.median([r["setup_s"] for r in rs])), 3), "sim_s": rs[0]["sim_s"],
            "real_time_factor": round(rs[0]["sim_s"] / wall, 3),
            "load_before": [float(np.median([r["load_before"][0] for r in rs]))],
            "first": min(r["when"] for r in rs), "last": max(r["when"] for r in rs),
            "score_agree": sum(abs(r["score"] - r["gpu_score_seed"]) <= 0.05 for r in rs
                               if r.get("gpu_score_seed") is not None),
            "previous_sessions": prev}


def queue_of(plant, task):
    """The plant's queue timing per level: the timing lines of its search rows and the steps' wall clocks."""
    q = {}
    for level in ("grasp", "full"):
        tf = R.ROWS_ROOT / plant / level / f"{task}_d{R.TASKS[task]:g}_timing.jsonl"
        st_ = steps().get(f"{plant}_{level}_{task}", {})
        ls = batch_lines(tf)
        if ls:
            s = summarize(ls)
            q[level] = dict(s, step_wall_s=st_.get("wall_s"),
                            step_ms_per_rollout=round(1000 * st_["wall_s"] / s["rollouts"], 2)
                            if st_.get("wall_s") else None)
    return q


def speedups(c, q):
    """One GPU against one CPU core: the CPU's wall seconds per rollout over the GPU's per-rollout time at three clocks,
    the GPU's reference being the plant's level-grasp search."""
    sp = {}
    ref = q.get("grasp") or q.get("full")
    for key, cc in c.items():
        if ref and ref.get("ms_per_rollout_run_full_batches"):
            sp[key] = {"vs_gpu_run": round(1000 * cc["wall_s"] / ref["ms_per_rollout_run_full_batches"], 1),
                       "vs_gpu_batch": round(1000 * cc["wall_s"] / ref["ms_per_rollout_batch"], 1),
                       "vs_gpu_step": round(1000 * cc["wall_s"] / ref["step_ms_per_rollout"], 1)
                       if ref.get("step_ms_per_rollout") else None}
    return sp


def cmd_report(a):
    sess = cpu_sessions(R.read_rows(COMPUTE / "cpu_bench.jsonl"))
    out = {"made": time.strftime("%Y-%m-%d %H:%M"), "cpu": {}, "gpu_bench": {}, "queue": {}, "speedup": {},
           "plants": {}}
    for plant in ("sv25", "k21", "k21mu0p4"):
        P = {"cpu": {}, "queue": {}, "speedup": {}}
        for task in R.TASKS:
            c = {f"{v}_p{procs}": cpu_summary(*sess[(pl, t, v, procs)])
                 for (pl, t, v, procs) in sorted(sess) if pl == plant and t == task}
            if c.get("plant_p1") and c.get("plant_p4"):
                c["plant_p4"]["per_core_slowdown"] = round(c["plant_p4"]["wall_s"] / c["plant_p1"]["wall_s"], 3)
            q = queue_of(plant, task)
            P["cpu"][task], P["queue"][task], P["speedup"][task] = c, q, speedups(c, q)
        out["plants"][plant] = P
    for task in R.TASKS:                          # the SCS0009 at the top level, as the page reads it
        out["cpu"][task] = out["plants"]["sv25"]["cpu"][task]
        out["queue"][task] = out["plants"]["sv25"]["queue"][task]
        out["speedup"][task] = out["plants"]["sv25"]["speedup"][task]
        tl = batch_lines(COMPUTE / f"gpu_bench_{task}_timing.jsonl")
        proc = {int(r["worlds"]): r["process_s"] for r in R.read_rows(COMPUTE / "gpu_bench_process.jsonl")
                if r["task"] == task}
        out["gpu_bench"][task] = [{"worlds": int(r["worlds"]), "run_ms": round(1000 * r["run_s"] / r["worlds"], 2),
                                   "batch_ms": round(1000 * (r["build_s"] + r["run_s"] + r["score_s"]) / r["worlds"], 2),
                                   "process_ms": round(1000 * proc[int(r["worlds"])] / r["worlds"], 2)
                                   if int(r["worlds"]) in proc else None, "when": r.get("when")} for r in tl]
    p = COMPUTE / f"{time.strftime('%Y%m%d')}-throughput.json"
    p.write_text(json.dumps(out, indent=1))
    for plant, P in out["plants"].items():
        for task in R.TASKS:
            c, s = P["cpu"][task], P["speedup"][task]
            if c:
                print(plant, task, {k: (v["wall_s"], v["real_time_factor"], v["load_before"][0]) for k, v in c.items()},
                      "speedup", s, flush=True)
    print("wrote", p.relative_to(ROOT))


def main():
    ap = argparse.ArgumentParser(description="HOM compute ledger and CPU/GPU throughput")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ledger")
    for name in ("cpu-bench", "cpu-worker"):
        p = sub.add_parser(name)
        p.add_argument("--plant", default="sv25", choices=list(R.PLANTS))
        p.add_argument("--tasks", nargs="+", default=list(R.TASKS), choices=list(R.TASKS))
        p.add_argument("--n", type=int, default=6)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--procs", type=int, default=1)
        if name == "cpu-bench":
            p.add_argument("--variants", nargs="+", default=["plant", "filter"],
                           choices=("plant", "matched", "filter", "solver"))
        else:
            p.add_argument("--variant", required=True, choices=("plant", "matched", "filter", "solver"))
            p.add_argument("--proc", type=int, default=0)
    g = sub.add_parser("gpu-bench")
    g.add_argument("--plant", default="sv25", choices=list(R.PLANTS))
    g.add_argument("--tasks", nargs="+", default=list(R.TASKS), choices=list(R.TASKS))
    g.add_argument("--n", type=int, default=12)
    g.add_argument("--force", action="store_true")
    sub.add_parser("report")
    a = ap.parse_args()
    {"ledger": cmd_ledger, "cpu-bench": cmd_cpu_bench, "cpu-worker": cmd_cpu_worker, "gpu-bench": cmd_gpu_bench,
     "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
