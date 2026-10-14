#!/usr/bin/env python3
"""Servo-plant RL scenes of the cheaper fingertip models and short PPO probes of their training cost (U1, 2026-10-14).

  scenes  writes the two RL morphology runs U5 evaluates, both derived from the servo-limited plant's box-tip run
          results/phase1/real_v1/20261011-sv1_u0308_b050_servo_tip (plant kp0.72_kv0_fr0.18_fl0_dp0.0144):
            ..._servo_tip_tpu2.7pads2    the TPU block as 2 mm sphere pads of radius 1.5 mm that touch the tool only
                                         (scripts/make_pad_morphology_run.py --spacing 0.002)
            ..._servo_tip_tpu2.7mesh_c4  the TPU block mesh with condim 4, priority 1 and torsional friction 2.9 mm, the
                                         pads' twist-onset arm at the policies' 5-8 N per finger
                                         (make_work_plant_runs.condim4_variant(spin=0.0029) with OUT and C4_DIR set)
          and runs make_work_plant_runs.check on them and on the 2026-10-11 mesh and 1 mm pad runs: the RL env with zero
          residual actions (scripted grasp and lift) at 64 envs, tool height, held worlds and tip contacts at policy
          steps 60 and 150. Rows: docs/experiments/20261014-pad_usability/cost/rl_scene_check.jsonl.
  probe   one PPO probe: the 2026-10-11 servo-plant trainer command (scripts/rl_contact_train_queue_20261011.sh) cut to
          12 iterations (589,824 steps at 2,048 envs x 24 steps), under /usr/bin/time -v, with nvidia-smi memory.used
          sampled every 2 s into logs/20261014-pad_usability/rl_probes/gpu_<variant>_r<k>.tsv. Afterwards one row:
          'Iteration time' of iterations 3-12 (median, interquartile range), collection and learning time, peak GPU
          memory (whole GPU and this process), host peak RSS, and whether the log mentions an overflow.
          Rows: docs/experiments/20261014-pad_usability/cost/ppo_probes.jsonl.

    MUJOCO_GL=egl WARP_CACHE_PATH=$(mktemp -d) ~/.claude/bin/resguard.sh run --mem 6G --cpu 400 -- taskset -c 17-19 \\
        uv run --extra rl --extra gpu python scripts/pad_usability_rl.py scenes
    MUJOCO_GL=egl WARP_CACHE_PATH=$(mktemp -d) ~/.claude/bin/resguard.sh run --mem 6G --cpu 400 -- taskset -c 17-19 \\
        uv run --extra rl --extra gpu python scripts/pad_usability_rl.py probe --variant servo_mesh --rep 0
  queue   runs probes one at a time (default: mesh, condim 4, 2 mm pads, mesh again, 1 mm pads twice), each in a fresh
          process with its own WARP_CACHE_PATH.
  probe --buffers N,J,M runs the same trainer with env_build._contact_buffers replaced (in the probe's own process,
  env_build.py unchanged) by nconmax N, njmax J and sensor maxmatch M per world, in place of the rule that gives every
  scene with more than 500 geoms 640 / 3,072 / 768.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

COST = ROOT / "docs/experiments/20261014-pad_usability/cost"
LOGS = ROOT / "logs/20261014-pad_usability"
BOX = ROOT / "results/phase1/real_v1/20261011-sv1_u0308_b050_servo_tip"
RUNS = {
    "servo_mesh": "results/phase1/real_v1/20261011-sv1_u0308_b050_servo_tip_tpu2.7mesh",
    "servo_meshc4": "results/phase1/real_v1/20261014-sv1_u0308_b050_servo_tip_tpu2.7mesh_c4",
    "servo_pads2": "results/phase1/real_v1/20261014-sv1_u0308_b050_servo_tip_tpu2.7pads2",
    "servo_pads1": "results/phase1/real_v1/20261011-sv1_u0308_b050_servo_tip_tpu2.7pads1",
}
C4_SPIN = 0.0029
PROBE_STEPS = 589824            # 12 iterations of 2,048 envs x 24 steps
TRAIN_ARGS = ["--recipe", "b_liveA", "--num-envs", "2048", "--total-timesteps", str(PROBE_STEPS), "--save-interval", "41",
              "--no-wandb", "--no-record-videos", "--closed-ctrl-from-keyframe", "open_ik", "--open-finger-from-keyframe",
              "--lift-target-z-above-init", "0.1", "--lift-delta-z", "0.1", "--finger-residual-scale", "0.5",
              "--finger-close-easing", "ease_out_quad", "--lift-phase-start-step", "58", "--reorient-start-step", "58",
              "--finger-residual-active-from-step", "58", "--term-tip-lost-steps", "15", "--friction-dr",
              "--grip-force-penalty-weight", "-2"]


def append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def scene_info(run: Path) -> dict:
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(run / "frozen_scene.xml"))
    s = json.load(open(run / "summary.json"))
    pads = [g for g in range(m.ngeom) if "_pad" in (m.geom(g).name or "")]
    tips = {}
    for f in ("thumb", "index", "middle"):
        try:
            g = m.geom(f"{f}_tipgeom")
            tips[f] = {"condim": int(g.condim[0]), "priority": int(g.priority[0]),
                       "friction": [float(x) for x in g.friction]}
        except KeyError:
            pass
    return {"ngeom": int(m.ngeom), "n_pad_geoms": len(pads), "fingertip": s.get("fingertip"), "tipgeoms": tips,
            "derived_from": s.get("derived_from")}


def cmd_scenes(a) -> int:
    import make_work_plant_runs as W
    pads2 = ROOT / RUNS["servo_pads2"]
    if not (pads2 / "frozen_scene.xml").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts/make_pad_morphology_run.py"), "--src", str(BOX), "--out",
                        str(pads2), "--spacing", "0.002"], check=True)
    W.OUT = BOX
    W.C4_DIR = ROOT / RUNS["servo_meshc4"]
    if not (W.C4_DIR / "frozen_scene.xml").exists():
        W.condim4_variant(spin=C4_SPIN)
    out = COST / "rl_scene_check.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(x)["variant"] for x in out.read_text().splitlines() if x.strip()}
    for v in a.variants:
        if v in done:
            continue
        run = ROOT / RUNS[v]
        info = scene_info(run)
        t0 = time.perf_counter()
        rows = W.check([run], num_envs=64)
        row = {"variant": v, **rows[0], **info, "num_envs": 64, "wall_s": round(time.perf_counter() - t0, 1),
               "when": time.strftime("%Y-%m-%d %H:%M")}
        append(out, row)
        print(json.dumps({k: row[k] for k in ("variant", "ngeom", "n_pad_geoms", "step60", "step150")}), flush=True)
    return 0


def gpu_sampler(path: Path, pid_box: list, stop: threading.Event, every: float = 2.0):
    """memory.used of the GPU and of the probe's processes every `every` s, as t, total_mb, proc_mb."""
    with open(path, "a") as fh:
        while not stop.is_set():
            try:
                tot = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                         capture_output=True, text=True, timeout=10).stdout.split()[0])
                apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                       "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                      timeout=10).stdout
                mine = descendants(pid_box[0]) if pid_box else set()
                proc = sum(int(b) for a_, b in (ln.split(",") for ln in apps.splitlines() if ln.strip())
                           if int(a_) in mine)
                fh.write(f"{time.time():.1f}\t{tot}\t{proc}\n")
                fh.flush()
            except Exception:
                pass
            stop.wait(every)


def descendants(pid: int) -> set:
    out = {pid}
    try:
        txt = subprocess.run(["ps", "-eo", "pid,ppid"], capture_output=True, text=True).stdout.splitlines()[1:]
        kids = {}
        for ln in txt:
            p, pp = (int(x) for x in ln.split())
            kids.setdefault(pp, []).append(p)
        stack = [pid]
        while stack:
            for c in kids.get(stack.pop(), []):
                if c not in out:
                    out.add(c)
                    stack.append(c)
    except Exception:
        pass
    return out


def parse_log(txt: str) -> dict:
    it = [float(x) for x in re.findall(r"Iteration time:\s*([0-9.]+)s", txt)]
    col = [float(x) for x in re.findall(r"Collection time:\s*([0-9.]+)s", txt)]
    lrn = [float(x) for x in re.findall(r"Learning time:\s*([0-9.]+)s", txt)]
    sps = [float(x) for x in re.findall(r"Steps per second:\s*([0-9.]+)", txt)]
    rss = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", txt)
    wall = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([0-9:.]+)", txt)

    def q(v, lo=2, hi=12):
        v = np.asarray(v[lo:hi], float)          # iterations 3-12 (0-based 2..11)
        if not v.size:
            return None
        return {"median": float(np.median(v)), "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75)),
                "n": int(v.size), "values": [float(x) for x in v]}
    out = {"iterations_logged": len(it), "iteration_s_all": it, "iteration_s": q(it), "collection_s": q(col),
           "learning_s": q(lrn), "steps_per_s": q(sps),
           "host_rss_peak_gb": round(int(rss.group(1)) / 1e6, 3) if rss else None, "wall": wall.group(1) if wall else None,
           "overflow_lines": len(re.findall(r"overflow", txt, flags=re.I)),
           "nan_lines": len(re.findall(r"\bnan\b", txt, flags=re.I))}
    return out


def cmd_probe(a) -> int:
    run = ROOT / RUNS[a.variant]
    pdir = LOGS / "rl_probes"
    pdir.mkdir(parents=True, exist_ok=True)
    suffix = f"_buf{a.buffers.replace(',', '-')}" if a.buffers else ""
    tag = f"20261014-probe_{a.variant}{suffix}_r{a.rep}"
    log = pdir / f"train_{a.variant}{suffix}_r{a.rep}.log"
    tsv = pdir / f"gpu_{a.variant}{suffix}_r{a.rep}.tsv"
    out = COST / "ppo_probes.jsonl"
    if out.exists() and any(json.loads(x).get("tag") == tag and json.loads(x).get("status") == "ok"
                            for x in out.read_text().splitlines() if x.strip()):
        print(f"{tag}: done", flush=True)
        return 0
    gpu0 = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.split()[0])
    trainer = [str(ROOT / "scripts/rl_train_cube.py")] if not a.buffers else \
        [str(Path(__file__).resolve()), "train", "--buffers", a.buffers, "--"]
    cmd = ["/usr/bin/time", "-v", sys.executable] + trainer + ["--morphology-run", str(run), "--tag", tag,
                                                               "--output-root", str(pdir)] + TRAIN_ARGS
    env = dict(os.environ, MUJOCO_GL="egl")
    stop = threading.Event()
    pid_box: list = []
    th = threading.Thread(target=gpu_sampler, args=(tsv, pid_box, stop), daemon=True)
    t0 = time.time()
    with open(log, "w") as fh:
        pr = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env)
        pid_box.append(pr.pid)
        th.start()
        rc = pr.wait()
    stop.set()
    th.join(timeout=10)
    wall = time.time() - t0
    txt = log.read_text(errors="replace")
    res = parse_log(txt)
    samp = [ln.split("\t") for ln in tsv.read_text().splitlines() if ln.strip()] if tsv.exists() else []
    tot = [int(s[1]) for s in samp if float(s[0]) >= t0]
    proc = [int(s[2]) for s in samp if float(s[0]) >= t0]
    row = {"tag": tag, "variant": a.variant, "rep": a.rep, "run": RUNS[a.variant], "rc": rc, "wall_s": round(wall, 1),
           "buffers": a.buffers or "env_build default",
           "gpu_before_mb": gpu0, "gpu_peak_mb": max(tot) if tot else None, "gpu_proc_peak_mb": max(proc) if proc else None,
           "gpu_samples": len(tot), "cpus": sorted(os.sched_getaffinity(0)), "train_args": TRAIN_ARGS,
           "when": time.strftime("%Y-%m-%d %H:%M"), **res}
    row["status"] = "ok" if rc == 0 and res["iterations_logged"] >= 12 and res["overflow_lines"] == 0 else "failed"
    append(out, row)
    it = res["iteration_s"] or {}
    print(f"{tag}: rc {rc}, {res['iterations_logged']} iterations, iteration {it.get('median')} s "
          f"[{it.get('q25')}-{it.get('q75')}], GPU peak {row['gpu_peak_mb']} MB (process {row['gpu_proc_peak_mb']}), "
          f"RSS {res['host_rss_peak_gb']} GB, overflow lines {res['overflow_lines']}", flush=True)
    return 0 if row["status"] == "ok" else 1


def cmd_queue(a) -> int:
    """Probes one after another, each `probe` in a fresh process with its own WARP_CACHE_PATH, waiting for the GPU
    memory of the last one to be released and for 13.5 GB free before a 1 mm pad probe. Jobs: variant:rep[:buffers]."""
    import tempfile

    def used_free():
        u, f = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.split(",")
        return int(u), int(f)
    def gpu_pids():
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout
        return {int(x) for x in out.split() if x.strip().isdigit()}
    base_pids = gpu_pids()
    for job in a.jobs:
        parts = job.split(":")
        v, rep = parts[0], parts[1]
        cmd = [sys.executable, str(Path(__file__).resolve()), "probe", "--variant", v, "--rep", rep]
        if len(parts) > 2:
            cmd += ["--buffers", parts[2]]
        need = 13500 if v == "servo_pads1" and len(parts) == 2 else 6000
        t_wait = time.time()
        while used_free()[1] < need and time.time() - t_wait < 1800:
            time.sleep(15)
        env = dict(os.environ, WARP_CACHE_PATH=tempfile.mkdtemp(prefix="warp_u1_probe_"), MUJOCO_GL="egl")
        t0 = time.time()
        rc = subprocess.run(cmd, env=env, cwd=str(ROOT)).returncode
        t1 = time.time()
        while gpu_pids() - base_pids and time.time() - t1 < 180:     # the probe's processes have left the GPU
            time.sleep(5)
        time.sleep(10)
        print(f"[queue] {job}: rc {rc}, {time.time() - t0:.0f} s, GPU used now {used_free()[0]} MB", flush=True)
    return 0


def cmd_train(a, rest) -> int:
    """rl_train_cube.py with env_build._contact_buffers replaced by exact per-world sizes (nconmax, njmax, sensor
    maxmatch), the probe's stand-in for overrides in env_build itself; env_build.py is not edited."""
    import runpy
    from morphohand.rl import env_build
    n, j, mm = (int(v) for v in a.buffers.split(","))
    env_build._contact_buffers = lambda cfg: (n, j, mm)
    print(f"[pad_usability_rl] contact buffers per world: nconmax {n}, njmax {j}, sensor maxmatch {mm}", flush=True)
    sys.argv = [str(ROOT / "scripts/rl_train_cube.py")] + rest
    runpy.run_path(str(ROOT / "scripts/rl_train_cube.py"), run_name="__main__")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scenes")
    s.add_argument("--variants", nargs="+", default=["servo_meshc4", "servo_pads2", "servo_mesh", "servo_pads1"])
    p = sub.add_parser("probe")
    p.add_argument("--variant", required=True, choices=list(RUNS))
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--buffers", default=None, help="nconmax,njmax,maxmatch per world in place of env_build's rule")
    q = sub.add_parser("queue")
    q.add_argument("--jobs", nargs="+", default=["servo_mesh:0", "servo_meshc4:0", "servo_pads2:0", "servo_mesh:1",
                                                  "servo_pads1:0", "servo_pads1:1"])
    t = sub.add_parser("train", help="internal: the trainer with exact contact buffers; trainer arguments after --")
    t.add_argument("--buffers", required=True)
    argv = sys.argv[1:]
    rest = []
    if "--" in argv:
        k = argv.index("--")
        argv, rest = argv[:k], argv[k + 1:]
    a = ap.parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", "egl")
    if a.cmd == "train":
        return cmd_train(a, rest)
    return {"scenes": cmd_scenes, "probe": cmd_probe, "queue": cmd_queue}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
