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
  cross     U5: each servo-plant final (TPU mesh and 1 mm pads, seeds 0 and 1) on the TPU-mesh, 1 mm pad, 2 mm pad and
            condim-4 plant scenes, nominal and at kp 0.36 N m/rad, a 0.12 N m torque limit and friction x0.5: closed loop
            (128 rollouts, grasp-pose noise 2 mm and 5 deg) and the open-loop replay of its 64 own-scene action sequences
            (docs/experiments/20261013-contact_feedback/acts/). Rows: docs/experiments/20261014-pad_usability/rl/cross.jsonl.
  numerics  U5: each final on its own scene at impratio 100 (written in place into the MuJoCo-Warp model) and at a 4 ms
            step (decimation 5), against the trainer's 2 ms and impratio 10. Rows: rl/numerics.jsonl.
  film      U5: the kp 0.36 cross-plant film, rl/media/20261014-rl_cross_kp036.mp4.
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


# ---------------------------------------------------------------------------------- U5: fingertip model in evaluation

RLX = ROOT / "docs/experiments/20261014-pad_usability/rl"
ACTS = ROOT / "docs/experiments/20261013-contact_feedback/acts"     # own-scene action sequences, 64 x 250 steps
FINALS = ("20261011-d6_servo_tpu27mesh_40M_s0", "20261011-d6_servo_tpu27mesh_40M_s1",
          "20261011-d6_servo_tpu27pads1_40M_s0", "20261011-d6_servo_tpu27pads1_40M_s1")
PLANT_SCENES = ("tpu27mesh", "tpu27pads1", "tpu27pads2", "tpu27meshc4")
CONDITIONS = {"nominal": (None, None), "kp=0.36": ("kp", 0.36), "fr=0.12": ("fr", 0.12),
              "friction=0.5": ("friction", 0.5)}
NOISE = (2.0, 5.0)              # closed-loop grasp-pose noise: uniform tool offset (mm) and yaw (deg)
NUMERICS = ("nominal", "impratio100", "dt4ms")


def eval_module():
    """rl_contact_eval with the 2026-10-11 study selected and the 2 mm pad and condim-4 scenes added as its arms."""
    import rl_contact_eval as E
    E.STUDIES["20261011"]["arms"].update(tpu27pads2=ROOT / RUNS["servo_pads2"], tpu27meshc4=ROOT / RUNS["servo_meshc4"])
    E.use_study("20261011")
    return E


def done_rows(path: Path, keys: tuple) -> set:
    if not path.exists():
        return set()
    return {tuple(r.get(k) for k in keys) for r in (json.loads(x) for x in path.read_text().splitlines() if x.strip())
            if r.get("status") == "ok"}


def cmd_cross(a) -> int:
    """Each final on each plant scene under each condition: closed loop (128 rollouts with grasp-pose noise) and the
    open-loop replay of its 64 own-scene action sequences (contact_feedback_replay.replay). One row per (tag, scene,
    mode, condition) in rl/cross.jsonl."""
    E = eval_module()
    import contact_feedback_replay as R
    out = RLX / "cross.jsonl"
    have = done_rows(out, ("tag", "scene", "mode", "condition"))
    pol = [j for j in E.finished() if j[0] in (a.tags or FINALS)]
    for scene in a.scenes:
        for mode in a.modes:
            todo = [(j, c) for j in pol for c in a.conditions if (j[0], scene, mode, c) not in have]
            if not todo:
                continue
            t_b = time.perf_counter()
            ev = E.Evaluator(E.ARMS[scene], E.final_ckpt(todo[0][0][0], todo[0][0][3]),
                             n=128 if mode == "closed" else 64, noise=NOISE if mode == "closed" else None)
            build_s = round(time.perf_counter() - t_b, 1)
            for (tag, arm, seed, steps), cond in todo:
                ck = E.final_ckpt(tag, steps)
                ev.load(ck)
                ev.perturb(*CONDITIONS[cond])
                print(f"[u5] {tag} on {scene}, {mode}, {cond}", flush=True)
                t0 = time.perf_counter()
                r = ev.rollout() if mode == "closed" else R.replay(ev, np.load(ACTS / f"{tag}.npz")["acts"])
                row = dict(tag=tag, arm=arm, seed=seed, iteration=E._it(ck), scene=scene, mode=mode, condition=cond,
                           noise_mm_deg=list(NOISE) if mode == "closed" else None, **E.metrics(r, ev.residual_from),
                           status="ok", build_s=build_s, wall_s=round(time.perf_counter() - t0, 1),
                           when=time.strftime("%Y-%m-%d %H:%M"))
                append(out, row)
                ev.perturb(None)
                print(f"[u5] {tag} on {scene}, {mode}, {cond}: held {row['n_held']}/{row['n']}, held cos "
                      f"{row['held_cos_mean']}", flush=True)
            ev.close()
    return 0


def set_impratio(ev, ir: float) -> float:
    """Write impratio into the MuJoCo-Warp model in place (opt.impratio_invsqrt, a per-world array that the captured
    CUDA graphs read at every step); returns the value read back."""
    arr = ev.u.sim.wp_model.opt.impratio_invsqrt
    arr.fill_(float(ir) ** -0.5)
    return float(np.asarray(arr.numpy()).ravel()[0]) ** -2


def cmd_numerics(a) -> int:
    """Each final on its own scene, closed loop (128 rollouts, grasp-pose noise, mech record for the hold-phase slide):
    the trainer's numerics (2 ms, impratio 10), impratio 100 written in place, and a 4 ms step (Evaluator(dt=0.004):
    decimation 5, same seconds of scripted grasp and lift). One row per (tag, numerics) in rl/numerics.jsonl."""
    E = eval_module()
    out = RLX / "numerics.jsonl"
    have = done_rows(out, ("tag", "numerics"))
    pol = [j for j in E.finished() if j[0] in (a.tags or FINALS)]
    for scene in ("tpu27mesh", "tpu27pads1"):
        mine = [j for j in pol if j[1] == scene]
        for dt, names in ((None, ("nominal", "impratio100")), (0.004, ("dt4ms",))):
            todo = [(j, nm) for j in mine for nm in names if nm in a.numerics and (j[0], nm) not in have]
            if not todo:
                continue
            t_b = time.perf_counter()
            ev = E.Evaluator(E.ARMS[scene], E.final_ckpt(todo[0][0][0], todo[0][0][3]), n=128, noise=NOISE, dt=dt,
                             mech=True)
            build_s = round(time.perf_counter() - t_b, 1)
            ir0 = set_impratio(ev, 10.0)
            for (tag, arm, seed, steps), nm in todo:
                ck = E.final_ckpt(tag, steps)
                ev.load(ck)
                ir = set_impratio(ev, 100.0 if nm == "impratio100" else 10.0)
                print(f"[u5] {tag} on {scene}, {nm} (impratio read back {ir:.4g})", flush=True)
                t0 = time.perf_counter()
                r = ev.rollout()
                set_impratio(ev, 10.0)
                mm = E.mech_metrics(r, ev.mech_consts())
                row = dict(tag=tag, arm=arm, seed=seed, iteration=E._it(ck), scene=scene, numerics=nm,
                           dt_s=dt or 0.002, impratio_set=100.0 if nm == "impratio100" else 10.0, impratio_read=ir,
                           impratio_build=ir0, noise_mm_deg=list(NOISE), **E.metrics(r, ev.residual_from, physics=True),
                           **{f"mech_{k}": v for k, v in mm.items()}, status="ok", build_s=build_s,
                           wall_s=round(time.perf_counter() - t0, 1), when=time.strftime("%Y-%m-%d %H:%M"))
                append(out, row)
                print(f"[u5] {tag} {nm}: held {row['n_held']}/128, held cos {row['held_cos_mean']}, hold slide "
                      f"{mm.get('slide_hold_path_um_s')} um/s", flush=True)
            ev.close()
    return 0


SCENE_NAMES = {"tpu27mesh": "TPU-mesh plant", "tpu27pads1": "1 mm pad plant", "tpu27pads2": "2 mm pad plant",
               "tpu27meshc4": "condim-4 plant"}
# film rows: (title, [(tag, scene, condition, dt), (tag, scene, condition, dt)])
FILM_ROWS = [("TPU-mesh policy, seed 0, servo gain halved (kp 0.36 N m/rad)",
              [("20261011-d6_servo_tpu27mesh_40M_s0", "tpu27mesh", "kp=0.36", None),
               ("20261011-d6_servo_tpu27mesh_40M_s0", "tpu27pads1", "kp=0.36", None)]),
             ("1 mm pad policy, seed 0, servo gain halved (kp 0.36 N m/rad)",
              [("20261011-d6_servo_tpu27pads1_40M_s0", "tpu27pads1", "kp=0.36", None),
               ("20261011-d6_servo_tpu27pads1_40M_s0", "tpu27mesh", "kp=0.36", None)])]


def cmd_film(a) -> int:
    """FILM_ROWS as one tiled film: per row a title bar and two panels, each the closed-loop world (of 128, grasp-pose
    noise) whose final cosine is the median, drawn by Evaluator.render_world at 480 x 360 with a label bar (plant and
    counts over the 128 rollouts) and a strip (time, cosine, height). Out: rl/media/20261014-rl_cross_kp036.mp4, .png
    poster at 75 % and a .json of the worlds shown."""
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    E = eval_module()
    jobs = {j[0]: j for j in E.finished()}
    W_, H_ = 480, 360
    f_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    f_lab = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19)
    TITLE_H, LAB_H, STRIP_H = 38, 32, 30
    rows, meta = [], []
    for title, panels in FILM_ROWS:
        out_panels = []
        for tag, scene, cond, dt in panels:
            _, arm, seed, steps = jobs[tag]
            ck = E.final_ckpt(tag, steps)
            ev = E.Evaluator(E.ARMS[scene], ck, n=128, noise=NOISE, dt=dt, watch=True)
            ev.load(ck)
            ev.perturb(*CONDITIONS[cond])
            r = ev.rollout()
            m = E.metrics(r, ev.residual_from)
            fc = r["cos"][-1]
            w = int(np.argsort(fc)[len(fc) // 2])
            held_t = (r["z"] > E.FLOOR_Z) & ((r["force"] >= E.HELD_N).sum(-1) >= 2)
            head = f"{SCENE_NAMES[scene]}: held {m['n_held']}/128" + (
                f", held cos {m['held_cos_mean']:.2f}" if m["held_cos_mean"] is not None else "")
            ks = list(range(0, 250, 2))
            frames = []
            for k, fr in zip(ks, ev.render_world(r, w, ks, W_, H_)):
                im = Image.new("RGB", (W_, LAB_H + H_), "white")
                im.paste(Image.fromarray(fr[:, :, :3]), (0, LAB_H))
                dr = ImageDraw.Draw(im, "RGBA")
                dr.text((10, 5), head, font=f_lab, fill=(0, 0, 0))
                dr.line([(0, LAB_H - 1), (W_, LAB_H - 1)], fill=(0, 0, 0), width=1)
                dropped = k >= ev.residual_from and not held_t[k, w]
                txt = (f"t {0.02 * (k + 1):.2f} s   cos {r['cos'][k, w]:+.2f}   z {1e3 * r['z'][k, w]:.0f} mm"
                       + ("   dropped" if dropped else ""))
                dr.rectangle([0, LAB_H + H_ - STRIP_H, W_, LAB_H + H_], fill=(255, 255, 255, 215))
                dr.text((10, LAB_H + H_ - STRIP_H + 4), txt, font=f_lab, fill=(150, 0, 0) if dropped else (0, 0, 0))
                frames.append(np.asarray(im))
            out_panels.append(frames)
            meta.append(dict(row=title, tag=tag, scene=scene, condition=cond, dt=dt, world=w, n_held=m["n_held"],
                             held_cos=m["held_cos_mean"], final_cos=float(fc[w]), held_end=bool(held_t[-1, w])))
            ev.close()
            print(f"[film] {tag} on {scene} {cond}: world {w}, held {m['n_held']}/128", flush=True)
        bar = Image.new("RGB", (2 * W_, TITLE_H), "white")
        ImageDraw.Draw(bar).text((10, 7), title, font=f_title, fill=(0, 0, 0))
        bar = np.asarray(bar)
        sep = np.zeros((LAB_H + H_, 2, 3), np.uint8)
        rows.append([np.concatenate([bar, np.concatenate([x[:, :-1], sep, y[:, 1:]], axis=1)], axis=0)
                     for x, y in zip(*out_panels)])
    frames = [np.concatenate(fr, axis=0) for fr in zip(*rows)]
    pad = (-frames[0].shape[0]) % 8
    if pad:
        frames = [np.concatenate([f, np.full((pad, f.shape[1], 3), 255, np.uint8)], axis=0) for f in frames]
    media = RLX / "media"
    media.mkdir(parents=True, exist_ok=True)
    out = media / "20261014-rl_cross_kp036.mp4"
    wr = imageio.get_writer(str(out), fps=25, codec="libx264", quality=None, pixelformat="yuv420p", macro_block_size=8,
                            ffmpeg_params=["-crf", "23", "-preset", "medium"], ffmpeg_log_level="error")
    for f in frames:
        wr.append_data(np.asarray(f, np.uint8))
    wr.close()
    Image.fromarray(frames[int(0.75 * (len(frames) - 1))]).save(out.with_suffix(".png"))
    out.with_suffix(".json").write_text(json.dumps(dict(panels=meta, when=time.strftime("%Y-%m-%d %H:%M")), indent=1))
    print(f"wrote {out.relative_to(ROOT)} ({len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]})", flush=True)
    return 0


def cmd_irtest(a) -> int:
    """Check that set_impratio reaches the constraint rows: after a closed-loop rollout on the 1 mm pad plant, one env step
    at impratio 10 and one at 100; per contact the friction rows' constraint mass D (1/R) over its normal row's. With
    elliptic cones MuJoCo-Warp scales the friction rows' diagonal by 1/impratio, so the ratio rises tenfold."""
    E = eval_module()
    tag = FINALS[3]
    j = next(x for x in E.finished() if x[0] == tag)
    ck = E.final_ckpt(tag, j[3])
    ev = E.Evaluator(E.ARMS["tpu27pads1"], ck, n=16)
    ev.load(ck)
    ev.rollout()
    import torch
    zero = torch.zeros((16, 9), device=ev.u.device)
    out = {}
    for ir in (10.0, 100.0, 10.0):
        read = set_impratio(ev, ir)
        ev.wrapped.step(zero)
        wd = ev.u.sim.wp_data
        nc = int(wd.nacon.numpy()[0])
        adr = wd.contact.efc_address.numpy()[:nc]
        dim = wd.contact.dim.numpy()[:nc]
        wid = wd.contact.worldid.numpy()[:nc]
        D = wd.efc.D.numpy()
        rat = [float(D[w, ad[1]] / D[w, ad[0]]) for w, ad, dm in zip(wid, adr, dim) if dm >= 3 and ad[0] >= 0 and ad[1] >= 0
               and D[w, ad[0]] > 0]
        out.setdefault("steps", []).append(dict(impratio_set=ir, impratio_read=read, contacts=len(rat),
                                                ratio_median=float(np.median(rat)) if rat else None))
        print(f"[irtest] impratio {read:g}: {len(rat)} contacts, friction D over normal D median "
              f"{out['steps'][-1]['ratio_median']}", flush=True)
    s = out["steps"]
    out.update(tag=tag, scene="tpu27pads1", rise=(s[1]["ratio_median"] / s[0]["ratio_median"]) if s[0]["ratio_median"] else None,
               when=time.strftime("%Y-%m-%d %H:%M"))
    (RLX / "impratio_check.json").write_text(json.dumps(out, indent=1) + "\n")
    ev.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("film", help="U5: the kp 0.36 cross-plant film (FILM_ROWS)")
    sub.add_parser("irtest", help="U5: does the in-place impratio write reach the friction rows?")
    c = sub.add_parser("cross", help="U5: finals x plant scenes x conditions, closed loop and replays")
    c.add_argument("--tags", nargs="*", default=[])
    c.add_argument("--scenes", nargs="+", default=list(PLANT_SCENES), choices=PLANT_SCENES)
    c.add_argument("--modes", nargs="+", default=["closed", "open"], choices=["closed", "open"])
    c.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    nm = sub.add_parser("numerics", help="U5: impratio 100 and a 4 ms step on each final's own scene")
    nm.add_argument("--tags", nargs="*", default=[])
    nm.add_argument("--numerics", nargs="+", default=list(NUMERICS), choices=NUMERICS)
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
    return {"scenes": cmd_scenes, "probe": cmd_probe, "queue": cmd_queue, "cross": cmd_cross,
            "numerics": cmd_numerics, "film": cmd_film, "irtest": cmd_irtest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
