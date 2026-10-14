#!/usr/bin/env python3
"""GPU cost per world-step of the fingertip contact models at the servo plant's held RL state (U1, 2026-10-14).

Variants (morphology runs of the servo-limited plant kp0.72_kv0_fr0.18_fl0_dp0.0144; scripts/pad_usability_rl.py
scenes writes the two 2026-10-14 ones):
  servo_mesh    the TPU block as one convex mesh, point contact (condim 3)
  servo_meshc4  the mesh with condim 4, priority 1, torsional friction 2.9 mm
  servo_pads2   2 mm sphere pads of radius 1.5 mm, touching the tool only
  servo_pads1   1 mm sphere pads (radius 0.75 mm), touching the tool only

  export     rl_state_export.export(variant, 64, 60) with its OUT set to logs/20261014-pad_usability: the RL env's
             scripted grasp and lift at zero residual action, 64 worlds, state after 60 policy steps (scene.xml with the
             trainer's options, state.npz, state.json). Added here: the broadphase candidate count of the last physics
             step (MuJoCo-Warp writes candidate pairs into the contact buffer, so a buffer sized from contacts alone
             drops pad contacts), scene_key.xml (world 0's qpos and commands as keyframe 0, for mjwarp-testspeed), and
             the buffer sizes the timing uses: nconmax = ceil(1.5 max(largest per-world contact count, broadphase
             candidates per world)), njmax = ceil(1.5 x largest per-world constraint-row count).
             Rows: docs/experiments/20261014-pad_usability/cost/gpu_export.jsonl
  time       same_state_timing.run_mjw on that state (its STATES and OPTS set here; --train-buffers: nconmax and
             njmax of env_build._contact_buffers, 640 / 3,072 above 500 geoms, 64 / 400 below): the 64 exported configurations
             tiled over `--nworld` worlds at rest, the trainer's solver options plus mjlab's parallel line search and 50
             CCD iterations, 50-step CUDA-graph blocks, one compile block, one captured block, then `--blocks` timed
             replays. Per point: us per world-step, contacts, constraint rows and broadphase candidates per world, the
             fraction of worlds whose tool stayed within 20 mm, the process's peak GPU memory (nvidia-smi, sampled every
             0.2 s) and buffer overflow. A point whose memory estimate (the variant's largest measured point scaled by
             world count) exceeds free GPU memory minus 2 GB is written as skipped. One process per point.
             Rows: docs/experiments/20261014-pad_usability/cost/gpu_timing.jsonl
  testspeed  .venv/bin/mjwarp-testspeed on scene_key.xml (keyframe 0, `--nworld`, `--nstep`, event trace, contact and
             row counts, memory report; mjlab's ls_parallel and ccd_iterations 50 as overrides) with JSON output.
             Rows: docs/experiments/20261014-pad_usability/cost/gpu_testspeed.jsonl; raw output in the logs folder.

    MUJOCO_GL=egl WARP_CACHE_PATH=$(mktemp -d) ~/.claude/bin/resguard.sh run --mem 6G --cpu 400 -- taskset -c 17-19 \\
        uv run --extra rl --extra gpu python scripts/pad_usability_gpu_cost.py export --variants servo_mesh servo_meshc4 \\
        servo_pads2 servo_pads1
    ... pad_usability_gpu_cost.py time --variant servo_pads1 --nworld 2048 --rep 0
    ... pad_usability_gpu_cost.py testspeed --variant servo_pads1 --nworld 2048 --nstep 200
    ... pad_usability_gpu_cost.py sweep --testspeed      (every time point and testspeed, one fresh process each)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import pad_usability_rl as PR  # noqa: E402

COST = ROOT / "docs/experiments/20261014-pad_usability/cost"
LOGS = ROOT / "logs/20261014-pad_usability"
RUNS = PR.RUNS
MARGIN_MB = 2048


def append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(row, default=float) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def smi(query: str) -> str:
    return subprocess.run(["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"], capture_output=True,
                          text=True, timeout=20).stdout


def gpu_used_free() -> tuple[int, int]:
    u, f = smi("gpu=memory.used,memory.free").split(",")
    return int(u), int(f)


def proc_gpu_mb(pid: int) -> int:
    out = 0
    for ln in smi("compute-apps=pid,used_memory").splitlines():
        if ln.strip():
            a, b = ln.split(",")
            if int(a) == pid:
                out += int(b)
    return out


class PeakSampler:
    """Peak GPU memory of this process and of the whole GPU, sampled every `every` s in a thread."""

    def __init__(self, every=0.2):
        self.every, self.peak_proc, self.peak_total = every, 0, 0
        self.stop = threading.Event()
        self.th = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop.is_set():
            try:
                self.peak_proc = max(self.peak_proc, proc_gpu_mb(os.getpid()))
                self.peak_total = max(self.peak_total, gpu_used_free()[0])
            except Exception:
                pass
            self.stop.wait(self.every)

    def __enter__(self):
        self.th.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.th.join(timeout=5)


# ---------------------------------------------------------------------------------------------------- export

def write_scene_key(sdir: Path) -> Path:
    """scene.xml with world 0's exported qpos, commands and activations as keyframe 0 (qvel zero)."""
    import mujoco
    st = dict(np.load(sdir / "state.npz"))
    root = ET.parse(sdir / "scene.xml").getroot()
    for kf in root.findall("keyframe"):
        root.remove(kf)
    kf = ET.SubElement(root, "keyframe")
    attrs = {"name": "u1_held", "qpos": " ".join(f"{v:.9g}" for v in st["qpos"][0])}
    if st["ctrl"].size:
        attrs["ctrl"] = " ".join(f"{v:.9g}" for v in st["ctrl"][0])
    if st["act"].size:
        attrs["act"] = " ".join(f"{v:.9g}" for v in st["act"][0])
    ET.SubElement(kf, "key", attrs)
    out = sdir / "scene_key.xml"
    ET.ElementTree(root).write(out)
    m = mujoco.MjModel.from_xml_path(str(out))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    return out


def cmd_export(a) -> int:
    import rl_state_export as E
    T = E.T
    E.OUT = LOGS
    T.RUNS.update(RUNS)
    cap = {}
    orig = T.make_env

    def make_env(*args, **kw):
        env = orig(*args, **kw)
        close0 = env.close

        def close():
            wd = env.sim.wp_data
            cap["ncollision"] = int(wd.ncollision.numpy()[0])
            cap["nacon"] = int(wd.nacon.numpy()[0])
            cap["naconmax"] = int(wd.naconmax)
            cap["nworld"] = int(env.num_envs)
            close0()
        env.close = close
        return env
    T.make_env = make_env
    out = COST / "gpu_export.jsonl"
    for v in a.variants:
        sdir = LOGS / f"state_{v}"
        if (sdir / "state.json").exists() and (sdir / "scene_key.xml").exists() and not a.force:
            print(f"{v}: exported already", flush=True)
            continue
        cap.clear()
        t0 = time.perf_counter()
        E.export(v, a.envs, a.steps)
        meta = json.loads((sdir / "state.json").read_text())
        st = dict(np.load(sdir / "state.npz"))
        cand = cap["ncollision"] / cap["nworld"]
        ncon_max, nefc_max = int(st["ncon_world"].max()), int(st["nefc_world"].max())
        meta.update(ncollision_total=cap["ncollision"], ncollision_per_world=cand, export_naconmax=cap["naconmax"],
                    ncon_world_max=ncon_max, nefc_world_max=nefc_max,
                    nconmax_timing=int(math.ceil(1.5 * max(ncon_max, cand))),
                    njmax_timing=int(math.ceil(1.5 * nefc_max)))
        (sdir / "state.json").write_text(json.dumps(meta, indent=1))
        key = write_scene_key(sdir)
        held = (st["tip_tool_contacts"] > 0).sum(1) >= 2
        row = {"variant": v, "run": RUNS[v], "state_dir": str(sdir.relative_to(ROOT)), "scene_key": str(key.relative_to(ROOT)),
               "num_envs": a.envs, "policy_steps": a.steps, "ngeom": meta["ngeom"], "held_worlds": int(held.sum()),
               "tool_z_mm_median": meta["tool_z_mm_median"], "ncon_world_mean": float(st["ncon_world"].mean()),
               "ncon_world_max": ncon_max, "nefc_world_mean": float(st["nefc_world"].mean()), "nefc_world_max": nefc_max,
               "candidates_per_world": cand, "tip_tool_contacts_mean": meta["tip_tool_contacts_mean"],
               "nconmax_timing": meta["nconmax_timing"], "njmax_timing": meta["njmax_timing"],
               "option": meta["option"], "wall_s": round(time.perf_counter() - t0, 1),
               "when": time.strftime("%Y-%m-%d %H:%M"), "status": "ok"}
        append(out, row)
        print(json.dumps({k: row[k] for k in ("variant", "ngeom", "held_worlds", "ncon_world_mean", "ncon_world_max",
                                              "nefc_world_max", "candidates_per_world", "nconmax_timing",
                                              "njmax_timing")}), flush=True)
    return 0


# ---------------------------------------------------------------------------------------------------- time

def estimate_mb(rows, variant, nworld):
    """Process GPU memory of the variant's largest measured point, scaled by world count (None if none measured)."""
    ok = [r for r in rows if r["variant"] == variant and r.get("status") == "ok" and r.get("gpu_proc_peak_mb")]
    if not ok:
        return None
    r = max(ok, key=lambda x: x["nworld"])
    return int(r["gpu_proc_peak_mb"] * nworld / r["nworld"])


def cmd_time(a) -> int:
    import same_state_timing as SS
    out = COST / "gpu_timing.jsonl"
    rows = [json.loads(x) for x in out.read_text().splitlines() if x.strip()] if out.exists() else []
    buffers = "training" if a.train_buffers else "exact"
    if any(r["variant"] == a.variant and r["nworld"] == a.nworld and r.get("rep", 0) == a.rep
           and r.get("buffers", "exact") == buffers and r.get("status") in ("ok", "skipped") for r in rows):
        print(f"{a.variant} {a.nworld} rep {a.rep}: done", flush=True)
        return 0
    SS.STATES = LOGS
    sdir, meta, st = SS.load_state(a.variant)
    nconmax, njmax = int(meta["nconmax_timing"]), int(meta["njmax_timing"])
    if a.train_buffers:          # env_build._contact_buffers' rule: 640 / 3,072 above 500 geoms, else 64 / 400
        nconmax, njmax = (640, 3072) if int(meta["ngeom"]) > 500 else (64, 400)
    SS.OPTS.update(mjlab=True, warm=False, nconmax=nconmax, njmax=njmax)
    used0, free0 = gpu_used_free()
    est = estimate_mb([r for r in rows if r.get("buffers", "exact") == buffers], a.variant, a.nworld)
    row = {"variant": a.variant, "run": RUNS[a.variant], "nworld": a.nworld, "rep": a.rep, "blocks": a.blocks,
           "buffers": buffers,
           "state": str(sdir.relative_to(ROOT)), "nconmax": SS.OPTS["nconmax"], "njmax": SS.OPTS["njmax"],
           "gpu_used_before_mb": used0, "gpu_free_before_mb": free0, "mem_estimate_mb": est,
           "cpus": sorted(os.sched_getaffinity(0)), "when": time.strftime("%Y-%m-%d %H:%M")}
    if est is not None and est > free0 - MARGIN_MB:
        row.update(status="skipped", reason=f"estimate {est} MB > free {free0} MB - {MARGIN_MB} MB")
        append(out, row)
        print(f"{a.variant} {a.nworld}: skipped ({row['reason']})", flush=True)
        return 0
    import mujoco_warp
    import warp
    cap = {}
    put0 = mujoco_warp.put_data

    def put_data(*args, **kw):
        cap["d"] = put0(*args, **kw)
        return cap["d"]
    mujoco_warp.put_data = put_data
    try:
        xml, m = SS.engine_xml(sdir)
        m.opt.ccd_iterations = 50
        row.update(dt=float(m.opt.timestep), iterations=int(m.opt.iterations), ls_iterations=int(m.opt.ls_iterations),
                   tolerance=float(m.opt.tolerance), impratio=float(m.opt.impratio), ccd_iterations=50,
                   warp=warp.__version__, mujoco_warp=getattr(mujoco_warp, "__version__", "?"),
                   exported_ncon_world=float(st["ncon_world"].mean()), exported_nefc_world=float(st["nefc_world"].mean()))
        t0 = time.perf_counter()
        with PeakSampler() as ps:
            r = SS.run_mjw(xml, m, st, meta, a.nworld, a.blocks)
        wd = cap["d"]
        ncoll = int(wd.ncollision.numpy()[0])
        nacon = int(wd.nacon.numpy()[0])
        bp = SS.OPTS["broadphase_used"]
        try:
            bp = mujoco_warp.BroadphaseType(int(bp)).name
        except (ValueError, TypeError):
            pass
        row.update(r, wall_s=round(time.perf_counter() - t0, 1), broadphase=bp,
                   candidates_per_world=ncoll / a.nworld, naconmax=int(wd.naconmax),
                   overflow=bool(ncoll > wd.naconmax or nacon >= wd.naconmax or r["nefc_world_max"] >= SS.OPTS["njmax"]),
                   gpu_proc_peak_mb=ps.peak_proc, gpu_total_peak_mb=ps.peak_total, status="ok")
        print(f"{a.variant:<13} {a.nworld:>5} rep {a.rep}: {r['us_per_world_step']:.3f} us/world-step, "
              f"{r['contacts_per_world']:.1f} contacts, {ncoll / a.nworld:.1f} candidates, {r['nefc_world_mean']:.0f} "
              f"rows/world, held {r['held_frac']:.3f}, GPU {ps.peak_proc} MB, overflow {row['overflow']}", flush=True)
    except Exception as e:
        row.update(status="error", error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-1500:])
        print(f"{a.variant} {a.nworld}: {row['error']}", flush=True)
    finally:
        mujoco_warp.put_data = put0
    append(out, row)
    return 0 if row["status"] == "ok" else 1


# ---------------------------------------------------------------------------------------------------- testspeed

def cmd_testspeed(a) -> int:
    out = COST / "gpu_testspeed.jsonl"
    sdir = LOGS / f"state_{a.variant}"
    meta = json.loads((sdir / "state.json").read_text())
    nconmax = int(meta["nconmax_timing"])
    njmax = int(a.njmax) if a.njmax else int(math.ceil(a.njmax_mult * meta["nefc_world_max"]))
    raw = LOGS / f"testspeed_{a.variant}_{a.nworld}_nj{a.njmax if a.njmax else f'{a.njmax_mult:g}x'}.txt"
    cmd = [str(ROOT / ".venv/bin/mjwarp-testspeed"), str(sdir / "scene_key.xml"), "--keyframe", "0", "--nworld",
           str(a.nworld), "--nstep", str(a.nstep), "--event_trace", "--measure_alloc", "--measure_solver", "--memory",
           "--nconmax", str(nconmax), "--njmax", str(njmax), "--format", "json",
           "-o", "opt.ls_parallel=True", "-o", "opt.ccd_iterations=50"]
    t0 = time.perf_counter()
    pr = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    raw.write_text(" ".join(cmd) + "\n\n" + pr.stdout + "\n" + pr.stderr)
    nover = pr.stdout.count("overflow")
    row = {"variant": a.variant, "nworld": a.nworld, "nstep": a.nstep, "nconmax": nconmax, "njmax": njmax,
           "njmax_mult": None if a.njmax else a.njmax_mult, "overflow_lines": nover, "control_noise": "testspeed's Ornstein-Uhlenbeck "
           "command noise, std 0.01 x half the command range, 0.1 s", "raw": str(raw.relative_to(ROOT)),
           "rc": pr.returncode, "wall_s": round(time.perf_counter() - t0, 1), "when": time.strftime("%Y-%m-%d %H:%M")}
    js = None
    for ln in pr.stdout.splitlines()[::-1]:
        if ln.strip().startswith("{"):
            try:
                js = json.loads(ln)
                break
            except json.JSONDecodeError:
                pass
    if pr.returncode == 0 and js is not None:
        row.update(metrics=js, status="ok" if nover == 0 else "overflow")
        print(f"{a.variant} {a.nworld}: {1e6 / js['steps_per_second']:.3f} us/world-step (event trace on), "
              f"ncon {js.get('ncon_mean')}, nefc {js.get('nefc_mean')}, memory {js.get('total_memory', 0) / 2**20:.0f} MiB",
              flush=True)
    else:
        row.update(status="error", error=(pr.stderr or pr.stdout)[-1500:])
        print(f"{a.variant}: testspeed failed rc {pr.returncode}", flush=True)
    append(out, row)
    return 0 if row["status"] == "ok" else 1


def wait_gpu_release(pid: int, used_before: int, timeout: float = 120.0) -> int:
    """Wait until process `pid` holds no GPU memory and the GPU's used memory is back near `used_before`."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if proc_gpu_mb(pid) == 0 and gpu_used_free()[0] <= used_before + 150:
                break
        except Exception:
            pass
        time.sleep(2)
    return gpu_used_free()[0]


def cmd_sweep(a) -> int:
    """Each (variant, worlds, repeat) of `time`, then `testspeed` per variant, each in a fresh process with its own
    WARP_CACHE_PATH, one at a time, waiting for the GPU memory of the last one to be released."""
    import tempfile
    jobs = []
    for v in ([] if a.testspeed_only else a.variants):
        for n in a.nworld:
            for k in range(a.reps if n == a.rep_nworld else 1):
                jobs.append(["time", "--variant", v, "--nworld", str(n), "--rep", str(k)])
    if a.testspeed or a.testspeed_only:
        jobs += [["testspeed", "--variant", v, "--nworld", str(a.rep_nworld), "--nstep", "200", "--njmax-mult",
                  str(a.njmax_mult)] for v in a.variants]
    for job in jobs:
        used0 = gpu_used_free()[0]
        env = dict(os.environ, WARP_CACHE_PATH=tempfile.mkdtemp(prefix="warp_u1_"), MUJOCO_GL="egl")
        t0 = time.time()
        pr = subprocess.Popen([sys.executable, str(Path(__file__).resolve())] + job, env=env, cwd=str(ROOT))
        rc = pr.wait()
        used1 = wait_gpu_release(pr.pid, used0)
        print(f"[sweep] {' '.join(job)}: rc {rc}, {time.time() - t0:.0f} s, GPU used {used0} -> {used1} MB", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("sweep")
    w.add_argument("--variants", nargs="+", default=list(RUNS))
    w.add_argument("--nworld", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    w.add_argument("--reps", type=int, default=2, help="repeats at --rep-nworld")
    w.add_argument("--rep-nworld", type=int, default=2048)
    w.add_argument("--testspeed", action="store_true")
    w.add_argument("--testspeed-only", action="store_true")
    w.add_argument("--njmax-mult", type=float, default=1.5)
    e = sub.add_parser("export")
    e.add_argument("--variants", nargs="+", default=list(RUNS))
    e.add_argument("--envs", type=int, default=64)
    e.add_argument("--steps", type=int, default=60)
    e.add_argument("--force", action="store_true")
    t = sub.add_parser("time")
    t.add_argument("--variant", required=True, choices=list(RUNS))
    t.add_argument("--nworld", type=int, required=True)
    t.add_argument("--rep", type=int, default=0)
    t.add_argument("--blocks", type=int, default=20)
    t.add_argument("--train-buffers", action="store_true",
                   help="the trainer's per-world buffers (env_build rule) in place of 1.5x the exported maxima")
    s = sub.add_parser("testspeed")
    s.add_argument("--variant", required=True, choices=list(RUNS))
    s.add_argument("--nworld", type=int, default=2048)
    s.add_argument("--nstep", type=int, default=200)
    s.add_argument("--njmax-mult", type=float, default=1.5,
                   help="njmax as this multiple of the exported per-world maximum (testspeed adds command noise)")
    s.add_argument("--njmax", type=int, default=None, help="njmax per world (overrides --njmax-mult)")
    a = ap.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    return {"export": cmd_export, "time": cmd_time, "testspeed": cmd_testspeed, "sweep": cmd_sweep}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
