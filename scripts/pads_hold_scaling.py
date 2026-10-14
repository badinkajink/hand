#!/usr/bin/env python3
"""Same-fixture control for scripts/newton_scaling.py: 1 mm MuJoCo-Warp sphere pads on the hold.

Calls gpu_scaling_mjwarp.run_point unchanged (legacy mapping, 1 mm pads, 1 ms step, per-world
capacities 256 contacts / 1024 constraint rows, one CUDA graph per 10 ms block) with the torque
pulses off (torque_peak 0) and a 0.5 s duration, i.e. the 0.5 s hold under the 20 N/m centring
guide that newton_scaling.py times for Newton hydroelastic, at the same world counts.
Rows are appended to docs/experiments/20261005-gpu_scaling/newton_scaling.jsonl with
kind 'control_hold' and model 'mj_pads1_legacy'. A point runs only when no other compute process
is on the GPU (the resident gnome-remote-desktop-daemon excepted) and utilization is <= 20 %;
a point that saw another process is repeated once.
Run with the repo uv env (mjlab 1.2, MuJoCo Warp 3.6).
"""

import argparse
import gc
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

import mujoco
import warp as wp

from gpu_scaling_mjwarp import run_point

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/experiments/20261005-gpu_scaling/newton_scaling.jsonl"
IGNORED_APPS = ("gnome-remote-desktop-daemon",)


def snapshot():
    me = os.getpid()
    snap = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.free,clocks.sm",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
    u, used, free, clk = (float(x) for x in q.strip().split(","))
    snap.update(util_pct=u, used_mib=used, free_mib=free, sm_clock_mhz=clk)
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                           "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
    others, own = [], None
    for line in apps.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 3 or not p[0].isdigit():
            continue
        if int(p[0]) == me:
            own = float(p[2])
        elif not any(s in p[1] for s in IGNORED_APPS):
            others.append(dict(pid=int(p[0]), name=p[1], used_mib=float(p[2])))
    snap.update(own_mib=own, others=others)
    return snap


def contended(snap, after=False):
    return bool(snap["others"]) or (not after and snap["util_pct"] > 20.0)


def wait_clear(max_wait_s):
    t0 = time.time()
    while True:
        s = snapshot()
        if not contended(s):
            time.sleep(3.0)
            s = snapshot()
            if not contended(s):
                return s, time.time() - t0
        if time.time() - t0 > max_wait_s:
            return s, time.time() - t0
        print("GPU busy", [o["name"] for o in s["others"]], s["util_pct"], flush=True)
        time.sleep(15.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nworld", type=int, nargs="+", default=[1, 64, 256, 1024, 4096, 8192])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--max-wait-s", type=float, default=1800.0)
    args = ap.parse_args()
    try:
        wp.set_mempool_release_threshold("cuda:0", 0)
    except Exception:  # noqa: BLE001
        pass
    rows = [json.loads(x) for x in args.out.read_text().splitlines()] if args.out.exists() else []
    rev = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    for n in args.nworld:
        for rep in range(args.repeats):
            key = f"control_hold_mj_pads1_legacy_n{n}_r{rep}"
            if any(r.get("key") == key and not r.get("contention") and r.get("status") == "complete" for r in rows):
                continue
            pt = dict(torque_peak=0.0, duration=0.5, solver_tolerance=1e-8, policy="legacy", spacing_mm=1.0,
                      timestep=0.001, nworld=n, nconmax=256, njmax=1024)
            for attempt in range(2):
                before, waited = wait_clear(args.max_wait_s)
                row = dict(task="gpu_scaling", kind="control_hold", key=key, model="mj_pads1_legacy",
                           rig_spec="mj:spheres:s1.0:rs0.75:tr0.03:ir100 + apply_legacy",
                           fixture="SR2 thumb-index pinch, distributed_contact_transfer.build; 0.5 s hold, "
                                   "20 N/m centring guide, no torque (gpu_scaling_mjwarp.run_point, torque_peak 0)",
                           dt_ms=1.0, nworld=n, repeat=rep, attempt=attempt, config=pt, gpu_before=before,
                           waited_s=waited, script="scripts/pads_hold_scaling.py", git_rev=rev,
                           versions=dict(warp=wp.__version__, mujoco=mujoco.__version__))
                try:
                    row.update(run_point(pt))
                except Exception as e:  # noqa: BLE001
                    row.update(status="error", error=repr(e)[:600], traceback=traceback.format_exc()[-2000:])
                    gc.collect()
                row["gpu_after"] = snapshot()
                row["contention"] = contended(before) or contended(row["gpu_after"], after=True)
                with open(args.out, "a") as fh:
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                print(key, attempt, row["status"], f"{row.get('world_steps_per_s', 0):.0f} ws/s",
                      f"{row.get('ms_per_step_median', 0):.3f} ms", f"proc {row.get('vram_process_mib')} MiB",
                      f"con {row.get('peak_contacts_per_world')}", "CONTENDED" if row["contention"] else "", flush=True)
                if not row["contention"]:
                    break


if __name__ == "__main__":
    main()
