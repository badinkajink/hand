#!/usr/bin/env python3
"""Throughput of the RL training env with each fingertip contact model (2026-10-06).

The env is the D6 reorientation's mjlab env on the calibrated plant (`deploy.make_env_cfg`, as
`bench_env_throughput.py` builds it: scripted grasp and lift, zero residual actions), on morphology runs that differ
only in the fingertip (`make_pad_morphology_run.py`):

  legacy    the 10.6 x 21.2 x 15 mm box tip every RL run so far trained on (MuJoCo point contact)
  mesh      the TPU block (2.7 mm fillets) as one convex mesh
  pads2     TPU block, sphere pads at 2 mm spacing (290 per tip)
  pads1f    TPU block, 1 mm pads on the front half of the block (684 per tip)
  pads1     TPU block, 1 mm pads on every face but the back and top (1059 per tip)

For each variant a calibration pass (nconmax 1024, njmax 4096 per world, 80 policy steps at 256 envs) measures the
largest per-world contact and constraint counts; the timed env then allocates 1.5x those. Timed after warm-up:

  env steps/s      policy steps per second over all envs (10 physics steps of 2 ms + observations and rewards)
  physics us       wall time of one batched physics step (env.sim.step) divided by the number of envs
  GPU MB           nvidia-smi memory in use with the timed env alive (Warp allocations included)

    WARP_CACHE_PATH=$(mktemp -d) uv run --extra rl --extra gpu python scripts/rl_contact_throughput.py \\
        --variants legacy pads1 --envs 1024 4096
Rows: docs/experiments/20261006-rl_contact/throughput.jsonl, one fsynced line per variant x batch.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/experiments/20261006-rl_contact/throughput.jsonl"
RUNS = {
    "legacy": "results/phase1/real_v1/20260920-sv1_u0308_b050_cal_tip",
    "mesh": "results/phase1/real_v1/20261006-sv1_u0308_b050_cal_tip_tpu2.7mesh",
    "pads2": "results/phase1/real_v1/20261006-sv1_u0308_b050_cal_tip_tpu2.7pads2",
    "pads1f": "results/phase1/real_v1/20261006-sv1_u0308_b050_cal_tip_tpu2.7pads1f",
    "pads1": "results/phase1/real_v1/20261006-sv1_u0308_b050_cal_tip_tpu2.7pads1",
}


def base_cfg(run: Path):
    from morphohand.rl.deploy import make_env_cfg
    keyframe = json.load(open(run / "summary.json")).get("keyframe", "open_ik")
    bfc = tuple(float(v) for v in np.load(run / "best_rollout.npz")["best_finger_ctrl"].reshape(-1))
    return make_env_cfg(run / "frozen_scene.xml", keyframe, run, bfc, enable_target_axis=True, num_steps=240,
                        open_finger_from_keyframe=True, num_envs=1)


def make_env(cfg, num_envs, nconmax, njmax, impratio=None):
    import torch  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from morphohand.rl.env_cfg import to_mjlab_cfg
    mj = to_mjlab_cfg(dataclasses.replace(cfg, num_envs=num_envs))
    mj.sim.nconmax = int(nconmax)
    mj.sim.njmax = int(njmax)
    if impratio is not None:
        mj.sim.mujoco.impratio = float(impratio)
    return ManagerBasedRlEnv(cfg=mj, device="cuda:0", render_mode=None)


def counts(env):
    """Largest per-world contact and constraint counts in the current state, and the total contact count."""
    d = env.sim.wp_data
    nacon = int(d.nacon.numpy()[0])
    wid = d.contact.worldid.numpy()[:nacon] if nacon else np.zeros(0, int)
    per_world = int(np.bincount(wid, minlength=env.num_envs).max()) if nacon else 0
    nefc = int(d.nefc.numpy().max())
    return per_world, nefc, nacon


def calibrate(cfg, steps=80, n=256, impratio=None):
    import torch
    env = make_env(cfg, n, 1024, 4096, impratio)
    try:
        act = torch.zeros((n, env.action_manager.total_action_dim), device="cuda:0")
        env.reset()
        mx_c = mx_e = 0
        for k in range(steps):
            env.step(act)
            if k % 4 == 0:
                c, e, _ = counts(env)
                mx_c, mx_e = max(mx_c, c), max(mx_e, e)
    finally:
        env.close()
        del env
        torch.cuda.empty_cache()
    return mx_c, mx_e


def timed(cfg, num_envs, nconmax, njmax, steps, warmup, phys_steps, impratio=None):
    import torch
    torch.cuda.reset_peak_memory_stats()
    env = make_env(cfg, num_envs, nconmax, njmax, impratio)
    try:
        act = torch.zeros((num_envs, env.action_manager.total_action_dim), device="cuda:0")
        env.reset()
        for _ in range(warmup):
            env.step(act)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        mx_c = mx_e = mx_tot = 0
        for k in range(steps):
            env.step(act)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        c, e, tot = counts(env)
        mx_c, mx_e, mx_tot = max(mx_c, c), max(mx_e, e), max(mx_tot, tot)
        t1 = time.perf_counter()
        for _ in range(phys_steps):
            env.sim.step()
        torch.cuda.synchronize()
        dp = time.perf_counter() - t1
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        gpu_mb = gpu_used_mb()
    finally:
        env.close()
        del env
        torch.cuda.empty_cache()
    return {"env_steps_per_s": num_envs * steps / dt, "physics_us_per_world_step": 1e6 * dp / phys_steps / num_envs,
            "physics_ms_per_batch_step": 1e3 * dp / phys_steps, "peak_gib": peak, "gpu_used_mb": gpu_mb,
            "ncon_world_max": mx_c,
            "nefc_world_max": mx_e, "ncon_total": mx_tot, "overflow": bool(mx_tot >= nconmax * num_envs or
                                                                         mx_e >= njmax)}


def gpu_used_mb():
    try:
        return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True).stdout.split()[0])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=list(RUNS))
    ap.add_argument("--envs", type=int, nargs="+", default=[1024, 2048, 4096])
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=60, help="policy steps; covers the grasp so contacts exist")
    ap.add_argument("--phys-steps", type=int, default=200)
    ap.add_argument("--impratio", type=float, default=None, help="default: the trainer's 10")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for v in a.variants:
        run = ROOT / RUNS[v]
        cfg = base_cfg(run)
        import mujoco
        m = mujoco.MjModel.from_xml_path(str(run / "frozen_scene.xml"))
        try:
            cmax, emax = calibrate(cfg, impratio=a.impratio)
        except Exception as e:
            print(f"{v}: calibration failed: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            continue
        nconmax = max(16, math.ceil(1.5 * cmax))
        njmax = max(64, math.ceil(1.5 * emax))
        print(f"{v}: ngeom {m.ngeom}, calibration max {cmax} contacts / {emax} constraints per world -> "
              f"nconmax {nconmax}, njmax {njmax}", flush=True)
        for n in a.envs:
            from morphohand.rl import env_build
            row = {"variant": v, "run": RUNS[v], "ngeom": m.ngeom, "num_envs": n, "nconmax": nconmax, "njmax": njmax,
                   "sensor_reduce": env_build.CONTACT_SENSOR_REDUCE, "sensor_maxmatch": env_build.CONTACT_SENSOR_MAXMATCH,
                   "impratio": a.impratio or 10.0, "cal_ncon_world_max": cmax, "cal_nefc_world_max": emax,
                   "gpu_used_before_mb": gpu_used_mb(), "when": time.strftime("%Y-%m-%d %H:%M")}
            try:
                for attempt in range(4):           # MuJoCo-Warp names the size an overflowing buffer needed
                    try:
                        row.update(timed(cfg, n, nconmax, njmax, a.steps, a.warmup, a.phys_steps, a.impratio))
                        break
                    except ValueError as e:
                        mm = re.search(r"(nconmax|njmax) must be >= (\d+)", str(e))
                        if not mm or attempt == 3:
                            raise
                        need = math.ceil(1.25 * int(mm.group(2)))
                        print(f"  {v} {n}: {mm.group(1)} -> {need} ({e})", flush=True)
                        if mm.group(1) == "nconmax":
                            nconmax = need
                        else:
                            njmax = need
                        import torch
                        torch.cuda.empty_cache()
                row.update(nconmax=nconmax, njmax=njmax)
                row["status"] = "ok"
                print(f"  {v:<7} {n:>5} envs: {row['env_steps_per_s']:>9,.0f} env steps/s, physics "
                      f"{row['physics_us_per_world_step']:.2f} us/world-step ({row['physics_ms_per_batch_step']:.2f} "
                      f"ms/batch), peak {row['peak_gib']:.1f} GiB, ncon/world {row['ncon_world_max']}, "
                      f"nefc/world {row['nefc_world_max']}{' OVERFLOW' if row['overflow'] else ''}", flush=True)
            except Exception as e:
                row.update(status="error", error=f"{type(e).__name__}: {e}")
                print(f"  {v} {n}: {row['error']}", flush=True)
                import torch
                torch.cuda.empty_cache()
            with open(a.out, "a") as fh:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
    return 0


if __name__ == "__main__":
    sys.exit(main())
